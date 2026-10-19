//! What a Qwen4-exp prefill forward takes from the device beside its state,
//! measured with nothing allowed to complete: the GPU waits on a shared
//! event encoded before the forward, so every buffer the forward releases
//! stays pending and none is reused. `live + pending` grows by what the
//! forward acquired — its transient with no completion inside it, the
//! upper bound the resident tier's admission must cover (no host read
//! inside the pass lets the pool drain early).
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident, State};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::metal_backend::{HostEvent, MetalDevice};
use lloom_tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The process-wide dispatch clock and the pool are shared by every test
/// in this binary's process; one measurement at a time.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

/// The pool's live and pending bytes, and the device's allocation.
#[derive(Debug, Clone, Copy)]
struct Pool {
    live: i64,
    pending: i64,
    device: i64,
}

fn pool(m: &MetalDevice) -> Pool {
    let (s, p) = m.pool_occupancy();
    Pool {
        live: (s.live_bytes + p.live_bytes) as i64,
        pending: (s.pending_bytes + p.pending_bytes) as i64,
        device: m.device().current_allocated_size() as i64,
    }
}

/// One forward of `prompt` positions from position 0 with the GPU held,
/// after a warm forward of the same prompt: `(after − before)` of the
/// pool and the device. The warm forward leaves its buffers on the free
/// list, which the held forward may take; `live + pending` counts what it
/// acquired whichever list served it.
fn held_forward(
    cfg: &Config,
    prompt: usize,
    free_budget: Option<usize>,
    witness_reset: bool,
) -> Result<Held> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    if let Some(b) = free_budget {
        m.set_free_budget(b);
    }
    assert!(
        !m.pass_plans_enabled(),
        "a pass plan serves activations from its base; this measures the pool"
    );
    let model: Qwen4Exp<CrownExact, Resident> =
        TinyCheckpoint::<Qwen4ExpTiny>::new(cfg.clone(), Weights::default()).model(&dev)?;
    let mut cache: State<CrownExact> = model.new_cache(cfg, &dev)?;
    let ids: Vec<u32> = (0..prompt)
        .map(|i| (i * 7 % (cfg.vocab_size - 3) + 3) as u32)
        .collect();
    let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
    // The ids noted on the host, as the engine's prefill notes them: a
    // forward with none reads them back from the device.
    model.note_input_ids(&mut cache, &ids)?;
    model.forward(&input, 0, &mut cache)?;
    dev.synchronize()?;
    cache.reset();
    // A second warm forward, so every lazily built table and the state's
    // first allocations exist before the measured one.
    model.note_input_ids(&mut cache, &ids)?;
    model.forward(&input, 0, &mut cache)?;
    dev.synchronize()?;
    cache.reset();
    dev.synchronize()?;
    let clock = m.gpu_clock();
    let reset_epoch = clock.current_epoch();
    let reset = pool(m);
    let reset_complete = clock.is_complete(reset_epoch);
    // Reset releases are stamped on the open epoch even when it has no
    // dispatches. Close and drain that prior epoch before measuring a pass.
    let reset_marker = m.flush_marker()?;
    m.wait_marker(&reset_marker)?;
    let settled = pool(m);
    assert!(clock.is_complete(reset_epoch), "reset epoch did not retire");
    assert_eq!(settled.pending, 0, "prior reset releases remain pending");
    if witness_reset {
        assert!(reset.pending > 0, "reset-release witness was vacuous");
        assert!(!reset_complete, "reset epoch was already complete");
    }
    eprintln!(
        "HELD_BASELINE actual={} prompt={prompt} reset_epoch={} reset_live={} \
         reset_pending={} reset_complete={reset_complete} settled_live={} \
         settled_pending={} completed={}",
        m.commands().actual_command_api(),
        reset_epoch.0,
        reset.live,
        reset.pending,
        settled.live,
        settled.pending,
        clock.completed_count().0
    );

    // One command buffer for the held forward: Metal blocks a queue's
    // `commandBuffer` once 64 are outstanding, and a held queue never
    // retires one. Which buffer a release parks on does not change what
    // stays pending while none completes.
    let period = m.compute_per_buffer();
    m.set_compute_per_buffer(usize::MAX >> 1);
    let before = pool(m);
    let gate = HostEvent::new(m)?;
    gate.encode_wait(1)?;
    let held_epoch = clock.current_epoch();
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let (gate, watchdog) = {
        let gate = Arc::new(gate);
        let (g, d, t) = (gate.clone(), done.clone(), timed_out.clone());
        let w = std::thread::spawn(move || {
            for _ in 0..600 {
                if d.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            t.store(true, Ordering::Release);
            g.signal(1);
        });
        (gate, w)
    };
    model.note_input_ids(&mut cache, &ids)?;
    let logits = model.forward(&input, 0, &mut cache);
    let after = pool(m);
    let held_complete = clock.is_complete(held_epoch);
    done.store(true, Ordering::Release);
    gate.signal(1);
    watchdog.join().expect("the watchdog");
    assert!(
        !timed_out.load(Ordering::Acquire),
        "the forward waited on the held GPU: a host read inside the pass"
    );
    assert!(
        !held_complete,
        "held epoch retired before its event was released"
    );
    eprintln!(
        "HELD_EPOCH actual={} prompt={prompt} epoch={} complete={held_complete} \
         before_live={} before_pending={} after_live={} after_pending={}",
        m.commands().actual_command_api(),
        held_epoch.0,
        before.live,
        before.pending,
        after.live,
        after.pending
    );
    drop(logits?);
    dev.synchronize()?;
    m.set_compute_per_buffer(period);
    // What stays live once everything completed: the state the forward
    // grew (its KV slabs, its recurrent state's copies), not its transient.
    let kept = pool(m).live - before.live;
    Ok(Held {
        acquired: (after.live - before.live) + (after.pending - before.pending),
        kept,
        device: after.device - before.device,
        derived: model.prefill_transient(),
    })
}

/// A held forward's figures: what it acquired, what of it stayed live after
/// completion, the device's growth, and the model's own derivation.
struct Held {
    acquired: i64,
    kept: i64,
    device: i64,
    derived: lloom::models::qwen4exp::PrefillTransient,
}

fn geometry(name: &str) -> Config {
    let mut cfg = Config::tiny();
    match name {
        "tiny" => {}
        "wide" => {
            assert!(cfg.set_tiny_axis("hidden_size", 64));
            assert!(cfg.set_tiny_axis("n_experts", 16));
            assert!(cfg.set_tiny_axis("n_experts_per_tok", 4));
            assert!(cfg.set_tiny_axis("moe_intermediate_size", 32));
            assert!(cfg.set_tiny_axis("shared_expert_intermediate_size", 48));
            assert!(cfg.set_tiny_axis("ssm_state_size", 16));
            assert!(cfg.set_tiny_axis("ssm_dt_rank", 8));
        }
        "deep" => {
            assert!(cfg.set_tiny_axis("n_layers", 8));
            assert!(cfg.set_tiny_axis("hc_count", 4));
        }
        // A head the fused attention and the paged sparse kernel take.
        "fused" => {
            assert!(cfg.set_tiny_axis("head_dim", 32));
        }
        other => panic!("no geometry {other}"),
    }
    cfg.tiny_derive();
    cfg
}

/// The derivation against the pool: per geometry and prompt, a forward
/// held so nothing completes acquires what `PrefillTransient` says a
/// resident forward keeps — every allocation of the pass, the KV slab
/// aside (`qwen4exp-kv`'s claim). The route is the model's own, with no
/// turn checkpoint inside the pass (none is hinted here; the claim's route
/// adds one, an upper bound checked below). Both directions:
/// - under: the measured transient is never above the derivation;
/// - over: the derivation is within the state copies it also counts — two
///   per DeltaNet layer, live after the pass — of what was measured.
#[test]
fn a_held_prefill_acquires_what_its_layers_allocate() -> Result<()> {
    use lloom::models::qwen4exp::{Lifetime, PrefillTransient};
    let _serial = SERIAL.lock();
    for g in ["tiny", "wide", "deep", "fused"] {
        let cfg = geometry(g);
        for prompt in [7, 64, 512, 600] {
            // An empty free list: every acquisition is a fresh buffer at its
            // own size, so `live + pending` is what the forward allocated —
            // a free bucket serving a smaller request is memory already held
            // (and admitted as the free list's cap), not the pass's.
            let h = held_forward(&cfg, prompt, Some(0), g == "tiny" && prompt == 7)?;
            let route = h.derived.route();
            assert_eq!(
                route.lifetime,
                Lifetime::Deferred,
                "a resident forward on Metal"
            );
            let exact = PrefillTransient::new(
                &cfg,
                lloom::models::qwen4exp::Route {
                    checkpoint: false,
                    ..route
                },
            )
            .forward(prompt, 0);
            let claim = h.derived.forward(prompt, 0).held as i64;
            let derived = exact.held as i64;
            let transient = h.acquired - h.kept;
            let dims = cfg.deltanet_dims();
            let linear = cfg
                .layer_types
                .iter()
                .filter(|k| **k == lloom::models::qwen4exp::LayerKind::Linear)
                .count() as i64;
            let state = |b: usize| (b as i64 * 4 + 127) / 128 * 128;
            let copies = linear
                * 2
                * (state(dims.h_v * dims.d_k * dims.d_k)
                    + state(cfg.conv_channels() * cfg.ssm_conv_kernel));
            eprintln!(
                "HELD geometry={g} prompt={prompt} acquired={} kept={} transient={transient} \
                 derived={derived} claim_route={claim} device={} fused_qsa={} paged={}",
                h.acquired, h.kept, h.device, route.qsa_fused, route.sparse_paged
            );
            assert!(
                transient <= derived,
                "{g} {prompt}: measured {transient} B over the derived {derived} B"
            );
            assert!(
                derived <= transient + copies,
                "{g} {prompt}: derived {derived} B over the measured {transient} B by more \
                 than the {copies} B of state copies it counts"
            );
            assert!(
                claim >= derived,
                "{g} {prompt}: the claim's route is below the pass"
            );
            // The device's own growth: Metal's allocation granule above the
            // pool's, on what stayed beside the transient.
            assert!(
                h.device <= derived + h.kept + (derived + h.kept) / 4,
                "{g} {prompt}: the device grew {} B against {derived} B derived and {} B kept",
                h.device,
                h.kept
            );
        }
    }
    Ok(())
}

/// The instrument behind the table: one forward with an empty free-list
/// cap, so every acquisition is a fresh allocation, which
/// `LLOOM_POOL_TRACE=1` prints with the frames that asked for it, between
/// the `TRACE-BEGIN` and `TRACE-END` markers. Not held (a traced forward
/// outlasts the GPU's timeout on a held queue) and not a gate: run by hand
/// with `--run-ignored only`; `QWEN4EXP_HELD_PROMPT` positions (64),
/// `QWEN4EXP_HELD_GEOMETRY` (tiny).
#[test]
#[ignore = "an attribution instrument, run by hand under LLOOM_POOL_TRACE=1"]
fn trace_one_forward() -> Result<()> {
    let _serial = SERIAL.lock();
    let prompt: usize = std::env::var("QWEN4EXP_HELD_PROMPT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let name = std::env::var("QWEN4EXP_HELD_GEOMETRY").unwrap_or_else(|_| "tiny".into());
    let cfg = geometry(&name);
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.set_free_budget(0);
    let model: Qwen4Exp<CrownExact, Resident> =
        TinyCheckpoint::<Qwen4ExpTiny>::new(cfg.clone(), Weights::default()).model(&dev)?;
    let mut cache: State<CrownExact> = model.new_cache(&cfg, &dev)?;
    let ids: Vec<u32> = (0..prompt)
        .map(|i| (i * 7 % (cfg.vocab_size - 3) + 3) as u32)
        .collect();
    let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
    model.note_input_ids(&mut cache, &ids)?;
    model.forward(&input, 0, &mut cache)?;
    dev.synchronize()?;
    cache.reset();
    dev.synchronize()?;
    eprintln!("TRACE-BEGIN geometry={name} prompt={prompt}");
    model.note_input_ids(&mut cache, &ids)?;
    model.forward(&input, 0, &mut cache)?;
    eprintln!("TRACE-END");
    dev.synchronize()?;
    Ok(())
}
