//! What an LFM2 prefill forward takes from the device, measured with nothing
//! allowed to complete, against the activation class admission charges it
//! (`admission::prefill_activation_bytes_at`, `memory.md` §9.5d).
//!
//! The forward's commands are committed only once a host-raised event
//! reaches its value (`HostEvent::commit_when_produced`), so nothing it
//! encodes completes, every buffer it releases stays pending and none is
//! reused: `live + pending` grows by what the forward acquired. What stays live once it
//! completes (the KV slab, the conv state) is the sequence's state, not the
//! pass's, and is taken off. The fused arm's class is linear in the pass's
//! positions, so two pass lengths difference the per-forward constants away
//! (the logits, the conv carry) and leave the per-position bytes, which are
//! held to the derivation exactly; the absolute figure is held to it as an
//! upper bound.
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Config, KvAppend, MemoryBudget, QkvLayout, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::{CrownExact, PoolExact, Profile};
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::metal_backend::admission::PrefillAttention;
use lloom_tensor::metal_backend::{HostEvent, MetalDevice};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The pool and the dispatch clock are process-wide: one measurement at a time.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

/// The pool's live plus pending bytes.
fn held_bytes(m: &MetalDevice) -> i64 {
    let (s, p) = m.pool_occupancy();
    (s.live_bytes + p.live_bytes + s.pending_bytes + p.pending_bytes) as i64
}

fn live_bytes(m: &MetalDevice) -> i64 {
    let (s, p) = m.pool_occupancy();
    (s.live_bytes + p.live_bytes) as i64
}

/// The tiny LFM2 (conv, conv, attention; head dim 64) at a serving arm, with
/// two KV heads: at one, the `[B, P, KV]` → `[B, KV heads, P, D]` view of `v`
/// is already contiguous and the page pool's copy of it cannot be seen.
fn config(arm: &str) -> Config {
    let mut cfg = <Lfm2Tiny as TinyFamily>::template();
    cfg.num_key_value_heads = 2;
    cfg.qkv_layout = QkvLayout::Fused;
    cfg.conv_fused = true;
    cfg.conv_prefill_ctx = true;
    match arm {
        // `lfm2_presets::crown`'s attention and KV.
        "crown" => {
            cfg.attn_impl = AttnImpl::Sdpa;
            cfg.kv_append = KvAppend::InPlace;
        }
        // The page pool `lloom chat` takes by default (a `Paged` rows profile).
        "paged" => {
            cfg.attn_impl = AttnImpl::FlashDecoding;
            cfg.kv_append = KvAppend::Paged;
        }
        other => panic!("no arm {other}"),
    }
    cfg
}

fn ids(n: usize, from: usize, cfg: &Config) -> Vec<u32> {
    (from..from + n)
        .map(|i| (i * 7 % (cfg.vocab_size - 3) + 3) as u32)
        .collect()
}

/// One held pass: `prefix` positions prefilled first (unheld), then
/// `positions` more with the GPU held. Returns what the held pass acquired
/// less what of it stayed live after completion.
fn held_pass<Q: Profile>(cfg: &Config, prefix: usize, positions: usize) -> Result<i64> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    // An empty free list: every acquisition is a fresh buffer at its own
    // size, so `live + pending` is what the pass allocated.
    m.set_free_budget(0);
    assert!(
        !m.pass_plans_enabled(),
        "a pass plan serves activations from its base; this measures the pool"
    );
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<Q>(&dev)?;
    let mut cache = State::<Q>::new(true, DType::F16, cfg, &dev)?;

    // A warm pass of the same shape, so every lazily built table exists
    // before the measured one; then the state is rebuilt from scratch.
    let warm = Tensor::new(ids(prefix + positions, 0, cfg).as_slice(), &dev)?.unsqueeze(0)?;
    model.forward(&warm, 0, &mut cache)?;
    dev.synchronize()?;
    let mut cache = State::<Q>::new(true, DType::F16, cfg, &dev)?;
    if prefix > 0 {
        let head = Tensor::new(ids(prefix, 0, cfg).as_slice(), &dev)?.unsqueeze(0)?;
        model.forward(&head, 0, &mut cache)?;
    }
    let input = Tensor::new(ids(positions, prefix, cfg).as_slice(), &dev)?.unsqueeze(0)?;
    dev.synchronize()?;
    let settle = m.flush_marker()?;
    m.wait_marker(&settle)?;

    // One command buffer for the held pass: a held queue retires none, and
    // Metal blocks `commandBuffer` once 64 are outstanding.
    let period = m.compute_per_buffer();
    m.set_compute_per_buffer(usize::MAX >> 1);
    let clock = m.gpu_clock();
    let before = held_bytes(m);
    let before_live = live_bytes(m);
    let gate = Arc::new(HostEvent::new(m)?);
    gate.commit_when_produced(1)?;
    let held_epoch = clock.current_epoch();
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let (g, d, t) = (gate.clone(), done.clone(), timed_out.clone());
        std::thread::spawn(move || {
            for _ in 0..600 {
                if d.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            t.store(true, Ordering::Release);
            g.signal(1);
        })
    };
    let logits = model.forward(&input, prefix, &mut cache);
    let after = held_bytes(m);
    let held_complete = clock.is_complete(held_epoch);
    done.store(true, Ordering::Release);
    gate.signal(1);
    watchdog.join().expect("the watchdog");
    assert!(
        !timed_out.load(Ordering::Acquire),
        "the pass waited on the held GPU: a host read inside it"
    );
    assert!(!held_complete, "the held epoch retired before its release");
    drop(logits?);
    dev.synchronize()?;
    m.set_compute_per_buffer(period);
    let kept = live_bytes(m) - before_live;
    let transient = (after - before) - kept;
    eprintln!(
        "HELD prefix={prefix} positions={positions} acquired={} kept={kept} transient={transient}",
        after - before
    );
    Ok(transient)
}

/// What admission charges a pass of `positions` over `keys` keys at `cfg`'s
/// arms: the production arithmetic, through the model's own budget.
fn derived(cfg: &Config, positions: usize, keys: usize) -> i64 {
    let b = State::<CrownExact>::admission_budget(
        cfg,
        &MemoryBudget::new(0).with_prefill(positions),
        keys,
    );
    b.footprint().activations as i64
}

/// A held pass onto a warm cache acquires exactly what admission charges a
/// pass of its length, on both serving arms: the derivation's worst pass is
/// a continuation (the left context's concatenation and, on the page pool,
/// the gathered cache). A first pass acquires less by the concatenation —
/// one `[B, H, P + L − 1]` copy per conv layer, the previous window's tail
/// among it — and, on the page pool, by its slot table, which it keeps as
/// state where a continuation's replaces the previous one; both are held
/// exactly too, so the bound is not loose by more than those terms.
#[test]
fn a_held_prefill_acquires_what_admission_charges() -> Result<()> {
    let _serial = SERIAL.lock();
    for arm in ["crown", "paged"] {
        let cfg = config(arm);
        let route = State::<CrownExact>::admission_budget(&cfg, &MemoryBudget::new(0), 1)
            .prefill_geometry
            .attention;
        assert_eq!(
            route,
            if arm == "crown" {
                PrefillAttention::Fused
            } else {
                PrefillAttention::FusedGathered
            },
            "{arm}: the arm admission reads"
        );
        let conv_layers = cfg
            .layer_types
            .iter()
            .filter(|k| **k == lloom::models::lfm2::LayerType::Conv)
            .count() as i64;
        let h = cfg.hidden_size as i64;
        // A continuation follows a pass of its own length, as a tile follows
        // a tile: the page pool's slot table it replaces is then its size.
        for (prefix, positions) in [(0, 64), (0, 128), (64, 64), (128, 128)] {
            let measured = match arm {
                "crown" => held_pass::<CrownExact>(&cfg, prefix, positions)?,
                _ => held_pass::<PoolExact>(&cfg, prefix, positions)?,
            };
            let charged = derived(&cfg, positions, prefix + positions);
            let first_pass = if prefix == 0 {
                let slots = if arm == "paged" { 4 * positions as i64 } else { 0 };
                conv_layers * h * 2 * (positions as i64 + cfg.conv_l_cache as i64 - 1) + slots
            } else {
                0
            };
            eprintln!(
                "CHARGED arm={arm} prefix={prefix} positions={positions} measured={measured} \
                 charged={charged} first_pass={first_pass} over={}",
                charged - measured
            );
            assert_eq!(
                measured + first_pass,
                charged,
                "{arm} prefix {prefix} positions {positions}: the pool against admission"
            );
        }
    }
    Ok(())
}

/// The instrument behind the derivation: one pass onto a warm cache with an
/// empty free-list cap, so every acquisition is a fresh allocation, which
/// `LLOOM_POOL_TRACE=1` prints with the frames that asked for it between the
/// `TRACE-BEGIN` and `TRACE-END` markers. Not held, and not a gate: run by
/// hand with `--run-ignored only`; `LFM2_HELD_ARM` (paged) and
/// `LFM2_HELD_PREFIX` (64) choose the pass.
#[test]
#[ignore = "an attribution instrument, run by hand under LLOOM_POOL_TRACE=1"]
fn trace_one_pass() -> Result<()> {
    let _serial = SERIAL.lock();
    let arm = std::env::var("LFM2_HELD_ARM").unwrap_or_else(|_| "paged".into());
    let prefix: usize = std::env::var("LFM2_HELD_PREFIX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let cfg = config(&arm);
    match arm.as_str() {
        "crown" => trace_pass::<CrownExact>(&cfg, prefix, 64),
        _ => trace_pass::<PoolExact>(&cfg, prefix, 64),
    }
}

fn trace_pass<Q: Profile>(cfg: &Config, prefix: usize, positions: usize) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<Q>(&dev)?;
    let mut cache = State::<Q>::new(true, DType::F16, cfg, &dev)?;
    if prefix > 0 {
        let head = Tensor::new(ids(prefix, 0, cfg).as_slice(), &dev)?.unsqueeze(0)?;
        model.forward(&head, 0, &mut cache)?;
    }
    let input = Tensor::new(ids(positions, prefix, cfg).as_slice(), &dev)?.unsqueeze(0)?;
    dev.synchronize()?;
    m.set_free_budget(0);
    eprintln!("TRACE-BEGIN prefix={prefix} positions={positions}");
    let before = held_bytes(m);
    let logits = model.forward(&input, prefix, &mut cache)?;
    let after = held_bytes(m);
    eprintln!("TRACE-END acquired={}", after - before);
    drop(logits);
    dev.synchronize()?;
    Ok(())
}
