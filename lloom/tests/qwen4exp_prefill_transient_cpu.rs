//! What a Qwen4-exp prefill forward holds on the CPU at its peak beside its
//! state: the heap's high-water over the forward, less what it held when
//! the forward began. The CPU frees a tensor when its last handle drops, so
//! this is the forward's live set at its largest — the CPU arm of the
//! transient the machine gate admits (`memory.md` §9.10), and the arm the
//! container CI runs the tiny servers on.
//!
//! Its own binary: a test binary carries at most one `#[global_allocator]`.
//! The counter is process-wide, so the GEMM's worker threads are counted;
//! a positive control (a known allocation inside a counted window) runs
//! first, so a figure below is the forward's and not a blind instrument's.

use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{Config, Lifetime, Qwen4Exp, Resident, State};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicI64, Ordering};

static CURRENT: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);

fn grew(by: i64) {
    let now = CURRENT.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

struct Peak;

// SAFETY: every method forwards to the system allocator unchanged; the
// counters are atomics, so counting allocates nothing and never reenters.
unsafe impl std::alloc::GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size() as i64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size() as i64);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) };
        grew(-(layout.size() as i64));
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { std::alloc::System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            grew(new_size as i64 - layout.size() as i64);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Peak = Peak;

/// `f`'s heap high-water above the heap at its start, and what it left.
fn high_water<T>(f: impl FnOnce() -> T) -> (T, i64, i64) {
    let start = CURRENT.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let out = f();
    let end = CURRENT.load(Ordering::Relaxed);
    (out, PEAK.load(Ordering::Relaxed) - start, end - start)
}

pub fn geometry(name: &str) -> Config {
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
        other => panic!("no geometry {other}"),
    }
    cfg.tiny_derive();
    cfg
}

/// One forward of `prompt` positions from position 0 after a warm forward
/// of the same prompt: `(peak above the start, left at the end, the
/// model's derived figure for it)`.
fn cpu_forward(cfg: &Config, prompt: usize) -> Result<(i64, i64, i64)> {
    let dev = Device::Cpu;
    let model: Qwen4Exp<CrownExact, Resident> =
        TinyCheckpoint::<Qwen4ExpTiny>::new(cfg.clone(), Weights::default()).model(&dev)?;
    let mut cache: State<CrownExact> = model.new_cache(cfg, &dev)?;
    let ids: Vec<u32> = (0..prompt)
        .map(|i| (i * 7 % (cfg.vocab_size - 3) + 3) as u32)
        .collect();
    let input = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
    model.note_input_ids(&mut cache, &ids)?;
    model.forward(&input, 0, &mut cache)?;
    cache.reset();
    model.note_input_ids(&mut cache, &ids)?;
    let derived = model.prefill_transient();
    assert_eq!(
        derived.route().lifetime,
        Lifetime::Eager,
        "the CPU frees eagerly"
    );
    let derived = derived.forward(prompt, 0).held as i64;
    let (logits, peak, left) = high_water(|| model.forward(&input, 0, &mut cache));
    drop(logits?);
    Ok((peak, left, derived))
}

#[test]
fn the_counter_sees_an_allocation_inside_its_window() {
    let ((), peak, left) = high_water(|| {
        let v = vec![1u8; 3 << 20];
        std::hint::black_box(&v);
    });
    assert!(peak >= 3 << 20, "a 3 MiB Vec read {peak} B of high-water");
    assert!(left.abs() < 1 << 16, "the Vec was freed: {left} B left");
}

/// The derivation's CPU arm against the heap: what the forward holds at
/// its peak beyond what it leaves (its KV slabs and state) is never above
/// one layer's allocations and the carries (`Lifetime::Eager`) — the
/// under-estimate direction, which is the one that lets a container's
/// limit be crossed. The CPU frees inside a layer, so its live set is a
/// fraction of what the layer allocates: the ratio is printed, and the
/// over-estimate is held against the constant this replaced.
#[test]
fn a_cpu_prefill_holds_one_layers_live_set_at_its_peak() -> Result<()> {
    for g in ["tiny", "wide", "deep"] {
        let cfg = geometry(g);
        for prompt in [7, 64, 512, 600] {
            let (peak, left, derived) = cpu_forward(&cfg, prompt)?;
            let transient = peak - left;
            eprintln!(
                "CPU geometry={g} prompt={prompt} peak={peak} left={left} transient={transient} \
                 derived={derived} ratio={:.2}",
                derived as f64 / transient.max(1) as f64
            );
            assert!(peak > 0, "{g} {prompt}: no high-water");
            assert!(
                transient <= derived,
                "{g} {prompt}: the CPU held {transient} B past its state, over the derived {derived} B"
            );
            // The over-estimate, against what it replaced: the claim
            // the family's constant made for a forward of this length
            // (3 MB a position, two tiles of the chunk at most) refused
            // three tiny servers on a 9.7 GB host. The derivation stays a
            // small fraction of it.
            let constant = 3_000_000 * prompt.min(512) as i64;
            assert!(
                derived * 4 <= constant,
                "{g} {prompt}: derived {derived} B is not under a quarter of the constant's {constant} B"
            );
        }
    }
    Ok(())
}
