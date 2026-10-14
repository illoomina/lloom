//! What a warmed decode step of the generic decoder costs the host, counted
//! apart from any timing (#993's step 3): the Rust allocations the stepping
//! thread makes per step, per window opening and closing, and, on Metal
//! with `CANDLE_METAL_PROFILE=1`, the dispatches per step. The model is the
//! tiny Qwen3.5 dense at the engine's shape (`No`, the standing window).
//!
//! Its own binary: a test binary carries at most one `#[global_allocator]`.
//! The allocator counts only on a thread that opted in, so nextest's and
//! the GPU completion threads' allocations are not the step's. A positive
//! control (a known allocation inside a counted region) runs first, so a
//! zero below is the code's and not a blind instrument's.

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result, Tensor};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

fn note(bytes: usize) {
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
            let _ = BYTES.try_with(|b| b.set(b.get() + bytes as u64));
        }
    });
}

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged; the
// counters are thread-local cells with const initializers, so counting
// allocates nothing and never reenters the allocator.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        note(layout.size());
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        note(layout.size());
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The allocations and bytes `f` makes on this thread.
fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (a0, b0) = (ALLOCS.with(Cell::get), BYTES.with(Cell::get));
    COUNTING.with(|on| on.set(true));
    let out = f();
    COUNTING.with(|on| on.set(false));
    (out, ALLOCS.with(Cell::get) - a0, BYTES.with(Cell::get) - b0)
}

fn model(dev: &Device) -> Result<(Qwen3_5<CrownExact>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        verify: 0,
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

const WARM: usize = 8;
const STEPS: usize = 32;

/// Per warmed step: allocations, bytes; per window opening and closing
/// outside a forward: allocations.
fn census(dev: &Device) -> Result<(f64, f64, f64)> {
    // The positive control, in this arm's process: a counted region with
    // one known allocation reads one.
    let (v, control, _) = counted(|| Vec::<u8>::with_capacity(8));
    assert_eq!(control, 1, "the counter is blind");
    drop(v);
    let (model, cfg) = model(dev)?;
    let mut st = model.new_cache(&cfg, dev)?;
    let prompt = Tensor::from_vec((0..12u32).map(|i| (i * 7 + 3) % 64).collect(), (1, 12), dev)?;
    model.forward(&prompt, 0, &mut st)?;
    let inputs = (0..WARM + STEPS)
        .map(|i| Tensor::from_vec(vec![(i as u32 * 5 + 1) % 64], (1, 1), dev))
        .collect::<Result<Vec<_>>>()?;
    for x in &inputs[..WARM] {
        let pos = st.pos;
        model.forward(x, pos, &mut st)?;
    }
    dev.synchronize()?;
    let (r, allocs, bytes) = counted(|| -> Result<()> {
        for x in &inputs[WARM..] {
            let pos = st.pos;
            let logits = model.forward(x, pos, &mut st)?;
            drop(logits);
        }
        Ok(())
    });
    r?;
    dev.synchronize()?;
    st.settle_standing()?;
    let (r, window_allocs, _) = counted(|| -> Result<()> {
        for _ in 0..STEPS {
            st.open_window()?;
            st.resolve_window()?;
        }
        Ok(())
    });
    r?;
    Ok((
        allocs as f64 / STEPS as f64,
        bytes as f64 / STEPS as f64,
        window_allocs as f64 / STEPS as f64,
    ))
}

#[test]
fn the_counter_sees_an_allocation() {
    let (v, allocs, bytes) = counted(|| Vec::<u64>::with_capacity(16));
    assert_eq!((allocs, bytes), (1, 128), "the positive control");
    drop(v);
}

#[test]
fn a_warmed_decode_step_on_the_cpu() -> Result<()> {
    let (per_step, bytes, window) = census(&Device::Cpu)?;
    eprintln!(
        "CENSUS cpu: {per_step:.2} allocations a step ({bytes:.0} bytes); {window:.2} a window \
         opening and closing"
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn a_warmed_decode_step_on_metal() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (per_step, bytes, window) = census(&dev)?;
    eprintln!(
        "CENSUS metal: {per_step:.2} allocations a step ({bytes:.0} bytes); {window:.2} a window \
         opening and closing"
    );
    // The dispatches, with `CANDLE_METAL_PROFILE=1` (a run of its own: the
    // census is instrumentation).
    let (model, cfg) = model(&dev)?;
    let mut st = model.new_cache(&cfg, &dev)?;
    let prompt = Tensor::from_vec(vec![1u32, 3, 5, 7, 9], (1, 5), &dev)?;
    model.forward(&prompt, 0, &mut st)?;
    let x = Tensor::from_vec(vec![11u32], (1, 1), &dev)?;
    let pos = st.pos;
    model.forward(&x, pos, &mut st)?;
    dev.synchronize()?;
    match dev.as_metal_device()?.dispatch_census().open() {
        Ok(c) => {
            let pos = st.pos;
            model.forward(&x, pos, &mut st)?;
            dev.synchronize()?;
            let n: u64 = c.close().iter().map(|(_, n)| n).sum();
            eprintln!("CENSUS metal: {n} dispatches a step");
            assert!(n > 0, "the dispatch census counted the step");
        }
        Err(_) => eprintln!("CENSUS metal: dispatch census off (CANDLE_METAL_PROFILE unset)"),
    }
    Ok(())
}
