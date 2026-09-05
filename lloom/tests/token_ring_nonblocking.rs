//! Is the non-blocking read actually non-blocking? (lloom #535, `DESIGN.md` §11.5)
//!
//! **This file is separate from `token_ring_tests.rs` for a measured reason,
//! and the reason is the finding.** The discriminator it needs is
//! `candle-core`'s `to_cpu` engagement counters -- and those are
//! **process-global statics**, while cargo runs integration tests on parallel
//! threads within one binary. The first version of this test lived beside its
//! siblings and **failed on its own first run**, reporting `(1 fast, 0 blit)`
//! for a `poll_eos` that is provably non-blocking when measured alone
//! (`to_cpu` delta `(0, 0)`). The increment was a *sibling test's* `to_vec1`,
//! landing inside this one's before/after window.
//!
//! That is `CONTRIBUTING.md` §3.1b's *"a test measuring a SHARED quantity must
//! attribute it"* -- #519's defect, where a fixture took one baseline against a
//! process-wide residency set -- reproduced inside the test written to catch a
//! different defect, and caught only because the first run was red.
//!
//! **A mutex was tried and rejected**: it is only sound if *every* test that
//! touches the counters takes it, which no lint enforces and a new sibling
//! would silently break. `--test-threads=1` was rejected too -- a test that is
//! correct only under a flag nobody passes is a latent flake, and `cargo test`
//! with no arguments is what every gate here runs. **A separate test binary is
//! its own process**, so the attribution holds by construction rather than by
//! convention.
//!
//! # Why a counter and not a clock
//!
//! This file exists because of a mutation that SURVIVED the value tests (`M4`
//! in `measurements/issue-535-raw/mutate.sh`): inserting a
//! `flush_and_wait_current` at the top of `peek_cpu` makes the "non-blocking"
//! read secretly blocking, and **no assertion over values can see it** -- a
//! blocking read returns exactly the same bytes and only *when* differs. That
//! is `DESIGN.md` §15.1a's INEXPRESSIVE class, whose repair is a different
//! instrument rather than more runs. The counters are that instrument: a
//! `to_vec1` increments `to_cpu_fast` or `to_cpu_blit`, and a `peek_vec1` must
//! increment **neither**.
//!
//! A timing assertion was considered and rejected: it would be a threshold on a
//! machine this project measures at load 2--26 (§3.2), which is how a test
//! becomes a flake.

use lloom_tensor::{Device, Result, Tensor};
use lloom::generation::token_ring::TokenRing;

const EOS: &[u32] = &[124900];

/// Counts every allocation the process makes, for #578's first defect.
///
/// **A global allocator is the only instrument that can see this.** The defect
/// is that `poll_eos` reached `peek_vec1`, which materialises a `Vec` of the
/// whole storage and then a second `Vec` for the slice -- two heap allocations
/// to deliver one `u32`, at ~283 000 poll iterations per token. No assertion
/// over the *value* can see it: the allocating and non-allocating reads return
/// exactly the same bytes, which is `DESIGN.md` §15.1a's INEXPRESSIVE class
/// again and the same shape as #535's M4. The repair is a different
/// instrument, and this is it.
///
/// This is process-global like the peek counters, which is why the assertion
/// lives in **this** binary rather than beside its siblings -- see the module
/// comment above: cargo runs integration tests on parallel threads within one
/// binary, so a shared counter must be attributed by construction.
struct CountingAllocator;

static ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// SAFETY: this forwards every call to the system allocator unchanged and only
// increments a counter, so it upholds `GlobalAlloc`'s contract exactly as
// `System` does.
unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn allocs() -> u64 {
    ALLOCS.load(std::sync::atomic::Ordering::Relaxed)
}

fn logits_peaked_at(at: usize, vocab: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; vocab];
    v[at] = 1.0;
    Tensor::from_vec(v, vocab, device)
}

/// The poll does not reach the blocking readback, and the blocking read does.
///
/// The second half is the positive control and is not optional: if `to_vec1`
/// also failed to move the counters, the first assertion would pass on an
/// instrument that reads nothing (§15.1 #1, and #336's own first sweep, which
/// measured the blit against itself and read `unresolved` on every dimension).
#[test]
fn the_poll_does_not_reach_the_blocking_readback() -> Result<()> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let ring = TokenRing::new(1, EOS, &device)?;
    let sampled = logits_peaked_at(99, 200_000, &device)?.argmax(lloom_tensor::D::Minus1)?;
    ring.publish_step(&sampled, Some(1))?;
    device.synchronize()?;

    // --- the arm under test -------------------------------------------------
    let (fast0, blit0) = lloom_tensor::metal_backend::to_cpu_counts();
    let (peek0, _) = lloom_tensor::metal_backend::peek_counts();
    let reads0 = lloom::generation::token_ring::poll_read_count();
    let polled = ring.poll_eos(1);
    let (fast1, blit1) = lloom_tensor::metal_backend::to_cpu_counts();
    let (peek1, _) = lloom_tensor::metal_backend::peek_counts();
    let reads1 = lloom::generation::token_ring::poll_read_count();

    assert_eq!(polled, Some(false), "the poll did not answer");
    assert_eq!(
        (fast1 - fast0, blit1 - blit0),
        (0, 0),
        "poll_eos reached the BLOCKING readback ({} fast, {} blit) -- it is not \
         non-blocking, whatever the RESULT line reports",
        fast1 - fast0,
        blit1 - blit0
    );
    // The poll must go through **one** of the two non-blocking reads, and which
    // one is an allocator property (§15.2 #12a) rather than a choice: the fast
    // path needs the allocation mapped, and #454 measures that as unstable run
    // to run. Before #578 there was one path and `peek_hit` was the whole
    // instrument; now a mapped run reads through a resolved address and moves
    // `poll_read_count` instead, so asserting `peek_hit` alone would fail on
    // the path that ships. What the test needs is unchanged -- that the poll
    // was observed at all -- so the assertion is over the pair.
    assert!(
        peek1 > peek0 || reads1 > reads0,
        "poll_eos went through neither non-blocking read (peek {peek0} -> \
         {peek1}, resolved-address reads {reads0} -> {reads1}), so this test is \
         not observing the path it names"
    );
    // And the arm that ran is the one this build should be taking, so a silent
    // fall back to the locking path is a visible failure rather than a quiet
    // loss of the property #578 built (§7.1a-i's wrong-arm shape).
    assert_eq!(
        ring.poll_is_mapped(),
        reads1 > reads0,
        "the ring reports mapped={} but the resolved-address counter moved by \
         {} -- the poll took an arm other than the one it reports",
        ring.poll_is_mapped(),
        reads1 - reads0
    );

    // --- the positive control ----------------------------------------------
    let (fast2, blit2) = lloom_tensor::metal_backend::to_cpu_counts();
    let _ = ring.eos_flag().to_vec1::<u8>()?;
    let (fast3, blit3) = lloom_tensor::metal_backend::to_cpu_counts();
    assert!(
        (fast3 - fast2) + (blit3 - blit2) > 0,
        "the blocking read did not move the readback counters, so the assertion \
         above proves nothing"
    );

    Ok(())
}

/// The poll allocates **nothing**, and the old spelling allocated (#578).
///
/// `DESIGN.md` §11.5n's non-blocking path exists to stop touching the allocator
/// on the decode path (`to_cpu_fast` 65 -> 1), and it was replacing a blocking
/// readback with an allocator storm: two `Vec`s and a lock per iteration, at
/// ~283 000 iterations per token.
///
/// **The positive control is not optional and is the point of the second
/// half.** A zero allocation count reads identically to an instrument that is
/// not attached (§3.1c: *a zero is not evidence unless something nonzero came
/// from the same command*), so `peek_vec1` -- the call `poll_eos` used to make
/// -- is exercised on the same tensors in the same test and must move the
/// counter. That is also what pins the finding: the two calls answer the same
/// question and only one of them allocates.
#[test]
fn the_poll_allocates_nothing_and_the_old_spelling_did() -> Result<()> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let ring = TokenRing::new(1, EOS, &device)?;
    let sampled = logits_peaked_at(99, 200_000, &device)?.argmax(lloom_tensor::D::Minus1)?;
    ring.publish_step(&sampled, Some(1))?;
    device.synchronize()?;

    // The fast path is only taken when the allocations are mapped, which is an
    // allocator property (§15.2 #12a) -- so it is read rather than assumed. If
    // it did not resolve, the assertion below would be about the fallback and
    // would say nothing about #578.
    assert!(
        ring.poll_is_mapped(),
        "the ring did not resolve its mapped addresses, so this test is \
         measuring the locking fallback rather than the path #578 built"
    );

    // Warm anything lazily initialised behind the first poll, so the measured
    // window contains steady-state polling only.
    let _ = ring.poll_eos(1);

    // --- the arm under test -------------------------------------------------
    const ITERS: usize = 1000;
    let a0 = allocs();
    let mut answered = 0usize;
    for _ in 0..ITERS {
        if ring.poll_eos(1).is_some() {
            answered += 1;
        }
    }
    let a1 = allocs();

    assert_eq!(
        answered, ITERS,
        "the poll declined mid-loop, so the allocation count below is not \
         measuring {ITERS} completed polls"
    );
    assert_eq!(
        a1 - a0,
        0,
        "poll_eos made {} allocations over {ITERS} iterations -- the decode \
         path is touching the allocator once per poll, which is #578's defect 1",
        a1 - a0
    );

    // --- the positive control ----------------------------------------------
    // `peek_vec1` is the call `poll_eos` used to make. Same tensors, same
    // question, same window: it must move the counter, or the zero above is an
    // instrument that is not attached.
    let step = ring.step_counter();
    let c0 = allocs();
    for _ in 0..ITERS {
        let _ = step.peek_vec1::<u32>();
    }
    let c1 = allocs();
    assert!(
        c1 - c0 >= ITERS as u64,
        "the allocating spelling made only {} allocations over {ITERS} \
         iterations, so the zero above does not demonstrate anything",
        c1 - c0
    );

    Ok(())
}

/// The peek fails **closed** on a shape it cannot serve, rather than blocking
/// or reading garbage.
///
/// `peek_vec1` returning `None` is the contract's escape hatch and it must be
/// reachable, or callers will be written as though it never fires -- which is
/// how §11.5c's `contents()` test came to be described as "failing closed"
/// while the path around it did not (§15.2 #12b).
#[test]
fn the_peek_declines_shapes_it_cannot_serve() -> Result<()> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    // Rank 2 is not served.
    let t = Tensor::from_vec(vec![1u32, 2, 3, 4], (2, 2), &device)?;
    assert!(
        t.peek_vec1::<u32>().is_none(),
        "a rank-2 tensor was served by a rank-1 peek"
    );

    // A dtype mismatch is not served: reinterpreting the bytes would be a
    // silent wrong answer, which is the class §10.4 exists for.
    let u = Tensor::from_vec(vec![1u32, 2, 3, 4], 4, &device)?;
    assert!(
        u.peek_vec1::<u8>().is_none(),
        "a U32 buffer was reinterpreted as U8"
    );
    // And the matching dtype IS served, which is the control that makes the
    // two assertions above readable rather than vacuous.
    assert_eq!(
        u.peek_vec1::<u32>(),
        Some(vec![1u32, 2, 3, 4]),
        "the matching dtype was declined, so the two refusals above prove nothing"
    );

    Ok(())
}

/// A CPU tensor is declined rather than silently served.
///
/// §14.1 makes not changing the CPU and CUDA backends a hard constraint. The
/// peek is a Metal capability; serving a CPU tensor here would make the arm
/// report a non-blocking read on a backend where the concept does not apply,
/// which is §7.1a-i's wrong-arm shape.
#[test]
fn a_cpu_tensor_is_declined() -> Result<()> {
    let t = Tensor::from_vec(vec![1u32, 2, 3, 4], 4, &Device::Cpu)?;
    assert!(
        t.peek_vec1::<u32>().is_none(),
        "the CPU backend was served by a Metal-only peek"
    );
    Ok(())
}
