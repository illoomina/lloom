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

use candle::{Device, Result, Tensor};
use candle_transformers::generation::token_ring::TokenRing;

const EOS: &[u32] = &[124900];

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
    let sampled = logits_peaked_at(99, 200_000, &device)?.argmax(candle::D::Minus1)?;
    ring.publish_step(&sampled, Some(1))?;
    device.synchronize()?;

    // --- the arm under test -------------------------------------------------
    let (fast0, blit0) = candle::metal_backend::to_cpu_counts();
    let (peek0, _) = candle::metal_backend::peek_counts();
    let polled = ring.poll_eos(1);
    let (fast1, blit1) = candle::metal_backend::to_cpu_counts();
    let (peek1, _) = candle::metal_backend::peek_counts();

    assert_eq!(polled, Some(false), "the poll did not answer");
    assert_eq!(
        (fast1 - fast0, blit1 - blit0),
        (0, 0),
        "poll_eos reached the BLOCKING readback ({} fast, {} blit) -- it is not \
         non-blocking, whatever the RESULT line reports",
        fast1 - fast0,
        blit1 - blit0
    );
    assert!(
        peek1 > peek0,
        "poll_eos did not go through the peek at all ({peek0} -> {peek1}), so \
         this test is not observing the path it names"
    );

    // --- the positive control ----------------------------------------------
    let (fast2, blit2) = candle::metal_backend::to_cpu_counts();
    let _ = ring.eos_flag().to_vec1::<u8>()?;
    let (fast3, blit3) = candle::metal_backend::to_cpu_counts();
    assert!(
        (fast3 - fast2) + (blit3 - blit2) > 0,
        "the blocking read did not move the readback counters, so the assertion \
         above proves nothing"
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
