#![cfg(feature = "metal")]
//! Ordering of blit-encoded copies against the last writer of their operands.
//!
//! `BlitCommandEncoder::copy_from_buffer` waited on the last writer of its
//! *source* but not of its *destination*, where the sibling `fill_buffer`
//! waited on its destination. These exercise both primitives after closing
//! that asymmetry.
//!
//! # Why the destination edge is not redundant
//!
//! `Commands::blit_command_encoder` already waits on every fence in
//! `live_fences` before handing a blit encoder out, which looks like it should
//! subsume any per-buffer wait. It does, for *compute* encoders: those register
//! their fence in `live_fences` in `Commands::end_encoding`.
//!
//! A *blit* encoder does not. `BlitCommandEncoder::end_encoding` registers its
//! outputs in `prev_ce_outputs` and updates its own fence, but never adds that
//! fence to `live_fences`. So a blit that writes buffer B followed by a second
//! blit that writes B again has a write-after-write dependency the blanket wait
//! cannot see; `prev_ce_outputs` is the only place it is recorded.
//!
//! Under `HazardTrackingModeUntracked` there is no safety net: a missed
//! dependency corrupts silently rather than failing.
//!
//! # What these tests do NOT show, stated up front
//!
//! **They do not fail when the destination wait is reverted.** Removing every
//! per-buffer wait from the blit encoder -- a strictly larger mutation -- leaves
//! all of them green, in release and in debug. So they are a regression guard
//! on the values, not a test of the ordering property, and they are labelled as
//! such rather than presented as evidence the edge was needed.
//!
//! The reason is measured, not assumed: instrumented on a real LFM2 decode,
//! `copy_from_buffer`'s destination has a registered last writer in **0** of
//! its calls. The edge is structurally right and, on the paths reachable here,
//! never live -- so there is nothing for a mutation to break. A test that could
//! fail would need a destination that genuinely carries a pending writer, and
//! no candle path observed here produces one.
//!
//! These also do **not** test the grouped-convolution corruption, which is a
//! different mechanism the fence machinery cannot observe at all, and which
//! this change does not fix.

use lloom_tensor::{DType, Device, Result, Tensor};

/// Repetitions per pattern. Enough that a frequent ordering failure would show
/// up; see the module note on why that is a weaker claim than it sounds.
const REPS: usize = 200;

/// Chained `Tensor::cat` copies. Each `cat` blits its inputs into a freshly
/// allocated destination, and the pool recycles destinations across iterations.
#[test]
fn chained_cat_copies_produce_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let a = Tensor::full(rep as f32, (128, 64), &device)?;
        let b = Tensor::full((rep + 1000) as f32, (128, 64), &device)?;

        // `cat` on a contiguous source takes the blit path in
        // `MetalStorage::copy_strided_src`, once per input.
        let joined = Tensor::cat(&[&a, &b], 0)?;
        let again = Tensor::cat(&[&joined, &joined], 1)?;

        let v = again.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v.len(), 256 * 128);
        let want_a = rep as f32;
        let want_b = (rep + 1000) as f32;
        assert!(
            v.iter().all(|&x| x == want_a || x == want_b),
            "rep {rep}: cat produced a value that is neither input: {:?}",
            v.iter()
                .find(|&&x| x != want_a && x != want_b)
                .copied()
                .unwrap_or(f32::NAN)
        );
        // Both halves present and in the right places, which whole-buffer
        // aliasing would violate.
        assert_eq!(v[0], want_a, "rep {rep}: first row wrong");
        assert_eq!(v[128 * 128], want_b, "rep {rep}: second half wrong");
    }
    Ok(())
}

/// Readback after a copy. The readback blit's destination is a fresh buffer
/// from the pool, which is the case where a pending writer is most likely.
#[test]
fn readback_after_copy_produces_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let src = Tensor::full(rep as f32, (256, 32), &device)?;
        let copied = Tensor::cat(&[&src], 0)?.contiguous()?;
        let v = copied.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|&x| x == rep as f32),
            "rep {rep}: readback saw a stale or foreign value; first bad = {:?}",
            v.iter().find(|&&x| x != rep as f32).copied()
        );
    }
    Ok(())
}

/// A `fill_buffer` (via `Tensor::zeros`) alongside a copy. This is the pairing
/// the two blit primitives are meant to be symmetric about: before the change
/// one waited on its destination and the other did not.
#[test]
fn fill_then_add_produces_the_right_values() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..REPS {
        let zeros = Tensor::zeros((128, 64), DType::F32, &device)?;
        let ones = Tensor::full(1f32, (128, 64), &device)?;
        let sum = (&zeros + &ones)?;
        let v = sum.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            v.iter().all(|&x| x == 1.0),
            "rep {rep}: zeros+ones != 1 everywhere; found {:?}",
            v.iter().find(|&&x| x != 1.0).copied()
        );
    }
    Ok(())
}

/// Interleaved copies across several destinations, so the pool cycles buffers
/// between them and a destination's previous writer is a *different* copy.
#[test]
fn interleaved_copies_do_not_alias_each_other() -> Result<()> {
    let device = Device::new_metal(0)?;

    for rep in 0..(REPS / 4) {
        let mut outs = Vec::new();
        for lane in 0..4usize {
            let v = (rep * 4 + lane) as f32;
            let t = Tensor::full(v, (64, 64), &device)?;
            outs.push((v, Tensor::cat(&[&t, &t], 0)?));
        }
        // Read back only after all four exist, so cross-lane aliasing
        // introduced while they were all in flight is still visible.
        for (want, t) in &outs {
            let got = t.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                got.iter().all(|&x| x == *want),
                "rep {rep}: lane expecting {want} saw {:?} -- another lane's value \
                 would indicate whole-buffer aliasing",
                got.iter().find(|&&x| x != *want).copied()
            );
        }
    }
    Ok(())
}
