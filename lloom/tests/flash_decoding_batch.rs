//! FlashDecoding at `B > 1`: every batch row must be computed.
//!
//! # The defect this pins
//!
//! `ops::flash_decoding` sized its output `[B, H, 1, D]` from `q_l.dim(0)` while
//! dispatching a grid whose `y` extent was `q_l.dim(1)` — the **q head count**,
//! with no batch axis. Every offset in `flash_decoding.metal` is computed from a
//! flat `head_idx` (`queries += head_idx * D`, `out + head_idx * D`,
//! `partials + head_idx * stride_chunks * D`), so the grid covered exactly row
//! 0 and **rows 1..B-1 were never written**.
//!
//! That is a silent wrong answer rather than a crash: the rows hold whatever
//! their freshly-allocated buffer happened to contain. Measured on the real
//! model before the fix, `lfm2-decode-profile --batch 2 --attn flash` diverged
//! at step 1 with row 0 emitting 4116 and row 1 emitting **127999 at B=2 and
//! 20392 at B=4** — a token that moves between runs, which is the signature of
//! reading unwritten memory rather than of a systematic mis-index.
//!
//! `AttnImpl::Sdpa` was **not** affected, which is what localises the defect:
//! `call_sdpa_vector` already flattens `b = q_shape[0] * q_shape[1]` into the
//! same grid axis. The repair makes FlashDecoding agree with it.
//!
//! # What this test asserts, and why it is an absolute reference
//!
//! `CONTRIBUTING.md` §3.1b rule 1: *prefer an absolute reference to a shape*. A
//! shape invariant here — *"all rows are equal"* — is exactly the assertion the
//! defect could satisfy, since B unwritten rows can hold equal garbage, and a
//! buffer that happens to be zeroed makes them equal by construction. So the
//! reference is the **B=1 answer computed by the same op**, and each row is
//! compared against it bit-for-bit.
//!
//! Feeding every row the *same* q makes the expected output exactly the B=1
//! output, which is the property `--batch-check` asserts on the real model at
//! the token level (`DESIGN.md` §13.4b) — this is that gate at the op level,
//! where it can run without a checkpoint.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::{DType, Device, Result, Tensor};

/// A deterministic filler. A fixed pattern rather than an RNG: the assertion is
/// bit-equality between two arms of one run, so what the values *are* does not
/// matter, and a reproducible fixture makes a failure re-derivable.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

/// Every batch row must reproduce the `B = 1` answer.
///
/// The rows carry identical queries against identical KV, so the correct output
/// is the B=1 output repeated `B` times. Before the repair, rows 1.. held
/// unwritten memory.
#[test]
fn flash_decoding_writes_every_batch_row() -> Result<()> {
    let device = match Device::new_metal(0) {
        Ok(d) => d,
        // Not a silent skip on a Metal machine: this file is `cfg(metal)`, so
        // reaching here means the feature is on and the device is absent.
        Err(e) => return Err(e),
    };

    const H: usize = 8;
    const H_KV: usize = 2;
    const D: usize = 64;
    // Two chunks at the page size below, so the combine actually merges rather
    // than degenerating to `n_chunks == 1` — which is the arm §10.4d shows is
    // "FlashDecoding not doing the thing it exists to do".
    const KV_LEN: usize = 64;
    const PAGE: usize = 32;
    const K: usize = 1;

    let scale = 1.0 / (D as f64).sqrt() as f32;
    let q1 = Tensor::from_vec(fill(H * D, 7), (1, H, 1, D), &device)?.to_dtype(DType::F16)?;
    let k1 = Tensor::from_vec(fill(H_KV * KV_LEN * D, 11), (1, H_KV, KV_LEN, D), &device)?
        .to_dtype(DType::F16)?;
    let v1 = Tensor::from_vec(fill(H_KV * KV_LEN * D, 13), (1, H_KV, KV_LEN, D), &device)?
        .to_dtype(DType::F16)?;

    let reference = lloom::nn::ops::flash_decoding(
        &q1,
        &k1,
        &v1,
        scale,
        1.0,
        PAGE,
        K,
        lloom::nn::ops::FlashScratchSizing::Grow,
        KV_LEN.div_ceil(PAGE * K),
    )?
    .flatten_all()?
    .to_vec1::<f16>()?;

    // The fixture must be able to express the failure: an all-zero reference
    // would be matched by an untouched buffer, so a passing test would say
    // nothing (§15.1a's VACUOUS class).
    assert!(
        reference.iter().any(|x| f16::to_f32(*x) != 0.0),
        "the reference output is entirely zero, so an unwritten row would match it \
         and this test could not fail"
    );

    for b in [2usize, 4] {
        let q = Tensor::cat(&vec![&q1; b], 0)?;
        let k = Tensor::cat(&vec![&k1; b], 0)?;
        let v = Tensor::cat(&vec![&v1; b], 0)?;

        let out = lloom::nn::ops::flash_decoding(
            &q,
            &k,
            &v,
            scale,
            1.0,
            PAGE,
            K,
            lloom::nn::ops::FlashScratchSizing::Grow,
            KV_LEN.div_ceil(PAGE * K),
        )?;
        assert_eq!(out.dims(), [b, H, 1, D]);
        let got = out.flatten_all()?.to_vec1::<f16>()?;

        for row in 0..b {
            let slice = &got[row * H * D..(row + 1) * H * D];
            assert_eq!(
                slice,
                reference.as_slice(),
                "B={b} row {row} does not reproduce the B=1 answer: the grid's `y` extent \
                 must be (batch x q head) flattened, not the q head count, or this row is \
                 never written"
            );
        }
    }
    Ok(())
}

/// The flattened grid reaches a batch row by head stride alone, so a `k`/`v`
/// whose batch stride is not `n_kv_heads` head-strides must be refused.
///
/// **This is the precondition made loud rather than assumed** (`DESIGN.md` §3.5:
/// under `HazardTrackingModeUntracked` a wrong address is silent). It holds for
/// the contiguous `[B, H_kv, capacity, D]` cache the model passes — including
/// under the `narrow(2, 0, len)` the KV slot hands out, since a dim-2 narrow
/// changes the length and no stride — and it is what a paged or per-sequence
/// cache would break.
#[test]
fn flash_decoding_refuses_a_kv_whose_batch_stride_is_not_head_major() -> Result<()> {
    let device = Device::new_metal(0)?;

    const H: usize = 8;
    const H_KV: usize = 2;
    const D: usize = 64;
    const KV_LEN: usize = 64;
    const PAGE: usize = 32;

    let scale = 1.0 / (D as f64).sqrt() as f32;
    let q = Tensor::from_vec(fill(2 * H * D, 7), (2, H, 1, D), &device)?.to_dtype(DType::F16)?;
    // Narrowing dim 1 leaves `stride[0]` at the FULL head count while `dims[1]`
    // is the narrowed one, so the batch row is wider than `n_kv_heads` strides —
    // exactly the shape the flattening cannot address.
    let wide = Tensor::from_vec(
        fill(2 * (H_KV * 2) * KV_LEN * D, 11),
        (2, H_KV * 2, KV_LEN, D),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let k = wide.narrow(1, 0, H_KV)?;
    let v = wide.narrow(1, 0, H_KV)?;

    let err = lloom::nn::ops::flash_decoding(
        &q,
        &k,
        &v,
        scale,
        1.0,
        PAGE,
        1,
        lloom::nn::ops::FlashScratchSizing::Grow,
        KV_LEN.div_ceil(PAGE),
    );
    let msg = match err {
        Ok(_) => panic!(
            "a k/v whose batch stride is not `n_kv_heads` head-strides was ACCEPTED: the \
             flattened grid would read the wrong row, silently (DESIGN.md 3.5)"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("batch stride"),
        "refused for the wrong reason: {msg}"
    );
    Ok(())
}
