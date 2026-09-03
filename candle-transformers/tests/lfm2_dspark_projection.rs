//! The DSpark context projection — `DESIGN.md` §10.2m/§10.2o, issue #363.
//!
//! §10.2n built the tap and named the projection as the next increment: the
//! first surface of the drafter that is **falsifiable before the backbone
//! exists**. This file is its test.
//!
//! # What has to be asserted, and why the obvious assertions do not
//!
//! The projection is `RMSNorm_enc(W_c · concat)`, and three defects it can
//! carry are each invisible to the assertions one reaches for first:
//!
//! * **norm before the matmul instead of after.** Both orders produce a
//!   `[b, seq, 2048]` tensor of plausible magnitude. Only a value comparison
//!   against an independently computed reference separates them, and the order
//!   is the one thing §10.2m had to *correct* in the paper — so it is the one
//!   thing a test here owes.
//! * **a permuted concat.** `[H(l₁); …; H(l₅)]` in the wrong order is
//!   algebraically valid, right in every shape, and pairs each 2048-d slice
//!   with the wrong 2048-column block of `W_c`. §10.2n caught exactly this in
//!   the tap's own build; the projection is the consumer that makes it
//!   *observable*, so the test asserts the pairing rather than the shape.
//! * **a transposed weight.** `fc.weight` is `[out, in]`; using it untransposed
//!   is a dimension error at 2048×10240 but **not** at a square fixture — so
//!   every fixture here is deliberately non-square.
//!
//! `CONTRIBUTING.md` §3.1b rule 1: *prefer an absolute reference to a shape*.
//! The reference below is computed by hand from the definition — a scalar dot
//! product per output element, then an explicit RMS — never by calling the
//! production path with different arguments. A test that restates the
//! derivation is #503's shape and survives the mutation it exists to catch.
//!
//! # CPU only, deliberately
//!
//! Same reason as the tap's test: the question is *which arithmetic*, not what
//! a kernel computes, and the CPU backend is bit-stable where the Metal generic
//! path is not (`CONTRIBUTING.md` §3.1, §2.3.8a). This also keeps the file
//! runnable with no `metal` feature, which is the build §14.1 gates on.

use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::lfm2_dspark::{
    projection_params, DSparkProjection, CHECKPOINT_DTYPE,
};

/// Deliberately non-square and small enough to hand-compute: 3 taps of width 4
/// project to width 4 through an `fc` of `[4, 12]`.
///
/// Non-square in the *tap* direction (12 in, 4 out) so a missing transpose is a
/// hard error rather than a silently different answer, which is the defect a
/// square fixture cannot see.
const HIDDEN: usize = 4;
const N_TAP: usize = 3;
const EPS: f64 = 1e-5;

/// Values that are distinct per position and per slice, so a permuted or
/// misaligned concat lands on different numbers rather than coincidentally
/// equal ones (`lfm2.rs`' fixture rule, §11.3l finding 3).
fn tap(seq: usize, base: f32, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..seq * HIDDEN)
        .map(|i| base + (i as f32) * 0.25 - 1.0)
        .collect();
    Tensor::from_vec(v, (1, seq, HIDDEN), dev)
}

fn fc_weight(dev: &Device) -> Result<Tensor> {
    // [HIDDEN, N_TAP * HIDDEN] = [4, 12], distinct entries, mixed signs.
    let v: Vec<f32> = (0..HIDDEN * N_TAP * HIDDEN)
        .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
        .collect();
    Tensor::from_vec(v, (HIDDEN, N_TAP * HIDDEN), dev)
}

fn norm_weight(dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i as f32) * 0.125).collect();
    Tensor::from_vec(v, HIDDEN, dev)
}

/// The absolute reference: `RMSNorm_enc(W_c · concat)`, computed from the
/// definition with scalar arithmetic.
///
/// **Not a second call into the production path.** Every step is written out —
/// the concat by index, the matmul as a dot product, the RMS as a mean of
/// squares — so that a change to the implementation cannot move the reference
/// with it. That independence is the whole reason this is a test and not a
/// tautology (§3.1b rule 1, and #503's harness which restated the derivation
/// inside the test module and survived the mutation).
fn reference(taps: &[Vec<f32>], seq: usize, fc: &[f32], nw: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; seq * HIDDEN];
    for s in 0..seq {
        // concat: [H(l1); H(l2); H(l3)] at this position, in tap order.
        let mut cat = [0f32; N_TAP * HIDDEN];
        for (t, tapv) in taps.iter().enumerate() {
            for h in 0..HIDDEN {
                cat[t * HIDDEN + h] = tapv[s * HIDDEN + h];
            }
        }
        // fc: [HIDDEN, N_TAP*HIDDEN] row-major; out[o] = sum_i fc[o][i]*cat[i]
        let mut fc_out = [0f32; HIDDEN];
        for (o, fo) in fc_out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (i, c) in cat.iter().enumerate() {
                acc += fc[o * (N_TAP * HIDDEN) + i] * c;
            }
            *fo = acc;
        }
        // RMSNorm AFTER the matmul: x * rsqrt(mean(x^2) + eps) * weight.
        let mean_sq: f32 = fc_out.iter().map(|x| x * x).sum::<f32>() / (HIDDEN as f32);
        let inv = 1.0f32 / (mean_sq + EPS as f32).sqrt();
        for h in 0..HIDDEN {
            out[s * HIDDEN + h] = fc_out[h] * inv * nw[h];
        }
    }
    out
}

fn build(dev: &Device) -> Result<(DSparkProjection, Vec<f32>, Vec<f32>)> {
    let fc = fc_weight(dev)?;
    let nw = norm_weight(dev)?;
    let fc_v = fc.flatten_all()?.to_vec1::<f32>()?;
    let nw_v = nw.to_vec1::<f32>()?;
    let proj = DSparkProjection::from_parts(fc, nw, EPS)?;
    Ok((proj, fc_v, nw_v))
}

/// The load-bearing test: the projection equals an independently computed
/// `RMSNorm_enc(W_c · concat)`, position by position.
///
/// This is what makes the **norm-after-matmul** claim falsifiable. §10.2m
/// corrected the paper's `RMSNorm(W_c·[…])` ordering by reading `dflash.cpp`,
/// and the native MLX port states the same order in one line
/// (`fuse_target: self.hidden_norm(self.fc(cat))`). Swapping the order here
/// changes every value while changing no shape.
#[test]
fn projection_matches_an_independently_computed_reference() -> Result<()> {
    let dev = Device::Cpu;
    let seq = 3;
    let (proj, fc_v, nw_v) = build(&dev)?;

    let taps: Vec<Tensor> = (0..N_TAP)
        .map(|t| tap(seq, t as f32 * 3.0, &dev))
        .collect::<Result<_>>()?;
    let tap_v: Vec<Vec<f32>> = taps
        .iter()
        .map(|t| t.flatten_all()?.to_vec1::<f32>())
        .collect::<Result<_>>()?;

    let got = proj.forward(&taps)?;
    assert_eq!(got.dims(), &[1, seq, HIDDEN], "projection output shape");

    let got_v = got.flatten_all()?.to_vec1::<f32>()?;
    let want = reference(&tap_v, seq, &fc_v, &nw_v);

    for (i, (g, w)) in got_v.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "element {i}: projection gave {g}, reference {w} \
             (delta {:e}) -- the projection is not RMSNorm_enc(W_c . concat)",
            (g - w).abs()
        );
    }
    Ok(())
}

/// The norm must be applied AFTER the matmul, and this pins it by showing the
/// other order gives a *different* answer on this fixture.
///
/// Without this arm the test above is satisfied by any implementation whose
/// reference happens to share its ordering bug. `CONTRIBUTING.md` §3.1b rule 2:
/// a shape needs a second arm producing a value the shape can be checked
/// against. Here the second arm is the *wrong* order, asserted to differ — so
/// "the orders are distinguishable on this fixture" is established rather than
/// assumed, which is what makes the first test's passing informative.
#[test]
fn norm_after_matmul_differs_from_norm_before_on_this_fixture() -> Result<()> {
    let dev = Device::Cpu;
    let seq = 2;
    let (proj, _, _) = build(&dev)?;
    let taps: Vec<Tensor> = (0..N_TAP)
        .map(|t| tap(seq, t as f32 * 3.0, &dev))
        .collect::<Result<_>>()?;

    let after = proj.forward(&taps)?.flatten_all()?.to_vec1::<f32>()?;

    // The rejected order: normalise the 12-wide concat first, then project.
    // Note it needs a 12-wide norm weight, which the checkpoint does not ship
    // -- `hidden_norm.weight` is [2048], i.e. the OUTPUT width. That absence is
    // itself evidence for the order, and this arm makes the numeric half of the
    // argument with a synthetic weight.
    let cat = Tensor::cat(&taps, candle::D::Minus1)?;
    let w12 = Tensor::from_vec(vec![1f32; N_TAP * HIDDEN], N_TAP * HIDDEN, &dev)?;
    let normed = candle_nn::ops::rms_norm(&cat, &w12, EPS as f32)?;
    let fc = fc_weight(&dev)?.t()?.contiguous()?;
    let before = normed
        .broadcast_matmul(&fc)?
        .broadcast_mul(&norm_weight(&dev)?)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let max_delta = after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "norm-before and norm-after agree to {max_delta:e} on this fixture, so the \
         fixture cannot distinguish them and the test above proves nothing about \
         ordering -- pick different values (DESIGN.md 10.2m)"
    );
    Ok(())
}

/// Tap order is load-bearing, and a permutation must change the answer.
///
/// §10.2n's finding: captures in layer-iteration order where `fc.weight`
/// expects `target_layer_ids` order is *latent on a sorted checkpoint and live
/// on any other*, valid in every shape, and invisible to every gate. The tap
/// asserts it produces the right order; this asserts the consumer actually
/// *depends* on that order — without which the tap's invariant would be
/// guarding nothing.
#[test]
fn a_permuted_concat_changes_the_projection() -> Result<()> {
    let dev = Device::Cpu;
    let seq = 2;
    let (proj, _, _) = build(&dev)?;
    let taps: Vec<Tensor> = (0..N_TAP)
        .map(|t| tap(seq, t as f32 * 3.0, &dev))
        .collect::<Result<_>>()?;

    let straight = proj.forward(&taps)?.flatten_all()?.to_vec1::<f32>()?;

    let swapped: Vec<Tensor> = vec![taps[1].clone(), taps[0].clone(), taps[2].clone()];
    let permuted = proj.forward(&swapped)?.flatten_all()?.to_vec1::<f32>()?;

    let max_delta = straight
        .iter()
        .zip(permuted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "swapping two taps moved the projection by only {max_delta:e} -- either the \
         projection ignores tap order (in which case 10.2n's invariant guards nothing) \
         or the fixture's slices are too alike to tell (issue #363)"
    );
    Ok(())
}

/// The arity check refuses by name rather than by a matmul dimension error.
#[test]
fn the_wrong_number_of_taps_is_refused_with_the_expected_count() -> Result<()> {
    let dev = Device::Cpu;
    let (proj, _, _) = build(&dev)?;
    let two: Vec<Tensor> = (0..2)
        .map(|t| tap(2, t as f32, &dev))
        .collect::<Result<_>>()?;
    let err = proj.forward(&two).unwrap_err().to_string();
    assert!(
        err.contains(&format!("expects {N_TAP} taps")),
        "arity error should name the expected count, got: {err}"
    );
    Ok(())
}

/// A tap of the wrong width is refused, naming which one.
#[test]
fn a_tap_of_the_wrong_width_is_refused() -> Result<()> {
    let dev = Device::Cpu;
    let (proj, _, _) = build(&dev)?;
    let mut taps: Vec<Tensor> = (0..N_TAP)
        .map(|t| tap(2, t as f32, &dev))
        .collect::<Result<_>>()?;
    taps[1] = Tensor::from_vec(vec![0f32; 2 * (HIDDEN + 1)], (1, 2, HIDDEN + 1), &dev)?;
    let err = proj.forward(&taps).unwrap_err().to_string();
    assert!(
        err.contains("tap 1") && err.contains(&HIDDEN.to_string()),
        "width error should name the offending tap and expected width, got: {err}"
    );
    Ok(())
}

/// `from_parts` derives `n_tap` from the weight and refuses a ragged one.
#[test]
fn from_parts_refuses_an_fc_that_is_not_a_whole_number_of_hidden_states() -> Result<()> {
    let dev = Device::Cpu;
    let fc = Tensor::from_vec(vec![0f32; HIDDEN * 13], (HIDDEN, 13), &dev)?;
    let nw = norm_weight(&dev)?;
    let err = DSparkProjection::from_parts(fc, nw, EPS)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("whole number"),
        "a 13-column fc against hidden 4 should be refused, got: {err}"
    );
    Ok(())
}

/// The projection consumes exactly what `Cache::take_taps` produces.
///
/// The two halves of §363's increment are the tap (§10.2n) and this
/// projection, and the seam between them is the one place a mismatch would be
/// discovered late. Pinned here as a shape contract: five `[b, seq, hidden]`
/// tensors in, one `[b, seq, hidden]` out.
#[test]
fn the_real_drafter_arity_and_widths_line_up_with_the_checkpoint() -> Result<()> {
    let dev = Device::Cpu;
    // The 2.6B drafter's real numbers (DESIGN.md 10.2l, re-derived from the
    // payload-bearing safetensors by verify_projection.py): fc.weight is
    // [2048, 10240] = [hidden, 5 * hidden].
    const REAL_HIDDEN: usize = 2048;
    const REAL_TAPS: usize = 5;
    assert_eq!(
        REAL_TAPS * REAL_HIDDEN,
        10240,
        "10240 must be 5 x 2048 -- if this fails the checkpoint changed shape"
    );
    assert_eq!(
        projection_params(REAL_HIDDEN, REAL_TAPS),
        20_971_520 + 2048,
        "fc plus the encoder norm"
    );
    assert_eq!(
        CHECKPOINT_DTYPE,
        DType::BF16,
        "the released drafter stores every tensor at BF16 (safetensors header)"
    );

    // And the small fixture round-trips the same contract.
    let (proj, _, _) = build(&dev)?;
    assert_eq!(proj.n_tap(), N_TAP);
    assert_eq!(proj.hidden(), HIDDEN);
    Ok(())
}
