//! The fused Shuffle decode conv block (`shortconv_step`, issue #672) against
//! the per-token `cat` + fold + `*C` it replaces, on the Metal device.
//!
//! The reference is the `ConvState::Shuffle` decode arm `lfm2.rs::convolve`
//! runs today — `bx = b * x`, shift it into the `[b, hidden, l_cache]` state
//! (`cat(state[..,1..], bx)`), the depthwise fold of that state against the
//! weight, then `* c` and the transpose the caller applies. Built from the same
//! tensor ops on the same device, so the assertion is **bit equality**, not
//! tolerance (`DESIGN.md` §2.3.5a: a fused arm that moves the digest is a
//! different kernel, and this one claims not to be).
//!
//! Three gates per `CONTRIBUTING.md` §3.1: parity (bit-equal to the unfused arm
//! at several batch/hidden/phase points, all three dtypes the kernel
//! instantiates), determinism (the fused arm agrees with itself across runs),
//! and the mutation — `measurements/issue-672-raw/mutate_shortconv_step.sh`:
//! the shift direction, the fold order, the state write-back and the forced
//! rounding each turn `parity_*` RED while `deterministic` stays green.
//!
//! Two things this file inherits from `shortconv_ring.rs`'s hard-won lessons:
//! `Tensor::copy` on Metal is a shallow clone (`MetalStorage::try_clone` shares
//! the buffer), so the reference's `slice_set`/`cat` must run on its own copy or
//! it edits the fused arm's input — here the reference builds a fresh `state`
//! via `cat`, so the fused arm's `state` is untouched; and under fast math a
//! `T(acc) * c` in one kernel is not the two-kernel `store, then multiply`
//! unless the rounding is forced, which `f16_multiply_rounding` pins.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor, D};

/// A deterministic filler in `[-1, 1)` — a fixed pattern so a failure is
/// re-derivable and the two arms see identical bytes.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

/// The unfused `ConvState::Shuffle` decode arm, up to but excluding the final
/// transpose the caller applies: `bx = b*x`, shift into the `l_cache`-wide
/// state, depthwise fold, `*c`. `seq_len == 1` — the decode step.
///
/// Returns `(conv_out `[b, hidden, 1]`, new_state `[b, hidden, l_cache]`)`, the
/// same pair `shortconv_step` returns. **The transpose is the CALLER's**
/// (`convolve` does `conv_out.transpose(1, 2)?.contiguous()` on the fused arm's
/// output too, `lfm2.rs`), so it is outside the kernel's contract and outside
/// this reference — comparing before it isolates exactly what the kernel does.
fn reference(
    b: &Tensor,
    x: &Tensor,
    c: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, hidden, _) = state.dims3()?;
    // `bx = (b * x).contiguous()`, exactly `convolve`'s line.
    let bx = (b * x)?.contiguous()?;
    // Shift the cache and add the new token: `cat(state[.., 1.., ..], bx)`.
    // This builds a FRESH tensor, so the fused arm's `state` input is untouched
    // (no shared-storage hazard — the reason `shortconv_ring` had to copy).
    let new_state = if l_cache > 1 {
        let tail = state.narrow(2, 1, l_cache - 1)?;
        Tensor::cat(&[tail, bx.clone()], 2)?
    } else {
        bx.clone()
    };
    // `conv_fold` under `conv_fused`: the depthwise kernel over the whole
    // `l_cache`-wide state.
    let conv_out = new_state
        .conv1d(&weight.unsqueeze(1)?, 0, 1, 1, hidden)?
        .contiguous()?;
    // The `*C` gate. NOT transposed — the caller applies that to both arms.
    let conv_out = (c * &conv_out)?;
    Ok((conv_out.contiguous()?, new_state))
}

struct Case {
    b_sz: usize,
    hidden: usize,
    l_cache: usize,
}

fn run(case: &Case, dtype: DType, device: &Device) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
    run_with(case, dtype, device, false)
}

/// As [`run`]; with `unit_c` the C block is all ones, so the gate multiply is
/// exact and what is compared is the shift and the fold alone.
fn run_with(
    case: &Case,
    dtype: DType,
    device: &Device,
    unit_c: bool,
) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
    let Case {
        b_sz,
        hidden,
        l_cache,
    } = *case;
    // seq_len == 1: shortconv_step is the decode step, one output per channel.
    let seq_len = 1usize;
    // b/x/c as sibling narrows of one `[b, 3*hidden, seq]` projection, exactly
    // the strided views `convolve` hands the kernel (in-proj output, transposed).
    let mut bcx_vec = fill(b_sz * 3 * hidden * seq_len, 1);
    if unit_c {
        for bi in 0..b_sz {
            for l in 0..seq_len {
                for ch in 0..hidden {
                    bcx_vec[(bi * seq_len + l) * 3 * hidden + hidden + ch] = 1.0;
                }
            }
        }
    }
    let bcx = Tensor::from_vec(bcx_vec, (b_sz, seq_len, 3 * hidden), device)?
        .to_dtype(dtype)?
        .transpose(1, 2)?;
    let b = bcx.narrow(1, 0, hidden)?;
    let c = bcx.narrow(1, hidden, hidden)?;
    let x = bcx.narrow(1, 2 * hidden, hidden)?;
    // The state is `[b, hidden, l_cache]` — contiguous, the Shuffle slot.
    let state = Tensor::from_vec(
        fill(b_sz * hidden * l_cache, 2),
        (b_sz, hidden, l_cache),
        device,
    )?
    .to_dtype(dtype)?;
    // The squeezed conv weight `[hidden, l_cache]`, as `convolve` passes it.
    let w = Tensor::from_vec(fill(hidden * l_cache, 3), (hidden, l_cache), device)?
        .to_dtype(dtype)?;

    // The fused arm first: it does not write its inputs. The reference builds a
    // fresh state via `cat`, so order is not load-bearing here — but kept for
    // symmetry with `shortconv_ring`'s contract.
    let got = lloom::nn::ops::shortconv_step(&b, &x, &c, &state, &w, l_cache)?;
    let want = reference(&b, &x, &c, &state, &w, l_cache)?;
    Ok((got, want))
}

fn assert_bits(name: &str, got: &Tensor, want: &Tensor) -> Result<()> {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{name}: dtype");
    let g = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let w = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bad: Vec<usize> = (0..g.len())
        .filter(|&i| g[i].to_bits() != w[i].to_bits())
        .collect();
    assert!(
        bad.is_empty(),
        "{name}: {} of {} elements differ; first at {} (fused {} vs loop {}); indices {:?}",
        bad.len(),
        g.len(),
        bad[0],
        g[bad[0]],
        w[bad[0]],
        &bad[..bad.len().min(48)]
    );
    Ok(())
}

const CASES: &[Case] = &[
    // The 2.6B decode step: l_cache 3, hidden 64, batch 1 — the crown Shuffle
    // conv the fused kernel replaces.
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 3,
    },
    // Batched, so the batch strides are exercised.
    Case {
        b_sz: 2,
        hidden: 32,
        l_cache: 3,
    },
    // A wider hidden, still l_cache 3.
    Case {
        b_sz: 1,
        hidden: 128,
        l_cache: 3,
    },
    // l_cache 2 (the `_k2` fold) and 4 (`_k4`), so every instantiated width is
    // covered — the `shortconv_step_k(k)` lookup is total for these.
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 2,
    },
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 4,
    },
    // l_cache 1: the degenerate ring where the state IS the new token (no shift).
    Case {
        b_sz: 1,
        hidden: 16,
        l_cache: 1,
    },
];

fn parity(dtype: DType) -> Result<()> {
    let device = Device::new_metal(0)?;
    for (i, case) in CASES.iter().enumerate() {
        // l_cache 1 has no `shortconv_step_k(1)` variant; the fused arm falls
        // back, so skip the direct-kernel parity there (covered by the fold
        // path elsewhere). Every l_cache >= 2 has a variant.
        if case.l_cache < 2 {
            continue;
        }
        let ((out, state), (want_out, want_state)) = run(case, dtype, &device)?;
        // State first: it isolates the `b * x` write + shift from the fold and gate.
        assert_bits(&format!("case {i} {dtype:?} state"), &state, &want_state)?;
        assert_bits(&format!("case {i} {dtype:?} out"), &out, &want_out)?;
        // The fused output, before the caller's transpose: `[b, hidden, 1]`.
        assert_eq!(out.dims3()?, (case.b_sz, case.hidden, 1));
        // The new state is the shifted ring: `[b, hidden, l_cache]`.
        assert_eq!(state.dims3()?, (case.b_sz, case.hidden, case.l_cache));
    }
    Ok(())
}

#[test]
fn parity_f16() -> Result<()> {
    parity(DType::F16)
}

#[test]
fn parity_f32() -> Result<()> {
    parity(DType::F32)
}

#[test]
fn parity_bf16() -> Result<()> {
    parity(DType::BF16)
}

/// The fold + shift alone (C = 1), per case and dtype: which stage disagrees
/// when `parity_*` does. Isolates the `b*x` write and the shifted fold from the
/// gate multiply's rounding.
#[test]
fn fold_only_parity() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32, DType::BF16] {
        for (i, case) in CASES.iter().enumerate() {
            if case.l_cache < 2 {
                continue;
            }
            let ((out, state), (want_out, want_state)) = run_with(case, dtype, &device, true)?;
            assert_bits(
                &format!("fold-only case {i} {dtype:?} state"),
                &state,
                &want_state,
            )?;
            assert_bits(
                &format!("fold-only case {i} {dtype:?} out"),
                &out,
                &want_out,
            )?;
        }
    }
    Ok(())
}

/// Candle's f16 elementwise multiply is the correctly rounded half product, and
/// the gate in the fused kernel is written to match THAT (`folded * c` in T
/// after a forced rounding of the accumulator). This pins the reference the
/// gate matches, the same invariant `shortconv_ring` depends on.
#[test]
fn f16_multiply_rounding() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 4096;
    let a = Tensor::from_vec(fill(n, 11), n, &device)?.to_dtype(DType::F16)?;
    let b = Tensor::from_vec(fill(n, 13), n, &device)?.to_dtype(DType::F16)?;
    let direct = (&a * &b)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let via_f32 = (a.to_dtype(DType::F32)? * b.to_dtype(DType::F32)?)?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let bad = (0..n)
        .filter(|&i| direct[i].to_bits() != via_f32[i].to_bits())
        .count();
    assert_eq!(
        bad, 0,
        "f16 a*b is not the correctly rounded product on {n} samples"
    );
    Ok(())
}

/// The reference must itself be non-vacuous: the shift writes the ring (the
/// state after differs from the state before) and the output is not a constant.
#[test]
fn reference_is_not_vacuous() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ((out, state), _) = run(&CASES[0], DType::F16, &device)?;
    let before = Tensor::from_vec(
        fill(CASES[0].b_sz * CASES[0].hidden * CASES[0].l_cache, 2),
        (CASES[0].b_sz, CASES[0].hidden, CASES[0].l_cache),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let moved = (state.to_dtype(DType::F32)? - before.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(moved > 0.0, "the shift wrote nothing into the ring");
    let spread = (out.to_dtype(DType::F32)?.max_all()? - out.to_dtype(DType::F32)?.min_all()?)?
        .to_scalar::<f32>()?;
    assert!(spread > 0.0, "the output is a constant");
    // Keep the unused import honest under every cfg.
    let _ = D::Minus1;
    Ok(())
}

/// Run to run, the fused arm gives the same bytes (§2.3.6).
#[test]
fn deterministic() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ((out0, st0), _) = run(&CASES[0], DType::F16, &device)?;
    for r in 1..4 {
        let ((out, st), _) = run(&CASES[0], DType::F16, &device)?;
        assert_bits(&format!("run {r} out"), &out, &out0)?;
        assert_bits(&format!("run {r} state"), &st, &st0)?;
    }
    Ok(())
}
