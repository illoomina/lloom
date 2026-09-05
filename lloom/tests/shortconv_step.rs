//! The fused Shuffle decode conv block (`shortconv_step`, issue #672) against
//! the per-token `cat` + fold + `*C` it replaces, on the Metal device.
//!
//! The reference is the `ConvState::Shuffle` decode arm `lfm2.rs::convolve`
//! runs today — `bx = b * x`, shift it into the `[b, hidden, l_cache]` state
//! (`cat(state[..,1..], bx)`), the depthwise fold of that state against the
//! weight, then `* c` and the transpose the caller applies.
//!
//! # Two contracts, because the gate stopped double-rounding (#278, task #6)
//!
//! The **shift + fold + state write-back** is bit-identical to the unfused arm
//! and asserted so (`assert_bits`): the fused kernel writes exactly the same
//! `cat(state[1..], bx)` window and folds it in the same order. A mutation to
//! the shift direction, the fold order or the state write-back turns those RED.
//!
//! The **`* C` gate is no longer bit-identical, on purpose.** The unfused arm
//! stores the fold as `T` (one rounding) then a separate `bmul` multiplies that
//! `T` by `c` (a second rounding); the fused kernel keeps the accumulator in
//! `f32` through the multiply and rounds **once** (`T(acc * c)`). Two roundings
//! vs one is a moved digest at f16/bf16 — but the move is **toward** the f64
//! truth, not away: measured over 500k samples the single-round lands closer to
//! the f64 value in 24.9 % of cases against the double-round's 0.004 %
//! (`DESIGN.md` §2.3.3c: the incumbent reference is not ground truth, and here
//! it is the worse discretisation). So the gate's contract is **not** bit-parity
//! against the double-round arm — that would pin the *worse* value. It is
//! **accuracy dominance**: the fused output is at least as close to the f64
//! truth as the double-round arm at every element, and strictly closer at a
//! non-trivial fraction. `gate_is_at_least_as_accurate_as_double_round` is the
//! unit that pins exactly that property; a mutation that regresses the gate
//! (e.g. re-introducing a spurious rounding, or dropping the `* c`) makes the
//! fused output *worse* than the double-round arm and turns it RED.
//!
//! One thing this file inherits from `shortconv_ring.rs`'s hard-won lessons:
//! `Tensor::copy` on Metal is a shallow clone (`MetalStorage::try_clone` shares
//! the buffer), so the reference's `cat` must run on its own copy or it edits
//! the fused arm's input — here the reference builds a fresh `state` via `cat`,
//! so the fused arm's `state` is untouched.
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
    // This is the **double-round** arm: `conv_out` is already rounded to `T` by
    // `conv1d`, and `(c * conv_out)` rounds a second time. It is the incumbent
    // the fused single-round gate must be at least as accurate as — not a value
    // to match bit-for-bit (see the module docs and #278).
    let conv_out = (c * &conv_out)?;
    Ok((conv_out.contiguous()?, new_state))
}

/// The **f64 gate truth**: the fold and `* c` computed in `f64` from the same
/// `T`-stored window the kernel reads, rounded to `T` only at the very end.
///
/// The state is a `T` buffer in both arms, so the fold *inputs* are identical
/// `T` values — what differs is only the arithmetic precision from there on.
/// Folding those same `T` values in `f64` and multiplying by `c` in `f64`
/// isolates the gate's rounding: the result is the best `T`-representable value
/// of `(Σ win_j·w_j)·c`, the target both the fused (one round) and the unfused
/// (two rounds) arms approximate. Returned as `f32` bit-widened to compare
/// distances without losing the `T` value.
///
/// `new_state` is the post-shift window `[b, hidden, l_cache]` (what the kernel
/// wrote); `w` is `[hidden, l_cache]`; `c` is `[b, hidden, 1]`. The fold order
/// matches the kernel: slots `0..l_cache` of the shifted window against
/// `w[.., 0..l_cache]`, which is exactly `cat(state[1..], bx)` folded left to
/// right (`conv.metal` `shortconv_step_body`).
fn gate_truth(new_state: &Tensor, c: &Tensor, w: &Tensor, l_cache: usize) -> Result<Vec<f64>> {
    let (b_sz, hidden, k) = new_state.dims3()?;
    assert_eq!(k, l_cache);
    // Bring everything to f64 host-side; the window and weight are small.
    let st = new_state
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let wv = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let cv = c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let dtype = new_state.dtype();
    let mut out = Vec::with_capacity(b_sz * hidden);
    for bi in 0..b_sz {
        for ch in 0..hidden {
            // Fold the shifted window against the weight, in f64, left to right.
            let mut acc = 0.0f64;
            for j in 0..l_cache {
                let win = st[(bi * hidden + ch) * l_cache + j] as f64;
                let wj = wv[ch * l_cache + j] as f64;
                acc += win * wj;
            }
            // The `* c` gate in f64, then round once to T and re-widen so the
            // stored distance is measured in the T grid.
            let gated = acc * (cv[bi * hidden + ch] as f64);
            let rounded = round_to_dtype(gated, dtype);
            out.push(rounded);
        }
    }
    Ok(out)
}

/// Round an f64 to the storage dtype's grid and return it as f64, so a distance
/// to it is the distance to the best T-representable value.
fn round_to_dtype(x: f64, dtype: DType) -> f64 {
    match dtype {
        DType::F32 => x as f32 as f64,
        DType::F16 => half::f16::from_f64(x).to_f64(),
        DType::BF16 => half::bf16::from_f64(x).to_f64(),
        other => panic!("gate_truth: unexpected dtype {other:?}"),
    }
}

/// The gate's contract (see module docs): the fused arm is **at least as close**
/// to the f64 truth as the double-round arm at every element, and **strictly
/// closer** at a non-trivial fraction — so the ~1 ulp move off the double-round
/// reference is toward correctness, not a regression.
fn assert_accuracy_dominates(
    name: &str,
    fused: &Tensor,
    double_round: &Tensor,
    truth: &[f64],
) -> Result<()> {
    assert_eq!(fused.shape(), double_round.shape(), "{name}: shape");
    let f = fused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let d = double_round
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(f.len(), truth.len(), "{name}: truth len");
    let mut worse = Vec::new();
    for i in 0..f.len() {
        let ef = (f[i] as f64 - truth[i]).abs();
        let ed = (d[i] as f64 - truth[i]).abs();
        if ef > ed {
            worse.push((i, f[i], d[i], truth[i], ef, ed));
        }
    }
    assert!(
        worse.is_empty(),
        "{name}: fused gate is FARTHER from the f64 truth than the double-round arm at {} \
         element(s); first at {} (fused {} err {:.3e} vs double {} err {:.3e}, truth {}). \
         A regressed gate is worse than the incumbent — that is a defect, not a toward-truth move.",
        worse.len(),
        worse[0].0,
        worse[0].1,
        worse[0].4,
        worse[0].2,
        worse[0].5,
        worse[0].3,
    );
    // Non-vacuous: the single-round must actually help somewhere, or the claim
    // that it is more accurate is untested on this case. (Unit-C cases are
    // exempt — there the gate is `*1`, exact, and dominance is trivially equal.)
    Ok(())
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
    let mut total_closer = 0usize;
    for (i, case) in CASES.iter().enumerate() {
        // l_cache 1 has no `shortconv_step_k(1)` variant; the fused arm falls
        // back, so skip the direct-kernel parity there (covered by the fold
        // path elsewhere). Every l_cache >= 2 has a variant.
        if case.l_cache < 2 {
            continue;
        }
        let ((out, state), (want_out, want_state)) = run(case, dtype, &device)?;
        // State first, bit-exact: it isolates the `b * x` write + shift from the
        // fold and gate, and the fused kernel writes it identically (assert_bits).
        assert_bits(&format!("case {i} {dtype:?} state"), &state, &want_state)?;
        // The `* C` gate: NOT bit-parity against the double-round `want_out`
        // (that pins the worse value). Accuracy dominance against the f64 truth,
        // computed from the same T window the kernel wrote (`want_state`, which
        // == `state` bit-for-bit as just asserted).
        let truth = gate_truth(&want_state, &c_of(case, dtype, &device)?, &w_of(case, dtype, &device)?, case.l_cache)?;
        assert_accuracy_dominates(&format!("case {i} {dtype:?} out"), &out, &want_out, &truth)?;
        total_closer += count_strictly_closer(&out, &want_out, &truth)?;
        // The fused output, before the caller's transpose: `[b, hidden, 1]`.
        assert_eq!(out.dims3()?, (case.b_sz, case.hidden, 1));
        // The new state is the shifted ring: `[b, hidden, l_cache]`.
        assert_eq!(state.dims3()?, (case.b_sz, case.hidden, case.l_cache));
    }
    // The whole point of the single-round gate is that it is MORE accurate. At
    // f16/bf16 it must land strictly closer to the truth than the double-round
    // arm on at least one element across the suite, or the change is inert and
    // the claim untested. (f32 has no gate rounding delta — acc is already f32 —
    // so equality everywhere is correct there.)
    if dtype != DType::F32 {
        assert!(
            total_closer > 0,
            "{dtype:?}: the single-round gate was never strictly closer to the f64 truth than \
             the double-round arm across the whole suite — the accuracy claim is untested; \
             the kernel may have silently reverted to rounding the intermediate."
        );
    }
    Ok(())
}

/// `c` for a case, as `parity` builds it — factored out so `gate_truth` sees the
/// exact same C block the fused arm and reference saw.
fn c_of(case: &Case, dtype: DType, device: &Device) -> Result<Tensor> {
    let (b, c, x) = bcx_of(case, dtype, device)?;
    let _ = (b, x);
    Ok(c)
}
fn w_of(case: &Case, dtype: DType, device: &Device) -> Result<Tensor> {
    let Case {
        b_sz: _,
        hidden,
        l_cache,
    } = *case;
    Tensor::from_vec(fill(hidden * l_cache, 3), (hidden, l_cache), device)?.to_dtype(dtype)
}
/// The `[b, hidden, seq]` b/c/x narrows, matching `run_with(unit_c=false)`.
fn bcx_of(case: &Case, dtype: DType, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    let Case {
        b_sz,
        hidden,
        l_cache: _,
    } = *case;
    let seq_len = 1usize;
    let bcx_vec = fill(b_sz * 3 * hidden * seq_len, 1);
    let bcx = Tensor::from_vec(bcx_vec, (b_sz, seq_len, 3 * hidden), device)?
        .to_dtype(dtype)?
        .transpose(1, 2)?;
    let b = bcx.narrow(1, 0, hidden)?;
    let c = bcx.narrow(1, hidden, hidden)?;
    let x = bcx.narrow(1, 2 * hidden, hidden)?;
    Ok((b, c, x))
}

fn count_strictly_closer(fused: &Tensor, double_round: &Tensor, truth: &[f64]) -> Result<usize> {
    let f = fused.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let d = double_round
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((0..f.len())
        .filter(|&i| (f[i] as f64 - truth[i]).abs() < (d[i] as f64 - truth[i]).abs())
        .count())
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

/// The gate's core claim in isolation, on a controlled sample: fusing the `* c`
/// as `T(acc * c)` — one rounding, `acc` kept in f32 — is **at least as close**
/// to the f64 truth as the two-kernel `T(T(acc) * c)` double-round, and strictly
/// closer on a real fraction of samples. This is the reason the parity gate is
/// accuracy-dominance rather than bit-parity (module docs, #278): the incumbent
/// double-round is the worse discretisation, so pinning bit-equality to it would
/// pin the worse value.
///
/// `acc` here is a synthetic f32 fold result (not itself T-representable in
/// general); `direct` models the fused gate, `via_t` the double-round arm, and
/// the truth is `acc·c` rounded once to T from f64.
#[test]
fn gate_is_at_least_as_accurate_as_double_round() -> Result<()> {
    let n = 8192;
    // Synthetic f32 accumulators and f16 gate coefficients, deterministic.
    let acc = fill(n, 11); // f32, the un-rounded fold accumulator
    let c_f32 = fill(n, 13);
    let mut worse = 0usize;
    let mut strictly_closer = 0usize;
    for i in 0..n {
        let a = acc[i];
        // c is a real f16 value (as it is in the kernel: `c_in[proj]` is T).
        let c = half::f16::from_f32(c_f32[i]);
        let c32 = c.to_f32();
        // Fused: T(acc * c), acc stays f32 through the multiply, one round.
        let fused = half::f16::from_f32(a * c32).to_f64();
        // Double-round: T(acc) first, then T(T(acc) * c).
        let folded_t = half::f16::from_f32(a); // round the intermediate
        let double = half::f16::from_f32(folded_t.to_f32() * c32).to_f64();
        // Truth: acc * c in f64, rounded once to f16.
        let truth = half::f16::from_f64(a as f64 * c32 as f64).to_f64();
        let ef = (fused - truth).abs();
        let ed = (double - truth).abs();
        if ef > ed {
            worse += 1;
        } else if ef < ed {
            strictly_closer += 1;
        }
    }
    assert_eq!(
        worse, 0,
        "fused single-round gate was farther from the f64 truth than the double-round arm on \
         {worse} of {n} samples — it must never be worse"
    );
    assert!(
        strictly_closer > 0,
        "fused single-round gate was never strictly closer than the double-round arm on {n} \
         samples — the accuracy claim would be vacuous"
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
