//! The fused RotatingRing conv block (`shortconv_ring`, issue #675) against
//! the per-position loop it replaces, on the Metal device.
//!
//! The reference is the loop the speculative verify writer runs today —
//! `bx = b * x` per position, `slice_set` into the ring slot, the depthwise
//! fold of the whole row against that slot's permuted weight, then `* c` —
//! built from the same tensor ops on the same device, so the assertion is
//! **bit equality**, not tolerance (`DESIGN.md` §2.3.5a: a fused arm that
//! moves the digest is a different kernel, and this one claims not to be).
//!
//! Three gates per `CONTRIBUTING.md` §3.1: parity (bit-equal to the loop at
//! several `seq_len`/batch/phase points, all three dtypes the kernel
//! instantiates), determinism (the fused arm agrees with itself across runs),
//! and the mutation — `measurements/issue-675-raw/mutate_shortconv_ring.sh`:
//! the fold order, the slot arithmetic and the forced rounding each turn
//! `parity_*` RED while `deterministic` stays green.
//!
//! Two things this file had to learn to be a test at all: `Tensor::copy` on
//! Metal is a shallow clone (`MetalStorage::try_clone` shares the buffer), so
//! the loop's `slice_set` writes the fused arm's input unless the fused arm
//! runs first; and under fast math a `T(acc) * c` in one kernel is not the
//! two-kernel `store, then multiply` unless the rounding is forced.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor, D};

/// A deterministic filler in `[-1, 1)` — a fixed pattern so a failure is
/// re-derivable and the two arms see identical bytes.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

/// The per-phase permuted weight table `[width, hidden, width]`, exactly as
/// `ShortConv::new` builds it: slot `s` at phase `phase` is `back` tokens old
/// where `back = (phase + width - s) % width`, live iff `back < l_cache`.
fn table(w: &Tensor, l_cache: usize, width: usize) -> Result<Tensor> {
    let (hidden, _) = w.dims2()?;
    let mut per_phase = Vec::with_capacity(width);
    for phase in 0..width {
        let mut cols = Vec::with_capacity(width);
        for s in 0..width {
            let back = (phase + width - s) % width;
            let col = if back < l_cache {
                w.narrow(1, l_cache - 1 - back, 1)?
            } else {
                Tensor::zeros((hidden, 1), w.dtype(), w.device())?
            };
            cols.push(col);
        }
        per_phase.push(Tensor::cat(&cols, 1)?.contiguous()?);
    }
    Tensor::stack(&per_phase, 0)?.contiguous()
}

/// The loop `lfm2.rs`'s speculative conv writer runs, one position at a time,
/// with the `*C` gate and the transpose the caller applies after it.
fn reference(
    b: &Tensor,
    x: &Tensor,
    c: &Tensor,
    state: &Tensor,
    table: &Tensor,
    phase_last: usize,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, hidden, width) = state.dims3()?;
    let seq_len = b.dim(2)?;
    let bx = (b * x)?.contiguous()?;
    // `slice_set` writes in place, and a `clone` shares storage: the loop
    // must run on its own copy or it edits the fused arm's input.
    let state = state.copy()?;
    let mut outs = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let phase = (phase_last + width - (seq_len - 1 - i)) % width;
        let col = bx.narrow(2, i, 1)?.contiguous()?;
        state.slice_set(&col, 2, phase)?;
        let w = table.get(phase)?;
        // `conv_fold` under `conv_fused`: the depthwise kernel over the whole
        // width-wide row.
        let _ = l_cache;
        outs.push(
            state
                .conv1d(&w.unsqueeze(1)?, 0, 1, 1, hidden)?
                .contiguous()?,
        );
    }
    let conv_out = Tensor::cat(&outs, 2)?.contiguous()?;
    let conv_out = (c * &conv_out)?;
    Ok((conv_out.transpose(1, 2)?.contiguous()?, state))
}

struct Case {
    b_sz: usize,
    hidden: usize,
    l_cache: usize,
    k: usize,
    seq_len: usize,
    phase_last: usize,
}

fn run(case: &Case, dtype: DType, device: &Device) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
    run_with(case, dtype, device, false)
}

/// As [`run`]; with `unit_c` the C block is all ones, so the gate multiply is
/// exact and what is compared is the ring write and the fold alone.
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
        k,
        seq_len,
        phase_last,
    } = *case;
    let width = l_cache + k;
    // b/x/c as sibling narrows of one `[b, 3*hidden, seq]` projection, exactly
    // the strided views `convolve` hands the kernel.
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
    let state = Tensor::from_vec(
        fill(b_sz * hidden * width, 2),
        (b_sz, hidden, width),
        device,
    )?
    .to_dtype(dtype)?;
    let w =
        Tensor::from_vec(fill(hidden * l_cache, 3), (hidden, l_cache), device)?.to_dtype(dtype)?;
    let table = table(&w, l_cache, width)?;

    // **The fused arm writes its state IN PLACE** — that is the ring's point
    // and what stops its state escaping a planned pass (`DESIGN.md` §9.2aa,
    // `pass-plan-design.md` §13). So each arm gets its own ring: the fused
    // one's is the tensor it mutates and then IS the result, and the loop's
    // is `reference`'s own copy. Comparing the two rings afterwards is the
    // stronger check it always was, now over a buffer the kernel edited
    // rather than one it produced.
    //
    // `Tensor::copy` is shallow on Metal (`try_clone` returns `self.clone()`),
    // so the fused arm's ring is built by its own `from_vec` rather than
    // copied from `state` — a shallow copy would have the two arms share one
    // buffer and the comparison would pass by aliasing.
    let fused_state = Tensor::from_vec(
        fill(b_sz * hidden * width, 2),
        (b_sz, hidden, width),
        device,
    )?
    .to_dtype(dtype)?;
    let out = lloom::nn::ops::shortconv_ring(&b, &x, &c, &fused_state, &table, phase_last)?;
    let want = reference(&b, &x, &c, &state, &table, phase_last, l_cache)?;
    Ok(((out, fused_state), want))
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
    // The DSpark verify window on the 2.6B ring: l_cache 3, K+1 = 10 slots,
    // 10 positions, batch 1, an arbitrary phase.
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 3,
        k: 10,
        seq_len: 10,
        phase_last: 4,
    },
    // A partial window and the wrap: the positions straddle slot 0.
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 3,
        k: 10,
        seq_len: 4,
        phase_last: 1,
    },
    // The K=1 decode step on the same ring.
    Case {
        b_sz: 1,
        hidden: 64,
        l_cache: 3,
        k: 10,
        seq_len: 1,
        phase_last: 12,
    },
    // Batched, so the batch strides are exercised.
    Case {
        b_sz: 2,
        hidden: 32,
        l_cache: 3,
        k: 5,
        seq_len: 6,
        phase_last: 7,
    },
    // A ring with no history slots (K = 0 → width = l_cache) at full width.
    Case {
        b_sz: 1,
        hidden: 16,
        l_cache: 3,
        k: 0,
        seq_len: 3,
        phase_last: 2,
    },
];

fn parity(dtype: DType) -> Result<()> {
    let device = Device::new_metal(0)?;
    for (i, case) in CASES.iter().enumerate() {
        let ((out, state), (want_out, want_state)) = run(case, dtype, &device)?;
        // State first: it isolates the `b * x` write from the fold and gate.
        assert_bits(&format!("case {i} {dtype:?} state"), &state, &want_state)?;
        assert_bits(&format!("case {i} {dtype:?} out"), &out, &want_out)?;
        // The fused output is what `out_proj` reads: `[b, seq, hidden]`.
        assert_eq!(out.dims3()?, (case.b_sz, case.seq_len, case.hidden));
    }
    Ok(())
}

#[test]
fn parity_f16() -> Result<()> {
    parity(DType::F16)
}

/// Windows that fill the ring, and the positions whose live taps a LATER
/// position of the same pass overwrites — the shape where an arm that reads
/// the ring after the pass (rather than as it is written) goes wrong. Every
/// variant must be bit-equal; the `k = 0` rows are the K=0 ring (width equals
/// `l_cache`, no history slot), where the loop's `conv1d` takes the `_k3`
/// depthwise variant rather than the generic one.
#[test]
fn full_windows_and_late_overwrites() -> Result<()> {
    let device = Device::new_metal(0)?;
    let variants = [
        (
            "k0 seq3 h16",
            Case {
                b_sz: 1,
                hidden: 16,
                l_cache: 3,
                k: 0,
                seq_len: 3,
                phase_last: 2,
            },
        ),
        (
            "k0 seq2 h16",
            Case {
                b_sz: 1,
                hidden: 16,
                l_cache: 3,
                k: 0,
                seq_len: 2,
                phase_last: 2,
            },
        ),
        (
            "k1 seq3 h16",
            Case {
                b_sz: 1,
                hidden: 16,
                l_cache: 3,
                k: 1,
                seq_len: 3,
                phase_last: 2,
            },
        ),
        (
            "k1 seq4 h16",
            Case {
                b_sz: 1,
                hidden: 16,
                l_cache: 3,
                k: 1,
                seq_len: 4,
                phase_last: 2,
            },
        ),
        (
            "k0 seq3 h64",
            Case {
                b_sz: 1,
                hidden: 64,
                l_cache: 3,
                k: 0,
                seq_len: 3,
                phase_last: 2,
            },
        ),
        (
            "k10 seq13 h64",
            Case {
                b_sz: 1,
                hidden: 64,
                l_cache: 3,
                k: 10,
                seq_len: 13,
                phase_last: 4,
            },
        ),
        (
            "k10 seq12 h64",
            Case {
                b_sz: 1,
                hidden: 64,
                l_cache: 3,
                k: 10,
                seq_len: 12,
                phase_last: 4,
            },
        ),
    ];
    for dtype in [DType::F16, DType::F32] {
        for (name, case) in variants.iter() {
            let ((out, state), (want_out, want_state)) = run_with(case, dtype, &device, false)?;
            assert_bits(&format!("{name} {dtype:?} state"), &state, &want_state)?;
            assert_bits(&format!("{name} {dtype:?} out"), &out, &want_out)?;
        }
    }
    Ok(())
}

/// Candle's f16 elementwise multiply is the correctly rounded half product —
/// `a * b` on the device against `(a.f32 * b.f32).f16`, bitwise. The gate in
/// the fused kernel is written to match THAT (`folded * c` in T after a forced
/// rounding of the accumulator); this pins the reference it matches.
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
    // The strided form the gate takes: `c` is a narrow of a transposed view.
    let big = Tensor::from_vec(fill(3 * n, 17), (1, n, 3), &device)?
        .to_dtype(DType::F16)?
        .transpose(1, 2)?;
    let c = big.narrow(1, 1, 1)?; // [1, 1, n], stride n... a strided view
    let d = Tensor::from_vec(fill(n, 19), (1, 1, n), &device)?.to_dtype(DType::F16)?;
    let direct = (&c * &d)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let via_f32 = (c.to_dtype(DType::F32)? * d.to_dtype(DType::F32)?)?
        .flatten_all()?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let bad2 = (0..n)
        .filter(|&i| direct[i].to_bits() != via_f32[i].to_bits())
        .count();
    assert_eq!(
        bad2, 0,
        "strided f16 c*d is not the correctly rounded product"
    );
    Ok(())
}

/// The fold alone (C = 1), per case and dtype: which stage disagrees when
/// `parity_*` does.
#[test]
fn fold_only_parity() -> Result<()> {
    let device = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32, DType::BF16] {
        for (i, case) in CASES.iter().enumerate() {
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

#[test]
fn parity_f32() -> Result<()> {
    parity(DType::F32)
}

#[test]
fn parity_bf16() -> Result<()> {
    parity(DType::BF16)
}

/// The loop is the reference, so the reference must itself be non-vacuous:
/// a state change is observable (the ring after the pass differs from the
/// ring before it) and the output is not a constant.
#[test]
fn reference_is_not_vacuous() -> Result<()> {
    let device = Device::new_metal(0)?;
    let ((out, state), _) = run(&CASES[0], DType::F16, &device)?;
    let width = CASES[0].l_cache + CASES[0].k;
    let before = Tensor::from_vec(
        fill(CASES[0].b_sz * CASES[0].hidden * width, 2),
        (CASES[0].b_sz, CASES[0].hidden, width),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let moved = (state.to_dtype(DType::F32)? - before.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(moved > 0.0, "the pass wrote nothing into the ring");
    let spread = (out.to_dtype(DType::F32)?.max_all()? - out.to_dtype(DType::F32)?.min_all()?)?
        .to_scalar::<f32>()?;
    assert!(spread > 0.0, "the output is a constant");
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

/// A window wider than the ring is refused, not truncated.
#[test]
fn refuses_a_window_wider_than_the_ring() -> Result<()> {
    let device = Device::new_metal(0)?;
    let case = Case {
        b_sz: 1,
        hidden: 8,
        l_cache: 3,
        k: 2,
        seq_len: 6,
        phase_last: 0,
    };
    let err = run(&case, DType::F16, &device);
    assert!(
        err.is_err(),
        "6 positions into a 5-wide ring must be refused"
    );
    // Keep the unused import honest under every cfg.
    let _ = D::Minus1;
    Ok(())
}
