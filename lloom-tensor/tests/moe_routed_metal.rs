//! The routed-expert quantized matmul (`QTensor::indexed_moe_forward`,
//! `kernel_mul_mv_id_*` on Metal; issue #747): parity of the Metal arm and
//! the CPU arm against a dense reference built from the dequantized experts,
//! over the weight dtypes the Qwen checkpoint routes through (Q4_0/Q8_0/Q4K
//! as quantizable stand-ins, IQ4_NL and Q2_0 hand-built), both activation
//! layouts (`in_dim1 == 1`, the gate/up shape; `in_dim1 == topk`, the down
//! shape), batch > 1, and run-to-run identity.
//!
//! The reference is `x · dequant(W[e])ᵀ` per selected expert, never the other
//! arm (CONTRIBUTING §3.1b). `mutate_747_routed.sh` breaks the dispatch and the
//! Q2_0 core in place and expects these to go red.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{
    iq_quants::{BlockIQ4nl, BlockQ2_0},
    GgmlDType, QStorage, QTensor,
};
use lloom_tensor::{DType, Device, Result, Tensor, D};
use std::borrow::Cow;

fn lcg(seed: u32) -> impl FnMut() -> u8 {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    }
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: the block structs are #[repr(C)] with no padding (sizes pinned
    // by const asserts in iq_quants); the byte view covers exactly the slice.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A stacked `[n_experts, n, k]` weight of `dtype` on `dev`, plus its f32
/// dequant on the CPU (the reference).
fn stacked_weight(
    dtype: GgmlDType,
    (e, n, k): (usize, usize, usize),
    seed: u32,
    dev: &Device,
) -> Result<(QTensor, Tensor)> {
    let raw: Vec<u8> = match dtype {
        GgmlDType::Q2_0 => {
            let mut next = lcg(seed);
            let blocks: Vec<BlockQ2_0> = (0..e * n * k / 64)
                .map(|i| {
                    let mut qs = [0u8; 16];
                    for b in qs.iter_mut() {
                        *b = next();
                    }
                    BlockQ2_0 {
                        d: f16::from_f32(0.02 + 0.01 * ((i as f32) * 0.31).sin()),
                        qs,
                    }
                })
                .collect();
            as_bytes(&blocks).to_vec()
        }
        GgmlDType::IQ4NL => {
            let mut next = lcg(seed);
            let blocks: Vec<BlockIQ4nl> = (0..e * n * k / 32)
                .map(|i| {
                    let mut qs = [0u8; 16];
                    for b in qs.iter_mut() {
                        *b = next();
                    }
                    BlockIQ4nl {
                        d: f16::from_f32(0.004 + 0.002 * ((i as f32) * 0.29).cos()),
                        qs,
                    }
                })
                .collect();
            as_bytes(&blocks).to_vec()
        }
        GgmlDType::IQ2XXS
        | GgmlDType::IQ2XS
        | GgmlDType::IQ2S
        | GgmlDType::IQ3XXS
        | GgmlDType::IQ3S
        | GgmlDType::IQ4XS => {
            // Raw blocks: LCG bytes with a sane f16 scale at offset 0 of every
            // block (every byte pattern is a valid codebook block).
            let (bs, ts) = (dtype.block_size(), dtype.type_size());
            let n_blocks = e * n * k / bs;
            let mut next = lcg(seed);
            let mut raw = vec![0u8; n_blocks * ts];
            for b in raw.iter_mut() {
                *b = next();
            }
            for i in 0..n_blocks {
                let d = f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
                raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
            }
            raw
        }
        _ => {
            // Quantizable types: quantize a smooth f32 weight on the CPU.
            let w = (0..e * n * k)
                .map(|i| ((i as f32) * 0.0037).sin() * 0.5 + ((i as f32) * 0.011).cos() * 0.25)
                .collect::<Vec<_>>();
            let w = Tensor::from_vec(w, (e, n, k), &Device::Cpu)?;
            let q = QTensor::quantize(&w, dtype)?;
            q.data()?.into_owned()
        }
    };
    let on_dev = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?,
        (e, n, k),
    )?;
    let on_cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, dtype)?,
        (e, n, k),
    )?;
    let reference = on_cpu.dequantize(&Device::Cpu)?; // [e, n, k] f32
    Ok((on_dev, reference))
}

fn activations((batch, in_dim1, k): (usize, usize, usize), seed: f32) -> Result<Tensor> {
    let data = (0..batch * in_dim1 * k)
        .map(|i| ((i as f32) * 0.013 + seed).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, (batch, in_dim1, k), &Device::Cpu)
}

/// Dense reference: `out[b, t] = x[b, t or 0] · W[ids[b, t]]ᵀ`.
fn dense_reference(x: &Tensor, w_ref: &Tensor, ids: &[u32], topk: usize) -> Result<Tensor> {
    let (batch, in_dim1, _k) = x.dims3()?;
    let n = w_ref.dim(1)?;
    let mut planes = Vec::with_capacity(batch * topk);
    for b in 0..batch {
        for t in 0..topk {
            let e = ids[b * topk + t] as usize;
            let xi = if in_dim1 == 1 { 0 } else { t };
            let xrow = x.get(b)?.get(xi)?.unsqueeze(0)?; // [1, k]
            let we = w_ref.get(e)?; // [n, k]
            planes.push(xrow.matmul(&we.t()?)?); // [1, n]
        }
    }
    Tensor::cat(&planes, 0)?.reshape((batch, topk, n))
}

fn mean_rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let want = want.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    ((&want - &got)?.abs()? / (want.abs()? + 1e-4)?)?
        .mean_all()?
        .to_scalar::<f32>()
}

/// Both arms against the dense reference for one dtype and one activation
/// layout. `ids` repeats an expert across slots and across the batch so the
/// same weight plane is read from two `(b, t)` positions.
fn routed_matches_reference(dtype: GgmlDType, in_dim1_is_topk: bool, tol: f32) -> Result<()> {
    routed_matches_reference_n(dtype, in_dim1_is_topk, 40, tol)
}

/// As above with the row count `n` chosen by the caller. `n = 37` is not a
/// multiple of the rows a threadgroup covers, so the last threadgroup of every
/// `(b, t)` plane has rows past `n`: a core that stores them unguarded writes
/// into the NEXT plane's first rows — invisible on a plain `[1, n]` GEMV
/// (the over-write lands in buffer padding), corrupting on the routed one.
fn routed_matches_reference_n(
    dtype: GgmlDType,
    in_dim1_is_topk: bool,
    n: usize,
    tol: f32,
) -> Result<()> {
    routed_matches_reference_nk(dtype, in_dim1_is_topk, n, 256, tol)
}

fn routed_matches_reference_nk(
    dtype: GgmlDType,
    in_dim1_is_topk: bool,
    n: usize,
    k: usize,
    tol: f32,
) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let e = 6usize;
    let (batch, topk) = (2usize, 3usize);
    let in_dim1 = if in_dim1_is_topk { topk } else { 1 };
    let (w, w_ref) = stacked_weight(dtype, (e, n, k), 17, &dev)?;
    let x = activations((batch, in_dim1, k), 0.2)?;
    let ids: Vec<u32> = vec![5, 0, 5, 2, 4, 5];
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &Device::Cpu)?;
    let want = dense_reference(&x, &w_ref, &ids, topk)?;

    // Metal arm (kernel_mul_mv_id).
    let got = w.indexed_moe_forward(&x.to_device(&dev)?, &ids_t.to_device(&dev)?)?;
    assert_eq!(got.dims(), &[batch, topk, n]);
    assert_eq!(got.dtype(), DType::F32);
    let err = mean_rel_err(&got, &want)?;
    if err >= tol {
        // on failure, show where: per-plane max |diff| and a few values
        let g = got.to_device(&Device::Cpu)?;
        for b in 0..batch {
            for t in 0..topk {
                let gp = g.get(b)?.get(t)?.to_vec1::<f32>()?;
                let wp = want.get(b)?.get(t)?.to_vec1::<f32>()?;
                let maxd = gp
                    .iter()
                    .zip(&wp)
                    .map(|(a, c)| (a - c).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "plane b={b} t={t} expert={} maxdiff={maxd:.3e} got[..4]={:?} want[..4]={:?}",
                    ids[b * topk + t],
                    &gp[..4],
                    &wp[..4]
                );
            }
        }
    }
    assert!(
        err < tol,
        "{dtype:?} in_dim1={in_dim1}: Metal routed matmul diverges from the dense \
         reference: mean rel err {err} >= {tol}"
    );

    // CPU arm (dequant-per-row), the same contract.
    let w_cpu = QTensor::new(
        QStorage::from_data(w.data()?, &Device::Cpu, dtype)?,
        (e, n, k),
    )?;
    let got_cpu = w_cpu.indexed_moe_forward(&x, &ids_t)?;
    assert_eq!(got_cpu.dims(), &[batch, topk, n]);
    let err_cpu = mean_rel_err(&got_cpu, &want)?;
    assert!(
        err_cpu < tol,
        "{dtype:?} in_dim1={in_dim1}: CPU routed matmul diverges from the dense \
         reference: mean rel err {err_cpu} >= {tol}"
    );
    Ok(())
}

#[test]
fn routed_q4_0_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q4_0, false, 1e-4)
}
#[test]
fn routed_q4_0_down_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q4_0, true, 1e-4)
}
#[test]
fn routed_q8_0_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q8_0, false, 1e-4)
}
#[test]
fn routed_q4k_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q4K, false, 1e-4)
}
#[test]
fn routed_iq4_nl_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ4NL, false, 1e-4)
}
#[test]
fn routed_iq4_nl_down_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ4NL, true, 1e-4)
}
#[test]
fn routed_q2_0_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q2_0, false, 1e-4)
}
#[test]
fn routed_q2_0_down_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::Q2_0, true, 1e-4)
}

#[test]
fn routed_iq2_xxs_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ2XXS, false, 1e-4)
}
#[test]
fn routed_iq3_s_down_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ3S, true, 1e-4)
}
#[test]
fn routed_iq4_xs_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ4XS, false, 1e-4)
}

/// Rows not a multiple of the threadgroup's row tile (see
/// `routed_matches_reference_n`): the cores whose store is unguarded upstream
/// (the K-quants and the IQ2/IQ3/IQ4_XS family) must not spill into the next
/// plane.
#[test]
fn routed_q4k_rows_not_multiple_of_tile() -> Result<()> {
    routed_matches_reference_n(GgmlDType::Q4K, false, 37, 1e-4)
}
#[test]
fn routed_iq2_xxs_rows_not_multiple_of_tile() -> Result<()> {
    routed_matches_reference_n(GgmlDType::IQ2XXS, true, 37, 1e-4)
}
#[test]
fn routed_iq3_s_rows_not_multiple_of_tile() -> Result<()> {
    routed_matches_reference_n(GgmlDType::IQ3S, false, 37, 1e-4)
}
#[test]
fn routed_q6k_rows_not_multiple_of_tile() -> Result<()> {
    routed_matches_reference_n(GgmlDType::Q6K, false, 37, 1e-4)
}
#[test]
fn routed_iq4_nl_rows_not_multiple_of_tile() -> Result<()> {
    routed_matches_reference_n(GgmlDType::IQ4NL, false, 37, 1e-4)
}

#[test]
fn routed_iq2_xxs_two_superblocks() -> Result<()> {
    routed_matches_reference_nk(GgmlDType::IQ2XXS, false, 40, 512, 1e-4)
}
#[test]
fn routed_iq2_xs_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ2XS, false, 1e-4)
}
#[test]
fn routed_iq2_s_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ2S, false, 1e-4)
}
#[test]
fn routed_iq3_xxs_gate_shape() -> Result<()> {
    routed_matches_reference(GgmlDType::IQ3XXS, false, 1e-4)
}

/// The `(f16, f32, f16)` routed arm (issue #748, #750 item B) against the
/// dense f32 reference: f16 activations in, f16 planes out, within the f16
/// store's rounding; at a row count that is not a multiple of the tile, so a
/// spill into the next plane would show; and against the f32 routed arm on
/// the same inputs, within the f16 store's rounding of the f32 result — the
/// two arms are the same core at two store widths, not two computations.
fn routed_f16_matches_reference(dtype: GgmlDType, in_dim1_is_topk: bool, n: usize) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let e = 6usize;
    let (batch, topk, k) = (2usize, 3usize, 256usize);
    let in_dim1 = if in_dim1_is_topk { topk } else { 1 };
    let (w, w_ref) = stacked_weight(dtype, (e, n, k), 23, &dev)?;
    let x = activations((batch, in_dim1, k), 0.4)?;
    let ids: Vec<u32> = vec![5, 0, 5, 2, 4, 5];
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &Device::Cpu)?.to_device(&dev)?;
    // The reference sees the f16-rounded activations the kernel reads.
    let x16 = x.to_dtype(DType::F16)?;
    let want = dense_reference(&x16.to_dtype(DType::F32)?, &w_ref, &ids, topk)?;

    let got16 = w.indexed_moe_forward(&x16.to_device(&dev)?, &ids_t)?;
    assert_eq!(got16.dims(), &[batch, topk, n]);
    assert_eq!(got16.dtype(), DType::F16, "the f16 arm must store f16");
    let err = mean_rel_err(&got16, &want)?;
    assert!(
        err < 4e-3,
        "{dtype:?} in_dim1={in_dim1} n={n}: f16 routed arm diverges from the dense reference: mean rel err {err}"
    );

    // Against the f32 arm on the same (f16-rounded) activations: the f32 arm's
    // planes rounded to f16 must equal the f16 arm's planes to the ulp scale.
    let got32 = w.indexed_moe_forward(&x16.to_dtype(DType::F32)?.to_device(&dev)?, &ids_t)?;
    assert_eq!(got32.dtype(), DType::F32);
    let a = got16
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = got32
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    // The f32 arm's planes rounded to f16 are the f16 arm's planes, to the
    // last place: the two arms are the same core, the f16 store is the only
    // rounding. Measured in units of the f16 ulp at the value's own exponent.
    let mut worst = 0f32;
    let mut worst_at = (0usize, 0f32, 0f32);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let y16 = half::f16::from_f32(*y).to_f32();
        let exp = if *y == 0.0 {
            -24
        } else {
            y.abs().log2().floor() as i32
        };
        let ulp = 2f32.powi((exp - 10).max(-24));
        let d = (x - y16).abs() / ulp;
        if d > worst {
            worst = d;
            worst_at = (i, *x, *y);
        }
    }
    assert!(
        worst <= 1.0,
        "{dtype:?} in_dim1={in_dim1} n={n}: f16 arm vs f32 arm rounded to f16 differ by {worst} f16 ulps at {worst_at:?}"
    );
    Ok(())
}

#[test]
fn routed_f16_q4_0_gate_shape() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::Q4_0, false, 40)
}
#[test]
fn routed_f16_q4k_down_shape() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::Q4K, true, 40)
}
#[test]
fn routed_f16_iq2_xxs_rows_not_multiple_of_tile() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::IQ2XXS, false, 37)
}
#[test]
fn routed_f16_iq4_nl_down_shape() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::IQ4NL, true, 40)
}
#[test]
fn routed_f16_iq3_s_rows_not_multiple_of_tile() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::IQ3S, true, 37)
}
#[test]
fn routed_f16_q2_0_gate_shape() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::Q2_0, false, 40)
}
#[test]
fn routed_f16_q2_0_down_shape() -> Result<()> {
    routed_f16_matches_reference(GgmlDType::Q2_0, true, 37)
}

/// A dense/Q8 weight has no f16 routed arm: an f16 activation is refused,
/// never read under the wrong dtype.
#[test]
fn routed_f16_refuses_dense_weight() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (e, n, k) = (2usize, 8usize, 64usize);
    let w = QTensor::quantize(&Tensor::zeros((e, n, k), DType::F32, &dev)?, GgmlDType::F32)?;
    let x = Tensor::zeros((1, 1, k), DType::F16, &dev)?;
    let ids = Tensor::from_vec(vec![0u32, 1], (1, 2), &dev)?;
    assert!(w.indexed_moe_forward(&x, &ids).is_err());
    Ok(())
}

/// The Qwen decode shape at reduced width: 512 experts are too many for a
/// fixture, but the expert-stride arithmetic (`nb02`) is what a wrong offset
/// corrupts, so use enough experts that a mis-stride lands on a different,
/// non-zero expert: e = 64, top-10, and a hidden of 640 (the checkpoint's
/// moe_intermediate_size) at k = 512.
#[test]
fn routed_q2_0_top10_over_64_experts() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (e, n, k) = (64usize, 640usize, 512usize);
    let (batch, topk) = (1usize, 10usize);
    let (w, w_ref) = stacked_weight(GgmlDType::Q2_0, (e, n, k), 23, &dev)?;
    let x = activations((batch, 1, k), 0.7)?;
    let ids: Vec<u32> = vec![63, 7, 31, 0, 8, 47, 15, 1, 62, 32];
    let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &Device::Cpu)?;
    let want = dense_reference(&x, &w_ref, &ids, topk)?;
    let got = w.indexed_moe_forward(&x.to_device(&dev)?, &ids_t.to_device(&dev)?)?;
    let err = mean_rel_err(&got, &want)?;
    assert!(err < 1e-4, "mean rel err {err}");
    // and the planes are distinct (a stride of zero would repeat one expert)
    let p0 = got.get(0)?.get(0)?;
    let p1 = got.get(0)?.get(1)?;
    let diff = (p0 - p1)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        diff > 1e-3,
        "expert planes are identical: stride is not reaching the experts"
    );
    Ok(())
}

/// Run-to-run identity of the routed dispatch (§2.3.6).
#[test]
fn routed_is_bit_identical_across_runs() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (e, n, k) = (8usize, 128usize, 512usize);
    let (batch, topk) = (2usize, 4usize);
    let (w, _) = stacked_weight(GgmlDType::IQ4NL, (e, n, k), 29, &dev)?;
    let x = activations((batch, 1, k), 0.4)?.to_device(&dev)?;
    let ids = Tensor::from_vec(vec![1u32, 7, 3, 3, 0, 6, 5, 2], (batch, topk), &dev)?;
    let bits = |t: &Tensor| -> Result<Vec<u32>> {
        Ok(t.to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let first = bits(&w.indexed_moe_forward(&x, &ids)?)?;
    assert!(first.iter().any(|&b| b != 0), "degenerate all-zero output");
    for run in 1..5 {
        let again = bits(&w.indexed_moe_forward(&x, &ids)?)?;
        assert_eq!(first, again, "run {run} differs from run 0");
    }
    let _ = D::Minus1;
    Ok(())
}

/// Diagnostic: per-plane error map for the routed IQ2_XXS one-superblock case.
#[test]
#[ignore]
fn diag_routed_iq2_xxs_planes() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (e, n, k) = (6usize, 40usize, 256usize);
    let (batch, topk) = (2usize, 3usize);
    let (w, w_ref) = stacked_weight(GgmlDType::IQ2XXS, (e, n, k), 17, &dev)?;
    let x = activations((batch, 1, k), 0.2)?;
    for ids in [
        vec![5u32, 0, 5, 2, 4, 5],
        vec![0u32, 0, 0, 0, 0, 0],
        vec![1u32, 1, 1, 1, 1, 1],
        vec![0u32, 1, 2, 3, 4, 5],
    ] {
        let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &Device::Cpu)?;
        let want = dense_reference(&x, &w_ref, &ids, topk)?;
        let got = w.indexed_moe_forward(&x.to_device(&dev)?, &ids_t.to_device(&dev)?)?;
        let g = got.to_device(&Device::Cpu)?;
        for b in 0..batch {
            for t in 0..topk {
                let gp = g.get(b)?.get(t)?.to_vec1::<f32>()?;
                let wp = want.get(b)?.get(t)?.to_vec1::<f32>()?;
                let maxd = gp
                    .iter()
                    .zip(&wp)
                    .map(|(a, c)| (a - c).abs())
                    .fold(0f32, f32::max);
                let first_bad = gp.iter().zip(&wp).position(|(a, c)| (a - c).abs() > 1e-3);
                eprintln!("ids={ids:?} plane b={b} t={t} expert={} maxdiff={maxd:.3e} first_bad_row={first_bad:?} got[0..3]={:?} want[0..3]={:?}", ids[b*topk+t], &gp[..3], &wp[..3]);
            }
        }
    }
    Ok(())
}
