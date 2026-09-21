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
    let dev = Device::new_metal(0)?;
    let (e, n, k) = (6usize, 40usize, 256usize);
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
