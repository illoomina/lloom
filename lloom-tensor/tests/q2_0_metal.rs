//! Q2_0 Metal kernels (issue #747, #750 item C): parity against the CPU
//! dequant reference, the f16 triple arm, the prefill GEMM, the row gather,
//! and run-to-run determinism.
//!
//! Q2_0 is read-only (no `from_float`), so the fixture is hand-built blocks —
//! the same bytes are dequantized on the CPU (`BlockQ2_0::to_float`, the
//! byte-exact port of the fork's `dequantize_row_q2_0`) for the reference.
//! Every assertion is against that absolute reference, never arm-vs-arm
//! (CONTRIBUTING §3.1b). The mutation harness `mutate_747_q2_0.sh` breaks the
//! kernels in place and expects these to go red.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{iq_quants::BlockQ2_0, GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{DType, Device, Module, Result, Tensor};
use std::borrow::Cow;

const QK2_0: usize = 64;

/// Deterministic Q2_0 blocks for an `[n, k]` weight: scales `d` spread over
/// (0.005, 0.045), quant codes from an LCG so every code {0,1,2,3} appears.
fn q2_0_blocks(n: usize, k: usize, seed: u32) -> Vec<BlockQ2_0> {
    assert_eq!(k % QK2_0, 0);
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    };
    (0..n * k / QK2_0)
        .map(|i| {
            let d = 0.025 + 0.02 * ((i as f32) * 0.37).sin();
            let mut qs = [0u8; QK2_0 / 4];
            for b in qs.iter_mut() {
                *b = next();
            }
            BlockQ2_0 {
                d: f16::from_f32(d),
                qs,
            }
        })
        .collect()
}

fn blocks_as_bytes(blocks: &[BlockQ2_0]) -> &[u8] {
    // SAFETY: BlockQ2_0 is #[repr(C)] with no padding (size 18 = 2 + 16, pinned
    // by a const assert in iq_quants); the byte view covers exactly the vector.
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    }
}

/// The `[n, k]` Q2_0 weight on `dev`, plus its f32 dequant (`[n, k]`, CPU) as
/// the reference.
fn q2_0_weight(n: usize, k: usize, seed: u32, dev: &Device) -> Result<(QTensor, Tensor)> {
    let blocks = q2_0_blocks(n, k, seed);
    let bytes = blocks_as_bytes(&blocks);
    let on_dev = QTensor::new(
        QStorage::from_data(Cow::Borrowed(bytes), dev, GgmlDType::Q2_0)?,
        (n, k),
    )?;
    let on_cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, GgmlDType::Q2_0)?,
        (n, k),
    )?;
    let reference = on_cpu.dequantize(&Device::Cpu)?;
    Ok((on_dev, reference))
}

fn activations(m: usize, k: usize, dev: &Device) -> Result<Tensor> {
    let data = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, (m, k), dev)
}

fn mean_rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let want = want.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    ((&want - &got)?.abs()? / (want.abs()? + 1e-4)?)?
        .mean_all()?
        .to_scalar::<f32>()
}

/// The hand-built fixture is not degenerate: the reference carries every code
/// (values at -d, 0, +d, +2d) and is not all-zero.
#[test]
fn q2_0_fixture_dequantizes_to_all_four_codes() -> Result<()> {
    let (_, reference) = q2_0_weight(16, 256, 1, &Device::Cpu)?;
    let v = reference.flatten_all()?.to_vec1::<f32>()?;
    let d0 = f16::from_f32(0.025 + 0.02 * 0f32.sin()).to_f32();
    let first_block = &v[..QK2_0];
    for code in [-1.0f32, 0.0, 1.0, 2.0] {
        assert!(
            first_block.iter().any(|&x| (x - code * d0).abs() < 1e-7),
            "code {code} missing from the first block: {first_block:?}"
        );
    }
    Ok(())
}

/// Decode GEMV (`m == 1`, the `fwd_mv` path) at the `(f32, f32, f32)` triple
/// against the CPU-dequant reference. `n = 37` is deliberately not a multiple
/// of the 16 rows a threadgroup covers, so the row guard is exercised.
#[test]
fn q2_0_gemv_f32_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (37usize, 640usize);
    let (w, w_ref) = q2_0_weight(n, k, 7, &dev)?;
    let x_cpu = activations(1, k, &Device::Cpu)?;
    let want = x_cpu.matmul(&w_ref.t()?)?; // [1, n]

    let mm = QMatMul::from_qtensor(w)?;
    let got = mm.forward(&x_cpu.to_device(&dev)?)?;
    assert_eq!(got.dtype(), DType::F32);
    assert_eq!(got.dims(), &[1, n]);
    let err = mean_rel_err(&got, &want)?;
    assert!(
        err < 1e-4,
        "Q2_0 f32 GEMV diverges from the dequant reference: mean rel err {err}"
    );
    Ok(())
}

/// The `(f16, f32, f16)` arm — f16 activations, f32 accumulate, f16 store —
/// against the SAME full-precision reference (not the f32 arm: a symmetric
/// core corruption would leave two arms agreeing, §3.1b).
#[test]
fn q2_0_gemv_f16_arm_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (48usize, 512usize);
    let (w, w_ref) = q2_0_weight(n, k, 11, &dev)?;
    let x_cpu = activations(1, k, &Device::Cpu)?;
    let want = x_cpu.matmul(&w_ref.t()?)?;

    let mm = QMatMul::from_qtensor(w)?;
    let got = mm.forward(&x_cpu.to_device(&dev)?.to_dtype(DType::F16)?)?;
    assert_eq!(got.dtype(), DType::F16, "f16 activation -> f16 store");
    let err = mean_rel_err(&got, &want)?;
    assert!(
        err < 5e-3,
        "Q2_0 f16 GEMV arm diverges from the dequant reference: mean rel err {err}"
    );
    Ok(())
}

/// Prefill shape (`m == 3`, the `mul_mm` GEMM path with `dequantize_q2_0`).
#[test]
fn q2_0_gemm_prefill_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (m, n, k) = (3usize, 64usize, 256usize);
    let (w, w_ref) = q2_0_weight(n, k, 3, &dev)?;
    let x_cpu = activations(m, k, &Device::Cpu)?;
    let want = x_cpu.matmul(&w_ref.t()?)?;

    let mm = QMatMul::from_qtensor(w)?;
    let got = mm.forward(&x_cpu.to_device(&dev)?)?;
    assert_eq!(got.dims(), &[m, n]);
    let err = mean_rel_err(&got, &want)?;
    assert!(
        err < 1e-3,
        "Q2_0 prefill GEMM diverges from the dequant reference: mean rel err {err}"
    );
    Ok(())
}

/// Row gather (`get_rows_q2_0`) against `dequantize + index_select`.
#[test]
fn q2_0_embedding_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (rows, hidden) = (8usize, 256usize);
    let (w, w_ref) = q2_0_weight(rows, hidden, 5, &dev)?;
    let ids = Tensor::from_vec(vec![3u32, 1, 3, 7], (2, 2), &dev)?;
    let got = w.embedding(&ids)?;
    let want = w_ref
        .index_select(&ids.to_device(&Device::Cpu)?.flatten_all()?, 0)?
        .reshape((2, 2, hidden))?;
    let got = got
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let want = want.flatten_all()?.to_vec1::<f32>()?;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "row gather mismatch at {i}: {g} != {w}"
        );
    }
    Ok(())
}

/// Run-to-run identity of the decode GEMV (§2.3.6): five dispatches, one bit
/// pattern.
#[test]
fn q2_0_gemv_is_bit_identical_across_runs() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (128usize, 1024usize);
    let (w, _) = q2_0_weight(n, k, 9, &dev)?;
    let x = activations(1, k, &dev)?;
    let mm = QMatMul::from_qtensor(w)?;
    let bits = |t: &Tensor| -> Result<Vec<u32>> {
        Ok(t.to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let first = bits(&mm.forward(&x)?)?;
    assert!(first.iter().any(|&b| b != 0), "degenerate all-zero output");
    for run in 1..5 {
        let again = bits(&mm.forward(&x)?)?;
        assert_eq!(first, again, "run {run} differs from run 0");
    }
    Ok(())
}
