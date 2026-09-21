//! The IQ-family Metal kernels (issue #747): parity of the plain matmul-vec at
//! the f32 and f16 triples, the prefill GEMM and the row gather against the
//! CPU dequant reference (`iq_quants`, ported byte-exact from ggml), for every
//! IQ type the Qwen checkpoint stores (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
//! IQ4_NL/IQ4_XS) plus Q2_0 through the same generic fixture.
//!
//! The types are read-only (no `from_float`), so the fixture is raw blocks:
//! LCG bytes with a sane f16 scale written at offset 0 of every block. Every
//! byte pattern is a valid block for these codebook types (indices are masked
//! to their grid), so the reference is exactly what the kernel must reproduce.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{DType, Device, Module, Result, Tensor};
use std::borrow::Cow;

const TYPES: [GgmlDType; 8] = [
    GgmlDType::IQ2XXS,
    GgmlDType::IQ2XS,
    GgmlDType::IQ2S,
    GgmlDType::IQ3XXS,
    GgmlDType::IQ3S,
    GgmlDType::IQ4NL,
    GgmlDType::IQ4XS,
    GgmlDType::Q2_0,
];

fn raw_blocks(dtype: GgmlDType, n: usize, k: usize, seed: u32) -> Vec<u8> {
    let (bs, ts) = (dtype.block_size(), dtype.type_size());
    assert_eq!(
        k % bs,
        0,
        "{dtype:?}: k {k} not a multiple of the block {bs}"
    );
    let n_blocks = n * k / bs;
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(777);
    let mut raw = vec![0u8; n_blocks * ts];
    for b in raw.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    for i in 0..n_blocks {
        let d = f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
        raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
    }
    raw
}

/// The `[n, k]` weight on `dev` plus its CPU f32 dequant (the reference).
fn weight(
    dtype: GgmlDType,
    n: usize,
    k: usize,
    seed: u32,
    dev: &Device,
) -> Result<(QTensor, Tensor)> {
    let raw = raw_blocks(dtype, n, k, seed);
    let on_dev = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?,
        (n, k),
    )?;
    let on_cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, dtype)?,
        (n, k),
    )?;
    let reference = on_cpu.dequantize(&Device::Cpu)?;
    Ok((on_dev, reference))
}

fn activations(m: usize, k: usize) -> Result<Tensor> {
    let data = (0..m * k)
        .map(|i| ((i as f32) * 0.013).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, (m, k), &Device::Cpu)
}

fn mean_rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let want = want.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    ((&want - &got)?.abs()? / (want.abs()? + 1e-4)?)?
        .mean_all()?
        .to_scalar::<f32>()
}

/// The fixture is not degenerate: the reference is finite and carries both
/// signs and a spread of magnitudes for every type.
#[test]
fn iq_fixtures_dequantize_to_nondegenerate_weights() -> Result<()> {
    for dtype in TYPES {
        let (_, w) = weight(dtype, 8, 512, 1, &Device::Cpu)?;
        let v = w.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "{dtype:?}: non-finite");
        assert!(
            v.iter().any(|&x| x > 0.0) && v.iter().any(|&x| x < 0.0),
            "{dtype:?}: one-signed"
        );
        let distinct = v
            .iter()
            .map(|x| x.to_bits())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(distinct > 8, "{dtype:?}: only {distinct} distinct values");
    }
    Ok(())
}

/// Decode GEMV (`m == 1`) at `(f32, f32, f32)`; `n = 37` exercises the row
/// guard of every core's threadgroup tiling.
#[test]
fn iq_gemv_f32_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (37usize, 512usize);
    for dtype in TYPES {
        let (w, w_ref) = weight(dtype, n, k, 7, &dev)?;
        let x = activations(1, k)?;
        let want = x.matmul(&w_ref.t()?)?;
        let got = QMatMul::from_qtensor(w)?.forward(&x.to_device(&dev)?)?;
        assert_eq!(got.dtype(), DType::F32);
        assert_eq!(got.dims(), &[1, n]);
        let err = mean_rel_err(&got, &want)?;
        assert!(err < 1e-4, "{dtype:?} f32 GEMV: mean rel err {err}");
    }
    Ok(())
}

/// The `(f16, f32, f16)` arm against the same full-precision reference.
#[test]
fn iq_gemv_f16_arm_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (48usize, 512usize);
    for dtype in TYPES {
        let (w, w_ref) = weight(dtype, n, k, 11, &dev)?;
        let x = activations(1, k)?;
        let want = x.matmul(&w_ref.t()?)?;
        let got = QMatMul::from_qtensor(w)?.forward(&x.to_device(&dev)?.to_dtype(DType::F16)?)?;
        assert_eq!(
            got.dtype(),
            DType::F16,
            "{dtype:?}: f16 activation -> f16 store"
        );
        let err = mean_rel_err(&got, &want)?;
        assert!(err < 5e-3, "{dtype:?} f16 GEMV arm: mean rel err {err}");
    }
    Ok(())
}

/// Prefill shape (`m == 3`, the `mul_mm` GEMM with each type's `dequantize_*`).
/// The tiled GEMM is `simdgroup_half8x8` over half-dequantized blocks (the
/// upstream template every quant type shares), so its error is f16-product
/// scale (~1e-3 on this fixture), not the f32 GEMV's.
#[test]
fn iq_gemm_prefill_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (m, n, k) = (3usize, 64usize, 512usize);
    for dtype in TYPES {
        let (w, w_ref) = weight(dtype, n, k, 3, &dev)?;
        let x = activations(m, k)?;
        let want = x.matmul(&w_ref.t()?)?;
        let got = QMatMul::from_qtensor(w)?.forward(&x.to_device(&dev)?)?;
        assert_eq!(got.dims(), &[m, n]);
        let err = mean_rel_err(&got, &want)?;
        assert!(err < 5e-3, "{dtype:?} prefill GEMM: mean rel err {err}");
    }
    Ok(())
}

/// Row gather (`get_rows_*`) against `dequantize + index_select` — the path the
/// IQ3_S embedding table takes.
#[test]
fn iq_embedding_matches_cpu_dequant_reference() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (rows, hidden) = (8usize, 512usize);
    for dtype in TYPES {
        let (w, w_ref) = weight(dtype, rows, hidden, 5, &dev)?;
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
                "{dtype:?} row gather mismatch at {i}: {g} != {w}"
            );
        }
    }
    Ok(())
}

/// The one-superblock case (`k == 256`, `nb == 1`): the K-quant-shaped cores
/// walk 32-blocks across simdgroup lanes and advance per-row pointers by `nb`
/// blocks, so `nb == 1` is the tightest layout.
#[test]
fn iq_gemv_f32_matches_reference_at_one_superblock() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (n, k) = (40usize, 256usize);
    for dtype in TYPES {
        let (w, w_ref) = weight(dtype, n, k, 13, &dev)?;
        let x = activations(1, k)?;
        let want = x.matmul(&w_ref.t()?)?;
        let got = QMatMul::from_qtensor(w)?.forward(&x.to_device(&dev)?)?;
        let err = mean_rel_err(&got, &want)?;
        assert!(
            err < 1e-4,
            "{dtype:?} f32 GEMV at k=256: mean rel err {err}"
        );
    }
    Ok(())
}
