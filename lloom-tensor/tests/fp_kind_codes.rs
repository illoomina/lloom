//! Every code of the float-code block kinds through the Metal decoders, bit
//! for bit against the CPU `to_float` — the cases the parity matrix's
//! quantized fixtures do not reach: all 256 E4M3FN codes (subnormals, ±448,
//! the two NaNs, -0), all 16 E2M1 codes, and E8M0 scales across the byte's
//! range.
//!
//! Two decoders per kind: the element-wise one (`dequantize_*`, behind
//! `get_rows`, `dequant_rows_f16` and every GEMM) and the matmul-vec core's
//! own, read back through one-hot activations, so each output is one
//! decoded weight times one (`0 · w` adds nothing, and the f32 sum of one
//! product is the product).
#![cfg(feature = "metal")]

use half::{bf16, f16};
use lloom_tensor::quantized::fp_quants::{e4m3_to_f32, e8m0_to_f32_half};
use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::borrow::Cow;

fn metal() -> Device {
    Device::new_metal(0).expect("a Metal device")
}

fn both(
    raw: &[u8],
    dtype: GgmlDType,
    rows: usize,
    k: usize,
    gpu: &Device,
) -> Result<(QTensor, Vec<f32>)> {
    let q = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), gpu, dtype)?,
        (rows, k),
    )?;
    let cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(raw), &Device::Cpu, dtype)?,
        (rows, k),
    )?;
    let w = cpu
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((q, w))
}

/// `get_rows` of every row equals the CPU decode bit for bit; a NaN code
/// is NaN on both sides (its payload is the GPU's to choose).
fn get_rows_matches(
    q: &QTensor,
    w: &[f32],
    rows: usize,
    k: usize,
    gpu: &Device,
    what: &str,
) -> Result<()> {
    let ids = Tensor::from_vec((0..rows as u32).collect::<Vec<_>>(), rows, gpu)?;
    let got = q
        .embedding(&ids)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut bad = Vec::new();
    for i in 0..rows * k {
        let (g, r) = (got[i], w[i]);
        let same = if r.is_nan() {
            g.is_nan()
        } else {
            g.to_bits() == r.to_bits()
        };
        if !same {
            bad.push(format!(
                "[{i}] got {g:e} ({:#010x}) want {r:e} ({:#010x})",
                g.to_bits(),
                r.to_bits()
            ));
        }
    }
    println!(
        "{what}: get_rows {} of {} elements differ",
        bad.len(),
        rows * k
    );
    assert!(
        bad.is_empty(),
        "{what}: get_rows differs:\n{}",
        bad[..bad.len().min(16)].join("\n")
    );
    Ok(())
}

/// `dequant_rows_f16` equals the CPU decode rounded to half, bit for bit
/// (NaN as NaN).
fn dequant_rows_f16_matches(
    q: &QTensor,
    w: &[f32],
    rows: usize,
    k: usize,
    dtype: GgmlDType,
    gpu: &Device,
    what: &str,
) -> Result<()> {
    let out = Tensor::zeros(rows * k, lloom_tensor::DType::F16, gpu)?;
    {
        let md = gpu.as_metal_device()?;
        let (s, _) = out.storage_and_layout();
        let dst = match &*s {
            lloom_tensor::Storage::Metal(m) => m.buffer(),
            _ => unreachable!("a Metal tensor"),
        };
        let enc = md.command_encoder()?;
        lloom_tensor::kernels::call_quantized_dequant_rows_f16(
            md.device(),
            &enc,
            md.kernels(),
            dtype.into(),
            (rows, k),
            q.metal_buffer()?,
            dst,
        )
        .map_err(lloom_tensor::Error::wrap)?;
    }
    let got = out.to_device(&Device::Cpu)?.to_vec1::<f16>()?;
    let mut differ = 0usize;
    for i in 0..rows * k {
        let r = f16::from_f32(w[i]);
        let same = if r.is_nan() {
            got[i].is_nan()
        } else {
            got[i].to_bits() == r.to_bits()
        };
        if !same {
            differ += 1;
            if differ <= 8 {
                println!(
                    "{what}: dequant_rows_f16 [{i}] got {} want {r} (f32 {:e})",
                    got[i], w[i]
                );
            }
        }
    }
    println!(
        "{what}: dequant_rows_f16 {differ} of {} elements differ",
        rows * k
    );
    assert_eq!(differ, 0, "{what}: dequant_rows_f16 differs");
    Ok(())
}

/// The matmul-vec core's decode: one-hot activations, eight a launch (the
/// matmul-vec's rows), so output `[r, j]` is weight `[j, c]` for the hot
/// column `c`. The weight must be finite (a NaN anywhere poisons a row).
fn matvec_matches(
    q: &QTensor,
    w: &[f32],
    rows: usize,
    k: usize,
    gpu: &Device,
    what: &str,
) -> Result<()> {
    let mut differ = 0usize;
    for c0 in (0..k).step_by(8) {
        let mut x = vec![0f32; 8 * k];
        for r in 0..8 {
            x[r * k + c0 + r] = 1.0;
        }
        let xt = Tensor::from_vec(x, (8, k), gpu)?;
        let got = xt
            .apply_op1_no_bwd(q)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for r in 0..8 {
            for j in 0..rows {
                let (g, want) = (got[r * rows + j], w[j * k + c0 + r]);
                if g.to_bits() != want.to_bits() && !(g == 0.0 && want == 0.0) {
                    differ += 1;
                    if differ <= 8 {
                        println!(
                            "{what}: matvec col {} row {j}: got {g:e} want {want:e}",
                            c0 + r
                        );
                    }
                }
            }
        }
    }
    println!("{what}: matvec {differ} of {} weights differ", rows * k);
    assert_eq!(differ, 0, "{what}: the matmul-vec core decodes differently");
    Ok(())
}

/// Eight blocks of 32 codes hold all 256 E4M3 codes, under a scale of one
/// and under an arbitrary bf16 scale.
#[test]
fn every_e4m3_code_decodes_as_the_cpu_does() -> Result<()> {
    let gpu = metal();
    let (rows, k) = (4usize, 64usize);
    for d in [1.0f32, 0.012_3, 3.0e-5] {
        let d = bf16::from_f32(d);
        let mut raw = Vec::new();
        for b in 0..8u32 {
            raw.extend_from_slice(&d.to_le_bytes());
            raw.extend((0..32u32).map(|j| (b * 32 + j) as u8));
        }
        let what = format!("fp8 d={}", d.to_f32());
        let (q, w) = both(&raw, GgmlDType::FP8E4M3, rows, k, &gpu)?;
        assert_eq!(
            w[0x38].to_bits(),
            (e4m3_to_f32(0x38) * d.to_f32()).to_bits()
        );
        get_rows_matches(&q, &w, rows, k, &gpu, &what)?;
        dequant_rows_f16_matches(&q, &w, rows, k, GgmlDType::FP8E4M3, &gpu, &what)?;
        // The NaN codes (0x7F, 0xFF) zeroed for the reduction.
        for blk in raw.chunks_mut(34) {
            for c in &mut blk[2..] {
                if *c & 0x7F == 0x7F {
                    *c = 0;
                }
            }
        }
        let (q, w) = both(&raw, GgmlDType::FP8E4M3, rows, k, &gpu)?;
        matvec_matches(&q, &w, rows, k, &gpu, &what)?;
    }
    Ok(())
}

/// Every E2M1 code under E8M0 scales across the byte, the two subnormal
/// scales apart (`subnormal_e8m0_scales_flush_to_zero_on_the_gpu`).
#[test]
fn every_e2m1_code_and_e8m0_scale_decodes_as_the_cpu_does() -> Result<()> {
    let gpu = metal();
    let es: Vec<u8> = (2..=254u8).step_by(3).chain([127, 128, 253, 254]).collect();
    let blocks = es.len();
    let mut raw = Vec::with_capacity(blocks * 17);
    for (i, &e) in es.iter().enumerate() {
        raw.push(e);
        // Each block holds all 16 codes twice, rotated by the block index.
        raw.extend((0..16u8).map(|j| ((j + i as u8) & 0x0F) | (((j + 3 + i as u8) & 0x0F) << 4)));
    }
    let k = 32 * blocks;
    let (q, w) = both(&raw, GgmlDType::MXFP4, 1, k, &gpu)?;
    assert_eq!(w[0], 0.0 * e8m0_to_f32_half(es[0]));
    get_rows_matches(&q, &w, 1, k, &gpu, "mxfp4")?;
    // Half holds the finite range of 2^(e-128)·12 only between e ≈ 103 and
    // 143; past it the rounding to half is the reference's own (inf, 0).
    dequant_rows_f16_matches(&q, &w, 1, k, GgmlDType::MXFP4, &gpu, "mxfp4")?;
    // The reduction over the scales an f32 sum holds without overflow.
    let es: Vec<u8> = (60..=200u8).step_by(5).collect();
    let mut raw = Vec::new();
    for (i, &e) in es.iter().enumerate() {
        raw.push(e);
        raw.extend((0..16u8).map(|j| ((j + i as u8) & 0x0F) | (((j + 7) & 0x0F) << 4)));
    }
    let k = 32 * es.len();
    let raw: Vec<u8> = raw.repeat(2);
    let (q, w) = both(&raw, GgmlDType::MXFP4, 2, k, &gpu)?;
    matvec_matches(&q, &w, 2, k, &gpu, "mxfp4")?;
    Ok(())
}

/// An E8M0 scale below f32's normal range (`e8m0_to_f32_half` of 0 and 1,
/// 2^-128 and 2^-127) decodes to zero on the GPU, which flushes f32
/// subnormals, where the CPU keeps them; 2 is the smallest scale the two
/// decode alike. A weight there is under 2^-125 in magnitude.
#[test]
fn subnormal_e8m0_scales_flush_to_zero_on_the_gpu() -> Result<()> {
    let gpu = metal();
    for e in [0u8, 1, 2] {
        let mut raw = vec![e];
        raw.extend((0..16u8).map(|j| j | (j << 4)));
        let (q, w) = both(&raw, GgmlDType::MXFP4, 1, 32, &gpu)?;
        let ids = Tensor::from_vec(vec![0u32], 1, &gpu)?;
        let got = q
            .embedding(&ids)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        println!("e={e}: cpu {:?}", &w[..8]);
        println!("e={e}: gpu {:?}", &got[..8]);
        for (g, r) in got.iter().zip(&w) {
            if e < 2 {
                assert!(
                    r.is_subnormal() || *r == 0.0,
                    "e={e}: {r:e} is not subnormal"
                );
                assert_eq!(*g, 0.0, "e={e}: the GPU kept {g:e}");
            } else {
                assert_eq!(g.to_bits(), r.to_bits(), "e={e}");
            }
        }
    }
    Ok(())
}
