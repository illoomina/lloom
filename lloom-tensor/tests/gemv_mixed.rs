//! The mixed-precision GEMV (`gemv_mixed_f16w_f32_*`): f16 weights, f32 in
//! and out, one launch — against the CPU reference (the weights widened,
//! an f32 dot per row, the epilogue at f32) on the Qwen4-exp dense shapes,
//! every tile `derive_gemv_tile` selects for them, both epilogues, batches
//! of 1 and 3 rows. Metal only.
#![cfg(feature = "metal")]

use lloom_tensor::quantized::gemv_mixed::MixedEpilogue;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor};

fn weight(n: usize, k: usize, seed: f32, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.05)
        .collect();
    Tensor::from_vec(v, (n, k), dev)?.to_dtype(DType::F16)
}

fn input(b: usize, k: usize, seed: f32, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..b * k)
        .map(|i| ((i as f32 * 0.021 + seed).cos()) * 0.7)
        .collect();
    Tensor::from_vec(v, (b, 1, k), dev)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let x: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let y: Vec<f32> = b.flatten_all()?.to_vec1()?;
    assert_eq!(x.len(), y.len());
    let scale = y.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = x
        .iter()
        .zip(&y)
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max);
    Ok((diff, scale))
}

#[test]
fn mixed_gemv_matches_the_f32_reference_on_every_tile() -> Result<()> {
    let Ok(gpu) = Device::new_metal(0) else {
        return Ok(());
    };
    let cpu = Device::Cpu;
    // (n_out, k): the HC down (bm1_bn8), up (bm8), inject (tm 4 at n 4),
    // DeltaNet α/β (bm1_bn8), the router (bm4), a tiny-K shape (sm8_sn4),
    // an n < 4 shape (tm1).
    let shapes = [
        (320usize, 10240usize),
        (10240, 320),
        (4, 10240),
        (48, 2560),
        (512, 2560),
        (7, 48),
        (3, 2560),
        (4100, 96),
    ];
    let cases = [
        (MixedEpilogue::None, 1.0f32),
        (MixedEpilogue::Silu, 0.25),
        (MixedEpilogue::Sigmoid, 0.25),
    ];
    let mut ran = 0;
    for &(n, k) in &shapes {
        let w_cpu = weight(n, k, 0.3, &cpu)?;
        let w_gpu = w_cpu.to_device(&gpu)?;
        for &b in &[1usize, 3] {
            let x_cpu = input(b, k, 1.1, &cpu)?;
            let x_gpu = x_cpu.to_device(&gpu)?;
            for &(ep, scale) in &cases {
                let fused = QMatMul::TensorF16(w_gpu.clone())
                    .gemv_mixed(&x_gpu, ep, scale)
                    .expect("the mixed arm applies on Metal at f32 with <= 4 rows")?;
                let reference = QMatMul::TensorF16(w_cpu.clone()).gemv_mixed(&x_cpu, ep, scale);
                // The CPU has no mixed arm: compose the reference there.
                assert!(reference.is_none(), "the CPU must not claim the mixed arm");
                let w32 = w_cpu.to_dtype(DType::F32)?;
                let y = x_cpu.reshape((b, k))?.matmul(&w32.t()?)?;
                let y: Vec<f32> = y.flatten_all()?.to_vec1()?;
                let y: Vec<f32> = y
                    .into_iter()
                    .map(|v| match ep {
                        MixedEpilogue::None => v,
                        MixedEpilogue::Silu => {
                            let t = scale * v;
                            t / (1.0 + (-t).exp())
                        }
                        MixedEpilogue::Sigmoid => {
                            let t = scale * v;
                            1.0 / (1.0 + (-t).exp())
                        }
                    })
                    .collect();
                let reference = Tensor::from_vec(y, (b, 1, n), &cpu)?;
                assert_eq!(fused.dims(), &[b, 1, n]);
                let (diff, scale_y) = max_abs_diff(&fused.to_device(&cpu)?, &reference)?;
                assert!(
                    diff <= 2e-5 * scale_y.max(1e-3) + 1e-6,
                    "shape ({n},{k}) b={b} {ep:?}: max diff {diff} at scale {scale_y}"
                );
                ran += 1;
            }
        }
    }
    assert_eq!(ran, shapes.len() * 2 * cases.len());
    Ok(())
}

/// `Module::forward` on a `TensorF16` weight takes the mixed arm at decode
/// row counts — and computes the f32 dot the reference does, not the
/// f16-rounded activation the classical route used.
#[test]
fn tensor_f16_forward_takes_the_mixed_arm_for_a_decode_row() -> Result<()> {
    let Ok(gpu) = Device::new_metal(0) else {
        return Ok(());
    };
    let cpu = Device::Cpu;
    let (n, k) = (320usize, 10240usize);
    let w_cpu = weight(n, k, 0.3, &cpu)?;
    let x_cpu = input(1, k, 2.2, &cpu)?;
    let out = QMatMul::TensorF16(w_cpu.to_device(&gpu)?).forward(&x_cpu.to_device(&gpu)?)?;
    let reference = x_cpu
        .reshape((1, k))?
        .matmul(&w_cpu.to_dtype(DType::F32)?.t()?)?
        .reshape((1, 1, n))?;
    let (diff, scale) = max_abs_diff(&out.to_device(&cpu)?, &reference)?;
    assert!(diff <= 2e-5 * scale, "forward: {diff} at {scale}");
    // A prefill's rows (> 4) keep the classical route and its shape.
    let x8 = Tensor::zeros((8, 1, k), DType::F32, &gpu)?;
    let out8 = QMatMul::TensorF16(w_cpu.to_device(&gpu)?).forward(&x8)?;
    assert_eq!(out8.dims(), &[8, 1, n]);
    assert_eq!(out8.dtype(), DType::F32);
    Ok(())
}

/// The mixed GEMV with a grouped RMSNorm of its input inside it
/// (`QMatMul::gemv_mixed_gnorm`, `PrologueGroupNorm`): against an f64
/// reference of the norm then the GEMV at the HyperConnection shapes it is
/// instantiated for (`[320, 10240]` with SiLU and the scales written,
/// `[4, 10240]` plain), four groups of 2 560; the scales it writes against the
/// reference's; and `gemv_mixed_gnorm_serves` exactly where the kernel runs.
#[test]
fn mixed_gemv_with_a_grouped_norm_matches_the_reference() -> Result<()> {
    let Ok(gpu) = Device::new_metal(0) else {
        return Ok(());
    };
    let (groups, eps) = (4usize, 1e-6f32);
    for &(n_out, k, epi, scale, scales) in &[
        (320usize, 10240usize, MixedEpilogue::Silu, 0.25f32, true),
        (4, 10240, MixedEpilogue::None, 1.0, false),
    ] {
        let group = k / groups;
        let w = weight(n_out, k, 1.3, &gpu)?;
        let x = input(1, k, 0.4, &gpu)?;
        let gamma: Vec<f32> = (0..k)
            .map(|i| 1.0 + 0.1 * ((i as f32) * 0.3).sin())
            .collect();
        let g = Tensor::from_vec(gamma.clone(), k, &gpu)?;
        let wv: Vec<f32> = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let mut xn = vec![0f64; k];
        let mut want_s = vec![0f64; groups];
        for gi in 0..groups {
            let seg = &xv[gi * group..(gi + 1) * group];
            let ms = seg.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / group as f64;
            want_s[gi] = 1.0 / (ms + eps as f64).sqrt();
            for j in 0..group {
                xn[gi * group + j] = seg[j] as f64 * want_s[gi] * gamma[gi * group + j] as f64;
            }
        }
        let want: Vec<f64> = (0..n_out)
            .map(|r| {
                let y: f64 = (0..k).map(|c| wv[r * k + c] as f64 * xn[c]).sum();
                match epi {
                    MixedEpilogue::None => y,
                    MixedEpilogue::Silu => {
                        let v = scale as f64 * y;
                        v / (1.0 + (-v).exp())
                    }
                    MixedEpilogue::Sigmoid => unreachable!(),
                }
            })
            .collect();
        let qm = QMatMul::TensorF16(w.clone());
        let got: Vec<f32> = qm
            .gemv_mixed_gnorm(&x, &g, eps, groups, epi, scale, scales)
            .expect("the kernel is instantiated for this shape")?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(got.len(), n_out + if scales { groups } else { 0 });
        // `f64::max` drops a NaN, so a non-finite output would read as no error.
        assert!(
            got.iter().all(|v| v.is_finite()),
            "gemv_mixed_gnorm [{n_out}, {k}]: a non-finite output"
        );
        let mag = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let err = got[..n_out]
            .iter()
            .zip(&want)
            .fold(0f64, |m, (a, b)| m.max((*a as f64 - b).abs()))
            / mag;
        println!("gemv_mixed_gnorm [{n_out}, {k}] {epi:?} rel_err {err:.3e}");
        assert!(err < 1e-5, "gemv_mixed_gnorm [{n_out}, {k}]: rel_err {err}");
        if scales {
            for gi in 0..groups {
                let e = (got[n_out + gi] as f64 - want_s[gi]).abs() / want_s[gi];
                assert!(
                    e < 1e-5,
                    "scale {gi}: {} against {}",
                    got[n_out + gi],
                    want_s[gi]
                );
            }
        }
    }
    use lloom_tensor::kernels::gemv_mixed_gnorm_serves as serves;
    assert!(serves(320, 10240, 4) && serves(4, 10240, 4));
    assert!(
        !serves(10240, 320, 4),
        "the up projection takes the bm8 tile"
    );
    assert!(!serves(320, 10240, 3), "10240 does not split into 3");
    assert!(!serves(320, 10240, 9), "more groups than the scratch holds");
    let qm = QMatMul::TensorF16(weight(10240, 320, 0.2, &gpu)?);
    let g = Tensor::ones(320, DType::F32, &gpu)?;
    assert!(
        qm.gemv_mixed_gnorm(
            &input(1, 320, 0.1, &gpu)?,
            &g,
            eps,
            4,
            MixedEpilogue::None,
            1.0,
            false
        )
        .is_none(),
        "declined where the tile is not instantiated"
    );
    Ok(())
}
