//! Packed activation views must survive the real multi-row attention layout.
use super::*;
use lloom_tensor::{DType, Device};

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + salt) % 29) as f32 / 64. - 0.21875)
        .collect()
}
fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= 2e-5, "{label}[{i}]: {a} != {e}");
    }
}
fn linear_reference(x: &[f32], w: &[f32], bias: &[f32], d: usize) -> Vec<f32> {
    x.chunks_exact(d)
        .flat_map(|row| {
            w.chunks_exact(d).zip(bias).map(move |(weight, &bias)| {
                row.iter().zip(weight).map(|(&a, &b)| a * b).sum::<f32>() + bias
            })
        })
        .collect()
}
fn head_reference(packed: &[f32], d: usize, arm: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for b in 0..2 {
        for h in 0..2 {
            for t in 0..3 {
                for k in 0..d / 2 {
                    out.push(packed[(b * 3 + t) * 3 * d + arm * d + h * (d / 2) + k]);
                }
            }
        }
    }
    out
}
fn dense(rows: usize, cols: usize, salt: usize) -> Result<Linear> {
    Ok(Linear::new(
        Tensor::from_vec(values(rows * cols, salt), (rows, cols), &Device::Cpu)?,
        Some(Tensor::from_vec(
            values(rows, salt + 1),
            rows,
            &Device::Cpu,
        )?),
    ))
}

#[test]
fn packed_dense_multirow_attention_layout_matches_scalar() -> Result<()> {
    let d = 4;
    let input = values(2 * 3 * d, 2);
    let x = Tensor::from_slice(&input, (2, 3, d), &Device::Cpu)?;
    let packed = PackedQkv::new(dense(3 * d, d, 3)?)?;
    let (q, k, v) = packed.project(&x)?;
    for part in [&q, &k, &v] {
        assert_eq!(part.dims(), [2, 3, d]);
        assert!(!part.is_contiguous());
        assert_eq!(part.stride(), [9 * d, 3 * d, 1]);
    }
    let expected = linear_reference(&input, &values(3 * d * d, 3), &values(3 * d, 4), d);
    let expected_v: Vec<_> = expected
        .chunks_exact(3 * d)
        .flat_map(|r| r[2 * d..].iter().copied())
        .collect();
    let expected_output = linear_reference(&expected_v, &values(d * d, 5), &values(d, 6), d);
    let attention = Attention::from_qkv(packed, dense(d, d, 5)?, 2)?;
    let mask = Tensor::from_slice(&[0f32, -2., -10000.], (1, 1, 1, 3), &Device::Cpu)?;
    let mut calls = 0;
    let out = attention.forward_with_attention(&x, Some(&mask), &mut |q, k, v, m, scale| {
        calls += 1;
        assert_eq!(m.map(|m| m as *const Tensor), Some(&mask as *const Tensor));
        assert_eq!(scale, (2f64).powf(-0.5));
        for (arm, p) in [q, k, v].into_iter().enumerate() {
            assert_eq!(p.dims(), [2, 2, 3, 2]);
            assert!(p.is_contiguous());
            close(
                &p.flatten_all()?.to_vec1::<f32>()?,
                &head_reference(&expected, d, arm),
                "dense QKV",
            );
        }
        Ok(v.clone())
    })?;
    assert_eq!(calls, 1);
    assert_eq!(out.dims(), [2, 3, d]);
    close(
        &out.flatten_all()?.to_vec1::<f32>()?,
        &expected_output,
        "dense output",
    );
    Ok(())
}

// The CPU Q8_0 route rounds BOTH operands to signed bytes with an f16 scale.
// One width32 block makes its integer dot and scale fold explicit here.
fn q8_block(row: &[f32]) -> (Vec<i32>, f32) {
    assert_eq!(row.len(), 32);
    let maximum = row.iter().fold(0f32, |a, x| a.max(x.abs()));
    let scale = maximum / 127.;
    let inverse = if scale == 0. { 0. } else { 1. / scale };
    (
        row.iter().map(|x| (x * inverse).round() as i32).collect(),
        half::f16::from_f32(scale).to_f32(),
    )
}
fn q8_reference(x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    x.chunks_exact(32)
        .flat_map(|row| {
            let (x, dx) = q8_block(row);
            w.chunks_exact(32).zip(bias).map(move |(weight, &bias)| {
                let (w, dw) = q8_block(weight);
                let dot: i32 = x.iter().zip(w).map(|(&x, w)| x * w).sum();
                dot as f32 * dw * dx + bias
            })
        })
        .collect()
}
fn quantized(rows: usize, salt: usize) -> Result<crate::quantized_nn::Linear> {
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    let weight = Tensor::from_vec(values(rows * 32, salt), (rows, 32), &Device::Cpu)?;
    let weight = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
    assert_eq!(weight.dtype(), GgmlDType::Q8_0);
    crate::quantized_nn::Linear::from_arc(
        std::sync::Arc::new(weight),
        Some(Tensor::from_vec(
            values(rows, salt + 1),
            rows,
            &Device::Cpu,
        )?),
    )
}

#[test]
fn packed_quantized_multirow_attention_layout_matches_scalar() -> Result<()> {
    // Normal CI uses nextest (one process/test); plain cargo test needs an exact
    // filter. install claims process ownership and fails if globals were already
    // initialized: never silently execute a dequantized or multiworker arm.
    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning.cpu.num_threads = 1;
    tuning.qmm.dequantize_all = false;
    tuning.qmm.dequantize_all_f16 = false;
    lloom_tensor::tuning::install(tuning).expect("isolated quantized fixture tuning");
    assert_eq!(lloom_tensor::utils::barrier_pool().n_workers(), 0);
    let d = 32;
    let input = values(2 * 3 * d, 2);
    let x = Tensor::from_slice(&input, (2, 3, d), &Device::Cpu)?;
    let packed = PackedQkv::new(quantized(3 * d, 3)?)?;
    let (q, k, v) = packed.project(&x)?;
    for part in [&q, &k, &v] {
        assert!(!part.is_contiguous());
        assert_eq!(part.stride(), [9 * d, 3 * d, 1]);
    }
    let expected = q8_reference(&input, &values(3 * d * d, 3), &values(3 * d, 4));
    // A known callback output isolates the output projection's own activation rounding.
    let attended = values(2 * 3 * d, 11);
    let expected_output = q8_reference(&attended, &values(d * d, 5), &values(d, 6));
    let attention = Attention::from_qkv(packed, quantized(d, 5)?, 2)?;
    let mut calls = 0;
    let out = attention.forward_with_attention(&x, None, &mut |q, k, v, mask, scale| {
        calls += 1;
        assert!(mask.is_none());
        assert_eq!(scale, (16f64).powf(-0.5));
        for (arm, p) in [q, k, v].into_iter().enumerate() {
            assert_eq!(p.dims(), [2, 2, 3, 16]);
            assert!(p.is_contiguous());
            close(
                &p.flatten_all()?.to_vec1::<f32>()?,
                &head_reference(&expected, d, arm),
                "quantized QKV",
            );
        }
        Tensor::from_slice(&attended, (2, 3, 2, 16), &Device::Cpu)?
            .transpose(1, 2)?
            .contiguous()
    })?;
    assert_eq!(calls, 1);
    close(
        &out.flatten_all()?.to_vec1::<f32>()?,
        &expected_output,
        "quantized output",
    );
    Ok(())
}

#[test]
fn packed_geometry_refuses_invalid_weights_bias_and_heads() -> Result<()> {
    for (rows, cols, bias) in [(0, 0, None), (11, 4, None), (12, 4, Some(11))] {
        let layer = Linear::new(
            Tensor::zeros((rows, cols), DType::F32, &Device::Cpu)?,
            bias.map(|n| Tensor::zeros(n, DType::F32, &Device::Cpu))
                .transpose()?,
        );
        assert!(PackedQkv::new(layer).is_err());
    }
    for heads in [0, 3, 5] {
        assert!(
            Attention::from_qkv(PackedQkv::new(dense(12, 4, 0)?)?, dense(4, 4, 1)?, heads).is_err()
        );
    }
    assert!(Attention::from_qkv(PackedQkv::new(dense(12, 4, 0)?)?, dense(3, 4, 1)?, 2).is_err());
    assert!(Attention::from_qkv(PackedQkv::new(dense(12, 4, 0)?)?, dense(4, 4, 1)?, 2).is_ok());
    Ok(())
}
