use super::*;
fn data(dims: (usize, usize, usize, usize), seed: usize, dtype: DType) -> Result<Tensor> {
    let n = dims.0 * dims.1 * dims.2 * dims.3;
    Tensor::from_vec(
        (0..n)
            .map(|i| (((i * 7 + seed) % 19) as f32 - 9.) / 8.)
            .collect::<Vec<_>>(),
        dims,
        &Device::Cpu,
    )?
    .to_dtype(dtype)
}
fn flat(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect())
}
// Scalar reference reads already-quantized fixture inputs, not unrounded F32
// originals. Independent order: materialize each score row then stable softmax.
fn scalar(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f64) -> Result<Vec<f64>> {
    let (b, h, t, d) = q.dims4()?;
    let (_, kh, n, _) = k.dims4()?;
    let (q, k, v, mask) = (flat(q)?, flat(k)?, flat(v)?, flat(mask)?);
    let mut output = vec![0.; b * h * t * d];
    for batch in 0..b {
        for head in 0..h {
            for query in 0..t {
                let kvhead = head / (h / kh);
                let mut scores = Vec::new();
                for key in 0..n {
                    let dot = (0..d)
                        .map(|i| {
                            q[((batch * h + head) * t + query) * d + i]
                                * k[((batch * kh + kvhead) * n + key) * d + i]
                        })
                        .sum::<f64>();
                    scores.push(dot * scale + mask[batch * n + key]);
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let total = probs.iter().sum::<f64>();
                for i in 0..d {
                    output[((batch * h + head) * t + query) * d + i] = (0..n)
                        .map(|key| {
                            probs[key] / total * v[((batch * kh + kvhead) * n + key) * d + i]
                        })
                        .sum();
                }
            }
        }
    }
    Ok(output)
}
#[test]
fn online_tiles_match_scalar_for_all_model_dtypes_gqa_offsets_and_value_strides() -> Result<()> {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        // Nonzero start and noncontiguous outer stride; unit channel stride.
        let q = data((2, 2, 4, 4), 1, dtype)?.narrow(2, 1, 3)?;
        let k = data((2, 1, 4, 4), 2, dtype)?;
        // Same shape, deliberately nonunit V channel stride (supported by CPU).
        let v = data((2, 1, 4, 4), 3, dtype)?.transpose(2, 3)?;
        let mask = Tensor::from_vec(
            vec![0f32, -0.75, -3., 0.25, 0., -1., -0.5, -2.],
            (2, 1, 1, 4),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?;
        let unmasked = Tensor::zeros((2, 1, 1, 4), dtype, &Device::Cpu)?;
        for n in [1, 3] {
            let q = q.narrow(2, 0, n)?;
            let expected = scalar(&q, &k, &v, &mask, 0.5)?;
            let ignored = scalar(&q, &k, &v, &unmasked, 0.5)?;
            assert!(
                expected
                    .iter()
                    .zip(ignored)
                    .any(|(a, b)| (a - b).abs() > 0.05),
                "mask discriminator is vacuous"
            );
            for tile in [1, 2, 8] {
                let plan = MaskedAttentionPlan::new(
                    &q,
                    &k,
                    &v,
                    &mask,
                    0.5,
                    NonZeroUsize::new(tile).unwrap(),
                )?;
                let mut previous = None;
                for _ in 0..3 {
                    let output = plan.execute()?;
                    assert_eq!(output.dtype(), dtype);
                    assert_eq!(output.dims(), q.dims());
                    let actual = flat(&output)?;
                    let tolerance = match dtype {
                        DType::F32 => 2e-6,
                        DType::F16 => 1e-3,
                        _ => 8e-3,
                    };
                    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (a - e).abs() < tolerance,
                            "{dtype:?} q{n} tile{tile} index{i}: {a} != {e}"
                        );
                    }
                    let bits: Vec<_> = actual.iter().map(|x| x.to_bits()).collect();
                    if let Some(prior) = &previous {
                        assert_eq!(&bits, prior);
                    }
                    previous = Some(bits);
                }
            }
        }
    }
    Ok(())
}
#[test]
fn negative_infinity_mask_selects_absolute_one_key_and_volume_counts_all_tiles() -> Result<()> {
    let q = data((2, 2, 3, 4), 1, DType::F32)?;
    let k = data((2, 1, 4, 4), 2, DType::F32)?;
    let v = data((2, 1, 4, 4), 3, DType::F32)?;
    let mask = Tensor::from_vec(
        vec![
            f32::NEG_INFINITY,
            0.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.,
            f32::NEG_INFINITY,
        ],
        (2, 1, 1, 4),
        &Device::Cpu,
    )?;
    for tile in [1, 3] {
        let plan =
            MaskedAttentionPlan::new(&q, &k, &v, &mask, 0.5, NonZeroUsize::new(tile).unwrap())?;
        assert_eq!(
            plan.scratch(),
            AttentionScratch {
                tensor_allocation_volume: 672,
                host_accumulator_volume: 192,
                opaque_workspace: OpaqueWorkspace::CpuRuntime
            }
        );
        let actual = flat(&plan.execute()?)?;
        let values = flat(&v)?;
        for batch in 0..2 {
            for head in 0..2 {
                for query in 0..3 {
                    for i in 0..4 {
                        assert_eq!(
                            actual[((batch * 2 + head) * 3 + query) * 4 + i],
                            values[(batch * 4 + batch + 1) * 4 + i]
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
#[test]
fn constructor_rejects_unsupported_contracts_without_dispatch() -> Result<()> {
    let q = data((2, 2, 3, 4), 1, DType::F32)?;
    let k = data((2, 1, 4, 4), 2, DType::F32)?;
    let v = data((2, 1, 4, 4), 3, DType::F32)?;
    let mask = Tensor::zeros((2, 1, 1, 4), DType::F32, &Device::Cpu)?;
    let tile = NonZeroUsize::new(2).unwrap();
    for scale in [f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(MaskedAttentionPlan::new(&q, &k, &v, &mask, scale, tile).is_err());
    }
    assert!(MaskedAttentionPlan::new(&q, &k.to_dtype(DType::F16)?, &v, &mask, 1., tile).is_err());
    assert!(
        MaskedAttentionPlan::new(&q, &k, &v, &mask.broadcast_as((2, 2, 3, 4))?, 1., tile).is_err()
    );
    let strided = data((2, 2, 4, 3), 4, DType::F32)?.transpose(2, 3)?;
    assert!(MaskedAttentionPlan::new(&strided, &k, &v, &mask, 1., tile).is_err());
    for malformed in [
        Tensor::zeros((2, 0, 3, 4), DType::F32, &Device::Cpu)?,
        data((2, 3, 3, 4), 1, DType::F32)?,
        data((2, 2, 3, 2), 1, DType::F32)?,
    ] {
        // Three Q heads are not divisible by the two KV heads in this case.
        let keys = if malformed.dim(1)? == 3 {
            k.broadcast_as((2, 2, 4, 4))?
        } else {
            k.clone()
        };
        let values = if malformed.dim(1)? == 3 {
            v.broadcast_as((2, 2, 4, 4))?
        } else {
            v.clone()
        };
        assert!(MaskedAttentionPlan::new(&malformed, &keys, &values, &mask, 1., tile).is_err());
    }
    assert!(mul(&[usize::MAX, 2]).is_err());
    assert!(sum(&[isize::MAX as usize, 1]).is_err());
    assert!(i32_dim(i32::MAX as usize + 1).is_err());
    if let Some(n) = (u32::MAX as usize).checked_add(1) {
        assert!(u32_count(n).is_err());
    }
    Ok(())
}

#[test]
fn huge_zero_stride_view_rejects_checked_shape_before_any_native_work() -> Result<()> {
    // Four backing scalar bytes per view; no giant tensor allocation. Every
    // dimension fits I32, but their product does not fit usize on 32/64-bit.
    let scalar = Tensor::new(0f32, &Device::Cpu)?;
    let huge = i32::MAX as usize;
    let q = scalar.broadcast_as((huge, huge, 8, 8))?;
    let k = scalar.broadcast_as((huge, 1, 8, 8))?;
    let mask = scalar.broadcast_as((huge, 1, 1, 8))?;
    let result = MaskedAttentionPlan::new(&q, &k, &k, &mask, 0.5, NonZeroUsize::new(1).unwrap());
    match result {
        Ok(_) => panic!("overflowing logical shape admitted"),
        Err(e) => assert!(e.to_string().contains("product/byte overflow"), "{e}"),
    }
    Ok(())
}
