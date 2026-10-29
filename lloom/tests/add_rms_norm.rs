use lloom::nn::ops::{add_rms_norm, rms_norm};
use lloom_tensor::{DType, Device, Result, Tensor};

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

fn check(dev: &Device, dtype: DType, rows: usize, width: usize, scale: f32) -> Result<()> {
    let n = rows * width;
    let input = |phase: usize| -> Result<Tensor> {
        let v: Vec<f32> = (0..n + width)
            .map(|i| (((i * 17 + phase) % 97) as f32 / 43.0 - 1.0) * scale)
            .collect();
        Tensor::new(v.as_slice(), dev)?
            .to_dtype(dtype)?
            .narrow(0, width, n)?
            .reshape((rows, 1, width))
    };
    let lhs = input(3)?;
    let rhs = input(29)?;
    let w: Vec<f32> = (0..width + 3)
        .map(|i| 0.5 + (i % 13) as f32 / 11.0)
        .collect();
    let alpha = Tensor::new(w.as_slice(), dev)?
        .to_dtype(dtype)?
        .narrow(0, 3, width)?;
    let want_sum = (&lhs + &rhs)?;
    let want_norm = rms_norm(&want_sum, &alpha, 1e-5)?;
    let (sum, norm) = add_rms_norm(&lhs, &rhs, &alpha, 1e-5)?;
    assert_eq!(sum.dims(), lhs.dims());
    assert_eq!(norm.dims(), lhs.dims());
    assert!(sum.is_contiguous() && norm.is_contiguous());
    assert_eq!(
        values(&sum)?,
        values(&want_sum)?,
        "rounded residual {dtype:?}/{rows}/{width}"
    );
    assert_eq!(
        values(&norm)?,
        values(&want_norm)?,
        "composed parity {dtype:?}/{rows}/{width}"
    );
    let got = values(&norm)?;
    assert!(got.windows(2).any(|w| w[0] != w[1]), "non-vacuity");
    assert!(got.iter().all(|v| v.is_finite()));
    if dev.is_metal() {
        let sums = values(&want_sum)?;
        let weights = values(&alpha)?;
        let tolerance = match dtype {
            DType::F16 => 0.003,
            DType::BF16 => 0.025,
            _ => 2e-5,
        };
        for (r, row) in sums.chunks(width).enumerate() {
            let ssq = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / width as f64;
            for c in 0..width {
                let truth = row[c] as f64 * weights[c] as f64 / (ssq + 1e-5).sqrt();
                assert!(
                    (got[r * width + c] as f64 - truth).abs() < tolerance,
                    "independent reference row={r} col={c} got={} truth={truth}",
                    got[r * width + c]
                );
            }
        }
    }
    for _ in 0..3 {
        let (s, n) = add_rms_norm(&lhs, &rhs, &alpha, 1e-5)?;
        assert_eq!(values(&s)?, values(&sum)?);
        assert_eq!(values(&n)?, got);
    }
    Ok(())
}

#[test]
fn composed_cpu_fallback_keeps_both_results() -> Result<()> {
    for rows in [1, 4, 9] {
        for width in [3, 33, 64] {
            check(&Device::Cpu, DType::F32, rows, width, 1.0)?;
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn metal_matrix() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32, DType::BF16] {
        for (rows, width) in [(1, 33), (4, 2048), (9, 64), (17, 257), (512, 2048)] {
            for scale in [1.0, 0.001] {
                check(&dev, dtype, rows, width, scale)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_split_pair_parity_independent_reference_and_determinism() -> Result<()> {
    lloom_tensor::isolation::claim("add-rms-norm parameter policy");
    lloom_tensor::kernels::set_default_param_style(lloom_tensor::kernels::ParamStyle::Split);
    metal_matrix()
}

#[cfg(feature = "metal")]
#[test]
fn metal_packed_pair_parity_independent_reference_and_determinism() -> Result<()> {
    lloom_tensor::isolation::claim("add-rms-norm parameter policy");
    lloom_tensor::kernels::set_default_param_style(lloom_tensor::kernels::ParamStyle::Packed);
    metal_matrix()
}

#[test]
fn noncontiguous_fallback_and_invalid_shapes_keep_existing_contract() -> Result<()> {
    let a = Tensor::arange(0f32, 24f32, &Device::Cpu)?
        .reshape((2, 3, 4))?
        .transpose(0, 1)?;
    let b = a.affine(0.125, -0.2)?;
    let alpha = Tensor::ones(4, DType::F32, &Device::Cpu)?;
    let (sum, normalized) = add_rms_norm(&a, &b, &alpha, 1e-5)?;
    let want = (&a + &b)?;
    assert_eq!(values(&sum)?, values(&want)?);
    assert_eq!(
        values(&normalized)?,
        values(&rms_norm(&want, &alpha, 1e-5)?)?
    );
    let bad = Tensor::ones((2, 4), DType::F32, &Device::Cpu)?;
    assert!(add_rms_norm(&a, &bad, &alpha, 1e-5).is_err());
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_noncontiguous_fallback_returns_contiguous_results() -> Result<()> {
    lloom_tensor::isolation::claim("add-rms-norm fallback");
    let dev = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32, DType::BF16] {
        let a = Tensor::arange(0f32, 24f32, &dev)?
            .to_dtype(dtype)?
            .reshape((2, 3, 4))?
            .transpose(0, 1)?;
        let b = a.affine(0.125, -0.2)?;
        let alpha = Tensor::ones(4, dtype, &dev)?;
        assert!(!lloom::nn::ops::add_rms_norm_applies(&a, &b, &alpha));
        let (sum, normalized) = add_rms_norm(&a, &b, &alpha, 1e-5)?;
        let want = (&a + &b)?;
        assert!(sum.is_contiguous() && normalized.is_contiguous());
        assert_eq!(values(&sum)?, values(&want)?);
        assert_eq!(
            values(&normalized)?,
            values(&rms_norm(&want, &alpha, 1e-5)?)?
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn paired_storage_keeps_either_plane_alive_until_its_last_reader() -> Result<()> {
    lloom_tensor::isolation::claim("add-rms-norm paired storage");
    let dev = Device::new_metal(0)?;
    let lhs = Tensor::arange(0f32, 256f32, &dev)?.reshape((4, 1, 64))?;
    let rhs = lhs.affine(-0.125, 0.33)?;
    let alpha = Tensor::ones(64, DType::F32, &dev)?;
    let (sum, norm) = add_rms_norm(&lhs, &rhs, &alpha, 1e-5)?;
    {
        let (ss, sl) = sum.storage_and_layout();
        let (ns, nl) = norm.storage_and_layout();
        let (lloom_tensor::Storage::Metal(ss), lloom_tensor::Storage::Metal(ns)) = (&*ss, &*ns)
        else {
            panic!("Metal outputs")
        };
        assert!(lloom_tensor::kernels::metal::BufferHandle::ptr_eq(
            ss.buffer_handle(),
            ns.buffer_handle()
        ));
        assert_eq!(nl.start_offset() - sl.start_offset(), lhs.elem_count());
        assert!(ss.buffer().length() >= 2 * lhs.elem_count() * 4);
    }
    let want_norm = values(&norm)?;
    drop(sum);
    dev.synchronize()?;
    let competing = Tensor::full(111f32, (2, lhs.elem_count()), &dev)?;
    assert_eq!(
        values(&norm)?,
        want_norm,
        "dropping the residual cannot release the normalized plane"
    );
    drop(competing);
    let (sum, norm) = add_rms_norm(&lhs, &rhs, &alpha, 1e-5)?;
    let want_sum = values(&sum)?;
    drop(norm);
    dev.synchronize()?;
    let competing = Tensor::full(-77f32, (2, lhs.elem_count()), &dev)?;
    assert_eq!(
        values(&sum)?,
        want_sum,
        "dropping the normalized plane cannot release the residual"
    );
    drop(competing);
    Ok(())
}
