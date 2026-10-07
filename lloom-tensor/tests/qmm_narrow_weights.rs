//! The quantized GEMM arms (`QMetalStorage::fwd` past the matmul-vec's rows:
//! the in-kernel unpack below the dequant-once threshold, the dequantised
//! f16 arm from it) read an activation that is a ROW VIEW — a contiguous
//! `narrow` whose first element sits inside its buffer — from that element.
//! Both arms bound the activation at offset 0 (their strides come from a
//! fresh contiguous layout), so a view read the rows before it: the tiny
//! Qwen4-exp indexer's `q_proj` over the chunk rows past the dense range
//! (45 rows of a 64-row prefill, offset 19 rows) read rows 0..45
//! (`tiny_manifest_drift`, CPU vs Metal 0.229). Each arm, each kind below,
//! a view at a nonzero offset against the same rows owned outright and
//! against the dequantised CPU product.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};

fn max_rel(got: &Tensor, want: &Tensor) -> Result<f32> {
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-3);
    Ok((got - want)?.abs()?.max_all()?.to_scalar::<f32>()? / scale)
}

/// `rows` rows taken at row `skip` of a `[skip + rows + 3, k]` activation
/// (a view with an offset and rows after it), through a `[n, k]` weight.
fn case(dtype: GgmlDType, n: usize, k: usize, rows: usize, skip: usize) -> Result<(f32, f32)> {
    let cpu = Device::Cpu;
    let gpu = Device::new_metal(0)?;
    let total = skip + rows + 3;
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i as f32) * 0.37).sin() * 0.5)
            .collect::<Vec<_>>(),
        (n, k),
        &cpu,
    )?;
    let x = Tensor::from_vec(
        (0..total * k)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect::<Vec<_>>(),
        (total, k),
        &cpu,
    )?;
    let wd = QTensor::quantize(&w, dtype)?.dequantize(&cpu)?;
    let want = x.narrow(0, skip, rows)?.matmul(&wd.t()?)?;
    let qm = QMatMul::from_qtensor(QTensor::quantize(&w.to_device(&gpu)?, dtype)?)?;
    let xg = x.to_device(&gpu)?;
    let view = xg.narrow(0, skip, rows)?;
    assert!(view.is_contiguous() && view.layout().start_offset() == skip * k);
    let got_view = qm.forward(&view)?.to_device(&cpu)?;
    let owned = Tensor::from_vec(
        x.narrow(0, skip, rows)?.flatten_all()?.to_vec1::<f32>()?,
        (rows, k),
        &gpu,
    )?;
    let got_owned = qm.forward(&owned)?.to_device(&cpu)?;
    Ok((max_rel(&got_view, &want)?, max_rel(&got_view, &got_owned)?))
}

#[test]
fn a_row_view_is_read_from_its_own_first_row() -> Result<()> {
    // 45 rows: the in-kernel unpack (above the matmul-vec's 8, below the
    // dequant-once threshold's 64); 80 rows: the dequantised f16 arm.
    let mut bad = Vec::new();
    for dtype in [GgmlDType::Q8_0, GgmlDType::Q4_0, GgmlDType::Q4K] {
        let k = if dtype == GgmlDType::Q4K { 256 } else { 32 };
        for &(n, rows, skip) in &[(16usize, 45usize, 19usize), (64, 45, 1), (16, 80, 19), (128, 80, 7)] {
            let (vs_cpu, vs_owned) = case(dtype, n, k, rows, skip)?;
            eprintln!("{dtype:?} n={n} k={k} rows={rows} skip={skip}: vs cpu {vs_cpu:e}, vs owned {vs_owned:e}");
            if vs_cpu > 2e-2 || vs_owned != 0.0 {
                bad.push(format!(
                    "{dtype:?} n={n} k={k} rows={rows} skip={skip}: vs cpu {vs_cpu:e}, vs owned {vs_owned:e}"
                ));
            }
        }
    }
    assert!(bad.is_empty(), "a row view read off its own rows: {bad:#?}");
    Ok(())
}
