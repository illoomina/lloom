//! `QMatMul::TensorF16::forward` past the mixed gemv's rows (the f16-weight
//! GEMM arm, P23) at the tiny test geometry: deterministic across runs and
//! within f16 rounding of the CPU. P30 (1): the tiny Qwen4-exp prefill with
//! f16 stems varied run to run on Metal.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Result, Tensor};

fn case(rows: usize, out: usize, k: usize) -> Result<()> {
    case_with(rows, out, k, 0)
}

/// `view_off > 0`: the weight is a row view (`narrow`) inside a larger
/// matrix starting `view_off` rows in — the stacked stems' shape.
fn case_with(rows: usize, out: usize, k: usize, view_off: usize) -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_metal(0)?;
    let big = out + 2 * view_off;
    let w_all = Tensor::from_vec(
        (0..big * k).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect::<Vec<_>>(),
        (big, k),
        &cpu,
    )?
    .to_dtype(DType::F16)?;
    let w = w_all.narrow(0, view_off, out)?;
    let x = Tensor::from_vec(
        (0..rows * k).map(|i| ((i as f32) * 0.11).cos()).collect::<Vec<_>>(),
        (1, rows, k),
        &cpu,
    )?;
    let want: Vec<f32> = x
        .broadcast_matmul(&w.to_dtype(DType::F32)?.t()?)?
        .flatten_all()?
        .to_vec1()?;
    let w_gpu = w_all.to_device(&gpu)?.narrow(0, view_off, out)?;
    let qm = QMatMul::TensorF16(w_gpu);
    let xg = x.to_device(&gpu)?;
    let mut runs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..5 {
        let y = qm.forward(&xg)?;
        runs.push(y.to_dtype(DType::F32)?.to_device(&cpu)?.flatten_all()?.to_vec1()?);
    }
    for r in &runs[1..] {
        assert_eq!(&runs[0], r, "rows {rows} out {out} k {k} off {view_off}: not deterministic");
    }
    let maxdiff = runs[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    assert!(maxdiff < 4e-3 * scale.max(1.0), "rows {rows} out {out} k {k} off {view_off}: maxdiff {maxdiff} vs cpu");
    Ok(())
}

#[test]
fn tiny_shapes_are_deterministic_and_match_cpu() -> Result<()> {
    for rows in [5usize, 7, 9, 16, 33] {
        for (out, k) in [(64usize, 32usize), (32, 64), (96, 32), (160, 32), (32, 96), (48, 32)] {
            case(rows, out, k)?;
        }
    }
    Ok(())
}

#[test]
fn odd_shapes_and_row_views() -> Result<()> {
    for rows in [7usize, 9] {
        for (out, k) in [(100usize, 32usize), (72, 32), (36, 32), (40, 32), (8, 32), (4, 32), (96, 40), (64, 24), (72, 8)] {
            for off in [0usize, 4, 36] {
                if let Err(e) = case_with(rows, out, k, off) {
                    panic!("{e}");
                }
            }
        }
    }
    Ok(())
}
