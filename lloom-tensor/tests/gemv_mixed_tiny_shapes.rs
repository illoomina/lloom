//! The mixed gemv (f16 weight, f32 activation) at shapes smaller than its
//! tile — the tiny test geometry of the Qwen4-exp HC bracket (down [4×32],
//! up [32×4]): deterministic across runs and equal to the CPU reference.
//! P30 (1): the tiny model's Metal prefill varied run to run from layer 0.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::gemv_mixed::MixedEpilogue;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Result, Tensor};

fn case(out: usize, k: usize, rows: usize) -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_metal(0)?;
    let w = Tensor::from_vec(
        (0..out * k).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect::<Vec<_>>(),
        (out, k),
        &cpu,
    )?
    .to_dtype(DType::F16)?;
    let x = Tensor::from_vec(
        (0..rows * k).map(|i| ((i as f32) * 0.11).cos()).collect::<Vec<_>>(),
        (1, rows, k),
        &cpu,
    )?;
    let want: Vec<f32> = x.broadcast_matmul(&w.to_dtype(DType::F32)?.t()?)?.flatten_all()?.to_vec1()?;
    let qm = QMatMul::TensorF16(w.to_device(&gpu)?);
    let xg = x.to_device(&gpu)?;
    let mut runs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..5 {
        let y = qm
            .gemv_mixed(&xg, MixedEpilogue::None, 1.0)
            .expect("the mixed gemv applies")?;
        runs.push(y.to_device(&cpu)?.flatten_all()?.to_vec1()?);
    }
    for r in &runs[1..] {
        assert_eq!(&runs[0], r, "out {out} k {k} rows {rows}: not deterministic");
    }
    let maxdiff = runs[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maxdiff < 2e-3, "out {out} k {k} rows {rows}: maxdiff {maxdiff} vs cpu");
    Ok(())
}

#[test]
fn tiny_shapes_are_deterministic_and_match_cpu() -> Result<()> {
    for (out, k) in [(4usize, 32usize), (32, 4), (8, 8), (16, 32), (320, 10240), (10240, 320)] {
        for rows in [1usize, 3] {
            case(out, k, rows)?;
        }
    }
    Ok(())
}
