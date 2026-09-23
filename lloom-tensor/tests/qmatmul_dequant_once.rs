//! A quantized stem at a prefill's rows: dequantised to half once and run
//! through the f16 GEMM arm (`LLOOM_QMM_DEQUANT_FROM`, lloom #749 P26)
//! against the in-kernel unpack — the same half weights, so within half's
//! rounding of each other (bit-identical when the unpack rounds the same
//! way), bit-stable, and the rows under the threshold untouched.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::metal::{set_dequant_gemm_steel, set_dequant_once_from};
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Module, Tensor};

fn run(dev: &Device, dtype: GgmlDType, rows: usize, from: usize) -> Vec<f32> {
    run_with(dev, dtype, rows, from, false)
}

/// [`run`], choosing the GEMM over the scratch: the quantized f16 arm
/// (`steel = false`) or the steel gemm with a float result.
fn run_with(dev: &Device, dtype: GgmlDType, rows: usize, from: usize, steel: bool) -> Vec<f32> {
    set_dequant_gemm_steel(steel);
    let (n, k) = (256usize, 512usize);
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i as f32) * 0.0173).sin() * 0.2)
            .collect::<Vec<f32>>(),
        (n, k),
        dev,
    )
    .unwrap();
    let q = QMatMul::from_qtensor(QTensor::quantize(&w, dtype).unwrap()).unwrap();
    let x = Tensor::from_vec(
        (0..rows * k)
            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
            .collect::<Vec<f32>>(),
        (1, rows, k),
        dev,
    )
    .unwrap();
    set_dequant_once_from(from);
    q.forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn dequantised_once_matches_the_in_kernel_unpack_and_is_stable() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::Q5K,
    ] {
        let a = run(&dev, dtype, 128, 0);
        let b = run(&dev, dtype, 128, 1);
        let b2 = run(&dev, dtype, 128, 1);
        assert_eq!(b, b2, "{dtype:?}: not bit-stable");
        let (mut worst, mut scale) = (0f32, 0f32);
        for (p, q) in a.iter().zip(&b) {
            assert!(p.is_finite() && q.is_finite());
            worst = worst.max((p - q).abs());
            scale = scale.max(p.abs());
        }
        assert!(
            worst <= 2e-3 * scale.max(1.0),
            "{dtype:?}: worst {worst} at scale {scale}"
        );
        println!(
            "{dtype:?}: worst |unpack - dequant-once| {worst:e} at {scale} (identical: {})",
            worst == 0.0
        );
        let below = run(&dev, dtype, 16, 64);
        let below2 = run(&dev, dtype, 16, 0);
        assert_eq!(below, below2, "{dtype:?}: rows under the threshold moved");

        // The steel gemm over the same scratch (half operands, a float
        // result): another tile loop, so within half's rounding of the
        // unpack, not bit-identical; bit-stable; and at an odd row count
        // (the tile's edge path).
        for rows in [128usize, 77] {
            let c = run_with(&dev, dtype, rows, 1, true);
            let c2 = run_with(&dev, dtype, rows, 1, true);
            assert_eq!(c, c2, "{dtype:?}: the steel route is not bit-stable");
            let a = run_with(&dev, dtype, rows, 0, false);
            let (mut worst, mut scale) = (0f32, 0f32);
            for (p, q) in a.iter().zip(&c) {
                assert!(p.is_finite() && q.is_finite());
                worst = worst.max((p - q).abs());
                scale = scale.max(p.abs());
            }
            assert!(
                worst <= 2e-3 * scale.max(1.0),
                "{dtype:?} steel at {rows} rows: worst {worst} at scale {scale}"
            );
            println!("{dtype:?} steel at {rows} rows: worst |unpack - steel| {worst:e} at {scale}");
        }
    }
    set_dequant_once_from(lloom_tensor::quantized::metal::DEFAULT_DEQUANT_ONCE_FROM);
    set_dequant_gemm_steel(lloom_tensor::quantized::metal::DEFAULT_DEQUANT_GEMM_STEEL);
}
