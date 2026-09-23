//! The routed GEMM with the rows grouped by expert (`kernel_mul_mm_id_mapped`,
//! lloom #749 P22) against the routed matmul-vec it replaces from
//! `LLOOM_MOE_MM_FROM` rows: the same `[batch, topk, n]` within the f16
//! dequantisation's tolerance, both activation layouts (`[batch, 1, k]` and
//! `[batch, topk, k]`), bit-stable across runs, and the matmul-vec untouched
//! below the threshold.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::metal::set_moe_mm_from;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{DType, Device, Tensor};

fn run(dev: &Device, dtype: GgmlDType, batch: usize, in_dim1: usize, from: usize) -> Vec<f32> {
    let (e, n, k, topk) = (8usize, 128usize, 256usize, 4usize);
    let w = Tensor::from_vec(
        (0..e * n * k)
            .map(|i| ((i as f32) * 0.0173).sin() * 0.2)
            .collect::<Vec<f32>>(),
        (e, n, k),
        dev,
    )
    .unwrap();
    let q = QTensor::quantize(&w, dtype).unwrap();
    let x = Tensor::from_vec(
        (0..batch * in_dim1 * k)
            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
            .collect::<Vec<f32>>(),
        (batch, in_dim1, k),
        dev,
    )
    .unwrap();
    let ids = Tensor::from_vec(
        (0..batch * topk)
            .map(|i| ((i * 5 + i / topk) % e) as u32)
            .collect::<Vec<u32>>(),
        (batch, topk),
        dev,
    )
    .unwrap();
    set_moe_mm_from(from);
    let y = q.indexed_moe_forward(&x, &ids).unwrap();
    assert_eq!(y.dims(), &[batch, topk, n]);
    assert_eq!(y.dtype(), DType::F32);
    y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn grouped_gemm_matches_the_matmul_vec_and_is_stable() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::F32] {
        for in_dim1 in [1usize, 4] {
            let batch = 70; // 280 pairs over 8 experts: several 32-row blocks and a part block
            let mv = run(&dev, dtype, batch, in_dim1, 0);
            let mm = run(&dev, dtype, batch, in_dim1, 1);
            let mm2 = run(&dev, dtype, batch, in_dim1, 1);
            assert_eq!(
                mm, mm2,
                "{dtype:?} in_dim1={in_dim1}: the GEMM is not bit-stable"
            );
            let mut worst = 0f32;
            let mut scale = 0f32;
            for (a, b) in mv.iter().zip(&mm) {
                assert!(a.is_finite() && b.is_finite(), "{dtype:?}: non-finite");
                worst = worst.max((a - b).abs());
                scale = scale.max(a.abs());
            }
            assert!(
                worst <= 2e-3 * scale.max(1.0),
                "{dtype:?} in_dim1={in_dim1}: worst |mv - mm| {worst} at scale {scale}"
            );
            assert!(
                worst > 0.0 || dtype == GgmlDType::F32,
                "{dtype:?}: the two arms are identical — did the GEMM run?"
            );
            // Below the threshold the matmul-vec answers, bit-identical to itself.
            let below = run(&dev, dtype, 8, in_dim1, 16);
            let below2 = run(&dev, dtype, 8, in_dim1, 0);
            assert_eq!(below, below2, "{dtype:?}: rows under the threshold moved");
        }
    }
    set_moe_mm_from(lloom_tensor::quantized::metal::DEFAULT_MOE_MM_FROM);
}
