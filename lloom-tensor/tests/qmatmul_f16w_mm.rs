//! `QMatMul::TensorF16` past the mixed gemv's rows on Metal: the quantized
//! GEMM's f16 arm (`matmul_f16_weight_f32`, lloom #749 P23) against the
//! classical cast-matmul-cast route — within half's rounding of each other
//! (the classical route rounds the activations to half; the GEMM keeps
//! them float), bit-stable, and the shape/batch conventions kept.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::metal::matmul_f16_weight_f32;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Module, Tensor};

#[test]
fn f16_weight_gemm_matches_the_cast_route() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let (n, k) = (320usize, 2560usize);
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i as f32) * 0.0031).sin() * 0.05)
            .collect::<Vec<f32>>(),
        (n, k),
        &dev,
    )
    .unwrap()
    .to_dtype(DType::F16)
    .unwrap();
    for dims in [vec![64usize, k], vec![1, 37, k], vec![2, 70, k]] {
        let x = Tensor::from_vec(
            (0..dims.iter().product::<usize>())
                .map(|i| ((i as f32) * 0.017).cos() * 0.5)
                .collect::<Vec<f32>>(),
            dims.clone(),
            &dev,
        )
        .unwrap();
        // The classical route, spelled out.
        let wt = match dims.len() {
            2 => w.t().unwrap(),
            _ => w.broadcast_left(dims[0]).unwrap().t().unwrap(),
        };
        let classical: Vec<f32> = x
            .to_dtype(DType::F16)
            .unwrap()
            .matmul(&wt)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mm = matmul_f16_weight_f32(&x, &w).unwrap();
        let mut od = dims.clone();
        *od.last_mut().unwrap() = n;
        assert_eq!(mm.dims(), &od[..]);
        let mmv: Vec<f32> = mm.flatten_all().unwrap().to_vec1().unwrap();
        let mm2: Vec<f32> = matmul_f16_weight_f32(&x, &w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(mmv, mm2, "{dims:?}: not bit-stable");
        let (mut worst, mut scale) = (0f32, 0f32);
        for (a, b) in classical.iter().zip(&mmv) {
            assert!(a.is_finite() && b.is_finite());
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }
        assert!(
            worst <= 4e-3 * scale.max(1.0),
            "{dims:?}: worst |classical - mm| {worst} at scale {scale}"
        );
        // Through `QMatMul` the route is taken at these rows.
        let q = QMatMul::TensorF16(w.clone());
        let via: Vec<f32> = q
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(via, mmv, "{dims:?}: QMatMul did not take the GEMM route");
    }
}
