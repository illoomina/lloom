//! The quantized dense stems at `m = 1` against `m = 4` rows, timed: a
//! batched decode step runs every stem once over its rows, and the question
//! is whether the `m = 4` path (the mm kernel or the mv kernel over rows)
//! costs less than four `m = 1` gemvs. Ignored: a measurement, run by hand
//! with `--ignored --nocapture` under `lloom-arb measure`.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Module, Tensor};
use std::time::Instant;

#[test]
#[ignore]
fn quantized_stems_m1_vs_m4() {
    let dev = Device::new_metal(0).unwrap();
    let (n, k) = (10240usize, 2560usize); // attn_qkv's shape
    for dtype in [
        GgmlDType::Q6K,
        GgmlDType::Q4K,
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
    ] {
        let w = Tensor::randn(0f32, 0.05, (n, k), &dev).unwrap();
        let q = QMatMul::from_qtensor(QTensor::quantize(&w, dtype).unwrap()).unwrap();
        let x1 = Tensor::randn(0f32, 1.0, (1, 1, k), &dev).unwrap();
        let x4 = Tensor::randn(0f32, 1.0, (1, 4, k), &dev).unwrap(); // `[1, B, k]`: the MoE's rows-as-positions shape
        let x16 = Tensor::randn(0f32, 1.0, (1, 16, k), &dev).unwrap(); // past the per-row gemv rows: the mm kernel
        for _ in 0..5 {
            q.forward(&x1).unwrap();
            q.forward(&x4).unwrap();
        }
        dev.synchronize().unwrap();
        let iters = 100;
        let t = Instant::now();
        for _ in 0..iters {
            q.forward(&x1).unwrap();
        }
        dev.synchronize().unwrap();
        let m1 = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            q.forward(&x4).unwrap();
        }
        dev.synchronize().unwrap();
        let m4 = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        for _ in 0..3 {
            q.forward(&x16).unwrap();
        }
        dev.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            q.forward(&x16).unwrap();
        }
        dev.synchronize().unwrap();
        let m16 = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // Identity of the m=4 path against four m=1 rows.
        let y4 = q.forward(&x4).unwrap();
        let mut worst = 0f32;
        for r in 0..4 {
            let y1 = q
                .forward(&x4.narrow(1, r, 1).unwrap().contiguous().unwrap())
                .unwrap();
            let a: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
            let b: Vec<f32> = y4
                .narrow(1, r, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let d = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(d / scale);
        }
        println!(
            "QMM dtype={dtype:?} [{n}x{k}] m1={m1:.3}ms m4(gemv rows)={m4:.3}ms m16(mm)={m16:.3}ms m4/(4*m1)={:.2} m16/(16*m1)={:.2} rel_diff={worst:e}",
            m4 / (4.0 * m1),
            m16 / (16.0 * m1)
        );
    }
}
