//! The quantized GEMM's `(f16, f32, f16)` arm (`kernel_mul_mm_<kind>_f16_f16`,
//! #1154 A): past the matmul-vec's rows an f16 activation runs the GEMM and
//! stores f16. The arm widens its half activations into the f32 arm's float
//! tile, so it computes exactly what the f32 arm computes from the same
//! half-rounded activations, rounded once at the store. That is the reference
//! here, bit for bit: the f32 arm on `x16` widened, cast to half. Both GEMM
//! routes (the in-kernel unpack and the dequantise-once scratch) at every
//! quantizable kind and the dense f16 weight, full and partial tiles.
#![cfg(feature = "metal")]
use lloom_tensor::quantized::metal::set_dequant_once_from;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Module, Tensor};

const KINDS: [GgmlDType; 11] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::F16,
];

fn weight(dev: &Device, dtype: GgmlDType, n: usize, k: usize) -> QMatMul {
    let w = Tensor::from_vec(
        (0..n * k)
            .map(|i| ((i as f32) * 0.0173).sin() * 0.2)
            .collect::<Vec<f32>>(),
        (n, k),
        dev,
    )
    .unwrap();
    // The packed weight as the quantized path holds it: `from_qtensor`
    // dequantizes the float kinds to a dense tensor, which this test is not
    // about.
    QMatMul::QTensor(std::sync::Arc::new(QTensor::quantize(&w, dtype).unwrap()))
}

fn bits(t: &Tensor) -> Vec<u16> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<half::f16>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn the_f16_gemm_is_the_f32_gemm_on_half_inputs_rounded_once() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let k = 512usize;
    let mut cells = 0usize;
    // `from` 0: the in-kernel unpack at every row count; 64: the
    // dequantise-once scratch from 64 rows.
    for from in [0usize, 64] {
        set_dequant_once_from(from);
        for dtype in KINDS {
            for n in [37usize, 256] {
                let q = weight(&dev, dtype, n, k);
                for rows in [9usize, 37, 70] {
                    let x = Tensor::from_vec(
                        (0..rows * k)
                            .map(|i| ((i as f32) * 0.011).cos() * 0.5)
                            .collect::<Vec<f32>>(),
                        (1, rows, k),
                        &dev,
                    )
                    .unwrap();
                    let x16 = x.to_dtype(DType::F16).unwrap();
                    let got = q.forward(&x16).unwrap();
                    assert_eq!(
                        got.dtype(),
                        DType::F16,
                        "{dtype:?} n={n} rows={rows}: the store"
                    );
                    let want = q
                        .forward(&x16.to_dtype(DType::F32).unwrap())
                        .unwrap()
                        .to_dtype(DType::F16)
                        .unwrap();
                    let (g, w) = (bits(&got), bits(&want));
                    let differ = g.iter().zip(&w).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differ, 0,
                        "{dtype:?} n={n} rows={rows} dequant_from={from}: {differ} of {} halves differ from the f32 arm rounded once",
                        g.len()
                    );
                    assert_eq!(
                        g,
                        bits(&q.forward(&x16).unwrap()),
                        "{dtype:?}: not bit-stable"
                    );
                    cells += 1;
                }
            }
        }
    }
    println!("f16 GEMM cells bit-identical to the f32 arm rounded once: {cells}");
    assert_eq!(cells, 2 * KINDS.len() * 2 * 3);
}
