//! The projection op (`nn::project`): a request at the dtype the model
//! computes in takes exactly the kernels the weight's own forward takes; a
//! declared triple at another activation dtype casts the input once and
//! stores at the triple's leg; an unserved triple is refused by name; and on
//! Metal each row band resolves to the route written beside it.
use lloom::nn::project::{describe, project, resolve, Epilogue, Spec, Triple, Weight};
use lloom::nn::Linear;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Module, Result, Tensor};

const N: usize = 96;
const K: usize = 256;

fn weights(dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..N * K)
        .map(|i| (((i * 7919) % 1013) as f32 / 1013.0 - 0.5) * 0.25)
        .collect();
    Tensor::from_vec(v, (N, K), dev)
}

fn input(rows: usize, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..rows * K)
        .map(|i| (((i * 104_729) % 977) as f32 / 977.0 - 0.5) * 2.0)
        .collect();
    Tensor::from_vec(v, (1, rows, K), dev)
}

fn blocks(dev: &Device) -> Result<Weight> {
    let q = QTensor::quantize(&weights(dev)?, GgmlDType::Q8_0)?;
    Ok(Weight::Blocks(QMatMul::QTensor(std::sync::Arc::new(q))))
}

fn spec(triple: Triple, rows: usize) -> Spec {
    Spec {
        triple,
        rows,
        epilogue: Epilogue::Store,
    }
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
fn a_dense_projection_is_linears_forward() -> Result<()> {
    let dev = Device::Cpu;
    let l = Linear::new(weights(&dev)?, None);
    let x = input(3, &dev)?;
    let y = project(&x, &Weight::Dense(l.clone()), spec(Triple::F32, 3))?;
    assert_eq!(bits(&y)?, bits(&l.forward(&x)?)?);
    Ok(())
}

#[test]
fn a_packed_projection_is_the_quantized_matmuls_forward() -> Result<()> {
    let dev = Device::Cpu;
    let w = blocks(&dev)?;
    let Weight::Blocks(q) = &w else {
        unreachable!()
    };
    let x = input(5, &dev)?;
    let y = project(&x, &w, spec(Triple::F32, 5))?;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(bits(&y)?, bits(&q.forward(&x)?)?);
    Ok(())
}

/// The head's shape: an f16 stream into an f32-declared projection is cast
/// once and stored at f32, the product of the cast input.
#[test]
fn a_declared_triple_casts_the_input_and_stores_at_its_leg() -> Result<()> {
    let dev = Device::Cpu;
    let w = blocks(&dev)?;
    let Weight::Blocks(q) = &w else {
        unreachable!()
    };
    let x16 = input(2, &dev)?.to_dtype(DType::F16)?;
    let y = project(&x16, &w, spec(Triple::F32, 2))?;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(bits(&y)?, bits(&q.forward(&x16.to_dtype(DType::F32)?)?)?);
    let r = resolve(&w, &spec(Triple::F32, 2), DType::F16)?;
    assert_eq!(r.cast_from, Some(DType::F16));
    Ok(())
}

#[test]
fn an_unserved_triple_is_refused_by_name() -> Result<()> {
    let dev = Device::Cpu;
    let w = blocks(&dev)?;
    let x = input(1, &dev)?;
    let half_acc = Triple {
        act: DType::F16,
        acc: DType::F16,
        store: DType::F16,
    };
    let wide_store = Triple {
        act: DType::F16,
        acc: DType::F32,
        store: DType::F32,
    };
    for (t, text) in [
        (half_acc, "accumulates in f32"),
        (wide_store, "stores at a dtype other than its activation's"),
    ] {
        let e = resolve(&w, &spec(t, 1), DType::F16)
            .unwrap_err()
            .to_string();
        assert!(e.contains(text) && e.contains("q8_0"), "{e}");
        assert!(project(&x, &w, spec(t, 1)).is_err());
    }
    let copy = Weight::Blocks(QMatMul::Tensor(weights(&dev)?));
    let e = project(&x.to_dtype(DType::F16)?, &copy, spec(Triple::F16, 1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("dequantize_all"), "{e}");
    let dense = Weight::Dense(Linear::new(weights(&dev)?, None));
    let e = resolve(&dense, &spec(Triple::F16, 1), DType::F16)
        .unwrap_err()
        .to_string();
    assert!(e.contains("read at its own dtype"), "{e}");
    let e = project(&x.to_dtype(DType::F16)?, &dense, spec(Triple::F16, 1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("read at its own dtype"), "{e}");
    Ok(())
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// The shipped policy's bands (`qmv.mm_from` 9, `qmm.dequant_from` 64)
    /// over a Q8_0 weight, written out: the stream's projection at f16, the
    /// head declared f32 over an f16 stream, and the f32 stream.
    #[test]
    fn each_row_band_resolves_as_written() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let w = blocks(&dev)?;
        assert_eq!(
            describe(&w, Triple::F16, DType::F16),
            "rows 1+ Matvec(F16F32F16), rows 9+ Gemm(F16F32F16), rows 64+ DequantOnceGemm(F16F32F16)"
        );
        assert_eq!(
            describe(&w, Triple::F32, DType::F16),
            "rows 1+ cast f16 + Matvec(F32F32F32), rows 9+ cast f16 + Gemm(F32F32F32), \
             rows 64+ cast f16 + DequantOnceGemm(F32F32F32)"
        );
        assert_eq!(
            describe(&w, Triple::F32, DType::F32),
            "rows 1+ Matvec(F32F32F32), rows 9+ Gemm(F32F32F32), rows 64+ DequantOnceGemm(F32F32F32)"
        );
        let bf16 = Triple::at(DType::BF16);
        assert!(
            describe(&w, bf16, DType::BF16).starts_with("rows 1+ refused: "),
            "{}",
            describe(&w, bf16, DType::BF16)
        );
        Ok(())
    }

    /// At the stream's dtype the op is the weight's own forward on Metal,
    /// in every band: the f16 arm stores f16.
    #[test]
    fn an_f16_request_is_the_quantized_matmuls_forward_in_every_band() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let w = blocks(&dev)?;
        let Weight::Blocks(q) = &w else {
            unreachable!()
        };
        for rows in [1, 9, 64] {
            let x = input(rows, &dev)?.to_dtype(DType::F16)?;
            let y = project(&x, &w, spec(Triple::F16, rows))?;
            assert_eq!(y.dtype(), DType::F16, "rows {rows}");
            assert_eq!(bits(&y)?, bits(&q.forward(&x)?)?, "rows {rows}");
        }
        Ok(())
    }
}
