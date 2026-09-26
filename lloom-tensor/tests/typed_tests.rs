//! `Typed<B, E>` view: the runtime half of the RESTRUCTURING §6 gates. The
//! compile-time half (a backend, dtype or triple mismatch is a type error) is
//! the `compile_fail` doctests on `Typed`; here we check the boundary checks,
//! the deref-through, the visible transfer and cast, and the triple's legs.

use half::f16;
use lloom_tensor::typed::{AnyDtype, Cpu, Element, Metal, Triple, Typed, F16F32F16, F32F32F32};
use lloom_tensor::{DType, Device, Result, Tensor, WithDType};

#[test]
fn try_from_accepts_matching_backend() -> Result<()> {
    let t = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let typed: Typed<Cpu> = Typed::try_from(t)?;
    // Deref-through: an untyped `Tensor` method resolves on the view unchanged,
    // with no wrapper method of its own.
    assert_eq!(typed.dims(), &[2, 3]);
    assert_eq!(typed.dtype(), DType::F32);
    assert_eq!(
        typed.to_vec2::<f32>()?,
        vec![vec![0., 1., 2.], vec![3., 4., 5.]]
    );
    Ok(())
}

#[test]
fn try_from_rejects_wrong_backend() -> Result<()> {
    let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
    // A CPU tensor tagged `Metal` must be refused at the boundary — the tag can
    // never lie, which is what lets the unchecked internal `wrap` be sound.
    let err = Typed::<Metal>::try_from(t).unwrap_err();
    assert!(
        err.to_string().contains("metal"),
        "error should name the requested backend, got: {err}"
    );
    Ok(())
}

#[test]
fn same_backend_op_preserves_values_and_tag() -> Result<()> {
    let a: Typed<Cpu> = Tensor::arange(0f32, 4f32, &Device::Cpu)?.try_into()?;
    let b: Typed<Cpu> = Tensor::ones(4, DType::F32, &Device::Cpu)?.try_into()?;
    // Same-backend `add` returns `Typed<Cpu>`; the compute is the untyped op
    // verbatim, so the values match `Tensor::add` exactly.
    let c = a.add(&b)?;
    assert_eq!(c.to_vec1::<f32>()?, vec![1., 2., 3., 4.]);
    Ok(())
}

#[test]
fn cpu_transfer_is_identity_on_cpu() -> Result<()> {
    let a: Typed<Cpu> = Tensor::arange(0f32, 4f32, &Device::Cpu)?.try_into()?;
    let same = a.cpu()?;
    assert_eq!(same.to_vec1::<f32>()?, vec![0., 1., 2., 3.]);
    Ok(())
}

/// The visible-transfer round-trip (RESTRUCTURING §6 gate): a `Typed<Metal>` to
/// `.cpu()` to `Typed<Cpu>` produces the same values, and back again. Skips
/// gracefully when no Metal device is present (the depthwise-conv convention).
#[cfg(feature = "metal")]
#[test]
fn gpu_cpu_round_trip_preserves_values() -> Result<()> {
    let Ok(metal) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return Ok(());
    };
    let src = Tensor::arange(0f32, 6f32, &metal)?.reshape((2, 3))?;
    let on_gpu: Typed<Metal> = src.try_into()?;

    // Metal -> CPU: values survive the visible transfer.
    let on_cpu: Typed<Cpu> = on_gpu.cpu()?;
    assert!(on_cpu.device().is_cpu());
    let cpu_vals = on_cpu.to_vec2::<f32>()?;
    assert_eq!(cpu_vals, vec![vec![0., 1., 2.], vec![3., 4., 5.]]);

    // CPU -> GPU (via the platform-gpu alias) and back: still identical.
    let back_on_gpu: Typed<Metal> = on_cpu.gpu()?;
    assert!(back_on_gpu.device().is_metal());
    let round_tripped = back_on_gpu.cpu()?.to_vec2::<f32>()?;
    assert_eq!(round_tripped, cpu_vals);
    Ok(())
}

#[test]
fn try_from_checks_the_dtype_when_the_view_names_one() -> Result<()> {
    let t = Tensor::arange(0f32, 4f32, &Device::Cpu)?;
    let typed: Typed<Cpu, f32> = t.clone().try_into()?;
    assert_eq!(typed.dtype(), DType::F32);
    // The same tensor tagged f16 is refused at the boundary, naming both
    // dtypes; the untyped-dtype view accepts it as before.
    let err = Typed::<Cpu, f16>::try_from(t.clone()).unwrap_err();
    assert_eq!(err.to_string(), "tensor is f32, not f16");
    let any: Typed<Cpu> = t.try_into()?;
    assert_eq!(any.dtype(), DType::F32);
    Ok(())
}

#[test]
fn the_backend_is_checked_before_the_dtype() -> Result<()> {
    let t = Tensor::zeros(2, DType::F16, &Device::Cpu)?;
    let err = Typed::<Metal, f32>::try_from(t).unwrap_err();
    assert!(err.to_string().contains("metal"), "{err}");
    Ok(())
}

#[test]
fn a_dtype_typed_op_keeps_both_tags_and_the_untyped_values() -> Result<()> {
    let a: Typed<Cpu, f16> = Tensor::arange(0f32, 4f32, &Device::Cpu)?
        .to_dtype(DType::F16)?
        .try_into()?;
    let b: Typed<Cpu, f16> = Tensor::ones(4, DType::F16, &Device::Cpu)?.try_into()?;
    let c: Typed<Cpu, f16> = a.add(&b)?;
    let untyped = a.as_tensor().add(b.as_tensor())?;
    assert_eq!(c.to_vec1::<f16>()?, untyped.to_vec1::<f16>()?);
    assert_eq!(
        c.to_vec1::<f16>()?,
        [1., 2., 3., 4.].map(f16::from_f32).to_vec()
    );
    // Transfers keep the dtype tag.
    let same: Typed<Cpu, f16> = c.cpu()?;
    assert_eq!(same.dtype(), DType::F16);
    Ok(())
}

#[test]
fn cast_is_the_visible_dtype_change_and_erase_drops_the_tag() -> Result<()> {
    let a: Typed<Cpu, f32> = Tensor::new(&[0.5f32, 1.5, -2.0], &Device::Cpu)?.try_into()?;
    let h: Typed<Cpu, f16> = a.cast::<f16>()?;
    assert_eq!(h.dtype(), DType::F16);
    assert_eq!(
        h.to_vec1::<f16>()?,
        [0.5, 1.5, -2.0].map(f16::from_f32).to_vec()
    );
    let back: Typed<Cpu> = h.erase_dtype();
    assert_eq!(back.dtype(), DType::F16);
    Ok(())
}

/// The legs each named triple carries, against the runtime dtypes they must
/// map to — an absolute reference per leg, not a comparison between triples.
#[test]
fn each_triple_names_its_legs() {
    fn legs<T: Triple>() -> (DType, DType, DType) {
        (
            <T::Act as WithDType>::DTYPE,
            <T::Acc as WithDType>::DTYPE,
            <T::Store as WithDType>::DTYPE,
        )
    }
    assert_eq!(legs::<F32F32F32>(), (DType::F32, DType::F32, DType::F32));
    assert_eq!(legs::<F16F32F16>(), (DType::F16, DType::F32, DType::F16));
    // The marker's runtime dtype is the one `WithDType` assigns the scalar.
    assert_eq!(<f16 as Element>::RUNTIME_DTYPE, Some(DType::F16));
    assert_eq!(<AnyDtype as Element>::RUNTIME_DTYPE, None);
    assert!(<AnyDtype as Element>::matches(DType::BF16));
    assert!(!<f16 as Element>::matches(DType::BF16));
}

/// The type-level triple and the runtime quantized matmul-vec selector name
/// the same activation and store legs.
#[cfg(feature = "metal")]
#[test]
fn each_triple_lowers_to_the_matching_gemv_selector() {
    use lloom_tensor::kernels::{GemvFloat, GemvTriple};
    fn leg(d: DType) -> GemvFloat {
        match d {
            DType::F32 => GemvFloat::F32,
            DType::F16 => GemvFloat::F16,
            other => panic!("no GEMV leg for {other:?}"),
        }
    }
    fn check<T: Triple>(expected: GemvTriple) {
        assert_eq!(T::GEMV, expected);
        assert!(T::GEMV.is_implemented());
        assert_eq!(T::GEMV.activation(), leg(<T::Act as WithDType>::DTYPE));
        assert_eq!(T::GEMV.store(), leg(<T::Store as WithDType>::DTYPE));
    }
    check::<F32F32F32>(GemvTriple::F32F32F32);
    check::<F16F32F16>(GemvTriple::F16F32F16);
}
