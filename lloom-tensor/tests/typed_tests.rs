//! `Typed<B>` view: the runtime half of the RESTRUCTURING §6 gates. The
//! compile-time half (mismatch is a type error) is the `compile_fail` doctest
//! on `Typed`; here we check the boundary check, the deref-through, and the
//! visible-transfer round-trip.

use lloom_tensor::typed::{Cpu, Metal, Typed};
use lloom_tensor::{DType, Device, Result, Tensor};

#[test]
fn try_from_accepts_matching_backend() -> Result<()> {
    let t = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let typed: Typed<Cpu> = Typed::try_from(t)?;
    // Deref-through: an untyped `Tensor` method resolves on the view unchanged,
    // with no wrapper method of its own.
    assert_eq!(typed.dims(), &[2, 3]);
    assert_eq!(typed.dtype(), DType::F32);
    assert_eq!(typed.to_vec2::<f32>()?, vec![vec![0., 1., 2.], vec![3., 4., 5.]]);
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
