//! The two-weight kernel's operand is the gate WEIGHT, `[n, k]` like the up
//! weight — and the host checks it (#732, `nn::ops::gemv_fused_metal`): a
//! gate of another shape is refused by name, never swept as if it were
//! `[n, k]`. Under `HazardTrackingModeUntracked` a wrong-shaped sweep is a
//! well-formed wrong number, which nothing reports (§3.5); this test is the
//! only thing that can fail when that check is removed.

#![cfg(feature = "metal")]

use lloom::nn::ops::gemv_swiglu;
use lloom_tensor::{DType, Device, Result, Tensor};

#[test]
fn a_gate_weight_of_another_shape_is_refused_and_the_right_one_runs() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let (n, k) = (512usize, 2048usize);
    let x = Tensor::randn(0f32, 1.0, (1, 1, k), &dev)?.to_dtype(DType::F16)?;
    let up = Tensor::randn(0f32, 0.05, (n, k), &dev)?.to_dtype(DType::F16)?;
    let gate = Tensor::randn(0f32, 0.05, (n, k), &dev)?.to_dtype(DType::F16)?;
    // The right shape runs and produces the output extent.
    let y = gemv_swiglu(&x, &up, &gate)?;
    assert_eq!(y.dims(), &[1, 1, n]);
    // Same element count, another shape: refused by name, not swept.
    let wrong = gate.reshape((n / 2, k * 2))?.contiguous()?;
    let err = match gemv_swiglu(&x, &up, &wrong) {
        Ok(_) => panic!(
            "a [{}, {}] gate must be refused against a [{n}, {k}] up weight",
            n / 2,
            k * 2
        ),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("gemv-swiglu"), "{err}");
    eprintln!("{err}");
    // Another element count is refused by the extent check that predates it.
    let short = gate.narrow(0, 0, n / 2)?.contiguous()?;
    assert!(gemv_swiglu(&x, &up, &short).is_err());
    Ok(())
}
