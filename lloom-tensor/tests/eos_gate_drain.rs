//! The EOS gate's lifecycle across runs (#993; the maintainer's review,
//! "verify that old gated work cannot observe a reset/reused terminal
//! flag"). The flag is one per device and the next run's
//! `eos_gate_begin` clears it with a host store. So a run that ended early
//! (an error out of the decode loop, the driver's guard disarming the gate)
//! must not leave gated work queued behind it: that work would run after
//! the clear, read a flag of 0 and take its plain path, writing past the
//! EOS into state the next run reads. `eos_gate_end` drains first.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};

/// The first run's gate fires on the device and completes; a forward
/// queued after it (a strided copy, the `copy2d` kernel a KV append
/// dispatches, which carries the gate) has not run when the run ends early
/// and the gate is armed again. The copy must run under the first run's
/// raised flag and keep its destination's poison, not under the next run's
/// cleared one. (A contiguous copy is a blit and carries no gate at all.)
#[test]
fn a_run_that_ends_early_leaves_no_gated_work_for_the_next_runs_cleared_flag() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let eos = Tensor::new(&[42u32], &dev)?;
    let token = Tensor::new(&[42u32], &dev)?;
    let src = Tensor::ones((4, 16), DType::F32, &dev)?;
    let dst = Tensor::full(-1f32, (4, 32), &dev)?;
    dev.synchronize()?;

    dev.eos_gate_begin()?;
    // The sampled token is the stop id: the gate kernel raises the flag, and
    // its command buffer completes.
    dev.eos_gate_step(&token, &eos)?;
    dev.synchronize()?;
    // A forward past the EOS, queued and not yet run: a strided copy into
    // the destination's first 16 columns.
    dst.slice_set(&src, 1, 0)?;
    // The run ends early: its guard disarms the gate, no drain of its own.
    let report = dev.eos_gate_end();
    // The next run arms the gate: its host store clears the flag.
    dev.eos_gate_begin()?;
    dev.synchronize()?;
    let _ = dev.eos_gate_end();

    let got: Vec<f32> = dst.flatten_all()?.to_vec1()?;
    assert!(
        got.iter().all(|&x| x == -1.0),
        "the first run's gated copy ran under the next run's cleared flag and wrote: {:?}",
        &got[..4]
    );
    assert!(
        report.is_some_and(|r| r.fired),
        "the first run's gate fired: {report:?}"
    );
    Ok(())
}
