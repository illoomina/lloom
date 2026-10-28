//! A device's dispatch trace (`trace::DeviceTrace`) records that device's
//! dispatches and no other's: two windows held open at once on two devices,
//! on two threads, each read exactly their own work -- kernel, count and
//! bindings -- while the process-wide trace over the same span reads both,
//! and names some of one thread's dispatches after the other's kernel.
//!
//! Its own binary: the trace switch is read once per process.
#![cfg(feature = "metal")]

use std::sync::Barrier;

use lloom_tensor::kernels::metal::trace::{self, TraceRefused};
use lloom_tensor::test_utils::Rendezvous;
use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Option<Device> {
    std::env::set_var("CANDLE_METAL_TRACE", "1");
    Device::new_metal(0).ok()
}

fn count(ds: &[trace::Dispatch], prefix: &str) -> usize {
    ds.iter().filter(|d| d.pipeline.starts_with(prefix)).count()
}

/// Open a window on a fresh device, wait for the other thread to open its
/// own, dispatch `op` `n` times, wait for the other thread to finish, close.
/// Both windows are therefore open across both threads' dispatches; a thread
/// that fails still keeps both meetings, so the other one reports rather
/// than hangs.
fn windowed(
    both: &Barrier,
    n: usize,
    op: fn(&Tensor) -> lloom_tensor::Result<Tensor>,
) -> Result<Vec<trace::Dispatch>> {
    let mut meet = Rendezvous::new(both, 2);
    let dev = device().expect("a Metal device");
    let mut x = Tensor::ones((1, 256), DType::F32, &dev)?;
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_trace()
        .open()
        .expect("a window on a fresh device");
    meet.meet();
    for _ in 0..n {
        x = op(&x)?;
    }
    meet.meet();
    let ds = window.close();
    x.to_vec2::<f32>()?;
    Ok(ds)
}

#[test]
fn two_windows_open_at_once_on_two_devices_each_record_only_their_own() -> Result<()> {
    let Some(_probe) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    assert!(
        trace::trace_requested(),
        "the switch did not reach the trace"
    );
    const EXPS: usize = 7;
    const SQRTS: usize = 11;

    let _ = trace::take_dispatches();
    trace::set_recording(true)?;
    let both = Barrier::new(2);
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| windowed(&both, EXPS, |t| t.exp()));
        let b = s.spawn(|| windowed(&both, SQRTS, |t| t.sqrt()));
        (a.join().unwrap(), b.join().unwrap())
    });
    trace::set_recording(false)?;
    let global = trace::take_dispatches();
    let (a, b) = (a?, b?);
    let names = |ds: &[trace::Dispatch]| ds.iter().map(|d| d.pipeline.clone()).collect::<Vec<_>>();
    eprintln!("device A: {:?}\ndevice B: {:?}", names(&a), names(&b));

    // The control: the process-wide trace over the same span recorded both
    // threads' dispatches, so a window reading it would have too. Only the
    // total is asserted: its one pipeline slot is shared by the threads, so
    // how the total splits between `exp` and `sqrt` is a race (printed).
    let (g_exp, g_sqrt) = (count(&global, "exp_"), count(&global, "sqrt_"));
    eprintln!("process-wide: exp {g_exp}, sqrt {g_sqrt} (the windows: {EXPS}, {SQRTS})");
    assert!(
        g_exp + g_sqrt >= EXPS + SQRTS,
        "the process-wide trace must see both devices' work: exp {g_exp}, sqrt {g_sqrt}"
    );

    // Each window: its own kernel, its own count, nothing else -- and every
    // dispatch carries the one input and one output its kernel bound, so no
    // binding of the other device's dispatch was attributed to it.
    for (what, ds, prefix, n) in [("A", &a, "exp_", EXPS), ("B", &b, "sqrt_", SQRTS)] {
        assert_eq!(count(ds, prefix), n, "device {what}: {:?}", names(ds));
        assert_eq!(
            ds.len(),
            n,
            "device {what} recorded another kernel: {:?}",
            names(ds)
        );
        for d in ds.iter() {
            let outs = d.bindings.iter().filter(|b| b.is_output).count();
            let ins = d.bindings.iter().filter(|b| !b.is_output).count();
            assert_eq!(
                (ins, outs),
                (1, 1),
                "device {what}: {} bound {ins} inputs and {outs} outputs",
                d.pipeline
            );
        }
    }
    Ok(())
}

#[test]
fn a_second_window_on_one_device_is_refused_until_the_first_closes() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let tr = dev.as_metal_device()?.dispatch_trace();
    let x = Tensor::ones((1, 256), DType::F32, &dev)?;
    dev.synchronize()?;

    let first = tr.open().expect("the first window");
    assert_eq!(tr.open().err(), Some(TraceRefused::WindowOpen));
    x.exp()?;
    assert_eq!(count(&first.close(), "exp_"), 1);

    // Closed, the trace records nothing; a new window starts empty.
    x.exp()?;
    let second = tr.open().expect("a window after the first closed");
    x.exp()?;
    assert_eq!(count(&second.close(), "exp_"), 1);
    Ok(())
}
