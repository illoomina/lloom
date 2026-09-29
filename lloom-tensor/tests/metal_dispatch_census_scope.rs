//! A device's dispatch census (`profile::DeviceCensus`) counts that device's
//! dispatches and no other's: two windows held open at once on two devices,
//! on two threads, each read exactly their own work, while the process-wide
//! census over the same span reads both.
//!
//! Its own binary: the profiler's switch is read once per process.
#![cfg(feature = "metal")]

use std::sync::Barrier;

use lloom_tensor::kernels::metal::profile::{self, CensusRefused};
use lloom_tensor::test_utils::Rendezvous;
use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Option<Device> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    Device::new_metal(0).ok()
}

fn count(census: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    census
        .iter()
        .filter(|(l, _)| pred(l))
        .map(|(_, n)| *n)
        .sum()
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
) -> Result<Vec<(String, u64)>> {
    let mut meet = Rendezvous::new(both, 2);
    let dev = device().expect("a Metal device");
    let mut x = Tensor::ones((1, 256), DType::F32, &dev)?;
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("a window on a fresh device");
    meet.meet();
    for _ in 0..n {
        x = op(&x)?;
    }
    dev.synchronize()?;
    meet.meet();
    let census = window.close();
    x.to_vec2::<f32>()?;
    Ok(census)
}

#[test]
fn two_windows_open_at_once_on_two_devices_each_count_only_their_own() -> Result<()> {
    let Some(_probe) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    assert!(profile::enabled(), "the switch did not reach the profiler");
    const EXPS: u64 = 7;
    const SQRTS: u64 = 11;
    let is_exp = |l: &str| l.starts_with("exp_");
    let is_sqrt = |l: &str| l.starts_with("sqrt_");

    let before = profile::snapshot().by_label;
    let both = Barrier::new(2);
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| windowed(&both, EXPS as usize, |t| t.exp()));
        let b = s.spawn(|| windowed(&both, SQRTS as usize, |t| t.sqrt()));
        (a.join().unwrap(), b.join().unwrap())
    });
    let (a, b) = (a?, b?);
    let after = profile::snapshot().by_label;
    eprintln!("device A: {a:?}\ndevice B: {b:?}");

    // The control: the process-wide census over the same span saw both
    // threads' dispatches, so a window reading it would have too.
    let delta = |pred: &dyn Fn(&str) -> bool| count(&after, pred) - count(&before, pred);
    assert!(
        delta(&is_exp) >= EXPS && delta(&is_sqrt) >= SQRTS,
        "the process-wide census must see both devices' work: exp +{}, sqrt +{}",
        delta(&is_exp),
        delta(&is_sqrt)
    );

    // Each window: its own kernel, its own count, nothing else.
    assert_eq!(count(&a, is_exp), EXPS, "device A's exps: {a:?}");
    assert_eq!(count(&b, is_sqrt), SQRTS, "device B's sqrts: {b:?}");
    assert_eq!(a.len(), 1, "device A counted another label: {a:?}");
    assert_eq!(b.len(), 1, "device B counted another label: {b:?}");
    Ok(())
}

#[test]
fn a_second_window_on_one_device_is_refused_until_the_first_closes() -> Result<()> {
    let Some(dev) = device() else {
        eprintln!("no Metal device; skipping");
        return Ok(());
    };
    let census = dev.as_metal_device()?.dispatch_census();
    let x = Tensor::ones((1, 256), DType::F32, &dev)?;
    dev.synchronize()?;

    let first = census.open().expect("the first window");
    assert_eq!(census.open().err(), Some(CensusRefused::WindowOpen));
    x.exp()?;
    assert_eq!(count(&first.close(), |l| l.starts_with("exp_")), 1);

    // Closed, the census counts nothing; a new window starts from zero.
    x.exp()?;
    let second = census.open().expect("a window after the first closed");
    x.exp()?;
    assert_eq!(count(&second.close(), |l| l.starts_with("exp_")), 1);
    Ok(())
}
