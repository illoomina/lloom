//! One Metal device shared by several threads: a thread that waits for the
//! device (`synchronize`) must not strip the ordering another thread's
//! still-running work depends on.
//!
//! Every worker runs a chain of dependent dispatches (`x + 1`, `K` times),
//! long enough to cross several command-buffer cuts, so each link after a cut
//! reads a value an earlier encoder wrote. The buffers are untracked
//! (`HazardTrackingModeUntracked`): only the host's cross-encoder fence map
//! orders those reads, and a reader that finds no entry for its input does
//! not wait. The syncer thread calls `synchronize` in a loop.
#![cfg(feature = "metal")]

use lloom_tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Elements per link: enough work per dispatch that consecutive encoders
/// overlap on the GPU when nothing orders them.
const N: usize = 1 << 20;
/// Links per chain: several cuts at the default period of 50.
const K: usize = 160;
const WORKERS: usize = 3;
const REPS: usize = 40;

/// Wrong elements of one chain: `x0 + K` is exact in f32 for these values.
fn chain(dev: &Device, seed: usize) -> Result<usize> {
    let base = (seed % 1000) as f32;
    let mut x = Tensor::full(base, N, dev)?;
    for _ in 0..K {
        x = x.affine(1.0, 1.0)?;
    }
    let got = x.to_vec1::<f32>()?;
    let want = base + K as f32;
    Ok(got.iter().filter(|&&v| v != want).count())
}

/// What the other thread does to the device in a loop.
#[derive(Clone, Copy, Debug)]
enum Wait {
    /// `Device::synchronize`: commit, wait for everything committed.
    Synchronize,
    /// Commit and wait for the buffer just committed (a readback's wait).
    Current,
    /// Commit without waiting.
    Prune,
}

fn wait(dev: &Device, how: Wait) -> Result<()> {
    let Device::Metal(m) = dev else {
        unreachable!("a Metal device")
    };
    match how {
        Wait::Synchronize => dev.synchronize(),
        Wait::Current => m.flush_and_wait_current(),
        Wait::Prune => m.flush_and_prune().map(|_| ()),
    }
}

/// `(chains with a wrong element, chains run)` over `WORKERS` threads, each
/// on `device(worker)`, with a thread waiting on `sync_on` in a loop while
/// they run, or none.
fn run(
    device: impl Fn(usize) -> Device,
    sync_on: Option<(Device, Wait)>,
) -> Result<(usize, usize)> {
    let stop = Arc::new(AtomicBool::new(false));
    let syncer = sync_on.map(|(dev, how)| {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || -> Result<u64> {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                wait(&dev, how)?;
                n += 1;
            }
            Ok(n)
        })
    });
    let workers: Vec<_> = (0..WORKERS)
        .map(|w| {
            let dev = device(w);
            std::thread::spawn(move || -> Result<usize> {
                let mut bad = 0;
                for rep in 0..REPS {
                    if chain(&dev, w * REPS + rep)? > 0 {
                        bad += 1;
                    }
                }
                Ok(bad)
            })
        })
        .collect();
    let mut bad = 0;
    for w in workers {
        bad += w.join().expect("worker panicked")?;
    }
    stop.store(true, Ordering::Relaxed);
    if let Some(s) = syncer {
        let syncs = s.join().expect("syncer panicked")?;
        assert!(syncs > 0, "the syncer never ran: the arm measured nothing");
        eprintln!("syncs={syncs}");
    }
    Ok((bad, WORKERS * REPS))
}

/// The control: the chains alone on one shared device compute `x0 + K`.
#[test]
fn chains_on_a_shared_device_are_exact_without_a_syncer() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (bad, total) = run(|_| dev.clone(), None)?;
    eprintln!("shared device, no syncer: {bad} of {total} chains wrong");
    assert_eq!(bad, 0, "{bad} of {total} chains wrong with no syncer");
    Ok(())
}

/// The defect's arm: a `synchronize` on another thread of the same device.
#[test]
fn a_synchronize_on_another_thread_keeps_the_chains_exact() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (bad, total) = run(|_| dev.clone(), Some((dev.clone(), Wait::Synchronize)))?;
    eprintln!("shared device, syncer on it: {bad} of {total} chains wrong");
    assert_eq!(
        bad, 0,
        "{bad} of {total} chains wrong while another thread synchronized the device"
    );
    Ok(())
}

/// A readback's wait on another thread: commits and waits as `synchronize`
/// does, and leaves the fence map alone.
#[test]
fn a_readback_wait_on_another_thread_keeps_the_chains_exact() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (bad, total) = run(|_| dev.clone(), Some((dev.clone(), Wait::Current)))?;
    eprintln!("shared device, readback waits on it: {bad} of {total} chains wrong");
    assert_eq!(
        bad, 0,
        "{bad} of {total} chains wrong under another thread's readback waits"
    );
    Ok(())
}

/// Commits at arbitrary points on another thread, with no wait.
#[test]
fn a_commit_on_another_thread_keeps_the_chains_exact() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (bad, total) = run(|_| dev.clone(), Some((dev.clone(), Wait::Prune)))?;
    eprintln!("shared device, commits on it: {bad} of {total} chains wrong");
    assert_eq!(
        bad, 0,
        "{bad} of {total} chains wrong under another thread's commits"
    );
    Ok(())
}
