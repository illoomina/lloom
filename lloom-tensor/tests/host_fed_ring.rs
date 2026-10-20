#![cfg(feature = "metal")]
//! `HostFedRing`: the commands that read a slot are committed once a host
//! thread has written the slot and signalled — and do not run before
//! (`design/docs/expert-streaming.md` §20.13, `docs/invariants.md` §15.1q).

use lloom_tensor::kernels::metal::HostInputArm;
use lloom_tensor::metal_backend::HostFedRing;
use lloom_tensor::{Device, Result};
use std::time::Duration;

const LEN: usize = 2560;

fn pattern(v: u64) -> Vec<f32> {
    (0..LEN).map(|i| (v as f32) * 0.5 + i as f32).collect()
}

/// A reader recorded after `commit_when_produced` sees the host's write,
/// value after value, through both slots: the reader's result cannot be the
/// zeros the slot started with, and the host writes only after the reader
/// was recorded and cut. On the commit arm the reader is held, uncommitted,
/// until the signal.
#[test]
fn a_held_reader_sees_the_slot_the_host_filled_before_signalling() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let commit_arm = m.commands().host_input_arm() == HostInputArm::Commit;
    let ring = std::sync::Arc::new(HostFedRing::new(m, 2, LEN)?);
    let base = ring.base();
    for k in 1..=6u64 {
        let v = base + k;
        ring.commit_when_produced(v)?;
        // A kernel reading the slot, recorded after the gate, into a buffer
        // of its own.
        let out = ring.tensor(v).affine(1.0, 0.0)?;
        // Cut what is recorded; nothing behind the gate may be committed yet.
        m.flush_and_prune()?;
        if commit_arm {
            assert!(
                m.commands().produced_stats().held_now > 0,
                "value {k}: the reader was committed before its input was produced"
            );
        }
        let filler = std::sync::Arc::clone(&ring);
        let host = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            // SAFETY: the reader of `v - 2` completed (its `to_vec1` below
            // returned before this value was recorded); this thread is the
            // only writer.
            unsafe { filler.slot_mut(v) }.copy_from_slice(&pattern(v));
            filler.signal(v);
        });
        let got = out.to_vec1::<f32>()?;
        host.join().expect("filler");
        assert_eq!(
            got,
            pattern(v),
            "value {k}: the reader did not see the host's write"
        );
    }
    if commit_arm {
        let stats = m.commands().produced_stats();
        assert!(stats.released >= 6, "{stats:?}");
        assert_eq!(stats.held_now, 0, "{stats:?}");
    }
    Ok(())
}

/// `release` ends a run whose producer will fill no more. On the commit arm
/// the held reader is cancelled — it never runs, its output keeps what was
/// there — and the stream stays usable. On the GPU-wait arm the jump lets the
/// parked stream through, as it always did.
#[test]
fn release_cancels_the_held_reader_and_the_stream_goes_on() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let commit_arm = m.commands().host_input_arm() == HostInputArm::Commit;
    let ring = std::sync::Arc::new(HostFedRing::new(m, 2, LEN)?);
    let base = ring.base();
    let sentinel = lloom_tensor::Tensor::full(-7.0f32, LEN, &dev)?;
    let before = sentinel.to_vec1::<f32>()?;
    ring.commit_when_produced(base + 3)?;
    // The reader writes into the sentinel's own storage.
    sentinel.slice_set(&ring.tensor(base + 3).affine(1.0, 1.0)?, 0, 0)?;
    m.flush_and_prune()?;
    let releaser = std::sync::Arc::clone(&ring);
    let host = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        releaser.release(base);
    });
    host.join().expect("releaser");
    assert!(
        ring.base() > base + 3,
        "the event did not move past the parked value"
    );
    let after = sentinel.to_vec1::<f32>()?;
    if commit_arm {
        assert_eq!(after, before, "the cancelled reader ran");
        let stats = m.commands().produced_stats();
        assert!(stats.cancelled >= 1, "{stats:?}");
        assert_eq!(stats.held_now, 0, "{stats:?}");
    } else {
        assert_eq!(after.len(), LEN);
    }
    // The stream goes on: fresh work commits and computes.
    let fresh = lloom_tensor::Tensor::full(2.0f32, LEN, &dev)?.affine(3.0, 1.0)?;
    assert!(fresh.to_vec1::<f32>()?.iter().all(|x| *x == 7.0));
    Ok(())
}
