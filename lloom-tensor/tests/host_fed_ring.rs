#![cfg(feature = "metal")]
//! `HostFedRing`: a command stream that waits on the ring's event reads what a
//! host thread wrote into the slot before it signalled — and does not run
//! until then (`design/docs/expert-streaming.md` §20.13).

use lloom_tensor::metal_backend::HostFedRing;
use lloom_tensor::{Device, Result};
use std::time::Duration;

const LEN: usize = 2560;

fn pattern(v: u64) -> Vec<f32> {
    (0..LEN).map(|i| (v as f32) * 0.5 + i as f32).collect()
}

/// A reader encoded after the wait sees the host's write, value after value,
/// through both slots; the stream holds at the wait until the signal (the
/// reader's result cannot be the zeros the slot started with, and the host
/// writes only after the reader was encoded and committed).
#[test]
fn a_waiting_reader_sees_the_slot_the_host_filled_before_signalling() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let ring = std::sync::Arc::new(HostFedRing::new(m, 2, LEN)?);
    let base = ring.base();
    for k in 1..=6u64 {
        let v = base + k;
        ring.encode_wait(v)?;
        // A kernel reading the slot, encoded after the wait, into a buffer
        // of its own.
        let out = ring.tensor(v).affine(1.0, 0.0)?;
        // Commit what is encoded; nothing may run past the wait yet.
        m.flush_and_prune()?;
        let filler = std::sync::Arc::clone(&ring);
        let host = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            // SAFETY: the reader of `v - 2` completed (its `to_vec1` below
            // returned before this value was encoded); this thread is the
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
    Ok(())
}

/// `release` lets every wait of a run through: a stream parked on a value no
/// host will fill completes once the ring is released.
#[test]
fn release_lets_a_parked_stream_through() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let ring = std::sync::Arc::new(HostFedRing::new(m, 2, LEN)?);
    let base = ring.base();
    ring.encode_wait(base + 3)?;
    let out = ring.tensor(base + 3).affine(1.0, 0.0)?;
    m.flush_and_prune()?;
    let releaser = std::sync::Arc::clone(&ring);
    let host = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        releaser.release(base);
    });
    let got = out.to_vec1::<f32>()?;
    host.join().expect("releaser");
    assert_eq!(got.len(), LEN);
    assert!(
        ring.base() > base + 3,
        "the event did not move past the parked value"
    );
    Ok(())
}
