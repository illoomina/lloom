//! The staged readbacks' contract (`metal_backend::StagedReadback`,
//! `StagedMetalRead`): a typed read must name the staged dtype's element and
//! is refused before any wait or byte read otherwise; a deferred stage reads
//! only bytes its own recording wrote, with no wait and no cut once a drain
//! has completed it, and an error when that recording never ran.
#![cfg(feature = "metal")]
use half::{bf16, f16};
use lloom_tensor::metal_backend::{
    readback_counts, stage_readback, stage_readback_deferred, stage_readback_into, HostEvent,
    ReadbackElement,
};
use lloom_tensor::{DType, Device, MetalDevice, StagedRead, Tensor};

fn metal() -> (Device, MetalDevice) {
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(m) = &dev else {
        unreachable!("new_metal")
    };
    let m = m.clone();
    (dev, m)
}

fn values<T: ReadbackElement>(n: usize) -> Vec<T> {
    (0..n).map(|i| T::from_f64((i % 97) as f64 + 1.0)).collect()
}

/// Every route returns exactly the staged values, for each element type.
fn every_route_reads_back<T: ReadbackElement + std::fmt::Debug>(dev: &Device, m: &MetalDevice) {
    let n = 1031;
    let want = values::<T>(n);
    let t = Tensor::from_vec(want.clone(), n, dev).unwrap();
    assert_eq!(t.dtype(), T::DTYPE);
    let got: Vec<T> = stage_readback(&t).unwrap().finish().unwrap();
    assert_eq!(got, want, "{:?} stage_readback", T::DTYPE);
    let dst = m
        .new_persistent_shared(n * T::DTYPE.size_in_bytes())
        .unwrap();
    let got: Vec<T> = stage_readback_into(&t, &dst).unwrap().finish().unwrap();
    assert_eq!(got, want, "{:?} stage_readback_into", T::DTYPE);
    let st = stage_readback_deferred(&t).unwrap();
    m.flush_and_wait_current().unwrap();
    let got: Vec<T> = st.finish().unwrap();
    assert_eq!(got, want, "{:?} stage_readback_deferred", T::DTYPE);
    let got: Vec<T> = t.stage_read().unwrap().to_vec1().unwrap();
    assert_eq!(got, want, "{:?} stage_read", T::DTYPE);
}

#[test]
fn a_typed_readback_returns_the_staged_values_for_every_element() {
    let (dev, m) = metal();
    every_route_reads_back::<u8>(&dev, &m);
    every_route_reads_back::<u32>(&dev, &m);
    every_route_reads_back::<i64>(&dev, &m);
    every_route_reads_back::<f16>(&dev, &m);
    every_route_reads_back::<bf16>(&dev, &m);
    every_route_reads_back::<f32>(&dev, &m);
}

/// Another element type than the staged dtype's is refused, before the wait
/// and before a byte is read: an F32 stage read as `u32` (same width, so a
/// missing dtype check would return the bits), and a U8 stage read as `u32`
/// (four times the staged bytes, which the extent check refuses on its own).
#[test]
fn a_readback_as_another_element_is_refused_before_it_reads() {
    let (dev, m) = metal();
    let f = Tensor::from_vec(values::<f32>(64), 64, &dev).unwrap();
    let b = Tensor::from_vec(values::<u8>(64), 64, &dev).unwrap();
    m.flush_and_wait_current().unwrap();
    let before = readback_counts();
    for (t, label) in [(&f, "F32"), (&b, "U8")] {
        let refused = stage_readback(t).unwrap().finish::<u32>();
        assert!(
            refused
                .as_ref()
                .is_err_and(|e| e.to_string().contains("asked as U32")),
            "{label} finished as u32: {refused:?}"
        );
        let st = stage_readback_deferred(t).unwrap();
        m.flush_and_wait_current().unwrap();
        assert!(
            st.finish::<u32>().is_err(),
            "{label} deferred finished as u32"
        );
        let StagedRead::Metal(staged) = t.stage_read().unwrap() else {
            unreachable!("a Metal tensor stages on Metal")
        };
        let refused = staged.read::<u32>();
        assert!(
            refused
                .as_ref()
                .is_err_and(|e| e.to_string().contains("asked as U32")),
            "{label} read as u32: {refused:?}"
        );
    }
    let after = readback_counts().since(&before);
    eprintln!("REFUSED {after:?}");
    assert_eq!(after.staged_readback, 0, "a refused finish read bytes");
    assert_eq!(after.staged, 0, "a refused read read bytes");
    assert_eq!(after.staged_wait, 0, "a refused finish waited");
    let _ = DType::U32;
}

/// The deferred route's point: once the caller's own drain has completed the
/// recording, `finish` reads with no wait and no cut — the clock takes no
/// epoch for it.
#[test]
fn a_deferred_stage_after_its_drain_reads_without_a_wait_or_a_cut() {
    let (dev, m) = metal();
    let want = values::<f32>(4096);
    let t = Tensor::from_vec(want.clone(), want.len(), &dev).unwrap();
    let y = t.affine(1.0, 0.0).unwrap();
    let st = stage_readback_deferred(&y).unwrap();
    // The drain: a plain readback of other work, as the host plan's.
    let _drain = y.narrow(0, 0, 1).unwrap().to_vec1::<f32>().unwrap();
    let clock = m.gpu_clock();
    let epoch = clock.current_epoch();
    let open = m.open_dispatches();
    let before = readback_counts();
    let got: Vec<f32> = st.finish().unwrap();
    let after = readback_counts().since(&before);
    eprintln!(
        "DEFERRED_DRAINED {after:?} epoch_before={} epoch_after={}",
        epoch.0,
        clock.current_epoch().0
    );
    assert_eq!(got, want);
    assert_eq!(after.staged_readback, 1);
    assert_eq!(after.staged_wait, 0, "a drained deferred stage waited");
    assert_eq!(after.deferred_undrained, 0);
    assert_eq!(clock.current_epoch(), epoch, "finish cut a command buffer");
    assert_eq!(m.open_dispatches(), open);
}

/// Finished before any drain, a deferred stage waits for its own recording
/// (submitting it) and reads what that recording wrote — never the bytes the
/// destination held before the copy ran.
#[test]
fn a_deferred_stage_finished_before_its_drain_waits_for_its_own_recording() {
    let (dev, m) = metal();
    let want = values::<f32>(4096);
    let t = Tensor::from_vec(want.clone(), want.len(), &dev).unwrap();
    m.flush_and_wait_current().unwrap();
    // A computed source: its bytes exist only once the recording runs.
    let y = t.affine(2.0, 1.0).unwrap();
    let want: Vec<f32> = want.iter().map(|v| 2.0 * v + 1.0).collect();
    let st = stage_readback_deferred(&y).unwrap();
    let before = readback_counts();
    let got: Vec<f32> = st.finish().unwrap();
    let after = readback_counts().since(&before);
    eprintln!("DEFERRED_UNDRAINED {after:?} first={:?}", &got[..3]);
    assert_eq!(got, want, "a deferred stage read before its copy ran");
    assert_eq!(after.deferred_undrained, 1);
    assert_eq!(after.staged_wait, 1);
}

/// A deferred stage whose recording was cancelled before it ran returns the
/// cancellation's error, not the destination's bytes; the stream goes on.
#[test]
fn a_deferred_stage_whose_recording_was_cancelled_fails_with_its_reason() {
    let (dev, m) = metal();
    if m.commands().host_input_arm() != lloom_tensor::kernels::metal::HostInputArm::Commit {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let t = Tensor::from_vec(values::<f32>(256), 256, &dev).unwrap();
    m.flush_and_wait_current().unwrap();
    let event = HostEvent::new(&m).unwrap();
    event.commit_when_produced(1).unwrap();
    let st = stage_readback_deferred(&t).unwrap();
    let cancelled = event
        .cancel_through(1, "test: the producer gave up")
        .unwrap();
    assert!(cancelled >= 1, "the gated recording was not cancelled");
    let got = st.finish::<f32>();
    eprintln!("DEFERRED_CANCELLED {got:?}");
    assert!(
        got.as_ref()
            .is_err_and(|e| e.to_string().contains("the producer gave up")),
        "{got:?}"
    );
    let y = t.affine(1.0, 1.0).unwrap();
    let back = y.to_vec1::<f32>().unwrap();
    assert_eq!(back[0], values::<f32>(1)[0] + 1.0);
}
