//! The staged readbacks' contract (`metal_backend::StagedReadback`,
//! `StagedMetalRead`): a typed read must name the staged dtype's element and
//! is refused before any wait or byte read otherwise; a deferred stage reads
//! only bytes its own recording wrote, with no wait and no cut once a drain
//! has completed it, and an error when that recording never ran. A stage
//! into a `ReadbackDst` keeps custody of its copy until a successful finish,
//! and a `ReadbackDstCache` returns its destination on every path.
#![cfg(feature = "metal")]
use half::{bf16, f16};
use lloom_tensor::metal_backend::{
    readback_counts, stage_readback, stage_readback_deferred, stage_readback_into, HostEvent,
    ReadbackDstCache, ReadbackDstStats, ReadbackElement,
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
    let mut dst = m.new_readback_dst(n * T::DTYPE.size_in_bytes()).unwrap();
    let got: Vec<T> = stage_readback_into(&t, &mut dst).unwrap().finish().unwrap();
    assert_eq!(got, want, "{:?} stage_readback_into", T::DTYPE);
    assert!(
        !dst.in_flight(),
        "{:?} a finished stage kept custody",
        T::DTYPE
    );
    let cache = ReadbackDstCache::default();
    let got: Vec<T> = cache.read(&t).unwrap();
    assert_eq!(got, want, "{:?} ReadbackDstCache::read", T::DTYPE);
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

fn u32s(n: usize, base: u32) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

fn stats(created: u64, grown: u64, reused: u64, busy: u64, foreign: u64) -> ReadbackDstStats {
    ReadbackDstStats {
        created,
        grown,
        reused,
        busy,
        foreign,
    }
}

/// A finished stage returns the destination ready; a dropped one and one
/// finished with an error keep custody, and the next stage into it is the
/// ordered kind and reads its own values.
#[test]
fn a_destination_stays_in_flight_until_a_stage_finishes() {
    let (dev, m) = metal();
    let a = Tensor::from_vec(u32s(256, 1000), 256, &dev).unwrap();
    let b = Tensor::from_vec(u32s(256, 5000), 256, &dev).unwrap();
    let c = Tensor::from_vec(u32s(256, 9000), 256, &dev).unwrap();
    m.flush_and_wait_current().unwrap();
    let mut dst = m.new_readback_dst(1024).unwrap();
    assert!(!dst.in_flight());

    let before = readback_counts();
    let got: Vec<u32> = stage_readback_into(&a, &mut dst).unwrap().finish().unwrap();
    assert_eq!(got, u32s(256, 1000));
    assert!(!dst.in_flight(), "a finished stage kept custody");

    // Dropped unfinished.
    drop(stage_readback_into(&b, &mut dst).unwrap());
    assert!(
        dst.in_flight(),
        "a dropped stage returned its destination ready"
    );

    // Finished with an error, before any wait.
    let refused = stage_readback_into(&b, &mut dst).unwrap().finish::<f32>();
    assert!(refused.is_err(), "{refused:?}");
    assert!(
        dst.in_flight(),
        "a failed finish returned its destination ready"
    );

    let got: Vec<u32> = stage_readback_into(&c, &mut dst).unwrap().finish().unwrap();
    let after = readback_counts().since(&before);
    eprintln!("DST_CUSTODY {after:?}");
    assert_eq!(
        got,
        u32s(256, 9000),
        "the ordered stage read another copy's bytes"
    );
    assert!(!dst.in_flight());
    assert_eq!(after.dst_leased, 4, "a, b dropped, b refused, c");
    assert_eq!(
        after.dst_ordered_reuse, 2,
        "the two stages after an unfinished one"
    );
    assert_eq!(
        after.staging_allocs, 0,
        "a stage into the destination allocated"
    );
}

/// The ordering an in-flight destination is reused under: the first copy is
/// held behind a host event and dropped unfinished, a second is staged into
/// the same destination, and once the event is raised the second's finish
/// reads the second's values, never the first's.
#[test]
fn a_copy_into_an_in_flight_destination_lands_after_the_pending_one() {
    let (dev, m) = metal();
    let first = Tensor::from_vec(u32s(4096, 1), 4096, &dev).unwrap();
    let second = Tensor::from_vec(u32s(4096, 70_000), 4096, &dev).unwrap();
    m.flush_and_wait_current().unwrap();
    let mut dst = m.new_readback_dst(4096 * 4).unwrap();
    let event = HostEvent::new(&m).unwrap();
    event.commit_when_produced(1).unwrap();
    let before = readback_counts();
    let held = stage_readback_into(&first, &mut dst).unwrap();
    assert!(
        !held.is_complete(),
        "the first copy ran: the control is vacuous"
    );
    drop(held);
    assert!(dst.in_flight());
    let st = stage_readback_into(&second, &mut dst).unwrap();
    assert!(!st.is_complete(), "the second copy ran before the event");
    event.signal(1);
    let got: Vec<u32> = st.finish().unwrap();
    let after = readback_counts().since(&before);
    eprintln!(
        "DST_ORDERED arm={:?} {after:?} first={:?}",
        m.commands().host_input_arm(),
        &got[..2]
    );
    assert_eq!(
        got,
        u32s(4096, 70_000),
        "the pending copy landed after the new one"
    );
    assert_eq!(after.dst_ordered_reuse, 1);
    assert!(!dst.in_flight());
}

/// A tensor on another queue is refused, whatever the custody, and the
/// destination is unchanged by the refusal.
#[test]
fn a_tensor_on_another_queue_is_refused() {
    let (dev, m) = metal();
    let (dev2, m2) = metal();
    assert_ne!(m.id(), m2.id(), "two devices share one queue");
    let mine = Tensor::from_vec(u32s(64, 3), 64, &dev).unwrap();
    let theirs = Tensor::from_vec(u32s(64, 300), 64, &dev2).unwrap();
    let mut dst = m.new_readback_dst(256).unwrap();
    let refused = stage_readback_into(&theirs, &mut dst).map(|_| ());
    eprintln!("DST_FOREIGN {refused:?}");
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.to_string().contains("another queue")),
        "{refused:?}"
    );
    assert!(!dst.in_flight(), "a refused stage took custody");
    let got: Vec<u32> = stage_readback_into(&mine, &mut dst)
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(got, u32s(64, 3));
}

/// The cache, through one sequence of requests: cold creation, reuse, a stage
/// error and a finish error (each returning the destination, the next
/// request reusing it), growth, a busy request (a lease held open) and a
/// request from another queue (each served by a buffer of its own, the
/// cached destination kept), and an unwind with a lease out.
/// Past any page rounding of the first destination, so the cache must grow.
const LARGE: usize = 65536;

#[test]
fn the_cache_returns_its_destination_on_every_path() {
    let (dev, m) = metal();
    let (dev2, _m2) = metal();
    let small = Tensor::from_vec(u32s(64, 10), 64, &dev).unwrap();
    let small2 = Tensor::from_vec(u32s(64, 20), 64, &dev).unwrap();
    let large = Tensor::from_vec(u32s(LARGE, 30), LARGE, &dev).unwrap();
    let skewed = Tensor::from_vec(u32s(64, 40), (8, 8), &dev)
        .unwrap()
        .t()
        .unwrap();
    let theirs = Tensor::from_vec(u32s(64, 50), 64, &dev2).unwrap();
    m.flush_and_wait_current().unwrap();
    let cache = ReadbackDstCache::default();
    let allocs = |before: &lloom_tensor::metal_backend::ReadbackCounts| {
        readback_counts().since(before).staging_allocs
    };

    // Cold: the first destination is made and cached.
    let before = readback_counts();
    assert_eq!(cache.read::<u32>(&small).unwrap(), u32s(64, 10));
    assert_eq!(cache.stats(), stats(1, 0, 0, 0, 0));
    // Warm: the same destination.
    assert_eq!(cache.read::<u32>(&small2).unwrap(), u32s(64, 20));
    assert_eq!(cache.stats(), stats(1, 0, 1, 0, 0));
    assert_eq!(
        allocs(&before),
        0,
        "a cached read allocated a staging buffer"
    );

    // A stage error (a non-contiguous tensor), then the next request.
    assert!(!skewed.is_contiguous());
    let refused = cache.read::<u32>(&skewed);
    assert!(refused.is_err(), "{refused:?}");
    assert_eq!(cache.read::<u32>(&small).unwrap(), u32s(64, 10));
    assert_eq!(
        cache.stats(),
        stats(1, 0, 3, 0, 0),
        "a stage error lost the destination"
    );

    // A finish error (another element), then the next request.
    let before = readback_counts();
    let refused = cache.read::<f32>(&small2);
    assert!(refused.is_err(), "{refused:?}");
    assert_eq!(cache.read::<u32>(&small2).unwrap(), u32s(64, 20));
    let after = readback_counts().since(&before);
    assert_eq!(
        cache.stats(),
        stats(1, 0, 5, 0, 0),
        "a finish error lost the destination"
    );
    assert_eq!(
        after.dst_ordered_reuse, 1,
        "the failed finish's copy was not in flight"
    );
    assert_eq!(after.staging_allocs, 0);

    // Growth.
    assert_eq!(cache.read::<u32>(&large).unwrap(), u32s(LARGE, 30));
    assert_eq!(cache.stats(), stats(1, 1, 5, 0, 0));
    assert_eq!(cache.read::<u32>(&small).unwrap(), u32s(64, 10));
    assert_eq!(
        cache.stats(),
        stats(1, 1, 6, 0, 0),
        "a smaller read after growth"
    );

    // Busy: a lease held open; the request gets a buffer of its own, and the
    // cached destination is the one returned, not replaced.
    let before = readback_counts();
    {
        let mut lease = cache.lease(&m, 256).unwrap().expect("an idle cache lends");
        assert!(lease.dst_mut().capacity() >= LARGE * 4);
        assert_eq!(cache.read::<u32>(&small2).unwrap(), u32s(64, 20));
        assert!(cache.lease(&m, 256).unwrap().is_none(), "two leases out");
    }
    assert_eq!(
        allocs(&before),
        1,
        "the busy request did not stage on its own"
    );
    assert_eq!(cache.stats(), stats(1, 1, 7, 2, 0));
    assert_eq!(cache.read::<u32>(&large).unwrap(), u32s(LARGE, 30));
    assert_eq!(
        cache.stats(),
        stats(1, 1, 8, 2, 0),
        "the busy fallback replaced the cache"
    );

    // Another queue: served by a buffer of its own; the cache is kept.
    assert_eq!(cache.read::<u32>(&theirs).unwrap(), u32s(64, 50));
    assert_eq!(cache.stats(), stats(1, 1, 8, 2, 1));
    assert_eq!(cache.read::<u32>(&small).unwrap(), u32s(64, 10));
    assert_eq!(cache.stats(), stats(1, 1, 9, 2, 1));

    // An unwind with a lease out returns it.
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lease = cache.lease(&m, 256).unwrap();
        panic!("test: unwinding with a lease out");
    }));
    assert!(unwound.is_err());
    assert_eq!(cache.read::<u32>(&small2).unwrap(), u32s(64, 20));
    let end = cache.stats();
    eprintln!("CACHE_PATHS {end:?}");
    assert_eq!(
        end,
        stats(1, 1, 11, 2, 1),
        "the unwind lost the destination"
    );
}
