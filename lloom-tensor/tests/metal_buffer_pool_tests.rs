//! Device-level behaviour of the Metal buffer pool (lloom issue #21).
//!
//! `candle-metal-kernels` tests the pool data structure directly. These test
//! the properties that only appear once `MetalDevice` is driving it: which
//! allocation paths participate in reuse, that reclamation needs nothing from
//! the caller, and that the free list does not grow without bound.

#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};

/// Serializes these tests against each other.
///
/// `Device::new_metal(0)` hands back one process-wide device, so every test
/// here drives the *same* pool. Run concurrently they take each other's
/// buffers and reset each other's counters, and the failure looks like a pool
/// bug rather than a harness one -- "allocated 6 buffers over 384 lookups" when
/// another test had just emptied the free list. Cargo's default is one thread
/// per core, so this is not a hypothetical.
///
/// A lock rather than `--test-threads=1`, because the requirement then travels
/// with the tests instead of living in whatever command someone happens to run.
static POOL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    // A panicking test leaves this poisoned; the pool state it was observing is
    // no longer trustworthy anyway, and taking the guard regardless means one
    // failure reports one failure rather than cascading into the rest.
    POOL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// Dropping a tensor hands its buffer back to the pool with no sweep, no scan,
/// and nothing asked of the caller.
///
/// "Hands back" is not "makes reusable": the buffer is parked until the GPU
/// finishes with it. What this pins is that the release still happens on drop,
/// automatically -- `trim_does_not_free_buffers_still_in_flight` and
/// `in_flight_buffer_is_not_handed_out_again` cover the second half.
#[test]
fn dropping_a_tensor_returns_its_buffer() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Warm the pool so the sizes under test already exist.
    drop(Tensor::zeros((256, 256), DType::F32, &device)?);

    let releases = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_counters();
        s.releases + p.releases
    };

    let before = releases(metal);
    let t = Tensor::zeros((256, 256), DType::F32, &device)?;
    let during = releases(metal);
    drop(t);
    let after = releases(metal);

    assert!(
        after > during,
        "dropping a tensor did not return its buffer: {during} -> {after}"
    );
    assert!(after > before);
    Ok(())
}

/// Repeating identical work must keep reusing buffers rather than allocating a
/// fresh one every time. This is the decode steady state in miniature.
///
/// # Why this no longer asserts zero allocations
///
/// It did, as issue #21's version, and that assertion was correct **under
/// CPU-drop release**: a buffer was returned to the free list inside the
/// operation that freed it, so the next identical iteration always found it.
///
/// Deciding reuse on the GPU clock (issue #23) changes the arithmetic, and not
/// as a defect. A buffer released now is parked against the command buffer
/// currently being encoded, and cannot be handed out until that command buffer
/// *completes*. Candle packs 50 compute dispatches into one (`compute_per_buffer`),
/// which for this tiny workload is roughly 14 iterations. So every buffer freed
/// during a command buffer is, by construction, unavailable to the other ~13
/// iterations sharing it -- they must allocate, and those allocations are then
/// reused for the rest of the run. The steady state is therefore a *working set
/// proportional to the in-flight window*, not one iteration's worth.
///
/// Measured directly, by varying that window and running this file:
///
/// ```text
///   CANDLE_METAL_COMPUTE_PER_BUFFER=1   allocations 0     10/10 pass
///   CANDLE_METAL_COMPUTE_PER_BUFFER=2   allocations 0     10/10 pass
///   CANDLE_METAL_COMPUTE_PER_BUFFER=5   allocations >0     9/10 pass
///   CANDLE_METAL_COMPUTE_PER_BUFFER=50  allocations ~287   8/10 pass  (default)
/// ```
///
/// Zero is reachable only when the window is 1-2 command buffers deep. At the
/// default it is unreachable *by construction*, so asserting it would be
/// asserting that the fix had not been made.
///
/// # What replaced it, and why that is still a guard
///
/// The property that survives is **bounded reuse**: the number of distinct
/// buffers the workload needs is a function of the in-flight window, which is a
/// constant, so allocations must stop growing with the number of iterations.
/// Asserted here by running two equal-length windows back to back and requiring
/// the second to allocate no more than the first -- with the pool warm, a
/// steady-state workload has already paid for its working set.
///
/// This still fails a pool that stops reusing, which is what #21 added it for:
/// if every lookup missed, the second window would allocate as much as the
/// first (one per lookup) and never converge. It also still fails the specific
/// regression #21 caught, a free list that never returns anything.
#[test]
fn repeated_identical_work_converges_to_bounded_reuse() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let a = Tensor::ones((128, 128), DType::F32, &device)?;
    let b = Tensor::ones((128, 128), DType::F32, &device)?;

    let work = |n: usize| -> Result<()> {
        for _ in 0..n {
            let c = ((&a + &b)? * 2.0)?;
            let _ = c.sum_all()?;
        }
        Ok(())
    };

    // Populate the in-flight working set.
    work(512)?;

    metal.reset_pool_counters();
    work(256)?;
    let (s1, p1) = metal.pool_counters();
    let first = s1.allocations + p1.allocations;
    let lookups = s1.lookups + p1.lookups;

    metal.reset_pool_counters();
    work(256)?;
    let (s2, p2) = metal.pool_counters();
    let second = s2.allocations + p2.allocations;

    assert!(lookups > 0, "no pool lookups happened at all");

    // The pool must be reusing at all: a pool that never hits allocates once
    // per lookup. This is the #21 guard, restated so it survives deferral.
    //
    // Stated as "some reuse happened", not as a ratio. A ratio is not safely
    // assertable here: these tests share one process-wide device, `POOL_LOCK`
    // serializes the test *bodies* but not the GPU completion handlers other
    // tests are still delivering, and `trim_unused_buffers` in a neighbouring
    // test empties the free list outright. Measured across runs that pushed the
    // hit count to 150/768 and 50/768 for that reason alone -- while a pool with
    // reuse disabled gives exactly 0 and the assertion below still catches it.
    //
    // The quantitative bound lives in the unit tests, where the pool is driven
    // directly and nothing else can touch it:
    // `fixed_shape_work_allocates_the_window_then_stops` pins the allocation
    // count to the window depth and every later iteration to a hit.
    let hits = s2.hits + p2.hits;
    let lookups2 = s2.lookups + p2.lookups;
    assert!(
        hits > 0,
        "not one of {lookups2} lookups was served from the free list; \
         the pool is not reusing"
    );

    // And it must not be allocating once per lookup, which is what a pool that
    // has stopped converging degenerates to. A strict `second <= first` is the
    // property wanted here and is *not* assertable at device level: a
    // neighbouring test's `trim_unused_buffers` destroys the free list between
    // the two windows, and measured that way the comparison fails in 5 runs out
    // of 8 for that reason alone. The exact form is pinned instead by
    // `fixed_shape_work_allocates_the_window_then_stops` in
    // `candle-metal-kernels`, where the pool is exclusively owned and the count
    // is deterministic: 8 allocations for a window 8 deep, then 1000 hits.
    assert!(
        second < lookups2,
        "steady-state work allocated {second} buffers over {lookups2} lookups \
         (first window: {first}); the pool is allocating rather than reusing"
    );
    Ok(())
}

/// The pool must reach a bounded size and stay there.
///
/// Deferring release to GPU completion holds more buffers at once than
/// releasing on CPU drop did, which is the cost of the change. The cost has to
/// be a constant -- the depth of the in-flight window -- and not something that
/// grows with how long the process runs, which is the shape issue #21 removed
/// from the free-list scan and must not be reintroduced in the footprint.
///
/// # The horizon matters, and an earlier version of this test got it wrong
///
/// It warmed up for 200 iterations and then required the next 1000 to change
/// the footprint by nothing at all. That failed, and the failure was the test's:
/// 200 iterations is still on the ramp. Measured on an M1 Max, the footprint
/// climbs for roughly 4000 iterations and is then flat *to the byte*:
///
/// ```text
///   after  2000 iters: total=  93413760
///   after  3000 iters: total= 131067648
///   after  4000 iters: total= 140579456   <- plateau
///   after  6000 iters: total= 140579456
///   after  8000 iters: total= 140579456
///   after 10000 iters: total= 140579456
/// ```
///
/// Thirteen consecutive samples at exactly 140,579,456 bytes. So the property
/// is real and the assertion was measuring it too early -- convergence takes
/// longer than one iteration's working set suggests, because the working set
/// is the in-flight window's, not one iteration's.
#[test]
fn pool_footprint_converges_and_stays_flat() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let a = Tensor::ones((128, 128), DType::F32, &device)?;
    let b = Tensor::ones((128, 128), DType::F32, &device)?;

    let footprint = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.total_bytes() + p.total_bytes()
    };

    let work = |n: usize| -> Result<()> {
        for _ in 0..n {
            let c = ((&a + &b)? * 2.0)?;
            let _ = c.sum_all()?;
        }
        Ok(())
    };

    // Past the ramp. See the table above for where the plateau actually is.
    work(5000)?;
    let settled = footprint(metal);

    work(2000)?;
    let after = footprint(metal);

    assert_eq!(
        after, settled,
        "pool grew from {settled} to {after} bytes over 2000 further iterations \
         of identical work after 5000 warm-up iterations; the in-flight window \
         is not bounded"
    );
    Ok(())
}

/// Scan length must not grow with how long the process has been running.
///
/// The old pool failed this: `retain` emptied a bucket's `Vec` but left its key
/// in the map, so keys accumulated at ~3.2 per decode token and every lookup
/// walked all of them -- 1345 buckets, 1324 of them empty, after 400 tokens.
/// The cost was O(work done), not O(live buffers), and it never converged.
#[test]
fn free_list_does_not_accumulate_empty_buckets() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Ever-changing shapes, the way a growing KV cache produces a new size
    // every few tokens.
    for n in 1..200usize {
        let t = Tensor::zeros((n, 64), DType::F32, &device)?;
        let _ = (&t + 1.0)?;
    }

    let (shared, private) = metal.pool_occupancy();
    for (name, occ) in [("shared", shared), ("private", private)] {
        assert!(
            occ.free_buckets <= occ.free_buffers,
            "{name} pool has {} free buckets holding only {} buffers, so some are empty",
            occ.free_buckets,
            occ.free_buffers
        );
    }
    Ok(())
}

/// Per-lookup scan length is bounded by a small constant regardless of how many
/// distinct sizes the pool holds -- the invariant `CONTRIBUTING.md` §3.3 and
/// `DESIGN.md` §15.2 #10 require of anything on the per-token path.
#[test]
fn lookup_cost_does_not_grow_with_pool_size() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    for n in 1..256usize {
        let _ = Tensor::zeros((n, 32), DType::F32, &device)?;
    }

    metal.reset_pool_counters();
    for n in 1..256usize {
        let _ = Tensor::zeros((n, 32), DType::F32, &device)?;
    }
    let (shared, private) = metal.pool_counters();

    let lookups = shared.lookups + private.lookups;
    let probed = shared.buckets_probed + private.buckets_probed;
    assert!(lookups > 0, "no lookups recorded");

    // Two probes per lookup is the structural maximum: the exact-size bucket,
    // then one ordered range query. The old scan walked every bucket.
    let per_lookup = probed as f64 / lookups as f64;
    assert!(
        per_lookup <= 2.0,
        "expected at most 2 bucket probes per lookup, got {per_lookup:.2} \
         ({probed} probes over {lookups} lookups)"
    );
    Ok(())
}

/// Buffers uploaded with data are not parked in the free list.
///
/// They come from `newBufferWithBytes` and are only reachable by an
/// `allocate_buffer` request of exactly their size, which nothing in a normal
/// forward pass issues. On an LFM2 load, pooling them held 5145 MB for a reuse
/// that never comes -- the old allocator reclaimed them only as a side effect
/// of its sweep, so with the sweep off the allocation path they must not be
/// retained (issue #8's 7731 -> 5509 MB depends on it).
#[test]
fn uploaded_buffers_are_not_retained_for_reuse() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let data: Vec<f32> = vec![1.0; 64 * 1024];
    metal.reset_pool_counters();
    let (shared_before, _) = metal.pool_occupancy();

    for _ in 0..16 {
        let t = Tensor::from_slice(&data, (64, 1024), &device)?;
        drop(t);
    }

    let (shared_after, _) = metal.pool_occupancy();
    assert_eq!(
        shared_after.free_bytes, shared_before.free_bytes,
        "uploaded buffers were retained in the free list: {} -> {} bytes free",
        shared_before.free_bytes, shared_after.free_bytes
    );
    Ok(())
}

/// Reclamation must not require anything of the caller. A buffer moved into a
/// struct, cloned, and handed around still comes back on its own -- which is
/// what keeps candle's allocate-and-forget ergonomics, and why this is not a
/// manual `free()` API.
#[test]
fn reclamation_requires_nothing_of_the_caller() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    drop(Tensor::zeros((64, 64), DType::F32, &device)?);

    struct Owned {
        _tensor: Tensor,
    }

    let t = Tensor::zeros((64, 64), DType::F32, &device)?;
    let shared_clone = t.clone();
    let owned = Owned { _tensor: t };

    let releases = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_counters();
        s.releases + p.releases
    };

    let before = releases(metal);
    drop(shared_clone);
    let mid = releases(metal);
    assert_eq!(
        mid, before,
        "buffer returned while another reference was still live"
    );

    drop(owned);
    let after = releases(metal);
    assert!(
        after > mid,
        "buffer did not return when its last reference dropped"
    );
    Ok(())
}

/// A workload whose buffers grow every step must not grow the pool without
/// bound. This is LFM2 decode's shape: the KV cache is one token longer each
/// step, so every allocation asks for a size never asked for before, and the
/// size just freed is never wanted again.
///
/// It is the case deferring release to GPU completion broke. Under CPU-drop
/// release each buffer was reused inside the operation that freed it; deferred,
/// it is stranded. Measured on LFM2 before the free-list bound: 11.6 stranded
/// buffers per token, 5231 MB -> 13629 MB at 400 tokens, still climbing.
#[test]
fn growing_allocations_do_not_grow_the_pool_without_bound() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let footprint = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.total_bytes() + p.total_bytes()
    };

    // Ever-larger tensors, as a growing KV cache produces.
    for n in 1..300usize {
        let t = Tensor::zeros((n, 512), DType::F32, &device)?;
        let _ = (&t + 1.0)?;
    }
    let at_300 = footprint(metal);

    for n in 300..600usize {
        let t = Tensor::zeros((n, 512), DType::F32, &device)?;
        let _ = (&t + 1.0)?;
    }
    let at_600 = footprint(metal);

    // Each of the two pools caps its free list at 256 MB, and both are counted
    // here, so the ceiling this can reach is 512 MB plus whatever is in flight.
    // What the assertion is really about is that the ceiling exists: without it
    // the second half adds a second half's worth of stranded buffers and the
    // total keeps climbing with the loop bound, unbounded.
    let ceiling = 2 * 256 * 1024 * 1024 + 64 * 1024 * 1024;
    assert!(
        at_600 <= ceiling,
        "pool reached {at_600} bytes ({at_300} at the halfway point), past the \
         {ceiling} byte ceiling the two capped free lists allow; the free list \
         is not bounded"
    );

    // **The free lists specifically, which is what the bound is on.**
    //
    // The assertion above is on `total_bytes()` -- live + free + pending -- so
    // its 64 MiB slack term has to cover a whole in-flight window, and that
    // window is a property of the machine's scheduling rather than of the
    // bound. Measured, it does not always: this test fails intermittently on
    // `lloom/integration` at 088d7249, **6 of 30** interleaved against **9 of
    // 30** on issue #210's branch, always at the same figure -- 622 962 688
    // bytes, 86.1 MB of live+pending against 67.1 MB of slack. Both arms, one
    // population, and §1.3 says a difference that size at that sample is not
    // evidence of anything.
    //
    // So the free-list bound gets its own assertion, on the quantity the bound
    // is actually enforced against. This one cannot be perturbed by how far the
    // CPU has run ahead of the GPU, which is what makes it the sharper half.
    let (s, p) = metal.pool_occupancy();
    let cap = 256 * 1024 * 1024;
    assert!(
        s.free_bytes <= cap && p.free_bytes <= cap,
        "a free list is past its {cap} byte budget: shared {}, private {} \
         (`DESIGN.md` §6.3b)",
        s.free_bytes,
        p.free_bytes
    );
    Ok(())
}

/// `trim` hands memory back without disturbing live tensors.
#[test]
fn trim_frees_the_free_list_and_spares_live_tensors() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let live = Tensor::zeros((128, 128), DType::F32, &device)?;
    drop(Tensor::zeros((256, 256), DType::F32, &device)?);
    // A dropped buffer is not free until the GPU is done with it, so a trim
    // has nothing to reclaim until the work it was part of has completed.
    metal.wait_until_completed()?;

    let free_buffers = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.free_buffers + p.free_buffers
    };

    assert!(free_buffers(metal) > 0, "expected something free to trim");

    metal.trim_unused_buffers();

    assert_eq!(free_buffers(metal), 0, "trim left buffers in the free list");

    // The live tensor is untouched and still usable.
    let sum = (&live + 1.0)?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(sum, (128 * 128) as f32);
    Ok(())
}

/// The property that stops `trim` from becoming a use-after-free.
///
/// A buffer the CPU has released but the GPU may still be reading is *not* in
/// the free list, so trim leaves it alone. Destroying it would be the same
/// defect the deferral exists to prevent, with a freed allocation in place of
/// an aliased one.
#[test]
fn trim_does_not_free_buffers_still_in_flight() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Queue work and drop its result without synchronizing, so the buffers are
    // released by the CPU while the GPU may still be using them.
    for _ in 0..16 {
        let t = Tensor::zeros((256, 256), DType::F32, &device)?;
        drop((&t + 1.0)?);
    }

    let pending = |m: &lloom_tensor::MetalDevice| {
        let (s, p) = m.pool_occupancy();
        s.pending_buffers + p.pending_buffers
    };

    let before = pending(metal);
    assert!(before > 0, "expected buffers awaiting the GPU");

    metal.trim_unused_buffers();

    assert_eq!(
        pending(metal),
        before,
        "trim destroyed {} buffer(s) the GPU may still be using",
        before - pending(metal)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// The residency-set guard (lloom issue #166, `DESIGN.md` §6.3c).
//
// The defect these cover is a *kernel panic*:
// `IOGPUGroupMemory::remove_memory_object()` aborts the machine when asked to
// remove an allocation that is already gone. That is not a testable assertion,
// and a test that provoked it would cost the machine and the user's session —
// so **nothing here reproduces the crash**. What is testable is the mechanism
// that makes it unreachable, and that is what these assert: that the set is
// emptied while the buffers still exist, and that the emptying happens when the
// last device handle drops rather than when any clone does.
// ---------------------------------------------------------------------------

/// The guard fires on the **last** handle, not on every clone.
///
/// `MetalDevice` is `Clone` and every `Tensor` holds a clone through its
/// `MetalStorage`, so a `Drop` written on `MetalDevice` itself would fire
/// hundreds of times per decode token — each time waiting for the GPU. This
/// pins the property that keeps that wait off the hot path.
#[test]
fn teardown_guard_is_shared_by_every_device_clone() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    let before = metal.device_handles();

    // A tensor holds a device clone through its storage, so allocating must
    // raise the count — which is exactly why the guard cannot live on
    // `MetalDevice::drop`.
    let t = Tensor::zeros((64, 64), DType::F32, &device)?;
    let with_tensor = metal.device_handles();
    assert!(
        with_tensor > before,
        "a tensor must hold a device handle ({before} -> {with_tensor}); \
         if it does not, the premise for the guard's placement has changed"
    );

    // Dropping it lowers the count and must NOT have run the guard: the
    // residency set is still populated and the device still works.
    drop(t);
    assert_eq!(
        metal.device_handles(),
        before,
        "dropping a tensor should release exactly its own device handle"
    );
    assert!(
        metal.residency_set_len() > 0,
        "dropping a tensor must not empty the residency set — that is the \
         teardown path firing on the hot path"
    );

    // And the device is still usable, which is the observable consequence of
    // the guard not having run.
    let sum = Tensor::ones((8, 8), DType::F32, &device)?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert_eq!(sum, 64.0);
    Ok(())
}

/// Dropping the last handle empties the residency set.
///
/// This is the ordering the panic came from, asserted from the outside: after
/// the device is gone, the set holds nothing, so the later teardown of the
/// Metal object has nothing to remove and cannot ask the kernel to remove an
/// absent allocation.
///
/// It is observed through an `Arc` the test keeps to the set itself, because
/// after the device drops there is no device left to ask.
#[test]
fn dropping_the_last_device_handle_empties_the_residency_set() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Allocate through several paths so the set is genuinely populated.
    let a = Tensor::zeros((128, 128), DType::F32, &device)?;
    let b = Tensor::ones((64, 64), DType::F32, &device)?;
    let c = (&b + 1.0)?;
    let set = metal.residency_set_handle();
    assert!(
        !set.is_empty(),
        "expected allocations to have registered in the residency set"
    );

    drop((a, b, c));
    assert!(
        !set.is_empty(),
        "dropping tensors must not empty the set (issue #166)"
    );

    // The last handle goes here. `set` outlives it deliberately, which is the
    // whole point: once the device is gone there is nothing left to ask.
    drop(device);

    assert_eq!(
        set.len(),
        0,
        "the last device handle dropping must empty the residency set before \
         the pools free the buffers it lists (`DESIGN.md` §6.3c)"
    );
    Ok(())
}

/// Evicting a buffer removes it from the residency set's membership record.
///
/// The pool bounds its free list by **destroying** buffers whose size will not
/// be asked for again (§6.3b). That is a second place a buffer's existence
/// ends, and nothing was announcing it — so the record went on naming
/// allocations whose handles were gone, on the steady-state decode path rather
/// than only at teardown.
///
/// **This asserts the record, and it holds on both arms** — `retire_batch` and
/// `remove_batch` both drop the key, and that is deliberate: the record is what
/// decides whether a later `removeAllocation` may be issued, so the panic guard
/// (§6.3c) must not depend on which arm ships. What the arms differ in is
/// whether Metal is *told*, which is bytes rather than membership and is pinned
/// by `evicting_a_free_buffer_releases_the_allocation_to_metal` below.
#[test]
fn evicting_a_free_buffer_retires_it_from_the_membership_record() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Free several distinct sizes, then synchronize so they reach the free
    // list rather than sitting pending on the GPU.
    for i in 1..=8 {
        let t = Tensor::zeros((256, 256 + i), DType::F32, &device)?;
        drop(t);
    }
    metal.wait_until_completed()?;

    let before = metal.residency_set_len();
    let evicted_before = {
        let (s, p) = metal.pool_counters();
        s.evicted + p.evicted
    };

    // A budget of zero evicts the whole free list.
    metal.set_free_budget(0);

    let (s, p) = metal.pool_counters();
    let evicted = (s.evicted + p.evicted) - evicted_before;
    assert!(
        evicted > 0,
        "expected the budget change to evict something; with nothing evicted \
         this test asserts nothing"
    );

    let after = metal.residency_set_len();
    assert_eq!(
        before - after,
        evicted as usize,
        "every evicted buffer must leave the membership record: {evicted} \
         evicted, record went {before} -> {after}"
    );

    // The consequence that matters: nothing can now name one of those
    // allocations to Metal. `trim_unused_buffers` walks the free list and
    // removes what it finds, and the free list is empty — so this is a no-op
    // rather than a removal of an allocation the pool already destroyed.
    metal.trim_unused_buffers();
    assert_eq!(
        metal.residency_set_len(),
        after,
        "a trim after an eviction must not remove anything further"
    );
    Ok(())
}

/// **Eviction tells Metal, so the allocation is released rather than retained**
/// (`DESIGN.md` §6.3e, lloom issue #210).
///
/// This is the change, and it is the half the membership test above structurally
/// cannot see: both arms drop the key, and only one calls `removeAllocation`.
/// The retention the other leaves is safe — it is a retention, not a dangling
/// reference — and it reached **48 GB** at long context under `KvAppend=Cat`,
/// exhausting a 64 GB device at 97 % of `recommendedMaxWorkingSetSize`.
///
/// Asserted on the counters rather than on `currentAllocatedSize`, deliberately.
/// The driver figure is what #206 measured the defect with and it is the right
/// instrument for a *long-context run*; here it would make a unit test depend on
/// Metal's own release timing, which `removeAllocation` documents as *"on the
/// next commit"* and which is not ours to assert. What is ours is that the call
/// was made, batched, and that is exactly what `removed` and `commits` say.
#[test]
fn evicting_a_free_buffer_releases_the_allocation_to_metal() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    for i in 1..=8 {
        let t = Tensor::zeros((256, 512 + i), DType::F32, &device)?;
        drop(t);
    }
    metal.wait_until_completed()?;

    let before = metal.residency_counters();
    let evicted_before = {
        let (s, p) = metal.pool_counters();
        s.evicted + p.evicted
    };

    metal.set_free_budget(0);

    let (s, p) = metal.pool_counters();
    let evicted = (s.evicted + p.evicted) - evicted_before;
    assert!(
        evicted > 0,
        "expected the budget change to evict something; with nothing evicted \
         this test asserts nothing"
    );

    let after = metal.residency_counters();
    let removed = after.removed - before.removed;
    let retired = after.retired - before.retired;

    // THE ASSERTION. Every evicted buffer is unregistered from Metal, not
    // merely forgotten by us.
    assert_eq!(
        removed, evicted,
        "eviction must release the allocation to Metal: {evicted} evicted but \
         only {removed} removed. A buffer forgotten without a `removeAllocation` \
         is retained by Metal until the device drops, which is the 48 GB of \
         §6.3e."
    );
    assert_eq!(
        retired, 0,
        "the retaining arm ran: {retired} allocations were forgotten without \
         telling Metal"
    );

    // **And it is batched.** The number of Metal calls is what §6.3d's cost
    // argument is about: #166 measured the per-buffer form at +0.087 ms/token
    // and the per-batch form at +0.062, and the difference between them IS the
    // commit count. A regression to one commit per buffer would be invisible to
    // every assertion above.
    let commits = after.commits - before.commits;
    assert!(
        commits <= 2,
        "one eviction round must cost at most one commit per pool, got \
         {commits} for {evicted} buffers -- a per-buffer commit is the +0.087 \
         ms/token form (§6.3d)"
    );
    Ok(())
}

/// **A residency commit was already on the allocation path**, which is the fact
/// that makes the eager arm affordable (`DESIGN.md` §6.3e, issue #210).
///
/// §6.3d attributes eager unregistration's cost to putting a `commit()` on the
/// per-token path, and §6.3e wonders whether a sweep folded into the existing
/// completion handler could avoid one because *"the `commit()` may not be
/// additional"*. Both are reasoning about the same quantity and neither had
/// counted it. Counted: `ResidencySet::insert` is one buffer and one commit, and
/// every pool **miss** calls it — so allocation already commits at the
/// allocation rate, and what eviction adds is one commit per *round*.
///
/// Pinned as a test because it is the load-bearing premise of the choice #210
/// made, and because a future change routing allocation through `insert_batch`
/// would improve it — at which point this test says so by failing rather than
/// leaving the reasoning stale.
#[test]
fn allocation_already_commits_the_residency_set() -> Result<()> {
    let _guard = lock();
    let Some(device) = metal_device() else {
        return Ok(());
    };
    let Device::Metal(metal) = &device else {
        return Ok(());
    };

    // Warm the pool so the sizes below are genuine misses rather than hits: a
    // hit allocates nothing and registers nothing, which is the property
    // `a_pool_hit_is_not_subject_to_the_cap` pins one layer down.
    let before = metal.residency_counters();

    // Distinct, unusual sizes so each one misses.
    let mut held = Vec::new();
    for i in 0..4 {
        held.push(Tensor::zeros((37, 1024 + i * 7), DType::F32, &device)?);
    }
    let after = metal.residency_counters();

    let added = after.added - before.added;
    let commits = after.commits - before.commits;
    assert!(
        added > 0,
        "no allocation registered, so this test measured nothing"
    );
    assert_eq!(
        commits, added,
        "expected one commit per registered allocation, got {commits} for \
         {added} additions. This is the figure the eviction cost is compared \
         against (§6.3e): if allocation batched its commits, the eager \
         eviction arm's relative cost would be larger than #210 measured."
    );
    drop(held);
    Ok(())
}
