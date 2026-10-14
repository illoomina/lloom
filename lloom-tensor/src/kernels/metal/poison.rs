//! ASan shadow-memory poisoning for allocator invariants the sanitizer cannot
//! otherwise see (`DESIGN.md` §6.3b, §9.2, issue #475).
//!
//! # What plain ASan misses, and why it is exactly our shape
//!
//! ASan checks accesses against *allocation* boundaries. Every allocator this
//! project has built sub-divides one allocation: the activation arena is one
//! `MTLBuffer` handed out by offset (§9.2), the pool hands back whole buffers
//! whose *reuse eligibility* is a state ASan has no concept of, and #472's
//! proposed ring is slots in one region whose whole correctness argument is
//! *"the consumer only reads what the producer published"*.
//!
//! In every one of those, a read of the wrong slot is a **legal read inside a
//! live allocation** and plain ASan sees nothing. Measured for #475: a 4096 B
//! region, one 64 B slot handed out, a read one byte past the slot —
//! **caught as use-after-poison when the region is poisoned, exit 0 when it is
//! not**, on otherwise identical code. The second arm is what makes the first
//! evidence rather than a demonstration (§15.1a).
//!
//! Poisoning turns those conventions into checked invariants: mark what a
//! caller must not touch, unmark it at publish, mark it again at recycle.
//!
//! # THREE BOUNDARIES, and they must be stated wherever this is used
//!
//! **1. The GPU never sees the shadow map.** Poison is a host-side convention
//! in ASan's shadow memory. A Metal kernel writing an arena slot **cannot trap
//! on it**, and the driver is uninstrumented. This instruments the *host's*
//! view of an allocator. **A GPU-side overrun stays out of reach, and an arena
//! bug could live entirely on the device side** — which is precisely the half
//! §3.7g's and §191's findings live on.
//!
//! **2. ASan changes the allocator, so any timing taken under it is
//! meaningless.** §11.5e measures `to_cpu`'s fast/blit split as
//! *allocator-state-dependent*, swinging blit 1 → 12 over four consecutive runs
//! of one binary with identical flags, and 3 → 26 (8.7×) across a committed
//! 12-run arm. A sanitizer build changes allocation addresses, sizes and
//! timing, so it moves exactly the state that split turns on. **This is a
//! CORRECTNESS arm. Do not report a duration taken under it as a result**
//! (`CONTRIBUTING.md` §3.2, `DESIGN.md` §2.4).
//!
//! **3. [`is_poisoned`] is a QUERY, not a barrier.** It answers about the
//! shadow map at that instant. §11.5e measures the arm selection drifting run
//! to run on one binary, so a poison-state check is a debugging aid rather than
//! a gate — nothing may branch on it in shipping code.
//!
//! # A fourth boundary this module MEASURED rather than inherited
//!
//! **A poison needs a CPU mapping, and the pool's buffers mostly have none.**
//! §15.2 #12a records a `StorageModePrivate` allocation as CPU-mapped below
//! 16384 B and never at or above it. Probed directly for #475 through
//! `newBufferWithLength:` (`measurements/issue-475-raw/mtl-poison-probe.txt`):
//!
//! ```text
//!   Shared   4096 B  -> mapped, poison round-trips 0/1/0
//!   Private  4096 B  -> mapped, poison round-trips 0/1/0
//!   Shared  16384 B  -> mapped
//!   Private 16384 B  -> contents() == NULL
//!   Private   1 MiB  -> contents() == NULL
//! ```
//!
//! A poisoned `MTLBuffer` byte **does** trap on a host read, so the shadow map
//! covers Metal's mappings when they exist. What it cannot cover is an
//! allocation with no mapping, and that is most of the pool by bytes.
//!
//! **So coverage is a number rather than an assumption**, and this module
//! counts it: [`Counters::unmappable`] is the buffers a poison could not
//! reach. **A run reporting zero poisons and zero unmappable has not verified
//! anything** — it has failed to engage, which is §2.4's rule after #69's
//! vacuous determinism run and #454's `to_cpu_fast=0` sweep.
//!
//! # Inert unless BOTH the feature and the sanitizer are on
//!
//! Following `run_telemetry` and `hazard_audit`: without the `asan-poison`
//! feature every entry point below is an empty inline function that drops its
//! arguments, so there is no call to elide and no state to allocate. With the
//! feature but without `-Zsanitizer=address` the interface symbols do not
//! exist and the crate would fail to link — which is why the feature's own
//! documentation says it is only ever enabled by the `sanitize` profile.

/// What the poisoning path did, so a run can prove the mechanism engaged.
///
/// **Every field is a count, so it is load-independent** and reportable under
/// contention (§6.6a) — unlike anything timed under a sanitizer, which
/// boundary 2 forbids.
///
/// The pair that matters is `poisoned` against `unmappable`: the first is
/// coverage collected, the second is coverage the 16384 B threshold denied.
/// Reporting one without the other invites reading an unmapped pool as a clean
/// one, which is §15.1a's VACUOUS class in the instrument written to prevent
/// it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Regions poisoned — a buffer entering the free list, or an arena region
    /// marked unallocated.
    pub poisoned: u64,
    /// Regions unpoisoned — a buffer handed to a caller, a slot published.
    pub unpoisoned: u64,
    /// Regions a poison could not reach because the allocation has no CPU
    /// mapping (`contents()` is null — boundary 4).
    ///
    /// **Not an error.** It is the measured share of the pool the 16384 B
    /// threshold puts outside the shadow map, and it is what stops a low
    /// `poisoned` count reading as a clean run.
    pub unmappable: u64,
    /// Regions skipped because they were empty. Kept separate from
    /// `unmappable` so a zero-length bug cannot masquerade as the threshold.
    pub empty: u64,
}

#[cfg(feature = "asan-poison")]
mod imp {
    use super::Counters;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ASan's documented manual-poisoning interface. These are exported by the
    // sanitizer runtime, so this is a documented API rather than a symbol
    // interposition or a link-order trick -- which is what makes it something
    // this project may rest an invariant on.
    //
    // Verified callable from Rust on aarch64-apple-darwin for #475; the probe
    // and its inverted arm are in `measurements/issue-475-raw/`.
    extern "C" {
        fn __asan_poison_memory_region(addr: *const u8, size: usize);
        fn __asan_unpoison_memory_region(addr: *const u8, size: usize);
        fn __asan_address_is_poisoned(addr: *const u8) -> i32;
    }

    static POISONED: AtomicU64 = AtomicU64::new(0);
    static UNPOISONED: AtomicU64 = AtomicU64::new(0);
    static UNMAPPABLE: AtomicU64 = AtomicU64::new(0);
    static EMPTY: AtomicU64 = AtomicU64::new(0);

    /// Whether `addr..addr + len` is addressable at all.
    ///
    /// A null pointer is boundary 4: the allocation exists and has no CPU
    /// mapping, so there is nothing to mark. Counted rather than ignored.
    fn reachable(addr: *const u8, len: usize) -> bool {
        if addr.is_null() {
            UNMAPPABLE.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if len == 0 {
            EMPTY.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Marks `addr..addr + len` as memory the host must not touch.
    ///
    /// # Safety
    ///
    /// The range must belong to an allocation that outlives every access to
    /// it, and the caller must guarantee **nothing else is reading or writing
    /// it** — poisoning a region another thread is using turns a correct
    /// access into a reported error.
    pub unsafe fn poison(addr: *const u8, len: usize) {
        if !reachable(addr, len) {
            return;
        }
        __asan_poison_memory_region(addr, len);
        POISONED.fetch_add(1, Ordering::Relaxed);
    }

    /// Clears the mark, making `addr..addr + len` accessible again.
    ///
    /// # Safety
    ///
    /// As [`poison`]. Unpoisoning a range that was never poisoned is
    /// harmless; unpoisoning one ASan itself poisoned (a redzone, a freed
    /// region) would suppress a real report, so the range must be one this
    /// module marked.
    pub unsafe fn unpoison(addr: *const u8, len: usize) {
        if !reachable(addr, len) {
            return;
        }
        __asan_unpoison_memory_region(addr, len);
        UNPOISONED.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether `addr` is currently marked.
    ///
    /// **A query, not a barrier** (boundary 3). For a debugger and for tests;
    /// shipping code must not branch on it.
    ///
    /// # Safety
    ///
    /// `addr` must be a pointer whose shadow ASan maps -- any live allocation.
    pub unsafe fn is_poisoned(addr: *const u8) -> bool {
        !addr.is_null() && __asan_address_is_poisoned(addr) != 0
    }

    pub fn counters() -> Counters {
        Counters {
            poisoned: POISONED.load(Ordering::Relaxed),
            unpoisoned: UNPOISONED.load(Ordering::Relaxed),
            unmappable: UNMAPPABLE.load(Ordering::Relaxed),
            empty: EMPTY.load(Ordering::Relaxed),
        }
    }

    pub const ENABLED: bool = true;
}

#[cfg(not(feature = "asan-poison"))]
mod imp {
    use super::Counters;

    /// # Safety
    /// See the enabled implementation. Inert here.
    #[inline(always)]
    pub unsafe fn poison(_addr: *const u8, _len: usize) {}

    /// # Safety
    /// See the enabled implementation. Inert here.
    #[inline(always)]
    pub unsafe fn unpoison(_addr: *const u8, _len: usize) {}

    /// # Safety
    /// See the enabled implementation. Inert here.
    #[inline(always)]
    pub unsafe fn is_poisoned(_addr: *const u8) -> bool {
        false
    }

    #[inline(always)]
    pub fn counters() -> Counters {
        Counters::default()
    }

    pub const ENABLED: bool = false;
}

pub use imp::{counters, is_poisoned, poison, unpoison, ENABLED};

/// Poisons a buffer's whole CPU mapping, if it has one.
///
/// The pool's use: a buffer entering the free list is one **no caller may
/// touch**, and until #475 nothing said so in a way any tool could check.
///
/// # Safety
///
/// The buffer must have no live user. In the pool this is guaranteed by
/// construction — a buffer reaches the free list only after its last handle
/// dropped *and* the GPU epoch it was released in completed (§6.3b), which is
/// strictly stronger than "the CPU is done with it".
pub unsafe fn poison_buffer(buffer: &super::Buffer) {
    // Do not evaluate Objective-C accessors for an inert diagnostic hook.
    // In debug builds their signature verification itself can allocate.
    #[cfg(feature = "asan-poison")]
    poison(buffer.data(), buffer.length());
    #[cfg(not(feature = "asan-poison"))]
    let _ = buffer;
}

/// Clears a buffer's mark as it is handed to a caller.
///
/// # Safety
///
/// As [`poison_buffer`]. Must be called before the caller can observe the
/// buffer, or the caller's first legitimate access is reported.
pub unsafe fn unpoison_buffer(buffer: &super::Buffer) {
    #[cfg(feature = "asan-poison")]
    unpoison(buffer.data(), buffer.length());
    #[cfg(not(feature = "asan-poison"))]
    let _ = buffer;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interface is callable and the counters move, or it is inert and
    /// they do not. **Both are legitimate; what is not is a silent third
    /// state** where the calls compile away under the feature.
    ///
    /// This is the engagement check §2.4 asks for, at the level of the
    /// mechanism rather than of a run: a build that claims `asan-poison` and
    /// counts nothing has not poisoned anything.
    #[test]
    fn counters_track_the_feature() {
        let mut region = vec![0u8; 128];
        let p = region.as_mut_ptr();
        let before = counters();
        unsafe {
            poison(p, 128);
            unpoison(p, 128);
        }
        let after = counters();

        if ENABLED {
            assert_eq!(
                after.poisoned - before.poisoned,
                1,
                "the `asan-poison` feature is on and a poison did not count -- \
                 the calls are compiling away, which makes every later coverage \
                 figure vacuous (DESIGN.md §15.1a)"
            );
            assert_eq!(after.unpoisoned - before.unpoisoned, 1);
        } else {
            assert_eq!(
                (after.poisoned, after.unpoisoned),
                (before.poisoned, before.unpoisoned),
                "the feature is off and something counted -- the inert arm is \
                 not inert"
            );
        }
    }

    /// A null pointer counts as unmappable rather than being silently dropped.
    ///
    /// **This is boundary 4's counter, and it is the one that stops a low
    /// coverage figure reading as a clean run.** A `StorageModePrivate`
    /// allocation at or above 16384 B has no CPU mapping (§15.2 #12a,
    /// reproduced for #475), so the pool hands this path nulls routinely.
    #[test]
    fn an_unmappable_region_is_counted_not_ignored() {
        let before = counters();
        unsafe {
            poison(std::ptr::null(), 4096);
        }
        let after = counters();
        if ENABLED {
            assert_eq!(
                after.unmappable - before.unmappable,
                1,
                "a null mapping was dropped without being counted -- an \
                 unmapped pool would then read exactly like a poisoned one"
            );
            assert_eq!(after.poisoned, before.poisoned);
        }
    }

    /// An empty range is its own bucket.
    ///
    /// Separate from `unmappable` so a zero-length bug cannot be read as the
    /// 16384 B threshold, which would attribute a defect to the machine.
    #[test]
    fn an_empty_region_is_counted_separately() {
        let x = 0u8;
        let before = counters();
        unsafe {
            poison(&x as *const u8, 0);
        }
        let after = counters();
        if ENABLED {
            assert_eq!(after.empty - before.empty, 1);
            assert_eq!(after.unmappable, before.unmappable);
            assert_eq!(after.poisoned, before.poisoned);
        }
    }
}

#[cfg(test)]
mod engagement {
    //! Does the pool's poisoning actually fire on a real Metal buffer?
    //!
    //! A clean sanitizer run over the decode path is #475's registered
    //! prediction, and a clean run is worth nothing if the mechanism was inert
    //! -- §15.1a's VACUOUS class, and the shape #454 hit with `to_cpu_fast=0`
    //! and #69 with a consumed `OnceLock`. So the coverage figure needs its own
    //! non-vacuity arm.
    use super::*;
    use crate::kernels::metal::{BufferPool, Device, GpuClock};
    use std::sync::Arc;

    /// A buffer in the pool's free list is marked, and is not marked once the
    /// pool hands it back.
    ///
    /// This is the mechanism, asserted end to end through the real pool on a
    /// real `MTLBuffer` rather than on a `Vec<u8>` stand-in.
    #[test]
    fn the_pool_marks_a_free_buffer_and_clears_it_on_acquire() {
        if !ENABLED {
            return;
        }
        let dev = Device::system_default().expect("no Metal device");
        let clock = Arc::new(GpuClock::new());
        let pool = BufferPool::with_clock(Arc::clone(&clock));

        // Small enough to be CPU-mapped: boundary 4 says a Private allocation
        // at or above 16384 B has no mapping to mark, and `RESOURCE_OPTIONS`
        // is Shared so it is mapped at every size -- but keeping it small
        // keeps this test honest about what it is checking.
        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        let addr = unsafe { raw.data() };
        assert!(
            !addr.is_null(),
            "the fixture buffer has no CPU mapping, so this test could not \
             have observed a poison either way (boundary 4)"
        );
        let b = pool.adopt(raw, 1024);

        assert!(
            !unsafe { is_poisoned(addr) },
            "a live buffer is marked -- the caller's own access would be \
             reported"
        );

        // Into the free list: dropped, and the GPU epoch completed.
        drop(b);
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();

        assert!(
            unsafe { is_poisoned(addr) },
            "a buffer in the free list is NOT marked, so the poisoning is \
             inert and every coverage figure taken from it is vacuous \
             (DESIGN.md §15.1a)"
        );

        // Back out to a caller: the mark must be gone before they touch it.
        let again = pool.acquire(1024).expect("the buffer was just freed");
        assert!(
            !unsafe { is_poisoned(addr) },
            "acquire handed out a marked buffer -- the caller's first \
             legitimate read is what gets reported"
        );
        drop(again);
    }

    /// The counters move on a real pool cycle, so a run can prove engagement
    /// without reading the shadow map.
    ///
    /// This is what a harness reports: `poisoned` against `unmappable` is
    /// coverage collected against coverage the 16384 B threshold denied, and
    /// **a run reporting zero of both has not verified anything**.
    #[test]
    fn a_pool_cycle_moves_the_counters() {
        if !ENABLED {
            return;
        }
        let dev = Device::system_default().expect("no Metal device");
        let clock = Arc::new(GpuClock::new());
        let pool = BufferPool::with_clock(Arc::clone(&clock));
        let before = counters();

        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .expect("buffer allocation");
        let b = pool.adopt(raw, 1024);
        drop(b);
        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();

        let after = counters();
        assert!(
            after.poisoned > before.poisoned,
            "a full release-and-drain cycle poisoned nothing: {before:?} -> \
             {after:?}"
        );
    }
}
