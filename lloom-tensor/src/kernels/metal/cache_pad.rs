//! Cache-line padding for **control state** -- struct fields written by
//! different threads (`DESIGN.md` §9.2, issue #442).
//!
//! # The rule, and it is dangerous without its unit
//!
//! **Pad the control state -- struct fields written by different threads -- at
//! 128 B. Never pad DATA at sub-tensor granularity.**
//!
//! The second half is not a caution, it is the larger number. 2048 f16 elements
//! padded individually to 128 B is 262144 bytes against 4096 -- a **64x
//! blowup** -- against a decode step that already moves 5.394 GB at §3.4's
//! measured 349.7 GB/s. At *tensor* granularity padding is free by arithmetic
//! rather than by luck: every LFM2 decode activation shape is already a
//! multiple of 128 B (hidden 4096, mlp_inter 21504, logits 512000, conv_state
//! 12288 -- 0.00 % waste), and `buf_size` rounds to `BUFFER_ALIGNMENT` before
//! allocation anyway.
//!
//! So this type is for **struct instances**, of which there is one each. It is
//! not a general "pad things" mechanism, and applying it per element, per
//! dtype slot or per tensor is a different proposal needing its own arithmetic.
//!
//! # Why 128 and not 64
//!
//! `hw.cachelinesize` is **128** on Apple silicon, not the 64 a portable
//! assumption would use. `BUFFER_ALIGNMENT = 128` already encodes this for
//! buffers, and §9.2 records that lowering the arena's alignment to 64 would
//! silently introduce false sharing between adjacent slots. Same constant, same
//! reason, applied to a struct field instead of a buffer offset.
//!
//! # What it does NOT pad
//!
//! **Read-mostly flags.** Measured rather than asserted: a reader of a line
//! another core writes reads **0.96x** against the same reader with the flag
//! padded away -- a wash, and if anything marginally worse. A read re-fetches
//! on invalidation but does not itself invalidate anyone; the expensive case is
//! **write/write**. `encoder.rs`, `trace.rs`, `run_telemetry.rs` and
//! `profile.rs` are mostly `AtomicBool` and are deliberately untouched.
//!
//! **The classification is what to check, not the padding** -- a "read-mostly"
//! flag written occasionally on a hot path is a write/write case in disguise.
//!
//! **And two writers is necessary rather than sufficient.**
//! `residency_set.rs`'s counters are written from the allocation path *and*
//! from the eviction path, which are two threads -- and both writes sit
//! **inside `members.lock()`**, so the two can never execute concurrently and
//! the line moves once per lock handoff, which the `Mutex` already forces.
//! Padding them would buy nothing. The test is *concurrent* write/write, and a
//! writer census alone does not establish it.

/// One cache line of alignment, for a field written by a different thread from
/// its neighbour.
///
/// `#[repr(align(128))]` emits **no instruction**: it changes where the field
/// lands, not what happens when it is touched. There is no branch, no
/// allocation and no arithmetic, which is why this cannot move a digest --
/// nothing computed changes.
///
/// Derefs to the inner value, so it substitutes for the field it replaces at
/// every use site without a rename.
#[derive(Debug, Default)]
#[repr(align(128))]
pub struct CachePadded<T>(pub T);

impl<T> CachePadded<T> {
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// The line this pads to. `hw.cachelinesize` on Apple silicon, and the same
/// constant `BUFFER_ALIGNMENT` uses.
///
/// Stated as a `const` so the assertions below compare against a name rather
/// than a repeated literal, and so a reader who changes one has to meet the
/// other.
pub const CACHE_LINE: usize = 128;

// The alignment is the whole mechanism, so it is asserted rather than trusted:
// `repr(align)` is a silent no-op if the attribute is dropped or the literal is
// edited, and nothing else in the build would notice. A compile-time check
// costs nothing and fails loudly.
const _: () = {
    assert!(std::mem::align_of::<CachePadded<u64>>() == CACHE_LINE);
    assert!(std::mem::align_of::<CachePadded<std::sync::atomic::AtomicU64>>() == CACHE_LINE);
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    /// The alignment holds for the types actually padded, not only for `u64`.
    ///
    /// `size_of` is checked too: a `repr(align(N))` type's size is rounded up
    /// to a multiple of its alignment, so two adjacent padded fields are
    /// provably on different lines rather than merely individually aligned.
    /// That is the property the padding exists for and it does not follow from
    /// alignment alone.
    #[test]
    fn padded_fields_occupy_distinct_lines() {
        assert_eq!(std::mem::align_of::<CachePadded<AtomicU64>>(), CACHE_LINE);
        assert_eq!(std::mem::align_of::<CachePadded<AtomicUsize>>(), CACHE_LINE);
        assert_eq!(std::mem::size_of::<CachePadded<AtomicU64>>(), CACHE_LINE);

        // Two in one struct: the offsets must differ by at least a line, which
        // is what "not false sharing" means and what a per-field alignment
        // assertion does not by itself say.
        #[derive(Default)]
        struct Two {
            a: CachePadded<AtomicU64>,
            b: CachePadded<AtomicU64>,
        }
        let two = Two::default();
        let a = &two.a as *const _ as usize;
        let b = &two.b as *const _ as usize;
        assert!(
            b.abs_diff(a) >= CACHE_LINE,
            "padded fields {a:#x} and {b:#x} share a {CACHE_LINE} B line"
        );
    }

    /// The wrapper is transparent to reads and writes: `Deref` reaches the
    /// atomic, and the value observed is the value stored.
    ///
    /// This is the check that the padding did not change *behaviour* -- the
    /// half a digest gate covers end to end and a unit test can state directly.
    #[test]
    fn padding_is_transparent_to_the_value() {
        let padded = CachePadded::new(AtomicU64::new(7));
        assert_eq!(padded.load(Ordering::Relaxed), 7);
        padded.fetch_add(35, Ordering::Relaxed);
        assert_eq!(padded.load(Ordering::Relaxed), 42);
    }
}
