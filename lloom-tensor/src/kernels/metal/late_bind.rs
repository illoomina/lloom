//! `tuning.instrument.late_bind` (`LLOOM_LATE_BIND`, read once): does anything
//! bind a pooled buffer after its last handle dropped? (§15.1m)
//!
//! The pool stamps a released buffer with the epoch open at the drop and
//! offers it again once that epoch has retired (§6.3b). That is sound iff
//! every GPU use of the buffer was encoded before the drop: a use encoded
//! later sits in a later command buffer, which the stamp does not cover, and
//! if the buffer has meanwhile gone to a new tenant — a host upload writes its
//! bytes the instant it is acquired — that use reads the new tenant's bytes.
//!
//! This keeps the set of released buffers (parked or free) and checks every
//! compute bind and blit against it. A hit is a use after release, counted
//! and, for the first few, printed with the kernel label and the epochs.
//! `report()` gives the totals.
//!
//! Off, every entry point is one relaxed load. Never the shipping path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use super::Buffer;

#[cfg(test)]
thread_local! {
    /// Armed by a test on its own thread, where the tuning knob cannot be set
    /// and a process-wide arm would reach every other test.
    static ARMED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn armed() -> bool {
    #[cfg(test)]
    if ARMED_HERE.with(|a| a.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().instrument.late_bind)
}

/// Released buffers by object address, with the epoch they were stamped with.
fn released() -> &'static parking_lot::Mutex<HashMap<usize, u64>> {
    static MAP: OnceLock<parking_lot::Mutex<HashMap<usize, u64>>> = OnceLock::new();
    MAP.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

static RELEASES: AtomicU64 = AtomicU64::new(0);
static BINDS: AtomicU64 = AtomicU64::new(0);
static LATE: AtomicU64 = AtomicU64::new(0);
static LATE_WRITES: AtomicU64 = AtomicU64::new(0);

/// What the instrument saw, since the process started.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LateBindReport {
    /// Whether the instrument ran at all.
    pub armed: bool,
    /// Buffers released to a pool.
    pub releases: u64,
    /// Compute binds and blit operands checked.
    pub binds: u64,
    /// Of those, uses of a buffer released and not yet handed out again.
    pub late: u64,
    /// Of `late`, the ones that write.
    pub late_writes: u64,
}

/// The totals.
pub fn report() -> LateBindReport {
    LateBindReport {
        armed: armed(),
        releases: RELEASES.load(Ordering::Relaxed),
        binds: BINDS.load(Ordering::Relaxed),
        late: LATE.load(Ordering::Relaxed),
        late_writes: LATE_WRITES.load(Ordering::Relaxed),
    }
}

/// A pool took `buffer` back, stamped with `epoch`.
pub fn on_release(buffer: &Buffer, epoch: u64) {
    if !armed() {
        return;
    }
    RELEASES.fetch_add(1, Ordering::Relaxed);
    released().lock().insert(buffer.raw_ptr() as usize, epoch);
}

/// A pool handed `buffer` to a new tenant, or destroyed it.
pub fn on_leave(buffer: &Buffer) {
    if !armed() {
        return;
    }
    released().lock().remove(&(buffer.raw_ptr() as usize));
}

/// `ptr` (a buffer's object address) is being bound, or used by a blit.
pub fn on_bind(ptr: usize, write: bool, label: &str) {
    if !armed() {
        return;
    }
    BINDS.fetch_add(1, Ordering::Relaxed);
    let Some(stamp) = released().lock().get(&ptr).copied() else {
        return;
    };
    let n = LATE.fetch_add(1, Ordering::Relaxed) + 1;
    if write {
        LATE_WRITES.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 20 {
        eprintln!(
            "[late bind] #{n} ptr=0x{ptr:x} {} by `{label}`; released at epoch {stamp}",
            if write { "written" } else { "read" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::metal::{BufferPool, CommandBuffer, Device, GpuClock};
    use objc2_metal::MTLCommandQueue;
    use std::sync::Arc;

    /// The instrument's positive control: a bind of a buffer the pool has
    /// taken back counts, and the same buffer bound after it is handed out
    /// again does not. Without it, `late=0` on a real run could mean the
    /// lookup never matches (a key that is not the bind's), not that nothing
    /// binds late.
    #[test]
    fn a_bind_of_a_released_buffer_counts_and_one_of_a_reacquired_buffer_does_not() {
        ARMED_HERE.with(|a| a.set(true));
        let dev = Device::system_default().expect("no Metal device");
        let clock = Arc::new(GpuClock::new());
        let pool = BufferPool::with_clock(Arc::clone(&clock));
        let raw = dev
            .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
            .unwrap();
        let handle = pool.adopt(raw, 1024);
        let stale: Buffer = (*handle).clone();
        drop(handle);

        let queue = dev.new_command_queue().expect("queue");
        let cb = CommandBuffer::new(queue.commandBuffer().unwrap());
        let enc = cb.compute_command_encoder_no_fence();
        let before_stale = report().late;
        enc.set_input_buffer(0, Some(&stale), 0);
        let after_stale = report().late;

        let epoch = clock.commit_epoch();
        clock.mark_completed(epoch);
        pool.drain_completed();
        let again = pool
            .acquire(1024)
            .expect("the released buffer is free again");
        let before_live = report().late;
        enc.set_output_buffer(1, Some(&again), 0);
        let after_live = report().late;
        // Ended before asserting: an encoder dropped open aborts the process
        // and hides which assertion failed.
        enc.end_encoding();

        assert_eq!(
            after_stale,
            before_stale + 1,
            "a bind of a released buffer was not seen"
        );
        assert_eq!(
            again.raw_ptr(),
            stale.raw_ptr(),
            "a different buffer came back"
        );
        assert_eq!(
            after_live, before_live,
            "a bind of a live buffer was called late"
        );
    }
}
