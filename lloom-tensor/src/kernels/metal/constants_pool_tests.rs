//! A pooled params slot is rewritten only when nothing in flight can read it.
//!
//! The pool gives dispatch position N the same buffer every step, which is what
//! lets an ICB command bind it by identity. The hazard is the CPU write into
//! that buffer: no barrier orders it against a GPU dispatch from the previous
//! step that is still reading the old bytes (`ConstantsPool`). These tests drive
//! the clock by hand, so "the GPU is still reading" and "the GPU has finished"
//! are states the test sets rather than races it hopes to catch.
use crate::kernels::kernels::params::ConstantsPool;
use crate::kernels::metal::{Buffer, Device, GpuClock};

fn device() -> Device {
    Device::system_default().unwrap()
}

fn read(buf: &Buffer, len: usize) -> Vec<u8> {
    // SAFETY: shared storage at least `len` long -- every pooled slot is
    // `CONSTANTS_SLOT_BYTES`, and `len` is a block the pool accepted.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, len) }.to_vec()
}

/// The three arms after allocation, in the order a lookahead loop meets them:
/// same bytes (hand the slot out untouched), different bytes with the previous
/// reader in flight (a private buffer, the slot untouched), different bytes with
/// the previous reader complete (rewrite in place, identity kept).
#[test]
fn a_slot_is_rewritten_only_after_its_last_reader_completes() {
    let device = device();
    let pool = ConstantsPool::new();
    let clock = GpuClock::new();
    let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let b = [9u8, 9, 9, 9, 9, 9, 9, 9];

    // Step 0: first sight of the position allocates the slot, in epoch 0.
    let slot = pool.next(&device, &a, Some(&clock)).unwrap();
    assert_eq!(read(&slot, a.len()), a);

    // Step 1, same bytes, epoch 0 still open: the slot itself, nothing written.
    pool.reset();
    let again = pool.next(&device, &a, Some(&clock)).unwrap();
    assert_eq!(
        again.raw_addr(),
        slot.raw_addr(),
        "unchanged bytes must keep the slot's identity"
    );

    // Step 2, new bytes, epoch 0 still open: the reader from step 1 may be
    // executing, so the dispatch gets a private buffer and the slot keeps `a`.
    pool.reset();
    let private = pool.next(&device, &b, Some(&clock)).unwrap();
    assert_ne!(
        private.raw_addr(),
        slot.raw_addr(),
        "a slot whose reader may be in flight must not be rewritten"
    );
    assert_eq!(read(&private, b.len()), b);
    assert_eq!(
        read(&slot, a.len()),
        a,
        "the slot must be left as its in-flight reader saw it"
    );

    // The GPU finishes epoch 0.
    let closed = clock.commit_epoch();
    assert_eq!(closed, 0);
    clock.mark_completed(closed);

    // Step 3, new bytes, every reader of the slot complete: rewritten in place,
    // so the position's identity is back on the slot.
    pool.reset();
    let rewritten = pool.next(&device, &b, Some(&clock)).unwrap();
    assert_eq!(
        rewritten.raw_addr(),
        slot.raw_addr(),
        "an idle slot is rewritten, not replaced"
    );
    assert_eq!(read(&slot, b.len()), b);

    let c = pool.counters();
    assert_eq!(
        (c.allocated, c.unchanged, c.detached, c.rewritten),
        (1, 1, 1, 1),
        "each arm exactly once: {c:?}"
    );
}

/// A slot handed out again with the same bytes takes the epoch of its newest
/// reader, so a later rewrite waits for that reader too -- not only for the
/// one that first wrote the bytes.
#[test]
fn an_unchanged_hand_out_still_advances_the_slots_epoch() {
    let device = device();
    let pool = ConstantsPool::new();
    let clock = GpuClock::new();
    let a = [1u8, 1, 1, 1];
    let b = [2u8, 2, 2, 2];

    let slot = pool.next(&device, &a, Some(&clock)).unwrap(); // epoch 0
    clock.mark_completed(clock.commit_epoch()); // epoch 0 done; epoch 1 open
    pool.reset();
    let _ = pool.next(&device, &a, Some(&clock)).unwrap(); // unchanged, read in epoch 1
    pool.reset();
    let got = pool.next(&device, &b, Some(&clock)).unwrap();
    assert_ne!(
        got.raw_addr(),
        slot.raw_addr(),
        "epoch 1 is open and its reader saw `a`; the slot must not be rewritten under it"
    );
    assert_eq!(read(&slot, a.len()), a);
}

/// Without a clock there is nothing in flight to order behind, and the slot is
/// rewritten unconditionally -- the arm the blocking harnesses ran on.
#[test]
fn without_a_clock_the_slot_is_rewritten_in_place() {
    let device = device();
    let pool = ConstantsPool::new();
    let a = [1u8, 2, 3, 4];
    let b = [5u8, 6, 7, 8];
    let slot = pool.next(&device, &a, None).unwrap();
    pool.reset();
    let again = pool.next(&device, &b, None).unwrap();
    assert_eq!(again.raw_addr(), slot.raw_addr());
    assert_eq!(read(&slot, b.len()), b);
    assert_eq!(pool.counters().detached, 0);
}
