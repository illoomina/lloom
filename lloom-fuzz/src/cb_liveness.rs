//! A stateful fuzzer over command-buffer liveness: does every buffer a command
//! buffer reads stay alive and unchanged until that command buffer completes,
//! with **unretained** command buffers as well as retaining ones? (lloom #706;
//! the hot-path handoff's item (b).)
//!
//! # The claim under test
//!
//! Metal's default command buffer retains every resource encoded into it and
//! releases them on completion. `commandBufferWithUnretainedReferences` does
//! not: if the host drops its last reference to a buffer while the GPU is
//! still reading it, the read is a use-after-free. The handoff argues the
//! unretained form is safe for us because nothing is dropped early — a pool
//! buffer whose last handle drops is parked in `pending` until the GPU epoch it
//! was released in completes (`BufferPool::drain_completed`), and only then
//! joins the free list. That is a claim about *our* discipline, not Metal's,
//! and it is exactly the shape §6.7 L4 / #23 already got wrong once.
//!
//! # The poison
//!
//! Under the `fuzzing` feature the pool writes a sentinel over a buffer the
//! moment it becomes free (`BufferPool::set_free_poison`). Every read the
//! harness encodes is a blit **copy** of a live buffer into a fresh destination
//! the harness owns; after the command buffer completes the destination is
//! read back on the host and compared with the bytes the harness wrote into the
//! source. A destination holding the sentinel is a command buffer that read a
//! buffer AFTER the pool freed it — the use-after-release, made loud. A
//! destination holding anything else that is not the pattern is corruption of
//! another kind and reported as such. With the unretained kind a freed-and-
//! destroyed buffer (`trim`) is a GPU fault instead, which the process reports
//! through the record and LibAFL's crash handler.
//!
//! # The alphabet is the pool's and the queue's together
//!
//! `buffer_pool_fuzz` drives the pool with a hand-turned clock and never
//! submits work; `commands_fuzz` submits work over one buffer it never frees.
//! This harness is the product: real command buffers through the shipping
//! `Commands` (whose `LLOOM_CB_UNRETAINED` switch picks the kind), a real pool
//! sharing that `Commands`' clock, and the host dropping, draining, trimming and
//! re-acquiring buffers while copies of them are in flight.
//!
//! # Every class the unretained kind leaves to the host
//!
//! The pool is one of three. A buffer allocated OUTSIDE the pool (an unpooled
//! upload, a persistent allocation: `AllocUnpooled`) is released by its last
//! handle unless the queue's `InFlightHold` takes it, and an encoder's
//! `MTLFence` is released when the cross-encoder map forgets it — which a
//! second write of the same destination (`CopyAgain`) does while the first
//! encoder's command buffer, and every encoder that waited on its fence, may
//! still be queued. Neither class can be poisoned: a released Metal object is
//! gone. The detector for them is the command buffer's own status, surfaced
//! by the wait as an error, and a copy that read freed memory.

use lloom_tensor::kernels::metal::buffer::Buffer;
use lloom_tensor::kernels::metal::buffer_pool::{BufferHandle, BufferPool, PooledBuffer};
use lloom_tensor::kernels::metal::commands::{CommandBufferKind, Commands};
use lloom_tensor::kernels::metal::completion::CompletionWait;
use lloom_tensor::kernels::metal::device::Device;
use lloom_tensor::kernels::metal::residency_set::ResidencySet;
use lloom_tensor::kernels::RESOURCE_OPTIONS;
use std::sync::Arc;

/// The sentinel the pool writes over a buffer when it frees it.
pub const FREE_POISON: u8 = 0xDE;

/// One symbol of the liveness alphabet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `acquire(size)` or allocate + `adopt`, then fill the buffer with a
    /// per-buffer pattern byte through the host mapping.
    Alloc { size: usize },
    /// Encode a blit copy of live buffer `index % live.len()` into a fresh
    /// destination the harness owns, in the CURRENT command buffer.
    Copy { index: usize },
    /// Drop the host handle of live buffer `index % live.len()`: the pool
    /// parks it in `pending` stamped with the OPEN epoch — the one the command
    /// buffer being encoded right now belongs to.
    Drop { index: usize },
    /// `Commands::flush()`: commit the current command buffer without waiting.
    Flush,
    /// `Commands::flush_and_prune()`: commit, and drop completed buffers from
    /// the in-flight queue (the non-blocking path).
    FlushAndPrune,
    /// `Commands::flush_and_wait()`: commit and wait for everything. The
    /// oracle runs here: every destination written so far is verified.
    FlushAndWait,
    /// `pool.drain_completed()`: pending buffers whose epoch completed become
    /// free — and get poisoned.
    Drain,
    /// `pool.trim()`: destroy every free buffer. Under the unretained kind a
    /// command buffer still reading one of them faults.
    Trim,
    /// Let the GPU retire work without the host waiting on it.
    Settle,
    /// Allocate outside the pool — the shape of an unpooled upload or a
    /// persistent buffer — resident, held in flight by the queue's hold when
    /// it has one, and filled with a per-buffer pattern.
    AllocUnpooled { size: usize },
    /// Copy live buffer `index % live.len()` into the destination of the most
    /// recent copy again, from a new blit encoder: the second write of one
    /// destination replaces the first encoder's fence in the cross-encoder
    /// map, so the map no longer holds it.
    CopyAgain { index: usize },
    /// An owner replacing the buffer it holds — an arena or region base —
    /// gives the old one up through `in_flight::hold_buffer`, as
    /// `MetalDevice` does, and drops it: live buffer `index % live.len()`
    /// when it is one the harness allocated outside the pool.
    ReplaceBase { index: usize },
}

/// One encoded copy, to be verified when its command buffer has completed.
struct Pending {
    dst: Buffer,
    size: usize,
    /// The pattern byte the source held when the copy was encoded.
    expect: u8,
    /// Which live buffer it read (for the report).
    src_id: u64,
}

/// What a run reached.
#[derive(Debug, Default, Clone, Copy)]
pub struct Engagement {
    pub allocs: u64,
    pub pool_hits: u64,
    pub copies: u64,
    pub copies_verified: u64,
    pub drops: u64,
    pub drops_while_in_flight: u64,
    pub drains_freeing: u64,
    pub trims_destroying: u64,
    pub flushes: u64,
    pub waits: u64,
    pub unpooled_allocs: u64,
    pub copies_again: u64,
    pub bases_replaced: u64,
    /// Fences and buffers the queue's hold took past their last host handle.
    pub held_fences: u64,
    pub held_buffers: u64,
    pub unretained: bool,
}

/// Run one sequence on the process's selected command-buffer kind
/// (`LLOOM_CB_UNRETAINED`). The oracle fires at every `FlushAndWait` and at
/// the end.
pub fn run_sequence(dev: &Device, ops: &[Op]) -> Result<Engagement, String> {
    run_sequence_with(dev, ops, CommandBufferKind::selected())
}

/// [`run_sequence`] on a stated kind, so one process can drive both.
pub fn run_sequence_with(
    dev: &Device,
    ops: &[Op],
    kind: CommandBufferKind,
) -> Result<Engagement, String> {
    // An autorelease pool per sequence: the queue, the command buffers and the
    // blit encoders come back autoreleased, and a thread with no pool never
    // releases them (see `commands_fuzz::run_sequence`).
    objc2::rc::autoreleasepool(|_| run_sequence_inner(dev, ops, kind))
}

fn run_sequence_inner(
    dev: &Device,
    ops: &[Op],
    kind: CommandBufferKind,
) -> Result<Engagement, String> {
    let queue = dev
        .new_command_queue()
        .map_err(|e| format!("new_command_queue: {e:?}"))?;
    let residency = Arc::new(ResidencySet::new(dev));
    let cmds = Commands::new_with(
        queue,
        Arc::clone(&residency),
        CompletionWait::Synchronize,
        kind,
    )
    .map_err(|e| format!("Commands::new_with: {e:?}"))?;
    let pool = BufferPool::with_clock(cmds.clock());
    pool.set_free_poison(Some(FREE_POISON));

    let mut live: Vec<(BufferHandle, u8, u64)> = Vec::new();
    // Buffers outside the pool, which an owner can replace: `ReplaceBase`'s
    // population. A raw `Buffer` beside its pattern and id.
    let mut bases: Vec<(Buffer, u8, u64)> = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut eng = Engagement {
        unretained: kind == CommandBufferKind::Unretained,
        ..Default::default()
    };
    let mut next_id = 1u64;
    // Copies encoded since the last blocking wait: a drop while this is
    // non-zero is a drop of a buffer that may be in flight.
    let mut in_flight_copies = 0usize;

    let verify =
        |pending: &mut Vec<Pending>, eng: &mut Engagement, at: &str| -> Result<(), String> {
            for p in pending.drain(..) {
                // SAFETY: the destination is shared storage the harness owns and
                // the command buffer that wrote it has completed (the caller waited).
                let got = unsafe { std::slice::from_raw_parts(p.dst.contents(), p.size) };
                if let Some(k) = got.iter().position(|&b| b != p.expect) {
                    let bad = got[k];
                    let poison = got.iter().filter(|&&b| b == FREE_POISON).count();
                    return Err(format!(
                    "{at}: copy of buffer #{} ({} B, pattern {:#04x}) reads {bad:#04x} at byte {k} \
                     ({poison} poison bytes of {}): {}",
                    p.src_id,
                    p.size,
                    p.expect,
                    p.size,
                    if bad == FREE_POISON {
                        "the command buffer read the buffer AFTER the pool freed it -- a \
                         use-after-release the epoch discipline must make impossible"
                    } else {
                        "the copy does not match what the host wrote"
                    }
                ));
                }
                eng.copies_verified += 1;
            }
            Ok(())
        };

    for (i, op) in ops.iter().enumerate() {
        let at = |what: String| format!("after op {i} ({op:?}): {what}");
        match *op {
            Op::Alloc { size } => {
                let size = size.max(16);
                let buf = match pool.acquire(size) {
                    Some(b) => {
                        eng.pool_hits += 1;
                        b
                    }
                    None => {
                        let raw = dev
                            .new_buffer(size, RESOURCE_OPTIONS)
                            .map_err(|e| at(format!("new_buffer({size}): {e:?}")))?;
                        residency.insert(&raw);
                        pool.adopt(raw, size)
                    }
                };
                let id = next_id;
                next_id += 1;
                // A pattern that is never the poison and differs per buffer.
                let pattern = ((id * 37) % 200) as u8 + 1;
                // SAFETY: shared storage, freshly acquired: no command buffer
                // can be reading a buffer the pool just handed out.
                unsafe { std::ptr::write_bytes(buf.contents(), pattern, buf.length()) };
                live.push((buf, pattern, id));
                eng.allocs += 1;
            }
            Op::Copy { index } => {
                // The pool's and the harness's buffers and the bases, one
                // index space.
                let n = live.len() + bases.len();
                if n == 0 {
                    continue;
                }
                let k = index % n;
                let (buf, size, pattern, id): (&Buffer, usize, &u8, &u64) = if k < live.len() {
                    let (b, p, i) = &live[k];
                    (b, b.size(), p, i)
                } else {
                    let (b, p, i) = &bases[k - live.len()];
                    (b, b.length(), p, i)
                };
                let dst = dev
                    .new_buffer(size, RESOURCE_OPTIONS)
                    .map_err(|e| at(format!("new_buffer(dst {size}): {e:?}")))?;
                // SAFETY: fresh shared buffer the harness owns.
                unsafe { std::ptr::write_bytes(dst.contents(), 0, size) };
                {
                    let mut blit = cmds
                        .blit_command_encoder()
                        .map_err(|e| at(format!("blit_command_encoder: {e:?}")))?;
                    // SAFETY: `buf` leaves the harness only through the pool's
                    // GPU-clock release or an in-flight hold, the mechanism this
                    // model checks; `dst` is read only after the wait.
                    unsafe { blit.copy_from_buffer(buf, 0, &dst, 0, size) };
                }
                pending.push(Pending {
                    dst,
                    size,
                    expect: *pattern,
                    src_id: *id,
                });
                eng.copies += 1;
                in_flight_copies += 1;
            }
            Op::Drop { index } => {
                if live.is_empty() {
                    continue;
                }
                let k = index % live.len();
                let (buf, _, _) = live.remove(k);
                drop(buf);
                eng.drops += 1;
                if in_flight_copies > 0 {
                    eng.drops_while_in_flight += 1;
                }
            }
            Op::Flush => {
                cmds.flush().map_err(|e| at(format!("flush: {e:?}")))?;
                eng.flushes += 1;
            }
            Op::FlushAndPrune => {
                cmds.flush_and_prune()
                    .map_err(|e| at(format!("flush_and_prune: {e:?}")))?;
                eng.flushes += 1;
            }
            Op::FlushAndWait => {
                cmds.flush_and_wait()
                    .map_err(|e| at(format!("flush_and_wait: {e:?}")))?;
                eng.waits += 1;
                in_flight_copies = 0;
                verify(&mut pending, &mut eng, &at("verify".to_string()))?;
            }
            Op::Drain => {
                let before = pool.occupancy().free_buffers;
                pool.drain_completed();
                if pool.occupancy().free_buffers > before {
                    eng.drains_freeing += 1;
                }
            }
            Op::Trim => {
                let destroyed = pool.trim();
                if !destroyed.is_empty() {
                    eng.trims_destroying += 1;
                    residency.remove_batch(destroyed.iter());
                }
                drop(destroyed);
            }
            Op::Settle => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Op::AllocUnpooled { size } => {
                let size = size.max(16);
                let raw = dev
                    .new_buffer(size, RESOURCE_OPTIONS)
                    .map_err(|e| at(format!("new_buffer(unpooled {size}): {e:?}")))?;
                residency.insert(&raw);
                let buf = BufferHandle::new(
                    PooledBuffer::unpooled_resident(raw, size, Arc::clone(&residency))
                        .held_in_flight(cmds.in_flight_hold()),
                );
                let id = next_id;
                next_id += 1;
                let pattern = ((id * 37) % 200) as u8 + 1;
                // SAFETY: shared storage, fresh: nothing reads it yet.
                unsafe { std::ptr::write_bytes(buf.contents(), pattern, buf.length()) };
                live.push((buf, pattern, id));
                eng.allocs += 1;
                eng.unpooled_allocs += 1;
            }
            Op::ReplaceBase { index } => {
                if let Some(k) = (!bases.is_empty()).then(|| index % bases.len()) {
                    let (old, _, _) = bases.remove(k);
                    lloom_tensor::kernels::metal::in_flight::hold_buffer(
                        cmds.in_flight_hold(),
                        &old,
                    );
                    drop(old);
                    eng.bases_replaced += 1;
                    if in_flight_copies > 0 {
                        eng.drops_while_in_flight += 1;
                    }
                }
                let size = [4096usize, 65536][index % 2];
                let raw = dev
                    .new_buffer(size, RESOURCE_OPTIONS)
                    .map_err(|e| at(format!("new_buffer(base {size}): {e:?}")))?;
                let id = next_id;
                next_id += 1;
                let pattern = ((id * 37) % 200) as u8 + 1;
                // SAFETY: shared storage, fresh: nothing reads it yet.
                unsafe { std::ptr::write_bytes(raw.contents(), pattern, size) };
                bases.push((raw, pattern, id));
            }
            Op::CopyAgain { index } => {
                if live.is_empty() {
                    continue;
                }
                let Some(last) = pending.last_mut() else {
                    continue;
                };
                let (buf, pattern, id) = &live[index % live.len()];
                let size = buf.size().min(last.size);
                {
                    let mut blit = cmds
                        .blit_command_encoder()
                        .map_err(|e| at(format!("blit_command_encoder: {e:?}")))?;
                    // SAFETY: as the copy above.
                    unsafe { blit.copy_from_buffer(buf, 0, &last.dst, 0, size) };
                }
                // The destination now holds this source's pattern over the
                // copied prefix; verify only that prefix.
                last.size = size;
                last.expect = *pattern;
                last.src_id = *id;
                eng.copies += 1;
                eng.copies_again += 1;
                in_flight_copies += 1;
            }
        }
    }

    // Everything in flight completes, and is verified, before the pool and
    // the queue are torn down.
    cmds.flush_and_wait()
        .map_err(|e| format!("final flush_and_wait: {e:?}"))?;
    eng.waits += 1;
    verify(&mut pending, &mut eng, "at the end")?;
    if let Some(hold) = cmds.in_flight_hold() {
        let held = hold.counts();
        eng.held_fences = held.fences;
        eng.held_buffers = held.buffers;
    }
    Ok(eng)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy encoded, its source dropped while the command buffer is still
    /// open, the buffer drained and trimmed after the wait: clean, and every
    /// interesting counter moved.
    #[test]
    fn the_round_trip_is_clean_and_reaches_the_states() {
        let dev = Device::system_default().expect("a Metal device");
        let ops = [
            Op::Alloc { size: 4096 },
            Op::Copy { index: 0 },
            Op::Drop { index: 0 },
            Op::Flush,
            Op::Alloc { size: 4096 },
            Op::Copy { index: 0 },
            Op::FlushAndWait,
            Op::Drain,
            Op::Trim,
        ];
        let e = run_sequence(&dev, &ops).expect("clean");
        assert_eq!(e.copies_verified, 2);
        assert_eq!(e.drops_while_in_flight, 1);
        assert!(e.drains_freeing >= 1, "{e:?}");
        assert!(e.trims_destroying >= 1, "{e:?}");
    }

    /// An unpooled buffer dropped and a base replaced while copies of them are
    /// queued, and a destination written twice so the first encoder's fence
    /// leaves the map: clean on both kinds, and on the unretained kind the
    /// hold took every class — absolute counts, so a hold that never engaged
    /// cannot pass as one with nothing to hold.
    #[test]
    fn the_classes_outside_the_pool_are_held_under_unretained_command_buffers() {
        let dev = Device::system_default().expect("a Metal device");
        let ops = [
            Op::ReplaceBase { index: 0 },
            Op::AllocUnpooled { size: 4096 },
            // Index 1: past the one live buffer, so the base.
            Op::Copy { index: 1 },
            Op::ReplaceBase { index: 0 },
            Op::Copy { index: 0 },
            Op::CopyAgain { index: 0 },
            Op::Drop { index: 0 },
            Op::Flush,
            Op::AllocUnpooled { size: 4096 },
            Op::Copy { index: 0 },
            Op::CopyAgain { index: 0 },
            Op::FlushAndWait,
        ];
        for kind in [CommandBufferKind::Retaining, CommandBufferKind::Unretained] {
            let e = run_sequence_with(&dev, &ops, kind).expect("clean");
            assert_eq!(e.copies_verified, 3, "{kind:?}: {e:?}");
            assert_eq!(e.drops_while_in_flight, 2, "{kind:?}: {e:?}");
            assert_eq!(e.bases_replaced, 1, "{kind:?}: {e:?}");
            match kind {
                CommandBufferKind::Retaining => {
                    assert_eq!((e.held_fences, e.held_buffers), (0, 0), "{e:?}")
                }
                CommandBufferKind::Unretained => {
                    // The replaced base and the dropped unpooled buffer; the
                    // second unpooled buffer is still live at the end.
                    assert_eq!(e.held_buffers, 2, "{e:?}");
                    // Each `CopyAgain` replaces one blit fence in the map.
                    assert!(e.held_fences >= 2, "{e:?}");
                }
            }
        }
    }
}
