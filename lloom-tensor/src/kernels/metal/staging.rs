//! A ring of host-writable staging buffers for a streamed weight tier: the
//! host fills one, a compute-encoder copy lands it in its slot, and the
//! buffer is reused once the command buffer holding that copy has completed
//! — the `depth+1` discipline of `design/docs/expert-streaming.md` §20.4a,
//! gated on the GPU clock (§6.3b) with a relaxed atomic load and never a
//! wait: a ring with nothing complete hands out a fresh buffer instead.
//!
//! Outside every pool on purpose: a pooled host upload is the tenant class
//! whose reuse gate leaks at long context (`invariants.md` §15.1m), and a
//! streamed tier writes hundreds of them per token.
//!
//! Bounded: the ring keeps at most `cap_bytes` of buffers. Past the cap a
//! fresh buffer is a one-shot, held from its stamp until its epoch completes
//! and freed then — never dropped while a command buffer may still read it,
//! so the liveness holds under `LLOOM_CB_UNRETAINED=1` as well (the
//! unretained form rests on OUR discipline, see `commands::cb_unretained`).
//! A prefill burst thus costs its transient bytes once; steady-state decode
//! allocates nothing.
//!
//! Acquisition is O(size classes), not O(entries) (CONTRIBUTING §3.3): the
//! entries of one size wait in the order their copies were stamped, and the
//! clock completes epochs as a watermark, so if the oldest stamp of a class
//! is not complete no later one is.

use super::buffer_pool::{GpuClock, PooledBuffer};
use crate::{MetalDevice, Result};
use std::collections::VecDeque;
use std::sync::Arc;

/// An entry acquired and not yet stamped: no epoch completes it.
const IN_USE: u64 = u64::MAX;

struct Entry {
    buf: Arc<PooledBuffer>,
    /// The epoch of the command buffer the last copy from this entry was
    /// encoded into (0 once released unused); reusable once the clock shows
    /// it complete.
    epoch: u64,
    /// Bumped on every change of `epoch`: a ticket for an older state is
    /// stale and is dropped, never honoured.
    gen: u64,
    /// Index into `StagingRing::classes`.
    class: usize,
}

/// A claim that `entries[idx]` is reusable once its epoch completes, live
/// while the entry's `gen` still matches.
#[derive(Clone, Copy)]
struct Ticket {
    idx: usize,
    gen: u64,
}

/// The entries of one buffer size that are waiting to be reused.
struct Class {
    size: usize,
    /// Stamped entries in stamp order. Stamps read the clock's current
    /// epoch, which only grows, so the epochs here are non-decreasing.
    stamped: VecDeque<Ticket>,
    /// Entries handed back unused (epoch 0).
    released: Vec<Ticket>,
}

/// Which list of a class holds its first reusable entry.
#[derive(Clone, Copy)]
enum Ready {
    Released,
    Stamped,
}

impl Class {
    /// The list holding a live ticket whose epoch is complete, dropping the
    /// stale tickets in front of it. Released entries go first: nothing on
    /// the GPU reads them.
    fn ready(&mut self, entries: &[Entry], clock: &GpuClock, probes: &mut u64) -> Option<Ready> {
        let live = |t: &Ticket| entries[t.idx].gen == t.gen;
        while let Some(t) = self.released.last() {
            *probes += 1;
            if live(t) {
                if clock.is_complete(entries[t.idx].epoch) {
                    return Some(Ready::Released);
                }
                break;
            }
            self.released.pop();
        }
        while let Some(t) = self.stamped.front() {
            *probes += 1;
            if live(t) {
                return clock
                    .is_complete(entries[t.idx].epoch)
                    .then_some(Ready::Stamped);
            }
            self.stamped.pop_front();
        }
        None
    }

    fn take(&mut self, from: Ready) -> Ticket {
        match from {
            Ready::Released => self.released.pop(),
            Ready::Stamped => self.stamped.pop_front(),
        }
        .expect("`ready` found a ticket in this list")
    }
}

pub struct StagingRing {
    entries: Vec<Entry>,
    classes: Vec<Class>,
    /// Stamped one-shots past the cap, in stamp order, freed once their
    /// epoch completes.
    overflow: VecDeque<(Arc<PooledBuffer>, u64)>,
    /// Bytes of `entries`.
    bytes: usize,
    cap_bytes: usize,
    clock: Arc<GpuClock>,
    allocations: u64,
    reuses: u64,
    overflows: u64,
    probes: u64,
}

/// A ring shared by every layer of a streamed tier: one bound, one clock.
pub type SharedStagingRing = Arc<parking_lot::Mutex<StagingRing>>;

/// Bytes a ring keeps by default: a decode step's loads several times over
/// (a miss is three ~0.6 MB planes on Qwen4-exp; ~30 per layer at 40 % hit
/// over 48 layers ≈ 0.5 GB per token).
pub const DEFAULT_CAP_BYTES: usize = 1 << 30;

impl StagingRing {
    pub fn new(device: &MetalDevice) -> Self {
        Self::with_cap_bytes(device, DEFAULT_CAP_BYTES)
    }

    pub fn with_cap_bytes(device: &MetalDevice, cap_bytes: usize) -> Self {
        Self::with_clock(device.gpu_clock(), cap_bytes)
    }

    fn with_clock(clock: Arc<GpuClock>, cap_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            classes: Vec::new(),
            overflow: VecDeque::new(),
            bytes: 0,
            cap_bytes,
            clock,
            allocations: 0,
            reuses: 0,
            overflows: 0,
            probes: 0,
        }
    }

    pub fn shared(device: &MetalDevice) -> SharedStagingRing {
        Arc::new(parking_lot::Mutex::new(Self::new(device)))
    }

    /// [`Self::shared`] at `cap_bytes` (a tier with a prefetcher wants room
    /// for what its reader threads stage ahead, `nn::expert_prefetch`).
    pub fn shared_with_cap(device: &MetalDevice, cap_bytes: usize) -> SharedStagingRing {
        Arc::new(parking_lot::Mutex::new(Self::with_cap_bytes(
            device, cap_bytes,
        )))
    }

    /// The cap the ring was built with.
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// A Shared buffer of at least `len` bytes whose last copy has completed
    /// — the smallest size that has one — or a fresh one, as `(entry,
    /// buffer)`. The entry is in use — never reusable — until [`Self::stamp`]
    /// records the epoch of the command buffer its copy was encoded into. At
    /// the cap with nothing complete the buffer is a one-shot (`None`), held
    /// from its stamp until its epoch completes; nobody waits.
    pub fn acquire(
        &mut self,
        device: &MetalDevice,
        len: usize,
    ) -> Result<(Option<usize>, Arc<PooledBuffer>)> {
        // A ring entry is the tier's, not a pass's: outside any pass plan.
        let _escape = super::arena::EscapeGuard::new();
        self.free_overflow();
        if let Some(idx) = self.take_reusable(len) {
            let e = &mut self.entries[idx];
            e.epoch = IN_USE;
            e.gen += 1;
            self.reuses += 1;
            return Ok((Some(idx), Arc::clone(&e.buf)));
        }
        let buf = device.new_unpooled_shared(len)?;
        if self.bytes + buf.size() > self.cap_bytes && !self.entries.is_empty() {
            self.overflows += 1;
            return Ok((None, buf));
        }
        let class = self.class_of(buf.size());
        self.bytes += buf.size();
        self.entries.push(Entry {
            buf: Arc::clone(&buf),
            epoch: IN_USE,
            gen: 0,
            class,
        });
        self.allocations += 1;
        Ok((Some(self.entries.len() - 1), buf))
    }

    /// Record the epoch the copy from `buf` was encoded into — a ring entry
    /// by index, a one-shot by its buffer. Read the clock AFTER taking the
    /// compute encoder for the copy: a command-buffer cut between the
    /// acquisition and the encode would otherwise leave the entry stamped one
    /// epoch early, and it would be reused while its copy is still pending.
    ///
    /// Stamps are expected in non-decreasing epoch order (the clock's current
    /// epoch, read under the ring's lock); an older epoch stamped behind a
    /// newer one is reused no earlier than the newer one, never early.
    pub fn stamp(&mut self, entry: Option<usize>, buf: &Arc<PooledBuffer>, epoch: u64) {
        match entry {
            Some(i) => {
                let e = &mut self.entries[i];
                debug_assert!(
                    Arc::ptr_eq(&e.buf, buf),
                    "stamp: entry {i} is another buffer"
                );
                e.epoch = epoch;
                e.gen += 1;
                let t = Ticket { idx: i, gen: e.gen };
                self.classes[e.class].stamped.push_back(t);
            }
            None => self.overflow.push_back((Arc::clone(buf), epoch)),
        }
    }

    /// Hand an acquired entry back UNUSED — no copy was encoded from it (a
    /// prefetch the layer never needed). A ring entry is reusable at once
    /// (epoch 0: nothing reads it); a one-shot is the caller's to drop,
    /// since the ring holds one only once it is stamped.
    pub fn release(&mut self, entry: Option<usize>, _buf: &Arc<PooledBuffer>) {
        if let Some(i) = entry {
            let e = &mut self.entries[i];
            e.epoch = 0;
            e.gen += 1;
            let t = Ticket { idx: i, gen: e.gen };
            self.classes[e.class].released.push(t);
        }
    }

    /// The smallest class of at least `len` bytes with an entry whose copy
    /// is complete, and that entry, taken off its list.
    fn take_reusable(&mut self, len: usize) -> Option<usize> {
        let mut best: Option<(usize, Ready)> = None;
        for c in 0..self.classes.len() {
            let size = self.classes[c].size;
            if size < len || best.is_some_and(|(b, _)| self.classes[b].size <= size) {
                continue;
            }
            if let Some(r) = self.classes[c].ready(&self.entries, &self.clock, &mut self.probes) {
                best = Some((c, r));
            }
        }
        best.map(|(c, r)| self.classes[c].take(r).idx)
    }

    fn class_of(&mut self, size: usize) -> usize {
        if let Some(c) = self.classes.iter().position(|c| c.size == size) {
            return c;
        }
        self.classes.push(Class {
            size,
            stamped: VecDeque::new(),
            released: Vec::new(),
        });
        self.classes.len() - 1
    }

    fn free_overflow(&mut self) {
        while self
            .overflow
            .front()
            .is_some_and(|(_, epoch)| self.clock.is_complete(*epoch))
        {
            self.overflow.pop_front();
        }
    }

    /// The clock the ring gates on.
    pub fn clock(&self) -> &Arc<GpuClock> {
        &self.clock
    }

    /// `(buffers allocated, acquisitions served by reuse, one-shots past the cap)`.
    pub fn counters(&self) -> (u64, u64, u64) {
        (self.allocations, self.reuses, self.overflows)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes the ring keeps.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// One-shots still held for a pending epoch.
    pub fn overflow_pending(&self) -> usize {
        self.overflow.len()
    }

    /// Tickets examined by acquisitions so far: at most the size classes
    /// per acquisition, plus each stale ticket once.
    pub fn probes(&self) -> u64 {
        self.probes
    }
}

impl std::fmt::Debug for StagingRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StagingRing {{ buffers: {}, allocations: {}, reuses: {}, overflows: {} }}",
            self.entries.len(),
            self.allocations,
            self.reuses,
            self.overflows
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use std::collections::HashMap;

    fn wait_gpu(dev: &MetalDevice) {
        // Something must be encoded for the epoch to close and complete.
        let t = crate::Tensor::zeros(16, crate::DType::F32, &Device::Metal(dev.clone())).unwrap();
        let _ = t.to_vec1::<f32>().unwrap(); // commits and waits
    }

    fn metal() -> Option<MetalDevice> {
        match Device::new_metal(0) {
            Ok(Device::Metal(m)) => Some(m),
            _ => None,
        }
    }

    /// Two acquisitions inside one open command buffer allocate twice; once
    /// that buffer has completed, the ring serves the next from the first
    /// entry without allocating; an unstamped entry is never reused.
    #[test]
    fn reuse_waits_for_the_epoch_and_never_blocks() {
        let Some(dev) = metal() else { return };
        let mut ring = StagingRing::with_cap_bytes(&dev, 4096 * 2 + (1 << 20));
        let epoch = ring.clock().current_epoch();
        let (ia, a) = ring.acquire(&dev, 4096).unwrap();
        let (ib, b) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(
            ring.counters(),
            (2, 0, 0),
            "an open epoch serves nothing by reuse"
        );
        assert!(!Arc::ptr_eq(&a, &b));
        ring.stamp(ia, &a, epoch);
        wait_gpu(&dev); // `epoch` is complete; b is still unstamped
        let (ic, c) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(
            ring.counters(),
            (2, 1, 0),
            "the stamped, completed entry is reused"
        );
        assert!(Arc::ptr_eq(&a, &c) && ic == ia);
        ring.stamp(ib, &b, epoch);
        ring.stamp(ic, &c, ring.clock().current_epoch());
        let (_, d) = ring.acquire(&dev, 4096).unwrap();
        assert!(
            Arc::ptr_eq(&b, &d),
            "b, stamped complete, is the next reuse"
        );
        // a larger request never fits an old entry: the last that fits the cap
        let (ie, e) = ring.acquire(&dev, 1 << 20).unwrap();
        assert_eq!(ring.counters().0, 3);
        assert!(ie.is_some());
        ring.stamp(ie, &e, ring.clock().current_epoch());
        // at the cap with nothing complete: a one-shot, held for its epoch
        let (i_over, f) = ring.acquire(&dev, 4096).unwrap();
        assert!(i_over.is_none());
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.counters().2, 1);
        ring.stamp(i_over, &f, ring.clock().current_epoch());
        assert_eq!(ring.overflow_pending(), 1);
        wait_gpu(&dev);
        // its epoch complete, the one-shot is released on the next acquire
        let (ig, _g) = ring.acquire(&dev, 4096).unwrap();
        assert!(ig.is_some(), "a completed entry serves it, not a one-shot");
        assert_eq!(ring.overflow_pending(), 0);
        assert_eq!(ring.len(), 3);
    }

    /// The invariant the ring exists for, under a random schedule of
    /// acquisitions, stamps, releases, commits and completions on a clock the
    /// test drives: a buffer comes back from `acquire` only once the epoch of
    /// its last stamp is complete, never while another holder has it, and a
    /// stamped one-shot lives until its epoch completes and no longer.
    #[test]
    fn a_buffer_is_never_reused_before_its_copy_completes() {
        let Some(dev) = metal() else { return };
        let clock = Arc::new(GpuClock::new());
        // Room for eight 4 KiB entries: the schedule runs into the cap.
        let mut ring = StagingRing::with_clock(Arc::clone(&clock), 8 * 4096);
        let sizes = [4096usize, 4096, 3000, 8192];
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let key = |b: &Arc<PooledBuffer>| Arc::as_ptr(b) as usize;
        let mut held: Vec<(Option<usize>, Arc<PooledBuffer>)> = Vec::new();
        // Buffer -> the epoch a reuse must wait for (0 once released).
        let mut last_stamp: HashMap<usize, u64> = HashMap::new();
        let mut one_shots: Vec<(std::sync::Weak<PooledBuffer>, u64)> = Vec::new();
        let (mut reused_early, mut reused_ok) = (0, 0);
        for _ in 0..4000 {
            match next(10) {
                // At most six held: the ring lives near its cap, not past it.
                0..=3 if held.len() < 6 => {
                    let len = sizes[next(sizes.len() as u64) as usize];
                    let (entry, buf) = ring.acquire(&dev, len).unwrap();
                    assert!(buf.size() >= len);
                    assert!(
                        held.iter().all(|(_, b)| !Arc::ptr_eq(b, &buf)),
                        "a buffer another holder has was handed out"
                    );
                    if let Some(epoch) = last_stamp.remove(&key(&buf)) {
                        if clock.is_complete(epoch) {
                            reused_ok += 1;
                        } else {
                            reused_early += 1;
                        }
                    }
                    held.push((entry, buf));
                }
                0..=5 if !held.is_empty() => {
                    let (entry, buf) = held.swap_remove(next(held.len() as u64) as usize);
                    let epoch = clock.current_epoch();
                    ring.stamp(entry, &buf, epoch);
                    match entry {
                        Some(_) => {
                            last_stamp.insert(key(&buf), epoch);
                        }
                        None => one_shots.push((Arc::downgrade(&buf), epoch)),
                    }
                }
                6 if !held.is_empty() => {
                    let (entry, buf) = held.swap_remove(next(held.len() as u64) as usize);
                    ring.release(entry, &buf);
                    if entry.is_some() {
                        last_stamp.insert(key(&buf), 0);
                    }
                }
                7..=8 => {
                    clock.commit_epoch();
                }
                _ => {
                    let (done, open) = (clock.completed_count(), clock.current_epoch());
                    if done < open {
                        clock.mark_completed(done + next(open - done));
                    }
                }
            }
            for (w, epoch) in &one_shots {
                if !clock.is_complete(*epoch) {
                    assert!(
                        w.upgrade().is_some(),
                        "a one-shot was freed before its copy completed"
                    );
                }
            }
        }
        assert_eq!(
            reused_early, 0,
            "{reused_early} buffers reused before their copy completed"
        );
        let (allocations, reuses, overflows) = ring.counters();
        eprintln!(
            "allocations={allocations} reuses={reuses} overflows={overflows} \
             reused_after_completion={reused_ok} one_shots={}",
            one_shots.len()
        );
        assert!(
            reused_ok > 100 && overflows > 10,
            "the schedule must reuse and overflow"
        );

        // Everything stamped and complete: the next acquisition of each size
        // allocates nothing, and every one-shot is freed.
        for (entry, buf) in held.drain(..) {
            ring.stamp(entry, &buf, clock.current_epoch());
        }
        clock.commit_epoch();
        clock.mark_completed(clock.current_epoch() - 1);
        for len in [4096, 3000, 8192] {
            let (entry, buf) = ring.acquire(&dev, len).unwrap();
            assert!(entry.is_some());
            ring.release(entry, &buf);
        }
        assert_eq!(ring.counters().0, allocations, "a complete ring reuses");
        assert_eq!(ring.overflow_pending(), 0);
        assert!(
            one_shots.iter().all(|(w, _)| w.upgrade().is_none()),
            "a one-shot leaked"
        );
    }

    /// A released entry reads as a stamp at epoch 0: reusable once the clock
    /// has completed anything, and no sooner.
    #[test]
    fn a_released_entry_is_reused_as_epoch_zero() {
        let Some(dev) = metal() else { return };
        let clock = Arc::new(GpuClock::new());
        let mut ring = StagingRing::with_clock(Arc::clone(&clock), 1 << 20);
        let (ia, a) = ring.acquire(&dev, 4096).unwrap();
        ring.release(ia, &a);
        let (_, b) = ring.acquire(&dev, 4096).unwrap();
        assert!(
            !Arc::ptr_eq(&a, &b),
            "nothing has completed: epoch 0 is pending"
        );
        clock.commit_epoch();
        clock.mark_completed(0);
        let (ic, c) = ring.acquire(&dev, 4096).unwrap();
        assert!(
            Arc::ptr_eq(&a, &c) && ic == ia,
            "epoch 0 complete: a is reused"
        );
    }

    /// An acquisition with every entry pending examines one ticket per size
    /// class, whatever the ring holds.
    #[test]
    fn acquire_does_not_scan_the_ring() {
        let Some(dev) = metal() else { return };
        let clock = Arc::new(GpuClock::new());
        let mut ring = StagingRing::with_clock(Arc::clone(&clock), 1 << 30);
        for i in 0..2000 {
            let len = if i % 2 == 0 { 4096 } else { 8192 };
            let (entry, buf) = ring.acquire(&dev, len).unwrap();
            ring.stamp(entry, &buf, clock.current_epoch());
        }
        assert_eq!(ring.len(), 2000);
        let before = ring.probes;
        let (entry, buf) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(ring.len(), 2001, "nothing complete: a fresh entry");
        assert!(
            ring.probes - before <= 2,
            "{} tickets examined for one acquisition",
            ring.probes - before
        );
        ring.stamp(entry, &buf, clock.current_epoch());
        clock.commit_epoch();
        clock.mark_completed(clock.current_epoch() - 1);
        let before = ring.probes;
        let (entry, _) = ring.acquire(&dev, 4096).unwrap();
        assert_eq!(entry, Some(0), "the oldest stamp of the class is reused");
        assert!(ring.probes - before <= 2);
    }
}
