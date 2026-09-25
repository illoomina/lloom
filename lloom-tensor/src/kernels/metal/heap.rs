//! Transient allocations from PLACEMENT `MTLHeap` chunks that are in the
//! residency set ONCE, at offsets this allocator chooses (`DESIGN.md` §6.3k,
//! §11.5z-ix item 1).
//!
//! A pool miss used to be an `MTLBuffer` of its own plus a staged
//! `addAllocation`, and every command-buffer cut then committed the residency
//! set — ~0.7 ms a commit over a ~4 000-allocation set, 1.1 s of a 5.2 s
//! streamed prefill (#749 P26). A heap's sub-allocations are resident with
//! the heap, so a miss served from a chunk makes no residency call at all.
//! The first form let Metal place the sub-buffers (`MTLHeapTypeAutomatic`)
//! and its allocator fragmented under the pool's churn: 2.87 GB used of 4.29
//! allocated. This form places them itself (`MTLHeapTypePlacement`,
//! `newBufferWithLength:options:offset:`): one free-list of ranges per chunk,
//! first-fit with coalescing, the range returned when the pool evicts the
//! buffer (`release`) — so the chunks hold what the pool holds and no more.
//!
//! What this does NOT change: the persistent allocations (weights, slot
//! pools, KV, the session region) keep the set as before; the pool's own
//! bookkeeping (free budget, pending epochs, eviction order) is untouched —
//! only where a miss's bytes come from and where an eviction's go.
//! `currentAllocatedSize` counts a chunk at its creation, so the admission
//! gauge (§9.5) reads a conservative figure; the pass admission credits the
//! chunks' free bytes back (`plan_pass`).
//!
//! Aliasing: a placement heap does not track it. A range is handed out once
//! and comes back only from the pool's eviction of a buffer whose GPU epoch
//! has completed (the pool evicts from its FREE list), so no two live
//! buffers overlap and no returned range is still read.

use super::{Buffer, ResidencySet};
use crate::kernels::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLResourceOptions, MTLStorageMode,
};
use std::collections::HashMap;
use std::sync::Arc;

/// The default chunk: 512 MiB. A 512-row prefill tile's transients are
/// ~1.3 GB with two tiles in flight (P18), so a warmed prefill sits in three
/// to six; the air is the tail of the last chunk of each list.
/// `LLOOM_HEAP_CHUNK_MB` overrides.
pub const DEFAULT_HEAP_CHUNK_BYTES: usize = 512 << 20;

/// The chunk size in force (`LLOOM_HEAP_CHUNK_MB`, default 512), read once.
pub fn heap_chunk_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| crate::tuning::tuning().metal.heap_chunk_mb.max(16) << 20)
}

/// Whether pool misses are served from heap chunks (`LLOOM_TRANSIENT_HEAPS=1`;
/// off by default — the per-buffer arm is what every recorded digest ran
/// on). Measured under the paired prefill (#749 P26): the seam's residency
/// commit 1 101 → 1.5 ms of host time, the wall −3–5 %, and the device
/// footprint **+0.9 GB at the peak, +1.85 GB retained after it** — the
/// chunks hold the pass's peak transient set and the holes the churn leaves
/// in it (2.89 GB used of 4.83 at 512 MiB chunks, best fit; Metal's own
/// automatic placement packed 2.87 of 4.29), where the per-buffer arm frees
/// an evicted buffer to the OS. On this machine's budget the peak is the
/// dearer side; the arm is opt-in, and the fix for the air is structural —
/// a per-pass bump region for the prefill's transients (§9.2's arena, at
/// prefill), which these chunks are the place for. Read once.
pub fn transient_heaps_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().metal.transient_heaps)
}

pub const DEFAULT_TRANSIENT_HEAPS: bool = false;

/// One placement chunk: the heap and its free ranges, `(offset, len)`,
/// sorted by offset and coalesced.
struct Chunk {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    size: usize,
    free: Vec<(usize, usize)>,
    used: usize,
}

impl Chunk {
    /// The smallest free range that fits `size` at `align` (best fit: the
    /// big holes are kept for the big requests), and its waste after the
    /// piece is taken.
    fn fit(&self, size: usize, align: usize) -> Option<(usize, usize)> {
        let align = align.max(1);
        let mut best: Option<(usize, usize)> = None;
        for (i, &(off, len)) in self.free.iter().enumerate() {
            let start = off.next_multiple_of(align);
            if start + size <= off + len {
                let waste = len - size;
                if best.is_none_or(|(_, w)| waste < w) {
                    best = Some((i, waste));
                }
            }
        }
        best
    }

    /// Take `size` at `align` from free range `i` (from [`Self::fit`]): the
    /// buffer's offset.
    fn take_at(&mut self, i: usize, size: usize, align: usize) -> usize {
        let align = align.max(1);
        let (off, len) = self.free[i];
        let start = off.next_multiple_of(align);
        // What is left of the range before and after the taken piece.
        let mut rest = Vec::with_capacity(2);
        if start > off {
            rest.push((off, start - off));
        }
        let end = start + size;
        if off + len > end {
            rest.push((end, off + len - end));
        }
        self.free.splice(i..i + 1, rest);
        self.used += size;
        start
    }

    fn take(&mut self, size: usize, align: usize) -> Option<usize> {
        let (i, _) = self.fit(size, align)?;
        Some(self.take_at(i, size, align))
    }

    /// Give `(offset, size)` back, coalescing with its neighbours.
    fn give_back(&mut self, offset: usize, size: usize) {
        let i = self.free.partition_point(|&(off, _)| off < offset);
        self.free.insert(i, (offset, size));
        // Merge with the next, then with the previous.
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            let next = self.free.remove(i + 1);
            self.free[i].1 += next.1;
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            let cur = self.free.remove(i);
            self.free[i - 1].1 += cur.1;
        }
        self.used = self.used.saturating_sub(size);
    }
}

/// Where a live sub-buffer sits: its chunk and its range.
#[derive(Clone, Copy)]
struct Placement {
    chunk: usize,
    offset: usize,
    size: usize,
}

struct State {
    chunks: Vec<Chunk>,
    /// Keyed by the `MTLBuffer` pointer (`Buffer::raw_ptr`).
    live: HashMap<usize, Placement>,
}

/// Chunks of one storage mode, and the residency set they are added to.
pub struct HeapChunks {
    device: super::Device,
    residency_set: Arc<ResidencySet>,
    options: MTLResourceOptions,
    storage: MTLStorageMode,
    chunk_bytes: usize,
    state: parking_lot::Mutex<State>,
}

// SAFETY: an `MTLHeap` is a thread-safe Metal object; the state is behind a
// mutex.
unsafe impl Send for HeapChunks {}
unsafe impl Sync for HeapChunks {}

/// What the chunks hold: their count, their total size, the bytes their live
/// sub-buffers occupy, and the live sub-buffer count.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub chunks: usize,
    pub size: usize,
    pub used: usize,
    pub live: usize,
}

impl HeapChunks {
    /// Chunks for buffers of `options` (their storage mode is the heaps'),
    /// each `chunk_bytes` unless a request is larger.
    pub fn new(
        device: &super::Device,
        residency_set: Arc<ResidencySet>,
        options: MTLResourceOptions,
        chunk_bytes: usize,
    ) -> Self {
        // The storage mode is bits 4..8 of the resource options.
        let storage = MTLStorageMode((options.0 >> 4) & 0xF);
        Self {
            device: device.clone(),
            residency_set,
            options,
            storage,
            chunk_bytes: chunk_bytes.max(1 << 20),
            state: parking_lot::Mutex::new(State {
                chunks: Vec::new(),
                live: HashMap::new(),
            }),
        }
    }

    /// A buffer of `length` bytes at a free range of some chunk (first fit,
    /// oldest chunk first), else in a new chunk (of `chunk_bytes`, or the
    /// request's size rounded up when larger).
    pub fn alloc(&self, length: usize) -> Result<Buffer, MetalKernelError> {
        let sa = self
            .device
            .as_ref()
            .heapBufferSizeAndAlignWithLength_options(length.max(1), self.options);
        let (size, align) = (sa.size, sa.align.max(1));
        let mut st = self.state.lock();
        // Best fit across every chunk.
        let mut best: Option<(usize, usize, usize)> = None;
        for (ci, c) in st.chunks.iter().enumerate() {
            if let Some((ri, waste)) = c.fit(size, align) {
                if best.is_none_or(|(_, _, w)| waste < w) {
                    best = Some((ci, ri, waste));
                }
            }
        }
        let placed = best.map(|(ci, ri, _)| {
            let off = st.chunks[ci].take_at(ri, size, align);
            (ci, off)
        });
        let (chunk, offset) = match placed {
            Some(p) => p,
            None => {
                let chunk_size = self.chunk_bytes.max(size.next_multiple_of(align));
                let desc = MTLHeapDescriptor::new();
                desc.setSize(chunk_size);
                desc.setStorageMode(self.storage);
                desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
                desc.setType(MTLHeapType::Placement);
                let heap = self
                    .device
                    .as_ref()
                    .newHeapWithDescriptor(&desc)
                    .ok_or_else(|| {
                        MetalKernelError::FailedToCreateResource(format!(
                            "placement heap chunk of {chunk_size} bytes"
                        ))
                    })?;
                // Resident with everything it will ever hold: one add, one
                // commit.
                self.residency_set.insert_allocation(
                    unsafe {
                        &*(&*heap as *const ProtocolObject<dyn MTLHeap>
                            as *const ProtocolObject<dyn objc2_metal::MTLAllocation>)
                    },
                    Retained::as_ptr(&heap) as usize,
                );
                let size_actual = heap.size();
                st.chunks.push(Chunk {
                    heap,
                    size: size_actual,
                    free: vec![(0, size_actual)],
                    used: 0,
                });
                let i = st.chunks.len() - 1;
                let off = st.chunks[i].take(size, align).ok_or_else(|| {
                    MetalKernelError::FailedToCreateResource(format!(
                        "{size} bytes at align {align} from a fresh {size_actual}-byte chunk"
                    ))
                })?;
                (i, off)
            }
        };
        // SAFETY: `[offset, offset + size)` is inside the chunk (the free
        // list is the chunk's own ranges) and overlaps no live buffer (a
        // range is handed out once and returns only from `release`).
        let raw = unsafe {
            st.chunks[chunk]
                .heap
                .newBufferWithLength_options_offset(size, self.options, offset)
        };
        let raw = match raw {
            Some(r) => r,
            None => {
                st.chunks[chunk].give_back(offset, size);
                return Err(MetalKernelError::FailedToCreateResource(format!(
                    "{size} bytes at offset {offset} of a {}-byte placement chunk",
                    st.chunks[chunk].size
                )));
            }
        };
        let buffer = Buffer::new(raw);
        st.live.insert(
            buffer.raw_ptr() as usize,
            Placement {
                chunk,
                offset,
                size,
            },
        );
        Ok(buffer)
    }

    /// Give a sub-buffer's range back (the pool evicting it, or a trim).
    /// Returns whether `buf` was one of ours; a buffer that is not is left
    /// to its own release.
    pub fn release(&self, buf: &Buffer) -> bool {
        let mut st = self.state.lock();
        let Some(p) = st.live.remove(&(buf.raw_ptr() as usize)) else {
            return false;
        };
        st.chunks[p.chunk].give_back(p.offset, p.size);
        true
    }

    pub fn stats(&self) -> HeapStats {
        let st = self.state.lock();
        HeapStats {
            chunks: st.chunks.len(),
            size: st.chunks.iter().map(|c| c.size).sum(),
            used: st.chunks.iter().map(|c| c.used).sum(),
            live: st.live.len(),
        }
    }
}

impl std::fmt::Debug for HeapChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HeapChunks {{ {:?}, chunk_bytes: {} }}",
            self.stats(),
            self.chunk_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    /// The range arithmetic alone, with no heap behind it.
    #[derive(Default)]
    struct Ranges {
        free: Vec<(usize, usize)>,
        used: usize,
    }

    impl Ranges {
        fn take(&mut self, size: usize, align: usize) -> Option<usize> {
            let mut pick = None;
            for (i, &(off, len)) in self.free.iter().enumerate() {
                let start = off.next_multiple_of(align);
                if start + size <= off + len {
                    pick = Some((i, off, len, start));
                    break;
                }
            }
            let (i, off, len, start) = pick?;
            let mut rest = Vec::with_capacity(2);
            if start > off {
                rest.push((off, start - off));
            }
            let end = start + size;
            if off + len > end {
                rest.push((end, off + len - end));
            }
            self.free.splice(i..i + 1, rest);
            self.used += size;
            Some(start)
        }
        fn give_back(&mut self, offset: usize, size: usize) {
            let i = self.free.partition_point(|&(off, _)| off < offset);
            self.free.insert(i, (offset, size));
            if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
                let next = self.free.remove(i + 1);
                self.free[i].1 += next.1;
            }
            if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
                let cur = self.free.remove(i);
                self.free[i - 1].1 += cur.1;
            }
            self.used = self.used.saturating_sub(size);
        }
    }

    /// The same arithmetic as `Chunk::take` / `give_back` (a copy, since a
    /// `Chunk` holds a heap): split at alignment, coalesce both ways, reuse.
    #[test]
    fn ranges_split_align_and_coalesce() {
        let mut c = Ranges {
            free: vec![(0, 1 << 20)],
            used: 0,
        };
        let a = c.take(1000, 256).unwrap();
        let b = c.take(1000, 256).unwrap();
        let d = c.take(1000, 256).unwrap();
        assert_eq!((a, b, d), (0, 1024, 2048));
        assert_eq!(c.used, 3000);
        c.give_back(b, 1000);
        assert_eq!(c.free.len(), 2, "{:?}", c.free);
        c.give_back(a, 1000);
        c.give_back(d, 1000);
        assert_eq!(c.free, vec![(0, 1 << 20)]);
        assert_eq!(c.used, 0);
        let x = c.take(1000, 256).unwrap();
        assert_eq!(x, 0);
        let y = c.take(3000, 4096).unwrap();
        assert_eq!(y, 4096);
    }
}
