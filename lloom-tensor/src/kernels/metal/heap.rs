//! Transient allocations from `MTLHeap` chunks that are in the residency set
//! ONCE (`DESIGN.md` §6.3k, §11.5z-ix item 1).
//!
//! A pool miss used to be an `MTLBuffer` of its own plus a staged
//! `addAllocation`, and every command-buffer cut then committed the residency
//! set — ~0.7 ms a commit over a ~4 000-allocation set, 1.1 s of a 5.2 s
//! streamed prefill (#749 P26). A heap's sub-allocations are resident with
//! the heap, so a miss served from a chunk makes no residency call at all,
//! and an eviction that drops the sub-buffer returns its range to the chunk
//! (Metal's own allocator, `MTLHeapTypeAutomatic`). The chunk is added to the
//! set when it is made: a few commits a process instead of one a cut.
//!
//! What this does NOT change: the persistent allocations (weights, slot
//! pools, KV, the session region) keep the set as before; the pool's own
//! bookkeeping (free budget, pending epochs, eviction order) is untouched —
//! only where a miss's bytes come from. `currentAllocatedSize` counts a
//! chunk at its creation, so the admission gauge (§9.5) reads a conservative
//! figure; `usedSize` beside it is what the sub-buffers hold.

use super::{Buffer, ResidencySet};
use crate::kernels::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLResourceOptions, MTLStorageMode,
};
use std::sync::Arc;

/// The default chunk: 512 MiB. A 512-row prefill tile's transients are
/// ~1.3 GB with two tiles in flight (P18), so a warmed prefill sits in five
/// or six; the last chunk of each list is the air `currentAllocatedSize`
/// counts and the pass admission credits back (`plan_pass`) — at most one
/// chunk a pool. Measured at 1 GiB: 4 chunks, 2.87 GB used of 4.29 (#749
/// P26); `LLOOM_HEAP_CHUNK_MB` overrides.
pub const DEFAULT_HEAP_CHUNK_BYTES: usize = 512 << 20;

/// The chunk size in force (`LLOOM_HEAP_CHUNK_MB`, default 512), read once.
pub fn heap_chunk_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("LLOOM_HEAP_CHUNK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb.max(16) << 20)
            .unwrap_or(DEFAULT_HEAP_CHUNK_BYTES)
    })
}

/// Whether pool misses are served from heap chunks (`LLOOM_TRANSIENT_HEAPS=1`;
/// off by default). Measured on the streamed Qwen4-exp prefill (#749 P26):
/// the seam's residency commit 1 101 → 1.5 ms of host time, the prefill wall
/// 5.19 → 5.02 s (−3 %: the host was not the critical path), and the device
/// footprint +1.0–1.4 GB — a chunk never shrinks and Metal's heap allocator
/// fragments (2.87 GB used of 4.29 at 512 MiB chunks) — which on this
/// machine's budget is the dearer side. On until the chunks are placed by
/// our own allocator (a placement heap), the arm is opt-in. Read once.
pub fn transient_heaps_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LLOOM_TRANSIENT_HEAPS")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Chunks of one storage mode, and the residency set they are added to.
pub struct HeapChunks {
    device: super::Device,
    residency_set: Arc<ResidencySet>,
    options: MTLResourceOptions,
    storage: MTLStorageMode,
    chunk_bytes: usize,
    chunks: parking_lot::Mutex<Vec<Retained<ProtocolObject<dyn MTLHeap>>>>,
}

// SAFETY: an `MTLHeap` is a thread-safe Metal object; the chunk list is
// behind a mutex.
unsafe impl Send for HeapChunks {}
unsafe impl Sync for HeapChunks {}

/// What the chunks hold: their count, their total size, and the bytes their
/// live sub-buffers occupy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub chunks: usize,
    pub size: usize,
    pub used: usize,
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
            chunks: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// A buffer of `length` bytes from the newest chunk that has room, else
    /// from a new chunk (of `chunk_bytes`, or `length` rounded up when the
    /// request is larger). A chunk that cannot serve the request for
    /// fragmentation opens the next one rather than failing.
    pub fn alloc(&self, length: usize) -> Result<Buffer, MetalKernelError> {
        let mut chunks = self.chunks.lock();
        if let Some(raw) = chunks
            .iter()
            .rev()
            .find_map(|h| h.newBufferWithLength_options(length, self.options))
        {
            return Ok(Buffer::new(raw));
        }
        let size = self.chunk_bytes.max(length.next_multiple_of(1 << 20));
        let desc = MTLHeapDescriptor::new();
        desc.setSize(size);
        desc.setStorageMode(self.storage);
        desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
        desc.setType(MTLHeapType::Automatic);
        let heap = self
            .device
            .as_ref()
            .newHeapWithDescriptor(&desc)
            .ok_or_else(|| {
                MetalKernelError::FailedToCreateResource(format!("heap chunk of {size} bytes"))
            })?;
        // Resident with everything it will ever hold: one add, one commit.
        self.residency_set.insert_allocation(
            unsafe {
                &*(&*heap as *const ProtocolObject<dyn MTLHeap>
                    as *const ProtocolObject<dyn objc2_metal::MTLAllocation>)
            },
            Retained::as_ptr(&heap) as usize,
        );
        let raw = heap
            .newBufferWithLength_options(length, self.options)
            .ok_or_else(|| {
                MetalKernelError::FailedToCreateResource(format!(
                    "{length} bytes from a fresh {size}-byte heap chunk"
                ))
            })?;
        chunks.push(heap);
        Ok(Buffer::new(raw))
    }

    pub fn stats(&self) -> HeapStats {
        let chunks = self.chunks.lock();
        HeapStats {
            chunks: chunks.len(),
            size: chunks.iter().map(|h| h.size()).sum(),
            used: chunks.iter().map(|h| h.usedSize()).sum(),
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
