//! Does Metal on this GPU offer a sparse buffer whose pages are mapped on
//! demand and which a kernel addresses from one base? (`memory-system-plan.md`
//! §4.4, R4; decision 108.)
//!
//! Raw `objc2-metal`, no engine wrapper, so what is measured is the platform:
//!
//! * `inventory` — the device's sparse capabilities: GPU families,
//!   `supportsPlacementSparse`, the sparse tile sizes, `maxBufferLength`, the
//!   default `maxCompatiblePlacementSparsePageSize` of a heap descriptor.
//! * `function` — create a placement sparse buffer, map ranges of it from two
//!   placement heaps (`MTL4CommandQueue::updateBufferMappings`), write them
//!   with a kernel that addresses the whole virtual range from one base,
//!   read them back on both command paths (Metal 4 and Metal 3), read an
//!   unmapped page (only where the buffer reports tier 1, which defines it),
//!   unmap and remap a tile at another virtual offset, order a Metal 3
//!   reader behind a Metal 4 mapping by an event, unmap while a reader is in
//!   flight, and give a heap's memory back. Footprint and wired bytes at
//!   every step.
//! * `cost` — the latency of mapping and unmapping N tiles, per page size,
//!   contiguous (one operation) and scattered (one operation per tile),
//!   against a null control (the event signal alone) and a dispatch control.
//! * `sparse-heap-buffer` — whether the Metal 3 sparse heap
//!   (`MTLHeapTypeSparse`) hands out a buffer. Its own subcommand because the
//!   call may raise.
//!
//! ```bash
//! lloom-arb build --priority 3 -- ./target/release/examples/sparse_buffer_probe inventory
//! lloom-arb measure -- ./target/release/examples/sparse_buffer_probe cost --reps 20
//! ```

use anyhow::{bail, ensure, Context, Result};
use block2::RcBlock;
use clap::{Parser, Subcommand};
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTL4UpdateSparseBufferMappingOperation, MTL4VisibilityOptions,
    MTLAllocation, MTLBuffer, MTLBufferSparseTier, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLHazardTrackingMode, MTLHeap,
    MTLHeapDescriptor, MTLHeapType, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLResourceOptions, MTLSharedEvent, MTLSize, MTLSparsePageSize, MTLSparseTextureMappingMode,
    MTLStages, MTLStorageMode,
};
use parking_lot::Mutex;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Heap = Retained<ProtocolObject<dyn MTLHeap>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;
type Enc4 = ProtocolObject<dyn MTL4ComputeCommandEncoder>;
type Enc3 = ProtocolObject<dyn MTLComputeCommandEncoder>;

#[derive(Parser, Debug)]
#[command(about = "Placement sparse buffers on this GPU (memory-system-plan.md §4.4, R4)")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// The device's sparse capabilities.
    Inventory,
    /// Create, map, write, read, unmap, remap; both command paths.
    Function {
        /// Virtual length of the sparse buffer, GiB.
        #[arg(long, default_value_t = 16)]
        virt_gib: usize,
        /// Size of each of the two placement heaps, MiB.
        #[arg(long, default_value_t = 256)]
        heap_mib: usize,
        /// The system's wired bytes may grow by at most this much, GiB.
        #[arg(long, default_value_t = 6)]
        wired_cap_gib: u64,
    },
    /// Map/unmap latency per page size and tile count.
    Cost {
        #[arg(long, default_value_t = 20)]
        reps: usize,
        /// Tile counts to map and unmap.
        #[arg(long, value_delimiter = ',', default_value = "1,16,256,1024")]
        tiles: Vec<usize>,
        /// Page sizes, KiB.
        #[arg(long, value_delimiter = ',', default_value = "16,64,256")]
        page_kib: Vec<usize>,
    },
    /// Ask a Metal 3 sparse heap (`MTLHeapTypeSparse`) for a buffer.
    SparseHeapBuffer,
    /// Does an unmap take effect when a reader of the tile is in flight, or
    /// when the heap tile is also mapped at another virtual tile?
    UnmapHazard {
        #[arg(long, default_value_t = 3)]
        trials: usize,
        /// Reader length in loop iterations per thread (~85 ms at 524288).
        #[arg(long, default_value_t = 524288)]
        iters: u32,
        /// Delay from the reader's commit to the unmap, ms.
        #[arg(long, default_value_t = 20)]
        delay_ms: u64,
    },
    /// A heap released while an unmap of its tile is still unpublished:
    /// what does the old virtual tile read? (Read only; tier 1 defines an
    /// unbacked read as zero.)
    HeapDropPending,
    /// Read bandwidth of a fully mapped sparse buffer against a dense and a
    /// placement-heap buffer of the same size.
    Bandwidth {
        #[arg(long, default_value_t = 512)]
        mib: usize,
        #[arg(long, default_value_t = 20)]
        reps: usize,
    },
    /// Ask for a placement sparse buffer in a non-Private storage mode.
    SparseBufferStorage {
        /// shared | managed
        #[arg(long)]
        mode: String,
    },
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct P {
    ulong words_per_tile;
    uint n_tiles;
    uint salt;
    uint iters;
    uint pad;
};

inline uint pat(ulong idx, uint salt) {
    uint lo = (uint)idx;
    uint hi = (uint)(idx >> 32);
    return (lo * 2654435761u) ^ (hi * 40503u) ^ salt ^ 0x9e3779b9u;
}

// Thread g writes word g % wpt of virtual tile tiles[g / wpt]. The index is
// 64-bit and taken from ONE base: the buffer's address at offset 0.
kernel void fill(device uint* base [[buffer(0)]],
                 device const uint* tiles [[buffer(1)]],
                 device const P* p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    uint wpt = (uint)p->words_per_tile;
    uint t = gid / wpt;
    if (t >= p->n_tiles) return;
    ulong idx = (ulong)tiles[t] * p->words_per_tile + (gid % wpt);
    base[idx] = pat(idx, p->salt);
}

kernel void read(device const uint* base [[buffer(0)]],
                 device const uint* tiles [[buffer(1)]],
                 device const P* p [[buffer(2)]],
                 device uint* out [[buffer(3)]],
                 uint gid [[thread_position_in_grid]]) {
    uint wpt = (uint)p->words_per_tile;
    uint t = gid / wpt;
    if (t >= p->n_tiles) return;
    ulong idx = (ulong)tiles[t] * p->words_per_tile + (gid % wpt);
    out[gid] = base[idx];
}

// Reads tile tiles[0] `iters` times per thread, at an address that moves with
// the loop so no load is hoisted; the CPU knows the sum a fully mapped tile
// gives.
kernel void spin(device const uint* base [[buffer(0)]],
                 device const uint* tiles [[buffer(1)]],
                 device const P* p [[buffer(2)]],
                 device uint* out [[buffer(3)]],
                 uint gid [[thread_position_in_grid]]) {
    uint wpt = (uint)p->words_per_tile;
    ulong t0 = (ulong)tiles[0] * p->words_per_tile;
    uint s = 0;
    uint z = 0;
    for (uint i = 0; i < p->iters; i++) {
        uint w = base[t0 + ((gid + i * 37u) % wpt)];
        s += w;
        z += (w == 0u) ? 1u : 0u;
    }
    out[2 * gid] = s;
    out[2 * gid + 1] = z;
}

// Streams `n4` uint4 from `src`, grid-stride; one word per thread out.
kernel void stream(device const uint4* src [[buffer(0)]],
                   device const P* p [[buffer(2)]],
                   device uint* out [[buffer(3)]],
                   uint gid [[thread_position_in_grid]],
                   uint grid [[threads_per_grid]]) {
    ulong n4 = p->words_per_tile;
    uint4 acc = 0;
    for (ulong i = gid; i < n4; i += grid) {
        acc ^= src[i];
    }
    out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

kernel void touch(device uint* out [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = gid;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct P {
    words_per_tile: u64,
    n_tiles: u32,
    salt: u32,
    iters: u32,
    pad: u32,
}

fn pat(idx: u64, salt: u32) -> u32 {
    let lo = idx as u32;
    let hi = (idx >> 32) as u32;
    lo.wrapping_mul(2654435761) ^ hi.wrapping_mul(40503) ^ salt ^ 0x9e3779b9
}

// ------------------------------------------------------------ host gauges

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// The system's wired bytes (`host_statistics64`, `wire_count`).
fn wired_bytes() -> u64 {
    // SAFETY: a zeroed vm_statistics64 is a valid out-parameter; the count is
    // the struct's size in integer_t units, which the kernel clamps to its own.
    unsafe {
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::vm_statistics64>()
            / std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        #[allow(deprecated)]
        let host = libc::mach_host_self();
        let kr = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as libc::host_info64_t,
            &mut count,
        );
        if kr != 0 {
            return 0;
        }
        stats.wire_count as u64 * page_size() as u64
    }
}

/// `task_vm_info` through rev 3 (xnu `osfmk/mach/task_info.h`), as
/// `host_table_probe` reads it; only `phys_footprint` and the graphics
/// ledgers are used.
#[repr(C)]
#[derive(Default)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
    min_address: u64,
    max_address: u64,
    ledger: [i64; 21],
}

const TASK_VM_INFO: libc::task_flavor_t = 22;
const LEDGER_GRAPHICS_FOOTPRINT: usize = 13;
const LEDGER_GRAPHICS_NOFOOTPRINT: usize = 15;

/// (phys_footprint, graphics footprint + nofootprint ledgers).
fn task_vm() -> (u64, i64) {
    let mut info = TaskVmInfo::default();
    let mut count = (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;
    // SAFETY: `info` is a repr(C) out-parameter of `count` natural_t words.
    let kr = unsafe {
        #[allow(deprecated)]
        libc::task_info(
            libc::mach_task_self(),
            TASK_VM_INFO,
            &mut info as *mut _ as libc::task_info_t,
            &mut count,
        )
    };
    if kr != 0 {
        return (0, 0);
    }
    (
        info.phys_footprint,
        info.ledger[LEDGER_GRAPHICS_FOOTPRINT] + info.ledger[LEDGER_GRAPHICS_NOFOOTPRINT],
    )
}

fn mib(b: i128) -> f64 {
    b as f64 / 1048576.0
}

/// A labelled snapshot of the memory gauges, printed as a delta from the
/// previous one.
struct Gauges {
    dev: Dev,
    last: (u64, i64, u64, usize),
}

impl Gauges {
    fn new(dev: &Dev) -> Self {
        let mut g = Self {
            dev: dev.clone(),
            last: (0, 0, 0, 0),
        };
        g.last = g.read();
        let (fp, gfx, wired, dalloc) = g.last;
        println!(
            "MEM baseline footprint_mib={:.1} graphics_mib={:.1} wired_mib={:.1} device_allocated_mib={:.1}",
            mib(fp as i128),
            mib(gfx as i128),
            mib(wired as i128),
            mib(dalloc as i128)
        );
        g
    }

    fn read(&self) -> (u64, i64, u64, usize) {
        let (fp, gfx) = task_vm();
        (fp, gfx, wired_bytes(), self.dev.currentAllocatedSize())
    }

    fn step(&mut self, label: &str) {
        let now = self.read();
        let d = |a: i128, b: i128| mib(a - b);
        println!(
            "MEM {label}: d_footprint_mib={:+.1} d_graphics_mib={:+.1} d_wired_mib={:+.1} d_device_allocated_mib={:+.1} (footprint_mib={:.1} device_allocated_mib={:.1})",
            d(now.0 as i128, self.last.0 as i128),
            d(now.1 as i128, self.last.1 as i128),
            d(now.2 as i128, self.last.2 as i128),
            d(now.3 as i128, self.last.3 as i128),
            mib(now.0 as i128),
            mib(now.3 as i128),
        );
        self.last = now;
    }
}

/// Exits the process if the system's wired bytes grow past `cap` over the
/// start: a probe that wires a whole virtual range must not take the machine.
struct Watchdog {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(cap: u64) -> Self {
        let base = wired_bytes();
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let w = wired_bytes();
                if w > base + cap {
                    eprintln!(
                        "WATCHDOG wired grew {:.1} MiB > cap {:.1} MiB: exiting",
                        mib(w as i128 - base as i128),
                        mib(cap as i128)
                    );
                    // SAFETY: _exit ends the process without unwinding,
                    // releasing every buffer and heap.
                    unsafe { libc::_exit(3) };
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ------------------------------------------------------------ the GPU rig

fn as_alloc_buf(b: &ProtocolObject<dyn MTLBuffer>) -> &ProtocolObject<dyn MTLAllocation> {
    // SAFETY: an MTLBuffer is an MTLAllocation (MTLResource refines it); the
    // cast mirrors `residency_set::as_allocation`.
    unsafe {
        &*(b as *const ProtocolObject<dyn MTLBuffer> as *const ProtocolObject<dyn MTLAllocation>)
    }
}

fn as_alloc_heap(h: &ProtocolObject<dyn MTLHeap>) -> &ProtocolObject<dyn MTLAllocation> {
    // SAFETY: MTLHeap refines MTLAllocation; the cast mirrors `heap.rs`.
    unsafe {
        &*(h as *const ProtocolObject<dyn MTLHeap> as *const ProtocolObject<dyn MTLAllocation>)
    }
}

fn page_enum(kib: usize) -> Result<MTLSparsePageSize> {
    Ok(match kib {
        16 => MTLSparsePageSize::Size16,
        64 => MTLSparsePageSize::Size64,
        256 => MTLSparsePageSize::Size256,
        _ => bail!("page size {kib} KiB: 16, 64 or 256"),
    })
}

/// One Metal 4 queue with one allocator, command buffer and argument table
/// (each submission is waited before the next reuses them), a Metal 3 queue,
/// one shared event, one residency set on both queues, and a shared scratch
/// buffer for the tile lists and parameters.
struct Rig {
    dev: Dev,
    q4: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    alloc: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    cb4: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    q3: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ev: SharedEvent,
    next: AtomicU64,
    rs: Retained<ProtocolObject<dyn MTLResidencySet>>,
    scratch: Buf,
    scratch_at: Mutex<usize>,
    fill: Pso,
    read: Pso,
    spin: Pso,
    touch: Pso,
    stream: Pso,
    errors: Arc<Mutex<Vec<String>>>,
}

const SCRATCH_BYTES: usize = 4 << 20;

impl Rig {
    fn new(dev: &Dev) -> Result<Self> {
        let q4 = dev.newMTL4CommandQueue().context("newMTL4CommandQueue")?;
        let alloc = dev.newCommandAllocator().context("newCommandAllocator")?;
        let cb4 = dev.newCommandBuffer().context("MTL4 newCommandBuffer")?;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(8);
        desc.setInitializeBindings(true);
        let table = dev
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| anyhow::anyhow!("argument table: {e}"))?;
        let q3 = dev.newCommandQueue().context("newCommandQueue")?;
        let ev = dev.newSharedEvent().context("newSharedEvent")?;
        let rs = dev
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| anyhow::anyhow!("residency set: {e}"))?;
        q4.addResidencySet(&rs);
        q3.addResidencySet(&rs);
        let scratch = dev
            .newBufferWithLength_options(SCRATCH_BYTES, MTLResourceOptions::StorageModeShared)
            .context("scratch")?;
        rs.addAllocation(as_alloc_buf(&scratch));
        rs.commit();
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
            .map_err(|e| anyhow::anyhow!("compile: {e}"))?;
        let pso = |name: &str| -> Result<Pso> {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .with_context(|| format!("no function {name}"))?;
            dev.newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow::anyhow!("pipeline {name}: {e}"))
        };
        Ok(Self {
            fill: pso("fill")?,
            read: pso("read")?,
            spin: pso("spin")?,
            touch: pso("touch")?,
            stream: pso("stream")?,
            dev: dev.clone(),
            q4,
            alloc,
            cb4,
            table,
            q3,
            ev,
            next: AtomicU64::new(0),
            rs,
            scratch,
            scratch_at: Mutex::new(0),
            errors: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn shared(&self, len: usize) -> Result<Buf> {
        let b = self
            .dev
            .newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared)
            .context("shared buffer")?;
        self.rs.addAllocation(as_alloc_buf(&b));
        self.rs.commit();
        Ok(b)
    }

    /// Copies `bytes` into the scratch buffer; returns its offset.
    fn stage<T: Copy>(&self, items: &[T]) -> usize {
        let len = std::mem::size_of_val(items);
        let mut at = self.scratch_at.lock();
        let off = (*at + 255) & !255;
        assert!(off + len <= SCRATCH_BYTES, "scratch exhausted");
        // SAFETY: the scratch buffer is Shared, `off + len` is inside it, and
        // no GPU work reading this range is in flight (every caller resets
        // the bump only after waiting).
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr().cast::<u8>(),
                self.scratch.contents().as_ptr().cast::<u8>().add(off),
                len,
            );
        }
        *at = off + len;
        off
    }

    fn reset_scratch(&self) {
        *self.scratch_at.lock() = 0;
    }

    fn signal(&self) -> u64 {
        let v = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.q4
            .signalEvent_value(ProtocolObject::from_ref(&*self.ev), v);
        v
    }

    fn wait(&self, v: u64) -> Result<()> {
        ensure!(
            self.ev.waitUntilSignaledValue_timeoutMS(v, 20_000),
            "event {v} not signalled in 20 s"
        );
        Ok(())
    }

    fn map(&self, buf: &Buf, heap: Option<&Heap>, ops: &[MTL4UpdateSparseBufferMappingOperation]) {
        // SAFETY: `ops` is a live slice of `ops.len()` operations; the heap is
        // a placement heap whose maxCompatiblePlacementSparsePageSize is the
        // buffer's page size (every caller creates both from one value), or
        // None for unmaps only.
        unsafe {
            self.q4.updateBufferMappings_heap_operations_count(
                buf,
                heap.map(|h| &**h),
                NonNull::new(ops.as_ptr() as *mut _).expect("non-empty ops"),
                ops.len(),
            );
        }
    }

    /// Encodes one compute encoder on the Metal 4 command buffer and commits
    /// it, with a queue barrier after every prior queue stage (mapping
    /// updates included) before its dispatches. Does not wait.
    fn commit4(&self, encode: impl FnOnce(&Enc4)) -> Result<()> {
        self.cb4.beginCommandBufferWithAllocator(&self.alloc);
        let enc = self
            .cb4
            .computeCommandEncoder()
            .context("MTL4 compute encoder")?;
        enc.setArgumentTable(Some(&self.table));
        enc.barrierAfterQueueStages_beforeStages_visibilityOptions(
            MTLStages::ResourceState | MTLStages::Dispatch | MTLStages::Blit,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        encode(&enc);
        enc.endEncoding();
        self.cb4.endCommandBuffer();
        let options = MTL4CommitOptions::new();
        let errors = self.errors.clone();
        let block = RcBlock::new(move |fb: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            // SAFETY: Metal passes a live feedback object for the call.
            let fb = unsafe { fb.as_ref() };
            if let Some(e) = fb.error() {
                errors
                    .lock()
                    .push(format!("MTL4 commit: {}", e.localizedDescription()));
            }
        });
        // SAFETY: Metal copies the block; it captures only an Arc.
        unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&block)) };
        let mut raw = NonNull::from(&*self.cb4);
        // SAFETY: one live, ended command buffer.
        unsafe {
            self.q4
                .commit_count_options(NonNull::from(&mut raw), 1, &options);
        }
        Ok(())
    }

    /// `commit4`, then the event, then a wait, then the allocator reset.
    fn run4(&self, encode: impl FnOnce(&Enc4)) -> Result<()> {
        self.commit4(encode)?;
        let v = self.signal();
        self.wait(v)?;
        self.alloc.reset();
        self.check_errors()
    }

    fn check_errors(&self) -> Result<()> {
        let errs = std::mem::take(&mut *self.errors.lock());
        if errs.is_empty() {
            Ok(())
        } else {
            bail!("GPU errors: {errs:?}")
        }
    }

    fn bind4(&self, b: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize) {
        // SAFETY: the table holds 8 buffer bindings; the buffer is in the
        // residency set and outlives the command buffer.
        unsafe {
            self.table
                .setAddress_atIndex(b.gpuAddress() + offset as u64, index)
        }
    }

    fn dispatch4(&self, enc: &Enc4, pso: &Pso, threads: usize) {
        enc.setComputePipelineState(pso);
        let tg = pso.maxTotalThreadsPerThreadgroup().min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
    }

    fn bind3(&self, enc: &Enc3, b: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize) {
        // SAFETY: the buffer outlives the command buffer (waited below).
        unsafe { enc.setBuffer_offset_atIndex(Some(b), offset, index) }
    }

    fn dispatch3(&self, enc: &Enc3, pso: &Pso, threads: usize) {
        enc.setComputePipelineState(pso);
        let tg = pso.maxTotalThreadsPerThreadgroup().min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
    }

    /// `fill` over `tiles` (virtual tile indices) of `buf`, Metal 4.
    fn fill4(&self, buf: &Buf, tiles: &[u32], wpt: u64, salt: u32) -> Result<()> {
        self.reset_scratch();
        let t_off = self.stage(tiles);
        let p_off = self.stage(&[P {
            words_per_tile: wpt,
            n_tiles: tiles.len() as u32,
            salt,
            ..P::default()
        }]);
        self.run4(|enc| {
            self.bind4(buf, 0, 0);
            self.bind4(&self.scratch, t_off, 1);
            self.bind4(&self.scratch, p_off, 2);
            self.dispatch4(enc, &self.fill, tiles.len() * wpt as usize);
        })
    }

    /// `read` of `tiles` into `out` (which it zeroes first), Metal 4.
    fn read4(&self, buf: &Buf, tiles: &[u32], wpt: u64, out: &Buf) -> Result<Vec<u32>> {
        self.reset_scratch();
        let n = tiles.len() * wpt as usize;
        zero(out, n);
        let t_off = self.stage(tiles);
        let p_off = self.stage(&[P {
            words_per_tile: wpt,
            n_tiles: tiles.len() as u32,
            ..P::default()
        }]);
        self.run4(|enc| {
            self.bind4(buf, 0, 0);
            self.bind4(&self.scratch, t_off, 1);
            self.bind4(&self.scratch, p_off, 2);
            self.bind4(out, 0, 3);
            self.dispatch4(enc, &self.read, n);
        })?;
        Ok(words(out, n))
    }

    /// The same read on the Metal 3 queue. With `after`, the command buffer
    /// first waits on the shared event for that value (committed before the
    /// caller signals it); the caller then waits with `finish3`.
    fn read3_commit(
        &self,
        buf: &Buf,
        tiles: &[u32],
        wpt: u64,
        out: &Buf,
        after: Option<u64>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.reset_scratch();
        let n = tiles.len() * wpt as usize;
        zero(out, n);
        let t_off = self.stage(tiles);
        let p_off = self.stage(&[P {
            words_per_tile: wpt,
            n_tiles: tiles.len() as u32,
            ..P::default()
        }]);
        let cb = self.q3.commandBuffer().context("Metal 3 command buffer")?;
        if let Some(v) = after {
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*self.ev), v);
        }
        let enc = cb.computeCommandEncoder().context("Metal 3 encoder")?;
        self.bind3(&enc, buf, 0, 0);
        self.bind3(&enc, &self.scratch, t_off, 1);
        self.bind3(&enc, &self.scratch, p_off, 2);
        self.bind3(&enc, out, 0, 3);
        self.dispatch3(&enc, &self.read, n);
        enc.endEncoding();
        cb.commit();
        Ok(cb)
    }

    fn finish3(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        out: &Buf,
        n: usize,
    ) -> Result<Vec<u32>> {
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            bail!(
                "Metal 3 command buffer error: {}",
                cb.error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_default()
            );
        }
        Ok(words(out, n))
    }

    fn placement_heap(&self, bytes: usize, page: MTLSparsePageSize) -> Result<Heap> {
        let desc = MTLHeapDescriptor::new();
        desc.setSize(bytes);
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
        desc.setType(MTLHeapType::Placement);
        desc.setMaxCompatiblePlacementSparsePageSize(page);
        self.dev
            .newHeapWithDescriptor(&desc)
            .context("placement heap")
    }

    fn sparse_buffer(&self, len: usize, page: MTLSparsePageSize) -> Option<Buf> {
        // SAFETY: plain creation call; nil is a refusal.
        unsafe {
            self.dev
                .newBufferWithLength_options_placementSparsePageSize(
                    len,
                    MTLResourceOptions::StorageModePrivate,
                    page,
                )
        }
    }
}

/// A buffer's CPU pointer, possibly null (Private storage), read without
/// objc2's non-null return type. Only reported, never dereferenced.
fn contents_ptr(b: &Buf) -> *mut std::ffi::c_void {
    // SAFETY: `contents` takes no arguments and returns a pointer, null for a
    // buffer the CPU cannot address.
    unsafe { msg_send![&**b, contents] }
}

fn zero(b: &Buf, n: usize) {
    // SAFETY: a Shared buffer at least `n` words long, not in use by the GPU.
    unsafe { std::ptr::write_bytes(b.contents().as_ptr().cast::<u32>(), 0, n) }
}

fn words(b: &Buf, n: usize) -> Vec<u32> {
    // SAFETY: a Shared buffer at least `n` words long, its writer completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<u32>(), n).to_vec() }
}

fn op(
    mode: MTLSparseTextureMappingMode,
    tile: usize,
    count: usize,
    heap_tile: usize,
) -> MTL4UpdateSparseBufferMappingOperation {
    MTL4UpdateSparseBufferMappingOperation {
        mode,
        bufferRange: NSRange::new(tile, count),
        heapOffset: heap_tile,
    }
}

const MAP: MTLSparseTextureMappingMode = MTLSparseTextureMappingMode::Map;
const UNMAP: MTLSparseTextureMappingMode = MTLSparseTextureMappingMode::Unmap;

/// Compares a read-back against the pattern `salt` wrote; returns
/// (matching words, zero words, other words).
fn check(got: &[u32], tiles: &[u32], wpt: u64, salt: u32) -> (usize, usize, usize) {
    let (mut ok, mut zeros, mut other) = (0, 0, 0);
    for (g, &w) in got.iter().enumerate() {
        let idx = tiles[g / wpt as usize] as u64 * wpt + (g as u64 % wpt);
        if w == pat(idx, salt) {
            ok += 1;
        } else if w == 0 {
            zeros += 1;
        } else {
            other += 1;
        }
    }
    (ok, zeros, other)
}

fn verdict(label: &str, got: (usize, usize, usize), want_ok: usize, want_zero: usize) -> bool {
    let pass = got.0 == want_ok && got.1 == want_zero && got.2 == 0;
    println!(
        "CHECK {label}: match={} zero={} other={} (expected match={want_ok} zero={want_zero}) -> {}",
        got.0,
        got.1,
        got.2,
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ------------------------------------------------------------ inventory

fn inventory() -> Result<()> {
    let _dog = Watchdog::start(6 << 30);
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    println!("DEVICE name={}", dev.name());
    let families = [
        ("Apple7", MTLGPUFamily::Apple7),
        ("Apple8", MTLGPUFamily::Apple8),
        ("Apple9", MTLGPUFamily::Apple9),
        ("Mac2", MTLGPUFamily::Mac2),
        ("Common3", MTLGPUFamily::Common3),
        ("Metal3", MTLGPUFamily::Metal3),
        ("Metal4", MTLGPUFamily::Metal4),
    ];
    for (name, f) in families {
        println!("FAMILY {name}={}", dev.supportsFamily(f));
    }
    let responds = dev.respondsToSelector(sel!(supportsPlacementSparse));
    let placement_sparse: Option<bool> = if responds {
        // SAFETY: the device responds to the selector, a BOOL property
        // (MTLDevice.h, macos(26.4)) that objc2-metal 0.3.2 does not bind.
        Some(unsafe { msg_send![&*dev, supportsPlacementSparse] })
    } else {
        None
    };
    println!("CAP supportsPlacementSparse responds={responds} value={placement_sparse:?}");
    println!("CAP sparseTileSizeInBytes={}", dev.sparseTileSizeInBytes());
    for (kib, p) in [
        (16, MTLSparsePageSize::Size16),
        (64, MTLSparsePageSize::Size64),
        (256, MTLSparsePageSize::Size256),
    ] {
        println!(
            "CAP sparseTileSizeInBytesForSparsePageSize({kib}KiB)={}",
            dev.sparseTileSizeInBytesForSparsePageSize(p)
        );
    }
    println!("CAP maxBufferLength={}", dev.maxBufferLength());
    println!(
        "CAP recommendedMaxWorkingSetSize={} hasUnifiedMemory={}",
        dev.recommendedMaxWorkingSetSize(),
        dev.hasUnifiedMemory()
    );
    let desc = MTLHeapDescriptor::new();
    println!(
        "CAP MTLHeapDescriptor default type={:?} maxCompatiblePlacementSparsePageSize={:?} sparsePageSize={:?}",
        desc.r#type(),
        desc.maxCompatiblePlacementSparsePageSize(),
        desc.sparsePageSize()
    );
    let sa = dev
        .heapBufferSizeAndAlignWithLength_options(64 << 10, MTLResourceOptions::StorageModePrivate);
    println!(
        "CAP heapBufferSizeAndAlign(64KiB, Private) size={} align={}",
        sa.size, sa.align
    );
    // Creation attempts: each length and page size, both storage modes; the
    // buffer is dropped at once and never bound.
    for (kib, p) in [
        (16, MTLSparsePageSize::Size16),
        (64, MTLSparsePageSize::Size64),
        (256, MTLSparsePageSize::Size256),
    ] {
        // Shared is asked in its own subcommand: Metal asserts on it.
        {
            let (mode, opts) = ("Private", MTLResourceOptions::StorageModePrivate);
            let len = 1usize << 30;
            // SAFETY: plain creation; nil is a refusal.
            let b =
                unsafe { dev.newBufferWithLength_options_placementSparsePageSize(len, opts, p) };
            match b {
                Some(b) => println!(
                    "CREATE sparse len=1GiB page={kib}KiB storage={mode}: ok length={} allocatedSize={} sparseBufferTier={:?} gpuAddress=0x{:x} contents_null={}",
                    b.length(),
                    MTLAllocation::allocatedSize(&*b),
                    b.sparseBufferTier(),
                    b.gpuAddress(),
                    contents_ptr(&b).is_null()
                ),
                None => println!("CREATE sparse len=1GiB page={kib}KiB storage={mode}: nil"),
            }
        }
    }
    // Ascending, and stop growing the moment a length wires memory: a
    // sparse buffer that wired its virtual length would take the machine at
    // 64 GiB.
    let max = dev.maxBufferLength();
    for gib in [1usize, 4, 16, 64, 256, 1024] {
        let len = gib << 30;
        let wired0 = wired_bytes();
        // SAFETY: plain creation; nil is a refusal.
        let b = unsafe {
            dev.newBufferWithLength_options_placementSparsePageSize(
                len,
                MTLResourceOptions::StorageModePrivate,
                MTLSparsePageSize::Size64,
            )
        };
        let d_wired = mib(wired_bytes() as i128 - wired0 as i128);
        match b {
            Some(b) => println!(
                "CREATE sparse len={gib}GiB (maxBufferLength {}): ok length={} allocatedSize={} d_wired_mib={d_wired:+.1}",
                if len > max { "exceeded" } else { "within" },
                b.length(),
                MTLAllocation::allocatedSize(&*b)
            ),
            None => println!(
                "CREATE sparse len={gib}GiB (maxBufferLength {}): nil",
                if len > max { "exceeded" } else { "within" }
            ),
        }
        if d_wired > 256.0 {
            println!("CREATE stopped: {gib} GiB wired {d_wired:.1} MiB");
            break;
        }
    }
    Ok(())
}

fn sparse_heap_buffer() -> Result<()> {
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let desc = MTLHeapDescriptor::new();
    desc.setSize(64 << 20);
    desc.setType(MTLHeapType::Sparse);
    desc.setStorageMode(MTLStorageMode::Private);
    let Some(heap) = dev.newHeapWithDescriptor(&desc) else {
        println!("SPARSEHEAP create: nil");
        return Ok(());
    };
    println!(
        "SPARSEHEAP create: ok size={} type={:?}",
        heap.size(),
        heap.r#type()
    );
    let b = heap.newBufferWithLength_options(64 << 10, MTLResourceOptions::StorageModePrivate);
    match b {
        Some(b) => println!(
            "SPARSEHEAP newBufferWithLength(64KiB): ok length={} sparseBufferTier={:?}",
            b.length(),
            b.sparseBufferTier()
        ),
        None => println!("SPARSEHEAP newBufferWithLength(64KiB): nil"),
    }
    Ok(())
}

fn sparse_buffer_storage(mode: &str) -> Result<()> {
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let opts = match mode {
        "shared" => MTLResourceOptions::StorageModeShared,
        "managed" => MTLResourceOptions::StorageModeManaged,
        _ => bail!("mode: shared | managed"),
    };
    println!("STORAGE asking for a 1 GiB 64 KiB-page placement sparse buffer, storage={mode}");
    // SAFETY: plain creation; nil is a refusal (an assertion aborts).
    let b = unsafe {
        dev.newBufferWithLength_options_placementSparsePageSize(
            1 << 30,
            opts,
            MTLSparsePageSize::Size64,
        )
    };
    match b {
        Some(b) => println!(
            "STORAGE {mode}: ok length={} contents_null={}",
            b.length(),
            contents_ptr(&b).is_null()
        ),
        None => println!("STORAGE {mode}: nil"),
    }
    Ok(())
}

// ------------------------------------------------------------ function

fn function(virt_gib: usize, heap_mib: usize, wired_cap_gib: u64) -> Result<()> {
    let _dog = Watchdog::start(wired_cap_gib << 30);
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let rig = Rig::new(&dev)?;
    let mut g = Gauges::new(&dev);
    let page = MTLSparsePageSize::Size64;
    let tile = dev.sparseTileSizeInBytesForSparsePageSize(page);
    println!("TILE bytes={tile} (sparseTileSizeInBytesForSparsePageSize(64KiB))");
    let wpt = (tile / 4) as u64;
    let virt = virt_gib << 30;
    let vtiles = virt / tile;
    let htiles = (heap_mib << 20) / tile;
    let mut all = true;
    // A virtual tile no check reads, mapped to heap A's last tile to publish
    // pending unmaps.
    let sentinel = vtiles / 2;

    // F1: the virtual buffer.
    let Some(sb) = rig.sparse_buffer(virt, page) else {
        println!("F1 sparse buffer {virt_gib} GiB: nil — NOT AVAILABLE");
        return Ok(());
    };
    println!(
        "F1 sparse buffer: length={} allocatedSize={} sparseBufferTier={:?} gpuAddress=0x{:x} virtual_tiles={vtiles}",
        sb.length(),
        MTLAllocation::allocatedSize(&*sb),
        sb.sparseBufferTier(),
        sb.gpuAddress()
    );
    let tier1 = sb.sparseBufferTier() == MTLBufferSparseTier::Tier1;
    g.step("F1 after sparse buffer creation");

    // F2: two placement heaps, the chunks a tenant would grow by.
    let ha = rig.placement_heap(heap_mib << 20, page)?;
    g.step("F2 after heap A creation");
    let hb = rig.placement_heap(heap_mib << 20, page)?;
    g.step("F2 after heap B creation");
    println!(
        "F2 heaps: A size={} B size={} maxCompatiblePlacementSparsePageSize={:?} heap_tiles={htiles}",
        ha.size(),
        hb.size(),
        page
    );
    rig.rs.addAllocation(as_alloc_buf(&sb));
    rig.rs.addAllocation(as_alloc_heap(&ha));
    rig.rs.addAllocation(as_alloc_heap(&hb));
    rig.rs.commit();
    rig.rs.requestResidency();
    g.step("F2 after residency commit (buffer + both heaps)");

    // F3: map 8 tiles at the start from A, 8 at the end from B.
    let low: Vec<u32> = (0..8).collect();
    let high: Vec<u32> = ((vtiles - 8) as u32..vtiles as u32).collect();
    let t = Instant::now();
    rig.map(&sb, Some(&ha), &[op(MAP, 0, 8, 0)]);
    rig.map(&sb, Some(&hb), &[op(MAP, vtiles - 8, 8, 0)]);
    let call_us = t.elapsed().as_secs_f64() * 1e6;
    let v = rig.signal();
    rig.wait(v)?;
    println!(
        "F3 mapped tiles [0,8) <- A[0,8) and [{},{}) <- B[0,8): calls {call_us:.1} us, to signalled {:.1} us",
        vtiles - 8,
        vtiles,
        t.elapsed().as_secs_f64() * 1e6
    );
    g.step("F3 after mapping 16 tiles");

    // F4: one kernel writes both ranges from one base; read back on Metal 4
    // and on Metal 3.
    let both: Vec<u32> = low.iter().chain(high.iter()).copied().collect();
    let out = rig.shared(both.len() * wpt as usize * 4 + 64)?;
    rig.fill4(&sb, &both, wpt, 0x51)?;
    let got = rig.read4(&sb, &both, wpt, &out)?;
    all &= verdict(
        "F4 Metal4 fill+read, tiles 0..8 and last 8 of the virtual range, one base",
        check(&got, &both, wpt, 0x51),
        got.len(),
        0,
    );
    let cb = rig.read3_commit(&sb, &both, wpt, &out, None)?;
    let got = rig.finish3(&cb, &out, both.len() * wpt as usize)?;
    all &= verdict(
        "F4 Metal3 read of the same tiles (setBuffer at offset 0)",
        check(&got, &both, wpt, 0x51),
        got.len(),
        0,
    );
    // The tile right after each mapped range is unmapped; its first word
    // must not carry the pattern (tile size check), and on tier 1 reads 0.
    if tier1 {
        let edge = vec![8u32, (vtiles - 9) as u32];
        let got = rig.read4(&sb, &edge, wpt, &out)?;
        all &= verdict(
            "F5 tier1 read of the unmapped tiles adjacent to the mapped ranges",
            check(&got, &edge, wpt, 0x51),
            0,
            got.len(),
        );
        // A write to an unmapped tile is dropped (tier 1).
        let lone = vec![100u32];
        rig.fill4(&sb, &lone, wpt, 0x77)?;
        let got = rig.read4(&sb, &lone, wpt, &out)?;
        all &= verdict(
            "F5 tier1 write then read of unmapped tile 100",
            check(&got, &lone, wpt, 0x77),
            0,
            got.len(),
        );
        rig.map(&sb, Some(&ha), &[op(MAP, 100, 1, 8)]);
        let v = rig.signal();
        rig.wait(v)?;
        let got = rig.read4(&sb, &lone, wpt, &out)?;
        let c = check(&got, &lone, wpt, 0x77);
        println!(
            "F5 tile 100 after mapping it to A[8]: carries the dropped write's pattern in {} of {} words (expected 0)",
            c.0,
            got.len()
        );
        all &= c.0 == 0;
    } else {
        println!("F5 skipped: the buffer is not tier 1, so an unmapped access is undefined");
    }

    // F6: unmap tile 0 and map A[0] at tile 200: the data moves with the tile
    // (the map also publishes the unmap).
    rig.map(&sb, None, &[op(UNMAP, 0, 1, 0)]);
    rig.map(&sb, Some(&ha), &[op(MAP, 200, 1, 0)]);
    let v = rig.signal();
    rig.wait(v)?;
    let t200 = vec![200u32];
    let got = rig.read4(&sb, &t200, wpt, &out)?;
    // Tile 200 now shows what tile 0's writes put in A[0]: tile 0's pattern.
    let mut moved = 0;
    for (w, &x) in got.iter().enumerate() {
        if x == pat(w as u64, 0x51) {
            moved += 1;
        }
    }
    println!(
        "CHECK F6 A[0] remapped from virtual tile 0 to 200 shows tile 0's data: {moved} of {} words -> {}",
        got.len(),
        if moved == got.len() { "PASS" } else { "FAIL" }
    );
    all &= moved == got.len();
    if tier1 {
        let t0 = vec![0u32];
        let got = rig.read4(&sb, &t0, wpt, &out)?;
        all &= verdict(
            "F6 tier1 read of tile 0 after its unmap",
            check(&got, &t0, wpt, 0x51),
            0,
            got.len(),
        );
    }

    // F7: a Metal 3 reader ordered behind a Metal 4 mapping by the event
    // alone, no host wait between them. Tile 300 maps A[1], which holds tile
    // 1's pattern.
    let v_next = rig.next.load(Ordering::Relaxed) + 1;
    let t300 = vec![300u32];
    let cb = rig.read3_commit(&sb, &t300, wpt, &out, Some(v_next))?;
    std::thread::sleep(Duration::from_millis(20));
    rig.map(&sb, Some(&ha), &[op(MAP, 300, 1, 1)]);
    let v = rig.signal();
    ensure!(v == v_next, "event sequence");
    let got = rig.finish3(&cb, &out, wpt as usize)?;
    let mut seen = 0;
    for (w, &x) in got.iter().enumerate() {
        if x == pat(wpt + w as u64, 0x51) {
            seen += 1;
        }
    }
    println!(
        "CHECK F7 Metal3 command buffer waiting on the MTL4 queue's event after the mapping reads A[1] at tile 300: {seen} of {} words -> {}",
        got.len(),
        if seen == got.len() { "PASS" } else { "FAIL" }
    );
    all &= seen == got.len();

    // F8: unmap while a reader is in flight (tier 1 only: an unmapped read
    // is defined as zero). The reader runs ~90 ms; the unmap is issued a
    // quarter of the way in, with no wait, on the Metal 4 queue. Arm `m4`:
    // the reader on the same Metal 4 queue. Arm `m3`: the reader on the
    // Metal 3 queue, which nothing orders against the mapping queue.
    if tier1 {
        let threads = 256usize;
        let spin_out = rig.shared(threads * 8)?;
        let expect = |tile: u64, iters: u32| -> Vec<u32> {
            (0..threads)
                .map(|g| {
                    (0..iters).fold(0u32, |s, i| {
                        let w = (g as u64 + i as u64 * 37) % wpt;
                        s.wrapping_add(pat(tile * wpt + w, 0x51))
                    })
                })
                .collect()
        };
        // Returns (wall ms, exact sums, zero reads).
        let spin_once = |m3: bool,
                         tile: u32,
                         iters: u32,
                         unmap_after_us: Option<u64>|
         -> Result<(f64, Vec<u32>)> {
            rig.reset_scratch();
            zero(&spin_out, threads * 2);
            let t_off = rig.stage(&[tile]);
            let p_off = rig.stage(&[P {
                words_per_tile: wpt,
                n_tiles: 1,
                iters,
                ..P::default()
            }]);
            let t = Instant::now();
            let cb3 = if m3 {
                let cb = rig.q3.commandBuffer().context("Metal 3 command buffer")?;
                let enc = cb.computeCommandEncoder().context("Metal 3 encoder")?;
                rig.bind3(&enc, &sb, 0, 0);
                rig.bind3(&enc, &rig.scratch, t_off, 1);
                rig.bind3(&enc, &rig.scratch, p_off, 2);
                rig.bind3(&enc, &spin_out, 0, 3);
                rig.dispatch3(&enc, &rig.spin, threads);
                enc.endEncoding();
                cb.commit();
                Some(cb)
            } else {
                rig.commit4(|enc| {
                    rig.bind4(&sb, 0, 0);
                    rig.bind4(&rig.scratch, t_off, 1);
                    rig.bind4(&rig.scratch, p_off, 2);
                    rig.bind4(&spin_out, 0, 3);
                    rig.dispatch4(enc, &rig.spin, threads);
                })?;
                None
            };
            if let Some(us) = unmap_after_us {
                std::thread::sleep(Duration::from_micros(us));
                // An unmap is applied with the next map operation
                // (`unmap-hazard` C1–C1i): publish it with a sentinel map.
                rig.map(
                    &sb,
                    Some(&ha),
                    &[
                        op(UNMAP, tile as usize, 1, 0),
                        op(MAP, sentinel, 1, htiles - 1),
                    ],
                );
            }
            let v = rig.signal();
            rig.wait(v)?;
            if let Some(cb) = cb3 {
                rig.finish3(&cb, &spin_out, 0)?;
            } else {
                rig.alloc.reset();
                rig.check_errors()?;
            }
            Ok((
                t.elapsed().as_secs_f64() * 1e3,
                words(&spin_out, threads * 2),
            ))
        };
        let score = |got: &[u32], want: &[u32]| -> (usize, u64) {
            let exact = (0..threads).filter(|&g| got[2 * g] == want[g]).count();
            let zeros = (0..threads).map(|g| got[2 * g + 1] as u64).sum();
            (exact, zeros)
        };
        let mut iters = 1u32 << 14;
        let mut ms = 0.0;
        for _ in 0..8 {
            ms = spin_once(false, 1, iters, None)?.0;
            if ms > 60.0 {
                break;
            }
            iters = iters.saturating_mul(2);
        }
        let reads = threads as u64 * iters as u64;
        println!("F8 reader calibrated: iters={iters} wall={ms:.1} ms reads_per_run={reads}");
        let want = expect(1, iters);
        // The instrument's two levels: a mapped tile reads no zero word and
        // every sum is exact; an unmapped tile (400) reads only zeros.
        for m3 in [false, true] {
            let api = if m3 { "m3" } else { "m4" };
            let (_, got) = spin_once(m3, 1, iters, None)?;
            let (exact, zeros) = score(&got, &want);
            let pass = exact == threads && zeros == 0;
            println!(
                "CHECK F8 {api} control, mapped tile, no unmap: sums exact {exact} of {threads}, zero reads {zeros} -> {}",
                if pass { "PASS" } else { "FAIL" }
            );
            all &= pass;
            let (_, got) = spin_once(m3, 400, iters, None)?;
            let (_, zeros) = score(&got, &want);
            let pass = zeros == reads;
            println!(
                "CHECK F8 {api} control, unmapped tile 400: zero reads {zeros} of {reads} -> {}",
                if pass { "PASS" } else { "FAIL" }
            );
            all &= pass;
        }
        let out1 = rig.shared(wpt as usize * 4)?;
        for trial in 0..5 {
            for m3 in [false, true] {
                let api = if m3 { "m3" } else { "m4" };
                let (wall, got) = spin_once(m3, 1, iters, Some((ms * 1e3 / 4.0) as u64))?;
                let (exact, zeros) = score(&got, &want);
                // Afterwards the tile is unmapped: it reads zero.
                let after = rig.read4(&sb, &[1u32], wpt, &out1)?;
                let after_zero = after.iter().filter(|&&w| w == 0).count();
                println!(
                    "OBSERVE F8 {api} trial {trial}: unmap issued {:.1} ms into the reader, wall={wall:.1} ms, sums exact {exact} of {threads}, zero reads {zeros} of {reads}; tile 1 afterwards reads zero in {after_zero} of {} words",
                    ms / 4.0,
                    after.len()
                );
                all &= after_zero == after.len();
                rig.map(&sb, Some(&ha), &[op(MAP, 1, 1, 1)]);
                let v = rig.signal();
                rig.wait(v)?;
            }
        }
    }

    // F9: give heap B's memory back: unmap its tiles, drop it from the
    // residency set, drop the heap.
    rig.map(
        &sb,
        Some(&ha),
        &[
            op(UNMAP, vtiles - 8, 8, 0),
            op(MAP, sentinel, 1, htiles - 1),
        ],
    );
    let v = rig.signal();
    rig.wait(v)?;
    rig.rs.removeAllocation(as_alloc_heap(&hb));
    rig.rs.commit();
    g.step("F9 after unmapping B's tiles (published) and removing B from the residency set");
    let fp0 = task_vm().0;
    let t = Instant::now();
    drop(hb);
    let mut back_ms = None;
    while t.elapsed() < Duration::from_secs(5) {
        if fp0.saturating_sub(task_vm().0) >= (heap_mib as u64) << 19 {
            back_ms = Some(t.elapsed().as_secs_f64() * 1e3);
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    println!("F9 footprint fell by half heap B's size or more after its drop: {back_ms:?} ms (polled every 10 ms, up to 5 s)");
    g.step("F9 after dropping heap B");
    // The low tiles still read through A.
    let still: Vec<u32> = (1..8).collect();
    let got = rig.read4(&sb, &still, wpt, &out)?;
    all &= verdict(
        "F9 tiles 1..8 (heap A) still read after heap B is gone",
        check(&got, &still, wpt, 0x51),
        got.len(),
        0,
    );
    // F10: grow again from a fresh heap C mapped into the same virtual range;
    // the buffer, and so the address every kernel binds, is unchanged.
    let addr_before = sb.gpuAddress();
    let hc = rig.placement_heap(heap_mib << 20, page)?;
    rig.rs.addAllocation(as_alloc_heap(&hc));
    rig.rs.commit();
    g.step("F10 after heap C creation + residency commit");
    rig.map(&sb, Some(&hc), &[op(MAP, vtiles - 8, 8, 0)]);
    let v = rig.signal();
    rig.wait(v)?;
    println!(
        "F10 gpuAddress before re-backing 0x{addr_before:x}, after 0x{:x}",
        sb.gpuAddress()
    );
    all &= addr_before == sb.gpuAddress();
    rig.fill4(&sb, &high, wpt, 0x99)?;
    let got = rig.read4(&sb, &high, wpt, &out)?;
    all &= verdict(
        "F10 the last 8 tiles re-backed by heap C, written and read",
        check(&got, &high, wpt, 0x99),
        got.len(),
        0,
    );
    println!("FUNCTION {}", if all { "PASS" } else { "FAIL" });
    if !all {
        std::process::exit(1);
    }
    Ok(())
}

// ------------------------------------------------------------ unmap hazard

/// Commits the `spin` reader over virtual tile `tile` on Metal 4 (`m3`
/// false) or Metal 3; returns the Metal 3 command buffer to wait on.
#[expect(
    clippy::too_many_arguments,
    reason = "a probe helper; each argument is one axis of the hazard cases"
)]
fn spin_commit(
    rig: &Rig,
    sb: &Buf,
    out: &Buf,
    tile: u32,
    wpt: u64,
    iters: u32,
    threads: usize,
    m3: bool,
    producer_barrier: bool,
) -> Result<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>> {
    rig.reset_scratch();
    zero(out, threads * 2);
    let t_off = rig.stage(&[tile]);
    let p_off = rig.stage(&[P {
        words_per_tile: wpt,
        n_tiles: 1,
        iters,
        ..P::default()
    }]);
    if m3 {
        let cb = rig.q3.commandBuffer().context("Metal 3 command buffer")?;
        let enc = cb.computeCommandEncoder().context("Metal 3 encoder")?;
        rig.bind3(&enc, sb, 0, 0);
        rig.bind3(&enc, &rig.scratch, t_off, 1);
        rig.bind3(&enc, &rig.scratch, p_off, 2);
        rig.bind3(&enc, out, 0, 3);
        rig.dispatch3(&enc, &rig.spin, threads);
        enc.endEncoding();
        cb.commit();
        Ok(Some(cb))
    } else {
        rig.commit4(|enc| {
            rig.bind4(sb, 0, 0);
            rig.bind4(&rig.scratch, t_off, 1);
            rig.bind4(&rig.scratch, p_off, 2);
            rig.bind4(out, 0, 3);
            rig.dispatch4(enc, &rig.spin, threads);
            if producer_barrier {
                // Later queue work of stage ResourceState (mapping updates)
                // waits for this encoder's dispatches.
                enc.barrierAfterStages_beforeQueueStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::ResourceState,
                    MTL4VisibilityOptions::Device,
                );
            }
        })?;
        Ok(None)
    }
}

fn settle(rig: &Rig) -> Result<()> {
    let v = rig.signal();
    rig.wait(v)?;
    rig.alloc.reset();
    rig.check_errors()
}

fn zeros_in(rig: &Rig, sb: &Buf, tile: u32, wpt: u64, out: &Buf) -> Result<usize> {
    let got = rig.read4(sb, &[tile], wpt, out)?;
    Ok(got.iter().filter(|&&w| w == 0).count())
}

fn unmap_hazard(trials: usize, iters: u32, delay_ms: u64) -> Result<()> {
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let rig = Rig::new(&dev)?;
    let page = MTLSparsePageSize::Size64;
    let tile = dev.sparseTileSizeInBytesForSparsePageSize(page);
    let wpt = (tile / 4) as u64;
    let sb = rig
        .sparse_buffer(1 << 30, page)
        .context("sparse buffer: nil")?;
    let heap = rig.placement_heap(64 << 20, page)?;
    rig.rs.addAllocation(as_alloc_buf(&sb));
    rig.rs.addAllocation(as_alloc_heap(&heap));
    rig.rs.commit();
    let other_sb = rig
        .sparse_buffer(1 << 30, page)
        .context("sparse buffer: nil")?;
    rig.rs.addAllocation(as_alloc_buf(&other_sb));
    rig.rs.commit();
    let out = rig.shared(wpt as usize * 4)?;
    let threads = 256usize;
    let spin_out = rig.shared(threads * 8)?;
    let w = wpt as usize;
    let next_heap_tile = std::cell::Cell::new(0usize);
    let next_heap_tile_peek = || next_heap_tile.get();
    let fresh = |v: u32| -> Result<usize> {
        let h = next_heap_tile.get();
        next_heap_tile.set(h + 1);
        rig.map(&sb, Some(&heap), &[op(MAP, v as usize, 1, h)]);
        settle(&rig)?;
        rig.fill4(&sb, &[v], wpt, 0x51)?;
        Ok(h)
    };
    println!("HAZARD tile_bytes={tile} words_per_tile={w} iters={iters} delay_ms={delay_ms}");
    let mut v = 10u32;
    for trial in 0..trials {
        // C1: plain unmap, nothing in flight.
        fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C1 plain unmap, idle: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        v += 1;
        // C1b..C1g: what makes an idle unmap visible to the next reader.
        fresh(v)?;
        rig.map(&sb, Some(&heap), &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C1b unmap passing the heap (not nil), idle: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        v += 1;
        fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        std::thread::sleep(Duration::from_millis(1000));
        println!(
            "HAZARD trial={trial} C1c unmap, then 1 s idle: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        v += 1;
        fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        let cb = rig.read3_commit(&sb, &[v], wpt, &out, None)?;
        let got = rig.finish3(&cb, &out, w)?;
        println!(
            "HAZARD trial={trial} C1d unmap, read on the Metal 3 queue: tile {v} reads zero in {} of {w}",
            got.iter().filter(|&&x| x == 0).count()
        );
        v += 1;
        // An unrelated map (another virtual tile, another heap tile) after
        // the unmap, in a separate call.
        fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        let h_other = next_heap_tile_peek();
        rig.map(
            &sb,
            Some(&heap),
            &[op(MAP, 4000 + v as usize, 1, 60 + h_other)],
        );
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C1e unmap, then an unrelated map call: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        rig.map(&sb, None, &[op(UNMAP, 4000 + v as usize, 1, 0)]);
        settle(&rig)?;
        v += 1;
        // The unrelated map in the same call as the unmap.
        fresh(v)?;
        rig.map(
            &sb,
            Some(&heap),
            &[
                op(UNMAP, v as usize, 1, 0),
                op(MAP, 4000 + v as usize, 1, 61 + h_other),
            ],
        );
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C1f unmap + unrelated map in one call: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        rig.map(&sb, None, &[op(UNMAP, 4000 + v as usize, 1, 0)]);
        settle(&rig)?;
        v += 1;
        // The unmapped tile's heap tile mapped elsewhere and overwritten:
        // does the old virtual tile see the new bytes?
        let h = fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        rig.map(&sb, Some(&heap), &[op(MAP, 5000 + v as usize, 1, h)]);
        settle(&rig)?;
        rig.fill4(&sb, &[5000 + v], wpt, 0x33)?;
        let got = rig.read4(&sb, &[v], wpt, &out)?;
        let (old, z, other) = check(&got, &[v], wpt, 0x51);
        let new_bytes = got
            .iter()
            .enumerate()
            .filter(|(i, &x)| x == pat((5000 + v as u64) * wpt + *i as u64, 0x33))
            .count();
        println!(
            "HAZARD trial={trial} C1g unmap, heap tile remapped at {} and rewritten: old tile {v} reads its old data {old}, zero {z}, the new bytes {new_bytes}, other {} of {w}",
            5000 + v,
            other - new_bytes
        );
        rig.map(&sb, None, &[op(UNMAP, 5000 + v as usize, 1, 0)]);
        settle(&rig)?;
        v += 1;
        // C1h: a translation the GPU never used. Heap tile h is filled
        // through u; u is unmapped; a map elsewhere publishes that; then v
        // is mapped to h (never touched by the GPU), unmapped, and read for
        // the first time. Zero: the unmap reached the page table and C1's
        // survivor is a cached translation. Data: the unmap itself waits.
        let u = v;
        let h = fresh(u)?;
        rig.map(&sb, None, &[op(UNMAP, u as usize, 1, 0)]);
        rig.map(
            &sb,
            Some(&heap),
            &[op(MAP, 6000 + u as usize, 1, 62 + next_heap_tile_peek())],
        );
        settle(&rig)?;
        rig.map(&sb, None, &[op(UNMAP, 6000 + u as usize, 1, 0)]);
        let v2 = u + 1;
        rig.map(&sb, Some(&heap), &[op(MAP, v2 as usize, 1, h)]);
        settle(&rig)?;
        rig.map(&sb, None, &[op(UNMAP, v2 as usize, 1, 0)]);
        settle(&rig)?;
        let got = rig.read4(&sb, &[v2], wpt, &out)?;
        let carries = got
            .iter()
            .enumerate()
            .filter(|(i, &x)| x == pat(u as u64 * wpt + *i as u64, 0x51))
            .count();
        println!(
            "HAZARD trial={trial} C1h map v to a filled heap tile, unmap before any GPU use, first read: tile {v2} reads zero in {} of {w}, the heap tile's data in {carries}",
            got.iter().filter(|&&x| x == 0).count()
        );
        rig.map(&sb, Some(&heap), &[op(MAP, 7000, 1, 63)]);
        settle(&rig)?;
        v += 2;
        // C1i: the publishing map made on ANOTHER sparse buffer.
        fresh(v)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        rig.map(&other_sb, Some(&heap), &[op(MAP, 0, 1, 64 + (trial % 8))]);
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C1i unmap, then a map on another sparse buffer: tile {v} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?
        );
        rig.map(&sb, Some(&heap), &[op(MAP, 7000, 1, 63)]);
        settle(&rig)?;
        v += 1;
        // C2: the heap tile also mapped at v+1; unmap v.
        let h = fresh(v)?;
        rig.map(&sb, Some(&heap), &[op(MAP, v as usize + 1, 1, h)]);
        settle(&rig)?;
        rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
        settle(&rig)?;
        println!(
            "HAZARD trial={trial} C2 unmap of an aliased tile, idle: tile {v} reads zero in {} of {w}; alias {} reads zero in {} of {w}",
            zeros_in(&rig, &sb, v, wpt, &out)?,
            v + 1,
            zeros_in(&rig, &sb, v + 1, wpt, &out)?
        );
        v += 2;
        // C3..C10: a reader of tile v in flight on either queue; the unmap
        // issued `delay_ms` (or 0) after its commit, alone or with a
        // publishing map (an unrelated tile) in the same call. Afterwards a
        // publishing map is made in every case, and the tile is read.
        for (case, m3, delay, publish, barrier) in [
            (
                "C3 reader m4, unmap+publish after delay",
                false,
                delay_ms,
                true,
                false,
            ),
            (
                "C4 reader m3, unmap+publish after delay",
                true,
                delay_ms,
                true,
                false,
            ),
            ("C5 reader m4, unmap+publish at once", false, 0, true, false),
            ("C6 reader m3, unmap+publish at once", true, 0, true, false),
            (
                "C7 reader m4, unmap alone after delay",
                false,
                delay_ms,
                false,
                false,
            ),
            (
                "C8 reader m3, unmap alone after delay",
                true,
                delay_ms,
                false,
                false,
            ),
            (
                "C9 reader m4 + producer barrier Dispatch->ResourceState, unmap+publish at once",
                false,
                0,
                true,
                true,
            ),
            (
                "C10 reader m4 + producer barrier Dispatch->ResourceState, unmap+publish after delay",
                false,
                delay_ms,
                true,
                true,
            ),
        ] {
            fresh(v)?;
            let t = Instant::now();
            let cb3 = spin_commit(&rig, &sb, &spin_out, v, wpt, iters, threads, m3, barrier)?;
            if delay > 0 {
                std::thread::sleep(Duration::from_millis(delay));
            }
            let t_unmap = t.elapsed().as_secs_f64() * 1e3;
            if publish {
                rig.map(
                    &sb,
                    Some(&heap),
                    &[op(UNMAP, v as usize, 1, 0), op(MAP, 7000, 1, 63)],
                );
            } else {
                rig.map(&sb, None, &[op(UNMAP, v as usize, 1, 0)]);
            }
            settle(&rig)?;
            if let Some(cb) = &cb3 {
                rig.finish3(cb, &spin_out, 0)?;
            }
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let got = words(&spin_out, threads * 2);
            let zr: u64 = (0..threads).map(|g| got[2 * g + 1] as u64).sum();
            let after = zeros_in(&rig, &sb, v, wpt, &out)?;
            rig.map(&sb, Some(&heap), &[op(MAP, 7000, 1, 63)]);
            settle(&rig)?;
            let published = zeros_in(&rig, &sb, v, wpt, &out)?;
            println!(
                "HAZARD trial={trial} {case}: unmap at {t_unmap:.1} ms, wall {wall:.1} ms, reader zero reads {zr} of {}; tile {v} afterwards reads zero in {after} of {w}; after a further publishing map: {published} of {w}",
                threads as u64 * iters as u64
            );
            v += 1;
        }
        // C11: a Metal 3 reader signals an event when it completes; the
        // Metal 4 queue waits for that event, then the unmap is issued at
        // once. Does the queue's wait hold the mapping update back?
        fresh(v)?;
        let done = dev.newSharedEvent().context("event")?;
        let t = Instant::now();
        let cb = spin_commit(&rig, &sb, &spin_out, v, wpt, iters, threads, true, false)?
            .context("Metal 3 command buffer")?;
        // The signal goes in a second command buffer, after the reader's.
        let sig = rig.q3.commandBuffer().context("Metal 3 command buffer")?;
        sig.encodeSignalEvent_value(ProtocolObject::from_ref(&*done), 1);
        sig.commit();
        rig.q4
            .waitForEvent_value(ProtocolObject::from_ref(&*done), 1);
        rig.map(
            &sb,
            Some(&heap),
            &[op(UNMAP, v as usize, 1, 0), op(MAP, 7000, 1, 63)],
        );
        settle(&rig)?;
        rig.finish3(&cb, &spin_out, 0)?;
        sig.waitUntilCompleted();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let got = words(&spin_out, threads * 2);
        let zr: u64 = (0..threads).map(|g| got[2 * g + 1] as u64).sum();
        let after = zeros_in(&rig, &sb, v, wpt, &out)?;
        println!(
            "HAZARD trial={trial} C11 reader m3 signals an event at completion, MTL4 queue waitForEvent, unmap+publish at once: wall {wall:.1} ms, reader zero reads {zr} of {}; tile {v} afterwards reads zero in {after} of {w}",
            threads as u64 * iters as u64
        );
        v += 1;
    }
    Ok(())
}

fn heap_drop_pending() -> Result<()> {
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let rig = Rig::new(&dev)?;
    let page = MTLSparsePageSize::Size64;
    let tile = dev.sparseTileSizeInBytesForSparsePageSize(page);
    let wpt = (tile / 4) as u64;
    let w = wpt as usize;
    let sb = rig
        .sparse_buffer(1 << 30, page)
        .context("sparse buffer: nil")?;
    let keep = rig.placement_heap(16 << 20, page)?;
    rig.rs.addAllocation(as_alloc_buf(&sb));
    rig.rs.addAllocation(as_alloc_heap(&keep));
    rig.rs.commit();
    let out = rig.shared(w * 4)?;
    for (case, publish) in [
        ("D1 unmap published, heap dropped", true),
        ("D2 unmap unpublished, heap dropped", false),
    ] {
        let doomed = rig.placement_heap(16 << 20, page)?;
        rig.rs.addAllocation(as_alloc_heap(&doomed));
        rig.rs.commit();
        rig.map(&sb, Some(&doomed), &[op(MAP, 10, 1, 0)]);
        settle(&rig)?;
        rig.fill4(&sb, &[10], wpt, 0x51)?;
        let before = check(&rig.read4(&sb, &[10], wpt, &out)?, &[10], wpt, 0x51).0;
        if publish {
            rig.map(&sb, Some(&keep), &[op(UNMAP, 10, 1, 0), op(MAP, 20, 1, 0)]);
        } else {
            rig.map(&sb, None, &[op(UNMAP, 10, 1, 0)]);
        }
        settle(&rig)?;
        rig.rs.removeAllocation(as_alloc_heap(&doomed));
        rig.rs.commit();
        drop(doomed);
        std::thread::sleep(Duration::from_millis(500));
        let got = rig.read4(&sb, &[10], wpt, &out);
        match got {
            Ok(got) => {
                let (old, z, other) = check(&got, &[10], wpt, 0x51);
                println!(
                    "DROP {case}: before the drop the tile read its data in {before} of {w}; 500 ms after the drop it reads its old data {old}, zero {z}, other {other} of {w}"
                );
            }
            Err(e) => println!("DROP {case}: the read after the drop failed: {e}"),
        }
        // Publish anything still pending before the next case.
        rig.map(&sb, Some(&keep), &[op(UNMAP, 20, 1, 0), op(MAP, 30, 1, 1)]);
        settle(&rig)?;
    }
    Ok(())
}

// ------------------------------------------------------------ bandwidth

fn bandwidth(mib_len: usize, reps: usize) -> Result<()> {
    ensure!(reps >= 5, "N >= 5");
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let rig = Rig::new(&dev)?;
    let len = mib_len << 20;
    let threads = 64 * 1024usize;
    let out = rig.shared(threads * 4)?;
    let mut arms: Vec<(String, Buf, Option<Heap>)> = Vec::new();
    let dense = dev
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
        .context("dense buffer")?;
    arms.push(("dense".into(), dense, None));
    let ph = rig.placement_heap(len, MTLSparsePageSize::Size64)?;
    // SAFETY: offset 0 of a placement heap of exactly `len` bytes.
    let placed = unsafe {
        ph.newBufferWithLength_options_offset(len, MTLResourceOptions::StorageModePrivate, 0)
    }
    .context("placement buffer")?;
    arms.push(("placement-heap".into(), placed, Some(ph)));
    for kib in [16usize, 64, 256] {
        let page = page_enum(kib)?;
        let tile = dev.sparseTileSizeInBytesForSparsePageSize(page);
        let heap = rig.placement_heap(len, page)?;
        let sb = rig.sparse_buffer(len, page).context("sparse buffer: nil")?;
        rig.map(&sb, Some(&heap), &[op(MAP, 0, len / tile, 0)]);
        settle(&rig)?;
        arms.push((format!("sparse-{kib}KiB"), sb, Some(heap)));
    }
    for (_, b, h) in &arms {
        rig.rs.addAllocation(as_alloc_buf(b));
        if let Some(h) = h {
            rig.rs.addAllocation(as_alloc_heap(h));
        }
    }
    rig.rs.commit();
    rig.rs.requestResidency();
    // Write every arm once so every page has been touched.
    for (_, b, _) in &arms {
        let wpt = 16384u64;
        let tiles: Vec<u32> = (0..(len / 65536) as u32).collect();
        rig.fill4(b, &tiles, wpt, 0x5)?;
    }
    let p = [P {
        words_per_tile: (len / 16) as u64,
        ..P::default()
    }];
    let mut gbs: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for round in 0..=reps {
        for (ai, (_, b, _)) in arms.iter().enumerate() {
            rig.reset_scratch();
            let p_off = rig.stage(&p);
            let cb = rig.q3.commandBuffer().context("Metal 3 command buffer")?;
            let enc = cb.computeCommandEncoder().context("Metal 3 encoder")?;
            rig.bind3(&enc, b, 0, 0);
            rig.bind3(&enc, &rig.scratch, p_off, 2);
            rig.bind3(&enc, &out, 0, 3);
            rig.dispatch3(&enc, &rig.stream, threads);
            enc.endEncoding();
            cb.commit();
            rig.finish3(&cb, &out, 0)?;
            let s = cb.GPUEndTime() - cb.GPUStartTime();
            if round > 0 {
                gbs[ai].push(len as f64 / s / 1e9);
            }
        }
    }
    for (ai, (name, _, _)) in arms.iter().enumerate() {
        let v = &mut gbs[ai];
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        println!(
            "BW {name} bytes={len} n={n} min_gbs={:.1} p50_gbs={:.1} max_gbs={:.1} mean_gbs={mean:.1}",
            pct(v, 0.0),
            pct(v, 0.5),
            pct(v, 1.0)
        );
    }
    Ok(())
}

// ------------------------------------------------------------ cost

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i.min(v.len() - 1)]
}

fn summary(name: &str, v: &mut [f64]) {
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64).sqrt();
    println!(
        "COST {name} n={n} min_us={:.2} p50_us={:.2} p90_us={:.2} max_us={:.2} mean_us={mean:.2} sd_us={sd:.2}",
        pct(v, 0.0),
        pct(v, 0.5),
        pct(v, 0.9),
        pct(v, 1.0),
    );
}

fn cost(reps: usize, tiles: &[usize], page_kib: &[usize]) -> Result<()> {
    ensure!(reps >= 5, "N >= 5");
    let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let rig = Rig::new(&dev)?;
    let max_tiles = *tiles.iter().max().context("tiles")?;
    let touch_out = rig.shared(4096)?;
    for &kib in page_kib {
        let page = page_enum(kib)?;
        let tile = dev.sparseTileSizeInBytesForSparsePageSize(page);
        // Virtual range: room for a scattered layout (every other tile) and
        // a sentinel tile past it; the heap has one tile more than the
        // largest count, for the sentinel. Every unmap is published by a
        // map of the sentinel in the same call (`unmap-hazard`).
        let virt = (2 * max_tiles + 2) * tile;
        let sentinel = 2 * max_tiles + 1;
        let sb = rig
            .sparse_buffer(virt, page)
            .context("sparse buffer: nil")?;
        let heap = rig.placement_heap((max_tiles + 1) * tile, page)?;
        rig.rs.addAllocation(as_alloc_buf(&sb));
        rig.rs.addAllocation(as_alloc_heap(&heap));
        rig.rs.commit();
        println!(
            "COSTCFG page={kib}KiB tile_bytes={tile} heap_bytes={}",
            heap.size()
        );
        let cells: Vec<String> = std::iter::once("null".to_string())
            .chain(std::iter::once("dispatch".to_string()))
            .chain(tiles.iter().flat_map(|n| {
                [
                    format!("map_contig_{n}"),
                    format!("unmap_contig_{n}"),
                    format!("map_scatter_{n}"),
                    format!("unmap_scatter_{n}"),
                    format!("map+dispatch_{n}"),
                ]
            }))
            .collect();
        let mut call: Vec<Vec<f64>> = vec![Vec::new(); cells.len()];
        let mut done: Vec<Vec<f64>> = vec![Vec::new(); cells.len()];
        // One warm-up round, then `reps` rounds; within a round every cell
        // runs once, so the cells are interleaved.
        for round in 0..=reps {
            for (ci, name) in cells.iter().enumerate() {
                let n: usize = name
                    .rsplit('_')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let contig_map = [op(MAP, 0, n.max(1), 0)];
                let publish = op(MAP, sentinel, 1, max_tiles);
                let contig_unmap = [op(UNMAP, 0, n.max(1), 0), publish];
                let scatter_map: Vec<_> = (0..n).map(|i| op(MAP, 2 * i, 1, i)).collect();
                let scatter_unmap: Vec<_> = (0..n)
                    .map(|i| op(UNMAP, 2 * i, 1, 0))
                    .chain(std::iter::once(publish))
                    .collect();
                // Unmap cells first map (untimed), so the unmap has work.
                if name.starts_with("unmap_contig") {
                    rig.map(&sb, Some(&heap), &contig_map);
                    let v = rig.signal();
                    rig.wait(v)?;
                } else if name.starts_with("unmap_scatter") {
                    rig.map(&sb, Some(&heap), &scatter_map);
                    let v = rig.signal();
                    rig.wait(v)?;
                }
                let t = Instant::now();
                let tc = match name.as_str() {
                    "null" => 0.0,
                    "dispatch" => {
                        rig.commit4(|enc| {
                            rig.bind4(&touch_out, 0, 3);
                            rig.dispatch4(enc, &rig.touch, 64);
                        })?;
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    s if s.starts_with("map_contig") => {
                        rig.map(&sb, Some(&heap), &contig_map);
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    s if s.starts_with("unmap_contig") => {
                        rig.map(&sb, Some(&heap), &contig_unmap);
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    s if s.starts_with("map_scatter") => {
                        rig.map(&sb, Some(&heap), &scatter_map);
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    s if s.starts_with("unmap_scatter") => {
                        rig.map(&sb, Some(&heap), &scatter_unmap);
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    s if s.starts_with("map+dispatch") => {
                        rig.map(&sb, Some(&heap), &contig_map);
                        rig.commit4(|enc| {
                            rig.bind4(&touch_out, 0, 3);
                            rig.dispatch4(enc, &rig.touch, 64);
                        })?;
                        t.elapsed().as_secs_f64() * 1e6
                    }
                    _ => unreachable!(),
                };
                let v = rig.signal();
                rig.wait(v)?;
                let td = t.elapsed().as_secs_f64() * 1e6;
                if name.starts_with("dispatch") || name.starts_with("map+dispatch") {
                    rig.alloc.reset();
                }
                rig.check_errors()?;
                // Leave nothing mapped for the next cell.
                if name.starts_with("map_contig") || name.starts_with("map+dispatch") {
                    rig.map(&sb, Some(&heap), &contig_unmap);
                } else if name.starts_with("map_scatter") {
                    rig.map(&sb, Some(&heap), &scatter_unmap);
                }
                if name.starts_with("map") {
                    let v = rig.signal();
                    rig.wait(v)?;
                }
                if round > 0 {
                    call[ci].push(tc);
                    done[ci].push(td);
                }
            }
        }
        for (ci, name) in cells.iter().enumerate() {
            summary(&format!("page={kib}KiB {name} host_call"), &mut call[ci]);
            summary(&format!("page={kib}KiB {name} to_signalled"), &mut done[ci]);
        }
        rig.rs.removeAllocation(as_alloc_buf(&sb));
        rig.rs.removeAllocation(as_alloc_heap(&heap));
        rig.rs.commit();
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Inventory => inventory(),
        Cmd::Function {
            virt_gib,
            heap_mib,
            wired_cap_gib,
        } => function(virt_gib, heap_mib, wired_cap_gib),
        Cmd::Cost {
            reps,
            tiles,
            page_kib,
        } => cost(reps, &tiles, &page_kib),
        Cmd::SparseHeapBuffer => sparse_heap_buffer(),
        Cmd::HeapDropPending => heap_drop_pending(),
        Cmd::Bandwidth { mib, reps } => bandwidth(mib, reps),
        Cmd::UnmapHazard {
            trials,
            iters,
            delay_ms,
        } => unmap_hazard(trials, iters, delay_ms),
        Cmd::SparseBufferStorage { mode } => sparse_buffer_storage(&mode),
    }
}
