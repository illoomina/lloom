//! Can the GPU read a host-produced table without the host standing between
//! two decode steps? (`design/docs/expert-streaming.md` §20.13)
//!
//! Two probes of the Metal mechanisms that question rests on, both on raw
//! Metal objects so what is measured is the platform, not an engine wrapper:
//!
//! * `nocopy` — option C. A `newBufferWithBytesNoCopy` buffer over a
//!   read-only memory map of a large file (the N-gram table's shard), read by
//!   a gather kernel. Is it created at all, does the GPU read pages the page
//!   cache does not hold, and how much memory does the system WIRE for it —
//!   at creation, while a command buffer that reads it runs, while one is
//!   parked on an event before reading it, and as a residency-set member?
//!   The sizes are staged small to large and the run stops growing the
//!   moment a stage wires a large fraction of its range: wiring the whole
//!   28.8 GB table on a 64 GB machine is the P18 panic class, so a watchdog
//!   thread exits the process if the system's wired bytes pass a cap.
//! * `event` — option A's device half. A command buffer that publishes a
//!   value, waits on an `MTLSharedEvent`, then reads a staging buffer the host
//!   wrote after the publish and before `setSignaledValue`: the latency from
//!   the CPU signal to the GPU resuming, a non-vacuity arm (no signal, no
//!   resume), a long park, the data ordering, and the commit-ordering arms
//!   (the producer committed before the host waits for it; the inverse is
//!   the deadlock the encoder thread must never build).
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/host_table_probe nocopy --file <shard>
//! lloom-arb measure -- ./target/release/examples/host_table_probe event
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Dev = Retained<ProtocolObject<dyn MTLDevice>>;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Cb = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

#[derive(Parser, Debug)]
#[command(about = "Metal mechanics for a device-side reader of a host table (§20.13)")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Option C: a NoCopy buffer over a file mmap — readable, and what it wires.
    Nocopy {
        /// The file to map (the PLE shard). Regions are taken from its tail
        /// half first, where the table's rows lie.
        #[arg(long)]
        file: std::path::PathBuf,
        /// Region sizes to probe, in MiB, smallest first.
        #[arg(long, value_delimiter = ',', default_values_t = [64usize, 1024, 4096])]
        sizes_mib: Vec<usize>,
        /// Exit if the system's wired bytes pass the baseline by this much.
        #[arg(long, default_value_t = 6.0)]
        wired_cap_gib: f64,
        /// Stop growing the size when a stage wires more than this fraction
        /// of its region.
        #[arg(long, default_value_t = 0.25)]
        stop_fraction: f64,
        /// Where the first region starts, GiB into the file (default: its
        /// middle). Move it between runs so every run maps cold pages.
        #[arg(long)]
        start_gib: Option<f64>,
        /// Largest region (MiB) put in a residency set with `requestResidency`.
        #[arg(long, default_value_t = 1024)]
        residency_set_max_mib: usize,
        /// Rows gathered per command buffer, and the row stride in bytes.
        #[arg(long, default_value_t = 16)]
        rows: usize,
        #[arg(long, default_value_t = 90)]
        row_bytes: usize,
        /// How long a parked command buffer is held before the signal.
        #[arg(long, default_value_t = 300)]
        park_ms: u64,
    },
    /// Option A's device half: a CPU-signalled event waited on mid-buffer.
    Event {
        /// Timed trials per arm.
        #[arg(long, default_value_t = 400)]
        trials: usize,
        /// Floats the host writes into the staging buffer per trial (16 rows
        /// of 160 is one token's PLE rows).
        #[arg(long, default_value_t = 2560)]
        floats: usize,
        /// The long-park arm's hold before the signal.
        #[arg(long, default_value_t = 3000)]
        long_park_ms: u64,
    },
}

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_rows(device const uchar* table [[buffer(0)]],
                        device const ulong* offs [[buffer(1)]],
                        device uchar* out [[buffer(2)]],
                        constant uint& row_bytes [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= n * row_bytes) return;
    uint r = gid / row_bytes;
    uint b = gid - r * row_bytes;
    out[gid] = table[offs[r] + b];
}

kernel void publish(device atomic_uint* flag [[buffer(0)]],
                    constant uint& slot [[buffer(1)]],
                    constant uint& v [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid == 0) atomic_store_explicit(&flag[slot], v, memory_order_relaxed);
}

kernel void consume(device const float* staging [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = staging[gid];
}
"#;

// ---------------------------------------------------------------- host state

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

/// `task_vm_info` through rev 3 (xnu `osfmk/mach/task_info.h`); libc does not
/// carry it. Only the fields read below are named for use.
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
/// Index of `ledger_tag_graphics_footprint` in the rev-3 ledger block.
const LEDGER_GRAPHICS_FOOTPRINT: usize = 13;
const LEDGER_GRAPHICS_NOFOOTPRINT: usize = 15;

/// This process's VM figures: (resident, file-backed resident, phys_footprint,
/// graphics footprint + nofootprint ledgers).
fn task_vm() -> (u64, u64, u64, i64) {
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
        return (0, 0, 0, 0);
    }
    (
        info.resident_size,
        info.external,
        info.phys_footprint,
        info.ledger[LEDGER_GRAPHICS_FOOTPRINT] + info.ledger[LEDGER_GRAPHICS_NOFOOTPRINT],
    )
}

fn task_line() -> String {
    let (res, ext, fp, gfx) = task_vm();
    format!(
        "task_resident_mib={:.1} task_file_backed_mib={:.1} task_footprint_mib={:.1} task_graphics_mib={:.1}",
        res as f64 / 1048576.0,
        ext as f64 / 1048576.0,
        fp as f64 / 1048576.0,
        gfx as f64 / 1048576.0
    )
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn mib_signed(d: i64) -> f64 {
    d as f64 / (1u64 << 20) as f64
}

/// Watches the system's wired bytes every 5 ms, keeps the peak since the last
/// `reset_peak`, and exits the process past `cap`.
struct Watchdog {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(cap: u64) -> Self {
        let peak = Arc::new(AtomicU64::new(wired_bytes()));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (peak.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let w = wired_bytes();
                p.fetch_max(w, Ordering::Relaxed);
                if w > cap {
                    eprintln!(
                        "WATCHDOG wired {:.2} GiB > cap {:.2} GiB: exiting",
                        gib(w),
                        gib(cap)
                    );
                    // SAFETY: _exit is async-signal-safe and ends the process
                    // without unwinding, releasing every mapping and buffer.
                    unsafe { libc::_exit(3) };
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        Self {
            peak,
            stop,
            handle: Some(handle),
        }
    }

    fn reset_peak(&self) {
        self.peak.store(wired_bytes(), Ordering::Relaxed);
    }

    fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
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

/// A read-only shared mapping of `[off, off + len)` of `file`.
struct Mapping {
    ptr: *mut c_void,
    len: usize,
}

impl Mapping {
    fn new(file: &std::fs::File, off: u64, len: usize) -> Result<Self> {
        // SAFETY: a fresh mapping of a file this process keeps open; checked
        // against MAP_FAILED below and unmapped once on drop.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                off as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            bail!("mmap {len} at {off}: {}", std::io::Error::last_os_error())
        }
        // SAFETY: advice over the range just mapped.
        unsafe { libc::madvise(ptr, len, libc::MADV_RANDOM) };
        Ok(Self { ptr, len })
    }

    /// Per-page residency (`mincore`), one byte a page.
    fn residency(&self) -> Vec<u8> {
        let pages = self.len.div_ceil(page_size());
        let mut v = vec![0u8; pages];
        // SAFETY: the range is this mapping; `v` holds one byte per page.
        unsafe {
            libc::mincore(self.ptr, self.len, v.as_mut_ptr() as *mut libc::c_char);
        }
        v
    }

    fn resident_fraction(&self) -> f64 {
        let r = self.residency();
        r.iter()
            .filter(|&&b| b as i32 & libc::MINCORE_INCORE != 0)
            .count() as f64
            / r.len().max(1) as f64
    }

    /// `msync(MS_INVALIDATE)`: drop the range's cached pages where the
    /// kernel may (a wired page stays).
    fn invalidate(&self) -> i32 {
        // SAFETY: the range is this mapping.
        unsafe { libc::msync(self.ptr, self.len, libc::MS_INVALIDATE) }
    }

    fn page_resident(&self, byte: usize) -> bool {
        let page = page_size();
        let start = (byte / page) * page;
        let mut v = [0u8; 1];
        // SAFETY: one page inside this mapping.
        unsafe {
            libc::mincore(
                (self.ptr as *mut u8).add(start) as *mut c_void,
                page,
                v.as_mut_ptr() as *mut libc::c_char,
            );
        }
        v[0] as i32 & libc::MINCORE_INCORE != 0
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping created in `new`, unmapped once.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

// --------------------------------------------------------------- Metal glue

struct Gpu {
    dev: Dev,
    queue: Queue,
    gather: Pso,
    publish: Pso,
    consume: Pso,
}

impl Gpu {
    fn new() -> Result<Self> {
        let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = dev.newCommandQueue().context("newCommandQueue")?;
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
            gather: pso("gather_rows")?,
            publish: pso("publish")?,
            consume: pso("consume")?,
            dev,
            queue,
        })
    }

    fn shared(&self, len: usize) -> Result<Buf> {
        self.dev
            .newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared)
            .context("newBuffer")
    }

    fn cb(&self) -> Result<Cb> {
        self.queue.commandBuffer().context("commandBuffer")
    }

    fn event(&self) -> Result<SharedEvent> {
        self.dev.newSharedEvent().context("newSharedEvent")
    }
}

fn set_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: u32, index: usize) {
    // SAFETY: `v` is live for the call; Metal copies the bytes.
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::from(&v).cast::<c_void>(),
            std::mem::size_of::<u32>(),
            index,
        )
    }
}

fn set_buf(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, b: &Buf, index: usize) {
    // SAFETY: the buffer outlives the command buffer (the caller holds it
    // until completion).
    unsafe { enc.setBuffer_offset_atIndex(Some(&**b), 0, index) }
}

fn dispatch(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, pso: &Pso, threads: usize) {
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

fn wait_for_event(cb: &Cb, ev: &SharedEvent, v: u64) {
    let e: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&**ev);
    cb.encodeWaitForEvent_value(e, v);
}

fn cb_error(cb: &Cb) -> Option<String> {
    if cb.status() == MTLCommandBufferStatus::Error {
        Some(
            cb.error()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "error (no NSError)".into()),
        )
    } else {
        None
    }
}

fn contents(b: &Buf) -> *mut u8 {
    b.contents().as_ptr() as *mut u8
}

// ------------------------------------------------------------------- nocopy

struct GatherCheck {
    offs: Vec<u64>,
    nonresident_before: usize,
}

/// `rows` row offsets inside `[0, len − row_bytes)`, spread over the region
/// and preferring rows whose page the page cache does not hold.
fn pick_rows(map: &Mapping, rows: usize, row_bytes: usize, salt: u64) -> GatherCheck {
    let span = (map.len - row_bytes) as u64;
    let mut offs = Vec::with_capacity(rows);
    let mut nonresident = 0;
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    let mut tries = 0;
    while offs.len() < rows {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let off = (x % span) / row_bytes as u64 * row_bytes as u64;
        let cold = !map.page_resident(off as usize);
        tries += 1;
        if cold || tries > rows * 64 {
            nonresident += cold as usize;
            offs.push(off);
        }
    }
    GatherCheck {
        offs,
        nonresident_before: nonresident,
    }
}

fn encode_gather(
    gpu: &Gpu,
    cb: &Cb,
    table: &Buf,
    offs_buf: &Buf,
    out: &Buf,
    rows: usize,
    row_bytes: usize,
) -> Result<()> {
    let enc = cb.computeCommandEncoder().context("encoder")?;
    enc.setComputePipelineState(&gpu.gather);
    set_buf(&enc, table, 0);
    set_buf(&enc, offs_buf, 1);
    set_buf(&enc, out, 2);
    set_u32(&enc, row_bytes as u32, 3);
    set_u32(&enc, rows as u32, 4);
    dispatch(&enc, &gpu.gather, rows * row_bytes);
    enc.endEncoding();
    Ok(())
}

/// One gather of `chk`'s rows: (wall ms, GPU ms, error or "none").
#[allow(clippy::too_many_arguments)]
fn gather_once(
    gpu: &Gpu,
    table: &Buf,
    chk: &GatherCheck,
    offs_buf: &Buf,
    out: &Buf,
    rows: usize,
    row_bytes: usize,
) -> Result<(f64, f64, String)> {
    // SAFETY: offs_buf holds `rows` u64s, host-written before the commit;
    // no command buffer reading it is in flight.
    unsafe {
        std::ptr::copy_nonoverlapping(chk.offs.as_ptr(), contents(offs_buf) as *mut u64, rows)
    };
    let cb = gpu.cb()?;
    encode_gather(gpu, &cb, table, offs_buf, out, rows, row_bytes)?;
    let t = Instant::now();
    cb.commit();
    cb.waitUntilCompleted();
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
    Ok((wall, gpu_ms, cb_error(&cb).unwrap_or_else(|| "none".into())))
}

/// The gathered bytes against the file's, read with `pread` AFTER the GPU
/// read so the check cannot warm the page cache ahead of it.
fn verify_rows(
    file: &std::fs::File,
    region_off: u64,
    chk: &GatherCheck,
    out: &Buf,
    row_bytes: usize,
) -> Result<usize> {
    use std::os::unix::fs::FileExt;
    let mut bad = 0;
    let got = contents(out);
    let mut want = vec![0u8; row_bytes];
    for (i, &o) in chk.offs.iter().enumerate() {
        file.read_exact_at(&mut want, region_off + o)?;
        // SAFETY: `out` holds rows × row_bytes bytes and the command buffer
        // that wrote them has completed.
        let g = unsafe { std::slice::from_raw_parts(got.add(i * row_bytes), row_bytes) };
        bad += (g != want.as_slice()) as usize;
    }
    Ok(bad)
}

#[allow(clippy::too_many_arguments)]
fn nocopy(
    file_path: &std::path::Path,
    sizes_mib: &[usize],
    wired_cap_gib: f64,
    stop_fraction: f64,
    start_gib: Option<f64>,
    rs_max_mib: usize,
    rows: usize,
    row_bytes: usize,
    park_ms: u64,
) -> Result<()> {
    let gpu = Gpu::new()?;
    let page = page_size();
    let file = std::fs::File::open(file_path)?;
    let file_len = file.metadata()?.len();
    let base = wired_bytes();
    let cap = base + (wired_cap_gib * (1u64 << 30) as f64) as u64;
    println!(
        "DEVICE name={} unified={} max_buffer_gib={:.2} working_set_gib={:.2} page={page}",
        gpu.dev.name(),
        gpu.dev.hasUnifiedMemory(),
        gib(gpu.dev.maxBufferLength() as u64),
        gib(gpu.dev.recommendedMaxWorkingSetSize()),
    );
    println!(
        "FILE path={} bytes={file_len} wired_base_gib={:.3} watchdog_cap_gib={:.3}",
        file_path.display(),
        gib(base),
        gib(cap)
    );
    let dog = Watchdog::start(cap);
    let offs_buf = gpu.shared(rows * 8)?;
    let out = gpu.shared(rows * row_bytes)?;
    let ev = gpu.event()?;
    let mut ev_value = 0u64;

    // Regions from the file's second half, one after another, so no stage
    // reads pages an earlier stage brought in.
    let start = match start_gib {
        Some(g) => (g * (1u64 << 30) as f64) as u64,
        None => file_len / 2,
    };
    let mut cursor = start / page as u64 * page as u64;
    for (i, &mib) in sizes_mib.iter().enumerate() {
        let len = mib << 20;
        if cursor + len as u64 > file_len {
            println!("SKIP size_mib={mib}: past the file");
            break;
        }
        let region_off = cursor;
        cursor += len as u64;
        let map = Mapping::new(&file, region_off, len)?;
        let frac0 = map.resident_fraction();
        let stage = |name: &str, before: u64, extra: &str| {
            let now = wired_bytes();
            let peak = dog.peak();
            println!(
                "NOCOPY size_mib={mib} stage={name} wired_delta_mib={:.1} peak_delta_mib={:.1} {extra}",
                mib_signed(now as i64 - before as i64),
                mib_signed(peak as i64 - before as i64),
            );
            (now as i64 - before as i64).max(peak as i64 - before as i64)
        };

        dog.reset_peak();
        let w0 = wired_bytes();
        println!(
            "NOCOPY size_mib={mib} stage=mapped region_off={region_off} resident_frac={frac0:.4} {}",
            task_line()
        );
        // The eviction instrument's control, with no buffer over the range:
        // read a few rows in, then `msync(MS_INVALIDATE)`. If this does not
        // drop them, the `invalidated` stage below cannot tell pinned pages
        // from an eviction that does not work.
        {
            use std::os::unix::fs::FileExt;
            let probe_rows = pick_rows(&map, rows, row_bytes, 0xC0 + i as u64);
            let mut b = vec![0u8; row_bytes];
            for &o in &probe_rows.offs {
                file.read_exact_at(&mut b, region_off + o)?;
            }
            let res_before = probe_rows
                .offs
                .iter()
                .filter(|&&o| map.page_resident(o as usize))
                .count();
            let rc = map.invalidate();
            let res_after = probe_rows
                .offs
                .iter()
                .filter(|&&o| map.page_resident(o as usize))
                .count();
            println!(
                "NOCOPY size_mib={mib} stage=evict_control rows_read={rows} resident_before_msync={res_before} msync_rc={rc} resident_after_msync={res_after}"
            );
        }
        // SAFETY: page-aligned mapping of `len` bytes (a multiple of the page)
        // that outlives the buffer: the buffer is dropped before `map`.
        let table = unsafe {
            gpu.dev.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(map.ptr).context("null map")?,
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };
        let Some(table) = table else {
            println!("NOCOPY size_mib={mib} stage=create result=REFUSED (nil buffer)");
            continue;
        };
        let mut worst = stage(
            "create",
            w0,
            &format!(
                "resident_frac_before={frac0:.4} buffer_len={} {}",
                table.length(),
                task_line()
            ),
        );

        // One gather of cold rows, committed and waited for: the first
        // submission of the buffer.
        let chk = pick_rows(&map, rows, row_bytes, i as u64 + 1);
        dog.reset_peak();
        let (wall_ms, gpu_ms, err) =
            gather_once(&gpu, &table, &chk, &offs_buf, &out, rows, row_bytes)?;
        let resident_after = chk
            .offs
            .iter()
            .filter(|&&o| map.page_resident(o as usize))
            .count();
        let frac1 = map.resident_fraction();
        let bad = verify_rows(&file, region_off, &chk, &out, row_bytes)?;
        worst = worst.max(stage(
            "gather",
            w0,
            &format!(
                "rows={rows} cold_rows_before={} rows_resident_after={resident_after} \
                 mismatched_rows={bad} error={err} wall_ms={wall_ms:.3} gpu_ms={gpu_ms:.3} \
                 resident_frac_after={frac1:.4} {}",
                chk.nonresident_before,
                task_line()
            ),
        ));

        // The same gather again, the region now resident: does every
        // submission pay for the whole buffer?
        let (mut walls, mut gpus) = (Vec::new(), Vec::new());
        for _ in 0..20 {
            let (w, g, _) = gather_once(&gpu, &table, &chk, &offs_buf, &out, rows, row_bytes)?;
            walls.push(w * 1e3);
            gpus.push(g * 1e3);
        }
        worst = worst.max(stage(
            "warm_repeat",
            w0,
            &format!(
                "n=20 wall_us_p50={:.1} wall_us_max={:.1} gpu_us_p50={:.2}",
                pct(&mut walls, 0.5),
                pct(&mut walls, 1.0),
                pct(&mut gpus, 0.5)
            ),
        ));

        // Drop the region's pages while the buffer lives. If the buffer held
        // them wired, the invalidate cannot take them; if it did not, the next
        // submission shows whether it pages the whole range in again.
        let inval = map.invalidate();
        let frac_inval = map.resident_fraction();
        worst = worst.max(stage(
            "invalidated",
            w0,
            &format!(
                "msync_rc={inval} resident_frac={frac_inval:.4} {}",
                task_line()
            ),
        ));
        let chk_b = pick_rows(&map, rows, row_bytes, 0xB0B + i as u64);
        dog.reset_peak();
        let (wall_ms, gpu_ms, err) =
            gather_once(&gpu, &table, &chk_b, &offs_buf, &out, rows, row_bytes)?;
        let frac_b = map.resident_fraction();
        let bad = verify_rows(&file, region_off, &chk_b, &out, row_bytes)?;
        worst = worst.max(stage(
            "regather",
            w0,
            &format!(
                "cold_rows_before={} mismatched_rows={bad} error={err} wall_ms={wall_ms:.3} \
                 gpu_ms={gpu_ms:.3} resident_frac_after={frac_b:.4} {}",
                chk_b.nonresident_before,
                task_line()
            ),
        ));

        // A buffer parked on an event before its gather, over a region with
        // its pages dropped again: does committing alone page it in?
        map.invalidate();
        let chk2 = pick_rows(&map, rows, row_bytes, 0x5151 + i as u64);
        // SAFETY: offs_buf holds `rows` u64s; the previous buffer completed.
        unsafe {
            std::ptr::copy_nonoverlapping(chk2.offs.as_ptr(), contents(&offs_buf) as *mut u64, rows)
        };
        ev_value += 1;
        dog.reset_peak();
        let cb = gpu.cb()?;
        wait_for_event(&cb, &ev, ev_value);
        encode_gather(&gpu, &cb, &table, &offs_buf, &out, rows, row_bytes)?;
        let frac_pre = map.resident_fraction();
        cb.commit();
        std::thread::sleep(Duration::from_millis(park_ms));
        let parked_status = cb.status();
        let frac_parked = map.resident_fraction();
        worst = worst.max(stage(
            "parked",
            w0,
            &format!(
                "status={parked_status:?} resident_frac_before_commit={frac_pre:.4} \
                 resident_frac_parked={frac_parked:.4} cold_rows_before={} {}",
                chk2.nonresident_before,
                task_line()
            ),
        ));
        let t = Instant::now();
        ev.setSignaledValue(ev_value);
        cb.waitUntilCompleted();
        let sig_ms = t.elapsed().as_secs_f64() * 1e3;
        let bad2 = verify_rows(&file, region_off, &chk2, &out, row_bytes)?;
        let frac_res = map.resident_fraction();
        worst = worst.max(stage(
            "resumed",
            w0,
            &format!(
                "signal_to_complete_ms={sig_ms:.3} mismatched_rows={bad2} error={} resident_frac={frac_res:.4}",
                cb_error(&cb).as_deref().unwrap_or("none")
            ),
        ));

        // The engine's own residency path: a residency-set member with
        // requestResidency (how `register_buffers` would hold the table),
        // over a region with its pages dropped.
        if mib <= rs_max_mib {
            map.invalidate();
            let frac_pre = map.resident_fraction();
            let desc = MTLResidencySetDescriptor::new();
            let set = gpu
                .dev
                .newResidencySetWithDescriptor_error(&desc)
                .map_err(|e| anyhow::anyhow!("residency set: {e}"))?;
            // SAFETY: an MTLBuffer is an MTLAllocation (MTLResource refines
            // it); the pointer cast mirrors `residency_set::as_allocation`.
            let alloc: &ProtocolObject<dyn MTLAllocation> = unsafe {
                &*(&*table as *const ProtocolObject<dyn MTLBuffer>
                    as *const ProtocolObject<dyn MTLAllocation>)
            };
            dog.reset_peak();
            let t = Instant::now();
            set.addAllocation(alloc);
            set.commit();
            set.requestResidency();
            let req_ms = t.elapsed().as_secs_f64() * 1e3;
            std::thread::sleep(Duration::from_millis(park_ms));
            let frac_rs = map.resident_fraction();
            let inval = map.invalidate();
            let frac_rs_inval = map.resident_fraction();
            worst = worst.max(stage(
                "residency_set",
                w0,
                &format!(
                    "request_ms={req_ms:.3} resident_frac_before={frac_pre:.4} resident_frac={frac_rs:.4} \
                     after_msync_rc={inval} resident_frac_after_msync={frac_rs_inval:.4} allocated_gib={:.3} {}",
                    gib(set.allocatedSize()),
                    task_line()
                ),
            ));
            set.endResidency();
            set.removeAllocation(alloc);
            set.commit();
            drop(set);
            stage("residency_set_released", w0, &task_line());
        } else {
            println!("NOCOPY size_mib={mib} stage=residency_set skipped (> {rs_max_mib} MiB)");
        }

        drop(table);
        std::thread::sleep(Duration::from_millis(50));
        stage("released", w0, &task_line());
        drop(map);

        let frac = worst as f64 / len as f64;
        println!(
            "NOCOPY size_mib={mib} verdict worst_wired_mib={:.1} of_region={frac:.4}",
            mib_signed(worst)
        );
        if frac > stop_fraction {
            println!(
                "STOP a stage wired {:.1} % of a {mib} MiB region: larger sizes not attempted",
                frac * 100.0
            );
            break;
        }
    }
    drop(dog);
    Ok(())
}

// -------------------------------------------------------------------- event

fn spin_until(word: *const u32, v: u32, timeout: Duration) -> Option<Instant> {
    let start = Instant::now();
    loop {
        // SAFETY: a mapped Shared buffer word the GPU writes; volatile so the
        // load is not folded.
        if unsafe { std::ptr::read_volatile(word) } == v {
            return Some(Instant::now());
        }
        if start.elapsed() > timeout {
            return None;
        }
        std::hint::spin_loop();
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i.min(v.len() - 1)]
}

fn summary(name: &str, v: &mut [f64]) {
    if v.is_empty() {
        println!("EVENT {name} n=0");
        return;
    }
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    println!(
        "EVENT {name} n={n} p5_us={:.2} p50_us={:.2} p90_us={:.2} p99_us={:.2} max_us={:.2} mean_us={mean:.2}",
        pct(v, 0.05),
        pct(v, 0.5),
        pct(v, 0.9),
        pct(v, 0.99),
        pct(v, 1.0),
    );
}

struct EventRig {
    gpu: Gpu,
    ev: SharedEvent,
    flag: Buf,
    staging: Buf,
    out: Buf,
    floats: usize,
}

impl EventRig {
    fn flag_word(&self, slot: usize) -> *const u32 {
        // SAFETY: the flag buffer holds 16 u32s.
        unsafe { (contents(&self.flag) as *const u32).add(slot) }
    }

    fn encode_publish(&self, cb: &Cb, slot: u32, v: u32) -> Result<()> {
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&self.gpu.publish);
        set_buf(&enc, &self.flag, 0);
        set_u32(&enc, slot, 1);
        set_u32(&enc, v, 2);
        dispatch(&enc, &self.gpu.publish, 1);
        enc.endEncoding();
        Ok(())
    }

    /// consume (staging → out), then publish `v` into slot 1: the host sees
    /// slot 1 only after the copy (serial dispatch within the encoder).
    fn encode_consume(&self, cb: &Cb, v: u32) -> Result<()> {
        let enc = cb.computeCommandEncoder().context("encoder")?;
        enc.setComputePipelineState(&self.gpu.consume);
        set_buf(&enc, &self.staging, 0);
        set_buf(&enc, &self.out, 1);
        set_u32(&enc, self.floats as u32, 2);
        dispatch(&enc, &self.gpu.consume, self.floats);
        enc.setComputePipelineState(&self.gpu.publish);
        set_buf(&enc, &self.flag, 0);
        set_u32(&enc, 1, 1);
        set_u32(&enc, v, 2);
        dispatch(&enc, &self.gpu.publish, 1);
        enc.endEncoding();
        Ok(())
    }

    fn write_staging(&self, v: u32) {
        let p = contents(&self.staging) as *mut f32;
        for i in 0..self.floats {
            // SAFETY: the staging buffer holds `floats` f32s.
            unsafe { *p.add(i) = (v as f32) * 0.5 + i as f32 };
        }
    }

    fn out_matches(&self, v: u32) -> bool {
        let p = contents(&self.out) as *const f32;
        (0..self.floats).all(|i| {
            // SAFETY: as above, read after the flag showed the copy done.
            unsafe { std::ptr::read_volatile(p.add(i)) == (v as f32) * 0.5 + i as f32 }
        })
    }
}

fn event(trials: usize, floats: usize, long_park_ms: u64) -> Result<()> {
    let gpu = Gpu::new()?;
    let rig = EventRig {
        ev: gpu.event()?,
        flag: gpu.shared(64)?,
        staging: gpu.shared(floats * 4)?,
        out: gpu.shared(floats * 4)?,
        gpu,
        floats,
    };
    let to = Duration::from_secs(5);
    let mut v: u32 = 0;
    let mut next = || {
        v += 1;
        v
    };

    // Host-side costs of the two event calls.
    let mut set_ns = Vec::new();
    let mut get_ns = Vec::new();
    for _ in 0..2000 {
        let t = Instant::now();
        let s = rig.ev.signaledValue();
        get_ns.push(t.elapsed().as_nanos() as f64 / 1e3);
        let t = Instant::now();
        rig.ev.setSignaledValue(s);
        set_ns.push(t.elapsed().as_nanos() as f64 / 1e3);
    }
    summary("host_signaledValue_read", &mut get_ns);
    summary("host_setSignaledValue", &mut set_ns);

    // Warm the pipelines and the queue.
    for _ in 0..50 {
        let x = next();
        let cb = rig.gpu.cb()?;
        rig.encode_publish(&cb, 0, x)?;
        rig.encode_consume(&cb, x)?;
        cb.commit();
        cb.waitUntilCompleted();
    }

    // Arm 1: publish → wait(event) → consume, one command buffer. The host
    // sees the publish, writes the staging buffer, signals; the latency of
    // interest is signal → consume visible. Arm 0 (control) is the same
    // buffer without the wait: publish visible → consume visible, the GPU's
    // own kernel-to-kernel time as the host observes it.
    let (mut ctl, mut resume, mut pub_after_commit) = (Vec::new(), Vec::new(), Vec::new());
    let (mut wrong, mut timeouts) = (0usize, 0usize);
    for trial in 0..trials * 2 {
        let waiting = trial % 2 == 1;
        let x = next();
        let ev_v = x as u64;
        let cb = rig.gpu.cb()?;
        rig.encode_publish(&cb, 0, x)?;
        if waiting {
            wait_for_event(&cb, &rig.ev, ev_v);
        }
        rig.encode_consume(&cb, x)?;
        let t_commit = Instant::now();
        cb.commit();
        let Some(t_pub) = spin_until(rig.flag_word(0), x, to) else {
            timeouts += 1;
            rig.ev.setSignaledValue(ev_v);
            cb.waitUntilCompleted();
            continue;
        };
        if waiting {
            pub_after_commit.push((t_pub - t_commit).as_secs_f64() * 1e6);
            rig.write_staging(x);
            let t_sig = Instant::now();
            rig.ev.setSignaledValue(ev_v);
            match spin_until(rig.flag_word(1), x, to) {
                Some(t_done) => resume.push((t_done - t_sig).as_secs_f64() * 1e6),
                None => timeouts += 1,
            }
            cb.waitUntilCompleted();
            wrong += (!rig.out_matches(x)) as usize;
        } else {
            match spin_until(rig.flag_word(1), x, to) {
                Some(t_done) => ctl.push((t_done - t_pub).as_secs_f64() * 1e6),
                None => timeouts += 1,
            }
            cb.waitUntilCompleted();
            // The event is monotone: keep it level with the values used.
            rig.ev.setSignaledValue(ev_v);
        }
        if let Some(e) = cb_error(&cb) {
            println!("EVENT trial={trial} cb_error={e}");
        }
    }
    // Arm 1b: the same buffer with the event ALREADY at its value when the
    // GPU reaches the wait (the host fill finished inside the window): what
    // the wait costs when nothing waits. Interleaved with its no-wait control.
    let (mut pre, mut pre_ctl) = (Vec::new(), Vec::new());
    for trial in 0..trials * 2 {
        let waiting = trial % 2 == 1;
        let x = next();
        let ev_v = x as u64;
        let cb = rig.gpu.cb()?;
        rig.encode_publish(&cb, 0, x)?;
        if waiting {
            wait_for_event(&cb, &rig.ev, ev_v);
            rig.write_staging(x);
            rig.ev.setSignaledValue(ev_v);
        }
        rig.encode_consume(&cb, x)?;
        cb.commit();
        let Some(t_pub) = spin_until(rig.flag_word(0), x, to) else {
            timeouts += 1;
            cb.waitUntilCompleted();
            continue;
        };
        match spin_until(rig.flag_word(1), x, to) {
            Some(t_done) => {
                let us = (t_done - t_pub).as_secs_f64() * 1e6;
                if waiting {
                    pre.push(us)
                } else {
                    pre_ctl.push(us)
                }
            }
            None => timeouts += 1,
        }
        cb.waitUntilCompleted();
        if !waiting {
            rig.ev.setSignaledValue(ev_v);
        }
    }
    summary(
        "presignalled_publish_to_consume_seen (wait already satisfied)",
        &mut pre,
    );
    summary(
        "presignalled_control_publish_to_consume_seen (no wait)",
        &mut pre_ctl,
    );
    summary("commit_to_publish_seen", &mut pub_after_commit);
    summary("control_publish_to_consume_seen (no wait)", &mut ctl);
    summary("signal_to_consume_seen (wait mid-buffer)", &mut resume);
    println!(
        "EVENT data_order trials={trials} wrong_outputs={wrong} (staging written after the publish, before the signal) timeouts={timeouts}"
    );

    // Arm 2: non-vacuity — no signal, no resume; the part before the wait
    // has run.
    let x = next();
    let cb = rig.gpu.cb()?;
    rig.encode_publish(&cb, 0, x)?;
    wait_for_event(&cb, &rig.ev, x as u64);
    rig.encode_consume(&cb, x)?;
    cb.commit();
    let published = spin_until(rig.flag_word(0), x, to).is_some();
    let resumed_unsignalled = spin_until(rig.flag_word(1), x, Duration::from_millis(500)).is_some();
    rig.write_staging(x);
    rig.ev.setSignaledValue(x as u64);
    let resumed_after = spin_until(rig.flag_word(1), x, to).is_some();
    cb.waitUntilCompleted();
    println!(
        "EVENT non_vacuity published_before_wait={published} resumed_without_signal_500ms={resumed_unsignalled} resumed_after_signal={resumed_after} output_ok={}",
        rig.out_matches(x)
    );

    // Arm 3: a long park — the buffer completes without error.
    let x = next();
    let cb = rig.gpu.cb()?;
    rig.encode_publish(&cb, 0, x)?;
    wait_for_event(&cb, &rig.ev, x as u64);
    rig.encode_consume(&cb, x)?;
    cb.commit();
    std::thread::sleep(Duration::from_millis(long_park_ms));
    let status_parked = cb.status();
    rig.write_staging(x);
    rig.ev.setSignaledValue(x as u64);
    cb.waitUntilCompleted();
    println!(
        "EVENT long_park ms={long_park_ms} status_while_parked={status_parked:?} final_status={:?} error={} output_ok={}",
        cb.status(),
        cb_error(&cb).as_deref().unwrap_or("none"),
        rig.out_matches(x)
    );

    // Arm 4: commit ordering. (a) producer and consumer in two buffers, the
    // producer committed first; (b) both in one committed buffer (the
    // adaptive commit's shape); (c) the inverse — the host waits for a
    // publish whose buffer it has not committed: nothing can ever arrive.
    let x = next();
    let producer = rig.gpu.cb()?;
    rig.encode_publish(&producer, 0, x)?;
    let consumer = rig.gpu.cb()?;
    wait_for_event(&consumer, &rig.ev, x as u64);
    rig.encode_consume(&consumer, x)?;
    producer.commit();
    consumer.commit();
    let seen = spin_until(rig.flag_word(0), x, to).is_some();
    rig.write_staging(x);
    rig.ev.setSignaledValue(x as u64);
    let done = spin_until(rig.flag_word(1), x, to).is_some();
    consumer.waitUntilCompleted();
    println!(
        "EVENT order_two_buffers producer_first publish_seen={seen} completed={done} output_ok={}",
        rig.out_matches(x)
    );

    let x = next();
    let cb = rig.gpu.cb()?;
    rig.encode_publish(&cb, 0, x)?;
    wait_for_event(&cb, &rig.ev, x as u64);
    rig.encode_consume(&cb, x)?;
    cb.commit();
    let seen = spin_until(rig.flag_word(0), x, to).is_some();
    rig.write_staging(x);
    rig.ev.setSignaledValue(x as u64);
    let done = spin_until(rig.flag_word(1), x, to).is_some();
    cb.waitUntilCompleted();
    println!(
        "EVENT order_one_buffer publish_seen={seen} completed={done} output_ok={}",
        rig.out_matches(x)
    );

    let x = next();
    let cb = rig.gpu.cb()?;
    rig.encode_publish(&cb, 0, x)?;
    wait_for_event(&cb, &rig.ev, x as u64);
    rig.encode_consume(&cb, x)?;
    let seen_uncommitted = spin_until(rig.flag_word(0), x, Duration::from_millis(500)).is_some();
    cb.commit();
    let seen_after_commit = spin_until(rig.flag_word(0), x, to).is_some();
    rig.write_staging(x);
    rig.ev.setSignaledValue(x as u64);
    let done = spin_until(rig.flag_word(1), x, to).is_some();
    cb.waitUntilCompleted();
    println!(
        "EVENT order_uncommitted_producer publish_seen_before_commit_500ms={seen_uncommitted} seen_after_commit={seen_after_commit} completed={done}"
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Nocopy {
            file,
            sizes_mib,
            wired_cap_gib,
            stop_fraction,
            start_gib,
            residency_set_max_mib,
            rows,
            row_bytes,
            park_ms,
        } => nocopy(
            &file,
            &sizes_mib,
            wired_cap_gib,
            stop_fraction,
            start_gib,
            residency_set_max_mib,
            rows,
            row_bytes,
            park_ms,
        ),
        Cmd::Event {
            trials,
            floats,
            long_park_ms,
        } => event(trials, floats, long_park_ms),
    }
}
