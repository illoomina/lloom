//! `MTLIOCommandQueue` loads — file bytes straight into a device buffer as
//! GPU-queued commands, ordered against the compute stream by shared events
//! (`design/docs/expert-streaming.md` §20.4a, candidate B).
//!
//! A load is issued on the IO queue and never waited for on the host. The
//! compute command buffer signals `ev_compute` after everything it has encoded
//! so far (the slot's earlier readers), the IO command buffer waits for that,
//! loads, and signals `ev_io`; the compute buffer then waits for `ev_io` before
//! its next dispatch. Both queues are in-order, so one monotone value per
//! `load_ordered` call serialises the whole exchange. No `waitUntilCompleted`
//! anywhere on this path — the GPU does the waiting.

use super::{Buffer, Commands, Device};
use crate::kernels::MetalKernelError;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLDevice, MTLIOCommandBuffer, MTLIOCommandQueue, MTLIOCommandQueueDescriptor,
    MTLIOCommandQueueType, MTLIOFileHandle, MTLIOPriority, MTLSharedEvent,
};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// A file opened for `MTLIOCommandQueue` loads.
pub struct IoFile {
    handle: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
}

// SAFETY: an `MTLIOFileHandle` is an immutable Metal object usable from any
// thread; it is only ever read by loads encoded on the queue.
unsafe impl Send for IoFile {}
unsafe impl Sync for IoFile {}

/// One load: `len` bytes of `file` at `src_offset` into `dst` at `dst_offset`.
pub struct IoLoad<'a> {
    pub dst: &'a Buffer,
    pub dst_offset: usize,
    pub file: &'a IoFile,
    pub src_offset: u64,
    pub len: usize,
}

/// An IO queue plus the pair of shared events that order it against the
/// compute stream.
///
/// A point-of-use batch (`load_ordered`) is split across `lanes` IO command
/// buffers, each with its own event: one IO command buffer runs its loads
/// in order at ~2.2 GB/s on this SSD (the one-in-flight figure of
/// `p25-ssdbench`), eight concurrent ones at ~8.6 (issue #749 P27). The
/// compute stream waits for every lane the batch used. The in-flight
/// hints (`issue` / `await_value`) stay on lane 0.
pub struct IoQueue {
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
    ev_compute: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    ev_io: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    next: AtomicU64,
    /// The extra lanes (lane 0 is `ev_io` / `next`).
    lanes: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, AtomicU64)>,
    /// The compute event's counter: one monotone value per exchange.
    compute_next: AtomicU64,
    device: Device,
    loads: AtomicU64,
    bytes: AtomicU64,
}

/// `LLOOM_IO_LANES` (default 8): IO command buffers a point-of-use batch is
/// split across. Read once.
pub fn io_lanes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().metal.io_lanes.max(1))
}

// SAFETY: the queue and events are thread-safe Metal objects; the counters are
// atomics.
unsafe impl Send for IoQueue {}
unsafe impl Sync for IoQueue {}

impl IoQueue {
    /// A concurrent, normal-priority IO queue on `device`.
    pub fn new(device: &Device) -> Result<Self, MetalKernelError> {
        let desc = MTLIOCommandQueueDescriptor::new();
        desc.setType(MTLIOCommandQueueType::Concurrent);
        desc.setPriority(MTLIOPriority::Normal);
        let queue = device
            .as_ref()
            .newIOCommandQueueWithDescriptor_error(&desc)
            .map_err(|e| {
                MetalKernelError::FailedToCreateResource(format!("newIOCommandQueue: {e}"))
            })?;
        let event = |what: &str| {
            device.as_ref().newSharedEvent().ok_or_else(|| {
                MetalKernelError::FailedToCreateResource(format!(
                    "newSharedEvent ({what}) returned nil"
                ))
            })
        };
        let mut lanes = Vec::new();
        for _ in 1..io_lanes() {
            lanes.push((event("io lane")?, AtomicU64::new(0)));
        }
        Ok(Self {
            queue,
            ev_compute: event("compute")?,
            ev_io: event("io")?,
            next: AtomicU64::new(0),
            lanes,
            compute_next: AtomicU64::new(0),
            device: device.clone(),
            loads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        })
    }

    /// The lane events with their counters, lane 0 first.
    fn lane(&self, i: usize) -> (&ProtocolObject<dyn MTLSharedEvent>, &AtomicU64) {
        if i == 0 {
            (&self.ev_io, &self.next)
        } else {
            let (e, n) = &self.lanes[i - 1];
            (e, n)
        }
    }

    /// One IO command buffer over `loads`: waits for the compute signal
    /// `compute_value`, loads, signals lane `lane` with its next value
    /// (returned), commits.
    fn commit_lane(
        &self,
        lane: usize,
        compute_value: u64,
        loads: &[IoLoad<'_>],
    ) -> Result<u64, MetalKernelError> {
        let (ev, next) = self.lane(lane);
        let value = next.fetch_add(1, Ordering::AcqRel) + 1;
        let cb = self.queue.commandBuffer();
        cb.waitForEvent_value(&self.ev_compute, compute_value);
        let mut total = 0u64;
        for l in loads {
            if l.dst_offset + l.len > l.dst.length() {
                return Err(MetalKernelError::FailedToCreateResource(format!(
                    "io load: {}+{} past the {}-byte buffer",
                    l.dst_offset,
                    l.len,
                    l.dst.length()
                )));
            }
            // SAFETY: as in `issue`.
            unsafe {
                cb.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    l.dst.as_ref(),
                    l.dst.base_offset() + l.dst_offset,
                    l.len,
                    &l.file.handle,
                    l.src_offset as usize,
                );
            }
            total += l.len as u64;
        }
        cb.signalEvent_value(ev, value);
        cb.commit();
        self.loads.fetch_add(loads.len() as u64, Ordering::Relaxed);
        self.bytes.fetch_add(total, Ordering::Relaxed);
        Ok(value)
    }

    /// Block until every lane's latest batch but one has landed (a pool
    /// warm's bound on what it leaves pending; never on a token's path).
    pub fn wait_all_but_latest(&self) {
        for i in 0..=self.lanes.len() {
            let (ev, next) = self.lane(i);
            let target = next.load(Ordering::Acquire).saturating_sub(1);
            while ev.signaledValue() < target {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
    }

    /// Open `path` for loads.
    pub fn open(&self, path: &Path) -> Result<IoFile, MetalKernelError> {
        let s = path.to_str().ok_or_else(|| {
            MetalKernelError::FailedToCreateResource(format!("{}: not UTF-8", path.display()))
        })?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(s));
        let handle = self
            .device
            .as_ref()
            .newIOFileHandleWithURL_error(&url)
            .map_err(|e| {
                MetalKernelError::FailedToCreateResource(format!(
                    "newIOHandle {}: {e}",
                    path.display()
                ))
            })?;
        Ok(IoFile { handle })
    }

    /// Issue `loads` as one IO command buffer, ordered after every command
    /// `commands` has encoded so far and before every command it encodes next.
    /// Returns without waiting.
    pub fn load_ordered(
        &self,
        commands: &Commands,
        loads: &[IoLoad<'_>],
    ) -> Result<(), MetalKernelError> {
        if loads.is_empty() {
            return Ok(());
        }
        let n_lanes = (self.lanes.len() + 1).min(loads.len());
        if n_lanes == 1 {
            let value = self.issue(commands, loads)?;
            return self.await_value(commands, value);
        }
        // The compute signal, committed before any IO buffer waits on it
        // (see `issue`); then one IO buffer per lane, the loads dealt round
        // robin; then a compute wait per lane.
        let compute_value = self.next_compute();
        commands.encode_signal_event(&self.ev_compute, compute_value)?;
        commands.flush_marker()?;
        let per_lane = loads.len().div_ceil(n_lanes);
        let mut waits = Vec::with_capacity(n_lanes);
        for (lane, chunk) in loads.chunks(per_lane).enumerate() {
            let value = self.commit_lane(lane, compute_value, chunk)?;
            waits.push((lane, value));
        }
        for (lane, value) in waits {
            let (ev, _) = self.lane(lane);
            commands.encode_wait_for_event(ev, value)?;
        }
        Ok(())
    }

    /// The next compute-signal value (the compute event is one monotone
    /// counter whatever the lane).
    fn next_compute(&self) -> u64 {
        self.compute_next.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The first half of [`Self::load_ordered`]: issue `loads` ordered after
    /// everything `commands` has encoded so far, and return the `ev_io`
    /// value the loads signal when landed — for [`Self::await_value`] at the
    /// point of use, possibly layers later (the in-flight entry of a hint).
    /// Nothing waits until then; the loads run under the compute between.
    pub fn issue(
        &self,
        commands: &Commands,
        loads: &[IoLoad<'_>],
    ) -> Result<u64, MetalKernelError> {
        let value = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        if loads.is_empty() {
            return Ok(value);
        }
        let compute_value = self.next_compute();
        commands.encode_signal_event(&self.ev_compute, compute_value)?;
        // Commit the compute buffer carrying the signal BEFORE the IO buffer
        // is committed: the IO buffer waits for a signal that runs only once
        // its buffer is committed, and the loads sit pending until then —
        // with their bytes, which the IO queue holds on the device while
        // they wait (measured: a 16 624-expert warm left 35 GB allocated at
        // load beside the 33.75 GB of the pools, #749 P26).
        commands.flush_marker()?;
        let cb = self.queue.commandBuffer();
        cb.waitForEvent_value(&self.ev_compute, compute_value);
        let mut total = 0u64;
        for l in loads {
            if l.dst_offset + l.len > l.dst.length() {
                return Err(MetalKernelError::FailedToCreateResource(format!(
                    "io load: {}+{} past the {}-byte buffer",
                    l.dst_offset,
                    l.len,
                    l.dst.length()
                )));
            }
            // SAFETY: the destination range was bounds-checked against the
            // buffer this handle addresses (`base_offset` folded in), the
            // buffer outlives the command buffer (the caller keeps it), and the
            // source range is the file's — a short read is Metal's error, not
            // memory unsafety.
            unsafe {
                cb.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    l.dst.as_ref(),
                    l.dst.base_offset() + l.dst_offset,
                    l.len,
                    &l.file.handle,
                    l.src_offset as usize,
                );
            }
            total += l.len as u64;
        }
        cb.signalEvent_value(&self.ev_io, value);
        cb.commit();
        self.loads.fetch_add(loads.len() as u64, Ordering::Relaxed);
        self.bytes.fetch_add(total, Ordering::Relaxed);
        Ok(value)
    }

    /// The value the most recent [`Self::issue`] returned (0 before any).
    pub fn last_issued(&self) -> u64 {
        self.next.load(Ordering::Acquire)
    }

    /// Block this thread until the loads issued with `value` have landed —
    /// for the one-off paths (a pool warm) that bound what they leave
    /// pending; never on a token's path.
    pub fn wait_landed(&self, value: u64) {
        while self.signaled() < value {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }

    /// The second half: every command `commands` encodes from here on waits
    /// for the loads [`Self::issue`] returned `value` for. The IO queue is
    /// in order, so waiting for the largest value of a set covers the set.
    pub fn await_value(&self, commands: &Commands, value: u64) -> Result<(), MetalKernelError> {
        commands.encode_wait_for_event(&self.ev_io, value)
    }

    /// The largest `ev_io` value landed so far (a host read of the shared
    /// event): loads issued with a value at or below it are complete, and
    /// need no wait.
    pub fn signaled(&self) -> u64 {
        self.ev_io.signaledValue()
    }

    /// `(loads issued, bytes issued)` so far.
    pub fn counters(&self) -> (u64, u64) {
        (
            self.loads.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }
}

impl std::fmt::Debug for IoQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (loads, bytes) = self.counters();
        write!(f, "IoQueue {{ loads: {loads}, bytes: {bytes} }}")
    }
}

impl std::fmt::Debug for IoFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IoFile")
    }
}
