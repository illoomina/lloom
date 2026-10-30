//! `MTLIOCommandQueue` loads — file bytes straight into a device buffer as
//! queued commands, ordered against the compute stream by shared events
//! (`design/docs/expert-streaming.md` §20.4a, candidate B).
//!
//! A load is issued on the IO queue and never waited for on the host. The
//! compute stream signals `ev_compute` after everything it has encoded so far
//! (the slot's earlier readers), and the IO command buffer waits for that — a
//! signal the GPU produces, committed before the IO buffer is
//! (`Commands::commit_io_after`) — then loads and signals `ev_io`. The compute
//! work that reads the loaded bytes does not wait for `ev_io` on the GPU: it
//! is recorded as before and committed once the event has reached the load's
//! value (`Commands::commit_when_produced`), by the event's own notification
//! (`docs/invariants.md` §15.1q). Both queues are in-order, so one monotone
//! value per `load_ordered` call serialises the whole exchange. No
//! `waitUntilCompleted` on this path.

use super::{Buffer, Commands, Device, HostInputArm};
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
    /// The file's length when it was opened: a load past it completes
    /// without an error and lands only the bytes that exist, so it is
    /// refused before it is issued (`checked_source_range`).
    len: u64,
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

/// Validate byte arithmetic without creating native objects. The raw extent is
/// checked separately because a view's logical extent is not the allocation.
fn checked_load_range(
    base: usize,
    view_len: usize,
    raw_len: usize,
    dst_offset: usize,
    src_offset: u64,
    len: usize,
) -> Result<(), MetalKernelError> {
    let refused = || {
        MetalKernelError::FailedToCreateResource(
            "io load byte range exceeds destination or address space".into(),
        )
    };
    let logical_end = dst_offset.checked_add(len).ok_or_else(refused)?;
    if logical_end > view_len {
        return Err(refused());
    }
    let physical_end = base.checked_add(logical_end).ok_or_else(refused)?;
    if physical_end > raw_len {
        return Err(refused());
    }
    let source = usize::try_from(src_offset).map_err(|_| refused())?;
    source.checked_add(len).ok_or_else(refused)?;
    Ok(())
}

/// A load's source range against the file's length. Metal reports a load
/// past the end as complete, with only the bytes that exist landed, so the
/// work behind it would read a slot the load never wrote.
fn checked_source_range(
    src_offset: u64,
    len: usize,
    file_len: u64,
) -> Result<(), MetalKernelError> {
    let end = u64::try_from(len)
        .ok()
        .and_then(|len| src_offset.checked_add(len));
    match end {
        Some(end) if end <= file_len => Ok(()),
        _ => Err(MetalKernelError::FailedToCreateResource(format!(
            "io load source range {src_offset}+{len} passes the file's end ({file_len} bytes)"
        ))),
    }
}

/// Entire-batch validation precedes the first compute signal or IO submission,
/// including ranges assigned to later lanes. No descriptor copy is needed:
/// IoLoad and Buffer extents remain immutable throughout the borrowed batch.
fn validate_loads(loads: &[IoLoad<'_>]) -> Result<(), MetalKernelError> {
    let mut total = 0u64;
    for load in loads {
        checked_load_range(
            load.dst.base_offset(),
            load.dst.length(),
            load.dst.native_allocation_length(),
            load.dst_offset,
            load.src_offset,
            load.len,
        )?;
        checked_source_range(load.src_offset, load.len, load.file.len)?;
        total = total
            .checked_add(u64::try_from(load.len).map_err(|_| {
                MetalKernelError::FailedToCreateResource("io load byte count overflow".into())
            })?)
            .ok_or_else(|| {
                MetalKernelError::FailedToCreateResource("io batch byte count overflow".into())
            })?;
    }
    Ok(())
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
        crate::tuning::try_tuning()
            .map_err(|error| MetalKernelError::InvalidInput(error.to_string()))?;
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
    /// (returned), committed once that signal is (`Commands::commit_io_after`).
    /// Caller has validated the ENTIRE batch before any lane is submitted; this
    /// method performs no fallible range discovery.
    fn commit_lane(
        &self,
        commands: &Commands,
        gpu_wait: bool,
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
            // SAFETY: as in `issue`.
            unsafe {
                cb.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    l.dst.raw_unchecked(),
                    l.dst.base_offset() + l.dst_offset,
                    l.len,
                    &l.file.handle,
                    l.src_offset as usize,
                );
            }
            total += l.len as u64;
        }
        cb.signalEvent_value(ev, value);
        if gpu_wait {
            // The control arm: the signal's buffer was committed by the
            // caller's flush, before the first lane.
            cb.commit();
        } else {
            commands.commit_io_after(&cb, &self.ev_compute, compute_value)?;
        }
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
            // Parked on the lane's event; the timeout only re-tests.
            while !ev.waitUntilSignaledValue_timeoutMS(target, 1_000) {}
        }
    }

    /// Open `path` for loads.
    pub fn open(&self, path: &Path) -> Result<IoFile, MetalKernelError> {
        let s = path.to_str().ok_or_else(|| {
            MetalKernelError::FailedToCreateResource(format!("{}: not UTF-8", path.display()))
        })?;
        let len = std::fs::metadata(path)
            .map_err(|e| {
                MetalKernelError::FailedToCreateResource(format!("{}: {e}", path.display()))
            })?
            .len();
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
        Ok(IoFile { handle, len })
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
        validate_loads(loads)?;
        // The compute signal, committed before any IO buffer waits on it
        // (see `issue`); then one IO buffer per lane, the loads dealt round
        // robin; then the compute work behind every lane's load.
        let compute_value = self.next_compute();
        commands.encode_signal_event(&self.ev_compute, compute_value)?;
        let gpu_wait = commands.host_input_arm() == HostInputArm::GpuWait;
        if gpu_wait {
            commands.flush_marker()?;
        }
        let per_lane = loads.len().div_ceil(n_lanes);
        let mut waits = Vec::with_capacity(n_lanes);
        for (lane, chunk) in loads.chunks(per_lane).enumerate() {
            let value = self.commit_lane(commands, gpu_wait, lane, compute_value, chunk)?;
            waits.push((lane, value));
        }
        for (lane, value) in waits {
            let (ev, _) = self.lane(lane);
            commands.commit_when_produced(ev, value)?;
            commands.notify_produced_at(ev, value)?;
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
        // Zero is already complete on every lane. An empty batch must not
        // reserve a fresh value that no command buffer will ever signal.
        if loads.is_empty() {
            return Ok(0);
        }
        validate_loads(loads)?;
        let value = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        let compute_value = self.next_compute();
        commands.encode_signal_event(&self.ev_compute, compute_value)?;
        // The compute buffer carrying the signal is committed BEFORE the IO
        // buffer (`commit_io_after` below): the IO buffer waits for a signal
        // that runs only once its buffer is committed, and the loads sit
        // pending until then — with their bytes, which the IO queue holds on
        // the device while they wait (measured: a 16 624-expert warm left
        // 35 GB allocated at load beside the 33.75 GB of the pools, #749 P26).
        let cb = self.queue.commandBuffer();
        cb.waitForEvent_value(&self.ev_compute, compute_value);
        let mut total = 0u64;
        for l in loads {
            // SAFETY: whole-batch preflight checked both logical and raw
            // destination extents plus source address arithmetic. The
            // buffer outlives the command buffer (the caller keeps it), and the
            // source range is the file's — a short read is Metal's error, not
            // memory unsafety.
            unsafe {
                cb.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                    l.dst.raw_unchecked(),
                    l.dst.base_offset() + l.dst_offset,
                    l.len,
                    &l.file.handle,
                    l.src_offset as usize,
                );
            }
            total += l.len as u64;
        }
        cb.signalEvent_value(&self.ev_io, value);
        commands.commit_io_after(&cb, &self.ev_compute, compute_value)?;
        self.loads.fetch_add(loads.len() as u64, Ordering::Relaxed);
        self.bytes.fetch_add(total, Ordering::Relaxed);
        Ok(value)
    }

    /// The latest reserved nonempty IO value (0 before any). Empty issues do
    /// not change it. Reservation does not itself prove native submission.
    pub fn last_issued(&self) -> u64 {
        self.next.load(Ordering::Acquire)
    }

    /// Block this thread until the loads issued with `value` have landed —
    /// for the one-off paths (a pool warm) that bound what they leave
    /// pending; never on a token's path.
    pub fn wait_landed(&self, value: u64) {
        // Parked on the IO event; the timeout only re-tests.
        while !self.ev_io.waitUntilSignaledValue_timeoutMS(value, 1_000) {}
    }

    /// The second half: every command `commands` encodes from here on is
    /// committed once the loads [`Self::issue`] returned `value` for have
    /// landed — by `ev_io`'s notification at that value, not by a GPU wait.
    /// The IO queue is in order, so the largest value of a set covers the set.
    pub fn await_value(&self, commands: &Commands, value: u64) -> Result<(), MetalKernelError> {
        commands.commit_when_produced(&self.ev_io, value)?;
        commands.notify_produced_at(&self.ev_io, value)
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

#[cfg(test)]
mod preflight_tests {
    use super::{checked_load_range, checked_source_range};

    #[test]
    fn a_source_range_past_the_files_end_is_refused() {
        assert!(checked_source_range(0, 16, 16).is_ok());
        assert!(checked_source_range(8, 8, 16).is_ok());
        assert!(checked_source_range(16, 0, 16).is_ok());
        assert!(checked_source_range(16, 1, 16).is_err());
        assert!(checked_source_range(8, 16, 16).is_err());
        assert!(checked_source_range(u64::MAX, 1, u64::MAX).is_err());
    }

    #[test]
    fn view_and_raw_extents_are_checked_independently() {
        assert!(checked_load_range(16, 32, 64, 8, 4, 24).is_ok());
        assert!(checked_load_range(16, 32, 64, 9, 4, 24).is_err());
        assert!(checked_load_range(48, 32, 64, 8, 4, 24).is_err());
        assert!(checked_load_range(64, 0, 64, 0, 0, 0).is_ok());
    }

    #[test]
    fn offset_additions_refuse_wrap_without_allocating() {
        assert!(checked_load_range(0, usize::MAX, usize::MAX, usize::MAX, 0, 1).is_err());
        assert!(checked_load_range(usize::MAX, 1, usize::MAX, 0, 0, 1).is_err());
        assert!(checked_load_range(0, 1, 1, 0, u64::MAX, 1).is_err());
        assert!(checked_load_range(0, 1, 1, 0, (usize::MAX - 1) as u64, 1).is_ok());
    }
}
