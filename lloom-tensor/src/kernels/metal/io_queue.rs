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
pub struct IoQueue {
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
    ev_compute: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    ev_io: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    next: AtomicU64,
    device: Device,
    loads: AtomicU64,
    bytes: AtomicU64,
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
        Ok(Self {
            queue,
            ev_compute: event("compute")?,
            ev_io: event("io")?,
            next: AtomicU64::new(0),
            device: device.clone(),
            loads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        })
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
        let value = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        commands.encode_signal_event(&self.ev_compute, value)?;
        let cb = self.queue.commandBuffer();
        cb.waitForEvent_value(&self.ev_compute, value);
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
        commands.encode_wait_for_event(&self.ev_io, value)?;
        self.loads.fetch_add(loads.len() as u64, Ordering::Relaxed);
        self.bytes.fetch_add(total, Ordering::Relaxed);
        Ok(())
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
