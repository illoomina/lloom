//! A host-fed input: a ring of Shared buffers a host thread writes in place,
//! and a shared event that orders the commands reading a slot after the write
//! (`design/docs/expert-streaming.md` §20.13, option A).
//!
//! It is the token ring's stamp run in reverse. For tokens out, the GPU writes
//! a slot and stamps, and the host polls the stamp; here the host writes a
//! slot and signals (`setSignaledValue`). The GPU never waits on that signal:
//! the encoding thread records the commands that read the slot as before, and
//! their commit waits for the value (`Commands::commit_when_produced`); the
//! host thread, once it has raised the event, commits what its value released
//! (`Commands::produced`). The value a forward reads is decided on the
//! encoding thread, and the host thread only ever raises the event
//! (`docs/invariants.md` §15.1q).
//!
//! Values are monotone and 1-based per [`HostFedRing::base`]: the forward that
//! consumes value `v` reads slot `v % slots`. The slots are persistent Shared
//! buffers outside every pool, so no allocation reaches a second tenant while
//! a pending command still reads it (`docs/invariants.md` §15.1m).

use super::MetalDevice;
use crate::kernels::metal::BufferHandle;
use crate::{DType, Result, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent, MTLSharedEventListener};

/// A value no wait of a live run reaches: signalled to release every wait a
/// failed run left encoded (`HostFedRing::release`).
const RELEASE_SPAN: u64 = 1 << 40;

/// A shared event a host thread raises and the command stream's commits wait
/// for — [`HostFedRing`]'s protocol without the ring: the caller owns the
/// buffers (a weight-group tier's slots, `lloom::nn::weight_groups`; an
/// expert fetch's slots, `lloom::nn::expert_fetch`) and the values (one per
/// load, monotone). The loader thread writes a slot, then `signal`s the
/// load's value; the encoding thread calls `commit_when_produced` with that
/// value before the slot's first reader — a cut when the host has not raised
/// it yet (`profiles-and-memory-design.md` §4.8 R1).
pub struct HostEvent {
    device: MetalDevice,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// SAFETY: an `MTLSharedEvent` is usable from any thread; it is what it is for.
unsafe impl Send for HostEvent {}
unsafe impl Sync for HostEvent {}

impl HostEvent {
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let event = device
            .metal_device()
            .as_ref()
            .newSharedEvent()
            .ok_or_else(|| crate::Error::Msg("newSharedEvent returned nil".into()))?;
        Ok(Self {
            device: device.clone(),
            event,
        })
    }

    /// The value the host has raised the event to.
    pub fn signaled(&self) -> u64 {
        self.event.signaledValue()
    }

    /// Raise the event to `v` and commit the work its value released; the
    /// host writes before this call are visible to that work. A release the
    /// queue refuses fails the queue (`Commands::produced`).
    pub fn signal(&self, v: u64) {
        // The writes the value guards before it, for a thread that commits
        // their reader on seeing the value (`Commands::produced`).
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        self.event.setSignaledValue(v);
        let _ = self.device.commands().produced(&self.event, v);
    }

    /// Every command encoded after this on the device's stream is committed
    /// only once the event has reached `v`. Cuts the open recording unless
    /// the host has raised it already.
    pub fn commit_when_produced(&self, v: u64) -> Result<()> {
        self.device
            .commands()
            .commit_when_produced(&self.event, v)
            .map_err(|e| crate::Error::Msg(format!("HostEvent: {e}")))
    }
}

impl std::fmt::Debug for HostEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HostEvent {{ signalled: {} }}",
            self.event.signaledValue()
        )
    }
}

pub struct HostFedRing {
    device: MetalDevice,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    buffers: Vec<BufferHandle>,
    tensors: Vec<Tensor>,
    len: usize,
    /// Every value up to this one belongs to a released run: refused.
    released_upto: std::sync::atomic::AtomicU64,
}

// SAFETY: the event and the buffers are Metal objects usable from any thread.
// A slot's bytes are written by one host thread at a time, and only for a
// value whose previous reader has run (the caller's protocol, `slot_mut`).
unsafe impl Send for HostFedRing {}
unsafe impl Sync for HostFedRing {}

impl HostFedRing {
    /// `slots` buffers of `len` f32 each, and a fresh event at 0.
    pub fn new(device: &MetalDevice, slots: usize, len: usize) -> Result<Self> {
        if slots == 0 || len == 0 {
            crate::bail!("HostFedRing: {slots} slots of {len} floats")
        }
        let event = device
            .metal_device()
            .as_ref()
            .newSharedEvent()
            .ok_or_else(|| crate::Error::Msg("newSharedEvent returned nil".into()))?;
        let bytes = len
            .checked_mul(DType::F32.size_in_bytes())
            .ok_or_else(|| crate::Error::Msg("HostFedRing byte size overflow".into()))?;
        let mut buffers = Vec::with_capacity(slots);
        let mut tensors = Vec::with_capacity(slots);
        for _ in 0..slots {
            let buf = device.new_persistent_shared(bytes)?;
            // SAFETY: the pointer is only compared with null.
            if unsafe { buf.contents() }.is_null() {
                crate::bail!("HostFedRing: a Shared buffer with no CPU mapping")
            }
            let storage = crate::Storage::Metal(super::MetalStorage::new_allocated(
                BufferHandle::clone(&buf),
                device.clone(),
                len,
                DType::F32,
            )?);
            tensors.push(crate::tensor::from_storage(
                storage,
                len,
                crate::op::BackpropOp::none(),
                false,
            ));
            buffers.push(buf);
        }
        Ok(Self {
            device: device.clone(),
            event,
            buffers,
            tensors,
            len,
            released_upto: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn slots(&self) -> usize {
        self.buffers.len()
    }

    /// Floats a slot holds.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The value the event holds now: a run's values start above it.
    pub fn base(&self) -> u64 {
        self.event.signaledValue()
    }

    /// The `[len]` f32 tensor over the slot value `v` fills.
    pub fn tensor(&self, v: u64) -> &Tensor {
        &self.tensors[(v % self.slots() as u64) as usize]
    }

    /// The slot value `v` fills, for the host to write.
    ///
    /// # Safety
    /// No command that reads this slot for an earlier value may still be
    /// pending, and no other thread may hold the slot: the caller writes value
    /// `v` only after the stream has shown that the reader of `v - slots` ran.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, v: u64) -> &mut [f32] {
        let buf = &self.buffers[(v % self.slots() as u64) as usize];
        // SAFETY: a CPU-mapped Shared buffer of `len` f32 (checked at `new`),
        // exclusive to the caller by the contract above.
        unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, self.len) }
    }

    /// Raise the event to `v` and commit the work every value `<= v`
    /// released. The slot writes before it are visible to that work. A
    /// release the queue refuses fails the queue (`Commands::produced`).
    pub fn signal(&self, v: u64) {
        // The slot writes before it, for a thread that commits their reader
        // on seeing the value (`Commands::produced`).
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        self.event.setSignaledValue(v);
        let _ = self.device.commands().produced(&self.event, v);
    }

    /// End a run whose producer will fill no more: the work held for its
    /// values is cancelled — none of it runs (`Commands::cancel_produced`) —
    /// a value of the run is refused from here on, and the event jumps
    /// [`RELEASE_SPAN`] past `base`, so the next run's values start above
    /// every one this run could have used. On the GPU-wait arm nothing is
    /// held, and the jump lets the encoded waits through as before.
    pub fn release(&self, base: u64) {
        let v = base.saturating_add(RELEASE_SPAN);
        self.released_upto
            .fetch_max(v, std::sync::atomic::Ordering::AcqRel);
        if let Err(e) = self.device.commands().cancel_produced_through(
            &self.event,
            v,
            "host feed: the run was released; the work held for its fill was cancelled",
        ) {
            eprintln!("[host-fed] cancelling the released run's held work: {e}");
        }
        if self.event.signaledValue() < v {
            self.event.setSignaledValue(v);
        }
    }

    /// Every command encoded after this on the device's stream is committed
    /// only once the event has reached `v`. Cuts the open recording unless
    /// the host has raised it already; refused for a value of a released run.
    pub fn commit_when_produced(&self, v: u64) -> Result<()> {
        let released = &self.released_upto;
        self.device
            .commands()
            .commit_when_produced_unless(&self.event, v, || {
                if v <= released.load(std::sync::atomic::Ordering::Acquire) {
                    Err(crate::kernels::MetalKernelError::QueueUnavailable(
                        "a value of a released host-fed run",
                    ))
                } else {
                    Ok(())
                }
            })
            .map_err(|e| crate::Error::Msg(format!("HostFedRing: {e}")))
    }

    /// A GPU wait for value `v`, which the host has raised already: the cost
    /// of a satisfied wait, for an instrument. Refused while `v` is pending —
    /// a GPU wait on a host signal not yet raised is what
    /// [`Self::commit_when_produced`] exists to avoid.
    pub fn encode_satisfied_wait(&self, v: u64) -> Result<()> {
        if self.event.signaledValue() < v {
            crate::bail!("HostFedRing: a GPU wait on value {v}, not yet raised")
        }
        self.device
            .commands()
            .encode_wait_for_event(&self.event, v)
            .map_err(|e| crate::Error::Msg(format!("HostFedRing wait: {e}")))
    }
}

impl std::fmt::Debug for HostFedRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HostFedRing {{ slots: {}, len: {}, signalled: {} }}",
            self.slots(),
            self.len,
            self.event.signaledValue()
        )
    }
}

/// A GPU→host event at a point in the command stream: the command stream
/// signals it after every command encoded before [`PublishEvent::encode_signal`],
/// and a host thread parks on the value (`waitUntilSignaledValue:timeoutMS:`)
/// instead of polling a word for it.
///
/// It is how a host waiter learns of the token ring's stamp without spinning
/// (`design/docs/expert-streaming.md` §20.14b): a compute kernel has no signal
/// primitive (§3.7b), but the command buffer signals at an encoder boundary,
/// right behind the stamp's dispatch, and the stamp's writes are visible to the
/// host once the value is observed. Values are the caller's, monotone.
pub struct PublishEvent {
    device: MetalDevice,
    clock: std::sync::Arc<crate::kernels::metal::GpuClock>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Delivers [`Self::notify_at`]'s blocks; held for the event's life, since
    /// a dropped listener silently stops delivering.
    listener: Retained<MTLSharedEventListener>,
    /// The command-buffer epoch the last signal was encoded in.
    encoded_epoch: std::sync::atomic::AtomicU64,
}

// SAFETY: as `HostFedRing`'s — an `MTLSharedEvent` is usable from any thread,
// and the only other state is an atomic.
unsafe impl Send for PublishEvent {}
unsafe impl Sync for PublishEvent {}

impl PublishEvent {
    /// A fresh event at 0 on `device`'s command stream.
    pub fn new(device: &MetalDevice) -> Result<Self> {
        let event = device
            .metal_device()
            .as_ref()
            .newSharedEvent()
            .ok_or_else(|| crate::Error::Msg("newSharedEvent returned nil".into()))?;
        Ok(Self {
            device: device.clone(),
            clock: device.gpu_clock(),
            event,
            listener: MTLSharedEventListener::new(),
            encoded_epoch: std::sync::atomic::AtomicU64::new(u64::MAX),
        })
    }

    /// Signal `value` once every command encoded so far on the device's
    /// stream has completed. Ends the open compute encoder.
    pub fn encode_signal(&self, value: u64) -> Result<()> {
        let epoch = self.clock.current_epoch();
        self.device
            .commands()
            .encode_signal_event(&self.event, value)
            .map_err(|e| crate::Error::Msg(format!("PublishEvent signal: {e}")))?;
        self.encoded_epoch
            .store(epoch.0, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Whether the last signal encoded is in a committed command buffer, so a
    /// wait for it can be satisfied without the waiter committing.
    pub fn committed(&self) -> bool {
        let e = self
            .encoded_epoch
            .load(std::sync::atomic::Ordering::Acquire);
        e == u64::MAX || self.clock.current_epoch().0 > e
    }

    /// The value the GPU has signalled so far.
    pub fn value(&self) -> u64 {
        self.event.signaledValue()
    }

    /// Park until the event reaches `value` or `timeout_ms` passes. `false`
    /// is a timeout, never completion.
    ///
    /// Woken by a listener block rather than by `waitUntilSignaledValue:`,
    /// which returned ~1 ms after the signal on the decode path
    /// (`expert-streaming.md` §20.14b).
    pub fn wait(&self, value: u64, timeout_ms: u64) -> bool {
        if self.event.signaledValue() >= value {
            return true;
        }
        let slot =
            std::sync::Arc::new((parking_lot::Mutex::new(false), parking_lot::Condvar::new()));
        let s = std::sync::Arc::clone(&slot);
        self.notify_at(value, move || {
            *s.0.lock() = true;
            s.1.notify_all();
        });
        let mut landed = slot.0.lock();
        if !*landed {
            let _ = slot.1.wait_while_for(
                &mut landed,
                |l| !*l,
                std::time::Duration::from_millis(timeout_ms),
            );
        }
        *landed || self.event.signaledValue() >= value
    }

    /// Run `f` on a dispatch-queue thread once the event reaches `value` (at
    /// once if it already has): for a waiter that parks on a wake of its own
    /// and has another reason to wake beside the GPU's.
    pub fn notify_at<F: Fn() + Send + Sync + 'static>(&self, value: u64, f: F) {
        let block = block2::RcBlock::new(
            move |_event: std::ptr::NonNull<ProtocolObject<dyn MTLSharedEvent>>, _seen: u64| f(),
        );
        // SAFETY: `block` is valid for the call and Metal copies it, so it may
        // run after this frame (or this event) is gone; it owns what it
        // captures.
        unsafe {
            self.event.notifyListener_atValue_block(
                &self.listener,
                value,
                &*block as *const _ as *mut _,
            );
        }
    }
}

impl std::fmt::Debug for PublishEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PublishEvent {{ value: {} }}",
            self.event.signaledValue()
        )
    }
}
