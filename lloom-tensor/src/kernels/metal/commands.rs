#[path = "entry_guard.rs"]
mod entry_guard;
#[path = "commands_lifecycle.rs"]
mod lifecycle;

use super::in_flight::{CompletedRelease, Held};
use super::residency_set::PublishedRemovals;
use entry_guard::{DeferredGuard, DeferredSlots};

use crate::current_command::Current;
use crate::kernels::metal::{
    cache_pad::CachePadded,
    completion::{CompletionWait, SharedEvent, WaitOutcome},
    contention::{Access, Censused, Side, SubmissionContention},
    executor::ExecutorSlot,
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    DispatchGate, Fence, GpuClock, InFlightHold, PrevCeOutputs, ResidencySet,
};
use crate::kernels::MetalKernelError;
use crate::retire::Epoch;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

// The cut periods' defaults (50 blocking/prefill, 512 pipelined) live in
// `crate::tuning` beside the knobs that override them, one definition each.
use crate::tuning::DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER;

pub use crate::command_policy::CutPolicy;

/// Whether the queue hands out command buffers that do NOT retain the
/// resources encoded into them (`commandBufferWithUnretainedReferences`).
///
/// Opt-in through `LLOOM_CB_UNRETAINED=1` (`--cb-unretained`), read once. A
/// retaining command buffer takes a strong reference to every buffer, fence and
/// pipeline it sees and releases them on completion: objc retain traffic per
/// encode that buys nothing when the host already holds each of them until the
/// command buffer has run. The unretained form rests on that claim, and the
/// claim is ours to keep, object class by object class:
///
/// - pool buffers are parked until their epoch completes (`BufferPool`);
/// - arena and region bases, weights and slot pools live for their owner, and
///   a base being replaced goes to the [`InFlightHold`];
/// - a buffer outside every pool (an unpooled upload, a persistent allocation)
///   goes to the hold when its last handle drops;
/// - an encoder's fence goes to the hold when its last handle drops, since
///   encoders in later command buffers may still wait on it;
/// - pipelines live for the process.
///
/// The hold releases each object once the epoch open at its drop has completed
/// (`in_flight.rs`). A path whose objects it cannot hold refuses the kind by
/// name. The validation layer checks the claim on a real run
/// (`MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES=5`), and
/// `lloom-fuzz`'s `cb_liveness` drives every class through both kinds.
pub fn cb_unretained() -> bool {
    static KIND: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| crate::tuning::tuning().metal.cb_unretained)
}

/// Which command buffers a [`Commands`] creates: Metal's retaining default,
/// or unretained with the host holding what they read ([`InFlightHold`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandBufferKind {
    Retaining,
    Unretained,
}

impl CommandBufferKind {
    /// The kind the process selected (`cb_unretained`).
    pub fn selected() -> Self {
        if cb_unretained() {
            CommandBufferKind::Unretained
        } else {
            CommandBufferKind::Retaining
        }
    }
}

fn create_command_buffer(
    command_queue: &CommandQueue,
    kind: CommandBufferKind,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>, MetalKernelError> {
    let raw = match kind {
        CommandBufferKind::Unretained => command_queue.commandBufferWithUnretainedReferences(),
        CommandBufferKind::Retaining => command_queue.commandBuffer(),
    };
    raw.ok_or_else(|| MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()))
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: EntryGuard<'a>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
    pub(crate) fn retain_namespace(
        &mut self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        if owner.already_recorded(&self.guard.namespace) {
            return Ok(());
        }
        self.guard
            .current
            .recording()
            .ok_or(MetalKernelError::QueueUnavailable(
                "namespace attachment requires recording generation",
            ))?
            .retain_namespace(owner.try_clone()?)?;
        owner.remember(&mut self.guard.namespace)
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.as_ref().set_compute_pipeline_state(pipeline);
    }

    #[cfg(feature = "debug-labels")]
    #[must_use = "the debug group is popped when the returned guard is dropped"]
    pub fn debug_group(&self, label: &str) -> crate::kernels::metal::DebugGroupGuard<'_> {
        self.as_ref().debug_group(label)
    }
}

/// RAII guard for blit command encoder operations.
pub struct BlitCommandsGuard<'a> {
    _guard: EntryGuard<'a>,
    state: Option<BlitCommandEncoder>,
    core: &'a CommandsCore,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self.state.as_ref().expect("live blit guard")
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        self.state.as_mut().expect("live blit guard")
    }
}

impl BlitCommandsGuard<'_> {
    pub(crate) fn retain_namespace(
        &mut self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        if owner.already_recorded(&self._guard.namespace) {
            return Ok(());
        }
        self._guard
            .current
            .recording()
            .ok_or(MetalKernelError::QueueUnavailable(
                "namespace attachment requires recording generation",
            ))?
            .retain_namespace(owner.try_clone()?)?;
        owner.remember(&mut self._guard.namespace)
    }

    /// Explicit completion returns refusal; unfinished native custody remains in
    /// Commands for retry or failed-queue quarantine, including unwind.
    pub fn finish(mut self) -> Result<(), MetalKernelError> {
        self.stage_close();
        self.core.close_pending_blit(&mut self._guard)
    }

    fn stage_close(&mut self) {
        if let Some(encoder) = self.state.take() {
            // Constructor drained any previous pending blit under this same
            // EntryState guard; no other producer can install one meanwhile.
            self._guard.pending_blit = Some(PendingBlit {
                encoder,
                error: None,
            });
        }
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    /// # Safety
    /// Legacy/raw interoperability only. Retain authentic originating reuse
    /// authority, native/accounting owners and all future emission/raw references
    /// through terminal work. A derived bare Buffer is not that authority.
    pub unsafe fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        // SAFETY: this explicitly unsafe guard API preserves the raw setter's
        // caller obligations; it is not a safe Tensor ownership-tracked route.
        unsafe {
            self.as_mut()
                .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
        }
    }

    /// # Safety
    /// Same originating reuse/terminal/future-emission duties as copy_from_buffer.
    /// No counted pin is installed by this legacy/raw interoperability API.
    pub unsafe fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        unsafe { self.as_mut().fill_buffer(buffer, range, value) };
    }

    pub(crate) fn copy_from_handles_aligned(
        &mut self,
        source: &super::BufferHandle,
        source_offset: usize,
        destination: &super::BufferHandle,
        destination_offset: usize,
        size: usize,
    ) -> Result<(), MetalKernelError> {
        self.as_mut().copy_from_handles_aligned(
            source,
            source_offset,
            destination,
            destination_offset,
            size,
        )
    }
    pub(crate) fn fill_handle_aligned(
        &mut self,
        destination: &super::BufferHandle,
        range: (usize, usize),
        value: u8,
    ) -> Result<(), MetalKernelError> {
        self.as_mut().fill_handle_aligned(destination, range, value)
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        if self.state.is_none() {
            return;
        }
        self.stage_close();
        if std::thread::panicking() {
            self.core
                .failure
                .record_static("blit guard unwound before close; custody retained");
            return;
        }
        if let Err(error) = self.core.close_pending_blit(&mut self._guard) {
            if !matches!(
                error,
                MetalKernelError::RecordingAdmission(
                    crate::command_lifecycle::LifecycleError::Busy
                )
            ) {
                self._guard
                    .pending_blit
                    .as_mut()
                    .expect("refused close retains custody")
                    .error = Some(error);
            }
        }
    }
}

/// A committed command buffer a caller can wait on by itself.
///
/// Taken by [`Commands::flush_marker`] once a result is fully encoded: waiting
/// on it waits for that result and for nothing committed after it. The value
/// [`Commands::flush_and_wait_current`] waits on is the queue's tail, which is
/// the right wait for a single stream and the wrong one for two interleaved.
#[derive(Clone, Debug)]
pub struct CommandMarker {
    cb: CommandBuffer,
    /// The shared-event value encoded on the buffer for the async completion
    /// arms; `None` on the blocking arm.
    signalled: Option<u64>,
}

impl CommandMarker {
    /// Whether the marker's buffer has completed (successfully or not).
    pub fn is_complete(&self) -> bool {
        self.cb.status().0 >= MTLCommandBufferStatus::Completed.0
    }

    /// Nonblocking known-error observation; false also includes pending work.
    pub fn is_failed(&self) -> bool {
        self.cb.status() == MTLCommandBufferStatus::Error
    }

    /// The buffer's GPU end time (`CLOCK_UPTIME_RAW` seconds), valid once
    /// [`Self::is_complete`].
    pub fn gpu_end_time(&self) -> f64 {
        self.cb.gpu_end_time()
    }
}

#[derive(Clone, Copy)]
struct CommitReceipt {
    epoch: Epoch,
    signalled: Option<u64>,
}

struct PendingBlit {
    encoder: BlitCommandEncoder,
    error: Option<MetalKernelError>,
}

struct EntryState {
    pending_blit: Option<PendingBlit>,
    current: Current<CommandBuffer, CommitReceipt>,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
    namespace: crate::metal_backend::device_retirement::NamespaceMemo,
    // Per-call lookahead bound: recovery + one explicit cut can consume TWO.
    // This is not a global generation/cardinality cap. Untracked Core keeps None.
    binding_credits: [Option<super::native_generation_bindings::PreparedGeneration>; 2],
}

// One acquisition can close one compute encoder and construct one replacement.
// A blit guard additionally closes one pending predecessor and its own encoder.
#[derive(Default)]
struct ClosedEncoders<'a> {
    compute: DeferredSlots<ComputeCommandEncoder, 1>,
    blit: DeferredSlots<PendingBlit, 2>,
    executor: DeferredSlots<Arc<ExecutorSlot>, 1>,
    removals: DeferredSlots<PublishedRemovals, 1>,
    // Last: closed wrappers can hand their fields to this same hold before
    // its existing completed-prefix drain runs. No release occurs under state.
    completed_hold: Option<CompletedRelease<'a, Held>>,
}

type EntryGuard<'a> = DeferredGuard<parking_lot::MutexGuard<'a, EntryState>, ClosedEncoders<'a>>;

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            pending_blit: None,
            current: Current::new(cb),
            in_flight: vec![],
            current_encoder: None,
            namespace: Default::default(),
            binding_credits: std::array::from_fn(|_| None),
        }
    }
}

pub struct Commands {
    core: lifecycle::OwnedCore<CommandsCore>,
}

struct CommandsCore {
    // Actual control owner; neither Generation/Page C retains this bank/Core.
    binding_bank: Option<super::native_generation_bindings::BindingBank>,
    // Retain service notification custody independently of installed state and
    // encoder-domain observers. Notifications flow through those observers;
    // this field is an owner, not a notification call site.
    _retirement_wake: Arc<crate::completion_wake::Wake>,
    encoder_domain: Arc<super::encoder::EncoderDomain>,
    /// The encoder state, and `compute_count` beside it, are on **different
    /// cache lines** (issue #442).
    ///
    /// A lock word and a counter, both touched on the per-dispatch path:
    /// `command_encoder` takes `state` and then does `compute_count.fetch_add`
    /// (`:244`, `:245`), **~555 times per token** at §13.1a's decode census.
    /// At B=1 that is one thread, so this is *structural rather than live* --
    /// and the structure is what §11.3o and #317 put multiple threads through
    /// one `Arc<Commands>` at B>1, where a `Mutex` handoff and a counter
    /// increment on one line is exactly the shape that costs.
    ///
    /// Modelled on this hardware -- a lock word plus a counter, one thread on
    /// each -- **15.89 Mops/s adjacent against 51.06 padded, so padding buys
    /// 3.21x**. Zero instructions, zero allocation, zero effect on what is
    /// computed.
    ///
    /// **Stated as latent rather than collected**: at B=1 the registered
    /// prediction (§17) is a null, and it is padded now because the cost is
    /// 128 B *once* for the process and retrofitting a layout after a
    /// concurrency change is how the measurement gets attributed to the wrong
    /// thing.
    state: CachePadded<Censused<parking_lot::Mutex<EntryState>>>,
    /// Its own line -- see `state` above.
    compute_count: CachePadded<AtomicUsize>,
    replacement_pending: CachePadded<AtomicBool>,
    command_queue: CommandQueue,
    modern_queue: Option<Arc<super::metal4::Queue>>,
    legacy_completions: Option<Arc<super::metal3_retirement::Completions>>,
    /// The Metal 3 queue's one completion block, copied into every buffer.
    legacy_handler: Option<Arc<super::metal3_retirement::NativeHandler>>,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: AtomicUsize,
    /// The pipelined loop's period (`CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER`,
    /// default 512); see `DEFAULT_CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER`.
    pipelined_compute_per_buffer: usize,
    /// Where command buffers are cut (`DESIGN.md` §13.5m, issue #420 stage 3).
    ///
    /// Read once from the environment at construction, like
    /// `compute_per_buffer` beside it. `DispatchCounter` unless
    /// `CANDLE_METAL_NODES_PER_CB` is set, so **the default path is the one
    /// that ran before this field existed** (§7.1a) rather than one equivalent
    /// to it.
    cut_policy: CutPolicy,
    /// Set by [`Commands::mark_graph_node`] when the model reaches a node the
    /// partition cuts at; consumed by the next `command_encoder`.
    ///
    /// A flag rather than a direct commit because the mark arrives *between*
    /// dispatches, where no encoder is open and cutting would mean closing an
    /// encoder only to reopen it for the same work. Deferring to the next
    /// encoder request puts the seam exactly at the first dispatch of the new
    /// run, which is the position the partition names.
    node_cut_pending: AtomicBool,
    device: Device,
    /// The EOS gate (lloom #735 step 4), shared with every encoder this
    /// `Commands` opens; disarmed and unallocated until a decode loop arms it.
    eos_gate: Arc<DispatchGate>,
    /// Shared with escaped encoders without a Commands/Core backedge.
    encoder_finalizer: Arc<EncoderFinalizer>,
    /// The queue's residency set, shared with the device that stages allocations
    /// into it. Held so a command-buffer cut can publish deferred allocations:
    /// `commit_swap_locked` calls [`ResidencySet::commit_pending_extracted`] before it
    /// commits, so a pool MISS's `add` (`DESIGN.md` §6.3b) is made resident
    /// before the command buffer that binds it executes, one commit per cut
    /// rather than one per missed buffer.
    residency_set: Arc<ResidencySet>,
    /// How far the GPU has got through the command buffers submitted here.
    ///
    /// Shared with the buffer pools, which use it to decide when a released
    /// buffer is safe to hand out again. This is the only place it advances.
    clock: Arc<GpuClock>,
    /// This queue's own dispatch census (`profile::DeviceCensus`), handed to
    /// every compute encoder this creates while profiling is on.
    census: Arc<crate::kernels::metal::profile::DeviceCensus>,
    /// This queue's own dispatch trace (`trace::DeviceTrace`), handed to
    /// every compute encoder this creates while the trace switch is on.
    trace: Arc<crate::kernels::metal::trace::DeviceTrace>,
    /// What every command buffer committed here runs on completion: the
    /// failure latch, the clock, the subscribers. Built once and installed in
    /// the completion adapter, so a commit allocates and locks nothing for it
    /// (`completion_hook.rs`).
    hook: Arc<super::completion_hook::CompletionHook>,
    /// How dispatches reach the GPU (`DESIGN.md` §11.1). Handed to every
    /// compute encoder this creates.
    ///
    /// `Mutex` rather than a plain field because installing an executor is a
    /// setup-time action on a `&Commands` that is already shared; it is never
    /// touched per dispatch, because the encoder holds its own `Arc` to the
    /// slot from creation.
    executor: parking_lot::Mutex<Arc<ExecutorSlot>>,
    /// How the host waits for a committed command buffer (`DESIGN.md` §11.5o,
    /// issue #559).
    ///
    /// Read once from the environment at construction, like `compute_per_buffer`
    /// and `cut_policy` beside it. [`CompletionWait::Synchronize`] unless
    /// `CANDLE_METAL_COMPLETION_WAIT` is set, so **the default path is the one
    /// that ran before this field existed** (§7.1a) rather than one equivalent
    /// to it -- every canonical digest and every `.bench/` row belongs to it.
    completion_wait: CompletionWait,
    /// The event the async arms signal, allocated **only** when an async arm is
    /// selected.
    ///
    /// `None` on the default path, where nothing is allocated and nothing is
    /// encoded, and `None` also when the device declines to create one --
    /// which is a capability absence rather than an error, so
    /// [`CompletionWait::wait_for`] falls back to the blocking wait.
    completion_event: Option<SharedEvent>,
    /// Polls performed by the async arms since construction (§2.4: an
    /// instrument that cannot be shown to have engaged has not measured
    /// anything).
    ///
    /// **Zero is a distinguishing value, not a null.** `Synchronize` reports 0
    /// because it never polls; an async arm reporting 0 has either not run or
    /// observed every signal on its first read, and `completion_signals`
    /// beside it separates the two. #500's whole finding is that a green board
    /// says nothing when nothing is counted.
    completion_polls: AtomicU64,
    /// Signals encoded by the async arms since construction.
    ///
    /// The engagement counter proper: a run selecting an async arm and
    /// reporting `completion_signals = 0` has **not taken that arm**, whatever
    /// its `RESULT` line says it selected. That is #513's `ring_publishes` in
    /// this axis, and §7.1a-i's wrong-arm shape is what it exists to catch.
    completion_signals: AtomicU64,
    /// Notification blocks that fired, for the listener arm (#583).
    ///
    /// **The listener arm's engagement proof, and `completion_polls` cannot be
    /// it.** That counter reads 0 on this arm whether the block arrived or
    /// never did, so on this arm the poll count is not evidence — a listener
    /// that silently stopped delivering and fell back to blocking is
    /// indistinguishable from a working park by every counter that existed
    /// before this one. §3.1c's rule in a counter: a zero is not evidence
    /// unless something nonzero came from the same command.
    completion_fires: AtomicU64,
    /// Waits that found the command buffer already complete, so no arm ran
    /// (#583).
    ///
    /// **Measured 0 on the decode path**, and kept because it is the arm that
    /// falsified the first explanation for `completion_fires` sitting an order
    /// of magnitude below `completion_signals`. The buffers are not already
    /// complete; the two counters simply count different things — commits
    /// against waits, ~13.5:1 under `CutPolicy::Counter(50)`.
    completion_already_complete: AtomicU64,
    /// Retaining or unretained command buffers, fixed at construction.
    kind: CommandBufferKind,
    /// What the unretained kind's command buffers cannot hold for themselves;
    /// `None` on the retaining kind, where Metal holds it.
    hold: Option<Arc<InFlightHold>>,
    /// The first command buffer that ended in error, latched by its completion
    /// handler so every later commit and wait reports it.
    failure: Arc<CommandBufferFailure>,
    // LAST: native wrapper fields release before this submit-capability pin.
    physical: crate::retirement_service::PhysicalAnchor<
        crate::metal_backend::device_retirement::NamespaceOwner,
    >,
}

/// A command buffer of this queue ended in error.
///
/// Raised on Metal's completion thread and read on the host. Once raised it
/// stays raised: the failed buffer's writes never happened, so nothing
/// computed after it can be trusted, whichever wait the host next makes.
#[derive(Default)]
pub(super) struct CommandBufferFailure {
    failed: AtomicBool,
    message: std::sync::OnceLock<CommandFailureMessage>,
}
enum CommandFailureMessage {
    Native(String),
    Host(&'static str),
}

/// A read-only observation of this queue's existing monotone failure latch.
/// Holds only the existing AtomicBool/OnceLock error-message owner, never a
/// Commands, native queue, completion adapter, subscriber, buffer or device.
/// False means no failure has been published here yet, NOT successful work.
#[derive(Clone)]
pub struct CommandFailureObserver(Arc<CommandBufferFailure>);
impl CommandFailureObserver {
    pub fn has_failed(&self) -> bool {
        self.0.failed.load(Ordering::Acquire)
    }
}
impl std::fmt::Debug for CommandFailureObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandFailureObserver")
            .field("has_failed", &self.has_failed())
            .finish()
    }
}

impl CommandBufferFailure {
    pub(super) fn record(&self, message: String) {
        let _ = self.message.set(CommandFailureMessage::Native(message));
        self.failed.store(true, Ordering::Release);
    }

    pub(super) fn record_static(&self, message: &'static str) {
        let _ = self.message.set(CommandFailureMessage::Host(message));
        self.failed.store(true, Ordering::Release);
    }

    pub(super) fn check(&self) -> Result<(), MetalKernelError> {
        if !self.failed.load(Ordering::Acquire) {
            return Ok(());
        }
        let msg = self
            .message
            .get()
            .map(|message| match message {
                CommandFailureMessage::Native(text) => text.as_str(),
                CommandFailureMessage::Host(text) => *text,
            })
            .unwrap_or("unknown error");
        Err(MetalKernelError::CommandBufferError(msg.to_string()))
    }
}

/// The host's fence bookkeeping beside `prev_ce_outputs`.
///
/// A session's fence completes when the queue's clock marks the epoch of the
/// command buffer that carried it ([`Fence::is_completed`]); the host removes
/// their map entries at its next encoder close or queue-wide wait ([`FenceBook::reap`]),
/// under the locks those take anyway. Nothing removes an entry whose session
/// may still be running: with several threads on one queue, a wait covers only
/// what was committed when it began. So no Metal completion thread ever takes the map's lock
/// or this one, and an encoding thread can never wait on a completion thread
/// descheduled while holding them (`DESIGN.md` §6.4d). Until the close, a bind
/// that finds a completed fence treats it as absent, as it was once the
/// handler had removed it.
#[derive(Default)]
struct FenceBook {
    live: Vec<LiveFence>,
    retiring: Vec<Retiring>,
}

/// An ended session's outputs, held until its command buffer has completed.
struct Retiring {
    fence: Arc<Fence>,
    outputs: HashSet<usize>,
}

impl FenceBook {
    /// Remove from `map` the outputs of every session whose command buffer has
    /// completed and whose fence is still each output's last writer.
    fn reap(&mut self, map: &mut HashMap<usize, Arc<Fence>>) {
        let mut i = 0;
        while i < self.retiring.len() {
            if !self.retiring[i].fence.is_completed() {
                i += 1;
                continue;
            }
            let r = self.retiring.swap_remove(i);
            let ptr = Arc::as_ptr(&r.fence) as usize;
            for buf in &r.outputs {
                if map.get(buf).is_some_and(|f| Arc::ptr_eq(f, &r.fence)) {
                    map.remove(buf);
                    release_fence(&mut self.live, ptr);
                }
            }
        }
    }
}

/// A fence still referenced by `prev_ce_outputs`, with its buffer count.
struct LiveFence {
    /// Identity of the fence, used to match without comparing Arc contents.
    ptr: usize,
    fence: Arc<Fence>,
    buffers: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

// SAFETY: CommandsCore preserves Commands native recording exclusion.
unsafe impl Send for CommandsCore {}
unsafe impl Sync for CommandsCore {}

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
    ) -> Result<Self, MetalKernelError> {
        Self::new_impl(command_queue, residency_set, None)
    }
    pub(crate) fn requested_binding_metadata_bytes() -> Result<usize, MetalKernelError> {
        super::native_generation_bindings::BindingBank::requested_bytes()
    }
    /// Cold shared-base enrollment. Existing new/new_with remain explicit
    /// untracked APIs; admitted replacements NEVER fall back to those branches.
    /// # Safety
    /// Caller quotes this initial tranche ONCE from its same bootstrap metadata
    /// domain before construction; full control/native pricing remains separate.
    /// Generation native/returned/future-emission closure satisfies root contract.
    pub(crate) unsafe fn new_with_binding_metadata(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
        domain: &crate::memory_admission::ModelDomain,
        context: &crate::memory_admission::AllocationContext,
    ) -> Result<Self, MetalKernelError> {
        Self::new_impl(command_queue, residency_set, Some((domain, context)))
    }
    fn new_impl(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
        binding_metadata: Option<(
            &crate::memory_admission::ModelDomain,
            &crate::memory_admission::AllocationContext,
        )>,
    ) -> Result<Self, MetalKernelError> {
        let tuning = &crate::tuning::tuning().metal;
        let compute_per_buffer = tuning.compute_per_buffer;

        // `DESIGN.md` §13.5m. Absent -- which is every run that does not opt in
        // -- leaves `DispatchCounter`, the path that ran before this existed.
        //
        // An unparseable or zero value is a *panic* rather than a silent
        // fallback, per §2.4: "make an invalid setting panic rather than fall
        // back silently". A run that meant to select the node-index arm and
        // silently got the counter would report a clean null for the wrong
        // reason -- #69's vacuous determinism run, in this axis.
        let cut_policy = match tuning.nodes_per_cb {
            n if n > 0 => CutPolicy::NodeIndex { nodes_per_cb: n },
            _ => CutPolicy::from_env_str(&tuning.cut_policy).unwrap_or_else(|e| panic!("{e}")),
        };

        // SAFETY: queue installation borrows only. This constructor transfers
        // the SAME Arc into Core; no allAllocations extraction or out-of-band
        // residency mutation occurs here. This is not an alias terminality proof.
        if let Some(raw) = unsafe { residency_set.raw() } {
            command_queue.addResidencySet(raw);
        }

        let kind = CommandBufferKind::selected();
        let device = Device::new(command_queue.device());
        // Accounted initial route reserves/initializes header credits before
        // creating its native CB/Slot. Preserve old raw constructor behavior.
        let cb = if binding_metadata.is_some() {
            None
        } else {
            Some(create_command_buffer(&command_queue, kind)?)
        };

        // `DESIGN.md` §11.5o. Absent -- every run that does not opt in --
        // leaves `Synchronize`, which is the path that ran before this field
        // existed. An unparseable value panics inside `from_env`, for
        // `cut_policy`'s reason above.
        //
        // The event is allocated **only** for an arm that needs one, so the
        // default path allocates nothing: `newSharedEvent` on a device that has
        // never been asked for one is not free, and more importantly a resource
        // the incumbent does not create cannot change what the incumbent does.
        let completion_wait = CompletionWait::from_env();

        Self::with_parts(
            command_queue,
            device,
            cb,
            compute_per_buffer,
            cut_policy,
            completion_wait,
            residency_set,
            kind,
            binding_metadata,
        )
    }

    /// As [`Self::new`], but with the completion arm supplied rather than read
    /// from the environment (`DESIGN.md` §11.5o).
    ///
    /// **This exists because the environment arm is a `OnceLock` and a test
    /// cannot select per-test through one.** #434 records the same constraint
    /// and #536 measured what happens without this split: a flag test that
    /// exercises the *mechanism* while the *routing* is mutated dead passes,
    /// because neither test touches the code path the flag runs through. The
    /// repair there was to split the arm decision into a function the test
    /// calls and assert **both** directions, which is what this is.
    ///
    /// `new` is a caller of this, so the two cannot drift: there is one
    /// construction path and the only difference is where the arm comes from.
    pub fn new_with_completion_wait(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
        completion_wait: CompletionWait,
    ) -> Result<Self, MetalKernelError> {
        Self::new_with(
            command_queue,
            residency_set,
            completion_wait,
            CommandBufferKind::selected(),
        )
    }

    /// As [`Self::new_with_completion_wait`], with the command-buffer kind
    /// supplied too: the kind is a `OnceLock` in production, and a test or a
    /// fuzzer exercises both in one process.
    pub fn new_with(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
        completion_wait: CompletionWait,
        kind: CommandBufferKind,
    ) -> Result<Self, MetalKernelError> {
        // SAFETY: queue installation borrows only. This constructor transfers
        // the SAME Arc into Core; no allAllocations extraction or out-of-band
        // residency mutation occurs here. This is not an alias terminality proof.
        if let Some(raw) = unsafe { residency_set.raw() } {
            command_queue.addResidencySet(raw);
        }
        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue, kind)?;
        Self::with_parts(
            command_queue,
            device,
            Some(cb),
            DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
            CutPolicy::DispatchCounter,
            completion_wait,
            residency_set,
            kind,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_parts(
        command_queue: CommandQueue,
        device: Device,
        cb: Option<Retained<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>>,
        compute_per_buffer: usize,
        cut_policy: CutPolicy,
        completion_wait: CompletionWait,
        residency_set: Arc<ResidencySet>,
        kind: CommandBufferKind,
        binding_metadata: Option<(
            &crate::memory_admission::ModelDomain,
            &crate::memory_admission::AllocationContext,
        )>,
    ) -> Result<Self, MetalKernelError> {
        // The listener is allocated with the event and only for the arm that
        // needs one (#583). Its lifetime is then the engine's, which is the
        // whole discipline: a deallocated listener silently stops delivering
        // blocks, and a mechanism that stops firing reads exactly like one that
        // never engaged (§2.4).
        let completion_event = if completion_wait.needs_event() {
            SharedEvent::with_listener(&device, completion_wait.needs_listener())
        } else {
            None
        };

        let requested_api = crate::tuning::tuning().metal.command_api.as_str();
        let supported = super::arms::for_device(&device).has(super::arms::Feature::Metal4);
        let counters = super::profile::counters_enabled();
        if requested_api == "metal4" && counters {
            return Err(MetalKernelError::InvalidInput(
                "Metal 4 encoder counter sampling is not implemented; select --metal-api metal3 (or auto) to retain counter profiling".into()));
        }
        let supported = supported && !counters;
        if requested_api == "auto" && counters {
            eprintln!("[metal] command_api=metal3 auto_fallback=encoder-counter-profiling");
        }

        let modern_queue = match requested_api {
            "metal4" if !supported => {
                return Err(MetalKernelError::InvalidInput(
                    "Metal 4 is unavailable or disabled on this device; select metal3 or auto"
                        .into(),
                ))
            }
            "metal4" | "auto" if supported => Some(super::metal4::Queue::new(device.clone())?),
            _ => None,
        };
        let legacy_completions = if modern_queue.is_none() {
            Some(super::metal3_retirement::Completions::new()?)
        } else {
            None
        };
        let legacy_handler = legacy_completions
            .as_ref()
            .map(|core| core.native_handler());
        // Clock identity precedes every actual bound native constructor.
        let clock = Arc::new(GpuClock::new());
        let binding_bank = binding_metadata
            .map(|(domain, context)| unsafe {
                super::native_generation_bindings::BindingBank::cold_in(
                    domain.clone(),
                    context,
                    modern_queue
                        .as_ref()
                        .map(|queue| Arc::as_ptr(queue) as usize)
                        .unwrap_or_else(|| {
                            legacy_completions
                                .as_ref()
                                .expect("actual M3 core")
                                .identity()
                        }),
                    |context| {
                        if let Some(queue) = &modern_queue {
                            queue.cold_binding_inventory(context, clock.clone())
                        } else {
                            legacy_completions
                                .as_ref()
                                .expect("actual M3 core")
                                .cold_binding_inventory(context, clock.clone())
                        }
                    },
                )
            })
            .transpose()?;
        let mut credits: [Option<super::native_generation_bindings::PreparedGeneration>; 2] =
            std::array::from_fn(|_| None);
        let initial = if let Some(bank) = &binding_bank {
            for credit in &mut credits {
                *credit = Some(bank.prepare_before_entry()?);
            }
            Some(bank.prepare_before_entry()?)
        } else {
            None
        };
        let cb = if let Some(queue) = &modern_queue {
            eprintln!("[metal] command_api=metal4 reusable_native_buffers=true");
            CommandBuffer::from_metal4(match initial {
                Some(prepared) => queue.acquire_bound(prepared)?,
                None => queue.acquire()?,
            })
        } else {
            let raw = match cb {
                Some(raw) => raw,
                None => create_command_buffer(&command_queue, kind)?,
            };
            let core = legacy_completions
                .as_ref()
                .expect("M3 core constructed")
                .clone();
            let handler = legacy_handler
                .as_ref()
                .expect("M3 handler constructed")
                .clone();
            CommandBuffer::from_metal3_associated(match initial {
                Some(prepared) => super::metal3_retirement::AssociatedBuffer::new_bound(
                    raw, core, handler, None, prepared,
                )?,
                None => super::metal3_retirement::AssociatedBuffer::new(raw, core, handler, None)?,
            })
        };

        let pipelined_compute_per_buffer =
            crate::tuning::tuning().metal.pipelined_compute_per_buffer;
        let hold = (kind == CommandBufferKind::Unretained)
            .then(|| Arc::new(InFlightHold::new(Arc::clone(&clock))));
        let failure: Arc<CommandBufferFailure> = Arc::default();
        let hook = Arc::new(super::completion_hook::CompletionHook::new(
            Arc::clone(&clock),
            Arc::clone(&failure),
        ));
        match (&modern_queue, &legacy_completions) {
            (Some(queue), _) => queue.install_hook(Arc::clone(&hook))?,
            (None, Some(core)) => core.install_hook(Arc::clone(&hook))?,
            (None, None) => unreachable!("one completion adapter per queue"),
        }
        let encoder_finalizer = Arc::new(EncoderFinalizer {
            prev_ce_outputs: Arc::new(Censused::new(parking_lot::RwLock::new(HashMap::new()))),
            live_fences: Censused::new(parking_lot::Mutex::new(FenceBook::default())),
            failure: Arc::clone(&failure),
        });
        let retirement_wake = crate::retirement_service::HostRetirementService::shared()
            .map_err(|_| MetalKernelError::QueueUnavailable("host teardown service unavailable"))?
            .wake();
        let encoder_domain = super::encoder::EncoderDomain::new(retirement_wake.clone());
        let mut state = EntryState::new(cb);
        state.binding_credits = credits;
        let core = Arc::new(CommandsCore {
            binding_bank,
            _retirement_wake: retirement_wake,
            encoder_domain,
            state: CachePadded::new(Censused::new(parking_lot::Mutex::new(state))),
            compute_count: CachePadded::new(AtomicUsize::new(0)),
            replacement_pending: CachePadded::new(AtomicBool::new(false)),
            command_queue,
            modern_queue,
            legacy_completions,
            legacy_handler,
            compute_per_buffer: AtomicUsize::new(compute_per_buffer.max(1)),
            pipelined_compute_per_buffer: pipelined_compute_per_buffer.max(1),
            cut_policy,
            node_cut_pending: AtomicBool::new(false),
            device,
            eos_gate: Arc::new(DispatchGate::new()),
            encoder_finalizer,
            residency_set,
            clock,
            census: Arc::default(),
            trace: Arc::default(),
            hook,
            executor: parking_lot::Mutex::new(Arc::new(ExecutorSlot::Classical)),
            completion_wait,
            completion_event,
            completion_polls: AtomicU64::new(0),
            completion_signals: AtomicU64::new(0),
            completion_fires: AtomicU64::new(0),
            completion_already_complete: AtomicU64::new(0),
            kind,
            hold,
            failure,
            physical: crate::retirement_service::PhysicalAnchor::default(),
        });
        let core = lifecycle::register(core)?;
        Ok(Self { core })
    }

    /// Cold attachment before a MetalDevice or encoder can escape. The
    /// independently registered owner must not retain Commands/modern Queue.
    pub(crate) fn attach_physical_owner(
        &self,
        owner: crate::metal_backend::device_retirement::NamespaceOwner,
    ) -> Result<(), MetalKernelError> {
        let state = self.lock_state()?;
        state
            .current
            .recording()
            .ok_or(MetalKernelError::QueueUnavailable(
                "physical attachment requires pristine recording",
            ))?
            .attach_physical_owner(owner.try_clone()?)?;
        self.core.physical.install(owner).map_err(|owner| {
            crate::kernels::err::CompletionRegistrationError::new(
                MetalKernelError::QueueUnavailable("physical owner already installed"),
                Some(owner),
            )
            .into()
        })
    }

    /// The command-buffer kind this queue creates.
    pub fn kind(&self) -> CommandBufferKind {
        self.core.kind
    }

    /// The hold for objects an unretained command buffer may still name after
    /// their last host handle; `None` on the retaining kind.
    pub fn in_flight_hold(&self) -> Option<&Arc<InFlightHold>> {
        self.core.hold.as_ref()
    }

    /// The completion-wait arm in force (`DESIGN.md` §11.5o).
    ///
    /// Read back from the **resolved value**, never echoed from the string that
    /// selected it. §7.1a-i measured why: `AttnImpl` was rendered by a two-armed
    /// conditional over a three-armed axis and `--attn flash` reported
    /// `AttnImpl=Generic` on every `RESULT` line since #116 -- and a *wrong* arm
    /// compares equal to every run of that arm, where an absent one compares
    /// equal to nothing.
    pub fn completion_wait(&self) -> CompletionWait {
        self.core.completion_wait
    }

    /// Whether an event was actually allocated for the selected arm.
    ///
    /// Distinguishes *"an async arm was selected"* from *"an async arm is
    /// running"*: a device that declined to create the event leaves this false
    /// and every wait takes the blocking fallback, which is correct but is not
    /// the arm the operator asked for. Reporting it is what stops that becoming
    /// a silent null (§2.4).
    pub fn completion_event_present(&self) -> bool {
        self.core.completion_event.is_some()
    }

    /// `(signals encoded, polls performed)` since construction.
    ///
    /// The engagement pair for the async arms. `(0, 0)` under an async arm
    /// means the arm did not run -- #500's finding is that the alternative
    /// reading, a mechanism that ran and cost nothing, is indistinguishable
    /// without a counter.
    pub fn completion_counts(&self) -> (u64, u64) {
        (
            self.core.completion_signals.load(Ordering::Relaxed),
            self.core.completion_polls.load(Ordering::Relaxed),
        )
    }

    /// Notification blocks that fired since construction (#583).
    ///
    /// Separate from [`Self::completion_counts`] rather than a third element of
    /// it, because it answers a question the other two cannot: on the listener
    /// arm both of those read `(nonzero, 0)` whether the block delivered or the
    /// wait timed out and blocked instead. **A listener arm reporting
    /// `completion_fires = 0` has not taken the arm its `RESULT` line names**,
    /// which is §7.1a-i's wrong-arm shape in the one place no existing counter
    /// could see it.
    pub fn completion_fires(&self) -> u64 {
        self.core.completion_fires.load(Ordering::Relaxed)
    }

    /// Waits that found the buffer already complete, so no arm ran (#583).
    ///
    /// Read beside [`Self::completion_fires`] and [`Self::completion_counts`],
    /// which it is the denominator for: an async arm engages only on the waits
    /// that actually had to wait.
    pub fn completion_already_complete(&self) -> u64 {
        self.core
            .completion_already_complete
            .load(Ordering::Relaxed)
    }

    /// Acquisitions of this queue's submission locks that had to wait, since
    /// construction (`DESIGN.md` §6.4d). Difference two reads for a window.
    pub fn submission_contention(&self) -> SubmissionContention {
        SubmissionContention {
            map_host_read: self
                .core
                .encoder_finalizer
                .prev_ce_outputs
                .contention(Side::Host, Access::Shared),
            map_host_write: self
                .core
                .encoder_finalizer
                .prev_ce_outputs
                .contention(Side::Host, Access::Exclusive),
            live_fences_host: self
                .core
                .encoder_finalizer
                .live_fences
                .contention(Side::Host, Access::Exclusive),
            state_host: self.core.state.contention(Side::Host, Access::Exclusive),
        }
    }

    /// Hold the encoder-state lock on the calling thread while `f` runs, so a
    /// test can make an encoding thread wait on it.
    #[cfg(test)]
    pub(crate) fn while_state_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let _g = self.core.state.lock(Side::Host);
        f()
    }

    /// Hold the fence map's write lock and the fence book on the calling
    /// thread while `f` runs, so a test can show a completion does not wait on
    /// them.
    #[cfg(test)]
    pub(crate) fn while_fence_locks_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let _map = self
            .core
            .encoder_finalizer
            .prev_ce_outputs
            .write(Side::Host);
        let _book = self.core.encoder_finalizer.live_fences.lock(Side::Host);
        f()
    }

    /// Hold the completion subscribers' lock on the calling thread while `f`
    /// runs, so a test can show a commit and a completion never take it.
    #[cfg(test)]
    pub(crate) fn while_subscribers_locked<R>(&self, f: impl FnOnce() -> R) -> R {
        self.core.hook.while_subscribers_locked(f)
    }

    /// Install the executor that subsequent compute encoders will submit
    /// through (`DESIGN.md` §11.1).
    ///
    /// Takes effect on the next encoder created, not on one already open: an
    /// encoder holds its slot from creation so that the per-dispatch path never
    /// has to take this lock. Call before encoding the work being measured.
    pub fn set_executor(&self, executor: Arc<ExecutorSlot>) {
        ExecutorSlot::replace_installed(&self.core.executor, executor);
    }

    /// The executor currently installed.
    pub fn executor(&self) -> Arc<ExecutorSlot> {
        Arc::clone(&self.core.executor.lock())
    }

    /// The EOS gate every encoder of this `Commands` binds while it is armed
    /// (lloom #735 step 4).
    pub fn eos_gate(&self) -> &Arc<DispatchGate> {
        &self.core.eos_gate
    }

    /// Bind the armed gate's flag on the encoder that is already open, if
    /// any — `begin` may arm the gate mid-encoder (after a prefill).
    pub fn bind_eos_gate_current(&self) -> Result<(), MetalKernelError> {
        let state = self.lock_state()?;
        self.core.failure.check()?;
        let Some(flag) = self.core.eos_gate.armed_flag() else {
            return Ok(());
        };
        if let Some(enc) = &state.current_encoder {
            let _unwind = CloseUnwind {
                failure: &self.core.failure,
                message: "compute encoder setup exited exceptionally; custody retained",
                buffer: Some(&enc.command_buffer),
            };
            // SAFETY: the flag is the gate's own buffer, held for this queue's life;
            // the host writes it only in `EosGate::begin`, before a round's first
            // gated dispatch.
            unsafe {
                enc.set_input_buffer(
                    crate::kernels::metal::eos_gate::EOS_GATE_FLAG_INDEX,
                    Some(flag),
                    0,
                );
            }
        }
        Ok(())
    }

    /// The clock the buffer pools decide reuse against.
    pub fn clock(&self) -> Arc<GpuClock> {
        Arc::clone(&self.core.clock)
    }

    /// The dispatch census of this queue alone, for a window that must not
    /// count another device's work (`profile::DeviceCensus`).
    pub fn dispatch_census(&self) -> &Arc<crate::kernels::metal::profile::DeviceCensus> {
        &self.core.census
    }

    /// The dispatch trace of this queue alone, for a window that must not
    /// record another device's work (`trace::DeviceTrace`).
    pub fn dispatch_trace(&self) -> &Arc<crate::kernels::metal::trace::DeviceTrace> {
        &self.core.trace
    }

    /// Registers `f` to run after every command buffer that completes from now
    /// on, after the clock has marked that buffer's epoch.
    ///
    /// Call during device setup, before any work is encoded: a subscriber
    /// added later misses the buffers that completed before it, and a buffer
    /// parked against one of those epochs is swept in only at a later one.
    /// Registration publishes a new snapshot; a completion reads the current
    /// one without a lock (`completion_hook.rs`).
    ///
    /// Runs on Metal's completion thread. Per CONTRIBUTING §3.3a, a callback
    /// must not acquire a lock the encoding thread takes or submit work that
    /// waits for that thread. In particular, native-slot backpressure can hold
    /// the Commands state lock while feedback is responsible for returning a
    /// slot. Keep callbacks bounded; advance a flag and drain host-owned work
    /// on the host, as the buffer pools do.
    pub fn on_command_buffer_complete<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        self.core.hook.subscribe(Arc::new(f));
    }

    /// The cut policy in force (`DESIGN.md` §13.5m).
    ///
    /// Read by the harness to state the arm on its `RESULT` line: §7.1a-i
    /// records `config_line()` rendering a three-armed axis with a two-armed
    /// conditional, so an arm is read back from the mechanism that decides it
    /// rather than from the flag that was passed.
    pub fn cut_policy(&self) -> CutPolicy {
        self.core.cut_policy
    }

    /// The dispatch period the counter cut uses, whether or not it is in force.
    ///
    /// Stated even under `NodeIndex`, where it decides nothing: an arm at its
    /// default and an arm that is inert are different facts, which is the
    /// distinction §7.1a-i's `ScratchSizing` note draws.
    pub fn compute_per_buffer(&self) -> usize {
        self.core.compute_per_buffer.load(Ordering::Relaxed)
    }

    /// The period a pipelined decode loop raises the cut to (see the defaults
    /// above).
    pub fn pipelined_compute_per_buffer(&self) -> usize {
        self.core.pipelined_compute_per_buffer
    }

    /// Change the counter cut's period. A relaxed store, read by the next
    /// `command_encoder`; the running count since the last cut is untouched,
    /// so a switch mid-session cuts when that count reaches the new period.
    /// A pipelined decode loop raises the period for its steps through this
    /// and restores the default after (`Device::pipelined_cut_period`).
    pub fn set_compute_per_buffer(&self, period: usize) {
        self.core
            .compute_per_buffer
            .store(period.max(1), Ordering::Relaxed);
    }

    /// Tell the executor the model has reached graph node `index`
    /// (`DESIGN.md` §13.5m, issue #420 stage 3).
    ///
    /// Under [`CutPolicy::DispatchCounter`] this is **a no-op with no lock and
    /// no store**, which is what makes the default path unchanged rather than
    /// merely equivalent: an unmarked build and a marked one that never opted
    /// in run the same instructions.
    ///
    /// Under [`CutPolicy::NodeIndex`] a node whose index begins a partition run
    /// arms a cut, taken at the next encoder request. Node 0 does **not** cut:
    /// the forward pass already opens with whatever encoder the previous
    /// token's tail left, and cutting there would add a seam the partition does
    /// not name -- `n_cb` runs have `n_cb - 1` interior boundaries, and the
    /// remaining seam per token is the one the sample's own
    /// `flush_and_wait_current` already forces.
    ///
    /// **The caller owns the node numbering, not this.** `Commands` cannot know
    /// what a node is -- candle is eager and has no graph (§11.1a) -- so it
    /// takes an index and a run length and does arithmetic. That keeps the
    /// model-shaped knowledge in the model, which is §7.1c's finding about
    /// mechanisms living on `lfm2::Config`.
    /// The decision itself is [`CutPolicy::cuts_at_node`], which is what the
    /// tests exercise -- the function under test is the function called, not a
    /// restatement of it beside the one that runs (§15.2 #12b's shape).
    #[inline]
    pub fn mark_graph_node(&self, index: usize) {
        let since_cut = self.core.compute_count.load(Ordering::Relaxed);
        if self
            .core
            .cut_policy
            .cuts_at_node(index, since_cut, self.compute_per_buffer())
        {
            self.core.node_cut_pending.store(true, Ordering::Relaxed);
        }
    }

    /// Whether this encoder request should cut a new command buffer.
    ///
    /// The two policies are deliberately exclusive rather than combined: under
    /// `NodeIndex` the dispatch counter is **not consulted at all**, because a
    /// cut that fires on either condition would be a node partition *plus* a
    /// counter, and its session count would be the sum -- which is exactly the
    /// held-count property the arm exists to have (§13.5j: barriers and
    /// sessions are collinear along the counter, so an arm that moves both
    /// separates nothing).
    ///
    /// `compute_count` is still incremented on both paths, so the counter stays
    /// a faithful dispatch count for anything else reading it.
    fn should_cut(&self, count: usize) -> bool {
        self.core.cut_policy.take_cut(
            count,
            self.compute_per_buffer(),
            &self.core.node_cut_pending,
        )
    }

    /// Native feedback can arrive before commit_swap_locked releases EntryState.
    /// Refuse only actual same-queue blocking; available state remains usable.
    fn lock_state(&self) -> Result<EntryGuard<'_>, MetalKernelError> {
        self.core
            .state
            .lock_checked(Side::Host, || {
                if let Some(queue) = &self.core.modern_queue {
                    crate::completion_scope::before_park(Arc::as_ptr(queue) as usize)
                        .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
                }
                if let Some(core) = &self.core.legacy_completions {
                    crate::completion_scope::before_park(core.identity())
                        .map_err(|_| MetalKernelError::CallbackWouldBlock)?;
                }
                Ok(())
            })
            .map(|guard| DeferredGuard::new(guard, ClosedEncoders::default()))
    }

    fn lock_recording_state(&self) -> Result<EntryGuard<'_>, MetalKernelError> {
        let Some(bank) = self.core.binding_bank.as_ref() else {
            return self.lock_state();
        };
        loop {
            let state = self.lock_state()?;
            if state.binding_credits.iter().all(Option::is_some) {
                return Ok(state);
            }
            drop(state);
            self.check_failed_atomic()?;
            // Typed Full/refill BEFORE EntryState/native acquisition. Any
            // raced unused metadata credit drops only after the guard releases.
            let mut prepared = Some(bank.prepare_before_entry()?);
            let mut state = self.lock_state()?;
            if let Some(credit) = state.binding_credits.iter_mut().find(|slot| slot.is_none()) {
                *credit = prepared.take();
            }
            drop(state);
            drop(prepared);
        }
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state_guard)?;
        let count = self.core.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = self.should_cut(count);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // The per-kernel timing arm: a counter-sampled encoder per dispatch,
        // so each encoder's interval is one kernel's (`profile::encoder_per_dispatch`).
        if crate::kernels::metal::profile::encoder_per_dispatch() {
            self.core.close_compute(&mut state_guard)?;
        }

        if state_guard.current_encoder.is_none() {
            let (state_guard, custody) = state_guard.parts_mut();
            let executor = custody
                .executor
                .reserve()
                .ok_or(MetalKernelError::QueueUnavailable(
                    "encoder construction executor custody bound (1) exhausted",
                ))?;
            *executor = Some(self.executor());
            let executor = executor.as_ref().expect("executor staged above");
            if self.core.hold.is_some() && !executor.is_classical() {
                // An executor's indirect command buffers and the buffers
                // their ranges keep resident are its own, replaced as it
                // records; nothing holds them for a command buffer that does
                // not retain them.
                return Err(MetalKernelError::ArmRefused(
                    "unretained command buffers (--cb-unretained) with an executor \
                     installed: its indirect command buffers are not held until \
                     the command buffers naming them complete"
                        .to_string(),
                ));
            }
            let fence = Arc::new(Fence::held_by(
                &self.core.device,
                Some(Arc::clone(&self.core.clock)),
                self.core.hold.clone(),
            ));
            // No blanket wait here. HazardTrackingModeUntracked means Metal does
            // not flush GPU caches at encoder boundaries, so ordering is ours to
            // enforce -- but the encoder already records every buffer it binds,
            // so it waits per buffer in set_input_buffer / set_output_buffer
            // instead. Waiting on every live fence would order this encoder
            // after work it never touches.
            let mut enc = state_guard
                .current
                .recording()
                .expect("recording recovered above")
                .compute_command_encoder_with_executor(
                    &fence,
                    &self.core.encoder_finalizer.prev_ce_outputs,
                    executor,
                )?
                .with_clock(Arc::clone(&self.core.clock))
                .with_finalizer(Arc::clone(&self.core.encoder_finalizer));
            if crate::kernels::metal::profile::enabled() {
                enc = enc.with_census(Arc::clone(&self.core.census));
            }
            if crate::kernels::metal::trace::trace_requested() {
                enc = enc.with_trace(Arc::clone(&self.core.trace));
            }
            state_guard.current_encoder =
                Some(enc.into_canonical(self.core.encoder_domain.clone()));
            let enc = state_guard
                .current_encoder
                .as_ref()
                .expect("encoder staged above");
            // A custom bind may unwind after partially changing native state.
            // The canonical wrapper stays rooted, but no later recording gate
            // may treat it as a successfully initialized encoder.
            let _unwind = CloseUnwind {
                failure: &self.core.failure,
                message: "compute encoder setup exited exceptionally; custody retained",
                buffer: Some(&enc.command_buffer),
            };
            if let Some(flag) = self.core.eos_gate.armed_flag() {
                // The EOS gate (lloom #735 step 4): the gated kernel variants
                // read the flag at this index; bound once per encoder.
                // SAFETY: the flag is the gate's own buffer, held for this queue's life;
                // the host writes it only in `EosGate::begin`, before a round's first
                // gated dispatch.
                unsafe {
                    enc.set_input_buffer(
                        crate::kernels::metal::eos_gate::EOS_GATE_FLAG_INDEX,
                        Some(flag),
                        0,
                    );
                }
            }
            // Hazard state is per encoder, so a barrier count is only
            // interpretable against the session boundaries (`DESIGN.md` §9.2e).
            crate::kernels::metal::trace::note_encoder_session();
            crate::kernels::metal::trace::record_encoder_begin(enc.trace.as_deref());
            crate::kernels::metal::profile::record_encoder();
            // The audit needs the seam as a first-class ordering primitive, not
            // only as context for a count: a cross-session edge is ordered by
            // `prev_ce_outputs` + `wait_for_buffer` and by no barrier, and #144
            // found 11 real edges that survive on that alone (issue #185).
            crate::kernels::metal::hazard_audit::record_encoder_begin();
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    /// Signal `event` to `value` on the current command buffer, after every
    /// command encoded so far — the compute side of an ordering with another
    /// queue (an `MTLIOCommandQueue` load into a buffer this queue has been
    /// reading, `io_queue`). Ends the open compute encoder, which Metal
    /// requires; the next dispatch opens a new one.
    pub fn encode_signal_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        self.core.close_compute(&mut state)?;
        if self.core.modern_queue.is_some() {
            // Queue the signal as part of this commit, before acquiring a
            // replacement slot can apply backpressure.
            state
                .current
                .recording()
                .expect("recording recovered above")
                .signal_event(event, value)?;
            self.commit_swap_locked(&mut state, 0)?;
        } else {
            state
                .current
                .recording()
                .expect("recording recovered above")
                .signal_event(event, value)?;
        }
        Ok(())
    }

    /// Make every command encoded after this one on the current command
    /// buffer wait for `event` to reach `value` (the other half of
    /// [`Self::encode_signal_event`]). Ends the open compute encoder.
    pub fn encode_wait_for_event(
        &self,
        event: &ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        self.core.close_compute(&mut state)?;
        if self.core.modern_queue.is_some() {
            self.commit_swap_locked(&mut state, 0)?;
        }
        state
            .current
            .recording()
            .expect("recording recovered above")
            .wait_for_event(event, value)?;
        Ok(())
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state_guard)?;
        let count = self.core.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = self.should_cut(count);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        self.core.close_compute(&mut state_guard)?;

        let fence = Arc::new(Fence::held_by(
            &self.core.device,
            Some(Arc::clone(&self.core.clock)),
            self.core.hold.clone(),
        ));
        let encoder = state_guard
            .current
            .recording()
            .expect("recording recovered above")
            .blit_command_encoder(&fence, &self.core.encoder_finalizer.prev_ce_outputs)?;

        // Wait for all prior encoder fences before any blit commands execute.
        // Required for HazardTrackingModeUntracked: GPU caches are not auto-flushed.
        //
        // This covers every *compute* encoder that has ended, because
        // `Commands::end_encoding` adds its fence to `live_fences`. It does not
        // cover a prior *blit*: `BlitCommandEncoder::end_encoding` registers its
        // outputs in `prev_ce_outputs` but never adds its fence here. That is
        // why the per-buffer waits in `copy_from_buffer` / `fill_buffer` are not
        // redundant with this one.
        {
            let book = self.core.encoder_finalizer.live_fences.lock(Side::Host);
            for live in book.live.iter().filter(|l| !l.fence.is_completed()) {
                encoder.wait_for_fence(&live.fence);
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: Some(encoder),
            core: &self.core,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        let (to_wait, retained_current, commit_error) = {
            let mut state = self.lock_recording_state()?;
            let commit_error = if state.current.submitted().is_none()
                && self.core.compute_count.load(Ordering::Acquire) > 0
            {
                self.commit_swap_locked(&mut state, 0).err()
            } else {
                None
            };
            let retained = state.current.submitted().map(|(cb, _)| cb.clone());
            (std::mem::take(&mut state.in_flight), retained, commit_error)
        };
        // The retained current is newer than every in-flight record. Waiting
        // it needs no allocation and remains possible after replacement fails.
        if let Some(last) = retained_current.as_ref().or_else(|| to_wait.last()) {
            Self::ensure_completed(last)?;
        }
        // Check earlier CBs for errors (no need to block — they're already done).
        for cb in &to_wait {
            if cb.status() == MTLCommandBufferStatus::Error {
                self.core.failure.record(cb_error_message(cb));
            }
        }
        self.core.failure.check()?;

        // Drop only what has completed. The state lock was released before the
        // wait, so another thread sharing this queue may have ended sessions
        // since, and those may still be running: their entries are the only
        // thing ordering a later encoder's reads of what they wrote (§6.4f).
        {
            let mut prev_ce_outputs = self
                .core
                .encoder_finalizer
                .prev_ce_outputs
                .write(Side::Host);
            self.core
                .encoder_finalizer
                .live_fences
                .lock(Side::Host)
                .reap(&mut prev_ce_outputs);
        }

        if let Some(error) = commit_error {
            return Err(error);
        }
        Ok(())
    }

    /// Commit the current command buffer and wait on that specific buffer, for CPU readbacks.
    /// [`Self::wait_until_completed`] waits on the last in-flight buffer, which a concurrent
    /// `flush_and_wait` on another thread may already have taken, returning before our work ran.
    ///
    /// **This is the per-token serialization point** (`DESIGN.md` §11.5a: 441
    /// µs/token, 2.43 % of a step) and therefore the one call routed through
    /// the completion seam (§11.5o). §11.5f's census is why it is the only one:
    /// the two production `flush_and_wait_current` sites are the two arms of
    /// the single `to_cpu` the sampler reaches, and nothing else on the decode
    /// path blocks.
    pub fn flush_and_wait_current(&self) -> Result<(), MetalKernelError> {
        let (cb, signalled) = {
            let mut state = self.lock_recording_state()?;
            let signalled = self.commit_swap_locked(&mut state, 0)?;
            (state.in_flight.last().cloned(), signalled)
        };
        if let Some(cb) = cb {
            // The seam. On the default arm this is `waitUntilCompleted` and the
            // call is what it always was; on an async arm the host polls the
            // shared event and never enters the kernel to wait.
            //
            // `ensure_completed`'s other duties -- committing a buffer the swap
            // never saw, and surfacing an `Error` status -- are kept on both
            // arms, because a dropped error presents as wrong output rather than
            // as a failure (§3.5).
            let outcome = Self::ensure_completed_via(
                &cb,
                self.core.completion_wait,
                self.core.completion_event.as_ref(),
                signalled,
            )?;
            self.core
                .completion_polls
                .fetch_add(outcome.polls, Ordering::Relaxed);
            self.core
                .completion_fires
                .fetch_add(outcome.fired, Ordering::Relaxed);
            self.core
                .completion_already_complete
                .fetch_add(outcome.already_complete, Ordering::Relaxed);
            // queue is FIFO: everything committed before cb is done too
            let mut state = self.lock_state()?;
            self.prune_in_flight(&mut state, Some(&cb))?;
        }
        Ok(())
    }

    /// Commit the current command buffer and return a marker for it.
    ///
    /// Everything encoded so far is at or before the marker (the queue is
    /// FIFO), so [`Self::wait_marker`] waits for exactly that work and not for
    /// anything committed afterwards. [`Self::flush_and_wait_current`] waits on
    /// the tail of `in_flight` instead, which drains every later commit too —
    /// the reason a readback in one sequence's round stalls behind another
    /// sequence's queued work.
    pub fn flush_marker(&self) -> Result<CommandMarker, MetalKernelError> {
        let mut state = self.lock_recording_state()?;
        let signalled = self.commit_swap_locked(&mut state, 0)?;
        let cb = state.in_flight.last().cloned().ok_or_else(|| {
            MetalKernelError::CommandBufferError(
                "flush_marker: commit_swap_locked left no buffer in flight".to_string(),
            )
        })?;
        Ok(CommandMarker { cb, signalled })
    }

    /// Whether the marker's buffer has already retired, without waiting.
    /// Close and commit the open buffer **only when the queue is down to the
    /// buffer the GPU is running** — the adaptive commit (lloom #735): refill
    /// the queue the moment nothing is waiting behind the running buffer, and
    /// never cut while a committed buffer is still queued. So a buffer holds
    /// however many dispatches the host encoded while the GPU ran the previous
    /// one, and the loop commits on no cadence of its own. `Scheduled` is not
    /// the signal: an in-order queue schedules a buffer the moment it is
    /// committed; "running" is the last buffer that is not yet `Completed`.
    /// `None` when there is nothing encoded or more than one committed buffer
    /// is still incomplete; the marker otherwise.
    pub fn flush_marker_if_started(&self) -> Result<Option<CommandMarker>, MetalKernelError> {
        if self.core.compute_count.load(Ordering::Acquire) == 0
            && !self.core.replacement_pending.load(Ordering::Acquire)
        {
            return Ok(None);
        }
        let mut state = self.lock_recording_state()?;
        let recovering = state.current.submitted().is_some();
        if !recovering && self.core.compute_count.load(Ordering::Acquire) == 0 {
            return Ok(None);
        }
        let incomplete = state
            .in_flight
            .iter()
            .filter(|cb| cb.status() != MTLCommandBufferStatus::Completed)
            .count();
        if !recovering && incomplete > 1 {
            return Ok(None);
        }
        let signalled = self.commit_swap_locked(&mut state, 0)?;
        let cb = state.in_flight.last().cloned().ok_or_else(|| {
            MetalKernelError::CommandBufferError(
                "flush_marker_if_started: commit_swap_locked left no buffer in flight".to_string(),
            )
        })?;
        Ok(Some(CommandMarker { cb, signalled }))
    }

    pub fn marker_complete(&self, marker: &CommandMarker) -> bool {
        marker.cb.status() == MTLCommandBufferStatus::Completed
    }

    /// Dispatches encoded into the open buffer and not yet committed.
    pub fn open_dispatches(&self) -> usize {
        self.core.compute_count.load(Ordering::Acquire)
    }

    /// The last buffer committed and not yet pruned, as a marker; `None`
    /// when every committed buffer has completed and been pruned.
    pub fn last_committed_marker(&self) -> Result<Option<CommandMarker>, MetalKernelError> {
        let state = self.lock_state()?;
        Ok(state.in_flight.last().cloned().map(|cb| CommandMarker {
            cb,
            signalled: None,
        }))
    }

    /// Whether the marker's buffer ended in error.
    pub fn marker_failed(&self, marker: &CommandMarker) -> bool {
        marker.cb.status() == MTLCommandBufferStatus::Error
    }

    /// Wait for the marker's buffer only; commits made after it keep running.
    ///
    /// Same completion seam and counters as [`Self::flush_and_wait_current`];
    /// only the buffer waited on differs.
    pub fn wait_marker(&self, marker: &CommandMarker) -> Result<(), MetalKernelError> {
        let outcome = Self::ensure_completed_via(
            &marker.cb,
            self.core.completion_wait,
            self.core.completion_event.as_ref(),
            marker.signalled,
        )?;
        self.core
            .completion_polls
            .fetch_add(outcome.polls, Ordering::Relaxed);
        self.core
            .completion_fires
            .fetch_add(outcome.fired, Ordering::Relaxed);
        self.core
            .completion_already_complete
            .fetch_add(outcome.already_complete, Ordering::Relaxed);
        let mut state = self.lock_state()?;
        self.prune_in_flight(&mut state, Some(&marker.cb))
    }

    /// Recovery boundary for a marker created by this associated queue.
    /// Joins the registered associated prefix plus this marker's data/error
    /// even if a concurrent drainer removed predecessors from in_flight. M3 does
    /// not cover untracked raw submissions; M4 also awaits its native queue front.
    /// M4 does not
    /// join arbitrary callbacks; M3's current stronger ordered-logical fallback
    /// can wait for them, and refuses callback self-wait rather than deadlocking.
    /// No EntryState guard is held across either wait. This does not prove final
    /// callback/physical teardown or absence of errors published after return.
    pub fn wait_marker_prefix_data(&self, marker: &CommandMarker) -> Result<(), MetalKernelError> {
        let identity = self
            .core
            .modern_queue
            .as_ref()
            .map(|queue| Arc::as_ptr(queue) as usize)
            .or_else(|| {
                self.core
                    .legacy_completions
                    .as_ref()
                    .map(|core| core.identity())
            });
        if identity.is_none() || marker.cb.associated_queue_identity() != identity {
            return Err(MetalKernelError::QueueUnavailable(
                "marker belongs to another queue",
            ));
        }
        marker.cb.try_wait_prefix_data_ready()?;
        self.core.failure.check()?;
        let mut state = self.lock_state()?;
        self.prune_in_flight(&mut state, Some(&marker.cb))
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.lock_recording_state()?;
        if state.current.submitted().is_some()
            || self.core.compute_count.load(Ordering::Acquire) > 0
        {
            self.commit_swap_locked(&mut state, 0)?;
        }
        Ok(())
    }

    /// Commit the current command buffer **without waiting on it**, and drop
    /// the `in_flight` entries whose GPU work has already finished.
    ///
    /// This exists for the non-blocking token path (`DESIGN.md` §11.5g): when
    /// the host stops calling [`Self::flush_and_wait_current`] once per token,
    /// nothing else prunes `in_flight`.
    ///
    /// **The pruning is load-bearing and is not bookkeeping.** `in_flight` is a
    /// `Vec<CommandBuffer>` that shrinks on exactly two paths --
    /// `wait_until_completed`'s `mem::take` and `flush_and_wait_current`'s
    /// `retain` -- both of which block. Buffer *reclamation* does not depend on
    /// either: `commit_swap_locked` attaches an `on_completion` handler that
    /// marks the epoch and wakes the pools, and Metal delivers it
    /// asynchronously. So the pools stay correct without a wait, and it is the
    /// `Vec` itself -- one retained `CommandBuffer` per token, for the length
    /// of a generation -- that grows without one.
    ///
    /// §13.5j measures peak `pending` at "exactly zero on every arm" and
    /// attributes it to `to_scalar`'s per-token drain. Removing that drain
    /// removes the reason, which is why this is called from the poll rather
    /// than left to the next blocking readback.
    ///
    /// Returns the number of command buffers still in flight after pruning, so
    /// a caller can assert the queue is bounded rather than assume it (§2.4:
    /// an instrument that cannot be shown to have engaged has not measured
    /// anything).
    pub fn flush_and_prune(&self) -> Result<usize, MetalKernelError> {
        let timed = crate::kernels::metal::profile::enabled().then(std::time::Instant::now);
        let mut state = self.lock_recording_state()?;
        if state.current.submitted().is_some()
            || self.core.compute_count.load(Ordering::Acquire) > 0
        {
            self.commit_swap_locked(&mut state, 0)?;
        }
        // Errors are surfaced here rather than swallowed: without a blocking
        // readback on this path, this is the only place a failed command buffer
        // is observed, and a dropped error would present later as wrong output
        // rather than as a failure (§3.5 -- a missing dependency is silent
        // corruption, not an error).
        self.prune_in_flight(&mut state, None)?;
        if let Some(t) = timed {
            crate::kernels::metal::profile::record_prune(t.elapsed());
        }
        Ok(state.in_flight.len())
    }

    /// How many command buffers are in flight, **without committing, pruning or
    /// waiting**.
    ///
    /// Test-only, and the reason it exists is a defect it caught in the
    /// instrument that wanted it (#508). A fuzzer checking the boundedness
    /// invariant after every operation had no way to read the queue's length
    /// except by calling [`Self::flush_and_prune`] -- which prunes. So the
    /// observation *consumed the event it existed to observe*: the queue was
    /// already empty by the time the next generated `flush_and_prune` ran, and
    /// the engagement counter read **`pruned_something = 0` across 208 prunes**.
    /// That is `DESIGN.md` §15.1a's VACUOUS class in the observer, and #504's
    /// own `evicted = 0` in this alphabet.
    ///
    /// Deliberately **not** public: nothing in production may branch on this.
    /// The length is a racing quantity -- Metal retires buffers on a driver
    /// thread -- so a caller acting on it would be reading a state that has
    /// already moved, which is §6.7 L4's two-clocks defect. It is sound for a
    /// *bound* check, which is the only thing that reads it: the queue cannot
    /// have grown without a commit the caller made.
    #[cfg(any(test, feature = "fuzzing"))]
    pub(crate) fn in_flight_len(&self) -> Result<usize, MetalKernelError> {
        Ok(self.lock_state()?.in_flight.len())
    }

    /// Cold control/plan boundary only. Reuse returned inventory for subsequent
    /// generation preparation; do not call per CB/token. Complete pricing and
    /// default constructor enrollment remain explicit prerequisites.
    pub(crate) unsafe fn cold_binding_inventory(
        &self,
        context: &crate::memory_admission::AllocationContext,
    ) -> Result<super::native_generation_bindings::BindingInventory, MetalKernelError> {
        if let Some(queue) = &self.core.modern_queue {
            unsafe { queue.cold_binding_inventory(context, self.core.clock.clone()) }
        } else {
            let core = self
                .core
                .legacy_completions
                .as_ref()
                .expect("actual M3 associated core");
            unsafe { core.cold_binding_inventory(context, self.core.clock.clone()) }
        }
    }
    /// Cold operation capacity from its explicit model/default tranche; bank
    /// uses only shared queue identity, not first-model payer inference.
    /// # Safety
    /// Caller carries complete operation pricing and real native-use closure.
    pub(crate) unsafe fn cold_binding_capacity(
        &self,
        context: &crate::memory_admission::AllocationContext,
    ) -> Result<super::native_generation_bindings::BindingCapacity, MetalKernelError> {
        let bank = self
            .core
            .binding_bank
            .as_ref()
            .ok_or(MetalKernelError::QueueUnavailable(
                "operation capacity requires enrolled shared binding control",
            ))?;
        unsafe { bank.cold_capacity(context) }
    }
    /// Preflight outside EntryState. A racing replacement causes later exact
    /// encoder/generation refusal, never relabeling the capability as current.
    pub(crate) fn prepare_tracked_buffer(
        &self,
        buffer: &super::Buffer,
    ) -> Result<super::native_generation_bindings::PreparedBuffer, MetalKernelError> {
        let current = {
            let state = self.lock_state()?;
            state
                .current
                .recording()
                .ok_or(MetalKernelError::QueueUnavailable(
                    "native binding needs recording generation",
                ))?
                .clone()
        };
        current.binding_generation()?.prepare_buffer(buffer)
    }
    /// Recover only an already submitted receipt, preserving no-double-submit
    /// semantics. Native acquisition/prepared header credit is outside EntryState.
    /// This additive route is not enabled by legacy/default recovery factories.
    pub(crate) fn recover_bound_recording(
        &self,
        inventory: &super::native_generation_bindings::BindingInventory,
    ) -> Result<(), MetalKernelError> {
        let old = {
            let state = self.lock_state()?;
            state
                .current
                .submitted()
                .ok_or(MetalKernelError::QueueUnavailable(
                    "bound recovery needs retained submission receipt",
                ))?
                .0
                .clone()
        };
        let prepared = inventory.prepare()?;
        let candidate = if let Some(queue) = &self.core.modern_queue {
            CommandBuffer::from_metal4(queue.acquire_bound(prepared)?)
        } else {
            // Real default pin clone BEFORE new native object creation. Any
            // later error retains/returns physical + raw custody off EntryState.
            let physical = self
                .core
                .physical
                .try_owner(crate::metal_backend::device_retirement::NamespaceOwner::try_clone)?;
            let raw = create_command_buffer(&self.core.command_queue, self.core.kind)?;
            CommandBuffer::from_metal3_associated(
                super::metal3_retirement::AssociatedBuffer::new_bound(
                    raw,
                    self.core
                        .legacy_completions
                        .as_ref()
                        .expect("M3 core constructed")
                        .clone(),
                    self.core
                        .legacy_handler
                        .as_ref()
                        .expect("M3 handler constructed")
                        .clone(),
                    physical,
                    prepared,
                )?,
            )
        };
        let mut candidate = Some(candidate);
        // candidate declared BEFORE guard: every refusal releases guard before
        // cancellation/Drop of unused prepared native Submission/Slot custody.
        let mut state = self.lock_state()?;
        if !state
            .current
            .submitted()
            .is_some_and(|(current, _)| current.identity() == old.identity())
        {
            return Err(MetalKernelError::QueueUnavailable(
                "bound recovery receipt changed",
            ));
        }
        let state = &mut *state;
        state.current.recover_recording(
            || Ok::<_, MetalKernelError>(candidate.take().expect("prepared bound replacement")),
            &mut state.in_flight,
            &self.core.compute_count,
            &self.core.replacement_pending,
            0,
        )?;
        state.namespace.reset();
        Ok(())
    }

    fn create_replacement(
        &self,
        prepared: Option<super::native_generation_bindings::PreparedGeneration>,
    ) -> Result<CommandBuffer, MetalKernelError> {
        if self.core.binding_bank.is_some() && prepared.is_none() {
            return Err(MetalKernelError::QueueUnavailable(
                "bound replacement lacks preflight credit",
            ));
        }
        match &self.core.modern_queue {
            Some(queue) => Ok(CommandBuffer::from_metal4(match prepared {
                Some(prepared) => queue.acquire_bound(prepared)?,
                None => queue.acquire()?,
            })),
            None => {
                // Counted owner refusal precedes raw CB creation. Never create
                // an unregistered native resource and then drop it on EntryState
                // merely because acquiring its metadata pin failed.
                let physical = self.core.physical.try_owner(
                    crate::metal_backend::device_retirement::NamespaceOwner::try_clone,
                )?;
                let raw = create_command_buffer(&self.core.command_queue, self.core.kind)?;
                let core = self
                    .core
                    .legacy_completions
                    .as_ref()
                    .expect("M3 core constructed")
                    .clone();
                let handler = self
                    .core
                    .legacy_handler
                    .as_ref()
                    .expect("M3 handler constructed")
                    .clone();
                match prepared {
                    Some(prepared) => super::metal3_retirement::AssociatedBuffer::new_bound(
                        raw, core, handler, physical, prepared,
                    ),
                    None => super::metal3_retirement::AssociatedBuffer::new(
                        raw, core, handler, physical,
                    ),
                }
                .map(CommandBuffer::from_metal3_associated)
            }
        }
    }

    /// Recovery precedes every recording operation, not just a policy cut.
    fn ensure_recording_locked(&self, state: &mut EntryGuard<'_>) -> Result<(), MetalKernelError> {
        self.core.close_pending_blit(state)?;
        let state = &mut **state;
        let credit = state.binding_credits.iter_mut().find(|slot| slot.is_some());
        let mut absent = None;
        let credit = credit.unwrap_or(&mut absent);
        if state
            .current
            .recover_recording(
                || self.create_replacement(credit.take()),
                &mut state.in_flight,
                &self.core.compute_count,
                &self.core.replacement_pending,
                0,
            )?
            .is_some()
        {
            state.namespace.reset();
        }
        // A consumed node cut belongs to the already submitted CB; do not
        // re-arm it for the dispatch whose encoder was never returned.
        Ok(())
    }

    /// Commit the current buffer, swap in a fresh one, and return the value the
    /// completion arm should wait for -- `None` on the incumbent path.
    ///
    /// The return is what makes the async arms possible without a second commit
    /// site: the signal has to be encoded on *this* buffer, between closing its
    /// encoder and committing it, and only this function is at that point.
    fn commit_swap_locked<'a>(
        &'a self,
        state: &mut EntryGuard<'a>,
        reset_to: usize,
    ) -> Result<Option<u64>, MetalKernelError> {
        let timed = crate::kernels::metal::profile::enabled().then(std::time::Instant::now);
        let receipt = self.core.commit_current_locked(state, timed)?;
        if let Some(hold) = &self.core.hold {
            // Preserve one drain per successful receipt and its original
            // completion snapshot, including replacement error/unwind. The
            // guard unlocks before this borrowed finalizer runs.
            state.parts_mut().1.completed_hold = Some(hold.defer_release_completed());
        }
        let created = timed.map(|_| std::time::Instant::now());
        let state = &mut **state;
        let credit = state.binding_credits.iter_mut().find(|slot| slot.is_some());
        let mut absent = None;
        let credit = credit.unwrap_or(&mut absent);
        let recovered = state
            .current
            .recover_recording(
                || self.create_replacement(credit.take()),
                &mut state.in_flight,
                &self.core.compute_count,
                &self.core.replacement_pending,
                reset_to,
            )?
            .expect("submit_marked retained its receipt");
        state.namespace.reset();
        debug_assert_eq!(receipt.epoch, recovered.epoch);
        if let Some(t) = created {
            crate::kernels::metal::profile::record_create_cb(t.elapsed());
        }
        if let Some(t) = timed {
            crate::kernels::metal::profile::record_commit(t.elapsed());
        }
        Ok(receipt.signalled)
    }

    /// Wait for `cb` by the blocking incumbent.
    ///
    /// A thin wrapper over [`Self::ensure_completed_via`] with the
    /// [`CompletionWait::Synchronize`] arm, rather than a second copy of the
    /// status machine. **One implementation, not two that can drift**: the
    /// status handling below -- committing a buffer the swap never saw,
    /// surfacing an `Error` -- is identical on every arm and only the wait
    /// differs, which is the property that makes this a seam rather than a
    /// fork.
    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        let result = Self::ensure_completed_via(cb, CompletionWait::Synchronize, None, None);
        // Full drain joins callbacks, even a native terminal error. Failed
        // admission/unknown handoff is not a quiescence proof.
        if result.is_ok()
            || matches!(
                cb.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        {
            cb.try_wait_until_completed()?;
        }
        result.map(|_| ())
    }

    /// Whether a listener was allocated for the selected arm (#583).
    ///
    /// The listener arm's counterpart of [`Self::completion_event_present`],
    /// and it exists for the same reason: *"the arm was selected"* and *"the
    /// arm can run"* are different facts, and a listener that was never created
    /// makes every wait take the blocking fallback -- correct, and not the arm
    /// the operator asked for.
    pub fn completion_listener_present(&self) -> bool {
        self.core
            .completion_event
            .as_ref()
            .is_some_and(|e| e.has_listener())
    }

    /// Wait for `cb` by `arm`, returning the number of polls performed.
    ///
    /// Every arm shares the status machine and differs only at the wait, which
    /// is delegated to [`CompletionWait::wait_for`]. A **third mechanism** is
    /// therefore a variant of that enum plus its wait -- no change here, no
    /// change at the call sites, and no change to the error handling. That is
    /// the "admits a third without restructuring" property, shown rather than
    /// asserted: `SharedEventSleep` was added exactly that way and touches this
    /// function not at all.
    fn ensure_completed_via(
        cb: &CommandBuffer,
        arm: CompletionWait,
        event: Option<&SharedEvent>,
        signalled: Option<u64>,
    ) -> Result<WaitOutcome, MetalKernelError> {
        let mut outcome = WaitOutcome::default();
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                // This path commits a buffer that `commit_swap_locked` never saw,
                // so it needs its own registration or its GPU time goes unrecorded.
                //
                // **And it carries no signal**, because the signal is encoded in
                // `commit_swap_locked` and this buffer bypassed it. Passing
                // `None` here is what routes it to the blocking wait, which is
                // the fail-safe direction: a poll for a value nothing will
                // publish is a hang.
                let prepared = cb.prepare_commit()?;
                prepared.record_gpu_time_on_completion()?;
                prepared.commit()?;
                cb.try_wait_data_ready()?;
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                outcome = arm.wait_for(cb, event, signalled)?;
            }
            // Already finished, so no arm runs. **Counted, and it measures 0 on
            // the decode path** (#583) -- which is what makes the counter worth
            // keeping rather than a reason to drop it: it is the arm that
            // *falsified* the first explanation offered for the gap between
            // `completion_signals` and `completion_fires`, and a hypothesis
            // excluded by measurement is why the real one is trustworthy.
            //
            // The real explanation is that the two counters have **different
            // denominators**: a signal is encoded in every
            // `commit_swap_locked`, of which there are five call sites, and
            // only `flush_and_wait_current`'s return reaches a wait. At 675
            // dispatches/token with `CutPolicy::Counter(50)` that is ~13.5
            // commits per token against **one** wait -- so `signals` counts
            // commits and `fires` counts waits, and they are ~16:1 by
            // construction rather than by anything failing.
            MTLCommandBufferStatus::Completed => {
                outcome.already_complete = 1;
            }
            MTLCommandBufferStatus::Error => return Err(Self::cb_error(cb)),
            _ => unreachable!(),
        }

        // Event visibility and Completed status do not alone establish own
        // callback return. Readback/marker callers require data plus native
        // error visibility; full drain explicitly joins logical completion.
        cb.try_wait_data_ready()?;
        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(Self::cb_error(cb));
        }

        Ok(outcome)
    }

    fn cb_error(cb: &CommandBuffer) -> MetalKernelError {
        MetalKernelError::CommandBufferError(cb_error_message(cb))
    }

    /// Drop the in-flight buffers that completed, and report one that failed.
    ///
    /// Every wait prunes through here, so a failure in ANY buffer the host has
    /// committed surfaces at the next wait, not only a failure in the buffer
    /// that wait happened to name: the queue keeps executing after a failed
    /// buffer, so the buffer waited on can complete cleanly over results the
    /// failed one never wrote. The latch catches the paths that never wait.
    fn prune_in_flight(
        &self,
        state: &mut EntryState,
        waited: Option<&CommandBuffer>,
    ) -> Result<(), MetalKernelError> {
        // Every buffer committed before the one just waited on has run (the
        // queue executes in order), but its final status is published by its
        // own completion, which can lag the later buffer's: settle those, or a
        // failure ahead of the waited buffer reads as still running.
        let ahead = waited.and_then(|w| {
            state
                .in_flight
                .iter()
                .position(|c| c.identity() == w.identity())
        });
        for c in &state.in_flight[..ahead.unwrap_or(0)] {
            if !matches!(
                c.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            ) {
                c.try_wait_data_ready()?;
            }
        }
        let mut failed = None;
        state.in_flight.retain(|c| match c.status() {
            MTLCommandBufferStatus::Completed => false,
            MTLCommandBufferStatus::Error => {
                if failed.is_none() {
                    failed = Some(cb_error_message(c));
                }
                false
            }
            _ => true,
        });
        if let Some(msg) = failed {
            self.core.failure.record(msg);
        }
        self.core.failure.check()
    }

    /// Device setup attaches the existing memory ledger before any encoding.
    pub(crate) fn attach_command_ledger(&self, ledger: Arc<super::ledger::Ledger>) {
        if let Some(queue) = &self.core.modern_queue {
            queue.attach_ledger(ledger);
        }
    }

    /// Fixed-data failure admission for warm metadata/staging preparation.
    /// Reads only the monotone failure atom; detailed diagnostics stay in the
    /// existing public check_failed() and are not allocated on this path.
    pub(crate) fn check_failed_atomic(&self) -> Result<(), MetalKernelError> {
        if self.core.failure.failed.load(Ordering::Acquire) {
            Err(MetalKernelError::QueueUnavailable("command queue failed"))
        } else {
            Ok(())
        }
    }

    /// Cold read-only diagnostic handle; clones only the existing failure Arc.
    pub fn failure_observer(&self) -> CommandFailureObserver {
        CommandFailureObserver(Arc::clone(&self.core.failure))
    }

    /// The first command buffer of this queue that ended in error, if any.
    pub fn check_failed(&self) -> Result<(), MetalKernelError> {
        self.core.failure.check()
    }
}

/// Native close can unwind before returning a Result. Custody was already
/// staged in EntryState; latch only a static failure without allocating here.
struct CloseUnwind<'a> {
    failure: &'a CommandBufferFailure,
    message: &'static str,
    buffer: Option<&'a CommandBuffer>,
}
impl Drop for CloseUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.failure.record_static(self.message);
            if let Some(buffer) = self.buffer {
                buffer.notify_lifecycle();
            }
        }
    }
}

impl CommandsCore {
    fn close_compute(&self, state: &mut EntryGuard<'_>) -> Result<(), MetalKernelError> {
        let (state, custody) = state.parts_mut();
        if let Some(encoder) = state.current_encoder.as_ref() {
            let destination =
                custody
                    .compute
                    .reserve()
                    .ok_or(MetalKernelError::QueueUnavailable(
                        "closed compute custody bound (1) exhausted",
                    ))?;
            self.end_encoding(encoder)?;
            *destination = state.current_encoder.take();
        }
        Ok(())
    }

    fn close_pending_blit(&self, state: &mut EntryGuard<'_>) -> Result<(), MetalKernelError> {
        self.failure.check()?;
        let (state, custody) = state.parts_mut();
        if let Some(pending) = state.pending_blit.as_mut() {
            let destination = custody
                .blit
                .reserve()
                .ok_or(MetalKernelError::QueueUnavailable(
                    "closed blit custody bound (2) exhausted",
                ))?;
            let _unwind = CloseUnwind {
                failure: &self.failure,
                message: "blit close exited exceptionally; custody retained",
                buffer: None,
            };
            if let Err(error) = pending.encoder.end_encoding() {
                if !matches!(
                    error,
                    MetalKernelError::RecordingAdmission(
                        crate::command_lifecycle::LifecycleError::Busy
                    )
                ) {
                    self.failure
                        .record_static("blit close refused; custody retained");
                }
                return Err(error);
            }
            *destination = state.pending_blit.take();
        }
        Ok(())
    }

    /// Submit once and retain the original receipt on retry. Replacement is a
    /// separate operation: draining already recorded work must not require
    /// acquiring another native slot.
    fn commit_current_locked(
        &self,
        state: &mut EntryGuard<'_>,
        timed: Option<std::time::Instant>,
    ) -> Result<CommitReceipt, MetalKernelError> {
        self.close_pending_blit(state)?;
        self.close_compute(state)?;
        let t_end_enc = timed.map(|t| t.elapsed());
        let (state, custody) = state.parts_mut();
        state
            .current
            .submit_marked(&self.replacement_pending, |cb| {
                self.failure.check()?;
                self.check_recording_status(cb)?;
                // Reserve queue order and native handler allocation before
                // epoch/signal/subscriber side effects. An unused preparation
                // consumes no ordered completion sequence.
                let destination =
                    custody
                        .removals
                        .reserve()
                        .ok_or(MetalKernelError::QueueUnavailable(
                            "published residency custody bound (1) exhausted",
                        ))?;
                let prepared = cb.prepare_commit()?;
                let (_, removed) = self.residency_set.commit_pending_extracted();
                *destination = Some(removed);
                let t_residency = timed.map(|t| t.elapsed());
                self.submit_prepared(prepared, timed, t_end_enc, t_residency)
            })
            .copied()
    }

    fn check_recording_status(&self, cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        if !matches!(
            cb.status(),
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return Err(MetalKernelError::CommandBufferError(
                "recording command buffer was already submitted outside Commands".into(),
            ));
        }
        Ok(())
    }

    /// Common native handoff for ordinary cuts and final no-replacement submit.
    /// Residency was published by the caller before this point. All selectable
    /// completion signals, the completion hook and GPU profiling use this one
    /// body.
    ///
    /// The epoch is taken after every fallible registration and before the
    /// native commit, both under the commit's exclusion: a refused
    /// registration consumes no epoch, so the hook's n-th run is epoch n
    /// (`completion_hook.rs`). A bound buffer's generation stages the epoch
    /// it was given, so staging follows it. A staging or commit refused after
    /// the epoch fails the queue, and no later record runs the hook.
    fn submit_prepared(
        &self,
        prepared: super::command_buffer::PreparedCommandCommit<'_>,
        timed: Option<std::time::Instant>,
        t_end_enc: Option<std::time::Duration>,
        t_residency: Option<std::time::Duration>,
    ) -> Result<CommitReceipt, MetalKernelError> {
        let signalled = self
            .completion_wait
            .encode_prepared_signal(&prepared, self.completion_event.as_ref())?;
        if signalled.is_some() {
            self.completion_signals.fetch_add(1, Ordering::Relaxed);
        }
        crate::kernels::metal::trace::note_command_buffer();
        prepared.run_completion_hook()?;
        prepared.record_gpu_time_on_completion()?;
        let epoch = self.clock.commit_epoch();
        prepared.stage_binding_epoch(epoch)?;
        prepared.commit()?;
        if let (Some(a), Some(b), Some(c)) = (t_end_enc, t_residency, timed.map(|t| t.elapsed())) {
            crate::kernels::metal::profile::record_commit_parts(a, b - a, c - b);
        }
        Ok(CommitReceipt { epoch, signalled })
    }

    fn end_encoding(&self, encoder: &ComputeCommandEncoder) -> Result<(), MetalKernelError> {
        encoder.end_encoding()
    }
}

/// Same end operation for a Commands cut and an explicit close of its escaped
/// encoder. Shared containers have no owner backedge; allocation occurs at setup.
pub(super) struct EncoderFinalizer {
    /// Global cross-encoder output map. Maps buffer pointer to the fence of the last encoder
    /// that wrote it, enabling cross-command-buffer ordering for HazardTrackingModeUntracked.
    prev_ce_outputs: PrevCeOutputs,
    /// The distinct fences currently referenced by `prev_ce_outputs`, each with
    /// the number of buffers pointing at it, and the ended sessions whose
    /// entries leave the map once their command buffer completes.
    ///
    /// A new blit encoder waits on every distinct prior fence. Deriving that set
    /// from `prev_ce_outputs` means scanning one entry per live buffer and
    /// allocating a HashSet to deduplicate them, on every encoder. Maintaining it
    /// here makes encoder creation proportional to the number of distinct fences
    /// instead, which is typically small.
    ///
    /// Host-only, like `prev_ce_outputs`: see [`FenceBook`].
    live_fences: Censused<parking_lot::Mutex<FenceBook>>,
    failure: Arc<CommandBufferFailure>,
}
impl EncoderFinalizer {
    pub(super) fn end_encoding(
        &self,
        encoder: &ComputeCommandEncoder,
    ) -> Result<(), MetalKernelError> {
        self.end_encoding_mode(encoder, false)
    }

    pub(super) fn try_end_encoding(
        &self,
        encoder: &ComputeCommandEncoder,
    ) -> Result<bool, MetalKernelError> {
        match self.end_encoding_mode(encoder, true) {
            Ok(()) => Ok(true),
            Err(MetalKernelError::RecordingAdmission(
                crate::command_lifecycle::LifecycleError::Busy,
            )) => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn end_encoding_mode(
        &self,
        encoder: &ComputeCommandEncoder,
        nonblocking: bool,
    ) -> Result<(), MetalKernelError> {
        self.failure.check()?;
        let _unwind = CloseUnwind {
            failure: &self.failure,
            message: "compute close exited exceptionally; custody retained",
            buffer: Some(&encoder.command_buffer),
        };
        encoder.end_encoding_prepared(
            nonblocking,
            || {
                let busy = || {
                    MetalKernelError::RecordingAdmission(
                        crate::command_lifecycle::LifecycleError::Busy,
                    )
                };
                let outputs = if nonblocking {
                    self.prev_ce_outputs.try_write().ok_or_else(busy)?
                } else {
                    self.prev_ce_outputs.write(Side::Host)
                };
                let book = if nonblocking {
                    self.live_fences.try_lock().ok_or_else(busy)?
                } else {
                    self.live_fences.lock(Side::Host)
                };
                Ok((outputs, book))
            },
            |closing, s, (mut prev_ce_outputs, mut book)| {
                let all_outputs = {
                    // Once per encoder close, off the per-bind path: the session's
                    // fence-probe counts into the profiler (lloom #732).
                    if crate::kernels::metal::profile::enabled() {
                        crate::kernels::metal::profile::record_fence_stats(
                            s.fence_probes,
                            s.fence_hits,
                            s.fence_waits,
                        );
                    }
                    std::mem::take(&mut s.all_outputs)
                };

                let retires = !all_outputs.is_empty();
                {
                    // Once per encoder close: drop what completed sessions wrote, then
                    // register this session's outputs so later encoders wait for it.
                    book.reap(&mut prev_ce_outputs);
                    let this_ptr = Arc::as_ptr(&encoder.fence) as usize;
                    let mut claimed = 0usize;
                    for output in all_outputs.iter() {
                        if let Some(prev) = prev_ce_outputs.insert(*output, encoder.fence.clone()) {
                            // Another fence owned this buffer; it loses one reference.
                            release_fence(&mut book.live, Arc::as_ptr(&prev) as usize);
                        }
                        claimed += 1;
                    }
                    if claimed > 0 {
                        match book.live.iter_mut().find(|f| f.ptr == this_ptr) {
                            Some(live) => live.buffers += claimed,
                            None => book.live.push(LiveFence {
                                ptr: this_ptr,
                                fence: encoder.fence.clone(),
                                buffers: claimed,
                            }),
                        }
                    }
                    if retires {
                        book.retiring.push(Retiring {
                            fence: encoder.fence.clone(),
                            outputs: all_outputs,
                        });
                    }
                }

                drop(book);
                drop(prev_ce_outputs);
                encoder.raw.updateFence(encoder.fence.raw());

                // A clocked fence completes with its buffer's epoch; only a
                // fence with no clock needs a handler of its own.
                let clocked = encoder.fence.close_on_clock();
                let registration = if retires && !clocked {
                    let completed = encoder.fence.completion_flag();
                    closing
                        .on_completion(move || {
                            completed.store(true, Ordering::Release);
                        })
                        .map_err(MetalKernelError::from)
                } else {
                    Ok(())
                };
                // Close the native encoder even when registration refused. The error
                // carries its rejected capture out through the caller's EntryState guard;
                // it is not formatted/dropped under that guard.
                encoder.raw.endEncoding();
                if registration.is_err() {
                    // Required fence retirement is missing. Preserve the closed encoder
                    // in EntryState and the rejected callback in the returned error;
                    // refuse future recording instead of retrying native endEncoding.
                    self.failure
                        .record("encoder fence completion registration refused".into());
                }
                registration
            },
        )
    }
}

/// The error a failed command buffer reports, as text.
fn cb_error_message(cb: &CommandBuffer) -> String {
    cb.error()
        .map(|e| e.into_owned())
        .unwrap_or_else(|| "unknown error".to_string())
}

/// Drop one buffer's reference to `ptr`, forgetting the fence once none remain.
fn release_fence(fences: &mut Vec<LiveFence>, ptr: usize) {
    if let Some(idx) = fences.iter().position(|f| f.ptr == ptr) {
        fences[idx].buffers = fences[idx].buffers.saturating_sub(1);
        if fences[idx].buffers == 0 {
            fences.swap_remove(idx);
        }
    }
}

// Final Drop is TeardownRequest's metadata-only publication. The service owns
// Core before public construction returns; no synchronous wait or replacement
// allocation is attempted from an arbitrary final-owner thread.

#[cfg(test)]
mod failure_latch_tests {
    use super::*;

    fn message(latch: &CommandBufferFailure) -> String {
        match latch.check() {
            Err(MetalKernelError::CommandBufferError(message)) => message,
            other => panic!("expected latched command failure, got {other:?}"),
        }
    }

    #[test]
    fn first_error_persists_across_checks_and_later_records() {
        let latch = CommandBufferFailure::default();
        assert!(latch.check().is_ok());
        latch.record("first native failure".into());
        for _ in 0..32 {
            assert_eq!(message(&latch), "first native failure");
            latch.record("later native failure".into());
            assert_eq!(message(&latch), "first native failure");
        }
    }

    #[test]
    fn concurrent_publication_never_exposes_unknown_or_changes_the_first_message() {
        for _ in 0..16 {
            let latch = Arc::new(CommandBufferFailure::default());
            let start = Arc::new(std::sync::Barrier::new(9));
            let mut workers = Vec::new();
            for id in 0..8 {
                let latch = latch.clone();
                let start = start.clone();
                workers.push(std::thread::spawn(move || {
                    start.wait();
                    latch.record(format!("writer-{id}"));
                }));
            }
            start.wait();
            let mut first_seen = None;
            for _ in 0..64 {
                if latch.check().is_err() {
                    let observed = message(&latch);
                    assert!((0..8).any(|id| observed == format!("writer-{id}")));
                    if let Some(first) = &first_seen {
                        assert_eq!(&observed, first);
                    }
                    first_seen = Some(observed);
                }
                std::thread::yield_now();
            }
            for worker in workers {
                worker.join().unwrap();
            }
            let final_message = message(&latch);
            assert!((0..8).any(|id| final_message == format!("writer-{id}")));
            if let Some(first) = first_seen {
                assert_eq!(final_message, first);
            }
            for _ in 0..16 {
                assert_eq!(message(&latch), final_message);
            }
        }
    }
}

impl lifecycle::FinalCore for CommandsCore {
    type Captures = lifecycle::FinalCaptures;
    fn install_retirement_wake(
        &self,
        wake: Arc<crate::completion_wake::Wake>,
    ) -> Result<(), MetalKernelError> {
        self.residency_set
            .install_retirement_wake(wake)
            .map_err(|_| MetalKernelError::QueueUnavailable("foreign residency retirement Wake"))
    }
    fn record_teardown_failure(&self) {
        self.failure
            .record_static("final command teardown failed; custody retained");
    }
    /// Sole lifecycle worker after final Commands handle relinquishes. External
    /// encoder capabilities retain their domain permit until every wrapper field
    /// has dropped; no producer is revoked while it may still bind/dispatch.
    fn poll_final_submission(
        &self,
        captures: &mut lifecycle::FinalCaptures,
    ) -> Result<bool, MetalKernelError> {
        self.failure.check()?;
        if self.encoder_domain.has_external_handles() {
            return Ok(false);
        }
        let Some(mut state) = self.state.try_lock() else {
            // EntryState is privately owned, and all public guards borrow
            // Commands. No such borrow survives final Commands::drop. Treat a
            // violated ownership invariant as failure, not an unwakeable park.
            return Err(MetalKernelError::QueueUnavailable(
                "final EntryState still borrowed",
            ));
        };
        if state.current.submitted().is_some() {
            return Ok(true);
        }
        if let Some(pending) = state.pending_blit.as_mut() {
            if pending.error.is_some() {
                return Err(MetalKernelError::QueueUnavailable(
                    "retained blit close failure",
                ));
            }
            let _unwind = CloseUnwind {
                failure: &self.failure,
                message: "final blit close exited exceptionally; custody retained",
                buffer: None,
            };
            if !pending.encoder.try_end_encoding()? {
                return Ok(false);
            }
            // Keep the closed encoder rooted until all native work is terminal.
        }
        if let Some(encoder) = state.current_encoder.as_ref() {
            if !self.encoder_finalizer.try_end_encoding(encoder)? {
                return Ok(false);
            }
            // Do NOT take/drop the canonical encoder under EntryState.
        }
        let submitted = state
            .current
            .submit_marked(&self.replacement_pending, |cb| {
                self.failure.check()?;
                self.check_recording_status(cb)?;
                // Busy returns before residency/epoch/event/callback side effects.
                let prepared = cb.try_prepare_commit()?;
                if captures.removals.is_some() {
                    return Err(MetalKernelError::QueueUnavailable(
                        "final residency custody already staged",
                    ));
                }
                let (_, removals) =
                    self.residency_set
                        .try_commit_pending_extracted()
                        .map_err(|_| {
                            MetalKernelError::RecordingAdmission(
                                crate::command_lifecycle::LifecycleError::Busy,
                            )
                        })?;
                // Empty C was staged before entering this method. Even an unwind
                // after this transfer leaves the opaque raw+charge payload rooted.
                captures.removals = Some(removals);
                self.submit_prepared(prepared, None, None, None)
            });
        match submitted {
            Ok(_) => Ok(true),
            Err(MetalKernelError::RecordingAdmission(
                crate::command_lifecycle::LifecycleError::Busy,
            )) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Every retained marker, including submitted current, requires its own
    /// returned handoff and ordered callback completion. Error/poison is never
    /// a terminality certificate. The final prefix also covers older pruned CBs.
    fn poll_final_terminality(&self) -> Result<bool, MetalKernelError> {
        self.failure.check()?;
        let state = self
            .state
            .try_lock()
            .ok_or(MetalKernelError::QueueUnavailable(
                "final EntryState still borrowed",
            ))?;
        let Some((current, _)) = state.current.submitted() else {
            return Err(MetalKernelError::QueueUnavailable("final receipt absent"));
        };
        if !current.poll_logical_completed()? {
            return Ok(false);
        }
        for cb in &state.in_flight {
            if !cb.poll_logical_completed()? {
                return Ok(false);
            }
        }
        self.failure.check()?;
        Ok(true)
    }

    /// Move C directly into the independently registered active record. All
    /// locks are acquired before the first transfer. No arbitrary Drop occurs
    /// under those locks; A remains in the lifecycle inbox throughout.
    fn transfer_final_captures(
        &self,
        captures: &mut lifecycle::FinalCaptures,
        empty_executor: &Arc<ExecutorSlot>,
    ) -> Result<(), MetalKernelError> {
        let mut state = self
            .state
            .try_lock()
            .ok_or(MetalKernelError::QueueUnavailable(
                "final EntryState still borrowed",
            ))?;
        let mut executor = self
            .executor
            .try_lock()
            .ok_or(MetalKernelError::QueueUnavailable(
                "final executor still borrowed",
            ))?;
        let subscribers =
            self.hook
                .try_take_subscribers()
                .ok_or(MetalKernelError::QueueUnavailable(
                    "final completion subscribers still borrowed",
                ))?;
        if self.encoder_domain.has_external_handles()
            || state
                .current_encoder
                .as_ref()
                .is_some_and(|encoder| !encoder.retirement_closed())
            || state.pending_blit.as_ref().is_some_and(|pending| {
                pending.error.is_some() || !pending.encoder.retirement_closed()
            })
        {
            return Err(MetalKernelError::QueueUnavailable(
                "final encoder is not exclusively closed",
            ));
        }
        // Only one terminal transition calls this method; all destinations are
        // empty, except removals already retained during final submission.
        debug_assert!(
            captures.executor.is_none()
                && captures.subscribers.is_empty()
                && captures.encoder_executor.is_none()
        );
        captures.executor = Some(std::mem::replace(&mut *executor, empty_executor.clone()));
        captures.subscribers = subscribers.take();
        // Inert native wrappers stay in Core. Custom executor captures instead
        // leave through C while Core remains independently rooted.
        if let Some(encoder) = state.current_encoder.as_mut() {
            captures.encoder_executor = Some(std::mem::replace(
                &mut encoder.executor,
                empty_executor.clone(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod entry_unwind_tests {
    use super::{CloseUnwind, CommandBufferFailure, DeferredGuard};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    #[test]
    fn setup_unwind_latches_failure_and_keeps_staged_owner() {
        let failure = CommandBufferFailure::default();
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let state = parking_lot::Mutex::new(Some(owner));
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _entry = DeferredGuard::new(state.lock(), ());
            let _setup = CloseUnwind {
                failure: &failure,
                message: "compute encoder setup exited exceptionally; custody retained",
                buffer: None,
            };
            panic!("controlled setup unwind");
        }))
        .is_err());
        assert!(failure
            .check()
            .unwrap_err()
            .to_string()
            .contains("compute encoder setup exited exceptionally"));
        assert!(weak.upgrade().is_some(), "partial setup lost durable owner");
        assert!(state
            .try_lock()
            .expect("setup unwind retained EntryState lock")
            .is_some());
    }
}

#[cfg(test)]
mod entry_release_tests {
    use super::{CompletedRelease, DeferredGuard};
    use crate::retire::{Completed, Epoch, Retire};
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::{Rc, Weak};

    struct Payload<'a> {
        entry: &'a parking_lot::Mutex<()>,
        held: Weak<parking_lot::Mutex<Retire<Payload<'a>>>>,
        drops: Rc<RefCell<Vec<&'static str>>>,
        name: &'static str,
    }
    impl Drop for Payload<'_> {
        fn drop(&mut self) {
            assert!(
                self.entry.try_lock().is_some(),
                "held payload destructor ran under EntryState lock"
            );
            if let Some(held) = self.held.upgrade() {
                assert!(
                    held.try_lock().is_some(),
                    "hold payload destructor ran under hold lock"
                );
            }
            self.drops.borrow_mut().push(self.name);
        }
    }

    fn queue<'a>(
        entry: &'a parking_lot::Mutex<()>,
        drops: &Rc<RefCell<Vec<&'static str>>>,
    ) -> Rc<parking_lot::Mutex<Retire<Payload<'a>>>> {
        let held = Rc::new(parking_lot::Mutex::new(Retire::new()));
        held.lock().retire(
            Epoch(3),
            Payload {
                entry,
                held: Rc::downgrade(&held),
                drops: Rc::clone(drops),
                name: "eligible",
            },
        );
        held
    }

    #[test]
    fn completed_release_runs_after_normal_entry_unlock() {
        let entry = parking_lot::Mutex::new(());
        let drops = Rc::new(RefCell::new(Vec::new()));
        let held = queue(&entry, &drops);
        let guard = DeferredGuard::new(entry.lock(), CompletedRelease::new(&held, Completed(4)));
        assert!(drops.borrow().is_empty());
        drop(guard);
        assert_eq!(*drops.borrow(), ["eligible"]);
        assert!(held.lock().is_empty());
    }

    #[test]
    fn completed_release_survives_replacement_error() {
        let entry = parking_lot::Mutex::new(());
        let drops = Rc::new(RefCell::new(Vec::new()));
        let held = queue(&entry, &drops);
        let replacement = || -> Result<(), &'static str> {
            let _guard =
                DeferredGuard::new(entry.lock(), CompletedRelease::new(&held, Completed(4)));
            Err("replacement refused")
        };
        assert_eq!(replacement(), Err("replacement refused"));
        assert_eq!(*drops.borrow(), ["eligible"]);
        assert!(held.lock().is_empty());
    }

    #[test]
    fn completed_release_survives_replacement_unwind() {
        let entry = parking_lot::Mutex::new(());
        let drops = Rc::new(RefCell::new(Vec::new()));
        let held = queue(&entry, &drops);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _guard =
                DeferredGuard::new(entry.lock(), CompletedRelease::new(&held, Completed(4)));
            panic!("controlled replacement unwind");
        }))
        .is_err());
        assert_eq!(*drops.borrow(), ["eligible"]);
        assert!(held.lock().is_empty());
    }

    #[test]
    fn completed_release_keeps_captured_cutoff_and_unready_head() {
        let entry = parking_lot::Mutex::new(());
        let drops = Rc::new(RefCell::new(Vec::new()));
        let held = queue(&entry, &drops);
        for (epoch, name) in [(Epoch(5), "later"), (Epoch(6), "last")] {
            held.lock().retire(
                epoch,
                Payload {
                    entry: &entry,
                    held: Rc::downgrade(&held),
                    drops: Rc::clone(&drops),
                    name,
                },
            );
        }
        let clock = super::GpuClock::new();
        clock.mark_completed(Epoch(3));
        let guard = DeferredGuard::new(
            entry.lock(),
            CompletedRelease::new(&held, clock.completed_count()),
        );
        clock.mark_completed(Epoch(6));
        drop(guard);
        assert_eq!(*drops.borrow(), ["eligible"], "captured cutoff was widened");
        assert_eq!(held.lock().held(), 2);
        drop(CompletedRelease::new(&held, clock.completed_count()));
        assert_eq!(*drops.borrow(), ["eligible", "later", "last"]);
        assert!(held.lock().is_empty());
    }
}
