use crate::kernels::metal::{
    cache_pad::CachePadded,
    completion::{CompletionWait, SharedEvent, WaitOutcome},
    executor::ExecutorSlot,
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    Fence, GpuClock, PrevCeOutputs, ResidencySet,
};
use crate::kernels::MetalKernelError;
use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;

/// Where a command buffer is cut (`DESIGN.md` §13.5m, issue #420 stage 3).
///
/// # Why there are two
///
/// The shipping cut is a **global dispatch counter**: `count >=
/// compute_per_buffer`, with `count` running from process start. §13.5i
/// measured what that produces — every seam at a multiple of 50 *counted from
/// process start*, so the flash arm's land at 51/101/152 rather than
/// 50/100/150, and of a token's seams 6 fall inside attention layers and 7
/// inside conv, none at a position the graph would have chosen. The placement
/// is a **counter's residue**, not a decision.
///
/// §13.5j then measured the cost of the cut itself and found it is **~7.2–7.4
/// µs per encoder session and not a function of the barrier count** — but
/// barriers and sessions are *collinear* along `COMPUTE_PER_BUFFER` (R² 0.983
/// and 0.985), because one cut produces both. **No setting of that knob can
/// separate the two terms.**
///
/// A node-index partition is the arm that can, and that is the only reason
/// this enum exists: it moves *where* the seams land while holding *how many*
/// there are, so the session term and the barrier term stop moving together.
///
/// # What a "node" is here, and why it is the layer
///
/// llama.cpp partitions a materialised graph — `n_nodes_per_cb = (n_nodes +
/// n_cb - 1) / n_cb`, shipping `n_cb` of 1 or 2 (#419). Candle is eager and has
/// no graph object to index, so the node index has to come from the only party
/// that knows the graph position: the model. LFM2's decoder is a loop over 30
/// layers, so **node := layer** and the partition is over the layer sequence.
///
/// That is what makes the count held rather than approximately held: cutting a
/// 30-layer sequence into `n_cb` contiguous runs yields **exactly `n_cb`
/// sessions per forward pass, by construction**, independent of how many
/// dispatches a layer happens to contain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutPolicy {
    /// Cut every `compute_per_buffer` dispatches, counted from process start.
    /// The default and the shipping path; §13.5j's arm and every recorded
    /// digest belong to it.
    DispatchCounter,
    /// Cut on a layer-index partition: `n_cb` contiguous runs of the layer
    /// sequence, so every seam lands on a layer boundary and the session count
    /// is `n_cb` by construction.
    ///
    /// `nodes_per_cb` is the run length in layers, derived from `n_cb` the same
    /// way llama.cpp derives it — `ceil(n_layers / n_cb)` — and computed by the
    /// caller, since only the model knows `n_layers`.
    NodeIndex { nodes_per_cb: usize },
}

impl CutPolicy {
    /// Whether reaching graph node `index` should arm a cut.
    ///
    /// A free function of the policy and the index so the partition arithmetic
    /// is testable **without a device** — `Commands` needs a live
    /// `MTLCommandQueue` and this needs none, which is `scratch_tests.rs`'s
    /// split between the arithmetic and what executes it.
    ///
    /// **Node 0 never cuts.** `n_cb` contiguous runs have `n_cb - 1` interior
    /// boundaries, and a cut at the first node would add a seam the partition
    /// does not name — the forward pass already opens against whatever encoder
    /// the previous token's `flush_and_wait_current` left. Cutting there would
    /// make the session count `n_cb + 1` and break the held-count property the
    /// arm exists to have.
    #[inline]
    pub fn cuts_at_node(&self, index: usize) -> bool {
        match self {
            // A dispatch counter has no notion of graph position, which is
            // §13.5i's whole finding. It cannot answer this question and does
            // not try.
            CutPolicy::DispatchCounter => false,
            CutPolicy::NodeIndex { nodes_per_cb } => {
                index > 0 && index.is_multiple_of(*nodes_per_cb)
            }
        }
    }

    /// How many sessions a forward pass over `n_nodes` nodes will cut into.
    ///
    /// The held-count property, as arithmetic rather than as a claim: a
    /// partition of `n_nodes` into runs of `nodes_per_cb` has
    /// `ceil(n_nodes / nodes_per_cb)` runs, and each run is one session. Used
    /// by the tests and by anything choosing a `nodes_per_cb` to match a
    /// counter arm's count.
    pub fn sessions_for(
        &self,
        n_nodes: usize,
        compute_per_buffer: usize,
        n_dispatches: usize,
    ) -> usize {
        match self {
            // The counter's count is a function of the dispatch total, which is
            // why it cannot be held while placement moves (§13.5j).
            CutPolicy::DispatchCounter => n_dispatches.div_ceil(compute_per_buffer.max(1)),
            CutPolicy::NodeIndex { nodes_per_cb } => n_nodes.div_ceil((*nodes_per_cb).max(1)),
        }
    }
}

/// Callbacks run after each command buffer completes. See
/// [`Commands::on_command_buffer_complete`].
type CompletionSubscribers = Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>>;

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    command_queue.commandBuffer().map(CommandBuffer::new).ok_or(
        MetalKernelError::FailedToCreateResource("CommandBuffer".to_string()),
    )
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: MutexGuard<'a, EntryState>,
}

impl AsRef<ComputeCommandEncoder> for CommandsGuard<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.guard.current_encoder.as_ref().unwrap()
    }
}

impl CommandsGuard<'_> {
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
    _guard: MutexGuard<'a, EntryState>,
    state: BlitCommandEncoder,
}

impl<'a> AsRef<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_ref(&self) -> &BlitCommandEncoder {
        &self.state
    }
}

impl<'a> AsMut<BlitCommandEncoder> for BlitCommandsGuard<'a> {
    fn as_mut(&mut self) -> &mut BlitCommandEncoder {
        &mut self.state
    }
}

impl BlitCommandsGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.as_ref().set_label(label);
    }

    pub fn copy_from_buffer(
        &mut self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        self.as_mut()
            .copy_from_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size)
    }

    pub fn fill_buffer(&mut self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.as_mut().fill_buffer(buffer, range, value);
    }
}

impl Drop for BlitCommandsGuard<'_> {
    fn drop(&mut self) {
        self.as_ref().end_encoding();
    }
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
    current_encoder: Option<ComputeCommandEncoder>,
}

impl EntryState {
    pub fn new(cb: CommandBuffer) -> EntryState {
        EntryState {
            current: cb,
            in_flight: vec![],
            current_encoder: None,
        }
    }
}

pub struct Commands {
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
    state: CachePadded<Mutex<EntryState>>,
    /// Its own line -- see `state` above.
    compute_count: CachePadded<AtomicUsize>,
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
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
    /// Global cross-encoder output map. Maps buffer pointer to the fence of the last encoder
    /// that wrote it, enabling cross-command-buffer ordering for HazardTrackingModeUntracked.
    prev_ce_outputs: PrevCeOutputs,
    /// The distinct fences currently referenced by `prev_ce_outputs`, each with
    /// the number of buffers pointing at it.
    ///
    /// A new encoder waits on every distinct prior fence. Deriving that set from
    /// `prev_ce_outputs` means scanning one entry per live buffer and allocating
    /// a HashSet to deduplicate them, on every encoder. Maintaining it here makes
    /// encoder creation proportional to the number of distinct fences instead,
    /// which is typically small.
    live_fences: Arc<Mutex<Vec<LiveFence>>>,
    /// How far the GPU has got through the command buffers submitted here.
    ///
    /// Shared with the buffer pools, which use it to decide when a released
    /// buffer is safe to hand out again. This is the only place it advances.
    clock: Arc<GpuClock>,
    /// Run after each command buffer completes, once per command buffer.
    ///
    /// The buffer pools subscribe here to sweep in what that command buffer was
    /// holding up. Kept as a callback rather than a direct reference to the
    /// pools so that `Commands` does not need to know they exist -- it owns the
    /// clock, not the things that read it.
    on_complete: CompletionSubscribers,
    /// How dispatches reach the GPU (`DESIGN.md` §11.1). Handed to every
    /// compute encoder this creates.
    ///
    /// `Mutex` rather than a plain field because installing an executor is a
    /// setup-time action on a `&Commands` that is already shared; it is never
    /// touched per dispatch, because the encoder holds its own `Arc` to the
    /// slot from creation.
    executor: Mutex<Arc<ExecutorSlot>>,
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

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: &ResidencySet,
    ) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        // `DESIGN.md` §13.5m. Absent -- which is every run that does not opt in
        // -- leaves `DispatchCounter`, the path that ran before this existed.
        //
        // An unparseable or zero value is a *panic* rather than a silent
        // fallback, per §2.4: "make an invalid setting panic rather than fall
        // back silently". A run that meant to select the node-index arm and
        // silently got the counter would report a clean null for the wrong
        // reason -- #69's vacuous determinism run, in this axis.
        let cut_policy = match std::env::var("CANDLE_METAL_NODES_PER_CB") {
            Ok(val) => {
                let n: usize = val.parse().unwrap_or_else(|_| {
                    panic!("CANDLE_METAL_NODES_PER_CB must be a positive integer, got {val:?}")
                });
                assert!(
                    n > 0,
                    "CANDLE_METAL_NODES_PER_CB must be > 0, got {n} (unset it to use the counter)"
                );
                CutPolicy::NodeIndex { nodes_per_cb: n }
            }
            _ => CutPolicy::DispatchCounter,
        };

        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
        }

        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;

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
        residency_set: &ResidencySet,
        completion_wait: CompletionWait,
    ) -> Result<Self, MetalKernelError> {
        if let Some(raw) = residency_set.raw() {
            command_queue.addResidencySet(raw);
        }
        let device = Device::new(command_queue.device());
        let cb = create_command_buffer(&command_queue)?;
        Self::with_parts(
            command_queue,
            device,
            cb,
            DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
            CutPolicy::DispatchCounter,
            completion_wait,
        )
    }

    fn with_parts(
        command_queue: CommandQueue,
        device: Device,
        cb: CommandBuffer,
        compute_per_buffer: usize,
        cut_policy: CutPolicy,
        completion_wait: CompletionWait,
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

        Ok(Self {
            state: CachePadded::new(Mutex::new(EntryState::new(cb))),
            compute_count: CachePadded::new(AtomicUsize::new(0)),
            command_queue,
            compute_per_buffer,
            cut_policy,
            node_cut_pending: AtomicBool::new(false),
            device,
            prev_ce_outputs: Arc::new(Mutex::new(HashMap::new())),
            live_fences: Arc::new(Mutex::new(Vec::new())),
            clock: Arc::new(GpuClock::new()),
            on_complete: Arc::new(Mutex::new(Vec::new())),
            executor: Mutex::new(Arc::new(ExecutorSlot::Classical)),
            completion_wait,
            completion_event,
            completion_polls: AtomicU64::new(0),
            completion_signals: AtomicU64::new(0),
            completion_fires: AtomicU64::new(0),
            completion_already_complete: AtomicU64::new(0),
        })
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
        self.completion_wait
    }

    /// Whether an event was actually allocated for the selected arm.
    ///
    /// Distinguishes *"an async arm was selected"* from *"an async arm is
    /// running"*: a device that declined to create the event leaves this false
    /// and every wait takes the blocking fallback, which is correct but is not
    /// the arm the operator asked for. Reporting it is what stops that becoming
    /// a silent null (§2.4).
    pub fn completion_event_present(&self) -> bool {
        self.completion_event.is_some()
    }

    /// `(signals encoded, polls performed)` since construction.
    ///
    /// The engagement pair for the async arms. `(0, 0)` under an async arm
    /// means the arm did not run -- #500's finding is that the alternative
    /// reading, a mechanism that ran and cost nothing, is indistinguishable
    /// without a counter.
    pub fn completion_counts(&self) -> (u64, u64) {
        (
            self.completion_signals.load(Ordering::Relaxed),
            self.completion_polls.load(Ordering::Relaxed),
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
        self.completion_fires.load(Ordering::Relaxed)
    }

    /// Waits that found the buffer already complete, so no arm ran (#583).
    ///
    /// Read beside [`Self::completion_fires`] and [`Self::completion_counts`],
    /// which it is the denominator for: an async arm engages only on the waits
    /// that actually had to wait.
    pub fn completion_already_complete(&self) -> u64 {
        self.completion_already_complete.load(Ordering::Relaxed)
    }

    /// Install the executor that subsequent compute encoders will submit
    /// through (`DESIGN.md` §11.1).
    ///
    /// Takes effect on the next encoder created, not on one already open: an
    /// encoder holds its slot from creation so that the per-dispatch path never
    /// has to take this lock. Call before encoding the work being measured.
    pub fn set_executor(&self, executor: Arc<ExecutorSlot>) {
        if let Ok(mut slot) = self.executor.lock() {
            *slot = executor;
        }
    }

    /// The executor currently installed.
    pub fn executor(&self) -> Arc<ExecutorSlot> {
        self.executor
            .lock()
            .map(|e| Arc::clone(&e))
            .unwrap_or_else(|_| Arc::new(ExecutorSlot::Classical))
    }

    /// The clock the buffer pools decide reuse against.
    pub fn clock(&self) -> Arc<GpuClock> {
        Arc::clone(&self.clock)
    }

    /// Registers `f` to run after every command buffer completes.
    ///
    /// Must be called during device setup, before any work is encoded:
    /// subscribers added later would miss the command buffers already in
    /// flight, and a buffer parked against one of those epochs would never be
    /// swept in.
    pub fn on_command_buffer_complete<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        if let Ok(mut subs) = self.on_complete.lock() {
            subs.push(Arc::new(f));
        }
    }

    /// The cut policy in force (`DESIGN.md` §13.5m).
    ///
    /// Read by the harness to state the arm on its `RESULT` line: §7.1a-i
    /// records `config_line()` rendering a three-armed axis with a two-armed
    /// conditional, so an arm is read back from the mechanism that decides it
    /// rather than from the flag that was passed.
    pub fn cut_policy(&self) -> CutPolicy {
        self.cut_policy
    }

    /// The dispatch period the counter cut uses, whether or not it is in force.
    ///
    /// Stated even under `NodeIndex`, where it decides nothing: an arm at its
    /// default and an arm that is inert are different facts, which is the
    /// distinction §7.1a-i's `ScratchSizing` note draws.
    pub fn compute_per_buffer(&self) -> usize {
        self.compute_per_buffer
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
        if self.cut_policy.cuts_at_node(index) {
            self.node_cut_pending.store(true, Ordering::Relaxed);
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
        match self.cut_policy {
            CutPolicy::DispatchCounter => count >= self.compute_per_buffer,
            CutPolicy::NodeIndex { .. } => self.node_cut_pending.swap(false, Ordering::Relaxed),
        }
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = self.should_cut(count);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            // No blanket wait here. HazardTrackingModeUntracked means Metal does
            // not flush GPU caches at encoder boundaries, so ordering is ours to
            // enforce -- but the encoder already records every buffer it binds,
            // so it waits per buffer in set_input_buffer / set_output_buffer
            // instead. Waiting on every live fence would order this encoder
            // after work it never touches.
            let enc = state_guard.current.compute_command_encoder_with_executor(
                &fence,
                &self.prev_ce_outputs,
                &self.executor(),
            );
            // Hazard state is per encoder, so a barrier count is only
            // interpretable against the session boundaries (`DESIGN.md` §9.2e).
            crate::kernels::metal::trace::record_encoder_begin();
            crate::kernels::metal::profile::record_encoder();
            // The audit needs the seam as a first-class ordering primitive, not
            // only as context for a count: a cross-session edge is ordered by
            // `prev_ce_outputs` + `wait_for_buffer` and by no barrier, and #144
            // found 11 real edges that survive on that alone (issue #185).
            crate::kernels::metal::hazard_audit::record_encoder_begin();
            state_guard.current_encoder = Some(enc);
        }

        Ok(CommandsGuard { guard: state_guard })
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock().unwrap();
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = self.should_cut(count);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // End compute encoder before starting blit.
        if let Some(enc) = state_guard.current_encoder.take() {
            self.end_encoding(enc);
        }

        let fence = Arc::new(Fence::new(&self.device));
        let encoder = state_guard
            .current
            .blit_command_encoder(&fence, &self.prev_ce_outputs);

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
            let fences = self.live_fences.lock().unwrap();
            for live in fences.iter() {
                encoder.wait_for_fence(&live.fence);
            }
        }

        Ok(BlitCommandsGuard {
            _guard: state_guard,
            state: encoder,
        })
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        let to_wait = {
            let mut state = self.state.lock()?;
            if self.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&mut state, 0)?;
            }
            std::mem::take(&mut state.in_flight)
        };

        // Wait only on the last CB. Metal executes CBs in queue order, so all earlier
        // CBs are guaranteed complete when the last one is. Calling waitUntilCompleted on
        // each CB individually pays OS notification latency (~1-2ms) N times unnecessarily.
        if let Some(last) = to_wait.last() {
            Self::ensure_completed(last)?;
        }
        // Check earlier CBs for errors (no need to block — they're already done).
        for cb in &to_wait[..to_wait.len().saturating_sub(1)] {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
        }

        self.prev_ce_outputs.lock()?.clear();
        self.live_fences.lock()?.clear();

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
            let mut state = self.state.lock()?;
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
                self.completion_wait,
                self.completion_event.as_ref(),
                signalled,
            )?;
            self.completion_polls
                .fetch_add(outcome.polls, Ordering::Relaxed);
            self.completion_fires
                .fetch_add(outcome.fired, Ordering::Relaxed);
            self.completion_already_complete
                .fetch_add(outcome.already_complete, Ordering::Relaxed);
            // queue is FIFO: everything committed before cb is done too
            let mut state = self.state.lock()?;
            state
                .in_flight
                .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
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
        let mut state = self.state.lock()?;
        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }
        // Errors are surfaced here rather than swallowed: without a blocking
        // readback on this path, this is the only place a failed command buffer
        // is observed, and a dropped error would present later as wrong output
        // rather than as a failure (§3.5 -- a missing dependency is silent
        // corruption, not an error).
        for cb in state.in_flight.iter() {
            if cb.status() == MTLCommandBufferStatus::Error {
                return Err(Self::cb_error(cb));
            }
        }
        state
            .in_flight
            .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
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
    #[cfg(test)]
    pub(crate) fn in_flight_len(&self) -> Result<usize, MetalKernelError> {
        Ok(self.state.lock()?.in_flight.len())
    }

    /// Commit the current buffer, swap in a fresh one, and return the value the
    /// completion arm should wait for -- `None` on the incumbent path.
    ///
    /// The return is what makes the async arms possible without a second commit
    /// site: the signal has to be encoded on *this* buffer, between closing its
    /// encoder and committing it, and only this function is at that point.
    fn commit_swap_locked(
        &self,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<Option<u64>, MetalKernelError> {
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }

        // The signal goes here and it can go nowhere else. `encodeSignalEvent:
        // value:` is documented "may only be called if there is no current
        // command encoder on the receiver" -- so after `end_encoding` above --
        // and it must precede the `commit` below, because Metal will not accept
        // an encode onto a committed buffer.
        //
        // Probed before this was written (`measurements/probes/issue-559/
        // shared_event.m`): the signal is observed from the host with **no**
        // `waitUntilCompleted`, it does **not** advance when no signal is
        // encoded, and the kernel's writes are visible at the instant the
        // signal is. That last arm is the one this seam's soundness rests on --
        // §3.5 makes a read-before-write silent corruption rather than an error.
        //
        // `None` on the incumbent path, where nothing is encoded at all: the
        // default arm's command buffer is byte-for-byte what it was before this
        // field existed.
        let signalled = self
            .completion_wait
            .encode_signal(&state.current, self.completion_event.as_ref());
        if signalled.is_some() {
            self.completion_signals.fetch_add(1, Ordering::Relaxed);
        }

        // Close the epoch this command buffer holds and arrange for the pools to
        // hear when it finishes. Every closed epoch must be reported exactly
        // once, or a buffer parked against it waits forever -- so the handler is
        // attached on the same branch that commits, and an already-committed
        // buffer (which Metal will not accept a handler for) has its epoch
        // retired here instead.
        //
        // Both happen under the state lock, so no release can observe the new
        // epoch before the old one is accounted for.
        let epoch = self.clock.commit_epoch();
        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                let clock = Arc::clone(&self.clock);
                let subs: Vec<_> = self
                    .on_complete
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_default();
                state.current.on_completion(move || {
                    // Order matters: the clock must show the epoch finished
                    // before subscribers look, or they find nothing drainable
                    // and the buffers wait for an unrelated later completion.
                    clock.mark_completed(epoch);
                    for f in &subs {
                        f();
                    }
                });
                // Registered before commit: GPUStartTime/GPUEndTime only become
                // valid once the buffer completes, so a completion handler is
                // the only place they can be read.
                state.current.record_gpu_time_on_completion();
                state.current.commit();
            }
            // Already committed or finished, so no handler can be attached. It
            // has been waited on or will be by `flush_and_wait`; retiring the
            // epoch now keeps the clock from stalling on an epoch that will
            // never be reported.
            _ => {
                self.clock.mark_completed(epoch);
            }
        }
        let new_cb = create_command_buffer(&self.command_queue)?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);

        Ok(signalled)
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
        Self::ensure_completed_via(cb, CompletionWait::Synchronize, None, None).map(|_| ())
    }

    /// Whether a listener was allocated for the selected arm (#583).
    ///
    /// The listener arm's counterpart of [`Self::completion_event_present`],
    /// and it exists for the same reason: *"the arm was selected"* and *"the
    /// arm can run"* are different facts, and a listener that was never created
    /// makes every wait take the blocking fallback -- correct, and not the arm
    /// the operator asked for.
    pub fn completion_listener_present(&self) -> bool {
        self.completion_event
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
                cb.record_gpu_time_on_completion();
                cb.commit();
                cb.wait_until_completed();
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

        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(Self::cb_error(cb));
        }

        Ok(outcome)
    }

    fn cb_error(cb: &CommandBuffer) -> MetalKernelError {
        let msg = cb
            .error()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        MetalKernelError::CommandBufferError(msg)
    }

    fn end_encoding(&self, encoder: ComputeCommandEncoder) {
        use objc2_metal::MTLCommandEncoder as _;
        use objc2_metal::MTLComputeCommandEncoder as _;

        let all_outputs = {
            let s = encoder.state.lock().unwrap();
            s.all_outputs.clone()
        };

        {
            let mut prev_ce_outputs = self.prev_ce_outputs.lock().unwrap();
            let mut fences = self.live_fences.lock().unwrap();
            let this_ptr = Arc::as_ptr(&encoder.fence) as usize;
            let mut claimed = 0usize;

            // Register our outputs so subsequent encoders can wait for us,
            // keeping `live_fences` consistent with the map.
            for output in all_outputs.iter() {
                if let Some(prev) = prev_ce_outputs.insert(*output, encoder.fence.clone()) {
                    // Another fence owned this buffer; it loses one reference.
                    release_fence(&mut fences, Arc::as_ptr(&prev) as usize);
                }
                claimed += 1;
            }

            if claimed > 0 {
                match fences.iter_mut().find(|f| f.ptr == this_ptr) {
                    Some(live) => live.buffers += claimed,
                    None => fences.push(LiveFence {
                        ptr: this_ptr,
                        fence: encoder.fence.clone(),
                        buffers: claimed,
                    }),
                }
            }
        }

        // Signal this encoder's completion fence and end encoding.
        encoder.raw.updateFence(encoder.fence.raw());

        // Schedule cleanup of our output entries once the GPU completes.
        if !all_outputs.is_empty() {
            let fence_for_cleanup = Arc::clone(&encoder.fence);
            let map_for_cleanup = Arc::clone(&self.prev_ce_outputs);
            let fences_for_cleanup = Arc::clone(&self.live_fences);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let mut map = map_for_cleanup.lock().unwrap();
                let mut fences = fences_for_cleanup.lock().unwrap();
                let ptr = Arc::as_ptr(&fence_for_cleanup) as usize;
                for &buf in &all_outputs {
                    if let Some(f) = map.get(&buf) {
                        if Arc::ptr_eq(f, &fence_for_cleanup) {
                            map.remove(&buf);
                            release_fence(&mut fences, ptr);
                        }
                    }
                }
            });
            unsafe {
                encoder
                    .command_buffer
                    .addCompletedHandler(RcBlock::as_ptr(&block))
            };
        }

        encoder.raw.endEncoding();
    }
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

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
