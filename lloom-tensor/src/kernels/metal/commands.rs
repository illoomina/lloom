use crate::kernels::metal::{
    cache_pad::CachePadded,
    completion::{CompletionWait, SharedEvent, WaitOutcome},
    contention::{Access, Censused, Side, SubmissionContention},
    executor::ExecutorSlot,
    BlitCommandEncoder, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipeline, Device,
    DispatchGate, Fence, GpuClock, PrevCeOutputs, ResidencySet,
};
use crate::kernels::MetalKernelError;
use block2::RcBlock;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// The cut period: dispatches per command buffer. 50 is the shipping path and
/// every recorded digest belongs to it; it is also the overlap granularity of a
/// loop that WAITS on every step -- the GPU runs the first fifty dispatches
/// while the host encodes the rest -- which is why a prefill and a blocking
/// loop want it small (lloom #715: `lfm2-decode-profile` at 512 read −8.5 %,
/// 20.0 vs 18.5 ms/token; a prompt in one buffer +1.6–2.6 ms of TTFT).
const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
/// The cut period for a PIPELINED decode loop -- one that encodes step `s+1`
/// while the GPU runs step `s` (`lookahead ≥ 1`), so a cut buys no overlap and
/// costs a seam. 512 is "at most one interior cut per decode step" for LFM2's
/// ~345 dispatches, with margin; measured on the crown loop (#715,
/// `measurements/issue-715-cut-period.md`): **+2.2 % [+1.9, +2.6] tok/s at
/// 350, 512 and 1024 against 50**, streams byte-identical, the visible cadence
/// the commit budget's, prefill memory unchanged. The decode loop raises the
/// period to this for its steps (`Commands::set_compute_per_buffer`) and
/// restores the default after; the prefill runs at the default.
/// `CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER` overrides it.
const DEFAULT_CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER: usize = 512;

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
    /// Cut at the first node boundary once `compute_per_buffer` dispatches
    /// have accumulated, and unconditionally at four periods.
    ///
    /// The counter's period as a *lower bound* on a session's length and the
    /// graph's own boundaries as the places a seam may fall: the count stays
    /// the counter's to within a node, the placement is the graph's, and a
    /// stream that marks no nodes (a model that never calls
    /// `mark_graph_node`) is still cut by the hard bound rather than growing
    /// one command buffer for its whole pass. `CANDLE_METAL_CUT_POLICY=
    /// node-aligned`.
    NodeAligned,
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
    ///
    /// `since_cut` is how many dispatches the open session holds when the node
    /// is reached; only `NodeAligned` reads it, against `compute_per_buffer`.
    #[inline]
    pub fn cuts_at_node(&self, index: usize, since_cut: usize, compute_per_buffer: usize) -> bool {
        match self {
            // A dispatch counter has no notion of graph position, which is
            // §13.5i's whole finding. It cannot answer this question and does
            // not try.
            CutPolicy::DispatchCounter => false,
            CutPolicy::NodeIndex { nodes_per_cb } => {
                index > 0 && index.is_multiple_of(*nodes_per_cb)
            }
            CutPolicy::NodeAligned => index > 0 && since_cut >= compute_per_buffer,
        }
    }

    /// The dispatch count at which `NodeAligned` cuts whether or not a node
    /// boundary has arrived, as a multiple of the period.
    pub const NODE_ALIGNED_HARD_BOUND: usize = 4;

    /// Whether a session `since_cut` dispatches long must be cut at this
    /// dispatch, node or no node.
    #[inline]
    pub fn cuts_at_dispatch(&self, since_cut: usize, compute_per_buffer: usize) -> bool {
        match self {
            CutPolicy::DispatchCounter => since_cut >= compute_per_buffer,
            CutPolicy::NodeIndex { .. } => false,
            CutPolicy::NodeAligned => {
                since_cut >= compute_per_buffer.saturating_mul(Self::NODE_ALIGNED_HARD_BOUND)
            }
        }
    }

    /// The arm as a `RESULT`-line token, with the period the counter arms
    /// use: `Counter(50)`, `Node(3)`, `NodeAligned(50)`.
    pub fn render(policy: CutPolicy, compute_per_buffer: usize) -> String {
        match policy {
            CutPolicy::DispatchCounter => format!("Counter({compute_per_buffer})"),
            CutPolicy::NodeIndex { nodes_per_cb } => format!("Node({nodes_per_cb})"),
            CutPolicy::NodeAligned => format!("NodeAligned({compute_per_buffer})"),
        }
    }

    /// Parse the environment's spelling: `counter`, `node-aligned`, or
    /// `nodes:<k>`.
    pub fn from_env_str(s: &str) -> Result<CutPolicy, String> {
        match s {
            "counter" => Ok(CutPolicy::DispatchCounter),
            "node-aligned" => Ok(CutPolicy::NodeAligned),
            other => match other.strip_prefix("nodes:") {
                Some(k) => {
                    let n: usize = k
                        .parse()
                        .map_err(|_| format!("nodes:<k> needs a positive integer, got {k:?}"))?;
                    if n == 0 {
                        return Err("nodes:0 would cut at every node and then some".to_string());
                    }
                    Ok(CutPolicy::NodeIndex { nodes_per_cb: n })
                }
                None => Err(format!(
                    "CANDLE_METAL_CUT_POLICY must be `counter`, `node-aligned` or `nodes:<k>`, \
                     got {other:?}"
                )),
            },
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
            // At most the counter's count, at least one; the exact figure
            // depends on where the nodes fall against the period and is
            // read from the run rather than predicted here.
            CutPolicy::NodeAligned => n_dispatches.div_ceil(compute_per_buffer.max(1)).max(1),
        }
    }
}

/// Callbacks run after each command buffer completes. See
/// [`Commands::on_command_buffer_complete`].
type CompletionSubscribers = Arc<parking_lot::Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>>;

/// Whether the queue hands out command buffers that do NOT retain the
/// resources encoded into them (`commandBufferWithUnretainedReferences`).
///
/// Opt-in through `LLOOM_CB_UNRETAINED=1`, read once. The default retaining
/// command buffer takes a strong reference to every buffer, pipeline and
/// encoder argument it sees and releases them on completion -- objc retain
/// traffic per encode that the hot path pays for nothing, because everything a
/// decode step encodes is already held for the generation: pool buffers are
/// parked in `pending` until their epoch completes (`BufferPool::drain_completed`),
/// arena buffers live for the arena, pipelines for the process, and residency
/// is the residency set's. That is the claim the unretained form rests on, and
/// it is a claim about OUR liveness discipline rather than Metal's, which is why
/// it is fuzzed rather than assumed: `lloom-fuzz`'s `cb_liveness` harness runs
/// both kinds over a pool with a device-visible poison on free
/// (`BufferPool::set_free_poison`) and reads every copy back.
pub fn cb_unretained() -> bool {
    static KIND: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| crate::tuning::tuning().metal.cb_unretained)
}

fn create_command_buffer(command_queue: &CommandQueue) -> Result<CommandBuffer, MetalKernelError> {
    let raw = if cb_unretained() {
        command_queue.commandBufferWithUnretainedReferences()
    } else {
        command_queue.commandBuffer()
    };
    raw.map(CommandBuffer::new)
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

/// RAII guard for compute command encoder operations.
pub struct CommandsGuard<'a> {
    guard: parking_lot::MutexGuard<'a, EntryState>,
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
    _guard: parking_lot::MutexGuard<'a, EntryState>,
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
    state: CachePadded<Censused<parking_lot::Mutex<EntryState>>>,
    /// Its own line -- see `state` above.
    compute_count: CachePadded<AtomicUsize>,
    command_queue: CommandQueue,
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
    /// The queue's residency set, shared with the device that stages allocations
    /// into it. Held so a command-buffer cut can publish deferred allocations:
    /// `commit_swap_locked` calls [`ResidencySet::commit_pending`] before it
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
}

/// The host's fence bookkeeping beside `prev_ce_outputs`.
///
/// A command buffer's completion handler only marks the fences of the sessions
/// it carried completed ([`Fence::mark_completed`]); the host removes their map
/// entries at its next encoder close or queue-wide wait ([`FenceBook::reap`]),
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

impl Commands {
    pub fn new(
        command_queue: CommandQueue,
        residency_set: Arc<ResidencySet>,
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
            residency_set,
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
            residency_set,
        )
    }

    fn with_parts(
        command_queue: CommandQueue,
        device: Device,
        cb: CommandBuffer,
        compute_per_buffer: usize,
        cut_policy: CutPolicy,
        completion_wait: CompletionWait,
        residency_set: Arc<ResidencySet>,
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

        let pipelined_compute_per_buffer =
            crate::tuning::tuning().metal.pipelined_compute_per_buffer;
        Ok(Self {
            state: CachePadded::new(Censused::new(parking_lot::Mutex::new(EntryState::new(cb)))),
            compute_count: CachePadded::new(AtomicUsize::new(0)),
            command_queue,
            compute_per_buffer: AtomicUsize::new(compute_per_buffer.max(1)),
            pipelined_compute_per_buffer: pipelined_compute_per_buffer.max(1),
            cut_policy,
            node_cut_pending: AtomicBool::new(false),
            device,
            eos_gate: Arc::new(DispatchGate::new()),
            prev_ce_outputs: Arc::new(Censused::new(parking_lot::RwLock::new(HashMap::new()))),
            live_fences: Censused::new(parking_lot::Mutex::new(FenceBook::default())),
            residency_set,
            clock: Arc::new(GpuClock::new()),
            census: Arc::default(),
            trace: Arc::default(),
            on_complete: Arc::new(parking_lot::Mutex::new(Vec::new())),
            executor: parking_lot::Mutex::new(Arc::new(ExecutorSlot::Classical)),
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

    /// Acquisitions of this queue's submission locks that had to wait, since
    /// construction (`DESIGN.md` §6.4d). Difference two reads for a window.
    pub fn submission_contention(&self) -> SubmissionContention {
        SubmissionContention {
            map_host_read: self.prev_ce_outputs.contention(Side::Host, Access::Shared),
            map_host_write: self
                .prev_ce_outputs
                .contention(Side::Host, Access::Exclusive),
            live_fences_host: self.live_fences.contention(Side::Host, Access::Exclusive),
            state_host: self.state.contention(Side::Host, Access::Exclusive),
        }
    }

    /// Hold the encoder-state lock on the calling thread while `f` runs, so a
    /// test can make an encoding thread wait on it.
    #[cfg(test)]
    pub(crate) fn while_state_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let _g = self.state.lock(Side::Host);
        f()
    }

    /// Hold the fence map's write lock and the fence book on the calling
    /// thread while `f` runs, so a test can show a completion does not wait on
    /// them.
    #[cfg(test)]
    pub(crate) fn while_fence_locks_held<R>(&self, f: impl FnOnce() -> R) -> R {
        let _map = self.prev_ce_outputs.write(Side::Host);
        let _book = self.live_fences.lock(Side::Host);
        f()
    }

    /// Install the executor that subsequent compute encoders will submit
    /// through (`DESIGN.md` §11.1).
    ///
    /// Takes effect on the next encoder created, not on one already open: an
    /// encoder holds its slot from creation so that the per-dispatch path never
    /// has to take this lock. Call before encoding the work being measured.
    pub fn set_executor(&self, executor: Arc<ExecutorSlot>) {
        // `parking_lot` guards are infallible -- no `Ok` to match.
        *self.executor.lock() = executor;
    }

    /// The executor currently installed.
    pub fn executor(&self) -> Arc<ExecutorSlot> {
        Arc::clone(&self.executor.lock())
    }

    /// The EOS gate every encoder of this `Commands` binds while it is armed
    /// (lloom #735 step 4).
    pub fn eos_gate(&self) -> &Arc<DispatchGate> {
        &self.eos_gate
    }

    /// Bind the armed gate's flag on the encoder that is already open, if
    /// any — `begin` may arm the gate mid-encoder (after a prefill).
    pub fn bind_eos_gate_current(&self) {
        let Some(flag) = self.eos_gate.armed_flag() else {
            return;
        };
        let state = self.state.lock(Side::Host);
        if let Some(enc) = &state.current_encoder {
            enc.set_input_buffer(
                crate::kernels::metal::eos_gate::EOS_GATE_FLAG_INDEX,
                Some(flag),
                0,
            );
        }
    }

    /// The clock the buffer pools decide reuse against.
    pub fn clock(&self) -> Arc<GpuClock> {
        Arc::clone(&self.clock)
    }

    /// The dispatch census of this queue alone, for a window that must not
    /// count another device's work (`profile::DeviceCensus`).
    pub fn dispatch_census(&self) -> &Arc<crate::kernels::metal::profile::DeviceCensus> {
        &self.census
    }

    /// The dispatch trace of this queue alone, for a window that must not
    /// record another device's work (`trace::DeviceTrace`).
    pub fn dispatch_trace(&self) -> &Arc<crate::kernels::metal::trace::DeviceTrace> {
        &self.trace
    }

    /// Registers `f` to run after every command buffer completes.
    ///
    /// Must be called during device setup, before any work is encoded:
    /// subscribers added later would miss the command buffers already in
    /// flight, and a buffer parked against one of those epochs would never be
    /// swept in.
    pub fn on_command_buffer_complete<F: Fn() + Send + Sync + 'static>(&self, f: F) {
        self.on_complete.lock().push(Arc::new(f));
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
        self.compute_per_buffer.load(Ordering::Relaxed)
    }

    /// The period a pipelined decode loop raises the cut to (see the defaults
    /// above).
    pub fn pipelined_compute_per_buffer(&self) -> usize {
        self.pipelined_compute_per_buffer
    }

    /// Change the counter cut's period. A relaxed store, read by the next
    /// `command_encoder`; the running count since the last cut is untouched,
    /// so a switch mid-session cuts when that count reaches the new period.
    /// A pipelined decode loop raises the period for its steps through this
    /// and restores the default after (`Device::pipelined_cut_period`).
    pub fn set_compute_per_buffer(&self, period: usize) {
        self.compute_per_buffer
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
        let since_cut = self.compute_count.load(Ordering::Relaxed);
        if self
            .cut_policy
            .cuts_at_node(index, since_cut, self.compute_per_buffer())
        {
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
            CutPolicy::DispatchCounter => count >= self.compute_per_buffer(),
            CutPolicy::NodeIndex { .. } => self.node_cut_pending.swap(false, Ordering::Relaxed),
            CutPolicy::NodeAligned => {
                self.node_cut_pending.swap(false, Ordering::Relaxed)
                    || self
                        .cut_policy
                        .cuts_at_dispatch(count, self.compute_per_buffer())
            }
        }
    }

    pub fn command_encoder(&self) -> Result<CommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock(Side::Host);
        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = self.should_cut(count);

        if flush {
            self.commit_swap_locked(&mut state_guard, 1)?;
        }

        // The per-kernel timing arm: a counter-sampled encoder per dispatch,
        // so each encoder's interval is one kernel's (`profile::encoder_per_dispatch`).
        if crate::kernels::metal::profile::encoder_per_dispatch() {
            if let Some(enc) = state_guard.current_encoder.take() {
                self.end_encoding(enc);
            }
        }

        if state_guard.current_encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device));
            // No blanket wait here. HazardTrackingModeUntracked means Metal does
            // not flush GPU caches at encoder boundaries, so ordering is ours to
            // enforce -- but the encoder already records every buffer it binds,
            // so it waits per buffer in set_input_buffer / set_output_buffer
            // instead. Waiting on every live fence would order this encoder
            // after work it never touches.
            let mut enc = state_guard
                .current
                .compute_command_encoder_with_executor(
                    &fence,
                    &self.prev_ce_outputs,
                    &self.executor(),
                )
                .with_clock(Arc::clone(&self.clock));
            if crate::kernels::metal::profile::enabled() {
                enc = enc.with_census(Arc::clone(&self.census));
            }
            if crate::kernels::metal::trace::trace_requested() {
                enc = enc.with_trace(Arc::clone(&self.trace));
            }
            if let Some(flag) = self.eos_gate.armed_flag() {
                // The EOS gate (lloom #735 step 4): the gated kernel variants
                // read the flag at this index; bound once per encoder.
                enc.set_input_buffer(
                    crate::kernels::metal::eos_gate::EOS_GATE_FLAG_INDEX,
                    Some(flag),
                    0,
                );
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
            state_guard.current_encoder = Some(enc);
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
        let mut state = self.state.lock(Side::Host);
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }
        state.current.signal_event(event, value);
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
        let mut state = self.state.lock(Side::Host);
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }
        state.current.wait_for_event(event, value);
        Ok(())
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandsGuard<'_>, MetalKernelError> {
        let mut state_guard = self.state.lock(Side::Host);
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
            let book = self.live_fences.lock(Side::Host);
            for live in book.live.iter().filter(|l| !l.fence.is_completed()) {
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
            let mut state = self.state.lock(Side::Host);
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

        // Drop only what has completed. The state lock was released before the
        // wait, so another thread sharing this queue may have ended sessions
        // since, and those may still be running: their entries are the only
        // thing ordering a later encoder's reads of what they wrote (§6.4f).
        {
            let mut prev_ce_outputs = self.prev_ce_outputs.write(Side::Host);
            self.live_fences.lock(Side::Host).reap(&mut prev_ce_outputs);
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
            let mut state = self.state.lock(Side::Host);
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
            let mut state = self.state.lock(Side::Host);
            state
                .in_flight
                .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
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
        let mut state = self.state.lock(Side::Host);
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
        if self.compute_count.load(Ordering::Acquire) == 0 {
            return Ok(None);
        }
        let mut state = self.state.lock(Side::Host);
        let incomplete = state
            .in_flight
            .iter()
            .filter(|cb| cb.status() != MTLCommandBufferStatus::Completed)
            .count();
        if incomplete > 1 {
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

    /// Wait for the marker's buffer only; commits made after it keep running.
    ///
    /// Same completion seam and counters as [`Self::flush_and_wait_current`];
    /// only the buffer waited on differs.
    pub fn wait_marker(&self, marker: &CommandMarker) -> Result<(), MetalKernelError> {
        let outcome = Self::ensure_completed_via(
            &marker.cb,
            self.completion_wait,
            self.completion_event.as_ref(),
            marker.signalled,
        )?;
        self.completion_polls
            .fetch_add(outcome.polls, Ordering::Relaxed);
        self.completion_fires
            .fetch_add(outcome.fired, Ordering::Relaxed);
        self.completion_already_complete
            .fetch_add(outcome.already_complete, Ordering::Relaxed);
        let mut state = self.state.lock(Side::Host);
        state
            .in_flight
            .retain(|c| c.status() != MTLCommandBufferStatus::Completed);
        Ok(())
    }

    pub fn flush(&self) -> Result<(), MetalKernelError> {
        let mut state = self.state.lock(Side::Host);
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
        let timed = crate::kernels::metal::profile::enabled().then(std::time::Instant::now);
        let mut state = self.state.lock(Side::Host);
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
        Ok(self.state.lock(Side::Host).in_flight.len())
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
        // Priced on the encoding thread under `CANDLE_METAL_PROFILE=1`: this
        // is the host-side cost of a command-buffer seam (issue #702).
        let timed = crate::kernels::metal::profile::enabled().then(std::time::Instant::now);
        if let Some(enc) = state.current_encoder.take() {
            self.end_encoding(enc);
        }
        let t_end_enc = timed.map(|t| t.elapsed());

        // Publish any allocations a pool MISS staged with `ResidencySet::add`
        // (`DESIGN.md` §6.3b) before committing this command buffer, so every
        // allocation its dispatches may bind is resident before the GPU executes
        // them (§3.7). A load-and-branch when nothing was staged, which is every
        // commit that did not follow a miss — the whole steady-state.
        self.residency_set.commit_pending();
        let t_residency = timed.map(|t| t.elapsed());

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
        crate::kernels::metal::trace::note_command_buffer();
        let epoch = self.clock.commit_epoch();
        match state.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                let clock = Arc::clone(&self.clock);
                let subs: Vec<_> = self.on_complete.lock().clone();
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
        let t_committed = timed.map(|t| t.elapsed());
        if let (Some(a), Some(b), Some(c)) = (t_end_enc, t_residency, t_committed) {
            crate::kernels::metal::profile::record_commit_parts(a, b - a, c - b);
        }
        let created = timed.map(|_| std::time::Instant::now());
        let new_cb = create_command_buffer(&self.command_queue)?;
        if let Some(t) = created {
            crate::kernels::metal::profile::record_create_cb(t.elapsed());
        }
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);
        if let Some(t) = timed {
            crate::kernels::metal::profile::record_commit(t.elapsed());
        }

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

        // The encoder is consumed here, so its output set is moved out rather
        // than cloned: a `HashSet` cloned per encoder close is 14 clones a
        // token (`CONTRIBUTING.md` §3.3a), and the capacity dies with the
        // encoder either way.
        let all_outputs = {
            let mut s = encoder.state.lock();
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
            let mut prev_ce_outputs = self.prev_ce_outputs.write(Side::Host);
            let mut book = self.live_fences.lock(Side::Host);
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

        encoder.raw.updateFence(encoder.fence.raw());

        if retires {
            let fence = Arc::clone(&encoder.fence);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                fence.mark_completed();
            });
            // SAFETY: Metal copies the block, so the `RcBlock` may drop at the
            // end of this scope; the buffer is the uncommitted current one,
            // which is when Metal accepts a handler.
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
