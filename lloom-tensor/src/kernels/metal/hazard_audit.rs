//! Ordering verification for a recorded command stream.
//!
//! Reconstructs every dependency edge between recorded GPU commands and asks
//! whether a synchronization the stream actually encoded orders it. Under
//! untracked hazard mode a missing edge is silent corruption, and the outcome
//! gates (digests, text) cannot see it (`docs/dispatch.md` §11.3t).
//!
//! An edge is covered only through these, and happens-before is transitive
//! through any chain of them:
//!
//! | primitive | orders |
//! |---|---|
//! | pass barrier | earlier commands of its pass in the `after` stages against later commands of that pass in the `before` stages |
//! | fence update / wait | the updating pass's earlier commands in the update's stages against the waiting pass's later commands in the wait's stages, for the one fence generation the wait observes |
//! | queue barrier | every earlier pass's commands in the `after` stages against the barrier pass's later commands in the `before` stages |
//! | event signal / wait | the signal's scope against the wait's scope, only when every recorded signal able to satisfy the wait is ordered after the writer |
//! | ICB `setBarrier` | a replayed command against its predecessors in its own run |
//!
//! Distinct passes are not ordered by being distinct: a reader in one pass and
//! a writer in a later pass need a dependency the writer's pass actually waits
//! on (`docs/dependency-ordered-reuse-design.md` §2, §4).
//!
//! The verdict errs toward reporting. An unknown extent covers its allocation,
//! a wait credits only the latest recorded update of its fence, a stage a
//! synchronization does not name is not ordered by it, an update followed by a
//! wait in one pass orders nothing, and an event wait that an unordered or
//! later-recorded signal could satisfy orders nothing.
//!
//! [`SeamPolicy::AssumedOrdered`] keeps the older model, in which every
//! cross-pass edge counts as ordered. It exists to replay committed traces that
//! record no synchronization; its verdict is a statement about that model, not
//! evidence of ordering.
//!
//! Not modelled: memory visibility options, queue identity (a region is
//! assumed to be one queue), and event signals issued by the host or by a
//! queue that is not recorded.
//!
//! With the `hazard-audit` feature off every recording hook is an empty inline
//! function and no state exists.

#[cfg(feature = "hazard-audit")]
use crate::kernels::metal::encoder::HazardKind;
use crate::kernels::metal::encoder::{BoundRange, HazardKinds};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLFence;
#[cfg(feature = "hazard-audit")]
use std::collections::HashMap;

/// Pipeline stages a command runs in, or a synchronization names.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct Stages(u8);

impl Stages {
    pub const NONE: Stages = Stages(0);
    pub const DISPATCH: Stages = Stages(1);
    pub const BLIT: Stages = Stages(2);
    pub const ALL: Stages = Stages(3);

    pub const fn union(self, other: Stages) -> Stages {
        Stages(self.0 | other.0)
    }

    /// Whether every stage of `stage` is named here. An empty `stage` is
    /// never covered.
    pub const fn covers(self, stage: Stages) -> bool {
        stage.0 != 0 && self.0 & stage.0 == stage.0
    }

    #[cfg(feature = "hazard-audit")]
    fn slots(self) -> impl Iterator<Item = usize> {
        [Stages::DISPATCH, Stages::BLIT]
            .into_iter()
            .enumerate()
            .filter(move |(_, s)| self.0 & s.0 != 0)
            .map(|(i, _)| i)
    }
}

/// The two sides of a pass barrier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StageScope {
    pub after: Stages,
    pub before: Stages,
}

impl StageScope {
    pub const ALL: StageScope = StageScope {
        after: Stages::ALL,
        before: Stages::ALL,
    };

    #[cfg(feature = "hazard-audit")]
    fn union(self, other: StageScope) -> StageScope {
        StageScope {
            after: self.after.union(other.after),
            before: self.before.union(other.before),
        }
    }
}

/// The commands an event signal waits for, or an event wait blocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EventScope {
    /// The commands of one command buffer, by identity.
    CommandBuffer(usize),
    /// Every command submitted to the queue.
    Queue,
}

/// One binding: the `(buffer, offset, len, direction)` an interval test needs.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AuditBinding {
    pub ptr: usize,
    pub offset: usize,
    pub len: usize,
    pub is_output: bool,
}

#[cfg(feature = "hazard-audit")]
impl AuditBinding {
    /// Whether the two bindings can touch the same byte. A zero length covers
    /// the whole allocation, as in `BoundRange`.
    fn overlaps(&self, other: &AuditBinding) -> bool {
        if self.ptr != other.ptr {
            return false;
        }
        if self.len == 0 || other.len == 0 {
            return true;
        }
        self.offset < other.offset + other.len && other.offset < self.offset + self.len
    }

    /// The hazard from `earlier` to this binding. Read-after-read is none.
    fn hazard_against(&self, earlier: &AuditBinding) -> Option<HazardKind> {
        if !self.overlaps(earlier) {
            return None;
        }
        match (earlier.is_output, self.is_output) {
            (true, false) => Some(HazardKind::Raw),
            (true, true) => Some(HazardKind::Waw),
            (false, true) => Some(HazardKind::War),
            (false, false) => None,
        }
    }
}

/// One recorded command.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug)]
pub struct AuditDispatch {
    pub seq: u64,
    pub kernel: String,
    pub bindings: Vec<AuditBinding>,
    /// Whether a pass barrier was encoded immediately before this command.
    pub barrier: bool,
    /// The stages that barrier orders; read only when `barrier` is set.
    pub barrier_scope: StageScope,
    /// The hazard directions the barrier was emitted for.
    pub kinds: HazardKinds,
    /// Whether a barrier was owed here and deferred rather than emitted.
    pub barrier_suppressed: bool,
    /// The pass this command belongs to.
    pub encoder: u64,
    /// The stage this command runs in.
    pub stage: Stages,
    /// The ICB run this position replays in, when it is replayed at all.
    pub run: Option<u64>,
    /// Whether this position heads its run; a head orders nothing in its run.
    pub is_run_head: bool,
    /// Whether the ICB encoded a `setBarrier` onto this position's command.
    pub icb_barrier: bool,
}

#[cfg(feature = "hazard-audit")]
impl AuditDispatch {
    /// A command in `encoder` and `stage` with no binding, barrier or replay.
    pub fn new(seq: u64, encoder: u64, stage: Stages, kernel: impl Into<String>) -> Self {
        AuditDispatch {
            seq,
            kernel: kernel.into(),
            bindings: Vec::new(),
            barrier: false,
            barrier_scope: StageScope::ALL,
            kinds: HazardKinds::NONE,
            barrier_suppressed: false,
            encoder,
            stage,
            run: None,
            is_run_head: false,
            icb_barrier: false,
        }
    }
}

/// A recorded synchronization, placed in the command stream.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug)]
pub struct AuditSync {
    /// The number of commands recorded before this; it sits between command
    /// `before_dispatch - 1` and command `before_dispatch`.
    pub before_dispatch: usize,
    pub kind: SyncKind,
}

#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncKind {
    /// A pass opened, on the named command buffer when it is known.
    Open {
        encoder: u64,
        command_buffer: Option<usize>,
    },
    /// The `generation`-th update of `fence`, covering the pass's earlier
    /// commands in the `after` stages.
    FenceUpdate {
        encoder: u64,
        fence: usize,
        generation: u64,
        after: Stages,
    },
    /// A wait on `fence` as of its `generation`-th update (0: never updated),
    /// blocking the pass's later commands in the `before` stages.
    FenceWait {
        encoder: u64,
        fence: usize,
        generation: u64,
        before: Stages,
    },
    /// A consumer queue barrier encoded in `encoder`.
    QueueBarrier {
        encoder: u64,
        after: Stages,
        before: Stages,
    },
    EventSignal {
        scope: EventScope,
        event: usize,
        value: u64,
    },
    /// Satisfied by any signal of `event` with a value of at least `value`.
    EventWait {
        scope: EventScope,
        event: usize,
        value: u64,
    },
}

/// A recorded stream: commands and the synchronizations between them.
///
/// The builder methods number fence generations, so a stream built through
/// them cannot pair a wait with an update it could not have observed.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug, Default)]
pub struct AuditStream {
    pub dispatches: Vec<AuditDispatch>,
    pub syncs: Vec<AuditSync>,
    /// Where a range starts holding a new logical value (its storage reused),
    /// as `(commands recorded before it, range)`, in recording order.
    pub values: Vec<(usize, AuditBinding)>,
    fence_generations: HashMap<usize, u64>,
}

/// Bytes of a new value read before any write of that value: the successor
/// did not initialize what its consumer reads (`dependency-ordered-reuse-
/// design.md` §3, obligation 3).
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UninitializedRead {
    pub reader_seq: u64,
    pub reader_kernel: String,
    pub ptr: usize,
    /// The first byte of the read no earlier write of the value covers, and
    /// the end of that gap.
    pub from: usize,
    pub to: usize,
}

#[cfg(feature = "hazard-audit")]
impl AuditStream {
    /// Append a command; its `seq` is left as given.
    pub fn push_dispatch(&mut self, dispatch: AuditDispatch) {
        self.dispatches.push(dispatch);
    }

    fn push_sync(&mut self, kind: SyncKind) {
        self.syncs.push(AuditSync {
            before_dispatch: self.dispatches.len(),
            kind,
        });
    }

    pub fn open_pass(&mut self, encoder: u64, command_buffer: Option<usize>) {
        self.push_sync(SyncKind::Open {
            encoder,
            command_buffer,
        });
    }

    pub fn fence_update(&mut self, encoder: u64, fence: usize, after: Stages) {
        let generation = self.fence_generations.entry(fence).or_insert(0);
        *generation += 1;
        let generation = *generation;
        self.push_sync(SyncKind::FenceUpdate {
            encoder,
            fence,
            generation,
            after,
        });
    }

    pub fn fence_wait(&mut self, encoder: u64, fence: usize, before: Stages) {
        let generation = self.fence_generations.get(&fence).copied().unwrap_or(0);
        self.push_sync(SyncKind::FenceWait {
            encoder,
            fence,
            generation,
            before,
        });
    }

    pub fn queue_barrier(&mut self, encoder: u64, after: Stages, before: Stages) {
        self.push_sync(SyncKind::QueueBarrier {
            encoder,
            after,
            before,
        });
    }

    pub fn event_signal(&mut self, scope: EventScope, event: usize, value: u64) {
        self.push_sync(SyncKind::EventSignal {
            scope,
            event,
            value,
        });
    }

    pub fn event_wait(&mut self, scope: EventScope, event: usize, value: u64) {
        self.push_sync(SyncKind::EventWait {
            scope,
            event,
            value,
        });
    }

    /// `ptr`'s bytes `offset..offset + len` (a zero `len`: the allocation
    /// from `offset`) start holding a new value here: its storage is reused.
    pub fn new_value(&mut self, ptr: usize, offset: usize, len: usize) {
        self.values.push((
            self.dispatches.len(),
            AuditBinding {
                ptr,
                offset,
                len,
                is_output: true,
            },
        ));
    }

    /// Every read of a new value's bytes that no earlier write of that value
    /// covers, up to the next new value of the same storage. A write in the
    /// reading command itself does not count: it is not ordered before the
    /// read. Ordering of the writes that do count is [`Self::cover`]'s
    /// question, not this one's.
    pub fn uninitialized_reads(&self) -> Vec<UninitializedRead> {
        let span = |b: &AuditBinding| {
            let end = if b.len == 0 {
                usize::MAX
            } else {
                b.offset.saturating_add(b.len)
            };
            (b.offset, end)
        };
        let mut found = Vec::new();
        for (vi, (start, value)) in self.values.iter().enumerate() {
            let end = self.values[vi + 1..]
                .iter()
                .find(|(_, later)| later.overlaps(value))
                .map_or(self.dispatches.len(), |(at, _)| *at);
            let (v0, v1) = span(value);
            for r in *start..end {
                let d = &self.dispatches[r];
                for b in d
                    .bindings
                    .iter()
                    .filter(|b| !b.is_output && b.overlaps(value))
                {
                    let (b0, b1) = span(b);
                    let (lo, hi) = (b0.max(v0), b1.min(v1));
                    let mut written: Vec<(usize, usize)> = self.dispatches[*start..r]
                        .iter()
                        .flat_map(|w| w.bindings.iter())
                        .filter(|w| w.is_output && w.ptr == value.ptr)
                        .map(span)
                        .collect();
                    written.sort_unstable();
                    let mut at = lo;
                    for (w0, w1) in written {
                        if w0 > at {
                            break;
                        }
                        at = at.max(w1);
                        if at >= hi {
                            break;
                        }
                    }
                    if at < hi {
                        found.push(UninitializedRead {
                            reader_seq: d.seq,
                            reader_kernel: d.kernel.clone(),
                            ptr: value.ptr,
                            from: at,
                            to: hi,
                        });
                    }
                }
            }
        }
        found
    }

    /// Cover the stream under `policy`.
    pub fn cover(&self, region: &str, policy: SeamPolicy) -> AuditReport {
        cover_stream(region, &self.dispatches, &self.syncs, policy)
    }
}

/// How an edge between two passes with no recorded dependency is judged.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SeamPolicy {
    /// Only recorded synchronization orders anything.
    #[default]
    Observed,
    /// Every cross-pass edge counts as ordered. For replaying traces that
    /// record no synchronization; not evidence of ordering.
    AssumedOrdered,
}

/// What ordered an edge.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cover {
    /// A pass barrier in the pass both ends belong to.
    Barrier,
    /// A fence the reader's pass waits on, updated after the writer.
    Fence,
    /// A queue barrier in the reader's pass.
    QueueBarrier,
    /// An event wait every satisfying signal of which follows the writer.
    Event,
    /// The ICB's `setBarrier` on the reader's command, within one run.
    Icb,
    /// Ordered, but by no single primitive above (a round trip through
    /// another pass back into the writer's own pass).
    Transitive,
    /// Assumed ordered because the ends are in different passes; only under
    /// [`SeamPolicy::AssumedOrdered`].
    Seam,
}

/// A dependency edge and what, if anything, orders it.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug)]
pub struct Edge {
    pub writer_seq: u64,
    pub reader_seq: u64,
    pub writer_kernel: String,
    pub reader_kernel: String,
    pub writer_encoder: u64,
    pub reader_encoder: u64,
    pub kind: HazardKind,
    pub ptr: usize,
    pub offset: usize,
    pub len: usize,
    /// `None` is the finding: an edge nothing orders.
    pub cover: Option<Cover>,
}

/// What one audited region concluded.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug, Default)]
pub struct AuditReport {
    pub region: String,
    pub positions: usize,
    pub edges: usize,
    pub edges_raw: usize,
    pub edges_waw: usize,
    pub edges_war: usize,
    pub by_barrier: usize,
    pub by_fence: usize,
    pub by_queue_barrier: usize,
    pub by_event: usize,
    pub by_icb: usize,
    pub by_transitive: usize,
    pub by_seam: usize,
    /// The edges nothing orders. Empty is the passing verdict.
    pub uncovered: Vec<Edge>,
    pub barriers: usize,
    pub barriers_raw: usize,
    pub barriers_waw: usize,
    pub barriers_war: usize,
    pub barriers_suppressed: usize,
    pub passes: usize,
    pub fence_updates: usize,
    pub fence_waits: usize,
    /// Waits on a fence with no recorded update in the region; they order
    /// nothing the audit can see.
    pub unmatched_fence_waits: usize,
    /// Waits whose observed update is in the waiting pass itself.
    pub same_pass_fence_waits: usize,
    pub queue_barriers: usize,
    pub event_signals: usize,
    pub event_waits: usize,
    /// Event waits credited with nothing: no recorded signal satisfies them,
    /// or one recorded after the wait could.
    pub unordered_event_waits: usize,
}

#[cfg(feature = "hazard-audit")]
impl AuditReport {
    /// Whether every dependency edge is ordered by something.
    pub fn is_clean(&self) -> bool {
        self.uncovered.is_empty()
    }

    /// A human-readable summary.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut o = String::new();
        let _ = writeln!(
            o,
            "region={} positions={} passes={} edges={} (raw={} waw={} war={})",
            self.region,
            self.positions,
            self.passes,
            self.edges,
            self.edges_raw,
            self.edges_waw,
            self.edges_war
        );
        let _ = writeln!(
            o,
            "  barriers={} (raw={} waw={} war={}) suppressed={}",
            self.barriers,
            self.barriers_raw,
            self.barriers_waw,
            self.barriers_war,
            self.barriers_suppressed
        );
        let _ = writeln!(
            o,
            "  sync: fence_updates={} fence_waits={} unmatched_waits={} same_pass_waits={} \
             queue_barriers={} event_signals={} event_waits={} unordered_event_waits={}",
            self.fence_updates,
            self.fence_waits,
            self.unmatched_fence_waits,
            self.same_pass_fence_waits,
            self.queue_barriers,
            self.event_signals,
            self.event_waits,
            self.unordered_event_waits
        );
        let _ = writeln!(
            o,
            "  cover: barrier={} fence={} queue_barrier={} event={} icb={} transitive={} \
             seam(assumed)={}  UNCOVERED={}",
            self.by_barrier,
            self.by_fence,
            self.by_queue_barrier,
            self.by_event,
            self.by_icb,
            self.by_transitive,
            self.by_seam,
            self.uncovered.len()
        );
        for e in self.uncovered.iter().take(40) {
            let _ = writeln!(
                o,
                "    !! #{} {} (pass {}) -> #{} {} (pass {})  {} buf={:#x} off={} len={}",
                e.writer_seq,
                e.writer_kernel,
                e.writer_encoder,
                e.reader_seq,
                e.reader_kernel,
                e.reader_encoder,
                e.kind.as_str(),
                e.ptr,
                e.offset,
                e.len
            );
        }
        if self.uncovered.len() > 40 {
            let _ = writeln!(o, "    ... and {} more", self.uncovered.len() - 40);
        }
        o
    }
}

/// The cover test over commands alone: only pass barriers and ICB runs can
/// order anything, so every cross-pass edge is uncovered.
#[cfg(feature = "hazard-audit")]
pub fn cover(region: &str, dispatches: &[AuditDispatch]) -> AuditReport {
    cover_stream(region, dispatches, &[], SeamPolicy::Observed)
}

/// The cover test: every dependency edge, and what orders it.
///
/// `syncs` must be in recording order; entries with equal `before_dispatch`
/// keep their relative order.
#[cfg(feature = "hazard-audit")]
pub fn cover_stream(
    region: &str,
    dispatches: &[AuditDispatch],
    syncs: &[AuditSync],
    policy: SeamPolicy,
) -> AuditReport {
    let mut rep = AuditReport {
        region: region.to_string(),
        positions: dispatches.len(),
        ..Default::default()
    };
    for d in dispatches {
        if d.barrier {
            rep.barriers += 1;
            if d.kinds.contains(HazardKind::Raw) {
                rep.barriers_raw += 1;
            }
            if d.kinds.contains(HazardKind::Waw) {
                rep.barriers_waw += 1;
            }
            if d.kinds.contains(HazardKind::War) {
                rep.barriers_war += 1;
            }
        }
        if d.barrier_suppressed {
            rep.barriers_suppressed += 1;
        }
    }

    let order = Order::build(dispatches, syncs, &mut rep);

    for (ri, r) in dispatches.iter().enumerate() {
        for (wi, w) in dispatches.iter().enumerate().take(ri) {
            let Some(kind) = hazard_between(w, r) else {
                continue;
            };
            let (ptr, offset, len) = witness(w, r);
            rep.edges += 1;
            match kind {
                HazardKind::Raw => rep.edges_raw += 1,
                HazardKind::Waw => rep.edges_waw += 1,
                HazardKind::War => rep.edges_war += 1,
            }
            let cover = order
                .cover(wi, ri, dispatches, syncs)
                .or_else(|| icb_orders(w, r).then_some(Cover::Icb))
                .or_else(|| {
                    (policy == SeamPolicy::AssumedOrdered && w.encoder != r.encoder)
                        .then_some(Cover::Seam)
                });
            match cover {
                Some(Cover::Barrier) => rep.by_barrier += 1,
                Some(Cover::Fence) => rep.by_fence += 1,
                Some(Cover::QueueBarrier) => rep.by_queue_barrier += 1,
                Some(Cover::Event) => rep.by_event += 1,
                Some(Cover::Icb) => rep.by_icb += 1,
                Some(Cover::Transitive) => rep.by_transitive += 1,
                Some(Cover::Seam) => rep.by_seam += 1,
                None => rep.uncovered.push(Edge {
                    writer_seq: w.seq,
                    reader_seq: r.seq,
                    writer_kernel: w.kernel.clone(),
                    reader_kernel: r.kernel.clone(),
                    writer_encoder: w.encoder,
                    reader_encoder: r.encoder,
                    kind,
                    ptr,
                    offset,
                    len,
                    cover: None,
                }),
            }
        }
    }
    rep
}

/// The ICB's `setBarrier` orders a replayed command against its predecessors
/// in its own run; a head's scan slice is empty, so a head orders nothing.
#[cfg(feature = "hazard-audit")]
fn icb_orders(w: &AuditDispatch, r: &AuditDispatch) -> bool {
    w.run.is_some() && w.run == r.run && r.icb_barrier && !r.is_run_head
}

/// The first hazard found between two commands; any one requires ordering.
#[cfg(feature = "hazard-audit")]
fn hazard_between(earlier: &AuditDispatch, later: &AuditDispatch) -> Option<HazardKind> {
    for lb in &later.bindings {
        for eb in &earlier.bindings {
            if let Some(k) = lb.hazard_against(eb) {
                return Some(k);
            }
        }
    }
    None
}

/// The bytes an edge is about, for the report.
#[cfg(feature = "hazard-audit")]
fn witness(earlier: &AuditDispatch, later: &AuditDispatch) -> (usize, usize, usize) {
    for lb in &later.bindings {
        for eb in &earlier.bindings {
            if lb.hazard_against(eb).is_some() {
                return (lb.ptr, lb.offset, lb.len);
            }
        }
    }
    (0, 0, 0)
}

/// A fixed-capacity set of command positions.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug)]
struct PosSet(Vec<u64>);

#[cfg(feature = "hazard-audit")]
impl PosSet {
    fn new(n: usize) -> Self {
        PosSet(vec![0; n.div_ceil(64)])
    }
    fn insert(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
    fn contains(&self, i: usize) -> bool {
        self.0[i / 64] & (1 << (i % 64)) != 0
    }
    fn union_with(&mut self, other: &PosSet) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= *b;
        }
    }
    fn intersect_with(&mut self, other: &PosSet) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a &= *b;
        }
    }
}

/// Per-pass happens-before state, indexed by stage slot.
#[cfg(feature = "hazard-audit")]
struct Pass {
    rank: usize,
    command_buffer: Option<usize>,
    /// Every command of the pass so far, with what precedes each, by stage.
    closure: [PosSet; 2],
    /// What every later command of the pass in that stage is ordered after.
    floor: [PosSet; 2],
    /// Pass barriers so far: (command position, scope).
    barriers: Vec<(usize, StageScope)>,
    /// Syncs that order later commands of this pass.
    inbound: Vec<usize>,
}

/// Happens-before over a recorded stream: for each command, the set of
/// commands that complete before it starts.
#[cfg(feature = "hazard-audit")]
struct Order {
    ancestors: Vec<PosSet>,
    sync_ancestors: Vec<Option<PosSet>>,
    passes: HashMap<u64, Pass>,
    cb_inbound: HashMap<usize, Vec<usize>>,
    queue_inbound: Vec<usize>,
}

#[cfg(feature = "hazard-audit")]
impl Order {
    fn build(dispatches: &[AuditDispatch], syncs: &[AuditSync], rep: &mut AuditReport) -> Order {
        let n = dispatches.len();
        let mut sync_order: Vec<usize> = (0..syncs.len()).collect();
        sync_order.sort_by_key(|&i| syncs[i].before_dispatch);
        let mut host_rank = vec![0usize; syncs.len()];
        for (rank, &i) in sync_order.iter().enumerate() {
            host_rank[i] = rank;
        }

        let mut o = Order {
            ancestors: Vec::with_capacity(n),
            sync_ancestors: vec![None; syncs.len()],
            passes: HashMap::new(),
            cb_inbound: HashMap::new(),
            queue_inbound: Vec::new(),
        };
        let mut cb_closure: HashMap<usize, PosSet> = HashMap::new();
        let mut cb_floor: HashMap<usize, PosSet> = HashMap::new();
        let mut queue_closure = PosSet::new(n);
        let mut queue_floor = PosSet::new(n);
        let mut updates: HashMap<(usize, u64), (u64, PosSet)> = HashMap::new();
        let mut signals: HashMap<usize, Vec<(u64, PosSet)>> = HashMap::new();

        let mut next = 0;
        for k in 0..=n {
            while next < sync_order.len() && syncs[sync_order[next]].before_dispatch <= k {
                let si = sync_order[next];
                next += 1;
                match &syncs[si].kind {
                    SyncKind::Open {
                        encoder,
                        command_buffer,
                    } => {
                        let p = o.pass(*encoder, n);
                        if command_buffer.is_some() {
                            p.command_buffer = *command_buffer;
                        }
                    }
                    SyncKind::FenceUpdate {
                        encoder,
                        fence,
                        generation,
                        after,
                    } => {
                        rep.fence_updates += 1;
                        let p = o.pass(*encoder, n);
                        let mut a = PosSet::new(n);
                        for s in after.slots() {
                            a.union_with(&p.closure[s]);
                        }
                        updates.insert((*fence, *generation), (*encoder, a.clone()));
                        o.sync_ancestors[si] = Some(a);
                    }
                    SyncKind::FenceWait {
                        encoder,
                        fence,
                        generation,
                        before,
                    } => {
                        rep.fence_waits += 1;
                        let a = match updates.get(&(*fence, *generation)) {
                            None => {
                                rep.unmatched_fence_waits += 1;
                                None
                            }
                            Some((updater, _)) if updater == encoder => {
                                rep.same_pass_fence_waits += 1;
                                None
                            }
                            Some((_, a)) => Some(a.clone()),
                        };
                        if let Some(a) = a {
                            let p = o.pass(*encoder, n);
                            for s in before.slots() {
                                p.floor[s].union_with(&a);
                            }
                            p.inbound.push(si);
                            o.sync_ancestors[si] = Some(a);
                        }
                    }
                    SyncKind::QueueBarrier {
                        encoder,
                        after,
                        before,
                    } => {
                        rep.queue_barriers += 1;
                        let rank = o.pass(*encoder, n).rank;
                        let mut a = PosSet::new(n);
                        for q in o.passes.values().filter(|q| q.rank < rank) {
                            for s in after.slots() {
                                a.union_with(&q.closure[s]);
                            }
                        }
                        let p = o.pass(*encoder, n);
                        for s in before.slots() {
                            p.floor[s].union_with(&a);
                        }
                        p.inbound.push(si);
                        o.sync_ancestors[si] = Some(a);
                    }
                    SyncKind::EventSignal {
                        scope,
                        event,
                        value,
                    } => {
                        rep.event_signals += 1;
                        let a = match scope {
                            EventScope::CommandBuffer(cb) => cb_closure
                                .get(cb)
                                .cloned()
                                .unwrap_or_else(|| PosSet::new(n)),
                            EventScope::Queue => queue_closure.clone(),
                        };
                        signals.entry(*event).or_default().push((*value, a.clone()));
                        o.sync_ancestors[si] = Some(a);
                    }
                    SyncKind::EventWait {
                        scope,
                        event,
                        value,
                    } => {
                        rep.event_waits += 1;
                        let later = syncs.iter().enumerate().any(|(j, s)| {
                            host_rank[j] > host_rank[si]
                                && matches!(&s.kind, SyncKind::EventSignal { event: e, value: v, .. }
                                    if e == event && v >= value)
                        });
                        let mut satisfying = signals
                            .get(event)
                            .into_iter()
                            .flatten()
                            .filter(|(v, _)| v >= value)
                            .map(|(_, a)| a);
                        let a = match (later, satisfying.next()) {
                            (false, Some(first)) => {
                                let mut a = first.clone();
                                for other in satisfying {
                                    a.intersect_with(other);
                                }
                                Some(a)
                            }
                            _ => {
                                rep.unordered_event_waits += 1;
                                None
                            }
                        };
                        if let Some(a) = a {
                            match scope {
                                EventScope::CommandBuffer(cb) => {
                                    cb_floor
                                        .entry(*cb)
                                        .or_insert_with(|| PosSet::new(n))
                                        .union_with(&a);
                                    o.cb_inbound.entry(*cb).or_default().push(si);
                                }
                                EventScope::Queue => {
                                    queue_floor.union_with(&a);
                                    o.queue_inbound.push(si);
                                }
                            }
                            o.sync_ancestors[si] = Some(a);
                        }
                    }
                }
            }
            let Some(d) = dispatches.get(k) else {
                break;
            };
            let p = o.pass(d.encoder, n);
            if d.barrier {
                let scope = d.barrier_scope;
                let mut a = PosSet::new(n);
                for s in scope.after.slots() {
                    a.union_with(&p.closure[s]);
                }
                for s in scope.before.slots() {
                    p.floor[s].union_with(&a);
                }
                p.barriers.push((k, scope));
            }
            let mut ancestors = PosSet::new(n);
            let mut first = true;
            for s in d.stage.slots() {
                if first {
                    ancestors = p.floor[s].clone();
                    first = false;
                } else {
                    ancestors.intersect_with(&p.floor[s]);
                }
            }
            ancestors.union_with(&queue_floor);
            let cb = p.command_buffer;
            if let Some(f) = cb.and_then(|cb| cb_floor.get(&cb)) {
                ancestors.union_with(f);
            }
            let mut closed = ancestors.clone();
            closed.insert(k);
            for s in d.stage.slots() {
                p.closure[s].union_with(&closed);
            }
            if let Some(cb) = cb {
                cb_closure
                    .entry(cb)
                    .or_insert_with(|| PosSet::new(n))
                    .union_with(&closed);
            }
            queue_closure.union_with(&closed);
            o.ancestors.push(ancestors);
        }
        rep.passes = o.passes.len();
        o
    }

    fn pass(&mut self, encoder: u64, n: usize) -> &mut Pass {
        let rank = self.passes.len();
        self.passes.entry(encoder).or_insert_with(|| Pass {
            rank,
            command_buffer: None,
            closure: [PosSet::new(n), PosSet::new(n)],
            floor: [PosSet::new(n), PosSet::new(n)],
            barriers: Vec::new(),
            inbound: Vec::new(),
        })
    }

    /// What orders `wi` before `ri`, if anything recorded does.
    fn cover(
        &self,
        wi: usize,
        ri: usize,
        dispatches: &[AuditDispatch],
        syncs: &[AuditSync],
    ) -> Option<Cover> {
        if !self.ancestors[ri].contains(wi) {
            return None;
        }
        let (w, r) = (&dispatches[wi], &dispatches[ri]);
        let pass = &self.passes[&r.encoder];
        if w.encoder == r.encoder {
            let direct = pass.barriers.iter().any(|&(p, scope)| {
                wi < p && p <= ri && scope.after.covers(w.stage) && scope.before.covers(r.stage)
            });
            return Some(if direct {
                Cover::Barrier
            } else {
                Cover::Transitive
            });
        }
        let reaches = |si: &usize| {
            syncs[*si].before_dispatch <= ri
                && self.sync_ancestors[*si]
                    .as_ref()
                    .is_some_and(|a| a.contains(wi))
        };
        let blocks_reader = |si: &usize| match &syncs[*si].kind {
            SyncKind::FenceWait { before, .. } => before.covers(r.stage),
            SyncKind::QueueBarrier { before, .. } => before.covers(r.stage),
            _ => true,
        };
        let cb_inbound = pass
            .command_buffer
            .and_then(|cb| self.cb_inbound.get(&cb))
            .into_iter()
            .flatten();
        pass.inbound
            .iter()
            .chain(cb_inbound)
            .chain(&self.queue_inbound)
            .find(|si| blocks_reader(si) && reaches(si))
            .map(|si| match syncs[*si].kind {
                SyncKind::FenceWait { .. } => Cover::Fence,
                SyncKind::QueueBarrier { .. } => Cover::QueueBarrier,
                _ => Cover::Event,
            })
            .or(Some(Cover::Transitive))
    }
}

// ---------------------------------------------------------------------------
// Recording.
// ---------------------------------------------------------------------------

/// A native fence the recorder keeps alive for the region, so that no later
/// fence can be allocated at its address and inherit its generations.
#[cfg(feature = "hazard-audit")]
struct RetainedFence(#[allow(dead_code)] objc2::rc::Retained<ProtocolObject<dyn MTLFence>>);

// SAFETY: the recorder only retains and releases the fence, never messages it,
// and Objective-C retain/release are thread-safe.
#[cfg(feature = "hazard-audit")]
unsafe impl Send for RetainedFence {}

#[cfg(feature = "hazard-audit")]
struct Pending {
    pass: u64,
    bindings: Vec<AuditBinding>,
    barrier: Option<StageScope>,
    kinds: HazardKinds,
    suppressed: bool,
}

#[cfg(feature = "hazard-audit")]
#[derive(Default)]
struct Recorder {
    recording: bool,
    region: String,
    stream: AuditStream,
    /// Open native encoders: (native address, pass id).
    open: Vec<(usize, u64)>,
    next_pass: u64,
    pending: Vec<Pending>,
    fences: Vec<RetainedFence>,
    commands_compute_passes: u64,
}

#[cfg(feature = "hazard-audit")]
impl Recorder {
    /// The pass a native encoder is recording into, opening one (with an
    /// unknown command buffer) if the recorder has not seen it open.
    fn pass(&mut self, native: usize) -> u64 {
        if let Some(&(_, pass)) = self.open.iter().find(|(n, _)| *n == native) {
            return pass;
        }
        self.open_pass(native, None)
    }

    fn open_pass(&mut self, native: usize, command_buffer: Option<usize>) -> u64 {
        self.close_pass(native);
        let pass = self.next_pass;
        self.next_pass += 1;
        self.open.push((native, pass));
        self.stream.open_pass(pass, command_buffer);
        pass
    }

    fn close_pass(&mut self, native: usize) {
        if let Some(i) = self.open.iter().position(|(n, _)| *n == native) {
            let (_, pass) = self.open.swap_remove(i);
            self.pending.retain(|p| p.pass != pass);
        }
    }

    fn pending(&mut self, native: usize) -> &mut Pending {
        let pass = self.pass(native);
        let i = match self.pending.iter().position(|p| p.pass == pass) {
            Some(i) => i,
            None => {
                self.pending.push(Pending {
                    pass,
                    bindings: Vec::new(),
                    barrier: None,
                    kinds: HazardKinds::NONE,
                    suppressed: false,
                });
                self.pending.len() - 1
            }
        };
        &mut self.pending[i]
    }

    fn push_command(&mut self, native: usize, kernel: &str, stage: Stages) {
        let pass = self.pass(native);
        let pending = self
            .pending
            .iter()
            .position(|p| p.pass == pass)
            .map(|i| self.pending.swap_remove(i));
        let seq = self.stream.dispatches.len() as u64;
        let mut d = AuditDispatch::new(seq, pass, stage, kernel);
        if let Some(p) = pending {
            d.bindings = p.bindings;
            d.barrier = p.barrier.is_some();
            d.barrier_scope = p.barrier.unwrap_or(StageScope::ALL);
            d.kinds = p.kinds;
            d.barrier_suppressed = p.suppressed;
        }
        self.stream.push_dispatch(d);
    }

    fn retain_fence(&mut self, fence: &ProtocolObject<dyn MTLFence>) -> usize {
        let address = fence as *const ProtocolObject<dyn MTLFence> as usize;
        // SAFETY: `fence` is a live Objective-C object borrowed for this call;
        // retaining it only extends its lifetime.
        if let Some(retained) =
            unsafe { objc2::rc::Retained::retain(std::ptr::from_ref(fence).cast_mut()) }
        {
            self.fences.push(RetainedFence(retained));
        }
        address
    }
}

#[cfg(feature = "hazard-audit")]
fn recorder() -> &'static parking_lot::Mutex<Recorder> {
    static STATE: std::sync::OnceLock<parking_lot::Mutex<Recorder>> = std::sync::OnceLock::new();
    STATE.get_or_init(|| parking_lot::Mutex::new(Recorder::default()))
}

/// Run `f` on the recorder when a region is being recorded.
#[cfg(feature = "hazard-audit")]
#[inline]
fn recording(f: impl FnOnce(&mut Recorder)) {
    let mut r = recorder().lock();
    if r.recording {
        f(&mut r);
    }
}

/// A finished recording.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug, Default)]
pub struct RecordedRegion {
    pub region: String,
    pub stream: AuditStream,
    /// Compute passes `Commands` reported opening, for cross-checking the
    /// passes the recorder saw open.
    pub commands_compute_passes: u64,
}

/// Start recording, labelling what follows and discarding anything prior.
#[cfg(feature = "hazard-audit")]
pub fn begin(region: &str) {
    let mut r = recorder().lock();
    *r = Recorder {
        recording: true,
        region: region.to_string(),
        ..Recorder::default()
    };
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn begin(_region: &str) {}

/// Stop recording and return what was recorded.
///
/// `runs` maps a compute command (by its ordinal among compute commands) to
/// `(run index, is_head, icb_emitted_a_setBarrier)` for every position the
/// executor replays; empty on the classical path.
#[cfg(feature = "hazard-audit")]
pub fn finish_recording(runs: &[(usize, u64, bool, bool)]) -> RecordedRegion {
    let mut r = recorder().lock();
    let taken = std::mem::take(&mut *r);
    drop(r);
    let mut stream = taken.stream;
    let compute: Vec<usize> = stream
        .dispatches
        .iter()
        .enumerate()
        .filter(|(_, d)| d.stage == Stages::DISPATCH)
        .map(|(i, _)| i)
        .collect();
    for &(pos, run, is_head, icb_barrier) in runs {
        if let Some(d) = compute.get(pos).and_then(|&i| stream.dispatches.get_mut(i)) {
            d.run = Some(run);
            d.is_run_head = is_head;
            d.icb_barrier = icb_barrier;
        }
    }
    RecordedRegion {
        region: taken.region,
        stream,
        commands_compute_passes: taken.commands_compute_passes,
    }
}

/// Stop recording and cover what was recorded, on observed synchronization.
#[cfg(feature = "hazard-audit")]
pub fn finish(runs: &[(usize, u64, bool, bool)]) -> AuditReport {
    let region = finish_recording(runs);
    region.stream.cover(&region.region, SeamPolicy::Observed)
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn finish(_runs: &[(usize, u64, bool, bool)]) {}

/// A native encoder opened on the command buffer `command_buffer`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_pass_open(encoder: usize, command_buffer: usize) {
    recording(|r| {
        r.open_pass(encoder, Some(command_buffer));
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_pass_open(_encoder: usize, _command_buffer: usize) {}

/// A native encoder ended; its address may now name a different encoder.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_pass_end(encoder: usize) {
    recording(|r| r.close_pass(encoder));
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_pass_end(_encoder: usize) {}

/// `Commands` opened a compute pass. A count only; pass identity comes from
/// the native encoder.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_encoder_begin() {
    recording(|r| r.commands_compute_passes += 1);
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_encoder_begin() {}

/// A buffer bound for the next command of `encoder`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_binding(encoder: usize, range: &BoundRange, is_output: bool) {
    recording(|r| {
        r.pending(encoder).bindings.push(AuditBinding {
            ptr: range.ptr,
            offset: range.offset,
            len: range.len,
            is_output,
        })
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_binding(_encoder: usize, _range: &BoundRange, _is_output: bool) {}

/// A pass barrier encoded natively in `encoder`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_pass_barrier(encoder: usize, scope: StageScope) {
    recording(|r| {
        let p = r.pending(encoder);
        p.barrier = Some(p.barrier.map_or(scope, |s| s.union(scope)));
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_pass_barrier(_encoder: usize, _scope: StageScope) {}

/// The hazard directions the barrier just encoded in `encoder` was owed to.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_barrier_kinds(encoder: usize, kinds: HazardKinds) {
    recording(|r| r.pending(encoder).kinds.merge(kinds));
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_barrier_kinds(_encoder: usize, _kinds: HazardKinds) {}

/// A barrier owed before the next command of `encoder` was deferred rather
/// than emitted. A deferred barrier orders nothing.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_barrier_suppressed(encoder: usize) {
    recording(|r| r.pending(encoder).suppressed = true);
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_barrier_suppressed(_encoder: usize) {}

/// A compute dispatch in `encoder`, consuming the bindings and barrier
/// recorded for it.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_dispatch(encoder: usize, kernel: &str) {
    recording(|r| r.push_command(encoder, kernel, Stages::DISPATCH));
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_dispatch(_encoder: usize, _kernel: &str) {}

/// A blit command in `encoder` over the given `(buffer, offset, len,
/// is_output)` ranges.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_blit(encoder: usize, kernel: &'static str, ranges: &[(usize, usize, usize, bool)]) {
    recording(|r| {
        let pending = r.pending(encoder);
        for &(ptr, offset, len, is_output) in ranges {
            pending.bindings.push(AuditBinding {
                ptr,
                offset,
                len,
                is_output,
            });
        }
        r.push_command(encoder, kernel, Stages::BLIT);
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_blit(
    _encoder: usize,
    _kernel: &'static str,
    _ranges: &[(usize, usize, usize, bool)],
) {
}

/// `encoder` updated `fence` after its earlier commands in `after`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_fence_update(encoder: usize, fence: &ProtocolObject<dyn MTLFence>, after: Stages) {
    recording(|r| {
        let pass = r.pass(encoder);
        let fence = r.retain_fence(fence);
        r.stream.fence_update(pass, fence, after);
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_fence_update(_encoder: usize, _fence: &ProtocolObject<dyn MTLFence>, _after: Stages) {
}

/// `encoder` waits on `fence` before its later commands in `before`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_fence_wait(encoder: usize, fence: &ProtocolObject<dyn MTLFence>, before: Stages) {
    recording(|r| {
        let pass = r.pass(encoder);
        let fence = r.retain_fence(fence);
        r.stream.fence_wait(pass, fence, before);
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_fence_wait(_encoder: usize, _fence: &ProtocolObject<dyn MTLFence>, _before: Stages) {}

/// A consumer queue barrier encoded in `encoder`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_queue_barrier(encoder: usize, after: Stages, before: Stages) {
    recording(|r| {
        let pass = r.pass(encoder);
        r.stream.queue_barrier(pass, after, before);
    });
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_queue_barrier(_encoder: usize, _after: Stages, _before: Stages) {}

/// An event signal of `value`, ordered after the commands of `scope`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_event_signal(scope: EventScope, event: usize, value: u64) {
    recording(|r| r.stream.event_signal(scope, event, value));
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_event_signal(_scope: EventScope, _event: usize, _value: u64) {}

/// An event wait for `value` or higher, blocking the later commands of `scope`.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_event_wait(scope: EventScope, event: usize, value: u64) {
    recording(|r| r.stream.event_wait(scope, event, value));
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_event_wait(_scope: EventScope, _event: usize, _value: u64) {}
