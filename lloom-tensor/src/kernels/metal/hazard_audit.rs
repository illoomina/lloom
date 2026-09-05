//! Strict ordering verification: reconstruct the dependency graph and assert
//! every edge is covered (issue #185).
//!
//! # What this answers, and why a digest cannot
//!
//! Under `HazardTrackingModeUntracked` (`DESIGN.md` §3.5) the driver does no
//! dependency analysis, so a wrongly-dropped ordering edge is **silent
//! corruption rather than an error**. The project's two gates -- §15.1 #7's
//! determinism digest and §15.1 #8's `lfm2-smoke` -- are both *outcome* tests,
//! and #144 measured that neither can see this class:
//!
//! > A predicate wrong in exactly the way §11.3p warns against passes too.
//! > Mutating production source to suppress the **30 run heads** moves
//! > `suppressed` to 32225 and leaves the digest **identical**, over three runs.
//!
//! The ordering survived by luck -- every run head happens to be preceded by a
//! classically-encoded gap position that emits -- and luck is not a property a
//! second model family inherits (#92, #90, #91). §11.0a states the rule this
//! module exists to satisfy: *can the instrument express the difference it is
//! meant to detect?* A digest expresses **inequality**; this expresses **which
//! edge, between which two dispatches, in which direction, and why it is or is
//! not ordered**.
//!
//! # Three primitives, and the cover must span all of them
//!
//! §11.3o: *an ordering count is only interpretable against a fixed primitive*,
//! and three appear in this one question. They are not commensurable, so the
//! audit models each separately and an edge is covered when **any** orders it:
//!
//! | primitive | what it is | what it orders |
//! |---|---|---|
//! | `memoryBarrierWithScope` | a fence **in the stream** | everything before it against everything after, *including* the interleaved gap positions |
//! | the ICB's `setBarrier` | a property of **one command** | that command against its predecessors **within its own run**; successors inherit nothing |
//! | an encoder-session seam | a fresh `EncoderState` | via `prev_ce_outputs` + `wait_for_buffer`, keyed on the **pointer alone** (§9.2j: no precision axis) |
//!
//! **Modelling fewer than three reports false positives.** #144 found the case
//! and it was not anticipated: 11 gap-writer edges out of one `sdpa_vector`
//! position are ordered by a session seam and by nothing else, so an audit
//! knowing only about barriers would report 11 failures on a configuration that
//! is correct. That is §11.3p's obstacle 3 defusing its obstacle 2, and no
//! barrier count shows it.
//!
//! # Why a feature, and why it composes with release
//!
//! `run-telemetry`'s shape (#171), for its reasons. An environment check is a
//! cached bool, but the code around it is still compiled and any state it
//! touches is still allocated; §6.4a measured what that costs when a hook is not
//! free -- a `String` per dispatch, 675 per token, ~3.6 % of forward-pass CPU,
//! *the same order as the thing being measured*.
//!
//! Off, every entry point below is an empty inline function that drops its
//! arguments, so there is no call to elide, no branch to predict, no counter to
//! contend on and no `Vec` to grow. The off-cost claim is therefore
//! **structural**, and the archive is the evidence rather than a timing.
//!
//! # What it does not do
//!
//! **It is not a gate.** §15.1's gates each exist because a specific failure
//! demanded them; this is diagnostic, reached for when an ordering decision is
//! being changed. Whether it becomes a gate is a later decision with its own
//! evidence.
//!
//! **It does not model `HazardKey`.** The audit asks whether the edges are
//! covered; *which* edges exist is `HazardKey`'s question, settled and measured
//! (§9.2e/§9.2f). #185 is about observing what either keying produces.
//!
//! **It under-reports nothing and over-reports deliberately.** An unknown extent
//! covers the whole allocation, so a pair whose overlap cannot be decided is
//! reported as an edge. A spurious edge costs a false positive a human reads; a
//! missed one costs the silence this exists to break.

#[cfg(feature = "hazard-audit")]
use crate::kernels::metal::encoder::{BoundRange, HazardKind, HazardKinds};
#[cfg(feature = "hazard-audit")]
use std::sync::{Mutex, OnceLock};

/// One binding as the audit sees it -- the full `(buffer, offset, len,
/// direction)` tuple, which is what an interval test needs on both sides.
///
/// `len` is the field #144's `edge-cover.py` did not have and had to work
/// around: keyed on `(buffer, offset)`, equality is *sufficient* for overlap and
/// not *necessary*, so that model under-reports edges and can only be read
/// against its mutation controls. `trace::Binding` gained the extent in the same
/// change as this module.
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
    /// Whether these two bindings can touch the same byte.
    ///
    /// The same rule as `BoundRange::overlaps`, including its deliberate
    /// asymmetry on a zero length.
    fn overlaps(&self, other: &AuditBinding) -> bool {
        if self.ptr != other.ptr {
            return false;
        }
        if self.len == 0 || other.len == 0 {
            return true;
        }
        self.offset < other.offset + other.len && other.offset < self.offset + self.len
    }

    /// The hazard direction from `earlier` to this binding, if the pair is
    /// orderable at all.
    ///
    /// Read-after-read is `None`: two reads never conflict, which is why the
    /// weights -- bound on every dispatch -- contribute no edges (issue #185).
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

/// One dispatch position, with everything the cover test needs about it.
#[cfg(feature = "hazard-audit")]
#[derive(Clone)]
pub struct AuditDispatch {
    pub seq: u64,
    pub kernel: String,
    pub bindings: Vec<AuditBinding>,
    /// Whether candle emitted `memoryBarrierWithScope` before this position.
    pub barrier: bool,
    /// The directions that barrier was owed to.
    pub kinds: HazardKinds,
    /// Whether candle had a barrier pending here and suppressed it (§11.3r).
    pub barrier_suppressed: bool,
    /// The encoder session this position belongs to. Two positions in different
    /// sessions cannot be ordered by a barrier at all, only by the seam.
    pub encoder: u64,
    /// The ICB run this position replays in, when it is replayed at all.
    pub run: Option<u64>,
    /// Whether this position is the head of its run. A head's scan slice is
    /// empty by construction, so its command orders nothing within the run.
    pub is_run_head: bool,
    /// Whether the ICB encoded a `setBarrier` onto this position's command.
    pub icb_barrier: bool,
}

/// Which primitive ordered an edge.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cover {
    /// A surviving `memoryBarrierWithScope` between the two ends.
    Barrier,
    /// The two ends are in different encoder sessions.
    Seam,
    /// The ICB's `setBarrier` on the reader's command, within one run.
    Icb,
}

/// A dependency edge and what, if anything, orders it.
#[cfg(feature = "hazard-audit")]
#[derive(Clone, Debug)]
pub struct Edge {
    pub writer_seq: u64,
    pub reader_seq: u64,
    pub writer_kernel: String,
    pub reader_kernel: String,
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
    pub by_seam: usize,
    pub by_icb: usize,
    /// The edges nothing orders. Empty is the passing verdict.
    pub uncovered: Vec<Edge>,
    /// Barriers observed, attributed by the direction that caused them --
    /// the artifact §11.3p could not produce (issue #185).
    pub barriers: usize,
    pub barriers_raw: usize,
    pub barriers_waw: usize,
    pub barriers_war: usize,
    pub barriers_suppressed: usize,
}

#[cfg(feature = "hazard-audit")]
impl AuditReport {
    /// Whether every dependency edge is ordered by something.
    pub fn is_clean(&self) -> bool {
        self.uncovered.is_empty()
    }

    /// A human-readable summary, including the per-kind attribution.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut o = String::new();
        let _ = writeln!(
            o,
            "region={} positions={} edges={} (raw={} waw={} war={})",
            self.region, self.positions, self.edges, self.edges_raw, self.edges_waw, self.edges_war
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
            "  cover: barrier={} seam={} icb={}  UNCOVERED={}",
            self.by_barrier,
            self.by_seam,
            self.by_icb,
            self.uncovered.len()
        );
        for e in self.uncovered.iter().take(40) {
            let _ = writeln!(
                o,
                "    !! #{} {} -> #{} {}  {} buf={:#x} off={} len={}",
                e.writer_seq,
                e.writer_kernel,
                e.reader_seq,
                e.reader_kernel,
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

#[cfg(feature = "hazard-audit")]
struct AuditState {
    recording: bool,
    region: String,
    dispatches: Vec<AuditDispatch>,
    pending: Vec<AuditBinding>,
    pending_barrier: bool,
    pending_kinds: HazardKinds,
    pending_suppressed: bool,
    encoder: u64,
}

#[cfg(feature = "hazard-audit")]
fn state() -> &'static Mutex<AuditState> {
    static STATE: OnceLock<Mutex<AuditState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(AuditState {
            recording: false,
            region: String::new(),
            dispatches: Vec::new(),
            pending: Vec::new(),
            pending_barrier: false,
            pending_kinds: HazardKinds::NONE,
            pending_suppressed: false,
            encoder: 0,
        })
    })
}

// ---------------------------------------------------------------------------
// Recording hooks.
//
// Each has an off-arm with an empty body, so with the feature off there is no
// call, no branch and no state -- `run-telemetry`'s shape (#171).
// ---------------------------------------------------------------------------

/// Start auditing, labelling what follows and discarding anything prior.
#[cfg(feature = "hazard-audit")]
pub fn begin(region: &str) {
    if let Ok(mut s) = state().lock() {
        s.recording = true;
        region.clone_into(&mut s.region);
        s.dispatches.clear();
        s.pending.clear();
        s.pending_barrier = false;
        s.pending_kinds = HazardKinds::NONE;
        s.pending_suppressed = false;
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn begin(_region: &str) {}

/// Stop auditing and run the cover test over what was recorded.
///
/// `runs` maps a position to `(run index, is_head, icb_emitted_a_setBarrier)`
/// for every position the executor replays. Empty on the classical path, where
/// only two of the three primitives are in play.
#[cfg(feature = "hazard-audit")]
pub fn finish(runs: &[(usize, u64, bool, bool)]) -> AuditReport {
    let mut s = match state().lock() {
        Ok(s) => s,
        Err(_) => return AuditReport::default(),
    };
    s.recording = false;
    let mut dispatches = std::mem::take(&mut s.dispatches);
    let region = s.region.clone();
    drop(s);

    for &(pos, run, is_head, icb_barrier) in runs {
        if let Some(d) = dispatches.get_mut(pos) {
            d.run = Some(run);
            d.is_run_head = is_head;
            d.icb_barrier = icb_barrier;
        }
    }
    cover(&region, &dispatches)
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn finish(_runs: &[(usize, u64, bool, bool)]) {}

/// The cover test: every dependency edge, and which primitive orders it.
///
/// Separated from [`finish`] so it can be driven from a unit test with a
/// synthetic stream -- which is how the known-good and known-bad predicates are
/// exercised without a GPU.
#[cfg(feature = "hazard-audit")]
pub fn cover(region: &str, dispatches: &[AuditDispatch]) -> AuditReport {
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

    // Positions carrying a surviving classical fence, ascending, so the "is
    // there a fence between w and r" test is a binary search rather than a scan.
    let fences: Vec<usize> = dispatches
        .iter()
        .enumerate()
        .filter(|(_, d)| d.barrier)
        .map(|(i, _)| i)
        .collect();

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

            // (1) A surviving fence at p with w < p <= r orders everything
            //     before p against everything after -- including the uncovered
            //     positions interleaved between ICB runs, which is the asymmetry
            //     §11.3m records and `setBarrier` does not have.
            let by_fence =
                fences.partition_point(|&p| p <= wi) < fences.partition_point(|&p| p <= ri);
            if by_fence {
                rep.by_barrier += 1;
                continue;
            }

            // (2) Different encoder sessions: hazard state is per session, so
            //     `auto_barrier` could not have ordered this pair even in
            //     principle. What does is `prev_ce_outputs` + `wait_for_buffer`
            //     on the shared pointer (§6.4, §9.2j).
            if w.encoder != r.encoder {
                rep.by_seam += 1;
                continue;
            }

            // (3) The ICB's `setBarrier` on the reader's command. It orders that
            //     command against its predecessors *within its own run*, so all
            //     four conditions are required: both ends replayed, in the same
            //     run, the reader carrying a barrier, and the reader not a head
            //     (a head's scan slice is empty by construction).
            let same_run = w.run.is_some() && w.run == r.run;
            if same_run && r.icb_barrier && !r.is_run_head {
                rep.by_icb += 1;
                continue;
            }

            rep.uncovered.push(Edge {
                writer_seq: w.seq,
                reader_seq: r.seq,
                writer_kernel: w.kernel.clone(),
                reader_kernel: r.kernel.clone(),
                kind,
                ptr,
                offset,
                len,
                cover: None,
            });
        }
    }
    rep
}

/// The strongest hazard between two dispatches, if any.
///
/// "Strongest" only in the sense of *first found*; a pair may carry several
/// bindings in different directions, and any one of them requires ordering, so
/// the edge exists either way. The kind is reported for attribution.
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

/// Record a binding against the dispatch that will consume it.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_binding(range: &BoundRange, is_output: bool) {
    if let Ok(mut s) = state().lock() {
        if !s.recording {
            return;
        }
        s.pending.push(AuditBinding {
            ptr: range.ptr,
            offset: range.offset,
            len: range.len,
            is_output,
        });
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_binding(_range: &crate::kernels::metal::encoder::BoundRange, _is_output: bool) {}

/// Record that candle emitted a barrier, and which directions it was owed to.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_barrier(kinds: HazardKinds) {
    if let Ok(mut s) = state().lock() {
        if !s.recording {
            return;
        }
        s.pending_barrier = true;
        s.pending_kinds = kinds;
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_barrier(_kinds: crate::kernels::metal::encoder::HazardKinds) {}

/// Record that a pending barrier was suppressed rather than emitted (§11.3r).
///
/// The audit must know, because a suppressed barrier is **not** a surviving
/// fence: the whole question #144 poses is whether the ICB's `setBarrier` and
/// the session seams cover what it would have ordered.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_barrier_suppressed() {
    if let Ok(mut s) = state().lock() {
        if !s.recording {
            return;
        }
        s.pending_suppressed = true;
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_barrier_suppressed() {}

/// Record that a new encoder session began, resetting hazard state.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_encoder_begin() {
    if let Ok(mut s) = state().lock() {
        if !s.recording {
            return;
        }
        s.encoder += 1;
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_encoder_begin() {}

/// Record a dispatch, consuming the bindings and barrier state before it.
#[cfg(feature = "hazard-audit")]
#[inline]
pub fn record_dispatch(kernel: &str) {
    if let Ok(mut s) = state().lock() {
        // Cleared even when not recording, so turning the audit on mid-stream
        // cannot attribute a previous dispatch's bindings to the first recorded
        // one -- `trace::record_dispatch`'s rule, for its reason.
        if !s.recording {
            s.pending.clear();
            s.pending_barrier = false;
            s.pending_suppressed = false;
            s.pending_kinds = HazardKinds::NONE;
            return;
        }
        let seq = s.dispatches.len() as u64;
        let bindings = std::mem::take(&mut s.pending);
        let barrier = std::mem::take(&mut s.pending_barrier);
        let barrier_suppressed = std::mem::take(&mut s.pending_suppressed);
        let kinds = s.pending_kinds.take();
        let encoder = s.encoder;
        s.dispatches.push(AuditDispatch {
            seq,
            kernel: kernel.to_string(),
            bindings,
            barrier,
            kinds,
            barrier_suppressed,
            encoder,
            run: None,
            is_run_head: false,
            icb_barrier: false,
        });
    }
}

#[cfg(not(feature = "hazard-audit"))]
#[inline(always)]
pub fn record_dispatch(_kernel: &str) {}
