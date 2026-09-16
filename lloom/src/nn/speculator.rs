//! The drafter surface: `Speculator`, `TargetProbe`, `KvSource`, `Draft`,
//! `TapSpec` (`RESTRUCTURING_DESIGN.md` §3.4).
//!
//! One target, many drafter families. The four families llama.cpp ships (DSpark/
//! DFlash, EAGLE3, MTP, n-gram) differ on four concrete axes, and the surface has
//! one slot per axis so a new family is one `Probe` impl plus one `Speculator`
//! impl, with the decode loop, the verify step and the target's hot loop
//! untouched (§3.4.2's "implement, not re-architect").
//!
//! | axis | slot |
//! |---|---|
//! | what it reads from the target | [`Speculator::Probe`] (a [`TargetProbe`]) |
//! | own weights | the impl |
//! | where its draft-KV comes from | [`Speculator::kv_source`] (a value, not a type) |
//! | cross-round state | private to the impl (`Vec<PerSeq>`) |
//!
//! The one thing all families share is a lifecycle — begin, observe
//! (=process+accept), draft — and that is the trait; everything else is
//! associated structure (`speculative.cpp:163`).
//!
//! The compile-time contract lives in [`Probes`]: a target can only pair with a
//! drafter whose probe it can satisfy, so an MTP drafter against a target with no
//! `h_nextn` head does not compile rather than failing at runtime (§3.4.3).

use crate::nn::model::{CacheReset, CacheRewind, CacheShare, CausalLM};
use lloom_tensor::{DType, Device, Result, Tensor};

/// A sequence slot on the serving side. The `Speculator` is constructed for a
/// fixed number of these (server slots), and per-seq state lives *inside* the
/// impl keyed by this id — never on the public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub usize);

// ---------------------------------------------------------------------------
// Layer 2 — `TargetProbe`: what a family reads from ONE target forward.
// ---------------------------------------------------------------------------

/// What a drafter family extracts from a single target forward pass — the axis
/// that makes DSpark ≠ MTP ≠ n-gram (`RESTRUCTURING_DESIGN.md` §3.4.2).
///
/// A probe is a zero-sized type per family; the target's hot loop is woven (via
/// the tap hooks, [`ProbeHooks`]) to fill its `Output`. This generalises §3.4's
/// original `draft!`/`TapSpec` to a compile-time description of what the target
/// must expose.
pub trait TargetProbe {
    /// The extracted tensors, this family's shape. For DSpark it is the `N`
    /// tapped residuals; for MTP the target's own `h_nextn`; for n-gram `()`.
    type Output;

    /// The hooks this probe weaves into the target's layer loop. Generalises
    /// §3.4's `begin_taps`/`tap_layer_input`.
    fn hooks() -> ProbeHooks;
}

/// The target-side weaving a [`TargetProbe`] requests. Named rather than a
/// closure so the target's loop can match on it at construction.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeHooks {
    /// N layer-residual taps at fixed indices and a side ([`TapSpec`]).
    Taps(TapSpec),
    /// The target's own next-token hidden head — no taps woven; the target
    /// already computes it and the probe just names it.
    NextnHead,
    /// Nothing woven.
    None,
}

/// DSpark / EAGLE3: `N` layer-residual taps at fixed indices and a side.
///
/// `<LayerTaps<5> as TargetProbe>::Output` — the five concatenated residuals — is
/// exactly what DSpark's projection reads: `H_ctx = RMSNorm_enc(W_c·[H(l₁);…;
/// H(l₅)])` (`lfm2_dspark.rs`), `fc.weight [hidden, n_tap×hidden]`.
#[derive(Debug, Clone, Copy)]
pub struct LayerTaps<const N: usize>;

/// MTP: the target's own next-token hidden state — NO taps; the target already
/// computes `h_nextn` and the probe names it.
#[derive(Debug, Clone, Copy)]
pub struct NextnHidden;

/// n-gram: nothing from the target.
#[derive(Debug, Clone, Copy)]
pub struct NoProbe;

impl<const N: usize> TargetProbe for LayerTaps<N> {
    /// The `N` tapped residuals, each `[b, seq, hidden]`. `Vec` rather than
    /// `[Tensor; N]` because the tap-order contract (`TapSpec::layers`) is a
    /// runtime slice the drafter checkpoint carries, and its length is checked
    /// against `N` where a `LayerTaps<N>` is paired with a `TapSpec`, not at the
    /// type level — `[Tensor; N]` would force every consumer through a
    /// fallible `try_into` at exactly the seam that already range-checks the
    /// indices (`Cache::new_with`).
    type Output = Vec<lloom_tensor::Tensor>;

    fn hooks() -> ProbeHooks {
        // The concrete layers/side/dtype are the drafter checkpoint's, supplied
        // where a `Speculator` is constructed (DSpark reads them from its
        // `DSparkConfig`); a bare `LayerTaps<N>` names only that it taps `N`
        // layers on some side. The placeholder here carries `N` and the
        // reference side so a caller that never overrides it still gets the
        // llama.cpp convention.
        ProbeHooks::Taps(TapSpec {
            layers: &[],
            side: TapSide::LayerOutput,
            dtype: DType::F32,
        })
    }
}

impl TargetProbe for NextnHidden {
    type Output = lloom_tensor::Tensor;
    fn hooks() -> ProbeHooks {
        ProbeHooks::NextnHead
    }
}

impl TargetProbe for NoProbe {
    type Output = ();
    fn hooks() -> ProbeHooks {
        ProbeHooks::None
    }
}

/// The `draft!` declaration's tap spec — the layers to tap, the side, and the
/// capture dtype (`RESTRUCTURING_DESIGN.md` §3.4.2).
///
/// `layers` is the drafter's own `target_layer_ids`-derived order, and that
/// order is load-bearing: a permuted concat pairs every residual with the wrong
/// block of `W_c`, is algebraically valid, and is invisible to every gate
/// (`DESIGN.md` §10.2n). The consumer never re-sorts.
#[derive(Debug, Clone, PartialEq)]
pub struct TapSpec {
    pub layers: &'static [usize],
    pub side: TapSide,
    pub dtype: DType,
}

/// Which residual a tap captures — the input to a layer or its output.
///
/// The reference convention is [`TapSide::LayerOutput`]: llama.cpp's converter
/// writes `dflash.target_layers = [l + 1 for l in target_layer_ids]`, so the
/// tapped representation is layer `l`'s **output** residual, captured as the
/// **input** of layer `l + 1`. In the tree's wired path this is
/// `DSparkConfig::tap_layers()` (which already applies the `+ 1`) at
/// `--tap-shift 0`; a `--tap-shift 1` moves the tap one layer LATE, off the
/// reference (`lfm2_dspark.rs`, and the drafter examples' NOTE lines;
/// `DESIGN.md` §10.2s). [`TapSide::LayerInput`] is the one-layer-early arm
/// (`--tap-shift -1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapSide {
    /// Capture the residual entering the named layer — one layer early relative
    /// to the reference.
    LayerInput,
    /// The reference: capture the named layer's output residual (delivered as
    /// the input of the next layer, `target_layer_ids + 1`).
    LayerOutput,
}

// ---------------------------------------------------------------------------
// Layer 1 — `Speculator`, the seam the decode loop names.
// ---------------------------------------------------------------------------

/// Where a family's draft-KV comes from — read by the driver, not encoded as a
/// type (`RESTRUCTURING_DESIGN.md` §3.4.3).
///
/// A single family (MTP) has both variants by checkpoint, so it cannot be an
/// associated type; llama.cpp derives it (`is_mem_shared = (ctx_dft == ctx_tgt)`,
/// `speculative.cpp:1374`) and it changes exactly two driver behaviours — skip
/// the catch-up decode and share the target's position for all draft tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSource {
    /// The drafter keeps its own draft-KV (DSpark's `CtxCache`, EAGLE3, most MTP).
    Own,
    /// The drafter shares the target's KV cache (gemma4's `is_mem_shared`).
    SharedWithTarget,
}

/// The decode cursor for one sequence: the last committed token and the target
/// position the verify window starts at.
///
/// Threaded by the driver rather than carried on [`Draft`] (§3.4.1 locks
/// `Draft` to `{tokens, confidence}`): `last` and `pos` are the *target's*
/// decode state, not the drafter's proposal, so they belong to the caller that
/// owns the target cache. `verify_step` builds `[last, proposed…]` at `pos`.
#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    /// The last committed token — the one whose logits produced `proposed[0]`.
    pub last: u32,
    /// The absolute target position the window begins at.
    pub pos: usize,
}

/// The ids of one drafted block, where the family produced them.
///
/// A drafter that samples on the device (DSpark's `sample_block` argmax
/// chain) hands its ids over as the `[k]` `U32` tensor they already are, and
/// the verify feeds that tensor straight into the target's window. Draining
/// them to the host first is a blocking readback the round does not need —
/// the host learns what was accepted from the verify's own readback, and the
/// draft ids ride along in it. A family that drafts on the host (n-gram, a
/// test double) hands over a `Vec` and the verify uploads it.
///
/// Either way `len()` is the block width and the verify's window is `k + 1`.
#[derive(Debug, Clone)]
pub enum DraftTokens {
    /// Host-side ids; the verify uploads them.
    Host(Vec<u32>),
    /// Device-resident ids: a rank-1 `U32` tensor of length `k`.
    Device(Tensor),
}

impl DraftTokens {
    /// The block width `k`.
    pub fn len(&self) -> usize {
        match self {
            Self::Host(v) => v.len(),
            Self::Device(t) => t.elem_count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keep the first `n` ids. A `Device` block is narrowed, not read back.
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        match self {
            Self::Host(v) => v.truncate(n),
            Self::Device(t) => {
                if n < t.elem_count() {
                    *t = t.narrow(0, 0, n)?;
                }
            }
        }
        Ok(())
    }
}

/// One drafted block for one sequence.
pub struct Draft {
    pub tokens: DraftTokens,
    /// Per-position `P(this token)` in `[0, 1]` if the family exposes a
    /// confidence signal (DSpark's confidence head) AND the driver asked for
    /// it ([`Speculator::want_confidence`]), EMPTY otherwise. NOT a
    /// probability the target must honour — a hint. The DRIVER owns the
    /// `p_min` threshold ([`Session::speculative_step`]); the family never
    /// sees it. One knob, many families.
    pub confidence: Vec<f32>,
}

impl Draft {
    /// A block drafted on the host, with no confidence hint.
    pub fn host(tokens: Vec<u32>) -> Self {
        Self {
            tokens: DraftTokens::Host(tokens),
            confidence: Vec::new(),
        }
    }
}

/// A speculative proposer: one target, one drafter family.
///
/// Constructed for a fixed number of sequences (server slots); per-seq state
/// lives INSIDE the impl as a private `Vec<PerSeq>`. That is what lets one
/// [`Speculator::draft`] build a batch across all active sequences in one call —
/// B>1 is the whole reason DSpark can pay (`DESIGN.md` §10.2r: at B=1 it does
/// not; the amortization is `B×K`), so the batch axis is the trait's native
/// shape rather than something bolted around a per-seq trait. The decode loop
/// drives it and never sees which family it is.
pub trait Speculator {
    /// What this family reads from ONE target forward. Associated, not a field,
    /// so a family that reads nothing (n-gram) pays for nothing (B8 applied).
    type Probe: TargetProbe;

    /// The proposer's slot count. Per-seq state is private (`Vec<PerSeq>`).
    fn n_seq(&self) -> usize;

    /// Where this family's draft-KV comes from. Default `Own`; a single family
    /// (MTP) overrides per checkpoint.
    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }

    /// The block ceiling — DSpark's `block_size − logits_start`, MTP's
    /// `n_mtp_layers`, the n-gram match length. The loop sizes the window from it.
    fn max_draft(&self) -> usize;

    /// Seed sequence `seq`'s state from its prompt.
    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()>;

    /// Propose for the flagged-active sequences — ONE call, ALL sequences, so
    /// the driver verifies them in one batched target forward (§10.2m's B>1 win).
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>>;

    /// Whether the driver will read [`Draft::confidence`] this round.
    ///
    /// The driver owns the `p_min` threshold and calls this with `p_min > 0`
    /// before each `draft`; the family never learns the threshold, only
    /// whether the hint is consumed. A hint nobody reads is a head forward
    /// and a blocking host readback per round for nothing, so a family with a
    /// confidence head produces it only when asked. Default: ignored — a
    /// family with no confidence signal leaves `confidence` empty either way.
    fn want_confidence(&mut self, _wanted: bool) {}

    /// Fold sequence `seq`'s verified rows back in. This is llama.cpp's
    /// `process` and `accept` merged: every family's "accept" is "update my
    /// state given what the target kept," and separating them needed a stashed
    /// `verify_h`. One call, one place the state moves. `probe` is what the
    /// target's forward extracted this round.
    fn observe(
        &mut self,
        seq: SeqId,
        probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Layer 3 — the verify/accept loop, family-blind; the pairing proof.
// ---------------------------------------------------------------------------

/// A target that can satisfy probe `P` — the compile-time pairing contract
/// (`RESTRUCTURING_DESIGN.md` §3.4.3).
///
/// `M: Probes<S::Probe>` in [`Session::speculative_step`] makes the §3.4 runtime
/// tap refusal (`Cache::new_with`) a **type error**: an MTP drafter
/// (`Probe = NextnHidden`) against a target with no `h_nextn` head does not
/// compile, because that target does not `impl Probes<NextnHidden>`. This is the
/// #619 "a mismatch is a compile error" principle applied to speculation.
///
/// `verify_and_probe` runs one target forward over `[last, draft…]`, accepts the
/// longest correct prefix per sequence, and extracts the probe in the SAME pass
/// (§10.2r amortization). The implementor wires it to its existing verify entry
/// (`Model::verify_step` + `Cache::take_taps` for LFM2).
///
/// # The mismatch is a compile error (the §3.4.3 gate)
///
/// LFM2 implements `Probes<LayerTaps<5>>` and NOT `Probes<NextnHidden>`, so an
/// MTP-shaped drafter (`Probe = NextnHidden`, needing an `h_nextn` head LFM2
/// does not have) cannot be paired with it — the bound `M: Probes<S::Probe>` on
/// [`Session::speculative_step`] fails to hold and the program does not compile:
///
/// ```compile_fail
/// use lloom::models::lfm2::Model;
/// use lloom::nn::speculator::{NextnHidden, Probes};
/// // A target must satisfy the probe a drafter reads. LFM2 does not have an
/// // `h_nextn` head, so it does not `impl Probes<NextnHidden>` — this bound is
/// // unsatisfiable and the call does not compile.
/// fn requires_nextn<M: Probes<NextnHidden>>() {}
/// requires_nextn::<Model>();
/// ```
///
/// The matching probe compiles (`Model: Probes<LayerTaps<5>>`):
///
/// ```
/// use lloom::models::lfm2::Model;
/// use lloom::nn::speculator::{LayerTaps, Probes};
/// fn requires_taps<M: Probes<LayerTaps<5>>>() {}
/// requires_taps::<Model>();
/// ```
pub trait Probes<P: TargetProbe>: CausalLM {
    /// Verify the drafts and extract the probe in one pass. Returns, per active
    /// sequence, the accepted tokens (never empty — the target's own prediction
    /// at position 0 is always correct), plus the probe output for the round.
    ///
    /// `eos_ids` is the run's stop set, and an EOS the target emits ends the
    /// accepted prefix at itself: the accepted length is what the cache keeps,
    /// what the probe is trimmed to and how far the driver's cursor moves, so
    /// a prefix past an EOS would carry all three past the reply's end
    /// (`design/docs/speculation.md` §10.2z-ii). Empty draws no boundary.
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, P::Output)>;

    /// The receipt of a verify pass that has been encoded and not read back.
    type Pending;

    /// Encode one sequence's verify — the target forward over `[last, draft…]`
    /// and its argmax — and return without waiting for the GPU. The host learns
    /// nothing here; [`Self::verify_complete`] reads the result back.
    ///
    /// The split is what lets a driver run another sequence's draft and
    /// observe while this verify is on the GPU
    /// ([`Session::decode_speculative_pipelined`]). [`Self::verify_and_probe`]
    /// is the two halves back to back and stays the single-sequence path.
    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Self::Cache,
    ) -> Result<Self::Pending>;

    /// Read back the pass [`Self::verify_enqueue`] encoded: the accepted tokens
    /// (never empty) and the round's probe, with the cache resolved to the
    /// accepted length. The wait is on the pass's own command buffer, so work
    /// queued behind it — another sequence's round — keeps running. `eos_ids`
    /// bounds the prefix as in [`Self::verify_and_probe`].
    fn verify_complete(
        &self,
        pending: Self::Pending,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, P::Output)>;

    /// Drain the probe the last non-verify pass (the prefill) left in `cache`,
    /// covering every position that pass ran. A drafter is keyed from the
    /// prompt with this before round 0 — the prompt is the context a draft
    /// copies from, and a drafter that never sees it draws from the generated
    /// tokens alone (`design/docs/speculation.md` §10.2w).
    fn take_probe(&self, cache: &mut Self::Cache) -> Result<P::Output>;

    /// How many probes a forward has discarded untaken so far — a target that
    /// overwrites its probe per forward reports the count, one that keeps
    /// none reports 0. The drivers refuse a run in which it moves: a drafter
    /// seeded past a dropped probe has missed positions, and only the accept
    /// rate would ever show it.
    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        let _ = cache;
        0
    }
}

/// How [`Session::speculative_round`] brackets its phases.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundProbe<'d> {
    /// Synchronize this device at the end of every phase, so a phase's GPU
    /// work is charged to it. `None` brackets the host only.
    pub sync: Option<&'d Device>,
    /// Also take the per-phase dispatch census by pipeline label (needs
    /// `sync` and the Metal profiler). Allocates; ask for one round, not all.
    pub census: bool,
}

/// One phase's dispatch census: `(pipeline label, dispatches)`, descending.
pub type PhaseCensus = Vec<(String, u64)>;

impl RoundProbe<'_> {
    /// Close a phase: synchronize if asked, and take `(gpu_busy_union_s,
    /// dispatches, census)` from the profiler if it is on, resetting it.
    fn close(&self) -> Result<(f64, u64, Option<PhaseCensus>)> {
        let Some(device) = self.sync else {
            return Ok((0.0, 0, None));
        };
        device.synchronize()?;
        #[cfg(feature = "metal")]
        {
            use lloom_tensor::metal_backend::profile;
            if profile::enabled() {
                let s = profile::snapshot();
                profile::reset();
                let census = if self.census { Some(s.by_label) } else { None };
                return Ok((s.gpu_busy_union_s, s.dispatches, census));
            }
        }
        Ok((0.0, 0, None))
    }
}

/// One speculative round's phase brackets, from [`Session::speculative_round`].
///
/// Wall is the host's clock around each phase. GPU-busy and dispatches are
/// per phase and only nonzero when the round was probed with `sync` and the
/// Metal profiler on; the census only when it was asked for.
#[derive(Debug, Clone, Default)]
pub struct RoundTiming {
    pub draft_ns: u64,
    pub verify_ns: u64,
    pub observe_ns: u64,
    /// Under the pipelined driver, the two halves of `verify_ns`: the host's
    /// encode of the verify, and the wait for its readback (which is where the
    /// other sequences' work overlaps). Zero on the serial round.
    pub verify_enqueue_ns: u64,
    pub verify_wait_ns: u64,
    /// Host readbacks (each one a wait for the GPU) per phase, from the Metal
    /// backend's `to_cpu` counters; 0 off Metal. The speculative round owes
    /// exactly one, in the verify — the count is what shows a draft that
    /// drained its ids to the host, since the digests cannot.
    pub draft_readbacks: u64,
    pub verify_readbacks: u64,
    pub observe_readbacks: u64,
    pub draft_gpu_s: f64,
    pub verify_gpu_s: f64,
    pub observe_gpu_s: f64,
    pub draft_disp: u64,
    pub verify_disp: u64,
    pub observe_disp: u64,
    pub draft_census: Option<PhaseCensus>,
    pub verify_census: Option<PhaseCensus>,
    pub observe_census: Option<PhaseCensus>,
}

impl RoundTiming {
    /// The round's wall, the three phases summed.
    pub fn wall_ns(&self) -> u64 {
        self.draft_ns + self.verify_ns + self.observe_ns
    }

    /// Host readbacks the round performed, the three phases summed.
    pub fn readbacks(&self) -> u64 {
        self.draft_readbacks + self.verify_readbacks + self.observe_readbacks
    }
}

/// Host readbacks performed so far by this process (Metal's two `to_cpu`
/// paths summed); 0 on other backends.
pub(crate) fn readbacks() -> u64 {
    #[cfg(feature = "metal")]
    {
        let (fast, blit) = lloom_tensor::metal_backend::to_cpu_counts();
        fast + blit + lloom_tensor::metal_backend::to_cpu_staged_count()
    }
    #[cfg(not(feature = "metal"))]
    {
        0
    }
}

/// One sequence: a cache, the token ids whose state it holds, and how far it
/// has been computed (issue #677, `design/docs/sequence-reuse-design.md`).
///
/// This is the unit both mature engines converge on — llama.cpp's slot
/// (`server-context.cpp:100`, cached tokens beside the KV) and vLLM's request
/// (`request.py`, `num_computed_tokens` beside the block table): **cached state
/// keyed by token content**, with no request identity in it. A continuation is
/// a prompt sharing this sequence's whole token log as a prefix; a divergent
/// request shares part of it; a fresh one shares none. [`Session::prepare`] is
/// that comparison, and it is the only place the three cases are told apart.
///
/// Separable from the [`Session`] that drives it so a serving layer can hold
/// sequences across requests without holding a borrow of the model: build a
/// `Session` from a sequence for one request and take it back afterwards
/// ([`Session::from_sequence`], [`Session::into_sequence`]).
pub struct Sequence<C> {
    pub(crate) cache: C,
    /// The ids at cache positions `0..pos - overrun`, in order.
    pub(crate) tokens: Vec<u32>,
    /// Positions in the cache; the next forward pass starts here.
    pub(crate) pos: usize,
    /// Positions past `tokens.len()` that a loop consumed after the sequence's
    /// last kept token — the forwards an asynchronous loop enqueued before it
    /// learned EOS. Discarded by [`Session::reconcile`]; a decode is refused
    /// while it is non-zero.
    pub(crate) overrun: usize,
    /// The overrun's forwards launched nothing on the device (lloom #735's
    /// EOS gate): `reconcile` moves the counters back
    /// (`CacheRewind::rewind_unwritten`) instead of rewinding contents.
    pub(crate) overrun_unwritten: bool,
    /// The logits of the last fed position, when the run ended with the cache
    /// exactly one prediction ahead of the token log and nothing overran — the
    /// state a continuation resumes from without re-feeding anything.
    pub(crate) last_logits: Option<Tensor>,
    /// Memory this sequence owns for `last_logits` when the device serves
    /// activations from a reusable base (`Session::hold_logits`): allocated
    /// once, outside every pass, and written by one copy per hold.
    pub(crate) logits_hold: Option<Tensor>,
    /// `false` for a cache prefilled by a caller that did not say with what
    /// ([`Session::with_pos`]); such a sequence cannot be matched, only reset.
    pub(crate) tokens_known: bool,
}

impl<C> Sequence<C> {
    /// A fresh sequence over an empty cache.
    pub fn new(cache: C) -> Self {
        Self {
            cache,
            tokens: Vec::new(),
            pos: 0,
            overrun: 0,
            overrun_unwritten: false,
            last_logits: None,
            logits_hold: None,
            tokens_known: true,
        }
    }

    /// A sequence over a cache the caller has already filled with exactly
    /// `tokens` — the state a serving layer restores, or a test builds.
    pub fn with_tokens(cache: C, tokens: Vec<u32>) -> Self {
        let pos = tokens.len();
        Self {
            cache,
            tokens,
            pos,
            overrun: 0,
            overrun_unwritten: false,
            last_logits: None,
            logits_hold: None,
            tokens_known: true,
        }
    }

    /// The ids whose state the cache holds, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Positions in the cache, the unreconciled overrun included.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Positions past the token log a loop consumed and nothing has discarded.
    pub fn overrun(&self) -> usize {
        self.overrun
    }

    /// Whether the token log describes the cache (false after [`Session::with_pos`]).
    pub fn tokens_known(&self) -> bool {
        self.tokens_known
    }

    /// The cache.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// The cache, mutably.
    pub fn cache_mut(&mut self) -> &mut C {
        &mut self.cache
    }

    /// The length of the longest common prefix of the token log and `full`.
    pub fn common_prefix(&self, full: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(full.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// What [`Session::prepare`] did to the sequence to make `full` decodable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseAction {
    /// The whole token log is a prefix of `full`; only the suffix is prefilled.
    Extend,
    /// `full` diverged inside the rewind horizon: the cache was rewound by the
    /// carried count and the suffix from the divergence is prefilled.
    Rewound(usize),
    /// `full` is exactly the token log and the last prediction is still held:
    /// nothing is prefilled and the decode resumes from those logits.
    Resume,
    /// Nothing reusable — no overlap, a divergence past the horizon, or a cache
    /// whose contents were unknown — so the cache was reset and `full` is
    /// prefilled whole.
    Fresh,
    /// The cache was reset, then the carried count of tokens was SHARED from
    /// another row's cache — its pages by reference, its conv state restored
    /// (`CacheShare`) — and the suffix after them is prefilled.
    Shared(usize),
}

/// The outcome of [`Session::prepare`]: what to prefill and what was reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    pub action: ReuseAction,
    /// Tokens of `full` whose state the cache already held and kept.
    pub reused: usize,
    /// The tokens to feed to [`Session::decode`] as its prompt. Empty only for
    /// [`ReuseAction::Resume`], which `decode` accepts as "continue from the
    /// held logits".
    pub suffix: Vec<u32>,
}

/// A target holding its own cache, driving speculative rounds. The typed pair
/// (target, drafter) is checked where [`Session::speculative_step`] is
/// instantiated — `M: Probes<S::Probe>` there — and the driver names neither
/// family nor probe shape.
///
/// Constructed with a cache the caller already built and prefilled
/// ([`CausalLM::new_cache`]); the model is borrowed rather than owned so a
/// session pins no weights alive beyond the round it drives. The cache, the
/// token log and the position travel together as a [`Sequence`].
pub struct Session<'m, M: CausalLM> {
    pub(crate) model: &'m M,
    pub(crate) seq: Sequence<M::Cache>,
    /// The in-flight depth measured at warmup (`0` until `warmup` runs) — the
    /// computed host/GPU balance the pacer uses instead of a static cap.
    pub(crate) warm_in_flight_depth: usize,
    /// The `(host_ns, gpu_ns)` the depth above was computed from — kept so a
    /// test can check the two are DISJOINT (`Session::measured_balance_ns`).
    pub(crate) warm_balance_ns: (u64, u64),
}

impl<'m, M: CausalLM> Session<'m, M> {
    /// Pair a target with a cache it has already built. The cache is taken as
    /// empty; a caller that prefilled it supplies the length with
    /// [`Self::with_pos`].
    pub fn new(model: &'m M, cache: M::Cache) -> Self {
        Self {
            model,
            seq: Sequence::new(cache),
            warm_in_flight_depth: 0,
            warm_balance_ns: (0, 0),
        }
    }

    /// As [`Self::new`], with `pos` positions already prefilled into `cache` by
    /// the caller. What they hold is not known here, so the sequence cannot be
    /// prefix-matched until it is reset; continuation by [`Self::decode`] works
    /// as before.
    pub fn with_pos(model: &'m M, cache: M::Cache, pos: usize) -> Self {
        let mut seq = Sequence::new(cache);
        seq.pos = pos;
        seq.tokens_known = pos == 0;
        Self {
            model,
            seq,
            warm_in_flight_depth: 0,
            warm_balance_ns: (0, 0),
        }
    }

    /// Drive an existing sequence — a slot a serving layer held across requests.
    pub fn from_sequence(model: &'m M, seq: Sequence<M::Cache>) -> Self {
        Self {
            model,
            seq,
            warm_in_flight_depth: 0,
            warm_balance_ns: (0, 0),
        }
    }

    /// Give the sequence back, for a caller that holds it across requests.
    pub fn into_sequence(self) -> Sequence<M::Cache> {
        self.seq
    }

    /// The sequence this session drives.
    pub fn sequence(&self) -> &Sequence<M::Cache> {
        &self.seq
    }

    /// Positions currently in the cache.
    pub fn pos(&self) -> usize {
        self.seq.pos
    }

    /// The ids whose state the cache holds.
    pub fn tokens(&self) -> &[u32] {
        &self.seq.tokens
    }

    /// Positions past the token log that a loop consumed and nothing has yet
    /// discarded — see [`Sequence::overrun`].
    pub fn overrun(&self) -> usize {
        self.seq.overrun
    }

    /// The target, for a caller composing a second seam over the same model.
    pub fn model(&self) -> &'m M {
        self.model
    }

    /// Discard the overrun, so the cache ends at the token log's last id.
    ///
    /// Returns how many positions were discarded. Refused when the cache's
    /// rewind horizon is shorter than the overrun, naming the count; a caller
    /// that can accept losing the sequence resets it instead ([`Self::reset`]).
    pub fn reconcile(&mut self) -> Result<usize>
    where
        M::Cache: CacheRewind,
    {
        let n = self.seq.overrun;
        if n == 0 {
            return Ok(0);
        }
        if self.seq.overrun_unwritten {
            // The gated forwards wrote nothing: counters only, no horizon.
            self.seq.cache.rewind_unwritten(n)?;
        } else {
            self.seq.cache.rewind(n)?;
        }
        self.seq.pos -= n;
        self.seq.overrun = 0;
        self.seq.overrun_unwritten = false;
        Ok(n)
    }

    /// Whether the pending overrun's forwards launched nothing on the device
    /// (lloom #735's EOS gate), so [`Self::reconcile`] needs no rewind
    /// horizon.
    pub fn overrun_unwritten(&self) -> bool {
        self.seq.overrun > 0 && self.seq.overrun_unwritten
    }

    /// Return the sequence to empty: the cache reset, the token log cleared.
    pub fn reset(&mut self)
    where
        M::Cache: CacheReset,
    {
        self.seq.cache.reset();
        self.seq.tokens.clear();
        self.seq.pos = 0;
        self.seq.overrun = 0;
        self.seq.overrun_unwritten = false;
        self.seq.last_logits = None;
        self.seq.tokens_known = true;
    }

    /// [`Self::prepare`], and when it finds nothing of its own to reuse, a
    /// share of `src`'s cached prefix: the longest common prefix of `src_log`
    /// (what `src` holds, token for token) and `full`, rounded down to what
    /// [`CacheShare::share_prefix_from`] can give (whole pages, at most
    /// `full.len() - 1` so a prediction is still produced), the token log and
    /// position set past it. `src` must hold at least `src_log.len()` tokens.
    pub fn prepare_shared(
        &mut self,
        full: &[u32],
        src: &M::Cache,
        src_log: &[u32],
    ) -> Result<Prepared>
    where
        M::Cache: CacheRewind + CacheReset + CacheShare,
    {
        let prepared = self.prepare(full)?;
        if prepared.action != ReuseAction::Fresh || full.len() < 2 {
            return Ok(prepared);
        }
        let lcp = src_log
            .iter()
            .zip(full.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let want = lcp.min(full.len() - 1);
        if want == 0 {
            return Ok(prepared);
        }
        let shared = self.seq.cache.share_prefix_from(src, want)?;
        if shared == 0 {
            return Ok(prepared);
        }
        self.seq.tokens = full[..shared].to_vec();
        self.seq.pos = shared;
        self.seq.tokens_known = true;
        self.seq.last_logits = None;
        Ok(Prepared {
            action: ReuseAction::Shared(shared),
            reused: shared,
            suffix: full[shared..].to_vec(),
        })
    }

    /// Make the sequence decodable for `full` — the whole token sequence a
    /// request wants continued — reusing whatever prefix the cache already
    /// holds, and say what is left to prefill.
    ///
    /// The comparison is content only: the longest common prefix of the token
    /// log and `full` (llama.cpp's `get_common_prefix`, `server-context.cpp:1584`).
    /// A continuation of this sequence, a request diverging from it, and a
    /// fresh request are the same operation at different prefix lengths —
    /// nothing here knows which client sent `full`.
    ///
    /// An unreconciled overrun is discarded first, or the sequence is reset if
    /// it cannot be. When `full` is entirely cached, the last id must still be
    /// re-fed to produce a prediction unless the run that cached it left its
    /// logits behind ([`ReuseAction::Resume`]).
    pub fn prepare(&mut self, full: &[u32]) -> Result<Prepared>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        if full.is_empty() {
            lloom_tensor::bail!("Session::prepare: empty sequence");
        }
        if self.seq.overrun > 0 && self.reconcile().is_err() {
            self.reset();
        }
        if !self.seq.tokens_known {
            self.reset();
        }
        let cached = self.seq.tokens.len();
        if cached == 0 {
            return Ok(Prepared {
                action: ReuseAction::Fresh,
                reused: 0,
                suffix: full.to_vec(),
            });
        }
        let lcp = self.seq.common_prefix(full);

        if lcp == full.len() && lcp == cached && self.seq.last_logits.is_some() {
            return Ok(Prepared {
                action: ReuseAction::Resume,
                reused: lcp,
                suffix: Vec::new(),
            });
        }
        // Everything cached: the last token must be re-fed for its prediction.
        let keep = if lcp == full.len() { lcp - 1 } else { lcp };
        let drop = cached - keep;
        let action = if drop == 0 {
            ReuseAction::Extend
        } else if drop <= self.seq.cache.rewind_horizon() {
            self.seq.cache.rewind(drop)?;
            self.seq.pos -= drop;
            self.seq.tokens.truncate(keep);
            self.seq.last_logits = None;
            ReuseAction::Rewound(drop)
        } else {
            self.reset();
            ReuseAction::Fresh
        };
        let reused = match action {
            ReuseAction::Fresh => 0,
            _ => keep,
        };
        Ok(Prepared {
            action,
            reused,
            suffix: full[reused..].to_vec(),
        })
    }

    /// Drive one speculative round across the active sequences.
    ///
    /// `S: Speculator` supplies the family; `M: Probes<S::Probe>` is the pairing
    /// bound that makes an unsatisfiable probe a compile error. The driver owns
    /// the `p_min` early-exit (§3.4.4): `p_min` is one shared config knob, not a
    /// per-family concept — every family that early-exits does the same
    /// comparison and only the *source* of the probability differs, so there is
    /// no `trait EarlyExit`; the family supplies `Draft.confidence` and the
    /// driver truncates.
    pub fn speculative_step<S>(
        &mut self,
        spec: &mut S,
        active: &[SeqId],
        anchors: &[Anchor],
        p_min: f32,
        eos_ids: &[u32],
    ) -> Result<Vec<Vec<u32>>>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        self.speculative_round(spec, active, anchors, p_min, eos_ids, RoundProbe::default())
            .map(|(accepted, _)| accepted)
    }

    /// [`Self::speculative_step`] with the round's three phases bracketed —
    /// draft, verify (the target's `forward_all` up to and including the
    /// accept compare's readback) and observe (the drafter's re-key). The
    /// decode loop's per-round breakdown (`DecodeStats::rounds`) is this.
    ///
    /// Without a `probe.sync` device the brackets are the host's view: the
    /// draft ends when the drafter has its tokens (it reads them back), the
    /// verify ends when the argmax has been read back, and the observe is its
    /// encode only. With `probe.sync` each phase ends on a `synchronize`, so a
    /// phase's GPU tail is charged to that phase and not to the next one's
    /// readback; and when the Metal profiler is on the per-phase GPU-busy
    /// union and dispatch count are taken too (the counters are reset per
    /// phase — a caller reading them itself must accumulate from here).
    /// The synchronize perturbs the round it measures, exactly as
    /// `DecodeOptions::sample_split` does for the baseline loop.
    pub fn speculative_round<S>(
        &mut self,
        spec: &mut S,
        active: &[SeqId],
        anchors: &[Anchor],
        p_min: f32,
        eos_ids: &[u32],
        probe: RoundProbe<'_>,
    ) -> Result<(Vec<Vec<u32>>, RoundTiming)>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        self.speculative_round_keyed(spec, active, anchors, p_min, eos_ids, probe, None)
    }

    /// [`Self::speculative_round`] with the VERIFY phase bracketed as a
    /// planned pass of `verify_key`'s shape (`DESIGN.md` §9.2aa).
    ///
    /// Only the verify is bracketed. The draft and the observe run on the
    /// DRAFTER's model and its own caches — a different allocation sequence
    /// that the target's key would not describe — and the drafter's taps are
    /// declared escapes at their producer (`lfm2.rs`, `tap_layer_input`), so
    /// the verify's plan never serves the values `observe` reads.
    #[allow(clippy::too_many_arguments)]
    pub fn speculative_round_keyed<S>(
        &mut self,
        spec: &mut S,
        active: &[SeqId],
        anchors: &[Anchor],
        p_min: f32,
        eos_ids: &[u32],
        probe: RoundProbe<'_>,
        verify_key: Option<(&Device, crate::nn::decode::PassShape)>,
    ) -> Result<(Vec<Vec<u32>>, RoundTiming)>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        let mut timing = RoundTiming::default();
        let t = std::time::Instant::now();
        let rb = readbacks();
        // The hint is produced only when this round will read it (§3.4.4: the
        // driver owns `p_min`; the family only learns whether it is consumed).
        spec.want_confidence(p_min > 0.0);
        let mut drafts = spec.draft(active)?;
        let (gpu, disp, census) = probe.close()?;
        timing.draft_ns = t.elapsed().as_nanos() as u64;
        timing.draft_readbacks = readbacks() - rb;
        timing.draft_gpu_s = gpu;
        timing.draft_disp = disp;
        timing.draft_census = census;

        // Driver-owned `p_min` truncation. `--spec-draft-p-min` is off at 0.0
        // (`speculative.cpp:796`), and a family with no confidence signal leaves
        // `confidence` empty, so `position` finds nothing and the whole block
        // stands — uniform across families (§3.4.4).
        if p_min > 0.0 {
            for d in drafts.iter_mut() {
                if let Some(cut) = d.confidence.iter().position(|&p| p < p_min) {
                    d.tokens.truncate(cut)?;
                    d.confidence.truncate(cut);
                }
            }
        }

        // ONE target forward over `[last, draft…]`, extracting the probe in the
        // SAME pass (§10.2r amortization). The round's one host readback is in
        // here: the verify's argmax, with the draft ids beside it.
        let t = std::time::Instant::now();
        let rb = readbacks();
        if let Some((device, shape)) = verify_key {
            crate::nn::decode::begin_pass(device, shape)?;
        }
        let verified = self
            .model
            .verify_and_probe(&drafts, anchors, &mut self.seq.cache, eos_ids);
        if let Some((device, _)) = verify_key {
            crate::nn::decode::end_pass(device)?;
        }
        let (accepted, probe_out) = verified?;
        let (gpu, disp, census) = probe.close()?;
        timing.verify_ns = t.elapsed().as_nanos() as u64;
        timing.verify_readbacks = readbacks() - rb;
        timing.verify_gpu_s = gpu;
        timing.verify_disp = disp;
        timing.verify_census = census;

        let t = std::time::Instant::now();
        let rb = readbacks();
        for (seq, acc) in active.iter().zip(&accepted) {
            spec.observe(*seq, &probe_out, acc)?;
        }
        let (gpu, disp, census) = probe.close()?;
        timing.observe_ns = t.elapsed().as_nanos() as u64;
        timing.observe_readbacks = readbacks() - rb;
        timing.observe_gpu_s = gpu;
        timing.observe_disp = disp;
        timing.observe_census = census;
        Ok((accepted, timing))
    }

    /// The target's cache, for a caller that prefills before the first round.
    pub fn cache_mut(&mut self) -> &mut M::Cache {
        &mut self.seq.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seam composes: LFM2 as the target and `DSparkSpeculator` as the
    /// family type-check through `Session::speculative_step`'s bounds
    /// (`M: Probes<S::Probe>` with `S::Probe = LayerTaps<5>`), which is the
    /// §3.4.3 pairing proof. Reachability the way §3.1c asks — the production
    /// decode loop is Wave F's (`verify_step` has no production caller yet), so
    /// a generic bound is what proves the surface is callable, not an end-to-end
    /// run. The digest gate is blind to drafter quality by construction
    /// (greedy verify accepts iff the token equals the target's argmax), so the
    /// seam's correctness is a *type* property here, not a numeric one.
    #[allow(dead_code)]
    fn assert_pairs<M, S>()
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
    }

    #[test]
    fn dspark_pairs_with_lfm2() {
        // Only instantiating the bound matters — if `Model` did not satisfy
        // `Probes<LayerTaps<5>>` (DSpark's probe), this would not compile.
        assert_pairs::<crate::models::lfm2::Model, crate::models::lfm2_dspark::DSparkSpeculator>();

        // The reference tap side is `LayerOutput` — `target_layer_ids + 1`,
        // which `DSparkConfig::tap_layers()` already bakes in (so `--tap-shift 0`
        // is the reference, NOT shift 1 as the issue body's gate line claimed).
        assert_eq!(
            <LayerTaps<5> as TargetProbe>::hooks(),
            ProbeHooks::Taps(TapSpec {
                layers: &[],
                side: TapSide::LayerOutput,
                dtype: DType::F32,
            })
        );
    }
}
