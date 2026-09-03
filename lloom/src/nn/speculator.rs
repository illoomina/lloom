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

use crate::nn::model::CausalLM;
use lloom_tensor::{DType, Result};

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

/// One drafted block for one sequence.
pub struct Draft {
    pub tokens: Vec<u32>,
    /// Per-position `P(this token)` in `[0, 1]` if the family exposes a
    /// confidence signal (DSpark's confidence head), EMPTY otherwise. NOT a
    /// probability the target must honour — a hint. The DRIVER owns the
    /// `p_min` threshold ([`Session::speculative_step`]); the family never
    /// sees it. One knob, many families.
    pub confidence: Vec<f32>,
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
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
    ) -> Result<(Vec<Vec<u32>>, P::Output)>;
}

/// A target holding its own cache, driving speculative rounds. The typed pair
/// (target, drafter) is checked where [`Session::speculative_step`] is
/// instantiated — `M: Probes<S::Probe>` there — and the driver names neither
/// family nor probe shape.
///
/// Constructed with a cache the caller already built and prefilled
/// ([`CausalLM::new_cache`]); the model is borrowed rather than owned so a
/// session pins no weights alive beyond the round it drives.
pub struct Session<'m, M: CausalLM> {
    model: &'m M,
    cache: M::Cache,
}

impl<'m, M: CausalLM> Session<'m, M> {
    /// Pair a target with a cache it has already built.
    pub fn new(model: &'m M, cache: M::Cache) -> Self {
        Self { model, cache }
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
    ) -> Result<Vec<Vec<u32>>>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        let mut drafts = spec.draft(active)?;

        // Driver-owned `p_min` truncation. `--spec-draft-p-min` is off at 0.0
        // (`speculative.cpp:796`), and a family with no confidence signal leaves
        // `confidence` empty, so `position` finds nothing and the whole block
        // stands — uniform across families (§3.4.4).
        if p_min > 0.0 {
            for d in drafts.iter_mut() {
                if let Some(cut) = d.confidence.iter().position(|&p| p < p_min) {
                    d.tokens.truncate(cut);
                    d.confidence.truncate(cut);
                }
            }
        }

        // ONE target forward over `[last, draft…]`, extracting the probe in the
        // SAME pass (§10.2r amortization).
        let (accepted, probe_out) =
            self.model
                .verify_and_probe(&drafts, anchors, &mut self.cache)?;
        for (seq, acc) in active.iter().zip(&accepted) {
            spec.observe(*seq, &probe_out, acc)?;
        }
        Ok(accepted)
    }

    /// The target's cache, for a caller that prefills before the first round.
    pub fn cache_mut(&mut self) -> &mut M::Cache {
        &mut self.cache
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
