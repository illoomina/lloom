//! The memory budget: predict the peak before allocating, and refuse.
//!
//! `DESIGN.md` §9.5, issue #186. This is §9.5d's admission check and §9.5k's
//! derived residual, and it is the whole of what those sections specify.
//!
//! # Why this exists, in one paragraph
//!
//! **Nothing in this engine bounded the sum of §9.1's five memory classes, and
//! the failure mode was a kernel panic — twice.** §6.3b's `set_free_budget`
//! bounds one class's *free list*; §6.2b's `DEFAULT_KV_CAPACITY` bounds one
//! class's *per-sequence reserve*. There was no mechanism above either, and
//! none anywhere that knew what the five classes sum to. The second panic
//! (§9.5a) died **during allocation** — 0.60 s user against 34.77 s system,
//! 692 467 page faults against **144 page-ins**, which is a process faulting in
//! fresh anonymous pages rather than reading a model from disk — with 60 MB
//! free and 54.91 GB wired, and with #166/#167's residency guard compiled in.
//! **That is the failure admission prevents**, and it is a different failure
//! from the one the guard closed.
//!
//! # The organizing idea: the unpredictable class is the residual
//!
//! Four terms are exactly known before anything is allocated — weights from the
//! checkpoint header (§5.5), KV from `B × capacity × 16 KiB` (§5.6's geometry,
//! *not* its table — see [`KV_BYTES_PER_TOKEN`]), conv state from §5.7, and
//! scratch from §9.1a's formula under the selected policy. What is left over is
//! a **derived** cap on everything else rather than an estimate of it:
//!
//! ```text
//! residual = budget − (weights + KV + conv + activations + scratch)
//! ```
//!
//! So the one class that grows on demand acquires a ceiling **by construction**,
//! needing no new measurement. That is what lets admission report *"this
//! configuration leaves N GB for intermediates"* instead of discovering the
//! answer by dying.
//!
//! # What this module does not close
//!
//! **§9.5f, stated here because it would be easy to over-read this code.** Every
//! reachable B=1 configuration predicts under 8.71 GB and the machine died with
//! 54.91 GB wired, so **admission as specified would have refused nothing** on
//! the run that crashed. It is necessary and it is not sufficient. §16 P0 #7 is
//! the open question, and the measurement it wants is #171's memory-by-class
//! timeline at long context — which has never been run.
//!
//! What the residual adds is a *boundary* for the second of §9.5f's three
//! candidate explanations: an allocation path growing without bound is still
//! invisible to admission, but crossing the derived ceiling is now a refusal at
//! a known site rather than a panic at an arbitrary one.

use super::buffer_pool::PoolOccupancySnapshot;
use super::scratch::{PartialsGeometry, Sizing};

/// KV bytes per token across all attention layers, from the geometry.
///
/// `2 (K,V) × 8 kv_heads × 64 head_dim × 2 B × 8 attention layers` = **16 384**.
///
/// **This is §5.6's *geometry*, not §5.6's table.** That table was computed with
/// **16 000** B/token and is therefore low by 2.4 % — `0.524288 =
/// 16000 × 32768 / 1e9` exactly, which is what identifies the arithmetic rather
/// than leaving it as a rounding question (§9.5b; the table is corrected in
/// place). The error is 50 MB at B=1 × 128k and **1.61 GB at B=32 × 128k**:
/// negligible where the table is usually quoted, and not negligible where a
/// memory budget uses it, which is how it was found. **A budget built by quoting
/// the table would under-predict in exactly the regime where the margin
/// matters**, so this constant is derived from primitives below and asserted
/// against them by `kv_geometry_is_16384_not_16000`.
pub const KV_BYTES_PER_TOKEN: usize = 2 * 8 * 64 * 2 * 8;

/// Conv state across all 22 conv layers, per sequence: `22 × 2048 × 3 × 2 B`.
///
/// §5.7, and §10.2g records that `ConvState::RotatingRing` is also this figure
/// where the sliding ring at `slack = 16` is 1.63 MiB. The larger arm is a
/// caller-supplied override rather than the default, for the reason [`Budget`]
/// gives: this crate cannot see `lfm2::Config`.
pub const CONV_STATE_BYTES: usize = 22 * 2048 * 3 * 2;

/// The activation arena at B=1 **and `seq_len == 1`**, measured by #68 (§9.2c).
///
/// Not an estimate: 68.00 KB (69 632 B) packed, against a maximum simultaneous
/// liveness of 67.00 KB. §5.9's "tens of KB" was right and nobody had turned it
/// into a verdict.
///
/// It is also the one term that is genuinely *outside* the pools —
/// `install_arena` calls the raw device (`device.rs:485`) where weights, KV and
/// scratch all go through `acquire` (§9.5k). That asymmetry is what
/// [`unplanned_bytes`] turns on.
///
/// # This figure is `seq_len == 1`'s and does not generalise — #306
///
/// **§13.4b flags it as *"a B=1 measurement that this deliberately does not
/// exercise"*, and the same caution applies along `seq_len` with far more
/// force.** At `seq_len = 2048` the activation class is not a stale 68 KB, it
/// is **a different quantity by five orders of magnitude** — [`prefill_activation_bytes`]
/// computes 18.66 GB — because the generic attention arm's score matrix is
/// `[B, 32, P, P]` and quadratic in `P` where every decode activation is
/// constant.
///
/// Summing this constant for a prefill is what let admission admit a
/// configuration that peaked at **30.5 GB with 77 MB of host memory free**
/// (§13.4c) — §9.5f's *"the term that overruns is not one of the five it
/// sums"*, except that it **is** one of the five and it was being counted at
/// its decode value.
pub const ACTIVATION_ARENA_BYTES: usize = 69_632;

/// The default fraction of `recommendedMaxWorkingSetSize` a process may claim.
///
/// **The denominator is `recommendedMaxWorkingSetSize` and not `hw.memsize`**,
/// per §9.5c, and the choice is argued rather than defaulted. Three reasons:
/// it is the only candidate that is *about* GPU-visible memory, which is what
/// §6.3c's three-participant lifetime finding makes the relevant quantity; it
/// is **already read** (`device.rs:225`), so this needs no new capability; and
/// it degrades honestly on a machine that is not this one, where a hardcoded
/// byte count does not.
///
/// The two rejected candidates were rejected **by measurement, not by
/// preference**: free-page sampling reads a quantity every other process is
/// moving (§3.4a-ii measured `python3.13` moving **13.0 GB during a 5.4 s
/// run**), and `memoryPressure` read **False at both panics** (§9.5a) — the one
/// boolean an engine might have consulted was false when the machine died,
/// twice.
///
/// 0.65 is the figure §9.5k's table is computed at. It is a **default and not a
/// finding**: no measurement chose it, and it is a caller-settable field on
/// [`Budget`] for that reason.
pub const DEFAULT_BUDGET_FRACTION: f64 = 0.65;

/// §6.3b's measured stranding: 5231 → 13629 MB accumulated over 400 tokens.
///
/// **The yardstick a residual is graded against, and it is measured rather than
/// round.** §9.5k: a residual below this is one that a *known and already
/// observed* allocation shape can exhaust, which is what deriving the cap buys
/// — "is this configuration safe" becomes a comparison against something
/// observed rather than against a number someone picked.
///
/// It is the largest unbounded term this project has seen, and it is
/// `pending_bytes`: 11.6 buffers per token, ~21 MB/token, capped by nothing.
/// `free_bytes` is capped by `set_free_budget`; `pending_bytes` is not.
pub const OBSERVED_STRANDING_BYTES: usize = (13_629 - 5_231) * 1_000_000;

/// LFM2's model geometry, for the one class whose size depends on it.
///
/// # Why this type exists in a crate that deliberately knows nothing about LFM2
///
/// [`Budget`]'s own doc comment says this crate is below `candle-transformers`
/// and must not see `lfm2::Config`, and that stands: this is **numbers the
/// caller resolves and passes**, exactly as `conv_state_bytes` and
/// [`PartialsGeometry`] already are (§15.2 #8 at the crate boundary). What is
/// new is that the *activation* class needs geometry at all, and it needs it
/// only because the term is quadratic in `seq_len` (see
/// [`prefill_activation_bytes`]).
///
/// [`Default`] is LFM2's, from §5.2 — so a caller that does not care gets the
/// model this project measures, and a caller with a different one says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillGeometry {
    /// §5.3: 8 of 30 for LFM2, from `layer_types` rather than hardcoded indices.
    pub attention_layers: usize,
    /// §5.3: the other 22.
    pub conv_layers: usize,
    /// Query heads — **32**, not the 8 KV heads. The score matrix is formed
    /// after `repeat_kv`'s expansion, so the KV-head count never appears; using
    /// it would under-size this class **4×**, which is §9.1a's own warning
    /// about the scratch class in a second place.
    pub query_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    /// §5.2's stated 10752, **not** candle's recomputed 8192 (§5.2's trap).
    pub intermediate: usize,
}

impl Default for PrefillGeometry {
    fn default() -> Self {
        // §5.2 and §5.3, read from the checkpoint on 2026-08-23.
        Self {
            attention_layers: 8,
            conv_layers: 22,
            query_heads: 32,
            head_dim: 64,
            hidden: 2048,
            intermediate: 10752,
        }
    }
}

impl PrefillGeometry {
    fn layers(&self) -> usize {
        self.attention_layers + self.conv_layers
    }
}

/// The activation class at `seq_len > 1`, which is the term that overruns.
///
/// # What this counts, and it is read from source rather than estimated
///
/// **Every prefill takes the generic attention arm.** `sdpa_applies`
/// (`lfm2.rs:1383`) and `flash_decoding_applies` (`:1399`) both require
/// `seq_len == 1`, and §13.4c verified the consequence by a **byte-identical
/// kernel census across all three `--attn` arms** rather than inferring it from
/// the predicates. So this arithmetic is over one arm, not three.
///
/// Per attention layer, from `Attention::forward`'s final `else`:
///
/// | tensor | shape | dtype |
/// |---|---|---|
/// | `repeat_kv(k)`, `repeat_kv(v)` | `[B, 32, P, 64]` | f16 |
/// | `q`/`k`/`v` upcast | `[B, 32, P, 64]` | f32 |
/// | **scores** | `[B, 32, P, P]` | **f32** |
/// | **masked scores** | `[B, 32, P, P]` | **f32** |
/// | **softmax output** | `[B, 32, P, P]` | **f32** |
/// | attention output | `[B, 32, P, 64]` | f16 |
///
/// **Three `[B, 32, P, P]` f32 tensors are live, not one**, and that is where
/// §13.2b's single-tensor arithmetic loses its factor: `masked_fill` is
/// `mask.where_cond(..)` (`lfm2.rs:1254`), which writes a **new** tensor rather
/// than editing in place, and `softmax_last_dim` is a `CustomOp1`
/// (`candle-nn/src/ops.rs:437`), which allocates its own output.
///
/// # Why it sums over layers rather than taking the maximum
///
/// **§6.3b: the pool defers a buffer's return to the free list until the
/// command buffer holding its last use completes**, and a prefill pass is
/// **14 command buffers for 660 dispatches** (#250's RESULT line). So a layer's
/// intermediates are not reusable by the next layer — they are retained until
/// the command buffer retires. §6.3b measured this shape at decode as *"11.6
/// stranded buffers per token"*; at prefill the stranded unit is a whole
/// layer's activations and the quantity is **quadratic in `P`**.
///
/// # Reconciliation, at two lengths
///
/// Against #250's committed raw series, taking the measured *increment* over
/// the settled process footprint (the peak also carries the load-time f32
/// staging transient and the RoPE tables, which are flat in `P`):
///
/// | `P` | measured increment | predicted | ratio |
/// |---|---|---|---|
/// | 1536 | 12.28 GB | 11.57 GB | **0.94** |
/// | 2048 | 19.06 GB | 18.66 GB | **0.98** |
///
/// **Against §13.2b's 0.537 GB at `P` = 2048 — 35× low — this is the
/// correction.** Both bounds matter: a model that over-predicted everywhere
/// would refuse configurations that fit, which is §8.1g's failure.
///
/// # The direction that is dangerous
///
/// **Under-prediction is the silent direction** (§9.5m, and #247's contract
/// check preserves the same asymmetry in its API). §3.5 reports no overrun, so
/// an under-prediction is discovered by dying. Where this arithmetic is
/// uncertain it is uncertain *downward* — it counts named tensors and cannot
/// see a temporary candle allocates inside an op — which is why the reconciled
/// ratios above are below 1.0 rather than above, and why the margin is stated
/// rather than tuned away.
pub fn prefill_activation_bytes(seq_len: usize, batch: usize, geom: PrefillGeometry) -> usize {
    if seq_len <= 1 {
        // The decode case is #68's measured arena, not this arithmetic. At
        // `seq_len == 1` the score matrix is `[B, 32, 1, 1]` and every term
        // below collapses; returning the measured figure keeps the two regimes
        // from disagreeing at their shared point.
        return batch * ACTIVATION_ARENA_BYTES;
    }

    const F16: usize = 2;
    const F32: usize = 4;

    let p = seq_len;
    let heads = batch * geom.query_heads * p * geom.head_dim;
    let scores = batch * geom.query_heads * p * p;

    // One attention layer: the KV expansion, the upcast, three score matrices,
    // the causal mask and the output.
    let attn = 2 * heads * F16          // repeat_kv(k), repeat_kv(v)
        + 3 * heads * F32               // q, k, v upcast to f32
        + 3 * scores * F32              // scores, masked scores, softmax out
        + p * p                         // causal mask, u8
        + heads * F16; // attention output

    // One conv layer: `in_proj` to [B, P, 6144], then B/C/X and `bx`.
    let conv = batch * p * 3 * geom.hidden * F16 + 2 * batch * geom.hidden * p * F16;

    // Every layer's SwiGLU: gate, up and their product are live together
    // because `bmul` is not in place — §9.2c's own finding about the decode
    // arena's peak moment, at `seq_len = P` instead of 1.
    let mlp = 3 * batch * p * geom.intermediate * F16 + batch * p * geom.hidden * F16;

    geom.attention_layers * attn + geom.conv_layers * conv + geom.layers() * mlp
}

/// What one configuration is predicted to need, class by class.
///
/// Every field is bytes, and every field is computable **before the first
/// allocation** — which is the property that makes admission free (§9.5d). This
/// struct allocates nothing, reads no device state, and is a pure function of
/// the configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Footprint {
    /// §5.5, from the checkpoint header. A **fact**, with one policy lever:
    /// dropping the vision tower for a text-only request (#162).
    pub weights: usize,
    /// `B × capacity × 16 384 B`. A **policy** — the capacity is chosen — and
    /// the class that can consume the machine on its own: `B=32` at 128k is
    /// **68.719 GB of KV alone, exactly `hw.memsize`**, before a weight is
    /// loaded (§9.5b).
    pub kv: usize,
    /// `B × 264 KiB` (§5.7). A **fact** given `ConvState`.
    pub conv: usize,
    /// `B × 68 KB` (§9.2c). A **fact**, and negligible.
    pub activations: usize,
    /// §9.1a's formula under the selected policy. A **policy** —
    /// `Reserve`/`Grow`/`Bucket` — and **not** where the budget binds at any
    /// reachable configuration: 0.4 % of the total at B=1 × 128k against KV's
    /// 25 % (§9.5h).
    pub scratch: usize,
}

impl Footprint {
    /// The predicted peak: the five classes summed.
    ///
    /// **There is deliberately no `pool` row here**, and §9.5k is why. The pool
    /// is not a sixth class sitting beside the others at 0.27 GB — that figure
    /// is `set_free_budget`'s *free-list* cap, and the pool is where three of
    /// the five classes above are actually **served from**. Adding it as a row
    /// would double-count exactly what §9.5k had to read the allocation paths
    /// to rule out. What the caller needs instead is [`Admission::residual`].
    pub fn predicted(&self) -> usize {
        self.weights + self.kv + self.conv + self.activations + self.scratch
    }
}

/// The configuration admission is asked about.
///
/// # Why this type does not know about LFM2
///
/// It takes `weights` as a byte count rather than reading a checkpoint, and
/// `conv_state_bytes` as a number rather than a `ConvState`. `candle-metal-
/// kernels` is below `candle-transformers` and cannot see `lfm2::Config`;
/// putting model geometry here would be the conflation §10.1 renames the whole
/// `SequenceState` abstraction to avoid, one crate down. **Policy on the CPU,
/// numbers on the GPU** (§15.2 #8) applied to the crate boundary: the caller
/// resolves the policy and passes numbers.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// Concurrent sequences. Every measurement in this project is B=1 (§13.2).
    pub batch: usize,
    /// KV tokens reserved per sequence — `DEFAULT_KV_CAPACITY`'s value, not
    /// `max_context`, because the reserve is what is *allocated* (§6.2b).
    pub kv_capacity: usize,
    /// The context the scratch class is sized for under `Reserve`.
    pub max_context: usize,
    /// Total weight bytes, from the checkpoint header (§5.5). Text-only drops
    /// the 0.86 GB vision tower and projector (#162).
    pub weights: usize,
    /// Per-sequence conv state. [`CONV_STATE_BYTES`] for `Shuffle` and
    /// `RotatingRing`; larger for `SlidingRing` at slack (§10.2g).
    pub conv_state_bytes: usize,
    /// §9.1a's sizing policy for the scratch class.
    pub scratch_sizing: Sizing,
    /// The partials geometry scratch is sized from. `n_heads` is the **query**
    /// head count — using the 8 KV heads under-sizes the class **4×** (§9.1a).
    pub partials: PartialsGeometry,
    /// Fraction of `recommendedMaxWorkingSetSize` this process may claim.
    pub fraction: f64,
    /// **The longest single forward pass this configuration will run** — #306.
    ///
    /// Not `max_context` and not `kv_capacity`: those size the *session* state
    /// that persists across steps, where this sizes the *activations* live
    /// within one pass. §4.1's table is the distinction — `seq` is what enters
    /// one forward pass and `kv_len` is what the cache holds — and admission
    /// summed only the second until now.
    ///
    /// `1` is decode and is the default, because that is what every recorded
    /// measurement in this project runs (§13.2). A caller that will prefill a
    /// `P`-token prompt in one pass **must set this to `P`**, and a caller that
    /// chunks its prefill sets it to the **chunk size** — which is the whole
    /// mechanism by which chunking makes a long prompt admissible.
    pub max_seq_len: usize,
    /// Model geometry for the activation class. LFM2's by default (§5.2).
    pub prefill_geometry: PrefillGeometry,
}

impl Budget {
    /// A B=1 budget with the defaults every field's doc comment cites.
    pub fn new(weights: usize, kv_capacity: usize, max_context: usize) -> Self {
        Self {
            batch: 1,
            kv_capacity,
            max_context,
            weights,
            conv_state_bytes: CONV_STATE_BYTES,
            scratch_sizing: Sizing::Reserve,
            partials: PartialsGeometry::default(),
            fraction: DEFAULT_BUDGET_FRACTION,
            // Decode. **Not** `max_context`: defaulting this to the context
            // length would refuse nearly every configuration that ships, which
            // is §8.1g's failure — and a budget that refuses everything is
            // indistinguishable from one that is broken. A caller that
            // prefills says so; see [`Budget::with_prefill`].
            max_seq_len: 1,
            prefill_geometry: PrefillGeometry::default(),
        }
    }

    /// The same budget, asked about a single forward pass of `seq_len` tokens.
    ///
    /// This is the call a prefill owes, and the one whose absence let a
    /// 2048-token prompt be admitted at a predicted 8.4 GB and peak at 30.5
    /// (§13.4c). A **chunked** prefill passes its chunk size here rather than
    /// its prompt length, which is what makes a long prompt admissible at all
    /// (§13.2b).
    pub fn with_prefill(mut self, seq_len: usize) -> Self {
        self.max_seq_len = seq_len;
        self
    }

    /// The five classes, each from the section that states it.
    ///
    /// Pure arithmetic: six multiplies and some adds, no device call and no
    /// allocation. That is what makes admission **free by inspection** rather
    /// than by measurement (§9.5e) — it is not on any per-token path.
    pub fn footprint(&self) -> Footprint {
        let mut partials = self.partials;
        partials.batch = self.batch;
        // §9.1a's table is per-layer-summed: the class is 8 attention layers'
        // regions. `Reserve` sizes for max_context's chunk count and allocates
        // once; `Grow` and `Bucket` size to the live kv_len, which admission
        // cannot know — so both are predicted at their worst case, which is
        // `Reserve`'s figure. Predicting `Grow` at a *smaller* number would be
        // admitting a configuration on a footprint it only has early on.
        let chunks = partials.chunks(self.max_context);
        let scratch = N_ATTENTION_LAYERS * partials.partials_bytes(chunks);
        Footprint {
            weights: self.weights,
            kv: self.batch * self.kv_capacity * KV_BYTES_PER_TOKEN,
            conv: self.batch * self.conv_state_bytes,
            // #306: a function of `seq_len`, not a constant. At `max_seq_len
            // == 1` this is exactly #68's measured 68 KB × B, so no decode
            // configuration's verdict moves; above it the class is the
            // quadratic term §13.2b identifies and §13.4c measured.
            activations: prefill_activation_bytes(
                self.max_seq_len,
                self.batch,
                self.prefill_geometry,
            ),
            scratch,
        }
    }

    /// Evaluate this configuration against `working_set` bytes.
    ///
    /// `working_set` is `MTLDevice::recommendedMaxWorkingSetSize` (§9.5c),
    /// passed in rather than read here so the arithmetic stays a pure function
    /// and is testable without a GPU.
    pub fn admit(&self, working_set: usize) -> Admission {
        let footprint = self.footprint();
        let budget = (working_set as f64 * self.fraction) as usize;
        let predicted = footprint.predicted();
        Admission {
            footprint,
            budget,
            working_set,
            fraction: self.fraction,
            // Saturating rather than wrapping: a refused configuration has a
            // negative residual conceptually, and `residual_signed` reports it.
            residual: budget.saturating_sub(predicted),
            fits: predicted <= budget,
            max_seq_len: self.max_seq_len,
            prefill_geometry: self.prefill_geometry,
            batch: self.batch,
        }
    }
}

/// Attention layers holding KV. 8 of 30 for LFM2 (§5.3, from `layer_types`).
const N_ATTENTION_LAYERS: usize = 8;

/// The verdict, with the arithmetic that produced it.
///
/// **A boolean would not be enough**, per §9.5g: the caller sees which class
/// dominates, so the next configuration is a decision rather than a guess.
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub footprint: Footprint,
    /// `fraction × working_set`.
    pub budget: usize,
    /// `recommendedMaxWorkingSetSize`, as read from the device.
    pub working_set: usize,
    pub fraction: f64,
    /// `budget − predicted`, saturating at zero. **The derived cap on
    /// everything the five classes do not name** (§9.5k) — the pool, every
    /// intermediate no planner owns, and the RoPE tables (see
    /// [`Admission::describe`]).
    pub residual: usize,
    pub fits: bool,
    /// The forward-pass length this verdict is for (#306). Carried so a
    /// refusal can solve for the chunk size that would fit, and so a caller
    /// cannot read a decode verdict as a prefill one.
    pub max_seq_len: usize,
    /// Carried for the same reason: the chunk-size solve needs the geometry.
    pub prefill_geometry: PrefillGeometry,
    /// Batch, for the same solve.
    pub batch: usize,
}

impl Admission {
    /// `budget − predicted` as a signed quantity, so a refusal can report how
    /// far over it is rather than reporting zero.
    pub fn residual_signed(&self) -> i128 {
        self.budget as i128 - self.footprint.predicted() as i128
    }

    /// Whether the residual is smaller than a measured allocation shape can
    /// consume.
    ///
    /// **Graded against §6.3b's stranding rather than a round number**
    /// ([`OBSERVED_STRANDING_BYTES`]), which is the point of deriving the cap.
    /// `B=16 ctx 128k` passes the sum-based table at **60.3 % of the machine**
    /// and fails *this*, because 3.495 GB of residual is less than the 8.398 GB
    /// a known shape has already been measured accumulating over 400 tokens
    /// (§9.5k). A refusal that did not reflect that would admit a configuration
    /// a known allocation shape can kill.
    pub fn is_tight(&self) -> bool {
        self.fits && self.residual < OBSERVED_STRANDING_BYTES
    }

    /// The report §9.5g specifies, in the shape it specifies.
    ///
    /// Refusals carry the per-class breakdown and the reductions that would fit;
    /// admissions carry the residual, **which is the more useful half because it
    /// is the case that happens**.
    pub fn describe(&self) -> String {
        let gb = |b: usize| b as f64 / 1e9;
        let f = &self.footprint;
        if self.fits {
            let mut s = format!(
                "memory budget OK: predicted peak {:.2} GB, budget {:.2} GB\n  \
                 residual {:.2} GB   for pool, intermediates, and anything unplanned\n\
                 {:>24}(§6.3b's stranding is {:.2} GB over 400 tokens)",
                gb(f.predicted()),
                gb(self.budget),
                gb(self.residual),
                "",
                gb(OBSERVED_STRANDING_BYTES),
            );
            if self.is_tight() {
                s.push_str(
                    "\n  TIGHT: the residual is below a stranding figure this project \
                     has measured.",
                );
            }
            return s;
        }
        // A refusal. §9.5g: report the arithmetic, name the dominant class, and
        // list the reductions in order of what they cost -- so the next
        // configuration is a decision rather than a guess.
        let mut s = format!(
            "memory budget exceeded: predicted peak {:.2} GB > budget {:.2} GB\n  \
             ({:.2} x recommendedMaxWorkingSetSize = {:.2} GB)\n",
            gb(f.predicted()),
            gb(self.budget),
            self.fraction,
            gb(self.working_set),
        );
        let classes = [
            ("weights", f.weights),
            ("KV", f.kv),
            ("scratch", f.scratch),
            ("conv", f.conv),
            ("act", f.activations),
        ];
        let dominant = classes
            .iter()
            .max_by_key(|(_, b)| *b)
            .map(|(n, _)| *n)
            .unwrap_or("");
        for (name, bytes) in classes {
            let mark = if name == dominant {
                "   <- dominant"
            } else {
                ""
            };
            s.push_str(&format!("    {name:<9} {:>7.2} GB{mark}\n", gb(bytes)));
        }
        s.push_str(&format!(
            "  residual {:>7.2} GB   budget - predicted   <- NEGATIVE: nothing left \
             to allocate into\n",
            self.residual_signed() as f64 / 1e9,
        ));
        s.push_str("  reductions that would fit, in order of what they cost:\n");
        // #306: when the activation class dominates, the lever is the CHUNK
        // SIZE and not the context length -- and it is offered first, because
        // halving `max_context` does not touch a term that depends on
        // `seq_len`. §13.2b: chunked prefill "is not an optimization for the
        // long-prompt case, it is the only way that case runs at all."
        if dominant == "act" && self.max_seq_len > 1 {
            // What chunk size would fit, solved rather than guessed. The class
            // is quadratic in `seq_len`, so this is a search over a monotone
            // function and a bisection is exact to one token.
            let headroom = self.budget.saturating_sub(f.predicted() - f.activations);
            let mut lo = 1usize;
            let mut hi = self.max_seq_len;
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                let need = prefill_activation_bytes(mid, self.batch, self.prefill_geometry);
                if need <= headroom {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            s.push_str(&format!(
                "    chunk the prefill at {lo} tokens   frees {:>6.2} GB   \
                 (§13.2b; kv_len grows by chunk, §4.1)\n",
                gb(f.activations - prefill_activation_bytes(lo, self.batch, self.prefill_geometry)),
            ));
        }
        // Refuse rather than silently reduce (§9.5g): these are reported for the
        // caller to choose between, never applied. A silent reduction is a
        // configuration the caller did not choose and the artifact does not
        // record, which is §2.4's "an instrument that cannot be shown to have
        // engaged" in a new quantity -- and *which* lever to give up is a policy
        // the engine does not have, since a summarization job wants context
        // where a serving job wants concurrency.
        s.push_str(&format!(
            "    halve max_context             frees {:>6.2} GB\n",
            gb(f.kv / 2 + f.scratch / 2),
        ));
        s.push_str(&format!(
            "    halve B                       frees {:>6.2} GB\n",
            gb((f.kv + f.conv + f.activations + f.scratch) / 2),
        ));
        s.push_str("    scratch Reserve -> Grow       (see §9.1a)\n");
        s.push_str("    text-only (no vision tower)   frees   0.86 GB   (#162)\n");
        s
    }
}

/// Bytes the pools hold that **no planner owns**.
///
/// # This is the one derived quantity, and why it is not `live_bytes`
///
/// §9.5k had to read the allocation paths rather than §9.1's class table to get
/// this right, and the correction is what makes the check safe:
///
/// | term | path | lands in |
/// |---|---|---|
/// | weights | `to_dtype` → `new_buffer_builder` → `new_buffer` | **`private_buffers`** |
/// | KV | `KvSlot::append` → `Tensor::zeros` → `allocate_buffer` | **`buffers`** |
/// | scratch | `ScratchPlan`, when wired | pool |
/// | **arena** | `install_arena` → the **raw** device | **outside the pools** |
///
/// So **`live_bytes` already contains the weights and the KV reserve**, and
/// comparing `live_bytes` against the residual would **refuse the weights
/// themselves**. Corroborated from a direction nobody took for this purpose:
/// §6.3a measures the pool at 5220 MB and §6.3b at 5459 MB flat at 400 tokens,
/// against 5394 MB of weights — the pool footprint *is* approximately the
/// weights, not a 256 MiB region beside them.
///
/// `free_bytes` and `pending_bytes` belong on the left because both are bytes
/// the process holds and the OS cannot have. **`pending_bytes` is the term that
/// matters**: `free_bytes` is capped by `set_free_budget`, and `pending_bytes`
/// is capped by **nothing** — it is §6.3b's stranding at ~21 MB/token, 8.398 GB
/// over 400 tokens, the largest unbounded term this project has observed. The
/// residual is what finally gives it a ceiling.
///
/// The subtraction saturates: early in a run the pools hold less than the
/// planned set (weights are still loading), and a wrapping subtraction there
/// would report a colossal `unplanned` and refuse a run that is fine.
pub fn unplanned_bytes(occupancy: &PoolOccupancySnapshot, planned_in_pool: usize) -> usize {
    let held = occupancy.live_bytes + occupancy.free_bytes + occupancy.pending_bytes;
    held.saturating_sub(planned_in_pool)
}

/// The classes that are served **by the pool**, and therefore already inside
/// `live_bytes`.
///
/// The arena is excluded because `install_arena` calls the raw device — it is
/// the one class of the five that is genuinely outside the pools (§9.5k).
pub fn planned_in_pool(footprint: &Footprint) -> usize {
    footprint.weights + footprint.kv + footprint.scratch + footprint.conv
}

/// What the weight prediction was, measured against what was actually
/// allocated (`DESIGN.md` §9.5m, issue #244).
///
/// # The term this checks is the one nothing validated
///
/// [`Budget::weights`] is a **byte count the caller passes**, and its own doc
/// comment says *"text-only drops the 0.86 GB vision tower"* — so admission is
/// correct by construction only if the caller knows which, and nothing told it.
/// The arithmetic above is correct for whatever it is handed; this is the check
/// on **what it is handed**.
///
/// # Two errors, and only one of them is loud
///
/// **Over-prediction** — the full checkpoint's 6.25 GB passed for a text-only
/// process — over-predicts by 0.825 GB (#162). It fails *loudly*: a
/// configuration is refused that would have fit, and somebody notices.
///
/// **Under-prediction is the dangerous one.** A caller passing the *language*
/// figure for a process that later does run the tower under-predicts by the
/// same 0.825 GB, and §3.5 says **nothing reports an overrun**: Metal does no
/// dependency analysis and there is no safety net. A budget that under-predicts
/// does not refuse anything — it simply is not the bound it claims to be, on a
/// machine this project has kernel-panicked twice (§6.3c, §6.3d).
///
/// So the direction is not symmetric and this type does not treat it as though
/// it were. [`WeightReconciliation::under_predicted`] is the one that means the
/// budget is not a bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightReconciliation {
    /// What the caller passed as [`Budget::weights`].
    pub predicted: usize,
    /// What the weight-bearing pool actually holds, less the non-weight classes
    /// it also serves. See [`reconcile_weights`] for why this is a floor.
    pub observed: usize,
    /// `observed − predicted`, signed. Positive is **under**-prediction.
    pub divergence: i128,
    /// The tolerance the verdict was taken at.
    pub tolerance: usize,
}

impl WeightReconciliation {
    /// The caller passed **fewer** bytes than were allocated, by more than the
    /// tolerance. **This is the silent direction** — see the type docs.
    pub fn under_predicted(&self) -> bool {
        self.divergence > self.tolerance as i128
    }

    /// The caller passed **more** bytes than were allocated, by more than the
    /// tolerance. Wasteful, and loud when it refuses something.
    pub fn over_predicted(&self) -> bool {
        self.divergence < -(self.tolerance as i128)
    }

    /// Neither direction exceeded the tolerance.
    pub fn agrees(&self) -> bool {
        !self.under_predicted() && !self.over_predicted()
    }

    /// The report, in §9.5g's shape: the arithmetic, then what it means.
    pub fn describe(&self) -> String {
        let gb = |b: i128| b as f64 / 1e9;
        let head = format!(
            "weight prediction {:.3} GB against {:.3} GB allocated \
             (divergence {:+.3} GB, tolerance {:.3} GB)",
            gb(self.predicted as i128),
            gb(self.observed as i128),
            gb(self.divergence),
            gb(self.tolerance as i128),
        );
        if self.under_predicted() {
            format!(
                "{head}\n  UNDER-PREDICTED: admission was given a smaller weight \
                 figure than the process allocated, so the budget it computed is \
                 not a bound on this run. §3.5: nothing else reports an overrun. \
                 A caller passing the language-model figure for a load that also \
                 brings the vision tower is short by 0.825 GB (#162)."
            )
        } else if self.over_predicted() {
            format!(
                "{head}\n  over-predicted: admission was given a larger weight \
                 figure than the process allocated, so it may refuse a \
                 configuration that would have fit. A caller passing the full \
                 checkpoint for a text-only load is long by 0.825 GB (#162)."
            )
        } else {
            head
        }
    }
}

/// Compare admission's weight term against what the weight-bearing pool holds
/// (`DESIGN.md` §9.5m, issue #244).
///
/// # Why this is measurable at all, and where the number comes from
///
/// **#162 established the reading and this uses the same one.** `pool_live`
/// read **5.436 GB** against the language model's 5.394 GB, and the 42 MB
/// residual is attributable — the RoPE `cos`/`sin` tables at 16.4 MB, plus KV,
/// conv state and activations. The instrument is the pool's own counters, which
/// come from **outside** the model's arithmetic: #162 declined to cite the
/// `RESULT` line's `weight_bytes` for exactly this reason, since that figure is
/// *computed* from the config geometry and agreeing with a prediction is not
/// observing one.
///
/// `observed` is `live + free + pending` less the non-weight classes the same
/// pool serves. Under §9.5k's mapping the weight-bearing pool is
/// `private_buffers`, which holds weights **and** scratch and every activation
/// intermediate (§6.3a's correction), so the caller passes what it knows those
/// come to and this subtracts them.
///
/// # It is a floor, not an equality, and the tolerance is why
///
/// `observed` cannot be exact and is not claimed to be. It is a **floor on what
/// was allocated** taken at a moment: intermediates come and go, and the 42 MB
/// #162 measured is real and attributable. So the verdict is taken at a
/// tolerance, and the tolerance is the caller's to set from what it knows is
/// unaccounted — [`WEIGHT_RECONCILE_TOLERANCE`] is the default and states what
/// it is made of.
///
/// **The asymmetry is the point.** A tolerance that hides the 0.825 GB vision
/// tower would defeat the check; one that fires on 42 MB of RoPE tables would
/// cry wolf. The default sits between them by an order of magnitude at each end.
pub fn reconcile_weights(
    predicted_weights: usize,
    occupancy: &PoolOccupancySnapshot,
    non_weight_classes_in_pool: usize,
    tolerance: usize,
) -> WeightReconciliation {
    let held = occupancy.live_bytes + occupancy.free_bytes + occupancy.pending_bytes;
    let observed = held.saturating_sub(non_weight_classes_in_pool);
    WeightReconciliation {
        predicted: predicted_weights,
        observed,
        divergence: observed as i128 - predicted_weights as i128,
        tolerance,
    }
}

/// The default tolerance for [`reconcile_weights`], and what it is made of.
///
/// **256 MB, and it is a bound on the unaccounted rather than a round number.**
/// #162 measured the whole unattributed residual at **42 MB** — RoPE `cos`/`sin`
/// at 16.4 MB plus KV, conv and activations — on a live decode. This is ~6× that
/// figure, which leaves room for a larger `max_position_embeddings` to grow the
/// RoPE term and for intermediates in flight at the moment of the reading.
///
/// **What it must not do is hide a tower.** The quantity this check exists to
/// catch is 0.825 GB (#162), which is **3.2× this tolerance** — so the default
/// cannot mask the error it was written for, and that ratio is asserted by
/// `tolerance_cannot_hide_the_vision_tower` rather than left as a claim.
pub const WEIGHT_RECONCILE_TOLERANCE: usize = 256 * 1_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1e9;
    /// `hw.memsize` on the development machine (§3.4). Used as a stand-in for
    /// `recommendedMaxWorkingSetSize`, which is read from the device at runtime
    /// and cannot be assumed in a unit test.
    const MACHINE: usize = 68_719_476_736;
    /// §5.5: language model 5.394 GB + vision 0.83 + projector 0.03.
    const WEIGHTS_FULL: usize = 6_254_000_000;

    fn b1(capacity: usize, max_context: usize) -> Budget {
        Budget::new(WEIGHTS_FULL, capacity, max_context)
    }

    /// §9.5b's correction, asserted so it cannot drift back.
    ///
    /// §5.6's table was computed with 16 000 B/token where the geometry on its
    /// own preceding line derives 16 384. A budget quoting the table would
    /// under-predict by 2.4 % — 1.61 GB at B=32 × 128k — in exactly the regime
    /// where the margin matters.
    #[test]
    fn kv_geometry_is_16384_not_16000() {
        assert_eq!(
            KV_BYTES_PER_TOKEN, 16_384,
            "2 x 8 heads x 64 dim x 2 B x 8 layers"
        );
        assert_ne!(
            KV_BYTES_PER_TOKEN, 16_000,
            "§5.6's table, which is low by 2.4 %"
        );
        // The understatement at the configuration where it stops being
        // negligible, from §9.5b.
        let understated = 32 * (KV_BYTES_PER_TOKEN - 16_000) * 131_072;
        assert!(
            (understated as f64 / GB - 1.61).abs() < 0.01,
            "§9.5b: 1.61 GB at B=32 x 128k, got {:.3}",
            understated as f64 / GB
        );
    }

    /// The class table reconciles with §9.5b's, row for row.
    ///
    /// Not a restatement: this is the arithmetic §9.5b's table was drawn from,
    /// so a divergence here means one of the two moved. The tolerance is 10 MB,
    /// tight enough that a class-sized error cannot hide in it.
    #[test]
    fn footprint_reconciles_with_design_9_5b() {
        // (label, B, capacity, max_context, expected TOTAL in GB from §9.5b,
        //  minus its `pool` column -- §9.5k removed that row, so the expected
        //  figure here is the table's total less its 0.268 GB pool entry.)
        let cases = [
            ("B=1 ctx 4k", 1, 4096, 4096, 6.591 - 0.268),
            ("B=1 ctx 32k", 1, 32768, 32768, 7.068 - 0.268),
            ("B=1 ctx 128k", 1, 131072, 131072, 8.705 - 0.268),
            ("B=16 ctx 128k", 16, 131072, 131072, 41.441 - 0.268),
        ];
        for (label, batch, cap, mc, expect_gb) in cases {
            let mut b = b1(cap, mc);
            b.batch = batch;
            let got = b.footprint().predicted() as f64 / GB;
            assert!(
                (got - expect_gb).abs() < 0.01,
                "{label}: §9.5b says {expect_gb:.3} GB, got {got:.3} GB"
            );
        }
    }

    /// **Both bounds, which is the whole point** (§8.1g, and #184's precedent).
    ///
    /// An admission check that refuses everything is not a check. The two arms
    /// are asserted together in one test so neither can be dropped without the
    /// other going red.
    #[test]
    fn admits_a_configuration_that_fits_and_refuses_one_that_does_not() {
        // The configuration that ships today (§9.5b: 9.6 % of the machine).
        let ok = b1(4096, 4096).admit(MACHINE);
        assert!(ok.fits, "B=1 ctx 4k must be ADMITTED:\n{}", ok.describe());
        assert!(
            ok.residual > OBSERVED_STRANDING_BYTES,
            "and it must be ample, not merely admitted: residual {:.2} GB",
            ok.residual as f64 / GB
        );

        // B=32 at 128k: §9.5b's `KV alone = 68.719 GB, exactly hw.memsize`,
        // before a weight is loaded.
        let mut over = b1(131_072, 131_072);
        over.batch = 32;
        let refused = over.admit(MACHINE);
        assert!(!refused.fits, "B=32 ctx 128k must be REFUSED");
        assert!(
            refused.residual_signed() < 0,
            "a refusal reports a negative residual, got {}",
            refused.residual_signed()
        );
        // §9.5b: KV x B is the only class that can eat the machine on its own.
        assert!(
            refused.footprint.kv as f64 / GB > 68.0,
            "KV alone at B=32 x 128k is 68.719 GB (§9.5b), got {:.3}",
            refused.footprint.kv as f64 / GB
        );
    }

    /// **`B=16 ctx 128k` passes the sum-based table and fails the residual
    /// test** — §9.5k, and the trap the issue names explicitly.
    ///
    /// It sums to 60.3 % of the machine, which looks comfortable, and its
    /// residual is 3.495 GB against §6.3b's **measured** 8.398 GB of stranding
    /// over 400 tokens. Grading against a measured quantity rather than a round
    /// number is what makes the difference, and a refusal that did not reflect
    /// it would admit a configuration a known allocation shape can kill.
    #[test]
    fn b16_ctx128k_fits_the_sum_and_is_tight_against_measured_stranding() {
        let mut b = b1(131_072, 131_072);
        b.batch = 16;
        let a = b.admit(MACHINE);

        // It fits the sum: §9.5b puts it at 60.3 % of the machine.
        assert!(a.fits, "it passes the sum-based table:\n{}", a.describe());
        let share = a.footprint.predicted() as f64 / MACHINE as f64;
        assert!(
            (share - 0.603).abs() < 0.01,
            "§9.5b: 60.3 % of the machine, got {:.1} %",
            share * 100.0
        );

        // And it is tight: §9.5k's residual at fraction 0.65 is 3.495 GB.
        assert!(
            a.is_tight(),
            "the residual must be graded against §6.3b's measured stranding, \
             not a round number:\n{}",
            a.describe()
        );
        assert!(
            (a.residual as f64 / GB - 3.495).abs() < 0.02,
            "§9.5k: residual 3.495 GB at fraction 0.65, got {:.3} GB",
            a.residual as f64 / GB
        );
        assert!(
            a.residual < OBSERVED_STRANDING_BYTES,
            "3.495 GB residual is less than 8.398 GB of measured stranding"
        );
    }

    /// §9.5k's residual table, reproduced row for row.
    #[test]
    fn residual_reconciles_with_design_9_5k() {
        let cases = [
            ("B=1 ctx 4k", 1, 4096, 4096, 38.345),
            ("B=1 ctx 128k", 1, 131072, 131072, 36.231),
            ("B=16 ctx 32k", 16, 32768, 32768, 29.680),
            ("B=16 ctx 128k", 16, 131072, 131072, 3.495),
        ];
        for (label, batch, cap, mc, expect_gb) in cases {
            let mut b = b1(cap, mc);
            b.batch = batch;
            let got = b.admit(MACHINE).residual as f64 / GB;
            assert!(
                (got - expect_gb).abs() < 0.02,
                "{label}: §9.5k says residual {expect_gb:.3} GB, got {got:.3} GB"
            );
        }
    }

    /// **Comparing `live_bytes` against the residual would refuse the weights
    /// themselves** — §9.5k's load-bearing correction, asserted.
    ///
    /// The mutation this guards against is the *obvious* form of the check, and
    /// it is the one the issue calls out: weights land in `private_buffers` and
    /// KV in `buffers`, so both are already inside `live_bytes`.
    #[test]
    fn unplanned_subtracts_the_classes_the_pool_already_holds() {
        let f = b1(4096, 4096).footprint();
        let planned = planned_in_pool(&f);

        // A pool holding exactly the planned set and nothing else.
        let occ = PoolOccupancySnapshot {
            live_bytes: planned,
            free_bytes: 0,
            pending_bytes: 0,
            ..Default::default()
        };
        assert_eq!(
            unplanned_bytes(&occ, planned),
            0,
            "a pool holding only the planned classes has ZERO unplanned bytes; \
             comparing live_bytes against the residual would refuse the weights"
        );

        // The same pool, plus a token's worth of §6.3b stranding.
        let strand = 21 * 1_000_000;
        let occ = PoolOccupancySnapshot {
            live_bytes: planned,
            pending_bytes: strand,
            ..Default::default()
        };
        assert_eq!(
            unplanned_bytes(&occ, planned),
            strand,
            "pending_bytes is the unbounded term (§6.3b: ~21 MB/token) and must \
             be counted"
        );
    }

    /// `free_bytes` and `pending_bytes` are on the left of the subtraction.
    ///
    /// Both are bytes the process holds and the OS cannot have (§9.5k). A
    /// version counting only `live_bytes` would miss the one term that is
    /// capped by nothing.
    #[test]
    fn unplanned_counts_free_and_pending_not_only_live() {
        let occ = PoolOccupancySnapshot {
            live_bytes: 100,
            free_bytes: 20,
            pending_bytes: 3,
            ..Default::default()
        };
        assert_eq!(unplanned_bytes(&occ, 100), 23);
        // And it saturates rather than wrapping: early in a run the pool holds
        // less than the planned set, and a wrapping subtraction would report a
        // colossal figure and refuse a run that is fine.
        assert_eq!(unplanned_bytes(&occ, 10_000), 0);
    }

    // ---- #244: the weight term is caller-supplied, and this is the check ----

    /// §5.5's language-model weights: what a text-only LFM2 load carries.
    const WEIGHTS_LANGUAGE: usize = 5_394_397_184;
    /// #162 §1, from the safetensors header: the vision tower alone.
    const VISION_TOWER: usize = 825_299_424;

    /// A pool holding `weights` plus #162's measured 42 MB of attributable
    /// non-weight residual, which is what a live decode actually looks like.
    fn pool_holding(weights: usize) -> PoolOccupancySnapshot {
        PoolOccupancySnapshot {
            live_bytes: weights + 42_000_000,
            ..Default::default()
        }
    }

    /// **A caller passing the WRONG constant must be caught — issue #244's
    /// whole point, in the direction that fails silently.**
    ///
    /// A caller that passes the *language* figure for a process that also loads
    /// the vision tower under-predicts by 0.825 GB, and §3.5 says nothing
    /// reports an overrun. The arithmetic in this module is correct for whatever
    /// it is handed; this is the check on what it is handed.
    #[test]
    fn a_caller_passing_the_language_figure_for_a_vl_load_is_caught() {
        // The process actually allocated the language model AND the tower.
        let occ = pool_holding(WEIGHTS_LANGUAGE + VISION_TOWER);
        // The caller told admission it would only be the language model.
        let r = reconcile_weights(WEIGHTS_LANGUAGE, &occ, 0, WEIGHT_RECONCILE_TOLERANCE);

        assert!(
            r.under_predicted(),
            "under-prediction is the silent direction and MUST be caught:\n{}",
            r.describe()
        );
        assert!(!r.over_predicted(), "{}", r.describe());
        assert!(!r.agrees(), "{}", r.describe());

        // It is short by the tower, and the report says so in bytes rather than
        // in a boolean (§9.5g).
        let short = r.divergence - 42_000_000;
        assert!(
            (short - VISION_TOWER as i128).abs() < 1_000_000,
            "#162: short by the tower's 0.825 GB, got {:.3} GB",
            short as f64 / 1e9
        );
        assert!(r.describe().contains("UNDER-PREDICTED"), "{}", r.describe());
    }

    /// **The other bound, and without it the check above is worthless**
    /// (§8.1g, #184's precedent — both arms in one test so neither can be
    /// dropped without the other going red).
    ///
    /// A check that reports every caller wrong is not a check. The correct
    /// caller must be quiet, and the *opposite* error must read as the opposite
    /// error rather than as this one.
    #[test]
    fn the_correct_caller_is_quiet_and_over_prediction_reads_as_over() {
        // Correct: told the truth about a text-only load.
        let occ = pool_holding(WEIGHTS_LANGUAGE);
        let right = reconcile_weights(WEIGHTS_LANGUAGE, &occ, 0, WEIGHT_RECONCILE_TOLERANCE);
        assert!(
            right.agrees(),
            "a caller that got it right must not be reported wrong:\n{}",
            right.describe()
        );
        assert!(!right.under_predicted() && !right.over_predicted());

        // #162's own case: the full checkpoint passed for a text-only process.
        // Real, and it fails LOUDLY -- so it must not be reported as the silent
        // direction.
        let over = reconcile_weights(
            WEIGHTS_LANGUAGE + VISION_TOWER,
            &occ,
            0,
            WEIGHT_RECONCILE_TOLERANCE,
        );
        assert!(over.over_predicted(), "{}", over.describe());
        assert!(
            !over.under_predicted(),
            "over-prediction must NOT be reported as the dangerous direction:\n{}",
            over.describe()
        );
        assert!(
            over.describe().contains("over-predicted"),
            "{}",
            over.describe()
        );
    }

    /// **The tolerance must not be able to hide the thing it exists to catch.**
    ///
    /// #162 measured the unattributed residual at 42 MB (RoPE at 16.4 MB plus
    /// KV, conv and activations) and the vision tower at 0.825 GB. The tolerance
    /// has to sit between them, and this asserts the ratio at both ends rather
    /// than leaving it as a claim in a doc comment.
    #[test]
    fn tolerance_cannot_hide_the_vision_tower() {
        // A `const` block, so the lower bound is checked at COMPILE time: the
        // tolerance is a constant and a constant relation between constants
        // wants no test run to establish it.
        const {
            assert!(
                WEIGHT_RECONCILE_TOLERANCE > 42_000_000 * 4,
                "the tolerance must clear #162's measured 42 MB residual"
            );
        }
        assert!(
            (VISION_TOWER as f64 / WEIGHT_RECONCILE_TOLERANCE as f64) > 3.0,
            "the tower must be at least 3x the tolerance or the check is \
             defeated by its own slack: ratio {:.2}",
            VISION_TOWER as f64 / WEIGHT_RECONCILE_TOLERANCE as f64
        );
        // And #162's residual alone must be quiet: an instrument that fires on
        // the RoPE tables would be turned off within a week.
        let occ = pool_holding(WEIGHTS_LANGUAGE);
        let r = reconcile_weights(WEIGHTS_LANGUAGE, &occ, 0, WEIGHT_RECONCILE_TOLERANCE);
        assert!(
            r.agrees(),
            "42 MB of RoPE and KV must not fire:\n{}",
            r.describe()
        );
    }

    /// The non-weight classes the same pool serves are subtracted.
    ///
    /// §9.5k's mapping puts scratch and every activation intermediate in
    /// `private_buffers` alongside the weights (§6.3a's correction), so a check
    /// that did not subtract them would read them as under-prediction and
    /// report a correct caller as wrong.
    #[test]
    fn non_weight_classes_in_the_same_pool_are_subtracted() {
        let scratch = 33 * 1024 * 1024;
        let occ = pool_holding(WEIGHTS_LANGUAGE + scratch);

        let unsubtracted = reconcile_weights(WEIGHTS_LANGUAGE, &occ, 0, WEIGHT_RECONCILE_TOLERANCE);
        let subtracted =
            reconcile_weights(WEIGHTS_LANGUAGE, &occ, scratch, WEIGHT_RECONCILE_TOLERANCE);
        assert!(
            subtracted.agrees(),
            "scratch is planned and must not read as under-prediction:\n{}",
            subtracted.describe()
        );
        assert_eq!(
            subtracted.divergence + scratch as i128,
            unsubtracted.divergence,
            "the subtraction is exactly the classes the caller named"
        );

        // And it saturates: early in a run the pool holds less than the classes
        // named, and a wrapping subtraction would report a colossal observed
        // figure and call a fine run under-predicted.
        let empty = PoolOccupancySnapshot::default();
        let early = reconcile_weights(WEIGHTS_LANGUAGE, &empty, 10_000, WEIGHT_RECONCILE_TOLERANCE);
        assert_eq!(early.observed, 0);
        assert!(!early.under_predicted(), "{}", early.describe());
    }

    /// The scratch class is sized from **query** heads, not KV heads.
    ///
    /// §9.1a: using 8 under-sizes the class **4×** — 1.03 MB at 128k where the
    /// truth is 4.12 — and §3.5 says nothing would report the overrun.
    #[test]
    fn scratch_uses_query_heads_and_reconciles_with_9_1a() {
        let f = b1(131_072, 131_072).footprint();
        // §9.1a: Reserve at 8 layers, max_context 128k, is 33.0 MiB.
        let mib = f.scratch as f64 / (1024.0 * 1024.0);
        assert!(
            (mib - 33.0).abs() < 0.1,
            "§9.1a: 33.0 MiB at 8 layers, got {mib:.2} MiB"
        );
        // The 4x error the KV-head reading would produce.
        let with_kv_heads = mib / 4.0;
        assert!(
            (with_kv_heads - 8.25).abs() < 0.1,
            "using 8 KV heads would give {with_kv_heads:.2} MiB -- the 4x \
             under-size §9.1a warns about"
        );
    }

    /// A refusal names the dominant class and does not silently reduce.
    ///
    /// §9.5g: refuse rather than reduce, because *which* lever to give up is a
    /// policy the engine does not have.
    #[test]
    fn refusal_reports_the_arithmetic_and_names_the_dominant_class() {
        let mut b = b1(131_072, 131_072);
        b.batch = 32;
        let a = b.admit(MACHINE);
        let msg = a.describe();
        assert!(msg.contains("memory budget exceeded"), "{msg}");
        assert!(
            msg.contains("KV"),
            "the dominant class must be named: {msg}"
        );
        assert!(msg.contains("<- dominant"), "{msg}");
        assert!(
            msg.contains("reductions that would fit"),
            "the caller needs the levers to choose between: {msg}"
        );
        // And the admitted case reports the residual, which §9.5g calls the
        // more useful half because it is the case that happens.
        let ok = b1(4096, 4096).admit(MACHINE).describe();
        assert!(ok.contains("memory budget OK"), "{ok}");
        assert!(ok.contains("residual"), "{ok}");
    }

    // ---------------------------------------------------------------- #306
    // The activation term at `seq_len > 1`. `DESIGN.md` §13.4c is the
    // measurement every figure below is checked against; the arithmetic is
    // `measurements/issue-306-raw/prefill_arithmetic.py` in the lloom repo,
    // and these tests are what stop the two drifting apart.

    /// `recommendedMaxWorkingSetSize` on the development machine.
    ///
    /// **55.663 GB, not `hw.memsize`'s 68.719** — §9.5l read it and recorded
    /// that the stand-in is optimistic by 19 %. Used here because #306's
    /// refusals are about a real machine's real ceiling.
    const RMWSS: usize = 55_663_000_000;

    /// **Bound 1 of 2: decode is unmoved.**
    ///
    /// The load-bearing half, and the one a suite testing only the refusal
    /// would miss (§8.1g, and §9.5l's own finding 3). At `seq_len == 1` the
    /// activation class must be **exactly** #68's measured 68 KB × B, because
    /// every verdict this project has ever recorded was taken there.
    #[test]
    fn decode_activation_term_is_unmoved_at_68kb() {
        for batch in [1, 8, 16, 32] {
            let got = prefill_activation_bytes(1, batch, PrefillGeometry::default());
            assert_eq!(
                got,
                batch * ACTIVATION_ARENA_BYTES,
                "seq_len == 1 must reproduce #68's measured arena at B={batch}"
            );
        }
        // And the whole footprint at decode is byte-identical to what §9.5b's
        // table records -- which `footprint_reconciles_with_design_9_5b` also
        // asserts, from the other direction.
        let f = b1(4096, 4096).footprint();
        assert_eq!(f.activations, ACTIVATION_ARENA_BYTES);
    }

    /// **Bound 2 of 2: the prefill term reconciles with §13.4c's measurement.**
    ///
    /// Two lengths, because one point cannot distinguish a correct model from
    /// a constant that happens to match. The measured quantity is the
    /// **increment over the settled process footprint**, since the peak also
    /// carries the load-time f32 staging transient and the RoPE tables, which
    /// are flat in `P` -- checked, and the residual against a
    /// weights-only baseline is flat at ~6.6 GB across both lengths, which is
    /// what identifies it as a constant rather than a missed term.
    #[test]
    fn prefill_activation_term_reconciles_with_design_13_4c() {
        // (P, measured peak, settled floor), from #250's raw series
        // (`measurements/issue-250-raw/mem/mem-p{1536,2048}.jsonl.gz`),
        // re-derived rather than quoted from the section.
        let cases = [
            (1536usize, 23.7426_f64, 11.46_f64),
            (2048, 30.5205, 11.4614),
        ];
        for (p, peak_gb, floor_gb) in cases {
            let measured = peak_gb - floor_gb;
            let predicted = prefill_activation_bytes(p, 1, PrefillGeometry::default()) as f64 / GB;
            let ratio = predicted / measured;
            assert!(
                (0.80..=1.20).contains(&ratio),
                "P={p}: predicted {predicted:.3} GB against a measured increment of \
                 {measured:.3} GB, ratio {ratio:.3} -- outside 20 %"
            );
        }
    }

    /// §13.2b's own arithmetic counts one tensor of one layer, and is 35× low.
    ///
    /// Asserted so the correction cannot be quietly reverted to the figure
    /// that under-predicted. Three `[B, 32, P, P]` f32 tensors are live per
    /// attention layer, not one, and there are 8 layers.
    #[test]
    fn one_score_matrix_is_not_the_activation_term() {
        let p = 2048;
        let one_tensor = 32 * p * p * 4; // §13.2b's `[32, P, P]` f32
        assert!(
            (one_tensor as f64 / GB - 0.537).abs() < 0.001,
            "§13.2b computes 0.537 GB for one layer's scores"
        );
        let whole = prefill_activation_bytes(p, 1, PrefillGeometry::default());
        let factor = whole as f64 / one_tensor as f64;
        assert!(
            factor > 30.0,
            "§13.4c measured the non-weight peak at ~35x §13.2b's single \
             tensor; got {factor:.1}x"
        );
    }

    /// The term is quadratic in `seq_len`, which is what makes it the one that
    /// overruns. A linear model would not refuse `P` = 4096.
    #[test]
    fn the_activation_term_is_quadratic_in_seq_len() {
        let g = PrefillGeometry::default();
        let a = prefill_activation_bytes(1024, 1, g) as f64;
        let b = prefill_activation_bytes(2048, 1, g) as f64;
        let c = prefill_activation_bytes(4096, 1, g) as f64;
        // Doubling P more than trebles the term (it is quadratic plus a
        // linear part), and the growth accelerates.
        assert!(b / a > 2.5, "1024 -> 2048 grew {:.2}x", b / a);
        assert!(c / b > b / a, "the growth must accelerate: quadratic");
    }

    /// **What the correction buys: `P` = 4096 is refused on arithmetic.**
    ///
    /// §13.4c stopped at `P` = 2048 **by prediction rather than by failure**
    /// (§9.5j) and deliberately did not attempt 4096. This is what makes that
    /// a decision rather than a gap: admission refuses it without the machine
    /// being asked for it.
    ///
    /// **Both directions asserted.** `P` = 2048 *ran* — 30.5 GB peak, and it
    /// completed — so refusing it would be a defect, and the margin it is
    /// admitted by is stated rather than tuned.
    #[test]
    fn prefill_at_4096_is_refused_and_2048_is_admitted() {
        let mut over = b1(4096, 4096).with_prefill(4096);
        over.weights = 5_394_397_184; // text-only, #162
        let a = over.admit(RMWSS);
        assert!(
            !a.fits,
            "P=4096 predicts {:.2} GB against a {:.2} GB budget and must be \
             refused (§9.5j: predicted, not attempted)",
            a.footprint.predicted() as f64 / GB,
            a.budget as f64 / GB
        );

        // The admitted arm is the load-bearing half (§8.1g): a check that
        // refuses everything is broken, and P=2048 is a configuration that
        // demonstrably ran.
        let mut ran = b1(4096, 4096).with_prefill(2048);
        ran.weights = 5_394_397_184;
        let ok = ran.admit(RMWSS);
        assert!(
            ok.fits,
            "P=2048 ran on this machine (§13.4c) and must be admitted; \
             predicted {:.2} GB against {:.2} GB",
            ok.footprint.predicted() as f64 / GB,
            ok.budget as f64 / GB
        );
        // ------------------------------------------------------------------
        // **The margin, and the direction it errs in, stated rather than
        // tuned.** This is the honest half of the result and it is asserted so
        // it cannot be quietly forgotten.
        //
        // At P=2048 admission now predicts ~24.1 GB where §13.4c **measured a
        // 30.52 GB peak `phys_footprint`**. So the corrected term still
        // UNDER-predicts the observed peak by ~6.4 GB, and under-prediction is
        // the silent direction (§9.5m, §3.5 reports no overrun).
        //
        // **That residual is a constant, not a missed `seq_len` term**, which
        // is why the correction is still worth taking: measured at two prompt
        // lengths it is 6.78 GB at P=1536 and 6.47 GB at P=2048 -- a ratio of
        // 0.95 across a 1.33x span of P, where a missed quadratic term would
        // have grown 1.78x. It is the load-time f32 staging transient (§2.4's
        // first touch), the RoPE tables, and pool bytes held against
        // allocations outside all five classes -- §9.5f's candidate 3, which
        // §6.3e establishes an accounting instrument cannot find.
        //
        // What this test pins is that the gap is BOUNDED and known, so a
        // future change that makes it grow is a failure rather than a drift.
        let predicted_gb = ok.footprint.predicted() as f64 / GB;
        let measured_peak_gb = 30.5205;
        let gap = measured_peak_gb - predicted_gb;
        assert!(
            (4.0..=9.0).contains(&gap),
            "P=2048: admission predicts {predicted_gb:.2} GB against §13.4c's \
             measured {measured_peak_gb:.2} GB peak, a gap of {gap:.2} GB. \
             This gap is the flat non-class residual and is expected to be \
             ~6.4 GB; outside 4-9 GB means either the arithmetic moved or the \
             residual stopped being flat."
        );
    }

    /// A refusal whose dominant class is the activations offers **chunking**,
    /// and the chunk size it names actually fits.
    ///
    /// §13.2b: chunked prefill *"is not an optimization for the long-prompt
    /// case, it is the only way that case runs at all"* — so halving
    /// `max_context` is the wrong lever for this refusal and the message must
    /// not lead with it.
    #[test]
    fn an_activation_refusal_names_a_chunk_size_that_fits() {
        let mut b = b1(4096, 4096).with_prefill(8192);
        b.weights = 5_394_397_184;
        let a = b.admit(RMWSS);
        assert!(!a.fits, "P=8192 must be refused");
        let msg = a.describe();
        assert!(
            msg.contains("act") && msg.contains("<- dominant"),
            "the activations must be named as dominant at P=8192: {msg}"
        );
        assert!(
            msg.contains("chunk the prefill at"),
            "an activation-dominated refusal must offer chunking: {msg}"
        );

        // And the size it names must actually fit -- a refusal that suggests a
        // reduction which does not fit is worse than one that suggests none.
        let chunk: usize = msg
            .split("chunk the prefill at ")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .expect("the message states a chunk size");
        assert!((1..8192).contains(&chunk), "chunk {chunk} out of range");
        let chunked = b.with_prefill(chunk).admit(RMWSS);
        assert!(
            chunked.fits,
            "the chunk size the refusal names must fit: {chunk} tokens \
             predicts {:.2} GB against {:.2} GB",
            chunked.footprint.predicted() as f64 / GB,
            chunked.budget as f64 / GB
        );
        // And one token larger must not, so the solve is exact rather than
        // conservative -- otherwise it would under-use the machine silently.
        let over = b.with_prefill(chunk + 1).admit(RMWSS);
        assert!(
            !over.fits,
            "the chunk solve must be exact to one token; {} also fits",
            chunk + 1
        );
    }

    /// §6.2's two defects are counted: `repeat_kv`'s 4× expansion and the f32
    /// upcast, both of which are **still live on the prefill arm** because
    /// §6.2a's fix is `seq_len == 1` only.
    ///
    /// # Why this test exists separately, and the coverage limit it records
    ///
    /// A mutation deleting both terms **survived every other test in this
    /// module**, and it is recorded rather than papered over (§11.0a's rule —
    /// can the instrument express the difference it is meant to detect?). The
    /// reason is arithmetic rather than negligence: at `P` = 2048 the two terms
    /// are **0.537 GB of an 18.66 GB class, 2.88 %**, which sits inside the
    /// 20 % window `prefill_activation_term_reconciles_with_design_13_4c`
    /// needs in order not to over-fit a two-point measurement. **Widening that
    /// test to catch this would make it assert a precision the measurement
    /// does not support**, so the term is checked here, at the level where it
    /// is visible.
    ///
    /// §9.2c's alignment lesson in a third quantity: check at the level where
    /// the rounding happens, not at an aggregate that inherits it.
    #[test]
    fn the_gqa_expansion_and_the_f32_upcast_are_counted() {
        let g = PrefillGeometry::default();
        let p = 2048usize;
        let whole = prefill_activation_bytes(p, 1, g) as f64;

        // The two terms, computed independently of the function under test:
        // 2 x [B,32,P,64] f16 for repeat_kv(k)/(v) and 3 x the same at f32 for
        // the q/k/v upcast, over 8 attention layers.
        let heads = 32 * p * 64;
        let expect = (8 * (2 * heads * 2 + 3 * heads * 4)) as f64;
        assert!(
            (expect / 1e9 - 0.537).abs() < 0.01,
            "§6.2's two defects come to 0.537 GB at P=2048; got {:.3}",
            expect / 1e9
        );

        // They are ~2.9 % of the class -- stated so the coverage limit above
        // is a number a reader can check rather than an assertion.
        let share = expect / whole;
        assert!(
            (0.02..0.04).contains(&share),
            "§6.2's terms are ~2.9 % of the class; got {share:.4}"
        );

        // ------------------------------------------------------------------
        // **The check that actually detects their deletion**, and it works by
        // isolating the linear-in-P part from the quadratic one rather than by
        // asserting the aggregate to a precision the measurement cannot carry.
        //
        // The class is `a*P^2 + b*P`. Evaluating at two lengths and
        // differencing out the quadratic term recovers `b`, which is where
        // these two terms live -- so deleting them moves `b` by their exact
        // size even though they are 2.9 % of the total.
        let at = |q: usize| prefill_activation_bytes(q, 1, g) as f64;
        let (p1, p2) = (1024.0_f64, 2048.0_f64);
        let (y1, y2) = (at(1024), at(2048));
        // a = (y2/p2^2 - y1/p1^2) / (1 - p1/p2) ... solve the 2x2 directly:
        // y = a*P^2 + b*P  =>  y/P = a*P + b, a line in P.
        let a = (y2 / p2 - y1 / p1) / (p2 - p1);
        let b = y1 / p1 - a * p1;
        // `b` is the per-token linear coefficient across all 30 layers: the
        // conv and MLP activations, plus the 8 attention layers' heads-shaped
        // tensors -- which is where `repeat_kv` and the upcast sit.
        //
        // **Asserted at its exact value**, because that is what makes the
        // deletion detectable. The terms are 262 144 B/token of a 2 803 712
        // B/token coefficient, so dropping them takes it to 2 541 568. **A
        // one-sided bound does not see that**: a first version of this test
        // asserted `b > heads_per_token` and the mutation survived, since
        // 2 541 568 still exceeds 262 144. Recorded because it is the same
        // shape as §11.3j's vacuous parity arm -- a check that compares
        // against nothing specific enough to fail.
        let heads_per_token = (8 * (2 * 32 * 64 * 2 + 3 * 32 * 64 * 4)) as f64;
        assert_eq!(heads_per_token, 262_144.0, "§6.2's terms, per token");
        assert!(
            (b - 2_803_712.0).abs() < 1.0,
            "the linear coefficient is 2 803 712 B/token WITH §6.2's terms and \
             2 541 568 without; recovered {b:.0}"
        );
        // And `a`, the quadratic coefficient: 8 layers x 3 score matrices x 32
        // heads x 4 B = 3072, plus 8 for the u8 causal mask.
        assert!(
            (a - 3080.0).abs() < 1.0,
            "the quadratic coefficient is 3080 B/token^2; got {a:.1}"
        );
    }

    /// The geometry is the caller's, and using the KV head count under-sizes
    /// the class 4× — §9.1a's own warning about the scratch class, in a
    /// second place.
    #[test]
    fn the_query_head_count_is_what_sizes_the_score_matrix() {
        let g = PrefillGeometry::default();
        assert_eq!(g.query_heads, 32, "§5.2: 32 query heads");
        let correct = prefill_activation_bytes(2048, 1, g);
        let wrong = prefill_activation_bytes(
            2048,
            1,
            PrefillGeometry {
                query_heads: 8, // the KV head count -- the mistake
                ..g
            },
        );
        // 2.18x on the WHOLE class rather than 4x, because only the attention
        // terms scale with the head count -- the conv and MLP terms do not.
        // Asserted at the value it actually has rather than at the headline,
        // since a test asserting 4x here would be asserting a figure about a
        // different quantity (§9.1a's 4x is about the scratch class alone).
        let whole = correct as f64 / wrong as f64;
        assert!(
            (2.0..2.4).contains(&whole),
            "the whole class moves 2.18x on the head count; got {whole:.2}x"
        );

        // The score matrix itself IS 4x, and that is the figure §9.1a's
        // warning is about. Isolated so the claim is checked where it holds.
        let p = 2048usize;
        let scores_32 = 32 * p * p * 4;
        let scores_8 = 8 * p * p * 4;
        assert_eq!(
            scores_32 / scores_8,
            4,
            "the score matrix is 4x on the query-head count"
        );
    }
}
