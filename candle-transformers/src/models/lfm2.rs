//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::ops::{FlashScratchSizing, KvQuant};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // FFN dimension configuration
    #[serde(default = "default_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f32,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: usize,
}

fn default_num_key_value_heads() -> usize {
    8
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    128000
}

fn default_conv_l_cache() -> usize {
    3
}

fn default_ffn_dim_multiplier() -> f32 {
    1.0
}

fn default_block_multiple_of() -> usize {
    256
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Compute the actual intermediate size for the FFN.
    /// LFM2 uses: hidden_size * 4 * block_ffn_dim_multiplier, rounded to block_multiple_of
    fn compute_intermediate_size(&self) -> usize {
        let base_size = (self.hidden_size as f32 * 4.0 * self.block_ffn_dim_multiplier) as usize;
        let multiple = self.block_multiple_of;
        base_size.div_ceil(multiple) * multiple
    }

    pub fn into_config(self, use_flash_attn: bool) -> Config {
        // Use computed intermediate size (matches actual weights) instead of config field
        let intermediate_size = self.compute_intermediate_size();
        Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types: self.layer_types,
            tie_embedding: self.tie_embedding,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            use_flash_attn,
            attn_impl: AttnImpl::default(),
            flash_page_size: 256,
            flash_pages_per_chunk: 1,
            // `Grow`, which is what #116's per-call allocation did before the
            // axis reached it (#234). Not a choice — see the field's note.
            flash_scratch_sizing: FlashScratchSizing::default(),
            kv_append: KvAppend::default(),
            kv_quant: KvQuant::default(),
            conv_state: ConvState::default(),
            // §9.5's admission is off unless a caller asks for it, per §7.1a.
            memory_budget: None,
            // Both GEMV epilogue fusions are off, per §7.1a. #536 and #509 both
            // say the default needs the decode-path number, which is not taken
            // here — this change makes the arm reachable, nothing more.
            gemv_silu: false,
            gemv_residual: false,
            // No tap, per §7.1a: an empty `tap_layers` is what every existing
            // caller had before #363 and costs an `is_empty()` per layer. The
            // indices belong to a *drafter* checkpoint, so there is no
            // meaningful default for a target config to carry.
            tap_layers: Vec::new(),
            tap_dtype: DType::F32,
        }
    }
}

/// How the decode path writes conv state.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms, in the shape §9.1a uses
/// for `ScratchSizing`: every arm is compiled, the selection is one field, and a
/// regression is revertible without a rebuild.
///
/// # Three arms, and the third moves the digests deliberately
///
/// §10.2a specifies "a rotating base into a `(l_cache + K)`-wide buffer" and
/// predicts the wrap is nearly free because the phase is uniform across the
/// simdgroup (§3.3). The uniformity argument is sound; what that section gets
/// wrong is the premise it rests on. It argues the ring "sums the same three
/// products in the same tap order, differing only in which address each tap
/// reads", and the second clause is false: `sum_keepdim` accumulates in *slot*
/// order, so rotating which slot holds which token rotates the summation order.
/// Float addition is not associative, so the low bits move.
///
/// **That makes the rotating arm a different summation order, not a defect —
/// established by measurement, not by assumption** (issue #141, `DESIGN.md`
/// §10.2g). §2.3.5a names CPU-backend parity as the only load-bearing
/// discriminator between a legitimate reduction-order change and a computational
/// bug, and the rotating arm passes it: exact against an independent f64
/// reference on the reduction it reorders, and its LFM2 error does not grow over
/// a full generation. So it is an axis in the shape `AttnImpl::Sdpa` is
/// (§2.3.8c) — the digests move, deliberately, and the text does not.
///
/// - [`ConvState::SlidingRing`] keeps the window contiguous and in the shuffle's
///   own slot order, so its output is **bit-identical** and it costs a periodic
///   compaction — §10.2a's own second fallback ("a doubled buffer"), generalised
///   with `slack` as how much doubling is paid for.
/// - [`ConvState::RotatingRing`] is §10.2a as specified. It never compacts, so
///   its dispatch count is **constant** and its write offsets take one of
///   `l_cache + k` fixed values — both of which the sliding arm gives up
///   (§10.2e). It moves the digests.
///
/// Neither dominates: the sliding arm buys bit-identity with `slack` bytes and a
/// non-constant dispatch count, the rotating arm buys a constant count and fixed
/// offsets with a moved digest. `Shuffle` remains the default, so an
/// unconfigured caller keeps the path every recorded digest belongs to.
///
/// # Why `k` is a parameter and not a constant
///
/// §16 6b: the ring's resident footprint crosses the snapshot's at
/// `K = l_cache`, so K is a real memory decision once a speculative scheme
/// (#89) can supply an acceptance-rate distribution to size it against — and
/// until then any specific `K > 0` is a number nobody measured. `K = 0` is the
/// default, so the live window is exactly `l_cache` wide and the mechanism costs
/// no resident bytes beyond `slack`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvState {
    /// `narrow` + `Tensor::cat` — reallocates and copies the whole state every
    /// token (`DESIGN.md` §6.1). The correctness bar the rings are compared
    /// against, and the arm every recorded digest belongs to.
    #[default]
    Shuffle,
    /// A sliding write index into an `l_cache + k + slack`-wide buffer
    /// (`DESIGN.md` §10.2a, §10.2b). The `cat` disappears: the write becomes one
    /// in-place `slice_set`, and the read is a `narrow` that never wraps.
    ///
    /// Bit-identical to `Shuffle`, because the live window is always `l_cache`
    /// contiguous slots in the shuffle's own order. Costs a compaction every
    /// `slack` tokens, which makes the per-token dispatch count non-constant.
    SlidingRing { k: usize, slack: usize },
    /// A rotating write index into an `l_cache + k`-wide buffer — §10.2a as
    /// originally specified.
    ///
    /// No slack, no compaction, and the write lands at one of `l_cache + k`
    /// fixed offsets, so the per-token dispatch count is constant. **It moves
    /// the digests**, because the summation order rotates with the slot the
    /// newest token occupies; that is a reduction-order change and not a bug —
    /// see this enum's note, and `DESIGN.md` §10.2g for the discriminator runs.
    RotatingRing { k: usize },
}

/// Slack slots when `--conv-state ring` is given without one.
///
/// 16 puts the amortised wrap cost at one extra copy per 17 tokens — 0.06
/// dispatches per conv layer per token against the 2 the shuffle pays — while
/// costing 16 × 2048 × 2 B × 22 = 1.375 MiB of resident state. Past ~64 the
/// curve is flat and the memory is not.
const DEFAULT_RING_SLACK: usize = 16;

impl ConvState {
    /// Extra history slots beyond the live window.
    fn history(&self) -> usize {
        match self {
            ConvState::Shuffle => 0,
            ConvState::SlidingRing { k, .. } | ConvState::RotatingRing { k } => *k,
        }
    }

    /// Total slots allocated: the live window, the speculative history, and —
    /// for the sliding arm only — the slack the window slides through before it
    /// must be compacted. The rotating arm needs none, which is its point.
    fn width(&self, l_cache: usize) -> usize {
        match self {
            ConvState::Shuffle => l_cache,
            ConvState::SlidingRing { k, slack } => l_cache + k + slack,
            ConvState::RotatingRing { k } => l_cache + k,
        }
    }

    /// Parse a harness flag.
    ///
    /// `shuffle` | `ring[:<K>[:<slack>]]` (the sliding arm; `ring` is kept as
    /// the spelling every recorded #141 artifact uses) | `rotating[:<K>]`.
    ///
    /// Lives here rather than in each harness so the harnesses that carry the
    /// flag cannot drift apart on what a spelling means -- the shape §8.1b
    /// argues for, applied to a flag instead of a kernel name.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        if s == "shuffle" {
            return Ok(ConvState::Shuffle);
        }
        let num = |v: &str, what: &str| {
            v.parse::<usize>()
                .map_err(|_| format!("--conv-state: {what} must be an integer, got {v:?}"))
        };
        // Split `<name>[:<field>...]` once, so both arms share one parse of the
        // field list and cannot disagree about how `K` is spelled.
        fn fields<'a>(rest: &'a str, s: &str) -> std::result::Result<Vec<&'a str>, String> {
            match rest {
                "" => Ok(vec![]),
                r => Ok(r
                    .strip_prefix(':')
                    .ok_or_else(|| format!("--conv-state: expected `<arm>:...`, got {s:?}"))?
                    .split(':')
                    .collect()),
            }
        }

        if let Some(rest) = s.strip_prefix("rotating") {
            let parts = fields(rest, s)?;
            let k = match parts.as_slice() {
                [] => 0,
                [k] => num(k, "K")?,
                // No `slack` field: the rotating window never slides, so slack
                // has no meaning here and accepting it silently would let a
                // caller believe they had configured something.
                _ => {
                    return Err(format!(
                        "--conv-state: `rotating` takes at most `:<K>` (it has no slack \
                         -- the window rotates in place), got {s:?}"
                    ))
                }
            };
            return Ok(ConvState::RotatingRing { k });
        }

        let Some(rest) = s.strip_prefix("ring") else {
            return Err(format!(
                "--conv-state must be `shuffle`, `ring`, `ring:<K>`, \
                 `ring:<K>:<slack>`, `rotating` or `rotating:<K>`, got {s:?}"
            ));
        };
        let parts = fields(rest, s)?;
        let (k, slack) = match parts.as_slice() {
            [] => (0, DEFAULT_RING_SLACK),
            [k] => (num(k, "K")?, DEFAULT_RING_SLACK),
            [k, slack] => (num(k, "K")?, num(slack, "slack")?),
            _ => return Err(format!("--conv-state: too many fields in {s:?}")),
        };
        if slack == 0 {
            // A zero-slack window cannot advance, so it would compact on every
            // token -- strictly worse than the shuffle it replaces. `rotating`
            // is the arm for "no slack at all", and it is spelled separately.
            return Err("--conv-state: slack must be >= 1".to_string());
        }
        Ok(ConvState::SlidingRing { k, slack })
    }
}

/// Which attention implementation the decode path takes.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms: it selects a kernel at
/// model-construction time and reaches the GPU as nothing at all. Both arms are
/// compiled, so the A/B is free and a regression is one field away — the same
/// discipline as `ParamStyle` and `ArenaLayout`.
///
/// This is deliberately *not* a `use_flash_attn`-style bool. A third arm
/// (`call_sdpa_vector_2pass`, the chunked variant) is a later issue, and a bool
/// would have to be widened into an enum at that point anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttnImpl {
    /// `repeat_kv` + F32 upcast + `matmul` + `softmax_last_dim`.
    ///
    /// The default, and the correctness bar every other arm is compared
    /// against. Works on every backend.
    #[default]
    Generic,
    /// `candle_nn::ops::sdpa`, which is GQA-native and accumulates in F32
    /// internally, so neither `repeat_kv` nor the upcast is needed.
    ///
    /// Metal only, and only for `seq_len == 1`; every other case falls back to
    /// `Generic` rather than failing, so this is safe to set unconditionally.
    Sdpa,
    /// FlashDecoding: attention split over independent contiguous KV chunks,
    /// with an index-ordered combine (`DESIGN.md` §10.4, issue #116).
    ///
    /// **A selectable arm and not a replacement.** §10.4's argument for it is
    /// *structural* rather than measured — at B=1 attention with one
    /// threadgroup per head is 32 threadgroups on a GPU wanting hundreds, and
    /// splitting KV *manufactures* parallelism — and the `kv_len` at which that
    /// starts to pay is **unmeasured**, because every measurement in this
    /// project is below the 2720 ceiling. #61's context curve is what decides
    /// the crossover, and keeping this an arm is what lets it: #71 is the
    /// precedent for the alternative, where three scratch sizing policies were
    /// compiled and **none chosen**, because the axis that separates them could
    /// not be exercised.
    ///
    /// Metal only, `seq_len == 1` only, f16/f32 only. Falls back to `Generic`
    /// on any other case, as `Sdpa` does.
    FlashDecoding,
}

/// How the KV cache grows as decode appends a token.
///
/// A construction-tier axis in `DESIGN.md` §7.1's terms, following `AttnImpl`'s
/// shape: both arms are compiled, the old one is the default, and the A/B is one
/// field apart. §7.2 places it here rather than at compile tier because it
/// changes neither addressing inside a kernel nor registers per thread — the
/// kernel sees a different `n` and a different stride, which are dispatch-tier
/// numbers in a descriptor (§15.2 #8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvAppend {
    /// `Tensor::cat(&[cache, &new], 2)` — reallocate and copy the whole cache,
    /// every layer, every token.
    ///
    /// The default and the correctness bar. `DESIGN.md` §6.2 measures what it
    /// costs: the copy grows with `kv_len`, and at 128k it is ~2 GB per token
    /// on top of reading it.
    #[default]
    Cat,
    /// Write the new token in place at a moving offset inside a pre-allocated
    /// buffer, and read the cache back as a `narrow` of it.
    ///
    /// The `cat` disappears, the copy becomes constant-size, and — the property
    /// §11.1a.1 actually wants — the buffer identity stops changing, because
    /// `narrow` shares storage and only rewrites the layout.
    InPlace,
}

/// How much KV a sequence may accumulate before [`KvAppend::InPlace`] declines.
///
/// **This value is not chosen on evidence, and that is deliberate.** See the
/// `Cache::new_with` documentation for why, and for what would decide it.
pub const DEFAULT_KV_CAPACITY: usize = 4096;

/// What admission needs that `Config` does not already carry (`DESIGN.md`
/// §9.5d).
///
/// Present on `Config` as an `Option`, defaulting to `None`, so every existing
/// caller keeps the path it had without naming it — the shape `AttnImpl`,
/// `KvAppend` and `ConvState` all use (§7.1a).
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    /// Total weight bytes, from the checkpoint header (§5.5).
    ///
    /// Taken as a number rather than read here: the model's weights are loaded
    /// through a `VarBuilder` whose backing store this type cannot see, and a
    /// text-only request legitimately drops the 0.86 GB vision tower (#162).
    /// The caller knows which; this does not.
    ///
    /// **This field is the one admission cannot check on its own, and §9.5m
    /// (#244) is the check that it is right.** A caller passing the language
    /// figure for a load that also brings the tower under-predicts by 0.825 GB,
    /// and §3.5 reports no overrun — so the error is silent. `weight_tolerance`
    /// is what governs when the divergence is reported.
    pub weight_bytes: usize,
    /// Concurrent sequences. Every measurement in this project is B=1 (§13.2).
    pub batch: usize,
    /// Fraction of `recommendedMaxWorkingSetSize` this process may claim.
    ///
    /// **0.65 is §9.5k's table value and not a measurement.** No evidence
    /// chooses it; it is a default that is visible and settable for that
    /// reason.
    pub fraction: f64,
    /// How far `weight_bytes` may diverge from what was allocated before
    /// §9.5m's reconciliation reports it (#244).
    ///
    /// **256 MB by default, and it is a bound on the unaccounted rather than a
    /// round number.** #162 measured the whole attributable non-weight residual
    /// at 42 MB — the RoPE `cos`/`sin` tables at 16.4 MB plus KV, conv state and
    /// activations — so the default clears that by ~6×, while the vision tower
    /// it exists to catch is 3.2× larger than the tolerance. A caller that knows
    /// its own load better may tighten it.
    pub weight_tolerance: usize,
    /// **The longest single forward pass this process will run** — #306.
    ///
    /// `1` is decode and is the default. A caller that prefills a `P`-token
    /// prompt in one pass must set this to `P`; a caller that **chunks** its
    /// prefill sets it to the chunk size, which is the whole mechanism by
    /// which chunking makes a long prompt admissible (§13.2b).
    ///
    /// # Why this is not inferable, and why the default is 1
    ///
    /// **Admission runs once, at the head of `Cache::new_with`, before any
    /// forward pass** (§9.5d) — so no `seq_len` exists to read, and the prompt
    /// length is the caller's knowledge arriving later. Defaulting to
    /// `max_position_embeddings` would refuse essentially every configuration,
    /// which is §8.1g's failure and the one the admitted arm exists to catch;
    /// defaulting to 1 leaves every existing verdict **exactly** where it was
    /// and lets a prefilling caller opt in.
    ///
    /// # The limit, stated rather than hidden
    ///
    /// A caller that prefills and does not set this gets the pre-#306
    /// behaviour — which is the behaviour that admitted a configuration
    /// peaking at **30.5 GB with 77 MB of host memory free** (§13.4c). **What
    /// #306 buys is that the figure is expressible and checked, not that an
    /// unwitting caller is safe.** §9.5m's weight term has the same shape and
    /// the same honest limit: the contract is checkable and the caller still
    /// has to state it.
    pub max_seq_len: usize,
}

impl MemoryBudget {
    /// The fraction §9.5k's residual table is computed at.
    ///
    /// Duplicated from `admission::DEFAULT_BUDGET_FRACTION` rather than
    /// referenced, because this type exists on every backend and that constant
    /// is Metal-only. The two are asserted equal by
    /// `default_fraction_matches_admissions` under the `metal` feature, so the
    /// copy cannot drift — which is §8.1b's checked-registry argument at the
    /// smallest possible scale.
    pub const DEFAULT_FRACTION: f64 = 0.65;

    /// §9.5m's default tolerance for the weight reconciliation (#244).
    ///
    /// Duplicated from `admission::WEIGHT_RECONCILE_TOLERANCE` for the same
    /// reason `DEFAULT_FRACTION` is — this type exists on every backend and
    /// that constant is Metal-only — and asserted equal by
    /// `default_weight_tolerance_matches_admissions` under the `metal` feature,
    /// so the copy cannot drift.
    pub const DEFAULT_WEIGHT_TOLERANCE: usize = 256 * 1_000_000;

    /// A B=1 budget at §9.5k's fraction, for a model whose weights are
    /// `weight_bytes`.
    pub fn new(weight_bytes: usize) -> Self {
        Self {
            weight_bytes,
            batch: 1,
            fraction: Self::DEFAULT_FRACTION,
            weight_tolerance: Self::DEFAULT_WEIGHT_TOLERANCE,
            // Decode. #306: see the field's own doc comment for why this is 1
            // and not `max_position_embeddings`.
            max_seq_len: 1,
        }
    }

    /// The same budget, told the longest forward pass it will be asked for.
    ///
    /// A single-shot prefill of `P` tokens passes `P`; a chunked one passes
    /// its **chunk size** (§13.2b, §4.1).
    pub fn with_prefill(mut self, seq_len: usize) -> Self {
        self.max_seq_len = seq_len;
        self
    }
}

/// Report a weight-term divergence §9.5m has detected (#244).
///
/// **Reported on `stderr` rather than returned as an error**, and the choice is
/// argued rather than defaulted. By the time this fires every caller has already
/// loaded the weights — the bytes are spent — so failing the `Cache` build would
/// turn a bookkeeping error into a refusal to run a model that fits, which is a
/// worse outcome than the silence §3.5 currently offers. What was missing is the
/// *report*, and that is what this is.
///
/// **Metal-only, like the check that calls it.** CPU and CUDA compile none of
/// admission (§14.1), so this is gated the same way the rest of it is.
#[cfg(feature = "metal")]
pub fn report_weight_divergence(r: &candle::metal_backend::admission::WeightReconciliation) {
    eprintln!("lfm2: {}", r.describe());
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub use_flash_attn: bool,
    /// Defaults to `Generic`, so `into_config` and every existing caller keep
    /// the path they had without naming it.
    pub attn_impl: AttnImpl,
    /// KV tokens per **page** — the allocation granularity — for
    /// `AttnImpl::FlashDecoding`.
    ///
    /// §10.4 proposes **256** and marks it **UNVERIFIED**. It is a field rather
    /// than a constant for exactly that reason: §10.3d establishes that page
    /// size enters as two dispatch-tier numbers, so 16, 256 and 1024 are one
    /// field apart, which is what *"must not foreclose it"* requires. Verifying
    /// which wins is #61's axis and is not done here.
    pub flash_page_size: usize,
    /// Pages per **chunk** — the `k` of `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk to one granularity **by fiat**;
    /// §9.1d establishes the general form and that a page (allocation) and a
    /// tile (computation) are optimised against disjoint cost functions — a
    /// page wants to be small and a tile wants to be large enough to fill the
    /// machine. **A sweep holding `k = 1` cannot separate a page-size effect
    /// from a tile-size one**, which is what makes this a field.
    ///
    /// 1, which is what §10.4 specifies. What is new is that it is a *stated*
    /// value on a selectable axis rather than an equality welded into a kernel.
    pub flash_pages_per_chunk: usize,
    /// How `AttnImpl::FlashDecoding` sizes the scratch class (`DESIGN.md`
    /// §9.1a, issue #234).
    ///
    /// **This is the wiring §7.1b names as the cheapest missing
    /// prerequisite.** #71 compiled three sizing policies and chose none,
    /// because the axis that separates them is `kv_len` and it was unreachable;
    /// #61 has since taken `kv_len` to 32 801, and in the interval #116 gave
    /// the class its first consumer on the LFM2 path — one that sized its
    /// buffers to the live `kv_len` on every call and read no policy at all. So
    /// the arm in force was `Grow`, selected by an allocation site rather than
    /// by anyone.
    ///
    /// Defaults to [`FlashScratchSizing::Grow`], which is a **statement of what
    /// shipped rather than a choice**: an unconfigured caller allocates exactly
    /// what #116 allocated. §7.1a's rule is that a default is flipped by its
    /// own argued decision, and wiring an axis is not that argument.
    ///
    /// **Inert unless `attn_impl` is `FlashDecoding`**, which is the only path
    /// that allocates from this class — *inert* and *unrecorded* being different
    /// facts, so it is still stated on a `RESULT` line (#241).
    pub flash_scratch_sizing: FlashScratchSizing,
    /// Defaults to `Cat`, for the same reason `attn_impl` defaults to `Generic`.
    pub kv_append: KvAppend,
    /// Whether the KV cache stores integer codes, and at which width
    /// (`DESIGN.md` §10.7, issue #218).
    ///
    /// **Defaults to `Off`, and this issue does not flip it** (§7.1a): every
    /// recorded digest in this repository belongs to the f16 arm, and a
    /// value-changing arm made default would move all of them silently.
    ///
    /// A *construction*-tier axis in §7.1's terms, following `KvAppend`'s
    /// shape: the arms are compiled, the old one is the default, and the A/B is
    /// one field apart. The bit width reaches the kernel as a template
    /// parameter (§7.2 — memory layout is compile-tier), but selecting between
    /// two pre-built instantiations is a pipeline choice.
    pub kv_quant: KvQuant,
    /// Defaults to `Shuffle`, so every existing caller keeps §6.1's path.
    pub conv_state: ConvState,
    /// `DESIGN.md` §9.5's memory budget. `None` — admission off — by default,
    /// per §7.1a's rule that no default is flipped without its own argued
    /// decision. See `Cache::admit_memory_budget`.
    pub memory_budget: Option<MemoryBudget>,
    /// Fuse the SiLU in `Mlp::forward` onto `w1`'s GEMV store (#484's kernel,
    /// wired by #509).
    ///
    /// **Defaults to off**, per §7.1a's rule that a default is flipped by its
    /// own argued decision and that wiring an axis is not that argument. #536
    /// says so in as many words: *"do not flip the default — that needs the
    /// decode-path number"*, which this change does not take.
    ///
    /// Metal, `seq_len == 1`. Every other case takes the eager path, because
    /// this is a decode-path GEMV fusion and prefill is a GEMM.
    ///
    /// **This arm moves the digests by construction** and owes §2.3.5a's
    /// discriminator: SiLU is applied to the f32 accumulator where the
    /// `silu_f16` dispatch it replaces applies it at `T` to the rounded store
    /// (§12.2d). It is not gated on an unchanged digest.
    pub gemv_silu: bool,
    /// Fuse the two residual adds onto their producing GEMV's store (#509's
    /// kernel, wired by #509).
    ///
    /// The producers are `out_proj`, `conv.out_proj` and `w2` — 60 dispatches
    /// per token (§12.2e). **Defaults to off**, for `gemv_silu`'s reason.
    ///
    /// **A dispatch-tier switch and nothing else, since #516.** It used to
    /// additionally require the metallib to carry
    /// `-DLLOOM_509_RESIDUAL_EPILOGUE`, so this run-time field selected a
    /// kernel whose existence was decided at build time — two sources of truth
    /// for one decision, failing at pipeline load with a Metal error naming a
    /// mangled kernel rather than the flag. The `_residual` names are now
    /// instantiated unconditionally, exactly as `gemv_silu`'s are.
    ///
    /// **This arm moves the digests by construction** (§12.2h): the add is at
    /// the f32 accumulator where `badd_f16` adds at `T` on the rounded store.
    pub gemv_residual: bool,
    /// Which layers' **input residual** to capture into [`Cache::taps`], for a
    /// hidden-state-typed drafter (`DESIGN.md` §10.2m, issue #363).
    ///
    /// # Why a tap is needed at all, which is not obvious from the seam
    ///
    /// §10.2l measured `LFM2.5-2.6B-DSpark` and found it has **no
    /// `embed_tokens` and no `lm_head`**: `fc.weight` is `[2048, 10240]` where
    /// `10240 = 5 × 2048`, so it consumes **five concatenated 2048-d hidden
    /// states** from this target's layers. It cannot be run from tokens at all,
    /// and [`Model::verify_step`]'s seam is `(last, proposed) -> Vec<u32>` —
    /// tokens in, tokens out. **The tap is the prerequisite**, and neither
    /// `forward` nor `forward_all` returned one.
    ///
    /// # Which representation, and why this one rather than another
    ///
    /// The **layer-input residual**, captured at layer entry *before* the block
    /// runs — not post-attention, not post-MLP. The witness is mainline
    /// llama.cpp PR #27383 (`src/models/lfm2.cpp`), which writes
    /// `t_layer_inp[il] = cur` at exactly that point. §10.2l item 1 recorded the
    /// representation as unspecifiable from public prose; §10.2m corrects that
    /// from the reference implementation, which is a source that section did not
    /// have.
    ///
    /// **Empty by default**, which is what makes this free when unused: the
    /// layer loop tests `is_empty()` on a slice and captures nothing. §7.1a's
    /// rule is that a default is flipped by its own argued decision, and there
    /// is no decode-path consumer yet to make that argument.
    ///
    /// For the 2.6B target the drafter's `dflash_config.target_layer_ids` is
    /// `[2, 9, 17, 21, 27]` against `num_target_layers: 30`, which is this
    /// model's layer count (§5.11). **The indices are the caller's**, not a
    /// constant here: they are a property of the *drafter* checkpoint, and
    /// welding them into the target would be the conflation §10.1 renames the
    /// abstraction to avoid.
    pub tap_layers: Vec<usize>,
    /// The dtype [`Cache::taps`] holds, when `tap_layers` is non-empty.
    ///
    /// **F32, and it is deliberate rather than incidental.** The target's
    /// residual is f16 and `dflash.cpp` upcasts the concat to `GGML_TYPE_F32`.
    /// The target verifies bit-exactly whatever the draft proposes (§10.2j), so
    /// this is **not** a correctness decision — it buys **acceptance rate**,
    /// which is the entire ROI of a speculative scheme (§10.2m). A
    /// lower-precision tap shifts the draft distribution and lowers accept.
    pub tap_dtype: DType,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// One attention layer's K or V, pre-allocated once and written at a moving
/// offset.
///
/// `DESIGN.md` §6.2's `Tensor::cat` reallocates and copies the whole cache every
/// token; this writes only the new token's bytes and hands back a `narrow` of
/// the buffer. Three consequences, and the third is the one §11.1a.1 is after:
///
/// * the `cat` is gone, so the `copy2d` per layer per token drops from 2 to 1;
/// * the surviving `copy2d` is **constant-size** — one token's `[1, 8, 1, 64]`
///   — where the `cat` copied `kv_len` tokens and grew by 64 rows per step;
/// * **the buffer identity stops changing.** `narrow` clones the storage `Arc`
///   and rewrites only the layout, so every token binds the same `MTLBuffer` at
///   the same base. That is the property ICB replay needs and the reason this
///   is worth more than the copy it removes.
///
/// The read is a *view*, never a copy. `call_sdpa_vector` takes `k_l.stride()`
/// and `n = k_shape[2]`, so a narrowed cache is exactly the shape it is written
/// to consume (`candle-metal-kernels/src/kernels/sdpa.rs`); the generic arm's
/// `repeat_kv` goes through `Tensor::cat`, which handles a strided source.
#[derive(Debug, Clone)]
struct KvSlot {
    /// `[b_sz, n_kv_heads, capacity, head_dim]`, allocated on first append.
    ///
    /// `None` until then, because the batch size is not known at `Cache::new`.
    /// This mirrors `candle_nn::kv_cache::Cache`, deliberately: that type is the
    /// same mechanism and was read before this was written (see the note on
    /// `Cache::new_with` for why it is not reused directly).
    all_data: Option<Tensor>,
    /// How many tokens of `all_data` are live. The `narrow` length.
    len: usize,
    capacity: usize,
}

impl KvSlot {
    fn new(capacity: usize) -> Self {
        Self {
            all_data: None,
            len: 0,
            capacity,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
        self.all_data = None;
    }

    /// Discard the last `count` appended positions.
    ///
    /// `DESIGN.md` §10.2a's KV half of `resolve`: *"`len += n`; discarded bytes
    /// **need not be cleared** — unreachable above `len`"*. The read is
    /// `all_data.narrow(2, 0, self.len)`, so a position above `len` is outside
    /// every view this slot hands out and no kernel can address it.
    ///
    /// **The allocation is deliberately not touched.** Rewinding frees nothing
    /// and must not: stable buffer identity is the property §6.2b exists to
    /// preserve (`buf#69` at the same base every token), and dropping the
    /// buffer on rollback would move it at exactly the moment a speculative
    /// scheme runs most often. A rewind is a length decrement and nothing else,
    /// which is what §10.2a's cost column means by *"a length decrement"*.
    fn rewind(&mut self, count: usize) -> Result<()> {
        if count > self.len {
            candle::bail!(
                "lfm2 KV rewind past the start: {count} > len {}. \
                 `resolve(n)` may only discard positions this speculation appended.",
                self.len
            )
        }
        self.len -= count;
        Ok(())
    }

    /// Append `src` along dim 2 and return the live prefix as a view.
    ///
    /// Returns `Err` rather than reallocating when the capacity is exhausted —
    /// see `Cache::new_with` for why that is a loud failure and not a `Grow`.
    fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(2)?;
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[2] = self.capacity;
            self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }
        let all_data = self.all_data.as_ref().unwrap();
        if self.len + seq_len > self.capacity {
            candle::bail!(
                "lfm2 KV cache exhausted: {} + {} > capacity {}. \
                 Raise `Config::kv_capacity`; see `DESIGN.md` §6.2a on why no policy is chosen.",
                self.len,
                seq_len,
                self.capacity
            )
        }
        // `slice_set` requires both sides contiguous and writes one `copy2d` of
        // exactly `src`'s bytes at `offset * block_size` (`tensor_cat.rs`). The
        // caller has already made `src` contiguous.
        all_data.slice_set(src, 2, self.len)?;
        self.len += seq_len;
        all_data.narrow(2, 0, self.len)
    }
}

/// One attention layer's K or V, stored as integer codes plus a (scale, bias)
/// pair per head vector (`DESIGN.md` §10.7, issue #218).
///
/// **The quantized sibling of [`KvSlot`], and it preserves that type's three
/// properties** (§6.2b): there is no `cat`, the write is constant-size, and the
/// buffer identity does not change — the planes are allocated once at capacity
/// and written at a moving offset, exactly as `all_data` is.
///
/// # The layout, and why the scales are a separate plane
///
/// ```text
/// codes   [B, H, capacity, D]   u8   (int8: 1 code/elem; int4: 2 codes/byte)
/// scales  [B, H, capacity]      f16  one per (head, token) — the group IS the
/// biases  [B, H, capacity]      f16  head vector, since gs == head_dim == 64
/// ```
///
/// §17 item 18(c) and §10.7a: interleaving the scale beside each head vector
/// breaks `sdpa_vector`'s contiguous walk, which §9.1d records as a
/// *requirement* rather than a preference — `head_dim × dtype` is one 128 B
/// cache line at f16, and a dim-outer order scatters a token's vector across 64
/// lines. A parallel plane preserves it.
///
/// # Determinism
///
/// A scale is computed once at write time and only read thereafter, so it sits
/// **outside** the per-token reduction §2.3.3 #1 governs — §16 #6 and §10.7a
/// both record that **int8 KV needs no determinism variant tier**, and that is
/// why this type adds no arm to `.bench/configurations.md`'s determinism
/// column. The digests move because the *values* differ (a tier-2 arm in
/// §2.3.3c's terms), not because the order does.
#[derive(Debug, Clone)]
struct QuantKvSlot {
    /// `[B, H, capacity, D]` codes, packed. `None` until the first append,
    /// because the batch size is not known at `Cache::new` — `KvSlot`'s reason.
    codes: Option<Tensor>,
    scales: Option<Tensor>,
    biases: Option<Tensor>,
    /// The `SdpaQuantParams` block, rebuilt per step because `n` moves.
    ///
    /// A caller-owned buffer rather than `setBytes` scalars, so this family is
    /// ICB-encodable from the first commit (`DESIGN.md` §3.7c, §11.3l finding
    /// 3): a params buffer allocated per call makes ICB coverage zero, because
    /// a command binds a buffer by identity at encode time.
    params: Option<Tensor>,
    len: usize,
    capacity: usize,
    quant: KvQuant,
}

impl QuantKvSlot {
    fn new(capacity: usize, quant: KvQuant) -> Self {
        Self {
            codes: None,
            scales: None,
            biases: None,
            params: None,
            len: 0,
            capacity,
            quant,
        }
    }

    fn reset(&mut self) {
        self.len = 0;
        self.codes = None;
        self.scales = None;
        self.biases = None;
        self.params = None;
    }

    /// Discard the last `count` appended positions.
    ///
    /// `KvSlot::rewind`'s contract exactly (§10.2a): a length decrement and
    /// nothing else. The discarded codes sit above `len` and no kernel can
    /// address them, and the allocation is deliberately not touched.
    fn rewind(&mut self, count: usize) -> Result<()> {
        if count > self.len {
            candle::bail!(
                "lfm2 quantized KV rewind past the start: {count} > len {}",
                self.len
            )
        }
        self.len -= count;
        Ok(())
    }

    /// Allocate the three planes on first use.
    fn ensure(
        &mut self,
        b_sz: usize,
        n_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<()> {
        if self.codes.is_some() {
            return Ok(());
        }
        let code_bytes = self
            .quant
            .code_bytes(b_sz * n_heads * self.capacity * head_dim)
            .ok_or_else(|| candle::Error::msg("QuantKvSlot::ensure on the Off arm"))?;
        // A u8 plane rather than a typed one: the packing is the kernel's, and
        // a `Tensor` of the code width does not exist at int4 anyway.
        self.codes = Some(Tensor::zeros(code_bytes, DType::U8, device)?);
        let scale_els = b_sz * n_heads * self.capacity;
        self.scales = Some(Tensor::zeros(scale_els, DType::F16, device)?);
        self.biases = Some(Tensor::zeros(scale_els, DType::F16, device)?);
        Ok(())
    }

    /// Quantize `src` and append it, advancing `len`.
    fn append(&mut self, src: &Tensor) -> Result<()> {
        let (b_sz, n_heads, seq_len, head_dim) = src.dims4()?;
        if self.len + seq_len > self.capacity {
            candle::bail!(
                "lfm2 quantized KV cache exhausted: {} + {} > capacity {}",
                self.len,
                seq_len,
                self.capacity
            )
        }
        self.ensure(b_sz, n_heads, head_dim, src.device())?;

        let bits = match self.quant {
            KvQuant::Off => candle::bail!("QuantKvSlot::append on the Off arm"),
            KvQuant::Int8 => 8usize,
            KvQuant::Int4 => 4usize,
        };
        let _ = bits;
        // **The params block is built by the metal side, not hand-packed
        // here.** An earlier draft wrote the struct as a `&[u32]` with manual
        // padding, which is precisely the class §11.3k's layout check exists to
        // catch: a field at the wrong offset does not crash, it reads a
        // well-formed number from the wrong place. `write_params` fills a
        // caller-owned buffer through the same `SdpaQuantWriteParams` the
        // kernel reads and the layout test pins.
        candle_nn::ops::quantize_kv_into(
            src,
            self.codes.as_ref().unwrap(),
            self.scales.as_ref().unwrap(),
            self.biases.as_ref().unwrap(),
            self.capacity,
            self.len,
            self.quant,
        )?;
        self.len += seq_len;
        Ok(())
    }
}

/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    // KV cache for attention layers: (key, value) per layer
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // The `KvAppend::InPlace` arm's storage, one (K, V) pair per layer.
    // Held beside `kvs` rather than replacing it so that both arms stay
    // compiled and the unselected one allocates nothing.
    kv_slots: Vec<(KvSlot, KvSlot)>,
    /// The `KvQuant` arm's storage, one (K, V) pair per layer.
    ///
    /// Held beside `kv_slots` rather than replacing it, so both arms stay
    /// compiled and the unselected one allocates nothing -- `kv_slots`' own
    /// argument, one axis further out.
    quant_slots: Vec<(QuantKvSlot, QuantKvSlot)>,
    kv_append: KvAppend,
    kv_quant: KvQuant,
    /// The largest `kv_len` this cache admits, in tokens.
    ///
    /// **What `FlashScratchSizing::Reserve` reserves against** (#234), and the
    /// reason it lives here rather than on `Config`: the reachable ceiling is
    /// `kv_capacity`, which is a `Cache::new_with` argument and defaults to
    /// `DEFAULT_KV_CAPACITY` (4096) — where `Config::max_position_embeddings`
    /// is **128000**, the model's positional ceiling and not this run's. A
    /// `Reserve` bounded by the second would reserve 31× what any step of this
    /// cache can reach, which is not the policy's cost but a mis-stated bound
    /// on it, and §9.1a's own table is what a reader would then compare it
    /// against.
    ///
    /// Held for **both** `KvAppend` arms, though only `InPlace` has a
    /// per-slot capacity: `Cat` reallocates and has no ceiling of its own, so
    /// its reserve bound is this same configured number rather than something
    /// derived from a slot that does not exist. That keeps the two arms
    /// comparable on this axis, which is what an A/B across them needs.
    kv_capacity: usize,
    // Conv state cache for convolution layers
    conv_states: Vec<Option<Tensor>>,
    /// The ring's rotating write index, in slots.
    ///
    /// **Deliberately here and not in a params struct** (`DESIGN.md` §10.2c,
    /// §10.5, §11.3): it is per-sequence *conv state*, so it migrates and evicts
    /// with the sequence exactly as the bytes do, and putting it in a struct
    /// named for KV is the conflation §10.1 renames the abstraction to avoid.
    ///
    /// One index for all 22 conv layers rather than one each: every layer is
    /// written exactly once per token, so their phases are equal at every point
    /// a layer reads them. Per-layer indices would be 22 copies of one number.
    conv_phase: usize,
    /// Set by `Model::forward` when the write slot has run out of slack, so the
    /// conv layers compact their window before writing this token.
    ///
    /// **The decision lives in exactly one place on purpose.** An earlier draft
    /// had `Model::forward` wrap the index *and* the layers test `phase >=
    /// width` — two owners of one decision, which is a shape that cannot be
    /// right: the wrap made the layers' test unreachable, so the compaction
    /// never ran and the window silently read stale slots from the 17th token
    /// on. Caught by the bitwise parity test, which is what it is for.
    conv_compact: bool,
    /// The conv arm and window width, copied from `Config` so that
    /// [`Cache::advance`] and [`Cache::resolve`] can check and move the phase
    /// without a `Model` in hand.
    ///
    /// Duplicated rather than referenced because a `Cache` outlives no `Model`
    /// and holding a borrow would infect every caller's lifetimes; the pair is
    /// read once at construction and neither is mutable, so they cannot drift
    /// from the arm the layers actually take.
    conv_state: ConvState,
    l_cache: usize,
    /// Set between [`Cache::advance`] and [`Cache::resolve`]: this pass is a
    /// speculative verify, so the conv layers must **ring-write** their
    /// `seq_len` positions rather than rebuilding the state from them.
    ///
    /// # Why this needs a flag rather than being inferred from `seq_len > 1`
    ///
    /// `seq_len > 1` has two callers with opposite requirements. A **prefill**
    /// wants the existing rebuild: it is establishing history from nothing, the
    /// pre-pass state is meaningless, and `Conv1d` over the whole prompt is the
    /// efficient way to do it. A **verify pass** wants the opposite: the
    /// pre-pass state is the sequence's real history, and the positions it
    /// writes must be individually discardable, which is what the ring's slots
    /// are for (§10.2a).
    ///
    /// Inferring from `seq_len` cannot separate them, and inferring from
    /// "`conv_states` is already populated" would silently change what a
    /// **turn-boundary prefill** does — §10.9 measures that as a `seq_len > 1`
    /// pass onto a warm cache, which is the prefill case and must keep the
    /// rebuild. So the caller says which it is, and `advance`/`resolve` are the
    /// only things that set it.
    speculative: bool,
    /// The layer-input residuals captured by this pass, one per entry of
    /// `Config::tap_layers` and **in that slice's order** (`DESIGN.md` §10.2m,
    /// issue #363).
    ///
    /// Each is `[b_sz, seq_len, hidden_size]` at `Config::tap_dtype`. The
    /// consumer concatenates along the last dim to reach the drafter's
    /// `fc.weight [2048, 10240]` — so **order is load-bearing**, not incidental:
    /// `fc` expects `[H(l₁); …; H(l₅)]` in the drafter's own
    /// `target_layer_ids` order, and a permuted concat pairs every slice with
    /// the wrong 2048-column block of `W_c`. That is the shape §10.2h found in
    /// the rotating ring's weight permutation — algebraically valid, silently
    /// wrong — so the order is asserted by a test rather than left to a
    /// convention.
    ///
    /// **Overwritten per pass, not accumulated.** A tap is a property of *this*
    /// forward pass; a speculative window that is rolled back must not leave a
    /// stale tap behind for the next one to read, and the cheapest way to
    /// guarantee that is for every pass that taps to replace the vector whole.
    /// [`Cache::take_taps`] is the consumer's accessor and drains it, so a
    /// second read of one pass's taps is impossible by construction rather than
    /// by discipline.
    ///
    /// Empty whenever `tap_layers` is empty, which is the default — so a run
    /// that does not tap allocates nothing here (§9.1's memory classes are
    /// unmoved).
    taps: Vec<Option<Tensor>>,
    /// `Config::tap_layers`, copied so the layer loop can test it without a
    /// `Model` in hand — `conv_state`/`l_cache`'s argument, one field further
    /// out.
    tap_layers: Vec<usize>,
    /// `Config::tap_dtype`, copied for `tap_layers`' reason.
    tap_dtype: DType,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        Self::new_with(use_kv_cache, dtype, config, device, DEFAULT_KV_CAPACITY)
    }

    /// As [`Cache::new`], with the [`KvAppend::InPlace`] capacity given
    /// explicitly.
    ///
    /// # The capacity is not decided, and this says so rather than picking
    ///
    /// A moving write offset needs a bound on how far it may move. `DESIGN.md`
    /// §9.1a faced the identical question for the scratch class and built
    /// `Reserve`, `Grow` and `Bucket` while **choosing none**, because the axis
    /// that separates them is `kv_len` and the largest value this project has
    /// ever recorded is **2720** against a 128k target (§13.2). This issue hits
    /// the same wall, and the same answer is the honest one.
    ///
    /// What the three policies would cost here, computed from §5.6's 16 KiB per
    /// token across all 8 attention layers at B=1, f16:
    ///
    /// | `kv_len` | `Reserve` (128k) | `Grow` | `Bucket` |
    /// |---|---|---|---|
    /// | 2,720 — the largest ever measured | 2048 MiB | 42.5 MiB | 64 MiB |
    /// | 32,768 | 2048 MiB | 512 MiB | 512 MiB |
    /// | 131,072 | 2048 MiB | 2048 MiB | 2048 MiB |
    ///
    /// They converge at the target and differ by ~48× in the regime we can
    /// actually reach — which is precisely the shape that makes a measurement
    /// here uninformative about the choice.
    ///
    /// **What is chosen instead is a parameter with a default and a loud
    /// failure.** `DEFAULT_KV_CAPACITY` is 4096: above the 2720 ceiling with
    /// headroom, 64 MiB against a 5.2 GB pool (§6.3b), and small enough that
    /// nobody mistakes it for a considered long-context answer. Exceeding it is
    /// an error, not a silent reallocation — a `Grow` arm would put the
    /// `Tensor::cat` this change exists to remove back on the path at exactly
    /// the moment the cache is largest, and would move the buffer identity that
    /// §11.1a.1 wants stable. Failing loudly keeps both properties true or
    /// visibly absent, which is `DESIGN.md` §2.4's rule about instruments that
    /// cannot be shown to have engaged.
    ///
    /// **What would decide it** is a context-length curve — issue #61, and
    /// `DESIGN.md` §13.2 records that no preset has evidence for the same
    /// reason. Concretely: ms/token and peak footprint at `kv_len` across
    /// 2k/8k/32k/128k, with the three policies as arms. Until that exists, any
    /// value here is a guess, and this one is labelled as one.
    ///
    /// # Why not `candle_nn::kv_cache::KvCache`
    ///
    /// It is the same mechanism — `slice_set` at a moving offset, `narrow` to
    /// read — and it was read before this was written; seven models in
    /// `candle-transformers` already use it. It is not reused here because its
    /// `append` implements exactly the `Grow` arm this function declines: on
    /// exhaustion it does `Tensor::cat` with a fresh block
    /// (`candle-nn/src/kv_cache.rs`), which reintroduces the reallocation and
    /// moves the buffer identity. Taking `KvCache` would have been the smaller
    /// diff and would have silently chosen the policy this section exists to
    /// leave open. Making that a *choice* rather than an inheritance is the
    /// reason for the local type, and it is a difference of about forty lines.
    pub fn new_with(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
        kv_capacity: usize,
    ) -> Result<Self> {
        // `DESIGN.md` §9.5d: admission at configuration time, BEFORE the first
        // allocation. It is `Ok(())` unless a caller has asked for it -- see
        // `admit_memory_budget` for why it is opt-in and what it costs.
        //
        // **The ordering is the point.** The RoPE `cos`/`sin` tables below are
        // built from `max_position_embeddings` through `Tensor` ops, i.e.
        // through the pool: at the shipped 128000 that is 16.4 MB resident and
        // ~49 MB transient (§9.5k). They are not one of §9.1's five classes,
        // and an admission check placed after them would not see them. Running
        // first means the refusal happens before any of it is allocated.
        Self::admit_memory_budget(config, device, kv_capacity)?;

        // #363's tap indices, checked here rather than left to `contains`.
        //
        // `target_layer_ids` is a property of the **drafter** checkpoint and the
        // layer count a property of the **target**, so a mismatched pair is the
        // first thing a consumer hits — DSpark's `[2, 9, 17, 21, 27]` against a
        // target that is not the 30-layer 2.6B. Left silent, an out-of-range
        // index captures nothing and the consumer receives four slices where
        // `fc.weight [2048, 10240]` expects five, failing as a shape error far
        // from its cause. `DEFAULT_KV_CAPACITY`'s own note is the precedent:
        // fail loudly at configuration rather than plausibly at use.
        for &l in &config.tap_layers {
            if l >= config.num_hidden_layers {
                candle::bail!(
                    "lfm2 tap_layers: layer {l} is past the last layer ({}). \
                     The tap indices belong to the drafter checkpoint \
                     (`dflash_config.target_layer_ids`) and must be indices \
                     into THIS target's stack (DESIGN.md 10.2m).",
                    config.num_hidden_layers - 1
                )
            }
        }

        let theta = calculate_default_inv_freq(config);
        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            kv_slots: (0..num_layers)
                .map(|_| (KvSlot::new(kv_capacity), KvSlot::new(kv_capacity)))
                .collect(),
            quant_slots: (0..num_layers)
                .map(|_| {
                    (
                        QuantKvSlot::new(kv_capacity, config.kv_quant),
                        QuantKvSlot::new(kv_capacity, config.kv_quant),
                    )
                })
                .collect(),
            kv_append: config.kv_append,
            kv_quant: config.kv_quant,
            kv_capacity,
            conv_states: vec![None; num_layers],
            conv_phase: 0,
            conv_compact: false,
            conv_state: config.conv_state,
            l_cache: config.conv_l_cache,
            speculative: false,
            // Reserved to the tap count rather than left to grow: the layer
            // loop pushes exactly `tap_layers.len()` entries per tapping pass,
            // so the capacity is known and the push cannot reallocate on the
            // per-token path (§15.2 #10 is about lookup, but a realloc in the
            // layer loop is the same shape).
            taps: Vec::with_capacity(config.tap_layers.len()),
            tap_layers: config.tap_layers.clone(),
            tap_dtype: config.tap_dtype,
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Take this pass's layer-input residuals, leaving the cache empty
    /// (`DESIGN.md` §10.2m, issue #363).
    ///
    /// One entry per [`Config::tap_layers`] index, **in that slice's order**,
    /// each `[b_sz, seq_len, hidden_size]` at [`Config::tap_dtype`]. Empty
    /// unless the preceding forward pass ran with a non-empty `tap_layers`.
    ///
    /// **It drains rather than borrows**, which is the property that matters:
    /// a tap belongs to the pass that produced it, so reading one pass's taps
    /// twice — or reading a rolled-back speculative window's taps as though
    /// they described the committed state — is impossible by construction
    /// rather than by the caller remembering. The alternative (a `&[Tensor]`
    /// accessor) makes both of those a caller discipline, and §6.7 L4's whole
    /// lesson is that a discipline nothing enforces is a defect waiting for a
    /// second consumer.
    /// Returns empty if the preceding pass did not tap. **Every slot is filled
    /// or none is**: the indices are range-checked at construction, so a slot
    /// can only be empty if no pass has run, and a partially-filled vector
    /// would mean a layer index the loop never reached — which the constructor
    /// has already refused.
    pub fn take_taps(&mut self) -> Vec<Tensor> {
        let slots = std::mem::take(&mut self.taps);
        slots.into_iter().flatten().collect()
    }

    /// `DESIGN.md` §9.5's admission check: predict the peak, refuse before
    /// allocating, and install §9.5k's derived residual as a runtime cap.
    ///
    /// # Opt-in, and why
    ///
    /// Returns `Ok(())` unless [`Config::memory_budget`] is set. Three reasons
    /// it is not on by default, and the first is this project's own rule:
    ///
    /// * **§7.1a: no default is flipped without its own argued decision.** A
    ///   budget that refuses is a behaviour change for every consumer of this
    ///   model, and the evidence for a *fraction* does not exist — 0.65 is
    ///   §9.5k's table value, not a measurement.
    /// * **The denominator is only meaningful on Metal.**
    ///   `recommendedMaxWorkingSetSize` has no CPU or CUDA analogue, so a
    ///   non-Metal device is admitted unconditionally rather than being given a
    ///   number that means nothing (§9.5c).
    /// * **§9.5f: admission is necessary and not sufficient.** Every reachable
    ///   B=1 configuration predicts under 8.71 GB and the machine died with
    ///   54.91 GB wired, so this would have refused nothing on the run that
    ///   crashed. Making it the default would suggest a guarantee it does not
    ///   give.
    ///
    /// # What it costs
    ///
    /// Once per process, and it is six multiplies and a comparison against a
    /// figure the device already exposes. **Free by inspection: it is not on
    /// any per-token path** (§9.5e). The residual cap it installs is one branch
    /// on counters the pool already maintains, and it calls into Metal nowhere.
    #[cfg(not(feature = "metal"))]
    fn admit_memory_budget(_: &Config, _: &Device, _: usize) -> Result<()> {
        // No Metal, no `recommendedMaxWorkingSetSize`, nothing to check
        // against. The CPU and CUDA backends compile none of the above, which
        // is the constraint upstreaming imposes (§14.1).
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn admit_memory_budget(config: &Config, device: &Device, kv_capacity: usize) -> Result<()> {
        use candle::metal_backend::admission;

        let Some(budget) = config.memory_budget else {
            return Ok(());
        };
        // Non-Metal device with the metal feature built in: no denominator
        // exists, so there is nothing to check against. Silent rather than an
        // error -- a CPU run is a legitimate thing to do and refusing it here
        // would be an unrelated regression.
        let Ok(metal) = device.as_metal_device() else {
            return Ok(());
        };

        let mut b = admission::Budget::new(
            budget.weight_bytes,
            kv_capacity,
            config.max_position_embeddings,
        );
        b.batch = budget.batch;
        b.fraction = budget.fraction;
        // #306: the activation class is a function of the longest forward pass,
        // not a constant. Taken from the caller because admission runs before
        // any pass exists to measure -- see `MemoryBudget::max_seq_len`.
        b.max_seq_len = budget.max_seq_len;
        // The geometry that class is computed from, taken from `Config` rather
        // than from `PrefillGeometry::default()` so a model with different
        // dimensions is sized correctly and cannot silently inherit LFM2's.
        b.prefill_geometry = admission::PrefillGeometry {
            attention_layers: config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            conv_layers: config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::Conv)
                .count(),
            query_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            hidden: config.hidden_size,
            intermediate: config.intermediate_size,
        };
        // §10.2g: `Shuffle` and `RotatingRing` hold exactly `l_cache` columns
        // (264 KiB); `SlidingRing` holds the slack it slides through as well,
        // which is 1.63 MiB at the default. Taken from `ConvState::width`
        // rather than recomputed, so the budget cannot disagree with the
        // allocation about how wide the state is -- the hand-sync §8.1b exists
        // to remove.
        b.conv_state_bytes = admission::CONV_STATE_BYTES
            * config.conv_state.width(config.conv_l_cache)
            / config.conv_l_cache;
        // The RoPE `cos`/`sin` tables (§9.5k's "sixth allocation the five
        // classes do not name"). Two f16 tables of
        // `max_position_embeddings x head_dim/2`, built through the pool in
        // this very function. Accounted rather than excluded: they are
        // negligible against the budget at 16.4 MB, and they **scale with the
        // context axis #161 sweeps**, which is the reason §9.5k records them
        // instead of folding them in silently.
        let rope_resident = 2 * config.max_position_embeddings * (config.head_dim() / 2) * 2;
        // The transient peak is larger than the resident figure -- an f32
        // `idx_theta` plus f32 `cos`/`sin` before the cast, ~49 MB at 128000 --
        // and it is the one that has to fit, since it is live inside this call.
        let rope_transient = 3 * config.max_position_embeddings * (config.head_dim() / 2) * 4;
        let rope = rope_resident + rope_transient;

        let admission = b.admit(metal.device().recommended_max_working_set_size());
        // The RoPE tables come out of the residual rather than being added to
        // the predicted classes: they are exactly the kind of allocation the
        // residual exists to cover (§9.5k -- "an allocation nothing planned"),
        // and putting them in `Footprint` would make its rows stop matching
        // §9.1's five classes.
        if !admission.fits || admission.residual < rope {
            candle::bail!(
                "{}\n  (plus {:.2} MB for the RoPE cos/sin tables at \
                 max_position_embeddings={}, which are built before the first \
                 token and are not one of §9.1's five classes)",
                admission.describe(),
                rope as f64 / 1e6,
                config.max_position_embeddings,
            );
        }

        // §9.5m (#244): check the weight term against what was ACTUALLY
        // allocated, which is the check §3.5 says does not exist.
        //
        // **`weight_bytes` is a byte count the caller passes**, and its own doc
        // comment says a text-only request drops the vision tower -- so
        // admission is correct by construction only if the caller knows which,
        // and until now nothing told it. Over-prediction fails loudly by
        // refusing something that would have fit; **under-prediction fails
        // silently**, because §3.5's Metal has no dependency analysis and
        // nothing anywhere reports an overrun.
        //
        // The reading is the pool's own counters, deliberately from outside the
        // model's arithmetic -- #162 declined to cite the harness's computed
        // `weight_bytes` for this, since a prediction agreeing with itself is
        // not an observation.
        let f = &admission.footprint;
        let (_, private) = metal.pool_occupancy();
        let reconciliation = admission::reconcile_weights(
            budget.weight_bytes,
            &private,
            // The non-weight classes `private_buffers` also serves (§9.5k, and
            // §6.3a's correction that it is not the weight pool but where both
            // weights and intermediates live).
            f.scratch,
            budget.weight_tolerance,
        );
        // **Reported, not refused, and the asymmetry is deliberate.** Refusing
        // here would turn a caller's bookkeeping error into a failure to build
        // a model that fits -- and this check runs where every caller has
        // already loaded the weights, so the bytes are spent either way.
        // Reporting is what §3.5 lacks; refusing is a policy nobody has argued.
        //
        // A caller that builds the `Cache` BEFORE loading the model sees an
        // empty pool. `reconcile_weights` floors at zero there, so that case
        // reports nothing rather than reporting a spurious under-prediction --
        // the direction that matters cannot be manufactured by call order.
        if reconciliation.under_predicted() {
            crate::models::lfm2::report_weight_divergence(&reconciliation);
        }

        // Install §9.5k's derived cap. `planned` differs per pool because the
        // two hold different classes and `live_bytes` already contains them:
        // the KV reserve is served from `buffers`, the weights from
        // `private_buffers`. The arena is in neither -- `install_arena` calls
        // the raw device.
        metal.set_residual_cap(
            admission.residual,
            /* shared  */ f.kv + f.conv,
            /* private */ f.weights + f.scratch,
        );
        Ok(())
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    /// Drop every sequence-state tensor, returning the cache to its pre-prefill
    /// state.
    ///
    /// Both KV arms are reset, whichever is selected. For `InPlace` that means
    /// the buffer is dropped rather than the length being rewound: a new
    /// sequence may have a different batch size, so keeping the allocation
    /// would pin `b_sz` from the previous one.
    pub fn clear(&mut self) {
        self.kvs.iter_mut().for_each(|v| *v = None);
        self.kv_slots.iter_mut().for_each(|(k, v)| {
            k.reset();
            v.reset();
        });
        self.quant_slots.iter_mut().for_each(|(k, v)| {
            k.reset();
            v.reset();
        });
        self.conv_states.iter_mut().for_each(|v| *v = None);
        self.conv_phase = 0;
        self.conv_compact = false;
    }

    /// The conv ring's current write slot.
    ///
    /// Exposed because the speculative rollback contract is a statement *about*
    /// this index, so a test asserting the contract has to be able to read it.
    /// Read-only: the phase is advanced by `Model::forward` alone, which is the
    /// one-owner property §10.2e records as having already been wrong once.
    pub fn conv_phase(&self) -> usize {
        self.conv_phase
    }

    /// One layer's conv state, for tests that assert what a code path left in
    /// the buffer rather than what it computed.
    ///
    /// `pub` rather than `pub(crate)` because the tests that need it are
    /// integration tests — the limitation being pinned is a property of the
    /// public forward path, and asserting it from inside the module would not
    /// exercise the same call.
    pub fn conv_state_for_test(&self, block_idx: usize) -> Option<Tensor> {
        self.conv_states.get(block_idx).cloned().flatten()
    }

    /// Open a speculative window of `k` positions — `DESIGN.md` §10.2a's
    /// `advance`.
    ///
    /// This does **not** write anything. It records the state a `resolve` must
    /// be able to return to, and the write happens when `Model::forward` runs
    /// the K-position pass. Splitting it that way is what lets one `SpecToken`
    /// cover both halves of the state: the KV length and the conv phase are
    /// both read here, before either is moved.
    ///
    /// Between this call and its [`Cache::resolve`], reads see all `len + k`
    /// positions, and **no second `advance` is permitted** — enforced by
    /// `SpecToken` not being `Copy`, so the caller cannot hold two.
    pub fn advance(&mut self, k: usize) -> Result<SpecToken> {
        if k == 0 {
            candle::bail!("lfm2 advance(0): a speculative window must have at least one position")
        }
        // The rotating ring's history must be wide enough that `k` speculative
        // writes do not overwrite the live window: the buffer is `l_cache + k_ring`
        // wide and a window of `k` consumes `k` slots, leaving `l_cache + k_ring - k`
        // for a window that needs `l_cache`. Checked here rather than at the
        // write, because a partial pass is not rollback-able and this is the
        // last point at which refusing is free.
        if let ConvState::RotatingRing { k: k_ring } = self.conv_state {
            if k > k_ring {
                candle::bail!(
                    "lfm2 advance({k}): `--conv-state rotating:{k_ring}` reserves {k_ring} \
                     history slots, so a {k}-position speculation would overwrite the live \
                     window. Use `rotating:{k}` or larger (`DESIGN.md` §16 6b: the ring's K \
                     and the verifier's K are the same number)."
                )
            }
        } else {
            // §10.2a's whole premise: the shuffle is destructive, so the column
            // leaving the window is gone and no rollback can restore it. The
            // sliding ring is rejected for a second reason -- its compaction
            // relocates the live window, so a phase decrement does not undo a
            // speculation that spanned one.
            candle::bail!(
                "lfm2 advance: speculation requires `--conv-state rotating:<K>`. \
                 `Shuffle` is destructive (`DESIGN.md` §10.2a) and `SlidingRing` \
                 compacts, which relocates the window a rewind would have to restore."
            )
        }
        // The conv layers must ring-write this pass rather than rebuilding from
        // it, and `Model::forward_all` must advance the phase by `seq_len`
        // rather than seeding it. Set here and cleared by `resolve`, so the
        // window's extent is exactly the token's lifetime.
        self.speculative = true;
        Ok(SpecToken {
            k,
            // The first *allocated* slot, not the first slot: 22 of 30 layers
            // are conv and never append, so `kv_slots[0]` is empty on a model
            // whose layer 0 is a conv layer -- which LFM2's is (§5.3).
            kv_len: self
                .kv_slots
                .iter()
                .find(|(k_slot, _)| k_slot.all_data.is_some())
                .map(|(k_slot, _)| k_slot.len),
            conv_phase: self.conv_phase,
        })
    }

    /// Accept the first `n` of the token's `k` positions and discard the rest —
    /// `DESIGN.md` §10.2a's `resolve`.
    ///
    /// `n == k` is full acceptance, `n == 0` full rejection. **One call reaching
    /// both halves of the state** is the contract's second structural property:
    /// a caller cannot satisfy the KV half and forget the conv half, because
    /// there is only one entry point and it iterates the layer list itself.
    ///
    /// `k − n` is derived rather than supplied, so the inconsistent case a
    /// `commit(n)` + `discard(k − n)` pair admits is unrepresentable.
    pub fn resolve(&mut self, tok: SpecToken, n: usize) -> Result<()> {
        if n > tok.k {
            candle::bail!(
                "lfm2 resolve({n}) on a {}-position speculation: cannot accept more than \
                 were proposed",
                tok.k
            )
        }
        // The window closes here whatever the outcome, so the conv layers stop
        // ring-writing and the next ordinary pass behaves as it always did.
        // Cleared before the checks below so that an error leaves the flag down
        // rather than stranding the cache in speculative mode.
        self.speculative = false;

        // The `Cat` arm has no length to rewind: `kvs` holds whatever
        // `Tensor::cat` produced, and the discarded positions are inside that
        // allocation. Checked before anything is mutated, and unconditionally
        // rather than only when `discard > 0` — a full acceptance on the `Cat`
        // arm is still a configuration whose *next* rejection would be silently
        // wrong, and reporting that at the first `resolve` is what makes the
        // constraint discoverable.
        if self.kv_append == KvAppend::Cat && self.use_kv_cache {
            candle::bail!(
                "lfm2 resolve: speculation requires `--kv-append in-place`. \
                 The `Cat` arm has no length to rewind -- the discarded positions are \
                 inside the concatenated allocation."
            )
        }

        // The phase must be where `k` positions of ring-writing would have left
        // it. Checked before any mutation, and on every resolve rather than only
        // on a rejection: a full acceptance that ran the wrong number of passes
        // is equally wrong and is the case a `discard == 0` early return would
        // have skipped.
        let width = self.conv_state.width(self.l_cache);
        let expected = (tok.conv_phase + tok.k) % width;
        if self.conv_phase != expected {
            candle::bail!(
                "lfm2 resolve: conv phase is {} but a {}-position speculation from phase {} \
                 should have left it at {expected}. The window ran a different number of \
                 passes than it was opened for.",
                self.conv_phase,
                tok.k,
                tok.conv_phase
            )
        }

        let discard = tok.k - n;
        if discard == 0 {
            return Ok(());
        }

        // KV: a length decrement per layer. The bytes above `len` are
        // unreachable through the `narrow` and need not be cleared.
        //
        // **Only the attention layers' slots.** `kv_slots` carries one entry per
        // layer so that `block_idx` indexes it directly, but LFM2 is hybrid: 22
        // of its 30 layers are conv and never append (§5.3). Their slots are
        // untouched — `all_data: None`, `len: 0` — and rewinding one is not a
        // no-op but an error, because `count > len`. Skipping unallocated slots
        // rather than tolerating a short rewind is what keeps `rewind`'s bound
        // check meaningful for the slots that *are* live.
        for (k_slot, v_slot) in self.kv_slots.iter_mut() {
            if k_slot.all_data.is_none() {
                continue;
            }
            k_slot.rewind(discard)?;
            v_slot.rewind(discard)?;
        }
        // The quantized arm rewinds by the same rule and for the same reason:
        // a length decrement, discarded codes left unreachable above `len`
        // (§10.2a). The `codes.is_none()` skip is `all_data`'s, for the 22 conv
        // layers whose slots never allocate.
        for (k_slot, v_slot) in self.quant_slots.iter_mut() {
            if k_slot.codes.is_none() {
                continue;
            }
            k_slot.rewind(discard)?;
            v_slot.rewind(discard)?;
        }

        // Conv: a pointer move, and this is the line §10.2a's cost column means
        // by *"a pointer move"*. The discarded slots keep their bytes and become
        // unreachable, exactly as the KV positions above `len` do — the ring is
        // `l_cache + k` wide precisely so that `k` speculative writes cannot
        // reach the live window a rollback returns to.
        self.conv_phase = (tok.conv_phase + n) % width;

        // The KV length is checked the same way and for the same reason. It is
        // recorded per `advance` from layer 0, and every layer moves together.
        if let (Some(before), Some((k_slot, _))) = (
            tok.kv_len,
            self.kv_slots
                .iter()
                .find(|(k_slot, _)| k_slot.all_data.is_some()),
        ) {
            debug_assert_eq!(
                k_slot.len,
                before + n,
                "resolve must leave exactly the accepted positions live"
            );
        }
        Ok(())
    }
}

/// The receipt for an open speculative window — `DESIGN.md` §10.2a.
///
/// **`#[must_use]` and not `Copy`, and both are load-bearing.** A caller who
/// advances and forgets to resolve gets a compile warning; one who resolves
/// twice gets a move error. `Cache` is the only place that can mint one, which
/// is what makes "no second `advance` before a `resolve`" a property of the
/// type rather than of anyone's discipline.
///
/// It carries the pre-advance state rather than only the width, so `resolve`
/// checks that the window ran the passes it was opened for instead of trusting
/// the caller — the same reason `resolve(n)` derives `k − n` rather than taking
/// it.
#[must_use = "an open speculative window must be resolved, or the state is left mid-speculation"]
#[derive(Debug)]
pub struct SpecToken {
    /// How many positions were proposed.
    k: usize,
    /// Layer 0's KV length before the advance. `None` when no slot has been
    /// allocated yet, which is the pre-prefill case.
    kv_len: Option<usize>,
    /// The conv ring's write slot before the advance.
    conv_phase: usize,
}

impl SpecToken {
    /// How many positions this window proposed.
    pub fn k(&self) -> usize {
        self.k
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

/// Whether a projection's residual add can be fused onto its GEMV store
/// (§12.2h, #509's kernel; §12.2e names the three producing sites).
///
/// `x` is the projection's **input** and `proj` the projection. The GEMV route
/// is `m == 1` and the fused arms are instantiated only at the non-transposed
/// `bm4`/`bm8` decode tiles, so a `seq_len > 1` prefill is a GEMM and takes the
/// eager path.
///
/// **Falls back rather than failing**, unlike `flash_decoding`: the eager
/// `forward` + `add` beside it is correct and is what ships, so a shape this
/// fusion does not serve is not an error. What *is* an error — raised inside
/// the op — is a fused request the kernel cannot honour, because that would
/// otherwise run unfused and silently skip the add.
fn gemv_residual_applies(enabled: bool, x: &Tensor, proj: &Linear, residual: &Tensor) -> bool {
    enabled
        && x.device().is_metal()
        && matches!(x.dims(), [_, 1, _] | [1, _])
        && x.dims().last() == Some(&proj.weight().dims()[1])
        && proj.bias().is_none()
        // `Linear::forward` reshapes a contiguous input and broadcasts a
        // non-contiguous one onto a different weight layout. The fused op
        // builds the transposed stride by hand, so it serves the contiguous
        // case only and defers the other to the eager path.
        && x.is_contiguous()
        && proj.weight().is_contiguous()
        && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
        && residual.dtype() == x.dtype()
        // The residual is read at the storing thread's own output row, so it
        // must be exactly the output extent and contiguous — a strided or
        // offset view would be indexed wrongly and produce a well-formed wrong
        // number that §3.5 says nothing reports. Checked again in the op.
        && residual.elem_count() == proj.weight().dims()[0]
        && residual.is_contiguous()
}

/// MLP layer with SwiGLU activation.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    /// Fuse SiLU onto `w1`'s GEMV store (§12.2d, #484's kernel; #536's site).
    ///
    /// Copied from [`Config::gemv_silu`] at construction, the same way
    /// `attn_impl` is — a construction-tier axis in §7.1's terms.
    gemv_silu: bool,
    /// Fuse the post-MLP residual add onto `w2`'s GEMV store (§12.2h). 30 of
    /// the 60 `badd_f16` dispatches per token.
    gemv_residual: bool,
    span: tracing::Span,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        // LFM2 uses w1 (gate), w3 (up), w2 (down) naming convention
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("w1"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("w3"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("w2"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gemv_silu: cfg.gemv_silu,
            gemv_residual: cfg.gemv_residual,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    /// Whether the fused `w1` + SiLU arm applies to this call.
    ///
    /// Metal, `seq_len == 1`, no bias, and a dtype the GEMV carries. **Every
    /// other case takes the eager path rather than failing**, because unlike
    /// `flash_decoding` this arm has a correct eager equivalent right beside
    /// it and prefill is a GEMM the fusion is not instantiated for. The same
    /// shape as `sdpa_applies` above.
    fn gemv_silu_applies(&self, x: &Tensor) -> bool {
        self.gemv_silu
            && x.device().is_metal()
            && x.dims().last() == Some(&self.gate_proj.weight().dims()[1])
            && matches!(x.dims(), [_, 1, _] | [1, _])
            && self.gate_proj.bias().is_none()
            // See `gemv_residual_applies`: the fused op serves the contiguous
            // layout `Linear::forward` reshapes, not the broadcast one.
            && x.is_contiguous()
            && self.gate_proj.weight().is_contiguous()
            && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.down_proj.forward(&self.gated(x)?)
    }

    /// `mlp(x) + residual`, fusing the add onto `w2`'s store when the arm is
    /// on (§12.2h). Otherwise **literally `forward(x)? + residual`** — the
    /// unfused branch calls `forward` rather than reimplementing it, so the
    /// two cannot drift and the shipping path has exactly one definition.
    fn forward_add(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if self.gemv_residual {
            let _enter = self.span.enter();
            let gated = self.gated(x)?;
            if gemv_residual_applies(true, &gated, &self.down_proj, residual) {
                return candle_nn::ops::gemv_residual(&gated, self.down_proj.weight(), residual);
            }
            return self.down_proj.forward(&gated)? + residual;
        }
        self.forward(x)? + residual
    }

    /// `silu(w1(x)) * w3(x)` — the SwiGLU gate, shared by both entry points so
    /// the two cannot drift in what they compute.
    fn gated(&self, x: &Tensor) -> Result<Tensor> {
        // The fused arm replaces `gemv` + `silu_f16` with one dispatch, and
        // computes SiLU on the f32 accumulator rather than on the rounded
        // store — so it MOVES THE DIGESTS by construction (§12.2d, §2.3.5a).
        let gate = if self.gemv_silu_applies(x) {
            candle_nn::ops::gemv_silu(x, self.gate_proj.weight())?
        } else {
            candle_nn::ops::silu(&self.gate_proj.forward(x)?)?
        };
        let up = self.up_proj.forward(x)?;
        gate * up
    }
}

/// Attention layer with per-head QK normalization and RoPE.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    attn_impl: AttnImpl,
    flash_page_size: usize,
    flash_pages_per_chunk: usize,
    /// §9.1a's sizing policy, wired to FlashDecoding's allocation by #234.
    flash_scratch_sizing: FlashScratchSizing,
    /// Fuse the post-attention residual add onto `out_proj`'s GEMV store
    /// (§12.2h). 8 of the 60 `badd_f16` dispatches per token.
    gemv_residual: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(
            num_attention_heads * head_dim,
            hidden_size,
            vb.pp("out_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            attn_impl: cfg.attn_impl,
            flash_page_size: cfg.flash_page_size,
            flash_pages_per_chunk: cfg.flash_pages_per_chunk,
            flash_scratch_sizing: cfg.flash_scratch_sizing,
            gemv_residual: cfg.gemv_residual,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    /// Whether this call can take the fused `sdpa_vector` kernel.
    ///
    /// Every condition is one `candle_nn::ops::sdpa` would otherwise `bail!` on
    /// (`candle-nn/src/ops.rs`, `Sdpa::metal_fwd`), so a false here is a
    /// fallback and never a failure. They are checked rather than assumed:
    ///
    /// * **Metal.** `Sdpa::cpu_fwd` bails outright; there is no CPU arm.
    /// * **`seq_len == 1`.** `supports_sdpa_vector` requires it. Prefill is
    ///   `call_sdpa_full`, whose mask handling differs — deliberately out of
    ///   scope, see the type-level note on `AttnImpl::Sdpa`.
    /// * **dtype.** The kernel is instantiated for f16/bf16/f32 only.
    ///
    /// `head_dim == 64` and `32 % 8 == 0` also hold for LFM2 (§5.2) and are
    /// checked by `ops::sdpa` itself; they are not re-asserted here, because a
    /// second copy of a condition is a second thing to keep in sync (§8.1b).
    fn sdpa_applies(&self, q: &Tensor, seq_len: usize) -> bool {
        self.attn_impl == AttnImpl::Sdpa
            && q.device().is_metal()
            && seq_len == 1
            && matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32)
    }

    /// Whether this call can take the FlashDecoding kernels (issue #116).
    ///
    /// The same shape as `sdpa_applies`, and every condition is one
    /// `ops::flash_decoding` would otherwise `bail!` on — so a false here is a
    /// fallback and never a failure.
    ///
    /// **`BF16` is absent where `sdpa_applies` allows it, and the stated reason
    /// is false — measured 2026-08-30 (#307, `DESIGN.md` §3.9).** This comment
    /// used to say `flash_decoding.metal` does not instantiate `bfloat` because
    /// reaching it needs the ~500-line `_MLX_BFloat16` shim, *"and LFM2 ships
    /// BF16 on disk and decode runs F16 (§9.1b)"*. Both premises fail:
    /// `__HAVE_BFLOAT__` **is defined** on this machine, so that shim is the
    /// `#else` branch and is inert; and bf16 decode is **reachable** — 12 of 12
    /// decode families dispatch a native bf16 sibling, and `lfm2-smoke` PASSes
    /// at `--dtype bf16`.
    ///
    /// **The condition is kept anyway**, and it is a real fallback rather than
    /// an oversight: `flash_decoding.metal` genuinely has no `bfloat`
    /// instantiation, so admitting `BF16` here would `LoadFunctionError` inside
    /// a forward pass. Instantiating it is **not recommended** — §10.4b measures
    /// this arm **+6.3 % slower** at `kv_len` 16 034, so a bf16 variant would be
    /// built-and-unused (§15.2 #11).
    ///
    /// **What a caller must know**: at `--attn flash --dtype bf16` this returns
    /// false and the layer takes the *generic* path — 674 dispatches/token
    /// against the flash arm's 562 — while `config_line()` still renders
    /// `AttnImpl=FlashDecoding`. That is a fourth species of §7.1a-i's
    /// axis-reporting defect: the axis is selected and the mechanism did not
    /// run, for a **dtype** reason. Attribute a flash run from the kernel
    /// census, never from the config line (§2.4).
    fn flash_decoding_applies(&self, q: &Tensor, seq_len: usize) -> bool {
        self.attn_impl == AttnImpl::FlashDecoding
            && q.device().is_metal()
            && seq_len == 1
            && matches!(q.dtype(), DType::F16 | DType::F32)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let y = self.attend(x, index_pos, block_idx, cache)?;
        self.o_proj.forward(&y)
    }

    /// `attn(x) + residual`, fusing the add onto `out_proj`'s store when the
    /// arm is on (§12.2h). Otherwise **literally `forward(..)? + residual`** —
    /// the unfused branch calls `forward`, so the shipping path has exactly one
    /// definition and the two cannot drift.
    fn forward_add(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if self.gemv_residual {
            let y = self.attend(x, index_pos, block_idx, cache)?;
            if gemv_residual_applies(true, &y, &self.o_proj, residual) {
                return candle_nn::ops::gemv_residual(&y, self.o_proj.weight(), residual);
            }
            return self.o_proj.forward(&y)? + residual;
        }
        self.forward(x, index_pos, block_idx, cache)? + residual
    }

    /// Everything up to but excluding `out_proj`, so the two entry points
    /// above share one body and cannot drift in what they compute.
    fn attend(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) then transpose to (batch, num_heads, seq, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply per-head QK normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // The quantized arm (issue #218, `DESIGN.md` §10.7).
        //
        // **It intercepts before the f16 cache**, because it *is* the cache on
        // this arm: K and V are quantized at write time and the f16 tensors are
        // not retained. `seq_len == 1` only -- a prefill takes the f16 path and
        // then hands its result over, which is why the append below runs on
        // every step and the read only at decode.
        if cache.use_kv_cache && cache.kv_quant.is_on() {
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            let (k_slot, v_slot) = &mut cache.quant_slots[block_idx];
            k_slot.append(&k)?;
            v_slot.append(&v)?;
            let n = k_slot.len;
            let capacity = k_slot.capacity;

            if seq_len == 1 {
                let y = candle_nn::ops::sdpa_quant(
                    &q,
                    k_slot.codes.as_ref().unwrap(),
                    k_slot.scales.as_ref().unwrap(),
                    k_slot.biases.as_ref().unwrap(),
                    v_slot.codes.as_ref().unwrap(),
                    v_slot.scales.as_ref().unwrap(),
                    v_slot.biases.as_ref().unwrap(),
                    self.num_attention_heads / self.num_key_value_heads,
                    n,
                    capacity,
                    1f32 / (self.head_dim as f32).sqrt(),
                    1.0,
                    cache.kv_quant,
                )?;
                return y.transpose(1, 2)?.reshape((
                    b_sz,
                    seq_len,
                    self.num_attention_heads * self.head_dim,
                ));
            }
        }

        // Handle KV cache
        let (k, v) = if !cache.use_kv_cache {
            (k, v)
        } else if cache.kv_quant.is_on() {
            // **Prefill on the quantized arm reads the whole cache back.**
            //
            // A turn boundary re-prefills onto the *accumulated* cache without
            // clearing it (§10.9: `Cache::clear()` is never called by either
            // multi-turn harness), so this pass must attend over the full
            // `kv_len` -- the causal mask below is
            // `[seq_len, index_pos + seq_len]` by construction. Attending over
            // only this pass's own K/V is a shape error at `--turns 3` and, if
            // the mask were made to fit, a silently truncated context.
            //
            // A quantized `seq_len > 1` attention kernel is out of scope (§17
            // item 18 scopes this to the decode path), so the honest
            // arrangement is that the cache stays quantized and prefill
            // dequantizes it. **The dequantized values are the ones the codes
            // hold**, not the originals, so prefill sees exactly the cache
            // decode will see -- which is what makes the two paths consistent
            // rather than merely both present.
            let (k_slot, v_slot) = &cache.quant_slots[block_idx];
            let n = k_slot.len;
            let capacity = k_slot.capacity;
            let dtype = q.dtype();
            let k = candle_nn::ops::dequantize_kv(
                k_slot.codes.as_ref().unwrap(),
                k_slot.scales.as_ref().unwrap(),
                k_slot.biases.as_ref().unwrap(),
                b_sz,
                self.num_key_value_heads,
                n,
                self.head_dim,
                capacity,
                dtype,
                cache.kv_quant,
            )?;
            let v = candle_nn::ops::dequantize_kv(
                v_slot.codes.as_ref().unwrap(),
                v_slot.scales.as_ref().unwrap(),
                v_slot.biases.as_ref().unwrap(),
                b_sz,
                self.num_key_value_heads,
                n,
                self.head_dim,
                capacity,
                dtype,
                cache.kv_quant,
            )?;
            (k, v)
        } else {
            match cache.kv_append {
                KvAppend::Cat => {
                    let (k, v) = match &cache.kvs[block_idx] {
                        Some((k_cache, v_cache)) if index_pos > 0 => {
                            let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                            let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                            (k, v)
                        }
                        _ => (k, v),
                    };
                    cache.kvs[block_idx] = Some((k.clone(), v.clone()));
                    (k, v)
                }
                KvAppend::InPlace => {
                    // `slice_set` requires a contiguous source. `k` comes from
                    // `rope`, which returns a fresh contiguous tensor; `v` was
                    // made contiguous above. Asserted rather than assumed,
                    // because a non-contiguous source is an error here and a
                    // silent full copy under `Tensor::cat`.
                    let k = k.contiguous()?;
                    let v = v.contiguous()?;
                    let (k_slot, v_slot) = &mut cache.kv_slots[block_idx];
                    // A turn boundary re-prefills without clearing, so this is
                    // a `seq_len > 1` append onto a non-empty slot. It is the
                    // same call — the offset simply advances by more than one.
                    let k = k_slot.append(&k)?;
                    let v = v_slot.append(&v)?;
                    (k, v)
                }
            }
        };

        // The fused kernel is GQA-native — it derives `gqa_factor` from the head
        // counts and indexes `kv_head_idx = head_idx / gqa_factor` in registers
        // — and it accumulates in F32 internally. So on that arm `repeat_kv` and
        // the F32 upcast below are not merely unnecessary, they are the two
        // things being removed (`DESIGN.md` §6.2, §8.1 principle 4). Both
        // therefore have to stay *inside* the arm that needs them.
        let y = if self.flash_decoding_applies(&q, seq_len) {
            // Chunked attention over the contiguous cache. `page_size` is the
            // ALLOCATION granularity and `pages_per_chunk` the `k` of
            // `chunk_size = k * page_size` (§9.1d); both come from the config
            // rather than being constants here, so a sweep can move either.
            //
            // §10.4 proposes page size 256 and marks it **UNVERIFIED**. It is
            // still unverified: this issue makes it a field one flag apart
            // rather than measuring it, because the axis that would decide it
            // is `kv_len` and that is #61's.
            //
            // `flash_scratch_sizing` is §9.1a's policy, reaching this
            // allocation for the first time (#234). `Reserve` is bounded by
            // the **cache's** capacity rather than by
            // `max_position_embeddings`: the first is the largest `kv_len` this
            // run can reach and the second is the model's positional ceiling,
            // 4096 against 128000 at the shipped default. The conversion to
            // chunks happens here because the chunk size is this caller's pair
            // of fields and a context length is the cache's number.
            let reserve_chunks = cache
                .kv_capacity
                .div_ceil((self.flash_page_size * self.flash_pages_per_chunk).max(1));
            candle_nn::ops::flash_decoding(
                &q,
                &k,
                &v,
                1f32 / (self.head_dim as f32).sqrt(),
                1.0,
                self.flash_page_size,
                self.flash_pages_per_chunk,
                self.flash_scratch_sizing,
                reserve_chunks,
            )?
        } else if self.sdpa_applies(&q, seq_len) {
            // Scale is applied to `q` inside the kernel before the dot product,
            // matching `att / sqrt(head_dim)` below. No mask: `seq_len == 1`
            // attends to the whole cache, which is the same reason the generic
            // arm skips `masked_fill` in that case.
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1f32 / (self.head_dim as f32).sqrt(),
                1.0,
            )?
        } else if self.use_flash_attn {
            let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
            let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            // Expand KV heads to match query heads
            let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
            let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        y.transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_attention_heads * self.head_dim))
    }
}

/// Short convolution layer for efficient sequence processing.
#[derive(Debug, Clone)]
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
    /// One weight permutation per rotation phase, built at load time for
    /// `ConvState::RotatingRing` and `None` otherwise. See `ShortConv::new`.
    rotating_weights: Option<Vec<Tensor>>,
    conv_state: ConvState,
    /// Fuse the post-conv residual add onto `out_proj`'s GEMV store (§12.2h).
    /// 22 of the 60 `badd_f16` dispatches per token — the largest of the three.
    gemv_residual: bool,
    span: tracing::Span,
}

impl ShortConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let l_cache = cfg.conv_l_cache;

        // in_proj projects to 3 * hidden_size for B, C, X components
        let in_proj = linear(hidden_size, 3 * hidden_size, vb.pp("in_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        // Conv weight shape: (hidden_size, 1, l_cache) or (hidden_size, l_cache)
        let conv_weight = vb.get((hidden_size, 1, l_cache), "conv.weight")?;

        // The rotating arm's per-phase weights, precomputed at load time.
        //
        // **This is what makes the rotating form compute the right convolution
        // rather than a different one.** A rotating buffer's slot `s` holds the
        // token of age `(s + width - phase - 1) mod width`, so multiplying slot
        // `s` by weight `s` -- which is what a naive rotation does -- pairs each
        // tap with the wrong weight. Measured, that is an **O(1)** error (~1e7
        // ulp), not a rounding: it is a different operator, and it agrees with
        // the shuffle only on the one phase in `width` where the rotation is the
        // identity.
        //
        // Permuting the weights by phase restores the pairing exactly, and the
        // residual against the shuffle is then bounded by **1 ulp** and flat in
        // run length -- a reduction-order difference and nothing more
        // (`DESIGN.md` §10.2g). The weight is a `[hidden, l_cache]` constant, so
        // all `width` permutations are built once here and cost nothing per
        // token; §10.2a lists this as its third fallback.
        let rotating_weights = match cfg.conv_state {
            ConvState::RotatingRing { .. } => {
                let width = cfg.conv_state.width(l_cache);
                let w = conv_weight.squeeze(1)?;
                let mut per_phase = Vec::with_capacity(width);
                for phase in 0..width {
                    // **The newest token is AT `phase`, and the live window is
                    // the `l_cache` slots ending there.** Counting `d` slots
                    // back from `phase` gives the token `d` steps older, which
                    // the shuffle would have held at column `l_cache - 1 - d`.
                    // Every other slot is history and gets a zero column, so a
                    // stale value cannot contribute however many are held.
                    //
                    // **This is the arithmetic #141 got wrong for `K > 0`**, and
                    // the error was invisible on the arm it shipped. Its formula
                    // — `age = (s + width - phase - 1) % width`, live iff
                    // `age < l_cache` — is *equivalent* to this one at
                    // `width == l_cache`, which is `K = 0`, the default and the
                    // only arm its digest gate ran against. At `K > 0` it selects
                    // the `l_cache` slots **following** `phase` rather than the
                    // ones ending at it, so the live window is read from slots
                    // that hold history and the newest token is weighted zero.
                    //
                    // Measured on the CPU backend at f32 — §2.3.5a's load-bearing
                    // discriminator — `shuffle` and `rotating:0` give an
                    // identical token stream and `rotating:2` gives a different
                    // one. That is a computational bug and not a reduction order,
                    // so §16 6b's *"K is measurably inert"* held only because
                    // nothing read the history slots yet.
                    let mut cols = Vec::with_capacity(width);
                    for s in 0..width {
                        // How many slots back from the newest, or `None` if this
                        // slot is outside the live window.
                        let back = (phase + width - s) % width;
                        let col = if back < l_cache {
                            w.narrow(1, l_cache - 1 - back, 1)?
                        } else {
                            Tensor::zeros((hidden_size, 1), w.dtype(), w.device())?
                        };
                        cols.push(col);
                    }
                    per_phase.push(Tensor::cat(&cols, 1)?.contiguous()?);
                }
                Some(per_phase)
            }
            _ => None,
        };

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            rotating_weights,
            l_cache,
            hidden_size,
            conv_state: cfg.conv_state,
            gemv_residual: cfg.gemv_residual,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    fn forward(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let conv_out = self.convolve(x, block_idx, cache)?;
        self.out_proj.forward(&conv_out)
    }

    /// `conv(x) + residual`, fusing the add onto `out_proj`'s store when the
    /// arm is on (§12.2h). Otherwise **literally `forward(..)? + residual`** —
    /// the unfused branch calls `forward`, so the shipping path has exactly one
    /// definition and the two cannot drift.
    fn forward_add(
        &self,
        x: &Tensor,
        block_idx: usize,
        cache: &mut Cache,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if self.gemv_residual {
            let conv_out = self.convolve(x, block_idx, cache)?;
            if gemv_residual_applies(true, &conv_out, &self.out_proj, residual) {
                return candle_nn::ops::gemv_residual(&conv_out, self.out_proj.weight(), residual);
            }
            return self.out_proj.forward(&conv_out)? + residual;
        }
        self.forward(x, block_idx, cache)? + residual
    }

    /// Everything up to but excluding `out_proj`, so the two entry points
    /// above share one body and cannot drift in what they compute.
    fn convolve(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        // Project input to B, C, X components
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;

        // Element-wise multiply B and X
        let bx = (b * &x_proj)?.contiguous()?;

        // Prepare conv weight: squeeze to (hidden_size, l_cache) for element-wise, or keep for Conv1d
        let conv_weight = self.conv_weight.squeeze(1)?;

        let conv_out = if cache.speculative && seq_len > 1 {
            // **The speculative verify path** — `DESIGN.md` §10.2a's `advance`,
            // for the conv half, at `seq_len = k`.
            //
            // The `seq_len > 1` branch below rebuilds the state from this pass's
            // own activations and zeroes everything past the live window
            // (`lfm2_spec_conv_shape.rs` pins both by execution). That is right
            // for a prefill and **unrollbackable by construction** for a verify:
            // the pre-pass history is discarded, so `resolve(n < k)` has nothing
            // to expose.
            //
            // This writes the `k` positions into the ring the way `k` decode
            // steps would, one slot each, so the state after position `i` is
            // exactly the state a decode of that token would have left — which
            // is what makes discarding a suffix a pointer move rather than a
            // recomputation.
            //
            // **Why a loop and not `Conv1d`.** `Conv1d` computes all `k` outputs
            // from one contiguous activation run, which is cheaper and is what
            // prefill wants. It cannot express *"position i reads the ring's
            // live window at phase i"*, and the ring is what rollback needs. The
            // loop is `k` iterations of the decode writer over a pass that reads
            // the weights **once** — the weight sweep is per pass, not per
            // position (§9.1c), which is the whole speculative thesis and is
            // unaffected by how the conv taps are gathered.
            let ConvState::RotatingRing { .. } = self.conv_state else {
                candle::bail!(
                    "lfm2 speculative conv: only `RotatingRing` can rewind \
                     (`DESIGN.md` §10.2a). `Cache::advance` refuses the other arms."
                )
            };
            let width = self.conv_state.width(self.l_cache);
            let state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => Tensor::zeros((b_sz, self.hidden_size, width), bx.dtype(), bx.device())?,
            };

            // `Model::forward_all` has already advanced `conv_phase` by
            // `seq_len` — it is set once per pass and read by all 22 conv
            // layers, which is the one-owner property §10.2e records as having
            // been wrong once already. So the *last* position sits at
            // `conv_phase` and position `i` sits `seq_len - 1 - i` slots before
            // it. Deriving each slot from the post-pass value rather than
            // re-deriving the pre-pass one keeps a single source for the index.
            let last = cache.conv_phase;
            let mut outs = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                // Position `i`'s slot: the same one the `i`-th decode step of
                // this window would have written.
                let phase = (last + width - (seq_len - 1 - i)) % width;
                let col = bx.narrow(2, i, 1)?.contiguous()?;
                state.slice_set(&col, 2, phase)?;

                // The same phase-permuted weight the decode path uses, so the
                // products are the shuffle's and only the accumulation order
                // differs (§10.2g). Reusing the table rather than rebuilding it
                // is what keeps this a per-position index rather than a
                // per-position permutation (§15.2 #10).
                let w = self
                    .rotating_weights
                    .as_ref()
                    .expect("rotating weights exist whenever the arm is RotatingRing")
                    .get(phase)
                    .expect("phase is taken modulo width, so it indexes the table");
                // `broadcast_mul` rather than `*`: see the decode arm below for
                // why, and note it is the same one-line change at all four
                // conv-weight sites.
                outs.push(state.broadcast_mul(&w.unsqueeze(0)?)?.sum_keepdim(2)?);
            }

            if cache.use_kv_cache {
                cache.conv_states[block_idx] = Some(state.clone());
            }
            Tensor::cat(&outs, 2)?.contiguous()?
        } else if seq_len == 1 {
            match self.conv_state {
                ConvState::RotatingRing { .. } => {
                    // §10.2a as specified: the window IS the buffer, and the
                    // write index rotates through it. No slack, so no compaction
                    // and a constant dispatch count -- and the write lands at one
                    // of `width` fixed offsets rather than at a sliding one.
                    //
                    // **The read is the whole buffer against a phase-permuted
                    // weight**, which is what keeps it the same convolution. A
                    // naive rotation multiplies slot `s` by weight `s` and so
                    // pairs each tap with the wrong weight -- an O(1) error, not
                    // a rounding. With the permutation the products are exactly
                    // the shuffle's; only the order `sum_keepdim` accumulates
                    // them in differs, so the output moves within 1 ulp and the
                    // LFM2 digests move with it. That is a reduction-order
                    // change, discharged against §2.3.5a's discriminators rather
                    // than assumed -- `DESIGN.md` §10.2g.
                    let width = self.conv_state.width(self.l_cache);
                    let state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => {
                            Tensor::zeros((b_sz, self.hidden_size, width), bx.dtype(), bx.device())?
                        }
                    };

                    // Advanced once per token by `Model::forward`, for the same
                    // reason the sliding arm's is: 22 conv layers share one index.
                    let phase = cache.conv_phase;
                    state.slice_set(&bx, 2, phase)?;

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // Built at load time, one per phase -- so the per-token cost
                    // is an index rather than a permutation (§15.2 #10: the
                    // answer is computed when the plan is built, not per token).
                    let w = self
                        .rotating_weights
                        .as_ref()
                        .expect("rotating weights are built whenever the arm is RotatingRing")
                        .get(phase)
                        .expect("phase is taken modulo width, so it indexes the table");
                    state
                        .broadcast_mul(&w.unsqueeze(0)?)?
                        .sum_keepdim(2)?
                        .contiguous()?
                }
                ConvState::SlidingRing { .. } => {
                    let width = self.conv_state.width(self.l_cache);
                    // The ring is written in place, so it must exist as storage
                    // before the write rather than being produced by it.
                    let mut state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => {
                            Tensor::zeros((b_sz, self.hidden_size, width), bx.dtype(), bx.device())?
                        }
                    };

                    // `conv_phase` is the write slot, advanced once per token by
                    // `Model::forward` -- not here, because all 22 conv layers
                    // share one index and advancing it per layer would move it 22
                    // times per token. `conv_compact` comes from the same place,
                    // so this layer never decides where the window sits.
                    let phase = cache.conv_phase;

                    // The window has run out of slack: slide it back to the
                    // front. **This is the entire wrap cost** (`DESIGN.md`
                    // §16 6a), and it is paid once every `slack` tokens rather
                    // than on every token. The live window keeps its slot order
                    // across the move, so the compaction is invisible to the
                    // arithmetic.
                    if cache.conv_compact {
                        let live_w = self.l_cache + self.conv_state.history();
                        let live = state.narrow(2, width - live_w, live_w)?.contiguous()?;
                        let pad = Tensor::zeros(
                            (b_sz, self.hidden_size, width - live_w),
                            bx.dtype(),
                            bx.device(),
                        )?;
                        state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
                    }

                    // The whole point: one in-place write at the sliding slot.
                    // `slice_set` is a single `copy2d` into storage that already
                    // exists, so the `narrow` and the `cat` both disappear -- 44
                    // dispatch positions per token become 22 (§10.2b).
                    state.slice_set(&bx, 2, phase)?;

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // The live window is `l_cache` *contiguous* slots ending at
                    // the write, in the same order the shuffle presents them --
                    // which is what keeps the three-term sum's accumulation order
                    // unchanged and the output bit-identical. A rotating index
                    // would not: see `ConvState`'s note on §10.2a.
                    let window = state.narrow(2, phase + 1 - self.l_cache, self.l_cache)?;
                    window
                        .broadcast_mul(&conv_weight.unsqueeze(0)?)?
                        .sum_keepdim(2)?
                        .contiguous()?
                }
                ConvState::Shuffle => {
                    // Token-by-token generation: use cached state
                    let mut state = match &cache.conv_states[block_idx] {
                        Some(s) => s.clone(),
                        None => Tensor::zeros(
                            (b_sz, self.hidden_size, self.l_cache),
                            bx.dtype(),
                            bx.device(),
                        )?,
                    };

                    // Shift cache and add new token
                    if self.l_cache > 1 {
                        let tail = state.narrow(2, 1, self.l_cache - 1)?;
                        state = Tensor::cat(&[tail, bx.clone()], 2)?;
                    } else {
                        state = bx.clone();
                    }

                    if cache.use_kv_cache {
                        cache.conv_states[block_idx] = Some(state.clone());
                    }

                    // Apply convolution as element-wise multiply and sum.
                    //
                    // **`broadcast_mul` rather than `*`, and this is the one
                    // place LFM2 decode genuinely assumed B=1** (issue #249).
                    // The state is `[b_sz, hidden, l_cache]` and the weight
                    // `unsqueeze(0)`s to `[1, hidden, l_cache]`; `Tensor::mul`
                    // requires the shapes to be *equal*, so the two match only
                    // when `b_sz == 1` and any `b_sz > 1` fails with
                    // `shape mismatch in mul`. Every other operator on the
                    // decode path was already batch-parametric — `b_sz` comes
                    // from `x.dims3()?` and threads through the projections,
                    // the KV slots, the attention and the MLP untouched — so
                    // this was the whole of it, at four sites that are the same
                    // line.
                    //
                    // **The B=1 path does not change.** `broadcast_binary_op`
                    // computes the broadcast shape and takes the
                    // `(false, false)` arm — the identical `mul` — when neither
                    // side needs broadcasting, which is exactly `b_sz == 1`.
                    // At `b_sz > 1` the weight becomes a stride-0 view rather
                    // than a copy, so the batch shares one weight read, which
                    // is §13.4a's claim in miniature: the conv weight is per
                    // step, not per sequence.
                    state
                        .broadcast_mul(&conv_weight.unsqueeze(0)?)?
                        .sum_keepdim(2)?
                        .contiguous()?
                }
            }
        } else {
            // Prefill: use Conv1d
            let conv = Conv1d::new(
                self.conv_weight.clone(),
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: self.hidden_size,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx)?;
            out = out.narrow(2, 0, seq_len)?;

            // Update cache with last l_cache tokens
            if cache.use_kv_cache && self.l_cache > 0 {
                let start = seq_len.saturating_sub(self.l_cache);
                let cache_len = seq_len - start;
                let mut cache_src = bx.narrow(2, start, cache_len)?;
                if cache_len < self.l_cache {
                    let pad = self.l_cache - cache_len;
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, pad),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                // Under the ring the buffer is `l_cache + K + slack` wide and
                // decode writes into it in place -- so prefill must hand over
                // storage of the *full* width, with the live window at the front
                // (write slot `live_w - 1`, which is what `Model::forward` seeds)
                // and everything after it zeroed. Prefill is off the per-token
                // path, so this `cat` is paid once per prompt, not once per token.
                let width = self.conv_state.width(self.l_cache);
                if width > self.l_cache {
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, width - self.l_cache),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[cache_src, zeros], 2)?;
                }
                cache.conv_states[block_idx] = Some(cache_src.contiguous()?);
            }

            out
        };

        // Multiply by C; the projection is the caller's, so the two entry
        // points can fuse it or not.
        let conv_out = (c * &conv_out)?;
        conv_out.transpose(1, 2)?.contiguous()
    }
}

/// Unified decoder layer supporting both attention and convolution.
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(Box<Attention>),
    ShortConv(ShortConv),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        // LFM2 uses operator_norm and ffn_norm naming
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        // LFM2 uses feed_forward naming for MLP
        let mlp = Mlp::new(cfg, vb.pp("feed_forward"))?;

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"))?),
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            kind,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        // `forward_add` is `forward(..)? + residual` unless the fused arm is on
        // AND applies, in which case the add happens at the producing GEMV's
        // f32 accumulator instead of in a `badd_f16` over the rounded store
        // (§12.2h). That is a DIFFERENT COMPUTATION and moves the digests.
        let x = match &self.kind {
            LayerKind::Attention(attn) => {
                attn.forward_add(&x, index_pos, block_idx, cache, residual)?
            }
            LayerKind::ShortConv(conv) => conv.forward_add(&x, block_idx, cache, residual)?,
        };

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        self.mlp.forward_add(&x, residual)
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
    conv_state: ConvState,
    l_cache: usize,
}

/// Announce that the forward pass has reached graph node `index`
/// (`DESIGN.md` §13.5m, issue #420 stage 3).
///
/// # Why the model is the party that can say this
///
/// Candle is eager: there is no graph object, so nothing below this can know
/// where in the model a dispatch sits (§11.1a). The command-buffer cut
/// therefore falls where a *global dispatch counter* happens to tick, and
/// §13.5i measured the consequence — every seam at a multiple of 50 counted
/// from process start, landing 0–18 dispatches into whichever block it hits.
/// The layer loop is the one place that knows the answer, so it is the one
/// place that can supply it.
///
/// # Why it is free when the arm is off
///
/// Under the default cut policy `mark_graph_node` returns without a lock or a
/// store, so this is a call and a branch on a path that runs 30 times per
/// token against 538 dispatches. It is *not* gated on an environment read here
/// deliberately: the policy is read once at `Commands::new` and a second read
/// on the per-token path would be the O(N)-on-the-hot-path shape §15.2 #10
/// forbids, in an env lookup.
#[cfg(feature = "metal")]
fn mark_graph_node(device: &Device, index: usize) {
    // A non-Metal device with the metal feature built in has no command
    // buffers to cut. Silent rather than an error, for `admit_memory_budget`'s
    // reason: a CPU run is a legitimate thing to do.
    if let Ok(metal) = device.as_metal_device() {
        metal.mark_graph_node(index);
    }
}

/// No Metal, no command buffers, nothing to cut. The CPU and CUDA backends
/// compile none of the above, which is the constraint upstreaming imposes
/// (§14.1).
#[cfg(not(feature = "metal"))]
fn mark_graph_node(_: &Device, _: usize) {}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            dtype: vb.dtype(),
            conv_state: cfg.conv_state,
            l_cache: cfg.conv_l_cache,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // The ring's write slot, advanced once per forward pass and read by all
        // 22 conv layers (`DESIGN.md` §10.2c -- it is conv state, so it lives in
        // `Cache` beside the bytes it indexes rather than in a params struct).
        //
        // Prefill writes the state directly rather than through the ring, and
        // leaves the newest token in the last live slot -- which is slot
        // `l_cache + K - 1`. Decode advances from there. Advancing *before* the
        // layers run rather than after is what makes the first decode token
        // after a prefill land past the newest token instead of overwriting it.
        //
        // **One owner for the wrap decision.** This computes both the slot and
        // whether the layers must compact to reach it; the layers only obey. An
        // earlier draft split it -- this wrapped the index while the layers
        // tested `phase >= width` -- and the wrap made the layers' test
        // unreachable, so the compaction never ran.
        // Seeding, wrapping and the compaction decision all live in
        // `advance_conv_phase`, shared with `forward_all` so the two entry
        // points cannot disagree about where the window sits. The rotating arm
        // never compacts; the sliding arm seeds prefill at `l_cache - 1` rather
        // than `live_w - 1`, which is what keeps history slots from being read
        // as live at `K > 0`.
        self.advance_conv_phase(seq_len, cache);

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        Self::begin_taps(cache);
        for (block_idx, layer) in self.layers.iter().enumerate() {
            // The graph position, announced before the layer's dispatches are
            // encoded so a seam lands at its first dispatch rather than after
            // it (`DESIGN.md` §13.5m). A no-op unless the node-index cut is
            // selected.
            mark_graph_node(&cache.device, block_idx);
            Self::tap_layer_input(&hidden_states, block_idx, cache)?;
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }

        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    /// Embed `input_ids` without running a single layer.
    ///
    /// This exists so a caller holding image features can build the residual
    /// stream itself — embed the text, substitute the projected image rows at
    /// the `image_token_id` positions, and hand the result to
    /// [`Self::forward_embeds`]. Exposing the embedding rather than taking
    /// `pixel_values` here is what keeps every vision type out of this file
    /// (`CONTRIBUTING.md` §4's constraint that the CPU and CUDA backends must
    /// still build, and `DESIGN.md` §5.5a's that a text-only model names 266
    /// tensors and no more).
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// `forward`, but starting from a residual stream the caller has already
    /// built.
    ///
    /// # Why this is a second entry point rather than a flag on `forward`
    ///
    /// The same reason `forward_all` is: `forward` is on the per-token path and
    /// its shape is what every recorded digest belongs to (`DESIGN.md` §2.3.8c).
    /// A branch inside it would put a caller decode does not have onto the
    /// decode path — §11.3l finding 4's shape, where a window drawn for one
    /// purpose silently changed another. **Text-only decode does not reach this
    /// function at all**, which is what makes "the text path is unregressed" a
    /// structural claim rather than a measured hope.
    ///
    /// `embeds` is `[b, seq, hidden]` and must already carry whatever the caller
    /// wanted substituted into it. Everything after the embedding is `forward`'s
    /// code exactly, including the conv-phase advance, so a prompt whose embeds
    /// happen to be the plain token embeddings produces `forward`'s own answer.
    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = embeds.dims3()?;
        self.advance_conv_phase(seq_len, cache);

        let mut hidden_states = embeds.clone();
        Self::begin_taps(cache);
        for (block_idx, layer) in self.layers.iter().enumerate() {
            mark_graph_node(&cache.device, block_idx);
            Self::tap_layer_input(&hidden_states, block_idx, cache)?;
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }

        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    /// Run the model and project **every** position, not only the last.
    ///
    /// # Why this exists, and why it is a separate entry point
    ///
    /// `forward` above narrows the residual stream to `seq_len - 1` before the
    /// `lm_head` — `DESIGN.md` §5.10 records it as *"`forward()` returns logits
    /// for the last position only, so prefill and decode share one call path"*,
    /// which reads as a prefill efficiency note. **For a speculative verify pass
    /// it is the mechanism rather than an efficiency note**: verifying K
    /// proposed tokens means comparing the target's argmax at each of the K
    /// positions, and one position's logits cannot answer K questions.
    ///
    /// It is a second entry point rather than a flag on the first because
    /// `forward` is on the per-token path and its shape is what every recorded
    /// digest belongs to. A branch inside it would put a `seq_len`-dependent
    /// narrow on the decode path to serve a caller decode does not have —
    /// §11.3l finding 4's shape, where a window drawn for one purpose silently
    /// changed another.
    ///
    /// # What it costs, stated rather than elided
    ///
    /// `lm_head` is `[128000, 2048]` and tied to the embedding (§5.1), so this
    /// runs a GEMV per position where `forward` runs one. **The weight is read
    /// once either way** — that is the whole speculative thesis (§9.1c: one
    /// sweep serves K tokens) — but the *output* is `K × 128000` f32, which is
    /// 512 KB per position and the one term in a verify pass that grows with K
    /// without amortising. §4 of `measurements/issue-89-verify-shape.md` costs
    /// it.
    ///
    /// Returns `[b_sz, seq_len, vocab]` in f32, where `forward` returns
    /// `[b_sz, vocab]`.
    pub fn forward_all(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        self.advance_conv_phase(seq_len, cache);

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        Self::begin_taps(cache);
        for (block_idx, layer) in self.layers.iter().enumerate() {
            // Marked here as well as in `forward`, for `advance_conv_phase`'s
            // reason below: two entry points over the same layer loop must not
            // disagree about the graph position they are at. A verify pass
            // encodes the same 30 nodes.
            mark_graph_node(&cache.device, block_idx);
            // Tapped here as well as in `forward`, for the same reason and with
            // the same helper — a drafter conditioned on a *verify* pass's
            // hidden states is the case #363 exists for, so the entry point
            // that runs `seq_len > 1` must not be the one that does not tap.
            Self::tap_layer_input(&hidden_states, block_idx, cache)?;
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }
        let hidden_states = self.embedding_norm.forward(&hidden_states)?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    /// The **raw** token embedding, for a drafter that ships none of its own
    /// (`DESIGN.md` §10.2p, issue #363).
    ///
    /// # Why a drafter needs this at all
    ///
    /// `LFM2.5-2.6B-DSpark` has **no `embed_tokens`** — verified from the
    /// payload-bearing safetensors header, which carries 62 tensors and none of
    /// that name (§10.2l/§10.2o). That is the design rather than a defect: a
    /// hidden-state drafter shares the target's token boundary, so the draft
    /// block's input embedding has to come from here.
    ///
    /// # Why RAW, and why that is not an oversight
    ///
    /// This deliberately does **not** apply `embedding_norm`, and the
    /// distinction is worth a sentence because the normed tensor is the one a
    /// reader would reach for — it is what `forward` feeds `lm_head`. It is the
    /// wrong one here twice over: the drafter's own `input_layernorm`
    /// normalizes its input already, and the residual the draft block carries
    /// is the raw table output, so the normed tensor double-normalizes the
    /// first and falsifies the second.
    ///
    /// **Measured elsewhere rather than argued here**: mlx-dspark's
    /// `Target.draft_embed` records *"Do NOT apply `embed_norm` here: feeding
    /// the normalized input embedding measured a clear acceptance loss (accept
    /// 3.73 raw vs 2.86 normed on code), because the drafter's own
    /// `input_layernorm` already normalizes and the residual it carries is the
    /// raw table output, not the normed one."* That is a **−23 % acceptance**
    /// defect with **no error anywhere** — the class §10.2n names as the one no
    /// gate in this tree can see, since the target verifies every token and the
    /// emitted stream is byte-identical either way.
    pub fn draft_embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// The **raw** `lm_head` projection, for a drafter that ships none of its
    /// own (`DESIGN.md` §10.2p, issue #363).
    ///
    /// `hidden [.., hidden_size]` → `[.., vocab_size]`, in the model's dtype
    /// rather than f32: this is a *proposal* head whose output is consumed by
    /// the Markov bias and an argmax, where `forward` and [`Self::forward_all`]
    /// cast because their output is compared against a digest.
    ///
    /// # Why the proposal head must stay the raw projection
    ///
    /// The drafter adds its Markov transition bias to *these* logits, at the
    /// scale `lm_head` produces them. Any output transform a verifier applies —
    /// a scale, a softcap — shrinks the base logits while the bias stays at raw
    /// scale, so the bias overwhelms the head; mlx-dspark measured exactly that
    /// on a target whose head scales by ~0.196, where first-position acceptance
    /// fell **~85 % → ~29 % with no error anywhere**. LFM2 applies no such
    /// transform, so on this target the two coincide **today**. The accessor is
    /// named for the property it must keep rather than for the code it
    /// currently equals, so that adding one later does not silently reach the
    /// drafter.
    pub fn draft_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden)
    }

    /// The residual stream at the entry of **every** layer — the reference the
    /// tap is tested against (`DESIGN.md` §10.2m, issue #363).
    ///
    /// # Why this exists rather than the test reaching into the tap
    ///
    /// `CONTRIBUTING.md` §3.1b: a test whose reference comes from the same
    /// mechanism as its subject *"looks like a comparison and is not."* The tap
    /// pushes from inside the layer loop; this returns what the layer loop was
    /// about to consume, for every layer rather than the configured few. So
    /// *"tap `i` is layer `tap_layers[i]`'s input"* becomes an equality against
    /// a value produced by a different code path, and a defect that shifted
    /// every tap by one layer — the class a shape invariant is blind to — fails
    /// it.
    ///
    /// Not on any per-token path: it materialises `num_hidden_layers` tensors
    /// and exists for the test and for whoever needs to pin the representation
    /// against llama.cpp's (§10.2m's `H_ctx` ulp-match is the eventual
    /// consumer).
    pub fn layer_input_residuals(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Vec<Tensor>> {
        let (_, seq_len) = input_ids.dims2()?;
        self.advance_conv_phase(seq_len, cache);
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut out = Vec::with_capacity(self.layers.len());
        for (block_idx, layer) in self.layers.iter().enumerate() {
            out.push(hidden_states.contiguous()?.to_dtype(cache.tap_dtype)?);
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }
        Ok(out)
    }

    /// Clear the previous pass's taps, once per forward pass
    /// (`DESIGN.md` §10.2m, issue #363).
    ///
    /// Factored out for `advance_conv_phase`'s reason: two entry points over
    /// one layer loop must not disagree, and *when the vector is cleared* is
    /// exactly the kind of decision that has already been got wrong once here
    /// by being written twice (the conv wrap, §10.2e).
    ///
    /// **Clearing at pass entry rather than at `take_taps`** is what makes a
    /// pass that is never drained safe: a rolled-back speculative window leaves
    /// its taps behind, and the next pass replaces them before any consumer can
    /// read them as though they described committed state.
    ///
    /// Free when not tapping — clearing an empty vector is a store of a length
    /// that is already zero.
    ///
    /// **It resizes to `tap_layers.len()` rather than emptying**, because the
    /// layer loop writes taps **by slot index** rather than by pushing: the
    /// consumer's order is `tap_layers`' order and the loop's order is ascending
    /// layer index, and those differ whenever the drafter's `target_layer_ids`
    /// is not sorted. Pushing produced ascending order, which
    /// `tap_order_follows_tap_layers_and_the_layers_are_distinguishable` caught
    /// — the §10.2h shape, where a permutation is algebraically valid and
    /// silently pairs every slice with the wrong block of `W_c`.
    fn begin_taps(cache: &mut Cache) {
        cache.taps.clear();
        cache.taps.resize(cache.tap_layers.len(), None);
    }

    /// Capture layer `block_idx`'s **input residual** if it is tapped
    /// (`DESIGN.md` §10.2m, issue #363).
    ///
    /// Called at layer entry, *before* the block runs — which is the whole
    /// specification of the representation, and the reason it is a call at this
    /// point rather than a field the layer fills in. The witness is llama.cpp
    /// PR #27383's `t_layer_inp[il] = cur`; post-attention or post-MLP would be
    /// a different tensor that `fc.weight` was not trained against, and nothing
    /// downstream could tell (§2.3.5a: a wrong-but-stable input produces a
    /// deterministic wrong drafter, and the identity gate cannot see a drafter
    /// at all, §10.2m).
    ///
    /// # Cost when off, which is the case that must stay free
    ///
    /// `tap_layers` is empty by default, so this is an `is_empty()` on a slice
    /// and a return — 30 times per token against §6.6's 675 dispatches. It is
    /// deliberately *not* gated on an environment read, for `mark_graph_node`'s
    /// reason: a per-token env lookup is the O(N)-on-the-hot-path shape §15.2
    /// #10 forbids.
    ///
    /// `contains` over a ≤5-element slice rather than a set: at this length a
    /// linear scan is faster than hashing, and §15.2 #10's target is a *lookup
    /// whose N grows*, which this one's cannot — it is the drafter's fixed
    /// `target_layer_ids`.
    /// **Written by slot, not pushed.** A layer may appear at any position of
    /// `tap_layers` — the drafter's `target_layer_ids` is its order, not
    /// ascending — so the slot is where the *consumer* expects this layer, and
    /// the loop's own order is irrelevant. A layer listed twice fills both its
    /// slots, which is what a consumer asking for it twice means.
    fn tap_layer_input(hidden: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<()> {
        if cache.tap_layers.is_empty() {
            return Ok(());
        }
        let mut tapped: Option<Tensor> = None;
        for slot in 0..cache.tap_layers.len() {
            if cache.tap_layers[slot] != block_idx {
                continue;
            }
            // `contiguous` before the cast: the residual arrives as whatever the
            // previous layer's store left, and a consumer concatenating five of
            // these needs each addressable as a plain `[b, seq, hidden]` block.
            // Doing it here rather than at the consumer keeps the layout
            // obligation with the party that knows the producer. Computed once
            // even when a layer fills two slots.
            if tapped.is_none() {
                tapped = Some(hidden.contiguous()?.to_dtype(cache.tap_dtype)?);
            }
            cache.taps[slot] = tapped.clone();
        }
        Ok(())
    }

    /// The ring's write slot, advanced once per forward pass and read by all 22
    /// conv layers.
    ///
    /// Factored out of `forward` when `forward_all` was added, so the two entry
    /// points cannot disagree about where the window sits. That is the same
    /// argument the body already carries for keeping the wrap decision in one
    /// place: two owners of one decision is a shape that cannot be right, and it
    /// has already been wrong here once.
    fn advance_conv_phase(&self, seq_len: usize, cache: &mut Cache) {
        if let ConvState::RotatingRing { .. } = self.conv_state {
            let width = self.conv_state.width(self.l_cache);
            cache.conv_compact = false;
            cache.conv_phase = if cache.speculative && seq_len > 1 {
                // A verify pass writes `seq_len` ring slots, one per position,
                // so it advances the phase by `seq_len` — exactly what
                // `seq_len` decode steps would have done. The layers derive
                // each position's slot from the *pre-pass* phase, so this
                // lands where the last position wrote.
                (cache.conv_phase + seq_len) % width
            } else if seq_len == 1 {
                (cache.conv_phase + 1) % width
            } else {
                self.l_cache - 1
            };
        } else if let ConvState::SlidingRing { .. } = self.conv_state {
            let width = self.conv_state.width(self.l_cache);
            let live_w = self.l_cache + self.conv_state.history();
            if seq_len == 1 {
                let next = cache.conv_phase + 1;
                cache.conv_compact = next >= width;
                cache.conv_phase = if cache.conv_compact { live_w } else { next };
            } else {
                cache.conv_compact = false;
                cache.conv_phase = self.l_cache - 1;
            }
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Verify `k` proposed tokens in **one** forward pass, accept the longest
    /// correct prefix, and roll the state back to it.
    ///
    /// This is `DESIGN.md` §17's speculative *verifier* — the half every
    /// speculative scheme shares. **Where `proposed` comes from is policy and is
    /// not here** (§7.7: policy is runtime-swappable, kernel structure is not).
    /// #90 (MTP) and #91 (DSpark) supply proposers; this takes a slice.
    ///
    /// # The accept test, and why greedy needs no epsilon
    ///
    /// Under greedy decoding a speculative scheme is **output-identical to
    /// non-speculative decoding by construction**: a draft token is accepted iff
    /// it equals what the target would have produced, which is
    /// `argmax(logits[i])`. So the check is exact rather than statistical, and
    /// **a scheme that changes the output is a bug, not a tradeoff** — which is
    /// why the canonical digest pair is this mechanism's test rather than a
    /// divergence bound. Distribution-preserving sampling is a later,
    /// separately-argued step and is deliberately absent.
    ///
    /// # The one-token overhang, which is the mechanism and not an off-by-one
    ///
    /// `last` is the token already committed — the one whose logits produced
    /// `proposed[0]`. The pass runs `[last, proposed[0..k-1]]`, so position `i`
    /// predicts what follows `proposed[i-1]`, and **a fully-accepted window
    /// yields `k` tokens from `k` positions**: the `k − 1` verified proposals
    /// plus the bonus token the last position predicts for free. That free token
    /// is why a `k`-position verify can advance `k` tokens rather than `k − 1`,
    /// and it is the reason the cost model in
    /// `measurements/issue-89-verify-shape.md` §4 compares a `k`-position verify
    /// against `k` decodes rather than `k − 1`.
    ///
    /// # Returns
    ///
    /// The accepted tokens — between 1 and `k`, never 0, because the target's
    /// own prediction at position 0 is always correct by definition.
    pub fn verify_step(
        &self,
        last: u32,
        proposed: &[u32],
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Vec<u32>> {
        let k = proposed.len();
        if k == 0 {
            candle::bail!("lfm2 verify_step: no proposed tokens")
        }
        // Opened before the pass, so the token records the state to return to
        // rather than a state the pass has already moved.
        let tok = cache.advance(k)?;

        // `[last, proposed[0..k-1]]`. The final proposal is not fed: nothing
        // would verify it, since its verifier is the token after it.
        let mut window = Vec::with_capacity(k);
        window.push(last);
        window.extend_from_slice(&proposed[..k - 1]);

        let device = &cache.device;
        let input = Tensor::new(window.as_slice(), device)?.reshape((1, k))?;

        // **One pass over the weights for k positions.** §9.1c: the weight sweep
        // is per step and not per token, so this is where the multiple comes
        // from -- 5.394 GB read once against k times.
        let logits = self.forward_all(&input, index_pos, cache)?;
        let argmax = logits.i(0)?.argmax(candle::D::Minus1)?.to_vec1::<u32>()?;

        // Accept the longest prefix where the target agrees with the proposal.
        // Position `i`'s argmax is what the target would emit after
        // `window[i]`, so it is compared against `proposed[i]`.
        let mut accepted = Vec::with_capacity(k);
        for i in 0..k {
            let target = argmax[i];
            accepted.push(target);
            if i + 1 < k && target != proposed[i] {
                // The first disagreement ends the prefix. The target's own token
                // is kept -- it is what non-speculative decoding would have
                // emitted here -- which is what makes a rejection cost nothing
                // rather than costing a token.
                break;
            }
        }

        // `n` positions of state survive. The remaining `k - n` were written by
        // the pass and are discarded: KV by a length decrement, conv by a phase
        // move, neither by clearing bytes (§10.2a).
        cache.resolve(tok, accepted.len())?;
        Ok(accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two arms' shapes and contents, built the way `Attention::forward`
    /// builds them: `[b_sz, n_kv_heads, seq, head_dim]`, appended along dim 2.
    fn token(step: usize, dev: &Device) -> Result<Tensor> {
        // Distinct values per step and per element, so a transposed or
        // misaligned write is visible rather than accidentally equal. #11.3l's
        // finding 3 is the rule: a fixture whose values coincide proves nothing.
        let n: Vec<f32> = (0..2 * 3).map(|i| (step * 100 + i) as f32).collect();
        Tensor::from_vec(n, (1, 2, 1, 3), dev)
    }

    /// The reference: what `Tensor::cat` produces, step by step.
    fn cat_reference(steps: usize, dev: &Device) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for s in 0..steps {
            let t = token(s, dev)?;
            acc = Some(match acc {
                None => t,
                Some(a) => Tensor::cat(&[&a, &t], 2)?.contiguous()?,
            });
        }
        acc.ok_or_else(|| candle::Error::Msg("no steps".into()))
    }

    /// **The parity test.** In-place append must equal `Tensor::cat` exactly.
    ///
    /// Bit-equality rather than a tolerance: this moves bytes and does no
    /// arithmetic, so `DESIGN.md` §2.3.5a's changed-digest procedure does not
    /// apply and anything but equality is a defect.
    #[test]
    fn in_place_append_matches_cat() -> Result<()> {
        let dev = Device::Cpu;
        for steps in 1..=6 {
            let mut slot = KvSlot::new(8);
            let mut got = None;
            for s in 0..steps {
                got = Some(slot.append(&token(s, &dev)?)?);
            }
            let got = got.unwrap();
            let want = cat_reference(steps, &dev)?;
            assert_eq!(got.dims(), want.dims(), "shape at {steps} steps");
            assert_eq!(
                got.flatten_all()?.to_vec1::<f32>()?,
                want.flatten_all()?.to_vec1::<f32>()?,
                "values at {steps} steps"
            );
        }
        Ok(())
    }

    /// A turn boundary: `seq_len > 1` appended onto a non-empty slot.
    ///
    /// This is the case `lfm2-smoke`'s turn 2 exercises and the one §11.1a's
    /// single-turn limitation is the precedent for — a second prefill writes
    /// several positions at a non-zero offset. It is the same `slice_set` call
    /// with a larger `src`, and the point of the test is that nothing special
    /// happens.
    #[test]
    fn in_place_survives_a_multi_token_append_at_a_nonzero_offset() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        // A 3-token "prefill" on top of 2 decoded tokens.
        let prefill = Tensor::cat(&[&token(2, &dev)?, &token(3, &dev)?, &token(4, &dev)?], 2)?
            .contiguous()?;
        let got = slot.append(&prefill)?;
        let want = cat_reference(5, &dev)?;
        assert_eq!(got.dims(), &[1, 2, 5, 3]);
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    /// The read is a *view*, and that is the property §11.1a.1 wants.
    ///
    /// A `narrow` clones the storage `Arc` and rewrites only the layout, so one
    /// allocation backs every step and the binding is the same `MTLBuffer` every
    /// token. If this silently became a copy, the dispatch count and every
    /// digest would be **unchanged** and only the buffer identity would move —
    /// which no gate in this project can see. Hence a test.
    ///
    /// Observed by aliasing rather than by pointer equality: `same_storage` is
    /// `pub(crate)` to `candle-core`, and widening a core API for one assertion
    /// is a larger change than the assertion is worth. A later `append` writes
    /// through the backing buffer, so if the earlier return value was a view it
    /// sees nothing (the write lands past its length) while its *own* bytes stay
    /// readable — and if it were a copy, `all_data` and the view would have
    /// diverged. The discriminating check is that the backing buffer carries
    /// every step's bytes at the offsets the views reported.
    #[test]
    fn the_read_is_a_view_of_the_preallocated_buffer() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        let first = slot.append(&token(0, &dev)?)?;
        let first_before = first.flatten_all()?.to_vec1::<f32>()?;
        let second = slot.append(&token(1, &dev)?)?;

        // The step-2 write must not have disturbed step 1's region.
        assert_eq!(
            first.flatten_all()?.to_vec1::<f32>()?,
            first_before,
            "appending must not rewrite the live prefix"
        );

        // Both views must be prefixes of one buffer: read the backing store at
        // full capacity and check each view equals its own leading slice.
        let backing = slot.all_data.as_ref().unwrap();
        for (steps, view) in [(1usize, &first), (2usize, &second)] {
            let want = backing
                .narrow(2, 0, steps)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(
                view.flatten_all()?.to_vec1::<f32>()?,
                want,
                "the {steps}-step view must be the buffer's own prefix"
            );
        }
        Ok(())
    }

    /// Exhaustion is an error, not a silent reallocation.
    ///
    /// `candle_nn::kv_cache::Cache` grows by `Tensor::cat` here; this declines,
    /// because growing would put back the reallocation this change removes and
    /// would move the buffer identity, both at the moment the cache is largest.
    /// See `Cache::new_with` for why no policy is chosen.
    #[test]
    fn exceeding_capacity_fails_loudly() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(2);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        let err = slot.append(&token(2, &dev)?).unwrap_err().to_string();
        assert!(
            err.contains("KV cache exhausted"),
            "expected a capacity error, got: {err}"
        );
        Ok(())
    }

    /// `clear()` must reach the in-place slots too.
    ///
    /// Resetting only `kvs` would leave a stale buffer that the next sequence
    /// appends *after*, so its first token would read `len` zeros in front of
    /// it — a wrong answer that every shape check passes.
    #[test]
    fn reset_returns_the_slot_to_its_pre_prefill_state() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(8);
        slot.append(&token(0, &dev)?)?;
        slot.append(&token(1, &dev)?)?;
        slot.reset();
        let got = slot.append(&token(0, &dev)?)?;
        assert_eq!(got.dims(), &[1, 2, 1, 3], "a reset slot starts at length 1");
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            cat_reference(1, &dev)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // A stateful fuzzer over the slot's append/rewind SEQUENCE (lloom #575).
    // -----------------------------------------------------------------------

    /// One symbol of the slot's alphabet, as the decode loop drives it.
    ///
    /// Deliberately the *loop's* operations: `KvSlot`'s own methods are covered
    /// by the five tests above, and what is not covered is what happens across
    /// a long run of them. §10.2a's `advance`/`resolve` pair is a length
    /// increment followed by a conditional decrement, and the five tests reach
    /// at most two windows with `k <= 3`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SlotOp {
        /// Decode `n` positions and keep them: the ordinary step.
        Append { n: usize },
        /// A speculative window of `k` positions of which `n` are accepted --
        /// `advance(k)` then `resolve(_, n)`, which at slot level is an append
        /// of `k` followed by `rewind(k - n)`.
        Speculate { k: usize, accept: usize },
        /// A rewind of more positions than are live. Must be refused, and the
        /// slot must be unchanged afterwards.
        RewindPastStart,
        /// Append exactly enough positions to reach capacity, then one more.
        ///
        /// **The boundary itself, drawn as a symbol rather than left to
        /// chance.** A generator that only draws window sizes lands *near* the
        /// capacity, essentially never *on* it, so the admitted side of the
        /// bound is never exercised at the one value that distinguishes `>`
        /// from `>=`. This symbol pins both sides: filling to exactly capacity
        /// must be ADMITTED, and one past it must be REFUSED.
        ///
        /// **It still does not kill a bound mutated to `> capacity + 1`, and
        /// that is a finding rather than a gap.** Under that mutant `KvSlot`
        /// admits one position too many and `Tensor::slice_set` refuses it on
        /// its own — `candle-core/src/tensor_cat.rs:280` bails on
        /// `*v2 + offset > *v1`. The write never lands, so **no sequence of
        /// appends can observe the difference**: this fuzzer asserts
        /// *behaviour*, and the behaviour is unchanged.
        ///
        /// `exceeding_capacity_fails_loudly` does kill it, because it asserts
        /// the error *text* — which becomes
        /// `"shape mismatch on target dim, dst: 2, src: 1 + 2"` instead of
        /// `"KV cache exhausted"`. That is a real and different property: the
        /// slot's own message names the knob (`Config::kv_capacity`) a caller
        /// must raise, where `slice_set`'s names two integers. So the two tests
        /// are **not** redundant and neither subsumes the other — this one
        /// catches a bound that lets the write through, that one catches a
        /// bound that stops naming itself. Recorded here because a doc comment
        /// promising more than its assertions deliver is the sentence a reader
        /// trusts (§3.1b rule 3).
        FillToCapacityThenOne,
        /// A turn boundary, and the next turn runs at `heads` KV heads.
        ///
        /// **The head count is carried on the symbol deliberately.** `reset`'s
        /// contract is that the *buffer* is dropped and not merely the length,
        /// and `Cache::clear`'s doc gives the reason in as many words: *"a new
        /// sequence may have a different batch size, so keeping the allocation
        /// would pin `b_sz` from the previous one."* A fixture whose every turn
        /// has the same shape cannot see that -- the stale buffer is the right
        /// size, the writes land, and the mutant is behaviourally identical.
        ///
        /// Measured: with a constant shape, `reset` mutated to keep `all_data`
        /// **survives this fuzzer and all 16 pre-existing tests**. It is
        /// #574's mutant 2 exactly -- a quantity that is constant across the
        /// fixture cannot be observed -- reproduced in the harness written to
        /// catch that class, and repaired the way §3.1a rule 2 says: by fixing
        /// the INPUT, not by adding a check.
        Reset { heads: usize },
    }

    /// What the slot's length and contents should be, from the documented
    /// contract rather than from `KvSlot`'s fields.
    ///
    /// Keeps the accepted tokens' *step ids* rather than tensors: the model has
    /// no business holding a `Tensor`, and a model that shares the
    /// implementation's data structures shares its bugs (§15.1h).
    #[derive(Debug)]
    struct SlotModel {
        /// Step id of every live position, in order.
        live: Vec<usize>,
        /// KV heads the current turn writes. Changes at a `Reset`, which is
        /// what makes a buffer kept across turns observable.
        heads: usize,
    }

    /// Transitions this state machine has, for #575's *report transitions, not
    /// lines* rule.
    #[derive(Debug, Default, Clone, Copy)]
    struct SlotTransitions {
        appends: u64,
        speculations: u64,
        full_accepts: u64,
        full_rejects: u64,
        partial_accepts: u64,
        rewinds_refused: u64,
        capacity_refusals: u64,
        resets_from_nonempty: u64,
        /// A rewind that returned the slot to a length it had previously held,
        /// then appended different content there -- the reuse-after-discard
        /// shape, and the one a length-only assertion cannot see.
        rewrote_discarded_positions: u64,
        /// A turn boundary where the next turn's shape differs from the last's.
        /// The transition that makes `reset` dropping the buffer observable.
        turns_changed_shape: u64,
        /// A sequence that landed EXACTLY on capacity. The transition that
        /// makes the `>` in the bound check observable.
        filled_to_capacity: u64,
    }

    /// Drive one sequence against the real slot and the model, comparing the
    /// full live contents after **every** operation.
    ///
    /// Contents and not just the length: a rewind is a length decrement, so a
    /// defect that decrements correctly and leaves the wrong bytes reachable is
    /// invisible to a length check. That is §3.1b's *shape invariant* trap --
    /// `[4,4,4]` is as flat as `[3,3,3]` -- and the reason this compares an
    /// absolute reference.
    fn run_slot_sequence(
        capacity: usize,
        ops: &[SlotOp],
        dev: &Device,
    ) -> std::result::Result<SlotTransitions, String> {
        let mut slot = KvSlot::new(capacity);
        let mut model = SlotModel {
            live: Vec::new(),
            heads: 2,
        };
        let mut t = SlotTransitions::default();
        // A monotonically increasing step id, so a position written after a
        // discard carries a *different* value from the one it replaced. Without
        // this the reuse case is invisible: rewriting a discarded slot with the
        // same bytes proves nothing (#11.3l finding 3, and #574's mutant 2).
        let mut next_step = 0usize;

        for (i, op) in ops.iter().enumerate() {
            match *op {
                SlotOp::Append { n } => {
                    let ids: Vec<usize> = (0..n)
                        .map(|_| {
                            let s = next_step;
                            next_step += 1;
                            s
                        })
                        .collect();
                    let src =
                        multi_token(&ids, model.heads, dev).map_err(|e| format!("op {i}: {e}"))?;
                    match slot.append(&src) {
                        Ok(_) => {
                            model.live.extend_from_slice(&ids);
                            t.appends += 1;
                        }
                        Err(_) => {
                            // Refusal is the documented behaviour past capacity.
                            if model.live.len() + n <= capacity {
                                return Err(format!(
                                    "op {i}: append({n}) refused at len {} with capacity \
                                     {capacity}, where it fits",
                                    model.live.len()
                                ));
                            }
                            t.capacity_refusals += 1;
                        }
                    }
                }
                SlotOp::Speculate { k, accept } => {
                    let ids: Vec<usize> = (0..k)
                        .map(|_| {
                            let s = next_step;
                            next_step += 1;
                            s
                        })
                        .collect();
                    let src =
                        multi_token(&ids, model.heads, dev).map_err(|e| format!("op {i}: {e}"))?;
                    let had = model.live.len();
                    match slot.append(&src) {
                        Ok(_) => {
                            model.live.extend_from_slice(&ids);
                            let discard = k - accept;
                            if discard > 0 {
                                slot.rewind(discard).map_err(|e| {
                                    format!("op {i}: rewind({discard}) of a {k}-window failed: {e}")
                                })?;
                                model.live.truncate(had + accept);
                                if had + accept < had + k {
                                    t.rewrote_discarded_positions += 1;
                                }
                            }
                            t.speculations += 1;
                            if accept == k {
                                t.full_accepts += 1;
                            } else if accept == 0 {
                                t.full_rejects += 1;
                            } else {
                                t.partial_accepts += 1;
                            }
                        }
                        Err(_) => {
                            if had + k <= capacity {
                                return Err(format!(
                                    "op {i}: a {k}-position window was refused at len {had} \
                                     with capacity {capacity}, where it fits"
                                ));
                            }
                            t.capacity_refusals += 1;
                        }
                    }
                }
                SlotOp::FillToCapacityThenOne => {
                    // Exactly to the boundary: this must be ADMITTED.
                    let room = capacity - model.live.len();
                    if room > 0 {
                        let ids: Vec<usize> = (0..room)
                            .map(|_| {
                                let s = next_step;
                                next_step += 1;
                                s
                            })
                            .collect();
                        let src = multi_token(&ids, model.heads, dev)
                            .map_err(|e| format!("op {i}: {e}"))?;
                        slot.append(&src).map_err(|e| {
                            format!(
                                "op {i}: filling to exactly capacity {capacity} from len {} \
                                 was refused: {e}. The bound is `len + seq_len > capacity`, \
                                 so landing ON the capacity must be admitted.",
                                model.live.len()
                            )
                        })?;
                        model.live.extend_from_slice(&ids);
                        t.filled_to_capacity += 1;
                    }
                    // One past it: this must be REFUSED.
                    let one = [next_step];
                    next_step += 1;
                    let src =
                        multi_token(&one, model.heads, dev).map_err(|e| format!("op {i}: {e}"))?;
                    if slot.append(&src).is_ok() {
                        return Err(format!(
                            "op {i}: a position past capacity {capacity} was ADMITTED at len \
                             {}. The slot returns Err rather than reallocating, and a write \
                             past the allocation is what DESIGN.md §3.5 says nothing catches.",
                            model.live.len()
                        ));
                    }
                    t.capacity_refusals += 1;
                }
                SlotOp::RewindPastStart => {
                    let too_many = model.live.len() + 1;
                    let before = model.live.len();
                    if slot.rewind(too_many).is_ok() {
                        return Err(format!(
                            "op {i}: rewind({too_many}) succeeded at len {before}; \
                             `resolve(n)` may only discard positions this speculation \
                             appended, and a rewind past the start silently exposes \
                             positions an earlier turn wrote"
                        ));
                    }
                    t.rewinds_refused += 1;
                }
                SlotOp::Reset { heads } => {
                    if !model.live.is_empty() {
                        t.resets_from_nonempty += 1;
                    }
                    if heads != model.heads {
                        t.turns_changed_shape += 1;
                    }
                    slot.reset();
                    model.live.clear();
                    model.heads = heads;
                }
            }

            // --- the comparison, after every operation ----------------------
            let want = &model.live;
            match slot.all_data.as_ref() {
                None => {
                    if !want.is_empty() {
                        return Err(format!(
                            "op {i} ({op:?}): slot has no buffer but the model holds {} \
                             positions",
                            want.len()
                        ));
                    }
                }
                Some(all) => {
                    if slot.len != want.len() {
                        return Err(format!(
                            "op {i} ({op:?}): slot.len={} model={}",
                            slot.len,
                            want.len()
                        ));
                    }
                    if slot.len > 0 {
                        let view = all
                            .narrow(2, 0, slot.len)
                            .map_err(|e| format!("op {i}: narrow failed: {e}"))?;
                        let got = view
                            .flatten_all()
                            .and_then(|v| v.to_vec1::<f32>())
                            .map_err(|e| format!("op {i}: readback failed: {e}"))?;
                        // The SHAPE is part of the comparison, not only the
                        // values: a buffer kept across a turn that changed
                        // shape has the wrong dims with the right length.
                        let dims = view.dims().to_vec();
                        if dims != [1, model.heads, want.len(), 3] {
                            return Err(format!(
                                "op {i} ({op:?}): live view has dims {dims:?}, expected \
                                 [1, {}, {}, 3]. A buffer kept across a turn boundary pins \
                                 the previous turn's shape -- `Cache::clear`'s doc gives \
                                 exactly this reason for dropping the allocation rather \
                                 than rewinding the length.",
                                model.heads,
                                want.len()
                            ));
                        }
                        let expect = expected_values(want, model.heads);
                        if got != expect {
                            return Err(format!(
                                "op {i} ({op:?}): live contents diverge at len {}.\n  \
                                 slot  = {:?}\n  model = {:?}\n\
                                 A length that matches with contents that do not is the \
                                 shape a length-only assertion cannot see (CONTRIBUTING \
                                 §3.1b).",
                                slot.len,
                                &got[..got.len().min(12)],
                                &expect[..expect.len().min(12)]
                            ));
                        }
                    }
                }
            }
            if slot.len > capacity {
                return Err(format!(
                    "op {i} ({op:?}): len {} exceeds capacity {capacity}",
                    slot.len
                ));
            }
        }
        Ok(t)
    }

    /// `[1, heads, seq, 3]` whose every element is a function of its step id
    /// and head, so a position written by a different step is a *different*
    /// value and a buffer kept across a shape change is a *shape* error.
    fn multi_token(ids: &[usize], heads: usize, dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(expected_values(ids, heads), (1, heads, ids.len(), 3), dev)
    }

    /// The flattened values a `narrow(2, 0, len)` view must hold.
    ///
    /// Dim order is `[b, h, seq, d]`, so the seq axis is second-innermost and
    /// the values are laid out h-major over seq.
    fn expected_values(ids: &[usize], heads: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(ids.len() * heads * 3);
        for h in 0..heads {
            for &id in ids {
                for d in 0..3 {
                    v.push((id * 100 + h * 3 + d) as f32);
                }
            }
        }
        v
    }

    use proptest::prelude::*;

    fn slot_op_strategy(capacity: usize) -> impl Strategy<Value = SlotOp> {
        prop_oneof![
            8 => (1usize..=4).prop_map(|n| SlotOp::Append { n }),
            6 => (1usize..=4).prop_flat_map(|k| (Just(k), 0..=k))
                .prop_map(|(k, accept)| SlotOp::Speculate { k, accept }),
            1 => Just(SlotOp::RewindPastStart),
            2 => Just(SlotOp::FillToCapacityThenOne),
            // The head count VARIES across turns. A constant one makes the
            // fixture blind to `reset` keeping its buffer -- measured, and the
            // reason the symbol carries the shape at all.
            1 => prop::sample::select(vec![1usize, 2, 4]).prop_map(|heads| SlotOp::Reset { heads }),
        ]
        .prop_filter("window must fit the capacity", move |op| match op {
            SlotOp::Speculate { k, .. } => *k <= capacity,
            SlotOp::Append { n } => *n <= capacity,
            _ => true,
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 4096, ..ProptestConfig::default() })]

        /// The slot's live contents match a model of its contract across an
        /// arbitrary sequence of appends, speculative windows and turns.
        ///
        /// **The axis no existing test carries.** The five tests above drive at
        /// most six appends and no rewind at all; the two `lfm2_spec_conv_*`
        /// integration files reach two windows with `k <= 3`. A partial accept
        /// followed by a re-append writes *different* content to a position a
        /// discarded speculation already wrote, and that is the in-flight-reuse
        /// shape #23 is the precedent for -- at the level of a length rather
        /// than of a buffer.
        #[test]
        fn the_slot_matches_its_model_across_speculation_and_turns(
            capacity in 4usize..=64,
            ops in prop::collection::vec(slot_op_strategy(64), 1..80),
        ) {
            let ops: Vec<SlotOp> = ops
                .into_iter()
                .filter(|op| match op {
                    SlotOp::Speculate { k, .. } => *k <= capacity,
                    SlotOp::Append { n } => *n <= capacity,
                    _ => true,
                })
                .collect();
            if let Err(e) = run_slot_sequence(capacity, &ops, &Device::Cpu) {
                return Err(TestCaseError::fail(e));
            }
        }
    }

    /// The generator reaches every transition, and names the ones it does not.
    ///
    /// #575's *report transitions taken, not lines executed*. Without this the
    /// property above is #504's `evicted = 0`: a green result over a generator
    /// that never produced a rejection.
    #[test]
    fn the_slot_generator_reaches_every_transition() {
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::{Config, TestRunner};

        let mut runner = TestRunner::new(Config {
            cases: 256,
            ..Config::default()
        });
        let strategy = prop::collection::vec(slot_op_strategy(16), 40..80);
        let mut total = SlotTransitions::default();
        for _ in 0..256 {
            let ops = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let t = run_slot_sequence(16, &ops, &Device::Cpu)
                .expect("engagement run diverged; the property should have caught it");
            total.appends += t.appends;
            total.speculations += t.speculations;
            total.full_accepts += t.full_accepts;
            total.full_rejects += t.full_rejects;
            total.partial_accepts += t.partial_accepts;
            total.rewinds_refused += t.rewinds_refused;
            total.capacity_refusals += t.capacity_refusals;
            total.resets_from_nonempty += t.resets_from_nonempty;
            total.rewrote_discarded_positions += t.rewrote_discarded_positions;
            total.turns_changed_shape += t.turns_changed_shape;
            total.filled_to_capacity += t.filled_to_capacity;
        }

        println!("FUZZ575 kvslot transitions: {total:?}");
        let named: [(&str, u64); 11] = [
            ("turns_changed_shape", total.turns_changed_shape),
            ("filled_to_capacity", total.filled_to_capacity),
            ("appends", total.appends),
            ("speculations", total.speculations),
            ("full_accepts", total.full_accepts),
            ("full_rejects", total.full_rejects),
            ("partial_accepts", total.partial_accepts),
            ("rewinds_refused", total.rewinds_refused),
            ("capacity_refusals", total.capacity_refusals),
            ("resets_from_nonempty", total.resets_from_nonempty),
            (
                "rewrote_discarded_positions",
                total.rewrote_discarded_positions,
            ),
        ];
        let unreached: Vec<&str> = named
            .iter()
            .filter(|(_, n)| *n == 0)
            .map(|(s, _)| *s)
            .collect();
        assert!(
            unreached.is_empty(),
            "the generator never reached: {unreached:?}. An unreached transition is the \
             finding (#575)."
        );
    }

    /// **`transpose(1, 2)` is CONTIGUOUS at `seq_len == 1`, so the Q/K/V layout
    /// change costs nothing on the decode path** (#601).
    ///
    /// `Attention::attend` reshapes each projection to
    /// `[b, seq, heads, head_dim]` and transposes to `[b, heads, seq, head_dim]`
    /// (`lfm2.rs:2015-2024`). At decode `seq` is 1, so the swapped axis has
    /// **extent 1**: its stride is unreachable and the layout is still row-major.
    /// `Tensor::contiguous()` therefore takes its `is_contiguous()` short-circuit
    /// (`tensor.rs:2670`) and returns an `Arc` clone — no `alloc_uninit`, no
    /// `copy_strided_src`, no dispatch.
    ///
    /// Pinned as an **absolute reference** rather than a shape
    /// (`CONTRIBUTING.md` §3.1b rule 1): the exact `seq_len` at which the
    /// property flips is asserted, so a change that made the decode transpose
    /// strided fails here with a sentence rather than passing an "all equal".
    /// The `seq_len > 1` arm is the **control that pins the level** (rule 2) —
    /// without it, `is_contiguous == true` everywhere would satisfy the test
    /// while meaning the probe had stopped reading the layout.
    ///
    /// Corroborated on the GPU by the dispatch census: of the 38 `copy2d_f16`
    /// in a steady-state decode token, **22 are the conv ring and 16 are the KV
    /// append** (`grid=(8, 64, 1)`, writing at `kv_len * head_dim * 2` and
    /// advancing 128 B per step) — **none is a transpose copy**.
    #[test]
    fn decode_transpose_is_contiguous_and_prefill_is_not() -> Result<()> {
        let dev = Device::Cpu;
        // LFM2.5-VL-3B geometry (§5): 32 query heads, 8 KV heads, head_dim 64.
        for (label, heads) in [("q", 32usize), ("k/v", 8usize)] {
            let decode = Tensor::zeros((1usize, 1usize, heads, 64usize), DType::F16, &dev)?
                .transpose(1, 2)?;
            assert!(
                decode.is_contiguous(),
                "{label}: [1, 1, {heads}, 64].transpose(1, 2) must be contiguous — this is \
                 why the decode path pays no layout copy (#601). If this fails, the copy \
                 #601 describes has become real and its two directions are worth building."
            );

            // The control that pins the level: the same expression at seq_len > 1
            // IS strided, so the assertion above is reading the layout rather
            // than reporting a constant.
            for seq in [2usize, 34, 736] {
                let prefill = Tensor::zeros((1usize, seq, heads, 64usize), DType::F16, &dev)?
                    .transpose(1, 2)?;
                assert!(
                    !prefill.is_contiguous(),
                    "{label}: [1, {seq}, {heads}, 64].transpose(1, 2) must be STRIDED. If \
                     this fails the probe has stopped reading the layout and the decode \
                     assertion above proves nothing."
                );
            }
        }
        Ok(())
    }
}
