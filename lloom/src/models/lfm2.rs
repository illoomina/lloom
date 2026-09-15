//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::nn::ops::{FlashScratchSizing, KvQuant};
use crate::nn::{
    BatchedCausalLM, CacheReset, CausalLM, Conv1d, Conv1dConfig, Model as ModelTrait, RaggedRow,
    Row, Speculative, VarBuilder,
};
use crate::utils::repeat_kv;
use lloom_macros::Config as ConfigDerive;
// The SiLU steer arm is a Metal kernel switch (`unary.rs`), so the type and its
// axis field are metal-gated exactly as its former crate dependency was.
#[cfg(feature = "metal")]
use lloom_tensor::kernels::metal::{GpuClock, KvPagePool, PageQuarantine};
#[cfg(feature = "metal")]
use lloom_tensor::kernels::SiluImpl;
use lloom_tensor::{DType, Device, IndexOp, Module, Result, Tensor};
use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "metal")]
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

/// The nested RoPE block newer LFM2 configs carry in place of a flat
/// `rope_theta` (`{"rope_theta": 1e7, "rope_type": "default"}`).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Lfm2RopeParameters {
    pub rope_theta: Option<f32>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    // Newer LFM2 checkpoints nest the base frequency under `rope_parameters`
    // and omit the flat `rope_theta` (LFM2.5-2.6B: `rope_parameters.rope_theta`
    // = 1e7). When present it wins in `into_config`; the flat field's default is
    // the fallback for older configs. Getting this wrong is silent — a default
    // 1e6 against a checkpoint's 1e7 corrupts every RoPE rotation without an
    // error, which is why the nested form is read here rather than in a harness.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_parameters: Option<Lfm2RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    // HF checkpoints spell this `tie_word_embeddings`, and one checkpoint
    // carries BOTH spellings (LFM2.5-VL-3B's `text_config` has `tie_embedding`
    // beside `tie_word_embeddings`), so the HF key is a second field rather
    // than a serde alias: an alias makes two keys one field, and serde refuses
    // a field it sees twice. A tied checkpoint (LFM2.5-2.6B: no
    // `lm_head.weight`, reuses `embed_tokens.weight`) loads through either key
    // without a harness normalizing it first; `into_config` folds them.
    #[serde(default)]
    pub tie_embedding: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // FFN dimension configuration.
    //
    // LFM2 has two ways to size the FFN, and a checkpoint picks one. When
    // `block_auto_adjust_ff_dim` is true (or absent), the dimension is DERIVED
    // as `hidden_size * 4 * block_ffn_dim_multiplier` rounded to
    // `block_multiple_of` (`compute_intermediate_size`). When it is false, the
    // checkpoint states `intermediate_size` EXPLICITLY and that value is used
    // verbatim — the derived formula does not match. LFM2.5-VL-3B is the
    // auto-adjust case; LFM2.5-2.6B sets `intermediate_size: 10752` with
    // `block_auto_adjust_ff_dim: false`, which the formula would misread as
    // 8192 and fail to load `feed_forward.w1` against.
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_block_auto_adjust_ff_dim")]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: f32,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: usize,
    /// Every checkpoint key not named by a typed field above, kept losslessly
    /// (pydantic `extra="allow"`). Zero runtime cost when a checkpoint carries
    /// no unknown keys: the flatten is resolved by serde, and an empty map
    /// allocates nothing. (`RESTRUCTURING_DESIGN.md` §1.6)
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
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

// Absent means "derive the FFN dim from the formula", which is how the
// checkpoints predating this field are read. A checkpoint that states
// `intermediate_size` explicitly also carries `block_auto_adjust_ff_dim: false`.
fn default_block_auto_adjust_ff_dim() -> bool {
    true
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
        // The FFN dimension: the explicit `intermediate_size` when the
        // checkpoint disables auto-adjust (LFM2.5-2.6B: 10752), else the derived
        // formula (LFM2.5-VL-3B: hidden*4 rounded to block_multiple_of = 8192).
        // Reading the explicit value is what lets a non-auto-adjust checkpoint
        // load its `feed_forward.w1` at all.
        let intermediate_size = match self.intermediate_size {
            Some(n) if !self.block_auto_adjust_ff_dim => n,
            _ => self.compute_intermediate_size(),
        };
        // Nested `rope_parameters.rope_theta` wins over the flat field's default
        // when a checkpoint carries it (LFM2.5-2.6B omits the flat one).
        let rope_theta = self
            .rope_parameters
            .as_ref()
            .and_then(|rp| rp.rope_theta)
            .unwrap_or(self.rope_theta);
        Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types: self.layer_types,
            // Tied iff either spelling says so: `tie_embedding` is the
            // candle spelling a harness may have normalized to, and
            // `tie_word_embeddings` the HF one.
            tie_embedding: self.tie_embedding || self.tie_word_embeddings.unwrap_or(false),
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
            gemv_gate: false,
            conv_fused: false,
            conv_prefill_ctx: false,
            fusion: FusionPolicy::default(),
            qkv_layout: QkvLayout::default(),
            // No tap, per §7.1a: an empty `tap_layers` is what every existing
            // caller had before #363 and costs an `is_empty()` per layer. The
            // indices belong to a *drafter* checkpoint, so there is no
            // meaningful default for a target config to carry.
            tap_layers: Vec::new(),
            tap_dtype: DType::F32,
            // The default SiLU arm, `Exact` — an unconfigured process is
            // byte-for-byte what it was (§7.1a). The kernel atomic is set from
            // this at construction (§1.6). Metal-only, like the kernel it steers.
            #[cfg(feature = "metal")]
            silu_impl: SiluImpl::default(),
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
    pub(crate) fn width(&self, l_cache: usize) -> usize {
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
    /// `crate::nn::ops::sdpa`, which is GQA-native and accumulates in F32
    /// internally, so neither `repeat_kv` nor the upcast is needed.
    ///
    /// Metal only. `seq_len == 1` takes `sdpa_vector`; `seq_len > 1` — prefill,
    /// a verify window, B>1 — takes the tiled `steel_attention`
    /// (`Attention::sdpa_full_applies`). Every other case falls back to
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
    /// Metal only, `seq_len == 1` only, f16/f32 only. `seq_len > 1` takes the
    /// tiled `steel_attention` as `Sdpa` does; any other case falls back to
    /// `Generic`.
    FlashDecoding,
}

impl AttnImpl {
    /// Parse the `--attn` flag arm. The spellings are the ones every harness
    /// used inline before `#[derive(Config)]` consolidated them (§3.6).
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "generic" => Ok(AttnImpl::Generic),
            "sdpa" => Ok(AttnImpl::Sdpa),
            "flash" => Ok(AttnImpl::FlashDecoding),
            other => Err(format!(
                "--attn must be `generic`, `sdpa` or `flash`, got `{other}`"
            )),
        }
    }
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
    /// One page pool per attention layer, shared by every sequence, addressed
    /// through per-sequence page tables (`DESIGN.md` §10.3c–§10.3h).
    ///
    /// The write is `Tensor::kv_scatter_into` at the slots the table names and
    /// the decode read is FlashDecoding over the pool through the table, so
    /// this arm requires `AttnImpl::FlashDecoding` and refuses anything else.
    /// A fresh pool hands out pages in order and is byte-identical to `InPlace`
    /// in what it stores; a recycled one is not, which is the arm §10.3h's
    /// hazard needs and a contiguous cache cannot produce.
    Paged,
}

impl KvAppend {
    /// Parse the `--kv-append` flag arm. The spellings match the harnesses'
    /// prior inline parse (§3.6).
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "cat" => Ok(KvAppend::Cat),
            "in-place" => Ok(KvAppend::InPlace),
            "paged" => Ok(KvAppend::Paged),
            other => Err(format!(
                "--kv-append must be `cat`, `in-place` or `paged`, got `{other}`"
            )),
        }
    }
}

/// How the attention projections reach their norms and the KV append: the
/// `reshape → transpose(1, 2)` view is a copy at `seq_len > 1` and free at
/// `seq_len == 1` (`DESIGN.md` §6.2c), and this axis decides what is done
/// with it (`DESIGN.md` §6.2c, §12.2 row 6).
///
/// A construction-tier axis like [`KvAppend`]: every arm is compiled, the
/// incumbent is the default, and the A/B is one field apart (§7.1a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QkvLayout {
    /// `.contiguous()` on all three views, then `q_norm`/`k_norm` and the
    /// append over dense tensors -- the shipping path. Three copies per
    /// attention layer per multi-token pass; three clones at decode.
    #[default]
    Contiguous,
    /// No copy: `q_norm`/`k_norm` read the transposed view through the
    /// strided form of the same kernel (`rmsnorm_*_strided`, bit for bit the
    /// contiguous kernel over a copy), and `V` is written into the KV slab
    /// from its own layout by one strided-to-strided copy (`copy_nd`).
    Strided,
    /// `Strided`, and `q_norm + rope` / `k_norm + rope` each ONE dispatch
    /// (`rmsnorm_rope_*`), so the norm's output never round-trips memory
    /// before the rotation: `DESIGN.md` §12.2 row 6 at the norm seam rather
    /// than the GEMV seam §12.2e rules out. Falls back to `Strided` where
    /// the rows attend at different positions (`attend_rows`).
    Fused,
}

impl QkvLayout {
    /// Parse the `--qkv-layout` flag arm.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "contiguous" => Ok(QkvLayout::Contiguous),
            "strided" => Ok(QkvLayout::Strided),
            "fused" => Ok(QkvLayout::Fused),
            other => Err(format!(
                "--qkv-layout must be `contiguous`, `strided` or `fused`, got `{other}`"
            )),
        }
    }

    /// The arm as a `RESULT`-line word.
    pub fn as_str(self) -> &'static str {
        match self {
            QkvLayout::Contiguous => "contiguous",
            QkvLayout::Strided => "strided",
            QkvLayout::Fused => "fused",
        }
    }
}

/// The KV ceiling the profiling harness and `lloom-serve` reserve by default,
/// in tokens.
///
/// **Not what [`Cache::new`] uses.** A cache built through `new` takes the
/// model's own `max_position_embeddings` as its ceiling and grows toward it in
/// buckets of [`DEFAULT_KV_BUCKET`] tokens, each bucket admitted against the
/// device before it is allocated (`Cache::prepare_pass`). This constant is the
/// explicit `--kv-capacity` default of the harnesses that sweep the ceiling as
/// an axis, kept at the value every recorded run of theirs was taken at.
pub const DEFAULT_KV_CAPACITY: usize = 4096;

/// How many tokens the KV slab grows by when a pass would overrun it.
///
/// # Why 4096
///
/// A grow costs one allocation plus one copy of the live slab per K and V
/// plane of every attention layer, so the bucket trades over-allocation
/// against how often that is paid. At LFM2's 16 KiB per position (§5.6's
/// geometry, `admission::KV_BYTES_PER_TOKEN`) the bucket is 64 MiB at B=1,
/// which is also the most a run can hold past what it uses. Against the full
/// 131 072 positional ceiling it is 31 grows, one every ~75 s of decode at
/// 55 tok/s, and the copies sum to ~31 GiB — a tenth of a second at §3.4's
/// 349.7 GB/s spread over a forty-minute reply. The config's
/// `block_multiple_of` (256) would pay 511 grows and hand the buffer pool 511
/// distinct slab sizes, each a bucket it cannot reuse for the next; coarser
/// (16 384) would hold 256 MiB for a one-line answer. 4096 is also the
/// ceiling every recorded digest was taken under, so a reply that fits in one
/// bucket allocates exactly what it did before growth existed.
pub const DEFAULT_KV_BUCKET: usize = 4096;

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
pub fn report_weight_divergence(r: &lloom_tensor::metal_backend::admission::WeightReconciliation) {
    eprintln!("lfm2: {}", r.describe());
}

/// Who decides which fusions a decode step takes (`DESIGN.md`
/// `optimizer-design.md` §0.2, issue #732).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusionPolicy {
    /// The per-site flags as set: `gemv_silu`, `gemv_residual`, `gemv_gate`.
    /// The incumbent, and the arm every recorded digest belongs to.
    #[default]
    Manual,
    /// The optimizer derives the plan from the step's IR and the fragment
    /// catalogue at construction (`crate::optimizer`), and the derived plan
    /// REPLACES the three epilogue flags. `conv_fused` is not an elementwise
    /// fusion and stays as set.
    Derived,
}

impl FusionPolicy {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "manual" => Ok(FusionPolicy::Manual),
            "derived" => Ok(FusionPolicy::Derived),
            other => Err(format!(
                "--fusion must be `manual` or `derived`, got `{other}`"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FusionPolicy::Manual => "manual",
            FusionPolicy::Derived => "derived",
        }
    }
}

/// What `Fusion=Derived` decided at construction, kept on the model for the
/// record (`optimizer-design.md` §10c): the plan, what the exclusion set said
/// about every transform the config selects — under the fp model each one's
/// kernel computes in — which stateful regions carry more than one transform
/// and whether a joint validation covers them, and the two closed-form gains
/// §5a names. Nothing here is read per token.
#[derive(Debug, Clone, Default)]
pub struct FusionReport {
    /// The derived plan (`FusionPlan`'s line).
    pub plan: String,
    pub gemv_silu: bool,
    pub gemv_residual: bool,
    pub gemv_gate: bool,
    /// The build's math mode the model was constructed under.
    pub math_mode: &'static str,
    /// Fragments the policy refused: (name, why).
    pub refused: Vec<(String, String)>,
    /// Selected transforms the set admits under the model they run: (name,
    /// the kernel family it was looked up as).
    pub validated: Vec<(String, String)>,
    /// Selected transforms and fragments with no verdict under the model they
    /// run: (name, why) — no entry, recorded under another model, or a
    /// kernel with no declared model. Reported, not refused.
    pub unvalidated: Vec<(String, String)>,
    /// Stateful regions on which the config selects more than one transform.
    pub compositions: Vec<Composition>,
    /// Fused sites the plan placed inside a recurrence: (site, fragment,
    /// region).
    pub fused_in_stateful: Vec<(String, String, String)>,
    /// The conv fold's gain, from the loaded weights.
    pub conv_fold_gain: Option<ConvFoldGain>,
}

/// Several transforms selected on one recurrence. Their separate verdicts
/// were each measured through the recurrence alone; they compose there only
/// through a JOINT entry (§5a), named here if the set has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    pub region: String,
    pub transforms: Vec<String>,
    /// The joint entry's name when the set validates the combination under
    /// the kernel that runs; `None` when it does not.
    pub joint: Option<String>,
}

/// The conv fold's gain on a discrepancy in its state window — `Σ_j |w[c, j]|`
/// per channel, the ∞-norm of the fold as a map from the window to its
/// output — over every conv layer's loaded weights (§5a's closed form). The
/// window itself is carried by a shift or a ring write: a permutation plus
/// one new column, gain exactly 1, and it holds `taps − 1` past values, so a
/// discrepancy leaves the window after `taps − 1` tokens by construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvFoldGain {
    pub layers: usize,
    pub taps: usize,
    pub channels: usize,
    pub max: f32,
    pub min: f32,
    /// Channels whose fold is expansive (`L > 1`).
    pub over_one: usize,
}

impl std::fmt::Display for FusionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lfm2 fusion=derived: {}; gemv_silu={} gemv_residual={} gemv_gate={} math={} ({} refused, {} validated, {} unvalidated",
            self.plan,
            self.gemv_silu,
            self.gemv_residual,
            self.gemv_gate,
            self.math_mode,
            self.refused.len(),
            self.validated.len(),
            self.unvalidated.len()
        )?;
        if !self.unvalidated.is_empty() {
            write!(f, " [")?;
            for (i, (name, why)) in self.unvalidated.iter().enumerate() {
                if i > 0 {
                    write!(f, "; ")?;
                }
                write!(f, "{name}: {why}")?;
            }
            write!(f, "]")?;
        }
        write!(f, ")")?;
        for c in &self.compositions {
            write!(f, " composition[{}: {}", c.region, c.transforms.join("+"))?;
            match &c.joint {
                Some(j) => write!(f, " validated jointly as `{j}`]")?,
                None => write!(f, " NOT validated jointly]")?,
            }
        }
        if !self.fused_in_stateful.is_empty() {
            write!(f, " fused-in-stateful[")?;
            for (i, (site, frag, region)) in self.fused_in_stateful.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{site}:{frag}@{region}")?;
            }
            write!(f, "]")?;
        }
        if let Some(g) = &self.conv_fold_gain {
            write!(
                f,
                " conv-ring[{} layers: fold gain L∞ max {:.3} min {:.3}, {} of {} channels expansive; window write L=1 (permutation), horizon {} tokens]",
                g.layers,
                g.max,
                g.min,
                g.over_one,
                g.channels,
                g.taps - 1
            )?;
        }
        Ok(())
    }
}

/// The runtime config. The `#[axis]`-tagged fields are the engine's
/// configuration axes (RESTRUCTURING §1.6): each contributes to `lloom::axes::AXES`
/// at link time and becomes a `ConfigArgs` clap flag. The shape fields
/// (`hidden_size`, `layer_types`, …) come from the checkpoint and are not axes.
#[derive(Debug, Clone, ConfigDerive)]
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
    #[axis(
        name = "AttnImpl",
        flag = "attn",
        variants = ["Generic", "Sdpa", "FlashDecoding"],
        default = "Generic",
        parser = "AttnImpl::parse"
    )]
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
    #[axis(
        name = "ScratchSizing",
        flag = "flash-scratch-sizing",
        variants = ["Grow", "Reserve", "Bucket"],
        default = "Grow",
        parser = "FlashScratchSizing::parse"
    )]
    pub flash_scratch_sizing: FlashScratchSizing,
    /// Defaults to `Cat`, for the same reason `attn_impl` defaults to `Generic`.
    #[axis(
        name = "KvAppend",
        flag = "kv-append",
        variants = ["Cat", "InPlace", "Paged"],
        default = "Cat",
        parser = "KvAppend::parse"
    )]
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
    #[axis(
        name = "KvQuant",
        flag = "kv-quant",
        variants = ["Off", "Int8", "Int4"],
        default = "Off",
        parser = "KvQuant::parse"
    )]
    pub kv_quant: KvQuant,
    /// Defaults to `Shuffle`, so every existing caller keeps §6.1's path.
    #[axis(
        name = "ConvState",
        flag = "conv-state",
        variants = ["Shuffle", "SlidingRing", "RotatingRing"],
        default = "Shuffle",
        parser = "ConvState::parse"
    )]
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
    #[axis(name = "GemvSilu", flag = "gemv-silu", variants = ["off", "on"], default = "off", bool)]
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
    #[axis(name = "GemvResidual", flag = "gemv-residual", variants = ["off", "on"], default = "off", bool)]
    pub gemv_residual: bool,
    /// Fuse the SwiGLU product `silu(w1(x)) * w3(x)` onto `w3`'s GEMV store
    /// (issue #732, `DESIGN.md` §12.2e row 3b at the producer end): the gate
    /// is read at the storing thread's own output row, the residual's shape
    /// with `*` for `+`. 30 `bmul_f16` dispatches per token.
    ///
    /// **Defaults to off**, per §7.1a. Metal, `seq_len == 1`; every other case
    /// takes the eager path. **Moves the digests by construction**: the
    /// product lands at the f32 accumulator on the unrounded up-projection
    /// where the pair multiplies two rounded stores at `T` (§2.3.5a).
    #[axis(name = "GemvGate", flag = "gemv-gate", variants = ["off", "on"], default = "off", bool)]
    pub gemv_gate: bool,
    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// `conv1d` dispatch (§12.2 row 1). Off by default (§7.1a): it MOVES the
    /// digest — the depthwise kernel accumulates serially ascending in f32 where
    /// `sum_keepdim` uses a `simd_sum` tree — so an unmoved-digest gate cannot
    /// clear it and the discriminator is CPU-backend parity (§2.3.5a). The win
    /// is on the dispatch axis: ~2–3 dispatches per conv layer become 1, and
    /// LFM2 has 22 conv layers, so it is the largest single dispatch reduction
    /// the catalogue leaves open (llama.cpp's decode conv is one kernel).
    #[axis(name = "ConvFused", flag = "conv-fused", variants = ["off", "on"], default = "off", bool)]
    pub conv_fused: bool,
    /// Carry the conv state across a multi-token pass onto a warm cache, as the
    /// convolution's left context (`DESIGN.md` §13.2f defect 1, issue #677).
    ///
    /// The `seq_len > 1` conv writer zero-pads on the left, which is exact for
    /// the first pass and wrong for every later one: the first `l_cache - 1`
    /// outputs of a follow-up turn — or of a chunk, or of a prefix-reuse suffix
    /// — convolve over zeros where the previous pass's last `bx` columns belong.
    /// `on` prepends those columns instead, so a pass onto a warm cache computes
    /// exactly what one pass over the concatenation computes. **Off by default**
    /// (§7.1a): every recorded `--turns 3` digest belongs to the zero-padded
    /// arm, and flipping it is its own argued decision.
    #[axis(name = "ConvPrefillCtx", flag = "conv-prefill-ctx", variants = ["off", "on"], default = "off", bool)]
    pub conv_prefill_ctx: bool,
    /// Whether the epilogue fusions are selected by their flags or derived by
    /// the optimizer at construction (issue #732, P1).
    ///
    /// **Defaults to `Manual`** (§7.1a): the derivation is a mechanism, and
    /// every recorded digest belongs to the flag-selected arms. Under
    /// `Derived` the plan is derived once in [`Model::new`] from
    /// [`crate::models::lfm2_ir::decode_step`] — nothing runs per token — and
    /// `gemv_silu`/`gemv_residual` are set from it, so a `Derived` run and the
    /// `Manual` run whose flags match the plan compute the same bits.
    #[axis(
        name = "Fusion",
        flag = "fusion",
        variants = ["Manual", "Derived"],
        default = "Manual",
        parser = "FusionPolicy::parse"
    )]
    pub fusion: FusionPolicy,
    /// How the Q/K/V `transpose` views reach the norms and the KV append
    /// (`DESIGN.md` §6.2c). Defaults to the copying path, per §7.1a.
    #[axis(
        name = "QkvLayout",
        flag = "qkv-layout",
        variants = ["Contiguous", "Strided", "Fused"],
        default = "Contiguous",
        parser = "QkvLayout::parse"
    )]
    pub qkv_layout: QkvLayout,
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
    /// Which SiLU arm the 30 `usilu` dispatches per token resolve to
    /// (`DESIGN.md` §12.2l, issue #599).
    ///
    /// **The steer value is registered engine-side, cached kernel-side**
    /// (RESTRUCTURING §1.6). Before Wave A the arm lived only as the
    /// `SILU_IMPL` process-wide atomic in `lloom-tensor::kernels`
    /// (`unary.rs:464`), switched by `--silu-impl` and captured on no run
    /// record — the "unlisted ambient switch" §7.1a records as a run-record
    /// corruption (B7). It now rides a `#[axis]` field here, so a run's SiLU
    /// arm is part of its `Config` and its `AXES` registration; the atomic is
    /// the *cache* of this field's value, pushed into the kernel once at model
    /// construction via `lloom_tensor::kernels::set_silu_impl`. The STEER stays
    /// a kernel switch; only its registration moved engine-side.
    ///
    /// **Defaults to `Exact`** (`SiluImpl::default`), so an unconfigured process
    /// is byte-for-byte what it was (§7.1a: no default flipped).
    #[cfg(feature = "metal")]
    #[axis(
        name = "SiluImpl",
        flag = "silu-impl",
        variants = ["Exact", "Recip", "Literal", "Coarse4", "Coarse1", "HardSwish", "Relu"],
        default = "Exact",
        parser = "parse_silu_impl"
    )]
    pub silu_impl: SiluImpl,
}

/// Parse the `--silu-impl` arm into the enum, adapting the kernel layer's
/// `Option`-returning `parse` to the `Result<_, String>` shape the derive's
/// flag apply expects. Metal-only, like the axis it serves.
#[cfg(feature = "metal")]
pub fn parse_silu_impl(s: &str) -> std::result::Result<SiluImpl, String> {
    SiluImpl::parse(s).ok_or_else(|| {
        format!(
            "--silu-impl must be one of `exact`, `recip`, `literal`, `coarse4`, \
             `coarse1`, `hardswish` or `relu`, got `{s}`"
        )
    })
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// A `Config` a test or a `..` update can build from, so a new field is not a
/// breaking change to every struct literal that names the others.
///
/// The six integration-test binaries #575 found dark had each stopped compiling
/// on a `missing field` error one field-add earlier (`DESIGN.md` §15.1k); this
/// is the repair that section names and does not take, deferred there because it
/// is a public-API decision. `#[derive(Default)]` cannot express it — `DType`
/// has no `Default` — so it is hand-written, and its axis fields carry the same
/// arms `into_config` selects (§7.1a: no default is flipped). The shape fields
/// are a minimal valid geometry, not any checkpoint's.
impl Default for Config {
    fn default() -> Self {
        Config {
            vocab_size: 0,
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            norm_eps: default_norm_eps(),
            rope_theta: default_rope_theta(),
            max_position_embeddings: default_max_position_embeddings(),
            conv_l_cache: default_conv_l_cache(),
            conv_bias: false,
            layer_types: Vec::new(),
            tie_embedding: false,
            bos_token_id: None,
            eos_token_id: None,
            use_flash_attn: false,
            attn_impl: AttnImpl::default(),
            flash_page_size: 256,
            flash_pages_per_chunk: 1,
            flash_scratch_sizing: FlashScratchSizing::default(),
            kv_append: KvAppend::default(),
            kv_quant: KvQuant::default(),
            conv_state: ConvState::default(),
            memory_budget: None,
            gemv_silu: false,
            gemv_residual: false,
            gemv_gate: false,
            conv_fused: false,
            conv_prefill_ctx: false,
            fusion: FusionPolicy::default(),
            qkv_layout: QkvLayout::default(),
            tap_layers: Vec::new(),
            tap_dtype: DType::F32,
            #[cfg(feature = "metal")]
            silu_impl: SiluImpl::default(),
        }
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
    /// This mirrors `crate::nn::kv_cache::Cache`, deliberately: that type is the
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
            lloom_tensor::bail!(
                "lfm2 KV rewind past the start: {count} > len {}. \
                 `resolve(n)` may only discard positions this speculation appended.",
                self.len
            )
        }
        self.len -= count;
        Ok(())
    }

    /// Bytes the slab holds. Zero before the first append.
    fn bytes(&self) -> usize {
        self.all_data
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0)
    }

    /// Widen the slab to `capacity` tokens, keeping every position where it is.
    ///
    /// Before the first append this is a number: the slab is allocated at the
    /// new width when the first token arrives. After it, a fresh slab is
    /// allocated at the new width and the old one is copied into its prefix
    /// whole — every position, live or rewound, lands at the offset it had, so
    /// the views this slot hands out afterwards read exactly the bytes the
    /// views before it did. The copy is `slice_set`'s one `copy2d` per plane;
    /// the old slab returns to the pool when the last view of it drops.
    ///
    /// **The buffer identity moves here, and only here.** §6.2b's property —
    /// one `MTLBuffer` at one base every token — holds between grows; a grow
    /// is the one event that re-binds it, paid once per bucket rather than
    /// once per token. A replay executor sees the step as stale and re-records
    /// (§11.3y); the arena sees a drift and re-records (§9.2z). Neither is a
    /// wrong answer, and both are the documented price.
    ///
    /// A `capacity` at or below the current one is a no-op, never a shrink:
    /// the caller decides the width, and this never discards a position.
    fn grow_to(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        if let Some(old) = self.all_data.as_ref() {
            let mut shape = old.dims().to_vec();
            shape[2] = capacity;
            let grown = Tensor::zeros(shape, old.dtype(), old.device())?;
            grown.slice_set(old, 2, 0)?;
            self.all_data = Some(grown);
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Append `src` along dim 2 and return the live prefix as a view.
    ///
    /// Returns `Err` rather than reallocating when the capacity is exhausted:
    /// the slab is widened by `Cache::prepare_pass` before a pass appends, so
    /// reaching this means an append ran outside a prepared pass.
    fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(2)?;
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[2] = self.capacity;
            self.all_data = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }
        let all_data = self.all_data.as_ref().unwrap();
        if self.len + seq_len > self.capacity {
            lloom_tensor::bail!(
                "lfm2 KV cache exhausted: {} + {} > capacity {}. \
                 A pass widens the slab before it appends (`Cache::prepare_pass`); this \
                 append ran outside a prepared pass.",
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
    /// The plane geometry, recorded by `ensure` so a grow can re-lay the
    /// planes at a wider stride without a source tensor in hand.
    b_sz: usize,
    n_heads: usize,
    head_dim: usize,
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
            b_sz: 0,
            n_heads: 0,
            head_dim: 0,
        }
    }

    /// Bytes the three planes hold. Zero before the first append.
    fn bytes(&self) -> usize {
        [&self.codes, &self.scales, &self.biases]
            .into_iter()
            .flatten()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    /// Widen the planes to `capacity` tokens, keeping every position where it
    /// is — `KvSlot::grow_to`'s contract on three planes.
    ///
    /// The planes are flat and `capacity` is their row stride (`[B, H,
    /// capacity, D]` codes, `[B, H, capacity]` scales and biases), so the copy
    /// is per `(batch, head)` row: each old row lands at the head of the
    /// wider row it now has, which is one `copy2d` per plane through a
    /// two-dimensional view of the same bytes. Positions above `len` are
    /// carried too; they are unreachable either way.
    fn grow_to(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        if let Some(codes) = self.codes.as_ref() {
            let rows = self.b_sz * self.n_heads;
            let row = |cap: usize| -> Result<usize> {
                self.quant
                    .code_bytes(cap * self.head_dim)
                    .ok_or_else(|| lloom_tensor::Error::msg("QuantKvSlot::grow_to on the Off arm"))
            };
            let (old_row, new_row) = (row(self.capacity)?, row(capacity)?);
            let old = codes.reshape((rows, old_row))?;
            let grown = Tensor::zeros((rows, new_row), DType::U8, codes.device())?;
            grown.slice_set(&old, 1, 0)?;
            self.codes = Some(grown.reshape(rows * new_row)?);
            for plane in [&mut self.scales, &mut self.biases] {
                if let Some(p) = plane.as_ref() {
                    let old = p.reshape((rows, self.capacity))?;
                    let grown = Tensor::zeros((rows, capacity), p.dtype(), p.device())?;
                    grown.slice_set(&old, 1, 0)?;
                    *plane = Some(grown.reshape(rows * capacity)?);
                }
            }
        }
        self.capacity = capacity;
        Ok(())
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
            lloom_tensor::bail!(
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
            .ok_or_else(|| lloom_tensor::Error::msg("QuantKvSlot::ensure on the Off arm"))?;
        // A u8 plane rather than a typed one: the packing is the kernel's, and
        // a `Tensor` of the code width does not exist at int4 anyway.
        self.codes = Some(Tensor::zeros(code_bytes, DType::U8, device)?);
        let scale_els = b_sz * n_heads * self.capacity;
        self.scales = Some(Tensor::zeros(scale_els, DType::F16, device)?);
        self.biases = Some(Tensor::zeros(scale_els, DType::F16, device)?);
        self.b_sz = b_sz;
        self.n_heads = n_heads;
        self.head_dim = head_dim;
        Ok(())
    }

    /// Quantize `src` and append it, advancing `len`.
    fn append(&mut self, src: &Tensor) -> Result<()> {
        let (b_sz, n_heads, seq_len, head_dim) = src.dims4()?;
        if self.len + seq_len > self.capacity {
            lloom_tensor::bail!(
                "lfm2 quantized KV cache exhausted: {} + {} > capacity {}",
                self.len,
                seq_len,
                self.capacity
            )
        }
        self.ensure(b_sz, n_heads, head_dim, src.device())?;

        let bits = match self.quant {
            KvQuant::Off => lloom_tensor::bail!("QuantKvSlot::append on the Off arm"),
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
        crate::nn::ops::quantize_kv_into(
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

// The `seam::KvCache` impl for LFM2's in-place slot pair
// (`RESTRUCTURING_DESIGN.md` §3.3, N2's "append in place" shape). A `KvSlot`
// stores one of K or V, so the trait — whose `append(k, v)` writes both — is
// implemented on the `(K, V)` pair the model already holds per layer
// (`Cache::kv_slots`). `append` returns the two live prefixes as a `(Tensor,
// Tensor)` view (the paired shape), and `rollback` is the length decrement
// `KvSlot::rewind` already implements: the in-place slot IS the cache
// speculation needs, which the `Cat`/rotating caches above cannot be.
impl crate::nn::kv_cache::seam::KvCache for (KvSlot, KvSlot) {
    type View<'a> = (Tensor, Tensor);

    fn append<'a>(&'a mut self, k: &Tensor, v: &Tensor) -> Result<Self::View<'a>> {
        let out_k = self.0.append(k)?;
        let out_v = self.1.append(v)?;
        Ok((out_k, out_v))
    }

    fn len(&self) -> usize {
        self.0.len
    }

    fn rollback(&mut self, to: usize) -> Result<()> {
        let len = self.0.len;
        if to > len {
            lloom_tensor::bail!("lfm2 KvSlot rollback to {to} past len {len}")
        }
        let discard = len - to;
        self.0.rewind(discard)?;
        self.1.rewind(discard)?;
        Ok(())
    }

    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
}

/// The `KvAppend::Paged` arm's storage (`DESIGN.md` §10.3c–§10.3h, #148).
///
/// One `KvPagePool` holds every sequence's page table — the tables are shared
/// by all attention layers because every layer appends the same positions in
/// the same pass — and each attention layer owns a (K, V) pool of
/// `[1, n_kv_heads, pool_pages * page_size, head_dim]`. A page's tokens are a
/// contiguous run inside a head, so a fresh pool stores exactly what the
/// contiguous cache stores, byte for byte, and the FlashDecoding kernel reads
/// it through the table with a sequence stride of 0.
///
/// Pages are allocated once per `page_size` tokens per sequence, at the first
/// attention layer of a pass; the slot each token of the pass writes to is
/// computed then, once, and reused by every layer. A released sequence's pages
/// go to the `PageQuarantine` against the current GPU epoch and return to the
/// free list only after that epoch completes (§10.3h's one non-fence
/// obligation, §6.7 L4).
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
struct PagedKv {
    page_size: usize,
    pages_per_chunk: usize,
    /// The ceiling, in tokens per sequence.
    kv_capacity: usize,
    /// Tokens per sequence the pool is sized for now; grows by the cache's
    /// bucket toward `kv_capacity` (`Cache::prepare_pass`).
    slab: usize,
    /// Pool size in pages when set; otherwise `n_seqs * ceil(slab /
    /// page_size)`. An explicit size is pinned: it does not grow.
    pool_pages_override: Option<usize>,
    /// Pages allocated and recycled before any sequence exists, so the
    /// sequences' tables are not the identity — the fixture §10.3h needs at
    /// the model level, reached through an ordinary allocation history.
    churn_pages: usize,
    pool: Option<KvPagePool>,
    quarantine: PageQuarantine,
    clock: Arc<GpuClock>,
    /// Per layer, allocated on the layer's first write.
    pools: Vec<Option<(Tensor, Tensor)>>,
    n_seqs: usize,
    /// Live tokens per sequence. Every sequence has the same length: the batch
    /// runs in lockstep (#249's scope).
    len: usize,
    /// Tokens this pass appends.
    pass_len: usize,
    /// `[n_seqs * pass_len]` U32: the token slot of each `(sequence, token)`
    /// this pass writes, and the largest of them (checked against the pool by
    /// the scatter without a readback).
    slots: Option<Tensor>,
    max_slot: usize,
    /// Every sequence's chunk table, `table_stride` entries each.
    table: Vec<u32>,
    table_stride: usize,
}

#[cfg(feature = "metal")]
impl PagedKv {
    fn new(
        config: &Config,
        kv_capacity: usize,
        slab: usize,
        num_layers: usize,
        clock: Arc<GpuClock>,
    ) -> Result<Self> {
        let env_usize = |name: &str| -> Result<Option<usize>> {
            match std::env::var(name) {
                Ok(v) => v.parse::<usize>().map(Some).map_err(|e| {
                    lloom_tensor::Error::msg(format!("{name}={v:?} is not a page count: {e}"))
                }),
                Err(_) => Ok(None),
            }
        };
        Ok(Self {
            page_size: config.flash_page_size.max(1),
            pages_per_chunk: config.flash_pages_per_chunk.max(1),
            kv_capacity,
            slab,
            pool_pages_override: env_usize("LLOOM_148_POOL_PAGES")?,
            churn_pages: env_usize("LLOOM_148_POOL_CHURN")?.unwrap_or(0),
            pool: None,
            quarantine: PageQuarantine::new(),
            clock,
            pools: vec![None; num_layers],
            n_seqs: 0,
            len: 0,
            pass_len: 0,
            slots: None,
            max_slot: 0,
            table: Vec::new(),
            table_stride: 0,
        })
    }

    fn pages_per_seq(&self) -> usize {
        self.slab.div_ceil(self.page_size)
    }

    fn pool_pages(&self) -> usize {
        self.pool_pages_override
            .unwrap_or(self.n_seqs.max(1) * self.pages_per_seq())
    }

    /// Bytes the layer pools hold. Zero before the first write.
    fn bytes(&self) -> usize {
        self.pools
            .iter()
            .flatten()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
    }

    /// Tokens the pool would hold in total at `slab` tokens per sequence.
    fn pool_tokens_at(&self, slab: usize) -> usize {
        match self.pool_pages_override {
            Some(pages) => pages * self.page_size,
            None => self.n_seqs.max(1) * slab.div_ceil(self.page_size) * self.page_size,
        }
    }

    /// Widen the pool to `slab` tokens per sequence — `KvSlot::grow_to`'s
    /// contract on a pool.
    ///
    /// Every layer's pool is reallocated wider and the old one copied into its
    /// prefix, so a page index names the same run of tokens before and after
    /// and every table stays valid; the kernel reads `base + page × stride`,
    /// and only the base moved. The free list gains the new pages *beneath*
    /// the ones it holds (`KvPagePool::grow`), so a recycled page is still
    /// handed out before a fresh one — the order a pool this wide from the
    /// start would have taken. Under `LLOOM_148_POOL_PAGES` the pool is pinned
    /// and exhaustion at it stays loud.
    fn grow_to(&mut self, slab: usize) -> Result<()> {
        let slab = slab.min(self.kv_capacity);
        if slab <= self.slab {
            return Ok(());
        }
        let old_pages = self.pool_pages();
        self.slab = slab;
        let new_pages = self.pool_pages();
        match self.pool.as_mut() {
            Some(pool) if new_pages > old_pages => pool.grow(new_pages - old_pages),
            _ => return Ok(()),
        }
        let tokens = new_pages * self.page_size;
        for layer in self.pools.iter_mut() {
            if let Some((pk, pv)) = layer.as_ref() {
                let mut shape = pk.dims().to_vec();
                shape[2] = tokens;
                let nk = Tensor::zeros(shape.clone(), pk.dtype(), pk.device())?;
                nk.slice_set(pk, 2, 0)?;
                let nv = Tensor::zeros(shape, pv.dtype(), pv.device())?;
                nv.slice_set(pv, 2, 0)?;
                *layer = Some((nk, nv));
            }
        }
        Ok(())
    }

    /// Allocate this pass's pages for every sequence and compute the slots its
    /// tokens write to. Runs once per forward pass, at the first attention
    /// layer, because that is the first point the batch size is known.
    fn begin_pass(&mut self, n_seqs: usize, seq_len: usize, device: &Device) -> Result<()> {
        if self.pool.is_none() {
            self.n_seqs = n_seqs;
            let pool_pages = self.pool_pages();
            // One extra table for the churn sequence, which allocates and
            // leaves before any real sequence exists; a table one entry longer
            // than the pool, so running out is reported as the pool's
            // exhaustion (which names the knob) and never as a full table.
            let mut pool = KvPagePool::new(self.page_size, pool_pages, n_seqs + 1, pool_pages + 1);
            if self.churn_pages > 0 {
                // No command buffer has read these pages, so recycling them
                // without an epoch is safe here and only here.
                pool.append(n_seqs, self.churn_pages * self.page_size)
                    .map_err(lloom_tensor::Error::wrap)?;
                let freed = pool.release(n_seqs);
                pool.recycle(&freed).map_err(lloom_tensor::Error::wrap)?;
            }
            self.pool = Some(pool);
        } else if n_seqs != self.n_seqs {
            lloom_tensor::bail!(
                "lfm2 paged KV: the batch changed from {} to {n_seqs} sequences mid-cache; \
                 clear the cache first",
                self.n_seqs
            )
        }
        // Pages whose last reader has completed come back first, so a long
        // session recycles rather than exhausts.
        let ready = self
            .quarantine
            .drain_completed(self.clock.completed_count());
        let (pool_pages, page_size) = (self.pool_pages(), self.page_size);
        let pool = self.pool.as_mut().expect("created above");
        pool.recycle(&ready).map_err(lloom_tensor::Error::wrap)?;

        let start = self.len;
        let mut slots = Vec::with_capacity(n_seqs * seq_len);
        for s in 0..n_seqs {
            pool.append(s, seq_len).map_err(|e| {
                lloom_tensor::Error::msg(format!(
                    "lfm2 paged KV, sequence {s}: {e}. The pool holds {pool_pages} pages of \
                     {page_size} tokens (LLOOM_148_POOL_PAGES pins it; unpinned it grows \
                     with the KV slab, B x ceil(slab / page_size), toward the ceiling)"
                ))
            })?;
            let pages = pool.seq(s).expect("sequence index within max_seqs").pages();
            for t in start..start + seq_len {
                slots.push(
                    (pages[t / self.page_size] * self.page_size as u32)
                        + (t % self.page_size) as u32,
                );
            }
        }
        self.len = start + seq_len;
        self.pass_len = seq_len;
        self.max_slot = slots.iter().copied().max().unwrap_or(0) as usize;
        self.slots = Some(Tensor::new(slots.as_slice(), device)?);

        // The chunk tables the reader is handed, one per sequence, all the
        // same length because the batch is lockstep.
        let n_chunks = self.len.div_ceil(self.page_size * self.pages_per_chunk);
        self.table_stride = n_chunks;
        self.table.clear();
        for s in 0..n_seqs {
            if !pool.chunk_pages_are_contiguous(s, self.pages_per_chunk) {
                lloom_tensor::bail!(
                    "lfm2 paged KV: sequence {s}'s pages do not form contiguous chunks at \
                     pages_per_chunk={}; a recycled pool needs `--flash-k 1` (DESIGN.md 10.3d)",
                    self.pages_per_chunk
                )
            }
            let t = pool.table_for(s, self.pages_per_chunk);
            debug_assert_eq!(t.len(), n_chunks);
            self.table.extend_from_slice(&t);
        }
        Ok(())
    }

    /// Write this pass's K and V for one layer at the pass's slots.
    fn write_layer(&mut self, block_idx: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let (b, n_kv_heads, seq_len, head_dim) = k.dims4()?;
        if b != self.n_seqs || seq_len != self.pass_len {
            lloom_tensor::bail!(
                "lfm2 paged KV: layer {block_idx} writes [{b}, {n_kv_heads}, {seq_len}, \
                 {head_dim}] against a pass of {} sequences x {} tokens",
                self.n_seqs,
                self.pass_len
            )
        }
        let tokens = self.pool_pages() * self.page_size;
        if self.pools[block_idx].is_none() {
            let shape = (1, n_kv_heads, tokens, head_dim);
            self.pools[block_idx] = Some((
                Tensor::zeros(shape, k.dtype(), k.device())?,
                Tensor::zeros(shape, v.dtype(), v.device())?,
            ));
        }
        let slots = self
            .slots
            .as_ref()
            .expect("begin_pass ran before the first layer");
        let (pk, pv) = self.pools[block_idx].as_ref().expect("allocated above");
        k.kv_scatter_into(pk, slots, self.max_slot, 0, tokens * head_dim, head_dim)?;
        v.kv_scatter_into(pv, slots, self.max_slot, 0, tokens * head_dim, head_dim)?;
        Ok(())
    }

    /// The first `n_prev` tokens of every sequence, gathered from the pool into
    /// a `[n_seqs, n_kv_heads, n_prev, head_dim]` tensor. Prefill only.
    fn gather_layer(
        &self,
        block_idx: usize,
        n_seqs: usize,
        n_prev: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let pool = self.pool.as_ref().expect("a pass has begun");
        let (pk, pv) = self.pools[block_idx]
            .as_ref()
            .expect("the layer has a pool");
        let mut ks = Vec::with_capacity(n_seqs);
        let mut vs = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let pages = pool.seq(s).expect("sequence index within max_seqs").pages();
            let ids: Vec<u32> = (0..n_prev)
                .map(|t| {
                    pages[t / self.page_size] * self.page_size as u32 + (t % self.page_size) as u32
                })
                .collect();
            let ids = Tensor::new(ids.as_slice(), device)?;
            ks.push(pk.index_select(&ids, 2)?);
            vs.push(pv.index_select(&ids, 2)?);
        }
        Ok((Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?))
    }

    /// Drop every sequence, quarantining its pages against the epoch a
    /// command buffer encoded now belongs to.
    fn release_all(&mut self) {
        if let Some(pool) = self.pool.as_mut() {
            let mut freed = Vec::new();
            for s in 0..self.n_seqs {
                freed.extend(pool.release(s));
            }
            self.quarantine
                .quarantine(self.clock.current_epoch(), freed);
        }
        self.len = 0;
        self.pass_len = 0;
        self.slots = None;
        self.max_slot = 0;
        self.table.clear();
        self.table_stride = 0;
    }

    /// Discard the last `n` positions of every sequence: a length decrement,
    /// the pages stay (§10.2a).
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n > self.len {
            lloom_tensor::bail!(
                "lfm2 paged KV rewind past the start: {n} > len {}",
                self.len
            )
        }
        if let Some(pool) = self.pool.as_mut() {
            for s in 0..self.n_seqs {
                pool.rewind(s, n).map_err(lloom_tensor::Error::wrap)?;
            }
        }
        self.len -= n;
        Ok(())
    }
}

/// The KV slab's state, from [`Cache::kv_report`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvReport {
    /// Positions held.
    pub kv_len: usize,
    /// Tokens the slab is sized for now.
    pub slab: usize,
    /// The ceiling the slab grows toward.
    pub capacity: usize,
    /// Tokens per grow.
    pub bucket: usize,
    /// Grows performed since construction or the last `clear`.
    pub grows: usize,
    /// Bytes the KV planes hold on the device.
    pub bytes: usize,
}

/// A grow `Cache::plan_pass` sized and has not performed: the slab it would
/// reach, from which, and what it costs. `Cache::apply_grow` performs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GrowPlan {
    pub(crate) from: usize,
    pub(crate) slab: usize,
    pub(crate) bytes: usize,
    pub(crate) kv_len: usize,
    pub(crate) seq_len: usize,
}

impl GrowPlan {
    /// Where the grow was asked for, for the refusal's message.
    pub(crate) fn describe(&self) -> String {
        format!(
            "at {} + {} positions, widening the slab from {} to {} tokens",
            self.kv_len, self.seq_len, self.from, self.slab
        )
    }
}

/// Bytes a KV grow may take the process's Metal allocation to: `fraction ×
/// working_set`, `working_set` being `recommendedMaxWorkingSetSize` (§9.5c).
pub fn kv_grow_budget(working_set: usize, fraction: f64) -> usize {
    (working_set as f64 * fraction) as usize
}

/// Whether a grow of `grow` bytes fits beside `allocated` under
/// [`kv_grow_budget`]. Pure, so `Cache::prepare_pass`'s memory refusal is
/// tested without a device; exactly on the budget is admitted.
pub fn kv_grow_fits(allocated: usize, working_set: usize, fraction: f64, grow: usize) -> bool {
    allocated.saturating_add(grow) <= kv_grow_budget(working_set, fraction)
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
    /// The `KvAppend::Paged` arm's storage: the page tables, the epoch
    /// quarantine and one (K, V) pool per attention layer.
    #[cfg(feature = "metal")]
    paged: Option<PagedKv>,
    /// Set once per forward pass so the first attention layer of the pass
    /// allocates this pass's pages and slots for every layer.
    paged_pass_pending: bool,
    kv_append: KvAppend,
    kv_quant: KvQuant,
    /// The largest `kv_len` this cache admits, in tokens: the smaller of
    /// `Config::max_position_embeddings` and what the caller asked for.
    ///
    /// **A ceiling, not an allocation.** The slab starts at `kv_slab` and
    /// grows toward this in buckets (`prepare_pass`). It is also what
    /// `FlashScratchSizing::Reserve` reserves against (#234) and what admission
    /// predicts the KV class at (`admit_memory_budget`): both want the largest
    /// value a step of this cache can reach, and that is this.
    ///
    /// Held for every `KvAppend` arm, though `Cat` has no slab: the ceiling
    /// bounds it too, which keeps the arms comparable on this axis.
    kv_capacity: usize,
    /// The grow granularity in tokens — [`DEFAULT_KV_BUCKET`] unless
    /// [`Cache::with_kv_bucket`] said otherwise.
    kv_bucket: usize,
    /// Tokens the slab (or the pool, per sequence) is sized for now. Every
    /// slot, plane and pool is at this width; only `prepare_pass` moves it.
    kv_slab: usize,
    /// Grows this cache has performed. Reported, never acted on.
    kv_grows: usize,
    /// The fraction of `recommendedMaxWorkingSetSize` a grow may take the
    /// process's Metal allocation to (§9.5c's denominator): the configured
    /// budget's fraction, else [`MemoryBudget::DEFAULT_FRACTION`], else
    /// [`Cache::with_kv_grow_fraction`]'s.
    kv_grow_fraction: f64,
    /// Bytes one token of KV costs across every attention layer at this
    /// cache's dtype and B=1 — the prediction a grow is admitted on before the
    /// first append, when no plane has a shape to read.
    kv_bytes_per_token: usize,
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
    /// `conv_phase` as it stood before this pass advanced it — the slot the
    /// previous window ended at. Read by the multi-token conv writer under
    /// `ConvPrefillCtx=on` to find the left context, since the advance that
    /// precedes the layer loop has already reseeded `conv_phase` by then.
    conv_phase_prev: usize,
    /// Positions the ring has written since it was last seeded by a multi-token
    /// pass, saturating at the ring's history depth. A rewind may only reach
    /// slots the ring actually wrote: right after a prefill the history slots
    /// are zeros, so the rewind horizon is `min(K, conv_history)`.
    conv_history: usize,
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
    /// The verify window's anchor, a `[1]` `U32` slot this cache owns and
    /// rewrites per round through the mapped pointer — where a `Tensor::new`
    /// per round was an unpooled driver allocation, a residency add and a
    /// residency commit at the next cut (§10.2z-iii's drains). Written only
    /// between rounds, after the previous window's readback, so no queued
    /// `cat` still reads it (the mapped write's own contract).
    anchor_slot: Option<Tensor>,
    /// Tap sets a forward cleared before anyone took them. A drafter seeded
    /// from a cache that dropped a set has missed those positions — the
    /// defect no gate but the accept rate sees (§10.2n, §10.2w) — so the
    /// speculative drivers refuse a run in which this moved.
    taps_dropped: u64,
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

impl CacheReset for Cache {
    /// Delegates to [`Cache::clear`] — the tested total reset (kvs, kv/quant
    /// slots, conv states, conv phase and compaction flag). This is what lets
    /// `Session::warmup` run a real forward for its compilation side effect and
    /// leave the session pre-prefill, so the first real prompt is unaffected.
    fn reset(&mut self) {
        self.clear();
    }
}

impl Cache {
    /// A cache whose KV grows in buckets toward the model's positional
    /// ceiling: `kv_capacity` is `max_position_embeddings`. See
    /// [`Cache::new_with`] for the ceiling and [`DEFAULT_KV_BUCKET`] for the
    /// bucket.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        Self::new_with(
            use_kv_cache,
            dtype,
            config,
            device,
            config.max_position_embeddings,
        )
    }

    /// As [`Cache::new`], with the KV ceiling given explicitly.
    ///
    /// # The slab grows in buckets, and each bucket is admitted first
    ///
    /// `kv_capacity` is the largest `kv_len` this cache will hold, at most
    /// `max_position_embeddings` — the RoPE tables are built to that, and a
    /// position past them has no rotation. It is a **ceiling, not an
    /// allocation**: the `InPlace` slab, the quantized planes and the paged
    /// pool start at one bucket ([`DEFAULT_KV_BUCKET`] tokens, or
    /// [`Cache::with_kv_bucket`]'s) and widen by a bucket whenever a pass
    /// would overrun them, so a short reply pays for what it uses and a long
    /// one is bounded by memory rather than by a constant. `Cat` has no slab
    /// and grows by `Tensor::cat` as it always did; the ceiling applies to it
    /// too.
    ///
    /// A grow happens in `prepare_pass`, at the top of a forward and **before
    /// any state moves** — before the conv phase advances and before a layer
    /// writes — because a refusal from inside a layer would leave the conv
    /// ring one step ahead of the KV and a retry would diverge. Two things can
    /// refuse there, and both leave the cache exactly as it was: the ceiling,
    /// and memory. Memory is checked against the device itself —
    /// `currentAllocatedSize` plus the bucket, against a fraction of
    /// `recommendedMaxWorkingSetSize` (§9.5c's denominator; the fraction is a
    /// declared budget's or [`MemoryBudget::DEFAULT_FRACTION`]) — so it holds
    /// whether or not a caller declared a [`MemoryBudget`]. This machine has
    /// been kernel-panicked twice by an allocation that found the limit by
    /// crossing it (§9.5a); a grow that overran would be that failure again,
    /// and the check is what makes it an error naming the numbers instead.
    ///
    /// # What the ceiling and the bucket cost, from §5.6's 16 KiB per token
    ///
    /// | `kv_len` | slab held | grows paid |
    /// |---|---|---|
    /// | 267 (a canonical reply) | 64 MiB | 0 |
    /// | 4 096 | 64 MiB | 0 |
    /// | 8 000 | 128 MiB | 1 |
    /// | 32 768 | 512 MiB | 7 |
    /// | 131 072 | 2048 MiB | 31 |
    ///
    /// Every grow copies the live slab into a wider one and re-binds the
    /// buffer (`KvSlot::grow_to` says what that costs a replay executor and
    /// the arena). Below one bucket nothing differs from a fixed 4096-token
    /// cache: the same slab, the same buffer, the same stream.
    ///
    /// # Admission at construction, when a budget is declared
    ///
    /// `admit_memory_budget` predicts the peak at the **ceiling** —
    /// `kv_capacity × 16 KiB` — because §9.5d's admission is a predicted peak
    /// checked once, and the peak of a run allowed to grow is its ceiling. A
    /// caller that declares a budget on a machine that cannot hold the ceiling
    /// is refused at construction and told to lower `kv_capacity`, which is
    /// the decision admission exists to force early (§9.5g). The per-grow
    /// check is the other half: it runs with or without a budget, and it sees
    /// what admission cannot — everything the process allocated since.
    ///
    /// # Why not `crate::nn::kv_cache::KvCache`
    ///
    /// It is the same mechanism — `slice_set` at a moving offset, `narrow` to
    /// read — and it grows by `Tensor::cat` on exhaustion, every time, by
    /// `max_seq_len`: a copy at the moment the cache is largest, with no
    /// admission and no ceiling. The bucketed grow here is that copy made
    /// rare, bounded and admitted, and the local type is what lets it keep
    /// `rewind`, the quantized planes and the paged pool on one contract.
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

        // The ceiling is bounded by the RoPE tables built below: a position
        // past `max_position_embeddings` has no rotation, and asking for a
        // cache wider than that would fail at the first narrow past it —
        // inside a layer, after the conv phase moved. Refused here instead.
        if kv_capacity == 0 || kv_capacity > config.max_position_embeddings {
            lloom_tensor::bail!(
                "lfm2 kv_capacity {kv_capacity} is outside 1..={}: the model's \
                 max_position_embeddings is the largest position the RoPE tables cover",
                config.max_position_embeddings
            )
        }
        let slab = DEFAULT_KV_BUCKET.min(kv_capacity);
        let kv_bytes_per_token = Self::kv_bytes_per_token(config, dtype);
        let kv_grow_fraction = config
            .memory_budget
            .map(|b| b.fraction)
            .unwrap_or(MemoryBudget::DEFAULT_FRACTION);

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
                lloom_tensor::bail!(
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
        if config.kv_append == KvAppend::Paged {
            if config.attn_impl != AttnImpl::FlashDecoding {
                lloom_tensor::bail!(
                    "lfm2 `KvAppend::Paged` reads the pool through the chunk table, which only \
                     the FlashDecoding kernels take: use `--attn flash` (DESIGN.md 10.3b)"
                )
            }
            if config.kv_quant.is_on() {
                lloom_tensor::bail!("lfm2 `KvAppend::Paged` has no quantized arm")
            }
            if !device.is_metal() {
                lloom_tensor::bail!(
                    "lfm2 `KvAppend::Paged` is Metal-only: the pool is read by a Metal kernel"
                )
            }
        }
        #[cfg(feature = "metal")]
        let paged = if config.kv_append == KvAppend::Paged && use_kv_cache {
            let clock = match device {
                Device::Metal(d) => d.gpu_clock(),
                _ => lloom_tensor::bail!("lfm2 `KvAppend::Paged` is Metal-only"),
            };
            Some(PagedKv::new(config, kv_capacity, slab, num_layers, clock)?)
        } else {
            None
        };
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            kv_slots: (0..num_layers)
                .map(|_| (KvSlot::new(slab), KvSlot::new(slab)))
                .collect(),
            quant_slots: (0..num_layers)
                .map(|_| {
                    (
                        QuantKvSlot::new(slab, config.kv_quant),
                        QuantKvSlot::new(slab, config.kv_quant),
                    )
                })
                .collect(),
            #[cfg(feature = "metal")]
            paged,
            paged_pass_pending: false,
            kv_append: config.kv_append,
            kv_quant: config.kv_quant,
            kv_capacity,
            kv_bucket: DEFAULT_KV_BUCKET,
            kv_slab: slab,
            kv_grows: 0,
            kv_grow_fraction,
            kv_bytes_per_token,
            conv_states: vec![None; num_layers],
            conv_phase: 0,
            conv_phase_prev: 0,
            conv_history: 0,
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
            anchor_slot: None,
            taps_dropped: 0,
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

    /// How many tap sets a forward has cleared untaken (see `taps_dropped`).
    pub fn taps_dropped(&self) -> u64 {
        self.taps_dropped
    }

    /// The verify anchor as a `[1]` device tensor: the cache's own slot,
    /// rewritten in place through the mapped pointer, or — where no mapped
    /// write is available (a non-Metal device) — a fresh upload.
    fn anchor_tensor(&mut self, last: u32) -> Result<Tensor> {
        // Created by an upload that carries the first value, not `zeros`: a
        // pooled `zeros` is a fill blit still queued when the host would write
        // through the mapping, and the fill lands last (the first verify's
        // anchor read 0 that way).
        if self.anchor_slot.is_none() {
            let slot = Tensor::new(&[last], &self.device)?;
            self.anchor_slot = Some(slot.clone());
            return Ok(slot);
        }
        let slot = self.anchor_slot.as_ref().expect("set above");
        // SAFETY: the slot is read only by this cache's verify window, whose
        // readback completed before the next call rewrites it.
        if unsafe { slot.write_host_slice::<u32>(0, &[last]) } {
            return Ok(slot.clone());
        }
        Tensor::new(&[last], &self.device)
    }

    /// Grow the KV slab in buckets of `bucket` tokens rather than
    /// [`DEFAULT_KV_BUCKET`].
    ///
    /// A knob for a measurement, not an axis: the bucket decides how often the
    /// slab is reallocated and how much it holds past what it uses, never what
    /// a kernel reads — a run at bucket 64 and a run at bucket 4096 emit the
    /// same stream, which `lfm2_kv_growth.rs` asserts. Set it before the first
    /// forward; a slab that is already allocated is refused rather than
    /// resized.
    pub fn with_kv_bucket(mut self, bucket: usize) -> Result<Self> {
        if bucket == 0 {
            lloom_tensor::bail!("lfm2 kv_bucket must be at least one token")
        }
        let allocated = self.kv_slots.iter().any(|(k, _)| k.all_data.is_some())
            || self.quant_slots.iter().any(|(k, _)| k.codes.is_some());
        if allocated {
            lloom_tensor::bail!(
                "lfm2 with_kv_bucket: the slab is already allocated; set the bucket before \
                 the first forward"
            )
        }
        self.kv_bucket = bucket;
        self.set_slab(bucket.min(self.kv_capacity));
        Ok(self)
    }

    /// The fraction of `recommendedMaxWorkingSetSize` the process's Metal
    /// allocation may reach through a KV grow (`DESIGN.md` §9.5c's
    /// denominator).
    ///
    /// Defaults to a declared budget's fraction, else
    /// [`MemoryBudget::DEFAULT_FRACTION`]; a caller with no full budget to
    /// declare — none of the shipping harnesses has one — can still bound the
    /// one class that grows. Ignored off Metal, where no denominator exists.
    pub fn with_kv_grow_fraction(mut self, fraction: f64) -> Result<Self> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            lloom_tensor::bail!("lfm2 kv_grow_fraction must be in (0, 1], got {fraction}")
        }
        self.kv_grow_fraction = fraction;
        Ok(self)
    }

    /// Positions the cache holds: the `kv_len` the next pass appends at.
    ///
    /// Read from the first plane that has one, which is the first attention
    /// layer's — 22 of LFM2's 30 layers are conv and hold nothing here (§5.3).
    /// Zero before the first pass and after `clear`.
    pub fn kv_len(&self) -> usize {
        if !self.use_kv_cache {
            return 0;
        }
        if self.kv_quant.is_on() {
            return self
                .quant_slots
                .iter()
                .find(|(k, _)| k.codes.is_some())
                .map(|(k, _)| k.len)
                .unwrap_or(0);
        }
        match self.kv_append {
            KvAppend::Cat => self
                .kvs
                .iter()
                .flatten()
                .next()
                .and_then(|(k, _)| k.dim(2).ok())
                .unwrap_or(0),
            KvAppend::InPlace => self
                .kv_slots
                .iter()
                .find(|(k, _)| k.all_data.is_some())
                .map(|(k, _)| k.len)
                .unwrap_or(0),
            KvAppend::Paged => {
                #[cfg(feature = "metal")]
                {
                    self.paged.as_ref().map(|p| p.len).unwrap_or(0)
                }
                #[cfg(not(feature = "metal"))]
                {
                    0
                }
            }
        }
    }

    /// The KV slab's state, for a run's memory report.
    pub fn kv_report(&self) -> KvReport {
        let bytes = if self.kv_quant.is_on() {
            self.quant_slots
                .iter()
                .map(|(k, v)| k.bytes() + v.bytes())
                .sum()
        } else {
            match self.kv_append {
                KvAppend::Cat => self
                    .kvs
                    .iter()
                    .flatten()
                    .map(|(k, v)| {
                        k.elem_count() * k.dtype().size_in_bytes()
                            + v.elem_count() * v.dtype().size_in_bytes()
                    })
                    .sum(),
                KvAppend::InPlace => self
                    .kv_slots
                    .iter()
                    .map(|(k, v)| k.bytes() + v.bytes())
                    .sum(),
                KvAppend::Paged => {
                    #[cfg(feature = "metal")]
                    {
                        self.paged.as_ref().map(|p| p.bytes()).unwrap_or(0)
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        0
                    }
                }
            }
        };
        KvReport {
            kv_len: self.kv_len(),
            slab: self.kv_slab,
            capacity: self.kv_capacity,
            bucket: self.kv_bucket,
            grows: self.kv_grows,
            bytes,
        }
    }

    /// Make room for a pass of `seq_len` positions, or refuse before anything
    /// moves.
    ///
    /// Called at the top of every entry point of `Model` — `forward`,
    /// `forward_embeds`, `forward_all`, `forward_rows` per row — **before**
    /// `advance_conv_phase`. That placement is the contract: a refusal here
    /// leaves the KV length, the conv phase and every buffer exactly as they
    /// were, so the same call can be retried (after memory returns, or with a
    /// wider ceiling) and produce the stream an unrefused run would have.
    /// Refusing from inside a layer would not: the conv ring is advanced once
    /// per pass before the layers run, and a pass that advanced it and then
    /// failed at layer 2's append leaves the ring one step ahead of the KV.
    ///
    /// Two refusals, in order. The ceiling — `kv_len + seq_len` past
    /// `kv_capacity` — is a property of the model and the caller's choice and
    /// names both. Memory is the device's: `currentAllocatedSize` plus the
    /// bytes the bucket costs, against `kv_grow_fraction ×
    /// recommendedMaxWorkingSetSize`. Both are `Err`, never a panic and never
    /// a silent truncation, which is `DESIGN.md` §3.5's loud stop.
    ///
    /// When neither refuses and the slab is too narrow, every plane is widened
    /// to the next bucket boundary (at most the ceiling) and the grow is
    /// counted. A grow that fails part-way — the pool declining a buffer at
    /// layer 5 — leaves the planes it widened widened and the rest as they
    /// were; every plane is self-describing, nothing was written, and the next
    /// call widens the remainder. Idempotent by construction.
    pub(crate) fn prepare_pass(&mut self, seq_len: usize) -> Result<()> {
        if let Some(plan) = self.plan_pass(seq_len)? {
            self.admit_grow(plan.bytes, &plan.describe())?;
            self.apply_grow(&plan)?;
        }
        Ok(())
    }

    /// The first half of `prepare_pass`: the ceiling, and the grow a pass of
    /// `seq_len` positions would need — sized, not performed. No side effects,
    /// so a caller with several caches (`forward_rows`) can refuse any of them
    /// before growing any, and admit their grows as one sum.
    pub(crate) fn plan_pass(&self, seq_len: usize) -> Result<Option<GrowPlan>> {
        if !self.use_kv_cache || seq_len == 0 {
            return Ok(None);
        }
        let kv_len = self.kv_len();
        let need = kv_len + seq_len;
        if need > self.kv_capacity {
            lloom_tensor::bail!(
                "lfm2 KV ceiling: {kv_len} + {seq_len} = {need} positions > kv_capacity {} \
                 (the smaller of max_position_embeddings {} and the configured ceiling). \
                 The reply stops here and nothing was written for this pass; raise the \
                 ceiling (`--kv-capacity`, at most the model's positional ceiling) or bound \
                 the run.",
                self.kv_capacity,
                self.cos.dim(0)?
            )
        }
        if self.kv_append == KvAppend::Cat && !self.kv_quant.is_on() {
            // No slab: `Tensor::cat` widens by one token per step and the
            // ceiling above is the only bound this arm has.
            return Ok(None);
        }
        if need <= self.kv_slab {
            return Ok(None);
        }
        let slab = (need.div_ceil(self.kv_bucket) * self.kv_bucket).min(self.kv_capacity);
        Ok(Some(GrowPlan {
            from: self.kv_slab,
            slab,
            bytes: self.grow_bytes(slab),
            kv_len,
            seq_len,
        }))
    }

    /// The second half: widen every plane to the plan's slab. Admission is the
    /// caller's — `prepare_pass` admits one plan, `forward_rows` the sum of its
    /// rows' — so this allocates and copies and nothing else.
    pub(crate) fn apply_grow(&mut self, plan: &GrowPlan) -> Result<()> {
        let slab = plan.slab;
        if self.kv_quant.is_on() {
            for (k, v) in self.quant_slots.iter_mut() {
                k.grow_to(slab)?;
                v.grow_to(slab)?;
            }
        } else {
            match self.kv_append {
                KvAppend::InPlace => {
                    for (k, v) in self.kv_slots.iter_mut() {
                        k.grow_to(slab)?;
                        v.grow_to(slab)?;
                    }
                }
                KvAppend::Paged =>
                {
                    #[cfg(feature = "metal")]
                    if let Some(p) = self.paged.as_mut() {
                        p.grow_to(slab)?;
                    }
                }
                KvAppend::Cat => {}
            }
        }
        self.kv_slab = slab;
        self.kv_grows += 1;
        Ok(())
    }

    /// Set every unallocated plane's width and the slab figure to `slab`. A
    /// paged pool that already exists keeps its width, and the figure follows
    /// it.
    fn set_slab(&mut self, slab: usize) {
        for (k, v) in self.kv_slots.iter_mut() {
            k.capacity = slab;
            v.capacity = slab;
        }
        for (k, v) in self.quant_slots.iter_mut() {
            k.capacity = slab;
            v.capacity = slab;
        }
        let mut slab = slab;
        #[cfg(feature = "metal")]
        if let Some(p) = self.paged.as_mut() {
            if p.pool.is_none() {
                p.slab = slab;
            } else {
                slab = p.slab;
            }
        }
        self.kv_slab = slab;
    }

    /// Bytes widening the slab to `slab` tokens will allocate, read from an
    /// allocated plane's shape when one exists (its batch is the truth) and
    /// predicted from the geometry at B=1 before the first append.
    fn grow_bytes(&self, slab: usize) -> usize {
        let extra = slab.saturating_sub(self.kv_slab);
        let elem = |t: &Tensor| t.dtype().size_in_bytes();
        let per_token: usize = if self.kv_quant.is_on() {
            self.quant_slots
                .iter()
                .flat_map(|(k, v)| [k, v])
                .filter(|s| s.codes.is_some())
                .map(|s| {
                    let rows = s.b_sz * s.n_heads;
                    rows * s.quant.code_bytes(s.head_dim).unwrap_or(0)
                        + 2 * rows * DType::F16.size_in_bytes()
                })
                .sum()
        } else {
            match self.kv_append {
                KvAppend::InPlace => self
                    .kv_slots
                    .iter()
                    .flat_map(|(k, v)| [k, v])
                    .filter_map(|s| s.all_data.as_ref())
                    .map(|t| {
                        let d = t.dims();
                        d[0] * d[1] * d[3] * elem(t)
                    })
                    .sum(),
                KvAppend::Paged => {
                    #[cfg(feature = "metal")]
                    {
                        // A pool grows by whole pages for every sequence, so the
                        // token count is the pool's, not the slab's.
                        if let Some(p) = self.paged.as_ref() {
                            let tokens = p
                                .pool_tokens_at(slab)
                                .saturating_sub(p.pool_tokens_at(self.kv_slab));
                            let per_token: usize = p
                                .pools
                                .iter()
                                .flatten()
                                .map(|(k, v)| {
                                    let (dk, dv) = (k.dims(), v.dims());
                                    dk[0] * dk[1] * dk[3] * elem(k)
                                        + dv[0] * dv[1] * dv[3] * elem(v)
                                })
                                .sum();
                            if per_token > 0 {
                                return per_token * tokens;
                            }
                            return self.kv_bytes_per_token * tokens;
                        }
                        0
                    }
                    #[cfg(not(feature = "metal"))]
                    {
                        0
                    }
                }
                KvAppend::Cat => 0,
            }
        };
        if per_token == 0 {
            self.kv_bytes_per_token * extra
        } else {
            per_token * extra
        }
    }

    /// One token's KV across every attention layer at B=1, from the geometry
    /// (§5.6): `2 × attention layers × kv_heads × head_dim × dtype`, or the
    /// code bytes plus a scale and a bias per head vector on a quantized arm.
    fn kv_bytes_per_token(config: &Config, dtype: DType) -> usize {
        let attention_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        let head_dim = config.head_dim();
        let per_vector = match config.kv_quant.code_bytes(head_dim) {
            Some(codes) => codes + 2 * DType::F16.size_in_bytes(),
            None => head_dim * dtype.size_in_bytes(),
        };
        2 * attention_layers * config.num_key_value_heads * per_vector
    }

    /// The memory half of `prepare_pass`'s refusal, against the device.
    ///
    /// `currentAllocatedSize` is the whole process's Metal allocation — the
    /// weights, the pool's retained free buffers, every intermediate in flight
    /// — which is exactly the figure admission's prediction cannot see and the
    /// one a grow has to fit beside. The comparison is [`kv_grow_fits`], kept
    /// pure so the arithmetic is tested without a device.
    #[cfg(feature = "metal")]
    pub(crate) fn admit_grow(&self, grow: usize, what: &str) -> Result<()> {
        let Ok(metal) = self.device.as_metal_device() else {
            return Ok(());
        };
        let dev = metal.device();
        let allocated = dev.current_allocated_size();
        let working_set = dev.recommended_max_working_set_size();
        if kv_grow_fits(allocated, working_set, self.kv_grow_fraction, grow) {
            return Ok(());
        }
        let gb = |b: usize| b as f64 / 1e9;
        lloom_tensor::bail!(
            "lfm2 KV grow refused {what}: it needs {:.3} GB more, and the process already \
             holds {:.2} GB of the {:.2} GB it may ({:.2} x recommendedMaxWorkingSetSize \
             {:.2} GB). The reply stops here, before anything is written, rather than risk \
             the machine (DESIGN.md 9.5, 3.5): raise the fraction (`--kv-grow-fraction`), \
             free memory, or bound the run.",
            gb(grow),
            gb(allocated),
            gb(kv_grow_budget(working_set, self.kv_grow_fraction)),
            self.kv_grow_fraction,
            gb(working_set),
        )
    }

    /// No Metal, no `recommendedMaxWorkingSetSize`, nothing to check against —
    /// `admit_memory_budget`'s stance (§9.5c).
    #[cfg(not(feature = "metal"))]
    pub(crate) fn admit_grow(&self, _grow: usize, _what: &str) -> Result<()> {
        Ok(())
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
        use lloom_tensor::metal_backend::admission;

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
            lloom_tensor::bail!(
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
        #[cfg(feature = "metal")]
        if let Some(p) = &mut self.paged {
            p.release_all();
        }
        // A new sequence starts at one bucket again: the slab was dropped with
        // the buffers, and the next one is sized by what the next sequence
        // needs. The paged pool is not dropped and keeps its width.
        self.set_slab(self.kv_bucket.min(self.kv_capacity));
        self.paged_pass_pending = false;
        self.conv_states.iter_mut().for_each(|v| *v = None);
        self.conv_phase = 0;
        self.conv_phase_prev = 0;
        self.conv_history = 0;
        self.conv_compact = false;
    }

    /// The page pool's live tokens, pages in use and pages held in quarantine,
    /// on the `Paged` arm. `None` on the other arms.
    #[cfg(feature = "metal")]
    pub fn paged_stats(&self) -> Option<(usize, usize, usize)> {
        self.paged.as_ref().map(|p| {
            (
                p.len,
                p.pool.as_ref().map(|pool| pool.pages_in_use()).unwrap_or(0),
                p.quarantine.held(),
            )
        })
    }

    /// How many positions [`Cache::rewind`] can currently discard.
    ///
    /// Non-zero only on the rewindable configuration — `KvAppend::InPlace`
    /// (a length to decrement) with `ConvState::RotatingRing { k }` (history
    /// slots a phase move can return to) — and bounded by both the ring's `k`
    /// and the positions the ring has written since its last seed: a multi-token
    /// pass rebuilds the state with zeroed history (`lfm2_spec_conv_shape.rs`),
    /// so there is nothing behind it to return to. Zero while a speculative
    /// window is open.
    pub fn rewind_horizon(&self) -> usize {
        if self.speculative {
            return 0;
        }
        if self.use_kv_cache && self.kv_append == KvAppend::Cat {
            return 0;
        }
        match self.conv_state {
            ConvState::RotatingRing { k } => k.min(self.conv_history),
            _ => 0,
        }
    }

    /// Discard the last `n` positions: KV by a length decrement, conv by a
    /// phase move — `Cache::resolve`'s arithmetic outside a speculative window.
    ///
    /// This is what makes an EOS overrun a sequence-boundary event rather than
    /// a reason to decode conservatively (issue #677): a loop that learns EOS
    /// late leaves `n` forwards past it in the cache, and the next request
    /// starts from the rewound state. Refused past [`Self::rewind_horizon`],
    /// before anything is mutated, naming what would admit it.
    pub fn rewind(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let horizon = self.rewind_horizon();
        if n > horizon {
            let why = if self.speculative {
                "a speculative window is open".to_string()
            } else if self.use_kv_cache && self.kv_append == KvAppend::Cat {
                "`KvAppend::Cat` has no length to rewind (`DESIGN.md` §10.2a)".to_string()
            } else {
                match self.conv_state {
                    ConvState::RotatingRing { k } => format!(
                        "`--conv-state rotating:{k}` keeps {k} history slots and {} have been \
                         written since the last multi-token pass; use `rotating:{n}` or larger",
                        self.conv_history
                    ),
                    other => format!(
                        "{other:?} cannot rewind — only `RotatingRing` keeps the history a \
                         rewind returns to (`DESIGN.md` §10.2a)"
                    ),
                }
            };
            lloom_tensor::bail!("lfm2 rewind({n}): horizon is {horizon} — {why}")
        }
        for (k_slot, v_slot) in self.kv_slots.iter_mut() {
            if k_slot.all_data.is_none() {
                continue;
            }
            k_slot.rewind(n)?;
            v_slot.rewind(n)?;
        }
        for (k_slot, v_slot) in self.quant_slots.iter_mut() {
            if k_slot.codes.is_none() {
                continue;
            }
            k_slot.rewind(n)?;
            v_slot.rewind(n)?;
        }
        #[cfg(feature = "metal")]
        if let Some(p) = &mut self.paged {
            p.rewind(n)?;
        }
        let width = self.conv_state.width(self.l_cache);
        self.conv_phase = (self.conv_phase + width - n) % width;
        self.conv_history -= n;
        Ok(())
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
            lloom_tensor::bail!(
                "lfm2 advance(0): a speculative window must have at least one position"
            )
        }
        // The rotating ring's history must be wide enough that `k` speculative
        // writes do not overwrite the live window: the buffer is `l_cache + k_ring`
        // wide and a window of `k` consumes `k` slots, leaving `l_cache + k_ring - k`
        // for a window that needs `l_cache`. Checked here rather than at the
        // write, because a partial pass is not rollback-able and this is the
        // last point at which refusing is free.
        if let ConvState::RotatingRing { k: k_ring } = self.conv_state {
            if k > k_ring {
                lloom_tensor::bail!(
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
            lloom_tensor::bail!(
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
            lloom_tensor::bail!(
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
            lloom_tensor::bail!(
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
            lloom_tensor::bail!(
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
        #[cfg(feature = "metal")]
        if let Some(p) = &mut self.paged {
            p.rewind(discard)?;
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
        // One row only: the fused op is a GEMV, and a `[B, 1, k]` input would
        // dispatch it `B` times over the weight where the batched step reads
        // the weight once through the `m = B` GEMM (§13.5).
        && matches!(x.dims(), [1, 1, _] | [1, _])
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
    /// Fuse the gate product onto `w3`'s GEMV store (§12.2e row 3b, #732).
    /// The 30 `bmul_f16` dispatches per token.
    gemv_gate: bool,
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
            gemv_gate: cfg.gemv_gate,
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
            // One row only, for `gemv_residual_applies`'s reason: at `B > 1`
            // the fused GEMV would sweep `w1` once per row.
            && matches!(x.dims(), [1, 1, _] | [1, _])
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
                return crate::nn::ops::gemv_residual(&gated, self.down_proj.weight(), residual);
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
            crate::nn::ops::gemv_silu(x, self.gate_proj.weight())?
        } else {
            crate::nn::ops::silu(&self.gate_proj.forward(x)?)?
        };
        // The product on `w3`'s store, the gate its row-indexed operand: one
        // dispatch for `w3` and the `bmul` (§12.2e row 3b, #732). The same
        // applicability as the residual arm — `w3` has `w1`'s shape, and the
        // gate is exactly `w3`'s output extent, contiguous, as `gemv_silu`
        // and `silu` both store it.
        if self.gemv_gate && gemv_residual_applies(true, x, &self.up_proj, &gate) {
            return crate::nn::ops::gemv_mul(x, self.up_proj.weight(), &gate);
        }
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
    /// How the projections' `transpose` views reach the norms and the append
    /// (`Config::qkv_layout`).
    qkv_layout: QkvLayout,
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
            qkv_layout: cfg.qkv_layout,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    /// Whether this call can take the fused `sdpa_vector` kernel.
    ///
    /// Every condition is one `crate::nn::ops::sdpa` would otherwise `bail!` on
    /// (`candle-nn/src/ops.rs`, `Sdpa::metal_fwd`), so a false here is a
    /// fallback and never a failure. They are checked rather than assumed:
    ///
    /// * **Metal.** `Sdpa::cpu_fwd` bails outright; there is no CPU arm.
    /// * **`seq_len == 1`.** `supports_sdpa_vector` requires it. `seq_len > 1`
    ///   is `sdpa_full_applies`'s.
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

    /// Whether this call can take the tiled `steel_attention` kernel — the
    /// `seq_len > 1` sibling of `sdpa_vector` (`DESIGN.md` §10.2v). One
    /// threadgroup per (query block, head), online softmax over K/V blocks,
    /// GQA-native, K/V at their own dtype, the causal mask applied in-kernel
    /// from the cache offset `kv_len - seq_len`. Verify, prefill and B>1 are
    /// all this shape.
    ///
    /// Every fused arm takes it: the arm names the DECODE kernel, and there is
    /// one multi-query kernel for both to share. A false is a fallback to the
    /// generic path and never a failure — each condition is one `ops::sdpa`
    /// would otherwise `bail!` on:
    ///
    /// * **Metal**, **`seq_len > 1`**, and every query has a key
    ///   (`seq_len <= kv_len`, the shape `Sdpa::metal_fwd` admits).
    /// * **dtype** f16/bf16/f32 and **`head_dim`** in the instantiated set;
    ///   f32 has no `head_dim == 512` instantiation.
    fn sdpa_full_applies(&self, q: &Tensor, seq_len: usize, kv_len: usize) -> bool {
        const HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];
        self.attn_impl != AttnImpl::Generic
            && q.device().is_metal()
            && seq_len > 1
            && seq_len <= kv_len
            && matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32)
            && HEAD_DIMS.contains(&self.head_dim)
            && !(self.head_dim == 512 && q.dtype() == DType::F32)
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
        crate::nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
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
                return crate::nn::ops::gemv_residual(&y, self.o_proj.weight(), residual);
            }
            return self.o_proj.forward(&y)? + residual;
        }
        self.forward(x, index_pos, block_idx, cache)? + residual
    }

    /// [`Self::forward_add`] for one step of `rows.len()` independent
    /// sequences (`BatchedCausalLM`): the projections run once over
    /// `[B, 1, hidden]`, each row attends against its own cache. The residual
    /// arm is decided as `forward_add` decides it, so a one-row batch is that
    /// path exactly.
    fn forward_rows_add(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, Cache>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = self.attend_rows(x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &y, &self.o_proj, residual) {
            return crate::nn::ops::gemv_residual(&y, self.o_proj.weight(), residual);
        }
        self.o_proj.forward(&y)? + residual
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
        let (q, k, v, rotated) = self.project(x, Some((index_pos, &*cache)))?;
        let y = self.attend_projected(&q, &k, v, rotated, index_pos, block_idx, cache)?;
        y.transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_attention_heads * self.head_dim))
    }

    /// `attend` over independent rows: one set of projections for the batch,
    /// then row `r` attends at `rows[r].pos` against `rows[r].cache` through
    /// the same kernels, shapes and strides it would see as a batch of one —
    /// `q`, `k` and `v` are narrowed per row, and a row of a contiguous
    /// `[B, heads, 1, head_dim]` is itself a contiguous `[1, heads, 1, head_dim]`.
    fn attend_rows(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;
        // Rows attend at different positions, so the rotation is per row: the
        // projections are normed here and rotated below, on every arm.
        let (q, k, v, rotated) = self.project(x, None)?;
        debug_assert!(!rotated);
        let mut ys = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter_mut().enumerate() {
            ys.push(self.attend_projected(
                &q.narrow(0, r, 1)?,
                &k.narrow(0, r, 1)?,
                v.narrow(0, r, 1)?,
                rotated,
                row.pos,
                block_idx,
                row.cache,
            )?);
        }
        let y = Tensor::cat(&ys, 0)?;
        y.transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_attention_heads * self.head_dim))
    }

    /// [`Self::forward_add`] over a ragged batch (`BatchedCausalLM::forward_ragged`):
    /// the projections run once over `[1, T, hidden]`, each row attends over
    /// its own run of tokens against its own cache. `T == 1` is one decode
    /// row, and takes the residual arm `forward_add` takes.
    fn forward_ragged_add(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, Cache>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = self.attend_ragged(x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &y, &self.o_proj, residual) {
            return crate::nn::ops::gemv_residual(&y, self.o_proj.weight(), residual);
        }
        self.o_proj.forward(&y)? + residual
    }

    /// `attend` over a ragged batch: one set of projections and one pair of
    /// QK norms over all `T` tokens (the norm is per position, so norming
    /// before the split is the same arithmetic), then row `r` attends over
    /// its `len` tokens at `rows[r].pos` against `rows[r].cache` in the
    /// `[1, heads, len, head_dim]` layout a pass of `len` tokens has. A
    /// one-token row's slice is already that layout (no copy); a longer
    /// row's takes the transpose copy `project` takes for a prefill.
    fn attend_ragged(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_, total, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = self.q_norm.forward(&q.reshape((
            1,
            total,
            self.num_attention_heads,
            self.head_dim,
        ))?)?;
        let k = self.k_norm.forward(&k.reshape((
            1,
            total,
            self.num_key_value_heads,
            self.head_dim,
        ))?)?;
        let v = v.reshape((1, total, self.num_key_value_heads, self.head_dim))?;
        let mut ys = Vec::with_capacity(rows.len());
        let mut off = 0usize;
        for row in rows.iter_mut() {
            let len = row.len;
            let q_r = q.narrow(1, off, len)?.transpose(1, 2)?.contiguous()?;
            let k_r = k.narrow(1, off, len)?.transpose(1, 2)?.contiguous()?;
            let v_r = v.narrow(1, off, len)?.transpose(1, 2)?.contiguous()?;
            let y = self.attend_projected(&q_r, &k_r, v_r, false, row.pos, block_idx, row.cache)?;
            ys.push(y.transpose(1, 2)?.reshape((
                1,
                len,
                self.num_attention_heads * self.head_dim,
            ))?);
            off += len;
        }
        if ys.len() == 1 {
            return Ok(ys.remove(0));
        }
        Tensor::cat(&ys, 1)
    }

    /// The three projections and the per-head QK norms: `x [b, seq, hidden]`
    /// to `(q, k, v, rotated)` in `[b, heads, seq, head_dim]`. `q`/`k` are
    /// normed; they are also rotated (`rotated == true`) on the `Fused` arm
    /// when `rope_at` names the position. `v` is contiguous on the
    /// `Contiguous` arm and the bare `transpose` view otherwise, which the
    /// in-place append writes from where it lies (`Tensor::slice_set`).
    fn project(
        &self,
        x: &Tensor,
        rope_at: Option<(usize, &Cache)>,
    ) -> Result<(Tensor, Tensor, Tensor, bool)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) then transpose to
        // (batch, num_heads, seq, head_dim): a view, contiguous at
        // `seq_len == 1` and strided otherwise (`DESIGN.md` §6.2c).
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        match self.qkv_layout {
            QkvLayout::Contiguous => {
                let v = v.contiguous()?;
                let q = self.q_norm.forward(&q.contiguous()?)?;
                let k = self.k_norm.forward(&k.contiguous()?)?;
                Ok((q, k, v, false))
            }
            QkvLayout::Strided => {
                let q = self.q_norm.forward_view(&q)?;
                let k = self.k_norm.forward_view(&k)?;
                Ok((q, k, v, false))
            }
            QkvLayout::Fused => match rope_at {
                Some((index_pos, cache)) => {
                    let cos = cache.cos.narrow(0, index_pos, seq_len)?;
                    let sin = cache.sin.narrow(0, index_pos, seq_len)?;
                    let q = self.q_norm.forward_rope(&q, &cos, &sin)?;
                    let k = self.k_norm.forward_rope(&k, &cos, &sin)?;
                    Ok((q, k, v, true))
                }
                None => {
                    let q = self.q_norm.forward_view(&q)?;
                    let k = self.k_norm.forward_view(&k)?;
                    Ok((q, k, v, false))
                }
            },
        }
    }

    /// RoPE (unless `rotated`), the KV append and the attention kernel
    /// against one cache: `q` and `k` normed, `v` as `project` left it, all
    /// `[b, heads, seq, head_dim]`; returns `[b, heads, seq, head_dim]`,
    /// un-transposed.
    #[allow(clippy::too_many_arguments)]
    fn attend_projected(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: Tensor,
        rotated: bool,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, _, seq_len, _) = q.dims4()?;

        // Apply rotary embeddings
        let (q, k) = if rotated {
            (q.clone(), k.clone())
        } else {
            (
                self.apply_rotary_emb(q, index_pos, cache)?,
                self.apply_rotary_emb(k, index_pos, cache)?,
            )
        };

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
                let y = crate::nn::ops::sdpa_quant(
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
                return Ok(y);
            }
        }

        // Handle KV cache
        let (k, v) = if !cache.use_kv_cache {
            // The attention kernels below want a dense `v`; a clone on the
            // arms that already made it so.
            (k, v.contiguous()?)
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
            let k = crate::nn::ops::dequantize_kv(
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
            let v = crate::nn::ops::dequantize_kv(
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
                    // `k` comes from the norm or the rotation, a fresh
                    // contiguous tensor, so `append`'s `slice_set` takes its
                    // `copy2d`. `v` is the bare `transpose` view on the
                    // `Strided`/`Fused` arms and `slice_set` writes it from
                    // where it lies (one `copy_nd`); on `Contiguous` it was
                    // copied in `project`.
                    debug_assert!(
                        k.is_contiguous(),
                        "KvAppend::InPlace: k should be contiguous here"
                    );
                    let (k_slot, v_slot) = &mut cache.kv_slots[block_idx];
                    // A turn boundary re-prefills without clearing, so this is
                    // a `seq_len > 1` append onto a non-empty slot. It is the
                    // same call — the offset simply advances by more than one.
                    let k = k_slot.append(&k)?;
                    let v = v_slot.append(&v)?;
                    (k, v)
                }
                KvAppend::Paged => {
                    #[cfg(not(feature = "metal"))]
                    {
                        let _ = block_idx;
                        lloom_tensor::bail!("lfm2 `KvAppend::Paged` is Metal-only")
                    }
                    #[cfg(feature = "metal")]
                    {
                        let k = k.contiguous()?;
                        let v = v.contiguous()?;
                        let device = cache.device.clone();
                        let pending = std::mem::take(&mut cache.paged_pass_pending);
                        let Some(paged) = cache.paged.as_mut() else {
                            lloom_tensor::bail!("lfm2 `KvAppend::Paged` with no page pool")
                        };
                        if pending {
                            paged.begin_pass(b_sz, seq_len, &device)?;
                        }
                        let n_prev = paged.len - paged.pass_len;
                        // The write: this pass's tokens to the slots the table
                        // names, one dispatch for K and one for V.
                        paged.write_layer(block_idx, &k, &v)?;
                        if seq_len == 1 {
                            // Decode reads the pool through the table below;
                            // the tensors handed on are the pool itself.
                            let (pk, pv) = paged.pools[block_idx]
                                .as_ref()
                                .expect("write_layer allocates the layer's pool");
                            (pk.clone(), pv.clone())
                        } else if n_prev == 0 {
                            // A first prefill attends over its own tokens.
                            (k, v)
                        } else {
                            // A turn-boundary prefill (§10.9) attends over the
                            // accumulated cache: gathered from the pool by
                            // physical slot, then this pass's tokens. O(kv_len)
                            // per prefill and never on the decode path.
                            let (prev_k, prev_v) =
                                paged.gather_layer(block_idx, b_sz, n_prev, &device)?;
                            let k = Tensor::cat(&[&prev_k, &k], 2)?.contiguous()?;
                            let v = Tensor::cat(&[&prev_v, &v], 2)?.contiguous()?;
                            (k, v)
                        }
                    }
                }
            }
        };

        // The paged decode read. `KvAppend::Paged` requires
        // `AttnImpl::FlashDecoding` at construction, so this is the only decode
        // path the arm has; a dtype the kernels do not instantiate is an error
        // here rather than a silent fallback to a kernel that cannot read a
        // pool (§2.4).
        #[cfg(feature = "metal")]
        let paged_decode = cache.use_kv_cache && cache.kv_append == KvAppend::Paged && seq_len == 1;
        #[cfg(not(feature = "metal"))]
        let paged_decode = false;

        // The fused kernel is GQA-native — it derives `gqa_factor` from the head
        // counts and indexes `kv_head_idx = head_idx / gqa_factor` in registers
        // — and it accumulates in F32 internally. So on that arm `repeat_kv` and
        // the F32 upcast below are not merely unnecessary, they are the two
        // things being removed (`DESIGN.md` §6.2, §8.1 principle 4). Both
        // therefore have to stay *inside* the arm that needs them.
        let y = if paged_decode {
            #[cfg(not(feature = "metal"))]
            unreachable!();
            #[cfg(feature = "metal")]
            {
                if !matches!(q.dtype(), DType::F16 | DType::F32) {
                    lloom_tensor::bail!(
                        "lfm2 `KvAppend::Paged` decode at {:?}: flash_decoding.metal instantiates \
                         f16 and f32 only",
                        q.dtype()
                    )
                }
                let paged = cache.paged.as_ref().expect("checked above");
                let reserve_chunks = cache
                    .kv_capacity
                    .div_ceil((self.flash_page_size * self.flash_pages_per_chunk).max(1));
                crate::nn::ops::flash_decoding_paged(
                    &q,
                    &k,
                    &v,
                    1f32 / (self.head_dim as f32).sqrt(),
                    1.0,
                    self.flash_page_size,
                    self.flash_pages_per_chunk,
                    self.flash_scratch_sizing,
                    reserve_chunks,
                    paged.table.clone(),
                    paged.table_stride,
                    paged.len,
                )?
            }
        } else if self.flash_decoding_applies(&q, seq_len) {
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
            crate::nn::ops::flash_decoding(
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
            crate::nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1f32 / (self.head_dim as f32).sqrt(),
                1.0,
            )?
        } else if self.sdpa_full_applies(&q, seq_len, k.dim(2)?) {
            // The causal mask is the kernel's own, from the cache offset
            // `kv_len - seq_len`; `Cache::mask` is the generic arm's.
            crate::nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                true,
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
            let att = crate::nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        Ok(y)
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
    /// The same permutations stacked `[width, hidden, width]`, the operand the
    /// fused ring kernel indexes by phase (`shortconv_ring_fused`).
    rotating_table: Option<Tensor>,
    conv_state: ConvState,
    /// Fuse the post-conv residual add onto `out_proj`'s GEMV store (§12.2h).
    /// 22 of the 60 `badd_f16` dispatches per token — the largest of the three.
    gemv_residual: bool,
    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// conv1d dispatch (§12.2 row 1, the largest open dispatch-axis win). The
    /// window is `[b, hidden, l_cache]` and the kernel is the same `[hidden, 1,
    /// l_cache]` weight prefill already convolves with, so this reuses the
    /// validated fused depthwise path rather than a new kernel. It MOVES the
    /// digest — the depthwise kernel accumulates serially ascending in f32 where
    /// `sum_keepdim` reduces with a `simd_sum` tree — so it is off by default and
    /// its correctness is a CPU-backend parity check, not an unmoved digest
    /// (§2.3.5a).
    conv_fused: bool,
    /// Carry the previous window across a multi-token pass as the left context
    /// (`Config::conv_prefill_ctx`, `DESIGN.md` §13.2f defect 1).
    prefill_ctx: bool,
    span: tracing::Span,
}

/// `true` when `LLOOM_SKIP_CONV_C_FUSION` is set, forcing the unfused `*C`
/// path (§12.2 row 1) so the fused kernel and its reference can be A/B'd for the
/// digest gate. Read once — the value is fixed for the process, like the other
/// `LLOOM_*` compile/behaviour toggles.
fn conv_c_fusion_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("LLOOM_SKIP_CONV_C_FUSION").is_ok())
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
        let rotating_table = match &rotating_weights {
            Some(per_phase) => Some(Tensor::stack(per_phase, 0)?.contiguous()?),
            None => None,
        };

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            rotating_weights,
            rotating_table,
            l_cache,
            hidden_size,
            conv_state: cfg.conv_state,
            gemv_residual: cfg.gemv_residual,
            conv_fused: cfg.conv_fused,
            prefill_ctx: cfg.conv_prefill_ctx,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    /// The newest `l_cache - 1` columns of the live window, oldest first — the
    /// left context a multi-token pass owes (`DESIGN.md` §13.2f). `prev_phase`
    /// is the slot the window ended at before this pass advanced it; the
    /// shuffle's window is the whole state and needs no phase.
    fn window_tail(&self, state: &Tensor, prev_phase: usize) -> Result<Tensor> {
        let n = self.l_cache - 1;
        let tail = match self.conv_state {
            ConvState::Shuffle => state.narrow(2, 1, n)?,
            ConvState::SlidingRing { .. } => state.narrow(2, prev_phase + 2 - self.l_cache, n)?,
            ConvState::RotatingRing { .. } => {
                let width = self.conv_state.width(self.l_cache);
                let mut cols = Vec::with_capacity(n);
                for back in (0..n).rev() {
                    let slot = (prev_phase + width - back) % width;
                    cols.push(state.narrow(2, slot, 1)?);
                }
                Tensor::cat(&cols, 2)?
            }
        };
        tail.contiguous()
    }

    /// Fold a conv window `[b, hidden, w]` against a `[hidden, w]` weight into
    /// the per-channel output `[b, hidden, 1]` — the operation every decode arm
    /// ends on.
    ///
    /// Unfused (default): `broadcast_mul + sum_keepdim`, whose `simd_sum` tree
    /// is the accumulation order every recorded digest belongs to. Fused
    /// (`--conv-fused`): one depthwise `conv1d` dispatch over the same window,
    /// reusing prefill's validated kernel. `weight` is the `[hidden, w]` slice
    /// (squeezed or phase-permuted); the fused arm `unsqueeze(1)`s it back to the
    /// `[hidden, 1, w]` conv-kernel layout, so a caller passes the SAME weight to
    /// both arms and the choice is this method's alone.
    fn conv_fold(&self, window: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if self.conv_fused {
            let w = window.dims3()?.2;
            // Depthwise: groups == in-channels, one tap window, no padding. The
            // fused Metal kernel (`conv1d_depthwise_k*`) is dispatched by
            // `Tensor::conv1d` when `groups == c_in && c_in_k == 1`, which this
            // is; on a backend without it the generic grouped path gives the
            // identical answer (§conv1d_depthwise fallback).
            let _ = w;
            window
                .conv1d(&weight.unsqueeze(1)?, 0, 1, 1, self.hidden_size)?
                .contiguous()
        } else {
            window
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()
        }
    }

    /// The fold with LFM2's `*C` gate folded into the kernel store (§12.2 row 1).
    ///
    /// Returns `Some(scale * fold(window, weight))` in `[b, hidden, l_out]` when
    /// the fused kernel applies — `conv_fused` on, a scaled `_k<K>` variant
    /// instantiated for this window's tap count, and a per-channel `c`. Returns
    /// `None` otherwise, which is the caller's signal to run `conv_fold` and the
    /// separate `(c * conv_out)` multiply (the arm below RotatingRing folds at
    /// `k = width`, outside the instantiated set, so it takes this path).
    ///
    /// Bit-identical to the two-step form: the accumulation order is `conv_fold`'s
    /// and only the trailing `* c` moves into the store. `c` is passed already
    /// shaped `[b, hidden, l_out]` and made contiguous here so the kernel's flat
    /// `scale[tid]` indexing is valid.
    fn conv_fold_c(&self, window: &Tensor, weight: &Tensor, c: &Tensor) -> Result<Option<Tensor>> {
        if !self.conv_fused {
            return Ok(None);
        }
        // Diagnostic escape hatch: `LLOOM_SKIP_CONV_C_FUSION=1` forces the
        // unfused `conv_fold` + separate `(c * conv_out)`, so the fused and
        // reference paths can be A/B'd for the digest gate in one build.
        if conv_c_fusion_disabled() {
            return Ok(None);
        }
        // The fused kernels are Metal-only; off Metal the scaled `ConvKernel`
        // family does not compile, so gate the whole path and fall back to the
        // unfused `conv_fold` + `(c * conv_out)` that every backend shares.
        #[cfg(not(feature = "metal"))]
        {
            let _ = (window, weight, c);
            return Ok(None);
        }
        #[cfg(feature = "metal")]
        {
            let k = window.dims3()?.2;
            if lloom_tensor::kernels::ConvKernel::conv1d_depthwise_k_scaled(k).is_none() {
                return Ok(None);
            }
            if !window.device().is_metal() {
                return Ok(None);
            }
            // The scaled kernel addresses the source as a DENSE `(b, c, l_in)` — its
            // row base is `(b*c + c_idx) * l_in`, which is only the real element when
            // dim-1's stride is `l_in`. A ring `window` is a `narrow` of a wider
            // buffer (dim-1 stride is `width`, not `l_cache`), so it fails this and
            // takes the unfused path; `Shuffle`'s state is exactly `[b, hidden,
            // l_cache]` and passes. Making it contiguous here would add the copy the
            // fusion exists to remove, so the guard refuses rather than repairs.
            if !window.is_contiguous() {
                return Ok(None);
            }
            // `weight` is `[hidden, k]` (squeezed); the scaled op reads it as the
            // same `[c, k]` the plain `_k` kernel does. `c` must be contiguous in
            // `[b, hidden, l_out]` for the flat `scale[tid]` read.
            let c = c.contiguous()?;
            let out = crate::nn::ops::conv1d_depthwise_scaled(window, weight, &c, k, 0)?;
            Ok(Some(out))
        }
    }

    /// The whole Shuffle decode conv step fused into one kernel (§12.2 row 1):
    /// `bx = b*x`, shift into `state`, fold against `weight`, `*c`. Returns
    /// `Some((conv_out, new_state))` when the fused path applies — `conv_fused`
    /// on, `Shuffle`, metal, a `shortconv_step` variant for `l_cache`, a
    /// contiguous state, and `b`/`x`/`c` sharing one layout so the kernel's
    /// single stride pair addresses all three. `None` otherwise, so the caller
    /// runs the unfused `cat` + `conv_fold` + `* c`, which is the parity oracle.
    ///
    /// Bit-identical: the fold order is `state[1]*w0 + … + bx*w_{k-1}`, exactly
    /// the shifted `conv_fold`'s, and `new_state = [state[1.. ], bx]` is exactly
    /// `cat(state[1..], bx)`.
    #[allow(clippy::type_complexity)]
    fn shortconv_fused(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        state: &Tensor,
        weight: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !self.conv_fused || conv_c_fusion_disabled() {
            return Ok(None);
        }
        // Metal-only, same reason as `conv_fold_c`: the `shortconv_step` family
        // does not compile off Metal. Fall back to the unfused Shuffle arm.
        #[cfg(not(feature = "metal"))]
        {
            let _ = (b, x, c, state, weight);
            return Ok(None);
        }
        #[cfg(feature = "metal")]
        {
            if !state.device().is_metal() {
                return Ok(None);
            }
            let k = state.dims3()?.2;
            if lloom_tensor::kernels::ConvKernel::shortconv_step_k(k).is_none() {
                return Ok(None);
            }
            // The kernel reads `state` as dense `[b, hidden, k]` and writes a dense
            // state back, so the incoming state must be contiguous (it is: prefill
            // seeds it contiguous and this path writes it back contiguous).
            if !state.is_contiguous() {
                return Ok(None);
            }
            // b/x/c must share one layout — they are sibling narrows of the same
            // `in_proj` output, so this holds — because the kernel takes a single
            // `(stride_b, stride_c)` pair for all three.
            if b.stride() != x.stride() || b.stride() != c.stride() {
                return Ok(None);
            }
            let (out, new_state) = crate::nn::ops::shortconv_step(b, x, c, state, weight, k)?;
            Ok(Some((out, new_state)))
        }
    }

    /// The RotatingRing conv block over this pass's `seq_len` positions as
    /// ONE kernel (issue #675, `DESIGN.md` §10.2u): the speculative verify
    /// writer's per-position `narrow + slice_set + fold` (5 dispatches per
    /// position per layer, the `110·k` term of §10.2i) and the K=1 ring decode
    /// step both collapse to it. Bit-identical to the loop it replaces — the
    /// same products in the same order, rounded at the same points — and the
    /// output arrives in `[b, seq, hidden]` with `*C` applied, so the gate
    /// multiply and the transpose copy go too.
    ///
    /// Returns `None` (the caller runs the loop) when the fusion is off, off
    /// Metal, or when the operands are not in the layout the kernel reads.
    fn shortconv_ring_fused(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        state: &Tensor,
        phase_last: usize,
    ) -> Result<Option<Tensor>> {
        if !self.conv_fused || conv_c_fusion_disabled() {
            return Ok(None);
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (b, x, c, state, phase_last);
            Ok(None)
        }
        #[cfg(feature = "metal")]
        {
            let Some(table) = &self.rotating_table else {
                return Ok(None);
            };
            if !state.device().is_metal() || !state.is_contiguous() {
                return Ok(None);
            }
            if b.stride() != x.stride() || b.stride() != c.stride() {
                return Ok(None);
            }
            // `state` is written in place, so the caller's tensor is the
            // updated ring and only the output comes back.
            let out = crate::nn::ops::shortconv_ring(b, x, c, state, table, phase_last)?;
            Ok(Some(out))
        }
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
                return crate::nn::ops::gemv_residual(&conv_out, self.out_proj.weight(), residual);
            }
            return self.out_proj.forward(&conv_out)? + residual;
        }
        self.forward(x, block_idx, cache)? + residual
    }

    /// [`Self::forward_add`] for one step of `rows.len()` independent
    /// sequences (`BatchedCausalLM`): `in_proj` runs once over
    /// `[B, 1, hidden]`, each row convolves against its own state. The
    /// residual arm is decided as `forward_add` decides it.
    fn forward_rows_add(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, Cache>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let conv_out = self.convolve_rows(x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &conv_out, &self.out_proj, residual) {
            return crate::nn::ops::gemv_residual(&conv_out, self.out_proj.weight(), residual);
        }
        self.out_proj.forward(&conv_out)? + residual
    }

    /// Everything up to but excluding `out_proj`, so the two entry points
    /// above share one body and cannot drift in what they compute.
    fn convolve(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Project input to B, C, X components
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        self.convolve_bcx(&bcx, block_idx, cache)
    }

    /// `convolve` over independent rows: one `in_proj` for the batch, then
    /// row `r` takes the whole conv block against `rows[r].cache`. A row of
    /// the projected `[B, 3·hidden, 1]` has the strides a batch of one has, so
    /// every arm below sees the layout it was written for.
    fn convolve_rows(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let mut outs = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter_mut().enumerate() {
            outs.push(self.convolve_bcx(&bcx.narrow(0, r, 1)?, block_idx, row.cache)?);
        }
        Tensor::cat(&outs, 0)
    }

    /// [`Self::forward_add`] over a ragged batch (`BatchedCausalLM::forward_ragged`):
    /// `in_proj` runs once over `[1, T, hidden]`, each row convolves over its
    /// own run against its own state.
    fn forward_ragged_add(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, Cache>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let conv_out = self.convolve_ragged(x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &conv_out, &self.out_proj, residual) {
            return crate::nn::ops::gemv_residual(&conv_out, self.out_proj.weight(), residual);
        }
        self.out_proj.forward(&conv_out)? + residual
    }

    /// `convolve` over a ragged batch: one `in_proj`, then row `r` takes the
    /// whole conv block over its `len` tokens against `rows[r].cache`. A
    /// row's `[1, 3·hidden, len]` is the transposed narrow a pass of `len`
    /// tokens sees (for `len == 1`, the strides a batch of one has — the
    /// fused decode arms' layout), so every arm below sees what it was
    /// written for.
    fn convolve_ragged(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bcx = self.in_proj.forward(x)?;
        let mut outs = Vec::with_capacity(rows.len());
        let mut off = 0usize;
        for row in rows.iter_mut() {
            let bcx_r = bcx.narrow(1, off, row.len)?.transpose(1, 2)?;
            outs.push(self.convolve_bcx(&bcx_r, block_idx, row.cache)?);
            off += row.len;
        }
        if outs.len() == 1 {
            return Ok(outs.remove(0));
        }
        Tensor::cat(&outs, 1)
    }

    /// The conv block after `in_proj`: `bcx` is `[b, 3·hidden, seq]` and the
    /// result `[b, seq, hidden]` with the `*C` gate applied.
    fn convolve_bcx(&self, bcx: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let (b_sz, _, seq_len) = bcx.dims3()?;
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;

        // Prepare conv weight: squeeze to (hidden_size, l_cache) for element-wise, or keep for Conv1d
        let conv_weight = self.conv_weight.squeeze(1)?;

        // The whole-block fused Shuffle decode path (§12.2 row 1), attempted
        // BEFORE `bx` is materialized so its `bmul` and `.contiguous()` are
        // skipped entirely — that is the win over `conv_fold_c`, which still
        // pays them. One kernel computes `bx = b*x`, shifts it into the state,
        // folds, applies `c`, and writes the next state; only the transpose for
        // `out_proj` remains. Fires only for `Shuffle` at `seq_len == 1` off the
        // speculative path; everything else falls through to the code below.
        if matches!(self.conv_state, ConvState::Shuffle)
            && seq_len == 1
            && !(cache.speculative && seq_len > 1)
        {
            let state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => Tensor::zeros(
                    (b_sz, self.hidden_size, self.l_cache),
                    c.dtype(),
                    c.device(),
                )?,
            };
            if let Some((conv_out, new_state)) =
                self.shortconv_fused(&b, &x_proj, &c, &state, &conv_weight)?
            {
                if cache.use_kv_cache {
                    cache.conv_states[block_idx] = Some(new_state);
                }
                // `c` is already applied in the kernel store; only the transpose
                // to `[b, seq, hidden]` for `out_proj` remains.
                return conv_out.transpose(1, 2)?.contiguous();
            }
        }

        // The whole RotatingRing block — a speculative verify's `seq_len`
        // ring writes or the K=1 decode step — as one kernel, attempted before
        // `bx` is materialized for `shortconv_fused`'s reason. Falls through
        // to the per-position loop below when it does not apply.
        if matches!(self.conv_state, ConvState::RotatingRing { .. })
            && ((cache.speculative && seq_len > 1) || seq_len == 1)
        {
            let width = self.conv_state.width(self.l_cache);
            let state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => Tensor::zeros((b_sz, self.hidden_size, width), c.dtype(), c.device())?,
            };
            // `Model::forward`/`forward_all` have advanced `conv_phase` to the
            // slot of this pass's LAST position (§10.2e's one owner).
            if let Some(conv_out) =
                self.shortconv_ring_fused(&b, &x_proj, &c, &state, cache.conv_phase)?
            {
                // The kernel wrote `state` IN PLACE. When the cache had none,
                // `state` is the buffer allocated just above and must be
                // kept; otherwise it is already the cache's own tensor and
                // this stores the same handle back — which is what makes the
                // ring's state stop escaping its pass (§9.2aa, §13).
                if cache.use_kv_cache {
                    cache.conv_states[block_idx] = Some(state);
                }
                return Ok(conv_out);
            }
        }

        // Element-wise multiply B and X
        let bx = (b * &x_proj)?.contiguous()?;

        // Set true by a decode arm that folds the `*C` gate into its conv kernel
        // (`conv_fold_c` / `shortconv_fused`, §12.2 row 1); the trailing
        // `(c * conv_out)` is then skipped. The prefill and speculative arms
        // leave it false and take the separate multiply, which is the unchanged
        // path and the parity oracle.
        let mut c_fused = false;
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
                lloom_tensor::bail!(
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
                // The same fold as the decode arm, per position: `conv_fold`
                // honours `--conv-fused` here too, so a speculative verify runs
                // the same conv kernel a decode of that token would.
                outs.push(self.conv_fold(&state, w)?);
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
                    // The rotating arm folds the WHOLE `width`-wide buffer against
                    // a phase-permuted weight (not an `l_cache` window), so
                    // `conv_fold`'s depthwise kernel runs at `k_size = width`. The
                    // permutation already lives in `w`, so the fused and unfused
                    // arms fold the same products and differ only in order.
                    self.conv_fold(&state, w)?
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
                    self.conv_fold(&window, &conv_weight)?
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
                    // step, not per sequence. (Under `--conv-fused` the same
                    // per-step weight sharing holds: the depthwise kernel reads
                    // one `[hidden, w]` weight for the whole batch.)
                    match self.conv_fold_c(&state, &conv_weight, &c)? {
                        Some(scaled) => {
                            c_fused = true;
                            scaled
                        }
                        None => self.conv_fold(&state, &conv_weight)?,
                    }
                }
            }
        } else {
            // Prefill: use Conv1d.
            //
            // The left context (`DESIGN.md` §13.2f defect 1): a pass onto a
            // warm cache owes the previous window's newest `l_cache - 1` `bx`
            // columns as the convolution's left context in place of the zero
            // padding, or its first `l_cache - 1` outputs convolve over zeros
            // where the previous turn's tokens belong. Behind `ConvPrefillCtx`;
            // off, this is the zero-padded pass every recorded digest belongs to.
            let ctx = match &cache.conv_states[block_idx] {
                Some(prev) if self.prefill_ctx && cache.use_kv_cache && self.l_cache > 1 => {
                    Some(self.window_tail(prev, cache.conv_phase_prev)?)
                }
                _ => None,
            };
            let (conv_in, padding) = match &ctx {
                Some(ctx) => (Tensor::cat(&[ctx, &bx], 2)?.contiguous()?, 0),
                None => (bx.clone(), self.l_cache.saturating_sub(1)),
            };
            let conv = Conv1d::new(
                self.conv_weight.clone(),
                None,
                Conv1dConfig {
                    padding,
                    groups: self.hidden_size,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&conv_in)?;
            out = out.narrow(2, 0, seq_len)?;

            // Update cache with last l_cache tokens
            if cache.use_kv_cache && self.l_cache > 0 {
                let src_len = conv_in.dim(2)?;
                let start = src_len.saturating_sub(self.l_cache);
                let cache_len = src_len - start;
                let mut cache_src = conv_in.narrow(2, start, cache_len)?;
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
                // The state OUTLIVES this pass: the next pass reads it. Under
                // a pass plan (`DESIGN.md` §9.2aa) this pass's activations —
                // `cache_src` among them — live in slots the next pass reuses,
                // so the carry is written INTO the storage the cache already
                // owns (one `copy2d`, ordered after this pass's own read of it
                // by the barrier chain), and a cache with no state yet gets
                // one allocated under an `EscapeGuard`, which no plan serves.
                // This is what makes a prefill tile allocate nothing that
                // escapes it: measured 2 undeclared escapes per tile on the
                // tiny fixture before, 0 after (`lfm2_pass_plans.rs`).
                let carried = cache_src.contiguous()?;
                let owned = match &cache.conv_states[block_idx] {
                    Some(prev)
                        if prev.shape() == carried.shape()
                            && prev.dtype() == carried.dtype()
                            && prev.is_contiguous() =>
                    {
                        prev.slice_set(&carried, 2, 0)?;
                        prev.clone()
                    }
                    _ => {
                        // Not `Tensor::copy`: on Metal `try_clone` shares
                        // the buffer, and the storage must be this cache's own.
                        let _declared = lloom_tensor::EscapeGuard::new();
                        let own =
                            Tensor::zeros(carried.shape(), carried.dtype(), carried.device())?;
                        own.slice_set(&carried, 2, 0)?;
                        own
                    }
                };
                cache.conv_states[block_idx] = Some(owned);
            }

            out
        };

        // Multiply by C, unless a decode arm already folded it into the conv
        // kernel's store (`c_fused`, §12.2 row 1). The transpose to `[b, seq,
        // hidden]` for `out_proj` is unchanged either way; only the elementwise
        // `*C` dispatch and its barrier are removed on the fused arm.
        let conv_out = if c_fused { conv_out } else { (c * &conv_out)? };
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

    /// [`Self::forward`] for one step of `rows.len()` independent sequences:
    /// the norms, the MLP and every projection run once over `[B, 1, hidden]`;
    /// the attention or conv block runs per row against that row's cache.
    fn forward_rows(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match &self.kind {
            LayerKind::Attention(attn) => attn.forward_rows_add(&x, block_idx, rows, residual)?,
            LayerKind::ShortConv(conv) => conv.forward_rows_add(&x, block_idx, rows, residual)?,
        };
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        self.mlp.forward_add(&x, residual)
    }

    /// [`Self::forward`] over a ragged batch: the norms, the MLP and every
    /// projection run once over `[1, T, hidden]`; the attention or conv block
    /// runs per row over that row's tokens against its cache.
    fn forward_ragged(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match &self.kind {
            LayerKind::Attention(attn) => attn.forward_ragged_add(&x, block_idx, rows, residual)?,
            LayerKind::ShortConv(conv) => conv.forward_ragged_add(&x, block_idx, rows, residual)?,
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
    /// What `Fusion=Derived` decided at construction; `None` under `Manual`.
    fusion_report: Option<FusionReport>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
    conv_state: ConvState,
    l_cache: usize,
}

/// Derive the epilogue plan for `cfg` and return the config with the two
/// epilogue flags set from it, with the report (`crate::optimizer`, issue
/// #732 P1/P2).
///
/// The plan is per site; the runtime seam is two model-wide bools. A plan
/// that fuses some sites of a kind and not others cannot be expressed by the
/// bools, so it is refused rather than approximated — that would be a
/// finding about the catalogue or the model, and it is reported as an error
/// naming the sites.
#[cfg(not(target_arch = "wasm32"))]
fn derive_fusion(cfg: &Config, dtype: DType) -> Result<(Config, FusionReport)> {
    use crate::optimizer::{
        derive, exclusions, numeric, ExclusionSet, Lookup, Policy, RegionKind, TargetFpModel,
        Verdict,
    };

    let set = exclusions::embedded();
    let mut report = FusionReport {
        math_mode: numeric::build_math_mode(),
        ..Default::default()
    };

    // The exclusion set speaks about every transform the offline validation
    // ran, not only the fragments the derivation composes. Each arm this
    // config selects is looked up UNDER the fp model of the kernel it runs:
    // an arm the validation refused is refused here, by name and with its
    // numbers; one it never measured, or measured under another kernel's
    // model, is reported as unvalidated — at construction, never per token.
    // (transform name, kernel family, the region it rewrites)
    let mut selected: Vec<(String, &'static str, RegionKind)> = Vec::new();
    #[cfg(feature = "metal")]
    if cfg.silu_impl != lloom_tensor::kernels::SiluImpl::Exact {
        selected.push((
            ExclusionSet::silu_transform(cfg.silu_impl.name()),
            "unary",
            RegionKind::Straight,
        ));
    }
    // The conv arms: the fold's kernel depends on both flags, and the pair is
    // ONE transform of the ring's kernel, not two composed.
    let conv_state_arm = match cfg.conv_state {
        ConvState::Shuffle => None,
        ConvState::RotatingRing { .. } => Some("conv_state=rotating"),
        ConvState::SlidingRing { .. } => Some("conv_state=ring"),
    };
    let conv_fused_family = match cfg.conv_state {
        ConvState::Shuffle => "shortconv_step",
        ConvState::RotatingRing { .. } => "shortconv_ring",
        ConvState::SlidingRing { .. } => "conv1d_depthwise",
    };
    let mut conv_transforms: Vec<String> = Vec::new();
    if cfg.conv_fused {
        conv_transforms.push("conv_fused".to_string());
    }
    if let Some(arm) = conv_state_arm {
        conv_transforms.push(arm.to_string());
    }
    match conv_transforms.as_slice() {
        [] => {}
        [one] => {
            let family = if cfg.conv_fused {
                conv_fused_family
            } else {
                "fast_sum"
            };
            selected.push((one.clone(), family, RegionKind::ConvRing));
        }
        _ => {
            // Both: the separate entries were each measured through the
            // recurrence alone and do not compose there by addition (§5a);
            // what runs is the fused ring kernel, and its own joint entry is
            // what admits it.
            let joint = ExclusionSet::joint_transform(
                &conv_transforms
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            );
            let covered = match TargetFpModel::of_family(conv_fused_family) {
                Some(m) => {
                    matches!(set.lookup(&joint, &m), Lookup::Validated(v) if v.verdict == Verdict::Admit)
                }
                None => false,
            };
            report.compositions.push(Composition {
                region: RegionKind::ConvRing.name().to_string(),
                transforms: conv_transforms.clone(),
                joint: covered.then(|| joint.clone()),
            });
            selected.push((joint, conv_fused_family, RegionKind::ConvRing));
        }
    }
    let mut kv_transforms: Vec<(String, &'static str)> = Vec::new();
    match cfg.attn_impl {
        AttnImpl::Generic => {}
        AttnImpl::Sdpa => kv_transforms.push(("attn=sdpa".to_string(), "sdpa_vector")),
        AttnImpl::FlashDecoding => kv_transforms.push(("attn=flash".to_string(), "flash_decoding")),
    }
    match cfg.kv_quant {
        KvQuant::Off => {}
        other => kv_transforms.push((
            format!("kv_quant={}", other.name().to_ascii_lowercase()),
            "dequantize_kv",
        )),
    }
    if kv_transforms.len() > 1 {
        let names: Vec<&str> = kv_transforms.iter().map(|(n, _)| n.as_str()).collect();
        let joint = ExclusionSet::joint_transform(&names);
        report.compositions.push(Composition {
            region: RegionKind::Kv.name().to_string(),
            transforms: names.iter().map(|n| n.to_string()).collect(),
            joint: set.get(&joint).map(|_| joint.clone()),
        });
    }
    for (name, family) in kv_transforms {
        selected.push((name, family, RegionKind::Kv));
    }
    match cfg.qkv_layout {
        QkvLayout::Contiguous => {}
        QkvLayout::Strided => selected.push((
            "qkv_layout=strided".to_string(),
            "rmsnorm",
            RegionKind::Straight,
        )),
        QkvLayout::Fused => selected.push((
            "qkv_layout=fused".to_string(),
            "rmsnorm_rope",
            RegionKind::Straight,
        )),
    }

    for (transform, family, _region) in &selected {
        let Some(model) = TargetFpModel::of_family(family) else {
            report.unvalidated.push((
                transform.clone(),
                format!("no fp model declared for `{family}`"),
            ));
            continue;
        };
        match set.lookup(transform, &model) {
            Lookup::Validated(v) if v.verdict == Verdict::Exclude => {
                lloom_tensor::bail!(
                    "lfm2 fusion=derived refuses `{transform}`: excluded by the offline validation \
                     ({:.2} f16 ulp, last/first {:.3}, overrides {}, completion {}; under {}); \
                     `design/measurements/issue-732-optimizer-p2.md`",
                    v.max_f16_ulp,
                    v.ratio_last_first,
                    v.overrides,
                    if v.text_identical { "identical" } else { "differs" },
                    v.model
                )
            }
            Lookup::Validated(_) => report
                .validated
                .push((transform.clone(), family.to_string())),
            Lookup::ModelMismatch(v) => report.unvalidated.push((
                transform.clone(),
                format!("recorded under {}, runs `{family}` under {model}", v.model),
            )),
            Lookup::Unvalidated => report
                .unvalidated
                .push((transform.clone(), "no entry".to_string())),
        }
    }

    let graph = crate::models::lfm2_ir::decode_step(cfg, dtype);
    let policy = Policy::default();
    let d = derive(&graph, &policy)?;
    let plan = &d.plan;

    let uniform = |kind: &str, fragment: &str| -> Result<bool> {
        let sites: Vec<&str> = graph
            .nodes()
            .iter()
            .filter(|n| n.op == crate::optimizer::Op::Weight && n.name.ends_with(kind))
            .map(|n| n.name.as_str())
            .collect();
        let fused = sites
            .iter()
            .filter(|s| plan.sites.get(**s).map(String::as_str) == Some(fragment))
            .count();
        if fused == 0 {
            Ok(false)
        } else if fused == sites.len() {
            Ok(true)
        } else {
            lloom_tensor::bail!(
                "lfm2 fusion=derived: the plan fuses `{fragment}` onto {fused} of {} `{kind}` \
                 sites, which the model-wide flag cannot express; plan: {plan}",
                sites.len()
            )
        }
    };
    let gemv_silu = uniform("feed_forward.w1.weight", "silu")?;
    let gemv_residual =
        uniform("feed_forward.w2.weight", "residual")? && uniform("out_proj.weight", "residual")?;
    let gemv_gate = uniform("feed_forward.w3.weight", "mul")?;
    // Every fused site must be one of the three kinds the flags name.
    for (site, fragment) in &plan.sites {
        let known = (fragment == "silu" && site.ends_with("feed_forward.w1.weight"))
            || (fragment == "mul" && site.ends_with("feed_forward.w3.weight"))
            || (fragment == "residual"
                && (site.ends_with("feed_forward.w2.weight") || site.ends_with("out_proj.weight")));
        if !known {
            lloom_tensor::bail!(
                "lfm2 fusion=derived: the plan fuses `{fragment}` onto `{site}`, a site no flag \
                 selects; plan: {plan}"
            )
        }
    }
    for (name, why) in policy.unvalidated() {
        report
            .unvalidated
            .push((ExclusionSet::fragment_transform(&name), why));
    }
    report.plan = plan.to_string();
    report.gemv_silu = gemv_silu;
    report.gemv_residual = gemv_residual;
    report.gemv_gate = gemv_gate;
    report.refused = d.refused.clone();
    report.fused_in_stateful = plan
        .regions
        .fused_in_stateful
        .iter()
        .map(|s| (s.site.clone(), s.fragment.clone(), s.region.to_string()))
        .collect();
    let mut out = cfg.clone();
    out.gemv_silu = gemv_silu;
    out.gemv_residual = gemv_residual;
    out.gemv_gate = gemv_gate;
    Ok((out, report))
}

/// The conv fold's ∞-norm gain per channel over every conv layer's loaded
/// weights (`FusionReport::conv_fold_gain`). Once, at construction, under
/// `Fusion=Derived` only.
fn conv_fold_gain(layers: &[DecoderLayer]) -> Result<Option<ConvFoldGain>> {
    let mut gain = ConvFoldGain {
        layers: 0,
        taps: 0,
        channels: 0,
        max: f32::MIN,
        min: f32::MAX,
        over_one: 0,
    };
    for layer in layers {
        let LayerKind::ShortConv(conv) = &layer.kind else {
            continue;
        };
        let (h, _, k) = conv.conv_weight.dims3()?;
        let w = conv
            .conv_weight
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        gain.layers += 1;
        gain.taps = k;
        for c in 0..h {
            let l: f32 = w[c * k..(c + 1) * k].iter().map(|x| x.abs()).sum();
            gain.channels += 1;
            gain.max = gain.max.max(l);
            gain.min = gain.min.min(l);
            if l > 1.0 {
                gain.over_one += 1;
            }
        }
    }
    Ok((gain.layers > 0).then_some(gain))
}

/// A browser build has no optimizer (`egglog` is host-only); the arm is
/// refused rather than silently manual.
#[cfg(target_arch = "wasm32")]
fn derive_fusion(_cfg: &Config, _dtype: DType) -> Result<(Config, FusionReport)> {
    lloom_tensor::bail!("lfm2 fusion=derived is not available on wasm32; use fusion=manual")
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
        // Under `Fusion=Derived` the optimizer decides the epilogue sites
        // once, here, and the layers below read the same bools they always
        // did: the derivation is construction-tier, the token path is
        // untouched (`CONTRIBUTING.md` §3.3a).
        let derived;
        let mut report = None;
        let cfg = match cfg.fusion {
            FusionPolicy::Manual => cfg,
            FusionPolicy::Derived => {
                let (d, r) = derive_fusion(cfg, vb.dtype())?;
                derived = d;
                report = Some(r);
                &derived
            }
        };

        // Push the SiLU steer arm into the kernel-layer cache (RESTRUCTURING
        // §1.6). The value is a `#[axis]` field on `Config` so a run records it;
        // the atomic in `lloom-tensor::kernels` is its cache, set once here at
        // construction. `Exact` leaves the pre-existing `silu_f16` path
        // byte-for-byte unchanged (§7.1a). Metal-only, like the kernel.
        #[cfg(feature = "metal")]
        lloom_tensor::kernels::set_silu_impl(cfg.silu_impl);

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

        // The closed-form gain §5a names wants the loaded weights, so it is
        // read here, once, and the report is printed complete.
        if let Some(r) = report.as_mut() {
            r.conv_fold_gain = conv_fold_gain(&layers)?;
            eprintln!("{r}");
        }

        Ok(Self {
            embed_tokens,
            layers,
            fusion_report: report,
            embedding_norm,
            lm_head,
            dtype: vb.dtype(),
            conv_state: cfg.conv_state,
            l_cache: cfg.conv_l_cache,
        })
    }

    /// What `Fusion=Derived` decided at construction; `None` under `Manual`.
    pub fn fusion_report(&self) -> Option<&FusionReport> {
        self.fusion_report.as_ref()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        // Room for this pass, or a refusal that leaves everything as it was —
        // BEFORE the conv phase advances below (`Cache::prepare_pass`).
        cache.prepare_pass(seq_len)?;

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

    /// One decode step for `rows.len()` independent sequences
    /// (`BatchedCausalLM`, issue #694): `input_ids` is `[B, 1]`, row `i` is fed
    /// to `rows[i]` at `rows[i].pos`, and the result is `[B, vocab]`.
    ///
    /// Every weight matrix — the projections, the MLP, `lm_head` — is applied
    /// to all `B` rows in one `m = B` GEMM, so the 5.394 GB weight sweep is
    /// paid once for the batch rather than once per token (§13.5). What is
    /// per sequence — RoPE at the row's position, the KV append and the
    /// attention read, the conv window — runs per row against that row's own
    /// cache through the same kernels a `forward` of that token would take, so
    /// a row's result does not depend on which rows sit beside it
    /// (§2.3.3 #7). The rows may have any lengths.
    ///
    /// The step is `seq_len == 1` only; a prompt is prefilled through
    /// [`Self::forward`] per sequence. Refused: a cache that taps (the taps are
    /// per pass and a batched pass would have to split them) and one with an
    /// open speculative window.
    pub fn forward_rows(&self, input_ids: &Tensor, rows: &mut [Row<'_, Cache>]) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        if seq_len != 1 {
            lloom_tensor::bail!(
                "lfm2 forward_rows: one token per row, got seq_len {seq_len}; prefill each \
                 sequence through `forward`"
            )
        }
        if b_sz != rows.len() {
            lloom_tensor::bail!(
                "lfm2 forward_rows: {b_sz} token rows for {} caches",
                rows.len()
            )
        }
        if rows.is_empty() {
            lloom_tensor::bail!("lfm2 forward_rows: no rows")
        }
        let mut plans = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter_mut().enumerate() {
            if !row.cache.tap_layers.is_empty() {
                lloom_tensor::bail!(
                    "lfm2 forward_rows: row {r}'s cache taps layers {:?}; a batched step does \
                     not tap",
                    row.cache.tap_layers
                )
            }
            if row.cache.speculative {
                lloom_tensor::bail!(
                    "lfm2 forward_rows: row {r}'s cache has an open speculative window"
                )
            }
            // Every row's ceiling before ANY row grows or moves: a refusal at
            // row 3 must not leave rows 0..2 widened or with their conv phase
            // advanced (`Cache::prepare_pass`).
            plans.push(row.cache.plan_pass(1)?);
        }
        // One admission for the step's grows together, so a refusal on memory
        // also comes before any row has grown.
        let bytes: usize = plans.iter().flatten().map(|p| p.bytes).sum();
        if bytes > 0 {
            rows[0].cache.admit_grow(
                bytes,
                &format!("for a batched step over {} rows", rows.len()),
            )?;
        }
        for (row, plan) in rows.iter_mut().zip(&plans) {
            if let Some(plan) = plan {
                row.cache.apply_grow(plan)?;
            }
        }
        for row in rows.iter_mut() {
            // Per row, as `forward` does per pass: the ring's write slot and
            // the paged pool's pass marker are the row's own.
            self.advance_conv_phase(1, row.cache);
            Self::begin_taps(row.cache);
        }

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let device = rows[0].cache.device.clone();
        for (block_idx, layer) in self.layers.iter().enumerate() {
            mark_graph_node(&device, block_idx);
            hidden_states = layer.forward_rows(&hidden_states, block_idx, rows)?;
        }

        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., 0, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    /// One forward over a RAGGED batch (`BatchedCausalLM::forward_ragged`):
    /// `input_ids` is `[1, T]`, the rows' token runs end to end; row `i`'s
    /// `rows[i].len` tokens are fed to `rows[i]` at `rows[i].pos`; the result
    /// is `[rows, vocab]`, each row's logits at its last token.
    ///
    /// The weight-bearing ops — every projection, the MLP, `lm_head` — run
    /// once over all `T` tokens, so a prefill piece and the decode rows beside
    /// it pay ONE weight sweep (§13.5; the fused chunked prefill). What is per
    /// sequence runs per row over that row's tokens against its own cache,
    /// through the layouts `forward` (a row of `len > 1`) and `forward_rows`
    /// (a row of `len == 1`) already take — so a row computes what it would
    /// compute alone, and a one-row batch of `len` tokens is `forward`.
    ///
    /// Refused as `forward_rows` refuses: a cache that taps, or one with an
    /// open speculative window; and a `T` that is not the rows' lengths' sum.
    pub fn forward_ragged(
        &self,
        input_ids: &Tensor,
        rows: &mut [RaggedRow<'_, Cache>],
    ) -> Result<Tensor> {
        let (b_sz, total) = input_ids.dims2()?;
        if b_sz != 1 {
            lloom_tensor::bail!("lfm2 forward_ragged: the input is `[1, T]`, got batch {b_sz}")
        }
        if rows.is_empty() {
            lloom_tensor::bail!("lfm2 forward_ragged: no rows")
        }
        let lens: usize = rows.iter().map(|r| r.len).sum();
        if lens != total || rows.iter().any(|r| r.len == 0) {
            lloom_tensor::bail!(
                "lfm2 forward_ragged: {total} tokens for rows of lengths {:?}",
                rows.iter().map(|r| r.len).collect::<Vec<_>>()
            )
        }
        let mut plans = Vec::with_capacity(rows.len());
        for (r, row) in rows.iter_mut().enumerate() {
            if !row.cache.tap_layers.is_empty() {
                lloom_tensor::bail!(
                    "lfm2 forward_ragged: row {r}'s cache taps layers {:?}; a batched pass does \
                     not tap",
                    row.cache.tap_layers
                )
            }
            if row.cache.speculative {
                lloom_tensor::bail!(
                    "lfm2 forward_ragged: row {r}'s cache has an open speculative window"
                )
            }
            plans.push(row.cache.plan_pass(row.len)?);
        }
        let bytes: usize = plans.iter().flatten().map(|p| p.bytes).sum();
        if bytes > 0 {
            rows[0].cache.admit_grow(
                bytes,
                &format!("for a ragged batched pass over {} rows", rows.len()),
            )?;
        }
        for (row, plan) in rows.iter_mut().zip(&plans) {
            if let Some(plan) = plan {
                row.cache.apply_grow(plan)?;
            }
        }
        for row in rows.iter_mut() {
            self.advance_conv_phase(row.len, row.cache);
            Self::begin_taps(row.cache);
        }

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let device = rows[0].cache.device.clone();
        for (block_idx, layer) in self.layers.iter().enumerate() {
            mark_graph_node(&device, block_idx);
            hidden_states = layer.forward_ragged(&hidden_states, block_idx, rows)?;
        }

        // Each row's last position, gathered in row order.
        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let mut last = Vec::with_capacity(rows.len());
        let mut off = 0u32;
        for row in rows.iter() {
            off += row.len as u32;
            last.push(off - 1);
        }
        let idx = Tensor::new(last.as_slice(), &device)?;
        let hidden_states = hidden_states
            .squeeze(0)?
            .index_select(&idx, 0)?
            .contiguous()?;
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
        cache.prepare_pass(seq_len)?;
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
        cache.prepare_pass(seq_len)?;
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

    /// The token embedding, for a DSpark drafter that ships none and shares the
    /// target's (`RESTRUCTURING_DESIGN.md` §3.4; `DSparkSpeculator::new`). A
    /// clone is cheap — the weight is `Arc`-backed.
    pub fn draft_embedding(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// The raw `lm_head`, shared with a DSpark drafter as its proposal head. See
    /// [`Self::draft_lm_head`] for why it must stay the raw projection.
    pub fn lm_head_linear(&self) -> &Linear {
        &self.lm_head
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
        // A set still here was written by the previous forward and taken by
        // nobody: counted, because dropping it silently is how a drafter ends
        // up seeded from part of a prompt.
        if cache.taps.iter().any(Option::is_some) {
            cache.taps_dropped += 1;
        }
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
                // The tap OUTLIVES this pass by construction: the drafter
                // reads it in `observe`, after the round's verify has been
                // closed, and the cross-pass detector cannot see that (it
                // looks one pass ahead, and the next pass is another
                // verify). Declared, so no plan ever serves it and a served
                // verify does not hand the drafter a slot the next verify
                // overwrites (`DESIGN.md` §9.2aa, escapes are declared).
                let _declared = lloom_tensor::EscapeGuard::new();
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
        // The page pool allocates this pass's pages at the first attention
        // layer; this is the once-per-pass point that says a new pass began.
        cache.paged_pass_pending = true;
        cache.conv_phase_prev = cache.conv_phase;
        if let ConvState::RotatingRing { k } = self.conv_state {
            let width = self.conv_state.width(self.l_cache);
            cache.conv_compact = false;
            cache.conv_phase = if cache.speculative && seq_len > 1 {
                // A verify pass writes `seq_len` ring slots, one per position,
                // so it advances the phase by `seq_len` — exactly what
                // `seq_len` decode steps would have done. The layers derive
                // each position's slot from the *pre-pass* phase, so this
                // lands where the last position wrote.
                cache.conv_history = (cache.conv_history + seq_len).min(k);
                (cache.conv_phase + seq_len) % width
            } else if seq_len == 1 {
                cache.conv_history = (cache.conv_history + 1).min(k);
                (cache.conv_phase + 1) % width
            } else {
                // A multi-token pass reseeds: the history slots are zeroed by
                // the writer, so nothing behind the new window can be returned to.
                cache.conv_history = 0;
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
    /// # The window is `k + 1` positions, and the last one is the bonus token
    ///
    /// `last` is the token already committed — the one whose logits produced
    /// `proposed[0]`. The pass runs `[last, proposed[0..k]]`: position `i < k`
    /// predicts what follows `window[i]` and is compared against `proposed[i]`,
    /// and position `k` predicts the token after the last proposal. So **a
    /// fully-accepted window advances `k + 1` tokens**: the `k` verified
    /// proposals plus that bonus. Feeding only `k` positions verifies the last
    /// proposal implicitly and forfeits the bonus whenever the whole block is
    /// right — a ceiling of `k` against the `k + 1` of llama.cpp's
    /// `speculative-simple`, which feeds `[id_last, draft…]` (§10.2s).
    ///
    /// The conv ring must reserve `k + 1` history slots (`Cache::advance`).
    ///
    /// # Returns
    ///
    /// The accepted tokens — between 1 and `k + 1`, never 0, because the
    /// target's own prediction at position 0 is always correct by definition.
    pub fn verify_step(
        &self,
        last: u32,
        proposed: &[u32],
        index_pos: usize,
        cache: &mut Cache,
        eos_ids: &[u32],
    ) -> Result<Vec<u32>> {
        if proposed.is_empty() {
            lloom_tensor::bail!("lfm2 verify_step: no proposed tokens")
        }
        let proposed = Tensor::new(proposed, &cache.device)?;
        self.verify_tokens(last, &proposed, index_pos, cache, eos_ids)
    }

    /// [`Self::verify_step`] over proposals that are already on the device — a
    /// rank-1 `U32` tensor of the `k` ids, as a drafter sampling on the GPU
    /// produces them.
    ///
    /// The window `[last, proposed…]` is assembled on the device and the pass
    /// runs without the host ever holding the proposals: the round's single
    /// readback is the target's argmax with the proposals beside it, one
    /// `to_vec1` at the end. A drafter that read its ids back first would pay
    /// a second wait for the GPU per round, and the draft→verify seam would be
    /// a device→host→device round-trip of ids the host does not use until
    /// this compare.
    ///
    /// `eos_ids` is the run's stop set: the accepted prefix ends at the first
    /// EOS the target emits ([`accept_prefix`]), so the cache never holds a
    /// position past the reply's end.
    pub fn verify_tokens(
        &self,
        last: u32,
        proposed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        eos_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let k = proposed.dims1()?;
        if k == 0 {
            lloom_tensor::bail!("lfm2 verify_tokens: no proposed tokens")
        }
        if proposed.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "lfm2 verify_tokens: proposals must be U32 ids, got {:?}",
                proposed.dtype()
            )
        }
        // Opened before the pass, so the token records the state to return to
        // rather than a state the pass has already moved.
        let tok = cache.advance(k + 1)?;

        // `[last, proposed[0..k]]`: every proposal is fed, and the position
        // after the last one is where the bonus token comes from.
        let anchor = cache.anchor_tensor(last)?;
        let input = Tensor::cat(&[&anchor, proposed], 0)?.reshape((1, k + 1))?;

        // **One pass over the weights for k + 1 positions.** §9.1c: the weight
        // sweep is per step and not per token, so this is where the multiple
        // comes from -- 5.394 GB read once against k + 1 times.
        let logits = self.forward_all(&input, index_pos, cache)?;
        let argmax = logits.i(0)?.argmax(lloom_tensor::D::Minus1)?;

        // The one readback: `[argmax[0..=k], proposed[0..k]]` in a single
        // buffer, so the host waits for the GPU once and compares locally.
        // Staged (`stage_read`): the wait is on the buffer that produced it,
        // not the queue's tail, and the staging copy is unconditional — a
        // `to_vec1` here took the tail wait and, when the pool handed the
        // 80-byte result an unmapped buffer, a second blit and wait behind it.
        let both = Tensor::cat(&[&argmax, proposed], 0)?
            .stage_read()?
            .to_vec1::<u32>()?;
        let (argmax, proposed) = both.split_at(k + 1);
        let accepted = accept_prefix(argmax, proposed, eos_ids);

        // `n` positions of state survive: `last` and the `n - 1` proposals the
        // target agreed with. The remaining `k + 1 - n` were written by the
        // pass and are discarded: KV by a length decrement, conv by a phase
        // move, neither by clearing bytes (§10.2a).
        cache.resolve(tok, accepted.len())?;
        Ok(accepted)
    }
}

/// A verify pass that has been encoded and not yet read back —
/// [`Model::verify_tokens_enqueue`]'s receipt, consumed by
/// [`Model::verify_tokens_complete`].
///
/// Holds the open window's `SpecToken`, so the pair inherits its discipline: an
/// enqueued verify that is never completed leaves the cache mid-speculation,
/// and the receipt is neither `Copy` nor droppable without a warning.
#[must_use = "an enqueued verify must be completed, or the cache is left mid-speculation"]
pub struct PendingVerify {
    tok: SpecToken,
    /// `[argmax[0..=k], proposed[0..k]]` — the round's one readback, staged
    /// behind the command buffer that wrote it, so completing waits for this
    /// pass and not for whatever was committed after it.
    both: lloom_tensor::StagedRead,
    k: usize,
    /// This pass's layer-input taps, every window position; trimmed to the
    /// accepted rows at completion.
    taps: Vec<Tensor>,
}

impl Model {
    /// The encode half of [`Self::verify_tokens`]: open the window, run the
    /// pass over `[last, proposed…]`, and leave `[argmax, proposed]` on the
    /// device behind a marker. Nothing here waits for the GPU.
    pub fn verify_tokens_enqueue(
        &self,
        last: u32,
        proposed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<PendingVerify> {
        let k = proposed.dims1()?;
        if k == 0 {
            lloom_tensor::bail!("lfm2 verify_tokens_enqueue: no proposed tokens")
        }
        if proposed.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "lfm2 verify_tokens_enqueue: proposals must be U32 ids, got {:?}",
                proposed.dtype()
            )
        }
        let tok = cache.advance(k + 1)?;
        let anchor = cache.anchor_tensor(last)?;
        let input = Tensor::cat(&[&anchor, proposed], 0)?.reshape((1, k + 1))?;
        let logits = self.forward_all(&input, index_pos, cache)?;
        let argmax = logits.i(0)?.argmax(lloom_tensor::D::Minus1)?;
        let both = Tensor::cat(&[&argmax, proposed], 0)?.stage_read()?;
        // The taps are this pass's; taken now so the cache carries none into
        // the next pass on it.
        let taps = cache.take_taps();
        Ok(PendingVerify { tok, both, k, taps })
    }

    /// The readback half of [`Self::verify_tokens`]: wait for the pass's own
    /// command buffer, accept, resolve the cache, and trim the taps to the
    /// accepted rows. Same accept rule, same `resolve`, same bytes as the
    /// one-call form; only where the host waits differs.
    pub fn verify_tokens_complete(
        &self,
        pending: PendingVerify,
        cache: &mut Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        let PendingVerify { tok, both, k, taps } = pending;
        let both = both.to_vec1::<u32>()?;
        let (argmax, proposed) = both.split_at(k + 1);
        let accepted = accept_prefix(argmax, proposed, eos_ids);
        cache.resolve(tok, accepted.len())?;
        let n_acc = accepted.len();
        let probe = taps
            .iter()
            .map(|t| {
                let seq = t.dim(1)?;
                t.narrow(1, 0, n_acc.min(seq))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((accepted, probe))
    }
}

/// The greedy accept rule over a verified window: the longest prefix where the
/// target's argmax agrees with the proposal, plus the target's own token at the
/// first disagreement — or, when every proposal agreed, the bonus at position
/// `k`.
///
/// `argmax` has `k + 1` entries (position `i` is what the target emits after
/// `window[i]`) and `proposed` has `k`. Returns between 1 and `k + 1` tokens,
/// never 0: position 0 is the target's own prediction and is always kept. This
/// is the whole of what makes speculation output-identical to greedy — every
/// returned token is a target argmax — and a rejection costs nothing, since
/// the target's token at the mismatch is what plain decoding would have
/// emitted there.
///
/// The prefix also ends at the first token in `eos_ids`, which is kept: a reply
/// has no positions after its EOS, and the accepted length is what the cache
/// keeps, what the drafter is keyed with and how far the cursor moves — so a
/// prefix running past an EOS the drafter happened to propose would feed all
/// three with tokens the sequence does not have (§10.2z-ii). An empty set draws
/// no boundary.
pub fn accept_prefix(argmax: &[u32], proposed: &[u32], eos_ids: &[u32]) -> Vec<u32> {
    let k = proposed.len();
    debug_assert_eq!(argmax.len(), k + 1, "argmax covers the k + 1 window");
    let mut accepted = Vec::with_capacity(k + 1);
    for i in 0..=k {
        let target = argmax[i];
        accepted.push(target);
        if eos_ids.contains(&target) || (i < k && target != proposed[i]) {
            break;
        }
    }
    accepted
}

// The engine's model traits over LFM2's existing entry points
// (`RESTRUCTURING_DESIGN.md` §3.1). No computation is threaded through them that
// the inherent methods above do not already do; each wires to one of them.
//
// `forward` is overridden rather than taking `CausalLM`'s provided
// `embed`+`forward_embeds` default: the inherent `forward` is on the per-token
// path and its shape is what every recorded digest belongs to (`DESIGN.md`
// §2.3.8c), so it stays byte-identical rather than routing through two calls.
impl ModelTrait for Model {
    type Config = Config;

    fn from_weights(vb: VarBuilder, cfg: &Self::Config) -> Result<Self> {
        Model::new(cfg, vb)
    }
}

impl CausalLM for Model {
    type Cache = Cache;

    fn new_cache(&self, cfg: &Self::Config, dev: &Device) -> Result<Self::Cache> {
        Cache::new(true, self.dtype, cfg, dev)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Model::embed(self, tokens)
    }

    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        Model::forward_embeds(self, embeds, pos, cache)
    }

    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        cache.prepare_pass(seq_len)
    }

    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        Model::forward(self, tokens, pos, cache)
    }

    fn vocab_size(&self) -> usize {
        // `lm_head` is `[vocab, hidden]` (`Model::new`: `linear(hidden, vocab)`
        // or the tied `[vocab, hidden]` embedding), so its leading dim is the
        // vocabulary. 2D by construction, so `dims()[0]` cannot be absent.
        self.lm_head.weight().dims()[0]
    }
}

impl BatchedCausalLM for Model {
    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Self::Cache>]) -> Result<Tensor> {
        Model::forward_rows(self, tokens, rows)
    }

    fn forward_ragged(
        &self,
        tokens: &Tensor,
        rows: &mut [RaggedRow<'_, Self::Cache>],
    ) -> Result<Tensor> {
        Model::forward_ragged(self, tokens, rows)
    }
}

impl crate::nn::model::CacheRewind for Cache {
    fn rewind_horizon(&self) -> usize {
        Cache::rewind_horizon(self)
    }

    fn rewind(&mut self, n: usize) -> Result<()> {
        Cache::rewind(self, n)
    }
}

impl Speculative for Model {
    type Window = SpecToken;

    fn forward_all(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        Model::forward_all(self, tokens, pos, cache)
    }

    fn advance(cache: &mut Self::Cache, k: usize) -> Result<Self::Window> {
        cache.advance(k)
    }

    fn resolve(cache: &mut Self::Cache, window: Self::Window, n: usize) -> Result<()> {
        cache.resolve(window, n)
    }
}

/// LFM2 satisfies `LayerTaps<5>` — it has the layer-input tap hooks a DSpark
/// drafter reads (`RESTRUCTURING_DESIGN.md` §3.4.3). This impl is the pairing
/// contract: a drafter whose `Probe` is `NextnHidden` (MTP, needing an
/// `h_nextn` head LFM2 does not have) does NOT compile against this target,
/// because LFM2 implements `Probes<LayerTaps<5>>` and NOT `Probes<NextnHidden>`.
///
/// `verify_and_probe` wires to the existing `verify_step` + `Cache::take_taps`,
/// adding no computation. LFM2's cache is one sequence (B=1), so `anchors`/
/// `drafts` carry exactly one entry; a second is a shape the cache cannot hold
/// today and is refused rather than silently ignored.
impl crate::nn::Probes<crate::nn::LayerTaps<5>> for Model {
    fn verify_and_probe(
        &self,
        drafts: &[crate::nn::Draft],
        anchors: &[crate::nn::Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Vec<Tensor>)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "lfm2 verify_and_probe: the LFM2 cache is single-sequence (B=1); \
                 got {} drafts and {} anchors. Batched speculation needs the \
                 batched cache (§10.2r, not built here).",
                drafts.len(),
                anchors.len()
            )
        }
        let anchor = anchors[0];
        let accepted = match &drafts[0].tokens {
            crate::nn::DraftTokens::Host(ids) => {
                self.verify_step(anchor.last, ids, anchor.pos, cache, eos_ids)?
            }
            crate::nn::DraftTokens::Device(ids) => {
                self.verify_tokens(anchor.last, ids, anchor.pos, cache, eos_ids)?
            }
        };

        // The probe: this pass's layer-input taps, drained. Trim to the accepted
        // rows — the rejected positions describe state the target just rolled
        // back (§10.2m). `take_taps` is empty unless `tap_layers` was set, so a
        // caller that did not build the cache with taps gets an empty probe and
        // the mismatch surfaces at the drafter, not here.
        let taps = cache.take_taps();
        let n_acc = accepted.len();
        let probe = taps
            .iter()
            .map(|t| {
                let seq = t.dim(1)?;
                t.narrow(1, 0, n_acc.min(seq))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((vec![accepted], probe))
    }

    type Pending = PendingVerify;

    fn verify_enqueue(
        &self,
        draft: &crate::nn::Draft,
        anchor: &crate::nn::Anchor,
        cache: &mut Self::Cache,
    ) -> Result<PendingVerify> {
        match &draft.tokens {
            crate::nn::DraftTokens::Host(ids) => {
                if ids.is_empty() {
                    lloom_tensor::bail!("lfm2 verify_enqueue: no proposed tokens")
                }
                let ids = Tensor::new(ids.as_slice(), &cache.device)?;
                self.verify_tokens_enqueue(anchor.last, &ids, anchor.pos, cache)
            }
            crate::nn::DraftTokens::Device(ids) => {
                self.verify_tokens_enqueue(anchor.last, ids, anchor.pos, cache)
            }
        }
    }

    fn verify_complete(
        &self,
        pending: PendingVerify,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        self.verify_tokens_complete(pending, cache, eos_ids)
    }

    fn take_probe(&self, cache: &mut Self::Cache) -> Result<Vec<Tensor>> {
        Ok(cache.take_taps())
    }

    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        cache.taps_dropped()
    }
}

/// The `KvSlot` sequence fuzzer's alphabet, model and driver (lloom #575),
/// outside `#[cfg(test)]` under the `fuzzing` feature so `lloom-fuzz` (#706)
/// can drive it coverage-guided. `KvSlot` is private, which is why this is a
/// child module of `lfm2` rather than a file of its own; the proptest
/// generator and the properties stay in `tests` below.
#[cfg(any(test, feature = "fuzzing"))]
pub mod kv_slot_fuzz {
    use super::*;

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
    pub enum SlotOp {
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
        /// slot's own message names the mechanism (an append outside a
        /// prepared pass) where `slice_set`'s names two integers. So the two tests
        /// are **not** redundant and neither subsumes the other — this one
        /// catches a bound that lets the write through, that one catches a
        /// bound that stops naming itself. Recorded here because a doc comment
        /// promising more than its assertions deliver is the sentence a reader
        /// trusts (§3.1b rule 3).
        FillToCapacityThenOne,
        /// Widen the slab by `by` positions: `Cache::prepare_pass`'s grow, at
        /// slot level. The model's capacity moves with it and nothing else
        /// does, which is the whole contract — every live position and every
        /// rewound one stays where it was, and the boundary the fill symbol
        /// above pins moves to the new width.
        Grow { by: usize },
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
    pub struct SlotTransitions {
        pub appends: u64,
        pub speculations: u64,
        pub full_accepts: u64,
        pub full_rejects: u64,
        pub partial_accepts: u64,
        pub rewinds_refused: u64,
        pub capacity_refusals: u64,
        pub resets_from_nonempty: u64,
        /// A rewind that returned the slot to a length it had previously held,
        /// then appended different content there -- the reuse-after-discard
        /// shape, and the one a length-only assertion cannot see.
        pub rewrote_discarded_positions: u64,
        /// A turn boundary where the next turn's shape differs from the last's.
        /// The transition that makes `reset` dropping the buffer observable.
        pub turns_changed_shape: u64,
        /// A sequence that landed EXACTLY on capacity. The transition that
        /// makes the `>` in the bound check observable.
        pub filled_to_capacity: u64,
        /// A grow of a slab that held positions: the copy, not the number.
        pub grows: u64,
    }

    /// Drive one sequence against the real slot and the model, comparing the
    /// full live contents after **every** operation.
    ///
    /// Contents and not just the length: a rewind is a length decrement, so a
    /// defect that decrements correctly and leaves the wrong bytes reachable is
    /// invisible to a length check. That is §3.1b's *shape invariant* trap --
    /// `[4,4,4]` is as flat as `[3,3,3]` -- and the reason this compares an
    /// absolute reference.
    pub fn run_slot_sequence(
        mut capacity: usize,
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
                SlotOp::Grow { by } => {
                    let to = capacity + by;
                    if slot.all_data.is_some() {
                        t.grows += 1;
                    }
                    slot.grow_to(to)
                        .map_err(|e| format!("op {i}: grow_to({to}) failed: {e}"))?;
                    if slot.capacity != to {
                        return Err(format!(
                            "op {i}: grow_to({to}) left the capacity at {}",
                            slot.capacity
                        ));
                    }
                    capacity = to;
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
    pub fn multi_token(ids: &[usize], heads: usize, dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(expected_values(ids, heads), (1, heads, ids.len(), 3), dev)
    }

    /// The flattened values a `narrow(2, 0, len)` view must hold.
    ///
    /// Dim order is `[b, h, seq, d]`, so the seq axis is second-innermost and
    /// the values are laid out h-major over seq.
    pub fn expected_values(ids: &[usize], heads: usize) -> Vec<f32> {
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
}

#[cfg(test)]
mod tests {
    use super::kv_slot_fuzz::*;
    use super::*;

    /// The trait impls dispatch to the inherent entry points, checked at the
    /// bound rather than at a forward pass.
    ///
    /// A keystone-trait PR earns its reachability the way §3.1c asks — the
    /// production consumers (`KvCache`, `VisionEncoder`, `candle-hub`, llumos)
    /// arrive in PRs 4–10, so here the callable-through-the-trait property is
    /// what a generic bound proves: this function only compiles if `Model`
    /// satisfies `Model`/`CausalLM`/`Speculative` and the associated types line
    /// up. The end-to-end computation is the digest gate's, not a unit test's.
    #[allow(dead_code)]
    fn assert_bounds<M>()
    where
        M: ModelTrait + CausalLM + Speculative,
        M::Config: Sized,
    {
    }

    #[test]
    fn lfm2_implements_the_model_traits() {
        assert_bounds::<Model>();
        // The associated types resolve to LFM2's own state, not a placeholder:
        // each closure only typechecks if the two sides are the same type.
        let _cache: fn(<Model as CausalLM>::Cache) -> Cache = |c| c;
        let _window: fn(<Model as Speculative>::Window) -> SpecToken = |w| w;
    }

    /// The accept rule, pinned on values: the longest agreeing prefix plus the
    /// target's token at the first disagreement, or the bonus when nothing
    /// disagrees. Position 0 is always kept.
    #[test]
    fn accept_prefix_keeps_the_agreeing_prefix_plus_the_targets_token() {
        let argmax = [10u32, 11, 12, 13];
        let no_eos: &[u32] = &[];
        assert_eq!(
            accept_prefix(&argmax, &[10, 11, 12], no_eos),
            vec![10, 11, 12, 13]
        );
        assert_eq!(accept_prefix(&argmax, &[99, 11, 12], no_eos), vec![10]);
        assert_eq!(accept_prefix(&argmax, &[10, 99, 12], no_eos), vec![10, 11]);
        assert_eq!(
            accept_prefix(&argmax, &[10, 11, 99], no_eos),
            vec![10, 11, 12]
        );
        // A proposal equal to the wrong position's argmax is still wrong.
        assert_eq!(accept_prefix(&argmax, &[11, 12, 13], no_eos), vec![10]);
        assert_eq!(accept_prefix(&[7], &[], no_eos), vec![7]);
    }

    /// An EOS the target emits ends the prefix at itself — kept, and nothing
    /// past it — wherever it falls: an agreed interior position, position 0,
    /// the bonus, or the mismatch position. An EOS that is only PROPOSED (the
    /// target emitted something else) is an ordinary mismatch, and a set that
    /// names no id in the window changes nothing.
    #[test]
    fn accept_prefix_ends_at_the_first_eos_the_target_emits() {
        let eos: &[u32] = &[7, 8];
        // Interior: the drafter proposed the EOS and the target agreed, then
        // both continued past it. Cut at the EOS.
        assert_eq!(
            accept_prefix(&[10, 7, 12, 13], &[10, 7, 12], eos),
            vec![10, 7]
        );
        // Position 0: the target's own first token is the EOS.
        assert_eq!(accept_prefix(&[7, 11, 12, 13], &[7, 11, 12], eos), vec![7]);
        // The bonus token: the whole block agreed and the target then stopped.
        assert_eq!(
            accept_prefix(&[10, 11, 12, 8], &[10, 11, 12], eos),
            vec![10, 11, 12, 8]
        );
        // The mismatch position: the target emits EOS where the drafter
        // proposed 99. One cut, same result as the mismatch alone.
        assert_eq!(
            accept_prefix(&[10, 7, 12, 13], &[10, 99, 12], eos),
            vec![10, 7]
        );
        // A proposed EOS the target rejects is a mismatch, not a boundary.
        assert_eq!(
            accept_prefix(&[10, 11, 12, 13], &[10, 7, 12], eos),
            vec![10, 11]
        );
        // The second id of the set cuts too.
        assert_eq!(
            accept_prefix(&[10, 8, 12, 13], &[10, 8, 12], eos),
            vec![10, 8]
        );
        // No EOS in the window: the plain rule.
        assert_eq!(
            accept_prefix(&[10, 11, 12, 13], &[10, 11, 12], eos),
            vec![10, 11, 12, 13]
        );
        // The empty set draws no boundary — the pre-boundary rule verbatim.
        assert_eq!(
            accept_prefix(&[10, 7, 12, 13], &[10, 7, 12], &[]),
            vec![10, 7, 12, 13]
        );
    }

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
        acc.ok_or_else(|| lloom_tensor::Error::Msg("no steps".into()))
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
    /// `crate::nn::kv_cache::Cache` grows by `Tensor::cat` here; this declines,
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

    /// Growing the slab keeps every position where it was, and the reference
    /// is a slot that was that wide from the start plus the `cat` the slab
    /// replaces.
    #[test]
    fn growing_the_slab_keeps_every_position_where_it_was() -> Result<()> {
        let dev = Device::Cpu;
        let mut grown = KvSlot::new(2);
        let mut wide = KvSlot::new(5);
        for i in 0..2 {
            grown.append(&token(i, &dev)?)?;
            wide.append(&token(i, &dev)?)?;
        }
        let before = grown
            .all_data
            .as_ref()
            .unwrap()
            .narrow(2, 0, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        grown.grow_to(5)?;
        assert_eq!(grown.capacity, 5);
        assert_eq!(grown.len, 2, "a grow moves no position");
        assert_eq!(
            grown.all_data.as_ref().unwrap().dims()[2],
            5,
            "the slab is five wide after the grow"
        );
        let after = grown
            .all_data
            .as_ref()
            .unwrap()
            .narrow(2, 0, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(before, after, "the live prefix reads the same bytes");
        for i in 2..5 {
            let g = grown.append(&token(i, &dev)?)?;
            let w = wide.append(&token(i, &dev)?)?;
            assert_eq!(
                g.flatten_all()?.to_vec1::<f32>()?,
                w.flatten_all()?.to_vec1::<f32>()?,
                "grown and wide-from-the-start slots differ at len {}",
                i + 1
            );
        }
        let view = grown.all_data.as_ref().unwrap().narrow(2, 0, 5)?;
        assert_eq!(
            view.flatten_all()?.to_vec1::<f32>()?,
            cat_reference(5, &dev)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    /// A grow to the current width or below is a no-op: the same buffer, the
    /// same width, never a shrink.
    #[test]
    fn a_grow_never_shrinks_and_never_reallocates_needlessly() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(4);
        slot.append(&token(0, &dev)?)?;
        let id = slot.all_data.as_ref().unwrap().id();
        slot.grow_to(4)?;
        slot.grow_to(2)?;
        assert_eq!(slot.capacity, 4);
        assert_eq!(
            slot.all_data.as_ref().unwrap().id(),
            id,
            "no-op grows keep the buffer"
        );
        slot.grow_to(6)?;
        assert_ne!(
            slot.all_data.as_ref().unwrap().id(),
            id,
            "a real grow re-binds it"
        );
        Ok(())
    }

    /// Before the first append a grow is a number: nothing is allocated, and
    /// the first append allocates at the new width.
    #[test]
    fn a_grow_before_the_first_append_sets_the_width_and_allocates_nothing() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = KvSlot::new(2);
        slot.grow_to(6)?;
        assert!(slot.all_data.is_none());
        assert_eq!(slot.capacity, 6);
        let view = slot.append(&token(0, &dev)?)?;
        assert_eq!(view.dims()[2], 1);
        assert_eq!(slot.all_data.as_ref().unwrap().dims()[2], 6);
        Ok(())
    }

    /// `prepare_pass`'s memory arithmetic: inside the budget admits, exactly
    /// on it admits, one byte past refuses, and the fraction is what moves it.
    #[test]
    fn a_kv_grow_fits_inside_the_fraction_and_not_past_it() {
        let ws = 1_000_000;
        assert_eq!(kv_grow_budget(ws, 0.5), 500_000);
        assert!(
            kv_grow_fits(0, ws, 0.5, 500_000),
            "exactly on the budget is admitted"
        );
        assert!(
            !kv_grow_fits(0, ws, 0.5, 500_001),
            "one byte past is refused"
        );
        assert!(kv_grow_fits(400_000, ws, 0.5, 100_000));
        assert!(!kv_grow_fits(400_001, ws, 0.5, 100_000));
        assert!(
            kv_grow_fits(400_001, ws, 0.6, 100_000),
            "the fraction moves the verdict"
        );
        assert!(
            !kv_grow_fits(usize::MAX, ws, 1.0, 1),
            "saturating, never wrapping"
        );
    }

    use proptest::prelude::*;

    fn slot_op_strategy(capacity: usize) -> impl Strategy<Value = SlotOp> {
        prop_oneof![
            8 => (1usize..=4).prop_map(|n| SlotOp::Append { n }),
            6 => (1usize..=4).prop_flat_map(|k| (Just(k), 0..=k))
                .prop_map(|(k, accept)| SlotOp::Speculate { k, accept }),
            1 => Just(SlotOp::RewindPastStart),
            2 => Just(SlotOp::FillToCapacityThenOne),
            1 => (1usize..=8).prop_map(|by| SlotOp::Grow { by }),
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
        #![proptest_config(ProptestConfig { max_shrink_iters: 4096, ..ProptestConfig::default() })]

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
            total.grows += t.grows;
        }

        println!("FUZZ575 kvslot transitions: {total:?}");
        let named: [(&str, u64); 12] = [
            ("turns_changed_shape", total.turns_changed_shape),
            ("filled_to_capacity", total.filled_to_capacity),
            ("grows", total.grows),
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

    /// `Config::default()` selects the same axis arm every construction path
    /// selects, so a `..Config::default()` update is byte-identical to naming
    /// the axis explicitly.
    ///
    /// The absolute reference (§3.1b): each axis is pinned to `into_config`'s
    /// literal, not to "some default". A mutation flipping one arm here — the
    /// change that silently reroutes every `..`-built test — flips exactly one
    /// assertion, which is the coverage §15.1k's repair owes.
    #[test]
    fn config_default_matches_the_shipped_axis_arms() {
        let c = Config::default();
        assert_eq!(c.attn_impl, AttnImpl::Generic, "AttnImpl default arm");
        assert_eq!(c.kv_append, KvAppend::Cat, "KvAppend default arm");
        assert_eq!(c.conv_state, ConvState::Shuffle, "ConvState default arm");
        assert_eq!(c.kv_quant, KvQuant::Off, "KvQuant default arm");
        assert!(c.memory_budget.is_none(), "MemoryBudget off");
        assert!(!c.gemv_silu, "GemvSilu off");
        assert!(!c.gemv_residual, "GemvResidual off");
        assert!(c.tap_layers.is_empty(), "no tap by default");
        assert_eq!(c.tap_dtype, DType::F32, "tap dtype");
        assert_eq!(c.flash_page_size, 256);
        assert_eq!(c.flash_pages_per_chunk, 1);
        // Div-by-zero guard: `head_dim` divides by `num_attention_heads`, so the
        // default must not be zero even though the geometry is otherwise empty.
        assert_eq!(c.num_attention_heads, 1);
        assert_eq!(c.head_dim(), 0);
    }

    /// `#[serde(flatten)] extra` keeps every checkpoint key a typed field does
    /// not name (pydantic `extra="allow"`), and takes none that a typed field
    /// does — the two properties that make it lossless without shadowing.
    #[test]
    fn lfm2_config_flatten_keeps_unknown_keys_and_takes_no_known_ones() {
        let json = serde_json::json!({
            "vocab_size": 65536,
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "layer_types": ["full_attention", "conv"],
            // Keys candle's typed struct does not name — must land in `extra`.
            "architectures": ["Lfm2ForCausalLM"],
            "torch_dtype": "bfloat16",
            "unforeseen_future_field": 42
        });
        let cfg: Lfm2Config = serde_json::from_value(json).expect("deserialize");

        // Known keys parsed into typed fields, not into `extra`.
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.num_attention_heads, 32);
        assert!(
            !cfg.extra.contains_key("vocab_size")
                && !cfg.extra.contains_key("num_attention_heads")
                && !cfg.extra.contains_key("layer_types"),
            "a typed field must not also appear in `extra`"
        );

        // Unknown keys preserved losslessly.
        assert_eq!(
            cfg.extra["architectures"],
            serde_json::json!(["Lfm2ForCausalLM"])
        );
        assert_eq!(cfg.extra["torch_dtype"], serde_json::json!("bfloat16"));
        assert_eq!(cfg.extra["unforeseen_future_field"], serde_json::json!(42));
    }
}
