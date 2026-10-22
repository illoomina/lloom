//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::nn::kv_quant::QuantKvSlab;
use crate::nn::ops::{FlashScratchSizing, KvQuant};
use crate::nn::zeros as zs;
use crate::nn::{
    BatchedCausalLM, CacheReset, CausalLM, Conv1d, Conv1dConfig, Model as ModelTrait, RaggedRow,
    Row, Speculative, VarBuilder,
};
use crate::utils::repeat_kv;
use lloom_macros::Config as ConfigDerive;
// The SiLU steer arm is a Metal kernel switch (`unary.rs`), so the type and its
// axis field are metal-gated exactly as its former crate dependency was.
use crate::profile::{Arm, ArmMut, Picked as _, Profile, Regime as _, Rewind as _, Rows as _};
#[cfg(feature = "metal")]
use lloom_tensor::kernels::metal::{GpuClock, PageError, PageIdx};
#[cfg(feature = "metal")]
use lloom_tensor::kernels::SiluImpl;
use lloom_tensor::{DType, Device, IndexOp, Module, Result, Tensor};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
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
            stream_layers: None,
            // Both GEMV epilogue fusions are off, per §7.1a. #536 and #509 both
            // say the default needs the decode-path number, which is not taken
            // here — this change makes the arm reachable, nothing more.
            gemv_silu: false,
            gemv_residual: false,
            gemv_gate: GateFusion::Off,
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
    /// An `l_cache`-wide window shifted one column per token, in place in the
    /// storage the cache owns: the tail gathered to a scratch and written
    /// back, the new column appended — three copies per layer per token
    /// (`DESIGN.md` §6.1, §11.5z-x). The correctness bar the rings are
    /// compared against, and the arm every recorded digest belongs to.
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
/// dispatches per conv layer per token against the 3 the shuffle pays — while
/// costing 16 × 2048 × 2 B × 22 = 1.375 MiB of resident state. Past ~64 the
/// curve is flat and the memory is not.
const DEFAULT_RING_SLACK: usize = 16;

impl ConvState {
    /// The verify window a rewind can reach on this arm: the rotating
    /// ring's history, less the one slot the rows' shared pool lays its
    /// state out in (`lfm2_presets::pool`: `RotatingRing { k: 1 }` is the
    /// pooled rows' layout, not a window). The sliding arm never rewinds
    /// (`Cache::rewind_horizon`).
    pub fn verify_depth(&self, paged: bool) -> usize {
        match self {
            ConvState::RotatingRing { k } => k.saturating_sub(paged as usize),
            ConvState::Shuffle | ConvState::SlidingRing { .. } => 0,
        }
    }

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
    /// The library default and the CPU's path. On Metal it is the reference
    /// arm: it materializes the scores, and serving refuses it.
    #[default]
    Generic,
    /// The prepared fused route (`nn::attention::route`), GQA-native with F32
    /// accumulation inside, so neither `repeat_kv` nor the upcast is needed.
    ///
    /// Metal only. `seq_len == 1` takes `sdpa_vector`, or the tiled kernel at
    /// a head dim with no one-query kernel; `seq_len > 1` — prefill, a verify
    /// window, B>1 — takes the tiled `steel_attention`. A shape no fused
    /// kernel covers is refused at load.
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
    /// tiled `steel_attention` as `Sdpa` does, and a decode step FlashDecoding
    /// does not instantiate takes the prepared route's one-query kernel.
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

/// The kernel an attention layer's `seq_len == 1` step takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeAttn {
    /// The composed attention: `repeat_kv`, f32 `matmul`, softmax. The CPU's
    /// path, and on Metal the reference arm only.
    Generic,
    /// `sdpa_vector` (and its two-pass sibling past the key threshold).
    SdpaVector,
    /// The tiled `steel_attention` over one query, at a head dim no one-query
    /// kernel is instantiated at.
    Steel,
    /// FlashDecoding, over the contiguous cache or the page pool.
    FlashDecoding,
}

/// The kernel an attention layer's `seq_len > 1` pass (prefill, a verify
/// window) takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillAttn {
    /// The composed attention, with the causal mask.
    Generic,
    /// The tiled `steel_attention`, causal in-kernel.
    Steel,
}

impl DecodeAttn {
    fn name(self) -> &'static str {
        match self {
            DecodeAttn::Generic => "generic",
            DecodeAttn::SdpaVector => "sdpa_vector",
            DecodeAttn::Steel => "steel_attention",
            DecodeAttn::FlashDecoding => "flash_decoding",
        }
    }
}

impl PrefillAttn {
    fn name(self) -> &'static str {
        match self {
            PrefillAttn::Generic => "generic",
            PrefillAttn::Steel => "steel_attention",
        }
    }
}

/// The attention kernels a model's layers take, decided once at load from
/// the requested arm, the device, the compute dtype and the geometry.
///
/// On Metal a fused arm prepares one route (`nn::attention::route`) that the
/// forward pass dispatches through. A decode phase whose requested kernel does
/// not cover the shape takes the nearest fused kernel that does, which
/// [`Self::warning`] names; a shape no fused kernel covers is refused at load
/// with the missing capability named. The composed attention is the CPU's
/// path and, on Metal, only the reference arm [`AttnImpl::Generic`], which
/// serving refuses. [`Self::line`] is printed at every load. The head-dim
/// checks are shape probes (`docs/porting-models.md` §21.8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnRoute {
    pub requested: AttnImpl,
    pub head_dim: usize,
    pub dtype: DType,
    pub decode: DecodeAttn,
    pub prefill: PrefillAttn,
    /// The decode kernel the arm requested, when the route took another.
    pub decode_refused: Option<DecodeAttn>,
    /// The prefill kernel the arm requested, when the route took another.
    pub prefill_refused: Option<PrefillAttn>,
    /// The prepared fused route; `None` off Metal and on the reference arm.
    pub prepared: Option<crate::nn::attention::route::PreparedAttention>,
}

impl AttnRoute {
    pub fn resolve(
        requested: AttnImpl,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let route = Self {
            requested,
            head_dim: cfg.head_dim(),
            dtype,
            decode: DecodeAttn::Generic,
            prefill: PrefillAttn::Generic,
            decode_refused: None,
            prefill_refused: None,
            prepared: None,
        };
        if requested == AttnImpl::Generic || !device.is_metal() {
            return Ok(route);
        }
        #[cfg(feature = "metal")]
        return route.against_kernel_sets(cfg, device);
        #[cfg(not(feature = "metal"))]
        Ok(route)
    }

    /// What the fused route serves: causal attention over a strided cache,
    /// any batch, windows and keys up to the model's positions.
    pub fn spec(cfg: &Config, dtype: DType) -> crate::nn::attention::route::AttentionSpec {
        use crate::nn::attention::route::{AttentionBounds, AttentionSpec, KvLayout, MaskKind};
        let positions = cfg.max_position_embeddings.max(1);
        AttentionSpec {
            q_heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            dtype,
            kv_dtype: dtype,
            mask: MaskKind::Causal,
            softcap: 1.0,
            layout: KvLayout::Strided,
            bounds: AttentionBounds {
                batch: usize::MAX,
                min_queries: 1,
                queries: positions,
                keys: positions,
            },
        }
    }

    /// Whether FlashDecoding is instantiated at this head dim and dtype, and
    /// its pipelines on `device` admit the threadgroup widths they are
    /// dispatched at (a pipeline that does not writes nothing; the admitted
    /// width falls with the head dim's register demand, per device). The
    /// one-query `sdpa_vector` kernels are the prepared route's question.
    #[cfg(feature = "metal")]
    fn flash_covers(&self, device: &Device) -> bool {
        use lloom_tensor::kernels::{
            flash_admits_its_widths, FlashDType, FLASH_F32_HEAD_DIMS, FLASH_HEAD_DIMS,
        };
        let Ok(metal) = device.as_metal_device() else {
            return false;
        };
        let (raw, kernels) = (metal.device(), metal.kernels());
        let head_dim = self.head_dim;
        let is_f32 = self.dtype == DType::F32;
        let dims: &[usize] = match self.dtype {
            DType::F16 => &FLASH_HEAD_DIMS,
            DType::F32 => &FLASH_F32_HEAD_DIMS,
            _ => &[],
        };
        let dtype = if is_f32 {
            FlashDType::F32
        } else {
            FlashDType::F16
        };
        lloom_tensor::shape_req!(
            flash_decoding_head_dim,
            dims.contains(&head_dim);
            head_dim,
            is_f32
        ) && flash_admits_its_widths(raw, kernels, head_dim, dtype).unwrap_or(false)
    }

    /// The prepared route serves prefill and windows; decode takes, in
    /// preference order, the requested fused kernel, the other fused decode
    /// kernel, then the route's own one-query kernel. An attention no fused
    /// kernel covers is refused.
    #[cfg(feature = "metal")]
    fn against_kernel_sets(mut self, cfg: &Config, device: &Device) -> Result<Self> {
        use crate::nn::attention::route::{AttentionKernel, MetalKernelSet, PreparedAttention};
        let metal = device.as_metal_device()?;
        let prepared = PreparedAttention::prepare(
            Self::spec(cfg, self.dtype),
            &MetalKernelSet { device: metal },
        )
        .map_err(|u| {
            lloom_tensor::Error::Msg(format!(
                "lfm2: {u}: no fused kernel covers this attention, and the composed attention \
                 does not serve on Metal (`--attn generic` is the reference arm)"
            ))
        })?;
        self.prepared = Some(prepared);
        self.prefill = PrefillAttn::Steel;
        let one = match prepared.kernel_for(1, 1, 1)? {
            AttentionKernel::Full => DecodeAttn::Steel,
            _ => DecodeAttn::SdpaVector,
        };
        let order = match self.requested {
            AttnImpl::Sdpa => [DecodeAttn::SdpaVector, DecodeAttn::FlashDecoding],
            _ => [DecodeAttn::FlashDecoding, DecodeAttn::SdpaVector],
        };
        self.decode = order
            .into_iter()
            .find(|&k| match k {
                DecodeAttn::SdpaVector => one == DecodeAttn::SdpaVector,
                DecodeAttn::FlashDecoding => self.flash_covers(device),
                DecodeAttn::Generic | DecodeAttn::Steel => false,
            })
            .unwrap_or(one);
        if self.decode != order[0] {
            self.decode_refused = Some(order[0]);
        }
        Ok(self)
    }

    /// The one line a load prints when a phase does not take the kernel the
    /// arm requested; `None` when every phase does.
    pub fn warning(&self) -> Option<String> {
        let mut phases = Vec::new();
        if let Some(k) = self.decode_refused {
            phases.push(format!(
                "decode requested {} and takes {} (the nearest fused kernel)",
                k.name(),
                self.decode.name()
            ));
        }
        if let Some(k) = self.prefill_refused {
            phases.push(format!(
                "prefill requested {} and takes {}",
                k.name(),
                self.prefill.name()
            ));
        }
        (!phases.is_empty()).then(|| {
            format!(
                "lfm2: attention head_dim {} at {:?} under {:?}: {}",
                self.head_dim,
                self.dtype,
                self.requested,
                phases.join("; ")
            )
        })
    }

    /// Why this route takes the composed attention on `device`, if it does —
    /// on Metal only the reference arm, `AttnImpl::Generic`, does (a fused arm
    /// a shape no kernel covers is refused by [`Self::resolve`]) — for
    /// `nn::attention::route::ExecutionPurpose::admit`.
    pub fn composed_reason(&self, device: &Device) -> Option<String> {
        (device.is_metal() && self.prepared.is_none()).then(|| {
            format!(
                "attention under {:?} is the composed reference attention",
                self.requested
            )
        })
    }

    /// The route, printed at every load.
    pub fn line(&self) -> String {
        match &self.prepared {
            Some(p) => format!(
                "lfm2: attention under {:?}: decode {}, prefill {}; {}",
                self.requested,
                self.decode.name(),
                self.prefill.name(),
                p.describe()
            ),
            None => format!(
                "lfm2: attention under {:?}: the composed attention, head_dim {} at {:?}",
                self.requested, self.head_dim, self.dtype
            ),
        }
    }

    /// [`Self::warning`] on the load-warning channel.
    pub fn load_warning(&self) -> Option<crate::load_warning::LoadWarning> {
        self.warning().map(|m| {
            crate::load_warning::LoadWarning::new(
                crate::load_warning::LoadWarningKind::AttentionRoute,
                m,
            )
        })
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

/// The admission budget a caller declares (`nn::memory::MemoryBudget`);
/// re-exported here because every LFM2 caller names it through this module.
pub use crate::nn::memory::MemoryBudget;

/// Report a weight-term divergence §9.5m has detected.
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

/// How the SwiGLU gate product `silu(w1·x) * (w3·x)` is dispatched (issue
/// #732, `DESIGN.md` §12.2e row 3b, §12.2 row 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateFusion {
    /// `w3`'s GEMV, then a `bmul_f16` against the (fused or not) SiLU'd gate.
    /// What ships.
    #[default]
    Off,
    /// The product rides `w3`'s store with the gate as a row-indexed operand
    /// (`gemv_mul`): one dispatch fewer, and the product's barrier re-seated
    /// between `w1`'s and `w3`'s sweeps — measured a loss on the crown.
    Mul,
    /// Both weights swept by one kernel (`gemv_swiglu`): `w1`, `w3`, the SiLU
    /// and the product in one dispatch, no barrier between the sweeps. Under
    /// it `gemv_silu` is moot — `w1` is never dispatched alone.
    Swiglu,
}

impl GateFusion {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "off" => Ok(GateFusion::Off),
            "mul" => Ok(GateFusion::Mul),
            "swiglu" => Ok(GateFusion::Swiglu),
            other => Err(format!(
                "--gemv-gate must be `off`, `mul` or `swiglu`, got `{other}`"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GateFusion::Off => "off",
            GateFusion::Mul => "mul",
            GateFusion::Swiglu => "swiglu",
        }
    }

    /// The fragment name the optimizer's plan spells this arm as.
    pub fn fragment(self) -> Option<&'static str> {
        match self {
            GateFusion::Off => None,
            GateFusion::Mul => Some("mul"),
            GateFusion::Swiglu => Some("swiglu"),
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
    pub gemv_gate: GateFusion,
    /// The norms inside their consumers' GEMVs (the norm prologue).
    pub norm_gemv: bool,
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
            "lfm2 fusion=derived: {}; gemv_silu={} gemv_residual={} gemv_gate={} norm_gemv={} math={} ({} refused, {} validated, {} unvalidated",
            self.plan,
            self.gemv_silu,
            self.gemv_residual,
            self.gemv_gate.name(),
            self.norm_gemv,
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
        if !self.refused.is_empty() {
            write!(f, " refused[")?;
            for (i, (name, why)) in self.refused.iter().enumerate() {
                if i > 0 {
                    write!(f, "; ")?;
                }
                write!(f, "{name}: {why}")?;
            }
            write!(f, "]")?;
        }
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

/// How the layers' weights stream (`Config::stream_layers`): the
/// checkpoint files the groups read from, the slots the tier holds (one
/// layer's matrices each, sized for the largest) and how many groups the
/// loader runs ahead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamLayers {
    pub files: Vec<std::path::PathBuf>,
    pub slots: usize,
    pub depth: usize,
}

/// One pass's view of the MLP's projections: the resident `Linear`s by
/// reference, or the slot views a streamed layer took for this pass.
struct Projections<'a> {
    gate: std::borrow::Cow<'a, Linear>,
    up: std::borrow::Cow<'a, Linear>,
    down: std::borrow::Cow<'a, Linear>,
}

/// One pass's view of the attention's four projections (the norms stay
/// resident).
struct AttnWeights<'a> {
    q: std::borrow::Cow<'a, Linear>,
    k: std::borrow::Cow<'a, Linear>,
    v: std::borrow::Cow<'a, Linear>,
    o: std::borrow::Cow<'a, Linear>,
}

/// One pass's view of the short conv's two projections (the taps stay
/// resident: `hidden x l_cache` elements, not worth a slot).
struct ConvWeights<'a> {
    in_proj: std::borrow::Cow<'a, Linear>,
    out_proj: std::borrow::Cow<'a, Linear>,
}

enum MixerWeights<'a> {
    Attn(AttnWeights<'a>),
    Conv(ConvWeights<'a>),
}

/// A layer's matrices for one pass — the group §4.8 streams: the mixer's
/// projections and the MLP's, one slot. A resident layer borrows its
/// fields; a streamed layer takes its views at the layer's entry, once.
struct LayerWeights<'a> {
    mixer: MixerWeights<'a>,
    mlp: Projections<'a>,
}

/// The part order inside a layer's group: the MLP's `w1`, `w3`, `w2`, then
/// the mixer's projections (`q`, `k`, `v`, `out` | `in`, `out`).
const GROUP_PART_W1: usize = 0;
const GROUP_PART_W3: usize = 1;
const GROUP_PART_W2: usize = 2;
const GROUP_PART_MIXER: usize = 3;

/// One layer on the weight-group tier: its group, the tier (shared by
/// every layer) and the source its bytes come from.
#[derive(Clone)]
struct StreamedLayer {
    group: crate::nn::slot_cache::WeightGroupId,
    tier: std::sync::Arc<parking_lot::Mutex<crate::nn::weight_groups::WeightGroups>>,
    source: std::sync::Arc<dyn crate::nn::weight_groups::GroupSource>,
    depth: usize,
}

impl std::fmt::Debug for StreamedLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamedLayer")
            .field("group", &self.group)
            .finish()
    }
}

/// The runtime config. The `#[axis]`-tagged fields are the engine's
/// configuration axes (RESTRUCTURING §1.6): each contributes to `lloom::axes::AXES`
/// at link time and becomes a `ConfigArgs` clap flag. The shape fields
/// (`hidden_size`, `layer_types`, …) come from the checkpoint and are not axes.
///
/// `#[tiny]` marks what the shape campaign may search from the tiny
/// template (`models::lfm2_testing`, `docs/porting-models.md` §21.8).
#[derive(Debug, Clone, ConfigDerive)]
pub struct Config {
    #[tiny(axis)]
    pub vocab_size: usize,
    #[tiny(axis)]
    pub hidden_size: usize,
    #[tiny(axis)]
    pub intermediate_size: usize,
    #[tiny(axis, max = 8)]
    pub num_hidden_layers: usize,
    #[tiny(axis)]
    pub num_attention_heads: usize,
    #[tiny(axis)]
    pub num_key_value_heads: usize,
    #[tiny(fixed)]
    pub norm_eps: f64,
    #[tiny(fixed)]
    pub rope_theta: f32,
    #[tiny(fixed)]
    pub max_position_embeddings: usize,
    #[tiny(axis, max = 8)]
    pub conv_l_cache: usize,
    #[tiny(fixed)]
    pub conv_bias: bool,
    #[tiny(derived = "Self::tiny_layer_types(num_hidden_layers)", cover)]
    pub layer_types: Vec<LayerType>,
    #[tiny(fixed)]
    pub tie_embedding: bool,
    #[tiny(fixed)]
    pub bos_token_id: Option<u32>,
    #[tiny(fixed)]
    pub eos_token_id: Option<u32>,
    pub use_flash_attn: bool,
    /// Defaults to `Generic`, so `into_config` and every existing caller keep
    /// the path they had without naming it.
    #[axis(
        name = "AttnImpl",
        flag = "attn",
        variants = ["Generic", "Sdpa", "FlashDecoding"],
        default = "Generic",
        parser = "AttnImpl::parse",
        path = "kernels.attention.impl",
        status = "advanced",
        cli = "attn",
        arms = ["generic", "sdpa", "flash"],
        doc = "The attention kernel a decode step takes: `generic` (the composed attention), `sdpa` (the vector kernel) or `flash` (FlashDecoding over the contiguous cache or the page pool); unset = the preset's."
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
        parser = "FlashScratchSizing::parse",
        path = "kernels.attention.flash.scratch",
        status = "arm",
        arms = ["grow", "reserve", "bucket"],
        doc = "How FlashDecoding's partial scratch is sized: `grow`, `reserve` or `bucket`; unset = the preset's."
    )]
    pub flash_scratch_sizing: FlashScratchSizing,
    /// Defaults to `Cat`, for the same reason `attn_impl` defaults to `Generic`.
    #[axis(
        name = "KvAppend",
        flag = "kv-append",
        variants = ["Cat", "InPlace", "Paged"],
        default = "Cat",
        parser = "KvAppend::parse",
        path = "context.kv.layout",
        status = "advanced",
        arms = ["cat", "in-place", "paged"],
        doc = "How a step's keys and values join the KV cache: `cat` (a grown tensor), `in-place` (a preallocated slab) or `paged` (the page pool); unset = the preset's."
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
        parser = "KvQuant::parse",
        path = "context.kv.dtype",
        status = "stable",
        cli = "kv-dtype",
        aliases = ["kv-cache-dtype"],
        arms = ["off", "f16", "int8", "int4"],
        doc = "The KV store: `off` (the family's own dtype), `f16`, `int8` or `int4` (codes, Metal only); unset = the family's."
    )]
    pub kv_quant: KvQuant,
    /// Defaults to `Shuffle`, so every existing caller keeps §6.1's path.
    #[axis(
        name = "ConvState",
        flag = "conv-state",
        variants = ["Shuffle", "SlidingRing", "RotatingRing"],
        default = "Shuffle",
        parser = "ConvState::parse",
        path = "models.lfm2.conv_state",
        status = "arm",
        arms = ["shuffle", "ring[:<K>[:<slack>]]", "rotating[:<K>]"],
        doc = "How the short convolution keeps its state: `shuffle`, `ring[:<K>[:<slack>]]` or `rotating[:<K>]`; unset = the preset's."
    )]
    pub conv_state: ConvState,
    /// `DESIGN.md` §9.5's memory budget. `None` — admission off — by default,
    /// per §7.1a's rule that no default is flipped without its own argued
    /// decision. See `Cache::admit_memory_budget`.
    pub memory_budget: Option<MemoryBudget>,
    /// The decoder layers' matrices (the mixer's projections and the
    /// MLP's, one group a layer) streamed through `nn::weight_groups`
    /// (`profiles-and-memory-design.md` §4.8, #892 A6a): `None` — every
    /// weight resident — unless the loader finds the weights do not fit the
    /// quota. A model that fits never streams. The embedding, the norms and
    /// the conv taps stay resident.
    pub stream_layers: Option<StreamLayers>,
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
    #[axis(name = "GemvSilu", flag = "gemv-silu", variants = ["off", "on"], default = "off", bool, path = "kernels.fusion.gemv_silu", status = "arm", doc = "The SiLU on the gate projection's GEMV store; unset = the preset's.")]
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
    #[axis(name = "GemvResidual", flag = "gemv-residual", variants = ["off", "on"], default = "off", bool, path = "kernels.fusion.gemv_residual", status = "arm", doc = "The two residual adds on their producing GEMV's store; unset = the preset's.")]
    pub gemv_residual: bool,
    /// How the SwiGLU product `silu(w1(x)) * w3(x)` is dispatched (issue
    /// #732, `DESIGN.md` §12.2e row 3b, §12.2 row 5): `off` — `w3`'s GEMV
    /// then a `bmul_f16`; `mul` — the product on `w3`'s store with the gate
    /// as a row-indexed operand (one dispatch fewer, the barrier re-seated
    /// between the two sweeps, measured a loss); `swiglu` — both weights in
    /// one kernel, no barrier between the sweeps (three dispatches fewer).
    ///
    /// **Defaults to off**, per §7.1a. Metal, `seq_len == 1`; every other
    /// case takes the eager path. **Both arms move the digests by
    /// construction**: the product lands on the f32 accumulators where the
    /// pair multiplies rounded stores at `T` (§2.3.5a).
    #[axis(
        name = "GemvGate",
        flag = "gemv-gate",
        variants = ["off", "mul", "swiglu"],
        default = "off",
        parser = "GateFusion::parse",
        path = "kernels.fusion.gemv_gate",
        status = "arm",
        arms = ["off", "mul", "swiglu"],
        doc = "How the SwiGLU product is dispatched: `off` (a GEMV and a multiply), `mul` (the multiply on the up projection's store) or `swiglu` (the whole product in one GEMV); unset = the preset's."
    )]
    pub gemv_gate: GateFusion,
    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// `conv1d` dispatch (§12.2 row 1). Off by default (§7.1a): it MOVES the
    /// digest — the depthwise kernel accumulates serially ascending in f32 where
    /// `sum_keepdim` uses a `simd_sum` tree — so an unmoved-digest gate cannot
    /// clear it and the discriminator is CPU-backend parity (§2.3.5a). The win
    /// is on the dispatch axis: ~2–3 dispatches per conv layer become 1, and
    /// LFM2 has 22 conv layers, so it is the largest single dispatch reduction
    /// the catalogue leaves open (llama.cpp's decode conv is one kernel).
    #[axis(name = "ConvFused", flag = "conv-fused", variants = ["off", "on"], default = "off", bool, path = "kernels.fusion.conv", status = "arm", doc = "The fused short-convolution kernels; unset = the preset's.")]
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
    #[axis(name = "ConvPrefillCtx", flag = "conv-prefill-ctx", variants = ["off", "on"], default = "off", bool, path = "models.lfm2.conv_prefill_context", status = "arm", doc = "Carry the conv state across a multi-token pass onto a warm cache (forced on wherever a carry is needed); unset = the preset's.")]
    pub conv_prefill_ctx: bool,
    /// Whether the epilogue fusions are selected by their flags or derived by
    /// the optimizer at construction (issue #732, P1).
    ///
    /// **Defaults to `Manual`** (§7.1a): the derivation is a mechanism, and
    /// every recorded digest belongs to the flag-selected arms. Under
    /// `Derived` the plan is derived once in [`Model::new`] over the walk of
    /// the model as built (`crate::optimizer::walk`) — nothing runs per
    /// token — and the blocks' epilogue arms are set from it, so a `Derived`
    /// run and the `Manual` run whose flags match the plan compute the same
    /// bits.
    #[axis(
        name = "Fusion",
        flag = "fusion",
        variants = ["Manual", "Derived"],
        default = "Manual",
        parser = "FusionPolicy::parse",
        path = "kernels.fusion.policy",
        status = "arm",
        arms = ["manual", "derived"],
        doc = "Who decides the epilogue fusions a decode step takes: `manual` (the flags) or `derived` (the optimizer's plan over the model's walk); unset = each family's own."
    )]
    pub fusion: FusionPolicy,
    /// How the Q/K/V `transpose` views reach the norms and the KV append
    /// (`DESIGN.md` §6.2c). Defaults to the copying path, per §7.1a.
    #[axis(
        name = "QkvLayout",
        flag = "qkv-layout",
        variants = ["Contiguous", "Strided", "Fused"],
        default = "Contiguous",
        parser = "QkvLayout::parse",
        path = "kernels.attention.qkv_layout",
        status = "arm",
        arms = ["contiguous", "strided", "fused"],
        doc = "How the attention projections reach their norms and the KV append: `contiguous`, `strided` or `fused`; unset = the preset's."
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
        parser = "parse_silu_impl",
        path = "kernels.activation.silu",
        status = "reference",
        arms = ["exact", "recip", "literal", "coarse4", "coarse1", "hardswish", "relu"],
        doc = "The SiLU's arithmetic: `exact` (the shipping arm) or one of the reference approximations `recip`, `literal`, `coarse4`, `coarse1`, `hardswish`, `relu`; unset = the preset's."
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
    /// The shape this config's axes name, as the profile's `(rewind, rows)`
    /// (`profiles-and-memory-design.md` §3): `Paged` rows are the paged KV
    /// (`KvAppend::Paged`); a `Ring` rewind is a rotating conv ring with a
    /// verify window's depth beyond the one slot the rows' pool lays its
    /// state out in (`ConvState::verify_depth`), over a KV with a length to
    /// rewind (`Cat` has none).
    pub fn shape(&self) -> (&'static str, &'static str) {
        let paged = self.kv_append == KvAppend::Paged;
        let rewindable =
            self.kv_append != KvAppend::Cat && self.conv_state.verify_depth(paged) >= 1;
        (
            if rewindable {
                crate::profile::Ring::NAME
            } else {
                crate::profile::NoRewind::NAME
            },
            if paged {
                crate::profile::Paged::NAME
            } else {
                crate::profile::One::NAME
            },
        )
    }

    /// The profile `P` a model of this config is built at: the shape from
    /// the axes (`shape`), the regime from the contract in force.
    pub fn profile_key(&self) -> crate::profile::ProfileKey {
        let (rewind, rows) = self.shape();
        crate::profile::ProfileKey {
            regime: match lloom_tensor::tuning::contract() {
                lloom_tensor::tuning::Contract::Deterministic => crate::profile::Exact::NAME,
                lloom_tensor::tuning::Contract::Fast => crate::profile::Free::NAME,
            },
            rewind,
            rows,
        }
    }

    /// The key at `Exact`, the regime stated here if none is in force yet
    /// (`tuning::pin_regime`): what a test asks, which never inherits the
    /// default (RESTRUCTURING §2). The engine and the harnesses, whose
    /// contract is installed before any model, take `profile_key`.
    pub fn exact_key(&self) -> crate::profile::ProfileKey {
        let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
        self.profile_key()
    }

    /// Refuse a config whose axes are not `P`'s shape: the profile names
    /// which state exists, the axes must agree (a mismatch is the shape the
    /// type was meant to rule out, caught at construction rather than
    /// silently served).
    pub fn check_shape<P: Profile>(&self) -> Result<()> {
        let (rewind, rows) = self.shape();
        let key = P::KEY;
        if key.rewind != rewind || key.rows != rows {
            lloom_tensor::bail!(
                "lfm2: a {key} model on a config whose axes name `{rewind} × {rows}` \
                 (kv_append {:?}, conv_state {:?}): the profile and the preset must agree \
                 (profiles-and-memory-design.md §3; `lfm2_presets`)",
                self.kv_append,
                self.conv_state
            )
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The tiny models' layer pattern: attention at every third layer, the
    /// rest conv — both kinds from three layers up.
    pub fn tiny_layer_types(n_layers: usize) -> Vec<LayerType> {
        (0..n_layers)
            .map(|l| {
                if l % 3 == 2 {
                    LayerType::FullAttention
                } else {
                    LayerType::Conv
                }
            })
            .collect()
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
            stream_layers: None,
            gemv_silu: false,
            gemv_residual: false,
            gemv_gate: GateFusion::Off,
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
/// offset: `nn::slab::Slab` along the sequence dim, at the bucket's width,
/// widened only by `Cache::prepare_pass` before a pass appends
/// (`Grow::Fixed`). The read is a view, the write one constant-size
/// `copy2d`, and the buffer identity moves only at a grow (§6.2b); the
/// rationale is the slab's own.
type KvSlot = crate::nn::slab::Slab;

fn kv_slot(capacity: usize) -> KvSlot {
    crate::nn::slab::Slab::new(2, capacity, crate::nn::slab::Grow::Fixed, zs::LFM2_KV_SLAB)
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
        self.0.len()
    }

    fn rollback(&mut self, to: usize) -> Result<()> {
        let len = self.0.len();
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
pub struct PagedKv {
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
    /// The allocator with its liveness (`nn::page_pool::PagePool`): created
    /// at the first pass, when the batch is known.
    pool: Option<crate::nn::page_pool::PagePool>,
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
    /// Host staging reused across passes; device uploads remain immutable while queued.
    slot_staging: Vec<u32>,
    /// One live count for every attention layer of a one-sequence decode pass.
    /// Replaced, never overwritten: preceding GPU readers retain the old upload.
    live_count: Option<Tensor>,
    max_slot: usize,
    /// Every sequence's chunk table, `table_stride` entries each.
    table: Vec<u32>,
    table_stride: usize,
    /// `table` resident on the device for the one-sequence read: `U32`, sized
    /// for the ceiling, its first `table_written` entries the host's. A
    /// chunk's entry never changes once written (pages do not move; a rewind
    /// shortens the length, not the table), so a new chunk is appended in
    /// place past what any in-flight step reads; a table that changed inside
    /// the written prefix (a reset's recycled pages) gets a fresh tensor —
    /// the in-flight steps hold the old buffer through their command buffers.
    table_t: Option<Tensor>,
    table_written: usize,
    /// What `table_t`'s first `table_written` entries hold — the host's copy
    /// of what it wrote, so the prefix check reads no device memory.
    table_shadow: Vec<u32>,
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
        Ok(Self {
            page_size: config.flash_page_size.max(1),
            pages_per_chunk: config.flash_pages_per_chunk.max(1),
            kv_capacity,
            slab,
            pool_pages_override: crate::tuning::tuning().lfm2.pool_pages,
            churn_pages: crate::tuning::tuning().lfm2.pool_churn,
            pool: None,
            clock,
            pools: vec![None; num_layers],
            n_seqs: 0,
            len: 0,
            pass_len: 0,
            slots: None,
            slot_staging: Vec::new(),
            live_count: None,
            max_slot: 0,
            table: Vec::new(),
            table_stride: 0,
            table_t: None,
            table_written: 0,
            table_shadow: Vec::new(),
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

    /// Pages per sequence a pass of `seq_len` tokens would fail to take from
    /// the pool as it stands: the pages every sequence's append pops, less the
    /// free list and what the quarantine releases at the pass's drain. Zero
    /// before the pool exists (it is created sized for the slab) and under a
    /// pinned pool (exhaustion at it stays loud).
    ///
    /// Positions alone under-size the pool after a `clear`: the cleared
    /// sequence's pages are held until the epoch open at the clear completes,
    /// and nothing commits between a chat's turns, so the next turn's first
    /// pass finds every one of them held. Sized by positions, a pool of one
    /// turn refuses the refill of an empty sequence; this is the term that
    /// widens it instead (§10.3h's liveness kept, never a wait on the GPU).
    fn pages_short(&self, seq_len: usize) -> usize {
        let Some(pool) = self.pool.as_ref() else {
            return 0;
        };
        if self.pool_pages_override.is_some() || seq_len == 0 {
            return 0;
        }
        let n_seqs = self.n_seqs.max(1);
        pool.pages_short((0..n_seqs).map(|s| (s, seq_len)))
            .div_ceil(n_seqs)
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
    ///
    /// The ceiling bounds positions, not the pool: `Cache::plan_pass` refuses
    /// a pass past `kv_capacity`, and may size the pool past it by the pages a
    /// quarantine still holds (`pages_short`), so `slab` is taken as planned.
    fn grow_to(&mut self, slab: usize) -> Result<()> {
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
                let nk = zs::zeros(zs::LFM2_KV_PAGES, shape.clone(), pk.dtype(), pk.device())?;
                nk.slice_set(pk, 2, 0)?;
                let nv = zs::zeros(zs::LFM2_KV_PAGES, shape, pv.dtype(), pv.device())?;
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
            let mut pool = crate::nn::page_pool::PagePool::with_clock(
                self.page_size,
                pool_pages,
                n_seqs + 1,
                pool_pages + 1,
                Arc::clone(&self.clock),
            );
            if self.churn_pages > 0 {
                // No command buffer has read these pages, so recycling them
                // without an epoch is safe here and only here.
                let pages = pool.pages_mut();
                pages
                    .append(n_seqs, self.churn_pages * self.page_size)
                    .map_err(lloom_tensor::Error::wrap)?;
                let freed = pages.release(n_seqs);
                pages.recycle(&freed).map_err(lloom_tensor::Error::wrap)?;
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
        let (pool_pages, page_size) = (self.pool_pages(), self.page_size);
        let pool = self.pool.as_mut().expect("created above");
        pool.recycle_ready().map_err(lloom_tensor::Error::wrap)?;
        let pool = pool.pages_mut();

        let start = self.len;
        self.slot_staging.clear();
        let slots = &mut self.slot_staging;
        slots.reserve(n_seqs * seq_len);
        for s in 0..n_seqs {
            pool.append(s, seq_len).map_err(|e| {
                lloom_tensor::Error::msg(format!(
                    "lfm2 paged KV, sequence {s}: {e}. The pool holds {pool_pages} pages of \
                     {page_size} tokens (LLOOM_148_POOL_PAGES pins it; unpinned it grows \
                     with the KV slab, B x ceil(slab / page_size), toward the ceiling)"
                ))
            })?;
            let seq = pool.seq(s).expect("sequence index within max_seqs");
            slots.extend(seq.cells(self.page_size, start..start + seq_len));
        }
        self.len = start + seq_len;
        self.pass_len = seq_len;
        self.max_slot = slots.iter().copied().max().unwrap_or(0) as usize;
        self.slots = Some(Tensor::new(slots.as_slice(), device)?);
        self.live_count = if n_seqs == 1 && seq_len == 1 {
            Some(Tensor::new(&[self.len as u32], device)?)
        } else {
            None
        };

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
            let before = self.table.len();
            pool.extend_table_for(s, self.pages_per_chunk, &mut self.table);
            debug_assert_eq!(self.table.len() - before, n_chunks);
        }
        self.sync_resident_table(n_seqs, device)?;
        Ok(())
    }

    /// Keep the one-sequence resident table in step with `table`: append the
    /// new entries in place, or start a fresh tensor when the written prefix
    /// changed or the batch is not one sequence (then the read binds by value).
    fn sync_resident_table(&mut self, n_seqs: usize, device: &Device) -> Result<()> {
        if n_seqs != 1 || !device.is_metal() {
            self.table_t = None;
            self.table_written = 0;
            return Ok(());
        }
        let capacity = self
            .kv_capacity
            .div_ceil((self.page_size * self.pages_per_chunk).max(1))
            .max(self.table.len());
        let fresh = match &self.table_t {
            Some(t) => {
                t.elem_count() < self.table.len() || self.table_written > self.table.len() || {
                    // The written prefix must still be the table's prefix
                    // (a host compare against what was written; never a
                    // device read on the step).
                    let written = self.table_written.min(self.table.len());
                    self.table_shadow[..written] != self.table[..written]
                }
            }
            None => true,
        };
        if fresh {
            let mut init = self.table.clone();
            init.resize(capacity, 0);
            self.table_t = Some(Tensor::new(init.as_slice(), device)?);
            self.table_written = self.table.len();
            self.table_shadow.clear();
            self.table_shadow.extend_from_slice(&self.table);
            return Ok(());
        }
        if self.table.len() > self.table_written {
            let t = self.table_t.as_ref().expect("checked above");
            let start = self.table_written;
            // SAFETY: entries past `table_written` are read by no queued
            // dispatch (each step reads its own chunk count), and the prefix
            // is untouched.
            if !unsafe { t.write_host_slice::<u32>(start, &self.table[start..]) } {
                let mut init = self.table.clone();
                init.resize(capacity, 0);
                self.table_t = Some(Tensor::new(init.as_slice(), device)?);
            }
            self.table_shadow.extend_from_slice(&self.table[start..]);
            self.table_written = self.table.len();
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
                zs::zeros(zs::LFM2_KV_PAGES, shape, k.dtype(), k.device())?,
                zs::zeros(zs::LFM2_KV_PAGES, shape, v.dtype(), v.device())?,
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
        let pool = self.pool.as_ref().expect("a pass has begun").pages();
        let (pk, pv) = self.pools[block_idx]
            .as_ref()
            .expect("the layer has a pool");
        let mut ks = Vec::with_capacity(n_seqs);
        let mut vs = Vec::with_capacity(n_seqs);
        for s in 0..n_seqs {
            let seq = pool.seq(s).expect("sequence index within max_seqs");
            let ids: Vec<u32> = seq.cells(self.page_size, 0..n_prev).collect();
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
            for s in 0..self.n_seqs {
                pool.release(s);
            }
        }
        self.len = 0;
        self.pass_len = 0;
        self.slots = None;
        self.live_count = None;
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
                pool.pages_mut()
                    .rewind(s, n)
                    .map_err(lloom_tensor::Error::wrap)?;
            }
        }
        self.len -= n;
        Ok(())
    }

    // ---- The SLOT-INDEXED pool (`batched-decode-design.md` §12.2) ----
    //
    // One pool shared by every row of a batched loop, each row a SLOT with
    // its own table and its own length; a pass names the rows it feeds and
    // how many tokens each, in row order. The lockstep fields above (`len`,
    // `pass_len`, `slots`, `table`) are the per-cache arm's and unused here.

    /// A pool for `slots` sequences, created at once (the per-cache arm
    /// creates its pool at the first pass, when it learns its batch).
    fn new_shared(
        config: &Config,
        kv_capacity: usize,
        slab: usize,
        num_layers: usize,
        clock: Arc<GpuClock>,
        slots: usize,
    ) -> Result<Self> {
        let mut kv = Self::new(config, kv_capacity, slab, num_layers, clock)?;
        kv.n_seqs = slots;
        let pool_pages = kv.pool_pages();
        let mut pool = crate::nn::page_pool::PagePool::with_clock(
            kv.page_size,
            pool_pages,
            slots + 1,
            pool_pages + 1,
            Arc::clone(&kv.clock),
        );
        if kv.churn_pages > 0 {
            let pages = pool.pages_mut();
            pages
                .append(slots, kv.churn_pages * kv.page_size)
                .map_err(lloom_tensor::Error::wrap)?;
            let freed = pages.release(slots);
            pages.recycle(&freed).map_err(lloom_tensor::Error::wrap)?;
        }
        kv.pool = Some(pool);
        Ok(kv)
    }

    /// Slot `slot`'s live length.
    fn row_len(&self, slot: usize) -> usize {
        self.pool
            .as_ref()
            .and_then(|p| p.pages().seq(slot))
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Bytes widening the pool to `slab` tokens per slot would allocate, at
    /// `per_token` bytes a token across the layers.
    fn grow_bytes_to(&self, slab: usize, per_token: usize) -> usize {
        let slab = slab.min(self.kv_capacity);
        self.pool_tokens_at(slab)
            .saturating_sub(self.pool_tokens_at(self.slab))
            .saturating_mul(per_token)
    }

    /// Open a pass over `rows` — `(slot, tokens)` each, in the order the
    /// pass's `[1, T]` tokens are laid out — appending every row's tokens to
    /// its table and computing the token slots the write scatters to and
    /// the tables the read walks. Runs once per pass, at its first attention
    /// layer; every layer of the pass reuses the result.
    fn begin_rows(&mut self, rows: &[(usize, usize)], device: &Device) -> Result<RowsPass> {
        let (pool_pages, page_size, ppc) =
            (self.pool_pages(), self.page_size, self.pages_per_chunk);
        let pool = self
            .pool
            .as_mut()
            .ok_or_else(|| lloom_tensor::Error::msg("lfm2 paged KV: the shared pool is unset"))?;
        pool.recycle_ready().map_err(lloom_tensor::Error::wrap)?;

        let mut slots: Vec<u32> = Vec::new();
        let mut prev = Vec::with_capacity(rows.len());
        for &(slot, n) in rows {
            let start = pool.seq(slot).map(|s| s.len()).unwrap_or(0);
            prev.push(start);
            let appended = match pool.append(slot, n) {
                // A full pool whose free pages sit in the quarantine — the
                // rows a burst just released, against an epoch nothing has
                // committed since (a reset between requests encodes no
                // dispatch of its own): wait the GPU out once and recycle
                // them, rather than refusing a pool that is free in fact.
                Err(PageError::Exhausted { .. }) if pool.holds_any() => {
                    device.synchronize()?;
                    pool.recycle_ready().map_err(lloom_tensor::Error::wrap)?;
                    pool.pages_mut().append(slot, n)
                }
                other => other,
            };
            appended.map_err(|e| {
                lloom_tensor::Error::msg(format!(
                    "lfm2 paged KV, slot {slot}: {e}. The pool holds {pool_pages} pages of \
                     {page_size} tokens (LLOOM_148_POOL_PAGES pins it; unpinned it grows \
                     with the KV slab, slots x ceil(slab / page_size), toward the ceiling)"
                ))
            })?;
            let seq = pool.seq(slot).expect("slot index within max_seqs");
            slots.extend(seq.cells(page_size, start..start + n));
        }
        let max_slot = slots.iter().copied().max().unwrap_or(0) as usize;

        let chunk = page_size * ppc;
        let mut n_keys = Vec::with_capacity(rows.len());
        let mut pages = Vec::with_capacity(rows.len());
        let mut per_row = Vec::with_capacity(rows.len());
        for &(slot, _) in rows {
            let seq = pool.seq(slot).expect("slot index within max_seqs");
            if !pool.chunk_pages_are_contiguous(slot, ppc) {
                lloom_tensor::bail!(
                    "lfm2 paged KV: slot {slot}'s pages do not form contiguous chunks at \
                     pages_per_chunk={ppc}; a recycled pool needs `--flash-k 1` (DESIGN.md 10.3d)"
                )
            }
            n_keys.push(seq.len() as u32);
            pages.push(PageIdx::as_u32s(seq.pages()).to_vec());
            per_row.push(pool.table_for(slot, ppc));
        }
        let table_stride = n_keys
            .iter()
            .map(|&n| (n as usize).div_ceil(chunk))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut tables = Vec::with_capacity(rows.len() * table_stride);
        for mut t in per_row {
            t.resize(table_stride, 0);
            tables.extend(t);
        }
        // Uploaded here, once, and bound by every attention layer of the pass
        // (`flash_decoding_paged_ragged_bound`): the per-layer upload was the
        // pool arm's per-row host cost (§12.8).
        let tables_t = Tensor::new(tables.as_slice(), device)?;
        let n_keys_t = Tensor::new(n_keys.as_slice(), device)?;
        let dec_rows: Vec<usize> = (0..rows.len()).filter(|&r| rows[r].1 == 1).collect();
        let dec = if dec_rows.is_empty() || dec_rows.len() == rows.len() {
            None
        } else {
            let mut t = Vec::with_capacity(dec_rows.len() * table_stride);
            let mut n = Vec::with_capacity(dec_rows.len());
            for &r in &dec_rows {
                t.extend_from_slice(&tables[r * table_stride..(r + 1) * table_stride]);
                n.push(n_keys[r]);
            }
            Some(DecRows {
                tables_t: Tensor::new(t.as_slice(), device)?,
                n_keys_t: Tensor::new(n.as_slice(), device)?,
                rows: dec_rows,
                tables: t,
                n_keys: n,
            })
        };
        Ok(RowsPass {
            rows: rows.to_vec(),
            slots: Tensor::new(slots.as_slice(), device)?,
            max_slot,
            tables,
            table_stride,
            n_keys,
            pages,
            prev,
            tables_t,
            n_keys_t,
            dec,
        })
    }

    /// Write one layer's K and V for a pass: `k`/`v` are `[1, H_kv, T, D]`
    /// (the rows' tokens end to end in the pass's row order) or
    /// `[B, H_kv, 1, D]` (a decode batch, one token per row): the scatter's
    /// row `r` is `(r / t, r % t)`, which is the pass's row order either way.
    fn write_rows(
        &mut self,
        block_idx: usize,
        k: &Tensor,
        v: &Tensor,
        pass: &RowsPass,
    ) -> Result<()> {
        let (b, n_kv_heads, t, head_dim) = k.dims4()?;
        let want = pass.slots.dim(0)?;
        if b * t != want {
            lloom_tensor::bail!(
                "lfm2 paged KV: layer {block_idx} writes [{b}, {n_kv_heads}, {t}, {head_dim}] \
                 against a pass of {want} tokens"
            )
        }
        let tokens = self.pool_pages() * self.page_size;
        if self.pools[block_idx].is_none() {
            let shape = (1, n_kv_heads, tokens, head_dim);
            self.pools[block_idx] = Some((
                zs::zeros(zs::LFM2_KV_PAGES, shape, k.dtype(), k.device())?,
                zs::zeros(zs::LFM2_KV_PAGES, shape, v.dtype(), v.device())?,
            ));
        }
        let (pk, pv) = self.pools[block_idx].as_ref().expect("allocated above");
        k.kv_scatter_into(
            pk,
            &pass.slots,
            pass.max_slot,
            0,
            tokens * head_dim,
            head_dim,
        )?;
        v.kv_scatter_into(
            pv,
            &pass.slots,
            pass.max_slot,
            0,
            tokens * head_dim,
            head_dim,
        )?;
        Ok(())
    }

    /// Slot `slot`'s first `n_prev` tokens of one layer, gathered from the
    /// pool into `[1, H_kv, n_prev, D]`. A turn-boundary prefill only.
    fn gather_row(
        &self,
        block_idx: usize,
        slot: usize,
        n_prev: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let pool = self
            .pool
            .as_ref()
            .expect("a shared pool exists from construction");
        let (pk, pv) = self.pools[block_idx]
            .as_ref()
            .expect("the layer has a pool once a token was written");
        let seq = pool.seq(slot).expect("slot index within max_seqs");
        let ids: Vec<u32> = seq.cells(self.page_size, 0..n_prev).collect();
        let ids = Tensor::new(ids.as_slice(), device)?;
        Ok((pk.index_select(&ids, 2)?, pv.index_select(&ids, 2)?))
    }

    /// Discard slot `slot`'s last `n` positions (the pages stay, §10.2a).
    fn rewind_slot(&mut self, slot: usize, n: usize) -> Result<()> {
        if let Some(pool) = self.pool.as_mut() {
            pool.rewind(slot, n).map_err(lloom_tensor::Error::wrap)?;
        }
        Ok(())
    }

    /// Drop slot `slot`'s pages into the quarantine; returns the pages whose
    /// last referent this was.
    fn release_slot(&mut self, slot: usize) -> Vec<PageIdx> {
        let Some(pool) = self.pool.as_mut() else {
            return Vec::new();
        };
        pool.release(slot)
    }

    /// Slot `slot`'s page table.
    fn row_pages(&self, slot: usize) -> Vec<PageIdx> {
        self.pool
            .as_ref()
            .and_then(|p| p.pages().seq(slot))
            .map(|s| s.pages().to_vec())
            .unwrap_or_default()
    }

    /// Point `dst`'s first `n_pages` at `src`'s (`KvPagePool::share_prefix`).
    fn share_pages(&mut self, src: usize, dst: usize, n_pages: usize) -> Result<()> {
        let pool = self
            .pool
            .as_mut()
            .ok_or_else(|| lloom_tensor::Error::msg("lfm2 paged KV: the shared pool is unset"))?;
        pool.share_prefix(src, dst, n_pages)
            .map_err(lloom_tensor::Error::wrap)
    }

    /// Pages allocated to at least one slot, and pages free.
    fn page_counts(&self) -> (usize, usize) {
        self.pool
            .as_ref()
            .map(|p| (p.pages_in_use(), p.pages_free()))
            .unwrap_or((0, 0))
    }
}

/// One pass of a slot-indexed batch through the shared pool: the rows it
/// fed, the token slots their tokens went to, and the tables the read walks
/// (`batched-decode-design.md` §12.2). Built by `PagedKv::begin_rows` at the
/// pass's first attention layer and read by every layer of the pass.
#[cfg(feature = "metal")]
#[derive(Debug)]
pub(crate) struct RowsPass {
    /// `(slot, tokens)` per row, in the pass's row order.
    rows: Vec<(usize, usize)>,
    /// The token slot of every `(row, token)` this pass writes, in row
    /// order — `[T]` u32, the scatter's slot list.
    slots: Tensor,
    max_slot: usize,
    /// Every row's chunk table, `table_stride` entries each (the longest
    /// row's chunk count; a row's spare entries name page 0 and are never
    /// read — a chunk past the row's live count is empty).
    tables: Vec<u32>,
    table_stride: usize,
    /// Every row's live length AFTER the pass.
    n_keys: Vec<u32>,
    /// Every row's page table (a multi-token row's steel read walks pages).
    pages: Vec<Vec<u32>>,
    /// Every row's live length BEFORE the pass.
    prev: Vec<usize>,
    /// `tables` and `n_keys` on the device, `U32`, uploaded once per pass.
    tables_t: Tensor,
    n_keys_t: Tensor,
    /// The one-token rows' own tables and counts when the pass also feeds
    /// longer rows (a piece, a verify window); `None` when every row is a
    /// decode row (the whole pass is the subset) or none is.
    dec: Option<DecRows>,
}

/// The decode rows of a pass with longer rows beside them: their pass
/// indices, tables and live counts, on the host and on the device.
#[cfg(feature = "metal")]
#[derive(Debug)]
struct DecRows {
    rows: Vec<usize>,
    tables: Vec<u32>,
    n_keys: Vec<u32>,
    tables_t: Tensor,
    n_keys_t: Tensor,
}

#[cfg(feature = "metal")]
impl RowsPass {
    /// Row `r`'s position in the pass's `[1, T]` token axis and its length.
    fn span(&self, r: usize) -> (usize, usize) {
        let off: usize = self.rows[..r].iter().map(|&(_, n)| n).sum();
        (off, self.rows[r].1)
    }
}

/// The state a batched loop's rows share: one paged KV pool with a table
/// per SLOT, and one ring pool per conv layer with a ring per slot
/// (`batched-decode-design.md` §12.2). A row's [`Cache`] holds a lease on a
/// slot ([`Cache::for_batch`]); the per-row kernels address the slot, and
/// the batched arms of the layer address every live row in one dispatch.
#[derive(Debug, Clone)]
pub struct BatchPool {
    #[cfg(feature = "metal")]
    inner: Arc<parking_lot::Mutex<BatchInner>>,
    #[cfg(not(feature = "metal"))]
    _off: (),
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct BatchInner {
    kv: PagedKv,
    /// Per conv layer, `[slots, hidden, width]`, allocated on first use.
    conv: Vec<Option<Tensor>>,
    slots: usize,
    free: Vec<usize>,
    /// The batched conv arm's `(slots, phases)` index buffers, reused while
    /// the rows and phases are those they were built for.
    row_index: Option<RowIndex>,
    /// The conv rings after a page's LAST token, keyed by the page: taken
    /// when a row's pass begins at a page boundary, dropped when the page's
    /// last referent goes. What a row sharing the page starts its own ring
    /// from (`batched-decode-design.md` §12.3).
    snapshots: HashMap<PageIdx, RingSnapshot>,
    /// Whether the layers take the batched arms (one dispatch per kernel for
    /// the batch) or the per-row loop over the same pool — the A/B, and the
    /// identity gate's two arms. `LLOOM_148_ROWS_ARM=per-row` turns them off.
    batched_arms: bool,
    /// Batched-arm dispatch groups taken (attention and conv layers), a
    /// counter the gate reads so the arm cannot be silently skipped.
    shared_calls: u64,
}

/// A row's conv rings — one `[1, hidden, width]` per conv layer — and the
/// ring's write phase, as they stood after a page's last token.
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
struct RingSnapshot {
    rings: Vec<Option<Tensor>>,
    phase: usize,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct RowIndex {
    slots: Vec<u32>,
    phases: Vec<u32>,
    slots_t: Tensor,
    phases_t: Tensor,
}

/// A lease on one slot of a [`BatchPool`]; the slot's pages and ring return
/// to the pool when the last holder drops it.
#[cfg(feature = "metal")]
#[derive(Debug)]
struct SlotLease {
    inner: Arc<parking_lot::Mutex<BatchInner>>,
    slot: usize,
}

#[cfg(feature = "metal")]
impl Drop for SlotLease {
    fn drop(&mut self) {
        let mut g = self.inner.lock();
        g.release(self.slot);
        g.free.push(self.slot);
    }
}

#[cfg(feature = "metal")]
impl BatchInner {
    /// Drop `slot`'s pages (into the quarantine) and the snapshots of the
    /// pages that had no other referent.
    fn release(&mut self, slot: usize) {
        let freed = self.kv.release_slot(slot);
        for p in freed {
            self.snapshots.remove(&p);
        }
    }
}

/// What a [`BatchPool`] holds, for a gauge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchPoolReport {
    pub slots: usize,
    pub slots_free: usize,
    pub pages_in_use: usize,
    pub pages_free: usize,
    pub kv_bytes: usize,
    pub conv_bytes: usize,
}

impl BatchPool {
    /// A pool for `slots` rows on `config`'s arms. Refused off Metal, and on
    /// any arm but `Paged × FlashDecoding × RotatingRing` — the one whose
    /// per-row kernels take a slot (§12.2).
    #[allow(unused_variables)]
    pub fn new(config: &Config, device: &Device, slots: usize, kv_capacity: usize) -> Result<Self> {
        #[cfg(not(feature = "metal"))]
        {
            lloom_tensor::bail!("lfm2 BatchPool is Metal-only: the pool is read by Metal kernels")
        }
        #[cfg(feature = "metal")]
        {
            if slots == 0 {
                lloom_tensor::bail!("lfm2 BatchPool: no slots")
            }
            if config.kv_append != KvAppend::Paged || config.attn_impl != AttnImpl::FlashDecoding {
                lloom_tensor::bail!(
                    "lfm2 BatchPool needs `KvAppend::Paged` and `AttnImpl::FlashDecoding` \
                     (batched-decode-design.md 12.2), got {:?} x {:?}",
                    config.kv_append,
                    config.attn_impl
                )
            }
            if !matches!(config.conv_state, ConvState::RotatingRing { .. }) {
                lloom_tensor::bail!(
                    "lfm2 BatchPool needs `ConvState::RotatingRing` (the in-place ring the \
                     slot-indexed conv kernel writes), got {:?}",
                    config.conv_state
                )
            }
            let clock = match device {
                Device::Metal(d) => d.gpu_clock(),
                _ => lloom_tensor::bail!("lfm2 BatchPool is Metal-only"),
            };
            let kv_capacity = kv_capacity.min(config.max_position_embeddings).max(1);
            let slab = DEFAULT_KV_BUCKET.min(kv_capacity);
            let kv = PagedKv::new_shared(
                config,
                kv_capacity,
                slab,
                config.num_hidden_layers,
                clock,
                slots,
            )?;
            Ok(Self {
                inner: Arc::new(parking_lot::Mutex::new(BatchInner {
                    kv,
                    conv: vec![None; config.num_hidden_layers],
                    slots,
                    free: (0..slots).rev().collect(),
                    row_index: None,
                    snapshots: HashMap::new(),
                    batched_arms: crate::tuning::tuning().lfm2.rows_arm != "per-row",
                    shared_calls: 0,
                })),
            })
        }
    }

    /// Tokens per page — the sharing granularity (§12.3).
    pub fn page_size(&self) -> usize {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().kv.page_size
        }
        #[cfg(not(feature = "metal"))]
        {
            0
        }
    }

    /// Ring snapshots held (one per completed page still referenced).
    pub fn snapshots(&self) -> usize {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().snapshots.len()
        }
        #[cfg(not(feature = "metal"))]
        {
            0
        }
    }

    /// The ceiling in tokens per slot.
    pub fn kv_capacity(&self) -> usize {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().kv.kv_capacity
        }
        #[cfg(not(feature = "metal"))]
        {
            0
        }
    }

    /// Slots, pages and bytes held now.
    pub fn report(&self) -> BatchPoolReport {
        #[cfg(not(feature = "metal"))]
        {
            BatchPoolReport::default()
        }
        #[cfg(feature = "metal")]
        {
            let g = self.inner.lock();
            let (pages_in_use, pages_free) = g.kv.page_counts();
            BatchPoolReport {
                slots: g.slots,
                slots_free: g.free.len(),
                pages_in_use,
                pages_free,
                kv_bytes: g.kv.bytes(),
                conv_bytes: g
                    .conv
                    .iter()
                    .flatten()
                    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                    .sum(),
            }
        }
    }

    /// Take the batched arms (`true`, the default) or the per-row loop over
    /// the same pool. The two are one arithmetic; the gate compares them.
    #[allow(unused_variables)]
    pub fn set_batched_arms(&self, on: bool) {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().batched_arms = on;
        }
    }

    /// Whether the batched arms are on.
    pub fn batched_arms(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().batched_arms
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    /// Batched-arm dispatch groups taken so far (one per attention or conv
    /// layer per pass that went through the batched arm).
    pub fn shared_calls(&self) -> u64 {
        #[cfg(feature = "metal")]
        {
            self.inner.lock().shared_calls
        }
        #[cfg(not(feature = "metal"))]
        {
            0
        }
    }

    #[cfg(feature = "metal")]
    fn lease(&self) -> Result<SlotLease> {
        let slot = {
            let mut g = self.inner.lock();
            g.free.pop()
        };
        match slot {
            Some(slot) => Ok(SlotLease {
                inner: self.inner.clone(),
                slot,
            }),
            None => lloom_tensor::bail!("lfm2 BatchPool: every slot is leased"),
        }
    }

    #[cfg(feature = "metal")]
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// A row's place in a [`BatchPool`]: its slot lease and, within a pass, the
/// pass it is part of.
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct BatchSlot {
    pool: BatchPool,
    lease: Arc<SlotLease>,
    /// This row's own pass (the per-row path), opened at the first attention
    /// layer of a pass and dropped when the next pass begins.
    pass: Option<Arc<RowsPass>>,
    /// The row's live length when the current pass began — the position of
    /// the pass's first token, which the conv layers (some of them before
    /// the pass's first attention layer, so before the pass's pages exist)
    /// need to place a page boundary inside a piece.
    pass_start: usize,
    /// Tokens per page, read once at the lease.
    page_size: usize,
    /// The conv rings after a page's last token for every page boundary
    /// INSIDE this pass's piece, by the token's position: each conv layer
    /// fills its ring as it writes the piece; `flush_ring_snapshots` keys
    /// them by page when the pass ends and its pages exist (§12.3, the
    /// share granularity as the page rather than the chunk).
    staged_snaps: Vec<(usize, Vec<Option<Tensor>>)>,
}

#[cfg(feature = "metal")]
impl BatchSlot {
    fn slot(&self) -> usize {
        self.lease.slot
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
    /// The grow is a shared [`BatchPool`]'s: rows of one pool plan the same
    /// grow, and a batched pass admits it once (`Model::forward_rows`).
    pub(crate) shared: bool,
}

/// The bytes a batched pass's grows take together: every private plan's,
/// and a shared pool's once (its rows all plan the same grow).
fn grow_bytes_of(plans: &[Option<GrowPlan>]) -> usize {
    let mut private = 0usize;
    let mut shared = 0usize;
    for p in plans.iter().flatten() {
        if p.shared {
            shared = shared.max(p.bytes);
        } else {
            private += p.bytes;
        }
    }
    private + shared
}

/// Where the grow was asked for, for the refusal's line.
impl std::fmt::Display for GrowPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
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

/// The most cached causal masks a cache holds before the map is emptied: a
/// decode step's `[1, kv_len]` mask and a prefill tile's `[tile, kv_len]` are
/// each keyed on their `kv_len`, so a long run names one new shape per step or
/// tile and nothing ever asks for an old one again.
const MASK_CACHE_ENTRIES: usize = 16;
/// The `One` shapes' KV: a sequence's own store. Which of the three it is
/// follows `KvAppend` × `KvQuant`, preset values the profile does not carry
/// (`profiles-and-memory-design.md` §3: a value, not a shape), so the three
/// are one enum rather than three types; a `Paged` config is not a `One`
/// state and is refused at construction.
#[derive(Debug, Clone)]
pub enum OwnKv {
    /// `KvAppend::Cat`: the live (K, V) per layer, grown by `Tensor::cat`.
    Cat(Vec<Option<(Tensor, Tensor)>>),
    /// `KvAppend::InPlace`: one (K, V) slot pair per layer.
    InPlace(Vec<(KvSlot, KvSlot)>),
    /// `KvQuant` on: one K + V quantized slab per layer (`nn::kv_quant`,
    /// the block Qwen4-exp's QSA stores through too).
    Quant(Vec<QuantKvSlab>),
}

impl OwnKv {
    fn new(config: &Config, slab: usize, num_layers: usize) -> Result<Self> {
        if config.kv_append == KvAppend::Paged {
            lloom_tensor::bail!(
                "lfm2: a `One` state on a `KvAppend::Paged` config: the paged KV is the `Paged` \
                 shapes' (profiles-and-memory-design.md §3)"
            )
        }
        if config.kv_quant.is_on() {
            return Ok(OwnKv::Quant(
                (0..num_layers)
                    .map(|_| {
                        QuantKvSlab::fixed(config.kv_quant, slab)
                            .scales_at(DType::F16)
                            .at_site(zs::LFM2_KV_QUANT)
                    })
                    .collect(),
            ));
        }
        Ok(match config.kv_append {
            KvAppend::Cat => OwnKv::Cat(vec![None; num_layers]),
            KvAppend::InPlace => OwnKv::InPlace(
                (0..num_layers)
                    .map(|_| (kv_slot(slab), kv_slot(slab)))
                    .collect(),
            ),
            KvAppend::Paged => unreachable!("refused above"),
        })
    }

    /// Whether any layer's planes exist (`Cat` holds no planes).
    fn is_allocated(&self) -> bool {
        match self {
            OwnKv::Cat(_) => false,
            OwnKv::InPlace(slots) => slots.iter().any(|(k, _)| k.is_allocated()),
            OwnKv::Quant(slabs) => slabs.iter().any(|s| s.is_allocated()),
        }
    }

    /// Positions held, read from the first layer that holds any — 22 of
    /// LFM2's 30 layers are conv and hold nothing here (§5.3).
    fn len(&self) -> usize {
        match self {
            OwnKv::Cat(kvs) => kvs
                .iter()
                .flatten()
                .next()
                .and_then(|(k, _)| k.dim(2).ok())
                .unwrap_or(0),
            OwnKv::InPlace(slots) => slots
                .iter()
                .find(|(k, _)| k.is_allocated())
                .map(|(k, _)| k.len())
                .unwrap_or(0),
            OwnKv::Quant(slabs) => slabs
                .iter()
                .find(|s| s.is_allocated())
                .map(|s| s.len())
                .unwrap_or(0),
        }
    }

    /// The in-place slots' live length, for the speculative window's check.
    fn slot_len(&self) -> Option<usize> {
        match self {
            OwnKv::InPlace(slots) => slots
                .iter()
                .find(|(k, _)| k.is_allocated())
                .map(|(k, _)| k.len()),
            _ => None,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            OwnKv::Cat(kvs) => kvs
                .iter()
                .flatten()
                .map(|(k, v)| {
                    k.elem_count() * k.dtype().size_in_bytes()
                        + v.elem_count() * v.dtype().size_in_bytes()
                })
                .sum(),
            OwnKv::InPlace(slots) => slots.iter().map(|(k, v)| k.bytes() + v.bytes()).sum(),
            OwnKv::Quant(slabs) => slabs.iter().map(|s| s.bytes()).sum(),
        }
    }

    /// Bytes one token costs across the allocated planes, read from their
    /// shapes (0 before the first append, and on `Cat`).
    fn bytes_per_token(&self) -> usize {
        match self {
            OwnKv::Cat(_) => 0,
            OwnKv::InPlace(slots) => slots
                .iter()
                .flat_map(|(k, v)| [k, v])
                .filter_map(|s| s.data().as_ref())
                .map(|t| {
                    let d = t.dims();
                    d[0] * d[1] * d[3] * t.dtype().size_in_bytes()
                })
                .sum(),
            OwnKv::Quant(slabs) => slabs.iter().map(|s| s.bytes_per_token()).sum(),
        }
    }

    /// Widen every plane to `slab` tokens (`Cat` has no slab).
    fn grow_to(&mut self, slab: usize) -> Result<()> {
        match self {
            OwnKv::Cat(_) => {}
            OwnKv::InPlace(slots) => {
                for (k, v) in slots.iter_mut() {
                    k.grow_to(slab)?;
                    v.grow_to(slab)?;
                }
            }
            OwnKv::Quant(slabs) => {
                for s in slabs.iter_mut() {
                    s.grow_to(slab)?;
                }
            }
        }
        Ok(())
    }

    /// Set every unallocated plane's width.
    fn set_capacity(&mut self, slab: usize) {
        match self {
            OwnKv::Cat(_) => {}
            OwnKv::InPlace(slots) => {
                for (k, v) in slots.iter_mut() {
                    k.set_capacity(slab);
                    v.set_capacity(slab);
                }
            }
            OwnKv::Quant(slabs) => {
                for s in slabs.iter_mut() {
                    s.set_capacity(slab);
                }
            }
        }
    }

    /// Drop every plane: a new sequence may have a different batch size, so
    /// keeping the allocation would pin the previous one's.
    fn clear(&mut self) {
        match self {
            OwnKv::Cat(kvs) => kvs.iter_mut().for_each(|v| *v = None),
            OwnKv::InPlace(slots) => slots.iter_mut().for_each(|(k, v)| {
                k.reset();
                v.reset();
            }),
            OwnKv::Quant(slabs) => slabs.iter_mut().for_each(|s| s.reset()),
        }
    }

    /// Every allocated plane's length moved back by `n` (§10.2a: the rows
    /// above the length are unreachable and never cleared). Only the
    /// attention layers' planes are allocated — a conv layer's never is,
    /// and rewinding one would be an error, not a no-op.
    fn rewind(&mut self, n: usize) -> Result<()> {
        match self {
            OwnKv::Cat(_) => {}
            OwnKv::InPlace(slots) => {
                for (k_slot, v_slot) in slots.iter_mut() {
                    if !k_slot.is_allocated() {
                        continue;
                    }
                    k_slot.rewind(n)?;
                    v_slot.rewind(n)?;
                }
            }
            OwnKv::Quant(slabs) => {
                for s in slabs.iter_mut() {
                    if !s.is_allocated() {
                        continue;
                    }
                    s.rewind(n)?;
                }
            }
        }
        Ok(())
    }
}

/// The `Paged` shapes' KV: the sequence's own page pool, or a row of a
/// shared [`BatchPool`] (`State::for_batch`) whose pages and conv ring live
/// in the pool, addressed by slot. `Off` is a `Paged` state built without a
/// cache. There is no quantized variant: a quantized `Paged` config is
/// refused at construction, since `KvQuant` is a value the type cannot see.
#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
// One per sequence, and the own pool's bookkeeping is what the state held
// before the arms were typed: no indirection on the per-layer read.
#[allow(clippy::large_enum_variant)]
pub enum PooledKv {
    Own(PagedKv),
    Shared(BatchSlot),
    Off,
}

/// Off Metal the `Paged` shapes have no store — the pool is read by a Metal
/// kernel — so the arm is a refusal at construction and holds nothing.
#[cfg(not(feature = "metal"))]
#[derive(Debug, Clone)]
pub struct PooledKv;

impl PooledKv {
    /// What a `Paged` config must be, whichever pool it is served from.
    fn check(config: &Config, device: &Device) -> Result<()> {
        if config.kv_append != KvAppend::Paged {
            lloom_tensor::bail!(
                "lfm2: a `Paged` state on a `{:?}` config: the `One` shapes hold a sequence's \
                 own KV (profiles-and-memory-design.md §3)",
                config.kv_append
            )
        }
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
        Ok(())
    }

    #[allow(unused_variables)]
    fn new(
        config: &Config,
        device: &Device,
        kv_capacity: usize,
        slab: usize,
        num_layers: usize,
        use_kv_cache: bool,
    ) -> Result<Self> {
        Self::check(config, device)?;
        #[cfg(not(feature = "metal"))]
        {
            lloom_tensor::bail!("lfm2 `KvAppend::Paged` is Metal-only")
        }
        #[cfg(feature = "metal")]
        {
            if !use_kv_cache {
                return Ok(PooledKv::Off);
            }
            let clock = match device {
                Device::Metal(d) => d.gpu_clock(),
                _ => lloom_tensor::bail!("lfm2 `KvAppend::Paged` is Metal-only"),
            };
            Ok(PooledKv::Own(PagedKv::new(
                config,
                kv_capacity,
                slab,
                num_layers,
                clock,
            )?))
        }
    }
}

#[cfg(feature = "metal")]
impl PooledKv {
    fn len(&self) -> usize {
        match self {
            PooledKv::Own(p) => p.len,
            PooledKv::Shared(b) => b.pool.inner.lock().kv.row_len(b.slot()),
            PooledKv::Off => 0,
        }
    }

    /// The pool's bytes: a shared pool's are every row's, which a gauge
    /// reads once through `BatchPool::report`.
    fn bytes(&self) -> usize {
        match self {
            PooledKv::Own(p) => p.bytes(),
            PooledKv::Shared(b) => b.pool.inner.lock().kv.bytes(),
            PooledKv::Off => 0,
        }
    }

    fn grow_to(&mut self, slab: usize) -> Result<()> {
        match self {
            PooledKv::Own(p) => p.grow_to(slab),
            PooledKv::Shared(b) => b.pool.inner.lock().kv.grow_to(slab),
            PooledKv::Off => Ok(()),
        }
    }

    /// Set the width a pool that does not exist yet will take, and return
    /// the width the state records: an existing pool keeps its own.
    fn set_slab(&mut self, slab: usize) -> usize {
        match self {
            PooledKv::Own(p) => {
                if p.pool.is_none() {
                    p.slab = slab;
                    slab
                } else {
                    p.slab
                }
            }
            PooledKv::Shared(b) => b.pool.inner.lock().kv.slab,
            PooledKv::Off => slab,
        }
    }

    /// Bytes widening the pool from `from` to `slab` tokens per sequence
    /// allocates: a pool grows by whole pages for every sequence, so the
    /// token count is the pool's; the per-token cost is read from an
    /// allocated layer's pool when one exists, else it is `per_token`.
    fn grow_bytes(&self, from: usize, slab: usize, per_token: usize) -> usize {
        match self {
            PooledKv::Own(p) => {
                let tokens = p
                    .pool_tokens_at(slab)
                    .saturating_sub(p.pool_tokens_at(from));
                let measured: usize = p
                    .pools
                    .iter()
                    .flatten()
                    .map(|(k, v)| {
                        let (dk, dv) = (k.dims(), v.dims());
                        dk[0] * dk[1] * dk[3] * k.dtype().size_in_bytes()
                            + dv[0] * dv[1] * dv[3] * v.dtype().size_in_bytes()
                    })
                    .sum();
                if measured > 0 {
                    measured * tokens
                } else {
                    per_token * tokens
                }
            }
            PooledKv::Shared(_) | PooledKv::Off => per_token * slab.saturating_sub(from),
        }
    }

    fn clear(&mut self) {
        match self {
            PooledKv::Own(p) => p.release_all(),
            PooledKv::Shared(b) => {
                b.pool.inner.lock().release(b.slot());
                b.pass = None;
            }
            PooledKv::Off => {}
        }
    }

    fn rewind(&mut self, n: usize) -> Result<()> {
        match self {
            PooledKv::Own(p) => p.rewind(n),
            PooledKv::Shared(b) => b.pool.inner.lock().kv.rewind_slot(b.slot(), n),
            PooledKv::Off => Ok(()),
        }
    }

    /// The unwritten rewind's KV half: the sequence's own pool moves its
    /// length back; a shared pool's row length advances only in the pass
    /// that writes it, which an unwritten position never opened.
    fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        match self {
            PooledKv::Own(p) => p.rewind(n),
            PooledKv::Shared(_) | PooledKv::Off => Ok(()),
        }
    }
}

#[cfg(not(feature = "metal"))]
impl PooledKv {
    fn len(&self) -> usize {
        0
    }
    fn bytes(&self) -> usize {
        0
    }
    fn grow_to(&mut self, _slab: usize) -> Result<()> {
        Ok(())
    }
    fn set_slab(&mut self, slab: usize) -> usize {
        slab
    }
    fn grow_bytes(&self, from: usize, slab: usize, per_token: usize) -> usize {
        per_token * slab.saturating_sub(from)
    }
    fn clear(&mut self) {}
    fn rewind(&mut self, _n: usize) -> Result<()> {
        Ok(())
    }
    fn rewind_unwritten(&mut self, _n: usize) -> Result<()> {
        Ok(())
    }
}

/// The family's two KV arms, picked by the rows axis (`profile::Fork`).
#[derive(Debug, Clone, Copy)]
pub struct KvFork;

impl crate::profile::Fork for KvFork {
    type One = OwnKv;
    type Paged = PooledKv;
}

/// The KV store the rows axis names: an [`OwnKv`] on the `One` shapes, a
/// [`PooledKv`] on `Paged`.
pub type KvOf<P> = <<P as Profile>::Rows as crate::profile::Rows>::Pick<KvFork>;

/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct State<P: Profile> {
    profile: PhantomData<P>,
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    /// Private retained-request continuation; never changes ordinary chat defaults.
    branch_continuation: bool,
    /// The KV, one representation per shape (`profiles-and-memory-design.md`
    /// §7.1): the other arm is not a field.
    kv: KvOf<P>,
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
    /// This cache's row on the device's ledger (`ledger.rs`): the slab as
    /// its floor, the ceiling as its want, a bucket as its unit, the KV
    /// bytes held. Dropped with the cache; a clone shares it.
    #[cfg(feature = "metal")]
    ledger: Option<Arc<lloom_tensor::kernels::metal::ledger::Gauge>>,
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
    /// Per conv layer, the `Shuffle` state the fused decode step writes next:
    /// the step reads `conv_states[l]` and writes this, and the two swap, so
    /// a decode step allocates no state (`DESIGN.md` §9.2aa). Scratch, kept
    /// across `reset`; allocated on a layer's first fused step, outside every
    /// plan.
    conv_spare: Vec<Option<Tensor>>,
    /// The turn checkpoints: the last [`TURN_CHECKPOINTS`] multi-token
    /// passes, each the tail of its conv input per conv layer, so the cache
    /// can be put back at any position within [`TURN_CHECKPOINT_SLACK`] of a
    /// pass's end ([`Cache::restore`]). The slots' tails live in `turn_ring`,
    /// one allocation per sequence; this is their bookkeeping.
    turn_checkpoints: Vec<TurnCheckpoint>,
    /// The checkpoints' tails: [`TURN_CHECKPOINTS`] frames of `[conv layers,
    /// b, hidden, l_cache + TURN_CHECKPOINT_SLACK]`, allocated once at a
    /// sequence's first checkpointing pass and kept across `clear` (the next
    /// sequence of the same geometry writes over it), so no pass allocates a
    /// tail (`nn::ring::Ring`).
    turn_ring: Option<crate::nn::ring::Ring>,
    /// The model's weight-group tier, when its layers stream (§4.8) —
    /// attached by the model at each pass's entry, so the KV grow's
    /// admission can apply the arbiter's pledge (the tier's yield) and
    /// ask again, as Qwen4-exp's `prepare_pass_as` does with its experts.
    layer_tier: Option<std::sync::Arc<parking_lot::Mutex<crate::nn::weight_groups::WeightGroups>>>,
    /// The ring slot the pass in flight writes its tails into, `None` on a
    /// pass that takes no checkpoint (a decode step, a verify, `Cat`).
    turn_ckpt_cur: Option<usize>,
    /// The slot the next checkpointing pass takes.
    turn_ckpt_next: usize,
    /// The ring holds one frame past its slots, [`TURN_CHECKPOINTS`], for a
    /// pinned checkpoint, which the slots' rotation never takes
    /// (`sessions.md` §8).
    turn_pin_frame: bool,
    /// The pinned checkpoint, in the pin's frame.
    turn_pinned: Option<TurnCheckpoint>,
    /// The open request's second target: a checkpoint at this position,
    /// taken into the pin's frame by the pass that covers it.
    turn_pin_want: Option<usize>,
    /// A request is open (`CausalLM::open_checkpoint_request`): a second
    /// target is scheduled only inside one.
    turn_request: bool,
    /// The pass in flight's start and length, and the second target it
    /// takes, if any.
    turn_pass: (usize, usize),
    turn_pin_cur: Option<usize>,
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
    /// The speculative round's readback words, resident and CPU-mapped
    /// (`VerifyWords`); `None` until the first verify, or off Metal. A cloned
    /// cache starts without them (two caches must not share one stamp).
    verify_words: VerifyWordsSlot,
}

/// Full current convolution state, not a normalized prefill checkpoint.
pub struct RetainedState {
    conv: Vec<Option<Tensor>>,
    cat: Option<Vec<Option<(Tensor, Tensor)>>>,
    kv_len: usize,
    phase: usize,
    phase_prev: usize,
    history: usize,
    compact: bool,
}

impl<P: Profile> crate::nn::branch::RetainState for State<P> {
    type Retained = RetainedState;
    fn prepare_branching(&mut self) -> Result<()> {
        if self.speculative || !self.use_kv_cache || self.kv_len() != 0 {
            lloom_tensor::bail!("lfm2 branching requires a fresh non-speculative KV cache")
        }
        self.branch_continuation = true;
        Ok(())
    }
    fn retain_state(&mut self) -> Result<Self::Retained> {
        if self.speculative || !self.use_kv_cache {
            lloom_tensor::bail!("lfm2 retention requires a settled KV-cached sequence")
        }
        let mut conv = crate::nn::branch::retained_vec(self.conv_states.len())?;
        let cat = match self.kv.arm() {
            Arm::One(OwnKv::Cat(kv)) => {
                let mut saved = crate::nn::branch::retained_vec(kv.len())?;
                // Cat replaces the old prefix, never mutates its backing.
                saved.extend(kv.iter().cloned());
                Some(saved)
            }
            _ => None,
        };
        let _persistent = lloom_tensor::EscapeGuard::new();
        for t in &self.conv_states {
            conv.push(t.as_ref().map(crate::nn::recurrent::own_copy).transpose()?);
        }
        Ok(RetainedState {
            conv,
            cat,
            kv_len: self.kv_len(),
            phase: self.conv_phase,
            phase_prev: self.conv_phase_prev,
            history: self.conv_history,
            compact: self.conv_compact,
        })
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        if self.speculative
            || !self.use_kv_cache
            || saved.kv_len > self.kv_len()
            || saved.conv.len() != self.conv_states.len()
        {
            lloom_tensor::bail!("lfm2 retention: invalid window, KV prefix or layer count")
        }
        match (self.kv.arm_mut(), &saved.cat) {
            (ArmMut::One(OwnKv::Cat(kv)), Some(cat)) => {
                if kv.len() != cat.len() {
                    lloom_tensor::bail!("lfm2 retained Cat prefix has different layer count")
                }
                kv.clone_from_slice(cat);
            }
            (ArmMut::One(OwnKv::Cat(_)), None) | (_, Some(_)) => {
                lloom_tensor::bail!("lfm2 retained KV representation changed")
            }
            (_, None) => self.rewind_kv(self.kv_len() - saved.kv_len)?,
        }
        self.restore_retained_conv(saved)
    }
}

/// A fresh cache adopts a trunk on the Cat arm only: Cat replaces a layer's
/// KV on every append and never writes into the retained tensors, so rows
/// continuing one trunk share its prefix. An in-place slab is one buffer a
/// row would write its own tokens into.
impl<P: Profile> crate::nn::branch::AdoptRetained for State<P> {
    fn adopt_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        if self.speculative
            || !self.use_kv_cache
            || self.kv_len() != 0
            || saved.conv.len() != self.conv_states.len()
        {
            lloom_tensor::bail!("lfm2 adoption: the cache is not a fresh one of this model")
        }
        match (self.kv.arm_mut(), &saved.cat) {
            (ArmMut::One(OwnKv::Cat(kv)), Some(cat)) if kv.len() == cat.len() => {
                kv.clone_from_slice(cat);
            }
            _ => lloom_tensor::bail!(
                "lfm2 adoption: rows share a retained trunk on the Cat KV arm only \
                 (kv_append = cat on both the trunk and the row)"
            ),
        }
        self.restore_retained_conv(saved)
    }
}

impl<P: Profile> State<P> {
    /// The conv rings, their phase and the per-pass marks [`RetainedState`]
    /// carries, put back after the KV was.
    fn restore_retained_conv(&mut self, saved: &RetainedState) -> Result<()> {
        for (l, t) in saved.conv.iter().enumerate() {
            match t {
                Some(t) => {
                    let dst = match &self.conv_states[l] {
                        Some(dst)
                            if dst.dims() == t.dims()
                                && dst.dtype() == t.dtype()
                                && dst.device().same_device(t.device()) =>
                        {
                            dst.clone()
                        }
                        _ => crate::nn::recurrent::own_copy(t)?,
                    };
                    let _persistent = lloom_tensor::EscapeGuard::new();
                    dst.slice_set(t, 0, 0)?;
                    self.set_conv_state(l, dst)?;
                }
                None => self.conv_states[l] = None,
            }
        }
        self.conv_phase = saved.phase;
        self.conv_phase_prev = saved.phase_prev;
        self.conv_history = saved.history;
        self.conv_compact = saved.compact;
        self.paged_pass_pending = false;
        self.turn_ckpt_cur = None;
        self.turn_pin_cur = None;
        for t in &mut self.taps {
            *t = None;
        }
        #[cfg(feature = "metal")]
        if let Some(b) = self.shared_mut() {
            b.pass = None;
        }
        self.forget_checkpoints_past(saved.kv_len);
        Ok(())
    }
}

#[derive(Default)]
struct VerifyWordsSlot(Option<VerifyWords>);

impl Clone for VerifyWordsSlot {
    fn clone(&self) -> Self {
        VerifyWordsSlot(None)
    }
}

impl std::fmt::Debug for VerifyWordsSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(w) => write!(f, "VerifyWords(cap {}, generation {})", w.cap, w.generation),
            None => f.write_str("VerifyWords(none)"),
        }
    }
}

/// The verify's `[argmax, proposed]` words as a PUBLISH rather than a
/// readback: a CPU-mapped U32 buffer `Device::words_publish` copies the
/// round's words into, then a stamp of the round's generation into `step`,
/// dispatched after the copy and bound to it, so a host that sees the stamp
/// reads the words behind it. The host polls the stamp (`PollMechanism`, the
/// ring's own wait) and reads the words in place into `host` — no wait on
/// the command buffer's completion handler, no `to_cpu`, no `Vec` per round.
/// A poll that outlives `POLL_BUDGET` falls back to `Device::synchronize`
/// and then reads the same words, so the path is never wrong, only slower.
/// The raw addresses are kept as `usize` beside the tensors that own them
/// (the ring's `Mapped` does the same): the buffers are written in place
/// and never reallocated while a verify is pending.
struct VerifyWords {
    out: Tensor,
    step: Tensor,
    out_ptr: usize,
    step_ptr: usize,
    cap: usize,
    generation: u32,
    host: Vec<u32>,
    /// Signalled behind each round's stamp, so the completion parks on it
    /// rather than polling the stamp (`LLOOM_VERIFY_PARK`).
    #[cfg(feature = "metal")]
    event: Option<lloom_tensor::metal_backend::PublishEvent>,
    /// The last value signalled on `event` (monotone; the generation wraps).
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    signalled: u64,
    waits: VerifyWaits,
}

/// A verify's published words: where to poll, what to expect, how many.
#[derive(Clone, Copy)]
struct Published {
    out_ptr: usize,
    step_ptr: usize,
    generation: u32,
    n: usize,
    /// The publish event's value for this round; 0 when none was signalled.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    signal: u64,
}

/// What a cache's verify completions did with the publish event: rounds that
/// signalled it, of those the ones whose stamp was already visible, the ones
/// that parked until it landed, and the ones whose park timed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerifyWaits {
    pub signalled: u64,
    pub already: u64,
    pub parked: u64,
    pub fell_back: u64,
}

impl VerifyWords {
    /// Mapped words for at least `need` u32s on `device`; `None` off Metal or
    /// on a buffer the host cannot map (the caller then reads back).
    fn ensure<'a>(
        slot: &'a mut VerifyWordsSlot,
        need: usize,
        device: &Device,
    ) -> Option<&'a mut VerifyWords> {
        // The mapped-words fast path is a Metal-only optimisation: it reads the
        // stamp and the words in place through pointers into `Shared` buffers
        // (`peek_scalar_ptr`, `flush_and_prune`). Off Metal there is nothing to
        // map, so every caller falls back to a staged readback (`stage_read`),
        // and the whole path is `#[cfg]`-compiled out rather than guarded only
        // at runtime — a no-metal build has no `peek_scalar_ptr` to name.
        #[cfg(not(feature = "metal"))]
        {
            let _ = (slot, need, device);
            None
        }
        #[cfg(feature = "metal")]
        {
            if !device.is_metal() {
                return None;
            }
            let slot = &mut slot.0;
            let fresh = !matches!(slot, Some(w) if w.cap >= need);
            if fresh {
                let cap = need.max(64).next_power_of_two();
                let out = Tensor::from_vec(vec![0u32; cap], cap, device).ok()?;
                let step = Tensor::from_vec(vec![0u32], 1, device).ok()?;
                // SAFETY: read only through `read_volatile`, after the stamp says
                // the GPU wrote the words; the tensors are held beside the addresses.
                let out_ptr = unsafe { out.peek_scalar_ptr::<u32>(0)? } as usize;
                let step_ptr = unsafe { step.peek_scalar_ptr::<u32>(0)? } as usize;
                let generation = slot.as_ref().map_or(0, |w| w.generation);
                let event = match device {
                    Device::Metal(m) if crate::tuning::tuning().ring.verify_park => {
                        lloom_tensor::metal_backend::PublishEvent::new(m).ok()
                    }
                    _ => None,
                };
                *slot = Some(VerifyWords {
                    out,
                    step,
                    out_ptr,
                    step_ptr,
                    cap,
                    generation,
                    host: Vec::with_capacity(cap),
                    event,
                    signalled: 0,
                    waits: slot.as_ref().map_or(VerifyWaits::default(), |w| w.waits),
                });
            }
            slot.as_mut()
        }
    }

    /// Encode the publish of `src`'s first `n` words for this round; the
    /// receipt the completion polls.
    fn publish(&mut self, src: &Tensor, n: usize, device: &Device) -> Result<Published> {
        self.generation = self.generation.wrapping_add(1).max(1);
        device.words_publish(src, n, &self.out, &self.step, self.generation)?;
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut signal = 0;
        #[cfg(feature = "metal")]
        if let Some(ev) = &self.event {
            self.signalled += 1;
            ev.encode_signal(self.signalled)?;
            signal = self.signalled;
        }
        // Committed now, without a wait — what `stage_read` did with its
        // marker: the poll below is on a buffer the GPU has, not on one the
        // host still holds open (that was a `POLL_BUDGET` of spinning on a
        // stamp nothing was going to write). Metal-only: `publish` is only
        // reached from the `Some(w)` arm, which `ensure` yields on Metal alone.
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            m.flush_and_prune()?;
        }
        Ok(Published {
            out_ptr: self.out_ptr,
            step_ptr: self.step_ptr,
            generation: self.generation,
            n,
            signal,
        })
    }

    /// Wait for the round's stamp and read its words into `host`.
    fn complete(&mut self, p: Published, device: &Device) -> Result<&[u32]> {
        let poll = crate::generation::poll::PollMechanism::default();
        let step = p.step_ptr as *const u32;
        let mut spins: u64 = 0;
        let t0 = std::time::Instant::now();
        let mut fell_back = false;
        let first = unsafe { std::ptr::read_volatile(step) };
        // Parked on the event behind the stamp: the round's buffer was
        // committed at publish, so the signal is on its way. A timeout is not
        // completion; the blocking wait below covers it.
        #[cfg(feature = "metal")]
        if p.signal > 0 {
            self.waits.signalled += 1;
            if first == p.generation {
                self.waits.already += 1;
            } else if let Some(ev) = &self.event {
                if ev.wait(p.signal, crate::generation::readback::EOS_PARK_TIMEOUT_MS) {
                    self.waits.parked += 1;
                } else {
                    self.waits.fell_back += 1;
                    device.synchronize()?;
                    fell_back = true;
                }
            }
        }
        loop {
            // SAFETY: `step` is the mapped stamp of a tensor this struct holds.
            let seen = unsafe { std::ptr::read_volatile(step) };
            if seen == p.generation {
                break;
            }
            spins += 1;
            if spins > crate::generation::readback::POLL_BUDGET {
                device.synchronize()?;
                fell_back = true;
                break;
            }
            // SAFETY: as above; `wait_on` returns when the word moves or a
            // timeout elapses, never touches it.
            unsafe { poll.wait_on(step, seen) };
        }
        if crate::tuning::tuning().lfm2.verify_poll_trace {
            eprintln!(
                "[verify poll: gen {} first {} now {} spins {spins} {:.3} ms fallback={fell_back} n={}]",
                p.generation,
                first,
                unsafe { std::ptr::read_volatile(step) },
                t0.elapsed().as_secs_f64() * 1e3,
                p.n
            );
        }
        self.host.clear();
        for i in 0..p.n {
            // SAFETY: `out_ptr` maps `cap >= n` words of `out`, which this
            // struct holds; the stamp is ordered after the words' copy.
            self.host
                .push(unsafe { std::ptr::read_volatile((p.out_ptr as *const u32).add(i)) });
        }
        Ok(&self.host)
    }
}

impl<P: Profile> crate::nn::model::CacheShare for State<P> {
    fn share_prefix_from(&mut self, src: &Self, n_tokens: usize) -> Result<usize> {
        State::<P>::share_prefix_from(self, src, n_tokens)
    }

    fn share_granularity(&self) -> usize {
        self.batch_pool().map(|p| p.page_size()).unwrap_or(0)
    }
}

/// Turn checkpoints a cache keeps: the last passes whose end a request can
/// be put back to. Two chat turns and a serving slot's recent requests fit;
/// the slots are a ring, so the count bounds the memory and the next pass
/// reuses the oldest slot's buffers.
pub const TURN_CHECKPOINTS: usize = 8;

/// One token's KV across every attention layer at B=1, from the geometry
/// (§5.6): `2 × attention layers × kv_heads × head_dim × dtype`, or the
/// code bytes plus a scale and a bias per head vector on a quantized arm.
pub fn kv_bytes_per_token(config: &Config, dtype: DType) -> usize {
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

/// The bytes one sequence's state holds before its KV grows past
/// `kv_floor` positions: the KV slab, the conv state at its width, and the
/// turn-checkpoint ring (`TURN_CHECKPOINTS` frames of every conv layer's
/// tail) — the floors the machine gate reserves (`memory.md` §9.10).
pub fn state_floor_bytes(
    config: &Config,
    dtype: DType,
    kv_floor: usize,
    pin_frame: bool,
) -> [(&'static str, usize); 3] {
    let conv_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::Conv)
        .count();
    let elem = dtype.size_in_bytes();
    let conv =
        conv_layers * config.hidden_size * config.conv_state.width(config.conv_l_cache) * elem;
    let ring = (TURN_CHECKPOINTS + usize::from(pin_frame))
        * conv_layers
        * config.hidden_size
        * (config.conv_l_cache + TURN_CHECKPOINT_SLACK)
        * elem;
    [
        ("KV floor", kv_floor * kv_bytes_per_token(config, dtype)),
        ("conv state", conv),
        ("turn-checkpoint ring", ring),
    ]
}

/// Positions before a checkpointed pass's end the checkpoint can still stand
/// at: its tail keeps `l_cache + TURN_CHECKPOINT_SLACK` input columns, and a
/// position `back` before the end needs the `l_cache` columns ending there.
/// A chat re-render drops one token (the generation prompt's `<think>`), so
/// one would do; a few more cost a column each and admit a template that
/// drops a little more.
pub const TURN_CHECKPOINT_SLACK: usize = 16;

/// The state a multi-token pass left, as far as a restore needs it: the
/// position it ended at and, per conv layer, the newest `tail_len` columns of
/// the input it convolved over (`[b, hidden, l_cache + TURN_CHECKPOINT_SLACK]`,
/// the valid columns at the front). The KV needs nothing kept — a restore is
/// a length decrement there.
#[derive(Debug, Clone, Copy)]
struct TurnCheckpoint {
    pos: usize,
    tail_len: usize,
    valid: bool,
}

/// `LLOOM_LFM2_LOGITS_DUMP=<path>`: append every forward's logits to `path` —
/// a record per forward of `pos: u32, rows: u32, vocab: u32` then `rows ×
/// vocab` little-endian f32 — so two decode paths' logits at one position can
/// be compared (issue #894 item 3: the margin at the first divergence). A
/// readback per forward; an instrument, nothing unless the variable is set.
fn dump_logits(logits: &Tensor, pos: usize) -> Result<()> {
    use std::io::Write;
    use std::sync::OnceLock;
    static SINK: OnceLock<Option<parking_lot::Mutex<std::fs::File>>> = OnceLock::new();
    let sink = SINK.get_or_init(|| {
        let path = crate::tuning::tuning().lfm2.logits_dump.clone()?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(parking_lot::Mutex::new(f))
    });
    let Some(sink) = sink else {
        return Ok(());
    };
    let l = logits.to_dtype(DType::F32)?;
    let (rows, vocab) = match l.dims() {
        [_, v] => (1usize, *v),
        [_, r, v] => (*r, *v),
        d => lloom_tensor::bail!("dump_logits: unexpected logits shape {d:?}"),
    };
    let vals = l.flatten_all()?.to_vec1::<f32>()?;
    let mut buf = Vec::with_capacity(12 + vals.len() * 4);
    buf.extend_from_slice(&(pos as u32).to_le_bytes());
    buf.extend_from_slice(&(rows as u32).to_le_bytes());
    buf.extend_from_slice(&(vocab as u32).to_le_bytes());
    for v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = sink.lock();
    f.write_all(&buf)
        .map_err(|e| lloom_tensor::Error::msg(format!("dump_logits: {e}")))?;
    Ok(())
}

impl<P: Profile> CacheReset for State<P> {
    /// Delegates to [`Cache::clear`] — the tested total reset (kvs, kv/quant
    /// slots, conv states, conv phase and compaction flag). This is what lets
    /// `Session::warmup` run a real forward for its compilation side effect and
    /// leave the session pre-prefill, so the first real prompt is unaffected.
    fn reset(&mut self) {
        self.clear();
    }
}

impl<P: Profile> State<P> {
    /// What this cache's verify completions did with the publish event
    /// (`VerifyWaits`); all zero before the first verify, or off Metal.
    pub fn verify_waits(&self) -> VerifyWaits {
        self.verify_words
            .0
            .as_ref()
            .map_or(VerifyWaits::default(), |w| w.waits)
    }
}

impl<P: Profile> State<P> {
    /// The sequence's own store, on the `One` shapes.
    #[inline]
    fn own(&self) -> Option<&OwnKv> {
        match self.kv.arm() {
            Arm::One(own) => Some(own),
            Arm::Paged(_) => None,
        }
    }

    #[inline]
    fn own_mut(&mut self) -> Option<&mut OwnKv> {
        match self.kv.arm_mut() {
            ArmMut::One(own) => Some(own),
            ArmMut::Paged(_) => None,
        }
    }

    /// The sequence's own page pool, on the `Paged` shapes.
    #[cfg(feature = "metal")]
    #[inline]
    fn paged(&self) -> Option<&PagedKv> {
        match self.kv.arm() {
            Arm::Paged(PooledKv::Own(p)) => Some(p),
            _ => None,
        }
    }

    /// The row's slot in a shared pool, on the `Paged` shapes.
    #[cfg(feature = "metal")]
    #[inline]
    fn shared(&self) -> Option<&BatchSlot> {
        match self.kv.arm() {
            Arm::Paged(PooledKv::Shared(b)) => Some(b),
            _ => None,
        }
    }

    #[cfg(feature = "metal")]
    #[inline]
    fn shared_mut(&mut self) -> Option<&mut BatchSlot> {
        match self.kv.arm_mut() {
            ArmMut::Paged(PooledKv::Shared(b)) => Some(b),
            _ => None,
        }
    }

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
        Self::with_kv(
            use_kv_cache,
            dtype,
            config,
            device,
            kv_capacity,
            |slab, num_layers| {
                <P::Rows as crate::profile::Rows>::pick::<KvFork, lloom_tensor::Error>(
                    || OwnKv::new(config, slab, num_layers),
                    || PooledKv::new(config, device, kv_capacity, slab, num_layers, use_kv_cache),
                )
            },
        )
    }

    /// [`Self::new_with`] with the KV arm built by `kv(slab, num_layers)`:
    /// the shape's own store, or a row of a shared pool ([`Self::for_batch`]).
    fn with_kv(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
        kv_capacity: usize,
        kv: impl FnOnce(usize, usize) -> Result<KvOf<P>>,
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
        let kv_bytes_per_token = kv_bytes_per_token(config, dtype);
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

        let (cos, sin) = crate::nn::rotary_emb::default_rope_tables(
            config.head_dim(),
            config.rope_theta,
            config.max_position_embeddings,
            dtype,
            device,
        )?;

        let num_layers = config.num_hidden_layers;
        let kv = kv(slab, num_layers)?;
        #[cfg(feature = "metal")]
        let ledger = match device {
            Device::Metal(m) if use_kv_cache => {
                use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
                let gauge = Gauge::new(
                    "lfm2-kv",
                    Claim {
                        floor: slab * kv_bytes_per_token,
                        want: kv_capacity * kv_bytes_per_token,
                        unit: DEFAULT_KV_BUCKET.min(kv_capacity) * kv_bytes_per_token,
                        class: Class::Capability,
                    },
                    0,
                );
                let tenant: Arc<dyn Tenant> = gauge.clone();
                m.ledger().register(&tenant);
                Some(gauge)
            }
            _ => None,
        };
        Ok(Self {
            profile: PhantomData,
            masks: HashMap::new(),
            use_kv_cache,
            branch_continuation: false,
            kv,
            paged_pass_pending: false,
            kv_append: config.kv_append,
            kv_quant: config.kv_quant,
            kv_capacity,
            kv_bucket: DEFAULT_KV_BUCKET,
            #[cfg(feature = "metal")]
            ledger,
            kv_slab: slab,
            kv_grows: 0,
            kv_grow_fraction,
            kv_bytes_per_token,
            conv_states: vec![None; num_layers],
            conv_spare: vec![None; num_layers],
            turn_checkpoints: Vec::with_capacity(TURN_CHECKPOINTS),
            turn_ring: None,
            layer_tier: None,
            turn_ckpt_cur: None,
            turn_ckpt_next: 0,
            turn_pin_frame: false,
            turn_pinned: None,
            turn_pin_want: None,
            turn_request: false,
            turn_pass: (0, 0),
            turn_pin_cur: None,
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
            verify_words: VerifyWordsSlot(None),
            tap_layers: config.tap_layers.clone(),
            tap_dtype: config.tap_dtype,
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// The row's slot in its [`BatchPool`], if it has one.
    pub fn batch_slot(&self) -> Option<usize> {
        #[cfg(feature = "metal")]
        {
            self.shared().map(|b| b.slot())
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    }

    /// The pool the row belongs to, if any.
    pub fn batch_pool(&self) -> Option<&BatchPool> {
        #[cfg(feature = "metal")]
        {
            self.shared().map(|b| &b.pool)
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
    }

    /// Whether `other` is a row of the same pool.
    #[cfg(feature = "metal")]
    fn shares_pool_with(&self, other: &Self) -> bool {
        match (self.shared(), other.shared()) {
            (Some(a), Some(b)) => a.pool.same(&b.pool),
            _ => false,
        }
    }

    /// Layer `layer`'s conv state for this row: on the batch arm the row's
    /// ring of the pool's `[slots, hidden, width]` — bound on first use and
    /// zeroed — and elsewhere whatever the cache holds. `None` off the
    /// batch arm when nothing has been written.
    #[allow(unused_variables)]
    fn conv_state_view(
        &mut self,
        layer: usize,
        hidden: usize,
        width: usize,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        if let Some(t) = &self.conv_states[layer] {
            return Ok(Some(t.clone()));
        }
        #[cfg(feature = "metal")]
        if let Some(b) = self.shared() {
            let slot = b.slot();
            let view = {
                let mut g = b.pool.inner.lock();
                let slots = g.slots;
                if g.conv[layer].is_none() {
                    g.conv[layer] =
                        Some(Tensor::zeros((slots, hidden, width), dtype, &self.device)?);
                }
                g.conv[layer]
                    .as_ref()
                    .expect("allocated above")
                    .narrow(0, slot, 1)?
            };
            if view.dtype() != dtype || view.dims() != [1, hidden, width] {
                lloom_tensor::bail!(
                    "lfm2 batch conv pool: layer {layer} is {:?} {:?}, this row needs [1, {hidden}, \
                     {width}] {dtype:?}",
                    view.dims(),
                    view.dtype()
                )
            }
            // A fresh binding is a fresh ring: the slot's previous owner left
            // its window here.
            view.zero_set()?;
            self.conv_states[layer] = Some(view.clone());
            return Ok(Some(view));
        }
        Ok(None)
    }

    /// On the batch arm, at a pass's start: record the row's length as the
    /// pass's start position and, when that length is a positive multiple
    /// of the page size and the page has no snapshot yet, snapshot every
    /// conv ring (after the page's last token) under the page, for a sharer
    /// (§12.3). A no-op otherwise.
    #[cfg(feature = "metal")]
    fn snapshot_page_boundary(&mut self) -> Result<()> {
        let Some(b) = self.shared() else {
            return Ok(());
        };
        let slot = b.slot();
        let pool = b.pool.clone();
        let mut g = pool.inner.lock();
        let p = g.kv.page_size;
        let len = g.kv.row_len(slot);
        if let Some(b) = self.shared_mut() {
            b.pass_start = len;
        }
        if len == 0 || len % p != 0 {
            return Ok(());
        }
        let pages = g.kv.row_pages(slot);
        let Some(&page) = pages.get(len / p - 1) else {
            return Ok(());
        };
        if g.snapshots.contains_key(&page) {
            // Taken inside the piece that completed the page.
            return Ok(());
        }
        let mut rings = Vec::with_capacity(g.conv.len());
        for t in g.conv.iter() {
            rings.push(match t {
                Some(pool_t) => {
                    let view = pool_t.narrow(0, slot, 1)?;
                    // A real copy (`Tensor::copy` is shallow on Metal).
                    let own = Tensor::zeros(view.dims(), view.dtype(), view.device())?;
                    own.slice_set(&view, 2, 0)?;
                    Some(own)
                }
                None => None,
            });
        }
        g.snapshots.insert(
            page,
            RingSnapshot {
                rings,
                phase: self.conv_phase,
            },
        );
        Ok(())
    }

    /// On the batch arm, inside a multi-token conv pass: for every page
    /// boundary the piece crosses, the ring after that page's last token —
    /// the last `l_cache` columns of `conv_in` (the left context and the
    /// piece's `bx`) ending at the boundary, at the front of a `width`-wide
    /// ring, the rest zero: byte for byte what the writer stores when a
    /// piece ENDS at that token — staged under the token's position for
    /// layer `layer`. Keyed by page at the pass's end
    /// (`flush_ring_snapshots`), when the pass's pages exist.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn stage_piece_snapshots(
        &mut self,
        layer: usize,
        conv_in: &Tensor,
        ctx_len: usize,
        seq_len: usize,
        l_cache: usize,
        width: usize,
        n_layers: usize,
    ) -> Result<()> {
        let Some(b) = self.shared() else {
            return Ok(());
        };
        let (start, p) = (b.pass_start, b.page_size);
        if p == 0 || l_cache == 0 {
            return Ok(());
        }
        let (_, hidden, _) = conv_in.dims3()?;
        for i in 0..seq_len {
            let t = start + i;
            if (t + 1) % p != 0 {
                continue;
            }
            let ci = ctx_len + i;
            let take = (ci + 1).min(l_cache);
            let cols = conv_in.narrow(2, ci + 1 - take, take)?;
            let ring = Tensor::zeros((1, hidden, width), conv_in.dtype(), conv_in.device())?;
            ring.slice_set(&cols.contiguous()?, 2, l_cache - take)?;
            let b = self.shared_mut().expect("checked above");
            let entry = match b.staged_snaps.iter_mut().find(|(pos, _)| *pos == t) {
                Some(e) => e,
                None => {
                    b.staged_snaps.push((t, vec![None; n_layers]));
                    b.staged_snaps.last_mut().expect("pushed")
                }
            };
            if entry.1.len() < n_layers {
                entry.1.resize(n_layers, None);
            }
            entry.1[layer] = Some(ring);
        }
        Ok(())
    }

    /// Key the rings staged by `stage_piece_snapshots` by their pages, once
    /// the pass that wrote them has appended its tokens (so the pages exist
    /// in the row's table). A no-op when nothing is staged.
    #[cfg(feature = "metal")]
    fn flush_ring_snapshots(&mut self) {
        let Some(b) = self.shared_mut() else {
            return;
        };
        if b.staged_snaps.is_empty() {
            return;
        }
        let staged = std::mem::take(&mut b.staged_snaps);
        let slot = b.slot();
        let pool = b.pool.clone();
        let phase = self.l_cache.saturating_sub(1);
        let mut g = pool.inner.lock();
        let p = g.kv.page_size;
        let pages = g.kv.row_pages(slot);
        let len = g.kv.row_len(slot);
        for (t, rings) in staged {
            // A boundary past the row's live length (the pass was rolled
            // back below it) names no page.
            if t >= len {
                continue;
            }
            let Some(&page) = pages.get(t / p) else {
                continue;
            };
            g.snapshots.insert(page, RingSnapshot { rings, phase });
        }
    }

    /// Share `src`'s first whole pages covering at most `n_tokens` tokens
    /// with this (empty) row — the pages by reference, the conv rings from
    /// the snapshot after the last shared page — and return the tokens
    /// shared (a multiple of the page size, 0 when nothing could be:
    /// `batched-decode-design.md` §12.3). The caller then feeds the tokens
    /// after them at that position, with the conv's left context restored.
    #[allow(unused_variables)]
    pub fn share_prefix_from(&mut self, src: &State<P>, n_tokens: usize) -> Result<usize> {
        #[cfg(not(feature = "metal"))]
        {
            Ok(0)
        }
        #[cfg(feature = "metal")]
        {
            let (Some(d), Some(s)) = (self.shared(), src.shared()) else {
                lloom_tensor::bail!(
                    "lfm2 share_prefix_from: both rows must be slots of a BatchPool"
                )
            };
            if !d.pool.same(&s.pool) {
                lloom_tensor::bail!("lfm2 share_prefix_from: the rows are in different pools")
            }
            let (dslot, sslot) = (d.slot(), s.slot());
            if dslot == sslot {
                lloom_tensor::bail!("lfm2 share_prefix_from: a row cannot share with itself")
            }
            if self.kv_len() != 0 {
                lloom_tensor::bail!(
                    "lfm2 share_prefix_from: the sharing row holds {} tokens; a share starts \
                     from an empty row",
                    self.kv_len()
                )
            }
            let pool = d.pool.clone();
            let snap = {
                let mut g = pool.inner.lock();
                let p = g.kv.page_size;
                let src_len = g.kv.row_len(sslot);
                let pages = g.kv.row_pages(sslot);
                let asked = (n_tokens / p).min(src_len / p).min(pages.len());
                let mut n_pages = asked;
                // Only a page with its ring snapshot can be the last shared
                // one: walk down until one has it.
                while n_pages > 0 && !g.snapshots.contains_key(&pages[n_pages - 1]) {
                    n_pages -= 1;
                }
                if crate::tuning::tuning().lfm2.share_trace {
                    let have: Vec<usize> = (0..pages.len())
                        .filter(|&i| g.snapshots.contains_key(&pages[i]))
                        .collect();
                    eprintln!(
                        "[share: slot {sslot} -> {dslot}: asked {n_tokens} tokens = {asked} pages of \
                         src_len {src_len} ({} pages), snapshots at pages {have:?}, sharing {n_pages}]",
                        pages.len()
                    );
                }
                if n_pages == 0 {
                    return Ok(0);
                }
                g.kv.share_pages(sslot, dslot, n_pages)?;
                let snap = g.snapshots[&pages[n_pages - 1]].clone();
                (snap, n_pages * p)
            };
            let (snap, shared) = snap;
            for (layer, ring) in snap.rings.iter().enumerate() {
                let Some(r) = ring else { continue };
                let (_, hidden, width) = r.dims3()?;
                let view = self
                    .conv_state_view(layer, hidden, width, r.dtype())?
                    .expect("a batch row's conv view exists");
                view.slice_set(r, 2, 0)?;
            }
            self.conv_phase = snap.phase;
            self.conv_phase_prev = snap.phase;
            self.conv_history = 0;
            self.conv_compact = false;
            self.paged_pass_pending = false;
            Ok(shared)
        }
    }

    /// Layer `layer`'s spare `Shuffle` state, shaped as `state`: the buffer
    /// the fused step writes the next state into. Allocated on the layer's
    /// first fused step under an `EscapeGuard` (it outlives every pass, so no
    /// plan serves it), and again only if the state's shape or dtype moved.
    fn conv_spare_for(&mut self, layer: usize, state: &Tensor) -> Result<Tensor> {
        #[cfg(test)]
        tests::SPARES_TAKEN.with(|n| n.set(n.get() + 1));
        if let Some(s) = self.conv_spare[layer].take() {
            if s.dims() == state.dims() && s.dtype() == state.dtype() && s.is_contiguous() {
                return Ok(s);
            }
        }
        #[cfg(feature = "metal")]
        let _declared = lloom_tensor::EscapeGuard::new();
        Tensor::zeros(state.dims(), state.dtype(), state.device())
    }

    /// Store layer `layer`'s conv state: on the batch arm into the row's
    /// ring of the pool (the pool row IS the state every kernel reads, so a
    /// state produced elsewhere is copied in), else as the cache's tensor.
    fn set_conv_state(&mut self, layer: usize, state: Tensor) -> Result<()> {
        #[cfg(feature = "metal")]
        if self.shared().is_some() {
            let (_, hidden, width) = state.dims3()?;
            let view = self
                .conv_state_view(layer, hidden, width, state.dtype())?
                .expect("a batch row's conv view exists");
            if view.id() != state.id() {
                if state.dims() != view.dims() {
                    lloom_tensor::bail!(
                        "lfm2 batch conv pool: a {:?} state for a {:?} ring",
                        state.dims(),
                        view.dims()
                    )
                }
                view.slice_set(&state.contiguous()?, 2, 0)?;
            }
            return Ok(());
        }
        self.conv_states[layer] = Some(state);
        Ok(())
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
        let allocated = self.own().is_some_and(OwnKv::is_allocated);
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
        match self.kv.arm() {
            Arm::One(own) => own.len(),
            Arm::Paged(p) => p.len(),
        }
    }

    /// The KV slab's state, for a run's memory report.
    pub fn kv_report(&self) -> KvReport {
        let bytes = match self.kv.arm() {
            Arm::One(own) => own.bytes(),
            Arm::Paged(p) => p.bytes(),
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
    /// The model's tier, for the admission below (once; cheap after).
    pub(crate) fn attach_layer_tier(
        &mut self,
        tier: &std::sync::Arc<parking_lot::Mutex<crate::nn::weight_groups::WeightGroups>>,
    ) {
        if self.layer_tier.is_none() {
            self.layer_tier = Some(std::sync::Arc::clone(tier));
        }
    }

    /// Apply the arbiter's grant to the tier at this pass boundary, if it
    /// moved (A6b: the dense tier yields as the expert tier does).
    pub(crate) fn apply_layer_grant(&self) -> Result<Option<usize>> {
        match &self.layer_tier {
            Some(t) => t.lock().apply_grant(),
            None => Ok(None),
        }
    }

    pub(crate) fn prepare_pass(&mut self, seq_len: usize) -> Result<()> {
        self.apply_layer_grant()?;
        if let Some(plan) = self.plan_pass(seq_len)? {
            self.admit_grow(plan.bytes, &plan)?;
            self.apply_grow(&plan)?;
        }
        // The planes allocate at the first append, at the slab's width: the
        // ledger hears that here, once per sequence, and at every grow after.
        if seq_len > 0 && self.kv_len() == 0 {
            self.ledger_set_held(self.kv_slab * self.kv_bytes_per_token);
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
        #[cfg(feature = "metal")]
        if let Some(b) = self.shared() {
            let g = b.pool.inner.lock();
            if need <= g.kv.slab {
                return Ok(None);
            }
            let slab = (need.div_ceil(self.kv_bucket) * self.kv_bucket).min(self.kv_capacity);
            return Ok(Some(GrowPlan {
                from: g.kv.slab,
                slab,
                bytes: g.kv.grow_bytes_to(slab, self.kv_bytes_per_token),
                kv_len,
                seq_len,
                shared: true,
            }));
        }
        let mut slab = if need <= self.kv_slab {
            self.kv_slab
        } else {
            (need.div_ceil(self.kv_bucket) * self.kv_bucket).min(self.kv_capacity)
        };
        // The paged pool is sized by the pages a pass can take, not by
        // positions alone: pages a quarantine still holds are neither the
        // sequence's nor free (`PagedKv::pages_short`). The headroom may take
        // the pool past the ceiling; the ceiling check above bounds positions.
        #[cfg(feature = "metal")]
        if let Some(p) = self.paged() {
            let short = p.pages_short(seq_len);
            if short > 0 {
                let wide = self.kv_slab + short * p.page_size;
                slab = slab.max(wide.div_ceil(self.kv_bucket) * self.kv_bucket);
            }
        }
        if slab <= self.kv_slab {
            return Ok(None);
        }
        Ok(Some(GrowPlan {
            from: self.kv_slab,
            slab,
            bytes: self.grow_bytes(slab),
            kv_len,
            seq_len,
            shared: false,
        }))
    }

    /// The second half: widen every plane to the plan's slab. Admission is the
    /// caller's — `prepare_pass` admits one plan, `forward_rows` the sum of its
    /// rows' — so this allocates and copies and nothing else. The copy
    /// carries the rows the sequence holds, so under an armed EOS gate it
    /// runs ungated: a grow can fall inside the forwards queued past an EOS
    /// (`sequence-reuse-design.md` §2.3e).
    pub(crate) fn apply_grow(&mut self, plan: &GrowPlan) -> Result<()> {
        let _ungated = self.device.eos_gate_suspended();
        let slab = plan.slab;
        match self.kv.arm_mut() {
            ArmMut::One(own) => own.grow_to(slab)?,
            ArmMut::Paged(p) => p.grow_to(slab)?,
        }
        self.kv_slab = slab;
        self.kv_grows += 1;
        self.ledger_note();
        Ok(())
    }

    /// Set every unallocated plane's width and the slab figure to `slab`. A
    /// paged pool that already exists keeps its width, and the figure follows
    /// it.
    fn set_slab(&mut self, slab: usize) {
        let slab = match self.kv.arm_mut() {
            ArmMut::One(own) => {
                own.set_capacity(slab);
                slab
            }
            ArmMut::Paged(p) => p.set_slab(slab),
        };
        self.kv_slab = slab;
        self.ledger_note();
    }

    /// What the ledger reads as held: the planes' bytes once they exist, or
    /// the width the first pass will allocate them at (`prepare_pass`).
    fn ledger_set_held(&self, bytes: usize) {
        #[cfg(feature = "metal")]
        if let Some(g) = &self.ledger {
            g.set_held(bytes);
        }
        #[cfg(not(feature = "metal"))]
        let _ = bytes;
    }

    /// Tell the ledger what this cache holds and needs now: the planes' bytes
    /// as held, the slab's as the floor. At a grow and a clear, never per token.
    fn ledger_note(&self) {
        #[cfg(feature = "metal")]
        if let Some(g) = &self.ledger {
            use lloom_tensor::kernels::metal::ledger::{Claim, Class};
            g.set_held(self.kv_report().bytes);
            g.set_claim(Claim {
                floor: self.kv_slab * self.kv_bytes_per_token,
                want: self.kv_capacity * self.kv_bytes_per_token,
                unit: self.kv_bucket.min(self.kv_capacity) * self.kv_bytes_per_token,
                class: Class::Capability,
            });
        }
    }

    /// Bytes widening the slab to `slab` tokens will allocate, read from an
    /// allocated plane's shape when one exists (its batch is the truth) and
    /// predicted from the geometry at B=1 before the first append.
    fn grow_bytes(&self, slab: usize) -> usize {
        let extra = slab.saturating_sub(self.kv_slab);
        match self.kv.arm() {
            Arm::One(own) => {
                let per_token = own.bytes_per_token();
                if per_token == 0 {
                    self.kv_bytes_per_token * extra
                } else {
                    per_token * extra
                }
            }
            Arm::Paged(p) => p.grow_bytes(self.kv_slab, slab, self.kv_bytes_per_token),
        }
    }

    /// The memory half of `prepare_pass`'s refusal, against the device.
    ///
    /// `currentAllocatedSize` is the whole process's Metal allocation — the
    /// weights, the pool's retained free buffers, every intermediate in flight
    /// — which is exactly the figure admission's prediction cannot see and the
    /// one a grow has to fit beside. The comparison is [`kv_grow_fits`], kept
    /// pure so the arithmetic is tested without a device.
    ///
    /// A refusal is `Error::Allocation(Refused { .. })`: the machine gate's
    /// when the machine refused, else the budget's, with the grow and the
    /// room. Its line (what, and the arithmetic) goes to stderr once.
    #[cfg(feature = "metal")]
    pub(crate) fn admit_grow(&self, grow: usize, what: &dyn std::fmt::Display) -> Result<()> {
        let Ok(metal) = self.device.as_metal_device() else {
            return Ok(());
        };
        let dev = metal.device();
        let working_set = dev.recommended_max_working_set_size();
        // The free lists are given back before a refusal stands
        // (`MetalDevice::admit_bytes`); the arithmetic is `kv_grow_fits`'.
        let mut a = metal.admit_bytes(grow, kv_grow_budget(working_set, self.kv_grow_fraction));
        if a.needs_yield_retry() {
            // The tenants pledged a yield (the layers' slots): apply it —
            // the tier resized to its grant — and ask again on the figure
            // after (the expert tier's shape, `prepare_pass_as`).
            self.apply_layer_grant()?;
            let (yielded, reclaimed) = (a.yield_report(), a.reclaimed);
            a = metal.admit_bytes(grow, kv_grow_budget(working_set, self.kv_grow_fraction));
            a.merge_yield_report(yielded)?;
            a.reclaimed += reclaimed;
        }
        if a.fits {
            debug_assert!(kv_grow_fits(
                a.committed,
                working_set,
                self.kv_grow_fraction,
                grow
            ));
            return Ok(());
        }
        let budget = kv_grow_budget(working_set, self.kv_grow_fraction);
        let refusal =
            a.machine
                .unwrap_or(lloom_tensor::memory_admission::AllocationError::Refused {
                    requested: grow,
                    available: budget.saturating_sub(a.committed),
                    domain: lloom_tensor::memory_admission::RefusalDomain::Budget,
                });
        let gb = |b: usize| b as f64 / 1e9;
        metal.report_refusal(
            &format_args!(
                "lfm2 KV grow refused {what}: it needs {:.3} GB more, and the ledger holds \
                 {:.2} GB of the {:.2} GB it may ({:.2} x recommendedMaxWorkingSetSize {:.2} GB) \
                 after giving back {:.3} GB of free lists. The reply stops here, before anything \
                 is written (DESIGN.md 9.5, 3.5): raise the fraction (`--kv-grow-fraction`), \
                 free memory, or bound the run. pledged={} yield_contended={} (at most one \
                 yield retry)",
                gb(grow),
                gb(a.committed),
                gb(budget),
                self.kv_grow_fraction,
                gb(working_set),
                gb(a.reclaimed),
                a.pledged,
                a.yield_contended,
            ),
            &refusal,
        );
        Err(refusal.into())
    }

    /// No Metal, no `recommendedMaxWorkingSetSize`, nothing to check against —
    /// `admit_memory_budget`'s stance (§9.5c).
    #[cfg(not(feature = "metal"))]
    pub(crate) fn admit_grow(&self, _grow: usize, _what: &dyn std::fmt::Display) -> Result<()> {
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

    /// What a prefill pass's attention allocates, by the arms `config`
    /// selects: the composed attention on the reference arm; on every other
    /// arm the fused kernel, which reads an in-place cache where it lies and
    /// every other store through a copy of its keys (the page pool's gather,
    /// `Cat`'s concatenation, the quantized store's dequantization).
    #[cfg(feature = "metal")]
    fn prefill_attention(
        config: &Config,
    ) -> lloom_tensor::metal_backend::admission::PrefillAttention {
        use lloom_tensor::metal_backend::admission::PrefillAttention;
        match config.attn_impl {
            AttnImpl::Generic => PrefillAttention::Composed,
            _ if config.kv_append == KvAppend::InPlace && !config.kv_quant.is_on() => {
                PrefillAttention::Fused
            }
            _ => PrefillAttention::FusedGathered,
        }
    }

    /// §9.5d's admission arithmetic for this configuration, as a pure
    /// function of `config`, `budget` and the KV ceiling — the figure
    /// `admit_memory_budget` refuses on, exposed so a caller deciding
    /// **before** construction (§4.8's weight streaming, #892 A6a: how many
    /// MLP slots the quota leaves once the resident rest and its predicted
    /// peak are in) sizes from the same arithmetic the KV cache will be
    /// admitted on, rather than from a second figure that can disagree with
    /// it. The RoPE tables are not in it: `rope_table_bytes` is their cost,
    /// taken from the residual (see below).
    #[cfg(feature = "metal")]
    pub fn admission_budget(
        config: &Config,
        budget: &MemoryBudget,
        kv_capacity: usize,
    ) -> lloom_tensor::metal_backend::admission::Budget {
        use lloom_tensor::metal_backend::admission;

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
        b.activation_rows = budget.activation_rows;
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
            kv_heads: config.num_key_value_heads,
            vocab: config.vocab_size,
            conv_l_cache: config.conv_l_cache,
            attention: Self::prefill_attention(config),
        };
        // §10.2g: `Shuffle` and `RotatingRing` hold exactly `l_cache` columns
        // (264 KiB); `SlidingRing` holds the slack it slides through as well,
        // which is 1.63 MiB at the default. Taken from `ConvState::width`
        // rather than recomputed, so the budget cannot disagree with the
        // allocation about how wide the state is -- the hand-sync §8.1b exists
        // to remove.
        //
        // Both per-sequence classes from THIS config's geometry: the
        // admission constants are LFM2-2.6B's (22 conv layers × 2048, 8
        // attention layers × 8 KV heads × 64), which a model of another size
        // would otherwise inherit. At the 2.6B's geometry these are the
        // constants exactly (`admission_budget_reads_the_configs_geometry`).
        // 2 bytes an element: the f16 store the constants assume (an int8 or
        // int4 KV arm is over-predicted, never under).
        let (conv_layers, attention_layers) = (
            b.prefill_geometry.conv_layers,
            b.prefill_geometry.attention_layers,
        );
        b.conv_state_bytes =
            conv_layers * config.hidden_size * config.conv_state.width(config.conv_l_cache) * 2;
        b.kv_bytes_per_token =
            2 * config.num_key_value_heads * config.head_dim() * 2 * attention_layers;
        b
    }

    /// The RoPE `cos`/`sin` tables' cost — resident plus the transient peak
    /// of building them — which admission takes from the residual rather
    /// than from the five classes (§9.5k). Pure in `config`.
    #[cfg(feature = "metal")]
    pub fn rope_table_bytes(config: &Config) -> usize {
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
        rope_resident + rope_transient
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

        let b = Self::admission_budget(config, &budget, kv_capacity);
        let rope = Self::rope_table_bytes(config);

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
            // Every distinct `(seq_len, kv_len)` is a new `[seq_len, kv_len]`
            // u8 tensor, and a prefill fed in tiles onto a growing cache names
            // a new pair per tile — unbounded, the entries sum to
            // `seq_len · kv_len² / (2 · tile)` bytes over a prompt (8.6 GB for
            // a 131k prompt in 512-token tiles). A mask is reused only across
            // passes at the same shape, so the map is bounded here and dropped
            // with the sequence state (`clear`).
            if self.masks.len() >= MASK_CACHE_ENTRIES {
                self.masks.clear();
            }
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
        self.masks.clear();
        match self.kv.arm_mut() {
            ArmMut::One(own) => own.clear(),
            ArmMut::Paged(p) => p.clear(),
        }
        // A new sequence starts at one bucket again: the slab was dropped with
        // the buffers, and the next one is sized by what the next sequence
        // needs. The paged pool is not dropped and keeps its width.
        self.set_slab(self.kv_bucket.min(self.kv_capacity));
        self.paged_pass_pending = false;
        // On the batch arm the ring is a pool row: `None` here, and the
        // next pass binds the row again and zeroes it (`conv_state_view`).
        self.conv_states.iter_mut().for_each(|v| *v = None);
        self.conv_phase = 0;
        self.conv_phase_prev = 0;
        self.conv_history = 0;
        self.conv_compact = false;
        self.speculative = false;
        self.taps.iter_mut().for_each(|tap| *tap = None);
        // The checkpoints describe the sequence that was just discarded; the
        // slots' buffers stay as scratch for the next one.
        self.turn_checkpoints
            .iter_mut()
            .for_each(|c| c.valid = false);
        self.turn_ckpt_cur = None;
        self.turn_pinned = None;
        self.turn_pin_want = None;
        self.turn_pin_cur = None;
        self.turn_request = false;
    }

    /// The page pool's live tokens, pages in use and pages held in quarantine,
    /// on the `Paged` arm. `None` on the other arms.
    #[cfg(feature = "metal")]
    pub fn paged_stats(&self) -> Option<(usize, usize, usize)> {
        self.paged().map(|p| {
            (
                p.len,
                p.pool
                    .as_ref()
                    .map(|pool| pool.pages().pages_in_use())
                    .unwrap_or(0),
                p.pool.as_ref().map(|pool| pool.held()).unwrap_or(0),
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
        self.rewind_kv(n)?;
        let width = self.conv_state.width(self.l_cache);
        self.conv_phase = (self.conv_phase + width - n) % width;
        self.conv_history -= n;
        self.forget_checkpoints_past(self.kv_len());
        Ok(())
    }

    /// The KV half of a rewind: every plane's length moved back by `n`
    /// (`DESIGN.md` §10.2a: discarded rows are unreachable above the length
    /// and are never cleared). The caller has decided the conv half.
    fn rewind_kv(&mut self, n: usize) -> Result<()> {
        match self.kv.arm_mut() {
            ArmMut::One(own) => own.rewind(n),
            ArmMut::Paged(p) => p.rewind(n),
        }
    }

    /// A multi-token pass of `seq_len` positions is beginning: take a
    /// checkpoint slot for it, so its conv layers' writers can leave the tail
    /// of their input there ([`Cache::turn_checkpoint_tail`]). A pass that
    /// cannot be returned to takes none: a decode step, a speculative verify
    /// (`resolve` owns its rollback), a `Cat` KV (no length to move back).
    fn turn_checkpoint_begin(&mut self, seq_len: usize) {
        self.turn_ckpt_cur = None;
        self.turn_pin_cur = None;
        if !self.use_kv_cache || seq_len < 2 || self.speculative || self.kv_append == KvAppend::Cat
        {
            return;
        }
        let start = self.kv_len();
        self.turn_pass = (start, seq_len);
        // The second target, inside the pass: its tail is the input columns
        // before it, so a position at the pass's start is not one.
        self.turn_pin_cur = self
            .turn_pin_want
            .filter(|&w| self.turn_pin_frame && w > start && w <= start + seq_len);
        let pos = start + seq_len;
        let slot = self.turn_ckpt_next % TURN_CHECKPOINTS;
        if slot < self.turn_checkpoints.len() {
            let c = &mut self.turn_checkpoints[slot];
            c.pos = pos;
            c.tail_len = 0;
            c.valid = false;
        } else {
            self.turn_checkpoints.push(TurnCheckpoint {
                pos,
                tail_len: 0,
                valid: false,
            });
        }
        self.turn_ckpt_cur = Some(slot);
        self.turn_ckpt_next = (slot + 1) % TURN_CHECKPOINTS;
    }

    /// Conv layer `layer` of the pass in flight convolves over `conv_in`
    /// (`[b, hidden, ctx + seq]`, the carried left context first): keep its
    /// newest `l_cache + TURN_CHECKPOINT_SLACK` columns in the pass's
    /// checkpoint slot, written into the slot's own buffer.
    fn turn_checkpoint_tail(
        &mut self,
        layer: usize,
        conv_in: &Tensor,
        l_cache: usize,
    ) -> Result<()> {
        let Some(slot) = self.turn_ckpt_cur else {
            return Ok(());
        };
        let (b_sz, hidden, src_len) = conv_in.dims3()?;
        let want = l_cache + TURN_CHECKPOINT_SLACK;
        let keep = src_len.min(want);
        let src = conv_in.narrow(2, src_len - keep, keep)?;
        let layers = self.conv_states.len();
        let frame = (layers, b_sz, hidden, want);
        // The whole ring at a sequence's first checkpointing pass — one
        // request, session state, outside every plan — and kept for the next
        // sequence of the same geometry.
        let frames = self.turn_frames();
        let fresh = match &self.turn_ring {
            Some(r) => !r.fits(frames, frame, conv_in.dtype()),
            None => true,
        };
        if fresh {
            self.turn_ring = Some(crate::nn::ring::Ring::allocate(
                frames,
                frame,
                conv_in.dtype(),
                conv_in.device(),
                zs::LFM2_CONV_STATE,
                "lfm2-turn-ring",
            )?);
        }
        // The strided source is written where it lies (`slice_set`): no
        // transient copy.
        self.turn_tail_view(slot, layer)?.slice_set(&src, 2, 0)?;
        let c = &mut self.turn_checkpoints[slot];
        c.tail_len = keep;
        c.valid = true;
        // The second target: the columns before it, from the same input.
        if let Some(w) = self.turn_pin_cur {
            let (start, seq) = self.turn_pass;
            let before = src_len - seq + (w - start);
            let keep = before.min(want);
            if keep >= l_cache {
                let src = conv_in.narrow(2, before - keep, keep)?;
                self.turn_tail_view(TURN_CHECKPOINTS, layer)?
                    .slice_set(&src, 2, 0)?;
                self.turn_pinned = Some(TurnCheckpoint {
                    pos: w,
                    tail_len: keep,
                    valid: true,
                });
                self.turn_pin_want = None;
            }
        }
        Ok(())
    }

    /// The frames the turn ring holds: the slots, and the pin's.
    fn turn_frames(&self) -> usize {
        TURN_CHECKPOINTS + usize::from(self.turn_pin_frame)
    }

    /// One frame more in the turn ring, for a pinned checkpoint
    /// ([`Self::pin_checkpoint`]). Set before the first checkpointing pass.
    pub fn with_pin_frame(mut self, on: bool) -> Self {
        self.turn_pin_frame = on;
        self
    }

    /// Whether the turn ring holds a frame for a pin, on an arm that keeps
    /// turn checkpoints at all.
    pub fn has_pin_frame(&self) -> bool {
        self.turn_pin_frame && self.use_kv_cache && self.kv_append != KvAppend::Cat
    }

    /// Where the pinned checkpoint stands.
    pub fn pinned_checkpoint(&self) -> Option<usize> {
        self.turn_pinned.filter(|c| c.valid).map(|c| c.pos)
    }

    /// Pin the turn checkpoint at `pos` (`CausalLM::pin_checkpoint`): one
    /// reaching it is copied into the pin's frame, else — while a request
    /// is open — the pass that covers it leaves it there beside its own.
    /// `None` releases the pin. Returns where the pinned checkpoint stands.
    pub fn pin_checkpoint(&mut self, pos: Option<usize>) -> Result<Option<usize>> {
        let Some(p) = pos else {
            self.turn_pinned = None;
            self.turn_pin_want = None;
            return Ok(None);
        };
        if !self.has_pin_frame() || self.pinned_checkpoint() == Some(p) {
            return Ok(self.pinned_checkpoint());
        }
        match self.slot_reaching(p) {
            Some(idx) if p <= self.kv_len() => {
                let c = self.turn_checkpoints[idx];
                let back = c.pos - p;
                let keep = (c.tail_len - back).min(self.l_cache + TURN_CHECKPOINT_SLACK);
                // Two frames of one ring share its storage: the source goes
                // through a copy of its own.
                for layer in 0..self.conv_states.len() {
                    let src = self
                        .turn_tail_view(idx, layer)?
                        .narrow(2, c.tail_len - back - keep, keep)?
                        .copy()?;
                    self.turn_tail_view(TURN_CHECKPOINTS, layer)?
                        .slice_set(&src, 2, 0)?;
                }
                self.turn_pinned = Some(TurnCheckpoint {
                    pos: p,
                    tail_len: keep,
                    valid: true,
                });
                self.turn_pin_want = None;
            }
            _ if self.turn_request => self.turn_pin_want = Some(p),
            _ => {}
        }
        Ok(self.pinned_checkpoint())
    }

    /// A request opens (`CausalLM::open_checkpoint_request`): the turn
    /// checkpoints are every multi-token pass's, whatever it asks; only a
    /// second target is scheduled inside it.
    pub fn open_checkpoint_request(&mut self) {
        self.turn_request = true;
        self.turn_pin_want = None;
    }

    /// The request's end: a second target no pass took is dropped.
    pub fn close_checkpoint_request(&mut self) {
        self.turn_request = false;
        self.turn_pin_want = None;
    }

    /// The checkpoint record of slot `idx`, the pin's at [`TURN_CHECKPOINTS`].
    fn turn_checkpoint(&self, idx: usize) -> Option<TurnCheckpoint> {
        if idx == TURN_CHECKPOINTS {
            self.turn_pinned
        } else {
            self.turn_checkpoints.get(idx).copied()
        }
    }

    /// Slot `slot`'s tail for conv layer `layer`: a contiguous
    /// `[b, hidden, l_cache + TURN_CHECKPOINT_SLACK]` view into the ring.
    fn turn_tail_view(&self, slot: usize, layer: usize) -> Result<Tensor> {
        let ring = self
            .turn_ring
            .as_ref()
            .ok_or_else(|| lloom_tensor::Error::msg("lfm2 turn ring: no ring allocated"))?;
        if slot >= ring.frames() {
            lloom_tensor::bail!("lfm2 turn ring: no frame {slot} of {}", ring.frames())
        }
        ring.frame(slot)?.narrow(0, layer, 1)?.squeeze(0)
    }

    /// The checkpoint [`Cache::restore`] would take `pos` from: a valid one
    /// whose pass ended at or after `pos`, within its tail's reach.
    fn checkpoint_for(&self, pos: usize) -> Option<usize> {
        self.slot_reaching(pos).or_else(|| {
            self.turn_pinned
                .filter(|c| self.reaches(c, pos))
                .map(|_| TURN_CHECKPOINTS)
        })
    }

    /// The turn slot (never the pin's) whose checkpoint reaches `pos`.
    fn slot_reaching(&self, pos: usize) -> Option<usize> {
        self.turn_checkpoints
            .iter()
            .position(|c| self.reaches(c, pos))
    }

    fn reaches(&self, c: &TurnCheckpoint, pos: usize) -> bool {
        c.valid
            && c.tail_len >= self.l_cache
            && pos <= c.pos
            && c.pos - pos <= c.tail_len - self.l_cache
    }

    /// Checkpoints past `pos` describe positions the cache no longer holds;
    /// a pin past it is released.
    fn forget_checkpoints_past(&mut self, pos: usize) {
        for c in self.turn_checkpoints.iter_mut() {
            if c.pos > pos {
                c.valid = false;
            }
        }
        if self.turn_pinned.is_some_and(|c| c.pos > pos) {
            self.turn_pinned = None;
        }
    }

    /// Where a request diverging at `pos` is put back
    /// (`CacheRewind::restore_point`): `pos` itself where a checkpoint
    /// reaches it, else the pinned checkpoint's own position at or before
    /// it, from which the request prefills the rest.
    pub fn restore_point(&self, pos: usize) -> Option<usize> {
        if self.can_restore(pos) {
            return Some(pos);
        }
        self.pinned_checkpoint()
            .filter(|&p| p <= pos && self.can_restore(p))
    }

    /// Whether [`Cache::restore`] can put the cache back at `pos`: the KV has
    /// a length to move, no speculative window is open, and a checkpoint's
    /// tail reaches `pos` ([`CacheRewind::can_restore`]).
    pub fn can_restore(&self, pos: usize) -> bool {
        self.use_kv_cache
            && !self.speculative
            && self.kv_append != KvAppend::Cat
            && pos <= self.kv_len()
            && self.checkpoint_for(pos).is_some()
    }

    /// Put the cache back at `pos` from a turn checkpoint: the KV planes to
    /// that length, every conv layer's state to what a multi-token pass ending
    /// at `pos` writes — its newest `l_cache` input columns at the front of
    /// the state, the ring regions past them as a prefill leaves them, the
    /// phase at the prefill's seed. The next pass then continues exactly as it
    /// would after a prefill of the first `pos` tokens.
    pub fn restore(&mut self, pos: usize) -> Result<()> {
        if !self.can_restore(pos) {
            lloom_tensor::bail!(
                "lfm2 restore({pos}): no turn checkpoint reaches it (kv_len {}, checkpoints at {:?})",
                self.kv_len(),
                self.turn_checkpoints
                    .iter()
                    .chain(self.turn_pinned.iter())
                    .filter(|c| c.valid)
                    .map(|c| (c.pos, c.tail_len.saturating_sub(self.l_cache)))
                    .collect::<Vec<_>>()
            )
        }
        let idx = self.checkpoint_for(pos).expect("checked by can_restore");
        let c = self
            .turn_checkpoint(idx)
            .expect("checkpoint_for names a held checkpoint");
        let n = self.kv_len() - pos;
        self.rewind_kv(n)?;
        let back = c.pos - pos;
        let tail_len = c.tail_len;
        let l_cache = self.l_cache;
        let width = self.conv_state.width(l_cache);
        let layers = self.conv_states.len();
        for layer in 0..layers {
            let tail = self.turn_tail_view(idx, layer)?;
            let window = tail
                .narrow(2, tail_len - back - l_cache, l_cache)?
                .contiguous()?;
            let (b_sz, hidden, _) = window.dims3()?;
            let owned = match &self.conv_states[layer] {
                Some(prev)
                    if prev.dims() == [b_sz, hidden, width]
                        && prev.dtype() == window.dtype()
                        && prev.is_contiguous() =>
                {
                    prev.clone()
                }
                _ => {
                    #[cfg(feature = "metal")]
                    let _declared = lloom_tensor::EscapeGuard::new();
                    zs::zeros(
                        zs::LFM2_CONV_STATE,
                        (b_sz, hidden, width),
                        window.dtype(),
                        window.device(),
                    )?
                }
            };
            owned.slice_set(&window, 2, 0)?;
            if width > l_cache {
                let region = owned.narrow(2, l_cache, width - l_cache)?;
                match self.conv_state {
                    ConvState::RotatingRing { .. } => {
                        zs::zero_region(zs::LFM2_CONV_RING_HISTORY, &region)?
                    }
                    _ => zs::poison_region(zs::LFM2_CONV_RING_SLACK, &region)?,
                }
            }
            self.set_conv_state(layer, owned)?;
        }
        // The seed a multi-token pass leaves (`advance_conv_phase`'s
        // multi-token arm): the shuffle's phase never moves, the rings' sits
        // at the newest column, nothing behind the window is history.
        if !matches!(self.conv_state, ConvState::Shuffle) {
            self.conv_phase = l_cache - 1;
        }
        self.conv_phase_prev = self.conv_phase;
        self.conv_history = 0;
        self.conv_compact = false;
        self.paged_pass_pending = false;
        #[cfg(feature = "metal")]
        if let Some(b) = self.shared_mut() {
            b.pass = None;
        }
        self.forget_checkpoints_past(pos);
        Ok(())
    }

    /// The conv ring's current write slot.
    ///
    /// Exposed because the speculative rollback contract is a statement *about*
    /// this index, so a test asserting the contract has to be able to read it.
    /// Read-only: the phase is advanced by `Model::forward` alone, which is the
    /// one-owner property §10.2e records as having already been wrong once.
    /// The last `n` positions were never written (lloom #735's EOS gate):
    /// move the counters back, touch no contents. See
    /// [`crate::nn::model::CacheRewind::rewind_unwritten`].
    ///
    /// Per arm: the KV slabs and the paged pool decrement their lengths (the
    /// rows past the new end were never written; the next append overwrites
    /// them). `Shuffle` keeps no per-step conv counter — its phase never moves
    /// on a single-token pass — so nothing to undo, provided a gated step left
    /// the state as it was: the fused step carries it through, the unfused
    /// step writes it in place with gated copies (§11.5z-x). `RotatingRing` advanced
    /// `conv_phase`/`conv_history` per step on the host and is stepped back
    /// the same way. Refused: `KvAppend::Cat` (its length is the tensor's; a
    /// gated `cat` still grew it with unwritten rows — narrowing it is a copy
    /// this seam does not make), `SlidingRing` (its compaction pass is a
    /// device write scheduled from the same counter), and an open speculative
    /// window.
    pub fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if self.speculative {
            lloom_tensor::bail!("lfm2 rewind_unwritten({n}): a speculative window is open");
        }
        if self.use_kv_cache && self.kv_append == KvAppend::Cat {
            lloom_tensor::bail!(
                "lfm2 rewind_unwritten({n}): `KvAppend::Cat` has no length to move back; the \
                 EOS gate needs `in-place` or `paged`"
            );
        }
        if let ConvState::SlidingRing { .. } = self.conv_state {
            lloom_tensor::bail!(
                "lfm2 rewind_unwritten({n}): the sliding conv ring's compaction is scheduled \
                 from the per-step counter; the EOS gate needs `shuffle` or `rotating`"
            );
        }
        match self.kv.arm_mut() {
            ArmMut::One(own) => own.rewind(n)?,
            ArmMut::Paged(p) => p.rewind_unwritten(n)?,
        }
        if let ConvState::RotatingRing { .. } = self.conv_state {
            if n > self.conv_history {
                lloom_tensor::bail!(
                    "lfm2 rewind_unwritten({n}): only {} rotating-ring steps since the last \
                     multi-token pass",
                    self.conv_history
                );
            }
            let width = self.conv_state.width(self.l_cache);
            self.conv_phase = (self.conv_phase + width - n) % width;
            self.conv_history -= n;
        }
        self.forget_checkpoints_past(self.kv_len());
        Ok(())
    }

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
    /// The move-only receipt prevents consuming the same value twice, but does
    /// not borrow this cache or prevent overlapping windows. Callers keep window
    /// order; reset discards the sequence without certifying old receipts.
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
            kv_len: self.own().and_then(OwnKv::slot_len),
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
        match self.kv.arm_mut() {
            ArmMut::One(own) => own.rewind(discard)?,
            ArmMut::Paged(p) => p.rewind(discard)?,
        }

        // Conv: a pointer move, and this is the line §10.2a's cost column means
        // by *"a pointer move"*. The discarded slots keep their bytes and become
        // unreachable, exactly as the KV positions above `len` do — the ring is
        // `l_cache + k` wide precisely so that `k` speculative writes cannot
        // reach the live window a rollback returns to.
        self.conv_phase = (tok.conv_phase + n) % width;

        // The KV length is checked the same way and for the same reason. It is
        // recorded per `advance` from layer 0, and every layer moves together.
        if let (Some(before), Some(len)) = (tok.kv_len, self.own().and_then(OwnKv::slot_len)) {
            debug_assert_eq!(
                len,
                before + n,
                "resolve must leave exactly the accepted positions live"
            );
        }
        Ok(())
    }
}

/// The state a driver builds for its shape (`profiles-and-memory-design.md`
/// §3), named once in generic code (`P::state`): the `One` shapes a
/// sequence's own cache, the `Paged` shapes a row of the shared pool when
/// one is given (`--batch N --rows pool`) and a single sequence on its own
/// page pool when none is (`lfm2-determinism --preset pool`).
pub trait Shape: Profile {
    fn state(
        config: &Config,
        dtype: DType,
        device: &Device,
        kv_capacity: usize,
        pool: Option<&BatchPool>,
    ) -> Result<State<Self>>;
}

impl<G: crate::profile::Regime, W: crate::profile::Rewind> Shape
    for crate::profile::P<G, W, crate::profile::One>
{
    fn state(
        config: &Config,
        dtype: DType,
        device: &Device,
        kv_capacity: usize,
        pool: Option<&BatchPool>,
    ) -> Result<State<Self>> {
        if pool.is_some() {
            lloom_tensor::bail!(
                "lfm2: a {} state is a sequence's own; the shared pool's rows are the `Paged` shapes",
                Self::KEY
            )
        }
        State::<Self>::new_with(true, dtype, config, device, kv_capacity)
    }
}

impl<G: crate::profile::Regime, W: crate::profile::Rewind> Shape
    for crate::profile::P<G, W, crate::profile::Paged>
{
    fn state(
        config: &Config,
        dtype: DType,
        device: &Device,
        kv_capacity: usize,
        pool: Option<&BatchPool>,
    ) -> Result<State<Self>> {
        match pool {
            Some(p) => State::<Self>::for_batch(true, dtype, config, device, p),
            None => State::<Self>::new_with(true, dtype, config, device, kv_capacity),
        }
    }
}

/// The rows' shape (`Paged`): a row of a shared [`BatchPool`].
impl<P: Profile<Rows = crate::profile::Paged>> State<P> {
    /// A cache whose KV pages and conv ring live in `pool`, at a slot leased
    /// here and returned when the cache drops (`batched-decode-design.md`
    /// §12.2). `config` must be the pool's (`Paged × FlashDecoding ×
    /// RotatingRing`); everything else — positions, taps, the token log's
    /// owner — is the row's own, as on any cache.
    #[allow(unused_variables)]
    pub fn for_batch(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        device: &Device,
        pool: &BatchPool,
    ) -> Result<Self> {
        #[cfg(not(feature = "metal"))]
        {
            lloom_tensor::bail!("lfm2 State::<P>::for_batch is Metal-only")
        }
        #[cfg(feature = "metal")]
        {
            if !use_kv_cache {
                lloom_tensor::bail!("lfm2 State::<P>::for_batch: a batch row keeps its KV")
            }
            PooledKv::check(config, device)?;
            let lease = pool.lease()?;
            let page_size = pool.page_size();
            let slot = BatchSlot {
                pool: pool.clone(),
                lease: Arc::new(lease),
                pass: None,
                pass_start: 0,
                page_size,
                staged_snaps: Vec::new(),
            };
            Self::with_kv(true, dtype, config, device, pool.kv_capacity(), |_, _| {
                Ok(crate::profile::PagedOf(PooledKv::Shared(slot)))
            })
        }
    }
}

/// The receipt for an open speculative window — `DESIGN.md` §10.2a.
///
/// `#[must_use]` warns about an ignored receipt, and a move prevents consuming
/// the same value twice. Neither prevents overlapping advances nor binds the
/// receipt to a cache/reset generation; that identity contract is not enforced.
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
    /// Resident: the three projections, loaded once. Streamed (§4.8): 1×1
    /// placeholders, never bound — the slot views are taken per pass
    /// (`DecoderLayer::weights`).
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
    /// How the gate product is dispatched (§12.2e row 3b, §12.2 row 5, #732).
    gemv_gate: GateFusion,
    /// `(hidden, intermediate)`: the projections' extents, which a streamed
    /// layer's placeholders do not carry (its walk reads them here).
    dims: (usize, usize),
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
            dims: (cfg.hidden_size, cfg.intermediate_size),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    /// A layer whose matrices live on the weight-group tier
    /// (`profiles-and-memory-design.md` §4.8): nothing is loaded here; the
    /// group's MLP parts are `w1`, `w3`, `w2` in the tier's slot, taken as
    /// views at each pass (`DecoderLayer::weights`).
    fn new_placeholder(cfg: &Config, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: placeholder_linear(vb)?,
            up_proj: placeholder_linear(vb)?,
            down_proj: placeholder_linear(vb)?,
            gemv_silu: cfg.gemv_silu,
            gemv_residual: cfg.gemv_residual,
            gemv_gate: cfg.gemv_gate,
            dims: (cfg.hidden_size, cfg.intermediate_size),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    /// The resident projections, borrowed (R1: a model that fits never
    /// streams — nothing else happens).
    fn resident(&self) -> Projections<'_> {
        use std::borrow::Cow;
        Projections {
            gate: Cow::Borrowed(&self.gate_proj),
            up: Cow::Borrowed(&self.up_proj),
            down: Cow::Borrowed(&self.down_proj),
        }
    }

    /// Whether the fused `w1` + SiLU arm applies to this call.
    ///
    /// Metal, `seq_len == 1`, no bias, and a dtype the GEMV carries. **Every
    /// other case takes the eager path rather than failing**, because unlike
    /// `flash_decoding` this arm has a correct eager equivalent right beside
    /// it and prefill is a GEMM the fusion is not instantiated for.
    fn gemv_silu_applies(&self, x: &Tensor, p: &Projections<'_>) -> bool {
        self.gemv_silu
            && x.device().is_metal()
            && x.dims().last() == Some(&p.gate.weight().dims()[1])
            // One row only, for `gemv_residual_applies`'s reason: at `B > 1`
            // the fused GEMV would sweep `w1` once per row.
            && matches!(x.dims(), [1, 1, _] | [1, _])
            && p.gate.bias().is_none()
            // See `gemv_residual_applies`: the fused op serves the contiguous
            // layout `Linear::forward` reshapes, not the broadcast one.
            && x.is_contiguous()
            && p.gate.weight().is_contiguous()
            && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
    }

    fn forward(&self, x: &Tensor, p: &Projections<'_>) -> Result<Tensor> {
        let _enter = self.span.enter();
        p.down.forward(&self.gated(x, p)?)
    }

    /// `mlp(x) + residual`, fusing the add onto `w2`'s store when the arm is
    /// on (§12.2h). Otherwise **literally `forward(x)? + residual`** — the
    /// unfused branch calls `forward` rather than reimplementing it, so the
    /// two cannot drift and the shipping path has exactly one definition.
    fn forward_add(&self, x: &Tensor, residual: &Tensor, p: &Projections<'_>) -> Result<Tensor> {
        if self.gemv_residual {
            let _enter = self.span.enter();
            let gated = self.gated(x, p)?;
            if gemv_residual_applies(true, &gated, &p.down, residual) {
                return crate::nn::ops::gemv_residual(&gated, p.down.weight(), residual);
            }
            return p.down.forward(&gated)? + residual;
        }
        self.forward(x, p)? + residual
    }

    /// Whether [`Self::forward_add_normed`] serves `x`: `w1` and `w3` each
    /// take the norm — in their own GEMVs with the gate arm off, inside the
    /// two-weight kernel under `--gemv-gate swiglu` (`gemv_norm_swiglu`, the
    /// two weights the same shape, as that arm requires). Not under
    /// `--gemv-gate mul`: no kernel carries the norm with `w3`'s product, so
    /// that arm keeps its own path and the norm stays a dispatch.
    fn norm_gemv_applies(&self, on: bool, x: &Tensor, norm: &RmsNorm, p: &Projections<'_>) -> bool {
        let gate = match self.gemv_gate {
            GateFusion::Off => true,
            GateFusion::Swiglu => p.gate.weight().dims() == p.up.weight().dims(),
            GateFusion::Mul => false,
        };
        gate && norm_gemv_applies(on, x, norm, &p.gate) && norm_gemv_applies(on, x, norm, &p.up)
    }

    /// `mlp(rmsnorm(x)) + residual` with the norm inside `w1`'s and `w3`'s
    /// GEMVs, the product a `bmul` — or inside `gemv_swiglu`'s one kernel
    /// under that gate arm. Every other choice is the one
    /// [`Self::forward_add`] makes under the same flags: the SiLU rides `w1`
    /// iff the SiLU arm would fuse it, `w2` adds the residual iff the
    /// residual arm is on — so the norm arm moves the norms and nothing
    /// else. The caller has checked [`Self::norm_gemv_applies`].
    fn forward_add_normed(
        &self,
        x: &Tensor,
        norm: &RmsNorm,
        residual: &Tensor,
        p: &Projections<'_>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gated = if self.gemv_gate == GateFusion::Swiglu {
            // `gated`'s first branch with the norm inside: both weights in
            // one kernel, the SiLU arm moot as it is there.
            crate::nn::ops::gemv_norm_swiglu(
                x,
                p.up.weight(),
                p.gate.weight(),
                norm.weight(),
                norm.eps() as f32,
            )?
        } else {
            let gate = if self.gemv_silu_applies(x, p) {
                norm_gemv(x, norm, &p.gate, true)?
            } else {
                crate::nn::ops::silu(&norm_gemv(x, norm, &p.gate, false)?)?
            };
            let up = norm_gemv(x, norm, &p.up, false)?;
            (gate * up)?
        };
        if self.gemv_residual && gemv_residual_applies(true, &gated, &p.down, residual) {
            return crate::nn::ops::gemv_residual(&gated, p.down.weight(), residual);
        }
        p.down.forward(&gated)? + residual
    }

    /// `silu(w1(x)) * w3(x)` — the SwiGLU gate, shared by both entry points so
    /// the two cannot drift in what they compute.
    fn gated(&self, x: &Tensor, p: &Projections<'_>) -> Result<Tensor> {
        // Both weights in one kernel: `w1`, `w3`, the SiLU and the product,
        // no barrier between the two sweeps (§12.2 row 5 with row 3b's
        // product, #732). Under it the SiLU arm is moot — `w1` is never
        // dispatched alone — so it is tried first.
        if self.gemv_swiglu_applies(x, p) {
            return crate::nn::ops::gemv_swiglu(x, p.up.weight(), p.gate.weight());
        }
        // The fused arm replaces `gemv` + `silu_f16` with one dispatch, and
        // computes SiLU on the f32 accumulator rather than on the rounded
        // store — so it MOVES THE DIGESTS by construction (§12.2d, §2.3.5a).
        let gate = if self.gemv_silu_applies(x, p) {
            crate::nn::ops::gemv_silu(x, p.gate.weight())?
        } else {
            crate::nn::ops::silu(&p.gate.forward(x)?)?
        };
        // The product on `w3`'s store, the gate its row-indexed operand: one
        // dispatch for `w3` and the `bmul` (§12.2e row 3b, #732). The same
        // applicability as the residual arm — `w3` has `w1`'s shape, and the
        // gate is exactly `w3`'s output extent, contiguous, as `gemv_silu`
        // and `silu` both store it.
        if self.gemv_gate == GateFusion::Mul && gemv_residual_applies(true, x, &p.up, &gate) {
            return crate::nn::ops::gemv_mul(x, p.up.weight(), &gate);
        }
        let up = p.up.forward(x)?;
        gate * up
    }

    /// Whether the two-weight kernel serves this call: `gemv_silu_applies`'s
    /// conditions on `w1`, and `w3` the same shape and layout — the kernel
    /// streams both under one leading dimension.
    fn gemv_swiglu_applies(&self, x: &Tensor, p: &Projections<'_>) -> bool {
        self.gemv_gate == GateFusion::Swiglu
            && x.device().is_metal()
            && x.dims().last() == Some(&p.gate.weight().dims()[1])
            && matches!(x.dims(), [1, 1, _] | [1, _])
            && p.gate.bias().is_none()
            && p.up.bias().is_none()
            && x.is_contiguous()
            && p.gate.weight().is_contiguous()
            && p.up.weight().is_contiguous()
            && p.gate.weight().dims() == p.up.weight().dims()
            && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
    }
}

/// The SwiGLU MLP's decode step: `w2(silu(w1(x)) * w3(x))`, the down
/// projection unadded — the unfused chain, whichever epilogue arms are set:
/// those are what a derivation over it decides.
#[cfg(not(target_arch = "wasm32"))]
impl crate::optimizer::Visit for Mlp {
    type In = crate::optimizer::NodeId;
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, xn: Self::In) -> Self::Out {
        use crate::optimizer::Shape;
        let (h, i) = self.dims;
        let dt = self.down_proj.weight().dtype();
        let w1 = w.weight("w1.weight", Shape::of(&[i, h]), dt);
        let w3 = w.weight("w3.weight", Shape::of(&[i, h]), dt);
        let w2 = w.weight("w2.weight", Shape::of(&[h, i]), dt);
        let gate = w.gemv(w1, xn);
        let gate = w.silu(gate);
        let up = w.gemv(w3, xn);
        let gated = w.mul(gate, up);
        w.gemv(w2, gated)
    }
}

/// The dense SwiGLU MLP as a level-0 block (`nn::blocks::Ffn`): the
/// resident projections, no per-sequence state. A streamed layer's views
/// are the family's own path (`DecoderLayer::weights`).
impl crate::nn::blocks::Ffn for Mlp {
    type State = ();
    fn execute(&self, x: &Tensor, _state: &mut ()) -> Result<Tensor> {
        self.forward(x, &self.resident())
    }
}

/// Attention layer with per-head QK normalization and RoPE.
#[derive(Debug, Clone)]
struct Attention<P: Profile> {
    profile: PhantomData<P>,
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
    /// The kernels this layer takes, decided at load (`AttnRoute::resolve`).
    route: AttnRoute,
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
    /// The model width, which a streamed layer's placeholders do not carry.
    hidden_size: usize,
    /// How the cache this model is configured for grows (`Config::kv_append`):
    /// the cache's choice, read by the walk only.
    kv_append: KvAppend,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl<P: Profile> Attention<P> {
    /// `streamed`: the projections live in the weight-group tier (§4.8);
    /// the fields are placeholders and `LayerWeights` carries the views.
    fn new(cfg: &Config, route: AttnRoute, vb: VarBuilder, streamed: bool) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let (q_proj, k_proj, v_proj, o_proj) = if streamed {
            (
                placeholder_linear(&vb)?,
                placeholder_linear(&vb)?,
                placeholder_linear(&vb)?,
                placeholder_linear(&vb)?,
            )
        } else {
            (
                linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?,
                linear(
                    num_attention_heads * head_dim,
                    hidden_size,
                    vb.pp("out_proj"),
                )?,
            )
        };

        let q_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            profile: PhantomData,
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
            route,
            flash_page_size: cfg.flash_page_size,
            flash_pages_per_chunk: cfg.flash_pages_per_chunk,
            flash_scratch_sizing: cfg.flash_scratch_sizing,
            gemv_residual: cfg.gemv_residual,
            qkv_layout: cfg.qkv_layout,
            hidden_size,
            kv_append: cfg.kv_append,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    /// Whether this call takes the FlashDecoding kernels (`DESIGN.md` §10.4):
    /// the route chose them at load, the call is one token, and the dtype is
    /// f16 or f32.
    ///
    /// `flash_decoding.metal` has no `bfloat` instantiation, so at
    /// `--attn flash --dtype bf16` the route takes `sdpa_vector` and warns,
    /// while `config_line()` still renders `AttnImpl=FlashDecoding` (§3.9,
    /// §7.1a). Attribute a flash run from the kernel census, never from the
    /// config line (§2.4).
    fn flash_decoding_applies(&self, q: &Tensor, seq_len: usize) -> bool {
        self.route.decode == DecodeAttn::FlashDecoding
            && seq_len == 1
            && matches!(q.dtype(), DType::F16 | DType::F32)
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &State<P>) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        crate::nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn resident(&self) -> AttnWeights<'_> {
        use std::borrow::Cow;
        AttnWeights {
            q: Cow::Borrowed(&self.q_proj),
            k: Cow::Borrowed(&self.k_proj),
            v: Cow::Borrowed(&self.v_proj),
            o: Cow::Borrowed(&self.o_proj),
        }
    }

    fn forward(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let y = self.attend(w, x, index_pos, block_idx, cache)?;
        w.o.forward(&y)
    }

    /// `attn(x) + residual`, fusing the add onto `out_proj`'s store when the
    /// arm is on (§12.2h). Otherwise **literally `forward(..)? + residual`** —
    /// the unfused branch calls `forward`, so the shipping path has exactly one
    /// definition and the two cannot drift.
    fn forward_add(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if self.gemv_residual {
            let y = self.attend(w, x, index_pos, block_idx, cache)?;
            if gemv_residual_applies(true, &y, &w.o, residual) {
                return crate::nn::ops::gemv_residual(&y, w.o.weight(), residual);
            }
            return w.o.forward(&y)? + residual;
        }
        self.forward(w, x, index_pos, block_idx, cache)? + residual
    }

    /// [`Self::forward_add`] over the un-normalized `x`, `norm` riding the
    /// q/k/v projections.
    #[allow(clippy::too_many_arguments)]
    fn forward_add_normed(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        norm: &RmsNorm,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = self.attend_normed(w, x, Some(norm), index_pos, block_idx, cache)?;
        if self.gemv_residual && gemv_residual_applies(true, &y, &w.o, residual) {
            return crate::nn::ops::gemv_residual(&y, w.o.weight(), residual);
        }
        w.o.forward(&y)? + residual
    }

    /// [`Self::forward_add`] for one step of `rows.len()` independent
    /// sequences (`BatchedCausalLM`): the projections run once over
    /// `[B, 1, hidden]`, each row attends against its own cache. The residual
    /// arm is decided as `forward_add` decides it, so a one-row batch is that
    /// path exactly.
    fn forward_rows_add(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = self.attend_rows(w, x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &y, &w.o, residual) {
            return crate::nn::ops::gemv_residual(&y, w.o.weight(), residual);
        }
        w.o.forward(&y)? + residual
    }

    /// Everything up to but excluding `out_proj`, so the two entry points
    /// above share one body and cannot drift in what they compute.
    fn attend(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        self.attend_normed(w, x, None, index_pos, block_idx, cache)
    }

    /// [`Self::attend`] with the input norm riding the projections.
    fn attend_normed(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        norm: Option<&RmsNorm>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q, k, v, rotated) = self.project_normed(w, x, norm, Some((index_pos, &*cache)))?;
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
        w: &AttnWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;
        #[cfg(feature = "metal")]
        if let Some(y) = self.attend_rows_shared(w, x, block_idx, rows)? {
            return Ok(y);
        }
        // Rows attend at different positions, so the rotation is per row: the
        // projections are normed here and rotated below, on every arm.
        let (q, k, v, rotated) = self.project(w, x, None)?;
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
        w: &AttnWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let y = self.attend_ragged(w, x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &y, &w.o, residual) {
            return crate::nn::ops::gemv_residual(&y, w.o.weight(), residual);
        }
        w.o.forward(&y)? + residual
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
        w: &AttnWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_, total, _) = x.dims3()?;
        let q = w.q.forward(x)?;
        let k = w.k.forward(x)?;
        let v = w.v.forward(x)?;
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
        #[cfg(feature = "metal")]
        if let Some(y) = self.attend_ragged_shared(&q, &k, &v, block_idx, rows)? {
            return Ok(y);
        }
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

    /// Whether every row is a slot of ONE [`BatchPool`] the batched arms can
    /// address in one dispatch (`batched-decode-design.md` §12.2): all on the
    /// pool, keeping their KV, unquantized, untapped, and the dtype and head
    /// dim the flash kernels instantiate.
    #[cfg(feature = "metal")]
    fn rows_share_a_pool<'a>(
        &self,
        caches: impl Iterator<Item = &'a State<P>>,
        dtype: DType,
    ) -> bool {
        if self.route.decode != DecodeAttn::FlashDecoding {
            return false;
        }
        let mut first: Option<&State<P>> = None;
        for c in caches {
            if c.shared().is_none() || !c.use_kv_cache || c.kv_quant.is_on() {
                return false;
            }
            match first {
                None => first = Some(c),
                Some(f) => {
                    if !f.shares_pool_with(c) {
                        return false;
                    }
                }
            }
        }
        match first {
            Some(f) => {
                matches!(dtype, DType::F16 | DType::F32)
                    && f.shared().expect("checked").pool.inner.lock().batched_arms
            }
            None => false,
        }
    }

    /// The pass the batched arm reads and writes: opened at the pass's first
    /// attention layer (every row's `paged_pass_pending` is set), reused by
    /// the layers after it. `rows` are `(slot, tokens)` in the pass's order.
    #[cfg(feature = "metal")]
    fn shared_pass(
        pool: &BatchPool,
        caches: &mut [&mut State<P>],
        rows: &[(usize, usize)],
        device: &Device,
    ) -> Result<Arc<RowsPass>> {
        let pending = caches.iter().any(|c| c.paged_pass_pending);
        if pending {
            for c in caches.iter_mut() {
                c.paged_pass_pending = false;
            }
            let pass = Arc::new(pool.inner.lock().kv.begin_rows(rows, device)?);
            for c in caches.iter_mut() {
                if let Some(b) = c.shared_mut() {
                    b.pass = Some(pass.clone());
                }
            }
            return Ok(pass);
        }
        let pass = caches[0]
            .shared()
            .and_then(|b| b.pass.clone())
            .ok_or_else(|| lloom_tensor::Error::msg("lfm2 batched arm: no pass is open"))?;
        if pass.rows != rows {
            lloom_tensor::bail!(
                "lfm2 batched arm: the rows changed mid-pass ({:?} against the open pass's {:?})",
                rows,
                pass.rows
            )
        }
        Ok(pass)
    }

    /// `attend_rows` in one dispatch per kernel for the whole batch, when the
    /// rows share a pool (§12.2): one rotation at every row's own position
    /// (`cos`/`sin` gathered per row), one scatter of the B tokens, one paged
    /// read over B tables with B live counts. `None` when the rows cannot
    /// batch, and the per-row loop runs.
    #[cfg(feature = "metal")]
    fn attend_rows_shared(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Option<Tensor>> {
        if !self.rows_share_a_pool(rows.iter().map(|r| &*r.cache), x.dtype()) {
            return Ok(None);
        }
        let (b_sz, seq_len, _) = x.dims3()?;
        if seq_len != 1 {
            return Ok(None);
        }
        let device = x.device().clone();
        let (q, k, v, rotated) = self.project(w, x, None)?;
        debug_assert!(!rotated);
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // RoPE at every row's own position: the tables' rows gathered per
        // batch row, `[B, 1, D/2]`, which the rope kernel indexes per batch.
        let pos: Vec<u32> = rows.iter().map(|r| r.pos as u32).collect();
        let pos_t = Tensor::new(pos.as_slice(), &device)?;
        let cos = rows[0].cache.cos.index_select(&pos_t, 0)?.unsqueeze(1)?;
        let sin = rows[0].cache.sin.index_select(&pos_t, 0)?.unsqueeze(1)?;
        let q = crate::nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = crate::nn::rotary_emb::rope(&k, &cos, &sin)?;

        let pool = rows[0].cache.shared().expect("checked").pool.clone();
        let slots: Vec<(usize, usize)> = rows
            .iter()
            .map(|r| (r.cache.shared().expect("checked").slot(), 1))
            .collect();
        let mut caches: Vec<&mut State<P>> = rows.iter_mut().map(|r| &mut *r.cache).collect();
        let pass = Self::shared_pass(&pool, &mut caches, &slots, &device)?;
        let reserve_chunks = caches[0]
            .kv_capacity
            .div_ceil((self.flash_page_size * self.flash_pages_per_chunk).max(1));
        drop(caches);

        let (pk, pv) = {
            let mut g = pool.inner.lock();
            g.shared_calls += 1;
            g.kv.write_rows(block_idx, &k, &v, &pass)?;
            let (pk, pv) = g.kv.pools[block_idx]
                .as_ref()
                .expect("write_rows allocates the layer's pool");
            (pk.clone(), pv.clone())
        };
        let y = crate::nn::ops::flash_decoding_paged_ragged_bound(
            &q,
            &pk,
            &pv,
            1f32 / (self.head_dim as f32).sqrt(),
            1.0,
            self.flash_page_size,
            self.flash_pages_per_chunk,
            self.flash_scratch_sizing,
            reserve_chunks,
            &pass.tables,
            pass.table_stride,
            &pass.n_keys,
            &pass.tables_t,
            &pass.n_keys_t,
        )?;
        Ok(Some(y.transpose(1, 2)?.reshape((
            b_sz,
            1,
            self.num_attention_heads * self.head_dim,
        ))?))
    }

    /// `attend_ragged` over a shared pool (§12.2): one rotation for all `T`
    /// tokens at their own positions, one scatter, the one-token rows read
    /// in one paged dispatch, and each longer row (a prompt piece) read by
    /// the steel kernel over its own page table with the causal mask inside
    /// its window. `q`/`k`/`v` are `[1, T, heads, D]`, normed.
    #[cfg(feature = "metal")]
    fn attend_ragged_shared(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Option<Tensor>> {
        if !self.rows_share_a_pool(rows.iter().map(|r| &*r.cache), q.dtype()) {
            return Ok(None);
        }
        let device = q.device().clone();
        let total = q.dim(1)?;
        let scale = 1f32 / (self.head_dim as f32).sqrt();

        // Every token's position, in the pass's order.
        let mut pos: Vec<u32> = Vec::with_capacity(total);
        for r in rows.iter() {
            pos.extend((r.pos..r.pos + r.len).map(|p| p as u32));
        }
        let pos_t = Tensor::new(pos.as_slice(), &device)?;
        let cos = rows[0].cache.cos.index_select(&pos_t, 0)?;
        let sin = rows[0].cache.sin.index_select(&pos_t, 0)?;
        let q_t = q.transpose(1, 2)?.contiguous()?; // [1, H, T, D]
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let q_t = crate::nn::rotary_emb::rope(&q_t, &cos, &sin)?;
        let k_t = crate::nn::rotary_emb::rope(&k_t, &cos, &sin)?;

        let pool = rows[0].cache.shared().expect("checked").pool.clone();
        let slots: Vec<(usize, usize)> = rows
            .iter()
            .map(|r| (r.cache.shared().expect("checked").slot(), r.len))
            .collect();
        let mut caches: Vec<&mut State<P>> = rows.iter_mut().map(|r| &mut *r.cache).collect();
        let pass = Self::shared_pass(&pool, &mut caches, &slots, &device)?;
        let reserve_chunks = caches[0]
            .kv_capacity
            .div_ceil((self.flash_page_size * self.flash_pages_per_chunk).max(1));
        drop(caches);

        let (pk, pv) = {
            let mut g = pool.inner.lock();
            g.shared_calls += 1;
            g.kv.write_rows(block_idx, &k_t, &v_t, &pass)?;
            let (pk, pv) = g.kv.pools[block_idx]
                .as_ref()
                .expect("write_rows allocates the layer's pool");
            (pk.clone(), pv.clone())
        };

        // The one-token rows, in one dispatch: their queries gathered to
        // `[nd, H, 1, D]`, their tables and live counts the pass's (every
        // row) or the pass's decode subset, uploaded once at `begin_rows`.
        let dec: Vec<usize> = match &pass.dec {
            Some(d) => d.rows.clone(),
            None => (0..rows.len()).filter(|&r| rows[r].len == 1).collect(),
        };
        let mut y_dec: Option<Tensor> = None;
        if !dec.is_empty() {
            let nd = dec.len();
            let idx: Vec<u32> = dec.iter().map(|&r| pass.span(r).0 as u32).collect();
            let prefix = idx.iter().enumerate().all(|(i, &t)| t as usize == i);
            let q_d = if prefix {
                q_t.narrow(2, 0, nd)?
            } else {
                q_t.index_select(&Tensor::new(idx.as_slice(), &device)?, 2)?
            };
            let q_d = q_d.transpose(1, 2)?.contiguous()?.reshape((
                nd,
                self.num_attention_heads,
                1,
                self.head_dim,
            ))?;
            // Borrowed from the pass: no `Vec` clone per layer per step.
            let (tables, n_keys, tables_t, n_keys_t) = match &pass.dec {
                Some(d) => (&d.tables[..], &d.n_keys[..], &d.tables_t, &d.n_keys_t),
                None => (
                    &pass.tables[..],
                    &pass.n_keys[..],
                    &pass.tables_t,
                    &pass.n_keys_t,
                ),
            };
            y_dec = Some(crate::nn::ops::flash_decoding_paged_ragged_bound(
                &q_d,
                &pk,
                &pv,
                scale,
                1.0,
                self.flash_page_size,
                self.flash_pages_per_chunk,
                self.flash_scratch_sizing,
                reserve_chunks,
                tables,
                pass.table_stride,
                n_keys,
                tables_t,
                n_keys_t,
            )?);
        }

        // The longer rows (verify windows, prompt pieces), grouped by length:
        // each group ONE steel dispatch over its rows' page tables, every row
        // at its own live count with the causal mask at the end of its own
        // key range (§12.2 change 3). A speculative round's windows are all
        // `K + 1` wide, so they are one group and one dispatch.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            if row.len > 1 {
                match groups.iter_mut().find(|(l, _)| *l == row.len) {
                    Some((_, g)) => g.push(r),
                    None => groups.push((row.len, vec![r])),
                }
            }
        }
        let mut y_long: Vec<Option<Tensor>> = (0..rows.len()).map(|_| None).collect();
        // One table / count scratch for every group (cleared per group), the
        // rows' page lists read in place.
        let mut tables: Vec<u32> = Vec::new();
        let mut n_keys: Vec<u32> = Vec::new();
        for (len, grp) in &groups {
            let n = grp.len();
            let offs: Vec<usize> = grp.iter().map(|&r| pass.span(r).0).collect();
            // `[n, H, len, D]`: the rows' spans, adjacent (one narrow) or
            // gathered (one index_select), then the heads brought forward.
            let adjacent = offs.windows(2).all(|w| w[1] == w[0] + len);
            let q_g = if n == 1 {
                q_t.narrow(2, offs[0], *len)?.contiguous()?
            } else {
                let flat = if adjacent {
                    q_t.narrow(2, offs[0], n * len)?
                } else {
                    let mut idx: Vec<u32> = Vec::with_capacity(n * len);
                    for &o in &offs {
                        idx.extend((o..o + len).map(|t| t as u32));
                    }
                    q_t.index_select(&Tensor::new(idx.as_slice(), &device)?, 2)?
                };
                flat.reshape((self.num_attention_heads, n, *len, self.head_dim))?
                    .transpose(0, 1)?
                    .contiguous()?
            };
            let stride = grp
                .iter()
                .map(|&r| pass.pages[r].len())
                .max()
                .unwrap_or(1)
                .max(1);
            tables.clear();
            n_keys.clear();
            for &r in grp {
                let p = &pass.pages[r];
                tables.extend_from_slice(p);
                tables.extend(std::iter::repeat_n(0, stride - p.len()));
                n_keys.push(pass.n_keys[r]);
            }
            let y_g = crate::nn::ops::sdpa_paged_ragged(
                &q_g,
                &pk,
                &pv,
                true,
                scale,
                self.flash_page_size,
                &tables,
                stride,
                &n_keys,
            )?;
            for (j, &r) in grp.iter().enumerate() {
                y_long[r] = Some(y_g.narrow(0, j, 1)?);
            }
        }

        // Assemble in row order: a decode row from the batched read, a
        // longer row from its group's read.
        let mut ys = Vec::with_capacity(rows.len());
        let mut j = 0usize;
        for r in 0..rows.len() {
            let (_, len) = pass.span(r);
            let y = if len == 1 {
                let y = y_dec
                    .as_ref()
                    .expect("a decode row was read")
                    .narrow(0, j, 1)?;
                j += 1;
                y
            } else {
                y_long[r]
                    .take()
                    .expect("a longer row was read with its group")
            };
            ys.push(y.transpose(1, 2)?.reshape((
                1,
                len,
                self.num_attention_heads * self.head_dim,
            ))?);
        }
        if ys.len() == 1 {
            return Ok(Some(ys.remove(0)));
        }
        Ok(Some(Tensor::cat(&ys, 1)?))
    }

    /// The three projections and the per-head QK norms: `x [b, seq, hidden]`
    /// to `(q, k, v, rotated)` in `[b, heads, seq, head_dim]`. `q`/`k` are
    /// normed; they are also rotated (`rotated == true`) on the `Fused` arm
    /// when `rope_at` names the position. `v` is contiguous on the
    /// `Contiguous` arm and the bare `transpose` view otherwise, which the
    /// in-place append writes from where it lies (`Tensor::slice_set`).
    fn project(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        rope_at: Option<(usize, &State<P>)>,
    ) -> Result<(Tensor, Tensor, Tensor, bool)> {
        self.project_normed(w, x, None, rope_at)
    }

    /// [`Self::project`]; with `norm`, `x` is the un-normalized input and the
    /// norm rides each projection's GEMV (`norm_gemv`).
    fn project_normed(
        &self,
        w: &AttnWeights<'_>,
        x: &Tensor,
        norm: Option<&RmsNorm>,
        rope_at: Option<(usize, &State<P>)>,
    ) -> Result<(Tensor, Tensor, Tensor, bool)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let (q, k, v) = match norm {
            Some(n) => (
                norm_gemv(x, n, &w.q, false)?,
                norm_gemv(x, n, &w.k, false)?,
                norm_gemv(x, n, &w.v, false)?,
            ),
            None => (w.q.forward(x)?, w.k.forward(x)?, w.v.forward(x)?),
        };

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
        cache: &mut State<P>,
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
            let Some(OwnKv::Quant(slabs)) = cache.own_mut() else {
                lloom_tensor::bail!("lfm2: the quantized KV is a `One` shape's store")
            };
            let slab = &mut slabs[block_idx];
            slab.append(&k, &v)?;

            if seq_len == 1 {
                return slab.attend(&q, None, 1f32 / (self.head_dim as f32).sqrt());
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
            match cache.own() {
                Some(OwnKv::Quant(slabs)) => slabs[block_idx].dequantize_to(q.dtype())?,
                _ => lloom_tensor::bail!("lfm2: the quantized KV is a `One` shape's store"),
            }
        } else {
            let device = cache.device.clone();
            let pending = std::mem::take(&mut cache.paged_pass_pending);
            match cache.kv.arm_mut() {
                ArmMut::One(OwnKv::Quant(_)) => {
                    lloom_tensor::bail!("lfm2: the quantized arm is read above")
                }
                ArmMut::One(OwnKv::Cat(kvs)) => {
                    let (k, v) = match &kvs[block_idx] {
                        Some((k_cache, v_cache)) if index_pos > 0 => {
                            let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                            let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                            (k, v)
                        }
                        _ => (k, v),
                    };
                    kvs[block_idx] = Some((k.clone(), v.clone()));
                    (k, v)
                }
                ArmMut::One(OwnKv::InPlace(slots)) => {
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
                    let (k_slot, v_slot) = &mut slots[block_idx];
                    // A turn boundary re-prefills without clearing, so this is
                    // a `seq_len > 1` append onto a non-empty slot. It is the
                    // same call — the offset simply advances by more than one.
                    let k = k_slot.append(&k)?;
                    let v = v_slot.append(&v)?;
                    (k, v)
                }
                ArmMut::Paged(pooled) => {
                    #[cfg(not(feature = "metal"))]
                    {
                        let _ = (block_idx, pooled, device, pending);
                        lloom_tensor::bail!("lfm2 `KvAppend::Paged` is Metal-only")
                    }
                    #[cfg(feature = "metal")]
                    {
                        let k = k.contiguous()?;
                        let v = v.contiguous()?;
                        // The FlashDecoding read walks the pool in place; any
                        // other read attends over a gathered copy (below).
                        let reads_pool =
                            seq_len == 1 && self.route.decode == DecodeAttn::FlashDecoding;
                        match pooled {
                            PooledKv::Shared(b) => {
                                // A row of a shared pool, attending alone: its own
                                // pass over its slot — the per-row path the
                                // batched arm (`attend_rows_shared`) is checked
                                // against, bit for bit.
                                if b_sz != 1 {
                                    lloom_tensor::bail!(
                                        "lfm2 batch row: a row's cache holds one sequence, got a \
                                         batch of {b_sz}"
                                    )
                                }
                                let pass = if pending {
                                    let slot = b.slot();
                                    let pass = Arc::new(
                                        b.pool
                                            .inner
                                            .lock()
                                            .kv
                                            .begin_rows(&[(slot, seq_len)], &device)?,
                                    );
                                    b.pass = Some(pass.clone());
                                    pass
                                } else {
                                    b.pass.clone().ok_or_else(|| {
                                        lloom_tensor::Error::msg(
                                            "lfm2 batch row: a layer ran before the pass opened",
                                        )
                                    })?
                                };
                                let pool = b.pool.clone();
                                let mut g = pool.inner.lock();
                                g.kv.write_rows(block_idx, &k, &v, &pass)?;
                                let n_prev = pass.prev[0];
                                if reads_pool {
                                    let (pk, pv) = g.kv.pools[block_idx]
                                        .as_ref()
                                        .expect("write_rows allocates the layer's pool");
                                    (pk.clone(), pv.clone())
                                } else if n_prev == 0 {
                                    (k, v)
                                } else {
                                    let slot = pass.rows[0].0;
                                    let (prev_k, prev_v) =
                                        g.kv.gather_row(block_idx, slot, n_prev, &device)?;
                                    let k = Tensor::cat(&[&prev_k, &k], 2)?.contiguous()?;
                                    let v = Tensor::cat(&[&prev_v, &v], 2)?.contiguous()?;
                                    (k, v)
                                }
                            }
                            PooledKv::Own(paged) => {
                                if pending {
                                    paged.begin_pass(b_sz, seq_len, &device)?;
                                }
                                let n_prev = paged.len - paged.pass_len;
                                // The write: this pass's tokens to the slots the table
                                // names, one dispatch for K and one for V.
                                paged.write_layer(block_idx, &k, &v)?;
                                if reads_pool {
                                    // Decode reads the pool through the table below;
                                    // the tensors handed on are the pool itself.
                                    let (pk, pv) = paged.pools[block_idx]
                                        .as_ref()
                                        .expect("write_layer allocates the layer's pool");
                                    (pk.clone(), pv.clone())
                                } else if n_prev == 0 {
                                    // A first pass attends over its own tokens.
                                    (k, v)
                                } else {
                                    // A turn-boundary prefill (§10.9), or a decode step
                                    // whose shape FlashDecoding does not instantiate,
                                    // attends over the accumulated cache: gathered from
                                    // the pool by physical slot, then this pass's
                                    // tokens. O(kv_len) per pass, and on the decode path
                                    // only on that fallback (`AttnRoute`).
                                    let (prev_k, prev_v) =
                                        paged.gather_layer(block_idx, b_sz, n_prev, &device)?;
                                    let k = Tensor::cat(&[&prev_k, &k], 2)?.contiguous()?;
                                    let v = Tensor::cat(&[&prev_v, &v], 2)?.contiguous()?;
                                    (k, v)
                                }
                            }
                            PooledKv::Off => {
                                lloom_tensor::bail!("lfm2 `KvAppend::Paged` with no page pool")
                            }
                        }
                    }
                }
            }
        };

        // The paged decode read. `KvAppend::Paged` requires
        // `AttnImpl::FlashDecoding` at construction, so this is the arm's decode
        // path whenever the route took FlashDecoding; off it, decode attended
        // over the gathered copy above, through `sdpa_vector` where that covers
        // the shape and the composed attention where nothing does. A dtype the
        // kernels do not instantiate is an error here rather than a silent
        // fallback to a kernel that cannot read a pool (§2.4).
        #[cfg(feature = "metal")]
        let paged_decode = cache.use_kv_cache
            && cache.kv_append == KvAppend::Paged
            && seq_len == 1
            && self.route.decode == DecodeAttn::FlashDecoding;
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
                let reserve_chunks = cache
                    .kv_capacity
                    .div_ceil((self.flash_page_size * self.flash_pages_per_chunk).max(1));
                if let Some(b) = cache.shared() {
                    let pass = b.pass.as_ref().ok_or_else(|| {
                        lloom_tensor::Error::msg("lfm2 batch row: reading before the pass opened")
                    })?;
                    crate::nn::ops::flash_decoding_paged_ragged(
                        &q,
                        &k,
                        &v,
                        1f32 / (self.head_dim as f32).sqrt(),
                        1.0,
                        self.flash_page_size,
                        self.flash_pages_per_chunk,
                        self.flash_scratch_sizing,
                        reserve_chunks,
                        pass.tables.clone(),
                        pass.table_stride,
                        pass.n_keys.clone(),
                    )?
                } else {
                    let paged = cache.paged().expect("checked above");
                    if let (Some(table), Some(count)) = (&paged.table_t, &paged.live_count) {
                        crate::nn::ops::flash_decoding_paged_ragged_bound(
                            &q,
                            &k,
                            &v,
                            1f32 / (self.head_dim as f32).sqrt(),
                            1.0,
                            self.flash_page_size,
                            self.flash_pages_per_chunk,
                            self.flash_scratch_sizing,
                            reserve_chunks,
                            &paged.table,
                            paged.table_stride,
                            &[paged.len as u32],
                            table,
                            count,
                        )?
                    } else {
                        crate::nn::ops::flash_decoding_paged_resident(
                            &q,
                            &k,
                            &v,
                            1f32 / (self.head_dim as f32).sqrt(),
                            1.0,
                            self.flash_page_size,
                            self.flash_pages_per_chunk,
                            self.flash_scratch_sizing,
                            reserve_chunks,
                            &paged.table,
                            paged.table_stride,
                            paged.len,
                            paged.table_t.as_ref(),
                        )?
                    }
                }
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
        } else if let Some(prepared) = &self.route.prepared {
            // The fused route prepared at load: one query takes the vector
            // kernel (two passes past its key threshold), or the tiled one at
            // a head dim no one-query kernel serves; a window takes the tiled
            // kernel, causal from the cache offset `kv_len - seq_len`. The
            // scale is applied to `q` inside the kernel, as `att /
            // sqrt(head_dim)` below. A request the route does not serve is an
            // error, never the composed attention.
            crate::nn::ops::attention(
                &q,
                &k,
                &v,
                None,
                1f32 / (self.head_dim as f32).sqrt(),
                prepared,
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
/// The attention block's decode step, the output projection unadded: three
/// projections, the QK norms and RoPE (or the fused pair), the two appends,
/// the decode kernel the route took, the output projection. The cache, the
/// appends and the attention over `kv_len` are layer `l`'s KV recurrence.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile> crate::optimizer::Visit for Attention<P> {
    type In = (crate::optimizer::NodeId, usize);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (xn, l): Self::In) -> Self::Out {
        use crate::optimizer::{Dim, Region, Shape};
        let (h, hd) = (self.hidden_size, self.head_dim);
        let (heads, kvh) = (self.num_attention_heads, self.num_key_value_heads);
        let kv = kvh * hd;
        let dt = self.o_proj.weight().dtype();
        let wq = w.weight("q_proj.weight", Shape::of(&[h, h]), dt);
        let wk = w.weight("k_proj.weight", Shape::of(&[kv, h]), dt);
        let wv = w.weight("v_proj.weight", Shape::of(&[kv, h]), dt);
        let q = w.gemv(wq, xn);
        let k = w.gemv(wk, xn);
        let v = w.gemv(wv, xn);
        let q = w.view("reshape", q, Shape::of(&[1, heads, 1, hd]));
        let k = w.view("reshape", k, Shape::of(&[1, kvh, 1, hd]));
        let v = w.view("reshape", v, Shape::of(&[1, kvh, 1, hd]));
        let qn_w = w.weight("q_layernorm.weight", Shape::of(&[hd]), dt);
        let kn_w = w.weight("k_layernorm.weight", Shape::of(&[hd]), dt);
        let cos = w.input(&format!("rope.cos@{l}"), Shape::of(&[1, hd / 2]), dt);
        let sin = w.input(&format!("rope.sin@{l}"), Shape::of(&[1, hd / 2]), dt);
        let (qr, kr) = match self.qkv_layout {
            QkvLayout::Contiguous | QkvLayout::Strided => {
                let qn = w.rms_norm(qn_w, q);
                let kn = w.rms_norm(kn_w, k);
                let qr = w.opaque("rope", &[qn, cos, sin], Shape::of(&[1, heads, 1, hd]), dt);
                let kr = w.opaque("rope", &[kn, cos, sin], Shape::of(&[1, kvh, 1, hd]), dt);
                (qr, kr)
            }
            QkvLayout::Fused => {
                let qr = w.opaque(
                    "rmsnorm_rope",
                    &[qn_w, q, cos, sin],
                    Shape::of(&[1, heads, 1, hd]),
                    dt,
                );
                let kr = w.opaque(
                    "rmsnorm_rope",
                    &[kn_w, k, cos, sin],
                    Shape::of(&[1, kvh, 1, hd]),
                    dt,
                );
                (qr, kr)
            }
        };
        let kv_shape = Shape(vec![
            Dim::Const(1),
            Dim::Const(kvh),
            Dim::Sym("kv_len"),
            Dim::Const(hd),
        ]);
        let k_cache = w.input(&format!("kv.k@{l}"), kv_shape.clone(), dt);
        let v_cache = w.input(&format!("kv.v@{l}"), kv_shape.clone(), dt);
        // The appends: in place, one strided write each; `Cat`, a
        // `Tensor::cat` of the cache and the row (two copies each); `Paged`,
        // the pool's scatter (named from the arm).
        let mut ring = vec![k_cache, v_cache];
        let mut append = |w: &mut crate::optimizer::Walker, row, cache| match self.kv_append {
            KvAppend::InPlace => w.opaque("copy2d", &[row, cache], kv_shape.clone(), dt),
            KvAppend::Cat => {
                let old = w.opaque("copy2d", &[cache], kv_shape.clone(), dt);
                ring.push(old);
                w.opaque("copy2d", &[row, old], kv_shape.clone(), dt)
            }
            KvAppend::Paged => w.opaque("flash_kv_scatter", &[row, cache], kv_shape.clone(), dt),
        };
        let k_all = append(w, kr, k_cache);
        let v_all = append(w, v, v_cache);
        ring.extend([k_all, v_all]);
        let kernel = match self.route.decode {
            DecodeAttn::SdpaVector => "sdpa_vector",
            DecodeAttn::Steel => "steel_attention",
            DecodeAttn::FlashDecoding => "flash_decoding",
            DecodeAttn::Generic => "attention_generic",
        };
        let out = Shape::of(&[1, heads, 1, hd]);
        // At a position whose keys reach the 2-pass length, the vector
        // kernel is two launches: the partials, then their merge.
        let two_pass = self.route.decode == DecodeAttn::SdpaVector
            && w.position()
                .is_some_and(|p| crate::nn::ops::sdpa_vector_two_pass(p + 1, hd, dt));
        let y = if two_pass {
            let partial = w.opaque("sdpa_vector_2pass_1", &[qr, k_all, v_all], out.clone(), dt);
            ring.push(partial);
            w.opaque("sdpa_vector_2pass_2", &[partial], out, dt)
        } else if self.route.decode == DecodeAttn::Generic {
            self.visit_generic(w, qr, k_all, v_all, dt, &mut ring)
        } else {
            w.opaque(kernel, &[qr, k_all, v_all], out, dt)
        };
        ring.push(y);
        w.mark(&ring, Region::Kv(l));
        let y = w.view("reshape", y, Shape::of(&[1, 1, h]));
        let wo = w.weight("out_proj.weight", Shape::of(&[h, h]), dt);
        w.gemv(wo, y)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile> Attention<P> {
    /// The generic arm's decode attention, composed: the KV heads expanded
    /// to the query heads (`repeat_kv`, a `cat` of `n_rep` copies each), all
    /// three read up to f32, the scores (a batched GEMV), the scale, the
    /// softmax, the product with `v` (the transposed GEMV), the result read
    /// back to the model's dtype. Every node that reads the cache's values
    /// joins `ring`, as the fused kernel does on the other arms; the query's
    /// own cast does not.
    fn visit_generic(
        &self,
        w: &mut crate::optimizer::Walker,
        q: crate::optimizer::NodeId,
        k: crate::optimizer::NodeId,
        v: crate::optimizer::NodeId,
        dt: DType,
        ring: &mut Vec<crate::optimizer::NodeId>,
    ) -> crate::optimizer::NodeId {
        use crate::optimizer::{Dim, Shape};
        let (heads, kvh, hd) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        );
        let rep = heads / kvh;
        let expanded = Shape(vec![
            Dim::Const(1),
            Dim::Const(heads),
            Dim::Sym("kv_len"),
            Dim::Const(hd),
        ]);
        let mut expand = |w: &mut crate::optimizer::Walker, x| {
            if rep == 1 {
                return x;
            }
            let mut c = w.opaque("copy2d", &[x], expanded.clone(), dt);
            ring.push(c);
            for _ in 1..rep {
                c = w.opaque("copy2d", &[x, c], expanded.clone(), dt);
                ring.push(c);
            }
            c
        };
        let k = expand(w, k);
        let v = expand(w, v);
        let up = |w: &mut crate::optimizer::Walker, x: crate::optimizer::NodeId| {
            if dt == DType::F32 {
                return x;
            }
            let shape = w.node(x).shape.clone();
            w.opaque("cast", &[x], shape, DType::F32)
        };
        let q = up(w, q);
        let (k, v) = (up(w, k), up(w, v));
        if dt != DType::F32 {
            ring.extend([k, v]);
        }
        let scores = Shape(vec![
            Dim::Const(1),
            Dim::Const(heads),
            Dim::Const(1),
            Dim::Sym("kv_len"),
        ]);
        let raw = w.opaque("gemv", &[q, k], scores, DType::F32);
        let scaled = w.scale(raw, (1.0 / (hd as f64).sqrt()) as f32);
        let att_shape = w.node(scaled).shape.clone();
        let att = w.opaque("softmax", &[scaled], att_shape, DType::F32);
        ring.extend([raw, scaled]);
        let y = w.opaque(
            "gemv_t",
            &[att, v],
            Shape::of(&[1, heads, 1, hd]),
            DType::F32,
        );
        ring.extend([att, y]);
        if dt == DType::F32 {
            return y;
        }
        let y = w.opaque("cast", &[y], Shape::of(&[1, heads, 1, hd]), dt);
        ring.push(y);
        y
    }
}

#[derive(Debug, Clone)]
struct ShortConv<P: Profile> {
    profile: PhantomData<P>,
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
    /// The engine's short-conv block over the `[hidden, l_cache]` taps, in the
    /// shift layout (`Shuffle`'s fused step).
    shift: crate::nn::short_conv::ShortConv<crate::nn::short_conv::Shift>,
    /// The same taps in the ring layout, for `ConvState::RotatingRing` (one
    /// permutation per phase, built at load time); `None` otherwise.
    ring: Option<crate::nn::short_conv::ShortConv<crate::nn::short_conv::Ring>>,
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
    *DISABLED.get_or_init(|| crate::tuning::tuning().lfm2.skip_conv_c_fusion)
}

impl<P: Profile> ShortConv<P> {
    /// `streamed`: the two projections live in the weight-group tier
    /// (§4.8); the fields are placeholders and `LayerWeights` carries the
    /// views. The taps are resident either way.
    fn new(cfg: &Config, vb: VarBuilder, streamed: bool) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let l_cache = cfg.conv_l_cache;

        // in_proj projects to 3 * hidden_size for B, C, X components
        let (in_proj, out_proj) = if streamed {
            (placeholder_linear(&vb)?, placeholder_linear(&vb)?)
        } else {
            (
                linear(hidden_size, 3 * hidden_size, vb.pp("in_proj"))?,
                linear(hidden_size, hidden_size, vb.pp("out_proj"))?,
            )
        };

        // Conv weight shape: (hidden_size, 1, l_cache) or (hidden_size, l_cache)
        let conv_weight = vb.get((hidden_size, 1, l_cache), "conv.weight")?;

        // The rotating arm's per-phase weights, precomputed at load time: a
        // rotating buffer's slot holds a different token's input each phase,
        // so the taps are permuted to match or the fold is a different
        // operator (`DESIGN.md` §10.2g). All `width` permutations cost nothing
        // per token.
        use crate::nn::short_conv::{Ring, Shift, ShortConv};
        let taps = conv_weight.squeeze(1)?;
        let ring = match cfg.conv_state {
            ConvState::RotatingRing { .. } => Some(ShortConv::<Ring>::new(
                taps.clone(),
                cfg.conv_state.width(l_cache),
            )?),
            _ => None,
        };
        let shift = ShortConv::<Shift>::new(taps, l_cache);

        Ok(Self {
            profile: PhantomData,
            in_proj,
            out_proj,
            conv_weight,
            shift,
            ring,
            l_cache,
            hidden_size,
            conv_state: cfg.conv_state,
            gemv_residual: cfg.gemv_residual,
            conv_fused: cfg.conv_fused,
            prefill_ctx: cfg.conv_prefill_ctx,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    /// The ring columns past the live window a prefill leaves, set for what
    /// decode reads of them (`pass-plan-design.md` §18). The sliding ring
    /// reads no slot outside its window before writing it, so its tail is
    /// left as it is. The rotating ring folds every slot, the history against
    /// a zero weight, so its tail must be finite, and it is zeroed: that is
    /// also what keeps a rewind from reaching a slot this prefill did not
    /// write (`sequence-state.md` §10.2a).
    fn ring_regions_after_prefill(&self, state: &Tensor) -> Result<()> {
        let width = self.conv_state.width(self.l_cache);
        if width == self.l_cache {
            return Ok(());
        }
        let tail = state.narrow(2, self.l_cache, width - self.l_cache)?;
        match self.conv_state {
            ConvState::RotatingRing { .. } => zs::zero_region(zs::LFM2_CONV_RING_HISTORY, &tail),
            _ => zs::poison_region(zs::LFM2_CONV_RING_SLACK, &tail),
        }
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
        next: Option<&Tensor>,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !self.conv_fused || conv_c_fusion_disabled() {
            return Ok(None);
        }
        // Metal-only, same reason as `conv_fold_c`: the `shortconv_step` family
        // does not compile off Metal. Fall back to the unfused Shuffle arm.
        #[cfg(not(feature = "metal"))]
        {
            let _ = (b, x, c, state, next);
            return Ok(None);
        }
        #[cfg(feature = "metal")]
        {
            if !state.device().is_metal() {
                return Ok(None);
            }
            let k = state.dims3()?.2;
            if k != self.shift.k()
                || lloom_tensor::kernels::ConvKernel::shortconv_step_k(k).is_none()
            {
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
            Ok(Some(match next {
                Some(n) => (self.shift.gated_step_into(b, x, c, state, n)?, n.clone()),
                None => self.shift.gated_step(b, x, c, state)?,
            }))
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
            let Some(ring) = &self.ring else {
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
            Ok(Some(ring.gated_pass(b, x, c, state, phase_last)?))
        }
    }

    fn resident(&self) -> ConvWeights<'_> {
        use std::borrow::Cow;
        ConvWeights {
            in_proj: Cow::Borrowed(&self.in_proj),
            out_proj: Cow::Borrowed(&self.out_proj),
        }
    }

    fn forward(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let conv_out = self.convolve(w, x, block_idx, cache)?;
        w.out_proj.forward(&conv_out)
    }

    /// `conv(x) + residual`, fusing the add onto `out_proj`'s store when the
    /// arm is on (§12.2h). Otherwise **literally `forward(..)? + residual`** —
    /// the unfused branch calls `forward`, so the shipping path has exactly one
    /// definition and the two cannot drift.
    fn forward_add(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        cache: &mut State<P>,
        residual: &Tensor,
    ) -> Result<Tensor> {
        if self.gemv_residual {
            let conv_out = self.convolve(w, x, block_idx, cache)?;
            if gemv_residual_applies(true, &conv_out, &w.out_proj, residual) {
                return crate::nn::ops::gemv_residual(&conv_out, w.out_proj.weight(), residual);
            }
            return w.out_proj.forward(&conv_out)? + residual;
        }
        self.forward(w, x, block_idx, cache)? + residual
    }

    /// [`Self::forward_add`] over the un-normalized `x`, `norm` riding
    /// `in_proj`'s GEMV.
    fn forward_add_normed(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        norm: &RmsNorm,
        block_idx: usize,
        cache: &mut State<P>,
        residual: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bcx = norm_gemv(x, norm, &w.in_proj, false)?.transpose(1, 2)?;
        let conv_out = self.convolve_bcx(&bcx, block_idx, cache)?;
        if self.gemv_residual && gemv_residual_applies(true, &conv_out, &w.out_proj, residual) {
            return crate::nn::ops::gemv_residual(&conv_out, w.out_proj.weight(), residual);
        }
        w.out_proj.forward(&conv_out)? + residual
    }

    /// [`Self::forward_add`] for one step of `rows.len()` independent
    /// sequences (`BatchedCausalLM`): `in_proj` runs once over
    /// `[B, 1, hidden]`, each row convolves against its own state. The
    /// residual arm is decided as `forward_add` decides it.
    fn forward_rows_add(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let conv_out = self.convolve_rows(w, x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &conv_out, &w.out_proj, residual) {
            return crate::nn::ops::gemv_residual(&conv_out, w.out_proj.weight(), residual);
        }
        w.out_proj.forward(&conv_out)? + residual
    }

    /// Everything up to but excluding `out_proj`, so the two entry points
    /// above share one body and cannot drift in what they compute.
    fn convolve(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        // Project input to B, C, X components
        let bcx = w.in_proj.forward(x)?.transpose(1, 2)?;
        self.convolve_bcx(&bcx, block_idx, cache)
    }

    /// `convolve` over independent rows: one `in_proj` for the batch, then
    /// row `r` takes the whole conv block against `rows[r].cache`. A row of
    /// the projected `[B, 3·hidden, 1]` has the strides a batch of one has, so
    /// every arm below sees the layout it was written for.
    fn convolve_rows(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bcx = w.in_proj.forward(x)?.transpose(1, 2)?;
        #[cfg(feature = "metal")]
        if let Some(out) = self.convolve_rows_shared(&bcx, block_idx, rows)? {
            return Ok(out);
        }
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
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
        residual: &Tensor,
    ) -> Result<Tensor> {
        let conv_out = self.convolve_ragged(w, x, block_idx, rows)?;
        if self.gemv_residual && gemv_residual_applies(true, &conv_out, &w.out_proj, residual) {
            return crate::nn::ops::gemv_residual(&conv_out, w.out_proj.weight(), residual);
        }
        w.out_proj.forward(&conv_out)? + residual
    }

    /// `convolve` over a ragged batch: one `in_proj`, then row `r` takes the
    /// whole conv block over its `len` tokens against `rows[r].cache`. A
    /// row's `[1, 3·hidden, len]` is the transposed narrow a pass of `len`
    /// tokens sees (for `len == 1`, the strides a batch of one has — the
    /// fused decode arms' layout), so every arm below sees what it was
    /// written for.
    fn convolve_ragged(
        &self,
        w: &ConvWeights<'_>,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bcx = w.in_proj.forward(x)?;
        #[cfg(feature = "metal")]
        if let Some(out) = self.convolve_ragged_shared(&bcx, block_idx, rows)? {
            return Ok(out);
        }
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

    /// Whether the rows are slots of one pool the slot-indexed ring kernel
    /// can write in one dispatch (§12.2): the fused `RotatingRing` arm, every
    /// row on the same pool, none speculating.
    #[cfg(feature = "metal")]
    fn rows_share_a_ring_pool<'a>(&self, caches: impl Iterator<Item = &'a State<P>>) -> bool {
        if !matches!(self.conv_state, ConvState::RotatingRing { .. })
            || !self.conv_fused
            || conv_c_fusion_disabled()
            || self.ring.is_none()
        {
            return false;
        }
        let mut first: Option<&State<P>> = None;
        for c in caches {
            if c.shared().is_none() || !c.use_kv_cache {
                return false;
            }
            match first {
                None => first = Some(c),
                Some(f) => {
                    if !f.shares_pool_with(c) {
                        return false;
                    }
                }
            }
        }
        match first {
            Some(f) => f.shared().expect("checked").pool.inner.lock().batched_arms,
            None => false,
        }
    }

    /// The ring write for `caches`' rows — `[B, 3·hidden, 1]` projections
    /// laid out row by row — in one dispatch over the pool's rings, each row
    /// at its own slot and phase. Returns `[B, 1, hidden]` with `*C` applied.
    #[cfg(feature = "metal")]
    fn ring_rows_shared(
        &self,
        bcx: &Tensor,
        block_idx: usize,
        caches: &mut [&mut State<P>],
    ) -> Result<Tensor> {
        let width = self.conv_state.width(self.l_cache);
        let dtype = bcx.dtype();
        let device = bcx.device().clone();
        let mut slots = Vec::with_capacity(caches.len());
        let mut phases = Vec::with_capacity(caches.len());
        for c in caches.iter_mut() {
            // Bind (and on a fresh lease zero) the row's ring, so the pool's
            // row is the state this row's every kernel reads.
            c.conv_state_view(block_idx, self.hidden_size, width, dtype)?;
            slots.push(c.shared().expect("checked").slot() as u32);
            phases.push(c.conv_phase as u32);
        }
        let pool = caches[0].shared().expect("checked").pool.clone();
        let (pool_t, slots_t, phases_t) = {
            let mut g = pool.inner.lock();
            g.shared_calls += 1;
            let pool_t = g.conv[block_idx]
                .clone()
                .expect("conv_state_view allocated the layer's pool");
            let fresh = match &g.row_index {
                Some(ix) => ix.slots != slots || ix.phases != phases,
                None => true,
            };
            if fresh {
                g.row_index = Some(RowIndex {
                    slots: slots.clone(),
                    phases: phases.clone(),
                    slots_t: Tensor::new(slots.as_slice(), &device)?,
                    phases_t: Tensor::new(phases.as_slice(), &device)?,
                });
            }
            let ix = g.row_index.as_ref().expect("set above");
            (pool_t, ix.slots_t.clone(), ix.phases_t.clone())
        };
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;
        let ring = self.ring.as_ref().expect("checked");
        ring.gated_rows(&b, &x, &c, &pool_t, &slots_t, &phases_t)
    }

    /// `convolve_rows` in one dispatch (§12.2). `bcx` is `[B, 3·hidden, 1]`.
    #[cfg(feature = "metal")]
    fn convolve_rows_shared(
        &self,
        bcx: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Option<Tensor>> {
        if !self.rows_share_a_ring_pool(rows.iter().map(|r| &*r.cache)) {
            return Ok(None);
        }
        let mut caches: Vec<&mut State<P>> = rows.iter_mut().map(|r| &mut *r.cache).collect();
        Ok(Some(self.ring_rows_shared(bcx, block_idx, &mut caches)?))
    }

    /// `convolve_ragged` over a shared pool: the one-token rows' ring writes
    /// in one dispatch, each longer row (a prompt piece) through the
    /// multi-token writer on its own ring. `bcx` is `[1, T, 3·hidden]`.
    #[cfg(feature = "metal")]
    fn convolve_ragged_shared(
        &self,
        bcx: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Option<Tensor>> {
        if !self.rows_share_a_ring_pool(rows.iter().map(|r| &*r.cache)) {
            return Ok(None);
        }
        let device = bcx.device().clone();
        let mut offs = Vec::with_capacity(rows.len());
        let mut off = 0usize;
        for r in rows.iter() {
            offs.push(off);
            off += r.len;
        }
        let dec: Vec<usize> = (0..rows.len()).filter(|&r| rows[r].len == 1).collect();
        let mut out_dec: Option<Tensor> = None;
        if !dec.is_empty() {
            let nd = dec.len();
            let prefix = dec.iter().enumerate().all(|(i, &r)| offs[r] == i);
            let bcx_d = if prefix {
                bcx.narrow(1, 0, nd)?
            } else {
                let idx: Vec<u32> = dec.iter().map(|&r| offs[r] as u32).collect();
                bcx.index_select(&Tensor::new(idx.as_slice(), &device)?, 1)?
            };
            // `[1, nd, 3·hidden]` contiguous is `[nd, 3·hidden, 1]` with the
            // strides a batch of one-token rows has.
            let bcx_d = bcx_d.contiguous()?.reshape((nd, 3 * self.hidden_size, 1))?;
            let mut caches: Vec<&mut State<P>> = Vec::with_capacity(nd);
            let mut taken: Vec<Option<&mut RaggedRow<'_, State<P>>>> =
                rows.iter_mut().map(Some).collect();
            for &r in &dec {
                caches.push(&mut *taken[r].take().expect("each row once").cache);
            }
            out_dec = Some(self.ring_rows_shared(&bcx_d, block_idx, &mut caches)?);
        }
        let mut outs = Vec::with_capacity(rows.len());
        let mut j = 0usize;
        for (r, row) in rows.iter_mut().enumerate() {
            if row.len == 1 {
                outs.push(
                    out_dec
                        .as_ref()
                        .expect("a decode row was written")
                        .narrow(0, j, 1)?,
                );
                j += 1;
            } else {
                let bcx_r = bcx.narrow(1, offs[r], row.len)?.transpose(1, 2)?;
                outs.push(self.convolve_bcx(&bcx_r, block_idx, row.cache)?);
            }
        }
        if outs.len() == 1 {
            return Ok(Some(outs.remove(0)));
        }
        Ok(Some(Tensor::cat(&outs, 1)?))
    }

    /// The conv block after `in_proj`: `bcx` is `[b, 3·hidden, seq]` and the
    /// result `[b, seq, hidden]` with the `*C` gate applied.
    fn convolve_bcx(&self, bcx: &Tensor, block_idx: usize, cache: &mut State<P>) -> Result<Tensor> {
        let (b_sz, _, seq_len) = bcx.dims3()?;
        // A batch row's ring is a row of the pool: bound before any arm below
        // reads `conv_states`, so what they read and write is the pool's row.
        #[cfg(feature = "metal")]
        if cache.shared().is_some() && cache.use_kv_cache {
            let width = self.conv_state.width(self.l_cache);
            cache.conv_state_view(block_idx, self.hidden_size, width, bcx.dtype())?;
        }
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
                None => zs::zeros(
                    zs::LFM2_CONV_FIRST_STATE,
                    (b_sz, self.hidden_size, self.l_cache),
                    c.dtype(),
                    c.device(),
                )?,
            };
            // A cache that keeps the state writes it into the layer's spare
            // and swaps the two; the batch arm's state is a pool row, copied
            // into by `set_conv_state`, so it keeps a fresh one. Asked for
            // only on the fused arm, and handed back when the step declines:
            // the unfused step never takes one, so it allocates none a token.
            let next = if self.conv_fused
                && !conv_c_fusion_disabled()
                && cache.use_kv_cache
                && cache.batch_slot().is_none()
            {
                Some(cache.conv_spare_for(block_idx, &state)?)
            } else {
                None
            };
            match self.shortconv_fused(&b, &x_proj, &c, &state, next.as_ref())? {
                Some((conv_out, new_state)) => {
                    if cache.use_kv_cache {
                        if next.is_some() {
                            cache.conv_spare[block_idx] = Some(state);
                        }
                        cache.set_conv_state(block_idx, new_state)?;
                    }
                    // `c` is already applied in the kernel store; only the
                    // transpose to `[b, seq, hidden]` for `out_proj` remains.
                    return conv_out.transpose(1, 2)?.contiguous();
                }
                None => {
                    if let Some(spare) = next {
                        cache.conv_spare[block_idx] = Some(spare);
                    }
                }
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
                None => zs::zeros(
                    zs::LFM2_CONV_FIRST_STATE,
                    (b_sz, self.hidden_size, width),
                    c.dtype(),
                    c.device(),
                )?,
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
                    cache.set_conv_state(block_idx, state)?;
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
                     (`DESIGN.md` §10.2a). `State::<P>::advance` refuses the other arms."
                )
            };
            let width = self.conv_state.width(self.l_cache);
            let state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => zs::zeros(
                    zs::LFM2_CONV_FIRST_STATE,
                    (b_sz, self.hidden_size, width),
                    bx.dtype(),
                    bx.device(),
                )?,
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
                    .ring
                    .as_ref()
                    .expect("the ring's weights exist whenever the arm is RotatingRing")
                    .phase_weight(phase);
                // The same fold as the decode arm, per position: `conv_fold`
                // honours `--conv-fused` here too, so a speculative verify runs
                // the same conv kernel a decode of that token would.
                outs.push(self.conv_fold(&state, w)?);
            }

            if cache.use_kv_cache {
                cache.set_conv_state(block_idx, state.clone())?;
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
                        None => zs::zeros(
                            zs::LFM2_CONV_FIRST_STATE,
                            (b_sz, self.hidden_size, width),
                            bx.dtype(),
                            bx.device(),
                        )?,
                    };

                    // Advanced once per token by `Model::forward`, for the same
                    // reason the sliding arm's is: 22 conv layers share one index.
                    let phase = cache.conv_phase;
                    state.slice_set(&bx, 2, phase)?;

                    if cache.use_kv_cache {
                        cache.set_conv_state(block_idx, state.clone())?;
                    }

                    // Built at load time, one per phase -- so the per-token cost
                    // is an index rather than a permutation (§15.2 #10: the
                    // answer is computed when the plan is built, not per token).
                    let w = self
                        .ring
                        .as_ref()
                        .expect("the ring's weights are built whenever the arm is RotatingRing")
                        .phase_weight(phase);
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
                        None => zs::zeros(
                            zs::LFM2_CONV_FIRST_STATE,
                            (b_sz, self.hidden_size, width),
                            bx.dtype(),
                            bx.device(),
                        )?,
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
                        let pad = zs::uninit(
                            zs::LFM2_CONV_COMPACT_PAD,
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
                        cache.set_conv_state(block_idx, state.clone())?;
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
                    // The shift is written INTO the storage the cache owns, as
                    // the prefill writes it, never into a fresh tensor swapped
                    // in: a step gated past an EOS must leave the state as it
                    // was, and it skips writes, not swaps. At `l_cache > 1`
                    // both writes are strided (a window narrower than the
                    // state), so they are the gated `copy2d`, never a blit.
                    // (`docs/dispatch.md` §11.5z-x)
                    let state = match &cache.conv_states[block_idx] {
                        Some(s) => s.contiguous()?,
                        None => {
                            let _declared = lloom_tensor::EscapeGuard::new();
                            zs::zeros(
                                zs::LFM2_CONV_FIRST_STATE,
                                (b_sz, self.hidden_size, self.l_cache),
                                bx.dtype(),
                                bx.device(),
                            )?
                        }
                    };
                    if self.l_cache > 1 {
                        let tail = self.window_tail(&state, cache.conv_phase)?;
                        state.slice_set(&tail, 2, 0)?;
                    }
                    state.slice_set(&bx, 2, self.l_cache - 1)?;

                    if cache.use_kv_cache {
                        cache.set_conv_state(block_idx, state.clone())?;
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
            // A private branching cache enables continuation before its first
            // trunk tile without changing the loaded model/chat flag.
            let ctx = match &cache.conv_states[block_idx] {
                Some(prev)
                    if (self.prefill_ctx || cache.branch_continuation)
                        && cache.use_kv_cache
                        && self.l_cache > 1 =>
                {
                    Some(self.window_tail(prev, cache.conv_phase_prev)?)
                }
                _ => None,
            };
            let (conv_in, padding) = match &ctx {
                Some(ctx) => (Tensor::cat(&[ctx, &bx], 2)?.contiguous()?, 0),
                None => (bx.clone(), self.l_cache.saturating_sub(1)),
            };
            // The pass's turn checkpoint keeps this layer's newest input
            // columns, so a later request diverging just before this pass's
            // end can be put back there (`Cache::restore`).
            if cache.use_kv_cache && self.l_cache > 0 {
                cache.turn_checkpoint_tail(block_idx, &conv_in, self.l_cache)?;
            }
            // A batch row's piece crossing a page boundary leaves the ring
            // after that page's last token for a sharer (§12.3).
            #[cfg(feature = "metal")]
            if cache.shared().is_some() && cache.use_kv_cache && b_sz == 1 {
                let ctx_len = ctx.as_ref().map(|c| c.dim(2)).transpose()?.unwrap_or(0);
                let width = self.conv_state.width(self.l_cache);
                cache.stage_piece_snapshots(
                    block_idx,
                    &conv_in,
                    ctx_len,
                    seq_len,
                    self.l_cache,
                    width,
                    cache.conv_states.len(),
                )?;
            }
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
                let carried = conv_in.narrow(2, start, cache_len)?.contiguous()?;
                // The state OUTLIVES this pass, so the carry is written INTO
                // the storage the cache owns (a cache with none yet allocates
                // it under an `EscapeGuard`, which no plan serves), and the
                // regions around it are set in place: nothing this tile
                // allocates escapes it, and nothing is allocated to pad it.
                // The live window sits at the front, the newest column at
                // `l_cache - 1` (what `Model::forward` seeds). What each
                // region must hold is `nn::zeros`'s classification
                // (`pass-plan-design.md` §18).
                let width = self.conv_state.width(self.l_cache);
                let pad = self.l_cache - cache_len;
                let owned = match &cache.conv_states[block_idx] {
                    Some(prev)
                        if prev.dims() == [b_sz, self.hidden_size, width]
                            && prev.dtype() == carried.dtype()
                            && prev.is_contiguous() =>
                    {
                        prev.clone()
                    }
                    _ => {
                        // Not `Tensor::copy`: on Metal `try_clone` shares
                        // the buffer, and the storage must be this cache's own.
                        let _declared = lloom_tensor::EscapeGuard::new();
                        zs::zeros(
                            zs::LFM2_CONV_STATE,
                            (b_sz, self.hidden_size, width),
                            carried.dtype(),
                            carried.device(),
                        )?
                    }
                };
                owned.slice_set(&carried, 2, pad)?;
                if pad > 0 {
                    zs::zero_region(zs::LFM2_CONV_LEFT_PAD, &owned.narrow(2, 0, pad)?)?;
                }
                self.ring_regions_after_prefill(&owned)?;
                cache.set_conv_state(block_idx, owned)?;
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
/// The short-conv block's decode step, the output projection unadded, at
/// the state arm and fold the block was built with: the fused kernels as one
/// node each, the unfused arms as their copies, products and fold. The state
/// window, whatever writes the next window and the fold over it are layer
/// `l`'s conv recurrence; the `*c` gate reads the fold's result and is
/// straight-line.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile> crate::optimizer::Visit for ShortConv<P> {
    type In = (crate::optimizer::NodeId, usize);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (xn, l): Self::In) -> Self::Out {
        use crate::optimizer::{Dim, NodeId, Region, Shape};
        let (h, lc) = (self.hidden_size, self.l_cache);
        let width = self.conv_state.width(lc);
        let dt = self.conv_weight.dtype();
        let w_in = w.weight("in_proj.weight", Shape::of(&[3 * h, h]), dt);
        let bcx = w.gemv(w_in, xn);
        let bcx = w.view("transpose", bcx, Shape::of(&[1, 3 * h, 1]));
        let b = w.view("narrow", bcx, Shape::of(&[1, h, 1]));
        let c = w.view("narrow", bcx, Shape::of(&[1, h, 1]));
        let x = w.view("narrow", bcx, Shape::of(&[1, h, 1]));
        let w_conv = w.weight("conv.weight", Shape::of(&[h, 1, lc]), dt);
        let w_conv = w.view("squeeze", w_conv, Shape::of(&[h, lc]));
        let state = w.input(&format!("conv.state@{l}"), Shape::of(&[1, h, width]), dt);
        let mut ring: Vec<NodeId> = vec![state];
        let out = match (self.conv_state, self.conv_fused) {
            (ConvState::Shuffle, true) => {
                let step = w.opaque(
                    "shortconv_step",
                    &[b, x, c, state, w_conv],
                    Shape::of(&[1, h, 1]),
                    dt,
                );
                ring.push(step);
                step
            }
            // The ring writer's output arrives transposed with `*c` applied.
            (ConvState::RotatingRing { .. }, true) => {
                let out = w.opaque(
                    "shortconv_ring",
                    &[b, x, c, state, w_conv],
                    Shape::of(&[1, 1, h]),
                    dt,
                );
                ring.push(out);
                w.mark(&ring, Region::ConvRing(l));
                let w_out = w.weight("out_proj.weight", Shape::of(&[h, h]), dt);
                return w.gemv(w_out, out);
            }
            (arm, fused) => {
                let bx = w.mul(b, x);
                let window = match arm {
                    ConvState::Shuffle => {
                        // The shift in place (`window_tail`, `docs/dispatch.md`
                        // §11.5z-x): the tail gathered to a scratch (a strided
                        // copy), written back to the first columns, `bx` to the
                        // last — two strided writes into the state.
                        let tail = w.view("narrow", state, Shape::of(&[1, h, lc - 1]));
                        let gathered =
                            w.opaque("copy_strided", &[tail], Shape::of(&[1, h, lc - 1]), dt);
                        let shifted =
                            w.opaque("copy2d", &[gathered, state], Shape::of(&[1, h, lc]), dt);
                        let next = w.opaque("copy2d", &[bx, shifted], Shape::of(&[1, h, lc]), dt);
                        ring.extend([tail, gathered, shifted, next]);
                        next
                    }
                    ConvState::RotatingRing { .. } | ConvState::SlidingRing { .. } => {
                        // `slice_set` at the write slot: one copy into the ring.
                        let written =
                            w.opaque("copy2d", &[bx, state], Shape::of(&[1, h, width]), dt);
                        ring.push(written);
                        if matches!(arm, ConvState::SlidingRing { .. }) {
                            let live = w.view("narrow", written, Shape::of(&[1, h, lc]));
                            ring.push(live);
                            live
                        } else {
                            written
                        }
                    }
                };
                let taps = match w.node(window).shape.last() {
                    Some(Dim::Const(t)) => t,
                    _ => lc,
                };
                let conv = if fused {
                    let wt = w.view("unsqueeze", w_conv, Shape::of(&[h, 1, taps]));
                    let conv =
                        w.opaque("conv1d_depthwise", &[window, wt], Shape::of(&[1, h, 1]), dt);
                    ring.push(conv);
                    conv
                } else {
                    let wt = w.view("unsqueeze", w_conv, Shape::of(&[1, h, taps]));
                    let prod = w.mul(window, wt);
                    let sum = w.opaque("fast_sum", &[prod], Shape::of(&[1, h, 1]), dt);
                    ring.extend([prod, sum]);
                    sum
                };
                w.mul(c, conv)
            }
        };
        w.mark(&ring, Region::ConvRing(l));
        let out = w.view("transpose", out, Shape::of(&[1, 1, h]));
        let w_out = w.weight("out_proj.weight", Shape::of(&[h, h]), dt);
        w.gemv(w_out, out)
    }
}

#[derive(Debug, Clone)]
enum LayerKind<P: Profile> {
    Attention(Box<Attention<P>>),
    ShortConv(ShortConv<P>),
}

#[derive(Debug, Clone)]
struct DecoderLayer<P: Profile> {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind<P>,
    /// Whether the layer's two norms may run inside their consumers' GEMVs
    /// (`nn::ops::gemv_norm`): `LLOOM_LFM2_NORM_GEMV` at construction, or a
    /// derived plan's verdict (`Fusion=Derived`).
    norm_gemv: bool,
    /// `Some`: this layer's matrices stream through the tier (§4.8).
    streamed: Option<StreamedLayer>,
    span: tracing::Span,
}

/// A 1x1 stand-in for a matrix that lives in the weight-group tier.
fn placeholder_linear(vb: &VarBuilder) -> Result<Linear> {
    Ok(Linear::from_weights(
        Tensor::zeros((1, 1), vb.dtype(), vb.device())?,
        None,
    ))
}

/// The parts of layer `l`'s weight group, in `GROUP_PART_*` order: the
/// MLP's `w1`, `w3`, `w2`, then the mixer's projections — `q`, `k`, `v`,
/// `out` for an attention layer, `in`, `out` for a conv layer. The
/// checkpoint names and shapes, as `nn::weight_groups::GroupSpec` takes them.
pub fn layer_group_parts(cfg: &Config, l: usize) -> Vec<(String, lloom_tensor::Shape)> {
    use lloom_tensor::Shape;
    let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
    let mut parts = vec![
        (
            format!("model.layers.{l}.feed_forward.w1.weight"),
            Shape::from((i, h)),
        ),
        (
            format!("model.layers.{l}.feed_forward.w3.weight"),
            Shape::from((i, h)),
        ),
        (
            format!("model.layers.{l}.feed_forward.w2.weight"),
            Shape::from((h, i)),
        ),
    ];
    let kind = cfg
        .layer_types
        .get(l)
        .copied()
        .unwrap_or(LayerType::FullAttention);
    match kind {
        LayerType::FullAttention => {
            let (q, kv) = (
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.num_key_value_heads * cfg.head_dim(),
            );
            parts.push((
                format!("model.layers.{l}.self_attn.q_proj.weight"),
                Shape::from((q, h)),
            ));
            parts.push((
                format!("model.layers.{l}.self_attn.k_proj.weight"),
                Shape::from((kv, h)),
            ));
            parts.push((
                format!("model.layers.{l}.self_attn.v_proj.weight"),
                Shape::from((kv, h)),
            ));
            parts.push((
                format!("model.layers.{l}.self_attn.out_proj.weight"),
                Shape::from((h, q)),
            ));
        }
        LayerType::Conv => {
            parts.push((
                format!("model.layers.{l}.conv.in_proj.weight"),
                Shape::from((3 * h, h)),
            ));
            parts.push((
                format!("model.layers.{l}.conv.out_proj.weight"),
                Shape::from((h, h)),
            ));
        }
    }
    parts
}

/// The bytes of layer `l`'s weight group at `dtype` — what one slot of the
/// tier holds for it. The loader's arithmetic (how many slots the quota
/// leaves) reads this rather than its own copy of the shapes.
pub fn layer_group_bytes(cfg: &Config, l: usize, dtype: DType) -> usize {
    layer_group_parts(cfg, l)
        .iter()
        .map(|(_, s)| s.elem_count())
        .sum::<usize>()
        * dtype.size_in_bytes()
}

/// `LLOOM_LFM2_NORM_GEMV`: a decode step's RMSNorm runs inside the GEMVs
/// that consume it (`nn::ops::gemv_norm`). Read once.
fn norm_gemv_arm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().lfm2.norm_gemv)
}

/// Whether `rmsnorm(x)` may be computed inside a GEMV over `proj`: the
/// block's switch (`on`), Metal, one contiguous row, a dtype the kernel carries, the norm's weight a
/// contiguous `[k]` of `x`'s dtype, and a tile the fused kernel is
/// instantiated at (`nn::ops::gemv_norm_serves`: at `k <= 64` or
/// `k >= 16 n` the GEMV takes another tile, and the unfused pair runs).
fn norm_gemv_applies(on: bool, x: &Tensor, norm: &RmsNorm, proj: &Linear) -> bool {
    on && x.device().is_metal()
        && matches!(x.dims(), [1, 1, _] | [1, _])
        && x.dims().last() == Some(&proj.weight().dims()[1])
        && crate::nn::ops::gemv_norm_serves(proj.weight().dims()[0], proj.weight().dims()[1])
        && proj.bias().is_none()
        && x.is_contiguous()
        && proj.weight().is_contiguous()
        && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
        && norm.weight().dtype() == x.dtype()
        && norm.weight().is_contiguous()
        && norm.weight().elem_count() == proj.weight().dims()[1]
}

/// `proj(rmsnorm(x))` (and its SiLU) in one dispatch.
fn norm_gemv(x: &Tensor, norm: &RmsNorm, proj: &Linear, silu: bool) -> Result<Tensor> {
    crate::nn::ops::gemv_norm(x, proj.weight(), norm.weight(), norm.eps() as f32, silu)
}

/// `LLOOM_LFM2_ABLATE` names `norm`: the decode step skips the layers'
/// RMSNorms (an instrument pricing a fusion's upper bound; the output is
/// wrong). Read once.
fn ablate_norms() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::tuning::tuning()
            .lfm2
            .ablate
            .split(',')
            .any(|s| s.trim() == "norm")
    })
}

impl<P: Profile> DecoderLayer<P> {
    /// The layer's RMSNorm, or `x` itself under the `norm` ablation at a
    /// one-token step.
    fn norm(&self, n: &RmsNorm, x: &Tensor) -> Result<Tensor> {
        if ablate_norms() && x.dim(x.rank() - 2)? == 1 {
            return Ok(x.clone());
        }
        n.forward(x)
    }

    fn new(
        cfg: &Config,
        layer_idx: usize,
        route: AttnRoute,
        vb: VarBuilder,
        streamed: Option<StreamedLayer>,
    ) -> Result<Self> {
        // LFM2 uses operator_norm and ffn_norm naming
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        // LFM2 uses feed_forward naming for MLP; on the weight-group tier
        // (§4.8) the layer's three matrices are its group's parts.
        let mlp = if streamed.is_some() {
            Mlp::new_placeholder(cfg, &vb)?
        } else {
            Mlp::new(cfg, vb.pp("feed_forward"))?
        };

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => LayerKind::Attention(Box::new(Attention::new(
                cfg,
                route,
                vb.pp("self_attn"),
                streamed.is_some(),
            )?)),
            LayerType::Conv => {
                LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"), streamed.is_some())?)
            }
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            kind,
            norm_gemv: norm_gemv_arm(),
            streamed,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    /// This layer's matrices for the pass: borrowed when resident. When
    /// streamed, the group is made resident in the tier (`ensure`), its
    /// parts taken as views over the slot, and the next `depth` groups of
    /// the plan prefetched — once per layer per pass, at the layer's entry,
    /// before the mixer. A group that moved slots since its last pass
    /// drops the cached pass plans (R2: a plan binds slots; the old one
    /// bound another).
    fn weights(&self) -> Result<LayerWeights<'_>> {
        use std::borrow::Cow;
        let Some(st) = &self.streamed else {
            let mixer = match &self.kind {
                LayerKind::Attention(a) => MixerWeights::Attn(a.resident()),
                LayerKind::ShortConv(c) => MixerWeights::Conv(c.resident()),
            };
            return Ok(LayerWeights {
                mixer,
                mlp: self.mlp.resident(),
            });
        };
        let mut tier = st.tier.lock();
        let (_slot, moved, wait) = tier.ensure(st.group, st.source.as_ref())?;
        tier.cut_after_ensure()?;
        let part = |i: usize| -> Result<Cow<'static, Linear>> {
            Ok(Cow::Owned(Linear::from_weights(
                tier.tensor(st.group, i)?,
                None,
            )))
        };
        let mlp = Projections {
            gate: part(GROUP_PART_W1)?,
            up: part(GROUP_PART_W3)?,
            down: part(GROUP_PART_W2)?,
        };
        let mixer = match &self.kind {
            LayerKind::Attention(_) => MixerWeights::Attn(AttnWeights {
                q: part(GROUP_PART_MIXER)?,
                k: part(GROUP_PART_MIXER + 1)?,
                v: part(GROUP_PART_MIXER + 2)?,
                o: part(GROUP_PART_MIXER + 3)?,
            }),
            LayerKind::ShortConv(_) => MixerWeights::Conv(ConvWeights {
                in_proj: part(GROUP_PART_MIXER)?,
                out_proj: part(GROUP_PART_MIXER + 1)?,
            }),
        };
        tier.prefetch(st.depth, st.source.as_ref())?;
        // The group's bytes may still be landing (the loader's thread):
        // every kernel of this layer is committed once they have — and
        // nothing is held when they have landed already (R1).
        if let Some(ticket) = wait {
            tier.commit_after_load(ticket)?;
        }
        drop(tier);
        #[cfg(feature = "metal")]
        if moved {
            if let Device::Metal(m) = mlp.gate.weight().device() {
                m.execution_device()?.drop_pass_plans();
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = moved;
        Ok(LayerWeights { mixer, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let w = self.weights()?;
        let residual = x;
        // Under the norm-GEMV arm each half of the layer takes its norm
        // inside its consumers' GEMVs when every one of them qualifies, and
        // the norm as a dispatch otherwise (§13.5t).
        let pre = &self.input_layernorm;
        let x = match (&self.kind, &w.mixer) {
            (LayerKind::Attention(attn), MixerWeights::Attn(aw))
                if norm_gemv_applies(self.norm_gemv, x, pre, &aw.q)
                    && norm_gemv_applies(self.norm_gemv, x, pre, &aw.k)
                    && norm_gemv_applies(self.norm_gemv, x, pre, &aw.v) =>
            {
                attn.forward_add_normed(aw, x, pre, index_pos, block_idx, cache, residual)?
            }
            (LayerKind::ShortConv(conv), MixerWeights::Conv(cw))
                if norm_gemv_applies(self.norm_gemv, x, pre, &cw.in_proj) =>
            {
                conv.forward_add_normed(cw, x, pre, block_idx, cache, residual)?
            }
            _ => {
                let x = self.norm(pre, x)?;
                // `forward_add` is `forward(..)? + residual` unless the fused
                // arm is on AND applies, in which case the add happens at the
                // producing GEMV's f32 accumulator instead of in a `badd_f16`
                // over the rounded store (§12.2h). That is a DIFFERENT
                // COMPUTATION and moves the digests.
                match (&self.kind, &w.mixer) {
                    (LayerKind::Attention(attn), MixerWeights::Attn(aw)) => {
                        attn.forward_add(aw, &x, index_pos, block_idx, cache, residual)?
                    }
                    (LayerKind::ShortConv(conv), MixerWeights::Conv(cw)) => {
                        conv.forward_add(cw, &x, block_idx, cache, residual)?
                    }
                    _ => lloom_tensor::bail!("lfm2: a layer's weights are not its mixer's kind"),
                }
            }
        };

        let residual = &x;
        let post = &self.post_attention_layernorm;
        if self.mlp.norm_gemv_applies(self.norm_gemv, &x, post, &w.mlp) {
            return self.mlp.forward_add_normed(&x, post, residual, &w.mlp);
        }
        let x = self.norm(post, &x)?;
        self.mlp.forward_add(&x, residual, &w.mlp)
    }

    /// [`Self::forward`] for one step of `rows.len()` independent sequences:
    /// the norms, the MLP and every projection run once over `[B, 1, hidden]`;
    /// the attention or conv block runs per row against that row's cache.
    fn forward_rows(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let w = self.weights()?;
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match (&self.kind, &w.mixer) {
            (LayerKind::Attention(attn), MixerWeights::Attn(aw)) => {
                attn.forward_rows_add(aw, &x, block_idx, rows, residual)?
            }
            (LayerKind::ShortConv(conv), MixerWeights::Conv(cw)) => {
                conv.forward_rows_add(cw, &x, block_idx, rows, residual)?
            }
            _ => lloom_tensor::bail!("lfm2: a layer's weights are not its mixer's kind"),
        };
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        self.mlp.forward_add(&x, residual, &w.mlp)
    }

    /// [`Self::forward`] over a ragged batch: the norms, the MLP and every
    /// projection run once over `[1, T, hidden]`; the attention or conv block
    /// runs per row over that row's tokens against its cache.
    fn forward_ragged(
        &self,
        x: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let w = self.weights()?;
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match (&self.kind, &w.mixer) {
            (LayerKind::Attention(attn), MixerWeights::Attn(aw)) => {
                attn.forward_ragged_add(aw, &x, block_idx, rows, residual)?
            }
            (LayerKind::ShortConv(conv), MixerWeights::Conv(cw)) => {
                conv.forward_ragged_add(cw, &x, block_idx, rows, residual)?
            }
            _ => lloom_tensor::bail!("lfm2: a layer's weights are not its mixer's kind"),
        };
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        self.mlp.forward_add(&x, residual, &w.mlp)
    }
}

/// One layer's decode step: the operator norm, the mixer, its residual add,
/// the FFN norm, the MLP, its residual add — the adds unfused, as the MLP's
/// epilogues are. `In` is the hidden state and the layer's index.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile> crate::optimizer::Visit for DecoderLayer<P> {
    type In = (crate::optimizer::NodeId, usize);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (hidden, l): Self::In) -> Self::Out {
        use crate::optimizer::Shape;
        let h = self.mlp.dims.0;
        let dt = self.mlp.down_proj.weight().dtype();
        let residual = hidden;
        let norm_w = w.weight("operator_norm.weight", Shape::of(&[h]), dt);
        let xn = w.rms_norm(norm_w, residual);
        let block = match &self.kind {
            LayerKind::Attention(attn) => w.scoped("self_attn", |w| attn.visit(w, (xn, l))),
            LayerKind::ShortConv(conv) => w.scoped("conv", |w| conv.visit(w, (xn, l))),
        };
        let hidden = w.add(block, residual);
        let residual = hidden;
        let ffn_w = w.weight("ffn_norm.weight", Shape::of(&[h]), dt);
        let xn = w.rms_norm(ffn_w, residual);
        let down = w.scoped("feed_forward", |w| self.mlp.visit(w, xn));
        w.add(down, residual)
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model<P: Profile> {
    profile: PhantomData<P>,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer<P>>,
    /// What `Fusion=Derived` decided at construction; `None` under `Manual`.
    fusion_report: Option<FusionReport>,
    /// The attention kernels every attention layer takes, decided once here.
    attn_route: AttnRoute,
    /// The warnings construction printed, in order.
    load_warnings: Vec<crate::load_warning::LoadWarning>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    /// Whether the final norm may run inside `lm_head`'s GEMV
    /// ([`DecoderLayer`]'s `norm_gemv`, for the head).
    norm_gemv_head: bool,
    dtype: DType,
    conv_state: ConvState,
    l_cache: usize,
}

/// The epilogue arms a derivation sets on the built model's blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EpilogueFlags {
    silu: bool,
    residual: bool,
    gate: GateFusion,
    /// The norms inside their consumers' GEMVs (the prologue).
    norm: bool,
}

/// Derive the epilogue plan over the walk of `model` as built
/// (`crate::optimizer::walk`), and return the flags it sets with the report
/// (`crate::optimizer`, issue #732 P1/P2). `cfg` is what the model was
/// built from: the arms the exclusion set is consulted about.
///
/// The plan is per site; the flags are uniform over a kind. A plan that
/// fuses some sites of a kind and not others is refused rather than
/// approximated — that would be a finding about the catalogue or the model,
/// and it is reported as an error naming the sites.
#[cfg(not(target_arch = "wasm32"))]
fn derive_fusion<P: Profile>(
    model: &Model<P>,
    cfg: &Config,
    budget: crate::optimizer::NumericClass,
) -> Result<(EpilogueFlags, FusionReport)> {
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

    let graph = crate::optimizer::walk(model);
    // The kernel fragments run in no LFM2 switch.
    let policy = Policy {
        budget,
        unavailable: crate::optimizer::fragments::kernels()
            .into_iter()
            .map(|k| (k.name.to_string(), "no LFM2 switch selects it".to_string()))
            .collect(),
        ..Policy::default()
    };
    let d = derive(&graph, &policy)?;
    let plan = &d.plan;
    if !plan.kernels.is_empty() {
        lloom_tensor::bail!("lfm2 fusion=derived: the plan places kernel fragments; plan: {plan}")
    }
    // A site's block, split into its prologue (the norm in its load) and its
    // epilogue (the block on its store).
    let block = |site: &str| -> (bool, String) {
        match plan.sites.get(site).map(String::as_str) {
            Some("norm") => (true, String::new()),
            Some(b) => match b.strip_prefix("norm_") {
                Some(e) => (true, e.to_string()),
                None => (false, b.to_string()),
            },
            None => (false, String::new()),
        }
    };
    let sites_of = |kind: &str| -> Vec<&str> {
        graph
            .nodes()
            .iter()
            .filter(|n| n.op == crate::optimizer::Op::Weight && n.name.ends_with(kind))
            .map(|n| n.name.as_str())
            .collect()
    };
    // Whether every `kind` site carries `want` (true), none does (false), or
    // some do — which no model-wide flag expresses.
    let uniform_by = |kind: &str, what: &str, want: &dyn Fn(&str) -> bool| -> Result<bool> {
        let sites = sites_of(kind);
        let fused = sites.iter().filter(|s| want(s)).count();
        if fused == 0 {
            Ok(false)
        } else if fused == sites.len() {
            Ok(true)
        } else {
            lloom_tensor::bail!(
                "lfm2 fusion=derived: the plan fuses `{what}` onto {fused} of {} `{kind}` sites, \
                 which the model-wide flag cannot express; plan: {plan}",
                sites.len()
            )
        }
    };
    let uniform =
        |kind: &str, fragment: &str| uniform_by(kind, fragment, &|s| block(s).1 == fragment);
    let gemv_silu = uniform("feed_forward.w1.weight", "silu")?;
    let gemv_residual =
        uniform("feed_forward.w2.weight", "residual")? && uniform("out_proj.weight", "residual")?;
    let gemv_gate = if uniform("feed_forward.w3.weight", "swiglu")? {
        GateFusion::Swiglu
    } else if uniform("feed_forward.w3.weight", "mul")? {
        GateFusion::Mul
    } else {
        GateFusion::Off
    };
    // The norm prologue: every site a norm feeds, or none — the forward's
    // switch is model-wide. Under the two-weight kernel `w1` is its operand,
    // not a site.
    let mut norm_kinds = vec![
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "conv.in_proj.weight",
        "feed_forward.w3.weight",
    ];
    if gemv_gate != GateFusion::Swiglu {
        norm_kinds.push("feed_forward.w1.weight");
    }
    let lm_head = model.lm_head_site();
    let mut norm_votes = Vec::new();
    for kind in &norm_kinds {
        if !sites_of(kind).is_empty() {
            norm_votes.push(uniform_by(kind, "norm", &|s| block(s).0)?);
        }
    }
    norm_votes.push(block(&lm_head).0);
    let norm = match (norm_votes.iter().all(|v| *v), norm_votes.iter().any(|v| *v)) {
        (true, _) => true,
        (false, false) => false,
        _ => lloom_tensor::bail!(
            "lfm2 fusion=derived: the plan puts the norm in the load of some of its consumers \
             and not others, which the model-wide switch cannot express; plan: {plan}"
        ),
    };
    // Every fused site must be one of the kinds the flags name.
    for (site, b) in &plan.sites {
        let (pro, fragment) = block(site);
        let fragment = fragment.as_str();
        let known_epilogue = fragment.is_empty()
            || (fragment == "silu" && site.ends_with("feed_forward.w1.weight"))
            || ((fragment == "mul" || fragment == "swiglu")
                && site.ends_with("feed_forward.w3.weight"))
            || (fragment == "residual"
                && (site.ends_with("feed_forward.w2.weight") || site.ends_with("out_proj.weight")));
        let known_prologue =
            !pro || *site == lm_head || norm_kinds.iter().any(|k| site.ends_with(k));
        if !(known_epilogue && known_prologue) {
            lloom_tensor::bail!(
                "lfm2 fusion=derived: the plan fuses `{b}` onto `{site}`, a site no flag \
                 selects; plan: {plan}"
            )
        }
    }
    for (name, why) in policy.unvalidated_in(&graph) {
        report
            .unvalidated
            .push((ExclusionSet::fragment_transform(&name), why));
    }
    report.plan = plan.to_string();
    report.gemv_silu = gemv_silu;
    report.gemv_residual = gemv_residual;
    report.gemv_gate = gemv_gate;
    report.norm_gemv = norm;
    report.refused = d.refused.clone();
    report.fused_in_stateful = plan
        .regions
        .fused_in_stateful
        .iter()
        .map(|s| (s.site.clone(), s.fragment.clone(), s.region.to_string()))
        .collect();
    let flags = EpilogueFlags {
        silu: gemv_silu,
        residual: gemv_residual,
        gate: gemv_gate,
        norm,
    };
    Ok((flags, report))
}

/// The conv fold's ∞-norm gain per channel over every conv layer's loaded
/// weights (`FusionReport::conv_fold_gain`). Once, at construction, under
/// `Fusion=Derived` only.
fn conv_fold_gain<P: Profile>(layers: &[DecoderLayer<P>]) -> Result<Option<ConvFoldGain>> {
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
fn derive_fusion<P: Profile>(
    _model: &Model<P>,
    _cfg: &Config,
    _budget: (),
) -> Result<(EpilogueFlags, FusionReport)> {
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

impl<P: Profile> Model<P> {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        lloom_tensor::memory_admission::begin_model_load();
        crate::profile::hold::<P::Regime>("lfm2")?;
        cfg.check_shape::<P>()?;
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

        // One route for every attention layer, and so one warning per load. A
        // model with no attention layer has nothing to route.
        let has_attention = (0..cfg.num_hidden_layers).any(|l| {
            cfg.layer_types
                .get(l)
                .copied()
                .unwrap_or(LayerType::FullAttention)
                == LayerType::FullAttention
        });
        let requested = if has_attention {
            cfg.attn_impl
        } else {
            AttnImpl::Generic
        };
        let attn_route = AttnRoute::resolve(requested, cfg, vb.dtype(), vb.device())?;
        if has_attention {
            eprintln!("{}", attn_route.line());
        }
        let mut load_warnings = Vec::new();
        if let Some(w) = attn_route.load_warning() {
            eprintln!("{}", w.message);
            load_warnings.push(w);
        }

        // The layers on the weight-group tier when the loader asked for it
        // (`Config::stream_layers`, §4.8): one group per layer — the MLP's
        // `w1`, `w3`, `w2` and the mixer's projections end to end in a slot
        // (`layer_group_parts`) — the plan their order, the source the
        // checkpoint's mapped files.
        let streaming = match &cfg.stream_layers {
            Some(plan) => {
                use crate::nn::slot_cache::WeightGroupId;
                use crate::nn::weight_groups::{GroupSpec, SafetensorsGroups, WeightGroups};
                let specs: Vec<GroupSpec> = (0..cfg.num_hidden_layers)
                    .map(|l| GroupSpec::new(WeightGroupId(l as u32), layer_group_parts(cfg, l)))
                    .collect();
                let names: Vec<Vec<String>> = specs
                    .iter()
                    .map(|g| g.parts.iter().map(|p| p.name.clone()).collect())
                    .collect();
                let mut tier = WeightGroups::new(
                    specs,
                    plan.slots,
                    vb.dtype(),
                    vb.device(),
                    zs::NN_WEIGHT_GROUPS,
                )?;
                tier.set_plan(
                    (0..cfg.num_hidden_layers as u32)
                        .map(WeightGroupId)
                        .collect(),
                );
                // SAFETY: the checkpoint files are mapped read-only for the
                // model's life, as `VarBuilder` maps them.
                let source = unsafe { SafetensorsGroups::open(&plan.files, names, vb.device())? };
                let source = std::sync::Arc::new(source)
                    as std::sync::Arc<dyn crate::nn::weight_groups::GroupSource>;
                // The non-blocking loader (§4.8): on when the device is
                // Metal and the checkpoint writes the tier's dtype; else the
                // tier loads as it is asked, on the calling thread.
                tier.start_loader(std::sync::Arc::clone(&source))?;
                Some((
                    std::sync::Arc::new(parking_lot::Mutex::new(tier)),
                    source,
                    plan.depth,
                ))
            }
            None => None,
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let streamed = streaming
                .as_ref()
                .map(|(tier, source, depth)| StreamedLayer {
                    group: crate::nn::slot_cache::WeightGroupId(layer_idx as u32),
                    tier: std::sync::Arc::clone(tier),
                    source: std::sync::Arc::clone(source),
                    depth: *depth,
                });
            let layer =
                DecoderLayer::new(cfg, layer_idx, attn_route, vb_l.pp(layer_idx), streamed)?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let mut model = Self {
            profile: PhantomData,
            embed_tokens,
            layers,
            fusion_report: None,
            attn_route,
            load_warnings,
            embedding_norm,
            lm_head,
            norm_gemv_head: norm_gemv_arm(),
            dtype: vb.dtype(),
            conv_state: cfg.conv_state,
            l_cache: cfg.conv_l_cache,
        };
        // Under `Fusion=Derived` the optimizer decides the epilogue sites
        // once, here, over the walk of the model as built, and sets them on
        // the blocks, which read them per call as they always did: the
        // derivation is construction-tier, the token path is untouched
        // (`CONTRIBUTING.md` §3.3a). The closed-form gain §5a names wants
        // the loaded weights, so it is read here too and the report printed
        // complete.
        if cfg.fusion == FusionPolicy::Derived {
            // The budget is the arm's (`--lfm2-fusion-budget`): `reorder`
            // admits the norm inside its consumers' GEMVs, whose sum of
            // squares takes another tree.
            #[cfg(not(target_arch = "wasm32"))]
            let budget = match crate::tuning::tuning().lfm2.fusion_budget.as_str() {
                "reorder" => crate::optimizer::NumericClass::ReorderEquiv,
                _ => crate::optimizer::NumericClass::Rounding,
            };
            #[cfg(target_arch = "wasm32")]
            let budget = ();
            let (flags, mut r) = derive_fusion(&model, cfg, budget)?;
            model.set_epilogues(flags);
            r.conv_fold_gain = conv_fold_gain(&model.layers)?;
            eprintln!("{r}");
            model.fusion_report = Some(r);
        }
        Ok(model)
    }

    /// Set every block's epilogue arms (`Fusion=Derived`'s plan).
    fn set_epilogues(&mut self, f: EpilogueFlags) {
        self.set_norm_gemv(f.norm);
        for layer in &mut self.layers {
            layer.mlp.gemv_silu = f.silu;
            layer.mlp.gemv_residual = f.residual;
            layer.mlp.gemv_gate = f.gate;
            match &mut layer.kind {
                LayerKind::Attention(attn) => attn.gemv_residual = f.residual,
                LayerKind::ShortConv(conv) => conv.gemv_residual = f.residual,
            }
        }
    }

    /// The weight name the walk gives `lm_head`'s GEMV: the embedding table
    /// when the two are tied.
    fn lm_head_site(&self) -> String {
        if self.lm_head.weight().id() == self.embed_tokens.embeddings().id() {
            "model.embed_tokens.weight".to_string()
        } else {
            "lm_head.weight".to_string()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// `Fusion=Derived` at `budget` on a model built from `cfg`: the plan
    /// derived over its walk and set on its blocks, as construction does
    /// under that policy; the report returned. For a caller that compares
    /// budgets in one process (the tuning's budget is read once).
    pub fn derive_fusion_at(
        &mut self,
        cfg: &Config,
        budget: crate::optimizer::NumericClass,
    ) -> Result<FusionReport> {
        let (flags, r) = derive_fusion(self, cfg, budget)?;
        self.set_epilogues(flags);
        Ok(r)
    }

    /// Whether every layer's norms and the final norm may run inside their
    /// consumers' GEMVs (`LLOOM_LFM2_NORM_GEMV`'s arm, set per block; a
    /// derived plan sets it from its prologue sites).
    pub fn set_norm_gemv(&mut self, on: bool) {
        for layer in &mut self.layers {
            layer.norm_gemv = on;
        }
        self.norm_gemv_head = on;
    }

    /// What `Fusion=Derived` decided at construction; `None` under `Manual`.
    pub fn fusion_report(&self) -> Option<&FusionReport> {
        self.fusion_report.as_ref()
    }

    /// The attention kernels the model's attention layers take.
    pub fn attn_route(&self) -> AttnRoute {
        self.attn_route
    }

    /// The warnings construction printed to stderr, in order: at most one per
    /// load for the attention route ([`AttnRoute::warning`]).
    /// A diagnosis of the streamed layers (§4.8): the groups whose slot
    /// bytes differ from the checkpoint's, and the tier's counters; `None`
    /// when every weight is resident.
    pub fn streamed_layers_audit(
        &self,
    ) -> Result<Option<(Vec<(u32, usize)>, crate::nn::weight_groups::GroupStats)>> {
        let Some(st) = self.layers.first().and_then(|l| l.streamed.as_ref()) else {
            return Ok(None);
        };
        let tier = st.tier.lock();
        let wrong = tier
            .audit(st.source.as_ref())?
            .into_iter()
            .map(|(g, p)| (g.0, p))
            .collect();
        Ok(Some((wrong, tier.stats())))
    }

    /// Hand the cache the tier's handle (once), so its admission can apply
    /// the arbiter's pledge to the slots and ask again.
    fn attach_layer_tier(&self, cache: &mut State<P>) {
        if let Some(st) = self.layers.first().and_then(|l| l.streamed.as_ref()) {
            cache.attach_layer_tier(&st.tier);
        }
    }

    /// The tier's shape and counters when the layers stream (§4.8):
    /// `(slots, groups, slot_bytes, stats)`; `None` when every weight is
    /// resident. Cheap — no audit.
    pub fn streamed_layers_stats(
        &self,
    ) -> Option<(usize, usize, usize, crate::nn::weight_groups::GroupStats)> {
        let st = self.layers.first().and_then(|l| l.streamed.as_ref())?;
        let tier = st.tier.lock();
        Some((
            tier.n_slots(),
            tier.n_groups(),
            tier.slot_bytes(),
            tier.stats(),
        ))
    }

    pub fn load_warnings(&self) -> &[crate::load_warning::LoadWarning] {
        &self.load_warnings
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        self.attach_layer_tier(cache);
        // Room for this pass, or a refusal that leaves everything as it was —
        // BEFORE the conv phase advances below (`Cache::prepare_pass`).
        cache.prepare_pass(seq_len)?;

        // The ring's write slot, advanced once per forward pass and read by all
        // 22 conv layers (`DESIGN.md` §10.2c -- it is conv state, so it lives in
        // `Cache` beside the bytes it indexes rather than in a params struct).
        //
        // Prefill writes the state directly rather than through the ring, and
        // leaves the newest token at slot `l_cache - 1`, the window at the
        // front; a decode step that opens a sequence writes that same slot.
        // Decode advances from there. Advancing *before* the
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

        #[cfg(feature = "metal")]
        cache.flush_ring_snapshots();
        if seq_len == 1
            && norm_gemv_applies(
                self.norm_gemv_head,
                &hidden_states,
                &self.embedding_norm,
                &self.lm_head,
            )
        {
            let last = hidden_states.squeeze(1)?;
            let logits = norm_gemv(&last, &self.embedding_norm, &self.lm_head, false)?;
            dump_logits(&logits, index_pos)?;
            return logits.to_dtype(DType::F32);
        }
        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        dump_logits(&logits, index_pos + seq_len - 1)?;
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
    pub fn forward_rows(
        &self,
        input_ids: &Tensor,
        rows: &mut [Row<'_, State<P>>],
    ) -> Result<Tensor> {
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
        // also comes before any row has grown. Rows of one shared pool plan
        // the same grow: admitted once.
        let bytes = grow_bytes_of(&plans);
        if bytes > 0 {
            self.attach_layer_tier(rows[0].cache);
            rows[0].cache.admit_grow(
                bytes,
                &format_args!("for a batched step over {} rows", rows.len()),
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
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Tensor> {
        self.forward_ragged_impl(input_ids, rows, false, false)
    }

    /// [`Self::forward_ragged`] for a SPECULATIVE round over several rows
    /// (`batched-decode-design.md` §12.4): every row's window is open
    /// (`Cache::advance`), the result is `[1, T, vocab]` — every position's
    /// logits, the verify's — and each row's layer-input taps are taken over
    /// its own span (`Cache::take_taps`, per row), so the drafter is fed per
    /// row exactly what a one-row verify feeds it.
    pub fn forward_ragged_all(
        &self,
        input_ids: &Tensor,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<Tensor> {
        let _window = lloom_tensor::tuning::VerifyWindow::open();
        self.forward_ragged_impl(input_ids, rows, true, true)
    }

    fn forward_ragged_impl(
        &self,
        input_ids: &Tensor,
        rows: &mut [RaggedRow<'_, State<P>>],
        all_positions: bool,
        speculative: bool,
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
            if row.cache.speculative != speculative {
                lloom_tensor::bail!(
                    "lfm2 forward_ragged: row {r}'s cache {} an open speculative window; this \
                     pass is {}",
                    if row.cache.speculative {
                        "has"
                    } else {
                        "has not"
                    },
                    if speculative {
                        "a verify (open every row first)"
                    } else {
                        "not a verify"
                    }
                )
            }
            plans.push(row.cache.plan_pass(row.len)?);
        }
        let bytes = grow_bytes_of(&plans);
        if bytes > 0 {
            self.attach_layer_tier(rows[0].cache);
            rows[0].cache.admit_grow(
                bytes,
                &format_args!("for a ragged batched pass over {} rows", rows.len()),
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
            Self::tap_layer_input_rows(&hidden_states, block_idx, rows)?;
            hidden_states = layer.forward_ragged(&hidden_states, block_idx, rows)?;
        }

        #[cfg(feature = "metal")]
        for row in rows.iter_mut() {
            row.cache.flush_ring_snapshots();
        }
        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        if all_positions {
            let logits = self.lm_head.forward(&hidden_states.contiguous()?)?;
            return logits.to_dtype(DType::F32);
        }

        // Each row's last position, gathered in row order.
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
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = embeds.dims3()?;
        self.attach_layer_tier(cache);
        cache.prepare_pass(seq_len)?;
        self.advance_conv_phase(seq_len, cache);

        let mut hidden_states = embeds.clone();
        Self::begin_taps(cache);
        for (block_idx, layer) in self.layers.iter().enumerate() {
            mark_graph_node(&cache.device, block_idx);
            Self::tap_layer_input(&hidden_states, block_idx, cache)?;
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }

        #[cfg(feature = "metal")]
        cache.flush_ring_snapshots();
        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        dump_logits(&logits, index_pos + seq_len - 1)?;
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
        cache: &mut State<P>,
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
        #[cfg(feature = "metal")]
        cache.flush_ring_snapshots();
        let hidden_states = self.embedding_norm.forward(&hidden_states)?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        dump_logits(&logits, index_pos)?;
        logits.to_dtype(DType::F32)
    }

    /// [`Self::forward_all`] as a speculative verify window
    /// ([`lloom_tensor::tuning::VerifyWindow`]): under `deterministic` its half
    /// matmuls take the GEMM, the window's rows identical per window shape;
    /// the K=1 step keeps the one-row reduction.
    fn window_forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let _window = lloom_tensor::tuning::VerifyWindow::open();
        self.forward_all(input_ids, index_pos, cache)
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
        cache: &mut State<P>,
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
    fn begin_taps(cache: &mut State<P>) {
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
    /// [`Self::tap_layer_input`] over a ragged pass: row `r`'s taps are its
    /// span of the `[1, T, hidden]` residual, so each row's probe is what a
    /// one-row pass would have tapped.
    fn tap_layer_input_rows(
        hidden: &Tensor,
        block_idx: usize,
        rows: &mut [RaggedRow<'_, State<P>>],
    ) -> Result<()> {
        let mut off = 0usize;
        for row in rows.iter_mut() {
            if !row.cache.tap_layers.is_empty() && row.cache.tap_layers.contains(&block_idx) {
                let span = hidden.narrow(1, off, row.len)?;
                Self::tap_layer_input(&span, block_idx, row.cache)?;
            }
            off += row.len;
        }
        Ok(())
    }

    fn tap_layer_input(hidden: &Tensor, block_idx: usize, cache: &mut State<P>) -> Result<()> {
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
    fn advance_conv_phase(&self, seq_len: usize, cache: &mut State<P>) {
        // The page pool allocates this pass's pages at the first attention
        // layer; this is the once-per-pass point that says a new pass began.
        cache.paged_pass_pending = true;
        cache.turn_checkpoint_begin(seq_len);
        #[cfg(feature = "metal")]
        if cache.shared().is_some() {
            if let Some(b) = cache.shared_mut() {
                b.pass = None;
            }
            // A pass beginning at a page boundary: the ring holds the state
            // after the page's last token — what a row sharing the page
            // starts from (§12.3). Taken BEFORE this pass writes anything.
            if let Err(e) = cache.snapshot_page_boundary() {
                // A snapshot is an optimisation for a sharer; the pass itself
                // is unaffected, so the failure is logged, not raised.
                eprintln!("lfm2 batch: ring snapshot failed: {e}");
            }
        }
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
                // Never below the prefill's seed: a decode step that opens the
                // sequence writes where a prefill leaves the newest column, so
                // a full `l_cache` window fits behind it and the fresh state's
                // zeros stand where the shuffle's zero history does. Every
                // later phase is already at or past that slot.
                let next = (cache.conv_phase + 1).max(self.l_cache - 1);
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
    /// `argmax(logits[i])`. So the check is exact rather than statistical.
    /// What "the target would have produced" means is the contract's
    /// (`architecture.md` §2.3.5b): the window's row runs on the GEMM and the
    /// tiled attention where greedy's one-row step runs on the GEMV and
    /// `sdpa_vector`, and the two reduce in a different order — so under the
    /// default `fast` contract the streams may part where greedy's own top two
    /// sit within that reordering (issue #894: one position in 200 on the
    /// prose prompt, greedy's margin 0.0156 against a max |Δlogit| of 0.023
    /// between the paths, invariant to fencing and run to run). A divergence
    /// that moves with the schedule, or one wider than the reordering, is a
    /// bug, not a tradeoff; the `deterministic` contract owes byte identity
    /// and does not have it yet (§10.2z-v records the price of each way to
    /// close the seam). Distribution-preserving sampling is a later,
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
        cache: &mut State<P>,
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
        cache: &mut State<P>,
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
        let logits = self.window_forward(&input, index_pos, cache)?;
        let argmax = logits.i(0)?.argmax(lloom_tensor::D::Minus1)?;

        // The one readback: `[argmax[0..=k], proposed[0..k]]` in a single
        // buffer, so the host waits for the GPU once and compares locally.
        // Staged (`stage_read`): the wait is on the buffer that produced it,
        // not the queue's tail, and the staging copy is unconditional — a
        // `to_vec1` here took the tail wait and, when the pool handed the
        // 80-byte result an unmapped buffer, a second blit and wait behind it.
        // As a publish where the host can map the words (`VerifyWords`): the
        // stamp is polled, the words read in place; the staged readback
        // where it cannot.
        let both_t = Tensor::cat(&[&argmax, proposed], 0)?;
        let n = 2 * k + 1;
        let device = cache.device.clone();
        let accepted = match VerifyWords::ensure(&mut cache.verify_words, n, &device) {
            Some(w) => {
                let p = w.publish(&both_t, n, &device)?;
                let both = w.complete(p, &device)?;
                let (argmax, proposed) = both.split_at(k + 1);
                accept_prefix(argmax, proposed, eos_ids)
            }
            None => {
                let both = both_t.stage_read()?.to_vec1::<u32>()?;
                let (argmax, proposed) = both.split_at(k + 1);
                accept_prefix(argmax, proposed, eos_ids)
            }
        };

        // `n` positions of state survive: `last` and the `n - 1` proposals the
        // target agreed with. The remaining `k + 1 - n` were written by the
        // pass and are discarded: KV by a length decrement, conv by a phase
        // move, neither by clearing bytes (§10.2a).
        cache.resolve(tok, accepted.len())?;
        Ok(accepted)
    }

    /// [`Self::verify_tokens`] where the target's token at each window
    /// position is **sampled** with `sampler` rather than read as the
    /// position's argmax (#943 §2).
    ///
    /// The acceptance rule is the same one
    /// ([`accept_prefix`]): the run accepts while the target's token equals
    /// the draft's and keeps the target's at the first disagreement, so the
    /// emitted sequence is a draw from the target's own distribution and the
    /// drafter moves only the acceptance rate. At `t = 0` the sampler's
    /// `ArgMax` arm makes this the greedy verify, which is why the greedy
    /// path stays [`Self::verify_tokens`] rather than routing through here:
    /// that one fuses the argmax and the proposals into a single mapped
    /// readback, and a sampled draw cannot (each row needs its own noise
    /// before its own reduction).
    ///
    /// The window's rows are keyed to absolute positions `index_pos ..
    /// index_pos + k`, so a row the verify rejects consumes nothing the next
    /// position's noise depends on and a sampled speculative run reproduces a
    /// sampled non-speculative one at one seed
    /// ([`crate::generation::LogitsProcessor::window_draw`]).
    pub fn verify_tokens_sampled(
        &self,
        last: u32,
        proposed: &Tensor,
        index_pos: usize,
        cache: &mut State<P>,
        eos_ids: &[u32],
        sampler: &mut crate::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        let k = proposed.dims1()?;
        if k == 0 {
            lloom_tensor::bail!("lfm2 verify_tokens_sampled: no proposed tokens")
        }
        if proposed.dtype() != DType::U32 {
            lloom_tensor::bail!(
                "lfm2 verify_tokens_sampled: proposals must be U32 ids, got {:?}",
                proposed.dtype()
            )
        }
        let tok = cache.advance(k + 1)?;
        let anchor = cache.anchor_tensor(last)?;
        let input = Tensor::cat(&[&anchor, proposed], 0)?.reshape((1, k + 1))?;
        // One pass over the weights for k + 1 positions, as the greedy verify.
        let logits = self.window_forward(&input, index_pos, cache)?;
        // One drawn token per position, each with its own position's noise.
        // Row `i` is the input at `index_pos + i` predicting the token AT
        // `index_pos + i + 1`, and the key is the drawn token's own position
        // — the same key the K=1 loop uses for the token it feeds at
        // `seq.pos` (`Session::decode`), so the two paths draw alike.
        let drawn = sampler
            .window_draw_device(&logits, index_pos as u64 + 1, Some(&input.flatten_all()?))?
            .to_vec1::<u32>()?;
        // The proposals come back beside them, as the greedy path's one
        // readback does; the compare is the same host function.
        let proposed_host = proposed.stage_read()?.to_vec1::<u32>()?;
        // What each row feeds: its draw, or a thinking budget's forced tail.
        let fed = sampler.window_fed(&drawn);
        let accepted = accept_prefix(&fed, &proposed_host, eos_ids);
        cache.resolve(tok, accepted.len())?;
        Ok(accepted)
    }
}

/// The decode step's IR (`optimizer::walk`): the token's embedding gather,
/// every layer, the final norm, the head and the cast of the logits to f32
/// — what a `seq_len == 1`, `B == 1` forward dispatches on Metal at the arms
/// this model was built with, the epilogue chain unfused.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile> crate::optimizer::Visit for Model<P> {
    type In = ();
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (): ()) -> Self::Out {
        use crate::optimizer::Shape;
        let table = self.embed_tokens.embeddings();
        let (vocab, h) = (table.dims()[0], table.dims()[1]);
        let dt = self.dtype;
        let ids = w.input("input_ids", Shape::of(&[1, 1]), DType::U32);
        let embed = w.scoped("model", |w| {
            w.weight("embed_tokens.weight", Shape::of(&[vocab, h]), dt)
        });
        let mut hidden = w.opaque("is_u32", &[embed, ids], Shape::of(&[1, 1, h]), dt);
        for (l, layer) in self.layers.iter().enumerate() {
            hidden = w.scoped(format!("model.layers.{l}"), |w| layer.visit(w, (hidden, l)));
        }
        let norm_w = w.scoped("model", |w| {
            w.weight("embedding_norm.weight", Shape::of(&[h]), dt)
        });
        let hn = w.rms_norm(norm_w, hidden);
        // `i((.., seq_len - 1, ..)).contiguous()`: at `seq_len == 1` a view.
        let last = w.view("index", hn, Shape::of(&[1, h]));
        let tied = self.lm_head.weight().id() == table.id();
        let lm_head = if tied {
            embed
        } else {
            w.weight("lm_head.weight", Shape::of(&[vocab, h]), dt)
        };
        let logits = w.gemv(lm_head, last);
        w.opaque("cast", &[logits], Shape::of(&[1, vocab]), DType::F32)
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
    both: Option<lloom_tensor::StagedRead>,
    /// The words as a publish instead (`VerifyWords`); `both` is the staged
    /// readback where the host cannot map the buffer.
    published: Option<Published>,
    k: usize,
    /// This pass's layer-input taps, every window position; trimmed to the
    /// accepted rows at completion.
    taps: Vec<Tensor>,
}

impl<P: Profile> Model<P> {
    /// The encode half of [`Self::verify_tokens`]: open the window, run the
    /// pass over `[last, proposed…]`, and leave `[argmax, proposed]` on the
    /// device behind a marker. Nothing here waits for the GPU.
    pub fn verify_tokens_enqueue(
        &self,
        last: u32,
        proposed: &Tensor,
        index_pos: usize,
        cache: &mut State<P>,
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
        let logits = self.window_forward(&input, index_pos, cache)?;
        let argmax = logits.i(0)?.argmax(lloom_tensor::D::Minus1)?;
        let both_t = Tensor::cat(&[&argmax, proposed], 0)?;
        let n = 2 * k + 1;
        let device = cache.device.clone();
        let published = VerifyWords::ensure(&mut cache.verify_words, n, &device)
            .map(|w| w.publish(&both_t, n, &device))
            .transpose()?;
        let both = if published.is_some() {
            None
        } else {
            Some(both_t.stage_read()?)
        };
        // The taps are this pass's; taken now so the cache carries none into
        // the next pass on it.
        let taps = cache.take_taps();
        Ok(PendingVerify {
            tok,
            both,
            published,
            k,
            taps,
        })
    }

    /// The readback half of [`Self::verify_tokens`]: wait for the pass's own
    /// command buffer, accept, resolve the cache, and trim the taps to the
    /// accepted rows. Same accept rule, same `resolve`, same bytes as the
    /// one-call form; only where the host waits differs.
    pub fn verify_tokens_complete(
        &self,
        pending: PendingVerify,
        cache: &mut State<P>,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        let PendingVerify {
            tok,
            both,
            published,
            k,
            taps,
        } = pending;
        let device = cache.device.clone();
        let accepted = match (published, both) {
            (Some(p), _) => {
                let words = cache
                    .verify_words
                    .0
                    .as_mut()
                    .ok_or_else(|| lloom_tensor::Error::msg("verify words vanished"))?;
                let both = words.complete(p, &device)?;
                let (argmax, proposed) = both.split_at(k + 1);
                accept_prefix(argmax, proposed, eos_ids)
            }
            (None, Some(both)) => {
                let both = both.to_vec1::<u32>()?;
                let (argmax, proposed) = both.split_at(k + 1);
                accept_prefix(argmax, proposed, eos_ids)
            }
            (None, None) => lloom_tensor::bail!("verify_tokens_complete: no readback"),
        };
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
/// The receipt of a ragged verify over several rows (`Model::verify_rows_enqueue`).
pub struct PendingRows {
    toks: Vec<SpecToken>,
    ks: Vec<usize>,
    /// `[argmax over every window position (T), every row's proposed (Σk)]`.
    both: Option<lloom_tensor::StagedRead>,
    /// As `PendingVerify::published`, on the first row's cache.
    published: Option<Published>,
    /// Per row, its taps over its window.
    taps: Vec<Vec<Tensor>>,
}

/// One row of a ragged verify: its cache, the anchor (last committed token
/// and the window's position) and the proposed ids (`[k]`, U32, on the
/// device).
pub struct VerifyRow<'a, P: Profile> {
    pub cache: &'a mut State<P>,
    pub last: u32,
    pub pos: usize,
    pub proposed: &'a Tensor,
}

impl<P: Profile> Model<P> {
    /// The verify of several rows' windows as ONE ragged pass
    /// (`batched-decode-design.md` §12.4): every row's window opened, the
    /// `[last, proposed…]` runs laid end to end, one forward over all of
    /// them, the argmax of every position and the proposals staged behind
    /// the pass. Nothing waits for the GPU; [`Self::verify_rows_complete`]
    /// reads back. Per row it is [`Self::verify_tokens_enqueue`]'s
    /// arithmetic — the same window against the same cache.
    pub fn verify_rows_enqueue(&self, rows: &mut [VerifyRow<'_, P>]) -> Result<PendingRows> {
        if rows.is_empty() {
            lloom_tensor::bail!("lfm2 verify_rows_enqueue: no rows")
        }
        let device = rows[0].cache.device.clone();
        let mut toks = Vec::with_capacity(rows.len());
        let mut ks = Vec::with_capacity(rows.len());
        let mut pieces = Vec::with_capacity(rows.len());
        let mut proposed_all = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let k = row.proposed.dims1()?;
            if k == 0 {
                lloom_tensor::bail!("lfm2 verify_rows_enqueue: a row proposes no tokens")
            }
            if row.proposed.dtype() != DType::U32 {
                lloom_tensor::bail!(
                    "lfm2 verify_rows_enqueue: proposals must be U32 ids, got {:?}",
                    row.proposed.dtype()
                )
            }
            toks.push(row.cache.advance(k + 1)?);
            ks.push(k);
            let anchor = row.cache.anchor_tensor(row.last)?;
            pieces.push(Tensor::cat(&[&anchor, row.proposed], 0)?);
            proposed_all.push(row.proposed.clone());
        }
        let input = Tensor::cat(&pieces, 0)?.unsqueeze(0)?;
        let mut ragged: Vec<RaggedRow<'_, State<P>>> = rows
            .iter_mut()
            .zip(ks.iter())
            .map(|(row, &k)| RaggedRow {
                pos: row.pos,
                len: k + 1,
                cache: &mut *row.cache,
            })
            .collect();
        let logits = self.forward_ragged_all(&input, &mut ragged)?;
        drop(ragged);
        let argmax = logits.i(0)?.argmax(lloom_tensor::D::Minus1)?;
        let mut both_parts: Vec<Tensor> = vec![argmax];
        both_parts.extend(proposed_all);
        let both_t = Tensor::cat(&both_parts, 0)?;
        let n = both_t.dims1()?;
        let published = VerifyWords::ensure(&mut rows[0].cache.verify_words, n, &device)
            .map(|w| w.publish(&both_t, n, &device))
            .transpose()?;
        let both = if published.is_some() {
            None
        } else {
            Some(both_t.stage_read()?)
        };
        let taps = rows.iter_mut().map(|row| row.cache.take_taps()).collect();
        Ok(PendingRows {
            toks,
            ks,
            both,
            published,
            taps,
        })
    }

    /// Read back a ragged verify: per row its accepted tokens (never empty)
    /// and its probe trimmed to them, the cache resolved to the accepted
    /// length — [`Self::verify_tokens_complete`] per row.
    pub fn verify_rows_complete(
        &self,
        pending: PendingRows,
        caches: &mut [&mut State<P>],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, Vec<Tensor>)>> {
        let PendingRows {
            toks,
            ks,
            both,
            published,
            taps,
        } = pending;
        if caches.len() != toks.len() || eos_sets.len() != toks.len() {
            lloom_tensor::bail!(
                "lfm2 verify_rows_complete: {} caches and {} EOS sets for a verify of {} rows",
                caches.len(),
                eos_sets.len(),
                toks.len()
            )
        }
        let both: Vec<u32> = match (published, both) {
            (Some(p), _) => {
                let device = caches[0].device.clone();
                let words = caches[0]
                    .verify_words
                    .0
                    .as_mut()
                    .ok_or_else(|| lloom_tensor::Error::msg("verify words vanished"))?;
                words.complete(p, &device)?.to_vec()
            }
            (None, Some(both)) => both.to_vec1::<u32>()?,
            (None, None) => lloom_tensor::bail!("verify_rows_complete: no readback"),
        };
        let t: usize = ks.iter().map(|k| k + 1).sum();
        let (argmax_all, proposed_all) = both.split_at(t);
        let mut out = Vec::with_capacity(caches.len());
        let (mut off_a, mut off_p) = (0usize, 0usize);
        for ((r, cache), tok) in caches.iter_mut().enumerate().zip(toks.into_iter()) {
            let k = ks[r];
            let argmax = &argmax_all[off_a..off_a + k + 1];
            let proposed = &proposed_all[off_p..off_p + k];
            off_a += k + 1;
            off_p += k;
            let accepted = accept_prefix(argmax, proposed, eos_sets[r]);
            cache.resolve(tok, accepted.len())?;
            let n_acc = accepted.len();
            let probe = taps[r]
                .iter()
                .map(|t| {
                    let seq = t.dim(1)?;
                    t.narrow(1, 0, n_acc.min(seq))
                })
                .collect::<Result<Vec<_>>>()?;
            out.push((accepted, probe));
        }
        Ok(out)
    }
}

pub use crate::nn::speculator::accept_prefix;

// The engine's model traits over LFM2's existing entry points
// (`RESTRUCTURING_DESIGN.md` §3.1). No computation is threaded through them that
// the inherent methods above do not already do; each wires to one of them.
//
// `forward` is overridden rather than taking `CausalLM`'s provided
// `embed`+`forward_embeds` default: the inherent `forward` is on the per-token
// path and its shape is what every recorded digest belongs to (`DESIGN.md`
// §2.3.8c), so it stays byte-identical rather than routing through two calls.
impl<P: Profile> crate::profile::Shaped for Model<P> {
    type Profile = P;
}

impl<P: Profile> ModelTrait for Model<P> {
    type Config = Config;

    fn from_weights(vb: VarBuilder, cfg: &Self::Config) -> Result<Self> {
        Model::new(cfg, vb)
    }
}

impl<P: Profile> CausalLM for Model<P> {
    type Cache = State<P>;

    fn new_cache(&self, cfg: &Self::Config, dev: &Device) -> Result<Self::Cache> {
        State::<P>::new(true, self.dtype, cfg, dev)
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

    /// Every multi-token pass leaves its own turn checkpoint
    /// (`sequence-reuse-design.md` §2.3a), whatever the request asks; the
    /// request scopes only the pin's second target.
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        _policy: crate::nn::CheckpointPolicy,
        _end: usize,
    ) -> Result<()> {
        cache.open_checkpoint_request();
        Ok(())
    }

    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        cache.close_checkpoint_request();
        Ok(())
    }

    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        cache.pin_checkpoint(pos)
    }

    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        cache.has_pin_frame()
    }

    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        cache.pinned_checkpoint()
    }

    fn vocab_size(&self) -> usize {
        // `lm_head` is `[vocab, hidden]` (`Model::new`: `linear(hidden, vocab)`
        // or the tied `[vocab, hidden]` embedding), so its leading dim is the
        // vocabulary. 2D by construction, so `dims()[0]` cannot be absent.
        self.lm_head.weight().dims()[0]
    }
}

impl<P: Profile> BatchedCausalLM for Model<P> {
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

impl<P: Profile> crate::nn::model::CacheRewind for State<P> {
    fn rewind_horizon(&self) -> usize {
        State::<P>::rewind_horizon(self)
    }

    fn rewind(&mut self, n: usize) -> Result<()> {
        State::<P>::rewind(self, n)
    }

    fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        State::<P>::rewind_unwritten(self, n)
    }

    fn can_restore(&self, pos: usize) -> bool {
        State::<P>::can_restore(self, pos)
    }

    fn restore_point(&self, pos: usize) -> Option<usize> {
        State::<P>::restore_point(self, pos)
    }

    fn restore(&mut self, pos: usize) -> Result<()> {
        State::<P>::restore(self, pos)
    }
}

impl<P: Profile> Speculative for Model<P> {
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
impl<P: Profile> crate::nn::Probes<crate::nn::LayerTaps<5>> for Model<P> {
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

    fn verify_and_probe_sampled(
        &self,
        drafts: &[crate::nn::Draft],
        anchors: &[crate::nn::Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
        sampler: &mut crate::generation::LogitsProcessor,
    ) -> Result<(Vec<Vec<u32>>, Vec<Tensor>)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "lfm2 verify_and_probe_sampled: the LFM2 cache is single-sequence (B=1); \
                 got {} drafts and {} anchors",
                drafts.len(),
                anchors.len()
            )
        }
        let anchor = anchors[0];
        let ids = match &drafts[0].tokens {
            crate::nn::DraftTokens::Host(ids) => {
                if ids.is_empty() {
                    lloom_tensor::bail!("lfm2 verify_and_probe_sampled: no proposed tokens")
                }
                Tensor::new(ids.as_slice(), &cache.device)?
            }
            crate::nn::DraftTokens::Device(ids) => ids.clone(),
        };
        let accepted =
            self.verify_tokens_sampled(anchor.last, &ids, anchor.pos, cache, eos_ids, sampler)?;
        // The probe, trimmed to the accepted rows, as the greedy verify's.
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

    type PendingRows = PendingRows;

    fn verify_rows_enqueue(
        &self,
        rows: &mut [crate::nn::VerifyRowRef<'_, Self::Cache>],
    ) -> Result<PendingRows> {
        // The drafts as device ids, one tensor per row (a host draft is
        // uploaded; a device one is taken as it is).
        let ids: Vec<Tensor> = rows
            .iter()
            .map(|r| match &r.draft.tokens {
                crate::nn::DraftTokens::Host(v) => {
                    if v.is_empty() {
                        lloom_tensor::bail!("lfm2 verify_rows_enqueue: a row proposes no tokens")
                    }
                    Tensor::new(v.as_slice(), &r.cache.device)
                }
                crate::nn::DraftTokens::Device(t) => Ok(t.clone()),
            })
            .collect::<Result<_>>()?;
        let mut vrows: Vec<VerifyRow<'_, P>> = rows
            .iter_mut()
            .zip(ids.iter())
            .map(|(r, t)| VerifyRow {
                cache: &mut *r.cache,
                last: r.anchor.last,
                pos: r.anchor.pos,
                proposed: t,
            })
            .collect();
        Model::verify_rows_enqueue(self, &mut vrows)
    }

    fn verify_rows_complete(
        &self,
        pending: PendingRows,
        caches: &mut [&mut Self::Cache],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, Vec<Tensor>)>> {
        Model::verify_rows_complete(self, pending, caches, eos_sets)
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
        /// Widen the slab by `by` positions: `State::<CrownExact>::prepare_pass`'s grow, at
        /// slot level. The model's capacity moves with it and nothing else
        /// does, which is the whole contract — every live position and every
        /// rewound one stays where it was, and the boundary the fill symbol
        /// above pins moves to the new width.
        Grow { by: usize },
        /// A turn boundary, and the next turn runs at `heads` KV heads.
        ///
        /// **The head count is carried on the symbol deliberately.** `reset`'s
        /// contract is that the *buffer* is dropped and not merely the length,
        /// and `State::<CrownExact>::clear`'s doc gives the reason in as many words: *"a new
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
        let mut slot = kv_slot(capacity);
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
                    if slot.is_allocated() {
                        t.grows += 1;
                    }
                    slot.grow_to(to)
                        .map_err(|e| format!("op {i}: grow_to({to}) failed: {e}"))?;
                    if slot.capacity() != to {
                        return Err(format!(
                            "op {i}: grow_to({to}) left the capacity at {}",
                            slot.capacity()
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
            match slot.data().as_ref() {
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
                    if slot.len() != want.len() {
                        return Err(format!(
                            "op {i} ({op:?}): slot.len()={} model={}",
                            slot.len(),
                            want.len()
                        ));
                    }
                    if slot.len() > 0 {
                        let view = all
                            .narrow(2, 0, slot.len())
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
                                 the previous turn's shape -- `State::<CrownExact>::clear`'s doc gives \
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
                                slot.len(),
                                &got[..got.len().min(12)],
                                &expect[..expect.len().min(12)]
                            ));
                        }
                    }
                }
            }
            if slot.len() > capacity {
                return Err(format!(
                    "op {i} ({op:?}): len {} exceeds capacity {capacity}",
                    slot.len()
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
#[path = "lfm2/reset_recovery_tests.rs"]
mod reset_recovery_tests;

#[cfg(test)]
mod tests {
    use super::kv_slot_fuzz::*;
    use super::*;
    use crate::profile::CrownExact;

    thread_local! {
        /// `Cache::conv_spare_for`'s calls on this thread.
        pub(super) static SPARES_TAKEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// Per-pass metadata is immutable across queued readers, while host staging
    /// keeps its capacity. Exercise page boundaries and a recycled sequence.
    #[cfg(feature = "metal")]
    #[test]
    fn paged_metadata_reuses_staging_without_overwriting_queued_counts() -> Result<()> {
        let _pool = lloom_tensor::AutoreleaseScope::new();
        let device = Device::new_metal(0)?;
        let Device::Metal(m) = &device else {
            unreachable!()
        };
        let clock = m.gpu_clock();
        let config = Config {
            flash_page_size: 4,
            flash_pages_per_chunk: 1,
            ..Default::default()
        };
        let mut paged = PagedKv::new(&config, 64, 64, 1, clock.clone())?;
        let mut reads = Vec::new();
        let mut first_table = None;
        for n in 1..=17usize {
            paged.begin_pass(1, 1, &device)?;
            let count = paged.live_count.as_ref().expect("one shared count");
            // An actual queued GPU reader, not just the host's current field.
            reads.push((n, count.affine(1., 0.)?));
            if n == 1 {
                first_table = paged.table_t.clone();
            }
            assert_eq!(
                paged.table_t.as_ref().unwrap().id(),
                first_table.as_ref().unwrap().id(),
                "an append-only table must remain resident across page boundaries"
            );
            assert_eq!(paged.table.len(), n.div_ceil(4));
            assert_eq!(paged.slot_staging.as_slice(), &[(n - 1) as u32]);
        }
        device.synchronize()?;
        for (n, read) in reads {
            assert_eq!(
                read.to_vec1::<u32>()?,
                vec![n as u32],
                "old pass count was overwritten"
            );
        }
        let slot_capacity = paged.slot_staging.capacity();
        let table_capacity = paged.table.capacity();
        paged.release_all();
        // Release stamps the open epoch; close it before recycling this fixture.
        device.synchronize()?;
        paged.begin_pass(1, 1, &device)?;
        assert_eq!(
            paged.live_count.as_ref().unwrap().to_vec1::<u32>()?,
            vec![1]
        );
        assert_eq!(paged.slot_staging.capacity(), slot_capacity);
        assert_eq!(paged.table.capacity(), table_capacity);
        assert_ne!(
            paged.table_t.as_ref().unwrap().id(),
            first_table.as_ref().unwrap().id(),
            "a reset must not shorten/overwrite an old table still held by readers"
        );
        println!("PAGED_METADATA queued_counts=17 page_boundaries=4 resident_table_reused=true reset_isolated=true staging_capacity_stable=true");
        Ok(())
    }

    /// The unfused `Shuffle` decode step takes no spare state: the spare is
    /// the fused step's (it writes the next state there and swaps), and a
    /// step that does not take the fused arm asked for one, zeroed it and
    /// dropped it — an allocation and a fill a conv layer a token
    /// (`docs/optimizer-design.md` §10i). Three decode steps of a two-conv-
    /// layer model on the CPU, where the fused arm never runs: none taken.
    #[test]
    fn the_unfused_shuffle_step_takes_no_spare() -> Result<()> {
        use crate::models::lfm2_testing::Lfm2Tiny;
        use crate::testing::tiny::{TinyCheckpoint, Weights};
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 64,
            intermediate_size: 64,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            conv_l_cache: 3,
            layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
            tie_embedding: true,
            conv_state: ConvState::Shuffle,
            conv_fused: false,
            ..Default::default()
        };
        let dev = Device::Cpu;
        let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(DType::F32)
            .model::<CrownExact>(&dev)?;
        let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?;
        let prompt = Tensor::new(&[1u32, 2, 3], &dev)?.reshape((1, 3))?;
        model.forward(&prompt, 0, &mut cache)?;
        let before = SPARES_TAKEN.with(|n| n.get());
        for pos in 3..6 {
            let tok = Tensor::new(&[pos as u32], &dev)?.reshape((1, 1))?;
            model.forward(&tok, pos, &mut cache)?;
        }
        let taken = SPARES_TAKEN.with(|n| n.get()) - before;
        assert_eq!(taken, 0, "spares taken by three unfused decode steps");
        Ok(())
    }

    /// The trait impls dispatch to the inherent entry points, checked at the
    /// bound rather than at a forward pass.
    ///
    /// A keystone-trait PR earns its reachability the way §3.1c asks — the
    /// production consumers (`KvCache`, `VisionEncoder`, `candle-hub`, llumos)
    /// arrive in PRs 4–10, so here the callable-through-the-trait property is
    /// what a generic bound proves: this function only compiles if `Model<CrownExact>`
    /// satisfies `Model<CrownExact>`/`CausalLM`/`Speculative` and the associated types line
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
        assert_bounds::<Model<CrownExact>>();
        // The associated types resolve to LFM2's own state, not a placeholder:
        // each closure only typechecks if the two sides are the same type.
        let _cache: fn(<Model<CrownExact> as CausalLM>::Cache) -> State<CrownExact> = |c| c;
        let _window: fn(<Model<CrownExact> as Speculative>::Window) -> SpecToken = |w| w;
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
            let mut slot = kv_slot(8);
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
        let mut slot = kv_slot(8);
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
        let mut slot = kv_slot(8);
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
        let backing = slot.data().as_ref().unwrap();
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
    /// See `State::<CrownExact>::new_with` for why no policy is chosen.
    #[test]
    fn exceeding_capacity_fails_loudly() -> Result<()> {
        let dev = Device::Cpu;
        let mut slot = kv_slot(2);
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
        let mut slot = kv_slot(8);
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
        let mut grown = kv_slot(2);
        let mut wide = kv_slot(5);
        for i in 0..2 {
            grown.append(&token(i, &dev)?)?;
            wide.append(&token(i, &dev)?)?;
        }
        let before = grown
            .data()
            .as_ref()
            .unwrap()
            .narrow(2, 0, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        grown.grow_to(5)?;
        assert_eq!(grown.capacity(), 5);
        assert_eq!(grown.len(), 2, "a grow moves no position");
        assert_eq!(
            grown.data().as_ref().unwrap().dims()[2],
            5,
            "the slab is five wide after the grow"
        );
        let after = grown
            .data()
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
        let view = grown.data().as_ref().unwrap().narrow(2, 0, 5)?;
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
        let mut slot = kv_slot(4);
        slot.append(&token(0, &dev)?)?;
        let id = slot.data().as_ref().unwrap().id();
        slot.grow_to(4)?;
        slot.grow_to(2)?;
        assert_eq!(slot.capacity(), 4);
        assert_eq!(
            slot.data().as_ref().unwrap().id(),
            id,
            "no-op grows keep the buffer"
        );
        slot.grow_to(6)?;
        assert_ne!(
            slot.data().as_ref().unwrap().id(),
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
        let mut slot = kv_slot(2);
        slot.grow_to(6)?;
        assert!(!slot.is_allocated());
        assert_eq!(slot.capacity(), 6);
        let view = slot.append(&token(0, &dev)?)?;
        assert_eq!(view.dims()[2], 1);
        assert_eq!(slot.data().as_ref().unwrap().dims()[2], 6);
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

#[cfg(all(test, feature = "metal"))]
mod admission_geometry_tests {
    use super::*;
    use lloom_tensor::metal_backend::admission;

    fn geometry(conv: usize, attn: usize, hidden: usize, heads: usize, kv_heads: usize) -> Config {
        let mut layer_types = vec![LayerType::Conv; conv];
        layer_types.extend(vec![LayerType::FullAttention; attn]);
        Config {
            hidden_size: hidden,
            num_hidden_layers: conv + attn,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            layer_types,
            ..Default::default()
        }
    }

    /// The per-sequence classes are the config's: at LFM2-2.6B's geometry the
    /// admission constants exactly (the numbers every recorded budget used),
    /// and another geometry its own bytes — not the 2.6B's.
    #[test]
    fn admission_budget_reads_the_configs_geometry() {
        let mb = MemoryBudget::new(1);
        let big = geometry(22, 8, 2048, 32, 8);
        let b = State::<crate::profile::CrownExact>::admission_budget(&big, &mb, 1024);
        assert_eq!(b.conv_state_bytes, admission::CONV_STATE_BYTES);
        assert_eq!(b.kv_bytes_per_token, admission::KV_BYTES_PER_TOKEN);
        let small = geometry(2, 1, 64, 4, 2);
        let s = State::<crate::profile::CrownExact>::admission_budget(&small, &mb, 1024);
        let width = small.conv_state.width(small.conv_l_cache);
        assert_eq!(s.conv_state_bytes, 2 * 64 * width * 2);
        assert_eq!(s.kv_bytes_per_token, 2 * 2 * 16 * 2);
        assert!(s.kv_bytes_per_token < b.kv_bytes_per_token);
    }
}
