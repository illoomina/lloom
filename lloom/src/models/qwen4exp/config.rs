//! Qwen3.8-Flash-Next (Qwen4-exp) runtime config — the geometry the blocks read.
//!
//! Field values and their GGUF metadata keys are grounded in the checkpoint
//! header (`qwen4exp.*`, read 2026-09-19) and the reference `load_arch_hparams`
//! (llama.cpp `src/models/qwen4exp.cpp`). This is the `Model::Config` associated
//! type — a runtime struct, not the wire format (per `nn::model` docs, LFM2's
//! precedent: the checkpoint layer deserializes, this is what the blocks see).

use super::deltanet::DeltaNetDims;
use crate::nn::ops::KvPrecision;
use lloom_macros::Config as ConfigDerive;
use lloom_tensor::DType;

/// Which token-mixer a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention (the `linear_attention` layers).
    Linear,
    /// Qwen Sparse Attention (the `full_attention` layers, every `n`th).
    Full,
}

/// The verified geometry — every field carries its GGUF key in a comment —
/// and the runtime axes (`#[axis]`, not from the checkpoint): each registers
/// in `lloom::axes::AXES` and becomes a `ConfigArgs` flag, as on LFM2's
/// `Config` (RESTRUCTURING §1.6). `#[tiny]` marks what the shape campaign
/// may search from [`Config::tiny`] (`docs/porting-models.md` §21.8).
#[derive(Debug, Clone, ConfigDerive)]
pub struct Config {
    // ---- top-level ----
    /// `qwen4exp.block_count` = 48.
    #[tiny(axis, max = 8)]
    pub n_layers: usize,
    /// `qwen4exp.embedding_length` = 2560.
    #[tiny(axis)]
    pub hidden_size: usize,
    /// `vocab_size` = 248320.
    #[tiny(axis)]
    pub vocab_size: usize,
    /// `qwen4exp.attention.layer_norm_rms_epsilon` = 1e-6.
    #[tiny(fixed)]
    pub norm_eps: f64,
    /// `qwen4exp.full_attention_interval` = 4 (every 4th layer is full/QSA).
    #[tiny(axis, max = 8)]
    pub full_attention_interval: usize,
    /// Per-layer kind, resolved from `layer_types` / the interval (never hardcoded).
    #[tiny(
        derived = "Self::resolve_layer_types(n_layers, full_attention_interval)",
        cover
    )]
    pub layer_types: Vec<LayerKind>,

    // ---- HyperConnection (Gated Residual) ----
    /// `hc_count` = 4 (n parallel streams).
    #[tiny(axis)]
    pub hc_count: usize,
    /// `qwen4exp` HC low-rank = 320.
    #[tiny(axis)]
    pub hc_lowrank: usize,

    // ---- attention (QSA / full) ----
    /// `qwen4exp.attention.head_count` = 24.
    #[tiny(axis)]
    pub n_heads: usize,
    /// `qwen4exp.attention.head_count_kv` = 2 (GQA).
    #[tiny(axis)]
    pub n_kv_heads: usize,
    /// `qwen4exp.attention.key_length` = `value_length` = 256.
    #[tiny(axis)]
    pub head_dim: usize,
    /// `partial_rotary_factor` = 0.25 (64 of 256 dims get RoPE).
    #[tiny(fixed)]
    pub partial_rotary_factor: f64,
    /// `qwen4exp.rope.freq_base` = 1e7.
    #[tiny(fixed)]
    pub rope_theta: f64,
    /// `qwen4exp.rope.dimension_sections` = [11,11,10,0] (mRoPE).
    #[tiny(fixed)]
    pub mrope_section: [usize; 4],
    /// `qwen4exp.attention.indexer.head_count` = 4.
    #[tiny(axis)]
    pub indexer_n_heads: usize,
    /// `qwen4exp.attention.indexer.key_length` = 128.
    #[tiny(axis)]
    pub indexer_head_dim: usize,
    /// `qwen4exp.attention.indexer.top_k` = 2048.
    #[tiny(axis)]
    pub indexer_budget: usize,
    /// `qwen4exp.attention.compress_ratios` — the indexer compression per layer
    /// (0 on linear layers, 4 on full). Reduces to a scalar 4 here.
    #[tiny(fixed)]
    pub indexer_compress_ratio: usize,

    // ---- DeltaNet (SSM) ----
    /// `qwen4exp.ssm.state_size` = 128 (= d_k = d_v).
    #[tiny(axis)]
    pub ssm_state_size: usize,
    /// `qwen4exp.ssm.group_count` = 16 (= n key/query heads).
    #[tiny(axis)]
    pub ssm_n_group: usize,
    /// `qwen4exp.ssm.time_step_rank` = 48 (= n value heads; the state carries these).
    #[tiny(axis)]
    pub ssm_dt_rank: usize,
    /// `qwen4exp.ssm.conv_kernel` = 4.
    #[tiny(fixed)]
    pub ssm_conv_kernel: usize,
    /// `qwen4exp.ssm.inner_size` = 6144 (= value_dim).
    #[tiny(derived = "ssm_state_size * ssm_dt_rank")]
    pub ssm_inner_size: usize,

    // ---- MoE ----
    /// `qwen4exp.expert_count` = 512.
    #[tiny(axis)]
    pub n_experts: usize,
    /// `qwen4exp.expert_used_count` = 10.
    #[tiny(axis)]
    pub n_experts_per_tok: usize,
    /// `qwen4exp.expert_feed_forward_length` = 640.
    #[tiny(axis)]
    pub moe_intermediate_size: usize,
    /// `qwen4exp.expert_shared_feed_forward_length` = 640.
    #[tiny(axis)]
    pub shared_expert_intermediate_size: usize,

    // ---- N-gram PLE ----
    /// `qwen4exp.ple.layers` — the single PLE layer (blk.1 in the GGUF; the i±1
    /// layer-offset the doc flags). Empty if the model has no PLE.
    #[tiny(fixed)]
    pub ple_layers: Vec<usize>,
    /// `qwen4exp.ple.ngram_size` = 3.
    #[tiny(fixed)]
    pub ple_ngram_size: usize,
    /// `qwen4exp.ple.heads_per_ngram` = 8.
    #[tiny(fixed)]
    pub ple_heads_per_ngram: usize,
    /// `qwen4exp.ple.conv_kernel` = 4.
    #[tiny(fixed)]
    pub ple_conv_kernel: usize,
    /// `qwen4exp.embedding_length_per_layer_input` = 160 (the table's row width).
    #[tiny(fixed)]
    pub ple_head_dim: usize,
    /// `qwen4exp.ple.image_token_id` — the id hashed for a position that
    /// carries an embedding but no token (an image); EOS when absent.
    #[tiny(fixed)]
    pub ple_image_token_id: Option<u32>,

    // ---- tokens ----
    /// `bos = eos` = 248044.
    #[tiny(fixed)]
    pub bos_token_id: u32,
    #[tiny(fixed)]
    pub eos_token_id: u32,

    // ---- runtime axes ----
    /// The full-attention layers' KV store (`DESIGN.md` §10.7). `Off` is the
    /// f32 cache that ships; `f16` is `Off` at f16; `int8` / `int4` are the
    /// `KvQuant` codes, read by the quantized kernels on the dense and the
    /// selected path. The registered axis is `KvQuant` (LFM2's, same arms):
    /// `Off`'s dtype is a parameter of that arm (`KvPrecision`). Read when a
    /// `Cache` is built; the DeltaNet state is not KV and stays f32.
    #[axis(
        name = "KvQuant",
        flag = "kv-quant",
        variants = ["Off", "Int8", "Int4"],
        default = "Off",
        parser = "parse_kv_precision"
    )]
    pub kv_precision: KvPrecision,
}

/// The KV store this family ships: `Off` at f32.
pub const DEFAULT_KV_PRECISION: KvPrecision = KvPrecision::dense(DType::F32);

/// `--kv-quant` for this family: `off` is the f32 store.
pub fn parse_kv_precision(s: &str) -> std::result::Result<KvPrecision, String> {
    KvPrecision::parse(s, DType::F32).map_err(|e| e.to_string())
}

impl Config {
    /// The DeltaNet head geometry derived from the SSM config.
    pub fn deltanet_dims(&self) -> DeltaNetDims {
        DeltaNetDims {
            d_k: self.ssm_state_size,
            d_v: self.ssm_state_size,
            h_k: self.ssm_n_group,
            h_v: self.ssm_dt_rank,
        }
    }

    /// The n-stream HC carry width, `hc_count * hidden_size`.
    pub fn hc_dim(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    /// The DeltaNet concat-QKV conv channel count:
    /// `key_dim*2 + value_dim = (state*n_group)*2 + state*dt_rank`.
    pub fn conv_channels(&self) -> usize {
        let key_dim = self.ssm_state_size * self.ssm_n_group;
        let value_dim = self.ssm_state_size * self.ssm_dt_rank;
        key_dim * 2 + value_dim
    }

    /// Resolve the per-layer kinds from `full_attention_interval`: linear
    /// everywhere except every `interval`-th layer (1-indexed: layer `i` is full
    /// iff `(i+1) % interval == 0`), matching the reference (qwen4exp.cpp:133) and
    /// the verified `layer_types` array `[L,L,L,F]×12`.
    pub fn resolve_layer_types(n_layers: usize, interval: usize) -> Vec<LayerKind> {
        (0..n_layers)
            .map(|i| {
                if interval > 0 && (i + 1) % interval == 0 {
                    LayerKind::Full
                } else {
                    LayerKind::Linear
                }
            })
            .collect()
    }

    /// Whether layer `l` is a full-attention (QSA) layer.
    pub fn is_full(&self, l: usize) -> bool {
        self.layer_types.get(l) == Some(&LayerKind::Full)
    }

    /// The verified Qwen3.8-Flash-Next geometry (the checkpoint's values), for
    /// tests and as the reference the GGUF loader reconciles against.
    pub fn flash_next() -> Self {
        let n_layers = 48;
        let full_attention_interval = 4;
        Config {
            n_layers,
            hidden_size: 2560,
            vocab_size: 248320,
            norm_eps: 1e-6,
            full_attention_interval,
            layer_types: Self::resolve_layer_types(n_layers, full_attention_interval),
            hc_count: 4,
            hc_lowrank: 320,
            n_heads: 24,
            n_kv_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10, 0],
            indexer_n_heads: 4,
            indexer_head_dim: 128,
            indexer_budget: 2048,
            indexer_compress_ratio: 4,
            ssm_state_size: 128,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ssm_conv_kernel: 4,
            ssm_inner_size: 6144,
            n_experts: 512,
            n_experts_per_tok: 10,
            moe_intermediate_size: 640,
            shared_expert_intermediate_size: 640,
            ple_layers: vec![1],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_conv_kernel: 4,
            ple_head_dim: 160,
            ple_image_token_id: Some(248056),
            bos_token_id: 248044,
            eos_token_id: 248044,
            kv_precision: DEFAULT_KV_PRECISION,
        }
    }

    /// A tiny synthetic config for CPU tests — the same SHAPE (all block types,
    /// the HC/DeltaNet/QSA/MoE seams exercised) at small dims, so a forward
    /// threads end-to-end without the 44GB checkpoint.
    pub fn tiny() -> Self {
        let n_layers = 4; // [L, L, L, F] — one of each path
        let full_attention_interval = 4;
        Config {
            n_layers,
            hidden_size: 32,
            vocab_size: 128,
            norm_eps: 1e-6,
            full_attention_interval,
            layer_types: Self::resolve_layer_types(n_layers, full_attention_interval),
            hc_count: 2,
            hc_lowrank: 8,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            mrope_section: [3, 3, 2, 0],
            indexer_n_heads: 2,
            indexer_head_dim: 8,
            indexer_budget: 16,
            indexer_compress_ratio: 4,
            ssm_state_size: 8,
            ssm_n_group: 2,
            ssm_dt_rank: 4,
            ssm_conv_kernel: 4,
            ssm_inner_size: 32, // state(8) * dt_rank(4)
            n_experts: 8,
            n_experts_per_tok: 2,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            ple_layers: vec![],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 2,
            ple_conv_kernel: 4,
            ple_head_dim: 4,
            ple_image_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            kv_precision: DEFAULT_KV_PRECISION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer-type pattern is `[L,L,L,F]×12` for the real geometry — verified
    /// against the checkpoint's `layer_types` array. A wrong pattern changes no
    /// shape but runs the wrong mixer on 12 layers (text-gate-only, §20.1a).
    #[test]
    fn layer_types_are_lllf_repeated() {
        let cfg = Config::flash_next();
        assert_eq!(cfg.layer_types.len(), 48);
        for (i, k) in cfg.layer_types.iter().enumerate() {
            let want = if (i + 1) % 4 == 0 {
                LayerKind::Full
            } else {
                LayerKind::Linear
            };
            assert_eq!(*k, want, "layer {i}");
        }
        // 36 linear + 12 full.
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|k| **k == LayerKind::Full)
                .count(),
            12
        );
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|k| **k == LayerKind::Linear)
                .count(),
            36
        );
    }

    /// The Step-0 traps that change no shape and pass every gate but the text
    /// gate (§20.1a): a wrong rope_theta / mrope_section, and the geometry that
    /// silently mis-runs. Pin the verified values so a regression is caught here.
    #[test]
    fn step0_traps_pinned() {
        let cfg = Config::flash_next();
        // rope_theta nested in rope_parameters in config.json (the LFM2 §5.2
        // nested-theta trap class) = 1e7.
        assert_eq!(cfg.rope_theta, 1e7);
        // mRoPE section [11,11,10] (+0 pad) — a wrong split corrupts positions.
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
        // partial rotary 0.25 → 64 of 256 head dims get RoPE.
        assert_eq!(cfg.partial_rotary_factor, 0.25);
        assert_eq!(
            (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize,
            64
        );
        // norm eps 1e-6.
        assert_eq!(cfg.norm_eps, 1e-6);
        // bos == eos == 248044 (the checkpoint has no distinct pad).
        assert_eq!(cfg.bos_token_id, cfg.eos_token_id);
        assert_eq!(cfg.eos_token_id, 248044);
        // The DeltaNet gate is SIGMOID (not silu) and S is fp32 — shape-blind
        // both (no parity against a composed chain sees them: the chain carries
        // the same choice). Pinned at the kernels, against hand-computed values
        // and at the boundary: `kernels::tests::the_gated_norm_gate_is_sigmoid`,
        // `the_step_refuses_a_state_that_is_not_f32` (#749 P33).
    }

    /// `--kv-quant` on this family: `off` is the f32 store, `f16` / `f32`
    /// name a dense store, `int8` / `int4` the codes; an unknown arm is an
    /// error, not a default. The flag is the derive's (`ConfigArgs`), and the
    /// axis it registers is LFM2's `KvQuant` with the same arms, so the
    /// registry does not grow.
    #[test]
    fn the_kv_axis_parses_this_familys_stores() {
        use crate::nn::ops::{KvPrecision, KvQuant};
        use clap::Parser;
        #[derive(Parser)]
        struct H {
            #[command(flatten)]
            axes: ConfigArgs,
        }
        let arm = |v: &str| -> Result<KvPrecision, String> {
            let mut cfg = Config::tiny();
            H::try_parse_from(["t", "--kv-quant", v])
                .map_err(|e| e.to_string())?
                .axes
                .apply(&mut cfg)?;
            Ok(cfg.kv_precision)
        };
        assert_eq!(Config::tiny().kv_precision, KvPrecision::dense(DType::F32));
        assert_eq!(arm("off").unwrap(), KvPrecision::dense(DType::F32));
        assert_eq!(arm("f32").unwrap(), KvPrecision::dense(DType::F32));
        assert_eq!(arm("f16").unwrap(), KvPrecision::dense(DType::F16));
        assert_eq!(arm("int8").unwrap().quant(), KvQuant::Int8);
        assert_eq!(arm("4").unwrap().quant(), KvQuant::Int4);
        assert!(arm("int9").is_err());
        assert_eq!(format!("{:?}", arm("f16").unwrap()), "Off:f16");
        let mut cfg = Config::tiny();
        cfg.set_axis("kv-quant", "int8").unwrap();
        assert_eq!(cfg.axes_line(), "KvQuant=Int8");
        let meta: Vec<_> = crate::axes::AXES
            .iter()
            .filter(|a| a.name == "KvQuant")
            .collect();
        assert!(meta.len() >= 2, "both families register KvQuant");
        assert!(
            meta.windows(2).all(|w| w[0] == w[1]),
            "the two KvQuant registrations differ: {meta:?}"
        );
    }

    /// The `#[tiny]` surface: every geometry field is tagged, the searched
    /// axes are read and set by name, and `tiny_derive` recomputes the
    /// derived fields from the axes — the layer pattern from the layer count
    /// and the interval, the DeltaNet value width from state × heads.
    #[test]
    fn the_tiny_surface_derives_the_dependent_fields() {
        use crate::testing::tiny::TinyRole;
        let axes: Vec<&str> = Config::TINY
            .iter()
            .filter(|f| f.role == TinyRole::Axis)
            .map(|f| f.name)
            .collect();
        assert_eq!(axes.len(), 19, "{axes:?}");
        assert!(Config::TINY
            .iter()
            .any(|f| f.name == "layer_types" && f.cover));
        let mut cfg = Config::tiny();
        assert_eq!(cfg.tiny_axis("n_layers"), Some(4));
        assert_eq!(
            cfg.tiny_axis("norm_eps"),
            None,
            "a fixed field is not an axis"
        );
        assert!(cfg.set_tiny_axis("n_layers", 8));
        assert!(cfg.set_tiny_axis("ssm_dt_rank", 6));
        assert!(!cfg.set_tiny_axis("vocab", 1));
        assert_eq!(cfg.layer_types.len(), 4, "stale until derived");
        cfg.tiny_derive();
        assert_eq!(cfg.layer_types, Config::resolve_layer_types(8, 4));
        assert_eq!(cfg.ssm_inner_size, 8 * 6);
        let mut same = Config::tiny();
        same.tiny_derive();
        assert_eq!(
            format!("{same:?}"),
            format!("{:?}", Config::tiny()),
            "tiny() is already derived"
        );
    }

    /// Derived dims reconcile with the GGUF tensor shapes (conv_channels must
    /// equal ssm_conv1d's channel dim = 10240; deltanet head geometry).
    #[test]
    fn derived_dims_match_checkpoint() {
        let cfg = Config::flash_next();
        // conv_channels = key_dim*2 + value_dim = (128*16)*2 + 128*48 = 4096+6144...
        // = 2048*2 + 6144 = 10240 (matches ssm_conv1d ne0).
        assert_eq!(cfg.conv_channels(), 10240);
        assert_eq!(cfg.hc_dim(), 4 * 2560); // 10240
        let d = cfg.deltanet_dims();
        assert_eq!((d.d_k, d.d_v, d.h_k, d.h_v), (128, 128, 16, 48));
        assert_eq!(cfg.ssm_inner_size, d.d_v * d.h_v); // 128*48 = 6144
    }
}
