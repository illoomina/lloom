//! The Qwen3.5 architecture's runtime config: `qwen3_5` (dense) and
//! `qwen3_5_moe`, one struct. Every field names the GGUF key it is read
//! from (`qwen35.*` / `qwen35moe.*`, llama.cpp `src/models/qwen35.cpp`,
//! `qwen35moe.cpp` `load_arch_hparams`) and the `config.json` field it
//! stands for (Qwen/Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B, read
//! 2026-10-07). This is the `Model::Config`: a runtime struct, not the wire
//! format.

use crate::models::qwen4exp::{AttnGeometry, DeltaNetDims, GdnGeometry, LayerKind, MoeGeometry};
use crate::nn::gdn::OutGate;
use crate::nn::ops::KvPrecision;
use lloom_macros::Config as ConfigDerive;
use lloom_tensor::DType;

/// The GGUF architecture names this family loads.
pub const ARCH_DENSE: &str = "qwen35";
pub const ARCH_MOE: &str = "qwen35moe";
/// Cloudflare's clef (`general.architecture = clef`): Qwen3.8-27B's text
/// stack under its own name, every `qwen35.*` key spelled `clef.*`, plus a
/// decision head's tensors and keys (`clef.decision.*`) the text path does
/// not read. Loaded as the dense architecture.
pub const ARCH_CLEF: &str = "clef";

/// The verified geometry and the runtime axes; `#[tiny]` marks what the
/// shape campaign may search from [`Config::tiny`].
#[derive(Debug, Clone, ConfigDerive)]
pub struct Config {
    // ---- top-level ----
    /// The decoder stack's depth (`num_hidden_layers`): 64 on the 27B, 40 on
    /// the 35B-A3B. The GGUF's `block_count` counts the MTP block too when
    /// the file carries one (`nextn_predict_layers`, llama.cpp's
    /// `n_layer = n_layer_all − n_layer_nextn`): unsloth's Qwen3.8-27B is
    /// `block_count 65`, `nextn_predict_layers 1`.
    #[tiny(axis, max = 8)]
    pub n_layers: usize,
    /// `nextn_predict_layers` (`mtp_num_hidden_layers`): the MTP blocks the
    /// file carries after the stack, `blk.{n_layers}..`; 0 when stripped
    /// (unsloth's Qwen3.6 quants) or absent. Not run by the decoder.
    #[tiny(fixed)]
    pub n_nextn_layers: usize,
    /// `embedding_length` (`hidden_size`): 5120 / 2048.
    #[tiny(axis)]
    pub hidden_size: usize,
    /// `token_embd.weight`'s row count (`vocab_size` 248320).
    #[tiny(axis)]
    pub vocab_size: usize,
    /// `feed_forward_length` (`intermediate_size`): 17408 on the dense
    /// models; the MoE's routed width is [`Self::moe_intermediate_size`].
    #[tiny(axis)]
    pub intermediate_size: usize,
    /// `attention.layer_norm_rms_epsilon` = 1e-6.
    #[tiny(fixed)]
    pub norm_eps: f64,
    /// `full_attention_interval` = 4: every 4th layer is full attention.
    #[tiny(axis, max = 8)]
    pub full_attention_interval: usize,
    /// Per-layer kind, from `layer_types` / the interval — never hardcoded.
    #[tiny(
        derived = "Self::resolve_layer_types(n_layers, full_attention_interval)",
        cover
    )]
    pub layer_types: Vec<LayerKind>,

    // ---- full attention (gated GQA) ----
    /// `attention.head_count` (`num_attention_heads`): 24 / 16.
    #[tiny(axis)]
    pub n_heads: usize,
    /// `attention.head_count_kv` (`num_key_value_heads`): 4 / 2.
    #[tiny(axis)]
    pub n_kv_heads: usize,
    /// `attention.key_length` = `value_length` (`head_dim`) = 256.
    #[tiny(axis)]
    pub head_dim: usize,
    /// `rope.dimension_count / head_dim` (`partial_rotary_factor` 0.25).
    #[tiny(fixed)]
    pub partial_rotary_factor: f64,
    /// `rope.freq_base` (`rope_theta`) = 1e7.
    #[tiny(fixed)]
    pub rope_theta: f64,
    /// `rope.dimension_sections` (`mrope_section` [11, 11, 10], + 0): text
    /// positions coincide, so decode ropes the scalar position.
    #[tiny(fixed)]
    pub mrope_section: [usize; 4],

    // ---- Gated DeltaNet (the `linear_attention` layers) ----
    /// `ssm.state_size` (`linear_key_head_dim` = `linear_value_head_dim`) = 128.
    #[tiny(axis)]
    pub ssm_state_size: usize,
    /// `ssm.group_count` (`linear_num_key_heads`) = 16.
    #[tiny(axis)]
    pub ssm_n_group: usize,
    /// `ssm.time_step_rank` (`linear_num_value_heads`): 48 / 32.
    #[tiny(axis)]
    pub ssm_dt_rank: usize,
    /// `ssm.conv_kernel` (`linear_conv_kernel_dim`) = 4.
    #[tiny(fixed)]
    pub ssm_conv_kernel: usize,
    /// `ssm.inner_size` = `state_size · dt_rank` (6144 / 4096).
    #[tiny(derived = "ssm_state_size * ssm_dt_rank")]
    pub ssm_inner_size: usize,
    /// The gated norm's output gate: `output_gate_type` is `swish` on every
    /// Qwen3.5 checkpoint (the GGUF carries no key; the architecture
    /// implies it, llama.cpp `build_norm_gated` → `ggml_silu`).
    #[tiny(fixed)]
    pub out_gate: OutGate,

    // ---- MoE (`qwen3_5_moe` only; `n_experts` 0 on the dense models) ----
    /// `expert_count` (`num_experts`): 256; 0 for a dense FFN.
    #[tiny(axis)]
    pub n_experts: usize,
    /// `expert_used_count` (`num_experts_per_tok`): 8.
    #[tiny(axis)]
    pub n_experts_per_tok: usize,
    /// `expert_feed_forward_length` (`moe_intermediate_size`): 512.
    #[tiny(axis)]
    pub moe_intermediate_size: usize,
    /// `expert_shared_feed_forward_length` (`shared_expert_intermediate_size`): 512.
    #[tiny(axis)]
    pub shared_expert_intermediate_size: usize,

    // ---- tokens ----
    /// `tokenizer.ggml.bos_token_id` (248044, `<|endoftext|>`).
    #[tiny(fixed)]
    pub bos_token_id: u32,
    /// `tokenizer.ggml.eos_token_id` (248046, `<|im_end|>`).
    #[tiny(fixed)]
    pub eos_token_id: u32,

    // ---- runtime axes ----
    /// The full-attention layers' KV store (`sequence-state.md` §10.7).
    /// `Off` is the f32 store that ships (the attention block is
    /// Qwen4-exp's); the reduced stores are built but ungated on this
    /// family, so the engine serves `off` only.
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
    /// Linear everywhere except every `interval`-th layer (`(i+1) % interval
    /// == 0`), the reference's rule (`qwen35.cpp` `load_arch_hparams`) and
    /// the checkpoints' `layer_types` array (`[L,L,L,F]` repeated).
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

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }

    /// The GGUF architecture this config loads as.
    pub fn arch(&self) -> &'static str {
        if self.is_moe() {
            ARCH_MOE
        } else {
            ARCH_DENSE
        }
    }

    pub fn deltanet_dims(&self) -> DeltaNetDims {
        DeltaNetDims {
            d_k: self.ssm_state_size,
            d_v: self.ssm_state_size,
            h_k: self.ssm_n_group,
            h_v: self.ssm_dt_rank,
        }
    }

    /// The DeltaNet concat-QKV conv channel count, `key_dim · 2 + value_dim`.
    pub fn conv_channels(&self) -> usize {
        self.gdn().conv_channels()
    }

    /// The Gated DeltaNet block's geometry.
    pub fn gdn(&self) -> GdnGeometry {
        GdnGeometry {
            dims: self.deltanet_dims(),
            conv_kernel: self.ssm_conv_kernel,
            norm_eps: self.norm_eps,
            gate: self.out_gate,
        }
    }

    /// The gated attention block's geometry.
    pub fn attn(&self) -> AttnGeometry {
        AttnGeometry {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            n_rot: (self.head_dim as f64 * self.partial_rotary_factor).round() as usize,
            rope_theta: self.rope_theta,
        }
    }

    /// The routed MoE's geometry (meaningful when [`Self::is_moe`]).
    pub fn moe(&self) -> MoeGeometry {
        MoeGeometry {
            n_experts: self.n_experts,
            n_experts_per_tok: self.n_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
        }
    }

    /// Whether layer `l` is a full-attention layer.
    pub fn is_full(&self, l: usize) -> bool {
        self.layer_types.get(l) == Some(&LayerKind::Full)
    }

    /// Qwen3.6-27B / Qwen3.8-27B (`config.json` `text_config`, read
    /// 2026-10-07; the GGUF headers of unsloth's quants agree: `block_count`
    /// 64 with the MTP block stripped, 65 with it kept).
    pub fn qwen3_6_27b() -> Self {
        let n_layers = 64;
        let full_attention_interval = 4;
        Config {
            n_layers,
            n_nextn_layers: 1,
            hidden_size: 5120,
            vocab_size: 248320,
            intermediate_size: 17408,
            norm_eps: 1e-6,
            full_attention_interval,
            layer_types: Self::resolve_layer_types(n_layers, full_attention_interval),
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10, 0],
            ssm_state_size: 128,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ssm_conv_kernel: 4,
            ssm_inner_size: 6144,
            out_gate: OutGate::Silu,
            n_experts: 0,
            n_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            bos_token_id: 248044,
            eos_token_id: 248046,
            kv_precision: DEFAULT_KV_PRECISION,
        }
    }

    /// Qwen3.6-35B-A3B (`config.json` `text_config`, read 2026-10-07).
    pub fn qwen3_6_35b_a3b() -> Self {
        let n_layers = 40;
        let full_attention_interval = 4;
        Config {
            n_layers,
            n_nextn_layers: 1,
            hidden_size: 2048,
            vocab_size: 248320,
            intermediate_size: 0,
            norm_eps: 1e-6,
            full_attention_interval,
            layer_types: Self::resolve_layer_types(n_layers, full_attention_interval),
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10, 0],
            ssm_state_size: 128,
            ssm_n_group: 16,
            ssm_dt_rank: 32,
            ssm_conv_kernel: 4,
            ssm_inner_size: 4096,
            out_gate: OutGate::Silu,
            n_experts: 256,
            n_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            bos_token_id: 248044,
            eos_token_id: 248046,
            kv_precision: DEFAULT_KV_PRECISION,
        }
    }

    /// The dense family at tiny geometry: every block type (`[L, L, L, F]`),
    /// small dims, a CPU forward in milliseconds.
    pub fn tiny() -> Self {
        let n_layers = 4;
        let full_attention_interval = 4;
        Config {
            n_layers,
            n_nextn_layers: 0,
            hidden_size: 32,
            vocab_size: 128,
            intermediate_size: 48,
            norm_eps: 1e-6,
            full_attention_interval,
            layer_types: Self::resolve_layer_types(n_layers, full_attention_interval),
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            mrope_section: [3, 3, 2, 0],
            ssm_state_size: 8,
            ssm_n_group: 2,
            ssm_dt_rank: 4,
            ssm_conv_kernel: 4,
            ssm_inner_size: 32,
            out_gate: OutGate::Silu,
            n_experts: 0,
            n_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            bos_token_id: 1,
            eos_token_id: 2,
            kv_precision: DEFAULT_KV_PRECISION,
        }
    }

    /// The MoE family at tiny geometry: [`Self::tiny`]'s layers with routed
    /// and shared experts.
    pub fn tiny_moe() -> Self {
        Config {
            intermediate_size: 0,
            n_experts: 8,
            n_experts_per_tok: 2,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            ..Self::tiny()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer-type pattern is `[L,L,L,F]` repeated — 48 + 16 on the 27B,
    /// 30 + 10 on the 35B-A3B — as the checkpoints' `layer_types` arrays say.
    #[test]
    fn layer_types_are_lllf_repeated() {
        for (cfg, n, full) in [
            (Config::qwen3_6_27b(), 64, 16),
            (Config::qwen3_6_35b_a3b(), 40, 10),
        ] {
            assert_eq!(cfg.layer_types.len(), n);
            for (i, k) in cfg.layer_types.iter().enumerate() {
                let want = if (i + 1) % 4 == 0 {
                    LayerKind::Full
                } else {
                    LayerKind::Linear
                };
                assert_eq!(*k, want, "layer {i}");
            }
            assert_eq!(
                cfg.layer_types
                    .iter()
                    .filter(|k| **k == LayerKind::Full)
                    .count(),
                full
            );
        }
    }

    /// The Step-0 traps that change no shape and pass every gate but the
    /// text gate: the rope, the partial rotation, the gate kind, the conv
    /// channel count.
    #[test]
    fn step0_traps_pinned() {
        let cfg = Config::qwen3_6_27b();
        assert_eq!(cfg.rope_theta, 1e7);
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
        assert_eq!(cfg.attn().n_rot, 64);
        assert_eq!(
            cfg.out_gate,
            OutGate::Silu,
            "swish, not Qwen4-exp's sigmoid"
        );
        // conv_channels = key_dim·2 + value_dim = 2048·2 + 6144 (ssm_conv1d's ne1).
        assert_eq!(cfg.conv_channels(), 10240);
        assert_eq!(
            cfg.ssm_inner_size,
            cfg.deltanet_dims().d_v * cfg.deltanet_dims().h_v
        );
        assert!(!cfg.is_moe());
        let moe = Config::qwen3_6_35b_a3b();
        assert!(moe.is_moe());
        assert_eq!(moe.conv_channels(), 2048 * 2 + 4096);
        assert_eq!(moe.moe().n_experts_per_tok, 8);
        assert_eq!((cfg.arch(), moe.arch()), (ARCH_DENSE, ARCH_MOE));
    }

    /// The `#[tiny]` surface derives the dependent fields from the axes.
    #[test]
    fn the_tiny_surface_derives_the_dependent_fields() {
        let mut cfg = Config::tiny();
        assert!(cfg.set_tiny_axis("n_layers", 8));
        assert!(cfg.set_tiny_axis("ssm_dt_rank", 6));
        cfg.tiny_derive();
        assert_eq!(cfg.layer_types, Config::resolve_layer_types(8, 4));
        assert_eq!(cfg.ssm_inner_size, 8 * 6);
        let mut same = Config::tiny_moe();
        same.tiny_derive();
        assert_eq!(format!("{same:?}"), format!("{:?}", Config::tiny_moe()));
    }
}
