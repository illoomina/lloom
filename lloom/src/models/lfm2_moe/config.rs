use crate::models::lfm2::{self, Lfm2Config};
use lloom_tensor::{bail, Result};

/// Checkpoint fields that distinguish LFM2 MoE from the dense LFM2 schema.
/// The dense width is explicit, independently of the legacy auto-adjust flag.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub model_type: String,
    pub rope_parameters: RopeParameters,
    pub num_dense_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub use_expert_bias: bool,
    #[serde(flatten)]
    pub base: Lfm2Config,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f32,
}

/// Validated FFN composition carried by the shared LFM runtime configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub num_dense_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub use_expert_bias: bool,
}

impl MoeConfig {
    pub fn validate(&self, cfg: &lfm2::Config) -> Result<()> {
        if cfg.hidden_size == 0
            || cfg.num_hidden_layers == 0
            || cfg.conv_bias
            || self.num_dense_layers > cfg.num_hidden_layers
            || self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.moe_intermediate_size == 0
            || !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.0
            || cfg.layer_types.len() != cfg.num_hidden_layers
        {
            bail!("lfm2_moe: invalid runtime FFN composition")
        }
        let size = self
            .num_experts
            .checked_mul(self.moe_intermediate_size)
            .and_then(|n| n.checked_mul(cfg.hidden_size))
            .and_then(|n| n.checked_mul(24));
        if size.is_none() || self.checked_activation_bound(cfg.hidden_size).is_none() {
            bail!("lfm2_moe: expert geometry overflows its byte budget")
        }
        Ok(())
    }

    pub fn is_dense(&self, layer: usize) -> bool {
        layer < self.num_dense_layers
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let b = &self.base;
        if self.model_type != "lfm2_moe" {
            bail!("expected model_type lfm2_moe, got {}", self.model_type)
        }
        if b.hidden_size == 0 || b.vocab_size == 0 || b.num_hidden_layers == 0 {
            bail!("lfm2_moe: hidden, vocabulary and layer count must be positive")
        }
        if b.num_attention_heads == 0
            || b.num_key_value_heads == 0
            || !b.hidden_size.is_multiple_of(b.num_attention_heads)
            || !b.num_attention_heads.is_multiple_of(b.num_key_value_heads)
            || !(b.hidden_size / b.num_attention_heads).is_multiple_of(2)
        {
            bail!("lfm2_moe: invalid GQA/rotary geometry")
        }
        if b.layer_types.len() != b.num_hidden_layers {
            bail!("lfm2_moe: layer_types must name every layer")
        }
        if self.num_dense_layers > b.num_hidden_layers
            || self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.moe_intermediate_size == 0
            || b.intermediate_size.is_none_or(|n| n == 0)
        {
            bail!("lfm2_moe: invalid dense/expert FFN geometry")
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            bail!("lfm2_moe: routed_scaling_factor must be finite and positive")
        }
        if b.conv_bias {
            bail!("lfm2_moe: conv_bias=true is not implemented by the shared LFM mixer")
        }
        if !b.norm_eps.is_finite()
            || b.norm_eps <= 0.0
            || b.conv_l_cache == 0
            || b.max_position_embeddings == 0
        {
            bail!("lfm2_moe: invalid normalization/conv geometry")
        }
        if self.rope_parameters.rope_type != "default" {
            bail!("lfm2_moe: only default RoPE is implemented")
        }
        if !self.rope_parameters.rope_theta.is_finite() || self.rope_parameters.rope_theta <= 0.0 {
            bail!("lfm2_moe: rope_theta must be finite and positive")
        }
        Ok(())
    }

    /// LFM's existing mixer/state configuration, without changing dense LFM's
    /// interpretation of an omitted legacy auto-adjust field.
    pub fn lfm_config(&self) -> Result<lfm2::Config> {
        self.validate()?;
        let mut base = self.base.clone();
        base.block_auto_adjust_ff_dim = false;
        base.rope_parameters = Some(lfm2::Lfm2RopeParameters {
            rope_theta: Some(self.rope_parameters.rope_theta),
        });
        let mut runtime = base.into_config(false);
        runtime.moe = Some(MoeConfig {
            num_dense_layers: self.num_dense_layers,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            use_expert_bias: self.use_expert_bias,
        });
        runtime
            .moe
            .expect("MoE composition was set")
            .validate(&runtime)?;
        Ok(runtime)
    }

    pub fn is_dense(&self, layer: usize) -> bool {
        layer < self.num_dense_layers
    }
}

/// Parse a text-only LFM checkpoint without importing dense FFN defaults into
/// the MoE architecture. VL callers first select their nested text_config.
pub fn parse_lfm_config(root: serde_json::Value) -> Result<lfm2::Config> {
    if root.get("model_type").and_then(|v| v.as_str()) == Some("lfm2_moe") {
        let cfg: Config = serde_json::from_value(root).map_err(lloom_tensor::Error::wrap)?;
        cfg.lfm_config()
    } else {
        let cfg: Lfm2Config = serde_json::from_value(root).map_err(lloom_tensor::Error::wrap)?;
        Ok(cfg.into_config(false))
    }
}

impl MoeConfig {
    /// Conservative retained-activation allowance for one position in a routed
    /// layer: gate/up/SiLU/product and staging copies, expert outputs, carries,
    /// and routing/sort/weight temporaries. Routed intermediates are F32 even
    /// when the LFM residual is F16. This is a bound, not measured liveness.
    pub fn activation_bound(&self, hidden: usize) -> usize {
        // Invalid programmatic geometry must over-predict, never wrap into
        // a small admission. Constructors reject it through validate().
        self.checked_activation_bound(hidden).unwrap_or(usize::MAX)
    }

    /// Load-time packing through the existing QTensor CPU-to-device adapter:
    /// a projection's source/stack/cast/packed/upload buffers, separately from
    /// resident weights. Only one projection is packed at a time.
    pub fn packing_bound(&self, hidden: usize, dtype: lloom_tensor::DType) -> usize {
        self.num_experts
            .checked_mul(self.moe_intermediate_size)
            .and_then(|n| n.checked_mul(hidden))
            .and_then(|n| n.checked_mul(4 + 5 * dtype.size_in_bytes()))
            .unwrap_or(usize::MAX)
    }

    fn checked_activation_bound(&self, hidden: usize) -> Option<usize> {
        let gate_up = self
            .num_experts_per_tok
            .checked_mul(self.moe_intermediate_size)?
            .checked_mul(6)?;
        let outputs = self
            .num_experts_per_tok
            .checked_mul(hidden)?
            .checked_mul(4)?;
        let carries = hidden.checked_mul(8)?;
        let routing = self.num_experts.checked_mul(10)?;
        gate_up
            .checked_add(outputs)?
            .checked_add(carries)?
            .checked_add(routing)?
            .checked_mul(4)
    }
}
