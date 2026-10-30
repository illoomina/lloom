//! An LFM2 GGUF (the layout llama.cpp's converter writes) as a checkpoint
//! for this model: the config from the header's `lfm2.*` metadata, and the
//! tensor names this model asks for mapped onto the file's through the one
//! GGUF schema (`quantized_lfm2::weights::Lfm2Weights`). The weights load
//! through `nn::weight_repr::GgufBackend`: packed kinds stay packed.
//!
//! The model computes at f32 over a GGUF, the numerical contract GGUF
//! serving has always had (`docs/models/lfm2.md` §5.10): Metal's quantized
//! GEMM reads f32 activations past the matmul-vec's rows.
use super::{Config, LayerType};
use crate::models::quantized_lfm2::weights::{Lfm2Weights, MixerWeights};
use crate::nn::gguf_weights::WeightPlan;
use crate::nn::var_builder::LoadRecord;
use crate::nn::weight_repr::GgufBackend;
use crate::nn::VarBuilder;
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::{bail, DType, Device, Result};
use std::collections::HashMap;

/// The dtype a model built from a GGUF computes in.
pub const GGUF_COMPUTE_DTYPE: DType = DType::F32;

/// A validated LFM2 GGUF header: the model's config and the name map.
#[derive(Debug, Clone)]
pub struct GgufCheckpoint {
    pub config: Config,
    names: HashMap<String, String>,
}

impl GgufCheckpoint {
    /// Resolve `ct` against the LFM2 GGUF schema and validate every tensor
    /// header the model will read, before any weight is read.
    pub fn read(ct: &Content) -> Result<Self> {
        let plan = Lfm2Weights::resolve(ct)?;
        plan.validate_headers(ct)?;
        let rows = |name: &str| -> Result<usize> {
            match ct.tensor_infos.get(name) {
                Some(info) => Ok(info.shape.dims2()?.0),
                None => bail!("lfm2 GGUF: no tensor header {name}"),
            }
        };
        let intermediate = rows(plan.layers[0].gate.name())?;
        let mut kv_heads = None;
        let mut layer_types = Vec::with_capacity(plan.layers.len());
        for (l, layer) in plan.layers.iter().enumerate() {
            if rows(layer.gate.name())? != intermediate {
                bail!(
                    "lfm2 GGUF: layer {l}'s FFN width differs from layer 0's {intermediate}; \
                     models::lfm2 carries one width"
                )
            }
            match &layer.mixer {
                MixerWeights::Attention(a) => {
                    if kv_heads.is_some_and(|kv| kv != a.kv_heads) {
                        bail!(
                            "lfm2 GGUF: attention layers carry different KV-head counts \
                             ({} and {}); models::lfm2 carries one",
                            kv_heads.unwrap_or_default(),
                            a.kv_heads
                        )
                    }
                    kv_heads = Some(a.kv_heads);
                    layer_types.push(LayerType::FullAttention);
                }
                MixerWeights::Conv(_) => layer_types.push(LayerType::Conv),
            }
        }
        let token = |key: &str| -> Result<Option<u32>> {
            ct.metadata.get(key).map(|v| v.to_u32()).transpose()
        };
        let config = Config {
            vocab_size: plan.vocab,
            hidden_size: plan.width,
            intermediate_size: intermediate,
            num_hidden_layers: plan.layers.len(),
            num_attention_heads: plan.heads,
            num_key_value_heads: kv_heads.unwrap_or(plan.heads),
            norm_eps: plan.epsilon,
            rope_theta: plan.rope_base,
            max_position_embeddings: plan.context,
            conv_l_cache: plan.kernel,
            conv_bias: false,
            layer_types,
            tie_embedding: plan.output.is_none(),
            bos_token_id: token("tokenizer.ggml.bos_token_id")?,
            eos_token_id: token("tokenizer.ggml.eos_token_id")?,
            ..Config::default()
        };
        Ok(Self {
            config,
            names: engine_names(&plan),
        })
    }

    /// The weights of `ct` (read through `reader`) as this model's
    /// `VarBuilder`, at [`GGUF_COMPUTE_DTYPE`] on `device`.
    pub fn var_builder<'a, R: std::io::Read + std::io::Seek + Send + 'a>(
        &self,
        ct: Content,
        reader: R,
        device: &Device,
    ) -> VarBuilder<'a> {
        self.recording_var_builder(ct, reader, device).0
    }

    /// [`Self::var_builder`], and the record of which tensors it read.
    pub fn recording_var_builder<'a, R: std::io::Read + std::io::Seek + Send + 'a>(
        &self,
        ct: Content,
        reader: R,
        device: &Device,
    ) -> (VarBuilder<'a>, LoadRecord) {
        let backend = GgufBackend::new(ct, reader, self.names.clone());
        let record = backend.record();
        let vb = VarBuilder::from_backend(Box::new(backend), GGUF_COMPUTE_DTYPE, device.clone());
        (vb, record)
    }

    /// The engine name → GGUF name map, for a test or a census.
    pub fn names(&self) -> &HashMap<String, String> {
        &self.names
    }
}

/// The names `models::lfm2` asks for (the HF checkpoint's), each mapped to
/// the tensor the schema resolved.
fn engine_names(plan: &Lfm2Weights) -> HashMap<String, String> {
    let mut names = HashMap::new();
    let mut put = |engine: String, stored: &str| {
        names.insert(engine, stored.to_string());
    };
    put("model.embed_tokens.weight".into(), plan.embedding.name());
    put("model.embedding_norm.weight".into(), plan.norm.name());
    if let Some(output) = &plan.output {
        put("lm_head.weight".into(), output.name());
    }
    for (l, layer) in plan.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        put(
            format!("{p}.operator_norm.weight"),
            layer.operator_norm.name(),
        );
        put(format!("{p}.ffn_norm.weight"), layer.ffn_norm.name());
        put(format!("{p}.feed_forward.w1.weight"), layer.gate.name());
        put(format!("{p}.feed_forward.w3.weight"), layer.up.name());
        put(format!("{p}.feed_forward.w2.weight"), layer.down.name());
        match &layer.mixer {
            MixerWeights::Attention(a) => {
                let a_ = format!("{p}.self_attn");
                put(format!("{a_}.q_proj.weight"), a.q.name());
                put(format!("{a_}.k_proj.weight"), a.k.name());
                put(format!("{a_}.v_proj.weight"), a.v.name());
                put(format!("{a_}.out_proj.weight"), a.output.name());
                put(format!("{a_}.q_layernorm.weight"), a.q_norm.name());
                put(format!("{a_}.k_layernorm.weight"), a.k_norm.name());
            }
            MixerWeights::Conv(c) => {
                put(format!("{p}.conv.in_proj.weight"), c.input.name());
                put(format!("{p}.conv.out_proj.weight"), c.output.name());
                put(format!("{p}.conv.conv.weight"), c.conv.name());
            }
        }
    }
    names
}

#[cfg(test)]
mod tests;
