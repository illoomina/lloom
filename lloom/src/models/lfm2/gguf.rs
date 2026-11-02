//! An LFM2 GGUF (the layout llama.cpp's converter writes) as a checkpoint
//! for this model: the config from the header's `lfm2.*` metadata, and the
//! tensor names this model asks for mapped onto the file's through the one
//! GGUF schema (`quantized_lfm2::weights::Lfm2Weights`). The weights load
//! through `nn::weight_repr::GgufBackend`: packed kinds stay packed.
//!
//! A GGUF computes at the model's dtype, as its safetensors checkpoint does
//! (`docs/models/lfm2.md` §5.10): at f16 the stream, the norms, the conv and
//! the projections run `(f16, f32, f16)`, and the packed head computes f32
//! logits (`docs/invariants.md` §15.2a). f32 is an arm, the contract of the
//! legacy GGUF model.
use super::{Config, LayerType};
use crate::models::quantized_lfm2::weights::{Lfm2Weights, MixerWeights};
use crate::nn::gguf_weights::WeightPlan;
use crate::nn::var_builder::LoadRecord;
use crate::nn::weight_repr::GgufBackend;
use crate::nn::VarBuilder;
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::{bail, DType, Device, Result};
use std::collections::HashMap;

/// The dtype the legacy GGUF model (`quantized_lfm2`) computes in: the f32
/// arm. [`GgufCheckpoint::var_builder`] builds at it, so a parity test
/// against that model compares like with like.
pub const LEGACY_COMPUTE_DTYPE: DType = DType::F32;

/// The dtypes a model built from a GGUF may compute in: f32, or f16 (the
/// quantized cores' `(f16, f32, f16)` arms). bf16 has no packed arm.
pub fn check_compute_dtype(dtype: DType) -> Result<DType> {
    match dtype {
        DType::F32 | DType::F16 => Ok(dtype),
        other => bail!(
            "lfm2 GGUF: no packed projection route computes at {}; f32 or f16",
            other.as_str()
        ),
    }
}

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
    /// `VarBuilder`, at [`LEGACY_COMPUTE_DTYPE`] on `device`.
    pub fn var_builder<'a, R: std::io::Read + std::io::Seek + Send + 'a>(
        &self,
        ct: Content,
        reader: R,
        device: &Device,
    ) -> VarBuilder<'a> {
        self.recording_var_builder(ct, reader, device, LEGACY_COMPUTE_DTYPE)
            .0
    }

    /// [`Self::var_builder`] at `dtype` ([`check_compute_dtype`]'s), and the
    /// record of which tensors it read.
    pub fn recording_var_builder<'a, R: std::io::Read + std::io::Seek + Send + 'a>(
        &self,
        ct: Content,
        reader: R,
        device: &Device,
        dtype: DType,
    ) -> (VarBuilder<'a>, LoadRecord) {
        let backend = GgufBackend::new(ct, reader, self.names.clone());
        let record = backend.record();
        let vb = VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        (vb, record)
    }

    /// The engine name → GGUF name map, for a test or a census.
    pub fn names(&self) -> &HashMap<String, String> {
        &self.names
    }

    /// The storage kinds `ct` holds the projections and the token table in
    /// (`f16`, `q8_0`, …), each with its count: the tensors
    /// [`super::Model::weight_kinds`] counts once built, read off the header
    /// before any weight is. A packed kind stays that kind in the model; a
    /// float kind is dequantized at the compute dtype by the load.
    pub fn stored_kinds(&self, ct: &Content) -> std::collections::BTreeMap<String, usize> {
        let mut kinds = std::collections::BTreeMap::new();
        for (engine, stored) in &self.names {
            if !is_projection_or_table(engine) {
                continue;
            }
            if let Some(info) = ct.tensor_infos.get(stored) {
                let kind = format!("{:?}", info.ggml_dtype).to_lowercase();
                *kinds.entry(kind).or_insert(0) += 1;
            }
        }
        kinds
    }
}

/// Whether `engine`, a name this model asks for, is a projection or the
/// token table: what `Model::weight_kinds` counts. The norms and the conv
/// taps are not.
fn is_projection_or_table(engine: &str) -> bool {
    const SUFFIXES: [&str; 9] = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.out_proj.weight",
        ".conv.in_proj.weight",
        ".conv.out_proj.weight",
        ".feed_forward.w1.weight",
        ".feed_forward.w2.weight",
        ".feed_forward.w3.weight",
    ];
    engine == "model.embed_tokens.weight"
        || engine == "lm_head.weight"
        || SUFFIXES.iter().any(|s| engine.ends_with(s))
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
