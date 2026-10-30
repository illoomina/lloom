//! An LFM2 checkpoint in the HF naming written as the GGUF llama.cpp's
//! converter writes: its tensor names, the conv taps at `[hidden, l_cache]`,
//! no `output.weight` when the head is tied, the `lfm2.*` metadata. The name
//! table here is its own, not the loader's schema, so a test of the loader
//! (`models::lfm2::gguf`) is not the loader checked against itself.
use crate::models::lfm2::{Config, LayerType};
use lloom_tensor::quantized::{gguf_file, GgmlDType, QTensor};
use lloom_tensor::{bail, Result, Tensor};
use std::collections::HashMap;

/// The llama.cpp name of the HF tensor `hf`.
pub fn llama_cpp_name(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.embedding_norm.weight" => return Some("token_embd_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, leaf) = rest.split_once('.')?;
    let leaf = match leaf {
        "operator_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "feed_forward.w1.weight" => "ffn_gate.weight",
        "feed_forward.w3.weight" => "ffn_up.weight",
        "feed_forward.w2.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.out_proj.weight" => "attn_output.weight",
        "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
        "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
        "conv.in_proj.weight" => "shortconv.in_proj.weight",
        "conv.out_proj.weight" => "shortconv.out_proj.weight",
        "conv.conv.weight" => "shortconv.conv.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{leaf}"))
}

/// `tensors` (HF names, any float dtype) as an LFM2 GGUF for `cfg`: the
/// projections and the token table at `packed`, every other tensor F32,
/// `extra` metadata (a tokenizer, a template) beside the `lfm2.*` keys.
pub fn write(
    cfg: &Config,
    tensors: &HashMap<String, Tensor>,
    packed: GgmlDType,
    extra: Vec<(String, gguf_file::Value)>,
) -> Result<Vec<u8>> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut qs = Vec::with_capacity(names.len());
    for hf in names {
        let Some(name) = llama_cpp_name(hf) else {
            bail!("no llama.cpp name for {hf}")
        };
        let mut t = tensors[hf].to_dtype(lloom_tensor::DType::F32)?;
        if hf.ends_with("conv.conv.weight") {
            t = t.squeeze(1)?;
        }
        let kind = if t.rank() == 2 && !hf.ends_with("conv.conv.weight") {
            packed
        } else {
            GgmlDType::F32
        };
        qs.push((name, QTensor::quantize(&t, kind)?));
    }
    use gguf_file::Value as V;
    let kv = cfg
        .layer_types
        .iter()
        .map(|k| {
            V::U32(match k {
                LayerType::FullAttention => cfg.num_key_value_heads as u32,
                LayerType::Conv => 0,
            })
        })
        .collect();
    let mut metadata: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::String("lfm2".into())),
        (
            "lfm2.attention.head_count".into(),
            V::U32(cfg.num_attention_heads as u32),
        ),
        ("lfm2.attention.head_count_kv".into(), V::Array(kv)),
        (
            "lfm2.embedding_length".into(),
            V::U32(cfg.hidden_size as u32),
        ),
        (
            "lfm2.feed_forward_length".into(),
            V::U32(cfg.intermediate_size as u32),
        ),
        (
            "lfm2.context_length".into(),
            V::U32(cfg.max_position_embeddings as u32),
        ),
        (
            "lfm2.block_count".into(),
            V::U32(cfg.num_hidden_layers as u32),
        ),
        (
            "lfm2.attention.layer_norm_rms_epsilon".into(),
            V::F32(cfg.norm_eps as f32),
        ),
        ("lfm2.rope.freq_base".into(), V::F32(cfg.rope_theta)),
        (
            "lfm2.shortconv.l_cache".into(),
            V::U32(cfg.conv_l_cache as u32),
        ),
        ("lfm2.vocab_size".into(), V::U32(cfg.vocab_size as u32)),
    ];
    metadata.extend(extra);
    let mut bytes = std::io::Cursor::new(Vec::new());
    gguf_file::write(
        &mut bytes,
        &metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        &qs.iter().map(|(k, v)| (k.as_str(), v)).collect::<Vec<_>>(),
    )?;
    Ok(bytes.into_inner())
}
