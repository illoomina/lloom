//! GGUF loader for Qwen3.8-Flash-Next: the `qwen4exp.*` header → `Config`, the
//! tensor table → the blocks, on the `Resident` path (every expert stack in
//! memory, packed).
//!
//! Tensor names follow the llama.cpp converter (PR #27742); shapes are candle
//! row-major (ggml `ne` reversed), so `ffn_gate_exps` is `[n_expert, n_ff,
//! hidden]` and `ffn_down_exps` `[n_expert, hidden, n_ff]` — the routed
//! kernel's `[E, n, k]`. Quantized matrices stay packed behind `QMatMul`; the
//! BF16 matrices (HyperConnection low-rank, routers, DeltaNet α/β) become f16
//! tensors (`QMatMul::TensorF16` — f16 GEMV, f32 accumulate) rather than f32,
//! which halves their footprint; f32 vectors (norms, conv, `ssm_a`, `dt.bias`)
//! are dequantized as they are.

use super::config::{Config, LayerKind};
use super::deltanet_block::GatedDeltaNet;
use super::hc::HyperConnection;
use super::model::{Layer, Mixer, Qwen4Exp};
use super::moe::{MoeLayer, Resident};
use super::ple::{MmapPleTable, NgramHash, PleLayer, PleTable};
use super::qsa::{Indexer, QsaAttention};
use crate::nn::RmsNorm;
use lloom_tensor::quantized::gguf_file::{Content, Value};
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

fn md_u32(md: &HashMap<String, Value>, key: &str) -> Result<usize> {
    match md.get(key) {
        Some(v) => Ok(v.to_u32()? as usize),
        None => lloom_tensor::bail!("gguf metadata has no `{key}`"),
    }
}
fn md_f32(md: &HashMap<String, Value>, key: &str) -> Result<f64> {
    match md.get(key) {
        Some(v) => Ok(v.to_f32()? as f64),
        None => lloom_tensor::bail!("gguf metadata has no `{key}`"),
    }
}
fn value_as_i64(v: &Value) -> Result<i64> {
    Ok(match v {
        Value::U8(x) => *x as i64,
        Value::I8(x) => *x as i64,
        Value::U16(x) => *x as i64,
        Value::I16(x) => *x as i64,
        Value::U32(x) => *x as i64,
        Value::I32(x) => *x as i64,
        Value::U64(x) => *x as i64,
        Value::I64(x) => *x,
        other => lloom_tensor::bail!("gguf metadata value {other:?} is not an integer"),
    })
}
/// An integer array of any width (the converter writes `ple.*` as u64, the
/// layer arrays as i32).
fn md_int_array(md: &HashMap<String, Value>, key: &str) -> Result<Vec<i64>> {
    match md.get(key) {
        Some(v) => v.to_vec()?.iter().map(value_as_i64).collect(),
        None => lloom_tensor::bail!("gguf metadata has no `{key}`"),
    }
}
fn md_i32_array(md: &HashMap<String, Value>, key: &str) -> Result<Vec<i64>> {
    md_int_array(md, key)
}

impl Config {
    /// The geometry from a `qwen4exp` GGUF header. Every field is read from its
    /// key (none hardcoded); the per-layer kinds come from
    /// `full_attention_interval` and are cross-checked against the
    /// `attention.compress_ratios` array (non-zero on exactly the full-attention
    /// layers) so a converter that disagrees with the interval is refused rather
    /// than silently run with the wrong mixer on 12 layers.
    pub fn from_gguf_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        if arch != "qwen4exp" {
            lloom_tensor::bail!("expected general.architecture = qwen4exp, got `{arch}`")
        }
        let n_layers = md_u32(md, "qwen4exp.block_count")?;
        let full_attention_interval = md_u32(md, "qwen4exp.full_attention_interval")?;
        let layer_types = Self::resolve_layer_types(n_layers, full_attention_interval);
        let ratios = md_i32_array(md, "qwen4exp.attention.compress_ratios")?;
        if ratios.len() != n_layers {
            lloom_tensor::bail!(
                "attention.compress_ratios has {} entries for {n_layers} layers",
                ratios.len()
            )
        }
        let mut compress_ratio = 0usize;
        for (l, (kind, r)) in layer_types.iter().zip(&ratios).enumerate() {
            let is_full = *r != 0;
            if is_full != (*kind == LayerKind::Full) {
                lloom_tensor::bail!(
                    "layer {l}: compress_ratios says {} but the interval says {kind:?}",
                    if is_full { "full" } else { "linear" }
                )
            }
            if is_full {
                compress_ratio = *r as usize;
            }
        }
        let sections = md_i32_array(md, "qwen4exp.rope.dimension_sections")?;
        let mut mrope_section = [0usize; 4];
        for (dst, src) in mrope_section.iter_mut().zip(&sections) {
            *dst = *src as usize;
        }
        let ple_layers = md
            .get("qwen4exp.ple.layers")
            .map(|v| -> Result<Vec<usize>> {
                v.to_vec()?
                    .iter()
                    .map(|x| Ok(x.to_i32()? as usize))
                    .collect()
            })
            .transpose()?
            .unwrap_or_default();
        let eos = md
            .get("qwen4exp.ple.eos_token_id")
            .or_else(|| md.get("tokenizer.ggml.eos_token_id"))
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(248044);
        let head_dim = md_u32(md, "qwen4exp.attention.key_length")?;
        let n_rot = md_u32(md, "qwen4exp.rope.dimension_count")?;
        Ok(Config {
            n_layers,
            hidden_size: md_u32(md, "qwen4exp.embedding_length")?,
            // The vocabulary is the embedding table's row count; read from the
            // tensor by the loader (the header carries no vocab_size key).
            vocab_size: 0,
            norm_eps: md_f32(md, "qwen4exp.attention.layer_norm_rms_epsilon")?,
            full_attention_interval,
            layer_types,
            hc_count: md_u32(md, "qwen4exp.hyper_connection.count")?,
            hc_lowrank: md_u32(md, "qwen4exp.hyper_connection.low_rank")?,
            n_heads: md_u32(md, "qwen4exp.attention.head_count")?,
            n_kv_heads: md_u32(md, "qwen4exp.attention.head_count_kv")?,
            head_dim,
            partial_rotary_factor: n_rot as f64 / head_dim as f64,
            rope_theta: md_f32(md, "qwen4exp.rope.freq_base")?,
            mrope_section,
            indexer_n_heads: md_u32(md, "qwen4exp.attention.indexer.head_count")?,
            indexer_head_dim: md_u32(md, "qwen4exp.attention.indexer.key_length")?,
            indexer_budget: md_u32(md, "qwen4exp.attention.indexer.top_k")?,
            indexer_compress_ratio: compress_ratio,
            ssm_state_size: md_u32(md, "qwen4exp.ssm.state_size")?,
            ssm_n_group: md_u32(md, "qwen4exp.ssm.group_count")?,
            ssm_dt_rank: md_u32(md, "qwen4exp.ssm.time_step_rank")?,
            ssm_conv_kernel: md_u32(md, "qwen4exp.ssm.conv_kernel")?,
            ssm_inner_size: md_u32(md, "qwen4exp.ssm.inner_size")?,
            n_experts: md_u32(md, "qwen4exp.expert_count")?,
            n_experts_per_tok: md_u32(md, "qwen4exp.expert_used_count")?,
            moe_intermediate_size: md_u32(md, "qwen4exp.expert_feed_forward_length")?,
            shared_expert_intermediate_size: md_u32(
                md,
                "qwen4exp.expert_shared_feed_forward_length",
            )?,
            ple_layers,
            ple_ngram_size: md_u32(md, "qwen4exp.ple.ngram_size").unwrap_or(3),
            ple_heads_per_ngram: md_u32(md, "qwen4exp.ple.heads_per_ngram").unwrap_or(8),
            ple_conv_kernel: md_u32(md, "qwen4exp.ple.conv_kernel").unwrap_or(4),
            ple_head_dim: md_u32(md, "qwen4exp.embedding_length_per_layer_input").unwrap_or(0),
            ple_image_token_id: md
                .get("qwen4exp.ple.image_token_id")
                .map(|v| v.to_u32())
                .transpose()?
                .filter(|&id| id != 0),
            bos_token_id: eos,
            eos_token_id: eos,
        })
    }
}

/// The loader's view of one open shard: header + reader + device.
struct Gguf<'a, R: Read + Seek> {
    content: Content,
    reader: &'a mut R,
    device: Device,
}

impl<R: Read + Seek> Gguf<'_, R> {
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content.tensor(self.reader, name, &self.device)
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        match self.content.tensor_infos.get(name) {
            Some(ti) => Ok(ti.shape.dims().to_vec()),
            None => lloom_tensor::bail!("gguf has no tensor `{name}`"),
        }
    }
    /// A matrix as the matmul operand: quantized types stay packed; BF16/F16/
    /// F32 become f16 tensors (dense f16 GEMV, f32 accumulate — §8.1.4).
    fn qmat(&mut self, name: &str) -> Result<QMatMul> {
        let qt = self.qtensor(name)?;
        match qt.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                Ok(QMatMul::TensorF16(qt.dequantize_f16(&self.device)?))
            }
            _ => QMatMul::from_arc(Arc::new(qt)),
        }
    }
    /// A quantized matrix kept packed whatever its type (the routed expert
    /// stacks and the embedding table, which must not be expanded).
    fn packed(&mut self, name: &str) -> Result<Arc<QTensor>> {
        Ok(Arc::new(self.qtensor(name)?))
    }
    /// An f32 vector/tensor (norm weights, conv, SSM scalars).
    fn f32(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.qtensor(name)?;
        qt.dequantize(&self.device)
    }
}

/// What besides the first shard a load may take.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// The GGUF shard holding `per_layer_token_embd.weight` (the N-gram PLE
    /// table). `None` loads the model without the PLE — the layer-1 injection
    /// is skipped, which is the #747 baseline the digests were recorded on.
    pub ple_shard: Option<std::path::PathBuf>,
    /// The routed experts' activation dtype (`None` = f32, the byte-identical
    /// routed arm; `Some(F16)` = the `(f16, f32, f16)` arm).
    pub moe_act: Option<lloom_tensor::DType>,
}

impl Qwen4Exp<Resident> {
    /// Load the model from an open GGUF shard, without the N-gram PLE (see
    /// [`Self::from_gguf_with`]).
    pub fn from_gguf<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with(content, reader, device, &LoadOptions::default())
    }

    /// Load the model from an open GGUF shard (the first shard carries the whole
    /// text model), plus the PLE table's shard when `opts.ple_shard` names it.
    pub fn from_gguf_with<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let mut cfg = Config::from_gguf_metadata(&content.metadata)?;
        let ple_hash = match (&opts.ple_shard, cfg.ple_layers.first()) {
            (Some(_), Some(_)) => {
                let md = &content.metadata;
                Some(NgramHash::new(
                    md_int_array(md, "qwen4exp.ple.layer_multipliers")?,
                    md_int_array(md, "qwen4exp.ple.head_vocab_sizes")?,
                    md_int_array(md, "qwen4exp.ple.head_offsets")?,
                    cfg.ple_heads_per_ngram,
                    cfg.eos_token_id,
                )?)
            }
            _ => None,
        };
        let mut gg = Gguf {
            content,
            reader,
            device: device.clone(),
        };

        let embed_shape = gg.shape("token_embd.weight")?;
        if embed_shape.len() != 2 || embed_shape[1] != cfg.hidden_size {
            lloom_tensor::bail!(
                "token_embd.weight has shape {embed_shape:?}, expected [vocab, {}]",
                cfg.hidden_size
            )
        }
        cfg.vocab_size = embed_shape[0];

        let d = cfg.hidden_size;
        let n = cfg.hc_count;
        let eps = cfg.norm_eps as f32;

        let embed_tokens = QMatMul::QTensor(gg.packed("token_embd.weight")?);
        let lm_head = gg.qmat("output.weight")?;

        // Per-layer brackets carry an inject predictor; the head mixer
        // (`output_hc_*`) has none — its collapse goes straight to lm_head
        // (the reference passes nullptr there).
        let hc = |gg: &mut Gguf<'_, R>, p: &str, inject: bool| -> Result<HyperConnection> {
            let inject = if inject {
                Some(gg.qmat(&format!("{p}_inject.weight"))?)
            } else {
                None
            };
            HyperConnection::new(
                gg.f32(&format!("{p}_norm.weight"))?,
                gg.qmat(&format!("{p}_down.weight"))?,
                gg.qmat(&format!("{p}_up.weight"))?,
                inject,
                n,
                d,
                eps,
            )
        };
        let hc_head = hc(&mut gg, "output_hc", false)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = format!("blk.{l}");
            let hc_attn = hc(&mut gg, &format!("{p}.hc_attn"), true)?;
            let hc_ffn = hc(&mut gg, &format!("{p}.hc_ffn"), true)?;
            let mixer = match cfg.layer_types[l] {
                LayerKind::Linear => Mixer::Linear(GatedDeltaNet::new(
                    gg.qmat(&format!("{p}.attn_qkv.weight"))?,
                    gg.qmat(&format!("{p}.attn_gate.weight"))?,
                    gg.f32(&format!("{p}.ssm_conv1d.weight"))?,
                    gg.f32(&format!("{p}.ssm_a"))?,
                    gg.f32(&format!("{p}.ssm_dt.bias"))?,
                    gg.qmat(&format!("{p}.ssm_alpha.weight"))?,
                    gg.qmat(&format!("{p}.ssm_beta.weight"))?,
                    gg.f32(&format!("{p}.ssm_norm.weight"))?,
                    gg.qmat(&format!("{p}.ssm_out.weight"))?,
                    &cfg,
                )),
                LayerKind::Full => {
                    let indexer = Indexer::new(
                        gg.qmat(&format!("{p}.indexer.q_proj.weight"))?,
                        gg.qmat(&format!("{p}.indexer.k_proj.weight"))?,
                        RmsNorm::new(gg.f32(&format!("{p}.indexer.q_norm.weight"))?, cfg.norm_eps),
                        RmsNorm::new(gg.f32(&format!("{p}.indexer.k_norm.weight"))?, cfg.norm_eps),
                        &cfg,
                    )?;
                    Mixer::Full(
                        QsaAttention::new(
                            gg.qmat(&format!("{p}.attn_q.weight"))?,
                            gg.qmat(&format!("{p}.attn_k.weight"))?,
                            gg.qmat(&format!("{p}.attn_v.weight"))?,
                            gg.qmat(&format!("{p}.attn_output.weight"))?,
                            RmsNorm::new(gg.f32(&format!("{p}.attn_q_norm.weight"))?, cfg.norm_eps),
                            RmsNorm::new(gg.f32(&format!("{p}.attn_k_norm.weight"))?, cfg.norm_eps),
                            &cfg,
                        )
                        .with_indexer(indexer),
                    )
                }
            };
            let experts = Resident::new(
                gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
            );
            let sh_router = gg
                .f32(&format!("{p}.ffn_gate_inp_shexp.weight"))?
                .reshape((1, d))?;
            let moe = MoeLayer::new(
                gg.qmat(&format!("{p}.ffn_gate_inp.weight"))?,
                experts,
                gg.qmat(&format!("{p}.ffn_gate_shexp.weight"))?,
                gg.qmat(&format!("{p}.ffn_up_shexp.weight"))?,
                gg.qmat(&format!("{p}.ffn_down_shexp.weight"))?,
                QMatMul::Tensor(sh_router),
                &cfg,
            );
            layers.push(Layer::new(hc_attn, mixer, hc_ffn, moe));
        }

        let mut model = Qwen4Exp::new(embed_tokens, layers, hc_head, lm_head, cfg.clone());
        if let Some(dt) = opts.moe_act {
            model.set_routed_act(dt);
        }
        if let (Some(hash), Some(shard), Some(&l)) =
            (ple_hash, &opts.ple_shard, cfg.ple_layers.first())
        {
            let p = format!("blk.{l}");
            let block = PleLayer::new(
                gg.qmat(&format!("{p}.ple_key.weight"))?,
                gg.qmat(&format!("{p}.ple_value.weight"))?,
                gg.f32(&format!("{p}.ple_norm_key.weight"))?,
                gg.f32(&format!("{p}.ple_norm_query.weight"))?,
                gg.f32(&format!("{p}.ple_norm_conv.weight"))?,
                // ggml `[k, hc_dim]` → candle `[hc_dim, k]`.
                gg.f32(&format!("{p}.ple_conv1d.weight"))?,
                n,
                d,
                cfg.ple_ngram_size,
                eps,
            )?;
            let table = MmapPleTable::open(shard)?;
            if table.rows() < hash.total_rows() as usize {
                lloom_tensor::bail!(
                    "ple table has {} rows, the hash needs {}",
                    table.rows(),
                    hash.total_rows()
                )
            }
            model.set_ple(l, hash, PleTable::Mmap(table), block)?;
        }
        Ok(model)
    }
}
