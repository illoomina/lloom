//! GGUF loader for Qwen3.8-Flash-Next: the `qwen4exp.*` header → `Config`, the
//! tensor table → the blocks, on the `Resident` path (every expert stack in
//! memory, packed) or the `Streamed` path (the expert stacks left in the file
//! and streamed into a slot pool, `nn::expert_stream`).
//!
//! Tensor names follow the llama.cpp converter (PR #27742); shapes are candle
//! row-major (ggml `ne` reversed), so `ffn_gate_exps` is `[n_expert, n_ff,
//! hidden]` and `ffn_down_exps` `[n_expert, hidden, n_ff]` — the routed
//! kernel's `[E, n, k]`. Quantized matrices stay packed behind `QMatMul`; the
//! BF16 matrices (HyperConnection low-rank, routers, DeltaNet α/β) become f16
//! tensors (`QMatMul::TensorF16` — f16 GEMV, f32 accumulate) rather than f32,
//! which halves their footprint; f32 vectors (norms, conv, `ssm_a`, `dt.bias`)
//! are dequantized as they are.
//!
//! A checkpoint may be split across shards (`split.count`); a tensor is read
//! from whichever shard holds it, and the PLE table from the shard that holds
//! `per_layer_token_embd.weight`.

use super::config::{Config, LayerKind};
use super::deltanet_block::GatedDeltaNet;
use super::hc::HyperConnection;
use super::model::{Layer, Mixer, Qwen4Exp};
use super::moe::{ExpertResidency, MoeLayer, Resident, Streamed};
use super::mtp::MtpHead;
use super::ple::{CachedPleTable, MmapPleTable, NgramHash, PleLayer, PleTable};
use super::qsa::{Indexer, QsaAttention};
use crate::nn::attention::route::ExecutionPurpose;
use crate::nn::expert_stream::{ExpertSource, LoaderArm};
use crate::nn::RmsNorm;
use crate::profile::Profile;
use lloom_tensor::quantized::gguf_file::{Content, TensorInfo, Value};
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) fn md_u32(md: &HashMap<String, Value>, key: &str) -> Result<usize> {
    match md.get(key) {
        Some(v) => Ok(v.to_u32()? as usize),
        None => lloom_tensor::bail!("gguf metadata has no `{key}`"),
    }
}
pub(crate) fn md_f32(md: &HashMap<String, Value>, key: &str) -> Result<f64> {
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
pub(crate) fn md_int_array(md: &HashMap<String, Value>, key: &str) -> Result<Vec<i64>> {
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
            kv_precision: super::config::DEFAULT_KV_PRECISION,
        })
    }
}

/// One open shard: header, reader, and the path it was opened from (`None`
/// for a caller-supplied reader).
pub(crate) struct Shard<'a, R: Read + Seek> {
    pub(crate) content: Content,
    pub(crate) reader: &'a mut R,
    pub(crate) path: Option<PathBuf>,
}

/// The loader's view of a checkpoint: its shards, a name → shard index, the
/// device.
pub(crate) struct Gguf<'a, R: Read + Seek> {
    pub(crate) shards: Vec<Shard<'a, R>>,
    index: HashMap<String, usize>,
    pub(crate) device: Device,
    /// Keep float matrices f32 (`QMatMul::Tensor`) rather than f16 stems:
    /// the CPU reference's precision ([`Qwen4Exp::from_gguf_reference`]).
    pub(crate) dense_f32: bool,
    /// Read no data: every tensor a ghost of its header's kind and shape
    /// ([`Qwen4Exp::ghost_of_manifest`]), a model to walk, never to run.
    pub(crate) ghost: bool,
    /// What the attention is built for (`LoadOptions::purpose`).
    pub(crate) purpose: ExecutionPurpose,
    /// Every tensor read, for the schema's debug check (`build`).
    #[cfg(debug_assertions)]
    read: std::cell::RefCell<std::collections::BTreeSet<String>>,
}

impl<'a, R: Read + Seek> Gguf<'a, R> {
    pub(crate) fn new(shards: Vec<Shard<'a, R>>, device: &Device) -> Result<Self> {
        let mut index = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for name in s.content.tensor_infos.keys() {
                if index.insert(name.clone(), i).is_some() {
                    lloom_tensor::bail!("tensor `{name}` appears in two shards")
                }
            }
        }
        Ok(Self {
            shards,
            index,
            device: device.clone(),
            dense_f32: false,
            ghost: false,
            purpose: ExecutionPurpose::Reference,
            #[cfg(debug_assertions)]
            read: Default::default(),
        })
    }
    pub(crate) fn metadata(&self) -> &HashMap<String, Value> {
        &self.shards[0].content.metadata
    }
    pub(crate) fn shard_of(&self, name: &str) -> Result<usize> {
        match self.index.get(name) {
            Some(&i) => Ok(i),
            None => lloom_tensor::bail!("gguf has no tensor `{name}`"),
        }
    }
    pub(crate) fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        let i = self.shard_of(name)?;
        #[cfg(debug_assertions)]
        self.read.borrow_mut().insert(name.to_string());
        let device = self.device.clone();
        let s = &mut self.shards[i];
        if self.ghost {
            // One block of storage under the header's kind and shape.
            let ti = &s.content.tensor_infos[name];
            return QTensor::ghost(ti.shape.clone(), ti.ggml_dtype, &device);
        }
        s.content.tensor(s.reader, name, &device)
    }

    /// A ghost dense matrix at `dtype` of the header's shape.
    fn ghost_dense(&mut self, name: &str, dtype: lloom_tensor::DType) -> Result<Tensor> {
        let shape = self.qtensor(name)?.shape().clone();
        crate::testing::ghost::tensor(shape, dtype, &self.device)
    }
    /// Every tensor of every shard, `(name, shape)`.
    pub(crate) fn table(&self) -> impl Iterator<Item = (&str, &[usize])> {
        self.shards.iter().flat_map(|s| {
            s.content
                .tensor_infos
                .iter()
                .map(|(n, ti)| (n.as_str(), ti.shape.dims()))
        })
    }
    pub(crate) fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let i = self.shard_of(name)?;
        match self.shards[i].content.tensor_infos.get(name) {
            Some(ti) => Ok(ti.shape.dims().to_vec()),
            None => lloom_tensor::bail!("gguf has no tensor `{name}`"),
        }
    }
    /// The shard holding `name`, as `(header, path)`; a shard without a path
    /// cannot be streamed from.
    pub(crate) fn locate(&self, name: &str) -> Result<(&Content, &Path)> {
        let i = self.shard_of(name)?;
        #[cfg(debug_assertions)]
        self.read.borrow_mut().insert(name.to_string());
        let s = &self.shards[i];
        match &s.path {
            Some(p) => Ok((&s.content, p)),
            None => lloom_tensor::bail!(
                "tensor `{name}` is in a shard opened from a reader, not a path; \
                 streaming needs the file"
            ),
        }
    }
    /// A matrix as the matmul operand: quantized types stay packed; BF16/F16/
    /// F32 become f16 tensors (dense f16 GEMV, f32 accumulate — §8.1.4).
    pub(crate) fn qmat(&mut self, name: &str) -> Result<QMatMul> {
        if self.ghost {
            let dtype = self.shards[self.shard_of(name)?].content.tensor_infos[name].ggml_dtype;
            return match dtype {
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 if self.dense_f32 => Ok(
                    QMatMul::Tensor(self.ghost_dense(name, lloom_tensor::DType::F32)?),
                ),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => Ok(QMatMul::TensorF16(
                    self.ghost_dense(name, lloom_tensor::DType::F16)?,
                )),
                _ => Ok(QMatMul::QTensor(Arc::new(self.qtensor(name)?))),
            };
        }
        let qt = self.qtensor(name)?;
        match qt.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 if self.dense_f32 => {
                Ok(QMatMul::Tensor(qt.dequantize(&self.device)?))
            }
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                Ok(QMatMul::TensorF16(qt.dequantize_f16(&self.device)?))
            }
            _ => QMatMul::from_arc(Arc::new(qt)),
        }
    }
    /// The named matrices as ONE f16 matrix, rows concatenated in order,
    /// when every one of them is an f16 stem (`TensorF16`) — the parts come
    /// back as row VIEWS of it, so a decode step can run the whole stack as
    /// one gemv and split the output (`GatedDeltaNet::with_stem_cat`,
    /// `QsaAttention::with_stem_cat`) while every other path keeps its
    /// part. `None` when any part is quantized (then the parts stand alone).
    pub(crate) fn qmat_stack(
        &mut self,
        names: &[String],
    ) -> Result<(Vec<QMatMul>, Option<Tensor>)> {
        let parts: Vec<QMatMul> = names.iter().map(|n| self.qmat(n)).collect::<Result<_>>()?;
        let f16: Vec<&Tensor> = parts
            .iter()
            .filter_map(|p| match p {
                QMatMul::TensorF16(t) => Some(t),
                _ => None,
            })
            .collect();
        if crate::tuning::tuning().qwen4exp.trace_load {
            let kinds: Vec<String> = parts
                .iter()
                .map(|p| match p {
                    QMatMul::TensorF16(_) => "f16".to_string(),
                    QMatMul::Tensor(_) => "f32".to_string(),
                    QMatMul::QTensor(q) => format!("{:?}", q.dtype()),
                })
                .collect();
            eprintln!("STACK {names:?} kinds={kinds:?}");
        }
        if f16.len() != parts.len() {
            return Ok((parts, None));
        }
        let cat = Tensor::cat(&f16, 0)?;
        let mut views = Vec::with_capacity(parts.len());
        let mut off = 0;
        for t in &f16 {
            let rows = t.dim(0)?;
            views.push(QMatMul::TensorF16(cat.narrow(0, off, rows)?));
            off += rows;
        }
        Ok((views, Some(cat)))
    }
    /// A quantized matrix kept packed whatever its type (the routed expert
    /// stacks and the embedding table, which must not be expanded).
    pub(crate) fn packed(&mut self, name: &str) -> Result<Arc<QTensor>> {
        Ok(Arc::new(self.qtensor(name)?))
    }
    /// An f32 vector/tensor (norm weights, conv, SSM scalars).
    pub(crate) fn f32(&mut self, name: &str) -> Result<Tensor> {
        if self.ghost {
            return self.ghost_dense(name, lloom_tensor::DType::F32);
        }
        let qt = self.qtensor(name)?;
        qt.dequantize(&self.device)
    }
}

/// Which loader primitive streams the experts (§20.4a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamLoader {
    /// `pread` + a compute-encoder copy into the slot (candidate A).
    #[default]
    Pread,
    /// `MTLIOCommandQueue` straight into the slot (candidate B, Metal only).
    Io,
}

/// How many experts a layer keeps resident (the slot pool per projection).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlotBudget {
    /// A fixed count.
    Fixed(usize),
    /// Sized from the device: the expert pool's grant from the arbiter
    /// (`profiles-and-memory-design.md` §4.2; #892 A5) over `fraction` of
    /// `recommendedMaxWorkingSetSize`, with the resident tier (every
    /// non-expert tensor, from the headers) and the session's claims
    /// ([`session_claims`]: the KV of `rows` rows to `kv_tokens`, the
    /// prefill transient, the loader's staging) as the other tenants;
    /// clamped to `[topk + 1, n_experts]`. The pool takes what the claims
    /// leave and yields under pressure (the admission's third rung parks
    /// slots) — the old session reserve's free-list and pass-retention
    /// constants, which kept "what fits" from being the whole budget
    /// (#749 P18) without being a policy, are gone.
    Auto {
        fraction: f64,
        /// Rows the session state is reserved for (1, or `--batch N`).
        rows: usize,
        /// KV positions per row the reserve covers (`--kv-capacity`).
        kv_tokens: usize,
        /// The prefill tile the transient is reserved for (0 = one tile of
        /// `PREFILL_CHUNK`).
        prefill_tile: usize,
    },
}

impl SlotBudget {
    /// `Auto` at `fraction` for one row at 4096 positions and the default
    /// tile — the harness defaults.
    pub fn auto(fraction: f64) -> Self {
        SlotBudget::Auto {
            fraction,
            rows: 1,
            kv_tokens: DEFAULT_KV_RESERVE_TOKENS,
            prefill_tile: super::model::PREFILL_CHUNK,
        }
    }
}

/// KV positions per row `SlotBudget::auto` reserves for.
pub const DEFAULT_KV_RESERVE_TOKENS: usize = 4096;

/// One position's KV over every full-attention layer at the config's store
/// (`Config::kv_precision`: f32 or f16 elements, or the codes with their
/// (scale, bias) planes counted — §10.7a's 0.531× argument, at this
/// geometry), plus the indexer's two f32 caches: the raw key per position
/// and the compressed key per `indexer_compress_ratio` positions.
pub fn kv_bytes_per_token(cfg: &Config) -> Result<usize> {
    let full = cfg
        .layer_types
        .iter()
        .filter(|k| matches!(k, LayerKind::Full))
        .count();
    // No full-attention layers means no KV predictor demand, regardless of
    // fields belonging to absent attention geometry. Config validation is separate.
    if full == 0 {
        return Ok(0);
    }
    let fail = || lloom_tensor::Error::Msg("Qwen KV bytes per token exceed usize".into());
    let kv = cfg.kv_precision.checked_bytes_per_token(
        cfg.n_kv_heads,
        cfg.head_dim,
        lloom_tensor::DType::F32,
    )?;
    let indexer = cfg
        .indexer_head_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(fail)?;
    let blocks = indexer.div_ceil(cfg.indexer_compress_ratio.max(1));
    kv.checked_add(indexer)
        .and_then(|v| v.checked_add(blocks))
        .and_then(|v| v.checked_mul(full))
        .ok_or_else(fail)
}

pub(crate) fn kv_slab_bytes(cfg: &Config) -> Result<usize> {
    kv_bytes_per_token(cfg)?
        .checked_mul(super::qsa::KV_SLAB_TOKENS)
        .ok_or_else(|| lloom_tensor::Error::Msg("Qwen KV slab unit exceeds usize".into()))
}

// Exact multiply before divide: saturating the product before /2 undercounts.
fn prefill_transient_bytes(tile: usize, tiles: usize) -> Result<(usize, usize)> {
    let fail = || lloom_tensor::Error::Msg("Qwen prefill transient exceeds usize".into());
    let tile_bytes = tile
        .checked_mul(super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN)
        .ok_or_else(fail)?;
    let further = (tile_bytes as u128) * (tiles.saturating_sub(2) as u128) / 2;
    let total = (tile_bytes as u128).checked_add(further).ok_or_else(fail)?;
    Ok((tile_bytes, usize::try_from(total).map_err(|_| fail())?))
}

/// The session's claims beside the expert pool, for the load-time
/// arbitration (`profiles-and-memory-design.md` §4.2; #892 A5): the KV of
/// `rows` rows to `kv_tokens` (a capability — the caller chose the
/// context), the prefill transient of one tile and the further tiles in
/// flight (scratch), and the loader's staging (a capability of the loader).
/// What the old session reserve also held — a free-list reserve and a
/// pass-retention reserve, two constants in place of a policy — is the
/// expert pool's to yield now: the admission's third rung parks slots when
/// a pass is short (`Qwen4Exp::apply_expert_grant`).
pub fn session_claims(
    cfg: &Config,
    rows: usize,
    kv_tokens: usize,
    prefill_tile: usize,
    loader: &StreamLoader,
    prefetch: bool,
) -> Result<Vec<SessionClaim>> {
    let tile = if prefill_tile == 0 {
        super::model::PREFILL_CHUNK
    } else {
        prefill_tile
    };
    let tiles = if super::model::prefill_pair_on() {
        super::model::prefill_tiles()
    } else {
        2
    };
    let (tile_bytes, transient) = prefill_transient_bytes(tile, tiles)?;
    let per = kv_bytes_per_token(cfg)?;
    let kv = per
        .checked_mul(kv_tokens)
        .and_then(|v| v.checked_mul(rows.max(1)))
        .ok_or_else(|| lloom_tensor::Error::Msg("Qwen session KV exceeds usize".into()))?;
    Ok(vec![
        SessionClaim {
            name: "qwen4exp-kv",
            bytes: kv,
            unit: per,
            capability: true,
        },
        SessionClaim {
            name: "qwen4exp-transient",
            bytes: transient,
            unit: tile_bytes,
            capability: false,
        },
        SessionClaim {
            name: "qwen4exp-staging",
            bytes: staging_reserve_bytes(loader, prefetch),
            unit: 1 << 20,
            capability: true,
        },
    ])
}

/// One of the session's fixed claims (`floor == want`): a capability
/// (never reduced) or scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionClaim {
    pub name: &'static str,
    pub bytes: usize,
    pub unit: usize,
    pub capability: bool,
}

/// The session's claims summed: what the plan leaves out of the pool.
pub fn session_floor_bytes(claims: &[SessionClaim]) -> Result<usize> {
    claims.iter().try_fold(0usize, |sum, c| {
        sum.checked_add(c.bytes)
            .ok_or_else(|| lloom_tensor::Error::Msg("Qwen session claim sum exceeds usize".into()))
    })
}

/// The staging ring's cap on the `pread` arm: `staging::DEFAULT_CAP_BYTES`
/// (1 GiB), or [`PREFETCH_RING_BYTES`] with a prefetcher — its reader
/// threads hold up to [`PREFETCH_STAGED_BYTES`] staged ahead of the layers
/// beside what the layers themselves stage, and a ring at its cap hands out
/// one-shots (measured at 1 GiB with the prefetcher: 32 792 one-shots over
/// a 937-token prompt, issue #749 P19).
pub fn staging_ring_bytes(loader: &StreamLoader, prefetch: bool) -> usize {
    match (loader, prefetch) {
        (StreamLoader::Pread, true) => PREFETCH_RING_BYTES,
        (StreamLoader::Pread, false) => 1 << 30,
        (StreamLoader::Io, _) => 0,
    }
}

/// The ring with a prefetcher, and what the prefetcher may hold of it.
pub const PREFETCH_RING_BYTES: usize = 2 << 30;
pub const PREFETCH_STAGED_BYTES: usize = 1 << 30;

/// The `io` arm's transient: the bytes of the loads pending on the IO queue,
/// which it holds on the device until they run — a warm keeps two layers'
/// pending, the hints a lead's worth of layers (~0.6 GB a layer at ~340
/// slots), a layer's own misses one (issue #749 P26).
pub const IO_INFLIGHT_RESERVE_BYTES: usize = 2 << 30;

/// The streamed tier's own transient memory beside its slots — the staging
/// ring's cap — left out of the slot plan like the session reserve;
/// measured before it was: a prefetched second tile refused at 35.12 GB on
/// a 36.18 GB budget (`p19-prefetch-on.txt`).
pub fn staging_reserve_bytes(loader: &StreamLoader, prefetch: bool) -> usize {
    match loader {
        StreamLoader::Pread => staging_ring_bytes(loader, prefetch),
        StreamLoader::Io => IO_INFLIGHT_RESERVE_BYTES,
    }
}

/// How the routed experts are streamed.
#[derive(Debug, Clone)]
pub struct StreamOptions {
    /// Resident experts per layer per projection (the slot pool).
    pub slots: SlotBudget,
    pub loader: StreamLoader,
    /// The prefetch thread on the `pread` arm (Metal): a multi-token pass
    /// stages each layer's likely misses while the layer before it runs
    /// (`nn::expert_prefetch`). Off, every miss is read at the layer.
    pub prefetch: bool,
    /// The prefetch hints on the `io` arm (Metal): the hinted experts take
    /// the pool's FREE slots at hint time and load on the IO queue, waited
    /// for at the layer (the in-flight residency entry, §20.4a). Off by
    /// default — measured net-negative on this pool (issue #749 P26): the
    /// arm's DMA is already asynchronous, and a pool smaller than the
    /// tile's working set has no room to read ahead into.
    pub io_hints: bool,
    /// Bytes a drafter loaded beside the model holds (its weights and the
    /// tables it keeps), claimed out of the slot plan under `Auto`; 0 for
    /// none (the MTP head is a shard of the model and counted with it).
    pub drafter_bytes: usize,
}

/// The slots `Auto` resolves to, and the terms it resolved them from.
#[derive(Debug, Clone, Copy)]
pub struct SlotPlan {
    pub slots: usize,
    pub working_set: usize,
    pub resident_bytes: usize,
    pub expert_bytes_per_slot: usize,
    /// The session's claims the plan left out of the pool (the KV, the
    /// transient, the staging: `session_claims`).
    pub reserve_bytes: usize,
}

/// Every tensor's packed size from the headers: the resident tier, one
/// slot's worth of expert planes (summed over the layers) and every expert
/// plane, as a [`LoadPlan`] for `slots` (`None` = every expert resident).
fn plan_load<Rd: Read + Seek>(
    gg: &Gguf<'_, Rd>,
    device: &Device,
    fraction: f64,
    slots: Option<usize>,
) -> Result<LoadPlan> {
    plan_load_holding(gg, device, fraction, slots, None)
}

// A transaction over header-derived device-weight terms, not an allocation or
// a bound on opaque native objects. PLE is excluded because this term omits the
// mmap tier; its mapping/page residency is not physically free.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct HeaderTotals {
    resident: usize,
    per_slot: usize,
    total: usize,
}

impl HeaderTotals {
    fn add(&mut self, name: &str, info: &TensorInfo, held_prefix: Option<&str>) -> Result<()> {
        if name == "per_layer_token_embd.weight" {
            return Ok(());
        }
        let overflow =
            |term| lloom_tensor::Error::Msg(format!("tensor `{name}`: {term} overflows usize"));
        // A zero dimension makes the exact product zero, even if a preceding
        // partial product would overflow. An empty shape is one scalar.
        let dims = info.shape.dims();
        let elements = if dims.contains(&0) {
            0
        } else {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| overflow("dimension product"))?
        };
        let block = info.ggml_dtype.block_size();
        if !elements.is_multiple_of(block) {
            lloom_tensor::bail!(
                "tensor `{name}`: {elements} elements are not divisible by block size {block}"
            )
        }
        let bytes = (elements / block)
            .checked_mul(info.ggml_dtype.type_size())
            .ok_or_else(|| overflow("packed bytes"))?;
        let mut next = *self;
        let held = held_prefix.is_some_and(|prefix| name.starts_with(prefix));
        if name.ends_with("_exps.weight") && !held {
            // Preserve the existing plane division. ExpertSource::from_gguf
            // additionally validates dims3 and k/block for actual streamed IO;
            // this header aggregate does not replace that reader's validation.
            let experts = dims.first().copied().unwrap_or(1).max(1);
            next.per_slot = next
                .per_slot
                .checked_add(bytes / experts)
                .ok_or_else(|| overflow("per-slot sum"))?;
            next.total = next
                .total
                .checked_add(bytes)
                .ok_or_else(|| overflow("expert total"))?;
        } else {
            next.resident = next
                .resident
                .checked_add(bytes)
                .ok_or_else(|| overflow("resident sum"))?;
        }
        *self = next;
        Ok(())
    }
}

/// [`plan_load`] with layer `held`'s routed experts counted as resident
/// bytes rather than as planes of every slot: the MTP head layer when its
/// experts are held whole (`qwen4exp.mtp_experts = resident`), so the pool's
/// per-slot unit, the slot plan and the admission all see the same split.
fn plan_load_holding<Rd: Read + Seek>(
    gg: &Gguf<'_, Rd>,
    device: &Device,
    fraction: f64,
    slots: Option<usize>,
    held: Option<usize>,
) -> Result<LoadPlan> {
    let held_prefix = held.map(|l| format!("blk.{l}."));
    let mut totals = HeaderTotals::default();
    for sh in &gg.shards {
        for (name, info) in &sh.content.tensor_infos {
            totals.add(name, info, held_prefix.as_deref())?;
        }
    }
    Ok(LoadPlan {
        resident_bytes: totals.resident,
        expert_bytes_per_slot: totals.per_slot,
        expert_bytes_total: totals.total,
        slots,
        working_set: working_set_of(device),
        fraction,
    })
}

/// `recommendedMaxWorkingSetSize` of a Metal device; 0 elsewhere.
pub(crate) fn working_set_of(device: &Device) -> usize {
    #[cfg(feature = "metal")]
    if let Device::Metal(d) = device {
        return d.device().recommended_max_working_set_size();
    }
    let _ = device;
    0
}

fn checked_expert_slot_floor(cfg: &Config) -> Result<usize> {
    let floor = cfg
        .n_experts_per_tok
        .checked_add(1)
        .ok_or_else(|| lloom_tensor::Error::Msg("expert slot floor overflows usize".into()))?;
    if floor > cfg.n_experts {
        lloom_tensor::bail!(
            "expert slot floor {floor} exceeds expert count {}",
            cfg.n_experts
        )
    }
    Ok(floor)
}

/// The slot pool `SlotBudget::Auto` resolves to: the expert pool's grant
/// from the arbiter (`ledger::arbitrate`, §4.2) over the budget, with the
/// resident tier and the session's claims as the other tenants — the
/// resident tier and the capabilities at their floors, the pool a cache
/// claiming `topk + 1` slots as its floor, every expert as its want and a
/// slot as its unit — in slots; `[topk + 1, n_experts]`. A budget the
/// floors alone exceed refuses the load with the arithmetic.
fn plan_slots<Rd: Read + Seek>(
    gg: &Gguf<'_, Rd>,
    cfg: &Config,
    device: &Device,
    fraction: f64,
    claims: &[SessionClaim],
    held: Option<usize>,
) -> Result<SlotPlan> {
    let plan = plan_load_holding(gg, device, fraction, None, held)?;
    let per_slot = plan.expert_bytes_per_slot.max(1);
    let reserve_bytes = session_floor_bytes(claims)?;
    let min_slots = checked_expert_slot_floor(cfg)?;
    #[cfg(not(feature = "metal"))]
    let slots = {
        let _ = (per_slot, reserve_bytes);
        0usize
    };
    #[cfg(feature = "metal")]
    let slots = arbitrate_slots(&plan, cfg, fraction, claims, reserve_bytes)?;
    let slots = slots.clamp(min_slots, cfg.n_experts);
    Ok(SlotPlan {
        slots,
        working_set: plan.working_set,
        resident_bytes: plan.resident_bytes,
        expert_bytes_per_slot: plan.expert_bytes_per_slot,
        reserve_bytes,
    })
}

// Pure production loader policy, also compiled by CPU unit tests. No model
// load or native object is needed to exercise the real Refused consumer.
#[cfg(any(feature = "metal", test))]
fn arbitrate_slots(
    plan: &LoadPlan,
    cfg: &Config,
    fraction: f64,
    claims: &[SessionClaim],
    reserve_bytes: usize,
) -> Result<usize> {
    use lloom_tensor::memory_arbitration::{
        arbitrate, Arbitration, Claim, Class, TenantId, TenantRow,
    };
    let per_slot = plan.expert_bytes_per_slot.max(1);
    let min_slots = checked_expert_slot_floor(cfg)?;
    let expert_floor = min_slots
        .checked_mul(per_slot)
        .ok_or_else(|| lloom_tensor::Error::Msg("expert slot floor bytes overflow usize".into()))?;
    let expert_want = cfg
        .n_experts
        .checked_mul(per_slot)
        .ok_or_else(|| lloom_tensor::Error::Msg("expert slot want bytes overflow usize".into()))?;
    let fixed = |id: u32, name: &str, bytes: usize, unit: usize, class: Class| TenantRow {
        id: TenantId(id),
        name: name.to_string(),
        claim: Claim {
            floor: bytes,
            want: bytes,
            unit: unit.max(1),
            class,
        },
        held: 0,
    };
    let mut rows = vec![fixed(
        1,
        "qwen4exp-resident",
        plan.resident_bytes,
        1 << 20,
        Class::Capability,
    )];
    for (i, c) in claims.iter().enumerate() {
        let class = if c.capability {
            Class::Capability
        } else {
            Class::Scratch
        };
        rows.push(fixed(2 + i as u32, c.name, c.bytes, c.unit, class));
    }
    let experts = TenantId(100);
    rows.push(TenantRow {
        id: experts,
        name: "qwen4exp-experts".to_string(),
        claim: Claim {
            floor: expert_floor,
            want: expert_want,
            unit: per_slot,
            class: Class::Cache,
        },
        held: 0,
    });
    Ok(if plan.working_set == 0 {
        0
    } else {
        match arbitrate(plan.budget(), &rows) {
            Arbitration::Granted(grants) => grants
                .iter()
                .find(|g| g.id == experts)
                .map_or(0, |g| g.bytes / per_slot),
            Arbitration::Refused { floors, quota } => {
                let gb = |b: usize| b as f64 / 1e9;
                lloom_tensor::bail!(
                    "the streamed load does not fit: the floors alone — the resident tier \
                 {:.2} GB, the session's claims {:.2} GB, the pool's {} slots {:.2} GB — \
                 are {:.2} GB against a budget of {:.2} GB ({fraction} x \
                 recommendedMaxWorkingSetSize {:.2} GB): a smaller context, a smaller \
                 tile, or a larger fraction",
                    gb(plan.resident_bytes),
                    gb(reserve_bytes),
                    min_slots,
                    gb(expert_floor),
                    floors as f64 / 1e9,
                    gb(quota),
                    gb(plan.working_set),
                )
            }
        }
    })
}

/// What besides the first shard a load may take.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// The GGUF shard holding `per_layer_token_embd.weight` (the N-gram PLE
    /// table). `None` loads the model without the PLE — the layer-1 injection
    /// is skipped, which is the #747 baseline the digests were recorded on.
    /// For a sharded load the table is found in the shards themselves when
    /// this is `None` and `ple` is on.
    pub ple_shard: Option<std::path::PathBuf>,
    /// Attach the PLE from the shards that were opened (multi-shard loads).
    pub ple: bool,
    /// The routed experts' activation dtype (`None` = f32, the byte-identical
    /// routed arm; `Some(F16)` = the `(f16, f32, f16)` arm).
    pub moe_act: Option<lloom_tensor::DType>,
    /// Rows of the PLE hot-row cache (`PleTable::Cached`); `None` reads the
    /// memory map on every gather.
    pub ple_cache_rows: Option<usize>,
    /// Load the MTP head from the shard that carries `blk.{n_layers}.nextn.*`
    /// (ds4's naming; the in-tree sidecar `qwen4exp-mtp-quantize` writes).
    /// Its routed experts take the same residency strategy as the model's.
    pub mtp: bool,
    /// The fraction of `recommendedMaxWorkingSetSize` the load may hold —
    /// `DESIGN.md` §9.5's admission at the one site this family allocates
    /// its weight class (`admit_load`): the resident tier plus every expert
    /// (`Resident`) or the slot pool (`Streamed`), summed from the headers,
    /// refused before a byte is loaded when it exceeds the budget. `None` =
    /// [`DEFAULT_MEMORY_FRACTION`]; `Some(0.0)` turns the check off. Nothing
    /// on the device but Metal reads a working set; elsewhere the check is
    /// inert. Read at load only, never per token.
    pub memory_fraction: Option<f64>,
    /// The full-attention KV store (`Config::kv_precision`, `DESIGN.md`
    /// §10.7); `None` is the checkpoint's f32. Set at load, so the slot plan
    /// reserves for the store the caches will hold.
    pub kv_precision: Option<crate::nn::ops::KvPrecision>,
    /// The machine gate the load passes before its first tensor
    /// (`memory.md` §9.10); `None` loads unadmitted (a test, a harness).
    pub machine: Option<crate::nn::memory::MachineLoad>,
    /// What the attention is built for ([`ExecutionPurpose`]): production
    /// refuses at load, on Metal, a full-attention layer whose attention
    /// takes the composed path; the reference (the default) keeps it, for a
    /// test or a harness to compare against.
    pub purpose: ExecutionPurpose,
}

impl LoadOptions {
    /// The admission fraction this load runs at.
    pub fn fraction(&self) -> f64 {
        self.memory_fraction.unwrap_or(DEFAULT_MEMORY_FRACTION)
    }
}

/// §9.5k's fraction, `MemoryBudget::DEFAULT_FRACTION` (0.65) — asserted
/// equal to it by `memory_fraction_matches_the_budgets`.
pub const DEFAULT_MEMORY_FRACTION: f64 = 0.65;

/// What a load would hold on the device, from the headers: the resident
/// tier (every non-expert tensor; the PLE table is memory-mapped and not
/// counted) and the routed experts — all of them under `Resident`, `slots`
/// of them per layer under `Streamed`. The weight term of §9.5d for this
/// family, and the whole of what the load itself allocates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadPlan {
    /// Every non-expert tensor's packed bytes.
    pub resident_bytes: usize,
    /// One expert's planes over every layer (the three projections summed).
    pub expert_bytes_per_slot: usize,
    /// Every expert tensor's packed bytes (`Resident` holds all of it).
    pub expert_bytes_total: usize,
    /// The slot pool (`Streamed`), `None` for `Resident`.
    pub slots: Option<usize>,
    /// `recommendedMaxWorkingSetSize`; 0 off Metal.
    pub working_set: usize,
    pub fraction: f64,
}

impl LoadPlan {
    /// Exact predicted bytes, including totals not representable by a native
    /// allocation size. Public return widened from usize to u128 for #952.
    /// On supported 32/64-bit targets max_slots * max_slot_bytes + max_resident
    /// fits u128. Upstream header/predictor inputs still require checked math.
    pub fn predicted(&self) -> u128 {
        let experts = match self.slots {
            Some(n) => n as u128 * self.expert_bytes_per_slot as u128,
            None => self.expert_bytes_total as u128,
        };
        self.resident_bytes as u128 + experts
    }

    fn physical_bytes(&self) -> Result<usize> {
        usize::try_from(self.predicted()).map_err(|_| lloom_tensor::Error::Msg(format!(
            "unrepresentable physical memory demand: {} bytes exceeds native usize maximum {}; reduce dimensions or slots; changing the quota or forcing admission cannot represent this size",
            self.predicted(), usize::MAX,
        )))
    }
    /// `fraction × working_set`.
    pub fn budget(&self) -> usize {
        (self.working_set as f64 * self.fraction) as usize
    }
    /// Whether the load is admitted: exactly on the budget is; no working
    /// set (not Metal) or nonpositive fraction bypasses ONLY the quota.
    /// An unrepresentable physical demand is refused before either bypass.
    pub fn fits(&self) -> bool {
        let Ok(predicted) = usize::try_from(self.predicted()) else {
            return false;
        };
        self.working_set == 0 || self.fraction <= 0.0 || predicted <= self.budget()
    }
    /// The slots a streamed pool could hold under this budget (at least
    /// `topk + 1` is the pool's own floor, checked by the loader).
    pub fn slots_that_fit(&self) -> usize {
        self.budget()
            .saturating_sub(self.resident_bytes)
            .checked_div(self.expert_bytes_per_slot)
            .unwrap_or(0)
    }
    /// The fraction this load needs.
    pub fn fraction_needed(&self) -> f64 {
        if self.working_set == 0 {
            return 0.0;
        }
        self.predicted() as f64 / self.working_set as f64
    }
    /// §9.5g's report: the arithmetic and, on a refusal, the reductions
    /// that would fit in the order of what they cost.
    pub fn describe(&self) -> String {
        let predicted = match self.physical_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return error.to_string(),
        };
        let gb = |b: usize| b as f64 / 1e9;
        let experts = match self.slots {
            Some(n) => format!(
                "{n} slots x {:.1} MB",
                self.expert_bytes_per_slot as f64 / 1e6
            ),
            None => format!(
                "every expert, {:.2} GB ({:.1} MB a slot)",
                gb(self.expert_bytes_total),
                self.expert_bytes_per_slot as f64 / 1e6
            ),
        };
        let head = format!(
            "predicted {:.2} GB = resident {:.2} GB + experts {experts}; budget {:.2} GB = {:.2} x recommendedMaxWorkingSetSize {:.2} GB",
            gb(predicted),
            gb(self.resident_bytes),
            gb(self.budget()),
            self.fraction,
            gb(self.working_set),
        );
        if self.fits() {
            return format!(
                "memory budget OK at load: {head}; residual {:.2} GB for the KV, the prefill passes and the pool",
                gb(self.budget().saturating_sub(predicted))
            );
        }
        let mut s = format!(
            "memory budget exceeded at load: {head}; over by {:.2} GB. Nothing was loaded (DESIGN.md 9.5g: refuse rather than silently reduce). What would fit:",
            gb(predicted.saturating_sub(self.budget()))
        );
        if self.slots.is_none() {
            s.push_str(&format!(
                "\n  --experts stream (--slots auto: {} slots at this fraction)",
                self.slots_that_fit()
            ));
        } else {
            s.push_str(&format!("\n  --slots {}", self.slots_that_fit()));
        }
        s.push_str(&format!(
            "\n  --memory-fraction {:.2} (this load's share of the working set; the machine then has {:.2} GB for everything else, the OS included)",
            (self.fraction_needed() * 100.0).ceil() / 100.0,
            gb(self.working_set.saturating_sub(predicted))
        ));
        s
    }
}

/// The reservation a load made on the machine, held by the model it built.
type Reservation = Option<Arc<lloom_tensor::machine::Holding>>;

/// Admit the load on the machine (`memory.md` §9.10) when `opts` names a
/// gate: its predicted peak is what `plan` holds on the device plus the
/// session's claims beside it — the same claims the slot plan leaves room
/// for (`session_claims`; the staging only on the streamed tier) and a
/// drafter's bytes.
fn admit_machine<Rd: Read + Seek>(
    gg: &Gguf<'_, Rd>,
    opts: &LoadOptions,
    plan: &LoadPlan,
    stream: Option<&StreamOptions>,
) -> Result<Reservation> {
    let physical = plan.physical_bytes()?;
    let resident = u64::try_from(plan.resident_bytes).map_err(lloom_tensor::Error::wrap)?;
    let experts =
        u64::try_from(physical - plan.resident_bytes).map_err(lloom_tensor::Error::wrap)?;
    let Some(m) = &opts.machine else {
        return Ok(None);
    };
    let mut cfg = Config::from_gguf_metadata(gg.metadata())?;
    if let Some(p) = opts.kv_precision {
        cfg.kv_precision = p;
    }
    let claims = match stream {
        Some(st) => session_claims(
            &cfg,
            m.rows,
            m.kv_tokens,
            m.prefill_tile,
            &st.loader,
            st.prefetch,
        )?,
        None => session_claims(
            &cfg,
            m.rows,
            m.kv_tokens,
            m.prefill_tile,
            &StreamLoader::Pread,
            false,
        )?
        .into_iter()
        .filter(|c| c.name != "qwen4exp-staging")
        .collect(),
    };
    let mut peak = lloom_tensor::machine::Peak::new()
        .with("weights (resident tier)", resident)
        .with(
            match plan.slots {
                Some(n) => format!("weights (experts, {n} slots)"),
                None => "weights (experts, every one resident)".to_string(),
            },
            experts,
        );
    for c in &claims {
        peak.push(c.name, c.bytes as u64);
    }
    if let Some(st) = stream {
        peak.push("drafter", st.drafter_bytes as u64);
    }
    Ok(m.admit(&gg.device, peak)?.holding)
}

/// Refuse a load the budget cannot take, with the arithmetic (§9.5g).
pub fn admit_load(plan: &LoadPlan) -> Result<()> {
    if plan.fits() {
        Ok(())
    } else {
        lloom_tensor::bail!("{}", plan.describe())
    }
}

/// Open every shard at `paths`, headers only.
pub(crate) fn open_shards(paths: &[PathBuf]) -> Result<(Vec<Content>, Vec<File>)> {
    let mut contents = Vec::with_capacity(paths.len());
    let mut files = Vec::with_capacity(paths.len());
    for p in paths {
        let mut f = File::open(p)?;
        contents.push(Content::read(&mut f)?);
        files.push(f);
    }
    Ok((contents, files))
}

/// Enter the existing domain before headers, eager maps or worker services.
/// Preserve every family option except the already-satisfied machine gate.
fn in_admitted_load<T>(
    device: &Device,
    load: crate::nn::memory::LoadContext,
    opts: &LoadOptions,
    build: impl FnOnce(&Device, &LoadOptions) -> Result<T>,
) -> Result<crate::nn::memory::Loaded<T>> {
    crate::nn::memory::Loaded::new(device, load, |resolved, _context| {
        let mut scoped = opts.clone();
        scoped.machine = None;
        build(resolved, &scoped)
    })
}

impl<P: Profile> Qwen4Exp<P, Resident> {
    /// Resident loading in one caller-supplied admitted model domain.
    /// Headers and the incumbent complete assembly run inside the scope; the
    /// caller's load peak includes their host metadata and transfer temporaries.
    /// No second machine Holding is created. Legacy constructors are unchanged.
    ///
    /// Source integration only, pending complete allocator/native/worker producer
    /// enrollment. Requests must enter the returned owner's pass scope, including
    /// fresh cache, PLE and lazy service creation; context alone is not a namespace.
    pub fn from_gguf_shards_admitted(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
        load: crate::nn::memory::LoadContext,
    ) -> Result<crate::nn::memory::Loaded<Self>> {
        in_admitted_load(device, load, opts, |resolved, scoped| {
            Self::from_gguf_shards(paths, resolved, scoped)
        })
    }

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
        let mut gg = Gguf::new(
            vec![Shard {
                content,
                reader,
                path: None,
            }],
            device,
        )?;
        let plan = plan_load(&gg, device, opts.fraction(), None)?;
        admit_load(&plan)?;
        let reservation = admit_machine(&gg, opts, &plan, None)?;
        let mut model = build(&mut gg, device, opts, |gg, l| {
            let p = format!("blk.{l}");
            Ok(Resident::new(
                gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
            ))
        })?;
        model.hold_reservation(reservation);
        Ok(model)
    }

    /// Load from an open GGUF shard with every float matrix kept f32
    /// (`QMatMul::Tensor`) instead of an f16 stem: f32 activations over the
    /// file's values, the precision a CPU reference runs at. Not a serving
    /// arm — the stems' f16 GEMV and its fused stacks are not taken.
    pub fn from_gguf_reference<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(
            vec![Shard {
                content,
                reader,
                path: None,
            }],
            device,
        )?;
        gg.dense_f32 = true;
        let opts = LoadOptions::default();
        admit_load(&plan_load(&gg, device, opts.fraction(), None)?)?;
        build(&mut gg, device, &opts, |gg, l| {
            let p = format!("blk.{l}");
            Ok(Resident::new(
                gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
            ))
        })
    }

    /// Load from a sharded checkpoint, every expert resident.
    pub fn from_gguf_shards(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let (contents, mut files) = open_shards(paths)?;
        let shards = contents
            .into_iter()
            .zip(files.iter_mut())
            .zip(paths)
            .map(|((content, reader), path)| Shard {
                content,
                reader,
                path: Some(path.clone()),
            })
            .collect();
        let mut gg = Gguf::new(shards, device)?;
        let plan = plan_load(&gg, device, opts.fraction(), None)?;
        admit_load(&plan)?;
        let reservation = admit_machine(&gg, opts, &plan, None)?;
        let mut model = build(&mut gg, device, opts, |gg, l| {
            let p = format!("blk.{l}");
            Ok(Resident::new(
                gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
            ))
        })?;
        model.hold_reservation(reservation);
        Ok(model)
    }
}

impl<P: Profile, R: ExpertResidency> Qwen4Exp<P, R> {
    /// The model a checkpoint's headers describe, built by the loader's own
    /// assembly with every tensor a ghost (`testing::ghost`): the stacks,
    /// the brackets, the PLE, the kinds each stem is stored as — the
    /// structure a walk reads — and none of the values. `experts(l)` is
    /// layer `l`'s residency. Walked, never run.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ghost_of_manifest(
        manifest: &crate::testing::scale::ShapeManifest,
        device: &Device,
        opts: &LoadOptions,
        mut experts: impl FnMut(usize) -> R,
    ) -> Result<Self> {
        use lloom_tensor::quantized::gguf_file::{TensorInfo, VersionedMagic};
        let contents: Vec<Content> = manifest
            .shards
            .iter()
            .map(|s| Content {
                magic: VersionedMagic::GgufV3,
                metadata: s.metadata.iter().cloned().collect(),
                tensor_infos: s
                    .tensors
                    .iter()
                    .map(|t| {
                        (
                            t.name.clone(),
                            TensorInfo {
                                ggml_dtype: t.dtype,
                                shape: lloom_tensor::Shape::from(t.dims.clone()),
                                offset: t.offset,
                            },
                        )
                    })
                    .collect(),
                tensor_data_offset: 0,
            })
            .collect();
        let mut readers: Vec<std::io::Cursor<Vec<u8>>> = contents
            .iter()
            .map(|_| std::io::Cursor::new(Vec::new()))
            .collect();
        let shards = contents
            .into_iter()
            .zip(readers.iter_mut())
            .map(|(content, reader)| Shard {
                content,
                reader,
                path: None,
            })
            .collect();
        let mut gg = Gguf::new(shards, device)?;
        gg.ghost = true;
        build(&mut gg, device, opts, |gg, l| {
            // The stacks are the residency's to hold; read as the loader's
            // paths read them, so the schema's check sees every tensor.
            for part in ["gate", "up", "down"] {
                gg.packed(&format!("blk.{l}.ffn_{part}_exps.weight"))?;
            }
            Ok(experts(l))
        })
    }
}

impl<P: Profile> Qwen4Exp<P, Streamed> {
    /// Streamed loading in the same caller-supplied domain as its resident
    /// weights, slot pools, staging ring, prefetcher and optional MTP head.
    /// Reuses the complete existing loader and its slot/working-set checks;
    /// skips only a duplicate legacy machine reservation.
    ///
    /// Source integration only. Before activation, IO/native custody and every
    /// worker request must retain that namespace and a live pass context. The
    /// lexical load scope does not automatically propagate into service threads.
    /// Caller load/pass predictions include all asynchronous retention and host
    /// metadata, not only physical expert payloads.
    pub fn from_gguf_streamed_admitted(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
        stream: &StreamOptions,
        load: crate::nn::memory::LoadContext,
    ) -> Result<crate::nn::memory::Loaded<Self>> {
        in_admitted_load(device, load, opts, |resolved, scoped| {
            Self::from_gguf_streamed(paths, resolved, scoped, stream)
        })
    }

    /// Load from a sharded checkpoint with the routed experts streamed from the
    /// shards into a per-layer slot pool (`stream.slots` experts per
    /// projection); everything else resident as on the `Resident` path.
    pub fn from_gguf_streamed(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
        stream: &StreamOptions,
    ) -> Result<Self> {
        let (contents, mut files) = open_shards(paths)?;
        let shards = contents
            .into_iter()
            .zip(files.iter_mut())
            .zip(paths)
            .map(|((content, reader), path)| Shard {
                content,
                reader,
                path: Some(path.clone()),
            })
            .collect();
        let mut gg = Gguf::new(shards, device)?;
        let mut cfg0 = Config::from_gguf_metadata(gg.metadata())?;
        if let Some(p) = opts.kv_precision {
            cfg0.kv_precision = p;
        }
        // The MTP head layer (`blk.{n_layers}`), held whole when asked: its
        // tier takes every expert as a slot and is warmed at load, and its
        // bytes are resident in the plan, not planes of every pool slot.
        let held = (opts.mtp && crate::tuning::tuning().qwen4exp.mtp_experts == "resident")
            .then_some(cfg0.n_layers);
        let slots = match stream.slots {
            SlotBudget::Fixed(n) => n,
            SlotBudget::Auto {
                fraction,
                rows,
                kv_tokens,
                prefill_tile,
            } => {
                let mut claims = session_claims(
                    &cfg0,
                    rows,
                    kv_tokens,
                    prefill_tile,
                    &stream.loader,
                    stream.prefetch,
                )?;
                if stream.drafter_bytes > 0 {
                    claims.push(SessionClaim {
                        name: "qwen4exp-drafter",
                        bytes: stream.drafter_bytes,
                        unit: stream.drafter_bytes,
                        capability: true,
                    });
                }
                let plan = plan_slots(&gg, &cfg0, device, fraction, &claims, held)?;
                if plan.working_set == 0 {
                    lloom_tensor::bail!(
                        "SlotBudget::Auto needs a Metal device (recommendedMaxWorkingSetSize)"
                    )
                }
                eprintln!(
                    "SLOTPLAN working_set={} fraction={fraction} resident_bytes={} expert_bytes_per_slot={} reserve_bytes={} rows={rows} kv_tokens={kv_tokens} prefill_tile={prefill_tile} slots={}",
                    plan.working_set, plan.resident_bytes, plan.expert_bytes_per_slot, plan.reserve_bytes, plan.slots
                );
                plan.slots
            }
        };
        // A fixed count is admitted like any load; `Auto` sized itself to
        // the budget and passes by construction (or sits at the pool's
        // `topk + 1` floor, which is then refused with the arithmetic).
        let plan = plan_load_holding(&gg, device, opts.fraction(), Some(slots), held)?;
        admit_load(&plan)?;
        let reservation = admit_machine(&gg, opts, &plan, Some(stream))?;
        let arm = match stream.loader {
            StreamLoader::Pread => LoaderArm::Pread,
            StreamLoader::Io => {
                #[cfg(feature = "metal")]
                {
                    match device {
                        Device::Metal(d) => LoaderArm::Io(Arc::new(
                            lloom_tensor::kernels::metal::io_queue::IoQueue::new(d.device())
                                .map_err(lloom_tensor::Error::wrap)?,
                        )),
                        _ => lloom_tensor::bail!("the io loader arm needs a Metal device"),
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    lloom_tensor::bail!("the io loader arm needs the metal feature")
                }
            }
        };
        // One staging ring for the whole tier: one byte bound, one clock.
        #[cfg(feature = "metal")]
        let staging = match (device, &arm) {
            (Device::Metal(d), LoaderArm::Pread) => Some(
                lloom_tensor::kernels::metal::staging::StagingRing::try_shared_with_cap(
                    d,
                    staging_ring_bytes(&stream.loader, stream.prefetch),
                )?,
            ),
            _ => None,
        };
        #[cfg(not(feature = "metal"))]
        let staging = None::<()>;
        // One prefetcher for the tier, over the same ring.
        #[cfg(feature = "metal")]
        let prefetcher = match (device, &staging, stream.prefetch) {
            (Device::Metal(d), Some(ring), true) => Some(Arc::new(
                crate::nn::expert_prefetch::Prefetcher::try_with_cap(
                    d,
                    Arc::clone(ring),
                    PREFETCH_STAGED_BYTES,
                )?,
            )),
            _ => None,
        };
        let mut model = build(&mut gg, device, opts, |gg, l| {
            let p = format!("blk.{l}");
            let src = |gg: &Gguf<'_, File>, name: String| -> Result<ExpertSource> {
                let (content, path) = gg.locate(&name)?;
                ExpertSource::from_gguf(content, path, &name)
            };
            #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
            let mut tier = Streamed::with_staging(
                src(gg, format!("{p}.ffn_gate_exps.weight"))?,
                src(gg, format!("{p}.ffn_up_exps.weight"))?,
                src(gg, format!("{p}.ffn_down_exps.weight"))?,
                if held == Some(l) {
                    cfg0.n_experts
                } else {
                    slots
                },
                arm.clone(),
                device,
                staging.clone(),
            )?;
            #[cfg(feature = "metal")]
            if let Some(p) = &prefetcher {
                tier.set_prefetcher(Arc::clone(p));
            }
            // The `io` arm's hints (`io_hints`): slots chosen now, the
            // loads on the IO queue (the in-flight entry).
            #[cfg(feature = "metal")]
            if let LoaderArm::Io(_) = &arm {
                tier.set_io_hints(stream.io_hints);
            }
            Ok(tier)
        })?;
        model.register_expert_ledger(
            device,
            plan.expert_bytes_per_slot,
            slots,
            cfg0.n_experts,
            cfg0.n_experts_per_tok,
        )?;
        if held.is_some() {
            if let Some(head) = model.mtp_head() {
                // In groups of at most half the bytes the loads pass through
                // (the staging ring, or the `io` arm's in-flight reserve):
                // one group of every expert (2.67 GB for the Q8 head) is past
                // a 2 GiB ring, and the ring's overflow is an unpooled
                // allocation per load, off the ledger (`staging.rs`).
                let through = match staging_ring_bytes(&stream.loader, stream.prefetch) {
                    0 => IO_INFLIGHT_RESERVE_BYTES,
                    b => b,
                };
                let per_expert = (plan_load_holding(&gg, device, opts.fraction(), None, None)?
                    .expert_bytes_total
                    - plan.expert_bytes_total)
                    / cfg0.n_experts.max(1);
                let group = (through / 2 / per_expert.max(1)).clamp(1, cfg0.n_experts.max(1));
                let all: Vec<u32> = (0..cfg0.n_experts as u32).collect();
                let mut loaded = 0usize;
                for group in all.chunks(group) {
                    loaded += head.routed().warm(group)?;
                }
                if loaded != cfg0.n_experts {
                    lloom_tensor::bail!(
                        "the MTP head's experts: {loaded} of {} loaded resident",
                        cfg0.n_experts
                    )
                }
            }
        }
        model.hold_reservation(reservation);
        Ok(model)
    }
}

/// The model from the opened shards, `experts(gg, l)` supplying layer `l`'s
/// residency strategy.
fn build<P: Profile, R: ExpertResidency, Rd: Read + Seek>(
    gg: &mut Gguf<'_, Rd>,
    _device: &Device,
    opts: &LoadOptions,
    mut experts: impl FnMut(&mut Gguf<'_, Rd>, usize) -> Result<R>,
) -> Result<Qwen4Exp<P, R>> {
    gg.purpose = opts.purpose;
    let mut cfg = Config::from_gguf_metadata(gg.metadata())?;
    let ple_hash = match (opts.ple_shard.is_some() || opts.ple, cfg.ple_layers.first()) {
        (true, Some(_)) => {
            let md = gg.metadata();
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

    let embed_shape = gg.shape("token_embd.weight")?;
    if embed_shape.len() != 2 || embed_shape[1] != cfg.hidden_size {
        lloom_tensor::bail!(
            "token_embd.weight has shape {embed_shape:?}, expected [vocab, {}]",
            cfg.hidden_size
        )
    }
    cfg.vocab_size = embed_shape[0];
    let schema = super::schema::tensor_schema(
        &cfg,
        &super::schema::SchemaOptions {
            mtp: opts.mtp,
            ple: ple_hash.is_some(),
        },
        &Default::default(),
    );
    schema.validate(gg.table())?;

    let d = cfg.hidden_size;
    let n = cfg.hc_count;
    let eps = cfg.norm_eps as f32;

    let embed_tokens = QMatMul::QTensor(gg.packed("token_embd.weight")?);
    let lm_head = gg.qmat("output.weight")?;

    // The head mixer (`output_hc_*`) has no inject — its collapse goes
    // straight to lm_head (the reference passes nullptr there).
    let hc_head = load_hc(gg, "output_hc", false, &cfg)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let experts = experts(gg, l)?;
        layers.push(load_layer(gg, l, cfg.layer_types[l], experts, &cfg)?);
    }

    let mut model = Qwen4Exp::new(embed_tokens, layers, hc_head, lm_head, cfg.clone())?;
    if let Some(p) = opts.kv_precision {
        model.set_kv_precision(p)?;
    }
    if let Some(dt) = opts.moe_act {
        model.set_routed_act(dt);
    }
    if opts.mtp {
        // ds4's layout: the MTP layer is `blk.{n_layers}` (a full-attention
        // layer in the main layers' names) plus `blk.{n_layers}.nextn.*`.
        let l = cfg.n_layers;
        let p = format!("blk.{l}");
        let experts = experts(gg, l)?;
        let layer = load_layer(gg, l, LayerKind::Full, experts, &cfg)?;
        let head = MtpHead::new(
            RmsNorm::new(gg.f32(&format!("{p}.nextn.enorm.weight"))?, cfg.norm_eps),
            RmsNorm::new(gg.f32(&format!("{p}.nextn.hnorm.weight"))?, cfg.norm_eps),
            gg.qmat(&format!("{p}.nextn.eh_proj.weight"))?,
            layer,
            load_hc(gg, &format!("{p}.nextn.hc_head"), false, &cfg)?,
            &cfg,
        );
        model.set_mtp(head);
        if let Some(dt) = opts.moe_act {
            model.set_routed_act(dt); // the head's MoE follows the model's routed arm
        }
    }
    if let (Some(hash), Some(&l)) = (ple_hash, cfg.ple_layers.first()) {
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
        let table = if gg.ghost {
            PleTable::Dense(
                gg.ghost_dense("per_layer_token_embd.weight", lloom_tensor::DType::F32)?,
            )
        } else {
            let table_path: PathBuf = match &opts.ple_shard {
                Some(p) => p.clone(),
                None => gg.locate("per_layer_token_embd.weight")?.1.to_path_buf(),
            };
            let table = MmapPleTable::open(&table_path)?;
            if table.rows() < hash.total_rows() as usize {
                lloom_tensor::bail!(
                    "ple table has {} rows, the hash needs {}",
                    table.rows(),
                    hash.total_rows()
                )
            }
            match opts.ple_cache_rows {
                Some(rows) => PleTable::Cached(CachedPleTable::new(table, rows)?),
                None => PleTable::Mmap(table),
            }
        };
        model.set_ple(l, hash, table, block)?;
    }
    #[cfg(debug_assertions)]
    for t in &schema.tensors {
        debug_assert!(
            gg.read.borrow().contains(&t.name),
            "the tensor schema lists `{}`, which the loader never read",
            t.name
        );
    }
    // The optimizer decides the brackets' epilogues over the model as
    // built, once (`qwen4exp.fusion = derived`); `manual` leaves them on.
    #[cfg(not(target_arch = "wasm32"))]
    if crate::tuning::tuning().qwen4exp.fusion == "derived" {
        let line = model.derive_fusion()?;
        eprintln!("{line}");
    }
    Ok(model)
}

/// One HyperConnection bracket `{p}_{norm,down,up[,inject]}`.
fn load_hc<Rd: Read + Seek>(
    gg: &mut Gguf<'_, Rd>,
    p: &str,
    inject: bool,
    cfg: &Config,
) -> Result<HyperConnection> {
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
        cfg.hc_count,
        cfg.hidden_size,
        cfg.norm_eps as f32,
    )
}

/// Layer `blk.{l}` of `kind` around `experts` — the model's layers and the
/// MTP layer (`blk.{n_layers}`) share this.
fn load_layer<P: Profile, R: ExpertResidency, Rd: Read + Seek>(
    gg: &mut Gguf<'_, Rd>,
    l: usize,
    kind: LayerKind,
    experts: R,
    cfg: &Config,
) -> Result<Layer<P, R>> {
    let p = format!("blk.{l}");
    let d = cfg.hidden_size;
    let hc_attn = load_hc(gg, &format!("{p}.hc_attn"), true, cfg)?;
    let hc_ffn = load_hc(gg, &format!("{p}.hc_ffn"), true, cfg)?;
    let mixer = match kind {
        LayerKind::Linear => {
            let (mut stems, cat) = gg.qmat_stack(&[
                format!("{p}.attn_qkv.weight"),
                format!("{p}.attn_gate.weight"),
                format!("{p}.ssm_alpha.weight"),
                format!("{p}.ssm_beta.weight"),
            ])?;
            let mut ssm_beta = stems.pop().expect("four stems");
            let mut ssm_alpha = stems.pop().expect("four stems");
            let gate = stems.pop().expect("four stems");
            let wqkv = stems.pop().expect("four stems");
            // The projections are quantized per tensor in this checkpoint
            // (`attn_qkv` / `attn_gate` IQ4_XS and friends, differing by
            // layer), so the four rarely stack; the two gate stems (α, β —
            // BF16 in the checkpoint, f16 here) always do.
            let ab_cat = if cat.is_none() {
                let (mut ab, ab_cat) = gg.qmat_stack(&[
                    format!("{p}.ssm_alpha.weight"),
                    format!("{p}.ssm_beta.weight"),
                ])?;
                if ab_cat.is_some() {
                    ssm_beta = ab.pop().expect("two stems");
                    ssm_alpha = ab.pop().expect("two stems");
                }
                ab_cat
            } else {
                None
            };
            let block = GatedDeltaNet::new(
                wqkv,
                gate,
                gg.f32(&format!("{p}.ssm_conv1d.weight"))?,
                gg.f32(&format!("{p}.ssm_a"))?,
                gg.f32(&format!("{p}.ssm_dt.bias"))?,
                ssm_alpha,
                ssm_beta,
                gg.f32(&format!("{p}.ssm_norm.weight"))?,
                gg.qmat(&format!("{p}.ssm_out.weight"))?,
                cfg,
            );
            Mixer::Linear(match (cat, ab_cat) {
                (Some(cat), _) => block.with_stem_cat(cat),
                (None, Some(ab)) => block.with_gate_stems_cat(ab),
                (None, None) => block,
            })
        }
        LayerKind::Full => {
            let indexer = Indexer::new(
                gg.qmat(&format!("{p}.indexer.q_proj.weight"))?,
                gg.qmat(&format!("{p}.indexer.k_proj.weight"))?,
                RmsNorm::new(gg.f32(&format!("{p}.indexer.q_norm.weight"))?, cfg.norm_eps),
                RmsNorm::new(gg.f32(&format!("{p}.indexer.k_norm.weight"))?, cfg.norm_eps),
                cfg,
            )?;
            let (mut stems, cat) = gg.qmat_stack(&[
                format!("{p}.attn_q.weight"),
                format!("{p}.attn_k.weight"),
                format!("{p}.attn_v.weight"),
            ])?;
            let wv = stems.pop().expect("three stems");
            let wk = stems.pop().expect("three stems");
            let wq = stems.pop().expect("three stems");
            let attn = QsaAttention::new(
                wq,
                wk,
                wv,
                gg.qmat(&format!("{p}.attn_output.weight"))?,
                RmsNorm::new(gg.f32(&format!("{p}.attn_q_norm.weight"))?, cfg.norm_eps),
                RmsNorm::new(gg.f32(&format!("{p}.attn_k_norm.weight"))?, cfg.norm_eps),
                cfg,
            )
            .with_indexer(indexer);
            attn.admit(gg.purpose, &gg.device)?;
            Mixer::Full(match cat {
                Some(cat) => attn.with_stem_cat(cat),
                None => attn,
            })
        }
    };
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
        cfg,
    );
    Ok(Layer::new(hc_attn, mixer, hc_ffn, moe))
}

#[cfg(test)]
mod tests {
    #[test]
    fn numeric_session_predictors_refuse_overflow_and_preserve_zero_attention() {
        use crate::nn::ops::{KvPrecision, KvQuant};
        use lloom_tensor::DType;
        let mut cfg = Config::tiny();
        assert!(kv_bytes_per_token(&cfg).unwrap() > 0);
        cfg.n_kv_heads = usize::MAX;
        assert!(kv_bytes_per_token(&cfg).is_err());
        assert!(kv_slab_bytes(&cfg).is_err());
        cfg.layer_types.clear();
        cfg.head_dim = usize::MAX;
        cfg.indexer_head_dim = usize::MAX;
        cfg.kv_precision = KvPrecision::quantized(KvQuant::Int4).unwrap();
        assert_eq!(kv_bytes_per_token(&cfg).unwrap(), 0);
        assert_eq!(kv_slab_bytes(&cfg).unwrap(), 0);
        let claims =
            session_claims(&cfg, usize::MAX, usize::MAX, 1, &StreamLoader::Io, false).unwrap();
        assert_eq!(claims[0].bytes, 0);
        let cfg = Config::tiny();
        assert!(session_claims(&cfg, usize::MAX, 1, 1, &StreamLoader::Io, false).is_err());
        assert!(session_claims(&cfg, 1, usize::MAX, 1, &StreamLoader::Io, false).is_err());
        for precision in [
            KvPrecision::dense(DType::F32),
            KvPrecision::quantized(KvQuant::Int8).unwrap(),
            KvPrecision::quantized(KvQuant::Int4).unwrap(),
        ] {
            assert!(precision
                .checked_bytes_per_token(usize::MAX, 4, DType::F32)
                .is_err());
        }
        let int4 = KvPrecision::quantized(KvQuant::Int4).unwrap();
        assert!(int4.checked_bytes_per_token(1, 3, DType::F32).is_err());
        assert_eq!(int4.checked_bytes_per_token(0, 3, DType::F32).unwrap(), 0);
        assert_eq!(int4.checked_bytes_per_token(2, 4, DType::F32).unwrap(), 40);
    }

    #[test]
    fn numeric_transient_and_claim_sum_are_exact_or_refused() {
        let per = super::super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN;
        let tile = usize::MAX / 3 / per;
        let (unit, bytes) = prefill_transient_bytes(tile, 6).unwrap();
        assert_eq!(bytes as u128, unit as u128 * 3);
        assert!(unit as u128 * 4 > usize::MAX as u128);
        assert!(prefill_transient_bytes(usize::MAX, 2).is_err());
        assert!(prefill_transient_bytes(1, usize::MAX).is_err());
        let c = |bytes| SessionClaim {
            name: "numeric",
            bytes,
            unit: 1,
            capability: true,
        };
        assert_eq!(session_floor_bytes(&[c(2), c(3)]).unwrap(), 5);
        assert!(session_floor_bytes(&[c(usize::MAX), c(1)]).is_err());
        let mut cfg = Config::tiny();
        cfg.n_experts_per_tok = usize::MAX;
        assert!(checked_expert_slot_floor(&cfg).is_err());
        cfg.n_experts_per_tok = cfg.n_experts;
        assert!(checked_expert_slot_floor(&cfg).is_err());
        cfg.n_experts_per_tok = 0;
        assert_eq!(checked_expert_slot_floor(&cfg).unwrap(), 1);
    }

    fn header(dtype: GgmlDType, dims: &[usize]) -> TensorInfo {
        TensorInfo {
            ggml_dtype: dtype,
            shape: dims.into(),
            offset: 0,
        }
    }

    #[test]
    fn numeric_header_totals_preserve_tiers_scalar_quantized_and_zero() -> Result<()> {
        let mut totals = HeaderTotals::default();
        totals.add("scalar", &header(GgmlDType::F32, &[]), None)?;
        assert_eq!(totals.resident, 4);
        totals.add(
            "blk.0.ffn_exps.weight",
            &header(GgmlDType::Q4_0, &[2, 1, 32]),
            None,
        )?;
        assert_eq!((totals.per_slot, totals.total), (18, 36));
        totals.add(
            "blk.1.ffn_exps.weight",
            &header(GgmlDType::Q4_0, &[2, 1, 32]),
            Some("blk.1."),
        )?;
        assert_eq!(
            totals,
            HeaderTotals {
                resident: 40,
                per_slot: 18,
                total: 36
            }
        );
        let before = totals;
        totals.add("empty", &header(GgmlDType::F32, &[usize::MAX, 2, 0]), None)?;
        // Mmap PLE remains outside this device-weight term, not an admitted or
        // physically free allocation; its invalid/huge shape is not read here.
        totals.add(
            "per_layer_token_embd.weight",
            &header(GgmlDType::F32, &[usize::MAX, 2]),
            None,
        )?;
        assert_eq!(totals, before);
        // Preserve bytes/experts truncation for a header rejected later by the
        // streamed reader's stricter k/block validation (2*16 is one Q4 block).
        let mut aggregate = HeaderTotals::default();
        aggregate.add(
            "blk.0.ffn_exps.weight",
            &header(GgmlDType::Q4_0, &[2, 1, 16]),
            None,
        )?;
        assert_eq!((aggregate.per_slot, aggregate.total), (9, 18));
        Ok(())
    }

    #[test]
    fn numeric_header_totals_refuse_overflow_without_partial_updates() {
        let ordinary = HeaderTotals {
            resident: 4,
            per_slot: 18,
            total: 36,
        };
        for (info, expected) in [
            (
                header(GgmlDType::F32, &[usize::MAX, 2]),
                "dimension product",
            ),
            (header(GgmlDType::F32, &[usize::MAX]), "packed bytes"),
            (header(GgmlDType::Q4_0, &[31]), "not divisible"),
        ] {
            let mut totals = ordinary;
            let err = totals.add("bad", &info, None).unwrap_err().to_string();
            assert!(err.contains("`bad`") && err.contains(expected), "{err}");
            assert_eq!(totals, ordinary);
        }
        for (before, name, expected) in [
            (
                HeaderTotals {
                    resident: usize::MAX - 3,
                    ..ordinary
                },
                "scalar",
                "resident sum",
            ),
            (
                HeaderTotals {
                    per_slot: usize::MAX - 3,
                    ..ordinary
                },
                "ffn_exps.weight",
                "per-slot sum",
            ),
            // The per-slot update succeeds in the local copy before total
            // fails: neither field may be published by this failed add.
            (
                HeaderTotals {
                    total: usize::MAX - 3,
                    ..ordinary
                },
                "ffn_exps.weight",
                "expert total",
            ),
        ] {
            let mut totals = before;
            let err = totals
                .add(name, &header(GgmlDType::F32, &[1]), None)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{err}");
            assert_eq!(totals, before);
        }
    }

    #[test]
    fn numeric_load_plan_extremes_refuse_before_quota_bypasses() {
        let max = usize::MAX;
        let mut plan = LoadPlan {
            resident_bytes: max,
            expert_bytes_per_slot: 1,
            expert_bytes_total: 0,
            slots: None,
            working_set: max,
            fraction: 1.0,
        };
        assert_eq!(plan.predicted(), max as u128);
        assert!(plan.fits());
        assert!(admit_load(&plan).is_ok());
        plan.expert_bytes_total = 1;
        assert_eq!(plan.predicted(), max as u128 + 1);
        for working_set in [0, max] {
            for fraction in [0.0, -1.0, 1.0, f64::INFINITY, f64::NAN] {
                plan.working_set = working_set;
                plan.fraction = fraction;
                assert!(!plan.fits());
                let err = admit_load(&plan).unwrap_err().to_string();
                assert!(
                    err.contains("unrepresentable physical memory demand"),
                    "{err}"
                );
                assert!(err.contains(&(max as u128 + 1).to_string()), "{err}");
                assert!(!err.contains("budget OK"));
                assert!(!err.contains("--memory-fraction"));
            }
        }
        plan.slots = Some(max);
        plan.expert_bytes_per_slot = max;
        assert_eq!(plan.predicted(), max as u128 * max as u128 + max as u128);
        assert!(!plan.fits());
        assert!(admit_load(&plan).is_err());
        assert!(plan.physical_bytes().is_err());
    }

    #[test]
    fn numeric_load_plan_fraction_compatibility_is_not_new_policy_validation() {
        let mut plan = LoadPlan {
            resident_bytes: 1,
            expert_bytes_per_slot: 0,
            expert_bytes_total: 0,
            slots: None,
            working_set: 10,
            fraction: -1.0,
        };
        // Existing policy, not a new recommendation: <=0 bypasses quota,
        // infinity casts its budget to MAX; NaN casts to zero.
        assert!(plan.fits());
        plan.fraction = f64::INFINITY;
        assert!(plan.fits());
        plan.fraction = f64::NAN;
        assert!(!plan.fits());
        plan.working_set = 0;
        assert!(plan.fits());
        plan.working_set = 10;
        plan.fraction = 0.5;
        plan.resident_bytes = 5;
        assert!(plan.fits());
        plan.resident_bytes = 6;
        assert!(!plan.fits());
    }

    #[test]
    fn numeric_arbitration_loader_consumer_handles_wide_refusal_and_overflow() {
        let mut cfg = Config::tiny();
        cfg.n_experts_per_tok = 1;
        cfg.n_experts = 4;
        let mut plan = LoadPlan {
            resident_bytes: 0,
            expert_bytes_per_slot: 1,
            expert_bytes_total: 4,
            slots: None,
            working_set: 4,
            fraction: 1.0,
        };
        assert_eq!(arbitrate_slots(&plan, &cfg, 1.0, &[], 0).unwrap(), 4);
        plan.working_set = 3;
        assert_eq!(arbitrate_slots(&plan, &cfg, 1.0, &[], 0).unwrap(), 3);
        plan.resident_bytes = usize::MAX;
        let err = arbitrate_slots(&plan, &cfg, 1.0, &[], 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("floors alone"), "{err}");
        assert!(err.contains("against a budget"), "{err}");
        plan.resident_bytes = 0;
        cfg.n_experts_per_tok = usize::MAX;
        assert!(arbitrate_slots(&plan, &cfg, 1.0, &[], 0)
            .unwrap_err()
            .to_string()
            .contains("floor overflows"));
        cfg.n_experts_per_tok = 1;
        plan.expert_bytes_per_slot = usize::MAX;
        assert!(arbitrate_slots(&plan, &cfg, 1.0, &[], 0)
            .unwrap_err()
            .to_string()
            .contains("floor bytes overflow"));
        cfg.n_experts_per_tok = 0;
        plan.expert_bytes_per_slot = 2;
        cfg.n_experts = usize::MAX;
        assert!(arbitrate_slots(&plan, &cfg, 1.0, &[], 0)
            .unwrap_err()
            .to_string()
            .contains("want bytes overflow"));
    }
    use super::*;

    /// Holding a layer's experts whole (the MTP head, `qwen4exp.mtp_experts
    /// = resident`) moves exactly that layer's expert bytes out of the
    /// per-slot unit and the expert total into the resident bytes: the slot
    /// plan, the admission and the ledger's unit see one split.
    #[test]
    fn a_held_layer_is_resident_bytes_not_a_plane_of_every_slot() -> Result<()> {
        let mut cfg = Config::tiny();
        cfg.n_experts = 12;
        let dev = Device::Cpu;
        let held = cfg.n_layers; // the MTP head's index
        let mut owned: Vec<(String, QTensor)> = Vec::new();
        for l in 0..=held {
            let [g, u, d] =
                super::super::testing::expert_stacks(&cfg, l % cfg.n_layers, &dev, GgmlDType::F32);
            owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
            owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
            owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
        }
        let held_bytes: usize = owned
            .iter()
            .filter(|(n, _)| n.starts_with(&format!("blk.{held}.")))
            .map(|(_, t)| t.storage_size_in_bytes())
            .sum();
        let path = std::env::temp_dir().join(format!("lloom-held-{}.gguf", std::process::id()));
        let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        lloom_tensor::quantized::gguf_file::write(&mut File::create(&path)?, &[], &refs)?;
        let mut f = File::open(&path)?;
        let content = Content::read(&mut f)?;
        let gg = Gguf::new(
            vec![Shard {
                content,
                reader: &mut f,
                path: Some(path.clone()),
            }],
            &dev,
        )?;
        let all = plan_load(&gg, &dev, 0.65, Some(4))?;
        let mtp = plan_load_holding(&gg, &dev, 0.65, Some(4), Some(held))?;
        std::fs::remove_file(&path).ok();
        assert!(held_bytes > 0);
        assert_eq!(mtp.resident_bytes, all.resident_bytes + held_bytes);
        assert_eq!(mtp.expert_bytes_total, all.expert_bytes_total - held_bytes);
        assert_eq!(
            mtp.expert_bytes_per_slot,
            all.expert_bytes_per_slot - held_bytes / cfg.n_experts
        );
        // 4 slots of the other layers plus every expert of the held one.
        assert_eq!(
            mtp.predicted(),
            (all.resident_bytes + held_bytes + 4 * mtp.expert_bytes_per_slot) as u128
        );
        Ok(())
    }

    /// The numbers this machine's SLOTPLAN lines carry (issue #749 P18): a
    /// 55.66 GB working set, a 4.12 GB resident tier, 83.8 MB a slot over
    /// 512 experts.
    fn this_machine(slots: Option<usize>, fraction: f64) -> LoadPlan {
        LoadPlan {
            resident_bytes: 4_115_863_040,
            expert_bytes_per_slot: 83_814_400,
            expert_bytes_total: 83_814_400 * 512,
            slots,
            working_set: 55_662_788_608,
            fraction,
        }
    }

    /// `--experts resident` is the run that panicked the machine: 47.0 GB
    /// against a 36.2 GB budget at 0.65. Refused, with the reductions.
    #[test]
    fn the_resident_load_that_panicked_is_refused() {
        let plan = this_machine(None, DEFAULT_MEMORY_FRACTION);
        assert_eq!(plan.predicted(), 47_028_835_840);
        assert!(!plan.fits());
        let e = admit_load(&plan).unwrap_err().to_string();
        assert!(e.contains("memory budget exceeded at load"), "{e}");
        assert!(e.contains("--experts stream"), "{e}");
        assert!(e.contains("--memory-fraction 0.85"), "{e}");
        // What `Auto` sizes to at this fraction is admitted by construction.
        let auto = this_machine(Some(plan.slots_that_fit()), DEFAULT_MEMORY_FRACTION);
        assert_eq!(auto.slots, Some(382));
        assert!(auto.fits() && admit_load(&auto).is_ok());
        assert!(auto.describe().starts_with("memory budget OK at load"));
    }

    /// Exactly on the budget is admitted; one byte over is not; no working
    /// set (not Metal) and a zero fraction never refuse.
    #[test]
    fn the_boundary_is_inclusive_and_the_check_is_metal_only() {
        let mut plan = this_machine(None, 1.0);
        plan.working_set = usize::try_from(plan.predicted()).unwrap();
        assert!(plan.fits());
        plan.working_set -= 1;
        assert!(!plan.fits());
        plan.working_set = 0;
        assert!(plan.fits());
        let off = this_machine(None, 0.0);
        assert!(off.fits());
    }

    /// The reserve at the harness defaults: 4096 positions of KV + two
    /// 512-tiles' transients + the pool's free list. The slot plan leaves it
    /// out of the pool.
    #[test]
    fn the_auto_plan_leaves_the_session_reserve() {
        let cfg = Config::tiny();
        let per = kv_bytes_per_token(&cfg).unwrap();
        let full = cfg
            .layer_types
            .iter()
            .filter(|k| matches!(k, LayerKind::Full))
            .count();
        assert_eq!(
            per,
            full * ((2 * cfg.n_kv_heads * cfg.head_dim + cfg.indexer_head_dim) * 4
                + cfg.indexer_head_dim)
        );
        let claims = |rows, tile| {
            session_floor_bytes(
                &session_claims(&cfg, rows, 4096, tile, &StreamLoader::Io, false).unwrap(),
            )
            .unwrap()
        };
        let r1 = claims(1, 512);
        let r4 = claims(4, 512);
        assert_eq!(r4 - r1, 3 * 4096 * per);
        // The KV, the transient and the loader's staging — and nothing in
        // place of a policy: the free-list and pass-retention reserves the
        // old session reserve carried are the pool's to yield (#892 A5).
        assert_eq!(
            r1,
            4096 * per
                + 512 * super::super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN
                + IO_INFLIGHT_RESERVE_BYTES
        );
        // A zero tile reserves one PREFILL_CHUNK's worth.
        assert_eq!(claims(1, 0), r1);
    }

    /// A position's bytes on this model's geometry, per KV store, against
    /// the arithmetic written out: 12 full-attention layers × (K and V of 2
    /// heads × 256, plus the indexer's 128-wide raw key and a quarter of a
    /// compressed key, f32). The int8 / int4 arms carry one f32 (scale, bias)
    /// pair per head vector — the term a 0.5× / 0.25× estimate leaves out.
    #[test]
    fn the_kv_bytes_follow_the_store_with_the_scales_counted() {
        use crate::nn::ops::{KvPrecision, KvQuant};
        use lloom_tensor::DType;
        let mut cfg = Config::flash_next();
        let indexer = 128 * 4 + 128;
        let per = |cfg: &Config| kv_bytes_per_token(cfg).unwrap();
        assert_eq!(per(&cfg), 12 * (2 * 2 * 256 * 4 + indexer));
        cfg.kv_precision = KvPrecision::dense(DType::F16);
        assert_eq!(per(&cfg), 12 * (2 * 2 * 256 * 2 + indexer));
        cfg.kv_precision = KvPrecision::quantized(KvQuant::Int8).unwrap();
        assert_eq!(per(&cfg), 12 * (2 * 2 * (256 + 8) + indexer));
        cfg.kv_precision = KvPrecision::quantized(KvQuant::Int4).unwrap();
        assert_eq!(per(&cfg), 12 * (2 * 2 * (128 + 8) + indexer));
    }

    /// The default fraction is §9.5k's, the one `MemoryBudget` and the Metal
    /// admission share.
    #[test]
    fn memory_fraction_matches_the_budgets() {
        assert_eq!(
            DEFAULT_MEMORY_FRACTION,
            crate::nn::memory::MemoryBudget::DEFAULT_FRACTION
        );
    }
}
