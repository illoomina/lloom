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
use crate::nn::expert_stream::{ExpertSource, LoaderArm};
use crate::nn::RmsNorm;
use lloom_tensor::quantized::gguf_file::{Content, Value};
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
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

/// One open shard: header, reader, and the path it was opened from (`None`
/// for a caller-supplied reader).
struct Shard<'a, R: Read + Seek> {
    content: Content,
    reader: &'a mut R,
    path: Option<PathBuf>,
}

/// The loader's view of a checkpoint: its shards, a name → shard index, the
/// device.
struct Gguf<'a, R: Read + Seek> {
    shards: Vec<Shard<'a, R>>,
    index: HashMap<String, usize>,
    device: Device,
}

impl<'a, R: Read + Seek> Gguf<'a, R> {
    fn new(shards: Vec<Shard<'a, R>>, device: &Device) -> Result<Self> {
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
        })
    }
    fn metadata(&self) -> &HashMap<String, Value> {
        &self.shards[0].content.metadata
    }
    fn shard_of(&self, name: &str) -> Result<usize> {
        match self.index.get(name) {
            Some(&i) => Ok(i),
            None => lloom_tensor::bail!("gguf has no tensor `{name}`"),
        }
    }
    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        let i = self.shard_of(name)?;
        let device = self.device.clone();
        let s = &mut self.shards[i];
        s.content.tensor(s.reader, name, &device)
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let i = self.shard_of(name)?;
        match self.shards[i].content.tensor_infos.get(name) {
            Some(ti) => Ok(ti.shape.dims().to_vec()),
            None => lloom_tensor::bail!("gguf has no tensor `{name}`"),
        }
    }
    /// The shard holding `name`, as `(header, path)`; a shard without a path
    /// cannot be streamed from.
    fn locate(&self, name: &str) -> Result<(&Content, &Path)> {
        let i = self.shard_of(name)?;
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
    fn qmat(&mut self, name: &str) -> Result<QMatMul> {
        let qt = self.qtensor(name)?;
        match qt.dtype() {
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
    fn qmat_stack(&mut self, names: &[String]) -> Result<(Vec<QMatMul>, Option<Tensor>)> {
        let parts: Vec<QMatMul> = names.iter().map(|n| self.qmat(n)).collect::<Result<_>>()?;
        let f16: Vec<&Tensor> = parts
            .iter()
            .filter_map(|p| match p {
                QMatMul::TensorF16(t) => Some(t),
                _ => None,
            })
            .collect();
        if std::env::var("QWEN4EXP_TRACE_LOAD").is_ok() {
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
    fn packed(&mut self, name: &str) -> Result<Arc<QTensor>> {
        Ok(Arc::new(self.qtensor(name)?))
    }
    /// An f32 vector/tensor (norm weights, conv, SSM scalars).
    fn f32(&mut self, name: &str) -> Result<Tensor> {
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
    /// Sized from the device: `fraction` of `recommendedMaxWorkingSetSize`,
    /// minus the resident tier (every non-expert tensor, from the headers),
    /// minus the SESSION RESERVE ([`session_reserve_bytes`]: the KV of
    /// `rows` rows to `kv_tokens`, two prefill tiles' transients, the pool's
    /// free list), divided across the layers' three expert planes; clamped
    /// to `[topk + 1, n_experts]`. ds4's byte-budget planner, without a
    /// policy beyond "what fits" (§20.4a) — and the reserve is what keeps
    /// "what fits" from being the whole budget: without it the pool sized
    /// itself to `fraction × working_set` exactly and the first pass read
    /// over budget (issue #749 P18, `p18-stream-short.txt`).
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

/// Bytes of the pool's free list a streamed load reserves beside the KV and
/// the transients (`buffer_pool`'s 256 MiB default free budget).
pub const POOL_FREE_RESERVE_BYTES: usize = 256 * 1024 * 1024;

/// One position's KV over every full-attention layer, f32 k and v
/// `[n_kv_heads, head_dim]` each — plus the indexer's raw key per position.
pub fn kv_bytes_per_token(cfg: &Config) -> usize {
    let full = cfg
        .layer_types
        .iter()
        .filter(|k| matches!(k, LayerKind::Full))
        .count();
    let f32 = std::mem::size_of::<f32>();
    full * (2 * cfg.n_kv_heads * cfg.head_dim + cfg.indexer_head_dim) * f32
}

/// What a streamed load keeps beside its weights: `rows` rows' KV to
/// `kv_tokens` positions, the prefill's transient at
/// [`super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN`] (measured with the
/// pacer's two tiles in flight, so one factor of the tile), what a pass
/// retains, and the pool's free list. The slot plan subtracts it from the budget so
/// a pass has what it needs by construction, not by luck.
pub fn session_reserve_bytes(
    cfg: &Config,
    rows: usize,
    kv_tokens: usize,
    prefill_tile: usize,
) -> usize {
    let tile = if prefill_tile == 0 {
        super::model::PREFILL_CHUNK
    } else {
        prefill_tile
    };
    // One tile's worth: `PREFILL_TRANSIENT_BYTES_PER_TOKEN` is the peak
    // measured WITH the pacer's two tiles in flight (1.34 GB over a 512
    // tile), so a second factor of two counted the same bytes twice — 1.5
    // GB, ~18 slots, on this machine (P26).
    rows.max(1) * kv_tokens * kv_bytes_per_token(cfg)
        + tile * super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN
        + POOL_FREE_RESERVE_BYTES
        + PASS_RETENTION_BYTES
}

/// What the device holds after a prefill tile beyond the KV and the free
/// list — the pool's `pending` at the tile's end and the buckets it keeps:
/// measured +1.0–1.1 GB over the load after one 512-tile (P18/P21). Without
/// it the `io` arm (no staging ring in its reserve) was refused its second
/// tile by 0.07 GB (`p21-io-cold.txt`).
pub const PASS_RETENTION_BYTES: usize = 1 << 30;

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
}

/// The slots `Auto` resolves to, and the terms it resolved them from.
#[derive(Debug, Clone, Copy)]
pub struct SlotPlan {
    pub slots: usize,
    pub working_set: usize,
    pub resident_bytes: usize,
    pub expert_bytes_per_slot: usize,
    /// The session reserve the plan left out of the pool.
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
) -> LoadPlan {
    let mut resident = 0usize;
    let mut per_slot = 0usize;
    let mut total = 0usize;
    for sh in &gg.shards {
        for (name, info) in &sh.content.tensor_infos {
            let bytes = info.shape.elem_count() / info.ggml_dtype.block_size()
                * info.ggml_dtype.type_size();
            if name == "per_layer_token_embd.weight" {
                continue; // memory-mapped, not resident
            }
            if name.ends_with("_exps.weight") {
                // [n_experts, n, k]: one plane per slot
                let n_experts = info.shape.dims().first().copied().unwrap_or(1).max(1);
                per_slot += bytes / n_experts;
                total += bytes;
            } else {
                resident += bytes;
            }
        }
    }
    LoadPlan {
        resident_bytes: resident,
        expert_bytes_per_slot: per_slot,
        expert_bytes_total: total,
        slots,
        working_set: working_set_of(device),
        fraction,
    }
}

/// `recommendedMaxWorkingSetSize` of a Metal device; 0 elsewhere.
fn working_set_of(device: &Device) -> usize {
    #[cfg(feature = "metal")]
    if let Device::Metal(d) = device {
        return d.device().recommended_max_working_set_size();
    }
    let _ = device;
    0
}

/// The slot pool `SlotBudget::Auto` resolves to: what the budget leaves
/// after the resident tier, divided by a slot, clamped to `[topk + 1,
/// n_experts]`.
fn plan_slots<Rd: Read + Seek>(
    gg: &Gguf<'_, Rd>,
    cfg: &Config,
    device: &Device,
    fraction: f64,
    reserve_bytes: usize,
) -> Result<SlotPlan> {
    let plan = plan_load(gg, device, fraction, None);
    let slots = plan
        .budget()
        .saturating_sub(plan.resident_bytes)
        .saturating_sub(reserve_bytes)
        .checked_div(plan.expert_bytes_per_slot)
        .unwrap_or(0)
        .clamp(cfg.n_experts_per_tok + 1, cfg.n_experts);
    Ok(SlotPlan {
        slots,
        working_set: plan.working_set,
        resident_bytes: plan.resident_bytes,
        expert_bytes_per_slot: plan.expert_bytes_per_slot,
        reserve_bytes,
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
}

impl LoadOptions {
    /// The admission fraction this load runs at.
    pub fn fraction(&self) -> f64 {
        self.memory_fraction.unwrap_or(DEFAULT_MEMORY_FRACTION)
    }
}

/// §9.5k's fraction, `MemoryBudget::DEFAULT_FRACTION` (0.65) — asserted
/// equal to it by `memory_fraction_matches_lfm2s`.
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
    /// Bytes the load holds on the device.
    pub fn predicted(&self) -> usize {
        let experts = match self.slots {
            Some(n) => n * self.expert_bytes_per_slot,
            None => self.expert_bytes_total,
        };
        self.resident_bytes + experts
    }
    /// `fraction × working_set`.
    pub fn budget(&self) -> usize {
        (self.working_set as f64 * self.fraction) as usize
    }
    /// Whether the load is admitted: exactly on the budget is; no working
    /// set (not Metal) or a zero fraction never refuses.
    pub fn fits(&self) -> bool {
        self.working_set == 0 || self.fraction <= 0.0 || self.predicted() <= self.budget()
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
            gb(self.predicted()),
            gb(self.resident_bytes),
            gb(self.budget()),
            self.fraction,
            gb(self.working_set),
        );
        if self.fits() {
            return format!(
                "memory budget OK at load: {head}; residual {:.2} GB for the KV, the prefill passes and the pool",
                gb(self.budget().saturating_sub(self.predicted()))
            );
        }
        let mut s = format!(
            "memory budget exceeded at load: {head}; over by {:.2} GB. Nothing was loaded (DESIGN.md 9.5g: refuse rather than silently reduce). What would fit:",
            gb(self.predicted().saturating_sub(self.budget()))
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
            gb(self.working_set.saturating_sub(self.predicted()))
        ));
        s
    }
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
fn open_shards(paths: &[PathBuf]) -> Result<(Vec<Content>, Vec<File>)> {
    let mut contents = Vec::with_capacity(paths.len());
    let mut files = Vec::with_capacity(paths.len());
    for p in paths {
        let mut f = File::open(p)?;
        contents.push(Content::read(&mut f)?);
        files.push(f);
    }
    Ok((contents, files))
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
        let mut gg = Gguf::new(
            vec![Shard {
                content,
                reader,
                path: None,
            }],
            device,
        )?;
        admit_load(&plan_load(&gg, device, opts.fraction(), None))?;
        build(&mut gg, device, opts, |gg, l| {
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
        admit_load(&plan_load(&gg, device, opts.fraction(), None))?;
        build(&mut gg, device, opts, |gg, l| {
            let p = format!("blk.{l}");
            Ok(Resident::new(
                gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
                gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
            ))
        })
    }
}

impl Qwen4Exp<Streamed> {
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
        let cfg0 = Config::from_gguf_metadata(gg.metadata())?;
        let slots = match stream.slots {
            SlotBudget::Fixed(n) => n,
            SlotBudget::Auto {
                fraction,
                rows,
                kv_tokens,
                prefill_tile,
            } => {
                let reserve = session_reserve_bytes(&cfg0, rows, kv_tokens, prefill_tile)
                    + staging_reserve_bytes(&stream.loader, stream.prefetch);
                let plan = plan_slots(&gg, &cfg0, device, fraction, reserve)?;
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
        admit_load(&plan_load(&gg, device, opts.fraction(), Some(slots)))?;
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
                lloom_tensor::kernels::metal::staging::StagingRing::shared_with_cap(
                    d,
                    staging_ring_bytes(&stream.loader, stream.prefetch),
                ),
            ),
            _ => None,
        };
        #[cfg(not(feature = "metal"))]
        let staging = None::<()>;
        // One prefetcher for the tier, over the same ring.
        #[cfg(feature = "metal")]
        let prefetcher = match (device, &staging, stream.prefetch) {
            (Device::Metal(d), Some(ring), true) => {
                Some(Arc::new(crate::nn::expert_prefetch::Prefetcher::with_cap(
                    d,
                    Arc::clone(ring),
                    PREFETCH_STAGED_BYTES,
                )))
            }
            _ => None,
        };
        build(&mut gg, device, opts, |gg, l| {
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
                slots,
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
        })
    }
}

/// The model from the opened shards, `experts(gg, l)` supplying layer `l`'s
/// residency strategy.
fn build<R: ExpertResidency, Rd: Read + Seek>(
    gg: &mut Gguf<'_, Rd>,
    _device: &Device,
    opts: &LoadOptions,
    mut experts: impl FnMut(&mut Gguf<'_, Rd>, usize) -> Result<R>,
) -> Result<Qwen4Exp<R>> {
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

    let mut model = Qwen4Exp::new(embed_tokens, layers, hc_head, lm_head, cfg.clone());
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
        let table = match opts.ple_cache_rows {
            Some(rows) => PleTable::Cached(CachedPleTable::new(table, rows)?),
            None => PleTable::Mmap(table),
        };
        model.set_ple(l, hash, table, block)?;
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
fn load_layer<R: ExpertResidency, Rd: Read + Seek>(
    gg: &mut Gguf<'_, Rd>,
    l: usize,
    kind: LayerKind,
    experts: R,
    cfg: &Config,
) -> Result<Layer<R>> {
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
    use super::*;

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
        plan.working_set = plan.predicted();
        assert!(plan.fits());
        plan.working_set -= 1;
        assert!(!plan.fits());
        plan.working_set = 0;
        assert!(plan.fits());
        let off = this_machine(None, 0.0);
        assert!(off.fits());
    }

    /// The reserve at the harness defaults on this model's geometry: 12
    /// full-attention layers, 2 kv heads × 256, indexer 128 → 50 176 B a
    /// position; 4096 positions + two 512-tiles' transients + the pool's
    /// free list. The slot plan leaves it out of the pool.
    #[test]
    fn the_auto_plan_leaves_the_session_reserve() {
        let cfg = Config::tiny();
        let per = kv_bytes_per_token(&cfg);
        let full = cfg
            .layer_types
            .iter()
            .filter(|k| matches!(k, LayerKind::Full))
            .count();
        assert_eq!(
            per,
            full * (2 * cfg.n_kv_heads * cfg.head_dim + cfg.indexer_head_dim) * 4
        );
        let r1 = session_reserve_bytes(&cfg, 1, 4096, 512);
        let r4 = session_reserve_bytes(&cfg, 4, 4096, 512);
        assert_eq!(r4 - r1, 3 * 4096 * per);
        assert_eq!(
            r1,
            4096 * per
                + 512 * super::super::model::PREFILL_TRANSIENT_BYTES_PER_TOKEN
                + POOL_FREE_RESERVE_BYTES
                + PASS_RETENTION_BYTES
        );
        // A zero tile reserves one PREFILL_CHUNK's worth.
        assert_eq!(session_reserve_bytes(&cfg, 1, 4096, 0), r1);
    }

    /// The default fraction is §9.5k's, the one LFM2's budget and the Metal
    /// admission share.
    #[test]
    fn memory_fraction_matches_lfm2s() {
        assert_eq!(
            DEFAULT_MEMORY_FRACTION,
            crate::models::lfm2::MemoryBudget::DEFAULT_FRACTION
        );
    }
}
