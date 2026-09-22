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

/// How the routed experts are streamed.
#[derive(Debug, Clone)]
pub struct StreamOptions {
    /// Resident experts per layer per projection (the slot pool).
    pub slots: usize,
    pub loader: StreamLoader,
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
        build(&mut gg, device, opts, |gg, l| {
            let p = format!("blk.{l}");
            let src = |gg: &Gguf<'_, File>, name: String| -> Result<ExpertSource> {
                let (content, path) = gg.locate(&name)?;
                ExpertSource::from_gguf(content, path, &name)
            };
            Streamed::new(
                src(gg, format!("{p}.ffn_gate_exps.weight"))?,
                src(gg, format!("{p}.ffn_up_exps.weight"))?,
                src(gg, format!("{p}.ffn_down_exps.weight"))?,
                stream.slots,
                arm.clone(),
                device,
            )
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
            cfg,
        )),
        LayerKind::Full => {
            let indexer = Indexer::new(
                gg.qmat(&format!("{p}.indexer.q_proj.weight"))?,
                gg.qmat(&format!("{p}.indexer.k_proj.weight"))?,
                RmsNorm::new(gg.f32(&format!("{p}.indexer.q_norm.weight"))?, cfg.norm_eps),
                RmsNorm::new(gg.f32(&format!("{p}.indexer.k_norm.weight"))?, cfg.norm_eps),
                cfg,
            )?;
            Mixer::Full(
                QsaAttention::new(
                    gg.qmat(&format!("{p}.attn_q.weight"))?,
                    gg.qmat(&format!("{p}.attn_k.weight"))?,
                    gg.qmat(&format!("{p}.attn_v.weight"))?,
                    gg.qmat(&format!("{p}.attn_output.weight"))?,
                    RmsNorm::new(gg.f32(&format!("{p}.attn_q_norm.weight"))?, cfg.norm_eps),
                    RmsNorm::new(gg.f32(&format!("{p}.attn_k_norm.weight"))?, cfg.norm_eps),
                    cfg,
                )
                .with_indexer(indexer),
            )
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
