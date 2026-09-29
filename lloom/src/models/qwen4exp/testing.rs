//! Synthetic Qwen4-exp models at tiny geometry, for tests: deterministic
//! weights from a seed, every block real, any residency strategy. Shared by
//! the unit tests here and the crate's integration tests.
#![doc(hidden)]

use super::config::{Config, LayerKind};
use super::deltanet_block::GatedDeltaNet;
use super::hc::HyperConnection;
use super::model::{Layer, Mixer, Qwen4Exp};
use super::moe::{ExpertResidency, MoeLayer, Resident};
use super::mtp::MtpHead;
use super::qsa::{Indexer, QsaAttention};
use crate::nn::RmsNorm;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::Arc;

thread_local! {
    static F16_STEMS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Build the dense matrices of this thread's models as `TensorF16` (the
/// real checkpoint's dense-stem kind — the mixed gemv's row tiles) instead
/// of f32 `Tensor` (which the Metal gemm re-orders at `m > 1`). A test that
/// asserts bit-identity of a rows pass against single rows sets it.
pub fn set_f16_stems(on: bool) {
    F16_STEMS.with(|f| f.set(on));
}

/// A dense `[out, inp]` matrix from a seed.
pub fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
    let v: Vec<f32> = (0..out * inp)
        .map(|i| ((i as f32 * 0.007 + seed).sin()) * 0.08)
        .collect();
    let t = Tensor::from_vec(v, (out, inp), dev).unwrap();
    if F16_STEMS.with(|f| f.get()) {
        QMatMul::TensorF16(t.to_dtype(DType::F16).unwrap())
    } else {
        QMatMul::Tensor(t)
    }
}

pub fn ones(n: usize, dev: &Device) -> Tensor {
    Tensor::ones(n, DType::F32, dev).unwrap()
}

pub fn wt(shape: (usize, usize), seed: f32, dev: &Device) -> Tensor {
    let (r, c) = shape;
    Tensor::from_vec(
        (0..r * c)
            .map(|i| ((i as f32 * 0.009 + seed).cos()) * 0.05)
            .collect::<Vec<f32>>(),
        (r, c),
        dev,
    )
    .unwrap()
}

/// A stacked `[e, out, inp]` expert weight from a seed, quantized to `dtype`
/// (`F32` keeps every arm exact).
pub fn stacked(
    e: usize,
    out: usize,
    inp: usize,
    seed: f32,
    dev: &Device,
    dtype: GgmlDType,
) -> QTensor {
    let v: Vec<f32> = (0..e * out * inp)
        .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
        .collect();
    let t = Tensor::from_vec(v, (e, out, inp), dev).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

/// The three expert stacks of layer `l` (gate, up, down), the seeds the
/// resident and streamed builds share.
pub fn expert_stacks(cfg: &Config, l: usize, dev: &Device, dtype: GgmlDType) -> [QTensor; 3] {
    let d = cfg.hidden_size;
    let ff = cfg.moe_intermediate_size;
    let ne = cfg.n_experts;
    let s = l as f32 * 0.37;
    [
        stacked(ne, ff, d, 1.0 + s, dev, dtype),
        stacked(ne, ff, d, 100.0 + s, dev, dtype),
        stacked(ne, d, ff, 200.0 + s, dev, dtype),
    ]
}

pub fn hc(cfg: &Config, seed: f32, dev: &Device) -> HyperConnection {
    hc_with(cfg, seed, dev, true)
}

pub fn hc_with(cfg: &Config, seed: f32, dev: &Device, inject: bool) -> HyperConnection {
    let hc_dim = cfg.hc_dim();
    HyperConnection::new(
        ones(hc_dim, dev),
        lin(cfg.hc_lowrank, hc_dim, seed, dev),
        lin(hc_dim, cfg.hc_lowrank, seed + 1.0, dev),
        inject.then(|| lin(cfg.hc_count, hc_dim, seed + 2.0, dev)),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.norm_eps as f32,
    )
    .unwrap()
}

pub fn deltanet(cfg: &Config, dev: &Device) -> GatedDeltaNet {
    let d = cfg.hidden_size;
    let cc = cfg.conv_channels();
    let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let hv = cfg.ssm_dt_rank;
    GatedDeltaNet::new(
        lin(cc, d, 1.0, dev),
        lin(vd, d, 2.0, dev),
        wt((cc, cfg.ssm_conv_kernel), 3.0, dev),
        Tensor::from_vec((0..hv).map(|_| -0.5f32).collect::<Vec<_>>(), hv, dev).unwrap(),
        Tensor::zeros(hv, DType::F32, dev).unwrap(),
        lin(hv, d, 4.0, dev),
        lin(hv, d, 5.0, dev),
        ones(cfg.ssm_state_size, dev),
        lin(d, vd, 6.0, dev),
        cfg,
    )
}

pub fn qsa(cfg: &Config, dev: &Device) -> QsaAttention {
    qsa_seeded(cfg, dev, 0.0)
}

pub fn qsa_seeded(cfg: &Config, dev: &Device, seed: f32) -> QsaAttention {
    let d = cfg.hidden_size;
    let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    QsaAttention::new(
        lin(nh * hd * 2, d, 1.0 + seed, dev),
        lin(nkv * hd, d, 2.0 + seed, dev),
        lin(nkv * hd, d, 3.0 + seed, dev),
        lin(d, nh * hd, 4.0 + seed, dev),
        RmsNorm::new(ones(hd, dev), cfg.norm_eps),
        RmsNorm::new(ones(hd, dev), cfg.norm_eps),
        cfg,
    )
}

/// One layer's MoE around `experts`.
pub fn moe_with<R: ExpertResidency>(cfg: &Config, dev: &Device, experts: R) -> MoeLayer<R> {
    let d = cfg.hidden_size;
    let ne = cfg.n_experts;
    let sh = cfg.shared_expert_intermediate_size;
    MoeLayer::new(
        lin(ne, d, 1.0, dev),
        experts,
        lin(sh, d, 2.0, dev),
        lin(sh, d, 3.0, dev),
        lin(d, sh, 4.0, dev),
        lin(1, d, 5.0, dev),
        cfg,
    )
}

/// Layer `l`'s resident experts.
pub fn resident_experts(cfg: &Config, l: usize, dev: &Device) -> Resident {
    let [g, u, dn] = expert_stacks(cfg, l, dev, GgmlDType::F32);
    Resident::new(Arc::new(g), Arc::new(u), Arc::new(dn))
}

pub fn moe(cfg: &Config, dev: &Device) -> MoeLayer<Resident> {
    moe_with(cfg, dev, resident_experts(cfg, 0, dev))
}

/// A whole model with `experts(l)` supplying layer `l`'s residency strategy.
pub fn model_with<R: ExpertResidency>(
    cfg: &Config,
    dev: &Device,
    experts: impl FnMut(usize) -> Result<R>,
) -> Result<Qwen4Exp<R>> {
    build(cfg, dev, experts, false)
}

/// [`model_with`] with every full-attention layer carrying the QSA indexer
/// (`cfg.indexer_*`), as the checkpoint's do: a decode step that completes a
/// block of `indexer_compress_ratio` keys runs the block's compression.
pub fn model_with_indexer<R: ExpertResidency>(
    cfg: &Config,
    dev: &Device,
    experts: impl FnMut(usize) -> Result<R>,
) -> Result<Qwen4Exp<R>> {
    build(cfg, dev, experts, true)
}

fn indexer(cfg: &Config, dev: &Device) -> Result<Indexer> {
    let (h, ih, id) = (cfg.hidden_size, cfg.indexer_n_heads, cfg.indexer_head_dim);
    Indexer::new(
        lin(ih * id, h, 5.0, dev),
        lin(id, h, 6.0, dev),
        RmsNorm::new(ones(id, dev), cfg.norm_eps),
        RmsNorm::new(ones(id, dev), cfg.norm_eps),
        cfg,
    )
}

fn build<R: ExpertResidency>(
    cfg: &Config,
    dev: &Device,
    mut experts: impl FnMut(usize) -> Result<R>,
    indexed: bool,
) -> Result<Qwen4Exp<R>> {
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let mixer = match cfg.layer_types[l] {
            LayerKind::Linear => Mixer::Linear(deltanet(cfg, dev)),
            LayerKind::Full if indexed => {
                Mixer::Full(qsa(cfg, dev).with_indexer(indexer(cfg, dev)?))
            }
            LayerKind::Full => Mixer::Full(qsa(cfg, dev)),
        };
        layers.push(Layer::new(
            hc(cfg, l as f32, dev),
            mixer,
            hc(cfg, l as f32 + 50.0, dev),
            moe_with(cfg, dev, experts(l)?),
        ));
    }
    Ok(Qwen4Exp::new(
        QMatMul::Tensor(wt((cfg.vocab_size, cfg.hidden_size), 0.5, dev)),
        layers,
        hc(cfg, 999.0, dev),
        lin(cfg.vocab_size, cfg.hidden_size, 7.0, dev),
        cfg.clone(),
    ))
}

/// A whole model, every expert resident.
pub fn resident_model(cfg: &Config, dev: &Device) -> Qwen4Exp<Resident> {
    model_with(cfg, dev, |l| Ok(resident_experts(cfg, l, dev))).expect("resident build")
}

/// A one-layer MTP head around `experts` (a full-attention layer).
pub fn mtp_head<R: ExpertResidency>(cfg: &Config, dev: &Device, experts: R) -> MtpHead<R> {
    let d = cfg.hidden_size;
    let nd = cfg.hc_dim();
    let layer = Layer::new(
        hc(cfg, 300.0, dev),
        Mixer::Full(qsa_seeded(cfg, dev, 11.0)),
        hc(cfg, 350.0, dev),
        moe_with(cfg, dev, experts),
    );
    MtpHead::new(
        RmsNorm::new(ones(d, dev), cfg.norm_eps),
        RmsNorm::new(ones(nd, dev), cfg.norm_eps),
        lin(d, 2 * d, 21.0, dev),
        layer,
        hc_with(cfg, 400.0, dev, false),
        cfg,
    )
}

/// Qwen4-exp at tiny geometry (`docs/porting-models.md` §21.8): the
/// [`TinyFamily`](crate::testing::tiny::TinyFamily) impl. Its schema is
/// declared (`super::schema::tensor_schema`, which the loader validates
/// against), its files are one GGUF with the family's header keys, and its
/// in-memory model is that GGUF's bytes through the real loader — the one
/// place the names are wired to the blocks.
pub struct Qwen4ExpTiny;

use crate::testing::tiny::{
    Schema as TinySchema, TinyCheckpoint, TinyFamily, TinyTokenizer, Workload,
};
use lloom_tensor::quantized::gguf_file::Value;

impl TinyFamily for Qwen4ExpTiny {
    type Config = Config;
    type Model = Qwen4Exp<Resident>;
    const NAME: &'static str = "qwen4exp";
    const DENSE_KINDS: &'static [GgmlDType] = &[
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q8_0,
        GgmlDType::Q4_0,
        GgmlDType::IQ4NL,
    ];
    const EXPERT_KINDS: &'static [GgmlDType] = &[
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
        GgmlDType::IQ4XS,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ2XXS,
        GgmlDType::Q2_0,
    ];

    fn template() -> Config {
        Config::tiny()
    }

    fn covers(template: &Config, cfg: &Config) -> bool {
        [LayerKind::Linear, LayerKind::Full]
            .iter()
            .all(|k| !template.layer_types.contains(k) || cfg.layer_types.contains(k))
    }

    fn tokenizer() -> TinyTokenizer {
        TinyTokenizer::qwen4exp()
    }

    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<TinySchema> {
        Ok(super::schema::tensor_schema(
            &ckpt.cfg,
            &super::schema::SchemaOptions::default(),
            &super::schema::Kinds {
                dense: ckpt.weights.dense,
                experts: ckpt.weights.experts,
            },
        ))
    }

    fn workloads(_cfg: &Config) -> Vec<Workload> {
        vec![
            // Decode only: the gemv arms, the one-query attention.
            Workload {
                prompt: 1,
                decode: 2,
            },
            // A short prefill: the f16-weight GEMM above the gemv's rows.
            Workload {
                prompt: 7,
                decode: 2,
            },
            // `LLOOM_MOE_MM_FROM` rows: the routed grouped GEMM.
            Workload {
                prompt: 64,
                decode: 1,
            },
        ]
    }

    fn build(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen4Exp<Resident>> {
        let bytes = gguf_bytes(ckpt, None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen4Exp::from_gguf(content, &mut r, dev)
    }

    fn write(ckpt: &TinyCheckpoint<Self>, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        let tok = Self::tokenizer();
        if ckpt.cfg.vocab_size < tok.vocab_size() {
            lloom_tensor::bail!(
                "a written checkpoint carries the tokenizer: vocab_size {} < its {} ids",
                ckpt.cfg.vocab_size,
                tok.vocab_size()
            )
        }
        let path = dir.join("tiny-qwen4exp.gguf");
        std::fs::write(&path, gguf_bytes(ckpt, Some(&tok))?)?;
        tok.write(dir)?;
        Ok(path)
    }

    fn load(path: &std::path::Path, dev: &Device) -> Result<Qwen4Exp<Resident>> {
        Qwen4Exp::from_gguf_shards(
            &[path.to_path_buf()],
            dev,
            &super::loader::LoadOptions::default(),
        )
    }

    /// The loader's f32 arm over the dequantized checkpoint: f32
    /// activations over the stems' stored values, where the served load's
    /// `TensorF16` stems would cast the activations to f16 on the CPU.
    fn reference(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen4Exp<Resident>> {
        let bytes = gguf_bytes(&ckpt.dequantized(), None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen4Exp::from_gguf_reference(content, &mut r, dev)
    }
}

/// The `qwen4exp.*` header keys for `cfg`, as `Config::from_gguf_metadata`
/// reads them; the end-of-text id from `tok` when the file carries one.
pub fn gguf_metadata(cfg: &Config, tok: Option<&TinyTokenizer>) -> Vec<(String, Value)> {
    let u = |v: usize| Value::U32(v as u32);
    let ints = |v: &[usize]| Value::Array(v.iter().map(|x| Value::I32(*x as i32)).collect());
    let ratios: Vec<usize> = cfg
        .layer_types
        .iter()
        .map(|k| match k {
            LayerKind::Full => cfg.indexer_compress_ratio,
            LayerKind::Linear => 0,
        })
        .collect();
    let n_rot = (cfg.head_dim as f64 * cfg.partial_rotary_factor).round() as usize;
    let eos = tok
        .and_then(|t| t.id("<|endoftext|>"))
        .unwrap_or(cfg.eos_token_id);
    let mut md: Vec<(String, Value)> = vec![
        (
            "general.architecture".into(),
            Value::String("qwen4exp".into()),
        ),
        ("general.name".into(), Value::String("tiny qwen4exp".into())),
        ("qwen4exp.block_count".into(), u(cfg.n_layers)),
        ("qwen4exp.embedding_length".into(), u(cfg.hidden_size)),
        (
            "qwen4exp.full_attention_interval".into(),
            u(cfg.full_attention_interval),
        ),
        ("qwen4exp.attention.compress_ratios".into(), ints(&ratios)),
        (
            "qwen4exp.rope.dimension_sections".into(),
            ints(&cfg.mrope_section),
        ),
        ("qwen4exp.rope.dimension_count".into(), u(n_rot)),
        (
            "qwen4exp.rope.freq_base".into(),
            Value::F32(cfg.rope_theta as f32),
        ),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon".into(),
            Value::F32(cfg.norm_eps as f32),
        ),
        ("qwen4exp.attention.head_count".into(), u(cfg.n_heads)),
        ("qwen4exp.attention.head_count_kv".into(), u(cfg.n_kv_heads)),
        ("qwen4exp.attention.key_length".into(), u(cfg.head_dim)),
        ("qwen4exp.attention.value_length".into(), u(cfg.head_dim)),
        (
            "qwen4exp.attention.indexer.head_count".into(),
            u(cfg.indexer_n_heads),
        ),
        (
            "qwen4exp.attention.indexer.key_length".into(),
            u(cfg.indexer_head_dim),
        ),
        (
            "qwen4exp.attention.indexer.top_k".into(),
            u(cfg.indexer_budget),
        ),
        ("qwen4exp.hyper_connection.count".into(), u(cfg.hc_count)),
        (
            "qwen4exp.hyper_connection.low_rank".into(),
            u(cfg.hc_lowrank),
        ),
        ("qwen4exp.ssm.state_size".into(), u(cfg.ssm_state_size)),
        ("qwen4exp.ssm.group_count".into(), u(cfg.ssm_n_group)),
        ("qwen4exp.ssm.time_step_rank".into(), u(cfg.ssm_dt_rank)),
        ("qwen4exp.ssm.conv_kernel".into(), u(cfg.ssm_conv_kernel)),
        ("qwen4exp.ssm.inner_size".into(), u(cfg.ssm_inner_size)),
        ("qwen4exp.expert_count".into(), u(cfg.n_experts)),
        (
            "qwen4exp.expert_used_count".into(),
            u(cfg.n_experts_per_tok),
        ),
        (
            "qwen4exp.expert_feed_forward_length".into(),
            u(cfg.moe_intermediate_size),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length".into(),
            u(cfg.shared_expert_intermediate_size),
        ),
        ("qwen4exp.ple.ngram_size".into(), u(cfg.ple_ngram_size)),
        (
            "qwen4exp.ple.heads_per_ngram".into(),
            u(cfg.ple_heads_per_ngram),
        ),
        ("qwen4exp.ple.conv_kernel".into(), u(cfg.ple_conv_kernel)),
        (
            "qwen4exp.embedding_length_per_layer_input".into(),
            u(cfg.ple_head_dim),
        ),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(eos)),
    ];
    if !cfg.ple_layers.is_empty() {
        md.push(("qwen4exp.ple.layers".into(), ints(&cfg.ple_layers)));
    }
    if let Some(id) = cfg.ple_image_token_id {
        md.push(("qwen4exp.ple.image_token_id".into(), Value::U32(id)));
    }
    md
}

/// The checkpoint as one GGUF file's bytes: the header keys, then every
/// tensor of the schema.
pub fn gguf_bytes(
    ckpt: &TinyCheckpoint<Qwen4ExpTiny>,
    tok: Option<&TinyTokenizer>,
) -> Result<Vec<u8>> {
    let schema = ckpt.schema()?;
    let tensors: Vec<(String, QTensor)> = schema
        .tensors
        .iter()
        .map(|t| Ok((t.name.clone(), ckpt.qtensor(t, &Device::Cpu)?)))
        .collect::<Result<_>>()?;
    let md = gguf_metadata(&ckpt.cfg, tok);
    let md_refs: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let t_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let mut out = std::io::Cursor::new(Vec::new());
    lloom_tensor::quantized::gguf_file::write(&mut out, &md_refs, &t_refs)?;
    Ok(out.into_inner())
}
