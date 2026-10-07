//! The Qwen3.5 GGUF loader: the `qwen35` / `qwen35moe` header into
//! [`Config`], every tensor validated against the schema before a byte
//! loads, the blocks assembled on the generic [`Decoder`] topology. The
//! shard reader is Qwen4-exp's (`qwen4exp::loader::Gguf`): one GGUF reader
//! in the crate, not a second.

use super::config::{Config, ARCH_DENSE, ARCH_MOE};
use super::{Qwen3_5, Qwen3_5Layer, Qwen3_5Mixer, Qwen3_5Moe, Qwen3_5MoeLayer};
use crate::models::qwen4exp::loader::{md_f32, md_u32, open_shards, working_set_of, Gguf, Shard};
use crate::models::qwen4exp::{GatedDeltaNet, LayerKind, MoeLayer, QsaAttention, Resident};
use crate::nn::blocks::SwiGlu;
use crate::nn::decoder::{Decoder, Hybrid, NormHead, PreNorm, StateCapacity, TokenEmbed};
use crate::nn::RmsNorm;
use crate::profile::Profile;
use lloom_tensor::quantized::gguf_file::{Content, Value};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Result};
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::PathBuf;

/// The fraction of the device's working set a resident load may hold
/// (`DESIGN.md` §9.5's admission; Qwen4-exp's default).
pub const DEFAULT_MEMORY_FRACTION: f64 = crate::models::qwen4exp::loader::DEFAULT_MEMORY_FRACTION;

/// What a load is asked for.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// The KV capacity in tokens every state built from the model takes
    /// (the full-attention layers' slab; it grows by this much when passed).
    pub kv_capacity: usize,
    /// The full-attention KV store; `None` is the checkpoint's f32.
    pub kv_precision: Option<crate::nn::ops::KvPrecision>,
    /// The admission fraction (`None` = [`DEFAULT_MEMORY_FRACTION`];
    /// `Some(0.0)` turns the check off).
    pub memory_fraction: Option<f64>,
    /// Keep every float matrix f32 (`QMatMul::Tensor`) instead of an f16
    /// stem: the CPU reference's precision, not a serving arm.
    pub dense_f32: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            kv_capacity: 4096,
            kv_precision: None,
            memory_fraction: None,
            dense_f32: false,
        }
    }
}

impl LoadOptions {
    pub fn fraction(&self) -> f64 {
        self.memory_fraction.unwrap_or(DEFAULT_MEMORY_FRACTION)
    }
}

impl Config {
    /// The config from a `qwen35` / `qwen35moe` GGUF header. The vocabulary
    /// is `token_embd.weight`'s row count, filled in by the loader.
    pub fn from_gguf_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.as_str())
            .unwrap_or("");
        let moe = match arch {
            ARCH_DENSE => false,
            ARCH_MOE => true,
            other => lloom_tensor::bail!(
                "general.architecture = {other:?}: not a Qwen3.5 checkpoint ({ARCH_DENSE} | {ARCH_MOE})"
            ),
        };
        let k = |s: &str| format!("{arch}.{s}");
        // `block_count` counts the MTP blocks the file keeps after the stack
        // (llama.cpp: `n_layer = n_layer_all - n_layer_nextn`).
        let n_layers_all = md_u32(md, &k("block_count"))?;
        let n_nextn_layers = md_u32(md, &k("nextn_predict_layers")).unwrap_or(0);
        if n_nextn_layers >= n_layers_all {
            lloom_tensor::bail!(
                "{arch}: nextn_predict_layers {n_nextn_layers} >= block_count {n_layers_all}"
            )
        }
        let n_layers = n_layers_all - n_nextn_layers;
        let interval = md_u32(md, &k("full_attention_interval")).unwrap_or(4);
        let head_dim = md_u32(md, &k("attention.key_length"))?;
        let n_rot = md_u32(md, &k("rope.dimension_count")).unwrap_or(head_dim / 4);
        let sections: [usize; 4] = match md.get(&k("rope.dimension_sections")) {
            Some(Value::Array(a)) if a.len() >= 3 => {
                let mut out = [0usize; 4];
                for (i, v) in a.iter().take(4).enumerate() {
                    out[i] = v
                        .to_i32()
                        .map(|x| x.max(0) as usize)
                        .or_else(|_| v.to_u32().map(|x| x as usize))?;
                }
                out
            }
            _ => [11, 11, 10, 0],
        };
        let ssm_state_size = md_u32(md, &k("ssm.state_size"))?;
        let ssm_dt_rank = md_u32(md, &k("ssm.time_step_rank"))?;
        let eos = md
            .get("tokenizer.ggml.eos_token_id")
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(248046);
        let bos = md
            .get("tokenizer.ggml.bos_token_id")
            .map(|v| v.to_u32())
            .transpose()?
            .unwrap_or(248044);
        let (n_experts, n_experts_per_tok, moe_ff, sh_ff) = if moe {
            (
                md_u32(md, &k("expert_count"))?,
                md_u32(md, &k("expert_used_count"))?,
                md_u32(md, &k("expert_feed_forward_length"))?,
                md_u32(md, &k("expert_shared_feed_forward_length"))?,
            )
        } else {
            (0, 0, 0, 0)
        };
        Ok(Config {
            n_layers,
            n_nextn_layers,
            hidden_size: md_u32(md, &k("embedding_length"))?,
            vocab_size: 0,
            intermediate_size: if moe {
                0
            } else {
                md_u32(md, &k("feed_forward_length"))?
            },
            norm_eps: md_f32(md, &k("attention.layer_norm_rms_epsilon"))?,
            full_attention_interval: interval,
            layer_types: Self::resolve_layer_types(n_layers, interval),
            n_heads: md_u32(md, &k("attention.head_count"))?,
            n_kv_heads: md_u32(md, &k("attention.head_count_kv"))?,
            head_dim,
            partial_rotary_factor: n_rot as f64 / head_dim as f64,
            rope_theta: md_f32(md, &k("rope.freq_base"))?,
            mrope_section: sections,
            ssm_state_size,
            ssm_n_group: md_u32(md, &k("ssm.group_count"))?,
            ssm_dt_rank,
            ssm_conv_kernel: md_u32(md, &k("ssm.conv_kernel"))?,
            ssm_inner_size: md_u32(md, &k("ssm.inner_size"))
                .unwrap_or(ssm_state_size * ssm_dt_rank),
            out_gate: crate::nn::gdn::OutGate::Silu,
            n_experts,
            n_experts_per_tok,
            moe_intermediate_size: moe_ff,
            shared_expert_intermediate_size: sh_ff,
            bos_token_id: bos,
            eos_token_id: eos,
            kv_precision: super::config::DEFAULT_KV_PRECISION,
        })
    }
}

/// The bytes a checkpoint's tensors hold, from the headers.
fn header_bytes<R: Read + Seek>(gg: &Gguf<'_, R>) -> usize {
    gg.shards
        .iter()
        .flat_map(|s| s.content.tensor_infos.values())
        .map(|ti| {
            let dt = ti.ggml_dtype;
            ti.shape.elem_count() / dt.block_size() * dt.type_size()
        })
        .sum()
}

/// Refuse a load whose weights alone exceed the device's budget (Metal
/// only; `fraction` 0 turns it off).
fn admit<R: Read + Seek>(gg: &Gguf<'_, R>, fraction: f64) -> Result<()> {
    let ws = working_set_of(&gg.device);
    if ws == 0 || fraction <= 0.0 {
        return Ok(());
    }
    let bytes = header_bytes(gg);
    let budget = (ws as f64 * fraction) as usize;
    if bytes > budget {
        lloom_tensor::bail!(
            "qwen3_5: the checkpoint's weights are {:.2} GB, over the budget of {:.2} GB ({fraction:.2} of the {:.2} GB working set); \
             a resident load would not fit (`--memory-fraction` raises the budget)",
            bytes as f64 / 1e9,
            budget as f64 / 1e9,
            ws as f64 / 1e9
        )
    }
    Ok(())
}

fn capacity(cfg: &Config, opts: &LoadOptions) -> StateCapacity {
    StateCapacity {
        kv_tokens: opts.kv_capacity.max(1),
        kv_precision: opts.kv_precision.unwrap_or(cfg.kv_precision),
    }
}

/// The mixer of layer `l`: the Gated DeltaNet or the gated attention, with
/// the input stems stacked into one f16 matrix where the loader can.
fn load_mixer<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    l: usize,
    kind: LayerKind,
    cfg: &Config,
) -> Result<Qwen3_5Mixer<P>> {
    let p = format!("blk.{l}");
    Ok(match kind {
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
            let block = GatedDeltaNet::with_geometry(
                wqkv,
                gate,
                gg.f32(&format!("{p}.ssm_conv1d.weight"))?,
                gg.f32(&format!("{p}.ssm_a"))?,
                gg.f32(&format!("{p}.ssm_dt.bias"))?,
                ssm_alpha,
                ssm_beta,
                gg.f32(&format!("{p}.ssm_norm.weight"))?,
                gg.qmat(&format!("{p}.ssm_out.weight"))?,
                &cfg.gdn(),
            );
            Hybrid::A(match (cat, ab_cat) {
                (Some(cat), _) => block.with_stem_cat(cat),
                (None, Some(ab)) => block.with_gate_stems_cat(ab),
                (None, None) => block,
            })
        }
        LayerKind::Full => {
            let (mut stems, cat) = gg.qmat_stack(&[
                format!("{p}.attn_q.weight"),
                format!("{p}.attn_k.weight"),
                format!("{p}.attn_v.weight"),
            ])?;
            let wv = stems.pop().expect("three stems");
            let wk = stems.pop().expect("three stems");
            let wq = stems.pop().expect("three stems");
            let attn = QsaAttention::with_geometry(
                wq,
                wk,
                wv,
                gg.qmat(&format!("{p}.attn_output.weight"))?,
                RmsNorm::new(gg.f32(&format!("{p}.attn_q_norm.weight"))?, cfg.norm_eps),
                RmsNorm::new(gg.f32(&format!("{p}.attn_k_norm.weight"))?, cfg.norm_eps),
                &cfg.attn(),
            );
            Hybrid::B(match cat {
                Some(cat) => attn.with_stem_cat(cat),
                None => attn,
            })
        }
    })
}

fn load_dense_ffn<R: Read + Seek>(gg: &mut Gguf<'_, R>, l: usize, cfg: &Config) -> Result<SwiGlu> {
    let p = format!("blk.{l}");
    Ok(SwiGlu::new(
        gg.qmat(&format!("{p}.ffn_gate.weight"))?,
        gg.qmat(&format!("{p}.ffn_up.weight"))?,
        gg.qmat(&format!("{p}.ffn_down.weight"))?,
        cfg.hidden_size,
        cfg.intermediate_size,
    ))
}

fn load_moe_ffn<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    l: usize,
    cfg: &Config,
) -> Result<MoeLayer<P, Resident>> {
    let p = format!("blk.{l}");
    let d = cfg.hidden_size;
    let experts = Resident::new(
        gg.packed(&format!("{p}.ffn_gate_exps.weight"))?,
        gg.packed(&format!("{p}.ffn_up_exps.weight"))?,
        gg.packed(&format!("{p}.ffn_down_exps.weight"))?,
    );
    let sh_router = gg
        .f32(&format!("{p}.ffn_gate_inp_shexp.weight"))?
        .reshape((1, d))?;
    Ok(MoeLayer::with_geometry(
        gg.qmat(&format!("{p}.ffn_gate_inp.weight"))?,
        experts,
        gg.qmat(&format!("{p}.ffn_gate_shexp.weight"))?,
        gg.qmat(&format!("{p}.ffn_up_shexp.weight"))?,
        gg.qmat(&format!("{p}.ffn_down_shexp.weight"))?,
        QMatMul::Tensor(sh_router),
        &cfg.moe(),
    ))
}

/// The config, the schema check, the embedding and the head — what the
/// dense and the MoE builds share.
fn open<R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    opts: &LoadOptions,
    want_moe: bool,
) -> Result<(Config, TokenEmbed, NormHead<RmsNorm>)> {
    let mut cfg = Config::from_gguf_metadata(gg.metadata())?;
    if cfg.is_moe() != want_moe {
        lloom_tensor::bail!(
            "qwen3_5: the checkpoint is {} and the loader asked for {}",
            cfg.arch(),
            if want_moe { ARCH_MOE } else { ARCH_DENSE }
        )
    }
    let embed_shape = gg.shape("token_embd.weight")?;
    if embed_shape.len() != 2 || embed_shape[1] != cfg.hidden_size {
        lloom_tensor::bail!(
            "token_embd.weight has shape {embed_shape:?}, expected [vocab, {}]",
            cfg.hidden_size
        )
    }
    cfg.vocab_size = embed_shape[0];
    let schema = super::schema::tensor_schema(&cfg, &Default::default());
    schema.validate(gg.table())?;
    admit(gg, opts.fraction())?;
    let embed = TokenEmbed::new(
        QMatMul::QTensor(gg.packed("token_embd.weight")?),
        cfg.hidden_size,
    );
    let head = NormHead::new(
        RmsNorm::new(gg.f32("output_norm.weight")?, cfg.norm_eps),
        gg.qmat("output.weight")?,
        cfg.vocab_size,
    );
    Ok((cfg, embed, head))
}

fn build_dense<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    opts: &LoadOptions,
) -> Result<Qwen3_5<P>> {
    let (cfg, embed, head) = open(gg, opts, false)?;
    let mut layers: Vec<Qwen3_5Layer<P>> = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let p = format!("blk.{l}");
        layers.push(PreNorm {
            norm_in: RmsNorm::new(gg.f32(&format!("{p}.attn_norm.weight"))?, cfg.norm_eps),
            mixer: load_mixer::<P, R>(gg, l, cfg.layer_types[l], &cfg)?,
            norm_mid: RmsNorm::new(
                gg.f32(&format!("{p}.post_attention_norm.weight"))?,
                cfg.norm_eps,
            ),
            ffn: load_dense_ffn(gg, l, &cfg)?,
        });
    }
    let cap = capacity(&cfg, opts);
    Decoder::new(ARCH_DENSE, cfg, embed, layers, head, cap)
}

fn build_moe<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    opts: &LoadOptions,
) -> Result<Qwen3_5Moe<P>> {
    let (cfg, embed, head) = open(gg, opts, true)?;
    let mut layers: Vec<Qwen3_5MoeLayer<P, Resident>> = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let p = format!("blk.{l}");
        layers.push(PreNorm {
            norm_in: RmsNorm::new(gg.f32(&format!("{p}.attn_norm.weight"))?, cfg.norm_eps),
            mixer: load_mixer::<P, R>(gg, l, cfg.layer_types[l], &cfg)?,
            norm_mid: RmsNorm::new(
                gg.f32(&format!("{p}.post_attention_norm.weight"))?,
                cfg.norm_eps,
            ),
            ffn: load_moe_ffn::<P, R>(gg, l, &cfg)?,
        });
    }
    let cap = capacity(&cfg, opts);
    Decoder::new(ARCH_MOE, cfg, embed, layers, head, cap)
}

/// The architecture a GGUF's header names, when it is this family's.
pub fn arch_of(content: &Content) -> Option<&'static str> {
    match content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.as_str())
    {
        Some(ARCH_DENSE) => Some(ARCH_DENSE),
        Some(ARCH_MOE) => Some(ARCH_MOE),
        _ => None,
    }
}

impl<P: Profile> Qwen3_5<P> {
    /// The dense model from one open GGUF.
    pub fn from_gguf<R: Read + Seek>(
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
        gg.dense_f32 = opts.dense_f32;
        build_dense::<P, R>(&mut gg, opts)
    }

    /// The dense model from a checkpoint's shards.
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
        gg.dense_f32 = opts.dense_f32;
        build_dense::<P, _>(&mut gg, opts)
    }
}

impl<P: Profile> Qwen3_5Moe<P> {
    /// The MoE model from one open GGUF, every expert resident.
    pub fn from_gguf<R: Read + Seek>(
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
        gg.dense_f32 = opts.dense_f32;
        build_moe::<P, R>(&mut gg, opts)
    }

    /// The MoE model from a checkpoint's shards, every expert resident.
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
        gg.dense_f32 = opts.dense_f32;
        build_moe::<P, _>(&mut gg, opts)
    }
}
