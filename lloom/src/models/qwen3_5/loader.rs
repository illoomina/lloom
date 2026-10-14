//! The Qwen3.5 GGUF loader: the `qwen35` / `qwen35moe` header into
//! [`Config`], every tensor validated against the schema before a byte
//! loads, the blocks assembled on the generic [`Decoder`] topology. The
//! shard reader is Qwen4-exp's (`qwen4exp::loader::Gguf`): one GGUF reader
//! in the crate, not a second.

use super::config::{Config, ARCH_CLEF, ARCH_DENSE, ARCH_MOE};
use super::{
    Qwen3_5, Qwen3_5Layer, Qwen3_5Mixer, Qwen3_5Moe, Qwen3_5MoeLayer, Qwen3_5MoeMtpHead,
    Qwen3_5MtpHead,
};
use crate::models::qwen4exp::loader::{md_f32, md_u32, open_shards, working_set_of, Gguf, Shard};
use crate::models::qwen4exp::{GatedDeltaNet, LayerKind, MoeLayer, QsaAttention, Resident};
use crate::nn::blocks::SwiGlu;
use crate::nn::decoder::{Decoder, Hybrid, NormHead, PreNorm, StateCapacity, TokenEmbed};
use crate::nn::mtp::NextnHead;
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
    /// Load the MTP block (`blk.{n_layers}` + its `nextn.*`) as a
    /// [`NextnHead`](crate::nn::mtp::NextnHead): from the main file when
    /// its header keeps one (`nextn_predict_layers`), else from the sidecar
    /// the `*_with_mtp` constructors take. Refused when neither carries it.
    pub mtp: bool,
    /// The longest verify window (positions, `k + 1`) a state built at the
    /// `Ring` profile runs as one pass over phase rings
    /// (`StateCapacity::window`); 0 runs a window as sequential steps.
    pub window: usize,
    /// The machine gate the load passes before its first tensor
    /// (`memory.md` §9.10); `None` loads unadmitted (a test, a harness).
    pub machine: Option<crate::nn::memory::MachineLoad>,
    /// Turn checkpoints a state keeps (`StateCapacity::checkpoints`): the
    /// recurrent layers' `S` and conv windows at the last `checkpoints`
    /// multi-token passes, so a request re-rendering the previous turn
    /// restores there instead of prefilling the conversation again
    /// (`sequence-reuse-design.md` §2.3b). Admitted beside the weights:
    /// as many as the budget fits, at most this.
    pub checkpoints: usize,
    /// Refuse at load, on Metal, a full-attention layer the fused paths do
    /// not serve, rather than keep the composed attention: what serving sets.
    /// Off, such a layer attends composed, the reference a test or a harness
    /// compares against.
    pub serve: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            kv_capacity: 4096,
            kv_precision: None,
            memory_fraction: None,
            dense_f32: false,
            mtp: false,
            window: 0,
            machine: None,
            checkpoints: 2,
            serve: false,
        }
    }
}

impl LoadOptions {
    pub fn fraction(&self) -> f64 {
        self.memory_fraction.unwrap_or(DEFAULT_MEMORY_FRACTION)
    }
}

/// Reuse an already admitted load without acquiring another legacy Holding.
/// Only the machine gate is removed from the copied options; the family's
/// schema, fraction/working-set checks, MTP and cache geometry stay unchanged.
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
            ARCH_DENSE | ARCH_CLEF => false,
            ARCH_MOE => true,
            other => lloom_tensor::bail!(
                "general.architecture = {other:?}: not a Qwen3.5 checkpoint ({ARCH_DENSE} | {ARCH_MOE} | {ARCH_CLEF})"
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

/// The turn checkpoints admitted beside the weights: `want`, or as many
/// frames of `frame_bytes` as the budget leaves after the weights (0 when
/// none fits; the engine says so). Unchecked where the budget is (Metal
/// only; `fraction` 0 turns it off).
fn admit_checkpoints<R: Read + Seek>(
    gg: &Gguf<'_, R>,
    fraction: f64,
    frame_bytes: usize,
    want: usize,
) -> usize {
    let ws = working_set_of(&gg.device);
    if ws == 0 || fraction <= 0.0 || frame_bytes == 0 {
        return want;
    }
    let budget = (ws as f64 * fraction) as usize;
    let free = budget.saturating_sub(header_bytes(gg));
    want.min(free / frame_bytes)
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

/// The share of a prefill pass's tensors live at once at its peak, as a
/// fraction: the pool hands a buffer back only when the command buffer that
/// last read it completes, so a pass holds nearly all it allocates
/// (`memory.md` §9.10).
pub const PREFILL_LIVE_SHARE: (usize, usize) = (91, 100);

/// The bytes a prefill pass allocates per position, over every layer, from
/// the blocks' own launches — the Gated DeltaNet's projections and chunked
/// operands, the attention's fused path, the feed-forward (dense or routed,
/// with its shared expert), the norms and residuals, at f32 with a 16-bit
/// stem's output counted beside its f32 copy — times
/// [`PREFILL_LIVE_SHARE`]. Linear in the positions: a pass's working set is
/// its positions' rows through every layer (`memory.md` §9.10).
pub fn prefill_bytes_per_position(cfg: &Config) -> usize {
    const CHUNK: usize = crate::models::qwen4exp::deltanet::CHUNK_SIZE;
    let h = cfg.hidden_size;
    let d = cfg.deltanet_dims();
    let (v, c, hv) = (d.d_v * d.h_v, cfg.conv_channels(), d.h_v);
    // The projections and their 16-bit stems; the gate; the q/k/v pack;
    // the decays; the scaled operands; the WY solve; the scan's output;
    // the gated norm; the out-projection.
    let gdn = (c + v + 2 * hv)
        + (c + v) / 2
        + 2 * hv
        + 3 * v
        + (2 * hv * CHUNK + hv)
        + 5 * v
        + 3 * hv * CHUNK
        + 2 * v
        + v
        + v
        + h;
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    // q|gate and k, v with their 16-bit copies; the prepared rows; the
    // attention's output, its contiguous copy and the gated product; the
    // out-projection.
    let attn = 3 * nh * hd + 3 * nkv * hd + 2 * (nh + nkv) * hd + 3 * nh * hd + h;
    let ffn = if cfg.is_moe() {
        let (k, mff, sff) = (
            cfg.n_experts_per_tok,
            cfg.moe_intermediate_size,
            cfg.shared_expert_intermediate_size,
        );
        cfg.n_experts + k * (3 * mff + h) + 4 * sff + 3 * h
    } else {
        4 * cfg.intermediate_size + h
    };
    let norms = 4 * h;
    let elems: usize = cfg
        .layer_types
        .iter()
        .map(|k| match k {
            LayerKind::Full => attn,
            LayerKind::Linear => gdn,
        } + ffn
            + norms)
        .sum();
    elems * std::mem::size_of::<f32>() * PREFILL_LIVE_SHARE.0 / PREFILL_LIVE_SHARE.1
}

/// What a load holds at its peak before anything grows past its floors —
/// the machine gate's figure (`memory.md` §9.10): the weights from the
/// headers; the KV every full-attention layer allocates at its first append
/// (`kv_capacity` positions, the MTP block's layer among them when loaded);
/// every linear layer's recurrent state over `phases` buffers a channel; one
/// prefill pass's working set at the tile (the whole capacity when the
/// prompt is one forward, `prefill_tile` 0); and, when pass plans serve the
/// passes, the base a planned pass holds, another working set. A pass past
/// these is admitted when it comes (`Decoder::prepare_pass`).
pub fn machine_peak(
    cfg: &Config,
    weights: usize,
    opts: &LoadOptions,
    prefill_tile: usize,
    phases: usize,
    pass_plans: bool,
) -> lloom_tensor::machine::Peak {
    let full_stack = cfg
        .layer_types
        .iter()
        .filter(|k| **k == LayerKind::Full)
        .count();
    let full = full_stack + usize::from(opts.mtp);
    let linear = cfg.n_layers - full_stack;
    let precision = opts.kv_precision.unwrap_or(cfg.kv_precision);
    let per_token =
        precision.bytes_per_token(cfg.n_kv_heads, cfg.head_dim, lloom_tensor::DType::F32);
    let kv = full * opts.kv_capacity.max(1) * per_token;
    let d = cfg.deltanet_dims();
    let state = d.d_k * d.d_v * d.h_v + cfg.conv_channels() * cfg.ssm_conv_kernel;
    let recurrent = linear * phases * state * std::mem::size_of::<f32>();
    let positions = match prefill_tile {
        0 => opts.kv_capacity.max(1),
        t => t,
    };
    let pass = positions * prefill_bytes_per_position(cfg);
    let mut peak = lloom_tensor::machine::Peak::new()
        .with("weights", weights as u64)
        .with(
            format!(
                "KV floor ({full} layers x {} positions)",
                opts.kv_capacity.max(1)
            ),
            kv as u64,
        )
        .with(
            format!("recurrent state ({linear} layers)"),
            recurrent as u64,
        )
        .with(
            format!("prefill working set ({positions}-position pass)"),
            pass as u64,
        );
    if pass_plans {
        peak.push("pass-plan base (one pass)", pass as u64);
    }
    peak
}

fn capacity(cfg: &Config, opts: &LoadOptions) -> StateCapacity {
    StateCapacity {
        kv_tokens: opts.kv_capacity.max(1),
        kv_precision: opts.kv_precision.unwrap_or(cfg.kv_precision),
        window: opts.window,
        checkpoints: opts.checkpoints,
    }
}

/// [`capacity`] with the turn checkpoints the budget admits beside the
/// weights, from the built layers' frame size.
fn capacity_admitted<R: Read + Seek, L: crate::nn::decoder::Layer>(
    gg: &Gguf<'_, R>,
    cfg: &Config,
    opts: &LoadOptions,
    layers: &[L],
) -> StateCapacity {
    let mut cap = capacity(cfg, opts);
    let frame = layers
        .iter()
        .map(|l| l.checkpoint_elems(&cap))
        .sum::<usize>()
        * lloom_tensor::DType::F32.size_in_bytes();
    cap.checkpoints = admit_checkpoints(gg, opts.fraction(), frame, cap.checkpoints);
    cap
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
            if gg.serve && gg.device.is_metal() {
                attn.prepare_for_serving(&gg.device)?;
            }
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
    phases: usize,
    preflight: impl FnOnce(&Config) -> Result<()>,
) -> Result<(Config, TokenEmbed, NormHead<RmsNorm>, Holds)> {
    let mut cfg = Config::from_gguf_metadata(gg.metadata())?;
    // A sidecar's MTP block (bartowski's `mtp-*.gguf`): the main header
    // says nothing of it, the table carries `blk.{n_layers}.nextn.*`.
    if cfg.n_nextn_layers == 0
        && gg
            .shape(&format!("blk.{}.nextn.eh_proj.weight", cfg.n_layers))
            .is_ok()
    {
        cfg.n_nextn_layers = 1;
    }
    if opts.mtp && cfg.n_nextn_layers == 0 {
        lloom_tensor::bail!(
            "qwen3_5: the checkpoint carries no MTP block (`nextn_predict_layers` 0 and no \
             `blk.{}.nextn.*` in the shards); the MTP needs the block in the main file or a \
             sidecar GGUF",
            cfg.n_layers
        )
    }
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
    // Same resolved schema/config, before the first weight/table producer.
    preflight(&cfg)?;
    admit(gg, opts.fraction())?;
    let reservation = match &opts.machine {
        Some(m) => {
            let plans = pass_plans_on(&gg.device)?;
            let peak = machine_peak(&cfg, header_bytes(gg), opts, m.prefill_tile, phases, plans);
            let per_tile_position = prefill_bytes_per_position(&cfg) * (1 + usize::from(plans));
            match m.admit(&gg.device, peak.clone()) {
                Ok(a) => a.holding,
                Err(e) => {
                    // The tile is the lever a refusal can solve for: the
                    // largest prefill tile whose passes fit the room.
                    let pool = crate::nn::memory::pool_bound_bytes(&gg.device) as i128;
                    let tile = match m.prefill_tile {
                        0 => opts.kv_capacity.max(1),
                        t => t,
                    };
                    let fixed = peak.total() as i128 + pool - (tile * per_tile_position) as i128;
                    let fits = m
                        .gate_for(&gg.device)
                        .room()
                        .map(|room| (room - fixed) / per_tile_position as i128);
                    match fits {
                        Some(n) if n >= 1 && (n as usize) < tile => lloom_tensor::bail!(
                            "{e}\n  A prefill tile of {n} positions fits the room: --prefill-tile {n} \
                             ({:.1} MB a position of working set{}).",
                            per_tile_position as f64 / 1e6,
                            if plans { ", its pass-plan base included" } else { "" }
                        ),
                        _ => return Err(e),
                    }
                }
            }
        }
        None => None,
    };
    let keep = if opts.mtp {
        mtp_tenant(gg, cfg.n_layers)
    } else {
        None
    };
    let embed = TokenEmbed::new(
        QMatMul::QTensor(gg.packed("token_embd.weight")?),
        cfg.hidden_size,
    );
    let head = NormHead::new(
        RmsNorm::new(gg.f32("output_norm.weight")?, cfg.norm_eps),
        gg.qmat("output.weight")?,
        cfg.vocab_size,
    );
    Ok((
        cfg,
        embed,
        head,
        Holds {
            reservation,
            keep: keep.into_iter().collect(),
        },
    ))
}

/// What a load registered that its model keeps for its life: the machine
/// reservation and the ledger rows of what the model owns.
struct Holds {
    reservation: Option<std::sync::Arc<lloom_tensor::machine::Holding>>,
    keep: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

/// Register the MTP block's weights (`blk.{layer}.*`, from the main file or
/// the sidecar) as a ledger row for the process's life: a capability the
/// caller chose with `speculative.method = mtp`. The row is the returned
/// handle's: the model keeps it.
fn mtp_tenant<R: Read + Seek>(
    gg: &Gguf<'_, R>,
    layer: usize,
) -> Option<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &gg.device {
        use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
        let prefix = format!("blk.{layer}.");
        let bytes: usize = gg
            .shards
            .iter()
            .flat_map(|s| s.content.tensor_infos.iter())
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(_, ti)| {
                ti.shape.elem_count() / ti.ggml_dtype.block_size() * ti.ggml_dtype.type_size()
            })
            .sum();
        let g = Gauge::new(
            "qwen3_5-mtp",
            Claim {
                floor: bytes,
                want: bytes,
                unit: bytes.max(1),
                class: Class::Capability,
            },
            bytes,
        );
        let t: std::sync::Arc<dyn Tenant> = g.clone();
        m.ledger().register(&t);
        return Some(g);
    }
    let _ = (gg, layer);
    None
}

/// Whether recorded pass plans serve the passes on `device` (each planned
/// pass shape then holds a base the size of its working set).
fn pass_plans_on(device: &Device) -> Result<bool> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Ok(m.execution_device()?.pass_plans_enabled());
    }
    let _ = device;
    Ok(false)
}

/// Buffers a recurrent channel takes at profile `P`: the ring's phases when
/// the state is pooled (`Ring` with a window), else the slab and its spare.
fn state_phases<P: Profile>(opts: &LoadOptions) -> usize {
    if <P::Rewind as crate::profile::Rewind>::REWINDABLE && opts.window > 0 {
        opts.window + 1
    } else {
        2
    }
}

/// The MTP block's attention layer, `blk.{n_layers}`: the full-attention
/// mixer of the family's shape (llama.cpp `load_block_mtp`).
fn load_mtp_attention<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    l: usize,
    cfg: &Config,
) -> Result<QsaAttention<P>> {
    match load_mixer::<P, R>(gg, l, LayerKind::Full, cfg)? {
        Hybrid::B(attn) => Ok(attn),
        Hybrid::A(_) => lloom_tensor::bail!("the MTP block is a full-attention layer"),
    }
}

/// The `nextn.*` projections around the MTP block's layer.
fn load_nextn<L: crate::nn::decoder::Layer, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    l: usize,
    layer: L,
    cfg: &Config,
) -> Result<NextnHead<L>> {
    let p = format!("blk.{l}.nextn");
    Ok(NextnHead::new(
        RmsNorm::new(gg.f32(&format!("{p}.enorm.weight"))?, cfg.norm_eps),
        RmsNorm::new(gg.f32(&format!("{p}.hnorm.weight"))?, cfg.norm_eps),
        gg.qmat(&format!("{p}.eh_proj.weight"))?,
        layer,
        RmsNorm::new(
            gg.f32(&format!("{p}.shared_head_norm.weight"))?,
            cfg.norm_eps,
        ),
        cfg.hidden_size,
    ))
}

fn build_dense<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    opts: &LoadOptions,
    preflight: impl FnOnce(&Config) -> Result<()>,
) -> Result<(Qwen3_5<P>, Option<Qwen3_5MtpHead<P>>)> {
    let (cfg, embed, head, holds) = open(gg, opts, false, state_phases::<P>(opts), preflight)?;
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
    let mtp = if opts.mtp {
        let l = cfg.n_layers;
        let p = format!("blk.{l}");
        let layer = PreNorm {
            norm_in: RmsNorm::new(gg.f32(&format!("{p}.attn_norm.weight"))?, cfg.norm_eps),
            mixer: load_mtp_attention::<P, R>(gg, l, &cfg)?,
            norm_mid: RmsNorm::new(
                gg.f32(&format!("{p}.post_attention_norm.weight"))?,
                cfg.norm_eps,
            ),
            ffn: load_dense_ffn(gg, l, &cfg)?,
        };
        Some(load_nextn(gg, l, layer, &cfg)?)
    } else {
        None
    };
    let cap = capacity_admitted(gg, &cfg, opts, &layers);
    let hidden = cfg.hidden_size;
    let device = gg.device.clone();
    let per_position = prefill_bytes_per_position(&cfg);
    let mut model = Decoder::new(ARCH_DENSE, cfg, embed, layers, head, cap, hidden, &device)?;
    model.set_prefill_bytes_per_position(per_position);
    model.hold_reservation(holds.reservation);
    for k in holds.keep {
        model.keep_alive(k);
    }
    Ok((model, mtp))
}

fn build_moe<P: Profile, R: Read + Seek>(
    gg: &mut Gguf<'_, R>,
    opts: &LoadOptions,
    preflight: impl FnOnce(&Config) -> Result<()>,
) -> Result<(Qwen3_5Moe<P>, Option<Qwen3_5MoeMtpHead<P>>)> {
    let (cfg, embed, head, holds) = open(gg, opts, true, state_phases::<P>(opts), preflight)?;
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
    let mtp = if opts.mtp {
        let l = cfg.n_layers;
        let p = format!("blk.{l}");
        let layer = PreNorm {
            norm_in: RmsNorm::new(gg.f32(&format!("{p}.attn_norm.weight"))?, cfg.norm_eps),
            mixer: load_mtp_attention::<P, R>(gg, l, &cfg)?,
            norm_mid: RmsNorm::new(
                gg.f32(&format!("{p}.post_attention_norm.weight"))?,
                cfg.norm_eps,
            ),
            ffn: load_moe_ffn::<P, R>(gg, l, &cfg)?,
        };
        Some(load_nextn(gg, l, layer, &cfg)?)
    } else {
        None
    };
    let cap = capacity_admitted(gg, &cfg, opts, &layers);
    let hidden = cfg.hidden_size;
    let device = gg.device.clone();
    let per_position = prefill_bytes_per_position(&cfg);
    let mut model = Decoder::new(ARCH_MOE, cfg, embed, layers, head, cap, hidden, &device)?;
    model.set_prefill_bytes_per_position(per_position);
    model.hold_reservation(holds.reservation);
    for k in holds.keep {
        model.keep_alive(k);
    }
    Ok((model, mtp))
}

/// The shards and, after them, an MTP sidecar GGUF (bartowski's
/// `mtp-*.gguf`) with the tensors it duplicates from the main file
/// (`token_embd`, `output`, `output_norm`) dropped: the head reads the
/// target's, as the reference does without `nextn.embed_tokens` /
/// `shared_head_head` in the block.
fn open_shards_with_sidecar(
    paths: &[PathBuf],
    sidecar: Option<&PathBuf>,
) -> Result<(Vec<Content>, Vec<std::fs::File>, Vec<PathBuf>)> {
    let mut all: Vec<PathBuf> = paths.to_vec();
    if let Some(s) = sidecar {
        all.push(s.clone());
    }
    let (mut contents, files) = open_shards(&all)?;
    if sidecar.is_some() {
        let main: std::collections::HashSet<String> = contents[..paths.len()]
            .iter()
            .flat_map(|c| c.tensor_infos.keys().cloned())
            .collect();
        let last = contents.len() - 1;
        contents[last]
            .tensor_infos
            .retain(|name, _| !main.contains(name));
    }
    Ok((contents, files, all))
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
        Some(ARCH_CLEF) => Some(ARCH_CLEF),
        _ => None,
    }
}

impl<P: Profile> Qwen3_5<P> {
    /// The dense model and optional MTP head in one caller-supplied domain.
    /// Header readers, schema checks and all eager weight/table construction run
    /// inside the existing load scope. No second machine reservation is made.
    ///
    /// Source integration only: complete host/native/worker producer enrollment
    /// and checked caller load prediction precede engine activation. Existing
    /// CPU/CUDA and legacy constructors remain unchanged. Request execution must
    /// enter the returned owner's pass scope, including fresh cache creation.
    pub fn from_gguf_shards_with_mtp_admitted(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
        load: crate::nn::memory::LoadContext,
    ) -> Result<crate::nn::memory::Loaded<(Self, Option<Qwen3_5MtpHead<P>>)>> {
        in_admitted_load(device, load, opts, |resolved, scoped| {
            Self::from_gguf_shards_with_mtp(paths, sidecar, resolved, scoped)
        })
    }

    /// The dense model from one open GGUF (`opts.mtp` off).
    pub fn from_gguf<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        no_mtp(opts)?;
        Self::from_gguf_with_mtp(content, reader, device, opts).map(|(m, _)| m)
    }

    /// The dense model from one open GGUF, with its MTP head when
    /// `opts.mtp` and the file keeps the block.
    pub fn from_gguf_with_mtp<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<(Self, Option<Qwen3_5MtpHead<P>>)> {
        let mut gg = Gguf::new(
            vec![Shard {
                content,
                reader,
                path: None,
            }],
            device,
        )?;
        gg.dense_f32 = opts.dense_f32;
        gg.serve = opts.serve;
        build_dense::<P, R>(&mut gg, opts, |_| Ok(()))
    }

    /// The dense model from a checkpoint's shards (`opts.mtp` off).
    pub fn from_gguf_shards(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        no_mtp(opts)?;
        Self::from_gguf_shards_with_mtp(paths, None, device, opts).map(|(m, _)| m)
    }

    /// The dense model from a checkpoint's shards, with its MTP head when
    /// `opts.mtp`: from the main file's block, else from `sidecar`.
    pub fn from_gguf_shards_with_mtp(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<(Self, Option<Qwen3_5MtpHead<P>>)> {
        Self::from_gguf_shards_with_mtp_preflight(paths, sidecar, device, opts, |_| Ok(()))
    }

    /// Family metadata/policy validation against the same resolved weight
    /// schema before weight/table materialization. No second schema, domain or
    /// tokenizer/model is created; the caller owns host metadata enrollment.
    pub fn from_gguf_shards_with_mtp_preflight(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
        preflight: impl FnOnce(&Config) -> Result<()>,
    ) -> Result<(Self, Option<Qwen3_5MtpHead<P>>)> {
        let (contents, mut files, all) = open_shards_with_sidecar(paths, sidecar)?;
        let shards = contents
            .into_iter()
            .zip(files.iter_mut())
            .zip(&all)
            .map(|((content, reader), path)| Shard {
                content,
                reader,
                path: Some(path.clone()),
            })
            .collect();
        let mut gg = Gguf::new(shards, device)?;
        gg.dense_f32 = opts.dense_f32;
        gg.serve = opts.serve;
        build_dense::<P, _>(&mut gg, opts, preflight)
    }
}

impl<P: Profile> Qwen3_5Moe<P> {
    /// The MoE model and optional MTP head in one caller-supplied domain.
    /// Header readers, schema checks and all eager weight/table construction run
    /// inside the existing load scope. No second machine reservation is made.
    ///
    /// Source integration only: complete host/native/worker producer enrollment
    /// and checked caller load prediction precede engine activation. Existing
    /// CPU/CUDA and legacy constructors remain unchanged. Request execution must
    /// enter the returned owner's pass scope, including fresh cache creation.
    pub fn from_gguf_shards_with_mtp_admitted(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
        load: crate::nn::memory::LoadContext,
    ) -> Result<crate::nn::memory::Loaded<(Self, Option<Qwen3_5MoeMtpHead<P>>)>> {
        in_admitted_load(device, load, opts, |resolved, scoped| {
            Self::from_gguf_shards_with_mtp(paths, sidecar, resolved, scoped)
        })
    }

    /// The MoE model from one open GGUF, every expert resident (`opts.mtp`
    /// off).
    pub fn from_gguf<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        no_mtp(opts)?;
        Self::from_gguf_with_mtp(content, reader, device, opts).map(|(m, _)| m)
    }

    /// The MoE model from one open GGUF, with its MTP head when `opts.mtp`
    /// and the file keeps the block.
    pub fn from_gguf_with_mtp<R: Read + Seek>(
        content: Content,
        reader: &mut R,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<(Self, Option<Qwen3_5MoeMtpHead<P>>)> {
        let mut gg = Gguf::new(
            vec![Shard {
                content,
                reader,
                path: None,
            }],
            device,
        )?;
        gg.dense_f32 = opts.dense_f32;
        gg.serve = opts.serve;
        build_moe::<P, R>(&mut gg, opts, |_| Ok(()))
    }

    /// The MoE model from a checkpoint's shards, every expert resident
    /// (`opts.mtp` off).
    pub fn from_gguf_shards(
        paths: &[PathBuf],
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        no_mtp(opts)?;
        Self::from_gguf_shards_with_mtp(paths, None, device, opts).map(|(m, _)| m)
    }

    /// The MoE model from a checkpoint's shards, with its MTP head when
    /// `opts.mtp`: from the main file's block, else from `sidecar`.
    pub fn from_gguf_shards_with_mtp(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<(Self, Option<Qwen3_5MoeMtpHead<P>>)> {
        Self::from_gguf_shards_with_mtp_preflight(paths, sidecar, device, opts, |_| Ok(()))
    }

    /// Family metadata/policy validation against the same resolved weight
    /// schema before weight/table materialization. No second schema, domain or
    /// tokenizer/model is created; the caller owns host metadata enrollment.
    pub fn from_gguf_shards_with_mtp_preflight(
        paths: &[PathBuf],
        sidecar: Option<&PathBuf>,
        device: &Device,
        opts: &LoadOptions,
        preflight: impl FnOnce(&Config) -> Result<()>,
    ) -> Result<(Self, Option<Qwen3_5MoeMtpHead<P>>)> {
        let (contents, mut files, all) = open_shards_with_sidecar(paths, sidecar)?;
        let shards = contents
            .into_iter()
            .zip(files.iter_mut())
            .zip(&all)
            .map(|((content, reader), path)| Shard {
                content,
                reader,
                path: Some(path.clone()),
            })
            .collect();
        let mut gg = Gguf::new(shards, device)?;
        gg.dense_f32 = opts.dense_f32;
        gg.serve = opts.serve;
        build_moe::<P, _>(&mut gg, opts, preflight)
    }
}

/// The constructors that return the model alone take no MTP request.
fn no_mtp(opts: &LoadOptions) -> Result<()> {
    if opts.mtp {
        lloom_tensor::bail!("qwen3_5: `LoadOptions::mtp` wants a `*_with_mtp` constructor")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-position figure at the two real geometries: the 27B's pass
    /// tensors sum to 51.0 MB a position and the 35B-A3B's to 18.3, of which
    /// the live share is what a pass holds at its peak (`memory.md` §9.10:
    /// 46.4 MB a position over a 267-position pass on the 27B).
    #[test]
    fn the_prefill_working_set_comes_from_the_geometry() {
        let mb = |b: usize| b as f64 / 1e6;
        let dense = prefill_bytes_per_position(&Config::qwen3_6_27b());
        let moe = prefill_bytes_per_position(&Config::qwen3_6_35b_a3b());
        assert!((mb(dense) - 46.44).abs() < 0.05, "27B: {:.2} MB", mb(dense));
        assert!((mb(moe) - 16.70).abs() < 0.05, "35B-A3B: {:.2} MB", mb(moe));
        // Twice the positions, twice the bytes; a tiny geometry, a tiny figure.
        assert!(prefill_bytes_per_position(&Config::tiny()) < 100_000);
    }

    /// The peak carries one pass at the tile and, with pass plans, its base;
    /// one forward over the prompt (`prefill_tile` 0) is a pass of the whole
    /// capacity.
    #[test]
    fn the_peak_carries_one_pass_and_the_plans_base() {
        let cfg = Config::qwen3_6_27b();
        let opts = LoadOptions::default();
        let row = |p: &lloom_tensor::machine::Peak, name: &str| {
            p.rows()
                .iter()
                .find(|(n, _)| n.starts_with(name))
                .map(|(_, b)| *b as usize)
        };
        let per = prefill_bytes_per_position(&cfg);
        let planned = machine_peak(&cfg, 1, &opts, 512, 2, true);
        assert_eq!(row(&planned, "prefill working set"), Some(512 * per));
        assert_eq!(row(&planned, "pass-plan base"), Some(512 * per));
        let pooled = machine_peak(&cfg, 1, &opts, 512, 2, false);
        assert_eq!(row(&pooled, "pass-plan base"), None);
        let one_shot = machine_peak(&cfg, 1, &opts, 0, 2, false);
        assert_eq!(
            row(&one_shot, "prefill working set"),
            Some(opts.kv_capacity * per)
        );
    }
}
