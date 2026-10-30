//! One LFM2 implementation over both containers: the same weights written as
//! safetensors and as a dense GGUF load bit-identically, and a packed GGUF
//! runs on its blocks and agrees with the legacy GGUF model
//! (`quantized_lfm2`) on the same file.
use super::*;
use crate::models::lfm2::{Model, State};
use crate::models::quantized_lfm2;
use crate::nn::CausalLM;
use crate::profile::CrownExact;
use lloom_tensor::quantized::{gguf_file, GgmlDType, QTensor};
use lloom_tensor::{Device, Tensor};
use std::io::Cursor;

const VOCAB: usize = 64;
const HIDDEN: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const FFN: usize = 512;
const L_CACHE: usize = 3;
const CONTEXT: usize = 64;
const KINDS: [LayerType; 3] = [LayerType::Conv, LayerType::FullAttention, LayerType::Conv];

fn config() -> Config {
    Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: FFN,
        num_hidden_layers: KINDS.len(),
        num_attention_heads: HEADS,
        num_key_value_heads: KV_HEADS,
        norm_eps: 1e-5,
        rope_theta: 1e6,
        max_position_embeddings: CONTEXT,
        conv_l_cache: L_CACHE,
        layer_types: KINDS.to_vec(),
        tie_embedding: true,
        ..Config::default()
    }
}

/// Uniform in `[-g, g]` with `g = 1/√fan_in`, keyed by `(seed, i)`.
fn values(seed: u64, n: usize, fan_in: usize) -> Vec<f32> {
    let g = 1.0 / (fan_in as f32).sqrt();
    (0..n as u64)
        .map(|i| {
            let mut z =
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * g
        })
        .collect()
}

/// One tensor in the HF checkpoint's naming; `packed` marks a projection or
/// the table (stored at the GGUF's packed kind), `norm` a norm's weight.
struct Spec {
    hf: String,
    shape: Vec<usize>,
    packed: bool,
    norm: bool,
}

fn specs() -> Vec<Spec> {
    let mut out = Vec::new();
    let mut add = |hf: String, shape: &[usize], packed, norm| {
        out.push(Spec {
            hf,
            shape: shape.to_vec(),
            packed,
            norm,
        })
    };
    let (h, kv, hd) = (HIDDEN, KV_HEADS * HIDDEN / HEADS, HIDDEN / HEADS);
    add("model.embed_tokens.weight".into(), &[VOCAB, h], true, false);
    add("model.embedding_norm.weight".into(), &[h], false, true);
    for (l, kind) in KINDS.iter().enumerate() {
        let p = format!("model.layers.{l}");
        add(format!("{p}.operator_norm.weight"), &[h], false, true);
        add(format!("{p}.ffn_norm.weight"), &[h], false, true);
        add(
            format!("{p}.feed_forward.w1.weight"),
            &[FFN, h],
            true,
            false,
        );
        add(
            format!("{p}.feed_forward.w3.weight"),
            &[FFN, h],
            true,
            false,
        );
        add(
            format!("{p}.feed_forward.w2.weight"),
            &[h, FFN],
            true,
            false,
        );
        match kind {
            LayerType::FullAttention => {
                for (leaf, rows) in [
                    ("q_proj", h),
                    ("k_proj", kv),
                    ("v_proj", kv),
                    ("out_proj", h),
                ] {
                    add(
                        format!("{p}.self_attn.{leaf}.weight"),
                        &[rows, h],
                        true,
                        false,
                    );
                }
                add(
                    format!("{p}.self_attn.q_layernorm.weight"),
                    &[hd],
                    false,
                    true,
                );
                add(
                    format!("{p}.self_attn.k_layernorm.weight"),
                    &[hd],
                    false,
                    true,
                );
            }
            LayerType::Conv => {
                add(format!("{p}.conv.in_proj.weight"), &[3 * h, h], true, false);
                add(format!("{p}.conv.out_proj.weight"), &[h, h], true, false);
                add(
                    format!("{p}.conv.conv.weight"),
                    &[h, 1, L_CACHE],
                    false,
                    false,
                );
            }
        }
    }
    out
}

/// The checkpoint's values, f32, keyed by the HF name. `round` passes each
/// packed tensor through the GGUF kind it will be stored at, so the
/// safetensors arm holds exactly what the GGUF arm dequantizes.
fn tensors(round: Option<GgmlDType>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (i, s) in specs().iter().enumerate() {
        let n: usize = s.shape.iter().product();
        let fan_in = *s.shape.last().unwrap_or(&1);
        let v = if s.norm {
            values(i as u64 + 1, n, 1)
                .iter()
                .map(|x| 1.0 + 0.1 * x)
                .collect()
        } else {
            values(i as u64 + 1, n, fan_in)
        };
        let mut t = Tensor::from_vec(v, s.shape.as_slice(), &Device::Cpu)?;
        if let (Some(kind), true) = (round, s.packed) {
            t = QTensor::quantize(&t, kind)?.dequantize(&Device::Cpu)?;
        }
        out.insert(s.hf.clone(), t);
    }
    Ok(out)
}

/// The same values as llama.cpp writes them (`testing::lfm2_gguf`): packed
/// tensors at `kind`, norms and conv taps F32, no `output.weight` (tied).
fn gguf(kind: GgmlDType) -> Result<Vec<u8>> {
    crate::testing::lfm2_gguf::write(&config(), &tensors(None)?, kind, Vec::new())
}

fn from_gguf(bytes: &[u8], dev: &Device) -> Result<(Model<CrownExact>, Config)> {
    from_gguf_with(bytes, dev, |_| ())
}

/// [`from_gguf`] with `arm` applied to the header's config before
/// construction.
fn from_gguf_with(
    bytes: &[u8],
    dev: &Device,
    arm: impl FnOnce(&mut Config),
) -> Result<(Model<CrownExact>, Config)> {
    let mut reader = Cursor::new(bytes.to_vec());
    let ct = gguf_file::Content::read(&mut reader)?;
    let ckpt = GgufCheckpoint::read(&ct)?;
    let mut cfg = ckpt.config.clone();
    arm(&mut cfg);
    let vb = ckpt.var_builder(ct, reader, dev);
    let model = Model::<CrownExact>::new(&cfg, vb)?;
    Ok((model, cfg))
}

fn from_safetensors(t: HashMap<String, Tensor>, dev: &Device) -> Result<Model<CrownExact>> {
    let vb = VarBuilder::from_tensors(t, DType::F32, dev);
    Model::<CrownExact>::new(&config(), vb)
}

const PROMPT: [u32; 5] = [1, 7, 3, 42, 9];
const STEPS: usize = 6;

/// Greedy: the prompt as one pass, then `STEPS` one-token steps; each
/// step's logits and the token it chose.
fn greedy<M: CausalLM>(
    model: &M,
    cache: &mut M::Cache,
    dev: &Device,
) -> Result<Vec<(Vec<f32>, u32)>> {
    let out = greedy_on(model, cache, &PROMPT, STEPS, dev)?;
    assert!(out.iter().all(|(l, _)| l.len() == VOCAB));
    Ok(out)
}

/// [`greedy`] over `prompt` for `steps` steps.
fn greedy_on<M: CausalLM>(
    model: &M,
    cache: &mut M::Cache,
    prompt: &[u32],
    steps: usize,
    dev: &Device,
) -> Result<Vec<(Vec<f32>, u32)>> {
    let mut out = Vec::new();
    let mut ids = Tensor::new(prompt, dev)?.unsqueeze(0)?;
    let mut pos = 0;
    for _ in 0..=steps {
        let len = ids.dim(1)?;
        let logits = model
            .forward(&ids, pos, cache)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        pos += len;
        ids = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        out.push((logits, next));
    }
    Ok(out)
}

fn lfm2_run(model: &Model<CrownExact>, cfg: &Config, dev: &Device) -> Result<Vec<(Vec<f32>, u32)>> {
    let mut cache = State::<CrownExact>::new(true, DType::F32, cfg, dev)?;
    greedy(model, &mut cache, dev)
}

fn legacy_run(bytes: &[u8], dev: &Device) -> Result<Vec<(Vec<f32>, u32)>> {
    let mut reader = Cursor::new(bytes);
    let model =
        quantized_lfm2::Model::from_gguf(gguf_file::Content::read(&mut reader)?, &mut reader, dev)?;
    let mut cache = model.new_cache()?;
    greedy(&model, &mut cache, dev)
}

/// The largest |Δ| between two runs' logits, and the first step whose
/// greedy tokens differ.
fn compare(a: &[(Vec<f32>, u32)], b: &[(Vec<f32>, u32)]) -> (f32, Option<usize>) {
    let mut worst = 0f32;
    let mut first = None;
    for (step, ((la, ta), (lb, tb))) in a.iter().zip(b).enumerate() {
        for (x, y) in la.iter().zip(lb) {
            worst = worst.max((x - y).abs());
        }
        if ta != tb && first.is_none() {
            first = Some(step);
        }
    }
    (worst, first)
}

/// The header's config is the checkpoint's: every geometry field the
/// safetensors arm states, read from `lfm2.*` metadata and tensor shapes.
#[test]
fn the_gguf_header_resolves_the_safetensors_config() -> Result<()> {
    let bytes = gguf(GgmlDType::Q8_0)?;
    let ct = gguf_file::Content::read(&mut Cursor::new(&bytes))?;
    let got = GgufCheckpoint::read(&ct)?.config;
    let want = config();
    assert_eq!(
        (
            got.vocab_size,
            got.hidden_size,
            got.intermediate_size,
            got.num_hidden_layers
        ),
        (
            want.vocab_size,
            want.hidden_size,
            want.intermediate_size,
            want.num_hidden_layers
        )
    );
    assert_eq!(
        (got.num_attention_heads, got.num_key_value_heads),
        (HEADS, KV_HEADS)
    );
    assert_eq!(got.layer_types, want.layer_types);
    assert_eq!(
        (got.conv_l_cache, got.max_position_embeddings),
        (L_CACHE, CONTEXT)
    );
    assert!(got.tie_embedding);
    assert_eq!((got.rope_theta, got.norm_eps as f32), (1e6, 1e-5));
    Ok(())
}

/// The same f32 weights through both containers: one model, bit-identical
/// logits and tokens, at every step.
#[test]
fn safetensors_and_an_f32_gguf_load_bit_identically() -> Result<()> {
    let dev = Device::Cpu;
    let (g, cfg) = from_gguf(&gguf(GgmlDType::F32)?, &dev)?;
    let s = from_safetensors(tensors(None)?, &dev)?;
    let (gs, ss) = (lfm2_run(&g, &cfg, &dev)?, lfm2_run(&s, &config(), &dev)?);
    let (worst, first) = compare(&gs, &ss);
    eprintln!("f32 gguf vs safetensors: max|dlogit|={worst:e} first_token_divergence={first:?}");
    assert_eq!(worst, 0.0);
    assert_eq!(first, None);
    assert_eq!(
        g.weight_kinds(),
        [("f32".to_string(), 18)].into_iter().collect()
    );
    Ok(())
}

/// An F16 GGUF and the safetensors of the same f16-rounded values.
#[test]
fn safetensors_and_an_f16_gguf_load_bit_identically() -> Result<()> {
    let dev = Device::Cpu;
    let (g, cfg) = from_gguf(&gguf(GgmlDType::F16)?, &dev)?;
    let s = from_safetensors(tensors(Some(GgmlDType::F16))?, &dev)?;
    let (worst, first) = compare(&lfm2_run(&g, &cfg, &dev)?, &lfm2_run(&s, &config(), &dev)?);
    eprintln!("f16 gguf vs safetensors: max|dlogit|={worst:e} first_token_divergence={first:?}");
    assert_eq!((worst, first), (0.0, None));
    Ok(())
}

/// A packed GGUF stays packed: every projection and the tied table hold
/// `kind`'s blocks, none a dense copy.
fn packed_stays_packed(kind: GgmlDType) -> Result<()> {
    let (g, _) = from_gguf(&gguf(kind)?, &Device::Cpu)?;
    let name = format!("{kind:?}").to_lowercase();
    assert_eq!(g.weight_kinds(), [(name, 18)].into_iter().collect());
    Ok(())
}

#[test]
fn a_q8_0_gguf_stays_packed() -> Result<()> {
    packed_stays_packed(GgmlDType::Q8_0)
}

#[test]
fn a_q4_k_gguf_stays_packed() -> Result<()> {
    packed_stays_packed(GgmlDType::Q4K)
}

/// The packed GGUF on `models::lfm2` against the legacy GGUF model on the
/// same file, both at f32 over the same blocks: greedy tokens identical at
/// every step, logits within `bound`. The bound is fixed here, before any
/// run: the two compose the norms, the conv and the attention differently
/// (`docs/porting-models.md` §21.14), and nothing else.
fn agrees_with_the_legacy_model(kind: GgmlDType, dev: &Device, bound: f32) -> Result<()> {
    agrees_with_the_legacy_model_at(kind, dev, bound, crate::models::lfm2::AttnImpl::Generic)
}

fn agrees_with_the_legacy_model_at(
    kind: GgmlDType,
    dev: &Device,
    bound: f32,
    attn: crate::models::lfm2::AttnImpl,
) -> Result<()> {
    let bytes = gguf(kind)?;
    let (g, cfg) = from_gguf_with(&bytes, dev, |c| c.attn_impl = attn)?;
    let (worst, first) = compare(&lfm2_run(&g, &cfg, dev)?, &legacy_run(&bytes, dev)?);
    eprintln!("{kind:?} on {dev:?} attn {attn:?}: lfm2 vs quantized_lfm2 max|dlogit|={worst:e} first_token_divergence={first:?}");
    assert_eq!(first, None, "greedy tokens diverge");
    assert!(worst <= bound, "max|dlogit| {worst} > {bound}");
    Ok(())
}

#[test]
fn a_q8_0_gguf_agrees_with_the_legacy_model_on_the_cpu() -> Result<()> {
    agrees_with_the_legacy_model(GgmlDType::Q8_0, &Device::Cpu, 1e-3)
}

#[test]
fn a_q4_k_gguf_agrees_with_the_legacy_model_on_the_cpu() -> Result<()> {
    agrees_with_the_legacy_model(GgmlDType::Q4K, &Device::Cpu, 1e-3)
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    fn device() -> Result<Device> {
        Device::new_metal(0)
    }

    #[test]
    fn a_q8_0_gguf_agrees_with_the_legacy_model_on_metal() -> Result<()> {
        agrees_with_the_legacy_model(GgmlDType::Q8_0, &device()?, 1e-3)
    }

    #[test]
    fn a_q4_k_gguf_agrees_with_the_legacy_model_on_metal() -> Result<()> {
        agrees_with_the_legacy_model(GgmlDType::Q4K, &device()?, 1e-3)
    }

    /// The fused attention arms at f32 over packed weights: the vector
    /// kernel and FlashDecoding, each against the legacy model's route.
    #[test]
    fn a_q8_0_gguf_agrees_with_the_legacy_model_on_the_fused_attention_arms() -> Result<()> {
        use crate::models::lfm2::AttnImpl;
        let dev = device()?;
        for attn in [AttnImpl::Sdpa, AttnImpl::FlashDecoding] {
            agrees_with_the_legacy_model_at(GgmlDType::Q8_0, &dev, 1e-3, attn)?;
        }
        Ok(())
    }

    /// The same packed file on Metal and on the CPU through `models::lfm2`:
    /// the same greedy tokens.
    #[test]
    fn a_q8_0_gguf_decodes_the_same_tokens_on_metal_and_the_cpu() -> Result<()> {
        let bytes = gguf(GgmlDType::Q8_0)?;
        let dev = device()?;
        let (m, cfg) = from_gguf(&bytes, &dev)?;
        let (c, _) = from_gguf(&bytes, &Device::Cpu)?;
        let (worst, first) = compare(
            &lfm2_run(&m, &cfg, &dev)?,
            &lfm2_run(&c, &cfg, &Device::Cpu)?,
        );
        eprintln!("q8_0 lfm2 metal vs cpu: max|dlogit|={worst:e} first_token_divergence={first:?}");
        assert_eq!(first, None);
        Ok(())
    }

    /// The GEMV epilogue arms read a dense weight: asked for over packed
    /// blocks they decline, named once at load, and the step computes what
    /// the composed model computes.
    #[test]
    fn fused_arms_decline_on_packed_weights_by_name() -> Result<()> {
        let dev = device()?;
        let bytes = gguf(GgmlDType::Q8_0)?;
        let mut reader = Cursor::new(bytes.clone());
        let ct = gguf_file::Content::read(&mut reader)?;
        let ckpt = GgufCheckpoint::read(&ct)?;
        let mut cfg = ckpt.config.clone();
        cfg.gemv_silu = true;
        cfg.gemv_residual = true;
        let fused = Model::<CrownExact>::new(&cfg, ckpt.var_builder(ct, reader, &dev))?;
        let warning = fused
            .load_warnings()
            .iter()
            .find(|w| w.kind == crate::load_warning::LoadWarningKind::FusedArm)
            .expect("a fused-arm warning");
        assert!(
            warning.message.contains("gemv_silu") && warning.message.contains("gemv_residual"),
            "{}",
            warning.message
        );
        assert_eq!(
            fused.declined_fused_arms(),
            vec!["gemv_silu", "gemv_residual"]
        );
        let (plain, _) = from_gguf(&bytes, &dev)?;
        let (worst, first) = compare(
            &lfm2_run(&fused, &cfg, &dev)?,
            &lfm2_run(&plain, &ckpt.config, &dev)?,
        );
        assert_eq!((worst, first), (0.0, None));
        Ok(())
    }
}

/// The real file `LLOOM_LFM2_GGUF` names (an LFM2 GGUF, e.g. d1-3B-Q8_0)
/// through `models::lfm2` and through the legacy model, on the device the
/// build has: a chat-shaped prompt from the GGUF's own vocabulary, then
/// `LLOOM_LFM2_GGUF_STEPS` greedy steps (default 64). Reports max |Δlogit|
/// per step and the first token divergence with the legacy model's top-1
/// margin there; asserts the greedy streams are identical. Skipped, by
/// name, when the variable is unset. A real checkpoint: run it under
/// `lloom-arb measure`.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn a_real_lfm2_gguf_agrees_with_the_legacy_model() -> Result<()> {
    use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
    let Ok(path) = std::env::var("LLOOM_LFM2_GGUF") else {
        eprintln!("FIXTURE test=a_real_lfm2_gguf_agrees_with_the_legacy_model SKIPPED: LLOOM_LFM2_GGUF unset");
        return Ok(());
    };
    let steps: usize = std::env::var("LLOOM_LFM2_GGUF_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;
    let mut file = std::fs::File::open(&path)?;
    let ct = gguf_file::Content::read(&mut file)?;
    let tokenizer = tokenizers::Tokenizer::from_gguf(&ct).map_err(lloom_tensor::Error::wrap)?;
    let text = "<|startoftext|><|im_start|>user\nWhat is the capital of France? Answer in one paragraph.<|im_end|>\n<|im_start|>assistant\n";
    let prompt = tokenizer
        .encode(text, false)
        .map_err(lloom_tensor::Error::wrap)?
        .get_ids()
        .to_vec();
    let ckpt = GgufCheckpoint::read(&ct)?;
    let mut cfg = ckpt.config.clone();
    cfg.max_position_embeddings = cfg.max_position_embeddings.min(4096);
    if dev.is_metal() {
        cfg.attn_impl = crate::models::lfm2::AttnImpl::Sdpa;
    }
    let t = std::time::Instant::now();
    let model = Model::<CrownExact>::new(&cfg, ckpt.var_builder(ct, file, &dev))?;
    eprintln!(
        "REAL lfm2 load {:.1}s kinds={:?}",
        t.elapsed().as_secs_f64(),
        model.weight_kinds()
    );
    let legacy = {
        let mut f = std::fs::File::open(&path)?;
        let ct = gguf_file::Content::read(&mut f)?;
        quantized_lfm2::Model::preflight_gguf_content(&ct, &mut f)?
            .with_context_limit(cfg.max_position_embeddings)?
            .materialize(&mut f, &dev)?
    };
    let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?;
    let t = std::time::Instant::now();
    let a = greedy_on(&model, &mut cache, &prompt, steps, &dev)?;
    eprintln!("REAL lfm2 run {:.2}s", t.elapsed().as_secs_f64());
    let mut lcache = legacy.new_cache()?;
    let t = std::time::Instant::now();
    let b = greedy_on(&legacy, &mut lcache, &prompt, steps, &dev)?;
    eprintln!("REAL legacy run {:.2}s", t.elapsed().as_secs_f64());
    let mut first = None;
    for (step, ((la, ta), (lb, tb))) in a.iter().zip(&b).enumerate() {
        let worst = la
            .iter()
            .zip(lb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let mut sorted = lb.clone();
        sorted.sort_by(|x, y| y.total_cmp(x));
        eprintln!("REAL step={step} tok_lfm2={ta} tok_legacy={tb} max|dlogit|={worst:e} legacy_top1_margin={:e}", sorted[0] - sorted[1]);
        if ta != tb && first.is_none() {
            first = Some(step);
        }
    }
    let tokens: Vec<u32> = a.iter().map(|(_, t)| *t).collect();
    eprintln!(
        "REAL prompt_tokens={} steps={steps} first_token_divergence={first:?}",
        prompt.len()
    );
    eprintln!(
        "REAL text={:?}",
        tokenizer
            .decode(&tokens, false)
            .map_err(lloom_tensor::Error::wrap)?
    );
    assert_eq!(first, None, "the greedy streams diverge");
    Ok(())
}
