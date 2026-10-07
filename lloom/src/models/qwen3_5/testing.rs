//! The Qwen3.5 family at tiny geometry (`docs/porting-models.md` §21.8): the
//! `TinyFamily` impls for the dense and the MoE variants. The schema is
//! declared (`super::schema::tensor_schema`), the files are one GGUF with
//! the `qwen35` / `qwen35moe` header keys, and the in-memory model is that
//! GGUF's bytes through the real loader — the one place the names are wired
//! to the blocks.
#![doc(hidden)]

use super::config::Config;
use super::loader::LoadOptions;
use super::{Qwen3_5, Qwen3_5Moe};
use crate::models::qwen4exp::LayerKind;
use crate::profile::Profile;
use crate::testing::tiny::{
    Schema as TinySchema, TinyCheckpoint, TinyFamily, TinyTokenizer, Workload,
};
use lloom_tensor::quantized::gguf_file::Value;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result};

/// `qwen3_5`, dense, at tiny geometry.
pub struct Qwen3_5Tiny;
/// `qwen3_5_moe` at tiny geometry.
pub struct Qwen3_5MoeTiny;

const DENSE_KINDS: &[GgmlDType] = &[
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q8_0,
    GgmlDType::Q4_0,
    GgmlDType::Q4K,
    GgmlDType::Q6K,
];
const EXPERT_KINDS: &[GgmlDType] = &[
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::Q8_0,
    GgmlDType::Q4K,
    GgmlDType::Q6K,
];

fn covers(template: &Config, cfg: &Config) -> bool {
    [LayerKind::Linear, LayerKind::Full]
        .iter()
        .all(|k| !template.layer_types.contains(k) || cfg.layer_types.contains(k))
}

fn workloads() -> Vec<Workload> {
    vec![
        // Decode only: the gemv arms, the one-query attention.
        Workload {
            prompt: 1,
            decode: 2,
        },
        // A short prefill: the f16-weight GEMM above the gemv's rows, the
        // chunked DeltaNet, the dense chunk attention.
        Workload {
            prompt: 7,
            decode: 2,
        },
        // Two DeltaNet chunks (the chunk size is 64).
        Workload {
            prompt: 70,
            decode: 1,
        },
    ]
}

fn write_checkpoint<F: TinyFamily<Config = Config>>(
    ckpt: &TinyCheckpoint<F>,
    dir: &std::path::Path,
    name: &str,
) -> Result<std::path::PathBuf> {
    let tok = F::tokenizer();
    if ckpt.cfg.vocab_size < tok.vocab_size() {
        lloom_tensor::bail!(
            "a written checkpoint carries the tokenizer: vocab_size {} < its {} ids",
            ckpt.cfg.vocab_size,
            tok.vocab_size()
        )
    }
    let path = dir.join(name);
    std::fs::write(&path, gguf_bytes(ckpt, Some(&tok))?)?;
    tok.write(dir)?;
    Ok(path)
}

impl TinyFamily for Qwen3_5Tiny {
    type Config = Config;
    type Model<P: Profile> = Qwen3_5<P>;
    const NAME: &'static str = "qwen3_5";
    const DENSE_KINDS: &'static [GgmlDType] = DENSE_KINDS;
    const EXPERT_KINDS: &'static [GgmlDType] = &[];

    fn template() -> Config {
        Config::tiny()
    }
    fn covers(template: &Config, cfg: &Config) -> bool {
        covers(template, cfg)
    }
    fn tokenizer() -> TinyTokenizer {
        TinyTokenizer::qwen3_5()
    }
    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<TinySchema> {
        Ok(super::schema::tensor_schema(
            &ckpt.cfg,
            &super::schema::Kinds {
                dense: ckpt.weights.dense,
                experts: ckpt.weights.experts,
            },
        ))
    }
    fn workloads(_cfg: &Config) -> Vec<Workload> {
        workloads()
    }
    fn build<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen3_5<P>> {
        let bytes = gguf_bytes(ckpt, None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen3_5::from_gguf(content, &mut r, dev, &LoadOptions::default())
    }
    fn write(ckpt: &TinyCheckpoint<Self>, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        write_checkpoint(ckpt, dir, "tiny-qwen3_5.gguf")
    }
    fn load<P: Profile>(path: &std::path::Path, dev: &Device) -> Result<Qwen3_5<P>> {
        Qwen3_5::from_gguf_shards(&[path.to_path_buf()], dev, &LoadOptions::default())
    }
    /// The loader's f32 arm over the dequantized checkpoint.
    fn reference<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen3_5<P>> {
        let bytes = gguf_bytes(&ckpt.dequantized(), None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen3_5::from_gguf(
            content,
            &mut r,
            dev,
            &LoadOptions {
                dense_f32: true,
                ..Default::default()
            },
        )
    }
}

impl TinyFamily for Qwen3_5MoeTiny {
    type Config = Config;
    type Model<P: Profile> = Qwen3_5Moe<P>;
    const NAME: &'static str = "qwen3_5_moe";
    const DENSE_KINDS: &'static [GgmlDType] = DENSE_KINDS;
    const EXPERT_KINDS: &'static [GgmlDType] = EXPERT_KINDS;

    fn template() -> Config {
        Config::tiny_moe()
    }
    fn covers(template: &Config, cfg: &Config) -> bool {
        covers(template, cfg)
    }
    fn tokenizer() -> TinyTokenizer {
        TinyTokenizer::qwen3_5()
    }
    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<TinySchema> {
        Ok(super::schema::tensor_schema(
            &ckpt.cfg,
            &super::schema::Kinds {
                dense: ckpt.weights.dense,
                experts: ckpt.weights.experts,
            },
        ))
    }
    fn workloads(_cfg: &Config) -> Vec<Workload> {
        workloads()
    }
    fn build<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen3_5Moe<P>> {
        let bytes = gguf_bytes(ckpt, None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen3_5Moe::from_gguf(content, &mut r, dev, &LoadOptions::default())
    }
    fn write(ckpt: &TinyCheckpoint<Self>, dir: &std::path::Path) -> Result<std::path::PathBuf> {
        write_checkpoint(ckpt, dir, "tiny-qwen3_5_moe.gguf")
    }
    fn load<P: Profile>(path: &std::path::Path, dev: &Device) -> Result<Qwen3_5Moe<P>> {
        Qwen3_5Moe::from_gguf_shards(&[path.to_path_buf()], dev, &LoadOptions::default())
    }
    fn reference<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Qwen3_5Moe<P>> {
        let bytes = gguf_bytes(&ckpt.dequantized(), None)?;
        let mut r = std::io::Cursor::new(bytes);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut r)?;
        Qwen3_5Moe::from_gguf(
            content,
            &mut r,
            dev,
            &LoadOptions {
                dense_f32: true,
                ..Default::default()
            },
        )
    }
}

/// The `qwen35.*` / `qwen35moe.*` header keys for `cfg`, as
/// `Config::from_gguf_metadata` reads them (llama.cpp's spellings); the
/// turn-end id from `tok` when the file carries one.
pub fn gguf_metadata(cfg: &Config, tok: Option<&TinyTokenizer>) -> Vec<(String, Value)> {
    let arch = cfg.arch();
    let u = |v: usize| Value::U32(v as u32);
    let k = |s: &str| format!("{arch}.{s}");
    let ints = |v: &[usize]| Value::Array(v.iter().map(|x| Value::I32(*x as i32)).collect());
    let eos = tok
        .and_then(|t| t.id("<|im_end|>"))
        .unwrap_or(cfg.eos_token_id);
    let bos = tok
        .and_then(|t| t.id("<|endoftext|>"))
        .unwrap_or(cfg.bos_token_id);
    let mut md: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(arch.into())),
        ("general.name".into(), Value::String(format!("tiny {arch}"))),
        (k("block_count"), u(cfg.n_layers + cfg.n_nextn_layers)),
        (k("embedding_length"), u(cfg.hidden_size)),
        (k("full_attention_interval"), u(cfg.full_attention_interval)),
        (k("rope.dimension_sections"), ints(&cfg.mrope_section)),
        (k("rope.dimension_count"), u(cfg.attn().n_rot)),
        (k("rope.freq_base"), Value::F32(cfg.rope_theta as f32)),
        (
            k("attention.layer_norm_rms_epsilon"),
            Value::F32(cfg.norm_eps as f32),
        ),
        (k("attention.head_count"), u(cfg.n_heads)),
        (k("attention.head_count_kv"), u(cfg.n_kv_heads)),
        (k("attention.key_length"), u(cfg.head_dim)),
        (k("attention.value_length"), u(cfg.head_dim)),
        (k("ssm.state_size"), u(cfg.ssm_state_size)),
        (k("ssm.group_count"), u(cfg.ssm_n_group)),
        (k("ssm.time_step_rank"), u(cfg.ssm_dt_rank)),
        (k("ssm.conv_kernel"), u(cfg.ssm_conv_kernel)),
        (k("ssm.inner_size"), u(cfg.ssm_inner_size)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(eos)),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(bos)),
    ];
    if cfg.n_nextn_layers > 0 {
        md.push((k("nextn_predict_layers"), u(cfg.n_nextn_layers)));
    }
    if cfg.is_moe() {
        md.push((k("expert_count"), u(cfg.n_experts)));
        md.push((k("expert_used_count"), u(cfg.n_experts_per_tok)));
        md.push((
            k("expert_feed_forward_length"),
            u(cfg.moe_intermediate_size),
        ));
        md.push((
            k("expert_shared_feed_forward_length"),
            u(cfg.shared_expert_intermediate_size),
        ));
    } else {
        md.push((k("feed_forward_length"), u(cfg.intermediate_size)));
    }
    md
}

/// The checkpoint as one GGUF file's bytes: the header keys, then every
/// tensor of the schema.
pub fn gguf_bytes<F: TinyFamily<Config = Config>>(
    ckpt: &TinyCheckpoint<F>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::model::CausalLM;
    use crate::profile::CrownExact;
    use crate::testing::tiny::Weights;

    /// The header round-trips: a tiny checkpoint's GGUF reads back to the
    /// config it was written from, on both variants.
    #[test]
    fn the_tiny_header_round_trips() -> Result<()> {
        let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
        let dense = TinyCheckpoint::<Qwen3_5Tiny>::template();
        let bytes = gguf_bytes(&dense, None)?;
        let content =
            lloom_tensor::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(&bytes))?;
        let mut cfg = Config::from_gguf_metadata(&content.metadata)?;
        cfg.vocab_size = dense.cfg.vocab_size;
        // The epsilon is an F32 header value.
        assert_eq!(cfg.norm_eps as f32, dense.cfg.norm_eps as f32);
        cfg.norm_eps = dense.cfg.norm_eps;
        assert_eq!(format!("{cfg:?}"), format!("{:?}", dense.cfg));
        let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::template();
        let bytes = gguf_bytes(&moe, None)?;
        let content =
            lloom_tensor::quantized::gguf_file::Content::read(&mut std::io::Cursor::new(&bytes))?;
        let mut cfg = Config::from_gguf_metadata(&content.metadata)?;
        cfg.vocab_size = moe.cfg.vocab_size;
        assert_eq!(cfg.norm_eps as f32, moe.cfg.norm_eps as f32);
        cfg.norm_eps = moe.cfg.norm_eps;
        assert_eq!(format!("{cfg:?}"), format!("{:?}", moe.cfg));
        Ok(())
    }

    /// Both tiny models build through the loader on the CPU and run a
    /// prompt and two decode steps; the logits are finite and the same
    /// shape as the vocabulary.
    #[test]
    fn the_tiny_models_build_and_decode_on_the_cpu() -> Result<()> {
        let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
        let dev = Device::Cpu;
        let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
        let m = dense.model::<CrownExact>(&dev)?;
        let out = crate::testing::tiny::run(
            &m,
            &dense.cfg,
            dense.cfg.vocab_size,
            &dev,
            &Workload {
                prompt: 7,
                decode: 2,
            },
        )?;
        assert_eq!(out.len(), 3);
        for step in &out {
            assert_eq!(step.len(), dense.cfg.vocab_size);
            assert!(step.iter().all(|x| x.is_finite()));
        }
        let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(Config::tiny_moe(), Weights::default());
        let m = moe.model::<CrownExact>(&dev)?;
        let out = crate::testing::tiny::run(
            &m,
            &moe.cfg,
            moe.cfg.vocab_size,
            &dev,
            &Workload {
                prompt: 7,
                decode: 2,
            },
        )?;
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|s| s.iter().all(|x| x.is_finite())));
        assert_eq!(m.vocab_size(), moe.cfg.vocab_size);
        Ok(())
    }
}
