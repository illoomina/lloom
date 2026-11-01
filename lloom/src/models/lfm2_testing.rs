//! LFM2 at tiny geometry (`docs/porting-models.md` §21.8): the
//! [`TinyFamily`] impl. The schema is recorded — `Model::new` is built
//! through the fabricating backend and its requests are the tensor table —
//! so nothing here lists a tensor name. The files are what an LFM2
//! checkpoint directory holds: `model.safetensors`, `config.json`,
//! `tokenizer.json`, `chat_template.jinja`.
#![doc(hidden)]

use super::lfm2::{AttnImpl, Config, LayerType, Lfm2Config, Model};
use crate::nn::var_builder::RecordingBackend;
use crate::nn::VarBuilder;
use crate::profile::Profile;
use crate::testing::tiny::backend::TinyBackend;
use crate::testing::tiny::{Schema, TinyCheckpoint, TinyFamily, TinyTokenizer, Workload};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{bail, DType, Device, Result, Shape};
use std::path::{Path, PathBuf};

/// The LFM2 family at tiny geometry.
pub struct Lfm2Tiny;

/// The dtype a model computes in when the checkpoint does not say: the
/// engine's rule (`f16` on Metal, `f32` on the CPU).
pub fn run_dtype(dev: &Device) -> DType {
    if dev.is_cpu() {
        DType::F32
    } else {
        DType::F16
    }
}

fn store_dtype(kind: GgmlDType) -> Result<DType> {
    Ok(match kind {
        GgmlDType::F32 => DType::F32,
        GgmlDType::F16 => DType::F16,
        GgmlDType::BF16 => DType::BF16,
        other => bail!("an LFM2 checkpoint stores {other:?}? safetensors holds F32 | F16 | BF16"),
    })
}

fn backend(
    ckpt: &TinyCheckpoint<Lfm2Tiny>,
) -> Result<(TinyBackend, crate::testing::tiny::backend::Recording)> {
    let w = &ckpt.weights;
    Ok(TinyBackend::with_store(
        w.law,
        w.seed,
        store_dtype(w.dense)?,
    ))
}

impl TinyFamily for Lfm2Tiny {
    type Config = Config;
    type Model<P: Profile> = Model<P>;
    const NAME: &'static str = "lfm2";
    const DENSE_KINDS: &'static [GgmlDType] = &[GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16];
    const EXPERT_KINDS: &'static [GgmlDType] = &[];

    /// Three layers (conv, conv, attention), head_dim 64 — inside the SDPA
    /// head-dim set, so the fused attention arms are reachable on Metal —
    /// and RoPE tables long enough for the engine's default KV capacity.
    fn template() -> Config {
        let num_hidden_layers = 3;
        Config {
            vocab_size: 64,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: Config::tiny_layer_types(num_hidden_layers),
            tie_embedding: true,
            bos_token_id: None,
            eos_token_id: None,
            ..Default::default()
        }
    }

    fn covers(template: &Config, cfg: &Config) -> bool {
        [LayerType::Conv, LayerType::FullAttention]
            .iter()
            .all(|k| !template.layer_types.contains(k) || cfg.layer_types.contains(k))
    }

    fn tokenizer() -> TinyTokenizer {
        TinyTokenizer::lfm2()
    }

    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<Schema> {
        Self::schema_for::<crate::profile::CrownExact>(ckpt)
    }

    fn schema_for<P: Profile>(ckpt: &TinyCheckpoint<Self>) -> Result<Schema> {
        let (be, rec) = backend(ckpt)?;
        let vb = VarBuilder::from_backend(Box::new(be), DType::F32, Device::Cpu);
        // The tensor reads do not vary by profile; recording still obeys its regime.
        Model::<P>::new(&ckpt.cfg, vb)?;
        Ok(rec.schema(ckpt.weights.dense))
    }

    fn workloads(_cfg: &Config) -> Vec<Workload> {
        vec![
            // Decode only: the gemv arms, `sdpa_vector`.
            Workload {
                prompt: 1,
                decode: 2,
            },
            // A short prefill: the multi-row arms under the skinny tile.
            Workload {
                prompt: 7,
                decode: 2,
            },
            // Past `SKINNY_TILE_MAX_M` rows: the tiled GEMM.
            Workload {
                prompt: 33,
                decode: 1,
            },
        ]
    }

    fn build<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Model<P>> {
        let (be, _) = backend(ckpt)?;
        let dtype = ckpt.dtype.unwrap_or_else(|| run_dtype(dev));
        let vb = VarBuilder::from_backend(Box::new(be), dtype, dev.clone());
        Model::new(&ckpt.cfg, vb)
    }

    fn write(ckpt: &TinyCheckpoint<Self>, dir: &Path) -> Result<PathBuf> {
        write_schema(ckpt, dir, ckpt.schema()?)
    }

    fn write_for<P: Profile>(ckpt: &TinyCheckpoint<Self>, dir: &Path) -> Result<PathBuf> {
        write_schema(ckpt, dir, ckpt.schema_for::<P>()?)
    }

    fn load<P: Profile>(path: &Path, dev: &Device) -> Result<Model<P>> {
        let (model, _) = load_recorded::<P>(path, dev, None)?;
        Ok(model)
    }

    /// f32 on the CPU over the values the in-memory model holds: its dense
    /// matrices rounded through the dtype it computes in (f16 on Metal by
    /// the engine's rule), so the reference differs from it by the
    /// activations' precision alone.
    fn reference<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Model<P>> {
        let run = ckpt.dtype.unwrap_or(DType::F16);
        let mut r = ckpt.dequantized().with_dtype(DType::F32);
        if r.weights.dense == GgmlDType::F32 {
            r.weights.dense = match run {
                DType::F16 => GgmlDType::F16,
                DType::BF16 => GgmlDType::BF16,
                _ => GgmlDType::F32,
            };
        }
        Self::build::<P>(&r, dev)
    }

    /// The attention arms: the generic path, `sdpa_vector` /
    /// `steel_attention`, FlashDecoding. Each falls back to the generic path
    /// where it does not apply (the CPU, an off-set head dim).
    const ARMS: &'static [&'static str] = &["generic", "sdpa", "flash"];

    fn set_arm(cfg: &mut Config, arm: usize) {
        cfg.attn_impl = match arm {
            1 => AttnImpl::Sdpa,
            2 => AttnImpl::FlashDecoding,
            _ => AttnImpl::Generic,
        };
    }
}

/// `config.json` for `cfg`, in the checkpoint's own keys, with the token ids
/// of the tokenizer the files carry.
pub fn config_json(cfg: &Config, tok: &TinyTokenizer) -> String {
    let mut j = geometry_json(cfg);
    j["bos_token_id"] = serde_json::json!(tok.id("<|startoftext|>"));
    j["eos_token_id"] = serde_json::json!(tok.id("<|im_end|>"));
    j["pad_token_id"] = serde_json::json!(tok.id("<|pad|>"));
    serde_json::to_string_pretty(&j).expect("a JSON value serializes")
}

/// `cfg`'s geometry in the checkpoint's own keys, the FFN width stated
/// rather than derived: what `Lfm2Config::into_config` reads back to `cfg`'s
/// geometry, and the form a real checkpoint's geometry is committed in
/// ([`SCALE_GEOMETRY`]).
pub fn geometry_json(cfg: &Config) -> serde_json::Value {
    let kinds: Vec<&str> = cfg
        .layer_types
        .iter()
        .map(|k| match k {
            LayerType::Conv => "conv",
            LayerType::FullAttention => "full_attention",
        })
        .collect();
    let mut json = serde_json::json!({
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "block_auto_adjust_ff_dim": false,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "conv_L_cache": cfg.conv_l_cache,
        "conv_bias": cfg.conv_bias,
        "layer_types": kinds,
        "tie_embedding": cfg.tie_embedding,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
    });
    if let Some(m) = cfg.moe {
        json["model_type"] = serde_json::json!("lfm2_moe");
        json["architectures"] = serde_json::json!(["Lfm2MoeForCausalLM"]);
        json["rope_parameters"] =
            serde_json::json!({"rope_theta":cfg.rope_theta,"rope_type":"default"});
        json["num_dense_layers"] = serde_json::json!(m.num_dense_layers);
        json["num_experts"] = serde_json::json!(m.num_experts);
        json["num_experts_per_tok"] = serde_json::json!(m.num_experts_per_tok);
        json["moe_intermediate_size"] = serde_json::json!(m.moe_intermediate_size);
        json["norm_topk_prob"] = serde_json::json!(m.norm_topk_prob);
        json["routed_scaling_factor"] = serde_json::json!(m.routed_scaling_factor);
        json["use_expert_bias"] = serde_json::json!(m.use_expert_bias);
    }
    json
}

/// The committed geometry of the checkpoint `lloom` serves by default
/// (LFM2.5-2.6B), under `lloom/tests/scale_manifests/`: what a synthetic
/// LFM2 is built at when the checkpoint is not on the machine.
pub const SCALE_GEOMETRY: &str = "lfm2.5-2.6b.json";

/// Where [`SCALE_GEOMETRY`] is in this source tree.
pub fn scale_geometry_path() -> PathBuf {
    std::path::PathBuf::from(lloom_manifest_dir())
        .join("tests/scale_manifests")
        .join(SCALE_GEOMETRY)
}

#[cfg(not(target_arch = "wasm32"))]
impl crate::testing::fixture::Scaled for Lfm2Tiny {
    const CHECKPOINT_ENV: &'static str = "LFM2_MODEL_DIR";
    const REPO: (&'static str, &'static str) = ("LiquidAI/LFM2.5-2.6B", "config.json");

    fn checkpoint_of(file: PathBuf) -> PathBuf {
        file.parent().map(Path::to_path_buf).unwrap_or(file)
    }

    fn scale_config(source: &crate::testing::fixture::Source) -> Result<Config> {
        use crate::testing::fixture::Source;
        let json = match source {
            Source::Real(dir) => std::fs::read(dir.join("config.json"))?,
            Source::Synthetic => std::fs::read(scale_geometry_path())?,
        };
        let cfg: Lfm2Config = serde_json::from_slice(&json).map_err(lloom_tensor::Error::wrap)?;
        Ok(cfg.into_config(false))
    }

    /// The real checkpoint through the loader the engine uses (the mapped
    /// safetensors), or the synthetic one in memory: the recording backend's
    /// keyed values at the real geometry, no file read.
    fn scale_model<P: Profile>(
        source: &crate::testing::fixture::Source,
        cfg: &Config,
        dtype: Option<DType>,
        weights: &crate::testing::tiny::Weights,
        dev: &Device,
    ) -> Result<Model<P>> {
        use crate::testing::fixture::Source;
        let dtype = dtype.unwrap_or_else(|| run_dtype(dev));
        match source {
            Source::Real(dir) => {
                let ckpt = crate::hub::Checkpoint::from_safetensors_dir(dir)?;
                // SAFETY: a checkpoint in the cache is not modified while a
                // test maps it.
                let vb = unsafe { ckpt.var_builder(dtype, dev)? };
                Model::new(cfg, vb)
            }
            Source::Synthetic => TinyCheckpoint::<Self>::new(cfg.clone(), *weights)
                .with_dtype(dtype)
                .model::<P>(dev),
        }
    }
}

/// A checkpoint directory through the loader the engine uses
/// (`RecordingBackend` over `model.safetensors`, `config.json` through
/// `Lfm2Config`), and the names the constructor loaded.
pub fn load_recorded<P: Profile>(
    dir: &Path,
    dev: &Device,
    dtype: Option<DType>,
) -> Result<(Model<P>, crate::nn::var_builder::LoadRecord)> {
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
        .map_err(lloom_tensor::Error::wrap)?;
    let cfg = super::lfm2_moe::config::parse_lfm_config(root)?;
    // SAFETY: the checkpoint is not modified while mapped; a test writes it
    // before loading and never after.
    let (be, record) =
        unsafe { RecordingBackend::over_safetensors(&[dir.join("model.safetensors")])? };
    let vb = VarBuilder::from_backend(
        Box::new(be),
        dtype.unwrap_or_else(|| run_dtype(dev)),
        dev.clone(),
    );
    Ok((Model::new(&cfg, vb)?, record))
}

/// The crate's manifest directory **at run time** — what cargo and nextest
/// export to the test process — falling back to the build-time path. A test
/// binary archived on one host and run on another (`cargo nextest archive` /
/// `--workspace-remap`) reads its fixtures where the archive was unpacked;
/// `env!("CARGO_MANIFEST_DIR")` would bake the building host's path in
/// (measured 2026-10-13: `every_test_file_is_a_module` panicked with NotFound
/// on silverberry from a Mac-built archive).
fn lloom_manifest_dir() -> String {
    std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string())
}

/// The checkpoint files for `ckpt` under `schema`.
fn write_schema(ckpt: &TinyCheckpoint<Lfm2Tiny>, dir: &Path, schema: Schema) -> Result<PathBuf> {
    let store = store_dtype(ckpt.weights.dense)?;
    let w = &ckpt.weights;
    let mut tensors = std::collections::HashMap::new();
    for t in &schema.tensors {
        let v = w
            .law
            .tensor(
                w.seed,
                &t.name,
                t.role,
                &Shape::from(t.shape.clone()),
                &Device::Cpu,
            )?
            .to_dtype(store)?;
        tensors.insert(t.name.clone(), v);
    }
    lloom_tensor::safetensors::save(&tensors, dir.join("model.safetensors"))?;
    let tok = Lfm2Tiny::tokenizer();
    std::fs::write(dir.join("config.json"), config_json(&ckpt.cfg, &tok))?;
    tok.write(dir)?;
    Ok(dir.to_path_buf())
}
