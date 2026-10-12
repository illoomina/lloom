//! The load-path gate (`RESTRUCTURING_DESIGN.md` §3.6, issue #642): safetensors
//! and GGUF both route through one `Checkpoint`.
//!
//! Both fixtures are built on disk and loaded through the ONE offline provider
//! (`LocalDir`), which dispatches by extension. The gate is that the two formats
//! reach the same staging type — same `Checkpoint`, same `manifest`-visible
//! config — so a model builder above them sees one shape. The safetensors arm is
//! carried all the way to a built model through `Pretrained`; the GGUF arm stops
//! at `Checkpoint` because the quantized model is a different type that does not
//! implement `Model` (documented on `Checkpoint::var_builder`).

use lloom::hub::{Checkpoint, Hub, LocalDir, Pretrained, TokenizerFromCheckpoint, Weights};
use lloom::nn::{Model, VarBuilder};
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// A model that reads exactly one tensor, so `from_weights` proves the
/// safetensors arm reached a real `VarBuilder` over the checkpoint's shards.
struct OneTensorModel {
    w: Tensor,
}

struct OneTensorConfig {
    hidden: usize,
}

impl Model for OneTensorModel {
    type Config = OneTensorConfig;
    fn from_weights(vb: VarBuilder, cfg: &Self::Config) -> Result<Self> {
        let w = vb.get((cfg.hidden,), "w")?;
        Ok(Self { w })
    }
}

/// Write a minimal safetensors directory: `config.json` + `model.safetensors`.
fn write_safetensors_dir(dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&serde_json::json!({ "hidden_size": 4 })).unwrap(),
    )
    .unwrap();
    let w = Tensor::arange(0f32, 4f32, &Device::Cpu)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("w".to_string(), w);
    lloom_tensor::safetensors::save(&tensors, dir.join("model.safetensors"))?;
    Ok(())
}

/// Write a minimal GGUF file carrying a gpt2-style tokenizer in its metadata, so
/// the GGUF arm's "tokenizer lives in the weights" (B11) is exercised.
fn write_gguf(path: &std::path::Path) -> Result<()> {
    use gguf_file::Value;
    let dummy = Tensor::zeros((4,), DType::F32, &Device::Cpu)?;
    let qtensor = QTensor::quantize(&dummy, GgmlDType::F32)?;

    // A two-token gpt2 vocabulary with no merges: enough for `from_gguf` to build
    // a `Tokenizer`, which is all the routing gate needs to prove.
    let tokens = Value::Array(vec![
        Value::String("a".to_string()),
        Value::String("b".to_string()),
    ]);
    let merges = Value::Array(vec![]);
    let arch = Value::String("lfm2".to_string());
    let embed_len = Value::U32(4);
    let tok_model = Value::String("gpt2".to_string());
    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("lfm2.embedding_length", &embed_len),
        ("tokenizer.ggml.model", &tok_model),
        ("tokenizer.ggml.tokens", &tokens),
        ("tokenizer.ggml.merges", &merges),
    ];
    let tensors: Vec<(&str, &QTensor)> = vec![("w", &qtensor)];

    let mut file = std::fs::File::create(path)
        .map_err(|e| lloom_tensor::Error::msg(format!("creating {}: {e}", path.display())))?;
    gguf_file::write(&mut file, &metadata, &tensors)?;
    Ok(())
}

#[test]
fn both_formats_route_through_checkpoint() -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("candle-hub-routing-{}", std::process::id()));
    let st_dir = tmp.join("st");
    let gguf_path = tmp.join("model.gguf");
    write_safetensors_dir(&st_dir)?;
    write_gguf(&gguf_path)?;

    // ONE provider, ONE call shape — dispatched by extension. This is the "same
    // load path" the gate asks for: the caller does not branch on format.
    let provider = LocalDir::new(&tmp);
    let cp_st: Checkpoint = provider.checkpoint("st")?;
    let cp_gguf: Checkpoint = provider.checkpoint("model.gguf")?;

    // Both reached a `Checkpoint` with a non-null config visible in the manifest.
    let config_present = |cp: &Checkpoint| {
        cp.manifest()
            .into_iter()
            .any(|(name, present)| name == "config" && present)
    };
    assert!(config_present(&cp_st), "safetensors config missing");
    assert!(config_present(&cp_gguf), "gguf config missing");

    // The two formats are distinguished only by `Weights`, everything above is
    // the same type.
    assert!(matches!(cp_st.weights, Weights::Safetensors { .. }));
    assert!(matches!(cp_gguf.weights, Weights::Gguf(_)));

    // GGUF carries its tokenizer in the weights; the safetensors dir had none.
    assert!(
        tokenizer_from(&cp_gguf).is_ok(),
        "gguf tokenizer should come from the weights' metadata"
    );

    // The safetensors arm is carried all the way to a built model through the
    // SAME checkpoint — `from_weights` only runs if the `VarBuilder` reached the
    // shard the checkpoint staged.
    let cfg = OneTensorConfig { hidden: 4 };
    let pretrained: Pretrained<OneTensorModel> =
        unsafe { Pretrained::from_checkpoint(cp_st, &cfg, DType::F32, &Device::Cpu)? };
    assert_eq!(pretrained.model.w.dims(), &[4]);

    // The GGUF arm stops at the checkpoint: `var_builder` refuses, by design.
    assert!(
        unsafe { cp_gguf.var_builder(DType::F32, &Device::Cpu) }.is_err(),
        "a GGUF checkpoint must not build a VarBuilder"
    );

    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

fn tokenizer_from(cp: &Checkpoint) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_checkpoint(cp)
}
