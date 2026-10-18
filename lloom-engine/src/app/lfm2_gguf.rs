//! The default family for an LFM2 GGUF (`general.architecture = lfm2`, the
//! layout llama.cpp's converter writes, d1-3B's official GGUFs among them):
//! the causal owner (`causal_loaded`) over the quantized LFM2, one sequence,
//! text chat. A native d1 GGUF answers `/v1/systemone` with the template and
//! calibration it carries, its images through the mmproj's LFM2-VL tower
//! when `--mmproj` names it; chat and decisions share the one loaded model.
use super::causal_loaded::served::{self, Config};
use super::causal_loaded::{BoundPolicy, TokenizerSource};
use super::common::{machine_label, open_device};
use super::{LoadConfig, Served, Shape};
use crate::decision::binding::CheckpointDecision;
use anyhow::{Context, Result};
use lloom_tensor::machine::Peak;
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::Device;
use std::path::{Path, PathBuf};

/// The GGUF's metadata a load needs before any weight: the vocabulary's
/// BOS/EOS, the chat template, and the sizes the machine admission prices.
struct Header {
    bos: Option<(u32, String)>,
    eos: u32,
    chat_template: Option<String>,
    width: u64,
    context: u64,
    /// Attention layers' KV heads (the conv layers carry 0).
    kv_heads: Vec<u64>,
    heads: u64,
    payload: u64,
    /// The token table's dense copy, 0 when it stays packed.
    dense_table: u64,
}

/// A GGUF count of any integer width (`Value::to_count`).
fn integer(value: &lloom_tensor::quantized::gguf_file::Value) -> Result<u64> {
    Ok(value.to_count()?)
}

impl Header {
    fn read(path: &Path, content: &Content) -> Result<Self> {
        let get = |key: &str| content.metadata.get(key);
        let number = |key: &str| -> Result<u64> {
            integer(get(key).with_context(|| format!("the GGUF has no {key}"))?)
        };
        let tokens = get("tokenizer.ggml.tokens")
            .context("the GGUF has no tokenizer.ggml.tokens")?
            .to_vec()?;
        let bos = match get("tokenizer.ggml.bos_token_id") {
            Some(id) => {
                let id = id.to_u32()?;
                let text = tokens
                    .get(id as usize)
                    .with_context(|| format!("BOS id {id} is outside the GGUF's vocabulary"))?
                    .to_string()?
                    .clone();
                Some((id, text))
            }
            None => None,
        };
        let mut kv_heads = match get("lfm2.attention.head_count_kv")
            .context("the GGUF has no lfm2.attention.head_count_kv")?
        {
            lloom_tensor::quantized::gguf_file::Value::Array(values) => {
                values.iter().map(integer).collect::<Result<Vec<_>>>()?
            }
            single => vec![integer(single)?],
        };
        if kv_heads.len() == 1 {
            kv_heads.resize(usize::try_from(number("lfm2.block_count")?)?, kv_heads[0]);
        }
        let file = std::fs::metadata(path)?.len();
        Ok(Self {
            bos,
            eos: get("tokenizer.ggml.eos_token_id")
                .context("the GGUF has no tokenizer.ggml.eos_token_id")?
                .to_u32()?,
            chat_template: get("tokenizer.chat_template")
                .map(|v| v.to_string().cloned())
                .transpose()?,
            width: number("lfm2.embedding_length")?,
            context: number("lfm2.context_length")?,
            heads: number("lfm2.attention.head_count")?,
            kv_heads,
            payload: file.saturating_sub(content.tensor_data_offset),
            dense_table: lloom::models::quantized_lfm2::dense_token_table_bytes(content)?,
        })
    }

    /// What the load holds at its peak: the file's tensor payload, the token
    /// table's dense copy if it has one, the requested KV store a sequence grows to
    /// at the effective runtime context length, and the mmproj's tower dequantized
    /// to `vision_bytes` per element.
    fn peak(
        &self,
        mmproj: Option<&Content>,
        vision_bytes: u64,
        context: usize,
        kv_dtype: lloom_tensor::DType,
    ) -> Peak {
        let head_dim = self.width / self.heads.max(1);
        let kv_per_token: u64 = self
            .kv_heads
            .iter()
            .map(|&kv| 2 * kv * head_dim * kv_dtype.size_in_bytes() as u64)
            .sum();
        let vision: u64 = mmproj.map_or(0, |content| {
            content
                .tensor_infos
                .values()
                .map(|info| info.shape.elem_count() as u64 * vision_bytes)
                .sum()
        });
        Peak::new()
            .with("weights (GGUF payload)", self.payload)
            .with("token table, dense copy", self.dense_table)
            .with(
                format!("KV at context.max_tokens={context}, {}", kv_dtype.as_str()),
                context as u64 * kv_per_token,
            )
            .with("vision tower (mmproj, dequantized)", vision)
    }
}

/// Load `first` for `serve` / `chat`. The tokenizer is `--tokenizer`, else a
/// `tokenizer.json` beside the GGUF, else the GGUF's own vocabulary (the one
/// llama.cpp reads from the same file).
pub fn load(first: PathBuf, model: &LoadConfig, shape: &Shape) -> Result<Box<dyn Served>> {
    lloom_tensor::memory_admission::begin_model_load();
    let content = {
        let mut file =
            std::fs::File::open(&first).with_context(|| format!("opening {}", first.display()))?;
        Content::read(&mut file)
            .with_context(|| format!("reading the GGUF header of {}", first.display()))?
    };
    let facts = CheckpointDecision::from_gguf(&content.metadata)?;
    let binding = BoundPolicy::resolve(&facts, model.decision_template.as_deref())?;
    let header = Header::read(&first, &content)?;
    drop(content);
    let beside = first.with_file_name("tokenizer.json");
    let tokenizer = model
        .tokenizer
        .clone()
        .or_else(|| beside.is_file().then_some(beside));
    let source = match &tokenizer {
        Some(path) => TokenizerSource::File(path),
        None => TokenizerSource::Embedded,
    };
    let device = if model.cpu() {
        Device::Cpu
    } else {
        open_device()?
    };
    let (mmproj_path, vision_missing) = media(model.mmproj.as_deref(), model.text_only);
    if let (Some(path), None) = (&model.mmproj, mmproj_path) {
        eprintln!(
            "[lfm2-gguf: text only: the mmproj {} is not loaded]",
            path.display()
        );
    }
    let mmproj = match mmproj_path {
        Some(path) => {
            let mut file = std::fs::File::open(path)
                .with_context(|| format!("opening the mmproj {}", path.display()))?;
            Some(Content::read(&mut file).with_context(|| {
                format!("reading the GGUF header of the mmproj {}", path.display())
            })?)
        }
        None => None,
    };
    let vision_bytes = if device.is_cpu() { 4 } else { 2 };
    let context = shape.memory.kv_capacity;
    anyhow::ensure!(context > 0 && context as u64 <= header.context, "lfm2-gguf context.max_tokens={context} must be in 1..={} (checkpoint context); set --ctx-size within that range", header.context);
    let kv_dtype = if shape.model.axes.get("kv-quant").is_some_and(|v| v == "f16") {
        lloom_tensor::DType::F16
    } else {
        lloom_tensor::DType::F32
    };
    let machine = lloom::nn::memory::MachineLoad {
        gate: shape.memory.machine_gate(),
        label: machine_label("lfm2", &first),
        rows: 1,
        kv_tokens: context,
        prefill_tile: shape.decode.prefill.tile,
    }
    .admit(
        &device,
        header.peak(mmproj.as_ref(), vision_bytes, context, kv_dtype),
    )?;
    drop(mmproj);
    let model_name = model.model_name.clone().unwrap_or_else(|| {
        first
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let config = Config {
        model_name,
        bos_token: header
            .bos
            .as_ref()
            .map(|(_, text)| text.clone())
            .unwrap_or_default(),
        bos_id: header.bos.as_ref().map(|(id, _)| *id),
        eos_ids: vec![header.eos],
        seed_text: shape.seed_text.clone(),
        chat_template: header.chat_template,
    };
    eprintln!(
        "[lfm2-gguf: tokenizer {}]",
        tokenizer
            .as_deref()
            .map_or("the GGUF's own vocabulary".to_string(), |p| p
                .display()
                .to_string())
    );
    let engine = served::load_lfm2_gguf_unadmitted_with_runtime(
        &first,
        mmproj_path,
        source,
        binding,
        &device,
        machine,
        shape.decode.clone(),
        config,
        shape.decision_ragged,
        vision_missing,
        Some(served::GgufRuntime { context, kv_dtype }),
    )?;
    Ok(Box::new(engine))
}

/// The mmproj a load reads and, when it reads none, why images are refused:
/// `model.text_only` loads no tower whatever `model.mmproj` names.
fn media(mmproj: Option<&Path>, text_only: bool) -> (Option<&Path>, Option<&'static str>) {
    match (mmproj, text_only) {
        (Some(path), false) => (Some(path), None),
        (_, true) => (None, Some(served::TEXT_ONLY)),
        (None, false) => (None, Some(served::NO_MMPROJ)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_uses_effective_context_and_selected_store_precision() {
        let header = Header {
            bos: None,
            eos: 0,
            chat_template: None,
            width: 2048,
            context: 65536,
            kv_heads: vec![0, 8, 0, 8],
            heads: 32,
            payload: 2_870_000_000,
            dense_table: 0,
        };
        let f32 = header.peak(None, 4, 4096, lloom_tensor::DType::F32);
        let f16 = header.peak(None, 4, 4096, lloom_tensor::DType::F16);
        assert_eq!(
            f32.rows()[1],
            (
                "KV at context.max_tokens=4096, f32".into(),
                4096 * 2 * 16 * 64 * 4
            )
        );
        assert_eq!(f32.rows()[1].1, 2 * f16.rows()[1].1);
        assert_eq!(
            header.peak(None, 4, 8192, lloom_tensor::DType::F32).rows()[1].1,
            2 * f32.rows()[1].1
        );
    }

    /// `--text-only` loads no tower even when an mmproj is named, and each
    /// missing tower is refused for its own reason.
    #[test]
    fn text_only_loads_no_tower_and_says_why() {
        let path = Path::new("mmproj.gguf");
        assert_eq!(media(Some(path), false), (Some(path), None));
        assert_eq!(media(Some(path), true), (None, Some(served::TEXT_ONLY)));
        assert_eq!(media(None, true), (None, Some(served::TEXT_ONLY)));
        assert_eq!(media(None, false), (None, Some(served::NO_MMPROJ)));
        assert_ne!(served::TEXT_ONLY, served::NO_MMPROJ);
    }
}
