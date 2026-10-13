//! `lloom serve` on a d1-omni checkpoint: a safetensors directory (its
//! `config.json` says `d1_omni`) or the official GGUF (`<arch>.decision.type =
//! lfm2-d1-omni`) with its mmproj. The model generates nothing, so it is
//! served by the decision-only listener (`serve::run_decisions_loaded`),
//! admitted on the machine at load as the other default families are.
use super::{Attention, DeviceRecovery, Engine, MachineAdmitted, Policy, TokenizerSource};
use crate::app::common::{machine_label, open_device};
use crate::app::decision_endpoint::DecisionEndpoint;
use crate::app::{LoadConfig, Shape};
use crate::decision::{NoulAliases, StateNullability};
use anyhow::{Context, Result};
use lloom::models::d1_omni::{attention, OmniConfig};
use lloom_tensor::machine::Peak;
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::{DType, Device};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The decision type llama.cpp's converter writes for d1-omni.
const DECISION_TYPE: &str = "lfm2-d1-omni";
/// What the template is given for each image, and once for an audio clip.
const MEDIA_MARKER: &str = "<__media__>";
/// Request limits: llama.cpp's, as for `lfm2-d1`; a noul names its sides
/// `true`/`false` or `yes`/`no`, as the reference reads them (`prompt.py`).
const MAX_OPTIONS: usize = 255;

pub(crate) enum Checkpoint {
    Safetensors(PathBuf),
    Gguf(PathBuf),
}

/// A d1-omni checkpoint at `path`, or `None` for anything else.
pub(crate) fn detect(path: &Path) -> Option<Checkpoint> {
    if path.is_dir() {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("config.json")).ok()?).ok()?;
        return (config["model_type"] == "d1_omni").then(|| Checkpoint::Safetensors(path.into()));
    }
    if path.extension().is_some_and(|e| e == "gguf") {
        let content = Content::read(&mut std::fs::File::open(path).ok()?).ok()?;
        let arch = content
            .metadata
            .get("general.architecture")?
            .to_string()
            .ok()?
            .clone();
        let kind = content.metadata.get(&format!("{arch}.decision.type"))?;
        return (kind.to_string().ok()? == DECISION_TYPE).then(|| Checkpoint::Gguf(path.into()));
    }
    None
}

/// The number of elements a safetensors file's tensors hold.
fn safetensors_elements(path: &Path) -> Result<u64> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = usize::try_from(u64::from_le_bytes(len))?;
    anyhow::ensure!(
        len < 1 << 28,
        "{}: implausible safetensors header",
        path.display()
    );
    let mut header = vec![0u8; len];
    file.read_exact(&mut header)?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)?;
    Ok(header
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .map(|(_, info)| {
            info["shape"].as_array().map_or(0, |dims| {
                dims.iter().map(|d| d.as_u64().unwrap_or(0)).product()
            })
        })
        .sum())
}

/// A GGUF file's tensor payload, and the f32 copies `QMatMul` makes of its
/// dense (F32/F16/BF16) matrices.
fn gguf_bytes(path: &Path) -> Result<(u64, u64)> {
    use lloom_tensor::quantized::GgmlDType;
    let content = Content::read(&mut std::fs::File::open(path)?)?;
    let payload = std::fs::metadata(path)?
        .len()
        .saturating_sub(content.tensor_data_offset);
    let dense = content
        .tensor_infos
        .values()
        .filter(|info| info.shape.rank() >= 2)
        .filter(|info| {
            matches!(
                info.ggml_dtype,
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            )
        })
        .map(|info| info.shape.elem_count() as u64 * 4)
        .sum();
    Ok((payload, dense))
}

/// What a request holds at its largest, beside the weights: one attention
/// query tile's scores against every key, and the trunk's live activations,
/// both at the model's full length.
fn request_peak(max_length: u64, width: u64, heads: u64) -> (u64, u64) {
    (256 * max_length * heads * 4, 8 * max_length * width * 4)
}

/// Load `load.model` as the d1-omni decision endpoint, or `None` when it is
/// not a d1-omni checkpoint.
pub(crate) fn load(load: &LoadConfig, shape: &Shape) -> Result<Option<Box<dyn DecisionEndpoint>>> {
    let Some(path) = load.model.as_deref() else {
        return Ok(None);
    };
    let Some(checkpoint) = detect(path) else {
        return Ok(None);
    };
    lloom_tensor::memory_admission::begin_model_load();
    let device = if load.cpu() {
        Device::Cpu
    } else {
        open_device()?
    };
    // The reference recommends f16 on a GPU (it keeps the f32 answers).
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };
    let policy = Arc::new(Policy::native(
        match &checkpoint {
            Checkpoint::Safetensors(dir) | Checkpoint::Gguf(dir) => dir,
        },
        MAX_OPTIONS,
        NoulAliases::TrueFalseAndYesNo,
        StateNullability::AllowNull,
        MEDIA_MARKER.into(),
    )?);
    // The checkpoint's name, as the machine reservation lists it.
    let name = load.model_name.clone().unwrap_or_else(|| {
        let label = machine_label("d1-omni", path);
        label.trim_start_matches("lloom d1-omni ").to_string()
    });
    let machine = |peak: Peak, max_length: usize| {
        lloom::nn::memory::MachineLoad {
            gate: shape.memory.machine_gate(),
            label: machine_label("d1-omni", path),
            rows: 1,
            kv_tokens: max_length,
            prefill_tile: shape.decode.prefill.tile,
        }
        .admit(&device, peak)
    };
    let recovery = DeviceRecovery(device.clone());
    let attention = Attention {
        trunk: attention::trunk,
        head: attention::masked,
        vision: attention::masked,
        audio: attention::relative,
    };
    let endpoint: Box<dyn DecisionEndpoint> = match checkpoint {
        Checkpoint::Safetensors(dir) => {
            let config: OmniConfig =
                serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
                    .with_context(|| format!("{}: config.json", dir.display()))?;
            let mut weights: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            weights.sort();
            anyhow::ensure!(
                !weights.is_empty(),
                "{}: no .safetensors weights",
                dir.display()
            );
            let elements = weights
                .iter()
                .map(|w| safetensors_elements(w))
                .sum::<Result<u64>>()?;
            let text = &config.core.text_config;
            let (scores, live) = request_peak(
                config.core.max_length as u64,
                text.hidden_size as u64,
                text.num_attention_heads as u64,
            );
            let admitted = machine(
                Peak::new()
                    .with(
                        format!("weights ({dtype:?})"),
                        elements * dtype.size_in_bytes() as u64,
                    )
                    .with("attention scores, one query tile at max_length", scores)
                    .with("trunk activations at max_length", live),
                config.core.max_length,
            )?;
            let tokenizer = dir.join("tokenizer.json");
            // SAFETY: the checkpoint's files are read-only snapshots for the
            // server's life; nothing in this process writes them.
            let engine = unsafe {
                Engine::load_safetensors_on_machine(
                    config,
                    &weights,
                    TokenizerSource::File(&tokenizer),
                    &device,
                    dtype,
                    admitted,
                    policy,
                    MachineAdmitted,
                    recovery,
                )?
            };
            eprintln!("[d1-omni: safetensors on {device:?}, {dtype:?}]");
            Box::new(engine.into_endpoint(name, attention))
        }
        Checkpoint::Gguf(main) => {
            let mmproj = load.mmproj.as_deref().with_context(|| {
                format!(
                    "{}: d1-omni's GGUF holds the trunk and head; its vision and audio towers are the mmproj (--mmproj)",
                    main.display()
                )
            })?;
            let (main_payload, main_dense) = gguf_bytes(&main)?;
            let (media_payload, media_dense) = gguf_bytes(mmproj)?;
            let max_length = 16384;
            let (scores, live) = request_peak(max_length, 1024, 16);
            let admitted = machine(
                Peak::new()
                    .with("weights (GGUF payload)", main_payload + media_payload)
                    .with("dense matrices, f32 copies", main_dense + media_dense)
                    .with("attention scores, one query tile at max_length", scores)
                    .with("trunk activations at max_length", live),
                max_length as usize,
            )?;
            let beside = main.with_file_name("tokenizer.json");
            let tokenizer = match load.tokenizer.as_deref() {
                Some(path) => TokenizerSource::File(path),
                None if beside.is_file() => TokenizerSource::File(&beside),
                None => TokenizerSource::Gguf(&main),
            };
            let engine = Engine::load_gguf_on_machine(
                &main,
                mmproj,
                tokenizer,
                &device,
                admitted,
                policy,
                MachineAdmitted,
                recovery,
            )?;
            eprintln!("[d1-omni: GGUF + mmproj on {device:?}]");
            Box::new(engine.into_endpoint(name, attention))
        }
    };
    eprintln!("[decision template: {DECISION_TYPE} (the checkpoint's)]");
    Ok(Some(endpoint))
}
