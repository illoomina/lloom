//! Decode across the extracted seam — the Wave F (#650) gate arm.
//!
//! Drives the same two-turn greedy decode `lfm2-smoke` runs (§3.1a, #121), but
//! through `lloom_engine::DecodeSession` instead of a hand-written loop. If the
//! seam is faithful the text is identical to `lfm2-smoke`'s on the same
//! checkpoint and `--attn` arm — the "token-stream comparison across the seam"
//! §14.2 asks Wave F for. It reuses amber's config normalization so it also
//! exercises the loader path amber's `Lfm2` takes.
//!
//! Greedy (`temperature = 0` collapses to argmax), seeded, two turns against one
//! cache — the shape that makes turn 2 a test of the accumulated cache.

use anyhow::{Context, Result};
use lloom::models::lfm2::{self, AttnImpl, Cache, Model};
use lloom::nn::VarBuilder;
use lloom_engine::{DecodeParams, DecodeSession};
use lloom_tensor::{DType, Device};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let attn = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--attn")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "generic".to_string());
    let attn_impl: AttnImpl = AttnImpl::parse(&attn).map_err(|e| anyhow::anyhow!("{e}"))?;

    let model_dir = default_model_dir().context(
        "no LFM2 checkpoint found; set the snapshot dir in $LFM2_DIR or place it under the HF cache",
    )?;
    let device = Device::new_metal(0).context("opening Metal device")?;
    let dtype = DType::F16;

    let config_json = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_json)?;
    config.attn_impl = attn_impl;

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::new(&config, vb).context("constructing LFM2 model")?;
    let cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

    println!("model     {}", model_dir.display());
    println!("attention {:?}", config.attn_impl);
    println!("seam      lloom_engine::DecodeSession (greedy, seed 0)");
    println!();

    // One session, one cache: turn 2 decodes on the accumulated state.
    let mut session = DecodeSession::new(&model, cache, device.clone(), 0);

    let turns = [
        "The quick brown fox jumps over the lazy",
        "What is the capital of France?",
    ];

    let mut ok = true;
    for (idx, prompt) in turns.iter().enumerate() {
        let templated =
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
        let ids = tokenizer
            .encode(templated.as_str(), idx == 0)
            .map_err(|e| anyhow::anyhow!("tokenizing turn {}: {e}", idx + 1))?
            .get_ids()
            .to_vec();

        let params = DecodeParams {
            seed: 0,
            temperature: None, // greedy argmax
            top_p: None,
            max_tokens: 64,
            eos_ids: eos_ids.clone(),
            stop_strings: Vec::new(),
        };

        let tk = &tokenizer;
        let detok = |ids: &[u32]| -> lloom_tensor::Result<String> {
            tk.decode(ids, false)
                .map_err(|e| lloom_tensor::Error::Msg(format!("detok: {e}")))
        };

        let decoded = session.decode(&ids, detok, &params, |_| {})?;
        println!("--- turn {} ---", idx + 1);
        println!("prompt: {prompt}");
        println!("output: {}", decoded.text.trim());
        println!("        ({} tokens, {:?})", decoded.tokens.len(), decoded.stop);

        // Turn 2's low-bar substring check, the §3.1a gate: a working cache
        // answers "Paris" against the accumulated state.
        if idx == 1 && !decoded.text.contains("Paris") {
            eprintln!("FAIL: turn 2 did not contain \"Paris\"");
            ok = false;
        }
    }

    if ok {
        println!("\nPASS");
        Ok(())
    } else {
        anyhow::bail!("smoke check failed")
    }
}

/// The HF-cache snapshot for LFM2.5-VL-3B (or `$LFM2_DIR`).
fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-3B/snapshots");
    std::fs::read_dir(&snaps).ok()?.flatten().next().map(|e| e.path())
}

/// Parse `config.json` into a runtime `lfm2::Config`, reusing the normalization
/// amber's loader does (nested VL config, `rope_theta`, `tie_embedding`).
fn parse_config(raw: &str) -> Result<lfm2::Config> {
    let root: Value = serde_json::from_str(raw).context("parsing config.json")?;
    let text = root.get("text_config").unwrap_or(&root);
    let mut obj = text
        .as_object()
        .context("config is not a JSON object")?
        .clone();

    if !obj.contains_key("rope_theta") {
        if let Some(theta) = obj
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .cloned()
        {
            obj.insert("rope_theta".into(), theta);
        }
    }
    if !obj.contains_key("tie_embedding") {
        if let Some(v) = obj.get("tie_word_embeddings").cloned() {
            obj.insert("tie_embedding".into(), v);
        }
    }
    for key in ["bos_token_id", "eos_token_id"] {
        if !obj.contains_key(key) {
            if let Some(v) = root.get(key).cloned() {
                obj.insert(key.into(), v);
            }
        }
    }

    let normalized = Value::Object(obj);
    let base: lfm2::Lfm2Config = serde_json::from_value(normalized.clone())
        .context("config.json does not match the LFM2 config schema")?;
    let mut config = base.into_config(false);

    // The stated intermediate_size is authoritative when present (the VL FFN is
    // [10752, 2048] where the formula yields 8192).
    if let Some(stated) = normalized
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    {
        if stated != config.intermediate_size {
            config.intermediate_size = stated;
        }
    }
    Ok(config)
}

/// The safetensors shards for a checkpoint, in stable order.
fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = std::fs::read_to_string(&index)?;
        let v: Value = serde_json::from_str(&json)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index has no weight_map")?;
        let mut names: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        names.sort();
        names.dedup();
        Ok(names.into_iter().map(|n| dir.join(n)).collect())
    } else {
        Ok(vec![dir.join("model.safetensors")])
    }
}

/// Tensor names from a safetensors header, without mapping the weights.
fn tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    anyhow::ensure!(
        header_len > 0 && header_len < 100 * 1024 * 1024,
        "implausible header length {header_len}"
    );
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let parsed: Value = serde_json::from_slice(&header)?;
    Ok(parsed
        .as_object()
        .context("header is not an object")?
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}
