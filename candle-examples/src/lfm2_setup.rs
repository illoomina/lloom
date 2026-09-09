//! Model setup shared by the LFM2 harnesses — `DESIGN.md` §14.5 migration step 2.
//!
//! These four functions were **byte-identical across four harnesses**, verified
//! by SHA rather than by reading (lloom #291, `measurements/issue-287-one-suite-two-modes.md`
//! §2.2): `parse_config` 47 lines, `tensor_names` 23, `weight_files` 19 and
//! `default_model_dir` 13 — 102 lines existing four times, so moving them here
//! deletes **306** and changes no behaviour.
//!
//! # Why this is part of #284 rather than a later tidy-up
//!
//! §14.5 orders the shared module as step 2 and the speculation port as step 1,
//! and #284 takes both **in one step deliberately**. The duplication is *why*
//! the axes drifted apart: with four copies of the setup, an axis reaches a
//! harness only if someone remembers that harness exists, which is how
//! `--speculate` ended up on the one binary without GPU timing while `--batch`
//! went to the one without a verify loop (§10.2i). Porting speculation without
//! collapsing the setup would add a **fifth** copy and a fifth place an axis can
//! go missing.
//!
//! # What is deliberately NOT here
//!
//! **Axis parsing.** The setup blocks differ *only* in which axes each harness
//! knows how to parse (#291 §2.2), and unifying that is §14.5 step 3, whose gate
//! is `lloom-runs`' `arms.rs` check. Moving it here now would fold a step with
//! design content into one with none — and §14.5's ordering exists precisely so
//! that a regression in the mechanical step is known before any interface
//! changes.
//!
//! **The loading sequence.** `tokenizer` → `weight_files` →
//! `VarBuilder::from_mmaped_safetensors` → `rename_f` → `Model::new` →
//! `Cache::new_with` is identical in all four (#291 §2.2), but it interleaves
//! per-harness axis decisions (`Cache::new_with` takes the resolved axes), so
//! collapsing it is step 3's work and not a byte-identical move.

use candle::{Error, Result};
use lloom::models::lfm2::{Config, Lfm2Config};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Normalize an LFM2.5-VL `config.json` into candle's schema.
///
/// Mirrors ambrogio's `parse_lfm2_config`, because a harness must measure the
/// configuration that actually runs, not a nearby one:
///
/// * the language config is nested under `text_config`,
/// * `rope_theta` lives in `rope_parameters` and candle would otherwise default
///   it to 10000 where this checkpoint uses 1e6,
/// * candle recomputes `intermediate_size` as 8192 while the FFN weights are
///   `[10752, 2048]`, so the stated value has to win.
///
/// Both of the last two change how much memory a token moves, so both matter to
/// a bandwidth argument as well as to a digest.
pub fn parse_config(raw: &str) -> Result<Config> {
    let root: Value =
        serde_json::from_str(raw).map_err(|e| Error::msg(format!("parsing config.json: {e}")))?;
    let text = root.get("text_config").unwrap_or(&root);
    let mut obj = text
        .as_object()
        .ok_or_else(|| Error::msg("expected a JSON object for the model config"))?
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
    let base: Lfm2Config = serde_json::from_value(normalized.clone()).map_err(|e| {
        Error::msg(format!(
            "config.json does not match candle's LFM2 config schema: {e}"
        ))
    })?;
    let mut config = base.into_config(false);

    // LFM2.5's `intermediate_size` is non-standard and the schema's derivation
    // does not reproduce it, so the stated value wins where one is given.
    if let Some(stated) = normalized
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    {
        config.intermediate_size = stated;
    }

    Ok(config)
}

/// The tensor names in a safetensors file, read from its header alone.
///
/// Reads the header rather than mapping the file, so this is cheap enough to
/// run over every shard before deciding anything.
pub fn tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(Error::wrap)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(Error::wrap)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len >= 100 * 1024 * 1024 {
        candle::bail!("implausible safetensors header length {header_len}")
    }

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(Error::wrap)?;
    let parsed: Value = serde_json::from_slice(&header).map_err(Error::wrap)?;
    Ok(parsed
        .as_object()
        .ok_or_else(|| Error::msg("safetensors header is not a JSON object"))?
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}

/// Every safetensors shard for a checkpoint, in sorted order.
///
/// Falls back to the single-file layout when there is no index, which is the
/// shape a re-exported checkpoint takes.
pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let raw = std::fs::read_to_string(&index).map_err(Error::wrap)?;
    let parsed: Value = serde_json::from_str(&raw).map_err(Error::wrap)?;
    let map = parsed
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::msg("safetensors index has no weight_map"))?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

/// The newest LFM2.5-VL-3B snapshot in the HF cache, if one is there.
///
/// The HF cache stores blobs as **symlinks**, so this tests for `config.json`
/// existing rather than walking for regular files.
pub fn default_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-3B/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// The process's `(phys_footprint, resident_size)` in bytes from
/// `proc_pid_rusage(RUSAGE_INFO_V6)`, or `None` off macOS or on a failed call.
///
/// Field offsets in `u64` words, read from `<sys/resource.h>`'s `struct
/// rusage_info_v6` rather than recalled: the struct opens with `uint8_t
/// ri_uuid[16]` (2 words), then `ri_user_time`, `ri_system_time`,
/// `ri_pkg_idle_wkups`, `ri_interrupt_wkups`, `ri_pageins`, `ri_wired_size`,
/// `ri_resident_size`, `ri_phys_footprint`. A wrong offset returns a plausible
/// number rather than an error, so a caller sanity-checks the pair against what
/// it knows the process holds.
#[cfg(target_os = "macos")]
pub fn proc_footprint() -> Option<(u64, u64)> {
    const RUSAGE_INFO_V6: i32 = 6;
    const W_RESIDENT: usize = 8;
    const W_PHYS: usize = 9;
    // 64 words = 512 B against the struct's ~472 B, so the call cannot write
    // past the buffer even if a later SDK appends a field.
    const WORDS: usize = 64;
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    let mut buf = [0u64; WORDS];
    // SAFETY: `proc_pid_rusage` writes at most `sizeof(rusage_info_v6)` bytes
    // (~472) into the buffer, and `WORDS * 8 = 512` is larger, so the write is
    // in bounds; the buffer outlives the call.
    let rc = unsafe {
        proc_pid_rusage(
            std::process::id() as i32,
            RUSAGE_INFO_V6,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    if rc != 0 {
        return None;
    }
    Some((buf[W_PHYS], buf[W_RESIDENT]))
}

#[cfg(not(target_os = "macos"))]
pub fn proc_footprint() -> Option<(u64, u64)> {
    None
}
