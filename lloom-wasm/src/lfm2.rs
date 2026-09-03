//! Real LFM2 forward pass — the fuller notebook's load-bearing cell (#623).
//!
//! The #656 spike proved lloom's CPU backend runs in a browser on a fixed
//! micro-op. This runs the *actual model*: a quantized LFM2 checkpoint loaded
//! from GGUF bytes, one forward pass, one greedily-sampled real token. The same
//! [`decode_one`] runs native and in wasm.
//!
//! The oracle is the sampled **token**: native and browser agree on it exactly
//! (the model works the same in both). The **logit fingerprint** is byte-exact
//! within a build and across wasm's scalar/simd128 arms, but differs between
//! native and wasm because the quantized `vec_dot` reduces in a different order
//! (native aarch64 NEON vs the wasm path) — the §2.3.5a distinction between "same
//! result" and "same arithmetic", and expected, not a defect.
//!
//! Why quantized: a raw bf16 LFM2 checkpoint is ~5 GB, over the wasm32 4 GiB
//! linear-memory ceiling before activations. A Q4_K_M GGUF of LFM2.5-350M is
//! ~219 MB and fits with room to spare — the only real-LFM2-in-browser path, and
//! the one candle's own browser examples take for every model they run.
//!
//! The tokenizer comes from the GGUF itself, so one file is self-contained.
//! `lloom-tensor`'s `TokenizerFromGguf` now builds on wasm too (its `tokenizers`
//! dep falls to the `unstable_wasm` feature there instead of the `onig` C
//! library), so the same construction runs native and in the browser.

use lloom::models::quantized_lfm2::ModelWeights;
use lloom_tensor::quantized::gguf_file;
use lloom_tensor::quantized::tokenizer::TokenizerFromGguf;
use lloom_tensor::{Device, Tensor};
use std::io::Cursor;
use tokenizers::Tokenizer;

/// The outcome of one real LFM2 forward pass, in a form both the native binary
/// and the wasm export can fingerprint identically.
pub struct Decoded {
    /// The greedily-sampled next-token id.
    pub token_id: u32,
    /// The decoded token text (may be a byte-level fragment).
    pub token_text: String,
    /// The prompt token ids that were fed to the forward pass.
    pub prompt_ids: Vec<u32>,
    /// The top-k `(token_id, token_text, logit)` triples, highest first — the
    /// rich-display payload (#623 point 3) and the audit surface for the argmax.
    pub topk: Vec<(u32, String, f32)>,
    /// FNV-1a over the full f32 logit-vector bit patterns. Byte-exact and
    /// order-sensitive within a build; see the module docs on why native and
    /// wasm fingerprints differ while the sampled token does not.
    pub logits_fp: u64,
    /// The prompt as fed (echoed back for display / reproducibility).
    pub prompt: String,
}

/// Load a quantized LFM2 checkpoint from GGUF bytes, tokenize `prompt` with the
/// GGUF's own embedded tokenizer, run one forward pass, and greedily sample.
///
/// `gguf_bytes` is the whole `.gguf` file. On wasm it arrives from a JS `fetch`;
/// natively it is the file read from disk. Everything after that is
/// target-agnostic, which is what makes the oracle check meaningful.
pub fn decode_one(gguf_bytes: Vec<u8>, prompt: &str, topk: usize) -> lloom_tensor::Result<Decoded> {
    let device = Device::Cpu;

    let mut cursor = Cursor::new(gguf_bytes);
    let content = gguf_file::Content::read(&mut cursor)
        .map_err(|e| lloom_tensor::Error::Msg(format!("reading GGUF: {e}")))?;

    // Build the tokenizer before `from_gguf` consumes `content`.
    let tokenizer = Tokenizer::from_gguf(&content)?;

    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| lloom_tensor::Error::Msg(format!("tokenizing: {e}")))?
        .get_ids()
        .to_vec();
    if prompt_ids.is_empty() {
        return Err(lloom_tensor::Error::Msg(
            "prompt tokenized to nothing".into(),
        ));
    }

    let mut model = ModelWeights::from_gguf(content, &mut cursor, &device)?;

    // One prefill over the whole prompt; `forward` returns logits for the last
    // position only (`quantized_lfm2.rs:626`), so a single call is the whole
    // "next token" computation.
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.to_dtype(lloom_tensor::DType::F32)?;
    let logits_v: Vec<f32> = logits.to_vec1()?;

    let logits_fp = fingerprint_f32(&logits_v);
    let (token_id, top_ids) = argmax_topk(&logits_v, topk.max(1));

    let decode = |id: u32| -> String { tokenizer.decode(&[id], false).unwrap_or_default() };
    let token_text = decode(token_id);
    let topk = top_ids
        .into_iter()
        .map(|(id, logit)| (id, decode(id), logit))
        .collect();

    Ok(Decoded {
        token_id,
        token_text,
        prompt_ids,
        topk,
        logits_fp,
        prompt: prompt.to_string(),
    })
}

/// A cheap order-sensitive fingerprint of raw f32 bit patterns (FNV-1a), the
/// same construction #656 used for its oracle check.
pub fn fingerprint_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for byte in x.to_bits().to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn argmax_topk(logits: &[f32], k: usize) -> (u32, Vec<(u32, f32)>) {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<(u32, f32)> = idx
        .iter()
        .take(k)
        .map(|&i| (i, logits[i as usize]))
        .collect();
    let argmax = idx.first().copied().unwrap_or(0);
    (argmax, top)
}
