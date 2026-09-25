//! The request seam the server and the chat REPL speak: one trait per served
//! family, `dyn` at the request level (RESTRUCTURING §4, `DESIGN.md` §15.2
//! #9 — one indirect call per request; the per-token loop behind each method
//! is `DecodeSession`'s, generic and monomorphized).
//!
//! Two implementors: the LFM2 engine (`lfm2.rs`, every arm the serving
//! binary had — the crown ring, DSpark, `--batch N`) and the Qwen4-exp
//! engine (`qwen4exp.rs`, the single-sequence path over the resident or the
//! streamed expert tier, MTP behind `--mtp`). A family that cannot serve a
//! shape refuses it with the reason (`check_batch`), never serves it wrong.

use super::common::{DecodeTiming, PrefillTiming};
use crate::{DecodeParams, Decoded, ReadbackArm};
use anyhow::Result;
use lloom_tensor::Device;
use serde_json::Value;

pub trait Served: Send + Sync {
    /// The device the model decodes on.
    fn device(&self) -> &Device;
    /// The `model` id `/v1/models` reports.
    fn model_name(&self) -> &str;
    /// KV positions per request; a request past it is refused.
    fn kv_capacity(&self) -> usize;
    /// The readback arm requests take (resolved per family: the LFM2 crown
    /// ring under `auto`; the blocking arm on Qwen4-exp, whose forward needs
    /// its input ids host-side).
    fn readback(&self) -> ReadbackArm;
    fn prefill_tile(&self) -> usize;
    /// The batch width `N`; `1` is the single-sequence path.
    fn batch(&self) -> usize;
    /// The ids a run stops on (the checkpoint's EOS set).
    fn eos_ids(&self) -> &[u32];
    /// The BOS the template starts with, if the checkpoint has one.
    fn bos_id(&self) -> Option<u32>;
    /// The family-specific half of `/health` (tile, chunk, rows, the pool,
    /// the streamed tier's counters …).
    fn health(&self) -> Value;
    /// The family-specific tail of the `[loaded …]` line.
    fn describe(&self) -> String;
    /// What the family can do (`family.rs`): the loaded line and `/health` say it.
    fn capabilities(&self) -> super::family::Capabilities;
    /// The checkpoint's `generation_config.json`, when it carries one (the
    /// sampling defaults under `--sampling-defaults checkpoint`).
    fn generation_config(&self) -> Option<lloom::hub::GenerationConfig> {
        None
    }

    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>>;
    fn token_str(&self, id: u32) -> String;
    fn render_chat(&self, messages: &Value, kwargs: Option<&Value>) -> Result<String>;
    /// The benchmark seed tiled to `n` ids, `row` seed tokens in.
    fn seed_prompt_row(&self, n: usize, row: usize) -> Vec<u32>;

    /// One decode over the whole token sequence, streaming fragments; the
    /// hook's `false` ends the run (`Stop::Cancelled`).
    fn complete_until(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded>;

    fn complete(
        &self,
        prompt_ids: &[u32],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) + Send),
    ) -> Result<Decoded> {
        self.complete_until(prompt_ids, params, &mut |frag| {
            on_fragment(frag);
            true
        })
    }

    fn prefill_throughput(&self, n: usize, batch: usize) -> Result<PrefillTiming>;
    fn decode_throughput(
        &self,
        prompt_tokens: usize,
        decode_tokens: usize,
        batch: usize,
    ) -> Result<DecodeTiming>;
    fn next_token_logprobs(&self, prompt_ids: &[u32], k: usize) -> Result<(u32, Vec<(u32, f32)>)>;
    /// A cell's `batch` must fit what the process was started with.
    fn check_batch(&self, batch: usize) -> Result<()>;
    /// Stop whatever the engine runs on its own threads.
    fn shutdown(&self);
}
