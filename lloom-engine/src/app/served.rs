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
use super::vision::DecodedImage;
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

    /// The text one attached image's `<image>` expands to (its delimiters, its
    /// tiles' token runs). A family or checkpoint without vision refuses.
    fn image_placeholder(&self, image: &DecodedImage) -> Result<String> {
        let _ = image;
        Err(self.no_vision())
    }

    /// [`Self::complete_until`] for a prompt whose image positions take the
    /// `images`' rows, in order. The same refusal as `image_placeholder`.
    fn complete_images_until(
        &self,
        prompt_ids: &[u32],
        images: &[DecodedImage],
        params: &DecodeParams,
        on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<Decoded> {
        let _ = (prompt_ids, images, params, on_fragment);
        Err(self.no_vision())
    }

    /// What the last request loaded and encoded: the model's tensors, whether
    /// the vision tower is resident, the image census. `Null` for a family
    /// without vision.
    fn vision_report(&self) -> Value {
        Value::Null
    }

    /// The refusal a family without vision gives, by the family's own reason.
    fn no_vision(&self) -> anyhow::Error {
        let c = self.capabilities();
        anyhow::anyhow!(
            "{} does not take image input: {}",
            c.family,
            c.vision_missing
        )
    }

    /// The executor arms armed at load, for their engagement reports
    /// (`report::print_executor`); `None` where nothing was armed.
    fn installed(&self) -> Option<&super::report::Installed> {
        None
    }

    /// The family's lines for a verbose run report: the state a turn left
    /// behind that only the family can read (the held sequence's KV, the
    /// pacer's depth against what warmup measured).
    fn diagnostics(&self) -> Vec<String> {
        Vec::new()
    }

    /// The multi-sequence measurement modes (`lloom bench --mode reuse |
    /// concurrent`): built where the family can drive several sequences of
    /// one model directly; refused by name elsewhere.
    fn run_mode(&self, spec: &super::modes::ModeSpec) -> Result<()> {
        anyhow::bail!(
            "{} has no `lloom bench --mode {}`: the reuse and concurrent modes drive LFM2's \
             sessions, caches and pools directly (its `Served` is the only one that builds them)",
            self.capabilities().family,
            spec.mode.name()
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::family::QWEN4EXP_CAPABILITIES;

    /// A family with no vision, reached only through the seam: every method
    /// the image path could call past the refusal panics.
    struct NoVision(Device);

    impl Served for NoVision {
        fn device(&self) -> &Device {
            &self.0
        }
        fn model_name(&self) -> &str {
            "stub"
        }
        fn kv_capacity(&self) -> usize {
            0
        }
        fn readback(&self) -> ReadbackArm {
            ReadbackArm::Blocking
        }
        fn prefill_tile(&self) -> usize {
            0
        }
        fn batch(&self) -> usize {
            1
        }
        fn eos_ids(&self) -> &[u32] {
            &[]
        }
        fn bos_id(&self) -> Option<u32> {
            None
        }
        fn health(&self) -> Value {
            Value::Null
        }
        fn describe(&self) -> String {
            String::new()
        }
        fn capabilities(&self) -> super::super::family::Capabilities {
            QWEN4EXP_CAPABILITIES
        }
        fn encode(&self, _: &str, _: bool) -> Result<Vec<u32>> {
            unreachable!("the refusal comes first")
        }
        fn token_str(&self, _: u32) -> String {
            unreachable!()
        }
        fn render_chat(&self, _: &Value, _: Option<&Value>) -> Result<String> {
            unreachable!()
        }
        fn seed_prompt_row(&self, _: usize, _: usize) -> Vec<u32> {
            unreachable!()
        }
        fn complete_until(
            &self,
            _: &[u32],
            _: &DecodeParams,
            _: &mut (dyn FnMut(&str) -> bool + Send),
        ) -> Result<Decoded> {
            unreachable!("an image request must not reach the text path")
        }
        fn prefill_throughput(&self, _: usize, _: usize) -> Result<PrefillTiming> {
            unreachable!()
        }
        fn decode_throughput(&self, _: usize, _: usize, _: usize) -> Result<DecodeTiming> {
            unreachable!()
        }
        fn next_token_logprobs(&self, _: &[u32], _: usize) -> Result<(u32, Vec<(u32, f32)>)> {
            unreachable!()
        }
        fn check_batch(&self, _: usize) -> Result<()> {
            unreachable!()
        }
        fn shutdown(&self) {}
    }

    /// A family without vision refuses an image at the seam, by its own
    /// reason, before anything is rendered or decoded.
    #[test]
    fn a_family_without_vision_refuses_an_image_by_name() {
        let engine = NoVision(Device::Cpu);
        let image = DecodedImage::from_rgb8(1, 1, vec![0, 0, 0]);
        let want = "qwen4exp does not take image input: no vision tower is built for Qwen4-exp";
        let err = engine.image_placeholder(&image).unwrap_err().to_string();
        assert!(err.starts_with(want), "{err}");
        let err = engine
            .complete_images_until(&[1, 2], &[image], &DecodeParams::default(), &mut |_| true)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with(want), "{err}");
        assert!(engine.vision_report().is_null());
    }
}
