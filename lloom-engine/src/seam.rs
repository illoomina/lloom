//! The serving request seam — the per-request boundary of the engine.
//!
//! RESTRUCTURING §4 names "the request seam" as the serving engine's public
//! surface, with `dyn` allowed here because it is per **request**, not per token
//! (`DESIGN.md` §15.2 #9). What §4 sketched as moving verbatim was amber's
//! `trait Llm`, whose signature is the assistant's conversation vocabulary
//! (`Message`/`ToolCall`/`ToolSpec`) and whose `Arc<dyn Llm>` the assistant's
//! tool loop holds. That trait stays in amber (see this PR's finding): the
//! engine's request seam speaks tokens and text, and amber's `Llm` is the
//! adapter that renders a prompt and parses tools around it.
//!
//! So this trait is the seam's engine-side half: given an already-rendered
//! prompt (a token string) and decode parameters, stream text fragments. It
//! carries no chat template, no tool protocol — those are the consumer's.

use crate::decode::{DecodeParams, Decoded};
use lloom_tensor::Result;

/// A model that serves a decode request over a rendered prompt.
///
/// `dyn`-safe on purpose: a server holds `Arc<dyn Completer>` and pays one
/// indirect call per request, which is where §15.2 #9 permits it. The hot
/// per-token loop it drives ([`crate::decode::DecodeSession::decode`]) is
/// generic and monomorphized — the `dyn` never reaches the token path.
///
/// The seam speaks strings on both sides: the prompt is already rendered
/// (encoding is the implementor's, so the trait names no tokenizer library —
/// the version that bit the extraction stays behind the impl) and the output is
/// text fragments.
pub trait Completer: Send + Sync {
    /// Decode `prompt`, streaming each completed text fragment to `on_fragment`.
    /// `prompt` is already rendered — a chat template is the consumer's concern.
    fn complete_stream(
        &self,
        prompt: &str,
        params: &DecodeParams,
        on_fragment: &mut dyn FnMut(&str),
    ) -> Result<Decoded>;

    /// Decode `prompt` to completion without streaming. The default collects
    /// [`Self::complete_stream`] so an implementor writes one method.
    fn complete(&self, prompt: &str, params: &DecodeParams) -> Result<Decoded> {
        self.complete_stream(prompt, params, &mut |_| {})
    }
}
