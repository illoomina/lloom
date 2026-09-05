//! The decode loop, generic over `CausalLM` — the serving engine's core.
//!
//! Extracted from amber's `Lfm2::generate_raw` (RESTRUCTURING §4, Wave F): the
//! prompt rendering and tool parsing that wrapped it are the assistant's, but
//! the loop between them — prefill, sample, detokenize, advance — is the
//! engine's. Since #677 the loop itself is `lloom`'s `Session::decode` over the
//! `Blocking` readback strategy (the host-side sampler, every sampling arm), so
//! this crate carries no second loop (§11.5v named the one it used to carry);
//! what it adds is the serving side: the incremental detokenize, the stop
//! strings, and **content-addressed reuse** — a request is a whole token
//! sequence, and `Session::prepare` reuses whatever prefix the session's cache
//! already holds.
//!
//! Static dispatch only: the loop is generic over `M: CausalLM` so the
//! per-token `forward` monomorphizes and LTO inlines it. `DESIGN.md` §15.2 #9 /
//! `CONTRIBUTING.md` §3.6 — no `dyn` on the decode path. The `dyn`-per-request
//! boundary is the [`crate::seam`] trait, one level out, where it is allowed.

use lloom::generation::{Blocking, LogitsProcessor};
use lloom::nn::{
    CacheReset, CacheRewind, CausalLM, DecodeOptions, IdentityProposer, ReuseAction, Sequence,
    Session,
};
use lloom_tensor::{Device, Result};

/// Why a decode run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// A configured stop string appeared in the decoded text.
    StopString,
    /// The token budget was exhausted.
    Length,
}

/// Sampling and stopping knobs for one decode run. Engine-level: token ids and
/// counts, no conversation vocabulary (that is the consumer's — [`crate::seam`]).
#[derive(Debug, Clone)]
pub struct DecodeParams {
    pub seed: u64,
    /// `None` (or `< 1e-7`) is greedy argmax; `Some(t)` samples at temperature
    /// `t`, with `top_p` when set. Matches [`LogitsProcessor::new`].
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: usize,
    /// End-of-sequence ids; the first one sampled stops the run.
    pub eos_ids: Vec<u32>,
    /// Text substrings that stop the run once they appear in the output.
    pub stop_strings: Vec<String>,
}

impl Default for DecodeParams {
    fn default() -> Self {
        DecodeParams {
            seed: 42,
            temperature: None,
            top_p: None,
            max_tokens: 512,
            eos_ids: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

/// The outcome of a decode run: the full text, the tokens generated, why it
/// stopped, and what the request reused of the session's cache.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stop: Stop,
    /// How the request was matched against the session's cached tokens.
    pub reuse: ReuseAction,
    /// Prompt tokens whose state the cache already held and kept.
    pub reused: usize,
    /// Prompt tokens actually prefilled — the request's length minus `reused`,
    /// except under [`ReuseAction::Resume`] where nothing is.
    pub prefilled: usize,
}

/// A decode session: a borrowed target and the sequence it drives — the cache,
/// the ids whose state it holds, and the position. The model is borrowed rather
/// than owned so a session pins no weights alive beyond the run it drives.
///
/// A request is the WHOLE token sequence to continue, not the new tokens: the
/// session reuses the prefix its cache already holds (`Session::prepare`) and
/// prefills the rest. A caller that decodes several turns against one session
/// therefore passes the growing conversation each time and pays only for the
/// new turn; a fresh session with an empty cache prefills everything, which is
/// what every request did before #677.
pub struct DecodeSession<'m, M: CausalLM> {
    session: Session<'m, M>,
    device: Device,
}

impl<'m, M: CausalLM> DecodeSession<'m, M> {
    /// Pair a target with a cache it has already built ([`CausalLM::new_cache`]).
    /// `pos` is the number of positions already prefilled into that cache — 0
    /// for a fresh one. A non-zero `pos` describes a cache whose contents this
    /// session does not know, so the first request resets it (`Session::with_pos`).
    pub fn new(model: &'m M, cache: M::Cache, device: Device, pos: usize) -> Self {
        DecodeSession {
            session: Session::with_pos(model, cache, pos),
            device,
        }
    }

    /// Drive a sequence a serving layer held across requests ([`crate::SlotPool`]).
    pub fn from_sequence(model: &'m M, seq: Sequence<M::Cache>, device: Device) -> Self {
        DecodeSession {
            session: Session::from_sequence(model, seq),
            device,
        }
    }

    /// Give the sequence back, to be held until the next request.
    pub fn into_sequence(self) -> Sequence<M::Cache> {
        self.session.into_sequence()
    }

    /// The cache, for a caller that inspects or seeds it between runs.
    pub fn cache_mut(&mut self) -> &mut M::Cache {
        self.session.cache_mut()
    }

    /// Positions currently in the cache.
    pub fn pos(&self) -> usize {
        self.session.pos()
    }

    /// The ids whose state the cache holds.
    pub fn tokens(&self) -> &[u32] {
        self.session.tokens()
    }

    /// Decode `prompt_ids` — the whole sequence to continue — streaming each
    /// completed text fragment to `on_fragment` as it is produced. Returns the
    /// full decoded run.
    ///
    /// The streaming primitive: a fragment is emitted only when the tokens
    /// since the last emission decode to complete text — a multi-byte character
    /// can span several tokens, so a run ending on a `U+FFFD` replacement char
    /// is held until the next token completes it. This fragment-boundary logic
    /// is the engine's; it lives here now (was amber's).
    ///
    /// `detokenize` maps a run of ids to text — the consumer supplies it so the
    /// engine names no tokenizer library (amber and the engine were on
    /// different `tokenizers` majors; the seam must not couple them). It is
    /// called on the not-yet-emitted tail each step, so it must decode a *slice*
    /// of ids, not assume it starts at position 0.
    ///
    /// `on_fragment` is a `FnMut(&str)` rather than a channel so the engine
    /// stays runtime-agnostic: the consumer wires whatever streaming transport
    /// it uses (amber's `spawn_blocking` + mpsc channel) around this call.
    pub fn decode(
        &mut self,
        prompt_ids: &[u32],
        mut detokenize: impl FnMut(&[u32]) -> Result<String>,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str),
    ) -> Result<Decoded>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::decode: eos_ids is empty; a run needs an EOS set");
        }
        let prepared = self.session.prepare(prompt_ids)?;

        let mut sampler = LogitsProcessor::new(params.seed, params.temperature, params.top_p);
        let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
        let mut spec = IdentityProposer;
        let opts = DecodeOptions::new(params.max_tokens);

        let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
        let mut text = String::new();
        // Start of the run of generated tokens not yet emitted as text.
        let mut pending = 0usize;
        let mut stopped_on_string = false;

        let out = self.session.decode_until(
            &mut spec,
            &mut readback,
            &mut sampler,
            &prepared.suffix,
            &opts,
            &mut |batch: &[u32]| {
                generated.extend_from_slice(batch);
                // Decode only the not-yet-emitted tokens, not the whole sequence
                // each step: the latter is quadratic in output length.
                let chunk = detokenize(&generated[pending..])?;
                if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                    on_fragment(&chunk);
                    text.push_str(&chunk);
                    pending = generated.len();
                }
                if !params.stop_strings.is_empty()
                    && params
                        .stop_strings
                        .iter()
                        .any(|s| text.contains(s.as_str()))
                {
                    stopped_on_string = true;
                    return Ok(false);
                }
                Ok(true)
            },
        )?;

        let stop = if stopped_on_string {
            Stop::StopString
        } else if out.eos {
            Stop::Eos
        } else {
            Stop::Length
        };

        // Trim anything past the earliest stop sequence.
        if let Some(cut) = params
            .stop_strings
            .iter()
            .filter_map(|s| text.find(s.as_str()))
            .min()
        {
            text.truncate(cut);
        }

        Ok(Decoded {
            text,
            tokens: out.ids,
            stop,
            reuse: prepared.action,
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode session drives any `CausalLM`. Reachability the way §3.1c
    /// asks: a real end-to-end run needs weights (that is the `lfm2-smoke`
    /// gate — a `lloom` example, run in this PR's evidence), so what a unit
    /// test proves is that the seam is *callable* against the production model
    /// — if `lfm2::Model` did not satisfy `CausalLM`, this would not compile.
    #[allow(dead_code)]
    fn drives<M: CausalLM>()
    where
        M::Cache: CacheRewind + CacheReset,
    {
    }

    #[test]
    fn decode_session_drives_lfm2() {
        drives::<lloom::models::lfm2::Model>();
    }

    /// The stop reasons a caller matches on are exhaustive and distinct.
    #[test]
    fn stop_variants_are_distinct() {
        assert_ne!(Stop::Eos, Stop::Length);
        assert_ne!(Stop::StopString, Stop::Length);
        assert_ne!(Stop::Eos, Stop::StopString);
    }
}
