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

use lloom::generation::poll::PollMechanism;
use lloom::generation::readback::RING_HISTORY_SLOTS;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::nn::{
    CacheReset, CacheRewind, CausalLM, DecodeOptions, DecodeOut, DecodeStats, IdentityProposer,
    ReuseAction, Sequence, Session,
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
    /// The consumer's fragment hook ended the run ([`DecodeSession::decode_until`]).
    Cancelled,
}

/// Which readback strategy a run takes — where the per-token round trip sits
/// (`design/docs/decode-driver-design.md`, seam A; `readback.rs`).
///
/// The serving crate used to hard-code [`Blocking`], the arm that reads every
/// token back to the host and waits (§11.5a's 441 µs seam). The crown decode
/// path is the device-resident [`Ring`] polled for EOS with one forward of
/// lookahead (§11.5g, §11.2b), which `lfm2-chat` selects by hand; this enum is
/// how a consumer selects it through the seam instead of re-writing that
/// selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackArm {
    /// The ring when the sampler can ride it (argmax or plain temperature),
    /// the blocking arm otherwise — the choice [`Ring::check_sampler`]
    /// would refuse is made here rather than failing the request.
    Auto,
    /// The host reads every token back; any sampler, no lookahead.
    Blocking,
    /// The device-resident token ring with the EOS flag polled.
    Ring {
        /// Forward passes the host enqueues before it has learned the
        /// previous step's EOS: `0` or `1`. `1` is the crown arm and leaves
        /// the cache one position past an EOS, which [`Session::prepare`]
        /// reconciles or resets on the next request.
        lookahead: usize,
        /// Steps between commits of the in-flight queue; `1` is per-token.
        commit_steps: usize,
    },
}

impl ReadbackArm {
    /// The crown selection: ring, one forward of lookahead, commit every 8
    /// steps (`lfm2-chat`'s defaults, #520's flat queue bound).
    pub const CROWN: ReadbackArm = ReadbackArm::Ring {
        lookahead: 1,
        commit_steps: 8,
    };

    /// Whether the ring arm can serve `sampling` without a host round trip
    /// it would not report (`Ring::check_sampler`).
    fn ring_admits(sampling: &Sampling) -> bool {
        matches!(
            sampling,
            Sampling::ArgMax | Sampling::All { .. } | Sampling::GumbelSoftmax { .. }
        )
    }

    /// The arm that runs for `sampling`: `Auto` resolves here, the explicit
    /// arms pass through.
    fn resolve(self, sampling: &Sampling) -> ReadbackArm {
        match self {
            ReadbackArm::Auto if Self::ring_admits(sampling) => ReadbackArm::CROWN,
            ReadbackArm::Auto => ReadbackArm::Blocking,
            other => other,
        }
    }

    /// The arm's name for a report line: `blocking` or `ring-nonblocking`
    /// with its lookahead and commit period, rendered from the resolved value
    /// rather than from the request (§7.1a-i).
    pub fn name(self) -> String {
        match self {
            ReadbackArm::Auto => "auto".to_string(),
            ReadbackArm::Blocking => "blocking".to_string(),
            ReadbackArm::Ring {
                lookahead,
                commit_steps,
            } => format!("ring-nonblocking lookahead={lookahead} commit_steps={commit_steps}"),
        }
    }
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
    /// Which readback arm the run takes; [`ReadbackArm::Auto`] by default.
    pub readback: ReadbackArm,
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
            readback: ReadbackArm::Auto,
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
    /// The readback arm that ran, resolved (never [`ReadbackArm::Auto`]).
    pub readback: ReadbackArm,
    /// The loop's own counters — prefill submit time, per-step host time,
    /// commits, ring engagement — so a consumer that reports throughput reads
    /// the loop's clock rather than bracketing the call.
    pub stats: DecodeStats,
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

    /// Return the sequence to empty: cache reset, token log cleared. A serving
    /// layer that must not reuse state between requests — a benchmark cell
    /// that measures a cold prefill — calls this before each one; a layer that
    /// wants prefix reuse does not, and [`Session::prepare`] decides.
    pub fn reset(&mut self)
    where
        M::Cache: CacheReset,
    {
        self.session.reset()
    }

    /// Compile the pipelines the first real forward would otherwise compile
    /// inside its own timed window, then reset ([`Session::warmup`]). Returns
    /// the device's `(libraries, pipelines)` cache sizes.
    pub fn warmup(&mut self) -> Result<(usize, usize)>
    where
        M::Cache: CacheReset,
    {
        self.session.warmup(&self.device)
    }

    /// The device this session decodes on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The target this session drives.
    pub fn model(&self) -> &'m M {
        self.session.model()
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
        detokenize: impl FnMut(&[u32]) -> Result<String>,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str),
    ) -> Result<Decoded>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        self.decode_until(prompt_ids, detokenize, params, |frag| {
            on_fragment(frag);
            true
        })
    }

    /// [`Self::decode`] whose fragment hook can end the run: a `false` stops
    /// after the fragment it was handed, reported as [`Stop::Cancelled`]. The
    /// serving layer's cancellation — a client that went away, a benchmark's
    /// early abort on a degenerate sample — rides the same seam a stop string
    /// does, so the loop needs no second path.
    pub fn decode_until(
        &mut self,
        prompt_ids: &[u32],
        mut detokenize: impl FnMut(&[u32]) -> Result<String>,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool,
    ) -> Result<Decoded>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::decode: eos_ids is empty; a run needs an EOS set");
        }
        let prepared = self.session.prepare(prompt_ids)?;

        let mut sampler = LogitsProcessor::new(params.seed, params.temperature, params.top_p);
        let arm = params.readback.resolve(sampler.sampling());
        let mut spec = IdentityProposer;

        let mut generated: Vec<u32> = Vec::with_capacity(params.max_tokens);
        let mut text = String::new();
        // Start of the run of generated tokens not yet emitted as text.
        let mut pending = 0usize;
        let mut stopped_on_string = false;
        let mut cancelled = false;

        // The streaming hook, shared by every arm: decode only the
        // not-yet-emitted tokens (the whole sequence each step is quadratic in
        // output length), hold a run ending mid-character, and let a stop
        // string end the run.
        let mut on_tokens = |batch: &[u32]| -> Result<bool> {
            generated.extend_from_slice(batch);
            let chunk = detokenize(&generated[pending..])?;
            if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                let go_on = on_fragment(&chunk);
                text.push_str(&chunk);
                pending = generated.len();
                if !go_on {
                    cancelled = true;
                    return Ok(false);
                }
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
        };

        let out: DecodeOut = match arm {
            ReadbackArm::Blocking => {
                let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
                let opts = DecodeOptions::new(params.max_tokens);
                self.session.decode_until(
                    &mut spec,
                    &mut readback,
                    &mut sampler,
                    &prepared.suffix,
                    &opts,
                    &mut on_tokens,
                )?
            }
            ReadbackArm::Ring {
                lookahead,
                commit_steps,
            } => {
                // Sized for the whole generation; past `RING_HISTORY_SLOTS` the
                // history is no longer CPU-mapped and each drain synchronises
                // (`readback.rs`), which is a slower run and not a refusal.
                let capacity = params.max_tokens.saturating_add(1).max(RING_HISTORY_SLOTS);
                let mut readback = Ring::new(
                    &params.eos_ids,
                    &self.device,
                    capacity,
                    EosRead::Poll(PollMechanism::default()),
                )?;
                let opts = DecodeOptions {
                    max_tokens: params.max_tokens,
                    budget: if commit_steps <= 1 {
                        CommitBudget::per_token()
                    } else {
                        CommitBudget::steps(commit_steps)?
                    },
                    lookahead,
                    sample_split: false,
                    // 0 = the pacer's floor (no warmup ran here) and the one-shot
                    // prefill: the incumbent behaviour on both dials.
                    in_flight_depth: 0,
                    prefill_ubatch: 0,
                };
                self.session.decode_until(
                    &mut spec,
                    &mut readback,
                    &mut sampler,
                    &prepared.suffix,
                    &opts,
                    &mut on_tokens,
                )?
            }
            ReadbackArm::Auto => unreachable!("resolved above"),
        };

        let stop = if cancelled {
            Stop::Cancelled
        } else if stopped_on_string {
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
            readback: arm,
            stats: out.stats,
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

    /// `Auto` takes the crown ring for the device-resident samplers and the
    /// blocking arm for the ones that need a host sort (`Ring::check_sampler`),
    /// and never leaves `Auto` in the resolved value a consumer reports.
    #[test]
    fn auto_resolves_by_sampler() {
        let argmax = Sampling::ArgMax;
        let temp = Sampling::All { temperature: 0.6 };
        let top_p = Sampling::TopP {
            p: 0.9,
            temperature: 0.6,
        };
        assert_eq!(ReadbackArm::Auto.resolve(&argmax), ReadbackArm::CROWN);
        assert_eq!(ReadbackArm::Auto.resolve(&temp), ReadbackArm::CROWN);
        assert_eq!(ReadbackArm::Auto.resolve(&top_p), ReadbackArm::Blocking);
        assert_eq!(
            ReadbackArm::Blocking.resolve(&argmax),
            ReadbackArm::Blocking
        );
        let ring = ReadbackArm::Ring {
            lookahead: 0,
            commit_steps: 1,
        };
        assert_eq!(ring.resolve(&top_p), ring);
        assert_ne!(ReadbackArm::CROWN.name(), ReadbackArm::Blocking.name());
        assert!(ReadbackArm::CROWN.name().contains("lookahead=1"));
    }

    /// The stop reasons a caller matches on are exhaustive and distinct.
    #[test]
    fn stop_variants_are_distinct() {
        assert_ne!(Stop::Eos, Stop::Length);
        assert_ne!(Stop::StopString, Stop::Length);
        assert_ne!(Stop::Eos, Stop::StopString);
        assert_ne!(Stop::Cancelled, Stop::StopString);
    }
}
