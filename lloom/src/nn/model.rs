//! The engine's model traits: `Model`, `CausalLM`, `Speculative`.
//!
//! These name the entry points a decoder-only language model already exposes,
//! so the serving engine, the VL composition and the speculative verify loop
//! can be written against a seam rather than against `lfm2::Model` by name
//! (`RESTRUCTURING_DESIGN.md` §3.1). They add no computation: an impl wires each
//! method to the model's existing inherent method.
//!
//! `Cache` is an associated type because a real cache is not a shared concrete
//! type — LFM2's bundles KV slots, quant slots, conv state, phase, taps and
//! masks (`RESTRUCTURING_DESIGN.md` §3.1, B8). The trait fixes the *shape* of
//! the entry points, not the state they thread.

use crate::nn::var_builder::VarBuilder;
use lloom_tensor::{Result, Tensor};

/// A model constructible from weights and a typed config.
///
/// `VarBuilder` is `candle-nn`'s, which is why the trait lives here rather than
/// in `candle-core` (`RESTRUCTURING_DESIGN.md` §3.1, B12: the zoo's direction is
/// `candle-transformers → candle-nn → candle-core`).
///
/// `Config` carries no `serde` bound. `RESTRUCTURING_DESIGN.md` §3.1 sketches
/// `type Config: serde::de::DeserializeOwned`, but LFM2's constructor takes a
/// *runtime* `Config` derived from the wire `Lfm2Config` by `into_config`, which
/// computes `intermediate_size` and folds in a caller-supplied `use_flash_attn`
/// — so the runtime config is not the deserialized one and deriving
/// `Deserialize` on it would read the wrong intermediate size. Deserialization
/// is the checkpoint layer's concern (`Checkpoint`/`Pretrained`, §3.6, PR 7),
/// not this trait's; see this PR's body for the reconciliation.
pub trait Model: Sized {
    type Config;

    fn from_weights(vb: VarBuilder, cfg: &Self::Config) -> Result<Self>;
}

/// The LLM domain: an autoregressive model that projects the residual stream to
/// vocabulary logits and threads a cache across positions.
///
/// `forward` is a provided method over `embed` + `forward_embeds` so a caller
/// that builds the residual stream itself — the VL path substituting image rows
/// at the image-token positions (`RESTRUCTURING_DESIGN.md` §5) — needs no second
/// trait. A model whose per-token path must stay byte-identical to its recorded
/// digests overrides `forward` with its hand-written one; the default exists for
/// the models that do not.
pub trait CausalLM: Model {
    /// The state threaded across positions. Bundled per model, not shared: see
    /// the module docs and B8.
    type Cache;

    fn new_cache(&self, cfg: &Self::Config, dev: &lloom_tensor::Device) -> Result<Self::Cache>;

    /// Embed `tokens` to `[b, seq, hidden]` without running a layer.
    fn embed(&self, tokens: &Tensor) -> Result<Tensor>;

    /// Run the model from a residual stream the caller has already built,
    /// projecting the last position to logits.
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor>;

    /// Embed and run in one call. Provided over `embed` + `forward_embeds`;
    /// override only where the per-token path must stay byte-identical to the
    /// recorded digests.
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        let embeds = self.embed(tokens)?;
        self.forward_embeds(&embeds, pos, cache)
    }

    fn vocab_size(&self) -> usize;
}

/// A cache that can be returned to its pre-prefill state.
///
/// The one method [`Session::warmup`](crate::nn::speculator::Session) needs and
/// the locked [`CausalLM`] surface does not carry: a warmup runs a real forward
/// to compile the Metal pipelines the first token would otherwise compile inside
/// the timed window (`DESIGN.md` §13.1i — pipeline creation is ~2 ms cold and
/// ~0.01 ms warm, and the prefill-tile / F32-attention / batched-conv pipelines
/// are prefill-exclusive, so they are all cold on the first real prefill), then
/// resets so the warmup leaves nothing behind. Kept off `CausalLM` because that
/// trait is API-locked (`RESTRUCTURING_DESIGN.md`) and a model that never warms
/// need not implement it; `warmup` is bounded `where M::Cache: CacheReset`
/// rather than widening the locked trait.
pub trait CacheReset {
    /// Return every position-threaded field to its empty, pre-prefill state.
    /// The post-condition the LFM2 impl already tests
    /// (`reset_returns_the_slot_to_its_pre_prefill_state`) is what makes a
    /// warmup non-observable: after it, the next real prefill starts from the
    /// same state it would have without the warmup.
    fn reset(&mut self);
}

/// A cache that can discard its most recent positions outside a speculative
/// window — the sequence-boundary rewind (issue #677).
///
/// An asynchronous decode loop learns EOS after it has enqueued the forwards
/// that consumed it, so the cache ends `n` positions past the last token the
/// sequence keeps. With this seam the overrun is discarded where the sequence
/// ends rather than avoided by decoding conservatively (`DecodeOptions::lookahead
/// = 0`), and a prefix-reusing request that diverges inside the horizon rewinds
/// to the divergence rather than starting from nothing. Kept off [`CausalLM`] for
/// [`CacheReset`]'s reason: the locked trait is not widened, and a cache that
/// cannot rewind reports a horizon of zero rather than being unable to compile.
pub trait CacheRewind {
    /// How many positions [`Self::rewind`] would currently accept. Zero for a
    /// cache with no rewindable arm, and for a rewindable one while a
    /// speculative window is open or nothing has been written since a
    /// multi-token pass reseeded it.
    fn rewind_horizon(&self) -> usize;

    /// Discard the last `n` positions. Refused, before any mutation, past
    /// [`Self::rewind_horizon`].
    fn rewind(&mut self, n: usize) -> Result<()>;
}

/// The verifier's entry and the speculative rollback seam.
///
/// Only a model that can rewind implements it. `forward_all` projects **every**
/// position rather than the last, which is what a verify pass needs: comparing
/// the target's argmax at each of K proposed positions, where one position's
/// logits cannot answer K questions (`lfm2.rs` `forward_all`'s doc).
///
/// Rollback is transactional rather than an absolute `rewind(to)`: `advance`
/// opens a window and returns a token recording the state to return to;
/// `resolve` keeps the accepted prefix and discards the rest (KV by a length
/// decrement, conv by a phase move, `RESTRUCTURING_DESIGN.md` §3.1 /
/// `DESIGN.md` §10.2a). The token is the seam, not a position index — see the
/// finding in this PR's body.
pub trait Speculative: CausalLM {
    /// A handle to an open speculative window, recording the state `resolve`
    /// returns to. Not `Copy`, so a caller cannot hold two windows at once.
    type Window;

    /// Run the model and project every position: `[b, seq, vocab]` where
    /// `forward` returns `[b, vocab]`.
    fn forward_all(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor>;

    /// Open a speculative window of `k` positions. Records the pre-window state;
    /// writes nothing until the K-position pass runs.
    fn advance(cache: &mut Self::Cache, k: usize) -> Result<Self::Window>;

    /// Close the window opened by `advance`, keeping the first `n` positions and
    /// discarding the rest.
    fn resolve(cache: &mut Self::Cache, window: Self::Window, n: usize) -> Result<()>;
}

/// A vision tower that encodes one host-preprocessed image to rows in the
/// language model's embedding space.
///
/// A domain trait added because its one impl exists (`RESTRUCTURING_DESIGN.md`
/// §3.2, §5): `lfm2_vl::VisionEncoder`. It composes with `CausalLM` through
/// `embed`/`forward_embeds` rather than through a second entry point — the host
/// merges the encoded rows into the residual stream at the image-token positions
/// before the prefill, so nothing here runs on the per-token path.
///
/// `Input` is the host stage, not raw pixels: patchification, resizing and
/// normalisation happen on the CPU (`ProcessedImage`), so `encode` takes an
/// already-patchified `[1, n_patches, C*P*P]` sequence with its `(rows, cols)`.
/// The result is `[n_image_tokens, text_hidden]` — one row per image token, in
/// the text model's hidden size, ready for the host-side merge.
///
/// The trait shares its name with `lfm2_vl::VisionEncoder`, the concrete tower
/// it is first implemented for; §5 names both, and they sit in different modules.
pub trait VisionEncoder: Model {
    /// The host-preprocessed image this tower consumes.
    type Input;

    /// Encode one image to `[n_image_tokens, text_hidden]`.
    fn encode(&self, input: &Self::Input) -> Result<Tensor>;
}
