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

    /// Make `cache` ready for a pass of `seq_len` tokens — every session
    /// allocation the pass would otherwise make on entry (a KV slab, a
    /// bucket grow), so that a planned pass allocates nothing of its own
    /// (`DESIGN.md` §9.2aa). A driver calls it before its pass bracket;
    /// `forward` may call it again and find nothing to do. The default is
    /// a no-op for models with no such state.
    fn prepare_pass(&self, _cache: &mut Self::Cache, _seq_len: usize) -> Result<()> {
        Ok(())
    }

    fn vocab_size(&self) -> usize;
}

/// One sequence's slot in a batched decode step: where its cache is and the
/// position the step writes at ([`BatchedCausalLM::forward_rows`]).
///
/// The rows of one step are independent sequences — different prompts,
/// different lengths, different caches — so the batch is a slice of these
/// rather than one cache with a batch dimension. Nothing a row holds is
/// shared with its neighbours, which is what makes a row's result independent
/// of them (`DESIGN.md` §2.3.3 #7).
pub struct Row<'a, C> {
    /// The position this step's token is written at; the row's `kv_len`.
    pub pos: usize,
    pub cache: &'a mut C,
}

/// One row of a RAGGED batched forward ([`BatchedCausalLM::forward_ragged`]):
/// `len` consecutive tokens of the flattened `[1, T]` input belong to this
/// row, written at positions `pos..pos + len` of its own cache. A row of
/// length 1 is a decode step and a longer one a prefill piece; they share the
/// forward's weight sweep and nothing else.
pub struct RaggedRow<'a, C> {
    /// The position this row's first token is written at; the row's `kv_len`.
    pub pos: usize,
    /// How many of the input's tokens are this row's, in order.
    pub len: usize,
    pub cache: &'a mut C,
}

/// A model that can decode one token for each of several independent
/// sequences in one forward pass (issue #694, `design/docs/batched-decode-design.md` §10).
///
/// `forward_rows` takes a `[B, 1]` token tensor and one [`Row`] per token and
/// returns `[B, vocab]` logits. Every weight matrix is applied to all `B` rows
/// as one `m = B` GEMM — the 5.394 GB weight sweep is paid once for the batch
/// (§13.5) — while the per-sequence state (KV, conv window, RoPE position) is
/// read and written per row against that row's own cache, so a row computes
/// exactly what a `forward` of that token against that cache computes.
///
/// Kept off [`CausalLM`] for [`CacheReset`]'s reason: the locked trait is not
/// widened, and a model that cannot batch need not implement it. The K=1
/// batched driver ([`Session::decode_batched`](crate::nn::speculator::Session))
/// is bounded on it.
pub trait BatchedCausalLM: CausalLM {
    /// One decode step for `rows.len()` sequences: `tokens` is `[B, 1]`, row
    /// `i` of it is fed to `rows[i]`, and the result is `[B, vocab]` in the
    /// same order. Every row's position advances by one.
    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Self::Cache>]) -> Result<Tensor>;

    /// One forward over a RAGGED batch: `tokens` is `[1, T]`, the rows' token
    /// runs laid end to end (`rows[i].len` each, summing to `T`), run `i` fed
    /// to `rows[i]` at `rows[i].pos`. Returns `[rows, vocab]`, each row's
    /// logits at its LAST token. Every weight matrix is applied to all `T`
    /// tokens in one GEMM, so a prefill piece and the decode rows beside it
    /// ride ONE weight sweep — the fused chunked prefill (vLLM's shape) — and
    /// what is per row runs per row against that row's cache exactly as
    /// [`Self::forward_rows`] and `forward` do. Refused by default; a model
    /// that can feed a ragged batch overrides it.
    fn forward_ragged(
        &self,
        tokens: &Tensor,
        rows: &mut [RaggedRow<'_, Self::Cache>],
    ) -> Result<Tensor> {
        let _ = (tokens, rows);
        lloom_tensor::bail!("this model has no ragged batched forward")
    }
}

/// A cache that can start from another row's cached prefix without
/// recomputing it — pages by reference, the conv state restored — as the
/// rows of one `BatchPool` can (`batched-decode-design.md` §12.3).
pub trait CacheShare {
    /// Share `src`'s first whole pages covering at most `n_tokens` tokens
    /// with this empty cache; returns the tokens shared (0 when none could
    /// be). The caller feeds the rest at that position.
    fn share_prefix_from(&mut self, src: &Self, n_tokens: usize) -> Result<usize>;

    /// The unit a share is rounded down to (the page), in tokens; 0 when
    /// this cache cannot share at all.
    fn share_granularity(&self) -> usize;
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

    /// Discard the last `n` positions that were **never written on the
    /// device**: the forwards that advanced the host's bookkeeping launched
    /// nothing (lloom #735's EOS gate, `DecodeOptions::eos_gate`), so the
    /// cache's contents already end where the sequence does and only the
    /// counters move back. Not bounded by [`Self::rewind_horizon`] — the
    /// horizon protects contents a real rewind would have to restore. A cache
    /// whose per-step bookkeeping cannot be undone this way refuses. The
    /// default is a real rewind.
    fn rewind_unwritten(&mut self, n: usize) -> Result<()> {
        self.rewind(n)
    }
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
