//! The DSpark drafter's context projection — `DESIGN.md` §10.2m/§10.2o, issue #363.
//!
//! §10.2n built the target-side hidden-state tap and stopped there, naming the
//! projection as the next increment: it is the **first falsifiable surface**
//! the drafter offers, and it is reachable *before* the backbone exists.
//! This module is that projection, and nothing more.
//!
//! # What it computes
//!
//! `H_ctx = RMSNorm_enc( W_c · [H(l₁); …; H(l₅)] )` — the drafter's `fc.weight`
//! `[2048, 10240]` applied to the five concatenated target hidden states, then
//! the encoder RMSNorm. **The norm is after the matmul**, which is a correction
//! §10.2m made to the paper's `RMSNorm(W_c·[…])` ordering by reading
//! `dflash.cpp`, and which a third witness confirms verbatim — the native MLX
//! port's [`fuse_target`] is one line:
//!
//! ```text
//! def fuse_target(self, target_hidden_cat):        # mlx-dspark, model.py:371
//!     return self.hidden_norm(self.fc(target_hidden_cat))
//! ```
//!
//! Two checkpoint facts make that mapping unambiguous rather than inferred, and
//! both are re-derived from the payload-bearing file by
//! `measurements/issue-363-raw/verify_projection.py`: `fc.weight` is
//! `[2048, 10240] = [hidden, 5 × hidden]` with **no bias**, and the encoder norm
//! `hidden_norm.weight` is `[2048]` — a norm over the *output* width, which is
//! only expressible after the matmul.
//!
//! # Why this is a `matmul` and not the fused prologue §10.2m specifies
//!
//! §10.2m maps this onto `gemv` + `rmsnorm_scale` + `gemv_rmsnorm`, marking the
//! latter two **"exists, exact fit"**, and concludes *"`H_ctx` never
//! materializes in DRAM"* with *"the only new artifacts are `[[host_name]]`
//! instantiations"*. **The kernels exist and the path to them does not** —
//! measured this run, recorded as §10.2o, and the reason this module is written
//! against `Tensor::matmul`:
//!
//! * Every construction site of `GemvRmsNorm` in the tree is in `tests.rs`;
//!   `call_rms_norm_scale` is not even re-exported. Production callers: **0**.
//! * `candle-nn`'s public fused surface is `gemv_silu` and `gemv_residual`
//!   (#509's *epilogues*). There is **no `gemv_rmsnorm`** — the *prologue* has
//!   no public entry point, so a tensor-level caller cannot request it.
//! * `Tensor::matmul` reaches Metal through exactly one door,
//!   `call_mlx_gemm` (`candle-core/src/metal_backend/mod.rs:1928`), and that
//!   function **takes no `fused_norm` parameter at all**. Its GEMV branch
//!   delegates to `call_mlx_gemv_with`, which passes `None, GemvEpilogue::None`
//!   hardcoded (`mlx_gemm.rs:668`).
//!
//! And the shape half is independent of the wiring half. `candle-nn` states it
//! at `ops.rs:2358`: *"a GEMV is the only route that carries a fused epilogue —
//! `takes_gemv_route` is `m == 1 || n == 1` … an `m != 1` request would silently
//! take the GEMM path and run unfused."* §10.2m sets this projection's
//! `M = accepted_len`, 1..n_max, and calls it *"deep skinny-M"*. **Skinny-M is a
//! GEMM regime; the fused prologue is a GEMV mechanism** — at `M ≥ 2` no
//! instantiation count reaches it, so §10.2m's named repair does not address the
//! defect. Fusing this is real work on `call_mlx_gemm`, not a new caller, and it
//! is deliberately **not** attempted here: a fusion measured against a consumer
//! that does not exist is #500's shape (§3.1c), and the consumer is the backbone.
//!
//! What is lost by materializing is the DRAM round-trip §12.1 names, on a
//! `[M, 2048]` F32 tensor — 8 KiB per position, against the 40 MB `W_c` read
//! that produces it. What is gained is a projection that **runs**, on every
//! backend, with an absolute-reference parity test.
//!
//! # The representation: the OUTPUT residual of `target_layer_ids`
//!
//! The tap mechanism captures the **layer-input residual** (§10.2n) — llama.cpp
//! PR #27383's `t_layer_inp[il]`, read before layer `il` runs. The drafter's
//! `target_layer_ids` `[2, 9, 17, 21, 27]` name the layers whose **output** it
//! was trained on, so the layer-input capture has to be requested at `l + 1`:
//! llama.cpp's converter writes `dflash.target_layers = [i + 1 for i in
//! target_layer_ids]` (`conversion/qwen.py`), the released GGUF carries
//! `[3, 10, 18, 22, 28]`, and mlx-dspark captures after the block with the
//! indices unshifted. The two witnesses agree; [`DSparkConfig::tap_layers`] is
//! that `+ 1`, and requesting `target_layer_ids` directly is one layer early
//! (§10.2s measured the cost: a different drafted block from slot 1 onward).
//!
//! This module projects whatever taps it is given. The producer's order and
//! layer choice are the caller's contract, checked where they are made.

use crate::models::with_tracing::{Embedding as TracedEmbedding, Linear as TracedLinear};
use crate::nn::speculator::{Draft, DraftTokens, KvSource, LayerTaps, SeqId, Speculator};
use crate::nn::{Linear, RmsNorm, VarBuilder};
#[cfg(feature = "metal")]
use lloom_tensor::kernels::metal::Epoch;
use lloom_tensor::{DType, Device, IndexOp, Module, Result, Tensor, D};
#[cfg(feature = "metal")]
use parking_lot::Mutex;
use std::sync::Arc;

/// The drafter's context projection: `RMSNorm_enc(W_c · concat)`.
///
/// Holds only what the projection needs — the drafter's `fc` weight and its
/// encoder norm. The backbone, Markov head and confidence head are §10.2m's
/// remaining increments and deliberately absent: this type is the projection,
/// so a caller that has one cannot believe it has a drafter.
#[derive(Debug)]
pub struct DSparkProjection {
    /// `fc.weight`, `[hidden, n_tap × hidden]`, **no bias** (checkpoint fact —
    /// the safetensors header carries `fc.weight` and no `fc.bias`, and
    /// mlx-dspark constructs `nn.Linear(…, bias=False)`).
    fc: Tensor,
    /// `fc` at the taps' dtype, transposed and contiguous — the operand
    /// `forward`'s matmul actually reads. Built on the first call and kept:
    /// the taps are F32 and the checkpoint is not, so rebuilding it per round
    /// re-casts and re-copies the whole 42 MB weight every speculative round.
    fc_t: std::sync::OnceLock<Tensor>,
    /// `hidden_norm.weight`, `[hidden]` — the encoder RMSNorm applied *after*
    /// the matmul.
    norm_weight: Tensor,
    eps: f64,
    /// How many target hidden states the concat carries — `target_layer_ids`'
    /// length, 5 for the 2.6B drafter. Kept so `forward` can reject a concat of
    /// the wrong arity by name rather than by a matmul dimension error.
    n_tap: usize,
    hidden: usize,
}

impl DSparkProjection {
    /// Load from the drafter checkpoint's own names.
    ///
    /// `fc.weight` and `hidden_norm.weight` are the tensor names in
    /// `LFM2.5-2.6B-DSpark/model.safetensors`, read from the payload-bearing
    /// file rather than assumed (§10.2n: the earlier sections' constants came
    /// from a committed 6480-byte header blob, and the full 655 421 522 B file
    /// hashes identically over that prefix).
    ///
    /// `hidden` and `n_tap` are the *caller's*, checked against the loaded
    /// shape rather than inferred from it: an `fc.weight` whose column count is
    /// not `n_tap × hidden` means the drafter and the tap disagree about how
    /// many layers are tapped, and that is a configuration error worth a
    /// sentence rather than a shape mismatch three calls later.
    pub fn new(hidden: usize, n_tap: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let fc = vb.get((hidden, n_tap * hidden), "fc.weight")?;
        let norm_weight = vb.get(hidden, "hidden_norm.weight")?;
        Ok(Self {
            fc,
            fc_t: std::sync::OnceLock::new(),
            norm_weight,
            eps,
            n_tap,
            hidden,
        })
    }

    /// Build from tensors already in hand — the fixture path, and the one a
    /// parity test uses to pin an absolute reference.
    pub fn from_parts(fc: Tensor, norm_weight: Tensor, eps: f64) -> Result<Self> {
        let (hidden, cols) = fc.dims2()?;
        if !cols.is_multiple_of(hidden) {
            lloom_tensor::bail!(
                "DSpark fc.weight is [{hidden}, {cols}]: the column count must be a whole \
                 number of {hidden}-wide target hidden states (issue #363)"
            );
        }
        Ok(Self {
            fc,
            fc_t: std::sync::OnceLock::new(),
            norm_weight,
            eps,
            n_tap: cols / hidden,
            hidden,
        })
    }

    /// `W_c` at `dtype`, transposed to `[n_tap × hidden, hidden]` and
    /// contiguous. Cached after the first call; a later call at a different
    /// dtype rebuilds without caching, so one instance serves one dtype fast
    /// and any other correctly.
    fn fc_t(&self, dtype: DType) -> Result<Tensor> {
        if let Some(t) = self.fc_t.get() {
            if t.dtype() == dtype {
                return Ok(t.clone());
            }
        }
        let t = self.fc.to_dtype(dtype)?.t()?.contiguous()?;
        let _ = self.fc_t.set(t.clone());
        Ok(t)
    }

    /// How many target hidden states this projection consumes.
    pub fn n_tap(&self) -> usize {
        self.n_tap
    }

    /// The projection's output width, and the target's hidden size.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Project the drained taps: `H_ctx = RMSNorm_enc(W_c · [H(l₁); …; H(l_n)])`.
    ///
    /// `taps` is [`crate::models::lfm2::Cache::take_taps`]' output — one
    /// `[b, seq, hidden]` tensor per tapped layer, **in `tap_layers`' order**,
    /// which is the drafter's own `target_layer_ids` order. That order is
    /// load-bearing and is the defect §10.2n caught in its own build: a
    /// permuted concat pairs every 2048-d slice with the wrong 2048-column
    /// block of `W_c`, is algebraically valid, and is invisible to every gate
    /// in this tree. **This function does not re-sort**, deliberately — the
    /// order is the producer's contract, and silently repairing it here would
    /// take the invariant out of the one place a test can hold it.
    ///
    /// Returns `[b, seq, hidden]` at the taps' own dtype (F32 per
    /// `Config::tap_dtype`, which §10.2m argues buys acceptance rate rather
    /// than correctness).
    pub fn forward(&self, taps: &[Tensor]) -> Result<Tensor> {
        if taps.len() != self.n_tap {
            lloom_tensor::bail!(
                "DSpark projection expects {} taps and got {} — `fc.weight` is \
                 [{}, {}] and consumes exactly {} concatenated hidden states \
                 (issue #363, DESIGN.md 10.2m)",
                self.n_tap,
                taps.len(),
                self.hidden,
                self.n_tap * self.hidden,
                self.n_tap
            );
        }
        for (i, t) in taps.iter().enumerate() {
            let (_, _, h) = t.dims3()?;
            if h != self.hidden {
                lloom_tensor::bail!(
                    "DSpark projection tap {i} is {h} wide, expected {} (issue #363)",
                    self.hidden
                );
            }
        }

        // Concatenate along the feature axis, which is what `fc.weight`'s
        // 10240 columns are: `[H(l1); ...; H(l5)]`, not a token-stream concat.
        // mlx-dspark's `_body_lfm2` does the same with `axis=-1`.
        let cat = Tensor::cat(taps, lloom_tensor::D::Minus1)?;

        // `W_c` is stored `[out, in]` and `matmul` wants `[in, out]` on the
        // right, so the transpose is the weight layout's, not a choice --
        // `Linear::forward` computes `x @ w.t()` for exactly this reason.
        // Cast to the activations' dtype rather than the reverse: the taps are
        // F32 by `Config::tap_dtype` and downcasting them here would discard
        // the precision §10.2m says is bought deliberately.
        let w = self.fc_t(cat.dtype())?;
        let fc_out = cat.broadcast_matmul(&w)?;

        // The norm is AFTER the matmul. See the module docs: this is 10.2m's
        // correction to the paper, and mlx-dspark's one-line `fuse_target`
        // states the same order independently.
        let norm_weight = self.norm_weight.to_dtype(fc_out.dtype())?;
        crate::nn::ops::rms_norm(&fc_out, &norm_weight, self.eps as f32)
    }
}

/// The projection's parameter count, for the memory accounting §9.1 wants.
///
/// `fc` dominates: `2048 × 10240 = 20 971 520` against the norm's 2048.
pub fn projection_params(hidden: usize, n_tap: usize) -> usize {
    hidden * n_tap * hidden + hidden
}

/// The dtype the drafter's weights are stored at in the released checkpoint.
///
/// **BF16**, read from the safetensors header — every one of the 62 tensors,
/// including `fc.weight` and `hidden_norm.weight`. Recorded because §10.2l and
/// §10.2m give the drafter's shapes and constants without naming its storage
/// dtype, and a loader that assumed F16 would read the right bytes as the wrong
/// numbers (§3.1c: count the artifact).
pub const CHECKPOINT_DTYPE: DType = DType::BF16;

// ---------------------------------------------------------------------------
// The backbone — §10.2p, issue #363 increment 3.
// ---------------------------------------------------------------------------

/// The drafter's constants, as its own `config.json` carries them.
///
/// Every field is read from
/// `LFM2.5-2.6B-DSpark/config.json` rather than defaulted, because the two
/// values a wrong default would silently change — `rope_theta` and
/// `rope_is_neox_style` — are exactly the two that alter no shape and no
/// dispatch count. §5.11 records the target-side form of that trap: *"a wrong
/// theta changes no shape and no dispatch count, so every gate in this document
/// except the text one would stay green while the model produced fluent wrong
/// output"*. On the drafter there is not even a text gate, because the target
/// verifies every token — so a wrong theta here is invisible to **every** gate
/// and presents only as a lower acceptance rate.
#[derive(Debug, Clone, PartialEq)]
pub struct DSparkConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// **false** on this checkpoint — the drafter runs *interleaved* RoPE where
    /// the LFM2 target runs neox. §10.2l item 3 names it as *"a draft-side
    /// convention a build must express, not a cast"*, and the two sides of the
    /// seam are now both read: `lfm2.rs`'s `apply_rotary_emb` calls
    /// `crate::nn::rotary_emb::rope` (neox) and this calls `rope_i`.
    pub rope_is_neox_style: bool,
    /// γ — the draft block width, 9 here.
    pub block_size: usize,
    /// The id whose embedding fills every block slot after the anchor.
    pub mask_token_id: u32,
    /// The target layers whose **output** residuals the projection consumes,
    /// **in this order** (§10.2n: the order is the consumer's contract). The
    /// layer-input tap that delivers them is requested at [`Self::tap_layers`].
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub markov_rank: usize,
    pub enable_confidence_head: bool,
    /// Which block slot holds the first *prediction*.
    ///
    /// **0 on this checkpoint** — anchor-as-pos0, so slot 0 both embeds the
    /// known token and predicts the first draft token, and `block_size` 9 yields
    /// **9** predictions. It is not a key in `config.json`; it is a property of
    /// the packaging, and mlx-dspark's loader sets it for this exact
    /// architecture (`"Lfm2DSparkDraftModel"`, ids nested in `dflash_config`,
    /// no `projector_type` tag) with the reasoning recorded there: *"the card's
    /// 'ceiling is 10' = 9 drafts + the target's bonus token, i.e. logits_start
    /// 0 — confirmed by the drafter's own accept length"*.
    ///
    /// **Getting this wrong costs acceptance and nothing else.** mlx-dspark
    /// measured the sibling case on a SpecForge head: *"accept 1.35 as
    /// logits_start=1 vs 3.42 as 0, same prompt and loop"*, and names the
    /// mechanism — *"shifts every draft one slot and collapses acceptance with
    /// no error anywhere"*. §10.2n's class again, from an independent source
    /// and with a number attached.
    pub logits_start: usize,
}

impl DSparkConfig {
    /// The 2.6B drafter's own values, for a caller that has the checkpoint but
    /// not a JSON parser in hand. Every field is re-derived from the committed
    /// `config.json` by `measurements/issue-363-raw/verify_backbone.py`.
    pub fn lfm2_2_6b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            intermediate_size: 6144,
            vocab_size: 128_000,
            rms_norm_eps: 1e-5,
            rope_theta: 1e7,
            rope_is_neox_style: false,
            block_size: 9,
            mask_token_id: 125_017,
            target_layer_ids: vec![2, 9, 17, 21, 27],
            num_target_layers: 30,
            markov_rank: 256,
            enable_confidence_head: true,
            logits_start: 0,
        }
    }

    /// The drafter's constants from its checkpoint directory's `config.json`
    /// (the `lfm2-chat` / `lfm2-dspark-verify` reader, hoisted so a serving
    /// binary loads the same drafter the same way).
    ///
    /// Never defaulted for the two fields a wrong default silently corrupts —
    /// `rope_theta` and `rope_is_neox_style` alter no shape and no dispatch
    /// count, and on a drafter there is not even a text gate to catch them
    /// because the target verifies every token (`speculation.md` §10.2n).
    pub fn read_from_dir(dir: &std::path::Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| lloom_tensor::Error::msg(format!("reading {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| lloom_tensor::Error::msg(format!("parsing {}: {e}", path.display())))?;
        let dfc = v.get("dflash_config").cloned().unwrap_or_else(|| v.clone());
        let missing = |k: &str| lloom_tensor::Error::msg(format!("drafter config has no {k}"));
        let usize_at = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .ok_or_else(|| missing(k))
        };
        let target_layer_ids: Vec<usize> = dfc
            .get("target_layer_ids")
            .and_then(|x| x.as_array())
            .ok_or_else(|| missing("dflash_config.target_layer_ids"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        Ok(DSparkConfig {
            hidden_size: usize_at("hidden_size")?,
            num_hidden_layers: usize_at("num_hidden_layers")?,
            num_attention_heads: usize_at("num_attention_heads")?,
            num_key_value_heads: usize_at("num_key_value_heads")?,
            head_dim: usize_at("head_dim")?,
            intermediate_size: usize_at("intermediate_size")?,
            vocab_size: usize_at("vocab_size")?,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-5),
            rope_theta: v
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| missing("rope_theta"))?,
            rope_is_neox_style: v
                .get("rope_is_neox_style")
                .and_then(|x| x.as_bool())
                .ok_or_else(|| missing("rope_is_neox_style"))?,
            block_size: usize_at("block_size")?,
            mask_token_id: dfc
                .get("mask_token_id")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| missing("dflash_config.mask_token_id"))?
                as u32,
            num_target_layers: dfc
                .get("num_target_layers")
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .unwrap_or(30),
            target_layer_ids,
            markov_rank: v
                .get("markov_rank")
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .unwrap_or(0),
            enable_confidence_head: v
                .get("enable_confidence_head")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            // Not a key in config.json — a property of the packaging (§10.2p).
            logits_start: 0,
        })
    }

    /// The `Config::tap_layers` that deliver `target_layer_ids`' output residuals.
    ///
    /// The tap captures the residual **entering** a layer, and the drafter is
    /// trained on the residual **leaving** `target_layer_ids`, so each index is
    /// shifted by one: the input of layer `l + 1` is the output of layer `l`.
    /// This is the same `+ 1` llama.cpp's converter applies when it writes
    /// `dflash.target_layers` (`[3, 10, 18, 22, 28]` in the released GGUF for
    /// `[2, 9, 17, 21, 27]`). Requesting `target_layer_ids` unshifted taps one
    /// layer early and is invisible to every gate but the accept rate (§10.2s).
    pub fn tap_layers(&self) -> Vec<usize> {
        self.target_layer_ids.iter().map(|l| l + 1).collect()
    }

    /// How many tokens one block can propose: `block_size - logits_start`.
    pub fn max_draft(&self) -> usize {
        self.block_size.saturating_sub(self.logits_start)
    }

    /// GQA expansion factor, 4 here (32 query heads over 8 KV heads).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Precomputed `cos`/`sin` for the drafter's **interleaved** RoPE.
///
/// Separate from the target's table rather than shared, because the drafter's
/// `rope_theta` is its own config's (1e7 here — the same value the 2.6B target
/// happens to carry, and *not* the 1e6 §5.2 records for VL-3B). Sharing would
/// make the drafter silently inherit whichever target it is bolted onto, which
/// is the §5.11 trap with the checkpoints swapped.
#[derive(Debug)]
pub struct DSparkRotary {
    cos: Tensor,
    sin: Tensor,
    interleaved: bool,
    /// The batched rows' position index (`at_positions`), resident: the
    /// positions it holds, the tensor, and the epoch of the last gather that
    /// read it — written in place once that epoch completed (the round's
    /// verify readback), a fresh tensor otherwise; the same rule as the
    /// context pool's slot tensors (`paged_ctx::SeqSlots`). Behind a mutex
    /// because the forward takes `&self`; one short section per draft.
    #[cfg(feature = "metal")]
    positions: Mutex<(Vec<u32>, Option<Tensor>, Epoch)>,
}

impl Clone for DSparkRotary {
    fn clone(&self) -> Self {
        Self {
            cos: self.cos.clone(),
            sin: self.sin.clone(),
            interleaved: self.interleaved,
            #[cfg(feature = "metal")]
            positions: Mutex::new((Vec::new(), None, Epoch::ZERO)),
        }
    }
}

impl DSparkRotary {
    pub fn new(cfg: &DSparkConfig, dtype: DType, dev: &Device, max_seq_len: usize) -> Result<Self> {
        let dim = cfg.head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            interleaved: !cfg.rope_is_neox_style,
            #[cfg(feature = "metal")]
            positions: Mutex::new((Vec::new(), None, Epoch::ZERO)),
        })
    }

    /// Apply RoPE to `[b, heads, seq, head_dim]` at absolute position `offset`.
    ///
    /// Dispatches on the convention rather than hardcoding it: `rope_i` is the
    /// interleaved kernel (`y0 = x0·cos − x1·sin` over adjacent pairs) and
    /// `rope` the neox one (over halves). Both already exist in `candle-nn`
    /// with a Metal kernel and a scalar `_slow` sibling, so this is a caller,
    /// not new code — checked before writing (§3.1c: grep the behaviour).
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        self.apply_with(x, &cos, &sin)
    }

    /// The `cos`/`sin` rows for `seq_len` positions starting at each of
    /// `offsets`: `[B, seq_len, head_dim / 2]`, one gather for a batch of
    /// rows at their own positions — the rope kernels index a 3-D table per
    /// batch row. Taken once per draft and shared by every layer.
    pub fn rows(&self, offsets: &[usize], seq_len: usize) -> Result<(Tensor, Tensor)> {
        let mut idx: Vec<u32> = Vec::with_capacity(offsets.len() * seq_len);
        for &o in offsets {
            idx.extend((o..o + seq_len).map(|p| p as u32));
        }
        let idx = Tensor::new(idx.as_slice(), self.cos.device())?;
        let half = self.cos.dim(1)?;
        let cos = self
            .cos
            .index_select(&idx, 0)?
            .reshape((offsets.len(), seq_len, half))?;
        let sin = self
            .sin
            .index_select(&idx, 0)?
            .reshape((offsets.len(), seq_len, half))?;
        Ok((cos, sin))
    }

    /// The table rows of `positions` (absolute, in any order): `cos`/`sin`
    /// `[n, head_dim / 2]`. The 2-D form [`Self::apply_with`] reads one row
    /// per sequence position, so B rows' context positions laid end to end
    /// as one sequence are rotated at every position's own offset in one
    /// call — the batched context re-key's shape.
    pub fn at_positions(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let idx = self.position_index(positions)?;
        Ok((
            self.cos.index_select(&idx, 0)?,
            self.sin.index_select(&idx, 0)?,
        ))
    }

    /// `positions` as a U32 device tensor — the resident one (see the
    /// field) on Metal, an upload elsewhere.
    fn position_index(&self, positions: &[u32]) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = self.cos.device() {
            let clock = m.gpu_clock();
            let n = positions.len();
            let mut g = self.positions.lock();
            let (held, tensor, epoch) = &mut *g;
            if let Some(t) = tensor.as_ref() {
                if held.as_slice() == positions {
                    *epoch = clock.current_epoch();
                    return t.narrow(0, 0, n);
                }
                // SAFETY: the last gather's epoch is complete.
                if t.dim(0)? >= n
                    && clock.is_complete(*epoch)
                    && unsafe { t.write_host_slice::<u32>(0, positions) }
                {
                    held.clear();
                    held.extend_from_slice(positions);
                    *epoch = clock.current_epoch();
                    return t.narrow(0, 0, n);
                }
            }
            let have = tensor.as_ref().and_then(|t| t.dim(0).ok()).unwrap_or(0);
            let cap = n.max(have).max(64).next_power_of_two();
            held.clear();
            held.extend_from_slice(positions);
            held.resize(cap, 0);
            let t = Tensor::new(held.as_slice(), self.cos.device())?;
            held.truncate(n);
            *tensor = Some(t.clone());
            *epoch = clock.current_epoch();
            return t.narrow(0, 0, n);
        }
        Tensor::new(positions, self.cos.device())
    }

    /// [`Self::apply`] with the table rows in hand: `cos`/`sin` are
    /// `[seq, head_dim / 2]` (every batch row at the same positions) or
    /// `[B, seq, head_dim / 2]` (each at its own).
    pub fn apply_with(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        if self.interleaved {
            crate::nn::rotary_emb::rope_i(&x, cos, sin)
        } else {
            crate::nn::rotary_emb::rope(&x, cos, sin)
        }
    }
}

/// One draft layer's cached context K/V — the projected `H_ctx`, keyed and
/// valued once per verified block and then attended by every draft round.
///
/// This is the drafter's own KV cache (§10.2m's *"O(block), not O(target
/// kv_len)"*), and it holds the **context** half: `H_ctx` becomes K and V
/// through each layer's own `k_proj`/`v_proj`, exactly as mlx-dspark's
/// `update_ctx` does — `k, v = self._kv(fused); cache.append(self.rope(k,
/// offset=ctx_offset), v)`. The block's own K/V are concatenated in front of it
/// per round and are not retained.
///
/// # Against `CONTRIBUTING.md` §3.6 / `DESIGN.md` §15.2 #15
///
/// That law names this type: *"`KvCache`, `Norm`, `Rope`, `Attention` should be
/// traits with concrete impls (`KvSlot`, `QuantKvSlot`, a paged cache, **the
/// DSpark draft-KV** — all one `KvCache`)"*. **The trait does not exist in the
/// tree yet** — grepped, there is no `trait KvCache` — and minting the first one
/// inside a drafter PR would decide a cross-cutting seam for `KvSlot`,
/// `QuantKvSlot` and the paged cache from the narrowest of its four consumers.
/// That belongs to #602, the live tracker the law points at.
///
/// **What is owed now is the seam, and it is held**: this type is *threaded as a
/// parameter* at every use — `&CtxCache` to read, `&mut [CtxCache]` to fill —
/// and never owned by a layer, never global, never reached through the model.
/// So substituting a trait later is a signature change with no forward-pass
/// edit, which is the property the law is protecting. `DSparkRotary` is the same
/// shape one level down: the rope *style* is a selectable variant dispatched on
/// the config, not a hardcoded call, so the interleaved/neox choice is already a
/// block rather than a branch.
#[derive(Debug, Clone, Default)]
pub struct CtxCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// The paged backing ([`CtxKv::Paged`]); `k`/`v` stay `None` on it.
    #[cfg(feature = "metal")]
    paged: Option<paged_ctx::PagedCtx>,
}

/// Where the drafter's context KV lives — the backing behind [`CtxCache`]
/// (issue #708). A construction-time selection; the forward pass reads the
/// cache through one seam whichever arm it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtxKv {
    /// One contiguous `[1, H_kv, ctx, D]` tensor per layer, grown by
    /// `Tensor::cat` on every append and concatenated with the block's own K/V
    /// on every draft — O(ctx) copies per layer per round. What every recorded
    /// digest belongs to.
    Contiguous,
    /// One page pool per layer (`KvPagePool`, `DESIGN.md` §10.3c), written by
    /// `kv_scatter_into` at table-chosen slots and read by `sdpa_paged` through
    /// the table: the context grows into pages (no copy) and the draft block is
    /// written into slots reserved past the context (no concatenation). Metal
    /// only.
    Paged {
        /// KV tokens per page; a multiple of the tiled kernel's 32-key block.
        page_size: usize,
        /// `true`: one pool per layer shared by every sequence — one buffer,
        /// so every sequence's append is a write to the buffer every other
        /// sequence reads, and the backend orders them (§3.5: buffer identity
        /// is the hazard unit). `false`: one pool per layer PER SEQUENCE —
        /// the sequences' buffers are disjoint and their command buffers can
        /// overlap, which is the mechanism the pipelined driver's B>1 win
        /// rests on (§10.2y).
        shared: bool,
    },
}

impl CtxKv {
    /// §10.4's page size, the one the target's `KvAppend::Paged` ships at.
    pub const DEFAULT_PAGE_SIZE: usize = 256;

    /// The tuning spelling that leaves the arm to [`CtxKv::auto`].
    pub const AUTO: &'static str = "auto";

    /// The arm nothing chose: paged, one pool set per sequence, on Metal —
    /// the context grows into pages and the draft block is written into
    /// reserved slots, so a steady-state round allocates nothing (issue #894;
    /// §10.2z measured it at B=1 as the contiguous arm's equal and at B>1 as
    /// what keeps the pipelined driver's overlap) — and contiguous on the
    /// CPU, where the paged kernel does not run.
    pub fn auto(device: &Device) -> Self {
        if device.is_metal() {
            Self::Paged {
                page_size: Self::DEFAULT_PAGE_SIZE,
                shared: false,
            }
        } else {
            Self::Contiguous
        }
    }

    /// The arm `tuning.dspark.ctx_kv` (`--dspark-ctx`, `LLOOM_DSPARK_CTX_KV`)
    /// names: `None` for `auto`. Read at construction, never on the hot path.
    /// A value that parses as neither is an error rather than a fallback, so
    /// an A/B arm cannot silently measure the other one (§2.4).
    pub fn from_tuning() -> Result<Option<Self>> {
        let v = &crate::tuning::tuning().dspark.ctx_kv;
        if v.trim().eq_ignore_ascii_case(Self::AUTO) {
            return Ok(None);
        }
        Self::parse(v).map(Some)
    }

    /// [`CtxKv::from_tuning`], the `auto` spelling resolved for `device`.
    pub fn resolve(device: &Device) -> Result<Self> {
        Ok(Self::from_tuning()?.unwrap_or_else(|| Self::auto(device)))
    }

    /// `contiguous` | `paged[:<page_size>][:shared]`.
    pub fn parse(v: &str) -> Result<Self> {
        let v = v.trim();
        if v.eq_ignore_ascii_case("contiguous") {
            return Ok(Self::Contiguous);
        }
        let Some(rest) = v.strip_prefix("paged") else {
            lloom_tensor::bail!(
                "DSpark ctx kv {v:?}: expected `contiguous` or `paged[:<page_size>][:shared]` \
                 (issue #708)"
            )
        };
        let mut page_size = Self::DEFAULT_PAGE_SIZE;
        let mut shared = false;
        for part in rest.split(':').filter(|p| !p.is_empty()) {
            if part.eq_ignore_ascii_case("shared") {
                shared = true;
            } else {
                page_size = part.parse().map_err(|e| {
                    lloom_tensor::Error::msg(format!("DSpark ctx kv `{v}`: `{part}`: {e}"))
                })?;
            }
        }
        Ok(Self::Paged { page_size, shared })
    }

    /// The arm's name, as a `RESULT` line spells it.
    pub fn name(self) -> String {
        match self {
            Self::Contiguous => "contiguous".to_string(),
            Self::Paged { page_size, shared } => {
                format!("paged:{page_size}{}", if shared { ":shared" } else { "" })
            }
        }
    }
}

impl CtxCache {
    pub fn len(&self) -> usize {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return p.len();
        }
        match &self.k {
            Some(k) => k.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            p.reset();
            return;
        }
        self.k = None;
        self.v = None;
    }

    /// The paged arm's page table for this layer — the pages in token order,
    /// reserved block pages included. `None` on the contiguous arm. For a
    /// test that must show the table is NOT the identity (§10.3h), and a report.
    pub fn page_table(&self) -> Option<Vec<u32>> {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return Some(p.table());
        }
        None
    }

    /// Batched appends (`append_rows` over a shared paged pool) the pool
    /// behind this cache has taken; `None` off the paged arm.
    pub fn batched_appends(&self) -> Option<u64> {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return Some(p.batched_appends());
        }
        None
    }

    /// Which arm this cache is.
    pub fn kind(&self) -> CtxKv {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return CtxKv::Paged {
                page_size: p.page_size(),
                shared: p.shared(),
            };
        }
        CtxKv::Contiguous
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return p.append(k, v);
        }
        self.k = Some(match &self.k {
            None => k.clone(),
            Some(prev) => Tensor::cat(&[prev, k], 2)?.contiguous()?,
        });
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(prev) => Tensor::cat(&[prev, v], 2)?.contiguous()?,
        });
        Ok(())
    }

    /// [`Self::append`] for B rows at once: `k`/`v` are `[1, H_kv, Σlens, D]`,
    /// row `r`'s `lens[r]` positions laid end to end in `caches`' order. When
    /// every row is a slot of ONE shared paged pool the rows' keys land in one
    /// scatter per pool through one slot list (`DSparkCtxPool::append_rows`);
    /// otherwise each row appends its own slice — the same append it takes
    /// alone.
    fn append_rows(
        caches: &mut [&mut CtxCache],
        lens: &[usize],
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if caches.len() != lens.len() {
            lloom_tensor::bail!(
                "DSpark append_rows: {} caches and {} lengths",
                caches.len(),
                lens.len()
            );
        }
        #[cfg(feature = "metal")]
        {
            let shared = match caches.first().and_then(|c| c.paged.as_ref()) {
                Some(first) if first.shared() => caches.iter().all(|c| {
                    c.paged
                        .as_ref()
                        .map(|p| p.same_pool(first))
                        .unwrap_or(false)
                }),
                _ => false,
            };
            if shared {
                let seqs: Vec<usize> = caches
                    .iter()
                    .map(|c| c.paged.as_ref().expect("checked above").seq())
                    .collect();
                let first = caches[0].paged.as_ref().expect("checked above");
                return first.append_rows(&seqs, lens, k, v);
            }
        }
        let mut off = 0usize;
        for (c, &n) in caches.iter_mut().zip(lens) {
            if n > 0 {
                c.append(
                    &k.narrow(2, off, n)?.contiguous()?,
                    &v.narrow(2, off, n)?.contiguous()?,
                )?;
            }
            off += n;
        }
        Ok(())
    }

    /// The paged arm's read of `[context ‖ block]`: the block's `k`/`v`
    /// (`[1, H_kv, block, D]`, roped) are written into the slots reserved
    /// past the context and the pool is returned with the table that names
    /// both. `None` on the contiguous arm, and on every non-Metal build.
    #[cfg(feature = "metal")]
    fn paged_attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        match &self.paged {
            Some(p) => p.attend_block(q, k, v, scale).map(Some),
            None => Ok(None),
        }
    }

    /// [`Self::paged_view`] for B sequences at once: every cache a slot of
    /// ONE shared pool set (`CtxKv::Paged { shared: true }`), the B blocks'
    /// `k`/`v` (`[B, H_kv, block, D]`) scattered into their reserved slots
    /// in one dispatch, the pool returned with B tables and B live counts —
    /// what `sdpa_paged_ragged` reads in one dispatch. `None` when the rows
    /// do not share a pool (the contiguous arm, one pool set per sequence),
    /// and the caller attends per row.
    #[cfg(feature = "metal")]
    fn paged_attend_rows(
        caches: &[&CtxCache],
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        active: Option<&Tensor>,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        let Some(first) = caches.first().and_then(|c| c.paged.as_ref()) else {
            return Ok(None);
        };
        if !first.shared() {
            return Ok(None);
        }
        let mut seqs = Vec::with_capacity(caches.len());
        for c in caches {
            match &c.paged {
                Some(p) if p.same_pool(first) => seqs.push(p.seq()),
                _ => return Ok(None),
            }
        }
        first
            .attend_block_rows(&seqs, q, k, v, active, scale)
            .map(Some)
    }

    /// Drop the last `n` context positions — the draft-side half of a rollback.
    ///
    /// The draft has **no conv state** (all five `layer_types` are
    /// `full_attention`, §10.2l), so a rejection rolls back KV only, which is
    /// what this does. The target's conv rollback is the other half and is
    /// §10.2m's K+1 snapshots. On the paged arm it is a length decrement — the
    /// pages stay, as the target's `rewind` keeps them (§10.2a).
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        let len = self.len();
        let keep = len.saturating_sub(n);
        if keep == len {
            return Ok(());
        }
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            return p.rollback(keep);
        }
        if keep == 0 {
            self.clear();
            return Ok(());
        }
        self.k = Some(self.k.as_ref().unwrap().narrow(2, 0, keep)?.contiguous()?);
        self.v = Some(self.v.as_ref().unwrap().narrow(2, 0, keep)?.contiguous()?);
        Ok(())
    }
}

/// The paged backing of [`CtxCache`] (issue #708): one `KvPagePool` per
/// drafter layer, shared by every sequence the speculator drives.
#[cfg(feature = "metal")]
mod paged_ctx {
    use super::{Arc, Device, Mutex, Result, Tensor};
    use lloom_tensor::kernels::metal::{Epoch, GpuClock, PageError, PageIdx};

    /// Tokens per sequence the pool grows by — `lfm2.rs`'s `DEFAULT_KV_BUCKET`,
    /// so a drafter and its target widen at the same cadence.
    const BUCKET_TOKENS: usize = 4096;

    /// One layer's pool: the allocator (every sequence's table), its epoch
    /// quarantine, and the `[1, H_kv, pool_pages × P, D]` K/V pools, allocated
    /// at the first append at that append's dtype and geometry.
    struct LayerPool {
        /// The allocator with its liveness, over the tier's clock.
        pool: crate::nn::page_pool::PagePool,
        kv: Option<(Tensor, Tensor)>,
        /// Per sequence: the `reserve` slots past its live length, as the U32
        /// tensor the block scatter takes, and the largest of them.
        block: Vec<Option<(Tensor, usize)>>,
        /// Every sequence's block slots in one `[n_seqs * reserve]` U32
        /// tensor, sequence `s` at `s * reserve`, kept in step with `block`
        /// through the mapped pointer: the batched draft's scatter takes the
        /// active rows' slice of it in one dispatch. `None` until a host
        /// write fails to map (CPU), when the batched view falls back to a
        /// per-round upload.
        block_all: Option<Tensor>,
        /// What `block_all` holds, on the host: the source of a fresh tensor
        /// when the in-place write is not safe.
        block_host: Vec<u32>,
        /// The epoch of the last dispatch that read or wrote `block_all`; an
        /// in-place write waits for it to be complete, else a fresh tensor.
        block_epoch: Epoch,
    }

    /// The slot tensor one sequence's layers share within a round.
    ///
    /// Every layer appends the same positions of the same sequence, and the
    /// layers' pools follow the same allocation history, so the slot vectors
    /// they compute are the same vectors; uploading them once per round rather
    /// than once per layer is what keeps the host side of an append at one
    /// small buffer per round. **Shared by value, not by assumption**: a layer
    /// reuses the tensor only when the vector it computed is equal to the one
    /// uploaded, and uploads its own otherwise. The block slots need no upload
    /// at all: they are written in place into the layer's `block_all`.
    ///
    /// The slot tensor is RESIDENT: sized for the append (grown by doubling
    /// when one outgrows it), written in place and narrowed to the count — no
    /// upload, no allocation per round. `slots` shadows what the tensor
    /// holds and `epoch` is the epoch of the last dispatch that read it (the
    /// scatter encoded right after the write). The rules, in order:
    /// the same slots again (the layers compute the same ones) → the tensor
    /// as it is; different slots and that epoch complete (the round's verify
    /// readback fenced it) → written in place; different slots and a reader
    /// possibly in flight → a fresh tensor, the old one living on in the
    /// queued command buffer. The third arm is what a caller with no fence
    /// between appends (a test keying two rounds back to back) hits; the
    /// clock decides, not an assumption about the caller.
    #[derive(Default)]
    struct SeqSlots {
        slots: Vec<u32>,
        slots_t: Option<Tensor>,
        epoch: Epoch,
    }

    /// Every layer's pool, one clock. Behind a `Mutex` because the forward
    /// pass reads the cache through `&CtxCache` and the draft block is written
    /// into the pool from there; five short critical sections per round.
    ///
    /// A `parking_lot::Mutex` (`Cargo.toml`): those critical sections are on
    /// the draft path, and std's would add a poison `Result` branch to each
    /// of them. It also removes the recovery this code used to need — there
    /// is no poisoning to recover from, so a panic mid-append no longer
    /// leaves a lock whose guard has to be forced open.
    pub(super) struct DSparkCtxPool {
        page_size: usize,
        /// Slots kept free past every sequence's context for the draft block.
        reserve: usize,
        n_seqs: usize,
        clock: Arc<GpuClock>,
        layers: Vec<LayerPool>,
        /// Per sequence, the slot tensors the layers share (see `SeqSlots`).
        shared: Vec<SeqSlots>,
        /// The batched append's flat slot list (`append_rows`), shared by the
        /// layers the same way.
        rows: SeqSlots,
        /// Host scratch kept across rounds (capacity retained; no `Vec` per
        /// call): a block's slots, the batched append's B blocks' slots laid
        /// end to end, the batched read's padded tables and live counts.
        scratch_slots: Vec<u32>,
        scratch_block: Vec<u32>,
        scratch_blocks: Vec<u32>,
        scratch_tables: Vec<u32>,
        scratch_keys: Vec<u32>,
        /// Batched appends taken (`append_rows` calls, one per layer per
        /// round), for a gate that asserts the one-scatter arm ran.
        batched_appends: u64,
    }

    impl DSparkCtxPool {
        fn new(
            n_layers: usize,
            n_seqs: usize,
            page_size: usize,
            reserve: usize,
            clock: Arc<GpuClock>,
        ) -> Self {
            let pages_per_seq = (BUCKET_TOKENS + reserve).div_ceil(page_size);
            let pool_pages = n_seqs.max(1) * pages_per_seq;
            let layers = (0..n_layers)
                .map(|_| LayerPool {
                    // A table may reach the whole pool: exhaustion is reported
                    // as the pool's, which names the grow, never as a full table.
                    pool: crate::nn::page_pool::PagePool::with_clock(
                        page_size,
                        pool_pages,
                        n_seqs.max(1),
                        pool_pages + 1,
                        Arc::clone(&clock),
                    ),
                    kv: None,
                    block: vec![None; n_seqs.max(1)],
                    block_all: None,
                    block_host: Vec::new(),
                    block_epoch: Epoch::ZERO,
                })
                .collect();
            Self {
                page_size,
                reserve,
                n_seqs: n_seqs.max(1),
                clock,
                layers,
                shared: (0..n_seqs.max(1)).map(|_| SeqSlots::default()).collect(),
                rows: SeqSlots::default(),
                scratch_slots: Vec::new(),
                scratch_block: Vec::new(),
                scratch_blocks: Vec::new(),
                scratch_tables: Vec::new(),
                scratch_keys: Vec::new(),
                batched_appends: 0,
            }
        }

        /// `slots` (computed into a scratch) as a device tensor, per
        /// `SeqSlots`'s rules: the resident tensor as it is, written in
        /// place, or a fresh one — the slots in its prefix, uploaded, so no
        /// fill dispatch races the host write. `slots` and the shadow are
        /// swapped, so the caller's scratch keeps a capacity either way.
        fn upload(
            cached: &mut SeqSlots,
            slots: &mut Vec<u32>,
            device: &Device,
            clock: &GpuClock,
        ) -> Result<Tensor> {
            let n = slots.len();
            if let Some(t) = cached.slots_t.as_ref() {
                if cached.slots == *slots {
                    cached.epoch = clock.current_epoch();
                    return t.narrow(0, 0, n);
                }
            }
            std::mem::swap(&mut cached.slots, slots);
            if let Some(t) = cached.slots_t.as_ref() {
                // SAFETY: the epoch of the last reader is complete.
                if t.dim(0)? >= n
                    && clock.is_complete(cached.epoch)
                    && unsafe { t.write_host_slice::<u32>(0, &cached.slots) }
                {
                    cached.epoch = clock.current_epoch();
                    return t.narrow(0, 0, n);
                }
            }
            let have = cached
                .slots_t
                .as_ref()
                .and_then(|t| t.dim(0).ok())
                .unwrap_or(0);
            let cap = n.max(have).max(64).next_power_of_two();
            let mut init = std::mem::take(&mut cached.slots);
            init.resize(cap, 0);
            let t = Tensor::new(init.as_slice(), device)?;
            init.truncate(n);
            cached.slots = init;
            cached.slots_t = Some(t.clone());
            cached.epoch = clock.current_epoch();
            t.narrow(0, 0, n)
        }

        /// Widen `layer`'s pool by one bucket: the allocator gains pages
        /// beneath its free list and the K/V pools are reallocated wider with
        /// the old contents in their prefix, so every page index and table
        /// stays valid (`KvPagePool::grow`'s contract; `PagedKv::grow_to`'s
        /// shape).
        fn grow(&mut self, layer: usize) -> Result<()> {
            let extra = self.n_seqs * (BUCKET_TOKENS + self.reserve).div_ceil(self.page_size);
            let lp = &mut self.layers[layer];
            lp.pool.grow(extra);
            if let Some((pk, pv)) = lp.kv.as_ref() {
                let mut shape = pk.dims().to_vec();
                shape[2] = lp.pool.pool_pages() * self.page_size;
                let nk = Tensor::zeros(shape.clone(), pk.dtype(), pk.device())?;
                nk.slice_set(pk, 2, 0)?;
                let nv = Tensor::zeros(shape, pv.dtype(), pv.device())?;
                nv.slice_set(pv, 2, 0)?;
                lp.kv = Some((nk, nv));
            }
            Ok(())
        }

        /// The token slot of position `t` in `pages`.
        fn slot(pages: &[PageIdx], t: usize, page_size: usize) -> u32 {
            pages[t / page_size].0 * page_size as u32 + (t % page_size) as u32
        }

        /// Append `[1, H_kv, n, D]` `k`/`v` to `(layer, seq)`: the context
        /// grows by `n` positions and `reserve` slots stay held past it for
        /// the block. Pages the GPU has finished reading come back first.
        fn append(&mut self, layer: usize, seq: usize, k: &Tensor, v: &Tensor) -> Result<()> {
            let (b, n_kv_heads, n, head_dim) = k.dims4()?;
            if b != 1 {
                lloom_tensor::bail!("DSpark paged ctx: one sequence per append, got batch {b}");
            }
            let page_size = self.page_size;
            let reserve = self.reserve;
            self.layers[layer]
                .pool
                .recycle_ready()
                .map_err(lloom_tensor::Error::wrap)?;
            // Room for the append and the block behind it, a bucket at a time.
            loop {
                let lp = &mut self.layers[layer];
                match lp.pool.reserve(seq, n + reserve) {
                    Ok(_) => break,
                    Err(PageError::Exhausted { .. }) => self.grow(layer)?,
                    Err(e) => return Err(lloom_tensor::Error::wrap(e)),
                }
            }
            let lp = &mut self.layers[layer];
            let start = lp.pool.seq(seq).map(|s| s.len()).unwrap_or(0);
            lp.pool.append(seq, n).map_err(lloom_tensor::Error::wrap)?;
            let sh = &mut self.shared[seq];
            let slots = &mut self.scratch_slots;
            let mut block = std::mem::take(&mut self.scratch_block);
            {
                // The page list read in place; the slots into the scratch.
                let pages = lp.pool.seq(seq).expect("seq within n_seqs").pages();
                slots.clear();
                slots.extend((start..start + n).map(|t| Self::slot(pages, t, page_size)));
                block.clear();
                block.extend(
                    (start + n..start + n + reserve).map(|t| Self::slot(pages, t, page_size)),
                );
            }
            let tokens = lp.pool.pool_pages() * page_size;
            if lp.kv.is_none() {
                let shape = (1, n_kv_heads, tokens, head_dim);
                lp.kv = Some((
                    Tensor::zeros(shape, k.dtype(), k.device())?,
                    Tensor::zeros(shape, v.dtype(), v.device())?,
                ));
            }
            let (pk, pv) = lp.kv.as_ref().expect("allocated above");
            let max_slot = slots.iter().copied().max().unwrap_or(0) as usize;
            let slots = Self::upload(sh, slots, k.device(), &self.clock)?;
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            k.kv_scatter_into(pk, &slots, max_slot, 0, tokens * head_dim, head_dim)?;
            v.kv_scatter_into(pv, &slots, max_slot, 0, tokens * head_dim, head_dim)?;
            let r = self.set_block(layer, seq, &block, k.device());
            self.scratch_block = block;
            r
        }

        /// [`Self::append`] for `seqs` at once: `k`/`v` are `[1, H_kv, Σlens,
        /// D]`, row `r`'s `lens[r]` positions laid end to end in `seqs`'s
        /// order. Every row's context grows by its own count with `append`'s
        /// bookkeeping; the rows' slots go up as ONE list (shared by the
        /// layers, as a row's are) and the keys land in one scatter per pool.
        fn append_rows(
            &mut self,
            layer: usize,
            seqs: &[usize],
            lens: &[usize],
            k: &Tensor,
            v: &Tensor,
        ) -> Result<()> {
            let (b, n_kv_heads, t, head_dim) = k.dims4()?;
            if b != 1 || seqs.len() != lens.len() || lens.iter().sum::<usize>() != t {
                lloom_tensor::bail!(
                    "DSpark paged ctx: a batched append of [{b}, _, {t}, _] for {} rows of \
                     {lens:?} positions",
                    seqs.len()
                );
            }
            let page_size = self.page_size;
            let reserve = self.reserve;
            self.layers[layer]
                .pool
                .recycle_ready()
                .map_err(lloom_tensor::Error::wrap)?;
            let mut slots = std::mem::take(&mut self.scratch_slots);
            slots.clear();
            let mut blocks = std::mem::take(&mut self.scratch_blocks);
            blocks.clear();
            for (&seq, &n) in seqs.iter().zip(lens) {
                loop {
                    let lp = &mut self.layers[layer];
                    match lp.pool.reserve(seq, n + reserve) {
                        Ok(_) => break,
                        Err(PageError::Exhausted { .. }) => self.grow(layer)?,
                        Err(e) => return Err(lloom_tensor::Error::wrap(e)),
                    }
                }
                let lp = &mut self.layers[layer];
                let start = lp.pool.seq(seq).map(|s| s.len()).unwrap_or(0);
                lp.pool.append(seq, n).map_err(lloom_tensor::Error::wrap)?;
                let pages = lp.pool.seq(seq).expect("seq within n_seqs").pages();
                slots.extend((start..start + n).map(|t| Self::slot(pages, t, page_size)));
                blocks.extend(
                    (start + n..start + n + reserve).map(|t| Self::slot(pages, t, page_size)),
                );
            }
            let lp = &mut self.layers[layer];
            let tokens = lp.pool.pool_pages() * page_size;
            if lp.kv.is_none() {
                let shape = (1, n_kv_heads, tokens, head_dim);
                lp.kv = Some((
                    Tensor::zeros(shape, k.dtype(), k.device())?,
                    Tensor::zeros(shape, v.dtype(), v.device())?,
                ));
            }
            let (pk, pv) = lp.kv.as_ref().expect("allocated above");
            let max_slot = slots.iter().copied().max().unwrap_or(0) as usize;
            let slots_t = Self::upload(&mut self.rows, &mut slots, k.device(), &self.clock)?;
            self.scratch_slots = slots;
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            k.kv_scatter_into(pk, &slots_t, max_slot, 0, tokens * head_dim, head_dim)?;
            v.kv_scatter_into(pv, &slots_t, max_slot, 0, tokens * head_dim, head_dim)?;
            let mut r = Ok(());
            for (&seq, block) in seqs.iter().zip(blocks.chunks_exact(reserve)) {
                r = self.set_block(layer, seq, block, k.device());
                if r.is_err() {
                    break;
                }
            }
            self.scratch_blocks = blocks;
            r?;
            self.batched_appends += 1;
            Ok(())
        }

        /// Record `(layer, seq)`'s block slots: the `reserve` entries written
        /// in place into the layer's `block_all`, whose row `seq` is then the
        /// per-sequence tensor the single-row view scatters through (a view,
        /// no upload) and whose active rows the batched view takes. Written
        /// between rounds only (an append, a rollback, a key), when no queued
        /// dispatch reads it — the previous draft's scatter completed with
        /// the verify that consumed it.
        fn set_block(
            &mut self,
            layer: usize,
            seq: usize,
            slots: &[u32],
            device: &Device,
        ) -> Result<()> {
            let (n_seqs, reserve) = (self.n_seqs, self.reserve);
            if slots.len() != reserve {
                lloom_tensor::bail!(
                    "DSpark paged ctx: {} block slots for a reserve of {reserve}",
                    slots.len()
                );
            }
            let block_max = slots.iter().copied().max().unwrap_or(0) as usize;
            let now = self.clock.current_epoch();
            let settled = self.clock.is_complete(self.layers[layer].block_epoch);
            let lp = &mut self.layers[layer];
            if lp.block_host.len() != n_seqs * reserve {
                lp.block_host = vec![0u32; n_seqs * reserve];
            }
            lp.block_host[seq * reserve..(seq + 1) * reserve].copy_from_slice(slots);
            // In place when every dispatch that read `block_all` has
            // completed (the round's verify readback); else a fresh tensor
            // from the host copy — an upload, not `zeros`, so no pooled fill
            // blit races the write — the old one living on in the queued
            // command buffer. The clock decides, as `SeqSlots` says; a CPU
            // device (no mapping) takes the fresh-tensor arm every time.
            let mapped = match lp.block_all.as_ref() {
                // SAFETY: `settled` — no queued dispatch reads the tensor.
                Some(all)
                    if settled && unsafe { all.write_host_slice::<u32>(seq * reserve, slots) } =>
                {
                    Some(all.narrow(0, seq * reserve, reserve)?)
                }
                _ => None,
            };
            let block = match mapped {
                Some(view) => view,
                None => {
                    let all = Tensor::new(lp.block_host.as_slice(), device)?;
                    let view = all.narrow(0, seq * reserve, reserve)?;
                    lp.block_all = Some(all);
                    view
                }
            };
            lp.block_epoch = now;
            lp.block[seq] = Some((block, block_max));
            Ok(())
        }

        /// Key `(layer, seq)` with an empty context: allocate the layer's pools
        /// if it has none, hold `reserve` slots for the block, upload the block
        /// slots. The length stays zero; `append` reserves on top of this and
        /// recomputes the block slots past whatever it appends, so keying
        /// early changes nothing about where the context lands.
        fn key_empty(
            &mut self,
            layer: usize,
            seq: usize,
            n_kv_heads: usize,
            head_dim: usize,
            dtype: lloom_tensor::DType,
            device: &Device,
        ) -> Result<()> {
            let page_size = self.page_size;
            let reserve = self.reserve;
            loop {
                let lp = &mut self.layers[layer];
                match lp.pool.reserve(seq, reserve) {
                    Ok(_) => break,
                    Err(PageError::Exhausted { .. }) => self.grow(layer)?,
                    Err(e) => return Err(lloom_tensor::Error::wrap(e)),
                }
            }
            let lp = &mut self.layers[layer];
            let start = lp.pool.seq(seq).map(|s| s.len()).unwrap_or(0);
            let mut block = std::mem::take(&mut self.scratch_block);
            {
                let pages = lp.pool.seq(seq).expect("seq within n_seqs").pages();
                block.clear();
                block.extend((start..start + reserve).map(|t| Self::slot(pages, t, page_size)));
            }
            let tokens = lp.pool.pool_pages() * page_size;
            if lp.kv.is_none() {
                let shape = (1, n_kv_heads, tokens, head_dim);
                lp.kv = Some((
                    Tensor::zeros(shape, dtype, device)?,
                    Tensor::zeros(shape, dtype, device)?,
                ));
            }
            let r = self.set_block(layer, seq, &block, device);
            self.scratch_block = block;
            r
        }

        /// Write the block's `[1, H_kv, width, D]` `k`/`v` into `(layer,
        /// seq)`'s reserved slots and read `q` over the context plus the
        /// block through the tiled paged kernel — dispatched here, under the
        /// lock, so the kernel reads the sequence's page list in place (no
        /// table copied out per draft per layer).
        #[allow(clippy::too_many_arguments)]
        fn attend_block(
            &mut self,
            layer: usize,
            seq: usize,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            scale: f32,
        ) -> Result<Tensor> {
            let (b, _, width, head_dim) = k.dims4()?;
            if b != 1 || width > self.reserve {
                lloom_tensor::bail!(
                    "DSpark paged ctx: a block of {width} positions (batch {b}) against {} \
                     reserved slots",
                    self.reserve
                );
            }
            let page_size = self.page_size;
            if self.layers[layer].kv.is_none() || self.layers[layer].block[seq].is_none() {
                // A draft before any context — the drafter warmup's shape
                // (`DSparkSpeculator::warmup` drafts before the prefill is
                // observed, to compile the drafter's pipelines off the clock),
                // and the contiguous arm's ordinary empty-context draft. Key an
                // empty context: the pools allocated from the block's shape,
                // `reserve` slots held for the block, the length still zero,
                // so the first `append` lands exactly where it would have.
                let (_, n_kv_heads, _, head_dim) = k.dims4()?;
                self.key_empty(layer, seq, n_kv_heads, head_dim, k.dtype(), k.device())?;
            }
            let now = self.clock.current_epoch();
            let lp = &mut self.layers[layer];
            lp.block_epoch = now;
            let lp = &*lp;
            let (pk, pv) = lp.kv.as_ref().expect("keyed above");
            let (block, block_max) = lp.block[seq].as_ref().expect("keyed above");
            let tokens = lp.pool.pool_pages() * page_size;
            let slots = block.narrow(0, 0, width)?;
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            k.kv_scatter_into(pk, &slots, *block_max, 0, tokens * head_dim, head_dim)?;
            v.kv_scatter_into(pv, &slots, *block_max, 0, tokens * head_dim, head_dim)?;
            let s = lp.pool.seq(seq).expect("seq within n_seqs");
            let table = PageIdx::as_u32s(s.pages());
            crate::nn::ops::sdpa_paged(
                q,
                pk,
                pv,
                false,
                scale,
                page_size,
                table,
                table.len(),
                s.len() + width,
            )
        }

        /// [`Self::attend_block`] for `seqs` at once: the B blocks' `k`/`v`
        /// (`[B, H_kv, width, D]`, row `b` sequence `seqs[b]`'s) scattered
        /// into their reserved slots in ONE dispatch through the layer's
        /// `block_all` (the active rows' slice when `seqs` is `0..B`, else a
        /// gather by `active`), then `q` read over every row's context plus
        /// block in one `sdpa_paged_ragged` dispatch; the rows' padded
        /// tables and live counts are built into the pool's scratch.
        #[allow(clippy::too_many_arguments)]
        fn attend_block_rows(
            &mut self,
            layer: usize,
            seqs: &[usize],
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            active: Option<&Tensor>,
            scale: f32,
        ) -> Result<Tensor> {
            let (b, n_kv_heads, width, head_dim) = k.dims4()?;
            if b != seqs.len() || width > self.reserve {
                lloom_tensor::bail!(
                    "DSpark paged ctx: blocks [{b}, _, {width}, _] for {} sequences against {} \
                     reserved slots",
                    seqs.len(),
                    self.reserve
                );
            }
            let page_size = self.page_size;
            let reserve = self.reserve;
            for &seq in seqs {
                if self.layers[layer].kv.is_none() || self.layers[layer].block[seq].is_none() {
                    self.key_empty(layer, seq, n_kv_heads, head_dim, k.dtype(), k.device())?;
                }
            }
            let now = self.clock.current_epoch();
            let lp = &mut self.layers[layer];
            lp.block_epoch = now;
            let lp = &*lp;
            let (pk, pv) = lp.kv.as_ref().expect("keyed above");
            let tokens = lp.pool.pool_pages() * page_size;
            let prefix = seqs.iter().enumerate().all(|(i, &s)| s == i);
            // The B rows' block slots as one `[B * width]` list, in row order.
            let slots = match (&lp.block_all, active) {
                (Some(all), _) if prefix && width == reserve => all.narrow(0, 0, b * reserve)?,
                (Some(all), Some(idx)) if width == reserve => all
                    .reshape((self.n_seqs, reserve))?
                    .index_select(idx, 0)?
                    .flatten_all()?,
                _ => {
                    let parts: Vec<Tensor> = seqs
                        .iter()
                        .map(|&s| {
                            lp.block[s]
                                .as_ref()
                                .expect("keyed above")
                                .0
                                .narrow(0, 0, width)
                        })
                        .collect::<Result<_>>()?;
                    Tensor::cat(&parts, 0)?
                }
            };
            let max_slot = seqs
                .iter()
                .map(|&s| lp.block[s].as_ref().expect("keyed above").1)
                .max()
                .unwrap_or(0);
            let k = k.contiguous()?;
            let v = v.contiguous()?;
            k.kv_scatter_into(pk, &slots, max_slot, 0, tokens * head_dim, head_dim)?;
            v.kv_scatter_into(pv, &slots, max_slot, 0, tokens * head_dim, head_dim)?;
            let stride = seqs
                .iter()
                .map(|&s| lp.pool.seq(s).map(|p| p.pages().len()).unwrap_or(0))
                .max()
                .unwrap_or(1)
                .max(1);
            let tables = &mut self.scratch_tables;
            let n_keys = &mut self.scratch_keys;
            tables.clear();
            n_keys.clear();
            for &s in seqs {
                let sp = lp.pool.seq(s).expect("seq within n_seqs");
                let p = sp.pages();
                tables.extend_from_slice(PageIdx::as_u32s(p));
                tables.extend(std::iter::repeat_n(0, stride - p.len()));
                n_keys.push((sp.len() + width) as u32);
            }
            crate::nn::ops::sdpa_paged_ragged(
                q, pk, pv, false, scale, page_size, tables, stride, n_keys,
            )
        }

        fn len(&self, layer: usize, seq: usize) -> usize {
            self.layers[layer]
                .pool
                .seq(seq)
                .map(|s| s.len())
                .unwrap_or(0)
        }

        /// The layer's K/V pools, once an append has allocated them.
        fn pools(&self, layer: usize) -> Result<(Tensor, Tensor)> {
            match self.layers[layer].kv.as_ref() {
                Some((k, v)) => Ok((k.clone(), v.clone())),
                None => lloom_tensor::bail!("DSpark paged ctx: layer {layer} has no pool yet"),
            }
        }

        /// Back to `to` positions: a length decrement, pages kept, the block
        /// slots moved to follow the new end.
        fn rollback(&mut self, layer: usize, seq: usize, to: usize) -> Result<()> {
            let page_size = self.page_size;
            let reserve = self.reserve;
            let lp = &mut self.layers[layer];
            let len = lp.pool.seq(seq).map(|s| s.len()).unwrap_or(0);
            if to > len {
                lloom_tensor::bail!("DSpark paged ctx: rollback to {to} past len {len}");
            }
            lp.pool
                .rewind(seq, len - to)
                .map_err(lloom_tensor::Error::wrap)?;
            if let Some((block, _)) = lp.block[seq].as_ref() {
                let device = block.device().clone();
                let mut slots = std::mem::take(&mut self.scratch_block);
                {
                    let pages = lp.pool.seq(seq).expect("seq within n_seqs").pages();
                    slots.clear();
                    slots.extend((to..to + reserve).map(|t| Self::slot(pages, t, page_size)));
                }
                let r = self.set_block(layer, seq, &slots, &device);
                self.scratch_block = slots;
                r?;
            }
            Ok(())
        }

        /// Drop `(layer, seq)`'s pages into the quarantine against the epoch a
        /// command buffer encoded now belongs to (§10.3h's one non-fence
        /// obligation). Idempotent.
        fn reset(&mut self, layer: usize, seq: usize) {
            let lp = &mut self.layers[layer];
            if lp
                .pool
                .seq(seq)
                .map(|s| s.pages().is_empty())
                .unwrap_or(true)
            {
                return;
            }
            lp.pool.release(seq);
            lp.block[seq] = None;
        }
    }

    /// One `(sequence, layer)` slot of a shared [`DSparkCtxPool`] — what a
    /// paged [`super::CtxCache`] holds.
    #[derive(Clone)]
    pub(super) struct PagedCtx {
        shared: Arc<Mutex<DSparkCtxPool>>,
        seq: usize,
        layer: usize,
        page_size: usize,
        /// Whether `shared` is one pool set for every sequence (`seq` indexes
        /// its table) or this sequence's own (`seq` is 0 there).
        pool_shared: bool,
    }

    impl std::fmt::Debug for PagedCtx {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PagedCtx")
                .field("seq", &self.seq)
                .field("layer", &self.layer)
                .field("page_size", &self.page_size)
                .finish()
        }
    }

    impl PagedCtx {
        /// `n_seqs × n_layers` handles over one pool set.
        pub(super) fn make(
            n_layers: usize,
            n_seqs: usize,
            page_size: usize,
            reserve: usize,
            shared: bool,
            device: &Device,
        ) -> Result<Vec<Vec<PagedCtx>>> {
            let clock = match device {
                Device::Metal(d) => d.gpu_clock(),
                _ => lloom_tensor::bail!(
                    "DSpark paged ctx is Metal-only: the pool is read by a Metal kernel"
                ),
            };
            // The tiled kernel seats its loaders once per key block, so a page
            // must hold whole blocks: 32 keys below head_dim 128, 16 at and
            // above (`sdpa_full_tile`) — 32 covers both. `sdpa_paged` checks
            // the exact tile at dispatch; this is the early, named refusal.
            if page_size == 0 || !page_size.is_multiple_of(32) {
                lloom_tensor::bail!(
                    "DSpark paged ctx: page_size {page_size} must be a multiple of the tiled \
                     kernel's 32-key block (issue #708)"
                );
            }
            // One pool set for all sequences, or one per sequence: the handles
            // are the same shape either way, and the difference is which
            // buffer a sequence's append writes — every other sequence's, or
            // its own.
            let pools: Vec<Arc<Mutex<DSparkCtxPool>>> = if shared {
                let one = Arc::new(Mutex::new(DSparkCtxPool::new(
                    n_layers, n_seqs, page_size, reserve, clock,
                )));
                vec![one; n_seqs]
            } else {
                (0..n_seqs)
                    .map(|_| {
                        Arc::new(Mutex::new(DSparkCtxPool::new(
                            n_layers,
                            1,
                            page_size,
                            reserve,
                            clock.clone(),
                        )))
                    })
                    .collect()
            };
            Ok((0..n_seqs)
                .map(|seq| {
                    (0..n_layers)
                        .map(|layer| PagedCtx {
                            shared: pools[seq].clone(),
                            seq: if shared { seq } else { 0 },
                            layer,
                            page_size,
                            pool_shared: shared,
                        })
                        .collect()
                })
                .collect())
        }

        fn lock(&self) -> parking_lot::MutexGuard<'_, DSparkCtxPool> {
            // No poison arm: `parking_lot` does not poison. What the std
            // version's recovery said — that a panic mid-append leaves the
            // pool's bookkeeping unknown — is unchanged and unhandled either
            // way; this simply stops pretending the `unwrap_or_else` was
            // doing something about it.
            self.shared.lock()
        }

        pub(super) fn page_size(&self) -> usize {
            self.page_size
        }

        pub(super) fn shared(&self) -> bool {
            self.pool_shared
        }

        pub(super) fn len(&self) -> usize {
            self.lock().len(self.layer, self.seq)
        }

        pub(super) fn append(&self, k: &Tensor, v: &Tensor) -> Result<()> {
            self.lock().append(self.layer, self.seq, k, v)
        }

        /// [`Self::append`] for the rows `seqs` of this handle's pool set at
        /// once (`DSparkCtxPool::append_rows`).
        pub(super) fn append_rows(
            &self,
            seqs: &[usize],
            lens: &[usize],
            k: &Tensor,
            v: &Tensor,
        ) -> Result<()> {
            self.lock().append_rows(self.layer, seqs, lens, k, v)
        }

        pub(super) fn attend_block(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            scale: f32,
        ) -> Result<Tensor> {
            self.lock()
                .attend_block(self.layer, self.seq, q, k, v, scale)
        }

        /// Whether `other` is a handle over the same pool set (the batched
        /// draft's one-dispatch condition).
        pub(super) fn same_pool(&self, other: &PagedCtx) -> bool {
            Arc::ptr_eq(&self.shared, &other.shared) && self.layer == other.layer
        }

        pub(super) fn seq(&self) -> usize {
            self.seq
        }

        /// [`Self::block_view`] for the rows `seqs` of this handle's pool set.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn attend_block_rows(
            &self,
            seqs: &[usize],
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            active: Option<&Tensor>,
            scale: f32,
        ) -> Result<Tensor> {
            self.lock()
                .attend_block_rows(self.layer, seqs, q, k, v, active, scale)
        }

        pub(super) fn rollback(&self, to: usize) -> Result<()> {
            self.lock().rollback(self.layer, self.seq, to)
        }

        /// Batched appends this handle's pool set has taken.
        pub(super) fn batched_appends(&self) -> u64 {
            self.lock().batched_appends
        }

        pub(super) fn reset(&self) {
            self.lock().reset(self.layer, self.seq)
        }

        /// This layer's K/V pools.
        pub(super) fn pools(&self) -> Result<(Tensor, Tensor)> {
            self.lock().pools(self.layer)
        }

        /// This sequence's pages on this layer, in token order.
        pub(super) fn table(&self) -> Vec<u32> {
            let g = self.lock();
            g.layers[self.layer]
                .pool
                .seq(self.seq)
                .map(|s| PageIdx::as_u32s(s.pages()).to_vec())
                .unwrap_or_default()
        }
    }
}

// The drafter's own KV cache under the engine seam (`RESTRUCTURING_DESIGN.md`
// §3.3: "the DSpark draft-KV — all one `KvCache`"). It is the `KvSource::Own`
// case a `Speculator` names: `append` keys/values the projected `H_ctx` and
// returns the live `(K, V)`; `rollback` is `truncate`'s draft-side half. This is
// the concrete `CtxCache: KvCache` that lets `KvSource::Own` name a real type.
impl crate::nn::kv_cache::seam::KvCache for CtxCache {
    type View<'a> = (Tensor, Tensor);

    fn append<'a>(&'a mut self, k: &Tensor, v: &Tensor) -> Result<Self::View<'a>> {
        CtxCache::append(self, k, v)?;
        #[cfg(feature = "metal")]
        if let Some(p) = &self.paged {
            // The live cache is the pool itself, read through the table.
            return p.pools();
        }
        Ok((
            self.k.as_ref().unwrap().clone(),
            self.v.as_ref().unwrap().clone(),
        ))
    }

    fn len(&self) -> usize {
        CtxCache::len(self)
    }

    fn rollback(&mut self, to: usize) -> Result<()> {
        let len = CtxCache::len(self);
        if to > len {
            lloom_tensor::bail!("CtxCache rollback to {to} past len {len}")
        }
        self.truncate(len - to)
    }

    fn reset(&mut self) {
        self.clear()
    }
}

#[derive(Debug)]
struct DSparkMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DSparkMlp {
    fn new(cfg: &DSparkConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: crate::nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: crate::nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: crate::nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = crate::nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// `true` when `LLOOM_DSPARK_EAGER_ATTN` is set, forcing the drafter's
/// bidirectional block attention onto the eager `matmul → softmax → matmul`
/// arm even on Metal, so the fused `sdpa` arm and its reference can be A/B'd
/// for a parity gate (`lfm2_dspark_backbone.rs`). Read once — fixed for the
/// process, like `conv_c_fusion_disabled` and the other `LLOOM_*` toggles, and
/// so off the per-draft hot path (§15.2 #10: no per-token env read).
fn dspark_eager_attn() -> bool {
    use std::sync::OnceLock;
    static EAGER: OnceLock<bool> = OnceLock::new();
    *EAGER.get_or_init(|| crate::tuning::tuning().dspark.eager_attn)
}

#[derive(Debug)]
struct DSparkAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl DSparkAttention {
    fn new(cfg: &DSparkConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        Ok(Self {
            q_proj: crate::nn::linear_no_bias(h, cfg.num_attention_heads * hd, vb.pp("q_proj"))?,
            k_proj: crate::nn::linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v_proj: crate::nn::linear_no_bias(h, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o_proj: crate::nn::linear_no_bias(cfg.num_attention_heads * hd, h, vb.pp("o_proj"))?,
            // Per-head RMS norm over `head_dim`, exactly the target's shape
            // (§5.4's `[64]` q/k layernorms) and the Qwen3 convention.
            q_norm: crate::nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: crate::nn::rms_norm(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_kv_groups(),
            head_dim: hd,
        })
    }

    /// Whether the fused `sdpa` kernel has an instantiation for this head dim.
    ///
    /// The same set the target's `sdpa_full_applies` gates on
    /// (`lfm2.rs`, `metal/kernels/sdpa.rs`); `head_dim` is `64` on the 2.6B
    /// drafter, which is in it. Kept as a named predicate so the Metal arm's
    /// guard reads as one condition and a checkpoint with an un-instantiated
    /// head dim falls to the eager arm rather than bailing in the kernel.
    /// A shape probe (`docs/porting-models.md` §21.8): the check is the
    /// annotation.
    fn head_dim_instantiated(head_dim: usize) -> bool {
        const HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];
        lloom_tensor::shape_req!(sdpa_head_dim, HEAD_DIMS.contains(&head_dim); head_dim)
    }

    /// Project `H_ctx` into this layer's context K/V and append to `cache`.
    ///
    /// `offset` is the context position the projected rows start at, so the
    /// context keys carry absolute RoPE positions — mlx-dspark's
    /// `cache.append(self.rope(k, offset=ctx_offset), v)`. **V is not roped**,
    /// which is the ordinary RoPE contract and is stated here because the
    /// symmetry of the line above invites adding it.
    fn update_ctx(
        &self,
        fused: &Tensor,
        offset: usize,
        rotary: &DSparkRotary,
        cache: &mut CtxCache,
    ) -> Result<()> {
        let (b, seq, _) = fused.dims3()?;
        let k = self
            .k_proj
            .forward(fused)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(fused)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.norm_heads(&k, &self.k_norm, b, seq, self.num_kv_heads)?;
        let k = rotary.apply(&k, offset)?;
        cache.append(&k, &v)
    }

    /// [`Self::update_ctx`] for B rows in one pass: `fused` is `[1, Σlens,
    /// hidden]`, row `r`'s `lens[r]` positions laid end to end; `cos`/`sin`
    /// are the table rows of every position (`[Σlens, D/2]`, each at its own
    /// row's context offset); `caches[r]` is row `r`'s cache for this layer.
    /// One `k_proj`/`v_proj`, one norm and one RoPE at `m = Σlens`, then
    /// [`CtxCache::append_rows`].
    fn update_ctx_rows(
        &self,
        fused: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rotary: &DSparkRotary,
        lens: &[usize],
        caches: &mut [&mut CtxCache],
    ) -> Result<()> {
        let (b, t, _) = fused.dims3()?;
        let k = self
            .k_proj
            .forward(fused)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(fused)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.norm_heads(&k, &self.k_norm, b, t, self.num_kv_heads)?;
        let k = rotary.apply_with(&k, cos, sin)?;
        CtxCache::append_rows(caches, lens, &k, &v)
    }

    /// Per-head RMS norm on `[b, seq, heads, head_dim]`, returning
    /// `[b, heads, seq, head_dim]`.
    fn norm_heads(
        &self,
        x: &Tensor,
        norm: &RmsNorm,
        b: usize,
        seq: usize,
        heads: usize,
    ) -> Result<Tensor> {
        let flat = x.reshape((b * seq * heads, self.head_dim))?;
        let normed = norm.forward(&flat)?;
        normed
            .reshape((b, seq, heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Attend the draft block over `[context ‖ block]`.
    ///
    /// **No mask.** The default DSpark head is bidirectional over the whole
    /// block and the whole context: mlx-dspark builds a mask only when
    /// `causal_block` or `sliding_window` is set, and this checkpoint sets
    /// neither (all five `layer_types` are `full_attention`, no
    /// `sliding_window` key). §10.2m states the same from `dflash.cpp` —
    /// *"non-causal, cache-aware"*. That is the whole point of block diffusion:
    /// position `i` of the block is allowed to see position `i+1`, which is
    /// what lets one pass emit γ tokens.
    fn forward(
        &self,
        hidden: &Tensor,
        block_offset: usize,
        rotary: &DSparkRotary,
        cache: &CtxCache,
    ) -> Result<Tensor> {
        let (b, seq, _) = hidden.dims3()?;
        let (q, k, v) = self.project(hidden)?;
        let q = rotary.apply(&q, block_offset)?;
        let k = rotary.apply(&k, block_offset)?;
        let out = self.attend_one(&q, &k, &v, cache)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    /// [`Self::forward`] over B rows in ONE pass: the projections, the norms,
    /// the RoPE (each row at its own position through `cos`/`sin`
    /// `[B, seq, D/2]`) and `o_proj` batched at `m = B·seq`; the attention
    /// one `sdpa_paged_ragged` dispatch when the rows' caches are slots of
    /// one shared paged pool, else per row over its own context — the same
    /// kernels a one-row draft takes, per row (`batched-decode-design.md`
    /// §12.8's batched draft). `caches[b]` is row `b`'s cache for this layer.
    fn forward_rows(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rotary: &DSparkRotary,
        caches: &[&CtxCache],
        active: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _) = hidden.dims3()?;
        if caches.len() != b {
            lloom_tensor::bail!("DSpark forward_rows: {b} rows and {} caches", caches.len());
        }
        let (q, k, v) = self.project(hidden)?;
        let q = rotary.apply_with(&q, cos, sin)?;
        let k = rotary.apply_with(&k, cos, sin)?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        #[cfg(feature = "metal")]
        let out = match CtxCache::paged_attend_rows(caches, &q, &k, &v, active, scale as f32)? {
            Some(out) => out,
            None => self.attend_rows_each(&q, &k, &v, caches)?,
        };
        #[cfg(not(feature = "metal"))]
        let out = {
            let _ = (active, scale);
            self.attend_rows_each(&q, &k, &v, caches)?
        };
        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    /// Each row's attention over its own context, the rows' outputs stacked:
    /// the arm when the caches share no pool.
    fn attend_rows_each(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        caches: &[&CtxCache],
    ) -> Result<Tensor> {
        let mut outs = Vec::with_capacity(caches.len());
        for (r, cache) in caches.iter().enumerate() {
            outs.push(self.attend_one(
                &q.narrow(0, r, 1)?,
                &k.narrow(0, r, 1)?,
                &v.narrow(0, r, 1)?,
                cache,
            )?);
        }
        if outs.len() == 1 {
            return Ok(outs.remove(0));
        }
        Tensor::cat(&outs, 0)
    }

    /// The three projections and the per-head norms over `[b, seq, hidden]`:
    /// `(q, k, v)` in `[b, heads, seq, head_dim]`, `q`/`k` normed and NOT yet
    /// rotated.
    fn project(&self, hidden: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = hidden.dims3()?;
        let q = self
            .q_proj
            .forward(hidden)?
            .reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(hidden)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(hidden)?
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.norm_heads(&q, &self.q_norm, b, seq, self.num_heads)?;
        let k = self.norm_heads(&k, &self.k_norm, b, seq, self.num_kv_heads)?;
        Ok((q, k, v))
    }

    /// One row's block attention over `[context ‖ block]`: `q`/`k`/`v` are
    /// the rotated `[1, heads, seq, D]` projections; the result is
    /// `[1, heads, seq, D]` before `o_proj`.
    fn attend_one(&self, q: &Tensor, k: &Tensor, v: &Tensor, cache: &CtxCache) -> Result<Tensor> {
        let scale = 1f64 / (self.head_dim as f64).sqrt();

        // The paged arm (#708): the block's K/V go into the slots reserved past
        // the context and the tiled kernel reads `[context ‖ block]` through
        // the page table — no concatenation, no copy of the context. Same
        // kernel as the fused arm below, its K/V address one page-walk apart,
        // bit-identical on the same keys (`lloom/tests/sdpa.rs`).
        #[cfg(feature = "metal")]
        if let Some(out) = cache.paged_attend(q, k, v, scale as f32)? {
            return Ok(out);
        }

        // Context first, block second — the key order the context's own RoPE
        // offsets assume, and the order mlx-dspark concatenates in.
        let (k, v) = match (&cache.k, &cache.v) {
            (Some(ck), Some(cv)) => (
                Tensor::cat(&[ck, k], 2)?.contiguous()?,
                Tensor::cat(&[cv, v], 2)?.contiguous()?,
            ),
            _ => (k.clone(), v.clone()),
        };

        // Two arms, one answer. The eager arm below is the reference and the CPU
        // path; the fused `sdpa` arm is Metal-only and byte-parity-tested against
        // it (`lfm2_dspark_backbone.rs`). The block attention is **bidirectional**
        // (§10.2m, no mask), so `do_causal = false` and `mask = None` — the full
        // kernel then attends every key block with no `-inf` masking
        // (`scaled_dot_product_attention.metal`'s `do_causal` is a function
        // constant; false compiles the causal clamp out). The kernel is GQA-native
        // (`kv_head_idx = head / gqa_factor`), so K/V stay at `num_kv_heads` and the
        // two `repeat_kv` expansions the eager arm needs are gone — a dispatch cut
        // on the draft, which is dispatch-bound (§10.2w). `Sdpa` has no CPU impl
        // (`nn/ops.rs`, "SDPA has no cpu impl"), and the drafter runs on both
        // backends (the backbone's CPU/Metal digests must agree, §10.2p), so the
        // arm is gated on the device rather than made the only path.
        let out = if q.device().is_metal()
            && Self::head_dim_instantiated(self.head_dim)
            && !dspark_eager_attn()
        {
            // `q`: [b, num_heads, seq, head_dim]; `k`/`v`: [b, num_kv_heads,
            // ctx+seq, head_dim] — the layout `sdpa` reads, and `q_seq = seq > 1`
            // always (the draft block is a fixed γ ≥ 2, §10.2m), so it takes the
            // full-attention route rather than the seq==1 vector one.
            crate::nn::ops::sdpa(q, &k, &v, None, false, scale as f32, 1.0)?
        } else {
            let k = crate::utils::repeat_kv(k, self.num_kv_groups)?;
            let v = crate::utils::repeat_kv(v, self.num_kv_groups)?;
            let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn = crate::nn::ops::softmax_last_dim(&attn)?;
            attn.matmul(&v)?
        };
        Ok(out)
    }
}

#[derive(Debug)]
struct DSparkLayer {
    self_attn: DSparkAttention,
    mlp: DSparkMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DSparkLayer {
    fn new(cfg: &DSparkConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: DSparkAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: DSparkMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: crate::nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: crate::nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    /// The qwen/llama two-norm block: norm → attn → add → norm → mlp → add.
    fn forward(
        &self,
        hidden: &Tensor,
        block_offset: usize,
        rotary: &DSparkRotary,
        cache: &CtxCache,
    ) -> Result<Tensor> {
        let residual = hidden;
        let h = self.input_layernorm.forward(hidden)?;
        let h = self.self_attn.forward(&h, block_offset, rotary, cache)?;
        let h = (residual + h)?;
        let residual = &h;
        let x = self.post_attention_layernorm.forward(&h)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    /// [`Self::forward`] over B rows at once (`DSparkAttention::forward_rows`).
    fn forward_rows(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rotary: &DSparkRotary,
        caches: &[&CtxCache],
        active: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden;
        let h = self.input_layernorm.forward(hidden)?;
        let h = self
            .self_attn
            .forward_rows(&h, cos, sin, rotary, caches, active)?;
        let h = (residual + h)?;
        let residual = &h;
        let x = self.post_attention_layernorm.forward(&h)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

/// The rank-256 previous-token correction: `logits += W₂·W₁[prev]`.
///
/// # The two weights are the same shape and are NOT interchangeable
///
/// Both `markov_w1.weight` and `markov_w2.weight` are `[128000, 256]` on disk —
/// read from the header, not assumed — and they play different roles:
///
/// * `w1` is an **embedding**, indexed on axis 0 by the previous token id:
///   `W₁[t] → [256]`.
/// * `w2` is a **linear**, contracted on axis 1: `W₂ · r → [128000]`.
///
/// mlx-dspark spells exactly that — `markov_w1 = nn.Embedding(vocab_size,
/// markov_rank)` and `markov_w2 = nn.Linear(markov_rank, out_vocab_size,
/// bias=False)` — and §10.2m's *"low-rank `B = W₁W₂`"* names the factorization
/// without pinning which factor is the lookup.
///
/// **Swapping them is shape-valid and silently wrong**: `[128000, 256]` indexed
/// on axis 0 and contracted on axis 1 both typecheck, the result is a
/// well-formed `[V]` bias, and the only symptom is a worse acceptance rate.
/// That is §10.2n's permutation defect in a second place, so it is pinned by a
/// test against an absolute reference rather than left to the loader's spelling.
#[derive(Debug)]
pub struct MarkovHead {
    /// `[vocab, rank]` — indexed by the previous token.
    w1: Tensor,
    /// `[vocab, rank]` — contracted against `w1`'s row to give a `[vocab]` bias.
    w2: Tensor,
}

impl MarkovHead {
    pub fn new(cfg: &DSparkConfig, vb: VarBuilder) -> Result<Self> {
        let w1 = vb.get((cfg.vocab_size, cfg.markov_rank), "markov_w1.weight")?;
        let w2 = vb.get((cfg.vocab_size, cfg.markov_rank), "markov_w2.weight")?;
        Ok(Self { w1, w2 })
    }

    pub fn from_parts(w1: Tensor, w2: Tensor) -> Self {
        Self { w1, w2 }
    }

    /// `W₁[t]` — the rank-256 embedding of the previous token, which is also
    /// the confidence head's second input block.
    pub fn prev_embedding(&self, token: u32) -> Result<Tensor> {
        self.w1.i(token as usize)
    }

    /// `W₂ · W₁[t]` — the `[vocab]` transition bias added to position `k`'s
    /// base logits.
    pub fn step_bias(&self, token: u32) -> Result<Tensor> {
        let r = self.prev_embedding(token)?;
        let w2 = self.w2.to_dtype(r.dtype())?;
        w2.matmul(&r.reshape((r.elem_count(), 1))?)?.squeeze(1)
    }

    /// `W₁[tokens]` for a `[n]` `U32` tensor of tokens — `[n, rank]`, the
    /// rows gathered on the device.
    pub fn prev_embeddings(&self, tokens: &Tensor) -> Result<Tensor> {
        self.w1.index_select(tokens, 0)
    }

    /// [`Self::step_bias`] for a token that is still on the device — a `[1]`
    /// `U32` tensor, such as the argmax of the previous position. The same
    /// products in the same order; only the row lookup moves off the host.
    pub fn step_bias_at(&self, token: &Tensor) -> Result<Tensor> {
        let r = self.prev_embeddings(token)?;
        let rank = r.dim(1)?;
        let w2 = self.w2.to_dtype(r.dtype())?;
        w2.matmul(&r.reshape((rank, 1))?)?.squeeze(1)
    }

    /// [`Self::step_bias_at`] for B previous tokens at once (`[B]` `U32`):
    /// `[B, vocab]`, one `W₂ · W₁[tokens]ᵀ` for the batch. The same products
    /// per row; the tile's accumulation order is the GEMM's (§2.3.3 tier 2),
    /// which moves a draft only at a near-tie and never the verified stream.
    pub fn step_bias_rows(&self, tokens: &Tensor) -> Result<Tensor> {
        let r = self.prev_embeddings(tokens)?; // [B, rank]
        let w2 = self.w2.to_dtype(r.dtype())?; // [vocab, rank]
        w2.matmul(&r.t()?.contiguous()?)?.t()?.contiguous() // [B, vocab]
    }
}

/// The confidence head's input width: `hidden_size + markov_rank`.
///
/// A free function rather than an expression inside
/// [`ConfidenceHead::new`], so that the constructor and any test resolve
/// through **one** site. `CONTRIBUTING.md` §3.1a rule 3 is that an arm must
/// assert what a site resolves to; with the width inlined, a test asserting
/// `hidden + rank == 2304` restates the arithmetic and a mutation of the
/// constructor survives it. This harness caught exactly that — arm 8 of
/// `mutate_backbone.sh` survived until the width moved here.
pub fn confidence_feature_width(cfg: &DSparkConfig) -> usize {
    cfg.hidden_size + cfg.markov_rank
}

/// Per-position accept probability, `σ(w⊤[h_k ; W₁[x_{k-1}]] + b)`.
///
/// # The checkpoint confirms the formula independently
///
/// `confidence_head.proj.weight` is **`[1, 2304]`** and `2304 = 2048 + 256` —
/// `hidden_size + markov_rank`. §10.2m derives the concatenation from the paper;
/// the shape derives it from the artifact, and the two agree. A head over the
/// hidden state alone would be `[1, 2048]` and the checkpoint ships none.
#[derive(Debug)]
pub struct ConfidenceHead {
    proj: Linear,
    in_dim: usize,
}

impl ConfidenceHead {
    pub fn new(cfg: &DSparkConfig, vb: VarBuilder) -> Result<Self> {
        let in_dim = confidence_feature_width(cfg);
        Ok(Self {
            proj: crate::nn::linear(in_dim, 1, vb.pp("proj"))?,
            in_dim,
        })
    }

    /// The feature width this head was built for.
    pub fn feature_width(&self) -> usize {
        self.in_dim
    }

    /// `features` is `[n, hidden + rank]`; returns `[n]` probabilities.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (_, d) = features.dims2()?;
        if d != self.in_dim {
            lloom_tensor::bail!(
                "DSpark confidence head expects {} features (hidden + markov_rank) and got {d} \
                 (issue #363, DESIGN.md 10.2p)",
                self.in_dim
            );
        }
        crate::nn::ops::sigmoid(&self.proj.forward(features)?.squeeze(D::Minus1)?)
    }
}

/// The DSpark drafter: projection, backbone, Markov head, confidence head.
///
/// # What it is and what it is not
///
/// This is the *proposer*. It has no `embed_tokens` and no `lm_head` — the
/// checkpoint ships neither (62 tensors, verified) — so both are supplied by the
/// **target** through [`crate::models::lfm2::Model::draft_embed`] and
/// [`crate::models::lfm2::Model::draft_lm_head`]. It is not wired into any
/// decode loop: §10.2o's remaining list has the accept-rate measurement after
/// this, and measuring a placement or a fusion whose consumer does not exist is
/// #500's shape (§3.1c).
///
/// # The round, in the order it runs
///
/// 1. [`Self::update_context`] — project the target's five tapped residuals to
///    `H_ctx` and key/value them into each layer's [`CtxCache`]. Once per
///    verified block.
/// 2. [`Self::block_hidden`] — embed `[anchor, mask, …, mask]` with the
///    *target's* table and run the five layers over the γ-wide block, attending
///    the context bidirectionally.
/// 3. [`Self::base_logits`] — the target's `lm_head` over the block's
///    prediction slots.
/// 4. [`Self::sample_block`] — the sequential semi-AR pass that adds the Markov
///    bias position by position.
#[derive(Debug)]
pub struct DSparkDrafter {
    projection: DSparkProjection,
    layers: Vec<DSparkLayer>,
    norm: RmsNorm,
    rotary: DSparkRotary,
    markov: Option<MarkovHead>,
    confidence: Option<ConfidenceHead>,
    cfg: DSparkConfig,
    /// The dtype the backbone computes in — the checkpoint's (BF16). Held so
    /// the projection's F32 output and the target's F16 embedding both meet the
    /// backbone in one place rather than at every call site.
    dtype: DType,
}

impl DSparkDrafter {
    /// Load every drafter weight from the checkpoint's own names.
    ///
    /// `max_seq_len` sizes the RoPE table and must cover the largest absolute
    /// position the draft will attend — context positions included, since the
    /// context K carries absolute offsets.
    ///
    /// # The compute dtype is the `VarBuilder`'s, and it is not the
    /// checkpoint's
    ///
    /// [`CHECKPOINT_DTYPE`] records what the 62 tensors are stored **at**
    /// (BF16); this loads them **as** whatever `vb` was built with, which is
    /// candle's ordinary seam and is deliberately left to the caller for a
    /// reason that is a hard constraint rather than a preference: **the CPU
    /// backend has no BF16 matmul** (`unsupported dtype BF16 for op matmul`),
    /// while the Metal backend does (`metal_backend/mod.rs:1921` maps it to
    /// `GemmDType::BF16`). §14.1's *"changes must not break CPU or CUDA
    /// backends"* means the drafter cannot pin the storage dtype as its compute
    /// dtype.
    ///
    /// **F16 is the sensible default and matches the target**, which the
    /// harnesses load at F16 — so the block embedding the target produces and
    /// the backbone that consumes it agree without a cast. Pass a
    /// `VarBuilder` at F32 for a bit-stable reference run.
    pub fn new(cfg: DSparkConfig, max_seq_len: usize, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device().clone();

        let projection = DSparkProjection::new(
            cfg.hidden_size,
            cfg.target_layer_ids.len(),
            cfg.rms_norm_eps,
            vb.clone(),
        )?;

        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DSparkLayer::new(&cfg, vb_l.pp(i))?);
        }

        let norm = crate::nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary = DSparkRotary::new(&cfg, dtype, &dev, max_seq_len)?;

        let markov = if cfg.markov_rank > 0 {
            Some(MarkovHead::new(&cfg, vb.pp("markov_head"))?)
        } else {
            None
        };
        let confidence = if cfg.enable_confidence_head {
            Some(ConfidenceHead::new(&cfg, vb.pp("confidence_head"))?)
        } else {
            None
        };

        Ok(Self {
            projection,
            layers,
            norm,
            rotary,
            markov,
            confidence,
            cfg,
            dtype,
        })
    }

    pub fn config(&self) -> &DSparkConfig {
        &self.cfg
    }

    pub fn projection(&self) -> &DSparkProjection {
        &self.projection
    }

    pub fn markov(&self) -> Option<&MarkovHead> {
        self.markov.as_ref()
    }

    pub fn confidence_head(&self) -> Option<&ConfidenceHead> {
        self.confidence.as_ref()
    }

    /// One [`CtxCache`] per layer, on the contiguous arm.
    pub fn make_ctx_caches(&self) -> Vec<CtxCache> {
        (0..self.layers.len())
            .map(|_| CtxCache::default())
            .collect()
    }

    /// `n_seqs` sequences' per-layer [`CtxCache`]s on `ctx_kv`.
    ///
    /// The paged arm builds one page pool per layer shared by every
    /// sequence, with `block_size` slots reserved past each context for the
    /// draft block (§10.2m: the block is attended with the context and not
    /// retained), and refuses a non-Metal `device`. The contiguous arm needs
    /// neither the device nor a shared anything.
    pub fn make_ctx_caches_for(
        &self,
        ctx_kv: CtxKv,
        n_seqs: usize,
        device: &Device,
    ) -> Result<Vec<Vec<CtxCache>>> {
        match ctx_kv {
            CtxKv::Contiguous => Ok((0..n_seqs).map(|_| self.make_ctx_caches()).collect()),
            CtxKv::Paged { page_size, shared } => {
                #[cfg(not(feature = "metal"))]
                {
                    let _ = (page_size, shared, device);
                    lloom_tensor::bail!(
                        "DSpark ctx kv `paged` is Metal-only: the pool is read by a Metal kernel \
                         (issue #708)"
                    )
                }
                #[cfg(feature = "metal")]
                {
                    let handles = paged_ctx::PagedCtx::make(
                        self.layers.len(),
                        n_seqs,
                        page_size,
                        self.cfg.block_size,
                        shared,
                        device,
                    )?;
                    Ok(handles
                        .into_iter()
                        .map(|per_layer| {
                            per_layer
                                .into_iter()
                                .map(|p| CtxCache {
                                    k: None,
                                    v: None,
                                    paged: Some(p),
                                })
                                .collect()
                        })
                        .collect())
                }
            }
        }
    }

    /// Project the tapped target residuals and key/value them into every
    /// layer's context cache.
    ///
    /// `taps` is [`crate::models::lfm2::Cache::take_taps`]' output — one tensor
    /// per tapped layer **in `tap_layers`' order**, which the projection
    /// deliberately does not re-sort (§10.2n).
    pub fn update_context(
        &self,
        taps: &[Tensor],
        ctx_offset: usize,
        caches: &mut [CtxCache],
    ) -> Result<()> {
        if caches.len() != self.layers.len() {
            lloom_tensor::bail!(
                "DSpark drafter has {} layers and got {} context caches (issue #363)",
                self.layers.len(),
                caches.len()
            );
        }
        // The projection runs at the TAPS' dtype -- F32 by `Config::tap_dtype`,
        // which §10.2m argues buys acceptance rate rather than correctness --
        // and the backbone runs at the checkpoint's. The cast is here, once,
        // rather than at every call site: a caller that got it wrong would hit
        // a dtype error at best and silently truncate the tap's precision at
        // worst, which is the half of §10.2m's F32 argument a shape check
        // cannot see.
        let fused = self.projection.forward(taps)?.to_dtype(self.dtype)?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            layer
                .self_attn
                .update_ctx(&fused, ctx_offset, &self.rotary, cache)?;
        }
        Ok(())
    }

    /// [`Self::update_context`] for B rows in ONE pass: `taps[r]` is row
    /// `r`'s tapped residuals (`n_tap` tensors of `[1, n_r, hidden]`),
    /// `offsets[r]` the context position its rows key at, `caches[r]` its
    /// per-layer context caches. The rows' taps are laid end to end as one
    /// `[1, Σn_r, ·]` sequence: one projection, then per layer one
    /// `k_proj`/`v_proj`, one norm, one RoPE at every position's own offset
    /// (`DSparkRotary::at_positions`) and one scatter per pool when the rows
    /// share one (`CtxCache::append_rows`). Per row this is `update_context`'s
    /// arithmetic at `m = Σn_r` instead of `m = n_r` — bit-identical on the
    /// GEMM tile set (`n_r ≥ 2`); a one-position row keys through the GEMM
    /// where alone it took the GEMV route, the §10.4 tier-2 seam, which moves
    /// the drafter's context by a reordering and never the verified stream.
    pub fn update_context_rows(
        &self,
        taps: &[&[Tensor]],
        offsets: &[usize],
        caches: &mut [&mut [CtxCache]],
    ) -> Result<()> {
        let b = taps.len();
        if offsets.len() != b || caches.len() != b {
            lloom_tensor::bail!(
                "DSpark update_context_rows: {b} rows of taps, {} offsets, {} cache sets",
                offsets.len(),
                caches.len()
            );
        }
        let n_tap = self.projection.n_tap();
        let mut lens = Vec::with_capacity(b);
        for (r, t) in taps.iter().enumerate() {
            if t.len() != n_tap {
                lloom_tensor::bail!(
                    "DSpark update_context_rows: row {r} has {} taps, the projection consumes \
                     {n_tap}",
                    t.len()
                );
            }
            if caches[r].len() != self.layers.len() {
                lloom_tensor::bail!(
                    "DSpark drafter has {} layers and row {r} has {} context caches",
                    self.layers.len(),
                    caches[r].len()
                );
            }
            let (_, n, _) = t[0].dims3()?;
            if n == 0 {
                lloom_tensor::bail!("DSpark update_context_rows: row {r} has no positions");
            }
            lens.push(n);
        }
        let cat_taps: Vec<Tensor> = (0..n_tap)
            .map(|i| {
                let parts: Vec<&Tensor> = taps.iter().map(|t| &t[i]).collect();
                Tensor::cat(&parts, 1)
            })
            .collect::<Result<_>>()?;
        let fused = self.projection.forward(&cat_taps)?.to_dtype(self.dtype)?;
        let mut positions: Vec<u32> = Vec::with_capacity(lens.iter().sum());
        for (&off, &n) in offsets.iter().zip(&lens) {
            positions.extend((off..off + n).map(|p| p as u32));
        }
        let (cos, sin) = self.rotary.at_positions(&positions)?;
        for (l, layer) in self.layers.iter().enumerate() {
            let mut row_caches: Vec<&mut CtxCache> = caches.iter_mut().map(|c| &mut c[l]).collect();
            layer.self_attn.update_ctx_rows(
                &fused,
                &cos,
                &sin,
                &self.rotary,
                &lens,
                &mut row_caches,
            )?;
        }
        Ok(())
    }

    /// The dtype the backbone computes in — the checkpoint's, BF16 here.
    ///
    /// Exposed because a caller embedding the draft block with the *target's*
    /// table gets the target's dtype (F16) and has to meet the drafter
    /// somewhere. [`Self::embed_block`] does it for them; this is for a caller
    /// that assembles the block itself.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Cast an externally-produced block embedding to the backbone's dtype.
    ///
    /// The block is embedded by the **target** (the drafter ships no
    /// `embed_tokens`), so the two dtypes meet here rather than at the caller.
    pub fn embed_block(&self, block_embeds: &Tensor) -> Result<Tensor> {
        block_embeds.to_dtype(self.dtype)
    }

    /// The block's input embedding: the anchor token, then `width - 1` mask
    /// tokens, embedded with the **target's** raw table.
    ///
    /// `embed` is [`crate::models::lfm2::Model::draft_embed`], passed in rather
    /// than held, because the drafter does not own a target and must not pin one
    /// alive.
    pub fn block_input_ids(&self, anchor: u32, width: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(width);
        ids.push(anchor);
        ids.resize(width, self.cfg.mask_token_id);
        ids
    }

    /// Run the backbone over an already-embedded block.
    ///
    /// Returns `[b, width, hidden]` after the final norm.
    pub fn block_hidden(
        &self,
        block_embeds: &Tensor,
        block_offset: usize,
        caches: &[CtxCache],
    ) -> Result<Tensor> {
        if caches.len() != self.layers.len() {
            lloom_tensor::bail!(
                "DSpark drafter has {} layers and got {} context caches (issue #363)",
                self.layers.len(),
                caches.len()
            );
        }
        let mut h = block_embeds.clone();
        for (layer, cache) in self.layers.iter().zip(caches.iter()) {
            h = layer.forward(&h, block_offset, &self.rotary, cache)?;
        }
        self.norm.forward(&h)
    }

    /// [`Self::block_hidden`] over B rows at once: `block_embeds` is
    /// `[B, width, hidden]`, row `b` attends its own `caches[b]` (one cache
    /// per layer) at its own `offsets[b]`. Returns `[B, width, hidden]` after
    /// the final norm. The projections, norms and MLPs run once at
    /// `m = B·width`; the attention is one dispatch over a shared paged pool,
    /// else per row (`DSparkAttention::forward_rows`). `active`, when the
    /// rows are a subset of a shared pool's sequences, is their `[B]` `U32`
    /// index into it.
    pub fn block_hidden_rows(
        &self,
        block_embeds: &Tensor,
        offsets: &[usize],
        caches: &[&[CtxCache]],
        active: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, width, _) = block_embeds.dims3()?;
        if caches.len() != b || offsets.len() != b {
            lloom_tensor::bail!(
                "DSpark block_hidden_rows: {b} rows, {} cache sets, {} offsets (issue #148)",
                caches.len(),
                offsets.len()
            );
        }
        for (r, c) in caches.iter().enumerate() {
            if c.len() != self.layers.len() {
                lloom_tensor::bail!(
                    "DSpark drafter has {} layers and row {r} has {} context caches",
                    self.layers.len(),
                    c.len()
                );
            }
        }
        let (cos, sin) = self.rotary.rows(offsets, width)?;
        let mut h = block_embeds.clone();
        for (l, layer) in self.layers.iter().enumerate() {
            let row_caches: Vec<&CtxCache> = caches.iter().map(|c| &c[l]).collect();
            h = layer.forward_rows(&h, &cos, &sin, &self.rotary, &row_caches, active)?;
        }
        self.norm.forward(&h)
    }

    /// The block positions whose logits are draft predictions:
    /// `[:, logits_start .. logits_start + cap, :]`.
    ///
    /// `logits_start` is **0** on this checkpoint (anchor-as-pos0), so slot 0
    /// predicts the first draft token. Slicing from 1 instead re-predicts the
    /// token already in hand and lands every draft one position late — see
    /// [`DSparkConfig::logits_start`] for the measured cost of getting it wrong.
    pub fn head_slice(&self, block_hidden: &Tensor, cap: usize) -> Result<Tensor> {
        let (_, width, _) = block_hidden.dims3()?;
        let s = self.cfg.logits_start;
        let cap = cap.min(width.saturating_sub(s));
        if cap == 0 {
            lloom_tensor::bail!(
                "DSpark head_slice: block is {width} wide with logits_start {s}, so there is no \
                 prediction slot (issue #363)"
            );
        }
        block_hidden.narrow(1, s, cap)
    }

    /// The sequential semi-AR pass: greedy argmax over
    /// `base_logits[k] + W₂·W₁[x_{k−1}]`, left to right.
    ///
    /// `base_logits` is `[cap, vocab]` — the target's `lm_head` over
    /// [`Self::head_slice`]'s output. `first_prev` is the anchor token, the
    /// last token the target actually emitted.
    ///
    /// **Sequential is the mechanism, not an implementation choice**: position
    /// `k`'s bias depends on the token sampled at `k−1`, which is precisely the
    /// inter-token dependency the parallel backbone drops and the Markov head
    /// restores (§10.2m step 3). With no Markov head this degenerates to a
    /// per-position argmax and the loop still gives the same answer.
    ///
    /// **The chain stays on the device, and so does the block.** Position
    /// `k`'s bias is looked up from position `k−1`'s argmax *as a tensor*, so
    /// the `cap` positions are one enqueued chain — not one blocking readback
    /// per position, which serialized the host against the GPU `cap` times
    /// per draft. The block is returned as the `[cap]` `U32` tensor it is; the
    /// verify feeds it into the target's window without a host round-trip,
    /// and the host reads the ids back beside the verify's argmax, once. The
    /// arithmetic is unchanged: the same rows, the same products, the same
    /// argmax.
    pub fn sample_block(&self, base_logits: &Tensor, first_prev: &Tensor) -> Result<Tensor> {
        let (cap, _) = base_logits.dims2()?;
        let mut picks = Vec::with_capacity(cap);
        let mut prev = first_prev.clone();
        for k in 0..cap {
            let row = base_logits.i(k)?;
            let row = match &self.markov {
                None => row,
                Some(m) => {
                    let bias = m.step_bias_at(&prev)?.to_dtype(row.dtype())?;
                    (row + bias)?
                }
            };
            let next = row.argmax_keepdim(D::Minus1)?;
            picks.push(next.clone());
            prev = next;
        }
        Tensor::cat(&picks, 0)
    }

    /// [`Self::sample_block`] over B rows: `base_logits` is `[B, cap, vocab]`,
    /// `first_prev` the B anchors (`[B]` `U32`); returns `[B, cap]` `U32`,
    /// the chain on the device — at each position one bias for the batch
    /// (`MarkovHead::step_bias_rows`) and one argmax over `[B, vocab]`.
    pub fn sample_block_rows(&self, base_logits: &Tensor, first_prev: &Tensor) -> Result<Tensor> {
        let (b, cap, _) = base_logits.dims3()?;
        if first_prev.dims1()? != b {
            lloom_tensor::bail!(
                "DSpark sample_block_rows: {b} rows and {} anchors",
                first_prev.dims1()?
            );
        }
        let mut picks = Vec::with_capacity(cap);
        let mut prev = first_prev.clone();
        for k in 0..cap {
            let rows = base_logits.narrow(1, k, 1)?.squeeze(1)?; // [B, vocab]
            let rows = match &self.markov {
                None => rows,
                Some(m) => {
                    let bias = m.step_bias_rows(&prev)?.to_dtype(rows.dtype())?;
                    (rows + bias)?
                }
            };
            let next = rows.argmax(D::Minus1)?; // [B]
            picks.push(next.unsqueeze(1)?);
            prev = next;
        }
        Tensor::cat(&picks, 1)
    }

    /// Per-position accept probabilities for a drafted block.
    ///
    /// `block_hidden` is [`Self::head_slice`]'s output squeezed to
    /// `[cap, hidden]`; `prev_tokens[k]` is the token preceding draft position
    /// `k` (the anchor for `k = 0`, then the drafted tokens), a `[cap]` `U32`
    /// tensor so a device-resident block feeds it without a readback.
    pub fn confidence_scores(
        &self,
        block_hidden: &Tensor,
        prev_tokens: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Some(head) = &self.confidence else {
            return Ok(None);
        };
        let Some(markov) = &self.markov else {
            lloom_tensor::bail!(
                "DSpark confidence head needs the Markov head's `w1` for its second feature \
                 block, and this drafter has no Markov head (issue #363)"
            );
        };
        let (cap, _) = block_hidden.dims2()?;
        if prev_tokens.dims1()? != cap {
            lloom_tensor::bail!(
                "DSpark confidence: {cap} positions and {} previous tokens (issue #363)",
                prev_tokens.dims1()?
            );
        }
        // Both halves of the concat are brought to the backbone's dtype rather
        // than to `block_hidden`'s, because the head's own weights are at that
        // dtype and a caller may hand us an F32 slice.
        let prev = markov.prev_embeddings(prev_tokens)?.to_dtype(self.dtype)?;
        let hidden = block_hidden.to_dtype(self.dtype)?;
        let feats = Tensor::cat(&[&hidden, &prev], D::Minus1)?.contiguous()?;
        Ok(Some(head.forward(&feats)?))
    }

    /// [`Self::confidence_scores`] over B rows: `block_hidden` `[B, cap,
    /// hidden]`, `prev_tokens` `[B, cap]`; returns `[B, cap]`. The head is
    /// per position, so the rows are one `[B·cap]` batch through it.
    pub fn confidence_scores_rows(
        &self,
        block_hidden: &Tensor,
        prev_tokens: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (b, cap, hidden) = block_hidden.dims3()?;
        let flat = block_hidden.reshape((b * cap, hidden))?;
        let prev = prev_tokens.reshape(b * cap)?;
        Ok(match self.confidence_scores(&flat, &prev)? {
            Some(scores) => Some(scores.reshape((b, cap))?),
            None => None,
        })
    }
}

/// Every drafter parameter, for the memory accounting §9.1 wants.
///
/// Re-derived against the checkpoint's own 327 707 521 by
/// `measurements/issue-363-raw/verify_backbone.py`, which is the point: a
/// parameter count computed from the config and a parameter count summed from
/// the header are two quantities, and agreeing is evidence the config describes
/// the file.
pub fn drafter_params(cfg: &DSparkConfig) -> usize {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let kv = cfg.num_key_value_heads * hd;
    let q = cfg.num_attention_heads * hd;

    let per_layer = h * q          // q_proj
        + h * kv                   // k_proj
        + h * kv                   // v_proj
        + q * h                    // o_proj
        + hd                       // q_norm
        + hd                       // k_norm
        + h                        // input_layernorm
        + h                        // post_attention_layernorm
        + 3 * h * cfg.intermediate_size; // gate, up, down

    let markov = if cfg.markov_rank > 0 {
        2 * cfg.vocab_size * cfg.markov_rank
    } else {
        0
    };
    let confidence = if cfg.enable_confidence_head {
        (h + cfg.markov_rank) + 1 // proj weight [1, hidden+rank] and bias [1]
    } else {
        0
    };

    projection_params(h, cfg.target_layer_ids.len())
        + cfg.num_hidden_layers * per_layer
        + h // final norm
        + markov
        + confidence
}

// ---------------------------------------------------------------------------
// The `Speculator` seam (`RESTRUCTURING_DESIGN.md` §3.4.1), over the DSpark
// family.
// ---------------------------------------------------------------------------

/// One sequence's cross-round drafter state — the `Vec<PerSeq>` the trait keeps
/// PRIVATE (`RESTRUCTURING_DESIGN.md` §3.4.1). Never leaves this module.
#[derive(Debug)]
struct PerSeq {
    /// Per-layer context KV for this sequence's drafter.
    ctx: Vec<CtxCache>,
    /// The absolute context position the next `update_context` keys at — grows
    /// by the accepted length each round (the verify example's `ctx_positions`).
    ctx_offset: usize,
    /// The last committed token — the block anchor for the next `draft`.
    last: u32,
    /// Whether this slot has been `begin`-seeded.
    active: bool,
    /// The ids at the positions the context has been keyed from, in order —
    /// the target's token log as this drafter saw it. What `begin_at`
    /// compares a request's reused prefix against before keeping the context
    /// for it: a slot that served another sequence holds a well-formed
    /// context for the wrong tokens, which no gate sees (issue #894).
    keyed: Vec<u32>,
    /// The prompt tokens `begin` was given and the seeding observes have not
    /// yet keyed, in order; they move to `keyed` as their taps arrive.
    pending: Vec<u32>,
    /// `[1, block_size]` `U32`: the draft block's input ids, the mask ids
    /// written once and slot 0 (the anchor) rewritten per round through the
    /// mapped pointer — where `Tensor::new` per draft was an unpooled driver
    /// allocation plus a residency add and commit (§10.2z-iii's drains).
    block_slot: Tensor,
    /// `[1]` `U32`: the anchor as a device tensor for `sample_block`'s first
    /// `prev` and the confidence head, same discipline.
    anchor_slot: Tensor,
}

impl PerSeq {
    /// The block ids for `anchor`, on the device: the slot rewritten in place
    /// when the mapped write is available, a fresh upload otherwise (CPU).
    /// Written between rounds only — the previous draft's gather has run by
    /// the time the verify that consumed it was read back.
    fn block_ids(&self, drafter: &DSparkDrafter, anchor: u32, width: usize) -> Result<Tensor> {
        // SAFETY: see above — no queued dispatch reads slot 0 at this point.
        if unsafe { self.block_slot.write_host_slice::<u32>(0, &[anchor]) } {
            return Ok(self.block_slot.clone());
        }
        let ids = drafter.block_input_ids(anchor, width);
        Tensor::new(ids.as_slice(), self.block_slot.device())?.unsqueeze(0)
    }

    /// The anchor as a `[1]` device tensor, same slot discipline.
    fn anchor_tensor(&self, anchor: u32) -> Result<Tensor> {
        // SAFETY: as `block_ids`.
        if unsafe { self.anchor_slot.write_host_slice::<u32>(0, &[anchor]) } {
            return Ok(self.anchor_slot.clone());
        }
        Tensor::new(&[anchor], self.anchor_slot.device())
    }
}

/// A DSpark proposer driving a fixed set of sequences — the [`Speculator`]
/// implementation of the DSpark family (`RESTRUCTURING_DESIGN.md` §3.4).
///
/// # It borrows the target's two projections rather than owning a target
///
/// DSpark ships no `embed_tokens` and no `lm_head` (`DSparkDrafter`'s docs); the
/// block is embedded with the target's raw table and the base logits are the
/// target's raw head. §3.4's `Speculator::draft(active)` gives the call no target
/// handle — it is modelled on llama.cpp's self-contained drafter, which carries
/// its own embed and head. **This is a §3.4/tree seam** (see this PR's finding):
/// the DSpark family shares the target's embed/`lm_head`, which §3.4 covers for
/// the *KV* cache ([`KvSource::SharedWithTarget`]) but not for the embed/head. The
/// faithful resolution that keeps "the drafter does not own a target" is to hold
/// **clones of the two shared projection tensors** (cheap — both are `Arc`-backed),
/// supplied at construction by the caller that has the target. A target is never
/// pinned alive.
///
/// # The round, and where the context is seeded
///
/// The driver loop is `draft → verify → observe` ([`crate::nn::Session::speculative_step`]).
/// The first `draft` needs the context already keyed, so a caller seeds it once
/// after the target's prefill: `begin(seq, prompt)` sets the block anchor to the
/// prompt's last token, then a single `observe(seq, prefill_taps, &[])` keys the
/// context from the prefill's taps (empty `accepted` leaves the anchor in place).
/// Thereafter each `observe` re-keys from the verified round's taps and advances
/// the anchor. This mirrors the `lfm2-dspark-verify` example's
/// seed-then-loop shape, factored behind the seam.
pub struct DSparkSpeculator {
    drafter: DSparkDrafter,
    /// The target's token embedding, shared (the drafter ships none).
    embed: TracedEmbedding,
    /// The target's raw `lm_head`, shared. The proposal head must stay the raw
    /// projection (`Model::draft_lm_head`'s contract).
    lm_head: TracedLinear,
    seqs: Vec<PerSeq>,
    /// The draft-block width `γ` and the prediction cap, read once from the
    /// drafter config.
    block_size: usize,
    max_draft: usize,
    /// Whether the driver reads `Draft::confidence` this round
    /// ([`Speculator::want_confidence`]). The head is a forward and a host
    /// readback per round; off, the round's only readback is the verify's.
    confidence_wanted: bool,
    /// The context KV arm every slot's caches were built on.
    ctx_kv: CtxKv,
    /// The batched draft's device slots, one row per sequence, rewritten in
    /// place through the mapped pointer between rounds (`PerSeq::block_slot`'s
    /// discipline): every sequence's block ids `[n_seq, block_size]`, its
    /// anchor `[n_seq]`, and the active rows' indices `[n_seq]`.
    blocks: Tensor,
    anchors: Tensor,
    active_slot: Tensor,
    /// Drafts taken batched (`draft` over more than one row), for a gate that
    /// asserts the batched arm ran.
    batched_drafts: u64,
    /// Context re-keys taken batched (`observe_rows` over more than one
    /// row), the same gate's other half.
    batched_observes: u64,
    /// `begin_at` calls on a reused prefix whose context could not be kept —
    /// the slot held another sequence's, or less of this one than the target
    /// kept — so the suffix was keyed over a gap. Zero on a chat session that
    /// kept its drafter; nonzero is where an acceptance rate went.
    ctx_resets: u64,
}

impl DSparkSpeculator {
    /// Build a proposer for `n_seq` server slots, the context KV arm read
    /// from the tuning ([`CtxKv::resolve`]: `auto` unset, paged per sequence
    /// on Metal).
    ///
    /// `embed` and `lm_head` are the **target's** — cloned here so the drafter
    /// borrows the two shared projections without owning the target
    /// ([`crate::models::lfm2::Model::draft_embedding`],
    /// [`crate::models::lfm2::Model::lm_head_linear`]).
    pub fn new(
        drafter: DSparkDrafter,
        embed: TracedEmbedding,
        lm_head: TracedLinear,
        n_seq: usize,
    ) -> Result<Self> {
        let ctx_kv = CtxKv::resolve(lm_head.weight().device())?;
        Self::new_with_ctx(drafter, embed, lm_head, n_seq, ctx_kv)
    }

    /// [`Self::new`] with the context KV arm chosen by the caller.
    pub fn new_with_ctx(
        drafter: DSparkDrafter,
        embed: TracedEmbedding,
        lm_head: TracedLinear,
        n_seq: usize,
        ctx_kv: CtxKv,
    ) -> Result<Self> {
        let block_size = drafter.config().block_size;
        let max_draft = drafter.config().max_draft();
        let device = lm_head.weight().device().clone();
        let mask_block = drafter.block_input_ids(0, block_size);
        let seqs = drafter
            .make_ctx_caches_for(ctx_kv, n_seq, &device)?
            .into_iter()
            .map(|ctx| {
                Ok(PerSeq {
                    ctx,
                    ctx_offset: 0,
                    last: 0,
                    active: false,
                    keyed: Vec::new(),
                    pending: Vec::new(),
                    // Uploads, not `zeros`: a pooled `zeros` is a fill blit that
                    // may still be queued when the first mapped write lands.
                    block_slot: Tensor::new(mask_block.as_slice(), &device)?.unsqueeze(0)?,
                    anchor_slot: Tensor::new(&[0u32], &device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = n_seq.max(1);
        let all_blocks: Vec<u32> = mask_block
            .iter()
            .copied()
            .cycle()
            .take(rows * block_size)
            .collect();
        let blocks = Tensor::new(all_blocks.as_slice(), &device)?.reshape((rows, block_size))?;
        let anchors = Tensor::new(vec![0u32; rows].as_slice(), &device)?;
        let active_slot = Tensor::new(vec![0u32; rows].as_slice(), &device)?;
        Ok(Self {
            drafter,
            embed,
            lm_head,
            seqs,
            block_size,
            max_draft,
            confidence_wanted: false,
            ctx_kv,
            blocks,
            anchors,
            active_slot,
            batched_drafts: 0,
            batched_observes: 0,
            ctx_resets: 0,
        })
    }

    /// Drafts taken over more than one row in one drafter pass.
    pub fn batched_drafts(&self) -> u64 {
        self.batched_drafts
    }

    /// Context re-keys taken over more than one row in one pass.
    /// Reused-prefix requests whose context had to be reset (`begin_at`).
    pub fn ctx_resets(&self) -> u64 {
        self.ctx_resets
    }

    /// Slot `seq`'s context length, in keyed positions, and the position the
    /// next key lands at — equal unless the context was reset over a gap.
    pub fn ctx_state(&self, seq: SeqId) -> Option<(usize, usize)> {
        self.seqs
            .get(seq.0)
            .map(|s| (s.ctx.first().map(|c| c.len()).unwrap_or(0), s.ctx_offset))
    }

    pub fn batched_observes(&self) -> u64 {
        self.batched_observes
    }

    /// One-scatter appends the drafter's shared paged pool has taken (per
    /// layer per batched re-key); `None` off the shared paged arm.
    pub fn batched_appends(&self) -> Option<u64> {
        let CtxKv::Paged { shared: true, .. } = self.ctx_kv else {
            return None;
        };
        self.seqs
            .first()
            .and_then(|s| s.ctx.first())
            .and_then(|c| c.batched_appends())
    }

    /// The context KV arm this proposer runs on.
    pub fn ctx_kv(&self) -> CtxKv {
        self.ctx_kv
    }

    /// Compile the drafter's pipelines off the clock, the drafter counterpart of
    /// [`crate::nn::decode::Session::warmup`].
    ///
    /// The target's warmup compiles the *target's* GEMM/attention pipelines, but
    /// the drafter runs its own: the backbone attention and projections at
    /// `m = block_size` (γ), the shared `lm_head` over the block, the encoder
    /// `fc` matmul, the Markov/confidence heads. None is a target shape, so all
    /// are cold on the first real round — a fixed cost the first `--dspark` TTFT
    /// carries every process (the pipeline cache does not outlive one). Running a
    /// real `draft_one` here pays that compile at load, where no user clock runs;
    /// `synchronize` forces it to *complete* rather than leak into round 0.
    ///
    /// A draft is **read-only** on the sequence (`draft_one(&self, &PerSeq)`; the
    /// KV write is `observe`'s), so this leaves nothing behind — no reset needed.
    /// It drafts on slot 0, which exists from construction (`last = 0`, empty
    /// context, `ctx_offset = 0`): a block off the anchor `0` over an empty
    /// context, which exercises every drafter pipeline a real block will.
    ///
    /// A no-op when there are no slots. `synchronize` is a device wait, so this
    /// is a load-time call, never a per-round one.
    pub fn warmup(&mut self, device: &Device) -> Result<()> {
        if self.seqs.is_empty() {
            return Ok(());
        }
        // Force the confidence head's pipeline too when the drafter has one, so
        // a run that later asks for confidence pays no cold compile mid-stream.
        // Restored after, so this does not change what round 0 computes.
        let was_wanted = self.confidence_wanted;
        self.confidence_wanted = self.drafter.confidence_head().is_some();
        // Slot 0 exists (checked above) and a draft is read-only on it, so this
        // borrow does not outlive the immutable draft and nothing is mutated.
        let result = self.draft_one(&self.seqs[0]);
        self.confidence_wanted = was_wanted;
        result?;
        // Complete the compile here rather than leaking it into round 0.
        device.synchronize()?;
        Ok(())
    }

    /// [`Self::warmup`] for the batched draft: one `n`-row draft over the
    /// first `n` slots (empty contexts), so the `m = n·γ` GEMMs, the ragged
    /// paged attention and the batched Markov chain compile off the clock.
    pub fn warmup_rows(&mut self, n: usize, device: &Device) -> Result<()> {
        let n = n.min(self.seqs.len());
        if n < 2 {
            return self.warmup(device);
        }
        let was_wanted = self.confidence_wanted;
        self.confidence_wanted = self.drafter.confidence_head().is_some();
        let active: Vec<SeqId> = (0..n).map(SeqId).collect();
        let result = self.draft_rows(&active);
        self.confidence_wanted = was_wanted;
        result?;
        device.synchronize()?;
        Ok(())
    }

    /// The batched draft: every row's block in ONE drafter pass — the
    /// block ids and anchors gathered from the per-sequence slots, one
    /// `[B, γ]` embedding, `block_hidden_rows`, one `lm_head` over the B
    /// blocks' prediction slots, the Markov chain over `[B, vocab]` per
    /// position — and B `Draft`s, each row's tokens a `[cap]` view of the
    /// `[B, cap]` block on the device.
    fn draft_rows(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let b = active.len();
        let device = self.blocks.device().clone();
        let mut mapped = true;
        let mut ids_host: Vec<u32> = Vec::new();
        let mut anchors_host: Vec<u32> = Vec::with_capacity(b);
        let mut offsets = Vec::with_capacity(b);
        for &seq in active {
            let s = self
                .seqs
                .get(seq.0)
                .ok_or_else(|| lloom_tensor::Error::msg(format!("DSpark: no slot {}", seq.0)))?;
            // SAFETY: between rounds — the previous draft's gathers have run
            // by the time the verify that consumed it was read back.
            mapped &= unsafe {
                self.blocks
                    .write_host_slice::<u32>(seq.0 * self.block_size, &[s.last])
            };
            mapped &= unsafe { self.anchors.write_host_slice::<u32>(seq.0, &[s.last]) };
            anchors_host.push(s.last);
            offsets.push(s.ctx_offset);
        }
        let prefix = active.iter().enumerate().all(|(i, s)| s.0 == i);
        let (ids, anchors, active_t) = if mapped && prefix {
            (
                self.blocks.narrow(0, 0, b)?,
                self.anchors.narrow(0, 0, b)?,
                None,
            )
        } else if mapped {
            let idx: Vec<u32> = active.iter().map(|s| s.0 as u32).collect();
            // SAFETY: as above.
            let ok = unsafe { self.active_slot.write_host_slice::<u32>(0, &idx) };
            let idx_t = if ok {
                self.active_slot.narrow(0, 0, b)?
            } else {
                Tensor::new(idx.as_slice(), &device)?
            };
            (
                self.blocks.index_select(&idx_t, 0)?,
                self.anchors.index_select(&idx_t, 0)?,
                Some(idx_t),
            )
        } else {
            // Unmapped storage (CPU): upload the batch.
            for &a in &anchors_host {
                ids_host.extend(self.drafter.block_input_ids(a, self.block_size));
            }
            let idx: Vec<u32> = active.iter().map(|s| s.0 as u32).collect();
            (
                Tensor::new(ids_host.as_slice(), &device)?.reshape((b, self.block_size))?,
                Tensor::new(anchors_host.as_slice(), &device)?,
                if prefix {
                    None
                } else {
                    Some(Tensor::new(idx.as_slice(), &device)?)
                },
            )
        };
        let caches: Vec<&[CtxCache]> = active
            .iter()
            .map(|s| self.seqs[s.0].ctx.as_slice())
            .collect();
        let block_embeds = self.drafter.embed_block(&self.embed.forward(&ids)?)?;
        let hidden =
            self.drafter
                .block_hidden_rows(&block_embeds, &offsets, &caches, active_t.as_ref())?;
        let pred = self.drafter.head_slice(&hidden, self.max_draft)?; // [B, cap, hidden]
        let (_, cap, h) = pred.dims3()?;
        let base_logits = self
            .lm_head
            .forward(
                &pred
                    .reshape((b * cap, h))?
                    .to_dtype(self.embed.embeddings().dtype())?,
            )?
            .to_dtype(DType::F32)?
            .reshape((b, cap, ()))?;
        let tokens = self.drafter.sample_block_rows(&base_logits, &anchors)?; // [B, cap]
        let confidence: Vec<Vec<f32>> = match self.drafter.confidence_head() {
            Some(_) if self.confidence_wanted => {
                let head = self.drafter.head_slice(&hidden, cap)?;
                let prev =
                    Tensor::cat(&[&anchors.unsqueeze(1)?, &tokens.narrow(1, 0, cap - 1)?], 1)?;
                match self.drafter.confidence_scores_rows(&head, &prev)? {
                    Some(scores) => scores.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    None => vec![Vec::new(); b],
                }
            }
            _ => vec![Vec::new(); b],
        };
        self.batched_drafts += 1;
        let mut out = Vec::with_capacity(b);
        for (r, conf) in confidence.into_iter().enumerate() {
            out.push(Draft {
                tokens: DraftTokens::Device(tokens.narrow(0, r, 1)?.squeeze(0)?),
                confidence: conf,
            });
        }
        Ok(out)
    }

    /// Draft one block for one already-seeded sequence — the verify example's
    /// per-round draft body, factored behind the seam.
    fn draft_one(&self, seq: &PerSeq) -> Result<Draft> {
        let block_tensor = seq.block_ids(&self.drafter, seq.last, self.block_size)?;
        let block_embeds = self
            .drafter
            .embed_block(&self.embed.forward(&block_tensor)?)?;
        let hidden = self
            .drafter
            .block_hidden(&block_embeds, seq.ctx_offset, &seq.ctx)?;
        let pred = self
            .drafter
            .head_slice(&hidden, self.max_draft)?
            .squeeze(0)?;
        let base_logits = self
            .lm_head
            .forward(&pred.to_dtype(self.embed.embeddings().dtype())?)?
            .to_dtype(DType::F32)?;
        // The block stays on the device: the verify takes it as a tensor, and
        // the host sees the ids in the verify's own readback.
        let anchor = seq.anchor_tensor(seq.last)?;
        let tokens = self.drafter.sample_block(&base_logits, &anchor)?;

        // The confidence head, when present AND asked for, is the family's
        // `Draft.confidence` hint (§3.4.1). `prev_tokens[k]` precedes draft
        // position `k`: the anchor for `k = 0`, then the drafted tokens —
        // assembled on the device from the block. The hint is host-side by
        // contract, so producing it is the draft's one readback; the driver
        // asks only when its `p_min` will read it. EMPTY otherwise.
        let confidence = match self.drafter.confidence_head() {
            Some(_) if self.confidence_wanted => {
                let cap = tokens.dims1()?;
                let head = self.drafter.head_slice(&hidden, cap)?.squeeze(0)?;
                let prev = Tensor::cat(&[&anchor, &tokens.narrow(0, 0, cap - 1)?], 0)?;
                match self.drafter.confidence_scores(&head, &prev)? {
                    Some(scores) => scores.to_dtype(DType::F32)?.to_vec1::<f32>()?,
                    None => Vec::new(),
                }
            }
            _ => Vec::new(),
        };

        Ok(Draft {
            tokens: DraftTokens::Device(tokens),
            confidence,
        })
    }
}

impl Speculator for DSparkSpeculator {
    type Probe = LayerTaps<5>;
    type Family = crate::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        self.seqs.len()
    }

    fn kv_source(&self) -> KvSource {
        // DSpark keeps its own draft-KV — the per-layer `CtxCache`
        // (`RESTRUCTURING_DESIGN.md` §3.4: `KvSource::Own` names a real
        // `CtxCache: KvCache`).
        KvSource::Own
    }

    fn max_draft(&self) -> usize {
        self.max_draft
    }

    fn want_confidence(&mut self, wanted: bool) {
        self.confidence_wanted = wanted;
    }

    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        self.begin_at(seq, prompt, &[])
    }

    fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
        let pos = kept.len();
        let ctx_resets = &mut self.ctx_resets;
        let s = self
            .seqs
            .get_mut(seq.0)
            .ok_or_else(|| lloom_tensor::Error::msg(format!("DSpark: no slot {}", seq.0)))?;
        // The context is kept for the reused prefix only if it was keyed from
        // exactly those tokens, at least that far; then it is cut back to
        // them (a rewind is a length decrement, §10.2z). Otherwise it is
        // reset and the suffix is keyed at its true positions over a gap —
        // the positions are the target's either way, the acceptance rate
        // pays for the gap and nothing else does.
        let have = s.ctx.first().map(|c| c.len()).unwrap_or(0);
        let follow = pos > 0
            && s.keyed.len() >= pos
            && s.keyed[..pos] == kept[..]
            && have >= pos
            && s.ctx_offset >= pos;
        if follow {
            if have > pos {
                for c in s.ctx.iter_mut() {
                    c.truncate(have - pos)?;
                }
            }
            s.keyed.truncate(pos);
        } else {
            for c in s.ctx.iter_mut() {
                c.clear();
            }
            s.keyed.clear();
            if pos > 0 {
                *ctx_resets += 1;
            }
        }
        s.ctx_offset = pos;
        s.pending.clear();
        s.pending.extend_from_slice(prompt);
        s.last = *prompt.last().or(kept.last()).unwrap_or(&0);
        s.active = true;
        Ok(())
    }

    fn context_len(&self, seq: SeqId) -> Option<usize> {
        self.seqs.get(seq.0).map(|s| s.ctx_offset)
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        for &seq in active {
            let s = self
                .seqs
                .get(seq.0)
                .ok_or_else(|| lloom_tensor::Error::msg(format!("DSpark: no slot {}", seq.0)))?;
            if !s.active {
                lloom_tensor::bail!("DSpark: draft on sequence {} before begin()", seq.0);
            }
        }
        // One row is the one-row draft, byte for byte; more than one is ONE
        // drafter pass over every row's block (`draft_rows`): the drafter's
        // weights swept once for the batch, the block attention of a shared
        // paged pool in one dispatch — the round's serial drafts were the
        // term that flattened continuous DSpark past B = 2 (§12.8).
        if active.len() == 1 {
            return Ok(vec![self.draft_one(&self.seqs[active[0].0])?]);
        }
        self.draft_rows(active)
    }

    fn observe(
        &mut self,
        seq: SeqId,
        probe: &<Self::Probe as crate::nn::speculator::TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        // `probe` is the target's per-round taps (`Cache::take_taps`), already
        // trimmed to the accepted rows by the driver. Re-key the drafter's
        // context from them and advance the context offset and anchor.
        //
        // The drafter and the per-seq caches are disjoint fields, so borrow them
        // separately rather than through one `&mut self` reach — `update_context`
        // reads `&self.drafter` while writing `&mut s.ctx`.
        let drafter = &self.drafter;
        let s = self
            .seqs
            .get_mut(seq.0)
            .ok_or_else(|| lloom_tensor::Error::msg(format!("DSpark: no slot {}", seq.0)))?;
        let rows = probe.first().map(|t| t.dim(1)).transpose()?.unwrap_or(0);
        drafter.update_context(probe, s.ctx_offset, &mut s.ctx)?;
        s.note_keyed(rows, accepted);
        Ok(())
    }

    fn observe_rows(
        &mut self,
        rows: &[crate::nn::speculator::ObserveRow<'_, Self::Probe>],
    ) -> Result<()> {
        // One row is the one-row re-key, byte for byte; more than one is ONE
        // pass over every row's accepted taps (`update_context_rows`): the
        // projection and each layer's K/V once for the batch, the keys in one
        // scatter per pool — where a call per row was B serial passes, the
        // round's term after the batched draft (§12.10).
        if rows.len() < 2 {
            for (seq, probe, accepted) in rows {
                self.observe(*seq, probe, accepted)?;
            }
            return Ok(());
        }
        let drafter = &self.drafter;
        let mut taken: Vec<Option<&mut PerSeq>> = self.seqs.iter_mut().map(Some).collect();
        let mut picked: Vec<&mut PerSeq> = Vec::with_capacity(rows.len());
        for (seq, _, _) in rows {
            match taken.get_mut(seq.0).and_then(|s| s.take()) {
                Some(s) => picked.push(s),
                None => lloom_tensor::bail!(
                    "DSpark: observe_rows on slot {} (no such slot, or one slot twice in a round)",
                    seq.0
                ),
            }
        }
        let taps: Vec<&[Tensor]> = rows.iter().map(|(_, p, _)| p.as_slice()).collect();
        let offsets: Vec<usize> = picked.iter().map(|s| s.ctx_offset).collect();
        let n_rows: Vec<usize> = rows
            .iter()
            .map(|(_, p, _)| {
                p.first()
                    .map(|t| t.dim(1))
                    .transpose()
                    .map(|n| n.unwrap_or(0))
            })
            .collect::<Result<_>>()?;
        {
            let mut caches: Vec<&mut [CtxCache]> =
                picked.iter_mut().map(|s| s.ctx.as_mut_slice()).collect();
            drafter.update_context_rows(&taps, &offsets, &mut caches)?;
        }
        for ((s, (_, _, accepted)), n) in picked.iter_mut().zip(rows).zip(n_rows) {
            s.note_keyed(n, accepted);
        }
        self.batched_observes += 1;
        Ok(())
    }
}

impl PerSeq {
    /// `rows` positions were just keyed from `rows` taps, after which the
    /// anchor is `accepted`'s last token. While the prompt is being seeded the
    /// rows are its tokens; in a round they are the window's input tokens the
    /// target kept — the anchor the round started from and the accepted
    /// tokens before the last, which becomes the next anchor.
    fn note_keyed(&mut self, rows: usize, accepted: &[u32]) {
        let from_prompt = rows.min(self.pending.len());
        self.keyed.extend(self.pending.drain(..from_prompt));
        let in_round = rows - from_prompt;
        if in_round > 0 {
            self.keyed.push(self.last);
            let n = in_round.saturating_sub(1).min(accepted.len());
            self.keyed.extend_from_slice(&accepted[..n]);
        }
        self.ctx_offset += rows;
        if let Some(&t) = accepted.last() {
            self.last = t;
        }
    }
}

impl DSparkSpeculator {
    /// Draft `k` positions per round instead of the drafter's block width:
    /// the block is still predicted whole and its first `k` proposed, so the
    /// verify window is `k + 1` positions. Clamped to `1..=block`. A lever
    /// for wide batches, where a `B × (K + 1)` verify against ~2.4 accepted
    /// of 9 on natural text costs more than it returns (lloom #148, record
    /// §8.3): fewer positions per row for the same accepted prefix.
    pub fn set_max_draft(&mut self, k: usize) {
        let block = self.drafter.config().max_draft();
        self.max_draft = k.clamp(1, block);
    }
}
