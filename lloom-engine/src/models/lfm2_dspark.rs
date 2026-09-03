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

use crate::nn::{Linear, RmsNorm, VarBuilder};
use lloom_tensor::{DType, Device, IndexOp, Module, Result, Tensor, D};

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
            norm_weight,
            eps,
            n_tap: cols / hidden,
            hidden,
        })
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
        let w = self.fc.to_dtype(cat.dtype())?.t()?.contiguous()?;
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
#[derive(Debug, Clone)]
pub struct DSparkRotary {
    cos: Tensor,
    sin: Tensor,
    interleaved: bool,
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
        let x = x.contiguous()?;
        if self.interleaved {
            crate::nn::rotary_emb::rope_i(&x, &cos, &sin)
        } else {
            crate::nn::rotary_emb::rope(&x, &cos, &sin)
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
}

impl CtxCache {
    pub fn len(&self) -> usize {
        match &self.k {
            Some(k) => k.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
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

    /// Drop the last `n` context positions — the draft-side half of a rollback.
    ///
    /// The draft has **no conv state** (all five `layer_types` are
    /// `full_attention`, §10.2l), so a rejection rolls back KV only, which is
    /// what this does. The target's conv rollback is the other half and is
    /// §10.2m's K+1 snapshots.
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        let len = self.len();
        let keep = len.saturating_sub(n);
        if keep == len {
            return Ok(());
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
        let q = rotary.apply(&q, block_offset)?;
        let k = rotary.apply(&k, block_offset)?;

        // Context first, block second — the key order the context's own RoPE
        // offsets assume, and the order mlx-dspark concatenates in.
        let (k, v) = match (&cache.k, &cache.v) {
            (Some(ck), Some(cv)) => (
                Tensor::cat(&[ck, &k], 2)?.contiguous()?,
                Tensor::cat(&[cv, &v], 2)?.contiguous()?,
            ),
            _ => (k, v),
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = crate::nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b, seq, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
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

    /// One [`CtxCache`] per layer.
    pub fn make_ctx_caches(&self) -> Vec<CtxCache> {
        (0..self.layers.len())
            .map(|_| CtxCache::default())
            .collect()
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
    pub fn sample_block(&self, base_logits: &Tensor, first_prev: u32) -> Result<Vec<u32>> {
        let (cap, _) = base_logits.dims2()?;
        let mut out = Vec::with_capacity(cap);
        let mut prev = first_prev;
        for k in 0..cap {
            let row = base_logits.i(k)?;
            let row = match &self.markov {
                None => row,
                Some(m) => {
                    let bias = m.step_bias(prev)?.to_dtype(row.dtype())?;
                    (row + bias)?
                }
            };
            let next = row.argmax(D::Minus1)?.to_scalar::<u32>()?;
            out.push(next);
            prev = next;
        }
        Ok(out)
    }

    /// Per-position accept probabilities for a drafted block.
    ///
    /// `block_hidden` is [`Self::head_slice`]'s output squeezed to
    /// `[cap, hidden]`; `prev_tokens[k]` is the token preceding draft position
    /// `k` (the anchor for `k = 0`, then the drafted tokens).
    pub fn confidence_scores(
        &self,
        block_hidden: &Tensor,
        prev_tokens: &[u32],
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
        if prev_tokens.len() != cap {
            lloom_tensor::bail!(
                "DSpark confidence: {cap} positions and {} previous tokens (issue #363)",
                prev_tokens.len()
            );
        }
        let mut rows = Vec::with_capacity(cap);
        for &t in prev_tokens {
            rows.push(markov.prev_embedding(t)?);
        }
        // Both halves of the concat are brought to the backbone's dtype rather
        // than to `block_hidden`'s, because the head's own weights are at that
        // dtype and a caller may hand us an F32 slice.
        let prev = Tensor::stack(&rows, 0)?.to_dtype(self.dtype)?;
        let hidden = block_hidden.to_dtype(self.dtype)?;
        let feats = Tensor::cat(&[&hidden, &prev], D::Minus1)?.contiguous()?;
        Ok(Some(head.forward(&feats)?))
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
