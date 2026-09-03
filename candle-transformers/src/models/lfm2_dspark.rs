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
//! # The representation, and a witness conflict this module does not resolve
//!
//! The tap feeding this is the **layer-input residual** (§10.2n), captured
//! before the block runs. That follows llama.cpp PR #27383 — `t_layer_inp[il]`,
//! whose consumer passes the drafter's indices unshifted and whose own comment
//! reads *"turn on extraction of the target layers' **input** embeddings"*
//! (`common/speculative.cpp:1046`).
//!
//! **mlx-dspark disagrees by one layer.** Its LFM2 body captures *after* the
//! block (`target.py:_body_lfm2` — `h = layer(h, …)` then `captured.append(h)`),
//! with no compensating index shift, so for `[2, 9, 17, 21, 27]` it feeds the
//! *outputs* of those layers where llama.cpp feeds their *inputs*. Both are
//! §10.2m-designated witnesses and they cannot both be right.
//!
//! **This module takes neither side**, because it does not have to: the tap is
//! its input, [`DSparkProjection::forward`] projects whatever it is given, and
//! the discriminator is the `H_ctx` ulp-match against a reference — which is
//! what this module makes reachable and what §10.2o registers as the open
//! question. Recorded rather than guessed (`CONTRIBUTING.md` §6): picking one
//! silently is precisely the class §10.2n calls *"the defect no gate here can
//! see"*, since a wrong-but-stable `H_ctx` degrades acceptance with every gate
//! green.

use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

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
            candle::bail!(
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
            candle::bail!(
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
                candle::bail!(
                    "DSpark projection tap {i} is {h} wide, expected {} (issue #363)",
                    self.hidden
                );
            }
        }

        // Concatenate along the feature axis, which is what `fc.weight`'s
        // 10240 columns are: `[H(l1); ...; H(l5)]`, not a token-stream concat.
        // mlx-dspark's `_body_lfm2` does the same with `axis=-1`.
        let cat = Tensor::cat(taps, candle::D::Minus1)?;

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
        candle_nn::ops::rms_norm(&fc_out, &norm_weight, self.eps as f32)
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
