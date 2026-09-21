//! Qwen Sparse Attention (QSA) — the full-attention layers (every 4th).
//!
//! Dense QK-norm GQA attention — QK-norm, partial RoPE, output gate, `wo` —
//! restricted, past the indexer's dense range, to the cells the Lightning
//! indexer selects. The selection is `budget + ratio − 1` cells wide
//! (`qwen4exp.cpp:665`, vLLM `QSAIndexer::output_width`), so for
//! `n_kv ≤ budget + ratio − 1` (2051) every cell is selected and the dense
//! path IS QSA, byte for byte; the indexer only cuts past that.
//!
//! The indexer ([`Indexer`]; reference `build_qsa_top_k`, vLLM
//! `indexer_qsa.py` + `ops/qsa_indexer.py`) is the three-stage pipeline the
//! model doc sketches, kept O(1) per token where the reference's first
//! implementation was O(n_kv) (llama.cpp #28734):
//!
//! 1. INDEX — a raw indexer key per token (`k_proj`, cached raw); one
//!    compressed key per COMPLETE block of `ratio` tokens: the block's raw
//!    keys averaged, RMS-normed (`k_norm`), roped at the block's first
//!    position, appended to a persistent compressed-key cache the moment the
//!    block completes — never rebuilt.
//! 2. SELECT — the query's `n_heads` indexer heads (`q_proj`, `q_norm`, roped
//!    at the query's position) score every visible complete block
//!    `Σ_h relu(q_h · K_C[b])` in fp32 (no learned head weights — neither
//!    reference has them); the top `budget / ratio` blocks by score, plus
//!    the incomplete tail block (always visible), are the selected cells.
//!    Only blocks entirely at or before the query are visible (causal before
//!    rank).
//! 3. ATTEND — the selected rows of K and V are gathered (`index_select`,
//!    O(budget)) and the dense attention runs over them.
//!
//! No novel Metal kernel: a selection of existing blocks (§3.6a) — the
//! projections, `rms_norm`, `arg_sort`, `index_select`, softmax + matmul.
//! The score is fp32 by construction (§15.2a: precision-mandatory).

use super::config::Config;
use crate::nn::{ops, RmsNorm};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Module, Result, Tensor, D};

/// Per-layer growing KV state for one sequence (full-attn layers only). Unlike
/// DeltaNet's fixed `S`, KV grows with context. The rows live in a
/// preallocated slab written in place (`nn::kv_cache`, grown by a slab when
/// full), so a decode step appends without copying the cache — no O(context)
/// per token (§3.3). The paged/pooled KV is the perf path, deferred with caching.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// `[n_kv_heads, seq, head_dim]` for k and v, the sequence on dim 1.
    inner: crate::nn::kv_cache::KvCache,
    /// The indexer's raw keys `[1, seq, idx_dim]`, one per token.
    raw_keys: crate::nn::kv_cache::Cache,
    /// The indexer's compressed keys `[1, n_blocks, idx_dim]`, one per
    /// COMPLETE block, normed and roped — the persistent incremental cache.
    blocks: crate::nn::kv_cache::Cache,
}

impl Default for KvCache {
    fn default() -> Self {
        Self::with_capacity(KV_SLAB_TOKENS)
    }
}

/// Tokens a KV slab holds before it grows by another slab.
pub const KV_SLAB_TOKENS: usize = 1024;

impl KvCache {
    pub fn with_capacity(tokens: usize) -> Self {
        let tokens = tokens.max(1);
        Self {
            inner: crate::nn::kv_cache::KvCache::new(1, tokens),
            raw_keys: crate::nn::kv_cache::Cache::new(1, tokens),
            blocks: crate::nn::kv_cache::Cache::new(1, tokens.div_ceil(4).max(1)),
        }
    }
    pub fn reset(&mut self) {
        self.inner.reset();
        self.raw_keys.reset();
        self.blocks.reset();
    }
    pub fn len(&self) -> usize {
        self.inner.current_seq_len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Compressed (complete) blocks the indexer holds.
    pub fn n_blocks(&self) -> usize {
        self.blocks.current_seq_len()
    }
    /// `(tokens, compressed blocks)` held — what a speculative window records.
    pub fn lens(&self) -> (usize, usize) {
        (self.len(), self.n_blocks())
    }
    /// Keep the first `tokens` positions (k, v, raw indexer keys) and the
    /// first `blocks` compressed keys — a rejected speculative tail.
    pub fn truncate(&mut self, tokens: usize, blocks: usize) {
        self.inner.truncate(tokens);
        self.raw_keys.truncate(tokens);
        self.blocks.truncate(blocks);
    }
    /// The live `(k, v, raw indexer keys, compressed keys)` views, for a
    /// state probe (`None` where nothing was written).
    pub fn state_views(&self) -> Result<[Option<Tensor>; 4]> {
        Ok([
            self.inner.k()?,
            self.inner.v()?,
            self.raw_keys.current_data()?,
            self.blocks.current_data()?,
        ])
    }
    /// Append `k`, `v` `[n_kv_heads, n, head_dim]`; returns views of the whole
    /// cache `[n_kv_heads, len, head_dim]`.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.inner.append(k, v)
    }
    /// The rows `sel [n_sel]` (u32, all `< len`) of k and v:
    /// `[n_kv_heads, n_sel, head_dim]` each. Gathers from the slab itself
    /// (contiguous), so the cost is the selection's size, not the context's.
    fn gather(&self, sel: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match self.inner.k_cache().all_data() {
            Some(k) => k.index_select(sel, 1)?,
            None => lloom_tensor::bail!("qsa gather on an empty cache"),
        };
        let v = match self.inner.v_cache().all_data() {
            Some(v) => v.index_select(sel, 1)?,
            None => lloom_tensor::bail!("qsa gather on an empty cache"),
        };
        Ok((k, v))
    }
}

/// The Lightning indexer of one full-attention layer (`indexer.*`).
#[derive(Debug, Clone)]
pub struct Indexer {
    /// `indexer.q_proj` `[n_heads·head_dim, hidden]`.
    q_proj: QMatMul,
    /// `indexer.k_proj` `[head_dim, hidden]` (one key head).
    k_proj: QMatMul,
    /// RMS over `head_dim` (`indexer.q_norm` / `k_norm`, converter-folded γ).
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    head_dim: usize,
    /// `indexer_budget` (2048): selected cells from complete blocks.
    budget: usize,
    /// `indexer_compress_ratio` (4): tokens per block.
    ratio: usize,
    /// Rotary dims (the main attention's `n_rot`, ≤ `head_dim`).
    n_rot: usize,
    rope_theta: f64,
}

impl Indexer {
    pub fn new(
        q_proj: QMatMul,
        k_proj: QMatMul,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        cfg: &Config,
    ) -> Result<Self> {
        let n_rot = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        if cfg.indexer_compress_ratio == 0
            || !cfg
                .indexer_budget
                .is_multiple_of(cfg.indexer_compress_ratio)
        {
            lloom_tensor::bail!(
                "indexer budget {} / ratio {}",
                cfg.indexer_budget,
                cfg.indexer_compress_ratio
            )
        }
        if n_rot > cfg.indexer_head_dim {
            lloom_tensor::bail!("indexer head_dim {} < n_rot {n_rot}", cfg.indexer_head_dim)
        }
        Ok(Self {
            q_proj,
            k_proj,
            q_norm,
            k_norm,
            n_heads: cfg.indexer_n_heads,
            head_dim: cfg.indexer_head_dim,
            budget: cfg.indexer_budget,
            ratio: cfg.indexer_compress_ratio,
            n_rot,
            rope_theta: cfg.rope_theta,
        })
    }

    /// Cells past which the selection is narrower than the context:
    /// `budget + ratio − 1`. At or below it every cell is selected.
    pub fn dense_limit(&self) -> usize {
        self.budget + self.ratio - 1
    }

    /// Blocks selected per query.
    fn block_topk(&self) -> usize {
        self.budget / self.ratio
    }

    /// Append the raw indexer keys of `x2 [seq, hidden]` and complete every
    /// block the new tokens finish: pooled raw keys → `k_norm` → rope at the
    /// block's first position → the compressed cache. O(new blocks), never a
    /// rebuild.
    fn observe(&self, x2: &Tensor, kv: &mut KvCache) -> Result<()> {
        let seq = x2.dim(0)?;
        let k_raw = self.k_proj.forward(x2)?.reshape((1, seq, self.head_dim))?;
        kv.raw_keys.append(&k_raw)?;
        let n_tokens = kv.raw_keys.current_seq_len();
        let complete = n_tokens / self.ratio;
        let have = kv.n_blocks();
        if complete <= have {
            return Ok(());
        }
        let n_new = complete - have;
        let raw = match kv.raw_keys.current_data()? {
            Some(r) => r,
            None => return Ok(()),
        };
        let start = have * self.ratio;
        let members = raw.narrow(1, start, n_new * self.ratio)?.reshape((
            n_new,
            self.ratio,
            self.head_dim,
        ))?;
        let pooled = (members.sum(1)? * (1.0 / self.ratio as f64))?; // [n_new, head_dim]
        let normed = self.k_norm.forward(&pooled.contiguous()?)?;
        let positions: Vec<usize> = (0..n_new).map(|b| (have + b) * self.ratio).collect();
        let (cos, sin) = rope_cs_at(self.n_rot, self.rope_theta, &positions, x2.device())?;
        let half = self.n_rot / 2;
        let roped = partial_rope_seq(
            &normed.reshape((n_new, 1, self.head_dim))?,
            &cos.reshape((n_new, 1, half))?,
            &sin.reshape((n_new, 1, half))?,
            self.n_rot,
        )?; // [n_new, 1, head_dim]
        kv.blocks.append(&roped.reshape((1, n_new, self.head_dim))?)
    }

    /// The selected cell indices for the query at `pos` (its `x1 [1, hidden]`),
    /// or `None` when the context is within the dense range: the top
    /// `budget / ratio` visible complete blocks by `Σ_h relu(q_h · K_C[b])`
    /// (fp32; ties by the sort's order), expanded to cells, plus the
    /// incomplete tail `[(pos+1)/ratio·ratio, pos]`. `[n_sel]` u32, ascending
    /// within each block, tail last.
    fn select(&self, x1: &Tensor, pos: usize, kv: &KvCache) -> Result<Option<Tensor>> {
        let n_kv = pos + 1;
        if n_kv <= self.dense_limit() {
            return Ok(None);
        }
        let dev = x1.device();
        let visible = n_kv / self.ratio; // complete blocks at or before pos
        if kv.n_blocks() < visible {
            lloom_tensor::bail!(
                "indexer: {} compressed blocks, {visible} visible at pos {pos}",
                kv.n_blocks()
            )
        }
        let k_c = match kv.blocks.current_data()? {
            Some(b) => b.narrow(1, 0, visible)?.reshape((visible, self.head_dim))?,
            None => lloom_tensor::bail!("indexer: no compressed keys at pos {pos}"),
        };
        // The query's indexer heads: q_proj → q_norm → rope(pos).
        let q = self
            .q_proj
            .forward(x1)?
            .reshape((self.n_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let (cos, sin) = rope_cs_at(self.n_rot, self.rope_theta, &[pos], dev)?;
        let half = self.n_rot / 2;
        let q = partial_rope_seq(
            &q.reshape((1, self.n_heads, self.head_dim))?,
            &cos.reshape((1, 1, half))?,
            &sin.reshape((1, 1, half))?,
            self.n_rot,
        )?
        .reshape((self.n_heads, self.head_dim))?;
        // fp32 score per block: Σ_h relu(K_C · q_hᵀ).
        let k32 = k_c.to_dtype(DType::F32)?.contiguous()?;
        let score = if score_without_matmul() {
            // A/B: the same dot as a broadcast product + last-dim sum.
            let prod = k32.reshape((visible, 1, self.head_dim))?.broadcast_mul(
                &q.to_dtype(DType::F32)?
                    .reshape((1, self.n_heads, self.head_dim))?,
            )?;
            prod.sum(2)?.relu()?.sum(1)? // [visible]
        } else {
            let q32 = q.to_dtype(DType::F32)?.t()?.contiguous()?; // [head_dim, n_heads]
            k32.matmul(&q32)?.relu()?.sum(1)? // [visible]
        };
        let topk = self.block_topk().min(visible);
        // Descending by score = ascending by −score (the ascending sort is the
        // arm every width runs on Metal).
        let order = score.neg()?.arg_sort_last_dim(true)?; // [visible] u32
        let blocks = order.narrow(0, 0, topk)?.contiguous()?; // [topk]
                                                              // Expand to cells: block·ratio + (0..ratio).
        let base = (blocks.to_dtype(DType::F32)? * self.ratio as f64)?
            .reshape((topk, 1))?
            .broadcast_as((topk, self.ratio))?;
        let offs = Tensor::arange(0f32, self.ratio as f32, dev)?
            .reshape((1, self.ratio))?
            .broadcast_as((topk, self.ratio))?;
        let cells = (base + offs)?
            .reshape(topk * self.ratio)?
            .to_dtype(DType::U32)?;
        let tail_start = n_kv / self.ratio * self.ratio;
        let sel = if tail_start < n_kv {
            let tail = Tensor::arange(tail_start as u32, n_kv as u32, dev)?;
            Tensor::cat(&[&cells, &tail], 0)?
        } else {
            cells
        };
        Ok(Some(sel))
    }
}

/// One QSA (full-attention) layer's weights.
#[derive(Debug, Clone)]
pub struct QsaAttention {
    /// Fused Q proj → `[q | gate]` interleaved per head (`attn_q`, 2×head_dim/head).
    wq: QMatMul,
    /// K, V projs (`attn_k`, `attn_v`).
    wk: QMatMul,
    wv: QMatMul,
    /// Output proj (`attn_output`).
    wo: QMatMul,
    /// QK-norm (RMS over head_dim), `attn_q_norm` / `attn_k_norm`.
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Rotary dims = `head_dim * partial_rotary_factor` (64 of 256).
    n_rot: usize,
    rope_theta: f64,
    /// The Lightning indexer; without it the layer is dense at every context
    /// (correct only within the dense range).
    indexer: Option<Indexer>,
}

impl QsaAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: QMatMul,
        wk: QMatMul,
        wv: QMatMul,
        wo: QMatMul,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        cfg: &Config,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            n_rot: (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize,
            rope_theta: cfg.rope_theta,
            indexer: None,
        }
    }

    /// Attach the layer's Lightning indexer.
    pub fn with_indexer(mut self, indexer: Indexer) -> Self {
        self.indexer = Some(indexer);
        self
    }

    pub fn indexer(&self) -> Option<&Indexer> {
        self.indexer.as_ref()
    }

    /// The context past which this layer's attention is sparse (`None`: no
    /// indexer, dense everywhere).
    pub fn dense_limit(&self) -> Option<usize> {
        self.indexer.as_ref().map(|i| i.dense_limit())
    }

    /// The cells the indexer would select for a query at `pos` (its input
    /// `x [1, 1, hidden]`) against the cache as it stands after that position's
    /// keys were observed — `None` within the dense range. Test/probe surface
    /// for the selection itself.
    pub fn selection(&self, x: &Tensor, pos: usize, kv: &KvCache) -> Result<Option<Tensor>> {
        match &self.indexer {
            Some(ix) => ix.select(&x.reshape((1, x.dim(D::Minus1)?))?, pos, kv),
            None => Ok(None),
        }
    }

    /// Precompute per-rotary-pair (cos, sin) for `pos`, each `[n_rot/2]`.
    /// Text-only decode: mRoPE reduces to standard RoPE (all sections share the
    /// text position), so partial RoPE over the first `n_rot` dims is correct.
    fn rope_cs(&self, pos: usize, dev: &lloom_tensor::Device) -> Result<(Tensor, Tensor)> {
        let (c, s) = rope_cs_at(self.n_rot, self.rope_theta, &[pos], dev)?;
        let half = self.n_rot / 2;
        Ok((c.reshape(half)?, s.reshape(half)?))
    }

    /// Partial RoPE on the first `n_rot` dims of `x [h, head_dim]`, the tail
    /// unrotated (see [`partial_rope_seq`]).
    fn partial_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (h, hd) = x.dims2()?;
        let half = self.n_rot / 2;
        partial_rope_seq(
            &x.reshape((1, h, hd))?,
            &cos.reshape((1, 1, half))?,
            &sin.reshape((1, 1, half))?,
            self.n_rot,
        )?
        .reshape((h, hd))
    }

    /// One decode step: `x [1, 1, hidden]` in, `[1, 1, hidden]` out, appending to
    /// this layer's `kv`. Dense QK-norm GQA attention (= QSA for ctx ≤ budget).
    pub fn decode(&self, x: &Tensor, kv: &mut KvCache) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let dev = x.device();
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let x1 = x.reshape((1, hidden))?; // [1, hidden]

        // wq → [q | gate] interleaved per head: layout is
        // [head0_q(hd), head0_gate(hd), head1_q(hd), ...]. Split per head.
        let qg = self.wq.forward(&x1)?.reshape((nh, 2 * hd))?; // [nh, 2*hd]
        let q = qg.narrow(1, 0, hd)?.contiguous()?; // [nh, hd]
        let gate = qg.narrow(1, hd, hd)?.contiguous()?; // [nh, hd]

        let k = self.wk.forward(&x1)?.reshape((nkv, hd))?; // [nkv, hd]
        let v = self.wv.forward(&x1)?.reshape((nkv, hd))?; // [nkv, hd]

        // QK-norm (RMS over head_dim).
        let q = self.q_norm.forward(&q)?; // [nh, hd]
        let k = self.k_norm.forward(&k)?; // [nkv, hd]

        // partial mRoPE at this position.
        let pos = kv.len();
        let (cos, sin) = self.rope_cs(pos, dev)?;
        let q = self.partial_rope(&q, &cos, &sin)?; // [nh, hd]
        let k = self.partial_rope(&k, &cos, &sin)?; // [nkv, hd]

        // Append k, v to the cache: [nkv, seq, hd].
        let k_step = k.reshape((nkv, 1, hd))?;
        let v_step = v.reshape((nkv, 1, hd))?;
        let (k_all, v_all) = kv.append(&k_step, &v_step)?;

        // The indexer sees this token, completes its block if it does, and —
        // past the dense range — names the cells to attend to.
        let sel = match &self.indexer {
            Some(ix) => {
                ix.observe(&x1, kv)?;
                ix.select(&x1, pos, kv)?
            }
            None => None,
        };
        if let Some(sel) = &sel {
            if check_selection() {
                let n_kv = pos + 1;
                let ids = sel.to_vec1::<u32>()?;
                let bad = ids.iter().filter(|&&c| c as usize >= n_kv).count();
                let mut seen = std::collections::HashSet::new();
                let dup = ids.iter().filter(|&&c| !seen.insert(c)).count();
                if bad > 0 || dup > 0 {
                    lloom_tensor::bail!(
                        "qsa selection at pos {pos}: {bad} cells >= {n_kv}, {dup} duplicates, of {} (first {:?})",
                        ids.len(),
                        &ids[..ids.len().min(8)]
                    )
                }
                eprintln!(
                    "SEL pos={pos} n={} blocks={} first={:?} last={:?}",
                    ids.len(),
                    kv.n_blocks(),
                    &ids[..4],
                    &ids[ids.len() - 4..]
                );
            }
        }
        let (k_all, v_all) = match &sel {
            Some(sel) => kv.gather(sel)?, // [nkv, n_sel, hd] each
            None => (k_all, v_all),
        };
        let seq = k_all.dim(1)?;

        // GQA expand kv heads nkv → nh (tile, the reference repeat), then dense
        // attention: scores = q·kᵀ·scale, softmax, ·v. No mask needed at decode
        // (the single query attends to all past keys, all causal).
        let rep = nh / nkv;
        let k_exp = k_all
            .reshape((nkv, 1, seq, hd))?
            .broadcast_as((nkv, rep, seq, hd))?
            .reshape((nh, seq, hd))?;
        let v_exp = v_all
            .reshape((nkv, 1, seq, hd))?
            .broadcast_as((nkv, rep, seq, hd))?
            .reshape((nh, seq, hd))?;

        let scale = 1.0 / (hd as f64).sqrt();
        let q_h = q.reshape((nh, 1, hd))?.to_dtype(lloom_tensor::DType::F32)?; // [nh,1,hd]
        let k_h = k_exp.to_dtype(lloom_tensor::DType::F32)?; // [nh,seq,hd]
        let scores = (q_h.matmul(&k_h.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh,1,seq]
        let probs = ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_exp.to_dtype(lloom_tensor::DType::F32)?)?; // [nh,1,hd]
        let ctx = ctx.reshape((1, nh * hd))?.to_dtype(x.dtype())?;

        // output gate: · sigmoid(gate), then wo.
        let gate_flat = gate.reshape((1, nh * hd))?;
        let gated = (ctx * ops::sigmoid(&gate_flat)?)?;
        self.wo.forward(&gated)?.reshape((1, 1, hidden))
    }

    /// A whole chunk `x [1, seq, hidden]` → `[1, seq, hidden]`: the same dense
    /// QK-norm GQA composition as [`Self::decode`] over every position at once,
    /// causal over the chunk, attending to everything already cached. Valid
    /// while the cache stays within the indexer's dense range.
    pub fn prefill(&self, x: &Tensor, kv: &mut KvCache) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if b != 1 {
            lloom_tensor::bail!("qsa prefill takes one sequence, got {:?}", x.shape())
        }
        let dev = x.device();
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let x2 = x.reshape((seq, hidden))?;

        let qg = self.wq.forward(&x2)?.reshape((seq, nh, 2 * hd))?;
        let q = qg.narrow(2, 0, hd)?.contiguous()?; // [seq, nh, hd]
        let gate = qg.narrow(2, hd, hd)?.contiguous()?;
        let k = self.wk.forward(&x2)?.reshape((seq, nkv, hd))?;
        let v = self.wv.forward(&x2)?.reshape((seq, nkv, hd))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let past = kv.len();
        let positions: Vec<usize> = (past..past + seq).collect();
        let (cos, sin) = rope_cs_at(self.n_rot, self.rope_theta, &positions, dev)?; // [seq, half]
        let half = self.n_rot / 2;
        let (cos, sin) = (cos.reshape((seq, 1, half))?, sin.reshape((seq, 1, half))?);
        let q = partial_rope_seq(&q, &cos, &sin, self.n_rot)?; // [seq, nh, hd]
        let k = partial_rope_seq(&k, &cos, &sin, self.n_rot)?; // [seq, nkv, hd]

        let k_new = k.transpose(0, 1)?.contiguous()?; // [nkv, seq, hd]
        let v_new = v.transpose(0, 1)?.contiguous()?;
        let (k_all, v_all) = kv.append(&k_new, &v_new)?;
        let kv_len = k_all.dim(1)?;
        if let Some(ix) = &self.indexer {
            // Every position of a chunk attends densely, so the chunk must
            // stay within the range where dense IS the selection; the model
            // runs positions past it one at a time.
            if kv_len > ix.dense_limit() {
                lloom_tensor::bail!(
                    "qsa prefill: {kv_len} cells exceed the dense range {}; decode past it",
                    ix.dense_limit()
                )
            }
            ix.observe(&x2, kv)?;
        }

        let rep = nh / nkv;
        let f32 = lloom_tensor::DType::F32;
        let k_exp = k_all
            .reshape((nkv, 1, kv_len, hd))?
            .broadcast_as((nkv, rep, kv_len, hd))?
            .reshape((nh, kv_len, hd))?
            .to_dtype(f32)?;
        let v_exp = v_all
            .reshape((nkv, 1, kv_len, hd))?
            .broadcast_as((nkv, rep, kv_len, hd))?
            .reshape((nh, kv_len, hd))?
            .to_dtype(f32)?;
        let q_h = q.transpose(0, 1)?.contiguous()?.to_dtype(f32)?; // [nh, seq, hd]
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q_h.matmul(&k_exp.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh, seq, kv_len]
                                                                                    // Causal over the chunk: query t (at position past + t) sees keys ≤ past + t.
        let scores = if seq > 1 {
            let mask = causal_mask(seq, kv_len, past, dev)?
                .broadcast_as((nh, seq, kv_len))?
                .contiguous()?;
            let neg = Tensor::full(f32::NEG_INFINITY, (nh, seq, kv_len), dev)?;
            mask.where_cond(&scores, &neg)?
        } else {
            scores
        };
        let probs = ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_exp)?; // [nh, seq, hd]
        let ctx = ctx
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq, nh * hd))?
            .to_dtype(x.dtype())?;
        let gated = (ctx * ops::sigmoid(&gate.reshape((seq, nh * hd))?)?)?;
        self.wo.forward(&gated)?.reshape((1, seq, hidden))
    }
}

/// A/B instrument (`QWEN4EXP_SCORE_NO_MATMUL`, read once): the block score
/// as a broadcast product + reduce instead of the `[n_blocks, 128] × [128, 4]`
/// matmul. Never the shipping path.
fn score_without_matmul() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_SCORE_NO_MATMUL").is_some())
}

/// Read the selection back and check it every position (`QWEN4EXP_CHECK_SEL`,
/// read once): a debugging instrument behind a gate (§3.3a) — a host sync per
/// full layer per token, never on the shipping path.
fn check_selection() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_CHECK_SEL").is_some())
}

/// `(cos, sin)` `[positions.len(), n_rot/2]` for partial RoPE at `positions`.
fn rope_cs_at(
    n_rot: usize,
    theta: f64,
    positions: &[usize],
    dev: &lloom_tensor::Device,
) -> Result<(Tensor, Tensor)> {
    let half = n_rot / 2;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for &pos in positions {
        for i in 0..half {
            let inv = 1f32 / (theta as f32).powf(2.0 * i as f32 / n_rot as f32);
            let f = inv * pos as f32;
            cos.push(f.cos());
            sin.push(f.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (positions.len(), half), dev)?,
        Tensor::from_vec(sin, (positions.len(), half), dev)?,
    ))
}

/// Partial RoPE on `x [seq, h, head_dim]` with `cos`/`sin` `[seq, 1, n_rot/2]`:
/// the first `n_rot` dims rotate in NEOX (rotate-half) pairs
/// `(x[i], x[i + n_rot/2])` — `y[i] = x[i]·cos_i − x[i+h]·sin_i`,
/// `y[i+h] = x[i]·sin_i + x[i+h]·cos_i` — the tail passes through.
///
/// The model's rope is IMRoPE: the mRoPE SECTIONS are interleaved across the
/// pair index (t, y, x, t, y, x, …), and the pairs themselves are NEOX-ordered
/// (ggml `GGML_ROPE_TYPE_IMROPE` → `rotate_pairs(n_dims, n_dims/2)`; vLLM's
/// rotary is `is_neox_style`). For text every section carries the same
/// position, so the interleaving is invisible and only the pairing matters.
/// Adjacent-pair (GPT-J) rotation is the wrong operator here: it changes no
/// shape and passes the short-context text gate, so it is pinned by a test.
fn partial_rope_seq(x: &Tensor, cos: &Tensor, sin: &Tensor, n_rot: usize) -> Result<Tensor> {
    let hd = x.dim(2)?;
    let half = n_rot / 2;
    let x_lo = x.narrow(2, 0, half)?;
    let x_hi = x.narrow(2, half, half)?;
    let y_lo = (x_lo.broadcast_mul(cos)? - x_hi.broadcast_mul(sin)?)?;
    let y_hi = (x_lo.broadcast_mul(sin)? + x_hi.broadcast_mul(cos)?)?;
    if n_rot == hd {
        return Tensor::cat(&[&y_lo, &y_hi], 2);
    }
    let pass = x.narrow(2, n_rot, hd - n_rot)?;
    Tensor::cat(&[&y_lo, &y_hi, &pass], 2)
}

/// `[seq, kv_len]` u8: 1 where key `j` is visible to query `t` at position
/// `past + t`, i.e. `j <= past + t`.
fn causal_mask(
    seq: usize,
    kv_len: usize,
    past: usize,
    dev: &lloom_tensor::Device,
) -> Result<Tensor> {
    let mut m = vec![0u8; seq * kv_len];
    for t in 0..seq {
        for j in 0..kv_len.min(past + t + 1) {
            m[t * kv_len + j] = 1;
        }
    }
    Tensor::from_vec(m, (seq, kv_len), dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{DType, Device};

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let n = out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.011 + seed).sin()) * 0.1)
            .collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }
    fn rms(dim: usize, dev: &Device) -> RmsNorm {
        RmsNorm::new(Tensor::ones(dim, DType::F32, dev).unwrap(), 1e-6)
    }

    /// QSA decodes end to end and its KV grows: shapes thread, output finite, and
    /// the cache length advances one per step (context accumulates).
    #[test]
    fn qsa_decodes_and_kv_grows() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let attn = QsaAttention::new(
            lin(nh * hd * 2, h, 1.0, &dev), // wq → [q|gate]
            lin(nkv * hd, h, 2.0, &dev),    // wk
            lin(nkv * hd, h, 3.0, &dev),    // wv
            lin(h, nh * hd, 4.0, &dev),     // wo
            rms(hd, &dev),
            rms(hd, &dev),
            &cfg,
        );
        let mut kv = KvCache::default();
        let tok = |s: f32| {
            Tensor::from_vec(
                (0..h)
                    .map(|i| ((i as f32 * 0.05 + s).sin()) * 0.3)
                    .collect::<Vec<f32>>(),
                (1, 1, h),
                &dev,
            )
            .unwrap()
        };
        assert_eq!(kv.len(), 0);
        let o0 = attn.decode(&tok(0.0), &mut kv).unwrap();
        assert_eq!(o0.dims(), &[1, 1, h]);
        assert_eq!(kv.len(), 1);
        let o1 = attn.decode(&tok(1.0), &mut kv).unwrap();
        assert_eq!(kv.len(), 2);
        let o1v = o1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(o1v.iter().all(|x| x.is_finite()));
        // step 1 attends to 2 keys (grows); a nonzero output.
        assert!(o1v.iter().any(|&x| x.abs() > 1e-9));
    }

    /// Partial RoPE rotates only the first n_rot dims, leaving the tail
    /// unchanged (partial_rotary_factor 0.25 → 64 of 256; tiny: 4 of 16).
    #[test]
    fn partial_rope_leaves_tail_unrotated() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let attn = QsaAttention::new(
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            rms(cfg.head_dim, &dev),
            rms(cfg.head_dim, &dev),
            &cfg,
        );
        let hd = cfg.head_dim;
        let n_rot = (hd as f64 * cfg.partial_rotary_factor) as usize;
        let x = Tensor::ones((1, hd), DType::F32, &dev).unwrap(); // [h=1, hd]
                                                                  // pos 0: cos=1, sin=0 → rope is identity, whole vector unchanged.
        let (c0, s0) = attn.rope_cs(0, &dev).unwrap();
        let y0 = attn.partial_rope(&x, &c0, &s0).unwrap();
        let y0v = y0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            y0v.iter().all(|&v| (v - 1.0).abs() < 1e-5),
            "pos0 rope must be identity"
        );
        // pos 5: the tail dims (>= n_rot) stay exactly 1.0 (unrotated).
        let (c, s) = attn.rope_cs(5, &dev).unwrap();
        let y = attn.partial_rope(&x, &c, &s).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &yv[n_rot..] {
            assert!((v - 1.0).abs() < 1e-5, "tail dim rotated: {yv:?}");
        }
    }
    /// The rope pairs dim `i` with dim `i + n_rot/2` (NEOX), at the pair's own
    /// frequency, and leaves the tail alone — pinned against a hand
    /// computation so an adjacent-pair rotation (same shapes, same short-text
    /// output) is caught here rather than by the text gate.
    #[test]
    fn partial_rope_is_neox_paired() {
        let dev = Device::Cpu;
        let (n_rot, hd, theta) = (8usize, 12usize, 1e7f64);
        let x: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32).collect();
        let xt = Tensor::from_vec(x.clone(), (1, 1, hd), &dev).unwrap();
        let pos = 5usize;
        let (cos, sin) = rope_cs_at(n_rot, theta, &[pos], &dev).unwrap();
        let y = partial_rope_seq(
            &xt,
            &cos.reshape((1, 1, 4)).unwrap(),
            &sin.reshape((1, 1, 4)).unwrap(),
            n_rot,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        for i in 0..4 {
            let f = pos as f32 / (theta as f32).powf(2.0 * i as f32 / n_rot as f32);
            let (c, s) = (f.cos(), f.sin());
            let want_lo = x[i] * c - x[i + 4] * s;
            let want_hi = x[i] * s + x[i + 4] * c;
            assert!(
                (y[i] - want_lo).abs() < 1e-4,
                "dim {i}: {} vs {want_lo}",
                y[i]
            );
            assert!(
                (y[i + 4] - want_hi).abs() < 1e-4,
                "dim {}: {} vs {want_hi}",
                i + 4,
                y[i + 4]
            );
        }
        for i in n_rot..hd {
            assert_eq!(y[i], x[i], "tail dim {i} must pass through");
        }
        // The adjacent-pair operator gives a different vector on the same
        // input — the two are not interchangeable.
        let adj0 = x[0] * (pos as f32).cos() - x[1] * (pos as f32).sin();
        assert!(
            (y[0] - adj0).abs() > 1e-3,
            "neox and adjacent pairing coincide on this fixture"
        );
    }

    /// The chunk path equals the per-token path: `prefill` over 9 positions
    /// after 2 decode steps gives the same outputs as 9 more decode steps, and
    /// leaves the cache at the same length. Pins the causal mask, the
    /// per-position rope and the slab append.
    #[test]
    fn prefill_matches_decode_replay() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let attn = QsaAttention::new(
            lin(nh * hd * 2, h, 1.0, &dev),
            lin(nkv * hd, h, 2.0, &dev),
            lin(nkv * hd, h, 3.0, &dev),
            lin(h, nh * hd, 4.0, &dev),
            rms(hd, &dev),
            rms(hd, &dev),
            &cfg,
        );
        let tok = |s: f32| -> Vec<f32> {
            (0..h)
                .map(|i| ((i as f32 * 0.05 + s).sin()) * 0.3)
                .collect()
        };
        // Small slabs so the append grows across the test.
        let (mut kv_a, mut kv_b) = (KvCache::with_capacity(4), KvCache::with_capacity(4));
        for t in 0..2 {
            let x = Tensor::from_vec(tok(t as f32), (1, 1, h), &dev).unwrap();
            attn.decode(&x, &mut kv_a).unwrap();
            attn.decode(&x, &mut kv_b).unwrap();
        }
        let seq = 9;
        let flat: Vec<f32> = (0..seq).flat_map(|t| tok(10.0 + t as f32)).collect();
        let x = Tensor::from_vec(flat, (1, seq, h), &dev).unwrap();
        let out = attn.prefill(&x, &mut kv_a).unwrap();
        assert_eq!(out.dims(), &[1, seq, h]);
        assert_eq!(kv_a.len(), 2 + seq);
        let mut rows = Vec::new();
        for t in 0..seq {
            rows.push(attn.decode(&x.narrow(1, t, 1).unwrap(), &mut kv_b).unwrap());
        }
        assert_eq!(kv_b.len(), 2 + seq);
        let out_ref = Tensor::cat(&rows, 1).unwrap();
        let (a, b) = (
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            out_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let maxdiff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            maxdiff <= 1e-5 * scale,
            "prefill vs decode differ by {maxdiff} at scale {scale}"
        );
        // The mask is live: a query must not see a later key. Break the
        // causality by checking position 0 of the chunk equals a decode that
        // saw only 3 keys — already covered by the equality above; assert the
        // mask itself for a small case.
        let m = causal_mask(3, 5, 2, &dev).unwrap().to_vec2::<u8>().unwrap();
        assert_eq!(
            m,
            vec![
                vec![1, 1, 1, 0, 0],
                vec![1, 1, 1, 1, 0],
                vec![1, 1, 1, 1, 1]
            ]
        );
    }

    /// The Metal chunk path matches the CPU chunk path on the real head
    /// geometry, over a 9-position chunk after 2 decode steps (the mask, the
    /// per-position rope and the slab append all on Metal).
    #[cfg(feature = "metal")]
    #[test]
    fn prefill_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 256;
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let build = |dev: &Device| {
            QsaAttention::new(
                lin(nh * hd * 2, h, 1.0, dev),
                lin(nkv * hd, h, 2.0, dev),
                lin(nkv * hd, h, 3.0, dev),
                lin(h, nh * hd, 4.0, dev),
                rms(hd, dev),
                rms(hd, dev),
                &cfg,
            )
        };
        let (a_cpu, a_gpu) = (build(&cpu), build(&gpu));
        let (mut kv_cpu, mut kv_gpu) = (KvCache::with_capacity(4), KvCache::with_capacity(4));
        let tok = |s: f32| -> Vec<f32> {
            (0..h)
                .map(|i| ((i as f32 * 0.05 + s).sin()) * 0.3)
                .collect()
        };
        for t in 0..2 {
            a_cpu
                .decode(
                    &Tensor::from_vec(tok(t as f32), (1, 1, h), &cpu).unwrap(),
                    &mut kv_cpu,
                )
                .unwrap();
            a_gpu
                .decode(
                    &Tensor::from_vec(tok(t as f32), (1, 1, h), &gpu).unwrap(),
                    &mut kv_gpu,
                )
                .unwrap();
        }
        let seq = 9;
        let flat: Vec<f32> = (0..seq).flat_map(|t| tok(10.0 + t as f32)).collect();
        let oc = a_cpu
            .prefill(
                &Tensor::from_vec(flat.clone(), (1, seq, h), &cpu).unwrap(),
                &mut kv_cpu,
            )
            .unwrap();
        let og = a_gpu
            .prefill(
                &Tensor::from_vec(flat, (1, seq, h), &gpu).unwrap(),
                &mut kv_gpu,
            )
            .unwrap();
        let va = oc.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = og
            .to_device(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(vb.iter().filter(|x| !x.is_finite()).count(), 0);
        let scale = va.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let md = va
            .iter()
            .zip(&vb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            md <= 1e-3 * scale,
            "Metal/CPU prefill differ by {md} at scale {scale}"
        );
        assert_eq!(kv_gpu.len(), 2 + seq);
    }

    /// A tiny indexer config: budget 8, ratio 4 → 2 blocks selected, dense
    /// range 11 cells.
    fn tiny_indexed() -> Config {
        let mut cfg = Config::tiny();
        cfg.indexer_budget = 8;
        cfg.indexer_compress_ratio = 4;
        cfg.indexer_n_heads = 2;
        cfg.indexer_head_dim = 8;
        cfg
    }

    fn indexed_attn(cfg: &Config, dev: &Device) -> QsaAttention {
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let (ih, id) = (cfg.indexer_n_heads, cfg.indexer_head_dim);
        QsaAttention::new(
            lin(nh * hd * 2, h, 1.0, dev),
            lin(nkv * hd, h, 2.0, dev),
            lin(nkv * hd, h, 3.0, dev),
            lin(h, nh * hd, 4.0, dev),
            rms(hd, dev),
            rms(hd, dev),
            cfg,
        )
        .with_indexer(
            Indexer::new(
                lin(ih * id, h, 5.0, dev),
                lin(id, h, 6.0, dev),
                rms(id, dev),
                rms(id, dev),
                cfg,
            )
            .unwrap(),
        )
    }

    fn tok(h: usize, s: f32) -> Vec<f32> {
        (0..h)
            .map(|i| ((i as f32 * 0.05 + s).sin()) * 0.3)
            .collect()
    }

    /// The selection equals an independent scalar reference of the indexer:
    /// raw keys → per-block mean → RMS(γ=1) → NEOX rope at the block start;
    /// query heads → RMS → rope(pos); `Σ_h relu(q·k)` per block; the top
    /// `budget/ratio` blocks (ties: lower index) expanded to cells plus the
    /// tail. Checked as a set at every position past the dense range, and the
    /// selection is causal (no cell > pos) and bounded (≤ budget + ratio − 1).
    #[test]
    fn selection_matches_scalar_reference() {
        let dev = Device::Cpu;
        let cfg = tiny_indexed();
        let attn = indexed_attn(&cfg, &dev);
        let ix = attn.indexer().unwrap();
        let (h, ih, id) = (cfg.hidden_size, cfg.indexer_n_heads, cfg.indexer_head_dim);
        let (ratio, budget) = (cfg.indexer_compress_ratio, cfg.indexer_budget);
        let n_rot = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let theta = cfg.rope_theta as f32;
        let eps = cfg.norm_eps as f32;
        // Reference pieces, all scalar.
        let proj = |w: &QMatMul, x: &[f32]| -> Vec<f32> {
            let w = match w {
                QMatMul::Tensor(t) => t.to_vec2::<f32>().unwrap(),
                _ => unreachable!(),
            };
            w.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let rms = |v: &[f32]| -> Vec<f32> {
            let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
            let r = 1.0 / (ms + eps).sqrt();
            v.iter().map(|x| x * r).collect()
        };
        let rope = |v: &[f32], pos: usize| -> Vec<f32> {
            let mut y = v.to_vec();
            let half = n_rot / 2;
            for i in 0..half {
                let f = pos as f32 / theta.powf(2.0 * i as f32 / n_rot as f32);
                let (c, s) = (f.cos(), f.sin());
                y[i] = v[i] * c - v[i + half] * s;
                y[i + half] = v[i] * s + v[i + half] * c;
            }
            y
        };
        let (q_w, k_w) = (&ix.q_proj, &ix.k_proj);
        let n_pos = 22;
        let inputs: Vec<Vec<f32>> = (0..n_pos).map(|t| tok(h, t as f32)).collect();
        let raw: Vec<Vec<f32>> = inputs.iter().map(|x| proj(k_w, x)).collect();
        let mut kv = KvCache::with_capacity(4);
        let mut checked = 0;
        for pos in 0..n_pos {
            let x = Tensor::from_vec(inputs[pos].clone(), (1, 1, h), &dev).unwrap();
            let out = attn.decode(&x, &mut kv).unwrap();
            assert!(out
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()));
            let n_kv = pos + 1;
            let sel = attn.selection(&x, pos, &kv).unwrap();
            assert_eq!(
                kv.n_blocks(),
                n_kv / ratio,
                "compressed blocks at pos {pos}"
            );
            if n_kv <= budget + ratio - 1 {
                assert!(sel.is_none(), "pos {pos}: dense range must not select");
                continue;
            }
            let sel = sel.unwrap().to_vec1::<u32>().unwrap();
            // Reference score per visible complete block.
            let visible = n_kv / ratio;
            let qh: Vec<Vec<f32>> = {
                let qa = proj(q_w, &inputs[pos]);
                (0..ih)
                    .map(|hh| rope(&rms(&qa[hh * id..(hh + 1) * id]), pos))
                    .collect()
            };
            let mut scores = Vec::with_capacity(visible);
            for b in 0..visible {
                let mut pooled = vec![0f32; id];
                for j in 0..ratio {
                    for (p, r) in pooled.iter_mut().zip(&raw[b * ratio + j]) {
                        *p += r / ratio as f32;
                    }
                }
                let kc = rope(&rms(&pooled), b * ratio);
                let sc: f32 = qh
                    .iter()
                    .map(|q| q.iter().zip(&kc).map(|(a, b)| a * b).sum::<f32>().max(0.0))
                    .sum();
                scores.push(sc);
            }
            let mut order: Vec<usize> = (0..visible).collect();
            order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            let topk = (budget / ratio).min(visible);
            let mut want: Vec<u32> = order[..topk]
                .iter()
                .flat_map(|&b| (0..ratio).map(move |j| (b * ratio + j) as u32))
                .collect();
            let tail_start = n_kv / ratio * ratio;
            want.extend((tail_start..n_kv).map(|c| c as u32));
            let mut got = sel.clone();
            want.sort_unstable();
            got.sort_unstable();
            // The reference's top-k is exact only when the scores are not tied
            // across the boundary; the fixture is checked for that.
            let boundary_tied =
                topk < visible && (scores[order[topk - 1]] - scores[order[topk]]).abs() < 1e-6;
            assert!(
                !boundary_tied,
                "fixture ties at the top-k boundary at pos {pos}"
            );
            assert_eq!(got, want, "pos {pos}: selected cells vs reference");
            assert!(
                sel.iter().all(|&c| (c as usize) < n_kv),
                "pos {pos}: a future cell was selected"
            );
            assert!(
                sel.len() <= budget + ratio - 1,
                "pos {pos}: {} cells selected",
                sel.len()
            );
            assert!(
                sel.len() >= budget,
                "pos {pos}: only {} cells selected",
                sel.len()
            );
            checked += 1;
        }
        assert!(checked >= 8, "only {checked} sparse positions checked");
    }

    /// Past the dense range the output equals the dense attention over
    /// exactly the selected cells (the gather is the only difference), and
    /// differs from the dense attention over all cells — the selection is
    /// live. Also: the same weights without an indexer give the dense-all
    /// answer, so a missing indexer is loud in the output, not silent.
    #[test]
    fn sparse_output_is_dense_over_selection() {
        let dev = Device::Cpu;
        let cfg = tiny_indexed();
        let attn = indexed_attn(&cfg, &dev);
        let dense = {
            let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
            QsaAttention::new(
                lin(nh * hd * 2, h, 1.0, &dev),
                lin(nkv * hd, h, 2.0, &dev),
                lin(nkv * hd, h, 3.0, &dev),
                lin(h, nh * hd, 4.0, &dev),
                rms(hd, &dev),
                rms(hd, &dev),
                &cfg,
            )
        };
        let h = cfg.hidden_size;
        let (mut kv_s, mut kv_d) = (KvCache::with_capacity(4), KvCache::with_capacity(4));
        let mut differed = false;
        for pos in 0..20 {
            let x = Tensor::from_vec(tok(h, pos as f32), (1, 1, h), &dev).unwrap();
            let os = attn
                .decode(&x, &mut kv_s)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let od = dense
                .decode(&x, &mut kv_d)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let n_kv = pos + 1;
            let md = os
                .iter()
                .zip(&od)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if n_kv <= cfg.indexer_budget + cfg.indexer_compress_ratio - 1 {
                assert!(md < 1e-6, "pos {pos}: dense range must equal dense ({md})");
            } else if md > 1e-4 {
                differed = true;
            }
        }
        assert!(
            differed,
            "the selection never changed the output past the dense range"
        );
    }

    /// The Metal decode with the indexer matches the CPU decode past the
    /// dense range (the compressed cache, the fp32 score, the device
    /// arg_sort, the gather all on Metal).
    #[cfg(feature = "metal")]
    #[test]
    fn sparse_decode_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let cfg = tiny_indexed();
        let (a_cpu, a_gpu) = (indexed_attn(&cfg, &cpu), indexed_attn(&cfg, &gpu));
        let h = cfg.hidden_size;
        let (mut kv_c, mut kv_g) = (KvCache::with_capacity(4), KvCache::with_capacity(4));
        for pos in 0..20 {
            let x = tok(h, pos as f32);
            let oc = a_cpu
                .decode(
                    &Tensor::from_vec(x.clone(), (1, 1, h), &cpu).unwrap(),
                    &mut kv_c,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let og = a_gpu
                .decode(&Tensor::from_vec(x, (1, 1, h), &gpu).unwrap(), &mut kv_g)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                og.iter().filter(|v| !v.is_finite()).count(),
                0,
                "pos {pos}: non-finite on Metal"
            );
            let scale = oc.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = oc
                .iter()
                .zip(&og)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 1e-3 * scale,
                "pos {pos}: Metal/CPU differ by {md} at scale {scale}"
            );
        }
        assert_eq!(kv_g.n_blocks(), 5);
    }

    /// Run-to-run identity of the sparse path at the REAL indexer geometry
    /// (idx 4 × 128, budget 2048, ratio 4, heads 24/2 × 256): a 2048-position
    /// chunk prefill then 620 decode steps (the last 617 past the dense
    /// range), twice from fresh caches; every decode output must be
    /// bit-identical between the two passes. The long-context gate on the
    /// real model read two logits digests for one token digest; this is the
    /// in-process half of locating that.
    #[cfg(feature = "metal")]
    #[test]
    fn sparse_decode_is_repeatable_at_real_geometry() {
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let attn = indexed_attn(&cfg, &gpu);
        let h = cfg.hidden_size;
        let prefill_len = 2048usize;
        let steps = 620usize;
        let flat: Vec<f32> = (0..prefill_len * h)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.4)
            .collect();
        let x_pre = Tensor::from_vec(flat, (1, prefill_len, h), &gpu).unwrap();
        let step_x: Vec<Tensor> = (0..steps)
            .map(|t| Tensor::from_vec(tok(h, 100.0 + t as f32), (1, 1, h), &gpu).unwrap())
            .collect();
        let run = || -> Vec<Vec<f32>> {
            let mut kv = KvCache::with_capacity(1024);
            attn.prefill(&x_pre, &mut kv).unwrap();
            let mut outs = Vec::with_capacity(steps);
            for x in &step_x {
                let o = attn.decode(x, &mut kv).unwrap();
                outs.push(
                    o.to_device(&Device::Cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
            }
            assert_eq!(kv.len(), prefill_len + steps);
            outs
        };
        let a = run();
        let b = run();
        for (t, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x.iter().all(|v| v.is_finite()), "step {t}: non-finite");
            let n_diff = x
                .iter()
                .zip(y)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            assert_eq!(
                n_diff,
                0,
                "step {t} (pos {}): {n_diff} of {} values differ between two identical passes",
                prefill_len + t,
                x.len()
            );
        }
    }

    /// The Metal decode matches the CPU decode over three positions on the
    /// real head geometry (24/2 heads, head_dim 256, 64 rotated dims): same
    /// weights on both devices, per-step outputs finite and equal within f32
    /// reduction noise. Position 0 is rope-blind (sin = 0), so the later
    /// positions are the ones that see the KV growth, the rope and the
    /// softmax over more than one key.
    #[cfg(feature = "metal")]
    #[test]
    fn qsa_metal_matches_cpu_over_three_tokens() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 256; // small in-proj, real head geometry
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let build = |dev: &Device| {
            QsaAttention::new(
                lin(nh * hd * 2, h, 1.0, dev),
                lin(nkv * hd, h, 2.0, dev),
                lin(nkv * hd, h, 3.0, dev),
                lin(h, nh * hd, 4.0, dev),
                rms(hd, dev),
                rms(hd, dev),
                &cfg,
            )
        };
        let (a_cpu, a_gpu) = (build(&cpu), build(&gpu));
        let (mut kv_cpu, mut kv_gpu) = (KvCache::default(), KvCache::default());
        for step in 0..3 {
            let x: Vec<f32> = (0..h)
                .map(|i| ((i as f32 * 0.05 + step as f32).sin()) * 0.3)
                .collect();
            let xc = Tensor::from_vec(x.clone(), (1, 1, h), &cpu).unwrap();
            let xg = Tensor::from_vec(x, (1, 1, h), &gpu).unwrap();
            let oc = a_cpu
                .decode(&xc, &mut kv_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let og = a_gpu
                .decode(&xg, &mut kv_gpu)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let nonfinite = og.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                nonfinite, 0,
                "step {step}: {nonfinite} non-finite values on Metal"
            );
            let maxdiff = oc
                .iter()
                .zip(&og)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = oc.iter().map(|a| a.abs()).fold(0f32, f32::max);
            assert!(
                maxdiff <= 1e-3 * scale.max(1e-3),
                "step {step}: Metal/CPU max diff {maxdiff} at scale {scale}"
            );
        }
    }
}
