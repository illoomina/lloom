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

use super::config::{Config, DEFAULT_KV_PRECISION};
use crate::nn::kv_quant::QuantKvSlab;
use crate::nn::ops::KvPrecision;
use crate::nn::{ops, RmsNorm};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Module, Result, Tensor, D};

/// Per-layer growing KV state for one sequence (full-attn layers only). Unlike
/// DeltaNet's fixed `S`, KV grows with context. The rows live in a
/// preallocated slab written in place (`nn::kv_cache`, grown by a slab when
/// full), so a decode step appends without copying the cache — no O(context)
/// per token (§3.3). The paged/pooled KV is the perf path, deferred with caching.
///
/// The k / v store is `precision` (`Config::kv_precision`, §10.7): the dense
/// slab at f32 or f16 (appended at the store's dtype, read back up to the
/// activation dtype), or the quantized slab (codes + a (scale, bias) pair per
/// head vector), attended straight from its codes. Every read a layer makes
/// is bounded by the indexer's budget, dense range or selection alike.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// The k / v tokens: this row's own slab (dense or quantized), or a slot
    /// of a batch's page pool (`KvPool`, §12.2, dense f32 only).
    kv: KvBacking,
    precision: KvPrecision,
    /// The indexer's raw keys `[1, seq, idx_dim]`, one per token.
    raw_keys: crate::nn::kv_cache::Cache,
    /// The indexer's compressed keys `[1, n_blocks, idx_dim]`, one per
    /// COMPLETE block, normed and roped — the persistent incremental cache.
    blocks: crate::nn::kv_cache::Cache,
}

/// Where a [`KvCache`]'s k / v live.
#[derive(Debug)]
enum KvBacking {
    /// `[n_kv_heads, seq, head_dim]` for k and v, the sequence on dim 1, in a
    /// preallocated slab written in place (grown by a slab when full).
    Slab(crate::nn::kv_cache::KvCache),
    /// The quantized arm: codes and (scale, bias) planes, grown likewise.
    Quant(QuantKvSlab),
    /// Slot `slot` of one layer of a [`KvPool`]: the tokens in the pool's
    /// pages, the row's table on the host (and on the device, for a gather).
    Pooled {
        pool: KvPool,
        slot: usize,
        layer: usize,
        /// The row's page table as last uploaded, and the page count it was
        /// uploaded at (`None` until a gather needs it).
        table_t: Option<(Tensor, usize)>,
        /// The slot's lease, shared by the row's layers; the slot returns to
        /// the pool when the last of them drops.
        lease: std::sync::Arc<SlotLease>,
    },
}

/// A row's lease on a [`KvPool`] slot.
#[derive(Debug)]
pub struct SlotLease {
    pool: KvPool,
    slot: usize,
}

impl SlotLease {
    pub fn slot(&self) -> usize {
        self.slot
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        self.pool.release(self.slot);
    }
}

impl Clone for KvBacking {
    /// A pooled row clones into its OWN SLAB (the same tokens, unpooled): a
    /// clone is a probe's or a snapshot's, never a batch row's.
    fn clone(&self) -> Self {
        match self {
            KvBacking::Slab(s) => KvBacking::Slab(s.clone()),
            KvBacking::Quant(s) => KvBacking::Quant(s.clone()),
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => {
                let mut slab = crate::nn::kv_cache::KvCache::new(1, KV_SLAB_TOKENS);
                if let Ok(Some((k, v))) = pool.live(*layer, *slot) {
                    let _ = slab.append(&k, &v);
                }
                KvBacking::Slab(slab)
            }
        }
    }
}

impl Drop for KvBacking {
    fn drop(&mut self) {
        if let KvBacking::Pooled {
            pool, slot, layer, ..
        } = self
        {
            pool.clear(*layer, *slot);
        }
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::with_capacity(KV_SLAB_TOKENS)
    }
}

/// The bytes a tile of a chunk's sparse rows may take in either of its two
/// row-proportional transients (`QsaAttention::sparse_rows`).
pub const SPARSE_TILE_BYTES: usize = 64 << 20;

/// Tokens a KV slab holds before it grows by another slab.
pub const KV_SLAB_TOKENS: usize = 1024;

/// KV tokens per page of a [`KvPool`] (LFM2's `flash_page_size`).
pub const KV_PAGE_TOKENS: usize = 256;

impl KvCache {
    pub fn with_capacity(tokens: usize) -> Self {
        Self::with_precision(tokens, DEFAULT_KV_PRECISION)
            .expect("the dense store needs no quantized slab")
    }

    /// A cache of `tokens` initial capacity storing its k / v at `precision`.
    pub fn with_precision(tokens: usize, precision: KvPrecision) -> Result<Self> {
        let tokens = tokens.max(1);
        let kv = match precision.dense_dtype() {
            Some(_) => KvBacking::Slab(crate::nn::kv_cache::KvCache::new(1, tokens)),
            None => KvBacking::Quant(QuantKvSlab::new(precision.quant(), tokens, tokens)?),
        };
        Ok(Self {
            kv,
            precision,
            raw_keys: crate::nn::kv_cache::Cache::new(1, tokens),
            blocks: crate::nn::kv_cache::Cache::new(1, tokens.div_ceil(4).max(1)),
        })
    }

    /// The k / v store.
    pub fn precision(&self) -> KvPrecision {
        self.precision
    }
    /// A cache whose k / v are slot `slot` of layer `layer` of `pool`
    /// (`Cache::in_pools`); the indexer's caches stay the row's own.
    pub fn in_pool(pool: &KvPool, lease: &std::sync::Arc<SlotLease>, layer: usize) -> Self {
        Self {
            kv: KvBacking::Pooled {
                pool: pool.clone(),
                slot: lease.slot,
                layer,
                table_t: None,
                lease: lease.clone(),
            },
            precision: DEFAULT_KV_PRECISION,
            raw_keys: crate::nn::kv_cache::Cache::new(1, KV_SLAB_TOKENS),
            blocks: crate::nn::kv_cache::Cache::new(1, KV_SLAB_TOKENS / 4),
        }
    }
    /// The pool, slot and layer of a pooled cache.
    pub fn pool_slot(&self) -> Option<(&KvPool, usize, usize)> {
        match &self.kv {
            KvBacking::Pooled {
                pool, lease, layer, ..
            } => Some((pool, lease.slot(), *layer)),
            KvBacking::Slab(_) | KvBacking::Quant(_) => None,
        }
    }
    pub fn is_pooled(&self) -> bool {
        matches!(self.kv, KvBacking::Pooled { .. })
    }
    pub fn reset(&mut self) {
        match &mut self.kv {
            KvBacking::Slab(s) => s.reset(),
            KvBacking::Quant(s) => s.reset(),
            KvBacking::Pooled {
                pool,
                slot,
                layer,
                table_t,
                ..
            } => {
                pool.clear(*layer, *slot);
                *table_t = None;
            }
        }
        self.raw_keys.reset();
        self.blocks.reset();
    }
    /// Bytes appending `n` tokens would allocate for this row's k / v
    /// (`nkv` heads of `hd`, at the store's bytes a token —
    /// `KvPrecision::bytes_per_token`, the scale planes counted): a slab
    /// widened by `KV_SLAB_TOKENS` steps (the widening is a `cat`, so the
    /// old slab is live beside the new one at the instant — counted), or the
    /// pool's next page slab when its free list runs out. 0 when nothing
    /// grows. Sized, not performed (`Qwen4Exp::plan_pass`).
    pub fn grow_bytes_for(&self, n: usize, nkv: usize, hd: usize) -> usize {
        let per_token = self.precision.bytes_per_token(nkv, hd, DType::F32);
        match &self.kv {
            KvBacking::Quant(s) => s.grow_bytes_for(n, per_token),
            KvBacking::Slab(s) => {
                let (len, cap) = (s.current_seq_len(), s.k_cache().max_seq_len());
                if len + n <= cap {
                    // A first append allocates the slab itself.
                    return if len == 0 && s.k_cache().all_data().is_none() {
                        cap * per_token
                    } else {
                        0
                    };
                }
                let steps = (len + n - cap).div_ceil(KV_SLAB_TOKENS);
                (cap + steps * KV_SLAB_TOKENS) * per_token + cap * per_token
            }
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => pool.grow_bytes_for(*layer, *slot, n, per_token),
        }
    }

    /// Bytes this row's k / v planes hold on the device now: the slab at
    /// its capacity once allocated, the quantized planes' bytes; a pooled
    /// row's are the pool's (its own tenant, not this row's).
    pub fn bytes_held(&self, nkv: usize, hd: usize) -> usize {
        let per_token = self.precision.bytes_per_token(nkv, hd, DType::F32);
        match &self.kv {
            KvBacking::Slab(s) if s.k_cache().all_data().is_some() => {
                s.k_cache().max_seq_len() * per_token
            }
            KvBacking::Slab(_) => 0,
            KvBacking::Quant(s) => s.bytes(),
            KvBacking::Pooled { .. } => 0,
        }
    }

    pub fn len(&self) -> usize {
        match &self.kv {
            KvBacking::Slab(s) => s.current_seq_len(),
            KvBacking::Quant(s) => s.len(),
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => pool.len(*layer, *slot),
        }
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
        match &mut self.kv {
            KvBacking::Slab(s) => s.truncate(tokens),
            KvBacking::Quant(s) => s.truncate(tokens),
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => pool.truncate(*layer, *slot, tokens),
        }
        self.raw_keys.truncate(tokens);
        self.blocks.truncate(blocks);
    }
    /// The live `(k, v, raw indexer keys, compressed keys)` views, for a
    /// state probe (`None` where nothing was written; a quantized store's
    /// k / v dequantized).
    pub fn state_views(&self) -> Result<[Option<Tensor>; 4]> {
        let (k, v) = match &self.kv {
            KvBacking::Slab(s) => (s.k()?, s.v()?),
            KvBacking::Quant(s) if s.is_empty() => (None, None),
            KvBacking::Quant(s) => {
                let (k, v) = s.dequantize()?;
                (Some(k), Some(v))
            }
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => match pool.live(*layer, *slot)? {
                Some((k, v)) => (Some(k), Some(v)),
                None => (None, None),
            },
        };
        Ok([
            k,
            v,
            self.raw_keys.current_data()?,
            self.blocks.current_data()?,
        ])
    }
    /// Append `k`, `v` `[n_kv_heads, n, head_dim]` at the store's precision
    /// (a dense f16 store rounds them to f16; a quantized one encodes them).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        match &mut self.kv {
            KvBacking::Slab(s) => {
                let dt = self.precision.dense_dtype().unwrap_or(k.dtype());
                s.append(&k.to_dtype(dt)?, &v.to_dtype(dt)?)?;
                Ok(())
            }
            KvBacking::Quant(s) => s.append(k, v),
            KvBacking::Pooled { .. } => self.append_pooled(k, v),
        }
    }

    /// The live k and v `[n_kv_heads, len, head_dim]`, or the cells `sel`
    /// of them (`[n_kv_heads, n_sel, head_dim]`): the dense store's views at
    /// its dtype (a pooled row's gathered from its pages), a quantized
    /// store's dequantized — the read a multi-token pass and the composed
    /// path attend through.
    fn dense(&mut self, sel: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        if let Some(sel) = sel {
            return self.gather(sel);
        }
        match &self.kv {
            KvBacking::Slab(s) => match (s.k()?, s.v()?) {
                (Some(k), Some(v)) => Ok((k, v)),
                _ => lloom_tensor::bail!("qsa: a read of an empty cache"),
            },
            KvBacking::Quant(s) => s.dequantize(),
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => match pool.live(*layer, *slot)? {
                Some(kv) => Ok(kv),
                None => lloom_tensor::bail!("qsa: an empty pooled row after an append"),
            },
        }
    }

    /// One decode query's attention: `q [1, nh, 1, hd]` over the live cache,
    /// or over the cells `sel` past the indexer's dense range, →
    /// `[1, nh, 1, hd]`. The quantized store reads its codes (the dense or
    /// the selected kernel); the dense store runs `ops::sdpa` over its views
    /// (the selection gathered first), a 16-bit store's read up to `q`'s
    /// dtype — at most `dense_limit` rows either way.
    fn attend(&mut self, q: &Tensor, sel: Option<&Tensor>, scale: f32) -> Result<Tensor> {
        if let KvBacking::Quant(s) = &self.kv {
            return s.attend(q, sel, scale);
        }
        let (k, v) = self.dense(sel)?;
        let (k, v) = (k.to_dtype(q.dtype())?, v.to_dtype(q.dtype())?);
        ops::sdpa(
            q,
            &k.unsqueeze(0)?,
            &v.unsqueeze(0)?,
            None,
            false,
            scale,
            1.0,
        )
    }
    /// Append one token to a pooled row without writing it: the pool token
    /// slot the rows arm scatters it to.
    fn append_slot(&mut self) -> Result<u32> {
        match &mut self.kv {
            KvBacking::Slab(_) | KvBacking::Quant(_) => {
                lloom_tensor::bail!("qsa: append_slot on a slab cache")
            }
            KvBacking::Pooled {
                pool,
                slot,
                layer,
                table_t,
                ..
            } => {
                let s = pool.append(*layer, *slot, 1)?[0];
                if let Some((_, pages)) = table_t {
                    if *pages != pool.pages(*layer, *slot).len() {
                        *table_t = None;
                    }
                }
                Ok(s)
            }
        }
    }
    /// Append `k`, `v` `[n_kv_heads, n, head_dim]` to a pooled row (the
    /// pool's pages, one scatter), returning nothing.
    fn append_pooled(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        match &mut self.kv {
            KvBacking::Slab(_) | KvBacking::Quant(_) => {
                lloom_tensor::bail!("qsa: append_pooled on a slab cache")
            }
            KvBacking::Pooled {
                pool,
                slot,
                layer,
                table_t,
                ..
            } => {
                let n = k.dim(1)?;
                let slots = pool.append(*layer, *slot, n)?;
                pool.write(*layer, k, v, &slots)?;
                if let Some((_, pages)) = table_t {
                    if *pages != pool.pages(*layer, *slot).len() {
                        *table_t = None;
                    }
                }
                Ok(())
            }
        }
    }
    /// The f32 k / v `[n_kv_heads, tokens, head_dim]` (contiguous) a chunk's
    /// selected cells are read from, and how a cell maps into them: the
    /// dense f32 slab itself (a cell is its row), a pooled row's layer pool
    /// (a cell through the row's page table), or — a 16-bit or quantized
    /// store — its live rows read up to f32 once for the chunk.
    fn cell_store(&mut self) -> Result<CellStore> {
        let f32 = DType::F32;
        if let KvBacking::Slab(s) = &self.kv {
            if let (Some(k), Some(v)) = (s.k_cache().all_data(), s.v_cache().all_data()) {
                if k.dtype() == f32 && v.dtype() == f32 {
                    return Ok(CellStore {
                        k: k.clone(),
                        v: v.clone(),
                        pages: None,
                    });
                }
            }
        }
        if let KvBacking::Pooled {
            pool,
            slot,
            layer,
            table_t,
            ..
        } = &mut self.kv
        {
            let pages = pool.pages(*layer, *slot);
            let pt = match table_t {
                Some((t, n)) if *n == pages.len() => t.clone(),
                _ => {
                    let dev = pool.device();
                    let t = Tensor::from_vec(pages.clone(), pages.len(), &dev)?;
                    *table_t = Some((t.clone(), pages.len()));
                    t
                }
            };
            return match pool.tensors(*layer)? {
                Some((k, v)) => Ok(CellStore {
                    k,
                    v,
                    pages: Some((pool.clone(), pt)),
                }),
                None => lloom_tensor::bail!("qsa cell_store on an empty pooled row"),
            };
        }
        let (k, v) = self.dense(None)?;
        Ok(CellStore {
            k: k.to_dtype(f32)?.contiguous()?,
            v: v.to_dtype(f32)?.contiguous()?,
            pages: None,
        })
    }

    /// The rows `sel [n_sel]` (u32, all `< len`) of k and v:
    /// `[n_kv_heads, n_sel, head_dim]` each. Gathers from the slab itself
    /// (contiguous), so the cost is the selection's size, not the context's;
    /// a pooled row maps the cells through its page table first. A quantized
    /// store dequantizes and gathers — the composed reference's read; its
    /// decode reads the selection from the codes ([`Self::attend`]).
    fn gather(&mut self, sel: &Tensor) -> Result<(Tensor, Tensor)> {
        match &mut self.kv {
            KvBacking::Quant(s) => {
                let (k, v) = s.dequantize()?;
                Ok((k.index_select(sel, 1)?, v.index_select(sel, 1)?))
            }
            KvBacking::Slab(s) => {
                let k = match s.k_cache().all_data() {
                    Some(k) => k.index_select(sel, 1)?,
                    None => lloom_tensor::bail!("qsa gather on an empty cache"),
                };
                let v = match s.v_cache().all_data() {
                    Some(v) => v.index_select(sel, 1)?,
                    None => lloom_tensor::bail!("qsa gather on an empty cache"),
                };
                Ok((k, v))
            }
            KvBacking::Pooled {
                pool,
                slot,
                layer,
                table_t,
                ..
            } => {
                let pages = pool.pages(*layer, *slot);
                let dev = sel.device();
                let table = match table_t {
                    Some((t, n)) if *n == pages.len() => t.clone(),
                    _ => {
                        let t = Tensor::from_vec(pages.clone(), pages.len(), dev)?;
                        *table_t = Some((t.clone(), pages.len()));
                        t
                    }
                };
                let slots = pool.slots_of(&table, sel)?;
                let (pk, pv) = match pool.tensors(*layer)? {
                    Some(t) => t,
                    None => lloom_tensor::bail!("qsa gather on an empty pooled row"),
                };
                Ok((pk.index_select(&slots, 1)?, pv.index_select(&slots, 1)?))
            }
        }
    }
}

/// Where a chunk's selected cells are read ([`KvCache::cell_store`]).
struct CellStore {
    k: Tensor,
    v: Tensor,
    /// A pooled row: its pool and page table, through which a cell maps to
    /// its pool slot.
    pages: Option<(KvPool, Tensor)>,
}

impl CellStore {
    /// `table`'s cells as rows of `k` / `v`.
    fn rows_of(&self, table: &Tensor) -> Result<Tensor> {
        match &self.pages {
            None => Ok(table.clone()),
            Some((pool, pt)) => pool
                .slots_of(pt, &table.flatten_all()?)?
                .reshape(table.shape()),
        }
    }
}

/// The rows' paged KV: one `[nkv, pages · page_size, hd]` (K, V) pair per
/// layer, the pages of a slot the SAME for every layer — a slot's page
/// table lives once, in the engine's allocator
/// (`lloom_tensor::kv_pages::KvPagePool`: the free list, the refcounts, the
/// per-slot tables, `append` / `rewind` / `release`), with a per-layer live
/// length because the layers of one pass append in turn. Released pages
/// wait in the engine's `PageQuarantine` until the GPU has passed the epoch
/// that could still read them, then return to the free list (the discipline
/// LFM2's `BatchPool` has; the pool's own bookkeeping recycled at once —
/// #749 P29 (8), the merge).
#[derive(Debug, Clone)]
pub struct KvPool {
    inner: std::sync::Arc<parking_lot::Mutex<KvPoolInner>>,
}

#[derive(Debug)]
struct KvPoolInner {
    slots: usize,
    free_slots: Vec<usize>,
    page_size: usize,
    grow_pages: usize,
    nkv: usize,
    hd: usize,
    dtype: DType,
    device: lloom_tensor::Device,
    /// A page table per slot, shared by the layers, with its liveness
    /// (`nn::page_pool::PagePool`): a cleared slot's pages wait for the GPU's
    /// epoch that could read them.
    pages: crate::nn::page_pool::PagePool,
    /// Pages every layer's storage holds (the layers grow together).
    storage_pages: usize,
    layers: Vec<LayerStore>,
}

#[derive(Debug, Default)]
struct LayerStore {
    kv: Option<(Tensor, Tensor)>,
    /// The layer's live length per slot.
    lens: Vec<usize>,
}

impl KvPool {
    pub fn new(
        slots: usize,
        n_layers: usize,
        nkv: usize,
        hd: usize,
        dtype: DType,
        device: &lloom_tensor::Device,
    ) -> Self {
        let page_size = KV_PAGE_TOKENS;
        let grow_pages = (slots.max(1) * KV_SLAB_TOKENS).div_ceil(page_size).max(1);
        Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(KvPoolInner {
                slots,
                free_slots: (0..slots).rev().collect(),
                page_size,
                grow_pages,
                nkv,
                hd,
                dtype,
                device: device.clone(),
                pages: crate::nn::page_pool::PagePool::new(
                    page_size,
                    0,
                    slots,
                    u32::MAX as usize,
                    device,
                ),
                storage_pages: 0,
                layers: (0..n_layers)
                    .map(|_| LayerStore {
                        kv: None,
                        lens: vec![0; slots],
                    })
                    .collect(),
            })),
        }
    }

    pub fn device(&self) -> lloom_tensor::Device {
        self.inner.lock().device.clone()
    }

    pub fn same(&self, other: &KvPool) -> bool {
        std::sync::Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Lease a free slot; dropped, the lease returns it.
    pub fn lease(&self) -> Result<std::sync::Arc<SlotLease>> {
        let slot = self
            .inner
            .lock()
            .free_slots
            .pop()
            .ok_or_else(|| lloom_tensor::Error::msg("KvPool: no free slot"))?;
        Ok(std::sync::Arc::new(SlotLease {
            pool: self.clone(),
            slot,
        }))
    }

    fn release(&self, slot: usize) {
        let mut g = self.inner.lock();
        // Whatever the slot still holds goes back through the quarantine.
        for l in 0..g.layers.len() {
            g.layers[l].lens[slot] = 0;
        }
        g.release_pages(slot);
        g.free_slots.push(slot);
    }

    pub fn slots(&self) -> usize {
        self.inner.lock().slots
    }

    pub fn page_size(&self) -> usize {
        self.inner.lock().page_size
    }

    /// Tokens the layer's storage holds (every layer's is the same).
    pub fn tokens(&self, layer: usize) -> usize {
        let g = self.inner.lock();
        let _ = layer;
        g.storage_pages * g.page_size
    }

    pub fn len(&self, layer: usize, slot: usize) -> usize {
        self.inner.lock().layers[layer].lens[slot]
    }

    /// The slot's page table (the same for every layer).
    pub fn pages(&self, layer: usize, slot: usize) -> Vec<u32> {
        let _ = layer;
        self.inner
            .lock()
            .pages
            .seq(slot)
            .map(|s| lloom_tensor::kv_pages::PageIdx::as_u32s(s.pages()).to_vec())
            .unwrap_or_default()
    }

    pub fn tensors(&self, layer: usize) -> Result<Option<(Tensor, Tensor)>> {
        Ok(self.inner.lock().layers[layer].kv.clone())
    }

    /// Bytes a `grow` for `n` more tokens of `slot` would allocate across
    /// the layers (0 when the free pages cover it), for admission.
    pub fn grow_bytes_for(&self, layer: usize, slot: usize, n: usize, per_token: usize) -> usize {
        let g = self.inner.lock();
        let ps = g.page_size;
        let len = g.layers[layer].lens[slot];
        let held = g.pages.seq(slot).map(|s| s.pages().len()).unwrap_or(0);
        let need_pages = (len + n).div_ceil(ps).saturating_sub(held);
        let short = need_pages.saturating_sub(g.pages.pages_free());
        if short == 0 {
            return 0;
        }
        let grows = short.div_ceil(g.grow_pages);
        (grows * g.grow_pages * ps + g.storage_pages * ps) * per_token
    }

    /// Append `n` tokens to `slot` in `layer`: the cells they land in
    /// (`page · page_size + offset`). The slot's table grows when the
    /// layer's length runs past it (the first layer of a pass grows it,
    /// the others reuse its pages).
    pub fn append(&self, layer: usize, slot: usize, n: usize) -> Result<Vec<u32>> {
        let mut g = self.inner.lock();
        let ps = g.page_size;
        let start = g.layers[layer].lens[slot];
        let need = start + n;
        let seq_len = g.pages.seq(slot).map(|s| s.len()).unwrap_or(0);
        if need > seq_len {
            g.recycle_quarantined();
            let held = g.pages.seq(slot).map(|s| s.pages().len()).unwrap_or(0);
            let need_pages = need.div_ceil(ps).saturating_sub(held);
            while g.pages.pages_free() < need_pages {
                g.grow_all()?;
            }
            g.pages
                .append(slot, need - seq_len)
                .map_err(lloom_tensor::Error::wrap)?;
        }
        let out: Vec<u32> = {
            let pages = g.pages.seq(slot).expect("a leased slot").pages();
            (start..need)
                .map(|t| pages[t / ps].0 * ps as u32 + (t % ps) as u32)
                .collect()
        };
        g.layers[layer].lens[slot] = need;
        Ok(out)
    }

    pub fn write(&self, layer: usize, k: &Tensor, v: &Tensor, slots: &[u32]) -> Result<()> {
        let (nkv, n, hd) = k.dims3()?;
        if n != slots.len() || v.dims3()? != (nkv, n, hd) {
            lloom_tensor::bail!(
                "KvPool: k {:?} / v {:?} for {} slots",
                k.shape(),
                v.shape(),
                slots.len()
            )
        }
        self.scatter(
            layer,
            &k.reshape((1, nkv, n, hd))?.contiguous()?,
            &v.reshape((1, nkv, n, hd))?.contiguous()?,
            slots,
        )
    }

    pub fn write_rows(&self, layer: usize, k: &Tensor, v: &Tensor, slots: &[u32]) -> Result<()> {
        let (n, nkv, hd) = k.dims3()?;
        if n != slots.len() || v.dims3()? != (n, nkv, hd) {
            lloom_tensor::bail!(
                "KvPool: k {:?} / v {:?} for {} slots",
                k.shape(),
                v.shape(),
                slots.len()
            )
        }
        self.scatter(
            layer,
            &k.reshape((n, nkv, 1, hd))?.contiguous()?,
            &v.reshape((n, nkv, 1, hd))?.contiguous()?,
            slots,
        )
    }

    fn scatter(&self, layer: usize, k4: &Tensor, v4: &Tensor, slots: &[u32]) -> Result<()> {
        let (pk, pv, tokens, hd, nkv) = {
            let g = self.inner.lock();
            match &g.layers[layer].kv {
                Some((pk, pv)) => (
                    pk.clone(),
                    pv.clone(),
                    g.storage_pages * g.page_size,
                    g.hd,
                    g.nkv,
                ),
                None => lloom_tensor::bail!("KvPool: writing layer {layer} before its pages exist"),
            }
        };
        if k4.dim(1)? != nkv || k4.dim(3)? != hd {
            lloom_tensor::bail!("KvPool: {:?} against a [{nkv}, _, {hd}] pool", k4.shape())
        }
        let max_slot = slots.iter().copied().max().unwrap_or(0) as usize;
        let slots_t = Tensor::from_vec(slots.to_vec(), slots.len(), k4.device())?;
        let pk4 = pk.reshape((1, nkv, tokens, hd))?;
        let pv4 = pv.reshape((1, nkv, tokens, hd))?;
        k4.kv_scatter_into(&pk4, &slots_t, max_slot, 0, tokens * hd, hd)?;
        v4.kv_scatter_into(&pv4, &slots_t, max_slot, 0, tokens * hd, hd)?;
        Ok(())
    }

    /// The slot's live K and V in `layer`, gathered contiguous
    /// (`[nkv, len, hd]` each); `None` while empty.
    pub fn live(&self, layer: usize, slot: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (pk, pv, ids) = {
            let g = self.inner.lock();
            let len = g.layers[layer].lens[slot];
            if len == 0 {
                return Ok(None);
            }
            let (pk, pv) = match &g.layers[layer].kv {
                Some(t) => t.clone(),
                None => return Ok(None),
            };
            let ps = g.page_size;
            let pages = g.pages.seq(slot).expect("a leased slot").pages();
            let ids: Vec<u32> = (0..len)
                .map(|t| pages[t / ps].0 * ps as u32 + (t % ps) as u32)
                .collect();
            (pk, pv, ids)
        };
        let n = ids.len();
        let ids = Tensor::from_vec(ids, n, pk.device())?;
        Ok(Some((pk.index_select(&ids, 1)?, pv.index_select(&ids, 1)?)))
    }

    /// Cells for `sel` (token positions of a slot) through its table.
    pub fn slots_of(&self, table: &Tensor, sel: &Tensor) -> Result<Tensor> {
        let ps = self.page_size() as f64;
        let cell = sel.to_dtype(DType::F32)?;
        let page = (&cell / ps)?.floor()?;
        let pg = table
            .index_select(&page.to_dtype(DType::U32)?, 0)?
            .to_dtype(DType::F32)?;
        let off = (cell - (&page * ps)?)?;
        ((pg * ps)? + off)?.to_dtype(DType::U32)
    }

    /// Keep the first `tokens` of `slot` in `layer`; the slot's table is
    /// rewound to the longest layer (its pages stay attached).
    pub fn truncate(&self, layer: usize, slot: usize, tokens: usize) {
        let mut g = self.inner.lock();
        if tokens < g.layers[layer].lens[slot] {
            g.layers[layer].lens[slot] = tokens;
        }
        g.rewind_table(slot);
    }

    /// Empty `slot` in `layer`; once every layer has, the slot's pages are
    /// released into the quarantine.
    pub fn clear(&self, layer: usize, slot: usize) {
        let mut g = self.inner.lock();
        if layer >= g.layers.len() || slot >= g.slots {
            return;
        }
        g.layers[layer].lens[slot] = 0;
        if g.layers.iter().all(|l| l.lens[slot] == 0) {
            g.release_pages(slot);
        } else {
            g.rewind_table(slot);
        }
    }
}

impl KvPoolInner {
    fn recycle_quarantined(&mut self) {
        self.pages
            .recycle_ready()
            .expect("quarantined pages are unallocated");
    }

    fn release_pages(&mut self, slot: usize) {
        self.pages.release(slot);
    }

    fn rewind_table(&mut self, slot: usize) {
        let longest = self.layers.iter().map(|l| l.lens[slot]).max().unwrap_or(0);
        let seq_len = self.pages.seq(slot).map(|s| s.len()).unwrap_or(0);
        if longest < seq_len {
            self.pages
                .rewind(slot, seq_len - longest)
                .expect("rewinding within the table");
        }
    }

    /// Grow every layer's storage by `grow_pages` pages, and the allocator.
    fn grow_all(&mut self) -> Result<()> {
        let (ps, gp, nkv, hd) = (self.page_size, self.grow_pages, self.nkv, self.hd);
        let (dtype, dev) = (self.dtype, self.device.clone());
        for lp in &mut self.layers {
            // Two allocations: a `clone` of a tensor shares its storage.
            let fresh_k = Tensor::zeros((nkv, gp * ps, hd), dtype, &dev)?;
            let fresh_v = Tensor::zeros((nkv, gp * ps, hd), dtype, &dev)?;
            lp.kv = Some(match lp.kv.take() {
                None => (fresh_k, fresh_v),
                Some((k, v)) => (
                    Tensor::cat(&[&k, &fresh_k], 1)?,
                    Tensor::cat(&[&v, &fresh_v], 1)?,
                ),
            });
        }
        self.storage_pages += gp;
        self.pages.grow(gp);
        Ok(())
    }
}

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

    /// Tokens per compressed block: a pass whose last token completes one
    /// runs the block's pooling, norm, rope and append.
    pub fn ratio(&self) -> usize {
        self.ratio
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
        let k_raw = self.k_proj.forward(x2)?;
        self.observe_keys(&k_raw, kv)
    }

    /// [`Self::observe`] from the projected raw keys `k_raw [seq, idx_dim]`
    /// (the rows arm projects every row's at once).
    fn observe_keys(&self, k_raw: &Tensor, kv: &mut KvCache) -> Result<()> {
        let seq = k_raw.dim(0)?;
        let k_raw = k_raw.reshape((1, seq, self.head_dim))?;
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
        let (cos, sin) = match positions[..] {
            [pos] => rope_cs_one(self.n_rot, self.rope_theta, pos, k_raw.device())?,
            _ => rope_cs_at(self.n_rot, self.rope_theta, &positions, k_raw.device())?,
        };
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
        let visible = n_kv / self.ratio;
        let score = self.score_one(x1, pos, kv)?;
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

    /// The fp32 score `Σ_h relu(q_h · K_C[b])` of every block complete at
    /// or before `pos` for the query `x1 [1, hidden]` at `pos` → `[visible]`.
    fn score_one(&self, x1: &Tensor, pos: usize, kv: &KvCache) -> Result<Tensor> {
        let dev = x1.device();
        let visible = (pos + 1) / self.ratio; // complete blocks at or before pos
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
        let (cos, sin) = rope_cs_one(self.n_rot, self.rope_theta, pos, dev)?;
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
        Ok(score)
    }

    /// [`Self::select`] for `s` consecutive queries at once — their inputs
    /// `x [s, hidden]`, the first at `pos0`, every one past the dense range —
    /// against a cache that has observed all of them. Row `t` is the
    /// selection `select` makes for the query at `pos0 + t`, cell for cell
    /// and in the same order: its top `budget / ratio` blocks among the
    /// `(pos0 + t + 1) / ratio` complete at or before it (the blocks the
    /// chunk's own earlier tokens completed included, the later ones masked
    /// below every visible score), expanded to cells, then its incomplete
    /// tail. The scores are one `[blocks, s · n_heads]` product.
    fn select_rows(&self, x: &Tensor, pos0: usize, kv: &KvCache) -> Result<RowSelection> {
        let s = x.dim(0)?;
        let dev = x.device();
        let (r, topk) = (self.ratio, self.block_topk());
        let n_blocks = (pos0 + s) / r;
        if pos0 < self.dense_limit() || s == 0 || kv.n_blocks() < n_blocks {
            lloom_tensor::bail!(
                "indexer rows: {s} queries from pos {pos0} (dense limit {}), {} of {n_blocks} blocks",
                self.dense_limit(),
                kv.n_blocks()
            )
        }
        let k_c = match kv.blocks.current_data()? {
            Some(b) => b
                .narrow(1, 0, n_blocks)?
                .reshape((n_blocks, self.head_dim))?,
            None => lloom_tensor::bail!("indexer rows: no compressed keys at pos {pos0}"),
        };
        let q = self
            .q_proj
            .forward(x)?
            .reshape((s, self.n_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let positions: Vec<usize> = (pos0..pos0 + s).collect();
        let (cos, sin) = rope_cs_at(self.n_rot, self.rope_theta, &positions, dev)?;
        let half = self.n_rot / 2;
        let q = partial_rope_seq(
            &q,
            &cos.reshape((s, 1, half))?,
            &sin.reshape((s, 1, half))?,
            self.n_rot,
        )?; // [s, n_heads, head_dim]
        let k32 = k_c.to_dtype(DType::F32)?.contiguous()?;
        let q32 = q
            .to_dtype(DType::F32)?
            .reshape((s * self.n_heads, self.head_dim))?
            .t()?
            .contiguous()?;
        let score = k32
            .matmul(&q32)?
            .relu()?
            .reshape((n_blocks, s, self.n_heads))?
            .sum(2)?
            .t()?
            .contiguous()?; // [s, n_blocks], every score >= 0
                            // A block completed after the query scores −1, below every visible
                            // block: a query past the dense range sees more than `topk` blocks.
        let visible: Vec<u32> = (0..s).map(|t| ((pos0 + t + 1) / r) as u32).collect();
        let visible = Tensor::from_vec(visible, (s, 1), dev)?;
        let block = Tensor::arange(0u32, n_blocks as u32, dev)?.reshape((1, n_blocks))?;
        let seen = block.broadcast_lt(&visible)?;
        let score = seen.where_cond(&score, &Tensor::full(-1f32, (s, n_blocks), dev)?)?;
        let order = score.neg()?.arg_sort_last_dim(true)?; // [s, n_blocks]
        let blocks = order.narrow(1, 0, topk)?.contiguous()?; // [s, topk]
        let base = (blocks.to_dtype(DType::F32)? * r as f64)?
            .reshape((s, topk, 1))?
            .broadcast_as((s, topk, r))?;
        let offs = Tensor::arange(0f32, r as f32, dev)?
            .reshape((1, 1, r))?
            .broadcast_as((s, topk, r))?;
        let cells = (base + offs)?
            .reshape((s, topk * r))?
            .to_dtype(DType::U32)?;
        // The tails, `[visible·r, pos]` per row, padded to `r − 1` cells.
        let mut tail = vec![0u32; s * (r - 1)];
        let mut n_keys = Vec::with_capacity(s);
        for t in 0..s {
            let n_kv = pos0 + t + 1;
            let start = n_kv / r * r;
            for (j, c) in (start..n_kv).enumerate() {
                tail[t * (r - 1) + j] = c as u32;
            }
            n_keys.push((topk * r + n_kv - start) as u32);
        }
        let table = if r > 1 {
            let tail = Tensor::from_vec(tail, (s, r - 1), dev)?;
            Tensor::cat(&[&cells, &tail], 1)?
        } else {
            cells
        };
        Ok(RowSelection {
            stride: topk * r + r - 1,
            table,
            n_keys,
        })
    }
}

/// The cells `s` queries attend to past the dense range
/// ([`Indexer::select_rows`]): row `t`'s first `n_keys[t]` entries of
/// `table`, the rest padding.
#[derive(Debug)]
pub struct RowSelection {
    /// `u32 [s, stride]`.
    pub table: Tensor,
    pub stride: usize,
    pub n_keys: Vec<u32>,
}

impl RowSelection {
    /// Row `t`'s cells, `u32 [n_keys[t]]`.
    pub fn row(&self, t: usize) -> Result<Tensor> {
        self.table
            .narrow(0, t, 1)?
            .narrow(1, 0, self.n_keys[t] as usize)?
            .reshape(self.n_keys[t] as usize)
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
    /// The three input stems (`wq | wk | wv`) as one f16 matrix when the
    /// loader could stack them (the parts above are row views of it): a
    /// decode step runs one gemv and splits the row.
    stem_cat: Option<QMatMul>,
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
            stem_cat: None,
        }
    }

    /// Attach the stacked input stems (`[n_heads·2·head_dim + 2·n_kv_heads·
    /// head_dim, hidden]` f16, `q | k | v` rows) for the one-gemv decode.
    pub fn with_stem_cat(mut self, cat: Tensor) -> Self {
        self.stem_cat = Some(QMatMul::TensorF16(cat));
        self
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

    /// The cells a chunk's queries `x [1, s, hidden]` at `pos0 ..` select,
    /// every one past the dense range, against a cache that has observed
    /// them ([`Indexer::select_rows`]) — `None` without an indexer.
    /// Test/probe surface for the chunk's selection.
    pub fn selection_rows(
        &self,
        x: &Tensor,
        pos0: usize,
        kv: &KvCache,
    ) -> Result<Option<RowSelection>> {
        match &self.indexer {
            Some(ix) => {
                let (_, s, h) = x.dims3()?;
                ix.select_rows(&x.reshape((s, h))?, pos0, kv).map(Some)
            }
            None => Ok(None),
        }
    }

    /// Precompute per-rotary-pair (cos, sin) for `pos`, each `[n_rot/2]`.
    /// Text-only decode: mRoPE reduces to standard RoPE (all sections share the
    /// text position), so partial RoPE over the first `n_rot` dims is correct.
    fn rope_cs(&self, pos: usize, dev: &lloom_tensor::Device) -> Result<(Tensor, Tensor)> {
        let (c, s) = rope_cs_one(self.n_rot, self.rope_theta, pos, dev)?;
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
        self.decode_rows(x, &mut [kv])
    }

    /// One decode step for `B` independent sequences at once: `x [B, 1,
    /// hidden]`, row `r` against `kvs[r]` (its own KV cache and position), →
    /// `[B, 1, hidden]`. The projections and the out-proj run once over all
    /// rows (`m = B`); the norms, the rope, the append and the attention run
    /// per row against that row's cache through the kernels a row of one
    /// sees, so a row computes exactly what `decode` of that row computes.
    pub fn decode_rows(&self, x: &Tensor, kvs: &mut [&mut KvCache]) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if seq != 1 || b != kvs.len() || b == 0 {
            lloom_tensor::bail!(
                "qsa decode_rows: x {:?} for {} rows (one token per row)",
                x.shape(),
                kvs.len()
            )
        }
        if let Some(out) = self.decode_rows_fused(x, kvs)? {
            return Ok(out);
        }
        let mut outs = Vec::with_capacity(b);
        for (r, kv) in kvs.iter_mut().enumerate() {
            outs.push(self.decode_composed(&x.narrow(0, r, 1)?, kv)?);
        }
        if b == 1 {
            Ok(outs.pop().expect("one row"))
        } else {
            Tensor::cat(&outs, 0)
        }
    }

    /// `T` POSITIONS of one sequence at once (a speculative verify window):
    /// `x [T, 1, hidden]` → `[T, 1, hidden]`, `kv` advanced `T` positions.
    /// The projections and the out-proj run once over the `T` rows; the
    /// norms, the rope, the append, the indexer and the attention run per
    /// position against the growing cache — what `decode` computes `T`
    /// times, launch for launch. Returns the cache's `(tokens, blocks)`
    /// after every position, for a partial accept (`KvCache::truncate`).
    pub fn decode_seq(
        &self,
        x: &Tensor,
        kv: &mut KvCache,
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (t, seq, _) = x.dims3()?;
        if seq != 1 || t == 0 {
            lloom_tensor::bail!("qsa decode_seq: x {:?} (positions on dim 0)", x.shape())
        }
        let mut lens = Vec::with_capacity(t);
        if t > 1 && window_batched() {
            if let Some(out) = self.decode_seq_batched(x, kv, &mut lens)? {
                return Ok((out, lens));
            }
        }
        if let Some(out) = self.decode_seq_fused(x, kv, &mut lens)? {
            return Ok((out, lens));
        }
        let mut outs = Vec::with_capacity(t);
        for p in 0..t {
            outs.push(self.decode_composed(&x.narrow(0, p, 1)?, kv)?);
            lens.push(kv.lens());
        }
        let out = if t == 1 {
            outs.pop().expect("one position")
        } else {
            Tensor::cat(&outs, 0)?
        };
        Ok((out, lens))
    }

    fn decode_seq_fused(
        &self,
        x: &Tensor,
        kv: &mut KvCache,
        lens: &mut Vec<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, sigmoid_mul, QsaPrep};
        let (t, _, hidden) = x.dims3()?;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || !lloom_tensor::shape_req!(sdpa_head_dim, matches!(hd, 32 | 64 | 96 | 128 | 256); hd)
            || nh % nkv != 0
        {
            return Ok(None);
        }
        let f32 = lloom_tensor::DType::F32;
        let xb = x.reshape((t, hidden))?;
        let (nq, nk) = (nh * 2 * hd, nkv * hd);
        let (qg_all, k_all_rows, v_all_rows) = match &self.stem_cat {
            Some(cat) => {
                let all = cat.forward(&xb)?.to_dtype(f32)?.reshape((t, nq + 2 * nk))?;
                (
                    all.narrow(1, 0, nq)?.contiguous()?,
                    all.narrow(1, nq, nk)?.contiguous()?,
                    all.narrow(1, nq + nk, nk)?.contiguous()?,
                )
            }
            None => (
                self.wq.forward(&xb)?.to_dtype(f32)?.reshape((t, nq))?,
                self.wk.forward(&xb)?.to_dtype(f32)?.reshape((t, nk))?,
                self.wv.forward(&xb)?.to_dtype(f32)?.reshape((t, nk))?,
            ),
        };
        let q_gamma = self.q_norm.weight().to_dtype(f32)?;
        let k_gamma = self.k_norm.weight().to_dtype(f32)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut gated_rows = Vec::with_capacity(t);
        for r in 0..t {
            let x1 = xb.narrow(0, r, 1)?;
            let qg = qg_all.narrow(0, r, 1)?.reshape((nh, 2 * hd))?;
            let k = k_all_rows.narrow(0, r, 1)?.reshape((nkv, hd))?;
            let v = v_all_rows.narrow(0, r, 1)?.reshape((nkv, 1, hd))?;
            let pos = kv.len();
            let prep = qg.apply_op2_no_bwd(
                &k,
                &QsaPrep {
                    hd,
                    n_rot: self.n_rot,
                    nh,
                    nkv,
                    pos,
                    theta: self.rope_theta as f32,
                    eps: self.q_norm.eps() as f32,
                    q_gamma: q_gamma.clone(),
                    k_gamma: k_gamma.clone(),
                },
            )?;
            let q = prep.narrow(0, 0, nh)?;
            let gate = prep.narrow(0, nh, nh)?;
            let k_step = prep.narrow(0, 2 * nh, nkv)?.reshape((nkv, 1, hd))?;
            kv.append(&k_step, &v)?;
            let sel = match &self.indexer {
                Some(ix) => {
                    ix.observe(&x1, kv)?;
                    ix.select(&x1, pos, kv)?
                }
                None => None,
            };
            if let Some(sel) = &sel {
                check_sel(sel, pos, kv)?;
            }
            let ctx = kv.attend(&q.reshape((1, nh, 1, hd))?, sel.as_ref(), scale as f32)?;
            let ctx = ctx.reshape((1, nh * hd))?;
            let gate = gate.reshape((1, nh * hd))?;
            let gated = match sigmoid_mul(&gate, &ctx) {
                Some(g) => g?,
                None => (ctx * ops::sigmoid(&gate)?)?,
            };
            gated_rows.push(gated);
            lens.push(kv.lens());
        }
        let gated = if t == 1 {
            gated_rows.pop().expect("one position")
        } else {
            Tensor::cat(&gated_rows, 0)?
        };
        Ok(Some(self.wo.forward(&gated)?.reshape((t, 1, hidden))?))
    }

    /// [`Self::decode_seq_fused`] with the window's positions in one launch
    /// per kernel (`QWEN4EXP_QSA_WINDOW=batched`): `kernels::QsaPrepBatch`
    /// at each row's position, one append of the window's k / v, the
    /// indexer's key projection once with each block completed at the row
    /// that completes it, and the rows within the dense range through one
    /// `kernels::SdpaVectorPaged` over the slab (its whole capacity one
    /// page) — per row the per-row path's arithmetic. A row past the dense
    /// range selects and attends as the per-row path does. Metal, f32, the
    /// dense f32 slab; `None` elsewhere.
    fn decode_seq_batched(
        &self,
        x: &Tensor,
        kv: &mut KvCache,
        lens: &mut Vec<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, sigmoid_mul, QsaPrepBatch};
        let (t, _, hidden) = x.dims3()?;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let f32 = DType::F32;
        let slab_f32 =
            matches!(&kv.kv, KvBacking::Slab(_)) && kv.precision.dense_dtype() == Some(f32);
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != f32
            || !slab_f32
            || !lloom_tensor::shape_req!(sdpa_head_dim, matches!(hd, 32 | 64 | 96 | 128 | 256); hd)
            || nh % nkv != 0
        {
            return Ok(None);
        }
        let dev = x.device();
        let xb = x.reshape((t, hidden))?;
        let (nq, nk) = (nh * 2 * hd, nkv * hd);
        let (qg_all, k_all_rows, v_all_rows) = match &self.stem_cat {
            Some(cat) => {
                let all = cat.forward(&xb)?.to_dtype(f32)?.reshape((t, nq + 2 * nk))?;
                (
                    all.narrow(1, 0, nq)?.contiguous()?,
                    all.narrow(1, nq, nk)?.contiguous()?,
                    all.narrow(1, nq + nk, nk)?.contiguous()?,
                )
            }
            None => (
                self.wq.forward(&xb)?.to_dtype(f32)?.reshape((t, nq))?,
                self.wk.forward(&xb)?.to_dtype(f32)?.reshape((t, nk))?,
                self.wv.forward(&xb)?.to_dtype(f32)?.reshape((t, nk))?,
            ),
        };
        let pos0 = kv.len();
        let pos: Vec<u32> = (pos0..pos0 + t).map(|p| p as u32).collect();
        let prep = qg_all.reshape((t, nh, 2 * hd))?.apply_op3_no_bwd(
            &k_all_rows.reshape((t, nkv, hd))?,
            &Tensor::from_vec(pos, t, dev)?,
            &QsaPrepBatch {
                hd,
                n_rot: self.n_rot,
                nh,
                nkv,
                theta: self.rope_theta as f32,
                eps: self.q_norm.eps() as f32,
                q_gamma: self.q_norm.weight().to_dtype(f32)?,
                k_gamma: self.k_norm.weight().to_dtype(f32)?,
            },
        )?; // flat: q [t, nh, hd] | gate [t, nh, hd] | k [t, nkv, hd]
        let q = prep.narrow(0, 0, t * nh * hd)?.reshape((t, nh, hd))?;
        let gate = prep
            .narrow(0, t * nh * hd, t * nh * hd)?
            .reshape((t, nh * hd))?;
        let k_new = prep
            .narrow(0, 2 * t * nh * hd, t * nkv * hd)?
            .reshape((t, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        let v_new = v_all_rows
            .reshape((t, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?;
        kv.append(&k_new, &v_new)?;
        // The indexer's keys: one projection, then each block completed at
        // the row that completes it, as the per-row path's observe does.
        let ratio = self.indexer.as_ref().map(|ix| ix.ratio);
        if let Some(ix) = &self.indexer {
            let k_raw = ix.k_proj.forward(&xb)?;
            let mut start = 0;
            for r in 0..t {
                if (pos0 + r + 1).is_multiple_of(ix.ratio) || r + 1 == t {
                    ix.observe_keys(&k_raw.narrow(0, start, r + 1 - start)?, kv)?;
                    start = r + 1;
                }
            }
        }
        for r in 0..t {
            let blocks = match ratio {
                Some(ratio) => (pos0 + r + 1) / ratio,
                None => kv.n_blocks(),
            };
            lens.push((pos0 + r + 1, blocks));
        }
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        // The rows within the dense range are a prefix of the window.
        let d = (0..t)
            .take_while(|&r| match &self.indexer {
                Some(ix) => pos0 + r < ix.dense_limit(),
                None => true,
            })
            .count();
        let mut parts = Vec::with_capacity(1 + t - d);
        if d > 0 {
            let KvBacking::Slab(s) = &kv.kv else {
                unreachable!("checked above")
            };
            let (pk, pv) = match (s.k_cache().all_data(), s.v_cache().all_data()) {
                (Some(k), Some(v)) => (k.clone(), v.clone()),
                _ => lloom_tensor::bail!("qsa window: the slab is empty after its append"),
            };
            let cap = pk.dim(1)?;
            let n_keys: Vec<u32> = (0..d).map(|r| (pos0 + r + 1) as u32).collect();
            let ctx = self.sdpa_paged_rows(
                &q.narrow(0, 0, d)?.contiguous()?,
                &pk,
                &pv,
                cap,
                &vec![vec![0u32]; d],
                &n_keys,
                scale,
            )?;
            parts.push(ctx.reshape((d, nh * hd))?);
        }
        for r in d..t {
            let ix = self.indexer.as_ref().expect("a row past the dense range");
            let pos = pos0 + r;
            let sel = ix.select(&xb.narrow(0, r, 1)?, pos, kv)?;
            if let Some(sel) = &sel {
                check_sel(sel, pos, kv)?;
            }
            let qr = q.narrow(0, r, 1)?.reshape((1, nh, 1, hd))?;
            // Past the dense range the row reads its selection, every cell of
            // which lies at or before it.
            parts.push(kv.attend(&qr, sel.as_ref(), scale)?.reshape((1, nh * hd))?);
        }
        let ctx = if parts.len() == 1 {
            parts.pop().expect("one part")
        } else {
            Tensor::cat(&parts, 0)?
        };
        let gated = match sigmoid_mul(&gate, &ctx) {
            Some(g) => g?,
            None => (ctx * ops::sigmoid(&gate)?)?,
        };
        use std::sync::atomic::Ordering::Relaxed;
        WINDOW_BATCHED.fetch_add(1, Relaxed);
        WINDOW_SPARSE_ROWS.fetch_add((t - d) as u64, Relaxed);
        Ok(Some(self.wo.forward(&gated)?.reshape((t, 1, hidden))?))
    }

    /// The composed decode of one row: the CPU path and the reference.
    fn decode_composed(&self, x: &Tensor, kv: &mut KvCache) -> Result<Tensor> {
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
        kv.append(&k_step, &v_step)?;

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
            check_sel(sel, pos, kv)?;
        }
        let (k_all, v_all) = kv.dense(sel.as_ref())?; // [nkv, n_sel | len, hd] each
        let ctx = self
            .attend_composed(&q, &k_all, &v_all)?
            .to_dtype(x.dtype())?;

        // output gate: · sigmoid(gate), then wo.
        let gate_flat = gate.reshape((1, nh * hd))?;
        let gated = (ctx * ops::sigmoid(&gate_flat)?)?;
        self.wo.forward(&gated)?.reshape((1, 1, hidden))
    }

    /// One query's attention as tensor ops, the composed reference: `q
    /// [nh, hd]` (normed, roped) over every row of `k_all`, `v_all`
    /// `[nkv, n, hd]` (all at or before it, so no mask) → `[1, nh·hd]` f32.
    /// GQA expands the kv heads by tiling, as the reference repeats them.
    fn attend_composed(&self, q: &Tensor, k_all: &Tensor, v_all: &Tensor) -> Result<Tensor> {
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let seq = k_all.dim(1)?;
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
        ctx.reshape((1, nh * hd))
    }

    /// The queries of a chunk at `past .. past + seq` that lie within the
    /// dense range — a prefix of the chunk; the rest attend to a selection.
    fn dense_rows(&self, past: usize, seq: usize) -> usize {
        match &self.indexer {
            Some(ix) => seq.min(ix.dense_limit().saturating_sub(past)),
            None => seq,
        }
    }

    /// The chunk's queries past the dense range — inputs `x [s, hidden]`,
    /// `q [s, nh, hd]` (normed, roped), the first at `pos0` — selected
    /// ([`Indexer::select_rows`]) and attended ([`Self::attend_rows_paged`]) in
    /// tiles of rows, so neither the score `[blocks, rows · n_heads]` nor the
    /// two-pass kernel's partials (`rows · nh · 32 · hd`) outgrow
    /// [`SPARSE_TILE_BYTES`] at any chunk or context → `[s, nh·hd]` f32.
    fn sparse_rows(
        &self,
        x: &Tensor,
        q: &Tensor,
        pos0: usize,
        kv: &mut KvCache,
        scale: f32,
    ) -> Result<Tensor> {
        let ix = self.indexer.as_ref().expect("rows past the dense range");
        let s = x.dim(0)?;
        let n_blocks = (pos0 + s) / ix.ratio;
        let score_row = n_blocks.max(1) * ix.n_heads * 4;
        // The two-pass kernel's partials: 32 blocks of `hd` f32 a head.
        let partials_row = self.n_heads * 32 * self.head_dim * 4;
        let tile = (SPARSE_TILE_BYTES / score_row.max(partials_row)).max(1);
        if sparse_masked() && self.paged_rows(q) {
            return self.sparse_rows_masked(ix, x, q, pos0, kv, scale, tile);
        }
        let store = if self.paged_rows(q) {
            Some(kv.cell_store()?)
        } else {
            None
        };
        let mut parts = Vec::with_capacity(s.div_ceil(tile));
        for t0 in (0..s).step_by(tile) {
            let n = tile.min(s - t0);
            let x_t = x.narrow(0, t0, n)?;
            let sel = ix.select_rows(&x_t, pos0 + t0, kv)?;
            check_rows(ix, &x_t, pos0 + t0, &sel, kv)?;
            let q_t = q.narrow(0, t0, n)?;
            parts.push(match &store {
                Some(store) => self.attend_rows_paged(&q_t, &sel, store, scale)?,
                None => self.attend_rows_composed(&q_t, &sel, kv)?,
            });
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one tile"));
        }
        Tensor::cat(&parts, 0)
    }

    /// [`Self::sparse_rows`]'s A/B arm (`QWEN4EXP_SPARSE_MASKED`): each tile
    /// of rows through the steel full-attention kernel over every cell of
    /// the cache, an additive mask `[rows, cells]` (0 on the row's selected
    /// cells, −∞ elsewhere) in place of the per-row cell lists. The same
    /// selection; O(rows · context) attention where the paged arm is
    /// O(rows · budget).
    #[allow(clippy::too_many_arguments)]
    fn sparse_rows_masked(
        &self,
        ix: &Indexer,
        x: &Tensor,
        q: &Tensor,
        pos0: usize,
        kv: &mut KvCache,
        scale: f32,
        tile: usize,
    ) -> Result<Tensor> {
        let (hd, nh) = (self.head_dim, self.n_heads);
        let f32 = DType::F32;
        let s = x.dim(0)?;
        let dev = q.device();
        let (k_all, v_all) = kv.dense(None)?;
        let (k_all, v_all) = (k_all.to_dtype(f32)?, v_all.to_dtype(f32)?);
        let n_kv = k_all.dim(1)?;
        // Rows of a tile are bounded by the mask's bytes as well.
        let tile = tile.min((SPARSE_TILE_BYTES / (n_kv * 4)).max(1));
        let mut parts = Vec::with_capacity(s.div_ceil(tile));
        for t0 in (0..s).step_by(tile) {
            let n = tile.min(s - t0);
            let x_t = x.narrow(0, t0, n)?;
            let sel = ix.select_rows(&x_t, pos0 + t0, kv)?;
            check_rows(ix, &x_t, pos0 + t0, &sel, kv)?;
            // 1 on each row's live entries of the table, 0 on its padding;
            // scattered onto the cells, any count > 0 is a selected cell.
            let live = Tensor::arange(0u32, sel.stride as u32, dev)?
                .reshape((1, sel.stride))?
                .broadcast_lt(&Tensor::from_vec(sel.n_keys.clone(), (n, 1), dev)?)?
                .to_dtype(f32)?;
            let hits = Tensor::zeros((n, n_kv), f32, dev)?.scatter_add(&sel.table, &live, 1)?;
            let mask = hits
                .gt(0f64)?
                .where_cond(
                    &Tensor::zeros((n, n_kv), f32, dev)?,
                    &Tensor::full(f32::NEG_INFINITY, (n, n_kv), dev)?,
                )?
                .reshape((1, 1, n, n_kv))?
                .broadcast_as((1, nh, n, n_kv))?;
            let q_t = q
                .narrow(0, t0, n)?
                .transpose(0, 1)?
                .contiguous()?
                .unsqueeze(0)?; // [1, nh, n, hd]
            let ctx = ops::sdpa(
                &q_t,
                &k_all.unsqueeze(0)?,
                &v_all.unsqueeze(0)?,
                Some(&mask),
                false,
                scale,
                1.0,
            )?; // [1, nh, n, hd]
            parts.push(
                ctx.reshape((nh, n, hd))?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((n, nh * hd))?,
            );
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one tile"));
        }
        Tensor::cat(&parts, 0)
    }

    /// Whether the chunk's queries past the dense range run the paged-vector
    /// kernel ([`Self::attend_rows_paged`]): Metal, f32 queries, a head width
    /// the kernel carries.
    fn paged_rows(&self, q: &Tensor) -> bool {
        q.device().is_metal()
            && q.dtype() == DType::F32
            && !super::kernels::fused_off()
            && matches!(self.head_dim, 32 | 64 | 96 | 128 | 256 | 512)
    }

    /// The composed reference of [`Self::attend_rows_paged`]: row by row,
    /// the row's cells gathered and [`Self::attend_composed`] over them.
    fn attend_rows_composed(
        &self,
        q: &Tensor,
        sel: &RowSelection,
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        let mut rows = Vec::with_capacity(q.dim(0)?);
        for t in 0..q.dim(0)? {
            let (k, v) = kv.dense(Some(&sel.row(t)?))?;
            rows.push(self.attend_composed(&q.narrow(0, t, 1)?.squeeze(0)?, &k, &v)?);
        }
        Tensor::cat(&rows, 0)
    }

    /// The attention of the chunk's queries past the dense range: `q
    /// [s, nh, hd]` (normed, roped, f32), row `t` over the cells
    /// `sel.row(t)` of `store` → `[s, nh·hd]`. `kernels::SdpaVectorPaged`
    /// with a page of one cell and the row's cells for its table, one launch
    /// per kernel arm: per row the arithmetic of the vector `ops::sdpa` a
    /// decode step runs over the same cells gathered in the same order.
    fn attend_rows_paged(
        &self,
        q: &Tensor,
        sel: &RowSelection,
        store: &CellStore,
        scale: f32,
    ) -> Result<Tensor> {
        let (hd, nh) = (self.head_dim, self.n_heads);
        let s = q.dim(0)?;
        let table = store.rows_of(&sel.table)?;
        let dev = q.device();
        let long = |t: usize| ops::sdpa_vector_two_pass(sel.n_keys[t] as usize, hd, q.dtype());
        let groups: [Vec<usize>; 2] = [
            (0..s).filter(|&t| !long(t)).collect(),
            (0..s).filter(|&t| long(t)).collect(),
        ];
        let mut parts: Vec<(Vec<usize>, Tensor)> = Vec::with_capacity(2);
        for (two_pass, grp) in groups.into_iter().enumerate() {
            if grp.is_empty() {
                continue;
            }
            let ng = grp.len();
            let (q_g, table_g) = if ng == s {
                (q.contiguous()?, table.clone())
            } else {
                let idx: Vec<u32> = grp.iter().map(|&t| t as u32).collect();
                let idx = Tensor::from_vec(idx, ng, dev)?;
                (q.index_select(&idx, 0)?, table.index_select(&idx, 0)?)
            };
            let keys: Vec<u32> = grp.iter().map(|&t| sel.n_keys[t]).collect();
            let ctx = q_g.apply_op3_no_bwd(
                &store.k,
                &store.v,
                &super::kernels::SdpaVectorPaged {
                    nh,
                    nkv: self.n_kv_heads,
                    hd,
                    page_size: 1,
                    table_stride: sel.stride,
                    scale,
                    two_pass: two_pass == 1,
                    table: table_g.flatten_all()?.contiguous()?,
                    n_keys: Tensor::from_vec(keys, ng, dev)?,
                },
            )?; // [ng, nh, hd]
            parts.push((grp, ctx.reshape((ng, nh * hd))?));
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("one group").1);
        }
        let mut by_row: Vec<Option<Tensor>> = vec![None; s];
        for (grp, ctx) in &parts {
            for (i, &t) in grp.iter().enumerate() {
                by_row[t] = Some(ctx.narrow(0, i, 1)?);
            }
        }
        let rows: Vec<Tensor> = by_row.into_iter().map(|r| r.expect("every row")).collect();
        Tensor::cat(&rows, 0)
    }

    /// [`Self::decode_rows`] as launches: the projections once over all rows, then per row `kernels::QsaPrep` (the
    /// q/gate split, both norms, the partial rope at `pos` — the angle on the
    /// device, no per-token cos/sin upload), the KV append, `ops::sdpa`'s
    /// vector kernel over the cache as it lies (GQA in the kernel, no head
    /// expansion), `kernels::SigmoidMul` for the output gate, the out-proj.
    /// Metal, f32; `None` where the composed path applies.
    fn decode_rows_fused(&self, x: &Tensor, kvs: &mut [&mut KvCache]) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, sigmoid_mul, QsaPrep};
        let (b, _, hidden) = x.dims3()?;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || !lloom_tensor::shape_req!(sdpa_head_dim, matches!(hd, 32 | 64 | 96 | 128 | 256); hd)
            || nh % nkv != 0
        {
            return Ok(None);
        }
        let f32 = lloom_tensor::DType::F32;
        let xb = x.reshape((b, hidden))?;
        let (nq, nk) = (nh * 2 * hd, nkv * hd);
        // The projections once over all rows (one gemv over the stacked stems
        // when they are stacked): `[B, nq] | [B, nk] | [B, nk]`.
        let (qg_all, k_all_rows, v_all_rows) = match &self.stem_cat {
            Some(cat) => {
                let all = cat.forward(&xb)?.to_dtype(f32)?.reshape((b, nq + 2 * nk))?;
                (
                    all.narrow(1, 0, nq)?.contiguous()?,
                    all.narrow(1, nq, nk)?.contiguous()?,
                    all.narrow(1, nq + nk, nk)?.contiguous()?,
                )
            }
            None => (
                self.wq.forward(&xb)?.to_dtype(f32)?.reshape((b, nq))?,
                self.wk.forward(&xb)?.to_dtype(f32)?.reshape((b, nk))?,
                self.wv.forward(&xb)?.to_dtype(f32)?.reshape((b, nk))?,
            ),
        };
        let q_gamma = self.q_norm.weight().to_dtype(f32)?;
        let k_gamma = self.k_norm.weight().to_dtype(f32)?;
        let scale = 1.0 / (hd as f64).sqrt();
        // The pooled rows within the dense range: one launch per kernel over
        // them (the prologue, the scatter, the paged attention, the gate).
        let mut pooled: Vec<usize> = Vec::new();
        let mut pool: Option<KvPool> = None;
        for (r, kv) in kvs.iter().enumerate() {
            if let Some((p, _, _)) = kv.pool_slot() {
                let dense = match &self.indexer {
                    Some(ix) => kv.len() < ix.dense_limit(),
                    None => true,
                };
                if dense && pool.as_ref().is_none_or(|q| q.same(p)) {
                    if pool.is_none() {
                        pool = Some(p.clone());
                    }
                    pooled.push(r);
                }
            }
        }
        let mut gated_by_row: Vec<Option<Tensor>> = vec![None; b];
        if let Some(pool) = &pool {
            let g = self.decode_pooled_rows(
                pool,
                &pooled,
                kvs,
                &xb,
                &qg_all,
                &k_all_rows,
                &v_all_rows,
                &q_gamma,
                &k_gamma,
                scale as f32,
            )?; // [nd, nh·hd]
            for (j, &r) in pooled.iter().enumerate() {
                gated_by_row[r] = Some(g.narrow(0, j, 1)?);
            }
        }
        for (r, kv) in kvs.iter_mut().enumerate() {
            if gated_by_row[r].is_some() {
                continue;
            }
            let x1 = xb.narrow(0, r, 1)?; // [1, hidden]
            let qg = qg_all.narrow(0, r, 1)?.reshape((nh, 2 * hd))?;
            let k = k_all_rows.narrow(0, r, 1)?.reshape((nkv, hd))?;
            let v = v_all_rows.narrow(0, r, 1)?.reshape((nkv, 1, hd))?;
            let pos = kv.len();
            let prep = qg.apply_op2_no_bwd(
                &k,
                &QsaPrep {
                    hd,
                    n_rot: self.n_rot,
                    nh,
                    nkv,
                    pos,
                    theta: self.rope_theta as f32,
                    eps: self.q_norm.eps() as f32,
                    q_gamma: q_gamma.clone(),
                    k_gamma: k_gamma.clone(),
                },
            )?; // [2nh + nkv, hd]
            let q = prep.narrow(0, 0, nh)?; // [nh, hd]
            let gate = prep.narrow(0, nh, nh)?; // [nh, hd]
            let k_step = prep.narrow(0, 2 * nh, nkv)?.reshape((nkv, 1, hd))?;
            kv.append(&k_step, &v)?;
            let sel = match &self.indexer {
                Some(ix) => {
                    ix.observe(&x1, kv)?;
                    ix.select(&x1, pos, kv)?
                }
                None => None,
            };
            if let Some(sel) = &sel {
                check_sel(sel, pos, kv)?;
            }
            let ctx = kv.attend(&q.reshape((1, nh, 1, hd))?, sel.as_ref(), scale as f32)?; // [1, nh, 1, hd]
            let ctx = ctx.reshape((1, nh * hd))?;
            let gate = gate.reshape((1, nh * hd))?;
            let gated = match sigmoid_mul(&gate, &ctx) {
                Some(g) => g?,
                None => (ctx * ops::sigmoid(&gate)?)?,
            };
            gated_by_row[r] = Some(gated);
        }
        let gated_rows: Vec<Tensor> = gated_by_row
            .into_iter()
            .map(|g| g.expect("every row gated"))
            .collect();
        let gated = if b == 1 {
            gated_rows.into_iter().next().expect("one row")
        } else {
            Tensor::cat(&gated_rows, 0)?
        }; // [B, nh·hd]
        Ok(Some(self.wo.forward(&gated)?.reshape((b, 1, hidden))?))
    }

    /// The pooled rows arm of [`Self::decode_rows_fused`]: the rows `rows`
    /// (each pooled in `pool`, within the dense range) through
    /// `kernels::QsaPrepBatch` (every row at its own position), one scatter
    /// of the step's k / v into the layer's pool, the indexer's observe per
    /// row, `kernels::SdpaVectorPaged` (every row over its own page table,
    /// one launch) and the output gate → `[rows, nh·hd]`. Row for row the
    /// per-row path's arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn decode_pooled_rows(
        &self,
        pool: &KvPool,
        rows: &[usize],
        kvs: &mut [&mut KvCache],
        xb: &Tensor,
        qg_all: &Tensor,
        k_rows: &Tensor,
        v_rows: &Tensor,
        q_gamma: &Tensor,
        k_gamma: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        use super::kernels::{sigmoid_mul, QsaPrepBatch};
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let nd = rows.len();
        let dev = xb.device();
        let all = nd == kvs.len();
        let idx_t = if all {
            None
        } else {
            let idx: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
            Some(Tensor::from_vec(idx, nd, dev)?)
        };
        let pick = |t: &Tensor| -> Result<Tensor> {
            match &idx_t {
                None => Ok(t.clone()),
                Some(i) => t.index_select(i, 0),
            }
        };
        let qg = pick(qg_all)?.reshape((nd, nh, 2 * hd))?;
        let k = pick(k_rows)?.reshape((nd, nkv, hd))?;
        let v = pick(v_rows)?.reshape((nd, nkv, hd))?;
        let pos: Vec<u32> = rows.iter().map(|&r| kvs[r].len() as u32).collect();
        let pos_t = Tensor::from_vec(pos, nd, dev)?;
        let prep = qg.apply_op3_no_bwd(
            &k,
            &pos_t,
            &QsaPrepBatch {
                hd,
                n_rot: self.n_rot,
                nh,
                nkv,
                theta: self.rope_theta as f32,
                eps: self.q_norm.eps() as f32,
                q_gamma: q_gamma.clone(),
                k_gamma: k_gamma.clone(),
            },
        )?; // flat: q [nd, nh, hd] | gate [nd, nh, hd] | k [nd, nkv, hd]
        let q = prep.narrow(0, 0, nd * nh * hd)?.reshape((nd, nh, hd))?;
        let gate = prep
            .narrow(0, nd * nh * hd, nd * nh * hd)?
            .reshape((nd, nh * hd))?;
        let k_step = prep
            .narrow(0, 2 * nd * nh * hd, nd * nkv * hd)?
            .reshape((nd, nkv, hd))?;
        // The step's tokens appended (a slot each), written as one scatter.
        let layer = kvs[rows[0]].pool_slot().expect("pooled").2;
        let mut slots = Vec::with_capacity(nd);
        for &r in rows {
            slots.push(kvs[r].append_slot()?);
        }
        pool.write_rows(layer, &k_step, &v, &slots)?;
        if let Some(ix) = &self.indexer {
            // The indexer's key projection once over the rows; each row
            // keeps its own raw keys and completes its own blocks.
            let k_raw = ix.k_proj.forward(&pick(xb)?)?; // [nd, idx_dim]
            for (j, &r) in rows.iter().enumerate() {
                ix.observe_keys(&k_raw.narrow(0, j, 1)?, kvs[r])?;
            }
        }
        // Every row's table and live count.
        let tables: Vec<Vec<u32>> = rows
            .iter()
            .map(|&r| pool.pages(layer, kvs[r].pool_slot().expect("pooled").1))
            .collect();
        let n_keys: Vec<u32> = rows.iter().map(|&r| kvs[r].len() as u32).collect();
        let (pk, pv) = match pool.tensors(layer)? {
            Some(t) => t,
            None => lloom_tensor::bail!("qsa pooled rows: the layer's pool was not written"),
        };
        let ctx = self.sdpa_paged_rows(&q, &pk, &pv, pool.page_size(), &tables, &n_keys, scale)?;
        let ctx = ctx.reshape((nd, nh * hd))?;
        match sigmoid_mul(&gate, &ctx) {
            Some(g) => g,
            None => ctx * ops::sigmoid(&gate)?,
        }
    }

    /// The paged attention of `q [nd, nh, hd]` over the layer's pool for
    /// `nd` rows (row `j`'s pages `tables[j]`, its live count `n_keys[j]`),
    /// → `[nd, nh, hd]`. The rows split by `ops::sdpa_vector_two_pass` at
    /// their key counts — the single-pass rows in one launch, the 2-pass
    /// rows in one pair — so every row runs the kernel the per-row
    /// `ops::sdpa` runs at its length: bit-identical per row.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_paged_rows(
        &self,
        q: &Tensor,
        pk: &Tensor,
        pv: &Tensor,
        page_size: usize,
        tables: &[Vec<u32>],
        n_keys: &[u32],
        scale: f32,
    ) -> Result<Tensor> {
        use super::kernels::SdpaVectorPaged;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let nd = n_keys.len();
        let dev = q.device();
        let long = |j: usize| ops::sdpa_vector_two_pass(n_keys[j] as usize, hd, q.dtype());
        let groups: [Vec<usize>; 2] = [
            (0..nd).filter(|&j| !long(j)).collect(),
            (0..nd).filter(|&j| long(j)).collect(),
        ];
        let mut by_row: Vec<Option<Tensor>> = vec![None; nd];
        for (two_pass, grp) in groups.iter().enumerate() {
            if grp.is_empty() {
                continue;
            }
            let ng = grp.len();
            let q_g = if ng == nd {
                q.clone()
            } else {
                let idx: Vec<u32> = grp.iter().map(|&j| j as u32).collect();
                q.index_select(&Tensor::from_vec(idx, ng, dev)?, 0)?
            };
            let stride = grp
                .iter()
                .map(|&j| tables[j].len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut table = Vec::with_capacity(ng * stride);
            let mut keys = Vec::with_capacity(ng);
            for &j in grp {
                table.extend_from_slice(&tables[j]);
                table.extend(std::iter::repeat_n(0u32, stride - tables[j].len()));
                keys.push(n_keys[j]);
            }
            let ctx = q_g.apply_op3_no_bwd(
                pk,
                pv,
                &SdpaVectorPaged {
                    nh,
                    nkv,
                    hd,
                    page_size,
                    table_stride: stride,
                    scale,
                    two_pass: two_pass == 1,
                    table: Tensor::from_vec(table, ng * stride, dev)?,
                    n_keys: Tensor::from_vec(keys, ng, dev)?,
                },
            )?; // [ng, nh, hd]
            if ng == nd {
                return Ok(ctx);
            }
            for (i, &j) in grp.iter().enumerate() {
                by_row[j] = Some(ctx.narrow(0, i, 1)?);
            }
        }
        let rows: Vec<Tensor> = by_row.into_iter().map(|t| t.expect("every row")).collect();
        Tensor::cat(&rows, 0)
    }

    /// A whole chunk `x [1, seq, hidden]` → `[1, seq, hidden]`: QSA over every
    /// position at once. The projections, norms, rope, the KV append and
    /// the indexer's observe run over the chunk; the queries within the
    /// dense range attend densely and causally to everything cached, and
    /// each query past it to the cells the indexer selects for it, the
    /// blocks the chunk's earlier tokens completed included
    /// ([`Indexer::select_rows`]) — what [`Self::decode`] computes position
    /// by position, up to the order of fp32 reductions.
    pub fn prefill(&self, x: &Tensor, kv: &mut KvCache) -> Result<Tensor> {
        let (b, seq, hidden) = x.dims3()?;
        if b != 1 {
            lloom_tensor::bail!("qsa prefill takes one sequence, got {:?}", x.shape())
        }
        if let Some(out) = self.prefill_fused(x, kv)? {
            return Ok(out);
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
        kv.append(&k_new, &v_new)?;
        let d = self.dense_rows(past, seq);
        let dense = if d > 0 { Some(kv.dense(None)?) } else { None };
        if let Some(ix) = &self.indexer {
            ix.observe(&x2, kv)?;
        }
        let f32 = lloom_tensor::DType::F32;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut parts = Vec::with_capacity(2);
        if let Some((k_all, v_all)) = dense {
            // Query t (at position past + t) sees keys ≤ past + t.
            let (k_all, v_all, q) = if d < seq {
                (
                    k_all.narrow(1, 0, past + d)?,
                    v_all.narrow(1, 0, past + d)?,
                    q.narrow(0, 0, d)?,
                )
            } else {
                (k_all, v_all, q.clone())
            };
            let kv_len = k_all.dim(1)?;
            let rep = nh / nkv;
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
            let q_h = q.transpose(0, 1)?.contiguous()?.to_dtype(f32)?; // [nh, d, hd]
            let scores = (q_h.matmul(&k_exp.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh, d, kv_len]
            let scores = if d > 1 {
                let mask = causal_mask(d, kv_len, past, dev)?
                    .broadcast_as((nh, d, kv_len))?
                    .contiguous()?;
                let neg = Tensor::full(f32::NEG_INFINITY, (nh, d, kv_len), dev)?;
                mask.where_cond(&scores, &neg)?
            } else {
                scores
            };
            let probs = ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v_exp)?; // [nh, d, hd]
            parts.push(
                ctx.transpose(0, 1)?
                    .contiguous()?
                    .reshape((d, nh * hd))?
                    .to_dtype(x.dtype())?,
            );
        }
        if d < seq {
            let x_s = x2.narrow(0, d, seq - d)?;
            let q_s = q.narrow(0, d, seq - d)?.contiguous()?.to_dtype(f32)?;
            let ctx = self.sparse_rows(&x_s, &q_s, past + d, kv, scale as f32)?;
            parts.push(ctx.to_dtype(x.dtype())?);
        }
        let ctx = if parts.len() == 1 {
            parts.pop().expect("one part")
        } else {
            Tensor::cat(&parts, 0)?
        };
        let gated = (ctx * ops::sigmoid(&gate.reshape((seq, nh * hd))?)?)?;
        self.wo.forward(&gated)?.reshape((1, seq, hidden))
    }
}

impl QsaAttention {
    /// [`Self::prefill`] as launches: the projections, `kernels::QsaPrepRows`
    /// (the q/gate split, both norms, the rope at each position, the head-first
    /// layouts the attention kernel and the cache take — no transposes), the KV
    /// append, `ops::sdpa`'s steel full-attention kernel over the cache as it
    /// lies (GQA and the causal window against the cached prefix in the
    /// kernel — no mask upload, no head expansion; the f32 arm at head 256
    /// on the reduced 8×8 tile), `kernels::SigmoidMul`, the out-proj. Metal,
    /// f32; `None` where the composed path applies.
    fn prefill_fused(&self, x: &Tensor, kv: &mut KvCache) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, sigmoid_mul, QsaPrepRows};
        let (_, seq, hidden) = x.dims3()?;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || !lloom_tensor::shape_req!(sdpa_head_dim, matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256); hd)
            || nh % nkv != 0
            || seq == 0
        {
            return Ok(None);
        }
        let f32 = lloom_tensor::DType::F32;
        let x2 = x.reshape((seq, hidden))?;
        let qg = self
            .wq
            .forward(&x2)?
            .to_dtype(f32)?
            .reshape((seq, nh, 2 * hd))?;
        let k = self
            .wk
            .forward(&x2)?
            .to_dtype(f32)?
            .reshape((seq, nkv, hd))?;
        let v = self
            .wv
            .forward(&x2)?
            .to_dtype(f32)?
            .reshape((seq, nkv, hd))?;
        let past = kv.len();
        let flat = qg.apply_op3_no_bwd(
            &k,
            &v,
            &QsaPrepRows {
                hd,
                n_rot: self.n_rot,
                nh,
                nkv,
                seq,
                pos0: past,
                theta: self.rope_theta as f32,
                eps: self.q_norm.eps() as f32,
                q_gamma: self.q_norm.weight().to_dtype(f32)?,
                k_gamma: self.k_norm.weight().to_dtype(f32)?,
            },
        )?;
        let (q_len, kv_len_e) = (nh * seq * hd, nkv * seq * hd);
        let q_h = flat.narrow(0, 0, q_len)?.reshape((1, nh, seq, hd))?;
        let gate = flat.narrow(0, q_len, q_len)?.reshape((seq, nh * hd))?;
        let k_new = flat
            .narrow(0, 2 * q_len, kv_len_e)?
            .reshape((nkv, seq, hd))?;
        let v_new = flat
            .narrow(0, 2 * q_len + kv_len_e, kv_len_e)?
            .reshape((nkv, seq, hd))?;
        kv.append(&k_new, &v_new)?;
        let d = self.dense_rows(past, seq);
        // [nkv, kv_len, hd]: the dense store's views (a 16-bit store read up
        // to f32), a quantized store's dequantized — the dense range's
        // `past + d ≤ dense_limit` rows are what the queries within it read.
        let dense = if d > 0 {
            let (k_all, v_all) = kv.dense(None)?;
            Some((k_all.to_dtype(f32)?, v_all.to_dtype(f32)?))
        } else {
            None
        };
        if let Some(ix) = &self.indexer {
            ix.observe(&x2, kv)?;
        }
        let scale = 1.0 / (hd as f64).sqrt();
        let mut parts = Vec::with_capacity(2);
        if let Some((k_all, v_all)) = dense {
            // The steel full-attention kernel over the cache as it lies: GQA
            // and the causal window against the cached prefix in the kernel
            // (query t at position past + t sees keys ≤ past + t: the
            // kernel's window is offset by kv_len − d), no mask tensor, no
            // head expansion.
            let (k_all, v_all, q_d) = if d < seq {
                (
                    k_all.narrow(1, 0, past + d)?,
                    v_all.narrow(1, 0, past + d)?,
                    q_h.narrow(2, 0, d)?.contiguous()?,
                )
            } else {
                (k_all, v_all, q_h.clone())
            };
            let ctx = ops::sdpa(
                &q_d,
                &k_all.unsqueeze(0)?,
                &v_all.unsqueeze(0)?,
                None,
                true,
                scale as f32,
                1.0,
            )?; // [1, nh, d, hd]
            parts.push(
                ctx.reshape((nh, d, hd))?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((d, nh * hd))?,
            );
        }
        if d < seq {
            let s = seq - d;
            let q_s = q_h
                .reshape((nh, seq, hd))?
                .narrow(1, d, s)?
                .transpose(0, 1)?
                .contiguous()?; // [s, nh, hd]
            parts.push(self.sparse_rows(&x2.narrow(0, d, s)?, &q_s, past + d, kv, scale as f32)?);
        }
        let ctx = if parts.len() == 1 {
            parts.pop().expect("one part")
        } else {
            Tensor::cat(&parts, 0)?
        };
        let gated = match sigmoid_mul(&gate, &ctx) {
            Some(g) => g?,
            None => (ctx * ops::sigmoid(&gate)?)?,
        };
        Ok(Some(self.wo.forward(&gated)?.reshape((1, seq, hidden))?))
    }
}

/// `QWEN4EXP_QSA_WINDOW` (read once): `batched` runs a verify window's
/// positions in one launch per kernel ([`QsaAttention::decode_seq_batched`]);
/// `rows` (the default) one position after another.
fn window_batched() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match WINDOW_ARM.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = crate::tuning::tuning().qwen4exp.qsa_window == "batched";
            WINDOW_ARM.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// 0 unread, 1 `rows`, 2 `batched`.
static WINDOW_ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Layer windows the batched arm ran, and their rows past the dense range.
static WINDOW_BATCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WINDOW_SPARSE_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(layer windows, rows past the dense range)` the batched arm has run
/// since process start: its engagement.
pub fn qsa_window_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        WINDOW_BATCHED.load(Relaxed),
        WINDOW_SPARSE_ROWS.load(Relaxed),
    )
}

/// Select the verify window's QSA arm for this process (a test runs both;
/// `QWEN4EXP_QSA_WINDOW` otherwise).
pub fn set_qsa_window_batched(on: bool) {
    WINDOW_ARM.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// A/B instrument (`QWEN4EXP_SCORE_NO_MATMUL`, read once): the block score
/// as a broadcast product + reduce instead of the `[n_blocks, 128] × [128, 4]`
/// matmul. Never the shipping path.
fn score_without_matmul() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.score_no_matmul)
}

/// A/B arm (`QWEN4EXP_SPARSE_MASKED`, read once): a prefill chunk's rows
/// past the dense range through one masked dense attention over every cell
/// (`QsaAttention::sparse_rows_masked`). Never the shipping path.
fn sparse_masked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.sparse_masked)
}

/// The `QWEN4EXP_CHECK_SEL` instrument on one selection (a no-op unless set).
fn check_sel(sel: &Tensor, pos: usize, kv: &KvCache) -> Result<()> {
    if !check_selection() {
        return Ok(());
    }
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
    Ok(())
}

/// The `QWEN4EXP_CHECK_SEL` instrument on a chunk's rows past the dense
/// range (a no-op unless set): each row's cells against the selection the
/// one-query path ([`Indexer::select`]) makes for the same input on the
/// same cache, as sets. A row that differs prints a `SELROWS` line naming
/// the blocks only one side took and their one-query scores beside the
/// score at its top-k boundary (a near-tie reads as equal scores); every
/// call prints a summary line.
fn check_rows(
    ix: &Indexer,
    x: &Tensor,
    pos0: usize,
    sel: &RowSelection,
    kv: &KvCache,
) -> Result<()> {
    if !check_selection() {
        return Ok(());
    }
    let s = x.dim(0)?;
    let mut differ = 0usize;
    for t in 0..s {
        let p = pos0 + t;
        let mut got = sel.row(t)?.to_vec1::<u32>()?;
        let bad = got.iter().filter(|&&c| c as usize > p).count();
        let mut want = match ix.select(&x.narrow(0, t, 1)?, p, kv)? {
            Some(w) => w.to_vec1::<u32>()?,
            None => lloom_tensor::bail!("SELROWS pos={p}: within the dense range"),
        };
        got.sort_unstable();
        want.sort_unstable();
        let dup = got.windows(2).filter(|w| w[0] == w[1]).count();
        if bad > 0 || dup > 0 {
            lloom_tensor::bail!("SELROWS pos={p}: {bad} cells past it, {dup} duplicates")
        }
        if got == want {
            continue;
        }
        differ += 1;
        let blocks = |v: &[u32]| -> std::collections::BTreeSet<usize> {
            v.iter().map(|&c| c as usize / ix.ratio).collect()
        };
        let (bg, bw) = (blocks(&got), blocks(&want));
        let only_rows: Vec<usize> = bg.difference(&bw).copied().collect();
        let only_one: Vec<usize> = bw.difference(&bg).copied().collect();
        let score = ix.score_one(&x.narrow(0, t, 1)?, p, kv)?.to_vec1::<f32>()?;
        let mut ranked = score.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));
        let edge = ranked[ix.block_topk().min(ranked.len()) - 1];
        let at = |b: &[usize]| -> Vec<(usize, f32)> { b.iter().map(|&i| (i, score[i])).collect() };
        eprintln!(
            "SELROWS pos={p} differ: rows-only {:?} one-query-only {:?} (one-query scores; top-k edge {edge})",
            at(&only_rows),
            at(&only_one)
        );
    }
    eprintln!("SELROWS pos0={pos0} rows={s} differ={differ}");
    Ok(())
}

/// Read the selection back and check it every position (`QWEN4EXP_CHECK_SEL`,
/// read once): a debugging instrument behind a gate (§3.3a) — a host sync per
/// full layer per token, never on the shipping path.
fn check_selection() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.check_sel)
}

/// `(cos, sin)` `[positions.len(), n_rot/2]` for partial RoPE at `positions`.
fn rope_cs_at(
    n_rot: usize,
    theta: f64,
    positions: &[usize],
    dev: &lloom_tensor::Device,
) -> Result<(Tensor, Tensor)> {
    let half = n_rot / 2;
    let (cos, sin) = rope_cs_host(n_rot, theta, positions.iter().copied());
    Ok((
        Tensor::from_vec(cos, (positions.len(), half), dev)?,
        Tensor::from_vec(sin, (positions.len(), half), dev)?,
    ))
}

/// The partial rope's `(cos, sin)` at each of `positions`, `n_rot / 2` each,
/// in the host's f32 arithmetic (which the device's is not, bit for bit).
fn rope_cs_host(
    n_rot: usize,
    theta: f64,
    positions: impl ExactSizeIterator<Item = usize>,
) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for pos in positions {
        for i in 0..half {
            let inv = 1f32 / (theta as f32).powf(2.0 * i as f32 / n_rot as f32);
            let f = inv * pos as f32;
            cos.push(f.cos());
            sin.push(f.sin());
        }
    }
    (cos, sin)
}

/// [`rope_cs_at`] at one position, `[1, n_rot / 2]` each: on Metal a view
/// of a table of every position up to a power of two, computed on the host
/// by the same arithmetic and uploaded once per growth, so a decode step's
/// rope asks no pool for an upload (`DESIGN.md` §9.2aa). One table per
/// device and `(n_rot, theta)`, kept for the process.
fn rope_cs_one(
    n_rot: usize,
    theta: f64,
    pos: usize,
    dev: &lloom_tensor::Device,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "metal")]
    if let lloom_tensor::Device::Metal(m) = dev {
        type Tables = std::collections::HashMap<
            (lloom_tensor::metal_backend::DeviceId, usize, u64),
            (Tensor, Tensor, usize),
        >;
        static TABLES: std::sync::OnceLock<parking_lot::Mutex<Tables>> = std::sync::OnceLock::new();
        let half = n_rot / 2;
        let mut g = TABLES
            .get_or_init(|| parking_lot::Mutex::new(Tables::new()))
            .lock();
        let key = (m.id(), n_rot, theta.to_bits());
        if g.get(&key).is_none_or(|t| t.2 <= pos) {
            let rows = (pos + 1).next_power_of_two().max(4096);
            let (cos, sin) = rope_cs_host(n_rot, theta, 0..rows);
            // The table outlives every pass: no plan serves it.
            let _declared = lloom_tensor::EscapeGuard::new();
            let table = (
                Tensor::from_vec(cos, (rows, half), dev)?,
                Tensor::from_vec(sin, (rows, half), dev)?,
                rows,
            );
            g.insert(key, table);
        }
        let (cos, sin, _) = g.get(&key).expect("made above");
        return Ok((cos.narrow(0, pos, 1)?, sin.narrow(0, pos, 1)?));
    }
    rope_cs_at(n_rot, theta, &[pos], dev)
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

/// A full-attention layer's decode step ([`QsaAttention::decode_rows`]'s
/// launches at one row, inside the indexer's dense range, no block
/// completing): the input stems as the loader stacked them, the prologue
/// (norms and partial rope), the append at the store's precision, the
/// indexer's key projection and its append, the attention, the output gate,
/// the output projection. The cache, the appends and the attention over
/// `kv_len` are layer `l`'s KV recurrence. `In` is the bracket's output, the
/// layer, and the KV store's precision (the cache's, which the model knows).
///
/// Not walked: the step that completes an indexer block (every `ratio`-th
/// position: the pooled keys' norm, rope and append), the step past the
/// dense range (the selection), and the two-pass attention past
/// `sdpa_vector`'s single-pass length — each adds launches at a position.
#[cfg(not(target_arch = "wasm32"))]
impl crate::optimizer::Visit for QsaAttention {
    type In = (crate::optimizer::NodeId, usize, KvPrecision);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (x, l, precision): Self::In) -> Self::Out {
        use crate::optimizer::{Dim, Region, Shape};
        let f32 = DType::F32;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let h = match w.node(x).shape.last() {
            Some(Dim::Const(h)) => h,
            _ => unreachable!("a decode step's input has a constant width"),
        };
        let (nq, nk) = (nh * 2 * hd, nkv * hd);
        let xb = w.view("reshape", x, Shape::of(&[1, h]));
        let (qg, k, v) = match &self.stem_cat {
            Some(cat) => {
                let all = w.project_qmat("attn_q.weight+attn_k.weight+attn_v.weight", cat, xb);
                (
                    w.view("narrow", all, Shape::of(&[1, nq])),
                    w.view("narrow", all, Shape::of(&[1, nk])),
                    w.view("narrow", all, Shape::of(&[1, nk])),
                )
            }
            None => (
                w.project_qmat("attn_q.weight", &self.wq, xb),
                w.project_qmat("attn_k.weight", &self.wk, xb),
                w.project_qmat("attn_v.weight", &self.wv, xb),
            ),
        };
        let qg = w.view("reshape", qg, Shape::of(&[nh, 2 * hd]));
        let k = w.view("reshape", k, Shape::of(&[nkv, hd]));
        let v = w.view("reshape", v, Shape::of(&[nkv, 1, hd]));
        let qn = self.q_norm.weight();
        let kn = self.k_norm.weight();
        let qn = w.weight("attn_q_norm.weight", Shape::of(qn.dims()), qn.dtype());
        let kn = w.weight("attn_k_norm.weight", Shape::of(kn.dims()), kn.dtype());
        let prep = w.opaque(
            "qsa_prep",
            &[qg, k, qn, kn],
            Shape::of(&[2 * nh + nkv, hd]),
            f32,
        );
        let q = w.view("narrow", prep, Shape::of(&[nh, hd]));
        let gate = w.view("narrow", prep, Shape::of(&[nh, hd]));
        let k_step = w.view("narrow", prep, Shape::of(&[nkv, 1, hd]));
        let cache = || Shape(vec![Dim::Const(nkv), Dim::Sym("kv_len"), Dim::Const(hd)]);
        let mut ring = Vec::new();
        let (k_all, v_all) = match precision.dense_dtype() {
            Some(dt) => {
                let shape = cache();
                let kc = w.input(&format!("qsa.k@{l}"), shape.clone(), dt);
                let vc = w.input(&format!("qsa.v@{l}"), shape.clone(), dt);
                let (copy, cast_in, cast_out) = if dt == f32 {
                    ("copy2d_f32", None, None)
                } else {
                    ("copy2d_f16", Some("cast_f32_f16"), Some("cast_f16_f32"))
                };
                let (mut ks, mut vs) = (k_step, v);
                if let Some(c) = cast_in {
                    ks = w.opaque(c, &[ks], Shape::of(&[nkv, 1, hd]), dt);
                    vs = w.opaque(c, &[vs], Shape::of(&[nkv, 1, hd]), dt);
                    ring.extend([ks, vs]);
                }
                let mut ka = w.opaque(copy, &[ks, kc], shape.clone(), dt);
                let mut va = w.opaque(copy, &[vs, vc], shape.clone(), dt);
                ring.extend([kc, vc, ka, va]);
                if let Some(c) = cast_out {
                    ka = w.opaque(c, &[ka], shape.clone(), f32);
                    va = w.opaque(c, &[va], shape, f32);
                    ring.extend([ka, va]);
                }
                (ka, va)
            }
            // A quantized store's append and its read are its own kernels,
            // named here from the arm and reconciled against no trace.
            None => {
                let shape = cache();
                let store = w.input(&format!("qsa.kv@{l}"), shape.clone(), DType::U8);
                let appended = w.opaque("kv_quant_append", &[k_step, v, store], shape, DType::U8);
                ring.extend([store, appended]);
                (appended, appended)
            }
        };
        if let Some(ix) = &self.indexer {
            let k_raw = w.project_qmat("indexer.k_proj.weight", &ix.k_proj, xb);
            let k_raw = w.view("reshape", k_raw, Shape::of(&[1, 1, ix.head_dim]));
            let raw = w.input(
                &format!("qsa.indexer_keys@{l}"),
                Shape(vec![
                    Dim::Const(1),
                    Dim::Sym("kv_len"),
                    Dim::Const(ix.head_dim),
                ]),
                f32,
            );
            let appended = w.opaque(
                "copy2d_f32",
                &[k_raw, raw],
                Shape(vec![
                    Dim::Const(1),
                    Dim::Sym("kv_len"),
                    Dim::Const(ix.head_dim),
                ]),
                f32,
            );
            // Read by a later step's selection, not by this one.
            w.state_out(appended);
            ring.extend([raw, appended]);
        }
        let qv = w.view("reshape", q, Shape::of(&[1, nh, 1, hd]));
        let kernel = if precision.dense_dtype().is_some() {
            "sdpa_vector"
        } else {
            "sdpa_quant"
        };
        let ctx = w.opaque(kernel, &[qv, k_all, v_all], Shape::of(&[1, nh, 1, hd]), f32);
        ring.push(ctx);
        w.mark(&ring, Region::Kv(l));
        let ctx = w.view("reshape", ctx, Shape::of(&[1, nh * hd]));
        let gate = w.view("reshape", gate, Shape::of(&[1, nh * hd]));
        let gated = w.opaque("sigmoid_mul", &[gate, ctx], Shape::of(&[1, nh * hd]), f32);
        let out = w.project_qmat("attn_output.weight", &self.wo, gated);
        w.view("reshape", out, Shape::of(&[1, 1, h]))
    }
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

    /// `KvPool`: a row past the pool's first slab grows the layer (the pages
    /// re-based), `live` gathers the tokens in order through the table,
    /// `truncate` keeps a prefix, `clear` returns the pages; `slots_of` maps
    /// cells as the host does. CPU (the scatter's reference arm).
    #[test]
    fn kv_pool_grows_gathers_and_maps() -> Result<()> {
        let dev = Device::Cpu;
        let (nkv, hd) = (2usize, 8usize);
        let pool = KvPool::new(1, 1, nkv, hd, DType::F32, &dev);
        let lease = pool.lease()?;
        let mut kv = KvCache::in_pool(&pool, &lease, 0);
        let tok = |t: usize| -> Tensor {
            Tensor::from_vec(
                (0..nkv * hd)
                    .map(|i| (t * 100 + i) as f32)
                    .collect::<Vec<f32>>(),
                (nkv, 1, hd),
                &dev,
            )
            .unwrap()
        };
        // One slab is KV_SLAB_TOKENS tokens; cross it (the second slab).
        let n = KV_SLAB_TOKENS + 37;
        for t in 0..n {
            kv.append(&tok(t), &(tok(t) + 0.5)?)?;
            let (k_all, _) = kv.dense(None)?;
            assert_eq!(k_all.dims(), &[nkv, t + 1, hd]);
        }
        assert_eq!(kv.len(), n);
        assert_eq!(pool.tokens(0), 2 * KV_SLAB_TOKENS);
        let [k, v, _, _] = kv.state_views()?;
        let k = k.expect("live");
        let v = v.expect("live");
        for t in [0usize, 255, 256, 1023, 1024, n - 1] {
            let row: Vec<f32> = k.narrow(1, t, 1)?.flatten_all()?.to_vec1()?;
            let want: Vec<f32> = (0..nkv * hd).map(|i| (t * 100 + i) as f32).collect();
            assert_eq!(row, want, "token {t}");
            let row: Vec<f32> = v.narrow(1, t, 1)?.flatten_all()?.to_vec1()?;
            let want_v: Vec<f32> = want.iter().map(|x| x + 0.5).collect();
            assert_eq!(row, want_v, "token {t} (v)");
        }
        // The cell → slot map through the row's table.
        let pages = pool.pages(0, 0);
        let table = Tensor::from_vec(pages.clone(), pages.len(), &dev)?;
        let cells: Vec<u32> = vec![0, 3, 255, 256, 700, (n - 1) as u32];
        let sel = Tensor::from_vec(cells.clone(), cells.len(), &dev)?;
        let got: Vec<u32> = pool.slots_of(&table, &sel)?.to_vec1()?;
        let ps = KV_PAGE_TOKENS;
        let want: Vec<u32> = cells
            .iter()
            .map(|&c| pages[c as usize / ps] * ps as u32 + c % ps as u32)
            .collect();
        assert_eq!(got, want);
        let (gk, _) = kv.gather(&sel)?;
        let row: Vec<f32> = gk.narrow(1, 4, 1)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = (0..nkv * hd).map(|i| (700 * 100 + i) as f32).collect();
        assert_eq!(row, want);
        kv.truncate(300, 0);
        assert_eq!(kv.len(), 300);
        kv.append(&tok(300), &(tok(300) + 0.5)?)?;
        let (k_all, _) = kv.dense(None)?;
        assert_eq!(k_all.dims(), &[nkv, 301, hd]);
        kv.reset();
        assert_eq!(kv.len(), 0);
        assert!(pool.pages(0, 0).is_empty());
        Ok(())
    }

    /// The pooled rows arm's two kernels equal the per-row kernels bit for
    /// bit: `QsaPrepBatch` over rows at their own positions = `QsaPrep` per
    /// row; `SdpaVectorPaged` over a pool whose pages are scattered = the
    /// vector `ops::sdpa` over each row's keys laid contiguously.
    #[cfg(feature = "metal")]
    #[test]
    fn pooled_rows_kernels_match_per_row_bit_for_bit() -> Result<()> {
        use super::super::kernels::{QsaPrep, QsaPrepBatch};
        let dev = Device::new_metal(0)?;
        let cfg = Config::flash_next();
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let rows = 4usize;
        let gamma = |seed: f32| {
            Tensor::from_vec(
                (0..hd)
                    .map(|i| 1.0 + 0.1 * ((i as f32 + seed) * 0.3).sin())
                    .collect::<Vec<f32>>(),
                hd,
                &dev,
            )
            .unwrap()
        };
        let (qg_w, kg_w) = (gamma(0.0), gamma(9.0));
        let qg = Tensor::randn(0f32, 1.0, (rows, nh, 2 * hd), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (rows, nkv, hd), &dev)?;
        let pos: Vec<u32> = vec![7, 300, 1500, 1100];
        let pos_t = Tensor::from_vec(pos.clone(), rows, &dev)?;
        let prep = qg.apply_op3_no_bwd(
            &k,
            &pos_t,
            &QsaPrepBatch {
                hd,
                n_rot: (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize,
                nh,
                nkv,
                theta: cfg.rope_theta as f32,
                eps: 1e-6,
                q_gamma: qg_w.clone(),
                k_gamma: kg_w.clone(),
            },
        )?;
        let q_reg = prep.narrow(0, 0, rows * nh * hd)?.reshape((rows, nh, hd))?;
        let g_reg = prep
            .narrow(0, rows * nh * hd, rows * nh * hd)?
            .reshape((rows, nh, hd))?;
        let k_reg = prep
            .narrow(0, 2 * rows * nh * hd, rows * nkv * hd)?
            .reshape((rows, nkv, hd))?;
        for r in 0..rows {
            let one = qg
                .narrow(0, r, 1)?
                .reshape((nh, 2 * hd))?
                .apply_op2_no_bwd(
                    &k.narrow(0, r, 1)?.reshape((nkv, hd))?,
                    &QsaPrep {
                        hd,
                        n_rot: (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize,
                        nh,
                        nkv,
                        pos: pos[r] as usize,
                        theta: cfg.rope_theta as f32,
                        eps: 1e-6,
                        q_gamma: qg_w.clone(),
                        k_gamma: kg_w.clone(),
                    },
                )?;
            let a: Vec<f32> = Tensor::cat(
                &[
                    q_reg.narrow(0, r, 1)?.flatten_all()?,
                    g_reg.narrow(0, r, 1)?.flatten_all()?,
                    k_reg.narrow(0, r, 1)?.flatten_all()?,
                ],
                0,
            )?
            .to_vec1()?;
            let b: Vec<f32> = one.flatten_all()?.to_vec1()?;
            assert!(a.iter().all(|x| x.is_finite()));
            assert_eq!(a, b, "prep row {r}");
        }
        // Three rows of different lengths in a pool whose pages are handed
        // out from the back (so no row is contiguous), one launch.
        let lens = [5usize, 600, 1300, 1030];
        let page = KV_PAGE_TOKENS;
        let pool_pages = 16;
        let pk = Tensor::randn(0f32, 1.0, (nkv, pool_pages * page, hd), &dev)?;
        let pv = Tensor::randn(0f32, 1.0, (nkv, pool_pages * page, hd), &dev)?;
        let mut free: Vec<u32> = (0..pool_pages as u32).collect();
        let mut tables: Vec<Vec<u32>> = Vec::new();
        for &n in &lens {
            let mut t = Vec::new();
            for _ in 0..n.div_ceil(page) {
                t.push(free.pop().expect("pages"));
            }
            tables.push(t);
        }
        let n_keys: Vec<u32> = lens.iter().map(|&n| n as u32).collect();
        let q = Tensor::randn(0f32, 1.0, (rows, nh, hd), &dev)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let attn = QsaAttention::new(
            lin(nh * hd * 2, 8, 1.0, &dev),
            lin(nkv * hd, 8, 2.0, &dev),
            lin(nkv * hd, 8, 3.0, &dev),
            lin(8, nh * hd, 4.0, &dev),
            rms(hd, &dev),
            rms(hd, &dev),
            &cfg,
        );
        // Rows 0 and 1 below the 2-pass threshold, rows 2 and 3 above: the
        // two arms, each against the per-row kernel of that length.
        let ctx = attn.sdpa_paged_rows(&q, &pk, &pv, page, &tables, &n_keys, scale)?; // [rows, nh, hd]
        for r in 0..rows {
            let ids: Vec<u32> = (0..lens[r])
                .map(|t| tables[r][t / page] * page as u32 + (t % page) as u32)
                .collect();
            let ids = Tensor::from_vec(ids, lens[r], &dev)?;
            let kr = pk.index_select(&ids, 1)?.unsqueeze(0)?; // [1, nkv, n, hd]
            let vr = pv.index_select(&ids, 1)?.unsqueeze(0)?;
            let qr = q.narrow(0, r, 1)?.reshape((1, nh, 1, hd))?;
            let want = ops::sdpa(&qr, &kr, &vr, None, false, scale, 1.0)?;
            let a: Vec<f32> = ctx.narrow(0, r, 1)?.flatten_all()?.to_vec1()?;
            let b: Vec<f32> = want.flatten_all()?.to_vec1()?;
            assert!(a.iter().all(|x| x.is_finite()));
            assert_eq!(a, b, "attention row {r} ({} keys)", lens[r]);
        }
        Ok(())
    }

    /// The fused decode (the prep kernel, the sdpa vector kernel over the
    /// cache as it lies, the gate kernel) matches the composed path — the
    /// CPU's — at the real head geometry (24 heads of 256 over 2 kv heads,
    /// rope on 64 dims, non-trivial γ) over 12 steps from a fresh cache.
    #[cfg(feature = "metal")]
    #[test]
    fn decode_fused_matches_composed_at_real_geometry() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let build = |dev: &Device| {
            let gamma = |seed: f32| {
                RmsNorm::new(
                    Tensor::from_vec(
                        (0..hd)
                            .map(|i| 1.0 + 0.1 * ((i as f32 + seed) * 0.3).sin())
                            .collect::<Vec<f32>>(),
                        hd,
                        dev,
                    )
                    .unwrap(),
                    1e-6,
                )
            };
            QsaAttention::new(
                lin(nh * hd * 2, d, 1.0, dev),
                lin(nkv * hd, d, 2.0, dev),
                lin(nkv * hd, d, 3.0, dev),
                lin(d, nh * hd, 4.0, dev),
                gamma(0.0),
                gamma(9.0),
                &cfg,
            )
        };
        let (a_cpu, a_gpu) = (build(&cpu), build(&gpu));
        let (mut kv_c, mut kv_g) = (KvCache::default(), KvCache::default());
        let mut worst = 0f32;
        for t in 0..12 {
            let x = Tensor::from_vec(
                (0..d)
                    .map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, 1, d),
                &cpu,
            )
            .unwrap();
            let a = a_cpu.decode(&x, &mut kv_c).unwrap();
            let b = a_gpu
                .decode(&x.to_device(&gpu).unwrap(), &mut kv_g)
                .unwrap();
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<f32> = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                bv.iter().all(|v| v.is_finite()),
                "step {t}: non-finite fused output"
            );
            let scale = av.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = av
                .iter()
                .zip(&bv)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale);
            assert!(
                diff <= 2e-4 * scale,
                "step {t}: fused differs by {diff} at scale {scale}"
            );
        }
        assert_eq!(kv_c.len(), kv_g.len());
        assert_eq!(kv_g.len(), 12);
        println!("qsa fused vs composed worst relative diff: {worst:e}");
    }

    /// The fused prefill over a CACHED PREFIX (the model's chunked prompt:
    /// a 40-position chunk, then a 25-position chunk attending to it — the
    /// causal window offset by the cache) matches the composed path, the
    /// CPU's, at the real head geometry.
    #[cfg(feature = "metal")]
    #[test]
    fn prefill_fused_matches_composed_over_a_cached_prefix() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let build = |dev: &Device| {
            let gamma = |seed: f32| {
                RmsNorm::new(
                    Tensor::from_vec(
                        (0..hd)
                            .map(|i| 1.0 + 0.1 * ((i as f32 + seed) * 0.3).sin())
                            .collect::<Vec<f32>>(),
                        hd,
                        dev,
                    )
                    .unwrap(),
                    1e-6,
                )
            };
            QsaAttention::new(
                lin(nh * hd * 2, d, 1.0, dev),
                lin(nkv * hd, d, 2.0, dev),
                lin(nkv * hd, d, 3.0, dev),
                lin(d, nh * hd, 4.0, dev),
                gamma(0.0),
                gamma(9.0),
                &cfg,
            )
        };
        let (a_cpu, a_gpu) = (build(&cpu), build(&gpu));
        let (mut kv_c, mut kv_g) = (KvCache::default(), KvCache::default());
        let mut worst = 0f32;
        for (round, seq) in [40usize, 25].into_iter().enumerate() {
            let x = Tensor::from_vec(
                (0..seq * d)
                    .map(|i| ((i as f32 * 0.05 + round as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, seq, d),
                &cpu,
            )
            .unwrap();
            let a = a_cpu.prefill(&x, &mut kv_c).unwrap();
            let b = a_gpu
                .prefill(&x.to_device(&gpu).unwrap(), &mut kv_g)
                .unwrap();
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<f32> = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                bv.iter().all(|v| v.is_finite()),
                "round {round}: non-finite fused output"
            );
            let scale = av.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = av
                .iter()
                .zip(&bv)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale);
            assert!(
                diff <= 2e-4 * scale,
                "round {round}: fused differs by {diff} at scale {scale}"
            );
            assert_eq!(kv_c.len(), kv_g.len());
        }
        assert_eq!(kv_g.len(), 65);
        println!("qsa prefill fused vs composed worst relative diff: {worst:e}");
    }

    /// The stacked f16 stems (one gemv, the row split `q | k | v`) decode as
    /// the three stems do — the same weights as row views of one matrix —
    /// at the real head geometry over 12 steps.
    #[cfg(feature = "metal")]
    #[test]
    fn stacked_stems_decode_as_the_three_stems() {
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let f16 = |out: usize, seed: f32| {
            Tensor::from_vec(
                (0..out * d)
                    .map(|i| ((i as f32 * 0.011 + seed).sin()) * 0.05)
                    .collect::<Vec<_>>(),
                (out, d),
                &gpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let parts = [
            f16(nh * hd * 2, 1.0),
            f16(nkv * hd, 2.0),
            f16(nkv * hd, 3.0),
        ];
        let cat = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap();
        let mut off = 0;
        let views: Vec<Tensor> = parts
            .iter()
            .map(|p| {
                let v = cat.narrow(0, off, p.dim(0).unwrap()).unwrap();
                off += p.dim(0).unwrap();
                v
            })
            .collect();
        let build = |stack: Option<Tensor>| {
            let attn = QsaAttention::new(
                QMatMul::TensorF16(views[0].clone()),
                QMatMul::TensorF16(views[1].clone()),
                QMatMul::TensorF16(views[2].clone()),
                lin(d, nh * hd, 4.0, &gpu),
                rms(hd, &gpu),
                rms(hd, &gpu),
                &cfg,
            );
            match stack {
                Some(c) => attn.with_stem_cat(c),
                None => attn,
            }
        };
        let (three, one) = (build(None), build(Some(cat.clone())));
        // Non-vacuity: a stack with the v rows zeroed must decode differently.
        let zeroed = Tensor::cat(
            &[
                &parts[0],
                &parts[1],
                &Tensor::zeros(parts[2].shape(), DType::F16, &gpu).unwrap(),
            ],
            0,
        )
        .unwrap();
        let bad = build(Some(zeroed));
        let (mut kv3, mut kv1, mut kvb) =
            (KvCache::default(), KvCache::default(), KvCache::default());
        let mut worst = 0f32;
        for t in 0..12 {
            let x = Tensor::from_vec(
                (0..d)
                    .map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, 1, d),
                &gpu,
            )
            .unwrap();
            let a: Vec<f32> = three
                .decode(&x, &mut kv3)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = one
                .decode(&x, &mut kv1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let bad_out: Vec<f32> = bad
                .decode(&x, &mut kvb)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_ne!(a, bad_out, "step {t}: the stacked path did not run");
            assert!(b.iter().all(|v| v.is_finite()), "step {t}: non-finite");
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale);
            assert!(
                diff <= 1e-4 * scale,
                "step {t}: stacked differs by {diff} at {scale}"
            );
        }
        println!("stacked vs three stems worst relative diff: {worst:e}");
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
    /// A single position's rope from the device table is the host's upload
    /// for that position, bit for bit, on either side of the table's first
    /// growth: the table is the same arithmetic, stored once.
    #[cfg(feature = "metal")]
    #[test]
    fn the_rope_table_is_the_hosts_rope_at_every_position() {
        let dev = Device::new_metal(0).unwrap();
        let (n_rot, theta) = (64usize, 1e7f64);
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        for pos in [0usize, 1, 4, 5, 2051, 4095, 4096, 4100, 9000] {
            let (c1, s1) = rope_cs_one(n_rot, theta, pos, &dev).unwrap();
            let (c0, s0) = rope_cs_at(n_rot, theta, &[pos], &dev).unwrap();
            assert_eq!(c1.dims(), c0.dims(), "pos {pos}");
            assert_eq!(bits(&c1), bits(&c0), "cos at pos {pos}");
            assert_eq!(bits(&s1), bits(&s0), "sin at pos {pos}");
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

    /// NaN-carrying max |a − b|: a NaN on either side is the answer (a fold
    /// over `f32::max` would drop it and read as agreement).
    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "compared rows differ in length");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, |m, d| {
                if m.is_nan() || d.is_nan() {
                    f32::NAN
                } else {
                    m.max(d)
                }
            })
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// [`indexed_attn`]'s layer without its indexer: dense over every cell.
    fn dense_twin(cfg: &Config, dev: &Device) -> QsaAttention {
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        QsaAttention::new(
            lin(nh * hd * 2, h, 1.0, dev),
            lin(nkv * hd, h, 2.0, dev),
            lin(nkv * hd, h, 3.0, dev),
            lin(h, nh * hd, 4.0, dev),
            rms(hd, dev),
            rms(hd, dev),
            cfg,
        )
    }

    /// A pooled row whose cells are NOT its pool slots: another row takes
    /// the pool's first page before it, so a read that skipped the page
    /// table would read that row's tokens. Returns the row's cache and the
    /// other row's (kept alive for the test).
    fn pooled_row_off_page_zero(cfg: &Config, dev: &Device) -> (KvCache, KvCache) {
        let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
        let pool = KvPool::new(2, 1, nkv, hd, DType::F32, dev);
        let other = pool.lease().unwrap();
        let mut kv_o = KvCache::in_pool(&pool, &other, 0);
        let k = Tensor::ones((nkv, 1, hd), DType::F32, dev).unwrap();
        kv_o.append(&(&k * 7.0).unwrap(), &(&k * -7.0).unwrap())
            .unwrap();
        let lease = pool.lease().unwrap();
        let kv = KvCache::in_pool(&pool, &lease, 0);
        (kv, kv_o)
    }

    /// The inputs `tok(h, s0 + p)` for `n` positions.
    fn inputs(h: usize, n: usize, s0: f32) -> Vec<Vec<f32>> {
        (0..n).map(|p| tok(h, s0 + p as f32)).collect()
    }

    /// Each of `xs` through `attn` as one decode step on `kv` (at `pos0 +
    /// i`): every output, and the selection made at each step.
    fn decode_steps(
        attn: &QsaAttention,
        xs: &[Vec<f32>],
        pos0: usize,
        kv: &mut KvCache,
        dev: &Device,
    ) -> (Vec<Vec<f32>>, Vec<Option<Vec<u32>>>) {
        let h = xs[0].len();
        let mut outs = Vec::new();
        let mut sels = Vec::new();
        for (i, x) in xs.iter().enumerate() {
            let x = Tensor::from_vec(x.clone(), (1, 1, h), dev).unwrap();
            outs.push(host(&attn.decode(&x, kv).unwrap()));
            sels.push(
                attn.selection(&x, pos0 + i, kv)
                    .unwrap()
                    .map(|s| s.to_device(&Device::Cpu).unwrap().to_vec1().unwrap()),
            );
        }
        (outs, sels)
    }

    /// `xs` as one chunk through `attn.prefill` on `kv`: one output row per
    /// position.
    fn prefill_chunk(
        attn: &QsaAttention,
        xs: &[Vec<f32>],
        kv: &mut KvCache,
        dev: &Device,
    ) -> Vec<Vec<f32>> {
        let h = xs[0].len();
        let x = Tensor::from_vec(xs.concat(), (1, xs.len(), h), dev).unwrap();
        let out = host(&attn.prefill(&x, kv).unwrap());
        out.chunks(h).map(|r| r.to_vec()).collect()
    }

    /// The chunk's selection for its rows past the dense range, read back:
    /// row `t` (position `pos0 + t`)'s cells.
    fn chunk_selection(
        attn: &QsaAttention,
        xs: &[Vec<f32>],
        pos0: usize,
        kv: &KvCache,
        dev: &Device,
    ) -> Vec<Vec<u32>> {
        let h = xs[0].len();
        let x = Tensor::from_vec(xs.concat(), (1, xs.len(), h), dev).unwrap();
        let sel = attn.selection_rows(&x, pos0, kv).unwrap().unwrap();
        (0..xs.len())
            .map(|t| {
                sel.row(t)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            })
            .collect()
    }

    /// A chunk that starts inside the dense range, crosses it and completes
    /// blocks on its way selects, for every query past the range, the same
    /// cells `select` names for that query on a cache built one decode step
    /// at a time: the same set, the tail in it, every cell at or before the
    /// query. Non-vacuous by construction, asserted: the reference itself
    /// takes blocks the chunk completed (a chunk blind to its own blocks
    /// would differ there), and some rows have a tail.
    #[test]
    fn chunk_selection_equals_the_decode_selection() {
        let dev = Device::Cpu;
        let cfg = tiny_indexed();
        let attn = indexed_attn(&cfg, &dev);
        let (h, r) = (cfg.hidden_size, cfg.indexer_compress_ratio);
        let limit = attn.dense_limit().unwrap();
        let (past, n) = (6usize, 31usize);
        let xs = inputs(h, n, 0.0);
        let mut kv_d = KvCache::with_capacity(4);
        let (_, want) = decode_steps(&attn, &xs, 0, &mut kv_d, &dev);
        let mut kv_c = KvCache::with_capacity(4);
        decode_steps(&attn, &xs[..past], 0, &mut kv_c, &dev);
        prefill_chunk(&attn, &xs[past..], &mut kv_c, &dev);
        assert_eq!(kv_c.lens(), kv_d.lens());
        assert_eq!(kv_c.lens(), (n, n / r));
        let got = chunk_selection(&attn, &xs[limit..], limit, &kv_c, &dev);
        let (mut chunk_blocks, mut tails) = (0, 0);
        for (t, g) in got.iter().enumerate() {
            let p = limit + t;
            let w = want[p].as_ref().expect("past the dense range");
            let (mut gs, mut ws) = (g.clone(), w.clone());
            gs.sort_unstable();
            ws.sort_unstable();
            assert_eq!(gs, ws, "pos {p}: the chunk's selection vs decode's");
            assert!(g.iter().all(|&c| c as usize <= p), "pos {p}: a future cell");
            let complete = (p + 1) / r * r;
            if w.iter()
                .any(|&c| (c as usize) < complete && c as usize / r * r + r > past)
            {
                chunk_blocks += 1;
            }
            if complete <= p {
                tails += 1;
            }
        }
        assert!(
            chunk_blocks > 0,
            "no query selects a block the chunk completed: the fixture is blind"
        );
        assert!(tails > 0, "no query past the range has a tail");
    }

    /// The chunk prefill past the dense range computes what the decode
    /// replay computes, within fp32 reordering: a chunk crossing the range,
    /// then one starting mid-block wholly past it, against one decode step
    /// per position — CPU, the composed reference. The level is pinned by
    /// a control: the same layer without an indexer (dense over every cell)
    /// is far further from the replay than the tolerance, so a prefill that
    /// ignored the selection fails. A pooled row computes the slab's bits.
    #[test]
    fn chunk_prefill_past_the_dense_range_matches_decode_replay() {
        let dev = Device::Cpu;
        let cfg = tiny_indexed();
        let attn = indexed_attn(&cfg, &dev);
        let dense = dense_twin(&cfg, &dev);
        let h = cfg.hidden_size;
        let (n, cuts) = (40usize, [3usize, 27]);
        let xs = inputs(h, n, 0.5);
        let mut kv_d = KvCache::with_capacity(4);
        let (want, _) = decode_steps(&attn, &xs, 0, &mut kv_d, &dev);
        let mut kv_all = KvCache::with_capacity(4);
        let (all, _) = decode_steps(&dense, &xs, 0, &mut kv_all, &dev);
        let run = |kv: &mut KvCache| -> Vec<Vec<f32>> {
            let mut out = decode_steps(&attn, &xs[..cuts[0]], 0, kv, &dev).0;
            out.extend(prefill_chunk(&attn, &xs[cuts[0]..cuts[1]], kv, &dev));
            out.extend(prefill_chunk(&attn, &xs[cuts[1]..], kv, &dev));
            out
        };
        let mut kv_c = KvCache::with_capacity(4);
        let got = run(&mut kv_c);
        assert_eq!(kv_c.lens(), kv_d.lens());
        let scale = want.iter().flatten().fold(0f32, |m, x| m.max(x.abs()));
        let limit = attn.dense_limit().unwrap();
        let err = (0..n)
            .map(|p| max_diff(&got[p], &want[p]))
            .fold(0f32, |m, d| if d.is_nan() { d } else { m.max(d) });
        let d_sel = (limit..n)
            .map(|p| max_diff(&all[p], &want[p]))
            .fold(0f32, f32::max);
        println!("chunk vs replay {err:e}, dense-over-all vs replay {d_sel:e}, scale {scale}");
        assert!(
            err <= 1e-5 * scale,
            "chunk vs replay: {err} at scale {scale}"
        );
        assert!(
            d_sel > 1e3 * 1e-5 * scale,
            "the selection barely moves the output ({d_sel}): the control is blind"
        );
        let (mut kv_p, _other) = pooled_row_off_page_zero(&cfg, &dev);
        let pooled = run(&mut kv_p);
        for p in 0..n {
            let same = pooled[p]
                .iter()
                .zip(&got[p])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "pos {p}: the pooled row is not the slab's bits");
        }
    }

    /// The Metal chunk prefill past the dense range (the paged-vector
    /// attention over each row's cells, single-pass at this budget) against
    /// the Metal decode replay and the CPU chunk, at the real head geometry
    /// (24/2 × 256, indexer 4 × 128); the chunk's selection equals the
    /// decode's per query; a pooled row computes the slab's bits.
    #[cfg(feature = "metal")]
    #[test]
    fn chunk_prefill_past_the_dense_range_metal() {
        let (cpu, gpu) = (Device::Cpu, Device::new_metal(0).unwrap());
        let cfg = kv_arm_cfg();
        let (a_gpu, a_cpu) = (indexed_attn(&cfg, &gpu), indexed_attn(&cfg, &cpu));
        let h = cfg.hidden_size;
        let limit = a_gpu.dense_limit().unwrap();
        let (n, cuts) = (64usize, [5usize, 43]);
        let xs = inputs(h, n, 0.37);
        let mut kv_d = KvCache::with_capacity(16);
        let (want, want_sel) = decode_steps(&a_gpu, &xs, 0, &mut kv_d, &gpu);
        let run = |attn: &QsaAttention, kv: &mut KvCache, dev: &Device| -> Vec<Vec<f32>> {
            let mut out = decode_steps(attn, &xs[..cuts[0]], 0, kv, dev).0;
            out.extend(prefill_chunk(attn, &xs[cuts[0]..cuts[1]], kv, dev));
            out.extend(prefill_chunk(attn, &xs[cuts[1]..], kv, dev));
            out
        };
        let mut kv_g = KvCache::with_capacity(16);
        let got = run(&a_gpu, &mut kv_g, &gpu);
        let mut kv_c = KvCache::with_capacity(16);
        let got_cpu = run(&a_cpu, &mut kv_c, &cpu);
        assert_eq!(kv_g.lens(), kv_d.lens());
        let scale = want.iter().flatten().fold(0f32, |m, x| m.max(x.abs()));
        let worst = |a: &[Vec<f32>], b: &[Vec<f32>]| {
            (0..n)
                .map(|p| max_diff(&a[p], &b[p]))
                .fold(0f32, |m, d| if d.is_nan() { d } else { m.max(d) })
        };
        let (e_replay, e_cpu) = (worst(&got, &want), worst(&got, &got_cpu));
        println!("metal chunk vs metal replay {e_replay:e}, vs cpu chunk {e_cpu:e}, scale {scale}");
        assert!(e_replay <= 1e-3 * scale, "vs replay: {e_replay}");
        assert!(e_cpu <= 1e-3 * scale, "vs cpu: {e_cpu}");
        let rows = chunk_selection(&a_gpu, &xs[limit..cuts[1]], limit, &kv_g, &gpu);
        for (t, g) in rows.iter().enumerate() {
            let p = limit + t;
            let (mut gs, mut ws) = (g.clone(), want_sel[p].clone().expect("sparse"));
            gs.sort_unstable();
            ws.sort_unstable();
            assert_eq!(
                gs, ws,
                "pos {p}: the chunk's selection vs decode's on Metal"
            );
        }
        let (mut kv_p, _other) = pooled_row_off_page_zero(&cfg, &gpu);
        let pooled = run(&a_gpu, &mut kv_p, &gpu);
        for p in 0..n {
            let same = pooled[p]
                .iter()
                .zip(&got[p])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "pos {p}: the pooled row is not the slab's bits on Metal"
            );
        }
    }

    /// At the model's own indexer (budget 2048, ratio 4: dense range 2051):
    /// a 2000-position prefix prefilled, then one chunk crossing the range
    /// (each sparse row's 2048–2051 cells take the two-pass arm), against
    /// the same prefix followed by decode steps. Outputs within fp32
    /// reordering; the selection the same set per query.
    #[cfg(feature = "metal")]
    #[test]
    fn chunk_prefill_crosses_the_real_dense_range_on_metal() {
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let attn = indexed_attn(&cfg, &gpu);
        let h = cfg.hidden_size;
        let limit = attn.dense_limit().unwrap();
        assert_eq!(limit, 2051);
        let (pre, n) = (2000usize, 2302usize);
        let xs = inputs(h, n, 0.013);
        let prefix = |kv: &mut KvCache| {
            for c in xs[..pre].chunks(500) {
                prefill_chunk(&attn, c, kv, &gpu);
            }
        };
        let mut kv_d = KvCache::with_capacity(1024);
        prefix(&mut kv_d);
        let (want, want_sel) = decode_steps(&attn, &xs[pre..], pre, &mut kv_d, &gpu);
        let mut kv_c = KvCache::with_capacity(1024);
        prefix(&mut kv_c);
        let got = prefill_chunk(&attn, &xs[pre..], &mut kv_c, &gpu);
        assert_eq!(kv_c.lens(), kv_d.lens());
        let scale = want.iter().flatten().fold(0f32, |m, x| m.max(x.abs()));
        let err = (0..n - pre)
            .map(|t| max_diff(&got[t], &want[t]))
            .fold(0f32, |m, d| if d.is_nan() { d } else { m.max(d) });
        println!("real indexer: chunk vs replay {err:e} at scale {scale}");
        assert!(
            err <= 1e-3 * scale,
            "chunk vs replay: {err} at scale {scale}"
        );
        let rows = chunk_selection(&attn, &xs[limit..], limit, &kv_c, &gpu);
        let mut differ = Vec::new();
        for (t, g) in rows.iter().enumerate() {
            let p = limit + t;
            assert!(
                g.len() >= 2048 && g.len() <= 2051,
                "pos {p}: {} cells",
                g.len()
            );
            let (mut gs, mut ws) = (g.clone(), want_sel[p - pre].clone().expect("sparse"));
            gs.sort_unstable();
            ws.sort_unstable();
            if gs != ws {
                differ.push(p);
            }
        }
        assert!(
            differ.is_empty(),
            "queries whose selection differs: {differ:?}"
        );
    }

    /// Each KV store's chunk past the dense range against its own decode
    /// replay: the f16 store's cells and the quantized stores' dequantized
    /// cells feed the paged attention, where decode reads the f16 gather
    /// and the codes. Within fp32 reordering of the same values — far under
    /// the gap a store that attended every cell would show.
    #[cfg(feature = "metal")]
    #[test]
    fn chunk_prefill_past_the_dense_range_on_every_store() {
        use crate::nn::ops::KvQuant;
        let gpu = Device::new_metal(0).unwrap();
        let cfg = kv_arm_cfg();
        let attn = indexed_attn(&cfg, &gpu);
        let h = cfg.hidden_size;
        let (n, cuts) = (56usize, [3usize, 37]);
        let xs = inputs(h, n, 0.21);
        let limit = attn.dense_limit().unwrap();
        let (sel32, _) = decode_steps(&attn, &xs, 0, &mut KvCache::with_capacity(16), &gpu);
        let (all32, _) = decode_steps(
            &dense_twin(&cfg, &gpu),
            &xs,
            0,
            &mut KvCache::with_capacity(16),
            &gpu,
        );
        let d_sel = (limit..n)
            .map(|q| max_diff(&all32[q], &sel32[q]))
            .fold(0f32, f32::max);
        let arms = [
            ("f16", KvPrecision::dense(DType::F16)),
            ("int8", KvPrecision::quantized(KvQuant::Int8).unwrap()),
            ("int4", KvPrecision::quantized(KvQuant::Int4).unwrap()),
        ];
        for (name, p) in arms {
            let mut kv_d = KvCache::with_precision(16, p).unwrap();
            let (want, _) = decode_steps(&attn, &xs, 0, &mut kv_d, &gpu);
            let mut kv_c = KvCache::with_precision(16, p).unwrap();
            let mut got = decode_steps(&attn, &xs[..cuts[0]], 0, &mut kv_c, &gpu).0;
            got.extend(prefill_chunk(&attn, &xs[cuts[0]..cuts[1]], &mut kv_c, &gpu));
            got.extend(prefill_chunk(&attn, &xs[cuts[1]..], &mut kv_c, &gpu));
            assert_eq!(kv_c.lens(), kv_d.lens(), "{name}");
            let scale = want.iter().flatten().fold(0f32, |m, x| m.max(x.abs()));
            let err = (0..n)
                .map(|q| max_diff(&got[q], &want[q]))
                .fold(0f32, |m, d| if d.is_nan() { d } else { m.max(d) });
            println!("{name}: chunk vs its decode replay {err:e}, dense-over-all {d_sel:e}, scale {scale}");
            assert!(
                err <= 1e-3 * scale,
                "{name}: chunk vs replay {err} at scale {scale}"
            );
            assert!(
                d_sel > 10.0 * 1e-3 * scale,
                "the control is blind ({d_sel})"
            );
        }
    }

    /// The A/B arm (`QWEN4EXP_SPARSE_MASKED`) computes what the shipping arm
    /// does: the same queries over the same cache, one masked dense attention
    /// over every cell against the paged kernel over each row's cells, within
    /// fp32 reordering — so a timing between the two compares one result.
    /// At the real head geometry, rows past the dense range in two tiles.
    #[cfg(feature = "metal")]
    #[test]
    fn the_masked_arm_attends_as_the_paged_arm() {
        let gpu = Device::new_metal(0).unwrap();
        let cfg = kv_arm_cfg();
        let attn = indexed_attn(&cfg, &gpu);
        let (h, hd, nh) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads);
        let n = 70usize;
        let xs = inputs(h, n, 0.61);
        let mut kv = KvCache::with_capacity(16);
        decode_steps(&attn, &xs, 0, &mut kv, &gpu);
        let pos0 = attn.dense_limit().unwrap() + 3;
        let s = n - pos0;
        let x = Tensor::from_vec(xs[pos0..].concat(), (s, h), &gpu).unwrap();
        let q = Tensor::from_vec(
            (0..s * nh * hd)
                .map(|i| ((i as f32) * 0.0137).sin())
                .collect::<Vec<f32>>(),
            (s, nh, hd),
            &gpu,
        )
        .unwrap();
        let scale = 1.0 / (hd as f32).sqrt();
        let ix = attn.indexer().unwrap();
        let tile = s.div_ceil(2);
        let paged = host(&attn.sparse_rows(&x, &q, pos0, &mut kv, scale).unwrap());
        let masked = host(
            &attn
                .sparse_rows_masked(ix, &x, &q, pos0, &mut kv, scale, tile)
                .unwrap(),
        );
        let sc = paged.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = max_diff(&masked, &paged);
        println!("masked vs paged {err:e} at scale {sc}");
        assert!(err <= 1e-5 * sc, "masked vs paged: {err} at scale {sc}");
        assert!(paged.iter().any(|v| v.abs() > 1e-3), "a degenerate output");
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

    /// The real attention geometry (24 / 2 heads × 256, indexer 4 × 128) at a
    /// narrow hidden size, with a budget-16 indexer: every position past 19
    /// cells reads a selection.
    #[cfg(feature = "metal")]
    fn kv_arm_cfg() -> Config {
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        cfg.indexer_budget = 16;
        cfg
    }

    #[cfg(feature = "metal")]
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[cfg(feature = "metal")]
    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / scale
    }

    /// Decode `steps` positions through `attn` into a cache at `precision`
    /// (initial capacity 16, so the store grows twice), every output read back.
    #[cfg(feature = "metal")]
    fn decode_arm(
        attn: &QsaAttention,
        precision: KvPrecision,
        dev: &Device,
        steps: usize,
    ) -> (Vec<Vec<f32>>, KvCache) {
        let mut kv = KvCache::with_precision(16, precision).unwrap();
        let outs = (0..steps)
            .map(|pos| {
                let x = Tensor::from_vec(tok(128, pos as f32 * 0.37), (1, 1, 128), dev).unwrap();
                attn.decode(&x, &mut kv)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();
        (outs, kv)
    }

    /// Each KV store decodes within its own error of the f32 store, on the
    /// dense range AND past it, where the selection is read (the f16 store
    /// rounds, int8 / int4 read their codes on the dense and the selected
    /// kernel). The control that pins the level: past the dense range the
    /// f32 selection differs from the f32 dense-over-everything by `d_sel`,
    /// and a store that attended every cell instead of the selection would be
    /// that far off — so each arm's error is asserted well under it.
    /// And the stores are what they say: f16 elements, quantized planes.
    #[cfg(feature = "metal")]
    #[test]
    fn each_kv_store_decodes_within_its_error_on_the_dense_and_the_selected_path() {
        use crate::nn::ops::KvQuant;
        let gpu = Device::new_metal(0).unwrap();
        let cfg = kv_arm_cfg();
        let attn = indexed_attn(&cfg, &gpu);
        let dense_twin = {
            let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
            QsaAttention::new(
                lin(nh * hd * 2, h, 1.0, &gpu),
                lin(nkv * hd, h, 2.0, &gpu),
                lin(nkv * hd, h, 3.0, &gpu),
                lin(h, nh * hd, 4.0, &gpu),
                rms(hd, &gpu),
                rms(hd, &gpu),
                &cfg,
            )
        };
        let steps = 48;
        let limit = attn.dense_limit().unwrap();
        let (f32_out, _) = decode_arm(&attn, DEFAULT_KV_PRECISION, &gpu, steps);
        let (all_out, _) = decode_arm(&dense_twin, DEFAULT_KV_PRECISION, &gpu, steps);
        // Errors are taken against the RUN's output scale: this synthetic
        // layer's output nearly cancels at some positions (max |out| 0.02
        // against ~1 elsewhere), where a per-position ratio reads a small
        // absolute error as tens of percent.
        let run_scale = f32_out.iter().flatten().fold(0f32, |m, x| m.max(x.abs()));
        let d_sel = (limit..steps)
            .map(|p| max_abs(&all_out[p], &f32_out[p]))
            .fold(0f32, f32::max)
            / run_scale;
        assert!(
            d_sel > 1e-2,
            "the selection barely moves the output ({d_sel}): the control is blind"
        );
        let arms = [
            ("f16", KvPrecision::dense(DType::F16)),
            ("int8", KvPrecision::quantized(KvQuant::Int8).unwrap()),
            ("int4", KvPrecision::quantized(KvQuant::Int4).unwrap()),
        ];
        let mut errs = Vec::new();
        for (name, p) in arms {
            let (out, kv) = decode_arm(&attn, p, &gpu, steps);
            assert_eq!(kv.len(), steps);
            match p.dense_dtype() {
                Some(dt) => {
                    let [k, v, _, _] = kv.state_views().unwrap();
                    assert_eq!(k.unwrap().dtype(), dt, "{name}: k is not stored at {dt:?}");
                    assert_eq!(v.unwrap().dtype(), dt, "{name}: v is not stored at {dt:?}");
                }
                None => assert!(
                    matches!(kv.kv, KvBacking::Quant(_)),
                    "{name}: not a quantized store"
                ),
            }
            let dense_err = (0..limit)
                .map(|p| max_abs(&out[p], &f32_out[p]))
                .fold(0f32, f32::max)
                / run_scale;
            let sel_err = (limit..steps)
                .map(|p| max_abs(&out[p], &f32_out[p]))
                .fold(0f32, f32::max)
                / run_scale;
            eprintln!("KVARM {name}: max err / run scale: dense range {dense_err:.3e}, selected range {sel_err:.3e}; the selection's own effect {d_sel:.3e}");
            assert!(
                out.iter().flatten().all(|x| x.is_finite()),
                "{name}: non-finite"
            );
            assert!(
                dense_err > 0.0,
                "{name}: bit-identical to f32 — the store was not engaged"
            );
            assert!(sel_err < d_sel / 4.0, "{name}: selected-range error {sel_err} is not well under the selection's own effect {d_sel}");
            errs.push((dense_err.max(sel_err), name));
        }
        assert!(
            errs[0].0 < errs[1].0 && errs[1].0 < errs[2].0,
            "errors not ordered f16 < int8 < int4: {errs:?}"
        );
        // Measured 9.5e-5 / 2.2e-3 / 1.7e-2 of the run's scale; the bounds
        // are a decade above, and each a decade under the next rung.
        assert!(
            errs[0].0 < 1e-3,
            "f16 error {} is above what rounding K/V to f16 explains",
            errs[0].0
        );
        assert!(
            errs[1].0 < 1e-2,
            "int8 error {} is above what 8-bit codes explain",
            errs[1].0
        );
        assert!(
            errs[2].0 < 1e-1,
            "int4 error {} is above what 4-bit codes explain",
            errs[2].0
        );
    }

    /// The quantized block's reads against dense attention over its own
    /// dequantized planes — same codes, so quantization cancels and only the
    /// kernels differ: over every live cell, and over a scattered selection
    /// (blocks out of order and a tail). The planes have grown twice from a
    /// 16-token start, so the widened head strides are what is read.
    #[cfg(feature = "metal")]
    #[test]
    fn the_quantized_block_reads_its_cells_as_dense_attention_over_them() {
        use crate::nn::ops::KvQuant;
        let gpu = Device::new_metal(0).unwrap();
        let (nh, nkv, hd, n) = (24usize, 2usize, 256usize, 41usize);
        for q_arm in [KvQuant::Int8, KvQuant::Int4] {
            let mut kv =
                KvCache::with_precision(16, KvPrecision::quantized(q_arm).unwrap()).unwrap();
            for t in 0..n {
                let k: Vec<f32> = (0..nkv * hd)
                    .map(|i| ((i as f32 * 0.021 + t as f32 * 0.7).sin()) * (1.0 + t as f32 * 0.03))
                    .collect();
                let v: Vec<f32> = (0..nkv * hd)
                    .map(|i| ((i as f32 * 0.017 - t as f32 * 0.4).cos()) * 0.8 - 0.1 * t as f32)
                    .collect();
                kv.append(
                    &Tensor::from_vec(k, (nkv, 1, hd), &gpu).unwrap(),
                    &Tensor::from_vec(v, (nkv, 1, hd), &gpu).unwrap(),
                )
                .unwrap();
            }
            let q = Tensor::from_vec(
                (0..nh * hd)
                    .map(|i| ((i as f32 * 0.013).sin()) * 0.5)
                    .collect::<Vec<f32>>(),
                (1, nh, 1, hd),
                &gpu,
            )
            .unwrap();
            let scale = 1.0 / (hd as f32).sqrt();
            let cells: Vec<u32> = [8u32, 9, 10, 11, 0, 1, 2, 3, 28, 29, 30, 31, 40].to_vec();
            let sel = Tensor::from_vec(cells.clone(), cells.len(), &gpu).unwrap();
            for s in [None, Some(&sel)] {
                let got = kv
                    .attend(&q, s, scale)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let (k, v) = kv.dense(s).unwrap();
                let want = ops::sdpa(
                    &q,
                    &k.unsqueeze(0).unwrap(),
                    &v.unsqueeze(0).unwrap(),
                    None,
                    false,
                    scale,
                    1.0,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                let e = max_rel(&got, &want);
                eprintln!("QREAD {q_arm:?} sel={} max rel {e:.3e}", s.is_some());
                assert!(
                    e < 1e-5,
                    "{q_arm:?} sel={}: codes read {e} off the dequantized dense attention",
                    s.is_some()
                );
            }
            let all = kv
                .attend(&q, None, scale)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let some = kv
                .attend(&q, Some(&sel), scale)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                max_rel(&some, &all) > 1e-3,
                "{q_arm:?}: the selection did not change the read"
            );
        }
    }

    /// A speculative rewind on each store: decode, keep the lengths, decode
    /// five more, truncate back, decode the same five again — bit for bit the
    /// first five (the codes above the length are unreachable, the planes
    /// are rewritten in place).
    #[cfg(feature = "metal")]
    #[test]
    fn each_kv_store_rewinds_to_a_bit_identical_replay() {
        use crate::nn::ops::KvQuant;
        let gpu = Device::new_metal(0).unwrap();
        let cfg = kv_arm_cfg();
        let attn = indexed_attn(&cfg, &gpu);
        for p in [
            KvPrecision::dense(DType::F16),
            KvPrecision::quantized(KvQuant::Int8).unwrap(),
            KvPrecision::quantized(KvQuant::Int4).unwrap(),
        ] {
            let (_, mut kv) = decode_arm(&attn, p, &gpu, 30);
            let lens = kv.lens();
            let step = |kv: &mut KvCache, t: usize| -> Vec<u32> {
                let x = Tensor::from_vec(tok(128, 50.0 + t as f32), (1, 1, 128), &gpu).unwrap();
                attn.decode(&x, kv)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            };
            let first: Vec<Vec<u32>> = (0..5).map(|t| step(&mut kv, t)).collect();
            kv.truncate(lens.0, lens.1);
            assert_eq!(kv.lens(), lens);
            let again: Vec<Vec<u32>> = (0..5).map(|t| step(&mut kv, t)).collect();
            assert_eq!(first, again, "{p:?}: the replay after a rewind differs");
        }
    }

    /// The KV class's memory per store at 4k and 32k positions, measured: 12
    /// full-attention layers at the real geometry (2 heads × 256, the
    /// indexer's 128-wide raw and compressed keys), filled through the real
    /// append path in 512-position chunks. Prints the planes' bytes (exact) and
    /// the device / footprint deltas (the buffer pool's view), and asserts
    /// the planes equal `loader::kv_bytes_per_token` at the store — the
    /// admission's arithmetic, checked against what was allocated.
    /// `cargo test … kv_class_bytes -- --ignored --nocapture`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "a measurement: 12 layers × 32k positions per arm"]
    fn kv_class_bytes_per_store_at_4k_and_32k() {
        use crate::nn::memory::MemSnapshot;
        use crate::nn::ops::KvQuant;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let ix = Indexer::new(
            lin(cfg.indexer_n_heads * cfg.indexer_head_dim, 128, 5.0, &gpu),
            lin(cfg.indexer_head_dim, 128, 6.0, &gpu),
            rms(cfg.indexer_head_dim, &gpu),
            rms(cfg.indexer_head_dim, &gpu),
            &cfg,
        )
        .unwrap();
        let (nkv, hd, idx) = (cfg.n_kv_heads, cfg.head_dim, cfg.indexer_head_dim);
        let chunk = 512usize;
        let k: Vec<f32> = (0..nkv * chunk * hd)
            .map(|i| ((i as f32) * 0.003).sin())
            .collect();
        let k = Tensor::from_vec(k, (nkv, chunk, hd), &gpu).unwrap();
        let v = (&k * 0.5).unwrap();
        let raw = Tensor::from_vec(
            (0..chunk * idx)
                .map(|i| ((i as f32) * 0.007).cos())
                .collect::<Vec<f32>>(),
            (chunk, idx),
            &gpu,
        )
        .unwrap();
        for tokens in [4096usize, 32768] {
            for p in [
                DEFAULT_KV_PRECISION,
                KvPrecision::dense(DType::F16),
                KvPrecision::quantized(KvQuant::Int8).unwrap(),
                KvPrecision::quantized(KvQuant::Int4).unwrap(),
            ] {
                cfg.kv_precision = p;
                gpu.synchronize().unwrap();
                let before = MemSnapshot::read(&gpu);
                let mut caches: Vec<KvCache> = (0..12)
                    .map(|_| KvCache::with_precision(KV_SLAB_TOKENS, p).unwrap())
                    .collect();
                for c in caches.iter_mut() {
                    for _ in 0..tokens / chunk {
                        c.append(&k, &v).unwrap();
                        ix.observe_keys(&raw, c).unwrap();
                    }
                }
                gpu.synchronize().unwrap();
                let after = MemSnapshot::read(&gpu);
                let planes: usize = caches
                    .iter()
                    .map(|c| {
                        let kv = match &c.kv {
                            KvBacking::Slab(s) => [s.k_cache().all_data(), s.v_cache().all_data()]
                                .into_iter()
                                .flatten()
                                .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                                .sum(),
                            KvBacking::Quant(s) => s.bytes(),
                            KvBacking::Pooled { .. } => unreachable!(),
                        };
                        let ixb: usize = [c.raw_keys.all_data(), c.blocks.all_data()]
                            .into_iter()
                            .flatten()
                            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
                            .sum();
                        kv + ixb
                    })
                    .sum();
                let want = tokens * super::super::loader::kv_bytes_per_token(&cfg);
                let d = |a: u64, b: u64| a as f64 / 1e9 - b as f64 / 1e9;
                println!(
                    "KVBYTES store={p:?} tokens={tokens} planes_bytes={planes} planes_gb={:.4} per_token={} device_delta_gb={:.4} pool_live_delta_gb={:.4} phys_delta_gb={:.4}",
                    planes as f64 / 1e9,
                    planes / tokens,
                    d(after.device_allocated, before.device_allocated),
                    d(after.pool_live, before.pool_live),
                    d(after.phys_footprint, before.phys_footprint),
                );
                assert_eq!(
                    planes, want,
                    "{p:?} at {tokens}: the planes are not the admission's arithmetic"
                );
                drop(caches);
                gpu.synchronize().unwrap();
            }
        }
    }
}
