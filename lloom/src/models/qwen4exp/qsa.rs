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
    /// The k / v tokens: this row's own slab, or a slot of a batch's page
    /// pool (`KvPool`, §12.2).
    kv: KvBacking,
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

/// Tokens a KV slab holds before it grows by another slab.
pub const KV_SLAB_TOKENS: usize = 1024;

/// KV tokens per page of a [`KvPool`] (LFM2's `flash_page_size`).
pub const KV_PAGE_TOKENS: usize = 256;

impl KvCache {
    pub fn with_capacity(tokens: usize) -> Self {
        let tokens = tokens.max(1);
        Self {
            kv: KvBacking::Slab(crate::nn::kv_cache::KvCache::new(1, tokens)),
            raw_keys: crate::nn::kv_cache::Cache::new(1, tokens),
            blocks: crate::nn::kv_cache::Cache::new(1, tokens.div_ceil(4).max(1)),
        }
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
            KvBacking::Slab(_) => None,
        }
    }
    pub fn is_pooled(&self) -> bool {
        matches!(self.kv, KvBacking::Pooled { .. })
    }
    pub fn reset(&mut self) {
        match &mut self.kv {
            KvBacking::Slab(s) => s.reset(),
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
    /// (`[nkv, tokens, hd]` each, `elem` bytes an element): a slab widened by
    /// `KV_SLAB_TOKENS` steps (the widening is a `cat`, so the old slab is
    /// live beside the new one at the instant — counted), or the pool's
    /// next page slab when its free list runs out. 0 when nothing grows.
    /// Sized, not performed (`Qwen4Exp::plan_pass`).
    pub fn grow_bytes_for(&self, n: usize, nkv: usize, hd: usize, elem: usize) -> usize {
        let per_token = 2 * nkv * hd * elem;
        match &self.kv {
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

    pub fn len(&self) -> usize {
        match &self.kv {
            KvBacking::Slab(s) => s.current_seq_len(),
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
            KvBacking::Pooled {
                pool, slot, layer, ..
            } => pool.truncate(*layer, *slot, tokens),
        }
        self.raw_keys.truncate(tokens);
        self.blocks.truncate(blocks);
    }
    /// The live `(k, v, raw indexer keys, compressed keys)` views, for a
    /// state probe (`None` where nothing was written).
    pub fn state_views(&self) -> Result<[Option<Tensor>; 4]> {
        let (k, v) = match &self.kv {
            KvBacking::Slab(s) => (s.k()?, s.v()?),
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
    /// Append `k`, `v` `[n_kv_heads, n, head_dim]`; returns views of the whole
    /// cache `[n_kv_heads, len, head_dim]` (a pooled row's: gathered from its
    /// pages — the per-row paths; the pooled rows arm appends without them,
    /// [`Self::append_pooled`]).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match &mut self.kv {
            KvBacking::Slab(s) => s.append(k, v),
            KvBacking::Pooled { .. } => {
                self.append_pooled(k, v)?;
                let (pool, slot, layer) = self.pool_slot().expect("pooled");
                match pool.live(layer, slot)? {
                    Some(kv) => Ok(kv),
                    None => lloom_tensor::bail!("qsa: an empty pooled row after an append"),
                }
            }
        }
    }
    /// Append one token to a pooled row without writing it: the pool token
    /// slot the rows arm scatters it to.
    fn append_slot(&mut self) -> Result<u32> {
        match &mut self.kv {
            KvBacking::Slab(_) => lloom_tensor::bail!("qsa: append_slot on a slab cache"),
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
            KvBacking::Slab(_) => lloom_tensor::bail!("qsa: append_pooled on a slab cache"),
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
    /// The rows `sel [n_sel]` (u32, all `< len`) of k and v:
    /// `[n_kv_heads, n_sel, head_dim]` each. Gathers from the slab itself
    /// (contiguous), so the cost is the selection's size, not the context's;
    /// a pooled row maps the cells through its page table first.
    fn gather(&mut self, sel: &Tensor) -> Result<(Tensor, Tensor)> {
        match &mut self.kv {
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
    /// A page table per slot, shared by the layers.
    pages: lloom_tensor::kv_pages::KvPagePool,
    /// Pages released by a cleared slot, held until the GPU's epoch that
    /// could read them has completed.
    quarantine: lloom_tensor::kv_pages::PageQuarantine,
    #[cfg(feature = "metal")]
    clock: Option<std::sync::Arc<lloom_tensor::kernels::metal::GpuClock>>,
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
        #[cfg(feature = "metal")]
        let clock = match device {
            lloom_tensor::Device::Metal(d) => Some(d.gpu_clock()),
            _ => None,
        };
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
                pages: lloom_tensor::kv_pages::KvPagePool::new(
                    page_size,
                    0,
                    slots,
                    u32::MAX as usize,
                ),
                quarantine: lloom_tensor::kv_pages::PageQuarantine::new(),
                #[cfg(feature = "metal")]
                clock,
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
            .map(|s| s.pages().to_vec())
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
                .map(|t| pages[t / ps] * ps as u32 + (t % ps) as u32)
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
                .map(|t| pages[t / ps] * ps as u32 + (t % ps) as u32)
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
    /// The GPU's current epoch (Metal), the epoch a released page waits for.
    fn epoch(&self) -> u64 {
        #[cfg(feature = "metal")]
        {
            self.clock.as_ref().map(|c| c.current_epoch()).unwrap_or(0)
        }
        #[cfg(not(feature = "metal"))]
        {
            0
        }
    }

    /// Epochs the GPU has completed (every one, without a clock).
    fn completed(&self) -> u64 {
        #[cfg(feature = "metal")]
        {
            self.clock.as_ref().map(|c| c.completed_count()).unwrap_or(u64::MAX)
        }
        #[cfg(not(feature = "metal"))]
        {
            u64::MAX
        }
    }

    fn recycle_quarantined(&mut self) {
        let ready = self.quarantine.drain_completed(self.completed());
        if !ready.is_empty() {
            self.pages.recycle(&ready).expect("quarantined pages are unallocated");
        }
    }

    fn release_pages(&mut self, slot: usize) {
        let freed = self.pages.release(slot);
        let epoch = self.epoch();
        self.quarantine.quarantine(epoch, freed);
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
        let (cos, sin) = rope_cs_at(self.n_rot, self.rope_theta, &positions, k_raw.device())?;
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
            || !matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
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
            let (k_all, v_all) = kv.append(&k_step, &v)?;
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
            let (k_all, v_all) = match &sel {
                Some(sel) => kv.gather(sel)?,
                None => (k_all, v_all),
            };
            let ctx = ops::sdpa(
                &q.reshape((1, nh, 1, hd))?,
                &k_all.unsqueeze(0)?,
                &v_all.unsqueeze(0)?,
                None,
                false,
                scale as f32,
                1.0,
            )?;
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
            check_sel(sel, pos, kv)?;
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
            || !matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
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
            let (k_all, v_all) = kv.append(&k_step, &v)?; // [nkv, seq, hd] slab views
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
            let (k_all, v_all) = match &sel {
                Some(sel) => kv.gather(sel)?,
                None => (k_all, v_all),
            };
            let ctx = ops::sdpa(
                &q.reshape((1, nh, 1, hd))?,
                &k_all.unsqueeze(0)?,
                &v_all.unsqueeze(0)?,
                None,
                false,
                scale as f32,
                1.0, // no softcapping (the kernel divides by it)
            )?; // [1, nh, 1, hd]
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
    /// → `[nd, nh, hd]`. The rows split at `ops::SDPA_TWO_PASS_K_THRESHOLD`
    /// keys — the single-pass rows in one launch, the 2-pass rows in one
    /// pair — so every row runs the kernel the per-row `ops::sdpa` runs at
    /// its length: bit-identical per row.
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
        let long = |j: usize| n_keys[j] as usize >= ops::SDPA_TWO_PASS_K_THRESHOLD;
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

    /// A whole chunk `x [1, seq, hidden]` → `[1, seq, hidden]`: the same dense
    /// QK-norm GQA composition as [`Self::decode`] over every position at once,
    /// causal over the chunk, attending to everything already cached. Valid
    /// while the cache stays within the indexer's dense range.
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
            || !matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256)
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
        let (k_all, v_all) = kv.append(&k_new, &v_new)?; // [nkv, kv_len, hd] slab views
        let kv_len = k_all.dim(1)?;
        if let Some(ix) = &self.indexer {
            if kv_len > ix.dense_limit() {
                lloom_tensor::bail!(
                    "qsa prefill: {kv_len} cells exceed the dense range {}; decode past it",
                    ix.dense_limit()
                )
            }
            ix.observe(&x2, kv)?;
        }
        // The steel full-attention kernel over the cache as it lies: GQA and
        // the causal window against the cached prefix in the kernel (query
        // t at position past + t sees keys ≤ past + t: the kernel's window
        // is offset by kv_len − seq), no mask tensor, no head expansion.
        let scale = 1.0 / (hd as f64).sqrt();
        let ctx = ops::sdpa(
            &q_h,
            &k_all.unsqueeze(0)?,
            &v_all.unsqueeze(0)?,
            None,
            true,
            scale as f32,
            1.0,
        )?; // [1, nh, seq, hd]
        let ctx = ctx
            .reshape((nh, seq, hd))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq, nh * hd))?;
        let gated = match sigmoid_mul(&gate, &ctx) {
            Some(g) => g?,
            None => (ctx * ops::sigmoid(&gate)?)?,
        };
        Ok(Some(self.wo.forward(&gated)?.reshape((1, seq, hidden))?))
    }
}

/// A/B instrument (`QWEN4EXP_SCORE_NO_MATMUL`, read once): the block score
/// as a broadcast product + reduce instead of the `[n_blocks, 128] × [128, 4]`
/// matmul. Never the shipping path.
fn score_without_matmul() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_SCORE_NO_MATMUL").is_some())
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
            let (k_all, _) = kv.append(&tok(t), &(tok(t) + 0.5)?)?;
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
        let (k_all, _) = kv.append(&tok(300), &(tok(300) + 0.5)?)?;
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
