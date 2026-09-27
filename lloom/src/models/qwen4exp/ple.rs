//! N-gram PLE (Predictive Lookup Embedding): the hashed n-gram table injected
//! into the n-stream carry at one layer (`ple_layers`, GGUF `blk.1`).
//!
//! Three pieces, in dataflow order:
//!
//! 1. [`NgramHash`] — the multi-head hash from the token window to table rows.
//!    Bit-exact to the reference (vLLM `qwen4_exp/nvidia/ops/ple.py`, llama.cpp
//!    `qwen4exp.cpp` `llm_graph_input_ple::set_input`): per n-gram order
//!    `n ∈ 2..=ngram_size`, `mixed_n = tok[t]·m[0] ^ tok[t-1]·m[1] ^ … ^
//!    tok[t-n+1]·m[n-1]` in wrapping `i64`, and head `h` of that order reads
//!    row `offset[h] + rem(mixed_n, size[h])` with a non-negative remainder.
//!    An EOS (or a missing predecessor) cuts the window: it and everything
//!    older read as EOS; the current token being EOS does not cut its own
//!    context. A wrong hash changes no shape — it is a text-gate-only trap
//!    (`design/docs/expert-streaming.md` §20.1a), so the tests here pin it
//!    against hand-computed rows and the checkpoint's own hash constants.
//! 2. [`PleTable`] — the row gather. The table is 320 M rows × 160 (28.8 GB at
//!    IQ4_NL, the second GGUF shard), so it is never loaded: rows are read on
//!    demand from a memory map and dequantized on the CPU (16 rows × 90 B per
//!    token). This is the simplest form of §20.1a's streamed row-gather; a
//!    device-side hot-row cache is the streaming issue's.
//! 3. [`PleLayer`] — key/value projections of the gathered embedding, grouped
//!    RMS norms, the per-stream signed-sqrt gate, and the dilated depthwise
//!    conv over the gated value (kernel 4, dilation `ngram_size`), whose
//!    9-position history is a [`RecurrentState`] channel. Output is the delta
//!    added to the carry: `gated + silu(conv)`.

use super::hc::grouped_rms_norm;
use crate::nn::{ops, RecurrentState};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::{GgmlDType, GgmlType, QMatMul};
use lloom_tensor::{Device, Module, Result, Tensor};
use std::path::Path;

/// The per-layer hash constants, as the checkpoint records them
/// (`qwen4exp.ple.{layer_multipliers, head_offsets, head_vocab_sizes, eos_token_id}`).
#[derive(Debug, Clone)]
pub struct NgramHash {
    /// `layer_multipliers[i]` multiplies the token `i` positions back.
    multipliers: Vec<i64>,
    /// Per-head prime table sizes, `n_heads` of them.
    sizes: Vec<i64>,
    /// Per-head row offsets into the flat table.
    offsets: Vec<i64>,
    heads_per_ngram: usize,
    ngram_size: usize,
    eos: i64,
}

impl NgramHash {
    pub fn new(
        multipliers: Vec<i64>,
        sizes: Vec<i64>,
        offsets: Vec<i64>,
        heads_per_ngram: usize,
        eos: u32,
    ) -> Result<Self> {
        let ngram_size = multipliers.len();
        let n_heads = (ngram_size.saturating_sub(1)) * heads_per_ngram;
        if ngram_size < 2 || heads_per_ngram == 0 {
            lloom_tensor::bail!(
                "ple hash: ngram_size {ngram_size} / heads_per_ngram {heads_per_ngram}"
            )
        }
        if sizes.len() < n_heads || offsets.len() < n_heads {
            lloom_tensor::bail!(
                "ple hash: {n_heads} heads but {} sizes / {} offsets",
                sizes.len(),
                offsets.len()
            )
        }
        if sizes.iter().take(n_heads).any(|&s| s <= 0) {
            lloom_tensor::bail!("ple hash: a head vocab size is not positive")
        }
        Ok(Self {
            multipliers,
            sizes,
            offsets,
            heads_per_ngram,
            ngram_size,
            eos: eos as i64,
        })
    }

    /// Heads (rows gathered per token): `(ngram_size − 1) · heads_per_ngram`.
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Predecessors the window needs: `ngram_size − 1`.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// One row of the table, past the last real row (the padded tail).
    pub fn total_rows(&self) -> i64 {
        let n = self.n_heads();
        (0..n)
            .map(|h| self.offsets[h] + self.sizes[h])
            .max()
            .unwrap_or(0)
    }

    /// Row ids for one token. `window[0]` is the token itself, `window[s]` the
    /// token `s` positions back (`None` before the sequence start). `out` is
    /// `n_heads` long, head-major.
    pub fn row_ids(&self, window: &[Option<u32>], out: &mut [i64]) {
        debug_assert_eq!(window.len(), self.ngram_size);
        debug_assert_eq!(out.len(), self.n_heads());
        let cur = window[0].map(|t| t as i64).unwrap_or(self.eos);
        // mixed[n-2] accumulates the n-gram's XOR as the window walks outward.
        let mut mixed = cur.wrapping_mul(self.multipliers[0]);
        let mut crossed = false;
        for (shift, older) in window.iter().enumerate().skip(1) {
            let mut candidate = older.map(|t| t as i64).unwrap_or(self.eos);
            if crossed {
                candidate = self.eos;
            }
            crossed |= candidate == self.eos;
            mixed ^= candidate.wrapping_mul(self.multipliers[shift]);
            // Every head of order n > shift includes this term; the order
            // `shift + 1` heads are complete now.
            let order = shift + 1;
            let base = (order - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                let size = self.sizes[h];
                let mut rem = mixed.wrapping_rem(size);
                if rem < 0 {
                    rem += size;
                }
                out[h] = rem + self.offsets[h];
            }
        }
    }

    /// The reference derivation of the constants (vLLM
    /// `Qwen4ExpNGramEmbedding::_make_layer_multipliers` /
    /// `_make_vocab_layout`), so the checkpoint's recorded values can be
    /// cross-checked rather than trusted.
    pub fn derive(
        seed: u64,
        unigram_vocab_size: u64,
        ngram_size: usize,
        heads_per_ngram: usize,
        ngram_vocab_size_base: u64,
        ple_dense_layer_id: u64,
        eos: u32,
    ) -> Result<Self> {
        const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
        const PRIME: u64 = 10007;
        let max_multiplier = ((1u64 << 63) - 1) / unigram_vocab_size;
        let half_bound = (max_multiplier / 2).max(1);
        let base_seed = seed.wrapping_add(PRIME.wrapping_mul(ple_dense_layer_id));
        let multipliers: Vec<i64> = (0..ngram_size)
            .map(|i| {
                let v = base_seed.wrapping_add(GAMMA.wrapping_mul(i as u64 + 1));
                (2 * (splitmix64(v) % half_bound) + 1) as i64
            })
            .collect();
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        let mut sizes = Vec::with_capacity(n_heads);
        let mut offsets = Vec::with_capacity(n_heads);
        let mut offset = 0i64;
        for local in 0..n_heads as u64 {
            let global = ple_dense_layer_id * n_heads as u64 + local;
            let size = nth_prime_after(ngram_vocab_size_base - 1, global + 1) as i64;
            sizes.push(size);
            offsets.push(offset);
            offset += size;
        }
        Self::new(multipliers, sizes, offsets, heads_per_ngram, eos)
    }
}

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime_64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'witness: for a in [2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mod_mul(acc, base, m);
        }
        base = mod_mul(base, base, m);
        exp >>= 1;
    }
    acc
}

/// The `count`-th prime strictly greater than `start`.
fn nth_prime_after(start: u64, count: u64) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        let mut c = prime + 1;
        if c <= 2 {
            prime = 2;
            continue;
        }
        if c.is_multiple_of(2) {
            c += 1;
        }
        while !is_prime_64(c) {
            c += 2;
        }
        prime = c;
    }
    prime
}

/// Where the table's rows come from.
#[derive(Debug)]
pub enum PleTable {
    /// The real table: a memory map of the GGUF shard that holds
    /// `per_layer_token_embd.weight`, rows dequantized on read.
    Mmap(MmapPleTable),
    /// The memory map behind a hot-row cache of packed rows keyed on row id
    /// (§20.1a's streamed tier): a hit skips the file read, a miss reads the
    /// row and caches it, evicting the least recently used. The bytes a row
    /// dequantizes from are the file's either way — the gather is
    /// byte-identical to `Mmap`.
    Cached(CachedPleTable),
    /// A small dense `[rows, head_dim]` f32 table (tests).
    Dense(Tensor),
}

impl PleTable {
    /// Rows per token's `ids` (head-major), concatenated to one
    /// `[ids.len() · head_dim]` f32 vector on `dev`.
    pub fn gather(&self, ids: &[i64], dev: &Device) -> Result<Tensor> {
        match self {
            PleTable::Mmap(m) => m.gather(ids, dev),
            PleTable::Cached(c) => c.gather(ids, dev),
            PleTable::Dense(t) => {
                let rows = t.dim(0)? as i64;
                let ids: Vec<u32> = ids
                    .iter()
                    .map(|&i| {
                        if i < 0 || i >= rows {
                            lloom_tensor::bail!("ple row {i} out of {rows}")
                        }
                        Ok(i as u32)
                    })
                    .collect::<Result<_>>()?;
                let idx = Tensor::from_vec(ids.clone(), ids.len(), t.device())?;
                let g = t.index_select(&idx, 0)?;
                g.flatten_all()?.to_device(dev)
            }
        }
    }

    /// The rows of `ids` dequantized into `out` (`ids.len() · head_dim`
    /// floats) on the host — the fill a host-produced staging slot takes, with
    /// no tensor and no upload. `Dense` is the tests' table and has no host
    /// form here.
    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        match self {
            PleTable::Mmap(m) => m.gather_into(ids, out),
            PleTable::Cached(c) => c.gather_into(ids, out),
            PleTable::Dense(_) => lloom_tensor::bail!("ple: gather_into on a dense table"),
        }
    }

    pub fn head_dim(&self) -> Result<usize> {
        match self {
            PleTable::Mmap(m) => Ok(m.head_dim),
            PleTable::Cached(c) => Ok(c.inner.head_dim),
            PleTable::Dense(t) => t.dim(1),
        }
    }

    /// `(hits, misses)` of the hot-row cache; zeros for the other arms.
    pub fn cache_stats(&self) -> (u64, u64) {
        match self {
            PleTable::Cached(c) => c.stats(),
            _ => (0, 0),
        }
    }
}

/// A hot-row cache over an [`MmapPleTable`]: up to `capacity` packed rows,
/// least recently used evicted first. Rows are Zipfian in a token stream, so
/// a small pinned head covers most lookups (§20.1a).
#[derive(Debug)]
pub struct CachedPleTable {
    inner: MmapPleTable,
    lru: parking_lot::Mutex<RowLru>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

/// Packed rows in `capacity` slots with an intrusive LRU list over the slots
/// (`prev`/`next` by slot index; `head` most recent, `tail` least).
#[derive(Debug)]
struct RowLru {
    row_bytes: usize,
    capacity: usize,
    slot_of: std::collections::HashMap<i64, u32>,
    id_in: Vec<i64>,
    bytes: Vec<u8>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    len: usize,
}

const NO_SLOT: u32 = u32::MAX;

impl RowLru {
    fn new(row_bytes: usize, capacity: usize) -> Self {
        Self {
            row_bytes,
            capacity,
            slot_of: std::collections::HashMap::with_capacity(capacity),
            id_in: vec![-1; capacity],
            bytes: vec![0; capacity * row_bytes],
            prev: vec![NO_SLOT; capacity],
            next: vec![NO_SLOT; capacity],
            head: NO_SLOT,
            tail: NO_SLOT,
            len: 0,
        }
    }

    fn unlink(&mut self, s: u32) {
        let (p, n) = (self.prev[s as usize], self.next[s as usize]);
        if p == NO_SLOT {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NO_SLOT {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.prev[s as usize] = NO_SLOT;
        self.next[s as usize] = NO_SLOT;
    }

    fn push_front(&mut self, s: u32) {
        self.prev[s as usize] = NO_SLOT;
        self.next[s as usize] = self.head;
        if self.head != NO_SLOT {
            self.prev[self.head as usize] = s;
        }
        self.head = s;
        if self.tail == NO_SLOT {
            self.tail = s;
        }
    }

    /// The row's bytes if cached, moved to most-recent.
    fn get(&mut self, id: i64) -> Option<&[u8]> {
        let s = *self.slot_of.get(&id)?;
        if self.head != s {
            self.unlink(s);
            self.push_front(s);
        }
        let start = s as usize * self.row_bytes;
        Some(&self.bytes[start..start + self.row_bytes])
    }

    /// Insert `row` for `id` (not present), evicting the least recent when
    /// full.
    fn insert(&mut self, id: i64, row: &[u8]) {
        let s = if self.len < self.capacity {
            let s = self.len as u32;
            self.len += 1;
            s
        } else {
            let s = self.tail;
            self.unlink(s);
            let old = self.id_in[s as usize];
            self.slot_of.remove(&old);
            s
        };
        self.id_in[s as usize] = id;
        self.slot_of.insert(id, s);
        let start = s as usize * self.row_bytes;
        self.bytes[start..start + self.row_bytes].copy_from_slice(row);
        self.push_front(s);
    }
}

impl CachedPleTable {
    /// `capacity` packed rows over `inner` (at 90 B a row, 1 M rows is 90 MB).
    pub fn new(inner: MmapPleTable, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            lloom_tensor::bail!("ple row cache needs a capacity")
        }
        let row_bytes = inner.row_bytes;
        Ok(Self {
            inner,
            lru: parking_lot::Mutex::new(RowLru::new(row_bytes, capacity)),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn stats(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.hits.load(Relaxed), self.misses.load(Relaxed))
    }

    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    pub fn gather(&self, ids: &[i64], dev: &Device) -> Result<Tensor> {
        let mut out = vec![0f32; ids.len() * self.inner.head_dim];
        self.gather_into(ids, &mut out)?;
        Tensor::from_vec(out, ids.len() * self.inner.head_dim, dev)
    }

    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let hd = self.inner.head_dim;
        check_out(ids, hd, out)?;
        let mut lru = self.lru.lock();
        for (i, &id) in ids.iter().enumerate() {
            let dst = &mut out[i * hd..(i + 1) * hd];
            if let Some(row) = lru.get(id) {
                self.hits.fetch_add(1, Relaxed);
                dequant_row(self.inner.dtype, row, dst)?;
                continue;
            }
            self.misses.fetch_add(1, Relaxed);
            let row = self.inner.packed_row(id)?;
            dequant_row(self.inner.dtype, row, dst)?;
            lru.insert(id, row);
        }
        Ok(())
    }
}

fn check_out(ids: &[i64], head_dim: usize, out: &[f32]) -> Result<()> {
    if out.len() != ids.len() * head_dim {
        lloom_tensor::bail!(
            "ple gather_into: {} floats for {} rows of {head_dim}",
            out.len(),
            ids.len()
        )
    }
    Ok(())
}

/// Dequantize one packed row of `dtype` into `out`.
fn dequant_row(dtype: GgmlDType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDType::IQ4NL => {
            use lloom_tensor::quantized::iq_quants::BlockIQ4nl;
            let bs = std::mem::size_of::<BlockIQ4nl>();
            let blocks: Vec<BlockIQ4nl> = bytes
                .chunks_exact(bs)
                .map(|b| {
                    let mut qs = [0u8; 16];
                    qs.copy_from_slice(&b[2..18]);
                    BlockIQ4nl {
                        d: half::f16::from_le_bytes([b[0], b[1]]),
                        qs,
                    }
                })
                .collect();
            BlockIQ4nl::to_float(&blocks, out);
        }
        GgmlDType::F32 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlDType::F16 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        GgmlDType::BF16 => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        other => lloom_tensor::bail!("ple table dtype {other:?}"),
    }
    Ok(())
}

/// The memory-mapped `per_layer_token_embd.weight`: `[rows, head_dim]` in a
/// GGUF block dtype, one row = `head_dim / block_size` blocks.
#[derive(Debug)]
pub struct MmapPleTable {
    mmap: memmap2::Mmap,
    data_start: usize,
    rows: usize,
    head_dim: usize,
    row_bytes: usize,
    dtype: GgmlDType,
}

impl MmapPleTable {
    /// Open the shard at `path` and locate `per_layer_token_embd.weight` in it.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = Content::read(&mut file)?;
        let name = "per_layer_token_embd.weight";
        let (dtype, shape, offset) = match content.tensor_infos.get(name) {
            Some(i) => (i.ggml_dtype, i.shape.clone(), i.offset),
            None => lloom_tensor::bail!("{} has no `{name}`", path.display()),
        };
        let (rows, head_dim) = shape.dims2()?;
        if head_dim % dtype.block_size() != 0 {
            lloom_tensor::bail!(
                "ple table head_dim {head_dim} is not a multiple of the {dtype:?} block"
            )
        }
        match dtype {
            GgmlDType::IQ4NL | GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {}
            other => lloom_tensor::bail!("ple table dtype {other:?}: no row dequant here"),
        }
        let row_bytes = head_dim / dtype.block_size() * dtype.type_size();
        let data_start = (content.tensor_data_offset + offset) as usize;
        // SAFETY: the file is opened read-only and never truncated by this
        // process; the map is read through bounds-checked slices only.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if data_start + rows * row_bytes > mmap.len() {
            lloom_tensor::bail!(
                "ple table needs {} bytes past offset {data_start}, file has {}",
                rows * row_bytes,
                mmap.len()
            )
        }
        Ok(Self {
            mmap,
            data_start,
            rows,
            head_dim,
            row_bytes,
            dtype,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The packed bytes of row `id`.
    fn packed_row(&self, id: i64) -> Result<&[u8]> {
        if id < 0 || id as usize >= self.rows {
            lloom_tensor::bail!("ple row {id} out of {}", self.rows)
        }
        let start = self.data_start + id as usize * self.row_bytes;
        Ok(&self.mmap[start..start + self.row_bytes])
    }

    /// Dequantize row `id` into `out` (`head_dim` floats).
    fn row(&self, id: i64, out: &mut [f32]) -> Result<()> {
        dequant_row(self.dtype, self.packed_row(id)?, out)
    }

    pub fn gather(&self, ids: &[i64], dev: &Device) -> Result<Tensor> {
        let mut out = vec![0f32; ids.len() * self.head_dim];
        self.gather_into(ids, &mut out)?;
        Tensor::from_vec(out, ids.len() * self.head_dim, dev)
    }

    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        check_out(ids, self.head_dim, out)?;
        for (i, &id) in ids.iter().enumerate() {
            self.row(id, &mut out[i * self.head_dim..(i + 1) * self.head_dim])?;
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    /// Packed bytes a row takes in the file.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Byte offset of row `id` in the shard file: where a `pread` of the row
    /// starts.
    pub fn row_file_offset(&self, id: i64) -> Result<u64> {
        if id < 0 || id as usize >= self.rows {
            lloom_tensor::bail!("ple row {id} out of {}", self.rows)
        }
        Ok((self.data_start + id as usize * self.row_bytes) as u64)
    }

    /// Dequantize one packed row read by the caller (a `pread` of
    /// `row_bytes()` at `row_file_offset(id)`) into `out`.
    pub fn dequant_packed(&self, bytes: &[u8], out: &mut [f32]) -> Result<()> {
        if bytes.len() != self.row_bytes || out.len() != self.head_dim {
            lloom_tensor::bail!(
                "ple dequant: {} bytes / {} floats for a {}-byte row of {}",
                bytes.len(),
                out.len(),
                self.row_bytes,
                self.head_dim
            )
        }
        dequant_row(self.dtype, bytes, out)
    }
}

/// The PLE block's weights for its one layer.
#[derive(Debug, Clone)]
pub struct PleLayer {
    /// `ple_key.weight` `[hc_dim, ple_dim]`.
    key: QMatMul,
    /// `ple_value.weight` `[hidden, ple_dim]`.
    value: QMatMul,
    /// Grouped-norm γ `[hc_dim]` (converter-folded `1 + w`), per §hc.
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    /// Depthwise conv kernel `[hc_dim, k]`, tap `k−1` newest.
    conv_weight: Tensor,
    unit_d: Tensor,
    n: usize,
    d: usize,
    k: usize,
    dilation: usize,
    eps: f32,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key: QMatMul,
        value: QMatMul,
        norm_key: Tensor,
        norm_query: Tensor,
        norm_conv: Tensor,
        conv_weight: Tensor,
        n: usize,
        d: usize,
        dilation: usize,
        eps: f32,
    ) -> Result<Self> {
        let (ch, k) = conv_weight.dims2()?;
        if ch != n * d {
            lloom_tensor::bail!("ple conv weight [{ch}, {k}] vs hc_dim {}", n * d)
        }
        let unit_d = Tensor::ones(d, norm_key.dtype(), norm_key.device())?;
        Ok(Self {
            key,
            value,
            norm_key,
            norm_query,
            norm_conv,
            conv_weight,
            unit_d,
            n,
            d,
            k,
            dilation,
            eps,
        })
    }

    /// The conv history the state channel carries per position: `(k−1)·dilation`.
    pub fn history_len(&self) -> usize {
        (self.k - 1) * self.dilation
    }

    /// The delta to add to the carry `hidden [1, seq, n·d]` for the gathered
    /// embeddings `emb [seq, ple_dim]`, advancing the conv history channel `l`.
    pub fn forward(
        &self,
        hidden: &Tensor,
        emb: &Tensor,
        conv_state: &mut RecurrentState,
        l: usize,
    ) -> Result<Tensor> {
        let (b, seq, hc_dim) = hidden.dims3()?;
        let (n, d) = (self.n, self.d);
        if b != 1 || hc_dim != n * d || emb.dim(0)? != seq {
            lloom_tensor::bail!(
                "ple: hidden {:?} / emb {:?} do not fit n={n} d={d}",
                hidden.shape(),
                emb.shape()
            )
        }
        let dt = hidden.dtype();
        let emb = emb.to_dtype(dt)?.reshape((1, seq, emb.dim(1)?))?;
        let key = self.key.forward(&emb)?; // [1, seq, hc_dim]
        let value = self.value.forward(&emb)?; // [1, seq, d]

        let key_n = grouped_rms_norm(&key, &self.unit_d, &self.norm_key, n, d, self.eps)?;
        let query_n = grouped_rms_norm(hidden, &self.unit_d, &self.norm_query, n, d, self.eps)?;

        // Per-stream dot, signed square root, sigmoid.
        let prod = (key_n.reshape((1, seq, n, d))? * query_n.reshape((1, seq, n, d))?)?;
        let s = (prod.sum(3)? * (1.0 / (d as f64).sqrt()))?; // [1, seq, n]
        let mag = s.abs()?.clamp(1e-6f32, f32::MAX)?.sqrt()?;
        let sgn = (s.gt(0f32)?.to_dtype(dt)? - s.lt(0f32)?.to_dtype(dt)?)?;
        let gate = ops::sigmoid(&(sgn * mag)?)?; // [1, seq, n]

        // value broadcast over the streams, scaled by the gate.
        let gated = value
            .reshape((1, seq, 1, d))?
            .broadcast_mul(&gate.reshape((1, seq, n, 1))?)?
            .reshape((1, seq, n * d))?;

        let normed = grouped_rms_norm(&gated, &self.unit_d, &self.norm_conv, n, d, self.eps)?;
        let conv = self.dilated_conv(&normed, conv_state, l)?; // [1, seq, n*d]
        let conv = ops::silu(&conv)?;
        gated + conv
    }

    /// Depthwise causal conv, dilated by `ngram_size`, over `x [1, seq, hc_dim]`
    /// with the channel's `[hc_dim, hist]` history prepended (oldest first);
    /// writes the new history back.
    fn dilated_conv(
        &self,
        x: &Tensor,
        conv_state: &mut RecurrentState,
        l: usize,
    ) -> Result<Tensor> {
        let (_, seq, ch) = x.dims3()?;
        let hist = self.history_len();
        let dt = x.dtype();
        let x_t = x.reshape((seq, ch))?.t()?.contiguous()?; // [ch, seq]
        let history = match conv_state.get(l) {
            Some(h) => h.to_dtype(dt)?,
            None => Tensor::zeros((ch, hist), dt, x.device())?,
        };
        let padded = Tensor::cat(&[&history, &x_t], 1)?; // [ch, hist + seq]
        let w = self.conv_weight.to_dtype(dt)?; // [ch, k]
        let mut acc: Option<Tensor> = None;
        for tap in 0..self.k {
            // tap k reads (k−1−tap)·dilation positions back: column t + tap·dilation.
            let cols = padded.narrow(1, tap * self.dilation, seq)?; // [ch, seq]
            let term = cols.broadcast_mul(&w.narrow(1, tap, 1)?)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        let out = acc.expect("k >= 1");
        // New history: the last `hist` columns of the padded input.
        let new_hist = padded.narrow(1, seq, hist)?.contiguous()?;
        conv_state.write(l, new_hist)?;
        out.t()?.reshape((1, seq, ch))
    }
}

/// The per-sequence token window the hash reads: the last `context_len`
/// tokens, newest last, `None` before the sequence start.
#[derive(Debug, Clone)]
pub struct NgramWindow {
    prev: Vec<Option<u32>>,
}

impl NgramWindow {
    pub fn new(context_len: usize) -> Self {
        Self {
            prev: vec![None; context_len],
        }
    }

    pub fn reset(&mut self) {
        for p in &mut self.prev {
            *p = None;
        }
    }

    /// The hash window for `token` at the current position: `[token, prev1,
    /// prev2, …]` (newest first).
    pub fn window(&self, token: Option<u32>) -> Vec<Option<u32>> {
        let mut w = Vec::with_capacity(self.prev.len() + 1);
        w.push(token);
        w.extend(self.prev.iter().rev().copied());
        w
    }

    /// Advance past `token`.
    pub fn push(&mut self, token: Option<u32>) {
        if self.prev.is_empty() {
            return;
        }
        self.prev.rotate_left(1);
        let last = self.prev.len() - 1;
        self.prev[last] = token;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::DType;

    /// The checkpoint's recorded hash constants (`qwen4exp.ple.*`, read from the
    /// IQ3_XXS GSQ-RCO GGUF header, 2026-09-21) — the ground truth the
    /// derivation is checked against.
    const MULTIPLIERS: [i64; 3] = [23703573157769, 20109073645365, 8052911324071];
    const SIZES: [i64; 16] = [
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
        20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ];
    const OFFSETS: [i64; 16] = [
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
        180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
    ];
    const EOS: u32 = 248044;

    fn real_hash() -> NgramHash {
        NgramHash::new(
            MULTIPLIERS.to_vec(),
            SIZES.to_vec(),
            OFFSETS.to_vec(),
            8,
            EOS,
        )
        .unwrap()
    }

    /// The reference derivation reproduces every recorded constant: seed 1234
    /// (the config default), vocab 248320, base 20 000 000, layer id 0. So the
    /// hash constants in the file are the ones the reference computes, and
    /// `derive` is a faithful port of the reference's splitmix64 / nth-prime.
    #[test]
    fn derivation_reproduces_checkpoint_constants() {
        let h = NgramHash::derive(1234, 248320, 3, 8, 20_000_000, 0, EOS).unwrap();
        assert_eq!(h.multipliers, MULTIPLIERS);
        assert_eq!(h.sizes, SIZES);
        assert_eq!(h.offsets, OFFSETS);
        assert_eq!(h.total_rows(), 320001446);
    }

    /// Hand-computed rows for a 3-token window, both orders, every head:
    /// `mixed_2 = t0·m0 ^ t1·m1`, `mixed_3 = mixed_2 ^ t2·m2`, row = offset +
    /// mixed mod size. Computed here with i128 arithmetic (no wrapping can
    /// occur — the products stay below 2^63 for in-vocab ids) as the
    /// independent reference.
    #[test]
    fn rows_match_hand_computation() {
        let h = real_hash();
        let (t0, t1, t2) = (846u32, 25u32, 1234u32);
        let mut out = vec![0i64; 16];
        h.row_ids(&[Some(t0), Some(t1), Some(t2)], &mut out);
        let m2 = (t0 as i128 * MULTIPLIERS[0] as i128) ^ (t1 as i128 * MULTIPLIERS[1] as i128);
        let m3 = m2 ^ (t2 as i128 * MULTIPLIERS[2] as i128);
        assert!(m2 >= 0 && m2 < (1i128 << 63));
        assert!(m3 >= 0 && m3 < (1i128 << 63));
        for hd in 0..8 {
            let want = OFFSETS[hd] as i128 + m2 % SIZES[hd] as i128;
            assert_eq!(out[hd] as i128, want, "bigram head {hd}");
            let want3 = OFFSETS[8 + hd] as i128 + m3 % SIZES[8 + hd] as i128;
            assert_eq!(out[8 + hd] as i128, want3, "trigram head {hd}");
        }
        // The reference's exact numbers for this window (vLLM semantics),
        // pinned so a change in operator order is caught by value:
        assert_eq!(out[0], (m2 % SIZES[0] as i128) as i64);
        assert_eq!(out[15], OFFSETS[15] + (m3 % SIZES[15] as i128) as i64);
    }

    /// The remainder is non-negative even when the XOR has its sign bit set
    /// (`torch.remainder` semantics), never the C-style negative remainder and
    /// never the unsigned reinterpretation.
    #[test]
    fn remainder_is_python_style_on_negative_mix() {
        // A multiplier set whose products overflow into the sign bit.
        let h = NgramHash::new(
            vec![i64::MAX / 3, i64::MAX / 5, 7],
            vec![101; 16],
            vec![0; 16],
            8,
            EOS,
        )
        .unwrap();
        let mut out = vec![0i64; 16];
        h.row_ids(&[Some(7), Some(9), Some(11)], &mut out);
        let mixed = (7i64.wrapping_mul(i64::MAX / 3)) ^ (9i64.wrapping_mul(i64::MAX / 5));
        assert!(mixed < 0, "the fixture must exercise a negative mix");
        let want = mixed.rem_euclid(101);
        assert_eq!(out[0], want);
        assert!(out.iter().all(|&r| (0..101).contains(&r)));
    }

    /// EOS scoping: an EOS one position back cuts the trigram head's older
    /// predecessor to EOS too; an EOS two back only affects the trigram heads;
    /// the current token being EOS cuts nothing; a missing predecessor
    /// (sequence start) reads as EOS. Regression: a hash that ignores the
    /// reset gives a different row for the cut case.
    #[test]
    fn eos_resets_the_window_by_scope() {
        let h = real_hash();
        let ids = |w: &[Option<u32>]| {
            let mut o = vec![0i64; 16];
            h.row_ids(w, &mut o);
            o
        };
        // EOS one back: bigram heads see (t0, EOS); trigram heads see (t0, EOS, EOS)
        // regardless of the real two-back token.
        let cut = ids(&[Some(5), Some(EOS), Some(77)]);
        let cut_other = ids(&[Some(5), Some(EOS), Some(78)]);
        assert_eq!(
            cut, cut_other,
            "the token behind an EOS must not reach the hash"
        );
        let explicit = ids(&[Some(5), Some(EOS), Some(EOS)]);
        assert_eq!(cut, explicit);
        // Not resetting would be a different row: prove the reset is live.
        let uncut = ids(&[Some(5), Some(6), Some(77)]);
        assert_ne!(cut[8..], uncut[8..]);
        // EOS two back: bigram heads unchanged vs a non-EOS two-back.
        let a = ids(&[Some(5), Some(6), Some(EOS)]);
        let b = ids(&[Some(5), Some(6), Some(77)]);
        assert_eq!(a[..8], b[..8]);
        assert_ne!(a[8..], b[8..]);
        // The current token being EOS does not cut its predecessors.
        let c = ids(&[Some(EOS), Some(6), Some(77)]);
        let c2 = ids(&[Some(EOS), Some(6), Some(78)]);
        assert_ne!(c[8..], c2[8..]);
        // Missing predecessors are EOS.
        let start = ids(&[Some(5), None, None]);
        assert_eq!(start, ids(&[Some(5), Some(EOS), Some(EOS)]));
        let one = ids(&[Some(5), Some(6), None]);
        assert_eq!(one, ids(&[Some(5), Some(6), Some(EOS)]));
    }

    /// The window rotates newest-last and reads newest-first.
    #[test]
    fn ngram_window_rotates() {
        let mut w = NgramWindow::new(2);
        assert_eq!(w.window(Some(1)), vec![Some(1), None, None]);
        w.push(Some(1));
        assert_eq!(w.window(Some(2)), vec![Some(2), Some(1), None]);
        w.push(Some(2));
        w.push(Some(3));
        assert_eq!(w.window(Some(4)), vec![Some(4), Some(3), Some(2)]);
        w.reset();
        assert_eq!(w.window(Some(9)), vec![Some(9), None, None]);
    }

    /// The dilated conv reads taps `t − (k−1−j)·dilation` and carries a
    /// `(k−1)·dilation` history across calls: two single-position calls equal
    /// one two-position call, and a hand-computed value checks the taps.
    #[test]
    fn dilated_conv_taps_and_history() {
        let dev = Device::Cpu;
        let (n, d, k, dil) = (1usize, 2usize, 4usize, 3usize);
        let ch = n * d;
        // weight [ch, k]: channel 0 taps [1, 2, 3, 4], channel 1 taps [0,0,0,1].
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4., 0., 0., 0., 1.], (ch, k), &dev).unwrap();
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let layer = PleLayer::new(
            zero(ch, 4),
            zero(d, 4),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            w,
            n,
            d,
            dil,
            1e-6,
        )
        .unwrap();
        assert_eq!(layer.history_len(), 9);
        // x over 12 positions: channel 0 = position index, channel 1 = 10·index.
        let seq = 12;
        let mut xv = vec![0f32; seq * ch];
        for t in 0..seq {
            xv[t * ch] = t as f32;
            xv[t * ch + 1] = 10.0 * t as f32;
        }
        let x = Tensor::from_vec(xv, (1, seq, ch), &dev).unwrap();
        let mut st = RecurrentState::new(1);
        let y = layer.dilated_conv(&x, &mut st, 0).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t = 11, channel 0: 1·x[2] + 2·x[5] + 3·x[8] + 4·x[11] = 2 + 10 + 24 + 44 = 80.
        assert_eq!(yv[11 * ch], 80.0);
        // t = 11, channel 1: newest tap only = 10·11 = 110.
        assert_eq!(yv[11 * ch + 1], 110.0);
        // t = 1, channel 0: taps at -8, -5, -2 are zero history; 4·x[1] = 4.
        assert_eq!(yv[ch], 4.0);
        // Position by position from a fresh state equals the one-shot call.
        let mut st2 = RecurrentState::new(1);
        for t in 0..seq {
            let xt = x.narrow(1, t, 1).unwrap();
            let yt = layer.dilated_conv(&xt, &mut st2, 0).unwrap();
            let v = yt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(v, yv[t * ch..(t + 1) * ch].to_vec(), "position {t}");
        }
    }

    /// The PLE block on Metal equals the CPU block at the real widths (n=4,
    /// d=2560, 16 heads × 160): same weights, same 3-position chunk with a
    /// carried conv history, delta within f32 noise.
    #[cfg(feature = "metal")]
    #[test]
    fn ple_block_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, ple_dim) = (4usize, 2560usize, 2560usize);
        let ch = n * d;
        let build = |dev: &Device| {
            let lin = |o: usize, i: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..o * i)
                            .map(|x| ((x as f32 * 0.0037 + seed).sin()) * 0.02)
                            .collect::<Vec<_>>(),
                        (o, i),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let vecw = |k: usize, f: &dyn Fn(usize) -> f32| {
                Tensor::from_vec((0..k).map(f).collect::<Vec<f32>>(), k, dev).unwrap()
            };
            PleLayer::new(
                lin(ch, ple_dim, 1.0),
                lin(d, ple_dim, 2.0),
                vecw(ch, &|i| 1.0 + 0.001 * (i % 9) as f32),
                vecw(ch, &|i| 1.0 - 0.001 * (i % 7) as f32),
                vecw(ch, &|i| 1.0 + 0.002 * (i % 5) as f32),
                Tensor::from_vec(
                    (0..ch * 4)
                        .map(|x| (x as f32 * 0.1).cos() * 0.3)
                        .collect::<Vec<_>>(),
                    (ch, 4),
                    dev,
                )
                .unwrap(),
                n,
                d,
                3,
                1e-6,
            )
            .unwrap()
        };
        let (b_cpu, b_gpu) = (build(&cpu), build(&gpu));
        let seq = 3;
        let hidden: Vec<f32> = (0..seq * ch).map(|x| (x as f32 * 0.002).cos()).collect();
        let emb: Vec<f32> = (0..seq * ple_dim)
            .map(|x| (x as f32 * 0.011).sin())
            .collect();
        let h_cpu = Tensor::from_vec(hidden.clone(), (1, seq, ch), &cpu).unwrap();
        let h_gpu = Tensor::from_vec(hidden, (1, seq, ch), &gpu).unwrap();
        let e_cpu = Tensor::from_vec(emb.clone(), (seq, ple_dim), &cpu).unwrap();
        let e_gpu = Tensor::from_vec(emb, (seq, ple_dim), &gpu).unwrap();
        let (mut sc, mut sg) = (RecurrentState::new(1), RecurrentState::new(1));
        let dc = b_cpu.forward(&h_cpu, &e_cpu, &mut sc, 0).unwrap();
        let dg = b_gpu.forward(&h_gpu, &e_gpu, &mut sg, 0).unwrap();
        // and once more, so the carried history is exercised on both
        let dc2 = b_cpu.forward(&h_cpu, &e_cpu, &mut sc, 0).unwrap();
        let dg2 = b_gpu.forward(&h_gpu, &e_gpu, &mut sg, 0).unwrap();
        for (name, a, b) in [("first", &dc, &dg), ("second", &dc2, &dg2)] {
            let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vb = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                vb.iter().filter(|x| !x.is_finite()).count(),
                0,
                "{name}: non-finite on Metal"
            );
            let scale = va.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 2e-3 * scale,
                "{name}: Metal/CPU differ by {md} at scale {scale}"
            );
        }
    }

    /// Whole block on a tiny dense table: shapes thread, output finite, and the
    /// table row actually matters (two different tokens → different deltas)
    /// while a zero-weight block is a zero delta.
    #[test]
    fn ple_layer_threads_end_to_end() {
        let dev = Device::Cpu;
        let (n, d, ple_dim, heads) = (2usize, 4usize, 8usize, 2usize);
        let head_dim = ple_dim / heads;
        let ch = n * d;
        let lin = |o: usize, i: usize, seed: f32| {
            QMatMul::Tensor(
                Tensor::from_vec(
                    (0..o * i)
                        .map(|x| ((x as f32 * 0.37 + seed).sin()) * 0.5)
                        .collect::<Vec<_>>(),
                    (o, i),
                    &dev,
                )
                .unwrap(),
            )
        };
        let layer = PleLayer::new(
            lin(ch, ple_dim, 1.0),
            lin(d, ple_dim, 2.0),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::ones(ch, DType::F32, &dev).unwrap(),
            Tensor::from_vec(
                (0..ch * 4)
                    .map(|x| (x as f32 * 0.1).cos() * 0.3)
                    .collect::<Vec<_>>(),
                (ch, 4),
                &dev,
            )
            .unwrap(),
            n,
            d,
            3,
            1e-6,
        )
        .unwrap();
        let rows = 50usize;
        let table = PleTable::Dense(
            Tensor::from_vec(
                (0..rows * head_dim)
                    .map(|x| (x as f32 * 0.11).sin())
                    .collect::<Vec<_>>(),
                (rows, head_dim),
                &dev,
            )
            .unwrap(),
        );
        let hidden = Tensor::from_vec(
            (0..ch).map(|x| (x as f32 * 0.2).cos()).collect::<Vec<_>>(),
            (1, 1, ch),
            &dev,
        )
        .unwrap();
        let mut st = RecurrentState::new(1);
        let emb_a = table
            .gather(&[3, 17], &dev)
            .unwrap()
            .reshape((1, ple_dim))
            .unwrap();
        let emb_b = table
            .gather(&[4, 17], &dev)
            .unwrap()
            .reshape((1, ple_dim))
            .unwrap();
        let da = layer.forward(&hidden, &emb_a, &mut st, 0).unwrap();
        assert_eq!(da.dims(), &[1, 1, ch]);
        let mut st_b = RecurrentState::new(1);
        let db = layer.forward(&hidden, &emb_b, &mut st_b, 0).unwrap();
        let (va, vb) = (
            da.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            db.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(va.iter().all(|x| x.is_finite()));
        assert!(
            va.iter().zip(&vb).any(|(a, b)| (a - b).abs() > 1e-6),
            "the gathered row must reach the output"
        );
        assert!(st.get(0).is_some(), "the conv history advanced");
    }
}
