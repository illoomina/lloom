//! An n-gram memory's table: hashed n-gram embedding rows, read a handful per
//! token from a table far too large to be resident — Qwen4-exp's N-gram PLE
//! (320 M rows, 28.8 GB) and the Engram shape DeepSeek-V4.1 shares with it
//! (`design/docs/expert-streaming.md` §20.1a, §20.13). The family owns the
//! hash's constants and the block that consumes the rows; this is the hash
//! ([`NgramHash`] over an [`NgramWindow`]), the storage — where the rows live
//! and how one token's rows are read — and the host fill ([`HostFeed`]).
//!
//! Storage arms ([`RowTable`]): a memory map of the checkpoint's table
//! ([`MmapRows`]), an LRU of packed rows in front of it ([`CachedRows`]), and a
//! small dense tensor for tests. Every arm dequantizes the same file bytes, so
//! the arms are byte-identical to each other.
//!
//! A token's rows are scattered over the whole file, so a row the page cache
//! does not hold is an SSD read of ~0.1 ms, and a token's sixteen of them read
//! one after another cost ~1.6-2.2 ms on the host. [`ReadPolicy::Adaptive`]
//! reads the rows the page cache holds through the map and the others
//! concurrently: ~0.35 ms for sixteen cold rows (§20.13).
//!
//! # The host fill (§20.13, option A)
//!
//! A decode step whose rows depend on the token the step before it sampled
//! cannot be encoded ahead by a host that must know the token first. The host
//! fill moves the wait onto the GPU: the forward encodes a wait on a shared
//! event just before its first reader of the rows and reads a staging slot
//! ([`FeedLink::encode_read`]); a producer thread learns each published token,
//! hashes its window, gathers the rows into the slot and signals
//! ([`NgramFill::fill`]). The rows are the ones the host path reads, through
//! the same math, so the arms are byte-identical by construction.
//!
//! Every wait is satisfiable when it is encoded: the forward that reads value
//! `k` consumes the `k`-th token the loop fed, and a forward is refused unless
//! that token is already in the command stream ahead of it. A producer that
//! fails, or a run that ends early, releases every wait it left
//! ([`FeedLink::release`]) rather than leaving the queue parked.

use crate::nn::slot_cache::{Hashed, ListLru, SlotCache};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::{GgmlDType, GgmlType};
use lloom_tensor::{Device, Result, Tensor};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering, Ordering::Relaxed};
use std::sync::Arc;

/// The per-layer token embedding table's GGUF name (Qwen4-exp's N-gram PLE).
pub const PER_LAYER_TOKEN_EMBD: &str = "per_layer_token_embd.weight";

/// How [`MmapRows`] reads a token's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadPolicy {
    /// Every row through the map on the calling thread; a row the page cache
    /// does not hold is a page fault there.
    Serial,
    /// Rows the page cache holds through the map on the calling thread, the
    /// others by `pread` on the read pool, concurrently. Serial when at most
    /// one row is cold (the pool's hand-off costs about one read).
    Adaptive,
    /// Every row by `pread` on the read pool: the concurrent path whatever
    /// the page cache holds (the arm the tests pin that path with).
    Concurrent,
}

impl ReadPolicy {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ReadPolicy::Serial,
            1 => ReadPolicy::Adaptive,
            _ => ReadPolicy::Concurrent,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            ReadPolicy::Serial => 0,
            ReadPolicy::Adaptive => 1,
            ReadPolicy::Concurrent => 2,
        }
    }
}

/// Threads of the read pool: a token's rows, one read each in flight.
const READ_THREADS: usize = 16;

/// The read pool's scheduling class, stated (`default`) and read back by its
/// threads at start: the host fill's producer waits on these reads, and the
/// GPU on the producer (`docs/invariants.md` §15.2 #10b).
static READ_POOL_QOS: AtomicU8 = AtomicU8::new(0);

/// The read pool's class (`Unknown` before it starts, or off macOS).
pub fn read_pool_qos() -> crate::generation::QosClass {
    crate::generation::QosClass::from_index(READ_POOL_QOS.load(Relaxed))
}

/// The pool cold rows are read on, built on first use.
#[cfg(unix)]
fn read_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(READ_THREADS)
            .thread_name(|i| format!("ngram-read-{i}"))
            .start_handler(|_| {
                let (class, _) = crate::generation::hot_thread::claim_for_spawned("default");
                READ_POOL_QOS.store(class as u8, Relaxed)
            })
            .build()
            .expect("ngram read pool")
    })
}

/// Where the table's rows come from.
#[derive(Debug)]
pub enum RowTable {
    /// A memory map of the checkpoint's table, rows dequantized on read.
    Mmap(MmapRows),
    /// The memory map behind a hot-row cache of packed rows keyed on row id:
    /// a hit skips the file read, a miss reads the row and caches it,
    /// evicting the least recently used.
    Cached(CachedRows),
    /// A small dense `[rows, head_dim]` f32 table (tests).
    Dense(Tensor),
}

impl RowTable {
    /// The rows of `ids`, concatenated to one `[ids.len() · head_dim]` f32
    /// vector on `dev`.
    pub fn gather(&self, ids: &[i64], dev: &Device) -> Result<Tensor> {
        match self {
            RowTable::Mmap(m) => m.gather(ids, dev),
            RowTable::Cached(c) => c.gather(ids, dev),
            RowTable::Dense(t) => {
                let rows = t.dim(0)? as i64;
                let ids: Vec<u32> = ids
                    .iter()
                    .map(|&i| {
                        if i < 0 || i >= rows {
                            lloom_tensor::bail!("ngram row {i} out of {rows}")
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
    /// floats) on the host — the form a host-filled staging slot takes.
    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        match self {
            RowTable::Mmap(m) => m.gather_into(ids, out),
            RowTable::Cached(c) => c.gather_into(ids, out),
            RowTable::Dense(_) => lloom_tensor::bail!("ngram rows: gather_into on a dense table"),
        }
    }

    pub fn head_dim(&self) -> Result<usize> {
        match self {
            RowTable::Mmap(m) => Ok(m.head_dim),
            RowTable::Cached(c) => Ok(c.inner.head_dim),
            RowTable::Dense(t) => t.dim(1),
        }
    }

    /// `(hits, misses)` of the hot-row cache; zeros for the other arms.
    pub fn cache_stats(&self) -> (u64, u64) {
        match self {
            RowTable::Cached(c) => c.stats(),
            _ => (0, 0),
        }
    }

    /// The memory map under the arm, if any (its read policy and counters).
    pub fn mmap(&self) -> Option<&MmapRows> {
        match self {
            RowTable::Mmap(m) => Some(m),
            RowTable::Cached(c) => Some(&c.inner),
            RowTable::Dense(_) => None,
        }
    }
}

/// A hot-row cache over an [`MmapRows`]: up to `capacity` packed rows, least
/// recently used evicted first.
#[derive(Debug)]
pub struct CachedRows {
    inner: MmapRows,
    lru: parking_lot::Mutex<RowLru>,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Packed rows in `capacity` slots: a [`SlotCache`] keyed by the row id
/// (hashed, the ids are sparse) with the exact-LRU list policy, over the
/// rows' bytes.
#[derive(Debug)]
struct RowLru {
    row_bytes: usize,
    slots: SlotCache<i64, Hashed<i64>, ListLru>,
    bytes: Vec<u8>,
}

impl RowLru {
    fn new(row_bytes: usize, capacity: usize) -> Self {
        Self {
            row_bytes,
            slots: SlotCache::with(Hashed::with_capacity(capacity), capacity),
            bytes: vec![0; capacity * row_bytes],
        }
    }

    fn contains(&self, id: i64) -> bool {
        self.slots.slot_of(id).is_some()
    }

    /// The row's bytes if cached, moved to most-recent.
    fn get(&mut self, id: i64) -> Option<&[u8]> {
        let s = self.slots.lookup(id)?;
        let start = s.index() * self.row_bytes;
        Some(&self.bytes[start..start + self.row_bytes])
    }

    /// Insert `row` for `id` (not present), evicting the least recent when
    /// full.
    fn insert(&mut self, id: i64, row: &[u8]) {
        let (s, _displaced) = self.slots.claim();
        let start = s.index() * self.row_bytes;
        self.bytes[start..start + self.row_bytes].copy_from_slice(row);
        self.slots.install(s, id);
    }
}

impl CachedRows {
    /// `capacity` packed rows over `inner`.
    pub fn new(inner: MmapRows, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            lloom_tensor::bail!("ngram row cache needs a capacity")
        }
        let row_bytes = inner.row_bytes;
        Ok(Self {
            inner,
            lru: parking_lot::Mutex::new(RowLru::new(row_bytes, capacity)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    pub fn stats(&self) -> (u64, u64) {
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

    /// The rows the cache does not hold on entry are read first, through the
    /// map's read policy, then the lookups run in order exactly as one at a
    /// time would (a row evicted by an earlier insert of the same call is a
    /// miss again, read through the map) — so the hits, the misses and the
    /// cache's final state are the sequential ones.
    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        let hd = self.inner.head_dim;
        let rb = self.inner.row_bytes;
        check_out(ids, hd, out)?;
        let mut lru = self.lru.lock();
        let mut absent: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|&id| !lru.contains(id))
            .collect();
        absent.sort_unstable();
        absent.dedup();
        let mut fetched = vec![0u8; absent.len() * rb];
        self.inner.read_packed(&absent, &mut fetched)?;
        for (i, &id) in ids.iter().enumerate() {
            let dst = &mut out[i * hd..(i + 1) * hd];
            if let Some(row) = lru.get(id) {
                self.hits.fetch_add(1, Relaxed);
                dequant_row(self.inner.dtype, row, dst)?;
                continue;
            }
            self.misses.fetch_add(1, Relaxed);
            let row = match absent.binary_search(&id) {
                Ok(k) => &fetched[k * rb..(k + 1) * rb],
                Err(_) => self.inner.packed_row(id)?,
            };
            dequant_row(self.inner.dtype, row, dst)?;
            lru.insert(id, row);
        }
        Ok(())
    }
}

fn check_out(ids: &[i64], head_dim: usize, out: &[f32]) -> Result<()> {
    if out.len() != ids.len() * head_dim {
        lloom_tensor::bail!(
            "ngram gather_into: {} floats for {} rows of {head_dim}",
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
        GgmlDType::MXFP4 => {
            use lloom_tensor::quantized::fp_quants::BlockMxfp4;
            let blocks: Vec<BlockMxfp4> = bytes
                .chunks_exact(std::mem::size_of::<BlockMxfp4>())
                .map(|b| {
                    let mut qs = [0u8; 16];
                    qs.copy_from_slice(&b[1..17]);
                    BlockMxfp4 { e: b[0], qs }
                })
                .collect();
            BlockMxfp4::to_float(&blocks, out);
        }
        GgmlDType::FP8E4M3 => {
            use lloom_tensor::quantized::fp_quants::BlockFp8E4M3;
            let blocks: Vec<BlockFp8E4M3> = bytes
                .chunks_exact(std::mem::size_of::<BlockFp8E4M3>())
                .map(|b| {
                    let mut qs = [0u8; 32];
                    qs.copy_from_slice(&b[2..34]);
                    BlockFp8E4M3 {
                        d: half::bf16::from_le_bytes([b[0], b[1]]),
                        qs,
                    }
                })
                .collect();
            BlockFp8E4M3::to_float(&blocks, out);
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
        other => lloom_tensor::bail!("ngram table dtype {other:?}"),
    }
    Ok(())
}

/// A table in a GGUF file, memory-mapped: `[rows, head_dim]` in a block
/// dtype, one row = `head_dim / block_size` blocks. The file is also kept
/// open for the concurrent reads of [`ReadPolicy::Adaptive`].
#[derive(Debug)]
pub struct MmapRows {
    mmap: memmap2::Mmap,
    file: std::fs::File,
    data_start: usize,
    rows: usize,
    head_dim: usize,
    row_bytes: usize,
    dtype: GgmlDType,
    policy: AtomicU8,
    /// Rows read through the map, and rows read by `pread` on the pool.
    mapped_rows: AtomicU64,
    pooled_rows: AtomicU64,
}

impl MmapRows {
    /// The per-layer token embedding table ([`PER_LAYER_TOKEN_EMBD`]) of the
    /// shard at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_named(path, PER_LAYER_TOKEN_EMBD)
    }

    /// The table `name` of the shard at `path`.
    pub fn open_named(path: &Path, name: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = Content::read(&mut file)?;
        let (dtype, shape, offset) = match content.tensor_infos.get(name) {
            Some(i) => (i.ggml_dtype, i.shape.clone(), i.offset),
            None => lloom_tensor::bail!("{} has no `{name}`", path.display()),
        };
        let (rows, head_dim) = shape.dims2()?;
        if head_dim % dtype.block_size() != 0 {
            lloom_tensor::bail!(
                "ngram table head_dim {head_dim} is not a multiple of the {dtype:?} block"
            )
        }
        match dtype {
            GgmlDType::IQ4NL
            | GgmlDType::MXFP4
            | GgmlDType::FP8E4M3
            | GgmlDType::F32
            | GgmlDType::F16
            | GgmlDType::BF16 => {}
            other => lloom_tensor::bail!("ngram table dtype {other:?}: no row dequant here"),
        }
        let row_bytes = head_dim / dtype.block_size() * dtype.type_size();
        let data_start = (content.tensor_data_offset + offset) as usize;
        // SAFETY: the file is opened read-only and never truncated by this
        // process; the map is read through bounds-checked slices only.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if data_start + rows * row_bytes > mmap.len() {
            lloom_tensor::bail!(
                "ngram table needs {} bytes past offset {data_start}, file has {}",
                rows * row_bytes,
                mmap.len()
            )
        }
        Ok(Self {
            mmap,
            file,
            data_start,
            rows,
            head_dim,
            row_bytes,
            dtype,
            policy: AtomicU8::new(ReadPolicy::Adaptive.as_u8()),
            mapped_rows: AtomicU64::new(0),
            pooled_rows: AtomicU64::new(0),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
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

    pub fn read_policy(&self) -> ReadPolicy {
        ReadPolicy::from_u8(self.policy.load(Relaxed))
    }

    pub fn set_read_policy(&self, policy: ReadPolicy) {
        self.policy.store(policy.as_u8(), Relaxed);
    }

    /// `(rows read through the map, rows read by pread on the pool)`.
    pub fn read_counters(&self) -> (u64, u64) {
        (
            self.mapped_rows.load(Relaxed),
            self.pooled_rows.load(Relaxed),
        )
    }

    /// Byte offset of row `id` in the file.
    pub fn row_file_offset(&self, id: i64) -> Result<u64> {
        self.check_id(id)?;
        Ok((self.data_start + id as usize * self.row_bytes) as u64)
    }

    /// Dequantize one packed row read by the caller (`row_bytes()` at
    /// `row_file_offset(id)`) into `out`.
    pub fn dequant_packed(&self, bytes: &[u8], out: &mut [f32]) -> Result<()> {
        if bytes.len() != self.row_bytes || out.len() != self.head_dim {
            lloom_tensor::bail!(
                "ngram dequant: {} bytes / {} floats for a {}-byte row of {}",
                bytes.len(),
                out.len(),
                self.row_bytes,
                self.head_dim
            )
        }
        dequant_row(self.dtype, bytes, out)
    }

    fn check_id(&self, id: i64) -> Result<()> {
        if id < 0 || id as usize >= self.rows {
            lloom_tensor::bail!("ngram row {id} out of {}", self.rows)
        }
        Ok(())
    }

    /// The packed bytes of row `id`, through the map.
    fn packed_row(&self, id: i64) -> Result<&[u8]> {
        self.check_id(id)?;
        let start = self.data_start + id as usize * self.row_bytes;
        Ok(&self.mmap[start..start + self.row_bytes])
    }

    pub fn gather(&self, ids: &[i64], dev: &Device) -> Result<Tensor> {
        let mut out = vec![0f32; ids.len() * self.head_dim];
        self.gather_into(ids, &mut out)?;
        Tensor::from_vec(out, ids.len() * self.head_dim, dev)
    }

    pub fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<()> {
        check_out(ids, self.head_dim, out)?;
        let hd = self.head_dim;
        let dtype = self.dtype;
        self.read_with(ids, |i, row| {
            dequant_row(dtype, row, &mut out[i * hd..(i + 1) * hd])
        })
    }

    /// The packed bytes of `ids` into `out` (`ids.len() · row_bytes`).
    fn read_packed(&self, ids: &[i64], out: &mut [u8]) -> Result<()> {
        let rb = self.row_bytes;
        if out.len() != ids.len() * rb {
            lloom_tensor::bail!(
                "ngram read_packed: {} bytes for {} rows",
                out.len(),
                ids.len()
            )
        }
        self.read_with(ids, |i, row| {
            out[i * rb..(i + 1) * rb].copy_from_slice(row);
            Ok(())
        })
    }

    /// Hand each row of `ids` to `sink(index, bytes)` on the calling thread,
    /// read under the table's [`ReadPolicy`].
    fn read_with(
        &self,
        ids: &[i64],
        mut sink: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        for &id in ids {
            self.check_id(id)?;
        }
        let pooled: Vec<usize> = match self.read_policy() {
            ReadPolicy::Serial => Vec::new(),
            ReadPolicy::Concurrent => (0..ids.len()).collect(),
            ReadPolicy::Adaptive => {
                let cold: Vec<usize> = (0..ids.len()).filter(|&i| !self.resident(ids[i])).collect();
                if cold.len() < 2 {
                    Vec::new()
                } else {
                    cold
                }
            }
        };
        let fetched = self.pread_rows(ids, &pooled)?;
        let mut next = 0;
        for (i, &id) in ids.iter().enumerate() {
            if next < pooled.len() && pooled[next] == i {
                let rb = self.row_bytes;
                sink(i, &fetched[next * rb..(next + 1) * rb])?;
                next += 1;
            } else {
                sink(i, self.packed_row(id)?)?;
            }
        }
        self.mapped_rows
            .fetch_add((ids.len() - pooled.len()) as u64, Relaxed);
        self.pooled_rows.fetch_add(pooled.len() as u64, Relaxed);
        Ok(())
    }

    /// `pread` the rows `ids[which]` concurrently on the read pool, packed in
    /// `which`'s order.
    #[cfg(unix)]
    fn pread_rows(&self, ids: &[i64], which: &[usize]) -> Result<Vec<u8>> {
        use rayon::prelude::*;
        use std::os::unix::fs::FileExt;
        let rb = self.row_bytes;
        let mut buf = vec![0u8; which.len() * rb];
        if which.is_empty() {
            return Ok(buf);
        }
        read_pool().install(|| {
            buf.par_chunks_mut(rb)
                .zip(which.par_iter())
                .try_for_each(|(dst, &i)| {
                    let off = self.data_start as u64 + ids[i] as u64 * rb as u64;
                    self.file.read_exact_at(dst, off)
                })
        })?;
        Ok(buf)
    }

    #[cfg(not(unix))]
    fn pread_rows(&self, ids: &[i64], which: &[usize]) -> Result<Vec<u8>> {
        let rb = self.row_bytes;
        let mut buf = vec![0u8; which.len() * rb];
        for (dst, &i) in buf.chunks_mut(rb).zip(which) {
            dst.copy_from_slice(self.packed_row(ids[i])?);
        }
        Ok(buf)
    }

    /// Whether every page row `id` spans is in memory (`mincore`); `true`
    /// where that cannot be asked, which reads the row through the map.
    fn resident(&self, id: i64) -> bool {
        let start = self.data_start + id as usize * self.row_bytes;
        let page = page_size();
        let first = start / page * page;
        let last = (start + self.row_bytes - 1) / page * page;
        let pages = (last - first) / page + 1;
        let mut v = [0u8; 2];
        // SAFETY: `[first, last + page)` lies inside the map (the row does,
        // and the map covers whole pages); `v` holds a byte per page (a row
        // spans at most two).
        let ok = unsafe { mincore(self.mmap.as_ptr().add(first), pages * page, v.as_mut_ptr()) };
        !ok || v[..pages].iter().all(|b| b & 1 != 0)
    }
}

fn page_size() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    // SAFETY: sysconf has no preconditions.
    *PAGE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize)
}

/// `mincore` over `len` bytes at `addr` into `vec`; `false` when it failed.
///
/// # Safety
/// `[addr, addr + len)` is mapped, page-aligned at `addr`, and `vec` holds a
/// byte per page.
#[cfg(target_os = "macos")]
unsafe fn mincore(addr: *const u8, len: usize, vec: *mut u8) -> bool {
    libc::mincore(addr as *const libc::c_void, len, vec as *mut libc::c_char) == 0
}

#[cfg(all(unix, not(target_os = "macos")))]
unsafe fn mincore(addr: *const u8, len: usize, vec: *mut u8) -> bool {
    libc::mincore(addr as *mut libc::c_void, len, vec as *mut libc::c_uchar) == 0
}

#[cfg(not(unix))]
unsafe fn mincore(_addr: *const u8, _len: usize, _vec: *mut u8) -> bool {
    false
}

/// The per-layer hash constants, as the checkpoint records them
/// (`qwen4exp.ple.{layer_multipliers, head_offsets, head_vocab_sizes, eos_token_id}`).
#[derive(Debug, Clone)]
pub struct NgramHash {
    /// `layer_multipliers[i]` multiplies the token `i` positions back.
    pub(crate) multipliers: Vec<i64>,
    /// Per-head prime table sizes, `n_heads` of them.
    pub(crate) sizes: Vec<i64>,
    /// Per-head row offsets into the flat table.
    pub(crate) offsets: Vec<i64>,
    pub(crate) heads_per_ngram: usize,
    pub(crate) ngram_size: usize,
    pub(crate) eos: i64,
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

/// The per-sequence token window the hash reads: the last `context_len`
/// tokens, newest last, `None` before the sequence start.
#[derive(Debug, Clone)]
pub struct NgramWindow {
    prev: Vec<Option<u32>>,
}

impl crate::nn::branch::RetainState for NgramWindow {
    type Retained = Vec<Option<u32>>;
    fn retain_state(&mut self) -> Result<Self::Retained> {
        let mut saved = crate::nn::branch::retained_vec(self.prev.len())?;
        saved.extend_from_slice(&self.prev);
        Ok(saved)
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        if self.prev.len() != saved.len() {
            lloom_tensor::bail!("retained n-gram state has incompatible context length")
        }
        self.prev.copy_from_slice(saved);
        Ok(())
    }
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

    /// [`Self::window`] into `out`, which keeps its capacity.
    pub fn window_into(&self, token: Option<u32>, out: &mut Vec<Option<u32>>) {
        out.clear();
        out.push(token);
        out.extend(self.prev.iter().rev().copied());
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

/// Staging slots a run's producer fills: two, so the slot a producer writes
/// is never the one the forward before it reads (the staging ring's
/// depth + 1).
pub const HOST_FILL_SLOTS: usize = 2;

/// What one run's three parties share: the forward encodes its waits through
/// it, the decode loop counts the tokens it fed, the producer signals.
///
/// Values are `base + k` for the `k`-th decode forward of the run (1-based),
/// which consumes the `k`-th token the loop fed: position `pos0 + k - 1`. A
/// forward replayed at the same position (a device-plan miss) waits for the
/// same value and reads the same slot. A draw the loop takes back
/// ([`FeedLink::retract`], a provisional step redone behind the next
/// forward) shifts every later value by one: the forward encoded again at
/// its position waits for the fill of the token drawn again, never for the
/// value the taken-back token already signalled.
#[derive(Debug)]
pub struct FeedLink {
    #[cfg(feature = "metal")]
    ring: Arc<lloom_tensor::metal_backend::HostFedRing>,
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    base: u64,
    pos0: usize,
    fed: AtomicU64,
    signalled: AtomicU64,
    demanded: AtomicU64,
    released: AtomicBool,
    /// Draws taken back this run: added to a position's index to give the
    /// value its forward waits for.
    shift: AtomicU64,
    /// Take-backs the loop asked of the producer, and those it has done.
    retract_req: AtomicU64,
    retract_ack: AtomicU64,
    /// Why the producer stopped, if it failed.
    failure: parking_lot::Mutex<Option<String>>,
    /// Wakes the producer when a forward demands a value or the run ends.
    wake: Arc<FeedWake>,
    demand_wakes: AtomicU64,
    /// Under `LLOOM_HOST_FEED_TRACE`: the host time ([`trace_now_ns`]) each
    /// value's wait was first encoded, for the run's first [`TRACE_CAP`].
    encoded_ns: Option<Box<[AtomicU64]>>,
}

/// What wakes a parked host-fill producer: a forward demanding a value
/// ([`FeedLink::encode_read`]), the GPU publishing a token (the ring's publish
/// event, through a listener block), or the run ending. A counter under a
/// lock and a condvar: a waker bumps it, a waiter parks only while it is
/// unchanged since it last looked, so a wake between the look and the park is
/// never lost.
#[derive(Debug, Default)]
pub struct FeedWake {
    epoch: parking_lot::Mutex<u64>,
    cv: parking_lot::Condvar,
}

impl FeedWake {
    /// Wake every parked waiter.
    pub fn wake(&self) {
        *self.epoch.lock() += 1;
        self.cv.notify_all();
    }

    /// The wake count now, to park against.
    pub fn epoch(&self) -> u64 {
        *self.epoch.lock()
    }

    /// Park while the count is still `seen`, for at most `timeout`.
    pub fn park(&self, seen: u64, timeout: std::time::Duration) {
        let mut g = self.epoch.lock();
        if *g == seen {
            let _ = self.cv.wait_for(&mut g, timeout);
        }
    }
}

/// Values a traced run keeps encode times for; past it nothing is recorded.
pub const TRACE_CAP: usize = 1 << 14;

/// Nanoseconds on one clock every thread of a run reads (the host fill's
/// trace), never 0.
pub fn trace_now_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos() as u64
        + 1
}

impl FeedLink {
    /// A run whose first decode forward is at `pos0`, over `ring`, whose
    /// event holds the previous run's last value.
    #[cfg(feature = "metal")]
    pub fn new(ring: Arc<lloom_tensor::metal_backend::HostFedRing>, pos0: usize) -> Self {
        let base = ring.base();
        Self {
            ring,
            base,
            pos0,
            fed: AtomicU64::new(0),
            signalled: AtomicU64::new(0),
            demanded: AtomicU64::new(0),
            released: AtomicBool::new(false),
            shift: AtomicU64::new(0),
            retract_req: AtomicU64::new(0),
            retract_ack: AtomicU64::new(0),
            failure: parking_lot::Mutex::new(None),
            wake: Arc::new(FeedWake::default()),
            demand_wakes: AtomicU64::new(0),
            encoded_ns: crate::tuning::tuning()
                .ring
                .feed_trace
                .then(|| (0..TRACE_CAP).map(|_| AtomicU64::new(0)).collect()),
        }
    }

    /// When value `k`'s wait was first encoded, on a traced run.
    pub fn encoded_ns(&self, k: u64) -> Option<u64> {
        let t = self.encoded_ns.as_ref()?;
        match t.get((k as usize).checked_sub(1)?)?.load(Ordering::Acquire) {
            0 => None,
            v => Some(v),
        }
    }

    /// Whether this run is traced (`LLOOM_HOST_FEED_TRACE`).
    pub fn traced(&self) -> bool {
        self.encoded_ns.is_some()
    }

    /// What wakes the run's producer.
    pub fn wake(&self) -> &Arc<FeedWake> {
        &self.wake
    }

    /// Wake the producer for a value a forward now demands, and count it.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    fn wake_for_demand(&self) {
        self.demand_wakes.fetch_add(1, Ordering::Relaxed);
        self.wake.wake();
    }

    /// The demand wakes posted this run.
    pub fn demand_wakes(&self) -> u64 {
        self.demand_wakes.load(Ordering::Relaxed)
    }

    /// The position of the run's first decode forward.
    pub fn pos0(&self) -> usize {
        self.pos0
    }

    /// The loop fed one more token into the command stream (published it
    /// ahead of every forward it encodes after this call).
    pub fn note_fed(&self) {
        self.fed.fetch_add(1, Ordering::Release);
    }

    /// Tokens the loop has fed this run.
    pub fn fed(&self) -> u64 {
        self.fed.load(Ordering::Acquire)
    }

    /// Values the producer has signalled this run.
    pub fn signalled(&self) -> u64 {
        self.signalled.load(Ordering::Acquire)
    }

    /// The highest value a forward of this run waits for.
    pub fn demanded(&self) -> u64 {
        self.demanded.load(Ordering::Acquire)
    }

    /// Whether [`Self::release`] ran: every wait of the run was let through,
    /// whatever its slot held.
    pub fn released(&self) -> bool {
        self.released.load(Ordering::Acquire)
    }

    /// The run-relative value (1-based) the forward at `pos` consumes.
    fn index_at(&self, pos: usize) -> Result<u64> {
        if pos < self.pos0 {
            lloom_tensor::bail!(
                "host feed: a forward at position {pos}, before the run's first decode step {}",
                self.pos0
            )
        }
        Ok((pos - self.pos0) as u64 + 1)
    }

    /// The run-relative value the forward at `pos` waits for: its position's
    /// index, shifted by one per draw the loop took back ([`Self::retract`]).
    pub fn value_at(&self, pos: usize) -> Result<u64> {
        Ok(self.index_at(pos)? + self.shift.load(Ordering::Acquire))
    }

    /// The rows the forward at `pos` reads: a wait for its value, encoded
    /// here, and the `[len]` f32 slot the producer fills. Refused when the
    /// loop has not yet fed the token that value is filled from — a wait
    /// nothing would ever signal.
    pub fn encode_read(&self, pos: usize) -> Result<Tensor> {
        if self.released() {
            lloom_tensor::bail!(
                "host feed: the fill of a run that ended early (its waits were released{}); \
                 the sequence's state past position {} is not the model's",
                match &*self.failure.lock() {
                    Some(why) => format!(": {why}"),
                    None => String::new(),
                },
                self.pos0
            )
        }
        let k = self.value_at(pos)?;
        let fed = self.fed();
        if k > fed {
            lloom_tensor::bail!(
                "host feed: the forward at position {pos} reads token {k} of the run, and the \
                 loop has fed {fed}: its wait would never be signalled"
            )
        }
        self.demanded.fetch_max(k, Ordering::AcqRel);
        if let Some(slot) = self.encoded_ns.as_ref().and_then(|t| t.get(k as usize - 1)) {
            let _ = slot.compare_exchange(0, trace_now_ns(), Ordering::AcqRel, Ordering::Relaxed);
        }
        #[cfg(feature = "metal")]
        {
            let v = self.base + k;
            self.ring.encode_wait(v)?;
            // The producer may be parked: a forward now waits on value `k`, so
            // its token is imminent.
            self.wake_for_demand();
            Ok(self.ring.tensor(v).clone())
        }
        #[cfg(not(feature = "metal"))]
        lloom_tensor::bail!("host feed: no host-fed ring off Metal")
    }

    /// Let every wait of the run through: the event jumps past any value the
    /// run could have encoded. For a run that fails, or ends, with waits its
    /// producer will not fill; the slots those forwards read are stale.
    pub fn release(&self) {
        self.released.store(true, Ordering::Release);
        #[cfg(feature = "metal")]
        self.ring.release(self.base);
        self.wake.wake();
    }

    /// Take back the last token the producer filled (the loop took back its
    /// draw: a provisional step redone behind the forward that consumed it,
    /// `Session::decode`): the producer rewinds its n-gram window and its
    /// reader by one, and every later forward's value shifts by one, so the
    /// forward encoded again waits for the fill of the token drawn again.
    /// Waits, at most `timeout`, for the producer to have done it.
    ///
    /// Precondition: the producer has filled the taken-back token — the
    /// forward that consumed it has completed — and no token after it is
    /// published.
    pub fn retract(&self, timeout: std::time::Duration) -> Result<()> {
        let want = self.retract_req.fetch_add(1, Ordering::AcqRel) + 1;
        self.shift.fetch_add(1, Ordering::AcqRel);
        self.wake.wake();
        let t0 = std::time::Instant::now();
        while self.retract_ack.load(Ordering::Acquire) < want {
            if self.released() {
                lloom_tensor::bail!(
                    "host feed: a draw taken back from a run whose fill ended{}",
                    match &*self.failure.lock() {
                        Some(why) => format!(": {why}"),
                        None => String::new(),
                    }
                )
            }
            if t0.elapsed() > timeout {
                lloom_tensor::bail!(
                    "host feed: the producer did not take a draw back in {timeout:?}"
                )
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
            self.wake.wake();
        }
        Ok(())
    }

    /// Take-backs asked of the producer and not yet done.
    pub fn retracts_pending(&self) -> u64 {
        self.retract_req
            .load(Ordering::Acquire)
            .saturating_sub(self.retract_ack.load(Ordering::Acquire))
    }

    /// The producer did one take-back.
    pub fn ack_retract(&self) {
        self.retract_ack.fetch_add(1, Ordering::AcqRel);
    }

    /// The producer failed with `why`: record it and release every wait.
    pub fn fail(&self, why: &dyn std::fmt::Display) {
        *self.failure.lock() = Some(why.to_string());
        self.release();
    }

    /// The producer filled run value `k`'s slot: signal it.
    fn signal(&self, k: u64) {
        if self.released() {
            return;
        }
        #[cfg(feature = "metal")]
        self.ring.signal(self.base + k);
        self.signalled.store(k, Ordering::Release);
    }

    /// The slot run value `k` fills, for the producer.
    ///
    /// # Safety
    /// As [`lloom_tensor::metal_backend::HostFedRing::slot_mut`]: value `k`'s
    /// token is published, so the forward that read the slot's previous value
    /// (`k - HOST_FILL_SLOTS`) has run; only the producer writes slots.
    #[cfg(feature = "metal")]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot_mut(&self, k: u64) -> &mut [f32] {
        // SAFETY: the caller's contract, forwarded.
        unsafe { self.ring.slot_mut(self.base + k) }
    }
}

/// One run's host fill, as a model hands it to the decode loop
/// (`CausalLM::host_feed`): the shared [`FeedLink`] and the producer's half.
#[derive(Debug)]
pub struct HostFeed {
    link: Arc<FeedLink>,
    fill: NgramFill,
}

impl HostFeed {
    /// The fill of `rows` hashed by `hash`, starting from `window` (the
    /// sequence's window at the run's first decode step), over `link`.
    pub fn new(
        link: Arc<FeedLink>,
        rows: Arc<RowTable>,
        hash: NgramHash,
        window: NgramWindow,
    ) -> Self {
        let n = hash.n_heads();
        Self {
            fill: NgramFill {
                link: Arc::clone(&link),
                rows,
                hash,
                window0: window.clone(),
                window,
                filled: 0,
                ids: vec![0; n],
                scratch: Vec::with_capacity(n + 1),
                consumed: Vec::new(),
            },
            link,
        }
    }

    pub fn link(&self) -> &Arc<FeedLink> {
        &self.link
    }

    pub fn into_parts(self) -> (Arc<FeedLink>, NgramFill) {
        (self.link, self.fill)
    }
}

/// The producer's half of a [`HostFeed`]: fills one slot per token the loop
/// published, in order.
#[derive(Debug)]
pub struct NgramFill {
    link: Arc<FeedLink>,
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    rows: Arc<RowTable>,
    hash: NgramHash,
    window: NgramWindow,
    /// The window the run began with: a take-back rebuilds from it.
    window0: NgramWindow,
    filled: u64,
    ids: Vec<i64>,
    scratch: Vec<Option<u32>>,
    consumed: Vec<u32>,
}

impl NgramFill {
    /// The next published token of the run: hash its window, gather its rows
    /// into the next value's slot, advance the window, signal. A run's tokens
    /// must arrive in the order they were published.
    pub fn fill(&mut self, token: u32) -> Result<()> {
        let k = self.filled + 1;
        self.window.window_into(Some(token), &mut self.scratch);
        self.hash.row_ids(&self.scratch, &mut self.ids);
        self.gather_into_slot(k)?;
        self.window.push(Some(token));
        self.consumed.push(token);
        self.filled = k;
        self.link.signal(k);
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn gather_into_slot(&self, k: u64) -> Result<()> {
        // SAFETY: token `k` is published (the caller read it off the ring's
        // stamp), so the forward that read value `k - 2`'s slot — two tokens
        // back — has run; this thread is the only writer.
        let slot = unsafe { self.link.slot_mut(k) };
        self.rows.gather_into(&self.ids, slot)
    }

    #[cfg(not(feature = "metal"))]
    fn gather_into_slot(&self, _k: u64) -> Result<()> {
        lloom_tensor::bail!("host feed: no host-fed ring off Metal")
    }

    /// Take back the last token filled ([`FeedLink::retract`]): out of the
    /// tokens consumed and the n-gram window. The fill count stays — it is
    /// the last value signalled, and the next fill signals the one after
    /// it. A rebuild from the run's first window, on the rare path.
    pub fn rewind_one(&mut self) -> Result<()> {
        if self.consumed.pop().is_none() {
            lloom_tensor::bail!("host feed: a take-back with nothing filled")
        }
        self.window = self.window0.clone();
        for &t in &self.consumed {
            self.window.push(Some(t));
        }
        Ok(())
    }

    /// Slots filled so far this run.
    pub fn filled(&self) -> u64 {
        self.filled
    }

    /// The tokens filled, in order: what the run's decode forwards consumed,
    /// and possibly the tokens past its last forward.
    pub fn into_consumed(self) -> Vec<u32> {
        self.consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::{gguf_file, QTensor};

    /// A `[rows, head_dim]` table written as `dtype` to a temporary GGUF:
    /// f32 values distinct per element, or IQ4_NL blocks built byte by byte
    /// (the type is read-only here) with a finite scale and varied nibbles.
    fn table_file(tag: &str, rows: usize, head_dim: usize, dtype: GgmlDType) -> std::path::PathBuf {
        use lloom_tensor::quantized::QStorage;
        let q = match dtype {
            GgmlDType::F32 => {
                let vals: Vec<f32> = (0..rows * head_dim)
                    .map(|i| ((i * 37 % 1009) as f32 - 504.0) * 0.01)
                    .collect();
                let t = Tensor::from_vec(vals, (rows, head_dim), &Device::Cpu).unwrap();
                QTensor::quantize(&t, dtype).unwrap()
            }
            GgmlDType::IQ4NL => {
                let blocks = rows * head_dim / dtype.block_size();
                let mut bytes = Vec::with_capacity(blocks * dtype.type_size());
                for b in 0..blocks {
                    let d = half::f16::from_f32(0.01 + (b % 13) as f32 * 0.003);
                    bytes.extend_from_slice(&d.to_le_bytes());
                    bytes.extend((0..16).map(|j| ((b * 7 + j * 11) % 256) as u8));
                }
                let st = QStorage::from_data(std::borrow::Cow::Owned(bytes), &Device::Cpu, dtype)
                    .unwrap();
                QTensor::new(st, (rows, head_dim)).unwrap()
            }
            GgmlDType::MXFP4 | GgmlDType::FP8E4M3 => {
                let vals: Vec<f32> = (0..rows * head_dim)
                    .map(|i| ((i * 37 % 1009) as f32 - 504.0) * 0.001 * (1 + i % 7) as f32)
                    .collect();
                let t = Tensor::from_vec(vals, (rows, head_dim), &Device::Cpu).unwrap();
                QTensor::quantize(&t, dtype).unwrap()
            }
            other => panic!("no fixture for {other:?}"),
        };
        let path =
            std::env::temp_dir().join(format!("lloom-ngram-{}-{tag}.gguf", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        gguf_file::write(&mut f, &[], &[(PER_LAYER_TOKEN_EMBD, &q)]).unwrap();
        path
    }

    /// Every read policy gathers the rows the map holds: the concurrent path
    /// (`pread` on the pool, landed by index) against the serial map read,
    /// on a scattered id set with a row repeated, at f32 and at IQ4_NL.
    /// Pins the pool path's addressing: a row read at the wrong offset or
    /// landed in the wrong slot differs from the map's.
    /// A token the loop takes back ([`FeedLink::retract`]) is filled again as
    /// if the run had never seen it: the producer's handshake rewinds the
    /// window, the token drawn again fills the NEXT value, whose rows are
    /// those a run that drew it the first time fills, the consumed tokens
    /// leave the taken-back one out, and the forward at the taken-back
    /// position waits for the new value, never the one the wrong token
    /// signalled.
    #[cfg(feature = "metal")]
    #[test]
    fn a_taken_back_token_is_filled_again_as_if_never_filled() {
        use lloom_tensor::metal_backend::HostFedRing;
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let m = dev.as_metal_device().unwrap().clone();
        let head_dim = 8;
        let path = table_file("takeback", 80, head_dim, GgmlDType::F32);
        let rows = Arc::new(RowTable::Mmap(MmapRows::open(&path).unwrap()));
        let hash = NgramHash::new(
            vec![1_000_003, 999_983, 7],
            vec![13, 17, 19, 23],
            vec![0, 13, 30, 49],
            2,
            0,
        )
        .unwrap();
        let len = hash.n_heads() * head_dim;
        let slot =
            |ring: &HostFedRing, v: u64| -> Vec<f32> { ring.tensor(v).to_vec1::<f32>().unwrap() };
        let feed = |ring: &Arc<HostFedRing>| {
            let link = Arc::new(FeedLink::new(Arc::clone(ring), 10));
            HostFeed::new(
                link,
                Arc::clone(&rows),
                hash.clone(),
                NgramWindow::new(hash.context_len()),
            )
            .into_parts()
        };
        // A fourth token's rows hash the three before it: the reference
        // fills 5, 6, 9, 11; the control 5, 6, 7, 11.
        let fill_all = |tokens: &[u32]| -> (Vec<f32>, Vec<u32>) {
            let ring = Arc::new(HostFedRing::new(&m, HOST_FILL_SLOTS, len).unwrap());
            let base = ring.base();
            let (_, mut fill) = feed(&ring);
            for &t in tokens {
                fill.fill(t).unwrap();
            }
            (
                slot(&ring, base + tokens.len() as u64),
                fill.into_consumed(),
            )
        };
        let (want, _) = fill_all(&[5, 6, 9, 11]);
        let (control, _) = fill_all(&[5, 6, 7, 11]);
        assert_ne!(
            want, control,
            "11's rows do not see the token before it: no test"
        );
        // 5, 6, 7 — then 7 taken back, and 9, 11 drawn after it.
        let ring = Arc::new(HostFedRing::new(&m, HOST_FILL_SLOTS, len).unwrap());
        let base = ring.base();
        let (link, mut fill) = feed(&ring);
        for t in [5, 6, 7] {
            fill.fill(t).unwrap();
        }
        assert_eq!(link.value_at(12).unwrap(), 3);
        // The loop's side of the handshake on a thread, the producer's here.
        let l = Arc::clone(&link);
        let asked = std::thread::spawn(move || l.retract(std::time::Duration::from_secs(5)));
        while link.retracts_pending() == 0 {
            std::thread::yield_now();
        }
        fill.rewind_one().unwrap();
        link.ack_retract();
        asked.join().unwrap().unwrap();
        assert_eq!(
            link.value_at(12).unwrap(),
            4,
            "the forward at the taken-back position waits for the next value"
        );
        assert_eq!(link.value_at(13).unwrap(), 5);
        fill.fill(9).unwrap();
        fill.fill(11).unwrap();
        assert_eq!(fill.filled(), 5);
        assert_eq!(
            slot(&ring, base + 5),
            want,
            "the token after the one drawn again hashed the taken-back token"
        );
        assert_eq!(fill.into_consumed(), vec![5, 6, 9, 11]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn every_read_policy_gathers_the_same_rows() {
        for (dtype, head_dim) in [(GgmlDType::F32, 8usize), (GgmlDType::IQ4NL, 160)] {
            let path = table_file(&format!("{dtype:?}"), 300, head_dim, dtype);
            let t = MmapRows::open(&path).unwrap();
            let ids: Vec<i64> = vec![
                0, 299, 17, 150, 17, 3, 298, 64, 65, 1, 222, 9, 100, 5, 77, 31,
            ];
            let mut want = vec![0f32; ids.len() * head_dim];
            t.set_read_policy(ReadPolicy::Serial);
            t.gather_into(&ids, &mut want).unwrap();
            // The reference is the file's rows, one by one.
            for (i, &id) in ids.iter().enumerate() {
                let mut row = vec![0f32; head_dim];
                t.dequant_packed(t.packed_row(id).unwrap(), &mut row)
                    .unwrap();
                assert_eq!(row, want[i * head_dim..(i + 1) * head_dim], "row {id}");
            }
            for policy in [ReadPolicy::Concurrent, ReadPolicy::Adaptive] {
                t.set_read_policy(policy);
                let mut got = vec![0f32; ids.len() * head_dim];
                t.gather_into(&ids, &mut got).unwrap();
                assert!(got == want, "{policy:?} differs at {dtype:?}");
            }
            let (_, pooled) = t.read_counters();
            assert!(
                pooled >= ids.len() as u64,
                "the concurrent policy must read through the pool: {pooled}"
            );
            std::fs::remove_file(&path).ok();
        }
    }

    /// A table stored in a float-code kind (the FP8 checkpoint stores its
    /// n-gram table as E4M3 under one bf16 scale) gathers, row by row, what
    /// the kind's own `to_float` of the whole table holds — an absolute
    /// reference, not the row path compared with itself.
    #[test]
    fn a_float_code_table_gathers_its_to_float_rows() {
        for dtype in [GgmlDType::MXFP4, GgmlDType::FP8E4M3] {
            let (rows, head_dim) = (300usize, 160usize);
            let path = table_file(&format!("fp-{dtype:?}"), rows, head_dim, dtype);
            let mut f = std::fs::File::open(&path).unwrap();
            let content = gguf_file::Content::read(&mut f).unwrap();
            let whole = content
                .tensor(&mut f, PER_LAYER_TOKEN_EMBD, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let t = MmapRows::open(&path).unwrap();
            let ids: Vec<i64> = vec![0, 299, 17, 150, 17, 3];
            let mut got = vec![0f32; ids.len() * head_dim];
            t.gather_into(&ids, &mut got).unwrap();
            for (i, &id) in ids.iter().enumerate() {
                let want = &whole[id as usize * head_dim..(id as usize + 1) * head_dim];
                assert_eq!(
                    &got[i * head_dim..(i + 1) * head_dim],
                    want,
                    "{dtype:?} row {id}"
                );
            }
            assert!(whole.iter().any(|v| *v != 0.0), "{dtype:?}: an empty table");
            std::fs::remove_file(&path).ok();
        }
    }

    /// The cache's reads-first order keeps the one-at-a-time semantics: the
    /// same hits and misses as a lookup-by-lookup reference LRU, including a
    /// row evicted by an insert of the same call and asked for again.
    #[test]
    fn the_cache_counts_as_one_lookup_at_a_time() {
        let path = table_file("lru", 64, 8, GgmlDType::F32);
        let cached = CachedRows::new(MmapRows::open(&path).unwrap(), 4).unwrap();
        cached.inner.set_read_policy(ReadPolicy::Concurrent);
        let plain = MmapRows::open(&path).unwrap();
        // Six distinct ids through a 4-row cache, then the first again: the
        // first is evicted within the call and must miss twice.
        let calls: Vec<Vec<i64>> = vec![vec![1, 2, 3, 4, 5, 6, 1], vec![6, 5, 9, 1, 1]];
        let (mut ref_hits, mut ref_misses) = (0u64, 0u64);
        let mut order: std::collections::VecDeque<i64> = Default::default();
        for ids in &calls {
            let mut a = vec![0f32; ids.len() * 8];
            let mut b = vec![0f32; ids.len() * 8];
            cached.gather_into(ids, &mut a).unwrap();
            plain.gather_into(ids, &mut b).unwrap();
            assert!(a == b, "cached rows differ from the map on {ids:?}");
            for &id in ids {
                if let Some(p) = order.iter().position(|&x| x == id) {
                    order.remove(p);
                    ref_hits += 1;
                } else {
                    ref_misses += 1;
                    if order.len() == 4 {
                        order.pop_back();
                    }
                }
                order.push_front(id);
            }
            // After each call, not only at the end: the two calls' totals
            // coincide between LRU and MRU — (0, 7) + (4, 1) and (1, 6) +
            // (3, 2) both sum to (4, 8) — so a sum-only check cannot tell
            // the policies apart (found by a displace-the-most-recent
            // mutant, #892).
            assert_eq!(cached.stats(), (ref_hits, ref_misses), "after {ids:?}");
        }
        std::fs::remove_file(&path).ok();
    }
}
