//! An n-gram memory's table: hashed n-gram embedding rows, read a handful per
//! token from a table far too large to be resident — Qwen4-exp's N-gram PLE
//! (320 M rows, 28.8 GB) and the Engram shape DeepSeek-V4.1 shares with it
//! (`design/docs/expert-streaming.md` §20.1a, §20.13). The family owns its
//! hash and the block that consumes the rows; this is the storage: where the
//! rows live and how one token's rows are read.
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

use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::{GgmlDType, GgmlType};
use lloom_tensor::{Device, Result, Tensor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

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

/// The pool cold rows are read on, built on first use.
#[cfg(unix)]
fn read_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(READ_THREADS)
            .thread_name(|i| format!("ngram-read-{i}"))
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

    fn contains(&self, id: i64) -> bool {
        self.slot_of.contains_key(&id)
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
            GgmlDType::IQ4NL | GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {}
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
        }
        assert_eq!(cached.stats(), (ref_hits, ref_misses));
        std::fs::remove_file(&path).ok();
    }
}
