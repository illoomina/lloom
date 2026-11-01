//! The host fill allocates nothing per token (`CONTRIBUTING.md` §3.3a): an
//! n-gram table's read through a caller's scratch on the fixed read threads,
//! hedged or not, and the producer's whole fill (`NgramFill::fill`: hash,
//! read, dequantize into the staging slot, signal) count zero allocations
//! over N calls, on every thread, the read threads included.
//!
//! The counter is process-wide, because the reads land on other threads, so
//! this file is its own binary with one test. Its positive control is the
//! same read without the scratch, which allocates the cold rows' indices and
//! bytes per call: a counter that reads 0 there is not counting.
//!
//! With debug assertions on, objc2 checks every message send's signature
//! against the runtime's (`objc2` `runtime/message_receiver.rs`,
//! `msg_send_check`), and parsing a pointer type in it allocates: the slot's
//! `contents` (a `void *`, sent by `Buffer::data`) is then one allocation a
//! token that the shipping build does not make (`setSignaledValue:` has no
//! pointer and allocates nothing). So the fill is held to the count of the
//! same two Metal calls made bare beside it, and to zero where debug
//! assertions are off.

use lloom::nn::ngram_memory::{
    read_worker_stats, MmapRows, NgramHash, ReadPolicy, ReadPool, ReadScratch, PER_LAYER_TOKEN_EMBD,
};
use lloom_tensor::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use lloom_tensor::{Device, Result};
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAllocator;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call is forwarded to the system allocator unchanged; the
// counter is the only addition.
unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn allocs() -> u64 {
    ALLOCS.load(Ordering::SeqCst)
}

const HEAD_DIM: usize = 160;
const HEADS_PER_NGRAM: usize = 8;
const HEAD_ROWS: i64 = 257;

/// Sixteen heads of 257 rows each: the checkpoint's shape (16 rows a token,
/// IQ4_NL rows of 160) over a table small enough for a test.
fn hash() -> Result<NgramHash> {
    let n = 2 * HEADS_PER_NGRAM;
    NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![HEAD_ROWS; n],
        (0..n as i64).map(|h| h * HEAD_ROWS).collect(),
        HEADS_PER_NGRAM,
        0,
    )
}

fn table_file() -> Result<std::path::PathBuf> {
    let rows = 2 * HEADS_PER_NGRAM * HEAD_ROWS as usize;
    let dtype = GgmlDType::IQ4NL;
    let blocks = rows * HEAD_DIM / dtype.block_size();
    let mut bytes = Vec::with_capacity(blocks * dtype.type_size());
    for b in 0..blocks {
        let d = half::f16::from_f32(0.01 + (b % 13) as f32 * 0.003);
        bytes.extend_from_slice(&d.to_le_bytes());
        bytes.extend((0..16).map(|j| ((b * 7 + j * 11) % 256) as u8));
    }
    let st = QStorage::from_data(std::borrow::Cow::Owned(bytes), &Device::Cpu, dtype)?;
    let q = QTensor::new(st, (rows, HEAD_DIM))?;
    let path = std::env::temp_dir().join(format!("lloom-ngram-alloc-{}.gguf", std::process::id()));
    gguf_file::write(
        &mut std::fs::File::create(&path)?,
        &[],
        &[(PER_LAYER_TOKEN_EMBD, &q)],
    )?;
    Ok(path)
}

/// `n` tokens' row ids, as the hash gives them over a window that walks a
/// fixed token stream.
fn token_rows(hash: &NgramHash, n: usize) -> Vec<Vec<i64>> {
    let mut w = lloom::nn::ngram_memory::NgramWindow::new(hash.context_len());
    (0..n)
        .map(|i| {
            let t = ((i * 7919 + 13) % 50_000) as u32;
            let mut ids = vec![0i64; hash.n_heads()];
            hash.row_ids(&w.window(Some(t)), &mut ids);
            w.push(Some(t));
            ids
        })
        .collect()
}

#[test]
fn the_host_fill_allocates_nothing_per_token() -> Result<()> {
    // More threads than a token's rows, so that a hedge at once finds idle
    // threads to re-read on (in a decode the threads whose rows landed are
    // the idle ones).
    assert!(lloom::nn::ngram_memory::start_read_workers(32));
    let path = table_file()?;
    let hash = hash()?;
    let sets = token_rows(&hash, 256);
    let t = MmapRows::open(&path)?;
    // Every row through the read threads: the page cache holds this small
    // file, so `Adaptive` would read it through the map.
    t.set_read_policy(ReadPolicy::Concurrent);
    t.set_read_pool(ReadPool::Workers);
    t.set_hedge_us(0);
    let mut scratch = ReadScratch::with_rows(hash.n_heads(), t.row_bytes());
    let mut out = vec![0f32; hash.n_heads() * HEAD_DIM];
    // Warm: the read threads start, the lazy statics settle.
    t.gather_into_with(&sets[0], &mut out, &mut scratch)?;

    let a0 = allocs();
    for ids in &sets {
        t.gather_into_with(ids, &mut out, &mut scratch)?;
    }
    let reads = allocs() - a0;

    // Every row in flight hedged at once, on the 16 idle threads: the second
    // reads allocate nothing either.
    t.set_hedge_us(1);
    let h0 = read_worker_stats().unwrap_or_default();
    let a0 = allocs();
    for ids in &sets {
        t.gather_into_with(ids, &mut out, &mut scratch)?;
    }
    let hedged_reads = allocs() - a0;
    let h1 = read_worker_stats().unwrap_or_default();
    t.set_hedge_us(0);

    // The positive control: the same reads without the scratch.
    let a0 = allocs();
    for ids in &sets {
        t.gather_into(ids, &mut out)?;
    }
    let control = allocs() - a0;

    // rayon's pool, reported, not asserted: the library's own bookkeeping.
    t.set_read_pool(ReadPool::Rayon);
    let a0 = allocs();
    for ids in &sets {
        t.gather_into_with(ids, &mut out, &mut scratch)?;
    }
    let rayon = allocs() - a0;
    t.set_read_pool(ReadPool::Workers);

    println!(
        "ALLOCS reads={reads} hedged={hedged_reads} (re-read {} rows, {} won) control={control} rayon={rayon} over {} tokens",
        h1.hedged - h0.hedged,
        h1.hedge_wins - h0.hedge_wins,
        sets.len()
    );
    assert!(
        control >= 2 * sets.len() as u64,
        "the scratch-less read counted {control} allocations over {} tokens: the counter is not counting",
        sets.len()
    );
    assert_eq!(reads, 0, "the read threads' reads allocated");
    assert!(
        h1.hedged > h0.hedged,
        "a 1 µs hedge re-read nothing: {h1:?}"
    );
    assert_eq!(hedged_reads, 0, "the hedged reads allocated");

    #[cfg(feature = "metal")]
    fill_on_metal(&path, &hash)?;
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// The producer's fill, start to signal, over a real staging ring.
#[cfg(feature = "metal")]
fn fill_on_metal(path: &std::path::Path, hash: &NgramHash) -> Result<()> {
    use lloom::nn::ngram_memory::{FeedLink, HostFeed, NgramWindow, RowTable, HOST_FILL_SLOTS};
    use lloom_tensor::metal_backend::HostFedRing;
    use std::sync::Arc;
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let m = dev.as_metal_device()?.clone();
    let rows = Arc::new(RowTable::Mmap(MmapRows::open(path)?));
    let mm = rows.mmap().expect("a mapped table");
    mm.set_read_policy(ReadPolicy::Concurrent);
    mm.set_read_pool(ReadPool::Workers);
    mm.set_hedge_us(0);
    let ring = Arc::new(HostFedRing::new(
        &m,
        HOST_FILL_SLOTS,
        hash.n_heads() * HEAD_DIM,
    )?);
    let link = Arc::new(FeedLink::new(ring, 0));
    let (_link, mut fill) = HostFeed::new(
        link,
        rows,
        hash.clone(),
        NgramWindow::new(hash.context_len()),
    )
    .into_parts();
    let tokens: Vec<u32> = (0..258u32).map(|i| (i * 7919 + 13) % 50_000).collect();
    fill.fill(tokens[0])?;
    fill.fill(tokens[1])?;
    let a0 = allocs();
    for &t in &tokens[2..] {
        fill.fill(t)?;
    }
    let fills = allocs() - a0;
    let n = tokens.len() - 2;
    // The fill's two Metal calls, the slot and the signal, bare, as many
    // times, on a ring of its own.
    let other = HostFedRing::new(&m, HOST_FILL_SLOTS, hash.n_heads() * HEAD_DIM)?;
    let base = other.base();
    other.signal(base + 1);
    let a0 = allocs();
    for k in 0..n as u64 {
        // SAFETY: nothing reads this ring; this thread is its only writer.
        let slot = unsafe { other.slot_mut(base + 2 + k) };
        std::hint::black_box(slot.as_mut_ptr());
        other.signal(base + 2 + k);
    }
    let calls = allocs() - a0;
    println!(
        "ALLOCS fill={fills} metal_calls={calls} over {n} tokens (debug assertions {})",
        cfg!(debug_assertions)
    );
    assert_eq!(
        fills, calls,
        "the producer's fill allocated beyond its slot's and its signal's Metal calls"
    );
    if !cfg!(debug_assertions) {
        assert_eq!(fills, 0, "the producer's fill allocated");
    }
    assert_eq!(fill.filled(), tokens.len() as u64);
    Ok(())
}
