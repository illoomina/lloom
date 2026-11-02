//! The host fill allocates nothing per token (`CONTRIBUTING.md` §3.3a): an
//! n-gram table's read through a caller's scratch on the fixed read threads,
//! hedged or not, and the producer's whole fill (`NgramFill::fill`: hash,
//! read, dequantize into the staging slot, signal) count zero allocations
//! over N calls, on every thread, the read threads included.
//!
//! The counter is process-wide, because the reads land on other threads, so
//! this file is its own binary with one test. Its positive control is the
//! same read without the scratch, which allocates the cold rows' indices and
//! bytes per call: a counter that reads 0 there is not counting. Each count
//! is also attributed to the threads that made it (reported, not asserted;
//! CONTRIBUTING §3.1b), so a red names its thread.
//!
//! A thread's one-time setup is not the window's: the Metal device starts
//! the shared retirement worker, and its first park sets up parking_lot's
//! per-thread data, which allocates whenever the worker is first scheduled.
//! Under load that landed inside the fill's window (#1205, census row 1), so
//! the fill's window opens only once the worker is parked
//! (`lloom_tensor::retirement_worker_parked`).
//!
//! The hedged reads are not left to the host's read latency: one row of each
//! hedged request is held on its read thread until the request's hedge has
//! run (`hold_read_worker_row`), so the hedge always has a row to re-read.
//! One row at a time, under each winner, the held row lands once: the hedge
//! re-reads it, the winning read lands it, and the loser is not counted.
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
    hold_read_worker_row, read_worker_hold_stats, read_worker_stats, settle_read_workers,
    HeldWinner, HoldStats, MmapRows, NgramHash, ReadPolicy, ReadPool, ReadScratch,
    PER_LAYER_TOKEN_EMBD,
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
        by_thread::charge();
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        by_thread::charge();
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

/// Each counted allocation charged to its thread, in a fixed table that
/// allocates nothing: a slot per thread (by `pthread_self`), its name, its
/// count. Read at a window's two ends and reported beside the window's total.
mod by_thread {
    use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
    pub const SLOTS: usize = 256;
    const NAME: usize = 32;
    static TIDS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
    static COUNTS: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
    static NAMES: [[AtomicU8; NAME]; SLOTS] = [const { [const { AtomicU8::new(0) }; NAME] }; SLOTS];
    /// Allocations on threads past the table's slots.
    static OVERFLOW: AtomicU64 = AtomicU64::new(0);
    /// The test's own thread, named so in the report.
    pub static TEST: AtomicUsize = AtomicUsize::new(0);

    pub fn me() -> usize {
        // SAFETY: takes no arguments; returns the calling thread's handle.
        unsafe { libc::pthread_self() as usize }
    }

    pub fn charge() {
        let tid = me();
        for i in 0..SLOTS {
            let t = TIDS[i].load(Ordering::Acquire);
            let mine = t == tid
                || (t == 0
                    && TIDS[i]
                        .compare_exchange(0, tid, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok());
            if mine {
                if NAMES[i][0].load(Ordering::Relaxed) == 0 {
                    let mut buf = [0u8; NAME];
                    // SAFETY: `tid` is the calling thread, `buf` holds NAME bytes.
                    unsafe {
                        libc::pthread_getname_np(
                            tid as libc::pthread_t,
                            buf.as_mut_ptr().cast(),
                            NAME,
                        )
                    };
                    for (j, b) in buf.iter().enumerate() {
                        NAMES[i][j].store(*b, Ordering::Relaxed);
                    }
                }
                COUNTS[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        OVERFLOW.fetch_add(1, Ordering::Relaxed);
    }

    /// Every slot's count and the overflow, copied without allocating.
    pub fn counts() -> ([u64; SLOTS], u64) {
        (
            std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed)),
            OVERFLOW.load(Ordering::Relaxed),
        )
    }

    /// `thread=count` for every thread that allocated between `a` and `b`.
    pub fn between(a: &([u64; SLOTS], u64), b: &([u64; SLOTS], u64)) -> String {
        let test = TEST.load(Ordering::Relaxed);
        let mut parts = Vec::new();
        for i in 0..SLOTS {
            let d = b.0[i] - a.0[i];
            if d > 0 {
                let who = if TIDS[i].load(Ordering::Relaxed) == test {
                    "test".to_string()
                } else {
                    let n: Vec<u8> = NAMES[i]
                        .iter()
                        .map(|c| c.load(Ordering::Relaxed))
                        .take_while(|&c| c != 0)
                        .collect();
                    String::from_utf8_lossy(&n).into_owned()
                };
                parts.push(format!("{who:?}={d}"));
            }
        }
        if b.1 > a.1 {
            parts.push(format!("unattributed={}", b.1 - a.1));
        }
        if parts.is_empty() {
            "none".into()
        } else {
            parts.join(" ")
        }
    }
}

fn allocs() -> u64 {
    ALLOCS.load(Ordering::SeqCst)
}

const HEAD_DIM: usize = 160;
const HEADS_PER_NGRAM: usize = 8;
const HEAD_ROWS: i64 = 257;
/// The row of each hedged request whose first read waits for the hedge.
const HELD: usize = 5;

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
    by_thread::TEST.store(by_thread::me(), Ordering::Relaxed);
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

    let r0 = read_worker_stats().unwrap_or_default();
    let c0 = by_thread::counts();
    let a0 = allocs();
    for ids in &sets {
        t.gather_into_with(ids, &mut out, &mut scratch)?;
    }
    let reads = allocs() - a0;
    let c1 = by_thread::counts();
    let r1 = read_worker_stats().unwrap_or_default();
    let reads_by = by_thread::between(&c0, &c1);

    // Every row in flight hedged at once, on the idle threads: the second
    // reads allocate nothing either. Row `HELD` waits for its request's
    // hedge, and with the threads settled the first request's hedge finds
    // the 16 it did not use idle, so it re-reads that row whatever the
    // host's read latency.
    let n = hash.n_heads() as u32;
    assert_eq!(
        settle_read_workers(),
        Some(n),
        "an unhedged request counted its rows other than once each"
    );
    assert!(hold_read_worker_row(Some((HELD, HeldWinner::Hedge))));
    t.set_hedge_us(1);
    let h0 = read_worker_stats().unwrap_or_default();
    let c0 = by_thread::counts();
    let a0 = allocs();
    for ids in &sets {
        t.gather_into_with(ids, &mut out, &mut scratch)?;
    }
    let hedged_reads = allocs() - a0;
    let c1 = by_thread::counts();
    let hedged_by = by_thread::between(&c0, &c1);
    let h1 = read_worker_stats().unwrap_or_default();
    let held = read_worker_hold_stats().unwrap_or_default();
    hold_read_worker_row(None);
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

    // A request the read threads refuse is read on rayon's pool, which
    // allocates: the refusals are reported beside each count.
    println!(
        "ALLOCS reads={reads} (refused {}) hedged={hedged_reads} (refused {}; re-read {} rows, {} won; {held:?}) control={control} rayon={rayon} over {} tokens",
        r1.refused - r0.refused,
        h1.refused - h0.refused,
        h1.hedged - h0.hedged,
        h1.hedge_wins - h0.hedge_wins,
        sets.len()
    );
    println!("ALLOCS BY THREAD reads: {reads_by} | hedged: {hedged_by}");
    assert!(
        control >= 2 * sets.len() as u64,
        "the scratch-less read counted {control} allocations over {} tokens: the counter is not counting",
        sets.len()
    );
    assert_eq!(reads, 0, "the read threads' reads allocated: {reads_by}");
    assert_eq!(
        held.gave_up,
        0,
        "a held read's hedge did not run within {:?}: {held:?}",
        lloom::nn::ngram_memory::HOLD_BOUND
    );
    assert!(
        h1.hedged > h0.hedged,
        "a 1 µs hedge re-read nothing: {h1:?}"
    );
    assert!(
        held.rehedged >= 1 && h1.hedge_wins > h0.hedge_wins,
        "the held row was not re-read and landed by its hedge: {held:?}, {h1:?}"
    );
    assert_eq!(hedged_reads, 0, "the hedged reads allocated: {hedged_by}");

    a_held_row_lands_once(&t, &sets, &mut scratch)?;

    #[cfg(feature = "metal")]
    fill_on_metal(&path, &hash)?;
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// One row a request, its first read held, under each winner: the hedge
/// re-reads it once, the winner lands it, and the loser writes nothing the
/// caller reads and is not counted (the request's count is 1 once every
/// read has finished). The reference is the row read through the map.
fn a_held_row_lands_once(t: &MmapRows, sets: &[Vec<i64>], scratch: &mut ReadScratch) -> Result<()> {
    let ids: Vec<i64> = sets.iter().take(4).map(|s| s[HELD]).collect();
    let mut want = vec![0f32; HEAD_DIM];
    let mut got = vec![0f32; HEAD_DIM];
    t.set_hedge_us(1);
    for winner in [HeldWinner::Hedge, HeldWinner::First] {
        for &id in &ids {
            t.set_read_policy(ReadPolicy::Serial);
            t.gather_into(&[id], &mut want)?;
            t.set_read_policy(ReadPolicy::Concurrent);
            settle_read_workers();
            assert!(hold_read_worker_row(Some((0, winner))));
            let s0 = read_worker_stats().unwrap_or_default();
            got.fill(f32::NAN);
            t.gather_into_with(&[id], &mut got, scratch)?;
            let done = settle_read_workers();
            let s1 = read_worker_stats().unwrap_or_default();
            let held = read_worker_hold_stats().unwrap_or_default();
            hold_read_worker_row(None);
            let tag = format!("row {id}, the {winner:?} read to win");
            assert_eq!(
                (s1.requests - s0.requests, s1.refused - s0.refused),
                (1, 0),
                "{tag}: the read threads did not take the request"
            );
            assert_eq!(
                held,
                HoldStats {
                    held: 1,
                    rehedged: 1,
                    gave_up: 0
                },
                "{tag}: the hedge did not re-read the held row with its other read waiting"
            );
            assert_eq!(
                s1.hedged - s0.hedged,
                1,
                "{tag}: the hedge re-read other than the held row"
            );
            assert_eq!(
                s1.hedge_wins - s0.hedge_wins,
                u64::from(winner == HeldWinner::Hedge),
                "{tag}: the other read landed it"
            );
            assert_eq!(done, Some(1), "{tag}: the row was counted other than once");
            assert!(
                got.iter()
                    .map(|v| v.to_bits())
                    .eq(want.iter().map(|v| v.to_bits())),
                "{tag}: the row landed is not the file's"
            );
        }
    }
    t.set_hedge_us(0);
    println!(
        "HELD a held row landed once, {} rows under each winner",
        ids.len()
    );
    Ok(())
}

/// How long the fill waits for the retirement worker to park.
#[cfg(feature = "metal")]
const PARK_BOUND: std::time::Duration = std::time::Duration::from_secs(10);

/// Whether the shared retirement worker was parked within `bound`
/// (`lloom_tensor::retirement_worker_parked`): `None` when no worker exists.
#[cfg(feature = "metal")]
fn retirement_worker_parked_within(bound: std::time::Duration) -> Option<bool> {
    let start = std::time::Instant::now();
    loop {
        match lloom_tensor::retirement_worker_parked() {
            Some(false) if start.elapsed() < bound => std::thread::yield_now(),
            other => return other,
        }
    }
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
    // The device started the shared retirement worker, whose first park
    // allocates its parking setup on its own thread when it is first
    // scheduled. Open the window once it is parked, so that is behind it.
    let parked = retirement_worker_parked_within(PARK_BOUND);
    let r0 = read_worker_stats().unwrap_or_default();
    let c0 = by_thread::counts();
    let a0 = allocs();
    for &t in &tokens[2..] {
        fill.fill(t)?;
    }
    let fills = allocs() - a0;
    let c1 = by_thread::counts();
    let r1 = read_worker_stats().unwrap_or_default();
    let fills_by = by_thread::between(&c0, &c1);
    let n = tokens.len() - 2;
    // The fill's two Metal calls, the slot and the signal, bare, as many
    // times, on a ring of its own.
    let other = HostFedRing::new(&m, HOST_FILL_SLOTS, hash.n_heads() * HEAD_DIM)?;
    let base = other.base();
    other.signal(base + 1);
    let c0 = by_thread::counts();
    let a0 = allocs();
    for k in 0..n as u64 {
        // SAFETY: nothing reads this ring; this thread is its only writer.
        let slot = unsafe { other.slot_mut(base + 2 + k) };
        std::hint::black_box(slot.as_mut_ptr());
        other.signal(base + 2 + k);
    }
    let calls = allocs() - a0;
    let c1 = by_thread::counts();
    let calls_by = by_thread::between(&c0, &c1);
    println!(
        "ALLOCS fill={fills} (refused {}) metal_calls={calls} over {n} tokens (debug assertions {})",
        r1.refused - r0.refused,
        cfg!(debug_assertions)
    );
    println!(
        "ALLOCS BY THREAD fill: {fills_by} | metal_calls: {calls_by} | retirement worker parked: {parked:?}"
    );
    assert_eq!(
        parked,
        Some(true),
        "the retirement worker was not parked within {PARK_BOUND:?} of the device's start \
         (None: the device started no worker)"
    );
    assert_eq!(
        fills, calls,
        "the producer's fill allocated beyond its slot's and its signal's Metal calls: \
         fill {fills_by} | calls {calls_by}"
    );
    if !cfg!(debug_assertions) {
        assert_eq!(fills, 0, "the producer's fill allocated");
    }
    assert_eq!(fill.filled(), tokens.len() as u64);
    Ok(())
}
