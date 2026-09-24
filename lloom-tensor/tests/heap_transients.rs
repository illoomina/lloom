//! Pool misses served from heap chunks (`LLOOM_TRANSIENT_HEAPS`, lloom #749
//! P26, `DESIGN.md` §6.3k): the residency set's traffic does not scale with
//! the misses (one add and one commit per CHUNK), the sub-buffers are
//! distinct and computed on correctly, an evicted sub-buffer's range comes
//! back (the chunk count is bounded), and the counters read what the code
//! did.
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Tensor};

#[test]
fn heap_chunks_serve_misses_without_residency_traffic() {
    // On by default; pinned on here whatever the environment says (read
    // once, before the device exists).
    std::env::set_var("LLOOM_TRANSIENT_HEAPS", "1");
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let before = m.residency_counters();
    let (hs0, hp0) = m.heap_stats();

    // Five 300 MB private allocations: more than one 1 GiB chunk, five pool
    // misses, and five residency adds on the per-buffer arm.
    let n = 300 * 1024 * 1024 / 4;
    let mut keep = Vec::new();
    for i in 0..5 {
        let t = Tensor::full(i as f32 + 1.0, n, &dev).unwrap();
        keep.push(t);
    }
    let after = m.residency_counters();
    let (_, hp) = m.heap_stats();
    let chunks = hp.chunks - hp0.chunks;
    assert!(
        chunks >= 2,
        "five 300 MB buffers must span ≥ 2 chunks: {hp:?}"
    );
    assert_eq!(
        after.added - before.added,
        chunks as u64,
        "one residency add per CHUNK, none per buffer: {before:?} → {after:?}"
    );
    assert_eq!(after.commits - before.commits, chunks as u64);
    assert!(
        hp.used >= 5 * n * 4,
        "the chunks hold the five buffers: {hp:?}"
    );

    // Distinct, and computed on: each holds its own fill.
    for (i, t) in keep.iter().enumerate() {
        let s = t.narrow(0, n - 1, 1).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(s[0], i as f32 + 1.0, "buffer {i} was overwritten");
    }
    let sum = Tensor::stack(
        &keep
            .iter()
            .map(|t| t.narrow(0, 0, 4).unwrap())
            .collect::<Vec<_>>(),
        0,
    )
    .unwrap()
    .sum_all()
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert_eq!(sum, 4.0 * (1.0 + 2.0 + 3.0 + 4.0 + 5.0));

    // Drop them; the pool keeps up to its free budget and evicts the rest —
    // an eviction returns the range to its chunk and removes NOTHING from the
    // residency set (a sub-buffer was never in it).
    drop(keep);
    // A released buffer waits for the epoch it was released in to complete,
    // and an epoch closes at a command-buffer commit: one small dispatch and
    // its readback close it. The pool then drains it on the completion
    // handler (Metal's thread) and evicts what is over its free budget —
    // wait for the chunks to show the ranges back.
    let _ = Tensor::zeros(4, DType::F32, &dev)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    while m.heap_stats().1.used > 300 * 1024 * 1024 {
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "the evicted sub-buffers never returned to their chunks: {:?}",
            m.heap_stats().1
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut again = Vec::new();
    for i in 0..5 {
        again.push(Tensor::full(-(i as f32) - 1.0, n, &dev).unwrap());
    }
    dev.synchronize().unwrap();
    let after2 = m.residency_counters();
    let (_, hp2) = m.heap_stats();
    assert_eq!(
        after2.removed, after.removed,
        "an evicted sub-buffer is not a residency remove"
    );
    assert_eq!(
        hp2.chunks, hp.chunks,
        "a second round of the same sizes must reuse the chunks' freed ranges: {hp:?} → {hp2:?}"
    );
    // Placed by us: what the chunks hold is what is live, to the rounding
    // of a buffer's size (the pool's 300 MB requests rounded by Metal's
    // size-and-align), and no more.
    assert!(
        hp2.used >= 5 * n * 4 && hp2.used <= 5 * n * 4 + 5 * (1 << 20),
        "the chunks' used bytes must be the five live buffers': {hp2:?}"
    );
    assert!(hp2.live >= 5, "{hp2:?}");
    for (i, t) in again.iter().enumerate() {
        let s = t.narrow(0, 0, 1).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(s[0], -(i as f32) - 1.0);
    }
    // The Shared pool's chunks too (a KV-shaped allocation goes there).
    let shared = Tensor::zeros((1024, 1024), DType::F32, &dev).unwrap();
    let _ = shared.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let (hs, _) = m.heap_stats();
    assert!(hs.chunks >= hs0.chunks, "{hs:?}");
    println!("heaps: shared {hs:?} private {hp2:?}; residency {after2:?}");
}
