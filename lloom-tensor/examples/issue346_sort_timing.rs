//! Times `Tensor::arg_sort_last_dim` at LFM2's vocabulary width against the
//! host sort it would replace (lloom issue #346, acceptance item 6).
//!
//! #277 timed this at 0.251 ms/call on a kernel that **returned early** — the
//! single-threadgroup dispatch was invalid above 1024 columns, so the figure
//! priced a kernel computing nothing. This re-takes it on the fixed path.
//!
//! Two GPU figures are reported and only one of them is the sort:
//!   - **synchronized per call** — what a caller waiting on the result pays.
//!   - **enqueue-only** (N enqueued, one sync) — what the host can *submit*.
//! #277 records the second at 0.033 ms and says plainly that the 90× it implies
//! is not quotable; both are printed here so the honest ratio is the one read
//! (`DESIGN.md` §2.4: check whether the measurement tool measures the thing its
//! output names).
use lloom_tensor::{DType, Device, Result, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128_000);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let device = Device::new_metal(0)?;

    // Deterministic, tie-free.
    let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let t = Tensor::from_vec(v.clone(), (n,), &device)?.to_dtype(DType::F32)?;

    // Correctness first: a timing on a wrong kernel is what #277 caught.
    let idx: Vec<u32> = t.arg_sort_last_dim(true)?.to_vec1()?;
    let mut seen = vec![false; n];
    for &i in &idx {
        assert!((i as usize) < n && !seen[i as usize], "not a permutation");
        seen[i as usize] = true;
    }
    let ascending = idx.windows(2).all(|w| v[w[0] as usize] <= v[w[1] as usize]);
    assert!(ascending, "not ascending");

    // Warm up the pipeline cache and the allocator.
    for _ in 0..5 {
        let _ = t.arg_sort_last_dim(true)?.to_vec1::<u32>()?;
    }

    // (a) synchronized per call.
    let start = Instant::now();
    for _ in 0..reps {
        let r = t.arg_sort_last_dim(true)?;
        device.synchronize()?;
        std::hint::black_box(&r);
    }
    let gpu_sync_ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // (b) enqueue-only: N enqueued, one sync. Prices submission, not the sort.
    let start = Instant::now();
    let mut keep = Vec::with_capacity(reps);
    for _ in 0..reps {
        keep.push(t.arg_sort_last_dim(true)?);
    }
    device.synchronize()?;
    let gpu_enqueue_ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
    std::hint::black_box(&keep);

    // (c) the host sort this replaces: the same `sort_by` over indices that
    // `Sampling::TopP` runs (§11.2c measured it at 2.99 ms standalone).
    let start = Instant::now();
    for _ in 0..reps {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&i, &j| v[i as usize].total_cmp(&v[j as usize]));
        std::hint::black_box(&order);
    }
    let host_ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!(
        "RESULT346 n={n} reps={reps} gpu_sync_ms={gpu_sync_ms:.4} \
         gpu_enqueue_ms={gpu_enqueue_ms:.4} host_sort_ms={host_ms:.4} \
         ratio_sync={:.2} ratio_enqueue_NOT_QUOTABLE={:.2} permutation=true ascending=true",
        host_ms / gpu_sync_ms,
        host_ms / gpu_enqueue_ms
    );
    Ok(())
}
