//! What one buffer costs the host per decode step, served by the pool and by
//! a pass plan's arena (`DESIGN.md` §6.4e, §9.2aa): a measurement, run under
//! `lloom-arb measure`.
//!
//! `pool`: a decode-shaped working set (48 sizes × 32 buffers) acquired from
//! a warm pool, released, the epoch completed, every round: the per-buffer
//! host cost of an acquire hit, a release and its drain. `arena`: the same
//! count of allocations offered to an installed arena and dropped: the cost
//! of a planned output. Uses the public pool and arena API only, so the same
//! source builds on a tree whose handle is `Arc<PooledBuffer>`.
//!
//!   cargo run --release -p lloom-tensor --features metal --example pool_round_trip_price

#[cfg(feature = "metal")]
fn main() {
    use lloom_tensor::kernels::metal::{
        plan_from_sizes, Arena, ArenaLayout, BufferPool, Device, GpuClock,
    };
    use lloom_tensor::kernels::RESOURCE_OPTIONS;
    use std::sync::Arc;
    use std::time::Instant;

    const SIZES: usize = 48;
    const PER_SIZE: usize = 32;
    const ROUNDS: usize = 400;
    let dev = Device::system_default().expect("no Metal device");
    let sizes: Vec<usize> = (0..SIZES)
        .flat_map(|i| std::iter::repeat_n(4096 + 256 * i, PER_SIZE))
        .collect();
    let n = sizes.len();

    let report = |what: &str, mut samples: Vec<f64>| {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r = samples.len();
        println!(
            "PRICE {what} buffers_per_round={n} rounds={r} ns_per_buffer min {:.1} median {:.1} p90 {:.1} max {:.1}",
            samples[0],
            samples[r / 2],
            samples[r * 9 / 10],
            samples[r - 1]
        );
    };

    // The pool: one warm-up round allocates, every later round hits.
    let clock = Arc::new(GpuClock::new());
    let pool = BufferPool::with_clock(Arc::clone(&clock));
    pool.set_free_budget(usize::MAX);
    let mut held = Vec::with_capacity(n);
    let mut samples = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS + 8 {
        let t = Instant::now();
        for &s in &sizes {
            let b = match pool.acquire(s) {
                Some(b) => b,
                None => pool.adopt(dev.new_buffer(s, RESOURCE_OPTIONS).expect("allocation"), s),
            };
            held.push(b);
        }
        held.clear();
        let e = clock.commit_epoch();
        clock.mark_completed(e);
        if round >= 8 {
            samples.push(t.elapsed().as_nanos() as f64 / n as f64);
        }
    }
    assert_eq!(
        pool.counters().allocations,
        n as u64,
        "the timed rounds missed the pool"
    );
    report("pool", samples);

    // The arena: every ordinal its own slot, offered in order each round.
    let plan = plan_from_sizes(&sizes, &(0..n).collect::<Vec<_>>(), ArenaLayout::Packed);
    let base = dev
        .new_buffer(plan.arena_bytes().max(1), RESOURCE_OPTIONS)
        .expect("arena base");
    let arena = Arena::new(base, plan, ArenaLayout::Packed).expect("arena");
    let mut samples = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS + 8 {
        let t = Instant::now();
        arena.begin_step();
        for &s in &sizes {
            held.push(arena.acquire(s).expect("planned"));
        }
        held.clear();
        arena.end_step();
        if round >= 8 {
            samples.push(t.elapsed().as_nanos() as f64 / n as f64);
        }
    }
    report("arena", samples);
}

#[cfg(not(feature = "metal"))]
fn main() {}
