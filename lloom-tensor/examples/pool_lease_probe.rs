//! Issue #494 step 1: is the buffer pool's state machine exercisable without a
//! GPU lease?
//!
//! The feasibility question the issue makes decisive. `BufferPool`'s bookkeeping
//! is device-independent -- `adopt` takes a `Buffer` and a size and updates
//! counters -- but `Buffer` wraps `Retained<ProtocolObject<dyn MTLBuffer>>`, so
//! a handle cannot be built without a real `MTLBuffer`. That is the coupling.
//! The question is therefore not "can the GPU be avoided" but **"does the
//! allocation path this forces need the exclusive lease `lloom-arb measure`
//! holds"** -- and that is an empirical question about cost and contention.
//!
//! This measures it **through the shipping `Device` and `BufferPool`**, not
//! through a hand-rolled `newBufferWithLength:`. #454 (§11.5e, and the pit)
//! measured a standalone probe over the same API reading MAPPED at every size
//! where candle's own path reads null at 16384 B: a standalone `MTLBuffer`
//! probe is a different instrument from the one that ships.
//!
//! No command queue is created, no command buffer is committed, and no kernel
//! is dispatched. The `GpuClock` is driven by hand exactly as the existing 30
//! tests drive it (`commit_epoch` / `mark_completed`), which is what makes the
//! claim testable: if the pool's alphabet never reaches the GPU's execution
//! units, a fuzzer over it contends for the allocator and nothing else.

use lloom_tensor::kernels::metal::device::Device;
use lloom_tensor::kernels::metal::{BufferHandle, BufferPool, GpuClock};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let ops: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    let t0 = Instant::now();
    let dev = Device::system_default().expect("no Metal device");
    let device_ms = t0.elapsed().as_secs_f64() * 1e3;

    let clock = Arc::new(GpuClock::new());
    let pool = BufferPool::with_clock(Arc::clone(&clock));
    // Small enough that the growing-size mix below actually exceeds it. A first
    // version used 8 MiB with a trim every 4096 ops and the budget never bound
    // -- `evicted=0` over 200k operations. The engagement assertion at the
    // bottom caught it, which is the whole reason it is there (§2.4).
    pool.set_free_budget(64 * 1024);

    // A sequence over the pool's alphabet, shaped like what a fuzzer generates:
    // acquire / adopt / drop / commit_epoch / mark_completed / drain / trim.
    // Deliberately mixes sizes so both the exact-bucket and the range probe
    // fire, and so eviction engages.
    let t1 = Instant::now();
    let mut live: Vec<BufferHandle> = Vec::new();
    let mut allocations = 0usize;
    for i in 0..ops {
        let size = 1024 + (i % 97) * 128;

        // acquire, else allocate and adopt -- the shape `MetalDevice` uses.
        let b = match pool.acquire(size) {
            Some(b) => b,
            None => {
                allocations += 1;
                let raw = dev
                    .new_buffer(size, lloom_tensor::kernels::RESOURCE_OPTIONS)
                    .expect("buffer allocation");
                pool.adopt(raw, size)
            }
        };
        live.push(b);

        // Release some, so `PooledBuffer::drop` runs and parks in `pending`.
        if live.len() > 64 {
            live.remove(0);
        }
        // Advance the clock by hand: no command buffer exists.
        if i % 16 == 0 {
            let e = clock.commit_epoch();
            clock.mark_completed(e);
            pool.drain_completed();
        }
        if i % 4096 == 0 {
            drop(pool.trim());
        }
    }
    let churn_s = t1.elapsed().as_secs_f64();
    drop(live);

    let c = pool.counters();
    let occ = pool.occupancy();
    println!("PROBE494 device_create_ms={device_ms:.3}");
    println!(
        "PROBE494 ops={ops} wall_s={churn_s:.4} ops_per_s={:.0} us_per_op={:.4}",
        ops as f64 / churn_s,
        churn_s * 1e6 / ops as f64
    );
    println!(
        "PROBE494 allocations={allocations} hits={} releases={} drained={} evicted={} trims={}",
        c.hits, c.releases, c.drained, c.evicted, c.trims
    );
    println!(
        "PROBE494 live_buffers={} free_buffers={} pending_buffers={} free_bytes={}",
        occ.live_buffers, occ.free_buffers, occ.pending_buffers, occ.free_bytes
    );
    // Engagement, per §2.4: a probe that exercised no reuse and no eviction has
    // not measured the state machine. Refuse rather than print a clean number.
    assert!(c.hits > 0, "no pool hit: the reuse path never engaged");
    assert!(
        c.evicted > 0,
        "nothing evicted: the eviction path never engaged"
    );
    assert!(
        c.drained > 0,
        "nothing drained: the epoch path never engaged"
    );
    println!("PROBE494 engaged=true");
}
