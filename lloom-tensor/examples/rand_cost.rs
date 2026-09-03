//! What does one element per thread cost `rand_uniform`?
//!
//! lloom #345 names two candidate fixes and asks that the choice be argued on
//! evidence:
//!
//! * **A -- one element per thread.** Dispatch `length` threads rather than
//!   `length/2`. Each element is one draw from its own stream.
//! * **B -- per-thread stream separation.** Keep both writes, but seed the
//!   second from an independently-derived state by folding the output index
//!   into the seed vector's two unused components.
//!
//! **Both are correct**, measured: with the seed advance fixed, B's
//! argmax-position chi-squared reads 3.8 / 12.4 / 17.9 / 243.6 at
//! n = 4 / 8 / 16 / 256 against critical values 16.27 / 24.32 / 37.70 / 330.52,
//! and A's reads 7.9 / 6.2 / 13.6 / 252.4. So the choice is a cost question,
//! and this is the cost.
//!
//! Both variants are compiled from source here rather than reached through
//! candle, so the two arms differ only in the kernel body and the thread count
//! -- no allocator, no pipeline cache, no encoder machinery between them.
//!
//! ```bash
//! cargo run --release --example rand_cost
//! ```

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

/// The generator, lifted verbatim from `random.metal` so the arms differ only
/// in how work is assigned to threads.
const PRELUDE: &str = r#"
#include <metal_stdlib>
#include <metal_integer>
#include <metal_atomic>
using namespace metal;

static constexpr constant ulong UNIF01_NORM32 = 4294967296;
static constexpr constant float UNIF01_INV32 = 2.328306436538696289e-10;
static constexpr constant int3 S1 = {13, 19, 12};
static constexpr constant int3 S2 = {2, 25, 4};
static constexpr constant int3 S3 = {3, 11, 17};
static constexpr constant uint64_t PHI[4] = {
    0x9E3779B97F4A7C15, 0xF39CC0605CEDC834,
    0x1082276BF3A27251, 0xF86C6A11D0C18E95,
};

struct HybridTaus {
    float state;
    METAL_FUNC static uint4 seed_per_thread(const ulong4 seeds) {
        return uint4(ulong4(seeds) * ulong4(PHI[0], PHI[1], PHI[2], PHI[3]) * ulong4(1099087573UL));
    }
    METAL_FUNC static uint taus(const uint z, const int3 s, const uint M) {
        uint b = (((z << s.x) ^ z) >> s.y);
        return (((z & M) << s.z) ^ b);
    }
    METAL_FUNC static uint lcg(const uint z) { return (1664525 * z + 1013904223UL); }
    METAL_FUNC static HybridTaus init(const ulong4 seeds) {
        uint4 seed = seed_per_thread(seeds);
        uint z1 = taus(seed.x, S1, 4294967294UL);
        uint z2 = taus(seed.y, S2, 4294967288UL);
        uint z3 = taus(seed.z, S3, 4294967280UL);
        uint z4 = lcg(seed.x);
        uint r1 = (z1^z2^z3^z4^seed.y);
        z1 = taus(r1, S1, 429496729UL); z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL); z4 = lcg(r1);
        r1 = (z1^z2^z3^z4^seed.z);
        z1 = taus(r1, S1, 429496729UL); z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL); z4 = lcg(r1);
        r1 = (z1^z2^z3^z4^seed.w);
        z1 = taus(r1, S1, 429496729UL); z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL); z4 = lcg(r1);
        HybridTaus rng;
        rng.state = (z1^z2^z3^z4) * UNIF01_INV32;
        return rng;
    }
    METAL_FUNC float rand() {
        uint seed = this->state * UNIF01_NORM32;
        uint z1 = taus(seed, S1, 429496729UL); uint z2 = taus(seed, S2, 4294967288UL);
        uint z3 = taus(seed, S3, 429496280UL); uint z4 = lcg(seed);
        thread float result = this->state;
        this->state = (z1^z2^z3^z4) * UNIF01_INV32;
        return result;
    }
};

// Arm A -- what #345 ships. One element per thread, `size` threads.
kernel void arm_a(
    constant size_t &size, device float *out, uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    HybridTaus rng = HybridTaus::init({299792458UL, tid, 1, 1});
    out[tid] = rng.rand();
}

// Arm B -- two elements per thread, the second stream independently seeded by
// folding the output index into the unused `.z`. `size/2` threads.
kernel void arm_b(
    constant size_t &size, device float *out, uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    uint off = 1 - size % 2;
    HybridTaus rng = HybridTaus::init({299792458UL, tid, 1, 1});
    out[tid] = rng.rand();
    uint j = size - off - tid;
    if (j != tid && j < size) {
        HybridTaus rng2 = HybridTaus::init({299792458UL, tid, 2, 1});
        out[j] = rng2.rand();
    }
}

// The pre-#345 kernel, as a reference point for what the defect bought.
kernel void arm_old(
    constant size_t &size, device float *out, uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    uint off = 1 - size % 2;
    HybridTaus rng = HybridTaus::init({299792458UL, tid, 1, 1});
    out[tid] = rng.rand();
    out[size - off - tid] = rng.rand();
}
"#;

struct Arm {
    name: &'static str,
    kernel: &'static str,
    /// threads dispatched, as a function of `length`
    threads: fn(usize) -> usize,
}

fn main() -> Result<()> {
    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let queue = device.newCommandQueue().context("no command queue")?;
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(PRELUDE), None)
        .map_err(|e| anyhow::anyhow!("compiling probe kernels: {e}"))?;

    let arms = [
        Arm {
            name: "A one-per-thread (ships)",
            kernel: "arm_a",
            threads: |n| n,
        },
        Arm {
            name: "B two-per-thread, split",
            kernel: "arm_b",
            threads: |n| n / 2 + (n % 2),
        },
        Arm {
            name: "old (pre-#345, WRONG)",
            kernel: "arm_old",
            threads: |n| n / 2 + (n % 2),
        },
    ];

    let mut pipelines = Vec::new();
    for arm in &arms {
        let f = library
            .newFunctionWithName(&NSString::from_str(arm.kernel))
            .with_context(|| format!("looking up {}", arm.kernel))?;
        let p = device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| anyhow::anyhow!("pipeline for {}: {e}", arm.kernel))?;
        pipelines.push(p);
    }

    println!("device  {}", device.name());
    println!("method  best of 50 timed runs after 10 warmup; whole encode+commit+wait");
    println!();
    println!(
        "{:<26} {:>10} {:>12} {:>12} {:>10}",
        "arm", "n", "best ms", "GB/s", "vs A"
    );

    // 128 000 is LFM2's vocab -- the width a sampler draws at. The rest bracket
    // it so the shape of any difference is visible rather than a single point.
    for &n in &[1024usize, 16_384, 128_000, 1_048_576, 8_388_608] {
        let out = device
            .newBufferWithLength_options((n * 4) as _, MTLResourceOptions::StorageModeShared)
            .context("allocating output")?;
        let mut best_a = f64::MAX;
        for (i, arm) in arms.iter().enumerate() {
            let pipeline = &pipelines[i];
            let threads = (arm.threads)(n);
            let width = pipeline.maxTotalThreadsPerThreadgroup().min(threads);
            let groups = threads.div_ceil(width);

            let mut best = f64::MAX;
            for rep in 0..60 {
                let t = std::time::Instant::now();
                let cb = queue.commandBuffer().context("command buffer")?;
                {
                    let enc = cb.computeCommandEncoder().context("encoder")?;
                    enc.setComputePipelineState(pipeline);
                    let size_val = n as u64;
                    unsafe {
                        enc.setBytes_length_atIndex(
                            std::ptr::NonNull::from(&size_val).cast(),
                            std::mem::size_of::<u64>(),
                            0,
                        );
                        enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: groups,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    enc.endEncoding();
                }
                cb.commit();
                cb.waitUntilCompleted();
                let e = t.elapsed().as_secs_f64() * 1e3;
                // discard warmup
                if rep >= 10 && e < best {
                    best = e;
                }
            }
            if i == 0 {
                best_a = best;
            }
            let gbs = (n as f64 * 4.0) / (best * 1e-3) / 1e9;
            let rel = if i == 0 {
                "--".to_string()
            } else {
                format!("{:+.1}%", (best - best_a) / best_a * 100.0)
            };
            println!(
                "{:<26} {:>10} {:>12.4} {:>12.2} {:>10}",
                arm.name, n, best, gbs, rel
            );
        }
        println!();
    }

    println!("Note: `best of` rather than a mean, deliberately -- this machine carries other");
    println!("agents (DESIGN.md 6.6a prices contention at +65.8 %) and the minimum is the");
    println!("statistic least perturbed by load. It is a LOWER BOUND on each arm's cost, and");
    println!("the comparison BETWEEN arms is what this is for rather than the absolute.");
    Ok(())
}
