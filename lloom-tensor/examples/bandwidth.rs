//! What memory bandwidth does this machine actually achieve?
//!
//! Measurement harness for lloom issue #7 / `DESIGN.md` §3.4 and §16 P0 #4.
//!
//! §3.4 lists ~400 GB/s for an M1 Max and marks it **UNVERIFIED for our specific
//! machine**. That figure is a spec number -- the theoretical peak of the memory
//! interface -- and no kernel reaches it, so comparing a measured decode time
//! against it says very little. This measures what a kernel can actually
//! sustain, which is the number a roofline argument needs.
//!
//! # What is measured
//!
//! A streaming read: every thread strides through a large buffer, accumulating
//! into registers, and writes one float per thread at the end. The accumulator
//! exists only to stop the compiler eliminating the loads; the write is 4 bytes
//! per thread against hundreds of megabytes read, so it does not meaningfully
//! enter the byte count.
//!
//! Three axes are varied, because "achieved bandwidth" is not a single number:
//!
//! * **vector width** -- `half`, `half4`, and `half4` x2 per iteration. A GEMV
//!   reading f16 weights is the workload in question, so f16 is what is measured.
//! * **threads** -- occupancy buys latency hiding (§3.2); too few threads leaves
//!   the memory system waiting.
//! * **buffer size** -- large enough that no cache holds it, or this measures
//!   cache bandwidth. LFM2's weights are 5.39 GB, so the sweep runs out to
//!   comparable sizes.
//!
//! # Why the peak of a sweep
//!
//! A single kernel shape may be limited by its own instruction mix rather than
//! by memory, which would understate the ceiling. The peak over a sweep is a
//! lower bound on what the hardware can do -- the direction of error a roofline
//! tolerates, since if decode achieves close to this peak then decode is
//! bandwidth-bound and nothing is unexplained.
//!
//! ```bash
//! cargo run --release --example bandwidth
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

#[derive(Parser, Debug)]
#[command(about = "Achieved memory bandwidth on this machine")]
struct Args {
    /// Buffer sizes to sweep, in MiB.
    ///
    /// The smallest should exceed the system-level cache or it measures the
    /// cache; the largest approaches LFM2's 5.39 GB working set.
    #[arg(long, value_delimiter = ',', default_values_t = [256usize, 1024, 2048, 5120])]
    sizes_mib: Vec<usize>,

    /// Timed iterations per configuration.
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Untimed iterations first, to let clocks settle.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Threads per threadgroup to sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [256usize, 512, 1024])]
    threadgroup: Vec<usize>,

    /// Print every configuration, not just the peak per size.
    #[arg(long)]
    verbose: bool,
}

/// Streaming-read kernels over an f16 buffer.
///
/// Each thread strides by the total thread count, so consecutive lanes touch
/// consecutive addresses and every access coalesces. Accumulating into registers
/// keeps the loads independent -- a dependent chain would measure latency rather
/// than bandwidth.
const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void stream_h1(device const half*  src [[buffer(0)]],
                      device float*       dst [[buffer(1)]],
                      constant uint&      n   [[buffer(2)]],
                      uint tid  [[thread_position_in_grid]],
                      uint nthr [[threads_per_grid]]) {
    float acc = 0.0f;
    for (uint i = tid; i < n; i += nthr) { acc += float(src[i]); }
    dst[tid] = acc;
}

kernel void stream_h4(device const half4* src [[buffer(0)]],
                      device float*       dst [[buffer(1)]],
                      constant uint&      n   [[buffer(2)]],
                      uint tid  [[thread_position_in_grid]],
                      uint nthr [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = tid; i < n; i += nthr) { acc += float4(src[i]); }
    dst[tid] = acc.x + acc.y + acc.z + acc.w;
}

kernel void stream_h4x2(device const half4* src [[buffer(0)]],
                        device float*       dst [[buffer(1)]],
                        constant uint&      n   [[buffer(2)]],
                        uint tid  [[thread_position_in_grid]],
                        uint nthr [[threads_per_grid]]) {
    // Two independent half4 loads per iteration, so more requests are in flight
    // per thread. This is what hides latency when thread count alone does not.
    float4 a = float4(0.0f);
    float4 b = float4(0.0f);
    uint stride = nthr * 2;
    for (uint i = tid * 2; i + 1 < n; i += stride) {
        a += float4(src[i]);
        b += float4(src[i + 1]);
    }
    float4 acc = a + b;
    dst[tid] = acc.x + acc.y + acc.z + acc.w;
}
"#;

struct Variant {
    name: &'static str,
    /// Bytes per index step, so the loop bound and the byte count agree for the
    /// vector loads.
    bytes_per_elem: usize,
}

const VARIANTS: [Variant; 3] = [
    Variant {
        name: "stream_h1",
        bytes_per_elem: 2,
    },
    Variant {
        name: "stream_h4",
        bytes_per_elem: 8,
    },
    Variant {
        name: "stream_h4x2",
        bytes_per_elem: 8,
    },
];

fn main() -> Result<()> {
    let args = Args::parse();

    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let queue = device.newCommandQueue().context("no command queue")?;

    println!("device                {}", device.name());
    println!(
        "unified memory        {}",
        if device.hasUnifiedMemory() {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "max threadgroup       {}",
        device.maxThreadsPerThreadgroup().width
    );
    println!(
        "recommended wset      {:.2} GB",
        device.recommendedMaxWorkingSetSize() as f64 / 1e9
    );
    println!();

    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .map_err(|e| anyhow::anyhow!("compiling bandwidth kernels: {e}"))?;

    // One f32 per thread of the largest grid, allocated once so it never enters
    // the timed region.
    let max_threads: usize = args.threadgroup.iter().max().copied().unwrap_or(1024) * 4096;
    let dst = device
        .newBufferWithLength_options(max_threads * 4, MTLResourceOptions::StorageModePrivate)
        .context("allocating output buffer")?;

    if args.verbose {
        println!(
            "{:<12} {:>10} {:>6} {:>10} {:>10} {:>9}",
            "kernel", "size", "tg", "threads", "GB/s", "ms"
        );
    } else {
        println!(
            "{:<12} {:>10} {:>6} {:>10} {:>10}",
            "best kernel", "size", "tg", "threads", "GB/s"
        );
    }

    let mut global_peak = 0.0f64;
    let mut global_peak_desc = String::new();

    for &mib in &args.sizes_mib {
        let bytes = mib * 1024 * 1024;
        let src = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
            .with_context(|| format!("allocating {mib} MiB source buffer"))?;

        let mut size_peak = 0.0f64;
        let mut size_peak_desc = String::new();

        for variant in &VARIANTS {
            let function = library
                .newFunctionWithName(&NSString::from_str(variant.name))
                .with_context(|| format!("looking up {}", variant.name))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| anyhow::anyhow!("pipeline for {}: {e}", variant.name))?;

            let max_tg = pipeline.maxTotalThreadsPerThreadgroup();

            for &tg in &args.threadgroup {
                if tg > max_tg {
                    continue;
                }
                // Enough threadgroups to fill the machine several times over, so
                // the result is not limited by having too little work resident.
                // Capped so the output buffer stays in bounds.
                let groups = (max_threads / tg).min(4096);
                let threads = groups * tg;
                let n_elems = (bytes / variant.bytes_per_elem) as u32;

                let mut best = f64::MAX;
                for it in 0..(args.warmup + args.iters) {
                    let cb = queue.commandBuffer().context("command buffer")?;
                    let enc = cb.computeCommandEncoder().context("encoder")?;
                    enc.setComputePipelineState(&pipeline);
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&src), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&dst), 0, 1);
                        enc.setBytes_length_atIndex(std::ptr::NonNull::from(&n_elems).cast(), 4, 2);
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: tg,
                            height: 1,
                            depth: 1,
                        },
                    );
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();

                    if it < args.warmup {
                        continue;
                    }
                    // GPU-reported time, so CPU submission overhead is excluded:
                    // this measures the kernel, not the queue.
                    let secs = cb.GPUEndTime() - cb.GPUStartTime();
                    if secs > 0.0 && secs < best {
                        best = secs;
                    }
                }

                anyhow::ensure!(
                    best.is_finite(),
                    "no usable timing for {} at {mib} MiB",
                    variant.name
                );

                // Best of N rather than the mean: contention and clock ramp can
                // only make a run slower, so the fastest is closest to what the
                // hardware sustains.
                let gbs = bytes as f64 / best / 1e9;
                if args.verbose {
                    println!(
                        "{:<12} {:>6} MiB {:>6} {:>10} {:>10.1} {:>9.3}",
                        variant.name,
                        mib,
                        tg,
                        threads,
                        gbs,
                        best * 1e3
                    );
                }
                if gbs > size_peak {
                    size_peak = gbs;
                    size_peak_desc = format!(
                        "{:<12} {:>6} MiB {:>6} {:>10}",
                        variant.name, mib, tg, threads
                    );
                }
            }
        }

        if args.verbose {
            println!("  -> peak at {mib} MiB: {size_peak:.1} GB/s");
        } else {
            println!("{size_peak_desc} {size_peak:>10.1}");
        }
        if size_peak > global_peak {
            global_peak = size_peak;
            global_peak_desc = size_peak_desc.clone();
        }
    }

    println!();
    println!("peak achieved         {global_peak:.1} GB/s");
    println!("  at                  {}", global_peak_desc.trim());
    println!("fraction of ~400 spec {:.0}%", 100.0 * global_peak / 400.0);
    println!();
    println!("RESULT achieved_gbs={global_peak:.1}");

    Ok(())
}
