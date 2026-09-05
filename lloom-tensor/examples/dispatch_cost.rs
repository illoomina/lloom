//! What does one Metal dispatch cost the CPU?
//!
//! Measurement harness for lloom issue #7 / `DESIGN.md` §11.2 and §16 P0 #1.
//!
//! §11.2 argues that per-token CPU dispatch cost is significant and marks the
//! fraction **UNVERIFIED**. The decode profile answers that in situ; this
//! answers it in isolation, so the two can be cross-checked. If
//! `dispatches_per_token x cost_per_dispatch` lands near the non-GPU time the
//! decode profile measured, both numbers are probably right. If they disagree,
//! at least one is measuring something other than what it claims.
//!
//! # What is measured
//!
//! The kernel is deliberately trivial -- it writes one float -- so the GPU
//! finishes almost immediately and what remains is the CPU-side cost of
//! encoding: setting a pipeline, binding buffers, and issuing the dispatch.
//!
//! Two quantities, and the distinction is the whole point:
//!
//! * **encode time** -- wall time to encode N dispatches, *not* including
//!   waiting for them. This is the cost that sits in front of GPU work in a
//!   serial decode loop and that the GPU cannot hide (§6.7 L4b).
//! * **round-trip time** -- encode, commit, and wait. Includes submission and
//!   completion-notification latency, which is paid per command buffer rather
//!   than per dispatch.
//!
//! Sweeping N separates the fixed per-command-buffer cost from the marginal
//! per-dispatch cost: fitting a line through (N, time) gives the slope as the
//! marginal dispatch cost and the intercept as the fixed overhead. Reporting
//! only `total / N` at one N would fold the two together and overstate the
//! per-dispatch figure at small N.
//!
//! ```bash
//! cargo run --release --example dispatch_cost
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

#[derive(Parser, Debug)]
#[command(about = "CPU cost of a Metal compute dispatch")]
struct Args {
    /// Dispatch counts to sweep.
    ///
    /// The decode path issues ~675 per token, so the sweep brackets that.
    #[arg(long, value_delimiter = ',', default_values_t = [50usize, 100, 200, 400, 675, 1000, 2000])]
    counts: Vec<usize>,

    /// Timed repetitions per count; the median is reported.
    #[arg(long, default_value_t = 30)]
    iters: usize,

    /// Untimed repetitions first.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Rebind buffers before every dispatch, as a real kernel call does.
    ///
    /// On by default because the question is what candle's decode path pays, and
    /// candle rebinds per call. `--no-rebind` isolates the dispatch call itself.
    #[arg(long)]
    no_rebind: bool,
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Deliberately trivial: the GPU cost should be negligible against the CPU cost
// of encoding the dispatch, which is what is being measured.
kernel void touch(device float* dst [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
    dst[tid] = float(tid);
}
"#;

/// Median of a slice, which is the statistic reported.
///
/// Median rather than mean because an occasional scheduler preemption produces a
/// large outlier that the mean would absorb and report as if it were typical.
fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n.is_multiple_of(2) {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    } else {
        xs[n / 2]
    }
}

/// Least-squares fit of `y = a + b*x`, returning `(intercept, slope)`.
///
/// The slope is the marginal cost of one more dispatch; the intercept is the
/// per-command-buffer overhead that does not scale with dispatch count.
fn linear_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return (0.0, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (intercept, slope)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let queue = device.newCommandQueue().context("no command queue")?;

    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .map_err(|e| anyhow::anyhow!("compiling probe kernel: {e}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("touch"))
        .context("looking up touch")?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| anyhow::anyhow!("creating pipeline: {e}"))?;

    let dst = device
        .newBufferWithLength_options(4096 * 4, MTLResourceOptions::StorageModePrivate)
        .context("allocating output buffer")?;

    println!("device                {}", device.name());
    println!(
        "rebind per dispatch   {}",
        if args.no_rebind { "no" } else { "yes" }
    );
    println!(
        "iterations            {} (+{} warmup)",
        args.iters, args.warmup
    );
    println!();
    println!(
        "{:>8} {:>12} {:>12} {:>14} {:>14}",
        "N", "encode ms", "roundtrip ms", "encode us/disp", "rt us/disp"
    );

    let grid = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

    let mut encode_points: Vec<(f64, f64)> = Vec::new();
    let mut rt_points: Vec<(f64, f64)> = Vec::new();

    for &n in &args.counts {
        let mut encode_samples = Vec::with_capacity(args.iters);
        let mut rt_samples = Vec::with_capacity(args.iters);

        for it in 0..(args.warmup + args.iters) {
            let start = std::time::Instant::now();

            let cb = queue.commandBuffer().context("command buffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;
            enc.setComputePipelineState(&pipeline);
            if args.no_rebind {
                unsafe { enc.setBuffer_offset_atIndex(Some(&dst), 0, 0) };
            }
            for _ in 0..n {
                if !args.no_rebind {
                    // What candle does: every kernel call rebinds its buffers.
                    unsafe { enc.setBuffer_offset_atIndex(Some(&dst), 0, 0) };
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            }
            enc.endEncoding();

            // Encode time: everything the CPU did before handing work to the GPU.
            // Measured before commit, because commit is where the CPU stops being
            // the critical path.
            let encode = start.elapsed().as_secs_f64();

            cb.commit();
            cb.waitUntilCompleted();
            let roundtrip = start.elapsed().as_secs_f64();

            if it < args.warmup {
                continue;
            }
            encode_samples.push(encode);
            rt_samples.push(roundtrip);
        }

        let enc_med = median(&mut encode_samples);
        let rt_med = median(&mut rt_samples);

        println!(
            "{n:>8} {:>12.3} {:>12.3} {:>14.2} {:>14.2}",
            enc_med * 1e3,
            rt_med * 1e3,
            enc_med * 1e6 / n as f64,
            rt_med * 1e6 / n as f64,
        );

        encode_points.push((n as f64, enc_med));
        rt_points.push((n as f64, rt_med));
    }

    let (enc_intercept, enc_slope) = linear_fit(&encode_points);
    let (rt_intercept, rt_slope) = linear_fit(&rt_points);

    println!();
    println!("=== linear fit (time = fixed + N * marginal) ===");
    println!(
        "encode                fixed {:.3} ms, marginal {:.3} us/dispatch",
        enc_intercept * 1e3,
        enc_slope * 1e6
    );
    println!(
        "roundtrip             fixed {:.3} ms, marginal {:.3} us/dispatch",
        rt_intercept * 1e3,
        rt_slope * 1e6
    );
    println!();
    println!(
        "At 675 dispatches/token, encode alone predicts {:.3} ms/token.",
        (enc_intercept + 675.0 * enc_slope) * 1e3
    );
    println!();
    println!(
        "RESULT encode_us_per_dispatch={:.4} encode_fixed_ms={:.4} \
         rt_us_per_dispatch={:.4} rt_fixed_ms={:.4}",
        enc_slope * 1e6,
        enc_intercept * 1e3,
        rt_slope * 1e6,
        rt_intercept * 1e3,
    );

    Ok(())
}
