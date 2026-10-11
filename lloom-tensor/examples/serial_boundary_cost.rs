//! What does one serial boundary cost the GPU? (#749 closing plan, E3)
//!
//! A fusion removes the boundary between two dependent dispatches and keeps
//! their work. So the quantity a fusion saves is the difference between N
//! dispatches that each wait for the previous one and the same N dispatches
//! when nothing makes them wait:
//!
//! * **dependent**: a ping-pong chain `a → b → a → …` through the production
//!   encoder, which puts a barrier before every dispatch (each reads what the
//!   last one wrote);
//! * **independent**: the same kernel on the same input, each writing its own
//!   output — no hazard, no barrier, free to overlap.
//!
//! Both arms are one command buffer; its `GPUEndTime − GPUStartTime` is the
//! arm's time. `c = (T_dependent − T_independent) / N` is the per-boundary
//! cost, swept over the element count (a decode vector is 2 560 wide; a
//! routed row set, 10× that) and over `N` (the slope is the cost, the
//! intercept the command buffer's own). Medians over `--iters` after
//! `--warmup` untimed repetitions, so the GPU's clock has ramped (P30 (2)).
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/serial_boundary_cost
//! ```

use anyhow::Result;
use clap::Parser;
use lloom_tensor::kernels::metal::{Buffer, CommandBuffer, ComputeCommandEncoder, Device};
use lloom_tensor::kernels::{
    call_unary_contiguous, unary, BufferOffset, Kernels, RESOURCE_OPTIONS,
};
use objc2_metal::MTLCommandQueue;

#[derive(Parser, Debug)]
#[command(about = "GPU cost of a serial (barrier-ordered) boundary between dispatches")]
struct Args {
    /// Elements per dispatch (f32).
    #[arg(long, value_delimiter = ',', default_values_t = [2560usize, 10240, 65536, 262144])]
    elems: Vec<usize>,
    /// Dispatches per command buffer.
    #[arg(long, value_delimiter = ',', default_values_t = [256usize, 1024, 4096])]
    counts: Vec<usize>,
    /// Timed repetitions per cell; the median is reported.
    #[arg(long, default_value_t = 30)]
    iters: usize,
    /// Untimed repetitions first.
    #[arg(long, default_value_t = 20)]
    warmup: usize,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    let kernels = Kernels::new();
    let queue = dev.new_command_queue()?;
    let max_n = *args.counts.iter().max().unwrap_or(&1);
    let max_e = *args.elems.iter().max().unwrap_or(&1);
    let buf = || dev.new_buffer(max_e * 4, RESOURCE_OPTIONS);
    let (src, ping, pong) = (buf()?, buf()?, buf()?);
    let outs: Vec<Buffer> = (0..max_n)
        .map(|_| buf())
        .collect::<std::result::Result<_, _>>()?;

    let cos = |enc: &ComputeCommandEncoder, len: usize, input: &Buffer, output: &Buffer| {
        call_unary_contiguous(
            &dev,
            enc,
            &kernels,
            unary::contiguous::cos::FLOAT,
            4,
            len,
            BufferOffset {
                buffer: input,
                offset_in_bytes: 0,
            },
            output,
        )
    };
    // One command buffer of `n` dispatches at `len` elements; its GPU interval in µs.
    let run = |len: usize, n: usize, dependent: bool| -> Result<f64> {
        let cb = CommandBuffer::new(queue.commandBuffer().unwrap());
        let enc = cb.compute_command_encoder_no_fence();
        for k in 0..n {
            if dependent {
                let (a, b) = if k % 2 == 0 {
                    (&ping, &pong)
                } else {
                    (&pong, &ping)
                };
                cos(&enc, len, a, b)?;
            } else {
                cos(&enc, len, &src, &outs[k])?;
            }
        }
        enc.end_encoding();
        cb.commit()?;
        cb.wait_until_completed()?;
        Ok((cb.gpu_end_time() - cb.gpu_start_time()) * 1e6)
    };

    println!("SERIAL elems n dependent_us independent_us c_us_per_boundary");
    for &len in &args.elems {
        let mut cells: Vec<(usize, f64, f64)> = Vec::new();
        for &n in &args.counts {
            for _ in 0..args.warmup {
                run(len, n, true)?;
                run(len, n, false)?;
            }
            let (mut dep, mut ind) = (Vec::new(), Vec::new());
            for _ in 0..args.iters {
                // Interleaved, so a clock excursion lands on both arms.
                dep.push(run(len, n, true)?);
                ind.push(run(len, n, false)?);
            }
            let (d, i) = (median(&mut dep), median(&mut ind));
            println!(
                "SERIAL elems={len} n={n} dependent_us={d:.1} independent_us={i:.1} c_us={:.3}",
                (d - i) / n as f64
            );
            cells.push((n, d, i));
        }
        // The slope between the smallest and largest count: the per-boundary
        // cost with the command buffer's fixed cost cancelled.
        if let (Some(lo), Some(hi)) = (cells.first(), cells.last()) {
            if hi.0 > lo.0 {
                let dn = (hi.0 - lo.0) as f64;
                println!(
                    "SLOPE elems={len} dependent_us_per_dispatch={:.3} independent_us_per_dispatch={:.3} c_us={:.3}",
                    (hi.1 - lo.1) / dn,
                    (hi.2 - lo.2) / dn,
                    ((hi.1 - lo.1) - (hi.2 - lo.2)) / dn
                );
            }
        }
    }
    Ok(())
}
