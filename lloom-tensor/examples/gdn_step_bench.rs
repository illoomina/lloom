//! `gdn_step_f32` at Qwen4-exp's geometry, timed warm (#749 closing plan, K1a).
//!
//! The DeltaNet step reads the layer's state `S [d, d, h_v]` and writes it
//! back: 3 MB each way at `d = 128`, `h_v = 48`, so 6 MB is the least a step
//! can move. The per-label profiler cannot see this kernel in situ (P29 (6)),
//! so this times it alone, after the GPU's clock has ramped (P30 (2)):
//!
//! * **chain**: `N` steps, each reading the state the previous one wrote — the
//!   recurrence's own shape, the barrier between them included;
//! * **independent**: `N` steps on the same input, each to its own output — no
//!   barrier, the steps free to overlap: the kernel's throughput.
//!
//! One command buffer per arm; its GPU interval over `N` is the per-step time,
//! and 6 MB over that is the state's achieved bandwidth.
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/gdn_step_bench
//! ```

use anyhow::Result;
use clap::Parser;
use lloom_tensor::kernels::metal::{Buffer, CommandBuffer, Device};
use lloom_tensor::kernels::{call_gdn_step, BufferOffset, Kernels, RESOURCE_OPTIONS};
use objc2_metal::MTLCommandQueue;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 128)]
    d: usize,
    #[arg(long, default_value_t = 48)]
    h_v: usize,
    /// Steps per command buffer.
    #[arg(long, default_value_t = 36)]
    n: usize,
    #[arg(long, default_value_t = 30)]
    iters: usize,
    #[arg(long, default_value_t = 30)]
    warmup: usize,
}

fn filled(dev: &Device, n: usize, seed: u32) -> Result<Buffer> {
    // A deterministic, bounded fill: the state stays finite over any chain.
    let v: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / 16_777_216.0
                - 0.5)
                * 0.1
        })
        .collect();
    Ok(dev.new_buffer_with_data(v.as_ptr().cast(), n * 4, RESOURCE_OPTIONS)?)
}

fn off(b: &Buffer) -> BufferOffset<'_> {
    BufferOffset {
        buffer: b,
        offset_in_bytes: 0,
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dev = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    let kernels = Kernels::new();
    let queue = dev.new_command_queue()?;
    let (d, h) = (a.d, a.h_v);
    let s_len = d * d * h;
    let heads = filled(&dev, 3 * h * d, 1)?;
    let alpha = filled(&dev, h, 2)?;
    let beta = filled(&dev, h, 3)?;
    let dt_bias = filled(&dev, h, 4)?;
    let ssm_a = filled(&dev, h, 5)?;
    let s0 = filled(&dev, s_len, 6)?;
    let s1 = filled(&dev, s_len, 7)?;
    let o = dev.new_buffer(h * d * 4, RESOURCE_OPTIONS)?;
    let outs: Vec<Buffer> = (0..a.n)
        .map(|_| dev.new_buffer(s_len * 4, RESOURCE_OPTIONS))
        .collect::<std::result::Result<_, _>>()?;
    // Its own `o` too, or the shared read-out is a write-after-write hazard
    // and the encoder puts a barrier between "independent" steps.
    let os: Vec<Buffer> = (0..a.n)
        .map(|_| dev.new_buffer(h * d * 4, RESOURCE_OPTIONS))
        .collect::<std::result::Result<_, _>>()?;

    // One command buffer of `n` steps in form `reg`; its GPU time per step.
    let run = |chain: bool, reg: bool| -> Result<f64> {
        let cb = CommandBuffer::new(queue.commandBuffer().unwrap());
        let enc = cb.compute_command_encoder_no_fence()?;
        for k in 0..a.n {
            let (src, dst, o) = if chain {
                if k % 2 == 0 {
                    (&s0, &s1, &o)
                } else {
                    (&s1, &s0, &o)
                }
            } else {
                (&s0, &outs[k], &os[k])
            };
            // SAFETY: the bench owns every buffer bound here and reads them only after
            // waiting for the command buffer that writes them.
            unsafe {
                call_gdn_step(
                    &dev,
                    &enc,
                    &kernels,
                    d,
                    h,
                    off(&heads),
                    off(&alpha),
                    off(&beta),
                    off(&dt_bias),
                    off(&ssm_a),
                    off(src),
                    o,
                    off(dst),
                    reg,
                )
            }?;
        }
        enc.end_encoding()?;
        cb.commit()?;
        cb.wait_until_completed()?;
        Ok((cb.gpu_end_time() - cb.gpu_start_time()) * 1e6 / a.n as f64)
    };

    // Identity first: one independent step in each form from the same state,
    // the read-out and the new state compared byte for byte.
    let one = |reg: bool, out: &Buffer, o: &Buffer| -> Result<(Vec<u8>, Vec<u8>)> {
        let cb = CommandBuffer::new(queue.commandBuffer().unwrap());
        let enc = cb.compute_command_encoder_no_fence()?;
        // SAFETY: the bench owns every buffer bound here and reads them only after
        // waiting for the command buffer that writes them.
        unsafe {
            call_gdn_step(
                &dev,
                &enc,
                &kernels,
                d,
                h,
                off(&heads),
                off(&alpha),
                off(&beta),
                off(&dt_bias),
                off(&ssm_a),
                off(&s0),
                o,
                off(out),
                reg,
            )
        }?;
        enc.end_encoding()?;
        cb.commit()?;
        cb.wait_until_completed()?;
        let bytes = |b: &Buffer, n: usize| unsafe {
            std::slice::from_raw_parts(b.contents() as *const u8, n).to_vec()
        };
        Ok((bytes(o, h * d * 4), bytes(out, s_len * 4)))
    };
    let (o_loop, s_loop) = one(false, &outs[0], &os[0])?;
    let (o_reg, s_reg) = one(true, &outs[1], &os[1])?;
    let (o_reg2, s_reg2) = one(true, &outs[2], &os[2])?;
    println!(
        "GDNSTEP determinism reg_o_repeat_equal={} reg_s_repeat_equal={}",
        o_reg == o_reg2,
        s_reg == s_reg2
    );
    // How far apart, when not equal: the largest absolute difference and the
    // largest in f32 ulps, and how many elements differ.
    let diff = |x: &[u8], y: &[u8]| -> (usize, f32, u32) {
        let f = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (x, y) = (f(x), f(y));
        let mut n = 0;
        let mut abs = 0f32;
        let mut ulp = 0u32;
        for (a, b) in x.iter().zip(&y) {
            if a.to_bits() != b.to_bits() {
                n += 1;
                abs = abs.max((a - b).abs());
                ulp = ulp.max((a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32);
            }
        }
        (n, abs, ulp)
    };
    let (on, oa, ou) = diff(&o_loop, &o_reg);
    let (sn, sa, su) = diff(&s_loop, &s_reg);
    println!(
        "GDNSTEP identity o_equal={} s_equal={} o_differ={on}/{} o_max_abs={oa:e} o_max_ulp={ou} s_differ={sn}/{} s_max_abs={sa:e} s_max_ulp={su}",
        o_loop == o_reg,
        s_loop == s_reg,
        o_loop.len() / 4,
        s_loop.len() / 4
    );

    for _ in 0..a.warmup {
        for reg in [false, true] {
            run(true, reg)?;
            run(false, reg)?;
        }
    }
    let bytes = 2.0 * (s_len * 4) as f64;
    let mut t = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
    for _ in 0..a.iters {
        // Interleaved: a clock excursion lands on every arm.
        for (ri, reg) in [false, true].into_iter().enumerate() {
            t[ri][0].push(run(true, reg)?);
            t[ri][1].push(run(false, reg)?);
        }
    }
    for (ri, name) in ["loop", "reg"].into_iter().enumerate() {
        for v in t[ri].iter_mut() {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        }
        let (cm, im) = (t[ri][0][a.iters / 2], t[ri][1][a.iters / 2]);
        println!(
            "GDNSTEP form={name} d={d} h_v={h} n={} chain_us={cm:.1} (min {:.1}) independent_us={im:.1} (min {:.1}) chain_gb_s={:.0} independent_gb_s={:.0}",
            a.n,
            t[ri][0][0],
            t[ri][1][0],
            bytes / cm / 1e3,
            bytes / im / 1e3
        );
    }
    Ok(())
}
