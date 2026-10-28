use anyhow::Result;
/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
use clap::{Parser, Subcommand};
use half::f16;
use lloom_tensor::kernels::{
    metal::{Commands, Device, ResidencySet},
    GemmDType, RESOURCE_OPTIONS,
};

fn run_gemm(f32: bool, n: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 2;
    const MIN_DUR: f64 = 4.;

    let device = Device::system_default().unwrap();

    let (b, m, n, k) = (1, n, n, n);
    let kernels = lloom_tensor::kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (lhs, rhs) = if f32 {
        let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
        let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    } else {
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    };
    let (dtype, sizeof) = if f32 {
        (GemmDType::F32, core::mem::size_of::<f32>())
    } else {
        (GemmDType::F16, core::mem::size_of::<f16>())
    };
    let output = device.new_buffer(b * m * n * sizeof, options).unwrap();

    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, residency_set.clone()).unwrap();
        let encoder = commands.command_encoder().unwrap();
        let start_time = std::time::Instant::now();
        // SAFETY: the example owns every buffer bound or mapped here and reads them
        // only after waiting for the command buffer that writes them.
        unsafe {
            lloom_tensor::kernels::call_mlx_gemm(
                &device,
                &encoder,
                &kernels,
                dtype,
                (b, m, n, k),
                &[m * k, k, 1],
                0,
                &lhs,
                &[n * k, n, 1],
                0,
                &rhs,
                &output,
            )
        }?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += 1;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let gflops = (2 * n * n * n * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?},      {n:6}      gflops {gflops:.0}");

    Ok(())
}

/// LFM2-2.6B's prefill projections, `x[m, k] · W[n, k]ᵀ` at f16 with `W`
/// row-major as the model stores it (the `nt` layout), `reps` launches in one
/// command buffer so its overhead is amortised: TFLOP/s per shape, through the
/// shipping tile selection (`LLOOM_383_TILE` overrides it). The launches write
/// `outputs` buffers in turn: one serializes them on the shared output's
/// hazard, more lets independent launches overlap. `warm_ms` of untimed
/// launches of the same shape run first, so a short shape is not timed while
/// the GPU clock is still ramping. Returns the rate, an FNV-1a hash of the
/// output's bits (two kernels' results compare without a reference), and the
/// host's encode time per launch in the fastest batch: the batch is committed
/// at the wait, so the rate's interval holds the encode and then the GPU work.
/// One command queue serves the shape, as the engine keeps one; with
/// `queue_per_batch` every timed batch builds its own, which adds ~3 ms a
/// batch and is how the rates in `docs/tuning.md` §13.5z were taken (§13.5za).
fn run_prefill_gemm(
    m: usize,
    n: usize,
    k: usize,
    reps: usize,
    outputs: usize,
    warm_ms: u64,
    queue_per_batch: bool,
) -> Result<(f64, u64, f64)> {
    let device = Device::system_default().unwrap();
    let kernels = lloom_tensor::kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;
    let lhs: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 97) as f32 - 48.0) / 512.0))
        .collect();
    let rhs: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(((i % 89) as f32 - 44.0) / 512.0))
        .collect();
    let lhs = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&lhs[..]),
            options,
        )
        .unwrap();
    let rhs = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&rhs[..]),
            options,
        )
        .unwrap();
    let outputs: Vec<_> = (0..outputs.max(1))
        .map(|_| device.new_buffer(m * n * 2, options).unwrap())
        .collect();
    let mut best = f64::MAX;
    let mut best_encode = 0f64;
    let warm_until = std::time::Instant::now() + std::time::Duration::from_millis(warm_ms);
    let mut timed = 0;
    let shared = (!queue_per_batch).then(|| {
        Commands::new(device.new_command_queue().unwrap(), residency_set.clone()).unwrap()
    });
    while timed < 5 {
        let warming = std::time::Instant::now() < warm_until;
        let fresh = if shared.is_none() {
            Some(Commands::new(device.new_command_queue().unwrap(), residency_set.clone()).unwrap())
        } else {
            None
        };
        let commands = shared.as_ref().or(fresh.as_ref()).unwrap();
        let start = std::time::Instant::now();
        for rep in 0..reps {
            let output = &outputs[rep % outputs.len()];
            let encoder = commands.command_encoder().unwrap();
            // SAFETY: the example owns every buffer bound here and reads none
            // of them before the command buffer that writes them completes.
            unsafe {
                lloom_tensor::kernels::call_mlx_gemm(
                    &device,
                    &encoder,
                    &kernels,
                    GemmDType::F16,
                    (1, m, n, k),
                    &[m * k, k, 1],
                    0,
                    &lhs,
                    &[n * k, 1, k],
                    0,
                    &rhs,
                    output,
                )
            }?;
        }
        let encoded = start.elapsed().as_secs_f64() / reps as f64;
        commands.wait_until_completed().unwrap();
        if !warming {
            let t = start.elapsed().as_secs_f64() / reps as f64;
            if t < best {
                best = t;
                best_encode = encoded;
            }
            timed += 1;
        }
    }
    // SAFETY: every command buffer writing the output has completed above.
    let bits = unsafe { std::slice::from_raw_parts(outputs[0].contents() as *const u16, m * n) };
    let hash = bits.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    Ok((2.0 * (m * n * k) as f64 / best / 1e12, hash, best_encode))
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Gemm,
    /// The prefill projections' shapes at m = 110, 344, 374, 512, 2048, 2392
    /// (the three prompts' last tiles, the prefill tile, two whole prompts),
    /// each beside the tile it ran on.
    PrefillGemm {
        /// Output buffers the 20 launches write in turn (1: serialized).
        #[arg(long, default_value_t = 1)]
        outputs: usize,
        /// Untimed launches of each shape, in ms, before its five timed batches.
        #[arg(long, default_value_t = 0)]
        warm_ms: u64,
        /// A new command queue for every timed batch (the rates published
        /// before `docs/tuning.md` §13.5za) instead of one for the shape.
        #[arg(long)]
        queue_per_batch: bool,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::PrefillGemm {
            outputs,
            warm_ms,
            queue_per_batch,
        } => {
            let shapes = [
                ("w1/w3 (MLP up)", 10752, 2048),
                ("w2 (MLP down)", 2048, 10752),
                ("conv in_proj", 6144, 2048),
                ("q / o / conv out_proj", 2048, 2048),
                ("k / v", 512, 2048),
            ];
            let device = Device::system_default().unwrap();
            for m in [110, 344, 374, 512, 2048, 2392] {
                for (name, n, k) in shapes {
                    let (tflops, hash, encode) =
                        run_prefill_gemm(m, n, k, 20, outputs, warm_ms, queue_per_batch)?;
                    let (bm, bn, bk, wm, wn) = lloom_tensor::kernels::nt_gemm_tile(
                        &device,
                        lloom_tensor::kernels::GemmDType::F16,
                        (m, n, k),
                    );
                    println!(
                        "PREFILL_GEMM m={m} n={n} k={k} {name}: {tflops:.2} TFLOP/s tile={bm}_{bn}_{bk}_{wm}_{wn} out_fnv={hash:016x} encode_us={:.1}",
                        encode * 1e6
                    );
                }
            }
        }
        Task::Gemm => {
            for f32 in [false, true] {
                for n in [512, 1024, 2048, 4096] {
                    run_gemm(f32, n)?;
                }
            }
        }
    }
    Ok(())
}
