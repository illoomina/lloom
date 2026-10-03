//! One decode step's attention, timed per arm: FlashDecoding, `sdpa_vector`
//! as routed, the single-pass and the 2-pass vector kernels pinned, and the
//! composed attention. Two questions: is a widened FlashDecoding head dim
//! slower than the kernel a route would take without it
//! (`docs/porting-models.md` §21.8), and where the single-pass and 2-pass
//! vector kernels cross over in `kv_len` (`docs/tuning.md` §13.5s).
//!
//! LFM2's shape by default: 32 query heads over 8 KV heads, f16 (`--dtype`), a contiguous
//! cache of `kv_len` keys. A sample is `calls` steps in one command stream;
//! the arms are interleaved per round, and the null slot is the first arm run
//! again, so the null ratio is the noise floor of a ratio.
//!
//! `--chain` feeds each step's output in as the next step's query, so every
//! call waits on the one before it at a barrier, as attention does inside a
//! decode step; without it the calls are independent and overlap. Under
//! `CANDLE_METAL_PROFILE=1` a sample also reads the GPU time (the union of its
//! command buffers' intervals) beside the host wall.
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/attn_decode_ab --head-dims 64,72,80,256,512
//! CANDLE_METAL_PROFILE=1 lloom-arb measure -- ./target/release/examples/attn_decode_ab \
//!     --head-dims 64 --arms single,two-pass --chain --kv-lens 64,256,1024
//! ```

use anyhow::{bail, Result};
use clap::Parser;
use lloom::nn::ops::{
    flash_decoding, sdpa, sdpa_vector_pinned, softmax_last_dim, FlashScratchSizing,
};
use lloom::utils::repeat_kv;
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{DType, Device, Tensor};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "64,72,80,256,512")]
    head_dims: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "256,1024,4096")]
    kv_lens: Vec<usize>,
    /// The arms, in slot order: `flash`, `reference` (`sdpa` as routed where
    /// the head dim is instantiated, the composed attention otherwise),
    /// `single`, `two-pass`. The first is also run in the null slot.
    #[arg(long, value_delimiter = ',', default_value = "flash,reference")]
    arms: Vec<String>,
    /// Each step's output is the next step's query (a dependent chain).
    #[arg(long)]
    chain: bool,
    /// Steps per sample.
    #[arg(long, default_value_t = 200)]
    calls: usize,
    /// Interleaved rounds; each round takes one sample per slot.
    #[arg(long, default_value_t = 15)]
    rounds: usize,
    #[arg(long, default_value_t = 32)]
    q_heads: usize,
    #[arg(long, default_value_t = 8)]
    kv_heads: usize,
    /// `f16` or `f32`.
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// Cycle the steps over enough distinct K/V copies to hold this many MB,
    /// so a step reads its cache from DRAM as in a decode step, where ~5 GB
    /// of weights pass between two layers' attention; 0 reads one copy,
    /// which stays in the system-level cache. At most 4096 copies.
    #[arg(long, default_value_t = 0)]
    cold_mb: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    Flash,
    Vector,
    Single,
    TwoPass,
    Composed,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Flash => "flash_decoding",
            Arm::Vector => "sdpa_vector",
            Arm::Single => "sdpa_vector_1pass",
            Arm::TwoPass => "sdpa_vector_2pass",
            Arm::Composed => "composed",
        }
    }

    fn parse(s: &str, hd: usize) -> Result<Arm> {
        let vector = lloom_tensor::kernels::SDPA_VECTOR_HEAD_DIMS.contains(&hd);
        Ok(match s {
            "flash" => Arm::Flash,
            "reference" if vector => Arm::Vector,
            "reference" => Arm::Composed,
            "single" | "two-pass" if !vector => bail!("hd {hd}: no sdpa_vector instantiation"),
            "single" => Arm::Single,
            "two-pass" => Arm::TwoPass,
            other => bail!("--arms: `{other}` is not flash, reference, single or two-pass"),
        })
    }
}

fn step(arm: Arm, q: &Tensor, k: &Tensor, v: &Tensor, n_rep: usize) -> Result<Tensor> {
    let hd = q.dim(3)?;
    let scale = 1f32 / (hd as f32).sqrt();
    Ok(match arm {
        Arm::Flash => flash_decoding(q, k, v, scale, 1.0, 256, 1, FlashScratchSizing::Grow, 0)?,
        Arm::Vector => sdpa(q, k, v, None, false, scale, 1.0)?,
        Arm::Single => sdpa_vector_pinned(q, k, v, scale, 1.0, false)?,
        Arm::TwoPass => sdpa_vector_pinned(q, k, v, scale, 1.0, true)?,
        Arm::Composed => {
            let k = repeat_kv(k.clone(), n_rep)?.to_dtype(DType::F32)?;
            let v = repeat_kv(v.clone(), n_rep)?.to_dtype(DType::F32)?;
            let att = (q.to_dtype(DType::F32)?.matmul(&k.t()?)? * scale as f64)?;
            softmax_last_dim(&att)?
                .matmul(&v.contiguous()?)?
                .to_dtype(q.dtype())?
        }
    })
}

/// One sample of `calls` steps: (host wall, GPU time) in microseconds per
/// step, the GPU time `None` unless `CANDLE_METAL_PROFILE=1`.
#[allow(clippy::too_many_arguments)]
fn sample(
    arm: Arm,
    q: &Tensor,
    kvs: &[(Tensor, Tensor)],
    n_rep: usize,
    calls: usize,
    chain: bool,
    dev: &Device,
) -> Result<(f64, Option<f64>)> {
    dev.synchronize()?;
    profile::reset();
    let t = Instant::now();
    let mut x = q.clone();
    for i in 0..calls {
        let (k, v) = &kvs[i % kvs.len()];
        let y = step(arm, &x, k, v, n_rep)?;
        if chain {
            x = y;
        }
    }
    dev.synchronize()?;
    let wall = t.elapsed().as_secs_f64() * 1e6 / calls as f64;
    let gpu = profile::enabled().then(|| profile::snapshot().gpu_busy_union_s * 1e6 / calls as f64);
    Ok((wall, gpu))
}

fn quantile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((s.len() - 1) as f64 * p).round() as usize;
    s[i]
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dev = Device::new_metal(0)?;
    let n_rep = a.q_heads / a.kv_heads;
    let profiled = profile::enabled();
    let dtype = match a.dtype.as_str() {
        "f16" => DType::F16,
        "f32" => DType::F32,
        other => bail!("--dtype: `{other}` is not f16 or f32"),
    };
    println!(
        "attn_decode_ab: q_heads {} kv_heads {} {dtype:?} calls {} rounds {} chain {} gpu_time {} \
         two_pass_k_override {} commit {}",
        a.q_heads,
        a.kv_heads,
        a.calls,
        a.rounds,
        a.chain,
        profiled,
        lloom::tuning::tuning().lfm2.sdpa_two_pass_k,
        option_env!("LLOOM_COMMIT").unwrap_or("-")
    );
    if a.arms.is_empty() {
        bail!("--arms is empty");
    }
    for &hd in &a.head_dims {
        let arms = a
            .arms
            .iter()
            .map(|s| Arm::parse(s, hd))
            .collect::<Result<Vec<_>>>()?;
        let mut slots: Vec<(Arm, String)> = arms
            .iter()
            .zip(&a.arms)
            .map(|(&arm, s)| (arm, s.clone()))
            .collect();
        slots.push((arms[0], "null".to_string()));
        for &kv in &a.kv_lens {
            let q = Tensor::randn(0f32, 1.0, (1, a.q_heads, 1, hd), &dev)?.to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1.0, (1, a.kv_heads, kv, hd), &dev)?.to_dtype(dtype)?;
            let v = Tensor::randn(0f32, 1.0, (1, a.kv_heads, kv, hd), &dev)?.to_dtype(dtype)?;
            let copy_bytes = 2 * k.elem_count() * dtype.size_in_bytes();
            // Capped: below ~32 keys a step reads a few KB and stays latency-bound.
            let copies = (a.cold_mb << 20).div_ceil(copy_bytes).clamp(1, 4096);
            let mut kvs = vec![(k.clone(), v.clone())];
            for _ in 1..copies {
                kvs.push((k.copy()?, v.copy()?));
            }
            // The arms agree with the first before any is timed.
            let x = step(arms[0], &q, &k, &v, n_rep)?.to_dtype(DType::F32)?;
            let mut diffs = Vec::new();
            for &arm in &arms[1..] {
                let y = step(arm, &q, &k, &v, n_rep)?.to_dtype(DType::F32)?;
                let d = (&x - y)?.abs()?.max_all()?.to_scalar::<f32>()?;
                diffs.push(format!("max|{}-{}|={d:.2e}", arms[0].name(), arm.name()));
            }
            // Warm: pipelines compiled, clocks up.
            for &arm in &arms {
                sample(arm, &q, &kvs, n_rep, a.calls, a.chain, &dev)?;
            }
            let mut wall: Vec<Vec<f64>> = vec![Vec::new(); slots.len()];
            let mut gpu: Vec<Vec<f64>> = vec![Vec::new(); slots.len()];
            for round in 0..a.rounds {
                // Rotate the slot order per round so no arm always runs first.
                for i in 0..slots.len() {
                    let s = (i + round) % slots.len();
                    let (w, g) = sample(slots[s].0, &q, &kvs, n_rep, a.calls, a.chain, &dev)?;
                    wall[s].push(w);
                    if let Some(g) = g {
                        gpu[s].push(g);
                    }
                }
            }
            let fmt = |v: &[f64]| {
                v.iter()
                    .map(|x| format!("{x:.3}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            for (s, (arm, slot)) in slots.iter().enumerate() {
                println!(
                    "SAMPLES hd={hd} kv={kv} slot={slot} arm={} wall_us={} gpu_us={}",
                    arm.name(),
                    fmt(&wall[s]),
                    fmt(&gpu[s])
                );
            }
            // The verdict reads GPU time when it was taken, the wall otherwise.
            let (series, by) = if profiled {
                (&gpu, "gpu")
            } else {
                (&wall, "wall")
            };
            let base = quantile(&series[0], 0.5);
            let mut line = format!(
                "RESULT hd={hd} kv={kv} chain={} kv_copies={copies} by={by}",
                a.chain
            );
            for (s, (arm, slot)) in slots.iter().enumerate() {
                let m = quantile(&series[s], 0.5);
                line.push_str(&format!(
                    " {slot}:{}={m:.3}us [{:.3}..{:.3}] ratio={:.3} wall={:.3}us",
                    arm.name(),
                    quantile(&series[s], 0.25),
                    quantile(&series[s], 0.75),
                    m / base,
                    quantile(&wall[s], 0.5)
                ));
            }
            for d in &diffs {
                line.push(' ');
                line.push_str(d);
            }
            println!("{line}");
        }
    }
    Ok(())
}
