//! One decode step's attention at a head dim, timed per arm: FlashDecoding,
//! `sdpa_vector` where it is instantiated, the composed attention otherwise.
//! The widened FlashDecoding dims' null check (`docs/porting-models.md`
//! §21.8): is the new instantiation slower than the kernel a route would
//! take without it?
//!
//! LFM2's shape: 32 query heads over 8 KV heads, f16, a contiguous cache of
//! `kv_len` keys. A sample is `calls` back-to-back steps in one command
//! stream, host encode included, wall time over `calls`; the arms are
//! interleaved per round, and the null arm is the first arm run again in a
//! second slot, so the null ratio is the noise floor of a ratio.
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/attn_decode_ab --head-dims 64,72,80,256,512
//! ```

use anyhow::Result;
use clap::Parser;
use lloom::nn::ops::{flash_decoding, sdpa, softmax_last_dim, FlashScratchSizing};
use lloom::utils::repeat_kv;
use lloom_tensor::{DType, Device, Tensor};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "64,72,80,256,512")]
    head_dims: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "256,1024,4096")]
    kv_lens: Vec<usize>,
    /// Steps per sample.
    #[arg(long, default_value_t = 200)]
    calls: usize,
    /// Interleaved rounds; each round takes one sample per arm.
    #[arg(long, default_value_t = 15)]
    rounds: usize,
    #[arg(long, default_value_t = 32)]
    q_heads: usize,
    #[arg(long, default_value_t = 8)]
    kv_heads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    Flash,
    Vector,
    Composed,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Flash => "flash_decoding",
            Arm::Vector => "sdpa_vector",
            Arm::Composed => "composed",
        }
    }
}

fn step(arm: Arm, q: &Tensor, k: &Tensor, v: &Tensor, n_rep: usize) -> Result<Tensor> {
    let hd = q.dim(3)?;
    let scale = 1f32 / (hd as f32).sqrt();
    Ok(match arm {
        Arm::Flash => flash_decoding(q, k, v, scale, 1.0, 256, 1, FlashScratchSizing::Grow, 0)?,
        Arm::Vector => sdpa(q, k, v, None, false, scale, 1.0)?,
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

/// Microseconds per step over one sample of `calls` steps.
fn sample(
    arm: Arm,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_rep: usize,
    calls: usize,
    dev: &Device,
) -> Result<f64> {
    dev.synchronize()?;
    let t = Instant::now();
    for _ in 0..calls {
        step(arm, q, k, v, n_rep)?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / calls as f64)
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dev = Device::new_metal(0)?;
    let n_rep = a.q_heads / a.kv_heads;
    println!(
        "attn_decode_ab: q_heads {} kv_heads {} f16 calls {} rounds {} commit {}",
        a.q_heads,
        a.kv_heads,
        a.calls,
        a.rounds,
        option_env!("LLOOM_COMMIT").unwrap_or("-")
    );
    for &hd in &a.head_dims {
        let reference = if lloom_tensor::kernels::SDPA_VECTOR_HEAD_DIMS.contains(&hd) {
            Arm::Vector
        } else {
            Arm::Composed
        };
        for &kv in &a.kv_lens {
            let q = Tensor::randn(0f32, 1.0, (1, a.q_heads, 1, hd), &dev)?.to_dtype(DType::F16)?;
            let k =
                Tensor::randn(0f32, 1.0, (1, a.kv_heads, kv, hd), &dev)?.to_dtype(DType::F16)?;
            let v =
                Tensor::randn(0f32, 1.0, (1, a.kv_heads, kv, hd), &dev)?.to_dtype(DType::F16)?;
            // The two arms agree before either is timed.
            let x = step(Arm::Flash, &q, &k, &v, n_rep)?.to_dtype(DType::F32)?;
            let y = step(reference, &q, &k, &v, n_rep)?.to_dtype(DType::F32)?;
            let diff = (x - y)?.abs()?.max_all()?.to_scalar::<f32>()?;
            // Warm: pipelines compiled, clocks up.
            for arm in [Arm::Flash, reference] {
                sample(arm, &q, &k, &v, n_rep, a.calls, &dev)?;
            }
            let slots = [
                (Arm::Flash, "flash"),
                (reference, "reference"),
                (Arm::Flash, "null"),
            ];
            let mut us: Vec<Vec<f64>> = vec![Vec::new(); slots.len()];
            for round in 0..a.rounds {
                // Rotate the slot order per round so no arm always runs first.
                for i in 0..slots.len() {
                    let s = (i + round) % slots.len();
                    us[s].push(sample(slots[s].0, &q, &k, &v, n_rep, a.calls, &dev)?);
                }
            }
            for (s, (arm, slot)) in slots.iter().enumerate() {
                println!(
                    "SAMPLES hd={hd} kv={kv} slot={slot} arm={} us={}",
                    arm.name(),
                    us[s]
                        .iter()
                        .map(|x| format!("{x:.3}"))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            let med: Vec<f64> = us
                .iter_mut()
                .map(|v| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    quantile(v, 0.5)
                })
                .collect();
            println!(
                "RESULT hd={hd} kv={kv} flash_us={:.3} [{:.3}..{:.3}] {}_us={:.3} [{:.3}..{:.3}] \
                 flash/{}={:.3} null flash/flash={:.3} max|flash-{}|={diff:.2e}",
                med[0],
                quantile(&us[0], 0.25),
                quantile(&us[0], 0.75),
                reference.name(),
                med[1],
                quantile(&us[1], 0.25),
                quantile(&us[1], 0.75),
                reference.name(),
                med[0] / med[1],
                med[2] / med[0],
                reference.name(),
            );
        }
    }
    Ok(())
}
