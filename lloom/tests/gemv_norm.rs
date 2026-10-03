//! `nn::ops::gemv_norm`: a decode GEMV with its RMSNorm computed inside it
//! (`docs/tuning.md` §13.5t). Checked against a CPU reference computed the
//! way the unfused pair defines it — `rmsnorm(x) = x * w / sqrt(mean(x^2) +
//! eps)`, then `y = W · rmsnorm(x)`, then SiLU when asked — at both decode
//! tiles (`bm4` below N = 4096, `bm8` from it), at an N that is not a multiple
//! of the tile and a K the K-walk does not divide (its leftover branch), and
//! against the unfused pair on the device.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};

const EPS: f32 = 1e-5;

fn rand_vec(rng: &mut rand::rngs::StdRng, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| rng.random_range(-1.0f32..1.0) * scale)
        .collect()
}

fn round(v: &[f32], dtype: DType) -> Result<Vec<f32>> {
    Tensor::new(v, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
}

/// The pair's arithmetic in f64 over the dtype-rounded inputs.
fn reference(x: &[f32], g: &[f32], w: &[f32], n: usize, k: usize, silu: bool) -> Vec<f32> {
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64;
    let inv = 1.0 / (ms + EPS as f64).sqrt();
    (0..n)
        .map(|r| {
            let dot: f64 = (0..k)
                .map(|c| w[r * k + c] as f64 * x[c] as f64 * g[c] as f64 * inv)
                .sum();
            let y = if silu {
                dot / (1.0 + (-dot).exp())
            } else {
                dot
            };
            y as f32
        })
        .collect()
}

/// `gemv_norm` at `(n, k)` and `dtype`, the largest error relative to the
/// largest reference magnitude.
fn rel_err(n: usize, k: usize, dtype: DType, silu: bool, seed: u64) -> Result<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x = round(&rand_vec(&mut rng, k, 3.0), dtype)?;
    let g = round(&rand_vec(&mut rng, k, 1.5), dtype)?;
    let w = round(&rand_vec(&mut rng, n * k, 0.05), dtype)?;
    let want = reference(&x, &g, &w, n, k, silu);
    let dev = Device::new_metal(0)?;
    let xt = Tensor::from_vec(x, (1, 1, k), &dev)?.to_dtype(dtype)?;
    let gt = Tensor::from_vec(g, k, &dev)?.to_dtype(dtype)?;
    let wt = Tensor::from_vec(w, (n, k), &dev)?.to_dtype(dtype)?;
    let got = lloom::nn::ops::gemv_norm(&xt, &wt, &gt, EPS, silu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got.len(), n);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    Ok(got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale)
}

#[test]
fn gemv_norm_matches_the_pair_reference() -> Result<()> {
    // (n, k): the bm4 tile (2048, 512 and 203 rows: not a multiple of the
    // tile's 16), the bm8 tile (6144),
    // and a K of 200, which the 128-wide K-walk leaves a remainder of.
    let shapes = [
        (2048, 2048),
        (512, 2048),
        (203, 2048),
        (6144, 2048),
        (40, 200),
    ];
    for &(n, k) in &shapes {
        for dtype in [DType::F32, DType::F16] {
            for silu in [false, true] {
                let e = rel_err(n, k, dtype, silu, (n * 31 + k) as u64)?;
                let tol = if dtype == DType::F32 { 1e-5 } else { 4e-3 };
                println!(
                    "gemv_norm n={n} k={k} {dtype:?} silu={silu} rel_err={e:.3e} (tol {tol:.0e})"
                );
                assert!(
                    e <= tol,
                    "gemv_norm n={n} k={k} {dtype:?} silu={silu}: rel_err {e} > {tol}"
                );
            }
        }
    }
    Ok(())
}

/// The fused kernel against the unfused pair ON THE DEVICE at LFM2's shape:
/// close (a different rounding, not a different function) and bit-stable
/// run to run.
#[test]
fn gemv_norm_agrees_with_the_pair_and_is_bit_stable() -> Result<()> {
    let (n, k) = (6144, 2048);
    let dev = Device::new_metal(0)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let x = Tensor::from_vec(rand_vec(&mut rng, k, 3.0), (1, 1, k), &dev)?.to_dtype(DType::F16)?;
    let g = Tensor::from_vec(rand_vec(&mut rng, k, 1.5), k, &dev)?.to_dtype(DType::F16)?;
    let w =
        Tensor::from_vec(rand_vec(&mut rng, n * k, 0.05), (n, k), &dev)?.to_dtype(DType::F16)?;
    let norm = lloom::nn::RmsNorm::new(g.clone(), EPS as f64);
    let pair = lloom_tensor::Module::forward(&norm, &x)?
        .broadcast_matmul(&w.t()?)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let fused = |_: usize| -> Result<Vec<u16>> {
        lloom::nn::ops::gemv_norm(&x, &w, &g, EPS, false)?
            .flatten_all()?
            .to_vec1::<half::f16>()
            .map(|v| v.iter().map(|h| h.to_bits()).collect())
    };
    let a = fused(0)?;
    for i in 1..4 {
        assert_eq!(
            a,
            fused(i)?,
            "gemv_norm is not bit-stable run to run (run {i})"
        );
    }
    let f: Vec<f32> = a
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();
    let scale = pair.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = f
        .iter()
        .zip(&pair)
        .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
        / scale;
    println!("gemv_norm vs rmsnorm+gemv at [{n}, {k}] f16: rel_err {err:.3e}");
    assert!(err < 4e-3, "gemv_norm drifts from the pair: {err}");
    Ok(())
}
