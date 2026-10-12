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
    rel_err_at(n, k, dtype, silu, seed, 3.0)
}

/// [`rel_err`] with the input's magnitude `x_scale`: at 1e-3, `mean(x^2)` is
/// below `eps` and the epsilon decides the result.
fn rel_err_at(
    n: usize,
    k: usize,
    dtype: DType,
    silu: bool,
    seed: u64,
    x_scale: f32,
) -> Result<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x = round(&rand_vec(&mut rng, k, x_scale), dtype)?;
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
    // `f32::max` drops a NaN, so a non-finite output would read as no error.
    assert!(
        got.iter().all(|v| v.is_finite()),
        "gemv_norm [{n}, {k}]: a non-finite output"
    );
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
    // A small input, where `eps` is most of the denominator.
    let e = rel_err_at(2048, 2048, DType::F32, false, 99, 1e-3)?;
    println!("gemv_norm n=2048 k=2048 F32 |x|~1e-3 rel_err={e:.3e}");
    assert!(
        e <= 1e-5,
        "gemv_norm at |x|~1e-3 (eps-dominated): rel_err {e}"
    );
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

/// `gemv_norm_serves` names exactly the shapes the fused kernel runs at: true
/// at every LFM2 consumer of a norm (`q`, `k`/`v`, conv `in_proj`, `w1`/`w3`,
/// `lm_head`), false where the GEMV takes another tile (`k <= 64`;
/// `k >= 16 n`) — and where it says true the kernel runs, where false it is
/// refused, so a caller that checks it never meets the refusal.
#[test]
fn gemv_norm_serves_the_tiles_it_runs_at() -> Result<()> {
    use lloom::nn::ops::gemv_norm_serves;
    let dev = Device::new_metal(0)?;
    for &(n, k, serves) in &[
        (2048, 2048, true),
        (512, 2048, true),
        (6144, 2048, true),
        (8192, 2048, true),
        (65536, 2048, true),
        (37, 2048, false),
        (16, 8, false),
        (40, 64, false),
    ] {
        assert_eq!(gemv_norm_serves(n, k), serves, "gemv_norm_serves({n}, {k})");
        let x = Tensor::ones((1, 1, k), DType::F32, &dev)?;
        let w = Tensor::ones((n, k), DType::F32, &dev)?;
        let g = Tensor::ones(k, DType::F32, &dev)?;
        let ran = lloom::nn::ops::gemv_norm(&x, &w, &g, EPS, false)
            .and_then(|y| y.flatten_all()?.to_vec1::<f32>());
        assert_eq!(
            ran.is_ok(),
            serves,
            "gemv_norm at [{n}, {k}]: serves={serves} but ran={:?}",
            ran.err()
        );
    }
    Ok(())
}

/// `silu(gate·n(x)) * (up·n(x))` in f64 over the dtype-rounded inputs.
fn reference_swiglu(
    x: &[f32],
    g: &[f32],
    up: &[f32],
    gate: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / k as f64;
    let inv = 1.0 / (ms + EPS as f64).sqrt();
    (0..n)
        .map(|r| {
            let dot = |w: &[f32]| -> f64 {
                (0..k)
                    .map(|c| w[r * k + c] as f64 * x[c] as f64 * g[c] as f64 * inv)
                    .sum()
            };
            let (u, a) = (dot(up), dot(gate));
            (a / (1.0 + (-a).exp()) * u) as f32
        })
        .collect()
}

/// `gemv_norm_swiglu` at `(n, k)`, the largest error relative to the largest
/// reference magnitude; `x_scale` as in [`rel_err_at`].
fn swiglu_rel_err(n: usize, k: usize, dtype: DType, seed: u64, x_scale: f32) -> Result<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x = round(&rand_vec(&mut rng, k, x_scale), dtype)?;
    let g = round(&rand_vec(&mut rng, k, 1.5), dtype)?;
    let up = round(&rand_vec(&mut rng, n * k, 0.05), dtype)?;
    let gate = round(&rand_vec(&mut rng, n * k, 0.05), dtype)?;
    let want = reference_swiglu(&x, &g, &up, &gate, n, k);
    let dev = Device::new_metal(0)?;
    let xt = Tensor::from_vec(x, (1, 1, k), &dev)?.to_dtype(dtype)?;
    let gt = Tensor::from_vec(g, k, &dev)?.to_dtype(dtype)?;
    let ut = Tensor::from_vec(up, (n, k), &dev)?.to_dtype(dtype)?;
    let at = Tensor::from_vec(gate, (n, k), &dev)?.to_dtype(dtype)?;
    let got = lloom::nn::ops::gemv_norm_swiglu(&xt, &ut, &at, &gt, EPS)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got.len(), n);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "gemv_norm_swiglu [{n}, {k}]: a non-finite output"
    );
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    Ok(got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale)
}

/// The two-weight kernel with the norm inside (`gemv_norm_swiglu`): against
/// the f64 reference at both tiles (`bm4` below N = 4096, `bm8` from it — LFM2's
/// `[6144, 2048]` `w1`/`w3`), an N off the tile, a K with a leftover, an
/// eps-dominated input; against the unfused `rmsnorm` + `gemv_swiglu` on the
/// device; bit-stable run to run.
#[test]
fn gemv_norm_swiglu_matches_the_reference_and_the_pair() -> Result<()> {
    for &(n, k) in &[(2048, 2048), (203, 2048), (6144, 2048), (40, 200)] {
        for dtype in [DType::F32, DType::F16] {
            let e = swiglu_rel_err(n, k, dtype, (n * 17 + k) as u64, 3.0)?;
            let tol = if dtype == DType::F32 { 1e-5 } else { 4e-3 };
            println!("gemv_norm_swiglu n={n} k={k} {dtype:?} rel_err={e:.3e} (tol {tol:.0e})");
            assert!(
                e <= tol,
                "gemv_norm_swiglu n={n} k={k} {dtype:?}: rel_err {e} > {tol}"
            );
        }
    }
    let e = swiglu_rel_err(2048, 2048, DType::F32, 98, 1e-3)?;
    println!("gemv_norm_swiglu n=2048 k=2048 F32 |x|~1e-3 rel_err={e:.3e}");
    assert!(
        e <= 1e-5,
        "gemv_norm_swiglu at |x|~1e-3 (eps-dominated): rel_err {e}"
    );

    let (n, k) = (6144, 2048);
    let dev = Device::new_metal(0)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let f16 = |v: Vec<f32>, s: &[usize]| -> Result<Tensor> {
        Tensor::from_vec(v, s, &dev)?.to_dtype(DType::F16)
    };
    let x = f16(rand_vec(&mut rng, k, 3.0), &[1, 1, k])?;
    let g = f16(rand_vec(&mut rng, k, 1.5), &[k])?;
    let up = f16(rand_vec(&mut rng, n * k, 0.05), &[n, k])?;
    let gate = f16(rand_vec(&mut rng, n * k, 0.05), &[n, k])?;
    let norm = lloom::nn::RmsNorm::new(g.clone(), EPS as f64);
    let pair = lloom::nn::ops::gemv_swiglu(&lloom_tensor::Module::forward(&norm, &x)?, &up, &gate)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let fused = || -> Result<Vec<u16>> {
        lloom::nn::ops::gemv_norm_swiglu(&x, &up, &gate, &g, EPS)?
            .flatten_all()?
            .to_vec1::<half::f16>()
            .map(|v| v.iter().map(|h| h.to_bits()).collect())
    };
    let a = fused()?;
    for i in 1..4 {
        assert_eq!(
            a,
            fused()?,
            "gemv_norm_swiglu is not bit-stable run to run (run {i})"
        );
    }
    let scale = pair.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = a
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .zip(&pair)
        .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
        / scale;
    println!("gemv_norm_swiglu vs rmsnorm+gemv_swiglu at [{n}, {k}] f16: rel_err {err:.3e}");
    assert!(err < 4e-3, "gemv_norm_swiglu drifts from the pair: {err}");
    Ok(())
}
