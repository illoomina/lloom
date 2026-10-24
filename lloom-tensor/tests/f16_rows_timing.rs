//! The f16 row axis timed at LFM2.5-2.6B's weight shapes (`docs/tuning.md`
//! §13.5y): a verify window's `m`-row matmul on the GEMM (`fast` with the
//! arm off: what ships), on the row axis at capacity `m`, as `m` one-row
//! GEMVs, and one row alone — wall per launch over `ITERS` launches, the
//! encode included, median of `REPS`, each launch on the next of enough
//! weight copies that no repeat is served from the system cache. The
//! `window` lines weight each shape by its count in one forward. A
//! measurement, ignored by default; run by hand under `lloom-arb measure`
//! with `--ignored --nocapture` (`F16_ROWS_M=2,10` picks the row counts,
//! `F16_ROWS_SHAPES=6144x2048` the shapes, `F16_ROWS_SITES=label:count:NxK;…`
//! another model's sites, `F16_ROWS_SCOPE=decode` a batched decode step's
//! route instead of a verify window's — the same kernels — and
//! `F16_ROWS_PER_ROW=0` drops the `m` one-row arm).
#![cfg(feature = "metal")]
use lloom_tensor::tuning::{
    f16_rows_decode_launches, set_f16_rows, set_f16_rows_decode, Contract, DecodeRows, VerifyWindow,
};
use lloom_tensor::{DType, Device, Tensor};
use std::time::Instant;

const ITERS: usize = 40;
const REPS: usize = 7;
/// Bytes of weight copies a shape cycles through: four times the M1 Max's
/// 48 MB system cache.
const ROTATION_BYTES: usize = 192 << 20;

/// `(label, count in one LFM2.5-2.6B forward, n_out, k)`.
const SITES: &[(&str, usize, usize, usize)] = &[
    ("conv.in_proj", 22, 6144, 2048),
    ("conv.out_proj", 22, 2048, 2048),
    ("attn.qkv", 8, 3072, 2048),
    ("attn.out_proj", 8, 2048, 2048),
    ("ffn.w1|w3", 60, 10752, 2048),
    ("ffn.w2", 30, 2048, 10752),
    ("lm_head", 1, 128000, 2048),
];

fn median_ms(dev: &Device, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..3 {
        f(i);
    }
    dev.synchronize().unwrap();
    let mut ms = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for i in 0..ITERS {
            f(i);
        }
        dev.synchronize().unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[REPS / 2]
}

fn list(var: &str) -> Option<Vec<String>> {
    std::env::var(var)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
}

/// `F16_ROWS_SITES`, `label:count:NxK` separated by `;`, or LFM2.5's.
fn sites() -> Vec<(String, usize, usize, usize)> {
    match std::env::var("F16_ROWS_SITES") {
        Ok(v) if !v.trim().is_empty() => v
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                let f: Vec<&str> = s.trim().split(':').collect();
                let (n, k) = f[2].split_once('x').expect("NxK");
                (
                    f[0].to_string(),
                    f[1].parse().unwrap(),
                    n.parse().unwrap(),
                    k.parse().unwrap(),
                )
            })
            .collect(),
        _ => SITES
            .iter()
            .map(|&(l, c, n, k)| (l.to_string(), c, n, k))
            .collect(),
    }
}

/// The route an `m`-row matmul is timed under: a verify window's, or a
/// batched decode step's with the decode rows' switch at `rows`.
fn scoped<R>(decode: bool, rows: bool, f: impl FnOnce() -> R) -> R {
    if decode {
        set_f16_rows_decode(rows);
        let _s = DecodeRows::open();
        f()
    } else {
        let _w = VerifyWindow::open();
        f()
    }
}

#[test]
#[ignore]
fn the_row_axis_against_the_gemm_at_lfm2s_shapes() {
    lloom_tensor::tuning::pin_contract(Contract::Fast).unwrap();
    let dev = Device::new_metal(0).unwrap();
    let ms: Vec<usize> = list("F16_ROWS_M")
        .map(|v| v.iter().map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2, 4, 6, 8, 10, 12, 16]);
    let shapes = list("F16_ROWS_SHAPES");
    let decode = std::env::var("F16_ROWS_SCOPE").is_ok_and(|v| v == "decode");
    let per_row_arm = std::env::var("F16_ROWS_PER_ROW").map_or(true, |v| v != "0");
    let sites = sites();
    println!(
        "F16_ROWS_TIMING contract=fast scope={} iters={ITERS} reps={REPS} rotation_bytes={ROTATION_BYTES} \
         median wall ms per launch (encode included)",
        if decode { "decode" } else { "verify" }
    );
    println!("| site | m | gemm (ships) | row axis | m x one-row | one row | axis / gemm | axis / one row |");
    println!("|---|---|---|---|---|---|---|---|");
    // window[m] = (gemm, axis, m one-rows) summed over a forward's sites.
    let mut window = vec![(0f64, 0f64, 0f64); 17];
    for (label, count, n, k) in sites.iter().map(|(l, c, n, k)| (l.as_str(), *c, *n, *k)) {
        if let Some(only) = &shapes {
            if !only.iter().any(|s| *s == format!("{n}x{k}")) {
                continue;
            }
        }
        let copies = (ROTATION_BYTES / (n * k * 2)).clamp(1, 32);
        let ws: Vec<Tensor> = (0..copies)
            .map(|c| {
                Tensor::randn(0f32, 0.02, (n, k), &dev)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap()
                    .affine(1.0, c as f64 * 1e-3)
                    .unwrap()
                    .t()
                    .unwrap()
            })
            .collect();
        let x1 = Tensor::randn(0f32, 1.0, (1, k), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let one = median_ms(&dev, |i| {
            x1.matmul(&ws[i % copies]).unwrap();
        });
        for &m in &ms {
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let rows: Vec<Tensor> = (0..m).map(|r| x.narrow(0, r, 1).unwrap()).collect();
            set_f16_rows(if decode { Some(16) } else { None });
            let before = (
                lloom_tensor::tuning::f16_rows_launches(),
                f16_rows_decode_launches(),
            );
            let gemm = median_ms(&dev, |i| {
                scoped(decode, false, || x.matmul(&ws[i % copies]).unwrap());
            });
            assert_eq!(
                (
                    lloom_tensor::tuning::f16_rows_launches(),
                    f16_rows_decode_launches()
                ),
                before,
                "{label} m={m}: the GEMM arm took the row axis"
            );
            set_f16_rows(Some(16));
            let axis = median_ms(&dev, |i| {
                scoped(decode, true, || x.matmul(&ws[i % copies]).unwrap());
            });
            let after = if decode {
                f16_rows_decode_launches() > before.1
            } else {
                lloom_tensor::tuning::f16_rows_launches() > before.0
            };
            assert!(after, "{label} m={m}: the row axis did not engage");
            set_f16_rows(None);
            let per_row = if per_row_arm {
                median_ms(&dev, |i| {
                    for x in &rows {
                        x.matmul(&ws[i % copies]).unwrap();
                    }
                })
            } else {
                f64::NAN
            };
            println!(
                "| {label} [{n}x{k}] | {m} | {gemm:.4} | {axis:.4} | {per_row:.4} | {one:.4} | {:.3} | {:.2} |",
                axis / gemm,
                axis / one
            );
            let w = &mut window[m];
            w.0 += count as f64 * gemm;
            w.1 += count as f64 * axis;
            w.2 += count as f64 * per_row;
        }
    }
    if shapes.is_none() {
        for &m in &ms {
            let (g, a, p) = window[m];
            println!(
                "F16_ROWS_WINDOW m={m} weight_matmuls_ms gemm={g:.2} row_axis={a:.2} one_row_x_m={p:.2} axis_over_gemm={:.3}",
                a / g
            );
        }
    }
}
