//! A direct kernel benchmark of the weight kinds at Qwen4-exp's decode
//! shapes: the plain matmul-vec over a dense stem, the routed gate|up pair
//! and down over stacked experts (one token, ten of them), and the GEMM at a
//! prefill tile — MXFP4 and FP8 E4M3 beside Q4_K and Q8_0, IQ4_NL (the core
//! MXFP4 shares) and a second Q8_0 arm as the null control.
//!
//! Each shape's weights are cycled over enough copies that a launch reads
//! from memory, not the system cache. The arms take turns within every
//! round, in an order that rotates per round, so drift is shared. GPU time
//! is the profiler's busy union (`CANDLE_METAL_PROFILE=1`), the wall beside
//! it; GB/s is over the weight bytes a launch reads.
//!
//! A measurement (ignored): `cargo test --release -p lloom --features metal
//! --test quant_kind_throughput -- --ignored --nocapture` under `lloom-arb
//! measure`. `LLOOM_KIND_BENCH_ROUNDS` (default 7).
#![cfg(feature = "metal")]

use lloom::testing::tiny::packed::{self, Synthesis};
use lloom::testing::tiny::{Role, ValueLaw};
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::quantized::{GgmlDType, MoeArm, QTensor};
use lloom_tensor::{Device, Shape, Tensor};
use std::time::Instant;

/// `(label, kind)`; the second Q8_0 is the null control.
const ARMS: [(&str, GgmlDType); 6] = [
    ("Q4K", GgmlDType::Q4K),
    ("Q8_0", GgmlDType::Q8_0),
    ("IQ4NL", GgmlDType::IQ4NL),
    ("MXFP4", GgmlDType::MXFP4),
    ("FP8E4M3", GgmlDType::FP8E4M3),
    ("Q8_0-null", GgmlDType::Q8_0),
];

const HIDDEN: usize = 2560;
const INTER: usize = 640;
const TOPK: usize = 10;
const EXPERTS: usize = 32;

fn weight(kind: GgmlDType, shape: &[usize], name: &str, dev: &Device) -> QTensor {
    let shape = Shape::from(shape.to_vec());
    let fan_in = *shape.dims().last().unwrap();
    packed::qtensor(
        ValueLaw::DEFAULT,
        7,
        name,
        Role::Matrix { fan_in },
        &shape,
        kind,
        Synthesis::Packed,
        dev,
    )
    .unwrap()
}

/// One timed batch of `iters` calls of `f(i)`: (GPU µs, wall µs) a call.
fn batch(dev: &Device, iters: usize, f: &dyn Fn(usize)) -> (f64, f64) {
    dev.synchronize().unwrap();
    profile::reset();
    let t = Instant::now();
    for i in 0..iters {
        f(i);
    }
    dev.synchronize().unwrap();
    let wall = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let gpu = profile::snapshot().gpu_busy_union_s * 1e6 / iters as f64;
    (gpu, wall)
}

struct Arm<'a> {
    label: &'static str,
    bytes: usize,
    run: Box<dyn Fn(usize) + 'a>,
}

/// Warm every arm, then `rounds` rounds of every arm in a rotating order;
/// one `KINDBENCH` line per (round, arm) and a summary per arm.
fn bench(dev: &Device, shape: &str, iters: usize, rounds: usize, arms: &[Arm]) {
    for a in arms {
        dev.synchronize().unwrap();
        profile::reset();
        (a.run)(0);
        dev.synchronize().unwrap();
        let labels: Vec<String> = profile::snapshot()
            .by_label
            .into_iter()
            .map(|(l, n)| format!("{l}x{n}"))
            .collect();
        println!(
            "KINDLABELS shape={shape} kind={} kernels={labels:?}",
            a.label
        );
    }
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 1.0 {
        for a in arms {
            batch(dev, iters / 4 + 1, &*a.run);
        }
    }
    let mut gpu: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for r in 0..rounds {
        for j in 0..arms.len() {
            let i = (j + r) % arms.len();
            let a = &arms[i];
            let (g, w) = batch(dev, iters, &*a.run);
            println!(
                "KINDBENCH shape={shape} kind={} round={r} slot={j} bytes={} gpu_us={g:.2} wall_us={w:.2}",
                a.label, a.bytes
            );
            gpu[i].push(g);
        }
    }
    for (a, g) in arms.iter().zip(&mut gpu) {
        g.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = g[g.len() / 2];
        println!(
            "KINDSUM shape={shape} kind={} bytes={} gpu_us_median={med:.2} gpu_us_min={:.2} gpu_us_max={:.2} gb_s_median={:.1} n={}",
            a.label,
            a.bytes,
            g[0],
            g[g.len() - 1],
            a.bytes as f64 / med / 1e3,
            g.len()
        );
    }
}

#[test]
#[ignore]
fn quant_kind_throughput() {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0).unwrap();
    assert!(profile::enabled(), "GPU time is the profiler's busy union");
    let rounds: usize = std::env::var("LLOOM_KIND_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    println!(
        "KINDBENCH-CONFIG rounds={rounds} hidden={HIDDEN} inter={INTER} topk={TOPK} experts={EXPERTS} profile=busy_union build={}",
        if cfg!(debug_assertions) { "debug" } else { "release" }
    );

    // The dense decode stems: [n, k] at m = 1, eight copies cycled.
    for (n, k) in [(4 * HIDDEN, HIDDEN), (HIDDEN, 4 * HIDDEN)] {
        let copies = 8;
        let x = Tensor::randn(0f32, 1.0, (1, k), &dev).unwrap();
        let w: Vec<Vec<QTensor>> = ARMS
            .iter()
            .map(|(l, kind)| {
                (0..copies)
                    .map(|c| weight(*kind, &[n, k], &format!("dense.{l}.{c}"), &dev))
                    .collect()
            })
            .collect();
        let arms: Vec<Arm> = ARMS
            .iter()
            .zip(&w)
            .map(|((l, kind), ws)| {
                let x = &x;
                Arm {
                    label: l,
                    bytes: n * k / kind.block_size() * kind.type_size(),
                    run: Box::new(move |i| {
                        let _ = x.apply_op1_no_bwd(&ws[i % ws.len()]).unwrap();
                    }),
                }
            })
            .collect();
        bench(&dev, &format!("mul_mv[{n}x{k}]"), 64, rounds, &arms);
    }

    // The routed experts: one token, TOPK of EXPERTS, six stacks cycled;
    // gate|up in one launch, then down.
    {
        let stacks = 6;
        let x = Tensor::randn(0f32, 1.0, (1, 1, HIDDEN), &dev).unwrap();
        let h = Tensor::randn(0f32, 1.0, (1, TOPK, INTER), &dev).unwrap();
        let ids: Vec<Tensor> = (0..16u32)
            .map(|s| {
                let mut v: Vec<u32> = (0..TOPK as u32)
                    .map(|i| (i * 7 + s * 3) % EXPERTS as u32)
                    .collect();
                v.sort_unstable();
                v.dedup();
                while v.len() < TOPK {
                    let next = (v.last().unwrap() + 1) % EXPERTS as u32;
                    if !v.contains(&next) {
                        v.push(next);
                    }
                    v.sort_unstable();
                }
                Tensor::from_vec(v, (1, TOPK), &dev).unwrap()
            })
            .collect();
        type Stack = (QTensor, QTensor, QTensor);
        let w: Vec<Vec<Stack>> = ARMS
            .iter()
            .map(|(l, kind)| {
                (0..stacks)
                    .map(|s| {
                        (
                            weight(
                                *kind,
                                &[EXPERTS, INTER, HIDDEN],
                                &format!("gate.{l}.{s}"),
                                &dev,
                            ),
                            weight(
                                *kind,
                                &[EXPERTS, INTER, HIDDEN],
                                &format!("up.{l}.{s}"),
                                &dev,
                            ),
                            weight(
                                *kind,
                                &[EXPERTS, HIDDEN, INTER],
                                &format!("down.{l}.{s}"),
                                &dev,
                            ),
                        )
                    })
                    .collect()
            })
            .collect();
        let expert = |kind: GgmlDType| INTER * HIDDEN / kind.block_size() * kind.type_size();
        let pair: Vec<Arm> = ARMS
            .iter()
            .zip(&w)
            .map(|((l, kind), ws)| {
                let (x, ids) = (&x, &ids);
                Arm {
                    label: l,
                    bytes: 2 * TOPK * expert(*kind),
                    run: Box::new(move |i| {
                        let (g, u, _) = &ws[i % ws.len()];
                        let _ = g
                            .indexed_moe_forward_pair_with(
                                u,
                                x,
                                &ids[i % ids.len()],
                                MoeArm::MatVec,
                            )
                            .unwrap();
                    }),
                }
            })
            .collect();
        bench(
            &dev,
            &format!("mul_mv_id2[gate|up {EXPERTS}x{INTER}x{HIDDEN} top{TOPK}]"),
            64,
            rounds,
            &pair,
        );
        let down: Vec<Arm> = ARMS
            .iter()
            .zip(&w)
            .map(|((l, kind), ws)| {
                let (h, ids) = (&h, &ids);
                Arm {
                    label: l,
                    bytes: TOPK * expert(*kind),
                    run: Box::new(move |i| {
                        let (_, _, d) = &ws[i % ws.len()];
                        let _ = d
                            .indexed_moe_forward_with(h, &ids[i % ids.len()], MoeArm::MatVec)
                            .unwrap();
                    }),
                }
            })
            .collect();
        bench(
            &dev,
            &format!("mul_mv_id[down {EXPERTS}x{HIDDEN}x{INTER} top{TOPK}]"),
            64,
            rounds,
            &down,
        );
    }

    // The prefill GEMM at a 512-row tile over one stem (two copies): as
    // served (the stem dequantised to half once, then the f16 GEMM), and the
    // kind's own GEMM with the dequant-once arm off.
    for direct in [false, true] {
        lloom_tensor::quantized::metal::set_dequant_once_from(if direct {
            0
        } else {
            lloom_tensor::quantized::metal::DEFAULT_DEQUANT_ONCE_FROM
        });
        let (m, n, k) = (512usize, 4 * HIDDEN, HIDDEN);
        let x = Tensor::randn(0f32, 1.0, (m, k), &dev).unwrap();
        let w: Vec<Vec<QTensor>> = ARMS
            .iter()
            .map(|(l, kind)| {
                (0..2)
                    .map(|c| weight(*kind, &[n, k], &format!("mm.{l}.{c}"), &dev))
                    .collect()
            })
            .collect();
        let arms: Vec<Arm> = ARMS
            .iter()
            .zip(&w)
            .map(|((l, kind), ws)| {
                let x = &x;
                Arm {
                    label: l,
                    bytes: n * k / kind.block_size() * kind.type_size(),
                    run: Box::new(move |i| {
                        let _ = x.apply_op1_no_bwd(&ws[i % ws.len()]).unwrap();
                    }),
                }
            })
            .collect();
        let arm = if direct { "direct" } else { "dequant-once" };
        bench(
            &dev,
            &format!("mul_mm[{arm} m{m} {n}x{k}]"),
            8,
            rounds,
            &arms,
        );
    }
}
