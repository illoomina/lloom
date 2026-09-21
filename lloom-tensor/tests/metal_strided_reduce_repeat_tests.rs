#![cfg(feature = "metal")]
//! `sum` over a MIDDLE dimension on Metal (`fast_sum_f32_strided`), repeated
//! under GPU load, must be bit-identical every time and equal to the same sum
//! expressed as a batched matmul (issue #748).
//!
//! # History
//!
//! The Qwen4-exp DeltaNet decode first read its state as
//! `S.broadcast_mul(k).sum(1)` over a `[48, 128, 128]` state. On the real
//! model's 3749-token prompt that form diverged between two runs of one
//! process in 8 of 9 executions, the first divergent channel one layer's `S`
//! with that layer's input identical, while the read-outs as batched matmuls
//! diverged less often and elsewhere (the next attention layer's inputs).
//! The kernel was the first suspect; this test (sampling one repetition in
//! ten) and `lloom/tests/metal_reduce_barrier_tests.rs` (every repetition,
//! 20 000 of them) found it bit-repeatable in isolation. The mechanism was
//! `reduce.metal`'s threadgroup barriers carrying no memory fence
//! (`mem_none`), shared by every ≥ 64-thread reduce and the softmax -- see
//! the barrier test and `design/docs/invariants.md`. This test stays as the
//! shape-specific parity + repeatability check for the strided reduce.

use lloom_tensor::{DType, Device, Result, Tensor};

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
fn strided_middle_dim_sum_is_repeatable_under_load() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let (h, dk, dv) = (48usize, 128usize, 128usize);
    let s = Tensor::from_vec(
        (0..h * dk * dv)
            .map(|i| ((i as f32) * 0.0137).sin() * 0.3)
            .collect::<Vec<_>>(),
        (h, dk, dv),
        &dev,
    )?;
    let k = Tensor::from_vec(
        (0..h * dk)
            .map(|i| ((i as f32) * 0.071).cos())
            .collect::<Vec<_>>(),
        (h, dk, 1),
        &dev,
    )?;
    // The matmul form of the same contraction: kᵀ S → [h, 1, dv].
    let k_row = k.reshape((h, 1, dk))?;
    let reference = bits(&k_row.matmul(&s)?.reshape((h, dv))?)?;
    let first = bits(&s.broadcast_mul(&k)?.sum(1)?)?;
    // Load: a large matmul and a big elementwise chain kept in flight between
    // the repetitions, never synced.
    let big = Tensor::from_vec(
        (0..2048 * 2048)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect::<Vec<_>>(),
        (2048, 2048),
        &dev,
    )?;
    let mut keep = Vec::new();
    let mut mismatched_reps = Vec::new();
    for rep in 0..400 {
        let noise = big.matmul(&big)?;
        keep.push(noise.narrow(0, 0, 1)?);
        let r = s.broadcast_mul(&k)?.sum(1)?;
        let x = (&r + 0.0)?;
        keep.push(x.narrow(0, 0, 1)?);
        if rep % 10 == 0 {
            let b = bits(&r)?;
            if b != first {
                mismatched_reps.push(rep);
            }
        }
        if keep.len() > 64 {
            keep.clear();
        }
    }
    assert!(
        mismatched_reps.is_empty(),
        "strided middle-dim sum differed from its first result at repetitions {mismatched_reps:?}"
    );
    // And the two formulations agree to f32 reorder noise (not bit-equal: a
    // different summation order), so the reference is a real reference.
    let a: Vec<f32> = first.iter().map(|&b| f32::from_bits(b)).collect();
    let b: Vec<f32> = reference.iter().map(|&b| f32::from_bits(b)).collect();
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
    let md = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        md <= 1e-4 * scale,
        "reduce vs matmul differ by {md} at scale {scale}"
    );
    let _ = DType::F32;
    Ok(())
}

/// Correctness of the strided reduce against the CPU at the shapes the
/// indexer's A/B arm used (`[937, 4].sum(1)`, `[937, 4, 128].sum(2)`) and the
/// DeltaNet state's (`[48, 128, 128].sum(1)`): the A/B arm went NaN on the
/// real model deterministically, so one of these must be wrong on Metal if
/// the arm's math is right.
#[test]
fn strided_reduce_matches_cpu_at_the_issue_shapes() -> Result<()> {
    let gpu = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let cases: Vec<(Vec<usize>, usize)> = vec![
        (vec![937, 4], 1),
        (vec![526, 4], 1),
        (vec![937, 4, 128], 2),
        (vec![937, 4, 128], 1),
        (vec![48, 128, 128], 1),
        (vec![3, 5, 7], 1),
    ];
    for (shape, dim) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.0173).sin() * 0.7 + 0.1)
            .collect();
        let c = Tensor::from_vec(data.clone(), shape.as_slice(), &cpu)?;
        let g = Tensor::from_vec(data, shape.as_slice(), &gpu)?;
        let want = c.sum(dim)?.flatten_all()?.to_vec1::<f32>()?;
        let got = g
            .sum(dim)?
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(want.len(), got.len(), "{shape:?} dim {dim}: length");
        let nonfinite = got.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(
            nonfinite, 0,
            "{shape:?} dim {dim}: {nonfinite} non-finite on Metal"
        );
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let md = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            md <= 1e-4 * scale,
            "{shape:?} dim {dim}: Metal differs from CPU by {md} at scale {scale}"
        );
        // the broadcast-mul + reduce the arm composed, end to end
        if shape.len() == 3 && dim == 2 {
            let q = Tensor::from_vec(
                (0..4 * 128)
                    .map(|i| ((i as f32) * 0.05).cos())
                    .collect::<Vec<_>>(),
                (1, 4, 128),
                &cpu,
            )?;
            let k = Tensor::from_vec(
                (0..937 * 128)
                    .map(|i| ((i as f32) * 0.011).sin())
                    .collect::<Vec<_>>(),
                (937, 1, 128),
                &cpu,
            )?;
            let want = k
                .broadcast_mul(&q)?
                .sum(2)?
                .relu()?
                .sum(1)?
                .to_vec1::<f32>()?;
            let got = k
                .to_device(&gpu)?
                .broadcast_mul(&q.to_device(&gpu)?)?
                .sum(2)?
                .relu()?
                .sum(1)?
                .to_device(&cpu)?
                .to_vec1::<f32>()?;
            let nonfinite = got.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(nonfinite, 0, "score arm: {nonfinite} non-finite on Metal");
            let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 1e-4 * scale,
                "score arm: Metal differs from CPU by {md} at scale {scale}"
            );
        }
    }
    Ok(())
}
