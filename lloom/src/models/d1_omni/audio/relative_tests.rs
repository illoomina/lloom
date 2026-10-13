use super::relative::*;
use lloom_tensor::{Device, Result, Tensor};

#[test]
fn relative_shift_window_matches_literal_pad_view_mapping() {
    for t in 1usize..=7 {
        let p = 2 * t - 1;
        let original = (0..t * p).collect::<Vec<_>>();
        let padded = (0..t)
            .flat_map(|q| {
                std::iter::once(usize::MAX).chain(original[q * p..(q + 1) * p].iter().copied())
            })
            .collect::<Vec<_>>();
        // Python: pad last axis(1,0), view[P+1,T], drop first row,
        // then view[T,P]. These operations shift the flat origin by T.
        let shifted = &padded[t..];
        for q in 0..t {
            for k in 0..t {
                assert_eq!(shifted[q * p + k], original[q * p + (t - 1 - q + k)]);
            }
        }
    }
}

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed) % 17) as f32 - 8.) / 8.)
        .collect()
}
fn scalar(
    qc: &[f32],
    qp: &[f32],
    k: &[f32],
    v: &[f32],
    p: &[f32],
    valid: usize,
    wrong_position: bool,
    omit_post_mask: bool,
) -> Vec<f64> {
    let (heads, t, d) = (2, 3, 2);
    let mut out = vec![0.; heads * t * d];
    for h in 0..heads {
        for q in 0..t {
            let mut scores = Vec::new();
            for j in 0..t {
                let pos = if wrong_position { t - 1 } else { t - 1 - q + j };
                let mut ac = 0.;
                let mut bd = 0.;
                for c in 0..d {
                    ac += qc[(h * t + q) * d + c] as f64 * k[(h * t + j) * d + c] as f64;
                    bd +=
                        qp[(h * t + q) * d + c] as f64 * p[(h * (2 * t - 1) + pos) * d + c] as f64;
                }
                scores.push(if q < valid && j < valid {
                    (ac + bd) / (d as f64).sqrt()
                } else {
                    -10000.
                });
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom = scores.iter().map(|x| (x - max).exp()).sum::<f64>();
            for j in 0..t {
                let a = if !omit_post_mask && (q >= valid || j >= valid) {
                    0.
                } else {
                    (scores[j] - max).exp() / denom
                };
                for c in 0..d {
                    out[(h * t + q) * d + c] += a * v[(h * t + j) * d + c] as f64;
                }
            }
        }
    }
    out
}

#[test]
fn relative_rows_match_scalar_and_mask_all_padding_without_nan() -> Result<()> {
    let qc = values(12, 1);
    let qp = values(12, 2);
    let k = values(12, 3);
    let v = values(12, 4);
    let p = values(20, 5);
    let tensor = |x: &[f32], n| Tensor::from_vec(x.to_vec(), (1, 2, n, 2), &Device::Cpu);
    let (qct, qpt, kt, vt, pt) = (
        tensor(&qc, 3)?,
        tensor(&qp, 3)?,
        tensor(&k, 3)?,
        tensor(&v, 3)?,
        tensor(&p, 5)?,
    );
    for valid in [0, 1, 2, 3] {
        let plan = RelativeAttentionPlan::new(RelativeInputs {
            content_query: &qct,
            position_query: &qpt,
            key: &kt,
            value: &vt,
            positions: &pt,
            valid,
        })?;
        // K48 + P80 + masks6 + constants48 + query/output rows144
        // + score rows432 + BD120 + final cat48 = 926 bytes.
        assert_eq!(plan.scratch().tensor_allocation_volume, 926);
        assert_eq!(plan.scratch().host_mask_volume, 3);
        let expected = scalar(&qc, &qp, &k, &v, &p, valid, false, false);
        assert_eq!(expected.len(), 12);
        let mut bits = None;
        for _ in 0..3 {
            let got = plan.execute()?;
            assert_eq!(got.dims(), [1, 2, 3, 2]);
            let got = got.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(got.len(), 12);
            for (i, (&a, &b)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (a as f64 - b).abs() < 2e-6,
                    "relative valid{valid} index{i}: {a} != {b}"
                );
            }
            let next = got.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            if let Some(ref prior) = bits {
                assert_eq!(&next, prior);
            }
            bits = Some(next);
        }
        if valid == 2 {
            assert!(
                expected
                    .iter()
                    .zip(scalar(&qc, &qp, &k, &v, &p, valid, true, false))
                    .any(|(a, b)| (a - b).abs() > 1e-3),
                "relative-position discriminator vacuous"
            );
            assert!(
                expected
                    .iter()
                    .zip(scalar(&qc, &qp, &k, &v, &p, valid, false, true))
                    .any(|(a, b)| (a - b).abs() > 1e-3),
                "post-softmax mask discriminator vacuous"
            );
        }
        if valid == 0 {
            assert!(expected.iter().all(|&x| x == 0.));
        }
    }
    assert!(RelativeAttentionPlan::new(RelativeInputs {
        content_query: &qct,
        position_query: &qpt,
        key: &kt,
        value: &vt,
        positions: &pt,
        valid: 4
    })
    .is_err());
    let wrong = pt.narrow(2, 0, 3)?;
    assert!(RelativeAttentionPlan::new(RelativeInputs {
        content_query: &qct,
        position_query: &qpt,
        key: &kt,
        value: &vt,
        positions: &wrong,
        valid: 2
    })
    .is_err());
    Ok(())
}
