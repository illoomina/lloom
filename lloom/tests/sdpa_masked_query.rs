//! A single query's additive mask is part of SDPA, not a dispatch hint.
//! CPU scalar oracle has no tensor operations or device dependency.

const BATCH: usize = 2;
const HEADS: usize = 2;
const DIM: usize = 64;

struct Fixture {
    keys: usize,
    queries: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mask: Vec<f32>,
}

impl Fixture {
    fn new(queries: usize, keys: usize) -> Self {
        // Dyadic values are exactly representable in both tested input dtypes.
        let q = (0..BATCH * HEADS * queries * DIM)
            .map(|i| ((i % 11) as f32 - 5.) / 16.)
            .collect();
        let k = (0..BATCH * keys * DIM)
            .map(|i| ((i % 13) as f32 - 6.) / 16.)
            .collect();
        let v = (0..BATCH * keys * DIM)
            .map(|i| ((i % 31) as f32 - 15.) / 8.)
            .collect();
        let mask = (0..BATCH * HEADS * queries * keys)
            .map(|i| {
                let key = i % keys;
                let row = i / keys;
                if key == row % keys {
                    0.5 // Every row has at least one admitted key.
                } else if key == (row + 1) % keys {
                    -0.75 // Finite additive bias, not merely a boolean mask.
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        Self {
            keys,
            queries,
            q,
            k,
            v,
            mask,
        }
    }

    fn reference(&self, use_mask: bool) -> Vec<f32> {
        let mut out = vec![0f32; BATCH * HEADS * self.queries * DIM];
        for b in 0..BATCH {
            for h in 0..HEADS {
                for query in 0..self.queries {
                    let row = (b * HEADS + h) * self.queries + query;
                    let mut scores = vec![0f64; self.keys];
                    for (key, score) in scores.iter_mut().enumerate() {
                        let kv = (b * self.keys + key) * DIM;
                        *score = (0..DIM)
                            .map(|d| f64::from(self.q[row * DIM + d]) * f64::from(self.k[kv + d]))
                            .sum::<f64>()
                            * 0.125;
                        if use_mask {
                            *score += f64::from(self.mask[row * self.keys + key]);
                        }
                    }
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<_> = scores.iter().map(|score| (score - max).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    for d in 0..DIM {
                        out[row * DIM + d] = (weights
                            .iter()
                            .enumerate()
                            .map(|(key, weight)| {
                                weight * f64::from(self.v[(b * self.keys + key) * DIM + d])
                            })
                            .sum::<f64>()
                            / sum) as f32;
                    }
                }
            }
        }
        out
    }
}

#[test]
fn scalar_fixture_discriminates_ignored_masks() {
    let single = Fixture::new(1, 1);
    let got = single.reference(true);
    for b in 0..BATCH {
        for h in 0..HEADS {
            assert_eq!(
                &got[(b * HEADS + h) * DIM..(b * HEADS + h + 1) * DIM],
                &single.v[b * DIM..(b + 1) * DIM]
            );
        }
    }
    for keys in [3, 32, 33, 1025, 1537] {
        let f = Fixture::new(1, keys);
        let masked = f.reference(true);
        let unmasked = f.reference(false);
        let delta = masked
            .iter()
            .zip(&unmasked)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            delta > 0.02,
            "fixture keys={keys} does not distinguish ignored mask: {delta}"
        );
        assert!(masked.iter().all(|x| x.is_finite()));
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use lloom::nn::ops;
    use lloom_tensor::{DType, Device, Result, Tensor};

    fn deterministic_contract() {
        use lloom_tensor::tuning::{self, Contract};
        assert_eq!(
            tuning::pin_regime(Contract::Deterministic).unwrap(),
            Contract::Deterministic
        );
        assert_eq!(
            tuning::gate_contract("masked SDPA parity").unwrap(),
            Contract::Deterministic
        );
    }

    fn inputs(f: &Fixture, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        Ok((
            Tensor::from_vec(f.q.clone(), (BATCH, HEADS, f.queries, DIM), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(f.k.clone(), (BATCH, 1, f.keys, DIM), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(f.v.clone(), (BATCH, 1, f.keys, DIM), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(f.mask.clone(), (BATCH, HEADS, f.queries, f.keys), dev)?
                .to_dtype(dtype)?,
        ))
    }

    fn values(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }

    fn assert_reference(got: &[f32], expected: &[f32], dtype: DType, keys: usize) {
        let tolerance = if dtype == DType::F16 { 0.003 } else { 0.00003 };
        assert_eq!(got.len(), expected.len());
        for (i, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            assert!(got.is_finite() && (got - expected).abs() <= tolerance,
                    "{dtype:?} keys={keys} element={i}: got={got}, expected={expected}, tolerance={tolerance}");
        }
    }

    #[test]
    fn masked_single_query_matches_scalar_and_repeats() -> Result<()> {
        deterministic_contract();
        let dev = Device::new_metal(0)?;
        for dtype in [DType::F16, DType::F32] {
            for keys in [1, 3, 32, 33, 1025, 1537] {
                let f = Fixture::new(1, keys);
                let expected = f.reference(true);
                let (q, k, v, mask) = inputs(&f, dtype, &dev)?;
                let mut first = None;
                for _ in 0..3 {
                    let got = values(&ops::sdpa(&q, &k, &v, Some(&mask), false, 0.125, 1.)?)?;
                    assert_reference(&got, &expected, dtype, keys);
                    let bits: Vec<_> = got.iter().map(|v| v.to_bits()).collect();
                    if let Some(first) = &first {
                        assert_eq!(&bits, first, "nondeterministic {dtype:?} keys={keys}");
                    } else {
                        first = Some(bits);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn masked_softcap_refuses_without_disabling_unmasked_vector_softcap() -> Result<()> {
        deterministic_contract();
        let dev = Device::new_metal(0)?;
        let f = Fixture::new(1, 3);
        let (q, k, v, mask) = inputs(&f, DType::F32, &dev)?;
        let error = ops::sdpa(&q, &k, &v, Some(&mask), false, 0.125, 2.).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("SDPA full requires softcapping to be disabled"),
            "{error}"
        );
        let got = values(&ops::sdpa(&q, &k, &v, None, false, 0.125, 2.)?)?;
        assert!(got.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
