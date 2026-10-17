//! Finite CPU controls through the public scalar/window/processor consumers.
//! Source-only increment: these tests have not been compiled or run.
use lloom::generation::{
    Blocking, LogitsProcessor, Processors, ReadbackStrategy, Sampling, ThinkingBudget, GREEDY_BELOW,
};
use lloom_tensor::{Device, Result, Tensor};

#[path = "../../lloom-tensor/tests/sampling_contract/fixtures.rs"]
mod fixtures;
#[path = "../../lloom-tensor/tests/sampling_contract/oracle.rs"]
mod oracle;

fn sampler(seed: u64, params: oracle::Params) -> LogitsProcessor {
    LogitsProcessor::from_sampling(
        seed,
        Sampling::TopKThenTopP {
            temperature: params.temperature,
            k: params.k,
            p: params.p,
        },
    )
}

#[test]
fn public_scalar_contract_meets_independent_fixtures_at_position_uniforms() -> Result<()> {
    let seed = 1036;
    for case in fixtures::small() {
        let row = Tensor::new(case.z.as_slice(), &Device::Cpu)?;
        for pos in 0..12 {
            let mut s = sampler(seed, case.params);
            s.at_position(Some(pos));
            let want = oracle::draw(&case.z, case.params, LogitsProcessor::uniform_at(seed, pos));
            match (want, s.sample(&row)) {
                (Ok(want), Ok(got)) => assert_eq!(got, want, "{} pos={pos}", case.name),
                (Err(_), Err(_)) => {}
                (want, got) => panic!("{} pos={pos}: want {want:?}, got {got:?}", case.name),
            }
        }
    }
    Ok(())
}

#[test]
fn all_and_disabled_filters_share_the_public_scalar_and_window_contract() -> Result<()> {
    let z = [-87.0f32, -1.0, 0.0, f32::NEG_INFINITY, 0.0, -0.25];
    let row = Tensor::new(&z, &Device::Cpu)?;
    let rows = Tensor::new(&[z, z, z, z], &Device::Cpu)?;
    let params = oracle::Params::new(0.7, 0, 1.0);
    let arms = [
        Sampling::All { temperature: 0.7 },
        Sampling::TopK {
            k: 0,
            temperature: 0.7,
        },
        Sampling::TopK {
            k: z.len(),
            temperature: 0.7,
        },
        Sampling::TopK {
            k: usize::MAX,
            temperature: 0.7,
        },
        Sampling::TopP {
            p: 1.0,
            temperature: 0.7,
        },
    ];
    for arm in arms {
        let mut s = LogitsProcessor::from_sampling(7, arm.clone());
        for pos in 0..24 {
            s.at_position(Some(pos));
            let want = oracle::draw(&z, params, LogitsProcessor::uniform_at(7, pos)).unwrap();
            assert_eq!(s.sample(&row)?, want, "{arm:?} pos={pos}");
        }
        let got = s.window_draw(&rows, 20)?;
        let want: Vec<_> = (20..24)
            .map(|pos| oracle::draw(&z, params, LogitsProcessor::uniform_at(7, pos)).unwrap())
            .collect();
        assert_eq!(got, want, "{arm:?}");
    }
    Ok(())
}

#[test]
fn conditional_nucleus_ties_signed_zero_and_p_zero_have_literal_public_pins() -> Result<()> {
    for (z, k, p, want) in [
        (vec![-0.0, 0.0, 0.0], 1, 1.0, 0),
        (vec![0.0, 2.0, 2.0, f32::NEG_INFINITY], 0, 0.0, 1),
        (
            vec![
                (0.4f64).ln() as f32,
                (0.35f64).ln() as f32,
                (0.25f64).ln() as f32,
            ],
            2,
            0.5,
            0,
        ),
    ] {
        let row = Tensor::new(z.as_slice(), &Device::Cpu)?;
        let mut s = sampler(7, oracle::Params::new(1.0, k, p));
        for pos in 0..24 {
            s.at_position(Some(pos));
            assert_eq!(s.sample(&row)?, want);
        }
    }
    for p in [
        0.5,
        f32::from_bits(0.5f32.to_bits() - 1) as f64,
        f32::from_bits(0.5f32.to_bits() + 1) as f64,
    ] {
        let z = [0f32; 4];
        let row = Tensor::new(&z, &Device::Cpu)?;
        let mut s = sampler(19, oracle::Params::new(1.0, 0, p));
        for pos in 0..32 {
            s.at_position(Some(pos));
            assert_eq!(
                s.sample(&row)?,
                oracle::draw(
                    &z,
                    oracle::Params::new(1.0, 0, p),
                    LogitsProcessor::uniform_at(19, pos)
                )
                .unwrap()
            );
        }
    }
    Ok(())
}

#[test]
fn rejected_rows_do_not_feed_record_digest_or_consume_running_uniforms() -> Result<()> {
    let processors = Processors {
        frequency_penalty: 1.0,
        ..Processors::default()
    };
    for arm in [
        Sampling::All { temperature: 1.0 },
        Sampling::FilteredGumbel {
            temperature: 1.0,
            k: 2,
            p: 0.9,
        },
    ] {
        let make = || {
            LogitsProcessor::from_sampling(8, arm.clone())
                .with_processors(processors.clone())
                .with_logits_digest(true)
        };
        let mut a = make();
        let mut b = make();
        for bad in [
            vec![f32::NAN, 0.0],
            vec![0.0, f32::INFINITY],
            vec![f32::NEG_INFINITY; 2],
        ] {
            let before = a.logits_digest();
            let mut feeds = 0;
            assert!(a
                .sample_fed(&Tensor::new(bad.as_slice(), &Device::Cpu)?, |id| {
                    feeds += 1;
                    id
                })
                .is_err());
            assert_eq!(feeds, 0);
            assert_eq!(a.digest_rows(), 0);
            assert_eq!(a.logits_digest(), before);
        }
        let row = Tensor::new(&[0f32, 0.25], &Device::Cpu)?;
        for _ in 0..12 {
            assert_eq!(
                a.sample(&row)?,
                b.sample(&row)?,
                "rejected row moved RNG/history for {arm:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn filtered_gumbel_preserves_nonuniform_support_and_original_position_noise() -> Result<()> {
    let z = [1.0f32, -20.0, 0.5, f32::NEG_INFINITY, 0.0];
    let row = Tensor::new(&z, &Device::Cpu)?;
    let rows = Tensor::new(&[z, z, z, z], &Device::Cpu)?;
    let arm = Sampling::resolve(Some(0.7), Some(2), Some(0.9), true)?;
    assert!(matches!(arm, Sampling::FilteredGumbel { .. }));
    assert!(arm.requires_host());
    let mut s = LogitsProcessor::from_sampling(7, arm.clone());
    let mut seen = Vec::new();
    for first in [3, 20, 99] {
        let mut w = LogitsProcessor::from_sampling(7, arm.clone());
        let _ = w.window_draw(&rows, 500)?; // discarded work cannot move position keys
        let drawn = w.window_draw(&rows, first)?;
        for (i, &id) in drawn.iter().enumerate() {
            s.at_position(Some(first + i as u64));
            assert_eq!(id, s.sample(&row)?);
            assert!([0, 2].contains(&id));
            seen.push(id);
        }
        assert_eq!(drawn, w.window_draw(&rows, first)?);
    }
    // Independent noise expectation: reuse neither production fill_gumbel nor
    // production stream_key. The key is the published mapping, transcribed here.
    use rand::{Rng, SeedableRng};
    for pos in 0..32u64 {
        let mut key = 7u64.wrapping_add(pos.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        key = (key ^ (key >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        key = (key ^ (key >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        key ^= key >> 31;
        let mut rng = rand::rngs::StdRng::seed_from_u64(key);
        let mut best = (0, f64::NEG_INFINITY);
        for (i, &score) in z.iter().enumerate() {
            let u: f64 = rng.random_range(1e-9..0.9999999);
            let g = (-(-u.ln()).ln()) as f32;
            if [0, 2].contains(&i) && score as f64 / 0.7 + g as f64 > best.1 {
                best = (i as u32, score as f64 / 0.7 + g as f64);
            }
        }
        s.at_position(Some(pos));
        assert_eq!(s.sample(&row)?, best.0);
        seen.push(best.0);
    }
    assert!(
        seen.contains(&0) && seen.contains(&2),
        "fixture must exercise both allowed ids"
    );
    Ok(())
}

#[test]
fn public_window_commits_only_accepted_history_and_forces_close_via_existing_consumer() -> Result<()>
{
    let dev = Device::Cpu;
    let processors = Processors {
        frequency_penalty: 2.0,
        ..Processors::default()
    };
    let arm = Sampling::FilteredGumbel {
        temperature: 1.0,
        k: 1,
        p: 1.0,
    };
    let mut window = LogitsProcessor::from_sampling(7, arm.clone())
        .with_processors(processors.clone())
        .with_logits_digest(true);
    let row = Tensor::new(&[3f32, 2.0, 0.0], &dev)?;
    let rows = Tensor::new(&[[3f32, 2.0, 0.0]; 3], &dev)?;
    let inputs = Tensor::new(&[0u32, 0, 0], &dev)?;
    let got = window
        .window_draw_device(&rows, 10, Some(&inputs))?
        .to_vec1::<u32>()?;
    assert_eq!(got, [0, 1, 1]); // row i counts inputs[1..=i], not previous hypothetical draws
    assert_eq!(window.digest_rows(), 0);
    assert_eq!(
        window
            .window_draw_device(&rows, 10, Some(&inputs))?
            .to_vec1::<u32>()?,
        got
    );
    window.commit_window(&[0])?;
    assert_eq!(window.digest_rows(), 1);
    window.at_position(Some(11));
    assert_eq!(window.sample(&row)?, 1); // exactly one committed 0; rejected rows did not count

    let budget = ThinkingBudget::new(0, vec![2], vec![2])?;
    let mut blocking = Blocking::new(&[99], &dev)?;
    blocking.set_thinking_budget(Some(&budget))?;
    let mut scalar = LogitsProcessor::from_sampling(7, arm).with_processors(processors);
    let fed = blocking
        .sample_and_feed(&mut scalar, &row)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    assert_eq!(fed, [2]);
    // Counting the drawn id0 instead would lower it below id1 on this row.
    assert_eq!(scalar.sample(&row)?, 0);
    Ok(())
}

#[test]
fn legacy_constructor_retains_pre_release_threshold_and_old_valid_defaults() -> Result<()> {
    let row = Tensor::new(&[0f32, 0.0, -1.0], &Device::Cpu)?;
    for t in [0.0, GREEDY_BELOW / 2.0] {
        let mut s = LogitsProcessor::new(7, Some(t), Some(0.9));
        assert_eq!(s.sampling(), &Sampling::ArgMax);
        assert_eq!(s.sample(&row)?, 0);
    }
    assert_eq!(
        Sampling::resolve(Some(GREEDY_BELOW), None, None, false)?,
        Sampling::All {
            temperature: GREEDY_BELOW
        }
    );
    assert_eq!(
        Sampling::resolve(None, Some(20), Some(0.9), true)?,
        Sampling::ArgMax
    );
    Ok(())
}

#[test]
fn window_budget_and_digest_advance_only_by_the_accepted_prefix() -> Result<()> {
    let budget = ThinkingBudget::new(0, vec![2], vec![1, 2])?;
    let mut s = LogitsProcessor::from_sampling(
        7,
        Sampling::FilteredGumbel {
            temperature: 1.0,
            k: 1,
            p: 1.0,
        },
    )
    .with_window_budget(Some(&budget))
    .with_logits_digest(true);
    let rows = Tensor::new(&[[3f32, 2.0, 0.0]; 3], &Device::Cpu)?;
    let drawn = s.window_draw(&rows, 10)?;
    assert_eq!(drawn, [0, 0, 0]);
    assert_eq!(s.window_fed(&drawn), [1, 2, 0]);
    assert_eq!(s.window_fed(&drawn), [1, 2, 0]); // a provisional window consumes no budget
    assert_eq!(s.digest_rows(), 0);
    s.commit_window(&[1])?;
    assert_eq!(s.digest_rows(), 1);
    assert_eq!(s.window_fed(&drawn), [2, 0, 0]);
    let _ = s.window_draw(&rows, 11)?;
    s.commit_window(&[2])?;
    assert_eq!(s.digest_rows(), 2);
    assert_eq!(s.window_fed(&drawn), [0, 0, 0]);
    Ok(())
}

#[test]
fn gumbel_p_zero_and_removed_negative_candidates_cannot_win_with_zero_noise() -> Result<()> {
    let dev = Device::Cpu;
    let row = Tensor::new(&[-101f32, -100.0, -100.1, f32::NEG_INFINITY], &dev)?;
    for arm in [
        Sampling::resolve(Some(10.0), Some(1), Some(1.0), true)?,
        Sampling::resolve(Some(10.0), None, Some(0.0), true)?,
    ] {
        let mut s = LogitsProcessor::from_sampling(7, arm);
        for pos in 0..64 {
            s.at_position(Some(pos));
            assert_eq!(
                s.sample(&row)?,
                1,
                "an excluded candidate participated at pos={pos}"
            );
        }
    }
    Ok(())
}

#[test]
fn arbitrary_unnormalized_nonzero_callback_weights_survive_the_host_route() -> Result<()> {
    let row = Tensor::new(&[1f32, 0.0], &Device::Cpu)?;
    let pos = (0..128)
        .find(|&p| (0.2..0.4).contains(&LogitsProcessor::uniform_at(7, p)))
        .unwrap();
    for arm in [
        Sampling::All { temperature: 1.0 },
        Sampling::TopK {
            temperature: 1.0,
            k: 1,
        },
        Sampling::TopP {
            temperature: 1.0,
            p: 0.8,
        },
        Sampling::TopKThenTopP {
            temperature: 1.0,
            k: 1,
            p: 0.8,
        },
    ] {
        let mut s = LogitsProcessor::from_sampling(7, arm);
        s.at_position(Some(pos));
        let mut calls = 0;
        let got = s.sample_f(&row, |p| {
            calls += 1;
            p.copy_from_slice(&[1.0, 9.0]);
        })?;
        assert_eq!(calls, 1);
        assert_eq!(got, 1); // both values nonzero; normalizing changes no ratio
    }
    Ok(())
}

#[test]
fn a_nonfinite_processor_result_is_rejected_before_feed_or_digest_commit() -> Result<()> {
    let mut s = LogitsProcessor::from_sampling(
        7,
        Sampling::TopK {
            temperature: 1.0,
            k: 1,
        },
    )
    .with_processors(Processors {
        logit_bias: vec![(1, f32::NAN)],
        ..Processors::default()
    })
    .with_logits_digest(true);
    let row = Tensor::new(&[1f32, 0.0], &Device::Cpu)?;
    let mut fed = false;
    assert!(s
        .sample_fed(&row, |id| {
            fed = true;
            id
        })
        .is_err());
    assert!(!fed);
    assert_eq!(s.digest_rows(), 0);
    Ok(())
}


// Root additive regressions: the same Gumbel draw is invariant to adding an
// equal finite offset to every supported score. No previous assertion changed.
fn root_filtered_gumbel_sampler() -> LogitsProcessor {
    LogitsProcessor::from_sampling(7, Sampling::FilteredGumbel {
        temperature: 1.0, k: 2, p: 1.0,
    })
}

#[test]
fn filtered_gumbel_scalar_keeps_noise_under_large_equal_finite_offsets() -> Result<()> {
    let dev = Device::Cpu;
    let zero = Tensor::new(&[0.0f32, 0.0, f32::NEG_INFINITY], &dev)?;
    let mut reference = root_filtered_gumbel_sampler();
    let mut expected = Vec::new();
    for pos in 0..32 {
        reference.at_position(Some(pos));
        expected.push(reference.sample(&zero)?);
    }
    assert!(expected.contains(&0) && expected.contains(&1),
        "zero-offset control must observe both equal-score candidates");
    for offset in [f32::MAX, -f32::MAX] {
        let row = Tensor::new(&[offset, offset, f32::NEG_INFINITY], &dev)?;
        let mut shifted = root_filtered_gumbel_sampler();
        for (pos, &want) in expected.iter().enumerate() {
            shifted.at_position(Some(pos as u64));
            assert_eq!(shifted.sample(&row)?, want,
                "common finite offset must not erase Gumbel noise: offset={offset}, pos={pos}");
        }
    }
    Ok(())
}

#[test]
fn filtered_gumbel_window_keeps_noise_under_large_equal_finite_offsets() -> Result<()> {
    let dev = Device::Cpu;
    let zero = [0.0f32, 0.0, f32::NEG_INFINITY];
    let zero_rows = Tensor::new(&[zero; 32], &dev)?;
    let mut reference = root_filtered_gumbel_sampler();
    let expected = reference.window_draw(&zero_rows, 0)?;
    assert!(expected.contains(&0) && expected.contains(&1),
        "zero-offset window control must observe both equal-score candidates");
    for offset in [f32::MAX, -f32::MAX] {
        let row = [offset, offset, f32::NEG_INFINITY];
        let rows = Tensor::new(&[row; 32], &dev)?;
        let mut shifted = root_filtered_gumbel_sampler();
        assert_eq!(shifted.window_draw(&rows, 0)?, expected,
            "window common finite offset must not erase Gumbel noise: offset={offset}");
    }
    Ok(())
}

#[test]
fn unfiltered_and_disabled_filter_gumbel_share_centered_f32_scalar_window_and_running_stream() -> Result<()> {
    let device = Device::Cpu;
    let rows = Tensor::new(&[[1e30f32, 1e30, f32::NEG_INFINITY]; 4], &device)?;
    let row = rows.get(0)?;
    let mut plain = LogitsProcessor::from_sampling(7, Sampling::GumbelSoftmax { temperature: 3.0 });
    let mut filtered = LogitsProcessor::from_sampling(7, Sampling::FilteredGumbel { temperature: 3.0, k: 3, p: 1.0 });
    let mut seen = Vec::new();
    for pos in 0..32u64 {
        plain.at_position(Some(pos));
        filtered.at_position(Some(pos));
        let id = plain.sample(&row)?;
        assert_eq!(id, filtered.sample(&row)?);
        assert_eq!(id, plain.sample_argmax_on_device(&row)?.to_scalar::<u32>()?);
        seen.push(id);
    }
    assert!(seen.contains(&0) && seen.contains(&1));
    assert_eq!(plain.window_draw(&rows, 10)?, filtered.window_draw(&rows, 10)?);
    plain.at_position(None);
    filtered.at_position(None);
    for _ in 0..16 { assert_eq!(plain.sample(&row)?, filtered.sample(&row)?); }
    Ok(())
}

#[test]
fn unfiltered_gumbel_device_return_preserves_multirow_shape_and_flattened_noise_stream() -> Result<()> {
    let device = Device::Cpu;
    let rows = Tensor::new(&[[1e30f32; 3]; 2], &device)?;
    let row = rows.get(0)?;
    let arm = Sampling::GumbelSoftmax { temperature: 3.0 };
    let mut matrix = LogitsProcessor::from_sampling(7, arm.clone());
    let mut host = LogitsProcessor::from_sampling(7, arm.clone());
    let mut scalar = LogitsProcessor::from_sampling(7, arm);
    for _ in 0..8 {
        let ids = matrix.sample_argmax_on_device(&rows)?;
        assert_eq!(ids.dims(), &[2]);
        let ids = ids.to_vec1::<u32>()?;
        assert_eq!(ids, [scalar.sample(&row)?, scalar.sample(&row)?]);
        assert_eq!(host.sample(&rows)?, ids[0]);
    }
    // A keyed matrix uses one key for all six variates, not a reset per row.
    // Calibrate against the unchanged original noise mapping independently.
    use rand::{Rng, SeedableRng};
    let pos = 11u64;
    let mut key = 7u64.wrapping_add(pos.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    key = (key ^ (key >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    key ^= key >> 31;
    let mut rng = rand::rngs::StdRng::seed_from_u64(key);
    let expected: Vec<u32> = (0..2).map(|_| {
        let mut best = (0u32, f32::NEG_INFINITY);
        for i in 0..3 {
            let u: f64 = rng.random_range(1e-9..0.9999999);
            let g = (-(-u.ln()).ln()) as f32;
            if g > best.1 { best = (i, g); }
        }
        best.0
    }).collect();
    matrix.at_position(Some(pos));
    assert_eq!(matrix.sample_argmax_on_device(&rows)?.to_vec1::<u32>()?, expected);
    Ok(())
}
