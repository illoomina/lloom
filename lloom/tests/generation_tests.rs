use lloom::generation::{LogitsProcessor, Sampling};
use lloom_tensor::{DType, Device, Result, Tensor};

/// The CPU, and the Metal device when the feature is on.
fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    v.push(Device::new_metal(0).expect("the `metal` feature is on and no Metal device opened"));
    v
}

/// The `ArgMax` arm returns the same index over f16 as over f32 (lloom #351).
///
/// `sample_f` used to `to_dtype(F32)` before matching on `Sampling`, so the
/// `ArgMax` arm paid a whole-vocabulary cast whose values it never reads. The
/// arm now runs over the uncast logits. Widening f16 -> f32 is monotone and
/// injective, so the index cannot move -- this asserts that rather than
/// assuming it, at a dtype where the two paths are genuinely different code.
///
/// **Absolute reference, not a shape** (`CONTRIBUTING.md` §3.1b): the expected
/// index is written as a literal, so this fails with the index it got rather
/// than with "the two arms disagreed".
#[test]
fn the_f16_argmax_index_matches_the_f32_path() -> Result<()> {
    let dev = Device::Cpu;
    // Deliberately not sorted, and the max is not at either end.
    let vals = [0.1f32, 0.9, 0.3, 0.85, 0.2];
    let f32_logits = Tensor::new(&vals, &dev)?;
    let f16_logits = f32_logits.to_dtype(DType::F16)?;

    let mut p = LogitsProcessor::new(1337, None, None);
    let from_f16 = p.sample(&f16_logits)?;
    let mut p = LogitsProcessor::new(1337, None, None);
    let from_f32 = p.sample(&f32_logits)?;

    assert_eq!(from_f16, 1, "f16 argmax picked the wrong index");
    assert_eq!(from_f32, 1, "f32 argmax picked the wrong index");
    assert_eq!(from_f16, from_f32);
    Ok(())
}

/// Ties break to the LOWEST index, on both dtypes.
///
/// This is the one place the two paths could legitimately disagree, so it is
/// pinned with a literal rather than by comparing the arms to each other: two
/// paths that both broke ties to the *highest* index would agree with each
/// other and still be a behaviour change.
#[test]
fn an_argmax_tie_breaks_to_the_lowest_index_at_both_dtypes() -> Result<()> {
    let dev = Device::Cpu;
    // Indices 1 and 3 tie at the maximum; 0.5 is exact in both f16 and f32, so
    // the tie survives the narrowing and this fixture tests what it claims to.
    let vals = [0.25f32, 0.5, 0.125, 0.5, 0.25];
    let f32_logits = Tensor::new(&vals, &dev)?;
    let f16_logits = f32_logits.to_dtype(DType::F16)?;
    let roundtripped = f16_logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(
        (roundtripped[1], roundtripped[3]),
        (0.5, 0.5),
        "the fixture's tie did not survive the f16 narrowing"
    );

    let mut p = LogitsProcessor::new(1337, None, None);
    assert_eq!(p.sample(&f16_logits)?, 1);
    let mut p = LogitsProcessor::new(1337, None, None);
    assert_eq!(p.sample(&f32_logits)?, 1);
    Ok(())
}

/// A host-sort arm still sees f32, which is what bounds #351's claim.
///
/// Only `ArgMax` skips the cast. `sample_f`'s closure receives the f32
/// probabilities a host-sort arm (top-k, top-p) draws from, and this asserts
/// the slice it is handed is the full vocabulary at f32 -- so a future change
/// extending the skip to such an arm fails here rather than silently
/// degrading the sampler. (Plain temperature is a device draw on every arm
/// and hands the host no probabilities.)
#[test]
fn a_temperature_arm_still_receives_f32_probabilities() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &dev)?.to_dtype(DType::F16)?;
    let mut p = LogitsProcessor::new(42, Some(0.9), Some(0.95));
    let mut seen = 0usize;
    let _ = p.sample_f(&logits, |prs| {
        seen = prs.len();
        // A softmax over four logits sums to 1.0 -- if the cast were skipped
        // here the values would not be probabilities at all.
        let sum: f32 = prs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "not a distribution: {sum}");
    })?;
    assert_eq!(seen, 4, "the closure did not see the whole vocabulary");
    Ok(())
}

/// `sample_f`'s hook reaches plain temperature's draw, as it did before the
/// draw moved to the device (#943): a hook that leaves one token's mass draws
/// that token at every position, and a hook that changes nothing draws what
/// `sample` draws at the same seed and positions (a four-token vocabulary,
/// where the host and device sums agree).
#[test]
fn a_hook_over_the_probabilities_reaches_the_temperature_draw() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &dev)?;
    let mut hooked = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1.0 });
    for pos in 0..32u64 {
        hooked.at_position(Some(pos));
        let id = hooked.sample_f(&logits, |prs| {
            for (i, p) in prs.iter_mut().enumerate() {
                if i != 1 {
                    *p = 0.0;
                }
            }
        })?;
        assert_eq!(id, 1, "position {pos}: the hook's mask was not drawn from");
    }
    let mut plain = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1.0 });
    let mut noop = LogitsProcessor::from_sampling(7, Sampling::All { temperature: 1.0 });
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for pos in 0..64u64 {
        plain.at_position(Some(pos));
        noop.at_position(Some(pos));
        a.push(plain.sample(&logits)?);
        b.push(noop.sample_f(&logits, |_| {})?);
    }
    assert_eq!(a, b, "a hook that changes nothing moved the stream");
    assert!(
        a.iter().any(|&t| t != a[0]),
        "too few distinct draws to show anything"
    );
    Ok(())
}

#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        lloom::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        lloom::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    // k = 2 draws from {2, 3} only, and from both: the support, not one
    // seed's first two tokens (the draw is the filter, then the inverse CDF).
    let drawn: Vec<u32> = (0..64)
        .map(|_| logits_process.sample(&logits))
        .collect::<Result<_>>()?;
    assert!(
        drawn.iter().all(|t| *t == 2 || *t == 3),
        "top-k 2 drew outside the top two: {drawn:?}"
    );
    assert!(
        drawn.contains(&2) && drawn.contains(&3),
        "top-k 2 drew one token only: {drawn:?}"
    );
    Ok(())
}

#[test]
fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    let logits = Tensor::new(&[-1.0, 0.0, 0.2, 1.0], &Device::Cpu)?;
    let sm = lloom::nn::ops::softmax(&logits, 0)?.to_vec1::<f64>()?;
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}

/// The host draw of the gumbel arm (`sample`, the blocking readback's) is
/// seeded by the processor's seed, as the device draw is: seed 7 twice draws
/// one sequence, seed 8 another. Without processors, so the property is the
/// sampler's own.
#[test]
fn a_seeded_gumbel_host_draw_reproduces() -> Result<()> {
    let logits = Tensor::new(
        &[
            0.1f32, 0.9, -0.3, 0.4, 0.0, 0.7, -1.0, 0.2, 0.5, 0.3, -0.2, 0.8, 0.6, -0.5, 0.05, 0.35,
        ],
        &Device::Cpu,
    )?;
    let draw = |seed: u64| -> Result<Vec<u32>> {
        let mut p = LogitsProcessor::from_sampling(
            seed,
            lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
        );
        (0..64).map(|_| p.sample(&logits)).collect()
    };
    let (a, b, c) = (draw(7)?, draw(7)?, draw(8)?);
    assert_eq!(a, b, "seed 7 drew two different sequences");
    assert_ne!(a, c, "seeds 7 and 8 drew the same 64 tokens");
    Ok(())
}

fn spread_logits() -> Result<Tensor> {
    Tensor::new(
        &[
            0.1f32, 0.9, -0.3, 0.4, 0.0, 0.7, -1.0, 0.2, 0.5, 0.3, -0.2, 0.8, 0.6, -0.5, 0.05, 0.35,
        ],
        &Device::Cpu,
    )
}

/// The default stream is keyed by DRAW COUNT: a draw's noise is a function of
/// how many draws preceded it. This is the incumbent contract — the one
/// `a_seeded_gumbel_host_draw_reproduces` above rests on — and it is what
/// makes a naive sampled verify non-reproducible, since a speculative round
/// draws rows for drafts it then rejects. Pinned so the position-keyed arm
/// below is a visible departure rather than a silent change of default.
#[test]
fn the_unkeyed_gumbel_draw_depends_on_how_many_draws_preceded_it() -> Result<()> {
    let logits = spread_logits()?;
    let draw_after = |skipped: usize| -> Result<u32> {
        let mut p = LogitsProcessor::from_sampling(
            42,
            lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
        );
        for _ in 0..skipped {
            let _ = p.sample(&logits)?;
        }
        p.sample(&logits)
    };
    let clean = draw_after(0)?;
    assert!(
        (1..=3).any(|skipped| draw_after(skipped).map(|t| t != clean).unwrap_or(false)),
        "the unkeyed stream gave the same token at a position however many \
         draws preceded it; if that is now true by construction the noise is \
         already position-keyed and this test has lost its subject"
    );
    Ok(())
}

/// Armed with a position, a draw is a function of `(seed, position)` ALONE:
/// the row at a position is the same row however many rows were drawn and
/// discarded before it. That is the property a sampled verify needs — a
/// rejected row must consume nothing the next position's noise depends on.
#[test]
fn a_position_keyed_gumbel_draw_ignores_the_draws_before_it() -> Result<()> {
    let logits = spread_logits()?;
    // The draw for absolute position 7, reached after `skipped` discarded
    // draws (the rows of a round whose drafts the verify rejected).
    let at_seven_after = |skipped: usize| -> Result<u32> {
        let mut p = LogitsProcessor::from_sampling(
            42,
            lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
        );
        for s in 0..skipped {
            p.at_position(Some(100 + s as u64));
            let _ = p.sample(&logits)?;
        }
        p.at_position(Some(7));
        p.sample(&logits)
    };
    let clean = at_seven_after(0)?;
    for skipped in 1..=5 {
        assert_eq!(
            clean,
            at_seven_after(skipped)?,
            "position 7's draw moved when {skipped} rows were drawn and \
             discarded before it; the noise is still keyed by draw count, so a \
             sampled speculative round cannot reproduce a sampled greedy one"
        );
    }
    // And the key is both halves: a different position or a different seed
    // draws a different stream, or the "keying" is a constant.
    let at = |seed: u64, pos: u64| -> Result<Vec<u32>> {
        let mut p = LogitsProcessor::from_sampling(
            seed,
            lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
        );
        p.at_position(Some(pos));
        // One position, drawn repeatedly: the stream is the position's, so
        // every draw of it is the same token.
        (0..4).map(|_| p.sample(&logits)).collect()
    };
    let p7 = at(42, 7)?;
    assert!(
        p7.iter().all(|t| *t == p7[0]),
        "one armed position drew {p7:?}: a position's stream must not advance \
         between draws of that same position"
    );
    // Both halves of the key matter — asserted over a STREAM of positions,
    // never over one. A single position is a 1-in-vocab coin flip: seeds 42
    // and 43 both draw token 7 at position 7 over these 16 logits, and they
    // agree at 30 of 512 positions, which is the 1/16 chance rate (0.0586
    // against 0.0625) rather than a shared stream. A one-position assertion
    // here failed for exactly that reason.
    let stream = |seed: u64, from: u64| -> Result<Vec<u32>> {
        (from..from + 32)
            .map(|pos| {
                let mut p = LogitsProcessor::from_sampling(
                    seed,
                    lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
                );
                p.at_position(Some(pos));
                p.sample(&logits)
            })
            .collect()
    };
    assert_eq!(
        stream(42, 0)?,
        stream(42, 0)?,
        "one seed drew two different streams"
    );
    assert_ne!(
        stream(42, 0)?,
        stream(43, 0)?,
        "seeds 42 and 43 drew identical 32-position streams: the seed is dropped"
    );
    assert_ne!(
        stream(42, 0)?,
        stream(42, 32)?,
        "positions 0..32 and 32..64 drew identical streams: the position is dropped"
    );
    // Disarming returns to the running stream.
    let mut p = LogitsProcessor::from_sampling(
        42,
        lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    p.at_position(Some(7));
    assert!(p.position_keyed());
    p.at_position(None);
    assert!(!p.position_keyed() && p.position().is_none());
    Ok(())
}

/// The position-keyed stream is a Gumbel: its mean is the Euler–Mascheroni
/// constant. A keying that avalanched the seed into a degenerate stream would
/// still reproduce and still decorrelate, and would sample from the wrong
/// distribution — so the draw's correctness is asserted against the
/// distribution's own constant, not against the mechanism's self-consistency
/// (`CONTRIBUTING.md` §3.1b rule 1: an absolute reference, not a shape).
#[test]
fn the_position_keyed_stream_is_gumbel_distributed() -> Result<()> {
    // `sample` over equal logits draws uniformly iff the noise is i.i.d.
    // Gumbel; a 16-way uniform over 16 000 position-keyed draws is the
    // argmax property gumbel-max needs (the one #345's χ² tested on the GPU).
    let logits = Tensor::new(&[0.0f32; 16], &Device::Cpu)?;
    let mut counts = vec![0usize; 16];
    let draws = 16_000usize;
    let mut p = LogitsProcessor::from_sampling(
        42,
        lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    for pos in 0..draws {
        p.at_position(Some(pos as u64));
        counts[p.sample(&logits)? as usize] += 1;
    }
    let expected = draws as f64 / 16.0;
    let chi2: f64 = counts
        .iter()
        .map(|c| {
            let d = *c as f64 - expected;
            d * d / expected
        })
        .sum();
    // 15 d.f., p = 0.001 critical value 37.70.
    eprintln!("position-keyed argmax chi2(15) = {chi2:.2} over {draws} draws: {counts:?}");
    assert!(
        chi2 < 37.70,
        "the position-keyed argmax is not uniform over equal logits: \
         chi2(15) = {chi2:.2} against a p=0.001 critical 37.70, counts {counts:?}"
    );
    Ok(())
}

/// The property a sampled verify rests on: drawing a WINDOW of rows with
/// position-keyed noise gives each position the same token a step-by-step
/// sampled decode would have drawn there.
///
/// This is the mechanism behind "sampled speculation is byte-identical to
/// sampled non-speculative decode at the same seed" (#943 §2). A speculative
/// round draws `t` rows at once and keeps only an accepted prefix; a
/// non-speculative run reaches those same absolute positions one at a time.
/// Keyed by position the two agree row for row, so what the drafter happened
/// to propose cannot change the reply — only how fast it is produced.
#[test]
fn a_sampled_window_draws_what_a_step_by_step_sampled_decode_draws() -> Result<()> {
    let dev = Device::Cpu;
    let vocab = 32usize;
    let t = 6usize;
    let first_pos = 11u64;
    // A window of distinct per-row logits, so a row's draw is informative and
    // a row-indexing error cannot be masked by identical rows.
    let mut v = Vec::with_capacity(t * vocab);
    for row in 0..t {
        for c in 0..vocab {
            v.push(((row * 7 + c * 3) % 11) as f32 * 0.21 - 1.0);
        }
    }
    let window = Tensor::from_vec(v.clone(), (1, t, vocab), &dev)?;

    // `GumbelSoftmax` and `ArgMax` only. `Sampling::All` is EXCLUDED on
    // purpose, and the reason is a pre-existing asymmetry this test found:
    // that arm draws by two different mechanisms depending on the path —
    // `sample` (the blocking arm's) takes a host softmax and
    // `sample_multinomial` on the running rng, while `draw_on_device` (the
    // ring arm's, and `window_draw`'s) takes gumbel-max. At one seed over 12
    // logits the two paths read `[1, 3, 1, 0, …]` against `[11, 1, 3, 1, …]`,
    // where `GumbelSoftmax` reads the same stream on both. So "the window
    // equals the step-by-step draw" is not expressible for `All` through
    // `sample`, and the honest subject here is the arms where one mechanism
    // serves both. The asymmetry itself is pinned by
    // `the_all_arm_draws_by_a_different_mechanism_on_each_path` below.
    for sampling in [
        lloom::generation::Sampling::GumbelSoftmax { temperature: 0.8 },
        lloom::generation::Sampling::ArgMax,
    ] {
        // The window, drawn in one call.
        let mut w = LogitsProcessor::from_sampling(1234, sampling.clone());
        let windowed = w.window_draw(&window, first_pos)?;
        assert_eq!(
            windowed.len(),
            t,
            "{sampling:?}: window_draw returned {windowed:?}"
        );

        // The same positions, reached one at a time as a non-speculative
        // decode reaches them — each row drawn by its own sampler armed at
        // that absolute position.
        let mut stepwise = Vec::with_capacity(t);
        for row in 0..t {
            let mut s = LogitsProcessor::from_sampling(1234, sampling.clone());
            s.at_position(Some(first_pos + row as u64));
            // One row as the rank-1 vocabulary vector `sample` takes.
            let one = window.narrow(1, row, 1)?.flatten_all()?;
            stepwise.push(s.sample(&one)?);
        }
        assert_eq!(
            windowed, stepwise,
            "{sampling:?}: the window's rows {windowed:?} differ from the \
             step-by-step draws {stepwise:?} at the same absolute positions"
        );

        // Non-vacuity for the sampled arms: a window whose rows all drew the
        // same token, or that matched the argmax everywhere, would make the
        // agreement above uninformative.
        if !matches!(sampling, lloom::generation::Sampling::ArgMax) {
            let mut a = LogitsProcessor::from_sampling(1234, lloom::generation::Sampling::ArgMax);
            let greedy = a.window_draw(&window, first_pos)?;
            eprintln!("{sampling:?}: sampled {windowed:?} vs greedy {greedy:?}");
            assert_ne!(
                windowed, greedy,
                "{sampling:?}: the sampled window equals the greedy one, so this \
                 test would pass even if the temperature were ignored"
            );
        }
    }
    Ok(())
}

/// A window drawn with position-keyed noise does not depend on the rows
/// drawn and DISCARDED before it — the rejected-draft case, at window
/// granularity. Round 2 of a speculative run reaches position `p` whether
/// round 1 accepted one token or five, and must draw the same token there.
#[test]
fn a_sampled_windows_rows_ignore_the_windows_drawn_before_them() -> Result<()> {
    let dev = Device::Cpu;
    let (vocab, t) = (24usize, 4usize);
    let mut v = Vec::with_capacity(t * vocab);
    for row in 0..t {
        for c in 0..vocab {
            v.push(((row * 5 + c * 2) % 13) as f32 * 0.17);
        }
    }
    let window = Tensor::from_vec(v, (1, t, vocab), &dev)?;
    let at = |discarded_windows: usize| -> Result<Vec<u32>> {
        let mut p = LogitsProcessor::from_sampling(
            99,
            lloom::generation::Sampling::GumbelSoftmax { temperature: 0.9 },
        );
        // Earlier rounds' windows, drawn at other positions and thrown away.
        for w in 0..discarded_windows {
            let _ = p.window_draw(&window, 500 + (w * t) as u64)?;
        }
        p.window_draw(&window, 40)
    };
    let clean = at(0)?;
    for discarded in 1..=4 {
        assert_eq!(
            clean,
            at(discarded)?,
            "the window at position 40 moved after {discarded} discarded \
             windows; a rejected round is still advancing the noise"
        );
    }
    Ok(())
}

/// `window_draw` refuses what it cannot serve: processors without the
/// window's inputs (each row's history is the output plus the drafts before
/// it, which the logits alone do not carry), and a batched window.
#[test]
fn window_draw_refuses_the_arms_it_cannot_key() -> Result<()> {
    let dev = Device::Cpu;
    let window = Tensor::from_vec(vec![0.1f32; 3 * 8], (1, 3, 8), &dev)?;
    let procs = lloom::generation::Processors {
        repetition_penalty: 1.1,
        ..Default::default()
    };
    let mut p = LogitsProcessor::from_sampling(7, lloom::generation::Sampling::ArgMax)
        .with_processors(procs);
    let err = p.window_draw(&window, 0).unwrap_err().to_string();
    assert!(
        err.contains("history"),
        "processors not refused by name: {err}"
    );
    // A batched window is not this seam's.
    let batched = Tensor::from_vec(vec![0.1f32; 2 * 3 * 8], (2, 3, 8), &dev)?;
    let mut p = LogitsProcessor::from_sampling(7, lloom::generation::Sampling::ArgMax);
    assert!(p.window_draw(&batched, 0).is_err(), "B=2 was accepted");
    Ok(())
}

/// The window draws, row for row, what step-by-step draws at the same
/// positions, on the arms #943 part 1 refused: multinomial (the default
/// `-t`) and the host-sort arms (top-k, top-p: the filter, then the
/// inverse CDF from the position's uniform). CPU and Metal.
#[test]
fn every_temperature_arm_draws_a_window_as_a_step_by_step_decode_does() -> Result<()> {
    use lloom::generation::Sampling;
    for dev in devices() {
        let (vocab, t, first_pos) = (40usize, 7usize, 23u64);
        let mut v = Vec::with_capacity(t * vocab);
        for row in 0..t {
            for c in 0..vocab {
                v.push(((row * 11 + c * 5) % 17) as f32 * 0.19 - 1.4);
            }
        }
        let window = Tensor::from_vec(v, (1, t, vocab), &dev)?;
        for sampling in [
            Sampling::All { temperature: 0.9 },
            Sampling::TopK {
                k: 6,
                temperature: 0.8,
            },
            Sampling::TopP {
                p: 0.7,
                temperature: 1.1,
            },
            Sampling::TopKThenTopP {
                k: 9,
                p: 0.8,
                temperature: 0.7,
            },
        ] {
            let mut w = LogitsProcessor::from_sampling(31, sampling.clone());
            let windowed = w.window_draw(&window, first_pos)?;
            let mut stepwise = Vec::with_capacity(t);
            for row in 0..t {
                let mut s = LogitsProcessor::from_sampling(31, sampling.clone());
                s.at_position(Some(first_pos + row as u64));
                stepwise.push(s.sample(&window.narrow(1, row, 1)?.flatten_all()?)?);
            }
            assert_eq!(windowed, stepwise, "{dev:?} {sampling:?}");
            let greedy = LogitsProcessor::from_sampling(31, Sampling::ArgMax)
                .window_draw(&window, first_pos)?;
            assert_ne!(
                windowed, greedy,
                "{dev:?} {sampling:?}: the sampled window is greedy's"
            );
        }
    }
    Ok(())
}

/// Processors under a window: row `i` is processed against the output so
/// far plus the drafts `1..=i` the rows before it were fed, exactly the
/// history a step-by-step decode has when it reaches that position along
/// the drafted path; the history moves only by what the round kept
/// (`commit_window`). CPU and Metal.
#[test]
fn a_windows_rows_read_the_processor_history_of_their_position() -> Result<()> {
    use lloom::generation::Sampling;
    for dev in devices() {
        let (vocab, t, first_pos) = (24usize, 5usize, 60u64);
        let mut v = Vec::with_capacity(t * vocab);
        for row in 0..t {
            for c in 0..vocab {
                v.push(((row * 7 + c * 3) % 13) as f32 * 0.31 - 1.9);
            }
        }
        let window = Tensor::from_vec(v, (1, t, vocab), &dev)?;
        let prompt = [1u32, 4, 9, 4];
        let output = [9u32, 2, 2];
        // The window's inputs: the anchor (the output's last token) and
        // drafts that repeat tokens the penalties see.
        let inputs = [2u32, 9, 9, 17, 4];
        let procs = lloom::generation::Processors {
            repetition_penalty: 1.3,
            presence_penalty: 0.4,
            frequency_penalty: 0.6,
            min_p: 0.05,
            logit_bias: vec![(3, 1.5)],
        };
        for sampling in [
            Sampling::ArgMax,
            Sampling::All { temperature: 0.9 },
            Sampling::GumbelSoftmax { temperature: 0.9 },
            Sampling::TopK {
                k: 7,
                temperature: 0.9,
            },
        ] {
            let one = window.narrow(1, 0, 1)?.flatten_all()?;
            let primed = || -> Result<LogitsProcessor> {
                let mut p = LogitsProcessor::from_sampling(8, sampling.clone())
                    .with_processors(procs.clone());
                p.observe_prompt(&prompt);
                // The output so far, recorded as a decode records its draws.
                for &id in &output {
                    let _ = p.sample_fed(&one, |_| id)?;
                }
                Ok(p)
            };
            let ids = Tensor::new(&inputs, &dev)?;
            let mut w = primed()?;
            let windowed = w
                .window_draw_device(&window, first_pos, Some(&ids))?
                .to_vec1::<u32>()?;
            let mut stepwise = Vec::with_capacity(t);
            for row in 0..t {
                let mut s = primed()?;
                // The drafts the rows before this one were fed are output.
                for &id in &inputs[1..=row] {
                    let _ = s.sample_fed(&one, |_| id)?;
                }
                s.at_position(Some(first_pos + row as u64));
                stepwise.push(s.sample(&window.narrow(1, row, 1)?.flatten_all()?)?);
            }
            assert_eq!(windowed, stepwise, "{dev:?} {sampling:?}");
            // Without the processors the same window draws differently, or
            // the fixture does not exercise them.
            let bare = LogitsProcessor::from_sampling(8, sampling.clone())
                .window_draw(&window, first_pos)?;
            assert_ne!(
                windowed, bare,
                "{dev:?} {sampling:?}: the processors moved nothing"
            );
            // The draw moved no history: a second window draws the same rows.
            let again = w
                .window_draw_device(&window, first_pos, Some(&ids))?
                .to_vec1::<u32>()?;
            assert_eq!(
                windowed, again,
                "{dev:?} {sampling:?}: the draw moved the history"
            );
            // The round's kept tokens move it as a step-by-step decode would.
            w.commit_window(&inputs[1..3])?;
            let mut s = primed()?;
            for &id in &inputs[1..3] {
                let _ = s.sample_fed(&one, |_| id)?;
            }
            s.at_position(Some(first_pos));
            let row0 = window.narrow(1, 0, 1)?;
            let w0 = w
                .window_draw_device(&row0, first_pos, Some(&Tensor::new(&[9u32], &dev)?))?
                .to_vec1::<u32>()?;
            assert_eq!(
                w0,
                vec![s.sample(&one)?],
                "{dev:?} {sampling:?}: commit_window"
            );
        }
    }
    Ok(())
}

/// `Sampling::All` (multinomial) draws ONE stream on both paths: the host
/// draw (`sample`, the blocking arm's) and the device draw
/// (`sample_argmax_on_device`, every ring arm's) are the same inverse-CDF
/// draw from the same uniform, so the readback arm never changes a seeded
/// reply. Until `generation.sampling.method` the host path was a softmax and
/// a multinomial on the running rng and the device path gumbel-max, two
/// streams at one seed (`speculation.md` §10.2z-viii). Unkeyed (the running
/// stream) and keyed by position both; CPU and Metal.
#[test]
fn the_all_arm_draws_one_stream_on_both_paths() -> Result<()> {
    for dev in devices() {
        let logits = Tensor::new(
            &[
                0.1f32, 0.9, -0.3, 0.4, 0.0, 0.7, -1.0, 0.2, 0.5, 0.3, -0.2, 0.8,
            ],
            &dev,
        )?;
        for keyed in [false, true] {
            let host = |sampling: lloom::generation::Sampling| -> Result<Vec<u32>> {
                let mut p = LogitsProcessor::from_sampling(5, sampling);
                (0..48)
                    .map(|i| {
                        if keyed {
                            p.at_position(Some(100 + i));
                        }
                        p.sample(&logits)
                    })
                    .collect()
            };
            let device = |sampling: lloom::generation::Sampling| -> Result<Vec<u32>> {
                let mut p = LogitsProcessor::from_sampling(5, sampling);
                (0..48)
                    .map(|i| {
                        if keyed {
                            p.at_position(Some(100 + i));
                        }
                        Ok(p.sample_argmax_on_device(&logits)?
                            .flatten_all()?
                            .to_vec1::<u32>()?[0])
                    })
                    .collect()
            };
            for sampling in [
                lloom::generation::Sampling::All { temperature: 1.0 },
                lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
            ] {
                let (h, d) = (host(sampling.clone())?, device(sampling.clone())?);
                assert_eq!(
                    h, d,
                    "{dev:?} {sampling:?} keyed={keyed}: host {h:?} against device {d:?}"
                );
                let distinct = h.iter().collect::<std::collections::BTreeSet<_>>().len();
                assert!(distinct > 4, "{sampling:?}: a degenerate stream {h:?}");
            }
        }
    }
    Ok(())
}

/// The multinomial draw samples `softmax(z / T)`: over 16 000 position-keyed
/// draws its frequencies meet the distribution (χ², 7 d.f.), at two
/// temperatures — a draw that ignored the temperature fails the second.
#[test]
fn the_multinomial_draw_samples_the_tempered_distribution() -> Result<()> {
    let z = [0.5f32, -0.2, 1.1, 0.0, -1.3, 0.8, 0.3, -0.6];
    let logits = Tensor::new(&z, &Device::Cpu)?;
    for t in [1.0f64, 0.5] {
        let mut p =
            LogitsProcessor::from_sampling(42, lloom::generation::Sampling::All { temperature: t });
        let draws = 16_000usize;
        let mut counts = [0usize; 8];
        for pos in 0..draws {
            p.at_position(Some(pos as u64));
            counts[p.sample(&logits)? as usize] += 1;
        }
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = z.iter().map(|&x| ((x as f64 - m) / t).exp()).collect();
        let total: f64 = e.iter().sum();
        let chi2: f64 = counts
            .iter()
            .zip(&e)
            .map(|(&c, &ei)| {
                let want = draws as f64 * ei / total;
                (c as f64 - want).powi(2) / want
            })
            .sum();
        eprintln!("T={t}: chi2(7) = {chi2:.2}, counts {counts:?}");
        // 7 d.f., p = 0.001 critical 24.32.
        assert!(
            chi2 < 24.32,
            "T={t}: chi2(7) = {chi2:.2}, counts {counts:?}"
        );
    }
    Ok(())
}

/// The per-row processor history at a real vocabulary (128 000) and a
/// window of ten rows, on the CPU and Metal: the window's draws are the
/// step-by-step draws with the same history, multinomial and greedy, with
/// drafts that repeat tokens the penalties and min-p act on.
#[test]
fn a_real_vocabulary_window_reads_each_rows_processor_history() -> Result<()> {
    use lloom::generation::Sampling;
    let (vocab, t, first_pos) = (128_000usize, 10usize, 300u64);
    let mut s = 0x1234_5678u32;
    let host: Vec<f32> = (0..t * vocab)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 12.0 - 6.0
        })
        .collect();
    // The rows' top tokens repeated among the drafts, so the penalties move
    // the draws.
    let top = |r: usize| -> u32 {
        let row = &host[r * vocab..(r + 1) * vocab];
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    let inputs: Vec<u32> = (0..t).map(|r| top((r + 3) % t)).collect();
    let prompt: Vec<u32> = vec![top(0), top(1), 5, 77, 1024];
    let output: Vec<u32> = vec![top(2), 9, top(2)];
    let procs = lloom::generation::Processors {
        repetition_penalty: 1.3,
        presence_penalty: 0.8,
        frequency_penalty: 0.5,
        min_p: 0.02,
        logit_bias: vec![(top(4), 2.0)],
    };
    for dev in devices() {
        let window = Tensor::from_vec(host.clone(), (1, t, vocab), &dev)?;
        let ids = Tensor::new(inputs.as_slice(), &dev)?;
        for sampling in [Sampling::All { temperature: 0.7 }, Sampling::ArgMax] {
            let one = window.narrow(1, 0, 1)?.flatten_all()?;
            let primed = || -> Result<LogitsProcessor> {
                let mut p = LogitsProcessor::from_sampling(11, sampling.clone())
                    .with_processors(procs.clone());
                p.observe_prompt(&prompt);
                for &id in &output {
                    let _ = p.sample_fed(&one, |_| id)?;
                }
                Ok(p)
            };
            let windowed = primed()?
                .window_draw_device(&window, first_pos, Some(&ids))?
                .to_vec1::<u32>()?;
            let mut stepwise = Vec::with_capacity(t);
            for row in 0..t {
                let mut p = primed()?;
                for &id in &inputs[1..=row] {
                    let _ = p.sample_fed(&one, |_| id)?;
                }
                p.at_position(Some(first_pos + row as u64));
                stepwise.push(p.sample(&window.narrow(1, row, 1)?.flatten_all()?)?);
            }
            assert_eq!(windowed, stepwise, "{dev:?} {sampling:?}");
            let bare = LogitsProcessor::from_sampling(11, sampling.clone())
                .window_draw(&window, first_pos)?;
            assert_ne!(
                windowed, bare,
                "{dev:?} {sampling:?}: the processors moved nothing"
            );
        }
    }
    Ok(())
}

/// The keyed uniform's derivation, pinned by value: a change to the stream
/// key or to how a position's generator yields its uniform moves every
/// multinomial stream, and fails here first.
#[test]
fn the_keyed_uniform_is_pinned() {
    let got: Vec<f32> = [(42u64, 0u64), (42, 1), (42, 1000), (7, 0)]
        .iter()
        .map(|&(s, p)| LogitsProcessor::uniform_at(s, p))
        .collect();
    eprintln!("uniform_at: {got:?}");
    for u in &got {
        assert!((0.0..1.0).contains(u), "{got:?}");
    }
    assert_eq!(
        got.iter().map(|u| u.to_bits()).collect::<Vec<_>>(),
        PINNED_UNIFORMS.to_vec(),
        "uniform_at moved: {got:?}"
    );
}

/// `uniform_at` at (42, 0), (42, 1), (42, 1000), (7, 0), as bits: 0.34492636,
/// 0.85419625, 0.6753466, 0.75972766 — and what an independent replica of
/// the derivation computes (`measurements/issue-943-raw/part2/procs/keyed_uniform.py`:
/// the stream key, rand_core's PCG32 seed expansion, ChaCha12's first word).
const PINNED_UNIFORMS: [u32; 4] = [1051761200, 1062907035, 1059906436, 1061322115];
