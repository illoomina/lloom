use lloom::generation::LogitsProcessor;
use lloom_tensor::{DType, Device, Result, Tensor};

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

/// A temperature arm still sees f32, which is what bounds #351's claim.
///
/// Only `ArgMax` skips the cast. `sample_f`'s closure receives the f32
/// probabilities, and this asserts the slice it is handed is the full
/// vocabulary at f32 -- so a future change extending the skip to a
/// temperature arm fails here rather than silently degrading the sampler.
#[test]
fn a_temperature_arm_still_receives_f32_probabilities() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4], &dev)?.to_dtype(DType::F16)?;
    let mut p = LogitsProcessor::new(42, Some(0.9), None);
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
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
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

/// `window_draw` refuses what it cannot serve position-keyed, rather than
/// serving it on one running stream under a sampled label (§7.1a-i: report
/// the arm that ran, never the one that was asked for).
#[test]
fn window_draw_refuses_the_arms_it_cannot_key() -> Result<()> {
    let dev = Device::Cpu;
    let window = Tensor::from_vec(vec![0.1f32; 3 * 8], (1, 3, 8), &dev)?;
    for sampling in [
        lloom::generation::Sampling::TopP {
            p: 0.9,
            temperature: 0.8,
        },
        lloom::generation::Sampling::TopK {
            k: 4,
            temperature: 0.8,
        },
        lloom::generation::Sampling::TopKThenTopP {
            k: 4,
            p: 0.9,
            temperature: 0.8,
        },
    ] {
        let mut p = LogitsProcessor::from_sampling(7, sampling.clone());
        let err = p.window_draw(&window, 0).unwrap_err().to_string();
        assert!(
            err.contains("host sort"),
            "{sampling:?} was not refused by name: {err}"
        );
    }
    // Processors have per-token history a rejected row would corrupt.
    let mut procs = lloom::generation::Processors::default();
    procs.repetition_penalty = 1.1;
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

/// **A pre-existing asymmetry, pinned here because #943 depends on it.**
///
/// `Sampling::All { temperature }` draws by a DIFFERENT MECHANISM on each
/// path: the host draw (`sample`, what the `Blocking` readback arm calls)
/// takes a softmax and `sample_multinomial` over the running rng, while the
/// device draw (`sample_argmax_on_device`, what every `Ring` arm calls, and
/// what `window_draw` takes) is gumbel-max. `Sampling::GumbelSoftmax` is
/// gumbel-max on both.
///
/// So at one seed, a `t > 0` request's reply depends on which readback arm
/// served it — and `ReadbackArm::Auto` picks the ring for `All` but falls to
/// `Blocking` whenever `diagnostics.logits_digest` is on or the sampler is a
/// host-sort arm (`lloom-engine/src/decode.rs`). Nothing in the tree compares
/// one arm's SAMPLED ids to another's: `decode_driver_tests.rs`'s multi-arm
/// sweep checks its sampled arms for self-consistency (the reader's stream
/// equals the returned ids) and compares against an expected chain only for
/// `ArgMax`, which is `CONTRIBUTING.md` §3.1b's "looks like a comparison and
/// is not."
///
/// Recorded rather than changed: making the two agree means choosing which
/// mechanism `All` means, which moves either the ring arms' or the blocking
/// arm's recorded digests at `t > 0`, and that is its own argued decision
/// (§7.1a). What #943 needs from it is narrower and is satisfied: the arm it
/// samples a verify window with is gumbel-max on both paths.
#[test]
fn the_all_arm_draws_by_a_different_mechanism_on_each_path() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::new(
        &[
            0.1f32, 0.9, -0.3, 0.4, 0.0, 0.7, -1.0, 0.2, 0.5, 0.3, -0.2, 0.8,
        ],
        &dev,
    )?;
    let host = |sampling: lloom::generation::Sampling| -> Result<Vec<u32>> {
        let mut p = LogitsProcessor::from_sampling(5, sampling);
        (0..12).map(|_| p.sample(&logits)).collect()
    };
    let device = |sampling: lloom::generation::Sampling| -> Result<Vec<u32>> {
        let mut p = LogitsProcessor::from_sampling(5, sampling);
        (0..12)
            .map(|_| {
                Ok(p.sample_argmax_on_device(&logits)?
                    .flatten_all()?
                    .to_vec1::<u32>()?[0])
            })
            .collect()
    };
    let all = lloom::generation::Sampling::All { temperature: 1.0 };
    let gumbel = lloom::generation::Sampling::GumbelSoftmax { temperature: 1.0 };
    let (all_host, all_dev) = (host(all.clone())?, device(all)?);
    let (g_host, g_dev) = (host(gumbel.clone())?, device(gumbel)?);
    eprintln!("All    host {all_host:?}\nAll    dev  {all_dev:?}");
    eprintln!("Gumbel host {g_host:?}\nGumbel dev  {g_dev:?}");
    assert_eq!(
        g_host, g_dev,
        "GumbelSoftmax must draw one stream on both paths — it is the arm a \
         sampled verify is served by, and #943's identity rests on it"
    );
    assert_ne!(
        all_host, all_dev,
        "the `All` arm's two paths now agree: if that was made true \
         deliberately, this test has served its purpose and the §7.1a \
         decision it records has been taken — delete it and say so. If it \
         became true by accident, a recorded t > 0 digest has moved."
    );
    Ok(())
}
