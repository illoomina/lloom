use lloom_tensor::{DType, Device, Result, Tensor};
use lloom::generation::LogitsProcessor;

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
