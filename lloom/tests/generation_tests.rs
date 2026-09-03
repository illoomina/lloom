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
