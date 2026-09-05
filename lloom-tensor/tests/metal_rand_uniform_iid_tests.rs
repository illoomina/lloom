//! `rand_uniform` must be i.i.d. WITHIN a vector, not merely uniform per position.
//!
//! lloom #345, following #277 (`DESIGN.md` §11.2c). The Metal kernel wrote two
//! elements per thread -- `out[tid]` and `out[size - off - tid]` -- from two
//! CONSECUTIVE states of one `HybridTaus` stream. Consecutive outputs of that
//! generator are a deterministic function of one another, so the pair is
//! supported on a curve in the unit square rather than filling it.
//!
//! **Every marginal check passes on that kernel.** Per-position means are
//! uniform, pairwise correlation is ~0, and successive calls are independent.
//! `candle-metal-kernels`' own `random` test asserts range and mean and is
//! structurally incapable of seeing this (`DESIGN.md` §15.1a's INEXPRESSIVE
//! class). What fails is the property GPU sampling actually depends on:
//!
//!   with equal weights, the argmax over a drawn vector must be uniform
//!   over positions.
//!
//! The maximum is a TAIL statistic, so a source can have correct marginals and
//! zero correlation and still place its maxima wrongly. That is why this
//! survived: nothing was asking the question the max asks.
//!
//! These tests are the mutation control for the fix. On the pre-#345 kernel
//! they read chi-squared 300 / 772 / 1485 / 29 200 at n = 4 / 8 / 16 / 256
//! against critical values 16.27 / 24.32 / 37.70 / 330.55, i.e. they FAIL by
//! one to two orders of magnitude.
//!
//! Metal only: the CPU and CUDA backends draw from their own generators and are
//! not affected by the `.metal` change this pins. The tests are `#[cfg]`-gated
//! rather than made backend-generic for that reason.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};

/// Upper-tail critical values of the chi-squared distribution at p = 0.001,
/// for the degrees of freedom this file uses.
///
/// Quoted rather than computed so the test carries no statistics
/// implementation of its own; each is `df` with the value from a standard
/// table, cross-checked against `scipy.stats.chi2.isf(0.001, df)` in
/// `measurements/issue-345-raw/critical.py`.
fn critical_p001(df: usize) -> f64 {
    match df {
        3 => 16.2662,
        7 => 24.3219,
        15 => 37.6973,
        255 => 330.5197,
        999 => 1142.8480,
        127_999 => 129_568.2429,
        _ => panic!("no tabulated critical value for df={df}; add it deliberately"),
    }
}

/// Draw `draws` vectors of `n` uniforms and count where the argmax lands.
///
/// Each draw is one `rand_uniform` dispatch, which is the unit the defect lives
/// in: the kernel's thread-to-element mapping is a property of a single call.
fn argmax_position_counts(device: &Device, n: usize, draws: usize) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; n];
    for _ in 0..draws {
        let v = Tensor::rand(0f32, 1f32, n, device)?.to_vec1::<f32>()?;
        let mut best = 0usize;
        for (i, x) in v.iter().enumerate() {
            if x > &v[best] {
                best = i;
            }
        }
        counts[best] += 1;
    }
    Ok(counts)
}

/// Pearson's chi-squared against the uniform expectation.
fn chi_squared_uniform(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    assert!(
        expected >= 5.0,
        "chi-squared needs an expected cell count of at least 5; got {expected:.2} \
         from {total} draws over {} cells. Raise the draw count rather than \
         lowering this bound -- below 5 the statistic is not chi-squared \
         distributed and no critical value applies.",
        counts.len()
    );
    counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum()
}

/// The core property, at the widths #277 measured.
///
/// 40 000 draws per width, matching #277's fixture so the numbers are directly
/// comparable to the ones recorded in `DESIGN.md` §11.2c.
#[test]
fn rand_uniform_argmax_is_uniform_over_positions() -> Result<()> {
    let device = Device::new_metal(0)?;
    let draws = 40_000;

    for n in [4usize, 8, 16, 256] {
        let counts = argmax_position_counts(&device, n, draws)?;
        let chi2 = chi_squared_uniform(&counts);
        let critical = critical_p001(n - 1);

        assert!(
            chi2 < critical,
            "argmax position is not uniform at n={n}: chi-squared {chi2:.1} against \
             a p=0.001 critical value of {critical:.2} ({:.1}x). Counts: {counts:?}. \
             With equal weights every position must win equally often; this is the \
             property gumbel-max sampling depends on, and marginal uniformity does \
             not imply it (lloom #345).",
            chi2 / critical
        );
    }
    Ok(())
}

/// The same property at the width that actually ships.
///
/// #344 measured at 256 and explicitly did not test here. LFM2's vocab is
/// 128 000, and the kernel's structure is width-dependent -- the mirror pairing
/// is `tid` against `size - off - tid`, so which elements share a stream is a
/// function of `size`. A fix verified only at 256 has not been verified where
/// it ships.
///
/// 640 000 draws keeps the expected cell count at 5.0 exactly, which is the
/// floor `chi_squared_uniform` enforces. This is the expensive test in the file
/// and is the reason the width question is answered rather than deferred.
#[test]
#[ignore = "640k draws at n=128000; run explicitly with --ignored for the width gate"]
fn rand_uniform_argmax_is_uniform_at_the_real_vocab_width() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 128_000usize;
    let draws = 640_000;

    let counts = argmax_position_counts(&device, n, draws)?;
    let chi2 = chi_squared_uniform(&counts);
    let critical = critical_p001(n - 1);

    assert!(
        chi2 < critical,
        "argmax position is not uniform at the real vocab width n={n}: \
         chi-squared {chi2:.1} against a p=0.001 critical value of {critical:.1} \
         ({:.2}x)",
        chi2 / critical
    );
    Ok(())
}

/// The pairing is mirrored, so a test comparing ADJACENT positions finds
/// nothing. This pins the mechanism rather than only the symptom.
///
/// Under the pre-#345 kernel, positions `i` and `size - off - i` came from two
/// consecutive states of one stream, so `v[size-1-i]` is a deterministic
/// function of `v[i]` at every `i` -- and therefore, over many draws, the sign
/// of `v[i] - v[size-1-i]` is not a fair coin. Adjacent positions `i` and `i+1`
/// belong to DIFFERENT threads and show no such structure, which is why the
/// defect hid.
#[test]
fn rand_uniform_mirror_pairs_are_exchangeable() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 256usize;
    let draws = 20_000;

    // Even `n`, so `off == 1` and the mirror of `i` is `n - 1 - i`.
    let mut mirror_wins = 0u64;
    let mut adjacent_wins = 0u64;
    let mut total = 0u64;

    for _ in 0..draws {
        let v = Tensor::rand(0f32, 1f32, n, &device)?.to_vec1::<f32>()?;
        // Compare position 1 against its mirror, and position 1 against 2.
        // Position 0 is skipped: it is the thread that also advances the seed.
        if v[1] > v[n - 1 - 1] {
            mirror_wins += 1;
        }
        if v[1] > v[2] {
            adjacent_wins += 1;
        }
        total += 1;
    }

    // Two-sided binomial bound at ~5 sigma, which at 20 000 draws is ~0.0177.
    let sigma = 0.5 / (total as f64).sqrt();
    let bound = 5.0 * sigma;

    let mirror_rate = mirror_wins as f64 / total as f64;
    let adjacent_rate = adjacent_wins as f64 / total as f64;

    // The adjacent arm is the NON-VACUITY CONTROL: it must pass on both the
    // broken and the fixed kernel. If it ever fails, the harness is measuring
    // something other than the mirror pairing and the mirror verdict below
    // cannot be read (`DESIGN.md` §15.1 #1).
    assert!(
        (adjacent_rate - 0.5).abs() < bound,
        "the adjacent-position control failed at {adjacent_rate:.4} (bound \
         {:.4} from 0.5). This control must hold on ANY kernel; its failure \
         means the harness is wrong rather than the kernel.",
        bound
    );

    assert!(
        (mirror_rate - 0.5).abs() < bound,
        "mirror-paired positions are not exchangeable: P(v[1] > v[{}]) = \
         {mirror_rate:.4}, outside 0.5 +/- {bound:.4}. Those two positions \
         shared one HybridTaus stream and were two consecutive states of it \
         (lloom #345).",
        n - 2
    );
    Ok(())
}

/// An odd length takes the `off == 0` branch, where thread 0 writes only one
/// element and every other thread writes two. The two branches are different
/// code paths and a fix must hold on both.
#[test]
fn rand_uniform_argmax_is_uniform_at_odd_lengths() -> Result<()> {
    let device = Device::new_metal(0)?;
    let draws = 40_000;

    for n in [5usize, 17, 255] {
        let counts = argmax_position_counts(&device, n, draws)?;
        let chi2 = chi_squared_uniform(&counts);
        // df = n - 1, which for these n is not in the small table above.
        // Use the widest tabulated value below df and note the test is
        // therefore conservative in the direction of accepting.
        let df = n - 1;
        let critical = match df {
            4 => 18.4668,
            16 => 39.2524,
            254 => 329.3828,
            _ => unreachable!(),
        };
        assert!(
            chi2 < critical,
            "argmax position is not uniform at odd n={n}: chi-squared {chi2:.1} \
             against critical {critical:.2}. Counts: {counts:?}"
        );
    }
    Ok(())
}

/// `rand_normal` shares the structure and is checked rather than assumed.
///
/// `normal()` has the same shape -- one `HybridTaus`, two `rand()` calls,
/// `out[tid]` and `out[size - off - tid]` -- but its two writes are the two
/// outputs of ONE Box-Muller transform, which is a different relationship from
/// the uniform case. Box-Muller's outputs are independent GIVEN two independent
/// uniforms, and here `u1` and `u2` are two consecutive states of one stream,
/// so the input assumption is violated upstream.
///
/// Prediction P8 in `measurements/issue-345-prediction.md` says this should
/// fail on the pre-fix kernel too.
#[test]
fn rand_normal_argmax_is_uniform_over_positions() -> Result<()> {
    let device = Device::new_metal(0)?;
    let draws = 40_000;

    for n in [4usize, 16, 256] {
        let mut counts = vec![0u64; n];
        for _ in 0..draws {
            let v = Tensor::randn(0f32, 1f32, n, &device)?.to_vec1::<f32>()?;
            let mut best = 0usize;
            for (i, x) in v.iter().enumerate() {
                if x > &v[best] {
                    best = i;
                }
            }
            counts[best] += 1;
        }
        let chi2 = chi_squared_uniform(&counts);
        let critical = critical_p001(n - 1);
        assert!(
            chi2 < critical,
            "rand_normal argmax position is not uniform at n={n}: chi-squared \
             {chi2:.1} against critical {critical:.2}. Counts: {counts:?}"
        );
    }
    Ok(())
}

/// A seeded stream must reproduce, and a DIFFERENT seed must give a different
/// stream.
///
/// The second half is what #277 found missing: tier-3 test 2 passed 3/3 on a
/// mechanism where `--seed` reached nothing at all, because
/// `Device::set_seed` wrote a native `u64` where the kernel reads word 0 as the
/// HIGH half. A generator ignoring its seed still reproduces, so reproduction
/// alone is a vacuous check (`DESIGN.md` §15.1a's VACUOUS class, and §2.3.3c's
/// own correction: the test "needs *and a different seed gives a different
/// stream*").
#[test]
fn a_seeded_stream_reproduces_and_a_different_seed_does_not() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 1024usize;

    device.set_seed(12345)?;
    let a = Tensor::rand(0f32, 1f32, n, &device)?.to_vec1::<f32>()?;

    device.set_seed(12345)?;
    let b = Tensor::rand(0f32, 1f32, n, &device)?.to_vec1::<f32>()?;

    assert_eq!(a, b, "a repeated seed must reproduce its stream");

    device.set_seed(999)?;
    let c = Tensor::rand(0f32, 1f32, n, &device)?.to_vec1::<f32>()?;

    assert_ne!(
        a, c,
        "a DIFFERENT seed must give a different stream. Equality here means the \
         seed is not an input -- which is exactly what #277 found, and which a \
         reproduction-only test passes 3/3 while being false."
    );
    Ok(())
}

/// The seed must survive the low 32 bits specifically.
///
/// `seed_per_thread` truncates to `uint4`, so a seed whose information sits
/// only in the HIGH 32 bits reaches the generator as zero. #277's defect was
/// exactly this: the host wrote a native `u64` and the kernel read word 0 as
/// the high half, turning every seed below 2^32 into `seed << 32` -- whose low
/// 32 bits are zero, and stay zero under any multiply.
#[test]
fn small_seeds_are_distinguishable() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 256usize;

    let mut streams = Vec::new();
    for seed in [1u64, 2, 3, 0xDEAD_BEEF] {
        device.set_seed(seed)?;
        streams.push(Tensor::rand(0f32, 1f32, n, &device)?.to_vec1::<f32>()?);
    }

    for i in 0..streams.len() {
        for j in (i + 1)..streams.len() {
            assert_ne!(
                streams[i], streams[j],
                "seeds {i} and {j} produced identical streams; small seeds are \
                 not reaching the generator"
            );
        }
    }
    Ok(())
}

/// Dtype coverage: the defect is in the shared `rand_uniform<T>` template, so
/// it is present at every instantiation. f16 has only 10 mantissa bits, so ties
/// are common at small `n` and the argmax test is run at a width where they are
/// rare enough not to dominate.
#[test]
fn rand_uniform_argmax_is_uniform_for_f16() -> Result<()> {
    let device = Device::new_metal(0)?;
    let n = 256usize;
    let draws = 40_000;

    let mut counts = vec![0u64; n];
    for _ in 0..draws {
        let v = Tensor::rand(0f32, 1f32, n, &device)?
            .to_dtype(DType::F16)?
            .to_vec1::<half::f16>()?;
        let mut best = 0usize;
        for (i, x) in v.iter().enumerate() {
            if x > &v[best] {
                best = i;
            }
        }
        counts[best] += 1;
    }
    let chi2 = chi_squared_uniform(&counts);
    let critical = critical_p001(n - 1);
    assert!(
        chi2 < critical,
        "f16 argmax position is not uniform: chi-squared {chi2:.1} against \
         critical {critical:.2}"
    );
    Ok(())
}
