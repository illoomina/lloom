//! Tier-3 test 3 (`DESIGN.md` §2.3.3c): does GPU gumbel-max sample the
//! reference softmax?
//!
//! This is the acceptance test lloom #277 specified and #345 must pass. It is
//! a **distributional** check, not a digest: at temperature > 0 the output is
//! stochastic by construction, so there is nothing to compare bit for bit.
//!
//! **Both controls run every time, and that is the load-bearing part.** A
//! goodness-of-fit test that has never rejected anything is `DESIGN.md`
//! §15.1 #1's vacuous comparison, and #265's p3 is this project's own case of a
//! confirming probe surviving three runs unchecked.
//!
//! | arm | what it is | must |
//! |---|---|---|
//! | `multinomial` | host `WeightedIndex` over the CPU softmax | **agree** |
//! | `biased` | `softmax(logits * 1.10)` -- deliberately wrong | **reject** |
//! | `gumbel` | `argmax(logits/T + gumbel_noise)` on the GPU | the thing under test |
//!
//! #277 rebuilt this as a throwaway and **did not commit it**, so the
//! instrument behind that issue's headline number was unreachable -- §15.1a's
//! own class, in the artifact rather than in the tree. It is committed here.
//!
//! ```sh
//! cargo run --release --features metal --example gumbel-distribution -- \
//!     --vocab 256 --draws 50000 --logit-scale 0.25
//! ```

use anyhow::{bail, Result};
use candle::{DType, Device, Tensor};
use rand::distr::Distribution;
use rand::SeedableRng;

/// Upper-tail chi-squared critical values at p = 0.001.
///
/// Tabulated rather than computed so this carries no statistics implementation;
/// checked against `scipy.stats.chi2.isf` in
/// `measurements/issue-345-raw/critical.py`.
fn critical_p001(df: usize) -> Option<f64> {
    Some(match df {
        3 => 16.2662,
        7 => 24.3219,
        15 => 37.6973,
        255 => 330.5197,
        999 => 1142.8480,
        _ => return None,
    })
}

struct Verdict {
    name: &'static str,
    chi2: f64,
    tv: f64,
    thin_cells: usize,
}

/// Pearson's chi-squared and total-variation distance of `counts` against
/// `reference` probabilities.
///
/// TV is reported beside chi-squared because **they fail differently**:
/// chi-squared is sensitive to a relative error in a rare cell, TV to an
/// absolute error in a common one (#277's own recommendation).
fn compare(name: &'static str, counts: &[u64], reference: &[f64]) -> Verdict {
    let total: u64 = counts.iter().sum();
    let mut chi2 = 0.0;
    let mut tv = 0.0;
    let mut thin_cells = 0;
    for (i, &c) in counts.iter().enumerate() {
        let expected = reference[i] * total as f64;
        if expected < 5.0 {
            thin_cells += 1;
        }
        let observed = c as f64;
        if expected > 0.0 {
            let d = observed - expected;
            chi2 += d * d / expected;
        }
        tv += (observed / total as f64 - reference[i]).abs();
    }
    Verdict {
        name,
        chi2,
        tv: tv / 2.0,
        thin_cells,
    }
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exp: Vec<f64> = logits.iter().map(|l| ((l - max) as f64).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

fn main() -> Result<()> {
    let mut vocab = 256usize;
    let mut draws = 50_000usize;
    let mut logit_scale = 0.25f32;
    let mut temperature = 1.0f64;
    let mut seed = 12345u64;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let mut next = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--vocab" => vocab = next(&mut i)?.parse()?,
            "--draws" => draws = next(&mut i)?.parse()?,
            "--logit-scale" => logit_scale = next(&mut i)?.parse()?,
            "--temperature" => temperature = next(&mut i)?.parse()?,
            "--seed" => seed = next(&mut i)?.parse()?,
            other => bail!("unknown argument {other:?}"),
        }
        i += 1;
    }

    let device = Device::new_metal(0)?;
    device.set_seed(seed)?;

    // A fixed logits vector. `--logit-scale` flattens it: at scale 1.0 the tail
    // is thin enough that most cells have an expected count below 5, where
    // chi-squared is not chi-squared distributed and no critical value applies.
    // #277's first run left 98 of 256 cells under that bar.
    let logits: Vec<f32> = (0..vocab)
        .map(|k| {
            let x = k as f32 / vocab as f32;
            logit_scale * (3.0 * (6.28 * x).sin() + 2.0 * (2.7 * x).cos() - 4.0 * x)
        })
        .collect();

    let reference = softmax(&logits);
    let logits_t = Tensor::from_slice(&logits, vocab, &device)?.to_dtype(DType::F32)?;

    println!("gumbel-distribution: tier-3 test 3 (DESIGN.md 2.3.3c)");
    println!(
        "  vocab={vocab} draws={draws} logit_scale={logit_scale} \
         temperature={temperature} seed={seed}"
    );
    println!("  device={device:?}");
    println!();

    // --- arm 1: GPU gumbel-max, the thing under test ---------------------
    //
    // argmax(logits/T + G) with G = -log(-log(U)), U ~ rand_uniform on the GPU.
    // Its noise comes from `rand_like` -- i.e. from the kernel #345 fixes.
    //
    // The shipping `lloom::nn::sampling::gumbel_softmax` is called DIRECTLY
    // rather than reimplemented here. A first version of this harness inlined
    // the same
    // arithmetic, and a mutation to the shipping function then left its number
    // **byte-identical** -- so the harness was testing its own copy and could
    // not have reported a defect in the thing it names. That is `DESIGN.md`
    // §2.4's "an instrument that cannot be shown to have engaged has measured
    // nothing", and it was caught by mutation-testing the harness rather than
    // by reading it.
    let mut gumbel_counts = vec![0u64; vocab];
    for _ in 0..draws {
        let idx = lloom::nn::sampling::gumbel_softmax(&logits_t, temperature, 0)?
            .to_scalar::<u32>()? as usize;
        gumbel_counts[idx] += 1;
    }

    // --- arm 2: host multinomial, the POSITIVE control -------------------
    //
    // Draws from the same reference softmax with a host RNG. It must AGREE:
    // if it does not, the harness is wrong rather than the kernel.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let weights: Vec<f64> = reference.clone();
    let dist = rand::distr::weighted::WeightedIndex::new(&weights)?;
    let mut multinomial_counts = vec![0u64; vocab];
    for _ in 0..draws {
        multinomial_counts[dist.sample(&mut rng)] += 1;
    }

    // --- arm 3: a deliberately biased sampler, the NEGATIVE control ------
    //
    // softmax(logits * 1.10) -- a 10 % sharpening. It must REJECT. A test that
    // has never rejected anything has not been shown able to (DESIGN.md 15.1 #1).
    let biased_logits: Vec<f32> = logits.iter().map(|l| l * 1.10).collect();
    let biased_ref = softmax(&biased_logits);
    let biased_dist = rand::distr::weighted::WeightedIndex::new(&biased_ref)?;
    let mut biased_counts = vec![0u64; vocab];
    for _ in 0..draws {
        biased_counts[biased_dist.sample(&mut rng)] += 1;
    }

    let verdicts = [
        compare("gumbel", &gumbel_counts, &reference),
        compare("multinomial", &multinomial_counts, &reference),
        compare("biased(x1.10)", &biased_counts, &reference),
    ];

    let df = vocab - 1;
    let critical = critical_p001(df);

    println!(
        "  {:<16} {:>12} {:>10} {:>16}",
        "arm", "chi2", "TV", "cells with E<5"
    );
    for v in &verdicts {
        println!(
            "  {:<16} {:>12.2} {:>10.5} {:>16}",
            v.name, v.chi2, v.tv, v.thin_cells
        );
    }
    println!();

    match critical {
        Some(c) => {
            println!("  df = {df}, chi2 critical at p=0.001 = {c:.2}");
            println!();
            let gumbel = &verdicts[0];
            let positive = &verdicts[1];
            let negative = &verdicts[2];

            // The controls are checked FIRST. If either misbehaves the gumbel
            // verdict cannot be read at all.
            let positive_ok = positive.chi2 < c;
            let negative_ok = negative.chi2 >= c;

            println!(
                "  positive control: {} ({:.2} {} {:.2})",
                if positive_ok {
                    "AGREES  (ok)"
                } else {
                    "REJECTS (HARNESS BROKEN)"
                },
                positive.chi2,
                if positive_ok { "<" } else { ">=" },
                c
            );
            println!(
                "  negative control: {} ({:.2} {} {:.2})",
                if negative_ok {
                    "REJECTS (ok)"
                } else {
                    "AGREES  (TEST IS VACUOUS)"
                },
                negative.chi2,
                if negative_ok { ">=" } else { "<" },
                c
            );
            println!();

            if !positive_ok || !negative_ok {
                println!("  VERDICT: INDETERMINATE -- a control misbehaved, so this run");
                println!("           says nothing about the sampler. Fix the harness.");
                std::process::exit(2);
            }

            if gumbel.chi2 < c {
                println!(
                    "  VERDICT: PASS -- GPU gumbel-max reads {:.2} against a critical {:.2} \
                     ({:.3}x), with both controls behaving.",
                    gumbel.chi2,
                    c,
                    gumbel.chi2 / c
                );
            } else {
                println!(
                    "  VERDICT: FAIL -- GPU gumbel-max reads {:.2} against a critical {:.2} \
                     ({:.1}x critical, {:.1}x the biased sampler).",
                    gumbel.chi2,
                    c,
                    gumbel.chi2 / c,
                    gumbel.chi2 / negative.chi2
                );
                std::process::exit(1);
            }
        }
        None => {
            println!("  df = {df}: no tabulated critical value.");
            println!("  Add one deliberately rather than interpolating -- and note that at");
            println!("  a large vocab most cells have an expected count far below 5, where");
            println!("  the statistic is not chi-squared distributed at all.");
            std::process::exit(2);
        }
    }

    Ok(())
}
