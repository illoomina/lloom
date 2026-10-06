//! The shape moves bytes, not math (`profiles-and-memory-design.md` §3,
//! #892 B3): the greedy stream of a Qwen4-exp model at `No × One` (its own
//! channels and KV) equals the stream of the same weights at `Ring × One`
//! (the recurrent channels in `spec_pools`' phase rings, the KV its own) —
//! the state a `--mtp` run's greedy reference now runs on. Two
//! instantiations of the same seeded tiny model, one profile each; the
//! profiles are named, never defaulted.

use lloom::models::qwen4exp::{testing, Config};
use lloom::nn::CausalLM;
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::{Device, Result, Tensor};

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0 as u32
}

fn greedy<P: lloom::models::qwen4exp::Shape>(
    model: &lloom::models::qwen4exp::Qwen4Exp<P, lloom::models::qwen4exp::Resident>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let mut cache = P::single(model, dev, 2)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits: Vec<f32> = model
        .forward(&input, 0, &mut cache)?
        .flatten_all()?
        .to_vec1()?;
    let (mut ids, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        let next = argmax(&logits);
        ids.push(next);
        rows.push(logits.clone());
        let pos = cache.pos;
        logits = model
            .forward(&Tensor::from_vec(vec![next], (1, 1), dev)?, pos, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        let _ = cfg;
    }
    Ok((ids, rows))
}

fn check(dev: &Device, tag: &str) -> Result<()> {
    let cfg = Config::tiny();
    let own = testing::resident_model::<CrownExact>(&cfg, dev);
    let ring = testing::resident_model::<RewindableExact>(&cfg, dev);
    let prompt = [3u32, 11, 5, 8, 2, 7];
    let (ids_a, logits_a) = greedy(&own, &cfg, dev, &prompt, 8)?;
    let (ids_b, logits_b) = greedy(&ring, &cfg, dev, &prompt, 8)?;
    assert_eq!(
        ids_a, ids_b,
        "{tag}: the streams diverge between the shapes"
    );
    let mut worst = 0f32;
    for (a, b) in logits_a.iter().zip(&logits_b) {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        worst = worst.max(d);
    }
    println!("SHAPES {tag} own-vs-ring max|dlogit|={worst:e} over 8 steps");
    // Non-vacuity: the stream is not a constant.
    assert!(
        ids_a.windows(2).any(|w| w[0] != w[1]),
        "{tag}: a constant stream"
    );
    Ok(())
}

#[test]
fn the_greedy_stream_is_the_same_at_no_x_one_and_ring_x_one_on_the_cpu() -> Result<()> {
    check(&Device::Cpu, "cpu")
}

#[cfg(feature = "metal")]
#[test]
fn the_greedy_stream_is_the_same_at_no_x_one_and_ring_x_one_on_metal() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    check(&dev, "metal")
}
