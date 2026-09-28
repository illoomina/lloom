//! The logits processors through the sampler's public paths: the host draw
//! (`sample`, the blocking arm) and the device draw (`sample_argmax_on_device`,
//! the ring arm), each against a host reference that recomputes the penalties
//! from the whole history every step.

use lloom::generation::{LogitsProcessor, Processors, Sampling};
use lloom_tensor::{Device, Tensor};

const VOCAB: usize = 61;
const STEPS: usize = 48;

fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    v.push(Device::new_metal(0).expect("the `metal` feature is on and no Metal device opened"));
    v
}

/// Per-step logits that favour a handful of tokens strongly, so an
/// unpenalized greedy run repeats and a penalized one must move off them.
fn step_logits(step: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_u32 ^ (step as u32).wrapping_mul(0x85eb_ca6b);
    (0..VOCAB)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((s >> 9) % 64) as f32 * 0.03125 - 1.0;
            let favoured = match i {
                4 => 6.0,
                9 => 5.5,
                17 => 5.0,
                _ => 0.0,
            };
            favoured + noise
        })
        .collect()
}

/// vLLM's penalties over the full history, recomputed per step.
fn reference_step(logits: &[f32], prompt: &[u32], output: &[u32], p: &Processors) -> Vec<f32> {
    let mut counts = vec![0f32; logits.len()];
    for &o in output {
        counts[o as usize] += 1.0;
    }
    let mut x = logits.to_vec();
    for &(id, b) in &p.logit_bias {
        x[id as usize] += b;
    }
    let r = p.repetition_penalty;
    for i in 0..x.len() {
        let seen = counts[i] > 0.0 || prompt.contains(&(i as u32));
        if r != 1.0 && seen {
            x[i] *= if x[i] > 0.0 {
                (1.0 / r) as f32
            } else {
                r as f32
            };
        }
        x[i] -= counts[i] * p.frequency_penalty as f32;
        if counts[i] > 0.0 {
            x[i] -= p.presence_penalty as f32;
        }
    }
    x
}

/// Lowest index among the maxima, as the device argmax breaks ties.
fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

fn penalties() -> Processors {
    Processors {
        repetition_penalty: 1.15,
        presence_penalty: 0.4,
        frequency_penalty: 0.35,
        min_p: 0.0,
        logit_bias: vec![(9, -0.25), (30, 1.0)],
    }
}

const PROMPT: &[u32] = &[17, 2, 3];

fn greedy_run(device: &Device, p: &Processors, on_device: bool) -> Vec<u32> {
    let mut sampler =
        LogitsProcessor::from_sampling(0, Sampling::ArgMax).with_processors(p.clone());
    sampler.observe_prompt(PROMPT);
    (0..STEPS)
        .map(|s| {
            let logits = Tensor::from_vec(step_logits(s), VOCAB, device).unwrap();
            if on_device {
                sampler
                    .sample_argmax_on_device(&logits)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap()
            } else {
                sampler.sample(&logits).unwrap()
            }
        })
        .collect()
}

/// Greedy with penalties: the device draw (history advanced from the device
/// token) and the host draw (from the host id) both equal the reference, and
/// the reference differs from the unpenalized run — the fixture is not blind.
#[test]
fn greedy_with_penalties_matches_the_reference_on_both_draws() {
    let p = penalties();
    let mut want = Vec::new();
    for s in 0..STEPS {
        let x = reference_step(&step_logits(s), PROMPT, &want, &p);
        want.push(argmax(&x));
    }
    let plain: Vec<u32> = (0..STEPS).map(|s| argmax(&step_logits(s))).collect();
    assert_ne!(want, plain, "the penalties changed nothing on this fixture");
    for device in devices() {
        assert_eq!(
            greedy_run(&device, &p, true),
            want,
            "{device:?}: the device draw"
        );
        assert_eq!(
            greedy_run(&device, &p, false),
            want,
            "{device:?}: the host draw"
        );
    }
}

/// Neutral settings install nothing: the sampler is the one it was, and its
/// draws are the plain argmax's.
#[test]
fn neutral_processors_are_the_plain_sampler() {
    let sampler =
        LogitsProcessor::from_sampling(0, Sampling::ArgMax).with_processors(Processors::default());
    assert!(sampler.processors().is_none());
    let plain: Vec<u32> = (0..STEPS).map(|s| argmax(&step_logits(s))).collect();
    for device in devices() {
        assert_eq!(
            greedy_run(&device, &Processors::default(), true),
            plain,
            "{device:?}"
        );
        assert_eq!(
            greedy_run(&device, &Processors::default(), false),
            plain,
            "{device:?}"
        );
    }
}

fn sampled_run(device: &Device, sampling: Sampling, seed: u64, on_device: bool) -> Vec<u32> {
    let p = Processors {
        min_p: 0.05,
        ..penalties()
    };
    let mut sampler = LogitsProcessor::from_sampling(seed, sampling).with_processors(p);
    sampler.observe_prompt(PROMPT);
    (0..STEPS)
        .map(|s| {
            let logits = Tensor::from_vec(step_logits(s), VOCAB, device).unwrap();
            if on_device {
                sampler
                    .sample_argmax_on_device(&logits)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap()
            } else {
                sampler.sample(&logits).unwrap()
            }
        })
        .collect()
}

/// A seeded sampled run with every processor reproduces, on each draw path
/// and sampler arm; another seed draws another run.
#[test]
fn seeded_sampled_runs_with_processors_reproduce() {
    let arms = [
        Sampling::All { temperature: 0.9 },
        Sampling::TopK {
            k: 8,
            temperature: 0.9,
        },
        Sampling::TopP {
            p: 0.9,
            temperature: 0.9,
        },
        Sampling::GumbelSoftmax { temperature: 0.9 },
    ];
    for device in devices() {
        for arm in &arms {
            for on_device in [false, true] {
                if on_device
                    && !matches!(arm, Sampling::All { .. } | Sampling::GumbelSoftmax { .. })
                {
                    continue;
                }
                let a = sampled_run(&device, arm.clone(), 7, on_device);
                let b = sampled_run(&device, arm.clone(), 7, on_device);
                let c = sampled_run(&device, arm.clone(), 8, on_device);
                let at = format!("{device:?} {arm:?} on_device={on_device}");
                assert_eq!(a, b, "{at}: seed 7 twice");
                assert_ne!(a, c, "{at}: seeds 7 and 8 drew the same {STEPS} tokens");
            }
        }
    }
}

/// Min-p drops every token below its threshold: where one token clears it
/// by a wide margin, a sampled run at temperature 1 draws nothing else.
#[test]
fn min_p_confines_a_sampled_run_to_the_tokens_it_keeps() {
    let p = Processors {
        min_p: 0.5,
        ..Default::default()
    };
    let peaked = |s: usize| {
        let mut x = step_logits(s);
        x[(s * 5) % VOCAB] = 20.0;
        x
    };
    for device in devices() {
        for on_device in [false, true] {
            let mut sampler = LogitsProcessor::from_sampling(3, Sampling::All { temperature: 1.0 })
                .with_processors(p.clone());
            for s in 0..STEPS {
                let logits = Tensor::from_vec(peaked(s), VOCAB, &device).unwrap();
                let id = if on_device {
                    sampler
                        .sample_argmax_on_device(&logits)
                        .unwrap()
                        .to_scalar::<u32>()
                        .unwrap()
                } else {
                    sampler.sample(&logits).unwrap()
                };
                assert_eq!(
                    id,
                    ((s * 5) % VOCAB) as u32,
                    "{device:?} on_device={on_device} step {s}"
                );
            }
        }
    }
}
