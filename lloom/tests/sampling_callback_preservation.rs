use lloom::generation::{LogitsProcessor, Processors, Sampling};
use lloom_tensor::{Device, Result, Tensor};

// Nonzero probability changes must reach the draw, not become a zero-only mask.
// The biased probabilities also pin callback order: after processing, before filtering.
#[test]
fn nonzero_reweighting_runs_once_after_processors_before_selection() -> Result<()> {
    let mut devices = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    devices.push(Device::new_metal(0)?);
    let seed = 7;
    let pos = (0..128)
        .find(|&p| (0.2..0.4).contains(&LogitsProcessor::uniform_at(seed, p)))
        .expect("the fixture requires a uniform strictly between 0.2 and 0.4");
    let arms = [
        Sampling::All { temperature: 1.0 },
        Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
        Sampling::TopP {
            p: 0.8,
            temperature: 1.0,
        },
        Sampling::TopKThenTopP {
            k: 1,
            p: 0.8,
            temperature: 1.0,
        },
    ];
    let processors = Processors {
        logit_bias: vec![(0, 4.0)],
        ..Processors::default()
    };
    for device in devices.drain(..) {
        let z = Tensor::new(&[0f32, 0.0], &device)?;
        for arm in &arms {
            let sampler = || {
                LogitsProcessor::from_sampling(seed, arm.clone())
                    .with_processors(processors.clone())
            };
            let mut plain = sampler();
            plain.at_position(Some(pos));
            assert_eq!(plain.sample(&z)?, 0, "unmodified {arm:?}");
            let mut hooked = sampler();
            hooked.at_position(Some(pos));
            let mut calls = 0;
            let id = hooked.sample_f(&z, |probabilities| {
                calls += 1;
                assert_eq!(probabilities.len(), 2);
                assert!(
                    probabilities[0] > 0.9 && probabilities[1] > 0.0,
                    "callback must see the bias before any filter: {probabilities:?}"
                );
                probabilities.copy_from_slice(&[0.1, 0.9]);
            })?;
            assert_eq!(calls, 1, "callback count for {arm:?}");
            assert_eq!(id, 1, "nonzero callback weights discarded for {arm:?}");
        }
    }
    Ok(())
}
