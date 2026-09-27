//! The processors on the device draw read nothing back to the host: a run of
//! steps with every processor on moves neither `to_cpu` counter. Alone in its
//! binary because the counters are process-wide — a sibling test's readback
//! would land in this one's delta.
#![cfg(feature = "metal")]

use lloom::generation::{LogitsProcessor, Processors, Sampling};
use lloom_tensor::{Device, Result, Tensor};

const VOCAB: usize = 4099;

fn step_logits(step: usize, device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..VOCAB)
        .map(|i| ((i * 31 + step * 17) % 97) as f32 * 0.0625 - 3.0)
        .collect();
    Tensor::from_vec(v, VOCAB, device)
}

/// Greedy and tempered, every processor on: 32 device draws after the first
/// (which builds the stage's device state from host uploads) leave both
/// readback counters where they were; reading one drawn token back moves
/// them, so the zero is not an instrument that reads nothing.
#[test]
fn the_device_draw_with_processors_reads_nothing_back() -> Result<()> {
    let device = Device::new_metal(0)?;
    let p = Processors {
        repetition_penalty: 1.2,
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        min_p: 0.05,
        logit_bias: vec![(3, -1.0), (4000, 2.0)],
    };
    for sampling in [Sampling::ArgMax, Sampling::All { temperature: 0.8 }] {
        let mut sampler =
            LogitsProcessor::from_sampling(11, sampling.clone()).with_processors(p.clone());
        sampler.observe_prompt(&[1, 2, 3, 4000]);
        let mut last = sampler.sample_argmax_on_device(&step_logits(0, &device)?)?;
        let inputs: Vec<Tensor> = (1..33)
            .map(|s| step_logits(s, &device))
            .collect::<Result<_>>()?;
        device.synchronize()?;

        let (fast0, blit0) = lloom_tensor::metal_backend::to_cpu_counts();
        for x in &inputs {
            last = sampler.sample_argmax_on_device(x)?;
        }
        device.synchronize()?;
        let (fast1, blit1) = lloom_tensor::metal_backend::to_cpu_counts();
        assert_eq!(
            (fast1 - fast0, blit1 - blit0),
            (0, 0),
            "{sampling:?}: 32 device draws with processors read back {} fast, {} blit",
            fast1 - fast0,
            blit1 - blit0
        );

        let _ = last.to_scalar::<u32>()?;
        let (fast2, blit2) = lloom_tensor::metal_backend::to_cpu_counts();
        assert!(
            (fast2 - fast1) + (blit2 - blit1) > 0,
            "reading the drawn token back did not move the counters, so the zero above proves nothing"
        );
    }
    Ok(())
}
