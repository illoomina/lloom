//! The host-sort draws read their probabilities from a host-read slot the
//! softmax writes (#814 part 5), and the ids they draw are the readback
//! path's. Alone in its binary because the readback counters are
//! process-wide — a sibling test's readback would land in this one's delta,
//! which is what `logits_processors_readback.rs` says of its own.
#![cfg(feature = "metal")]

use lloom::generation::{LogitsProcessor, Sampling};
use lloom_tensor::{Device, Result, Tensor};

const VOCAB: usize = 4099;

/// The readback counters are process-wide, and the test harness runs a
/// binary's tests on threads: each test holds this for its whole body, so
/// one test's readbacks never land in another's delta.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn step_logits(step: usize, device: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..VOCAB)
        .map(|i| ((i * 31 + step * 17) % 97) as f32 * 0.0625 - 3.0)
        .collect();
    Tensor::from_vec(v, VOCAB, device)
}

/// The host-sort arms read their probabilities from a host-read slot the
/// softmax writes (#814 part 5): over 32 draws nothing blits and no staging
/// buffer is allocated, the slot is read once a draw, and the ids are exactly
/// the ones the readback path drew — the slot changes where the values are
/// read, not what they are.
#[test]
fn a_host_sort_draw_reads_its_probabilities_from_a_slot() -> Result<()> {
    let _serial = SERIAL.lock();
    let device = Device::new_metal(0)?;
    let inputs: Vec<Tensor> = (0..32)
        .map(|s| step_logits(s, &device))
        .collect::<Result<_>>()?;
    for sampling in [
        Sampling::TopP {
            p: 0.9,
            temperature: 0.7,
        },
        Sampling::TopK {
            k: 40,
            temperature: 0.7,
        },
    ] {
        // The same seed, so the two arms draw from the same rng sequence.
        let mut sampler = LogitsProcessor::from_sampling(7, sampling.clone());
        device.synchronize()?;
        let before = lloom_tensor::metal_backend::readback_counts();
        let got: Vec<u32> = inputs
            .iter()
            .map(|x| sampler.sample(x))
            .collect::<Result<_>>()?;
        let r = lloom_tensor::metal_backend::readback_counts().since(&before);
        eprintln!("[prs slot] {sampling:?}: {r:?}");
        assert_eq!(r.blit, 0, "{sampling:?}: a draw blitted its logits: {r:?}");
        assert_eq!(
            r.staging_allocs, 0,
            "{sampling:?}: a draw allocated a staging buffer: {r:?}"
        );
        assert_eq!(
            r.slot,
            inputs.len() as u64,
            "{sampling:?}: the draws did not read the slot once each: {r:?}"
        );
        // The reference: the same draws with the slot off, read back.
        let mut reference = LogitsProcessor::from_sampling(7, sampling.clone());
        let want: Vec<u32> = inputs
            .iter()
            .map(|x| {
                let prs = lloom::nn::ops::softmax_last_dim(&(x / 1.0)?)?;
                let _ = prs.to_vec1::<f32>()?;
                reference.sample(x)
            })
            .collect::<Result<_>>()?;
        assert_eq!(got, want, "{sampling:?}: the slot moved the draw");
    }
    Ok(())
}

/// The blocking arm's greedy draw (#814 part 5, item A): the id the host reads
/// from the draw's slot, and the device tensor fed as the next input, are
/// both the host's own argmax of the logits, at every step. The reference is
/// computed here from the host copy of the logits, not by any sampler, so a
/// slot the draw never writes (its reads stale or zero) cannot agree with it.
#[test]
fn the_greedy_draw_reads_its_id_from_the_slot_it_wrote() -> Result<()> {
    let _serial = SERIAL.lock();
    let device = Device::new_metal(0)?;
    let mut sampler = LogitsProcessor::from_sampling(3, Sampling::ArgMax);
    let mut seen = std::collections::BTreeSet::new();
    for step in 0..24 {
        // A different winner each step, never at index 0.
        let mut v: Vec<f32> = (0..VOCAB)
            .map(|i| ((i * 31 + step * 17) % 97) as f32 * 0.0625 - 3.0)
            .collect();
        let hot = 1 + (step * 173) % (VOCAB - 1);
        v[hot] = 100.0;
        let want = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        let x = Tensor::from_vec(v, VOCAB, &device)?;
        let before = lloom_tensor::metal_backend::readback_counts();
        let (draw, id) = sampler
            .draw_and_read(&x)?
            .expect("a Metal device has a draw slot");
        let r = lloom_tensor::metal_backend::readback_counts().since(&before);
        assert_eq!(id, want, "step {step}: the id read from the slot");
        assert_eq!(
            r.slot, 1,
            "step {step}: the id did not come from the slot: {r:?}"
        );
        assert_eq!(
            (r.fast, r.blit, r.staging_allocs),
            (0, 0, 0),
            "step {step}: the id was read back: {r:?}"
        );
        let fed: Vec<u32> = draw.flatten_all()?.to_vec1()?;
        assert_eq!(fed, vec![want], "step {step}: the fed tensor");
        seen.insert(id);
    }
    assert!(
        seen.len() > 20,
        "the steps did not exercise distinct ids: {seen:?}"
    );
    Ok(())
}

/// The device draws a blocking arm makes at a temperature — multinomial (the
/// inverse-CDF dispatch) and gumbel-max — hand the host their id through the
/// draw slot, as the greedy draw does: one slot read a draw, nothing read
/// back, nothing blitted. The reference is the device draw read back
/// directly, at the same positions.
#[test]
fn a_device_draw_at_a_temperature_reads_its_id_from_the_slot() -> Result<()> {
    let _serial = SERIAL.lock();
    let device = Device::new_metal(0)?;
    let inputs: Vec<Tensor> = (0..32)
        .map(|s| step_logits(s, &device))
        .collect::<Result<_>>()?;
    for sampling in [
        Sampling::All { temperature: 0.8 },
        Sampling::GumbelSoftmax { temperature: 0.8 },
    ] {
        let mut sampler = LogitsProcessor::from_sampling(7, sampling.clone());
        device.synchronize()?;
        let before = lloom_tensor::metal_backend::readback_counts();
        let got: Vec<u32> = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                sampler.at_position(Some(i as u64));
                sampler.sample(x)
            })
            .collect::<Result<_>>()?;
        let r = lloom_tensor::metal_backend::readback_counts().since(&before);
        eprintln!("[draw slot] {sampling:?}: {r:?}");
        assert_eq!(
            (r.slot, r.fast, r.blit, r.staging_allocs),
            (inputs.len() as u64, 0, 0, 0),
            "{sampling:?}: {r:?}"
        );
        let mut reference = LogitsProcessor::from_sampling(7, sampling.clone());
        let want: Vec<u32> = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                reference.at_position(Some(i as u64));
                Ok(reference
                    .sample_argmax_on_device(x)?
                    .flatten_all()?
                    .to_vec1::<u32>()?[0])
            })
            .collect::<Result<_>>()?;
        assert_eq!(got, want, "{sampling:?}: the slot moved the draw");
        assert!(
            got.iter().collect::<std::collections::BTreeSet<_>>().len() > 8,
            "{sampling:?}: a degenerate stream {got:?}"
        );
    }
    Ok(())
}
