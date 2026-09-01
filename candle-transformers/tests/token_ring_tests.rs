//! The device-resident token slot against the host round trip it replaces
//! (lloom #493, `DESIGN.md` §11.5g).
//!
//! The mechanism under test is the pair §11.5g designs: `argmax` left as a
//! **device** tensor, written into a persistent input slot in place with
//! `slice_set`, and compared against a loop-invariant EOS set device-side. The
//! reference is the path that ships today -- `to_scalar` then `Tensor::new` --
//! and the bar is **bit-identity**, because this changes how bytes move and not
//! what they are. §11.5c states it for exactly this seam: *"a moved digest
//! would be a defect. Tier 1."*
//!
//! **What these tests are careful about.** §15.1 #1's sub-rule (#53, §11.3j) is
//! that a parity test must assert its comparison was not vacuous -- *"two
//! kernels that both write nothing agree perfectly"* -- and §15.1a records 11
//! vacuous instances against 6 inexpressive and 5 coarse. The failure shape
//! here is specific and easy to hit: a slot that is never written and a
//! reference that happens to hold the same value agree, and a test asserting
//! only "the ring holds the right token" passes on a `publish` that does
//! nothing when the token is unchanged. So every parity arm below **moves the
//! value it checks**, and the mutation arms assert the tests can fail.

use candle::{DType, Device, Result, Tensor};
use candle_transformers::generation::token_ring::TokenRing;

/// EOS ids from `DESIGN.md` §5.1: bos / eos / pad = 124894 / 124900 / 124893.
const EOS: &[u32] = &[124900];

/// The shipping path, as `lfm2-release`'s loop runs it: read the token to the
/// host, then upload it again as the next step's input (§11.5f's four lines).
fn round_trip_reference(token: &Tensor, device: &Device) -> Result<(u32, Tensor)> {
    let next = token.to_scalar::<u32>()?;
    let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
    Ok((next, input))
}

/// Builds a logits row whose argmax is `at`, the way a decode step produces one.
fn logits_peaked_at(at: usize, vocab: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; vocab];
    v[at] = 1.0;
    Tensor::from_vec(v, vocab, device)
}

fn devices() -> Vec<Device> {
    let mut ds = vec![Device::Cpu];
    if let Ok(d) = Device::new_metal(0) {
        ds.push(d);
    }
    ds
}

/// The ring's input slot holds what the round trip would have uploaded, and the
/// slot is the **same allocation** across steps.
///
/// §10.3e names buffer-identity stability as one of three properties a
/// replacement append must not regress, so it is asserted rather than assumed.
#[test]
fn ring_matches_the_round_trip_and_keeps_one_allocation() -> Result<()> {
    for device in devices() {
        let vocab = 4096;
        let ring = TokenRing::new(7, EOS, &device)?;
        let first_id = ring.input().id();

        // Three steps at three *different* tokens. Moving the value every step
        // is what makes this non-vacuous: a `publish` that did nothing would
        // agree at step 1 and disagree at step 2.
        for &tok in &[11u32, 2044, 3] {
            let logits = logits_peaked_at(tok as usize, vocab, &device)?;
            let sampled = logits.argmax(candle::D::Minus1)?;
            assert_eq!(sampled.dtype(), DType::U32, "argmax must give U32");

            let (want, want_input) = round_trip_reference(&sampled, &device)?;
            assert_eq!(want, tok, "the fixture's argmax is not the token it names");

            ring.publish(&sampled)?;

            assert_eq!(
                ring.input().flatten_all()?.to_vec1::<u32>()?,
                want_input.flatten_all()?.to_vec1::<u32>()?,
                "ring slot disagrees with the round trip on {device:?}"
            );
            assert_eq!(
                ring.input().dims(),
                want_input.dims(),
                "ring slot shape disagrees with the round trip"
            );
        }

        // The slot is the same `Tensor` throughout: `publish` writes through
        // `slice_set` and never rebinds the field. A `Tensor::new` per step --
        // the shipping path's shape -- would give a fresh id every step.
        assert_eq!(
            ring.input().id(),
            first_id,
            "the input slot was replaced; the write leg must be in place"
        );
    }
    Ok(())
}

/// The EOS flag is set device-side, and it distinguishes both ways.
///
/// The both-ways part is the test: an arm that only checks "EOS sets the flag"
/// passes on a flag that is always set. §8.1g records the shape -- *"an
/// error-case test that only checks the bad input is refused passes when
/// everything is refused"*.
#[test]
fn eos_flag_is_set_device_side_and_clears() -> Result<()> {
    for device in devices() {
        let vocab = 128_100;
        let ring = TokenRing::new(1, EOS, &device)?;

        let eos_logits = logits_peaked_at(EOS[0] as usize, vocab, &device)?;
        ring.publish(&eos_logits.argmax(candle::D::Minus1)?)?;
        let hit = ring.eos_flag().flatten_all()?.to_vec1::<u8>()?[0];
        assert_eq!(hit, 1, "EOS token did not set the flag on {device:?}");

        // And it must come back down: a latched flag would end every
        // generation at its first non-EOS token after an EOS.
        let plain = logits_peaked_at(42, vocab, &device)?;
        ring.publish(&plain.argmax(candle::D::Minus1)?)?;
        let miss = ring.eos_flag().flatten_all()?.to_vec1::<u8>()?[0];
        assert_eq!(miss, 0, "flag stayed set after a non-EOS token");
    }
    Ok(())
}

/// Every id in a multi-id EOS set is matched, not just the first.
///
/// `lfm2-release` builds `eos_ids` from config **plus** two tokenizer lookups
/// (`<|im_end|>`, `<|endoftext|>`), so the set is routinely larger than one and
/// a comparison that only read element 0 would pass every single-id test.
#[test]
fn every_eos_id_matches() -> Result<()> {
    let ids: &[u32] = &[124_893, 124_894, 124_900];
    for device in devices() {
        let vocab = 128_100;
        let ring = TokenRing::new(1, ids, &device)?;
        for &id in ids {
            let l = logits_peaked_at(id as usize, vocab, &device)?;
            ring.publish(&l.argmax(candle::D::Minus1)?)?;
            assert_eq!(
                ring.eos_flag().flatten_all()?.to_vec1::<u8>()?[0],
                1,
                "eos id {id} did not match on {device:?}"
            );
        }
        // A token adjacent to a real id must not match -- otherwise the
        // comparison is a range test rather than an equality.
        let near = logits_peaked_at(124_899, vocab, &device)?;
        ring.publish(&near.argmax(candle::D::Minus1)?)?;
        assert_eq!(
            ring.eos_flag().flatten_all()?.to_vec1::<u8>()?[0],
            0,
            "a non-EOS neighbour matched on {device:?}"
        );
    }
    Ok(())
}

/// An empty EOS set is refused at construction.
///
/// §11.5f measured that the model never checks EOS -- all three
/// `eos_token_id` references in `lfm2.rs` are plumbing -- so there is no
/// model-side counterpart to fall back on and a silently non-terminating loop
/// is the alternative.
#[test]
fn empty_eos_set_is_refused() {
    let device = Device::Cpu;
    assert!(TokenRing::new(0, &[], &device).is_err());
}

/// `publish` refuses a non-`U32` tensor rather than coercing it.
///
/// The argmax output is `U32` by construction (`reduce_op` sets
/// `return_index`), so a different dtype means the caller passed something
/// other than the sampled token -- a `to_dtype` here would silently accept a
/// logits row.
#[test]
fn publish_refuses_a_non_u32_tensor() -> Result<()> {
    let device = Device::Cpu;
    let ring = TokenRing::new(0, EOS, &device)?;
    let wrong = Tensor::from_vec(vec![3f32], 1, &device)?;
    assert!(ring.publish(&wrong).is_err());
    Ok(())
}

/// **Mutation arm.** The parity assertion above must fail when the ring is
/// wrong, or it is not a test (`CONTRIBUTING.md` §3.1 #2).
///
/// The mutation is the one a real defect would produce: publishing a token
/// while the slot keeps its previous value -- i.e. `publish` not writing. It is
/// applied to the *reference* side here rather than by editing the production
/// source, so the check is that the comparison discriminates at all; the
/// source-level mutation is recorded in the PR.
#[test]
fn the_parity_comparison_can_fail() -> Result<()> {
    for device in devices() {
        let vocab = 4096;
        let ring = TokenRing::new(7, EOS, &device)?;
        let logits = logits_peaked_at(11, vocab, &device)?;
        ring.publish(&logits.argmax(candle::D::Minus1)?)?;

        // A stale slot: what the ring would hold if `publish` were a no-op.
        let stale = Tensor::from_vec(vec![7u32], (1, 1), &device)?;
        assert_ne!(
            ring.input().flatten_all()?.to_vec1::<u32>()?,
            stale.flatten_all()?.to_vec1::<u32>()?,
            "the comparison cannot distinguish a written slot from a stale one, \
             so the parity test above proves nothing"
        );
    }
    Ok(())
}

/// The token never goes through the host on the ring path.
///
/// This is the property the whole design rests on and it is not observable
/// from the values -- both paths produce the same `u32`. What distinguishes
/// them is that the ring's `publish` consumes a device tensor, so this asserts
/// the *shape of the API*: the sampled tensor's storage stays on the device it
/// was computed on, and the input slot after publishing is on that same device.
#[test]
fn the_published_token_stays_on_the_device() -> Result<()> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let ring = TokenRing::new(1, EOS, &device)?;
    let logits = logits_peaked_at(99, 4096, &device)?;
    let sampled = logits.argmax(candle::D::Minus1)?;
    assert!(sampled.device().is_metal(), "argmax left the device");
    ring.publish(&sampled)?;
    assert!(
        ring.input().device().is_metal(),
        "the input slot left the device"
    );
    assert!(
        ring.eos_flag().device().is_metal(),
        "the EOS flag left the device"
    );
    Ok(())
}
