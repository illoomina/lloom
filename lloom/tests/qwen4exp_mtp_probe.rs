//! `QWEN4EXP_MTP_PROBE` on the generation rows. A round's last row is the
//! one the next draft is taken from, and the probe scores that draft against
//! the next observe's first token — which is what the verify compared it
//! with. So at K = 1 the probe's last-row figures are fixed by the run's own
//! rounds, an absolute reference from a different route (CONTRIBUTING
//! §3.1b): one scored row per round, a hit exactly when the round accepted
//! its draft; and every accepted draft's row is scored once more as an
//! inner row.
//!
//! Its own test binary: the knob is read once per process.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, MtpDrafter};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, Session};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::{Device, Result};

#[test]
fn the_generation_probe_scores_every_round_and_agrees_with_the_verify() -> Result<()> {
    std::env::set_var("QWEN4EXP_MTP_PROBE", "1");
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg.eos_token_id = 127;
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let head =
        testing::mtp_head::<CrownExact, _>(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let eos = [cfg.eos_token_id];
    let max = 24;
    let mut checked = 0;
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..6).map(|i| (seed * 13 + i * 5 + 2) % 120).collect();
        let mut cache = RewindableExact::single(&model, &dev, 1)?;
        cache.want_nextn = true;
        let mut session = Session::new(&model, cache);
        let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut mtp,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(max),
            &eos,
            &mut |_| Ok(()),
        )?;
        if out.eos {
            // An EOS cuts the accepted prefix at itself, not at the draft.
            continue;
        }
        let p = mtp.generation_probe();
        let rounds = &out.stats.rounds;
        let accepted_drafts = rounds.iter().filter(|r| r.accepted > 1).count() as u64;
        let inner: u64 = rounds.iter().map(|r| r.accepted as u64 - 1).sum();
        println!(
            "MTPPROBEGEN seed={seed} rounds={} accepted_drafts={accepted_drafts} {p:?}",
            rounds.len()
        );
        assert_eq!(p.last_rows, rounds.len() as u64, "one last row a round");
        assert_eq!(
            p.last_hits, accepted_drafts,
            "a last row hits exactly when its round accepted the draft"
        );
        assert_eq!(
            p.inner_rows, inner,
            "every accepted draft's row is an inner row"
        );
        assert!(p.inner_hits <= p.inner_rows);
        checked += 1;
    }
    assert!(checked >= 2, "only {checked} runs reached the cap");
    Ok(())
}
