//! What a run hands its consumer is the text the server accepted
//! (`sequence-reuse-design.md` §2.3f, decision 42): through
//! `DecodeSession::decode_until` on the tiny Qwen3.5, a consumer that refuses
//! a fragment (a client gone) leaves that fragment out of the reply's text,
//! and a stop string leaves out itself and what follows; the run's ids stay
//! every id it generated.

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::model::CausalLM;
use lloom::nn::Sequence;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{DecodeParams, DecodeSession, ReadbackArm, Stop};
use lloom_tensor::{Device, Result};

type Model = Qwen3_5<CrownExact>;

fn model(dev: &Device) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    Ok((ckpt.model::<CrownExact>(dev)?, ckpt.cfg.clone()))
}

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

const P: [u32; 8] = [1, 3, 5, 7, 9, 11, 13, 15];

fn params(max: usize, stop: &[&str]) -> DecodeParams {
    DecodeParams {
        max_tokens: max,
        eos_ids: vec![u32::MAX],
        readback: ReadbackArm::Blocking,
        stop_strings: stop.iter().map(|s| s.to_string()).collect(),
        ..DecodeParams::default()
    }
}

#[test]
fn a_refused_fragment_is_not_in_the_delivered_text() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::Cpu;
    let (model, cfg) = model(&dev)?;
    let fresh =
        |m: &Model| -> Result<_> { Ok(Sequence::new(m.new_cache(&cfg, &dev)?)) };

    // The reference: every fragment, in order.
    let mut s = DecodeSession::from_sequence(&model, fresh(&model)?, dev.clone());
    let mut all = Vec::new();
    let whole = s.decode_until(&P, detok, &params(6, &[]), |f| {
        all.push(f.to_string());
        true
    })?;
    assert_eq!(whole.text, all.concat(), "every fragment handed out");
    assert!(all.len() >= 4, "{all:?}");

    // The consumer refuses its third fragment: the client never had it.
    let mut s = DecodeSession::from_sequence(&model, fresh(&model)?, dev.clone());
    let mut taken = Vec::new();
    let cut = s.decode_until(&P, detok, &params(6, &[]), |f| {
        if taken.len() == 2 {
            return false;
        }
        taken.push(f.to_string());
        true
    })?;
    assert_eq!(cut.stop, Stop::Cancelled);
    assert_eq!(
        cut.text,
        taken.concat(),
        "the delivered text is what the consumer took, not the refused fragment"
    );
    assert!(
        cut.tokens.len() >= 3,
        "the ids are every id generated: {:?}",
        cut.tokens
    );

    // A stop string spanning two ids' text: the text ends where it begins.
    let stop = format!("{}><", &all[1][..all[1].len() - 1]);
    let mut s = DecodeSession::from_sequence(&model, fresh(&model)?, dev.clone());
    let stopped = s.decode_until(&P, detok, &params(6, &[stop.as_str()]), |_| true)?;
    assert_eq!(stopped.stop, Stop::StopString);
    let at = whole.text.find(&stop).unwrap();
    assert_eq!(stopped.text, whole.text[..at], "cut at the stop's first byte");
    Ok(())
}
