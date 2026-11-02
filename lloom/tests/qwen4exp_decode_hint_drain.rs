//! The decode hint's deferred stage (`stage_readback_deferred`) is finished
//! after the layer's own host-plan routing readback has drained the queue
//! past it: on the streamed tier every host-planned layer reads its routing
//! back (`Streamed::plan`, `Streamed::plan_window`), so no hint's `finish`
//! has to submit its own recording, and none adds a cut.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom_tensor::metal_backend::readback_counts;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::Path;

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 24;
const SLOTS: usize = 3;

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    cfg.head_dim = 64;
    cfg.indexer_head_dim = 16;
    cfg.indexer_budget = 64;
    cfg.indexer_compress_ratio = 4;
    cfg
}

fn write_experts(cfg: &Config, path: &Path) -> Result<()> {
    let cpu = Device::Cpu;
    let mut owned = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &cpu, GgmlDType::F32);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let refs: Vec<(&str, &_)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    Ok(())
}

fn streamed(cfg: &Config, dev: &Device, path: &Path) -> Result<Qwen4Exp<CrownExact, Streamed>> {
    let content = Content::read(&mut File::open(path)?)?;
    testing::model_with_indexer::<CrownExact, _>(cfg, dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            SLOTS,
            LoaderArm::Pread,
            dev,
        )
    })
}

#[test]
fn every_decode_hint_is_finished_after_its_layers_drain() -> Result<()> {
    let mut t = lloom::tuning::Tuning::default();
    t.stream.plan = "host".into();
    t.stream.decode_hint = true;
    let _ = lloom::tuning::install(t);
    assert!(
        lloom::tuning::tuning().stream.decode_hint,
        "the hint is not on"
    );
    let cfg = tiny();
    let path = std::env::temp_dir().join(format!("lloom-944-{}-hint.gguf", std::process::id()));
    write_experts(&cfg, &path)?;
    let dev = Device::new_metal(0)?;
    let model = streamed(&cfg, &dev, &path)?;
    let hints = Qwen4Exp::<CrownExact, Streamed>::decode_hints();
    let before = readback_counts();
    let ids = {
        let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&[u32::MAX], &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(STEPS);
        let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
        session
            .decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
            .ids
    };
    let after = readback_counts().since(&before);
    let hints = Qwen4Exp::<CrownExact, Streamed>::decode_hints() - hints;
    let _ = std::fs::remove_file(&path);
    eprintln!(
        "DECODE_HINT_DRAIN ids={} hints={hints} counts={after:?}",
        ids.len()
    );
    assert!(hints > 0, "no decode hint landed: the arm did not engage");
    assert_eq!(
        after.deferred_undrained, 0,
        "a decode hint was finished before its layer drained the queue"
    );
    Ok(())
}
