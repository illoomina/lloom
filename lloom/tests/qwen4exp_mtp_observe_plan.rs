//! The drafter's observe with the MTP head's own streamed experts planned on
//! the DEVICE (`QWEN4EXP_MTP_OBSERVE_PLAN=device`, `expert-streaming.md`
//! §20.16): no readback at the head's layer, its plan record settled after
//! the draft's argmax, a miss rewinding the head's cache and running the
//! rows again on the host plan. On a head pool a third of its experts the
//! observes miss and redo; the stream must still be greedy's, and the head's
//! lane must have been keyed with every accepted row exactly once.
//!
//! Its own test binary: the knob is read once per process.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, MtpDrafter, Qwen4Exp, Resident, State, Streamed};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::PathBuf;

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 12;
    cfg.n_experts_per_tok = 3;
    cfg.eos_token_id = 127;
    cfg
}

fn greedy(
    model: &Qwen4Exp<RewindableExact, Resident>,
    cfg: &Config,
    prompt: &[u32],
    max: usize,
) -> Result<Vec<u32>> {
    let dev = Device::Cpu;
    let cache = RewindableExact::single(&model, &dev, 8)?;
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &mut |_| Ok(()),
    )?;
    Ok(out.ids)
}

#[test]
fn the_observe_on_the_heads_device_plan_is_greedy_and_keys_every_row_once() -> Result<()> {
    std::env::set_var("QWEN4EXP_MTP_OBSERVE_PLAN", "device");
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model::<RewindableExact>(&cfg, &dev);
    let max = 24;
    let mut found = None;
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..5).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let ids = greedy(&model, &cfg, &prompt, max)?;
        if ids.len() >= 8 {
            found = Some((prompt, ids));
            break;
        }
    }
    let (prompt, ref_ids) = found.expect("a prompt with a stream of >= 8 tokens");

    // The head's experts streamed from a GGUF (layer 0's stacks, any plane
    // serves a synthetic head) through a 4-slot pool for 12 experts at
    // top-3: the observes' rows miss.
    let path: PathBuf =
        std::env::temp_dir().join(format!("lloom-mtp-observe-{}.gguf", std::process::id()));
    {
        let [g, u, d] = testing::expert_stacks(&cfg, 0, &dev, GgmlDType::F32);
        let refs: Vec<(&str, &QTensor)> = vec![
            ("blk.0.ffn_gate_exps.weight", &g),
            ("blk.0.ffn_up_exps.weight", &u),
            ("blk.0.ffn_down_exps.weight", &d),
        ];
        gguf_file::write(&mut File::create(&path)?, &[], &refs)?;
    }
    let content = Content::read(&mut File::open(&path)?)?;
    let src = |n: &str| ExpertSource::from_gguf(&content, &path, &format!("blk.0.{n}"));
    for k in [1usize, 2] {
        let experts = Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            4,
            LoaderArm::Pread,
            &dev,
        )?;
        let head = testing::mtp_head::<CrownExact, _>(&cfg, &dev, experts);
        let mut mtp =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let mut cache: State<RewindableExact> =
            State::<RewindableExact>::in_ring(&cfg, &model.spec_pools(1, k, &dev))?;
        cache.want_nextn = true;
        let mut session = Session::new(&model, cache);
        let eos = [cfg.eos_token_id];
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
        assert_eq!(out.ids, ref_ids, "k = {k}: not the greedy stream");
        let (device, host, redos) = mtp.observe_plans();
        let stats = head.layer().moe().experts().stats();
        println!(
            "MTPOBSERVE k={k} device={device} host={host} redos={redos} misses={} evictions={} rows_seen={} pos={}",
            stats.misses,
            stats.evictions,
            mtp.rows_seen(),
            mtp.cache().pos
        );
        // Non-vacuity: the device plan ran, and missed (the redo path ran).
        assert!(device > 0, "k = {k}: no observe was planned on the device");
        assert!(
            redos > 0,
            "k = {k}: no device-planned observe missed on a 4-slot pool"
        );
        assert!(
            stats.misses > 0 && stats.evictions > 0,
            "k = {k}: {stats:?}"
        );
        // Every accepted row keyed once: the head's cache holds exactly the
        // rows the drafter saw (a redo rewound before running them again).
        assert_eq!(
            mtp.cache().pos,
            mtp.rows_seen(),
            "k = {k}: the head's cache holds rows a redo did not rewind"
        );
        assert!(
            mtp.rows_seen() >= prompt.len() + ref_ids.len() - 1,
            "k = {k}: keyed with {} rows for {} prompt + {} generated",
            mtp.rows_seen(),
            prompt.len(),
            ref_ids.len()
        );
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
