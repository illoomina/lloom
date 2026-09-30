//! Qwen4-exp's streamed decode on the HOST plan under the pass-plan cache
//! (`DESIGN.md` §9.2aa, `docs/pass-plan-design.md` §19): the regime
//! `lloom chat` runs at `--slots auto`, where the tier misses every token
//! and the device plan never arms. Each MoE layer reads its routing back and
//! uploads the one-row group's slot ids. Both round trips go through buffers
//! the tier owns (`stage_readback_into` into its destination; the slot ids
//! through its buffer, epoch-gated), so a served host-planned pass asks the
//! pools for nothing, and the stream is the unplanned one.
//!
//! Its own binary: the host plan is pinned through the process-wide tuning
//! (`QWEN4EXP_PLAN=host`), which the device-plan tests must not see.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::expert_stream::{ExpertResidency, ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKind, PassOpener};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// The tests share the device's global recorder and the tuning pin.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn pin_host_plan() {
    let mut t = lloom::tuning::Tuning::default();
    t.stream.plan = "host".into();
    let _ = lloom::tuning::install(t);
    assert_eq!(
        lloom::tuning::tuning().stream.plan,
        "host",
        "the host plan is not pinned"
    );
}

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 40;
/// Fewer slots than experts: the tier misses, as it does at `--slots auto`.
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

fn streamed(cfg: &Config, dev: &Device, path: &Path) -> Result<Qwen4Exp<Streamed>> {
    let content = Content::read(&mut File::open(path)?)?;
    testing::model_with_indexer(cfg, dev, |l| {
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

fn resident(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<Resident>> {
    testing::model_with_indexer(cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })
}

fn decode<R: ExpertResidency>(model: &Qwen4Exp<R>, dev: &Device) -> Result<Vec<u32>> {
    let mut session = Session::new(model, model.new_cache(&tiny(), dev)?);
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&[u32::MAX], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(STEPS);
    let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
    Ok(session
        .decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
        .ids)
}

#[test]
fn a_missing_host_planned_decode_is_served_and_asks_the_pools_for_nothing() -> Result<()> {
    let _serial = SERIAL.lock();
    pin_host_plan();
    let cfg = tiny();
    let path = std::env::temp_dir().join(format!("lloom-814-{}-host.gguf", std::process::id()));
    write_experts(&cfg, &path)?;
    let want = {
        let dev = Device::new_metal(0)?;
        decode(&streamed(&cfg, &dev, &path)?, &dev)?
    };
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("a Metal device")
    };
    let m: &MetalDevice = m;
    m.enable_pass_plans(ArenaLayout::Packed);
    let model = streamed(&cfg, &dev, &path)?;
    lloom_tensor::metal_backend::set_served_request_trace(true);
    let _ = lloom_tensor::metal_backend::served_pass_request_sites();
    let before = m.served_pass_pool_requests();
    let got = decode(&model, &dev)?;
    let after = m.served_pass_pool_requests();
    let sites = lloom_tensor::metal_backend::served_pass_request_sites();
    lloom_tensor::metal_backend::set_served_request_trace(false);
    let _ = std::fs::remove_file(&path);
    assert_eq!(got, want, "a served decode step moved the stream");
    let c = m.pass_plan_counters().expect("the cache is on");
    let (keys, _) = m.pass_plan_report().expect("the cache is on");
    let plans: Vec<(u32, usize, usize)> = keys
        .into_iter()
        .filter(|(k, ..)| {
            matches!(k.kind, PassKind::Decode { b: 1 }) && k.opener == PassOpener::Driver
        })
        .map(|(k, covered, total, _)| (k.variant, covered, total))
        .collect();
    eprintln!(
        "[host plan] counters {c:?} decode plans {plans:?} served-pass requests {:?}",
        (after.0 - before.0, after.1 - before.1)
    );
    for (pool, site, n) in &sites {
        eprintln!("[host plan] served-pass request site: {n} pool={pool} {site}");
    }
    assert!(
        plans.iter().all(|p| p.0 & 2 == 0),
        "a decode step ran the device plan: {plans:?}"
    );
    assert!(
        plans.iter().any(|p| p.1 == p.2 && p.2 > 0),
        "no host-planned decode was planned whole: {plans:?}"
    );
    assert_eq!(c.evicted, 0, "a decode plan drifted and was dropped");
    assert!(
        c.served >= (STEPS - 12) as u64,
        "the decode was not served: {c:?}"
    );
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (0, 0),
        "a served host-planned pass asked the pools: {sites:?}"
    );
    Ok(())
}

/// The routing ids of a one-row step are written by the router straight into
/// the tier's host-read slot and read there, in place (#814 part 5): every
/// decode step's MoE layers read the slot, none copies into a staging
/// destination, and the stream is the `Resident` tier's, which reads nothing
/// back — so a router that wrote anywhere but the slot the plan reads would
/// route on stale ids and move it.
#[test]
fn the_routing_ids_are_read_in_place_from_the_tiers_slot() -> Result<()> {
    let _serial = SERIAL.lock();
    pin_host_plan();
    let cfg = tiny();
    let want = {
        let dev = Device::new_metal(0)?;
        decode(&resident(&cfg, &dev)?, &dev)?
    };
    let path = std::env::temp_dir().join(format!("lloom-814-{}-slot.gguf", std::process::id()));
    write_experts(&cfg, &path)?;
    let dev = Device::new_metal(0)?;
    let model = streamed(&cfg, &dev, &path)?;
    let before = lloom_tensor::metal_backend::readback_counts();
    let got = decode(&model, &dev)?;
    let r = lloom_tensor::metal_backend::readback_counts().since(&before);
    let _ = std::fs::remove_file(&path);
    eprintln!("[ids slot] readbacks over {STEPS} steps: {r:?}");
    assert_eq!(got, want, "the host-planned stream is not the resident one");
    assert_eq!(
        r.staged_readback, 0,
        "a one-row step copied its ids into a staging destination: {r:?}"
    );
    assert!(
        r.slot >= STEPS as u64,
        "the decode steps did not read the ids slot: {r:?}"
    );
    assert_eq!(r.blit, 0, "a readback blitted: {r:?}");
    Ok(())
}
