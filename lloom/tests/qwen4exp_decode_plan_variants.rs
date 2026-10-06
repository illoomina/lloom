//! Qwen4-exp's decode steps under the pass-plan cache (`DESIGN.md` §9.2aa):
//! one `Decode { b: 1 }` kind, several allocation sequences, each named by
//! the model's variant (`PassKey::variant`) so each is served its own plan.
//!
//! - A step that completes an indexer block (`(pos + 1) % ratio == 0`) runs
//!   the block's pooling, norm, rope and append; a plan recorded at another
//!   phase drifts on it (the "residual drift" #749 P28 measured, §16).
//! - A step whose routing is resolved on the device (the streamed tier's
//!   device plan) allocates the plan and its settle; a host-planned step
//!   reads each layer's ids back instead.
//!
//! Every stream here must be the plans-off stream, the served decode steps
//! must drift nowhere, and a warm streamed decode must ask the pools for
//! nothing while it is served.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident};
use lloom::nn::expert_stream::{ExpertResidency, ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKind, PassOpener};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The recorder's bind observer and the dispatch clock are process globals.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
/// Decode steps: positions 9..49, all inside the indexer's dense range
/// (`budget + ratio - 1` = 67), so the block phase is the only thing that
/// moves the sequence.
const STEPS: usize = 40;

/// The tiny fixture at a head dim the fused attention takes (64), so the
/// decode runs the kernels the checkpoint's does: the composed fallback
/// sizes its scores by the KV length, which no plan can serve.
fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    cfg.head_dim = 64;
    // The indexer's rope covers the attention's rotated dims (`n_rot` = 16).
    cfg.indexer_head_dim = 16;
    cfg.indexer_budget = 64;
    cfg.indexer_compress_ratio = 4;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

fn resident(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<CrownExact, Resident>> {
    testing::model_with_indexer::<CrownExact, _>(cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })
}

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-814-{}-{tag}.gguf", std::process::id()))
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

/// Every expert fits its layer's pool: after the first loads nothing
/// misses, so the device plan arms and stays armed.
fn streamed(cfg: &Config, dev: &Device, path: &Path) -> Result<Qwen4Exp<CrownExact, Streamed>> {
    let content = Content::read(&mut File::open(path)?)?;
    testing::model_with_indexer::<CrownExact, _>(cfg, dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            cfg.n_experts,
            LoaderArm::Pread,
            dev,
        )
    })
}

/// `STEPS` greedy ids for `PROMPT` through the driver (`Session::decode`),
/// which prepares and brackets every decode step.
fn decode<R: ExpertResidency>(model: &Qwen4Exp<CrownExact, R>, dev: &Device) -> Result<Vec<u32>> {
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

/// The driver's decode keys with a plan: `(variant, covered, total)`.
fn decode_plans(m: &MetalDevice) -> Vec<(u32, usize, usize)> {
    let (keys, _) = m.pass_plan_report().expect("the cache is on");
    let mut v: Vec<(u32, usize, usize)> = keys
        .into_iter()
        .filter(|(k, ..)| {
            matches!(k.kind, PassKind::Decode { b: 1 }) && k.opener == PassOpener::Driver
        })
        .map(|(k, covered, total, _)| (k.variant, covered, total))
        .collect();
    v.sort();
    v
}

/// Every block phase of a resident decode is served its own plan: no plan
/// is dropped for drifting, both phases are planned, and the stream is the
/// pool path's.
#[test]
fn every_indexer_block_phase_is_served_its_own_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = tiny();
    let want = {
        let dev = Device::new_metal(0)?;
        decode(&resident(&cfg, &dev)?, &dev)?
    };
    assert!(
        want.iter().any(|&t| t != want[0]),
        "a degenerate stream cannot tell the arms apart: {want:?}"
    );
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let model = resident(&cfg, &dev)?;
    let before = m.served_pass_pool_requests();
    let got = decode(&model, &dev)?;
    let after = m.served_pass_pool_requests();
    assert_eq!(got, want, "a served decode step moved the stream");
    let c = m.pass_plan_counters().expect("the cache is on");
    let plans = decode_plans(m);
    let requests = (after.0 - before.0, after.1 - before.1);
    eprintln!(
        "[resident] counters {c:?} decode plans (variant, covered, total) {plans:?} \
         served-pass requests {requests:?}"
    );
    for (pool, site, n) in lloom_tensor::metal_backend::served_pass_request_sites() {
        eprintln!("[resident] served-pass request site: {n} pool={pool} {site}");
    }
    assert_eq!(
        c.evicted, 0,
        "a decode plan drifted and was dropped: one key served a step of the other \
         block phase"
    );
    let variants: Vec<u32> = plans.iter().map(|p| p.0).collect();
    assert_eq!(
        variants,
        vec![0, 1],
        "one plan per block phase (bit 0), none for the device plan (a resident tier)"
    );
    assert!(
        c.served >= (STEPS - 8) as u64,
        "the decode was not served: {c:?}"
    );
    // A resident tier's routing stays on the device (one group of one row
    // a step), the rope of a completed block comes from the table: a served
    // step asks the pools for nothing.
    assert_eq!(
        requests,
        (0, 0),
        "a served pass asked the pools (shared, private)"
    );
    Ok(())
}

/// A warm streamed decode — its routing resolved on the device, nothing
/// missing — is served every step from its plans and asks the pools for
/// nothing: the routed GEMMs' outputs and the plan's record are the pass's
/// shape and the plan serves them (`DESIGN.md` §9.2aa, I1).
#[test]
fn a_warm_streamed_decode_is_served_and_asks_the_pools_for_nothing() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = tiny();
    let path = temp_gguf("warm");
    write_experts(&cfg, &path)?;
    let want = {
        let dev = Device::new_metal(0)?;
        decode(&streamed(&cfg, &dev, &path)?, &dev)?
    };
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let model = streamed(&cfg, &dev, &path)?;
    lloom_tensor::metal_backend::set_served_request_trace(true);
    let _ = lloom_tensor::metal_backend::served_pass_request_sites();
    let before = m.served_pass_pool_requests();
    let reads = lloom_tensor::metal_backend::readback_counts();
    let got = decode(&model, &dev)?;
    let reads = lloom_tensor::metal_backend::readback_counts().since(&reads);
    let after = m.served_pass_pool_requests();
    let sites = lloom_tensor::metal_backend::served_pass_request_sites();
    lloom_tensor::metal_backend::set_served_request_trace(false);
    let _ = std::fs::remove_file(&path);
    assert_eq!(got, want, "a served decode step moved the stream");
    // The settle reads each layer's record in place from its tier's slot
    // (#814 part 5): no staged copy, no blit, a slot read per layer a step.
    eprintln!("[streamed] readbacks {reads:?}");
    assert_eq!(
        reads.staged_readback, 0,
        "a record was copied to be read: {reads:?}"
    );
    assert_eq!(reads.blit, 0, "a readback blitted: {reads:?}");
    assert!(reads.slot > 0, "no record was read in place: {reads:?}");
    let c = m.pass_plan_counters().expect("the cache is on");
    let plans = decode_plans(m);
    eprintln!(
        "[streamed] counters {c:?} decode plans {plans:?} served-pass requests {:?}",
        (after.0 - before.0, after.1 - before.1)
    );
    for (pool, site, n) in &sites {
        eprintln!("[streamed] served-pass request site: {n} pool={pool} {site}");
    }
    assert_eq!(c.evicted, 0, "a decode plan drifted and was dropped");
    assert_eq!(
        c.diverged, 0,
        "a step missed and replayed: the pool is not warm"
    );
    assert!(
        plans.iter().any(|p| p.0 & 2 != 0),
        "no device-planned decode was planned: {plans:?}"
    );
    assert!(
        c.served >= (STEPS - 12) as u64,
        "the decode was not served: {c:?}"
    );
    // What a served pass may still ask for is the tier's residency table,
    // rebuilt (uploaded) after a settle that loaded: a value of the tier's,
    // outliving every pass, declared. Nothing from the private pool, and no
    // other site.
    assert_eq!(
        after.1 - before.1,
        0,
        "a served pass asked the private pool: {sites:?}"
    );
    let table: u64 = sites
        .iter()
        .filter(|(pool, site, _)| *pool == 0 && site.contains("::plan_device"))
        .map(|(_, _, n)| n)
        .sum();
    assert_eq!(
        after.0 - before.0,
        table,
        "a served pass asked the Shared pool for something other than the residency \
         table: {sites:?}"
    );
    Ok(())
}
