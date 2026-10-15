//! The request-scoped turn-checkpoint policy (`sequence-reuse-design.md`
//! §2.3b, `CheckpointRing::open_request`): every request states where its
//! passes leave their checkpoint — none, the prefill's end, or a position —
//! and nothing an earlier request asked carries over.
//!
//! One live `Session` per family, never reset, drives the request sequence
//! a chat server does: prepare the conversation, open the request's policy,
//! decode, close. The requests alternate a stripped turn (the checkpoint a
//! tail before the prompt's end) with a persisted one (at its end), tiled
//! and untiled, through an extend and a restore. After every request the
//! positions the state's checkpoints stand at are checked against the
//! positions the requests' own prompts name; the text is never read. The
//! regression is the stripped → persisted transition: through the hint
//! alone, the persisted request after a hinted one left no checkpoint, and
//! a tiled prefill on an unhinted state checkpointed every tile's end and
//! evicted the turns before it.
//!
//! The tiny Qwen3.5 (the generic decoder at the engine's shape: a standing
//! window and two checkpoint slots) and the tiny resident Qwen4-exp, on the
//! CPU and on Metal; a zero-slot capacity; and on Metal the pass-plan key:
//! requests of one shape under each policy, served from plans, compute what
//! they compute with the cache off.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::Config;
use lloom::models::qwen4exp::testing::resident_model;
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::{CheckpointPolicy, DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result};

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

/// `n` ids of a 64-id vocabulary span, distinct per `salt`.
fn ids(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 2 + (i * 7 + salt) % 60).collect()
}

/// What one request asks for, as a server's front end resolves it: a
/// stripped turn's checkpoint `tail` ids before the prompt's end, a
/// persisted turn's at its end, or none.
#[derive(Clone, Copy, Debug)]
enum Ask {
    Tail(usize),
    End,
    Keep0,
}

impl Ask {
    fn policy(self, pos: usize, end: usize) -> CheckpointPolicy {
        match self {
            Ask::Tail(t) => CheckpointPolicy::At(end.saturating_sub(t).max(pos)),
            Ask::End => CheckpointPolicy::End,
            Ask::Keep0 => CheckpointPolicy::None,
        }
    }
}

/// One request on the live session: `full` is the whole conversation, its
/// prefill tiled by `tile` (0: one forward), `max` greedy ids decoded. The
/// policy is opened after `prepare` (whose restore or reset comes first)
/// and closed when the run ends, failed or not. Returns the reuse action
/// and the absolute position the policy resolved to.
fn request<M: CausalLM>(
    s: &mut Session<'_, M>,
    model: &M,
    full: &[u32],
    ask: Ask,
    tile: usize,
    max: usize,
    dev: &Device,
) -> Result<(ReuseAction, Option<usize>)>
where
    M::Cache: CacheRewind,
{
    let p = s.prepare(full)?;
    let pos = s.pos();
    let end = pos + p.suffix.len();
    let policy = ask.policy(pos, end);
    model.open_checkpoint_request(s.cache_mut(), policy, end)?;
    let eos = vec![u32::MAX];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let run = s.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        &p.suffix,
        &DecodeOptions {
            prefill_tile: tile,
            ..DecodeOptions::new(max)
        },
        &mut |_: &[u32]| Ok(()),
    );
    model.close_checkpoint_request(s.cache_mut())?;
    run?;
    let at = match policy {
        CheckpointPolicy::None => None,
        CheckpointPolicy::End => Some(end),
        CheckpointPolicy::At(p) => Some(p),
    };
    Ok((p.action, at))
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

/// The live session's sequence on one family; `positions` reads the
/// state's checkpoint positions.
fn live_session<M: CausalLM>(
    model: &M,
    cache: M::Cache,
    positions: fn(&M::Cache) -> Vec<usize>,
    dev: &Device,
) -> Result<()>
where
    M::Cache: CacheRewind,
{
    let mut s = Session::new(model, cache);
    let at = |s: &mut Session<'_, M>| sorted(positions(s.cache_mut()));

    // 1. Stripped, untiled: two ids before the prompt's end.
    let p1 = ids(20, 1);
    let (a, e1) = request(&mut s, model, &p1, Ask::Tail(2), 0, 4, dev)?;
    assert_eq!(a, ReuseAction::Fresh);
    let e1 = e1.unwrap();
    assert_eq!(e1, 18);
    assert_eq!(at(&mut s), vec![e1], "1: the stripped request's checkpoint");

    // 2. Persisted, no reset: the prompt's end, whatever the request
    // before it asked.
    let full2 = [s.tokens().to_vec(), ids(5, 2)].concat();
    let (a, e2) = request(&mut s, model, &full2, Ask::End, 0, 4, dev)?;
    assert_eq!(a, ReuseAction::Extend, "2");
    let e2 = e2.unwrap();
    assert_eq!(e2, full2.len());
    assert_eq!(
        at(&mut s),
        vec![e1, e2],
        "2: stripped -> persisted leaves the persisted request its checkpoint"
    );

    // 3. Persisted, tiled by 4: one checkpoint, at the end, not one per
    // tile (which would evict the earlier turns').
    let full3 = [s.tokens().to_vec(), ids(13, 3)].concat();
    let (a, e3) = request(&mut s, model, &full3, Ask::End, 4, 4, dev)?;
    assert_eq!(a, ReuseAction::Extend, "3");
    let e3 = e3.unwrap();
    assert_eq!(
        at(&mut s),
        vec![e2, e3],
        "3: a tiled persisted request leaves one"
    );

    // 4. Stripped, tiled, the last reply re-rendered away: restored to the
    // checkpoint at the previous prompt's end, and this request's own
    // stands ten ids before its end — inside an earlier tile than the
    // last, so the tiles after it must take none.
    let full4 = [full3.clone(), ids(14, 4)].concat();
    let before = s.tokens().len();
    let (a, e4) = request(&mut s, model, &full4, Ask::Tail(10), 4, 4, dev)?;
    assert_eq!(
        a,
        ReuseAction::Rewound(before - e3),
        "4: restores the previous prompt's end"
    );
    let e4 = e4.unwrap();
    assert_eq!(e4, full4.len() - 10);
    assert_eq!(
        at(&mut s),
        vec![e3, e4],
        "4: the restored-to checkpoint survives, the oldest goes, and no tile after the target takes one"
    );

    // 5. A request that keeps none: the standing ones are untouched.
    let full5 = [s.tokens().to_vec(), ids(9, 5)].concat();
    let (a, _) = request(&mut s, model, &full5, Ask::Keep0, 4, 3, dev)?;
    assert_eq!(a, ReuseAction::Extend, "5");
    assert_eq!(
        at(&mut s),
        vec![e3, e4],
        "5: a request that keeps none takes none"
    );

    // 6. Persisted, untiled, after all of it: its end.
    let full6 = [s.tokens().to_vec(), ids(7, 6)].concat();
    let (a, e6) = request(&mut s, model, &full6, Ask::End, 0, 3, dev)?;
    assert_eq!(a, ReuseAction::Extend, "6");
    assert_eq!(at(&mut s), vec![e4, e6.unwrap()], "6: persisted again");

    // 7. After the last request closes, a caller that opens none sees a
    // fresh state's default: a multi-token pass checkpoints its end.
    let end7 = s.pos() + 6;
    let x = lloom_tensor::Tensor::from_vec(ids(6, 7), (1, 6), dev)?;
    let pos = s.pos();
    model.forward(&x, pos, s.cache_mut())?;
    assert_eq!(
        at(&mut s),
        vec![e6.unwrap(), end7],
        "7: closed, the cache's own default resumes"
    );
    Ok(())
}

fn qwen3_5(
    dev: &Device,
    slots: usize,
) -> Result<(lloom::models::qwen3_5::Qwen3_5<CrownExact>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        recovery: DecodeRecovery::Standing,
        checkpoints: slots,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn decoder_live(dev: &Device) -> Result<()> {
    let (model, cfg) = qwen3_5(dev, 2)?;
    let cache = model.new_cache(&cfg, dev)?;
    live_session(&model, cache, |c| c.checkpoint_positions(), dev)
}

fn qwen4exp_live(dev: &Device) -> Result<()> {
    let cfg = lloom::models::qwen4exp::Config::tiny();
    let mut model = resident_model::<CrownExact>(&cfg, dev);
    model.set_turn_checkpoints(2);
    let cache = model.new_cache(&cfg, dev)?;
    live_session(&model, cache, |c| c.checkpoint_positions(), dev)
}

/// No checkpoint slots: every policy takes nothing and every request runs.
fn zero_slots(dev: &Device) -> Result<()> {
    let (model, cfg) = qwen3_5(dev, 0)?;
    let mut s = Session::new(&model, model.new_cache(&cfg, dev)?);
    let p = ids(20, 9);
    request(&mut s, &model, &p, Ask::Tail(2), 0, 3, dev)?;
    let full = [s.tokens().to_vec(), ids(8, 10)].concat();
    request(&mut s, &model, &full, Ask::End, 4, 3, dev)?;
    assert!(
        s.cache_mut().checkpoint_positions().is_empty(),
        "no slots, no checkpoints"
    );
    Ok(())
}

#[test]
fn every_request_sets_its_own_checkpoint_policy_on_the_decoder_cpu() -> Result<()> {
    pin();
    decoder_live(&Device::Cpu)
}

#[test]
fn every_request_sets_its_own_checkpoint_policy_on_qwen4exp_cpu() -> Result<()> {
    pin();
    qwen4exp_live(&Device::Cpu)
}

#[test]
fn no_checkpoint_slots_keep_none_cpu() -> Result<()> {
    pin();
    zero_slots(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn every_request_sets_its_own_checkpoint_policy_on_the_decoder_metal() -> Result<()> {
    pin();
    decoder_live(&Device::new_metal(0)?)
}

#[cfg(feature = "metal")]
#[test]
fn every_request_sets_its_own_checkpoint_policy_on_qwen4exp_metal() -> Result<()> {
    pin();
    qwen4exp_live(&Device::new_metal(0)?)
}

#[cfg(feature = "metal")]
#[test]
fn no_checkpoint_slots_keep_none_metal() -> Result<()> {
    pin();
    zero_slots(&Device::new_metal(0)?)
}

/// Three rounds of four fresh requests of one length on `model`, with the
/// pass-plan cache on or off: the ring's own end checkpoint (no request),
/// `End`, a position two before the end, and `None` — each greedy. Returns
/// the streams, the plans served and the plans evicted for drift.
#[cfg(feature = "metal")]
fn policy_rounds<M: CausalLM>(
    model: &M,
    cfg: &M::Config,
    dev: &Device,
) -> Result<(Vec<Vec<u32>>, Option<(u64, u64)>)>
where
    M::Cache: CacheRewind,
{
    let Device::Metal(m) = dev else {
        unreachable!("a Metal device")
    };
    let p = ids(24, 3);
    let mut s = Session::new(model, model.new_cache(cfg, dev)?);
    let mut out = Vec::new();
    for _ in 0..3 {
        for ask in [None, Some(Ask::End), Some(Ask::Tail(2)), Some(Ask::Keep0)] {
            s.reset();
            match ask {
                Some(ask) => out.push({
                    request(&mut s, model, &p, ask, 0, 4, dev)?;
                    s.tokens()[p.len()..].to_vec()
                }),
                None => {
                    let eos = vec![u32::MAX];
                    let mut rb = Blocking::new(&eos, dev)?;
                    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                    let o = s.decode(
                        &mut IdentityProposer,
                        &mut rb,
                        &mut sampler,
                        &p,
                        &DecodeOptions::new(4),
                        &mut |_: &[u32]| Ok(()),
                    )?;
                    out.push(o.ids);
                }
            }
        }
    }
    Ok((out, m.pass_plan_counters().map(|c| (c.served, c.evicted))))
}

/// The pass-plan key follows the policy: an end checkpoint is one key
/// whoever asked for it (the ring's own or `End`), an interior one another,
/// none a third. Served from plans, every request computes what it computes
/// with the cache off, and no plan drifts — or a key named two computations.
#[cfg(feature = "metal")]
fn plans_follow_the_policy<M: CausalLM>(
    build: impl Fn(&Device) -> Result<(M, M::Config)>,
) -> Result<()>
where
    M::Cache: CacheRewind,
{
    use lloom_tensor::metal_backend::ArenaLayout;
    let off = Device::new_metal(0)?;
    let (model, cfg) = build(&off)?;
    let (want, _) = policy_rounds(&model, &cfg, &off)?;
    let on = Device::new_metal(0)?;
    let Device::Metal(m) = &on else {
        unreachable!("a Metal device")
    };
    m.enable_pass_plans(ArenaLayout::Packed);
    let (model, cfg) = build(&on)?;
    let (got, counters) = policy_rounds(&model, &cfg, &on)?;
    let (served, evicted) = counters.expect("the cache is on");
    eprintln!("POLICY-PLANS served {served} evicted {evicted}");
    assert!(served > 0, "no plan was served: the comparison is vacuous");
    assert_eq!(
        evicted, 0,
        "a plan drifted: one key served another computation"
    );
    assert_eq!(
        got, want,
        "a served pass computed something the pool pass did not"
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn the_plan_key_follows_the_policy_on_the_decoder() -> Result<()> {
    pin();
    plans_follow_the_policy(|dev| qwen3_5(dev, 2))
}

#[cfg(feature = "metal")]
#[test]
fn the_plan_key_follows_the_policy_on_qwen4exp() -> Result<()> {
    pin();
    plans_follow_the_policy(|dev| {
        let cfg = lloom::models::qwen4exp::Config::tiny();
        let mut model = resident_model::<CrownExact>(&cfg, dev);
        model.set_turn_checkpoints(2);
        Ok((model, cfg))
    })
}
