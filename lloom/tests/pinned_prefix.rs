//! The pinned prefix (`sessions.md` §8): a request leaves, in the pass it
//! already runs, a turn checkpoint at the session's head beside its own and
//! pins it, so that a later request which keeps the head — a compaction that
//! truncates the conversation to its system prompt — restores there instead
//! of prefilling the conversation again.
//!
//! Three things are pinned here, on the tiny Qwen3.5 (the generic decoder's
//! Gated DeltaNet and KV layers) and the tiny LFM2 (its own ring, with reach):
//! - the pass that takes the second target computes what the pass that takes
//!   only the request's own computes — the logits bit for bit, the state it
//!   leaves, the request's own checkpoint — so no digest of a first turn
//!   moves;
//! - the head's checkpoint is the state a prefill of exactly the head leaves;
//! - the pin outlives more turns than the ring has slots, and a compaction
//!   restores it (without the pin the same request prefills from nothing).
//!
//! Every reference is another mechanism than the subject (CONTRIBUTING
//! §3.1b): a pass without the second target, a fresh prefill of the head,
//! the same session without a pin frame.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{
    Config as Lfm2Config, ConvState, KvAppend, LayerType, Model as Lfm2, Shape as Lfm2Shape,
    State as Lfm2State,
};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::Config;
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{
    DecodeRecovery, DecoderState, HybridState, Layer, PreNormState, StateCapacity,
};
use lloom::nn::model::{CacheReset, CacheRewind, CausalLM};
use lloom::nn::{
    CheckpointPolicy, DecodeOptions, IdentityProposer, ReuseAction, Session, VarBuilder,
};
use lloom::profile::{CrownExact, Profile};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;

fn pin_regime() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn tokens(n: usize, vocab: usize, seed: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 7 + seed) % vocab as u32)
        .collect()
}

fn f32s(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

/// The largest difference between two tensors of one shape, reduced on
/// their device (a real model's state is too large to read back whole).
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    assert_eq!(a.dims(), b.dims(), "shapes differ");
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

/// The SHA-256 of a tensor's values, as the gates digest logits.
fn digest(t: &Tensor) -> Result<String> {
    let mut h = Sha256::new();
    for v in f32s(t)? {
        h.update(v.to_le_bytes());
    }
    Ok(format!("{:x}", h.finalize()))
}

fn logits<M: CausalLM>(
    model: &M,
    st: &mut M::Cache,
    pos: usize,
    toks: &[u32],
    dev: &Device,
) -> Result<Tensor> {
    let input = Tensor::from_vec(toks.to_vec(), (1, toks.len()), dev)?;
    model.forward(&input, pos, st)
}

// ---- the generic decoder (Qwen3.5) -------------------------------------------

/// Every layer of `a` against `b`: the largest difference over the recurrent
/// layers' `S` and windows and over the attention layers' live KV rows.
fn state_diff<L: Layer<State = LayerState>>(
    a: &DecoderState<L>,
    b: &DecoderState<L>,
) -> Result<(f32, f32)> {
    assert_eq!(a.pos, b.pos, "position");
    let (mut rec, mut kv) = (0f32, 0f32);
    for (l, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        match (&x.mixer, &y.mixer) {
            (HybridState::A(p), HybridState::A(q)) => {
                for (what, s, t) in [
                    ("S", p.s.get(p.layer), q.s.get(q.layer)),
                    ("conv", p.conv.get(p.layer), q.conv.get(q.layer)),
                ] {
                    let (Some(s), Some(t)) = (s, t) else {
                        panic!("layer {l}: {what} written on one side only");
                    };
                    rec = rec.max(max_abs_diff(s, t)?);
                }
            }
            (HybridState::B(p), HybridState::B(q)) => {
                assert_eq!(p.len(), q.len(), "layer {l}: KV length");
                let n = p.len();
                let [pk, pv, _, _] = p.state_views()?;
                let [qk, qv, _, _] = q.state_views()?;
                for (s, t) in [(pk, qk), (pv, qv)] {
                    let (Some(s), Some(t)) = (s, t) else { continue };
                    if s.rank() == 3 && t.rank() == 3 && s.dim(1)? >= n && t.dim(1)? >= n {
                        kv = kv.max(max_abs_diff(&s.narrow(1, 0, n)?, &t.narrow(1, 0, n)?)?);
                    }
                }
            }
            _ => panic!("layer {l}: the two states are of different layer kinds"),
        }
    }
    Ok((rec, kv))
}

fn dense<P: Profile>(
    dev: &Device,
    pin: bool,
) -> Result<(lloom::models::qwen3_5::Qwen3_5<P>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        checkpoints: 2,
        pin,
        recovery: DecodeRecovery::Standing,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

/// One prefill of `tiles`' positions under the request's own `policy`, and
/// with the head's checkpoint pinned beside it when `head` is given.
/// Returns the state and the last tile's logits.
fn first_turn<M>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    tiles: &[usize],
    policy: CheckpointPolicy,
    head: Option<usize>,
) -> Result<(M::Cache, Tensor)>
where
    M: CausalLM<Config = Config>,
{
    let n: usize = tiles.iter().sum();
    let mut st = model.new_cache(cfg, dev)?;
    model.open_checkpoint_request(&mut st, policy, n)?;
    if let Some(h) = head {
        assert_eq!(
            model.pin_checkpoint(&mut st, Some(h))?,
            None,
            "nothing stands at the head before the pass"
        );
    }
    let mut at = 0;
    let mut last = None;
    for &len in tiles {
        last = Some(logits(model, &mut st, at, &prompt[at..at + len], dev)?);
        at += len;
    }
    model.close_checkpoint_request(&mut st)?;
    Ok((st, last.expect("one tile at least")))
}

/// The first turn with the head's checkpoint and without it: the same
/// logits, bit for bit and by digest; the same state after the pass; the
/// request's own checkpoint where it was and restoring to the same state;
/// the same continuation. And the head's checkpoint, pinned, is the state
/// the request's own checkpoint leaves when the request asks for it at the
/// head instead, over the same tiles (`deterministic` holds within a pass
/// split, not across one: a fresh one-pass prefill of the head is printed).
fn two_targets_case<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    tiles: &[usize],
    head: usize,
    tail: usize,
) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
{
    let n: usize = tiles.iter().sum();
    let own = n - tail;
    let policy = match tail {
        0 => CheckpointPolicy::End,
        t => CheckpointPolicy::At(n - t),
    };
    let prompt = tokens(n + 8, cfg.vocab_size, 3);
    let (mut one, la) = first_turn(model, cfg, dev, &prompt, tiles, policy, None)?;
    let (mut two, lb) = first_turn(model, cfg, dev, &prompt, tiles, policy, Some(head))?;
    let tag = format!("tiles {tiles:?} head {head} tail {tail}");
    let (da, db) = (digest(&la)?, digest(&lb)?);
    eprintln!("PIN-TWO {tag}: logits sha256 one-target {da} two-target {db}");
    assert_eq!(
        da, db,
        "{tag}: the pass's logits moved with the head's checkpoint"
    );
    assert_eq!(max_abs_diff(&la, &lb)?, 0.0);
    assert_eq!(
        state_diff(&one, &two)?,
        (0.0, 0.0),
        "{tag}: the state the pass leaves moved"
    );
    assert_eq!(one.checkpoint_positions(), vec![own]);
    assert_eq!(sorted(two.checkpoint_positions()), vec![head, own]);
    assert_eq!(
        two.pinned_checkpoint(),
        Some(head),
        "{tag}: pinned when taken"
    );
    // The request's own checkpoint: the same state on both.
    one.restore(own)?;
    two.restore(own)?;
    assert_eq!(
        state_diff(&one, &two)?,
        (0.0, 0.0),
        "{tag}: the request's own checkpoint"
    );
    assert_eq!(
        two.pinned_checkpoint(),
        Some(head),
        "a restore above the pin keeps it"
    );
    let ca = logits(model, &mut one, own, &prompt[own..n + 4], dev)?;
    let cb = logits(model, &mut two, own, &prompt[own..n + 4], dev)?;
    assert_eq!(max_abs_diff(&ca, &cb)?, 0.0, "{tag}: the continuation");
    // The head's checkpoint against the request's own one there.
    two.restore(head)?;
    assert_eq!(
        two.pinned_checkpoint(),
        Some(head),
        "a restore at the pin keeps it"
    );
    let (mut same, _) = first_turn(
        model,
        cfg,
        dev,
        &prompt,
        tiles,
        CheckpointPolicy::At(head),
        None,
    )?;
    same.restore(head)?;
    assert_eq!(
        state_diff(&two, &same)?,
        (0.0, 0.0),
        "{tag}: the pinned head is the state a checkpoint asked at the head leaves"
    );
    let ha = logits(model, &mut two, head, &prompt[head..head + 6], dev)?;
    let hb = logits(model, &mut same, head, &prompt[head..head + 6], dev)?;
    assert_eq!(
        max_abs_diff(&ha, &hb)?,
        0.0,
        "{tag}: the continuation from the head"
    );
    let mut fresh = model.new_cache(cfg, dev)?;
    logits(model, &mut fresh, 0, &prompt[..head], dev)?;
    two.restore(head)?;
    let (rec, kv) = state_diff(&two, &fresh)?;
    eprintln!(
        "PIN-TWO {tag}: head restore vs a fresh one-pass prefill of the head (another split, \
         printed): recurrent {rec:e} kv {kv:e}"
    );
    Ok(())
}

/// The head and the request's own in one chunk, in two chunks of one pass,
/// in two tiles, and the request's own inside the pass (a think tail).
fn two_targets_cases(dev: &Device, heads: [usize; 2]) -> Result<()> {
    let (mut model, cfg) = dense::<CrownExact>(dev, true)?;
    for (tiles, head, tail) in [
        (vec![70], heads[0], 0),
        (vec![70], heads[0], 2),
        (vec![70], heads[1], 4),
        (vec![40, 30], heads[0], 0),
    ] {
        two_targets_case(&model, &cfg, dev, &tiles, head, tail)?;
    }
    model.set_prefill_chunk(16);
    two_targets_case(&model, &cfg, dev, &[70], heads[0], 2)?;
    Ok(())
}

#[test]
fn the_two_target_pass_computes_what_the_one_target_pass_computes_on_the_cpu() -> Result<()> {
    pin_regime();
    two_targets_cases(&Device::Cpu, [20, 9])
}

#[cfg(feature = "metal")]
#[test]
fn the_two_target_pass_computes_what_the_one_target_pass_computes_on_metal() -> Result<()> {
    pin_regime();
    two_targets_cases(&Device::new_metal(0)?, [50, 9])
}

fn greedy_decode<M: CausalLM>(
    session: &mut Session<'_, M>,
    prompt: &[u32],
    n: usize,
    dev: &Device,
) -> Result<Vec<u32>>
where
    M::Cache: CacheRewind,
{
    let eos = vec![u32::MAX];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(n),
        &mut |_: &[u32]| Ok(()),
    )?;
    Ok(out.ids)
}

/// A request as the engine runs it (`DecodeSession`): prepare, open the
/// request's policy (its end), pin the head where asked, decode, close.
fn request<M: CausalLM>(
    session: &mut Session<'_, M>,
    full: &[u32],
    head: Option<usize>,
    n: usize,
    dev: &Device,
) -> Result<(lloom::nn::Prepared, Vec<u32>)>
where
    M::Cache: CacheRewind + CacheReset,
{
    request_with(session, full, None, head, n, dev)
}

/// [`request`] whose own checkpoint stands at `at` instead of its end.
fn request_with<M: CausalLM>(
    session: &mut Session<'_, M>,
    full: &[u32],
    at: Option<usize>,
    head: Option<usize>,
    n: usize,
    dev: &Device,
) -> Result<(lloom::nn::Prepared, Vec<u32>)>
where
    M::Cache: CacheRewind + CacheReset,
{
    let prepared = session.prepare(full)?;
    let model = session.model();
    let end = session.pos() + prepared.suffix.len();
    let policy = at.map_or(CheckpointPolicy::End, CheckpointPolicy::At);
    model.open_checkpoint_request(session.cache_mut(), policy, end)?;
    if let Some(h) = head {
        model.pin_checkpoint(session.cache_mut(), Some(h))?;
    }
    let out = greedy_decode(session, &prepared.suffix, n, dev);
    model.close_checkpoint_request(session.cache_mut())?;
    Ok((prepared, out?))
}

/// A conversation of `turns` extending turns on one session (a system
/// prompt of `head` ids, then per turn a user message and a reply), then a
/// compaction: the system prompt, a summary, a new turn. Returns the
/// session's preparation of the compaction and the state it left.
fn conversation<'m, M>(
    model: &'m M,
    cfg: &Config,
    dev: &Device,
    head: usize,
    turns: usize,
    pin: bool,
) -> Result<(lloom::nn::Prepared, Session<'m, M>, Vec<u32>)>
where
    M: CausalLM<
        Config = Config,
        Cache = DecoderState<lloom::models::qwen3_5::Qwen3_5Layer<CrownExact>>,
    >,
{
    let sys = tokens(head, cfg.vocab_size, 3);
    let mut session = Session::new(model, model.new_cache(cfg, dev)?);
    let mut full = sys.clone();
    for k in 0..turns {
        full.extend(tokens(5, cfg.vocab_size, 11 + k as u32));
        let (p, reply) = request(&mut session, &full, pin.then_some(head), 4, dev)?;
        if k > 0 {
            assert_eq!(p.action, ReuseAction::Extend, "turn {k} extends");
        }
        full.extend(reply);
        if pin {
            assert_eq!(
                session.cache_mut().pinned_checkpoint(),
                Some(head),
                "turn {k}: the head stays pinned"
            );
        }
    }
    let ckpts = sorted(session.cache_mut().checkpoint_positions());
    eprintln!("PIN-CONV head {head} pin {pin}: after {turns} turns, checkpoints at {ckpts:?}");
    let mut compact = sys.clone();
    let summary_first = (sys[head - 1] + 1 + full[head]) % cfg.vocab_size as u32;
    compact.push(summary_first);
    compact.extend(tokens(9, cfg.vocab_size, 29));
    assert_ne!(
        compact[head], full[head],
        "the compaction leaves the log at the head"
    );
    let prepared = session.prepare(&compact)?;
    Ok((prepared, session, compact))
}

/// The pin outlives more turns than the ring has slots, and the compaction
/// restores it: `Rewound` to the head, the state a prefill of the head
/// leaves, and the same continuation. The control: the same conversation
/// with no pin frame prefills the compaction from nothing.
fn compaction_case(dev: &Device, head: usize) -> Result<()> {
    let (model, cfg) = dense::<CrownExact>(dev, true)?;
    let turns = 5;
    let (prepared, mut session, compact) = conversation(&model, &cfg, dev, head, turns, true)?;
    eprintln!(
        "PIN-CONV head {head}: the compaction prepared {:?}, reused {}, suffix {}",
        prepared.action,
        prepared.reused,
        prepared.suffix.len()
    );
    assert!(
        matches!(prepared.action, ReuseAction::Rewound(_)),
        "the compaction restores the pinned head: {:?}",
        prepared.action
    );
    assert_eq!(prepared.reused, head);
    assert_eq!(prepared.suffix, compact[head..]);
    assert_eq!(session.cache_mut().pinned_checkpoint(), Some(head));
    // The reference: the first turn run the same way (the same passes),
    // its own checkpoint asked at the head, restored there.
    let mut turn1 = compact[..head].to_vec();
    turn1.extend(tokens(5, cfg.vocab_size, 11));
    let mut rs = Session::new(&model, model.new_cache(&cfg, dev)?);
    request_with(&mut rs, &turn1, Some(head), None, 4, dev)?;
    let same = rs.cache_mut();
    same.restore(head)?;
    assert_eq!(
        state_diff(session.cache_mut(), same)?,
        (0.0, 0.0),
        "the restored head is the first turn's state there"
    );
    let a = logits(&model, session.cache_mut(), head, &compact[head..], dev)?;
    let b = logits(&model, same, head, &compact[head..], dev)?;
    assert_eq!(
        max_abs_diff(&a, &b)?,
        0.0,
        "the compaction's prefill from the restored head"
    );
    let mut fresh = model.new_cache(&cfg, dev)?;
    logits(&model, &mut fresh, 0, &compact[..head], dev)?;
    same.restore(head)?;
    let (rec, kv) = state_diff(same, &fresh)?;
    eprintln!(
        "PIN-CONV head {head}: the head's state vs a fresh one-pass prefill of it (another \
         split, printed): recurrent {rec:e} kv {kv:e}"
    );
    // Without a pin frame the same conversation keeps nothing at the head.
    let (model, cfg) = dense::<CrownExact>(dev, false)?;
    let (control, _, _) = conversation(&model, &cfg, dev, head, turns, false)?;
    assert_eq!(
        control.action,
        ReuseAction::Fresh,
        "the control: without the pin the compaction prefills from nothing"
    );
    Ok(())
}

#[test]
fn a_pin_survives_more_turns_than_the_ring_has_slots_and_a_compaction_restores_it() -> Result<()> {
    pin_regime();
    compaction_case(&Device::Cpu, 20)
}

#[cfg(feature = "metal")]
#[test]
fn a_pin_survives_more_turns_than_the_ring_has_slots_and_a_compaction_restores_it_on_metal(
) -> Result<()> {
    pin_regime();
    compaction_case(&Device::new_metal(0)?, 50)
}

/// Below the pin it is released: a request that leaves the log before the
/// head restores an earlier checkpoint or resets, and the pin goes with
/// what the state no longer holds. A reset releases it.
#[test]
fn a_restore_below_the_pin_or_a_reset_releases_it() -> Result<()> {
    pin_regime();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, true)?;
    let head = 20;
    let prompt = tokens(40, cfg.vocab_size, 3);
    // A checkpoint at 10 (the request's own, inside the pass) and the head's.
    let mut st = model.new_cache(&cfg, &dev)?;
    model.open_checkpoint_request(&mut st, CheckpointPolicy::At(10), 40)?;
    model.pin_checkpoint(&mut st, Some(head))?;
    logits(&model, &mut st, 0, &prompt, &dev)?;
    model.close_checkpoint_request(&mut st)?;
    assert_eq!(sorted(st.checkpoint_positions()), vec![10, head]);
    assert_eq!(st.pinned_checkpoint(), Some(head));
    st.restore(10)?;
    assert_eq!(
        st.pinned_checkpoint(),
        None,
        "a restore below the pin releases it"
    );
    assert_eq!(st.checkpoint_positions(), vec![10]);
    // Through a session: a request diverging before the head resets.
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let full: Vec<u32> = tokens(30, cfg.vocab_size, 3);
    request(&mut session, &full, Some(head), 3, &dev)?;
    assert_eq!(session.cache_mut().pinned_checkpoint(), Some(head));
    let mut other = full[..5].to_vec();
    other.extend(tokens(12, cfg.vocab_size, 40));
    let (p, _) = request(&mut session, &other, None, 3, &dev)?;
    assert_eq!(p.action, ReuseAction::Fresh);
    assert_eq!(
        session.cache_mut().pinned_checkpoint(),
        None,
        "a reset releases it"
    );
    // Releasing on purpose (a session's `reset: prefix`).
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    request(&mut session, &full, Some(head), 3, &dev)?;
    assert_eq!(
        model.pin_checkpoint(session.cache_mut(), None)?,
        None,
        "unpinned"
    );
    assert!(
        !sorted(session.cache_mut().checkpoint_positions()).contains(&head),
        "the released pin is not a checkpoint any more"
    );
    Ok(())
}

/// Each request's reply, and the plans served and evicted.
#[cfg(feature = "metal")]
type Rounds = (Vec<Vec<u32>>, Option<(u64, u64)>);

/// One process's requests on the tiny dense model with the pass-plan cache
/// on or off: rounds of fresh requests of one length, one leaving only its
/// own checkpoint at its end, one leaving the head's beside it, and one
/// leaving its own at the head's position alone (the second recurrence
/// without the end's copy). With the cache on, the plans served and the
/// drift evictions; else `None`.
#[cfg(feature = "metal")]
fn mixed_rounds(plans: bool) -> Result<Rounds> {
    use lloom_tensor::metal_backend::ArenaLayout;
    let dev = Device::new_metal(0)?;
    let Device::Metal(m) = &dev else {
        unreachable!("a Metal device")
    };
    if plans {
        m.enable_pass_plans(ArenaLayout::Packed);
    }
    let (model, cfg) = dense::<CrownExact>(&dev, true)?;
    let prompt = tokens(60, cfg.vocab_size, 5);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let mut out = Vec::new();
    for _ in 0..3 {
        for (at, head) in [(None, None), (None, Some(30)), (Some(30), None)] {
            session.reset();
            out.push(request_with(&mut session, &prompt, at, head, 4, &dev)?.1);
        }
    }
    Ok((out, m.pass_plan_counters().map(|c| (c.served, c.evicted))))
}

/// A pass that leaves the head's checkpoint beside its own runs another
/// allocation sequence than one that leaves its own alone (the second
/// recurrence over the head), so each is served its own plan
/// (`CheckpointRing::pass_variant`): the tokens with the cache on are the
/// cache-off run's, plans were served, and no plan drifted.
#[cfg(feature = "metal")]
#[test]
fn a_pass_with_the_heads_checkpoint_is_served_its_own_plan() -> Result<()> {
    pin_regime();
    let (want, _) = mixed_rounds(false)?;
    let (got, counters) = mixed_rounds(true)?;
    eprintln!("PIN-PLANS served/evicted {counters:?}");
    let (served, evicted) = counters.expect("the cache was on");
    assert!(served > 0, "no plan was served: the comparison is vacuous");
    assert_eq!(
        evicted, 0,
        "a plan drifted under a pass it was not recorded from"
    );
    assert_eq!(
        got, want,
        "a served pass computed something the pool pass did not"
    );
    Ok(())
}

// ---- LFM2 (its own ring, with reach) -----------------------------------------

const L_HIDDEN: usize = 64;
const L_VOCAB: usize = 48;

fn lfm2_config() -> Lfm2Config {
    Lfm2Config {
        vocab_size: L_VOCAB,
        hidden_size: L_HIDDEN,
        intermediate_size: 128,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 512,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(7),
        use_flash_attn: false,
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::Shuffle,
        // The served presets carry the left context (`sequence-reuse-design.md`
        // §2.5): a turn after a restore is then the pass split it is.
        conv_prefill_ctx: true,
        gemv_silu: false,
        gemv_residual: false,
        tap_dtype: DType::F32,
        tap_layers: Vec::new(),
        ..Default::default()
    }
}

/// Fixed-ramp weights (`turn_checkpoint.rs`'s): deterministic, so two states
/// over one model agree exactly and a difference is the paths'.
fn lfm2_build<P: Lfm2Shape>(cfg: &Lfm2Config, dev: &Device) -> Result<Lfm2<P>> {
    let mut tensors = HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Lfm2::<P>::new(cfg, vb)
}

/// Every conv layer's state and the KV length of `a` against `b`.
fn lfm2_state_diff<P: Profile>(a: &Lfm2State<P>, b: &Lfm2State<P>, layers: usize) -> Result<f32> {
    assert_eq!(a.kv_len(), b.kv_len(), "KV length");
    assert_eq!(a.conv_phase(), b.conv_phase(), "conv phase");
    let mut d = 0f32;
    for l in 0..layers {
        if let (Some(x), Some(y)) = (a.conv_state_for_test(l), b.conv_state_for_test(l)) {
            d = d.max(max_abs_diff(&x, &y)?);
        }
    }
    Ok(d)
}

fn lfm2_state<P: Lfm2Shape>(cfg: &Lfm2Config, dev: &Device, pin: bool) -> Result<Lfm2State<P>> {
    Ok(Lfm2State::<P>::new(true, DType::F32, cfg, dev)?.with_pin_frame(pin))
}

/// LFM2's first turn with the head's tail and without it: the same logits
/// and state, the pass's own checkpoint where it was. The head's tail,
/// pinned, holds the input columns before the head: within the pass's own
/// checkpoint's reach (`TURN_CHECKPOINT_SLACK`) a restore from the pin is a
/// restore from the pass's own tail, bit for bit; past it only the pin
/// reaches the head.
#[test]
fn lfm2_the_pass_that_takes_the_heads_tail_computes_what_it_computed() -> Result<()> {
    pin_regime();
    let dev = Device::Cpu;
    let cfg = lfm2_config();
    let model = lfm2_build::<CrownExact>(&cfg, &dev)?;
    let layers = cfg.num_hidden_layers;
    let prompt: Vec<u32> = (0..48u32).map(|i| (i * 5 + 3) % 46).collect();
    let run = |n: usize, head: Option<usize>| -> Result<(Lfm2State<CrownExact>, Tensor)> {
        let mut st = lfm2_state::<CrownExact>(&cfg, &dev, head.is_some())?;
        model.open_checkpoint_request(&mut st, CheckpointPolicy::End, n)?;
        if let Some(h) = head {
            assert_eq!(model.pin_checkpoint(&mut st, Some(h))?, None);
        }
        let l = logits(&model, &mut st, 0, &prompt[..n], &dev)?;
        model.close_checkpoint_request(&mut st)?;
        Ok((st, l))
    };
    for (n, head) in [(24, 12), (40, 10)] {
        let tag = format!("n {n} head {head}");
        let (mut one, la) = run(n, None)?;
        let (mut two, lb) = run(n, Some(head))?;
        let (da, db) = (digest(&la)?, digest(&lb)?);
        eprintln!("PIN-LFM2 {tag}: logits sha256 without the head's tail {da} with it {db}");
        assert_eq!(
            da, db,
            "{tag}: the pass's logits moved with the head's tail"
        );
        assert_eq!(
            lfm2_state_diff(&one, &two, layers)?,
            0.0,
            "{tag}: the state"
        );
        assert_eq!(two.pinned_checkpoint(), Some(head));
        one.restore(n)?;
        two.restore(n)?;
        assert_eq!(
            lfm2_state_diff(&one, &two, layers)?,
            0.0,
            "{tag}: the pass's own checkpoint"
        );
        let in_reach = n - head <= lloom::models::lfm2::TURN_CHECKPOINT_SLACK;
        assert_eq!(
            one.can_restore(head),
            in_reach,
            "{tag}: the pass's own reach"
        );
        assert!(two.can_restore(head), "{tag}: the pin reaches the head");
        if in_reach {
            // Leave the pin the only checkpoint reaching the head: a restore
            // between the head and the pass's end forgets the pass's own.
            two.restore(head + 1)?;
            assert_eq!(two.pinned_checkpoint(), Some(head));
            two.restore(head)?;
            one.restore(head)?;
            let d = lfm2_state_diff(&one, &two, layers)?;
            assert_eq!(d, 0.0, "{tag}: the pin's tail is the pass's own columns");
            let a = logits(&model, &mut two, head, &prompt[head..head + 6], &dev)?;
            let b = logits(&model, &mut one, head, &prompt[head..head + 6], &dev)?;
            assert_eq!(max_abs_diff(&a, &b)?, 0.0, "{tag}: the continuation");
        } else {
            two.restore(head)?;
            assert_eq!(
                two.pinned_checkpoint(),
                Some(head),
                "a restore at the pin keeps it"
            );
            let mut fresh = lfm2_state::<CrownExact>(&cfg, &dev, false)?;
            logits(&model, &mut fresh, 0, &prompt[..head], &dev)?;
            let d = lfm2_state_diff(&two, &fresh, layers)?;
            eprintln!(
                "PIN-LFM2 {tag}: past the pass's reach, the pinned head vs a fresh prefill of it \
                 (another split, printed): {d:e}"
            );
        }
        two.restore(head - 1)?;
        assert_eq!(
            two.pinned_checkpoint(),
            None,
            "{tag}: a restore below the pin releases it"
        );
    }
    Ok(())
}

/// More extending turns than LFM2's ring has slots, then a compaction: the
/// pinned head restores (`restore_point` falls back to it past every turn
/// checkpoint's reach) and the continuation is the first turn's own. Each
/// turn's pass is 16 ids, so a pass whose tail were written into the pin's
/// frame would cover the columns a restore at the head reads (a 5-id pass
/// writes 7 columns and leaves them standing). The control: without the pin
/// frame the compaction is a fresh prefill.
#[test]
fn lfm2_a_pin_survives_more_turns_than_the_ring_has_slots() -> Result<()> {
    pin_regime();
    let dev = Device::Cpu;
    let cfg = lfm2_config();
    let model = lfm2_build::<CrownExact>(&cfg, &dev)?;
    let head = 12;
    let turns = lloom::models::lfm2::TURN_CHECKPOINTS + 2;
    let sys: Vec<u32> = (0..head as u32).map(|i| (i * 5 + 3) % 46).collect();
    let run =
        |pin: bool| -> Result<(lloom::nn::Prepared, Session<'_, Lfm2<CrownExact>>, Vec<u32>)> {
            let mut session = Session::new(&model, lfm2_state::<CrownExact>(&cfg, &dev, pin)?);
            let mut full = sys.clone();
            for k in 0..turns {
                full.extend((0..16u32).map(|i| (i * 3 + 11 + k as u32) % 46));
                let (p, reply) = request(&mut session, &full, pin.then_some(head), 3, &dev)?;
                if k > 0 {
                    assert_eq!(p.action, ReuseAction::Extend, "turn {k} extends");
                }
                full.extend(reply);
                if pin {
                    assert_eq!(
                        session.sequence().cache().pinned_checkpoint(),
                        Some(head),
                        "turn {k}: the head stays pinned"
                    );
                }
            }
            let mut compact = sys.clone();
            compact.push((full[head] + 1) % 46);
            compact.extend((0..9u32).map(|i| (i * 7 + 1) % 46));
            let prepared = session.prepare(&compact)?;
            Ok((prepared, session, compact))
        };
    let (prepared, mut session, compact) = run(true)?;
    eprintln!(
        "PIN-LFM2-CONV the compaction after {turns} turns prepared {:?}, reused {}",
        prepared.action, prepared.reused
    );
    assert!(matches!(prepared.action, ReuseAction::Rewound(_)));
    assert_eq!(prepared.reused, head);
    // The reference: the first turn's pass (the same split), restored to
    // the head through its own tail's reach.
    let mut turn1 = compact[..head].to_vec();
    turn1.extend((0..16u32).map(|i| (i * 3 + 11) % 46));
    let mut rs = Session::new(&model, lfm2_state::<CrownExact>(&cfg, &dev, false)?);
    request(&mut rs, &turn1, None, 3, &dev)?;
    let same = rs.cache_mut();
    same.restore(head)?;
    let d = lfm2_state_diff(session.sequence().cache(), same, cfg.num_hidden_layers)?;
    assert_eq!(d, 0.0, "the restored head is the first turn's state there");
    let a = logits(&model, session.cache_mut(), head, &compact[head..], &dev)?;
    let b = logits(&model, same, head, &compact[head..], &dev)?;
    assert_eq!(
        max_abs_diff(&a, &b)?,
        0.0,
        "the compaction's prefill from the head"
    );
    let (control, _, _) = run(false)?;
    assert_eq!(
        control.action,
        ReuseAction::Fresh,
        "the control: without the pin the compaction prefills from nothing"
    );
    Ok(())
}

// ---- a real checkpoint ---------------------------------------------------------

/// The first turn's prompt ids and head length, from `LLOOM_PIN_REAL_IDS`
/// (`{"ids": [...], "head": n}`, what a serve run's first turn reported),
/// else `n` ids of the vocabulary with the head at a third of them.
#[cfg(feature = "metal")]
fn real_ids(vocab: usize, n: usize) -> Result<(Vec<u32>, usize)> {
    match std::env::var_os("LLOOM_PIN_REAL_IDS") {
        Some(p) => {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&p)?).map_err(lloom_tensor::Error::wrap)?;
            let ids = v["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|x| x.as_u64().expect("an id") as u32)
                .collect();
            Ok((ids, v["head"].as_u64().expect("head") as usize))
        }
        None => Ok((tokens(n, vocab, 3), n / 3)),
    }
}

/// [`two_targets_case`] on a real Qwen3.5 / 3.6 checkpoint
/// (`LLOOM_PIN_REAL_QWEN3_5`, a GGUF's first shard), the first turn's ids as
/// a serve run sent them: the pass with the head's checkpoint beside its own
/// computes the logits the pass without it computes, by digest; the state
/// it leaves and its own checkpoint are the same; the head's is the state a
/// checkpoint asked at the head leaves over the same pass. Skipped, with a
/// line, where the variable is unset.
#[cfg(feature = "metal")]
#[test]
fn the_two_target_pass_on_a_real_qwen3_5_checkpoint() -> Result<()> {
    use lloom::models::qwen3_5::{LoadOptions, Qwen3_5};
    let Some(path) = std::env::var_os("LLOOM_PIN_REAL_QWEN3_5") else {
        eprintln!("PIN-REAL qwen3_5 SKIPPED: LLOOM_PIN_REAL_QWEN3_5 unset");
        return Ok(());
    };
    pin_regime();
    let dev = Device::new_metal(0)?;
    let shards = lloom::hub::gguf_shards(std::path::Path::new(&path));
    let opts = LoadOptions {
        kv_capacity: 4096,
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
        pin: true,
        purpose: lloom::nn::attention::route::ExecutionPurpose::Production,
        ..LoadOptions::default()
    };
    let t = std::time::Instant::now();
    let model = Qwen3_5::<CrownExact>::from_gguf_shards(&shards, &dev, &opts)?;
    let cfg = model.config().clone();
    eprintln!(
        "PIN-REAL qwen3_5 loaded {} in {:.1} s: {} checkpoint slots, pin frame {}",
        std::path::Path::new(&path).display(),
        t.elapsed().as_secs_f64(),
        model.capacity().checkpoints,
        model.capacity().pin
    );
    let (ids, head) = real_ids(cfg.vocab_size, 600)?;
    let n = ids.len();
    for (tiles, tail) in [(vec![n], 0), (vec![n], 4)] {
        real_case(&model, &cfg, &dev, &ids, &tiles, head, tail)?;
    }
    Ok(())
}

/// One real first turn, with the head's checkpoint and without it.
#[cfg(feature = "metal")]
fn real_case<M, L>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    ids: &[u32],
    tiles: &[usize],
    head: usize,
    tail: usize,
) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
{
    let n: usize = tiles.iter().sum();
    let own = n - tail;
    let policy = match tail {
        0 => CheckpointPolicy::End,
        t => CheckpointPolicy::At(n - t),
    };
    let tag = format!("n {n} tiles {tiles:?} head {head} tail {tail}");
    let (mut one, la) = first_turn(model, cfg, dev, ids, tiles, policy, None)?;
    let (mut two, lb) = first_turn(model, cfg, dev, ids, tiles, policy, Some(head))?;
    let (da, db) = (digest(&la)?, digest(&lb)?);
    eprintln!("PIN-REAL qwen3_5 {tag}: first-turn logits sha256 without the head's checkpoint {da} with it {db}");
    assert_eq!(
        da, db,
        "{tag}: the pass's logits moved with the head's checkpoint"
    );
    let after = state_diff(&one, &two)?;
    eprintln!("PIN-REAL qwen3_5 {tag}: the state after the pass, without vs with: {after:?}");
    assert_eq!(after, (0.0, 0.0), "{tag}: the state the pass leaves moved");
    assert_eq!(two.pinned_checkpoint(), Some(head), "{tag}");
    one.restore(own)?;
    two.restore(own)?;
    let own_d = state_diff(&one, &two)?;
    eprintln!("PIN-REAL qwen3_5 {tag}: the request's own checkpoint, without vs with: {own_d:?}");
    assert_eq!(own_d, (0.0, 0.0), "{tag}: the request's own checkpoint");
    drop(one);
    two.restore(head)?;
    let (mut same, _) = first_turn(
        model,
        cfg,
        dev,
        ids,
        tiles,
        CheckpointPolicy::At(head),
        None,
    )?;
    same.restore(head)?;
    let head_d = state_diff(&two, &same)?;
    eprintln!(
        "PIN-REAL qwen3_5 {tag}: the pinned head vs a checkpoint asked at the head: {head_d:?}"
    );
    assert_eq!(head_d, (0.0, 0.0), "{tag}: the pinned head");
    Ok(())
}

/// [`lfm2_the_pass_that_takes_the_heads_tail_computes_what_it_computed`] on
/// the real LFM2.5-2.6B (`LLOOM_FIXTURE=real`, through the fixture), the first
/// turn's ids from `LLOOM_PIN_REAL_IDS` where set: the logits by digest, the
/// state and the pass's own checkpoint, with the head's tail and without.
#[cfg(feature = "metal")]
#[test]
fn the_pass_that_takes_the_heads_tail_on_a_real_lfm2_checkpoint() -> Result<()> {
    use lloom::models::lfm2_testing::Lfm2Tiny;
    use lloom::testing::fixture::Fixture;
    if !lloom::testing::fixture::Backend::current().at_scale() {
        eprintln!("PIN-REAL lfm2 SKIPPED: LLOOM_FIXTURE is not real");
        return Ok(());
    }
    let Some(fx) = Fixture::<Lfm2Tiny>::new("pin-real-lfm2", lfm2_config())? else {
        return Ok(());
    };
    let Some(dev) = fx.device(Device::Cpu) else {
        return Ok(());
    };
    pin_regime();
    let cfg = fx.config(|c| {
        c.kv_append = KvAppend::InPlace;
        c.conv_state = ConvState::Shuffle;
        c.conv_prefill_ctx = true;
    });
    let model = fx.model::<CrownExact>(&cfg, &dev)?;
    let (ids, head) = real_ids(cfg.vocab_size, 600)?;
    let n = ids.len();
    let dtype = lloom::models::lfm2_testing::run_dtype(&dev);
    let run = |pin: bool| -> Result<(Lfm2State<CrownExact>, Tensor)> {
        let mut st = Lfm2State::<CrownExact>::new(true, dtype, &cfg, &dev)?.with_pin_frame(pin);
        model.open_checkpoint_request(&mut st, CheckpointPolicy::End, n)?;
        if pin {
            model.pin_checkpoint(&mut st, Some(head))?;
        }
        let l = logits(&model, &mut st, 0, &ids, &dev)?;
        model.close_checkpoint_request(&mut st)?;
        Ok((st, l))
    };
    let (mut one, la) = run(false)?;
    let (mut two, lb) = run(true)?;
    let (da, db) = (digest(&la)?, digest(&lb)?);
    eprintln!("PIN-REAL lfm2 n {n} head {head}: first-turn logits sha256 without the head's tail {da} with it {db}");
    assert_eq!(da, db, "the pass's logits moved with the head's tail");
    let layers = cfg.num_hidden_layers;
    assert_eq!(lfm2_state_diff(&one, &two, layers)?, 0.0, "the state");
    assert_eq!(two.pinned_checkpoint(), Some(head));
    one.restore(n)?;
    two.restore(n)?;
    assert_eq!(
        lfm2_state_diff(&one, &two, layers)?,
        0.0,
        "the pass's own checkpoint"
    );
    two.restore(head)?;
    assert_eq!(two.pinned_checkpoint(), Some(head));
    eprintln!("PIN-REAL lfm2 n {n} head {head}: restored at the pinned head");
    Ok(())
}
