//! The sequence layer (issue #677, `design/docs/sequence-reuse-design.md`):
//! a `Session` owns its token log, `prepare` reuses whatever prefix the cache
//! holds, an asynchronous loop's EOS overrun is rewound at the sequence
//! boundary, and a multi-token pass onto a warm cache is exact only when the
//! conv writer carries the left context (`DESIGN.md` §13.2f defect 1).
//!
//! CPU only, on a tiny LFM2 with fixed-ramp weights (`dspark_decode_identity.rs`'s
//! fixture) and on a position-counter fake for the bookkeeping. The reference
//! in every test is a **different mechanism** than the subject
//! (`CONTRIBUTING.md` §3.1b): one pass over a concatenation against two passes,
//! a solo run against an interleaved one, a fresh session against a reused one.

use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{
    Config, ConvState, KvAppend, LayerType, Model, Shape as Lfm2Shape, State,
};
use lloom::nn::{
    CacheReset, CacheRewind, CausalLM, DecodeOptions, IdentityProposer, Model as ModelTrait,
    ReuseAction, Session, VarBuilder,
};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::{DType, Device, IndexOp, Result, Tensor};

// ---------------------------------------------------------------------------
// The tiny LFM2 fixture.
// ---------------------------------------------------------------------------

fn tiny_config(conv_state: ConvState, kv_append: KvAppend, conv_prefill_ctx: bool) -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 256,
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
        eos_token_id: None,
        use_flash_attn: false,
        kv_append,
        conv_state,
        conv_prefill_ctx,
        ..Default::default()
    }
}

fn build<P: Lfm2Shape>(cfg: &Config, dev: &Device) -> Result<Model<P>> {
    let mut tensors = std::collections::HashMap::new();
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
    Model::<P>::new(cfg, vb)
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shapes differ");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

// ---------------------------------------------------------------------------
// §13.2f defect 1: the left context of a multi-token pass onto a warm cache.
// ---------------------------------------------------------------------------

/// One pass over `a ++ b` against a pass over `a` then a pass over `b`, per
/// conv arm. With `ConvPrefillCtx=on` the last position's logits agree to
/// float tolerance; with it off — the shipping zero-padded writer — they do
/// not, which is the control that makes the agreement evidence rather than a
/// tolerance wide enough to pass anything. Both the shipping multi-turn path
/// and prefix reuse are the second shape.
#[test]
fn a_pass_onto_a_warm_cache_matches_one_pass_over_the_concatenation_iff_the_context_is_carried(
) -> Result<()> {
    let dev = Device::Cpu;
    let a: Vec<u32> = vec![3, 9, 14, 21, 5, 30, 2];
    let b: Vec<u32> = vec![11, 4, 40, 8, 19, 6];
    let full: Vec<u32> = [a.clone(), b.clone()].concat();
    for (arm, kv) in [
        (ConvState::Shuffle, KvAppend::Cat),
        (ConvState::Shuffle, KvAppend::InPlace),
        (ConvState::RotatingRing { k: 4 }, KvAppend::InPlace),
        (
            ConvState::SlidingRing { k: 0, slack: 16 },
            KvAppend::InPlace,
        ),
    ] {
        let mut deltas = Vec::new();
        for carry in [false, true] {
            let cfg = tiny_config(arm, kv, carry);
            let key = cfg.exact_key();
            let delta = lloom::profile::dispatch!(key, lloom::profile::LFM2, P => warm_case::<P>(&cfg, &dev, &a, &b, &full))?;
            deltas.push(delta);
        }
        let (off, on) = (deltas[0], deltas[1]);
        eprintln!("{arm:?} x {kv:?}: max|Δlogit| off={off:.3e} on={on:.3e}");
        assert!(
            on < 1e-5,
            "{arm:?} x {kv:?}: with the context carried the two shapes must agree; \
             max|Δ| = {on:.3e}"
        );
        assert!(
            off > 1e-3,
            "{arm:?} x {kv:?}: the control — the zero-padded writer — must DISAGREE, or \
             the tolerance above passes anything; max|Δ| = {off:.3e}"
        );
    }
    Ok(())
}

/// One cell of the warm-cache test, at the profile its config names: the
/// max |Δlogit| between one pass over the concatenation and `a` then `b`.
fn warm_case<P: Lfm2Shape>(
    cfg: &Config,
    dev: &Device,
    a: &[u32],
    b: &[u32],
    full: &[u32],
) -> Result<f32> {
    let model = build::<P>(&cfg, &dev)?;

    // Reference: one pass over the concatenation, every position's
    // logits, of which `b`'s positions are compared.
    let mut cache = State::<P>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(full, &dev)?.unsqueeze(0)?;
    let one_pass = model
        .forward_all(&inp, 0, &mut cache)?
        .narrow(1, a.len(), b.len())?
        .contiguous()?;

    // Subject: `a` (prefill), then a decode step so the ring has
    // moved off its seed, then `b` onto the warm cache.
    let mut cache = State::<P>::new(true, DType::F32, &cfg, &dev)?;
    let (head, tail) = a.split_at(a.len() - 1);
    let inp = Tensor::new(head, &dev)?.unsqueeze(0)?;
    model.forward(&inp, 0, &mut cache)?;
    let inp = Tensor::new(tail, &dev)?.unsqueeze(0)?;
    model.forward(&inp, head.len(), &mut cache)?;
    let inp = Tensor::new(b, &dev)?.unsqueeze(0)?;
    let two_pass = model.forward_all(&inp, a.len(), &mut cache)?;

    max_abs_diff(&one_pass, &two_pass)
}

/// The conv state itself after the second pass equals the state after the
/// single pass — the hand-over half §13.2f says must be preserved — on the
/// shuffle arm where the state is the window.
#[test]
fn the_state_after_a_warm_pass_is_the_single_pass_state_when_the_context_is_carried() -> Result<()>
{
    let dev = Device::Cpu;
    let a: Vec<u32> = vec![3, 9, 14, 21];
    // A short second pass, shorter than `l_cache`, so the state must keep
    // columns from BEFORE it. (The ramp weights repeat every 17 vocabulary
    // rows, so ids 17 apart embed identically — 21 and 4 would — and the
    // "columns differ" sanity check below needs ids that do not.)
    let b: Vec<u32> = vec![11, 5];
    let cfg = tiny_config(ConvState::Shuffle, KvAppend::InPlace, true);
    let model = build::<CrownExact>(&cfg, &dev)?;

    let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new([a.clone(), b.clone()].concat().as_slice(), &dev)?.unsqueeze(0)?;
    model.forward(&inp, 0, &mut cache)?;
    let one = cache.conv_state_for_test(0).unwrap();

    let mut cache = State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?;
    let inp = Tensor::new(a.as_slice(), &dev)?.unsqueeze(0)?;
    model.forward(&inp, 0, &mut cache)?;
    let inp = Tensor::new(b.as_slice(), &dev)?.unsqueeze(0)?;
    model.forward(&inp, a.len(), &mut cache)?;
    let two = cache.conv_state_for_test(0).unwrap();

    let d = max_abs_diff(&one, &two)?;
    assert!(
        d < 1e-6,
        "the carried state must be the single-pass state; max|Δ| = {d:.3e}"
    );
    // And the last column is this pass's own last `bx` — the state is not a
    // stale copy of the first pass's.
    let last_one = one.i((0, .., 2))?;
    let first_one = one.i((0, .., 0))?;
    assert!(max_abs_diff(&last_one, &first_one)? > 1e-6);
    Ok(())
}

// ---------------------------------------------------------------------------
// The bookkeeping, on a position-counter fake.
// ---------------------------------------------------------------------------

const VOCAB: usize = 64;

fn next_of(tok: u32) -> u32 {
    (tok * 7 + 3) % VOCAB as u32
}

fn chain(start: u32, max: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut tok = start;
    while ids.len() < max {
        tok = next_of(tok);
        ids.push(tok);
    }
    ids
}

/// A cache that is a position counter with a fixed rewind horizon.
struct Counter {
    pos: usize,
    horizon: usize,
}

impl CacheReset for Counter {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl CacheRewind for Counter {
    fn rewind_horizon(&self) -> usize {
        self.horizon.min(self.pos)
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n > self.rewind_horizon() {
            lloom_tensor::bail!("Counter: rewind {n} past horizon {}", self.rewind_horizon());
        }
        self.pos -= n;
        Ok(())
    }
}

/// `decode_driver_tests.rs`'s device-side chain model over a `Counter` cache.
struct Chain {
    table: Tensor,
    eye: Tensor,
    horizon: usize,
}

impl Chain {
    fn new(device: &Device, horizon: usize) -> Result<Self> {
        let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
        Ok(Self {
            table: Tensor::from_vec(table, VOCAB, device)?,
            eye: Tensor::eye(VOCAB, DType::F32, device)?,
            horizon,
        })
    }
}

impl ModelTrait for Chain {
    type Config = ();
    fn from_weights(_vb: VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("Chain has no weights")
    }
}

impl CausalLM for Chain {
    type Cache = Counter;
    fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<Counter> {
        Ok(Counter {
            pos: 0,
            horizon: self.horizon,
        })
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }
    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut Counter) -> Result<Tensor> {
        let (_b, seq) = embeds.dims2()?;
        assert_eq!(
            cache.pos, pos,
            "forward at pos {pos} against a cache of {}",
            cache.pos
        );
        cache.pos += seq;
        let last = embeds.narrow(1, seq - 1, 1)?.flatten_all()?;
        let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
        self.eye.index_select(&next, 0)
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

fn sink(_: &[u32]) -> Result<()> {
    Ok(())
}

/// `prepare` tells the three cases apart by prefix length alone, and a resumed
/// or rewound sequence continues exactly where a fresh one would be.
#[test]
fn prepare_extends_resumes_rewinds_or_resets_by_prefix_length() -> Result<()> {
    let dev = Device::Cpu;
    let never: Vec<u32> = vec![63];
    let model = Chain::new(&dev, 2)?;
    let mut session = Session::new(&model, model.new_cache(&(), &dev)?);
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&never, &dev)?;
    let mut sampler = greedy();

    // Turn 1: prompt [0], six tokens. The log is the prompt plus the ids.
    let first = session.decode(
        &mut spec,
        &mut rb,
        &mut sampler,
        &[0],
        &DecodeOptions::new(6),
        &mut sink,
    )?;
    assert_eq!(first.ids, chain(0, 6));
    let log: Vec<u32> = [vec![0], chain(0, 6)].concat();
    assert_eq!(session.tokens(), &log[..]);
    assert_eq!(session.pos(), 7);
    assert_eq!(session.overrun(), 0);

    // Resume: the request IS the log, and the last logits are held.
    let p = session.prepare(&log)?;
    assert_eq!(p.action, ReuseAction::Resume);
    assert_eq!((p.reused, p.suffix.len()), (7, 0));
    let more = session.decode(
        &mut spec,
        &mut rb,
        &mut sampler,
        &[],
        &DecodeOptions::new(3),
        &mut sink,
    )?;
    assert_eq!(
        more.ids,
        chain(0, 9)[6..],
        "a resume continues the chain exactly"
    );
    assert_eq!(session.pos(), 10);

    // Extend: the request is the log plus a new turn — only the turn is fed.
    let log: Vec<u32> = [vec![0], chain(0, 9)].concat();
    let req: Vec<u32> = [log.clone(), vec![5, 1]].concat();
    let p = session.prepare(&req)?;
    assert_eq!(p.action, ReuseAction::Extend);
    assert_eq!(p.reused, 10);
    assert_eq!(p.suffix, vec![5, 1]);
    let out = session.decode(
        &mut spec,
        &mut rb,
        &mut sampler,
        &p.suffix,
        &DecodeOptions::new(4),
        &mut sink,
    )?;
    assert_eq!(out.ids, chain(1, 4));
    assert_eq!(session.pos(), 12 + 4);

    // Rewound: diverge two tokens back, inside the horizon of 2.
    let log = session.tokens().to_vec();
    let mut req = log[..log.len() - 2].to_vec();
    req.extend_from_slice(&[9, 9, 9]);
    let p = session.prepare(&req)?;
    assert_eq!(p.action, ReuseAction::Rewound(2));
    assert_eq!(p.reused, log.len() - 2);
    assert_eq!(p.suffix, vec![9, 9, 9]);
    assert_eq!(session.pos(), log.len() - 2);
    let out = session.decode(
        &mut spec,
        &mut rb,
        &mut sampler,
        &p.suffix,
        &DecodeOptions::new(2),
        &mut sink,
    )?;
    assert_eq!(out.ids, chain(9, 2));

    // Everything cached but no held logits (the last run's were consumed by
    // the divergence): the last id is re-fed through a one-token rewind.
    let log = session.tokens().to_vec();
    session.prepare(&[log.clone(), vec![2]].concat())?; // Extend, consumes nothing
    let p = session.prepare(&log)?;
    assert_eq!(
        p.action,
        ReuseAction::Resume,
        "the held logits survive a prepare that fed nothing"
    );
    // Force the no-logits case: a divergence at the very end.
    let mut req = log.clone();
    *req.last_mut().unwrap() ^= 1;
    let p = session.prepare(&req)?;
    assert_eq!(p.action, ReuseAction::Rewound(1));
    assert_eq!(p.suffix.len(), 1);
    session.decode(
        &mut spec,
        &mut rb,
        &mut sampler,
        &p.suffix,
        &DecodeOptions::new(1),
        &mut sink,
    )?;

    // Fresh: a divergence deeper than the horizon resets the sequence.
    let log = session.tokens().to_vec();
    let mut req = log[..3].to_vec();
    req.extend_from_slice(&[7, 7, 7, 7]);
    let p = session.prepare(&req)?;
    assert_eq!(p.action, ReuseAction::Fresh);
    assert_eq!(p.reused, 0);
    assert_eq!(p.suffix, req);
    assert_eq!(session.pos(), 0);
    assert!(session.tokens().is_empty());

    // A cache prefilled by a caller that did not say with what is only ever
    // reset.
    let mut unknown = Session::with_pos(&model, Counter { pos: 5, horizon: 2 }, 5);
    let p = unknown.prepare(&[1, 2, 3])?;
    assert_eq!(p.action, ReuseAction::Fresh);
    assert_eq!(unknown.pos(), 0);
    Ok(())
}

/// An asynchronous loop's EOS overrun is discarded at the sequence boundary,
/// and the continuation is byte-identical to the conservative loop's.
#[test]
fn an_eos_overrun_is_rewound_and_the_continuation_is_byte_identical() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config(ConvState::RotatingRing { k: 8 }, KvAppend::InPlace, true);
    let model = build::<RewindableExact>(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let never: Vec<u32> = vec![47];

    // The reference stream, with no EOS reachable, to pick an EOS from.
    let stream = {
        let mut s = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut rb = Blocking::new(&never, &dev)?;
        s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(24),
            &mut sink,
        )?
        .ids
    };
    assert!(stream.len() == 24);
    // The EOS: the first id (past the opening tokens) that has not appeared
    // earlier in the stream, so the run stops exactly there. The tiny model's
    // greedy stream repeats itself quickly, so this is searched rather than
    // fixed; the position is reported so a change in the fixture is visible.
    let cut = (1..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a stream with a fresh id somewhere");
    let eos = stream[cut];
    eprintln!("stream {stream:?}; EOS {eos} at {cut}");
    let eos_ids = vec![eos];
    let turn2: Vec<u32> = vec![10, 20, 30];

    // The conservative reference: lookahead 0 on the blocking arm, turn 1 to
    // EOS, then turn 2 appended (the cache ends at the last kept token).
    let reference = {
        let mut s = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let t1 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(24),
            &mut sink,
        )?;
        assert!(t1.eos);
        assert_eq!(t1.ids, stream[..cut]);
        assert_eq!(t1.stats.overrun, 0);
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(p.action, ReuseAction::Extend);
        assert_eq!(p.suffix, turn2);
        s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &p.suffix,
            &DecodeOptions::new(8),
            &mut sink,
        )?
        .ids
    };
    assert_eq!(reference.len(), 8);

    // The asynchronous arms: each overruns the EOS, reports it, and continues
    // identically after the rewind.
    for (name, eos_read, budget) in [
        (
            "poll lookahead 1",
            EosRead::Poll(PollMechanism::default()),
            CommitBudget::per_token(),
        ),
        (
            "host-on-drain steps:4",
            EosRead::HostOnDrain,
            CommitBudget::steps(4)?,
        ),
    ] {
        let mut s = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut rb = Ring::new(&eos_ids, &dev, 128, eos_read)?.with_poll_budget(20_000);
        let opts = DecodeOptions {
            lookahead: 1,
            budget,
            ..DecodeOptions::new(24)
        };
        let t1 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &opts,
            &mut sink,
        )?;
        assert!(t1.eos, "{name}");
        assert_eq!(t1.ids, stream[..cut], "{name}");
        assert!(
            t1.stats.overrun >= 1,
            "{name}: an asynchronous arm overruns an EOS"
        );
        assert_eq!(s.pos(), prompt.len() + cut + t1.stats.overrun, "{name}");
        assert_eq!(s.overrun(), t1.stats.overrun);
        // A decode is refused until the overrun is dealt with.
        let err = s
            .decode(
                &mut IdentityProposer,
                &mut rb,
                &mut greedy(),
                &turn2,
                &opts,
                &mut sink,
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("overrun"), "{name}: {err}");
        // `prepare` reconciles: the overrun is rewound, the turn is the suffix.
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(p.action, ReuseAction::Extend, "{name}");
        assert_eq!(p.suffix, turn2, "{name}");
        assert_eq!(
            s.pos(),
            prompt.len() + cut,
            "{name}: the cache ends at the last kept token"
        );
        assert_eq!(s.overrun(), 0);
        let t2 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &p.suffix,
            &DecodeOptions {
                lookahead: 1,
                budget,
                ..DecodeOptions::new(8)
            },
            &mut sink,
        )?;
        assert_eq!(
            t2.ids, reference,
            "{name}: the continuation after the rewind must be byte-identical"
        );
    }

    // The unrewindable cache — the crown's `Cat` × `Shuffle` — reports the
    // overrun and refuses the rewind, so `prepare` starts fresh.
    {
        let cfg = tiny_config(ConvState::Shuffle, KvAppend::Cat, true);
        let model = build::<CrownExact>(&cfg, &dev)?;
        let mut s = Session::new(
            &model,
            State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut rb = Ring::new(&eos_ids, &dev, 128, EosRead::Poll(PollMechanism::default()))?
            .with_poll_budget(20_000);
        let opts = DecodeOptions {
            lookahead: 1,
            ..DecodeOptions::new(24)
        };
        let t1 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &opts,
            &mut sink,
        )?;
        assert_eq!(t1.stats.overrun, 1);
        assert!(s.reconcile().is_err(), "the crown cache cannot rewind");
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(p.action, ReuseAction::Fresh);
        assert_eq!(p.suffix, full);
    }
    Ok(())
}

/// Two sequences interleaved on one model, each continuing from its own held
/// logits, produce exactly their solo streams — the concurrent-request test.
#[test]
fn two_sequences_interleaved_are_byte_identical_to_their_solo_runs() -> Result<()> {
    let dev = Device::Cpu;
    // The crown-shaped cache: `Resume` needs no rewind, so this works on the
    // unrewindable arms too.
    let cfg = tiny_config(ConvState::Shuffle, KvAppend::Cat, false);
    let model = build::<CrownExact>(&cfg, &dev)?;
    let never: Vec<u32> = vec![47];
    let prompts: Vec<Vec<u32>> = vec![
        vec![1, 2, 3, 4, 5],
        vec![9, 8, 7],
        vec![21, 22, 23, 24, 25, 26],
    ];
    let total = 20usize;
    let chunk = 4usize;

    let solo: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| -> Result<Vec<u32>> {
            let mut s = Session::new(
                &model,
                State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?,
            );
            let mut rb = Ring::new(&never, &dev, 64, EosRead::Poll(PollMechanism::default()))?
                .with_poll_budget(20_000);
            let opts = DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(total)
            };
            Ok(s.decode(
                &mut IdentityProposer,
                &mut rb,
                &mut greedy(),
                p,
                &opts,
                &mut sink,
            )?
            .ids)
        })
        .collect::<Result<_>>()?;
    assert!(solo.iter().all(|s| s.len() == total));
    assert_ne!(solo[0], solo[1], "the fixture must give different streams");

    // Interleaved: one session and one ring per sequence, `chunk` tokens at a
    // time, round-robin, continuing through `prepare` on each visit.
    let mut sessions: Vec<Session<'_, Model<CrownExact>>> = prompts
        .iter()
        .map(|_| {
            Ok(Session::new(
                &model,
                State::<CrownExact>::new(true, DType::F32, &cfg, &dev)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut rings: Vec<Ring> = prompts
        .iter()
        .map(|_| {
            Ok(
                Ring::new(&never, &dev, 64, EosRead::Poll(PollMechanism::default()))?
                    .with_poll_budget(20_000),
            )
        })
        .collect::<Result<_>>()?;
    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut resumes = 0usize;
    while streams.iter().any(|s| s.len() < total) {
        for i in 0..prompts.len() {
            if streams[i].len() >= total {
                continue;
            }
            let full: Vec<u32> = [prompts[i].clone(), streams[i].clone()].concat();
            let p = sessions[i].prepare(&full)?;
            if streams[i].is_empty() {
                assert_eq!(p.action, ReuseAction::Fresh);
            } else {
                assert_eq!(
                    p.action,
                    ReuseAction::Resume,
                    "seq {i}: a visit resumes from the held logits"
                );
                resumes += 1;
            }
            let n = chunk.min(total - streams[i].len());
            let opts = DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(n)
            };
            let out = sessions[i].decode(
                &mut IdentityProposer,
                &mut rings[i],
                &mut greedy(),
                &p.suffix,
                &opts,
                &mut sink,
            )?;
            assert_eq!(out.ids.len(), n);
            streams[i].extend_from_slice(&out.ids);
        }
    }
    assert!(resumes > 0);
    for (i, (got, want)) in streams.iter().zip(solo.iter()).enumerate() {
        assert_eq!(got, want, "sequence {i} diverged from its solo run");
        assert_eq!(
            sessions[i].tokens(),
            [prompts[i].clone(), want.clone()].concat()
        );
    }
    // The rings' publishes reconcile per sequence.
    for (i, r) in rings.iter().enumerate() {
        let c = r.counters();
        assert_eq!(c.ring_publishes, total as u64, "seq {i}: {c:?}");
    }
    Ok(())
}

/// Retained branching reuses the production paced prefill loop, but keeps its
/// recurrent snapshot outside the finite turn ring. The fresh reference repeats
/// the SAME tile split with ordinary context carry explicitly enabled. This is
/// not a cross-partition bit-identity assertion.
#[test]
fn retained_request_survives_long_suffixes_and_preserves_lfm2_carry() -> Result<()> {
    for (conv, kv) in [
        (ConvState::Shuffle, KvAppend::Cat),
        (ConvState::Shuffle, KvAppend::InPlace),
        (ConvState::RotatingRing { k: 4 }, KvAppend::InPlace),
        (ConvState::SlidingRing { k: 0, slack: 8 }, KvAppend::InPlace),
    ] {
        // Shipping flag remains false: only the private branching cache may
        // enable context carry. Reuse the incumbent bounded fixture/loader.
        let cfg = tiny_config(conv, kv, false);
        let key = cfg.exact_key();
        lloom::profile::dispatch!(key, lloom::profile::LFM2, P => retained_request_case::<P>(&cfg))?;
    }
    Ok(())
}

fn retained_request_case<P: Lfm2Shape>(cfg: &Config) -> Result<()> {
    use lloom::nn::branch::BranchSession;
    let dev = Device::Cpu;
    let model = build::<P>(cfg, &dev)?;
    let mut carried_cfg = cfg.clone();
    carried_cfg.conv_prefill_ctx = true;
    let reference = build::<P>(&carried_cfg, &dev)?;
    let opts = DecodeOptions {
        prefill_tile: 2,
        ..DecodeOptions::new(1)
    };
    for trunk in [vec![], vec![3], vec![3, 9, 14, 21, 5, 30, 2]] {
        let mut branch = BranchSession::new(&model, cfg, &dev)?;
        let mut rb = Blocking::new(&[47], &dev)?;
        branch.begin(&mut rb, &trunk, &opts)?;
        let mut first = None;
        for q in 0..12u32 {
            // Last branch repeats the first after many destructive recurrent
            // updates and turn-ring evictions; suffix length exceeds K/slack.
            let arm = if q == 11 { 0 } else { q };
            let suffix: Vec<u32> = (0..13).map(|i| 1 + (arm * 5 + i * 7) % 46).collect();
            let got = branch.suffix(&mut rb, &suffix, &opts)?.to_vec1::<f32>()?;
            let mut fresh = Session::new(&reference, reference.new_cache(&carried_cfg, &dev)?);
            let mut reference_rb = Blocking::new(&[47], &dev)?;
            fresh.prefill(&mut reference_rb, &trunk, &opts)?;
            fresh.prefill(&mut reference_rb, &suffix, &opts)?;
            let expected = fresh.last_logits().unwrap().to_vec1::<f32>()?;
            assert_eq!(got.len(), 48);
            assert_eq!(expected.len(), 48);
            assert_eq!(got.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                "retained state changed phase/history or KV prefix: {:?}/{:?}, trunk={}, branch={q}",cfg.conv_state,cfg.kv_append,trunk.len());
            if q == 0 {
                first = Some(got);
            } else if q == 11 {
                assert_eq!(got, first.take().unwrap(), "later suffix overwrote trunk");
            }
        }
        branch.reset()?;
        assert!(branch.suffix(&mut rb, &[2], &opts).is_err());
    }
    // Explicit source/config control: construction does not flip the model flag.
    assert!(!cfg.conv_prefill_ctx);
    Ok(())
}
