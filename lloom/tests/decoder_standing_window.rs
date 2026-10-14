//! The standing window on the generic decoder (`DecoderState`'s
//! `CacheRewind`, `sequence-reuse-design.md` §2.3b): a state built with
//! `StateCapacity::window >= 1` leaves every decode step's window open until
//! the next forward, so the driver can undo exactly that step — the forward
//! the ring queued past an EOS it had not yet learned. Qwen4-exp's contract
//! (`qwen4exp_engine_identity.rs`'s `a_decode_step_is_undone_by_the_standing_window`)
//! on the topology every Qwen3.5 checkpoint runs on.
//!
//! The reference in every identity test is a run that never overran
//! (CONTRIBUTING §3.1b): its logits and its whole state against the
//! subject's after the rewind, bit for bit. The controls: a state built
//! without the window rewinds nothing, and through `Session::prepare` it
//! resets where the windowed one extends.

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen3_5::Config;
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{DecoderState, HybridState, Layer, PreNormState, StateCapacity};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::{CrownExact, Profile, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::{Device, Result, Tensor};

/// Both tiny variants' layer state: the hybrid mixer's, no FFN state.
type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn tokens(n: usize, vocab: usize, seed: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| (i * 7 + seed) % vocab as u32)
        .collect()
}

fn dense<P: Profile>(
    dev: &Device,
    window: usize,
) -> Result<(lloom::models::qwen3_5::Qwen3_5<P>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        checkpoints: 2,
        window,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn moe<P: Profile>(
    dev: &Device,
    window: usize,
) -> Result<(lloom::models::qwen3_5::Qwen3_5Moe<P>, Config)> {
    let ckpt =
        TinyCheckpoint::<Qwen3_5MoeTiny>::new(Qwen3_5MoeTiny::template(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        checkpoints: 2,
        window,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn flat(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let (a, b) = (flat(a)?, flat(b)?);
    assert_eq!(a.len(), b.len(), "shapes differ");
    Ok(a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

/// Every layer of `a` against `b`: the largest difference over the
/// recurrent layers' `S` and conv windows, and over the attention layers'
/// live KV rows (their lengths must agree).
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

/// A fresh state with `prompt` prefilled in one forward.
fn prefill<M: CausalLM>(
    model: &M,
    cfg: &M::Config,
    dev: &Device,
    prompt: &[u32],
) -> Result<M::Cache> {
    let mut st = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    model.forward(&input, 0, &mut st)?;
    Ok(st)
}

/// One decode step of `tok` at the state's own position.
fn step<M, L>(model: &M, st: &mut DecoderState<L>, tok: u32, dev: &Device) -> Result<Tensor>
where
    M: CausalLM<Cache = DecoderState<L>>,
    L: Layer,
{
    let input = Tensor::from_vec(vec![tok], (1, 1), dev)?;
    let pos = st.pos;
    model.forward(&input, pos, st)
}

/// The step after an undone overrun is the never-overrun run's step: a
/// prefill, step `a`, the overrun `x`, `rewind(1)`, step `b` — against a
/// prefill, `a`, `b`. The logits bit for bit, then the whole state.
fn undo_case<M, L>(model: &M, cfg: &Config, dev: &Device, arm: &str) -> Result<()>
where
    M: CausalLM<Config = Config, Cache = DecoderState<L>>,
    L: Layer<State = LayerState>,
{
    let prompt = tokens(12, cfg.vocab_size, 3);
    let (a, x, b) = (5u32, 9u32, 7u32);

    let mut want = prefill(model, cfg, dev, &prompt)?;
    let want_a = step(model, &mut want, a, dev)?;
    let want_b = step(model, &mut want, b, dev)?;

    let mut st = prefill(model, cfg, dev, &prompt)?;
    assert!(st.keeps_standing_window(), "{arm}: built with a window");
    assert_eq!(st.rewind_horizon(), 0, "{arm}: a prefill leaves no horizon");
    let got_a = step(model, &mut st, a, dev)?;
    assert_eq!(
        flat(&got_a)?,
        flat(&want_a)?,
        "{arm}: a step under the standing window computes what a step without one does"
    );
    assert_eq!(
        st.rewind_horizon(),
        1,
        "{arm}: a decode step leaves a horizon of one"
    );
    assert!(
        !st.window_open(),
        "{arm}: the standing window is not a verify's"
    );
    // The overrun: the forward the ring queued past the EOS.
    let _ = step(model, &mut st, x, dev)?;
    assert_eq!(
        st.rewind_horizon(),
        1,
        "{arm}: the overrun's forward settled the step before it"
    );
    let pos = st.pos;
    let refused = st.rewind(2).expect_err("past the horizon").to_string();
    assert!(
        refused.contains("standing horizon of 1"),
        "{arm}: {refused}"
    );
    assert_eq!(
        (st.pos, st.rewind_horizon()),
        (pos, 1),
        "{arm}: a refused rewind changes nothing"
    );
    st.rewind(1)?;
    assert_eq!(st.rewind_horizon(), 0, "{arm}: the window is spent");
    let got_b = step(model, &mut st, b, dev)?;
    let d = max_abs_diff(&got_b, &want_b)?;
    eprintln!("STANDING {arm}: the step after the undone overrun vs never overrun: {d:e}");
    assert_eq!(
        flat(&got_b)?,
        flat(&want_b)?,
        "{arm}: the step after the undone overrun is the never-overrun run's, bit for bit"
    );
    assert_eq!(
        st.pos, want.pos,
        "{arm}: the undone overrun left no position"
    );
    let (rec, kv) = state_diff(&st, &want)?;
    assert_eq!(
        (rec, kv),
        (0.0, 0.0),
        "{arm}: every layer's state is the never-overrun run's"
    );
    Ok(())
}

/// Dense and MoE at the shipped `No` shape (an unpooled state, the window a
/// handle on the base), and dense at the `Ring` shape (the state on phase
/// rings, the window a phase), on the CPU.
#[test]
fn a_decode_step_is_undone_by_the_standing_window() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, 1)?;
    undo_case(&model, &cfg, &dev, "dense No")?;
    let (model, cfg) = dense::<RewindableExact>(&dev, 2)?;
    undo_case(&model, &cfg, &dev, "dense Ring")?;
    let (model, cfg) = moe::<CrownExact>(&dev, 1)?;
    undo_case(&model, &cfg, &dev, "moe No")?;
    Ok(())
}

/// What leaves no horizon: a prefill, a multi-token pass after a step (which
/// settles the step), a step inside a verify's window, and every step of a
/// state built without the window — the control the rewind tests read
/// against.
#[test]
fn a_prefill_a_verify_step_and_a_windowless_state_leave_no_horizon() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, 1)?;
    let prompt = tokens(12, cfg.vocab_size, 3);

    let mut st = prefill(&model, &cfg, &dev, &prompt)?;
    assert_eq!(st.rewind_horizon(), 0, "a prefill leaves no horizon");
    step(&model, &mut st, 5, &dev)?;
    assert_eq!(st.rewind_horizon(), 1);
    let more = Tensor::from_vec(vec![4u32, 6, 8], (1, 3), &dev)?;
    let pos = st.pos;
    model.forward(&more, pos, &mut st)?;
    assert_eq!(
        st.rewind_horizon(),
        0,
        "a multi-token pass settles the step and leaves no horizon of its own"
    );
    assert!(st.rewind(1).is_err());

    // Inside a verify's window the step is the verify's to undo.
    st.open_window()?;
    let pos = st.pos;
    step(&model, &mut st, 5, &dev)?;
    assert!(st.window_open(), "the verify's window stays the verify's");
    assert_eq!(
        st.rewind_horizon(),
        0,
        "a step inside a verify leaves no horizon"
    );
    assert!(st.rewind(1).is_err());
    st.rewind_window()?;
    assert_eq!(st.pos, pos);

    // Built without the window: nothing to undo, as before.
    let (model, cfg) = dense::<CrownExact>(&dev, 0)?;
    let mut st = prefill(&model, &cfg, &dev, &prompt)?;
    assert!(!st.keeps_standing_window());
    step(&model, &mut st, 5, &dev)?;
    assert_eq!(
        st.rewind_horizon(),
        0,
        "a windowless state keeps no horizon"
    );
    let refused = st.rewind(1).expect_err("nothing to rewind").to_string();
    assert!(refused.contains("standing horizon of 0"), "{refused}");
    Ok(())
}

/// What the window costs in memory. On the CPU the step writes the state in
/// place, so the window adds one buffer per recurrent channel — the spare
/// a step under a window writes into: exactly one more recurrent frame,
/// the KV unchanged. And it does not grow: many steps hold what one does.
#[test]
fn the_standing_window_holds_one_spare_frame_on_the_cpu() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let prompt = tokens(12, Config::tiny().vocab_size, 3);
    let held = |window: usize, steps: usize| -> Result<lloom::nn::decoder::StateBytes> {
        let (model, cfg) = dense::<CrownExact>(&dev, window)?;
        let mut st = prefill(&model, &cfg, &dev, &prompt)?;
        for i in 0..steps {
            step(&model, &mut st, 5 + i as u32, &dev)?;
        }
        Ok(model.held_bytes(&st))
    };
    let without = held(0, 1)?;
    let with = held(1, 1)?;
    let with_many = held(1, 9)?;
    let without_many = held(0, 9)?;
    let (model, _) = dense::<CrownExact>(&dev, 1)?;
    let frame = model.checkpoint_frame_bytes();
    eprintln!(
        "STANDING held after 1 step: without {without:?} with {with:?}; after 9: without \
         {without_many:?} with {with_many:?}; one recurrent frame {frame} B"
    );
    assert_eq!(
        without.recurrent, frame,
        "without the window: the slabs alone"
    );
    assert_eq!(
        with.recurrent,
        2 * frame,
        "with it: the slabs and one spare each"
    );
    assert_eq!(
        with.kv - without.kv,
        0,
        "the KV's window is its lengths: no bytes"
    );
    assert_eq!(
        (with_many.recurrent, without_many.recurrent),
        (with.recurrent, without.recurrent),
        "a step's window is released at the next: no growth"
    );
    Ok(())
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

fn sink(_: &[u32]) -> Result<()> {
    Ok(())
}

/// The driver's side, through `Session`: the ring with one forward of
/// lookahead overruns an EOS by one position, `prepare` reconciles it
/// through the standing window and the next turn EXTENDS — its stream
/// byte-identical to the conservative loop's (lookahead 0, the blocking
/// arm, no overrun). The control: the same run on a state built without the
/// window cannot take the overrun back, and `prepare` resets.
#[test]
fn an_eos_overrun_is_undone_and_the_next_turn_extends() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (model, cfg) = dense::<CrownExact>(&dev, 1)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let never: Vec<u32> = vec![u32::MAX];

    let stream = {
        let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
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
    assert_eq!(stream.len(), 24);
    // The EOS: the first id from the sixth on not seen earlier in the
    // stream, so the run stops exactly there after several decode steps,
    // each settling the window the one before it left standing.
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a stream with a fresh id somewhere");
    let eos_ids = vec![stream[cut]];
    eprintln!("STANDING stream {stream:?}; EOS {} at {cut}", stream[cut]);
    let turn2: Vec<u32> = vec![10, 20, 30];

    let reference = {
        let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
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

    let ring_turns = |window: usize| -> Result<(ReuseAction, Vec<u32>)> {
        let (model, cfg) = dense::<CrownExact>(&dev, window)?;
        let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let mut rb = Ring::new(&eos_ids, &dev, 128, EosRead::Poll(PollMechanism::default()))?
            .with_poll_budget(20_000);
        let opts = |n: usize| DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::per_token(),
            ..DecodeOptions::new(n)
        };
        let t1 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &opts(24),
            &mut sink,
        )?;
        assert!(t1.eos, "window {window}");
        assert_eq!(t1.ids, stream[..cut], "window {window}");
        assert_eq!(
            t1.stats.overrun, 1,
            "window {window}: the ring's lookahead overruns the EOS by one forward"
        );
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(s.overrun(), 0, "window {window}");
        let t2 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &p.suffix,
            &opts(8),
            &mut sink,
        )?;
        Ok((p.action, t2.ids))
    };

    let (action, ids) = ring_turns(1)?;
    assert_eq!(
        action,
        ReuseAction::Extend,
        "the overrun is undone and the next turn extends"
    );
    assert_eq!(
        ids, reference,
        "the continuation after the undone overrun is the conservative loop's, byte for byte"
    );
    // The control's stream is a different pass partition (one prefill of
    // the whole conversation), so only its action is asserted.
    let (action, ids) = ring_turns(0)?;
    eprintln!("STANDING control turn 2 {action:?}: {ids:?} (reference {reference:?})");
    // Without the window the overrun cannot be taken back; since PR992
    // `prepare` restores to the latest turn checkpoint and replays the
    // suffix instead of resetting (`Rewound(n)`, n the count rewound),
    // and before PR992 it was `Fresh`. Either way it is not `Extend`: the
    // window is what makes the next turn extend.
    assert!(
        matches!(action, ReuseAction::Rewound(_) | ReuseAction::Fresh),
        "the control: without the window the overrun cannot be taken back, got {action:?}"
    );
    assert_ne!(
        action,
        ReuseAction::Extend,
        "the control must not extend: only the window undoes the overrun"
    );
    Ok(())
}

/// The same on Metal, where the step is the fused decode (its spare is the
/// kernels' out-of-place write either way): the undone overrun bit for bit,
/// no memory beyond what a windowless state holds, and, with the census on
/// (`CANDLE_METAL_PROFILE=1`), the same dispatches per step.
#[cfg(feature = "metal")]
#[test]
fn the_standing_window_on_metal_undoes_a_step_at_no_extra_cost() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let (model, cfg) = dense::<CrownExact>(&dev, 1)?;
    undo_case(&model, &cfg, &dev, "dense No metal")?;
    let (model, cfg) = moe::<CrownExact>(&dev, 1)?;
    undo_case(&model, &cfg, &dev, "moe No metal")?;

    let prompt = tokens(12, cfg.vocab_size, 3);
    type Census = Option<Vec<(String, u64)>>;
    let run = |window: usize| -> Result<(lloom::nn::decoder::StateBytes, Census)> {
        let (model, cfg) = dense::<CrownExact>(&dev, window)?;
        let mut st = prefill(&model, &cfg, &dev, &prompt)?;
        step(&model, &mut st, 5, &dev)?;
        dev.synchronize()?;
        let census = dev.as_metal_device()?.dispatch_census().open().ok();
        step(&model, &mut st, 6, &dev)?;
        dev.synchronize()?;
        let counted = census.map(|c| c.close());
        Ok((model.held_bytes(&st), counted))
    };
    let (held0, census0) = run(0)?;
    let (held1, census1) = run(1)?;
    eprintln!("STANDING metal held: without {held0:?} with {held1:?}");
    assert_eq!(
        held1, held0,
        "the fused step's spare is there either way: the window holds nothing more"
    );
    match (census0, census1) {
        (Some(a), Some(b)) => {
            let total = |c: &[(String, u64)]| c.iter().map(|(_, n)| n).sum::<u64>();
            eprintln!(
                "STANDING metal dispatches per step: without {} with {}",
                total(&a),
                total(&b)
            );
            assert!(total(&a) > 0, "the census counted the step");
            assert_eq!(a, b, "the window adds no dispatch to a step");
        }
        _ => eprintln!("STANDING metal dispatch census off (CANDLE_METAL_PROFILE unset)"),
    }
    Ok(())
}
