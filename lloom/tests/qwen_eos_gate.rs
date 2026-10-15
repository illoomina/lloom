//! The shared EOS gate on the generic decoder's Qwen3.5 (#993's arm B,
//! `sequence-reuse-design.md` §2.3e): the GDN's gated decode step carries
//! its state into the buffer the host swaps in (`gdn.metal`), the KV's
//! gated appends are taken back by length, and `DecoderState`'s
//! `rewind_unwritten` moves the counters back.
//!
//! The subject is the crown's flow: the ring polled, the adaptive commit,
//! the fused publish, the gate. It is forced `EXTRA + 1` forwards past an
//! EOS, with no reader, so `max_tokens` stops it. That runs at every EOS
//! position in a span, at three `run_ahead`s (the buffers cut at different
//! steps), at the shipped `No` shape and at the `Ring` shape (the state on
//! phase rings). After `Session::reconcile`, the state is the blocking
//! reference's at the boundary, bit for bit: every recurrent layer's `S`
//! and conv window, the KV lengths and rows, the position. The next turn
//! extends to the reference's continuation.
//!
//! A KV that grows inside the gated forwards copies its rows ungated
//! (`nn::slab::Slab::grow_to`): the growth case sizes the KV just past the
//! boundary, so one growth lands in the overrun, and the prefix it carries
//! is compared as above. The family's KV is dense f32 only
//! (`QWEN3_5_CAPABILITIES.kv_quant`).
//!
//! The spare a gated step writes holds an older state, so a step that left
//! its output unwritten and was swapped in anyway would leave a stale state
//! behind, and the comparison sees it. The mutation harness's bare-return
//! mutant is that defect. The controls:
//! - the same run ungated cannot be reconciled (the window holds one step);
//! - armed but unfired, the run is the reference's (the gated variants are
//!   the plain kernels while the flag is clear).
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{
    DecodeRecovery, DecoderState, HybridState, Layer, PreNormState, StateCapacity,
};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::ops::KvPrecision;
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::{CrownExact, Profile, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result, Tensor};

type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;

const EXTRA: usize = 12;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

/// The KV's size and growth step, and its store, where a case sets them.
type Kv = Option<(usize, KvPrecision)>;

fn model<P: Profile>(dev: &Device, verify: usize, kv: Kv) -> Result<(Qwen3_5<P>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    let (kv_tokens, kv_precision) = kv.unwrap_or((cap.kv_tokens, cap.kv_precision));
    model.set_capacity(StateCapacity {
        kv_tokens,
        kv_precision,
        verify,
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
    });
    Ok((model, ckpt.cfg.clone()))
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

fn sink(_: &[u32]) -> Result<()> {
    Ok(())
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_dtype(lloom_tensor::DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// Every layer's state as bits: the recurrent layers' `S` and window, the
/// attention layers' length and live rows.
fn state_bits<L: Layer<State = LayerState>>(st: &DecoderState<L>) -> Result<Vec<Vec<u32>>> {
    let mut out = vec![vec![st.pos as u32]];
    for l in &st.layers {
        match &l.mixer {
            HybridState::A(g) => {
                for t in [g.s.get(g.layer), g.conv.get(g.layer)] {
                    out.push(match t {
                        Some(t) => bits(t)?,
                        None => Vec::new(),
                    });
                }
            }
            HybridState::B(kv) => {
                let n = kv.len();
                out.push(vec![n as u32]);
                let [k, v, _, _] = kv.state_views()?;
                for t in [k, v].into_iter().flatten() {
                    out.push(bits(&t.narrow(1, 0, n)?)?);
                }
            }
        }
    }
    Ok(out)
}

fn ring(eos: &[u32], dev: &Device) -> Result<Ring> {
    Ring::new(
        eos,
        dev,
        lloom::generation::readback::RING_HISTORY_SLOTS,
        EosRead::Poll(PollMechanism::default()),
    )
}

fn flow(n: usize, run_ahead: usize, gate: bool) -> DecodeOptions {
    DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        adaptive_commit: true,
        run_ahead,
        eos_gate: gate,
        ..DecodeOptions::new(n)
    }
}

/// What one case found.
#[derive(Debug, Default)]
struct Case {
    overrun: usize,
    fired: bool,
    unwritten: bool,
    state_matches: bool,
    /// The steps the reconciled state can still undo: none (its standing
    /// window's base is a gated step's).
    horizon: Option<usize>,
    reuse: Option<ReuseAction>,
    turn2_matches: bool,
}

/// One EOS position `k`, one `run_ahead`, gated or not, at profile `P`.
#[allow(clippy::too_many_arguments)]
fn case<P: Profile>(
    dev: &Device,
    verify: usize,
    kv: Kv,
    stream: &[u32],
    k: usize,
    run_ahead: usize,
    gate: bool,
    max_tokens: usize,
) -> Result<Case>
where
    Qwen3_5<P>:
        CausalLM<Config = Config, Cache = DecoderState<lloom::models::qwen3_5::Qwen3_5Layer<P>>>,
    lloom::models::qwen3_5::Qwen3_5Layer<P>: Layer<State = LayerState>,
{
    let (model, cfg) = model::<P>(dev, verify, kv)?;
    let prompt: Vec<u32> = PROMPT.to_vec();
    let eos = vec![stream[k]];
    let turn2: Vec<u32> = vec![10, 20, 30];
    let full: Vec<u32> = [prompt.clone(), stream[..k].to_vec(), turn2].concat();

    // The reference: blocking, nothing fed past the EOS; then its next turn.
    let mut reference = Session::new(&model, model.new_cache(&cfg, dev)?);
    let r1 = reference.decode(
        &mut IdentityProposer,
        &mut Blocking::new(&eos, dev)?,
        &mut greedy(),
        &prompt,
        &DecodeOptions::new(64),
        &mut sink,
    )?;
    assert_eq!(r1.ids, &stream[..k]);
    let want_state = state_bits(reference.sequence().cache())?;
    let p = reference.prepare(&full)?;
    assert_eq!(p.action, ReuseAction::Extend);
    let want_turn2 = reference
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[u32::MAX], dev)?,
            &mut greedy(),
            &p.suffix,
            &DecodeOptions::new(8),
            &mut sink,
        )?
        .ids;

    let mut s = Session::new(&model, model.new_cache(&cfg, dev)?);
    let mut rb = ring(&eos, dev)?;
    let out = s.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        &prompt,
        &flow(max_tokens, run_ahead, gate),
        &mut sink,
    )?;
    let mut c = Case {
        overrun: out.stats.overrun,
        fired: out.stats.eos_gate.is_some_and(|r| r.fired),
        unwritten: out.stats.overrun_unwritten,
        ..Case::default()
    };
    assert_eq!(out.ids, &stream[..k.min(max_tokens)], "the ids");
    if s.reconcile().is_err() {
        return Ok(c);
    }
    c.state_matches = state_bits(s.sequence().cache())? == want_state;
    c.horizon = Some(s.sequence().cache().rewind_horizon());
    let p = s.prepare(&full)?;
    c.reuse = Some(p.action);
    let got_turn2 = s
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[u32::MAX], dev)?,
            &mut greedy(),
            &p.suffix,
            &DecodeOptions::new(8),
            &mut sink,
        )?
        .ids;
    c.turn2_matches = got_turn2 == want_turn2;
    Ok(c)
}

const PROMPT: [u32; 5] = [1, 3, 5, 7, 9];

fn stream(dev: &Device) -> Result<Vec<u32>> {
    let (model, cfg) = model::<CrownExact>(dev, 0, None)?;
    let ids = Session::new(&model, model.new_cache(&cfg, dev)?)
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[u32::MAX], dev)?,
            &mut greedy(),
            &PROMPT,
            &DecodeOptions::new(32),
            &mut sink,
        )?
        .ids;
    Ok(ids)
}

/// The EOS positions to stop on: every id from the third on that is its
/// first occurrence in the stream (an earlier one would stop the run there).
fn cuts(stream: &[u32]) -> Vec<usize> {
    (2..stream.len())
        .filter(|&k| !stream[..k].contains(&stream[k]))
        .take(6)
        .collect()
}

/// What a gated run past an EOS leaves once reconciled.
fn assert_reconciled(at: &str, g: &Case) {
    assert_eq!(g.overrun, 1 + EXTRA, "{at}: the forced overrun");
    assert!(
        g.fired && g.unwritten,
        "{at}: the gate fired, the overrun unwritten"
    );
    assert!(
        g.state_matches,
        "{at}: the reconciled state is the reference's, bit for bit"
    );
    assert_eq!(g.horizon, Some(0), "{at}: no gated step left undoable");
    assert_eq!(
        g.reuse,
        Some(ReuseAction::Extend),
        "{at}: the next turn extends"
    );
    assert!(g.turn2_matches, "{at}: and continues as the reference does");
}

fn run_matrix<P: Profile>(shape: &str, verify: usize) -> Result<()>
where
    Qwen3_5<P>:
        CausalLM<Config = Config, Cache = DecoderState<lloom::models::qwen3_5::Qwen3_5Layer<P>>>,
    lloom::models::qwen3_5::Qwen3_5Layer<P>: Layer<State = LayerState>,
{
    pin();
    let dev = Device::new_metal(0)?;
    let stream = stream(&dev)?;
    let cuts = cuts(&stream);
    assert!(cuts.len() >= 3, "too few EOS positions in {stream:?}");
    for &k in &cuts {
        for run_ahead in [1usize, 3, 8] {
            let at = format!("{shape} EOS at {k}, run_ahead {run_ahead}");
            let g = case::<P>(
                &dev,
                verify,
                None,
                &stream,
                k,
                run_ahead,
                true,
                k + 1 + EXTRA,
            )?;
            eprintln!("QWEN-GATE {at}: gated {g:?}");
            assert_reconciled(&at, &g);
        }
        let at = format!("{shape} EOS at {k}");
        let u = case::<P>(&dev, verify, None, &stream, k, 8, false, k + 1 + EXTRA)?;
        eprintln!("QWEN-GATE {at}: ungated {u:?}");
        assert!(
            !u.unwritten && u.reuse.is_none(),
            "{at}: ungated, the window cannot take it back"
        );
        let a = case::<P>(&dev, verify, None, &stream, k, 8, true, k)?;
        eprintln!("QWEN-GATE {at}: armed, unfired {a:?}");
        assert!(!a.fired && a.overrun == 0, "{at}: the EOS never sampled");
        assert!(a.state_matches, "{at}: armed but unfired is the reference");
    }
    Ok(())
}

#[test]
fn the_gated_overrun_reconciles_to_the_reference_on_the_shipped_shape() -> Result<()> {
    run_matrix::<CrownExact>("No", 0)
}

#[test]
fn the_gated_overrun_reconciles_to_the_reference_on_phase_rings() -> Result<()> {
    run_matrix::<RewindableExact>("Ring", 2)
}

/// The KV sized and grown in steps of the boundary + 2 positions: the
/// reference never grows before the boundary, the gated run grows once
/// inside its overrun (the overrun appends `EXTRA + 1` > 2 positions past
/// it).
#[test]
fn a_kv_grow_inside_the_gated_overrun_keeps_the_prefix() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let stream = stream(&dev)?;
    let cuts = cuts(&stream);
    assert!(cuts.len() >= 3, "too few EOS positions in {stream:?}");
    let precision = KvPrecision::dense(lloom_tensor::DType::F32);
    for &k in &cuts[..3] {
        let kv = Some((PROMPT.len() + k + 2, precision));
        let at = format!("KV of {} at EOS {k}", PROMPT.len() + k + 2);
        let g = case::<CrownExact>(&dev, 0, kv, &stream, k, 8, true, k + 1 + EXTRA)?;
        eprintln!("QWEN-GATE grow {at}: gated {g:?}");
        assert_reconciled(&at, &g);
    }
    Ok(())
}
