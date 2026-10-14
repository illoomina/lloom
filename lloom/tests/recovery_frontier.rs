//! The recovery frontier under the adaptive commit (`sequence-reuse-design.md`
//! §2.3e): how far a sequence's persistent state runs past an EOS when the
//! stop flag the loop reads (`DecodeOptions::stop`) is raised late — by a
//! reader thread that observes the EOS and is slow to publish it (a blocked
//! streaming callback, `lloom-engine`'s reader runs the hook before it
//! raises the flag), or by a reader that is starved and polls slowly.
//!
//! The claim under test is the one #993's handoff calls a hypothesis: the
//! pacer bounds work in flight against GPU completion, not distance from the
//! stop, so `DecodeOptions::adaptive_lead` is not a bound on the overrun. A
//! finite `max_tokens` or a ring lap would end such a run too, and neither is
//! a bound on recoverable state, so the runs here are given room for neither
//! and the flag must be what stopped them.
//!
//! The control is the fixed cadence (`adaptive_commit` off, lookahead 1): the
//! loop learns the EOS itself, per step, so a late reader cannot move its
//! overrun off one. The model is the tiny Qwen3.5 on the generic decoder, at
//! the shape the engine serves (`No`, the standing window on), on Metal: the
//! fused decode step, whose next state is written out of place and swapped
//! in. It has no EOS gate, so every forward past the EOS writes its state;
//! the run records how much of it and what the session can still undo.
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::readback::RING_HISTORY_SLOTS;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{DecoderState, HybridState, PreNormState, StateCapacity};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::nn::{DecodeOptions, IdentityProposer, ReuseAction, Session};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type Model = Qwen3_5<CrownExact>;
type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;
type Layer = lloom::models::qwen3_5::Qwen3_5Layer<CrownExact>;

/// The decode budget: far past any overrun the delays below can cause, so a
/// run that is not stopped by the flag is a failure, not a cap.
const ROOM: usize = 3000;

/// The long delay: long enough that the host runs more than `adaptive_lead`
/// forwards inside it at this fixture's step time, which in a debug test
/// build is the host's encode (several ms a step), not the GPU's.
const LONG: Duration = Duration::from_millis(1000);

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn model(dev: &Device) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    // The engine's load: the standing window and two turn checkpoints.
    model.set_capacity(StateCapacity {
        window: 1,
        checkpoints: 2,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

fn sink(_: &[u32]) -> Result<()> {
    Ok(())
}

/// How the reader thread is late.
#[derive(Debug, Clone, Copy)]
enum Late {
    /// It polls every 100 µs and raises the flag as soon as it reads the EOS.
    No,
    /// It reads promptly, then holds the flag back for the delay after it has
    /// read the EOS: the engine reader's order, whose streaming hook runs on
    /// the batch the EOS ended before the next iteration raises the flag.
    Callback(Duration),
    /// It reads promptly until the EOS (the `.1`-th id after the run's start)
    /// is published, sleeps the delay, then reads it and raises the flag at
    /// once: a reader the scheduler starves at the moment it matters.
    Starved(Duration, usize),
}

/// What one reader-owned run past an EOS left.
#[derive(Debug)]
struct Overrun {
    produced: usize,
    overrun: usize,
    stopped_by_flag: bool,
    stopped_by_eos_watch: bool,
    /// Tokens the GPU had published past the EOS when the reader first saw
    /// it, and when it raised the flag.
    gpu_past_eos_at_seen: usize,
    gpu_past_eos_at_stop: usize,
    /// From the reader reading the EOS to the flag.
    stop_latency: Duration,
    /// The position, every attention layer's KV length, and the standing
    /// window's horizon after the run.
    pos: usize,
    kv_lens: Vec<usize>,
    horizon: usize,
    /// The run's median host time a step (recorded in debug and `profile`
    /// builds only; 0 otherwise).
    step_median: Duration,
}

/// One run on `rb` from `prompt` under `opts`, a reader thread raising the
/// flag as `late` says. The session is left as the run left it.
fn reader_run<'m>(
    session: &mut Session<'m, Model>,
    rb: &mut Ring,
    prompt: &[u32],
    mut opts: DecodeOptions,
    late: Late,
) -> Result<(Overrun, Vec<u32>)> {
    let mut reader = rb.reader().expect("a mapped ring");
    reader.resume_at_published();
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    opts.stop = Some(Arc::clone(&stop));
    let watcher = {
        let (stop, done) = (Arc::clone(&stop), Arc::clone(&done));
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut seen = Vec::new();
            let mut at_seen = None;
            let mut at_stop = None;
            let mut latency = Duration::ZERO;
            let mut starved = false;
            let start = reader.cursor();
            loop {
                let last = done.load(Ordering::Acquire);
                // Starved: descheduled from the moment the EOS is published.
                if let Late::Starved(d, n) = late {
                    let eos_out = reader.published().is_some_and(|p| p as usize > start + n);
                    if eos_out && !starved {
                        starved = true;
                        std::thread::sleep(d);
                    }
                }
                buf.clear();
                if reader.read_new(&mut buf).is_err() {
                    stop.store(true, Ordering::Release);
                    break;
                }
                seen.extend_from_slice(&buf);
                if reader.ended() {
                    let eos = reader.cursor();
                    let t = Instant::now();
                    at_seen = reader.published().map(|p| p as usize - eos);
                    if let Late::Callback(d) = late {
                        std::thread::sleep(d);
                    }
                    at_stop = reader.published().map(|p| p as usize - eos);
                    stop.store(true, Ordering::Release);
                    latency = t.elapsed();
                    break;
                }
                if last {
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
            (seen, at_seen, at_stop, latency)
        })
    };
    let out = session.decode(
        &mut IdentityProposer,
        rb,
        &mut greedy(),
        prompt,
        &opts,
        &mut sink,
    );
    done.store(true, Ordering::Release);
    let (seen, at_seen, at_stop, latency) = watcher.join().expect("reader thread");
    let out = out?;
    let st: &DecoderState<Layer> = session.sequence().cache();
    let kv_lens = st
        .layers
        .iter()
        .filter_map(|l: &LayerState| match &l.mixer {
            HybridState::B(kv) => Some(kv.len()),
            HybridState::A(_) => None,
        })
        .collect();
    Ok((
        Overrun {
            produced: out.stats.produced,
            overrun: out.stats.overrun,
            stopped_by_flag: out.stats.stopped_by_flag,
            stopped_by_eos_watch: out.stats.stopped_by_eos_watch,
            gpu_past_eos_at_seen: at_seen.unwrap_or(0),
            gpu_past_eos_at_stop: at_stop.unwrap_or(0),
            stop_latency: latency,
            pos: session.pos(),
            kv_lens,
            horizon: st.rewind_horizon(),
            step_median: {
                let mut v = out.stats.step_ns.clone();
                v.sort_unstable();
                Duration::from_nanos(v.get(v.len() / 2).copied().unwrap_or(0))
            },
        },
        seen,
    ))
}

/// The largest difference over every recurrent layer's `S` and conv window.
fn recurrent_diff(a: &DecoderState<Layer>, b: &DecoderState<Layer>) -> Result<f32> {
    let flat = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1::<f32>() };
    let mut d = 0f32;
    for (x, y) in a.layers.iter().zip(&b.layers) {
        if let (HybridState::A(p), HybridState::A(q)) = (&x.mixer, &y.mixer) {
            for (s, t) in [
                (p.s.get(p.layer), q.s.get(q.layer)),
                (p.conv.get(p.layer), q.conv.get(q.layer)),
            ] {
                let (Some(s), Some(t)) = (s, t) else {
                    panic!("a recurrent layer written on one side only")
                };
                for (u, v) in flat(s)?.iter().zip(flat(t)?.iter()) {
                    d = d.max((u - v).abs());
                }
            }
        }
    }
    Ok(d)
}

fn adaptive(n: usize) -> DecodeOptions {
    DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        adaptive_commit: true,
        run_ahead: 8,
        ..DecodeOptions::new(n)
    }
}

fn fixed(n: usize) -> DecodeOptions {
    DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        budget: CommitBudget::steps(8).expect("8 > 0"),
        ..DecodeOptions::new(n)
    }
}

/// The falsifier. On the adaptive commit the overrun follows the flag: with
/// the reader on time it is a few forwards, and with it late by [`LONG`] —
/// either way the reader is late — the host has run more than
/// `adaptive_lead` forwards past the EOS, every one of them written into the
/// state (no gate on this family): the KV lengths are the position, the
/// recurrent state is not the boundary's, and the standing window undoes one
/// of them, so the session cannot be reconciled and the next turn is
/// `Fresh`. On the fixed cadence the same late readers leave an overrun of
/// one, which the window undoes, and the next turn extends.
#[test]
fn the_adaptive_lead_does_not_bound_the_overrun_past_a_late_stop() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let never = vec![u32::MAX];

    let stream = Session::new(&model, model.new_cache(&cfg, &dev)?)
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&never, &dev)?,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(32),
            &mut sink,
        )?
        .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id in the stream to stop on");
    let eos = vec![stream[cut]];
    let want = stream[..cut].to_vec();
    eprintln!("FRONTIER stream {stream:?}; EOS {} at {cut}", eos[0]);

    // The boundary: the blocking arm feeds nothing past the EOS.
    let mut boundary = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let ref_out = boundary.decode(
        &mut IdentityProposer,
        &mut Blocking::new(&eos, &dev)?,
        &mut greedy(),
        &prompt,
        &DecodeOptions::new(ROOM),
        &mut sink,
    )?;
    assert_eq!(ref_out.ids, want);
    assert_eq!(ref_out.stats.overrun, 0);
    let turn2: Vec<u32> = vec![10, 20, 30];
    let full: Vec<u32> = [prompt.clone(), want.clone(), turn2].concat();

    let lead = adaptive(ROOM).adaptive_lead();
    let mut table = Vec::new();
    for (cadence, opts) in [("adaptive", adaptive(ROOM)), ("fixed", fixed(ROOM))] {
        for late in [Late::No, Late::Callback(LONG), Late::Starved(LONG, cut)] {
            let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
            let mut rb = Ring::new(
                &eos,
                &dev,
                RING_HISTORY_SLOTS,
                EosRead::Poll(PollMechanism::default()),
            )?
            .with_poll_budget(20_000);
            let (o, seen) = reader_run(&mut s, &mut rb, &prompt, opts.clone(), late)?;
            let at = format!("{cadence} {late:?}");
            let diff = recurrent_diff(s.sequence().cache(), boundary.sequence().cache())?;
            // A refused reconcile moves nothing; `prepare` then resets.
            let reconciled = s.reconcile().map_err(|e| e.to_string());
            let action = s.prepare(&full)?.action;
            eprintln!(
                "FRONTIER {at}: produced {} overrun {} (lead {lead}) stopped_by_flag {} | gpu past \
                 EOS: at seen {} at stop {} | stop latency {:?} | pos {} kv {:?} horizon {} | \
                 recurrent vs boundary {diff:e} | reconcile {reconciled:?} | next turn {action:?} \
                 | median step {:?}",
                o.produced,
                o.overrun,
                o.stopped_by_flag,
                o.gpu_past_eos_at_seen,
                o.gpu_past_eos_at_stop,
                o.stop_latency,
                o.pos,
                o.kv_lens,
                o.horizon,
                o.step_median,
            );
            assert_eq!(seen, want, "{at}: the reader's stream ends at the EOS");
            assert_eq!(o.produced, cut, "{at}: the ids end at the EOS");
            assert!(o.overrun >= 1, "{at}: the EOS's own forward ran");
            assert_eq!(
                o.pos,
                prompt.len() + cut + o.overrun,
                "{at}: the position is the prompt, the ids and the overrun"
            );
            for &kv in &o.kv_lens {
                assert_eq!(
                    kv, o.pos,
                    "{at}: every forward past the EOS appended its KV"
                );
            }
            assert!(diff > 0.0, "{at}: the recurrent state is past the boundary");
            assert_eq!(o.horizon, 1, "{at}: the standing window holds one step");
            table.push((at.clone(), o.overrun, reconciled.is_ok(), action));
            match (cadence, late) {
                ("fixed", _) => {
                    assert_eq!(o.overrun, 1, "{at}: the loop learns the EOS itself");
                    assert!(reconciled.is_ok(), "{at}: the window undoes one forward");
                    assert_eq!(action, ReuseAction::Extend, "{at}");
                }
                ("adaptive", Late::Callback(_) | Late::Starved(..)) => {
                    assert!(o.stopped_by_flag, "{at}: the flag, not the cap, stopped it");
                    assert!(
                        o.overrun > lead,
                        "{at}: a stop late by {LONG:?} lets the host past the lead"
                    );
                    assert!(reconciled.is_err(), "{at}: one step undoes one forward");
                    // `Fresh` here; a turn-checkpoint restore where the
                    // front end restores a failed reconcile (#946).
                    assert_ne!(action, ReuseAction::Extend, "{at}");
                }
                _ => assert!(o.stopped_by_flag, "{at}: the flag stopped it"),
            }
        }
    }
    eprintln!("FRONTIER summary (arm, overrun, reconciled, next turn): {table:?}");
    Ok(())
}

/// The bound the falsifier above finds missing, built (#993's arm D
/// prerequisite, `DecodeOptions::eos_watch`): the loop reads the ring before
/// each step and stops at a stop id the GPU has published. With the same
/// late readers, the overrun is at most `adaptive_lead + 1` and the watch,
/// not the flag, stopped the run; the ids and the reader's stream are the
/// boundary's. Without the gate the forwards past the EOS still write the
/// state, so this bounds them; it does not undo them.
#[test]
fn the_eos_watch_bounds_the_overrun_past_a_late_stop() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let never = vec![u32::MAX];
    let stream = Session::new(&model, model.new_cache(&cfg, &dev)?)
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&never, &dev)?,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(32),
            &mut sink,
        )?
        .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id in the stream to stop on");
    let eos = vec![stream[cut]];
    let want = stream[..cut].to_vec();
    let watched = DecodeOptions {
        eos_watch: Some(true),
        ..adaptive(ROOM)
    };
    let lead = watched.adaptive_lead();
    for late in [Late::No, Late::Callback(LONG), Late::Starved(LONG, cut)] {
        let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let mut rb = Ring::new(
            &eos,
            &dev,
            RING_HISTORY_SLOTS,
            EosRead::Poll(PollMechanism::default()),
        )?
        .with_poll_budget(20_000);
        let (o, seen) = reader_run(&mut s, &mut rb, &prompt, watched.clone(), late)?;
        let at = format!("watched {late:?}");
        eprintln!(
            "FRONTIER-WATCH {at}: produced {} overrun {} (lead {lead}) stopped_by_eos_watch {} \
             stopped_by_flag {} | pos {} | median step {:?}",
            o.produced, o.overrun, o.stopped_by_eos_watch, o.stopped_by_flag, o.pos, o.step_median,
        );
        assert_eq!(seen, want, "{at}: the reader's stream ends at the EOS");
        assert_eq!(o.produced, cut, "{at}: the ids end at the EOS");
        assert!(o.overrun >= 1, "{at}: the EOS's own forward ran");
        assert!(
            o.overrun <= lead + 1,
            "{at}: the watch bounds the overrun by the lead, however late the reader"
        );
        assert_eq!(o.pos, prompt.len() + cut + o.overrun, "{at}: the position");
        if matches!(late, Late::Callback(_) | Late::Starved(..)) {
            assert!(
                o.stopped_by_eos_watch && !o.stopped_by_flag,
                "{at}: the watch, not the late flag, stopped it"
            );
        }
        // The ring again, as serve's pool reuses it: the watch starts at this
        // run's first token, not at the last run's EOS.
        let mut again = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let (o2, seen2) = reader_run(&mut again, &mut rb, &prompt, watched.clone(), Late::No)?;
        eprintln!(
            "FRONTIER-WATCH {at}, the ring reused: produced {} overrun {}",
            o2.produced, o2.overrun
        );
        assert_eq!(seen2, want, "{at}, the ring reused: the stream");
        assert_eq!(o2.produced, cut, "{at}, the ring reused: the ids end at the EOS");
        assert!(o2.overrun <= lead + 1, "{at}, the ring reused: bounded");
    }
    Ok(())
}
