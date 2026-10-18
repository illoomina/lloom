//! The recovery frontier on the serving path (`sequence-reuse-design.md`
//! §2.3e): a consumer that is slow to take a fragment, through
//! `DecodeSession::decode_until` as `lloom serve` drives it — one sequence
//! and one pooled ring across requests.
//!
//! The engine's reader raises the stop flag before a hook on a batch whose
//! read found EOS. A hook on an earlier batch still holds the reader back:
//! completing the reply's text does not mean EOS was already published.
//! On the adaptive commit the loop keeps encoding until it reads the flag;
//! on the fixed cadence it has
//! learnt the EOS itself and stopped. Without the EOS gate (the family's
//! default, `family::QWEN3_5_CAPABILITIES`) the forwards it runs past the
//! EOS write the state, and the next request on the session extends only
//! if the standing window can undo them all (one step). With the gate (the
//! `models.qwen3_5.eos_gate` arm, #993's arm B) they write nothing the state
//! keeps, and the next request extends whatever the overrun.
//!
//! Every reply is held to the blocking arm's.
#![cfg(feature = "metal")]

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::{ReuseAction, Sequence};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{DecodeConfig, DecodeParams, DecodeSession, ReadbackConfig, RingPool, Stop};
use lloom_tensor::{Device, Result};
use std::time::Duration;

type Model = Qwen3_5<CrownExact>;

/// How long the consumer holds the fragment that ends the reply's text.
const SLOW: Duration = Duration::from_millis(1000);

fn model(dev: &Device) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn config(spelling: &str, gate: bool) -> DecodeConfig {
    let mut readback = ReadbackConfig::default();
    ReadbackConfig::apply_spelling(&mut readback, spelling).expect("a readback spelling");
    // The family's clamp (`family::clamp_ungated_eos_gate`): off unless the
    // arm is selected.
    readback.eos_gate = gate;
    DecodeConfig {
        readback,
        ..DecodeConfig::default()
    }
}

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

/// One request: the whole conversation, the budget, the stop strings, and
/// the delivered ids after which the consumer cancels (none if `None`).
#[derive(Debug, Clone)]
struct Req {
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: Vec<String>,
    cancel_after: Option<usize>,
    /// This request's stop id, in place of the run's.
    eos: Option<u32>,
    /// Sample at this temperature (seed [`SEED`]); greedy if `None`.
    temperature: Option<f64>,
}

/// The sampled requests' seed: a draw is a function of the seed and the
/// absolute position (#943), so every arm draws the same ids.
const SEED: u64 = 7;

fn req(prompt: Vec<u32>, max_tokens: usize) -> Req {
    Req {
        prompt,
        max_tokens,
        stop: Vec::new(),
        cancel_after: None,
        eos: None,
        temperature: None,
    }
}

/// What one request returned and did.
#[derive(Debug)]
struct Turn {
    ids: Vec<u32>,
    text: String,
    stop: Stop,
    overrun: usize,
    stopped_by_flag: bool,
    /// The EOS gate fired and the overrun's forwards wrote nothing.
    gated: bool,
    stopped_by_eos_watch: bool,
    reuse: ReuseAction,
    prefilled: usize,
    /// The session's token log after the request.
    log: usize,
}

/// `requests` on one sequence and one ring pool, the consumer holding the
/// fragment that completes `slow_after` ids for `slow` (none if `None`).
#[allow(clippy::too_many_arguments)]
fn serve(
    model: &Model,
    cfg: &Config,
    dev: &Device,
    spelling: &str,
    gate: bool,
    eos: u32,
    requests: &[Req],
    slow_after: Option<usize>,
    slow: Duration,
) -> Result<Vec<Turn>> {
    let mut seq = Some(Sequence::new(model.new_cache(cfg, dev)?));
    let mut rings = Some(RingPool::new());
    let config = config(spelling, gate);
    let readback = config.readback.arm();
    let mut out = Vec::new();
    for r in requests {
        let mut session = DecodeSession::from_parts(
            model,
            seq.take().unwrap(),
            dev.clone(),
            rings.take().unwrap(),
        )
        .with_config(config.clone());
        let params = DecodeParams {
            max_tokens: r.max_tokens,
            eos_ids: vec![r.eos.unwrap_or(eos)],
            readback,
            stop_strings: r.stop.clone(),
            seed: SEED,
            temperature: r.temperature,
            ..DecodeParams::default()
        };
        let mut delivered = 0usize;
        let decoded = session.decode_until(&r.prompt, detok, &params, |frag| {
            delivered += frag.matches('<').count();
            if slow_after.is_some_and(|n| delivered >= n) {
                std::thread::sleep(slow);
            }
            !r.cancel_after.is_some_and(|n| delivered >= n)
        })?;
        let log = session.tokens().len();
        let (s, p) = session.into_parts();
        seq = Some(s);
        rings = Some(p);
        out.push(Turn {
            ids: decoded.tokens.clone(),
            text: decoded.text.clone(),
            log,
            stop: decoded.stop,
            overrun: decoded.stats.overrun,
            stopped_by_flag: decoded.stats.stopped_by_flag,
            gated: decoded.stats.eos_gate.is_some_and(|g| g.fired)
                && decoded.stats.overrun_unwritten,
            stopped_by_eos_watch: decoded.stats.stopped_by_eos_watch,
            reuse: decoded.reuse,
            prefilled: decoded.prefilled,
        });
    }
    Ok(out)
}

/// A reply ending on EOS, then the conversation continued, on the blocking
/// arm, the fixed ring and the adaptive commit, each with an on-time and a
/// slow consumer. The fixed ring overruns one forward whatever the consumer
/// does and the next request extends; the adaptive commit with the slow
/// consumer runs past its lead and the next request is `Fresh`.
#[test]
fn a_slow_consumer_moves_the_adaptive_overrun_and_not_the_fixed_one() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];

    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    let reply = stream[..cut].to_vec();
    let next: Vec<u32> = [prompt.clone(), reply.clone(), vec![10, 20, 30]].concat();
    let requests = vec![req(prompt.clone(), 3000), req(next, 8)];

    let want = serve(
        &model, &cfg, &dev, "blocking", false, eos, &requests, None, SLOW,
    )?;
    assert_eq!(want[0].ids, reply);
    assert_eq!(want[0].stop, Stop::Eos);
    assert_eq!(
        want[1].reuse,
        ReuseAction::Extend,
        "the blocking arm extends"
    );
    // A `Fresh` request prefills the whole conversation in one pass, which a
    // DeltaNet family need not answer bit for bit as the extended one does
    // (`sequence-reuse-design.md` §2.3b): its reference is a fresh session's.
    let fresh = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        eos,
        &requests[1..],
        None,
        SLOW,
    )?
    .remove(0);
    assert_eq!(fresh.reuse, ReuseAction::Fresh);
    let lead = lloom::nn::DecodeOptions {
        adaptive_commit: true,
        run_ahead: 8,
        ..lloom::nn::DecodeOptions::new(1)
    }
    .adaptive_lead();

    for spelling in ["ring-fixed", "ring-nonblocking"] {
        for slow in [None, Some(cut)] {
            let got = serve(
                &model, &cfg, &dev, spelling, false, eos, &requests, slow, SLOW,
            )?;
            let at = format!("{spelling} slow={}", slow.is_some());
            eprintln!(
                "FRONTIER-SERVED {at}: overrun {} (lead {lead}) stopped_by_flag {} | next request \
                 {:?} prefilled {} (blocking {})",
                got[0].overrun,
                got[0].stopped_by_flag,
                got[1].reuse,
                got[1].prefilled,
                want[1].prefilled,
            );
            assert_eq!(got[0].ids, want[0].ids, "{at}: the reply");
            assert_eq!(got[0].stop, Stop::Eos, "{at}");
            let continued = if got[1].reuse == ReuseAction::Fresh {
                &fresh.ids
            } else {
                &want[1].ids
            };
            assert_eq!(&got[1].ids, continued, "{at}: the continued reply");
            match (spelling, slow) {
                ("ring-fixed", _) => {
                    assert_eq!(got[0].overrun, 1, "{at}: the loop learns the EOS itself");
                    assert_eq!(got[1].reuse, ReuseAction::Extend, "{at}");
                }
                (_, Some(_)) => {
                    assert!(got[0].stopped_by_flag, "{at}: the reader's flag stopped it");
                    assert!(
                        got[0].overrun > lead,
                        "{at}: a consumer slow by {SLOW:?} lets the host past the lead"
                    );
                    // `Fresh` here; a turn-checkpoint restore where the
                    // front end restores a failed reconcile (#946).
                    assert_ne!(got[1].reuse, ReuseAction::Extend, "{at}");
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// A host stop (a stop string, decided on the detokenized text) is not a
/// device-known EOS: the loop stops when it reads the flag the reader raised
/// after the hook said so, and every token it produced before that is in
/// the session's log, though the reply's text ends before the stop string.
/// The next request carries the text the client saw, so it diverges from
/// the log where the stop string began. Recorded per arm: the tokens the log
/// keeps past the stop, and what the next request reuses.
#[test]
fn a_host_stop_keeps_what_the_loop_produced_past_it() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let never = u32::MAX;
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        never,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let at = (3..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id for the stop string");
    let stop = format!("<{}>", stream[at]);
    let seen = &stream[..at];
    let first = Req {
        stop: vec![stop.clone()],
        ..req(prompt.clone(), 3000)
    };
    let next: Vec<u32> = [prompt.clone(), seen.to_vec(), vec![10, 20, 30]].concat();
    let requests = vec![first, req(next, 8)];
    for spelling in ["blocking", "ring-fixed", "ring-nonblocking"] {
        for slow in [None, Some(at)] {
            let got = serve(
                &model, &cfg, &dev, spelling, false, never, &requests, slow, SLOW,
            )?;
            let label = format!("{spelling} slow={}", slow.is_some());
            let text: String = seen.iter().map(|t| format!("<{t}>")).collect();
            eprintln!(
                "FRONTIER-HOSTSTOP {label}: stop {:?} at reply token {at} | ids {} log {} (prompt \
                 {} + kept {}) kept past the stop {} | next request {:?} prefilled {}",
                got[0].stop,
                got[0].ids.len(),
                got[0].log,
                prompt.len(),
                got[0].log - prompt.len(),
                (got[0].log - prompt.len()).saturating_sub(at),
                got[1].reuse,
                got[1].prefilled,
            );
            assert_eq!(got[0].stop, Stop::StopString, "{label}");
            assert_eq!(
                got[0].text, text,
                "{label}: the text ends before the stop string"
            );
            assert!(
                got[0].log >= prompt.len() + at,
                "{label}: the log holds at least what the client saw"
            );
        }
    }
    Ok(())
}

/// The same conversation on the adaptive commit with the EOS gate (#993's
/// arm B): the slow consumer still lets the loop past its lead (the gate
/// bounds no overrun), but the forwards past the EOS write nothing the state
/// keeps, so the session takes them back by their counters and the next
/// request extends, answering as the blocking arm's extended request does.
/// The on-time consumer likewise. The fixed cadence carries no gate (the
/// engine arms it under the adaptive commit only): its one forward past the
/// EOS is the standing window's, as without the arm.
#[test]
fn under_the_eos_gate_the_next_request_extends_past_any_overrun() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    let reply = stream[..cut].to_vec();
    let next: Vec<u32> = [prompt.clone(), reply.clone(), vec![10, 20, 30]].concat();
    let requests = vec![req(prompt.clone(), 3000), req(next, 8)];
    let want = serve(
        &model, &cfg, &dev, "blocking", false, eos, &requests, None, SLOW,
    )?;
    assert_eq!(want[0].ids, reply);
    assert_eq!(want[1].reuse, ReuseAction::Extend);
    let lead = lloom::nn::DecodeOptions {
        adaptive_commit: true,
        run_ahead: 8,
        ..lloom::nn::DecodeOptions::new(1)
    }
    .adaptive_lead();

    for spelling in ["ring-fixed", "ring-nonblocking"] {
        for slow in [None, Some(cut)] {
            let got = serve(
                &model, &cfg, &dev, spelling, true, eos, &requests, slow, SLOW,
            )?;
            let at = format!("gated {spelling} slow={}", slow.is_some());
            eprintln!(
                "FRONTIER-GATED {at}: overrun {} (lead {lead}) stopped_by_flag {} gated {} | next \
                 request {:?} prefilled {} (blocking {})",
                got[0].overrun,
                got[0].stopped_by_flag,
                got[0].gated,
                got[1].reuse,
                got[1].prefilled,
                want[1].prefilled,
            );
            assert_eq!(got[0].ids, want[0].ids, "{at}: the reply");
            assert_eq!(got[0].stop, Stop::Eos, "{at}");
            assert_eq!(
                got[1].reuse,
                ReuseAction::Extend,
                "{at}: the next request extends"
            );
            assert_eq!(
                got[1].prefilled, want[1].prefilled,
                "{at}: and prefills what blocking does"
            );
            assert_eq!(
                got[1].ids, want[1].ids,
                "{at}: and answers as blocking does"
            );
            let adaptive = spelling == "ring-nonblocking";
            assert_eq!(
                got[0].gated, adaptive,
                "{at}: the gate fires, the overrun unwritten, under the adaptive commit only"
            );
            if adaptive && slow.is_some() {
                assert!(
                    got[0].overrun > lead,
                    "{at}: the gate bounds no overrun; the slow consumer still passes the lead"
                );
            }
        }
    }
    Ok(())
}

/// One live session under the gate arm through every stop kind, a request
/// after each, the client sending back what it kept (#993's acceptance:
/// "continue the same live session after terminal EOS, length stop, host
/// stop, cancellation"). What it keeps is fixed by the turn, so every
/// request's prompt is the same on every run: an EOS or length reply whole;
/// nothing of a stop-string reply (the stop string `<` matches its first
/// text); the first id of a cancelled one (cancelled after one). How many
/// ids a stop-string or cancelled run returns past that depends on how the
/// reader batched them; the log's extra ids are the reuse's to undo. The
/// length turns stop on an id that never comes:
/// EOS → a length stop → a stop string → a cancellation → EOS → a length
/// stop. Every request completes with the stop it was forced to; the gate
/// fires on the EOS turns only, and the request after each EOS extends,
/// prefilling just what it adds. The first EOS turn and the turn after it
/// answer as the blocking arm's do. Each turn is recorded with its reuse.
#[test]
fn under_the_eos_gate_one_live_session_takes_every_stop_kind() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    // The conversation, turn by turn: each request is everything the client
    // has seen plus a user turn. A stop-string or cancelled reply is what the
    // client saw (its text, here its ids); the session's log may hold more.
    let mut seq = Some(Sequence::new(model.new_cache(&cfg, &dev)?));
    let mut rings = Some(RingPool::new());
    let config = config("ring-nonblocking", true);
    let readback = config.readback.arm();
    let mut convo = prompt.clone();
    let mut seen_eos_turns = 0usize;
    let mut extend_after_eos = 0usize;
    let mut last_was_eos = false;
    let kinds: [(&str, usize, Vec<String>, Option<usize>); 6] = [
        ("eos", 3000, vec![], None),
        ("length", 4, vec![], None),
        ("stop-string", 3000, vec!["<".into()], None),
        ("cancel", 3000, vec![], Some(1)),
        ("eos", 3000, vec![], None),
        ("length", 8, vec![], None),
    ];
    let mut record = Vec::new();
    for (i, (kind, max_tokens, stop, cancel_after)) in kinds.iter().enumerate() {
        let stop_id = if *kind == "length" { u32::MAX } else { eos };
        if i > 0 {
            convo.extend_from_slice(&[10 + i as u32, 20 + i as u32]);
        }
        let mut session = DecodeSession::from_parts(
            &model,
            seq.take().unwrap(),
            dev.clone(),
            rings.take().unwrap(),
        )
        .with_config(config.clone());
        let params = DecodeParams {
            max_tokens: *max_tokens,
            eos_ids: vec![stop_id],
            readback,
            stop_strings: stop.clone(),
            ..DecodeParams::default()
        };
        let mut delivered = 0usize;
        let decoded = session.decode_until(&convo, detok, &params, |frag| {
            delivered += frag.matches('<').count();
            !cancel_after.is_some_and(|n| delivered >= n)
        })?;
        let (s, p) = session.into_parts();
        seq = Some(s);
        rings = Some(p);
        let gated =
            decoded.stats.eos_gate.is_some_and(|g| g.fired) && decoded.stats.overrun_unwritten;
        eprintln!(
            "LIVE turn {i} {kind}: stop {:?} ids {} overrun {} gated {gated} | reuse {:?} prefilled {} of {}",
            decoded.stop,
            decoded.tokens.len(),
            decoded.stats.overrun,
            decoded.reuse,
            decoded.prefilled,
            convo.len(),
        );
        if last_was_eos {
            assert_eq!(
                decoded.reuse,
                ReuseAction::Extend,
                "turn {i}: the request after an EOS extends"
            );
            assert_eq!(
                decoded.prefilled, 2,
                "turn {i}: and prefills only the user turn"
            );
            extend_after_eos += 1;
        }
        let want_stop = match *kind {
            "eos" => Stop::Eos,
            "length" => Stop::Length,
            "stop-string" => Stop::StopString,
            _ => Stop::Cancelled,
        };
        // A forced EOS turn may meet its EOS later than the first turn's
        // position, or a length turn may meet it first: the kinds below
        // are what the conversation was driven to, and are asserted.
        assert_eq!(decoded.stop, want_stop, "turn {i} ({kind})");
        assert_eq!(
            gated,
            want_stop == Stop::Eos,
            "turn {i} ({kind}): the gate fires on an EOS only"
        );
        last_was_eos = want_stop == Stop::Eos;
        seen_eos_turns += last_was_eos as usize;
        record.push((i, kind.to_string(), decoded.tokens.clone(), decoded.reuse));
        // What the client kept goes back (above: fixed by the turn).
        let kept = match want_stop {
            Stop::StopString => &decoded.tokens[..0],
            Stop::Cancelled => &decoded.tokens[..decoded.tokens.len().min(1)],
            _ => &decoded.tokens[..],
        };
        convo.extend_from_slice(kept);
    }
    assert_eq!((seen_eos_turns, extend_after_eos), (2, 2));
    // The first EOS turn and the request after it, against the blocking arm.
    let first: Vec<u32> = [prompt.clone(), record[0].2.clone(), vec![11, 21]].concat();
    let want = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        eos,
        &[
            req(prompt.clone(), 3000),
            Req {
                eos: Some(u32::MAX),
                ..req(first, 4)
            },
        ],
        None,
        SLOW,
    )?;
    assert_eq!(
        record[0].2, want[0].ids,
        "the first EOS turn answers as blocking does"
    );
    assert_eq!(record[1].2, want[1].ids, "and so does the request after it");
    Ok(())
}

/// The gate on a pooled ring that has wrapped (#993's acceptance: "ring
/// wrapping"): a first request of more tokens than the ring's history
/// (`RING_HISTORY_SLOTS`) stops on its length, so the ring's stamp has run
/// past its slots when the second request, on the same ring and session,
/// meets its EOS. The second request's stop id is the first fresh id of the
/// blocking arm's continuation (a length stop leaves no overrun, so both
/// arms extend to the same stream). The gated adaptive arm's reply, its
/// reconcile and the third request match the blocking arm's.
#[test]
fn under_the_eos_gate_the_overrun_reconciles_on_a_wrapped_pooled_ring() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let slots = lloom::generation::readback::RING_HISTORY_SLOTS;
    let long = slots + 108;
    let never = u32::MAX;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let first = req(prompt.clone(), long);
    let reply1 = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        never,
        std::slice::from_ref(&first),
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    assert_eq!(reply1.len(), long, "the first request runs to its length");
    let next: Vec<u32> = [prompt.clone(), reply1.clone(), vec![10, 20, 30]].concat();
    let probe = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        never,
        &[first.clone(), req(next.clone(), 64)],
        None,
        SLOW,
    )?;
    let reply2 = &probe[1].ids;
    let k = (2..reply2.len())
        .find(|&i| !reply2[..i].contains(&reply2[i]))
        .expect("a fresh id in the continuation to stop on");
    let eos2 = reply2[k];
    let third: Vec<u32> = [next.clone(), reply2[..k].to_vec(), vec![40, 50]].concat();
    let requests = vec![
        first,
        Req {
            eos: Some(eos2),
            ..req(next, 3000)
        },
        Req {
            eos: Some(eos2),
            ..req(third, 8)
        },
    ];
    let want = serve(
        &model, &cfg, &dev, "blocking", false, never, &requests, None, SLOW,
    )?;
    assert_eq!(
        want[1].ids,
        &reply2[..k],
        "the blocking arm stops at the EOS"
    );
    assert_eq!(want[2].reuse, ReuseAction::Extend);
    for slow in [None, Some(k)] {
        let got = serve(
            &model,
            &cfg,
            &dev,
            "ring-nonblocking",
            true,
            never,
            &requests,
            slow,
            SLOW,
        )?;
        let at = format!("gated, the ring wrapped, slow={}", slow.is_some());
        eprintln!(
            "FRONTIER-WRAP {at}: first {:?} {} ids (ring {slots}) | second {:?} overrun {} gated {} | third {:?} prefilled {} (blocking {})",
            got[0].stop,
            got[0].ids.len(),
            got[1].stop,
            got[1].overrun,
            got[1].gated,
            got[2].reuse,
            got[2].prefilled,
            want[2].prefilled,
        );
        assert_eq!(got[0].ids, reply1, "{at}: the first reply");
        assert_eq!(
            got[1].ids, want[1].ids,
            "{at}: the second reply, past the wrap"
        );
        assert_eq!(got[1].stop, Stop::Eos, "{at}");
        assert!(got[1].gated, "{at}: the gate fired, the overrun unwritten");
        assert_eq!(
            got[2].reuse,
            ReuseAction::Extend,
            "{at}: the third request extends"
        );
        assert_eq!(got[2].prefilled, want[2].prefilled, "{at}");
        assert_eq!(
            got[2].ids, want[2].ids,
            "{at}: and answers as blocking does"
        );
    }
    Ok(())
}

/// The adaptive commit with the loop's own EOS watch (the `decode.ring.eos_watch`
/// arm, installed into this test's process before anything reads the
/// tuning): the consumer slow by 1 s no longer moves the overrun; the watch
/// stops the loop within `adaptive_lead + 1` forwards of the EOS while the
/// engine's reader is still in the hook, and the reader then finishes the
/// reply. Without the gate those forwards still write the state, so the
/// next request is not `Extend` unless the overrun is the window's one step;
/// it answers as its reuse says.
#[test]
fn the_eos_watch_bounds_the_served_overrun_past_a_slow_consumer() -> Result<()> {
    lloom::tuning::install(lloom::tuning::Tuning {
        ring: lloom::tuning::RingTuning {
            eos_watch: true,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("the tuning is installed before anything reads it");
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    let reply = stream[..cut].to_vec();
    let next: Vec<u32> = [prompt.clone(), reply.clone(), vec![10, 20, 30]].concat();
    let requests = vec![req(prompt.clone(), 3000), req(next, 8)];
    let want = serve(
        &model, &cfg, &dev, "blocking", false, eos, &requests, None, SLOW,
    )?;
    let fresh = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        eos,
        &requests[1..],
        None,
        SLOW,
    )?
    .remove(0);
    let lead = lloom::nn::DecodeOptions {
        adaptive_commit: true,
        run_ahead: 8,
        ..lloom::nn::DecodeOptions::new(1)
    }
    .adaptive_lead();
    for slow in [None, Some(cut)] {
        let got = serve(
            &model,
            &cfg,
            &dev,
            "ring-nonblocking",
            false,
            eos,
            &requests,
            slow,
            SLOW,
        )?;
        let at = format!("watched ring-nonblocking slow={}", slow.is_some());
        eprintln!(
            "FRONTIER-WATCH-SERVED {at}: overrun {} (lead {lead}) stopped_by_eos_watch {} \
             stopped_by_flag {} | next request {:?} prefilled {}",
            got[0].overrun,
            got[0].stopped_by_eos_watch,
            got[0].stopped_by_flag,
            got[1].reuse,
            got[1].prefilled,
        );
        assert_eq!(got[0].ids, want[0].ids, "{at}: the reply");
        assert_eq!(got[0].stop, Stop::Eos, "{at}");
        assert!(
            got[0].overrun <= lead + 1,
            "{at}: the watch bounds the overrun by the lead"
        );
        if slow.is_some() {
            assert!(
                got[0].stopped_by_eos_watch && !got[0].stopped_by_flag,
                "{at}: the watch, not the slow consumer's flag, stopped it"
            );
        }
        let continued = if got[1].reuse == ReuseAction::Fresh {
            &fresh.ids
        } else {
            &want[1].ids
        };
        assert_eq!(&got[1].ids, continued, "{at}: the continued reply");
    }
    Ok(())
}

/// Both arms (#993: the gate, `models.qwen3_5.eos_gate`'s effect, and the
/// watch, `decode.ring.eos_watch`): with a consumer slow by 1 s the watch
/// stops the loop within `adaptive_lead + 1` forwards of the EOS, and the
/// gate leaves those forwards unwritten, so the next request extends and
/// answers as the blocking arm's does. Neither arm alone does both.
#[test]
fn the_watch_and_the_gate_together_bound_the_overrun_and_keep_the_state() -> Result<()> {
    lloom::tuning::install(lloom::tuning::Tuning {
        ring: lloom::tuning::RingTuning {
            eos_watch: true,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("the tuning is installed before anything reads it");
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[req(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    let reply = stream[..cut].to_vec();
    let next: Vec<u32> = [prompt.clone(), reply.clone(), vec![10, 20, 30]].concat();
    let requests = vec![req(prompt.clone(), 3000), req(next, 8)];
    let want = serve(
        &model, &cfg, &dev, "blocking", false, eos, &requests, None, SLOW,
    )?;
    let lead = lloom::nn::DecodeOptions {
        adaptive_commit: true,
        run_ahead: 8,
        ..lloom::nn::DecodeOptions::new(1)
    }
    .adaptive_lead();
    for slow in [None, Some(cut)] {
        let got = serve(
            &model,
            &cfg,
            &dev,
            "ring-nonblocking",
            true,
            eos,
            &requests,
            slow,
            SLOW,
        )?;
        let at = format!("watched and gated slow={}", slow.is_some());
        eprintln!(
            "FRONTIER-BOTH {at}: overrun {} (lead {lead}) stopped_by_eos_watch {} gated {} | next request {:?} prefilled {} (blocking {})",
            got[0].overrun,
            got[0].stopped_by_eos_watch,
            got[0].gated,
            got[1].reuse,
            got[1].prefilled,
            want[1].prefilled,
        );
        assert_eq!(got[0].ids, want[0].ids, "{at}: the reply");
        assert!(
            got[0].overrun <= lead + 1,
            "{at}: the watch bounds the overrun"
        );
        assert!(got[0].gated, "{at}: the gate left it unwritten");
        assert_eq!(
            got[1].reuse,
            ReuseAction::Extend,
            "{at}: the next request extends"
        );
        assert_eq!(
            got[1].ids, want[1].ids,
            "{at}: and answers as blocking does"
        );
    }
    Ok(())
}

/// The composed arms with a sampled continuation (the maintainer's review:
/// "sampled — not only greedy — continuations"): at temperature 0.8 the
/// draws are keyed by position, so the blocking arm and the watched, gated
/// adaptive arm draw the same ids. With a consumer slow by 1 s the reply,
/// the reconcile and the next request's reply are the blocking arm's, and
/// the next request extends.
#[test]
fn the_watch_and_the_gate_together_keep_a_sampled_continuation() -> Result<()> {
    lloom::tuning::install(lloom::tuning::Tuning {
        ring: lloom::tuning::RingTuning {
            eos_watch: true,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("the tuning is installed before anything reads it");
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let sampled = |prompt: Vec<u32>, n: usize| Req {
        temperature: Some(0.8),
        ..req(prompt, n)
    };
    let stream = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
        false,
        u32::MAX,
        &[sampled(prompt.clone(), 32)],
        None,
        SLOW,
    )?
    .remove(0)
    .ids;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    let reply = stream[..cut].to_vec();
    let next: Vec<u32> = [prompt.clone(), reply.clone(), vec![10, 20, 30]].concat();
    let requests = vec![sampled(prompt.clone(), 3000), sampled(next, 8)];
    let want = serve(
        &model, &cfg, &dev, "blocking", false, eos, &requests, None, SLOW,
    )?;
    assert_eq!(
        want[0].ids, reply,
        "the blocking arm's sampled reply stops at the EOS"
    );
    let lead = lloom::nn::DecodeOptions {
        adaptive_commit: true,
        run_ahead: 8,
        ..lloom::nn::DecodeOptions::new(1)
    }
    .adaptive_lead();
    for slow in [None, Some(cut)] {
        let got = serve(
            &model,
            &cfg,
            &dev,
            "ring-nonblocking",
            true,
            eos,
            &requests,
            slow,
            SLOW,
        )?;
        let at = format!("sampled, watched and gated slow={}", slow.is_some());
        eprintln!(
            "FRONTIER-SAMPLED {at}: overrun {} (lead {lead}) stopped_by_eos_watch {} gated {} | next request {:?} prefilled {} (blocking {})",
            got[0].overrun,
            got[0].stopped_by_eos_watch,
            got[0].gated,
            got[1].reuse,
            got[1].prefilled,
            want[1].prefilled,
        );
        assert_eq!(got[0].ids, want[0].ids, "{at}: the sampled reply");
        assert!(
            got[0].overrun <= lead + 1,
            "{at}: the watch bounds the overrun"
        );
        assert!(got[0].gated, "{at}: the gate left it unwritten");
        assert_eq!(
            got[1].reuse,
            ReuseAction::Extend,
            "{at}: the next request extends"
        );
        assert_eq!(got[1].ids, want[1].ids, "{at}: and draws as blocking does");
    }
    Ok(())
}
