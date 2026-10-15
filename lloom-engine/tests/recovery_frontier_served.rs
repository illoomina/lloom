//! The recovery frontier on the serving path (`sequence-reuse-design.md`
//! §2.3e): a consumer that is slow to take a fragment, through
//! `DecodeSession::decode_until` as `lloom serve` drives it — one sequence
//! and one pooled ring across requests.
//!
//! The engine's reader runs the streaming hook on the batch it read and
//! raises the loop's stop flag at its next iteration, so a hook blocked on
//! the batch the EOS ended holds the flag back. On the adaptive commit the
//! loop keeps encoding until it reads the flag; on the fixed cadence it has
//! learnt the EOS itself and stopped. The tiny Qwen3.5 has no EOS gate
//! (`family::QWEN3_5_CAPABILITIES`), so the forwards it runs past the EOS
//! write the state, and the next request on the session extends only if
//! the standing window can undo them all (one step).
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

fn config(spelling: &str) -> DecodeConfig {
    let mut readback = ReadbackConfig::default();
    ReadbackConfig::apply_spelling(&mut readback, spelling).expect("a readback spelling");
    // The family's clamp: no gated kernels.
    readback.eos_gate = false;
    DecodeConfig {
        readback,
        ..DecodeConfig::default()
    }
}

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

/// One request: the whole conversation, the budget, the stop strings.
#[derive(Debug, Clone)]
struct Req {
    prompt: Vec<u32>,
    max_tokens: usize,
    stop: Vec<String>,
}

fn req(prompt: Vec<u32>, max_tokens: usize) -> Req {
    Req {
        prompt,
        max_tokens,
        stop: Vec::new(),
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
    eos: u32,
    requests: &[Req],
    slow_after: Option<usize>,
    slow: Duration,
) -> Result<Vec<Turn>> {
    let mut seq = Some(Sequence::new(model.new_cache(cfg, dev)?));
    let mut rings = Some(RingPool::new());
    let config = config(spelling);
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
            eos_ids: vec![eos],
            readback,
            stop_strings: r.stop.clone(),
            ..DecodeParams::default()
        };
        let mut delivered = 0usize;
        let decoded = session.decode_until(&r.prompt, detok, &params, |frag| {
            delivered += frag.matches('<').count();
            if slow_after.is_some_and(|n| delivered >= n) {
                std::thread::sleep(slow);
            }
            true
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

    let want = serve(&model, &cfg, &dev, "blocking", eos, &requests, None, SLOW)?;
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
            let got = serve(&model, &cfg, &dev, spelling, eos, &requests, slow, SLOW)?;
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
            let got = serve(&model, &cfg, &dev, spelling, never, &requests, slow, SLOW)?;
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
    let want = serve(&model, &cfg, &dev, "blocking", eos, &requests, None, SLOW)?;
    let fresh = serve(
        &model,
        &cfg,
        &dev,
        "blocking",
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
