//! A failed request, then the next one, on one live session through the
//! serving path (`DecodeSession::decode_until`, one sequence and one pooled
//! ring across requests), under the EOS gate and the EOS watch, each on or
//! off, on the adaptive commit (#993; the maintainer's review 6178758177
//! item 2 and 6188708788 §6).
//!
//! What a server does with a failed request is what the driver below does:
//! the error goes to the client, and the request's lease resets the session
//! before the next request (`app::request_slot`). The failure is injected
//! into the model's forward at a chosen decode step past the reply's EOS,
//! while the reader is slow: an error out of the decode loop with the gate
//! armed (whether its flag is already set when the failing forward is
//! encoded is the device's race; the drain below holds either way).
//! Without the watch the slow reader leaves the loop running, so the
//! failure is three forwards past the EOS with gated forwards queued behind
//! it; the watch stops encoding a forward or two past the EOS, so there it
//! is the first forward past it, the one the watch cannot have stopped (it
//! is encoded before the EOS is published).
//!
//! Per arm, one session takes: a reply ending on EOS; the next request,
//! failing in its overrun; the same request again; a cancelled request (a
//! client gone: the hook's `false`, what the server's failed write gives);
//! its continuation; a length stop; its continuation; a stop string that
//! spans two ids' text; its continuation; and a sampled request. Every
//! request that succeeds returns the blocking arm's ids for the same
//! sequence of requests (the reference ran the same failure and reset where
//! it reached it). And when a gated run fails, its disarm has drained: the
//! device holds no committed buffer that has not completed and no encoded
//! dispatch that was not committed, so no gated work of the failed run can
//! run under the next run's cleared flag (PR1027's drain; its mutant fails
//! here).
//!
//! Run once per command API (`LLOOM_METAL_API=metal3 | metal4`); each line
//! names the API it ran on.
#![cfg(feature = "metal")]

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::{CausalLM, Model as ModelTrait, Provisional};
use lloom::nn::{ReuseAction, Sequence, VarBuilder};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{DecodeConfig, DecodeParams, DecodeSession, ReadbackConfig, RingPool, Stop};
use lloom_tensor::{Device, Result, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

type Inner = Qwen3_5<CrownExact>;

/// The tiny Qwen3.5 with a fault: armed with `n`, its `n`-th single-position
/// forward after that fails, as a device error would surface out of the
/// loop. Every other call is the model's own.
struct Faulty {
    inner: Inner,
    left: AtomicUsize,
}

impl Faulty {
    fn arm(&self, n: Option<usize>) {
        self.left.store(n.unwrap_or(0), Ordering::Relaxed);
    }
}

impl ModelTrait for Faulty {
    type Config = Config;
    fn from_weights(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("built from the tiny checkpoint")
    }
}

impl CausalLM for Faulty {
    type Cache = <Inner as CausalLM>::Cache;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        if embeds.dim(1)? == 1 {
            let left = self.left.load(Ordering::Relaxed);
            if left == 1 {
                self.left.store(0, Ordering::Relaxed);
                lloom_tensor::bail!("injected: the decode step's forward failed");
            }
            if left > 1 {
                self.left.store(left - 1, Ordering::Relaxed);
            }
        }
        self.inner.forward_embeds(embeds, pos, cache)
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.inner.prepare_pass(cache, seq_len)
    }
    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        self.inner.note_input_ids(cache, ids)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        self.inner.checkpoint_at(cache, pos)
    }
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: lloom::nn::CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        self.inner.open_checkpoint_request(cache, policy, end)
    }
    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.close_checkpoint_request(cache)
    }
    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        self.inner.pin_checkpoint(cache, pos)
    }
    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        self.inner.pins_checkpoints(cache)
    }
    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        self.inner.pinned_checkpoint(cache)
    }
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.settle_pending(cache)
    }
    fn provisional(&self, cache: &mut Self::Cache, mode: Provisional) -> Result<Provisional> {
        self.inner.provisional(cache, mode)
    }
    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        self.inner.settle_step(cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

fn model(dev: &Device) -> Result<(Faulty, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut inner = ckpt.model::<CrownExact>(dev)?;
    let cap = inner.capacity();
    inner.set_capacity(StateCapacity {
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
        ..cap
    });
    Ok((
        Faulty {
            inner,
            left: AtomicUsize::new(0),
        },
        ckpt.cfg.clone(),
    ))
}

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

const SEED: u64 = 7;
const SLOW: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Default)]
struct Req {
    prompt: Vec<u32>,
    max_tokens: usize,
    eos: Option<u32>,
    stop: Vec<String>,
    /// The consumer refuses the fragment after this many delivered ids.
    cancel_after: Option<usize>,
    /// The consumer holds the fragment that completes this many ids for
    /// [`SLOW`].
    slow_after: Option<usize>,
    /// The model's forward fails at this decode step.
    fail_at: Option<usize>,
    temperature: Option<f64>,
}

#[derive(Debug)]
struct Turn {
    /// `Err`'s text when the request failed.
    error: Option<String>,
    ids: Vec<u32>,
    text: String,
    stop: Option<Stop>,
    reuse: Option<ReuseAction>,
    overrun: usize,
    gated: bool,
    /// At a failure's return: the device held no committed, incomplete
    /// buffer and no uncommitted dispatch.
    drained: Option<bool>,
}

fn config(spelling: &str, gate: bool) -> DecodeConfig {
    let mut readback = ReadbackConfig::default();
    ReadbackConfig::apply_spelling(&mut readback, spelling).expect("a readback spelling");
    readback.eos_gate = gate;
    DecodeConfig {
        readback,
        ..DecodeConfig::default()
    }
}

/// Whether the device has finished everything the run encoded.
fn drained(dev: &Device) -> Result<bool> {
    let Device::Metal(m) = dev else {
        return Ok(true);
    };
    let open = m.open_dispatches();
    let incomplete = m
        .last_committed_marker()?
        .is_some_and(|mk| !mk.is_complete());
    Ok(open == 0 && !incomplete)
}

/// `requests` on one live session and ring pool, as a server runs them: a
/// failed request's lease resets the session before the next one.
fn serve(
    model: &Faulty,
    cfg: &Config,
    dev: &Device,
    spelling: &str,
    gate: bool,
    eos: u32,
    requests: &[Req],
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
        model.arm(r.fail_at);
        let mut delivered = 0usize;
        let result = session.decode_until(&r.prompt, detok, &params, |frag| {
            delivered += frag.matches('<').count();
            if r.slow_after.is_some_and(|n| delivered >= n) {
                std::thread::sleep(SLOW);
            }
            r.cancel_after.is_none_or(|n| delivered < n)
        });
        model.arm(None);
        let turn = match result {
            Ok(d) => Turn {
                error: None,
                ids: d.tokens.clone(),
                text: d.text.clone(),
                stop: Some(d.stop),
                reuse: Some(d.reuse),
                overrun: d.stats.overrun,
                gated: d.stats.eos_gate.is_some_and(|g| g.fired) && d.stats.overrun_unwritten,
                drained: None,
            },
            Err(e) => {
                let drained = drained(dev)?;
                // The lease: a request that did not complete resets the
                // session before the next one.
                session.reset();
                Turn {
                    error: Some(e.to_string()),
                    ids: Vec::new(),
                    text: String::new(),
                    stop: None,
                    reuse: None,
                    overrun: 0,
                    gated: false,
                    drained: Some(drained),
                }
            }
        };
        let (s, p) = session.into_parts();
        seq = Some(s);
        rings = Some(p);
        out.push(turn);
    }
    Ok(out)
}

/// The first id of `stream` from `from` on that is its first occurrence.
fn fresh(stream: &[u32], from: usize) -> usize {
    (from..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on")
}

/// One request's greedy stream on the blocking arm, never stopping.
fn stream_of(
    model: &Faulty,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    n: usize,
) -> Result<Vec<u32>> {
    let r = Req {
        prompt: prompt.to_vec(),
        max_tokens: n,
        ..Req::default()
    };
    Ok(serve(model, cfg, dev, "blocking", false, u32::MAX, &[r])?
        .remove(0)
        .ids)
}

/// The request sequence, built from the blocking arm's streams so every
/// stop lands where the scenario says.
fn requests(model: &Faulty, cfg: &Config, dev: &Device, watch: bool) -> Result<(u32, Vec<Req>)> {
    let p1: Vec<u32> = vec![1, 3, 5, 7, 9];
    let s1 = stream_of(model, cfg, dev, &p1, 32)?;
    let cut1 = fresh(&s1, 6);
    let eos = s1[cut1];
    let reply1 = s1[..cut1].to_vec();
    // The next request, whose own EOS comes early: its failure is injected
    // past it, inside the overrun (three forwards past it, where a slow
    // reader leaves the loop running; under the watch the first forward past
    // it, encoded before the EOS is published).
    let p2: Vec<u32> = [p1.clone(), reply1.clone(), vec![10, 20, 30]].concat();
    let s2 = stream_of(model, cfg, dev, &p2, 32)?;
    let cut2 = fresh(&s2, 3);
    let eos2 = s2[cut2];
    let reqs = vec![
        Req {
            prompt: p1.clone(),
            max_tokens: 3000,
            ..Req::default()
        },
        Req {
            prompt: p2.clone(),
            max_tokens: 3000,
            eos: Some(eos2),
            slow_after: Some(cut2),
            fail_at: Some(if watch { cut2 + 1 } else { cut2 + 3 }),
            ..Req::default()
        },
        Req {
            prompt: p2.clone(),
            max_tokens: 3000,
            eos: Some(eos2),
            ..Req::default()
        },
    ];
    Ok((eos, reqs))
}

/// The matrix for the arms this process can run: `gates` on the adaptive
/// commit, the watch as installed.
fn matrix(watch: bool) -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let api = lloom_tensor::tuning::tuning().metal.command_api.clone();
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let (eos, first) = requests(&model, &cfg, &dev, watch)?;

    // The reference, then the arms, over the same requests. The scenario's
    // later requests carry the ids each continuation was given, so they are
    // built from the reference's replies and every arm must match them.
    let want_first = serve(&model, &cfg, &dev, "blocking", false, eos, &first)?;
    let mut reqs = first.clone();
    let base = [reqs[2].prompt.clone(), want_first[2].ids.clone()].concat();
    let s4 = stream_of(
        &model,
        &cfg,
        &dev,
        &[base.clone(), vec![11, 12]].concat(),
        32,
    )?;
    // A cancellation after two delivered ids, and its continuation.
    reqs.push(Req {
        prompt: [base.clone(), vec![11, 12]].concat(),
        max_tokens: 3000,
        cancel_after: Some(2),
        ..Req::default()
    });
    let c4 = [base.clone(), vec![11, 12], s4[..2].to_vec(), vec![13, 14]].concat();
    reqs.push(Req {
        prompt: c4.clone(),
        max_tokens: 6,
        ..Req::default()
    });
    // A length stop at 6 (the line above), its continuation.
    let s5 = stream_of(&model, &cfg, &dev, &c4, 6)?;
    let c5 = [c4.clone(), s5.clone(), vec![15, 16]].concat();
    // A stop string over the second and third ids' text, its continuation.
    let s6 = stream_of(&model, &cfg, &dev, &c5, 32)?;
    let stop = format!("{}><{}", s6[1], s6[2]);
    reqs.push(Req {
        prompt: c5.clone(),
        max_tokens: 3000,
        stop: vec![stop.clone()],
        ..Req::default()
    });
    let c6 = [c5.clone(), s6[..1].to_vec(), vec![17, 18]].concat();
    reqs.push(Req {
        prompt: c6.clone(),
        max_tokens: 8,
        ..Req::default()
    });
    // A sampled request on the same session.
    reqs.push(Req {
        prompt: [c6.clone(), vec![19]].concat(),
        max_tokens: 8,
        temperature: Some(0.8),
        ..Req::default()
    });

    let want = serve(&model, &cfg, &dev, "blocking", false, eos, &reqs)?;
    for (i, t) in want.iter().enumerate() {
        assert!(
            i == 1 || t.error.is_none(),
            "the reference's request {i} failed: {:?}",
            t.error
        );
    }
    assert_eq!(want[3].stop, Some(Stop::Cancelled));
    assert_eq!(want[4].stop, Some(Stop::Length));
    assert_eq!(want[5].stop, Some(Stop::StopString));
    assert_eq!(
        want[5].text,
        format!("<{}><", s6[0]),
        "the text ends where the stop begins"
    );

    for gate in [false, true] {
        let got = serve(&model, &cfg, &dev, "ring-nonblocking", gate, eos, &reqs)?;
        let arm = format!("api {api} watch {watch} gate {gate}");
        for (i, t) in got.iter().enumerate() {
            eprintln!(
                "FAILURE-NEXT {arm} request {i}: error {:?} drained {:?} stop {:?} reuse {:?} overrun {} gated {} ids {}",
                t.error.as_deref().map(|e| &e[..e.len().min(60)]),
                t.drained,
                t.stop,
                t.reuse,
                t.overrun,
                t.gated,
                t.ids.len(),
            );
        }
        // The failure, injected past the reply's EOS, fires on every arm.
        let e = got[1]
            .error
            .as_deref()
            .unwrap_or_else(|| panic!("{arm}: the request did not fail"));
        assert!(e.contains("injected"), "{arm}: {e}");
        if gate {
            assert_eq!(
                got[1].drained,
                Some(true),
                "{arm}: a failed gated run leaves no gated work for the next run's cleared flag"
            );
        }
        // The stops the device decides (EOS, length) return the blocking
        // arm's ids; a host stop (a cancellation, a stop string) returns
        // what the loop produced before it read the flag, so its delivered
        // text is what is compared, and the continuation, which carries
        // what the client was given, returns the blocking arm's ids.
        for i in [0usize, 2, 4, 6, 7] {
            assert!(
                got[i].error.is_none(),
                "{arm} request {i}: {:?}",
                got[i].error
            );
            assert_eq!(
                got[i].ids, want[i].ids,
                "{arm} request {i}: the blocking arm's ids"
            );
            assert_eq!(got[i].stop, want[i].stop, "{arm} request {i}: the stop");
        }
        assert_eq!(got[3].stop, Some(Stop::Cancelled), "{arm}: the client gone");
        assert_eq!(
            got[5].stop,
            Some(Stop::StopString),
            "{arm}: the stop string"
        );
        assert_eq!(got[5].text, want[5].text, "{arm}: the stop string's text");
        if gate {
            assert!(got[0].gated, "{arm}: the first reply's overrun was gated");
        }
    }
    Ok(())
}

#[test]
fn a_failed_request_then_the_next_on_one_session_without_the_watch() -> Result<()> {
    matrix(false)
}

#[test]
fn a_failed_request_then_the_next_on_one_session_with_the_watch() -> Result<()> {
    lloom::tuning::install(lloom::tuning::Tuning {
        ring: lloom::tuning::RingTuning {
            eos_watch: true,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("the tuning is installed before anything reads it");
    matrix(true)
}
