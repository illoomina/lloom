//! Qwen4-exp's filled ring on the serving path: `--readback ring-fixed` (the
//! ring on the fixed cadence, the PLE fed by the driver's host fill), served
//! as `app/qwen4exp.rs` serves it. A serving process runs every request on
//! one sequence and one pooled ring (`DecodeSession::from_parts` /
//! `into_parts`), so the fill's producer reads a ring whose stamps ran across
//! earlier runs, and the next request reconciles a run's lookahead overrun
//! through the standing window. The host-fill suite
//! (`lloom/tests/qwen4exp_host_feed.rs`) builds a ring per run and meets
//! neither.
//!
//! On the tiny Qwen4-exp with the PLE, every reply on `ring-fixed` must equal
//! the blocking arm's on the same request order and the blocking arm's on a
//! fresh session, the fill must have fed every decode forward, and the EOS
//! gate must stay off (it rides the adaptive commit). The adaptive commit
//! asked for by name (`--readback ring-nonblocking`) is held to the same
//! replies, with the gate off as the family serves it (Qwen4-exp's kernels
//! carry no gated variant: `family::QWEN4EXP_CAPABILITIES`).
//!
//! Every run is watched by a timer: a wait nothing signals parks the GPU
//! queue, and the symptom is a hang, not a failure.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident,
};
use lloom::nn::model::CausalLM;
use lloom::nn::Sequence;
use lloom::profile::CrownExact;
use lloom_engine::{
    DecodeConfig, DecodeParams, DecodeSession, Decoded, ReadbackArm, ReadbackConfig, RingArm,
    RingPool, Stop,
};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result};
use std::fs::File;
use std::path::Path;
use std::time::Duration;

/// The longest a request order may take before it counts as parked.
const PARKED: Duration = Duration::from_secs(120);

/// One Metal device per thread (the host-fill suite's rule): every request
/// order runs on a thread of its own (`watched`).
fn metal() -> Device {
    thread_local! {
        static DEV: Device = Device::new_metal(0).expect("a Metal device");
    }
    DEV.with(|d| d.clone())
}

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg
}

/// A PLE table in a GGUF, as the loader names it.
fn write_table(cfg: &Config, rows: usize, path: &Path) -> Result<()> {
    let table = testing::wt((rows, cfg.ple_head_dim), 3.0, &Device::Cpu);
    let q = QTensor::quantize(&table, GgmlDType::F32)?;
    let mut f = File::create(path)?;
    gguf_file::write(&mut f, &[], &[("per_layer_token_embd.weight", &q)])?;
    let _ = Content::read(&mut File::open(path)?)?;
    Ok(())
}

/// The resident tiny model on this thread's Metal device, the PLE at layer 1
/// over the table at `path`.
fn model_with_ple(path: &Path) -> Result<Qwen4Exp<CrownExact, Resident>> {
    let cfg = tiny();
    let dev = metal();
    let mut model = testing::resident_model::<CrownExact>(&cfg, &dev);
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::<CrownExact>::new(
        testing::lin(hc_dim, ple_dim, 21.0, &dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::wt((hc_dim, 4), 23.0, &dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, PleTable::Mmap(MmapPleTable::open(path)?), block)?;
    Ok(model)
}

/// Run `f` on a thread and wait at most [`PARKED`] for it.
fn watched<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(PARKED) {
        Ok(v) => v,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: no answer in {PARKED:?} — a wait was left parked")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("{what}: the run panicked (above)")
        }
    }
}

/// The session group each arm runs with, and the arm its requests ask for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    /// `--readback blocking`.
    Blocking,
    /// `--readback ring-fixed`: the fixed cadence with the fill.
    Fixed,
    /// `--readback ring-nonblocking`: the adaptive commit, the EOS gate
    /// clamped off as the family clamps it.
    Adaptive,
}

impl Arm {
    fn config(self) -> DecodeConfig {
        let mut readback = ReadbackConfig::default();
        match self {
            Arm::Blocking => ReadbackConfig::apply_spelling(&mut readback, "blocking").unwrap(),
            Arm::Fixed => ReadbackConfig::apply_spelling(&mut readback, "ring-fixed").unwrap(),
            Arm::Adaptive => {
                ReadbackConfig::apply_spelling(&mut readback, "ring-nonblocking").unwrap();
                readback.eos_gate = false;
            }
        }
        DecodeConfig {
            readback,
            ..DecodeConfig::default()
        }
    }
}

/// One reply, as a consumer sees it.
#[derive(Debug, Clone, PartialEq)]
struct Reply {
    ids: Vec<u32>,
    text: String,
    streamed: String,
    stop: Stop,
}

/// What a request ran, beside what it returned.
struct Ran {
    reply: Reply,
    decoded: Decoded,
}

/// The detokenizer: one bracketed id per token, so a text is its ids.
fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

/// `requests` on one sequence and one pooled ring, as `app/qwen4exp.rs`
/// serves them: the sequence and the rings handed from each request's
/// session to the next.
fn serve(
    model: &Qwen4Exp<CrownExact, Resident>,
    arm: Arm,
    eos: u32,
    requests: &[(Vec<u32>, usize)],
) -> Result<Vec<Ran>> {
    let dev = metal();
    let mut seq = Some(Sequence::new(model.new_cache(&tiny(), &dev)?));
    let mut rings = Some(RingPool::new());
    let config = arm.config();
    let readback = config.readback.arm();
    let mut out = Vec::new();
    for (prompt, max_tokens) in requests {
        let mut session = DecodeSession::from_parts(
            model,
            seq.take().unwrap(),
            dev.clone(),
            rings.take().unwrap(),
        )
        .with_config(config.clone());
        let params = DecodeParams {
            max_tokens: *max_tokens,
            eos_ids: vec![eos],
            readback,
            ..DecodeParams::default()
        };
        let mut streamed = String::new();
        let decoded = session.decode_until(prompt, detok, &params, |f| {
            streamed.push_str(f);
            true
        })?;
        let (s, r) = session.into_parts();
        seq = Some(s);
        rings = Some(r);
        out.push(Ran {
            reply: Reply {
                ids: decoded.tokens.clone(),
                text: decoded.text.clone(),
                streamed,
                stop: decoded.stop,
            },
            decoded,
        });
    }
    Ok(out)
}

#[test]
fn every_served_request_on_the_filled_ring_is_the_blocking_arms_reply() -> Result<()> {
    let path = std::env::temp_dir().join(format!("lloom-829-served-{}.gguf", std::process::id()));
    write_table(&tiny(), 80, &path)?;
    let p1: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let p2: Vec<u32> = vec![7, 4, 6, 2, 9];

    // The EOS: the first token seen once past position 2 of p1's stream, so
    // a p1 request ends on it inside its budget.
    let stream = {
        let (path, p1) = (path.clone(), p1.clone());
        watched("stream", move || -> Result<Vec<u32>> {
            let model = model_with_ple(&path)?;
            Ok(serve(&model, Arm::Blocking, u32::MAX, &[(p1, 16)])?
                .remove(0)
                .reply
                .ids)
        })?
    };
    let at = (2..stream.len())
        .find(|&i| stream[..i].iter().all(|&t| t != stream[i]))
        .expect("a token first seen past position 2");
    let eos = stream[at];

    // Every EOS run followed by another request on the same ring, and a
    // conversation continued past the first EOS run's overrun: the EOS run,
    // its continuation, a length run, the EOS run again, the length run again.
    let p3: Vec<u32> = p1
        .iter()
        .chain(&stream[..at])
        .chain(&[eos, 4, 7])
        .copied()
        .collect();
    let requests: Vec<(Vec<u32>, usize)> = vec![
        (p1.clone(), 16),
        (p3, 6),
        (p2.clone(), 10),
        (p1, 16),
        (p2, 10),
    ];

    let run = |arm: Arm, requests: Vec<(Vec<u32>, usize)>| {
        let path = path.clone();
        watched(&format!("{arm:?}"), move || -> Result<Vec<Ran>> {
            let model = model_with_ple(&path)?;
            serve(&model, arm, eos, &requests)
        })
    };
    let want = run(Arm::Blocking, requests.clone())?;
    let fresh: Vec<Reply> = requests
        .iter()
        .map(|r| run(Arm::Blocking, vec![r.clone()]).map(|mut v| v.remove(0).reply))
        .collect::<Result<_>>()?;
    assert_eq!(
        want[0].reply.stop,
        Stop::Eos,
        "the EOS request must end on EOS"
    );
    assert_eq!(
        want[2].reply.stop,
        Stop::Length,
        "the length request must end on length"
    );
    assert!(
        want[1].decoded.prefilled < requests[1].0.len(),
        "the continued conversation must extend the cache: {:?}",
        want[1].decoded.reuse
    );
    for (i, (w, f)) in want.iter().zip(&fresh).enumerate() {
        assert_eq!(
            w.reply, *f,
            "blocking request {i}: the served order moved the reply"
        );
        assert!(
            w.decoded.stats.host_feed.is_none(),
            "blocking request {i} took a fill"
        );
        assert_eq!(
            w.reply.streamed, w.reply.text,
            "blocking request {i}: the stream"
        );
    }

    for arm in [Arm::Fixed, Arm::Adaptive] {
        let got = run(arm, requests.clone())?;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let at = format!("{arm:?} request {i}");
            assert_eq!(g.reply, w.reply, "{at}: the reply");
            let feed = g
                .decoded
                .stats
                .host_feed
                .as_ref()
                .unwrap_or_else(|| panic!("{at}: the ring run took no host fill"));
            assert!(
                feed.forwards > 0 && feed.fills >= feed.forwards,
                "{at}: {feed:?}"
            );
            match arm {
                Arm::Fixed => {
                    assert_eq!(g.decoded.readback, ReadbackArm::RING_FIXED, "{at}");
                    assert!(
                        g.decoded.stats.eos_gate.is_none(),
                        "{at}: the gate rode the fixed cadence"
                    );
                }
                _ => {
                    assert_eq!(
                        g.decoded.readback,
                        ReadbackArm::Ring(RingArm::CROWN),
                        "{at}"
                    );
                    assert!(
                        g.decoded.stats.eos_gate.is_none(),
                        "{at}: the gate rode, clamped off"
                    );
                }
            }
        }
        // The conversation continued past the EOS run's overrun extends the
        // cache (the standing window undoes the one written forward) on the
        // fixed cadence, as on the blocking arm.
        if arm == Arm::Fixed {
            assert_eq!(
                got[1].decoded.prefilled, want[1].decoded.prefilled,
                "{arm:?}: the reuse"
            );
        }
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
