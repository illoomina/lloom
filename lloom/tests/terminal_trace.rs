//! The terminal-boundary diagnostic (`nn::terminal_trace`,
//! `design/measurements/issue-993.md` §14) on the tiny Qwen3.5 on Metal,
//! under the profile recorder: each run's line, against references the
//! trace does not compute.
//!
//! - The stop's step is the stream's own EOS position, found by a blocking
//!   reference run.
//! - The run's final retirement is checked against the device after a
//!   synchronize the test makes once the run has returned: every buffer the
//!   run committed or left open has then completed, and the latest GPU end
//!   among them is the retirement. Nothing else runs in between.
//! - The times are ordered as the events are, and the producer's stop is the
//!   arm's own: the fixed cadence reads the EOS itself, the adaptive commit
//!   stops on the reader's flag, a length stop on its count.
//!
//! Every process installs the recorder before anything reads the tuning
//! (nextest runs each test in its own).
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::readback::RING_HISTORY_SLOTS;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::terminal_trace;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::{Device, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

type Model = Qwen3_5<CrownExact>;

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(profile::enabled(), "the profile recorder is on");
}

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

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

fn sink(_: &[u32]) -> Result<()> {
    Ok(())
}

/// One run on a fresh poll ring with a reader thread that raises the stop
/// flag when it reads the EOS (the engine's reader, on time). Returns the
/// run's produced count; the session is left as the run left it.
fn run(
    session: &mut Session<'_, Model>,
    eos: &[u32],
    prompt: &[u32],
    mut opts: DecodeOptions,
    dev: &Device,
) -> Result<usize> {
    let mut rb = Ring::new(
        eos,
        dev,
        RING_HISTORY_SLOTS,
        EosRead::Poll(PollMechanism::default()),
    )?
    .with_poll_budget(20_000);
    let mut reader = rb.reader().expect("a mapped ring");
    reader.resume_at_published();
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    opts.stop = Some(Arc::clone(&stop));
    let watcher = {
        let (stop, done) = (Arc::clone(&stop), Arc::clone(&done));
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            loop {
                let last = done.load(Ordering::Acquire);
                buf.clear();
                if reader.read_new(&mut buf).is_err() || reader.ended() {
                    stop.store(true, Ordering::Release);
                    break;
                }
                if last {
                    break;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        })
    };
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        prompt,
        &opts,
        &mut sink,
    );
    done.store(true, Ordering::Release);
    watcher.join().expect("reader thread");
    Ok(out?.stats.produced)
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

/// The run's line, resolved after the test's synchronize, and the device's
/// own retirement: the latest GPU end among the buffers committed since
/// `since`.
fn line_after(dev: &Device, since: f64) -> Result<(serde_json::Value, f64)> {
    dev.synchronize()?;
    // A completion handler can run after the wait returns: the line resolves
    // once the run's last buffer has recorded.
    let t0 = std::time::Instant::now();
    let mut lines = Vec::new();
    while lines.is_empty() && t0.elapsed() < Duration::from_secs(5) {
        terminal_trace::flush();
        lines = terminal_trace::take_lines();
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(50));
    let retired = profile::command_buffer_records()
        .iter()
        .filter(|c| c.committed >= since)
        .map(|c| c.interval.end)
        .fold(f64::MIN, f64::max);
    assert_eq!(lines.len(), 1, "one resolved run: {lines:?}");
    let v: serde_json::Value = serde_json::from_str(&lines.remove(0)).expect("a JSON line");
    Ok((v, retired))
}

fn num(v: &serde_json::Value, k: &str) -> f64 {
    v[k].as_f64()
        .unwrap_or_else(|| panic!("{k} is a number: {v}"))
}

/// Ordered as the events are: the run starts, decodes, the stop's buffer
/// is committed in the decode and its GPU interval holds the publication
/// (a reader cannot see it before the buffer starts), the producer stops,
/// the run returns, and its retirement is the device's.
fn check(v: &serde_json::Value, retired_ref: f64, stop_by: &str, eos_step: Option<u64>) {
    let (start, decode, ret, retired) = (
        num(v, "start"),
        num(v, "decode"),
        num(v, "return"),
        num(v, "retired"),
    );
    assert!(
        start <= decode && decode <= num(v, "stop") && num(v, "stop") <= ret,
        "{v}"
    );
    assert_eq!(v["stop_by"], stop_by, "{v}");
    // The line prints nanoseconds: equal to within its rounding.
    assert!(
        (retired - retired_ref).abs() <= 1e-9,
        "the run's retirement is the device's last buffer ({retired_ref:.9}): {v}"
    );
    assert!(num(v, "last_submit") >= start, "{v}");
    match eos_step {
        Some(e) => {
            assert_eq!(
                v["eos_step"].as_u64(),
                Some(e),
                "the stop's step is the stream's EOS: {v}"
            );
            assert!(v["eos_buf_step"].as_u64().unwrap() >= e, "{v}");
            let (b0, b1) = (num(v, "eos_buf_start"), num(v, "eos_buf_end"));
            assert!(
                num(v, "eos_buf_committed") >= decode && b0 <= b1 && b1 <= retired,
                "{v}"
            );
            // Every arm here streams through a reader thread: it saw the
            // stop, and not before the buffer that published it started.
            let seen = num(v, "reader_eos");
            assert!(
                seen >= b0,
                "a reader saw the stop before its buffer started: {v}"
            );
        }
        None => assert!(v["eos_buf_end"].is_null() && v["eos"] == false, "{v}"),
    }
}

/// The fixed cadence (the producer reads the EOS), the adaptive commit (the
/// reader's flag) and a length stop, each on a fresh session, each line
/// against the device.
#[test]
fn each_run_s_terminal_line_is_the_device_s() -> Result<()> {
    install();
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
        .expect("a fresh id to stop on");
    let eos = vec![stream[cut]];
    dev.synchronize()?;
    terminal_trace::flush();
    terminal_trace::take_lines();

    for (arm, opts, by) in [
        ("fixed", fixed(3000), "eos_of_last"),
        ("adaptive", adaptive(3000), "flag"),
    ] {
        let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
        let since = profile::now_s();
        let produced = run(&mut s, &eos, &prompt, opts, &dev)?;
        assert_eq!(produced, cut, "{arm}");
        let (v, retired) = line_after(&dev, since)?;
        eprintln!("TRACE-TEST {arm}: {v}");
        check(&v, retired, by, Some(cut as u64 + 1));
    }
    let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let since = profile::now_s();
    run(&mut s, &never, &prompt, fixed(4), &dev)?;
    let (v, retired) = line_after(&dev, since)?;
    eprintln!("TRACE-TEST length: {v}");
    check(&v, retired, "length", None);
    // The blocking arm, ended on its count: it returns with its last
    // forward encoded and not committed (the held logits' forward), so its
    // retirement is the buffer that carries that work later.
    let mut s = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let since = profile::now_s();
    s.decode(
        &mut IdentityProposer,
        &mut Blocking::new(&never, &dev)?,
        &mut greedy(),
        &prompt,
        &DecodeOptions::new(6),
        &mut sink,
    )?;
    let (v, retired) = line_after(&dev, since)?;
    eprintln!("TRACE-TEST blocking length: {v}");
    assert!(
        v["open_dispatches"].as_u64().unwrap() > 0,
        "the case this arm exists for: work left encoded at the return: {v}"
    );
    assert!(num(&v, "retired") > num(&v, "return"), "{v}");
    check(&v, retired, "length", None);
    Ok(())
}
