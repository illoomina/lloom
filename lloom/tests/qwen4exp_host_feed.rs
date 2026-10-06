//! The PLE's host fill on the decode driver (`design/docs/expert-streaming.md`
//! §20.13, option A). On a ring arm the tiny Qwen4-exp model's decode
//! forwards wait on the GPU for the rows a producer thread fills from the
//! token before them, where the blocking arm hashes a host-side id. The rows
//! are the table's bytes through the same math either way, so:
//!
//! - every ring arm returns the blocking arm's ids, on the fixed cadence at
//!   lookahead 0 and 1 and under the adaptive commit, with EOS or without,
//!   and each reports a fill per forward;
//! - a second turn after a filled run decodes what it decodes after a
//!   blocking one: the n-gram window the fed forwards skipped is caught up;
//! - a forward whose token the loop has not fed is refused before its wait
//!   is encoded;
//! - a producer that fails releases every wait: the run returns an error in
//!   bounded time and the queue drains.
//!
//! Every run is watched by a timer: a wait nothing signals parks the GPU
//! queue, and the symptom is a hang, not a failure.
#![cfg(feature = "metal")]

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{
    Config, MmapPleTable, NgramHash, PleLayer, PleTable, Qwen4Exp, Resident, State,
};
use lloom::nn::model::{CausalLM, Model};
use lloom::nn::ngram_memory::{HostFeed, NgramWindow, RowTable};
use lloom::nn::{DecodeOptions, DecodeOut, FeedWait, IdentityProposer, Session, VarBuilder};
use lloom::profile::CrownExact;
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// The longest a run may take before it counts as parked.
const PARKED: Duration = Duration::from_secs(120);

/// One Metal device per thread: a tensor of one device cannot meet a tensor
/// of another, so a run's model, cache and strategy share one; and two runs
/// on one device would interleave their passes in one command stream, which
/// the per-device pass bracket does not serve. Every run here is a thread of
/// its own (`watched`).
fn metal() -> Device {
    thread_local! {
        static DEV: Device = Device::new_metal(0).expect("a Metal device");
    }
    DEV.with(|d| d.clone())
}

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-host-feed-{}-{tag}.gguf", std::process::id()))
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

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg
}

/// A resident tiny model on Metal with the PLE attached at layer 1 over the
/// table at `path` (the fixture of `qwen4exp_engine_identity`).
fn model_with_ple(cfg: &Config, path: &Path) -> Result<Qwen4Exp<CrownExact, Resident>> {
    let dev = metal();
    let mut model = testing::resident_model::<CrownExact>(cfg, &dev);
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

/// The readback arm a run takes.
#[derive(Clone, Copy, Debug)]
enum Arm {
    Blocking,
    /// The ring polled for EOS on the fixed cadence, `lookahead` 0 or 1.
    Fixed(usize),
    /// The ring reading its EOS flag back (`EosRead::Blocking`) on the fixed
    /// cadence: no host thread but the producer waits on the stamp.
    Flag(usize),
    /// The ring under the adaptive commit, the drain the loop's at the end.
    Adaptive,
}

enum Strategy {
    Blocking(Box<Blocking>),
    Ring(Box<Ring>),
}

fn options(arm: Arm, steps: usize) -> Result<DecodeOptions> {
    let mut opts = DecodeOptions::new(steps);
    match arm {
        Arm::Blocking => {}
        Arm::Fixed(lookahead) | Arm::Flag(lookahead) => {
            opts.lookahead = lookahead;
            opts.budget = CommitBudget::steps(4)?;
        }
        Arm::Adaptive => {
            opts.lookahead = 1;
            opts.budget = CommitBudget::steps(8)?;
            opts.adaptive_commit = true;
            opts.stream_via_reader = true;
        }
    }
    Ok(opts)
}

fn strategy(arm: Arm, eos: u32, dev: &Device) -> Result<Strategy> {
    Ok(match arm {
        Arm::Blocking => Strategy::Blocking(Box::new(Blocking::new(&[eos], dev)?)),
        Arm::Flag(_) => Strategy::Ring(Box::new(Ring::new(&[eos], dev, 64, EosRead::Blocking)?)),
        Arm::Fixed(_) | Arm::Adaptive => Strategy::Ring(Box::new(
            // A short poll budget: the tiny model's forward is below the
            // command-buffer cut, so a lookahead-1 poll falls back to the
            // blocking read rather than spinning on an uncommitted stamp.
            Ring::new(&[eos], dev, 64, EosRead::Poll(PollMechanism::default()))?
                .with_poll_budget(20_000),
        )),
    })
}

/// One decode of `prompt` (then of each `follow` turn on the same session)
/// on `arm`, greedy, `steps` tokens a turn at most. The ids of every turn.
fn decode_turns<M: CausalLM>(
    model: &M,
    cache: M::Cache,
    arm: Arm,
    eos: u32,
    prompt: &[u32],
    follow: &[&[u32]],
    steps: usize,
) -> Result<Vec<DecodeOut>> {
    decode_turns_waiting(model, cache, arm, None, eos, prompt, follow, steps)
}

/// [`decode_turns`] with the producer's wait chosen for the run.
#[allow(clippy::too_many_arguments)]
fn decode_turns_waiting<M: CausalLM>(
    model: &M,
    cache: M::Cache,
    arm: Arm,
    wait: Option<FeedWait>,
    eos: u32,
    prompt: &[u32],
    follow: &[&[u32]],
    steps: usize,
) -> Result<Vec<DecodeOut>> {
    let dev = metal();
    let mut session = Session::new(model, cache);
    let mut spec = IdentityProposer;
    let mut opts = options(arm, steps)?;
    opts.host_feed_wait = wait;
    // A run that asks for the parked producer parks the EOS wait too.
    opts.eos_park = wait.map(|w| matches!(w, FeedWait::Event { .. }));
    let mut outs = Vec::new();
    for turn in std::iter::once(prompt).chain(follow.iter().copied()) {
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut sink = |_: &[u32]| -> Result<()> { Ok(()) };
        let out = match strategy(arm, eos, &dev)? {
            Strategy::Blocking(mut rb) => {
                session.decode(&mut spec, &mut *rb, &mut sampler, turn, &opts, &mut sink)?
            }
            Strategy::Ring(mut rb) => {
                session.decode(&mut spec, &mut *rb, &mut sampler, turn, &opts, &mut sink)?
            }
        };
        outs.push(out);
    }
    Ok(outs)
}

/// Run `f` on a thread and wait at most [`PARKED`] for it: a wait nothing
/// signals parks the queue (until Metal's command-buffer timeout fails it),
/// and the run does not return.
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

/// The arms a filled run is checked on.
const RING_ARMS: [Arm; 3] = [Arm::Fixed(0), Arm::Fixed(1), Arm::Adaptive];

#[test]
fn every_ring_arm_fills_the_rows_the_blocking_arm_hashes() -> Result<()> {
    let path = temp_gguf("arms");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let steps = 12;
    // No EOS in the stream: every arm runs `steps` tokens.
    let never = u32::MAX;
    let want = {
        let (path, prompt) = (path.clone(), prompt.clone());
        watched("blocking", move || -> Result<Vec<u32>> {
            let model = model_with_ple(&tiny(), &path)?;
            let cache = model.new_cache(&tiny(), &metal())?;
            let out = decode_turns(&model, cache, Arm::Blocking, never, &prompt, &[], steps)?;
            assert!(
                out[0].stats.host_feed.is_none(),
                "the blocking arm took a host fill"
            );
            Ok(out[0].ids.clone())
        })?
    };
    assert_eq!(want.len(), steps);
    assert!(
        want.iter().any(|&t| t != want[0]),
        "a degenerate stream cannot tell the arms apart: {want:?}"
    );
    for arm in RING_ARMS {
        let (path, prompt) = (path.clone(), prompt.clone());
        let got = watched(&format!("{arm:?}"), move || -> Result<DecodeOut> {
            let model = model_with_ple(&tiny(), &path)?;
            let cache = model.new_cache(&tiny(), &metal())?;
            Ok(decode_turns(&model, cache, arm, never, &prompt, &[], steps)?.remove(0))
        })?;
        assert_eq!(got.ids, want, "{arm:?}: the filled rows moved the stream");
        let feed = got
            .stats
            .host_feed
            .as_ref()
            .unwrap_or_else(|| panic!("{arm:?}: the ring run took no host fill"));
        // One forward per token past the prompt's (the last token's forward
        // is the one whose logits go unused), each fed by its own fill.
        assert_eq!(feed.forwards, steps as u64, "{arm:?}: {feed:?}");
        assert!(feed.fills >= feed.forwards, "{arm:?}: {feed:?}");
        assert_eq!(got.stats.readback.ring_publishes, steps as u64, "{arm:?}");
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// The producer parks on the ring's publish event, at its stated class, and
/// says so in the run's report. On the fixed cadence at lookahead 0 a forward
/// is encoded only after its token is published, so the producer's waits are
/// parks (a spin only in the instant a token lands) and it never sleeps; the
/// loop's EOS wait parks on the same event. A spinning core costs the streamed tier's
/// host-planned pass ~30 ms a token of GPU time (`expert-streaming.md`
/// §20.14a/b), and a spawned thread is at `default` unless it claims
/// (`invariants.md` §15.2 #10b). The report carries what the producer did.
#[test]
fn the_producer_parks_on_the_publish_event_at_its_stated_class() -> Result<()> {
    let path = temp_gguf("class");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let want = {
        let (path, prompt) = (path.clone(), prompt.clone());
        watched("class-blocking", move || -> Result<Vec<u32>> {
            let model = model_with_ple(&tiny(), &path)?;
            let cache = model.new_cache(&tiny(), &metal())?;
            Ok(
                decode_turns(&model, cache, Arm::Blocking, u32::MAX, &prompt, &[], 10)?
                    .remove(0)
                    .ids,
            )
        })?
    };
    let out = watched("class", move || -> Result<DecodeOut> {
        let model = model_with_ple(&tiny(), &path)?;
        let cache = model.new_cache(&tiny(), &metal())?;
        let out = decode_turns_waiting(
            &model,
            cache,
            Arm::Fixed(0),
            Some(FeedWait::Event { idle_us: 50 }),
            u32::MAX,
            &prompt,
            &[],
            10,
        )?;
        std::fs::remove_file(&path).ok();
        Ok(out.into_iter().next().unwrap())
    })?;
    assert_eq!(out.ids, want, "the parked producer moved the stream");
    let feed = out.stats.host_feed.expect("the ring run took no host fill");
    assert_eq!(
        feed.poll,
        FeedWait::Event { idle_us: 50 },
        "the producer's wait: {feed:?}"
    );
    assert!(feed.parked > 0, "the producer never parked: {feed:?}");
    assert_eq!(
        feed.slept, 0,
        "a producer with an armed event slept: {feed:?}"
    );
    assert!(
        feed.demand_wakes > 0,
        "no forward woke the producer: {feed:?}"
    );
    // The loop's own EOS wait parks on the same event, every step, and never
    // takes the blocking read.
    let rb = out.stats.readback;
    assert!(rb.publish_parks > 0, "the EOS wait never parked: {rb:?}");
    assert_eq!(rb.poll_fallbacks, 0, "the EOS wait fell back: {rb:?}");
    #[cfg(target_os = "macos")]
    assert_eq!(
        (feed.qos, feed.qos_errno),
        (lloom::generation::QosClass::UserInteractive, 0),
        "the producer's class: {feed:?}"
    );
    assert_eq!(feed.forwards, 10);
    Ok(())
}

/// The producer's default wait is the one measured fastest end to end
/// (`expert-streaming.md` §20.14b): `WFE` while a forward waits on a value it
/// has not filled, a 50 µs sleep otherwise, at its stated class. The report
/// carries what the producer did.
#[test]
fn the_producer_waits_on_demand_by_default_at_its_stated_class() -> Result<()> {
    let path = temp_gguf("default");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let out = watched("default", move || -> Result<DecodeOut> {
        let model = model_with_ple(&tiny(), &path)?;
        let cache = model.new_cache(&tiny(), &metal())?;
        let out = decode_turns(&model, cache, Arm::Fixed(1), u32::MAX, &prompt, &[], 10)?;
        std::fs::remove_file(&path).ok();
        Ok(out.into_iter().next().unwrap())
    })?;
    let feed = out.stats.host_feed.expect("the ring run took no host fill");
    assert_eq!(feed.poll, FeedWait::OnDemand { idle_us: 50 }, "{feed:?}");
    assert_eq!(feed.parked, 0, "the default wait parked: {feed:?}");
    assert_eq!(
        out.stats.readback.publish_parks, 0,
        "the default EOS wait parked"
    );
    assert!(
        feed.spun + feed.slept > 0,
        "the producer never waited: {feed:?}"
    );
    #[cfg(target_os = "macos")]
    assert_eq!(
        (feed.qos, feed.qos_errno),
        (lloom::generation::QosClass::UserInteractive, 0),
        "the producer's class: {feed:?}"
    );
    Ok(())
}

/// The event is armed for the producer alone where nothing else waits on the
/// stamp: the flag-reading ring on the fixed cadence learns EOS by a blocking
/// read, and the producer still parks.
#[test]
fn the_producer_parks_where_it_is_the_only_waiter() -> Result<()> {
    let path = temp_gguf("only");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let out = watched("only", move || -> Result<DecodeOut> {
        let model = model_with_ple(&tiny(), &path)?;
        let cache = model.new_cache(&tiny(), &metal())?;
        let out = decode_turns_waiting(
            &model,
            cache,
            Arm::Flag(1),
            Some(FeedWait::Event { idle_us: 50 }),
            u32::MAX,
            &prompt,
            &[],
            10,
        )?;
        std::fs::remove_file(&path).ok();
        Ok(out.into_iter().next().unwrap())
    })?;
    let feed = out
        .stats
        .host_feed
        .expect("the flag ring took no host fill");
    assert!(feed.parked > 0, "the producer never parked: {feed:?}");
    assert_eq!(
        feed.slept, 0,
        "the producer slept with an event armed: {feed:?}"
    );
    assert!(
        feed.demand_wakes > 0,
        "no forward woke the producer: {feed:?}"
    );
    assert_eq!(out.stats.readback.publish_parks, 0, "the flag read parked");
    Ok(())
}

/// The event wait spins only for an imminent token and only for a bounded
/// number of retirements, parks otherwise, and sleeps only where the run armed
/// no event; a fixed arm is that arm.
#[test]
fn the_event_wait_spins_only_for_an_imminent_token() {
    use lloom::nn::{FeedStep, FEED_SPIN_BOUND};
    let w = FeedWait::Event { idle_us: 50 };
    let sleep = PollMechanism::Sleep(Duration::from_micros(50));
    let spin = FeedStep::Poll(PollMechanism::DEFAULT);
    assert_eq!(w.step(5, 4, 0, true), spin, "a forward waits on value 5");
    assert_eq!(
        w.step(5, 4, FEED_SPIN_BOUND, true),
        FeedStep::Park,
        "the spin is bounded"
    );
    assert_eq!(
        w.step(4, 4, 0, true),
        FeedStep::Park,
        "every demanded value is filled"
    );
    assert_eq!(
        w.step(4, 4, 0, false),
        FeedStep::Poll(sleep),
        "no event armed"
    );
    assert_ne!(spin, FeedStep::Poll(sleep), "the two arms must differ");
    let od = FeedWait::OnDemand { idle_us: 50 };
    assert_eq!(od.step(5, 4, 0, true), spin);
    assert_eq!(
        od.step(4, 4, 0, true),
        FeedStep::Poll(sleep),
        "on-demand never parks"
    );
    let fixed = FeedWait::Fixed(sleep);
    assert_eq!(fixed.step(9, 0, 0, true), FeedStep::Poll(sleep));
    assert_eq!("event".parse::<FeedWait>().unwrap(), w);
    assert_eq!(
        "event:10".parse::<FeedWait>().unwrap(),
        FeedWait::Event { idle_us: 10 }
    );
    assert_eq!("on-demand".parse::<FeedWait>().unwrap(), od);
    assert_eq!("sleep:50".parse::<FeedWait>().unwrap(), fixed);
    for x in [w, od, fixed] {
        assert_eq!(x.to_string().parse::<FeedWait>().unwrap(), x);
    }
    assert!("event:".parse::<FeedWait>().is_err());
}

/// With the host fill off the ring arm reads its input ids back: the same
/// stream, no fill — the A/B's second arm.
#[test]
fn with_the_host_fill_off_a_ring_arm_reads_its_ids_back() -> Result<()> {
    let path = temp_gguf("off");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let (on, off) = watched("off", move || -> Result<(DecodeOut, DecodeOut)> {
        let mut model = model_with_ple(&tiny(), &path)?;
        let dev = metal();
        let on = decode_turns(
            &model,
            model.new_cache(&tiny(), &dev)?,
            Arm::Fixed(1),
            u32::MAX,
            &prompt,
            &[],
            10,
        )?;
        model.set_host_fill(false);
        let off = decode_turns(
            &model,
            model.new_cache(&tiny(), &dev)?,
            Arm::Fixed(1),
            u32::MAX,
            &prompt,
            &[],
            10,
        )?;
        std::fs::remove_file(&path).ok();
        Ok((
            on.into_iter().next().unwrap(),
            off.into_iter().next().unwrap(),
        ))
    })?;
    assert_eq!(on.ids, off.ids);
    assert!(on.stats.host_feed.is_some());
    assert!(
        off.stats.host_feed.is_none(),
        "set_host_fill(false) still filled"
    );
    Ok(())
}

/// An EOS inside the stream: the fixed cadence at lookahead 1 enqueues the
/// forward that consumes the EOS before it learns it was one, and the
/// adaptive commit runs on to its cap — every one of those forwards waits for
/// a fill, so the producer fills past the EOS or the final drain parks.
#[test]
fn a_run_that_ends_on_eos_fills_the_forwards_past_it() -> Result<()> {
    let path = temp_gguf("eos");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
    let stream = {
        let (path, prompt) = (path.clone(), prompt.clone());
        watched("stream", move || -> Result<Vec<u32>> {
            let model = model_with_ple(&tiny(), &path)?;
            let cache = model.new_cache(&tiny(), &metal())?;
            Ok(
                decode_turns(&model, cache, Arm::Blocking, u32::MAX, &prompt, &[], 16)?
                    .remove(0)
                    .ids,
            )
        })?
    };
    // The first token that appears once in the stream past position 2, as
    // the EOS: the run stops before it.
    let at = (2..stream.len())
        .find(|&i| stream[..i].iter().all(|&t| t != stream[i]))
        .expect("a token first seen past position 2");
    let eos = stream[at];
    for arm in [Arm::Blocking, Arm::Fixed(0), Arm::Fixed(1), Arm::Adaptive] {
        let (path, prompt) = (path.clone(), prompt.clone());
        let got = watched(&format!("eos {arm:?}"), move || -> Result<DecodeOut> {
            let model = model_with_ple(&tiny(), &path)?;
            let cache = model.new_cache(&tiny(), &metal())?;
            Ok(decode_turns(&model, cache, arm, eos, &prompt, &[], 16)?.remove(0))
        })?;
        assert_eq!(got.ids, stream[..at], "{arm:?}");
        assert!(got.eos, "{arm:?}");
        if let Some(feed) = &got.stats.host_feed {
            assert!(feed.fills >= feed.forwards, "{arm:?}: {feed:?}");
        }
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// A second turn on the session after a filled run: its prompt is hashed
/// against the window the first turn's tokens left, which the fed forwards
/// never pushed — the model catches it up at the run's end. Stopped by the
/// cap, so no forward overran and the second turn extends the cache.
#[test]
fn a_second_turn_after_a_filled_run_hashes_the_window_the_first_left() -> Result<()> {
    let path = temp_gguf("turns");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    let run = |arm: Arm, path: PathBuf| {
        watched(
            &format!("turns {arm:?}"),
            move || -> Result<Vec<Vec<u32>>> {
                let model = model_with_ple(&tiny(), &path)?;
                let cache = model.new_cache(&tiny(), &metal())?;
                let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
                let out = decode_turns(&model, cache, arm, u32::MAX, &prompt, &[&[7, 4], &[2]], 6)?;
                Ok(out.into_iter().map(|o| o.ids).collect())
            },
        )
    };
    let want = run(Arm::Blocking, path.clone())?;
    for arm in [Arm::Fixed(0), Arm::Fixed(1)] {
        assert_eq!(run(arm, path.clone())?, want, "{arm:?}");
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// A forward armed for the host fill whose token the loop has not fed is
/// refused before it encodes a wait: that wait could never be signalled.
#[test]
fn a_forward_whose_token_was_not_fed_is_refused() -> Result<()> {
    let path = temp_gguf("refused");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    watched("refused", move || -> Result<()> {
        let dev = metal();
        let model = model_with_ple(&tiny(), &path)?;
        let mut cache = model.new_cache(&tiny(), &dev)?;
        let prompt = Tensor::new(&[3u32, 5, 2, 8], &dev)?.unsqueeze(0)?;
        model.forward(&prompt, 0, &mut cache)?;
        let feed = model
            .host_feed(&mut cache)?
            .expect("a PLE on Metal offers a host fill");
        let next = Tensor::new(&[9u32], &dev)?.unsqueeze(0)?;
        let err = model
            .forward(&next, cache.pos, &mut cache)
            .expect_err("a forward nothing was fed for must be refused");
        assert!(
            err.to_string().contains("never be signalled"),
            "refused for the wrong reason: {err}"
        );
        assert_eq!(
            feed.link().demanded(),
            0,
            "the refused forward encoded a wait"
        );
        model.host_feed_end(&mut cache, &[])?;
        // Nothing was left parked: the queue drains.
        Tensor::zeros(4, DType::F32, &dev)?.to_vec1::<f32>()?;
        std::fs::remove_file(&path).ok();
        Ok(())
    })
}

/// The tiny model with a producer that cannot fill: its table has no host
/// read (`RowTable::Dense`), so the first fill fails.
struct FailingFill<'a>(&'a Qwen4Exp<CrownExact, Resident>);

impl Model for FailingFill<'_> {
    type Config = Config;
    fn from_weights(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("a test wrapper")
    }
}

impl CausalLM for FailingFill<'_> {
    type Cache = State<CrownExact>;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<State<CrownExact>> {
        self.0.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        CausalLM::embed(self.0, tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut State<CrownExact>,
    ) -> Result<Tensor> {
        CausalLM::forward_embeds(self.0, embeds, pos, cache)
    }
    fn forward(
        &self,
        tokens: &Tensor,
        pos: usize,
        cache: &mut State<CrownExact>,
    ) -> Result<Tensor> {
        self.0.forward(tokens, pos, cache)
    }
    fn prepare_pass(&self, cache: &mut State<CrownExact>, seq_len: usize) -> Result<()> {
        self.0.prepare_pass(cache, seq_len)
    }
    fn note_input_ids(&self, cache: &mut State<CrownExact>, ids: &[u32]) -> Result<()> {
        self.0.note_input_ids(cache, ids)
    }
    fn host_feed(&self, cache: &mut State<CrownExact>) -> Result<Option<HostFeed>> {
        let Some(real) = self.0.host_feed(cache)? else {
            return Ok(None);
        };
        let hash = self.0.ple_hash().expect("a PLE").clone();
        let dense = RowTable::Dense(Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?);
        let window = NgramWindow::new(hash.context_len());
        Ok(Some(HostFeed::new(
            Arc::clone(real.link()),
            Arc::new(dense),
            hash,
            window,
        )))
    }
    fn host_feed_end(&self, cache: &mut State<CrownExact>, published: &[u32]) -> Result<()> {
        self.0.host_feed_end(cache, published)
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
}

/// A producer that fails releases every wait the run encoded: the run
/// returns the error instead of parking the queue, the device drains, and
/// the sequence refuses to run on past it rather than read stale rows.
#[test]
fn a_producer_that_fails_releases_its_waits() -> Result<()> {
    let path = temp_gguf("fails");
    let cfg = tiny();
    write_table(&cfg, 80, &path)?;
    for arm in RING_ARMS {
        let path = path.clone();
        watched(&format!("fails {arm:?}"), move || -> Result<()> {
            let dev = metal();
            let inner = model_with_ple(&tiny(), &path)?;
            let model = FailingFill(&inner);
            let cache = model.new_cache(&tiny(), &dev)?;
            let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9, 3, 5];
            let err = decode_turns(&model, cache, arm, u32::MAX, &prompt, &[], 8)
                .err()
                .unwrap_or_else(|| panic!("{arm:?}: a run whose producer failed returned ids"));
            eprintln!("HOSTFEED {arm:?} failed run: {err}");
            let err = err.to_string();
            assert!(
                err.contains("host feed") || err.contains("gather_into on a dense table"),
                "{arm:?}: the run failed, but not on its producer: {err}"
            );
            // Every wait was released: nothing on the queue is parked.
            Tensor::zeros(4, DType::F32, &dev)?.to_vec1::<f32>()?;
            Ok(())
        })?;
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
