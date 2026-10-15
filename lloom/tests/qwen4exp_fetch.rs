//! The fetch inside the token (`QWEN4EXP_FETCH=inside`,
//! `design/docs/expert-streaming.md` §20.14c): a device-planned decode step
//! whose router chose experts the pool does not hold waits on the device,
//! at each layer, for a worker that reads them beside the pass — and
//! decodes the ids and the logits the resident model decodes, bit for bit,
//! with no step replayed and none taken on the host plan.
//!
//! Metal only (the worker writes Shared slot pools and a live table the
//! device reads). Every test here sets the arm before anything reads the
//! tuning, so the whole binary runs under it.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, State, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::profile::{CrownExact, Profile};
use lloom::testing::fixture::Fixture;
use lloom_tensor::{Device, Result, Tensor};

fn arm() {
    std::env::set_var("QWEN4EXP_FETCH", "inside");
}

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: true,
        io_hints: false,
        drafter_bytes: 0,
    }
}

type Decoded<P> = (Vec<u32>, Vec<Vec<f32>>, State<P>);

fn decode<P: Profile, R: ExpertResidency>(
    model: &Qwen4Exp<P, R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<Decoded<P>> {
    let mut cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let (mut ids, mut rows) = (Vec::new(), Vec::new());
    for _ in 0..steps {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = row
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &v)| if v > m.1 { (i, v) } else { m },
            )
            .0 as u32;
        rows.push(row);
        ids.push(next);
        let input = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
    }
    // The last forward's logits are never read, so it may still be running
    // when the caller reads the fetch worker's counters, and the worker takes
    // a layer's job only once the layer before it was served: its last jobs
    // are not counted yet. Every counter the tests read is complete only once
    // the work it counts has run.
    dev.synchronize()?;
    Ok((ids, rows, cache))
}

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 24;

fn fixture(name: &str) -> Result<Option<(Fixture<Qwen4ExpTiny>, Device)>> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(None);
    };
    if lloom::testing::fixture::Backend::current().at_scale() {
        println!("at scale: the tiny model's fetch tests do not run the checkpoint");
        return Ok(None);
    }
    Ok(Fixture::<Qwen4ExpTiny>::new(name, Config::tiny())?.map(|fx| (fx, dev)))
}

/// 6 slots of 8 experts at top-2: steps miss. Every one fetches its misses
/// inside the step — the first because the cache is fresh, the rest because
/// the pool keeps missing — and none is replayed or host-planned: the
/// worker always finds a slot not routed to now (in place, its last reader
/// before the step). Each fetching step waits once a layer.
#[test]
fn a_fetching_decode_is_the_resident_stream_with_no_replay_metal() -> Result<()> {
    arm();
    let Some((fx, dev)) = fixture("qwen4exp_fetch_stream_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let (want_ids, want_rows) = {
        let resident = fx.resident::<CrownExact>(&cfg, &dev, &opts)?;
        let (ids, rows, _) = decode(&resident, &cfg, &dev, &PROMPT, STEPS)?;
        (ids, rows)
    };
    let streamed = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let (ids, rows, cache) = decode(&streamed, &cfg, &dev, &PROMPT, STEPS)?;
    let st = streamed.stream_stats();
    let f = streamed.fetch_stats();
    println!(
        "FETCHTEST stream steps={STEPS} passes={} entries={} exits={} redos={} fetch_redos={} host_plan_passes={} pre_waits={} jobs={} missed_layers={} loads={} short={} timeouts={} misses={} device_plans={}",
        cache.fetch_passes,
        cache.fetch_entries,
        cache.fetch_exits,
        cache.redos,
        cache.fetch_redos,
        cache.host_plan_passes,
        f.floor_waits,
        f.jobs,
        f.missed_layers,
        f.loads,
        f.short,
        f.timeouts,
        st.misses,
        st.device_plans
    );
    assert_eq!(ids, want_ids, "the fetching decode's ids differ");
    for (t, (a, b)) in rows.iter().zip(&want_rows).enumerate() {
        assert_eq!(a, b, "step {t}: the fetching decode's logits differ");
    }
    assert!(cache.fetch_passes > 0, "no step fetched");
    assert!(
        f.loads > 0,
        "the worker loaded nothing: the pool never missed"
    );
    assert_eq!(
        (cache.redos, cache.host_plan_passes, f.short, f.timeouts),
        (0, 0, 0, 0),
        "a step was replayed or host-planned (redos, host plans, shorts, timeouts)"
    );
    assert_eq!(
        f.jobs,
        cache.fetch_passes * cfg.n_layers as u64,
        "a fetching step waits once at every layer"
    );
    Ok(())
}

/// 8 slots of 8 experts: once the pool holds every expert the steps stop
/// fetching — after `fetch_exit` (8) clean ones the waits are gone and the
/// warm step is the plain device plan again.
#[test]
fn the_waits_stop_once_the_pool_holds_the_stream_metal() -> Result<()> {
    arm();
    let Some((fx, dev)) = fixture("qwen4exp_fetch_exit_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let n_experts = cfg.n_experts;
    let streamed = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(n_experts))?;
    let (_, _, cache) = decode(&streamed, &cfg, &dev, &PROMPT, STEPS)?;
    let f = streamed.fetch_stats();
    println!(
        "FETCHTEST exit steps={STEPS} passes={} entries={} exits={} mode={} redos={} jobs={} loads={}",
        cache.fetch_passes,
        cache.fetch_entries,
        cache.fetch_exits,
        cache.fetch_mode,
        cache.redos,
        f.jobs,
        f.loads
    );
    assert_eq!(cache.fetch_exits, 1, "the fetching regime never ended");
    assert!(!cache.fetch_mode, "the last step still fetched");
    assert!(
        (cache.fetch_passes as usize) < STEPS,
        "every step fetched on a pool that holds every expert"
    );
    assert_eq!(
        f.jobs,
        cache.fetch_passes * cfg.n_layers as u64,
        "a step that does not fetch encodes no wait"
    );
    // The steps after the regime ended are plain device plans, and the
    // first of them settles the last fetching step's trailing settle.
    assert!(
        !streamed.fetch_settle_trailing(),
        "a non-fetching step ran with a fetching step's settle still trailing"
    );
    Ok(())
}

/// The settle trails (`QWEN4EXP_FETCH_SETTLE=behind`, the default inside
/// the arm): after every fetching step its records wait for the next step,
/// so the host encodes step t+1 while the GPU runs step t; the model's next
/// entry or the driver's run end (`CausalLM::settle_pending`) settles it.
/// Through `Session::decode` the run ends with nothing trailing.
#[test]
fn a_fetching_steps_settle_trails_and_the_run_end_settles_it_metal() -> Result<()> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, IdentityProposer, Session};
    arm();
    let Some((fx, dev)) = fixture("qwen4exp_fetch_trailing_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let streamed = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let mut cache = streamed.new_cache(&cfg, &dev)?;
    let input = Tensor::from_vec(PROMPT.to_vec(), (1, PROMPT.len()), &dev)?;
    let mut logits = streamed.forward(&input, 0, &mut cache)?;
    for step in 0..8 {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = row
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &v)| if v > m.1 { (i, v) } else { m },
            )
            .0 as u32;
        let input = Tensor::from_vec(vec![next], (1, 1), &dev)?;
        logits = streamed.forward(&input, cache.pos, &mut cache)?;
        assert!(
            streamed.fetch_settle_trailing(),
            "step {step}: a fetching step settled before returning"
        );
    }
    streamed.settle_pending(&mut cache)?;
    assert!(
        !streamed.fetch_settle_trailing(),
        "settle_pending left a step trailing"
    );
    // Through the session: the run's end settles the last step.
    let cache = streamed.new_cache(&cfg, &dev)?;
    let mut session = Session::new(&streamed, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        &PROMPT,
        &DecodeOptions::new(12),
        &mut |_| Ok(()),
    )?;
    assert!(!out.ids.is_empty());
    assert!(
        out.stats.provisional,
        "the blocking arm's run did not settle its fetching steps itself"
    );
    assert!(
        !streamed.fetch_settle_trailing(),
        "the session's run ended with a fetching step's settle still trailing"
    );
    Ok(())
}

/// A settle behind the next step's encode reads its own step's records:
/// each tier alternates its record slot per plan and the settle waits for
/// its own pass only. Through the per-token loop here every fetching step
/// settles behind the next one's encode; through the session (the blocking
/// arm) every step is settled by the driver before the next is encoded.
/// The two count the same experts used — the tiers' usage profile, which
/// the settles write from the records — and decode the same ids. With one
/// record slot a tier, the settle behind read the next step's record: the
/// first decode step's experts went uncounted and the last step's counted
/// twice.
#[test]
fn a_settle_behind_the_next_encode_reads_its_own_steps_records_metal() -> Result<()> {
    use lloom::generation::{Blocking, LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, IdentityProposer, Session};
    arm();
    let Some((fx, dev)) = fixture("qwen4exp_fetch_own_records_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let behind = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let (ids, _, mut cache) = decode(&behind, &cfg, &dev, &PROMPT, STEPS)?;
    behind.settle_pending(&mut cache)?;
    let driven = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let mut session = Session::new(&driven, driven.new_cache(&cfg, &dev)?);
    let mut rb = Blocking::new(&[u32::MAX - 1], &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        &PROMPT,
        &DecodeOptions::new(STEPS),
        &mut |_| Ok(()),
    )?;
    assert!(out.stats.provisional);
    // Both loops draw STEPS ids, each from the pass before it, and run a
    // decode forward after each draw: the same passes.
    assert_eq!(out.ids, ids, "the two loops decode different ids");
    assert!(cache.fetch_passes > 0, "no step fetched");
    assert_eq!(
        behind.stream_usage(),
        driven.stream_usage(),
        "the settle behind the next encode counted other experts than the step's own"
    );
    Ok(())
}

/// On a lookahead ring (`lookahead = 1`), a fetching step's settle held for
/// the next step's encode — by the loop, `Provisional::Behind` (#927) —
/// runs while the GPU runs that next step and the worker serves its layers:
/// the two must not collide. The held settle is
/// the tiers' bookkeeping (the step's misses were loaded inside it), so it
/// loads nothing beside the worker; the decode is the resident model's,
/// every step fetched, nothing timed out. Run on a thread of its own and
/// bounded: a collision parks the queue rather than failing.
#[test]
fn a_fetching_settle_held_on_a_lookahead_ring_is_the_resident_stream_metal() {
    arm();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(ring_run());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(120)) {
        Ok(r) => r.unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("the ring run did not return in 120 s: a held settle and the worker parked")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("the ring run panicked"),
    }
}

fn ring_run() -> Result<()> {
    use lloom::generation::poll::PollMechanism;
    use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
    use lloom::nn::{DecodeOptions, IdentityProposer, Session};
    let Some((fx, dev)) = fixture("qwen4exp_fetch_ring_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let no_eos = [u32::MAX - 1];
    let want = {
        let resident = fx.resident::<CrownExact>(&cfg, &dev, &opts)?;
        let mut session = Session::new(&resident, resident.new_cache(&cfg, &dev)?);
        let mut rb = Blocking::new(&no_eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        session
            .decode(
                &mut IdentityProposer,
                &mut rb,
                &mut sampler,
                &PROMPT,
                &DecodeOptions::new(STEPS),
                &mut |_| Ok(()),
            )?
            .ids
    };
    let streamed = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let mut session = Session::new(&streamed, streamed.new_cache(&cfg, &dev)?);
    let mut rb = Ring::new(&no_eos, &dev, 64, EosRead::Poll(PollMechanism::default()))?
        .with_poll_budget(20_000);
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut o = DecodeOptions::new(STEPS);
    o.lookahead = 1;
    o.budget = CommitBudget::steps(4)?;
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        &PROMPT,
        &o,
        &mut |_| Ok(()),
    )?;
    let f = streamed.fetch_stats();
    let cache = session.cache_mut();
    println!(
        "FETCHTEST ring steps={STEPS} provisional={} passes={} redos={} jobs={} loads={} short={} timeouts={} floor_waits={}",
        out.stats.provisional, cache.fetch_passes, cache.redos, f.jobs, f.loads, f.short, f.timeouts, f.floor_waits
    );
    // Since #927 the ring's fixed cadence takes its steps provisionally
    // (`Provisional::Behind`): the held settle is the driver's, after the
    // next step's encode — the same overlap with the worker this test pins.
    assert!(
        out.stats.provisional,
        "the ring's fixed cadence did not settle its steps behind the next forward"
    );
    assert_eq!(
        out.ids, want,
        "the fetching ring's ids differ from the resident model's"
    );
    assert!(
        cache.fetch_passes > 0 && f.loads > 0,
        "no step fetched: {f:?}"
    );
    assert_eq!(
        (f.short, f.timeouts),
        (0, 0),
        "a layer went short or timed out"
    );
    assert_eq!(
        streamed.stream_stats().settle_loads,
        0,
        "a held settle loaded beside the worker"
    );
    assert!(!streamed.fetch_settle_trailing());
    Ok(())
}
