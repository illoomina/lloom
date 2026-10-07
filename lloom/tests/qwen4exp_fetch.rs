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
    Ok(())
}
