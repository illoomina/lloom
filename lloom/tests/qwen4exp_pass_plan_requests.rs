//! A replay inside a pass, on either side of a plan (`DESIGN.md` §9.2aa).
//! The streamed tier's device plan computes a step, settles it, and on a
//! miss rewinds and replays it on the host plan INSIDE the pass the driver
//! holds. So one key's pass has two shapes, and a plan recorded from one is
//! served to the other:
//!
//! - recorded without the replay, served to a pass that replays: the replay
//!   reads the step's embedding again, whose slot the plan had handed on;
//! - recorded with the replay, served to a pass that does not: the replay
//!   superseded the first computation's logits, so the plan handed their
//!   slot to the next allocation, and they are this pass's output.
//!
//! Each arm is driven as a driver drives it — one bracket around each
//! forward — over requests that reset the sequence, on pools small enough
//! that tokens miss and replay. The single-sequence decode (`decode_embed`),
//! the batched rows (`forward_rows`) and the verify window (`verify_window`)
//! each replay. Every id must be the plans-off run's, and an arm must have
//! served a pass that replayed (and, for decode and rows, run one on the
//! pool), or it says nothing.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, Qwen4Exp, Resident, State};
use lloom::nn::expert_stream::{ExpertResidency, ExpertSource, LoaderArm, Streamed};
use lloom::nn::model::{CausalLM, Row};
use lloom::profile::RewindableExact;
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKey, PassKind};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The recorder's bind observer and the dispatch clock are process globals.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 24;
const REQUESTS: usize = 4;

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 8;
    cfg.n_experts_per_tok = 2;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lloom-832-{}-{tag}.gguf", std::process::id()))
}

fn write_experts(cfg: &Config, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &dev, GgmlDType::F32);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    Ok(())
}

fn streamed(
    cfg: &Config,
    dev: &Device,
    path: &Path,
    slots: usize,
) -> Result<Qwen4Exp<RewindableExact, Streamed>> {
    let content = Content::read(&mut File::open(path)?)?;
    testing::model_with::<RewindableExact, _>(cfg, dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            slots,
            LoaderArm::Pread,
            dev,
        )
    })
}

fn resident(cfg: &Config, dev: &Device) -> Result<Qwen4Exp<RewindableExact, Resident>> {
    testing::model_with::<RewindableExact, _>(cfg, dev, |l| {
        let [g, u, d] = testing::expert_stacks(cfg, l, dev, GgmlDType::F32);
        Ok(Resident::new(Arc::new(g), Arc::new(u), Arc::new(d)))
    })
}

fn argmax(t: &Tensor) -> Result<u32> {
    t.flatten_all()?.argmax(0)?.to_scalar::<u32>()
}

/// A driver's bracket: `f` inside a pass of `kind` when `plans`, and whether
/// the cache served that pass.
fn in_pass<T>(
    dev: &Device,
    plans: bool,
    kind: PassKind,
    f: impl FnOnce() -> Result<T>,
) -> Result<(T, bool)> {
    let m = metal(dev);
    let served = || m.pass_plan_counters().map_or(0, |c| c.served);
    let before = served();
    if plans {
        m.begin_pass(PassKey::new(kind))?;
    }
    let out = f()?;
    if plans {
        m.finish_pass()?;
    }
    Ok((out, served() > before))
}

fn prefill_kind(len: usize) -> PassKind {
    PassKind::Prefill {
        w: len as u32,
        fresh: true,
    }
}

/// What a run decoded, and per step whether the cache served it and whether
/// it replayed.
struct Run {
    ids: Vec<Vec<u32>>,
    steps: Vec<(bool, bool)>,
}

impl Run {
    /// Served passes that replayed: the replay read its input from memory
    /// a plan recorded without the replay had freed.
    fn served_replays(&self) -> usize {
        self.steps.iter().filter(|&&(s, r)| s && r).count()
    }

    /// Passes on the pool that replayed: each would have been a key's plan,
    /// served to passes that do not replay.
    fn pool_replays(&self) -> usize {
        self.steps.iter().filter(|&&(s, r)| !s && r).count()
    }
}

/// Every request of a plans-on run against the plans-off ids, and the two
/// counts that say the run reached both sides of the replay: a served pass
/// that replayed, and a pass on the pool that replayed.
fn check_requests(
    arm: &str,
    slots: usize,
    want: &[Vec<u32>],
    mut request: impl FnMut() -> Result<Run>,
) -> Result<(usize, usize)> {
    let (mut served, mut pool) = (0, 0);
    for r in 0..REQUESTS {
        let run = request()?;
        served += run.served_replays();
        pool += run.pool_replays();
        assert_eq!(
            run.ids, want,
            "{arm} slots {slots} request {r} moved its ids under the plans; \
             per step (served, replayed): {:?}",
            run.steps
        );
    }
    eprintln!("{arm} slots {slots}: served replays {served}, pool replays {pool}");
    Ok((served, pool))
}

/// A served pass replayed: the replay's input was at stake.
fn assert_served_replay(arm: &str, (served, _): (usize, usize)) {
    assert!(
        served > 0,
        "{arm}: no served pass replayed, so the replay's input was never at stake"
    );
}

/// And a pass on the pool replayed: a recording that must not become a
/// plan was refused.
fn assert_pool_replay(arm: &str, (_, pool): (usize, usize)) {
    assert!(
        pool > 0,
        "{arm}: no pass on the pool replayed, so no replayed recording was refused"
    );
}

/// The mechanism, with no model: a pass that diverged closes nothing wrong
/// and is never made a plan; its key records again on its next sight, and
/// a pass that did not diverge plans as before.
#[test]
fn a_pass_that_diverged_is_never_a_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let x = Tensor::arange(0f32, 4096f32, &dev)?;
    let key = PassKey::new(PassKind::Decode { b: 1 });
    let pass = |diverge: bool| -> Result<Vec<f32>> {
        m.begin_pass(key)?;
        let y = x.affine(2.0, 1.0)?.affine(1.0, 3.0)?;
        if diverge {
            m.pass_diverged();
        }
        m.finish_pass()?;
        y.narrow(0, 0, 4)?.to_vec1::<f32>()
    };
    let counters = || m.pass_plan_counters().expect("cache on");

    // Every sight diverges: recorded each time, never planned.
    for _ in 0..3 {
        assert_eq!(pass(true)?, [4.0, 6.0, 8.0, 10.0]);
    }
    let c = counters();
    assert_eq!(
        (c.recorded, c.plans, c.served, c.diverged),
        (3, 0, 0, 3),
        "a diverged pass became a plan: {c:?}"
    );
    assert!(!m.pass_plan_cached(&key));

    // Two clean sights plan the key and the third is served.
    for _ in 0..3 {
        assert_eq!(pass(false)?, [4.0, 6.0, 8.0, 10.0]);
    }
    let c = counters();
    assert_eq!((c.recorded, c.plans, c.served), (4, 1, 1), "{c:?}");

    // A served pass that diverges keeps the plan: it was recorded from a
    // pass that did not.
    assert_eq!(pass(true)?, [4.0, 6.0, 8.0, 10.0]);
    let c = counters();
    assert_eq!((c.served, c.evicted, c.diverged), (2, 0, 4), "{c:?}");
    assert!(m.pass_plan_cached(&key));

    // Outside a pass it is a no-op.
    m.pass_diverged();
    assert_eq!(counters().diverged, 4);
    Ok(())
}

/// One request through the single-sequence decode: reset, the prefill and
/// `STEPS` greedy steps.
fn decode_request<R: ExpertResidency>(
    model: &Qwen4Exp<RewindableExact, R>,
    cache: &mut State<RewindableExact>,
    dev: &Device,
    plans: bool,
) -> Result<Run> {
    cache.reset();
    let input = Tensor::new(PROMPT.as_slice(), dev)?.unsqueeze(0)?;
    let (mut logits, _) = in_pass(dev, plans, prefill_kind(PROMPT.len()), || {
        model.forward(&input, 0, cache)
    })?;
    let mut ids = Vec::with_capacity(STEPS);
    let mut steps = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let next = argmax(&logits)?;
        // Read; held across the next pass it would be an undeclared escape.
        drop(logits);
        ids.push(next);
        let redos = cache.redos;
        let input = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        let (l, served) = in_pass(dev, plans, PassKind::Decode { b: 1 }, || {
            model.forward(&input, cache.pos, cache)
        })?;
        steps.push((served, cache.redos > redos));
        logits = l;
    }
    Ok(Run {
        ids: vec![ids],
        steps,
    })
}

#[test]
fn a_replay_inside_a_served_decode_step_is_on_neither_side_of_its_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = tiny();
    let path = temp_gguf("decode");
    write_experts(&cfg, &path)?;

    let dev = Device::new_metal(0)?;
    let model = resident(&cfg, &dev)?;
    let mut cache = RewindableExact::single(&model, &dev, 8)?;
    let want = decode_request(&model, &mut cache, &dev, false)?.ids;
    assert!(
        want[0].iter().any(|&t| t != want[0][0]),
        "a degenerate stream cannot tell the arms apart: {want:?}"
    );

    let mut reached = (0, 0);
    // 8 experts at top-2 over 3, 5 and 7 slots: tokens miss, settles load,
    // passes replay, at rates that differ with the pool.
    for slots in [3usize, 5, 7] {
        // The tier with no plans decodes the resident stream: the reference
        // is the tier's own.
        let dev = Device::new_metal(0)?;
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut cache = RewindableExact::single(&model, &dev, 8)?;
        let control = decode_request(&model, &mut cache, &dev, false)?;
        assert_eq!(
            control.ids, want,
            "slots {slots}: the streamed tier differs with no plans"
        );

        let dev = Device::new_metal(0)?;
        metal(&dev).enable_pass_plans(ArenaLayout::Packed);
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut cache = RewindableExact::single(&model, &dev, 8)?;
        let (s, p) = check_requests("decode", slots, &want, || {
            decode_request(&model, &mut cache, &dev, true)
        })?;
        reached = (reached.0 + s, reached.1 + p);
    }
    std::fs::remove_file(&path).ok();
    assert_served_replay("decode", reached);
    assert_pool_replay("decode", reached);
    Ok(())
}

const ROW_PROMPTS: [&[u32]; 3] = [&[3, 5, 2, 8, 8, 1, 9], &[7, 1, 4], &[2, 2, 6, 9, 1]];

/// One request through the batched rows: every row reset and prefilled
/// alone, then `STEPS` steps of all rows as one `forward_rows`.
fn rows_request<R: ExpertResidency>(
    model: &Qwen4Exp<RewindableExact, R>,
    caches: &mut [State<RewindableExact>],
    dev: &Device,
    plans: bool,
) -> Result<Run> {
    let b = caches.len();
    let mut next = Vec::with_capacity(b);
    for (cache, prompt) in caches.iter_mut().zip(ROW_PROMPTS) {
        cache.reset();
        let input = Tensor::new(prompt, dev)?.unsqueeze(0)?;
        let (logits, _) = in_pass(dev, plans, prefill_kind(prompt.len()), || {
            model.forward(&input, 0, cache)
        })?;
        next.push(argmax(&logits)?);
    }
    let mut ids = vec![Vec::with_capacity(STEPS); b];
    let mut steps = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        for (row, &t) in ids.iter_mut().zip(&next) {
            row.push(t);
        }
        let redos: u64 = caches.iter().map(|c| c.redos).sum();
        let tokens = Tensor::from_vec(next.clone(), (b, 1), dev)?;
        let mut rows: Vec<Row<'_, State<RewindableExact>>> = caches
            .iter_mut()
            .map(|c| Row {
                pos: c.pos,
                cache: c,
            })
            .collect();
        let (logits, served) = in_pass(dev, plans, PassKind::Decode { b: b as u16 }, || {
            model.forward_rows(&tokens, &mut rows)
        })?;
        let replayed = caches.iter().map(|c| c.redos).sum::<u64>() > redos;
        steps.push((served, replayed));
        for (r, n) in next.iter_mut().enumerate() {
            *n = argmax(&logits.narrow(0, r, 1)?)?;
        }
    }
    Ok(Run { ids, steps })
}

#[test]
fn a_replay_inside_a_served_rows_step_is_on_neither_side_of_its_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let mut cfg = tiny();
    // The rows arm's attention, as `qwen4exp_batched_rows` runs it on Metal.
    cfg.head_dim = 32;
    let path = temp_gguf("rows");
    write_experts(&cfg, &path)?;
    let new_caches = |model: &Qwen4Exp<RewindableExact, Streamed>,
                      dev: &Device|
     -> Result<Vec<State<RewindableExact>>> {
        (0..ROW_PROMPTS.len())
            .map(|_| RewindableExact::single(&model, &dev, 8))
            .collect()
    };

    let mut reached = (0, 0);
    // 3 rows at top-2 of 8 over 6 and 7 slots: a step's experts always fit,
    // the device plan runs and misses.
    for slots in [6usize, 7] {
        let dev = Device::new_metal(0)?;
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut caches = new_caches(&model, &dev)?;
        let want = rows_request(&model, &mut caches, &dev, false)?.ids;

        let dev = Device::new_metal(0)?;
        metal(&dev).enable_pass_plans(ArenaLayout::Packed);
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut caches = new_caches(&model, &dev)?;
        let (s, p) = check_requests("rows", slots, &want, || {
            rows_request(&model, &mut caches, &dev, true)
        })?;
        reached = (reached.0 + s, reached.1 + p);
    }
    std::fs::remove_file(&path).ok();
    assert_served_replay("rows", reached);
    assert_pool_replay("rows", reached);
    Ok(())
}

/// The ids a verify window carries after its first; any fixed ids do, since
/// each window is held to the same window with no plans.
const DRAFTS: [u32; 3] = [4, 11, 7];
const ROUNDS: usize = 12;

/// One request through the verify window: reset, the prefill, then `ROUNDS`
/// windows of the next id and `DRAFTS`, each accepted whole.
fn verify_request<R: ExpertResidency>(
    model: &Qwen4Exp<RewindableExact, R>,
    cache: &mut State<RewindableExact>,
    dev: &Device,
    plans: bool,
) -> Result<Run> {
    cache.reset();
    let input = Tensor::new(PROMPT.as_slice(), dev)?.unsqueeze(0)?;
    let (logits, _) = in_pass(dev, plans, prefill_kind(PROMPT.len()), || {
        model.forward(&input, 0, cache)
    })?;
    let mut next = argmax(&logits)?;
    drop(logits);
    let t = DRAFTS.len() + 1;
    let mut ids = Vec::with_capacity(ROUNDS);
    let mut steps = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let mut window = vec![next];
        window.extend_from_slice(&DRAFTS);
        let redos = cache.redos;
        let (argmaxes, served) = in_pass(dev, plans, PassKind::Verify { k: t as u16 }, || {
            model.verify_window(&window, cache)
        })?;
        steps.push((served, cache.redos > redos));
        cache.accept_window(t)?;
        next = argmaxes[t - 1];
        ids.push(argmaxes);
    }
    Ok(Run { ids, steps })
}

#[test]
fn a_replay_inside_a_served_verify_window_is_on_neither_side_of_its_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let mut cfg = tiny();
    cfg.head_dim = 32;
    let path = temp_gguf("verify");
    write_experts(&cfg, &path)?;
    let spec_cache = |model: &Qwen4Exp<RewindableExact, Streamed>,
                      dev: &Device|
     -> Result<State<RewindableExact>> {
        let pools = model.spec_pools(1, DRAFTS.len(), dev);
        State::<RewindableExact>::in_ring(&cfg, &pools)
    };

    let mut reached = (0, 0);
    // Windows of 4 positions at top-2 of 8 over 4 and 6 slots: windows that
    // settle clean arm the device plan, and later ones miss.
    for slots in [4usize, 6] {
        let dev = Device::new_metal(0)?;
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut cache = spec_cache(&model, &dev)?;
        let want = verify_request(&model, &mut cache, &dev, false)?.ids;

        let dev = Device::new_metal(0)?;
        metal(&dev).enable_pass_plans(ArenaLayout::Packed);
        let model = streamed(&cfg, &dev, &path, slots)?;
        let mut cache = spec_cache(&model, &dev)?;
        let (s, p) = check_requests("verify", slots, &want, || {
            verify_request(&model, &mut cache, &dev, true)
        })?;
        reached = (reached.0 + s, reached.1 + p);
    }
    std::fs::remove_file(&path).ok();
    // The verify's replays land in served windows at these pools, so this
    // arm holds the replay's input; refusing a replayed recording is the
    // same `pass_diverged` call the decode and rows arms reach.
    assert_served_replay("verify", reached);
    Ok(())
}
