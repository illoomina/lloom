//! The carries an MTP drafter reads (`NextnHidden`, the target's pre-mixer
//! `res_hc` at each kept position) are produced inside a planned pass — the
//! verify window, the prefill, the decode step — and read by `observe`
//! after it. A plan serves a slot to the next value whose interval does not
//! overlap, and the recorder cannot see a read made between passes
//! (`docs/pass-plan-design.md` §3.3, I3), so a carry left in a plan's slot
//! is overwritten by whatever the plan hands that slot next. The carries
//! are therefore declared escapes at the producer.
//!
//! The reference is the pool path: the same requests with no plan cache,
//! where every value is a buffer of its own. Under the plan cache — a
//! recording round and served rounds — every probe the drafter reads must
//! be the pool path's, bit for bit, in three reading orders:
//!
//! - **at the observe**, as the serial round reads it today (the drafter's
//!   own host wait precedes the next verify);
//! - **one round late**, after the next served verify has run (what a
//!   lookahead, a device-planned observe that stops reading back, or any
//!   later reader sees);
//! - **across two lanes**, the pipelined driver's interleave: lane B's
//!   verify is encoded between lane A's verify and A's observe.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Cache, Config, Qwen4Exp, Resident};
use lloom::nn::speculator::{Draft, NextnHidden, SeqId, Speculator};
use lloom::nn::{DecodeOptions, Session};
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKind};
use lloom_tensor::{Device, Result, Tensor};

/// The recorder's bind observer and the dispatch clock are process globals.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

const K: usize = 1;
const MAX: usize = 16;
const ROUNDS: usize = 3;

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg.eos_token_id = 127;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

/// When the drafter reads the probe it is handed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Read {
    /// In `observe`, before the next verify is encoded.
    Now,
    /// At the next `observe`: after the next round's verify pass has run.
    Late,
}

/// Proposes one fixed token a round, and records every probe it observes.
/// A wrong draft is rejected, so a round keeps one carry and `take_nextn`
/// hands it back as the pass produced it, with no `cat` between passes to
/// copy it out of the plan's slot.
struct Recorder {
    read: Read,
    token: u32,
    held: Vec<(SeqId, Tensor)>,
    seen: Vec<Vec<Vec<f32>>>,
}

impl Recorder {
    fn new(read: Read, token: u32, n: usize) -> Self {
        Self {
            read,
            token,
            held: Vec::new(),
            seen: vec![Vec::new(); n],
        }
    }

    fn record(&mut self, seq: SeqId, probe: &Tensor) -> Result<()> {
        self.seen[seq.0].push(probe.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }

    /// The probes still held at the end of a run (the `Late` arm's last).
    fn finish(&mut self) -> Result<Vec<Vec<Vec<f32>>>> {
        for (seq, t) in std::mem::take(&mut self.held) {
            self.record(seq, &t)?;
        }
        Ok(std::mem::take(&mut self.seen))
    }
}

impl Speculator for Recorder {
    type Probe = NextnHidden;
    fn n_seq(&self) -> usize {
        self.seen.len()
    }
    fn max_draft(&self) -> usize {
        K
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active
            .iter()
            .map(|_| Draft::host(vec![self.token; K]))
            .collect())
    }
    fn observe(&mut self, seq: SeqId, probe: &Tensor, _: &[u32]) -> Result<()> {
        match self.read {
            Read::Now => self.record(seq, probe),
            Read::Late => {
                if let Some(i) = self.held.iter().position(|(s, _)| *s == seq) {
                    let (_, prev) = self.held.remove(i);
                    self.record(seq, &prev)?;
                }
                self.held.push((seq, probe.clone()));
                Ok(())
            }
        }
    }
}

fn spec_cache(model: &Qwen4Exp<Resident>, cfg: &Config, dev: &Device) -> Result<Cache> {
    let mut cache = Cache::in_pools(cfg, &model.spec_pools(1, K, dev))?;
    cache.want_nextn = true;
    Ok(cache)
}

/// One request through the serial speculative round on a fresh cache: the
/// stream and every probe the drafter read.
fn serial(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    read: Read,
    token: u32,
) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    let mut session = Session::new(model, spec_cache(model, cfg, dev)?);
    let mut spec = Recorder::new(read, token, 1);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode_speculative(
        &mut spec,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(MAX),
        &eos,
        &mut |_| Ok(()),
    )?;
    let seen = spec.finish()?.pop().expect("one lane");
    Ok((out.ids, seen))
}

/// Every probe one lane read, flattened, in order.
type Probes = Vec<Vec<f32>>;

/// Two requests through the pipelined driver, one lane each.
fn pipelined(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    dev: &Device,
    prompts: &[&[u32]],
    token: u32,
) -> Result<(Vec<Vec<u32>>, Vec<Probes>)> {
    let mut sessions: Vec<Session<'_, Qwen4Exp<Resident>>> = prompts
        .iter()
        .map(|_| Ok(Session::new(model, spec_cache(model, cfg, dev)?)))
        .collect::<Result<_>>()?;
    let mut spec = Recorder::new(Read::Now, token, prompts.len());
    let outs = Session::decode_speculative_pipelined(
        &mut sessions,
        &mut spec,
        prompts,
        &DecodeOptions::new(MAX),
        &[cfg.eos_token_id],
        dev,
        &mut |_, _| Ok(()),
    )?;
    Ok((outs.into_iter().map(|o| o.ids).collect(), spec.finish()?))
}

fn prompt(seed: u32) -> Vec<u32> {
    (0..7).map(|i| (seed * 11 + i * 7 + 1) % 120).collect()
}

/// Every probe of `got` equal to the pool path's `want`, with the first
/// differing element named.
fn same_probes(arm: &str, round: usize, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{arm} round {round}: {} observes against the pool path's {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{arm} round {round}: probe {i}'s width");
        if let Some(j) = (0..g.len()).find(|&j| g[j].to_bits() != w[j].to_bits()) {
            panic!(
                "{arm} round {round}: probe {i} was overwritten after the pass that produced it \
                 (element {j}: {} against the pool path's {})",
                g[j], w[j]
            );
        }
    }
}

/// The served verify passes and the plans' escapes over `f`.
fn served_over<T>(m: &MetalDevice, f: impl FnOnce() -> Result<T>) -> Result<(T, u64)> {
    let before = m.pass_plan_counters().map_or(0, |c| c.served);
    let out = f()?;
    let after = m.pass_plan_counters().map_or(0, |c| c.served);
    Ok((out, after - before))
}

/// A planned verify keeps the drafter's carries, read at the observe or a
/// round late: every probe of a recording round and two served rounds is
/// the pool path's, and the verify key was served.
#[test]
fn a_planned_verify_keeps_the_mtp_carries() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = config();
    let p = prompt(0);
    let token = 0u32;
    for read in [Read::Now, Read::Late] {
        let dev = Device::new_metal(0)?;
        let model = testing::resident_model(&cfg, &dev);
        let (want_ids, want) = serial(&model, &cfg, &dev, &p, read, token)?;
        let one_carry = want.iter().filter(|v| v.len() == cfg.hc_dim()).count();
        assert!(
            want.len() >= 4 && one_carry >= 2,
            "{read:?}: the run must span rounds that keep one carry: {} observes, {one_carry} \
             of one carry, ids {want_ids:?}",
            want.len()
        );

        let dev = Device::new_metal(0)?;
        let m = metal(&dev);
        m.enable_pass_plans(ArenaLayout::Packed);
        let model = testing::resident_model(&cfg, &dev);
        let mut served = 0;
        for round in 0..ROUNDS {
            let ((ids, got), s) = served_over(m, || serial(&model, &cfg, &dev, &p, read, token))?;
            assert_eq!(ids, want_ids, "{read:?} round {round}: the stream moved");
            same_probes(&format!("{read:?}"), round, &got, &want);
            served += s;
        }
        let (keys, _) = m.pass_plan_report().expect("the plan cache is on");
        assert!(
            keys.iter()
                .any(|(k, ..)| matches!(k.kind, PassKind::Verify { k: w } if w as usize == K + 1)),
            "{read:?}: no verify key was planned: {:?}",
            keys.iter().map(|(k, ..)| k.kind).collect::<Vec<_>>()
        );
        let c = m.pass_plan_counters().expect("on");
        eprintln!(
            "MTPCARRIES read={read:?} observes={} served={served} {c:?}",
            want.len()
        );
        assert!(
            served as usize >= want.len(),
            "{read:?}: {served} passes served over {ROUNDS} requests of {} observes: the plans \
             never held the carries",
            want.len()
        );
    }
    Ok(())
}

/// Two lanes through the pipelined driver: lane B's verify is encoded
/// between lane A's verify and A's observe, so A's carries sit through a
/// served pass of the same key before they are read. Every probe of either
/// lane is its pool-path probe.
#[test]
fn a_second_lanes_verify_does_not_overwrite_the_firsts_carries() -> Result<()> {
    let _serial = SERIAL.lock();
    let cfg = config();
    let (p0, p1) = (prompt(0), prompt(3));
    let prompts: Vec<&[u32]> = vec![&p0, &p1];
    let token = 0u32;
    let dev = Device::new_metal(0)?;
    let model = testing::resident_model(&cfg, &dev);
    let (want_ids, want) = pipelined(&model, &cfg, &dev, &prompts, token)?;
    assert_ne!(
        want[0], want[1],
        "the two lanes must carry different values"
    );

    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let model = testing::resident_model(&cfg, &dev);
    let mut served = 0;
    for round in 0..ROUNDS {
        let ((ids, got), s) = served_over(m, || pipelined(&model, &cfg, &dev, &prompts, token))?;
        assert_eq!(ids, want_ids, "round {round}: a stream moved");
        for lane in 0..2 {
            same_probes(&format!("lane {lane}"), round, &got[lane], &want[lane]);
        }
        served += s;
    }
    let c = m.pass_plan_counters().expect("on");
    eprintln!(
        "MTPCARRIES lanes=2 observes={:?} served={served} {c:?}",
        (want[0].len(), want[1].len())
    );
    assert!(
        served > 0,
        "no pass was served: the plans never held the carries"
    );
    Ok(())
}
