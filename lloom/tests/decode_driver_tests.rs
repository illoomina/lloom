//! `Session::decode` over the readback strategies (lloom #673, stage 1 of
//! `design/docs/decode-driver-design.md`).
//!
//! The target is a fake `CausalLM` whose next token is a fixed function of the
//! last one, computed entirely on the device — two `index_select`s, no
//! readback — so the loop's own round trips are the only ones a counter would
//! see, and so the same chain reproduces on the CPU and Metal backends. The
//! chain is `next = (tok * 7 + 3) % VOCAB`, whose orbit from token 1 passes
//! the EOS id at the ninth step and whose orbit from token 0 never does, so
//! every test's expected ids are derived rather than recorded.
//!
//! What is asserted is the loop's contract, not a timing: the three strategies
//! produce the same ids, EOS ends the run at the right token and is not in the
//! ids, the lookahead's overrun is exactly what the docs say it is, the counters
//! reconcile (`ring_publishes == ids + eos`, `commits` follows the budget), and
//! the refusals refuse.

use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
    WAKE_LIVE_EVERY_DEFAULT,
};
use lloom::nn::decode::IN_FLIGHT_STEPS;
use lloom::nn::speculator::{Draft, NoProbe, SeqId, TargetProbe};
use lloom::nn::{CausalLM, DecodeOptions, IdentityProposer, Model, Session, Speculator};
use lloom_tensor::{DType, Device, Result, Tensor};

const VOCAB: usize = 64;

/// Tests that read a GPU-load-dependent quantity (the pacer's in-flight
/// count) or that put a load on the GPU (the repeated hazard sweep) take this
/// so they do not run beside each other: `cargo test` runs the file's tests
/// in parallel on one device, and a busy GPU retires buffers late — the
/// pacer test read `4 in flight, bound 3` beside the sweep, and passes alone.
static GPU_QUIET: std::sync::Mutex<()> = std::sync::Mutex::new(());
const EOS: u32 = 42;

fn next_of(tok: u32) -> u32 {
    (tok * 7 + 3) % VOCAB as u32
}

/// The chain from `start`, stopping before EOS, capped at `max`.
fn chain(start: u32, max: usize) -> (Vec<u32>, bool) {
    let mut ids = Vec::new();
    let mut tok = start;
    while ids.len() < max {
        tok = next_of(tok);
        if tok == EOS {
            return (ids, true);
        }
        ids.push(tok);
    }
    (ids, false)
}

/// A decoder whose logits peak at `next_of(last token)`, device-side.
struct Chain {
    /// `[VOCAB]` f32: `table[t] = next_of(t)`.
    table: Tensor,
    /// `[VOCAB, VOCAB]` f32 identity: row `t` is the one-hot logits for `t`.
    eye: Tensor,
}

impl Chain {
    fn new(device: &Device) -> Result<Self> {
        let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
        Ok(Self {
            table: Tensor::from_vec(table, VOCAB, device)?,
            eye: Tensor::eye(VOCAB, DType::F32, device)?,
        })
    }
}

impl Model for Chain {
    type Config = ();
    fn from_weights(_vb: lloom::nn::VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("Chain has no weights")
    }
}

impl CausalLM for Chain {
    /// Positions seen, so a test can check `pos` against the cache.
    type Cache = usize;

    fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<usize> {
        Ok(0)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }

    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        let (_b, seq) = embeds.dims2()?;
        assert_eq!(
            *cache, pos,
            "forward at pos {pos} against a cache of {cache}"
        );
        *cache += seq;
        // The last position's token, `[1]` U32, then its successor, then the
        // one-hot row: all on the device.
        let last = embeds.narrow(1, seq - 1, 1)?.flatten_all()?;
        let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
        self.eye.index_select(&next, 0)
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    if let Ok(m) = Device::new_metal(0) {
        v.push(m);
    }
    v
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

#[derive(Debug, Clone, Copy)]
enum Arm {
    Blocking,
    RingBlocking,
    RingPoll,
    /// The device never computes the EOS flag; the host learns it from the
    /// drained ids on the commit cadence, so the run overruns an EOS by up to a
    /// commit period (#677).
    RingHostDrain,
    /// One `eos_index` dispatch per window, read with a tail wait.
    RingGpuBatch,
    /// The same index, staged ahead of the lookahead forward and read only
    /// when its buffer has completed; the host scans otherwise.
    RingGpuBatchNonblocking,
}

/// One of the three arms, behind an enum rather than a `dyn`: the loop stays
/// generic and monomorphizes, which is what §15.2 #9 requires of production
/// and what this test exercises.
enum AnyArm {
    Blocking(Blocking),
    Ring(Ring),
}

fn strategy(arm: &Arm, device: &Device, capacity: usize) -> Result<AnyArm> {
    Ok(match arm {
        Arm::Blocking => AnyArm::Blocking(Blocking::new(&[EOS], device)?),
        Arm::RingBlocking => AnyArm::Ring(Ring::new(&[EOS], device, capacity, EosRead::Blocking)?),
        // A short poll budget: the fake forward is three dispatches, so under
        // lookahead the publish sits below the command-buffer cut and the poll
        // can only ever fall back — which the counters then show.
        Arm::RingPoll => AnyArm::Ring(
            Ring::new(
                &[EOS],
                device,
                capacity,
                EosRead::Poll(PollMechanism::default()),
            )?
            .with_poll_budget(20_000),
        ),
        Arm::RingHostDrain => {
            AnyArm::Ring(Ring::new(&[EOS], device, capacity, EosRead::HostOnDrain)?)
        }
        Arm::RingGpuBatch => {
            AnyArm::Ring(Ring::new(&[EOS], device, capacity, EosRead::GpuBatched)?)
        }
        Arm::RingGpuBatchNonblocking => AnyArm::Ring(Ring::new(
            &[EOS],
            device,
            capacity,
            EosRead::GpuBatchedNonblocking,
        )?),
    })
}

impl ReadbackStrategy for AnyArm {
    fn device(&self) -> &Device {
        match self {
            AnyArm::Blocking(b) => b.device(),
            AnyArm::Ring(r) => r.device(),
        }
    }
    fn arms(&self) -> lloom::generation::ReadbackArms {
        match self {
            AnyArm::Blocking(b) => b.arms(),
            AnyArm::Ring(r) => r.arms(),
        }
    }
    fn admits_lookahead(&self) -> bool {
        match self {
            AnyArm::Blocking(b) => b.admits_lookahead(),
            AnyArm::Ring(r) => r.admits_lookahead(),
        }
    }
    fn loop_commits(&self) -> bool {
        match self {
            AnyArm::Blocking(b) => b.loop_commits(),
            AnyArm::Ring(r) => r.loop_commits(),
        }
    }
    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        match self {
            AnyArm::Blocking(b) => b.check_sampler(sampling),
            AnyArm::Ring(r) => r.check_sampler(sampling),
        }
    }
    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        match self {
            AnyArm::Blocking(b) => b.sample_and_feed(sampler, logits),
            AnyArm::Ring(r) => r.sample_and_feed(sampler, logits),
        }
    }
    fn eos_of_last(&mut self) -> Result<bool> {
        match self {
            AnyArm::Blocking(b) => b.eos_of_last(),
            AnyArm::Ring(r) => r.eos_of_last(),
        }
    }
    fn take_ids(&mut self) -> Result<Vec<u32>> {
        match self {
            AnyArm::Blocking(b) => b.take_ids(),
            AnyArm::Ring(r) => r.take_ids(),
        }
    }
    fn drain_all(&mut self) -> Result<Vec<u32>> {
        match self {
            AnyArm::Blocking(b) => b.drain_all(),
            AnyArm::Ring(r) => r.drain_all(),
        }
    }
    // The two defaulted methods are forwarded too: a wrapper that left them at
    // the trait's no-op would drive a ring whose marker is never written, and
    // every drain-arm assertion below would be about the wrapper.
    fn stage_prompt_chunk(&mut self, tokens: &[u32]) -> Result<Tensor> {
        match self {
            AnyArm::Blocking(b) => b.stage_prompt_chunk(tokens),
            AnyArm::Ring(r) => r.stage_prompt_chunk(tokens),
        }
    }
    fn publish_marker(&mut self, force_live: bool) -> Result<()> {
        match self {
            AnyArm::Blocking(b) => b.publish_marker(force_live),
            AnyArm::Ring(r) => r.publish_marker(force_live),
        }
    }
    fn close_window(&mut self, in_ttft_ramp: bool) -> Result<()> {
        match self {
            AnyArm::Blocking(b) => b.close_window(in_ttft_ramp),
            AnyArm::Ring(r) => r.close_window(in_ttft_ramp),
        }
    }
    fn eos_in_drained(&self, batch: &[u32]) -> Option<usize> {
        match self {
            AnyArm::Blocking(b) => b.eos_in_drained(batch),
            AnyArm::Ring(r) => r.eos_in_drained(batch),
        }
    }
    fn note_eos_truncation(&mut self, dropped_in_batch: usize) {
        match self {
            AnyArm::Blocking(b) => b.note_eos_truncation(dropped_in_batch),
            AnyArm::Ring(r) => r.note_eos_truncation(dropped_in_batch),
        }
    }
    fn eos_read_arm(&self) -> &'static str {
        match self {
            AnyArm::Blocking(b) => b.eos_read_arm(),
            AnyArm::Ring(r) => r.eos_read_arm(),
        }
    }
    fn counters(&self) -> lloom::generation::ReadbackCounters {
        match self {
            AnyArm::Blocking(b) => b.counters(),
            AnyArm::Ring(r) => r.counters(),
        }
    }
}

/// The streaming hook these tests do not exercise: the loop calls it with
/// each confirmed batch, and the returned `ids` carry the whole run either way.
fn sink(_batch: &[u32]) -> Result<()> {
    Ok(())
}

fn run(
    device: &Device,
    arm: &Arm,
    prompt: &[u32],
    opts: &DecodeOptions,
) -> Result<(lloom::nn::DecodeOut, usize)> {
    let model = Chain::new(device)?;
    let mut session = Session::new(&model, 0usize);
    let mut spec = IdentityProposer;
    let mut rb = strategy(arm, device, opts.max_tokens + 4)?;
    let mut sampler = greedy();
    let out = session.decode(&mut spec, &mut rb, &mut sampler, prompt, opts, &mut sink)?;
    Ok((out, session.pos()))
}

/// The chain from 1 reaches EOS in a handful of steps; from 0 it runs long.
#[test]
fn the_fixture_reaches_eos_from_one_and_not_from_zero() {
    let (ids, eos) = chain(1, 64);
    assert!(
        eos,
        "the chain from 1 must reach EOS for the EOS tests to mean anything"
    );
    assert!(
        ids.len() >= 2,
        "the chain from 1 must produce some tokens first: {ids:?}"
    );
    let (ids0, eos0) = chain(0, 32);
    assert!(!eos0 && ids0.len() == 32);
}

#[test]
fn every_arm_produces_the_same_ids_as_the_host_chain() -> Result<()> {
    for device in devices() {
        let (want, _) = chain(0, 24);
        for (arm, lookahead) in [
            (Arm::Blocking, 0),
            (Arm::RingBlocking, 0),
            (Arm::RingBlocking, 1),
            (Arm::RingPoll, 0),
            (Arm::RingPoll, 1),
            (Arm::RingHostDrain, 1),
            (Arm::RingGpuBatch, 1),
            (Arm::RingGpuBatchNonblocking, 1),
        ] {
            let opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(24)
            };
            let (out, pos) = run(&device, &arm, &[0], &opts)?;
            assert_eq!(out.ids, want, "{device:?} lookahead {lookahead}");
            assert!(!out.eos);
            assert_eq!(out.stats.produced, 24);
            // Prompt of 1, then one forward per token (the last one's logits
            // go unused, as in every harness loop).
            assert_eq!(pos, 1 + 24, "{device:?} lookahead {lookahead}");
            assert_eq!(out.stats.step_ns.len(), 24);
            assert_eq!(out.stats.overrun, 0);
        }
    }
    Ok(())
}

/// The host-on-drain arm learns EOS from the drained ids, so it overruns by
/// as many forwards as the commit period let it enqueue past the EOS — and
/// it must report exactly that count, drop the overrun from the ids, and leave
/// the ring clean for a second run on the same strategy (#677: at HEAD the
/// arm reported `overrun = 0` and left the undrained overrun slots behind, so
/// a second `decode` on the same ring would have returned them as tokens).
#[test]
fn the_host_drain_arm_reports_its_overrun_and_a_second_run_starts_clean() -> Result<()> {
    for device in devices() {
        let (want, hit) = chain(1, 64);
        assert!(hit);
        for period in [1usize, 4, 8] {
            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut spec = IdentityProposer;
            let mut rb = strategy(&Arm::RingHostDrain, &device, 128)?;
            let mut sampler = greedy();
            let opts = DecodeOptions {
                lookahead: 1,
                budget: if period == 1 {
                    CommitBudget::per_token()
                } else {
                    CommitBudget::steps(period)?
                },
                ..DecodeOptions::new(64)
            };
            let out = session.decode(&mut spec, &mut rb, &mut sampler, &[1], &opts, &mut sink)?;
            assert_eq!(out.ids, want, "{device:?} steps:{period}");
            assert!(out.eos);
            // The EOS token was fed, plus everything enqueued before the drain
            // that showed it. The marker goes LIVE every `WAKE_LIVE_EVERY_DEFAULT`
            // commits (one commit per `period` steps), and a drain reads the
            // marker the GPU has executed — the previous live one, at worst —
            // so the EOS can be seen up to two live windows late, plus the
            // forwards the pacer lets the host keep in flight.
            let bound = 2 * period * WAKE_LIVE_EVERY_DEFAULT + IN_FLIGHT_STEPS;
            assert!(
                out.stats.overrun >= 1 && out.stats.overrun <= bound,
                "{device:?} steps:{period}: overrun {} outside [1, {bound}]",
                out.stats.overrun
            );
            // The cache position is the token log plus the overrun, exactly.
            assert_eq!(session.pos(), 1 + want.len() + out.stats.overrun);
            assert_eq!(session.overrun(), out.stats.overrun);
            assert_eq!(session.tokens(), [&[1u32][..], &want[..]].concat());
            // Publishes reconcile: every publish is a kept id, the EOS, or a
            // discarded overrun.
            let c = out.stats.readback;
            assert_eq!(
                c.ring_publishes,
                out.ids.len() as u64 + c.ring_eos_publishes + c.ring_discarded,
                "{device:?} steps:{period}: {c:?} against {} ids",
                out.ids.len()
            );
            assert_eq!(c.ring_eos_publishes, 1);
            assert_eq!(c.ring_discarded as usize, out.stats.overrun - 1);

            // A second run on the SAME ring, after the sequence is reset, must
            // not see the first run's overrun slots as its own tokens. The
            // position-counter cache resets by assignment.
            session = Session::new(&model, 0usize);
            let (want0, _) = chain(0, 12);
            let opts2 = DecodeOptions {
                max_tokens: 12,
                ..opts
            };
            let out2 = session.decode(&mut spec, &mut rb, &mut sampler, &[0], &opts2, &mut sink)?;
            assert_eq!(
                out2.ids, want0,
                "{device:?} steps:{period}: second run polluted"
            );
            assert_eq!(out2.stats.overrun, 0);
            assert_eq!(rb.eos_read_arm(), "drain");
        }
    }
    Ok(())
}

#[test]
fn eos_ends_the_run_at_the_right_token_and_is_not_in_the_ids() -> Result<()> {
    for device in devices() {
        let (want, hit) = chain(1, 64);
        assert!(hit);
        for (arm, lookahead) in [
            (Arm::Blocking, 0),
            (Arm::RingBlocking, 0),
            (Arm::RingPoll, 0),
            (Arm::RingBlocking, 1),
            (Arm::RingPoll, 1),
        ] {
            let opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(64)
            };
            let (out, pos) = run(&device, &arm, &[1], &opts)?;
            assert_eq!(out.ids, want, "{device:?} lookahead {lookahead}");
            assert!(out.eos);
            assert!(!out.ids.contains(&EOS));
            // Without lookahead the cache stops at the last returned token;
            // with it, one forward was enqueued past the EOS — reported, not
            // hidden.
            assert_eq!(out.stats.overrun, lookahead, "{device:?}");
            assert_eq!(
                pos,
                1 + want.len() + lookahead,
                "{device:?} lookahead {lookahead}"
            );
            // EOS ends the step without a timing entry, as the harness loop does.
            assert_eq!(out.stats.step_ns.len(), want.len());
            assert_eq!(out.stats.steps as usize, want.len() + 1);
        }
    }
    Ok(())
}

#[test]
fn ring_publishes_reconcile_against_the_ids_and_the_eos() -> Result<()> {
    for device in devices() {
        for (arm, prompt, lookahead) in [
            (Arm::RingPoll, 1u32, 0usize),
            (Arm::RingPoll, 1, 1),
            (Arm::RingBlocking, 0, 0),
            (Arm::RingPoll, 0, 1),
        ] {
            let opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(20)
            };
            let (out, _) = run(&device, &arm, &[prompt], &opts)?;
            let c = out.stats.readback;
            assert_eq!(
                c.ring_publishes,
                out.ids.len() as u64 + c.ring_eos_publishes,
                "{device:?}: published {} against {} ids + {} eos",
                c.ring_publishes,
                out.ids.len(),
                c.ring_eos_publishes
            );
            assert_eq!(c.ring_eos_publishes, u64::from(out.eos));
            assert!(
                c.poll_iters >= out.stats.steps,
                "every step waits at least once"
            );
        }
        // The blocking arm publishes nothing, and says so.
        let (out, _) = run(&device, &Arm::Blocking, &[0], &DecodeOptions::new(8))?;
        assert_eq!(out.stats.readback, Default::default());
    }
    Ok(())
}

#[test]
fn the_commit_cadence_follows_the_budget_only_with_lookahead() -> Result<()> {
    for device in devices() {
        // Lookahead on: the loop commits on the budget, PLUS the TTFT ramp —
        // the first `period` tokens each commit (so the first token streams at
        // its compute time, not after a full period; see `decode.rs`), then the
        // steady `steps:4` cadence resumes. 12 steps at steps:4: 4 ramp commits
        // (tokens 1..=4) + 2 steady commits (tokens 8 and 12) = 6. Without the
        // ramp it was 3; the ramp trades 3 extra commits (each reading finished
        // work, no wait) for a ~3× lower first-token latency.
        let opts = DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::steps(4)?,
            ..DecodeOptions::new(12)
        };
        let (out, _) = run(&device, &Arm::RingPoll, &[0], &opts)?;
        assert_eq!(
            out.stats.commits, 6,
            "{device:?}: 12 steps at steps:4 + TTFT ramp"
        );
        // Lookahead off: the publish must be committed before the wait, so the
        // loop commits per token whatever the budget says — the counter is
        // what makes that visible rather than silently ignored.
        let opts = DecodeOptions {
            lookahead: 0,
            budget: CommitBudget::steps(4)?,
            ..DecodeOptions::new(12)
        };
        let (out, _) = run(&device, &Arm::RingPoll, &[0], &opts)?;
        assert_eq!(out.stats.commits, 12, "{device:?}");
        // Per-token, lookahead on: one commit per step.
        let opts = DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::per_token(),
            ..DecodeOptions::new(12)
        };
        let (out, _) = run(&device, &Arm::RingPoll, &[0], &opts)?;
        assert_eq!(out.stats.commits, 12, "{device:?}");
        // The blocking arm's readback commits for itself; the loop makes none,
        // so its command stream is the incumbent's.
        let (out, _) = run(&device, &Arm::Blocking, &[0], &DecodeOptions::new(12))?;
        assert_eq!(out.stats.commits, 0, "{device:?}");
        assert_eq!(out.stats.max_in_flight, 0);
    }
    Ok(())
}

#[test]
fn a_second_decode_continues_from_the_cache_position() -> Result<()> {
    for device in devices() {
        for arm in [Arm::Blocking, Arm::RingPoll] {
            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut spec = IdentityProposer;
            let mut rb = strategy(&arm, &device, 64)?;
            let mut sampler = greedy();
            let opts = DecodeOptions::new(6);
            let first = session.decode(&mut spec, &mut rb, &mut sampler, &[0], &opts, &mut sink)?;
            assert_eq!(first.ids, chain(0, 6).0);
            assert_eq!(session.pos(), 7);
            // The follow-up prompt is prefilled at the cache position; the
            // chain restarts from its last token.
            let second =
                session.decode(&mut spec, &mut rb, &mut sampler, &[5, 1], &opts, &mut sink)?;
            let (want, hit) = chain(1, 6);
            assert_eq!(second.ids, want);
            assert_eq!(second.eos, hit);
            // Only this call's ids, never the first call's again.
            assert_eq!(second.ids.len(), want.len());
            assert_eq!(session.pos(), 7 + 2 + want.len());
            // The ring's ordinal keeps counting across calls; the blocking arm
            // has nothing to count and reports zero both times.
            if rb.arms().token_ring {
                assert_eq!(
                    second.stats.readback.ring_publishes,
                    first.stats.readback.ring_publishes + want.len() as u64 + u64::from(hit),
                    "counters accumulate across calls on {device:?}"
                );
            } else {
                assert_eq!(second.stats.readback, Default::default());
            }
        }
    }
    Ok(())
}

#[test]
fn max_tokens_bounds_the_run_and_a_ring_smaller_than_the_run_is_reused() -> Result<()> {
    let device = Device::Cpu;
    let (out, _) = run(&device, &Arm::RingPoll, &[0], &DecodeOptions::new(5))?;
    assert_eq!(out.ids.len(), 5);
    let model = Chain::new(&device)?;
    let mut session = Session::new(&model, 0usize);
    // `Ring::new` floors the history at `RING_HISTORY_SLOTS`; the history is a
    // wrapping ring whose live window is the commit period, so a run longer
    // than the whole ring reuses it rather than being refused — every id
    // still comes back, in order, and the count reconciles.
    let floor = lloom::generation::readback::RING_HISTORY_SLOTS;
    let (want, _) = chain(0, floor + 2);
    let mut rb = Ring::new(&[EOS], &device, 3, EosRead::Blocking)?;
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        &[0],
        &DecodeOptions::new(floor + 2),
        &mut sink,
    )?;
    assert_eq!(out.ids.len(), floor + 2);
    assert_eq!(out.ids, want);
    assert_eq!(out.stats.readback.ring_publishes, (floor + 2) as u64);
    Ok(())
}

#[test]
fn the_refusals_refuse() -> Result<()> {
    let device = Device::Cpu;
    let model = Chain::new(&device)?;

    // A nucleus sampler cannot ride the ring.
    let mut session = Session::new(&model, 0usize);
    let mut rb = Ring::new(&[EOS], &device, 8, EosRead::Blocking)?;
    let mut top_p = LogitsProcessor::from_sampling(
        0,
        Sampling::TopP {
            p: 0.9,
            temperature: 0.7,
        },
    );
    let err = session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut top_p,
            &[0],
            &DecodeOptions::new(4),
            &mut sink,
        )
        .unwrap_err();
    assert!(err.to_string().contains("device-resident"), "{err}");
    // ...and the blocking arm takes it.
    let mut session = Session::new(&model, 0usize);
    let mut rb = Blocking::new(&[EOS], &device)?;
    session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut top_p,
        &[0],
        &DecodeOptions::new(4),
        &mut sink,
    )?;

    // Lookahead on the blocking arm has nothing to overlap.
    let mut session = Session::new(&model, 0usize);
    let mut rb = Blocking::new(&[EOS], &device)?;
    let opts = DecodeOptions {
        lookahead: 1,
        ..DecodeOptions::new(4)
    };
    let err = session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &[0],
            &opts,
            &mut sink,
        )
        .unwrap_err();
    assert!(err.to_string().contains("lookahead"), "{err}");

    // Lookahead above one is refused on every arm.
    let mut session = Session::new(&model, 0usize);
    let mut rb = Ring::new(&[EOS], &device, 8, EosRead::Blocking)?;
    let opts = DecodeOptions {
        lookahead: 2,
        ..DecodeOptions::new(4)
    };
    let err = session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &[0],
            &opts,
            &mut sink,
        )
        .unwrap_err();
    assert!(err.to_string().contains("lookahead 2"), "{err}");

    // An empty prompt, and an empty EOS set.
    let mut session = Session::new(&model, 0usize);
    let mut rb = Blocking::new(&[EOS], &device)?;
    assert!(session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &[],
            &DecodeOptions::new(4),
            &mut sink,
        )
        .is_err());
    assert!(Blocking::new(&[], &device).is_err());

    // A proposer that drafts is stage 2 and is refused, not run as K=1.
    struct Drafts;
    impl Speculator for Drafts {
        type Probe = NoProbe;
        fn n_seq(&self) -> usize {
            1
        }
        fn max_draft(&self) -> usize {
            3
        }
        fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
            Ok(())
        }
        fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
            Ok(vec![])
        }
        fn observe(
            &mut self,
            _: SeqId,
            _: &<NoProbe as TargetProbe>::Output,
            _: &[u32],
        ) -> Result<()> {
            Ok(())
        }
    }
    let mut session = Session::new(&model, 0usize);
    let mut rb = Blocking::new(&[EOS], &device)?;
    let err = session
        .decode(
            &mut Drafts,
            &mut rb,
            &mut greedy(),
            &[0],
            &DecodeOptions::new(4),
            &mut sink,
        )
        .unwrap_err();
    assert!(err.to_string().contains("decode_speculative"), "{err}");
    Ok(())
}

/// The arms render from the strategy, so a harness cannot report one and run
/// another (§7.1a-i).
#[test]
fn the_arms_render_from_the_strategy() -> Result<()> {
    let device = Device::Cpu;
    let b = Blocking::new(&[EOS], &device)?;
    assert!(!b.arms().token_ring && !b.arms().nonblocking_eos);
    let r = Ring::new(&[EOS], &device, 4, EosRead::Blocking)?;
    assert!(r.arms().token_ring && !r.arms().nonblocking_eos);
    let p = Ring::new(&[EOS], &device, 4, EosRead::Poll(PollMechanism::default()))?;
    assert!(p.arms().token_ring && p.arms().nonblocking_eos);
    assert!(!b.admits_lookahead() && r.admits_lookahead() && p.admits_lookahead());
    // The EOS-learning arm is the strategy's own answer, one spelling each.
    let d = Ring::new(&[EOS], &device, 4, EosRead::HostOnDrain)?;
    assert_eq!(b.eos_read_arm(), "host");
    assert_eq!(r.eos_read_arm(), "flag");
    assert_eq!(p.eos_read_arm(), "poll");
    assert_eq!(d.eos_read_arm(), "drain");
    assert!(d.arms().token_ring && d.arms().nonblocking_eos);
    let g = Ring::new(&[EOS], &device, 4, EosRead::GpuBatched)?;
    let gn = Ring::new(&[EOS], &device, 4, EosRead::GpuBatchedNonblocking)?;
    assert_eq!(g.eos_read_arm(), "gpubatch");
    assert_eq!(gn.eos_read_arm(), "gpubatch-nonblocking");
    assert!(gn.arms().token_ring && gn.arms().nonblocking_eos);
    Ok(())
}

/// The non-blocking batched arm answers every drain from a staged index whose
/// buffer had completed, or from the host scan — and it counts which. Off
/// Metal there is no `eos_index` kernel, so every drain is a host scan; on
/// Metal the staged indices must answer at least some drains, or the arm is
/// the host-drain arm wearing another name. Either way the ids and the EOS
/// position are the host chain's, and the drain never waits on the tail.
#[test]
fn the_nonblocking_batched_arm_counts_which_path_answered_each_drain() -> Result<()> {
    for device in devices() {
        let (want, hit) = chain(1, 64);
        assert!(hit);
        let opts = DecodeOptions {
            lookahead: 1,
            budget: lloom::generation::CommitBudget::steps(4)?,
            ..DecodeOptions::new(64)
        };
        let (out, _) = run(&device, &Arm::RingGpuBatchNonblocking, &[1], &opts)?;
        assert_eq!(out.ids, want, "{device:?}");
        assert!(out.eos, "{device:?}: the chain's EOS ends the run");
        let c = out.stats.readback;
        let drains = c.eos_index_hits + c.eos_index_late;
        assert!(drains > 0, "{device:?}: no drain was ever answered");
        match device {
            Device::Cpu => assert_eq!(c.eos_index_hits, 0, "no eos_index kernel off Metal"),
            _ => assert!(
                c.eos_index_hits > 0,
                "{device:?}: every drain fell back to the host scan (hits 0, late {})",
                c.eos_index_late
            ),
        }
        assert_eq!(
            c.ring_publishes,
            out.ids.len() as u64 + c.ring_eos_publishes + c.ring_discarded,
            "{device:?}: publishes reconcile"
        );
    }
    Ok(())
}

/// The ring feeds the model its own slot, not a fresh upload: the next
/// forward's input is the same allocation every step (§10.3e).
#[test]
fn the_ring_feeds_the_same_slot_every_step() -> Result<()> {
    let device = Device::Cpu;
    let mut rb = Ring::new(&[EOS], &device, 8, EosRead::Blocking)?;
    let logits = Tensor::eye(VOCAB, DType::F32, &device)?
        .narrow(0, 3, 1)?
        .squeeze(0)?;
    let a = rb.sample_and_feed(&mut greedy(), &logits)?;
    assert!(!rb.eos_of_last()?);
    let b = rb.sample_and_feed(&mut greedy(), &logits)?;
    assert!(!rb.eos_of_last()?);
    assert_eq!(a.to_vec2::<u32>()?, vec![vec![3]]);
    // The feed is a `[1, width]` view of the ring's input tile, so two feeds
    // are two views; the invariant is that they are views of ONE allocation.
    {
        let (sa, _) = a.storage_and_layout();
        let (sb, _) = b.storage_and_layout();
        assert!(
            std::ptr::eq(&*sa, &*sb),
            "two feeds returned two allocations"
        );
    }
    assert_eq!(rb.take_ids()?, vec![3, 3]);
    assert_eq!(rb.take_ids()?, Vec::<u32>::new());
    Ok(())
}

/// The host's lead over the GPU is bounded by the step pacer on the arms
/// whose commits the loop owns, and the bound engages only where the
/// per-token wait does not already hold it.
///
/// `Session::decode` takes one marker per forward and waits on the marker
/// `IN_FLIGHT_STEPS` back before encoding the next. On the poll arm the
/// per-token EOS wait retires that forward first, so the pacer finds every
/// marker complete and waits on none; on the drain arm nothing else paces the
/// host, so the pacer is what bounds it, and the queue `flush_and_prune`
/// observes stays within the depth plus the commit the loop itself makes.
#[test]
fn the_pacer_bounds_the_drain_arm_and_is_a_null_on_the_poll_arm() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        let opts = DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::steps(8)?,
            ..DecodeOptions::new(24)
        };
        let (out, _) = run(&device, &Arm::RingHostDrain, &[0], &opts)?;
        // Markers are command buffers, so only a device with a queue takes
        // them; the CPU reports none rather than a count it did not observe.
        // One per forward: the prompt's one unit (`prefill_ubatch = 0`) and a
        // step's, since the prefill rides the same pacer as the decode.
        let want_markers = if matches!(device, Device::Metal(_)) {
            out.stats.steps + 1
        } else {
            0
        };
        assert_eq!(
            out.stats.step_markers, want_markers,
            "{device:?}: one marker per forward"
        );
        if matches!(device, Device::Metal(_)) {
            // **`max_in_flight` counts COMMAND BUFFERS, not forwards.** It is
            // `flush_and_prune`'s return, and a single forward commits several
            // buffers — the pacer's own marker, plus a `CutPolicy` cut every
            // `compute_per_buffer` dispatches, plus the commit budget's. So it
            // is NOT the pacer's depth and must not be compared against
            // `IN_FLIGHT_STEPS`: that comparison held only because this
            // fixture's steps are tiny, and it failed under machine load
            // whenever the driver retired buffers more slowly (lloom #727,
            // measurements §12 — a real run shows `max_in_flight=13` at
            // `depth 2`).
            //
            // What the pacer actually bounds is FORWARDS in flight, and
            // `step_markers` / `step_waits` above already pin that. What is
            // asserted here is the property this arm exists to have: the
            // count is BOUNDED — it does not grow with the run — which is
            // what distinguishes the pacer from the unbounded drain it
            // replaced. A generous ceiling, because the number of buffers per
            // forward is a function of the cut period, not of the pacer.
            assert!(
                out.stats.max_in_flight <= 64,
                "{device:?}: {} command buffers in flight on the drain arm — unbounded growth \
                 is what the pacer exists to prevent (the bound is on FORWARDS; see step_waits)",
                out.stats.max_in_flight,
            );
        }
        let (out, _) = run(&device, &Arm::RingPoll, &[0], &opts)?;
        assert_eq!(out.stats.step_markers, want_markers, "{device:?}");
        assert_eq!(
            out.stats.step_waits, 0,
            "{device:?}: the poll arm's own wait retires the marker before the pacer asks"
        );
        // The blocking arm's readback drains every step: no markers, its
        // command stream the incumbent's.
        let (out, _) = run(&device, &Arm::Blocking, &[0], &DecodeOptions::new(12))?;
        assert_eq!(out.stats.step_markers, 0, "{device:?}");
    }
    Ok(())
}

/// #722 follow-up, hypothesis 9: `TokenRing::publish_marker` stages the
/// generation into `marker_src` with a HOST write and no ordering against the
/// GPU copy that reads it (`write_host_slice`'s SAFETY clause is the caller's).
/// `marker_src` has one tile on the non-streamed geometry. The poll arm's
/// per-step wait orders the rewrite after the copy; the deferred arms have no
/// per-step wait, so at a `per-token` budget under lookahead 1 the tile may be
/// rewritten while the previous step's copy is still queued — the stamp would
/// then read a NEWER generation, and the drain would clamp past a slot the GPU
/// has not written (§3.5). If that happens the ids are not the host chain's.
/// Repeated, because a race does not fail on demand. Reports; does not fix.
#[test]
fn deferred_arms_at_a_per_token_budget_never_drain_an_unwritten_slot() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        for arm in [
            Arm::RingHostDrain,
            Arm::RingGpuBatch,
            Arm::RingGpuBatchNonblocking,
        ] {
            for rep in 0..20 {
                for (start, n) in [(1u32, 64usize), (0, 200)] {
                    let (want, _) = chain(start, n);
                    let opts = DecodeOptions {
                        lookahead: 1,
                        budget: lloom::generation::CommitBudget::per_token(),
                        ..DecodeOptions::new(n)
                    };
                    let (out, _) = run(&device, &arm, &[start], &opts)?;
                    assert_eq!(
                        out.ids, want,
                        "{device:?} {arm:?} rep {rep} from {start}: the drain returned a slot the \
                         GPU had not written (a stale or unwritten token)"
                    );
                }
            }
        }
    }
    Ok(())
}

/// A [`RingReader`] taken before the run reads the same stream the loop
/// returned, straight off the device ring, without touching the loop: the
/// tokens up to the EOS (not included), `ended()` set by the stop id, and the
/// published count equal to the loop's publishes. `None` off Metal.
#[test]
fn a_ring_reader_sees_the_loops_stream_off_the_ring() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        for (start, n) in [(1u32, 64usize), (0, 24)] {
            let (want, hit) = chain(start, n);
            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut spec = IdentityProposer;
            let mut rb = Ring::new(
                &[EOS],
                &device,
                n + 4,
                EosRead::Poll(PollMechanism::default()),
            )?
            .with_poll_budget(20_000);
            let reader = rb.reader();
            let opts = DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(n)
            };
            let mut sampler = greedy();
            let out =
                session.decode(&mut spec, &mut rb, &mut sampler, &[start], &opts, &mut sink)?;
            assert_eq!(out.ids, want);
            match (&device, reader) {
                (Device::Cpu, r) => assert!(r.is_none(), "no mapped words off Metal"),
                (_, None) => panic!("{device:?}: the ring's words were not mapped"),
                (_, Some(mut r)) => {
                    let mut seen = Vec::new();
                    let n_read = r.read_new(&mut seen).expect("the reader was not lapped");
                    assert_eq!(seen, out.ids, "{device:?}: the reader's stream");
                    assert_eq!(n_read, out.ids.len());
                    assert_eq!(r.ended(), hit, "{device:?}: the stop id ends the reader");
                    assert_eq!(
                        r.published().map(u64::from),
                        Some(out.stats.readback.ring_publishes),
                        "{device:?}: the published word is the loop's publish count"
                    );
                    assert_eq!(
                        r.read_new(&mut seen).expect("no lap"),
                        0,
                        "nothing new after"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The TILE (lloom #728): a tiled prefill takes `ceil(prompt / tile)`
/// forwards — counted off the run, not computed from the option — and the
/// ids, the position and the token log are one-shot's. On a ring arm under
/// Metal the pacer takes one marker per forward, prompt units included.
#[test]
fn a_tiled_prefill_takes_the_forwards_it_names_and_changes_nothing_else() -> Result<()> {
    for device in devices() {
        for arm in [Arm::Blocking, Arm::RingPoll] {
            let lookahead = usize::from(matches!(arm, Arm::RingPoll));
            let prompt: Vec<u32> = vec![9, 9, 9, 9, 9, 9, 9, 9, 9, 0];
            let base_opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(8)
            };
            let (base, base_pos) = run(&device, &arm, &prompt, &base_opts)?;
            assert_eq!(base.stats.prefill_units, 1, "one-shot is one forward");
            assert_eq!(base.ids, chain(0, 8).0);
            for (tile, want_units) in [(3usize, 4usize), (5, 2), (10, 1), (64, 1)] {
                let opts = DecodeOptions {
                    prefill_tile: tile,
                    ..base_opts.clone()
                };
                let (out, pos) = run(&device, &arm, &prompt, &opts)?;
                assert_eq!(out.ids, base.ids, "{device:?} {arm:?} tile {tile}: ids");
                assert_eq!(
                    out.stats.prefill_units, want_units,
                    "{device:?} {arm:?} tile {tile}: forwards"
                );
                assert_eq!(pos, base_pos, "{device:?} {arm:?} tile {tile}: position");
                assert_eq!(out.stats.steps, base.stats.steps);
                if matches!(device, Device::Metal(_)) && matches!(arm, Arm::RingPoll) {
                    assert_eq!(
                        out.stats.step_markers,
                        out.stats.steps + want_units as u64,
                        "{device:?} tile {tile}: one marker per forward, prompt units included"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The CHUNK seam (lloom #728): a prompt fed in pieces through
/// `Session::prefill`, then a decode with an empty prompt resuming from the
/// held logits, is one decode over the whole prompt — same ids, same
/// position, same token log — and an empty prefill is a no-op.
#[test]
fn a_prompt_prefilled_in_chunks_then_resumed_is_one_decode() -> Result<()> {
    for device in devices() {
        for arm in [Arm::Blocking, Arm::RingPoll] {
            let lookahead = usize::from(matches!(arm, Arm::RingPoll));
            let opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(6)
            };
            let prompt = [9u32, 9, 9, 9, 9, 9, 9, 0];
            let (reference, ref_pos) = run(&device, &arm, &prompt, &opts)?;

            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut spec = IdentityProposer;
            let mut rb = strategy(&arm, &device, opts.max_tokens + 4)?;
            let mut sampler = greedy();
            assert_eq!(
                session.prefill(&mut rb, &[], &opts)?,
                lloom::nn::PrefillOut::default(),
                "an empty prefill takes no forward"
            );
            let mut done = 0usize;
            for chunk in [3usize, 3, 2] {
                let out = session.prefill(&mut rb, &prompt[done..done + chunk], &opts)?;
                assert_eq!(out.units, 1, "{device:?} {arm:?}: one forward per chunk");
                done += chunk;
                assert_eq!(session.pos(), done);
                assert_eq!(session.tokens(), &prompt[..done]);
            }
            // The tile applies to a chunk too: a 4-token chunk at tile 3 is two forwards.
            let mut tiled = Session::new(&model, 0usize);
            let tiled_opts = DecodeOptions {
                prefill_tile: 3,
                ..opts.clone()
            };
            let mut rb_t = strategy(&arm, &device, opts.max_tokens + 4)?;
            assert_eq!(
                tiled.prefill(&mut rb_t, &prompt[..4], &tiled_opts)?.units,
                2
            );
            assert_eq!(
                tiled.prefill(&mut rb_t, &prompt[4..], &tiled_opts)?.units,
                2
            );

            // Resume: the empty prompt continues from the held logits.
            let out = session.decode(&mut spec, &mut rb, &mut sampler, &[], &opts, &mut sink)?;
            assert_eq!(
                out.ids, reference.ids,
                "{device:?} {arm:?}: the chunked run's ids"
            );
            assert_eq!(out.eos, reference.eos);
            assert_eq!(
                out.stats.prefill_units, 0,
                "a resume takes no prefill forward"
            );
            assert_eq!(session.pos(), ref_pos);
            let out_t = tiled.decode(&mut spec, &mut rb_t, &mut sampler, &[], &opts, &mut sink)?;
            assert_eq!(out_t.ids, reference.ids);
            assert_eq!(tiled.pos(), ref_pos);
        }
    }
    Ok(())
}

/// One run under the adaptive commit with the EOS gate, a reader thread as
/// the loop's EOS signal (the shape `lfm2-chat --stream-reader
/// --adaptive-commit --eos-gate` and `lfm2-release --turns N` run). Returns
/// the run and what the reader saw. `resume` positions the reader at the
/// current stamp (`RingReader::resume_at_published`) — what a follow-up turn
/// needs; a reader at zero would re-read the earlier run and end on its EOS.
fn adaptive_gated_run(
    session: &mut Session<'_, Chain>,
    rb: &mut Ring,
    prompt: &[u32],
    n: usize,
    resume: bool,
) -> Result<(lloom::nn::DecodeOut, Vec<u32>)> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let mut reader = rb.reader().expect("mapped ring");
    if resume {
        reader.resume_at_published();
    }
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let opts = DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        adaptive_commit: true,
        run_ahead: 8,
        eos_gate: true,
        stop: Some(Arc::clone(&stop)),
        ..DecodeOptions::new(n)
    };
    let watcher = {
        let stop = Arc::clone(&stop);
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut seen = Vec::new();
            loop {
                buf.clear();
                if reader.read_new(&mut buf).is_err() {
                    break;
                }
                seen.extend_from_slice(&buf);
                if reader.ended() {
                    stop.store(true, Ordering::Release);
                    break;
                }
                if done.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            seen
        })
    };
    let mut spec = IdentityProposer;
    let mut sampler = greedy();
    let out = session.decode(&mut spec, rb, &mut sampler, prompt, &opts, &mut sink)?;
    done.store(true, Ordering::Release);
    let seen = watcher.join().expect("watcher");
    Ok((out, seen))
}

fn adaptive_ring(device: &Device, arm: EosRead) -> Result<Ring> {
    let geometry = lloom::generation::Geometry {
        batch: 1,
        max_draft: 0,
        period: 8,
        lookahead: 1,
        max_input_width: 1,
        in_flight_depth: 4 + 3 * 8,
    };
    Ring::with_geometry(&[EOS], device, geometry, arm)
}

/// A reader taken for a LATER run on the same ring resumes at the current
/// stamp (`RingReader::resume_at_published`, lloom #735): it sees only the
/// second run's tokens, not the first run's — whose EOS would have ended it
/// at once. Two adaptive runs, as two turns are.
#[test]
fn a_resumed_reader_sees_only_the_later_run() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the reader and the gate are Metal
        }
        let model = Chain::new(&device)?;
        let mut rb = adaptive_ring(&device, EosRead::HostOnDrain)?;
        // Run 1 from 1: reaches EOS.
        let (want1, hit1) = chain(1, 64);
        assert!(hit1);
        let mut session = Session::new(&model, 0usize);
        let (out1, seen1) = adaptive_gated_run(&mut session, &mut rb, &[1], 64, false)?;
        assert_eq!(out1.ids, want1);
        assert_eq!(seen1, want1);
        // Run 2 from 0 on a fresh session, the same ring, a resumed reader.
        let (want2, hit2) = chain(0, 24);
        assert!(!hit2);
        let mut session = Session::new(&model, 0usize);
        let (out2, seen2) = adaptive_gated_run(&mut session, &mut rb, &[0], 24, true)?;
        assert_eq!(out2.ids, want2);
        assert_eq!(
            seen2, want2,
            "{device:?}: the resumed reader's stream is run 2's"
        );
        // The un-resumed reader starts at zero, reads run 1's ids and ends
        // on run 1's EOS — the failure `resume_at_published` exists for.
        let mut stale = rb.reader().expect("mapped ring");
        let mut seen = Vec::new();
        stale.read_new(&mut seen).expect("no lap");
        assert_eq!(seen, want1, "{device:?}: the stale reader reads run 1");
        assert!(stale.ended());
    }
    Ok(())
}

/// The adaptive commit with the EOS gate on the toy (lloom #735 step 4): the
/// reader thread's flag stops the loop, the gate reports it fired, and the
/// overrun is marked unwritten — the plumbing `lfm2-release --turns 3
/// --adaptive-commit --eos-gate` relies on to continue the crown's cache
/// across turns (`issue-735-eos-gate.md` §4). The toy's own kernels are not
/// in the gated set, so what this pins is the loop, the flag and the stats,
/// not the kernels' early-out (`eos_gate_gpu_tests.rs` pins that).
#[test]
fn the_adaptive_commit_with_the_eos_gate_stops_on_the_readers_flag() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the gate and the reader are Metal
        }
        for arm in [
            EosRead::HostOnDrain,
            EosRead::Poll(PollMechanism::default()),
        ] {
            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut rb = adaptive_ring(&device, arm)?;
            let (want, hit) = chain(1, 200);
            assert!(hit);
            let (out, seen) = adaptive_gated_run(&mut session, &mut rb, &[1], 200, false)?;
            assert_eq!(out.ids, want, "{arm:?}: the ids up to the EOS");
            assert!(out.eos, "{arm:?}: EOS reported");
            assert_eq!(seen, want, "{arm:?}: the reader's stream");
            assert!(
                out.stats.stopped_by_flag,
                "{arm:?}: the reader's flag stopped the loop"
            );
            let gate = out.stats.eos_gate.expect("the gate ran");
            assert!(gate.fired, "{arm:?}: the gate saw the EOS: {gate:?}");
            assert_eq!(
                gate.steps, out.stats.steps,
                "{arm:?}: one gate kernel per step"
            );
            assert_eq!(
                out.stats.overrun_unwritten,
                out.stats.overrun > 0,
                "{arm:?}: an overrun past a fired gate is unwritten: {:?}",
                out.stats
            );
            assert_eq!(session.overrun_unwritten(), out.stats.overrun > 0);
            // The EOS's own forward is the first of the overrun (nothing after
            // it is kept), so the position is the prompt, the ids and the overrun.
            assert_eq!(
                session.pos(),
                1 + want.len() + out.stats.overrun,
                "{arm:?}: the position is the prompt, the ids and the overrun"
            );
        }
    }
    Ok(())
}
