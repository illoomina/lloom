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
use lloom::nn::{CausalLM, DecodeOptions, IdentityProposer, Session, Speculator};
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

use lloom as subject;
#[path = "support/chain.rs"]
mod chain_fixture;
use chain_fixture::{Chain, RecordingChain};

/// The CPU, and under `metal` the Metal device. A Metal device that fails to
/// open is a failure, not a skip: every Metal-only assertion in this file
/// would otherwise vanish and the file would still read green.
fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    v.push(Device::new_metal(0).expect("the `metal` feature is on and no Metal device opened"));
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
    fn fed_host_ids(&self) -> Option<&[u32]> {
        match self {
            AnyArm::Blocking(b) => b.fed_host_ids(),
            AnyArm::Ring(r) => r.fed_host_ids(),
        }
    }
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        match self {
            AnyArm::Blocking(b) => b.take_ids_into(out),
            AnyArm::Ring(r) => r.take_ids_into(out),
        }
    }
    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        match self {
            AnyArm::Blocking(b) => b.eos_ids_tensor(),
            AnyArm::Ring(r) => r.eos_ids_tensor(),
        }
    }
    fn reader(&self) -> Option<lloom::generation::readback::RingReader> {
        match self {
            AnyArm::Blocking(b) => b.reader(),
            AnyArm::Ring(r) => ReadbackStrategy::reader(r),
        }
    }
    fn set_fused_publish(&mut self, on: bool) {
        match self {
            AnyArm::Blocking(b) => b.set_fused_publish(on),
            AnyArm::Ring(r) => r.set_fused_publish(on),
        }
    }
    fn publishes_fused(&self) -> bool {
        match self {
            AnyArm::Blocking(b) => b.publishes_fused(),
            AnyArm::Ring(r) => r.publishes_fused(),
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

/// Whether this build records `DecodeStats::step_ns`: the per-step clock is
/// compiled in under `debug_assertions` or `lloom/profile` and out of a plain
/// `--release` build, the build that ships.
const STEP_CLOCK: bool = cfg!(any(debug_assertions, feature = "profile"));

/// `DecodeStats::step_ns` holds one entry per returned token where the clock
/// is compiled in and none where it is not. Asserted both ways, so a release
/// run checks the clock stays off the hot path rather than skipping the check,
/// and every arm after the first still runs.
fn assert_step_clock(stats: &lloom::nn::DecodeStats, tokens: usize, what: &str) {
    let want = if STEP_CLOCK { tokens } else { 0 };
    assert_eq!(
        stats.step_ns.len(),
        want,
        "{what}: step_ns entries (step clock {})",
        if STEP_CLOCK {
            "compiled in"
        } else {
            "compiled out"
        }
    );
    if !STEP_CLOCK {
        assert_eq!(
            stats.step_ns.capacity(),
            0,
            "{what}: an unavailable step clock must not reserve token-sized storage"
        );
        assert!(stats.prefill_census.is_none(), "{what}: prefill census");
        assert!(stats.step0_census.is_none(), "{what}: step-0 census");
    }
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
            assert_step_clock(
                &out.stats,
                24,
                &format!("{device:?} {arm:?} lookahead {lookahead}"),
            );
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
            assert_step_clock(
                &out.stats,
                want.len(),
                &format!("{device:?} {arm:?} lookahead {lookahead}"),
            );
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
        type Family = lloom::nn::speculator::Identity;
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

/// Side of the drain-arm fixture's per-forward ballast matmul: large enough
/// that one forward's GPU time exceeds the host's encode of the next, so a
/// host that nothing paces runs ahead of the device.
const PACER_BALLAST: usize = 2048;

/// The host's lead over the GPU is bounded by the step pacer on the arms
/// whose commits the loop owns, and the bound engages only where the
/// per-token wait does not already hold it (`docs/dispatch.md` §11.5x).
///
/// `Session::decode` commits each forward at its end and, before encoding the
/// next, waits on the marker `pacer_depth` forwards back. On the poll arm the
/// per-token EOS wait retires that forward first, so the pacer finds every
/// marker complete and waits on none. On the drain arm nothing else paces the
/// host, so the pacer is the bound, and a bound is visible only where the
/// host could exceed it: the drain arm runs on a fixture whose forward
/// carries GPU ballast, so the pacer must wait, and without it the host would
/// queue the whole run.
///
/// The bound in the unit `max_in_flight` counts — command buffers not yet
/// complete at a loop commit, not forwards. The queue is FIFO, so once the
/// marker `d` forwards back has completed every buffer before it has too;
/// after it come at most the commit that followed it and two buffers for each
/// of the `d` forwards since (the forward's own marker, and the loop commit
/// carrying the drain marker): `2d + 1`. It holds at the default pipelined
/// cut period, where the fixture's forward never cuts inside a step.
#[test]
fn the_pacer_bounds_the_drain_arm_and_is_a_null_on_the_poll_arm() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        let metal = matches!(device, Device::Metal(_));
        let opts = DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::steps(8)?,
            ..DecodeOptions::new(24)
        };
        let depth = opts.pacer_depth();
        // Ballast on Metal only: markers are command buffers, so the pacer is
        // Metal's, and a CPU matmul of this size would only slow the CPU arm.
        let model = Chain::with_ballast(&device, if metal { PACER_BALLAST } else { 0 })?;
        let mut session = Session::new(&model, 0usize);
        let mut rb = strategy(&Arm::RingHostDrain, &device, opts.max_tokens + 4)?;
        let out = session.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &[0],
            &opts,
            &mut sink,
        )?;
        assert_eq!(
            out.ids,
            chain(0, 24).0,
            "{device:?}: the ballast moves no id"
        );
        // One marker per forward: the prompt's one unit (`prefill_ubatch = 0`)
        // and a step's, since the prefill rides the same pacer as the decode.
        // The CPU reports none rather than a count it did not observe.
        let want_markers = if metal { out.stats.steps + 1 } else { 0 };
        assert_eq!(
            out.stats.step_markers, want_markers,
            "{device:?}: one marker per forward"
        );
        if metal {
            eprintln!(
                "drain arm, ballast {PACER_BALLAST}: depth {depth}, step_waits {}, \
                 max_in_flight {}, commits {}, steps {}",
                out.stats.step_waits, out.stats.max_in_flight, out.stats.commits, out.stats.steps
            );
            assert!(
                out.stats.max_in_flight <= 2 * depth + 1,
                "{device:?}: {} command buffers in flight on the drain arm, bound 2 x depth {depth} \
                 + 1 — the host ran past the marker it should have waited on",
                out.stats.max_in_flight,
            );
            // The precondition, not the property: a host that never had to
            // wait never tested the bound above.
            assert!(
                out.stats.step_waits > 0,
                "{device:?}: the pacer never waited on the drain arm — either it does not wait, \
                 or the ballast did not outrun the host and the bound was not exercised (raise \
                 PACER_BALLAST)"
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

#[test]
fn actual_prefill_work_session_cpu_records_literal_forwards() -> Result<()> {
    eprintln!("EXECUTABLE {}", std::env::current_exe().unwrap().display());
    let model = RecordingChain::cpu()?;
    let mut session = Session::with_pos(&model, 17, 17);
    let mut rb = Blocking::new(&[EOS], &Device::Cpu)?;
    let mut options = DecodeOptions::new(1);
    options.prefill_ubatch = 3;
    options.prefill_tile = 4;
    options.in_flight_depth = 65;
    let prompt = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0];
    assert_eq!(session.prefill(&mut rb, &prompt, &options)?.units, 3);
    assert_eq!(*model.forwards.borrow(), [(17, 3), (20, 4), (24, 4)]);
    eprintln!(
        "ACTUAL Session::prefill tiled forwards={:?}",
        model.forwards.borrow()
    );
    assert_eq!(session.pos(), 28);
    assert_eq!(session.tokens(), prompt);
    assert_eq!(*session.sequence().cache(), 28);
    let held = session.last_logits().unwrap().to_vec1::<f32>()?;
    let mut expected = vec![0.0; VOCAB];
    expected[3] = 1.0;
    assert_eq!(held, expected);
    assert_eq!(
        session.prefill(&mut rb, &[], &options)?,
        lloom::nn::PrefillOut::default()
    );
    assert_eq!(*model.forwards.borrow(), [(17, 3), (20, 4), (24, 4)]);
    assert_eq!(session.last_logits().unwrap().to_vec1::<f32>()?, expected);

    model.forwards.borrow_mut().clear();
    options.prefill_tile = 0;
    options.prefill_schedule = Some(vec![3]);
    options.prefill_bulk = 4;
    options.prefill_pacer_depth = 1;
    assert_eq!(session.prefill(&mut rb, &[9; 9], &options)?.units, 4);
    assert_eq!(
        *model.forwards.borrow(),
        [(28, 3), (31, 4), (35, 1), (36, 1)]
    );
    eprintln!(
        "ACTUAL Session::prefill scheduled forwards={:?}",
        model.forwards.borrow()
    );
    assert_eq!(session.pos(), 37);
    assert_eq!(*session.sequence().cache(), 37);

    model.forwards.borrow_mut().clear();
    options.prefill_schedule = None;
    assert_eq!(session.prefill(&mut rb, &prompt, &options)?.units, 2);
    assert_eq!(*model.forwards.borrow(), [(37, 3), (40, 8)]);
    eprintln!(
        "ACTUAL Session::prefill tile0 forwards={:?}",
        model.forwards.borrow()
    );
    assert_eq!(session.pos(), 48);
    Ok(())
}

#[test]
fn actual_prefill_work_k1_cpu_records_literal_forwards() -> Result<()> {
    let model = RecordingChain::cpu()?;
    let mut session = Session::with_pos(&model, 17, 17);
    let mut rb = Blocking::new(&[EOS], &Device::Cpu)?;
    let options = DecodeOptions {
        prefill_ubatch: 3,
        prefill_tile: 4,
        in_flight_depth: 3,
        prefill_pacer_depth: 8,
        ..DecodeOptions::new(1)
    };
    let prompt = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0];
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        &prompt,
        &options,
        &mut sink,
    )?;
    assert_eq!(out.stats.prefill_units, 3);
    assert_eq!(out.ids, [3]);
    assert!(!out.eos);
    assert_eq!(
        *model.forwards.borrow(),
        [(17, 3), (20, 4), (24, 4), (28, 1)]
    );
    eprintln!(
        "ACTUAL Session::decode K=1 forwards={:?}",
        model.forwards.borrow()
    );
    assert_eq!(session.pos(), 29);
    assert_eq!(*session.sequence().cache(), 29);
    assert_eq!(session.tokens(), [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 3]);
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
    let opts = DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        adaptive_commit: true,
        run_ahead: 8,
        eos_gate: true,
        ..DecodeOptions::new(n)
    };
    reader_run(session, rb, prompt, opts, resume)
}

/// One reader-owned run under `opts` (`stream_via_reader` on): a reader
/// thread streams off the ring and raises the loop's stop flag on the EOS it
/// reads, as `DecodeSession::decode_until` does for `lloom serve`, and takes
/// one last read once the loop has returned. Returns the run and what the
/// reader saw.
fn reader_run<M: CausalLM>(
    session: &mut Session<'_, M>,
    rb: &mut Ring,
    prompt: &[u32],
    opts: DecodeOptions,
    resume: bool,
) -> Result<(lloom::nn::DecodeOut, Vec<u32>)> {
    reader_run_sampled(session, rb, prompt, opts, resume, &mut greedy())
}

/// [`reader_run`] with the caller's sampler.
fn reader_run_sampled<M: CausalLM>(
    session: &mut Session<'_, M>,
    rb: &mut Ring,
    prompt: &[u32],
    mut opts: DecodeOptions,
    resume: bool,
    sampler: &mut LogitsProcessor,
) -> Result<(lloom::nn::DecodeOut, Vec<u32>)> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let mut reader = rb.reader().expect("mapped ring");
    if resume {
        reader.resume_at_published();
    }
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    opts.stop = Some(Arc::clone(&stop));
    let watcher = {
        let stop = Arc::clone(&stop);
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut seen = Vec::new();
            loop {
                let last = done.load(Ordering::Acquire);
                buf.clear();
                if reader.read_new(&mut buf).is_err() {
                    break;
                }
                seen.extend_from_slice(&buf);
                if reader.ended() {
                    stop.store(true, Ordering::Release);
                    break;
                }
                if last {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            seen
        })
    };
    let mut spec = IdentityProposer;
    let out = session.decode(&mut spec, rb, sampler, prompt, &opts, &mut sink)?;
    done.store(true, Ordering::Release);
    let seen = watcher.join().expect("watcher");
    Ok((out, seen))
}

fn adaptive_ring(device: &Device, arm: EosRead) -> Result<Ring> {
    adaptive_ring_for(&[EOS], device, arm)
}

fn adaptive_ring_for(eos: &[u32], device: &Device, arm: EosRead) -> Result<Ring> {
    let geometry = lloom::generation::Geometry {
        batch: 1,
        max_draft: 0,
        period: 8,
        lookahead: 1,
        max_input_width: 1,
        in_flight_depth: 4 + 3 * 8,
    };
    Ring::with_geometry(eos, device, geometry, arm)
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

/// A reader-owned run on the fixed cadence learns EOS per step and retracts
/// the EOS token from the history, whose slot the next write reuses, but not
/// from the stamp — so after every such run the ring's stamps run one further
/// ahead of its slots. A reader taken for a later run and resumed
/// (`RingReader::resume_at_published`) must read that run exactly anyway, on
/// the same arm and on the adaptive one: the shape of `lloom serve --readback
/// ring-la0 | ring-fixed --stream-reader` and of its default arm, request after
/// request on the pool's ring (`Ring::new` at `RING_HISTORY_SLOTS`, as
/// `RingPool` builds it). A reader indexing the slots by the stamp alone reads,
/// from the second run on, the slot after each token's.
#[test]
fn a_resumed_reader_reads_each_later_run_after_retracted_eos() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the reader is Metal
        }
        let model = Chain::new(&device)?;
        let mut rb = Ring::new(
            &[EOS],
            &device,
            lloom::generation::readback::RING_HISTORY_SLOTS,
            EosRead::Poll(PollMechanism::default()),
        )?;
        let fixed = |lookahead: usize, n: usize| DecodeOptions {
            lookahead,
            stream_via_reader: true,
            budget: CommitBudget::steps(8).expect("8 > 0"),
            ..DecodeOptions::new(n)
        };
        let adaptive = |n: usize| DecodeOptions {
            lookahead: 1,
            stream_via_reader: true,
            adaptive_commit: true,
            run_ahead: 8,
            ..DecodeOptions::new(n)
        };
        // (label, opts, start, max, whether the loop retracts the EOS)
        let runs = [
            ("la0 eos", fixed(0, 64), 1u32, 64usize, true),
            ("la0 length", fixed(0, 24), 0, 24, false),
            ("fixed eos", fixed(1, 64), 1, 64, true),
            ("fixed length", fixed(1, 24), 0, 24, false),
            ("adaptive eos", adaptive(64), 1, 64, false),
            ("adaptive length", adaptive(24), 0, 24, false),
        ];
        let mut retracted = 0usize;
        for (label, opts, start, max, retracts) in runs {
            let (want, hit) = chain(start, max);
            let before = rb.generation() as usize;
            let mut session = Session::new(&model, 0usize);
            let (out, seen) = reader_run(&mut session, &mut rb, &[start], opts, true)?;
            assert_eq!(out.ids, want, "{device:?} {label}: the loop's ids");
            assert_eq!(out.eos, hit, "{device:?} {label}: EOS");
            assert_eq!(
                seen, want,
                "{device:?} {label}: the resumed reader's stream, after {retracted} retracted EOS"
            );
            if retracts {
                // The shift the next reader absorbs: the EOS was stamped and
                // its slot was retracted, so the run stamped one more
                // generation than the ids it kept.
                assert_eq!(
                    rb.generation() as usize - before,
                    want.len() + 1,
                    "{device:?} {label}: the retracted EOS was stamped"
                );
                retracted += 1;
            }
        }
        assert_eq!(retracted, 2);
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

/// The adaptive commit's options for a run the test stops itself: no reader
/// thread (the loop's drain is not the reader's, so the test reads the ids off
/// the returned run), `run_ahead` steps a buffer at most, the gate as asked.
fn adaptive_opts(n: usize, run_ahead: usize, eos_gate: bool) -> DecodeOptions {
    DecodeOptions {
        lookahead: 1,
        stream_via_reader: true,
        adaptive_commit: true,
        run_ahead,
        eos_gate,
        ..DecodeOptions::new(n)
    }
}

/// An adaptive run longer than the ring's history must not lap it. Under the
/// adaptive commit (and under `stream_via_reader` on the fixed cadence) the loop
/// does not drain the history per commit — the reader streams — so nothing
/// advances the history's drain cursor until the run ends, and a run of more
/// steps than the ring has tiles meets the lap guard. `Ring::new` is what the
/// serving binary builds (`RingPool`, `RING_HISTORY_SLOTS` tiles, no consumer
/// slack), so the length that meets it is 4092 tokens.
#[test]
fn a_reader_owned_run_longer_than_the_ring_does_not_lap_it() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the adaptive commit and the reader are Metal
        }
        let floor = lloom::generation::readback::RING_HISTORY_SLOTS;
        let n = floor + 900;
        let (want, hit) = chain(0, n);
        assert!(!hit, "the chain from 0 never reaches EOS");
        for (name, opts) in [
            ("adaptive", adaptive_opts(n, 8, false)),
            (
                "fixed cadence, reader-owned",
                DecodeOptions {
                    lookahead: 1,
                    budget: CommitBudget::steps(8)?,
                    stream_via_reader: true,
                    ..DecodeOptions::new(n)
                },
            ),
        ] {
            let model = Chain::new(&device)?;
            let mut session = Session::new(&model, 0usize);
            let mut rb = Ring::new(&[EOS], &device, 8, EosRead::Poll(PollMechanism::default()))?
                .with_poll_budget(20_000);
            let out = session
                .decode(
                    &mut IdentityProposer,
                    &mut rb,
                    &mut greedy(),
                    &[0],
                    &opts,
                    &mut sink,
                )
                .map_err(|e| lloom_tensor::Error::Msg(format!("{name}: {e}")))?;
            assert_eq!(out.ids.len(), n, "{name}: every token comes back");
            assert_eq!(out.ids, want, "{name}: the ids are the chain's");
            assert_eq!(session.pos(), 1 + n, "{name}: the position");
        }
    }
    Ok(())
}

/// The overrun past an EOS is reconciled for any depth (lloom #735): the loop
/// derives it from the cache position, so however many forwards the host had
/// queued behind the EOS, the ids end at the EOS, the position is the prompt,
/// the ids and the overrun, and the publishes reconcile. The overrun is forced
/// here rather than raced: no reader thread raises the stop flag, so the run
/// takes exactly `max_tokens` steps and the EOS is found in the drained ids at
/// the end, `extra` forwards before the cap. With the gate on, every one of
/// those forwards is marked unwritten.
///
/// Also pins what the ring's input side holds at that depth: one tile per
/// step the host may lead by, so no input write lands on a tile a queued
/// forward still reads (`TokenRing::input_slot_count`).
#[test]
fn the_adaptive_overrun_is_exact_at_every_depth() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the adaptive commit and the gate are Metal
        }
        let (want, hit) = chain(1, 64);
        assert!(hit);
        for run_ahead in [1usize, 2, 4, 8, 16] {
            for extra in [0usize, 1, 5, 23] {
                for gate in [false, true] {
                    let max_tokens = want.len() + 1 + extra;
                    let opts = adaptive_opts(max_tokens, run_ahead, gate);
                    let model = Chain::with_ballast(&device, 256)?;
                    let mut session = Session::new(&model, 0usize);
                    let mut rb = adaptive_ring(&device, EosRead::Poll(PollMechanism::default()))?;
                    assert!(
                        rb.token_ring().input_slot_count() > opts.adaptive_lead(),
                        "{} input tiles for a lead of {} steps",
                        rb.token_ring().input_slot_count(),
                        opts.adaptive_lead()
                    );
                    let out = session.decode(
                        &mut IdentityProposer,
                        &mut rb,
                        &mut greedy(),
                        &[1],
                        &opts,
                        &mut sink,
                    )?;
                    let tag = format!("run_ahead {run_ahead} extra {extra} gate {gate}");
                    assert_eq!(out.ids, want, "{tag}: the ids end at the EOS");
                    assert!(out.eos, "{tag}: EOS found in the drained ids");
                    assert_eq!(out.stats.steps as usize, max_tokens, "{tag}: steps");
                    assert_eq!(
                        out.stats.overrun,
                        1 + extra,
                        "{tag}: the EOS's own forward and every one after it"
                    );
                    assert_eq!(session.pos(), 1 + want.len() + 1 + extra, "{tag}: position");
                    assert_eq!(session.overrun(), 1 + extra, "{tag}");
                    assert_eq!(
                        out.stats.overrun_unwritten, gate,
                        "{tag}: unwritten iff gated"
                    );
                    if gate {
                        let g = out.stats.eos_gate.expect("the gate ran");
                        assert!(g.fired, "{tag}: the gate saw the EOS");
                    }
                    let c = out.stats.readback;
                    assert_eq!(
                        c.ring_publishes,
                        out.ids.len() as u64 + c.ring_eos_publishes + c.ring_discarded,
                        "{tag}: publishes reconcile: {c:?}"
                    );
                    assert!(
                        out.stats.max_steps_per_buffer <= run_ahead,
                        "{tag}: {} steps in one buffer",
                        out.stats.max_steps_per_buffer
                    );
                }
            }
        }
    }
    Ok(())
}

/// A two-layer LFM2 on the crown's cache (`Shuffle` conv state, in-place KV —
/// rewind horizon 0), deterministic weights: one conv block and one attention
/// block, so both kinds of persistent state a forward writes are present.
#[cfg(feature = "metal")]
mod tiny_lfm2 {
    use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model};
    use lloom::nn::VarBuilder;
    use lloom_tensor::{DType, Device, Result, Tensor};

    pub fn config(conv_state: ConvState, conv_fused: bool) -> Config {
        Config {
            vocab_size: 64,
            hidden_size: 512,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 128,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: vec![LayerType::Conv, LayerType::FullAttention],
            tie_embedding: true,
            kv_append: KvAppend::InPlace,
            conv_state,
            conv_fused,
            ..Default::default()
        }
    }

    pub fn build<P: lloom::models::lfm2::Shape>(cfg: &Config, dev: &Device) -> Result<Model<P>> {
        let mut tensors = std::collections::HashMap::new();
        let mut fill = |name: &str, shape: Vec<usize>, salt: usize| -> Result<()> {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 7 + salt * 13) % 23) as f32 - 11.0) / 48.0)
                .collect();
            tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
            Ok(())
        };
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        fill("model.embed_tokens.weight", vec![cfg.vocab_size, h], 1)?;
        fill("model.embedding_norm.weight", vec![h], 2)?;
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let s = 10 * (l + 1);
            fill(&format!("{p}.operator_norm.weight"), vec![h], s)?;
            fill(&format!("{p}.ffn_norm.weight"), vec![h], s + 1)?;
            fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h], s + 2)?;
            fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h], s + 3)?;
            fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i], s + 4)?;
            match cfg.layer_types[l] {
                LayerType::Conv => {
                    fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h], s + 5)?;
                    fill(
                        &format!("{p}.conv.conv.weight"),
                        vec![h, 1, cfg.conv_l_cache],
                        s + 6,
                    )?;
                    fill(&format!("{p}.conv.out_proj.weight"), vec![h, h], s + 7)?;
                }
                LayerType::FullAttention => {
                    fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h], s + 5)?;
                    fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h], s + 6)?;
                    fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h], s + 7)?;
                    fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h], s + 8)?;
                    let hd = cfg.head_dim();
                    fill(
                        &format!("{p}.self_attn.q_layernorm.weight"),
                        vec![hd],
                        s + 9,
                    )?;
                    fill(
                        &format!("{p}.self_attn.k_layernorm.weight"),
                        vec![hd],
                        s + 10,
                    )?;
                }
            }
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F16, dev);
        Model::<P>::new(cfg, vb)
    }

    /// Every conv block's live window — the `l_cache` columns the next decode
    /// step reads — as f32 bits. The whole state on `Shuffle`; on a rotating
    /// ring the slots ending at the phase, since the ring's other slots hold
    /// history a rewind has already stepped back over.
    pub fn conv_windows<P: lloom::profile::Profile>(
        cache: &lloom::models::lfm2::State<P>,
        cfg: &Config,
    ) -> Result<Vec<Vec<u32>>> {
        let l = cfg.conv_l_cache;
        let mut out = Vec::new();
        for b in 0..cfg.num_hidden_layers {
            let Some(t) = cache.conv_state_for_test(b) else {
                continue;
            };
            let window = match cfg.conv_state {
                ConvState::RotatingRing { k } => {
                    let width = l + k;
                    let phase = cache.conv_phase();
                    let cols = (0..l)
                        .rev()
                        .map(|back| t.narrow(2, (phase + width - back) % width, 1))
                        .collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&cols, 2)?
                }
                _ => t,
            };
            let v = window
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            out.push(v.into_iter().map(f32::to_bits).collect());
        }
        Ok(out)
    }
}

/// What one conv-state arm of the EOS-gate cache test observed; see
/// [`gate_cache_case`].
#[cfg(feature = "metal")]
struct GateCase {
    /// The gate armed, the cap reached before the EOS is sampled: it never
    /// fires. The conv state against the reference's.
    armed_unfired_matches: bool,
    /// The gate fired with `EXTRA + 1` forwards queued past the EOS; after
    /// the reconcile: the position, the KV length, the conv state, and a
    /// one-token second turn — the turn that reads the conv state (a
    /// multi-token prefill rebuilds it from its own activations).
    fired_reconciled: std::result::Result<(), String>,
    pos_matches: bool,
    kv_len_matches: bool,
    conv_matches: bool,
    turn2_one_token_matches: bool,
    turn2_prefill_matches: bool,
    /// The same run without the gate: the reconcile's outcome (a rewind, which
    /// only a ring can do) and whether the conv state then matched.
    ungated_reconciled: std::result::Result<usize, String>,
    ungated_conv_matches: bool,
}

/// One run of the crown's flow on `model` from `cache`: the ring polled, the
/// adaptive commit, the fused publish, the gate as asked.
#[cfg(feature = "metal")]
fn gated_flow<'m, P: lloom::models::lfm2::Shape>(
    model: &'m lloom::models::lfm2::Model<P>,
    cache: Result<lloom::models::lfm2::State<P>>,
    eos: u32,
    dev: &Device,
    gate: bool,
    max_tokens: usize,
) -> Result<(
    Session<'m, lloom::models::lfm2::Model<P>>,
    lloom::nn::DecodeOut,
)> {
    let mut s = Session::new(model, cache?);
    let mut rb = adaptive_ring_for(&[eos], dev, EosRead::Poll(PollMechanism::default()))?;
    let out = s.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        &[1, 2, 3, 4, 5],
        &adaptive_opts(max_tokens, 8, gate),
        &mut sink,
    )?;
    assert!(
        out.stats.readback.fused_publishes > 0,
        "the crown's fused publish ran"
    );
    Ok((s, out))
}

/// The EOS gate on the real decode kernels (lloom #735 step 4's definition of
/// done), on a two-layer LFM2 with in-place KV and the given conv arm. The
/// reference is the blocking arm, which never feeds the EOS token. The subject
/// is the crown's flow — the ring polled, the adaptive commit, the fused
/// publish, the gate — made to run `EXTRA` forwards past the EOS (the cap is
/// reached before any stop flag: there is no reader). After
/// `Session::reconcile` the subject's cache should be the reference's.
#[cfg(feature = "metal")]
fn gate_cache_case(
    conv_state: lloom::models::lfm2::ConvState,
    conv_fused: bool,
) -> Result<GateCase> {
    let dev = Device::new_metal(0).expect("metal");
    let cfg = tiny_lfm2::config(conv_state, conv_fused);
    let key = cfg.exact_key();
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => gate_cache_case_at::<P>(&cfg, &dev))
}

/// `gate_cache_case` at the profile the config names.
#[cfg(feature = "metal")]
fn gate_cache_case_at<P: lloom::models::lfm2::Shape>(
    cfg: &lloom::models::lfm2::Config,
    dev: &Device,
) -> Result<GateCase> {
    use lloom::models::lfm2::State;
    const EXTRA: usize = 12;
    let prompt = [1u32, 2, 3, 4, 5];
    let model = tiny_lfm2::build::<P>(&cfg, &dev)?;
    let fresh = || State::<P>::new(true, DType::F16, &cfg, &dev);
    let never = [u32::MAX];
    let greedy_blocking = |s: &mut Session<'_, lloom::models::lfm2::Model<P>>,
                           eos: &[u32],
                           p: &[u32],
                           n: usize|
     -> Result<Vec<u32>> {
        Ok(s.decode(
            &mut IdentityProposer,
            &mut Blocking::new(eos, &dev)?,
            &mut greedy(),
            p,
            &DecodeOptions::new(n),
            &mut sink,
        )?
        .ids)
    };

    // The stream with no stop token; the EOS is its first token not seen
    // earlier, at least four in.
    let stream = greedy_blocking(&mut Session::new(&model, fresh()?), &never, &prompt, 32)?;
    let k = (4..stream.len())
        .find(|&k| !stream[..k].contains(&stream[k]))
        .unwrap_or_else(|| panic!("no fresh token in the stream to stop on: {stream:?}"));
    let eos = stream[k];
    let want = &stream[..k];

    // The reference stops at the EOS, feeding nothing past it.
    let mut reference = Session::new(&model, fresh()?);
    assert_eq!(greedy_blocking(&mut reference, &[eos], &prompt, 64)?, want);
    let ref_pos = reference.pos();
    let ref_kv = reference.sequence().cache().kv_len();
    let ref_conv = tiny_lfm2::conv_windows(reference.sequence().cache(), &cfg)?;
    assert!(!ref_conv.is_empty(), "the fixture has a conv block");
    // Two second turns, each from the reference's state: one token (the
    // decode path, which reads the conv state) and a prefill.
    let ref_turn2_one = {
        let mut r = Session::new(&model, fresh()?);
        greedy_blocking(&mut r, &[eos], &prompt, 64)?;
        greedy_blocking(&mut r, &never, &[9], 12)?
    };
    let ref_turn2_prefill = greedy_blocking(&mut reference, &never, &[9, 3], 12)?;

    let flow =
        |gate: bool, max_tokens: usize| gated_flow(&model, fresh(), eos, &dev, gate, max_tokens);

    // Armed, not fired: `k` steps feed `want` and never sample the EOS.
    let (armed, out) = flow(true, k)?;
    assert_eq!(out.ids, want);
    assert!(!out.eos && out.stats.overrun == 0);
    assert!(!out.stats.eos_gate.expect("armed").fired);
    let armed_unfired_matches = armed.pos() == ref_pos
        && tiny_lfm2::conv_windows(armed.sequence().cache(), &cfg)? == ref_conv;

    // Fired: `EXTRA + 1` forwards queued past the EOS, the EOS's own first.
    let (mut fired, out) = flow(true, k + 1 + EXTRA)?;
    assert_eq!(out.ids, want, "the ids end at the EOS");
    assert!(out.eos);
    assert_eq!(out.stats.overrun, 1 + EXTRA, "the forced overrun");
    assert!(
        out.stats.eos_gate.expect("armed").fired,
        "the gate saw the EOS"
    );
    assert!(
        out.stats.overrun_unwritten,
        "every forward past the EOS is unwritten"
    );
    let fired_reconciled = fired.reconcile().map(|_| ()).map_err(|e| e.to_string());
    let pos_matches = fired.pos() == ref_pos;
    let kv_len_matches = fired.sequence().cache().kv_len() == ref_kv;
    let conv_matches = tiny_lfm2::conv_windows(fired.sequence().cache(), &cfg)? == ref_conv;
    let (turn2_one_token_matches, turn2_prefill_matches) = if fired_reconciled.is_ok() {
        let one = greedy_blocking(&mut fired, &never, &[9], 12)? == ref_turn2_one;
        let (mut again, _) = flow(true, k + 1 + EXTRA)?;
        again.reconcile()?;
        let prefill = greedy_blocking(&mut again, &never, &[9, 3], 12)? == ref_turn2_prefill;
        (one, prefill)
    } else {
        (false, false)
    };

    // The control: the same run, no gate. The overrun wrote the cache.
    let (mut ungated, out) = flow(false, k + 1 + EXTRA)?;
    assert!(!out.stats.overrun_unwritten && out.stats.overrun == 1 + EXTRA);
    let ungated_reconciled = ungated.reconcile().map_err(|e| e.to_string());
    let ungated_conv_matches =
        tiny_lfm2::conv_windows(ungated.sequence().cache(), &cfg)? == ref_conv;

    Ok(GateCase {
        armed_unfired_matches,
        fired_reconciled,
        pos_matches,
        kv_len_matches,
        conv_matches,
        turn2_one_token_matches,
        turn2_prefill_matches,
        ungated_reconciled,
        ungated_conv_matches,
    })
}

/// On a rotating conv ring the gate's overrun reconciles to the reference's
/// cache exactly: position, KV length, conv state and both second turns. The
/// armed-but-unfired arm pins that the gated kernel variants are the plain
/// ones while the flag is clear; the ungated control reconciles too, through
/// the ring's own history (a rewind), which is what `RotatingRing` is for.
#[cfg(feature = "metal")]
#[test]
fn the_eos_gate_reconciles_to_the_blocking_arms_cache_on_a_conv_ring() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let c = gate_cache_case(
        lloom::models::lfm2::ConvState::RotatingRing { k: 32 },
        false,
    )?;
    report_gate_case("rotating:32", &c);
    assert_gate_case(&c, false);
    assert!(c.ungated_reconciled.is_ok(), "ungated: the ring rewinds");
    assert!(
        c.ungated_conv_matches,
        "ungated: rewound to the reference's state"
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn assert_gate_case(c: &GateCase, horizon_zero: bool) {
    assert!(
        c.armed_unfired_matches,
        "armed, unfired: the reference's state"
    );
    assert_eq!(c.fired_reconciled, Ok(()), "fired: the reconcile");
    assert!(
        c.pos_matches && c.kv_len_matches,
        "fired: position and KV length"
    );
    assert!(
        c.conv_matches,
        "fired: every conv block's live window, bit for bit"
    );
    assert!(c.turn2_one_token_matches, "fired: a one-token second turn");
    assert!(c.turn2_prefill_matches, "fired: a prefilled second turn");
    if horizon_zero {
        assert!(
            c.ungated_reconciled.is_err(),
            "ungated: a horizon-0 cache cannot rewind"
        );
        assert!(
            !c.ungated_conv_matches,
            "ungated: the overrun moved the state"
        );
    }
}

#[cfg(feature = "metal")]
fn report_gate_case(name: &str, c: &GateCase) {
    eprintln!(
        "{name}: armed-unfired {} | fired: reconcile {:?} pos {} kv {} conv {} turn2(1 token) {} \
         turn2(prefill) {} | ungated: reconcile {:?} conv {}",
        c.armed_unfired_matches,
        c.fired_reconciled,
        c.pos_matches,
        c.kv_len_matches,
        c.conv_matches,
        c.turn2_one_token_matches,
        c.turn2_prefill_matches,
        c.ungated_reconciled,
        c.ungated_conv_matches
    );
}

/// The crown's conv arm — `Shuffle` with the fused step (`conv_fused`) — under
/// the gate: the reconcile is counters only (horizon 0), so the cache can
/// match the reference only if nothing past the EOS reached it. The fused step
/// writes the next state into a buffer the model swaps into the cache every
/// step, which is why its gated variant carries the state through rather than
/// returning bare (`conv.metal` `shortconv_state_carry`). The ungated control
/// is the level: its reconcile is refused and its state has moved.
#[cfg(feature = "metal")]
#[test]
fn the_eos_gate_reconciles_to_the_blocking_arms_cache_on_the_crown_shuffle() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let c = gate_cache_case(lloom::models::lfm2::ConvState::Shuffle, true)?;
    report_gate_case("shuffle, fused step", &c);
    assert_gate_case(&c, true);
    Ok(())
}

/// `Shuffle` WITHOUT the fused step, under the gate: the reconcile is counters
/// only (horizon 0), so the cache matches the reference only if no write past
/// the EOS reached the state. The unfused step shifts the state in place, in
/// the cache's own storage, with gated strided copies (`lfm2.rs`
/// `convolve_bcx`); a step that built the next state in a fresh tensor and
/// swapped it in would leave the cache holding a tensor no kernel wrote
/// (`docs/dispatch.md` §11.5z-x). The second turns are no discriminator on
/// this fixture — its greedy ids came out equal with the state wrong — and a
/// prefilled turn rebuilds the state from its own activations whatever it was,
/// which is why a multi-turn digest cannot see this: the conv state is the
/// assertion. The ungated control is the level: its reconcile is refused and
/// its state has moved.
#[cfg(feature = "metal")]
#[test]
fn the_eos_gate_reconciles_to_the_blocking_arms_cache_on_the_unfused_shuffle() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let c = gate_cache_case(lloom::models::lfm2::ConvState::Shuffle, false)?;
    report_gate_case("shuffle, unfused step", &c);
    assert_gate_case(&c, true);
    Ok(())
}

/// What the gate's overrun did to a KV slab that grew inside it; see
/// [`gate_growth_case`].
#[cfg(feature = "metal")]
#[derive(Debug)]
struct GrowthCase {
    /// The slab's width before the run and after it, and the KV length at
    /// the boundary (the reference's).
    slab_before: usize,
    slab_after: usize,
    boundary_kv: usize,
    /// The subject's reconcile, then its next step's logits against the
    /// reference's next step at the same position (max |diff|).
    reconciled: std::result::Result<(), String>,
    next_step_diff: f32,
}

/// The EOS gate on the real decode kernels when the KV slab grows inside the
/// gated overrun (`Cache::prepare_pass` widens it by a bucket before the
/// forward that needs it; the grow copies the old slab into the new one's
/// prefix, `nn::slab::Slab::grow_to`). The gate is armed for the whole run,
/// and the copy is a gated kernel (`copy2d`). The bucket is chosen so the
/// reference (the blocking arm, which never feeds the EOS) never grows and
/// the subject grows three forwards past the EOS, after the gate fired. The
/// subject is then reconciled and both take the same next step; the
/// ungated control grows at the same forward, ungated, and is rewound (a
/// rotating conv ring rewinds).
#[cfg(feature = "metal")]
fn gate_growth_case(gate: bool) -> Result<GrowthCase> {
    let dev = Device::new_metal(0).expect("metal");
    let cfg = tiny_lfm2::config(
        lloom::models::lfm2::ConvState::RotatingRing { k: 32 },
        false,
    );
    let key = cfg.exact_key();
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => gate_growth_case_at::<P>(&cfg, &dev, gate))
}

#[cfg(feature = "metal")]
fn gate_growth_case_at<P: lloom::models::lfm2::Shape>(
    cfg: &lloom::models::lfm2::Config,
    dev: &Device,
    gate: bool,
) -> Result<GrowthCase> {
    use lloom::models::lfm2::State;
    const EXTRA: usize = 12;
    let prompt = [1u32, 2, 3, 4, 5];
    let model = tiny_lfm2::build::<P>(cfg, dev)?;
    let never = [u32::MAX];
    let blocking = |s: &mut Session<'_, lloom::models::lfm2::Model<P>>,
                    eos: &[u32],
                    p: &[u32],
                    n: usize|
     -> Result<Vec<u32>> {
        Ok(s.decode(
            &mut IdentityProposer,
            &mut Blocking::new(eos, dev)?,
            &mut greedy(),
            p,
            &DecodeOptions::new(n),
            &mut sink,
        )?
        .ids)
    };
    let fresh = |bucket: usize| State::<P>::new(true, DType::F16, cfg, dev)?.with_kv_bucket(bucket);
    let stream = blocking(&mut Session::new(&model, fresh(64)?), &never, &prompt, 32)?;
    let k = (4..stream.len())
        .find(|&k| !stream[..k].contains(&stream[k]))
        .unwrap_or_else(|| panic!("no fresh token in the stream to stop on: {stream:?}"));
    let eos = stream[k];
    // The boundary holds the prompt and the `k` ids; the EOS's forward is
    // the overrun's first, so a bucket two past the boundary is crossed by
    // the overrun's third forward.
    let boundary_kv = prompt.len() + k;
    let bucket = boundary_kv + 2;

    let mut reference = Session::new(&model, fresh(bucket)?);
    assert_eq!(blocking(&mut reference, &[eos], &prompt, 64)?, &stream[..k]);
    assert_eq!(reference.sequence().cache().kv_len(), boundary_kv);
    let next = Tensor::new(&[[9u32]], dev)?;
    let pos = reference.pos();
    let mut want = reference.cache_mut().clone();
    let want_logits = CausalLM::forward(&model, &next, pos, &mut want)?;

    let slab_before = fresh(bucket)?.kv_report().slab;
    let (mut subject, out) = gated_flow(&model, fresh(bucket), eos, dev, gate, k + 1 + EXTRA)?;
    assert_eq!(out.ids, &stream[..k]);
    assert_eq!(out.stats.overrun, 1 + EXTRA);
    assert_eq!(out.stats.overrun_unwritten, gate);
    let slab_after = subject.sequence().cache().kv_report().slab;
    let reconciled = subject.reconcile().map(|_| ()).map_err(|e| e.to_string());
    let next_step_diff = if reconciled.is_ok() {
        assert_eq!(subject.pos(), pos);
        let got = CausalLM::forward(&model, &next, pos, subject.cache_mut())?;
        let (a, b) = (
            got.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
            want_logits
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
        );
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    } else {
        f32::NAN
    };
    Ok(GrowthCase {
        slab_before,
        slab_after,
        boundary_kv,
        reconciled,
        next_step_diff,
    })
}

/// A KV grow inside the gate's overrun keeps the slab's prefix: after the
/// reconcile, the next step is the reference's, bit for bit, gated or not.
#[cfg(feature = "metal")]
#[test]
fn a_kv_grow_inside_the_gated_overrun_keeps_the_prefix() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for gate in [false, true] {
        let c = gate_growth_case(gate)?;
        eprintln!("GATE-GROWTH gate={gate}: {c:?}");
        assert!(
            c.boundary_kv < c.slab_before,
            "gate={gate}: the boundary fits the slab, so the growth is the overrun's: {c:?}"
        );
        assert!(
            c.slab_after > c.slab_before,
            "gate={gate}: the overrun grew the slab: {c:?}"
        );
        assert_eq!(c.reconciled, Ok(()), "gate={gate}: the reconcile");
        assert_eq!(
            c.next_step_diff, 0.0,
            "gate={gate}: the step after the grown, reconciled overrun is the reference's"
        );
    }
    Ok(())
}

/// A reader-owned run on an arm that learns EOS late — the drain arms, the
/// adaptive commit's arms — ends its ids at the EOS whatever stopped it: the
/// loop does not stream (the reader does), so the EOS and the forwards past it
/// are in the ring's ids when the run ends, and the loop finds it there. Here
/// nothing raises the stop flag, so the cap stops the run `EXTRA` steps past
/// the EOS, on the fixed cadence and on the adaptive commit alike.
#[test]
fn a_reader_owned_run_ends_its_ids_at_the_eos_on_every_cadence() -> Result<()> {
    const EXTRA: usize = 6;
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        let metal = matches!(device, Device::Metal(_));
        let (want, hit) = chain(1, 64);
        assert!(hit);
        let n = want.len() + 1 + EXTRA;
        let mut arms = vec![(
            "fixed cadence, host drain",
            Arm::RingHostDrain,
            DecodeOptions {
                lookahead: 1,
                budget: CommitBudget::steps(4)?,
                stream_via_reader: true,
                ..DecodeOptions::new(n)
            },
        )];
        if metal {
            arms.push((
                "adaptive, host drain",
                Arm::RingHostDrain,
                adaptive_opts(n, 4, false),
            ));
            arms.push(("adaptive, poll", Arm::RingPoll, adaptive_opts(n, 4, false)));
        }
        for (name, arm, opts) in arms {
            let (out, pos) = run(&device, &arm, &[1], &opts)?;
            assert_eq!(out.ids, want, "{device:?} {name}: the ids end at the EOS");
            assert!(out.eos, "{device:?} {name}: EOS reported");
            assert_eq!(
                out.stats.overrun,
                1 + EXTRA,
                "{device:?} {name}: the overrun"
            );
            assert_eq!(pos, 1 + n, "{device:?} {name}: every step ran a forward");
            let c = out.stats.readback;
            assert_eq!(
                c.ring_publishes,
                out.ids.len() as u64 + c.ring_eos_publishes + c.ring_discarded,
                "{device:?} {name}: publishes reconcile: {c:?}"
            );
        }
    }
    Ok(())
}

/// What one run handed its consumer.
struct Delivered {
    out: lloom::nn::DecodeOut,
    /// The consumer's stream: the reader's reads on a reader-owned run,
    /// `on_tokens`' batches concatenated on a run the loop drains.
    stream: Vec<u32>,
    /// `on_tokens`' batch sizes, in order; empty on a reader-owned run.
    batches: Vec<usize>,
}

/// One run of `arm` under `opts` on the chain fixture, streamed the way the
/// options say: through a reader thread when `stream_via_reader` is on, else
/// through `on_tokens`.
fn deliver(device: &Device, arm: &Arm, prompt: &[u32], opts: DecodeOptions) -> Result<Delivered> {
    let model = Chain::new(device)?;
    let mut session = Session::new(&model, 0usize);
    let mut rb = strategy(arm, device, opts.max_tokens + 4)?;
    if opts.stream_via_reader {
        let AnyArm::Ring(ring) = &mut rb else {
            panic!("a reader-owned run needs a ring, not {arm:?}");
        };
        let (out, stream) = reader_run(&mut session, ring, prompt, opts, false)?;
        return Ok(Delivered {
            out,
            stream,
            batches: Vec::new(),
        });
    }
    let mut stream = Vec::new();
    let mut batches = Vec::new();
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut greedy(),
        prompt,
        &opts,
        &mut |batch: &[u32]| {
            stream.extend_from_slice(batch);
            batches.push(batch.len());
            Ok(())
        },
    )?;
    Ok(Delivered {
        out,
        stream,
        batches,
    })
}

/// The fixed cadence at `lookahead`, `steps:8`: the `ring-la0` / `ring-fixed`
/// presets' loop options (`lloom-engine` `ReadbackConfig::apply_spelling`).
fn fixed_cadence(n: usize, lookahead: usize) -> DecodeOptions {
    DecodeOptions {
        lookahead,
        budget: CommitBudget::steps(8).expect("8 > 0"),
        ..DecodeOptions::new(n)
    }
}

/// Every arm the loop keeps, named as `lloom --readback` spells it: the
/// blocking reference, the fixed cadence on each EOS read, and — where there
/// is a reader, Metal — the fixed cadence streamed by a reader and the flow.
fn cadence_arms(n: usize, metal: bool) -> Vec<(&'static str, Arm, DecodeOptions)> {
    let mut arms = vec![
        ("blocking", Arm::Blocking, DecodeOptions::new(n)),
        ("ring-la0", Arm::RingPoll, fixed_cadence(n, 0)),
        ("ring-fixed", Arm::RingPoll, fixed_cadence(n, 1)),
        (
            "ring (the flag read)",
            Arm::RingBlocking,
            fixed_cadence(n, 1),
        ),
        (
            "ring-hostdrain, fixed",
            Arm::RingHostDrain,
            fixed_cadence(n, 1),
        ),
        (
            "ring-gpubatch, fixed",
            Arm::RingGpuBatch,
            fixed_cadence(n, 1),
        ),
        (
            "ring-gpubatch-nonblocking, fixed",
            Arm::RingGpuBatchNonblocking,
            fixed_cadence(n, 1),
        ),
    ];
    if metal {
        let reader = |opts: DecodeOptions| DecodeOptions {
            stream_via_reader: true,
            ..opts
        };
        arms.extend([
            (
                "ring-la0 --stream-reader",
                Arm::RingPoll,
                reader(fixed_cadence(n, 0)),
            ),
            (
                "ring-fixed --stream-reader",
                Arm::RingPoll,
                reader(fixed_cadence(n, 1)),
            ),
            ("the flow", Arm::RingPoll, adaptive_opts(n, 8, true)),
            (
                "the flow, --eos-gate=false",
                Arm::RingPoll,
                adaptive_opts(n, 8, false),
            ),
            (
                "the flow, --no-fused-publish",
                Arm::RingPoll,
                DecodeOptions {
                    fused_publish: false,
                    ..adaptive_opts(n, 8, true)
                },
            ),
            (
                "the flow on ring-hostdrain",
                Arm::RingHostDrain,
                adaptive_opts(n, 8, true),
            ),
        ]);
    }
    arms
}

/// The flow and the fixed cadence deliver one stream (`docs/dispatch.md`
/// §11.5z-xv). The fixed cadence stays as a selectable arm beside the flow,
/// and neither is allowed to rot: every arm the loop keeps must hand its
/// consumer the host chain's ids — through `on_tokens` where the loop drains,
/// through the reader where it does not — on an EOS-ended run and on a
/// length-capped one that crosses the TTFT ramp and several commit windows.
///
/// Each arm must also run the mechanism its name promises, or two arms that
/// silently became one would pass: the fixed cadence makes cadence commits and
/// no adaptive close, and `ring-la0` one commit every step; `ring-fixed`'s
/// consumer receives the first eight tokens one commit each (the TTFT ramp)
/// and then eight a commit; the flow makes no cadence commit and closes its
/// buffers adaptively, and its gate fires exactly when the run met its EOS.
/// Off Metal there is no reader, so the CPU runs the fixed-cadence arms only.
#[test]
fn the_flow_and_the_fixed_cadence_deliver_one_stream() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        let metal = matches!(device, Device::Metal(_));
        for (start, n) in [(1u32, 200usize), (0, 40)] {
            let (want, hit) = chain(start, n);
            let mut flows = 0usize;
            for (name, arm, opts) in cadence_arms(n, metal) {
                let tag = format!("{device:?} from {start}: {name}");
                let (flow, gated, lookahead) =
                    (opts.adaptive_commit, opts.eos_gate, opts.lookahead);
                let d = deliver(&device, &arm, &[start], opts)?;
                assert_eq!(d.out.ids, want, "{tag}: the ids");
                assert_eq!(d.out.eos, hit, "{tag}: EOS");
                assert_eq!(d.stream, want, "{tag}: the stream its consumer saw");

                let s = &d.out.stats;
                let closes = s.adaptive_commits + s.run_ahead_closes;
                if flow {
                    flows += 1;
                    assert_eq!(s.commits, 0, "{tag}: a cadence commit on the flow");
                    assert!(closes > 0, "{tag}: the flow closed no buffer");
                } else {
                    assert_eq!(closes, 0, "{tag}: an adaptive close on the fixed cadence");
                    if !matches!(arm, Arm::Blocking) {
                        assert!(s.commits > 0, "{tag}: the fixed cadence made no commit");
                    }
                }
                if lookahead == 0 && !matches!(arm, Arm::Blocking) {
                    assert_eq!(s.commits, s.steps, "{tag}: a commit every step");
                }
                if name == "ring-fixed" && !hit {
                    let ramp_then_cadence = [vec![1usize; 8], vec![8; (n - 8) / 8]].concat();
                    assert_eq!(
                        d.batches, ramp_then_cadence,
                        "{tag}: the TTFT ramp, then one batch a period"
                    );
                }
                match (&s.eos_gate, gated) {
                    (Some(g), true) => assert_eq!(g.fired, hit, "{tag}: the gate fired"),
                    (None, false) => {}
                    (g, _) => panic!("{tag}: the gate report {g:?} with the gate {gated}"),
                }
            }
            assert_eq!(
                flows,
                if metal { 4 } else { 0 },
                "{device:?}: flow arms run"
            );
        }
    }
    Ok(())
}

/// Two turns of one conversation on one arm, continued through
/// `Session::prepare` as the engine continues a chat: turn one from `prompt`,
/// then the whole conversation — the prompt, turn one's ids, the EOS that
/// ended it, `user2` — prepared against what the cache holds and the suffix
/// decoded. The ring is kept across the turns, as the engine's pool keeps it.
#[cfg(feature = "metal")]
struct Conversation {
    turn1: Vec<u32>,
    action: lloom::nn::ReuseAction,
    turn2: Vec<u32>,
}

#[cfg(feature = "metal")]
fn converse<P: lloom::models::lfm2::Shape>(
    model: &lloom::models::lfm2::Model<P>,
    cache: lloom::models::lfm2::State<P>,
    rb: &mut AnyArm,
    opts: &dyn Fn(usize) -> DecodeOptions,
    prompt: &[u32],
    eos: u32,
    user2: &[u32],
) -> Result<Conversation> {
    fn turn<P: lloom::models::lfm2::Shape>(
        s: &mut Session<'_, lloom::models::lfm2::Model<P>>,
        rb: &mut AnyArm,
        prompt: &[u32],
        opts: DecodeOptions,
        resume: bool,
    ) -> Result<Vec<u32>> {
        if opts.stream_via_reader {
            let AnyArm::Ring(ring) = rb else {
                panic!("a reader-owned run needs a ring");
            };
            let (out, seen) = reader_run(s, ring, prompt, opts, resume)?;
            assert_eq!(seen, out.ids, "the reader's stream is the run's ids");
            return Ok(out.ids);
        }
        Ok(s.decode(
            &mut IdentityProposer,
            rb,
            &mut greedy(),
            prompt,
            &opts,
            &mut sink,
        )?
        .ids)
    }
    let mut s = Session::new(model, cache);
    let turn1 = turn(&mut s, rb, prompt, opts(64), false)?;
    let full = [prompt, &turn1[..], &[eos][..], user2].concat();
    let prepared = s.prepare(&full)?;
    let turn2 = turn(&mut s, rb, &prepared.suffix, opts(12), true)?;
    Ok(Conversation {
        turn1,
        action: prepared.action,
        turn2,
    })
}

/// The flow and the fixed cadence on the real decode kernels (§11.5z-xv): a
/// two-layer LFM2 on the crown's cache — in-place KV, `Shuffle` conv state
/// with the fused step, rewind horizon 0 — and a two-turn conversation
/// through `Session::prepare`. Every arm's turns are the blocking arm's, so
/// the gated kernel variants, the fused publish and the cadence change no
/// token. What differs is how turn two reuses the cache, and that is pinned
/// too: the arms that leave nothing written past the EOS — the blocking arm,
/// `ring-la0`, the gated flow — extend it; `ring-fixed` and the ungated flow
/// leave a written overrun this cache cannot rewind, and re-prefill the
/// conversation (`dispatch.md` §11.5z-xiv's finding, on the toy).
#[cfg(feature = "metal")]
#[test]
fn the_flow_and_the_fixed_cadence_agree_on_the_real_decode_kernels_across_turns() -> Result<()> {
    use lloom::models::lfm2::{ConvState, State};
    use lloom::nn::ReuseAction;
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new_metal(0).expect("metal");
    let cfg = tiny_lfm2::config(ConvState::Shuffle, true);
    let model = tiny_lfm2::build::<lloom::profile::CrownExact>(&cfg, &dev)?;
    let fresh = || State::<lloom::profile::CrownExact>::new(true, DType::F16, &cfg, &dev);
    let prompt = [1u32, 2, 3, 4, 5];

    // The EOS is the stream's first token not seen earlier, at least four in,
    // so turn one stops on it and every arm has an overrun to handle.
    let stream = Session::new(&model, fresh()?)
        .decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[u32::MAX], &dev)?,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(32),
            &mut sink,
        )?
        .ids;
    let k = (4..stream.len())
        .find(|&k| !stream[..k].contains(&stream[k]))
        .unwrap_or_else(|| panic!("no fresh token in the stream to stop on: {stream:?}"));
    let eos = stream[k];

    // The second message is the first candidate whose blocking turn two runs
    // at least four tokens before the EOS or the cap, so turn two is decode
    // steps reading the continued cache, not one prediction.
    let user2 = [[9u32, 3], [7, 11], [13, 6], [21, 2], [30, 17], [44, 5]]
        .into_iter()
        .map(|u| -> Result<(Vec<u32>, [u32; 2])> {
            let mut rb = AnyArm::Blocking(Blocking::new(&[eos], &dev)?);
            let c = converse(
                &model,
                fresh()?,
                &mut rb,
                &DecodeOptions::new,
                &prompt,
                eos,
                &u,
            )?;
            Ok((c.turn2, u))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .find(|(t, _)| t.len() >= 4)
        .map(|(_, u)| u)
        .expect("no candidate second message runs four tokens on this fixture");

    let slots = lloom::generation::readback::RING_HISTORY_SLOTS;
    let ring = |read: EosRead| -> Result<AnyArm> {
        Ok(AnyArm::Ring(Ring::new(&[eos], &dev, slots, read)?))
    };
    let poll = || EosRead::Poll(PollMechanism::default());
    type Opts = Box<dyn Fn(usize) -> DecodeOptions>;
    let arms: Vec<(&str, AnyArm, Opts, ReuseAction)> = vec![
        (
            "blocking",
            AnyArm::Blocking(Blocking::new(&[eos], &dev)?),
            Box::new(DecodeOptions::new),
            ReuseAction::Extend,
        ),
        (
            "ring-la0",
            ring(poll())?,
            Box::new(|n| fixed_cadence(n, 0)),
            ReuseAction::Extend,
        ),
        (
            "ring-fixed",
            ring(poll())?,
            Box::new(|n| fixed_cadence(n, 1)),
            ReuseAction::Fresh,
        ),
        (
            "ring-hostdrain, fixed",
            ring(EosRead::HostOnDrain)?,
            Box::new(|n| fixed_cadence(n, 1)),
            ReuseAction::Fresh,
        ),
        (
            "the flow",
            ring(poll())?,
            Box::new(|n| adaptive_opts(n, 8, true)),
            ReuseAction::Extend,
        ),
        (
            "the flow, --eos-gate=false",
            ring(poll())?,
            Box::new(|n| adaptive_opts(n, 8, false)),
            ReuseAction::Fresh,
        ),
    ];

    // The blocking arm runs first; its turn two is every other arm's reference.
    let mut reference: Option<Vec<u32>> = None;
    for (name, mut rb, opts, action) in arms {
        let c = converse(&model, fresh()?, &mut rb, &*opts, &prompt, eos, &user2)?;
        eprintln!(
            "{name}: turn 1 {:?}, turn 2 {:?} after {:?}",
            c.turn1, c.turn2, c.action
        );
        assert_eq!(c.turn1, stream[..k], "{name}: turn one ends at the EOS");
        assert_eq!(c.action, action, "{name}: how turn two reuses the cache");
        let r = reference.get_or_insert_with(|| c.turn2.clone());
        assert_eq!(&c.turn2, r, "{name}: turn two against the blocking arm's");
    }
    assert!(
        !reference.expect("the blocking arm ran").is_empty(),
        "turn two produced nothing"
    );
    Ok(())
}

/// A [`RingReader`] streams exactly the ids its run returns on every EOS read,
/// every cadence that can host one and every device-resident sampler, request
/// after request on one pooled ring (`Ring::new` at `RING_HISTORY_SLOTS`, as
/// `RingPool` builds it; each reader resumed at its run's first token).
///
/// What each arm stamps is what the reader can stream: the per-step reads
/// (flag, poll) stamp every token; the deferred reads stamp a window per
/// commit on the fixed cadence and the last window at the final drain, and
/// every token under the adaptive commit — through the fused kernel for the
/// argmax sampler, through the loop's `publish_marker` for any other. The
/// sampled arm (seeded gumbel-max) is the one every greedy gate is blind to:
/// the fused kernel serves argmax only. Because the final drain stamps what
/// is left, a stream that was never per-token still ends equal to the ids;
/// so under the adaptive commit an early EOS must also have stopped the run
/// through the reader's flag, which only a per-token stamp raises in time.
/// Where the adaptive commit carries the EOS gate, the gate must fire on the
/// EOS whichever path published it.
#[test]
fn a_ring_reader_streams_what_every_arm_returns() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the reader is Metal
        }
        let reads = [
            ("flag", EosRead::Blocking),
            ("poll", EosRead::Poll(PollMechanism::default())),
            ("drain", EosRead::HostOnDrain),
            ("gpubatch", EosRead::GpuBatched),
            ("gpubatch-nonblocking", EosRead::GpuBatchedNonblocking),
        ];
        let fixed = |lookahead: usize, budget: CommitBudget, n: usize| DecodeOptions {
            lookahead,
            budget,
            stream_via_reader: true,
            ..DecodeOptions::new(n)
        };
        let adaptive = |fused: bool, gate: bool, n: usize| DecodeOptions {
            fused_publish: fused,
            ..adaptive_opts(n, 8, gate)
        };
        type Cadence = (&'static str, bool, Box<dyn Fn(usize) -> DecodeOptions>);
        let cadences: Vec<Cadence> = vec![
            (
                "la0 steps:8",
                false,
                Box::new(move |n| fixed(0, CommitBudget::steps(8).expect("8"), n)),
            ),
            (
                "la1 steps:8",
                false,
                Box::new(move |n| fixed(1, CommitBudget::steps(8).expect("8"), n)),
            ),
            (
                "la1 per-token",
                false,
                Box::new(move |n| fixed(1, CommitBudget::per_token(), n)),
            ),
            (
                "adaptive",
                true,
                Box::new(move |n| adaptive(true, false, n)),
            ),
            (
                "adaptive unfused",
                true,
                Box::new(move |n| adaptive(false, false, n)),
            ),
            (
                "adaptive gated",
                true,
                Box::new(move |n| adaptive(true, true, n)),
            ),
        ];
        // Gumbel-max at t = 0.1 is a sampler the fused kernel does not serve
        // whose draws, against the fixture's one-hot logits, still follow the
        // chain to its early EOS: the sampled arm the per-token check reads.
        // At t = 0.6 the draws are near uniform and an EOS is rare and late.
        let samplers = [
            ("argmax", Sampling::ArgMax),
            ("t=0.1", Sampling::All { temperature: 0.1 }),
            ("t=0.6", Sampling::All { temperature: 0.6 }),
            ("gumbel t=0.6", Sampling::GumbelSoftmax { temperature: 0.6 }),
        ];
        // (start, max): ends on EOS, on length, EOS again, then length.
        let runs: [(u32, usize); 4] = [(1, 64), (0, 24), (1, 64), (0, 24)];
        let model = Chain::new(&device)?;
        // The blocking arm's run of each (sampler, run) at the seed: what
        // every ring arm must return. One method is one draw on every arm
        // (`generation.sampling.method`), keyed by position, so the arm that
        // reads the token back cannot change the reply.
        let mut reference = std::collections::HashMap::new();
        for (sname, sampling) in &samplers {
            for (i, (start, max)) in runs.into_iter().enumerate() {
                let mut session = Session::new(&model, 0usize);
                let mut rb = Blocking::new(&[EOS], &device)?;
                let mut sampler = LogitsProcessor::from_sampling(805, sampling.clone());
                let out = session.decode(
                    &mut IdentityProposer,
                    &mut rb,
                    &mut sampler,
                    &[start],
                    &DecodeOptions::new(max),
                    &mut |_: &[u32]| Ok(()),
                )?;
                reference.insert((*sname, i), (out.ids, out.eos));
            }
        }
        let mut arms_equal = 0usize;
        let mut failures: Vec<String> = Vec::new();
        let mut checked = 0usize;
        let mut sampled_early_eos = 0usize;
        for (read, eos) in reads {
            for (cadence, is_adaptive, opts_for) in &cadences {
                if *is_adaptive && matches!(eos, EosRead::Blocking) {
                    continue; // the flag reads every token back: refused
                }
                for (sname, sampling) in &samplers {
                    let mut rb = Ring::new(
                        &[EOS],
                        &device,
                        lloom::generation::readback::RING_HISTORY_SLOTS,
                        eos,
                    )?;
                    let mut sampler = LogitsProcessor::from_sampling(805, sampling.clone());
                    for (i, (start, max)) in runs.into_iter().enumerate() {
                        let at = format!("{read} {cadence} {sname} run {i}");
                        let opts = opts_for(max);
                        let gated = opts.eos_gate;
                        let mut session = Session::new(&model, 0usize);
                        let (out, seen) = match reader_run_sampled(
                            &mut session,
                            &mut rb,
                            &[start],
                            opts,
                            true,
                            &mut sampler,
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                failures.push(format!("{at}: refused: {e}"));
                                break;
                            }
                        };
                        checked += 1;
                        if matches!(sampling, Sampling::ArgMax) {
                            let (want, hit) = chain(start, max);
                            if out.ids != want || out.eos != hit {
                                failures.push(format!(
                                    "{at}: the loop's ids {:?} (eos {}) against the chain {want:?} (eos {hit})",
                                    out.ids, out.eos
                                ));
                            }
                        }
                        if seen != out.ids {
                            failures.push(format!(
                                "{at}: the reader streamed {seen:?}, the run returned {:?}",
                                out.ids
                            ));
                        }
                        let want = &reference[&(*sname, i)];
                        if (&out.ids, out.eos) != (&want.0, want.1) {
                            failures.push(format!(
                                "{at}: the ids {:?} (eos {}) against the blocking arm's {:?} \
                                 (eos {})",
                                out.ids, out.eos, want.0, want.1
                            ));
                        } else {
                            arms_equal += 1;
                        }
                        // Per token, not only by the end: under the adaptive
                        // commit the reader's flag is the loop's EOS, and it
                        // is raised in time only off a per-token stamp. The
                        // loop runs at most two run-ahead windows (16 steps)
                        // past an EOS the reader has seen, so an EOS 32 short
                        // of the cap is margin; a run that reached the cap
                        // instead streamed its tokens from the final drain.
                        let early_eos = *is_adaptive && out.eos && out.ids.len() + 32 <= max;
                        if early_eos && !matches!(sampling, Sampling::ArgMax) {
                            sampled_early_eos += 1;
                        }
                        if early_eos && !out.stats.stopped_by_flag {
                            failures.push(format!(
                                "{at}: the EOS at {} did not stop the run through the reader \
                                 (it ran {} steps of {max})",
                                out.ids.len(),
                                out.stats.steps
                            ));
                        }
                        if gated {
                            let gate = out.stats.eos_gate.expect("the gate ran");
                            if out.eos && !gate.fired {
                                failures.push(format!("{at}: an EOS the gate did not see"));
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "{device:?}: {checked} runs checked ({sampled_early_eos} sampled, adaptive, early EOS), \
             {} failed",
            failures.len()
        );
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        assert!(checked >= 150, "{device:?}: only {checked} runs ran");
        assert_eq!(
            arms_equal
                + failures
                    .iter()
                    .filter(|f| f.contains("blocking arm"))
                    .count(),
            checked,
            "every run was compared with the blocking arm's"
        );
        assert!(
            sampled_early_eos > 0,
            "{device:?}: no sampled run ended on an early EOS under the adaptive commit, so the \
             per-token check never read the unfused publish"
        );
        assert!(
            failures.is_empty(),
            "{} runs failed (listed above)",
            failures.len()
        );
    }
    Ok(())
}

/// A rate, not a gate: a reader that saw a stamp and read the slot behind it
/// before the slot's write was visible — the previous content (the zero fill
/// on a slot never written, #735's shape; a stale token on a reused one).
/// Pairs of an EOS request then a length request on one pooled ring, as
/// `lloom serve` runs them. `LLOOM_READER_STRESS=<pairs>` per arm (default
/// 200); `LLOOM_READER_STRESS_ARMS` filters arms by substring. Prints, per
/// arm, the runs whose stream was not the ids and how: `stale` (one slot
/// differs), `past-eos` (the EOS slot read stale, the stream ran on), other.
/// Asserts nothing: a count of a race is machine state (§3.2c).
#[test]
#[ignore = "a stress rate; run with --ignored"]
fn reader_stale_slot_rate_after_an_eos_on_a_pooled_ring() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    let pairs: usize = std::env::var("LLOOM_READER_STRESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let filter = std::env::var("LLOOM_READER_STRESS_ARMS").unwrap_or_default();
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue;
        }
        let model = Chain::new(&device)?;
        type Opts = Box<dyn Fn(usize) -> DecodeOptions>;
        let cadences: Vec<(&str, Opts)> = vec![
            ("adaptive fused", Box::new(|n| adaptive_opts(n, 8, false))),
            (
                "adaptive fused ra1",
                Box::new(|n| adaptive_opts(n, 1, false)),
            ),
            (
                "adaptive unfused",
                Box::new(|n| DecodeOptions {
                    fused_publish: false,
                    ..adaptive_opts(n, 8, false)
                }),
            ),
            (
                "fixed la1 steps:8",
                Box::new(|n| DecodeOptions {
                    lookahead: 1,
                    budget: CommitBudget::steps(8).expect("8"),
                    stream_via_reader: true,
                    ..DecodeOptions::new(n)
                }),
            ),
        ];
        for (read, eos) in [
            ("poll", EosRead::Poll(PollMechanism::default())),
            ("drain", EosRead::HostOnDrain),
            ("gpubatch-nonblocking", EosRead::GpuBatchedNonblocking),
        ] {
            for (cadence, opts_for) in &cadences {
                let arm = format!("{read} {cadence}");
                if !filter.is_empty() && !arm.contains(&filter) {
                    continue;
                }
                let mut rb = Ring::new(
                    &[EOS],
                    &device,
                    lloom::generation::readback::RING_HISTORY_SLOTS,
                    eos,
                )?;
                let (mut runs, mut tokens, mut stale, mut past, mut other) = (0, 0, 0, 0, 0);
                let mut first: Option<String> = None;
                for _ in 0..pairs {
                    for (start, max) in [(1u32, 64usize), (0, 24)] {
                        let mut session = Session::new(&model, 0usize);
                        let (out, seen) =
                            reader_run(&mut session, &mut rb, &[start], opts_for(max), true)?;
                        runs += 1;
                        tokens += out.ids.len();
                        if seen == out.ids {
                            continue;
                        }
                        let diffs: Vec<usize> = (0..seen.len().min(out.ids.len()))
                            .filter(|&i| seen[i] != out.ids[i])
                            .collect();
                        if seen.len() == out.ids.len() && diffs.len() == 1 {
                            stale += 1;
                        } else if seen.len() > out.ids.len() && diffs.is_empty() {
                            past += 1;
                        } else {
                            other += 1;
                        }
                        first.get_or_insert_with(|| {
                            format!("start {start}: streamed {seen:?}, ids {:?}", out.ids)
                        });
                    }
                }
                eprintln!(
                    "STRESS {arm}: {} wrong of {runs} runs ({tokens} tokens): {stale} stale, \
                     {past} past-eos, {other} other{}",
                    stale + past + other,
                    first.map(|f| format!("; first: {f}")).unwrap_or_default()
                );
            }
        }
    }
    Ok(())
}

/// The stamp after the fused publish never runs ahead of the slot it
/// certifies, as seen by a host reader: under the adaptive commit at a
/// one-step run-ahead — where the kernel's own in-dispatch stamp let a reader
/// read the slot's previous content in about 0.8 % of runs (§11.5z-xvi) — a
/// resumed reader streams exactly the ids, and the ids are the chain, for
/// every run of 1 000 EOS-then-length pairs on each deferred and per-step
/// read. A race, so a red here is a defect and a green is a bound.
#[test]
fn the_fused_publishs_stamp_never_runs_ahead_of_its_slot() -> Result<()> {
    let _quiet = GPU_QUIET.lock().unwrap_or_else(|e| e.into_inner());
    for device in devices() {
        if matches!(device, Device::Cpu) {
            continue; // the fused publish and the reader are Metal
        }
        let model = Chain::new(&device)?;
        let mut failures: Vec<String> = Vec::new();
        let mut runs = 0usize;
        for (read, eos) in [
            ("poll", EosRead::Poll(PollMechanism::default())),
            ("drain", EosRead::HostOnDrain),
            ("gpubatch-nonblocking", EosRead::GpuBatchedNonblocking),
        ] {
            let mut rb = Ring::new(
                &[EOS],
                &device,
                lloom::generation::readback::RING_HISTORY_SLOTS,
                eos,
            )?;
            for pair in 0..1000 {
                for (start, max) in [(1u32, 64usize), (0, 24)] {
                    let (want, _) = chain(start, max);
                    let mut session = Session::new(&model, 0usize);
                    let (out, seen) = reader_run(
                        &mut session,
                        &mut rb,
                        &[start],
                        adaptive_opts(max, 1, false),
                        true,
                    )?;
                    runs += 1;
                    assert!(
                        out.stats.readback.fused_publishes > 0,
                        "{read}: the fused publish did not run"
                    );
                    if out.ids != want || seen != want {
                        failures.push(format!(
                            "{read} pair {pair} from {start}: ids {:?}, streamed {seen:?}, chain {want:?}",
                            out.ids
                        ));
                    }
                }
            }
        }
        eprintln!("{device:?}: {runs} runs, {} wrong", failures.len());
        for f in failures.iter().take(8) {
            eprintln!("FAIL {f}");
        }
        assert!(
            failures.is_empty(),
            "{} of {runs} runs wrong",
            failures.len()
        );
    }
    Ok(())
}

/// A draw the loop takes back on the ring (`Ring::retract_draw`, a
/// provisional step redone behind the next forward): its history slot is
/// written again by the next draw, which carries the NEXT stamp, so a reader
/// that read the taken-back token reads the new one as new once it shifts
/// its mapping (`RingReader::retract_one`); the drain returns the history
/// without the taken-back token; `eos_of_last` answers for the draw made
/// again; the publishes reconcile with the retraction counted.
#[cfg(feature = "metal")]
#[test]
fn a_draw_taken_back_on_the_ring_is_read_as_the_one_drawn_again() -> Result<()> {
    let Ok(device) = Device::new_metal(0) else {
        return Ok(());
    };
    let mut rb = Ring::new(&[EOS], &device, 64, EosRead::Poll(PollMechanism::default()))?
        .with_poll_budget(20_000);
    let mut reader = rb.reader().expect("mapped ring");
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let logits = |hot: usize| -> Result<Tensor> {
        let mut v = vec![0f32; 16];
        v[hot] = 5.0;
        Tensor::from_vec(v, 16, &device)
    };
    assert!(rb.retracts_draw());
    rb.sample_and_feed(&mut sampler, &logits(3)?)?;
    assert!(!rb.eos_of_last()?);
    rb.sample_and_feed(&mut sampler, &logits(7)?)?;
    device.synchronize()?;
    let mut seen = Vec::new();
    reader.read_all_new(&mut seen).expect("no lap");
    assert_eq!(seen, vec![3, 7]);
    // 7 taken back before its EOS read; 9 drawn in its place.
    rb.retract_draw()?;
    reader.retract_one();
    rb.sample_and_feed(&mut sampler, &logits(9)?)?;
    assert!(!rb.eos_of_last()?);
    device.synchronize()?;
    seen.clear();
    reader.read_all_new(&mut seen).expect("no lap");
    assert_eq!(
        seen,
        vec![9],
        "the reader read the slot under the new stamp"
    );
    rb.sample_and_feed(&mut sampler, &logits(11)?)?;
    assert!(!rb.eos_of_last()?);
    device.synchronize()?;
    seen.clear();
    reader.read_all_new(&mut seen).expect("no lap");
    assert_eq!(seen, vec![11]);
    assert_eq!(
        rb.drain_all()?,
        vec![3, 9, 11],
        "the history kept the taken-back token"
    );
    let c = rb.counters();
    assert_eq!(c.ring_retracted, 1);
    assert_eq!(c.ring_publishes, 4);
    // A later reader of this ring maps the same stamps to the same slots.
    Ok(())
}

#[test]
fn filtered_gumbel_ring_refuses_before_session_sampler_or_feed_moves() -> Result<()> {
    let device = Device::Cpu;
    let model = Chain::new(&device)?;
    let mut session = Session::new(&model, 0usize);
    let mut rb = Ring::new(&[EOS], &device, 8, EosRead::Blocking)?;
    let mut sampler = LogitsProcessor::from_sampling(
        7,
        Sampling::FilteredGumbel {
            temperature: 0.7,
            k: 1,
            p: 0.6,
        },
    )
    .with_processors(lloom::generation::Processors {
        presence_penalty: 0.5,
        ..Default::default()
    })
    .with_logits_digest(true);
    sampler.observe_prompt(&[5, 1]);
    let digest = sampler.logits_digest();
    let counters = rb.counters();
    let error = session
        .decode_until(
            &mut IdentityProposer,
            &mut rb,
            &mut sampler,
            &[5, 1],
            &DecodeOptions::new(3),
            &mut |_| panic!("refused request reached token callback"),
        )
        .unwrap_err();
    assert!(error.to_string().contains("explicit host draw"), "{error}");
    assert_eq!(session.pos(), 0);
    assert_eq!(*session.sequence().cache(), 0);
    assert!(session.tokens().is_empty());
    assert_eq!(sampler.digest_rows(), 0);
    assert_eq!(sampler.logits_digest(), digest);
    assert_eq!(rb.counters(), counters);
    // A successful supported draw after refusal is the control that also
    // observes processor history/RNG; a refusal must not have advanced either.
    let mut fresh = LogitsProcessor::from_sampling(7, sampler.sampling().clone())
        .with_processors(lloom::generation::Processors {
            presence_penalty: 0.5,
            ..Default::default()
        })
        .with_logits_digest(true);
    fresh.observe_prompt(&[5, 1]);
    let logits = Tensor::new(&[0f32; VOCAB], &device)?;
    assert_eq!(sampler.sample(&logits)?, fresh.sample(&logits)?);
    assert_eq!(sampler.logits_digest(), fresh.logits_digest());
    Ok(())
}

#[test]
fn token_only_any_arm_keeps_the_token_input_contract() {
    use lloom_tensor::backend::PassInput;
    assert_eq!(AnyArm::PASS_INPUT, PassInput::TokenIds);
    assert_eq!(Blocking::PASS_INPUT, PassInput::TokenIds);
    assert_eq!(Ring::PASS_INPUT, PassInput::TokenIds);
}
