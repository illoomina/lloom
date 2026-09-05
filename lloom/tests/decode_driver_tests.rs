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
};
use lloom::nn::speculator::{Draft, NoProbe, SeqId, TargetProbe};
use lloom::nn::{CausalLM, DecodeOptions, IdentityProposer, Model, Session, Speculator};
use lloom_tensor::{DType, Device, Result, Tensor};

const VOCAB: usize = 64;
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

enum Arm {
    Blocking,
    RingBlocking,
    RingPoll,
    /// The device never computes the EOS flag; the host learns it from the
    /// drained ids on the commit cadence, so the run overruns an EOS by up to a
    /// commit period (#677).
    RingHostDrain,
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
    fn apply_backpressure(&mut self, cap: usize) -> Result<()> {
        match self {
            AnyArm::Blocking(b) => b.apply_backpressure(cap),
            AnyArm::Ring(r) => r.apply_backpressure(cap),
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
            // that showed it: at least 1, at most a commit period's worth.
            assert!(
                out.stats.overrun >= 1 && out.stats.overrun <= period,
                "{device:?} steps:{period}: overrun {} outside [1, {period}]",
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
        // Lookahead on: the loop commits on the budget and nowhere else.
        let opts = DecodeOptions {
            lookahead: 1,
            budget: CommitBudget::steps(4)?,
            ..DecodeOptions::new(12)
        };
        let (out, _) = run(&device, &Arm::RingPoll, &[0], &opts)?;
        assert_eq!(out.stats.commits, 3, "{device:?}: 12 steps at steps:4");
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
fn max_tokens_bounds_the_run_and_a_ring_too_small_is_refused() -> Result<()> {
    let device = Device::Cpu;
    let (out, _) = run(&device, &Arm::RingPoll, &[0], &DecodeOptions::new(5))?;
    assert_eq!(out.ids.len(), 5);
    let model = Chain::new(&device)?;
    let mut session = Session::new(&model, 0usize);
    // `Ring::new` floors the history at `RING_HISTORY_SLOTS` (a few pages, so
    // a short run gets a generous ring), so a refusal needs a run longer than
    // the floor, not longer than the 3 asked for.
    let floor = lloom::generation::readback::RING_HISTORY_SLOTS;
    let mut rb = Ring::new(&[EOS], &device, 3, EosRead::Blocking)?;
    let err = session
        .decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &[0],
            &DecodeOptions::new(floor + 2),
            &mut sink,
        )
        .unwrap_err();
    assert!(err.to_string().contains("capacity"), "{err}");
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
    assert_eq!(a.id(), b.id(), "two feeds returned two allocations");
    assert_eq!(rb.take_ids()?, vec![3, 3]);
    assert_eq!(rb.take_ids()?, Vec::<u32>::new());
    Ok(())
}
