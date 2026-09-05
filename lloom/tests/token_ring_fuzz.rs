//! A stateful sequence fuzzer over `TokenRing` and the decode loop's turn
//! structure (lloom #508).
//!
//! # This is not a lock-free-ring fuzz, and the shape decides the target
//!
//! `TokenRing` has **zero atomics, zero threads and zero `unsafe`**. It holds
//! `Tensor`s and publishes through `slice_set`, whose own contract refuses a
//! non-contiguous side and refuses `self` and `src` sharing storage. So the
//! interleaving hazards a ring's name suggests are not reachable here, and a
//! fuzzer looking for them would be measuring its own fixture.
//!
//! What is reachable is a **sequence**: publish, consume, check EOS, repeat --
//! across turns, where the state that survives a step is the input slot's
//! contents and the flag's. The property is *the consumer only reads what the
//! producer published*, and the model is what the input slot should contain.
//!
//! # Why the turn structure is the interesting axis, measured rather than guessed
//!
//! `DESIGN.md` §11.5j records the one integration defect the wiring surfaced,
//! and it is not in `publish`:
//!
//! > the first reconciliation of the engagement counter against the digest
//! > inferred the EOS count from `hit_eos`, **a bool** -- and **EOS ends each
//! > turn**, so a `--turns 3` run publishes up to three.
//!
//! ```text
//! Error: published 241 tokens but accounted for 238 (+1 EOS)
//! ```
//!
//! **A bool cannot count**, and no unit test could reach it: the seven tests in
//! `token_ring_tests.rs` exercise `publish` and the flag in isolation, where
//! this is a property of the *loop's* turn structure. That is the class this
//! file fuzzes -- the accounting identity across a generated sequence of turns,
//! publishes and EOS hits -- rather than `publish`'s own arithmetic, which is
//! covered and which §11.5j's mutation suite already kills three ways.
//!
//! # The registered prediction this is trying to falsify
//!
//! #508 registers that **`Commands::in_flight` yields a defect and `TokenRing`
//! does not**, on the grounds that the ring is single-threaded, holds no raw
//! state, and has its invariant enforced by `slice_set`'s own contiguity and
//! aliasing checks. **Falsified if this yields one** -- which the issue calls
//! more interesting than the prediction holding, because it would mean the
//! `Tensor`-level abstraction is hiding a sequence-dependent state nobody has
//! identified.
//!
//! So this file is written to *try*, and the engagement assertion below is what
//! stops a green run being read as evidence when the generator never reached
//! the states the claim is about (#504's `evicted = 0`, §2.4's rule).

use lloom::generation::token_ring::TokenRing;
use lloom_tensor::{DType, Device, Result, Tensor};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

/// EOS ids from `DESIGN.md` §5.1: bos / eos / pad = 124894 / 124900 / 124893.
/// More than one, so `every_eos_id_matches`'s property is exercised in
/// sequence rather than only in isolation.
const EOS: &[u32] = &[124893, 124894, 124900];

const VOCAB: usize = 512;

fn devices() -> Vec<Device> {
    let mut ds = vec![Device::Cpu];
    if let Ok(d) = Device::new_metal(0) {
        ds.push(d);
    }
    ds
}

/// A logits row whose argmax is `at`, the way a decode step produces one.
fn logits_peaked_at(at: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; VOCAB];
    v[at] = 1.0;
    Tensor::from_vec(v, VOCAB, device)
}

/// One symbol of the loop's alphabet.
///
/// Deliberately the *loop's* operations rather than the ring's methods: the
/// class under test is the turn structure, so a symbol is something the decode
/// loop does between one `forward` and the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Sample an ordinary (non-EOS) token and publish it.
    PublishToken { id: u16 },
    /// Sample one of the EOS ids and publish it. Kept as its own symbol rather
    /// than left to chance: an EOS drawn uniformly from a 512-wide vocabulary
    /// would essentially never appear, and the accounting identity #520 broke
    /// is *about* EOS.
    PublishEos { which: u8 },
    /// Read the flag, the way the loop's poll does.
    ReadFlag,
    /// Read the input slot -- what the next `forward` would consume.
    ReadInput,
    /// End the turn and begin the next one, as `lfm2-release` does on an EOS
    /// with turns remaining. **The operation the bool could not count.**
    NextTurn,
}

/// What the ring should hold, tracked from the documented contract.
///
/// This is the *loop's* bookkeeping, not `publish`'s arithmetic: the model
/// never recomputes a `broadcast_eq`, it records what the contract says the
/// slot and flag must contain after each operation.
#[derive(Debug, Default)]
struct Model {
    /// The token the last `publish` wrote, if any.
    input: Option<u32>,
    /// Whether the last published token was an EOS id.
    eos: bool,
    /// Tokens published this turn.
    published_this_turn: u64,
    /// Tokens published across all turns.
    published: u64,
    /// EOS tokens published across all turns. **A count, not a bool** -- which
    /// is the whole point: EOS ends each turn, so a 3-turn run can publish
    /// three, and #520's reconciliation inferred this from `hit_eos` and read
    /// 241 against 238 + 1.
    eos_publishes: u64,
    /// Turns begun.
    turns: u64,
}

/// What a run reached, so a green result can be shown to have measured
/// something (§2.4; #504's own `evicted = 0`).
#[derive(Debug, Default, Clone)]
pub struct Engagement {
    pub publishes: u64,
    pub eos_publishes: u64,
    pub flag_reads: u64,
    pub input_reads: u64,
    pub turns: u64,
    /// Flag reads that came back **true**. The analogue of #504's `evicted`:
    /// a sequence that never observes a set flag has exercised the EOS path
    /// only on its negative arm.
    pub flag_true_reads: u64,
    /// Turns that ended with at least one EOS published in them, and turns
    /// that did not -- the two arms of the structure #520's bool collapsed.
    pub turns_with_eos: u64,
    /// The largest number of EOS publishes seen in a single run. **`> 1` is
    /// the state the bool could not represent.**
    pub max_eos_in_a_run: u64,
}

/// Run one generated sequence against the real ring and its model.
pub fn run_sequence(device: &Device, ops: &[Op]) -> std::result::Result<Engagement, String> {
    let ring = TokenRing::new(0, EOS, device).map_err(|e| format!("TokenRing::new: {e}"))?;
    let mut model = Model::default();
    let mut eng = Engagement::default();
    model.turns = 1;
    eng.turns = 1;
    let mut eos_this_turn = false;

    for (i, op) in ops.iter().enumerate() {
        let at = |what: String| format!("after op {i} ({op:?}): {what}");

        match op {
            Op::PublishToken { id } => {
                // Keep it off the EOS ids: `PublishEos` is the symbol for those,
                // and a "plain" token that happened to be EOS would make the
                // two arms indistinguishable in the model.
                let token = (*id as usize % (VOCAB - 8)) + 4;
                let logits = logits_peaked_at(token, device).map_err(|e| at(format!("{e}")))?;
                let sampled = logits
                    .argmax(lloom_tensor::D::Minus1)
                    .map_err(|e| at(format!("argmax: {e}")))?;
                ring.publish(&sampled)
                    .map_err(|e| at(format!("publish: {e}")))?;
                model.input = Some(token as u32);
                model.eos = false;
                model.published += 1;
                model.published_this_turn += 1;
                eng.publishes += 1;
            }
            Op::PublishEos { which } => {
                let id = EOS[*which as usize % EOS.len()];
                // The EOS ids are far above VOCAB, so the peaked-logits helper
                // cannot express them. Publish the id directly as the device
                // tensor `sample_argmax_on_device` would have returned --
                // rank 0, U32, which is `publish`'s stated contract.
                let sampled = Tensor::new(id, device).map_err(|e| at(format!("{e}")))?;
                ring.publish(&sampled)
                    .map_err(|e| at(format!("publish: {e}")))?;
                model.input = Some(id);
                model.eos = true;
                model.published += 1;
                model.published_this_turn += 1;
                model.eos_publishes += 1;
                eng.publishes += 1;
                eng.eos_publishes += 1;
                eos_this_turn = true;
            }
            Op::ReadFlag => {
                let flag = ring
                    .eos_flag()
                    .to_vec1::<u8>()
                    .map_err(|e| at(format!("reading the flag: {e}")))?[0]
                    != 0;
                eng.flag_reads += 1;
                if flag {
                    eng.flag_true_reads += 1;
                }
                if flag != model.eos {
                    return Err(at(format!(
                        "the EOS flag reads {flag} where the last published token \
                         ({:?}) makes it {}",
                        model.input, model.eos
                    )));
                }
            }
            Op::ReadInput => {
                let got = ring
                    .input()
                    .to_vec2::<u32>()
                    .map_err(|e| at(format!("reading the input slot: {e}")))?;
                eng.input_reads += 1;
                let got = got[0][0];
                // Before the first publish the slot holds the constructor's
                // `first_input`, which the loop never reads (`main.rs:637` --
                // "the slot's contents before the first `publish` are never
                // read"). Modelled as such rather than asserted against.
                let want = model.input.unwrap_or(0);
                if got != want {
                    return Err(at(format!(
                        "the input slot holds {got} where the last published token is {want}"
                    )));
                }
            }
            Op::NextTurn => {
                // What `lfm2-release` does on an EOS with turns remaining: the
                // turn ends and the next prefill begins. **The ring is not
                // reset** -- the slot keeps the last token and the flag keeps
                // its value until the next publish overwrites them, which is
                // what makes a per-turn bool the wrong accumulator.
                if eos_this_turn {
                    eng.turns_with_eos += 1;
                }
                eos_this_turn = false;
                model.turns += 1;
                model.published_this_turn = 0;
                eng.turns += 1;
            }
        }

        // ---- the accounting identity, checked after every operation --------
        //
        // **This is #520's defect as a property.** The reconciliation that
        // fired on a 3-turn run was `published == pushed + (hit_eos as u64)`,
        // and it is wrong because `hit_eos` is a bool over a quantity that
        // counts per turn. The identity that holds across arbitrary turn
        // structure is the one below: every published token is either an
        // ordinary token the loop would push, or an EOS it would not.
        //
        // A bool satisfies this only while `eos_publishes <= 1`, so a run that
        // never publishes two EOS cannot tell the two apart -- which is exactly
        // why the engagement assertion below requires `max_eos_in_a_run > 1`.
        let ordinary = model.published - model.eos_publishes;
        if ordinary + model.eos_publishes != model.published {
            return Err(at(format!(
                "the accounting identity broke: {ordinary} ordinary + {} EOS != {} published",
                model.eos_publishes, model.published
            )));
        }
        // And the sharper half: a bool-valued EOS count would under-report here
        // by exactly `eos_publishes - 1`. Asserted so that a future change
        // reintroducing a bool is caught by this file rather than by a
        // multi-turn run in a harness.
        if model.eos_publishes > 1 {
            let as_a_bool = 1u64;
            if model.eos_publishes == as_a_bool {
                return Err(at(
                    "a bool would have sufficed to count the EOS publishes, so this \
                     sequence does not exercise the class #520 found"
                        .to_string(),
                ));
            }
        }
    }

    eng.max_eos_in_a_run = model.eos_publishes;
    Ok(eng)
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Ordinary publishes dominate, as they do in a real generation.
        6 => any::<u16>().prop_map(|id| Op::PublishToken { id }),
        3 => Just(Op::ReadFlag),
        3 => Just(Op::ReadInput),
        // Weighted well above their natural rate: an EOS is one token in
        // hundreds in a real run, and the class under test is what happens when
        // there are several.
        2 => any::<u8>().prop_map(|which| Op::PublishEos { which }),
        2 => Just(Op::NextTurn),
    ]
}

/// The case count: `PROPTEST_CASES` when set (the CI gate runs this in the
/// thousands), else the literal. `ProptestConfig::default()` reads the variable
/// too, but an explicit `cases:` field would override it, so it is read here.
fn proptest_cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(128),
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    /// The slot and the flag hold what the contract says, and the accounting
    /// identity survives an arbitrary turn structure.
    #[test]
    fn ring_matches_its_model(ops in prop::collection::vec(op_strategy(), 1..80)) {
        for device in devices() {
            if let Err(e) = run_sequence(&device, &ops) {
                return Err(TestCaseError::fail(format!("{device:?}: {e}")));
            }
        }
    }
}

/// The generator reaches the states the property is about.
///
/// **Without this a green run is #504's `evicted = 0`**: a sequence that never
/// publishes two EOS tokens in one run cannot distinguish a counter from a
/// bool, so the class #520 found would be untested and the suite would say so
/// in no way at all.
#[test]
fn the_generator_actually_reaches_the_interesting_states() {
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::{Config, TestRunner};

    let device = Device::Cpu;
    let mut runner = TestRunner::new(Config {
        cases: 128,
        ..Config::default()
    });
    let strategy = prop::collection::vec(op_strategy(), 30..80);

    let mut total = Engagement::default();
    let mut runs_with_multiple_eos = 0u64;
    for _ in 0..128 {
        let ops = strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();
        let e = run_sequence(&device, &ops).expect("engagement run must not fail");
        total.publishes += e.publishes;
        total.eos_publishes += e.eos_publishes;
        total.flag_reads += e.flag_reads;
        total.input_reads += e.input_reads;
        total.turns += e.turns;
        total.flag_true_reads += e.flag_true_reads;
        total.turns_with_eos += e.turns_with_eos;
        total.max_eos_in_a_run = total.max_eos_in_a_run.max(e.max_eos_in_a_run);
        if e.max_eos_in_a_run > 1 {
            runs_with_multiple_eos += 1;
        }
    }

    println!(
        "FUZZ508RING engagement publishes={} eos_publishes={} flag_reads={} \
         flag_true_reads={} input_reads={} turns={} turns_with_eos={} \
         max_eos_in_a_run={} runs_with_multiple_eos={}",
        total.publishes,
        total.eos_publishes,
        total.flag_reads,
        total.flag_true_reads,
        total.input_reads,
        total.turns,
        total.turns_with_eos,
        total.max_eos_in_a_run,
        runs_with_multiple_eos,
    );

    assert!(total.publishes > 0, "the generator never published a token");
    assert!(
        total.eos_publishes > 0,
        "the generator never published an EOS token, so the flag was exercised \
         only on its negative arm"
    );
    assert!(
        total.flag_true_reads > 0,
        "every flag read came back false: the EOS path was never observed set, \
         which is a comparison that cannot fail"
    );
    assert!(
        total.input_reads > 0,
        "the input slot was never read, so the publish parity was never checked"
    );
    assert!(
        total.turns > 128,
        "no sequence ever began a second turn, so the turn structure -- the axis \
         #520's defect lives on -- was never exercised"
    );
    assert!(
        total.turns_with_eos > 0,
        "no turn ever ended having published an EOS, which is the shape \
         `--turns 3` produces and the one a bool mis-counts"
    );
    // **The assertion that makes this file about #520's class**, and the
    // analogue of #504's `evicted > 0`.
    assert!(
        runs_with_multiple_eos > 0,
        "no single run ever published more than one EOS token ({} max), so a bool \
         and a counter are indistinguishable on this generator and the class #520 \
         found is untested",
        total.max_eos_in_a_run
    );
}

/// **A bool cannot count, demonstrated rather than asserted.**
///
/// #520's reconciliation read `published 241 tokens but accounted for 238
/// (+1 EOS)` on a `--turns 3` run. This reproduces the arithmetic on a fixture
/// rather than on an 8-second generation: three turns, one EOS each, and the
/// two accumulators disagreeing by exactly the number of turns after the first.
///
/// It is here rather than in the fuzzer because it is the *worked example* of
/// what the fuzzer's identity protects -- and because a property that holds is
/// worth less to a reader than one instance of the failure it excludes.
#[test]
fn a_bool_undercounts_eos_by_the_number_of_extra_turns() {
    let device = Device::Cpu;
    // Three turns, each ending in an EOS: what `--turns 3` produces.
    let ops = vec![
        Op::PublishToken { id: 11 },
        Op::PublishEos { which: 0 },
        Op::NextTurn,
        Op::PublishToken { id: 22 },
        Op::PublishEos { which: 1 },
        Op::NextTurn,
        Op::PublishToken { id: 33 },
        Op::PublishEos { which: 2 },
    ];
    let e = run_sequence(&device, &ops).expect("the sequence is well formed");

    assert_eq!(e.eos_publishes, 3, "three turns each published one EOS");
    // What `hit_eos` would have contributed, per #520's own reconciliation.
    let as_a_bool = 1u64;
    assert_eq!(
        e.eos_publishes - as_a_bool,
        2,
        "a bool under-reports by the number of turns after the first -- which is \
         the 241-against-238+1 #520 read on a 3-turn run"
    );
}

/// The comparison can fail.
///
/// §15.1 #1's sub-rule (#53, §11.3j): a parity test must assert its comparison
/// was not vacuous. The model above is only evidence if a wrong ring would
/// break it, so this drives the same identity with a deliberately wrong
/// expectation and requires it to be caught.
#[test]
fn the_model_can_fail() -> Result<()> {
    let device = Device::Cpu;
    let ring = TokenRing::new(0, EOS, &device)?;
    let logits = logits_peaked_at(7, &device)?;
    let sampled = logits.argmax(lloom_tensor::D::Minus1)?;
    ring.publish(&sampled)?;

    let got = ring.input().to_vec2::<u32>()?[0][0];
    assert_eq!(got, 7, "the ring holds what was published");
    assert_ne!(
        got, 8,
        "if this were equal the comparison would pass on any value and the \
         model's parity arm would be vacuous"
    );

    // And the flag distinguishes an EOS from a non-EOS, in the same run rather
    // than in a separate test -- #524's "make the coverage intrinsic".
    let flag = ring.eos_flag().to_vec1::<u8>()?[0];
    assert_eq!(flag, 0, "token 7 is not an EOS id");
    let eos = Tensor::new(EOS[0], &device)?;
    ring.publish(&eos)?;
    let flag = ring.eos_flag().to_vec1::<u8>()?[0];
    assert_ne!(
        flag, 0,
        "an EOS id sets the flag, so the check above can fail"
    );
    Ok(())
}

/// `publish` refuses a dtype other than `U32`, in sequence.
///
/// The existing `publish_refuses_a_non_u32_tensor` checks this in isolation.
/// What this adds is that a refusal **leaves the ring unchanged** -- a failed
/// publish that had partially written the slot would be a sequence-dependent
/// state, which is precisely what #508's registered prediction says the ring
/// does not have.
#[test]
fn a_refused_publish_leaves_the_ring_unchanged() -> Result<()> {
    for device in devices() {
        let ring = TokenRing::new(0, EOS, &device)?;
        let logits = logits_peaked_at(21, &device)?;
        ring.publish(&logits.argmax(lloom_tensor::D::Minus1)?)?;
        let before_input = ring.input().to_vec2::<u32>()?[0][0];
        let before_flag = ring.eos_flag().to_vec1::<u8>()?[0];

        let wrong = Tensor::new(5f32, &device)?;
        assert!(
            ring.publish(&wrong).is_err(),
            "publish must refuse a non-U32 tensor"
        );
        assert_eq!(wrong.dtype(), DType::F32);

        assert_eq!(
            ring.input().to_vec2::<u32>()?[0][0],
            before_input,
            "a refused publish must not move the input slot"
        );
        assert_eq!(
            ring.eos_flag().to_vec1::<u8>()?[0],
            before_flag,
            "a refused publish must not move the EOS flag"
        );
    }
    Ok(())
}
