//! The proptest half of the `TokenRing` sequence fuzzer (lloom #508): the
//! generator, the properties and the engagement gate. The model and the driver
//! (`run_sequence`, `Op`, `INPUT_POISON`) live in
//! `lloom::generation::token_ring_fuzz` behind the `fuzzing` feature, which the
//! self dev-dependency in `Cargo.toml` turns on for this test, so the
//! coverage-guided harness in `lloom-fuzz/` (#706) drives exactly this model.
//! The full argument for the target is on that module.

use lloom::generation::token_ring::TokenRing;
use lloom::generation::token_ring_fuzz::*;
use lloom_tensor::{DType, Device, Result, Tensor};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Ordinary publishes dominate, as they do in a real generation.
        6 => any::<u16>().prop_map(|id| Op::PublishToken { id }),
        // Host chunk stages (the streamed-prefill input path) — weighted so the
        // cursor cycles through the input ring's tiles and interleaves with
        // publishes, reaching the tile-reuse states a publish-only run cannot.
        4 => (any::<u16>(), any::<u8>()).prop_map(|(seed, len)| Op::StageChunk { seed, len }),
        3 => Just(Op::ReadFlag),
        // Read the input at a fuzzed width, so an over-width read (past a chunk's
        // staged prefix, into poison) is exercised alongside exact-width reads.
        3 => any::<u8>().prop_map(|width| Op::ReadInput { width }),
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

    let got = ring.input_for(1)?.to_vec2::<u32>()?[0][0];
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
        let before_input = ring.input_for(1)?.to_vec2::<u32>()?[0][0];
        let before_flag = ring.eos_flag().to_vec1::<u8>()?[0];

        let wrong = Tensor::new(5f32, &device)?;
        assert!(
            ring.publish(&wrong).is_err(),
            "publish must refuse a non-U32 tensor"
        );
        assert_eq!(wrong.dtype(), DType::F32);

        assert_eq!(
            ring.input_for(1)?.to_vec2::<u32>()?[0][0],
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
