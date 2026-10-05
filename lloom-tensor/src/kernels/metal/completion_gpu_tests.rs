//! The completion seam, **executed** (`DESIGN.md` §11.5o, issue #559).
//!
//! `completion.rs`'s own tests cover the routing arithmetic, which needs no GPU:
//! parsing, name distinctness, the default, and the fail-safe direction. These
//! are the properties that can only be established by running, and each is one
//! #500 or #536 would have caught had it existed:
//!
//! 1. **The async arm actually signals.** #500's finding is that a mechanism
//!    with no production callers reports a clean null, and #536's is that a test
//!    of the *mechanism* passes while the *routing* is mutated dead. So the
//!    assertion is on the engagement counters, which are a property of the path
//!    taken rather than of the code existing.
//! 2. **The async arm computes the same bytes.** A wait that returns early is a
//!    read-before-write, which §3.5 makes silent corruption rather than an
//!    error -- so it presents as a wrong value, never as a failure, and the only
//!    thing that catches it is comparing against a reference.
//! 3. **The default arm does not signal at all.** The incumbent must be
//!    byte-for-byte what it was, and "the flag is off" is a claim about a
//!    counter reading zero rather than about intent.
//!
//! Every test asserts an **absolute** reference rather than a shape (§3.1b rule
//! 1): `signals == 1` rather than *"signals did not decrease"*, and the read-back
//! value against the arithmetic the kernel was asked to do rather than against
//! the other arm's answer -- a defect shifting both arms equally is exactly what
//! a same-run comparison cannot see (#519).

use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{call_affine, BufferOffset, Device, Kernels};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device, arm: CompletionWait) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new_with_completion_wait(queue, residency_set, arm).unwrap()
}

/// Dispatch one `affine` over `n` f32 and read the result back through the
/// seam. Returns `(values, signals, polls)`.
///
/// `affine` is chosen because it is a real shipping kernel with a known closed
/// form -- `mul * x + add` -- so the result can be checked against arithmetic
/// rather than against the other arm. A fixture whose expected value comes from
/// the run is #519's "looks like a comparison and is not".
///
/// For the listener arm see [`run_affine_full`], which additionally returns the
/// **fired** count -- the only engagement proof that arm has, since it polls
/// zero times whether it worked or fell back to blocking (#583).
fn run_affine(arm: CompletionWait, n: usize, mul: f32, add: f32) -> (Vec<f32>, u64, u64) {
    let (out, signals, polls, _fired) = run_affine_full(arm, n, mul, add);
    (out, signals, polls)
}

/// As [`run_affine`], also returning the notification blocks that fired (#583).
///
/// Separate rather than folded into every call site because `fired` is
/// meaningful on exactly one arm, and a fourth element every other test ignores
/// invites reading a structural zero as a result.
fn run_affine_full(arm: CompletionWait, n: usize, mul: f32, add: f32) -> (Vec<f32>, u64, u64, u64) {
    let device = device();
    let cmds = commands(&device, arm);
    let kernels = Kernels::new();

    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bytes = std::mem::size_of_val(&input[..]);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, SHARED)
        .unwrap();
    let dst = device.new_buffer(bytes, SHARED).unwrap();

    {
        let encoder = cmds.command_encoder().unwrap();
        call_affine(
            &device,
            &encoder,
            &kernels,
            "affine_f32",
            std::mem::size_of::<f32>(),
            n,
            BufferOffset::zero_offset(&src),
            &dst,
            mul,
            add,
        )
        .unwrap();
    }

    // The seam. On `Synchronize` this is `waitUntilCompleted`; on an async arm
    // the host polls the shared event and never enters the kernel to wait.
    cmds.flush_and_wait_current().unwrap();

    let p = dst.contents() as *const f32;
    assert!(!p.is_null(), "destination has no CPU mapping");
    // SAFETY: shared storage sized for `n` f32, and the wait above has returned.
    let out = unsafe { std::slice::from_raw_parts(p, n) }.to_vec();
    let (signals, polls) = cmds.completion_counts();
    (out, signals, polls, cmds.completion_fires())
}

fn expected(n: usize, mul: f32, add: f32) -> Vec<f32> {
    (0..n).map(|i| i as f32 * mul + add).collect()
}

/// **The incumbent signals nothing**, which is what "byte-for-byte what it was"
/// means as a checkable claim rather than an intention.
///
/// The absolute reference is `0`, not "fewer than the async arm": a default that
/// signalled once per token would be a behaviour change invisible to a
/// comparison against another arm that also signals.
#[test]
fn the_default_arm_signals_nothing_and_is_correct() {
    let (out, signals, polls) = run_affine(CompletionWait::Synchronize, 64, 3.0, 1.5);
    assert_eq!(signals, 0, "the incumbent must encode no signal");
    assert_eq!(polls, 0, "the incumbent must not poll");
    assert_eq!(out, expected(64, 3.0, 1.5));
    assert!(
        !CompletionWait::Synchronize.needs_event(),
        "the incumbent must not allocate an event"
    );
}

/// **The async arm engages and computes the same bytes.**
///
/// `signals >= 1` is the engagement proof #500's finding demands: a mechanism
/// that exists and is never reached reports a clean null, and the counter is
/// what separates *"the arm ran and cost nothing"* from *"the arm did not run"*.
///
/// The value assertion is the one that matters for soundness. A poll that
/// returns before the GPU has written produces a wrong value and **no error**
/// (§3.5), so the arithmetic check is the only thing standing between a broken
/// wait and a green board. The probe behind this seam
/// (`measurements/probes/issue-559/shared_event.m`) establishes the ordering
/// property directly; this is the same property asserted through the shipping
/// path.
#[test]
fn the_shared_event_arm_engages_and_agrees() {
    let (out, signals, _polls) = run_affine(CompletionWait::SharedEvent, 64, 3.0, 1.5);
    assert!(
        signals >= 1,
        "the shared-event arm encoded no signal, so it did not run \
         (DESIGN.md §11.5i: a mechanism with no callers reports a clean null)"
    );
    assert_eq!(
        out,
        expected(64, 3.0, 1.5),
        "the async wait returned before the GPU finished: a wrong value and no \
         error is what a read-before-write looks like (§3.5)"
    );
}

/// The sleeping arm is the **third mechanism**, and it is here to show the seam
/// admits one rather than to assert it is better.
///
/// Issue #559 asks that the seam admit a third *without restructuring* and that
/// this be shown rather than asserted. This test is the showing: adding
/// `SharedEventSleep` touched `CompletionWait` and nothing else -- not
/// `commands.rs`'s call sites, not `ensure_completed_via`'s status machine, not
/// the error handling.
#[test]
fn the_third_arm_engages_and_agrees() {
    let (out, signals, _polls) = run_affine(
        CompletionWait::SharedEventSleep { micros: 50 },
        64,
        3.0,
        1.5,
    );
    assert!(signals >= 1, "the sleeping arm encoded no signal");
    assert_eq!(out, expected(64, 3.0, 1.5));
}

/// **Every arm agrees with the arithmetic, and the arithmetic is external
/// to all of them.**
///
/// §3.1b rule 2: a shape invariant ("the arms agree") is blind to a defect that
/// shifts every arm equally, and a per-operation defect moves every member by
/// construction. So the reference is the closed form, and arm-to-arm agreement
/// is a consequence rather than the assertion.
///
/// **The list is checked against the parser rather than maintained by hand**
/// (#583). This test read *"all three arms"* while a fourth existed and would
/// have gone on passing over three -- §3.1b rule 4's incidental coverage, where
/// the blindness is silent in both directions. `ARMS` below is the shared list
/// and `every_arm_is_covered_here` is what pins it to reality.
#[test]
fn every_arm_agrees_with_the_closed_form() {
    let want = expected(128, -2.0, 7.25);
    for arm in ARMS {
        let (out, _, _) = run_affine(arm, 128, -2.0, 7.25);
        assert_eq!(out, want, "arm {} disagrees", arm.name());
    }
}

/// Every arm this file executes: **the enum's own list**, not a copy (#583).
///
/// This was a hand-written `const` array of four, checked against a
/// hand-written list of four parser names. **Both went stale on the next arm
/// added** -- the fifth arm was selectable, shipped and executed by no test
/// here, while `every_arm_is_covered_here` stayed green, because the test
/// compared two hand-written lists **to each other**. That is #519's *"a
/// comparison that looks like one and is not"* (§3.1b) inside the check written
/// to prevent exactly this, and it is why the reference now lives on the enum
/// (`CompletionWait::ALL_ARMS`) with `no_arm_is_missing_from_all_arms` pinning
/// it to the compiler.
const ARMS: [CompletionWait; CompletionWait::ALL_ARMS.len()] = CompletionWait::ALL_ARMS;

/// **The coverage list is not allowed to fall behind the enum** (#583).
///
/// `ARMS` is now `CompletionWait::ALL_ARMS` itself, so this can no longer fail
/// by the route it was written for -- and it is kept, because what it asserts
/// is the property rather than the mechanism: **every arm a user can select is
/// executed by this file**. If someone re-introduces a local list, this goes red.
///
/// The check is against the **parser**, which is the enumeration a user can
/// actually select, and the parser names come from `ALL_ARMS` rather than from a
/// literal -- which is the half that went stale last time.
#[test]
fn every_arm_is_covered_here() {
    for arm in CompletionWait::ALL_ARMS {
        let parsed = CompletionWait::from_str_arm(arm.name())
            .unwrap_or_else(|e| panic!("{} no longer parses: {e}", arm.name()));
        assert!(
            ARMS.iter().any(|a| a.name() == parsed.name()),
            "arm {} is selectable and is executed by no test in this file",
            arm.name()
        );
    }
    // The control: a name that is *not* an arm must not be found in the list,
    // or the assertion above would pass for any string (§3.1a rule 3).
    assert!(CompletionWait::from_str_arm("shared-event-park").is_err());
    assert_eq!(
        ARMS.len(),
        CompletionWait::ALL_ARMS.len(),
        "ARMS has an untested member"
    );
}

/// **The listener arm is woken by a block, and the block is proved to fire**
/// (#583).
///
/// This is the issue's own definition of done: *"the block firing is asserted,
/// not assumed -- a listener that stopped firing reads as a clean run."* That is
/// not a stylistic preference. Apple's guidance is that a deallocated listener
/// silently stops delivering blocks, and this arm falls back to a blocking wait
/// when nothing arrives -- so **a broken listener computes the right answer**.
/// Every value assertion in this file would pass, both digests would be
/// unmoved, and `completion_polls` would read the same structural 0 it reads
/// when the arm works.
///
/// `fired >= 1` is the only observation that separates the two, which is why
/// the counter exists (§3.1c: a zero is not evidence unless something nonzero
/// came from the same command; here the *nonzero* is the thing being asserted
/// and its control is the arm below, which fires nothing on the same work).
///
/// The dispatch is large enough that the GPU is still busy when the host
/// arrives, so a fired block cannot be explained by the work already being done
/// -- the same discriminator `the_parking_arm_engages_without_a_single_poll`
/// uses.
#[test]
fn the_listener_arm_fires_its_block_and_agrees() {
    let n = 1 << 20;
    let (out, signals, polls, fired) = run_affine_full(
        CompletionWait::SharedEventListener { timeout_ms: 2_000 },
        n,
        1.0,
        0.0,
    );
    assert!(
        signals >= 1,
        "the listener arm encoded no signal, so it did not run (§11.5i)"
    );
    assert!(
        fired >= 1,
        "the listener arm's notification block never fired: the wait fell back \
         to blocking and computed the right answer anyway, which is exactly \
         what a deallocated listener looks like and what no value assertion \
         can see"
    );
    assert_eq!(
        polls, 0,
        "the listener arm polled: it is woken by a block, and a nonzero count \
         means it took a loop instead"
    );
    assert_eq!(
        out[n - 1],
        (n - 1) as f32,
        "the notified wait returned before the GPU finished: a wrong value and \
         no error is what a read-before-write looks like (§3.5)"
    );
    assert_eq!(out[0], 0.0);
}

/// **No other arm fires a block**, which is what makes the count above evidence
/// (#583).
///
/// §3.1c's rule in both directions: the test above asserts a nonzero from a
/// counter, and this is the control showing the counter is not simply always
/// nonzero. An arm that does not register a listener must report `fired = 0` on
/// the same work through the same counter.
#[test]
fn only_the_listener_arm_fires_a_block() {
    let n = 1 << 20;
    for arm in [
        CompletionWait::Synchronize,
        CompletionWait::SharedEvent,
        CompletionWait::SharedEventWait { timeout_ms: 2_000 },
    ] {
        let (_, _, _, fired) = run_affine_full(arm, n, 1.0, 0.0);
        assert_eq!(
            fired,
            0,
            "arm {} fired a notification block and registers no listener",
            arm.name()
        );
    }
}

/// The listener is allocated for exactly the arm that needs one (#583).
///
/// *"Hold a strong reference for the engine's life"* is the research's first
/// production constraint, and this is the mechanical half of it: the listener's
/// lifetime is the event's, and the event's is `Commands`'. A listener created
/// per wait would be dropped between tokens and the arm would silently become
/// the blocking fallback -- correct output, wrong arm.
#[test]
fn only_the_listener_arm_allocates_a_listener() {
    let device = device();
    for arm in CompletionWait::ALL_ARMS {
        let cmds = commands(&device, arm);
        assert_eq!(
            cmds.completion_listener_present(),
            arm.needs_listener(),
            "arm {} disagrees with its own needs_listener()",
            arm.name()
        );
    }
}

/// **The parking arm performs ZERO polls, and that is the point of it** (#583).
///
/// This is issue #583's own definition of done: *"an `MTLSharedEvent` blocking
/// wait arm exists and `completion_polls` reads 0 on it"*. The polling arms read
/// 8.2 M on a release run; this one reads 0 **structurally**, because there is
/// no loop to count.
///
/// **The zero is only readable beside the signal count**, which is why both are
/// asserted: `(signals >= 1, polls == 0)` is a wait that engaged and did not
/// iterate, while `(0, 0)` is an arm that never ran and is what §11.5o warns a
/// bare zero cannot be told apart from (§3.1c: a zero is not evidence unless
/// something nonzero came from the same command).
///
/// The dispatch is deliberately large enough that the GPU is still busy when the
/// host arrives, so the zero cannot be explained by the work being already
/// finished -- the same reasoning as
/// `the_async_arm_waits_for_work_that_is_not_already_done`, and it is the
/// discriminator that makes this a test of *parking* rather than of *luck*.
#[test]
fn the_parking_arm_engages_without_a_single_poll() {
    let n = 1 << 20;
    let (out, signals, polls) = run_affine(
        CompletionWait::SharedEventWait { timeout_ms: 2_000 },
        n,
        1.0,
        0.0,
    );
    assert!(
        signals >= 1,
        "the parking arm encoded no signal, so it did not run (§11.5i)"
    );
    assert_eq!(
        polls, 0,
        "the parking arm polled: it is supposed to hand the condition to the \
         runtime, and a nonzero count means it fell back to a loop"
    );
    assert_eq!(
        out[n - 1],
        (n - 1) as f32,
        "the parked wait returned before the GPU finished: a wrong value and no \
         error is what a read-before-write looks like (§3.5)"
    );
    assert_eq!(out[0], 0.0);
}

/// The polling arm on the **same** dispatch does poll, which is what makes the
/// zero above a measurement rather than a broken counter (#583).
///
/// §3.1c: *"a zero is not evidence unless something nonzero came from the same
/// command."* If `completion_polls` were wired to nothing, the parking arm's 0
/// would read identically -- so the discriminator is a second arm, on the same
/// size of work, through the same counter, reading nonzero.
#[test]
fn the_polling_arm_polls_on_the_same_work() {
    let n = 1 << 20;
    let (_, signals, polls) = run_affine(CompletionWait::SharedEvent, n, 1.0, 0.0);
    assert!(signals >= 1, "the polling arm encoded no signal");
    assert!(
        polls > 0,
        "the polling arm reported no polls on {n} elements, so the counter is \
         not attached and the parking arm's zero means nothing"
    );
}

/// The polling arm polls a buffer about to complete and parks on the rest: on
/// work that outlasts its bound it stops at the bound, and the value is right
/// (`completion::SHARED_EVENT_POLL_BOUND`, `expert-streaming.md` §20.14b).
///
/// The work is ~128 MB of traffic (~0.4 ms), against a bound of ~40 µs of
/// reads, so an arm that polled through the buffer would read far past it.
#[test]
fn the_polling_arm_parks_once_its_bound_is_spent() {
    let n = 1 << 24;
    let (out, signals, polls) = run_affine(CompletionWait::SharedEvent, n, 1.0, 0.0);
    assert!(signals >= 1);
    assert!(polls > 0, "the arm never polled");
    assert!(
        polls <= crate::kernels::metal::completion::SHARED_EVENT_POLL_BOUND,
        "the arm polled {polls} times: it spun through the buffer instead of parking"
    );
    assert_eq!(out[n - 1], (n - 1) as f32);
    assert_eq!(out[0], 0.0);
}

/// A larger dispatch, so the wait is not trivially satisfied on its first read.
///
/// The 64-element cases above may complete before the host looks, which would
/// make a broken poll loop indistinguishable from a working one -- the same
/// shape as §3.7h's "woke on its own and happened to see it". This runs enough
/// work that the GPU is genuinely still busy, and asserts the value.
#[test]
fn the_async_arm_waits_for_work_that_is_not_already_done() {
    let n = 1 << 20;
    let (out, signals, _polls) = run_affine(CompletionWait::SharedEvent, n, 1.0, 0.0);
    assert!(signals >= 1);
    // Spot-check rather than comparing a megabyte: the last element is the one
    // written latest, so it is the one an early return is most likely to miss.
    assert_eq!(out[n - 1], (n - 1) as f32);
    assert_eq!(out[0], 0.0);
}
