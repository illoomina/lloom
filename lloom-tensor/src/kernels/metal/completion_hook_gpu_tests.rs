//! The completion hook on a real queue (`completion_hook.rs`): the clock
//! marks exactly the command buffers that completed, in commit order, while
//! a later one is held by the GPU; a session's fence completes with its own
//! buffer; subscribers run once per completed buffer; and neither a commit
//! nor a completion takes the subscribers' lock.
//!
//! A buffer is held by a shared-event wait the host signals, so "while it is
//! held" is a state the test controls rather than a race it hopes to win.

use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::Device;
use objc2_metal::{MTLDevice, MTLResourceOptions, MTLSharedEvent};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// How long a held buffer is watched for a clock that runs ahead of it.
const WATCH: Duration = Duration::from_millis(100);

/// Arm before encoding a host wait or starting a possibly blocked producer.
/// Drop never joins/drains while its event/lock is still held.
struct ReleaseBeforeDrain<R: FnMut(), D: FnMut()> {
    release: R,
    drain: D,
}
impl<R: FnMut(), D: FnMut()> Drop for ReleaseBeforeDrain<R, D> {
    fn drop(&mut self) {
        (self.release)();
        (self.drain)();
    }
}

#[test]
fn fixture_release_precedes_drain_on_assertion_unwind() {
    let released = std::cell::Cell::new(false);
    let observed = std::cell::Cell::new(false);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ReleaseBeforeDrain {
            release: || released.set(true),
            drain: || observed.set(released.get()),
        };
        panic!("controlled completion fixture unwind");
    }));
    assert!(result.is_err());
    assert!(observed.get(), "fixture drain preceded release");
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    let commands = Commands::new(queue, residency_set).unwrap();
    let actual = commands.actual_command_api_for_test();
    let requested = &crate::tuning::tuning().metal.command_api;
    if requested != "auto" {
        assert_eq!(actual, requested);
    }
    if let Ok(requested) = std::env::var("LLOOM_METAL_API") {
        if requested != "auto" {
            assert_eq!(actual, requested);
        }
    }
    eprintln!(
        "HOOK_FIXTURE_ARM actual_api={actual} resolved_wait={}",
        commands.completion_wait().name()
    );
    commands
}

/// Spin until `done` holds, or fail after ten seconds naming `what`.
fn eventually(what: &str, done: impl Fn() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "never happened: {what}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// **The n-th completion is epoch n, and the clock never runs ahead of the
/// buffers that completed.** Two buffers complete; the third is held by an
/// event wait in front of a session writing `x`, and a fourth is queued
/// behind it. While the third is held the clock reads exactly the buffers
/// before it, and the writer's fence is incomplete; once the event is
/// signalled the clock reads every commit and the fence is complete.
///
/// Absolute references (§3.1b): the held epoch is read from the clock before
/// the writer encodes, so the test holds on either command API (Metal 4
/// commits at an event boundary, Metal 3 records the wait into the open
/// buffer). Killed by a hook that marks one epoch ahead, and by a fence that
/// claims its buffer's predecessor's epoch.
#[test]
fn the_clock_marks_completed_buffers_in_commit_order_and_no_further() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = Arc::new(commands(&device));
    let clock = cmds.clock();
    let x = device.new_buffer(4096, SHARED).unwrap();
    let event = device.as_ref().newSharedEvent().expect("a shared event");
    // Independent owners survive a refusal/unwind whose terminality cannot
    // be proved. Arm before even the first marker or host wait is encoded.
    let mut pins = Some((device.clone(), cmds.clone(), x.clone(), event.clone()));
    let cleanup = ReleaseBeforeDrain {
        release: || event.setSignaledValue(1),
        drain: || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cmds.flush_and_wait()));
            if matches!(result, Ok(Ok(()))) {
                drop(pins.take());
            } else {
                std::mem::forget(pins.take());
                eprintln!("clock fixture cleanup unknown; independent owners retained");
            }
        },
    };

    for _ in 0..2 {
        drop(cmds.command_encoder().unwrap());
        let m = cmds.flush_marker().unwrap();
        cmds.wait_marker(&m).unwrap();
    }
    cmds.encode_wait_for_event(&event, 1).unwrap();
    let held = clock.current_epoch().0;
    let writer = {
        let g = cmds.command_encoder().unwrap();
        // SAFETY: the test owns `x`, binds it only for its fence (no
        // dispatch reads or writes it), and drops it after every wait.
        unsafe { g.as_ref().set_output_buffer(0, Some(&x), 0) };
        Arc::clone(&g.as_ref().fence)
    };
    let held_marker = cmds.flush_marker().unwrap();
    drop(cmds.command_encoder().unwrap());
    let behind = cmds.flush_marker().unwrap();
    let committed = clock.current_epoch().0;
    assert_eq!(committed, held + 2, "one buffer held, one behind it");

    eventually("every buffer before the held one completed", || {
        clock.completed_count().0 == held
    });
    std::thread::sleep(WATCH);
    assert_eq!(
        clock.completed_count().0,
        held,
        "the clock ran ahead of the held buffer"
    );
    assert!(
        !cmds.marker_complete(&held_marker),
        "the event holds the buffer"
    );
    assert!(
        !writer.is_completed(),
        "the fence of a session in the held buffer reads complete"
    );
    assert!(cmds.check_failed().is_ok());

    event.setSignaledValue(1);
    cmds.wait_marker(&behind).unwrap();
    eventually("the clock reached every commit", || {
        clock.completed_count().0 == committed
    });
    assert!(
        writer.is_completed(),
        "the fence did not complete with its buffer"
    );
    std::thread::sleep(WATCH);
    assert_eq!(
        clock.completed_count().0,
        committed,
        "the clock ran past its commits"
    );
    assert!(cmds.check_failed().is_ok());
    drop(cleanup);
}

/// **A subscriber runs once per completed buffer, and a commit and a
/// completion both proceed while the subscribers' lock is held elsewhere.**
/// The test thread holds the lock that guards registration and teardown
/// while a producer commits eight buffers and it watches the clock; each
/// completion runs the subscriber regardless. A lock mutant stalls progress
/// during this bounded observation. The fixture releases the lock before
/// joining/draining, then fails the intended assertion instead of hanging.
#[test]
fn subscribers_run_per_buffer_without_their_lock() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = Arc::new(commands(&device));
    let runs = Arc::new(AtomicU64::new(0));
    let r = Arc::clone(&runs);
    cmds.on_command_buffer_complete(move || {
        r.fetch_add(1, Ordering::Relaxed);
    })
    .unwrap();
    let clock = cmds.clock();
    let buffers = 8u64;
    cmds.seal_completion_subscribers().unwrap();
    let join = std::cell::RefCell::new(None::<std::thread::JoinHandle<_>>);
    let joined = std::cell::RefCell::new(None);
    let mut lock = Some(cmds.lock_subscribers_for_test());
    let mut pins = Some((device.clone(), cmds.clone()));
    let cleanup = ReleaseBeforeDrain {
        release: || drop(lock.take()),
        drain: || {
            let outcome = join.borrow_mut().take().map(|handle| handle.join());
            let drained =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cmds.flush_and_wait()));
            if matches!(&outcome, Some(Ok(Ok(())))) && matches!(drained, Ok(Ok(()))) {
                drop(pins.take());
            } else {
                std::mem::forget(pins.take());
                eprintln!("subscriber fixture cleanup unknown; independent owners retained");
            }
            *joined.borrow_mut() = outcome;
        },
    };
    let owner = cmds.clone();
    *join.borrow_mut() = Some(std::thread::spawn(move || {
        for _ in 0..buffers {
            drop(owner.command_encoder()?);
            owner.flush()?;
        }
        Ok::<(), crate::kernels::MetalKernelError>(())
    }));
    let start = Instant::now();
    while clock.completed_count().0 < buffers && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let reached = clock.completed_count().0 == buffers;
    // A commit-lock mutant may still be blocked. Unlock BEFORE join/drain;
    // only then report the intended progress assertion, never a timeout kill.
    drop(cleanup);
    joined.into_inner().unwrap().unwrap().unwrap();
    assert!(
        reached,
        "completions stalled while the test held the subscribers' lock"
    );
    eventually("the subscriber ran for every buffer", || {
        runs.load(Ordering::Relaxed) == buffers
    });
    std::thread::sleep(WATCH);
    assert_eq!(
        runs.load(Ordering::Relaxed),
        buffers,
        "a subscriber ran more than once per buffer"
    );
}
