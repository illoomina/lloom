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

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
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
    let cmds = commands(&device);
    let clock = cmds.clock();
    let x = device.new_buffer(4096, SHARED).unwrap();
    let event = device.as_ref().newSharedEvent().expect("a shared event");

    for _ in 0..2 {
        drop(cmds.command_encoder().unwrap());
        let m = cmds.flush_marker().unwrap();
        cmds.wait_marker(&m).unwrap();
    }
    cmds.encode_wait_for_event(&event, 1).unwrap();
    let held = clock.current_epoch().0;
    let writer = {
        let g = cmds.command_encoder().unwrap();
        g.as_ref().set_output_buffer(0, Some(&x), 0);
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
}

/// **A subscriber runs once per completed buffer, and a commit and a
/// completion both proceed while the subscribers' lock is held elsewhere.**
/// The test thread holds the lock that guards registration and teardown
/// while it commits and waits on eight buffers; each completion runs the
/// subscriber regardless. Killed by a commit or a completion that takes that
/// lock: the held lock then stalls the clock and the wait times out.
#[test]
fn subscribers_run_per_buffer_without_their_lock() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let runs = Arc::new(AtomicU64::new(0));
    let r = Arc::clone(&runs);
    cmds.on_command_buffer_complete(move || {
        r.fetch_add(1, Ordering::Relaxed);
    });
    let clock = cmds.clock();
    let buffers = 8u64;
    let reached = cmds.while_subscribers_locked(|| {
        for _ in 0..buffers {
            drop(cmds.command_encoder().unwrap());
            let m = cmds.flush_marker().unwrap();
            cmds.wait_marker(&m).unwrap();
        }
        let start = Instant::now();
        while clock.completed_count().0 < buffers {
            if start.elapsed() > Duration::from_secs(10) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    });
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
