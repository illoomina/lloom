//! The submission-lock census on the production locks (`DESIGN.md` §6.4d).
//!
//! `contention.rs`'s own tests show the wrapper counts a wait and nothing
//! else. These show each production site routes through it: a bind's
//! fence-map read blocked by a writer elsewhere is one `map_host_read` wait,
//! and an encoder request blocked on the encoder-state lock is one
//! `state_host` wait, each asserted absolutely beside a control in which the
//! same sites run uncontended and record zero while the fence-probe count
//! shows the reads happened. And they show the completion thread takes none
//! of these locks, and that an entry leaves the map only after its writer's
//! command buffer completed.

use crate::kernels::metal::contention::{Side, SubmissionContention};
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::Device;
use objc2_metal::{MTLDevice, MTLResourceOptions, MTLSharedEvent};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// Long enough that the blocked side is certain to arrive while the holder
/// has the lock, on a loaded machine.
const HOLD: Duration = Duration::from_millis(200);

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

/// Every field but the named ones is zero.
fn only(d: &SubmissionContention, allowed: &[&str]) {
    for (name, c) in [
        ("map_host_read", d.map_host_read),
        ("map_host_write", d.map_host_write),
        ("live_fences_host", d.live_fences_host),
        ("state_host", d.state_host),
    ] {
        if !allowed.contains(&name) {
            assert_eq!(c.waits, 0, "{name} recorded a wait nothing caused: {d:?}");
        }
    }
}

/// Binds, encoder closes and encoder requests on one thread, with nothing
/// committed and so no completion thread: every site runs, none waits, and
/// the census reads zero while the probe count shows the map reads ran.
#[test]
fn uncontended_submission_records_nothing() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let x = device.new_buffer(4096, SHARED).unwrap();
    let y = device.new_buffer(4096, SHARED).unwrap();
    let mut probes = 0;
    for _ in 0..8 {
        {
            let g = cmds.command_encoder().unwrap();
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe { g.as_ref().set_input_buffer(0, Some(&x), 0) };
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe { g.as_ref().set_output_buffer(1, Some(&y), 0) };
            probes += g.as_ref().fence_stats().0;
        }
        // Ends the compute session: its outputs are written into the map.
        drop(cmds.blit_command_encoder().unwrap());
    }
    assert_eq!(
        probes, 16,
        "the map was read: two buffers a session, eight sessions"
    );
    assert_eq!(
        cmds.submission_contention(),
        SubmissionContention::default(),
        "an uncontended acquisition is not a wait"
    );
    cmds.flush_and_wait().unwrap();
}

/// A bind whose map read meets a write held on another thread is one
/// `map_host_read` wait, with the time it blocked.
#[test]
fn a_bind_blocked_by_a_map_writer_is_one_host_read_wait() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let x = device.new_buffer(4096, SHARED).unwrap();
    let before = cmds.submission_contention();
    {
        let g = cmds.command_encoder().unwrap();
        let enc = g.as_ref();
        let map = Arc::clone(&enc.prev_ce_outputs);
        let (held, is_held) = mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _w = map.write(Side::Host);
            held.send(()).unwrap();
            std::thread::sleep(HOLD);
        });
        is_held.recv().unwrap();
        let t = Instant::now();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(0, Some(&x), 0) };
        let waited = t.elapsed();
        holder.join().unwrap();
        assert!(waited >= HOLD / 4, "the bind returned after {waited:?}");
    }
    let d = cmds.submission_contention().since(&before);
    assert_eq!(d.map_host_read.waits, 1, "{d:?}");
    assert!(d.map_host_read.blocked_ns > 0);
    only(&d, &["map_host_read"]);
    cmds.flush_and_wait().unwrap();
}

/// A command buffer completes while the host holds the fence map and the
/// fence book: the completion thread takes neither. The GPU clock's handler,
/// registered after the session's, advances under them -- it could not if the
/// session's handler were waiting on a lock the test holds -- and the session's
/// entry leaves the map at the host's next encoder close.
#[test]
fn a_completion_never_waits_on_the_hosts_fence_locks() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let x = device.new_buffer(4096, SHARED).unwrap();
    let event = device.as_ref().newSharedEvent().expect("a shared event");
    let before = cmds.submission_contention();

    let map = {
        let g = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { g.as_ref().set_output_buffer(0, Some(&x), 0) };
        Arc::clone(&g.as_ref().prev_ce_outputs)
    };
    // Closes the session (x enters the map; its handler is attached) and
    // holds the command buffer until the event reaches 1.
    cmds.encode_wait_for_event(&event, 1).unwrap();
    let marker = cmds.flush_marker().unwrap();
    let clock = cmds.clock();
    let completed_before = clock.completed_count();

    let advanced = cmds.while_fence_locks_held(|| {
        event.setSignaledValue(1);
        let start = Instant::now();
        while clock.completed_count() == completed_before {
            if start.elapsed() > Duration::from_secs(10) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    });
    assert!(
        advanced,
        "the buffer's completion did not get past its handlers while the host \
         held the fence map and the fence book"
    );
    assert!(cmds.marker_complete(&marker));
    let key = x.raw_ptr() as usize;
    assert!(
        map.read(Side::Host).contains_key(&key),
        "the entry is removed by the host at its next encoder close, not by the \
         completion thread"
    );
    // Until then a bind treats the completed writer as absent: one probe, no
    // hit, no wait -- what it found once the handler had removed the entry.
    {
        let g = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { g.as_ref().set_input_buffer(0, Some(&x), 0) };
        assert_eq!(
            g.as_ref().fence_stats(),
            (1, 0, 0),
            "a bind waited on a writer whose command buffer had completed"
        );
    }
    // The close reaps it.
    drop(cmds.blit_command_encoder().unwrap());
    assert!(
        !map.read(Side::Host).contains_key(&key),
        "a completed session's entry survived an encoder close"
    );
    only(&cmds.submission_contention().since(&before), &[]);
    cmds.flush_and_wait().unwrap();
}

/// An entry leaves the map only after its command buffer completed: a close
/// while the writer's buffer is still held keeps it, pointing at the writer's
/// fence, and a close after the buffer completes removes it.
#[test]
fn an_incomplete_writer_is_never_reaped() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let x = device.new_buffer(4096, SHARED).unwrap();
    let event = device.as_ref().newSharedEvent().expect("a shared event");
    let key = x.raw_ptr() as usize;

    let (map, writer) = {
        let g = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { g.as_ref().set_output_buffer(0, Some(&x), 0) };
        (
            Arc::clone(&g.as_ref().prev_ce_outputs),
            Arc::clone(&g.as_ref().fence),
        )
    };
    cmds.encode_wait_for_event(&event, 1).unwrap();
    let marker = cmds.flush_marker().unwrap();

    // Two closes while the writer's buffer cannot complete; a bind in each
    // still finds the writer and waits on it.
    for _ in 0..2 {
        {
            let g = cmds.command_encoder().unwrap();
            // SAFETY: the test owns every buffer bound or mapped here and reads its
            // results only after waiting for the command buffer that writes them.
            unsafe { g.as_ref().set_input_buffer(0, Some(&x), 0) };
            assert_eq!(
                g.as_ref().fence_stats(),
                (1, 1, 1),
                "a bind did not wait on a writer whose command buffer is incomplete"
            );
        }
        drop(cmds.blit_command_encoder().unwrap());
    }
    assert!(!cmds.marker_complete(&marker), "the event holds the buffer");
    assert!(
        map.read(Side::Host)
            .get(&key)
            .is_some_and(|f| Arc::ptr_eq(f, &writer)),
        "an entry was removed while its writer's command buffer was incomplete"
    );

    event.setSignaledValue(1);
    cmds.wait_marker(&marker).unwrap();
    // `wait_marker` returns on the buffer's status; its handlers run on
    // Metal's thread around then. Close until the host sees the flag.
    let start = Instant::now();
    while map.read(Side::Host).contains_key(&key) {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "a completed session's entry was never reaped"
        );
        drop(cmds.command_encoder().unwrap());
        drop(cmds.blit_command_encoder().unwrap());
        std::thread::sleep(Duration::from_millis(1));
    }
    cmds.flush_and_wait().unwrap();
}

/// An encoder request made while another thread holds the encoder-state lock
/// is one `state_host` wait.
#[test]
fn an_encoder_request_blocked_on_the_state_lock_is_one_state_wait() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = Arc::new(commands(&device));
    let before = cmds.submission_contention();
    let (held, is_held) = mpsc::channel();
    let other = Arc::clone(&cmds);
    let holder = std::thread::spawn(move || {
        other.while_state_held(|| {
            held.send(()).unwrap();
            std::thread::sleep(HOLD);
        })
    });
    is_held.recv().unwrap();
    drop(cmds.command_encoder().unwrap());
    holder.join().unwrap();
    let d = cmds.submission_contention().since(&before);
    assert_eq!(d.state_host.waits, 1, "{d:?}");
    assert!(d.state_host.blocked_ns > 0);
    only(&d, &["state_host"]);
    cmds.flush_and_wait().unwrap();
}
