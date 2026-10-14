//! The cross-encoder fence probe, counted (lloom #732).
//!
//! `wait_for_buffer` orders an encoder session after the last writer of each
//! buffer it binds, through the cross-encoder map. On a real LFM2 decode step
//! that map was read **1101 times, 46 of which could matter** (lloom #727
//! §27): 67 % were repeat binds of a buffer the session had already probed,
//! 29 % were weights nothing writes. The change under test reads the map once
//! per buffer per session and waits once per fence, under the lock the bind
//! already takes.
//!
//! Under `HazardTrackingModeUntracked` a missing wait is silent corruption
//! rather than a failure (`DESIGN.md` §3.5), and `tests/metal_blit_ordering_tests.rs`
//! records that value-level tests stay green with every per-buffer wait
//! removed. So the gate here is the **engagement counters**, asserted
//! absolutely (§3.1b rule 1): how many times the map was read, how many reads
//! found a writer, and how many `waitForFence` calls were issued. Each number
//! is one a named mutation flips:
//!
//! - probe every bind again (the short-circuit removed): reads 2 → 4;
//! - never read the map (the probe dropped): hits and waits 1 → 0;
//! - wait per hit rather than per fence (the dedup removed): waits 1 → 2.
//!
//! The first session is ended by opening a blit encoder rather than by a
//! commit, so its outputs are registered into the map (that happens at
//! `end_encoding`) while the command buffer stays uncommitted -- a committed
//! buffer's completion handler would *remove* the entries on a driver thread,
//! and the second session's hit would be a race rather than a fact.

use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::Device;
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

#[test]
fn the_map_is_read_once_per_buffer_per_session_and_a_fence_is_waited_once() {
    let Some(device) = Device::system_default() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmds = commands(&device);
    let x = device.new_buffer(4096, SHARED).unwrap();
    let z = device.new_buffer(4096, SHARED).unwrap();
    let y = device.new_buffer(4096, SHARED).unwrap();

    // Session 1 writes X and Z. Nothing has written anything before it, so
    // every probe misses and no fence is waited.
    {
        let guard = cmds.command_encoder().unwrap();
        let enc = guard.as_ref();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_output_buffer(0, Some(&x), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_output_buffer(1, Some(&z), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_output_buffer(2, Some(&x), 0) };
        assert_eq!(
            enc.fence_stats(),
            (2, 0, 0),
            "session 1: two distinct buffers, no prior writer"
        );
    }
    // End session 1 without committing: the blit encoder ends the compute
    // encoder, which registers X and Z against session 1's fence.
    {
        let _blit = cmds.blit_command_encoder().unwrap();
    }

    // Session 2 reads X three times, Z once, Y (never written) once, and
    // writes X. Five binds of X/Z/Y, three distinct buffers: three map reads,
    // two hits (X, Z -- one writer, one fence), one wait.
    {
        let guard = cmds.command_encoder().unwrap();
        let enc = guard.as_ref();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(0, Some(&x), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(1, Some(&x), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(2, Some(&y), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(3, Some(&z), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_output_buffer(4, Some(&x), 0) };
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { enc.set_input_buffer(5, Some(&x), 0) };
        let (probes, hits, waits) = enc.fence_stats();
        assert_eq!(
            probes, 3,
            "one map read per distinct buffer, not per bind (6)"
        );
        assert_eq!(hits, 2, "X and Z have a registered writer; Y has none");
        assert_eq!(
            waits, 1,
            "X and Z share session 1's fence: one wait, not two"
        );
    }

    cmds.flush_and_wait().unwrap();
}
