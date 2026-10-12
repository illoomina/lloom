//! Unretained command buffers, executed: what the host must hold for them, and
//! what happens when a command buffer fails (`in_flight.rs`).
//!
//! A command buffer from `commandBufferWithUnretainedReferences` names objects
//! it does not own; the test below asserts the queue's hold keeps what the
//! engine itself drops, with absolute counts (§3.1b rule 1). The failure
//! injection — a command buffer Metal fails, and the wait that must report
//! it — is `tests/metal_cb_failure_tests.rs`, in a process of its own.

use crate::kernels::metal::commands::CommandBufferKind;
use crate::kernels::metal::completion::CompletionWait;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::Device;
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device, kind: CommandBufferKind) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new_with(queue, residency_set, CompletionWait::Synchronize, kind).unwrap()
}

fn filled(device: &Device, len: usize, byte: u8) -> crate::kernels::metal::Buffer {
    let b = device.new_buffer(len, SHARED).unwrap();
    // SAFETY: fresh shared buffer, `len` bytes, no GPU reader.
    unsafe { std::ptr::write_bytes(b.contents(), byte, len) };
    b
}

fn bytes(b: &crate::kernels::metal::Buffer, len: usize) -> Vec<u8> {
    // SAFETY: shared storage of at least `len` bytes; callers read after a wait.
    unsafe { std::slice::from_raw_parts(b.contents(), len) }.to_vec()
}

/// **A fence the map forgets is held until its command buffer has run.** Two
/// blit encoders write one destination; the second replaces the first's fence
/// in the cross-encoder map, after waiting on it, while both are still in the
/// open command buffer. The hold takes the fence; the copy lands; and once the
/// epoch completes the hold lets it go.
#[test]
fn an_unretained_queue_holds_a_forgotten_fence_until_its_epoch_completes() {
    let device = device();
    let len = 4096;
    let cmds = commands(&device, CommandBufferKind::Unretained);
    let hold = Arc::clone(
        cmds.in_flight_hold()
            .expect("the unretained kind has a hold"),
    );
    let a = filled(&device, len, 0x11);
    let b = filled(&device, len, 0x22);
    let dst = filled(&device, len, 0);
    for src in [&a, &b] {
        let mut blit = cmds.blit_command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { blit.copy_from_buffer(src, 0, &dst, 0, len) };
    }
    let held = hold.counts();
    assert_eq!(
        held.fences, 1,
        "the first copy's fence left the map: {held:?}"
    );
    assert_eq!(held.pending, 1, "{held:?}");

    cmds.flush_and_wait().expect("clean");
    assert!(bytes(&dst, len).iter().all(|&x| x == 0x22));
    assert_eq!(hold.release_completed(), 1, "its epoch has completed");
    assert_eq!(hold.counts().pending, 0);

    let retaining = commands(&device, CommandBufferKind::Retaining);
    assert!(
        retaining.in_flight_hold().is_none(),
        "Metal holds it on this kind"
    );
}
