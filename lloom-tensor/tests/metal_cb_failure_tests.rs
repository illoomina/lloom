#![cfg(feature = "metal")]
//! A command buffer that ends in error fails the run, whichever buffer the
//! host waits on next (`commands.rs`, the failure latch).
//!
//! The failure is injected: under unretained command buffers a copy whose
//! source the host released before the commit is one Metal fails with an
//! invalid-resource error. Its own binary because it makes the GPU fail work on
//! purpose, and a fault must not share a process with tests it could disturb.

use lloom_tensor::kernels::metal::commands::CommandBufferKind;
use lloom_tensor::kernels::metal::completion::CompletionWait;
use lloom_tensor::kernels::metal::{Buffer, CommandMarker, Commands, Device, ResidencySet};
use lloom_tensor::kernels::{call_affine, BufferOffset, Kernels, MetalKernelError};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn commands(device: &Device, kind: CommandBufferKind) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new_with(queue, residency_set, CompletionWait::Synchronize, kind).unwrap()
}

fn filled(device: &Device, len: usize, byte: u8) -> Buffer {
    let b = device.new_buffer(len, SHARED).unwrap();
    // SAFETY: fresh shared buffer, `len` bytes, no GPU reader.
    unsafe { std::ptr::write_bytes(b.contents(), byte, len) };
    b
}

/// One `affine` (`2x + 1` over 16 f32) through `cmds`, waited by
/// `flush_and_wait_current`: a later, healthy command buffer whose own wait
/// is clean.
fn affine_then_wait(device: &Device, cmds: &Commands) -> Result<Vec<f32>, MetalKernelError> {
    let kernels = Kernels::new();
    let n = 16usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let len = std::mem::size_of_val(&input[..]);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, len, SHARED)
        .unwrap();
    let dst = device.new_buffer(len, SHARED).unwrap();
    {
        let encoder = cmds.command_encoder()?;
        call_affine(
            device,
            &encoder,
            &kernels,
            "affine_f32",
            std::mem::size_of::<f32>(),
            n,
            BufferOffset::zero_offset(&src),
            &dst,
            2.0,
            1.0,
        )?;
    }
    cmds.flush_and_wait_current()?;
    // SAFETY: shared, `n` f32, waited above.
    Ok(unsafe { std::slice::from_raw_parts(dst.contents() as *const f32, n) }.to_vec())
}

/// Encode a blit copy whose source the host releases before the command
/// buffer is committed, and commit it. Returns the marker for that buffer —
/// or the error its commit returned, when Metal reported the failed copy
/// before the queue registered its next buffer: a queue with a native error
/// takes no new work, so the failure can surface at its own commit.
fn commit_a_copy_of_a_released_buffer(
    device: &Device,
    cmds: &Commands,
    len: usize,
) -> Result<CommandMarker, MetalKernelError> {
    let src = filled(device, len, 0x5a);
    let dst = filled(device, len, 0);
    {
        let mut blit = cmds.blit_command_encoder().unwrap();
        blit.copy_from_buffer(&src, 0, &dst, 0, len);
    }
    drop(src);
    cmds.flush_marker()
}

/// The queue's latch reports a failure: set by the failed buffer's
/// completion, which can trail the error a producer saw first.
fn latched(cmds: &Commands) -> bool {
    let start = std::time::Instant::now();
    while cmds.check_failed().is_ok() {
        if start.elapsed() > std::time::Duration::from_secs(5) {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}

/// **The injected failure.** Under the unretained kind, a command buffer whose
/// copy reads a buffer the host released before the commit is one Metal fails
/// with an invalid-resource error. The wait that follows names a LATER,
/// healthy buffer, which completes cleanly; it must fail exactly when the
/// earlier one did, and every wait and commit after it must fail too.
///
/// Metal does not fail the released read every time: alone in a process it
/// failed 4 of 4 copies, beside other GPU tests most completed. So the
/// assertion is per attempt — the wait's verdict equals the earlier buffer's
/// own status — over attempts until one failed, and a run in which Metal
/// never failed one is reported, not passed.
///
/// When Metal reports the failure before the queue has registered its next
/// buffer, the commit of the failed buffer returns it (`docs/tuning.md`
/// §13.5x): the run fails at the earliest point a host can see the failure.
/// That attempt must carry the injected error, and the latch must then fail
/// every later commit and wait, as on the wait's path.
#[test]
fn a_failed_command_buffer_fails_the_next_wait_whichever_buffer_it_names() {
    let device = Device::system_default().expect("no Metal device");
    let len = 64 << 20;
    let mut injected = 0;
    for attempt in 0..16 {
        let cmds = commands(&device, CommandBufferKind::Unretained);
        let first = match commit_a_copy_of_a_released_buffer(&device, &cmds, len) {
            Ok(first) => first,
            Err(error) => {
                let msg = error.to_string();
                assert!(
                    msg.contains("Invalid Resource"),
                    "attempt {attempt}: the commit failed, and not with the injected error: {msg}"
                );
                assert!(
                    latched(&cmds),
                    "attempt {attempt}: the failure never latched"
                );
                assert!(affine_then_wait(&device, &cmds).is_err());
                assert!(cmds.flush_and_wait().is_err());
                println!("INJECTED attempt={attempt} surfaced=commit");
                injected += 1;
                if injected == 2 {
                    break;
                }
                continue;
            }
        };
        let waited = affine_then_wait(&device, &cmds);
        let first_failed = cmds.marker_failed(&first);
        assert_eq!(
            waited.is_err(),
            first_failed,
            "attempt {attempt}: the wait on a healthy later buffer returned {waited:?} while \
             the earlier buffer {} failed",
            if first_failed { "had" } else { "had not" }
        );
        if first_failed {
            let msg = waited.unwrap_err().to_string();
            assert!(msg.contains("Invalid Resource"), "{msg}");
            // Latched: the queue reports it at every later commit and wait.
            assert!(cmds.check_failed().is_err());
            assert!(affine_then_wait(&device, &cmds).is_err());
            assert!(cmds.flush_and_wait().is_err());
            println!("INJECTED attempt={attempt} surfaced=wait");
            injected += 1;
            if injected == 2 {
                break;
            }
        }
    }
    assert!(
        injected > 0,
        "Metal failed none of the 16 injected command buffers: the test injected nothing"
    );

    // The retaining kind is the control: the command buffer holds the source
    // itself, so the same sequence is clean and the copy lands.
    let cmds = commands(&device, CommandBufferKind::Retaining);
    let first = commit_a_copy_of_a_released_buffer(&device, &cmds, len)
        .expect("the retaining kind holds the source");
    let out = affine_then_wait(&device, &cmds).expect("the retaining kind holds the source");
    assert_eq!(out[3], 7.0);
    assert!(!cmds.marker_failed(&first));
}
