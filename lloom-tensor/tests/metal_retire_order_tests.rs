#![cfg(feature = "metal")]
//! The premise `GpuClock` rests on: command buffers on one queue **retire** in
//! commit order even when the GPU finishes their work out of order (§15.1m).
//!
//! The pool stamps a released buffer with the open epoch and reuses it once the
//! clock says that epoch completed; the clock takes the highest completion
//! report. That is sound only if a later buffer is never reported complete
//! while an earlier one still runs. The GPU does not order the two — a later
//! buffer sharing nothing with an earlier one runs beside it and can end first
//! — so the property is about retirement, and it is Metal's, not ours. This
//! pins it.
//!
//! Its own binary rather than a unit test: it issues tens of thousands of
//! dispatches, and the arena's dispatch counter is process-wide, so beside the
//! arena's unit tests it would move their positions.

use lloom_tensor::kernels::metal::{Buffer, CommandBuffer, ComputeCommandEncoder, Device};
use lloom_tensor::kernels::{
    call_unary_contiguous, unary, BufferOffset, Kernels, RESOURCE_OPTIONS,
};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// A heavy buffer (a serial chain of tiny dependent dispatches, so most cores
/// stay idle) and then a tiny unrelated one. In every trial the tiny one must
/// not be `Completed`, nor its handler run, before the heavy one; and in at
/// least one trial the GPU must have finished the tiny one first, or the two
/// never overlapped and the retirement assertions tested nothing.
#[test]
fn command_buffers_retire_in_commit_order_when_the_gpu_finishes_them_out_of_order() {
    let dev = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new();
    let queue = dev.new_command_queue().expect("queue");
    let buf = || dev.new_buffer(4096, RESOURCE_OPTIONS).unwrap();
    let (ping, pong, small_in, small_out) = (buf(), buf(), buf(), buf());
    let cos = |enc: &ComputeCommandEncoder, input: &Buffer, output: &Buffer| {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_unary_contiguous(
                &dev,
                enc,
                &kernels,
                unary::contiguous::cos::FLOAT,
                4,
                256,
                BufferOffset {
                    buffer: input,
                    offset_in_bytes: 0,
                },
                output,
            )
        }
        .unwrap();
    };

    let (trials, mut gpu_inverted) = (8, 0u32);
    for trial in 0..trials {
        let heavy = CommandBuffer::new(queue.commandBuffer().unwrap());
        let tiny = CommandBuffer::new(queue.commandBuffer().unwrap());
        {
            let enc = heavy.compute_command_encoder_no_fence().unwrap();
            for k in 0..4000 {
                let (a, b) = if k % 2 == 0 {
                    (&ping, &pong)
                } else {
                    (&pong, &ping)
                };
                cos(&enc, a, b);
            }
            enc.end_encoding().unwrap();
        }
        {
            let enc = tiny.compute_command_encoder_no_fence().unwrap();
            cos(&enc, &small_in, &small_out);
            enc.end_encoding().unwrap();
        }
        let heavy_done_at_tiny_handler = Arc::new(AtomicBool::new(false));
        let handlers = Arc::new(AtomicU32::new(0));
        {
            let h = Arc::clone(&handlers);
            heavy
                .on_completion(move || {
                    h.fetch_add(1, Ordering::AcqRel);
                })
                .unwrap();
        }
        {
            let (h, seen, heavy) = (
                Arc::clone(&handlers),
                Arc::clone(&heavy_done_at_tiny_handler),
                heavy.clone(),
            );
            tiny.on_completion(move || {
                seen.store(
                    heavy.status() == MTLCommandBufferStatus::Completed,
                    Ordering::Release,
                );
                h.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
        }
        heavy.commit().unwrap();
        tiny.commit().unwrap();
        tiny.wait_until_completed().unwrap();
        let heavy_status_at_tiny_wait = heavy.status();
        heavy.wait_until_completed().unwrap();
        while handlers.load(Ordering::Acquire) < 2 {
            std::hint::spin_loop();
        }
        assert_eq!(
            heavy_status_at_tiny_wait,
            MTLCommandBufferStatus::Completed,
            "trial {trial}: the later buffer retired while the earlier one still ran"
        );
        assert!(
            heavy_done_at_tiny_handler.load(Ordering::Acquire),
            "trial {trial}: the later buffer's handler ran before the earlier one completed"
        );
        let heavy_end = heavy.gpu_end_time();
        let tiny_end = tiny.gpu_end_time();
        gpu_inverted += u32::from(tiny_end < heavy_end);
    }
    eprintln!("RETIREORDER trials={trials} tiny_gpu_end_first={gpu_inverted}");
    assert!(
        gpu_inverted > 0,
        "the GPU never finished the tiny buffer first: the two were not overlapped, \
         so the retirement-order assertions above tested nothing"
    );
}
