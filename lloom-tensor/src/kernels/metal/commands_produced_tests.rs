//! Work held behind a host-produced input (`commands_produced.rs`): committed
//! only once the producer raises it, on both command APIs. The held-compute
//! gate: a 6 s producer, the dependent compute and the four submissions behind
//! it never reach the GPU before the signal, and complete correctly after it.
//! Run on each API: `LLOOM_METAL_API=metal3|metal4`. The GPU-wait arm
//! (`LLOOM_HOST_INPUT=gpu-wait`, the eager commit with a GPU wait) is the
//! mutant these tests exist to kill.
use super::*;
use objc2_metal::{MTLDevice, MTLSharedEvent};
use std::time::{Duration, Instant};

type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;

/// An event a producer thread raises.
struct Shared(Event);
// SAFETY: an `MTLSharedEvent` is usable from any thread; it is what it is for.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

struct Fixture {
    native: Device,
    commands: Arc<Commands>,
    src: Buffer,
    dst: Buffer,
    gate: Event,
    progress: Event,
}

const ELEMENTS: usize = 16;
const WARM: u8 = 0x31;

/// A standalone queue on the process's API and host-input arm, its
/// destination warmed to a known byte through a drained submission.
fn fixture() -> Fixture {
    let native = Device::system_default().unwrap();
    let queue = native.new_command_queue().unwrap();
    let residency = Arc::new(ResidencySet::new(&native));
    let commands = Arc::new(Commands::new(queue, residency).unwrap());
    if let Ok(requested) = std::env::var("LLOOM_METAL_API") {
        if requested != "auto" {
            assert_eq!(commands.actual_command_api(), requested);
        }
    }
    let input = [3.0f32; ELEMENTS];
    let src = native
        .new_buffer_with_data(
            input.as_ptr().cast(),
            std::mem::size_of_val(&input),
            super::super::MTLResourceOptions::StorageModeShared,
        )
        .unwrap();
    let dst = native
        .new_buffer(4096, super::super::MTLResourceOptions::StorageModeShared)
        .unwrap();
    {
        let mut blit = commands.blit_command_encoder().unwrap();
        // SAFETY: the fixture owns this Shared allocation through every drain.
        unsafe { blit.fill_buffer(&dst, (0, 4096), WARM) };
    }
    commands.flush_and_wait().unwrap();
    assert!(dst_bytes(&dst).iter().all(|b| *b == WARM));
    let gate = native.as_ref().newSharedEvent().unwrap();
    let progress = native.as_ref().newSharedEvent().unwrap();
    // One recording for the held work: no policy cut inside it.
    commands.set_compute_per_buffer(usize::MAX);
    eprintln!(
        "HELD_SETUP actual_api={} host_input={}",
        commands.actual_command_api(),
        commands.host_input_arm().as_str()
    );
    Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    }
}

fn dst_bytes(dst: &Buffer) -> &[u8] {
    // SAFETY: a CPU-mapped Shared buffer of 4096 bytes; read only while no
    // submission writing it can be running (the callers' sequencing).
    unsafe { std::slice::from_raw_parts(dst.contents(), 4096) }
}

/// The dependent compute: `dst[i] = 2 * src[i] + 1 = 7`, then a progress
/// signal behind it on the same stream.
fn encode_dependent(f: &Fixture) {
    let kernels = crate::kernels::Kernels::new();
    {
        let encoder = f.commands.command_encoder().unwrap();
        // SAFETY: the fixture retains source and destination through every
        // wait; nothing reads the destination before its writer completed.
        unsafe {
            crate::kernels::call_affine(
                &f.native,
                &encoder,
                &kernels,
                "affine_f32",
                std::mem::size_of::<f32>(),
                ELEMENTS,
                crate::kernels::BufferOffset::zero_offset(&f.src),
                &f.dst,
                2.0,
                1.0,
            )
            .unwrap();
        }
    }
    f.commands.encode_signal_event(&f.progress, 1).unwrap();
}

fn computed(dst: &Buffer) -> bool {
    // SAFETY: as `dst_bytes`.
    let values = unsafe { std::slice::from_raw_parts(dst.contents().cast::<f32>(), ELEMENTS) };
    values.iter().all(|v| v.to_bits() == 7.0f32.to_bits())
}

/// Metal 4 cuts at a signal (`encode_signal_event`), Metal 3 records it in
/// the open buffer: one more held recording behind a signal on Metal 4.
fn signal_cut(commands: &Commands) -> usize {
    usize::from(commands.actual_command_api() == "metal4")
}

fn native_calls(commands: &Commands) -> usize {
    commands
        .core
        .submission_control
        .native_calls
        .load(Ordering::Relaxed)
}

/// Every slot of a Metal 4 queue is back in its pool or the one recording
/// current: none is stranded by a held or cancelled recording. Waits for the
/// host retirement service to discard abandoned slots.
fn metal4_slots_settle(commands: &Commands) -> Option<(usize, usize)> {
    let queue = commands.core.modern_queue.as_ref()?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (allocated, available) = queue.slots_for_test();
        if allocated == available + 1 || Instant::now() > deadline {
            return Some((allocated, available));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The held-compute gate, rewritten for the contract: the producer raises the
/// gate 6 s after the dependent compute was recorded, with four submissions
/// cut behind it. Through the whole hold nothing behind the gate reaches the
/// GPU — no native commit, no progress signal, the destination untouched —
/// and after the release the output is right with no timeout or error.
#[test]
fn held_compute_and_following_submissions_wait_for_their_producer() {
    const HOLD_MS: u64 = 6000;
    const FOLLOWERS: usize = 4;
    let f = fixture();
    let calls_before = native_calls(&f.commands);
    let held_epoch = f.commands.clock().current_epoch();
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    for _ in 0..FOLLOWERS {
        f.commands.flush_marker().unwrap();
    }
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let producer = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            go_rx.recv().unwrap();
            gate.0.setSignaledValue(1);
            commands.produced(&gate.0, 1).unwrap()
        })
    };
    let held_at = Instant::now();
    let deadline = held_at + Duration::from_millis(HOLD_MS);
    let mut ran_early = None;
    while Instant::now() < deadline {
        if f.progress.signaledValue() != 0 && ran_early.is_none() {
            ran_early = Some(held_at.elapsed());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let observed = held_at.elapsed();
    let stats = f.commands.produced_stats();
    let calls_during = native_calls(&f.commands) - calls_before;
    let dst_untouched = dst_bytes(&f.dst).iter().all(|b| *b == WARM);
    let held_complete = f.commands.clock().is_complete(held_epoch);
    eprintln!(
        "HELD_HOLD hold_requested_ms={HOLD_MS} observed_ms={} progress={} ran_early_ms={:?} \
         native_calls_during_hold={calls_during} dst_untouched={dst_untouched} \
         held_epoch={} held_complete={held_complete} queue_failure={:?} stats={stats:?}",
        observed.as_millis(),
        f.progress.signaledValue(),
        ran_early.map(|d| d.as_millis()),
        held_epoch.0,
        f.commands.check_failed().err().map(|e| e.to_string()),
    );
    assert_eq!(
        ran_early, None,
        "held work passed its producer: progress signalled before the gate"
    );
    assert_eq!(
        calls_during, 0,
        "held work reached the GPU before its input"
    );
    assert!(dst_untouched, "the dependent compute ran before its input");
    assert!(!held_complete, "the held epoch completed before its input");
    let held = FOLLOWERS + signal_cut(&f.commands);
    assert_eq!(stats.held_now, held, "{stats:?}");
    go_tx.send(()).unwrap();
    let released = producer.join().unwrap();
    assert_eq!(released, held, "the producer released the held prefix");
    f.commands.flush_and_wait().unwrap();
    assert!(
        f.progress.waitUntilSignaledValue_timeoutMS(1, 2000),
        "progress did not follow the release"
    );
    assert!(
        computed(&f.dst),
        "the dependent compute read the wrong input"
    );
    assert!(f.commands.clock().is_complete(held_epoch));
    assert_eq!(
        f.commands.clock().completed_count().0,
        f.commands.clock().current_epoch().0,
        "every epoch taken completed"
    );
    f.commands.check_failed().unwrap();
    let stats = f.commands.produced_stats();
    assert_eq!(stats.held_now, 0, "{stats:?}");
    assert_eq!(stats.released, held as u64, "{stats:?}");
    eprintln!(
        "HELD_RELEASED actual_api={} host_input={} elements={ELEMENTS} expected=7 bits_equal=true \
         native_error=None stats={stats:?}",
        f.commands.actual_command_api(),
        f.commands.host_input_arm().as_str()
    );
    if let Some((allocated, available)) = metal4_slots_settle(&f.commands) {
        assert_eq!(allocated, available + 1, "a Metal 4 slot was stranded");
    }
}

/// Cancelled mid-hold: none of the held work runs, the stream stays usable,
/// every epoch completes, no Metal 4 slot is stranded, and the producer's
/// late signal after the cancellation commits nothing (counted, reported
/// once).
#[test]
fn a_cancelled_hold_runs_nothing_leaks_nothing_and_ignores_a_late_input() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE cancellation holds nothing on the gpu-wait arm");
        return;
    }
    let calls_before = native_calls(&f.commands);
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let held_marker = f.commands.flush_marker().unwrap();
    f.commands.flush_marker().unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let held = 2 + signal_cut(&f.commands);
    assert_eq!(f.commands.produced_stats().held_now, held);
    let cancelled = f
        .commands
        .cancel_produced(&f.gate, "test: the producer will never run")
        .unwrap();
    // The held recordings and the fresh current behind them.
    assert_eq!(cancelled, held + 1);
    let stats = f.commands.produced_stats();
    assert_eq!(stats.held_now, 0, "{stats:?}");
    // A wait on a cancelled recording fails, saying why.
    let waited = f.commands.wait_marker(&held_marker);
    assert!(
        waited
            .as_ref()
            .is_err_and(|e| e.to_string().contains("the producer will never run")),
        "{waited:?}"
    );
    f.commands.flush_and_wait().unwrap();
    // Nothing that was held ran.
    std::thread::sleep(Duration::from_millis(1000));
    assert_eq!(f.progress.signaledValue(), 0, "cancelled work ran");
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "cancelled work ran"
    );
    let calls_after_cancel = native_calls(&f.commands);
    // The late input: nothing to commit, counted.
    for _ in 0..2 {
        f.gate.setSignaledValue(1);
        assert_eq!(f.commands.produced(&f.gate, 1).unwrap(), 0);
    }
    assert_eq!(native_calls(&f.commands), calls_after_cancel);
    let stats = f.commands.produced_stats();
    assert_eq!(stats.late_after_cancel, 2, "{stats:?}");
    assert!(f.commands.core.produced.late_logged_for_test());
    // The stream goes on: fresh work on the same queue computes, and the
    // epochs the cancelled recordings took complete with it.
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst));
    assert!(native_calls(&f.commands) > calls_before);
    assert_eq!(
        f.commands.clock().completed_count().0,
        f.commands.clock().current_epoch().0,
        "an epoch a cancelled recording took never completed"
    );
    f.commands.check_failed().unwrap();
    if let Some((allocated, available)) = metal4_slots_settle(&f.commands) {
        assert_eq!(
            allocated,
            available + 1,
            "a cancelled recording stranded its slot"
        );
    }
    eprintln!(
        "HELD_CANCELLED actual_api={} cancelled={cancelled} stats={:?}",
        f.commands.actual_command_api(),
        f.commands.produced_stats()
    );
}

/// Two inputs produced in reverse order: nothing behind the first commits on
/// the second's signal, and both commit in queue order on the first's.
#[test]
fn inputs_produced_out_of_order_commit_in_queue_order() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE ordering of held commits on the gpu-wait arm");
        return;
    }
    let second = f.native.as_ref().newSharedEvent().unwrap();
    let calls_before = native_calls(&f.commands);
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    f.commands.flush_marker().unwrap();
    f.commands.commit_when_produced(&second, 1).unwrap();
    {
        let mut blit = f.commands.blit_command_encoder().unwrap();
        // SAFETY: as the fixture's fill.
        unsafe { blit.fill_buffer(&f.dst, (2048, 4096), 0x5c) };
    }
    f.commands.flush_marker().unwrap();
    let held = 2 + signal_cut(&f.commands);
    assert_eq!(f.commands.produced_stats().held_now, held);
    second.setSignaledValue(1);
    assert_eq!(f.commands.produced(&second, 1).unwrap(), 0);
    assert_eq!(native_calls(&f.commands), calls_before);
    f.gate.setSignaledValue(1);
    assert_eq!(f.commands.produced(&f.gate, 1).unwrap(), held);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst));
    assert!(dst_bytes(&f.dst)[2048..].iter().all(|b| *b == 0x5c));
    assert_eq!(
        f.commands.clock().completed_count().0,
        f.commands.clock().current_epoch().0
    );
}

/// More cuts behind an unproduced input than a Metal 4 queue has slots, and
/// than the held queue holds: the owner parks holding the state lock, the
/// producer's release wakes it, and it submits the released work itself.
#[test]
fn a_hold_deeper_than_the_slots_drains_from_the_parked_owner() {
    const CUTS: usize = produced::HELD_CAP + 4;
    let f = fixture();
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let producer = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            std::thread::sleep(Duration::from_millis(300));
            gate.0.setSignaledValue(1);
            commands.produced(&gate.0, 1)
        })
    };
    for _ in 0..CUTS {
        f.commands.flush_marker().unwrap();
    }
    producer.join().unwrap().unwrap();
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst));
    let stats = f.commands.produced_stats();
    eprintln!(
        "HELD_DEEP actual_api={} cuts={CUTS} stats={stats:?}",
        f.commands.actual_command_api()
    );
    // The park this test exists for engaged: a slot (Metal 4) or the cap.
    assert!(stats.parks > 0, "{stats:?}");
    assert_eq!(stats.held_now, 0);
    // Every held recording took its after-list from the spares and gave it
    // back, storage kept: the next held IO attach allocates nothing.
    let lanes = crate::kernels::metal::io_queue::io_lanes() + 1;
    let state = f.commands.lock_state().unwrap();
    assert_eq!(state.spare_after.len(), produced::HELD_CAP + 1);
    assert!(state.spare_after.iter().all(|list| list.is_empty() && list.capacity() >= lanes));
    drop(state);
    assert_eq!(
        f.commands.clock().completed_count().0,
        f.commands.clock().current_epoch().0
    );
}

/// An input produced before its consumer is recorded holds nothing.
#[test]
fn an_input_already_produced_holds_nothing() {
    let f = fixture();
    f.gate.setSignaledValue(1);
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let marker = f.commands.flush_marker().unwrap();
    assert_eq!(f.commands.produced_stats().held, 0);
    f.commands.wait_marker(&marker).unwrap();
    assert!(computed(&f.dst));
}
