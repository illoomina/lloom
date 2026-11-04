//! Work held behind a host-produced input (`commands_produced.rs`): committed
//! only once the producer raises it, on both command APIs. The held-compute
//! gate: a 6 s producer, the dependent compute and the four submissions behind
//! it never reach the GPU before the signal, and complete correctly after it.
//! Run on each API: `LLOOM_METAL_API=metal3|metal4`. The GPU-wait arm
//! (`LLOOM_HOST_INPUT=gpu-wait`, the eager commit with a GPU wait) is the
//! mutant these tests exist to kill.
use super::*;
use objc2_metal::{MTLDevice, MTLSharedEvent};
use std::ptr::NonNull;
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
    fixture_with(|_| {})
}

/// As [`fixture`], with `setup` run on the queue before its first submission
/// seals it (a completion subscriber).
fn fixture_with(setup: impl FnOnce(&Commands)) -> Fixture {
    let native = Device::system_default().unwrap();
    let queue = native.new_command_queue().unwrap();
    let residency = Arc::new(ResidencySet::new(&native));
    let commands = Arc::new(Commands::new(queue, residency).unwrap());
    setup(&commands);
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
        unsafe { blit.fill_buffer(&dst, (0, 4096), WARM) }.unwrap();
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
    encode_compute(f);
    f.commands.encode_signal_event(&f.progress, 1).unwrap();
}

/// The dependent compute alone, with no signal (a signal may cut on Metal 4).
fn encode_compute(f: &Fixture) {
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
        unsafe { blit.fill_buffer(&f.dst, (2048, 2048), 0x5c) }.unwrap();
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
    assert!(state
        .spare_after
        .iter()
        .all(|list| list.is_empty() && list.capacity() >= lanes));
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

/// How the inputs a parked owner waits for reach the queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Delivery {
    /// The IO path's: each event's notification, on the queue's one
    /// `MTLSharedEventListener` (`Commands::notify_produced_at`).
    Listener,
    /// One host thread raises the second input, calls `produced`, then the
    /// first: a fill or fetch worker producing in order.
    OneThread,
    /// A host thread per input. The control: no delivery can wait behind
    /// another's.
    ThreadEach,
}

/// Where the inputs sit in the held queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// One held recording reads both inputs: a point-of-use batch over two IO
    /// lanes (`IoQueue::load_ordered`).
    Lanes,
    /// The head reads the first input and a later recording the second,
    /// produced first: concurrent IO completing later-before-earlier.
    LaterFirst,
}

/// When the first input is produced, relative to the parked owner's look
/// after the second input woke it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum When {
    /// After the owner has looked and parked again.
    AfterPark,
    /// Before the owner's wake snapshot (`schedule-hooks` only).
    #[cfg_attr(not(feature = "schedule-hooks"), allow(dead_code))]
    BeforeSnapshot,
    /// After the owner's look, before it parks (`schedule-hooks` only).
    #[cfg_attr(not(feature = "schedule-hooks"), allow(dead_code))]
    BeforePark,
}

/// The owner is parked behind held work: for a slot on Metal 4, at the held
/// queue's cap on Metal 3.
fn owner_parked(commands: &Commands) -> bool {
    match commands.core.modern_queue.as_ref() {
        Some(queue) => queue.ready_parked_for_test(),
        None => commands.core.produced.wake.test_is_parked(),
    }
}

/// The owner: cuts past every Metal 4 slot and past the held queue's cap,
/// then reports how its cuts ended.
fn spawn_owner(
    commands: &Arc<Commands>,
) -> (
    std::thread::JoinHandle<()>,
    std::sync::mpsc::Receiver<Result<(), String>>,
) {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let commands = Arc::clone(commands);
    let owner = std::thread::spawn(move || {
        #[cfg(feature = "schedule-hooks")]
        position::this_thread_is_the_owner();
        let mut result = Ok(());
        for _ in 0..produced::HELD_CAP + 2 {
            if let Err(error) = commands.flush_marker() {
                result = Err(error.to_string());
                break;
            }
        }
        let _ = done_tx.send(result);
    });
    (owner, done_rx)
}

/// The owner is parked with nothing but a producer or a canceller able to
/// move it: every Metal 4 slot held by recordings only a producer releases,
/// or the held queue at its cap on Metal 3.
fn stuck(commands: &Commands) -> bool {
    owner_parked(commands)
        && match commands.core.modern_queue.as_ref() {
            Some(queue) => {
                let (allocated, available) = queue.slots_for_test();
                available == 0 && commands.produced_stats().held_now + 1 == allocated
            }
            None => commands.produced_stats().held_now == produced::HELD_CAP,
        }
}

/// Poll `ready` until it holds or `within` passes.
fn until(within: Duration, mut ready: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + within;
    while !ready() {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

#[cfg(feature = "schedule-hooks")]
mod position {
    //! Raise the first input from the owner thread itself, at one schedule
    //! point of the park that follows the second input's wake.
    use super::{Event, Shared, When};
    use crate::schedule_hook::Site;
    use objc2_metal::MTLSharedEvent;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU8, Ordering};

    thread_local! {
        static OWNER: Cell<bool> = const { Cell::new(false) };
    }
    /// 0 idle, 1 the second input raised, 2 the first raised by the hook.
    pub(super) static PHASE: AtomicU8 = AtomicU8::new(0);
    static AT: AtomicU8 = AtomicU8::new(0);
    static FIRST: parking_lot::Mutex<Option<Shared>> = parking_lot::Mutex::new(None);

    fn hook(site: Site) {
        if !OWNER.with(Cell::get) || PHASE.load(Ordering::Acquire) != 1 {
            return;
        }
        let at = match site {
            Site::WakeParkReturned => When::BeforeSnapshot as u8,
            Site::WakeRegistered => When::BeforePark as u8,
            _ => return,
        };
        if AT.load(Ordering::Acquire) == at {
            if let Some(first) = FIRST.lock().as_ref() {
                first.0.setSignaledValue(1);
            }
            PHASE.store(2, Ordering::Release);
        }
    }

    pub(super) fn arm(when: When, first: &Event) {
        AT.store(when as u8, Ordering::Release);
        *FIRST.lock() = Some(Shared(first.clone()));
        crate::schedule_hook::install(Some(hook));
    }

    pub(super) fn this_thread_is_the_owner() {
        OWNER.with(|owner| owner.set(true));
    }
}

/// Park the owner behind two inputs with every way forward held — every
/// Metal 4 slot held by recordings only a producer can release, or the held
/// queue at its cap on Metal 3 — then produce the second input, and the first
/// once the owner has woken, looked and parked again. The owner must wake for
/// the first and finish: a hang is the defect, bounded here by the test's
/// own deadline, never by the code under test.
fn a_parked_owner_wakes_for_every_input(shape: Shape, delivery: Delivery, when: When) {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let first = f.gate.clone();
    let second = f.native.as_ref().newSharedEvent().unwrap();
    let listen = |event: &Event| {
        if delivery == Delivery::Listener {
            f.commands.notify_produced_at(event, 1).unwrap();
        }
    };
    match shape {
        Shape::Lanes => {
            f.commands.commit_when_produced(&first, 1).unwrap();
            f.commands.commit_when_produced(&second, 1).unwrap();
            listen(&first);
            listen(&second);
            encode_dependent(&f);
        }
        Shape::LaterFirst => {
            f.commands.commit_when_produced(&first, 1).unwrap();
            listen(&first);
            encode_dependent(&f);
            f.commands.commit_when_produced(&second, 1).unwrap();
            listen(&second);
            let mut blit = f.commands.blit_command_encoder().unwrap();
            // SAFETY: as the fixture's fill.
            unsafe { blit.fill_buffer(&f.dst, (2048, 2048), 0x5c) }.unwrap();
        }
    }
    let (owner, done_rx) = spawn_owner(&f.commands);
    assert!(
        until(Duration::from_secs(10), || stuck(&f.commands)),
        "the owner never parked behind the held work: {:?}",
        f.commands.produced_stats()
    );
    let parks = f.commands.produced_stats().parks;
    #[cfg(feature = "schedule-hooks")]
    if when != When::AfterPark {
        position::arm(when, &first);
        position::PHASE.store(1, Ordering::Release);
    }
    #[cfg(not(feature = "schedule-hooks"))]
    assert_eq!(when, When::AfterPark, "positions need schedule-hooks");
    let produce = |event: &Event| {
        let event = Shared(event.clone());
        let commands = Arc::clone(&f.commands);
        move || {
            let event = event;
            event.0.setSignaledValue(1);
            commands.produced(&event.0, 1).map_err(|e| e.to_string())
        }
    };
    let mut producers = Vec::new();
    match delivery {
        Delivery::Listener => second.setSignaledValue(1),
        Delivery::OneThread => {
            let produce_second = produce(&second);
            let produce_first = produce(&first);
            producers.push(std::thread::spawn(move || {
                produce_second()?;
                produce_first()
            }));
        }
        Delivery::ThreadEach => producers.push(std::thread::spawn(produce(&second))),
    }
    if delivery != Delivery::OneThread && when == When::AfterPark {
        // The second input woke the owner; it looked, and parked again.
        assert!(
            until(Duration::from_secs(10), || {
                f.commands.produced_stats().parks > parks && owner_parked(&f.commands)
            }),
            "the second input never woke the owner: {:?}",
            f.commands.produced_stats()
        );
        match delivery {
            Delivery::Listener => first.setSignaledValue(1),
            Delivery::ThreadEach => producers.push(std::thread::spawn(produce(&first))),
            Delivery::OneThread => unreachable!(),
        }
    }
    let finished = done_rx.recv_timeout(Duration::from_secs(20));
    let stats = f.commands.produced_stats();
    eprintln!(
        "PARKED_OWNER api={} shape={shape:?} delivery={delivery:?} when={when:?} \
         finished={} first={} second={} owner_parked={} stats={stats:?}",
        f.commands.actual_command_api(),
        finished.is_ok(),
        first.signaledValue(),
        second.signaledValue(),
        owner_parked(&f.commands),
    );
    assert!(
        finished.is_ok(),
        "HANG: the owner parked behind held work was never woken for the first \
         input (shape={shape:?} delivery={delivery:?} when={when:?}): {stats:?}"
    );
    finished.unwrap().unwrap();
    owner.join().unwrap();
    for producer in producers {
        producer.join().unwrap().unwrap();
    }
    f.commands.flush_and_wait().unwrap();
    assert!(
        computed(&f.dst),
        "the dependent compute read the wrong input"
    );
    if shape == Shape::LaterFirst {
        assert!(dst_bytes(&f.dst)[2048..].iter().all(|b| *b == 0x5c));
    }
    f.commands.check_failed().unwrap();
    let stats = f.commands.produced_stats();
    assert_eq!(stats.held_now, 0, "{stats:?}");
    assert!(
        stats.delegated >= 1,
        "no producer found the parked owner's lock held: the schedule was not reached: {stats:?}"
    );
    assert_eq!(
        f.commands.clock().completed_count().0,
        f.commands.clock().current_epoch().0,
        "every epoch taken completed"
    );
    if let Some((allocated, available)) = metal4_slots_settle(&f.commands) {
        assert_eq!(allocated, available + 1, "a Metal 4 slot was stranded");
    }
}

/// The IO path's shape: one recording behind two lanes, the second lane's
/// notification first. Its producer waits for the state lock the parked owner
/// holds, and must not keep the first lane's notification from waking it.
#[test]
fn a_parked_owner_wakes_for_every_io_lane() {
    a_parked_owner_wakes_for_every_input(Shape::Lanes, Delivery::Listener, When::AfterPark);
}

/// Concurrent IO completing later-before-earlier: a later held recording's
/// input lands first, and cannot release anything while the head waits.
#[test]
fn a_parked_owner_wakes_when_the_head_input_lands_last() {
    a_parked_owner_wakes_for_every_input(Shape::LaterFirst, Delivery::Listener, When::AfterPark);
}

/// A host producer that produces in order: its `produced` for the second
/// input must return before the owner releases anything, or it never
/// produces the first.
#[test]
fn a_parked_owner_wakes_for_an_in_order_host_producer() {
    a_parked_owner_wakes_for_every_input(Shape::Lanes, Delivery::OneThread, When::AfterPark);
}

/// The control: one thread per input, so neither delivery can wait behind
/// the other. The wake protocol alone moves the owner.
#[test]
fn a_parked_owner_wakes_for_a_producer_thread_per_input() {
    a_parked_owner_wakes_for_every_input(Shape::Lanes, Delivery::ThreadEach, When::AfterPark);
}

/// The first input produced before the owner's wake snapshot: its look sees
/// it, whatever the notification does.
#[cfg(feature = "schedule-hooks")]
#[test]
fn a_parked_owner_sees_an_input_produced_before_its_snapshot() {
    a_parked_owner_wakes_for_every_input(Shape::Lanes, Delivery::Listener, When::BeforeSnapshot);
}

/// The first input produced after the owner's look, before it parks: only the
/// notification can wake it.
#[cfg(feature = "schedule-hooks")]
#[test]
fn a_parked_owner_wakes_for_an_input_produced_before_it_parks() {
    a_parked_owner_wakes_for_every_input(Shape::Lanes, Delivery::Listener, When::BeforePark);
}

/// A producer that finds the state lock held by an owner that is not parked
/// leaves its release to that owner and returns at once; the owner submits
/// the released work as it unlocks, with no cut or wait of its own.
#[test]
fn a_producer_leaves_its_release_to_the_lock_holder() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    f.commands.flush_marker().unwrap();
    let held = 1 + signal_cut(&f.commands);
    assert_eq!(f.commands.produced_stats().held_now, held);
    let calls_before = native_calls(&f.commands);
    let state = f.commands.lock_state().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let producer = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            gate.0.setSignaledValue(1);
            let _ = tx.send(commands.produced(&gate.0, 1).map_err(|e| e.to_string()));
        })
    };
    let returned = rx.recv_timeout(Duration::from_secs(5));
    let calls_while_held = native_calls(&f.commands) - calls_before;
    let released_while_held = f.commands.produced_stats().released;
    drop(state);
    // Native calls are per segment (a Metal 4 signal ends one), so the
    // recordings are counted by `released`.
    let calls_after_unlock = native_calls(&f.commands) - calls_before;
    let stats = f.commands.produced_stats();
    eprintln!(
        "LEFT_TO_HOLDER api={} returned={returned:?} calls_while_held={calls_while_held} \
         released_while_held={released_while_held} calls_after_unlock={calls_after_unlock} \
         stats={stats:?}",
        f.commands.actual_command_api()
    );
    assert_eq!(
        returned,
        Ok(Ok(0)),
        "the producer waited for the state lock, or released work itself"
    );
    producer.join().unwrap();
    assert_eq!(
        (calls_while_held, released_while_held),
        (0, 0),
        "held work was submitted under another thread's lock"
    );
    assert!(
        calls_after_unlock > 0,
        "the holder did not submit the released work as it unlocked"
    );
    assert_eq!(stats.delegated, 1, "{stats:?}");
    assert_eq!(stats.released_by_producer, 0, "{stats:?}");
    assert_eq!(stats.released, held as u64, "{stats:?}");
    assert_eq!(stats.held_now, 0, "{stats:?}");
    f.commands.flush_and_wait().unwrap();
    assert!(
        computed(&f.dst),
        "the dependent compute read the wrong input"
    );
    f.commands.check_failed().unwrap();
}

/// A cancellation never waits for the state lock either. With the owner
/// parked behind work whose input will never come, the canceller returns at
/// once, the owner applies the cancellation in its park and its cut fails
/// with the reason; nothing held runs, and the queue goes on.
#[test]
fn a_cancel_reaches_an_owner_parked_behind_the_work_it_cancels() {
    const REASON: &str = "test: the producer failed";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let (owner, done) = spawn_owner(&f.commands);
    assert!(
        until(Duration::from_secs(10), || stuck(&f.commands)),
        "the owner never parked behind the held work: {:?}",
        f.commands.produced_stats()
    );
    let held = f.commands.produced_stats().held_now;
    let (tx, rx) = std::sync::mpsc::channel();
    let canceller = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            let _ = tx.send(
                commands
                    .cancel_produced(&gate.0, REASON)
                    .map_err(|e| e.to_string()),
            );
        })
    };
    let returned = rx.recv_timeout(Duration::from_secs(5));
    let finished = done.recv_timeout(Duration::from_secs(20));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_PARKED api={} held={held} returned={returned:?} finished={finished:?} stats={stats:?}",
        f.commands.actual_command_api()
    );
    assert_eq!(
        returned,
        Ok(Ok(0)),
        "HANG or wrong owner: the canceller waited for the lock the parked owner holds, \
         or applied the cancellation itself"
    );
    let finished = finished.expect("HANG: the parked owner never applied the cancellation");
    assert!(
        finished.as_ref().is_err_and(|e| e.contains(REASON)),
        "the parked cut did not fail with the cancellation's reason: {finished:?}"
    );
    owner.join().unwrap();
    canceller.join().unwrap();
    assert!(stats.cancels_delegated >= 1, "{stats:?}");
    assert_eq!(stats.released, 0, "held work ran: {stats:?}");
    assert!(stats.cancelled > held as u64, "{stats:?}");
    assert_eq!(stats.held_now, 0, "{stats:?}");
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(f.progress.signaledValue(), 0, "cancelled work ran");
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "cancelled work ran"
    );
    // The queue goes on: fresh work computes, every epoch taken completes.
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst));
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
}

/// A cancellation that finds the state lock held by an owner that is not
/// parked returns at once; the holder applies it as it unlocks.
#[test]
fn a_cancel_left_to_the_lock_holder_runs_as_it_unlocks() {
    const REASON: &str = "test: cancelled under another thread's lock";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let held_marker = f.commands.flush_marker().unwrap();
    let held = 1 + signal_cut(&f.commands);
    assert_eq!(f.commands.produced_stats().held_now, held);
    let state = f.commands.lock_state().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let canceller = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            let _ = tx.send(
                commands
                    .cancel_produced(&gate.0, REASON)
                    .map_err(|e| e.to_string()),
            );
        })
    };
    let returned = rx.recv_timeout(Duration::from_secs(5));
    let cancelled_while_held = f.commands.produced_stats().cancelled;
    drop(state);
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_LEFT_TO_HOLDER api={} returned={returned:?} cancelled_while_held={cancelled_while_held} stats={stats:?}",
        f.commands.actual_command_api()
    );
    assert_eq!(
        returned,
        Ok(Ok(0)),
        "the canceller waited for the state lock, or applied the cancellation itself"
    );
    canceller.join().unwrap();
    assert_eq!(
        cancelled_while_held, 0,
        "cancelled under another thread's lock"
    );
    // The held recordings and the current one behind them.
    assert_eq!(stats.cancelled, held as u64 + 1, "{stats:?}");
    assert_eq!(stats.cancels_delegated, 1, "{stats:?}");
    assert_eq!(stats.held_now, 0, "{stats:?}");
    let waited = f.commands.wait_marker(&held_marker);
    assert!(
        waited
            .as_ref()
            .is_err_and(|e| e.to_string().contains(REASON)),
        "{waited:?}"
    );
    f.gate.setSignaledValue(1);
    assert_eq!(f.commands.produced(&f.gate, 1).unwrap(), 0);
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(f.progress.signaledValue(), 0, "cancelled work ran");
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst));
    f.commands.check_failed().unwrap();
}

/// A cancellation left to a holder is applied before the raise its caller
/// makes next (`HostFedRing::release` cancels, then raises the event past
/// the run): the raise does not release what the cancellation covers.
#[test]
fn a_cancel_is_applied_before_the_raise_that_follows_it() {
    const REASON: &str = "test: the run was released";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    f.commands.flush_marker().unwrap();
    let held = 1 + signal_cut(&f.commands);
    let state = f.commands.lock_state().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let canceller = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            let cancelled = commands
                .cancel_produced_through(&gate.0, 1, REASON)
                .map_err(|e| e.to_string());
            gate.0.setSignaledValue(1);
            let released = commands.produced(&gate.0, 1).map_err(|e| e.to_string());
            let _ = tx.send((cancelled, released));
        })
    };
    let returned = rx.recv_timeout(Duration::from_secs(5));
    drop(state);
    canceller.join().unwrap();
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_THEN_RAISE api={} returned={returned:?} progress={} stats={stats:?}",
        f.commands.actual_command_api(),
        f.progress.signaledValue()
    );
    assert_eq!(
        returned,
        Ok((Ok(0), Ok(0))),
        "the canceller or the producer waited for the state lock"
    );
    assert_eq!(
        f.progress.signaledValue(),
        0,
        "the raise released cancelled work"
    );
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "the raise released cancelled work"
    );
    assert_eq!(stats.released, 0, "{stats:?}");
    assert_eq!(stats.cancelled, held as u64 + 1, "{stats:?}");
    f.commands.check_failed().unwrap();
}

/// A cancellation through a value covers no input above it, whenever it is
/// applied: a later run's work on the same event is not cancelled.
#[test]
fn a_cancel_through_a_value_spares_a_later_one() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 3).unwrap();
    encode_dependent(&f);
    let state = f.commands.lock_state().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let canceller = {
        let gate = Shared(f.gate.clone());
        let commands = Arc::clone(&f.commands);
        std::thread::spawn(move || {
            let gate = gate;
            let _ = tx.send(
                commands
                    .cancel_produced_through(&gate.0, 2, "test: an earlier run")
                    .map_err(|e| e.to_string()),
            );
        })
    };
    let returned = rx.recv_timeout(Duration::from_secs(5));
    drop(state);
    canceller.join().unwrap();
    f.gate.setSignaledValue(3);
    f.commands.produced(&f.gate, 3).unwrap();
    f.commands.flush_and_wait().unwrap();
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_THROUGH api={} returned={returned:?} stats={stats:?}",
        f.commands.actual_command_api()
    );
    assert_eq!(
        returned,
        Ok(Ok(0)),
        "the canceller waited for the state lock, or applied the cancellation itself"
    );
    assert_eq!(stats.cancelled, 0, "{stats:?}");
    assert!(computed(&f.dst), "a later value's work was cancelled");
    f.commands.check_failed().unwrap();
}

/// Two values of one event in the current recording, nothing held: a
/// cancellation through the lower one takes the recording itself, and
/// raising the event past both then runs nothing.
#[test]
fn a_cancel_through_the_lower_of_two_values_takes_the_current_recording() {
    const REASON: &str = "test: the lower value will never be produced";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    f.commands.commit_when_produced(&f.gate, 5).unwrap();
    encode_compute(&f);
    let gated = f.commands.produced_stats();
    let cancelled = f.commands.cancel_produced_through(&f.gate, 1, REASON);
    f.gate.setSignaledValue(5);
    let released = f.commands.produced(&f.gate, 5);
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_LOWER_CURRENT api={} debug_assertions={} gated={gated:?} cancelled={cancelled:?} released={released:?} stats={stats:?}",
        f.commands.actual_command_api(),
        cfg!(debug_assertions)
    );
    assert_eq!(
        (gated.held, gated.held_now),
        (0, 1),
        "the recording was not isolated: {gated:?}"
    );
    assert_eq!(gated.beside_its_event, 1, "{gated:?}");
    assert_eq!(
        cancelled.map_err(|e| e.to_string()),
        Ok(1),
        "the current recording escaped"
    );
    assert_eq!(released.map_err(|e| e.to_string()), Ok(0));
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "the cancelled recording ran"
    );
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst), "the stream did not go on");
    f.commands.check_failed().unwrap();
}

/// A held recording gated on 5 and then on 1, one event: a cancellation
/// through 1 takes it, whichever order the values came in.
#[test]
fn a_cancel_through_a_lower_value_gated_second_takes_the_held_recording() {
    const REASON: &str = "test: the lower value will never be produced";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 5).unwrap();
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    let held_marker = f.commands.flush_marker().unwrap();
    let held = 1 + signal_cut(&f.commands);
    assert_eq!(f.commands.produced_stats().held_now, held);
    let cancelled = f.commands.cancel_produced_through(&f.gate, 1, REASON);
    f.gate.setSignaledValue(5);
    let released = f.commands.produced(&f.gate, 5);
    let waited = f.commands.wait_marker(&held_marker);
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_DESCENDING_HELD api={} debug_assertions={} cancelled={cancelled:?} released={released:?} waited={waited:?} progress={} stats={stats:?}",
        f.commands.actual_command_api(),
        cfg!(debug_assertions),
        f.progress.signaledValue()
    );
    // The held recordings and the fresh current behind them.
    assert_eq!(cancelled.map_err(|e| e.to_string()), Ok(held + 1));
    assert_eq!(released.map_err(|e| e.to_string()), Ok(0));
    assert!(
        waited
            .as_ref()
            .is_err_and(|e| e.to_string().contains(REASON)),
        "{waited:?}"
    );
    assert_eq!(f.progress.signaledValue(), 0, "the cancelled recording ran");
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "the cancelled recording ran"
    );
    assert_eq!(stats.released, 0, "{stats:?}");
    f.commands.check_failed().unwrap();
}

/// A held recording gated on 1, 3 and 5 of one event, the event since raised
/// to 2: a cancellation through 3 takes it, by its interior value.
#[test]
fn a_cancel_through_an_interior_value_after_partial_production_takes_the_held_recording() {
    const REASON: &str = "test: the interior value will never be produced";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    for value in [1, 3, 5] {
        f.commands.commit_when_produced(&f.gate, value).unwrap();
    }
    encode_dependent(&f);
    f.commands.flush_marker().unwrap();
    let held = 1 + signal_cut(&f.commands);
    f.gate.setSignaledValue(2);
    let partial = f.commands.produced(&f.gate, 2);
    let cancelled = f.commands.cancel_produced_through(&f.gate, 3, REASON);
    f.gate.setSignaledValue(5);
    let released = f.commands.produced(&f.gate, 5);
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_INTERIOR_HELD api={} debug_assertions={} partial={partial:?} cancelled={cancelled:?} released={released:?} progress={} stats={stats:?}",
        f.commands.actual_command_api(),
        cfg!(debug_assertions),
        f.progress.signaledValue()
    );
    assert_eq!(
        partial.map_err(|e| e.to_string()),
        Ok(0),
        "released before 5"
    );
    assert_eq!(cancelled.map_err(|e| e.to_string()), Ok(held + 1));
    assert_eq!(released.map_err(|e| e.to_string()), Ok(0));
    assert_eq!(f.progress.signaledValue(), 0, "the cancelled recording ran");
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "the cancelled recording ran"
    );
    f.commands.check_failed().unwrap();
}

/// A recording held on 5 alone, then one gated on 1 and 5 behind it: a
/// cancellation through 1 takes the second and spares the first, which runs
/// once the event reaches 5.
#[test]
fn a_cancel_through_a_lower_value_spares_a_recording_gated_on_a_later_one_alone() {
    const REASON: &str = "test: the lower value will never be produced";
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 5).unwrap();
    encode_compute(&f);
    f.commands.flush_marker().unwrap();
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    f.commands.commit_when_produced(&f.gate, 5).unwrap();
    f.commands.encode_signal_event(&f.progress, 2).unwrap();
    let before = f.commands.produced_stats();
    let cancelled = f.commands.cancel_produced_through(&f.gate, 1, REASON);
    f.gate.setSignaledValue(5);
    let released = f.commands.produced(&f.gate, 5);
    f.commands.flush_and_wait().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let stats = f.commands.produced_stats();
    eprintln!(
        "CANCEL_SPARES_AHEAD api={} debug_assertions={} before={before:?} cancelled={cancelled:?} released={released:?} progress={} stats={stats:?}",
        f.commands.actual_command_api(),
        cfg!(debug_assertions),
        f.progress.signaledValue()
    );
    assert!(cancelled.as_ref().is_ok_and(|n| *n >= 1), "{cancelled:?}");
    assert_eq!(f.progress.signaledValue(), 0, "the cancelled recording ran");
    assert!(
        computed(&f.dst),
        "the recording gated on 5 alone did not run"
    );
    assert_eq!(stats.released, 1, "{stats:?}");
    f.commands.check_failed().unwrap();
}

/// The platform premise behind the listener arm: an `MTLSharedEventListener`
/// runs its blocks one at a time, so a block that waits holds back every
/// notification queued behind it, on any event.
#[test]
fn an_event_listener_runs_one_block_at_a_time() {
    use std::sync::atomic::AtomicBool;
    let native = Device::system_default().unwrap();
    let listener = objc2_metal::MTLSharedEventListener::new();
    let first = native.as_ref().newSharedEvent().unwrap();
    let second = native.as_ref().newSharedEvent().unwrap();
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let ran = Arc::new(AtomicBool::new(false));
    let blocking = {
        let (entered, release) = (entered.clone(), release.clone());
        block2::RcBlock::new(
            move |_: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
                entered.store(true, Ordering::Release);
                let deadline = Instant::now() + Duration::from_secs(10);
                while !release.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
            },
        )
    };
    let marking = {
        let ran = ran.clone();
        block2::RcBlock::new(
            move |_: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
                ran.store(true, Ordering::Release);
            },
        )
    };
    // SAFETY: the blocks are valid for the calls; Metal copies them.
    unsafe {
        first.notifyListener_atValue_block(&listener, 1, block2::RcBlock::as_ptr(&blocking));
        second.notifyListener_atValue_block(&listener, 1, block2::RcBlock::as_ptr(&marking));
    }
    first.setSignaledValue(1);
    assert!(until(Duration::from_secs(5), || entered.load(Ordering::Acquire)));
    second.setSignaledValue(1);
    let overtook = until(Duration::from_millis(500), || ran.load(Ordering::Acquire));
    release.store(true, Ordering::Release);
    let delivered = until(Duration::from_secs(5), || ran.load(Ordering::Acquire));
    eprintln!(
        "LISTENER_SERIAL second_ran_while_first_blocked={overtook} delivered_after={delivered}"
    );
    assert!(delivered, "the second notification was never delivered");
    assert!(
        !overtook,
        "the listener ran a second block beside a blocked first: the listener \
         arm's reproduction no longer reaches the defect"
    );
}

/// Real Commands boundary, controlled actual-bank terminal metadata. This does
/// NOT exercise native file IO; the native short-read suite remains separate.
#[test]
fn validated_io_failure_cancels_v_plus_one_only_gate_and_recovers_next_generation() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    assert_eq!(f.commands.host_input_arm(), HostInputArm::Commit);
    let mut old = TestIoGeneration::new(f.native.as_ref(), 2);
    let later_only = old.dependency(1);
    f.commands
        .commit_when_io_validated(later_only.clone())
        .unwrap();
    encode_dependent(&f);
    // The gate contains ONLY v+1. No v prerequisite can accidentally make
    // snapshot cancellation appear to cover the failing generation.
    old.terminal(0, 2);
    f.commands
        .core
        .produced_on(later_only.notification(), u64::MAX)
        .unwrap();
    {
        let state = f.commands.core.state.lock(Side::Host);
        assert!(
            state.current_abandoned,
            "IO_GENERATION_FAILURE_COVERS_V_PLUS_ONE"
        );
        assert!(state.held.is_empty(), "IO_FAILURE_NO_DEPENDENT_SUBMISSION");
    }
    assert!(
        f.commands.commit_when_io_validated(later_only).is_err(),
        "IO_FAILED_GENERATION_REFUSES_LATER_ADMISSION"
    );
    assert!(
        f.commands.core.failure.check().is_ok(),
        "IO_FAILURE_NOT_WHOLE_QUEUE_POISON"
    );
    let mut fresh = TestIoGeneration::new(f.native.as_ref(), 1);
    let ready = fresh.dependency(0);
    fresh.terminal(0, 3);
    f.commands.commit_when_io_validated(ready).unwrap();
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst), "IO_NEXT_GENERATION_COMPUTED");
    assert!(
        f.commands.core.failure.check().is_ok(),
        "IO_NEXT_GENERATION_RECOVERS"
    );
}

#[test]
fn validated_io_notification_is_not_data_and_preserves_successful_prefix() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    let mut generation = TestIoGeneration::new(f.native.as_ref(), 2);
    let earlier = generation.dependency(0);
    let later = generation.dependency(1);
    generation.terminal(0, 3);
    generation.terminal(1, 2);
    assert_eq!(later.notification().signaledValue(), u64::MAX);
    assert!(
        earlier.ready(),
        "IO_VALIDATED_PREFIX_SURVIVES_LATER_FAILURE"
    );
    assert!(!later.ready(), "IO_TERMINAL_NOTICE_NOT_DATA_READY");
    assert!(later.failed());
    assert!(
        f.commands.commit_when_io_validated(earlier).is_err(),
        "IO_FAILED_GENERATION_REJECTS_NEW_ADMISSION_EVEN_READY"
    );
}

#[test]
fn enrolled_io_compute_authority_refuses_independent_commands_queue() {
    let a = fixture();
    let b = fixture();
    let authority = a.commands.new_io_compute_authority().unwrap();
    assert!(
        b.commands.signal_enrolled_io(&authority).is_err(),
        "IO_COMPUTE_CERTIFICATE_OWNER"
    );
    assert_eq!(a.commands.signal_enrolled_io(&authority).unwrap(), 1);
    a.commands.flush_and_wait().unwrap();
}

#[test]
fn cancelled_validated_deferred_io_is_refused_not_rehomed() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    assert_eq!(f.commands.host_input_arm(), HostInputArm::Commit);
    let mut generation = TestIoGeneration::new(f.native.as_ref(), 1);
    let dependency = generation.dependency(0);
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_compute(&f);
    f.commands
        .commit_validated_io_after(generation.deferred(0))
        .unwrap();
    assert!(
        !dependency.generation_failed(),
        "IO_DEFERRED_STILL_PREPARED"
    );
    f.commands
        .cancel_produced(&f.gate, "test cancellation before native handoff")
        .unwrap();
    assert!(
        dependency.generation_failed(),
        "IO_DEFERRED_CANCEL_REFUSES_INSTEAD_OF_REHOME"
    );
    assert!(!dependency.ready(), "IO_REFUSAL_IS_NOT_DATA");
    assert!(
        f.commands.core.failure.check().is_ok(),
        "IO_REFUSAL_NOT_QUEUE_POISON"
    );
    // Exercise the existing replacement and final receipt path before fixture
    // teardown; no native IO callback was invented for the refused record.
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(computed(&f.dst), "IO_REFUSAL_NEXT_RECORDING_RECOVERS");
}

#[test]
fn validated_io_listener_wakes_failure_without_manual_produced_call() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    let mut generation = TestIoGeneration::new(f.native.as_ref(), 2);
    f.commands
        .commit_when_io_validated(generation.dependency(1))
        .unwrap();
    encode_compute(&f);
    generation.terminal(0, 2);
    // No produced_on call: exercises the real registered Metal event listener.
    assert!(
        until(Duration::from_secs(5), || f
            .commands
            .core
            .state
            .lock(Side::Host)
            .current_abandoned),
        "IO_REAL_LISTENER_FAILURE_WAKE"
    );
    assert!(
        dst_bytes(&f.dst).iter().all(|b| *b == WARM),
        "IO_LISTENER_DID_NOT_SUBMIT_INVALID_READ"
    );
    assert!(f.commands.core.failure.check().is_ok());
}

/// A failed IO generation cancels the current recording that waited for it,
/// then the last handle drops with no recording operation between: the final
/// owner retires by accounting, with no native call and no epoch of its own,
/// and runs nothing the cancellation took (decision 99).
#[test]
fn validated_io_direct_teardown_after_cancellation_retires_by_accounting() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    let mut generation = TestIoGeneration::new(f.native.as_ref(), 2);
    let dependency = generation.dependency(1);
    f.commands
        .commit_when_io_validated(dependency.clone())
        .unwrap();
    encode_compute(&f);
    generation.terminal(0, 2);
    f.commands
        .core
        .produced_on(dependency.notification(), u64::MAX)
        .unwrap();
    assert!(f.commands.core.state.lock(Side::Host).current_abandoned);
    let weak = f.commands.core.produced.me_for_test();
    let dst = f.dst.clone();
    let clock = f.commands.clock();
    let opened = clock.current_epoch().0;
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    // Deliberately no caller encode/flush repair before final public drop.
    drop(f);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (after, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "IO_FINAL_ABANDONED retired={retired} native_calls={calls} epochs={completed}/{after} opened_before={opened}"
    );
    assert!(retired, "IO_CANCELLED_FINAL_OWNER_RETIRES");
    assert_eq!(calls, 0, "IO_FINAL_ABANDONMENT_MAKES_NO_NATIVE_CALL");
    assert_eq!(after, opened, "IO_FINAL_ABANDONMENT_OPENS_NO_EPOCH");
    assert_eq!(completed, after, "IO_FINAL_EVERY_EPOCH_COMPLETES");
    assert!(
        dst_bytes(&dst).iter().all(|b| *b == WARM),
        "IO_FINAL_ABANDONMENT_DID_NOT_EXECUTE_CANCELLED_BODY"
    );
}

/// The notification argument may be the subscribed threshold, not the later
/// terminal event value. Invoke the actual IO block without a competing native
/// listener so the old argument-only callback cannot pass through timing luck.
#[test]
fn validated_io_threshold_callback_observes_later_terminal_notice() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    let mut generation = TestIoGeneration::new(f.native.as_ref(), 2);
    let dependency = generation.dependency(1);
    assert!(f.commands.gate_when_io_validated(&dependency).unwrap());
    encode_compute(&f);
    generation.terminal(0, 2);
    assert_eq!(dependency.notification().signaledValue(), u64::MAX);
    assert_eq!(dependency.value(), 2);
    f.commands
        .core
        .produced
        .io_notification()
        .call((NonNull::from(dependency.notification()), dependency.value()));
    let cancelled = f.commands.core.state.lock(Side::Host).current_abandoned;
    let poison = f.commands.core.failure.check().is_err();
    // Cleanup is explicit even for the argument-only causal mutant. Preserve
    // the observation before this helper; it cannot make the assertion pass.
    f.commands
        .core
        .produced_on(dependency.notification(), u64::MAX)
        .unwrap();
    let mut fresh = TestIoGeneration::new(f.native.as_ref(), 1);
    let ready = fresh.dependency(0);
    fresh.terminal(0, 3);
    f.commands.commit_when_io_validated(ready).unwrap();
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(
        cancelled,
        "IO_THRESHOLD_CALLBACK_MUST_OBSERVE_TERMINAL_NOTICE"
    );
    assert!(!poison, "IO_THRESHOLD_CALLBACK_NOT_QUEUE_POISON");
    assert!(
        computed(&f.dst),
        "IO_THRESHOLD_CALLBACK_NEXT_GENERATION_RECOVERS"
    );
}

/// Actual service fanout and real Commands event listener, with NO native IO
/// terminal callback. Empty prepared IOCBs remain rooted until explicit cleanup.
#[test]
fn validated_io_service_failure_without_callback_notifies_commands() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    assert_eq!(f.commands.host_input_arm(), HostInputArm::Commit);
    let generation = TestIoGeneration::new(f.native.as_ref(), 2);
    let dependency = generation.dependency(1);
    let before = generation.untouched_custody();
    f.commands
        .commit_when_io_validated(dependency.clone())
        .unwrap();
    encode_compute(&f);
    generation.fail_service_without_callback();
    // Saved observations precede ANY manual repair/pass/terminal injection.
    let cancelled = until(Duration::from_secs(5), || {
        f.commands.core.state.lock(Side::Host).current_abandoned
    });
    let notice = dependency.notification().signaledValue();
    let custody = generation.untouched_custody();
    let unchanged = dst_bytes(&f.dst).iter().all(|b| *b == WARM);
    let poisoned = f.commands.core.failure.check().is_err();
    // Cleanup for the missing-edge mutant cannot make the saved oracle pass.
    f.commands
        .core
        .produced_on(dependency.notification(), u64::MAX)
        .unwrap();
    let mut fresh = TestIoGeneration::new(f.native.as_ref(), 1);
    let ready = fresh.dependency(0);
    fresh.terminal(0, 3);
    f.commands.commit_when_io_validated(ready).unwrap();
    encode_dependent(&f);
    f.commands.flush_and_wait().unwrap();
    assert!(
        cancelled,
        "IO_SERVICE_FAILURE_WITHOUT_CALLBACK_NOTIFIES_COMMANDS"
    );
    assert_eq!(notice, u64::MAX, "IO_SERVICE_FAILURE_PUBLISHES_NOTICE");
    assert_eq!(
        custody, before,
        "IO_SERVICE_FAILURE_NOTICE_NOT_NATIVE_RETIREMENT"
    );
    assert!(
        unchanged,
        "IO_SERVICE_FAILURE_NO_INVALID_CONSUMER_EXECUTION"
    );
    assert!(!poisoned, "IO_SERVICE_FAILURE_NOT_WHOLE_COMMANDS_POISON");
    assert!(
        computed(&f.dst),
        "IO_SERVICE_FAILURE_FRESH_GENERATION_RECOVERS"
    );
}

/// A recording that gates a second value of its event is counted, and the
/// gate path prints nothing: it runs under the state lock, so the count is
/// left to the run's report (`[host input since load: … beside_its_event=]`).
///
/// The test runs the gate in a child of this binary (its own harness name,
/// `--exact`, output not captured) and reads the child's stderr: the
/// child's `BESIDE_COUNT` line is the positive control that the gate path
/// ran there, and it must be the only line about it.
#[test]
fn a_value_beside_its_event_is_counted_and_the_gate_path_prints_nothing() {
    const CHILD: &str = "LLOOM_TEST_BESIDE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let f = fixture();
        if f.commands.host_input_arm() == HostInputArm::GpuWait {
            eprintln!("NOT_APPLICABLE nothing is gated on the gpu-wait arm");
            return;
        }
        f.commands.commit_when_produced(&f.gate, 1).unwrap();
        f.commands.commit_when_produced(&f.gate, 5).unwrap();
        let stats = f.commands.produced_stats();
        let cancelled = f
            .commands
            .cancel_produced_through(&f.gate, 5, "test: never produced");
        f.gate.setSignaledValue(5);
        let _ = f.commands.produced(&f.gate, 5);
        f.commands.flush_and_wait().unwrap();
        eprintln!(
            "BESIDE_COUNT {} cancelled={cancelled:?}",
            stats.beside_its_event
        );
        return;
    }
    let name = std::thread::current().name().unwrap().to_string();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture", "--test-threads", "1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("CHILD status={:?} stderr:\n{stderr}", out.status);
    assert!(out.status.success(), "the child failed: {stderr}");
    if stderr.contains("NOT_APPLICABLE") {
        return;
    }
    assert!(
        stderr.lines().any(|l| l.starts_with("BESIDE_COUNT 1 ")),
        "the child's gate path did not count one value beside its event: {stderr}"
    );
    assert!(
        !stderr.contains("beside another value of its event"),
        "the gate path printed under the state lock: {stderr}"
    );
}

/// Final teardown commits held work as recorded, whatever its gate: with
/// no producer left there is nothing to wait for (`release_held_for_teardown`).
/// A host producer never leaves work there that waited for a value it gave
/// up: it cancels through the value, then raises it (`HostEvent::cancel_through`,
/// `HostFedRing::release`), so that work runs neither before teardown nor at
/// it. The control is the same held work with its value neither produced nor
/// cancelled: it runs once the last handle drops, which is what shows the
/// teardown was reached at all.
#[test]
fn final_teardown_runs_nothing_a_host_producer_gave_up() {
    let mut control_latency = None;
    for gave_up in [false, true] {
        let f = fixture();
        if f.commands.host_input_arm() == HostInputArm::GpuWait {
            eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
            return;
        }
        f.commands.commit_when_produced(&f.gate, 1).unwrap();
        encode_dependent(&f);
        f.commands.flush_marker().unwrap();
        assert!(f.commands.produced_stats().held_now > 0, "nothing was held");
        if gave_up {
            f.commands
                .cancel_produced_through(&f.gate, 1, "test: the producer gave up")
                .unwrap();
            f.gate.setSignaledValue(1);
            f.commands.produced(&f.gate, 1).unwrap();
            let stats = f.commands.produced_stats();
            assert_eq!(stats.held_now, 0, "{stats:?}");
            // Nothing is held, so this waits for nothing the producer owes;
            // it settles the recording the cancellation replaced.
            f.commands.flush_and_wait().unwrap();
        }
        let Fixture {
            native,
            commands,
            src,
            dst,
            gate,
            progress,
        } = f;
        let api = commands.actual_command_api();
        drop(commands);
        let bound = control_latency.map_or(Duration::from_secs(10), |l: Duration| {
            (l * 10).max(Duration::from_secs(1))
        });
        let t0 = Instant::now();
        while progress.signaledValue() == 0 && t0.elapsed() < bound {
            std::thread::sleep(Duration::from_millis(5));
        }
        let waited = t0.elapsed();
        let ran = progress.signaledValue() >= 1;
        if ran {
            // The progress signal follows the compute on the same stream.
            assert!(computed(&dst), "the progress signal ran before its compute");
        }
        eprintln!(
            "FINAL_TEARDOWN api={api} gave_up={gave_up} ran={ran} waited_ms={} bound_ms={} warm={}",
            waited.as_millis(),
            bound.as_millis(),
            dst_bytes(&dst).iter().all(|b| *b == WARM)
        );
        if gave_up {
            assert!(
                !ran,
                "work behind a value its producer gave up ran at teardown"
            );
            assert!(
                dst_bytes(&dst).iter().all(|b| *b == WARM),
                "cancelled work ran"
            );
        } else {
            assert!(
                ran,
                "the control's held work never ran: teardown was not reached, so the arm \
                 after it reads nothing"
            );
            control_latency = Some(waited);
        }
        drop((native, src, dst, gate));
    }
}

/// A cancellation that takes the current recording, then the last handle
/// drops with no recording operation between: the final owner must still
/// retire, by accounting, with no native call and no epoch of its own, and
/// run nothing the cancellation took. The ordinary host-producer counterpart
/// of the validated-IO control above
/// (`validated_io_direct_teardown_after_cancellation_retires_by_accounting`).
#[test]
fn final_owner_retires_after_a_cancellation_took_the_current_recording() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_dependent(&f);
    f.commands
        .cancel_produced(&f.gate, "test: the producer gave up")
        .unwrap();
    assert!(f.commands.core.state.lock(Side::Host).current_abandoned);
    let weak = f.commands.core.produced.me_for_test();
    let clock = f.commands.clock();
    let opened = clock.current_epoch().0;
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    let api = f.commands.actual_command_api();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (after, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "FINAL_AFTER_CANCEL api={api} retired={retired} native_calls={calls} epochs={completed}/{after} opened_before={opened} progress={} warm={}",
        progress.signaledValue(),
        dst_bytes(&dst).iter().all(|b| *b == WARM)
    );
    assert!(retired, "FINAL_OWNER_RETIRES_AFTER_CANCELLED_CURRENT");
    assert_eq!(calls, 0, "FINAL_ABANDONMENT_MAKES_NO_NATIVE_CALL");
    assert_eq!(after, opened, "FINAL_ABANDONMENT_OPENS_NO_EPOCH");
    assert_eq!(completed, after, "FINAL_ABANDONMENT_EVERY_EPOCH_COMPLETES");
    assert_eq!(
        progress.signaledValue(),
        0,
        "FINAL_ABANDONMENT_RAN_NOTHING_CANCELLED"
    );
    assert!(
        dst_bytes(&dst).iter().all(|b| *b == WARM),
        "FINAL_CANCELLED_BODY_NOT_EXECUTED"
    );
    drop((native, src, dst, gate, progress));
}

/// Teardown with an IO load still pending: ordinary held work ahead of it
/// runs as recorded, the work behind the load is cancelled (its bytes never
/// landed), the final owner retires by accounting — the ordinary held work is
/// the only native call, and no epoch is opened — and every epoch the
/// cancelled recordings took completes once that work has.
#[test]
fn final_teardown_cancels_work_behind_pending_validated_io() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let generation = TestIoGeneration::new(f.native.as_ref(), 1);
    let dependency = generation.dependency(0);
    let custody = generation.untouched_custody();
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_compute(&f);
    f.commands.flush_marker().unwrap();
    f.commands
        .commit_when_io_validated(dependency.clone())
        .unwrap();
    f.commands.encode_signal_event(&f.progress, 1).unwrap();
    // Hold the IO-dependent recording on both APIs (Metal 4 already cut at
    // the signal), so teardown cancels a HELD recording and owes its epoch.
    f.commands.flush_marker().unwrap();
    let held = f.commands.produced_stats().held_now;
    assert!(
        held >= 2,
        "the IO-dependent recording was not held behind the first"
    );
    let clock = f.commands.clock();
    let opened_before = clock.current_epoch().0;
    let weak = f.commands.core.produced.me_for_test();
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    let api = f.commands.actual_command_api();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (opened, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "TEARDOWN_PENDING_IO api={api} held={held} retired={retired} native_calls={calls} epochs={completed}/{opened} opened_before={opened_before} progress={} computed={}",
        progress.signaledValue(),
        computed(&dst)
    );
    assert!(retired, "TEARDOWN_PENDING_IO_FINAL_OWNER_RETIRES");
    assert_eq!(
        calls, 1,
        "TEARDOWN_PENDING_IO_ONLY_THE_ORDINARY_HELD_WORK_IS_SUBMITTED"
    );
    assert_eq!(opened, opened_before, "TEARDOWN_PENDING_IO_OPENS_NO_EPOCH");
    assert!(computed(&dst), "TEARDOWN_ORDINARY_HELD_RUNS_AS_RECORDED");
    assert_eq!(
        progress.signaledValue(),
        0,
        "TEARDOWN_WORK_BEHIND_PENDING_IO_NOT_RUN"
    );
    assert_eq!(completed, opened, "TEARDOWN_EVERY_EPOCH_COMPLETES");
    assert!(!dependency.ready(), "TEARDOWN_CANCELLATION_IS_NOT_DATA");
    assert_eq!(
        generation.untouched_custody(),
        custody,
        "TEARDOWN_LEAVES_IO_CUSTODY_TO_THE_BANK"
    );
    drop((native, src, dst, gate, progress));
}

/// As above with the IO-dependent work still in the current recording, its
/// compute encoder open when the last handle drops: teardown closes it,
/// cancels the recording and retires by accounting, running nothing and
/// making no native call.
#[test]
fn final_teardown_cancels_a_current_recording_behind_pending_validated_io() {
    use crate::io_terminal::TestIoGeneration;
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let generation = TestIoGeneration::new(f.native.as_ref(), 1);
    let dependency = generation.dependency(0);
    f.commands
        .commit_when_io_validated(dependency.clone())
        .unwrap();
    encode_compute(&f);
    assert!(!f.commands.core.state.lock(Side::Host).current_abandoned);
    let clock = f.commands.clock();
    let opened_before = clock.current_epoch().0;
    let weak = f.commands.core.produced.me_for_test();
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    let api = f.commands.actual_command_api();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (opened, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "TEARDOWN_PENDING_IO_CURRENT api={api} retired={retired} native_calls={calls} epochs={completed}/{opened} opened_before={opened_before} warm={}",
        dst_bytes(&dst).iter().all(|b| *b == WARM)
    );
    assert!(retired, "TEARDOWN_PENDING_IO_CURRENT_FINAL_OWNER_RETIRES");
    assert_eq!(calls, 0, "TEARDOWN_PENDING_IO_CURRENT_NO_NATIVE_CALL");
    assert_eq!(
        opened, opened_before,
        "TEARDOWN_PENDING_IO_CURRENT_OPENS_NO_EPOCH"
    );
    assert!(
        dst_bytes(&dst).iter().all(|b| *b == WARM),
        "TEARDOWN_PENDING_IO_CURRENT_BODY_NOT_RUN"
    );
    assert_eq!(
        completed, opened,
        "TEARDOWN_PENDING_IO_CURRENT_EVERY_EPOCH_COMPLETES"
    );
    drop((native, src, dst, gate, progress, generation));
}

/// The owed epochs of an abandoned final ride the last submission still in
/// flight. A submission held on the GPU by a wait, then a held reader whose
/// producer gives up: until the submission completes, neither its epoch nor
/// the cancelled one is covered (`GpuClock::mark_completed` is a maximum, so an
/// early mark would cover both) and the final owner stays pending; once it
/// completes, every epoch completes and the final owner retires with no
/// native call of its own (decision 99).
#[test]
fn final_abandonment_rides_the_last_submission_still_in_flight() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    // A GPU wait on an event this test raises well inside the queue timeout.
    let block = f.native.as_ref().newSharedEvent().unwrap();
    let clock = f.commands.clock();
    f.commands.encode_wait_for_event(&block, 1).unwrap();
    encode_compute(&f);
    // Metal 4 cuts before a wait: the blocked recording's epoch is the one
    // its own cut takes, read after the wait was encoded.
    let in_flight = clock.current_epoch();
    let marker = f.commands.flush_marker().unwrap();
    assert!(marker.is_submitted(), "the blocked submission was held");
    let (_held, held_epoch) = hold_on(&f, &f.gate);
    f.commands
        .cancel_produced(&f.gate, "test: the producer gave up")
        .unwrap();
    assert!(f.commands.core.state.lock(Side::Host).current_abandoned);
    assert_eq!(owed_now(&f), 1, "the cancelled reader's epoch is not owed");
    let weak = f.commands.core.produced.me_for_test();
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    let opened = clock.current_epoch().0;
    let api = f.commands.actual_command_api();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    std::thread::sleep(Duration::from_millis(300));
    let pending = weak.upgrade().is_some();
    let in_flight_done = clock.is_complete(in_flight);
    let held_done = clock.is_complete(held_epoch);
    block.setSignaledValue(1);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (after, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "FINAL_RIDES api={api} pending={pending} in_flight_done={in_flight_done} held_done={held_done} retired={retired} native_calls={calls} epochs={completed}/{after} opened_before={opened}"
    );
    assert!(pending, "FINAL_ABANDONMENT_PENDING_WHILE_EARLIER_WORK_RUNS");
    assert!(
        !in_flight_done,
        "FINAL_ABANDONMENT_COVERS_NO_EPOCH_STILL_IN_FLIGHT"
    );
    assert!(!held_done, "FINAL_ABANDONED_EPOCH_WAITS_FOR_THE_PREFIX");
    assert!(
        retired,
        "FINAL_ABANDONMENT_RETIRES_ONCE_THE_PREFIX_COMPLETES"
    );
    assert_eq!(calls, 0, "FINAL_ABANDONMENT_MAKES_NO_NATIVE_CALL");
    assert_eq!(after, opened, "FINAL_ABANDONMENT_OPENS_NO_EPOCH");
    assert_eq!(completed, after, "FINAL_ABANDONMENT_EVERY_EPOCH_COMPLETES");
    drop((native, src, dst, gate, progress, block));
}

/// A completion subscriber blocks on the completion thread inside the last
/// submission's hook run. That submission's epoch is marked before its
/// subscribers run, but its record has not finished: the abandoned epoch
/// stays uncovered and the final owner pending until the subscriber
/// returns. A prefix proof read from the hook's run count alone would
/// certify here, with a subscriber still running on a snapshot the final
/// captures are about to take.
#[test]
fn final_abandonment_waits_for_the_last_completion_to_return() {
    struct Blocker {
        armed: std::sync::atomic::AtomicBool,
        entered: std::sync::atomic::AtomicBool,
        released: std::sync::atomic::AtomicBool,
    }
    let block = Arc::new(Blocker {
        armed: false.into(),
        entered: false.into(),
        released: false.into(),
    });
    let observed = block.clone();
    let f = fixture_with(move |commands| {
        commands
            .on_command_buffer_complete(move || {
                if !observed.armed.swap(false, Ordering::AcqRel) {
                    return;
                }
                observed.entered.store(true, Ordering::Release);
                let t0 = Instant::now();
                while !observed.released.load(Ordering::Acquire)
                    && t0.elapsed() < Duration::from_secs(10)
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
            .unwrap_or_else(|_| panic!("subscriber registration refused"));
    });
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        block.released.store(true, Ordering::Release);
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    block.armed.store(true, Ordering::Release);
    encode_compute(&f);
    let marker = f.commands.flush_marker().unwrap();
    assert!(marker.is_submitted(), "the last submission was held");
    assert!(
        until(Duration::from_secs(5), || block
            .entered
            .load(Ordering::Acquire)),
        "the subscriber never ran"
    );
    let (_held, held_epoch) = hold_on(&f, &f.gate);
    f.commands
        .cancel_produced(&f.gate, "test: the producer gave up")
        .unwrap();
    assert!(f.commands.core.state.lock(Side::Host).current_abandoned);
    let clock = f.commands.clock();
    let opened = clock.current_epoch().0;
    let weak = f.commands.core.produced.me_for_test();
    let counters = f.commands.core.submission_control.clone();
    let before = counters.native_calls.load(Ordering::Acquire);
    let api = f.commands.actual_command_api();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    std::thread::sleep(Duration::from_millis(300));
    let pending = weak.upgrade().is_some();
    let held_done = clock.is_complete(held_epoch);
    block.released.store(true, Ordering::Release);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    let calls = counters.native_calls.load(Ordering::Acquire) - before;
    let (after, completed) = (clock.current_epoch().0, clock.completed_count().0);
    eprintln!(
        "FINAL_WAITS_FOR_RETURN api={api} pending={pending} held_done={held_done} retired={retired} native_calls={calls} epochs={completed}/{after} opened_before={opened}"
    );
    assert!(pending, "FINAL_ABANDONMENT_PENDING_WHILE_A_COMPLETION_RUNS");
    assert!(
        !held_done,
        "FINAL_ABANDONED_EPOCH_WAITS_FOR_THE_COMPLETION_TO_RETURN"
    );
    assert!(retired, "FINAL_ABANDONMENT_RETIRES_ONCE_IT_RETURNS");
    assert_eq!(calls, 0, "FINAL_ABANDONMENT_MAKES_NO_NATIVE_CALL");
    assert_eq!(after, opened, "FINAL_ABANDONMENT_OPENS_NO_EPOCH");
    assert_eq!(completed, after, "FINAL_ABANDONMENT_EVERY_EPOCH_COMPLETES");
    drop((native, src, dst, gate, progress));
}

/// A cancellation whose owed-epoch bound refuses, and whose close of the
/// current encoder refuses after it, still fails the queue for good: the
/// refusal is latched on that early exit. No later submission is accepted,
/// no epoch is marked, and the cancelled held recording is abandoned in
/// custody, never run (lane C, #944 6556673766 §3).
#[test]
fn a_bound_refusal_is_latched_when_a_close_refuses_after_it() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let (held, held_epoch) = hold_on(&f, &f.gate);
    // An open compute encoder on the current recording: the close refuses.
    encode_compute(&f);
    {
        let mut state = f.commands.core.state.lock(Side::Host);
        assert!(state.current_encoder.is_some(), "no encoder to close");
        let flagged = state.owed_epochs.flagged();
        state.owed_epochs = produced::Owed::for_test(produced::Owed::MAX_OWED, flagged);
    }
    let clock = f.commands.clock();
    let completed = clock.completed_count().0;
    let calls = native_calls(&f.commands);
    let api = f.commands.actual_command_api();
    f.commands
        .core
        .submission_control
        .arm(submission_test::Fault::CloseRefusal);
    let cancelled = f.commands.cancel_produced(&f.gate, "test: bound");
    let sticky = f.commands.core.failure.check();
    eprintln!(
        "BOUND_REFUSAL_LATCH api={api} cancel={cancelled:?} sticky={sticky:?} held_abandoned={}",
        held.cb.abandoned()
    );
    assert!(cancelled.is_err(), "BOUND_REFUSAL_REPORTED");
    assert!(sticky.is_err(), "BOUND_REFUSAL_LATCHED_ON_THE_EARLY_EXIT");
    let flushed = f.commands.flush_and_wait();
    f.gate.setSignaledValue(1);
    let released = f.commands.produced(&f.gate, 1);
    std::thread::sleep(Duration::from_millis(200));
    let after_calls = native_calls(&f.commands);
    let after_completed = clock.completed_count().0;
    eprintln!(
        "BOUND_REFUSAL_AFTER api={api} flushed={flushed:?} released={released:?} calls={calls}->{after_calls} completed={completed}->{after_completed} held_epoch_done={}",
        clock.is_complete(held_epoch)
    );
    assert!(flushed.is_err(), "NO_SUBMISSION_ACCEPTED_AFTER_THE_REFUSAL");
    assert_eq!(after_calls, calls, "NO_NATIVE_CALL_AFTER_THE_REFUSAL");
    assert_eq!(after_completed, completed, "NO_EPOCH_MARKED_BY_THE_REFUSAL");
    assert!(
        !clock.is_complete(held_epoch),
        "THE_CANCELLED_EPOCH_IS_NOT_COMPLETE"
    );
    assert!(
        held.cb.abandoned(),
        "THE_CANCELLED_HELD_RECORDING_IS_ABANDONED_IN_CUSTODY"
    );
    // A failed queue's teardown quarantines and retains its Core by design.
}

/// The re-homed IO behind a cancelled signal survives a close refusal. A
/// held recording carries an IO load's compute signal (`commit_io_after`
/// deferred the IO buffer behind it); its producer gives up while the
/// current recording has an open encoder whose close refuses. The load is
/// still committed behind a signal re-issued on the queue, and lands. Before
/// the re-homed list was placed ahead of the closes, the refusal's early exit
/// dropped it and the IO buffer was never committed (#944 decision-99 lane,
/// the dev defect beside lane C's §3).
#[test]
fn a_close_refusal_keeps_the_rehomed_io_behind_a_cancelled_signal() {
    use crate::kernels::metal::io_queue::{IoLoad, IoQueue};
    const LEN: usize = 64 * 1024;
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let bytes: Vec<u8> = (0..LEN).map(|i| (i * 31 % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!(
        "lloom-d99-rehome-{}-{}.bin",
        f.commands.actual_command_api(),
        std::process::id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let queue = IoQueue::new(&f.native).unwrap();
    let file = queue.open(&path).unwrap();
    let dst = f
        .native
        .new_buffer(LEN, super::super::MTLResourceOptions::StorageModeShared)
        .unwrap();
    // The held recording: the gated compute, then the load's compute signal.
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_compute(&f);
    let before = queue.signaled();
    let value = queue
        .issue(
            &f.commands,
            &[IoLoad {
                dst: &dst,
                dst_offset: 0,
                file: &file,
                src_offset: 0,
                len: LEN,
            }],
        )
        .unwrap();
    let held = f.commands.produced_stats().held_now;
    assert!(held >= 1, "the signal's recording was not held");
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(queue.signaled(), before, "the load ran before its signal");
    // An open encoder on the current recording; its close refuses.
    encode_compute(&f);
    f.commands
        .core
        .submission_control
        .arm(submission_test::Fault::CloseRefusal);
    let cancelled = f
        .commands
        .cancel_produced(&f.gate, "test: the producer gave up");
    let landed = until(Duration::from_secs(5), || queue.signaled() >= value);
    // SAFETY: a CPU-mapped Shared buffer of LEN bytes; its only writer, the
    // load, has signalled (or never ran).
    let got = unsafe { std::slice::from_raw_parts(dst.contents() as *const u8, LEN) };
    let right = got == bytes.as_slice();
    eprintln!(
        "REHOME_AFTER_CLOSE_REFUSAL api={} held={held} cancel={cancelled:?} landed={landed} bytes_right={right}",
        f.commands.actual_command_api()
    );
    std::fs::remove_file(&path).ok();
    assert!(cancelled.is_err(), "the close refusal was not reported");
    assert!(landed, "REHOMED_IO_SURVIVES_A_CLOSE_REFUSAL");
    assert!(right, "REHOMED_IO_LANDS_THE_FILES_BYTES");
}

/// The gated current recording with an open encoder, the close of which is
/// armed to refuse with `fault`; then its producer gives up, and raises the
/// event anyway, as #1099's producers may once the cancel returns.
struct RefusedClose {
    cancelled: Result<usize, MetalKernelError>,
    sticky: Result<(), MetalKernelError>,
    pending: bool,
    calls: (usize, usize),
    abandoned: bool,
    ran: bool,
}

fn cancel_through_a_refused_close(f: &Fixture, fault: submission_test::Fault) -> RefusedClose {
    f.commands.commit_when_produced(&f.gate, 1).unwrap();
    encode_compute(f);
    let calls = native_calls(&f.commands);
    f.commands.core.submission_control.arm(fault);
    let cancelled = f.commands.cancel_produced(&f.gate, "test: the producer gave up");
    let sticky = f.commands.core.failure.check();
    let pending = f.commands.core.produced_cancel_pending();
    f.gate.setSignaledValue(1);
    let _ = f.commands.produced(&f.gate, 1);
    let abandoned = f.commands.core.state.lock(Side::Host).current_abandoned;
    let _ = f.commands.flush_and_wait();
    std::thread::sleep(Duration::from_millis(200));
    let after = native_calls(&f.commands);
    RefusedClose {
        cancelled,
        sticky,
        pending,
        calls: (calls, after),
        abandoned,
        ran: computed(&f.dst),
    }
}

/// A cancellation whose close of the current encoder refuses for good must
/// not lose its effect. The queue fails (a visible error), and raising the
/// event afterwards releases nothing: the dependent work never runs over the
/// input its producer gave up on. On dev before this fix the refusal's early
/// exit dropped the cancellation unlatched, and the raise released the
/// current recording (#944 ledger item 10; lane A 6548125286).
#[test]
fn a_refused_close_fails_the_queue_rather_than_lose_the_cancellation() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let RefusedClose {
        cancelled,
        sticky,
        pending,
        calls: (calls, after),
        abandoned,
        ran,
    } = cancel_through_a_refused_close(&f, submission_test::Fault::CloseRefusal);
    eprintln!(
        "REFUSED_CLOSE_CANCEL api={} cancel={cancelled:?} sticky={sticky:?} pending={pending} calls={calls}->{after} abandoned={abandoned} ran={ran}",
        f.commands.actual_command_api()
    );
    assert!(cancelled.is_err(), "REFUSED_CLOSE_REPORTED");
    assert!(!ran, "THE_CANCELLED_WORK_NEVER_RUNS");
    assert_eq!(after, calls, "NO_NATIVE_CALL_FOR_THE_CANCELLED_WORK");
    assert!(sticky.is_err(), "A_REFUSED_CLOSE_FAILS_THE_QUEUE");
}

/// A close that refuses only for now (`Busy`, a recording edit in progress)
/// keeps the cancellation pending instead of losing it, and does not fail
/// the queue. The raise that follows applies it first (a pending cancel is
/// applied before any release), so the current recording is abandoned, the
/// cancelled work never runs, and the queue still runs the next work.
#[test]
fn a_busy_close_keeps_the_cancellation_until_it_applies() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let RefusedClose {
        cancelled,
        sticky,
        pending,
        calls: (calls, after),
        abandoned,
        ran,
    } = cancel_through_a_refused_close(&f, submission_test::Fault::CloseBusy);
    // The queue still runs work that waits for nothing.
    encode_compute(&f);
    let next = f.commands.flush_and_wait();
    let next_ran = computed(&f.dst);
    eprintln!(
        "BUSY_CLOSE_CANCEL api={} cancel={cancelled:?} sticky={sticky:?} pending={pending} calls={calls}->{after} abandoned={abandoned} ran={ran} next={next:?} next_ran={next_ran}",
        f.commands.actual_command_api()
    );
    assert!(cancelled.is_err(), "BUSY_CLOSE_REPORTED");
    assert!(!ran, "THE_CANCELLED_WORK_NEVER_RUNS");
    assert_eq!(after, calls, "NO_NATIVE_CALL_FOR_THE_CANCELLED_WORK");
    assert!(pending, "THE_CANCELLATION_IS_KEPT_PENDING");
    assert!(abandoned, "THE_RAISE_APPLIES_THE_KEPT_CANCELLATION");
    assert!(sticky.is_ok(), "A_BUSY_CLOSE_DOES_NOT_FAIL_THE_QUEUE");
    assert!(next.is_ok() && next_ran, "THE_QUEUE_RUNS_THE_NEXT_WORK");
}

/// One held reader per gate, each on an event of its own: the dependent
/// compute recorded behind `commit_when_produced(event, 1)`, then cut. Its
/// marker and the epoch the cut took.
fn hold_on(f: &Fixture, event: &Event) -> (CommandMarker, Epoch) {
    let epoch = f.commands.clock().current_epoch();
    f.commands.commit_when_produced(event, 1).unwrap();
    encode_compute(f);
    let marker = f.commands.flush_marker().unwrap();
    assert!(
        !marker.is_submitted(),
        "a gated reader was submitted at its cut"
    );
    (marker, epoch)
}

fn raise(f: &Fixture, event: &Event) -> usize {
    event.setSignaledValue(1);
    f.commands.produced(event, 1).unwrap()
}

/// The clock once the completions a wait returned on have run: polled past
/// `at_least`, then given time for any further hook run of the same
/// completion to land, so an over-marking completion is read at its end.
fn settled_completed(f: &Fixture, at_least: u64) -> u64 {
    let clock = f.commands.clock();
    assert!(
        until(Duration::from_secs(5), || clock.completed_count().0
            >= at_least),
        "epoch {} never completed",
        at_least - 1
    );
    std::thread::sleep(Duration::from_millis(200));
    clock.completed_count().0
}

/// A pool buffer of `size` bytes released now: parked at the clock's current
/// epoch, the one the next cut takes, until that epoch completes
/// (`BufferPool`'s reuse rule, `GpuClock::is_complete`).
fn park(f: &Fixture, pool: &crate::kernels::metal::BufferPool, size: usize) {
    let raw = f
        .native
        .new_buffer(size, super::super::MTLResourceOptions::StorageModeShared)
        .unwrap();
    drop(pool.adopt(raw, size));
}

/// Whether the pool would hand a parked `size` buffer out again now.
fn reusable(pool: &crate::kernels::metal::BufferPool, size: usize) -> bool {
    pool.acquire(size).is_some()
}

fn owed_now(f: &Fixture) -> u32 {
    f.commands.core.state.lock(Side::Host).owed_epochs.count()
}

/// The counterexample on #944 (H0/H1/H2 on gates A/B/C, cancel C, produce
/// A). H0's completion completes H0's epoch alone: H1 is still held, so its
/// epoch is not covered and anything the pool parked on it stays parked
/// (`BufferPool`'s reuse test is `GpuClock::is_complete`). Once B is produced
/// and H1 completes, every epoch has completed, the cancelled H2's with H1's.
#[test]
fn a_cancelled_tail_leaves_a_still_held_reader_uncovered() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b, c] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let pool = crate::kernels::metal::BufferPool::with_clock(f.commands.clock());
    const H1_FRAME: usize = 3 * 4096;
    let (m0, e0) = hold_on(&f, &a);
    // Memory H1 reads, released as H1 is recorded: parked at H1's epoch.
    park(&f, &pool, H1_FRAME);
    let (m1, e1) = hold_on(&f, &b);
    let (m2, e2) = hold_on(&f, &c);
    assert_eq!((e1.0, e2.0), (e0.0 + 1, e0.0 + 2));
    assert_eq!(f.commands.produced_stats().held_now, 3);
    let calls = native_calls(&f.commands);
    // H2 and the current recording behind it.
    assert_eq!(
        f.commands.cancel_produced(&c, "test: C gave up").unwrap(),
        2
    );
    assert_eq!(owed_now(&f), 1);
    assert_eq!(raise(&f, &a), 1, "A releases H0 alone");
    f.commands.wait_marker(&m0).unwrap();
    let completed = settled_completed(&f, e0.0 + 1);
    let clock = f.commands.clock();
    eprintln!(
        "OWED_H0_DONE api={} completed={completed} e0={} e1={} h1_submitted={} owed={} held={}",
        f.commands.actual_command_api(),
        e0.0,
        e1.0,
        m1.is_submitted(),
        owed_now(&f),
        f.commands.produced_stats().held_now
    );
    assert_eq!(
        native_calls(&f.commands) - calls,
        1,
        "H0 alone was submitted"
    );
    assert!(!m1.is_submitted(), "H1 left the held queue before B");
    assert_eq!(f.commands.produced_stats().held_now, 1);
    assert_eq!(
        completed,
        e0.0 + 1,
        "H0's completion marked an epoch past its own"
    );
    assert!(
        !clock.is_complete(e1),
        "H1's epoch completed while H1 was still held"
    );
    assert!(
        !reusable(&pool, H1_FRAME),
        "the pool handed out memory a still-held recording reads"
    );
    assert_eq!(
        owed_now(&f),
        1,
        "the owed epoch went with a non-tail commit"
    );
    assert_eq!(raise(&f, &b), 1, "B releases H1");
    f.commands.wait_marker(&m1).unwrap();
    let completed = settled_completed(&f, e2.0 + 1);
    eprintln!(
        "OWED_H1_DONE api={} completed={completed} current={} owed={}",
        f.commands.actual_command_api(),
        clock.current_epoch().0,
        owed_now(&f)
    );
    assert_eq!(completed, e2.0 + 1, "H1 completes its epoch and H2's");
    assert!(
        reusable(&pool, H1_FRAME),
        "H1's memory stayed parked after H1"
    );
    assert_eq!(clock.current_epoch().0, e2.0 + 1);
    assert!(clock.is_complete(e1) && clock.is_complete(e2));
    assert_eq!(owed_now(&f), 0);
    assert!(computed(&f.dst));
    let waited = f.commands.wait_marker(&m2);
    assert!(
        waited
            .as_ref()
            .is_err_and(|e| e.to_string().contains("C gave up")),
        "{waited:?}"
    );
    // The stream goes on.
    encode_compute(&f);
    f.commands.flush_and_wait().unwrap();
    assert_eq!(
        clock.completed_count().0,
        clock.current_epoch().0,
        "every epoch taken completed"
    );
    f.commands.check_failed().unwrap();
}

/// Held work added after a cancellation (H0; H1 cancelled; H2, H3 held
/// after it): H0's completion completes H0's epoch; H2 — the first commit
/// above the cancelled epoch — completes the cancelled epoch and its own at
/// its own completion, with H3 still held; H3 completes its own.
#[test]
fn work_held_after_a_cancellation_completes_its_own_epoch_at_its_completion() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b, c, d] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let (m0, e0) = hold_on(&f, &a);
    let (_m1, e1) = hold_on(&f, &b);
    assert_eq!(
        f.commands.cancel_produced(&b, "test: B gave up").unwrap(),
        2
    );
    assert_eq!(owed_now(&f), 1);
    let (m2, e2) = hold_on(&f, &c);
    let (m3, e3) = hold_on(&f, &d);
    assert_eq!((e1.0, e2.0, e3.0), (e0.0 + 1, e0.0 + 2, e0.0 + 3));
    let clock = f.commands.clock();
    assert_eq!(raise(&f, &a), 1);
    f.commands.wait_marker(&m0).unwrap();
    assert_eq!(settled_completed(&f, e0.0 + 1), e0.0 + 1);
    assert_eq!(owed_now(&f), 1);
    assert_eq!(raise(&f, &c), 1);
    f.commands.wait_marker(&m2).unwrap();
    let completed = settled_completed(&f, e2.0 + 1);
    eprintln!(
        "OWED_ADDED api={} completed={completed} e2={} h3_submitted={} owed={}",
        f.commands.actual_command_api(),
        e2.0,
        m3.is_submitted(),
        owed_now(&f)
    );
    assert!(!m3.is_submitted());
    assert_eq!(
        completed,
        e2.0 + 1,
        "H2's completion left its own epoch open"
    );
    assert!(!clock.is_complete(e3));
    assert_eq!(owed_now(&f), 0);
    assert_eq!(raise(&f, &d), 1);
    f.commands.wait_marker(&m3).unwrap();
    assert_eq!(settled_completed(&f, e3.0 + 1), e3.0 + 1);
    assert_eq!(clock.current_epoch().0, e3.0 + 1);
    f.commands.check_failed().unwrap();
}

/// Two cancellations, the later one shallower in the queue (H0..H3; cancel
/// D, then C): H0 completes its own epoch; H1, the last live one,
/// completes its own and both cancelled ones.
#[test]
fn repeated_cancellations_complete_with_the_last_live_reader() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b, c, d] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let (m0, e0) = hold_on(&f, &a);
    let (m1, e1) = hold_on(&f, &b);
    let (_, _) = hold_on(&f, &c);
    let (_, e3) = hold_on(&f, &d);
    assert_eq!(
        f.commands.cancel_produced(&d, "test: D gave up").unwrap(),
        2
    );
    assert_eq!(
        f.commands.cancel_produced(&c, "test: C gave up").unwrap(),
        1
    );
    assert_eq!(owed_now(&f), 2);
    let clock = f.commands.clock();
    assert_eq!(raise(&f, &a), 1);
    f.commands.wait_marker(&m0).unwrap();
    assert_eq!(settled_completed(&f, e0.0 + 1), e0.0 + 1);
    assert!(!clock.is_complete(e1));
    assert!(!m1.is_submitted());
    assert_eq!(raise(&f, &b), 1);
    f.commands.wait_marker(&m1).unwrap();
    assert_eq!(settled_completed(&f, e3.0 + 1), e3.0 + 1);
    assert_eq!(clock.current_epoch().0, e3.0 + 1);
    assert_eq!(owed_now(&f), 0);
    f.commands.check_failed().unwrap();
}

/// A held release the owner's submission refuses before its epoch keeps
/// what is owed: the retry carries it, and the mapping is exact.
#[test]
fn a_refused_release_keeps_the_owed_epochs_for_its_retry() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let (m0, _) = hold_on(&f, &a);
    let (_, e1) = hold_on(&f, &b);
    assert_eq!(
        f.commands.cancel_produced(&b, "test: B gave up").unwrap(),
        2
    );
    assert_eq!(owed_now(&f), 1);
    // A produced but not announced: the owner's wait releases H0.
    a.setSignaledValue(1);
    let calls = native_calls(&f.commands);
    let control = &f.commands.core.submission_control;
    control.arm(submission_test::Fault::Registration);
    let refused = f.commands.wait_marker(&m0);
    assert!(
        matches!(
            refused,
            Err(MetalKernelError::QueueUnavailable(
                "injected completion registration refusal"
            ))
        ),
        "{refused:?}"
    );
    assert_eq!(
        native_calls(&f.commands),
        calls,
        "a refused release reached the GPU"
    );
    assert!(!m0.is_submitted());
    assert_eq!(owed_now(&f), 1, "a refused release consumed the owed epoch");
    assert_eq!(f.commands.clock().current_epoch().0, e1.0 + 1);
    f.commands.wait_marker(&m0).unwrap();
    assert_eq!(settled_completed(&f, e1.0 + 1), e1.0 + 1);
    assert_eq!(owed_now(&f), 0);
    f.commands.check_failed().unwrap();
}

/// The ordinary current commit after a cancellation emptied the held queue
/// carries what is owed: its completion completes every epoch taken.
#[test]
fn the_next_current_commit_carries_the_owed_epochs() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let (_, e0) = hold_on(&f, &a);
    assert_eq!(
        f.commands.cancel_produced(&a, "test: A gave up").unwrap(),
        2
    );
    assert_eq!(owed_now(&f), 1);
    let clock = f.commands.clock();
    assert_eq!(clock.completed_count().0, e0.0);
    encode_compute(&f);
    f.commands.flush_and_wait().unwrap();
    assert_eq!(settled_completed(&f, e0.0 + 2), e0.0 + 2);
    assert_eq!(clock.current_epoch().0, e0.0 + 2);
    assert_eq!(owed_now(&f), 0);
    assert!(computed(&f.dst));
    f.commands.check_failed().unwrap();
}

/// Final teardown with a cancelled tail and a live reader not produced: the
/// teardown commits H0 and H1 as recorded, the owed epoch rides H1, and
/// every epoch has completed once the owner retires.
#[test]
fn teardown_after_a_cancelled_tail_completes_every_epoch() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b, c] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let (_, _) = hold_on(&f, &a);
    let (_, _) = hold_on(&f, &b);
    let (_, e2) = hold_on(&f, &c);
    assert_eq!(
        f.commands.cancel_produced(&c, "test: C gave up").unwrap(),
        2
    );
    let clock = f.commands.clock();
    let weak = f.commands.core.produced.me_for_test();
    let Fixture {
        native,
        commands,
        src,
        dst,
        gate,
        progress,
    } = f;
    drop(commands);
    let retired = until(Duration::from_secs(5), || weak.upgrade().is_none());
    eprintln!(
        "OWED_TEARDOWN retired={retired} completed={} current={} e2={}",
        clock.completed_count().0,
        clock.current_epoch().0,
        e2.0
    );
    assert!(retired);
    // H0, H1; H2's epoch owed and completed with H1. The abandoned final
    // opens no epoch of its own (decision 99).
    assert_eq!(clock.current_epoch().0, e2.0 + 1);
    assert_eq!(clock.completed_count().0, clock.current_epoch().0);
    assert!(computed(&dst));
    drop((native, src, dst, gate, progress, a, b, c));
}

/// Interleaved holes (H0 | cancelled H1 | H2 | cancelled H3 | H4, the review
/// on #944): H2's completion completes the hole below it and its own epoch
/// while H4's producer has not run, so memory parked at H2's epoch is
/// reusable then; H4 completes the hole above H2 and its own.
#[test]
fn a_reader_between_two_cancelled_ranges_completes_at_its_own_completion() {
    let f = fixture();
    if f.commands.host_input_arm() == HostInputArm::GpuWait {
        eprintln!("NOT_APPLICABLE nothing is held on the gpu-wait arm");
        return;
    }
    let [a, b, c, d, e] = std::array::from_fn(|_| f.native.as_ref().newSharedEvent().unwrap());
    let pool = crate::kernels::metal::BufferPool::with_clock(f.commands.clock());
    const H2_FRAME: usize = 5 * 4096;
    let (m0, e0) = hold_on(&f, &a);
    let (_, e1) = hold_on(&f, &b);
    assert_eq!(
        f.commands.cancel_produced(&b, "test: B gave up").unwrap(),
        2
    );
    park(&f, &pool, H2_FRAME);
    let (m2, e2) = hold_on(&f, &c);
    let (_, e3) = hold_on(&f, &d);
    assert_eq!(
        f.commands.cancel_produced(&d, "test: D gave up").unwrap(),
        2
    );
    let (m4, e4) = hold_on(&f, &e);
    assert_eq!(
        (e1.0, e2.0, e3.0, e4.0),
        (e0.0 + 1, e0.0 + 2, e0.0 + 3, e0.0 + 4)
    );
    assert_eq!(owed_now(&f), 2);
    let clock = f.commands.clock();
    assert_eq!(raise(&f, &a), 1);
    f.commands.wait_marker(&m0).unwrap();
    assert_eq!(settled_completed(&f, e0.0 + 1), e0.0 + 1);
    assert!(!reusable(&pool, H2_FRAME), "H2's memory left before H2 ran");
    assert_eq!(raise(&f, &c), 1);
    f.commands.wait_marker(&m2).unwrap();
    let completed = settled_completed(&f, e2.0 + 1);
    let h2_frame = reusable(&pool, H2_FRAME);
    eprintln!(
        "OWED_INTERLEAVED api={} completed={completed} e2={} e4={} h4_submitted={} owed={} h2_frame_reusable={h2_frame}",
        f.commands.actual_command_api(),
        e2.0,
        e4.0,
        m4.is_submitted(),
        owed_now(&f)
    );
    assert!(!m4.is_submitted(), "H4 ran without its producer");
    assert_eq!(completed, e2.0 + 1, "H2's completion did not complete H2");
    assert!(!clock.is_complete(e3) && !clock.is_complete(e4));
    assert!(
        h2_frame,
        "memory H2 read stayed parked behind H4's producer"
    );
    assert_eq!(owed_now(&f), 1);
    assert_eq!(raise(&f, &e), 1);
    f.commands.wait_marker(&m4).unwrap();
    assert_eq!(settled_completed(&f, e4.0 + 1), e4.0 + 1);
    assert_eq!(clock.current_epoch().0, e4.0 + 1);
    assert_eq!(owed_now(&f), 0);
    f.commands.check_failed().unwrap();
}
