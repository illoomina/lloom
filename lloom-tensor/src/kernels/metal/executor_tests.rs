//! That the executor seam sees every dispatch, and that the default changes
//! nothing.
use crate::kernels::metal::{
    Commands, ComputePipeline, DispatchRecord, Executor, ExecutorSlot, Grid, ResidencySet,
};
use crate::kernels::{call_unary_contiguous, unary, BufferOffset, Device, Kernels};
use std::sync::{Arc, Mutex};

fn device() -> Device {
    Device::system_default().unwrap()
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

/// Existing process-level tuning authority, not an in-test API override.
/// Broad gates keep every scenario; the native matrix runs it on both APIs.
fn assert_process_api(commands: &Commands) {
    let actual = commands.actual_command_api();
    let requested = &crate::tuning::tuning().metal.command_api;
    if requested != "auto" {
        assert_eq!(actual, requested);
    }
    if let Ok(requested) = std::env::var("LLOOM_METAL_API") {
        if requested != "auto" {
            assert_eq!(actual, requested);
        }
    }
    eprintln!("COMMON_EXECUTOR_SETUP requested={requested} actual_api={actual}");
}

/// Records what it is shown and forwards everything, so a run under it must
/// produce the same numbers as a run without it.
#[derive(Default)]
struct Recorder {
    dispatches: Mutex<Vec<DispatchRecord>>,
    binds: Mutex<Vec<(usize, usize, bool)>>,
    pipelines: Mutex<Vec<String>>,
}

impl Executor for Recorder {
    fn dispatch(&self, record: &DispatchRecord) -> bool {
        self.dispatches.lock().unwrap().push(record.clone());
        true
    }

    fn will_bind_buffer(
        &self,
        index: usize,
        _buffer: &crate::kernels::Buffer,
        offset: usize,
        out: bool,
    ) {
        self.binds.lock().unwrap().push((index, offset, out));
    }

    fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        self.pipelines
            .lock()
            .unwrap()
            .push(pipeline.name().unwrap_or("<unnamed>").to_string());
    }
}

/// Run `cos` over `input` and return the result, optionally through `executor`.
fn run_cos(executor: Option<Arc<ExecutorSlot>>, input: &[f32]) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    assert_process_api(&cmds);
    if let Some(e) = executor {
        cmds.set_executor(e);
    }

    let options = crate::kernels::RESOURCE_OPTIONS;
    let bytes = std::mem::size_of_val(input);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    let dst = device.new_buffer(bytes, options).unwrap();

    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_unary_contiguous(
                &device,
                &guard,
                &kernels,
                unary::contiguous::cos::FLOAT,
                std::mem::size_of::<f32>(),
                input.len(),
                BufferOffset::zero_offset(&src),
                &dst,
            )
        }
        .unwrap();
    }
    cmds.flush_and_wait().unwrap();

    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let ptr = unsafe { dst.contents() } as *const f32;
    // SAFETY: the command buffer has completed and `dst` holds `input.len()`
    // f32 in shared storage.
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

/// The seam sees the dispatch, its grid, and the kernel that ran -- and the
/// numbers are unchanged, which is the property `DESIGN.md` §11.1 calls "must
/// not regress" stated at unit scale.
#[test]
fn executor_observes_dispatches_without_changing_results() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 8.0).collect();

    let baseline = run_cos(None, &input);
    let recorder = Arc::new(Recorder::default());
    let observed = run_cos(
        Some(Arc::new(ExecutorSlot::Custom(recorder.clone()))),
        &input,
    );

    assert_eq!(
        baseline, observed,
        "installing a forwarding executor must not change results"
    );

    let dispatches = recorder.dispatches.lock().unwrap();
    assert_eq!(dispatches.len(), 1, "expected exactly one dispatch");
    assert_eq!(
        dispatches[0].kernel.as_deref(),
        Some(unary::contiguous::cos::FLOAT.0),
        "dispatch should be attributed to the kernel that ran"
    );
    assert!(
        matches!(dispatches[0].grid, Grid::Threadgroups(_)),
        "call_unary_contiguous dispatches threadgroups, got {:?}",
        dispatches[0].grid
    );

    assert!(
        !recorder.binds.lock().unwrap().is_empty(),
        "the executor should see buffer binds"
    );
    assert_eq!(
        recorder.pipelines.lock().unwrap().as_slice(),
        &[unary::contiguous::cos::FLOAT.0.to_string()],
    );
}

/// Returning `false` suppresses the dispatch, which is what a record-only or
/// replaying executor needs. Checked by the output *not* being written: if
/// suppression did not work the buffer would hold `cos(x)`.
#[test]
fn executor_can_suppress_a_dispatch() {
    struct Suppress;
    impl Executor for Suppress {
        fn dispatch(&self, _record: &DispatchRecord) -> bool {
            false
        }
    }

    let input: Vec<f32> = (0..64).map(|i| i as f32 / 8.0).collect();
    let out = run_cos(
        Some(Arc::new(ExecutorSlot::Custom(Arc::new(Suppress)))),
        &input,
    );
    let computed = run_cos(None, &input);

    assert_ne!(
        out, computed,
        "a suppressed dispatch must not produce the computed result"
    );
    assert!(
        out.iter().all(|&v| v == 0.0),
        "a fresh Metal buffer reads as zero, so suppression should leave zeros; got {:?}",
        &out[..8]
    );
}

/// The default slot is the classical path, and `is_classical` is what the
/// encoder branches on per dispatch -- so it must stay true by default or the
/// no-regression argument silently stops applying.
#[test]
fn default_executor_is_classical() {
    assert!(ExecutorSlot::default().is_classical());
    let device = device();
    assert!(commands(&device).executor().is_classical());
}

/// An executor is refused on the unretained kind only: its indirect command
/// buffers are not held for a command buffer that does not retain them. A
/// retaining queue carrying the unlisting hold (`kernels.residency.metal4_cover
/// = held`, §15.1r) is not that kind, and the refusal keys on the kind, not on
/// the hold.
#[test]
fn an_executor_is_refused_on_the_unretained_kind_only() {
    use crate::kernels::metal::commands::CommandBufferKind;
    use crate::kernels::metal::completion::CompletionWait;
    let device = device();
    let refused = |kind| {
        let queue = device.new_command_queue().unwrap();
        let residency_set = Arc::new(ResidencySet::new(&device));
        let cmds =
            Commands::new_with(queue, residency_set, CompletionWait::Synchronize, kind).unwrap();
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(Arc::new(Recorder::default()))));
        let refused = matches!(
            cmds.command_encoder(),
            Err(crate::kernels::MetalKernelError::ArmRefused(_))
        );
        (refused, cmds.in_flight_hold().is_some())
    };
    let (unretained, unretained_hold) = refused(CommandBufferKind::Unretained);
    assert!(unretained && unretained_hold, "the unretained kind refuses it");
    let (retaining, _) = refused(CommandBufferKind::Retaining);
    assert!(!retaining, "a retaining queue runs it, a hold or none");
}


/// Positive native witness for the same two ComputeCommandEncoder raw sites
/// observed by the refusal control. Run in its own exact-name process.
#[test]
fn common_raw_observer_counts_one_real_classical_dispatch() {
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 8.0).collect();
    let (out, raw) = super::encoder::observe_raw_dispatches_for_test(|| run_cos(None, &input));
    assert_eq!(raw, 1, "disconnected raw observer cannot prove refusal zeroes");
    assert_eq!(out.len(), input.len());
    assert_eq!(out[0], 1.0, "the genuine cos kernel did not write its known value");
    assert_ne!(out[0], out[8], "positive output is constant or uninitialized");
}

/// Refusal is an exceptional disposition, not an instruction to encode the
/// next offered dispatch or flush a later mismatch. The owner queue must fail.
#[test]
fn common_checked_refusal_suppresses_later_encode_and_owner_submission() {
    use super::encoder::observe_raw_dispatches_for_test;
    use super::executor::DispatchAction;
    use crate::kernels::MetalKernelError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct RefuseFirst {
        offers: AtomicUsize,
        fails: AtomicUsize,
    }
    impl Executor for RefuseFirst {
        fn dispatch_action(&self, _: &DispatchRecord) -> Result<DispatchAction, MetalKernelError> {
            if self.offers.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(MetalKernelError::QueueUnavailable("controlled common refusal"))
            } else {
                // The causal omission of the pre-offer latch would reach this
                // healthy Encode. Its real output is still in-bounds.
                Ok(DispatchAction::Encode)
            }
        }
        fn fail_step(&self) { self.fails.fetch_add(1, Ordering::Relaxed); }
    }

    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    assert_process_api(&cmds);
    let input = [0.25f32; 64];
    let zeros = [0f32; 64];
    let options = crate::kernels::RESOURCE_OPTIONS;
    let src = device.new_buffer_with_data(input.as_ptr().cast(), std::mem::size_of_val(&input), options).unwrap();
    let dst = device.new_buffer_with_data(zeros.as_ptr().cast(), std::mem::size_of_val(&zeros), options).unwrap();
    let probe = Arc::new(RefuseFirst::default());
    cmds.set_executor(Arc::new(ExecutorSlot::Custom(probe.clone())));
    // Intentional, bounded process-lifetime custody of this poisoned fixture.
    // Retain BEFORE the first failure/assertion, including mutant unwind.
    let fixture = Box::leak(Box::new((device, kernels, cmds, src, dst, probe)));
    let (device, kernels, cmds, src, dst, probe) = fixture;
    let before_completed = cmds.clock().completed_count();
    let before_current = cmds.clock().current_epoch();
    let guard = cmds.command_encoder().unwrap();
    let ((), raw) = observe_raw_dispatches_for_test(|| {
        for _ in 0..2 {
            // SAFETY: both real allocations hold all 64 F32 elements; even a
            // causal extra-Encode mutant remains in-bounds. No host read races.
            unsafe {
                call_unary_contiguous(device, &guard, kernels,
                    unary::contiguous::cos::FLOAT, std::mem::size_of::<f32>(), input.len(),
                    BufferOffset::zero_offset(src), dst)
            }.unwrap();
        }
    });
    assert_eq!(probe.offers.load(Ordering::Relaxed), 1,
        "a later Encode was offered after the first recording refusal");
    assert_eq!(probe.fails.load(Ordering::Relaxed), 2,
        "both failed offers must detach through the captured executor");
    assert_eq!(raw, 0, "checked refusal or later Encode entered raw dispatch");
    assert!(cmds.check_recording_failure().is_err(), "genuine recording latch was not published");
    drop(guard);
    assert!(cmds.flush_marker().is_err(), "owner submitted the poisoned recording");
    assert_eq!(cmds.clock().current_epoch(), before_current, "failed owner reserved a submission epoch");
    assert_eq!(cmds.clock().completed_count(), before_completed, "known-unissued refusal fabricated completion");
    eprintln!("COMMON_REFUSAL actual_api={} offers=1 fail_step=2 raw=0 owner_refused=true", cmds.actual_command_api());
}
