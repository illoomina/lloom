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
    Commands::new(queue, &residency_set).unwrap()
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
        .unwrap();
    }
    cmds.flush_and_wait().unwrap();

    let ptr = dst.contents() as *const f32;
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
