use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Kernel function name, when it was recorded at load time.
    ///
    /// An executor that validates a recorded dispatch sequence before replaying
    /// it (`DESIGN.md` §11.1a) has to compare *what* ran, and
    /// `MTLComputePipelineState` gives no way to read the function back: its
    /// `label` is settable but candle never sets it, and reading it through the
    /// Obj-C bridge per dispatch would cost more than keeping the name that was
    /// already in hand at load time.
    ///
    /// `Arc<str>` rather than `String` because pipelines are cloned out of the
    /// cache on every kernel call, and cloning a refcount is free where cloning
    /// the name would not be.
    name: Option<Arc<str>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> ComputePipeline {
        ComputePipeline { raw, name: None }
    }

    /// Attach the kernel function name this pipeline was built from.
    pub fn with_name(mut self, name: impl AsRef<str>) -> ComputePipeline {
        self.name = Some(Arc::from(name.as_ref()));
        self
    }

    /// The kernel function name, if it was recorded at load time.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The kernel function name as a shared handle, if it was recorded.
    ///
    /// The borrowing [`Self::name`] above is the right shape for a caller that
    /// only reads; this one is for the caller that has to *keep* the name past
    /// the pipeline's borrow. Cloning the `Arc` bumps a refcount where
    /// `Arc::from(self.name())` would allocate and copy the bytes a second
    /// time -- and the field is already an `Arc<str>` for exactly the reason
    /// this method exists (see its doc comment above).
    ///
    /// `encoder::note_pipeline` is the hot consumer: under
    /// `CANDLE_METAL_PROFILE=1` it fires once per dispatch, ~677 per decode
    /// token (`DESIGN.md` §11.2), and it was re-allocating at every one.
    pub fn name_handle(&self) -> Option<Arc<str>> {
        self.name.clone()
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
