use crate::kernels::{
    Buffer, CommandQueue, ComputePipeline, Function, Library, MTLResourceOptions, MetalKernelError,
};
use objc2::{rc::Retained, runtime::AnyObject, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice};
use std::{ffi::c_void, ptr};

/// Metal device type classification based on Apple Silicon architecture.
///
/// MLX uses the last character of the architecture name to determine device type:
/// - 'p': phone (iPhone, small device)
/// - 'g': base/pro (M1/M2/M3 base and Pro variants)
/// - 's': max (M1/M2/M3 Max)
/// - 'd': ultra (M1/M2 Ultra)
///
/// Reference: refs/mlx/mlx/backend/metal/device.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalDeviceType {
    /// Small device (iPhone, 'p' suffix)
    Phone,
    /// Base/Pro device (M1/M2/M3 base and Pro, 'g' suffix)
    BasePro,
    /// Max device (M1/M2/M3 Max, 's' suffix)
    Max,
    /// Ultra device (M1/M2 Ultra, 'd' suffix)
    Ultra,
    /// Unknown or medium device (default)
    Medium,
}

/// Whether new pipelines are built with `supportIndirectCommandBuffers`.
static PIPELINES_SUPPORT_ICB: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether any pipeline has been built yet, so the switch above can refuse to
/// change under one.
static ANY_PIPELINE_BUILT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Build subsequent pipelines with `supportIndirectCommandBuffers` (issue #115).
///
/// Must be called before any pipeline is built. `Kernels` caches pipelines per
/// `(Source, name, constants)` and hands the cached one back forever, so a
/// pipeline created before this switch is set stays unsupported for the life of
/// the process -- and encoding *that* into an ICB is §3.7d's segfault, at encode
/// time, with no error to catch. Returns an error rather than flipping the
/// switch and hoping, because "the flag was set but some pipelines predate it"
/// is indistinguishable from success until the crash.
pub fn set_pipelines_support_icb(enabled: bool) -> Result<(), MetalKernelError> {
    use std::sync::atomic::Ordering;
    if ANY_PIPELINE_BUILT.load(Ordering::Relaxed)
        && PIPELINES_SUPPORT_ICB.load(Ordering::Relaxed) != enabled
    {
        return Err(MetalKernelError::FailedToCreatePipeline(
            "supportIndirectCommandBuffers must be selected before any pipeline is built: \
             pipelines are cached per (Source, name, constants) and the ones already built \
             would keep the old setting, which is DESIGN.md §3.7d's segfault rather than a \
             wrong answer"
                .to_string(),
        ));
    }
    PIPELINES_SUPPORT_ICB.store(enabled, Ordering::Relaxed);
    Ok(())
}

/// Whether new pipelines will carry `supportIndirectCommandBuffers`.
pub fn pipelines_support_icb() -> bool {
    PIPELINES_SUPPORT_ICB.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record that a pipeline exists, so [`set_pipelines_support_icb`] can refuse to
/// change afterwards.
pub(crate) fn note_pipeline_built() {
    ANY_PIPELINE_BUILT.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl AsRef<ProtocolObject<dyn MTLDevice>> for Device {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

impl Device {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Device { raw }
    }

    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    /// Returns all Metal devices in the system.
    pub fn all() -> Vec<Self> {
        MTLCopyAllDevices()
            .to_vec()
            .into_iter()
            .map(|raw| Device { raw })
            .collect()
    }

    /// Returns the system default Metal device, if available.
    ///
    /// Falls back to first device from `all` if `MTLCreateSystemDefaultDevice`
    /// returns nil.
    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice()
            .map(|raw| Device { raw })
            .or_else(|| Device::all().first().cloned())
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(Buffer::new)
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Buffer".to_string(),
            ))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void)
            .ok_or_else(|| MetalKernelError::InvalidInput("Null pointer".to_string()))?;
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(Buffer::new)
                .ok_or(MetalKernelError::FailedToCreateResource(
                    "Buffer".to_string(),
                ))
        }
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library, MetalKernelError> {
        let raw = self
            .as_ref()
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?;

        Ok(Library::new(raw))
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, MetalKernelError> {
        note_pipeline_built();
        if pipelines_support_icb() {
            return self.new_compute_pipeline_state_supporting_icb(function);
        }
        let raw = self
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
        Ok(ComputePipeline::new(raw))
    }

    /// As above, but built through a descriptor with
    /// `supportIndirectCommandBuffers` set.
    ///
    /// # Why this is not simply always on
    ///
    /// The flag is a property of the *pipeline*, so it has to be decided when
    /// the pipeline is built, which is long before anyone knows whether a given
    /// kernel will be encoded into an ICB. Setting it unconditionally would
    /// change every pipeline in every process on the strength of an opt-in path
    /// nothing takes by default, and Apple documents it as a constraint the
    /// compiler may honour by generating different code. So it follows the same
    /// discipline as every other axis here: off unless asked for.
    ///
    /// # Why forgetting it is not a graceful failure
    ///
    /// `DESIGN.md` §3.7d, measured on this machine: a pipeline without the flag,
    /// encoded into an ICB by the CPU-side route this executor uses, **segfaults
    /// inside `setComputePipelineState:`** at encode time. Issue #32 originally
    /// recorded it as silent all-zero output; #35 measured otherwise, and the
    /// GPU-side route hangs the device instead. None of the three is an error
    /// return, so there is nothing to propagate and no test that can survive
    /// getting this wrong -- which is why the ICB executor refuses to install
    /// unless this switch was set before any pipeline was built.
    fn new_compute_pipeline_state_supporting_icb(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let desc = objc2_metal::MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function.as_ref()));
        desc.setSupportIndirectCommandBuffers(true);
        let raw = self
            .as_ref()
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                objc2_metal::MTLPipelineOption::None,
                None,
            )
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
        Ok(ComputePipeline::new(raw))
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, MetalKernelError> {
        self.as_ref()
            .newCommandQueue()
            .ok_or_else(|| MetalKernelError::FailedToCreateResource("CommandQueue".to_string()))
    }

    pub fn recommended_max_working_set_size(&self) -> usize {
        self.as_ref().recommendedMaxWorkingSetSize() as usize
    }

    pub fn current_allocated_size(&self) -> usize {
        self.as_ref().currentAllocatedSize()
    }

    /// Get the device architecture name (e.g., "applegpu_g13g", "applegpu_g14d").
    ///
    /// This returns the full architecture string from the Metal device.
    /// The last character indicates the device type:
    /// - 'p': phone
    /// - 'g': base/pro
    /// - 's': max
    /// - 'd': ultra
    pub fn architecture_name(&self) -> String {
        // On tvOS/iOS simulators the emulated Metal device returns NULL from
        // -[MTLDevice architecture], which causes objc2 to panic.  Guard
        // against this by checking the raw pointer before dereferencing.
        let raw_arch: *const AnyObject = unsafe { objc2::msg_send![self.as_ref(), architecture] };
        if raw_arch.is_null() {
            return "unknown".to_string();
        }
        let arch = self.as_ref().architecture();
        arch.name().to_string()
    }

    /// Get the device type based on architecture name.
    ///
    /// This implements the same logic as MLX's device type detection.
    /// Reference: refs/mlx/mlx/backend/metal/device.cpp
    pub fn device_type(&self) -> MetalDeviceType {
        // Cached per device (keyed on `registryID`): `mlx_gemm` asks on EVERY
        // GEMM dispatch, and the answer costs `-[MTLDevice architecture]` plus
        // its name as an `NSString` -- two autoreleased objects per matmul that
        // leaked on a thread with no pool (665 `MTLArchitecture` per 60 tokens,
        // `measurements/issue-706-raw/probes.log`) and a String allocation on
        // the hot path either way. The architecture of a device does not change.
        static CACHE: std::sync::OnceLock<(u64, MetalDeviceType)> = std::sync::OnceLock::new();
        let id = self.registry_id();
        if let Some((cached_id, ty)) = CACHE.get() {
            if *cached_id == id {
                return *ty;
            }
        }
        let ty = self.device_type_uncached();
        let _ = CACHE.set((id, ty));
        ty
    }

    fn device_type_uncached(&self) -> MetalDeviceType {
        let arch = self.architecture_name();
        match arch.chars().last() {
            Some('p') => MetalDeviceType::Phone,
            Some('g') => MetalDeviceType::BasePro,
            Some('s') => MetalDeviceType::Max,
            Some('d') => MetalDeviceType::Ultra,
            _ => MetalDeviceType::Medium,
        }
    }
}
