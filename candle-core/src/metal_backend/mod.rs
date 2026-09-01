//! Implementation of Backend traits for Metal
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, CpuStorageRef, DType, Error, Layout, Result, Shape};
use candle_metal_kernels::kernels::binary::contiguous;
use candle_metal_kernels::{
    metal::{Buffer, BufferPool, Commands, Device, PooledBuffer, ResidencySet},
    BufferOffset, CallConvTranspose2dCfg, ConvKernel, IndexingKernel, Kernels, RESOURCE_OPTIONS,
};
use objc2_foundation::NSRange;
#[cfg(feature = "metal-debug-labels")]
use objc2_foundation::NSString;
#[cfg(feature = "metal-debug-labels")]
use objc2_metal::MTLCommandQueue;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, PoisonError, RwLock, TryLockError};

mod device;
pub use device::{DeviceId, MetalDevice};
use device::{DeviceTeardown, ResidencyEvictionObserver};

/// The activation arena's public vocabulary (`DESIGN.md` §9.2).
///
/// Re-exported for the same reason `trace` is below: a harness selecting a
/// layout should not need a direct dependency on `candle-metal-kernels`.
pub use candle_metal_kernels::metal::{
    hazard_key, set_hazard_key, ArenaCounters, ArenaLayout, HazardKey, Slot, StepPlan,
};

/// Where a command buffer is cut (`DESIGN.md` §13.5m, issue #420 stage 3).
///
/// Re-exported for the same reason `ArenaLayout` above is: a harness stating
/// the arm on its `RESULT` line should not need a direct dependency on
/// `candle-metal-kernels`. Unlike `hazard_key()` beside it this is **not**
/// process-global — it is resolved once per `Commands` — so it is read from the
/// device (`MetalDevice::cut_policy`) rather than from a free function, and a
/// harness that reads the environment variable instead is reading the flag
/// rather than the mechanism (§7.1a-i).
pub use candle_metal_kernels::metal::CutPolicy;

/// The memory budget: predict the peak before allocating, and refuse (§9.5).
///
/// Re-exported for the same reason `trace` is: a model deciding whether its
/// configuration fits should not need a direct dependency on
/// `candle-metal-kernels`. It is the model that knows its own geometry, and it
/// is `candle-core` that both it and the pools already depend on.
pub use candle_metal_kernels::metal::admission;

/// Per-dispatch recording of the Metal command stream.
///
/// Re-exported so a measurement harness can drive it without taking a direct
/// dependency on `candle-metal-kernels`. Inert unless `CANDLE_METAL_TRACE=1`.
pub use candle_metal_kernels::metal::trace;

/// Opt-in GPU-time and dispatch-count profiling of the Metal submission path.
///
/// Re-exported so a consumer that already depends on `candle-core` can read the
/// counters without taking a direct dependency on `candle-metal-kernels`, which
/// sits outside the workspace and is not otherwise in an example's dependency
/// graph.
pub use candle_metal_kernels::metal::profile;

/// Restore the blit in [`MetalStorage::to_cpu`], for A/B and for a fallback.
///
/// `CANDLE_METAL_TO_CPU_BLIT=1` selects the pre-#336 path. Off by default, so
/// the fast path is what ships and what every recorded figure after this change
/// belongs to -- but the old arm stays *compiled and selectable* rather than
/// deleted, which is §12.3's rule (variants coexist) and what makes the
/// comparison in `measurements/issue-336-*` re-runnable rather than historical.
///
/// Read once through a `OnceLock`: this sits on the per-token path, and #69's
/// vacuous determinism run is what a per-call `var()` risks (§2.4).
/// Engagement counters for #336's two `to_cpu` paths (§2.4: an instrument that
/// cannot be shown to have engaged has measured nothing). Free: two relaxed
/// increments on a path that already waits on the GPU.
pub static TO_CPU_FAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TO_CPU_BLIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(fast, blit)` readback counts since process start.
pub fn to_cpu_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (TO_CPU_FAST.load(Relaxed), TO_CPU_BLIT.load(Relaxed))
}

pub fn to_cpu_force_blit() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| {
        std::env::var("CANDLE_METAL_TO_CPU_BLIT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// #366: readbacks whose buffer was null *after* the wait, and so took the blit.
///
/// This is the case that used to segfault. It is a counter rather than an
/// assertion because the fallback is correct — the blit computes the same bytes
/// — and because a number on the `RESULT` line is what makes the frequency
/// visible instead of being discovered by a crash report. **A run with
/// `to_cpu_fast > 0` and this at 0 is the healthy shape**; a nonzero value says
/// the window this fix closes was live on that run.
pub static TO_CPU_NULL_AFTER_WAIT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Readbacks that found the buffer unmapped after the GPU wait (#366).
pub fn to_cpu_null_after_wait() -> u64 {
    TO_CPU_NULL_AFTER_WAIT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn buffer_o<'a>(buffer: &'a Buffer, l: &Layout, dtype: DType) -> BufferOffset<'a> {
    BufferOffset {
        buffer,
        offset_in_bytes: l.start_offset() * dtype.size_in_bytes(),
    }
}

/// Resolve a `conv.metal` kernel name for `dtype`, or report the op as
/// unimplemented for it.
///
/// The name comes from [`ConvKernel`], which is the single Rust-side
/// declaration of the conv family's `[[host_name]]` strings and is checked
/// against the compiled Metal library by `conv_names_resolve` in
/// `candle-metal-kernels`. Previously each of these call sites carried its own
/// `match dtype` table of string literals, so a rename in `conv.metal` or a
/// dtype added on one side only surfaced as a runtime `LoadFunctionError`.
///
/// `op` is the caller's name for the error message, which is not always the
/// kernel stem — `conv1d` dispatches the `im2col1d` kernel, and reporting
/// "im2col1d not implemented" to a user who called `conv1d` would be worse than
/// the string duplication it saves.
fn conv_kernel_name(kernel: ConvKernel, dtype: DType, op: &'static str) -> Result<&'static str> {
    match kernel.name(dtype.as_str()) {
        Some(name) => Ok(name),
        None => crate::bail!("Metal {op} {dtype:?} not implemented"),
    }
}

/// Resolve an `indexing.metal` kernel name for an `(ids dtype, value dtype)`
/// pair, or report the op as unimplemented for it.
///
/// The `conv_kernel_name` above, for the family whose names were chosen *here*
/// rather than in `candle-metal-kernels`. Each of these five call sites carried
/// its own `match (ids.dtype, self.dtype)` table of string literals — 72 arms
/// between them — and nothing compared those spellings with the
/// `[[host_name]]`s `indexing.metal` declares.
///
/// **That had already gone wrong**: `is_i64_u8` and `is_i64_u32` were named here
/// and declared nowhere, so `index_select` on a `U8` or `U32` tensor with `I64`
/// indices failed at runtime. [`IndexingKernel`] is now the single Rust-side
/// declaration and `indexing_names_resolve` checks every name in it against the
/// compiled library.
///
/// Two dtypes rather than one, because an indexing kernel's name carries both —
/// `is_u32_f16` is `index` over an f16 tensor with u32 indices.
fn indexing_kernel_name(
    kernel: IndexingKernel,
    ids_dtype: DType,
    value_dtype: DType,
    op: &'static str,
) -> Result<&'static str> {
    match kernel.name(ids_dtype.as_str(), value_dtype.as_str()) {
        Some(name) => Ok(name),
        None => crate::bail!("Metal {op} {ids_dtype:?} {value_dtype:?} not implemented"),
    }
}
/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for MetalError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => MetalError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => MetalError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for MetalError {
    fn from(p: PoisonError<T>) -> Self {
        MetalError::LockError(LockError::Poisoned(p.to_string()))
    }
}

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),
    #[error("{0:?}")]
    LockError(LockError),
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    /// The actual buffer containing the data.
    buffer: Arc<PooledBuffer>,
    /// a reference to the device owning this buffer
    device: MetalDevice,
    /// The count of allocated elements in the buffer
    count: usize,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I16 => Ok(CpuStorage::I16(self.to_cpu()?)),
            DType::I32 => Ok(CpuStorage::I32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
            DType::F8E4M3 => Ok(CpuStorage::F8E4M3(self.to_cpu()?)),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(crate::Error::UnsupportedDTypeForOp(self.dtype, "to_cpu_storage").bt())
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device
            .new_buffer_builder()
            .with_size_for(el, self.dtype)
            .with_label("affine")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, dtype);
        if layout.is_contiguous() {
            let name = match self.dtype {
                DType::F32 => "affine_f32",
                DType::F16 => "affine_f16",
                DType::BF16 => "affine_bf16",
                DType::U8 => "affine_u8",
                DType::U32 => "affine_u32",
                DType::I64 => "affine_i64",
                dtype => crate::bail!("Metal contiguous affine {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_affine(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                self.dtype.size_in_bytes(),
                el,
                src,
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "affine_f32_strided",
                DType::F16 => "affine_f16_strided",
                DType::BF16 => "affine_bf16_strided",
                DType::U8 => "affine_u8_strided",
                DType::U32 => "affine_u32_strided",
                DType::I64 => "affine_i64_strided",
                dtype => crate::bail!("Metal strided affine {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_affine_strided(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                layout.dims(),
                src,
                layout.stride(),
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), el, dtype))
    }

    fn powf(&self, layout: &Layout, pow: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device
            .new_buffer_builder()
            .with_size_for(el, self.dtype)
            .with_label("powf")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, dtype);
        if layout.is_contiguous() {
            let name = match self.dtype {
                DType::F32 => "powf_f32",
                DType::F16 => "powf_f16",
                DType::BF16 => "powf_bf16",
                dtype => crate::bail!("Metal contiguous powf {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_powf(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                self.dtype.size_in_bytes(),
                el,
                src,
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "powf_f32_strided",
                DType::F16 => "powf_f16_strided",
                DType::BF16 => "powf_bf16_strided",
                dtype => crate::bail!("Metal strided powf {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_powf_strided(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                layout.dims(),
                src,
                layout.stride(),
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), el, dtype))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device
            .new_buffer_builder()
            .with_size_for(el, self.dtype)
            .with_label("elu")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, self.dtype);
        if layout.is_contiguous() {
            let name = match self.dtype {
                DType::F32 => "elu_f32",
                DType::F16 => "elu_f16",
                DType::BF16 => "elu_bf16",
                dtype => crate::bail!("Metal contiguous elu {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_elu(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                self.dtype.size_in_bytes(),
                el,
                src,
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "elu_f32_strided",
                DType::F16 => "elu_f16_strided",
                DType::BF16 => "elu_bf16_strided",
                dtype => crate::bail!("Metal strided elu {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_elu_strided(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                layout.dims(),
                src,
                layout.stride(),
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), el, dtype))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();

        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !sum_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }

        for &dim_idx in sum_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }

        let reduction_shape = Shape::from(dims.clone());

        if layout.is_contiguous() && reduction_shape.is_contiguous(&stride) {
            let (name, check_empty, return_index) = match (op, self.dtype) {
                (ReduceOp::Sum, DType::F32) => ("fast_sum_f32", false, false),
                (ReduceOp::Min, DType::F32) => ("fast_min_f32", true, false),
                (ReduceOp::Max, DType::F32) => ("fast_max_f32", true, false),
                (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_f32", true, true),
                (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_f32", true, true),
                (ReduceOp::Sum, DType::U32) => ("fast_sum_u32", false, false),
                (ReduceOp::Min, DType::U32) => ("fast_min_u32", true, false),
                (ReduceOp::Max, DType::U32) => ("fast_max_u32", true, false),
                (ReduceOp::ArgMin, DType::U32) => ("fast_argmin_u32", true, true),
                (ReduceOp::ArgMax, DType::U32) => ("fast_argmax_u32", true, true),
                (ReduceOp::Sum, DType::F16) => ("fast_sum_f16", false, false),
                (ReduceOp::Min, DType::F16) => ("fast_min_f16", true, false),
                (ReduceOp::Max, DType::F16) => ("fast_max_f16", true, false),
                (ReduceOp::ArgMin, DType::F16) => ("fast_argmin_f16", true, true),
                (ReduceOp::ArgMax, DType::F16) => ("fast_argmax_f16", true, true),
                (ReduceOp::Sum, DType::BF16) => ("fast_sum_bf16", false, false),
                (ReduceOp::Min, DType::BF16) => ("fast_min_bf16", true, false),
                (ReduceOp::Max, DType::BF16) => ("fast_max_bf16", true, false),
                (ReduceOp::ArgMin, DType::BF16) => ("fast_argmin_bf16", true, true),
                (ReduceOp::ArgMax, DType::BF16) => ("fast_argmax_bf16", true, true),
                (ReduceOp::Sum, DType::I64) => ("fast_sum_i64", false, false),
                (ReduceOp::Min, DType::I64) => ("fast_min_i64", true, false),
                (ReduceOp::Max, DType::I64) => ("fast_max_i64", true, false),
                (ReduceOp::ArgMin, DType::I64) => ("fast_argmin_i64", true, true),
                (ReduceOp::ArgMax, DType::I64) => ("fast_argmax_i64", true, true),
                (ReduceOp::Sum, DType::U8) => ("fast_sum_u8", false, false),
                (ReduceOp::Min, DType::U8) => ("fast_min_u8", true, false),
                (ReduceOp::Max, DType::U8) => ("fast_max_u8", true, false),
                (ReduceOp::ArgMin, DType::U8) => ("fast_argmin_u8", true, true),
                (ReduceOp::ArgMax, DType::U8) => ("fast_argmax_u8", true, true),
                (k, dtype) => {
                    crate::bail!("Metal contiguous reduce op {k:?} {dtype:?} not implemented")
                }
            };
            if check_empty && layout.shape().elem_count() == 0 {
                Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
            }
            let dtype = if return_index { DType::U32 } else { self.dtype };
            let buffer = device
                .new_buffer_builder()
                .with_size_for(dst_el, dtype)
                .with_label("reduce")
                .build()?;
            let encoder = self.device.command_encoder()?;
            let src = buffer_o(&self.buffer, layout, self.dtype);
            candle_metal_kernels::call_reduce_contiguous(
                &device.device,
                &encoder,
                &device.kernels,
                name,
                src_dims,
                dst_el,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;

            return Ok(Self::new(buffer, device, dst_el, dtype));
        }

        let (name, check_empty, return_index) = match (op, self.dtype) {
            (ReduceOp::Sum, DType::F32) => ("fast_sum_f32_strided", false, false),
            (ReduceOp::Min, DType::F32) => ("fast_min_f32_strided", true, false),
            (ReduceOp::Max, DType::F32) => ("fast_max_f32_strided", true, false),
            (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_f32_strided", true, true),
            (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_f32_strided", true, true),
            (ReduceOp::Sum, DType::U32) => ("fast_sum_u32_strided", false, false),
            (ReduceOp::Min, DType::U32) => ("fast_min_u32_strided", true, false),
            (ReduceOp::Max, DType::U32) => ("fast_max_u32_strided", true, false),
            (ReduceOp::ArgMin, DType::U32) => ("fast_argmin_u32_strided", true, true),
            (ReduceOp::ArgMax, DType::U32) => ("fast_argmax_u32_strided", true, true),
            (ReduceOp::Sum, DType::F16) => ("fast_sum_f16_strided", false, false),
            (ReduceOp::Min, DType::F16) => ("fast_min_f16_strided", true, false),
            (ReduceOp::Max, DType::F16) => ("fast_max_f16_strided", true, false),
            (ReduceOp::ArgMin, DType::F16) => ("fast_argmin_f16_strided", true, true),
            (ReduceOp::ArgMax, DType::F16) => ("fast_argmax_f16_strided", true, true),
            (ReduceOp::Sum, DType::BF16) => ("fast_sum_bf16_strided", false, false),
            (ReduceOp::Min, DType::BF16) => ("fast_min_bf16_strided", true, false),
            (ReduceOp::Max, DType::BF16) => ("fast_max_bf16_strided", true, false),
            (ReduceOp::ArgMin, DType::BF16) => ("fast_argmin_bf16_strided", true, true),
            (ReduceOp::ArgMax, DType::BF16) => ("fast_argmax_bf16_strided", true, true),
            (ReduceOp::Sum, DType::I64) => ("fast_sum_i64_strided", false, false),
            (ReduceOp::Min, DType::I64) => ("fast_min_i64_strided", true, false),
            (ReduceOp::Max, DType::I64) => ("fast_max_i64_strided", true, false),
            (ReduceOp::ArgMin, DType::I64) => ("fast_argmin_i64_strided", true, true),
            (ReduceOp::ArgMax, DType::I64) => ("fast_argmax_i64_strided", true, true),
            (ReduceOp::Sum, DType::U8) => ("fast_sum_u8_strided", false, false),
            (ReduceOp::Min, DType::U8) => ("fast_min_u8_strided", true, false),
            (ReduceOp::Max, DType::U8) => ("fast_max_u8_strided", true, false),
            (ReduceOp::ArgMin, DType::U8) => ("fast_argmin_u8_strided", true, true),
            (ReduceOp::ArgMax, DType::U8) => ("fast_argmax_u8_strided", true, true),
            (k, dtype) => crate::bail!("Metal strided reduce op {k:?} {dtype:?} not implemented"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        let buffer = device
            .new_buffer_builder()
            .with_size_for(dst_el, dtype)
            .with_label("reduce")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, self.dtype);
        candle_metal_kernels::call_reduce_strided(
            &device.device,
            &encoder,
            &device.kernels,
            name,
            &dims,
            &stride,
            dst_el,
            src,
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(buffer, device, dst_el, dtype))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let name = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
        };
        self.binary(name, rhs, lhs_l, rhs_l)
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        fn set<S: crate::WithDType + candle_metal_kernels::utils::EncoderParam>(
            self_: &mut MetalStorage,
            s: S,
            l: &Layout,
        ) -> Result<()> {
            let device = self_.device();
            let dtype = self_.dtype;
            let shape = l.shape();
            let el_count = shape.elem_count();
            let encoder = device.command_encoder()?;
            let dst = buffer_o(&self_.buffer, l, self_.dtype);

            if l.is_contiguous() {
                use candle_metal_kernels::unary::contiguous;
                let kernel_name = match dtype {
                    DType::F16 => contiguous::const_set::HALF,
                    DType::BF16 => contiguous::const_set::BFLOAT,
                    DType::F32 => contiguous::const_set::FLOAT,
                    DType::I64 => contiguous::const_set::I64,
                    DType::U32 => contiguous::const_set::U32,
                    DType::U8 => contiguous::const_set::U8,
                    DType::F8E4M3 => crate::bail!("unsupported const-set f8e4m3"),
                    DType::F64 => crate::bail!("unsupported const-set f64"),
                    DType::F4
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F8E8M0
                    | DType::I16
                    | DType::I32 => {
                        return Err(Error::UnsupportedDTypeForOp(dtype, "const-set").bt())
                    }
                };
                candle_metal_kernels::call_const_set_contiguous(
                    &device.device,
                    &encoder,
                    &device.kernels,
                    kernel_name,
                    dtype.size_in_bytes(),
                    el_count,
                    s,
                    dst,
                )
                .map_err(MetalError::from)?;
            } else {
                use candle_metal_kernels::unary::strided;
                let kernel_name = match dtype {
                    DType::F16 => strided::const_set::HALF,
                    DType::BF16 => strided::const_set::BFLOAT,
                    DType::F32 => strided::const_set::FLOAT,
                    DType::I64 => strided::const_set::I64,
                    DType::U32 => strided::const_set::U32,
                    DType::U8 => strided::const_set::U8,
                    DType::F8E4M3 => crate::bail!("unsupported const-set f8e4m3"),
                    DType::F64 => crate::bail!("unsupported const-set f64"),
                    DType::F4
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F8E8M0
                    | DType::I16
                    | DType::I32 => {
                        return Err(Error::UnsupportedDTypeForOp(dtype, "const-set").bt())
                    }
                };
                candle_metal_kernels::call_const_set_strided(
                    &device.device,
                    &encoder,
                    &device.kernels,
                    kernel_name,
                    l.dims(),
                    s,
                    l.stride(),
                    dst,
                )
                .map_err(MetalError::from)?;
            }
            Ok(())
        }
        match (self.dtype, s) {
            (DType::U8, Scalar::U8(s)) => set(self, s, l),
            (DType::U32, Scalar::U32(s)) => set(self, s, l),
            (DType::I64, Scalar::I64(s)) => set(self, s, l),
            (DType::F16, Scalar::F16(s)) => set(self, s, l),
            (DType::BF16, Scalar::BF16(s)) => set(self, s, l),
            (DType::F32, Scalar::F32(s)) => set(self, s, l),
            (DType::F64, Scalar::F64(s)) => set(self, s, l),
            _ => crate::bail!("dtype mismatch, expected {:?}, got {:?}", self.dtype, s),
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(el_count, dtype)
            .with_label("to_dtype")
            .build()?;
        let encoder = device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, self.dtype);
        if layout.is_contiguous() {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::BF16) => "cast_u32_bf16",
                (DType::U32, DType::F16) => "cast_u32_f16",
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::U32, DType::I64) => "cast_u32_i64",
                (DType::U32, DType::U8) => "cast_u32_u8",

                (DType::U8, DType::BF16) => "cast_u8_bf16",
                (DType::U8, DType::F16) => "cast_u8_f16",
                (DType::U8, DType::F32) => "cast_u8_f32",
                (DType::U8, DType::I64) => "cast_u8_i64",
                (DType::U8, DType::U32) => "cast_u8_u32",

                (DType::F32, DType::BF16) => "cast_f32_bf16",
                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F32, DType::I64) => "cast_f32_i64",
                (DType::F32, DType::U32) => "cast_f32_u32",
                (DType::F32, DType::U8) => "cast_f32_u8",

                (DType::I64, DType::BF16) => "cast_i64_bf16",
                (DType::I64, DType::F16) => "cast_i64_f16",
                (DType::I64, DType::F32) => "cast_i64_f32",
                (DType::I64, DType::U32) => "cast_i64_u32",
                (DType::I64, DType::U8) => "cast_i64_u8",

                (DType::F16, DType::BF16) => "cast_f16_bf16",
                (DType::F16, DType::F32) => "cast_f16_f32",
                (DType::F16, DType::I64) => "cast_f16_i64",
                (DType::F16, DType::U32) => "cast_f16_u32",
                (DType::F16, DType::U8) => "cast_f16_u8",

                (DType::BF16, DType::F16) => "cast_bf16_f16",
                (DType::BF16, DType::F32) => "cast_bf16_f32",
                (DType::BF16, DType::I64) => "cast_bf16_i64",
                (DType::BF16, DType::U32) => "cast_bf16_u32",
                (DType::BF16, DType::U8) => "cast_bf16_u8",

                (left, right) => {
                    crate::bail!("Metal contiguous to_dtype {left:?} {right:?} not implemented")
                }
            };
            candle_metal_kernels::call_cast_contiguous(
                &device.device,
                &encoder,
                &device.kernels,
                kernel_name,
                self.dtype.size_in_bytes(),
                el_count,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            let kernel_name = match (self.dtype, dtype) {
                (DType::BF16, DType::F16) => "cast_bf16_f16_strided",
                (DType::BF16, DType::F32) => "cast_bf16_f32_strided",
                (DType::BF16, DType::I64) => "cast_bf16_i64_strided",
                (DType::BF16, DType::U32) => "cast_bf16_u32_strided",
                (DType::BF16, DType::U8) => "cast_bf16_u8_strided",

                (DType::F16, DType::BF16) => "cast_f16_bf16_strided",
                (DType::F16, DType::F32) => "cast_f16_f32_strided",
                (DType::F16, DType::I64) => "cast_f16_i64_strided",
                (DType::F16, DType::U32) => "cast_f16_u32_strided",
                (DType::F16, DType::U8) => "cast_f16_u8_strided",

                (DType::F32, DType::BF16) => "cast_f32_bf16_strided",
                (DType::F32, DType::F16) => "cast_f32_f16_strided",
                (DType::F32, DType::I64) => "cast_f32_i64_strided",
                (DType::F32, DType::U32) => "cast_f32_u32_strided",
                (DType::F32, DType::U8) => "cast_f32_u8_strided",

                (DType::I64, DType::F32) => "cast_i64_f32_strided",
                (DType::I64, DType::BF16) => "cast_i64_bf16_strided",
                (DType::I64, DType::F16) => "cast_i64_f16_strided",
                (DType::I64, DType::U32) => "cast_i64_u32_strided",
                (DType::I64, DType::U8) => "cast_i64_u8_strided",

                (DType::U32, DType::BF16) => "cast_u32_bf16_strided",
                (DType::U32, DType::F16) => "cast_u32_f16_strided",
                (DType::U32, DType::F32) => "cast_u32_f32_strided",
                (DType::U32, DType::I64) => "cast_u32_i64_strided",
                (DType::U32, DType::U8) => "cast_u32_u8_strided",

                (DType::U8, DType::BF16) => "cast_u8_bf16_strided",
                (DType::U8, DType::F16) => "cast_u8_f16_strided",
                (DType::U8, DType::F32) => "cast_u8_f32_strided",
                (DType::U8, DType::I64) => "cast_u8_i64_strided",
                (DType::U8, DType::U32) => "cast_u8_u32_strided",

                (left, right) => {
                    crate::bail!("Metal strided to_dtype {left:?} {right:?} not implemented")
                }
            };
            candle_metal_kernels::call_cast_strided(
                &device.device,
                &encoder,
                &device.kernels,
                kernel_name,
                layout.dims(),
                src,
                layout.stride(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), el_count, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(el_count, dtype)
            .with_label(B::KERNEL)
            .build()?;
        let encoder = device.command_encoder()?;
        let src = buffer_o(&self.buffer, layout, self.dtype);

        if layout.is_contiguous() {
            use candle_metal_kernels::unary::contiguous;
            let kernel_name = match (B::KERNEL, dtype) {
                ("uabs", DType::F16) => contiguous::abs::HALF,
                ("uabs", DType::F32) => contiguous::abs::FLOAT,
                ("uabs", DType::BF16) => contiguous::abs::BFLOAT,
                ("uceil", DType::F16) => contiguous::ceil::HALF,
                ("uceil", DType::F32) => contiguous::ceil::FLOAT,
                ("uceil", DType::BF16) => contiguous::ceil::BFLOAT,
                ("ucos", DType::F16) => contiguous::cos::HALF,
                ("ucos", DType::F32) => contiguous::cos::FLOAT,
                ("ucos", DType::BF16) => contiguous::cos::BFLOAT,
                ("uerf", DType::F16) => contiguous::erf::HALF,
                ("uerf", DType::F32) => contiguous::erf::FLOAT,
                ("uerf", DType::BF16) => contiguous::erf::BFLOAT,
                ("uexp", DType::F16) => contiguous::exp::HALF,
                ("uexp", DType::F32) => contiguous::exp::FLOAT,
                ("uexp", DType::BF16) => contiguous::exp::BFLOAT,
                ("ufloor", DType::F16) => contiguous::floor::HALF,
                ("ufloor", DType::F32) => contiguous::floor::FLOAT,
                ("ufloor", DType::BF16) => contiguous::floor::BFLOAT,
                ("ugelu_erf", DType::F16) => contiguous::gelu_erf::HALF,
                ("ugelu_erf", DType::F32) => contiguous::gelu_erf::FLOAT,
                ("ugelu_erf", DType::BF16) => contiguous::gelu_erf::BFLOAT,
                ("ugelu", DType::F16) => contiguous::gelu::HALF,
                ("ugelu", DType::F32) => contiguous::gelu::FLOAT,
                ("ugelu", DType::BF16) => contiguous::gelu::BFLOAT,
                ("ulog", DType::F16) => contiguous::log::HALF,
                ("ulog", DType::F32) => contiguous::log::FLOAT,
                ("ulog", DType::BF16) => contiguous::log::BFLOAT,
                ("uneg", DType::F16) => contiguous::neg::HALF,
                ("uneg", DType::F32) => contiguous::neg::FLOAT,
                ("uneg", DType::BF16) => contiguous::neg::BFLOAT,
                ("urecip", DType::F16) => contiguous::recip::HALF,
                ("urecip", DType::F32) => contiguous::recip::FLOAT,
                ("urecip", DType::BF16) => contiguous::recip::BFLOAT,
                ("urelu", DType::F16) => contiguous::relu::HALF,
                ("urelu", DType::F32) => contiguous::relu::FLOAT,
                ("urelu", DType::BF16) => contiguous::relu::BFLOAT,
                ("uround", DType::F16) => contiguous::round::HALF,
                ("uround", DType::F32) => contiguous::round::FLOAT,
                ("uround", DType::BF16) => contiguous::round::BFLOAT,
                ("usilu", DType::F16) => contiguous::silu::HALF,
                ("usilu", DType::F32) => contiguous::silu::FLOAT,
                ("usilu", DType::BF16) => contiguous::silu::BFLOAT,
                ("usin", DType::F16) => contiguous::sin::HALF,
                ("usin", DType::F32) => contiguous::sin::FLOAT,
                ("usin", DType::BF16) => contiguous::sin::BFLOAT,
                ("usqr", DType::F16) => contiguous::sqr::HALF,
                ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                ("usqr", DType::BF16) => contiguous::sqr::BFLOAT,
                ("usqrt", DType::F16) => contiguous::sqrt::HALF,
                ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                ("usqrt", DType::BF16) => contiguous::sqrt::BFLOAT,
                ("utanh", DType::F16) => contiguous::tanh::HALF,
                ("utanh", DType::F32) => contiguous::tanh::FLOAT,
                ("utanh", DType::BF16) => contiguous::tanh::BFLOAT,
                ("usign", DType::F16) => contiguous::sign::HALF,
                ("usign", DType::F32) => contiguous::sign::FLOAT,
                ("usign", DType::BF16) => contiguous::sign::BFLOAT,
                ("usign", DType::I64) => contiguous::sign::I64,
                (name, dtype) => {
                    crate::bail!("Metal contiguous unary {name} {dtype:?} not implemented")
                }
            };

            candle_metal_kernels::call_unary_contiguous(
                &device.device,
                &encoder,
                &device.kernels,
                kernel_name,
                dtype.size_in_bytes(),
                el_count,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => strided::cos::FLOAT,
                ("usin", DType::F32) => strided::sin::FLOAT,
                ("usqr", DType::F32) => strided::sqr::FLOAT,
                ("usqrt", DType::F32) => strided::sqrt::FLOAT,
                ("uneg", DType::F32) => strided::neg::FLOAT,
                ("uexp", DType::F32) => strided::exp::FLOAT,
                ("ulog", DType::F32) => strided::log::FLOAT,
                ("ugelu", DType::F32) => strided::gelu::FLOAT,
                ("ugelu_erf", DType::F32) => strided::gelu_erf::FLOAT,
                ("uerf", DType::F32) => strided::erf::FLOAT,
                ("usilu", DType::F32) => strided::silu::FLOAT,
                ("uabs", DType::F32) => strided::abs::FLOAT,
                ("uceil", DType::F32) => strided::ceil::FLOAT,
                ("ufloor", DType::F32) => strided::floor::FLOAT,
                ("urelu", DType::F32) => strided::relu::FLOAT,
                ("uround", DType::F32) => strided::round::FLOAT,
                ("utanh", DType::F32) => strided::tanh::FLOAT,

                ("ucos", DType::F16) => strided::cos::HALF,
                ("usin", DType::F16) => strided::sin::HALF,
                ("usqr", DType::F16) => strided::sqr::HALF,
                ("usqrt", DType::F16) => strided::sqrt::HALF,
                ("uneg", DType::F16) => strided::neg::HALF,
                ("uexp", DType::F16) => strided::exp::HALF,
                ("ulog", DType::F16) => strided::log::HALF,
                ("ugelu", DType::F16) => strided::gelu::HALF,
                ("ugelu_erf", DType::F16) => strided::gelu_erf::HALF,
                ("uerf", DType::F16) => strided::erf::HALF,
                ("usilu", DType::F16) => strided::silu::HALF,
                ("uabs", DType::F16) => strided::abs::HALF,
                ("uceil", DType::F16) => strided::ceil::HALF,
                ("ufloor", DType::F16) => strided::floor::HALF,
                ("urelu", DType::F16) => strided::relu::HALF,
                ("uround", DType::F16) => strided::round::HALF,
                ("utanh", DType::F16) => strided::tanh::HALF,

                ("ucos", DType::BF16) => strided::cos::BFLOAT,
                ("usin", DType::BF16) => strided::sin::BFLOAT,
                ("usqr", DType::BF16) => strided::sqr::BFLOAT,
                ("usqrt", DType::BF16) => strided::sqrt::BFLOAT,
                ("uneg", DType::BF16) => strided::neg::BFLOAT,
                ("uexp", DType::BF16) => strided::exp::BFLOAT,
                ("ulog", DType::BF16) => strided::log::BFLOAT,
                ("ugelu", DType::BF16) => strided::gelu::BFLOAT,
                ("ugelu_erf", DType::BF16) => strided::gelu_erf::BFLOAT,
                ("uerf", DType::BF16) => strided::erf::BFLOAT,
                ("usilu", DType::BF16) => strided::silu::BFLOAT,
                ("uabs", DType::BF16) => strided::abs::BFLOAT,
                ("uceil", DType::BF16) => strided::ceil::BFLOAT,
                ("ufloor", DType::BF16) => strided::floor::BFLOAT,
                ("urelu", DType::BF16) => strided::relu::BFLOAT,
                ("uround", DType::BF16) => strided::round::BFLOAT,
                ("utanh", DType::BF16) => strided::tanh::BFLOAT,

                (name, dtype) => {
                    crate::bail!("Metal strided unary {name} {dtype:?} not implemented")
                }
            };
            let dst = BufferOffset::zero_offset(&buffer);
            candle_metal_kernels::call_unary_strided(
                &device.device,
                &encoder,
                &device.kernels,
                kernel_name,
                layout.dims(),
                src,
                layout.stride(),
                dst,
            )
            .map_err(MetalError::from)?;
        }

        Ok(Self::new(buffer, device.clone(), el_count, dtype))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.binary(B::KERNEL, rhs, lhs_l, rhs_l)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device.clone();
        let shape = t_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dtype = t.dtype;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(el, dtype)
            .with_label("where")
            .build()?;
        let encoder = self.device.command_encoder()?;
        if t.dtype() != f.dtype() {
            crate::bail!(
                "Invalid where: different dtypes for values {:?} != {:?}",
                t.dtype(),
                f.dtype()
            );
        }
        let name = match (self.dtype, t.dtype()) {
            (DType::U8, DType::F32) => "where_u8_f32",
            (DType::U32, DType::F32) => "where_u32_f32",
            (DType::U8, DType::BF16) => "where_u8_bf16",
            (DType::U8, DType::F16) => "where_u8_f16",
            (DType::U8, DType::I64) => "where_u8_i64",
            (DType::U8, DType::U32) => "where_u8_u32",
            (DType::U8, DType::U8) => "where_u8_u8",
            (left, right) => crate::bail!("Metal where_cond {left:?} {right:?} not implemented"),
        };
        let src = buffer_o(&self.buffer, layout, self.dtype);
        let t = buffer_o(&t.buffer, t_l, t.dtype);
        let f = buffer_o(&f.buffer, f_l, f.dtype);
        candle_metal_kernels::call_where_cond(
            &device.device,
            &encoder,
            &device.kernels,
            name,
            dtype.size_in_bytes(),
            dims,
            src,
            layout.stride(),
            layout.is_contiguous(),
            t,
            t_l.stride(),
            t_l.is_contiguous(),
            f,
            f_l.stride(),
            f_l.is_contiguous(),
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device, el, dtype))
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let shape = layout.shape();
        let dims = shape.dims();
        let strides = layout.stride();

        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let k_size = params.k_size;
        let l_out = (dims[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
        let dst_el = dims[0] * l_out * dims[1] * k_size;
        let dst = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("conv1d_im2col")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let name = conv_kernel_name(ConvKernel::IM2COL1D, self.dtype, "conv1d")?;
        let src = buffer_o(&self.buffer, layout, self.dtype);
        candle_metal_kernels::call_im2col1d_strided(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            layout.shape().dims(),
            strides,
            (k_size, stride, padding, dilation),
            src,
            &dst,
        )
        .map_err(MetalError::from)?;
        drop(encoder);
        let col = Self {
            buffer: dst,
            device,
            count: dst_el,
            dtype: self.dtype,
        };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        // `copy_strided_src` below writes every element of the destination, so
        // allocating it zeroed adds a blit fill whose result is immediately
        // discarded. This is per convolution, and models that run convolution
        // in most of their layers pay it repeatedly.
        let el_count = res_l.shape().elem_count();
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(el_count, res.dtype())
            .with_label("conv1d_out")
            .build()?;
        let mut res_t = Self::new(buffer, self.device().clone(), el_count, res.dtype());
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv1d_depthwise(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        // Only the contiguous-kernel case is fused; anything else falls back.
        if !kernel_l.is_contiguous() {
            return Err(crate::Error::UnsupportedDTypeForOp(
                self.dtype,
                "conv1d_depthwise (non-contiguous kernel)",
            ));
        }

        // Unlike the other conv entry points this returns `UnsupportedDTypeForOp`
        // rather than bailing: `conv1d` inspects the error to decide whether to
        // fall back to the generic im2col path, so the dtype refusal has to stay
        // a typed error.
        let Some(name) = ConvKernel::CONV1D_DEPTHWISE.name(self.dtype.as_str()) else {
            return Err(crate::Error::UnsupportedDTypeForOp(
                self.dtype,
                "conv1d_depthwise",
            ));
        };

        let dims = layout.shape().dims();
        let (b, c, _l_in) = (dims[0], dims[1], dims[2]);
        let l_out = params.l_out();
        let dst_el = b * c * l_out;

        // Written in destination order, so no transpose copy afterwards.
        let dst = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("conv1d_depthwise")
            .build()?;

        // Prefer the k_size-specialized variant when every precondition it was
        // compiled under holds. All three are required together: the kernel
        // hard-codes stride and dilation to 1 and indexes the source as a
        // contiguous (b, c, l_in), so a violation would read the wrong elements
        // rather than merely run slower.
        let specialized = if params.stride == 1 && params.dilation == 1 && layout.is_contiguous() {
            ConvKernel::conv1d_depthwise_k(params.k_size)
                .and_then(|family| family.name(self.dtype.as_str()))
        } else {
            None
        };

        let encoder = self.device.command_encoder()?;
        match specialized {
            Some(name) => candle_metal_kernels::call_conv1d_depthwise_k(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                name,
                dims,
                (params.k_size, params.padding),
                buffer_o(&self.buffer, layout, self.dtype),
                buffer_o(&kernel.buffer, kernel_l, kernel.dtype),
                &dst,
            )
            .map_err(MetalError::from)?,
            None => candle_metal_kernels::call_conv1d_depthwise(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                name,
                dims,
                layout.stride(),
                (
                    params.k_size,
                    params.stride,
                    params.padding,
                    params.dilation,
                ),
                buffer_o(&self.buffer, layout, self.dtype),
                buffer_o(&kernel.buffer, kernel_l, kernel.dtype),
                &dst,
            )
            .map_err(MetalError::from)?,
        }

        Ok(Self::new(dst, self.device.clone(), dst_el, self.dtype))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        k: &Self,
        k_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        const USE_COL2IM_CONV1D_TR: bool = true;

        let can_use_col2im = k_layout.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        let l_out = params.l_out();
        let dst_el = params.c_out * l_out * params.b_size;

        let buffer = if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = layout.shape().dims3()?;
            let (c_in2, c_out, k_size) = k_layout.shape().dims3()?;
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    layout.shape(),
                    k_layout.shape()
                )
            }
            let buffer = self
                .device
                .new_buffer_builder()
                .with_size_for(dst_el, self.dtype)
                .with_label("conv_transpose1d")
                .build()?;

            let name = conv_kernel_name(ConvKernel::COL2IM1D, self.dtype, "col2im1d")?;
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1],
                    k_layout.start_offset(),
                );
                self.matmul(
                    k,
                    (b_size, l_in, c_out * k_size, c_in),
                    &layout.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            // It is important for the command encoder to be obtained *after* the matmul
            // kernel has run, otherwise we might use a command-buffer that has been committed
            // already resulting in the following error.
            // _status < MTLCommandBufferStatusCommitted >
            // -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
            let encoder = self.device.command_encoder()?;
            candle_metal_kernels::call_col2im1d(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                name,
                &[b_size, l_in, c_out, k_size],
                params.k_size,
                params.stride,
                BufferOffset::zero_offset(&col.buffer),
                &buffer,
            )
            .map_err(MetalError::from)?;
            buffer
        } else {
            let buffer = self
                .device
                .new_buffer_builder()
                .with_size_for(dst_el, self.dtype)
                .with_label("conv_transpose1d")
                .build()?;

            let encoder = self.device.command_encoder()?;
            let name =
                conv_kernel_name(ConvKernel::CONV_TRANSPOSE1D, self.dtype, "conv_transpose1d")?;
            candle_metal_kernels::call_conv_transpose1d(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                name,
                params.dilation,
                params.stride,
                params.padding,
                params.output_padding,
                params.c_out,
                l_out,
                params.b_size,
                layout.dims(),
                layout.stride(),
                k_layout.dims(),
                k_layout.stride(),
                &self.buffer,
                layout.start_offset() * self.dtype.size_in_bytes(),
                &k.buffer,
                k_layout.start_offset() * k.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
            buffer
        };
        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let shape = layout.shape();
        let dims = shape.dims();

        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let h_k = params.k_h;
        let w_k = params.k_w;
        let h = dims[2];
        let w = dims[3];
        let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
        let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;
        let dst_el = dims[0] * h_out * w_out * dims[1] * h_k * w_k;

        let dst = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("conv2d_im2col")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let name = conv_kernel_name(ConvKernel::IM2COL, self.dtype, "conv2d")?;
        let src = buffer_o(&self.buffer, layout, self.dtype);
        candle_metal_kernels::call_im2col_strided(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            layout.shape().dims(),
            layout.stride(),
            (h_k, w_k, stride, padding, dilation),
            src,
            &dst,
        )
        .map_err(MetalError::from)?;
        drop(encoder);
        let col = Self {
            buffer: dst,
            device,
            count: dst_el,
            dtype: self.dtype,
        };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        // Kernel shape: (c_in_k, c_out, h_k, w_k)
        // Input shape: (b_size, c_in, h_in, w_in)
        let (out_w, out_h) = (params.out_w(), params.out_h());
        let dst_el = params.c_out * out_w * out_h * params.b_size;

        let dims = l.dims();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for conv_transpose2d {dims:?}, expected 4")
        }

        let k_dims = kernel_l.dims();
        if k_dims.len() != 4 {
            crate::bail!("unexpected kernel shape for conv_transpose2d {k_dims:?}, expected 4")
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("conv_transpose2d")
            .build()?;

        let encoder = self.device.command_encoder()?;

        let name = conv_kernel_name(ConvKernel::CONV_TRANSPOSE2D, self.dtype, "conv_transpose2d")?;

        candle_metal_kernels::call_conv_transpose2d(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            CallConvTranspose2dCfg {
                dilation: params.dilation,
                stride: params.stride,
                padding: params.padding,
                output_padding: params.output_padding,
                c_out: params.c_out,
                out_h,
                out_w,
                b_size: params.b_size,
                input_dims: l.dims(),
                input_stride: l.stride(),
                kernel_dims: kernel_l.dims(),
                kernel_stride: kernel_l.stride(),
                input_offset: l.start_offset() * self.dtype.size_in_bytes(),
                kernel_offset: kernel_l.start_offset() * kernel.dtype.size_in_bytes(),
            },
            &self.buffer,
            &kernel.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn avg_pool2d(
        &self,
        inp_l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        let shape = inp_l.shape();
        let (b_size, channels, width, height) = shape.dims4()?;
        let strides = inp_l.stride();
        let name = conv_kernel_name(ConvKernel::AVG_POOL2D, self.dtype, "avg_pool2d")?;
        let out_w = (width - w_k) / w_stride + 1;
        let out_h = (height - h_k) / h_stride + 1;
        let dst_el = out_w * out_h * b_size * channels;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("avg_pool2d")
            .build()?;
        let encoder = self.device.command_encoder()?;
        candle_metal_kernels::call_pool2d(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            inp_l.dims(),
            strides,
            out_w,
            out_h,
            w_k,
            h_k,
            w_stride,
            h_stride,
            &self.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn max_pool2d(
        &self,
        inp_l: &Layout,
        (w_k, h_k): (usize, usize),
        (w_stride, h_stride): (usize, usize),
    ) -> Result<Self> {
        let shape = inp_l.shape();
        let (b_size, channels, width, height) = shape.dims4()?;
        let strides = inp_l.stride();
        let name = conv_kernel_name(ConvKernel::MAX_POOL2D, self.dtype, "max_pool2d")?;
        let out_w = (width - w_k) / w_stride + 1;
        let out_h = (height - h_k) / h_stride + 1;
        let dst_el = out_w * out_h * b_size * channels;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("max_pool2d")
            .build()?;
        let encoder = self.device.command_encoder()?;
        candle_metal_kernels::call_pool2d(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            inp_l.dims(),
            strides,
            out_w,
            out_h,
            w_k,
            h_k,
            w_stride,
            h_stride,
            &self.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("Metal upsample_nearest1d not implemented")
    }

    fn upsample_nearest2d(&self, inp_l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        // let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let strides = inp_l.stride();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        }
        let name = conv_kernel_name(
            ConvKernel::UPSAMPLE_NEAREST2D,
            self.dtype,
            "upsample_nearest2d",
        )?;

        let dst_el = out_w * out_h * dims[0] * dims[1];
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("upsample_nearest2d")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, inp_l, self.dtype);
        candle_metal_kernels::call_upsample_nearest_2d(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            dims,
            strides,
            out_w,
            out_h,
            src,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn upsample_bilinear2d(
        &self,
        inp_l: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let shape = inp_l.shape();
        let dims = shape.dims();
        let strides = inp_l.stride();

        if dims.len() != 4 {
            crate::bail!("unexpected input shape for upsample_bilinear2d {dims:?}")
        }

        let name = conv_kernel_name(
            ConvKernel::UPSAMPLE_BILINEAR2D,
            self.dtype,
            "upsample_bilinear2d",
        )?;

        let dst_el = out_w * out_h * dims[0] * dims[1];
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(dst_el, self.dtype)
            .with_label("upsample_bilinear2d")
            .build()?;

        let encoder = self.device.command_encoder()?;

        let src = buffer_o(&self.buffer, inp_l, self.dtype);
        candle_metal_kernels::call_upsample_bilinear_2d(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            dims,
            strides,
            out_w,
            out_h,
            align_corners,
            scale_h,
            scale_w,
            src,
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(buffer, self.device.clone(), dst_el, self.dtype))
    }

    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        // `src_l` is checked as well as `ids_l`, and that check is the fix for
        // #87. `gather` was the only op in this family guarding one operand:
        // `scatter_set`, `scatter_add_set` and `index_add` all check every
        // operand they take.
        //
        // The kernel it dispatches computes a flat contiguous offset --
        // `(left_rank_i * src_dim_size + input_i) * right_size + right_rank_i`
        // -- and receives neither `src_dims` nor `src_strides`, so it has no way
        // to address a strided source. Guarding `ids_l` alone therefore accepted
        // a non-contiguous source and read it as though it were contiguous,
        // returning a plausible wrong tensor where an error belongs.
        //
        // **Both reference backends decline this**, which is what makes the
        // guard the fix rather than a strided arm. The CPU bails in
        // `cpu_backend/mod.rs`'s `Gather::f`, and so does CUDA:
        // `cuda_backend/mod.rs`'s `Gather::f` has a second `RequiresContiguous`
        // on the source, distinct from the one on the ids, and `indexing.cu`'s
        // `gather` takes no `num_dims`/`info` and computes the same flat offset
        // this kernel does. Accepting a strided source would make Metal the only
        // backend that does.
        //
        // Note the contrast with `index_select` below, which *does* carry a
        // strided arm. CUDA's `index_select` kernel takes `num_dims` and
        // branches on `is_contiguous`, so there the reference computes and
        // matching it was correct (#82). `gather` is the opposite case, and the
        // two fixes point in opposite directions for that reason.
        if !ids_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "gather" }.bt());
        };
        let ids_el = ids_l.dims()[dim];
        let dst_el = ids_l.shape().elem_count();
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(dst_el, dtype)
            .with_label("gather")
            .build()?;
        let name = indexing_kernel_name(IndexingKernel::GATHER, ids.dtype, self.dtype, "gather")?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, src_l, dtype);
        let ids = buffer_o(&ids.buffer, ids_l, ids.dtype);
        candle_metal_kernels::call_gather(
            &device.device,
            &encoder,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            src,
            ids,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dst_el, dtype))
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if !l.is_contiguous() || !ids_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "scatter" }.bt());
        };
        // `UnexpectedDType` rather than `indexing_kernel_name`'s `bail!`: this
        // site reports against the *ids* dtype specifically, which is a better
        // message than a generic pair, and changing it would be a behaviour
        // change beside a name-table change.
        let Some(name) = IndexingKernel::SCATTER.name(ids.dtype.as_str(), self.dtype.as_str())
        else {
            Err(MetalError::UnexpectedDType {
                msg: "scatter ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?
        };
        let encoder = self.device.command_encoder()?;
        let dst = buffer_o(&self.buffer, l, self.dtype);
        let src = buffer_o(&src.buffer, src_l, src.dtype);
        let ids = buffer_o(&ids.buffer, ids_l, ids.dtype);
        candle_metal_kernels::call_scatter(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            src_l.dims(),
            l.dims(),
            dim,
            src,
            ids,
            dst,
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if !l.is_contiguous() || !ids_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt());
        };
        let Some(name) = IndexingKernel::SCATTER_ADD.name(ids.dtype.as_str(), self.dtype.as_str())
        else {
            Err(MetalError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?
        };
        let encoder = self.device.command_encoder()?;
        let dst = buffer_o(&self.buffer, l, self.dtype);
        let src = buffer_o(&src.buffer, src_l, src.dtype);
        let ids = buffer_o(&ids.buffer, ids_l, ids.dtype);
        candle_metal_kernels::call_scatter(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            src_l.dims(),
            l.dims(),
            dim,
            src,
            ids,
            dst,
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if !ids_l.is_contiguous() {
            crate::bail!("Metal index_select requires contiguous ids")
        }
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(dst_el, dtype)
            .with_label("index_select")
            .build()?;
        // The site that had the defect: `is_i64_u8` and `is_i64_u32` were named
        // in the table this replaces and declared nowhere in `indexing.metal`,
        // so these two pairs reached `load_pipeline` as names that could not
        // resolve. Both are declared now, and `indexing_names_resolve` is what
        // keeps them so.
        let name = indexing_kernel_name(
            IndexingKernel::INDEX_SELECT,
            ids.dtype,
            self.dtype,
            "contiguous index_select",
        )?;
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&self.buffer, src_l, dtype);
        let ids = buffer_o(&ids.buffer, ids_l, ids.dtype);
        candle_metal_kernels::call_index_select(
            &device.device,
            &encoder,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            src_l.is_contiguous(),
            src_l.dims(),
            src_l.stride(),
            src,
            ids,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dst_el, dtype))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let mut acc = self.device.zeros_impl(l.shape(), self.dtype())?;
        self.copy_strided_src(&mut acc, 0, l)?;
        if !ids_l.is_contiguous() || !src_l.is_contiguous() {
            return Err(crate::Error::RequiresContiguous { op: "index-add" }.bt());
        };
        let Some(name) = IndexingKernel::INDEX_ADD.name(ids.dtype.as_str(), self.dtype.as_str())
        else {
            Err(MetalError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?
        };
        let encoder = self.device.command_encoder()?;
        let src = buffer_o(&src.buffer, src_l, src.dtype);
        let ids = buffer_o(&ids.buffer, ids_l, ids.dtype);
        candle_metal_kernels::call_index_add(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            name,
            src_l.dims(),
            l.dims(),
            ids_l.dims(),
            dim,
            src,
            ids,
            &acc.buffer,
        )
        .map_err(MetalError::from)?;
        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size_for(b * m * n, self.dtype)
            .with_label("matmul")
            .build()?;
        let encoder = self.device.command_encoder()?;
        let dtype = match self.dtype {
            DType::F32 => candle_metal_kernels::GemmDType::F32,
            DType::F16 => candle_metal_kernels::GemmDType::F16,
            DType::BF16 => candle_metal_kernels::GemmDType::BF16,
            dtype => {
                return Err(
                    MetalError::Message(format!("mlx matmul doesn't support {dtype:?}")).into(),
                )
            }
        };
        candle_metal_kernels::call_mlx_gemm(
            &self.device.device,
            &encoder,
            &self.device.kernels,
            dtype,
            (b, m, n, k),
            lhs_l.stride(),
            lhs_l.start_offset() * self.dtype.size_in_bytes(),
            &self.buffer,
            rhs_l.stride(),
            rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
            &rhs.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(
            buffer,
            self.device.clone(),
            b * m * n,
            self.dtype(),
        ))
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if self.dtype() != dst.dtype() {
            crate::bail!(
                "copy2d with inconsistent dtypes {:?} {:?}",
                self.dtype(),
                dst.dtype()
            )
        }
        if src_s == d2 && dst_s == d2 {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("copy2d_contiguous");
            let src_offset = src_o * self.dtype.size_in_bytes();
            let length = d1 * d2 * self.dtype.size_in_bytes();
            let dst_offset = dst_o * dst.dtype().size_in_bytes();
            blit.copy_from_buffer(&self.buffer, src_offset, dst.buffer(), dst_offset, length);
        } else {
            let el_count = d1 * d2;
            if el_count == 0 {
                return Ok(());
            }
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::copy2d::FLOAT,
                DType::F16 => candle_metal_kernels::copy2d::HALF,
                DType::BF16 => candle_metal_kernels::copy2d::BFLOAT,
                DType::I64 => candle_metal_kernels::copy2d::I64,
                DType::I32 => candle_metal_kernels::copy2d::I32,
                DType::I16 => candle_metal_kernels::copy2d::I16,
                DType::U32 => candle_metal_kernels::copy2d::U32,
                DType::U8 => candle_metal_kernels::copy2d::U8,
                dtype => crate::bail!("Metal copy2d {dtype:?} not implemented"),
            };
            let encoder = self.device.command_encoder()?;
            candle_metal_kernels::call_copy2d(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                kernel_name,
                &self.buffer,
                &dst.buffer,
                d1,
                d2,
                src_s,
                dst_s,
                src_o * self.dtype.size_in_bytes(),
                dst_o * self.dtype.size_in_bytes(),
            )
            .map_err(MetalError::from)?;
        }
        Ok(())
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if src_l.is_contiguous() && self.dtype == dst.dtype() {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("copy_contiguous");
            let src_offset = src_l.start_offset() * self.dtype.size_in_bytes();
            let length = src_l.shape().elem_count() * self.dtype.size_in_bytes();
            let dst_offset = dst_offset * dst.dtype().size_in_bytes();
            blit.copy_from_buffer(&self.buffer, src_offset, dst.buffer(), dst_offset, length);
        } else {
            let src_shape = src_l.shape();
            let el_count = src_shape.elem_count();
            if el_count == 0 {
                return Ok(());
            }
            if self.dtype == dst.dtype() {
                let strides = src_l.stride();
                let ndim = strides.len();
                // Inner blocks are contiguous - eligible for copy2d
                if ndim == 1 || strides[ndim - 1] == 1 {
                    if let crate::StridedBlocks::UniformBlocks {
                        start_offset,
                        block_len,
                        count,
                        src_stride,
                    } = src_l.strided_blocks()
                    {
                        return self.copy2d(
                            dst,
                            count,
                            block_len,
                            src_stride,
                            block_len,
                            start_offset,
                            dst_offset,
                        );
                    }
                }
            }
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
                DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
                DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
                DType::I64 => candle_metal_kernels::unary::strided::copy::I64,
                DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
                DType::U8 => candle_metal_kernels::unary::strided::copy::U8,
                dtype => crate::bail!("Metal copy_strided {dtype:?} not implemented"),
            };
            let src = buffer_o(&self.buffer, src_l, self.dtype);
            let dst = BufferOffset {
                buffer: &dst.buffer,
                offset_in_bytes: dst_offset * dst.dtype.size_in_bytes(),
            };
            let encoder = self.device.command_encoder()?;
            candle_metal_kernels::call_unary_strided(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                kernel_name,
                src_l.dims(),
                src,
                src_l.stride(),
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: Arc<PooledBuffer>, device: MetalDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn binary(
        &self,
        op: &'static str,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        fn kernel_name(op: &'static str, dtype: &DType, suffix: &str) -> String {
            format!("{op}_{}{}", dtype.as_str(), suffix)
        }
        let device = self.device();
        let shape = lhs_l.shape();
        let el_count = shape.elem_count();
        let encoder = device.command_encoder()?;
        let lhs = buffer_o(&self.buffer, lhs_l, self.dtype);
        let rhs = buffer_o(&rhs.buffer, rhs_l, rhs.dtype);

        let dtype = match op {
            "eq" | "ne" | "le" | "lt" | "ge" | "gt" => DType::U8,
            _ => self.dtype,
        };
        let lhs_is_scalar = lhs_l.is_scalar_like();
        let rhs_is_scalar = rhs_l.is_scalar_like();
        let lhs_contiguous = lhs_l.is_contiguous();
        let rhs_contiguous = rhs_l.is_contiguous();

        let contiguous_kernel = kernel_name(op, &self.dtype, "");
        let kernel = match (lhs_is_scalar, rhs_is_scalar, lhs_contiguous, rhs_contiguous) {
            (true, true, _, _) => kernel_name(op, &self.dtype, "_scalar"),
            (true, false, _, true) => kernel_name(op, &self.dtype, "_sc"),
            (true, false, _, false) => kernel_name(op, &self.dtype, "_rss"),
            (false, true, true, _) => kernel_name(op, &self.dtype, "_cs"),
            (false, true, false, _) => kernel_name(op, &self.dtype, "_lss"),
            (false, false, true, true) => contiguous_kernel.clone(),
            (false, false, true, false) => kernel_name(op, &self.dtype, "_rstrided"),
            (false, false, false, true) => kernel_name(op, &self.dtype, "_lstrided"),
            (false, false, false, false) => kernel_name(op, &self.dtype, "_strided"),
        };

        let buffer = if kernel == contiguous_kernel {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(el_count, dtype)
                .with_label(op)
                .build()?;
            candle_metal_kernels::call_binary_contiguous(
                &device.device,
                &encoder,
                &device.kernels,
                kernel,
                self.dtype.size_in_bytes(),
                el_count,
                lhs,
                rhs,
                &buffer,
            )
            .map_err(MetalError::from)?;
            buffer
        } else {
            let buffer = device
                .new_buffer_builder()
                .with_size_for(el_count, dtype)
                .with_label(op)
                .build()?;
            candle_metal_kernels::call_binary_strided(
                &device.device,
                &encoder,
                &device.kernels,
                kernel,
                self.dtype.size_in_bytes(),
                lhs_l.dims(),
                lhs,
                lhs_l.stride(),
                rhs,
                rhs_l.stride(),
                &buffer,
            )
            .map_err(MetalError::from)?;
            buffer
        };
        Ok(Self::new(buffer, device.clone(), el_count, dtype))
    }

    pub(crate) fn to_cpu<T: Clone>(&self) -> Result<Vec<T>> {
        let size = self.count * self.dtype.size_in_bytes();

        // Fast path: read the source directly, no destination and no blit.
        //
        // `DESIGN.md` §15.2 #12 forbids blits on the decode path, and this
        // function has violated it once per token for as long as it has existed
        // (§11.5b finding 1) -- `to_scalar` on the sampled argmax reaches here
        // for **4 bytes**. The blit exists to land the bytes somewhere the CPU
        // can read; when the source is already CPU-readable there is nothing
        // for it to do.
        //
        // **The test is `contents()`, and what it decides is not the storage
        // mode -- it is whether Metal happened to MAP this allocation.**
        // Measured on M1 Max / macOS 26.6.2 (`measurements/issue-336-raw/`):
        // a `StorageModePrivate` buffer is CPU-mapped below **16384 bytes** and
        // never at or above it, exactly. So `buffer.rs`'s "null for a
        // Private-storage allocation" is wrong for small allocations and right
        // for large ones, and neither a mode test nor a size test is the honest
        // predicate: the pool rounds, so the same 4-byte `to_scalar` lands in a
        // 12288 B buffer on one arm and a 731136 B one on another, and only the
        // first is readable.
        //
        // Asking the pointer asks the only question whose answer is
        // load-bearing, and it fails **closed**: an unmapped buffer takes the
        // blit, which is the pre-existing path. That is why this is safe to
        // ship even though which arm a given readback takes is a property of
        // the allocator rather than of this function.
        //
        // What is NOT skipped is the wait. The GPU still has to finish writing
        // the value before the CPU may read it, so `flush_and_wait_current`
        // stays on both paths; §11.2a measures that half at 0.241-0.265 ms
        // against the blit's own share, and only the blit is removed here.
        // Skipping the wait would be a read-before-write race, which §3.5 makes
        // silent corruption rather than an error.
        //
        // # The pointer is read ONCE, and after the wait (#366)
        //
        // The first version of this path read `contents()` twice — once to test
        // for null and once inside `read_to_vec` — with `flush_and_wait_current`
        // between them. **The value tested and the value used were different
        // reads of an ObjC property**, and the second is what `memmove` gets.
        // Measured, 25 of 25 crash reports read `memmove(dst, src=0x0, n=4)`
        // with `esr` a *read* fault at `far=0x0`, on a stack of
        // `sample_argmax -> to_scalar -> to_cpu_storage`.
        //
        // **And `read_to_vec`'s `assert!(!ptr.is_null())` does not survive
        // optimization here**: the assert's panic string is absent from the
        // release binary, and the emitted code goes `Buffer::contents` →
        // `cbz x23` (the *length* test) → `memcpy` with no null test on the
        // pointer at all. So the second read had no check of its own and the
        // first read's check did not cover it.
        //
        // Reading once removes the window by construction rather than narrowing
        // it: there is one value, it is tested, and it is the one used. The
        // ordering is the fix — the wait must come first, because the pointer is
        // only meaningful once the GPU has finished — which is the same
        // §6.7 L4 shape as the pool's own two clocks, in a readback.
        if !crate::metal_backend::to_cpu_force_blit() {
            // Wait first: `contents()` is only worth asking after the GPU has
            // finished writing, and asking before means the answer can change
            // before it is used.
            self.device.flush_and_wait_current()?;
            // `black_box` is load-bearing under `lto = "thin"` and is not a
            // micro-optimisation guard — measured 2026-09-01 (#445).
            //
            // The doc comment on `read_to_vec` below says this branch is kept
            // because it is "a branch on a value the compiler cannot fold
            // away". That is true within one codegen unit and FALSE across a
            // thin-LTO boundary: with `[profile.release] lto = "thin"` the
            // optimiser inlines `Buffer::contents()`, concludes the returned
            // pointer is non-null, and deletes this test, the
            // `TO_CPU_NULL_AFTER_WAIT` counter and the blit fallback with it —
            // so a null `contents()` reaches `memmove` and segfaults, 8 of 8
            // decode runs. Structurally: the null-branch-only counter symbol is
            // present in a non-LTO build and ABSENT under LTO
            // (`measurements/issue-445-raw/lto_mechanism.sh`).
            //
            // `black_box` makes the pointer opaque, so the test survives
            // whatever the optimiser can prove about the ObjC getter. It costs
            // one unfoldable move on a path that has just waited for the GPU.
            let src = std::hint::black_box(self.buffer.contents());
            if !src.is_null() {
                TO_CPU_FAST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The same pointer that was tested, not a second read of it.
                // SAFETY: `src` is non-null and CPU-mapped, the GPU work that
                // writes it has completed above, and `self.buffer` keeps the
                // allocation alive for the length of this borrow.
                let slice = unsafe { std::slice::from_raw_parts(src as *const T, self.count) };
                return Ok(slice.to_vec());
            }
            // Mapped-then-null is the case #366 crashed on. It is reachable and
            // it is now a fallback rather than a fault: the blit below is the
            // pre-#336 path and computes the same bytes.
            TO_CPU_NULL_AFTER_WAIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        TO_CPU_BLIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(size)
            .with_label("blit_to_cpu_dst")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, size);
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec(&buffer, self.count))
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = Device::all().swap_remove(ordinal);
        let command_queue = device.new_command_queue().map_err(MetalError::from)?;
        #[cfg(feature = "metal-debug-labels")]
        command_queue.setLabel(Some(&NSString::from_str("candle-metal")));
        let kernels = Arc::new(Kernels::new());
        let residency_set = Arc::new(ResidencySet::new(&device));
        let seed_buf = device
            .new_buffer_with_data(
                [299792458u64].as_ptr() as *const c_void,
                std::mem::size_of::<u64>(),
                RESOURCE_OPTIONS,
            )
            .map_err(MetalError::from)?;
        #[cfg(feature = "metal-debug-labels")]
        seed_buf.set_label("rng_seed");
        residency_set.insert(&seed_buf);
        let seed = Arc::new(Mutex::new(seed_buf));
        let commands = Commands::new(command_queue, &residency_set).map_err(MetalError::from)?;

        // Both pools decide reuse against the GPU's progress through the command
        // queue, not against CPU refcounts. `Commands` owns that clock and is
        // the only thing that advances it.
        let buffers = BufferPool::with_clock(commands.clock());
        let private_buffers = BufferPool::with_clock(commands.clock());
        {
            // Subscribed here, before any work is encoded, so no command buffer
            // completes without a subscriber to hear it.
            let (shared, private) = (buffers.clone(), private_buffers.clone());
            commands.on_command_buffer_complete(move || {
                shared.drain_completed();
                private.drain_completed();
            });
        }

        {
            // Installed before any allocation, so no buffer is evicted without
            // the residency set hearing about it. The pool bounds its free list
            // by destroying buffers (§6.3b), which is the second place a
            // buffer's existence ends and the one nothing was announcing
            // (`DESIGN.md` §6.3c).
            let observer = Arc::new(ResidencyEvictionObserver::new(Arc::clone(&residency_set)));
            buffers.set_eviction_observer(observer.clone());
            private_buffers.set_eviction_observer(observer);
        }

        let commands = Arc::new(commands);
        // The guard holds its own `Arc` to the two things it needs, so it fires
        // when the last `MetalDevice` handle drops rather than when any clone
        // does -- every `Tensor` holds a clone, so the difference is a teardown
        // guard against a per-tensor GPU synchronize (`DESIGN.md` §6.3c).
        let teardown = Arc::new(DeviceTeardown::new(
            Arc::clone(&commands),
            Arc::clone(&residency_set),
        ));

        Ok(Self {
            teardown,
            id: DeviceId::new(),
            device,
            commands,
            buffers,
            private_buffers,
            kernels,
            seed,
            seed_value: Arc::new(RwLock::new(299792458)),
            residency_set,
            // No arena until one is installed: the classical path is the
            // default, and the cost of the seam is one `Option` test per
            // allocation (`DESIGN.md` §9.2a, §11.1).
            arena: Arc::new(RwLock::new(None)),
            arena_recorder: Arc::new(Mutex::new(None)),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let buffer = self
            .new_buffer_builder()
            .with_size_for(shape.elem_count(), dtype)
            .with_label("alloc-uninit")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let size = shape.elem_count() * dtype.size_in_bytes();
        let buffer = self
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("zeros")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let label = "storage_from_slice";
        let (count, buffer) = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::U32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::I16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::I32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::I64(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::BF16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::F16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::F32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::F64(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::F8E4M3(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F4(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(Error::UnsupportedDTypeForOp(T::DTYPE, "to_dtype").bt())
            }
        };
        Ok(Self::Storage::new(buffer?, self.clone(), count, T::DTYPE))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let label = "storage_from_cpu_storage";
        let (count, buffer) = match storage {
            CpuStorage::U8(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::U32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::I16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::I32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::I64(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::BF16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::F16(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::F32(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::F64(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::F8E4M3(storage) => (
                storage.len(),
                self.new_buffer_builder()
                    .with_data(storage)
                    .with_label(label)
                    .build(),
            ),
            CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F4(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(Error::UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
            }
        };
        Ok(Self::Storage::new(
            buffer?,
            self.clone(),
            count,
            storage.dtype(),
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_uniform_f32",
            DType::F16 => "rand_uniform_f16",
            DType::BF16 => "rand_uniform_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self
            .new_buffer_builder()
            .with_size_for(shape.elem_count(), dtype)
            .with_label("rand_uniform")
            .build()?;
        let encoder = self.command_encoder()?;
        candle_metal_kernels::call_random_uniform(
            &self.device,
            &encoder,
            &self.kernels,
            name,
            min as f32,
            max as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::Storage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_normal_f32",
            DType::F16 => "rand_normal_f16",
            DType::BF16 => "rand_normal_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self
            .new_buffer_builder()
            .with_size_for(shape.elem_count(), dtype)
            .with_label("rand_normal")
            .build()?;
        let encoder = self.command_encoder()?;
        candle_metal_kernels::call_random_normal(
            &self.device,
            &encoder,
            &self.kernels,
            name,
            mean as f32,
            stddev as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::Storage::new(
            buffer,
            self.clone(),
            shape.elem_count(),
            dtype,
        ))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed_value.write().unwrap() = seed;

        let seed_buffer = self.seed.try_lock().map_err(MetalError::from)?;
        let contents = seed_buffer.data();

        // **The two words are written in the order the KERNEL reads them, which
        // is not native `u64` order** (lloom #277).
        //
        // `random.metal`'s `atomic_load_seed` reconstructs the seed as
        // `(ulong)seed[0] << 32 | seed[1]` -- word 0 is the HIGH half. Copying a
        // native `u64` on a little-endian host puts the LOW half in word 0, so
        // the kernel read `swap32(seed)`, and for every seed below 2^32 that is
        // `seed << 32`.
        //
        // That is not merely a permuted seed, because of what happens next:
        // `HybridTaus::seed_per_thread` computes `uint4(ulong4(seeds) * ...)`,
        // and `uint4` **truncates to the low 32 bits**. A value of the form
        // `seed << 32` has 32 zero low bits, and any multiple of it still does
        // -- so component `.x`, the only component carrying the seed at all
        // (the other three are `tid`, `1`, `1`), was **identically zero for
        // every seed the API can express**.
        //
        // Measured before this fix, on `Tensor::rand(0,1,(4,))`: seeds `1`,
        // `999`, `12345` and `0xDEADBEEF` all produced
        // `[0.0030706443, 0.3755374, 0.676627, 0.69485384]` -- byte-identical to
        // the default -- while seeds `1<<32` and `3<<32` each moved it. The
        // probe was therefore capable of the other answer (§15.1 #1), which is
        // what makes the null a measurement rather than a blind instrument.
        // Artifact: `measurements/issue-277-raw/seed-probe.txt`.
        //
        // **Fixed on the host side rather than in the kernel, deliberately.**
        // Swapping `atomic_load_seed`'s words would work too and would change
        // what every existing seed means -- `random.metal` is shared with the
        // CPU/CUDA parity tests and with every other model that calls
        // `set_seed`. Writing the words the kernel's way is one file, no
        // `.metal` change, and leaves the kernel's own contract untouched.
        //
        // The default seed installed in `MetalDevice::new` is unaffected: it is
        // written by `new_buffer_with_data` and never passed through here, so a
        // run that does not call `set_seed` is byte-for-byte what it was.
        let hi = (seed >> 32) as u32;
        let lo = (seed & 0xFFFF_FFFF) as u32;
        unsafe {
            std::ptr::copy_nonoverlapping([hi, lo].as_ptr(), contents as *mut u32, 2);
        }
        seed_buffer.did_modify_range(NSRange::new(0, 8));

        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        self.wait_until_completed()
    }
}

/// Copies `n` elements out of a CPU-mapped buffer.
///
/// # The assertion here did not survive optimization (#366)
///
/// This function read `contents()` and `assert!`ed the result non-null. In the
/// release binary that panic **does not exist** — its message string is absent
/// and the emitted code runs `contents()` → a *length* test → `memcpy`, with no
/// test on the pointer. So a null buffer reached `memmove` as `src=0x0` and
/// segfaulted, 25 times out of 25 with an identical stack.
///
/// `panic!` inside an `if` is kept rather than `assert!` because it is a branch
/// on a value the compiler cannot fold away, and the caller above no longer
/// relies on it: the fast path tests its own pointer and takes the blit when it
/// is null. This is the second line of defence, not the first.
///
/// **"Cannot fold away" holds within a codegen unit and fails across a thin-LTO
/// boundary — measured 2026-09-01 (#445).** Under `lto = "thin"` the panic's
/// message string is absent from the binary here too, exactly as this comment
/// records for the pre-#366 `assert!`. The `black_box` below is what keeps the
/// test, and it is the same repair the fast path in `to_cpu` now carries.
fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = std::hint::black_box(buffer.contents()) as *const T;
    if ptr.is_null() {
        panic!(
            "read_to_vec: buffer has no CPU mapping (len={}, n={n})",
            buffer.length()
        );
    }
    // SAFETY: `ptr` is non-null and CPU-mapped, and the caller has waited for
    // the GPU work that writes it.
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

#[cfg(test)]
mod issue_336_split {
    //! Where does the readback's time go: the encoder break, or the wait?
    //!
    //! §11.2a measures the readback half at 0.227-0.256 ms and does not
    //! separate the two. This times the three components on the arm the issue
    //! specifies -- the queue ALREADY DRAINED, which is what
    //! `--sample-in-buffer` produces -- so `flush_and_wait_current` has nothing
    //! left to wait for and whatever remains is the blit's own cost.
    use crate::{DType, Device, Tensor, D};
    use std::time::Instant;

    #[test]
    fn where_the_readback_time_goes() {
        let Ok(dev) = Device::new_metal(0) else {
            return;
        };
        let Device::Metal(mdev) = &dev else {
            unreachable!()
        };
        let n = 256usize;

        let mut t_fast = Vec::new();
        let mut t_blit = Vec::new();
        let mut t_alloc = Vec::new();
        let mut t_encode = Vec::new();
        let mut t_wait = Vec::new();

        for i in 0..n {
            let logits = Tensor::rand(0f32, 1f32, (1, 128000), &dev).unwrap();
            let am = logits.argmax(D::Minus1).unwrap();
            let (storage, _) = am.storage_and_layout();
            let crate::Storage::Metal(ms) = &*storage else {
                unreachable!()
            };

            // The arm the issue specifies: queue already drained, as
            // `--sample-in-buffer` leaves it.
            mdev.wait_until_completed().unwrap();

            // FAST: direct read + a wait that has nothing to do.
            let s = Instant::now();
            let v: Vec<u32> = ms.to_cpu().unwrap();
            t_fast.push(s.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(v);

            // BLIT, decomposed.
            let size = ms.count * ms.dtype.size_in_bytes();
            let s = Instant::now();
            let dst = ms
                .device
                .new_buffer_builder()
                .with_size(size)
                .with_label("blit_to_cpu_dst")
                .build()
                .unwrap();
            let a = s.elapsed().as_secs_f64() * 1e3;

            let s2 = Instant::now();
            {
                let mut blit = ms.device.blit_command_encoder().unwrap();
                blit.set_label("blit_to_cpu");
                blit.copy_from_buffer(&ms.buffer, 0, &dst, 0, size);
            }
            let e = s2.elapsed().as_secs_f64() * 1e3;

            let s3 = Instant::now();
            ms.device.flush_and_wait_current().unwrap();
            let w = s3.elapsed().as_secs_f64() * 1e3;
            let v2: Vec<u32> = super::read_to_vec(&dst, ms.count);
            std::hint::black_box(v2);

            if i >= 32 {
                // warmup
                t_alloc.push(a);
                t_encode.push(e);
                t_wait.push(w);
                t_blit.push(a + e + w);
            }
            let _ = DType::U32;
        }

        let med = |v: &mut Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let (f, b) = (med(&mut t_fast.split_off(32)), med(&mut t_blit));
        let (al, en, wa) = (med(&mut t_alloc), med(&mut t_encode), med(&mut t_wait));
        println!("ISSUE336-SPLIT n={} (queue pre-drained)", t_wait.len());
        println!("ISSUE336-SPLIT fast_total_ms={f:.4}");
        println!("ISSUE336-SPLIT blit_total_ms={b:.4}  alloc={al:.4} encode={en:.4} wait={wa:.4}");
        println!("ISSUE336-SPLIT blit_minus_fast_ms={:.4}", b - f);
    }
}

#[cfg(test)]
mod issue_336_to_cpu {
    //! `to_cpu` without the blit (lloom #336, `DESIGN.md` §15.2 #12).
    //!
    //! The parity arms compare the two paths **against each other on the same
    //! tensor**, which is what makes them a comparison rather than a
    //! restatement: `CANDLE_METAL_TO_CPU_BLIT` is read once per process
    //! (§2.4/#69), so the arms are selected by calling the two code paths
    //! directly rather than by setting the variable per test.
    use crate::{DType, Device, Tensor, D};

    /// Read back through the blit, unconditionally -- the pre-#336 path, kept
    /// reachable so a test can compare against it in one process.
    fn via_blit<T: Clone>(ms: &super::MetalStorage) -> Vec<T> {
        let size = ms.count * ms.dtype.size_in_bytes();
        let buffer = ms
            .device
            .new_buffer_builder()
            .with_size(size)
            .with_label("blit_to_cpu_dst")
            .build()
            .unwrap();
        {
            let mut blit = ms.device.blit_command_encoder().unwrap();
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&ms.buffer, 0, &buffer, 0, size);
        }
        ms.device.flush_and_wait_current().unwrap();
        super::read_to_vec(&buffer, ms.count)
    }

    fn metal() -> Option<Device> {
        Device::new_metal(0).ok()
    }

    /// The decode case this issue is about: a 4-byte `to_scalar` on the argmax.
    #[test]
    fn argmax_readback_agrees_with_the_blit() {
        let Some(dev) = metal() else { return };
        let logits = Tensor::rand(0f32, 1f32, (1, 128000), &dev).unwrap();
        let am = logits.argmax(D::Minus1).unwrap();
        let (storage, _) = am.storage_and_layout();
        let crate::Storage::Metal(ms) = &*storage else {
            unreachable!()
        };

        let fast = am.to_vec1::<u32>().unwrap();
        let blit: Vec<u32> = via_blit(ms);
        assert_eq!(fast, blit, "fast path disagrees with the blit");

        // Non-vacuity (§15.1 #1): an argmax over a random 128000-wide row must
        // not be a constant the comparison would pass on trivially.
        assert_eq!(fast.len(), 1);
        assert!(fast[0] < 128000);
    }

    /// The generic contract: every dtype and a size past one element, since
    /// `to_cpu` is generic over both and only the 4-byte case motivated this.
    #[test]
    fn readback_agrees_with_the_blit_across_dtypes_and_sizes() {
        let Some(dev) = metal() else { return };
        let mut checked = 0usize;

        for n in [1usize, 3, 1024, 128000] {
            let base = Tensor::rand(0f32, 1f32, n, &dev).unwrap();

            for dtype in [DType::F32, DType::F16, DType::BF16, DType::U32, DType::U8] {
                let t = base.to_dtype(dtype).unwrap();
                let (storage, _) = t.storage_and_layout();
                let crate::Storage::Metal(ms) = &*storage else {
                    unreachable!()
                };

                // Compare the raw bytes, so the check is dtype-agnostic and
                // does not depend on a float comparison.
                let fast: Vec<u8> = ms.to_cpu().unwrap();
                let blit: Vec<u8> = via_blit(ms);
                assert_eq!(
                    fast.len() * std::mem::size_of::<u8>(),
                    blit.len(),
                    "length mismatch at {dtype:?} n={n}"
                );
                assert_eq!(fast, blit, "byte mismatch at {dtype:?} n={n}");

                // The comparison must be capable of failing: a buffer of one
                // repeated byte would agree under a broken read.
                if n > 1 && dtype == DType::F32 {
                    let distinct: std::collections::HashSet<_> = fast.iter().collect();
                    assert!(
                        distinct.len() > 1,
                        "fixture is uniform at {dtype:?} n={n}; the comparison proves nothing"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 20, "expected 4 sizes x 5 dtypes");
    }

    /// The fast path must read the value the GPU wrote, not a stale one.
    ///
    /// Without the wait this races; the assertion is that the *sequence*
    /// (dispatch, wait, read) lands the new value, run repeatedly so a
    /// read-before-write would show up as a disagreement rather than as luck.
    #[test]
    fn readback_sees_the_current_value_not_a_stale_one() {
        let Some(dev) = metal() else { return };
        let mut seen = std::collections::HashSet::new();
        for i in 0..32u32 {
            let t = (Tensor::ones(4, DType::U32, &dev).unwrap() * (i as f64)).unwrap();
            let got = t.to_vec1::<u32>().unwrap();
            assert_eq!(got, vec![i; 4], "stale or wrong readback at iteration {i}");
            seen.insert(got[0]);
        }
        // The loop must actually vary the value, or it proves nothing.
        assert_eq!(seen.len(), 32, "fixture did not vary");
    }

    /// **Which allocations are CPU-mapped, pinned as a fact about the machine.**
    ///
    /// This is what decides how often the fast path fires, and it is a property
    /// of *(device, OS)* rather than of candle (`CONTRIBUTING.md` §2.7). A
    /// Private allocation is mapped below 16 KiB and unmapped at or above it,
    /// so the same 4-byte `to_scalar` is readable or not depending on what size
    /// the pool rounded its buffer to.
    ///
    /// If this changes, the fast path's *hit rate* changes and nothing else --
    /// `to_cpu` still returns the right bytes either way, because the null test
    /// fails closed onto the blit.
    #[test]
    fn private_allocations_are_mapped_below_16k_and_not_above() {
        let Some(dev) = metal() else { return };
        let Device::Metal(m) = &dev else {
            unreachable!()
        };

        // Sized in u32 elements. The pool rounds up to `BUFFER_ALIGNMENT`
        // (128 B), so 16380 would become 16384 and land on the wrong side --
        // one element below the boundary is not one *allocation* below it.
        // 4064 u32 = 16256 B, which rounds to itself.
        let mapped = m.new_buffer(4064, DType::U32, "probe").unwrap(); // 16256 B
        let unmapped = m.new_buffer(4096, DType::U32, "probe").unwrap(); // 16384 B

        // `black_box` is load-bearing under `lto = "thin"` — measured
        // 2026-09-01 (#445, `DESIGN.md` §14.5c). Without it the second
        // assertion FAILS under LTO while the pointer it claims to test is
        // still `0x0`: the optimiser inlines `Buffer::contents()` across the
        // crate boundary, proves the result non-null, and folds `is_null()`
        // to `false`. Probed with the pointers printed, the two arms read
        // 16256 B -> mapped and 16384 B -> `0x0` IDENTICALLY, so the threshold
        // this test guards did NOT move — the test was being deleted, and
        // without the probe it would have been read as the behaviour change it
        // is written to detect. Same elision as the two production sites in
        // `to_cpu` and `read_to_vec`, landing on the check for it.
        let mapped_ptr = std::hint::black_box(mapped.contents());
        let unmapped_ptr = std::hint::black_box(unmapped.contents());

        assert!(
            !mapped_ptr.is_null(),
            "a 16256 B private allocation was expected to be CPU-mapped"
        );
        assert!(
            unmapped_ptr.is_null(),
            "a 16384 B private allocation was expected NOT to be CPU-mapped; \
             if this fires the threshold moved and the fast path's hit rate \
             changed -- see measurements/issue-336-raw/threshold.txt"
        );

        // Shared is mapped at both sizes, which is what makes the above a
        // statement about Private rather than about size alone.
        let big_shared = m.allocate_buffer(1 << 20).unwrap();
        assert!(
            !std::hint::black_box(big_shared.contents()).is_null(),
            "shared must be mapped"
        );
    }

    /// The env switch resolves, and it defaults to the fast path.
    #[test]
    fn force_blit_defaults_off() {
        // Only meaningful when the variable is unset, which is the shipping
        // configuration and how CI runs.
        if std::env::var("CANDLE_METAL_TO_CPU_BLIT").is_err() {
            assert!(
                !super::to_cpu_force_blit(),
                "the blit must be off by default"
            );
        }
    }
}

#[cfg(test)]
mod issue_336_storage_probe {
    //! Probe for lloom #336: is the argmax output's buffer CPU-readable?
    //!
    //! `to_cpu` blits for a 4-byte read (`DESIGN.md` §15.2 #12). Whether that
    //! blit is removable turns on one fact: does the buffer the argmax writes
    //! expose a CPU pointer that sees the GPU's write? This asks candle, through
    //! candle's own allocation paths, rather than a hand-rolled `MTLBuffer`.
    use crate::{DType, Device, Tensor, D};

    #[test]
    fn argmax_output_buffer_storage_and_readability() {
        let Ok(dev) = Device::new_metal(0) else {
            eprintln!("no metal device; skipping");
            return;
        };
        let Device::Metal(mdev) = &dev else {
            unreachable!()
        };

        // The real decode shape: argmax over a [1, 128000] f32 row, which is
        // what `sample_argmax` dispatches (§11.2a).
        let logits = Tensor::rand(0f32, 1f32, (1, 128000), &dev).unwrap();
        let am = logits.argmax(D::Minus1).unwrap();

        let (storage, _) = am.storage_and_layout();
        let crate::Storage::Metal(ms) = &*storage else {
            unreachable!()
        };

        let ptr = ms.buffer().contents();
        println!(
            "ISSUE336 argmax_out dtype={:?} count={}",
            am.dtype(),
            am.elem_count()
        );
        println!("ISSUE336 contents_ptr={ptr:?} null={}", ptr.is_null());

        // Drain so the GPU write has certainly landed, then read directly.
        mdev.wait_until_completed().unwrap();
        if !ptr.is_null() {
            let direct = unsafe { *(ptr as *const u32) };
            let viablit = am.to_vec1::<u32>().unwrap()[0];
            println!(
                "ISSUE336 direct={direct} via_blit={viablit} agree={}",
                direct == viablit
            );
        }
        let _ = DType::U32;
    }
}
