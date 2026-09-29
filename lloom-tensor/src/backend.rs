//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    fn powf(&self, _: &Layout, _: f64) -> Result<Self>;

    fn elu(&self, _: &Layout, _: f64) -> Result<Self>;

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self>;

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    /// Depthwise 1D convolution, writing directly in `(b, c, l_out)` layout.
    ///
    /// Optional. The default declines and callers fall back to the generic
    /// grouped path, so backends without a fused implementation are unaffected.
    fn conv1d_depthwise(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(crate::Error::UnsupportedDTypeForOp(
            self.dtype(),
            "conv1d_depthwise (no fused implementation for this backend)",
        ))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self>;
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self>;
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self>;

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self>;

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self>;
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self>;

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;

    /// Copy `src_l`'s elements of `self` into `dst_l`'s positions of `dst`,
    /// both layouts of one shape and either of them strided. The `Option` is
    /// the same shape as `conv1d_depthwise`'s decline: `Ok(false)` means the
    /// backend has no such copy and the caller materialises the source; a
    /// backend that returns `Ok(true)` has written every element.
    fn copy_strided_to(&self, _: &mut Self, _src_l: &Layout, _dst_l: &Layout) -> Result<bool> {
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    // Similar to cudaMemcpy2D, though values are in elements and not in bytes.
    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()>;

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()>;
}

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    // TODO: Make the usize generic and part of a generic DeviceLocation.
    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::DeviceLocation;

    fn same_device(&self, _: &Self) -> bool;

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage>;

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    fn set_seed(&self, _: u64) -> Result<()>;
    fn get_current_seed(&self) -> Result<u64>;

    /// Synchronize should block until all the operations on the device are completed.
    fn synchronize(&self) -> Result<()>;

    /// Open a forward pass of shape `key` on this device, so its activations
    /// can be served from a plan cached for that shape rather than allocated
    /// (`DESIGN.md` §9.2aa, "planned passes").
    ///
    /// # Why this is a defaulted no-op rather than a required method
    ///
    /// The plan is *base + offsets*: a backend implements it by handing out
    /// slices of one buffer it grew outside every pass. That shape is
    /// expressible on any allocator (CUDA: one `cudaMalloc` base and
    /// `DevicePtr` offsets), but nothing about the driver's correctness
    /// depends on a backend taking it up — a backend that declines allocates
    /// exactly as it did before, which is what CPU and CUDA do here.
    /// Defaulting rather than requiring is §3.6's *add a variant behind a
    /// seam, never rip out the block*: the pass brackets can be wired
    /// through every driver once, and each backend opts in when it has the
    /// mechanism, with no `#[cfg]` at the call sites.
    ///
    /// Returning `Ok(false)` means "not planned" and is not a failure. An
    /// `Err` is a real refusal — a base grow the device cannot admit
    /// (§9.5) — and stops the run before anything is allocated.
    fn begin_pass(&self, _key: crate::backend::PassShapeKey) -> Result<bool> {
        Ok(false)
    }

    /// Close the pass [`Self::begin_pass`] opened. A no-op on a backend that
    /// declined it.
    fn finish_pass(&self) -> Result<()> {
        Ok(())
    }

    /// Admit computing at `dtype` on this device: `Ok` for a dtype it
    /// executes natively, and for one it emulates in software once the
    /// operator has opted in (one warning, at the first admission); otherwise
    /// the refusal, naming the reason and the opt-in (`DESIGN.md` §15.2a).
    /// A model load asks before its first weight. CPU and CUDA run every
    /// dtype they accept natively, so they admit all.
    fn admit_compute_dtype(&self, _dtype: DType) -> Result<()> {
        Ok(())
    }

    /// Whether [`Self::admit_compute_dtype`] would admit `dtype`, asked
    /// without admitting it (no warning is printed).
    fn admits_compute_dtype(&self, _dtype: DType) -> bool {
        true
    }
}

/// The shape of a forward pass, backend-neutrally: what a plan is keyed by
/// (`DESIGN.md` §9.2aa). Mirrors the Metal backend's `PassKind` without
/// depending on it, so a driver names a pass shape once and every backend
/// reads the same key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassShapeKey {
    /// One decode step over `b` lockstep rows.
    Decode { b: u16 },
    /// One prefill forward of `w` tokens; `fresh` is the first pass on an
    /// empty cache, which allocates the session state every later pass finds
    /// in place.
    Prefill { w: u32, fresh: bool },
    /// One speculative verify pass of `k` positions.
    Verify { k: u16 },
    /// One batched step of `live` decode rows with `pieces` prompt pieces
    /// riding it, whose widths hash to `layout`.
    Ragged { live: u16, pieces: u16, layout: u64 },
}
