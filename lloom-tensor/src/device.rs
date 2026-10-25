use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};

/// [`Device::eos_gate_suspended`]'s guard.
#[must_use]
pub struct EosGateSuspended {
    #[cfg(feature = "metal")]
    _scope: Option<crate::kernels::kernel::UngatedScope>,
}

/// What a run of the EOS gate did (lloom #735 step 4; `Device::eos_gate_end`).
/// Defined here so a `DecodeStats` compiles off Metal too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EosGateReport {
    /// The sticky flag was up when the run ended: an EOS was seen, and every
    /// gated kernel of every forward encoded after it returned at once.
    pub fired: bool,
    /// Gate kernels dispatched (`step` calls).
    pub steps: u64,
    /// The cores arm was on: the quantized mat-vec cores were gated too.
    pub cores: bool,
}

/// A `DeviceLocation` represents a physical device whereas multiple `Device`
/// can live on the same location (typically for cuda devices).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
}

/// Cpu, Cuda, or Metal
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "metal",
    expect(
        clippy::large_enum_variant,
        reason = "Keep the public MetalDevice payload inline; boxing adds allocation and changes the variant API."
    )
)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
}

pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for Vec<&[S]> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let data = self.iter().copied().flatten().copied().collect::<Vec<_>>();
        S::to_cpu_storage_owned(data)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<S>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self.iter().map(|v| v.len()).sum();
        let mut dst = Vec::with_capacity(len);
        for v in self.iter() {
            dst.extend(v.iter().copied());
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<S>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        if self.is_empty() {
            return S::to_cpu_storage_owned(vec![]);
        }
        let len: usize = self
            .iter()
            .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                dst.extend(v2.iter().copied());
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<Vec<S>>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
                    .sum::<usize>()
            })
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                for v3 in v2.iter() {
                    dst.extend(v3.iter().copied());
                }
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
    }

    pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
        match self {
            Self::Cuda(d) => Ok(d),
            Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
            Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
        }
    }

    pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
        match self {
            Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
            Self::Cpu => crate::bail!("expected a metal device, got cpu"),
            Self::Metal(d) => Ok(d),
        }
    }

    pub fn new_cuda_with_stream(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(crate::CudaDevice::new_with_stream(ordinal)?))
    }

    pub fn new_metal(ordinal: usize) -> Result<Self> {
        Ok(Self::Metal(crate::MetalDevice::new(ordinal)?))
    }

    /// Run `f` with device specific context.
    ///
    /// On CPU this installs candle's private rayon thread pool for the
    /// duration of `f`, keeping worker threads warm across the many short
    /// parallel sections in a model forward pass. Currently noop for other backends.
    pub fn with_context<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self {
            Self::Cpu => crate::utils::with_threadpool(f),
            _ => f(),
        }
    }

    pub fn set_seed(&self, seed: u64) -> Result<()> {
        match self {
            Self::Cpu => CpuDevice.set_seed(seed),
            Self::Cuda(c) => c.set_seed(seed),
            Self::Metal(m) => m.set_seed(seed),
        }
    }

    pub fn get_current_seed(&self) -> Result<u64> {
        match self {
            Self::Cpu => CpuDevice.get_current_seed(),
            Self::Cuda(c) => c.get_current_seed(),
            Self::Metal(m) => m.get_current_seed(),
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
            (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
            _ => false,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Cuda(device) => device.location(),
            Device::Metal(device) => device.location(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal(_))
    }

    /// Whether this device computes at bf16: on Metal, natively or by the
    /// operator's opt-in to the emulated arm (§15.2a), so a caller choosing
    /// its dtype here is never handed one the device then refuses.
    pub fn supports_bf16(&self) -> bool {
        match self {
            Self::Cuda(d) => d.admits_compute_dtype(DType::BF16),
            Self::Metal(d) => d.admits_compute_dtype(DType::BF16),
            Self::Cpu => false,
        }
    }

    /// Return `BF16` for devices that support it, otherwise default to `F32`.
    pub fn bf16_default_to_f32(&self) -> DType {
        if self.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::cuda_is_available() {
            Self::new_cuda(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub fn metal_if_available(ordinal: usize) -> Result<Self> {
        if crate::utils::metal_is_available() {
            Self::new_metal(ordinal)
        } else {
            Ok(Self::Cpu)
        }
    }

    pub(crate) fn rand_uniform_f64(
        &self,
        lo: f64,
        up: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_uniform(shape, DType::F32, lo, up)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_uniform(shape, dtype, lo, up)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_uniform(shape, dtype, lo, up)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) fn rand_uniform<T: crate::FloatDType>(
        &self,
        lo: T,
        up: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_uniform_f64(lo.to_f64(), up.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn rand_normal_f64(
        &self,
        mean: f64,
        std: f64,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                // TODO: Remove the special case if we start supporting generating f16/bf16 directly.
                if dtype == DType::F16 || dtype == DType::BF16 {
                    let storage = device.rand_normal(shape, DType::F32, mean, std)?;
                    Storage::Cuda(storage).to_dtype(&crate::Layout::contiguous(shape), dtype)
                } else {
                    let storage = device.rand_normal(shape, dtype, mean, std)?;
                    Ok(Storage::Cuda(storage))
                }
            }
            Device::Metal(device) => {
                let storage = device.rand_normal(shape, dtype, mean, std)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) fn rand_normal<T: crate::FloatDType>(
        &self,
        mean: T,
        std: T,
        shape: &Shape,
    ) -> Result<Storage> {
        self.rand_normal_f64(mean.to_f64(), std.to_f64(), shape, T::DTYPE)
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.zeros_impl(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.zeros_impl(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Device::Cpu => {
                let storage = CpuDevice.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            Device::Cuda(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.alloc_uninit(shape, dtype)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(data.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = device.storage_from_slice(data)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(array.to_cpu_storage())),
            Device::Cuda(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = array.to_cpu_storage();
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        match self {
            Device::Cpu => Ok(Storage::Cpu(S::to_cpu_storage_owned(data))),
            Device::Cuda(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Cuda(storage))
            }
            Device::Metal(device) => {
                let storage = S::to_cpu_storage_owned(data);
                let storage = device.storage_from_cpu_storage_owned(storage)?;
                Ok(Storage::Metal(storage))
            }
        }
    }

    /// Asks the OS to keep the device's memory resident: Metal's residency
    /// set, which is unwired after a few idle seconds otherwise. Nothing to
    /// do on the other backends.
    pub fn request_residency(&self) {
        #[cfg(feature = "metal")]
        if let Self::Metal(d) = self {
            d.request_residency();
        }
    }

    /// Tells the OS the device's memory need not stay resident (Metal's
    /// residency set): under memory pressure. Nothing to do on the other
    /// backends.
    pub fn end_residency(&self) {
        #[cfg(feature = "metal")]
        if let Self::Metal(d) = self {
            d.end_residency();
        }
    }

    /// The Metal 4 queue's per-slot residency counts, as `(adds, commits,
    /// last_bytes, max_bytes)` (`kernels.residency.slot_trace`; `None` off
    /// Metal 4 or with the instrument off).
    pub fn metal4_slot_residency(&self) -> Option<(u64, u64, u64, u64)> {
        #[cfg(feature = "metal")]
        if let Self::Metal(d) = self {
            return d
                .metal4_slot_residency()
                .map(|s| (s.adds, s.commits, s.last_bytes, s.max_bytes));
        }
        None
    }

    /// The residency set's attachment and size, for a loaded line (`None`
    /// off Metal).
    pub fn residency_line(&self) -> Option<String> {
        #[cfg(feature = "metal")]
        if let Self::Metal(d) = self {
            return Some(d.residency_line());
        }
        None
    }

    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.synchronize(),
            Self::Metal(d) => d.synchronize(),
        }
    }

    /// Open a forward pass of shape `key` (`BackendDevice::begin_pass`,
    /// `DESIGN.md` §9.2aa). `Ok(false)` means the device declined and the
    /// pass allocates exactly as it did before — the CPU's answer, and
    /// Metal's until the plan cache is enabled. Pair every `true` with
    /// [`Self::finish_pass`].
    pub fn begin_pass(&self, key: crate::backend::PassShapeKey) -> Result<bool> {
        match self {
            Self::Cpu => Ok(false),
            Self::Cuda(d) => d.begin_pass(key),
            // The inherent `MetalDevice::begin_pass` takes the backend's own
            // key and shadows the trait's; name the trait explicitly.
            Self::Metal(d) => crate::backend::BackendDevice::begin_pass(d, key),
        }
    }

    /// Close the pass [`Self::begin_pass`] opened.
    pub fn finish_pass(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.finish_pass(),
            Self::Metal(d) => crate::backend::BackendDevice::finish_pass(d),
        }
    }

    /// Admit computing at `dtype` here (`BackendDevice::admit_compute_dtype`):
    /// Metal refuses a type its GPU only emulates unless the operator opted
    /// in (§15.2a).
    pub fn admit_compute_dtype(&self, dtype: DType) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Cuda(d) => d.admit_compute_dtype(dtype),
            Self::Metal(d) => d.admit_compute_dtype(dtype),
        }
    }

    /// Raise the Metal command-buffer cut period to the PIPELINED loop's for
    /// the enclosing scope, restoring the default on drop (lloom #715). For a
    /// loop that encodes step `s+1` while the GPU runs step `s` a cut buys no
    /// overlap and costs a seam; for a loop that waits on every step -- and
    /// for a prefill -- the default period IS the overlap, so this is held
    /// only around a pipelined step loop (`Commands`' two defaults). A no-op
    /// on other backends.
    /// Hold the cut period at `period` until the returned guard drops — the
    /// adaptive-commit loop (lloom #735) holds it at `usize::MAX`, so the only
    /// cuts a decode step sees are the ones the loop makes when the GPU has
    /// started the previous buffer. The prefill and every other caller keep
    /// the default. No-op off Metal.
    /// Arm the EOS gate for a decode run (lloom #735 step 4,
    /// `kernels::metal::DispatchGate`): allocate its flag once, clear it,
    /// switch the decode kernels to their gated variants. Refused off Metal
    /// or with an ICB executor installed. Precondition: no gated work in
    /// flight (the previous run drained).
    pub fn eos_gate_begin(&self) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => d.eos_gate_begin(),
            _ => crate::bail!("eos gate: not a Metal device"),
        }
    }

    /// Between a step's argmax and the next forward: encode the gate kernel
    /// on `token` (the U32 scalar the next forward consumes) against `eos_ids`
    /// (U32, the stop set). Every gated kernel encoded after it reads the flag
    /// the kernel left.
    pub fn eos_gate_step(&self, token: &crate::Tensor, eos_ids: &crate::Tensor) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                if token.dtype() != DType::U32 || eos_ids.dtype() != DType::U32 {
                    crate::bail!(
                        "eos gate: token and eos_ids must be U32 (got {:?}, {:?})",
                        token.dtype(),
                        eos_ids.dtype()
                    );
                }
                let n_eos = eos_ids.elem_count();
                let (t_storage, t_layout) = token.storage_and_layout();
                let (e_storage, e_layout) = eos_ids.storage_and_layout();
                match (&*t_storage, &*e_storage) {
                    (Storage::Metal(t), Storage::Metal(e)) => d.eos_gate_step(
                        t,
                        t_layout.start_offset() * DType::U32.size_in_bytes(),
                        e,
                        e_layout.start_offset() * DType::U32.size_in_bytes(),
                        n_eos,
                    ),
                    _ => crate::bail!("eos gate: token and eos_ids must live on the Metal device"),
                }
            }
            _ => {
                let _ = (token, eos_ids);
                crate::bail!("eos gate: not a Metal device")
            }
        }
    }

    /// `words_publish` (Metal): `n` U32 words of `src` copied into the
    /// CPU-mapped `out`, then `generation` stamped into `step` by a dispatch
    /// ordered after the copy (a `ring_stamp` bound to `out`), for a host
    /// that polls `step` and reads `out` in place — the speculative verify's
    /// readback. A host that sees the stamp reads this publish's words.
    pub fn words_publish(
        &self,
        src: &crate::Tensor,
        n: usize,
        out: &crate::Tensor,
        step: &crate::Tensor,
        generation: u32,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                for (name, t) in [("src", src), ("out", out), ("step", step)] {
                    if t.dtype() != DType::U32 {
                        crate::bail!("words_publish: {name} must be U32, got {:?}", t.dtype());
                    }
                    if !t.is_contiguous() {
                        crate::bail!("words_publish: {name} must be contiguous");
                    }
                }
                if src.elem_count() < n || out.elem_count() < n {
                    crate::bail!(
                        "words_publish: {n} words from {} into {}",
                        src.elem_count(),
                        out.elem_count()
                    );
                }
                let (ss, sl) = src.storage_and_layout();
                let (os, ol) = out.storage_and_layout();
                let (ps, pl) = step.storage_and_layout();
                let u = DType::U32.size_in_bytes();
                match (&*ss, &*os, &*ps) {
                    (Storage::Metal(s), Storage::Metal(o), Storage::Metal(p)) => d.words_publish(
                        (s, sl.start_offset() * u),
                        n,
                        (o, ol.start_offset() * u),
                        (p, pl.start_offset() * u),
                        generation,
                    ),
                    _ => crate::bail!("words_publish: every tensor must live on the Metal device"),
                }
            }
            _ => {
                let _ = (src, n, out, step, generation);
                crate::bail!("words_publish: Metal only")
            }
        }
    }

    /// A token ring's generation stamp that certifies slots other dispatches
    /// wrote: `generation` into `step` (U32, one element), ordered after the
    /// last writer of every tensor in `certified` (one or two, any dtype). The
    /// certified tensors are bound and not read; the command stream orders a
    /// dispatch only through what it binds (`DESIGN.md` §6.4), and a stamp
    /// bound to nothing but `step` may run before the slots it vouches for when
    /// the slots are written by compute kernels. Metal only.
    pub fn ring_stamp(
        &self,
        step: &crate::Tensor,
        generation: u32,
        certified: &[&crate::Tensor],
    ) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                if step.dtype() != DType::U32 || step.elem_count() != 1 {
                    crate::bail!(
                        "ring_stamp: step must be one U32, got {:?} {:?}",
                        step.dtype(),
                        step.shape()
                    );
                }
                let (a, b) = match certified {
                    [a] => (*a, *a),
                    [a, b] => (*a, *b),
                    _ => crate::bail!(
                        "ring_stamp: one or two certified tensors, got {}",
                        certified.len()
                    ),
                };
                let (ss, sl) = step.storage_and_layout();
                let (as_, al) = a.storage_and_layout();
                let (bs, bl) = b.storage_and_layout();
                match (&*ss, &*as_, &*bs) {
                    (Storage::Metal(s), Storage::Metal(sa), Storage::Metal(sb)) => d.ring_stamp(
                        (s, sl.start_offset() * DType::U32.size_in_bytes()),
                        generation,
                        [
                            (sa, al.start_offset() * a.dtype().size_in_bytes()),
                            (sb, bl.start_offset() * b.dtype().size_in_bytes()),
                        ],
                    ),
                    _ => crate::bail!("ring_stamp: every tensor must live on the Metal device"),
                }
            }
            _ => {
                let _ = (step, generation, certified);
                crate::bail!("ring_stamp: not a Metal device")
            }
        }
    }

    /// The fused publish (lloom #735 step 5): one dispatch that takes the
    /// argmax of `logits` (F16 or F32, contiguous), writes the token to
    /// `token_slot` and `history_slot` (U32 views of one element), raises the
    /// EOS gate's flag if the token is in `eos_ids` (U32; only while the gate
    /// is armed), fences, and stamps `generation` into `step` (U32, one
    /// element) unless `generation` is 0 — the chain of `cast`, `argmax`, three
    /// blits and the gate kernel as one launch. The fence orders the stamp
    /// after the slots for the GPU, not for a host reading mapped memory; a
    /// stamp a host polls is `ring_stamp`'s, bound to the slots' buffer.
    /// The argmax follows `reduce.metal`'s rule (the
    /// greatest value, the lowest index on ties). Metal only.
    pub fn argmax_publish(
        &self,
        logits: &crate::Tensor,
        token_slot: &crate::Tensor,
        history_slot: &crate::Tensor,
        eos_ids: &crate::Tensor,
        step: &crate::Tensor,
        generation: u32,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                for (name, t) in [
                    ("token_slot", token_slot),
                    ("history_slot", history_slot),
                    ("eos_ids", eos_ids),
                    ("step", step),
                ] {
                    if t.dtype() != DType::U32 {
                        crate::bail!("argmax_publish: {name} must be U32, got {:?}", t.dtype());
                    }
                }
                if !logits.is_contiguous() {
                    crate::bail!("argmax_publish: logits must be contiguous");
                }
                let n = logits.elem_count();
                let n_eos = eos_ids.elem_count();
                let (ls, ll) = logits.storage_and_layout();
                let (ts, tl) = token_slot.storage_and_layout();
                let (hs, hl) = history_slot.storage_and_layout();
                let (es, el) = eos_ids.storage_and_layout();
                let (ss, sl) = step.storage_and_layout();
                let u = DType::U32.size_in_bytes();
                match (&*ls, &*ts, &*hs, &*es, &*ss) {
                    (
                        Storage::Metal(l),
                        Storage::Metal(t),
                        Storage::Metal(h),
                        Storage::Metal(e),
                        Storage::Metal(s),
                    ) => d.argmax_publish(
                        (l, ll.start_offset() * logits.dtype().size_in_bytes()),
                        n,
                        (t, tl.start_offset() * u),
                        (h, hl.start_offset() * u),
                        (e, el.start_offset() * u, n_eos),
                        (s, sl.start_offset() * u),
                        generation,
                    ),
                    _ => crate::bail!("argmax_publish: every tensor must live on the Metal device"),
                }
            }
            _ => {
                let _ = (logits, token_slot, history_slot, eos_ids, step, generation);
                crate::bail!("argmax_publish: not a Metal device")
            }
        }
    }

    /// While the returned guard lives, the kernels this thread encodes run
    /// whatever the EOS gate decided (`kernels::kernel::UngatedScope`); a
    /// no-op off Metal.
    pub fn eos_gate_suspended(&self) -> EosGateSuspended {
        EosGateSuspended {
            #[cfg(feature = "metal")]
            _scope: matches!(self, Self::Metal(_)).then(crate::kernels::kernel::UngatedScope::open),
        }
    }

    /// Whether the EOS gate is armed: the decode kernels that carry it run
    /// their gated variants. Never off Metal.
    pub fn eos_gate_armed(&self) -> bool {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => d.eos_gate_armed(),
            _ => false,
        }
    }

    /// The gate's cores arm for the armed run (`MetalDevice::eos_gate_cores`):
    /// the quantized mat-vec cores are gated too. No-op off Metal.
    pub fn eos_gate_cores(&self, on: bool) {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => d.eos_gate_cores(on),
            _ => {
                let _ = on;
            }
        }
    }

    /// Disarm the gate and report what it did. `None` off Metal. Precondition
    /// as `eos_gate_begin`'s: the run drained, so `fired` is the last gate's.
    pub fn eos_gate_end(&self) -> Option<EosGateReport> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => Some(d.eos_gate_end()),
            _ => None,
        }
    }

    pub fn cut_period_hold(&self, period: usize) -> PipelinedCutPeriod {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                let restore = d.compute_per_buffer();
                d.set_compute_per_buffer(period);
                PipelinedCutPeriod {
                    device: Some(d.clone()),
                    restore,
                }
            }
            _ => PipelinedCutPeriod {
                #[cfg(feature = "metal")]
                device: None,
                restore: period,
            },
        }
    }

    pub fn pipelined_cut_period(&self) -> PipelinedCutPeriod {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(d) => {
                let restore = d.compute_per_buffer();
                d.set_compute_per_buffer(d.pipelined_compute_per_buffer());
                PipelinedCutPeriod {
                    device: Some(d.clone()),
                    restore,
                }
            }
            _ => PipelinedCutPeriod {
                #[cfg(feature = "metal")]
                device: None,
                restore: 0,
            },
        }
    }
}

/// The guard [`Device::pipelined_cut_period`] returns; restores the period it
/// found on drop.
#[must_use = "the period is restored when this is dropped; bind it to `_pipelined`, not `_`"]
pub struct PipelinedCutPeriod {
    #[cfg(feature = "metal")]
    device: Option<crate::MetalDevice>,
    #[allow(dead_code)]
    restore: usize,
}

impl Drop for PipelinedCutPeriod {
    fn drop(&mut self) {
        #[cfg(feature = "metal")]
        if let Some(d) = &self.device {
            d.set_compute_per_buffer(self.restore);
        }
    }
}
