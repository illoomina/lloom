//! The fused ops of the Qwen4-exp blocks: one Metal launch each
//! (`lloom-tensor/src/kernels/metal_src/qwen4exp.metal`), with the composed
//! tensor-op form as the CPU path AND the reference every kernel is tested
//! against (`deltanet.rs`'s tests, `qwen4exp_dispatch_census.rs`).
//!
//! Why: the decode step was ~5.7k launches a token and the chunk prefill
//! ~2.6k `copy2d` per prompt token, 89 % of the step GPU-busy on launches of
//! `[1, 1, 2560]` elementwise kernels (`design/.dev/serving-catalogue.md`
//! §0). Each op here takes one block's chain of launches down to one.

use crate::nn::recurrent::PoolLayer;
use crate::nn::short_conv::{Shift, ShortConv};
use lloom_tensor::{shape_req, CpuStorage, Layout, Result, Shape, Tensor};

/// `(I + A)⁻¹` for strictly-lower `A [.., n, n]` (f32, contiguous) as one
/// launch — `deltanet::unit_lower_inverse`'s forward substitution.
pub(crate) struct UnitLowerInverse;

impl lloom_tensor::CustomOp1 for UnitLowerInverse {
    fn name(&self) -> &'static str {
        "qwen4exp-unit-lower-inverse"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "unit_lower_inverse: no cpu kernel; `deltanet::unit_lower_inverse` composes it there"
        )
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let dims = l.dims();
        let n = dims[dims.len() - 1];
        if dims.len() < 2 || dims[dims.len() - 2] != n {
            lloom_tensor::bail!("unit_lower_inverse: not square {:?}", l.shape())
        }
        if !l.is_contiguous() {
            lloom_tensor::bail!("unit_lower_inverse: the input must be contiguous")
        }
        if s.dtype() != lloom_tensor::DType::F32 {
            lloom_tensor::bail!("unit_lower_inverse: f32 only, got {:?}", s.dtype())
        }
        let batch: usize = dims[..dims.len() - 2].iter().product();
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("unit_lower_inverse");
        let elem_count = l.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s.dtype())
            .with_label("unit_lower_inverse")
            .build()?;
        lloom_tensor::kernels::call_unit_lower_inverse(
            device.metal_device(),
            &encoder,
            device.kernels(),
            batch,
            n,
            lloom_tensor::kernels::BufferOffset {
                buffer: s.buffer(),
                offset_in_bytes: l.start_offset() * s.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            l.shape().clone(),
        ))
    }
}

/// Whether the fused inverse serves `a`: Metal, f32, contiguous, square.
pub(crate) fn unit_lower_inverse_applies(a: &Tensor) -> bool {
    let dims = a.dims();
    !fused_off()
        && a.device().is_metal()
        && a.dtype() == lloom_tensor::DType::F32
        && a.is_contiguous()
        && dims.len() >= 2
        && dims[dims.len() - 1] == dims[dims.len() - 2]
}

pub(crate) fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    a.apply_op1_no_bwd(&UnitLowerInverse)
}

// ---- HyperConnection brackets ---------------------------------------------------

#[cfg(feature = "metal")]
fn metal_bo<'a>(
    st: &'a lloom_tensor::MetalStorage,
    l: &Layout,
) -> lloom_tensor::kernels::BufferOffset<'a> {
    use lloom_tensor::backend::BackendStorage;
    lloom_tensor::kernels::BufferOffset {
        buffer: st.buffer(),
        offset_in_bytes: l.start_offset() * st.dtype().size_in_bytes(),
    }
}

/// `QWEN4EXP_FUSED_OFF=1` (read once): every fused op declines and the
/// composed path runs — the A/B arm for the fused kernels, an instrument
/// behind a gate (§3.3a), never the shipping path.
pub(crate) fn fused_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| crate::tuning::tuning().qwen4exp.fused_off)
}

/// A fused op serves a carry-shaped f32 tensor on Metal when it is
/// contiguous; the composed path takes everything else.
fn fused_applies(t: &Tensor) -> bool {
    fused_applies_float(t) && t.dtype() == lloom_tensor::DType::F32
}

/// [`fused_applies`] for an op instantiated at f32 and f16 (`silu_mul`, the
/// routed output `moe_combine` reads): either float dtype.
fn fused_applies_float(t: &Tensor) -> bool {
    !fused_off()
        && t.device().is_metal()
        && matches!(
            t.dtype(),
            lloom_tensor::DType::F32 | lloom_tensor::DType::F16
        )
        && t.is_contiguous()
}

/// The kernels' float type of a tensor dtype a `fused_applies_float` op took.
#[cfg(feature = "metal")]
fn kernel_float(dt: lloom_tensor::DType) -> Result<lloom_tensor::kernels::DType> {
    match dt {
        lloom_tensor::DType::F32 => Ok(lloom_tensor::kernels::DType::F32),
        lloom_tensor::DType::F16 => Ok(lloom_tensor::kernels::DType::F16),
        other => lloom_tensor::bail!("qwen4exp fused op: no {other:?} instantiation"),
    }
}

/// The grouped RMS norm `x [.., n·d] · rsqrt(mean_d(x²) + eps) · γ[n·d]`.
pub(crate) struct HcNorm {
    pub n: usize,
    pub d: usize,
    pub eps: f32,
}

impl lloom_tensor::CustomOp2 for HcNorm {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-norm"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("hc_norm: no cpu kernel; `hc::grouped_rms_norm` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        if l1.dims().last() != Some(&nd) || l2.shape().elem_count() != nd {
            lloom_tensor::bail!(
                "hc_norm: x {:?} / gamma {:?} vs n·d {nd}",
                l1.shape(),
                l2.shape()
            )
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            lloom_tensor::bail!("hc_norm: contiguous inputs only")
        }
        let rows = l1.shape().elem_count() / nd;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_norm");
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_norm")
            .build()?;
        lloom_tensor::kernels::call_hc_norm(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            l1.shape().clone(),
        ))
    }
}

/// `mean_s(xn · gate)` over the `n` streams: `[.., n·d] → [.., d]`.
/// The HC prologue's first fused launch (`hc_mix_a_f32`): over the carry
/// `h [rows, n·d]` — the grouped norm (γ), the down projection with SiLU and
/// `scale`, the inject projection — into one flat buffer `[rows·n·d] xn |
/// [rows·lr] lo | [rows·n] inject`. The weights ride in the op (as
/// `GemvMixed` carries its weight): contiguous f16 `[lr, n·d]` and, when
/// present, `[n, n·d]`.
pub(crate) struct HcMixA {
    pub gamma: Tensor,
    pub w_down: Tensor,
    pub w_inject: Option<Tensor>,
    pub n: usize,
    pub d: usize,
    pub eps: f32,
    pub scale: f32,
}

#[cfg(feature = "metal")]
fn metal_weight<'a>(
    w: &'a Tensor,
    what: &str,
) -> Result<(
    parking_lot::RwLockReadGuard<'a, lloom_tensor::Storage>,
    &'a Layout,
)> {
    use lloom_tensor::backend::BackendStorage;
    let (ws, wl) = w.storage_and_layout();
    match &*ws {
        lloom_tensor::Storage::Metal(m) if m.dtype() == lloom_tensor::DType::F16 => {}
        _ => lloom_tensor::bail!("{what}: the weight must be f16 on the metal device"),
    }
    if !wl.is_contiguous() {
        lloom_tensor::bail!("{what}: the weight must be contiguous")
    }
    Ok((ws, wl))
}

#[cfg(feature = "metal")]
fn metal_bo_of<'a>(
    ws: &'a lloom_tensor::Storage,
    wl: &Layout,
) -> lloom_tensor::kernels::BufferOffset<'a> {
    use lloom_tensor::backend::BackendStorage;
    match ws {
        lloom_tensor::Storage::Metal(m) => lloom_tensor::kernels::BufferOffset {
            buffer: m.buffer(),
            offset_in_bytes: wl.start_offset() * m.dtype().size_in_bytes(),
        },
        _ => unreachable!("checked by metal_weight"),
    }
}

impl lloom_tensor::CustomOp1 for HcMixA {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-mix-a"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("hc_mix_a: no cpu kernel; `HyperConnection::mix` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        if l.dims().last() != Some(&nd) || !l.is_contiguous() {
            lloom_tensor::bail!("hc_mix_a: h {:?} vs n·d {nd}, contiguous", l.shape())
        }
        let rows = l.shape().elem_count() / nd;
        let (lr, k) = self.w_down.dims2()?;
        if k != nd {
            lloom_tensor::bail!("hc_mix_a: w_down {:?} vs n·d {nd}", self.w_down.shape())
        }
        if let Some(w) = &self.w_inject {
            if w.dims2()? != (self.n, nd) {
                lloom_tensor::bail!("hc_mix_a: w_inject {:?} vs [{}, {nd}]", w.shape(), self.n)
            }
        }
        let (gs, gl) = self.gamma.storage_and_layout();
        let gs_m = match &*gs {
            lloom_tensor::Storage::Metal(m)
                if m.dtype() == lloom_tensor::DType::F32 && gl.is_contiguous() =>
            {
                m
            }
            _ => lloom_tensor::bail!("hc_mix_a: γ must be contiguous f32 on the metal device"),
        };
        if gl.shape().elem_count() != nd {
            lloom_tensor::bail!("hc_mix_a: γ {:?} vs n·d {nd}", gl.shape())
        }
        let (ds, dl) = metal_weight(&self.w_down, "hc_mix_a")?;
        let inj = match &self.w_inject {
            Some(w) => Some(metal_weight(w, "hc_mix_a")?),
            None => None,
        };
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_mix_a");
        let elem_count = rows * (nd + lr + self.n);
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_mix_a")
            .build()?;
        let w_down = metal_bo_of(&ds, dl);
        // Unread when there is no inject weight: the down weight stands in.
        let w_inject = match &inj {
            Some((is_, il)) => metal_bo_of(is_, il),
            None => metal_bo_of(&ds, dl),
        };
        lloom_tensor::kernels::call_hc_mix_a(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            lr,
            if inj.is_some() { self.n } else { 0 },
            self.eps,
            self.scale,
            metal_bo(s, l),
            lloom_tensor::kernels::BufferOffset {
                buffer: gs_m.buffer(),
                offset_in_bytes: gl.start_offset() * 4,
            },
            w_down,
            w_inject,
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from(elem_count),
        ))
    }
}

/// The HC prologue's second fused launch (`hc_mix_b_f32`): from `HcMixA`'s
/// buffer, the up projection (sigmoid) and the gated collapse over the
/// streams → `[rows, d]`. `w_up` contiguous f16 `[n·d, lr]`.
pub(crate) struct HcMixB {
    pub w_up: Tensor,
    pub n: usize,
    pub d: usize,
    pub lr: usize,
}

impl lloom_tensor::CustomOp1 for HcMixB {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-mix-b"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("hc_mix_b: no cpu kernel; `HyperConnection::mix` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        let per_row = nd + self.lr + self.n;
        let total = l.shape().elem_count();
        if !l.is_contiguous() || total % per_row != 0 {
            lloom_tensor::bail!("hc_mix_b: packed {:?} vs rows × {per_row}", l.shape())
        }
        let rows = total / per_row;
        if self.w_up.dims2()? != (nd, self.lr) {
            lloom_tensor::bail!(
                "hc_mix_b: w_up {:?} vs [{nd}, {}]",
                self.w_up.shape(),
                self.lr
            )
        }
        let (us, ul) = metal_weight(&self.w_up, "hc_mix_b")?;
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_mix_b");
        let elem_count = rows * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_mix_b")
            .build()?;
        lloom_tensor::kernels::call_hc_mix_b(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            self.lr,
            1.0 / self.n as f32,
            metal_bo(s, l),
            metal_bo_of(&us, ul),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from((rows, self.d)),
        ))
    }
}

/// `QWEN4EXP_HC_MIX=1` (read once) takes the fused two-launch prologue;
/// the default is the composed chain. Measured in situ (#749 P29 (3), a
/// rotation on one binary): the pair removes three launches a bracket and
/// costs +1.06 ms of GPU a token (+3.6 % on the decode) — the mixed gemv
/// runs the bracket's 13 MB near bandwidth and the fused dots do not
/// (306 / 340 GB/s against 520 in the bench). Kept as the arm the next
/// kernel cut is measured against.
pub(crate) fn hc_mix_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.hc_mix)
}

/// The most rows the fused HC prologue takes: one — the kernels' grid is a
/// threadgroup set per row, each streaming the weights again, where the
/// mixed gemv's tile shares them across up to four rows (measured: 4 rows
/// fused 133 µs against 82 composed, `prologue_bench`). A batch's rows and a
/// prefill's take the composed chain.
pub(crate) const HC_MIX_MAX_ROWS: usize = 1;

/// The most streams the fused HC prologue takes: `hc_mix_a` holds one scale
/// per stream in a threadgroup array of `HC_MAX_N` (`qwen4exp.metal`), and a
/// carry of more streams writes past it. The checkpoint has four.
pub(crate) const HC_MIX_MAX_N: usize = 8;

/// The HC prologue as two launches (`HcMixA`, `HcMixB`) when the bracket's
/// weights are f16 on Metal and the carry is a decode's rows: `(mixed
/// [b, seq, d], inject [b, seq, n])`. `None` elsewhere (the composed chain).
pub(crate) fn hc_mix_fused(
    h: &Tensor,
    gamma: &Tensor,
    down: &lloom_tensor::quantized::QMatMul,
    up: &lloom_tensor::quantized::QMatMul,
    inject: Option<&lloom_tensor::quantized::QMatMul>,
    n: usize,
    d: usize,
    eps: f32,
) -> Option<Result<(Tensor, Tensor)>> {
    if !hc_mix_on() {
        return None;
    }
    hc_mix_pair(h, gamma, down, up, inject, n, d, eps)
}

/// [`hc_mix_fused`] without the arm's switch: the pair wherever it applies.
/// The parity test calls this — through [`hc_mix_fused`] it would test the
/// composed chain, the arm being off by default (#749 P33 (6b): the pair's
/// kernels survived a mutation because no test reached them).
#[allow(clippy::too_many_arguments)] // `hc_mix_fused`'s signature, which it serves
pub(crate) fn hc_mix_pair(
    h: &Tensor,
    gamma: &Tensor,
    down: &lloom_tensor::quantized::QMatMul,
    up: &lloom_tensor::quantized::QMatMul,
    inject: Option<&lloom_tensor::quantized::QMatMul>,
    n: usize,
    d: usize,
    eps: f32,
) -> Option<Result<(Tensor, Tensor)>> {
    use lloom_tensor::quantized::QMatMul;
    if !fused_applies(h) || h.rank() != 3 {
        return None;
    }
    let w_down = match down {
        QMatMul::TensorF16(w) if w.rank() == 2 && w.is_contiguous() => w.clone(),
        _ => return None,
    };
    let w_up = match up {
        QMatMul::TensorF16(w) if w.rank() == 2 && w.is_contiguous() => w.clone(),
        _ => return None,
    };
    let w_inject = match inject {
        Some(QMatMul::TensorF16(w)) if w.rank() == 2 && w.is_contiguous() => Some(w.clone()),
        Some(_) => return None,
        None => None,
    };
    let (b, seq, nd) = h.dims3().ok()?;
    let rows = b * seq;
    if nd != n * d
        || rows == 0
        || !shape_req!(hc_mix_fused, rows <= HC_MIX_MAX_ROWS; rows)
        || !shape_req!(hc_mix_fused, n <= HC_MIX_MAX_N; n)
    {
        return None;
    }
    let lr = w_down.dim(0).ok()?;
    if w_down.dim(1).ok()? != nd
        || w_up.dims2().ok()? != (nd, lr)
        || !shape_req!(hc_mix_fused, lr <= 1024; lr)
    {
        return None;
    }
    // The kernels' vector loads: d a multiple of 4, lr even, every operand
    // at a 16-byte-aligned start (whole tensors are; a narrow may not be).
    let aligned = |t: &Tensor, elem: usize| (t.layout().start_offset() * elem) % 16 == 0;
    if !shape_req!(hc_mix_fused, d % 4 == 0; d)
        || !shape_req!(hc_mix_fused, lr % 2 == 0; lr)
        || !shape_req!(hc_mix_fused, aligned(h, 4); offset = h.layout().start_offset())
        || !shape_req!(hc_mix_fused, aligned(gamma, 4); offset = gamma.layout().start_offset())
        || !shape_req!(hc_mix_fused, aligned(&w_down, 2); offset = w_down.layout().start_offset())
        || !shape_req!(hc_mix_fused, aligned(&w_up, 2); offset = w_up.layout().start_offset())
        || !w_inject.as_ref().map_or(
            true,
            |w| shape_req!(hc_mix_fused, aligned(w, 2); offset = w.layout().start_offset()),
        )
    {
        return None;
    }
    if !gamma.is_contiguous()
        || gamma.elem_count() != nd
        || gamma.dtype() != lloom_tensor::DType::F32
    {
        return None;
    }
    Some((|| {
        let packed = h.apply_op1_no_bwd(&HcMixA {
            gamma: gamma.clone(),
            w_down,
            w_inject,
            n,
            d,
            eps,
            scale: 1.0 / n as f32,
        })?; // flat
        let mixed = packed
            .apply_op1_no_bwd(&HcMixB { w_up, n, d, lr })?
            .reshape((b, seq, d))?;
        let inject = packed
            .narrow(0, rows * (nd + lr), rows * n)?
            .reshape((b, seq, n))?;
        Ok((mixed, inject))
    })())
}

pub(crate) struct HcCollapse {
    pub n: usize,
    pub d: usize,
}

impl lloom_tensor::CustomOp2 for HcCollapse {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-collapse"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("hc_collapse: no cpu kernel; `HyperConnection::mix` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        if l1.dims().last() != Some(&nd) || l1.shape() != l2.shape() {
            lloom_tensor::bail!(
                "hc_collapse: xn {:?} / gate {:?} vs n·d {nd}",
                l1.shape(),
                l2.shape()
            )
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            lloom_tensor::bail!("hc_collapse: contiguous inputs only")
        }
        let rows = l1.shape().elem_count() / nd;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_collapse");
        let elem_count = rows * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_collapse")
            .build()?;
        lloom_tensor::kernels::call_hc_collapse(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        let mut dims = l1.dims().to_vec();
        *dims.last_mut().unwrap() = self.d;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from(dims),
        ))
    }
}

/// [`HcCollapse`] over the carry `h` `[.., n·d]` itself, `xn` recomputed in
/// the kernel from γ (held here: a fourth input) and the per-row stream
/// `scales` `[.., n]` a grouped-norm GEMV wrote (`docs/tuning.md` §13.5u).
// The fields are read by the Metal arm only.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) struct HcCollapseScaled {
    pub n: usize,
    pub d: usize,
    pub gamma: Tensor,
}

impl lloom_tensor::CustomOp3 for HcCollapseScaled {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-collapse-scaled"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("hc_collapse_scaled: no cpu kernel; `HyperConnection::mix` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        sh: &lloom_tensor::MetalStorage,
        lh: &Layout,
        ss: &lloom_tensor::MetalStorage,
        ls: &Layout,
        sg: &lloom_tensor::MetalStorage,
        lg: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        if lh.dims().last() != Some(&nd) || lh.shape() != lg.shape() {
            lloom_tensor::bail!(
                "hc_collapse_scaled: h {:?} / gate {:?} vs n·d {nd}",
                lh.shape(),
                lg.shape()
            )
        }
        let rows = lh.shape().elem_count() / nd;
        if ls.shape().elem_count() != rows * self.n {
            lloom_tensor::bail!(
                "hc_collapse_scaled: {} scales for {rows} rows of {} streams",
                ls.shape().elem_count(),
                self.n
            )
        }
        if !lh.is_contiguous() || !ls.is_contiguous() || !lg.is_contiguous() {
            lloom_tensor::bail!("hc_collapse_scaled: contiguous inputs only")
        }
        let (gs, gl) = self.gamma.storage_and_layout();
        let gs = match &*gs {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("hc_collapse_scaled: γ must be on Metal"),
        };
        if gs.dtype() != lloom_tensor::DType::F32 || !gl.is_contiguous() || gl.shape().elem_count() != nd {
            lloom_tensor::bail!("hc_collapse_scaled: γ must be a contiguous f32 [{nd}]")
        }
        let device = sh.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_collapse_scaled");
        let elem_count = rows * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_collapse_scaled")
            .build()?;
        lloom_tensor::kernels::call_hc_collapse_scaled(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            metal_bo(sh, lh),
            metal_bo(gs, gl),
            metal_bo(ss, ls),
            metal_bo(sg, lg),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        let mut dims = lh.dims().to_vec();
        *dims.last_mut().unwrap() = self.d;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, sh.dtype()),
            Shape::from(dims),
        ))
    }
}

/// `h + bo ⊗ 2·sigmoid(inject / n)`: `[.., n·d]`, `[.., d]`, `[.., n]` → `[.., n·d]`.
pub(crate) struct HcCombine {
    pub n: usize,
    pub d: usize,
}

impl lloom_tensor::CustomOp3 for HcCombine {
    fn name(&self) -> &'static str {
        "qwen4exp-hc-combine"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "hc_combine: no cpu kernel; `HyperConnection::combine` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let nd = self.n * self.d;
        if l1.dims().last() != Some(&nd)
            || l2.dims().last() != Some(&self.d)
            || l3.dims().last() != Some(&self.n)
        {
            lloom_tensor::bail!(
                "hc_combine: h {:?} / bo {:?} / inject {:?} vs n {} d {}",
                l1.shape(),
                l2.shape(),
                l3.shape(),
                self.n,
                self.d
            )
        }
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            lloom_tensor::bail!("hc_combine: contiguous inputs only")
        }
        let rows = l1.shape().elem_count() / nd;
        if l2.shape().elem_count() != rows * self.d || l3.shape().elem_count() != rows * self.n {
            lloom_tensor::bail!("hc_combine: row counts differ")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("hc_combine");
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("hc_combine")
            .build()?;
        lloom_tensor::kernels::call_hc_combine(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            self.n,
            self.d,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            l1.shape().clone(),
        ))
    }
}

/// The fused grouped norm, or `None` where the composed path applies.
pub(crate) fn hc_norm(
    x: &Tensor,
    gamma: &Tensor,
    n: usize,
    d: usize,
    eps: f32,
) -> Option<Result<Tensor>> {
    (fused_applies(x) && gamma.is_contiguous() && gamma.dtype() == lloom_tensor::DType::F32)
        .then(|| x.apply_op2_no_bwd(gamma, &HcNorm { n, d, eps }))
}

pub(crate) fn hc_collapse(
    xn: &Tensor,
    gate: &Tensor,
    n: usize,
    d: usize,
) -> Option<Result<Tensor>> {
    (fused_applies(xn) && fused_applies(gate))
        .then(|| xn.apply_op2_no_bwd(gate, &HcCollapse { n, d }))
}

/// Whether [`hc_collapse_scaled`] would run for this carry and γ (the gate
/// is the up GEMV's f32 output, contiguous by construction) — decided before
/// anything is dispatched.
pub(crate) fn hc_collapse_scaled_applies(h: &Tensor, gamma: &Tensor) -> bool {
    fused_applies(h) && gamma.is_contiguous() && gamma.dtype() == lloom_tensor::DType::F32
}

/// The scaled collapse ([`HcCollapseScaled`]), or `None` where the composed
/// path applies.
pub(crate) fn hc_collapse_scaled(
    h: &Tensor,
    gamma: &Tensor,
    scales: &Tensor,
    gate: &Tensor,
    n: usize,
    d: usize,
) -> Option<Result<Tensor>> {
    (fused_applies(h)
        && fused_applies(scales)
        && fused_applies(gate)
        && gamma.is_contiguous()
        && gamma.dtype() == lloom_tensor::DType::F32)
        .then(|| {
            h.apply_op3_no_bwd(
                scales,
                gate,
                &HcCollapseScaled {
                    n,
                    d,
                    gamma: gamma.clone(),
                },
            )
        })
}

pub(crate) fn hc_combine(
    h: &Tensor,
    bo: &Tensor,
    inject: &Tensor,
    n: usize,
    d: usize,
) -> Option<Result<Tensor>> {
    (fused_applies(h) && fused_applies(bo) && fused_applies(inject))
        .then(|| h.apply_op3_no_bwd(bo, inject, &HcCombine { n, d }))
}

// ---- Mixture of experts ------------------------------------------------------------

/// The router's top-`k`: from `logits [.., n_experts]` the ids `[.., k]` u32
/// (the op's output) and their renormalised softmax weights, written into
/// `w` (`[.., k]` f32, the caller's, allocated ahead).
pub(crate) struct MoeTopk {
    pub k: usize,
    pub w: Tensor,
    /// Where the ids go, when the caller owns that buffer: a host-read slot
    /// the host reads in place after the pass, with no copy (#814 part 5).
    /// `None` allocates the output as any op does.
    pub ids: Option<Tensor>,
}

impl lloom_tensor::CustomOp1 for MoeTopk {
    fn name(&self) -> &'static str {
        "qwen4exp-moe-topk"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("moe_topk: no cpu kernel; `MoeLayer::forward` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let dims = l.dims();
        let n_experts = *dims.last().unwrap_or(&0);
        if !l.is_contiguous() || n_experts == 0 {
            lloom_tensor::bail!("moe_topk: contiguous logits only")
        }
        let rows = l.shape().elem_count() / n_experts;
        let (ws, wl) = self.w.storage_and_layout();
        let ws = match &*ws {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("moe_topk: the weight output must be on the metal device"),
        };
        if wl.shape().elem_count() != rows * self.k || !wl.is_contiguous() || wl.start_offset() != 0
        {
            lloom_tensor::bail!("moe_topk: w must be a fresh contiguous [rows, k]")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("moe_topk");
        let elem_count = rows * self.k;
        // The caller's buffer when it brought one (a host-read slot), else
        // this op's own. Checked like `w`: the kernel writes every entry, so
        // a destination of another extent would write past it.
        let ids = match &self.ids {
            Some(t) => {
                let (is, il) = t.storage_and_layout();
                let is = match &*is {
                    lloom_tensor::Storage::Metal(m) => m,
                    _ => {
                        lloom_tensor::bail!("moe_topk: the ids output must be on the metal device")
                    }
                };
                if il.shape().elem_count() != elem_count
                    || !il.is_contiguous()
                    || il.start_offset() != 0
                {
                    lloom_tensor::bail!("moe_topk: ids must be a fresh contiguous [rows, k]")
                }
                if is.dtype() != lloom_tensor::DType::U32 {
                    lloom_tensor::bail!("moe_topk: ids must be U32")
                }
                is.buffer_handle().clone()
            }
            None => device
                .new_buffer_builder()
                .with_size_for(elem_count, lloom_tensor::DType::U32)
                .with_label("moe_topk_ids")
                .build()?,
        };
        lloom_tensor::kernels::call_moe_topk(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            n_experts,
            self.k,
            metal_bo(s, l),
            &ids,
            ws.buffer(),
        )
        .map_err(lloom_tensor::Error::wrap)?;
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = self.k;
        Ok((
            lloom_tensor::MetalStorage::new(
                ids,
                device.clone(),
                elem_count,
                lloom_tensor::DType::U32,
            ),
            Shape::from(out_dims),
        ))
    }
}

/// `silu(g) · u`, elementwise.
pub(crate) struct SiluMul;

impl lloom_tensor::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "qwen4exp-silu-mul"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("silu_mul: no cpu kernel; `silu(g) * u` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        if l1.shape() != l2.shape() || !l1.is_contiguous() || !l2.is_contiguous() {
            lloom_tensor::bail!("silu_mul: same-shape contiguous inputs only")
        }
        if s1.dtype() != s2.dtype() {
            lloom_tensor::bail!("silu_mul: g {:?} and u {:?} differ", s1.dtype(), s2.dtype())
        }
        let dtype = kernel_float(s1.dtype())?;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("silu_mul");
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("silu_mul")
            .build()?;
        lloom_tensor::kernels::call_silu_mul(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            elem_count,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            l1.shape().clone(),
        ))
    }
}

/// `Σ_k w · y + shared · sigmoid(gate)`: `y [.., k, d]`, `w [.., k]`,
/// `shared [.., d]`, `gate [.., 1]` (captured) → `[.., d]`.
pub(crate) struct MoeCombine {
    pub k: usize,
    pub d: usize,
    pub gate: Tensor,
}

impl lloom_tensor::CustomOp3 for MoeCombine {
    fn name(&self) -> &'static str {
        "qwen4exp-moe-combine"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("moe_combine: no cpu kernel; `MoeLayer::forward` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let rows = l3.shape().elem_count() / self.d;
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            lloom_tensor::bail!("moe_combine: contiguous inputs only")
        }
        if l1.shape().elem_count() != rows * self.k * self.d
            || l2.shape().elem_count() != rows * self.k
            || l3.shape().elem_count() != rows * self.d
        {
            lloom_tensor::bail!(
                "moe_combine: y {:?} w {:?} shared {:?} vs k {} d {}",
                l1.shape(),
                l2.shape(),
                l3.shape(),
                self.k,
                self.d
            )
        }
        let (gs, gl) = self.gate.storage_and_layout();
        let gs = match &*gs {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("moe_combine: the gate must be on the metal device"),
        };
        if gl.shape().elem_count() != rows || !gl.is_contiguous() {
            lloom_tensor::bail!("moe_combine: gate must be a contiguous [rows]")
        }
        let y_dtype = kernel_float(s1.dtype())?;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("moe_combine");
        let elem_count = rows * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("moe_combine")
            .build()?;
        lloom_tensor::kernels::call_moe_combine(
            device.metal_device(),
            &encoder,
            device.kernels(),
            y_dtype,
            rows,
            self.k,
            self.d,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            metal_bo(gs, gl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(
                output,
                device.clone(),
                elem_count,
                lloom_tensor::DType::F32,
            ),
            l3.shape().clone(),
        ))
    }
}

/// The fused router selection: `(ids [.., k] u32, w [.., k] f32)`, or `None`
/// where the composed path applies.
pub(crate) fn moe_topk(logits: &Tensor, k: usize) -> Option<Result<(Tensor, Tensor)>> {
    moe_topk_into(logits, k, None)
}

/// [`moe_topk`] writing the ids into `ids` when given — a host-read slot the
/// caller reads in place after the pass, with no copy (#814 part 5).
pub(crate) fn moe_topk_into(
    logits: &Tensor,
    k: usize,
    ids: Option<Tensor>,
) -> Option<Result<(Tensor, Tensor)>> {
    if !fused_applies(logits) || logits.rank() < 1 {
        return None;
    }
    let n_experts = logits.dim(logits.rank() - 1).ok()?;
    if n_experts > 1024 || k == 0 || k > n_experts {
        return None;
    }
    let mut dims = logits.dims().to_vec();
    *dims.last_mut().unwrap() = k;
    Some((|| {
        // Unfilled on purpose: `moe_topk_f32` writes every `[row, k]` entry
        // (one per round, then the renormalisation over all `k`), and a
        // `Tensor::zeros` here was a blit `fill_buffer` on Metal — one
        // encoder break, a full GPU drain, per MoE layer per token (48 of a
        // step's 84 sessions, #749 P31 (1)).
        // SAFETY: every element is written by the kernel before any read;
        // the Metal arm refuses a `w` that is not a fresh contiguous
        // `[rows, k]`, and the CPU arm never takes this path.
        let w =
            unsafe { Tensor::empty(dims.as_slice(), lloom_tensor::DType::F32, logits.device())? };
        let ids = logits.apply_op1_no_bwd(&MoeTopk {
            k,
            w: w.clone(),
            ids,
        })?;
        Ok((ids, w))
    })())
}

/// `silu(g) · u` as one launch, at f32 or f16 (`g` and `u` of one dtype);
/// `None` where the composed `silu` and multiply apply.
pub(crate) fn silu_mul(g: &Tensor, u: &Tensor) -> Option<Result<Tensor>> {
    (fused_applies_float(g)
        && fused_applies_float(u)
        && g.dtype() == u.dtype()
        && g.shape() == u.shape())
    .then(|| g.apply_op2_no_bwd(u, &SiluMul))
}

/// The routed combine plus the gated shared expert as one launch, into f32:
/// `y` f32 or f16 (the routed store, read without a cast), the rest f32;
/// `None` where the composed path applies.
pub(crate) fn moe_combine(
    y: &Tensor,
    w: &Tensor,
    shared: &Tensor,
    gate: &Tensor,
    k: usize,
    d: usize,
) -> Option<Result<Tensor>> {
    (fused_applies_float(y) && fused_applies(w) && fused_applies(shared) && fused_applies(gate))
        .then(|| {
            y.apply_op3_no_bwd(
                w,
                shared,
                &MoeCombine {
                    k,
                    d,
                    gate: gate.clone(),
                },
            )
        })
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::nn::ops;
    use lloom_tensor::{Device, D};

    fn vals(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.0173 + seed).sin() * 7.1).cos() * scale)
            .collect()
    }

    fn close(name: &str, a: &Tensor, b: &Tensor, tol: f32) {
        let x: Vec<f32> = a
            .flatten_all()
            .unwrap()
            .to_dtype(lloom_tensor::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let y: Vec<f32> = b
            .flatten_all()
            .unwrap()
            .to_dtype(lloom_tensor::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(x.len(), y.len(), "{name}: lengths");
        let scale = y.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        let diff = x
            .iter()
            .zip(&y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(
            diff <= tol * scale,
            "{name}: max diff {diff} at scale {scale}"
        );
        assert!(scale > 1e-3, "{name}: degenerate");
    }

    /// The DeltaNet step's three kernels at the real geometry, each timed
    /// alone (ignored: a measurement, `--ignored --nocapture` under
    /// `lloom-arb measure`).
    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn gdn_kernels_timed_at_real_geometry() {
        let dev = Device::new_metal(0).unwrap();
        let (d, h_k, h_v, kk) = (128usize, 16usize, 48usize, 4usize);
        let conv_ch = (2 * h_k + h_v) * d;
        let f32 = lloom_tensor::DType::F32;
        let qkv = Tensor::randn(0f32, 0.3, conv_ch, &dev).unwrap();
        let conv =
            ShortConv::<Shift>::new(Tensor::randn(0f32, 0.2, (conv_ch, kk), &dev).unwrap(), kk);
        let win = Tensor::zeros((conv_ch, kk), f32, &dev).unwrap();
        let win_out = Tensor::zeros((conv_ch, kk), f32, &dev).unwrap();
        let slab = Tensor::zeros((d, d, h_v), f32, &dev).unwrap();
        let slab_out = Tensor::zeros((d, d, h_v), f32, &dev).unwrap();
        let alpha = Tensor::randn(0f32, 0.3, h_v, &dev).unwrap();
        let beta = Tensor::randn(0f32, 0.3, h_v, &dev).unwrap();
        let dt_bias = Tensor::zeros(h_v, f32, &dev).unwrap();
        let ssm_a = Tensor::full(-0.5f32, h_v, &dev).unwrap();
        let gamma = Tensor::ones(d, f32, &dev).unwrap();
        let z = Tensor::randn(0f32, 0.3, h_v * d, &dev).unwrap();
        let heads = qkv
            .apply_op1_no_bwd(&GdnPrologue {
                d,
                h_k,
                h_v,
                eps: 1e-6,
                q_scale: 0.088,
                conv: conv.clone(),
                win: win.clone(),
                win_out: win_out.clone(),
            })
            .unwrap();
        let o = Tensor::zeros((h_v, d), f32, &dev).unwrap();
        let step =
            crate::nn::gdn::GdnStep::<crate::nn::gdn::LoopForm>::new(d, h_v, &dt_bias, &ssm_a);
        let time = |name: &str, f: &dyn Fn()| {
            for _ in 0..5 {
                f();
            }
            dev.synchronize().unwrap();
            let n = 200;
            let t = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            dev.synchronize().unwrap();
            println!(
                "GDNTIME {name} {:.1} us",
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            );
        };
        time("prologue", &|| {
            let _ = qkv
                .apply_op1_no_bwd(&GdnPrologue {
                    d,
                    h_k,
                    h_v,
                    eps: 1e-6,
                    q_scale: 0.088,
                    conv: conv.clone(),
                    win: win.clone(),
                    win_out: win_out.clone(),
                })
                .unwrap();
        });
        time("step", &|| {
            let _ = step.step(&heads, &alpha, &beta, &slab, &slab_out).unwrap();
        });
        time("gated_norm", &|| {
            let _ = o
                .apply_op3_no_bwd(&gamma, &z, &GdnGatedNorm { d, h_v, eps: 1e-6 })
                .unwrap();
        });
    }

    /// The router's fused top-k picks the ids the composed sort picks and
    /// renormalises their weights the same way, at the real width (512
    /// experts, top 10) over several rows.
    #[test]
    fn moe_topk_matches_the_composed_selection() {
        let dev = Device::new_metal(0).unwrap();
        let (rows, e, k) = (3usize, 512usize, 10usize);
        let logits = Tensor::from_vec(vals(rows * e, 0.3, 3.0), (rows, e), &dev).unwrap();
        let (ids, w) = moe_topk(&logits, k).expect("metal f32").unwrap();
        let probs = ops::softmax_last_dim(&logits).unwrap();
        let ids_ref = probs
            .arg_sort_last_dim(false)
            .unwrap()
            .narrow(D::Minus1, 0, k)
            .unwrap()
            .contiguous()
            .unwrap();
        let w_ref = probs.gather(&ids_ref, D::Minus1).unwrap();
        let w_ref = w_ref
            .broadcast_div(&w_ref.sum_keepdim(D::Minus1).unwrap())
            .unwrap();
        let a: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<u32> = ids_ref.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b, "the selected experts differ");
        close("weights", &w, &w_ref, 1e-5);
        let ws: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        for r in 0..rows {
            let s: f32 = ws[r * k..(r + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} weights sum {s}");
        }
    }

    #[test]
    fn silu_mul_and_combine_match_composed() {
        let dev = Device::new_metal(0).unwrap();
        let n = 3 * 10 * 640;
        let g = Tensor::from_vec(vals(n, 1.0, 2.0), (3, 10, 640), &dev).unwrap();
        let u = Tensor::from_vec(vals(n, 2.0, 1.5), (3, 10, 640), &dev).unwrap();
        let fused = silu_mul(&g, &u).expect("metal f32").unwrap();
        let composed = (ops::silu(&g).unwrap() * &u).unwrap();
        close("silu_mul", &fused, &composed, 1e-6);

        let (rows, k, d) = (3usize, 10usize, 2560usize);
        let y = Tensor::from_vec(vals(rows * k * d, 3.0, 1.0), (rows, k, d), &dev).unwrap();
        let w = Tensor::from_vec(vals(rows * k, 4.0, 0.3), (rows, k), &dev).unwrap();
        let shared = Tensor::from_vec(vals(rows * d, 5.0, 0.8), (rows, d), &dev).unwrap();
        let gate = Tensor::from_vec(vals(rows, 6.0, 2.0), (rows, 1), &dev).unwrap();
        let fused = moe_combine(&y, &w, &shared, &gate, k, d)
            .expect("metal f32")
            .unwrap();
        let routed = y
            .broadcast_mul(&w.reshape((rows, k, 1)).unwrap())
            .unwrap()
            .sum(1)
            .unwrap();
        let composed =
            (routed + shared.broadcast_mul(&ops::sigmoid(&gate).unwrap()).unwrap()).unwrap();
        close("moe_combine", &fused, &composed, 1e-5);
    }

    fn bits16(t: &Tensor) -> Vec<u16> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn bits32(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// The f16 SwiGLU in one launch is `silu_f16` then `bmul_f16` bit for
    /// bit — the two launches the f16 routed step ran — over every finite
    /// f16 and both infinities as `g`, each against several `u`.
    #[test]
    fn silu_mul_f16_is_silu_then_mul_bit_for_bit() {
        let dev = Device::new_metal(0).unwrap();
        let gs: Vec<half::f16> = (0..=u16::MAX)
            .map(half::f16::from_bits)
            .filter(|v| !v.is_nan())
            .collect();
        let us = [1.0f32, -0.75, 3.1e-2, 6.5e4, 1.0e-5, -2.4];
        let n = gs.len() * us.len();
        let g: Vec<half::f16> = (0..n).map(|i| gs[i % gs.len()]).collect();
        let u: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32(us[i / gs.len()]))
            .collect();
        let g = Tensor::from_vec(g, (us.len(), gs.len()), &dev).unwrap();
        let u = Tensor::from_vec(u, (us.len(), gs.len()), &dev).unwrap();
        let fused = silu_mul(&g, &u).expect("metal f16").unwrap();
        assert_eq!(fused.dtype(), lloom_tensor::DType::F16);
        let composed = (ops::silu(&g).unwrap() * &u).unwrap();
        let (a, b) = (bits16(&fused), bits16(&composed));
        let differ: Vec<usize> = (0..n).filter(|&i| a[i] != b[i]).collect();
        assert!(
            differ.is_empty(),
            "silu_mul_f16 differs from silu_f16 · bmul_f16 at {} of {n}, first at g={:?} u={:?}: {:#06x} vs {:#06x}",
            differ.len(),
            gs[differ[0] % gs.len()],
            us[differ[0] / gs.len()],
            a[differ[0]],
            b[differ[0]]
        );
        // The real step's shape, `[rows, topk, n_ff]`, through the same check.
        let g = Tensor::from_vec(vals(10 * 640, 1.0, 6.0), (1, 10, 640), &dev)
            .unwrap()
            .to_dtype(lloom_tensor::DType::F16)
            .unwrap();
        let u = Tensor::from_vec(vals(10 * 640, 2.0, 1.5), (1, 10, 640), &dev)
            .unwrap()
            .to_dtype(lloom_tensor::DType::F16)
            .unwrap();
        let fused = silu_mul(&g, &u).expect("metal f16").unwrap();
        let composed = (ops::silu(&g).unwrap() * &u).unwrap();
        assert_eq!(bits16(&fused), bits16(&composed));
        assert!(silu_mul(&g, &u.to_dtype(lloom_tensor::DType::F32).unwrap()).is_none());
    }

    /// The combine reading an f16 routed output is the combine of that output
    /// cast to f32 bit for bit (the cast launch it replaces), its own output
    /// f32.
    #[test]
    fn moe_combine_reads_an_f16_y_as_its_f32_cast() {
        let dev = Device::new_metal(0).unwrap();
        let (rows, k, d) = (3usize, 10usize, 2560usize);
        let y16 = Tensor::from_vec(vals(rows * k * d, 3.0, 40.0), (rows, k, d), &dev)
            .unwrap()
            .to_dtype(lloom_tensor::DType::F16)
            .unwrap();
        let y32 = y16.to_dtype(lloom_tensor::DType::F32).unwrap();
        let w = Tensor::from_vec(vals(rows * k, 4.0, 0.3), (rows, k), &dev).unwrap();
        let shared = Tensor::from_vec(vals(rows * d, 5.0, 0.8), (rows, d), &dev).unwrap();
        let gate = Tensor::from_vec(vals(rows, 6.0, 2.0), (rows, 1), &dev).unwrap();
        let from16 = moe_combine(&y16, &w, &shared, &gate, k, d)
            .expect("metal f16 y")
            .unwrap();
        assert_eq!(from16.dtype(), lloom_tensor::DType::F32);
        let from32 = moe_combine(&y32, &w, &shared, &gate, k, d)
            .expect("metal f32 y")
            .unwrap();
        assert_eq!(bits32(&from16), bits32(&from32));
        assert!(bits32(&from32).iter().any(|&b| b != 0));
    }

    /// The Step-0 trap "the DeltaNet output gate is SIGMOID, not silu", pinned
    /// on the shipping kernel against a hand-computed value: at `z = 0` a
    /// sigmoid gate is 0.5 and a silu gate 0, so the output is half the normed
    /// value, not zero. `o` is one head of ones, so its RMS norm is 1 and the
    /// expected output is exactly `γ · 0.5`.
    #[cfg(feature = "metal")]
    #[test]
    fn the_gated_norm_gate_is_sigmoid() {
        let dev = Device::new_metal(0).unwrap();
        let (d, h_v) = (128usize, 2usize);
        let f32 = lloom_tensor::DType::F32;
        let o = Tensor::ones((h_v, d), f32, &dev).unwrap();
        let gamma = Tensor::full(3.0f32, d, &dev).unwrap();
        let z = Tensor::zeros(h_v * d, f32, &dev).unwrap();
        let out = o
            .apply_op3_no_bwd(&gamma, &z, &GdnGatedNorm { d, h_v, eps: 0.0 })
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(
                *v, 1.5,
                "element {i}: γ·σ(0) = 3·0.5, got {v} (silu would give 0)"
            );
        }
    }
}

// ---- Gated DeltaNet decode ---------------------------------------------------------

/// The decode prologue: `qkv [conv_ch]` against the layer's conv window
/// (`win [conv_ch, kk]`, captured, written in place) → `heads [3, h_v, d]`.
pub(crate) struct GdnPrologue {
    pub d: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub eps: f32,
    pub q_scale: f32,
    pub conv: ShortConv<Shift>,
    /// The window as it stands (read) and the window after this step
    /// (written): two buffers of the channel, never the same one.
    pub win: Tensor,
    pub win_out: Tensor,
}

impl lloom_tensor::CustomOp1 for GdnPrologue {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-prologue"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "gdn_prologue: no cpu kernel; `GatedDeltaNet::decode` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let conv_ch = (2 * self.h_k + self.h_v) * self.d;
        if l.shape().elem_count() != conv_ch || !l.is_contiguous() {
            lloom_tensor::bail!("gdn_prologue: qkv {:?} vs {conv_ch} channels", l.shape())
        }
        let kk = self.conv.k();
        let (ws, wl) = self.conv.weight().storage_and_layout();
        let (ns, nl) = self.win.storage_and_layout();
        let (os, ol) = self.win_out.storage_and_layout();
        let (ws, ns, os) = match (&*ws, &*ns, &*os) {
            (
                lloom_tensor::Storage::Metal(a),
                lloom_tensor::Storage::Metal(b),
                lloom_tensor::Storage::Metal(c),
            ) => (a, b, c),
            _ => lloom_tensor::bail!(
                "gdn_prologue: conv weight / window must be on the metal device"
            ),
        };
        if wl.shape().elem_count() != conv_ch * kk
            || nl.shape().elem_count() != conv_ch * kk
            || ol.shape().elem_count() != conv_ch * kk
            || !wl.is_contiguous()
            || !nl.is_contiguous()
            || !ol.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_prologue: conv weight / window must be contiguous [conv_ch, kk]"
            )
        }
        if self.win.id() == self.win_out.id() {
            lloom_tensor::bail!("gdn_prologue: the window in and out must be two buffers")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_prologue");
        let elem_count = 3 * self.h_v * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_heads")
            .build()?;
        lloom_tensor::kernels::call_gdn_prologue(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.d,
            self.h_k,
            self.h_v,
            kk,
            self.eps,
            self.q_scale,
            metal_bo(s, l),
            metal_bo(ws, wl),
            metal_bo(ns, nl),
            &output,
            metal_bo(os, ol),
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from((3, self.h_v, self.d)),
        ))
    }
}

/// The gated norm: `o [h_v, d]`, `γ [d]`, `z [h_v·d]` → `[h_v·d]`.
pub(crate) struct GdnGatedNorm {
    pub d: usize,
    pub h_v: usize,
    pub eps: f32,
}

impl lloom_tensor::CustomOp3 for GdnGatedNorm {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-gated-norm"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "gdn_gated_norm: no cpu kernel; `GatedDeltaNet::gated_rms_norm` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (d, h_v) = (self.d, self.h_v);
        if l1.shape().elem_count() != h_v * d
            || l2.shape().elem_count() != d
            || l3.shape().elem_count() != h_v * d
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!("gdn_gated_norm: o [h_v, d], γ [d], z [h_v·d], contiguous")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_gated_norm");
        let elem_count = h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_gated")
            .build()?;
        lloom_tensor::kernels::call_gdn_gated_norm(
            device.metal_device(),
            &encoder,
            device.kernels(),
            d,
            h_v,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from(h_v * d),
        ))
    }
}

// ---- QSA decode -------------------------------------------------------------------

/// The decode prologue: `qg [nh, 2·hd]` and `k [nkv, hd]` → `[2·nh + nkv, hd]`
/// (q normed + roped, the gate, k normed + roped); γ_q / γ_k captured.
pub(crate) struct QsaPrep {
    pub hd: usize,
    pub n_rot: usize,
    pub nh: usize,
    pub nkv: usize,
    pub pos: usize,
    pub theta: f32,
    pub eps: f32,
    pub q_gamma: Tensor,
    pub k_gamma: Tensor,
}

impl lloom_tensor::CustomOp2 for QsaPrep {
    fn name(&self) -> &'static str {
        "qwen4exp-qsa-prep"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("qsa_prep: no cpu kernel; `QsaAttention::decode` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (hd, nh, nkv) = (self.hd, self.nh, self.nkv);
        if l1.shape().elem_count() != nh * 2 * hd
            || l2.shape().elem_count() != nkv * hd
            || !l1.is_contiguous()
            || !l2.is_contiguous()
        {
            lloom_tensor::bail!(
                "qsa_prep: qg {:?} / k {:?} vs nh {nh} nkv {nkv} hd {hd}",
                l1.shape(),
                l2.shape()
            )
        }
        let (qs, ql) = self.q_gamma.storage_and_layout();
        let (ks, kl) = self.k_gamma.storage_and_layout();
        let (qs, ks) = match (&*qs, &*ks) {
            (lloom_tensor::Storage::Metal(a), lloom_tensor::Storage::Metal(b)) => (a, b),
            _ => lloom_tensor::bail!("qsa_prep: the norm weights must be on the metal device"),
        };
        if ql.shape().elem_count() != hd
            || kl.shape().elem_count() != hd
            || !ql.is_contiguous()
            || !kl.is_contiguous()
        {
            lloom_tensor::bail!("qsa_prep: γ_q / γ_k must be contiguous [hd]")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("qsa_prep");
        let elem_count = (2 * nh + nkv) * hd;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("qsa_prep")
            .build()?;
        lloom_tensor::kernels::call_qsa_prep(
            device.metal_device(),
            &encoder,
            device.kernels(),
            hd,
            self.n_rot,
            nh,
            nkv,
            self.pos,
            self.theta,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(qs, ql),
            metal_bo(ks, kl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((2 * nh + nkv, hd)),
        ))
    }
}

/// The decode prologue over `B` rows at their own positions (the pooled rows
/// arm): `qg [B, nh, 2·hd]`, `k [B, nkv, hd]`, `pos [B]` (u32) → the flat
/// `[B · (2·nh + nkv) · hd]` of three row-major regions `q [B, nh, hd] |
/// gate [B, nh, hd] | k [B, nkv, hd]`, row for row [`QsaPrep`]'s result.
pub(crate) struct QsaPrepBatch {
    pub hd: usize,
    pub n_rot: usize,
    pub nh: usize,
    pub nkv: usize,
    pub theta: f32,
    pub eps: f32,
    pub q_gamma: Tensor,
    pub k_gamma: Tensor,
}

impl lloom_tensor::CustomOp3 for QsaPrepBatch {
    fn name(&self) -> &'static str {
        "qwen4exp-qsa-prep-batch"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("qsa_prep_batch: no cpu kernel; the rows compose it per row there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (hd, nh, nkv) = (self.hd, self.nh, self.nkv);
        let rows = l3.shape().elem_count();
        if rows == 0
            || l1.shape().elem_count() != rows * nh * 2 * hd
            || l2.shape().elem_count() != rows * nkv * hd
            || s3.dtype() != lloom_tensor::DType::U32
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "qsa_prep_batch: qg {:?} / k {:?} / pos {:?} vs nh {nh} nkv {nkv} hd {hd}",
                l1.shape(),
                l2.shape(),
                l3.shape()
            )
        }
        let (qs, ql) = self.q_gamma.storage_and_layout();
        let (ks, kl) = self.k_gamma.storage_and_layout();
        let (qs, ks) = metal_pair(&qs, &ks, "qsa_prep_batch")?;
        if ql.shape().elem_count() != hd
            || kl.shape().elem_count() != hd
            || !ql.is_contiguous()
            || !kl.is_contiguous()
        {
            lloom_tensor::bail!("qsa_prep_batch: γ_q / γ_k must be contiguous [hd]")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("qsa_prep_batch");
        let elem_count = rows * (2 * nh + nkv) * hd;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("qsa_prep_batch")
            .build()?;
        lloom_tensor::kernels::call_qsa_prep_batch(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            hd,
            self.n_rot,
            nh,
            nkv,
            self.theta,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            metal_bo(qs, ql),
            metal_bo(ks, kl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from(elem_count),
        ))
    }
}

/// `sdpa_vector` for `B` rows over one layer's page pool in ONE launch:
/// `q [B, nh, hd]`, `k_pool` / `v_pool [nkv, tokens, hd]` → `[B, nh, hd]`;
/// row `r` attends its `n_keys[r]` keys through its page table (`table[r ·
/// table_stride ..]`), bit-identical to that row's own `ops::sdpa` over its
/// keys laid contiguously (`kernels::sdpa_vector_paged`).
pub(crate) struct SdpaVectorPaged {
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub page_size: usize,
    pub table_stride: usize,
    pub scale: f32,
    /// The 2-pass kernel (the per-row arm's choice,
    /// `ops::sdpa_vector_two_pass`; every row of the call takes it).
    pub two_pass: bool,
    /// `u32 [B · table_stride]`.
    pub table: Tensor,
    /// `u32 [B]` (bound as the kernel's `int` counts).
    pub n_keys: Tensor,
}

impl lloom_tensor::CustomOp3 for SdpaVectorPaged {
    fn name(&self) -> &'static str {
        "qwen4exp-sdpa-vector-paged"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("sdpa_vector_paged: no cpu kernel; the rows gather and compose there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let rows = match l1.shape().dims() {
            [b, h, d] if *h == nh && *d == hd => *b,
            _ => lloom_tensor::bail!("sdpa_vector_paged: q {:?} vs nh {nh} hd {hd}", l1.shape()),
        };
        let tokens = match l2.shape().dims() {
            [h, t, d] if *h == nkv && *d == hd => *t,
            _ => lloom_tensor::bail!(
                "sdpa_vector_paged: k pool {:?} vs nkv {nkv} hd {hd}",
                l2.shape()
            ),
        };
        if l3.shape().dims() != l2.shape().dims()
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
            || nh % nkv != 0
            || rows == 0
        {
            lloom_tensor::bail!(
                "sdpa_vector_paged: q {:?} / k {:?} / v {:?} (contiguous, nh {nh} a multiple of nkv {nkv})",
                l1.shape(),
                l2.shape(),
                l3.shape()
            )
        }
        let (ts, tl) = self.table.storage_and_layout();
        let (ns, nl) = self.n_keys.storage_and_layout();
        let (ts, ns) = metal_pair(&ts, &ns, "sdpa_vector_paged")?;
        if ts.dtype() != lloom_tensor::DType::U32
            || ns.dtype() != lloom_tensor::DType::U32
            || tl.shape().elem_count() != rows * self.table_stride
            || nl.shape().elem_count() != rows
            || !tl.is_contiguous()
            || !nl.is_contiguous()
        {
            lloom_tensor::bail!(
                "sdpa_vector_paged: table {:?} (u32 [{rows} × {}]) / n_keys {:?} (u32 [{rows}])",
                tl.shape(),
                self.table_stride,
                nl.shape()
            )
        }
        // The host-side bound: every page a live key reaches lies in the pool.
        let itype = match s1.dtype() {
            lloom_tensor::DType::F32 => lloom_tensor::kernels::SdpaDType::F32,
            lloom_tensor::DType::F16 => lloom_tensor::kernels::SdpaDType::F16,
            other => lloom_tensor::bail!("sdpa_vector_paged: {other:?} (f32 | f16)"),
        };
        if s2.dtype() != s1.dtype() || s3.dtype() != s1.dtype() {
            lloom_tensor::bail!("sdpa_vector_paged: q / k / v dtypes differ")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("sdpa_vector_paged");
        let elem_count = rows * nh * hd;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("sdpa_vector_paged")
            .build()?;
        let esz = s1.dtype().size_in_bytes();
        let scratch = if self.two_pass {
            let blocks = lloom_tensor::kernels::SDPA_2PASS_BLOCKS;
            let f32 = lloom_tensor::DType::F32;
            let partials = device
                .new_buffer_builder()
                .with_size_for(rows * nh * blocks * hd, f32)
                .with_label("sdpa_vector_paged_partials")
                .build()?;
            let sums = device
                .new_buffer_builder()
                .with_size_for(rows * nh * blocks, f32)
                .with_label("sdpa_vector_paged_sums")
                .build()?;
            let maxs = device
                .new_buffer_builder()
                .with_size_for(rows * nh * blocks, f32)
                .with_label("sdpa_vector_paged_maxs")
                .build()?;
            Some((partials, sums, maxs))
        } else {
            None
        };
        lloom_tensor::kernels::call_sdpa_vector_paged(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            nh,
            nh / nkv,
            hd,
            l1.start_offset() * esz,
            s1.buffer(),
            l2.start_offset() * esz,
            s2.buffer(),
            tokens * hd,
            l3.start_offset() * esz,
            s3.buffer(),
            tokens * hd,
            ts.buffer(),
            tl.start_offset() * 4,
            self.table_stride,
            ns.buffer(),
            nl.start_offset() * 4,
            self.page_size,
            &output,
            self.scale,
            itype,
            scratch.as_ref().map(|(p, s, m)| (&***p, &***s, &***m)),
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((rows, nh, hd)),
        ))
    }
}

/// `sigmoid(a) · b`, elementwise.
pub(crate) struct SigmoidMul;

impl lloom_tensor::CustomOp2 for SigmoidMul {
    fn name(&self) -> &'static str {
        "qwen4exp-sigmoid-mul"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("sigmoid_mul: no cpu kernel; `b * sigmoid(a)` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        if l1.shape().elem_count() != l2.shape().elem_count()
            || !l1.is_contiguous()
            || !l2.is_contiguous()
        {
            lloom_tensor::bail!("sigmoid_mul: same-size contiguous inputs only")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("sigmoid_mul");
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("sigmoid_mul")
            .build()?;
        lloom_tensor::kernels::call_sigmoid_mul(
            device.metal_device(),
            &encoder,
            device.kernels(),
            elem_count,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            l2.shape().clone(),
        ))
    }
}

pub(crate) fn sigmoid_mul(a: &Tensor, b: &Tensor) -> Option<Result<Tensor>> {
    (fused_applies(a) && fused_applies(b) && a.elem_count() == b.elem_count())
        .then(|| a.apply_op2_no_bwd(b, &SigmoidMul))
}

// ---- DeltaNet prefill (Kernel B) ---------------------------------------------------

#[cfg(feature = "metal")]
fn metal_pair<'a>(
    a: &'a lloom_tensor::Storage,
    b: &'a lloom_tensor::Storage,
    what: &str,
) -> Result<(
    &'a lloom_tensor::MetalStorage,
    &'a lloom_tensor::MetalStorage,
)> {
    match (a, b) {
        (lloom_tensor::Storage::Metal(a), lloom_tensor::Storage::Metal(b)) => Ok((a, b)),
        _ => lloom_tensor::bail!("{what}: the captured tensors must be on the metal device"),
    }
}

/// `alpha`, `beta_raw` `[seq, h_v]` → `[2, h_v, nc, cs]` (g, β), zero past `seq`.
pub(crate) struct GdnGate {
    pub seq: usize,
    pub h_v: usize,
    pub nc: usize,
    pub cs: usize,
    pub dt_bias: Tensor,
    pub ssm_a: Tensor,
}

impl lloom_tensor::CustomOp2 for GdnGate {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-gate"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_gate: no cpu kernel; `gate_and_beta` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let n = self.seq * self.h_v;
        if l1.shape().elem_count() != n
            || l2.shape().elem_count() != n
            || !l1.is_contiguous()
            || !l2.is_contiguous()
        {
            lloom_tensor::bail!("gdn_gate: alpha / beta must be contiguous [seq, h_v]")
        }
        let (ds, dl) = self.dt_bias.storage_and_layout();
        let (as_, al) = self.ssm_a.storage_and_layout();
        let (ds, as_) = metal_pair(&ds, &as_, "gdn_gate")?;
        if dl.shape().elem_count() != self.h_v || al.shape().elem_count() != self.h_v {
            lloom_tensor::bail!("gdn_gate: dt_bias / ssm_a must be [h_v]")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_gate");
        let elem_count = 2 * self.h_v * self.nc * self.cs;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_gate")
            .build()?;
        lloom_tensor::kernels::call_gdn_gate(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.seq,
            self.h_v,
            self.nc,
            self.cs,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(ds, dl),
            metal_bo(as_, al),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((2, self.h_v, self.nc, self.cs)),
        ))
    }
}

/// `qkv [seq, C]`, the window `[C, kk]`, the taps `[C, kk]` → `[3, h_v, nc, cs, d]`.
pub(crate) struct GdnPrefillPrologue {
    pub seq: usize,
    pub d: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub kk: usize,
    pub cs: usize,
    pub nc: usize,
    pub eps: f32,
    pub q_scale: f32,
}

impl lloom_tensor::CustomOp3 for GdnPrefillPrologue {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-prefill-prologue"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "gdn_prefill_prologue: no cpu kernel; `GatedDeltaNet::prefill` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let channels = (2 * self.h_k + self.h_v) * self.d;
        if l1.shape().elem_count() != self.seq * channels
            || l2.shape().elem_count() != channels * self.kk
            || l3.shape().elem_count() != channels * self.kk
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_prefill_prologue: qkv {:?} / window {:?} / taps {:?} vs seq {} × {channels} channels, kk {}",
                l1.shape(),
                l2.shape(),
                l3.shape(),
                self.seq,
                self.kk
            )
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_prefill_prologue");
        let elem_count = 3 * self.h_v * self.nc * self.cs * self.d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_prefill_qkv")
            .build()?;
        lloom_tensor::kernels::call_gdn_prefill_prologue(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.seq,
            self.d,
            self.h_k,
            self.h_v,
            self.kk,
            self.cs,
            self.nc,
            self.eps,
            self.q_scale,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((3, self.h_v, self.nc, self.cs, self.d)),
        ))
    }
}

/// The window `[C, kk]` (applied to) and the chunk's `qkv [seq, C]` (captured)
/// → the window after the chunk, a new `[C, kk]`.
pub(crate) struct GdnConvWindow {
    pub seq: usize,
    pub channels: usize,
    pub kk: usize,
    pub qkv: Tensor,
}

impl lloom_tensor::CustomOp1 for GdnConvWindow {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-conv-window"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_conv_window: no cpu kernel; `conv_prefill` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let n = self.channels * self.kk;
        if l.shape().elem_count() != n || !l.is_contiguous() {
            lloom_tensor::bail!("gdn_conv_window: the window must be contiguous [C, kk]")
        }
        let (qs, ql) = self.qkv.storage_and_layout();
        let qs = match &*qs {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("gdn_conv_window: qkv must be on the metal device"),
        };
        if ql.shape().elem_count() != self.seq * self.channels || !ql.is_contiguous() {
            lloom_tensor::bail!("gdn_conv_window: qkv must be contiguous [seq, C]")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_conv_window");
        let output = device
            .new_buffer_builder()
            .with_size_for(n, lloom_tensor::DType::F32)
            .with_label("gdn_conv_window")
            .build()?;
        lloom_tensor::kernels::call_gdn_conv_window(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.seq,
            self.channels,
            self.kk,
            metal_bo(s, l),
            metal_bo(qs, ql),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), n, s.dtype()),
            l.shape().clone(),
        ))
    }
}

/// `g [h_v, nc, cs]` → the flat decay pack: `g_exp [h_v·nc·cs] | g_diff
/// [h_v·nc·cs] | g_last [h_v·nc] | decay [h_v·nc·cs·cs] | decay_strict [same]`.
pub(crate) struct GdnChunkDecay {
    pub h_v: usize,
    pub nc: usize,
    pub cs: usize,
}

impl GdnChunkDecay {
    /// Offsets into the flat pack: `(g_exp, g_diff, g_last, decay, decay_strict, len)`.
    pub fn offsets(&self) -> (usize, usize, usize, usize, usize, usize) {
        let n = self.h_v * self.nc;
        let (cs, sq) = (self.cs, self.cs * self.cs);
        (
            0,
            n * cs,
            2 * n * cs,
            2 * n * cs + n,
            2 * n * cs + n + n * sq,
            2 * n * cs + n + 2 * n * sq,
        )
    }
}

impl lloom_tensor::CustomOp1 for GdnChunkDecay {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-chunk-decay"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_chunk_decay: no cpu kernel; `chunk_forward` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        if l.shape().elem_count() != self.h_v * self.nc * self.cs || !l.is_contiguous() {
            lloom_tensor::bail!("gdn_chunk_decay: g must be contiguous [h_v, nc, cs]")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_chunk_decay");
        let elem_count = self.offsets().5;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_chunk_decay")
            .build()?;
        lloom_tensor::kernels::call_gdn_chunk_decay(
            device.metal_device(),
            &encoder,
            device.kernels(),
            self.h_v,
            self.nc,
            self.cs,
            metal_bo(s, l),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from(elem_count),
        ))
    }
}

/// The chunked q/k/v `[3, n, d]`, the gate pack `[2, n]`, the decay pack →
/// flat `[5, n, d]`: k_beta | w_in | q_g | kg | v_beta.
pub(crate) struct GdnChunkScale {
    pub n: usize,
    pub d: usize,
}

impl lloom_tensor::CustomOp3 for GdnChunkScale {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-chunk-scale"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_chunk_scale: no cpu kernel; `chunk_forward` composes it there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (n, d) = (self.n, self.d);
        if l1.shape().elem_count() != 3 * n * d
            || l2.shape().elem_count() != 2 * n
            || l3.shape().elem_count() < 2 * n
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!("gdn_chunk_scale: operands vs n {n} d {d}")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_chunk_scale");
        let elem_count = 5 * n * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_chunk_scale")
            .build()?;
        lloom_tensor::kernels::call_gdn_chunk_scale(
            device.metal_device(),
            &encoder,
            device.kernels(),
            n,
            d,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((5, n, d)),
        ))
    }
}

/// `o [h_v, padded, d]`, `z [seq, h_v·d]`, γ captured → `[seq, h_v·d]`.
pub(crate) struct GdnChunkEpilogue {
    pub seq: usize,
    pub padded: usize,
    pub d: usize,
    pub h_v: usize,
    pub eps: f32,
    pub gamma: Tensor,
}

impl lloom_tensor::CustomOp2 for GdnChunkEpilogue {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-chunk-epilogue"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "gdn_chunk_epilogue: no cpu kernel; `GatedDeltaNet::prefill` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (seq, padded, d, h_v) = (self.seq, self.padded, self.d, self.h_v);
        if l1.shape().elem_count() != h_v * padded * d
            || l2.shape().elem_count() != seq * h_v * d
            || seq > padded
            || !l1.is_contiguous()
            || !l2.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_chunk_epilogue: o {:?} / z {:?} vs seq {seq} padded {padded} h_v {h_v} d {d}",
                l1.shape(),
                l2.shape()
            )
        }
        let (gs, gl) = self.gamma.storage_and_layout();
        let gs = match &*gs {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("gdn_chunk_epilogue: γ must be on the metal device"),
        };
        if gl.shape().elem_count() != d || !gl.is_contiguous() {
            lloom_tensor::bail!("gdn_chunk_epilogue: γ must be contiguous [d]")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_chunk_epilogue");
        let elem_count = seq * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_chunk_epilogue")
            .build()?;
        lloom_tensor::kernels::call_gdn_chunk_epilogue(
            device.metal_device(),
            &encoder,
            device.kernels(),
            seq,
            padded,
            d,
            h_v,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(gs, gl),
            metal_bo(s2, l2),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((seq, h_v * d)),
        ))
    }
}

// ---- QSA prefill ----------------------------------------------------------------

/// `qg [seq, nh, 2·hd]`, `k [seq, nkv, hd]`, `v [seq, nkv, hd]` → the flat
/// pack: `q_h [nh, seq, hd] | gate [seq, nh·hd] | k_new [nkv, seq, hd] | v_new
/// [nkv, seq, hd]`; γ_q / γ_k captured.
pub(crate) struct QsaPrepRows {
    pub hd: usize,
    pub n_rot: usize,
    pub nh: usize,
    pub nkv: usize,
    pub seq: usize,
    pub pos0: usize,
    pub theta: f32,
    pub eps: f32,
    pub q_gamma: Tensor,
    pub k_gamma: Tensor,
}

impl lloom_tensor::CustomOp3 for QsaPrepRows {
    fn name(&self) -> &'static str {
        "qwen4exp-qsa-prep-rows"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!(
            "qsa_prep_rows: no cpu kernel; `QsaAttention::prefill` composes it there"
        )
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (hd, nh, nkv, seq) = (self.hd, self.nh, self.nkv, self.seq);
        if l1.shape().elem_count() != seq * nh * 2 * hd
            || l2.shape().elem_count() != seq * nkv * hd
            || l3.shape().elem_count() != seq * nkv * hd
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "qsa_prep_rows: qg {:?} / k {:?} / v {:?} vs seq {seq} nh {nh} nkv {nkv} hd {hd}",
                l1.shape(),
                l2.shape(),
                l3.shape()
            )
        }
        let (qs, ql) = self.q_gamma.storage_and_layout();
        let (ks, kl) = self.k_gamma.storage_and_layout();
        let (qs, ks) = metal_pair(&qs, &ks, "qsa_prep_rows")?;
        if ql.shape().elem_count() != hd
            || kl.shape().elem_count() != hd
            || !ql.is_contiguous()
            || !kl.is_contiguous()
        {
            lloom_tensor::bail!("qsa_prep_rows: γ_q / γ_k must be contiguous [hd]")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("qsa_prep_rows");
        let elem_count = (2 * nh + 2 * nkv) * seq * hd;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("qsa_prep_rows")
            .build()?;
        lloom_tensor::kernels::call_qsa_prep_rows(
            device.metal_device(),
            &encoder,
            device.kernels(),
            hd,
            self.n_rot,
            nh,
            nkv,
            seq,
            self.pos0,
            self.theta,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            metal_bo(qs, ql),
            metal_bo(ks, kl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from(elem_count),
        ))
    }
}

// ---- DeltaNet decode over the rows of one state pool, and over the positions of one row ----

#[cfg(feature = "metal")]
fn metal_of(t: &Tensor, what: &str) -> Result<(lloom_tensor::MetalStorage, Layout)> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        lloom_tensor::Storage::Metal(m) => Ok((m.clone(), l.clone())),
        _ => lloom_tensor::bail!("{what}: every operand on the metal device"),
    }
}

/// The batched prologue: `qkv [B, C]` (applied to), the row tables `slots`
/// / `phases` (`[B]` u32) and the window pool (captured) → `heads [B, 3,
/// h_v, d]`; each row's window after the step at its next phase.
pub(crate) struct GdnPrologueRows {
    pub rows: usize,
    pub d: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub eps: f32,
    pub q_scale: f32,
    pub conv: ShortConv<Shift>,
    pub win: PoolLayer,
    pub slots: Tensor,
    pub phases: Tensor,
}

impl lloom_tensor::CustomOp1 for GdnPrologueRows {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-prologue-rows"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_prologue_rows: no cpu kernel; the per-row path composes it")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (rows, d, h_k, h_v, kk) = (self.rows, self.d, self.h_k, self.h_v, self.conv.k());
        let conv_ch = (2 * h_k + h_v) * d;
        if l.shape().elem_count() != rows * conv_ch || !l.is_contiguous() {
            lloom_tensor::bail!(
                "gdn_prologue_rows: qkv {:?} vs [{rows}, {conv_ch}]",
                l.shape()
            )
        }
        let (ws, wl) = metal_of(self.conv.weight(), "gdn_prologue_rows")?;
        let (ps, pl) = metal_of(&self.win.tensor, "gdn_prologue_rows")?;
        let (ss, sl) = metal_of(&self.slots, "gdn_prologue_rows")?;
        let (hs, hl) = metal_of(&self.phases, "gdn_prologue_rows")?;
        if wl.shape().elem_count() != conv_ch * kk
            || pl.shape().elem_count() != self.win.n_phases * self.win.n_slots * conv_ch * kk
            || sl.shape().elem_count() != rows
            || hl.shape().elem_count() != rows
            || !wl.is_contiguous()
            || !pl.is_contiguous()
            || !sl.is_contiguous()
            || !hl.is_contiguous()
        {
            lloom_tensor::bail!("gdn_prologue_rows: the taps / pool / tables' shapes")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_prologue_rows");
        let elem_count = rows * 3 * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_heads_rows")
            .build()?;
        lloom_tensor::kernels::call_gdn_prologue_rows(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            d,
            h_k,
            h_v,
            kk,
            self.eps,
            self.q_scale,
            self.win.n_slots,
            self.win.n_phases,
            metal_bo(s, l),
            metal_bo(&ws, &wl),
            metal_bo(&ps, &pl),
            metal_bo(&ss, &sl),
            metal_bo(&hs, &hl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from((rows, 3, h_v, d)),
        ))
    }
}

/// The multi-position prologue: `qkv [T, C]` (applied to) against row
/// `slot` at `phase` of the window pool (captured) → `heads [T, 3, h_v,
/// d]`; the window after position t at phase `phase + 1 + t`.
pub(crate) struct GdnPrologueSeq {
    pub steps: usize,
    pub d: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub eps: f32,
    pub q_scale: f32,
    pub conv: ShortConv<Shift>,
    pub win: PoolLayer,
    pub slot: usize,
    pub phase: usize,
}

impl lloom_tensor::CustomOp1 for GdnPrologueSeq {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-prologue-seq"
    }
    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_prologue_seq: no cpu kernel; the per-position path composes it")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (steps, d, h_k, h_v, kk) = (self.steps, self.d, self.h_k, self.h_v, self.conv.k());
        let conv_ch = (2 * h_k + h_v) * d;
        if l.shape().elem_count() != steps * conv_ch || !l.is_contiguous() {
            lloom_tensor::bail!(
                "gdn_prologue_seq: qkv {:?} vs [{steps}, {conv_ch}]",
                l.shape()
            )
        }
        if steps + 1 > self.win.n_phases {
            lloom_tensor::bail!(
                "gdn_prologue_seq: {steps} positions over {} phases",
                self.win.n_phases
            )
        }
        let (ws, wl) = metal_of(self.conv.weight(), "gdn_prologue_seq")?;
        let (ps, pl) = metal_of(&self.win.tensor, "gdn_prologue_seq")?;
        if wl.shape().elem_count() != conv_ch * kk
            || pl.shape().elem_count() != self.win.n_phases * self.win.n_slots * conv_ch * kk
            || !wl.is_contiguous()
            || !pl.is_contiguous()
        {
            lloom_tensor::bail!("gdn_prologue_seq: the taps / pool shapes")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_prologue_seq");
        let elem_count = steps * 3 * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_heads_seq")
            .build()?;
        lloom_tensor::kernels::call_gdn_prologue_seq(
            device.metal_device(),
            &encoder,
            device.kernels(),
            steps,
            d,
            h_k,
            h_v,
            kk,
            self.eps,
            self.q_scale,
            self.win.n_slots,
            self.win.n_phases,
            self.slot,
            self.phase,
            metal_bo(s, l),
            metal_bo(&ws, &wl),
            metal_bo(&ps, &pl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from((steps, 3, h_v, d)),
        ))
    }
}

/// The batched gated norm: `o [B, h_v, d]`, γ, `z [B, h_v·d]` → `[B, h_v·d]`.
pub(crate) struct GdnGatedNormRows {
    pub rows: usize,
    pub d: usize,
    pub h_v: usize,
    pub eps: f32,
}

impl lloom_tensor::CustomOp3 for GdnGatedNormRows {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-gated-norm-rows"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_gated_norm_rows: no cpu kernel")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (rows, d, h_v) = (self.rows, self.d, self.h_v);
        if l1.shape().elem_count() != rows * h_v * d
            || l2.shape().elem_count() != d
            || l3.shape().elem_count() != rows * h_v * d
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!("gdn_gated_norm_rows: o / γ / z vs rows {rows} d {d} h_v {h_v}")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_gated_norm_rows");
        let elem_count = rows * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_gated_rows")
            .build()?;
        lloom_tensor::kernels::call_gdn_gated_norm_rows(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            d,
            h_v,
            self.eps,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((rows, h_v * d)),
        ))
    }
}

/// The DeltaNet prefill's inter-chunk recurrence as one launch
/// (`gdn_chunk_scan_f32`): inputs `w`, `u`, `q_g` `[h_v, nc, cs, d]` (the op's
/// three), `kq [h_v, nc, cs, cs]`, `kg [h_v, nc, cs, d]`, `g_last [h_v, nc]`,
/// `s_in [h_v, d, d]` (fields); output flat `[o (h_v·nc·cs·d) | s_out
/// (h_v·d·d)]`. Metal only; the composed loop is the CPU's and the A/B arm
/// (`QWEN4EXP_CHUNK_SCAN=0`).
pub struct GdnChunkScan {
    pub h_v: usize,
    pub nc: usize,
    pub cs: usize,
    pub d: usize,
    pub kq: Tensor,
    pub kg: Tensor,
    pub g_last: Tensor,
    pub s_in: Tensor,
}

/// `QWEN4EXP_CHUNK_SCAN=0` (read once) keeps the host loop; `set_chunk_scan`
/// for the tests.
pub fn chunk_scan_on() -> bool {
    use std::sync::atomic::Ordering;
    match CHUNK_SCAN.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = crate::tuning::tuning().qwen4exp.chunk_scan;
            CHUNK_SCAN.store(on as u8, Ordering::Relaxed);
            on
        }
    }
}

/// Threadgroups a head's scan is split over (column slices of `S`): 48
/// heads alone leave most of a 32-core GPU idle; 4 slices of 32 columns.
pub const CHUNK_SCAN_SPLITS: usize = 4;

pub fn set_chunk_scan(on: bool) {
    CHUNK_SCAN.store(if on { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
}

static CHUNK_SCAN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

impl lloom_tensor::CustomOp3 for GdnChunkScan {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-chunk-scan"
    }
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        lloom_tensor::bail!("gdn_chunk_scan: no cpu kernel; `prefill_fused` loops the chunks there")
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &lloom_tensor::MetalStorage,
        l1: &Layout,
        s2: &lloom_tensor::MetalStorage,
        l2: &Layout,
        s3: &lloom_tensor::MetalStorage,
        l3: &Layout,
    ) -> Result<(lloom_tensor::MetalStorage, Shape)> {
        use lloom_tensor::backend::BackendStorage;
        let (h_v, nc, cs, d) = (self.h_v, self.nc, self.cs, self.d);
        let n = h_v * nc * cs * d;
        for (l, want, name) in [(l1, n, "w"), (l2, n, "u"), (l3, n, "q_g")] {
            if l.shape().elem_count() != want || !l.is_contiguous() {
                lloom_tensor::bail!("gdn_chunk_scan: {name} {:?} vs {want}", l.shape())
            }
        }
        let (kqs, kql) = self.kq.storage_and_layout();
        let (kgs, kgl) = self.kg.storage_and_layout();
        let (gls, gll) = self.g_last.storage_and_layout();
        let (sis, sil) = self.s_in.storage_and_layout();
        let (kqm, kgm, glm, sim) = match (&*kqs, &*kgs, &*gls, &*sis) {
            (
                lloom_tensor::Storage::Metal(a),
                lloom_tensor::Storage::Metal(b),
                lloom_tensor::Storage::Metal(c),
                lloom_tensor::Storage::Metal(e),
            ) => (a, b, c, e),
            _ => lloom_tensor::bail!("gdn_chunk_scan: every operand must be on the metal device"),
        };
        if kql.shape().elem_count() != h_v * nc * cs * cs
            || kgl.shape().elem_count() != n
            || gll.shape().elem_count() != h_v * nc
            || sil.shape().elem_count() != h_v * d * d
            || !kql.is_contiguous()
            || !kgl.is_contiguous()
            || !gll.is_contiguous()
            || !sil.is_contiguous()
        {
            lloom_tensor::bail!("gdn_chunk_scan: field operands vs h_v {h_v} nc {nc} cs {cs} d {d}")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_chunk_scan");
        let out_count = n + h_v * d * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(out_count, lloom_tensor::DType::F32)
            .with_label("gdn_chunk_scan")
            .build()?;
        let vn = device
            .new_buffer_builder()
            .with_size_for(h_v * cs * d, lloom_tensor::DType::F32)
            .with_label("gdn_chunk_scan_vn")
            .build()?;
        // `s_out` is the tail of the output: one buffer, two regions.
        let s_out_bo = lloom_tensor::kernels::BufferOffset {
            buffer: &output,
            offset_in_bytes: n * 4,
        };
        lloom_tensor::kernels::call_gdn_chunk_scan_at(
            device.metal_device(),
            &encoder,
            device.kernels(),
            (h_v, nc, cs, d),
            CHUNK_SCAN_SPLITS,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            metal_bo(kqm, &kql),
            metal_bo(kgm, &kgl),
            metal_bo(glm, &gll),
            metal_bo(sim, &sil),
            &output,
            s_out_bo,
            &vn,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), out_count, s1.dtype()),
            Shape::from(out_count),
        ))
    }
}
