//! The fused ops of the Qwen4-exp blocks: one Metal launch each
//! (`lloom-tensor/src/kernels/metal_src/qwen4exp.metal`), with the composed
//! tensor-op form as the CPU path AND the reference every kernel is tested
//! against (`deltanet.rs`'s tests, `qwen4exp_dispatch_census.rs`).
//!
//! Why: the decode step was ~5.7k launches a token and the chunk prefill
//! ~2.6k `copy2d` per prompt token, 89 % of the step GPU-busy on launches of
//! `[1, 1, 2560]` elementwise kernels (`design/.dev/serving-catalogue.md`
//! §0). Each op here takes one block's chain of launches down to one.

use lloom_tensor::{CpuStorage, Layout, Result, Shape, Tensor};

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
    *OFF.get_or_init(|| std::env::var_os("QWEN4EXP_FUSED_OFF").is_some())
}

/// A fused op serves a carry-shaped f32 tensor on Metal when it is
/// contiguous; the composed path takes everything else.
fn fused_applies(t: &Tensor) -> bool {
    !fused_off()
        && t.device().is_metal()
        && t.dtype() == lloom_tensor::DType::F32
        && t.is_contiguous()
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
        let ids = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::U32)
            .with_label("moe_topk_ids")
            .build()?;
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
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("silu_mul");
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("silu_mul")
            .build()?;
        lloom_tensor::kernels::call_silu_mul(
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
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            l3.shape().clone(),
        ))
    }
}

/// The fused router selection: `(ids [.., k] u32, w [.., k] f32)`, or `None`
/// where the composed path applies.
pub(crate) fn moe_topk(logits: &Tensor, k: usize) -> Option<Result<(Tensor, Tensor)>> {
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
        let w = Tensor::zeros(dims.as_slice(), lloom_tensor::DType::F32, logits.device())?;
        let ids = logits.apply_op1_no_bwd(&MoeTopk { k, w: w.clone() })?;
        Ok((ids, w))
    })())
}

pub(crate) fn silu_mul(g: &Tensor, u: &Tensor) -> Option<Result<Tensor>> {
    (fused_applies(g) && fused_applies(u) && g.shape() == u.shape())
        .then(|| g.apply_op2_no_bwd(u, &SiluMul))
}

pub(crate) fn moe_combine(
    y: &Tensor,
    w: &Tensor,
    shared: &Tensor,
    gate: &Tensor,
    k: usize,
    d: usize,
) -> Option<Result<Tensor>> {
    (fused_applies(y) && fused_applies(w) && fused_applies(shared) && fused_applies(gate)).then(
        || {
            y.apply_op3_no_bwd(
                w,
                shared,
                &MoeCombine {
                    k,
                    d,
                    gate: gate.clone(),
                },
            )
        },
    )
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
}

// ---- Gated DeltaNet decode ---------------------------------------------------------

/// The decode prologue: `qkv [conv_ch]` against the layer's conv window
/// (`win [conv_ch, kk]`, captured, written in place) → `heads [3, h_v, d]`.
pub(crate) struct GdnPrologue {
    pub d: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub kk: usize,
    pub eps: f32,
    pub q_scale: f32,
    pub conv_w: Tensor,
    pub win: Tensor,
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
        let (ws, wl) = self.conv_w.storage_and_layout();
        let (ns, nl) = self.win.storage_and_layout();
        let (ws, ns) = match (&*ws, &*ns) {
            (lloom_tensor::Storage::Metal(a), lloom_tensor::Storage::Metal(b)) => (a, b),
            _ => lloom_tensor::bail!(
                "gdn_prologue: conv weight / window must be on the metal device"
            ),
        };
        if wl.shape().elem_count() != conv_ch * self.kk
            || nl.shape().elem_count() != conv_ch * self.kk
            || !wl.is_contiguous()
            || !nl.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_prologue: conv weight / window must be contiguous [conv_ch, kk]"
            )
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
            self.kk,
            self.eps,
            self.q_scale,
            metal_bo(s, l),
            metal_bo(ws, wl),
            metal_bo(ns, nl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s.dtype()),
            Shape::from((3, self.h_v, self.d)),
        ))
    }
}

/// The recurrence: `heads`, the raw `alpha`/`beta` logits, the captured
/// `dt_bias`/`ssm_a` and the `S` slab (in place) → `o [h_v, d]`.
pub(crate) struct GdnStep {
    pub d: usize,
    pub h_v: usize,
    pub dt_bias: Tensor,
    pub ssm_a: Tensor,
    pub s: Tensor,
}

impl lloom_tensor::CustomOp3 for GdnStep {
    fn name(&self) -> &'static str {
        "qwen4exp-gdn-step"
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
        lloom_tensor::bail!("gdn_step: no cpu kernel; `deltanet::decode_step` composes it there")
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
        if l1.shape().elem_count() != 3 * h_v * d
            || l2.shape().elem_count() != h_v
            || l3.shape().elem_count() != h_v
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_step: heads {:?} / alpha {:?} / beta {:?} vs d {d} h_v {h_v}",
                l1.shape(),
                l2.shape(),
                l3.shape()
            )
        }
        let (bs, bl) = self.dt_bias.storage_and_layout();
        let (as_, al) = self.ssm_a.storage_and_layout();
        let (ss, sl) = self.s.storage_and_layout();
        let (bs, as_, ss) = match (&*bs, &*as_, &*ss) {
            (
                lloom_tensor::Storage::Metal(a),
                lloom_tensor::Storage::Metal(b),
                lloom_tensor::Storage::Metal(c),
            ) => (a, b, c),
            _ => lloom_tensor::bail!("gdn_step: dt_bias / ssm_a / S must be on the metal device"),
        };
        if bl.shape().elem_count() != h_v
            || al.shape().elem_count() != h_v
            || sl.shape().elem_count() != d * d * h_v
            || !bl.is_contiguous()
            || !al.is_contiguous()
            || !sl.is_contiguous()
        {
            lloom_tensor::bail!("gdn_step: dt_bias / ssm_a [h_v], S [d, d, h_v], contiguous")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_step");
        let elem_count = h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lloom_tensor::DType::F32)
            .with_label("gdn_o")
            .build()?;
        lloom_tensor::kernels::call_gdn_step(
            device.metal_device(),
            &encoder,
            device.kernels(),
            d,
            h_v,
            metal_bo(s1, l1),
            metal_bo(s2, l2),
            metal_bo(s3, l3),
            metal_bo(bs, bl),
            metal_bo(as_, al),
            metal_bo(ss, sl),
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), elem_count, s1.dtype()),
            Shape::from((h_v, d)),
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
