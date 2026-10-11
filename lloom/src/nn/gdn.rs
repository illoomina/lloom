//! The Gated DeltaNet step block: the recurrence on a layer's `S [d, d, h_v]`
//! slab (head-last),
//!
//! ```text
//! g = ssm_a · softplus(α + dt_bias);  β = sigmoid(β_raw)
//! S ← e^g · S;  r = kᵀ S;  δ = β (v − r);  S ← S + k ⊗ δ;  o = qᵀ S
//! ```
//!
//! over one row, the rows of a [`StatePool`](super::StatePool), or the
//! positions of one row (a speculative window, every intermediate `S` kept by
//! phase). A DeltaNet mixer owns one per layer; the prologue before it is the
//! short-conv block's ([`super::short_conv`]).
//!
//! The step's forms are types ([`StepForm`], RESTRUCTURING §1.5): a form is a
//! variant of the block, not a branch in it. The arm is chosen once, at
//! construction ([`GdnStepArm`]), and every call through it is monomorphized.
//! The kernels are `metal_src/gdn/gdn.metal`'s.

use crate::nn::recurrent::PoolLayer;
use lloom_tensor::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::LoopForm {}
    impl Sealed for super::RegisterForm {}
}

/// The step's forms, as a value: what a configuration names and a
/// [`GdnStepArm`] is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdnStepForm {
    /// [`LoopForm`].
    Loop,
    /// [`RegisterForm`].
    Register,
}

impl GdnStepForm {
    /// The form `tuning.gdn.step_reg` names.
    pub fn configured() -> Self {
        if lloom_tensor::tuning::tuning().gdn.step_reg {
            GdnStepForm::Register
        } else {
            GdnStepForm::Loop
        }
    }
}

/// The gated RMS norm's output gate, `rms_norm(o) · γ · gate(z)`: `sigmoid`
/// (Qwen4-exp, `output_gate_type: sigmoid`) or `silu` (Qwen3.5, `swish`).
/// A variant of the block, chosen at construction from the checkpoint's
/// config; the Metal kernels are instantiated per gate, the composed path
/// switches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OutGate {
    #[default]
    Sigmoid,
    Silu,
}

impl OutGate {
    pub fn is_silu(self) -> bool {
        matches!(self, OutGate::Silu)
    }
    /// `gate(z)` as tensor ops (the composed path).
    pub fn apply(self, z: &Tensor) -> Result<Tensor> {
        match self {
            OutGate::Sigmoid => crate::nn::ops::sigmoid(z),
            OutGate::Silu => crate::nn::ops::silu(z),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            OutGate::Sigmoid => "sigmoid",
            OutGate::Silu => "silu",
        }
    }
    /// The gate `output_gate_type` names (`swish` = `silu`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sigmoid" => Some(OutGate::Sigmoid),
            "silu" | "swish" => Some(OutGate::Silu),
            _ => None,
        }
    }
}

/// A form of the step.
pub trait StepForm:
    sealed::Sealed + Copy + Default + std::fmt::Debug + Send + Sync + 'static
{
    /// This form as a value.
    const FORM: GdnStepForm;
}

/// Reads `S` twice: the read-out's pass, then the update's.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopForm;

/// Holds each thread's slice of `S` in registers between the read-out and the
/// update, so a row's slab is read once. Not the loop form bit for bit: its
/// rounding is the compiler's for its shape (`docs/architecture.md` §2.3.5a).
/// One row, the rows and the positions of a window run one column, so a
/// sequence computes the same bits alone, batched or verified.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegisterForm;

impl StepForm for LoopForm {
    const FORM: GdnStepForm = GdnStepForm::Loop;
}

impl StepForm for RegisterForm {
    const FORM: GdnStepForm = GdnStepForm::Register;
}

/// One layer's step in form `F`: the geometry and the gate constants it
/// closes over (`dt_bias`, `ssm_a`, `[h_v]`, read as f32).
#[derive(Debug, Clone)]
pub struct GdnStep<F: StepForm> {
    d: usize,
    h_v: usize,
    dt_bias: Tensor,
    ssm_a: Tensor,
    _form: PhantomData<F>,
}

impl<F: StepForm> GdnStep<F> {
    /// `d` the head size (`d_k == d_v`), `h_v` the value heads. The shapes
    /// are checked where the kernel is dispatched.
    pub fn new(d: usize, h_v: usize, dt_bias: &Tensor, ssm_a: &Tensor) -> Self {
        Self {
            d,
            h_v,
            dt_bias: dt_bias.clone(),
            ssm_a: ssm_a.clone(),
            _form: PhantomData,
        }
    }

    fn gates(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.dt_bias.to_dtype(DType::F32)?,
            self.ssm_a.to_dtype(DType::F32)?,
        ))
    }

    /// One row: `heads [3, h_v, d]` (q, k, v), the raw gate logits `alpha`,
    /// `beta [h_v]`, the slab `s` (read) and `s_out` (the slab after the step,
    /// written; never `s`) → `o [h_v, d]`.
    pub fn step(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: &Tensor,
        s_out: &Tensor,
    ) -> Result<Tensor> {
        let (dt_bias, ssm_a) = self.gates()?;
        heads.apply_op3_no_bwd(
            alpha,
            beta,
            &StepOp {
                form: F::FORM,
                d: self.d,
                h_v: self.h_v,
                dt_bias,
                ssm_a,
                s: s.clone(),
                s_out: s_out.clone(),
            },
        )
    }

    /// `B` rows of one pool: `heads [B, 3, h_v, d]`, `alpha` / `beta` `[B,
    /// h_v]`; row `b` reads its slab at `phases[b]` of `slots[b]` (`u32 [B]`)
    /// and writes the next phase → `o [B, h_v, d]`.
    pub fn rows(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: PoolLayer,
        slots: Tensor,
        phases: Tensor,
    ) -> Result<Tensor> {
        let (dt_bias, ssm_a) = self.gates()?;
        heads.apply_op3_no_bwd(
            alpha,
            beta,
            &StepRowsOp {
                form: F::FORM,
                rows: slots.elem_count(),
                d: self.d,
                h_v: self.h_v,
                dt_bias,
                ssm_a,
                s,
                slots,
                phases,
            },
        )
    }

    /// `T` positions of one row: `heads [T, 3, h_v, d]`, `alpha` / `beta` `[T,
    /// h_v]`; position `t` reads phase `phase + t` of `slot` and writes the
    /// next, so an accept of `a` positions is phase `phase + a` → `o [T, h_v,
    /// d]`. The pool needs `T + 1` phases.
    pub fn seq(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: PoolLayer,
        slot: usize,
        phase: usize,
    ) -> Result<Tensor> {
        let (dt_bias, ssm_a) = self.gates()?;
        heads.apply_op3_no_bwd(
            alpha,
            beta,
            &StepSeqOp {
                form: F::FORM,
                steps: alpha.dim(0)?,
                d: self.d,
                h_v: self.h_v,
                dt_bias,
                ssm_a,
                s,
                slot,
                phase,
            },
        )
    }
}

/// The form the construction tier chose, each arm a [`GdnStep`] of its type:
/// a `match` per call, never a `dyn`.
#[derive(Debug, Clone)]
pub enum GdnStepArm {
    Loop(GdnStep<LoopForm>),
    Register(GdnStep<RegisterForm>),
}

impl GdnStepArm {
    pub fn new(form: GdnStepForm, d: usize, h_v: usize, dt_bias: &Tensor, ssm_a: &Tensor) -> Self {
        match form {
            GdnStepForm::Loop => GdnStepArm::Loop(GdnStep::new(d, h_v, dt_bias, ssm_a)),
            GdnStepForm::Register => GdnStepArm::Register(GdnStep::new(d, h_v, dt_bias, ssm_a)),
        }
    }

    /// The arm `tuning.gdn.step_reg` names.
    pub fn configured(d: usize, h_v: usize, dt_bias: &Tensor, ssm_a: &Tensor) -> Self {
        Self::new(GdnStepForm::configured(), d, h_v, dt_bias, ssm_a)
    }

    pub fn form(&self) -> GdnStepForm {
        match self {
            GdnStepArm::Loop(_) => GdnStepForm::Loop,
            GdnStepArm::Register(_) => GdnStepForm::Register,
        }
    }

    /// [`GdnStep::step`] in this arm's form.
    pub fn step(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: &Tensor,
        s_out: &Tensor,
    ) -> Result<Tensor> {
        match self {
            GdnStepArm::Loop(g) => g.step(heads, alpha, beta, s, s_out),
            GdnStepArm::Register(g) => g.step(heads, alpha, beta, s, s_out),
        }
    }

    /// [`GdnStep::rows`] in this arm's form.
    pub fn rows(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: PoolLayer,
        slots: Tensor,
        phases: Tensor,
    ) -> Result<Tensor> {
        match self {
            GdnStepArm::Loop(g) => g.rows(heads, alpha, beta, s, slots, phases),
            GdnStepArm::Register(g) => g.rows(heads, alpha, beta, s, slots, phases),
        }
    }

    /// [`GdnStep::seq`] in this arm's form.
    pub fn seq(
        &self,
        heads: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        s: PoolLayer,
        slot: usize,
        phase: usize,
    ) -> Result<Tensor> {
        match self {
            GdnStepArm::Loop(g) => g.seq(heads, alpha, beta, s, slot, phase),
            GdnStepArm::Register(g) => g.seq(heads, alpha, beta, s, slot, phase),
        }
    }
}

#[cfg(feature = "metal")]
fn metal_of(t: &Tensor, what: &str) -> Result<(lloom_tensor::MetalStorage, Layout)> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        lloom_tensor::Storage::Metal(m) => Ok((m.clone(), l.clone())),
        _ => lloom_tensor::bail!("{what}: every operand on the metal device"),
    }
}

#[cfg(feature = "metal")]
fn bo<'a>(
    st: &'a lloom_tensor::MetalStorage,
    l: &Layout,
) -> lloom_tensor::kernels::BufferOffset<'a> {
    use lloom_tensor::backend::BackendStorage;
    lloom_tensor::metal_backend::buffer_o(st.buffer(), l, st.dtype())
}

/// One row's step (`gdn_step_f32` or its register form).
struct StepOp {
    form: GdnStepForm,
    d: usize,
    h_v: usize,
    dt_bias: Tensor,
    ssm_a: Tensor,
    s: Tensor,
    s_out: Tensor,
}

impl lloom_tensor::CustomOp3 for StepOp {
    fn name(&self) -> &'static str {
        "gdn-step"
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
        lloom_tensor::bail!("gdn_step: no cpu kernel; the mixer's composed path is the CPU's")
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
        // Every term that reaches `S` enters at f32 (the model doc's "Gated
        // DeltaNet"): the kernel reads these buffers as float, so a narrower
        // input is refused here rather than misread.
        let f32 = DType::F32;
        if [s1.dtype(), s2.dtype(), s3.dtype()] != [f32; 3]
            || self.dt_bias.dtype() != f32
            || self.ssm_a.dtype() != f32
            || self.s.dtype() != f32
            || self.s_out.dtype() != f32
        {
            lloom_tensor::bail!(
                "gdn_step: heads / alpha / beta / dt_bias / ssm_a / S must be f32, got {:?} / {:?} / {:?} / {:?} / {:?} / {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype(),
                self.dt_bias.dtype(),
                self.ssm_a.dtype(),
                self.s.dtype()
            )
        }
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
        let (bs, bl) = metal_of(&self.dt_bias, "gdn_step")?;
        let (as_, al) = metal_of(&self.ssm_a, "gdn_step")?;
        let (ss, sl) = metal_of(&self.s, "gdn_step")?;
        let (os, ol) = metal_of(&self.s_out, "gdn_step")?;
        if bl.shape().elem_count() != h_v
            || al.shape().elem_count() != h_v
            || sl.shape().elem_count() != d * d * h_v
            || ol.shape().elem_count() != d * d * h_v
            || !bl.is_contiguous()
            || !al.is_contiguous()
            || !sl.is_contiguous()
            || !ol.is_contiguous()
        {
            lloom_tensor::bail!("gdn_step: dt_bias / ssm_a [h_v], S [d, d, h_v], contiguous")
        }
        if self.s.id() == self.s_out.id() {
            lloom_tensor::bail!("gdn_step: S in and out must be two buffers")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_step");
        let elem_count = h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, f32)
            .with_label("gdn_o")
            .build()?;
        lloom_tensor::kernels::call_gdn_step(
            device.metal_device(),
            &encoder,
            device.kernels(),
            d,
            h_v,
            bo(s1, l1),
            bo(s2, l2),
            bo(s3, l3),
            bo(&bs, &bl),
            bo(&as_, &al),
            bo(&ss, &sl),
            &output,
            bo(&os, &ol),
            self.form == GdnStepForm::Register,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new_allocated(output, device.clone(), elem_count, f32)?,
            Shape::from((h_v, d)),
        ))
    }
}

/// The rows' step (`gdn_step_rows_f32` or its register form).
struct StepRowsOp {
    form: GdnStepForm,
    rows: usize,
    d: usize,
    h_v: usize,
    dt_bias: Tensor,
    ssm_a: Tensor,
    s: PoolLayer,
    slots: Tensor,
    phases: Tensor,
}

impl lloom_tensor::CustomOp3 for StepRowsOp {
    fn name(&self) -> &'static str {
        "gdn-step-rows"
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
        lloom_tensor::bail!("gdn_step_rows: no cpu kernel; the per-row path composes it")
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
        if l1.shape().elem_count() != rows * 3 * h_v * d
            || l2.shape().elem_count() != rows * h_v
            || l3.shape().elem_count() != rows * h_v
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_step_rows: heads / alpha / beta vs rows {rows} d {d} h_v {h_v}"
            )
        }
        let (bs, bl) = metal_of(&self.dt_bias, "gdn_step_rows")?;
        let (as_, al) = metal_of(&self.ssm_a, "gdn_step_rows")?;
        let (ps, pl) = metal_of(&self.s.tensor, "gdn_step_rows")?;
        let (ss, sl) = metal_of(&self.slots, "gdn_step_rows")?;
        let (hs, hl) = metal_of(&self.phases, "gdn_step_rows")?;
        let slab = d * d * h_v;
        if bl.shape().elem_count() != h_v
            || al.shape().elem_count() != h_v
            || pl.shape().elem_count() != self.s.n_phases * self.s.n_slots * slab
            || sl.shape().elem_count() != rows
            || hl.shape().elem_count() != rows
            || !pl.is_contiguous()
            || !sl.is_contiguous()
            || !hl.is_contiguous()
        {
            lloom_tensor::bail!("gdn_step_rows: the pool / tables' shapes")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_step_rows");
        let elem_count = rows * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, DType::F32)
            .with_label("gdn_o_rows")
            .build()?;
        lloom_tensor::kernels::call_gdn_step_rows(
            device.metal_device(),
            &encoder,
            device.kernels(),
            rows,
            d,
            h_v,
            self.s.n_slots,
            self.s.n_phases,
            bo(s1, l1),
            bo(s2, l2),
            bo(s3, l3),
            bo(&bs, &bl),
            bo(&as_, &al),
            bo(&ps, &pl),
            bo(&ss, &sl),
            bo(&hs, &hl),
            &output,
            self.form == GdnStepForm::Register,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new_allocated(
                output,
                device.clone(),
                elem_count,
                s1.dtype(),
            )?,
            Shape::from((rows, h_v, d)),
        ))
    }
}

/// The multi-position step (`gdn_step_seq_f32` or its register form).
struct StepSeqOp {
    form: GdnStepForm,
    steps: usize,
    d: usize,
    h_v: usize,
    dt_bias: Tensor,
    ssm_a: Tensor,
    s: PoolLayer,
    slot: usize,
    phase: usize,
}

impl lloom_tensor::CustomOp3 for StepSeqOp {
    fn name(&self) -> &'static str {
        "gdn-step-seq"
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
        lloom_tensor::bail!("gdn_step_seq: no cpu kernel; the per-position path composes it")
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
        let (steps, d, h_v) = (self.steps, self.d, self.h_v);
        if l1.shape().elem_count() != steps * 3 * h_v * d
            || l2.shape().elem_count() != steps * h_v
            || l3.shape().elem_count() != steps * h_v
            || !l1.is_contiguous()
            || !l2.is_contiguous()
            || !l3.is_contiguous()
        {
            lloom_tensor::bail!(
                "gdn_step_seq: heads / alpha / beta vs steps {steps} d {d} h_v {h_v}"
            )
        }
        if steps + 1 > self.s.n_phases {
            lloom_tensor::bail!(
                "gdn_step_seq: {steps} positions over {} phases",
                self.s.n_phases
            )
        }
        let (bs, bl) = metal_of(&self.dt_bias, "gdn_step_seq")?;
        let (as_, al) = metal_of(&self.ssm_a, "gdn_step_seq")?;
        let (ps, pl) = metal_of(&self.s.tensor, "gdn_step_seq")?;
        let slab = d * d * h_v;
        if bl.shape().elem_count() != h_v
            || al.shape().elem_count() != h_v
            || pl.shape().elem_count() != self.s.n_phases * self.s.n_slots * slab
            || !pl.is_contiguous()
        {
            lloom_tensor::bail!("gdn_step_seq: the pool's shape")
        }
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gdn_step_seq");
        let elem_count = steps * h_v * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, DType::F32)
            .with_label("gdn_o_seq")
            .build()?;
        lloom_tensor::kernels::call_gdn_step_seq(
            device.metal_device(),
            &encoder,
            device.kernels(),
            steps,
            d,
            h_v,
            self.s.n_slots,
            self.s.n_phases,
            self.slot,
            self.phase,
            bo(s1, l1),
            bo(s2, l2),
            bo(s3, l3),
            bo(&bs, &bl),
            bo(&as_, &al),
            bo(&ps, &pl),
            &output,
            self.form == GdnStepForm::Register,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new_allocated(
                output,
                device.clone(),
                elem_count,
                s1.dtype(),
            )?,
            Shape::from((steps, h_v, d)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::Device;

    /// The arm the construction tier picks carries its form.
    #[test]
    fn the_arm_is_the_form_it_was_built_with() -> Result<()> {
        let dev = Device::Cpu;
        let v = Tensor::zeros(4, DType::F32, &dev)?;
        for form in [GdnStepForm::Loop, GdnStepForm::Register] {
            assert_eq!(GdnStepArm::new(form, 8, 4, &v, &v).form(), form);
        }
        Ok(())
    }

    /// `mamba_ssm_dtype: float32`, enforced: the step refuses a state, and an
    /// input, that is not f32 rather than read its bytes as float.
    #[cfg(feature = "metal")]
    #[test]
    fn the_step_refuses_a_state_that_is_not_f32() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let (d, h_v) = (16usize, 4usize);
        let (f32, f16) = (DType::F32, DType::F16);
        let ab = Tensor::zeros(h_v, f32, &dev)?;
        let step = GdnStep::<LoopForm>::new(d, h_v, &ab, &ab);
        let run = |heads_dtype: DType, s_dtype: DType| -> Result<Tensor> {
            step.step(
                &Tensor::zeros(3 * h_v * d, heads_dtype, &dev)?,
                &ab,
                &ab,
                &Tensor::zeros((d, d, h_v), s_dtype, &dev)?,
                &Tensor::zeros((d, d, h_v), s_dtype, &dev)?,
            )
        };
        assert!(run(f32, f32).is_ok(), "f32 inputs and state must run");
        let err = run(f32, f16).unwrap_err().to_string();
        assert!(
            err.contains("must be f32"),
            "an f16 state was not refused: {err}"
        );
        let err = run(f16, f32).unwrap_err().to_string();
        assert!(
            err.contains("must be f32"),
            "an f16 input was not refused: {err}"
        );
        Ok(())
    }
}
