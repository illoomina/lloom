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
    a.device().is_metal()
        && a.dtype() == lloom_tensor::DType::F32
        && a.is_contiguous()
        && dims.len() >= 2
        && dims[dims.len() - 1] == dims[dims.len() - 2]
}

pub(crate) fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    a.apply_op1_no_bwd(&UnitLowerInverse)
}
