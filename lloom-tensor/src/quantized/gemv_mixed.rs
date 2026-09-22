//! `f16` weights × `f32` activations → `f32`, one launch (Metal): the
//! [`QMatMul::TensorF16`](super::QMatMul::TensorF16) decode path without the
//! two cast launches the classical route pays around every `gemv_float16`.
//! Optionally `silu(s · y)` or `sigmoid(s · y)` at the store
//! ([`MixedEpilogue`]). The CPU path composes the same arithmetic in f32
//! (the f16 weights widened, the f32 vector as is) and is the reference the
//! kernel is tested against (`tests/gemv_mixed.rs`).

use crate::{CpuStorage, DType, Layout, Result, Shape, Tensor};

#[cfg(feature = "metal")]
pub use crate::kernels::MixedEpilogue;

/// The epilogue names, for the CPU path when the kernel crate is not built.
#[cfg(not(feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MixedEpilogue {
    #[default]
    None,
    Silu,
    Sigmoid,
}

/// The op: `w` is `[n_out, k]` f16, contiguous; the input is `[.., k]` f32 with
/// every leading dim folded into the batch; the output `[.., n_out]` f32.
pub(crate) struct GemvMixed {
    pub w: Tensor,
    pub epilogue: MixedEpilogue,
    pub scale: f32,
}

fn apply_epilogue(y: f32, epilogue: MixedEpilogue, scale: f32) -> f32 {
    match epilogue {
        MixedEpilogue::None => y,
        MixedEpilogue::Silu => {
            let v = scale * y;
            v / (1.0 + (-v).exp())
        }
        MixedEpilogue::Sigmoid => {
            let v = scale * y;
            1.0 / (1.0 + (-v).exp())
        }
    }
}

impl crate::CustomOp1 for GemvMixed {
    fn name(&self) -> &'static str {
        "gemv-mixed-f16w-f32"
    }

    /// The reference: the f16 weights widened to f32, a plain f32 dot per
    /// output row, the epilogue at f32.
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        use crate::backend::BackendStorage;
        let (n_out, k) = self.w.dims2()?;
        let w: Vec<f32> = self.w.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let x = match s {
            CpuStorage::F32(v) => v,
            other => crate::bail!("gemv_mixed: f32 input only, got {:?}", other.dtype()),
        };
        if !l.is_contiguous() {
            crate::bail!("gemv_mixed: the input must be contiguous")
        }
        let dims = l.dims();
        if dims.last() != Some(&k) {
            crate::bail!("gemv_mixed: input {:?} vs weight [{n_out}, {k}]", l.shape())
        }
        let b = l.shape().elem_count() / k;
        let x = &x[l.start_offset()..l.start_offset() + b * k];
        let mut out = vec![0f32; b * n_out];
        for bi in 0..b {
            let xr = &x[bi * k..(bi + 1) * k];
            for r in 0..n_out {
                let wr = &w[r * k..(r + 1) * k];
                let mut acc = 0f32;
                for i in 0..k {
                    acc += wr[i] * xr[i];
                }
                out[bi * n_out + r] = apply_epilogue(acc, self.epilogue, self.scale);
            }
        }
        let mut shape = dims.to_vec();
        *shape.last_mut().unwrap() = n_out;
        Ok((CpuStorage::F32(out), Shape::from(shape)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &crate::MetalStorage,
        l: &Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        use crate::backend::BackendStorage;
        let (n_out, k) = self.w.dims2()?;
        if s.dtype() != DType::F32 {
            crate::bail!("gemv_mixed: f32 input only, got {:?}", s.dtype())
        }
        if !l.is_contiguous() {
            crate::bail!("gemv_mixed: the input must be contiguous")
        }
        let dims = l.dims();
        if dims.last() != Some(&k) {
            crate::bail!("gemv_mixed: input {:?} vs weight [{n_out}, {k}]", l.shape())
        }
        let b = l.shape().elem_count() / k;
        let (ws, wl) = self.w.storage_and_layout();
        let ws = match &*ws {
            crate::Storage::Metal(m) => m,
            _ => crate::bail!("gemv_mixed: the weight must be on the metal device"),
        };
        if ws.dtype() != DType::F16 || !wl.is_contiguous() {
            crate::bail!("gemv_mixed: the weight must be contiguous f16")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("gemv_mixed");
        let elem_count = b * n_out;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, DType::F32)
            .with_label("gemv_mixed")
            .build()?;
        crate::kernels::call_mlx_gemv_mixed(
            device.metal_device(),
            &encoder,
            device.kernels(),
            (b, n_out, k),
            ws.buffer(),
            wl.start_offset() * DType::F16.size_in_bytes(),
            s.buffer(),
            l.start_offset() * DType::F32.size_in_bytes(),
            k,
            &output,
            self.epilogue,
            self.scale,
        )
        .map_err(crate::Error::wrap)?;
        let mut shape = dims.to_vec();
        *shape.last_mut().unwrap() = n_out;
        Ok((
            crate::MetalStorage::new(output, device.clone(), elem_count, DType::F32),
            Shape::from(shape),
        ))
    }
}

/// The most vectors the mixed arm serves in one dispatch: a decode step's
/// rows. Past it a prefill is a GEMM's job (the classical cast route).
pub const GEMV_MIXED_MAX_ROWS: usize = 4;
