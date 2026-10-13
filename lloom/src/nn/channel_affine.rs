//! Inference-only channel affine map for already-folded normalization weights.
//! Unlike BatchNorm this stores no running statistics and performs no folding.
//! The converter's supplied arithmetic is exactly `x * scale + bias`.
use lloom_tensor::{DType, Module, Result, Tensor};

#[derive(Clone, Debug)]
pub struct ChannelAffine {
    scale: Tensor,
    bias: Tensor,
}

impl ChannelAffine {
    /// Retain one-dimensional, common-device/common-dtype floating parameters.
    /// Their values are not scanned or copied to the host for validation.
    pub fn new(scale: Tensor, bias: Tensor) -> Result<Self> {
        if scale.rank() != 1 || scale.dim(0)? == 0 || bias.dims() != scale.dims() {
            lloom_tensor::bail!("channel affine requires matching nonempty [C] parameters")
        }
        if !matches!(
            scale.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) || bias.dtype() != scale.dtype()
            || !bias.device().same_device(scale.device())
        {
            lloom_tensor::bail!(
                "channel affine parameters require common floating dtype and device"
            )
        }
        Ok(Self { scale, bias })
    }

    pub fn scale(&self) -> &Tensor {
        &self.scale
    }
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}

impl Module for ChannelAffine {
    /// Apply to [B,C,...], including rank two, without changing axis order.
    /// Broadcasted parameter views own no new numeric storage. Multiply and add
    /// each produce an output-sized tensor; the caller admits their lifetimes.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() < 2
            || x.dim(1)? != self.scale.dim(0)?
            || x.dtype() != self.scale.dtype()
            || !x.device().same_device(self.scale.device())
        {
            lloom_tensor::bail!("channel affine input must match [B,C,...], dtype and device")
        }
        x.dims()
            .iter()
            .try_fold(x.dtype().size_in_bytes(), |n, &d| n.checked_mul(d))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg("channel affine output byte size overflow".into())
            })?;
        let mut shape = vec![1; x.rank()];
        shape[1] = self.scale.dim(0)?;
        x.broadcast_mul(&self.scale.reshape(shape.as_slice())?)?
            .broadcast_add(&self.bias.reshape(shape.as_slice())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BatchNorm, ModuleT};
    use lloom_tensor::Device;

    #[test]
    fn folded_parameters_match_scalar_and_existing_eval_on_strided_input() -> Result<()> {
        let means = [0.25f32, -0.5, 0.125];
        let vars = [0.25f32, 1., 2.];
        let gamma = [0.75f32, -1.25, 0.5];
        let beta = [0.2f32, -0.1, 0.3];
        let scale: Vec<f32> = gamma
            .iter()
            .zip(vars)
            .map(|(&g, v)| g / (v + 1e-5f32).sqrt())
            .collect();
        let bias: Vec<f32> = beta
            .iter()
            .zip(means)
            .zip(&scale)
            .map(|((&b, m), &a)| b - m * a)
            .collect();
        let tensor = |v: &[f32]| Tensor::from_slice(v, v.len(), &Device::Cpu);
        let layer = ChannelAffine::new(tensor(&scale)?, tensor(&bias)?)?;
        let bn = BatchNorm::new(
            3,
            tensor(&means)?,
            tensor(&vars)?,
            tensor(&gamma)?,
            tensor(&beta)?,
            1e-5,
        )?;
        let input: Vec<f32> = (0..24).map(|i| (i as f32 - 8.) / 7.).collect();
        let x = Tensor::from_vec(input.clone(), (2, 4, 3), &Device::Cpu)?.transpose(1, 2)?;
        assert!(!x.is_contiguous());
        let actual = layer.forward(&x)?.to_vec3::<f32>()?;
        let legacy = bn.forward_t(&x, false)?.to_vec3::<f32>()?;
        for b in 0..2 {
            for c in 0..3 {
                for t in 0..4 {
                    let expected = input[(b * 4 + t) * 3 + c] * scale[c] + bias[c];
                    assert!(
                        (actual[b][c][t] - expected).abs() < 1e-6,
                        "scalar b{b} c{c} t{t}"
                    );
                    // Folding changes arithmetic order; do not assert bit identity.
                    assert!(
                        (actual[b][c][t] - legacy[b][c][t]).abs() < 1e-6,
                        "eval b{b} c{c} t{t}"
                    );
                }
            }
        }
        assert_eq!(layer.scale().dims(), [3]);
        assert_eq!(layer.bias().dims(), [3]);
        Ok(())
    }

    #[test]
    fn rank_two_and_half_dtypes_keep_channel_axis() -> Result<()> {
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let scale = Tensor::new(&[2f32, -0.5], &Device::Cpu)?.to_dtype(dtype)?;
            let bias = Tensor::new(&[0.25f32, 1.], &Device::Cpu)?.to_dtype(dtype)?;
            let layer = ChannelAffine::new(scale, bias)?;
            let x = Tensor::new(&[[1f32, 2.], [-1., 4.]], &Device::Cpu)?.to_dtype(dtype)?;
            let output = layer.forward(&x)?;
            assert_eq!(output.dtype(), dtype);
            assert_eq!(
                output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                [[2.25, 0.], [-1.75, -1.]]
            );
        }
        Ok(())
    }

    #[test]
    fn malformed_geometry_and_logical_overflow_refuse_before_arithmetic() -> Result<()> {
        let vector = Tensor::ones(3, DType::F32, &Device::Cpu)?;
        assert!(ChannelAffine::new(vector.unsqueeze(0)?, vector.clone()).is_err());
        assert!(
            ChannelAffine::new(vector.clone(), Tensor::ones(2, DType::F32, &Device::Cpu)?).is_err()
        );
        assert!(ChannelAffine::new(vector.clone(), vector.to_dtype(DType::F16)?).is_err());
        assert!(
            ChannelAffine::new(vector.to_dtype(DType::U32)?, vector.to_dtype(DType::U32)?).is_err()
        );
        let layer = ChannelAffine::new(vector.clone(), vector.clone())?;
        assert!(layer.forward(&vector).is_err());
        assert!(layer
            .forward(&Tensor::ones((1, 2, 1), DType::F32, &Device::Cpu)?)
            .is_err());
        assert!(layer
            .forward(&Tensor::ones((1, 3, 1), DType::F16, &Device::Cpu)?)
            .is_err());
        // Scalar storage only; never execute a giant tensor operation.
        let huge = Tensor::new(1f32, &Device::Cpu)?.broadcast_as((
            i32::MAX as usize,
            3,
            i32::MAX as usize,
            3,
        ))?;
        let error = layer.forward(&huge).unwrap_err().to_string();
        assert!(error.contains("output byte size overflow"), "{error}");
        Ok(())
    }
}
