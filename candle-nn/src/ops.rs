//! Tensor ops.
//!

use candle::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

struct Sigmoid;

impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use candle::cuda_backend::SlicePtrOrNull;
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let shape = layout.shape();
                let dims = shape.dims();
                let el_count = shape.elem_count();
                let cfg = LaunchConfig::for_num_elems(el_count as u32);
                let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("usigmoid"), &kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<T>(el_count)? };

                let mut builder = func.builder();
                candle::builder_arg!(builder, el_count, dims.len());
                ds.builder_arg(&mut builder);
                builder.arg(src);
                builder.arg(&out);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;
        let device = storage.device();
        let dtype = storage.dtype();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(el_count, dtype)
            .with_label("sigmoid")
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("sigmoid");
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };

        if layout.is_contiguous() {
            use candle_metal_kernels::unary::contiguous;
            let kernel_name = match dtype {
                DType::F16 => contiguous::sigmoid::HALF,
                DType::F32 => contiguous::sigmoid::FLOAT,
                DType::BF16 => contiguous::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal contiguous unary sigmoid {dtype:?} not implemented")
                }
            };
            candle_metal_kernels::call_unary_contiguous(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                dtype.size_in_bytes(),
                el_count,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match dtype {
                DType::F16 => strided::sigmoid::HALF,
                DType::F32 => strided::sigmoid::FLOAT,
                DType::BF16 => strided::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal strided unary sigmoid {dtype:?} not implemented")
                }
            };
            let dst = candle_metal_kernels::BufferOffset::zero_offset(&buffer);
            candle_metal_kernels::call_unary_strided(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                layout.dims(),
                src,
                layout.stride(),
                dst,
            )
            .map_err(MetalError::from)?;
        }

        let new_storage = candle::MetalStorage::new(buffer, device.clone(), el_count, dtype);
        Ok((new_storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    xs * (1.0 + xs.exp()?)?.log()?.tanh()
}

pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn selu(xs: &Tensor, alpha: f32, gamma: f32) -> Result<Tensor> {
    let is_pos = xs.gt(0f32)?;
    let alpha_t = Tensor::full(alpha, xs.dims(), xs.device())?;
    let neg = xs.exp()?.mul(&alpha_t)?.sub(&alpha_t)?;
    let selu = is_pos.where_cond(xs, &neg)?;
    let gamma_t = Tensor::full(gamma, xs.dims(), xs.device())?;
    selu.broadcast_mul(&gamma_t)
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                candle::builder_arg!(builder, n_cols as i32);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = storage.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax");
        let kernels = device.kernels();
        let name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle::bail!("softmax-last-dim is not implemented for {dtype:?}"),
        };

        let n = layout.stride().len();
        if !(layout.is_contiguous() && layout.stride()[n - 1] == 1) {
            candle::bail!("Non contiguous softmax-last-dim is not implemented");
        }

        let last_dim = layout.dims()[layout.shape().rank() - 1];
        let elem_count = layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, storage.dtype())
            .with_label("softmax")
            .build()?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, storage.dtype());
        Ok((newstorage, layout.shape().clone()))
    }
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let n_rows = el_count / dim_m1;
            let mut dst = vec![T::zero(); el_count];

            fn rms_row<
                T: candle::WithDType
                    + num_traits::Float
                    + num_traits::AsPrimitive<f32>
                    + num_traits::FromPrimitive,
            >(
                src: &[T],
                alpha: &[T],
                n: usize,
                eps: f32,
                dst: &mut [T],
            ) {
                let sum2 = src
                    .iter()
                    .map(|&v| {
                        let v = v.as_();
                        v * v
                    })
                    .sum::<f32>();
                let m = (sum2 / n as f32 + eps).sqrt();
                let m = T::from_f32(m).unwrap_or_else(T::nan);
                for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                    *d = *s / m * *alpha
                }
            }

            if n_rows <= 32 {
                let n = dim_m1;
                for row in 0..n_rows {
                    let src = &src[row * n..(row + 1) * n];
                    let dst = &mut dst[row * n..(row + 1) * n];
                    rms_row(src, alpha, n, eps, dst);
                }
            } else {
                src.par_chunks(dim_m1)
                    .zip(dst.par_chunks_mut(dim_m1))
                    .for_each(|(src, dst)| {
                        let n = src.len();
                        rms_row(src, alpha, n, eps, dst);
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("rmsnorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rmsnorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle::bail!("rmsnorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous rmsnorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("rmsnorm")
            .build()?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
}

impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map3, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map3 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                beta: &CudaSlice<T>,
                beta_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let beta = match beta_layout.contiguous_offsets() {
                    None => candle::bail!("beta has to be contiguous"),
                    Some((o1, o2)) => beta.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("layernorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                builder.arg(&beta);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, &s3.slice, l3, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("layernorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("layernorm")
            .build()?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

pub fn layer_norm(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
    mask: Option<Tensor>,
    do_causal: bool,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
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
        candle::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();
        let out_shape = Shape::from_dims(&out_dims);
        let out_layout = Layout::contiguous(out_shape.clone());

        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, q.dtype())
            .with_label("sdpa_o")
            .build()?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;
        let k_seq = k_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim = q_head == 32
            || q_head == 64
            || q_head == 72
            || q_head == 80
            || q_head == 96
            || q_head == 128
            || q_head == 256
            || q_head == 512;

        let supports_sdpa_full_mask = self.mask.is_none() || q_seq <= k_seq;
        // F32 full attention at head_dim=512 exceeds 32KB Metal threadgroup memory
        let supports_sdpa_full_dtype = !(q_head == 512 && q.dtype() == DType::F32);
        let supports_sdpa_full =
            q_seq > 1 && supported_head_dim && supports_sdpa_full_mask && supports_sdpa_full_dtype;
        let supports_sdpa_vector = q_seq == 1 && supported_head_dim && q_seq <= k_seq;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("unsupported sdpa type {other:?}"),
        };

        let encoder = q.device().command_encoder()?;
        if supports_sdpa_vector {
            // Route to the 2 pass fused attention if the k seqlen is large.
            // https://github.com/ml-explore/mlx/pull/1597
            const TWO_PASS_K_THRESHOLD: usize = 1024;
            if k_seq >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_intermediate")
                    .build()?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_sums")
                    .build()?;
                let maxs = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_maxs")
                    .build()?;

                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector_2pass(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset() * q.dtype().size_in_bytes(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset() * k.dtype().size_in_bytes(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset() * v.dtype().size_in_bytes(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            } else {
                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset() * q.dtype().size_in_bytes(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset() * k.dtype().size_in_bytes(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset() * v.dtype().size_in_bytes(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            encoder.set_label("full_attention");
            if self.softcapping != 1. {
                candle::bail!("SDPA full requires softcapping to be disabled (1.0)");
            }

            let mask_s_l = self.mask.as_ref().map(|m| m.storage_and_layout());

            let (mask_type, mask_buffer, mask_strides) = if let Some(mask) = &self.mask {
                let (mask_s, mask_l) = mask_s_l.as_ref().unwrap();

                let mask_buffer = match &**mask_s {
                    candle::Storage::Metal(m) => m.buffer(),
                    _ => candle::bail!("Expected metal device for mask"),
                };

                let mask_type = match mask.dtype() {
                    DType::BF16 => SdpaDType::BF16,
                    DType::F16 => SdpaDType::F16,
                    DType::F32 => SdpaDType::F32,
                    other => candle::bail!("unsupported sdpa type {other:?}"),
                };
                if mask_type != itype {
                    candle::bail!("Mask type {mask_type:?} must match q type {itype:?}");
                }

                if mask_l.dims() != [q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, k_seq] {
                    candle::bail!(
                        "Mask shape must be {:?} (bs, qheads, qseq, kseq), got {:?}",
                        [q_l.dim(0)?, q_head, q_l.dim(2)?, k_seq],
                        mask_l.dims()
                    );
                }

                (
                    Some(mask_type),
                    Some(mask_buffer),
                    Some(mask_l.stride().to_vec()),
                )
            } else {
                (None, None, None)
            };

            candle_metal_kernels::call_sdpa_full(
                q.device().device(),
                &encoder,
                q.device().kernels(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                q_l.dims(),
                q_l.stride(),
                q.buffer(),
                k_l.start_offset() * k.dtype().size_in_bytes(),
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset() * v.dtype().size_in_bytes(),
                v.buffer(),
                v_l.stride(),
                mask_type,
                mask_buffer,
                mask_strides.as_deref(),
                &output,
                out_layout.stride(),
                self.scale,
                self.do_causal,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else {
            candle::bail!("must be vector or full sdpa kernel");
        }

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `mask`: (bs, qhead, seq, kv_seq)
/// - `do_causal`: Apply causal masking. If this is true, the mask does not need to be provided.
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// Note: For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Masking is supported
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
///     - Softcapping is not supported.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping,
            mask: mask.cloned(),
            do_causal,
        },
    )
}

/// Which width a quantized KV cache stores its codes at, or `Off`
/// (`DESIGN.md` §10.7, §17 Phase 5 item 18, issue #218).
///
/// **A backend-neutral declaration converted at the metal boundary**, which is
/// `FlashScratchSizing`'s shape and for its reason: this enum is compiled on
/// every backend, so naming `candle_metal_kernels::KvQuantBits` here would put
/// a metal-only type in a struct CPU and CUDA builds construct, and §14.1's
/// *"must not break other backends"* is a hard constraint rather than a
/// preference.
///
/// **The default is `Off` and nothing here flips it** (§7.1a): every axis this
/// project has added defaults to the arm that shipped, and this one has no old
/// path to be conservative against other than not existing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQuant {
    /// The f16 cache. What ships, and what every recorded digest belongs to.
    #[default]
    Off,
    /// 8-bit codes, one byte per element.
    Int8,
    /// 4-bit codes, two elements per byte.
    Int4,
}

impl KvQuant {
    /// Every arm, for a harness and for exhaustive tests (`Sizing::ALL`'s
    /// argument, #58 / §11.3i).
    pub const ALL: [KvQuant; 3] = [KvQuant::Off, KvQuant::Int8, KvQuant::Int4];

    /// The arm's name, rendered from the resolved enum through an exhaustive
    /// `match` -- #241's rule, after a two-armed conditional over a three-armed
    /// axis mislabelled every `RESULT` line since #116.
    pub fn name(self) -> &'static str {
        match self {
            KvQuant::Off => "Off",
            KvQuant::Int8 => "Int8",
            KvQuant::Int4 => "Int4",
        }
    }

    /// Parse an arm from a flag value.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "off" | "f16" => Ok(KvQuant::Off),
            "int8" | "8" => Ok(KvQuant::Int8),
            "int4" | "4" => Ok(KvQuant::Int4),
            other => candle::bail!("unknown kv-quant {other:?}; expected off, int8 or int4"),
        }
    }

    /// Whether this arm stores codes at all.
    pub fn is_on(self) -> bool {
        !matches!(self, KvQuant::Off)
    }

    /// Bits per code, or `None` on the `Off` arm.
    pub fn bits(self) -> Option<usize> {
        match self {
            KvQuant::Off => None,
            KvQuant::Int8 => Some(8),
            KvQuant::Int4 => Some(4),
        }
    }

    /// Bytes a code plane of `n` elements occupies, or `None` on `Off`.
    ///
    /// **The backend-neutral copy of `KvQuantBits::code_bytes`**, because a
    /// `Cache` sizes its planes on every backend while the kernel constant is
    /// metal-only. `kv_quant_code_bytes_agree_across_the_boundary` asserts the
    /// two are the same function, which is §8.1b's checked-registry argument at
    /// the smallest possible scale -- the same shape
    /// `MemoryBudget::DEFAULT_FRACTION` uses for its duplicated constant.
    pub fn code_bytes(self, n: usize) -> Option<usize> {
        match self {
            KvQuant::Off => None,
            KvQuant::Int8 => Some(n),
            KvQuant::Int4 => {
                debug_assert_eq!(n % 2, 0, "an int4 plane needs an even code count");
                Some(n / 2)
            }
        }
    }

    /// The metal-side width this names.
    ///
    /// Exhaustive by construction, so an arm added to one enum and not the
    /// other is a compile error rather than a silently unreachable width
    /// (#58's mechanism, and #241's finding).
    #[cfg(feature = "metal")]
    fn to_metal(self) -> Option<candle_metal_kernels::KvQuantBits> {
        match self {
            KvQuant::Off => None,
            KvQuant::Int8 => Some(candle_metal_kernels::KvQuantBits::Eight),
            KvQuant::Int4 => Some(candle_metal_kernels::KvQuantBits::Four),
        }
    }
}

/// Quantize `src` into a caller-owned code plane and scale/bias planes
/// (issue #218).
///
/// `src` is `[B, H, seq, D]` contiguous. The planes are written at
/// `dst_offset` tokens in, which is the destination slot's live length before
/// this append -- so a `seq > 1` prefill and a one-token decode step are the
/// same call, exactly as §6.2b's in-place append is.
///
/// **Writes through to the caller's tensors rather than returning new ones.**
/// The planes are allocated once at capacity and written at a moving offset,
/// which is what preserves §6.2b's stable buffer identity -- returning fresh
/// tensors per token would put back the reallocation that section removed.
///
/// Metal only, `head_dim` 64 or 128, f16 or f32. Every other case is an
/// **error rather than a fallback**, per `flash_decoding`'s rule: a silent
/// fallback is what makes an A/B arm measure the wrong thing (§2.4).
pub fn quantize_kv_into(
    src: &Tensor,
    codes: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    capacity: usize,
    dst_offset: usize,
    quant: KvQuant,
) -> Result<()> {
    #[cfg(not(feature = "metal"))]
    {
        let _ = (src, codes, scales, biases, capacity, dst_offset, quant);
        candle::bail!("quantize_kv has no cpu impl (issue #218)")
    }
    #[cfg(feature = "metal")]
    {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let bits = match quant.to_metal() {
            Some(b) => b,
            None => candle::bail!("quantize_kv_into called on the Off arm (issue #218)"),
        };

        let (_b, n_heads, seq_len, head_dim) = src.dims4()?;
        let itype = match src.dtype() {
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("quantize_kv: unsupported dtype {other:?}"),
        };

        let (src_s, src_l) = src.storage_and_layout();
        let src_s = match &*src_s {
            candle::Storage::Metal(s) => s,
            _ => candle::bail!("quantize_kv expects a metal device"),
        };
        let (codes_s, _) = codes.storage_and_layout();
        let codes_s = match &*codes_s {
            candle::Storage::Metal(s) => s,
            _ => candle::bail!("quantize_kv expects a metal device"),
        };
        let (scales_s, _) = scales.storage_and_layout();
        let scales_s = match &*scales_s {
            candle::Storage::Metal(s) => s,
            _ => candle::bail!("quantize_kv expects a metal device"),
        };
        let (biases_s, _) = biases.storage_and_layout();
        let biases_s = match &*biases_s {
            candle::Storage::Metal(s) => s,
            _ => candle::bail!("quantize_kv expects a metal device"),
        };
        let device = src_s.device();

        // **The struct is built through its own type, never hand-packed.** Field
        // offsets and the byte/element unit split are what
        // `every_family_params_layout_matches_metal` pins; a caller assembling the
        // bytes itself would be outside that check, which is the failure §11.3k
        // records — a field at the wrong offset reads a well-formed number from the
        // wrong place and computes a plausible wrong answer.
        let wp = candle_metal_kernels::SdpaQuantWriteParams::for_append(
            seq_len * head_dim,
            capacity,
            head_dim,
            dst_offset,
            bits,
        );
        let params_buf = device.new_buffer_with_data(&[wp])?;

        let encoder = device.command_encoder()?;
        encoder.set_label("quantize_kv");

        candle_metal_kernels::call_quantize_kv(
            device.device(),
            &encoder,
            device.kernels(),
            src_l.start_offset() * src.dtype().size_in_bytes(),
            src_s.buffer(),
            codes_s.buffer(),
            scales_s.buffer(),
            biases_s.buffer(),
            &params_buf,
            n_heads,
            seq_len,
            head_dim,
            bits,
            itype,
        )
        .map_err(candle::Error::wrap)?;
        Ok(())
    }
}

/// Read a quantized KV cache back as a dense tensor (issue #218).
///
/// **What the prefill path needs.** A turn boundary re-prefills onto the
/// accumulated cache (§2.3.8d, §10.9) and must attend over the whole `kv_len`
/// -- the causal mask is `[seq_len, index_pos + seq_len]` by construction --
/// while a quantized `seq_len > 1` kernel is out of scope (§17 item 18 scopes
/// this to the decode path). So the cache stays quantized and prefill reads it
/// back; the decode steps that follow read codes directly, which is where the
/// mechanism pays.
struct DequantizeKv {
    scales: Tensor,
    biases: Tensor,
    b_sz: usize,
    n_heads: usize,
    n_tokens: usize,
    head_dim: usize,
    capacity: usize,
    dtype: DType,
    quant: KvQuant,
}

impl candle::CustomOp1 for DequantizeKv {
    fn name(&self) -> &'static str {
        "metal-dequantize-kv"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("dequantize_kv has no cpu impl (issue #218)")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        codes: &candle::MetalStorage,
        _l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let bits = match self.quant.to_metal() {
            Some(b) => b,
            None => candle::bail!("dequantize_kv reached on the Off arm (issue #218)"),
        };
        let itype = match self.dtype {
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("dequantize_kv: unsupported dtype {other:?}"),
        };

        let (scales_s, _sl) = self.scales.storage_and_layout();
        let (biases_s, _bl) = self.biases.storage_and_layout();
        let scales_m = match &*scales_s {
            candle::Storage::Metal(m) => m,
            _ => candle::bail!("dequantize_kv expects a metal scales plane"),
        };
        let biases_m = match &*biases_s {
            candle::Storage::Metal(m) => m,
            _ => candle::bail!("dequantize_kv expects a metal biases plane"),
        };

        let device = codes.device();
        let elem_count = self.b_sz * self.n_heads * self.n_tokens * self.head_dim;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, self.dtype)
            .with_label("dequantize_kv_o")
            .build()?;

        // `src_head_stride` carries the DESTINATION head stride here, because
        // the direction is reversed. The struct is shared with the write path
        // and its layout is pinned by
        // `every_family_params_layout_matches_metal`, so the field is reused
        // rather than a near-duplicate struct minted -- and it is named at both
        // ends rather than left to be inferred from the call.
        let wp = candle_metal_kernels::SdpaQuantWriteParams::for_append(
            self.n_tokens * self.head_dim,
            self.capacity,
            self.head_dim,
            0,
            bits,
        );
        let params_buf = device.new_buffer_with_data(&[wp])?;

        let encoder = device.command_encoder()?;
        encoder.set_label("dequantize_kv");
        candle_metal_kernels::call_dequantize_kv(
            device.device(),
            &encoder,
            device.kernels(),
            codes.buffer(),
            scales_m.buffer(),
            biases_m.buffer(),
            &output,
            &params_buf,
            self.b_sz * self.n_heads,
            self.n_tokens,
            self.head_dim,
            bits,
            itype,
        )
        .map_err(candle::Error::wrap)?;

        let storage = candle::MetalStorage::new(output, device.clone(), elem_count, self.dtype);
        Ok((
            storage,
            Shape::from_dims(&[self.b_sz, self.n_heads, self.n_tokens, self.head_dim]),
        ))
    }
}

/// Read a quantized KV cache back as `[b, h, n_tokens, d]` at `dtype`.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_kv(
    codes: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    b_sz: usize,
    n_heads: usize,
    n_tokens: usize,
    head_dim: usize,
    capacity: usize,
    dtype: DType,
    quant: KvQuant,
) -> Result<Tensor> {
    codes.apply_op1_no_bwd(&DequantizeKv {
        scales: scales.clone(),
        biases: biases.clone(),
        b_sz,
        n_heads,
        n_tokens,
        head_dim,
        capacity,
        dtype,
        quant,
    })
}

/// Attention over a quantized KV cache (issue #218).
///
/// The six planes are captured rather than passed as operands, which is
/// `Sdpa`'s own shape for its `mask`: `CustomOp3` takes three tensors and this
/// path reads seven, so the alternative would be a new op arity for one caller.
#[allow(dead_code)]
struct SdpaQuant {
    k_codes: Tensor,
    k_scales: Tensor,
    k_biases: Tensor,
    v_codes: Tensor,
    v_scales: Tensor,
    v_biases: Tensor,
    /// The read params, built through `SdpaQuantParams::for_step` at the call
    /// site rather than hand-packed -- §11.3k's layout check only covers the
    /// struct, so a caller assembling the bytes is outside it.
    gqa_factor: usize,
    n: usize,
    capacity: usize,
    scale: f32,
    softcapping: f32,
    quant: KvQuant,
}

impl candle::CustomOp1 for SdpaQuant {
    fn name(&self) -> &'static str {
        "metal-sdpa-quant"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("sdpa_quant has no cpu impl (issue #218)")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let bits = match self.quant.to_metal() {
            Some(b) => b,
            None => candle::bail!("sdpa_quant reached on the Off arm (issue #218)"),
        };
        let itype = match q.dtype() {
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("sdpa_quant: unsupported dtype {other:?}"),
        };
        // `seq_len == 1` only, like `sdpa_vector` itself: the kernel is a
        // vector kernel and a `seq > 1` request must be an error rather than a
        // silent fallback (§2.4, and `flash_decoding`'s own rule).
        if q_l.dim(2)? != 1 {
            candle::bail!(
                "sdpa_quant is a decode kernel: q seq must be 1, got {}",
                q_l.dim(2)?
            );
        }

        let device = q.device();
        let out_dims = q_l.dims().to_vec();
        let elem_count: usize = out_dims.iter().product();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, q.dtype())
            .with_label("sdpa_quant_o")
            .build()?;

        // The guards are bound to locals and held for the whole encode: a
        // `storage_and_layout()` temporary dropped at the end of its statement
        // would release the lock while the buffer it yielded is still being
        // bound, which is the shape §6.3b's whole argument is about at the GPU
        // clock and is just as wrong at the host one.
        let (kc_s, _kc_l) = self.k_codes.storage_and_layout();
        let (ks_s, _ks_l) = self.k_scales.storage_and_layout();
        let (kb_s, _kb_l) = self.k_biases.storage_and_layout();
        let (vc_s, _vc_l) = self.v_codes.storage_and_layout();
        let (vs_s, _vs_l) = self.v_scales.storage_and_layout();
        let (vb_s, _vb_l) = self.v_biases.storage_and_layout();

        fn metal_buf<'a>(s: &'a candle::Storage, what: &str) -> Result<&'a candle::MetalStorage> {
            match s {
                candle::Storage::Metal(m) => Ok(m),
                _ => candle::bail!("sdpa_quant expects a metal {what} plane"),
            }
        }
        let kc = metal_buf(&kc_s, "k_codes")?;
        let ks = metal_buf(&ks_s, "k_scales")?;
        let kb = metal_buf(&kb_s, "k_biases")?;
        let vc = metal_buf(&vc_s, "v_codes")?;
        let vs = metal_buf(&vs_s, "v_scales")?;
        let vb = metal_buf(&vb_s, "v_biases")?;

        let sp = candle_metal_kernels::SdpaQuantParams::for_step(
            self.gqa_factor,
            self.n,
            self.capacity,
            q_l.dim(3)?,
            self.scale,
            self.softcapping,
            bits,
        );
        let params_buf = device.new_buffer_with_data(&[sp])?;

        let encoder = device.command_encoder()?;
        encoder.set_label("sdpa_quant");

        candle_metal_kernels::call_sdpa_vector_quant(
            device.device(),
            &encoder,
            device.kernels(),
            q_l.start_offset() * q.dtype().size_in_bytes(),
            q_l.dims(),
            q.buffer(),
            kc.buffer(),
            ks.buffer(),
            kb.buffer(),
            vc.buffer(),
            vs.buffer(),
            vb.buffer(),
            &output,
            &params_buf,
            bits,
            itype,
        )
        .map_err(candle::Error::wrap)?;

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, Shape::from_dims(&out_dims)))
    }
}

/// Attention over a quantized KV cache (issue #218, `DESIGN.md` §10.7).
///
/// `q` is `[B, H_q, 1, D]`; the planes are the caller's, written by
/// [`quantize_kv_into`]. Returns `[B, H_q, 1, D]`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_quant(
    q: &Tensor,
    k_codes: &Tensor,
    k_scales: &Tensor,
    k_biases: &Tensor,
    v_codes: &Tensor,
    v_scales: &Tensor,
    v_biases: &Tensor,
    gqa_factor: usize,
    n: usize,
    capacity: usize,
    scale: f32,
    softcapping: f32,
    quant: KvQuant,
) -> Result<Tensor> {
    q.apply_op1_no_bwd(&SdpaQuant {
        k_codes: k_codes.clone(),
        k_scales: k_scales.clone(),
        k_biases: k_biases.clone(),
        v_codes: v_codes.clone(),
        v_scales: v_scales.clone(),
        v_biases: v_biases.clone(),
        gqa_factor,
        n,
        capacity,
        scale,
        softcapping,
        quant,
    })
}

/// How FlashDecoding sizes the scratch class it allocates (`DESIGN.md` §9.1a,
/// issue #234).
///
/// # Why this enum exists beside `candle_metal_kernels::Sizing` rather than
/// being it
///
/// The policy is `Sizing`, it has been compiled in three arms since #71, and it
/// lives in `candle-metal-kernels` — a **metal-only** crate. `FlashDecoding`'s
/// fields are compiled on every backend, so naming `Sizing` here would put a
/// metal type in a struct CPU and CUDA builds construct, and §14.1's "must not
/// break other backends" is a hard constraint rather than a preference. This is
/// the same shape as the depthwise-conv PR's optional trait method: a
/// backend-neutral declaration, converted at the metal boundary.
///
/// **The conversion is one exhaustive `match`** (`to_metal`, below), so an arm
/// added to `Sizing` and not to this enum is a compile error rather than a
/// silently unreachable policy — which is the failure mode #241 found in
/// `config_line`'s two-armed conditional over a three-armed axis, and #58's
/// mechanism for preventing it.
///
/// The default is `Grow`, and that is a **statement of what shipped rather than
/// a choice**: #116 sized these buffers to the live `kv_len` on every call,
/// which its own comment calls *"`Sizing::Grow` in effect … a consequence of
/// allocating per call and stated rather than chosen"*. Defaulting to anything
/// else here would flip a default by wiring an axis, which §7.1a forbids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlashScratchSizing {
    /// Size to the current `kv_len`, every step. **What #116 does today**, and
    /// therefore the default — see the type-level note.
    #[default]
    Grow,
    /// Reserve for the configured maximum context, once. The region is then a
    /// constant size, which is the only arm whose allocation does not vary with
    /// `kv_len`.
    Reserve,
    /// Reserve the smallest rung of `BUCKET_LADDER` that covers `kv_len`.
    Bucket,
}

impl FlashScratchSizing {
    /// Every arm, for a harness and for exhaustive tests.
    ///
    /// An array rather than a derive so that adding an arm without extending a
    /// consumer is visible, which is `Sizing::ALL`'s own argument (#58,
    /// §11.3i).
    pub const ALL: [FlashScratchSizing; 3] = [
        FlashScratchSizing::Grow,
        FlashScratchSizing::Reserve,
        FlashScratchSizing::Bucket,
    ];

    /// The arm's name, as a `RESULT` line and an error message spell it.
    ///
    /// Rendered from the resolved enum through an exhaustive `match`, which is
    /// #241's rule: a conditional narrower than its axis renders a **wrong**
    /// arm, and a wrong arm compares *equal* to real runs of another
    /// configuration where an absent one would only have been `UNRECORDED`.
    pub fn name(self) -> &'static str {
        match self {
            FlashScratchSizing::Grow => "Grow",
            FlashScratchSizing::Reserve => "Reserve",
            FlashScratchSizing::Bucket => "Bucket",
        }
    }

    /// Parse an arm from a flag value.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "grow" => Ok(FlashScratchSizing::Grow),
            "reserve" => Ok(FlashScratchSizing::Reserve),
            "bucket" => Ok(FlashScratchSizing::Bucket),
            other => {
                candle::bail!("unknown scratch sizing {other:?}; expected grow, reserve or bucket")
            }
        }
    }

    /// The `candle-metal-kernels` policy this names.
    ///
    /// Exhaustive by construction, which is what makes this a translation
    /// rather than a second copy of the policy: the arithmetic stays in
    /// [`candle_metal_kernels::Sizing::sized_chunks`] and both consumers reach
    /// it through here.
    #[cfg(feature = "metal")]
    fn to_metal(self) -> candle_metal_kernels::Sizing {
        match self {
            FlashScratchSizing::Grow => candle_metal_kernels::Sizing::Grow,
            FlashScratchSizing::Reserve => candle_metal_kernels::Sizing::Reserve,
            FlashScratchSizing::Bucket => candle_metal_kernels::Sizing::Bucket,
        }
    }
}

/// FlashDecoding: attention split over independent contiguous KV chunks, with
/// an index-ordered combine (`DESIGN.md` §10.4, issue #116).
///
/// # Why a second op rather than a branch inside [`Sdpa`]
///
/// `Sdpa::metal_fwd` already routes to `call_sdpa_vector_2pass` at
/// `k_seq >= 1024`, so a branch here would read as *"the chunked path, now with
/// a different chunk count"*. It is not: `sdpa_vector_2pass`'s blocks read a
/// **strided interleave** at a **fixed count of 32**, so a block owns no
/// contiguous range and cannot be a page (`DESIGN.md` §10.3b). A separate op
/// keeps the existing routing untouched — the 2pass path is a legitimate
/// long-`kv_len` optimisation and this issue does not disturb it — and makes
/// the arm **selectable** rather than reached by a threshold, which is what
/// lets #61's context curve decide the crossover instead of a constant deciding
/// it now.
#[allow(dead_code)]
struct FlashDecoding {
    scale: f32,
    softcapping: f32,
    /// KV tokens per page — the **allocation** unit.
    page_size: usize,
    /// Pages per chunk: the `k` of `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk to one granularity **by fiat**;
    /// §9.1d establishes the general form and that the two are optimised
    /// against disjoint cost functions. Carried so a sweep can separate a
    /// page-size effect from a tile-size one, which a sweep holding `k = 1`
    /// cannot.
    pages_per_chunk: usize,
    /// How the scratch class is sized (`DESIGN.md` §9.1a, issue #234).
    ///
    /// Three arms have been compiled since #71 and none reached this
    /// allocation, which sized to the live `kv_len` on every call — `Grow` in
    /// effect, by consequence rather than by choice. This field is the wiring
    /// §7.1b names as the cheapest missing prerequisite, and `Grow` is what it
    /// defaults to, so an unconfigured caller allocates byte-for-byte what
    /// shipped.
    scratch_sizing: FlashScratchSizing,
    /// Chunks [`FlashScratchSizing::Reserve`] reserves for.
    ///
    /// Derived by the caller from its configured max context, because the max
    /// context is the *model's* number and the chunk size is this op's:
    /// `ceil(max_context / (page_size * pages_per_chunk))`. Ignored by the
    /// other two arms, and carried rather than recomputed so the op does not
    /// need to know what a context length is.
    reserve_chunks: usize,
}

impl candle::CustomOp3 for FlashDecoding {
    fn name(&self) -> &'static str {
        "metal-flash-decoding"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for metal-flash-decoding")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::{flash_chunk_count, ChunkTable, FlashDType};

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();
        let out_shape = Shape::from_dims(&out_dims);
        let out_layout = Layout::contiguous(out_shape.clone());

        // Every condition below is one this kernel cannot serve, and each fails
        // loudly rather than computing something plausible — §3.5 makes a wrong
        // answer here silent, so the caller's fallback has to be a *decision*
        // and not a discovery.
        if q_l.dim(2)? != 1 {
            candle::bail!("flash-decoding is a decode kernel: q seq must be 1");
        }
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }
        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }
        // The batch row must be exactly `n_kv_heads` head-strides wide, on both
        // K and V.
        //
        // **This is the precondition the flattened grid below rests on**, and it
        // is asserted rather than assumed because §3.5 makes the failure silent:
        // the kernel reaches a batch row only via `kv_head_idx * k_head_stride`
        // with `kv_head_idx = head_idx / gqa_factor`, so row `b`'s KV sits at
        // `b * n_kv_heads * stride[1]`. That is the right address iff
        // `stride[0] == n_kv_heads * stride[1]`.
        //
        // It holds for the shapes this path is given — a contiguous
        // `[B, H_kv, capacity, D]` cache, including under the `narrow(2, 0, len)`
        // the KV slot hands out, since a dim-2 narrow changes the length and no
        // stride (§10.3b: *"`N` is the live length and the stride is the
        // reserved capacity"*). It would **not** hold for a view offset or
        // narrowed on dim 0 or 1, which is precisely what a paged or
        // per-sequence cache could introduce — so this check is what such a
        // change trips on rather than corrupts.
        // `q` is reached as `queries += head_idx * D` with a flat `head_idx`, so
        // the (batch, head) plane must be densely packed. At `q seq == 1` — which
        // the first check above already enforces — contiguity is exactly that.
        if !q_l.is_contiguous() {
            candle::bail!(
                "flash-decoding needs a contiguous `q`: the grid flattens (batch, head) into one \
                 axis and indexes queries by `head_idx * head_dim` (DESIGN.md 3.5)"
            );
        }
        for (name, l) in [("k", k_l), ("v", v_l)] {
            let (st, dims) = (l.stride(), l.dims());
            if st[0] != dims[1] * st[1] {
                candle::bail!(
                    "flash-decoding needs `{name}`'s batch stride to be `n_kv_heads` head-strides \
                     ({} * {} = {}), got {}: the grid flattens (batch, head) into one axis and \
                     reaches a row by head stride alone (DESIGN.md 3.5, 10.3b)",
                    dims[1],
                    st[1],
                    dims[1] * st[1],
                    st[0]
                );
            }
        }
        let itype = match q.dtype() {
            DType::F16 => FlashDType::F16,
            DType::F32 => FlashDType::F32,
            other => candle::bail!("unsupported flash-decoding type {other:?}"),
        };

        // The grid's `y` axis is **(batch x q head) flattened**, not the q head
        // count — which is what `call_sdpa_vector` already does
        // (`sdpa.rs`: `let b = q_shape[0] * q_shape[1]`), and the reason the two
        // must agree is that every offset in `flash_decoding.metal` is computed
        // from a flat `head_idx`: `queries += head_idx * D`,
        // `partials + head_idx * stride_chunks * D`, `out + head_idx * D`.
        //
        // **Passing `q_l.dim(1)` here is what left rows 1..B-1 unwritten**: the
        // output is sized `B*H*1*D` by `out_dims` above while the grid covered
        // only `H`, so at B=2 row 1 read whatever its buffer happened to hold —
        // measured as a `--batch-check` divergence at step 1 emitting a token
        // that varied between runs (127999 at B=2, 20392 at B=4), the signature
        // of unwritten memory rather than of a mis-index.
        //
        // # Why flattening is correct for the KV mapping, and its condition
        //
        // The kernel derives `kv_head_idx = head_idx / gqa_factor`. For
        // `head_idx = b*H + h` that is `b*(H/g) + h/g`, which equals the wanted
        // `b*H_kv + h/g` **iff `H/g == H_kv`** — exactly what the
        // `n_heads % n_kv_heads == 0` guard above already establishes, since
        // `g = H/H_kv`. So the same guard that makes GQA well-formed is what
        // makes the flattening sound; there is no new precondition.
        //
        // `gqa_factor` itself is unaffected: `FlashPartialParams::for_step`
        // derives it from `q_shape[1] / k_shape[1]`, the true head counts.
        let b_sz = q_l.dim(0)?;
        let n_q_heads = b_sz * q_l.dim(1)?;
        let head_dim = q_l.dim(D::Minus1)?;
        let n_keys = k_l.dim(2)?;
        let chunk_size = self.page_size * self.pages_per_chunk;
        let n_chunks = flash_chunk_count(n_keys, chunk_size);
        if n_chunks == 0 {
            candle::bail!("flash-decoding needs at least one key");
        }

        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, q.dtype())
            .with_label("flash_decoding_o")
            .build()?;

        // The scratch class, §9.1a — **and it now reads the policy** (#234).
        //
        // # What the mismatch was, and what resolving it needed
        //
        // The comment that stood here said the class was sized *"here rather
        // than through `ScratchPlan` because that plan owns the whole step's
        // regions and this op sees one layer"*, and named wiring the planned
        // arena as what a `ScratchSizing` decision would need. **The premise is
        // right and the conclusion does not follow.** A plan carries two things
        // — how many chunks a region reserves, and where each layer's region
        // sits in one arena — and only the first is a *policy*. The second is
        // placement, which an op allocating one region per call does not have
        // and does not need. `Sizing::sized_chunks` is the first alone, so it
        // is reachable from here; `plan_scratch` calls the same function, so
        // the two cannot disagree about what an arm means.
        //
        // So this is no longer `Grow` **in effect**: it is whichever arm the
        // caller selected, and `Grow` when nobody selects — which is what
        // shipped, byte for byte.
        let sizing = self.scratch_sizing.to_metal();
        let rungs: Vec<usize> = candle_metal_kernels::BUCKET_LADDER
            .iter()
            .map(|&kv| flash_chunk_count(kv, chunk_size))
            .collect();
        let chunk_capacity = sizing
            .sized_chunks(n_chunks, self.reserve_chunks, &rungs)
            .map_err(candle::Error::msg)?;

        // Sized for the **capacity** and written for the **live** count. Both
        // numbers reach the kernel: `chunk_capacity` is the stride the partial
        // pass lays the region out at, and `n_chunks` is the dispatch depth and
        // the combine's loop bound. Merging over the reservation would fold in
        // uninitialised memory, which §10.4 and §10.3d both name as a silent
        // wrong answer no size check catches.
        //
        // `walk_order` is sized for the live count and not the capacity,
        // deliberately: it records the chunks the combine **walked**, and the
        // combine walks `n_chunks` of them. Sizing it larger would leave the
        // tail unwritten and give the merge-order assertion a region it cannot
        // interpret.
        let partials = device
            .new_buffer_builder()
            .with_size_for(n_q_heads * chunk_capacity * head_dim, DType::F32)
            .with_label("flash_decoding_partials")
            .build()?;
        let sums = device
            .new_buffer_builder()
            .with_size_for(n_q_heads * chunk_capacity, DType::F32)
            .with_label("flash_decoding_sums")
            .build()?;
        let maxs = device
            .new_buffer_builder()
            .with_size_for(n_q_heads * chunk_capacity, DType::F32)
            .with_label("flash_decoding_maxs")
            .build()?;
        let walk_order = device
            .new_buffer_builder()
            .with_size_for(n_q_heads * n_chunks, DType::U32)
            .with_label("flash_decoding_walk_order")
            .build()?;

        // At B=1 over a contiguous cache the table is `c * pages_per_chunk`.
        // **It is built and bound even though it is derivable**, because a
        // combine that computes the range instead of reading it is a *walk*,
        // and §10.3h records that a B=1 gate structurally cannot tell a walk
        // from an index — at B=1 the two coincide. Binding it now is what makes
        // the permuted-table test possible and what a paged cache would fill
        // differently.
        let table = ChunkTable::Contiguous.entries(n_chunks, self.pages_per_chunk);
        let chunk_table = device.new_buffer_with_data(&table)?;

        let partial_params = device.new_buffer_with_data(std::slice::from_ref(
            &candle_metal_kernels::FlashPartialParams::for_step(
                q_l.dims(),
                k_l.dims(),
                k_l.stride(),
                v_l.stride(),
                self.page_size,
                self.pages_per_chunk,
                n_chunks,
                chunk_capacity,
                self.scale,
                self.softcapping,
            )
            .map_err(candle::Error::wrap)?,
        ))?;
        let combine_params = device.new_buffer_with_data(std::slice::from_ref(
            // The pair this call has taken since #116, and the reason the
            // partial pass needed the same one: the combine reads at the
            // capacity stride and merges to the live count. Both sides of the
            // region now agree about its shape.
            &candle_metal_kernels::FlashCombineParams::for_step(n_chunks, chunk_capacity),
        ))?;

        let encoder = q.device().command_encoder()?;
        encoder.set_label("flash_decoding");
        candle_metal_kernels::call_flash_decoding(
            q.device().device(),
            &encoder,
            q.device().kernels(),
            q_l.start_offset() * q.dtype().size_in_bytes(),
            q.buffer(),
            k_l.start_offset() * k.dtype().size_in_bytes(),
            k.buffer(),
            v_l.start_offset() * v.dtype().size_in_bytes(),
            v.buffer(),
            &output,
            &partials,
            &sums,
            &maxs,
            &chunk_table,
            &partial_params,
            &combine_params,
            &walk_order,
            n_q_heads,
            head_dim,
            n_chunks,
            itype,
        )
        .map_err(candle::Error::wrap)?;

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, out_layout.shape().clone()))
    }
}

/// FlashDecoding attention over a contiguous KV cache.
///
/// `q` is `[b, n_q_heads, 1, head_dim]` — a decode step. `k`/`v` are
/// `[b, n_kv_heads, kv_len, head_dim]`, and may be a `narrow` of a
/// pre-allocated cache: the kernel takes the head stride and the token stride
/// as parameters, so the reserved capacity and the live length are separate
/// numbers (`DESIGN.md` §10.3b).
///
/// `page_size` is the **allocation** granularity and `pages_per_chunk` is the
/// `k` of `chunk_size = k * page_size` — the **computation** granularity. §10.4
/// fixes them equal by fiat and §9.1d records that they are optimised against
/// disjoint cost functions; both are parameters here so a sweep can separate
/// them.
///
/// `scratch_sizing` is `DESIGN.md` §9.1a's policy, wired to this allocation by
/// #234 — three arms compiled since #71 and, until now, none of them reachable
/// from a model path. `reserve_chunks` is what
/// [`FlashScratchSizing::Reserve`] reserves for, in **chunks**: the caller
/// converts its max context because a context length is the model's number and
/// a chunk size is this op's. It is ignored by the other two arms.
///
/// Metal only, `seq == 1` only. Every other case is an error rather than a
/// fallback: the caller decides, because a silent fallback is what makes an
/// A/B arm measure the wrong thing (§2.4).
#[allow(clippy::too_many_arguments)]
pub fn flash_decoding(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    softcapping: f32,
    page_size: usize,
    pages_per_chunk: usize,
    scratch_sizing: FlashScratchSizing,
    reserve_chunks: usize,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &FlashDecoding {
            scale,
            softcapping,
            page_size,
            pages_per_chunk,
            scratch_sizing,
            reserve_chunks,
        },
    )
}

/// Which transform a fused GEMV epilogue applies at the accumulator (issue
/// #509 wiring; the kernels are #484's and #509's).
///
/// **This is the wiring, not the kernel.** `GemvEpilogue` has existed in
/// `candle-metal-kernels` since #484 (`Silu`) and #509 (`Residual`), measured
/// and correct, with **zero production callers** — `Tensor::matmul` reaches
/// `call_mlx_gemv` through a wrapper that pins the epilogue to `None`
/// (`mlx_gemm.rs:667`), so no model could select one. `DESIGN.md` §12.2h
/// records that state and names the wiring as what it waits on.
///
/// Both arms **move the digests by construction** and owe §2.3.5a's
/// discriminator rather than an unchanged-digest gate: the fused kernel
/// applies the transform to the **f32 accumulator** where the dispatch pair it
/// replaces applies it at `T` to the **already-rounded store**. Strictly fewer
/// roundings, and a different computation. See `DESIGN.md` §12.2d and §12.2h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GemvFusion {
    /// No fusion. The caller takes the eager path it always took.
    #[default]
    Off,
    /// SiLU on the accumulator, replacing a `silu_f16` dispatch (#484).
    Silu,
    /// Residual add on the accumulator, replacing a `badd_f16` dispatch
    /// (#509). Reads the residual at the storing thread's own output row.
    Residual,
}

/// A GEMV with an elementwise transform fused onto its store.
///
/// `x` is `[.., 1, k]` and `weight` is `[n, k]` — the **untransposed**
/// `candle_nn::Linear` layout, transposed here exactly as `Linear::forward`
/// does it, so the unfused arm is the same `[[host_name]]` on the same
/// operands. `residual` is `[.., 1, n]` and is bound only by
/// [`GemvFusion::Residual`].
///
/// Metal only, `m == 1` only, no bias. Every other case is an **error rather
/// than a fallback**: the caller decides, because a silent fallback is what
/// makes an A/B arm measure the wrong thing (`DESIGN.md` §2.4, and the same
/// rule `flash_decoding` above is written to).
///
/// `dead_code` because both members are read only from `metal_fwd`, which is
/// `#[cfg(feature = "metal")]`. The same shape as `Sdpa` above; `cfg`-gating
/// the struct instead would take the `cpu_fwd` that reports *no cpu support*
/// away with it, which is the message a non-Metal caller should get.
#[allow(dead_code)]
struct GemvFused {
    fusion: GemvFusion,
}

#[allow(dead_code)]
impl GemvFused {
    /// The shape checks both arities share, run before either dispatches.
    ///
    /// Returns `(b, n, k)` with `m` verified to be 1. A GEMV is the only route
    /// that carries a fused epilogue — `takes_gemv_route` is `m == 1 || n == 1`
    /// (`mlx_gemm.rs:246`) and the instantiated tiles are the non-transposed
    /// `bm4`/`bm8` decode tiles — so an `m != 1` request would silently take
    /// the GEMM path and run unfused. Refused here instead.
    fn checked_dims(x_l: &Layout, w_l: &Layout) -> Result<(usize, usize, usize)> {
        let (b, m, k) = match *x_l.dims() {
            [b, m, k] => (b, m, k),
            [m, k] => (1, m, k),
            ref other => {
                candle::bail!("gemv-fused expects x rank 2 or 3, got {other:?} (issue #509)")
            }
        };
        if m != 1 {
            candle::bail!(
                "gemv-fused is a decode-path fusion: x seq must be 1, got {m} \
                 (issue #509)"
            );
        }
        let (n, w_k) = match *w_l.dims() {
            [n, w_k] => (n, w_k),
            ref other => {
                candle::bail!("gemv-fused expects weight rank 2 [n, k], got {other:?} (issue #509)")
            }
        };
        if w_k != k {
            candle::bail!("gemv-fused: x k={k} does not match weight k={w_k} (issue #509)");
        }
        Ok((b, n, k))
    }
}

#[cfg(feature = "metal")]
fn gemv_fused_metal(
    fusion: GemvFusion,
    x: &candle::MetalStorage,
    x_l: &Layout,
    w: &candle::MetalStorage,
    w_l: &Layout,
    residual: Option<(&candle::MetalStorage, &Layout)>,
) -> Result<(candle::MetalStorage, Shape)> {
    use candle::backend::BackendStorage;
    use candle_metal_kernels::{GemmDType, GemvEpilogue};

    let (b, n, k) = GemvFused::checked_dims(x_l, w_l)?;
    let device = x.device();

    if x.dtype() != w.dtype() {
        candle::bail!(
            "gemv-fused: x dtype {:?} and weight dtype {:?} must match (issue #509)",
            x.dtype(),
            w.dtype()
        );
    }
    let dtype = match x.dtype() {
        DType::F32 => GemmDType::F32,
        DType::F16 => GemmDType::F16,
        DType::BF16 => GemmDType::BF16,
        other => candle::bail!("gemv-fused does not support {other:?} (issue #509)"),
    };

    let epilogue = match fusion {
        GemvFusion::Off => GemvEpilogue::None,
        GemvFusion::Silu => GemvEpilogue::Silu,
        GemvFusion::Residual => GemvEpilogue::Residual,
    };

    // The residual is bound at the storing thread's own output position, so it
    // must be exactly the output extent and contiguous. Checked rather than
    // trusted: an offset view would be read at the wrong index and produce a
    // well-formed wrong number, which under `HazardTrackingModeUntracked`
    // (§3.5) nothing reports.
    let operand = match residual {
        None => None,
        Some((r, r_l)) => {
            if r.dtype() != x.dtype() {
                candle::bail!(
                    "gemv-fused: residual dtype {:?} must match x dtype {:?} (issue #509)",
                    r.dtype(),
                    x.dtype()
                );
            }
            if r_l.shape().elem_count() != b * n {
                candle::bail!(
                    "gemv-fused: residual has {} elements, expected b*n = {} (issue #509)",
                    r_l.shape().elem_count(),
                    b * n
                );
            }
            if !r_l.is_contiguous() {
                candle::bail!("gemv-fused: residual must be contiguous (issue #509)");
            }
            Some((r.buffer(), r_l.start_offset() * r.dtype().size_in_bytes()))
        }
    };

    let out_shape = match *x_l.dims() {
        [b, m, _] => Shape::from_dims(&[b, m, n]),
        _ => Shape::from_dims(&[1, n]),
    };
    let elem_count = b * n;
    let output = device
        .new_buffer_builder()
        .with_size_for(elem_count, x.dtype())
        .with_label("gemv_fused")
        .build()?;

    // `Linear::forward` computes `x @ w.t()`, so the strides handed to the
    // kernel are the transposed weight's. Built here rather than by calling
    // `Tensor::t()` because this op has storages, not tensors — and it is the
    // same `(lhs, rhs)` order `call_mlx_gemm` would have seen.
    let x_stride = x_l.stride().to_vec();
    let w_t_stride = vec![1usize, k];

    let encoder = device.command_encoder()?;
    encoder.set_label("gemv_fused");
    candle_metal_kernels::call_mlx_gemv_epilogue(
        device.device(),
        &encoder,
        device.kernels(),
        dtype,
        (b, 1, n, k),
        &x_stride,
        x_l.start_offset() * x.dtype().size_in_bytes(),
        x.buffer(),
        &w_t_stride,
        w_l.start_offset() * w.dtype().size_in_bytes(),
        w.buffer(),
        &output,
        candle_metal_kernels::ParamStyle::Split,
        None,
        epilogue,
        operand,
    )
    .map_err(candle::Error::wrap)?;

    let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, x.dtype());
    Ok((newstorage, out_shape))
}

impl candle::CustomOp2 for GemvFused {
    fn name(&self) -> &'static str {
        "metal-gemv-fused"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for metal-gemv-fused (issue #509)")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle::MetalStorage,
        x_l: &Layout,
        w: &candle::MetalStorage,
        w_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        gemv_fused_metal(self.fusion, x, x_l, w, w_l, None)
    }
}

impl candle::CustomOp3 for GemvFused {
    fn name(&self) -> &'static str {
        "metal-gemv-fused-residual"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for metal-gemv-fused-residual (issue #509)")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x: &candle::MetalStorage,
        x_l: &Layout,
        w: &candle::MetalStorage,
        w_l: &Layout,
        r: &candle::MetalStorage,
        r_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        gemv_fused_metal(self.fusion, x, x_l, w, w_l, Some((r, r_l)))
    }
}

/// `silu(x @ weight.t())` in one dispatch — #484's epilogue, on the decode path.
///
/// The consuming site is `lfm2.rs`'s `silu(gate_proj.forward(x))`: a GEMV
/// immediately followed by a `silu_f16`, which is exactly the pair #484
/// measured winning at 7 of 7 sizes. `weight` is `[n, k]`, the `Linear`
/// layout.
///
/// **Moves the digests**: SiLU is computed at the f32 accumulator where the
/// pair computes it at `T` on the rounded store (`DESIGN.md` §12.2d, §2.3.5a).
///
/// Metal, `m == 1`, no bias. Anything else is an error — see [`GemvFused`].
pub fn gemv_silu(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    x.apply_op2_no_bwd(
        weight,
        &GemvFused {
            fusion: GemvFusion::Silu,
        },
    )
}

/// `(x @ weight.t()) + residual` in one dispatch — #509's epilogue, on the
/// decode path.
///
/// The producing sites are `lfm2.rs`'s two residual adds: `out_proj`,
/// `conv.out_proj` and `w2`, 60 dispatches per token (`DESIGN.md` §12.2e).
/// `weight` is `[n, k]`; `residual` is the layer input, `[.., 1, n]`.
///
/// **No compile flag is required, since #516.** The `_residual` names are
/// instantiated unconditionally, as `gemv_silu`'s are; this used to need
/// `-DLLOOM_509_RESIDUAL_EPILOGUE` on the metallib and failed at pipeline load
/// without it.
///
/// **Moves the digests**: the add happens at the f32 accumulator where the
/// `badd_f16` it replaces adds at `T` on the rounded store (§12.2h, §2.3.5a).
pub fn gemv_residual(x: &Tensor, weight: &Tensor, residual: &Tensor) -> Result<Tensor> {
    x.apply_op3_no_bwd(
        weight,
        residual,
        &GemvFused {
            fusion: GemvFusion::Residual,
        },
    )
}
