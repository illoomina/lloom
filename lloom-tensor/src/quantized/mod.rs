use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::{borrow::Cow, sync::OnceLock};

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod fp_quants;
pub mod gemv_mixed;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod iq_grids;
pub mod iq_quants;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

fn as_t_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
    /// Lazily initialized storage for repacked quantized data. Currently raw bits, could be `QStorage` in the future.
    /// Not always used.
    #[allow(dead_code)]
    repacked_qs: OnceLock<Option<Vec<u8>>>,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        let data: &[u8] = &data;
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(Cow::Borrowed(data)))),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(data)),
                // IQ / Q2_0: raw blocks uploaded as-is; the Metal `mul_mv_iq*`
                // kernels dequantize on the fly (dequant-in-kernel, no separate
                // dequant pass). `load_quantized` is generic over the block type.
                GgmlDType::IQ2XXS => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ2xxs>(data))
                }
                GgmlDType::IQ2XS => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ2xs>(data))
                }
                GgmlDType::IQ2S => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ2s>(data))
                }
                GgmlDType::IQ3XXS => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ3xxs>(data))
                }
                GgmlDType::IQ3S => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ3s>(data))
                }
                GgmlDType::IQ4NL => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ4nl>(data))
                }
                GgmlDType::IQ4XS => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockIQ4xs>(data))
                }
                GgmlDType::Q2_0 => {
                    metal::load_quantized(d, as_t_slice::<iq_quants::BlockQ2_0>(data))
                }
                GgmlDType::MXFP4 => {
                    metal::load_quantized(d, as_t_slice::<fp_quants::BlockMxfp4>(data))
                }
                GgmlDType::FP8E4M3 => {
                    metal::load_quantized(d, as_t_slice::<fp_quants::BlockFp8E4M3>(data))
                }
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),
                // IQ / Q2_0 are not wired for the CUDA matmul path in lloom; the
                // GSQ-RCO checkpoints these serve run on Metal / CPU. Refuse
                // rather than upload blocks no CUDA kernel can read.
                GgmlDType::IQ2XXS
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ3S
                | GgmlDType::IQ4NL
                | GgmlDType::IQ4XS
                | GgmlDType::Q2_0
                | GgmlDType::MXFP4
                | GgmlDType::FP8E4M3 => {
                    crate::bail!("{dtype:?} is not supported on the CUDA quantized path")
                }
            },
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?)?;
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row)?;
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?)?;
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row)?;
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    /// Overwrite `len` packed bytes at `offset` with whatever `fill` writes
    /// into the slice it is handed — a streamed weight tier landing one
    /// expert's blocks in a resident slot (`nn::expert_stream`). On the CPU
    /// the slice is the storage itself; on Metal it is a pooled Shared staging
    /// buffer copied into the storage by a compute-encoder copy, so the write
    /// is ordered by the encoder's hazard chain after every earlier reader of
    /// the slot (a blit would sit outside that chain, `MetalStorage::copy2d`).
    fn write_with(
        &mut self,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        match self {
            QStorage::Cpu(storage) => {
                let size = storage.storage_size_in_bytes();
                if offset + len > size {
                    crate::bail!(
                        "write_with: {offset}+{len} past the {size}-byte quantized storage"
                    )
                }
                // SAFETY: `as_mut_ptr` addresses `size` initialised bytes of
                // plain-old-data blocks and the range was bounds-checked; the
                // `&mut self` borrow excludes every other access.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(storage.as_mut_ptr().add(offset), len)
                };
                fill(dst)
            }
            QStorage::Metal(storage) => storage.write_with(offset, len, fill),
            QStorage::Cuda(_) => crate::bail!("write_with is not implemented on CUDA"),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr_with_guard(stream),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // IQ-family and the GSQ-RCO fork's Q2_0 (type code 42). Read-only: lloom
    // dequantizes these (CPU `to_float` in `iq_quants`, Metal `mul_mv_iq*`), it
    // does not quantize to them. Codes per ggml.h; Q2_0 per the qwen4exp fork.
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4NL,
    IQ4XS,
    Q2_0,
    // ggml's MXFP4 (type 39): 32 E2M1 codes under one E8M0 scale.
    MXFP4,
    // 32 E4M3FN codes under one bf16 scale. No ggml type exists; its GGUF
    // id is lloom's own (`GGUF_TYPE_FP8_E4M3`).
    FP8E4M3,
}

/// The GGUF tensor type id of [`GgmlDType::FP8E4M3`]. ggml has no FP8 type,
/// so the id is lloom's: upstream assigns ids densely from 0 (43 so far) and
/// refuses one at or past its count, so an id at `1 << 24` can neither
/// collide with a future upstream type nor be misread by another reader.
pub const GGUF_TYPE_FP8_E4M3: u32 = 1 << 24;

impl GgmlDType {
    /// Reject activation-only encodings before allocating a file payload.
    /// This says nothing about a backend's matrix-multiplication support.
    pub(crate) fn validate_file_weight(self) -> Result<()> {
        match self {
            Self::Q8_1 | Self::Q8K => crate::bail!("quantized type {self:?} is not supported yet"),
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::IQ3S
            | Self::IQ4NL
            | Self::IQ4XS
            | Self::Q2_0
            | Self::MXFP4
            | Self::FP8E4M3 => Ok(()),
        }
    }

    /// Whether lloom only reads blocks of this type. For the IQ family and
    /// Q2_0 only the dequant is ported: there is no quantizer to produce them
    /// and no CPU dot product to multiply by them, so both are refused by name
    /// (§15.2a); their matmul runs on Metal.
    pub fn is_read_only(self) -> bool {
        matches!(
            self,
            Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ2S
                | Self::IQ3XXS
                | Self::IQ3S
                | Self::IQ4NL
                | Self::IQ4XS
                | Self::Q2_0
        )
    }

    fn refuse_read_only(self, op: &str) -> Result<()> {
        if self.is_read_only() {
            crate::bail!(
                "{op} {self:?} is refused: lloom reads {self:?} blocks but ports only \
                 their dequant, not a quantizer or a CPU dot product"
            )
        }
        Ok(())
    }

    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            // Q2_0: the qwen4exp / GSQ-RCO fork's own type (GGML_TYPE_Q2_0=42).
            42 => Self::Q2_0,
            39 => Self::MXFP4,
            GGUF_TYPE_FP8_E4M3 => Self::FP8E4M3,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
            Self::Q2_0 => 42,
            Self::MXFP4 => 39,
            Self::FP8E4M3 => GGUF_TYPE_FP8_E4M3,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::IQ2XXS => Box::new(vec![
                iq_quants::BlockIQ2xxs::zeros();
                elem_count / iq_quants::BlockIQ2xxs::BLCK_SIZE
            ]),
            Self::IQ2XS => Box::new(vec![
                iq_quants::BlockIQ2xs::zeros();
                elem_count / iq_quants::BlockIQ2xs::BLCK_SIZE
            ]),
            Self::IQ2S => Box::new(vec![
                iq_quants::BlockIQ2s::zeros();
                elem_count / iq_quants::BlockIQ2s::BLCK_SIZE
            ]),
            Self::IQ3XXS => Box::new(vec![
                iq_quants::BlockIQ3xxs::zeros();
                elem_count / iq_quants::BlockIQ3xxs::BLCK_SIZE
            ]),
            Self::IQ3S => Box::new(vec![
                iq_quants::BlockIQ3s::zeros();
                elem_count / iq_quants::BlockIQ3s::BLCK_SIZE
            ]),
            Self::IQ4NL => Box::new(vec![
                iq_quants::BlockIQ4nl::zeros();
                elem_count / iq_quants::BlockIQ4nl::BLCK_SIZE
            ]),
            Self::IQ4XS => Box::new(vec![
                iq_quants::BlockIQ4xs::zeros();
                elem_count / iq_quants::BlockIQ4xs::BLCK_SIZE
            ]),
            Self::Q2_0 => Box::new(vec![
                iq_quants::BlockQ2_0::zeros();
                elem_count / iq_quants::BlockQ2_0::BLCK_SIZE
            ]),
            Self::MXFP4 => Box::new(vec![
                fp_quants::BlockMxfp4::zeros();
                elem_count / fp_quants::BlockMxfp4::BLCK_SIZE
            ]),
            Self::FP8E4M3 => Box::new(vec![
                fp_quants::BlockFp8E4M3::zeros();
                elem_count / fp_quants::BlockFp8E4M3::BLCK_SIZE
            ]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        let data: &[u8] = &data;
        match self {
            Self::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
            Self::IQ2XXS => Box::new(as_t_slice::<iq_quants::BlockIQ2xxs>(data).to_vec()),
            Self::IQ2XS => Box::new(as_t_slice::<iq_quants::BlockIQ2xs>(data).to_vec()),
            Self::IQ2S => Box::new(as_t_slice::<iq_quants::BlockIQ2s>(data).to_vec()),
            Self::IQ3XXS => Box::new(as_t_slice::<iq_quants::BlockIQ3xxs>(data).to_vec()),
            Self::IQ3S => Box::new(as_t_slice::<iq_quants::BlockIQ3s>(data).to_vec()),
            Self::IQ4NL => Box::new(as_t_slice::<iq_quants::BlockIQ4nl>(data).to_vec()),
            Self::IQ4XS => Box::new(as_t_slice::<iq_quants::BlockIQ4xs>(data).to_vec()),
            Self::Q2_0 => Box::new(as_t_slice::<iq_quants::BlockQ2_0>(data).to_vec()),
            Self::MXFP4 => Box::new(as_t_slice::<fp_quants::BlockMxfp4>(data).to_vec()),
            Self::FP8E4M3 => Box::new(as_t_slice::<fp_quants::BlockFp8E4M3>(data).to_vec()),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
            Self::IQ2XXS => std::mem::size_of::<iq_quants::BlockIQ2xxs>(),
            Self::IQ2XS => std::mem::size_of::<iq_quants::BlockIQ2xs>(),
            Self::IQ2S => std::mem::size_of::<iq_quants::BlockIQ2s>(),
            Self::IQ3XXS => std::mem::size_of::<iq_quants::BlockIQ3xxs>(),
            Self::IQ3S => std::mem::size_of::<iq_quants::BlockIQ3s>(),
            Self::IQ4NL => std::mem::size_of::<iq_quants::BlockIQ4nl>(),
            Self::IQ4XS => std::mem::size_of::<iq_quants::BlockIQ4xs>(),
            Self::Q2_0 => std::mem::size_of::<iq_quants::BlockQ2_0>(),
            Self::MXFP4 => std::mem::size_of::<fp_quants::BlockMxfp4>(),
            Self::FP8E4M3 => std::mem::size_of::<fp_quants::BlockFp8E4M3>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
            // IQ super-block types use QK_K=256; IQ4_NL and Q2_0 are small blocks.
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S | Self::IQ3XXS | Self::IQ3S | Self::IQ4XS => {
                k_quants::QK_K
            }
            Self::IQ4NL => iq_quants::QK4_NL,
            Self::Q2_0 => iq_quants::QK2_0,
            Self::MXFP4 => fp_quants::QK_MXFP4,
            Self::FP8E4M3 => fp_quants::QK_FP8,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage>;
    /// Routed matmul over stacked experts (see `QTensor::indexed_moe_forward`):
    /// `x` is `[batch, in_dim1, k]` flattened, `ids` `[batch, topk]` flattened,
    /// `dst` `[batch, topk, n]` flattened.
    #[allow(clippy::too_many_arguments)]
    fn matmul_t_indexed(
        &self,
        enk: (usize, usize, usize),
        x: &[f32],
        in_dim1: usize,
        ids: &[u32],
        bt: (usize, usize),
        dst: &mut [f32],
    ) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    /// The packed bytes, writable: the seam a streamed weight tier fills a
    /// resident slot through (`QTensor::write_with`).
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]) -> Result<()>;
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(
        &mut self,
        xs: &[f32],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()>;
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        T::DTYPE.refuse_read_only("the CPU matmul of")?;
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        T::DTYPE.refuse_read_only("the CPU matmul of")?;
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn matmul_t_indexed(
        &self,
        (n_experts, n, k): (usize, usize, usize),
        x: &[f32],
        in_dim1: usize,
        ids: &[u32],
        (batch, topk): (usize, usize),
        dst: &mut [f32],
    ) -> Result<()> {
        if !k.is_multiple_of(T::BLCK_SIZE) {
            crate::bail!(
                "routed matmul k {k} is not divisible by block size {}",
                T::BLCK_SIZE
            )
        }
        let row_blocks = k / T::BLCK_SIZE;
        if self.len() != n_experts * n * row_blocks {
            crate::bail!(
                "quantized tensor has {} blocks, expected {}",
                self.len(),
                n_experts * n * row_blocks
            )
        }
        // Dequantize one weight row at a time and dot it against the activation:
        // the reference arm, correct for every block type (`to_float` exists for
        // the read-only quants where `vec_dot` does not).
        let mut row = vec![0f32; k];
        for b in 0..batch {
            for t in 0..topk {
                let e = ids[b * topk + t] as usize;
                if e >= n_experts {
                    crate::bail!("expert id {e} is out of range for {n_experts} experts")
                }
                let xi = if in_dim1 == 1 { 0 } else { t };
                let xrow = &x[(b * in_dim1 + xi) * k..(b * in_dim1 + xi + 1) * k];
                let out = &mut dst[(b * topk + t) * n..(b * topk + t + 1) * n];
                for (r, o) in out.iter_mut().enumerate() {
                    let blocks = &self[(e * n + r) * row_blocks..(e * n + r + 1) * row_blocks];
                    T::to_float(blocks, &mut row);
                    *o = row.iter().zip(xrow).map(|(w, x)| w * x).sum();
                }
            }
        }
        Ok(())
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(T::BLCK_SIZE) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                T::BLCK_SIZE
            )
        }
        let row_blocks = hidden / T::BLCK_SIZE;
        if self.len() != rows * row_blocks {
            crate::bail!(
                "quantized tensor has {} blocks, expected {}",
                self.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                crate::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &self[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            T::to_float(src, dst);
        }
        Ok(CpuStorage::F32(out))
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        T::DTYPE.refuse_read_only("quantizing to")?;
        T::from_float(xs, self);
        Ok(())
    }

    fn from_float_imatrix(
        &mut self,
        xs: &[f32],
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        T::DTYPE.refuse_read_only("quantizing to")?;
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row);
        Ok(())
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr() as *mut u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self {
            storage,
            shape,
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, hidden) = self.shape.dims2()?;
        if !hidden.is_multiple_of(self.dtype().block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype().block_size()
            )
        }
        let mut out_shape = ids.dims().to_vec();
        out_shape.push(hidden);
        let device = self.device();
        let ids = ids
            .to_device(&device)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .contiguous()?;
        let storage = match &self.storage {
            QStorage::Cpu(storage) => {
                let ids = ids.to_vec1::<u32>()?;
                Storage::Cpu(storage.embedding(&ids, rows, hidden)?)
            }
            QStorage::Metal(storage) => match &*ids.storage() {
                Storage::Metal(ids_storage) => {
                    Storage::Metal(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
            QStorage::Cuda(storage) => match &*ids.storage() {
                Storage::Cuda(ids_storage) => {
                    Storage::Cuda(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
        };
        let none = crate::op::BackpropOp::none();
        Ok(crate::tensor::from_storage(storage, out_shape, none, false))
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    /// An all-zero packed tensor of `dtype` on `device`, to be filled in
    /// place with [`Self::write_with`] — the resident slot pool of a streamed
    /// weight tier.
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: GgmlDType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, dtype.block_size())?;
        let storage = device.qzeros(shape.elem_count(), dtype)?;
        Self::new(storage, shape)
    }

    /// A tensor of `shape` and `dtype` over ONE block of storage: what a
    /// model's structure says about a weight (its kind and extents) with
    /// none of its bytes — `lloom::testing::ghost`, a model built to be
    /// walked. No kernel may read it: a dispatch would read past its block.
    #[doc(hidden)]
    pub fn ghost<S: Into<Shape>>(shape: S, dtype: GgmlDType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, dtype.block_size())?;
        let storage = device.qzeros(dtype.block_size(), dtype)?;
        Self::new(storage, shape)
    }

    /// [`Self::zeros`] in storage that is not a pool bucket — a persistent
    /// Private allocation on Metal (`QMetalStorage::persistent`), the CPU
    /// storage as it is. The slot pool of a streamed weight tier.
    pub fn zeros_persistent<S: Into<Shape>>(
        shape: S,
        dtype: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        Self::zeros_persistent_with(shape, dtype, device, false)
    }

    /// [`Self::zeros_persistent`], Shared storage on Metal when `shared` (the
    /// CPU writes it in place); the same as `zeros_persistent` elsewhere.
    pub fn zeros_persistent_with<S: Into<Shape>>(
        shape: S,
        dtype: GgmlDType,
        device: &Device,
        shared: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, dtype.block_size())?;
        let storage = match device {
            Device::Metal(d) => QStorage::Metal(metal::QMetalStorage::persistent_with(
                d,
                shape.elem_count(),
                dtype,
                shared,
            )?),
            other => other.qzeros(shape.elem_count(), dtype)?,
        };
        Self::new(storage, shape)
    }

    /// [`Self::write_with`] staged through a [`StagingRing`] on Metal (a
    /// reused buffer instead of an allocation per call); the other backends
    /// take the plain path.
    #[cfg(feature = "metal")]
    pub fn write_with_ring(
        &mut self,
        offset: usize,
        len: usize,
        ring: &mut crate::kernels::metal::staging::StagingRing,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let exact = self.shape.elem_count() / self.dtype().block_size() * self.dtype().type_size();
        if offset + len > exact {
            crate::bail!("write_with_ring: {offset}+{len} past the {exact}-byte tensor")
        }
        match &mut self.storage {
            QStorage::Metal(s) => s.write_with_ring(offset, len, ring, fill),
            _ => self.storage.write_with(offset, len, fill),
        }
    }

    /// [`Self::write_with_ring`] for bytes a prefetch thread already put in
    /// `staging` (`QMetalStorage::write_staged`): the copy alone. Metal only.
    #[cfg(feature = "metal")]
    pub fn write_staged(
        &mut self,
        offset: usize,
        len: usize,
        staging: crate::kernels::metal::BufferHandle,
        ring: &mut crate::kernels::metal::staging::StagingRing,
        entry: Option<usize>,
    ) -> Result<()> {
        let exact = self.shape.elem_count() / self.dtype().block_size() * self.dtype().type_size();
        if offset + len > exact {
            crate::bail!("write_staged: {offset}+{len} past the {exact}-byte tensor")
        }
        match &mut self.storage {
            QStorage::Metal(s) => s.write_staged(offset, len, staging, ring, entry),
            _ => crate::bail!("write_staged: not a Metal quantized tensor"),
        }
    }

    /// Overwrite `len` packed bytes at byte `offset` with what `fill` writes
    /// into the slice it is handed. Bounds-checked against the exact packed
    /// size; the Metal arm goes through the compute encoder so the write is
    /// ordered after every earlier reader of those bytes (`QStorage::write_with`).
    pub fn write_with(
        &mut self,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let exact = self.shape.elem_count() / self.dtype().block_size() * self.dtype().type_size();
        if offset + len > exact {
            crate::bail!("write_with: {offset}+{len} past the {exact}-byte tensor")
        }
        self.storage.write_with(offset, len, fill)
    }

    /// The Metal buffer behind this tensor, for a subsystem that writes it
    /// through another queue (`kernels::metal::io_queue`).
    #[cfg(feature = "metal")]
    pub fn metal_buffer(&self) -> Result<&crate::kernels::metal::Buffer> {
        match &self.storage {
            QStorage::Metal(s) => Ok(s.buffer()),
            _ => crate::bail!("metal_buffer: not a Metal quantized tensor"),
        }
    }

    /// The packed bytes, exactly `elem_count / block_size * type_size` of
    /// them: a pool-backed storage (Metal) hands back its whole bucket, which
    /// is rounded up past the tensor (§11.5n), and a byte view that carries the
    /// padding is not a multiple of the block size for most types.
    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        let exact = self.shape.elem_count() / self.dtype().block_size() * self.dtype().type_size();
        Ok(match self.storage.data()? {
            Cow::Borrowed(b) if b.len() > exact => Cow::Borrowed(&b[..exact]),
            Cow::Owned(mut v) if v.len() > exact => {
                v.truncate(exact);
                Cow::Owned(v)
            }
            other => other,
        })
    }

    /// Gate and up of one routed layer: `(self · x, up · x)`, each
    /// `[batch, topk, n]` and each bit for bit what
    /// [`Self::indexed_moe_forward_with`] computes for it on the same arm.
    /// The arm is `batch`'s ([`MoeArm::for_rows`]);
    /// [`Self::indexed_moe_forward_pair_with`] takes it from the caller.
    pub fn indexed_moe_forward_pair(
        &self,
        up: &QTensor,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.indexed_moe_forward_pair_with(up, x, ids, MoeArm::for_rows(x.dim(0)?))
    }

    /// [`Self::indexed_moe_forward_pair`] on the caller's `arm`. On
    /// [`MoeArm::MatVec`] (a decode step) the two halves are ONE Metal launch,
    /// `kernel_mul_mv_id2`, when both weights share a kind and shape and the
    /// activations are f32 or f16 (each its own triple, as the single
    /// launch's), unless `LLOOM_MOE_PAIR=off`. The GEMM has no paired form,
    /// so on [`MoeArm::Gemm`] (a prefill tile) the halves are two calls on
    /// that arm, and the pair never takes a tile off the GEMM. The choice
    /// depends on the arm, the weights and the activation dtype, never on
    /// `batch`, so every group of a tile makes the same one
    /// (`docs/expert-streaming.md` §20.15a).
    pub fn indexed_moe_forward_pair_with(
        &self,
        up: &QTensor,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
    ) -> Result<(Tensor, Tensor)> {
        self.indexed_moe_forward_pair_as(up, x, ids, arm, x.dtype())
    }

    /// [`Self::indexed_moe_forward_pair_with`] computed in `act`: each half is
    /// [`Self::indexed_moe_forward_as`]'s for it. An f32 `x` computed in f16
    /// is the pair of the f16 launches over `x` cast to f16, bit for bit,
    /// without the cast (on Metal the rounding is in the launch's load).
    pub fn indexed_moe_forward_pair_as(
        &self,
        up: &QTensor,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.indexed_moe_forward_pair_in(up, x, ids, arm, act, PairOrder::configured())
    }

    /// [`Self::indexed_moe_forward_pair_as`] with the one launch's grid in
    /// `order` rather than `LLOOM_MOE_PAIR_ORDER`'s. The halves are the same
    /// bits in every order.
    pub fn indexed_moe_forward_pair_in(
        &self,
        up: &QTensor,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
        act: DType,
        order: PairOrder,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "metal")]
        if arm == MoeArm::MatVec
            && moe_pair_on()
            && self.dtype() == up.dtype()
            && self.shape() == up.shape()
            && matches!(
                (x.dtype(), act),
                (DType::F32, DType::F32) | (DType::F16, DType::F16) | (DType::F32, DType::F16)
            )
        {
            if let (QStorage::Metal(a), QStorage::Metal(b)) = (&self.storage, &up.storage) {
                let (xs, is) = (x.storage(), ids.storage());
                if let (Storage::Metal(xs), Storage::Metal(is)) = (&*xs, &*is) {
                    let (gs, us, shape) = a.indexed_moe_forward_pair(
                        b,
                        self.shape(),
                        xs,
                        x.layout(),
                        is,
                        ids.layout(),
                        order,
                        act,
                    )?;
                    let tensor = |s, shape| {
                        crate::tensor::from_storage(
                            Storage::Metal(s),
                            shape,
                            crate::op::BackpropOp::none(),
                            false,
                        )
                    };
                    return Ok(match us {
                        None => {
                            let both = tensor(gs, shape);
                            (
                                both.narrow(0, 0, 1)?.squeeze(0)?,
                                both.narrow(0, 1, 1)?.squeeze(0)?,
                            )
                        }
                        Some(us) => (tensor(gs, shape.clone()), tensor(us, shape)),
                    });
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = order;
        Ok((
            self.indexed_moe_forward_as(x, ids, arm, act)?,
            up.indexed_moe_forward_as(x, ids, arm, act)?,
        ))
    }

    /// Routed-expert matmul over a stacked quantized weight `[n_experts, n, k]`:
    /// `x` `[batch, in_dim1, k]` with `in_dim1 ∈ {1, topk}`, `ids` `[batch, topk]`
    /// → `[batch, topk, n]`, where plane `(b, t)` is `x[b, t or 0] · Wᵀ[ids[b, t]]`.
    /// The weight stays packed on every backend; the CPU arm dequantizes rows on
    /// the fly and is the reference the Metal `mul_mv_id` arm is gated against.
    /// On Metal the kernel is `batch`'s ([`MoeArm::for_rows`]).
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        self.indexed_moe_forward_with(x, ids, MoeArm::for_rows(x.dim(0)?))
    }

    /// [`Self::indexed_moe_forward`] on the caller's `arm`: a caller that
    /// splits a tile into groups decides the arm once for the tile, so each
    /// row is computed as the unsplit tile would compute it
    /// (`docs/expert-streaming.md` §20.15).
    pub fn indexed_moe_forward_with(
        &self,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
    ) -> Result<Tensor> {
        self.indexed_moe_forward_as(x, ids, arm, x.dtype())
    }

    /// [`Self::indexed_moe_forward_with`] computed in `act`: what it computes
    /// for `x` cast to `act`, bit for bit. On Metal an f32 `x` computed in f16
    /// takes the f16 launch with the rounding in its load (no cast launch);
    /// elsewhere `x` is cast first.
    pub fn indexed_moe_forward_as(
        &self,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
        act: DType,
    ) -> Result<Tensor> {
        let folds = matches!(self.storage, QStorage::Metal(_))
            && (x.dtype(), act) == (DType::F32, DType::F16);
        if act != x.dtype() && !folds {
            return self.indexed_moe_forward_as(&x.to_dtype(act)?, ids, arm, act);
        }
        let none = crate::op::BackpropOp::none();
        match &self.storage {
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        none,
                        false,
                    ))
                }
                _ => crate::bail!("indexed_moe_forward: x and ids must be on the weight's device"),
            },
            QStorage::Metal(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Metal(x_storage), Storage::Metal(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                        arm,
                        act,
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Metal(storage),
                        out_shape,
                        none,
                        false,
                    ))
                }
                _ => crate::bail!("indexed_moe_forward: x and ids must be on the weight's device"),
            },
            QStorage::Cpu(s) => {
                let (n_experts, n, k) = self.shape().dims3()?;
                let (batch, in_dim1, k_in) = x.dims3()?;
                let (batch_ids, topk) = ids.dims2()?;
                if k_in != k || batch != batch_ids || (in_dim1 != 1 && in_dim1 != topk) {
                    crate::bail!(
                        "indexed_moe_forward: x {:?} / ids {:?} do not fit weight {:?}",
                        x.shape(),
                        ids.shape(),
                        self.shape()
                    )
                }
                let x = x.to_dtype(DType::F32)?.contiguous()?;
                let x = x.flatten_all()?.to_vec1::<f32>()?;
                let ids = ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1::<u32>()?;
                let mut out = vec![0f32; batch * topk * n];
                s.matmul_t_indexed(
                    (n_experts, n, k),
                    &x,
                    in_dim1,
                    &ids,
                    (batch, topk),
                    &mut out,
                )?;
                let storage = Storage::Cpu(CpuStorage::F32(out));
                Ok(crate::tensor::from_storage(
                    storage,
                    (batch, topk, n),
                    none,
                    false,
                ))
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.storage.device_ptr_with_guard(stream)
    }
}

/// A block on the quantized matvec's store (`Source::QuantizedStore`,
/// `metal_src/quantized_store.metal`): the f32 store's type swapped for one
/// whose conversion or assignment carries the block, the cores unedited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MvStore {
    /// `silu(W·x)`.
    Silu,
    /// `dst[i] · (W·x)[i]` into `dst`: the SwiGLU's up projection multiplied
    /// into the gate's buffer.
    MulInto,
}

impl MvStore {
    /// The MSL store type.
    pub fn store_type(self) -> &'static str {
        match self {
            MvStore::Silu => "MvStoreSilu",
            MvStore::MulInto => "MvStoreMulInto",
        }
    }

    /// The `[[host_name]]` suffix after `_f32`.
    pub fn name_suffix(self) -> &'static str {
        match self {
            MvStore::Silu => "_silu",
            MvStore::MulInto => "_mul",
        }
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = crate::tuning::tuning().qmm.dequantize_all;
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = crate::tuning::tuning().qmm.dequantize_all_f16;
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    /// [`QTensor::indexed_moe_forward_pair`] on two quantized weights.
    pub fn indexed_moe_forward_pair(
        &self,
        up: &QMatMul,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        match (self, up) {
            (Self::QTensor(g), Self::QTensor(u)) => g.indexed_moe_forward_pair(u, x, ids),
            _ => Ok((
                self.indexed_moe_forward(x, ids)?,
                up.indexed_moe_forward(x, ids)?,
            )),
        }
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            _ => {
                panic!("Not implemented!")
            }
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.embedding(ids),
            Self::Tensor(w) | Self::TensorF16(w) => {
                let mut final_dims = ids.dims().to_vec();
                final_dims.push(w.dim(D::Minus1)?);
                let ids = ids.to_device(w.device())?.flatten_all()?;
                w.index_select(&ids, 0)?.reshape(final_dims)
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];

                // Try the 8-column BlockQ4Kx8 repacked path.
                #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
                if self_storage.dtype() == GgmlDType::Q4K && n.is_multiple_of(8) {
                    use zerocopy::{FromBytes, IntoBytes};

                    let total_blocks =
                        self_storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ4K>();
                    let repacked = self.repacked_qs.get_or_init(|| {
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                self_storage.as_ptr() as *const BlockQ4K,
                                total_blocks,
                            )
                        };
                        let packed = k_quants::pack_to_q4kx8(blocks, n);
                        Some(packed.as_bytes().to_vec())
                    });
                    if let Some(repacked_bytes) = repacked {
                        let block_x8: &[BlockQ4Kx8] =
                            <[BlockQ4Kx8]>::ref_from_bytes(repacked_bytes).map_err(|_| {
                                crate::Error::Msg(
                                    "repacked_qs alignment invariant violated".to_string(),
                                )
                            })?;

                        k_quants::matmul_q4k_x8(
                            (dst_shape.elem_count() / n, k, n),
                            slice,
                            block_x8,
                            &mut dst_storage,
                        )?;
                        return Ok((crate::CpuStorage::F32(dst_storage), dst_shape));
                    }
                }

                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                // A decode step's rows on Metal: the mixed arm, f32 in and
                // out, no cast launches (`gemv_mixed`). Everything else — a
                // prefill's rows, the CPU, a half activation — the classical
                // route.
                if let Some(out) = self.gemv_mixed(xs, gemv_mixed::MixedEpilogue::None, 1.0) {
                    return out;
                }
                #[cfg(feature = "metal")]
                if xs.device().is_metal()
                    && xs.dtype() == DType::F32
                    && w.dims().len() == 2
                    // The GEMM's f16 arm reads the weight row in 32-wide
                    // blocks (the quantized template's block): a K off the
                    // block read past the row — garbage that varied run to
                    // run at the tiny test geometry (#749 P30 (1)). Such a
                    // K takes the classical route below.
                    && crate::shape_req!(f16_weight_gemm, w.dims()[1] % 32 == 0; k = w.dims()[1])
                    && metal::f16_weight_mm_route()
                {
                    // A prefill's rows: the quantized GEMM's f16 arm reads
                    // the weights as half and the activations as float —
                    // no cast launch on either side (lloom #749 P23).
                    return metal::matmul_f16_weight_f32(xs, w);
                }
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}

impl QMatMul {
    /// Whether the quantized matvec takes a block on its store for `xs`:
    /// a packed weight whose kind has a genericized core
    /// (`kernels::mv_store_instantiations`), Metal, one contiguous f32 row of
    /// the weight's `k`. One predicate for both store entry points.
    pub fn mv_store_serves(&self, xs: &Tensor) -> bool {
        #[cfg(feature = "metal")]
        {
            let Self::QTensor(q) = self else {
                return false;
            };
            let kind: crate::kernels::GgmlDType = q.dtype().into();
            let Ok((_, k)) = q.shape().dims2() else {
                return false;
            };
            matches!(
                kind.kind().mul_mv,
                crate::kernels::kernels::quantized::MulMv::Quant(_)
            ) && matches!(q.storage, QStorage::Metal(_))
                && xs.device().is_metal()
                && xs.dtype() == DType::F32
                && xs.is_contiguous()
                && xs.elem_count() == k
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = xs;
            false
        }
    }

    /// `silu(W·x)` with the SiLU on the quantized matvec's store, one launch
    /// (`MvStore::Silu`). `None` where [`Self::mv_store_serves`] is false —
    /// the caller composes.
    pub fn forward_silu(&self, xs: &Tensor) -> Option<Result<Tensor>> {
        #[cfg(feature = "metal")]
        {
            if !self.mv_store_serves(xs) {
                return None;
            }
            let Self::QTensor(q) = self else {
                return None;
            };
            Some(xs.apply_op1_no_bwd(&MvSilu(q.clone())))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = xs;
            None
        }
    }

    /// `gate ← gate · (W·x)`: the products multiplied into `gate`'s own
    /// buffer at the store (`MvStore::MulInto`), one launch — the SwiGLU's up
    /// projection landing as `silu(g)·u` where the gate's SiLU stored
    /// `silu(g)`. `gate` is overwritten: it must have no other reader. `None`
    /// where the store block does not serve (`gate` an f32 contiguous
    /// `[.., n]` on Metal too) — the caller composes.
    pub fn forward_mul_into(&self, xs: &Tensor, gate: &Tensor) -> Option<Result<()>> {
        #[cfg(feature = "metal")]
        {
            if !self.mv_store_serves(xs) {
                return None;
            }
            let Self::QTensor(q) = self else {
                return None;
            };
            let n = q.shape().dims2().ok()?.0;
            if !gate.device().is_metal()
                || gate.dtype() != DType::F32
                || !gate.is_contiguous()
                || gate.elem_count() != n
            {
                return None;
            }
            Some(gate.inplace_op2(xs, &MvMulInto(q.clone())))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (xs, gate);
            None
        }
    }
}

impl QMatMul {
    /// `epilogue(scale · xs · Wᵀ)` for f16 weights and an f32 `xs` of at most
    /// [`gemv_mixed::GEMV_MIXED_MAX_ROWS`] rows, as ONE launch on Metal
    /// (`gemv_mixed`), `None` where the arm does not apply — the CPU, another
    /// weight kind, a half activation, a prefill's row count — so the caller
    /// composes the classical route instead. Never a silent fallback to a
    /// different arithmetic: the caller sees which arm ran.
    pub fn gemv_mixed(
        &self,
        xs: &Tensor,
        epilogue: gemv_mixed::MixedEpilogue,
        scale: f32,
    ) -> Option<Result<Tensor>> {
        let w = self.mixed_weight(xs)?;
        Some(xs.apply_op1_no_bwd(&gemv_mixed::GemvMixed {
            w: w.clone(),
            epilogue,
            scale,
            gnorm: None,
        }))
    }

    /// The f16 weight when the mixed arm serves `xs`: Metal, an f32 `xs` whose
    /// last dim is the weight's `k`, contiguous, at most
    /// [`gemv_mixed::GEMV_MIXED_MAX_ROWS`] rows. One predicate for both mixed
    /// entry points, so they cannot disagree on what they take.
    fn mixed_weight(&self, xs: &Tensor) -> Option<&Tensor> {
        let w = match self {
            Self::TensorF16(w) => w,
            _ => return None,
        };
        if !xs.device().is_metal()
            || xs.dtype() != DType::F32
            || w.dtype() != DType::F16
            || w.rank() != 2
            || !w.is_contiguous()
            || !xs.is_contiguous()
            || xs.rank() < 1
        {
            return None;
        }
        let k = w.dim(1).ok()?;
        if xs.dim(xs.rank() - 1).ok()? != k {
            return None;
        }
        let rows = xs.elem_count() / k;
        if rows == 0 || rows > gemv_mixed::GEMV_MIXED_MAX_ROWS {
            return None;
        }
        Some(w)
    }

    /// [`Self::gemv_mixed`] over `norm(xs)`, the grouped RMSNorm computed
    /// inside the launch (`docs/tuning.md` §13.5u): `xs` `[.., k]` normalized
    /// per group of `k / groups`, then scaled by `gamma` `[k]` f32. With
    /// `scales`, the `groups` scales `rsqrt(mean(x_g^2) + eps)` follow the
    /// `n_out` outputs (`[.., n_out + groups]`, one row only). `None` where
    /// [`Self::gemv_mixed_gnorm_applies`] is false — the caller composes.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_mixed_gnorm(
        &self,
        xs: &Tensor,
        gamma: &Tensor,
        eps: f32,
        groups: usize,
        epilogue: gemv_mixed::MixedEpilogue,
        scale: f32,
        scales: bool,
    ) -> Option<Result<Tensor>> {
        if !self.gemv_mixed_gnorm_applies(xs, gamma, groups, epilogue, scales) {
            return None;
        }
        let w = self.mixed_weight(xs)?;
        Some(xs.apply_op1_no_bwd(&gemv_mixed::GemvMixed {
            w: w.clone(),
            epilogue,
            scale,
            gnorm: Some(gemv_mixed::GroupNorm {
                gamma: gamma.clone(),
                eps,
                groups,
                scales,
            }),
        }))
    }

    /// Whether [`Self::gemv_mixed_gnorm`] serves these arguments, decided
    /// without dispatching anything: [`Self::gemv_mixed`]'s conditions, the
    /// instantiated tile (`kernels::gemv_mixed_gnorm_serves`), an epilogue
    /// other than the sigmoid, one row when the scales are asked for, and a
    /// contiguous f32 `gamma` of `k` on Metal. A caller fusing several
    /// consumers of one normalized input checks them all first.
    pub fn gemv_mixed_gnorm_applies(
        &self,
        xs: &Tensor,
        gamma: &Tensor,
        groups: usize,
        epilogue: gemv_mixed::MixedEpilogue,
        scales: bool,
    ) -> bool {
        #[cfg(feature = "metal")]
        {
            let Some(w) = self.mixed_weight(xs) else {
                return false;
            };
            let Ok((n_out, k)) = w.dims2() else {
                return false;
            };
            crate::kernels::gemv_mixed_gnorm_serves(n_out, k, groups)
                && epilogue != gemv_mixed::MixedEpilogue::Sigmoid
                && (!scales || xs.elem_count() == k)
                && gamma.dtype() == DType::F32
                && gamma.device().is_metal()
                && gamma.is_contiguous()
                && gamma.elem_count() == k
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (xs, gamma, groups, epilogue, scales);
            false
        }
    }
}

/// The kernel a routed matmul runs on Metal: the matmul-vec (a plane per
/// `(row, slot)`, the decode step's) or the GEMM with the rows grouped by
/// expert (a prefill tile's). The two are not bit-identical — the GEMM
/// multiplies the weights as f16 — so whoever splits a tile into groups
/// decides the arm once, for the tile, and hands it to every group
/// (`docs/expert-streaming.md` §20.15). The CPU and CUDA arms have one
/// kernel and ignore it; a weight or activation the GEMM does not take
/// runs the matmul-vec under either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeArm {
    MatVec,
    Gemm,
}

impl MoeArm {
    /// The arm of a tile of `rows` rows: the GEMM from `LLOOM_MOE_MM_FROM`
    /// rows (default 64; 0 = never), the matmul-vec below.
    pub fn for_rows(rows: usize) -> Self {
        #[cfg(feature = "metal")]
        let from = metal::moe_mm_from();
        #[cfg(not(feature = "metal"))]
        let from = crate::tuning::tuning().moe.mm_from;
        if from > 0 && rows >= from {
            MoeArm::Gemm
        } else {
            MoeArm::MatVec
        }
    }
}

/// How the routed gate|up pair lays its two halves out on Metal
/// (`LLOOM_MOE_PAIR_ORDER`): one launch with the half next to the plane (the
/// default), on the grid's slowest axis, or on its fastest axis; or the
/// pair's kernel launched once a half. Every order computes each plane with
/// the same core, so the bits never depend on it; the default is the one the
/// decode step runs fastest (`docs/expert-streaming.md` §20.15e).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairOrder {
    /// Gate's planes, then up's, the half read by a division (the previous
    /// default).
    Half,
    /// [`PairOrder::Half`]'s grid, the half read by a compare.
    HalfCmp,
    /// One expert's gate and up planes adjacent: the default.
    Expert,
    /// One row tile's gate and up threadgroups adjacent.
    Tile,
    /// The pair's kernel launched twice, one half each.
    Split,
}

impl PairOrder {
    pub const ALL: [PairOrder; 5] = [
        PairOrder::Half,
        PairOrder::HalfCmp,
        PairOrder::Expert,
        PairOrder::Tile,
        PairOrder::Split,
    ];

    pub fn parse(v: &str) -> std::result::Result<PairOrder, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "half" => Ok(PairOrder::Half),
            "half-cmp" => Ok(PairOrder::HalfCmp),
            "expert" => Ok(PairOrder::Expert),
            "tile" => Ok(PairOrder::Tile),
            "split" => Ok(PairOrder::Split),
            other => Err(format!(
                "moe pair order: `{other}` is not one of half, half-cmp, expert, tile, split"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PairOrder::Half => "half",
            PairOrder::HalfCmp => "half-cmp",
            PairOrder::Expert => "expert",
            PairOrder::Tile => "tile",
            PairOrder::Split => "split",
        }
    }

    /// The process's order (`LLOOM_MOE_PAIR_ORDER`, read once).
    pub fn configured() -> PairOrder {
        static ORDER: std::sync::OnceLock<PairOrder> = std::sync::OnceLock::new();
        *ORDER.get_or_init(|| {
            PairOrder::parse(&crate::tuning::tuning().moe.pair_order)
                .expect("the parser accepted it")
        })
    }
}

/// The routed gate|up pair launch on the matmul-vec arm (`LLOOM_MOE_PAIR`,
/// read once): on unless `off` | `0`.
#[cfg(feature = "metal")]
fn moe_pair_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().moe.pair)
}

/// The quantized matvec with `silu` on its store ([`QMatMul::forward_silu`]).
#[cfg(feature = "metal")]
struct MvSilu(std::sync::Arc<QTensor>);

#[cfg(feature = "metal")]
impl crate::CustomOp1 for MvSilu {
    fn name(&self) -> &'static str {
        "qmatmul-mv-silu"
    }

    fn cpu_fwd(
        &self,
        _: &crate::CpuStorage,
        _: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        crate::bail!("the quantized matvec store block is Metal's")
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let QStorage::Metal(w) = &self.0.storage else {
            crate::bail!("the quantized matvec store block needs a Metal weight")
        };
        let out = w
            .fwd_mv_store(
                &self.0.shape,
                storage,
                layout,
                crate::kernels::MvStore::Silu,
                None,
            )?
            .expect("a fresh store returns its storage");
        let n = self.0.shape.dims2()?.0;
        let mut dims = layout.shape().dims().to_vec();
        *dims.last_mut().expect("a row") = n;
        Ok((out, Shape::from(dims)))
    }
}

/// The quantized matvec's products multiplied into the gate's buffer
/// ([`QMatMul::forward_mul_into`]).
#[cfg(feature = "metal")]
struct MvMulInto(std::sync::Arc<QTensor>);

#[cfg(feature = "metal")]
impl crate::InplaceOp2 for MvMulInto {
    fn name(&self) -> &'static str {
        "qmatmul-mv-mul-into"
    }

    fn cpu_fwd(
        &self,
        _: &mut crate::CpuStorage,
        _: &crate::Layout,
        _: &crate::CpuStorage,
        _: &crate::Layout,
    ) -> Result<()> {
        crate::bail!("the quantized matvec store block is Metal's")
    }

    fn metal_fwd(
        &self,
        gate: &mut crate::MetalStorage,
        gate_layout: &crate::Layout,
        xs: &crate::MetalStorage,
        xs_layout: &crate::Layout,
    ) -> Result<()> {
        let QStorage::Metal(w) = &self.0.storage else {
            crate::bail!("the quantized matvec store block needs a Metal weight")
        };
        w.fwd_mv_store(
            &self.0.shape,
            xs,
            xs_layout,
            crate::kernels::MvStore::MulInto,
            Some((gate, gate_layout)),
        )?;
        Ok(())
    }
}
