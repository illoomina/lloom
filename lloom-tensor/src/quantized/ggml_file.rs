//! Support for the GGML file format.

use super::{iq_quants, k_quants, GgmlDType, QStorage};
use crate::{Context, Device, Result, Shape};
use byteorder::{LittleEndian, ReadBytesExt};
use std::borrow::Cow;
use std::collections::HashMap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x67676a74 => Self::Ggjt,
            0x67676c61 => Self::Ggla,
            0x67676d66 => Self::Ggmf,
            0x67676d6c => Self::Ggml,
            0x6767736e => Self::Ggsn,
            _ => crate::bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        if magic == Magic::Ggml {
            return Ok(Self::GgmlUnversioned);
        }
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Ggmf, 1) => Self::GgmfV1,
            (Magic::Ggjt, 1) => Self::GgjtV1,
            (Magic::Ggjt, 2) => Self::GgjtV2,
            (Magic::Ggjt, 3) => Self::GgjtV3,
            _ => crate::bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn align32(&self) -> bool {
        match self {
            Self::GgmlUnversioned | Self::GgmfV1 => false,
            Self::GgjtV1 | Self::GgjtV2 | Self::GgjtV3 => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl HParams {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let n_vocab = reader.read_u32::<LittleEndian>()?;
        let n_embd = reader.read_u32::<LittleEndian>()?;
        let n_mult = reader.read_u32::<LittleEndian>()?;
        let n_head = reader.read_u32::<LittleEndian>()?;
        let n_layer = reader.read_u32::<LittleEndian>()?;
        let n_rot = reader.read_u32::<LittleEndian>()?;
        let ftype = reader.read_u32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_embd,
            n_mult,
            n_head,
            n_layer,
            n_rot,
            ftype,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub token_score_pairs: Vec<(Vec<u8>, f32)>,
}

impl Vocab {
    fn read<R: std::io::Read>(reader: &mut R, n_vocab: usize) -> Result<Self> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
        let mut token_score_pairs = Vec::with_capacity(n_vocab);
        for _index in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let score = reader.read_f32::<LittleEndian>()?;
            token_score_pairs.push((word, score))
        }
        Ok(Self { token_score_pairs })
    }
}

fn checked_tensor_bytes(ggml_dtype: GgmlDType, shape: &Shape) -> Result<usize> {
    ggml_dtype.validate_file_weight()?;
    let block_size = ggml_dtype.block_size();
    super::check_shape(shape, block_size)?;
    let tensor_elems = shape.dims().iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .context("ggml: tensor element count overflow")
    })?;
    let size_in_bytes = (tensor_elems / block_size)
        .checked_mul(ggml_dtype.type_size())
        .context("ggml: tensor byte size overflow")?;
    if size_in_bytes > isize::MAX as usize {
        crate::bail!("ggml: tensor byte size {size_in_bytes} exceeds host allocation limit")
    }
    Ok(size_in_bytes)
}

fn from_raw_data<T>(raw_data: &[u8], shape: Shape, device: &Device) -> Result<super::QTensor>
where
    T: super::GgmlType + Send + Sync + IntoBytes + Immutable + FromBytes + KnownLayout + 'static,
{
    let block_bytes = std::mem::size_of::<T>();
    if !raw_data.len().is_multiple_of(block_bytes) {
        crate::bail!(
            "ggml: payload byte count is not a multiple of the host block size {block_bytes}"
        )
    }
    let data: Cow<'_, [T]> =
        if (raw_data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            let data = <[T]>::ref_from_bytes(raw_data)
                .map_err(|_| crate::Error::Msg("ggml: invalid aligned block payload".into()))?;
            Cow::Borrowed(data)
        } else {
            let n_blocks = raw_data.len() / block_bytes;
            let mut data = Vec::<T>::new();
            data.try_reserve_exact(n_blocks).map_err(|e| {
                crate::Error::Msg(format!(
                    "ggml: cannot reserve {n_blocks} aligned blocks: {e}"
                ))
            })?;
            data.resize_with(n_blocks, T::zeros);
            data.as_mut_slice().as_mut_bytes().copy_from_slice(raw_data);
            Cow::Owned(data)
        };
    let data: QStorage = match device {
        Device::Cpu => QStorage::Cpu(Box::new(data.into_owned())),
        Device::Metal(metal) => super::metal::load_quantized(metal, data.as_ref())?,
        Device::Cuda(cuda) => super::cuda::load_quantized(cuda, data.as_ref())?,
    };
    super::QTensor::new(data, shape)
}

/// Creates a quantized tensor from the shape-sized prefix of a GGML payload.
///
/// The shape must be non-scalar with a block-divisible last dimension. The
/// supplied bytes must contain the checked block extent; trailing bytes are
/// ignored, as before. Unaligned input is copied into owned aligned blocks
/// without changing the packed representation. Payload byte order is unchanged.
pub fn qtensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let shape = Shape::from(dims);
    let size_in_bytes = checked_tensor_bytes(ggml_dtype, &shape)?;
    let raw_data = raw_data.get(..size_in_bytes).with_context(|| {
        format!(
            "ggml: payload length {} is shorter than expected {size_in_bytes} bytes",
            raw_data.len()
        )
    })?;

    match ggml_dtype {
        GgmlDType::F32 => from_raw_data::<f32>(raw_data, shape, device),
        GgmlDType::F16 => from_raw_data::<half::f16>(raw_data, shape, device),
        GgmlDType::BF16 => from_raw_data::<half::bf16>(raw_data, shape, device),
        GgmlDType::Q4_0 => from_raw_data::<k_quants::BlockQ4_0>(raw_data, shape, device),
        GgmlDType::Q4_1 => from_raw_data::<k_quants::BlockQ4_1>(raw_data, shape, device),
        GgmlDType::Q5_0 => from_raw_data::<k_quants::BlockQ5_0>(raw_data, shape, device),
        GgmlDType::Q5_1 => from_raw_data::<k_quants::BlockQ5_1>(raw_data, shape, device),
        GgmlDType::Q8_0 => from_raw_data::<k_quants::BlockQ8_0>(raw_data, shape, device),
        GgmlDType::Q2K => from_raw_data::<k_quants::BlockQ2K>(raw_data, shape, device),
        GgmlDType::Q3K => from_raw_data::<k_quants::BlockQ3K>(raw_data, shape, device),
        GgmlDType::Q4K => from_raw_data::<k_quants::BlockQ4K>(raw_data, shape, device),
        GgmlDType::Q5K => from_raw_data::<k_quants::BlockQ5K>(raw_data, shape, device),
        GgmlDType::Q6K => from_raw_data::<k_quants::BlockQ6K>(raw_data, shape, device),
        GgmlDType::IQ2XXS => from_raw_data::<iq_quants::BlockIQ2xxs>(raw_data, shape, device),
        GgmlDType::IQ2XS => from_raw_data::<iq_quants::BlockIQ2xs>(raw_data, shape, device),
        GgmlDType::IQ2S => from_raw_data::<iq_quants::BlockIQ2s>(raw_data, shape, device),
        GgmlDType::IQ3XXS => from_raw_data::<iq_quants::BlockIQ3xxs>(raw_data, shape, device),
        GgmlDType::IQ3S => from_raw_data::<iq_quants::BlockIQ3s>(raw_data, shape, device),
        GgmlDType::IQ4NL => from_raw_data::<iq_quants::BlockIQ4nl>(raw_data, shape, device),
        GgmlDType::IQ4XS => from_raw_data::<iq_quants::BlockIQ4xs>(raw_data, shape, device),
        GgmlDType::Q2_0 => from_raw_data::<iq_quants::BlockQ2_0>(raw_data, shape, device),
        GgmlDType::MXFP4 => from_raw_data::<super::fp_quants::BlockMxfp4>(raw_data, shape, device),
        GgmlDType::FP8E4M3 => {
            from_raw_data::<super::fp_quants::BlockFp8E4M3>(raw_data, shape, device)
        }
        // The activation-side kinds; no file stores a weight in them. No
        // wildcard: a new kind must choose here, or a GGUF holding it builds
        // and is refused only when read.
        GgmlDType::Q8_1 | GgmlDType::Q8K => {
            crate::bail!("quantized type {ggml_dtype:?} is not supported yet")
        }
    }
}

fn read_one_tensor<R: std::io::Seek + std::io::Read>(
    reader: &mut R,
    magic: VersionedMagic,
    device: &Device,
) -> Result<(String, super::QTensor)> {
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let name_len = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
    let mut dims = vec![0u32; n_dims as usize];
    reader.read_u32_into::<LittleEndian>(&mut dims)?;
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse();
    let mut name = vec![0u8; name_len as usize];
    reader.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    if magic.align32() {
        let pos = reader.stream_position()?;
        reader.seek(std::io::SeekFrom::Current(((32 - pos % 32) % 32) as i64))?;
    }
    let shape = Shape::from(dims.iter().map(|&u| u as usize).collect::<Vec<_>>());
    let size_in_bytes = checked_tensor_bytes(ggml_dtype, &shape)?;
    // TODO: Mmap version to avoid copying the data around?
    let mut raw_data = Vec::new();
    raw_data.try_reserve_exact(size_in_bytes).map_err(|e| {
        crate::Error::Msg(format!(
            "ggml: cannot reserve {size_in_bytes} tensor bytes: {e}"
        ))
    })?;
    raw_data.resize(size_in_bytes, 0u8);
    reader.read_exact(&mut raw_data)?;
    match qtensor_from_ggml(ggml_dtype, &raw_data, shape.into_dims(), device) {
        Ok(tensor) => Ok((name, tensor)),
        Err(e) => crate::bail!("Error creating tensor {name}: {e}"),
    }
}

pub struct Content {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
    pub tensors: HashMap<String, super::QTensor>,
    pub device: Device,
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        device: &Device,
    ) -> Result<Content> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let magic = VersionedMagic::read(reader)?;
        let hparams = HParams::read(reader)?;
        let vocab = Vocab::read(reader, hparams.n_vocab as usize)?;
        let mut tensors = HashMap::new();

        while reader.stream_position()? != last_position {
            let (name, tensor) = read_one_tensor(reader, magic, device)?;
            tensors.insert(name, tensor);
        }
        let device = device.clone();
        Ok(Self {
            magic,
            hparams,
            vocab,
            tensors,
            device,
        })
    }

    pub fn remove(&mut self, name: &str) -> Result<super::QTensor> {
        match self.tensors.remove(name) {
            None => crate::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}

#[cfg(test)]
#[path = "ggml_file_tests.rs"]
mod tests;
