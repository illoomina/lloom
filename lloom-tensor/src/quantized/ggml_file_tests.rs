//! Source-only loader controls: UNCOMPILED / UNRUN.

use super::{qtensor_from_ggml, GgmlDType};
use crate::quantized::{fp_quants::*, iq_quants::*, k_quants::*, QTensor};
use crate::Device;
#[cfg(feature = "metal")]
use crate::Result;
use half::{bf16, f16};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

const H1: [u8; 2] = 0x3c00u16.to_ne_bytes();
const HMIN: [u8; 2] = 0x3800u16.to_ne_bytes();
const HM2: [u8; 2] = 0xc000u16.to_ne_bytes();
const B1: [u8; 2] = 0x3f80u16.to_ne_bytes();

fn literal(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn offset_payload(bytes: &[u8]) -> Vec<u64> {
    let mut words = vec![0u64; (bytes.len() + 1).div_ceil(8)];
    words.as_mut_slice().as_mut_bytes()[1..1 + bytes.len()].copy_from_slice(bytes);
    words
}

fn load(bytes: &[u8], dtype: GgmlDType, dims: Vec<usize>, device: &Device) -> QTensor {
    qtensor_from_ggml(dtype, bytes, dims, device).unwrap()
}

fn refusal(bytes: &[u8], dtype: GgmlDType, dims: Vec<usize>, device: &Device, text: &str) {
    let error = qtensor_from_ggml(dtype, bytes, dims, device).unwrap_err();
    assert!(error.to_string().contains(text), "{dtype:?}: {error}");
}

fn check_layout<T>(
    block: T,
    size: usize,
    alignment: usize,
    elems: usize,
    bytes: Vec<u8>,
    device: &Device,
) where
    T: GgmlType + FromBytes + KnownLayout + IntoBytes + Immutable + 'static,
{
    assert_eq!(std::mem::size_of::<T>(), size, "{:?}", T::DTYPE);
    assert_eq!(std::mem::align_of::<T>(), alignment, "{:?}", T::DTYPE);
    assert_eq!(T::DTYPE.type_size(), size);
    assert_eq!(T::BLCK_SIZE, elems);
    assert_eq!(T::DTYPE.block_size(), elems);
    assert_eq!(block.as_bytes(), bytes, "literal block field order");

    let mut second = block.clone();
    *second.as_mut_bytes().last_mut().unwrap() ^= 1;
    let blocks = [block, second];
    let mut expected = bytes.repeat(2);
    *expected.last_mut().unwrap() ^= 1;
    let aligned = blocks.as_slice().as_bytes();
    assert_eq!(aligned.len(), 2 * size);
    assert_eq!((aligned.as_ptr() as usize) % alignment, 0);
    let offset = offset_payload(&expected);
    let unaligned = &offset.as_slice().as_bytes()[1..1 + expected.len()];
    if alignment > 1 {
        assert_ne!((unaligned.as_ptr() as usize) % alignment, 0, "recovery arm");
    }
    // MXFP4 has alignment one: an offset slice is legal but cannot be misaligned.
    for input in [aligned, unaligned] {
        let tensor = load(input, T::DTYPE, vec![2, elems], device);
        assert_eq!(tensor.dtype(), T::DTYPE);
        assert_eq!(tensor.shape().dims(), &[2, elems]);
        assert!(
            tensor.device().same_device(device),
            "supplied device changed"
        );
        assert!(tensor.storage_size_in_bytes() >= expected.len());
        if device.is_cpu() {
            assert_eq!(tensor.storage_size_in_bytes(), 2 * size);
        }
        assert_eq!(
            tensor.data().unwrap().as_ref(),
            expected,
            "packed byte identity"
        );
        refusal(
            &input[..input.len() - 1],
            T::DTYPE,
            vec![2, elems],
            device,
            "payload length",
        );
    }
    for extra in [size, 1] {
        let mut padded = expected.clone();
        padded.resize(padded.len() + extra, 0);
        // The original API consumed only the shape-sized prefix. Both whole
        // extra blocks and partial trailing bytes remain legal and unobserved.
        let offset_padded = offset_payload(&padded);
        for input in [
            padded.as_slice(),
            &offset_padded.as_slice().as_bytes()[1..1 + padded.len()],
        ] {
            let tensor = load(input, T::DTYPE, vec![2, elems], device);
            assert_eq!(tensor.shape().dims(), &[2, elems]);
            assert!(tensor.device().same_device(device));
            assert_eq!(tensor.data().unwrap().as_ref(), expected);
        }
    }
    let empty_words = offset_payload(&[]);
    let empty_unaligned = &empty_words.as_slice().as_bytes()[1..1];
    for dims in [vec![0], vec![0, elems], vec![2, 0]] {
        let empty = load(empty_unaligned, T::DTYPE, dims.clone(), device);
        assert_eq!(empty.shape().dims(), dims);
        assert_eq!(empty.data().unwrap().as_ref(), &[] as &[u8]);
        assert!(empty.device().same_device(device));
        let empty_prefix = load(&[0xa5], T::DTYPE, dims, device);
        assert_eq!(empty_prefix.data().unwrap().as_ref(), &[] as &[u8]);
    }
    refusal(&[], T::DTYPE, vec![], device, "scalar tensor");
    if elems > 1 {
        refusal(&[], T::DTYPE, vec![elems - 1], device, "last dim divisible");
        refusal(
            aligned,
            T::DTYPE,
            vec![4, elems / 2],
            device,
            "last dim divisible",
        );
        refusal(
            &[],
            T::DTYPE,
            vec![0, elems - 1],
            device,
            "last dim divisible",
        );
    }
}

macro_rules! file_layouts {
    ($check:ident, $device:expr) => {{
        let device = $device;
        let d = f16::from_f32(1.0);
        let dmin = f16::from_f32(0.5);
        let m = f16::from_f32(-2.0);
        $check(1.25f32, 4, 4, 1, 1.25f32.to_ne_bytes().to_vec(), device);
        $check(d, 2, 2, 1, H1.to_vec(), device);
        $check(bf16::from_f32(1.0), 2, 2, 1, B1.to_vec(), device);
        $check(
            BlockQ4_0 { d, qs: [0x91; 16] },
            18,
            2,
            32,
            literal(&[&H1, &[0x91; 16]]),
            device,
        );
        $check(
            BlockQ4_1 {
                d,
                m,
                qs: [0x92; 16],
            },
            20,
            2,
            32,
            literal(&[&H1, &HM2, &[0x92; 16]]),
            device,
        );
        $check(
            BlockQ5_0 {
                d,
                qh: [0x53; 4],
                qs: [0x93; 16],
            },
            22,
            2,
            32,
            literal(&[&H1, &[0x53; 4], &[0x93; 16]]),
            device,
        );
        $check(
            BlockQ5_1 {
                d,
                m,
                qh: [0x54; 4],
                qs: [0x94; 16],
            },
            24,
            2,
            32,
            literal(&[&H1, &HM2, &[0x54; 4], &[0x94; 16]]),
            device,
        );
        $check(
            BlockQ8_0 { d, qs: [-3; 32] },
            34,
            2,
            32,
            literal(&[&H1, &[0xfd; 32]]),
            device,
        );
        $check(
            BlockQ2K {
                scales: [0x21; 16],
                qs: [0x96; 64],
                d,
                dmin,
            },
            84,
            2,
            256,
            literal(&[&[0x21; 16], &[0x96; 64], &H1, &HMIN]),
            device,
        );
        $check(
            BlockQ3K {
                hmask: [0x57; 32],
                qs: [0x97; 64],
                scales: [0x22; 12],
                d,
            },
            110,
            2,
            256,
            literal(&[&[0x57; 32], &[0x97; 64], &[0x22; 12], &H1]),
            device,
        );
        $check(
            BlockQ4K {
                d,
                dmin,
                scales: [0x23; 12],
                qs: [0x98; 128],
            },
            144,
            2,
            256,
            literal(&[&H1, &HMIN, &[0x23; 12], &[0x98; 128]]),
            device,
        );
        $check(
            BlockQ5K {
                d,
                dmin,
                scales: [0x24; 12],
                qh: [0x59; 32],
                qs: [0x99; 128],
            },
            176,
            2,
            256,
            literal(&[&H1, &HMIN, &[0x24; 12], &[0x59; 32], &[0x99; 128]]),
            device,
        );
        $check(
            BlockQ6K {
                ql: [0x9a; 128],
                qh: [0x5a; 64],
                scales: [-5; 16],
                d,
            },
            210,
            2,
            256,
            literal(&[&[0x9a; 128], &[0x5a; 64], &[0xfb; 16], &H1]),
            device,
        );
        $check(
            BlockIQ2xxs {
                d,
                qs: [0x1234; 32],
            },
            66,
            2,
            256,
            literal(&[&H1, &0x1234u16.to_ne_bytes().repeat(32)]),
            device,
        );
        $check(
            BlockIQ2xs {
                d,
                qs: [0x1235; 32],
                scales: [0x25; 8],
            },
            74,
            2,
            256,
            literal(&[&H1, &0x1235u16.to_ne_bytes().repeat(32), &[0x25; 8]]),
            device,
        );
        $check(
            BlockIQ2s {
                d,
                qs: [0x9b; 64],
                qh: [0x5b; 8],
                scales: [0x26; 8],
            },
            82,
            2,
            256,
            literal(&[&H1, &[0x9b; 64], &[0x5b; 8], &[0x26; 8]]),
            device,
        );
        $check(
            BlockIQ3xxs { d, qs: [0x9c; 96] },
            98,
            2,
            256,
            literal(&[&H1, &[0x9c; 96]]),
            device,
        );
        $check(
            BlockIQ3s {
                d,
                qs: [0x9d; 64],
                qh: [0x5d; 8],
                signs: [0x55; 32],
                scales: [0x27; 4],
            },
            110,
            2,
            256,
            literal(&[&H1, &[0x9d; 64], &[0x5d; 8], &[0x55; 32], &[0x27; 4]]),
            device,
        );
        $check(
            BlockIQ4nl { d, qs: [0x9e; 16] },
            18,
            2,
            32,
            literal(&[&H1, &[0x9e; 16]]),
            device,
        );
        $check(
            BlockIQ4xs {
                d,
                scales_h: 0x1236,
                scales_l: [0x28; 4],
                qs: [0x9f; 128],
            },
            136,
            2,
            256,
            literal(&[&H1, &0x1236u16.to_ne_bytes(), &[0x28; 4], &[0x9f; 128]]),
            device,
        );
        $check(
            BlockQ2_0 { d, qs: [0xe4; 16] },
            18,
            2,
            64,
            literal(&[&H1, &[0xe4; 16]]),
            device,
        );
        $check(
            BlockMxfp4 {
                e: 128,
                qs: [0x21; 16],
            },
            17,
            1,
            32,
            literal(&[&[128], &[0x21; 16]]),
            device,
        );
        $check(
            BlockFp8E4M3 {
                d: bf16::from_f32(1.0),
                qs: [0x38; 32],
            },
            34,
            2,
            32,
            literal(&[&B1, &[0x38; 32]]),
            device,
        );
    }};
}

#[test]
fn all_file_layouts_preserve_aligned_and_offset_payloads() {
    file_layouts!(check_layout, &Device::Cpu);
}

#[test]
fn shape_and_byte_overflows_refuse_before_payload_conversion() {
    for (dtype, dims, message) in [
        (
            GgmlDType::F32,
            vec![usize::MAX, 2],
            "element count overflow",
        ),
        (
            GgmlDType::Q4K,
            vec![usize::MAX, 256],
            "element count overflow",
        ),
        (
            GgmlDType::F32,
            vec![usize::MAX / 4 + 1],
            "byte size overflow",
        ),
        (
            GgmlDType::F32,
            vec![isize::MAX as usize / 4 + 1],
            "host allocation limit",
        ),
        (
            GgmlDType::F32,
            vec![usize::MAX, 2, 0],
            "element count overflow",
        ),
    ] {
        refusal(&[], dtype, dims, &Device::Cpu, message);
    }
}

#[test]
fn activation_formats_keep_their_original_early_error() {
    for dtype in [GgmlDType::Q8_1, GgmlDType::Q8K] {
        for dims in [vec![], vec![0], vec![usize::MAX, 2]] {
            refusal(&[], dtype, dims, &Device::Cpu, "not supported yet");
        }
    }
}

#[test]
fn literal_cpu_references_cover_scalar_types_and_quantized_blocks() {
    let q4 = literal(&[&HMIN, &[0x91; 16]]);
    let q2 = literal(&[&HM2, &[0xe4; 16]]);
    let cases = [
        (
            GgmlDType::F32,
            literal(&[&1.25f32.to_ne_bytes(), &(-2.5f32).to_ne_bytes()]),
            vec![1.25, -2.5],
        ),
        (GgmlDType::F16, literal(&[&H1, &HM2]), vec![1.0, -2.0]),
        (
            GgmlDType::BF16,
            literal(&[&B1, &0xc000u16.to_ne_bytes()]),
            vec![1.0, -2.0],
        ),
        (
            GgmlDType::Q4_0,
            q4,
            [vec![-3.5; 16], vec![0.5; 16]].concat(),
        ),
        (GgmlDType::Q2_0, q2, [2.0f32, 0.0, -2.0, -4.0].repeat(16)),
    ];
    for (dtype, bytes, expected) in cases {
        let offset = offset_payload(&bytes);
        let input = &offset.as_slice().as_bytes()[1..1 + bytes.len()];
        let tensor = load(input, dtype, vec![expected.len()], &Device::Cpu);
        assert_eq!(
            tensor
                .dequantize(&Device::Cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
        );
        assert_eq!(tensor.data().unwrap().as_ref(), bytes);
    }
}

#[test]
fn owned_cpu_recovery_preserves_bits_after_input_changes() {
    let bytes = literal(&[&0x7fc0_1234u32.to_ne_bytes(), &0x8000_0000u32.to_ne_bytes()]);
    let mut input = offset_payload(&bytes);
    let tensor = load(
        &input.as_slice().as_bytes()[1..9],
        GgmlDType::F32,
        vec![2],
        &Device::Cpu,
    );
    input.as_mut_slice().as_mut_bytes().fill(0);
    assert_eq!(
        tensor.data().unwrap().as_ref(),
        bytes,
        "NaN payload / negative zero bits"
    );
}

#[cfg(not(feature = "metal"))]
#[test]
fn unavailable_metal_keeps_backend_error_after_valid_offset_recovery() {
    let device = Device::Metal(crate::MetalDevice);
    file_layouts!(check_unavailable_layout, &device);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn unavailable_cuda_keeps_backend_error_after_valid_offset_recovery() {
    let device = Device::Cuda(crate::cuda::CudaDevice);
    file_layouts!(check_unavailable_layout, &device);
}

#[cfg(any(not(feature = "cuda"), not(feature = "metal")))]
fn check_unavailable_layout<T>(
    block: T,
    _size: usize,
    _alignment: usize,
    elems: usize,
    bytes: Vec<u8>,
    device: &Device,
) where
    T: GgmlType + FromBytes + KnownLayout + IntoBytes + Immutable + 'static,
{
    let aligned = [block];
    let offset = offset_payload(&bytes);
    for input in [
        aligned.as_slice().as_bytes(),
        &offset.as_slice().as_bytes()[1..1 + bytes.len()],
    ] {
        let error = qtensor_from_ggml(T::DTYPE, input, vec![elems], device).unwrap_err();
        assert!(
            matches!(
                (device, &error),
                (Device::Metal(_), crate::Error::NotCompiledWithMetalSupport)
                    | (Device::Cuda(_), crate::Error::NotCompiledWithCudaSupport)
            ),
            "supplied backend error changed: {error}"
        );
        refusal(
            &input[..input.len() - 1],
            T::DTYPE,
            vec![elems],
            device,
            "payload length",
        );
        refusal(input, T::DTYPE, vec![], device, "scalar tensor");
        if elems > 1 {
            refusal(
                input,
                T::DTYPE,
                vec![2, elems / 2],
                device,
                "last dim divisible",
            );
        }
        refusal(
            &[],
            T::DTYPE,
            vec![usize::MAX, elems.max(2)],
            device,
            "element count overflow",
        );
        refusal(
            input,
            GgmlDType::Q8_1,
            vec![32],
            device,
            "not supported yet",
        );
    }
}

#[test]
fn gguf_reader_excludes_intertensor_padding_from_public_payload() {
    use crate::quantized::gguf_file::TensorInfo;
    use std::io::Cursor;
    let bytes = literal(&[&1.25f32.to_ne_bytes(), &(-2.5f32).to_ne_bytes()]);
    let mut file = vec![0xa5; 96];
    file[64..72].copy_from_slice(&bytes);
    let info = TensorInfo {
        ggml_dtype: GgmlDType::F32,
        shape: vec![2].into(),
        offset: 32,
    };
    let tensor = info.read(&mut Cursor::new(file), 32, &Device::Cpu).unwrap();
    assert_eq!(tensor.data().unwrap().as_ref(), bytes);
}

#[test]
fn legacy_ggml_reader_checks_shape_before_reading_payload() {
    use std::io::{Cursor, Read, Seek, SeekFrom};
    struct HeaderOnly {
        header: Cursor<Vec<u8>>,
    }
    impl Read for HeaderOnly {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            assert!(
                self.header.position() < self.header.get_ref().len() as u64,
                "unexpected payload read"
            );
            self.header.read(bytes)
        }
    }
    impl Seek for HeaderOnly {
        fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
            self.header.seek(at)
        }
    }
    let header = literal(&[
        &1u32.to_le_bytes(),
        &1u32.to_le_bytes(),
        &2u32.to_le_bytes(),
        &31u32.to_le_bytes(),
        b"w",
    ]);
    let mut reader = HeaderOnly {
        header: Cursor::new(header),
    };
    let error = super::read_one_tensor(
        &mut reader,
        super::VersionedMagic::GgmlUnversioned,
        &Device::Cpu,
    )
    .unwrap_err();
    assert!(error.to_string().contains("last dim divisible"), "{error}");
}

#[test]
fn legacy_ggml_reader_keeps_adjacent_tensor_payloads_separate() {
    use std::io::Cursor;
    let bytes = 1.25f32.to_ne_bytes();
    let mut file = literal(&[&0x67676d6cu32.to_le_bytes(), &[0; 28]]);
    for name in [b"a", b"b"] {
        file.extend(literal(&[
            &1u32.to_le_bytes(),
            &1u32.to_le_bytes(),
            &0u32.to_le_bytes(),
            &1u32.to_le_bytes(),
            name,
            &bytes,
        ]));
    }
    let content = super::Content::read(&mut Cursor::new(file), &Device::Cpu).unwrap();
    assert_eq!(content.tensors.len(), 2);
    for name in ["a", "b"] {
        assert_eq!(content.tensors[name].data().unwrap().as_ref(), bytes);
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires separately admitted tiny native Metal upload/readback"]
fn native_supplied_metal_device_preserves_all_file_layouts() -> Result<()> {
    let device = Device::new_metal(0)?;
    let requested = crate::tuning::tuning().metal.command_api.as_str();
    let actual = device
        .as_metal_device()?
        .commands()
        .actual_command_api_for_test();
    eprintln!("GGML_LOADER_NATIVE requested={requested} actual={actual} layouts=23");
    if matches!(requested, "metal3" | "metal4") {
        assert_eq!(actual, requested, "native API must not silently fall back");
    }
    file_layouts!(check_layout, &device);
    Ok(())
}
