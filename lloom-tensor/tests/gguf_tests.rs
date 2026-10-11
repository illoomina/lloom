//! Regression tests for the allocation caps added in huggingface/candle#3533.

use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::Device;
use std::io::Cursor;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_V3: u32 = 3;

fn header(tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&GGUF_V3.to_le_bytes());
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_kv_count.to_le_bytes());
    buf
}

fn length_prefixed(s: &[u8]) -> Vec<u8> {
    let mut buf = (s.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(s);
    buf
}

/// Pad past the upfront `count * min_per_entry` sufficient-bytes check so
/// the inner per-field cap is what fires.
fn pad(buf: &mut Vec<u8>, n: usize) {
    buf.resize(buf.len() + n, 0);
}

fn assert_rejects(buf: Vec<u8>, msg_contains: &str) {
    let mut cursor = Cursor::new(buf);
    let err = Content::read(&mut cursor).expect_err("expected Err");
    let msg = format!("{err}");
    assert!(msg.contains(msg_contains), "unexpected error: {msg}");
}

#[test]
fn empty_header_loads() {
    let mut cursor = Cursor::new(header(0, 0));
    let content = Content::read(&mut cursor).expect("empty header should parse");
    assert!(content.metadata.is_empty());
    assert!(content.tensor_infos.is_empty());
}

#[test]
fn rejects_oversized_metadata_kv_count() {
    assert_rejects(header(0, 1u64 << 31), "metadata_kv_count");
}

#[test]
fn rejects_oversized_tensor_count() {
    assert_rejects(header(1u64 << 31, 0), "tensor_count");
}

#[test]
fn rejects_oversized_string_length() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

#[test]
fn rejects_oversized_array_length() {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    buf.extend_from_slice(&0u32.to_le_bytes()); // inner type = U8
    buf.extend_from_slice(&(1u64 << 31).to_le_bytes()); // element count
    pad(&mut buf, 64);
    assert_rejects(buf, "array length");
}

#[test]
fn rejects_oversized_n_dimensions() {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    pad(&mut buf, 64);
    assert_rejects(buf, "dimensions");
}

#[test]
fn rejects_deeply_nested_arrays() {
    // 256 nested arrays of arrays, well above GGUF_MAX_VALUE_DEPTH (64).
    // Each layer is type=Array(9) + inner_type(u32) + len=1(u64) = 16 bytes
    // except the innermost which terminates with a U8 (type=0) + len=0(u64).
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"k"));
    buf.extend_from_slice(&9u32.to_le_bytes()); // outer value_type = Array
    for _ in 0..256 {
        buf.extend_from_slice(&9u32.to_le_bytes()); // inner type = Array
        buf.extend_from_slice(&1u64.to_le_bytes()); // len = 1
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // terminal inner type = U8
    buf.extend_from_slice(&0u64.to_le_bytes()); // terminal len = 0
    assert_rejects(buf, "nesting depth");
}

#[test]
fn empty_v1_header_loads() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&1u32.to_le_bytes()); // version 1: u32 length prefixes
    buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u32.to_le_bytes()); // metadata_kv_count
    let mut cursor = Cursor::new(buf);
    Content::read(&mut cursor).expect("empty v1 header should parse");
}

#[test]
fn rejects_tensor_size_exceeding_file() {
    // Create a gguf tensor that claims shape [1_073_741_824] F32 (4 GiB), but has no actual tensor data.
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&1u32.to_le_bytes()); // n dims
    buf.extend_from_slice(&1_073_741_824u64.to_le_bytes()); // 1 GiB elements
    buf.extend_from_slice(&0u32.to_le_bytes()); // F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // no offset
    let mut cursor = Cursor::new(buf);
    let content = Content::read(&mut cursor).expect("header should parse");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .expect_err("expected Err from oversized tensor load");
    let msg = format!("{err}");
    // Both hosts inspect the metadata. 64-bit reaches the file-range check;
    // on 32-bit the 4-GiB byte count cannot be materialized even before I/O.
    #[cfg(target_pointer_width = "64")]
    assert!(msg.contains("remaining"), "unexpected error: {msg}");
    #[cfg(target_pointer_width = "32")]
    assert!(
        msg.contains("tensor byte size exceeds host usize"),
        "unexpected error: {msg}"
    );
}

#[test]
fn rejects_string_length_above_remaining_file_bytes() {
    let mut buf = header(1, 0);
    buf.extend_from_slice(&(1u64 << 20).to_le_bytes()); // 1 MB, below cap, above file size
    pad(&mut buf, 64);
    assert_rejects(buf, "string length");
}

/// `write` is `write_header` at `packed_offsets` followed by each tensor's
/// bytes at its offset: the split a sparse writer places the data through
/// (lloom `testing::scale`) produces the file `write` does, byte for byte,
/// and the reader finds each tensor where the header says.
#[test]
fn write_is_its_header_then_each_tensor_at_its_offset() -> lloom_tensor::Result<()> {
    use lloom_tensor::quantized::gguf_file::{
        packed_offsets, write, write_header, HeaderTensor, Value,
    };
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    use lloom_tensor::Tensor;
    use std::io::{Seek, SeekFrom, Write};
    let dev = Device::Cpu;
    let a = QTensor::quantize(
        &Tensor::arange(0f32, 15.0, &dev)?.reshape((3, 5))?,
        GgmlDType::F32,
    )?;
    let b = QTensor::quantize(
        &(Tensor::arange(0f32, 128.0, &dev)?.reshape((2, 64))? / 7.0)?,
        GgmlDType::Q8_0,
    )?;
    let md_v = Value::U32(7);
    let md = [("general.test", &md_v)];
    let mut whole = Cursor::new(Vec::new());
    write(&mut whole, &md, &[("a", &a), ("b", &b)])?;
    let whole = whole.into_inner();

    let offsets = packed_offsets([a.storage_size_in_bytes(), b.storage_size_in_bytes()]);
    assert_eq!(offsets, vec![0, 64], "15 f32 = 60 B, padded to 64");
    let entries = [
        HeaderTensor {
            name: "a",
            dims: a.shape().dims(),
            dtype: GgmlDType::F32,
            offset: offsets[0],
        },
        HeaderTensor {
            name: "b",
            dims: b.shape().dims(),
            dtype: GgmlDType::Q8_0,
            offset: offsets[1],
        },
    ];
    let mut split = Cursor::new(Vec::new());
    let start = write_header(&mut split, &md, &entries)?;
    for (t, off) in [&a, &b].iter().zip(&offsets) {
        split.seek(SeekFrom::Start(start + off))?;
        split.write_all(&t.data()?)?;
    }
    let len = whole.len();
    let mut split = split.into_inner();
    split.resize(len, 0);
    assert_eq!(split, whole, "header + placed data is `write`'s file");

    let content = Content::read(&mut Cursor::new(&whole))?;
    assert_eq!(content.tensor_data_offset, start);
    assert_eq!(content.tensor_infos["b"].offset, 64);
    let back = content.tensor(&mut Cursor::new(&whole), "b", &dev)?;
    assert_eq!(back.data()?, b.data()?);
    Ok(())
}

// All adversarial geometry below is metadata only. No huge tensor, sparse file,
// native device or raw-data conversion is needed to exercise these guards.
fn tensor_header(dims: &[u64], dtype: u32, offset: u64) -> Vec<u8> {
    let mut buf = header(1, 0);
    buf.extend(length_prefixed(b"t"));
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dim in dims.iter().rev() {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
    buf
}

fn alignment_header(alignment: u32) -> Vec<u8> {
    let mut buf = header(0, 1);
    buf.extend(length_prefixed(b"general.alignment"));
    buf.extend_from_slice(&4u32.to_le_bytes()); // U32
    buf.extend_from_slice(&alignment.to_le_bytes());
    buf
}

#[test]
fn rejects_zero_alignment_preserves_positive_non_power_of_two() {
    assert_rejects(alignment_header(0), "alignment must be nonzero");
    for alignment in [1, 8, 24, 32] {
        let buf = alignment_header(alignment);
        let expected = (buf.len() as u64).div_ceil(alignment as u64) * alignment as u64;
        let content = Content::read(&mut Cursor::new(buf)).unwrap();
        assert_eq!(content.tensor_data_offset, expected);
    }
}

#[test]
fn rejects_tensor_element_and_byte_overflow_before_payload() {
    // On 32-bit these dimensions themselves do not fit Shape; on 64-bit
    // parsing reaches the checked u64 product/byte calculations.
    #[cfg(target_pointer_width = "64")]
    {
        assert_rejects(
            tensor_header(&[u64::MAX, 2], 0, 0),
            "element count overflow",
        );
        assert_rejects(
            tensor_header(&[u64::MAX / 4 + 1], 0, 0),
            "byte size overflow",
        );
    }
    #[cfg(target_pointer_width = "32")]
    assert_rejects(tensor_header(&[u32::MAX as u64 + 1], 0, 0), "dimension");
}

#[test]
fn large_header_remains_inspectable_but_materialization_refuses() {
    // Each dimension fits usize on both 32/64-bit; bytes exceed isize::MAX
    // but fit u64 including header offset. No tensor payload is present.
    let dims = [(isize::MAX as u64) / 4 + 1];
    let mut cursor = Cursor::new(tensor_header(&dims, 0, 0));
    let content = Content::read(&mut cursor).expect("inspection is not allocation");
    let err = content
        .tensor(&mut cursor, "t", &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(err.contains("host allocation limit"), "{err}");
}

#[test]
fn preserves_scalar_zero_axis_and_header_only_tensor_metadata() {
    // Existing header inspection accepts these shapes. Materializing a scalar
    // QTensor is a separate unsupported operation; do not convert these here.
    for dims in [vec![], vec![0], vec![2, 0, 3], vec![1024]] {
        let mut cursor = Cursor::new(tensor_header(&dims, 0, 0));
        let content = Content::read(&mut cursor).unwrap();
        assert_eq!(
            content.tensor_infos["t"].shape.dims(),
            dims.iter().map(|&d| d as usize).collect::<Vec<_>>()
        );
    }
}

#[test]
fn rejects_tensor_absolute_start_and_end_overflow() {
    assert_rejects(
        tensor_header(&[1], 0, u64::MAX),
        "byte range offset overflow",
    );
    // This header is 57 bytes, aligned to 64. Start = MAX-2 is representable,
    // but adding the four-byte F32 payload is not.
    assert_eq!(tensor_header(&[1], 0, 0).len(), 57);
    assert_rejects(
        tensor_header(&[1], 0, u64::MAX - 66),
        "byte range offset overflow",
    );
}

#[test]
fn direct_tensor_info_read_revalidates_geometry_and_file_range() {
    use lloom_tensor::quantized::{gguf_file::TensorInfo, GgmlDType};
    for (dims, offset, base, expected) in [
        (vec![usize::MAX, 2], 0, 0, "element count overflow"),
        (vec![usize::MAX / 4 + 1], 0, 0, "byte size"),
        (vec![1], u64::MAX, 1, "start offset overflow"),
        (vec![0], 1, 0, "exceeds file size"),
        (vec![1], 0, 0, "remaining"),
    ] {
        let info = TensorInfo {
            ggml_dtype: GgmlDType::F32,
            shape: dims.into(),
            offset,
        };
        let err = info
            .read(&mut Cursor::new(Vec::<u8>::new()), base, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "expected {expected}, got {err}");
    }
}

#[test]
fn rejects_alignment_roundup_overflow_with_tiny_virtual_header() {
    use std::io::{Read, Seek, SeekFrom};
    // A positioned reader can start near u64::MAX without storing a giant
    // prefix. All reads consume only the ordinary 24-byte empty header.
    struct Positioned {
        bytes: Cursor<Vec<u8>>,
        base: u64,
    }
    impl Read for Positioned {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            self.bytes.read(out)
        }
    }
    impl Seek for Positioned {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            let relative = match pos {
                SeekFrom::Start(n) => SeekFrom::Start(
                    n.checked_sub(self.base)
                        .ok_or_else(|| std::io::Error::other("before virtual header"))?,
                ),
                other => other,
            };
            self.bytes
                .seek(relative)?
                .checked_add(self.base)
                .ok_or_else(|| std::io::Error::other("virtual position overflow"))
        }
    }
    let mut reader = Positioned {
        bytes: Cursor::new(header(0, 0)),
        base: u64::MAX - 31,
    };
    let err = Content::read(&mut reader).unwrap_err().to_string();
    assert!(err.contains("aligned tensor data offset overflow"), "{err}");
}

/// Seek-only virtual extents prove preflight does not read payload or construct
/// Tensor storage. Extra eager entries are checked even if no model asks for them.
#[test]
fn eager_payload_preflight_checks_extras_without_reading_or_allocating_tensors() {
    use lloom_tensor::quantized::gguf_file::{TensorInfo, VersionedMagic};
    use lloom_tensor::quantized::GgmlDType;
    use lloom_tensor::Shape;
    use std::collections::HashMap;
    use std::io::{Seek, SeekFrom};
    struct Extent(u64);
    impl Seek for Extent {
        fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
            assert_eq!(at, SeekFrom::End(0), "preflight only observes extent");
            Ok(self.0)
        }
    }
    let mut content = Content {
        magic: VersionedMagic::GgufV3,
        metadata: HashMap::new(),
        tensor_infos: HashMap::from([
            (
                "required".into(),
                TensorInfo {
                    shape: Shape::from(4),
                    ggml_dtype: GgmlDType::F32,
                    offset: 0,
                },
            ),
            (
                "extra".into(),
                TensorInfo {
                    shape: Shape::from(4),
                    ggml_dtype: GgmlDType::F32,
                    offset: 48,
                },
            ),
        ]),
        tensor_data_offset: 0,
    };
    content.validate_payloads(&mut Extent(64)).unwrap();
    for dtype in [GgmlDType::Q8_1, GgmlDType::Q8K] {
        let extra = content.tensor_infos.get_mut("extra").unwrap();
        extra.ggml_dtype = dtype;
        extra.shape = Shape::from(dtype.block_size());
        let error = content
            .validate_payloads(&mut Extent(u64::MAX))
            .unwrap_err()
            .to_string();
        assert!(error.contains("not supported yet"), "{error}");
        // The direct shared reader must select the same refusal before it can
        // inspect these deliberately empty raw bytes or attempt conversion.
        let error = lloom_tensor::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &[],
            vec![dtype.block_size()],
            &Device::Cpu,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not supported yet"), "{error}");
    }
    let extra = content.tensor_infos.get_mut("extra").unwrap();
    extra.ggml_dtype = GgmlDType::F32;
    extra.shape = Shape::from(4);

    let error = content
        .validate_payloads(&mut Extent(63))
        .unwrap_err()
        .to_string();
    assert!(error.contains("only 15 remaining"), "{error}");
    // Preserve zero-element header semantics at EOF, not a blanket positive cap.
    content.tensor_infos.get_mut("extra").unwrap().shape = Shape::from(0);
    content.tensor_infos.get_mut("extra").unwrap().offset = 64;
    content.validate_payloads(&mut Extent(64)).unwrap();
    content.tensor_data_offset = 1;
    content.tensor_infos.get_mut("extra").unwrap().offset = u64::MAX;
    assert!(content
        .validate_payloads(&mut Extent(u64::MAX))
        .unwrap_err()
        .to_string()
        .contains("offset overflow"));
    content.tensor_data_offset = 0;
    let extra = content.tensor_infos.get_mut("extra").unwrap();
    extra.offset = 0;
    extra.shape = Shape::from(31);
    extra.ggml_dtype = GgmlDType::Q4_0;
    assert!(content
        .validate_payloads(&mut Extent(64))
        .unwrap_err()
        .to_string()
        .contains("not divisible"));
    let extra = content.tensor_infos.get_mut("extra").unwrap();
    extra.shape = Shape::from(isize::MAX as usize / 4 + 1);
    extra.ggml_dtype = GgmlDType::F32;
    assert!(content
        .validate_payloads(&mut Extent(u64::MAX))
        .unwrap_err()
        .to_string()
        .contains("host allocation limit"));
}
