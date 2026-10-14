//! Every `Error` variant's `Display` text, pinned to the strings the enum
//! printed before its cold payloads were boxed, so where a payload lives
//! cannot change what a caller reads. `Debug` is `Display` (`error.rs`).

use lloom_tensor::memory_admission::{AllocationError, RefusalDomain};
use lloom_tensor::{DType, DeviceLocation, Error, Shape};

fn s(dims: &[usize]) -> Shape {
    Shape::from(dims.to_vec())
}

/// The variant's name. No wildcard: a variant added to `Error` fails to
/// compile here until it is named, and then fails the test until it has a
/// fixture and a pinned text.
fn variant(e: &Error) -> &'static str {
    match e {
        Error::UnexpectedDType { .. } => "UnexpectedDType",
        Error::DTypeMismatchBinaryOp { .. } => "DTypeMismatchBinaryOp",
        Error::UnsupportedDTypeForOp(..) => "UnsupportedDTypeForOp",
        Error::DimOutOfRange(_) => "DimOutOfRange",
        Error::DuplicateDimIndex(_) => "DuplicateDimIndex",
        Error::UnexpectedNumberOfDims { .. } => "UnexpectedNumberOfDims",
        Error::UnexpectedShape(_) => "UnexpectedShape",
        Error::ShapeMismatch { .. } => "ShapeMismatch",
        Error::ShapeMismatchBinaryOp(_) => "ShapeMismatchBinaryOp",
        Error::ShapeMismatchCat(_) => "ShapeMismatchCat",
        Error::ShapeMismatchSplit { .. } => "ShapeMismatchSplit",
        Error::OnlySingleDimension { .. } => "OnlySingleDimension",
        Error::EmptyTensor { .. } => "EmptyTensor",
        Error::DeviceMismatchBinaryOp(_) => "DeviceMismatchBinaryOp",
        Error::NarrowInvalidArgs(_) => "NarrowInvalidArgs",
        Error::Conv1dInvalidArgs(_) => "Conv1dInvalidArgs",
        Error::InvalidIndex { .. } => "InvalidIndex",
        Error::BroadcastIncompatibleShapes(_) => "BroadcastIncompatibleShapes",
        Error::CannotSetVar { .. } => "CannotSetVar",
        Error::MatMulUnexpectedStriding(_) => "MatMulUnexpectedStriding",
        Error::RequiresContiguous { .. } => "RequiresContiguous",
        Error::OpRequiresAtLeastOneTensor { .. } => "OpRequiresAtLeastOneTensor",
        Error::OpRequiresAtLeastTwoTensors { .. } => "OpRequiresAtLeastTwoTensors",
        Error::BackwardNotSupported { .. } => "BackwardNotSupported",
        Error::NotCompiledWithCudaSupport => "NotCompiledWithCudaSupport",
        Error::NotCompiledWithMetalSupport => "NotCompiledWithMetalSupport",
        Error::CannotFindTensor { .. } => "CannotFindTensor",
        Error::Cuda(_) => "Cuda",
        Error::Metal(_) => "Metal",
        Error::Allocation(_) => "Allocation",
        #[cfg(feature = "ug")]
        Error::Ug(_) => "Ug",
        Error::TryFromIntError(_) => "TryFromIntError",
        Error::Npy(_) => "Npy",
        Error::Zip(_) => "Zip",
        Error::ParseInt(_) => "ParseInt",
        Error::FromUtf8(_) => "FromUtf8",
        Error::Io(_) => "Io",
        Error::SafeTensor(_) => "SafeTensor",
        Error::UnsupportedSafeTensorDtype(_) => "UnsupportedSafeTensorDtype",
        Error::Wrapped(_) => "Wrapped",
        Error::WrappedContext { .. } => "WrappedContext",
        Error::Context { .. } => "Context",
        Error::WithPath { .. } => "WithPath",
        Error::WithBacktrace { .. } => "WithBacktrace",
        Error::Msg(_) => "Msg",
        Error::UnwrapNone => "UnwrapNone",
    }
}

/// Every variant but `Ug` (behind the `ug` feature, which this tree does not
/// build), with fixed payloads.
fn every_variant() -> Vec<(&'static str, Error)> {
    #[allow(unused_mut)]
    let mut all: Vec<(&str, Error)> = vec![
        (
            "UnexpectedDType",
            Error::UnexpectedDType {
                msg: "m",
                expected: DType::F16,
                got: DType::BF16,
            },
        ),
        (
            "DTypeMismatchBinaryOp",
            Error::DTypeMismatchBinaryOp {
                lhs: DType::F32,
                rhs: DType::U8,
                op: "add",
            },
        ),
        (
            "UnsupportedDTypeForOp",
            Error::UnsupportedDTypeForOp(DType::I64, "erf"),
        ),
        (
            "DimOutOfRange",
            Error::DimOutOfRange(Box::new(lloom_tensor::error::DimOutOfRange {
                shape: s(&[2, 3]),
                dim: -4,
                op: "narrow",
            })),
        ),
        (
            "DuplicateDimIndex",
            Error::DuplicateDimIndex(Box::new(lloom_tensor::error::DuplicateDimIndex {
                shape: s(&[2, 3, 4]),
                dims: vec![1, 1],
                op: "permute",
            })),
        ),
        (
            "UnexpectedNumberOfDims",
            Error::UnexpectedNumberOfDims {
                expected: 3,
                got: 2,
                shape: s(&[5, 6]),
            },
        ),
        (
            "UnexpectedShape",
            Error::UnexpectedShape(Box::new(lloom_tensor::error::UnexpectedShape {
                msg: "weight".to_string(),
                expected: s(&[4]),
                got: s(&[4, 1]),
            })),
        ),
        (
            "ShapeMismatch",
            Error::ShapeMismatch {
                buffer_size: 7,
                shape: s(&[8]),
            },
        ),
        (
            "ShapeMismatchBinaryOp",
            Error::ShapeMismatchBinaryOp(Box::new(lloom_tensor::error::ShapeMismatchBinaryOp {
                lhs: s(&[1, 2]),
                rhs: s(&[3]),
                op: "mul",
            })),
        ),
        (
            "ShapeMismatchCat",
            Error::ShapeMismatchCat(Box::new(lloom_tensor::error::ShapeMismatchCat {
                dim: 1,
                first_shape: s(&[2, 2]),
                n: 3,
                nth_shape: s(&[3, 3]),
            })),
        ),
        (
            "ShapeMismatchSplit",
            Error::ShapeMismatchSplit {
                shape: s(&[9]),
                dim: 0,
                n_parts: 2,
            },
        ),
        (
            "OnlySingleDimension",
            Error::OnlySingleDimension {
                op: "squeeze",
                dims: vec![0, 2],
            },
        ),
        ("EmptyTensor", Error::EmptyTensor { op: "max" }),
        (
            "DeviceMismatchBinaryOp",
            Error::DeviceMismatchBinaryOp(Box::new(lloom_tensor::error::DeviceMismatchBinaryOp {
                lhs: DeviceLocation::Cpu,
                rhs: DeviceLocation::Metal { gpu_id: 1 },
                op: "where",
            })),
        ),
        (
            "NarrowInvalidArgs",
            Error::NarrowInvalidArgs(Box::new(lloom_tensor::error::NarrowInvalidArgs {
                shape: s(&[4, 5]),
                dim: 1,
                start: 3,
                len: 4,
                msg: "start + len > dim_len",
            })),
        ),
        (
            "Conv1dInvalidArgs",
            Error::Conv1dInvalidArgs(Box::new(lloom_tensor::error::Conv1dInvalidArgs {
                inp_shape: s(&[1, 2, 3]),
                k_shape: s(&[4, 2, 5]),
                padding: 6,
                stride: 7,
                msg: "kernel too large",
            })),
        ),
        (
            "InvalidIndex",
            Error::InvalidIndex {
                op: "index-select",
                index: 9,
                size: 4,
            },
        ),
        (
            "BroadcastIncompatibleShapes",
            Error::BroadcastIncompatibleShapes(Box::new(
                lloom_tensor::error::BroadcastIncompatibleShapes {
                    src_shape: s(&[2, 3]),
                    dst_shape: s(&[4, 3, 2]),
                },
            )),
        ),
        (
            "CannotSetVar",
            Error::CannotSetVar {
                msg: "shape mismatch",
            },
        ),
        (
            "MatMulUnexpectedStriding",
            Error::MatMulUnexpectedStriding(Box::new(
                lloom_tensor::error::MatMulUnexpectedStriding {
                    lhs_l: lloom_tensor::Layout::contiguous(s(&[2, 3])),
                    rhs_l: lloom_tensor::Layout::contiguous(s(&[3, 4])),
                    bmnk: (1, 2, 4, 3),
                    msg: "non-contiguous lhs",
                },
            )),
        ),
        (
            "RequiresContiguous",
            Error::RequiresContiguous { op: "copy2d" },
        ),
        (
            "OpRequiresAtLeastOneTensor",
            Error::OpRequiresAtLeastOneTensor { op: "cat" },
        ),
        (
            "OpRequiresAtLeastTwoTensors",
            Error::OpRequiresAtLeastTwoTensors { op: "stack" },
        ),
        (
            "BackwardNotSupported",
            Error::BackwardNotSupported { op: "argmax" },
        ),
        (
            "NotCompiledWithCudaSupport",
            Error::NotCompiledWithCudaSupport,
        ),
        (
            "NotCompiledWithMetalSupport",
            Error::NotCompiledWithMetalSupport,
        ),
        (
            "CannotFindTensor",
            Error::CannotFindTensor {
                path: "model.embed.weight".to_string(),
            },
        ),
        (
            "Cuda",
            Error::Cuda(Box::new(std::io::Error::other("cuda fixture"))),
        ),
        (
            "Metal.Message",
            Error::Metal(lloom_tensor::MetalError::Message(
                "metal fixture".to_string(),
            )),
        ),
        (
            "Allocation.Refused",
            Error::Allocation(AllocationError::Refused {
                requested: 17,
                available: 16,
                domain: RefusalDomain::Promise,
            }),
        ),
        (
            "Allocation.ReuseDomainMismatch",
            Error::Allocation(AllocationError::ReuseDomainMismatch {
                expected: "a",
                actual: "b",
            }),
        ),
        (
            "Allocation.ConfigurationConflict",
            Error::Allocation(AllocationError::ConfigurationConflict {
                active: Some(1),
                requested: None,
            }),
        ),
        (
            "TryFromIntError",
            Error::TryFromIntError(u8::try_from(300u32).unwrap_err()),
        ),
        ("Npy", Error::Npy("npy fixture".to_string())),
        ("Zip", Error::Zip(zip::result::ZipError::FileNotFound)),
        ("ParseInt", Error::ParseInt("x".parse::<u32>().unwrap_err())),
        (
            "FromUtf8",
            Error::FromUtf8(String::from_utf8(vec![0xff, 0x41]).unwrap_err()),
        ),
        (
            "Io",
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "io fixture",
            )),
        ),
        (
            "SafeTensor",
            Error::SafeTensor(safetensors::SafeTensorError::HeaderTooLarge),
        ),
        (
            "UnsupportedSafeTensorDtype",
            Error::UnsupportedSafeTensorDtype(safetensors::Dtype::BOOL),
        ),
        ("Wrapped", Error::Wrapped(Box::new("wrapped fixture"))),
        (
            "WrappedContext",
            Error::WrappedContext {
                wrapped: Box::new(std::io::Error::other("inner io")),
                context: "while loading".to_string(),
            },
        ),
        (
            "Context",
            Error::Context {
                inner: Box::new(Error::UnwrapNone),
                context: Box::new("ctx fixture"),
            },
        ),
        (
            "WithPath",
            Error::WithPath {
                inner: Box::new(Error::Msg("inner msg".to_string())),
                path: std::path::PathBuf::from("/tmp/x.safetensors"),
            },
        ),
        (
            "WithBacktrace",
            Error::WithBacktrace {
                inner: Box::new(Error::Msg("bt inner".to_string())),
                backtrace: Box::new(std::backtrace::Backtrace::disabled()),
            },
        ),
        ("Msg", Error::Msg("msg fixture".to_string())),
        ("UnwrapNone", Error::UnwrapNone),
    ];
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::{err::CommandLifecycleError, MetalKernelError};
        use lloom_tensor::MetalError;
        all.push((
            "Metal.UnexpectedDType",
            Error::Metal(MetalError::UnexpectedDType {
                msg: "mm",
                expected: DType::F16,
                got: DType::F32,
            }),
        ));
        all.push((
            "Metal.Kernel.Allocation",
            Error::Metal(MetalError::KernelError(MetalKernelError::Allocation(
                AllocationError::DeviceOom { requested: 4096 },
            ))),
        ));
        all.push((
            "Metal.Kernel.RecordingAdmission",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::RecordingAdmission(CommandLifecycleError::Busy),
            )),
        ));
        all.push((
            "Metal.Kernel.QueueUnavailable",
            Error::Metal(MetalError::KernelError(MetalKernelError::QueueUnavailable(
                "closed",
            ))),
        ));
        all.push((
            "Metal.Kernel.CommandBufferError",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::CommandBufferError("cb fixture".to_string()),
            )),
        ));
        all.push((
            "Metal.Kernel.UnsupportedDTypeForOp",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::UnsupportedDTypeForOp("f64", "gemv"),
            )),
        ));
        all.push((
            "Metal.Kernel.MatMulNonContiguous",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::MatMulNonContiguous(Box::new(
                    lloom_tensor::kernels::err::MatMulNonContiguous {
                        lhs_stride: vec![3, 1],
                        rhs_stride: vec![1, 4],
                        mnk: (2, 4, 3),
                    },
                )),
            )),
        ));
        all.push((
            "Metal.Kernel.SdpaHeadSizeMismatch",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                    lloom_tensor::kernels::err::SdpaHeadSizeMismatch {
                        variation: "vector",
                        got: 80,
                        expected: vec![64, 128],
                    },
                )),
            )),
        ));
        all.push((
            "Metal.Kernel.FailedToCreatePipeline",
            Error::Metal(MetalError::KernelError(
                MetalKernelError::FailedToCreatePipeline("p".to_string()),
            )),
        ));
    }
    all
}

/// What each fixture printed on the enum before the boxing (`42ae34386`).
const EXPECTED: &[(&str, &str)] = &[
    ("UnexpectedDType", "m, expected: F16, got: BF16"),
    ("DTypeMismatchBinaryOp", "dtype mismatch in add, lhs: F32, rhs: U8"),
    ("UnsupportedDTypeForOp", "unsupported dtype I64 for op erf"),
    ("DimOutOfRange", "narrow: dimension index -4 out of range for shape [2, 3]"),
    ("DuplicateDimIndex", "permute: duplicate dim index [1, 1] for shape [2, 3, 4]"),
    ("UnexpectedNumberOfDims", "unexpected rank, expected: 3, got: 2 ([5, 6])"),
    ("UnexpectedShape", "weight, expected: [4], got: [4, 1]"),
    ("ShapeMismatch", "Shape mismatch, got buffer of size 7 which is compatible with shape [8]"),
    ("ShapeMismatchBinaryOp", "shape mismatch in mul, lhs: [1, 2], rhs: [3]"),
    ("ShapeMismatchCat", "shape mismatch in cat for dim 1, shape for arg 1: [2, 2] shape for arg 3: [3, 3]"),
    ("ShapeMismatchSplit", "Cannot divide tensor of shape [9] equally along dim 0 into 2"),
    ("OnlySingleDimension", "squeeze can only be performed on a single dimension"),
    ("EmptyTensor", "empty tensor for max"),
    ("DeviceMismatchBinaryOp", "device mismatch in where, lhs: Cpu, rhs: Metal { gpu_id: 1 }"),
    ("NarrowInvalidArgs", "narrow invalid args start + len > dim_len: [4, 5], dim: 1, start: 3, len:4"),
    ("Conv1dInvalidArgs", "conv1d invalid args kernel too large: inp: [1, 2, 3], k: [4, 2, 5], pad: 6, stride: 7"),
    ("InvalidIndex", "index-select invalid index 9 with dim size 4"),
    ("BroadcastIncompatibleShapes", "cannot broadcast [2, 3] to [4, 3, 2]"),
    ("CannotSetVar", "cannot set variable shape mismatch"),
    ("MatMulUnexpectedStriding", "MatMulUnexpectedStriding { lhs_l: Layout { shape: [2, 3], stride: [3, 1], start_offset: 0 }, rhs_l: Layout { shape: [3, 4], stride: [4, 1], start_offset: 0 }, bmnk: (1, 2, 4, 3), msg: \"non-contiguous lhs\" }"),
    ("RequiresContiguous", "copy2d only supports contiguous tensors"),
    ("OpRequiresAtLeastOneTensor", "cat expects at least one tensor"),
    ("OpRequiresAtLeastTwoTensors", "stack expects at least two tensors"),
    ("BackwardNotSupported", "backward is not supported for argmax"),
    ("NotCompiledWithCudaSupport", "the candle crate has not been built with cuda support"),
    ("NotCompiledWithMetalSupport", "the candle crate has not been built with metal support"),
    ("CannotFindTensor", "cannot find tensor model.embed.weight"),
    ("Cuda", "cuda fixture"),
    ("Metal.Message", "Metal error metal fixture"),
    ("Allocation.Refused", "Promise memory refused 17 bytes: 16 bytes available"),
    ("Allocation.ReuseDomainMismatch", "reusable allocation domain mismatch: container=a, active payer=b; accounted domains require identical origin"),
    ("Allocation.ConfigurationConflict", "process memory cap already initialized to Some(1), requested None; configure before the first load"),
    ("TryFromIntError", "out of range integral type conversion attempted"),
    ("Npy", "npy/npz error npy fixture"),
    ("Zip", "specified file not found in archive"),
    ("ParseInt", "invalid digit found in string"),
    ("FromUtf8", "invalid utf-8 sequence of 1 bytes from index 0"),
    ("Io", "io fixture"),
    ("SafeTensor", "header too large"),
    ("UnsupportedSafeTensorDtype", "unsupported safetensor dtype BOOL"),
    ("Wrapped", "wrapped fixture"),
    ("WrappedContext", "Custom { kind: Other, error: \"inner io\" }\n\"while loading\""),
    ("Context", "ctx fixture\nunwrap none"),
    ("WithPath", "path: \"/tmp/x.safetensors\" inner msg"),
    ("WithBacktrace", "bt inner\ndisabled backtrace"),
    ("Msg", "msg fixture"),
    ("UnwrapNone", "unwrap none"),
    ("Metal.UnexpectedDType", "Metal error mm, expected: F16, got: F32"),
    ("Metal.Kernel.Allocation", "Metal error the device refused a 4096-byte allocation"),
    ("Metal.Kernel.RecordingAdmission", "Metal error command buffer recording admission refused: recording operation currently busy"),
    ("Metal.Kernel.QueueUnavailable", "Metal error GPU queue unavailable: closed"),
    ("Metal.Kernel.CommandBufferError", "Metal error Command buffer had following error: cb fixture"),
    ("Metal.Kernel.UnsupportedDTypeForOp", "Metal error Unsupported dtype f64 for operation gemv"),
    ("Metal.Kernel.MatMulNonContiguous", "Metal error Invalid matmul arguments [3, 1] [1, 4] (2, 4, 3)"),
    ("Metal.Kernel.SdpaHeadSizeMismatch", "Metal error Sdpa vector head size was 80, expectd [64, 128]"),
    ("Metal.Kernel.FailedToCreatePipeline", "Metal error Failed to create pipeline"),
];

#[test]
fn every_variant_displays_as_before() {
    let all = every_variant();
    let mut named = std::collections::BTreeSet::new();
    for (name, e) in &all {
        named.insert(variant(e));
        let expected = EXPECTED
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} has no pinned text"))
            .1;
        assert_eq!(e.to_string(), expected, "{name}: Display moved");
        assert_eq!(format!("{e:?}"), expected, "{name}: Debug is not Display");
    }
    let listed = all.len();
    #[cfg(feature = "metal")]
    assert_eq!(listed, EXPECTED.len(), "a pinned text has no fixture");
    assert!(listed > 40, "only {listed} fixtures");
    assert_eq!(named.len(), 45, "variants without a fixture: {named:?}");
}
