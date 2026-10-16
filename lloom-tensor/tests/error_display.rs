//! Every `Error` variant's `Display` text, and through it every
//! `AllocationError`, `MetalError` and `MetalKernelError` variant's, pinned to
//! the strings the enums printed before their cold payloads were boxed and
//! `AllocationError` was packed, so where a payload lives cannot change what a
//! caller reads. `Debug` is `Display` (`error.rs`).

use lloom_tensor::memory_admission::{AllocationError, RefusalDomain, ReuseDomains};
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
        Error::UnexpectedNumberOfDims(_) => "UnexpectedNumberOfDims",
        Error::UnexpectedShape(_) => "UnexpectedShape",
        Error::ShapeMismatch(_) => "ShapeMismatch",
        Error::ShapeMismatchBinaryOp(_) => "ShapeMismatchBinaryOp",
        Error::ShapeMismatchCat(_) => "ShapeMismatchCat",
        Error::ShapeMismatchSplit(_) => "ShapeMismatchSplit",
        Error::OnlySingleDimension(_) => "OnlySingleDimension",
        Error::EmptyTensor { .. } => "EmptyTensor",
        Error::DeviceMismatchBinaryOp(_) => "DeviceMismatchBinaryOp",
        Error::NarrowInvalidArgs(_) => "NarrowInvalidArgs",
        Error::Conv1dInvalidArgs(_) => "Conv1dInvalidArgs",
        Error::InvalidIndex(_) => "InvalidIndex",
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
        Error::WrappedContext(_) => "WrappedContext",
        Error::Context { .. } => "Context",
        Error::WithPath(_) => "WithPath",
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
            Error::UnexpectedNumberOfDims(Box::new(lloom_tensor::error::UnexpectedNumberOfDims {
                expected: 3,
                got: 2,
                shape: s(&[5, 6]),
            })),
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
            Error::ShapeMismatch(Box::new(lloom_tensor::error::ShapeMismatch {
                buffer_size: 7,
                shape: s(&[8]),
            })),
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
            Error::ShapeMismatchSplit(Box::new(lloom_tensor::error::ShapeMismatchSplit {
                shape: s(&[9]),
                dim: 0,
                n_parts: 2,
            })),
        ),
        (
            "OnlySingleDimension",
            Error::OnlySingleDimension(Box::new(lloom_tensor::error::OnlySingleDimension {
                op: "squeeze",
                dims: vec![0, 2],
            })),
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
            Error::InvalidIndex(Box::new(lloom_tensor::error::InvalidIndex {
                op: "index-select",
                index: 9,
                size: 4,
            })),
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
                domains: &ReuseDomains {
                    expected: "a",
                    actual: "b",
                },
            }),
        ),
        (
            "Allocation.ConfigurationConflict",
            Error::Allocation(AllocationError::configuration_conflict(Some(1), None)),
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
            Error::from(String::from_utf8(vec![0xff, 0x41]).unwrap_err()),
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
            Error::from(safetensors::SafeTensorError::HeaderTooLarge),
        ),
        (
            "UnsupportedSafeTensorDtype",
            Error::UnsupportedSafeTensorDtype(safetensors::Dtype::BOOL),
        ),
        ("Wrapped", Error::Wrapped(Box::new("wrapped fixture"))),
        (
            "WrappedContext",
            Error::WrappedContext(Box::new(lloom_tensor::error::WrappedContext {
                wrapped: Box::new(std::io::Error::other("inner io")),
                context: "while loading".to_string(),
            })),
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
            Error::Msg("inner msg".to_string()).with_path("/tmp/x.safetensors"),
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
                MetalKernelError::UnsupportedDTypeForOp(Box::new(
                    lloom_tensor::kernels::err::UnsupportedDTypeForOp {
                        dtype: "f64",
                        op: "gemv",
                    },
                )),
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
    all.extend(nested_fixtures());
    all
}

/// One value of every `AllocationError`, `MetalError` and `MetalKernelError`
/// variant the fixtures above leave out, every `ConfigurationConflict` cap
/// presence, and the three origin pairs `AllocationOrigin::validate_reuse`
/// refuses, built through it.
fn nested_fixtures() -> Vec<(&'static str, Error)> {
    use lloom_tensor::memory_admission::{
        process_account, AllocationOrigin, MachineFailure, PredictionBasis,
    };
    let a = Error::Allocation;
    let domain =
        || AllocationOrigin::Domain(process_account().unadmitted_domain(Some(8)).unwrap().key());
    let (one, two, legacy) = (domain(), domain(), AllocationOrigin::Legacy);
    let reuse = |container: &AllocationOrigin, payer: &AllocationOrigin| {
        Error::Allocation(container.validate_reuse(payer).unwrap_err())
    };
    #[allow(unused_mut)]
    let mut all = vec![
        (
            "Allocation.CountExhausted",
            a(AllocationError::CountExhausted),
        ),
        ("Allocation.Cancelled", a(AllocationError::Cancelled)),
        (
            "Allocation.DeviceOom",
            a(AllocationError::DeviceOom { requested: 4096 }),
        ),
        (
            "Allocation.HostAllocation",
            a(AllocationError::HostAllocation),
        ),
        ("Allocation.Overflow", a(AllocationError::Overflow)),
        (
            "Allocation.Geometry",
            a(AllocationError::Geometry("geometry")),
        ),
        (
            "Allocation.PredictionExceeded",
            a(AllocationError::PredictionExceeded {
                predicted: 1,
                actual: 2,
                basis: PredictionBasis::HeapDescriptorSizeEstimate,
            }),
        ),
        (
            "Allocation.Unsupported",
            a(AllocationError::Unsupported("unsupported")),
        ),
        (
            "Allocation.Retirement",
            a(AllocationError::Retirement("retirement")),
        ),
        (
            "Allocation.InvalidCustody",
            a(AllocationError::InvalidCustody),
        ),
        (
            "Allocation.UnsupportedLayout",
            a(AllocationError::UnsupportedLayout),
        ),
        (
            "Allocation.DifferentDomain",
            a(AllocationError::DifferentDomain),
        ),
        (
            "Allocation.DifferentAllocationDevice",
            a(AllocationError::DifferentAllocationDevice),
        ),
        (
            "Allocation.UnboundAllocationDevice",
            a(AllocationError::UnboundAllocationDevice),
        ),
        (
            "Allocation.Machine.Unavailable",
            a(AllocationError::Machine(MachineFailure::Unavailable)),
        ),
        (
            "Allocation.Machine.Binding",
            a(AllocationError::Machine(MachineFailure::Binding)),
        ),
        (
            "Allocation.ConfigurationConflict.none-none",
            a(AllocationError::configuration_conflict(None, None)),
        ),
        (
            "Allocation.ConfigurationConflict.none-some",
            a(AllocationError::configuration_conflict(None, Some(8))),
        ),
        (
            "Allocation.ConfigurationConflict.some-some",
            a(AllocationError::configuration_conflict(
                Some(4),
                Some(usize::MAX),
            )),
        ),
        (
            "Allocation.ReuseDomainMismatch.domain-domain",
            reuse(&one, &two),
        ),
        (
            "Allocation.ReuseDomainMismatch.domain-legacy",
            reuse(&one, &legacy),
        ),
        (
            "Allocation.ReuseDomainMismatch.legacy-domain",
            reuse(&legacy, &one),
        ),
    ];
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::kernels::SdpaDType;
        use lloom_tensor::kernels::MetalKernelError as K;
        use lloom_tensor::metal_backend::LockError;
        use lloom_tensor::MetalError;
        let k = |e: K| Error::Metal(MetalError::KernelError(e));
        all.extend([
            (
                "Metal.Kernel.OwnerSubmissionRequired",
                k(K::OwnerSubmissionRequired),
            ),
            (
                "Metal.Kernel.SubscriberConfigurationSealed",
                k(K::SubscriberConfigurationSealed),
            ),
            (
                "Metal.Kernel.SubscriberSetupBusy",
                k(K::SubscriberSetupBusy),
            ),
            ("Metal.Kernel.CallbackWouldBlock", k(K::CallbackWouldBlock)),
            (
                "Metal.Kernel.LockError",
                k(K::LockError("lock fixture".into())),
            ),
            (
                "Metal.Kernel.LoadLibraryError",
                k(K::LoadLibraryError("library fixture".into())),
            ),
            (
                "Metal.Kernel.LoadFunctionError",
                k(K::LoadFunctionError("function fixture".into())),
            ),
            (
                "Metal.Kernel.ArmRefused",
                k(K::ArmRefused("arm fixture".into())),
            ),
            (
                "Metal.Kernel.EmulatedRefused",
                k(K::EmulatedRefused("emulated fixture".into())),
            ),
            (
                "Metal.Kernel.FailedToCreateComputeFunction",
                k(K::FailedToCreateComputeFunction),
            ),
            (
                "Metal.Kernel.InvalidInput",
                k(K::InvalidInput("input fixture".into())),
            ),
            (
                "Metal.Kernel.FailedToCreateResource",
                k(K::FailedToCreateResource("resource fixture".into())),
            ),
            (
                "Metal.Kernel.SdpaHeadDTypeMismatch",
                k(K::SdpaHeadDTypeMismatch {
                    variation: "full",
                    got: SdpaDType::BF16,
                }),
            ),
            (
                "Metal.Kernel.WithBacktrace",
                k(K::WithBacktrace {
                    inner: Box::new(K::CallbackWouldBlock),
                    backtrace: Box::new(std::backtrace::Backtrace::disabled()),
                }),
            ),
            (
                "Metal.LockError.Poisoned",
                Error::Metal(MetalError::LockError(LockError::Poisoned(
                    "poison fixture".into(),
                ))),
            ),
            (
                "Metal.LockError.WouldBlock",
                Error::Metal(MetalError::LockError(LockError::WouldBlock)),
            ),
        ]);
    }
    all
}

/// The nested variant a fixture reaches, `None` for an `Error`-level one. No
/// wildcard below the `Error` level: a variant added to `AllocationError`,
/// `MetalError` or `MetalKernelError` fails to compile here until it is named,
/// and then fails the test until it has a fixture.
fn nested_variant(e: &Error) -> Option<&'static str> {
    match e {
        Error::Allocation(a) => Some(allocation_variant(a)),
        #[cfg(feature = "metal")]
        Error::Metal(m) => Some(metal_variant(m)),
        _ => None,
    }
}

fn allocation_variant(e: &AllocationError) -> &'static str {
    match e {
        AllocationError::Refused { .. } => "Allocation.Refused",
        AllocationError::CountExhausted => "Allocation.CountExhausted",
        AllocationError::Cancelled => "Allocation.Cancelled",
        AllocationError::DeviceOom { .. } => "Allocation.DeviceOom",
        AllocationError::HostAllocation => "Allocation.HostAllocation",
        AllocationError::Overflow => "Allocation.Overflow",
        AllocationError::Geometry(_) => "Allocation.Geometry",
        AllocationError::PredictionExceeded { .. } => "Allocation.PredictionExceeded",
        AllocationError::Unsupported(_) => "Allocation.Unsupported",
        AllocationError::Retirement(_) => "Allocation.Retirement",
        AllocationError::InvalidCustody => "Allocation.InvalidCustody",
        AllocationError::UnsupportedLayout => "Allocation.UnsupportedLayout",
        AllocationError::DifferentDomain => "Allocation.DifferentDomain",
        AllocationError::DifferentAllocationDevice => "Allocation.DifferentAllocationDevice",
        AllocationError::UnboundAllocationDevice => "Allocation.UnboundAllocationDevice",
        AllocationError::ReuseDomainMismatch { .. } => "Allocation.ReuseDomainMismatch",
        AllocationError::Machine(_) => "Allocation.Machine",
        AllocationError::ConfigurationConflict { .. } => "Allocation.ConfigurationConflict",
    }
}

#[cfg(feature = "metal")]
fn metal_variant(e: &lloom_tensor::MetalError) -> &'static str {
    use lloom_tensor::kernels::MetalKernelError as K;
    use lloom_tensor::MetalError as M;
    match e {
        M::Message(_) => "Metal.Message",
        M::LockError(_) => "Metal.LockError",
        M::UnexpectedDType { .. } => "Metal.UnexpectedDType",
        M::KernelError(k) => match k {
            K::Allocation(a) => allocation_variant(a),
            K::RecordingAdmission(_) => "Metal.Kernel.RecordingAdmission",
            // `CompletionRegistrationError::new` is crate-private; its text is
            // pinned by `kernels::err`'s `registration_error_tests`.
            K::CompletionRegistration(_) => "Metal.Kernel.CompletionRegistration",
            K::OwnerSubmissionRequired => "Metal.Kernel.OwnerSubmissionRequired",
            K::SubscriberConfigurationSealed => "Metal.Kernel.SubscriberConfigurationSealed",
            K::SubscriberSetupBusy => "Metal.Kernel.SubscriberSetupBusy",
            K::CallbackWouldBlock => "Metal.Kernel.CallbackWouldBlock",
            K::QueueUnavailable(_) => "Metal.Kernel.QueueUnavailable",
            K::CommandBufferError(_) => "Metal.Kernel.CommandBufferError",
            K::LockError(_) => "Metal.Kernel.LockError",
            K::LoadLibraryError(_) => "Metal.Kernel.LoadLibraryError",
            K::LoadFunctionError(_) => "Metal.Kernel.LoadFunctionError",
            K::ArmRefused(_) => "Metal.Kernel.ArmRefused",
            K::EmulatedRefused(_) => "Metal.Kernel.EmulatedRefused",
            K::UnsupportedDTypeForOp(_) => "Metal.Kernel.UnsupportedDTypeForOp",
            K::FailedToCreateComputeFunction => "Metal.Kernel.FailedToCreateComputeFunction",
            K::InvalidInput(_) => "Metal.Kernel.InvalidInput",
            K::FailedToCreateResource(_) => "Metal.Kernel.FailedToCreateResource",
            K::FailedToCreatePipeline(_) => "Metal.Kernel.FailedToCreatePipeline",
            K::MatMulNonContiguous(_) => "Metal.Kernel.MatMulNonContiguous",
            K::SdpaHeadSizeMismatch(_) => "Metal.Kernel.SdpaHeadSizeMismatch",
            K::SdpaHeadDTypeMismatch { .. } => "Metal.Kernel.SdpaHeadDTypeMismatch",
            K::WithBacktrace { .. } => "Metal.Kernel.WithBacktrace",
        },
    }
}

/// What each fixture printed before the boxing: the `Error`-level ones on
/// `42ae34386`, the nested ones (`nested_fixtures`) on `d583361c8`.
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
    ("Allocation.CountExhausted", "allocation reference count exhausted"),
    ("Allocation.Cancelled", "allocation context was cancelled"),
    ("Allocation.DeviceOom", "the device refused a 4096-byte allocation"),
    ("Allocation.HostAllocation", "cold host allocation returned null"),
    ("Allocation.Overflow", "memory admission arithmetic exceeds usize"),
    ("Allocation.Geometry", "allocation geometry: geometry"),
    ("Allocation.PredictionExceeded", "allocation prediction exceeded: 2 actual bytes > 1 predicted (HeapDescriptorSizeEstimate); raw allocation released; transient peak was not prevented"),
    ("Allocation.Unsupported", "admitted allocation unsupported: unsupported"),
    ("Allocation.Retirement", "allocation retirement registration: retirement"),
    ("Allocation.InvalidCustody", "allocation custody invariant violated"),
    ("Allocation.UnsupportedLayout", "owned Arc quote requires an audited std toolchain/source; legacy allocation APIs remain available"),
    ("Allocation.DifferentDomain", "cannot return a promise to a different model domain"),
    ("Allocation.DifferentAllocationDevice", "reusable allocation container and operation target different native devices"),
    ("Allocation.UnboundAllocationDevice", "clock-only allocation container has no native device binding"),
    ("Allocation.Machine.Unavailable", "machine admission: the registry or the host's memory could not be read; admission cannot establish room"),
    ("Allocation.Machine.Binding", "machine admission: the machine holding is absent without an explicit force/off override, or already belongs to another admission domain; reuse that domain or obtain a distinct holding with Gate::admit"),
    ("Allocation.ConfigurationConflict.none-none", "process memory cap already initialized to None, requested None; configure before the first load"),
    ("Allocation.ConfigurationConflict.none-some", "process memory cap already initialized to None, requested Some(8); configure before the first load"),
    ("Allocation.ConfigurationConflict.some-some", "process memory cap already initialized to Some(4), requested Some(18446744073709551615); configure before the first load"),
    ("Allocation.ReuseDomainMismatch.domain-domain", "reusable allocation domain mismatch: container=accounted domain, active payer=accounted domain; accounted domains require identical origin"),
    ("Allocation.ReuseDomainMismatch.domain-legacy", "reusable allocation domain mismatch: container=accounted domain, active payer=legacy/unadmitted; accounted domains require identical origin"),
    ("Allocation.ReuseDomainMismatch.legacy-domain", "reusable allocation domain mismatch: container=legacy/unadmitted, active payer=accounted domain; accounted domains require identical origin"),
    ("Metal.Kernel.OwnerSubmissionRequired", "Metal error bound command buffer requires its owning Commands submission transaction"),
    ("Metal.Kernel.SubscriberConfigurationSealed", "Metal error queue completion subscribers are sealed; register during device setup"),
    ("Metal.Kernel.SubscriberSetupBusy", "Metal error queue subscriber setup publication is active; finish setup and retry"),
    ("Metal.Kernel.CallbackWouldBlock", "Metal error a completion callback cannot block on its own queue"),
    ("Metal.Kernel.LockError", "Metal error Could not lock resource: lock fixture"),
    ("Metal.Kernel.LoadLibraryError", "Metal error Error while loading library: library fixture"),
    ("Metal.Kernel.LoadFunctionError", "Metal error Error while loading function: function fixture"),
    ("Metal.Kernel.ArmRefused", "Metal error arm fixture"),
    ("Metal.Kernel.EmulatedRefused", "Metal error emulated fixture"),
    ("Metal.Kernel.FailedToCreateComputeFunction", "Metal error Failed to create compute function"),
    ("Metal.Kernel.InvalidInput", "Metal error Invalid input: input fixture"),
    ("Metal.Kernel.FailedToCreateResource", "Metal error Failed to create metal resource: resource fixture"),
    ("Metal.Kernel.SdpaHeadDTypeMismatch", "Metal error Sdpa full got dtype BF16"),
    ("Metal.Kernel.WithBacktrace", "Metal error a completion callback cannot block on its own queue\ndisabled backtrace"),
    ("Metal.LockError.Poisoned", "Metal error Poisoned(\"poison fixture\")"),
    ("Metal.LockError.WouldBlock", "Metal error WouldBlock"),
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
    let nested: std::collections::BTreeSet<_> =
        all.iter().filter_map(|(_, e)| nested_variant(e)).collect();
    let allocation = nested
        .iter()
        .filter(|n| n.starts_with("Allocation."))
        .count();
    assert_eq!(
        allocation, 18,
        "AllocationError variants with a fixture: {nested:?}"
    );
    #[cfg(feature = "metal")]
    {
        let kernel = nested
            .iter()
            .filter(|n| n.starts_with("Metal.Kernel."))
            .count();
        // Twenty-three variants: `Allocation` is counted above and
        // `CompletionRegistration` is pinned in `kernels::err`.
        assert_eq!(
            kernel, 21,
            "MetalKernelError variants with a fixture: {nested:?}"
        );
        let metal = nested
            .iter()
            .filter(|n| !n.starts_with("Allocation."))
            .count()
            - kernel;
        assert_eq!(
            metal, 3,
            "MetalError variants beside KernelError with a fixture: {nested:?}"
        );
    }
}
