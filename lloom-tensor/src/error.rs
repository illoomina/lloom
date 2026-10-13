//! Candle-specific Error and Result
use std::{convert::Infallible, fmt::Display};

use crate::{DType, DeviceLocation, Layout, MetalError, Shape};

#[derive(Debug, Clone)]
pub struct MatMulUnexpectedStriding {
    pub lhs_l: Layout,
    pub rhs_l: Layout,
    pub bmnk: (usize, usize, usize, usize),
    pub msg: &'static str,
}

// Payloads of the cold variants whose inline size would exceed the hot
// refusals', boxed so they never set `size_of::<Error>()`. Each keeps its
// variant's name and `Display` text (`docs/invariants.md` §15.2 #15a).

#[derive(Debug, thiserror::Error)]
#[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
pub struct DimOutOfRange {
    pub shape: Shape,
    pub dim: i32,
    pub op: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
pub struct DuplicateDimIndex {
    pub shape: Shape,
    pub dims: Vec<usize>,
    pub op: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("{msg}, expected: {expected:?}, got: {got:?}")]
pub struct UnexpectedShape {
    pub msg: String,
    pub expected: Shape,
    pub got: Shape,
}

#[derive(Debug, thiserror::Error)]
#[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
pub struct ShapeMismatchBinaryOp {
    pub lhs: Shape,
    pub rhs: Shape,
    pub op: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
pub struct ShapeMismatchCat {
    pub dim: usize,
    pub first_shape: Shape,
    pub n: usize,
    pub nth_shape: Shape,
}

#[derive(Debug, thiserror::Error)]
#[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
pub struct DeviceMismatchBinaryOp {
    pub lhs: DeviceLocation,
    pub rhs: DeviceLocation,
    pub op: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
pub struct NarrowInvalidArgs {
    pub shape: Shape,
    pub dim: usize,
    pub start: usize,
    pub len: usize,
    pub msg: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("conv1d invalid args {msg}: inp: {inp_shape:?}, k: {k_shape:?}, pad: {padding}, stride: {stride}")]
pub struct Conv1dInvalidArgs {
    pub inp_shape: Shape,
    pub k_shape: Shape,
    pub padding: usize,
    pub stride: usize,
    pub msg: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
pub struct BroadcastIncompatibleShapes {
    pub src_shape: Shape,
    pub dst_shape: Shape,
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Main library error type.
///
/// Every `Result<T>` carries this enum by value, so its size is a per-return
/// cost. The refusals a warm path can meet stay inline and allocation-free:
/// `Allocation` (`Copy`), and `Metal`, which carries the kernel layer's
/// allocation and command-buffer recording refusals. A cold variant whose
/// payload is larger than those holds it behind one `Box`.
#[derive(thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error(transparent)]
    DimOutOfRange(Box<DimOutOfRange>),

    #[error(transparent)]
    DuplicateDimIndex(Box<DuplicateDimIndex>),

    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error(transparent)]
    UnexpectedShape(Box<UnexpectedShape>),

    #[error(
        "Shape mismatch, got buffer of size {buffer_size} which is compatible with shape {shape:?}"
    )]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error(transparent)]
    ShapeMismatchBinaryOp(Box<ShapeMismatchBinaryOp>),

    #[error(transparent)]
    ShapeMismatchCat(Box<ShapeMismatchCat>),

    #[error("Cannot divide tensor of shape {shape:?} equally along dim {dim} into {n_parts}")]
    ShapeMismatchSplit {
        shape: Shape,
        dim: usize,
        n_parts: usize,
    },

    #[error("{op} can only be performed on a single dimension")]
    OnlySingleDimension { op: &'static str, dims: Vec<usize> },

    #[error("empty tensor for {op}")]
    EmptyTensor { op: &'static str },

    // === Device Errors ===
    #[error(transparent)]
    DeviceMismatchBinaryOp(Box<DeviceMismatchBinaryOp>),

    // === Op Specific Errors ===
    #[error(transparent)]
    NarrowInvalidArgs(Box<NarrowInvalidArgs>),

    #[error(transparent)]
    Conv1dInvalidArgs(Box<Conv1dInvalidArgs>),

    #[error("{op} invalid index {index} with dim size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error(transparent)]
    BroadcastIncompatibleShapes(Box<BroadcastIncompatibleShapes>),

    #[error("cannot set variable {msg}")]
    CannotSetVar { msg: &'static str },

    // Box indirection to avoid large variant.
    #[error("{0:?}")]
    MatMulUnexpectedStriding(Box<MatMulUnexpectedStriding>),

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("{op} expects at least two tensors")]
    OpRequiresAtLeastTwoTensors { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    // === Other Errors ===
    #[error("the candle crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error("the candle crate has not been built with metal support")]
    NotCompiledWithMetalSupport,

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    // === Wrapped Errors ===
    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error("Metal error {0}")]
    Metal(#[from] MetalError),

    /// An allocation entry point refused, on any backend. `Copy` and built
    /// without allocating, so a warm refusal stays allocation-free.
    #[error(transparent)]
    Allocation(#[from] crate::memory_admission::AllocationError),

    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios"), feature = "ug"))]
    #[error(transparent)]
    Ug(#[from] candle_ug::Error),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),

    #[error("npy/npz error {0}")]
    Npy(String),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// SafeTensor error.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("unsupported safetensor dtype {0:?}")]
    UnsupportedSafeTensorDtype(safetensors::Dtype),

    /// Arbitrary errors wrapping.
    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    /// Arbitrary errors wrapping with context.
    #[error("{wrapped:?}\n{context:?}")]
    WrappedContext {
        wrapped: Box<dyn std::error::Error + Send + Sync>,
        context: String,
    },

    #[error("{context}\n{inner}")]
    Context {
        inner: Box<Self>,
        context: Box<dyn std::fmt::Display + Send + Sync>,
    },

    /// Adding path information to an error.
    #[error("path: {path:?} {inner}")]
    WithPath {
        inner: Box<Self>,
        path: std::path::PathBuf,
    },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::fmt::Display) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    pub fn debug(err: impl std::fmt::Debug) -> Self {
        Self::Msg(format!("{err:?}")).bt()
    }

    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }

    pub fn with_path<P: AsRef<std::path::Path>>(self, p: P) -> Self {
        Self::WithPath {
            inner: Box::new(self),
            path: p.as_ref().to_path_buf(),
        }
    }

    pub fn context(self, c: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Context {
            inner: Box::new(self),
            context: Box::new(c),
        }
    }

    /// The allocation refusal this error carries, under any context, path or
    /// backtrace it gained on the way up.
    pub fn allocation(&self) -> Option<crate::memory_admission::AllocationError> {
        match self {
            Self::Allocation(refusal) => Some(*refusal),
            Self::Context { inner, .. }
            | Self::WithPath { inner, .. }
            | Self::WithBacktrace { inner, .. } => inner.allocation(),
            _ => None,
        }
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub fn zip<T, U>(r1: Result<T>, r2: Result<U>) -> Result<(T, U)> {
    match (r1, r2) {
        (Ok(r1), Ok(r2)) => Ok((r1, r2)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

pub(crate) mod private {
    pub trait Sealed {}

    impl<T, E> Sealed for std::result::Result<T, E> where E: std::error::Error {}
    impl<T> Sealed for Option<T> {}
}

/// Attach more context to an error.
///
/// Inspired by [`anyhow::Context`].
pub trait Context<T, E>: private::Sealed {
    /// Wrap the error value with additional context.
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static;

    /// Wrap the error value with additional context that is evaluated lazily
    /// only once an error does occur.
    fn with_context<C, F>(self, f: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> Context<T, E> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using map_err to save 2 useless frames off the captured backtrace
        // in ext_context.
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::WrappedContext {
                wrapped: Box::new(error),
                context: context.to_string(),
            }
            .bt()),
        }
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::WrappedContext {
                wrapped: Box::new(error),
                context: context().to_string(),
            }
            .bt()),
        }
    }
}

impl<T> Context<T, Infallible> for Option<T> {
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using ok_or_else to save 2 useless frames off the captured
        // backtrace.
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::msg(context).bt()),
        }
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(context()).bt()),
        }
    }
}
