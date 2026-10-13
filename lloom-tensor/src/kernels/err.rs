use crate::kernels::kernels::sdpa::SdpaDType;

pub use crate::command_lifecycle::LifecycleError as CommandLifecycleError;

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    /// An allocation refused inside a kernel-layer path; widened unchanged.
    #[error(transparent)]
    Allocation(#[from] crate::memory_admission::AllocationError),
    #[error("command buffer recording admission refused: {0}")]
    RecordingAdmission(#[from] CommandLifecycleError),
    #[error("completion registration refused: {0}")]
    CompletionRegistration(#[from] CompletionRegistrationError),
    #[error("a completion callback cannot block on its own queue")]
    CallbackWouldBlock,
    #[error("GPU queue unavailable: {0}")]
    QueueUnavailable(&'static str),
    #[error("Command buffer had following error: {0}")]
    CommandBufferError(String),
    #[error("Could not lock resource: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0}")]
    LoadFunctionError(String),
    /// An arm this device does not execute, refused by name (§15.2a).
    #[error("{0}")]
    ArmRefused(String),
    /// A type this device runs only through software emulation, refused
    /// until the operator opts in to it (§15.2a).
    #[error("{0}")]
    EmulatedRefused(String),
    #[error("Unsupported dtype {0} for operation {1}")]
    UnsupportedDTypeForOp(&'static str, &'static str),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Failed to create metal resource: {0}")]
    FailedToCreateResource(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    #[error("Sdpa {variation} head size was {got}, expectd {expected:?}")]
    SdpaHeadSizeMismatch {
        variation: &'static str,
        got: usize,
        expected: Vec<usize>,
    },
    #[error("Sdpa {variation} got dtype {got:?}")]
    SdpaHeadDTypeMismatch {
        variation: &'static str,
        got: SdpaDType,
    },
    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },
}

impl MetalKernelError {
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
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

/// Registration failed without invoking or discarding the submitted callback.
///
/// The rejected owner is deliberately opaque: error formatting never runs user
/// code. Propagate this error out of recording locks before dropping it. The
/// payload may capture the final Device, custom Executor, or arbitrary user Drop.
pub struct CompletionRegistrationError {
    cause: Box<MetalKernelError>,
    rejected: Option<Box<dyn Send + Sync>>,
}
impl CompletionRegistrationError {
    pub(crate) fn new<T: Send + Sync + 'static>(
        cause: MetalKernelError,
        rejected: Option<T>,
    ) -> Self {
        Self {
            cause: Box::new(cause),
            rejected: rejected.map(|value| Box::new(value) as Box<dyn Send + Sync>),
        }
    }
    pub fn cause(&self) -> &MetalKernelError {
        &self.cause
    }
    pub fn into_parts(self) -> (MetalKernelError, Option<Box<dyn Send + Sync>>) {
        (*self.cause, self.rejected)
    }
}
impl std::fmt::Debug for CompletionRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionRegistrationError")
            .field("cause", &self.cause)
            .field("has_rejected_callback", &self.rejected.is_some())
            .finish()
    }
}
impl std::fmt::Display for CompletionRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for CompletionRegistrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.cause)
    }
}

#[cfg(test)]
mod registration_error_tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct RejectedOwner {
        recording: Arc<parking_lot::Mutex<()>>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for RejectedOwner {
        fn drop(&mut self) {
            let _guard = self
                .recording
                .try_lock()
                .expect("rejected owner dropped under recording lock");
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn refused(
        recording: &parking_lot::Mutex<()>,
        owner: RejectedOwner,
    ) -> Result<(), MetalKernelError> {
        let _recording = recording.lock();
        Err(CompletionRegistrationError::new(
            MetalKernelError::QueueUnavailable("fixture refusal"),
            Some(owner),
        ))?;
        Ok(())
    }
    #[test]
    fn rejected_capture_survives_error_conversions_until_caller_drop() {
        let recording = Arc::new(parking_lot::Mutex::new(()));
        let drops = Arc::new(AtomicUsize::new(0));
        let error = refused(
            &recording,
            RejectedOwner {
                recording: recording.clone(),
                drops: drops.clone(),
            },
        )
        .unwrap_err();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(format!("{error:?}").contains("has_rejected_callback"));
        let error: crate::metal_backend::MetalError = error.into();
        let error: crate::Error = error.into();
        assert!(format!("{error}").contains("fixture refusal"));
        // The outer tensor Error intentionally delegates Debug to Display.
        assert_eq!(format!("{error:?}"), format!("{error}"));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "formatting or conversion destroyed rejected capture"
        );
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn rejected_capture_can_be_separated_from_printable_cause() {
        let recording = Arc::new(parking_lot::Mutex::new(()));
        let drops = Arc::new(AtomicUsize::new(0));
        let error = CompletionRegistrationError::new(
            MetalKernelError::CallbackWouldBlock,
            Some(RejectedOwner {
                recording,
                drops: drops.clone(),
            }),
        );
        let (cause, payload) = error.into_parts();
        assert!(matches!(cause, MetalKernelError::CallbackWouldBlock));
        drop(cause);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(payload);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
