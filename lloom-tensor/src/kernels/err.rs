use crate::kernels::kernels::sdpa::SdpaDType;

pub use crate::command_lifecycle::LifecycleError as CommandLifecycleError;

#[derive(Debug, thiserror::Error)]
#[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
pub struct MatMulNonContiguous {
    pub lhs_stride: Vec<usize>,
    pub rhs_stride: Vec<usize>,
    pub mnk: (usize, usize, usize),
}

#[derive(Debug, thiserror::Error)]
#[error("Unsupported dtype {dtype} for operation {op}")]
pub struct UnsupportedDTypeForOp {
    pub dtype: &'static str,
    pub op: &'static str,
}

#[derive(Debug, thiserror::Error)]
#[error("Sdpa {variation} head size was {got}, expectd {expected:?}")]
pub struct SdpaHeadSizeMismatch {
    pub variation: &'static str,
    pub got: usize,
    pub expected: Vec<usize>,
}

/// Kernel-layer failures. Carried inline in [`crate::Error::Metal`], so the
/// refusals a dispatch can meet (`Allocation`, `RecordingAdmission`) widen
/// without allocating; payloads larger than [`AllocationError`] are boxed so
/// they do not set this enum's size (`docs/invariants.md` §15.2 #15a).
///
/// [`AllocationError`]: crate::memory_admission::AllocationError
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
    #[error("bound command buffer requires its owning Commands submission transaction")]
    OwnerSubmissionRequired,
    #[error("queue completion subscribers are sealed; register during device setup")]
    SubscriberConfigurationSealed,
    #[error("queue subscriber setup publication is active; finish setup and retry")]
    SubscriberSetupBusy,
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
    #[error(transparent)]
    UnsupportedDTypeForOp(Box<UnsupportedDTypeForOp>),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Failed to create metal resource: {0}")]
    FailedToCreateResource(String),
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error(transparent)]
    MatMulNonContiguous(Box<MatMulNonContiguous>),
    #[error(transparent)]
    SdpaHeadSizeMismatch(Box<SdpaHeadSizeMismatch>),
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
        // The text the enums printed before their payloads were boxed, as
        // `tests/error_display.rs` pins every other variant's.
        assert_eq!(
            format!("{error}"),
            "Metal error completion registration refused: GPU queue unavailable: fixture refusal"
        );
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
