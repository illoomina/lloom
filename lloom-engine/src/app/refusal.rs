//! The HTTP status of a request that failed before its reply's first
//! fragment, when the response can still carry one: a memory refusal is
//! 503 with `Retry-After`, the client's cue to send it again (decision 83,
//! #1145); a request the server cannot serve as written is 400; anything
//! else is 500. Each is found through the error's context chain by its type
//! (`AllocationError` as it travels, `memory.md` §9.10a; [`InputRefused`]),
//! never by matching its text.

use lloom_tensor::memory_admission::AllocationError;

/// Seconds a 503 asks the client to wait before retrying: a hint, not a
/// promise that the memory will be there. The memory admission that refused
/// the request has no estimate of when it will be (`memory.md` §9.10).
pub const RETRY_AFTER_SECONDS: u32 = 1;

/// The request asks for what this server cannot do with its input: an image
/// to a model without vision. The client's error, so a 400.
#[derive(Debug)]
pub struct InputRefused(pub String);

impl std::fmt::Display for InputRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InputRefused {}

/// The status a pre-first-fragment failure answers with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailureStatus {
    pub code: u16,
    /// `Retry-After` in seconds, sent with a 503.
    pub retry_after: Option<u32>,
}

impl FailureStatus {
    /// The status for `error`: 400 when the request's input was refused;
    /// 503 with `Retry-After` when memory refused the request — a quota, the
    /// machine, the device, or the admission's own host reading — else 500.
    pub fn of(error: &anyhow::Error) -> Self {
        if error.chain().any(|cause| cause.is::<InputRefused>()) {
            return Self {
                code: 400,
                retry_after: None,
            };
        }
        match allocation_refusal(error) {
            Some(refusal) if is_memory_refusal(&refusal) => Self {
                code: 503,
                retry_after: Some(RETRY_AFTER_SECONDS),
            },
            _ => Self {
                code: 500,
                retry_after: None,
            },
        }
    }
}

/// The typed allocation refusal anywhere in `error`'s chain.
pub fn allocation_refusal(error: &anyhow::Error) -> Option<AllocationError> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<lloom_tensor::Error>()
            .and_then(lloom_tensor::Error::allocation)
            .or_else(|| cause.downcast_ref::<AllocationError>().copied())
    })
}

/// Memory that is not there now and may be later. The other refusals are
/// arithmetic, geometry or custody defects a retry cannot change.
fn is_memory_refusal(refusal: &AllocationError) -> bool {
    matches!(
        refusal,
        AllocationError::Refused { .. }
            | AllocationError::DeviceOom { .. }
            | AllocationError::CountExhausted
            | AllocationError::Machine(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use lloom_tensor::memory_admission::RefusalDomain;

    fn refused(domain: RefusalDomain) -> AllocationError {
        AllocationError::Refused {
            requested: 7_595_648,
            available: 0,
            domain,
        }
    }

    /// The refusal a pass meets on the machine (#1128's 35B: `Machine
    /// memory refused 7595648 bytes`), as the engine returns it: a tensor
    /// error under the request's own contexts.
    #[test]
    fn a_machine_refusal_under_contexts_is_503_with_retry_after() {
        let error: anyhow::Error =
            Err::<(), _>(lloom_tensor::Error::from(refused(RefusalDomain::Machine)))
                .context("decoding the turn")
                .context("/v1/completions")
                .unwrap_err();
        assert_eq!(
            allocation_refusal(&error),
            Some(refused(RefusalDomain::Machine))
        );
        assert_eq!(
            FailureStatus::of(&error),
            FailureStatus {
                code: 503,
                retry_after: Some(RETRY_AFTER_SECONDS)
            }
        );
    }

    #[test]
    fn every_memory_refusal_is_503_and_every_defect_500() {
        let status = |e: AllocationError| {
            FailureStatus::of(&anyhow::Error::from(lloom_tensor::Error::from(e))).code
        };
        for domain in [
            RefusalDomain::Process,
            RefusalDomain::Model,
            RefusalDomain::Promise,
            RefusalDomain::Machine,
            RefusalDomain::Budget,
        ] {
            assert_eq!(status(refused(domain)), 503, "{domain:?}");
        }
        assert_eq!(status(AllocationError::DeviceOom { requested: 1 }), 503);
        assert_eq!(status(AllocationError::CountExhausted), 503);
        assert_eq!(status(AllocationError::Overflow), 500);
        assert_eq!(status(AllocationError::Geometry("shape")), 500);
        assert_eq!(status(AllocationError::InvalidCustody), 500);
        // A bare `AllocationError` in the chain is found too.
        assert_eq!(
            FailureStatus::of(&anyhow::Error::from(refused(RefusalDomain::Budget))).code,
            503
        );
    }

    /// A refused input under the request's contexts is the client's: 400,
    /// nothing to retry.
    #[test]
    fn a_refused_input_under_contexts_is_400() {
        let error = anyhow::Error::new(InputRefused(
            "qwen4exp does not take image input: no vision tower".into(),
        ))
        .context("decoding the turn");
        assert_eq!(
            FailureStatus::of(&error),
            FailureStatus {
                code: 400,
                retry_after: None
            }
        );
        assert_eq!(
            format!("{error:#}"),
            "decoding the turn: qwen4exp does not take image input: no vision tower"
        );
    }

    /// A failure that is not an allocation's — the same words in a message
    /// included — is 500: the status follows the type, not the text.
    #[test]
    fn an_untyped_failure_is_500_whatever_it_says() {
        let error = anyhow::anyhow!("Machine memory refused 7595648 bytes: 0 bytes available");
        assert_eq!(allocation_refusal(&error), None);
        assert_eq!(
            FailureStatus::of(&error),
            FailureStatus {
                code: 500,
                retry_after: None
            }
        );
    }
}
