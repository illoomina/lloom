//! Predicted managed-byte admission, not an OS cap over opaque driver overhead.
//! The native caller supplies a documented charge basis and observes actual
//! allocated bytes. Overprediction returns unused demand. Underprediction is a
//! typed failure AFTER destroying raw storage; it cannot undo a transient peak.

use crate::allocation_owner::AllocationOwner;
use crate::memory_admission::{AdmissionError, AllocationContext, AllocationTicket};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionBasis {
    /// Heap size/alignment queried for the same options, at least requested
    /// bytes, rounded with checked arithmetic. For standalone buffers this is
    /// an estimator, NOT Apple's guarantee for heap suballocations.
    HeapSizeAndAlignEstimate,
    /// Checked placement chunk request; actual heap allocation may differ.
    HeapDescriptorSizeEstimate,
}

#[derive(Debug)]
pub enum AllocationError<E> {
    Admission(AdmissionError),
    Geometry(&'static str),
    Retirement(String),
    PredictionExceeded {
        predicted: usize,
        actual: usize,
        basis: PredictionBasis,
    },
    Native(E),
    Unsupported(&'static str),
}
impl<E: std::fmt::Display> std::fmt::Display for AllocationError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => write!(f, "{e}"),
            Self::Geometry(e) => write!(f, "allocation geometry: {e}"),
            Self::Retirement(e) => write!(f, "allocation retirement registration: {e}"),
            Self::PredictionExceeded { predicted, actual, basis } => write!(f,
                "allocation prediction exceeded: {actual} actual bytes > {predicted} predicted ({basis:?}); raw allocation released; transient peak was not prevented"),
            Self::Native(e) => write!(f, "native allocation: {e}"),
            Self::Unsupported(e) => write!(f, "admitted allocation unsupported: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for AllocationError<E> {}
impl<E> AllocationError<E> {
    pub(crate) fn permits_fallback(&self, native_refusal: impl FnOnce(&E) -> bool) -> bool {
        match self {
            Self::Unsupported(_) => true,
            Self::Native(e) => native_refusal(e),
            _ => false,
        }
    }
}

pub(crate) struct Prediction {
    pub requested: usize,
    pub bytes: usize,
    pub basis: PredictionBasis,
}
impl Prediction {
    pub(crate) fn heap(bytes: usize) -> Self {
        Self {
            requested: bytes,
            bytes,
            basis: PredictionBasis::HeapDescriptorSizeEstimate,
        }
    }
    pub(crate) fn standalone<E>(
        requested: usize,
        heap_size: usize,
        alignment: usize,
    ) -> Result<Self, AllocationError<E>> {
        if alignment == 0 {
            return Err(AllocationError::Geometry("zero estimator alignment"));
        }
        let requested = requested.max(1);
        let size = requested.max(heap_size);
        let remainder = size % alignment;
        let bytes = if remainder == 0 {
            size
        } else {
            size.checked_add(alignment - remainder)
                .ok_or(AllocationError::Geometry(
                    "predicted byte rounding overflow",
                ))?
        };
        Ok(Self {
            requested,
            bytes,
            basis: PredictionBasis::HeapSizeAndAlignEstimate,
        })
    }
}

/// Uncommitted native allocation. Drop order is raw first, ticket second: an
/// unused speculative allocation returns the full entitlement to its parent.
pub(crate) struct StagedAllocation<R> {
    raw: R,
    ticket: AllocationTicket,
    actual: usize,
}
impl<R> StagedAllocation<R> {
    pub(crate) fn raw(&self) -> &R {
        &self.raw
    }
    pub(crate) fn commit<E>(self) -> Result<(R, AllocationOwner), AllocationError<E>> {
        commit_raw(self.raw, self.ticket, self.actual)
    }
}

pub(crate) fn stage<R, E>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Result<R, E>,
    actual_size: impl FnOnce(&R) -> usize,
) -> Result<StagedAllocation<R>, AllocationError<E>> {
    let ticket = context
        .take(prediction.bytes)
        .map_err(AllocationError::Admission)?;
    let raw = create(prediction.requested).map_err(AllocationError::Native)?;
    let actual = actual_size(&raw);
    if actual > prediction.bytes {
        drop(raw);
        drop(ticket);
        return Err(AllocationError::PredictionExceeded {
            predicted: prediction.bytes,
            actual,
            basis: prediction.basis,
        });
    }
    Ok(StagedAllocation {
        raw,
        ticket,
        actual,
    })
}

/// A single ticket is issued before native creation. Return raw before owner in
/// tuple drop order until the caller installs its persistent AllocationLease.
pub(crate) fn allocate<R, E>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Result<R, E>,
    actual_size: impl FnOnce(&R) -> usize,
) -> Result<(R, AllocationOwner), AllocationError<E>> {
    stage(context, prediction, create, actual_size)?.commit()
}

fn commit_raw<R, E>(
    raw: R,
    mut ticket: AllocationTicket,
    actual: usize,
) -> Result<(R, AllocationOwner), AllocationError<E>> {
    match ticket.try_commit(actual) {
        Ok(charge) => Ok((raw, AllocationOwner::new(charge))),
        Err(error) => {
            // ANY refused transfer keeps the ticket until physical raw release.
            drop(raw);
            drop(ticket);
            Err(AllocationError::Admission(error))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{process_account, AccountSnapshot, ModelDomain};
    use std::cell::Cell;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct RawOracle {
        domain: ModelDomain,
        expected: AccountSnapshot,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for RawOracle {
        fn drop(&mut self) {
            assert_eq!(
                self.domain.snapshot(),
                self.expected,
                "raw release precedes ticket restoration or final charge release"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn predicted(bytes: usize) -> Prediction {
        Prediction::standalone::<&'static str>(bytes, bytes, 1).unwrap()
    }
    #[test]
    fn numeric_prediction_checks_rounding_and_preserves_native_length() {
        let p = Prediction::standalone::<&str>(5, 9, 8).unwrap();
        assert_eq!((p.requested, p.bytes), (5, 16));
        assert_eq!(Prediction::standalone::<&str>(0, 0, 1).unwrap().bytes, 1);
        assert_eq!(Prediction::standalone::<&str>(17, 4, 8).unwrap().bytes, 24);
        assert!(matches!(
            Prediction::standalone::<&str>(usize::MAX, 0, 2),
            Err(AllocationError::Geometry(_))
        ));
        assert!(matches!(
            Prediction::standalone::<&str>(1, 1, 0),
            Err(AllocationError::Geometry(_))
        ));
    }
    #[test]
    fn numeric_padded_upload_demand_copies_only_native_length() {
        let input = [1u8, 2, 3, 4, 5];
        let domain = process_account().unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(16).unwrap();
        let prediction = Prediction::standalone::<&str>(input.len(), 9, 8).unwrap();
        let (copied, owner) = allocate(
            &pass.context(),
            prediction,
            |length| {
                assert_eq!(length, 5, "creator must never read padded 16-byte demand");
                Ok::<_, &str>(input[..length].to_vec())
            },
            |_| 8,
        )
        .unwrap();
        assert_eq!(copied, input);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 8,
                committed: 8
            }
        );
        drop(copied);
        drop(owner);
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn numeric_admission_and_cancel_refuse_before_native_call() {
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let context = pass.context();
        let called = Cell::new(false);
        let result = allocate(
            &context,
            predicted(16),
            |_| {
                called.set(true);
                Ok::<_, &str>(0)
            },
            |_| 0,
        );
        assert!(matches!(
            result,
            Err(AllocationError::Admission(
                AdmissionError::PromiseTooSmall { .. }
            ))
        ));
        assert!(!called.get());
        drop(pass);
        let result = allocate(
            &context,
            predicted(8),
            |_| {
                called.set(true);
                Ok::<_, &str>(0)
            },
            |_| 0,
        );
        assert!(matches!(
            result,
            Err(AllocationError::Admission(AdmissionError::Cancelled))
        ));
        assert!(!called.get());
    }
    #[test]
    fn numeric_native_refusal_restores_same_open_context() {
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let context = pass.context();
        let result = allocate(
            &context,
            predicted(8),
            |_| {
                assert_eq!(domain.snapshot().pending, 8);
                Err::<(), _>("native refused")
            },
            |_| 0,
        );
        assert!(matches!(
            result,
            Err(AllocationError::Native("native refused"))
        ));
        let returned = context.take(8).unwrap();
        drop(pass);
        assert_eq!(domain.snapshot().pending, 8);
        drop(returned);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn numeric_prediction_overrun_destroys_raw_before_returning_demand() {
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let result = allocate(
            &context,
            predicted(8),
            |_| {
                drop(pass); // Issued ticket survives cancellation during native creation.
                Ok::<_, &str>(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 8,
                        committed: 0,
                    },
                    drops: drops.clone(),
                })
            },
            |_| 12,
        );
        assert!(matches!(
            result,
            Err(AllocationError::PredictionExceeded {
                predicted: 8,
                actual: 12,
                ..
            })
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn numeric_commit_refusal_keeps_ticket_until_raw_is_destroyed() {
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let ticket = pass.context().take(8).unwrap();
        drop(pass);
        let drops = Arc::new(AtomicUsize::new(0));
        let raw = RawOracle {
            domain: domain.clone(),
            expected: AccountSnapshot {
                pending: 8,
                committed: 0,
            },
            drops: drops.clone(),
        };
        // Directly exercise the actual bridge's transfer-failure cleanup arm.
        let result = commit_raw::<_, &str>(raw, ticket, 12);
        assert!(matches!(
            result,
            Err(AllocationError::Admission(
                AdmissionError::PromiseTooSmall {
                    remaining: 8,
                    requested: 12
                }
            ))
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn numeric_success_reconciles_actual_and_survives_parent_cancellation() {
        let domain = process_account().unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(16).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let allocation = allocate(
            &context,
            predicted(16),
            |length| {
                assert_eq!(length, 16);
                assert_eq!(domain.snapshot().pending, 16);
                drop(pass);
                Ok::<_, &str>(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 0,
                        committed: 8,
                    },
                    drops: drops.clone(),
                })
            },
            |_| 8,
        )
        .unwrap();
        assert_eq!(allocation.1.bytes(), 8);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 8
            }
        );
        drop(allocation); // Tuple raw-first order, as used before Buffer wrapping.
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn numeric_unused_staged_native_returns_full_parent_entitlement() {
        let domain = process_account().unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(16).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let staged = stage(
            &context,
            predicted(16),
            |_| {
                Ok::<_, &str>(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 16,
                        committed: 0,
                    },
                    drops: drops.clone(),
                })
            },
            |_| 8,
        )
        .unwrap();
        assert_eq!(domain.snapshot().committed, 0);
        assert!(matches!(
            context.take(1),
            Err(AdmissionError::PromiseTooSmall { .. })
        ));
        drop(staged);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let fully_returned = context.take(16).unwrap();
        drop(pass);
        assert_eq!(domain.snapshot().pending, 16);
        drop(fully_returned);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn numeric_impossible_staged_capacity_returns_full_entitlement() {
        let domain = process_account().unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(16).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let staged = stage(
            &context,
            predicted(16),
            |_| {
                Ok::<_, &str>((
                    RawOracle {
                        domain: domain.clone(),
                        expected: AccountSnapshot {
                            pending: 16,
                            committed: 0,
                        },
                        drops: drops.clone(),
                    },
                    4usize,
                ))
            },
            |raw| raw.1,
        )
        .unwrap();
        let mut ranges = crate::heap_ranges::Ranges::new(staged.raw().1);
        assert_eq!(ranges.take(8, 4), None);
        drop(ranges);
        drop(staged);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let returned = context.take(16).unwrap();
        drop(pass);
        assert_eq!(domain.snapshot().pending, 16);
        drop(returned);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn numeric_fallback_is_explicit_and_never_retries_admission_errors() {
        assert!(AllocationError::Native(true).permits_fallback(|refused| *refused));
        assert!(!AllocationError::Native(false).permits_fallback(|refused| *refused));
        assert!(AllocationError::<bool>::Unsupported("heap pending").permits_fallback(|_| false));
        for error in [
            AllocationError::Admission(AdmissionError::Cancelled),
            AllocationError::Admission(AdmissionError::PromiseTooSmall {
                remaining: 4,
                requested: 8,
            }),
            AllocationError::Geometry("overflow"),
            AllocationError::PredictionExceeded {
                predicted: 4,
                actual: 8,
                basis: PredictionBasis::HeapSizeAndAlignEstimate,
            },
        ] {
            assert!(!error.permits_fallback(|_: &bool| true));
        }
    }
}
