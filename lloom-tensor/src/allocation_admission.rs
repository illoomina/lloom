//! Predicted managed-byte admission, not an OS cap over opaque driver overhead.
//! The native caller supplies a documented charge basis and observes actual
//! allocated bytes. Overprediction returns unused demand. Underprediction is a
//! typed failure AFTER destroying raw storage; it cannot undo a transient peak.

use crate::allocation_owner::AllocationOwner;
use crate::memory_admission::{AllocationContext, AllocationTicket};
pub use crate::memory_admission::{AllocationError, PredictionBasis};

impl AllocationError {
    /// Whether a refusal lets a fallback backing try: a native creation the
    /// device refused, or a route this backing does not serve. An admission
    /// refusal is the caller's, and no other backing may override it.
    pub(crate) fn permits_fallback(&self) -> bool {
        matches!(self, Self::Unsupported(_) | Self::DeviceOom { .. })
    }
}

/// The VM page size, read once: the granule of a standalone native buffer's
/// `allocatedSize`.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn native_page_bytes() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        #[cfg(unix)]
        {
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page > 0 {
                return page as usize;
            }
        }
        4096
    })
}

pub(crate) struct Prediction {
    pub requested: usize,
    pub bytes: usize,
    pub basis: PredictionBasis,
}
impl Prediction {
    /// Native estimator term plus exactly ONE physical charge-control Layout.
    /// This leaves requested native length and actual native bound unchanged.
    pub(crate) fn requested_owned_bytes(&self) -> Result<usize, AllocationError> {
        let control = AllocationTicket::requested_charge_control_bytes()?;
        self.bytes
            .checked_add(control)
            .ok_or(AllocationError::Overflow)
    }
    pub(crate) fn heap(bytes: usize) -> Self {
        Self {
            requested: bytes,
            bytes,
            basis: PredictionBasis::HeapDescriptorSizeEstimate,
        }
    }
    pub(crate) fn standalone(
        requested: usize,
        heap_size: usize,
        alignment: usize,
    ) -> Result<Self, AllocationError> {
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
    owned_control: bool,
}
impl<R> StagedAllocation<R> {
    pub(crate) fn raw(&self) -> &R {
        &self.raw
    }
    pub(crate) fn commit(self) -> Result<(R, AllocationOwner), AllocationError> {
        commit_raw_with_control(self.raw, self.ticket, self.actual, self.owned_control)
    }
}

#[cfg(test)]
pub(crate) fn stage<R>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Option<R>,
    actual_size: impl FnOnce(&R) -> usize,
) -> Result<StagedAllocation<R>, AllocationError> {
    stage_with_control(context, prediction, create, actual_size, false)
}

/// Actual native factories account for the charge-control requested Layout
/// before native entry. Supplied paths split the SAME already-issued promise;
/// default factories separately reserve the SAME summed quote exactly once.
/// Arc allocation retains std global OOM semantics; this is not an RSS bound.
pub(crate) fn stage_owned<R>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Option<R>,
    actual_size: impl FnOnce(&R) -> usize,
) -> Result<StagedAllocation<R>, AllocationError> {
    stage_with_control(context, prediction, create, actual_size, true)
}
fn stage_with_control<R>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Option<R>,
    actual_size: impl FnOnce(&R) -> usize,
    owned_control: bool,
) -> Result<StagedAllocation<R>, AllocationError> {
    let quote = if owned_control {
        prediction.requested_owned_bytes()?
    } else {
        prediction.bytes
    };
    let ticket = context.take(quote)?;
    let raw = create(prediction.requested).ok_or(AllocationError::DeviceOom {
        requested: prediction.requested,
    })?;
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
        owned_control,
    })
}

/// A single ticket is issued before native creation. Return raw before owner in
/// tuple drop order until the caller installs its persistent AllocationLease.
#[cfg(test)]
pub(crate) fn allocate<R>(
    context: &AllocationContext,
    prediction: Prediction,
    create: impl FnOnce(usize) -> Option<R>,
    actual_size: impl FnOnce(&R) -> usize,
) -> Result<(R, AllocationOwner), AllocationError> {
    stage(context, prediction, create, actual_size)?.commit()
}

#[cfg(test)]
fn commit_raw<R>(
    raw: R,
    ticket: AllocationTicket,
    actual: usize,
) -> Result<(R, AllocationOwner), AllocationError> {
    commit_raw_with_control(raw, ticket, actual, false)
}
fn commit_raw_with_control<R>(
    raw: R,
    mut ticket: AllocationTicket,
    actual: usize,
    owned_control: bool,
) -> Result<(R, AllocationOwner), AllocationError> {
    let committed = if owned_control {
        ticket.try_commit_with_control(actual)
    } else {
        ticket.try_commit(actual)
    };
    match committed {
        Ok(charge) => Ok((raw, AllocationOwner::new(charge))),
        Err(error) => {
            // ANY refused transfer keeps the ticket until physical raw release.
            drop(raw);
            drop(ticket);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{process_account, AccountSnapshot, ModelDomain, RefusalDomain};
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
        Prediction::standalone(bytes, bytes, 1).unwrap()
    }
    /// Where std's Arc layout is not audited (`build.rs`), the owned quote
    /// refuses by policy: `stage_owned` returns `UnsupportedLayout` before it
    /// draws on the promise and before any native create.
    #[cfg(not(lloom_audited_arc_layout))]
    fn unaudited_stage_refuses_before_create(bytes: usize) {
        assert_eq!(
            AllocationTicket::requested_charge_control_bytes(),
            Err(AllocationError::UnsupportedLayout)
        );
        let domain = ModelDomain::isolated_numeric_fixture(bytes);
        let pass = domain.reserve(bytes).unwrap();
        let context = pass.context();
        let called = Cell::new(false);
        let result = stage_owned(
            &context,
            predicted(bytes),
            |_| {
                called.set(true);
                Some(0usize)
            },
            |_| bytes,
        );
        assert!(
            matches!(result, Err(AllocationError::UnsupportedLayout)),
            "{:?}",
            result.err()
        );
        assert!(!called.get(), "the refusal precedes any native create");
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: bytes,
                committed: 0
            },
            "the refusal draws nothing on the promise"
        );
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn owned_native_control_success_reconciles_same_promise() {
        unaudited_stage_refuses_before_create(16);
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn owned_native_control_success_reconciles_same_promise() {
        let control = AllocationTicket::requested_charge_control_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(16 + control);
        let pass = domain.reserve(16 + control).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let staged = stage_owned(
            &context,
            predicted(16),
            |requested| {
                assert_eq!(requested, 16, "control quote must not change native length");
                drop(pass); // issued full ticket survives genuine cancellation
                Some(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 0,
                        committed: 8 + control,
                    },
                    drops: drops.clone(),
                })
            },
            |_| 8,
        )
        .unwrap();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 16 + control,
                committed: 0
            }
        );
        let (raw, owner) = staged.commit().unwrap();
        assert_eq!(
            owner.bytes(),
            8,
            "logical native bytes exclude control Layout"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 8 + control
            }
        );
        drop(raw);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn owned_native_control_missing_promise_refuses_before_creator() {
        unaudited_stage_refuses_before_create(8);
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn owned_native_control_missing_promise_refuses_before_creator() {
        let domain = ModelDomain::isolated_numeric_fixture(8);
        let pass = domain.reserve(8).unwrap();
        let context = pass.context();
        let called = Cell::new(false);
        let result = stage_owned(
            &context,
            predicted(8),
            |_| {
                called.set(true);
                Some(0usize)
            },
            |_| 8,
        );
        assert!(matches!(
            result,
            Err(AllocationError::Refused {
                domain: RefusalDomain::Promise,
                ..
            })
        ));
        assert!(
            !called.get(),
            "control entitlement precedes real native create"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 8,
                committed: 0
            }
        );
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn owned_native_control_overrun_keeps_full_ticket_through_raw_drop() {
        unaudited_stage_refuses_before_create(8);
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn owned_native_control_overrun_keeps_full_ticket_through_raw_drop() {
        let control = AllocationTicket::requested_charge_control_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(8 + control);
        let pass = domain.reserve(8 + control).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let result = stage_owned(
            &context,
            predicted(8),
            |_| {
                drop(pass);
                Some(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 8 + control,
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
    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn owned_native_control_unused_stage_returns_original_total() {
        unaudited_stage_refuses_before_create(16);
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn owned_native_control_unused_stage_returns_original_total() {
        let control = AllocationTicket::requested_charge_control_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(16 + control);
        let pass = domain.reserve(16 + control).unwrap();
        let context = pass.context();
        let drops = Arc::new(AtomicUsize::new(0));
        let staged = stage_owned(
            &context,
            predicted(16),
            |_| {
                Some(RawOracle {
                    domain: domain.clone(),
                    expected: AccountSnapshot {
                        pending: 16 + control,
                        committed: 0,
                    },
                    drops: drops.clone(),
                })
            },
            |_| 8,
        )
        .unwrap();
        drop(staged);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let returned = context
            .take(16 + control)
            .expect("same parent receives native+control unused demand");
        drop(pass);
        drop(returned);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn numeric_prediction_checks_rounding_and_preserves_native_length() {
        let p = Prediction::standalone(5, 9, 8).unwrap();
        assert_eq!((p.requested, p.bytes), (5, 16));
        assert_eq!(Prediction::standalone(0, 0, 1).unwrap().bytes, 1);
        assert_eq!(Prediction::standalone(17, 4, 8).unwrap().bytes, 24);
        assert!(matches!(
            Prediction::standalone(usize::MAX, 0, 2),
            Err(AllocationError::Geometry(_))
        ));
        assert!(matches!(
            Prediction::standalone(1, 1, 0),
            Err(AllocationError::Geometry(_))
        ));
    }
    #[test]
    fn numeric_padded_upload_demand_copies_only_native_length() {
        let input = [1u8, 2, 3, 4, 5];
        let domain = process_account().unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(16).unwrap();
        let prediction = Prediction::standalone(input.len(), 9, 8).unwrap();
        let (copied, owner) = allocate(
            &pass.context(),
            prediction,
            |length| {
                assert_eq!(length, 5, "creator must never read padded 16-byte demand");
                Some(input[..length].to_vec())
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
                Some(0)
            },
            |_| 0,
        );
        assert!(matches!(
            result,
            Err(AllocationError::Refused {
                domain: RefusalDomain::Promise,
                ..
            })
        ));
        assert!(!called.get());
        drop(pass);
        let result = allocate(
            &context,
            predicted(8),
            |_| {
                called.set(true);
                Some(0)
            },
            |_| 0,
        );
        assert!(matches!(result, Err(AllocationError::Cancelled)));
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
                None::<()>
            },
            |_| 0,
        );
        assert!(matches!(
            result,
            Err(AllocationError::DeviceOom { requested: 8 })
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
                Some(RawOracle {
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
        let result = commit_raw(raw, ticket, 12);
        assert!(matches!(
            result,
            Err(AllocationError::Refused {
                requested: 12,
                available: 8,
                domain: RefusalDomain::Promise,
            })
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
                Some(RawOracle {
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
                Some(RawOracle {
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
            Err(AllocationError::Refused {
                domain: RefusalDomain::Promise,
                ..
            })
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
                Some((
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
        assert!(AllocationError::DeviceOom { requested: 8 }.permits_fallback());
        assert!(AllocationError::Unsupported("heap pending").permits_fallback());
        for error in [
            AllocationError::Cancelled,
            AllocationError::Refused {
                requested: 8,
                available: 4,
                domain: RefusalDomain::Promise,
            },
            AllocationError::Refused {
                requested: 8,
                available: 0,
                domain: RefusalDomain::Machine,
            },
            AllocationError::Geometry("overflow"),
            AllocationError::PredictionExceeded {
                predicted: 4,
                actual: 8,
                basis: PredictionBasis::HeapSizeAndAlignEstimate,
            },
            AllocationError::CountExhausted,
        ] {
            assert!(!error.permits_fallback(), "{error:?}");
        }
    }
}
