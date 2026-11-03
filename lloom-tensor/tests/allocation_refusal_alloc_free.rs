//! A refused allocation allocates nothing on the host, and widens into
//! `lloom_tensor::Error::Allocation` unchanged (`memory_admission::AllocationError`).
//!
//! Its own binary because it installs a counting global allocator. The count
//! is per thread: only allocations made on the thread running the refusal are
//! counted, so the test harness's own threads cannot move it.

use lloom_tensor::memory_admission::{process_account, AllocationError, RefusalDomain};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

struct Counting;

static COUNTED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    if COUNTING.with(|c| c.get()) {
        COUNTED.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` unchanged; the count is a side
// effect on an atomic and a const-initialised thread-local, neither of which
// allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `f`'s result and the host allocations made on this thread while it ran.
fn counted<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = COUNTED.load(Ordering::Relaxed);
    COUNTING.with(|c| c.set(true));
    let out = f();
    COUNTING.with(|c| c.set(false));
    (out, COUNTED.load(Ordering::Relaxed) - before)
}

/// Called outside counted windows; compare the existing typed diagnostic chain.
fn diagnostic_sources(error: &dyn std::error::Error) -> Vec<String> {
    let mut sources = Vec::new();
    let mut next = error.source();
    while let Some(source) = next {
        sources.push(source.to_string());
        next = source.source();
    }
    sources
}

/// The storage boundary's widening, as a caller writes it.
fn widen<T>(r: Result<T, AllocationError>) -> lloom_tensor::Result<T> {
    Ok(r?)
}

#[test]
fn a_refused_reserve_take_and_commit_allocate_nothing_and_widen_typed() {
    let domain = process_account().unadmitted_domain(Some(16)).unwrap();
    let pass = domain.reserve(16).unwrap();
    let context = pass.context();

    let (refused, n) = counted(|| widen(domain.reserve(1).map(drop)));
    assert_eq!(n, 0, "a refused reserve allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 1,
                available: 0,
                domain: RefusalDomain::Model,
            }))
        ),
        "{refused:?}"
    );

    let (refused, n) = counted(|| widen(context.take(17).map(drop)));
    assert_eq!(n, 0, "a refused take allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 17,
                available: 16,
                domain: RefusalDomain::Promise,
            }))
        ),
        "{refused:?}"
    );

    let mut ticket = context.take(8).unwrap();
    let (refused, n) = counted(|| widen(ticket.try_commit(9).map(drop)));
    assert_eq!(n, 0, "a refused commit allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Refused {
                requested: 9,
                available: 8,
                domain: RefusalDomain::Promise,
            }))
        ),
        "{refused:?}"
    );
    drop(ticket);

    drop(pass);
    let (refused, n) = counted(|| widen(context.take(1).map(drop)));
    assert_eq!(n, 0, "a cancelled take allocated {n} times");
    assert!(
        matches!(
            refused,
            Err(lloom_tensor::Error::Allocation(AllocationError::Cancelled))
        ),
        "{refused:?}"
    );
}

/// The instrument can see an allocation: a positive control, so the zeros
/// above are a measurement and not a counter that never moves.
#[test]
fn the_counter_sees_a_formatted_refusal() {
    let (_, n) = counted(|| lloom_tensor::Error::Msg(format!("{}", AllocationError::Cancelled)));
    assert!(n > 0, "the counting allocator did not observe format!");
}

/// One value of every `AllocationError` variant. The `match` has no wildcard,
/// so a variant added to the enum fails to compile here until it is listed.
fn every_allocation_refusal() -> Vec<AllocationError> {
    use lloom_tensor::memory_admission::{MachineFailure, PredictionBasis, ReuseDomains};
    let all = vec![
        AllocationError::Refused {
            requested: 9,
            available: 8,
            domain: RefusalDomain::Budget,
        },
        AllocationError::CountExhausted,
        AllocationError::Cancelled,
        AllocationError::DeviceOom { requested: 4096 },
        AllocationError::HostAllocation,
        AllocationError::Overflow,
        AllocationError::Geometry("geometry"),
        AllocationError::PredictionExceeded {
            predicted: 1,
            actual: 2,
            basis: PredictionBasis::HeapSizeAndAlignEstimate,
        },
        AllocationError::Unsupported("unsupported"),
        AllocationError::Retirement("retirement"),
        AllocationError::InvalidCustody,
        AllocationError::UnsupportedLayout,
        AllocationError::DifferentDomain,
        AllocationError::DifferentAllocationDevice,
        AllocationError::UnboundAllocationDevice,
        AllocationError::ReuseDomainMismatch {
            domains: &ReuseDomains {
                expected: "a",
                actual: "b",
            },
        },
        AllocationError::Machine(MachineFailure::Unavailable),
        AllocationError::configuration_conflict(Some(1), None),
    ];
    for e in &all {
        match e {
            AllocationError::Refused { .. }
            | AllocationError::CountExhausted
            | AllocationError::Cancelled
            | AllocationError::DeviceOom { .. }
            | AllocationError::HostAllocation
            | AllocationError::Overflow
            | AllocationError::Geometry(_)
            | AllocationError::PredictionExceeded { .. }
            | AllocationError::Unsupported(_)
            | AllocationError::Retirement(_)
            | AllocationError::InvalidCustody
            | AllocationError::UnsupportedLayout
            | AllocationError::DifferentDomain
            | AllocationError::DifferentAllocationDevice
            | AllocationError::UnboundAllocationDevice
            | AllocationError::ReuseDomainMismatch { .. }
            | AllocationError::Machine(_)
            | AllocationError::ConfigurationConflict { .. } => {}
        }
    }
    all
}

/// Every allocation refusal widens into `Error::Allocation` through `?`
/// without a host allocation, and arrives unchanged.
#[test]
fn every_allocation_refusal_widens_into_error_without_allocating() {
    for refusal in every_allocation_refusal() {
        let (widened, n) = counted(|| widen::<()>(Err(refusal)));
        assert_eq!(n, 0, "{refusal:?} allocated {n} times widening into Error");
        let widened = widened.unwrap_err();
        assert_eq!(widened.allocation(), Some(refusal), "{widened}");
        assert!(
            matches!(widened, lloom_tensor::Error::Allocation(r) if r == refusal),
            "{refusal:?} did not arrive as Error::Allocation: {widened}"
        );
    }
}

/// The kernel layer's refusals reach `Error` through `MetalKernelError` and
/// `MetalError`, as the device's `?` writes it; none of those hops allocates.
#[cfg(feature = "metal")]
mod kernel_layer {
    use super::*;
    use lloom_tensor::kernels::{err::CommandLifecycleError, MetalKernelError};
    use lloom_tensor::MetalError;

    fn widen_kernel<T>(r: Result<T, MetalKernelError>) -> lloom_tensor::Result<T> {
        Ok(r.map_err(MetalError::from)?)
    }

    fn every_recording_refusal() -> [CommandLifecycleError; 6] {
        let all = [
            CommandLifecycleError::NotSubmitted,
            CommandLifecycleError::Busy,
            CommandLifecycleError::Submitted,
            CommandLifecycleError::Abandoned,
            CommandLifecycleError::Exhausted,
            CommandLifecycleError::StaleOwner,
        ];
        for e in all {
            match e {
                CommandLifecycleError::NotSubmitted
                | CommandLifecycleError::Busy
                | CommandLifecycleError::Submitted
                | CommandLifecycleError::Abandoned
                | CommandLifecycleError::Exhausted
                | CommandLifecycleError::StaleOwner => {}
            }
        }
        all
    }

    #[test]
    fn a_recording_refusal_widens_into_error_without_allocating() {
        for refusal in every_recording_refusal() {
            let (widened, n) = counted(|| widen_kernel::<()>(Err(MetalKernelError::from(refusal))));
            assert_eq!(n, 0, "{refusal:?} allocated {n} times widening into Error");
            let widened = widened.unwrap_err();
            assert!(
                matches!(
                    widened,
                    lloom_tensor::Error::Metal(MetalError::KernelError(
                        MetalKernelError::RecordingAdmission(r)
                    )) if r == refusal
                ),
                "{refusal:?} did not arrive as a recording refusal: {widened}"
            );
        }
    }

    fn recording_source(error: &lloom_tensor::Error) -> Option<CommandLifecycleError> {
        let mut next = std::error::Error::source(error);
        while let Some(source) = next {
            if let Some(recording) = source.downcast_ref::<CommandLifecycleError>() {
                return Some(*recording);
            }
            next = source.source();
        }
        None
    }

    /// Exercise both inline kernel refusal families without touching a device.
    #[test]
    fn kernel_refusal_step_annotations_are_allocation_free_and_monotone() {
        use lloom_tensor::StateTouched;
        for touched in [StateTouched::No, StateTouched::Yes] {
            let errors = every_allocation_refusal()
                .into_iter()
                .map(MetalKernelError::from)
                .chain(
                    every_recording_refusal()
                        .into_iter()
                        .map(MetalKernelError::from),
                );
            for error in errors {
                let original = widen_kernel::<()>(Err(error)).unwrap_err();
                let expected_text = original.to_string();
                let expected_origin = original.allocation();
                let expected_sources = diagnostic_sources(&original);
                let expected_recording_source = recording_source(&original);
                let (annotated, n) =
                    counted(|| std::hint::black_box(original.with_state_touched(touched)));
                assert_eq!(
                    n, 0,
                    "annotating kernel refusal allocated {n} times: {expected_text}"
                );
                assert_eq!(annotated.state_touched(), Some(touched));
                assert_eq!(annotated.allocation(), expected_origin);
                assert_eq!(annotated.to_string(), expected_text);
                assert_eq!(
                    diagnostic_sources(&annotated),
                    expected_sources,
                    "annotation lost its diagnostic source chain"
                );
                assert_eq!(
                    recording_source(&annotated),
                    expected_recording_source,
                    "annotation lost the typed recording-admission source"
                );
                let (annotated, n) = counted(|| {
                    std::hint::black_box(annotated.with_state_touched(StateTouched::No))
                });
                assert_eq!(
                    n, 0,
                    "re-annotating kernel refusal allocated {n} times: {expected_text}"
                );
                assert_eq!(annotated.state_touched(), Some(touched));
                assert_eq!(annotated.allocation(), expected_origin);
                assert_eq!(annotated.to_string(), expected_text);
                assert_eq!(
                    diagnostic_sources(&annotated),
                    expected_sources,
                    "annotation lost its diagnostic source chain"
                );
                assert_eq!(
                    recording_source(&annotated),
                    expected_recording_source,
                    "annotation lost the typed recording-admission source"
                );
            }
        }
    }

    #[test]
    fn a_kernel_layer_allocation_refusal_widens_into_error_without_allocating() {
        for refusal in every_allocation_refusal() {
            let (widened, n) = counted(|| widen_kernel::<()>(Err(MetalKernelError::from(refusal))));
            assert_eq!(n, 0, "{refusal:?} allocated {n} times widening into Error");
            let widened = widened.unwrap_err();
            assert!(
                matches!(
                    widened,
                    lloom_tensor::Error::Metal(MetalError::KernelError(
                        MetalKernelError::Allocation(r)
                    )) if r == refusal
                ),
                "{refusal:?} did not arrive as a kernel-layer refusal: {widened}"
            );
        }
    }
}

/// Annotation cannot turn an inline memory refusal into a heap allocation.
/// Existing origin/text inspection and monotone touched-state stay intact.
#[test]
fn allocation_refusal_step_annotations_are_allocation_free_and_monotone() {
    use lloom_tensor::{Error, StateTouched};
    for refusal in every_allocation_refusal() {
        let expected = Error::Allocation(refusal).to_string();
        let expected_sources = diagnostic_sources(&Error::Allocation(refusal));
        for touched in [StateTouched::No, StateTouched::Yes] {
            let (annotated, n) = counted(|| {
                std::hint::black_box(Error::Allocation(refusal).with_state_touched(touched))
            });
            assert_eq!(
                n, 0,
                "annotating {refusal:?} as {touched:?} allocated {n} times"
            );
            assert_eq!(annotated.state_touched(), Some(touched));
            assert_eq!(annotated.allocation(), Some(refusal));
            assert_eq!(annotated.to_string(), expected);
            assert_eq!(
                diagnostic_sources(&annotated),
                expected_sources,
                "annotation lost its diagnostic source chain"
            );
            let (annotated, n) =
                counted(|| std::hint::black_box(annotated.with_state_touched(StateTouched::No)));
            assert_eq!(n, 0, "re-annotating {refusal:?} allocated {n} times");
            assert_eq!(annotated.state_touched(), Some(touched));
            assert_eq!(annotated.allocation(), Some(refusal));
            assert_eq!(annotated.to_string(), expected);
            assert_eq!(
                diagnostic_sources(&annotated),
                expected_sources,
                "annotation lost its diagnostic source chain"
            );
        }
    }
}

/// Existing cold diagnostics retain their box instead of nesting a fresh box
/// at every outer step boundary. This does not price initial cold construction.
#[test]
fn existing_cold_step_box_is_reused_without_an_allocation_or_source_change() {
    use lloom_tensor::{Error, StateTouched};
    fn inner_address(error: &Error) -> usize {
        let Error::StepState { inner, .. } = error else {
            panic!("cold diagnostic lost its typed step wrapper");
        };
        inner.as_ref() as *const Error as usize
    }
    let mut error = Error::Msg("existing diagnostic".into()).with_state_touched(StateTouched::No);
    let address = inner_address(&error);
    let sources = diagnostic_sources(&error);
    for requested in [StateTouched::No, StateTouched::Yes, StateTouched::No] {
        let expected = if error.state_touched() == Some(StateTouched::Yes) {
            StateTouched::Yes
        } else {
            requested
        };
        let (annotated, n) = counted(|| std::hint::black_box(error.with_state_touched(requested)));
        assert_eq!(
            n, 0,
            "reannotating an already boxed cold diagnostic allocated {n} times"
        );
        assert_eq!(
            inner_address(&annotated),
            address,
            "existing diagnostic box was replaced"
        );
        assert_eq!(annotated.state_touched(), Some(expected));
        assert_eq!(annotated.to_string(), "existing diagnostic");
        assert_eq!(diagnostic_sources(&annotated), sources);
        error = annotated;
    }
}
