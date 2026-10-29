//! Exact requested-layout fixed host scratch. No Vec growth or post-allocation
//! capacity guess. Existing byte-shadow callers retain the same API/behavior.
use crate::memory_admission::{
    AllocationContext, AllocationError, AllocationTicket, PhysicalCharge,
};
use std::alloc::{alloc_zeroed, Layout};

#[derive(Debug)]
pub(crate) enum ColdBytesError {
    Admission(AllocationError),
    Layout,
    Allocation,
}

mod scalar {
    // Closed set: zero bits are valid, no destructor, alignment known by Layout.
    pub trait Sealed: Copy {}
    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
}
pub(crate) trait ScratchScalar: scalar::Sealed {}
impl ScratchScalar for u8 {}
impl ScratchScalar for u32 {}
impl ScratchScalar for f32 {}

/// Field order is custody: exact slice storage frees before its physical charge
/// and control allocation. Final Drop belongs off engine/encoder/producer guards.
/// This host-only owner is not a native-use/retirement capability.
pub(crate) struct ColdScratch<T: ScratchScalar> {
    values: Box<[T]>,
    _charge: PhysicalCharge,
}
/// Preserve the existing private byte-shadow API without a second allocator.
pub(crate) type ColdBytes = ColdScratch<u8>;

impl<T: ScratchScalar> ColdScratch<T> {
    fn layout(len: usize) -> Result<Layout, ColdBytesError> {
        if len == 0 {
            return Err(ColdBytesError::Layout);
        }
        Layout::array::<T>(len).map_err(|_| ColdBytesError::Layout)
    }
    pub(crate) fn requested_bytes(len: usize) -> Result<usize, ColdBytesError> {
        Self::layout(len)?
            .size()
            .checked_add(
                AllocationTicket::requested_charge_control_bytes()
                    .map_err(ColdBytesError::Admission)?,
            )
            .ok_or(ColdBytesError::Layout)
    }
    pub(crate) fn zeroed_in(
        context: &AllocationContext,
        len: usize,
    ) -> Result<Self, ColdBytesError> {
        // SAFETY: Global zeroed allocator obeys exact Layout/deallocation and
        // null semantics; the production commit retains existing ticket policy.
        unsafe {
            Self::zeroed_with(context, len, alloc_zeroed, |ticket, bytes| {
                ticket.try_commit_with_control(bytes)
            })
        }
    }

    /// Narrow cold birth seam shared with fault controls, not an allocator API.
    /// # Safety
    /// `allocate` returns null without allocation, or unique Global-compatible
    /// zero-initialized storage for exactly Layout. It never unwinds after birth.
    /// `commit` uses this ticket for the supplied payload bytes, or returns Err
    /// without creating an unrelated charge; it must not retain the ticket.
    unsafe fn zeroed_with(
        context: &AllocationContext,
        len: usize,
        allocate: unsafe fn(Layout) -> *mut u8,
        commit: impl FnOnce(&mut AllocationTicket, usize) -> Result<PhysicalCharge, AllocationError>,
    ) -> Result<Self, ColdBytesError> {
        let quote = Self::requested_bytes(len)?;
        let mut ticket = context.take(quote).map_err(ColdBytesError::Admission)?;
        let layout = Self::layout(len)?;
        // SAFETY: checked nonzero Layout and caller's exact zeroed Global contract.
        let raw = std::ptr::NonNull::new(unsafe { allocate(layout) }.cast::<T>())
            .ok_or(ColdBytesError::Allocation)?;
        // SAFETY: sealed scalar set u8/u32/f32 accepts zero bits. Alignment and
        // size derive from T; Box<[T]> frees this same Global Layout exactly.
        let values =
            unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw.as_ptr(), len)) };
        let charge = match commit(&mut ticket, layout.size()) {
            Ok(charge) => charge,
            Err(error) => {
                drop(values); // actual storage free BEFORE unused ticket refund
                return Err(ColdBytesError::Admission(error));
            }
        };
        Ok(Self {
            values,
            _charge: charge,
        })
    }
}
impl<T: ScratchScalar> std::ops::Deref for ColdScratch<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T: ScratchScalar> std::ops::DerefMut for ColdScratch<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{AccountSnapshot, ModelDomain};

    #[test]
    fn fixed_scalar_zero_and_overflow_refuse_before_quote() {
        for result in [
            ColdBytes::requested_bytes(0),
            ColdScratch::<u32>::requested_bytes(0),
            ColdScratch::<f32>::requested_bytes(0),
            ColdBytes::requested_bytes(usize::MAX),
            ColdScratch::<u32>::requested_bytes(usize::MAX / 4 + 1),
            ColdScratch::<f32>::requested_bytes(usize::MAX / 4 + 1),
        ] {
            assert!(matches!(result, Err(ColdBytesError::Layout)));
        }
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_quote_is_exact_layout_plus_existing_control() {
        let control = AllocationTicket::requested_charge_control_bytes().unwrap();
        assert_eq!(ColdBytes::requested_bytes(7).unwrap(), 7 + control);
        assert_eq!(
            ColdScratch::<u32>::requested_bytes(7).unwrap(),
            28 + control
        );
        assert_eq!(
            ColdScratch::<f32>::requested_bytes(7).unwrap(),
            28 + control
        );
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_u32_f32_alignment_zero_and_mutation() {
        let quote = ColdScratch::<u32>::requested_bytes(7).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(2 * quote);
        let pass = domain.reserve(2 * quote).unwrap();
        let mut ints = ColdScratch::<u32>::zeroed_in(&pass.context(), 7).unwrap();
        let mut floats = ColdScratch::<f32>::zeroed_in(&pass.context(), 7).unwrap();
        assert_eq!(ints.as_ptr() as usize % std::mem::align_of::<u32>(), 0);
        assert_eq!(floats.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
        assert_eq!(&*ints, &[0; 7]);
        assert!(floats.iter().all(|value| value.to_bits() == 0));
        ints[6] = u32::MAX;
        floats[6] = -3.5;
        assert_eq!(ints[6], u32::MAX);
        assert_eq!(floats[6].to_bits(), (-3.5f32).to_bits());
        drop(pass); // already issued charges survive cancellation
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 2 * quote
            }
        );
        drop(ints);
        drop(floats);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_existing_bytes_api_keeps_zeroed_mutable_slice() {
        let quote = ColdBytes::requested_bytes(3).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let mut bytes = ColdBytes::zeroed_in(&pass.context(), 3).unwrap();
        assert_eq!(&*bytes, &[0, 0, 0]);
        bytes.copy_from_slice(&[7, 0, 255]);
        assert_eq!(&*bytes, &[7, 0, 255]);
        drop(bytes);
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    unsafe fn null_without_allocation(_: Layout) -> *mut u8 {
        std::ptr::null_mut()
    }
    unsafe fn must_not_allocate(_: Layout) -> *mut u8 {
        panic!("allocation before entitlement")
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_short_promise_and_closed_context_never_allocate() {
        let quote = ColdScratch::<f32>::requested_bytes(7).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote - 1).unwrap();
        let context = pass.context();
        // SAFETY: sentinel never returns storage and only panics if invoked;
        // caller must reject before allocation. Commit is unchanged production.
        let attempt = || unsafe {
            ColdScratch::<f32>::zeroed_with(&context, 7, must_not_allocate, |ticket, bytes| {
                ticket.try_commit_with_control(bytes)
            })
        };
        assert!(matches!(attempt(), Err(ColdBytesError::Admission(_))));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: quote - 1,
                committed: 0
            }
        );
        drop(pass);
        assert!(matches!(attempt(), Err(ColdBytesError::Admission(_))));
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_null_returns_issued_demand() {
        let quote = ColdScratch::<u32>::requested_bytes(7).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        // SAFETY: injected null allocates nothing; production commit unchanged.
        let failed = unsafe {
            ColdScratch::<u32>::zeroed_with(
                &pass.context(),
                7,
                null_without_allocation,
                |ticket, bytes| ticket.try_commit_with_control(bytes),
            )
        };
        assert!(matches!(failed, Err(ColdBytesError::Allocation)));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: quote,
                committed: 0
            }
        );
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    fn assert_clean_observer(report: &crate::free_observer::Report) {
        assert!(
            !report.overflow
                && !report.reallocated
                && !report.wrong_layout
                && !report.duplicate_free
        );
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_actual_free_precedes_charge_refund() {
        let quote = ColdScratch::<f32>::requested_bytes(7).unwrap();
        let control = AllocationTicket::requested_charge_control_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let report = crate::free_observer::observe(&domain, |watch| {
            let scratch = ColdScratch::<f32>::zeroed_in(&pass.context(), 7).unwrap();
            watch.watch();
            drop(scratch);
        });
        assert_clean_observer(&report);
        assert_eq!((report.born, report.freed), (2, 2));
        assert_eq!(
            (report.blocks[0].size, report.blocks[0].align),
            (28, std::mem::align_of::<f32>())
        );
        assert_eq!(report.blocks[1].size, control);
        assert_eq!(
            &report.order[..2],
            &[0, 1],
            "scratch must free before charge control"
        );
        for snapshot in &report.snapshots[..2] {
            assert_eq!(
                *snapshot,
                Some(AccountSnapshot {
                    pending: 0,
                    committed: quote
                }),
                "actual storage free must precede charge refund"
            );
        }
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn fixed_scalar_commit_refusal_frees_before_ticket_refund() {
        let quote = ColdScratch::<u32>::requested_bytes(7).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let context = pass.context();
        let report = crate::free_observer::observe(&domain, |watch| {
            // SAFETY: actual Global zeroed birth; synthetic commit refusal does
            // not create a charge or modify/retain the issued ticket.
            let failed = unsafe {
                ColdScratch::<u32>::zeroed_with(&context, 7, alloc_zeroed, |_, _| {
                    // Close unused issuance while the actual ticket still owns
                    // the full demand: premature ticket Drop now refunds to zero.
                    drop(pass);
                    watch.watch();
                    Err(AllocationError::Overflow)
                })
            };
            assert!(matches!(
                failed,
                Err(ColdBytesError::Admission(AllocationError::Overflow))
            ));
        });
        assert_clean_observer(&report);
        assert_eq!((report.born, report.freed), (1, 1));
        assert_eq!(
            (report.blocks[0].size, report.blocks[0].align),
            (28, std::mem::align_of::<u32>())
        );
        assert_eq!(
            report.snapshots[0],
            Some(AccountSnapshot {
                pending: quote,
                committed: 0
            }),
            "rejected storage must free while its ticket still retains entitlement"
        );
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn fixed_scalar_unsupported_control_layout_refuses_before_birth() {
        assert!(matches!(
            ColdScratch::<f32>::requested_bytes(7),
            Err(ColdBytesError::Admission(
                AllocationError::UnsupportedLayout
            ))
        ));
    }
}
