//! Concrete fixed sampler-slot ownership; proposed cross-crate preparation API.
use super::prepared::PreparedSampler;
use crate::memory_admission::{
    AllocationContext, AllocationError, AllocationTicket, PhysicalCharge,
};
use crate::metal_backend::{allocation_scope, MetalDevice};
use crate::Result;
use std::{
    alloc::{alloc, Layout},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

// Private initialized-prefix storage. Unlike ColdScratch, T may own resources;
// neither zero bytes nor an uninitialized element is ever exposed as T.
struct InitializedSlots<T> {
    storage: Box<[MaybeUninit<T>]>,
    initialized: usize,
    charge: Option<PhysicalCharge>,
}
impl<T> InitializedSlots<T> {
    fn layout(count: usize) -> std::result::Result<Layout, AllocationError> {
        if count == 0 {
            return Err(AllocationError::Overflow);
        }
        let layout = Layout::array::<T>(count).map_err(|_| AllocationError::Overflow)?;
        if layout.size() == 0 {
            return Err(AllocationError::Overflow);
        }
        Ok(layout)
    }
    fn requested_bytes(count: usize) -> std::result::Result<usize, AllocationError> {
        Self::layout(count)?
            .size()
            .checked_add(AllocationTicket::requested_charge_control_bytes()?)
            .ok_or(AllocationError::Overflow)
    }
    fn into_values(self) -> SlotValues<T> {
        assert_eq!(
            self.initialized,
            self.storage.len(),
            "only complete storage may publish"
        );
        // Move the exact Global allocation into the existing fixed Vec owner,
        // without allocation/copy/shrink. No operation after suppression can fail.
        let owner = std::mem::ManuallyDrop::new(self);
        let count = owner.initialized;
        // SAFETY: `owner` is never dropped, so each field is moved out exactly
        // once and no destructor of `InitializedSlots` runs afterwards.
        let storage = unsafe { std::ptr::read(&owner.storage) };
        let charge = unsafe { std::ptr::read(&owner.charge) };
        let raw = Box::into_raw(storage).cast::<MaybeUninit<T>>().cast::<T>();
        // SAFETY: `raw` is the Global allocation of `Layout::array::<T>(count)`
        // (`MaybeUninit<T>` has `T`'s size and alignment), all `count`
        // elements are initialized (asserted above), and the capacity equals
        // the allocation's element count, so `Vec` frees it with that layout.
        SlotValues {
            values: unsafe { Vec::from_raw_parts(raw, count, count) },
            _charge: charge,
        }
    }
    fn new(
        context: &AllocationContext,
        count: usize,
        make: impl FnMut() -> Result<T>,
    ) -> Result<Self> {
        // SAFETY: Global alloc obeys exact Layout/null/Box deallocation contract.
        unsafe {
            Self::new_with(context, count, make, alloc, |ticket, bytes| {
                ticket.try_commit_with_control(bytes)
            })
        }
    }
    // Narrow cold fault seam; no public allocation API. Allocator cannot unwind
    // after birth and returned storage must be Global-compatible for Layout.
    unsafe fn new_with(
        context: &AllocationContext,
        count: usize,
        mut make: impl FnMut() -> Result<T>,
        allocate: unsafe fn(Layout) -> *mut u8,
        commit: impl FnOnce(
            &mut AllocationTicket,
            usize,
        ) -> std::result::Result<PhysicalCharge, AllocationError>,
    ) -> Result<Self> {
        let layout = Self::layout(count)?;
        let mut ticket = context.take(Self::requested_bytes(count)?)?;
        // SAFETY: the caller supplies a Global-compatible allocator; `layout`
        // has a nonzero size (checked by `Self::layout`).
        let raw = std::ptr::NonNull::new(unsafe { allocate(layout) }.cast::<MaybeUninit<T>>())
            .ok_or(AllocationError::HostAllocation)?;
        // SAFETY: `raw` is a live, uniquely owned allocation of exactly
        // `count` `MaybeUninit<T>`, which `Box` frees with the same layout.
        let storage =
            unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw.as_ptr(), count)) };
        let mut owner = Self {
            storage,
            initialized: 0,
            charge: None,
        };
        // Storage is charged BEFORE child creation; later child refusal/unwind
        // destroys initialized prefix + exact backing before refunding this charge.
        owner.charge = match commit(&mut ticket, layout.size()) {
            Ok(charge) => Some(charge),
            Err(error) => {
                drop(owner);
                return Err(error.into());
            }
        };
        for _ in 0..count {
            let value = make()?;
            owner.storage[owner.initialized].write(value);
            owner.initialized += 1;
        }
        Ok(owner)
    }
}
impl<T> Drop for InitializedSlots<T> {
    fn drop(&mut self) {
        // Slice drop glue also drops remaining initialized elements if one
        // destructor unwinds. Box<MaybeUninit> then frees the exact allocation,
        // and only afterward does the final charge field release its receipt.
        // SAFETY: exactly the first `initialized` elements were written and
        // none has been moved out (publication suppresses this destructor).
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.storage.as_mut_ptr().cast::<T>(),
                self.initialized,
            ));
        }
    }
}
impl<T> Deref for InitializedSlots<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the first `initialized` elements are initialized `T`.
        unsafe { std::slice::from_raw_parts(self.storage.as_ptr().cast::<T>(), self.initialized) }
    }
}
impl<T> DerefMut for InitializedSlots<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: the first `initialized` elements are initialized `T`.
        unsafe {
            std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast::<T>(), self.initialized)
        }
    }
}

/// Proposed concrete API, not a public generic allocator or admission policy.
/// Fixed count; no growth or allocation in indexing/reuse. Native resources keep
/// their existing independent retirement custody; this pays host slot storage.
/// Fields free the actual Vec backing BEFORE its accounting receipt.
struct SlotValues<T> {
    values: Vec<T>,
    // Held for its drop: the receipt is released after `values` frees.
    _charge: Option<PhysicalCharge>,
}
pub struct PreparedSamplerSlots(SlotValues<PreparedSampler>);
impl PreparedSamplerSlots {
    pub fn with_noise(
        device: &MetalDevice,
        rows: usize,
        vocab: usize,
        count: usize,
    ) -> Result<Self> {
        if count == 0 {
            crate::bail!("native sampling: no slots");
        }
        PreparedSampler::payload_bytes(rows, vocab)?;
        if let Some(scope) = allocation_scope::active(device)? {
            let owner = InitializedSlots::new(&scope.context, count, || {
                PreparedSampler::with_noise(device, rows, vocab)
            })?;
            return Ok(Self(owner.into_values()));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| crate::Error::Msg(format!("native sampling slots: {e}")))?;
        for _ in 0..count {
            values.push(PreparedSampler::with_noise(device, rows, vocab)?);
        }
        Ok(Self(SlotValues {
            values,
            _charge: None,
        }))
    }
}
impl Deref for PreparedSamplerSlots {
    type Target = [PreparedSampler];
    fn deref(&self) -> &Self::Target {
        &self.0.values
    }
}
impl DerefMut for PreparedSamplerSlots {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{ModelDomain, RefusalDomain};
    use std::{cell::Cell, rc::Rc};
    struct Probe {
        dropped: Rc<Cell<usize>>,
        domain: ModelDomain,
        min_live: usize,
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            assert!(
                self.domain.snapshot().committed >= self.min_live,
                "SLOT_REFUND_BEFORE_CHILD_DROP"
            );
            self.dropped.set(self.dropped.get() + 1);
        }
    }
    unsafe fn null(_: Layout) -> *mut u8 {
        std::ptr::null_mut()
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn slot_prefix_error_unwind_and_final_drop_keep_backing_paid() {
        for stop in [0, 1, 3] {
            let domain = ModelDomain::isolated_numeric_fixture(1 << 20);
            let base = domain.snapshot();
            let quote = InitializedSlots::<Probe>::requested_bytes(3).unwrap();
            let pass = domain.reserve(quote).unwrap();
            let dropped = Rc::new(Cell::new(0));
            let mut made = 0;
            let result = InitializedSlots::new(&pass.context(), 3, || {
                if made == stop {
                    crate::bail!("injected child refusal");
                }
                made += 1;
                Ok(Probe {
                    dropped: dropped.clone(),
                    domain: domain.clone(),
                    min_live: quote,
                })
            });
            if stop < 3 {
                assert!(result.is_err());
                assert_eq!(dropped.get(), stop);
            } else {
                let owner = result.unwrap();
                let address = owner.storage.as_ptr().cast::<Probe>();
                let published = owner.into_values();
                let values = &published.values;
                assert_eq!(values.as_ptr(), address, "SLOT_CONVERSION_REALLOCATED");
                assert_eq!(values.capacity(), 3);
                drop(pass);
                assert_eq!(values.len(), 3);
                assert_eq!(domain.snapshot().committed, quote);
                drop(published);
                assert_eq!(dropped.get(), 3);
                assert_eq!(domain.snapshot(), base);
                continue;
            }
            drop(pass);
            assert_eq!(domain.snapshot(), base);
        }
        let domain = ModelDomain::isolated_numeric_fixture(1 << 20);
        let base = domain.snapshot();
        let quote = InitializedSlots::<Probe>::requested_bytes(3).unwrap();
        let pass = domain.reserve(quote).unwrap();
        let dropped = Rc::new(Cell::new(0));
        let mut made = 0;
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = InitializedSlots::new(&pass.context(), 3, || {
                if made == 1 {
                    panic!("injected child unwind");
                }
                made += 1;
                Ok(Probe {
                    dropped: dropped.clone(),
                    domain: domain.clone(),
                    min_live: quote,
                })
            });
        }));
        assert!(panic.is_err());
        assert_eq!(dropped.get(), 1);
        drop(pass);
        assert_eq!(domain.snapshot(), base);
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn slot_refusal_null_and_commit_failure_never_create_children() {
        let domain = ModelDomain::isolated_numeric_fixture(1 << 20);
        let base = domain.snapshot();
        let quote = InitializedSlots::<u32>::requested_bytes(3).unwrap();
        let called = Cell::new(0);
        let pass = domain.reserve(quote - 1).unwrap();
        let error = InitializedSlots::new(&pass.context(), 3, || {
            called.set(called.get() + 1);
            Ok(7u32)
        })
        .err()
        .unwrap();
        assert!(
            matches!(error,crate::Error::Allocation(AllocationError::Refused {requested,available,domain:RefusalDomain::Promise}) if requested==quote&&available==quote-1)
        );
        drop(pass);
        assert_eq!(domain.snapshot(), base);
        for fail_commit in [false, true] {
            let pass = domain.reserve(quote).unwrap();
            let result = unsafe {
                InitializedSlots::new_with(
                    &pass.context(),
                    3,
                    || {
                        called.set(called.get() + 1);
                        Ok(7u32)
                    },
                    if fail_commit { alloc } else { null },
                    |ticket, bytes| {
                        if fail_commit {
                            Err(AllocationError::Cancelled)
                        } else {
                            ticket.try_commit_with_control(bytes)
                        }
                    },
                )
            };
            assert!(result.is_err());
            assert_eq!(called.get(), 0);
            drop(pass);
            assert_eq!(domain.snapshot(), base);
        }
        assert!(InitializedSlots::<u32>::layout(0).is_err());
        assert!(InitializedSlots::<u32>::layout(usize::MAX).is_err());
    }

    fn clean_report(report: &crate::free_observer::Report) {
        assert!(
            !report.overflow
                && !report.reallocated
                && !report.wrong_layout
                && !report.duplicate_free
        );
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn slot_actual_free_precedes_refund_before_and_after_publication() {
        use crate::memory_admission::AccountSnapshot;
        for publish in [false, true] {
            let quote = InitializedSlots::<u32>::requested_bytes(3).unwrap();
            let domain = ModelDomain::isolated_numeric_fixture(quote);
            let pass = domain.reserve(quote).unwrap();
            let report = crate::free_observer::observe(&domain, |watch| {
                let initialized = InitializedSlots::new(&pass.context(), 3, || Ok(7u32)).unwrap();
                let address = initialized.storage.as_ptr().cast::<u32>();
                watch.watch();
                if publish {
                    let values = initialized.into_values();
                    assert_eq!(values.values.as_ptr(), address);
                    assert_eq!(values.values.capacity(), 3);
                    assert_eq!(values.values.as_slice(), &[7, 7, 7]);
                    drop(values);
                } else {
                    drop(initialized);
                }
            });
            clean_report(&report);
            assert_eq!((report.born, report.freed), (2, 2));
            assert_eq!(
                (report.blocks[0].size, report.blocks[0].align),
                (12, std::mem::align_of::<u32>())
            );
            assert_eq!(
                report.blocks[1].size,
                AllocationTicket::requested_charge_control_bytes().unwrap()
            );
            assert_eq!(
                &report.order[..2],
                &[0, 1],
                "SLOT_ACTUAL_BACKING_FREE_BEFORE_CHARGE"
            );
            for snapshot in &report.snapshots[..2] {
                assert_eq!(
                    *snapshot,
                    Some(AccountSnapshot {
                        pending: 0,
                        committed: quote
                    }),
                    "SLOT_ACTUAL_FREE_REFUNDED_EARLY"
                );
            }
            drop(pass);
            assert_eq!(domain.snapshot(), AccountSnapshot::default());
        }
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn slot_commit_refusal_actual_free_keeps_ticket() {
        use crate::memory_admission::AccountSnapshot;
        let quote = InitializedSlots::<u32>::requested_bytes(3).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let context = pass.context();
        let report = crate::free_observer::observe(&domain, |watch| {
            let result = unsafe {
                InitializedSlots::<u32>::new_with(
                    &context,
                    3,
                    || panic!("no child on commit refusal"),
                    alloc,
                    |_, _| {
                        drop(pass);
                        watch.watch();
                        Err(AllocationError::Cancelled)
                    },
                )
            };
            assert!(matches!(
                result,
                Err(crate::Error::Allocation(AllocationError::Cancelled))
            ));
        });
        clean_report(&report);
        assert_eq!((report.born, report.freed), (1, 1));
        assert_eq!(
            report.snapshots[0],
            Some(AccountSnapshot {
                pending: quote,
                committed: 0
            }),
            "SLOT_REJECTED_FREE_AFTER_TICKET_REFUND"
        );
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
}
