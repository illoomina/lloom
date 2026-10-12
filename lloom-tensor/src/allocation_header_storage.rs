//! Explicit requested-layout storage for a private, independently rooted header slab.
//! This is storage only: native alias admission and safe rooted handles are NOT
//! supplied here. No Buffer/PooledBuffer factory enrolls through this helper yet.
use crate::allocation_header::{HeaderError, HeaderInventory};
use std::{
    alloc::{alloc, dealloc, Layout},
    cell::UnsafeCell,
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeaderStorageError {
    Header(HeaderError),
    Layout,
    Allocation,
    Index,
    Initialized,
    Empty,
}
struct PayloadCell<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    initialized: AtomicBool,
}

/// Owns exactly one checked Layout allocation. Arc/control/wrapper storage and
/// allocator rounding remain separate accounting obligations. Construction is
/// cold: the eventual adapter must admit requested_layout before allocating.
/// All payload operations require the unsafe contracts below; there is no safe
/// conversion from an arbitrary index/ticket into a native-usable reference.
pub(crate) struct HeaderStorage<T, const N: usize> {
    allocation: NonNull<u8>,
    layout: Layout,
    payload_offset: usize,
    _owned: PhantomData<T>,
}
// SAFETY: allocation stays at one stable address until exclusive final Drop.
// Mutation/shared payload access is possible only through the documented unsafe
// counted-root/PREPARING/DETACHING contracts. Atomics supply metadata handoff.
unsafe impl<T: Send, const N: usize> Send for HeaderStorage<T, N> {}
unsafe impl<T: Send + Sync, const N: usize> Sync for HeaderStorage<T, N> {}

impl<T, const N: usize> HeaderStorage<T, N> {
    pub(crate) fn requested_layout() -> Result<Layout, HeaderStorageError> {
        if N == 0 || N > 64 {
            return Err(HeaderStorageError::Header(HeaderError::InvalidCapacity));
        }
        HeaderInventory::<N>::storage_layout::<PayloadCell<T>>()
            .map(|(layout, _)| layout)
            .map_err(|_| HeaderStorageError::Layout)
    }
    pub(crate) fn allocate() -> Result<Self, HeaderStorageError> {
        // SAFETY: std::alloc::alloc supplies null or storage compatible with
        // dealloc for exactly this checked, nonzero Layout.
        unsafe { Self::allocate_using(alloc) }
    }
    /// Cold allocator seam; production uses std::alloc::alloc, scalar tests may
    /// return null without allocating. It never runs on a producer warm path.
    /// # Safety
    /// Allocator returns null OR a valid allocation for its supplied Layout,
    /// owned uniquely by this result and compatible with std::alloc::dealloc.
    unsafe fn allocate_using(
        allocator: unsafe fn(Layout) -> *mut u8,
    ) -> Result<Self, HeaderStorageError> {
        let inventory = HeaderInventory::<N>::new().map_err(HeaderStorageError::Header)?;
        let (layout, payload_offset) = HeaderInventory::<N>::storage_layout::<PayloadCell<T>>()
            .map_err(|_| HeaderStorageError::Layout)?;
        // SAFETY: new() checked N1..64; metadata makes this Layout nonzero.
        // Exact checked array/extend/padding Layout is retained for deallocation.
        let allocation =
            NonNull::new(unsafe { allocator(layout) }).ok_or(HeaderStorageError::Allocation)?;
        // No T constructor/arbitrary code runs after allocation.
        // SAFETY: the allocator contract supplies unique storage with this exact
        // Layout's alignment/size; checked offset/array stride fits every cell.
        // Metadata and all N cells are initialized before any reference escapes.
        unsafe {
            allocation
                .cast::<HeaderInventory<N>>()
                .as_ptr()
                .write(inventory);
            for index in 0..N {
                allocation
                    .as_ptr()
                    .add(payload_offset)
                    .cast::<PayloadCell<T>>()
                    .add(index)
                    .write(PayloadCell {
                        value: UnsafeCell::new(MaybeUninit::uninit()),
                        initialized: AtomicBool::new(false),
                    });
            }
        }
        Ok(Self {
            allocation,
            layout,
            payload_offset,
            _owned: PhantomData,
        })
    }
    pub(crate) fn inventory(&self) -> &HeaderInventory<N> {
        // SAFETY: allocate initialized metadata before returning, and &self
        // prevents storage destruction through this borrowed accessor.
        unsafe { self.allocation.cast::<HeaderInventory<N>>().as_ref() }
    }
    fn cell(&self, index: usize) -> Result<&PayloadCell<T>, HeaderStorageError> {
        if index >= N {
            return Err(HeaderStorageError::Index);
        }
        // SAFETY: exact offset/stride comes from checked storage Layout; all N
        // cells were initialized before publication. Cell address remains fixed.
        Ok(unsafe {
            &*self
                .allocation
                .as_ptr()
                .add(self.payload_offset)
                .cast::<PayloadCell<T>>()
                .add(index)
        })
    }
    /// # Safety
    /// Caller owns the exclusive genuine PREPARING reservation in THIS root,
    /// no payload accessor can exist, and the root stays alive through this call.
    /// Publish LIVE only after this returns; on refusal/unwind, rollback takes
    /// and roots/destroys T before cancelling the reservation/free-bit claim.
    pub(crate) unsafe fn initialize(
        &self,
        index: usize,
        value: T,
    ) -> Result<(), (HeaderStorageError, T)> {
        let cell = match self.cell(index) {
            Ok(cell) => cell,
            Err(e) => return Err((e, value)),
        };
        if cell.initialized.load(Ordering::Acquire) {
            return Err((HeaderStorageError::Initialized, value));
        }
        // SAFETY: sole PREPARING owner has exclusive uninitialized payload cell.
        unsafe {
            (*cell.value.get()).write(value);
        }
        cell.initialized.store(true, Ordering::Release);
        Ok(())
    }
    /// # Safety
    /// Caller owns one of: (a) genuine counted strong with exact root/generation
    /// and acquired LIVE publication; (b) service sole genuine claimed DETACHING
    /// authority in this root, acquired from cleanup; (c) sole initialized
    /// unpublished PREPARING reservation in this root. Holder/authority/root
    /// outlive reference. No raw reference escapes or mutation/destruction
    /// overlaps; PREPARING borrow ends before publish/cancel/unwind cleanup,
    /// DETACHING borrow ends before take/requeue. PREPARING additionally permits
    /// no native/future-submit capability to escape before actual publish-last.
    pub(crate) unsafe fn payload(&self, index: usize) -> Result<&T, HeaderStorageError> {
        let cell = self.cell(index)?;
        if !cell.initialized.load(Ordering::Acquire) {
            return Err(HeaderStorageError::Empty);
        }
        // SAFETY: genuine strong prevents mutation/destruction, or sole
        // DETACHING/PREPARING authority excludes overlapping payload mutation.
        Ok(unsafe { (&*cell.value.get()).assume_init_ref() })
    }
    /// # Safety
    /// Caller owns exclusive genuine PREPARING rollback OR the service-claimed
    /// DETACHING authority in THIS root. All strong accesses ended (Release
    /// decrements/Acquire fence), upgrade cannot resurrect zero, and no payload
    /// reference remains. Move returned T into root-owned active custody before
    /// arbitrary Drop; retain anchors/charge on unwind. Root spans every access.
    pub(crate) unsafe fn take(&self, index: usize) -> Result<T, HeaderStorageError> {
        let cell = self.cell(index)?;
        if !cell.initialized.swap(false, Ordering::AcqRel) {
            return Err(HeaderStorageError::Empty);
        }
        // SAFETY: exclusive authority transfers T once; there is no arbitrary
        // operation between initialized=false and the nonpanicking value read.
        Ok(unsafe { (&*cell.value.get()).assume_init_read() })
    }
}
impl<T, const N: usize> Drop for HeaderStorage<T, N> {
    fn drop(&mut self) {
        // Published-root final destruction is service-only after its rooted
        // census staged the charge. Empty unpublished cold candidates may drop
        // off-guard too. Unexpected ownership panics BEFORE freeing
        // payload/storage; service retains the charge across that unwind.
        for index in 0..N {
            if self
                .cell(index)
                .expect("constructed index")
                .initialized
                .load(Ordering::Acquire)
            {
                panic!("header storage still owns initialized payload");
            }
        }
        // SAFETY: final &mut self plus independently rooted census excludes all
        // accessors. No T remains initialized; MaybeUninit never drops T. Drop
        // only the real metadata/cell objects before exact same-Layout free.
        unsafe {
            ptr::drop_in_place(self.allocation.cast::<HeaderInventory<N>>().as_ptr());
            for index in 0..N {
                ptr::drop_in_place(
                    self.allocation
                        .as_ptr()
                        .add(self.payload_offset)
                        .cast::<PayloadCell<T>>()
                        .add(index),
                );
            }
            dealloc(self.allocation.as_ptr(), self.layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[repr(align(64))]
    #[derive(Debug, PartialEq, Eq)]
    struct Wide([u64; 8]);

    #[test]
    fn over_aligned_payload_is_published_then_taken_once() {
        let storage = HeaderStorage::<Wide, 3>::allocate().unwrap();
        let claim = storage.inventory().reserve().unwrap().unwrap();
        let index = claim.index();
        // SAFETY: sole exact-root PREPARING reservation; no payload borrower.
        unsafe { storage.initialize(index, Wide([17; 8])) }.unwrap();
        let ticket = claim.publish().unwrap();
        {
            // SAFETY: this genuine counted strong ticket belongs to storage;
            // borrowed payload ends before its last strong is released.
            let payload = unsafe { storage.payload(index) }.unwrap();
            assert_eq!((payload as *const Wide as usize) % 64, 0);
            assert_eq!(payload, &Wide([17; 8]));
        }
        let authority = storage.inventory().release_strong(ticket).unwrap().unwrap();
        storage.inventory().publish_cleanup(authority).unwrap();
        let authority = storage.inventory().claim_cleanup().unwrap().unwrap();
        assert_eq!(authority.index(), index);
        // SAFETY: sole service-claimed DETACHING authority, no strong references.
        let payload = unsafe { storage.take(index) }.unwrap();
        assert_eq!(payload, Wide([17; 8]));
        // SAFETY: same exclusive authority still held; cell is already empty.
        assert_eq!(
            unsafe { storage.take(index) },
            Err(HeaderStorageError::Empty)
        );
        drop(payload);
        storage.inventory().detached(authority).unwrap();
        assert_eq!(storage.inventory().available(), 3);
        drop(storage);
    }

    #[test]
    fn null_allocation_and_unusable_capacity_are_refused() {
        unsafe fn refuse(_: Layout) -> *mut u8 {
            ptr::null_mut()
        }
        // SAFETY: null-only cold allocator performs no allocation or side effect.
        assert!(matches!(
            unsafe { HeaderStorage::<u64, 3>::allocate_using(refuse) },
            Err(HeaderStorageError::Allocation)
        ));
        for result in [
            HeaderStorage::<u64, 0>::requested_layout(),
            HeaderStorage::<u64, 65>::requested_layout(),
        ] {
            assert_eq!(
                result,
                Err(HeaderStorageError::Header(HeaderError::InvalidCapacity))
            );
        }
    }

    #[derive(Debug)]
    struct Witness<'a>(&'a AtomicUsize);
    impl Drop for Witness<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn preparing_rollback_clears_payload_before_restoring_credit() {
        let drops = AtomicUsize::new(0);
        let storage = HeaderStorage::<Witness<'_>, 3>::allocate().unwrap();
        let claim = storage.inventory().reserve().unwrap().unwrap();
        let index = claim.index();
        // SAFETY: sole genuine exact-root PREPARING reservation.
        unsafe { storage.initialize(index, Witness(&drops)) }.unwrap();
        // SAFETY: same exclusive PREPARING rollback reservation.
        let payload = unsafe { storage.take(index) }.unwrap();
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        // SAFETY: same reservation, empty payload; refusal does not transfer T.
        assert!(matches!(
            unsafe { storage.take(index) },
            Err(HeaderStorageError::Empty)
        ));
        assert_eq!(storage.inventory().available(), 2);
        drop(payload);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        claim.cancel().unwrap();
        assert_eq!(storage.inventory().available(), 3);
        drop(storage);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    /// Bounded fixture cleanup, NOT production retry: the known early Drop
    /// panic leaves one raw block allocated. Only after observing that exact
    /// panic do we arm this independent guard. No HeaderStorage is reconstructed
    /// or re-dropped; no arbitrary payload destructor runs (payload is u64).
    struct QuarantinedBlock {
        allocation: NonNull<u8>,
        layout: Layout,
        offset: usize,
        index: usize,
        armed: bool,
    }
    impl Drop for QuarantinedBlock {
        fn drop(&mut self) {
            if !self.armed {
                return;
            }
            // SAFETY: exact initialized-payload panic precedes all raw free/drop;
            // storage owner has unwound, no other holder exists, saved Layout and
            // pointers describe this one live allocation. T=u64 has no Drop.
            unsafe {
                let first = self
                    .allocation
                    .as_ptr()
                    .add(self.offset)
                    .cast::<PayloadCell<u64>>();
                let cell = &*first.add(self.index);
                cell.initialized.store(false, Ordering::Relaxed);
                let _payload = (&*cell.value.get()).assume_init_read();
                ptr::drop_in_place(self.allocation.cast::<HeaderInventory<3>>().as_ptr());
                for index in 0..3 {
                    ptr::drop_in_place(first.add(index));
                }
                dealloc(self.allocation.as_ptr(), self.layout);
            }
        }
    }
    #[test]
    fn actual_initialized_final_drop_quarantines_before_storage_free() {
        let storage = HeaderStorage::<u64, 3>::allocate().unwrap();
        let claim = storage.inventory().reserve().unwrap().unwrap();
        let index = claim.index();
        // SAFETY: sole genuine exact-root PREPARING reservation.
        unsafe { storage.initialize(index, 41) }.unwrap();
        // Dropped unfinished claim intentionally quarantines metadata; no credit.
        drop(claim);
        let mut cleanup = QuarantinedBlock {
            allocation: storage.allocation,
            layout: storage.layout,
            offset: storage.payload_offset,
            index,
            armed: false,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(storage)));
        let known_early_panic = result.as_ref().err().is_some_and(|panic| {
            panic.downcast_ref::<&str>().copied()
                == Some("header storage still owns initialized payload")
        });
        // Normal return means Drop already freed the allocation: NEVER arm/free
        // again. Unexpected panic keeps at most this one fixed block quarantined
        // and fails; no unbounded allocation/retry loop exists in this fixture.
        cleanup.armed = known_early_panic;
        assert!(
            known_early_panic,
            "actual storage Drop must refuse initialized payload before free"
        );
        // SAFETY: known early panic proves block/cell remain initialized/live;
        // independent cleanup guard owns all remaining raw storage exclusively.
        let cell = unsafe {
            &*cleanup
                .allocation
                .as_ptr()
                .add(cleanup.offset)
                .cast::<PayloadCell<u64>>()
                .add(index)
        };
        assert!(cell.initialized.load(Ordering::Acquire));
        // SAFETY: uniquely owned retained u64 cell; no mutation overlaps read.
        assert_eq!(unsafe { (&*cell.value.get()).assume_init_ref() }, &41);
        drop(cleanup);
    }
}
