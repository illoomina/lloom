//! Header-slot metadata for independently rooted allocation slabs.
//! Weak headers pin only their slot, not native-alias reuse eligibility.
//! Payload moves/admission/service rooting belong to the allocator adapter.
use std::alloc::{Layout, LayoutError};
use std::sync::atomic::{fence, AtomicU64, AtomicUsize, Ordering};

const SHIFT: usize = 3;
const MASK: usize = (1 << SHIFT) - 1;
const VACANT: usize = 0;
const PREPARING: usize = 1;
const LIVE: usize = 2;
const DETACHING: usize = 3;
const WEAK_ONLY: usize = 4;
const EXHAUSTED: usize = 5;
const MAX_GENERATION: usize = usize::MAX >> SHIFT;
const MAX_REFS: usize = isize::MAX as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeaderTicket {
    index: usize,
    generation: usize,
}
impl HeaderTicket {
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}
/// Exclusive last-strong custody. A producer must place this authority in the
/// rooted cleanup cell before publishing its ready bit. The service consumes it
/// only after sidecar destruction. It is not cloneable or a replacement strong
/// handle; dropping it unfinished retains/quarantines DETACHING and its implicit
/// weak instead of republishing header storage.
#[must_use]
pub(crate) struct HeaderDetachment {
    ticket: HeaderTicket,
}
impl HeaderDetachment {
    pub(crate) fn index(&self) -> usize {
        self.ticket.index
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeaderError {
    InvalidCapacity,
    Stale,
    Phase,
    Count,
}
/// A counted handle whose count would overflow is a refusal; every other
/// header error is a custody invariant the caller broke.
impl From<HeaderError> for crate::memory_admission::AllocationError {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::Count => Self::CountExhausted,
            _ => Self::InvalidCustody,
        }
    }
}
/// Exclusive cold claim. It cannot be cloned into competing publishers or
/// cancellers. Dropping an unfinished claim quarantines its slot; the payload
/// adapter must install a rollback guard before initialization to cancel after
/// payload cleanup on refusal/unwind. No implicit Drop may republish live data.
#[must_use]
pub(crate) struct HeaderReservation<'a, const N: usize> {
    inventory: &'a HeaderInventory<N>,
    ticket: HeaderTicket,
}
impl<const N: usize> HeaderReservation<'_, N> {
    pub(crate) fn index(&self) -> usize {
        self.ticket.index
    }
    pub(crate) fn cancel(self) -> Result<(), HeaderError> {
        self.inventory.cancel(self.ticket)
    }
    pub(crate) fn publish(self) -> Result<HeaderTicket, HeaderError> {
        self.inventory.publish(self.ticket)
    }
}
struct Counts {
    control: AtomicUsize,
    strong: AtomicUsize,
    weak: AtomicUsize,
}
impl Counts {
    fn new() -> Self {
        Self {
            control: AtomicUsize::new(VACANT),
            strong: AtomicUsize::new(0),
            weak: AtomicUsize::new(0),
        }
    }
    fn state(&self, generation: usize) -> Result<usize, HeaderError> {
        let state = self.control.load(Ordering::Acquire);
        if state >> SHIFT != generation {
            return Err(HeaderError::Stale);
        }
        Ok(state & MASK)
    }
    fn transition(&self, generation: usize, from: usize, to: usize) -> Result<(), HeaderError> {
        self.control
            .compare_exchange(
                generation << SHIFT | from,
                generation << SHIFT | to,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|state| {
                if state >> SHIFT != generation {
                    HeaderError::Stale
                } else {
                    HeaderError::Phase
                }
            })
    }
    fn increment(count: &AtomicUsize) -> Result<(), HeaderError> {
        count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n != 0 && n < MAX_REFS).then(|| n + 1)
            })
            .map(|_| ())
            .map_err(|_| HeaderError::Count)
    }
    fn decrement(count: &AtomicUsize) -> Result<bool, HeaderError> {
        count
            .fetch_update(Ordering::Release, Ordering::Relaxed, |n| n.checked_sub(1))
            .map(|n| {
                if n == 1 {
                    fence(Ordering::Acquire);
                    true
                } else {
                    false
                }
            })
            .map_err(|_| HeaderError::Count)
    }
}

/// Reservations remove one free bit before touching its slot. Successful
/// payload detachment precedes implicit-weak release; final weak release alone
/// republishes the bit. Arc-held slab custody must cover every method call.
/// Clone/downgrade/release require the adapter's existing counted handle;
/// HeaderTicket is an identity check, never an independently acquirable handle.
/// The adapter must not expose payloads before publish or restore stale tickets.
pub(crate) struct HeaderInventory<const N: usize> {
    free: AtomicU64,
    // Producer initializes its existing header cleanup payload before this
    // Release bit. Single service claim transfers DETACHING authority; the
    // implicit weak remains until the claimed sidecars are actually destroyed.
    cleanup: AtomicU64,
    counts: [Counts; N],
}
impl<const N: usize> HeaderInventory<N> {
    pub(crate) fn new() -> Result<Self, HeaderError> {
        if N == 0 || N > 64 {
            return Err(HeaderError::InvalidCapacity);
        }
        Ok(Self {
            free: AtomicU64::new(u64::MAX >> (64 - N)),
            cleanup: AtomicU64::new(0),
            counts: std::array::from_fn(|_| Counts::new()),
        })
    }
    /// Explicit requested storage for metadata plus N payload slots, including
    /// padding. Arc/control headers and allocator rounding are NOT covered.
    pub(crate) fn storage_layout<T>() -> Result<(Layout, usize), LayoutError> {
        let (layout, payload_offset) = Layout::new::<Self>().extend(Layout::array::<T>(N)?)?;
        Ok((layout.pad_to_align(), payload_offset))
    }
    fn resolve(&self, ticket: HeaderTicket) -> Result<&Counts, HeaderError> {
        let cell = self.counts.get(ticket.index).ok_or(HeaderError::Stale)?;
        cell.state(ticket.generation)?;
        Ok(cell)
    }
    /// Diagnostic only while a genuine holder roots this ticket. Not uniqueness,
    /// raw/native terminality, or permission to mutate/drop a payload.
    pub(crate) fn strong_count(&self, ticket: HeaderTicket) -> Result<usize, HeaderError> {
        Ok(self.resolve(ticket)?.strong.load(Ordering::Relaxed))
    }
    #[cfg(test)]
    pub(crate) fn test_strong_count(&self, ticket: HeaderTicket) -> Result<usize, HeaderError> {
        Ok(self.resolve(ticket)?.strong.load(Ordering::Acquire))
    }
    pub(crate) fn reserve(&self) -> Result<Option<HeaderReservation<'_, N>>, HeaderError> {
        loop {
            let old = match self
                .free
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
                    (bits != 0).then(|| bits & (bits - 1))
                }) {
                Ok(bits) => bits,
                Err(_) => return Ok(None),
            };
            let index = old.trailing_zeros() as usize;
            let cell = &self.counts[index];
            let previous = cell.control.load(Ordering::Acquire);
            if previous & MASK != VACANT {
                return Err(HeaderError::Phase);
            }
            let generation = previous >> SHIFT;
            let Some(next) = generation.checked_add(1).filter(|n| *n <= MAX_GENERATION) else {
                cell.transition(generation, VACANT, EXHAUSTED)?;
                // This exhausted slot stays out of the free bitmap. Remaining
                // candidates are still usable; exhaustion routes cold growth.
                continue;
            };
            cell.control
                .compare_exchange(
                    previous,
                    next << SHIFT | PREPARING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map_err(|_| HeaderError::Phase)?;
            return Ok(Some(HeaderReservation {
                inventory: self,
                ticket: HeaderTicket {
                    index,
                    generation: next,
                },
            }));
        }
    }
    fn cancel(&self, ticket: HeaderTicket) -> Result<(), HeaderError> {
        let cell = self.resolve(ticket)?;
        // No payload may remain initialized when the reservation is cancelled.
        cell.transition(ticket.generation, PREPARING, VACANT)?;
        self.free.fetch_or(1 << ticket.index, Ordering::Release);
        Ok(())
    }
    fn publish(&self, ticket: HeaderTicket) -> Result<HeaderTicket, HeaderError> {
        let cell = self.resolve(ticket)?;
        if cell.state(ticket.generation)? != PREPARING {
            return Err(HeaderError::Phase);
        }
        cell.strong.store(1, Ordering::Relaxed);
        cell.weak.store(1, Ordering::Relaxed);
        cell.transition(ticket.generation, PREPARING, LIVE)?;
        Ok(ticket)
    }
    pub(crate) fn clone_strong(&self, ticket: HeaderTicket) -> Result<(), HeaderError> {
        let cell = self.resolve(ticket)?;
        if cell.state(ticket.generation)? != LIVE {
            return Err(HeaderError::Phase);
        }
        Counts::increment(&cell.strong)
    }
    pub(crate) fn downgrade(&self, ticket: HeaderTicket) -> Result<(), HeaderError> {
        let cell = self.resolve(ticket)?;
        if cell.state(ticket.generation)? != LIVE {
            return Err(HeaderError::Phase);
        }
        Counts::increment(&cell.weak)
    }
    pub(crate) fn clone_weak(&self, ticket: HeaderTicket) -> Result<(), HeaderError> {
        let cell = self.resolve(ticket)?;
        match cell.state(ticket.generation)? {
            LIVE | DETACHING | WEAK_ONLY => Counts::increment(&cell.weak),
            _ => Err(HeaderError::Phase),
        }
    }
    pub(crate) fn upgrade(&self, ticket: HeaderTicket) -> Result<bool, HeaderError> {
        let cell = self.resolve(ticket)?;
        match cell.state(ticket.generation)? {
            DETACHING | WEAK_ONLY => return Ok(false),
            LIVE => {}
            _ => return Err(HeaderError::Phase),
        }
        match cell
            .strong
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |n| {
                (n != 0 && n < MAX_REFS).then(|| n + 1)
            }) {
            Ok(_) => Ok(true),
            Err(0) => Ok(false),
            Err(_) => Err(HeaderError::Count),
        }
    }
    pub(crate) fn release_strong(
        &self,
        ticket: HeaderTicket,
    ) -> Result<Option<HeaderDetachment>, HeaderError> {
        let cell = self.resolve(ticket)?;
        if cell.state(ticket.generation)? != LIVE {
            return Err(HeaderError::Phase);
        }
        let last = Counts::decrement(&cell.strong)?;
        if last {
            cell.transition(ticket.generation, LIVE, DETACHING)?;
        }
        Ok(last.then_some(HeaderDetachment { ticket }))
    }
    /// Publish last: all cleanup payload/physical custody writes precede this
    /// Release bit. Consuming the non-Clone authority transfers its sole custody
    /// to the independent service root. The producer keeps its root Arc through
    /// this call, then touches no slot state or payload after publication.
    pub(crate) fn publish_cleanup(&self, authority: HeaderDetachment) -> Result<(), HeaderError> {
        let cell = self.resolve(authority.ticket)?;
        if cell.state(authority.ticket.generation)? != DETACHING {
            return Err(HeaderError::Phase);
        }
        let bit = 1 << authority.ticket.index;
        if self.cleanup.fetch_or(bit, Ordering::Release) & bit != 0 {
            return Err(HeaderError::Phase);
        }
        Ok(())
    }
    /// Service only. The Acquire claim observes producer payload initialization;
    /// bit removal transfers one exclusive authority, not header credit. Native
    /// cleanup/panic custody stays rooted until this claim is finally consumed.
    #[cfg(test)]
    pub(crate) fn claim_cleanup(&self) -> Result<Option<HeaderDetachment>, HeaderError> {
        self.claim_cleanup_matching(u64::MAX)
    }
    /// Service-only bounded readiness scan. eligible excludes a cell already
    /// found nonready in this pass; unselected pending bits remain published.
    /// This mask selects work, not header credit or a native terminal verdict.
    pub(crate) fn claim_cleanup_matching(
        &self,
        eligible: u64,
    ) -> Result<Option<HeaderDetachment>, HeaderError> {
        let bits = match self
            .cleanup
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
                let candidates = bits & eligible;
                (candidates != 0).then(|| bits & !(1 << candidates.trailing_zeros()))
            }) {
            Ok(bits) => bits & eligible,
            Err(_) => return Ok(None),
        };
        let index = bits.trailing_zeros() as usize;
        let state = self.counts[index].control.load(Ordering::Acquire);
        if state & MASK != DETACHING {
            // Never restore or recycle an unexplained claim. The header's
            // implicit weak still retains storage; the service fails closed.
            return Err(HeaderError::Phase);
        }
        Ok(Some(HeaderDetachment {
            ticket: HeaderTicket {
                index,
                generation: state >> SHIFT,
            },
        }))
    }
    /// Consume transferred last-strong authority only after the independent
    /// service has destroyed all header cleanup sidecars and physical payload
    /// custody is rooted separately. Producer-to-service transfer publishes the
    /// payload and authority before its cleanup bit. Old weak never sees payload
    /// or upgrades zero; implicit weak protects metadata until this completion.
    /// Authority, like HeaderTicket, requires the inseparable correct root; its
    /// index/generation is not a cross-inventory identity discriminator.
    pub(crate) fn detached(&self, authority: HeaderDetachment) -> Result<(), HeaderError> {
        let ticket = authority.ticket;
        let cell = self.resolve(ticket)?;
        cell.transition(ticket.generation, DETACHING, WEAK_ONLY)?;
        self.release_weak(ticket)
    }
    pub(crate) fn release_weak(&self, ticket: HeaderTicket) -> Result<(), HeaderError> {
        let cell = self.resolve(ticket)?;
        if !matches!(cell.state(ticket.generation)?, LIVE | DETACHING | WEAK_ONLY) {
            return Err(HeaderError::Phase);
        }
        if Counts::decrement(&cell.weak)? {
            cell.transition(ticket.generation, WEAK_ONLY, VACANT)?;
            self.free.fetch_or(1 << ticket.index, Ordering::Release);
        }
        Ok(())
    }
    /// Service-only quiescence observation, not an acquisition capability.
    /// Concurrent false positives are harmless only under the service's shared
    /// Arc census: any producer/root caller owns an already-counted root Arc.
    /// PREPARING/DETACHING abandonment never becomes empty/free credit.
    pub(crate) fn quiescent(&self) -> bool {
        self.cleanup.load(Ordering::Acquire) == 0
            && self.counts.iter().all(|cell| {
                matches!(
                    cell.control.load(Ordering::Acquire) & MASK,
                    VACANT | EXHAUSTED
                ) && cell.strong.load(Ordering::Acquire) == 0
                    && cell.weak.load(Ordering::Acquire) == 0
            })
    }
    #[cfg(test)]
    pub(crate) fn available(&self) -> usize {
        self.free.load(Ordering::Acquire).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weak_header_does_not_block_replacement_and_never_resurrects() {
        let slots = HeaderInventory::<3>::new().unwrap();
        let old = slots.reserve().unwrap().unwrap().publish().unwrap();
        slots.clone_strong(old).unwrap();
        slots.downgrade(old).unwrap();
        assert!(slots.release_strong(old).unwrap().is_none());
        let authority = slots.release_strong(old).unwrap().unwrap();
        assert_eq!(authority.index(), old.index);
        assert!(!slots.upgrade(old).unwrap());
        assert_eq!(slots.available(), 2);
        slots.publish_cleanup(authority).unwrap();
        assert_eq!(slots.available(), 2, "queued cleanup retains header credit");
        assert!(
            !slots.upgrade(old).unwrap(),
            "queued cleanup cannot upgrade"
        );
        let service_authority = slots.claim_cleanup().unwrap().unwrap();
        assert_eq!(
            slots.available(),
            2,
            "claimed cleanup retains header credit"
        );
        assert!(
            !slots.upgrade(old).unwrap(),
            "claimed cleanup cannot upgrade"
        );
        assert!(slots.claim_cleanup().unwrap().is_none());
        slots.detached(service_authority).unwrap();
        assert!(!slots.upgrade(old).unwrap());
        slots.clone_weak(old).unwrap();
        let replacement = slots.reserve().unwrap().unwrap();
        assert_ne!(replacement.index(), old.index);
        let replacement = replacement.publish().unwrap();
        slots.release_weak(old).unwrap();
        assert_eq!(slots.available(), 1);
        slots.release_weak(old).unwrap();
        assert_eq!(slots.available(), 2);
        let next = slots.reserve().unwrap().unwrap();
        assert_eq!(next.index(), old.index);
        assert_ne!(next.ticket.generation, old.generation);
        assert_eq!(slots.clone_weak(old), Err(HeaderError::Stale));
        next.cancel().unwrap();
        let authority = slots.release_strong(replacement).unwrap().unwrap();
        slots.publish_cleanup(authority).unwrap();
        let service_authority = slots.claim_cleanup().unwrap().unwrap();
        assert!(slots.claim_cleanup().unwrap().is_none());
        slots.detached(service_authority).unwrap();
        assert_eq!(slots.available(), 3);
    }
    #[test]
    fn cold_reservation_rollback_and_generation_exhaustion_keep_other_credit() {
        let slots = HeaderInventory::<3>::new().unwrap();
        let ticket = slots.reserve().unwrap().unwrap();
        assert_eq!(slots.available(), 2);
        let stale = ticket.ticket;
        ticket.cancel().unwrap();
        assert_eq!(slots.available(), 3);
        slots.counts[0]
            .control
            .store(MAX_GENERATION << SHIFT | VACANT, Ordering::Release);
        let usable = slots.reserve().unwrap().unwrap();
        assert_eq!(usable.index(), 1);
        assert_eq!(
            slots.counts[0].control.load(Ordering::Acquire) & MASK,
            EXHAUSTED
        );
        usable.cancel().unwrap();
        assert_eq!(slots.available(), 2);
        assert_eq!(slots.clone_strong(stale), Err(HeaderError::Stale));
    }
    #[test]
    fn explicit_storage_layout_and_count_boundaries_are_checked() {
        assert!(matches!(
            HeaderInventory::<0>::new(),
            Err(HeaderError::InvalidCapacity)
        ));
        assert!(matches!(
            HeaderInventory::<65>::new(),
            Err(HeaderError::InvalidCapacity)
        ));
        let (layout, offset) = HeaderInventory::<3>::storage_layout::<u128>().unwrap();
        assert_eq!(offset % std::mem::align_of::<u128>(), 0);
        assert!(layout.size() >= offset + 3 * std::mem::size_of::<u128>());
        let slots = HeaderInventory::<3>::new().unwrap();
        let ticket = slots.reserve().unwrap().unwrap().publish().unwrap();
        slots.counts[ticket.index]
            .strong
            .store(MAX_REFS, Ordering::Relaxed);
        assert_eq!(slots.clone_strong(ticket), Err(HeaderError::Count));
        assert_eq!(slots.upgrade(ticket), Err(HeaderError::Count));
        slots.counts[ticket.index]
            .strong
            .store(1, Ordering::Relaxed);
        let authority = slots.release_strong(ticket).unwrap().unwrap();
        slots.publish_cleanup(authority).unwrap();
        let service_authority = slots.claim_cleanup().unwrap().unwrap();
        assert!(slots.claim_cleanup().unwrap().is_none());
        slots.detached(service_authority).unwrap();
        assert_eq!(slots.available(), 3);
    }
}
