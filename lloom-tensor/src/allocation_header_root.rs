//! Exact-root counted header capabilities and cold Layout admission.
//! Native physical-ready/alias closure and Buffer enrollment remain separate.
use crate::{
    allocation_header::{HeaderDetachment, HeaderError, HeaderReservation, HeaderTicket},
    allocation_header_storage::{HeaderStorage, HeaderStorageError, VacantPayload},
    deferred_failure_relay::DeferredFailureRelay,
    memory_admission::{AllocationContext, AllocationError, ModelDomain, PhysicalCharge},
    retirement_owners::{Failure, FailureSignal},
    retirement_service::{CleanupInbox, HostRetirementService, ServiceError},
};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) enum RootError {
    Storage(HeaderStorageError),
    Admission(AllocationError),
    Service(ServiceError),
    Header(HeaderError),
    Full,
    Failed(Failure),
}
pub(crate) struct RootRejected<C, A> {
    pub(crate) error: RootError,
    // None means payload is already rooted/quarantined, not destroyed/returned.
    pub(crate) owners: Option<(C, A)>,
}
struct Payload<C, A> {
    captures: C,
    anchors: A,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Staged,
    Dropping,
    CapturesDropped,
}
struct Active<C, A> {
    captures: Option<C>,
    anchors: Option<A>,
    authority: HeaderDetachment,
    phase: Phase,
}
struct HeaderRoot<C, A, const N: usize> {
    // Storage MUST precede charge for empty unpublished cold-drop refusal.
    storage: HeaderStorage<Payload<C, A>, N>,
    charge: Mutex<Option<PhysicalCharge>>,
    // Only host service accesses active; no producer method takes this mutex.
    active: Mutex<Option<Active<C, A>>>,
    failure: FailureSignal,
    // Cold installed trusted engine predicate; service only, never producer.
    ready: fn(&C, &A) -> bool,
    // Immutable cold opt-in. Independent existing failure/Wake identities only;
    // no Queue/Core/namespace/page/native backedge. Producers publish atoms,
    // actual service notifications occur outside active/registry guards.
    relay: Option<Arc<DeferredFailureRelay>>,
}
/// Opaque pool capability: its internal Arc never escapes, so safe callers
/// cannot create Weak<HeaderRoot>, raw pointers or late zero-count acquisitions.
/// Wrapper allocation by a caller does not expose the root's independent Arc.
pub(crate) struct HeaderPool<C, A, const N: usize> {
    root: Arc<HeaderRoot<C, A, N>>,
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize> HeaderPool<C, A, N> {
    /// Exact requested storage Layout only. Arc/control/registry/allocator
    /// overhead remains separately priced; this is not a total RSS quote.
    pub(crate) fn requested_bytes() -> Result<usize, RootError> {
        HeaderStorage::<Payload<C, A>, N>::requested_layout()
            .map(|layout| layout.size())
            .map_err(RootError::Storage)
    }
    /// Owned requested terms: slab Layout + audited std Arc root Layout +
    /// exact registration Box Layout + this allocation's charge-control Layout.
    /// Shared service/bootstrap/pass/domain/map and opaque allocator/native
    /// overhead are NOT hidden inside this quote. Their factory terms remain
    /// independently required; this is not an RSS quote.
    pub(crate) fn requested_owned_bytes() -> Result<usize, RootError> {
        let storage = Self::requested_bytes()?;
        let root = crate::managed_layout::arc_requested_layout::<HeaderRoot<C, A, N>>()
            .map_err(|error| RootError::Admission(error.into()))?
            .size();
        let control = crate::memory_admission::AllocationTicket::requested_charge_control_bytes()
            .map_err(RootError::Admission)?;
        storage
            .checked_add(root)
            .and_then(|bytes| {
                bytes.checked_add(HostRetirementService::requested_registration_bytes())
            })
            .and_then(|bytes| bytes.checked_add(control))
            .ok_or(RootError::Admission(AllocationError::Overflow))
    }
    /// Cold owned requested-layout enrollment, using the SAME existing promise.
    /// # Safety
    /// Same C/A/ready/relay custody as cold_when_in_with_relay (None allowed).
    /// Audited std Arc layout dependency and separately shared factory terms
    /// must be validated; requested_owned_bytes is included exactly once.
    /// Arc births retain std global allocation-error behavior; numerical
    /// admission is not graceful recovery from system OOM or an RSS guarantee.
    pub(crate) unsafe fn cold_owned_when_in(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        relay: Option<Arc<DeferredFailureRelay>>,
    ) -> Result<Self, RootError> {
        let service = HostRetirementService::shared().map_err(RootError::Service)?;
        Ok(Self {
            root: unsafe {
                HeaderRoot::cold_with_owned_context_and_service(context, ready, relay, service)?
            },
        })
    }
    /// Manual-service control of the genuine owned constructor. No root Arc,
    /// ticket or native authority is exported; quote/admission/publication and
    /// final service-charge custody are the same production bodies.
    /// # Safety
    /// Same C/A/ready/relay contract as cold_owned_when_in. The supplied service
    /// must own all registered roots through genuine cleanup; test controls use
    /// the existing manual service rather than starting the global worker.
    #[cfg(test)]
    pub(crate) unsafe fn cold_owned_when_in_service(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        relay: Option<Arc<DeferredFailureRelay>>,
        service: &Arc<HostRetirementService>,
    ) -> Result<Self, RootError> {
        Ok(Self {
            // SAFETY: same constructor contract, only service selection differs.
            root: unsafe {
                HeaderRoot::cold_with_owned_context_and_service(context, ready, relay, service)?
            },
        })
    }
    /// Partition an existing cold load promise instead of reserving it twice.
    /// # Safety
    /// Same C/A and native-use contract as cold. The existing prediction must
    /// include requested_bytes exactly once, plus separately priced overhead.
    pub(crate) unsafe fn cold_in(context: &AllocationContext) -> Result<Self, RootError> {
        // SAFETY: cold_in's caller supplies the complete native-use/C-A proof
        // before cleanup; true introduces no independent native inference.
        unsafe { Self::cold_when_in(context, |_, _| true) }
    }
    /// Partition the same cold promise with an explicit per-cell readiness gate.
    /// # Safety
    /// Same C/A, ready and native/future-submit contracts as cold_when. The
    /// prediction includes requested_bytes exactly once plus separately priced
    /// overhead. False keeps the initialized payload/anchors/implicit weak and
    /// polling enrollment intact; it does not permit Drop, refund or reuse.
    pub(crate) unsafe fn cold_when_in(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
    ) -> Result<Self, RootError> {
        let service = HostRetirementService::shared().map_err(RootError::Service)?;
        // SAFETY: the caller supplies the per-cell readiness and anchor proof;
        // existing context partition precedes storage allocation/publication.
        Ok(Self {
            root: unsafe { HeaderRoot::cold_with_context_and_service(context, ready, service)? },
        })
    }
    /// Cold engine-only failure progress enrollment. Existing constructors keep
    /// None; this does not infer native readiness or broaden any raw boundary.
    /// # Safety
    /// Same C/A/readiness contracts as cold_when_in. Relay preholds the SAME
    /// actual queue/completion FailureSignal and Wake observed by its blocked
    /// waiters, not an arbitrary pair. Caller identity/enrollment remains a
    /// separate proof; no native caller is enrolled by this constructor alone.
    /// Every relay producer must borrow through a genuine HeaderStrong/root
    /// custody; independently escaped
    /// relay Arcs are not producer capabilities. Private readiness payload may
    /// prehold the same relay cold, but never export it. Prediction separately
    /// covers relay/Arc/root metadata; requested_bytes remains storage-only.
    pub(crate) unsafe fn cold_when_in_with_relay(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        relay: Arc<DeferredFailureRelay>,
    ) -> Result<Self, RootError> {
        let service = HostRetirementService::shared().map_err(RootError::Service)?;
        // SAFETY: same native/anchor contract plus exact relay custody above.
        Ok(Self {
            root: unsafe {
                HeaderRoot::cold_with_context_relay_and_service(
                    context,
                    ready,
                    Some(relay),
                    service,
                )?
            },
        })
    }
    /// Cold admitted setup. Same trusted anchor/native-use contract as internal
    /// HeaderRoot::cold; no root Arc or root-Weak capability is returned.
    /// # Safety
    /// A contains trusted nonpanicking engine anchors without a namespace/pool/
    /// root backedge. C may panic; A retains needed physical custody throughout.
    /// Native terminal/future-submit closure is an independent caller obligation
    /// before header cleanup releases A; no raw/legacy alias proof is inferred.
    pub(crate) unsafe fn cold(domain: &ModelDomain) -> Result<Self, RootError> {
        // SAFETY: caller supplies the same A/native contract to this opaque
        // constructor; Arc remains internal and cannot gain Weak/raw holders.
        Ok(Self {
            root: unsafe { HeaderRoot::cold(domain, |_, _| true)? },
        })
    }
    /// Cold native-readiness enrollment, not a native terminality inference.
    /// # Safety
    /// The same anchor/backedge contract as cold applies. ready is trusted
    /// nonpanicking engine code: no locks, allocations, service re-entry, raw
    /// escape or arbitrary callback. It returns true ONLY after all actual
    /// future-submit aliases/borrowed native handoffs OF THIS C/A CUSTODY CELL
    /// are closed and the safely selected last-use cutoff is complete. This
    /// authorizes only these held leases/anchors, not blanket namespace closure
    /// or global residency removal. Rust header strong0 alone is not that proof;
    /// raw/legacy aliases require their independently stated boundary.
    /// False retains payload/anchors/implicit weak and polling enrollment.
    pub(crate) unsafe fn cold_when(
        domain: &ModelDomain,
        ready: fn(&C, &A) -> bool,
    ) -> Result<Self, RootError> {
        // SAFETY: caller supplies cold's anchors and this trusted readiness
        // contract; private root Arc still has no Weak/raw acquisition path.
        Ok(Self {
            root: unsafe { HeaderRoot::cold(domain, ready)? },
        })
    }
    pub(crate) fn cloned(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
    /// Claim header capacity before creating native primary/sidecar owners.
    /// Readonly atomic failure observation through this genuine root owner.
    /// Does not reserve credit or authorize concurrent recording/cleanup.
    pub(crate) fn check_failed(&self) -> Result<(), RootError> {
        match self.root.failure.get() {
            Some(reason) => Err(RootError::Failed(reason)),
            None => Ok(()),
        }
    }
    /// Cold slab growth is the caller's admitted, unlocked operation on Full.
    pub(crate) fn reserve_header(&self) -> Result<EmptyHeader<'_, C, A, N>, RootError> {
        if let Some(reason) = self.root.failure.get() {
            return Err(RootError::Failed(reason));
        }
        let claim = self
            .root
            .storage
            .inventory()
            .reserve()
            .map_err(RootError::Header)?
            .ok_or(RootError::Full)?;
        Ok(EmptyHeader {
            root: &self.root,
            preparation: Preparation {
                root: &self.root,
                claim: Some(claim),
                initialized: false,
            },
        })
    }
    /// Reserve and validate payload storage BEFORE building native sidecars.
    /// Capacity/admission refusal constructs no sidecars. An impossible stale
    /// initialized-cell observation quarantines its claim/root, never cancels
    /// an unexplained payload. Initialization of the returned
    /// exact-root token cannot return arbitrary C/A custody to its producer.
    pub(crate) fn reserve_validated_header(
        &self,
    ) -> Result<ValidatedEmptyHeader<'_, C, A, N>, RootError> {
        self.reserve_header()?.validate()
    }
    pub(crate) fn insert(
        &self,
        captures: C,
        anchors: A,
    ) -> Result<HeaderStrong<C, A, N>, RootRejected<C, A>> {
        self.root.insert(captures, anchors)
    }
}
/// Exact-root, unpublished credit. Drop before initialize cancels only atoms.
/// No payload/reference/native capability exists at this stage.
pub(crate) struct EmptyHeader<
    'a,
    C: Send + Sync + 'static,
    A: Send + Sync + 'static,
    const N: usize,
> {
    root: &'a Arc<HeaderRoot<C, A, N>>,
    preparation: Preparation<'a, C, A, N>,
}
/// Exact-root empty credit plus exclusive validated vacant payload cell.
/// Private fields inseparably bind the storage token to its live PREPARING
/// reservation. No C/A has been constructed by this type or its validation.
pub(crate) struct ValidatedEmptyHeader<
    'a,
    C: Send + Sync + 'static,
    A: Send + Sync + 'static,
    const N: usize,
> {
    empty: EmptyHeader<'a, C, A, N>,
    vacant: VacantPayload<'a, Payload<C, A>>,
}
impl<'a, C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize>
    ValidatedEmptyHeader<'a, C, A, N>
{
    /// Move owners directly into the already-validated service-rooted cell.
    /// No callback/lock/allocation/native action, fallible path, or arbitrary
    /// Drop occurs before initialized publication. Subsequent unwind uses the
    /// existing initialized Preparation cleanup publication, not rollback.
    pub(crate) fn initialize(self, captures: C, anchors: A) -> PreparedHeader<'a, C, A, N> {
        let Self { mut empty, vacant } = self;
        // SAFETY: private fields were constructed together by validate; the
        // genuine same-root PREPARING reservation stays in empty through write.
        unsafe {
            vacant.initialize(Payload { captures, anchors });
        }
        empty.preparation.initialized = true;
        PreparedHeader {
            root: empty.root,
            preparation: empty.preparation,
        }
    }
}
/// Initialized but unpublished. Drop queues rooted service cleanup, NOT credit
/// rollback; native C/A cannot be silently destroyed on the producer thread.
pub(crate) struct PreparedHeader<
    'a,
    C: Send + Sync + 'static,
    A: Send + Sync + 'static,
    const N: usize,
> {
    root: &'a Arc<HeaderRoot<C, A, N>>,
    preparation: Preparation<'a, C, A, N>,
}
impl<'a, C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize>
    EmptyHeader<'a, C, A, N>
{
    fn validate(mut self) -> Result<ValidatedEmptyHeader<'a, C, A, N>, RootError> {
        let index = self
            .preparation
            .claim
            .as_ref()
            .expect("exclusive unpublished credit")
            .index();
        // SAFETY: exact root borrowed for 'a, sole genuine PREPARING reservation
        // stays inside self; no payload owner/accessor exists for this credit.
        let vacant = match unsafe { self.root.storage.vacant(index) } {
            Ok(vacant) => vacant,
            Err(error) => {
                // No new C/A exists. Do not return unexplained initialized
                // storage to free credit: an unfinished reservation has no
                // implicit cancellation; failed root remains service-retained.
                self.root.failed();
                let _quarantined = self.preparation.claim.take();
                return Err(RootError::Storage(error));
            }
        };
        Ok(ValidatedEmptyHeader {
            empty: self,
            vacant,
        })
    }
    pub(crate) fn initialize(
        mut self,
        captures: C,
        anchors: A,
    ) -> Result<PreparedHeader<'a, C, A, N>, RootRejected<C, A>> {
        let index = self
            .preparation
            .claim
            .as_ref()
            .expect("exclusive unpublished credit")
            .index();
        // SAFETY: exact-root sole PREPARING reservation, root borrow spans it.
        if let Err((error, payload)) = unsafe {
            self.root
                .storage
                .initialize(index, Payload { captures, anchors })
        } {
            return Err(RootRejected {
                error: RootError::Storage(error),
                owners: Some((payload.captures, payload.anchors)),
            });
        }
        self.preparation.initialized = true;
        Ok(PreparedHeader {
            root: self.root,
            preparation: self.preparation,
        })
    }
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize>
    PreparedHeader<'_, C, A, N>
{
    /// Engine transaction hook only; no arbitrary user callback is installed.
    /// # Safety
    /// The caller uses A only for exact-root atomic publication preparation;
    /// no native/future-submit capability may escape before publish. Reference
    /// cannot outlive this borrowed unpublished header or overlap publication.
    /// Physical transaction A remains rooted here across any unwind.
    pub(crate) unsafe fn anchors(&self) -> &A {
        let index = self
            .preparation
            .claim
            .as_ref()
            .expect("unpublished credit")
            .index();
        // SAFETY: this initialized exact-root sole PREPARING reservation spans
        // the returned borrow; storage payload remains immutable until publish.
        &unsafe { self.root.storage.payload(index) }
            .expect("initialized unpublished payload")
            .anchors
    }
    /// Publish last after actual physical transaction commit. Contains only
    /// checked metadata transition and Arc increment, no allocation/callback.
    /// Unexpected phase failure retains initialized storage/anchors and marks
    /// root failed, rather than returning FREE or running arbitrary C/A Drop.
    pub(crate) fn publish(mut self) -> Result<HeaderStrong<C, A, N>, RootError> {
        let ticket = match self
            .preparation
            .claim
            .take()
            .expect("unpublished credit")
            .publish()
        {
            Ok(ticket) => ticket,
            Err(error) => {
                self.root.failed();
                return Err(RootError::Header(error));
            }
        };
        Ok(HeaderStrong {
            root: self.root.clone(),
            ticket,
        })
    }
}
pub(crate) struct HeaderStrong<C, A, const N: usize> {
    root: Arc<HeaderRoot<C, A, N>>,
    ticket: HeaderTicket,
}
pub(crate) struct HeaderWeak<C, A, const N: usize> {
    root: Arc<HeaderRoot<C, A, N>>,
    ticket: HeaderTicket,
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize> HeaderRoot<C, A, N> {
    /// Cold setup only. No allocation caller may treat this as a machine/RSS cap.
    /// # Safety
    /// A contains trusted nonpanicking engine anchors and no root/namespace/pool
    /// backedge. C destruction may panic; A retains all needed physical custody
    /// across it. Native payloads require independent terminal/future-use proof
    /// before header cleanup may release A. Raw/legacy escapes are NOT proved by
    /// this header strong/weak census. Actual Buffer enrollment is not supplied.
    /// ready additionally upholds cold_when's trusted, per-C/A-cell readiness
    /// contract (or the caller has already supplied that proof for true).
    unsafe fn cold(
        domain: &ModelDomain,
        ready: fn(&C, &A) -> bool,
    ) -> Result<Arc<Self>, RootError> {
        let service = HostRetirementService::shared().map_err(RootError::Service)?;
        // SAFETY: cold caller supplies anchors and the trusted ready contract;
        // this shared service is already independently rooted before enrollment.
        unsafe { Self::cold_with_service(domain, ready, service) }
    }
    /// Same cold/ready contracts; private service seam lets bounded scalar
    /// controls use the actual admission/enrollment path without a worker thread.
    unsafe fn cold_with_service(
        domain: &ModelDomain,
        ready: fn(&C, &A) -> bool,
        service: &Arc<HostRetirementService>,
    ) -> Result<Arc<Self>, RootError> {
        let bytes = HeaderStorage::<Payload<C, A>, N>::requested_layout()
            .map_err(RootError::Storage)?
            .size();
        let pass = domain.reserve(bytes).map_err(RootError::Admission)?;
        // SAFETY: same trusted contract and rooted service; partition the new
        // pass through the same constructor used by an existing load promise.
        unsafe { Self::cold_with_context_and_service(&pass.context(), ready, service) }
    }
    /// Cold partition, never an implicit extra domain reservation. The context
    /// split validates cancellation and available bytes before storage allocation.
    /// # Safety
    /// Same C/A, ready and service-root contract as cold_with_service.
    unsafe fn cold_with_context_and_service(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        service: &Arc<HostRetirementService>,
    ) -> Result<Arc<Self>, RootError> {
        // SAFETY: preserve all existing constructors with no enrolled relay.
        unsafe { Self::cold_with_context_relay_and_service(context, ready, None, service) }
    }
    /// Cold optional relay setup; same storage/C-A and inseparable producer-root
    /// contract as cold_when_in_with_relay. Shared service registered last.
    unsafe fn cold_with_context_relay_and_service(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        relay: Option<Arc<DeferredFailureRelay>>,
        service: &Arc<HostRetirementService>,
    ) -> Result<Arc<Self>, RootError> {
        let bytes = HeaderPool::<C, A, N>::requested_bytes()?;
        let mut ticket = context.take(bytes).map_err(RootError::Admission)?;
        let storage = HeaderStorage::allocate().map_err(RootError::Storage)?;
        let charge = match ticket.try_commit(bytes) {
            Ok(charge) => charge,
            Err(error) => {
                drop(storage); // Physical block free precedes ticket cancellation.
                return Err(RootError::Admission(error));
            }
        };
        let root = Arc::new(Self {
            storage,
            charge: Mutex::new(Some(charge)),
            active: Mutex::new(None),
            failure: FailureSignal::default(),
            ready,
            relay,
        });
        // Cold Arc/control/registry/allocator rounding are still unenrolled.
        let inbox: Arc<dyn CleanupInbox> = root.clone();
        service.register(&inbox).map_err(RootError::Service)?;
        Ok(root)
    }
    /// Same owned cold contracts; caller service is already rooted. All four
    /// owned terms are promised before storage/root/charge/registration allocation.
    unsafe fn cold_with_owned_context_and_service(
        context: &AllocationContext,
        ready: fn(&C, &A) -> bool,
        relay: Option<Arc<DeferredFailureRelay>>,
        service: &Arc<HostRetirementService>,
    ) -> Result<Arc<Self>, RootError> {
        let quote = HeaderPool::<C, A, N>::requested_owned_bytes()?;
        let control = crate::memory_admission::AllocationTicket::requested_charge_control_bytes()
            .map_err(RootError::Admission)?;
        let payload = quote
            .checked_sub(control)
            .ok_or(RootError::Admission(AllocationError::Overflow))?;
        let mut ticket = context.take(quote).map_err(RootError::Admission)?;
        let storage = HeaderStorage::allocate().map_err(RootError::Storage)?;
        let charge = match ticket.try_commit_with_control(payload) {
            Ok(charge) => charge,
            Err(error) => {
                drop(storage); // owned Layout free precedes ticket cancellation
                return Err(RootError::Admission(error));
            }
        };
        let root = Arc::new(Self {
            storage,
            charge: Mutex::new(Some(charge)),
            active: Mutex::new(None),
            failure: FailureSignal::default(),
            ready,
            relay,
        });
        let inbox: Arc<dyn CleanupInbox> = root.clone();
        match service.register_owned(&inbox) {
            Ok(()) => {
                drop(inbox);
                Ok(root)
            }
            Err(error) => {
                drop(inbox);
                // Failed registration retained no foreign root; its candidate
                // Box was freed before returning. into_inner frees root Arc
                // storage BEFORE returning inline root for storage/charge Drop.
                drop(Arc::into_inner(root).expect("unpublished private root"));
                Err(RootError::Service(error))
            }
        }
    }
    fn insert(
        self: &Arc<Self>,
        captures: C,
        anchors: A,
    ) -> Result<HeaderStrong<C, A, N>, RootRejected<C, A>> {
        if let Some(reason) = self.failure.get() {
            return Err(RootRejected {
                error: RootError::Failed(reason),
                owners: Some((captures, anchors)),
            });
        }
        let claim = match self.storage.inventory().reserve() {
            Ok(Some(claim)) => claim,
            Ok(None) => {
                return Err(RootRejected {
                    error: RootError::Full,
                    owners: Some((captures, anchors)),
                })
            }
            Err(error) => {
                return Err(RootRejected {
                    error: RootError::Header(error),
                    owners: Some((captures, anchors)),
                })
            }
        };
        let index = claim.index();
        let mut preparation = Preparation {
            root: self,
            claim: Some(claim),
            initialized: false,
        };
        // SAFETY: exact-root sole PREPARING reservation, root Arc covers call.
        if let Err((error, payload)) = unsafe {
            self.storage
                .initialize(index, Payload { captures, anchors })
        } {
            return Err(RootRejected {
                error: RootError::Storage(error),
                owners: Some((payload.captures, payload.anchors)),
            });
        }
        preparation.initialized = true;
        let ticket = match preparation
            .claim
            .take()
            .expect("owned preparation")
            .publish()
        {
            Ok(ticket) => ticket,
            Err(error) => {
                // Impossible contract violation retains initialized storage;
                // no unsafe take from an unexplained phase or arbitrary Drop.
                self.failed();
                return Err(RootRejected {
                    error: RootError::Header(error),
                    owners: None,
                });
            }
        };
        Ok(HeaderStrong {
            root: self.clone(),
            ticket,
        })
    }
    fn failed(&self) {
        self.publish_failure(Failure::AdapterFailed);
    }
}
impl<C, A, const N: usize> HeaderRoot<C, A, N> {
    fn publish_failure(&self, reason: Failure) {
        self.failure.publish(reason, || {});
        if let Some(relay) = &self.relay {
            relay.publish(reason);
        }
    }
}
struct Preparation<'a, C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize> {
    root: &'a HeaderRoot<C, A, N>,
    claim: Option<HeaderReservation<'a, N>>,
    initialized: bool,
}
impl<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize> Drop
    for Preparation<'_, C, A, N>
{
    fn drop(&mut self) {
        let Some(claim) = self.claim.take() else {
            return;
        };
        if !self.initialized {
            if claim.cancel().is_err() {
                self.root.failed();
            }
            return;
        }
        // Unwind cleanup never drops C/A or takes a lock on producer. Convert
        // its sole initialized preparation into queued host cleanup custody.
        let result = claim.publish().and_then(|ticket| {
            self.root
                .storage
                .inventory()
                .release_strong(ticket)
                .and_then(|authority| authority.ok_or(HeaderError::Phase))
                .and_then(|authority| self.root.storage.inventory().publish_cleanup(authority))
        });
        if result.is_err() {
            self.root.failed();
        }
    }
}
impl<C, A, const N: usize> HeaderStrong<C, A, N> {
    /// Same fixed-data atomic failure check as the owning pool; no message read.
    pub(crate) fn check_failed(&self) -> Result<(), RootError> {
        match self.root.failure.get() {
            Some(reason) => Err(RootError::Failed(reason)),
            None => Ok(()),
        }
    }

    /// Counted payload identity, not native allocation/namespace equivalence.
    /// Root identity remains opaque; no pointer or acquisition path escapes.
    pub(crate) fn same(left: &Self, right: &Self) -> bool {
        Arc::ptr_eq(&left.root, &right.root) && left.ticket == right.ticket
    }
    /// Atoms-only refusal publication under this genuine counted root holder.
    /// No wake/native call, callback, allocation or detached relay export.
    /// Failure retains custody; it is never release/terminality authorization.
    pub(crate) fn publish_failure(&self, reason: Failure) {
        self.root.publish_failure(reason);
    }
    /// Approximate diagnostic count; this holder prevents detachment/recycling.
    /// Never a substitute for future-use or native terminal custody.
    pub(crate) fn strong_count(&self) -> usize {
        self.root
            .storage
            .inventory()
            .strong_count(self.ticket)
            .expect("genuine strong header ticket")
    }
    pub(crate) fn captures(&self) -> &C {
        // SAFETY: private ticket inseparable from correct counted root; this
        // strong holder/root outlives returned reference and blocks detachment.
        &unsafe { self.root.storage.payload(self.ticket.index()) }
            .expect("counted strong initialized payload")
            .captures
    }
    pub(crate) fn try_clone(&self) -> Result<Self, HeaderError> {
        self.root.storage.inventory().clone_strong(self.ticket)?;
        Ok(Self {
            root: self.root.clone(),
            ticket: self.ticket,
        })
    }
    /// Checked two-count acquisition used by the concrete physical External
    /// role. First obtain a genuine header retain; failed alias CAS drops that
    /// temporary retain, leaving the original and both counts unchanged.
    /// # Safety
    /// selector is an atoms-only immutable projection to the external census
    /// INSIDE this exact C payload, never a counter from another root. Caller
    /// retains this genuine issuer/source throughout; on success it constructs
    /// one external owner nonfallibly and releases the alias before header Drop.
    pub(crate) unsafe fn try_clone_with_external(
        &self,
        selector: fn(&C) -> &crate::allocation_alias::AliasPins<()>,
    ) -> Result<Self, HeaderError> {
        let retained = self.try_clone()?;
        if !selector(self.captures()).try_clone_external() {
            return Err(HeaderError::Count);
        }
        Ok(retained)
    }
    pub(crate) fn downgrade(&self) -> Result<HeaderWeak<C, A, N>, HeaderError> {
        self.root.storage.inventory().downgrade(self.ticket)?;
        Ok(HeaderWeak {
            root: self.root.clone(),
            ticket: self.ticket,
        })
    }
}
impl<C, A, const N: usize> HeaderWeak<C, A, N> {
    /// Exact root+generation identity without upgrading or exposing root Arc.
    pub(crate) fn same(left: &Self, right: &Self) -> bool {
        Arc::ptr_eq(&left.root, &right.root) && left.ticket == right.ticket
    }
    /// Mixed counted identity, without upgrade/downgrade or root exposure.
    /// This is not native allocation identity or a terminality permission.
    pub(crate) fn same_strong(&self, strong: &HeaderStrong<C, A, N>) -> bool {
        Arc::ptr_eq(&self.root, &strong.root) && self.ticket == strong.ticket
    }
    pub(crate) fn try_clone(&self) -> Result<Self, HeaderError> {
        self.root.storage.inventory().clone_weak(self.ticket)?;
        Ok(Self {
            root: self.root.clone(),
            ticket: self.ticket,
        })
    }
    pub(crate) fn upgrade(&self) -> Result<Option<HeaderStrong<C, A, N>>, HeaderError> {
        Ok(self
            .root
            .storage
            .inventory()
            .upgrade(self.ticket)?
            .then(|| HeaderStrong {
                root: self.root.clone(),
                ticket: self.ticket,
            }))
    }
}
impl<C, A, const N: usize> Drop for HeaderStrong<C, A, N> {
    fn drop(&mut self) {
        let result = self
            .root
            .storage
            .inventory()
            .release_strong(self.ticket)
            .and_then(|authority| match authority {
                Some(authority) => self.root.storage.inventory().publish_cleanup(authority),
                None => Ok(()),
            });
        if result.is_err() {
            self.root.publish_failure(Failure::AdapterFailed);
        }
        // Automatic Arc release follows final slot access; service root remains.
    }
}
impl<C, A, const N: usize> Drop for HeaderWeak<C, A, N> {
    fn drop(&mut self) {
        if self
            .root
            .storage
            .inventory()
            .release_weak(self.ticket)
            .is_err()
        {
            self.root.publish_failure(Failure::AdapterFailed);
        }
    }
}
// SAFETY: single host worker stages C,A and sole cleanup authority inside this
// independently registered root before arbitrary C Drop. No native proof is
// inferred here: unsafe cold constructor requires caller's anchor/terminal
// contract. Warm publishers are atoms only and retain root Arc through access.
unsafe impl<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize> CleanupInbox
    for HeaderRoot<C, A, N>
{
    fn begin_retire(&self) -> bool {
        let mut active = self.active.lock();
        assert!(active.is_none(), "header cleanup cannot retry active Drop");
        let mut eligible = u64::MAX;
        loop {
            let Some(authority) = self
                .storage
                .inventory()
                .claim_cleanup_matching(eligible)
                .expect("exclusive header cleanup authority")
            else {
                return false;
            };
            let index = authority.index();
            // SAFETY: exact-root sole claimed DETACHING authority; no strong
            // borrower can survive/resurrect. Root/implicit weak span this
            // scoped read; reference ends before either take or requeue.
            let ready = {
                let payload = unsafe { self.storage.payload(index) }
                    .expect("claimed initialized header payload");
                (self.ready)(&payload.captures, &payload.anchors)
            };
            if !ready {
                // One attempt per slot/pass, no spin on an outstanding alias.
                // Requeue the same consumed authority; no free/header credit,
                // arbitrary Drop, epoch rewrite or new producer operation.
                eligible &= !(1 << index);
                self.storage
                    .inventory()
                    .publish_cleanup(authority)
                    .expect("nonready authority stays DETACHING");
                continue;
            }
            // SAFETY: same-root sole service-claimed DETACHING authority; ready
            // additionally supplies caller's native/alias/cutoff contract. Move
            // into rooted active custody before any arbitrary captured Drop.
            let payload =
                unsafe { self.storage.take(index) }.expect("claimed header cleanup payload");
            *active = Some(Active {
                captures: Some(payload.captures),
                anchors: Some(payload.anchors),
                authority,
                phase: Phase::Staged,
            });
            return true;
        }
    }
    fn drop_captures(&self) {
        let captures = {
            let mut active = self.active.lock();
            let active = active.as_mut().expect("active header cleanup");
            assert!(active.phase == Phase::Staged, "header cleanup phase");
            active.phase = Phase::Dropping;
            active.captures.take()
        };
        drop(captures); // Unwind retains rooted A/authority; never retry.
        self.active
            .lock()
            .as_mut()
            .expect("active header cleanup")
            .phase = Phase::CapturesDropped;
    }
    fn finish_retire(&self) {
        let anchors = {
            let mut active = self.active.lock();
            let active = active.as_mut().expect("active header cleanup");
            assert!(
                active.phase == Phase::CapturesDropped,
                "header cleanup phase"
            );
            active.anchors.take()
        };
        drop(anchors); // Trusted nonpanicking engine-only anchors, offguard.
        let active = self.active.lock().take().expect("active header cleanup");
        self.storage
            .inventory()
            .detached(active.authority)
            .expect("finished header cleanup");
    }
    fn can_unregister(&self) -> bool {
        self.failure.get().is_none()
            && self.active.lock().is_none()
            && self.storage.inventory().quiescent()
            && self.relay.as_ref().is_none_or(|relay| relay.quiet())
    }
    fn service_notify(&self) {
        // Actual service invokes after begin_retire/fail returns and outside
        // active and registry guards, including while genuine strongs live.
        if let Some(relay) = &self.relay {
            relay.notify_on_service();
        }
    }
    fn requires_polling(&self) -> bool {
        true
    }
    fn worker_final_metadata_drop(&self) -> bool {
        true
    }
    fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
        self.charge.lock().take()
    }
    fn fail(&self, reason: Failure) {
        self.publish_failure(reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{process_account, AccountSnapshot};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn scalar_pool<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize>(
        domain: &ModelDomain,
        service: &Arc<HostRetirementService>,
        ready: fn(&C, &A) -> bool,
    ) -> HeaderPool<C, A, N> {
        // SAFETY: every caller below uses scalar anchors/probes only, no native
        // payload or root/namespace/pool backedge. Predicates are atoms only.
        HeaderPool {
            root: unsafe { HeaderRoot::cold_with_service(domain, ready, service) }
                .expect("bounded scalar admitted root"),
        }
    }
    fn inserted<C: Send + Sync + 'static, A: Send + Sync + 'static, const N: usize>(
        pool: &HeaderPool<C, A, N>,
        captures: C,
        anchors: A,
    ) -> HeaderStrong<C, A, N> {
        match pool.insert(captures, anchors) {
            Ok(handle) => handle,
            Err(rejected) => panic!("unexpected scalar insert refusal: {:?}", rejected.error),
        }
    }
    /// Where std's Arc layout is not audited (`build.rs`), the owned quote
    /// refuses by policy, and the owned constructor refuses with it before it
    /// draws on the promise or allocates.
    #[cfg(not(lloom_audited_arc_layout))]
    #[test]
    fn owned_root_actual_free_precedes_same_account_refund() {
        type Pool = HeaderPool<(), (), 3>;
        assert!(matches!(
            Pool::requested_owned_bytes(),
            Err(RootError::Admission(AllocationError::UnsupportedLayout))
        ));
        let service = HostRetirementService::manual();
        let domain = ModelDomain::isolated_numeric_fixture(4096);
        let pass = domain.reserve(4096).unwrap();
        let context = pass.context();
        // SAFETY: scalar empty C/A only, no native use/backedge, true readiness;
        // the constructor refuses before any root or registration exists.
        let result =
            unsafe { Pool::cold_owned_when_in_service(&context, |_, _| true, None, &service) };
        assert!(matches!(
            result,
            Err(RootError::Admission(AllocationError::UnsupportedLayout))
        ));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 4096,
                committed: 0
            },
            "the refusal draws nothing on the promise"
        );
        drop(context);
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[cfg(lloom_audited_arc_layout)]
    #[test]
    fn owned_root_actual_free_precedes_same_account_refund() {
        type Pool = HeaderPool<(), (), 3>;
        let service = HostRetirementService::manual();
        let quote = Pool::requested_owned_bytes().unwrap();
        let storage = Pool::requested_bytes().unwrap();
        let control =
            crate::memory_admission::AllocationTicket::requested_charge_control_bytes().unwrap();
        let root_layout =
            crate::managed_layout::arc_requested_layout::<HeaderRoot<(), (), 3>>().unwrap();
        let registration = HostRetirementService::requested_registration_bytes();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let context = pass.context();
        // Arm only around the actual four owned births. Domain/pass/service and
        // their independently disclosed bootstrap allocations already exist.
        let report = crate::free_observer::observe(&domain, |observer| {
            // SAFETY: scalar empty C/A only, no native use/backedge, true readiness;
            // real manual service owns the actual priced registration/root cleanup.
            let pool =
                unsafe { Pool::cold_owned_when_in_service(&context, |_, _| true, None, &service) }
                    .unwrap();
            observer.watch();
            assert_eq!(
                domain.snapshot(),
                AccountSnapshot {
                    pending: 0,
                    committed: quote
                }
            );
            drop(context);
            drop(pass); // unrelated allocations, not tracked targets
            drop(pool); // service still owns root and actual four committed terms
            assert_eq!(domain.snapshot().committed, quote);
            service.manual_pass(); // same production unlink/free/final-charge path
        }); // internal Reset disarms before assertion formatting
        assert_eq!(
            report.born, 4,
            "owned constructor requested birth census drift"
        );
        assert!(
            !report.overflow
                && !report.reallocated
                && !report.wrong_layout
                && !report.duplicate_free
        );
        assert_eq!(
            report.blocks[..4]
                .iter()
                .map(|b| b.size)
                .collect::<Vec<_>>(),
            vec![storage, control, root_layout.size(), registration]
        );
        assert_eq!(report.blocks[2].align, root_layout.align());
        assert_eq!(
            report.freed, 4,
            "every promised owned block must actually free"
        );
        for observation in &report.snapshots[..4] {
            assert_eq!(
                *observation,
                Some(AccountSnapshot {
                    pending: 0,
                    committed: quote
                }),
                "same committed charge must survive actual System free"
            );
        }
        assert_eq!(
            &report.order[..4],
            &[3, 0, 2, 1],
            "node then slab/root then charge-control free"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot::default(),
            "refund only after all four real frees"
        );
    }

    #[test]
    fn weak_rooted_header_does_not_resurrect_replacement_or_lose_charge() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<u64, (), 3>(&domain, &service, |_, _| true);
        let bytes = HeaderStorage::<Payload<u64, ()>, 3>::requested_layout()
            .unwrap()
            .size();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: bytes
            }
        );
        let old = inserted(&pool, 17, ());
        let old_clone = old.try_clone().unwrap();
        let old_weak = old.downgrade().unwrap();
        let weak_clone = old_weak.try_clone().unwrap();
        assert!(HeaderStrong::same(&old, &old_clone));
        assert!(HeaderWeak::same(&old_weak, &weak_clone));
        assert!(old_weak.same_strong(&old));
        assert!(weak_clone.same_strong(&old_clone));
        assert_eq!(*old_clone.captures(), 17);
        drop(old);
        assert_eq!(*old_weak.upgrade().unwrap().unwrap().captures(), 17);
        drop(old_clone);
        assert!(old_weak.upgrade().unwrap().is_none());
        assert_eq!(pool.root.storage.inventory().available(), 2);
        assert!(service.manual_pass());
        assert!(old_weak.upgrade().unwrap().is_none());
        assert_eq!(pool.root.storage.inventory().available(), 2);
        let replacement = inserted(&pool, 23, ());
        assert_ne!(replacement.ticket.index(), old_weak.ticket.index());
        let replacement_weak = replacement.downgrade().unwrap();
        assert!(!HeaderWeak::same(&old_weak, &replacement_weak));
        assert!(!old_weak.same_strong(&replacement));
        assert!(replacement_weak.same_strong(&replacement));
        drop(replacement_weak);
        assert_eq!(*replacement.captures(), 23);
        assert!(old_weak.upgrade().unwrap().is_none());
        drop(replacement);
        assert!(service.manual_pass());
        drop(pool);
        assert!(!service.manual_pass());
        assert_eq!(
            domain.snapshot().committed,
            bytes,
            "old header weak retains slab charge"
        );
        drop(old_weak);
        assert!(!service.manual_pass());
        assert_eq!(domain.snapshot().committed, bytes);
        drop(weak_clone);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn checked_external_refusal_restores_real_header_and_original_custody() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<crate::allocation_alias::AliasPins<()>, (), 3>(
            &domain,
            &service,
            |c, _| !c.has_external(),
        );
        let pins = crate::allocation_alias::AliasPins::new();
        assert!(pins.claim_primary());
        let original = inserted(&pool, pins, ());
        let weak = original.downgrade().unwrap();
        let roots = Arc::strong_count(&original.root);
        let strongs = original
            .root
            .storage
            .inventory()
            .test_strong_count(original.ticket)
            .unwrap();
        original.captures().test_external_count(isize::MAX as usize);
        // SAFETY: SAME actual scalar C, immutable atoms-only identity selector.
        assert!(matches!(
            unsafe { original.try_clone_with_external(|c| c) },
            Err(HeaderError::Count)
        ));
        assert_eq!(
            original
                .root
                .storage
                .inventory()
                .test_strong_count(original.ticket)
                .unwrap(),
            strongs
        );
        assert_eq!(Arc::strong_count(&original.root), roots);
        assert_eq!(
            original.captures().test_current_external_count(),
            isize::MAX as usize
        );
        let upgraded = weak.upgrade().unwrap().unwrap();
        assert!(HeaderStrong::same(&original, &upgraded));
        drop(upgraded);
        assert_eq!(pool.root.storage.inventory().available(), 2);
        // Synthetic numeric-bound injection has no actual phantom owners.
        original.captures().test_external_count(0);
        let counted = unsafe { original.try_clone_with_external(|c| c) }.unwrap();
        assert_eq!(original.captures().test_current_external_count(), 1);
        assert!(counted.captures().release_external().is_none());
        drop(counted);
        assert_eq!(
            original
                .root
                .storage
                .inventory()
                .test_strong_count(original.ticket)
                .unwrap(),
            strongs
        );
        assert_eq!(original.captures().test_current_external_count(), 0);
        drop(weak);
        drop(original);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    struct Captures {
        bit: usize,
        ready: Arc<AtomicBool>,
        attempts: Arc<AtomicUsize>,
        dropped: Arc<AtomicUsize>,
    }
    impl Drop for Captures {
        fn drop(&mut self) {
            self.dropped.fetch_or(self.bit, Ordering::SeqCst);
        }
    }
    struct Anchors {
        bit: usize,
        captures_dropped: Arc<AtomicUsize>,
        dropped: Arc<AtomicUsize>,
        domain: ModelDomain,
        bytes: usize,
    }
    impl Drop for Anchors {
        fn drop(&mut self) {
            assert_ne!(
                self.captures_dropped.load(Ordering::SeqCst) & self.bit,
                0,
                "actual anchor Drop follows captured Drop"
            );
            assert_eq!(
                self.domain.snapshot().committed,
                self.bytes,
                "actual anchor Drop retains slab charge"
            );
            self.dropped.fetch_or(self.bit, Ordering::SeqCst);
        }
    }
    fn readiness(c: &Captures, _: &Anchors) -> bool {
        c.attempts.fetch_add(1, Ordering::Relaxed);
        c.ready.load(Ordering::Acquire)
    }
    fn owners(
        bit: usize,
        ready: Arc<AtomicBool>,
        attempts: Arc<AtomicUsize>,
        captures_dropped: &Arc<AtomicUsize>,
        anchors_dropped: &Arc<AtomicUsize>,
        domain: &ModelDomain,
        bytes: usize,
    ) -> (Captures, Anchors) {
        (
            Captures {
                bit,
                ready,
                attempts,
                dropped: captures_dropped.clone(),
            },
            Anchors {
                bit,
                captures_dropped: captures_dropped.clone(),
                dropped: anchors_dropped.clone(),
                domain: domain.clone(),
                bytes,
            },
        )
    }
    #[test]
    fn nonready_cleanup_keeps_authority_credit_and_charge_without_blocking_sibling() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<Captures, Anchors, 3>(&domain, &service, readiness);
        let bytes = HeaderStorage::<Payload<Captures, Anchors>, 3>::requested_layout()
            .unwrap()
            .size();
        let captures_dropped = Arc::new(AtomicUsize::new(0));
        let anchors_dropped = Arc::new(AtomicUsize::new(0));
        let first_ready = Arc::new(AtomicBool::new(false));
        let first_attempts = Arc::new(AtomicUsize::new(0));
        let second_attempts = Arc::new(AtomicUsize::new(0));
        let (c, a) = owners(
            1,
            first_ready.clone(),
            first_attempts.clone(),
            &captures_dropped,
            &anchors_dropped,
            &domain,
            bytes,
        );
        let first = inserted(&pool, c, a);
        let weak = first.downgrade().unwrap();
        let (c, a) = owners(
            2,
            Arc::new(AtomicBool::new(true)),
            second_attempts.clone(),
            &captures_dropped,
            &anchors_dropped,
            &domain,
            bytes,
        );
        let second = inserted(&pool, c, a);
        drop(first);
        drop(second);
        assert_eq!(pool.root.storage.inventory().available(), 1);
        assert!(service.manual_pass());
        assert_eq!(
            first_attempts.load(Ordering::Relaxed),
            1,
            "one readiness attempt per slot/pass"
        );
        assert_eq!(second_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            captures_dropped.load(Ordering::SeqCst),
            2,
            "nonready payload stays initialized"
        );
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 2);
        assert_eq!(
            pool.root.storage.inventory().available(),
            2,
            "nonready header credit is not returned"
        );
        assert!(weak.upgrade().unwrap().is_none());
        assert!(!service.manual_pass());
        assert_eq!(first_attempts.load(Ordering::Relaxed), 2);
        first_ready.store(true, Ordering::Release);
        assert!(service.manual_pass());
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 3);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 3);
        assert_eq!(
            pool.root.storage.inventory().available(),
            2,
            "old weak pins only its header"
        );
        drop(weak);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn cold_budget_refusal_leaves_no_pending_or_committed_slab() {
        let service = HostRetirementService::manual();
        let bytes = HeaderStorage::<Payload<u64, ()>, 3>::requested_layout()
            .unwrap()
            .size();
        let domain = process_account()
            .unadmitted_domain(Some(bytes - 1))
            .unwrap();
        // SAFETY: scalar u64/unit, no native use/backedges, atoms-free true predicate.
        let refused =
            unsafe { HeaderRoot::<u64, (), 3>::cold_with_service(&domain, |_, _| true, &service) };
        assert!(matches!(refused, Err(RootError::Admission(_))));
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
        assert!(!service.manual_pass());
    }
    #[test]
    fn initialized_preparation_unwind_queues_custody_without_producer_drop() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<Captures, Anchors, 3>(&domain, &service, readiness);
        let bytes = HeaderStorage::<Payload<Captures, Anchors>, 3>::requested_layout()
            .unwrap()
            .size();
        let captures_dropped = Arc::new(AtomicUsize::new(0));
        let anchors_dropped = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(AtomicBool::new(true));
        let attempts = Arc::new(AtomicUsize::new(0));
        let (c, a) = owners(
            1,
            ready,
            attempts,
            &captures_dropped,
            &anchors_dropped,
            &domain,
            bytes,
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let claim = pool.root.storage.inventory().reserve().unwrap().unwrap();
            let index = claim.index();
            let mut preparation = Preparation {
                root: &pool.root,
                claim: Some(claim),
                initialized: false,
            };
            // SAFETY: same root sole PREPARING reservation; actual production guard
            // owns rollback/host-queue custody. No borrower exists before publish.
            if unsafe {
                pool.root.storage.initialize(
                    index,
                    Payload {
                        captures: c,
                        anchors: a,
                    },
                )
            }
            .is_err()
            {
                panic!("unexpected scalar initialization refusal");
            }
            preparation.initialized = true;
            panic!("bounded preparation unwind");
        }));
        assert!(result
            .as_ref()
            .err()
            .is_some_and(|payload| payload.downcast_ref::<&str>().copied()
                == Some("bounded preparation unwind")));
        assert_eq!(
            captures_dropped.load(Ordering::SeqCst),
            0,
            "producer unwind must not drop initialized captures"
        );
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(
            pool.root.storage.inventory().available(),
            2,
            "queued rollback is not completed credit"
        );
        assert_eq!(domain.snapshot().committed, bytes);
        assert!(service.manual_pass());
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    // Proposed additions INSIDE allocation_header_root::tests; source-only bodies.
    // Shared scalar_pool/inserted/owners helpers are the actual existing module's.
    #[test]
    fn prepared_empty_credit_drop_restores_only_header_eligibility() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<u64, (), 3>(&domain, &service, |_, _| true);
        let bytes = domain.snapshot().committed;
        let empty = pool.reserve_header().unwrap();
        assert_eq!(pool.root.storage.inventory().available(), 2);
        assert_eq!(domain.snapshot().committed, bytes);
        drop(empty);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        assert!(!service.manual_pass());
        assert_eq!(domain.snapshot().committed, bytes);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn prepared_publish_uses_actual_atomic_anchor_hook_and_counted_identity() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<u64, AtomicUsize, 3>(&domain, &service, |_, _| true);
        let empty = pool.reserve_header().unwrap();
        let prepared = match empty.initialize(7, AtomicUsize::new(0)) {
            Ok(prepared) => prepared,
            Err(rejected) => panic!("unexpected prepared scalar refusal: {:?}", rejected.error),
        };
        // SAFETY: exact-root initialized PREPARING borrow, scalar atom only; no
        // native/future capability or reference escapes before actual publish-last.
        unsafe { prepared.anchors() }.store(13, Ordering::Release);
        assert_eq!(pool.root.storage.inventory().available(), 2);
        let strong = prepared.publish().unwrap();
        assert_eq!(*strong.captures(), 7);
        // SAFETY: exact rooted counted LIVE strong spans this scalar-only borrow.
        assert_eq!(
            unsafe { strong.root.storage.payload(strong.ticket.index()) }
                .unwrap()
                .anchors
                .load(Ordering::Acquire),
            13
        );

        let same = strong.try_clone().unwrap();
        let weak = strong.downgrade().unwrap();
        assert!(HeaderStrong::same(&strong, &same));
        assert!(weak.same_strong(&strong));
        drop(same);
        drop(strong);
        assert!(weak.upgrade().unwrap().is_none());
        service.manual_pass();
        assert_eq!(pool.root.storage.inventory().available(), 2);
        drop(weak);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn prepared_initialized_drop_queues_custody_not_early_credit_or_owner_drop() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<Captures, Anchors, 3>(&domain, &service, readiness);
        let bytes = domain.snapshot().committed;
        let captures_dropped = Arc::new(AtomicUsize::new(0));
        let anchors_dropped = Arc::new(AtomicUsize::new(0));
        let (c, a) = owners(
            1,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicUsize::new(0)),
            &captures_dropped,
            &anchors_dropped,
            &domain,
            bytes,
        );
        let prepared = match pool.reserve_header().unwrap().initialize(c, a) {
            Ok(prepared) => prepared,
            Err(rejected) => panic!("unexpected prepared scalar refusal: {:?}", rejected.error),
        };
        drop(prepared);
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(
            pool.root.storage.inventory().available(),
            2,
            "queued is not completed rollback"
        );
        assert_eq!(domain.snapshot().committed, bytes);
        assert!(service.manual_pass());
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn prepared_initialized_unwind_uses_actual_wrapper_service_handoff() {
        let service = HostRetirementService::manual();
        let domain = process_account().unadmitted_domain(None).unwrap();
        let pool = scalar_pool::<Captures, Anchors, 3>(&domain, &service, readiness);
        let bytes = domain.snapshot().committed;
        let captures_dropped = Arc::new(AtomicUsize::new(0));
        let anchors_dropped = Arc::new(AtomicUsize::new(0));
        let (c, a) = owners(
            1,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicUsize::new(0)),
            &captures_dropped,
            &anchors_dropped,
            &domain,
            bytes,
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _prepared = match pool.reserve_header().unwrap().initialize(c, a) {
                Ok(prepared) => prepared,
                Err(rejected) => panic!("unexpected prepared scalar refusal: {:?}", rejected.error),
            };
            panic!("bounded prepared-wrapper unwind");
        }));
        assert!(result.as_ref().err().is_some_and(
            |p| p.downcast_ref::<&str>().copied() == Some("bounded prepared-wrapper unwind")
        ));
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(
            pool.root.storage.inventory().available(),
            2,
            "unwind retains queued credit"
        );
        assert_eq!(domain.snapshot().committed, bytes);
        assert!(service.manual_pass());
        assert_eq!(captures_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(anchors_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(pool.root.storage.inventory().available(), 3);
        drop(pool);
        service.manual_pass();
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
    #[test]
    fn context_cold_partition_conserves_tight_cap_and_final_service_refund() {
        let service = HostRetirementService::manual();
        let bytes = HeaderPool::<u64, (), 3>::requested_bytes().unwrap();
        let total = bytes.checked_mul(2).unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(total);
        let pass = domain.reserve(total).unwrap();
        let context = pass.context();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: total,
                committed: 0
            }
        );
        // SAFETY: scalar payload/anchors only; atomics-free true predicate needs
        // no native terminal receipt. Actual manual service roots the slab.
        let root = unsafe {
            HeaderRoot::<u64, (), 3>::cold_with_context_and_service(&context, |_, _| true, &service)
        }
        .unwrap();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: bytes,
                committed: bytes
            }
        );
        let remainder = context.take(bytes).unwrap();
        assert!(matches!(
            context.take(1),
            Err(AllocationError::Refused {
                domain: crate::memory_admission::RefusalDomain::Promise,
                ..
            })
        ));
        drop(remainder); // Returns exactly the remaining demand to open parent.
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: bytes,
                committed: bytes
            }
        );
        drop(root);
        assert!(!service.manual_pass()); // Empty storage final destruction/refund.
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: bytes,
                committed: 0
            }
        );
        drop(pass);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 0
            }
        );
        drop(context);
        assert!(!service.manual_pass());
    }

    #[test]
    fn context_cold_underbound_refuses_without_storage_charge() {
        let service = HostRetirementService::manual();
        let bytes = HeaderPool::<u64, (), 3>::requested_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(bytes);
        let pass = domain.reserve(bytes - 1).unwrap();
        let context = pass.context();
        // SAFETY: same bounded scalar-only contract; take must refuse before
        // allocation. Committed accounting is checked, not an allocator spy.
        let result = unsafe {
            HeaderRoot::<u64, (), 3>::cold_with_context_and_service(&context, |_, _| true, &service)
        };
        assert!(matches!(
            result,
            Err(RootError::Admission(AllocationError::Refused {
                domain: crate::memory_admission::RefusalDomain::Promise,
                ..
            }))
        ));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: bytes - 1,
                committed: 0
            }
        );
        assert!(!service.manual_pass());
        drop(pass);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 0
            }
        );
    }

    #[test]
    fn context_cold_cancelled_context_refuses_without_storage_charge() {
        let service = HostRetirementService::manual();
        let bytes = HeaderPool::<u64, (), 3>::requested_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(bytes);
        let pass = domain.reserve(bytes).unwrap();
        let context = pass.context();
        drop(pass);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 0
            }
        );
        // SAFETY: bounded scalar-only contract; cancelled authoritative issuance
        // boundary precedes allocation, not a guessed native readiness proof.
        let result = unsafe {
            HeaderRoot::<u64, (), 3>::cold_with_context_and_service(&context, |_, _| true, &service)
        };
        assert!(matches!(
            result,
            Err(RootError::Admission(AllocationError::Cancelled))
        ));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 0
            }
        );
        assert!(!service.manual_pass());
    }
    // Combined managed-layout controls use only fake payloads and manual service.
    // These use the actual context constructor, header publication and manual service.
    fn combined_pool_quote<C: Send + Sync + 'static, A: Send + Sync + 'static>() -> usize {
        HeaderPool::<Arc<crate::host_anchor::HostAnchorPermit<C, A>>, (), 1>::requested_bytes()
            .unwrap()
            .checked_add(
                crate::host_anchor::HostAnchorRegistry::<C, A>::requested_slab_bytes().unwrap(),
            )
            .unwrap()
    }
    struct CombinedCapture(Arc<AtomicUsize>);
    impl Drop for CombinedCapture {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct CombinedAnchor(Arc<AtomicUsize>);
    impl Drop for CombinedAnchor {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    type CombinedPermit = crate::host_anchor::HostAnchorPermit<CombinedCapture, CombinedAnchor>;
    fn combined_drain(service: &HostRetirementService) {
        for _ in 0..8 {
            service.manual_pass();
        }
    }
    #[test]
    fn combined_managed_quote_partitions_once_and_retains_charge_through_old_weak() {
        let service = HostRetirementService::manual();
        let quote = combined_pool_quote::<CombinedCapture, CombinedAnchor>();
        let outer_bytes = HeaderPool::<Arc<CombinedPermit>, (), 1>::requested_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote).unwrap();
        let context = pass.context();
        // SAFETY: outer C owns only final fake permit metadata; its scalar C/A are
        // independently registered below. No native use, raw root or pool backedge.
        let pool = HeaderPool {
            root: unsafe {
                HeaderRoot::<Arc<CombinedPermit>, (), 1>::cold_with_context_and_service(
                    &context,
                    |_, _| true,
                    &service,
                )
            }
            .unwrap(),
        };
        let empty = pool.reserve_header().unwrap();
        let registry = crate::host_anchor::HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            service.wake(),
        );
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let permit = registry
            .reserve_in(
                &context,
                CombinedCapture(captures.clone()),
                CombinedAnchor(anchors.clone()),
            )
            .unwrap_or_else(|_| panic!("combined admitted fake permit refused"));
        // Actual live capacities and committed slab quote, not physical malloc bytes.
        let (cells, ordered, actual_bytes, charged_bytes) = registry.managed_layout_fixture();
        assert_eq!(
            (cells, ordered),
            (32, 32),
            "actual requested vector capacities changed"
        );
        assert_eq!(
            actual_bytes, charged_bytes,
            "actual slab layout charge mismatch"
        );
        assert_eq!(
            outer_bytes + actual_bytes,
            quote,
            "combined quote mismatches actual managed layouts"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: quote
            },
            "combined metadata reserved a second load promise"
        );
        let prepared = empty
            .initialize(Arc::new(permit), ())
            .unwrap_or_else(|_| panic!("combined fake header initialize refused"));
        let strong = prepared.publish().unwrap();
        let weak = strong.downgrade().unwrap();
        drop(registry);
        drop(strong);
        assert_eq!(
            captures.load(Ordering::SeqCst),
            0,
            "producer destroyed independent captures"
        );
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            0,
            "producer destroyed independent anchors"
        );
        combined_drain(&service);
        assert_eq!(
            captures.load(Ordering::SeqCst),
            1,
            "service missed inner captures"
        );
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            1,
            "service missed inner anchors"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: outer_bytes
            },
            "old weak lost outer storage charge or inner slab refund"
        );
        drop(pool);
        combined_drain(&service);
        assert_eq!(
            domain.snapshot().committed,
            outer_bytes,
            "old weak failed to retain metadata charge"
        );
        drop(weak);
        combined_drain(&service);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot::default(),
            "final service lost combined refund"
        );
        drop(pass);
    }
    #[test]
    fn combined_underbound_inner_refusal_returns_payload_and_refunds_outer_storage() {
        let service = HostRetirementService::manual();
        let quote = combined_pool_quote::<CombinedCapture, CombinedAnchor>();
        let outer_bytes = HeaderPool::<Arc<CombinedPermit>, (), 1>::requested_bytes().unwrap();
        let domain = ModelDomain::isolated_numeric_fixture(quote);
        let pass = domain.reserve(quote - 1).unwrap();
        let context = pass.context();
        // SAFETY: same fake metadata-only contract as normal control; outer empty
        // storage may be allocated before the INNER undersized ticket refuses.
        let pool = HeaderPool {
            root: unsafe {
                HeaderRoot::<Arc<CombinedPermit>, (), 1>::cold_with_context_and_service(
                    &context,
                    |_, _| true,
                    &service,
                )
            }
            .unwrap(),
        };
        let empty = pool.reserve_header().unwrap();
        let registry = crate::host_anchor::HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            service.wake(),
        );
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let rejected = match registry.reserve_in(
            &context,
            CombinedCapture(captures.clone()),
            CombinedAnchor(anchors.clone()),
        ) {
            Ok(_) => panic!("undersized inner quote accepted"),
            Err(rejected) => rejected,
        };
        assert!(matches!(
            rejected.error,
            crate::host_anchor::AnchorError::Registry(
                crate::retirement_registry::RegistryError::Admission(AllocationError::Refused {
                    domain: crate::memory_admission::RefusalDomain::Promise,
                    ..
                })
            )
        ));
        assert_eq!(
            registry.managed_layout_fixture(),
            (0, 0, 0, 0),
            "inner refusal allocated/published managed slab"
        );
        assert_eq!(
            captures.load(Ordering::SeqCst),
            0,
            "refusal lost returned capture custody"
        );
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            0,
            "refusal lost returned anchor custody"
        );
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: quote - 1 - outer_bytes,
                committed: outer_bytes
            },
            "inner refusal changed admission totals"
        );
        drop(rejected.owner.expect("inner refusal must return fake C/A"));
        assert_eq!(captures.load(Ordering::SeqCst), 1);
        assert_eq!(anchors.load(Ordering::SeqCst), 1);
        drop(empty); // Only uninitialized exact-root credit restored, no payload.
        drop(registry);
        drop(pool);
        combined_drain(&service);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: quote - 1 - outer_bytes,
                committed: 0
            },
            "outer service refund missing"
        );
        drop(pass);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }
}
