//! Experimental private bank binding; production enrollment remains explicit.
//! Cold enrollment/diagnostic service plumbing and controls remain unfinished.
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLIOCommandBuffer, MTLSharedEvent};
use std::ptr::NonNull;

pub(crate) struct NativeIoCapture(Retained<ProtocolObject<dyn MTLIOCommandBuffer>>);
// SAFETY obligation at private creation: CB was freshly created/encoded by the
// engine, never externally committed/exported. After handoff only the unique
// private commit borrow and Metal's status callback access it; host service Drop
// requires actual terminal feedback plus returned handoff. This is not arbitrary
// concurrent mutation of a user-provided raw native CB.
unsafe impl Send for NativeIoCapture {}
unsafe impl Sync for NativeIoCapture {}

pub(crate) struct PersistentIoCohort {
    // Actual NativeBufferLease owners, not raw pointers/erased fake anchors.
    // Created cold for verified persistent expert allocations; all Vec capacity
    // and cohort metadata have an independently held real admission charge.
    destinations: Vec<crate::kernels::metal::buffer::NativeBufferLease>,
}
// Missing enrollment constructor must supply accounted lifetime of this cohort's
// actual metadata allocation and guarantee destinations are persistent/rich-use
// qualified. No constructor is fabricated from arbitrary &Buffer here.
type NativeBank = Bank<NativeIoCapture, Arc<PersistentIoCohort>>;
pub(crate) type NativePrepared = Prepared<NativeIoCapture, Arc<PersistentIoCohort>>;

struct NativeHandler(block2::RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLIOCommandBuffer>>)>);
// Exactly the existing M3 NativeHandler contract: closure captures only a
// counted metadata handle and cell index. Native block copies/releases retain it.
unsafe impl Send for NativeHandler {}
unsafe impl Sync for NativeHandler {}

fn cold_handler(bank: BankHandle<NativeIoCapture, Arc<PersistentIoCohort>>, index: usize)
    -> NativeHandler
{
    NativeHandler(block2::RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLIOCommandBuffer>>| {
        // SAFETY: callback invocation supplies a valid MTLIOCommandBuffer for
        // this invocation, as in metal3_retirement::native_handler. No borrowed
        // bank payload is read. Actual ticket/key are revalidated by callback.
        let raw = unsafe { cb.as_ref() };
        let status = RawStatus(raw.status().0 as usize);
        let result = bank.callback(index, cb.as_ptr() as *const () as usize, status);
        if let Err(error) = result { bank.note_callback_error(&error); }
    }))
}

// Bank service custody, not this unique lease, owns the native capture/cohort.
unsafe impl crate::io_submission::IoSubmissionOwner for NativePrepared {
    type Payload = (NativeIoCapture, Arc<PersistentIoCohort>);
    fn core(&self) -> &IoTerminal { &self.bank.core }
    fn cell(&self) -> &IoCell<Self::Payload> { &self.bank.slots[self.index].cell }
    fn refuse_before_handoff(&self, ticket: Ticket) -> Result<(), Error> {
        self.bank.refuse(self.index, ticket)
    }
    fn commit_native(&self, ticket: Ticket) {
        // Unique DeferredIo has already won this exact record's handoff CAS.
        // Native return has not been published, so strict service detachment
        // cannot destroy the bank-owned CB. Borrow the native object directly,
        // not a concurrently edited OwnedRecord and not a cloned native owner.
        assert_eq!(ticket, self.ticket, "private native IO capability ticket");
        let cb = self.native_key as *const ProtocolObject<dyn MTLIOCommandBuffer>;
        unsafe { (&*cb).commit(); }
    }
    fn request_progress(&self) { self.bank.request_progress(); }
    fn report_protocol_error(&self, error: Error) {
        self.bank.note_protocol_error(&error);
    }
}

// Immutable cold enrollment. Construct only after bank/service rooting, with
// exactly one correctly bound cold_handler per slot and admitted backing for
// this Vec/RcBlock set. The metadata cost and producer owner-qualified compute
// event must be supplied by the protected owner integration; no default is chosen.
pub(crate) struct NativeProducer {
    bank: BankHandle<NativeIoCapture, Arc<PersistentIoCohort>>,
    handlers: Vec<NativeHandler>,
    cohort: Arc<PersistentIoCohort>,
}

pub(crate) enum NativePrepareError {
    Admission(PrepareError<NativeIoCapture, Arc<PersistentIoCohort>>),
    Rooted(Error),
}
impl NativeProducer {
    /// The owning IoQueue creates/encodes cb freshly under existing authority;
    /// this private function receives it before any commit/raw export. Existing
    /// IOCB creation cost remains; RcBlock::new is NOT called in this method.
    pub(crate) fn prepare_native(
        &self,
        cb: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>,
    ) -> Result<(NativeDeferredIo, NativeIoDependency), NativePrepareError> {
        if self.bank.notice.get().is_none() {
            return Err(NativePrepareError::Admission(PrepareError::NotIssued {
                error: Error::Mailbox(MailboxError::Busy), captures: NativeIoCapture(cb),
                anchors: self.cohort.clone(),
            }));
        }
        let key = Retained::as_ptr(&cb) as *const () as usize;
        let prepared = NativeBank::try_prepare(
            &self.bank, NativeIoCapture(cb), self.cohort.clone(), key,
        ).map_err(NativePrepareError::Admission)?;
        // The actual CB is now rooted in the record and still Prepared. The
        // producer's immutable cold handler set is bound to this same bank.
        let raw = key as *const ProtocolObject<dyn MTLIOCommandBuffer>;
        unsafe { (&*raw).addCompletedHandler(block2::RcBlock::as_ptr(&self.handlers[prepared.index].0)); }
        // SAFETY: try_prepare is the sole constructor of the non-Clone prepared
        // capability. It has not escaped, been copied, or entered native code.
        let ticket = prepared.ticket;
        let dependency = NativeIoDependency { bank: prepared.bank.clone(), value: prepared.sequence + 1 };
        match unsafe { crate::io_submission::DeferredIo::from_prepared(prepared, ticket) } {
            Ok(lease) => Ok((lease, dependency)),
            Err(rejected) => {
                let ticket = rejected.owner.ticket;
                let bank = &rejected.owner.bank;
                if let Err(error) = bank.refuse(rejected.owner.index, ticket) {
                    bank.note_protocol_error(&error);
                }
                Err(NativePrepareError::Rooted(rejected.error))
            }
        }
    }
}

/// Notification-only event. Under the bank's existing metadata exclusion every
/// publication is monotone; failure uses MAX to wake every registered target.
/// No GPU consumer may wait on this event as a data-validity certificate.
pub(super) struct NativeNotice {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    published: AtomicU64,
}
// Shared events are explicitly usable across threads; the mutable publication
// metadata is atomic and writes are serialized by the bank metadata owner.
unsafe impl Send for NativeNotice {}
unsafe impl Sync for NativeNotice {}
impl NativeNotice {
    pub(super) fn failure_published(&self) -> bool {
        self.published.load(Ordering::Acquire) == u64::MAX
    }
    pub(super) fn publish(&self, successful: u64, failed: bool) {
        let value = if failed { u64::MAX } else { successful };
        if value > self.published.load(Ordering::Relaxed) {
            self.event.setSignaledValue(value);
            self.published.store(value, Ordering::Release);
        }
    }
}

pub(crate) type NativeDeferredIo = crate::io_submission::DeferredIo<NativePrepared>;

/// Retains the actual generation bank, not a copied readiness Boolean. Each
/// dependency is one lane's accepted sequence; another lane cannot satisfy it.
#[derive(Clone)]
pub(crate) struct NativeIoDependency {
    bank: BankHandle<NativeIoCapture, Arc<PersistentIoCohort>>,
    value: u64,
}
impl NativeIoDependency {
    pub(crate) fn value(&self) -> u64 { self.value }
    pub(crate) fn successful(&self) -> u64 { self.bank.core.successful() }
    pub(crate) fn ready(&self) -> bool { self.successful() >= self.value }
    pub(crate) fn generation_failed(&self) -> bool {
        self.bank.core.failed() || self.bank.failure.get().is_some()
    }
    pub(crate) fn failed(&self) -> bool { self.generation_failed() && !self.ready() }
    pub(crate) fn notification(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.bank.notice.get().expect("cold IO notice installed before dependency publication").event
    }
}

impl NativePrepareError {
    pub(crate) fn rooted(&self) -> bool {
        !matches!(self, Self::Admission(PrepareError::NotIssued { .. }))
    }
}

/// Test-only status fault injector for ACTUAL bank metadata and actual Commands
/// gates. It creates empty native IO buffers but does not commit them; injected
/// callback feedback is not evidence of native file IO/error behavior.
#[cfg(test)]
pub(crate) struct TestIoGeneration {
    bank: BankHandle<NativeIoCapture, Arc<PersistentIoCohort>>,
    prepared: Vec<Option<NativePrepared>>,
    service: Arc<crate::retirement_service::HostRetirementService>,
    _domain: crate::memory_admission::ModelDomain,
}
#[cfg(test)]
impl TestIoGeneration {
    pub(crate) fn new(device: &ProtocolObject<dyn objc2_metal::MTLDevice>, capacity: usize) -> Self {
        use objc2_metal::{MTLDevice, MTLIOCommandQueue, MTLIOCommandQueueDescriptor};
        use crate::memory_admission::ModelDomain;
        use crate::retirement_service::HostRetirementService;
        let domain = ModelDomain::isolated_numeric_fixture(1 << 20);
        let pass = domain.reserve(1 << 20).unwrap();
        let mut admission = pass.context().take(1 << 20).unwrap();
        let service = HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let wake = service.wake();
        let core = IoTerminal::new(RuntimeStorage::try_new(capacity).unwrap()).unwrap();
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity).unwrap();
        for _ in 0..capacity {
            slots.push(Slot { cell: IoCell::new(&core, failure.clone(), wake.clone()).unwrap(), native_key: AtomicUsize::new(0) });
        }
        let bytes = std::alloc::Layout::array::<Slot<NativeIoCapture, Arc<PersistentIoCohort>>>(slots.capacity()).unwrap().size()
            + std::mem::size_of::<NativeBank>() + core.order.lock().frontier.allocation_layout().size()
            + HostRetirementService::requested_registration_bytes();
        let charge = admission.try_commit(bytes).unwrap();
        let bank = unsafe { NativeBank::from_admitted(ColdParts { core, slots, failure, wake, charge }, &service) }
            .unwrap_or_else(|_| panic!("test bank admission"));
        let notice = device.newSharedEvent().unwrap();
        assert!(bank.notice.set(NativeNotice { event: notice, published: AtomicU64::new(0) }).is_ok());
        let descriptor = MTLIOCommandQueueDescriptor::new();
        let queue = device.newIOCommandQueueWithDescriptor_error(&descriptor).unwrap();
        // This empty test cohort owns no destinations. It cannot authorize real
        // loads and does not certify production metadata/cohort admission.
        let cohort = Arc::new(PersistentIoCohort { destinations: Vec::new() });
        let mut prepared = Vec::new();
        for _ in 0..capacity {
            let cb = queue.commandBuffer();
            let key = Retained::as_ptr(&cb) as *const () as usize;
            prepared.push(Some(NativeBank::try_prepare(&bank, NativeIoCapture(cb), cohort.clone(), key)
                .unwrap_or_else(|_| panic!("test prepared record"))));
        }
        Self { bank, prepared, service, _domain: domain }
    }
    pub(crate) fn dependency(&self, index: usize) -> NativeIoDependency {
        let p = self.prepared[index].as_ref().expect("dependency before injected terminal");
        NativeIoDependency { bank: self.bank.clone(), value: p.sequence + 1 }
    }
    pub(crate) fn deferred(&mut self, index: usize) -> NativeDeferredIo {
        let p = self.prepared[index].take().unwrap();
        let ticket = p.ticket;
        unsafe { crate::io_submission::DeferredIo::from_prepared(p, ticket) }
            .unwrap_or_else(|_| panic!("test unique deferred capability"))
    }
    pub(crate) fn fail_service_without_callback(&self) {
        // Same fanout as production WorkerFailed, no terminal injection/pass.
        self.service.close_admission();
    }
    pub(crate) fn untouched_custody(&self) -> (u64, u64, u64, usize, usize) {
        (self.bank.issued.load(Ordering::Acquire), self.bank.retired.load(Ordering::Acquire),
         self.bank.core.successful(), self.prepared.iter().filter(|p| p.is_some()).count(),
         self.bank.slots.iter().filter(|s| s.cell.mailbox.pending_ticket().is_some()).count())
    }
    pub(crate) fn terminal(&mut self, index: usize, status: usize) {
        let p = self.prepared[index].take().unwrap();
        self.bank.slots[p.index].cell.begin_handoff(p.ticket).unwrap();
        self.bank.callback(p.index, p.native_key, RawStatus(status)).unwrap();
        self.bank.slots[p.index].cell.handoff_returned(p.ticket).unwrap();
        self.bank.request_progress();
        while self.service.manual_pass() {}
    }
}
#[cfg(test)]
impl Drop for TestIoGeneration {
    fn drop(&mut self) {
        for p in &mut self.prepared {
            if let Some(p) = p.take() { let _ = self.bank.refuse(p.index, p.ticket); }
        }
        while self.service.manual_pass() {}
    }
}


#[cfg(test)]
mod failure_notice_tests {
    use super::*;

    fn generation(capacity: usize) -> TestIoGeneration {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal device");
        TestIoGeneration::new(device.as_ref(), capacity)
    }

    #[test]
    fn service_failure_notice_waits_for_metadata_holder_not_native_callback() {
        let generation = generation(2);
        let dependency = generation.dependency(1);
        let before = generation.untouched_custody();
        let mut during = None;
        // Actual holder path, deterministic losing try-lock; no extra scheduler.
        let retired = generation.bank.begin_retire_with(|| {
            generation.fail_service_without_callback();
            during = Some((dependency.notification().signaledValue(),
                           generation.bank.pending.load(Ordering::SeqCst)));
        });
        let after = (dependency.notification().signaledValue(), generation.untouched_custody());
        assert_eq!(during, Some((0, true)), "IO_FAILURE_BUSY_HOLDER_PRESERVES_PENDING");
        assert_eq!(after.0, u64::MAX, "IO_FAILURE_HOLDER_ASSISTS_WITHOUT_CALLBACK");
        assert_eq!(after.1, before, "IO_FAILURE_NOTICE_DOES_NOT_RETIRE_OR_HANDOFF");
        assert!(!retired);
        assert!(!dependency.ready());
    }

    #[test]
    fn service_failure_notice_cannot_regress_to_captured_lower_prefix() {
        let mut generation = generation(2);
        let dependency = generation.dependency(1);
        generation.terminal(0, 3);
        let captured_prefix = dependency.successful();
        assert_eq!(captured_prefix, 1);
        let before = generation.untouched_custody();
        let mut during = None;
        generation.bank.begin_retire_with(|| {
            generation.fail_service_without_callback();
            // Existing lower-prefix holder still owns the setter exclusion.
            generation.bank.notice.get().unwrap().publish(captured_prefix, false);
            during = Some(dependency.notification().signaledValue());
        });
        assert_eq!(during, Some(1), "IO_FAILURE_LOWER_HOLDER_FINISHES_BEFORE_MAX");
        assert_eq!(dependency.notification().signaledValue(), u64::MAX);
        {
            let _order = generation.bank.core.order.lock();
            // Explicit stale captured argument boundary, NOT new data validity.
            generation.bank.notice.get().unwrap().publish(captured_prefix, false);
        }
        assert_eq!(dependency.notification().signaledValue(), u64::MAX,
                   "IO_FAILURE_MAX_CANNOT_REGRESS_TO_STALE_PREFIX");
        assert_eq!(generation.untouched_custody(), before);
        assert_eq!(dependency.successful(), 1);
        assert!(!dependency.ready());
    }

    #[test]
    fn healthy_foreign_callback_diagnostic_does_not_publish_failure_notice() {
        let generation = generation(1);
        let dependency = generation.dependency(0);
        let before = generation.untouched_custody();
        let error = generation.bank.callback(usize::MAX, 1, RawStatus(3)).unwrap_err();
        assert!(matches!(error, CallbackError::WrongNativeIdentity));
        generation.bank.note_callback_error(&error);
        assert!(!dependency.generation_failed(), "IO_FOREIGN_DIAGNOSTIC_NOT_GENERATION_FAILURE");
        assert_eq!(dependency.notification().signaledValue(), 0);
        assert_eq!(generation.untouched_custody(), before);
        assert_eq!(generation.bank.diagnostics.load(Ordering::Acquire) & 2, 2);
    }
}
