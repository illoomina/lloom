//! Unapplied child of the private bank module; actual objc2/Metal binding.
//! Cold enrollment/diagnostic service plumbing and controls remain unfinished.
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLIOCommandBuffer;
use std::ptr::NonNull;

pub(super) struct NativeIoCapture(Retained<ProtocolObject<dyn MTLIOCommandBuffer>>);
// SAFETY obligation at private creation: CB was freshly created/encoded by the
// engine, never externally committed/exported. After handoff only the unique
// private commit borrow and Metal's status callback access it; host service Drop
// requires actual terminal feedback plus returned handoff. This is not arbitrary
// concurrent mutation of a user-provided raw native CB.
unsafe impl Send for NativeIoCapture {}
unsafe impl Sync for NativeIoCapture {}

pub(super) struct PersistentIoCohort {
    // Actual NativeBufferLease owners, not raw pointers/erased fake anchors.
    // Created cold for verified persistent expert allocations; all Vec capacity
    // and cohort metadata have an independently held real admission charge.
    destinations: Vec<crate::kernels::metal::buffer::NativeBufferLease>,
}
// Missing enrollment constructor must supply accounted lifetime of this cohort's
// actual metadata allocation and guarantee destinations are persistent/rich-use
// qualified. No constructor is fabricated from arbitrary &Buffer here.
type NativeBank = Bank<NativeIoCapture, Arc<PersistentIoCohort>>;
type NativePrepared = Prepared<NativeIoCapture, Arc<PersistentIoCohort>>;

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
struct NativeProducer {
    bank: BankHandle<NativeIoCapture, Arc<PersistentIoCohort>>,
    handlers: Vec<NativeHandler>,
    cohort: Arc<PersistentIoCohort>,
}

enum NativePrepareError {
    Admission(PrepareError<NativeIoCapture, Arc<PersistentIoCohort>>),
    Rooted(Error),
}
impl NativeProducer {
    /// The owning IoQueue creates/encodes cb freshly under existing authority;
    /// this private function receives it before any commit/raw export. Existing
    /// IOCB creation cost remains; RcBlock::new is NOT called in this method.
    fn prepare_native(
        &self,
        cb: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>,
    ) -> Result<crate::io_submission::DeferredIo<NativePrepared>, NativePrepareError> {
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
        match unsafe { crate::io_submission::DeferredIo::from_prepared(prepared, ticket) } {
            Ok(lease) => Ok(lease),
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
