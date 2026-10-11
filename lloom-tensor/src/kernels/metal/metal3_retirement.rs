//! Queue-associated Metal 3 logical completion prefix over reusable slab cells.
//!
//! This metadata core owns no native Queue, Device, callback capture, or physical
//! context. Registry/service custody is independent. Associated wrappers and
//! producer commit transactions are the only clients allowed to append links.
use super::{
    metal4::Feedback,
    retirement::{Captures, Metal3Anchor, Metal3RawOwners, NativeOwners},
};
use crate::{
    completion_wake::Wake,
    host_anchor::HostAnchorRegistry,
    retirement_mailbox::{CleanupPolicy, FeedbackPublicationError, MailboxError},
    retirement_owners::{Failure, FailureSignal},
    retirement_registry::{OwnerRegistry, PinnedRegistration, Registration},
};
use objc2_metal::MTLCommandBuffer;
use parking_lot::{Mutex, MutexGuard};
use std::sync::{Arc, OnceLock};

type Record = Registration<Captures, NativeOwners, Feedback>;
type Marker = PinnedRegistration<Captures, NativeOwners, Feedback>;

/// Unregistered captures/anchors return to caller custody. Caller must release
/// them outside any recording/commit lock. None means cancellation was routed
/// to the independently rooted host service, never dropped at this callsite.
pub(super) struct ReserveError {
    pub error: crate::kernels::MetalKernelError,
    pub rejected: Option<(Captures, NativeOwners)>,
}

pub(super) struct Completions {
    registry: OwnerRegistry<Captures, NativeOwners, Feedback>,
    raw_owners: HostAnchorRegistry<(), Metal3RawOwners>,
    // Host only: held from preparation through all raw commit calls/RETURNED.
    // Completion callbacks never acquire it. Queue identity is this Arc address.
    tail: Mutex<Marker>,
    // Native feedback only; never acquired by a producer/native handoff path.
    drain: Mutex<Drain>,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    // Native error is terminal evidence, not adapter poison. Drain still runs
    // every genuinely ready callback to release known-completed owners.
    native_error: OnceLock<String>,
}
struct Drain {
    cursor: Marker,
    next: Option<Marker>,
    elected: bool,
}
impl Completions {
    pub(super) fn new() -> Result<Arc<Self>, crate::kernels::MetalKernelError> {
        let failure = Arc::new(FailureSignal::default());
        let wake = Arc::new(Wake::default());
        let registry = OwnerRegistry::new_with_waiters(
            failure.clone(),
            wake.clone(),
            CleanupPolicy::ReturnedHandoff,
        )
        .map_err(|_| unavailable("M3 owner registry unavailable"))?;
        let raw_owners = HostAnchorRegistry::new(failure.clone(), wake.clone())
            .map_err(|_| unavailable("M3 host anchor registry unavailable"))?;
        Self::from_registry(registry, raw_owners, failure, wake)
    }
    fn from_registry(
        registry: OwnerRegistry<Captures, NativeOwners, Feedback>,
        raw_owners: HostAnchorRegistry<(), Metal3RawOwners>,
        failure: Arc<FailureSignal>,
        wake: Arc<Wake>,
    ) -> Result<Arc<Self>, crate::kernels::MetalKernelError> {
        let sentinel = registry
            .prepare(Captures(Vec::new()), NativeOwners::vacant(), |_| Ok(()))
            .map_err(|_| unavailable("M3 sentinel registration refused"))?;
        let pins = (|| {
            let tail = sentinel
                .pin()
                .map_err(|_| unavailable("M3 sentinel metadata refused"))?;
            let cursor = tail
                .try_clone()
                .map_err(|_| unavailable("M3 sentinel cursor refused"))?;
            Ok::<_, crate::kernels::MetalKernelError>((tail, cursor))
        })();
        // Always cancel after registration, including either pin failure. No
        // constructor error may strand a Prepared sentinel in service custody.
        let canceled = sentinel.cancel_before_handoff();
        let (tail, cursor) = pins?;
        canceled.map_err(|_| unavailable("M3 sentinel cancellation refused"))?;
        Ok(Arc::new(Self {
            registry,
            raw_owners,
            tail: Mutex::new(tail),
            drain: Mutex::new(Drain {
                cursor,
                next: None,
                elected: false,
            }),
            failure,
            wake,
            native_error: OnceLock::new(),
        }))
    }
    pub(super) fn identity(&self) -> usize {
        self as *const Self as usize
    }
    pub(super) fn check(&self) -> Result<(), crate::kernels::MetalKernelError> {
        if self.failure.get().is_some() {
            return Err(unavailable("M3 retirement queue failed"));
        }
        if let Some(error) = self.native_error.get() {
            return Err(crate::kernels::MetalKernelError::CommandBufferError(
                error.clone(),
            ));
        }
        Ok(())
    }
    pub(super) fn reserve(
        &self,
        captures: Captures,
        owners: NativeOwners,
    ) -> Result<Marker, ReserveError> {
        if let Err(error) = self.check() {
            return Err(ReserveError {
                error,
                rejected: Some((captures, owners)),
            });
        }
        let record = self
            .registry
            .prepare(captures, owners, |_| Ok(()))
            .map_err(|rejected| ReserveError {
                error: unavailable("M3 owner registration refused"),
                rejected: Some((rejected.captures, rejected.anchors)),
            })?;
        match record.pin() {
            Ok(pin) => Ok(pin),
            Err(_) => {
                if record.cancel_before_handoff().is_err() {
                    self.failure
                        .publish(Failure::AdapterFailed, || self.wake.notify());
                }
                Err(ReserveError {
                    error: unavailable("M3 marker pin refused"),
                    rejected: None,
                })
            }
        }
    }
    /// Holds actual raw-commit exclusion, not merely a sequence counter. A
    /// caller must retain this guard through begin_handoff/append/native entry/
    /// RETURNED. Unused preparations append nothing.
    pub(super) fn prepare_order(
        &self,
    ) -> Result<MutexGuard<'_, Marker>, crate::kernels::MetalKernelError> {
        self.check()?;
        if let Some(guard) = self.tail.try_lock() {
            return Ok(guard);
        }
        crate::completion_scope::before_park(self.identity())
            .map_err(|_| crate::kernels::MetalKernelError::CallbackWouldBlock)?;
        let guard = self.tail.lock();
        self.check()?;
        Ok(guard)
    }
    /// Both pins were obtained before any irreversible submission side effect.
    /// Caller supplies its held prepare_order guard and retains it across raw
    /// native commit. Publishing a link alone is not a native-order guarantee.
    pub(super) fn append(
        &self,
        tail: &mut MutexGuard<'_, Marker>,
        link: Marker,
        new_tail: Marker,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        tail.append_successor(link)
            .map_err(|_| unavailable("duplicate M3 successor publication"))?;
        **tail = new_tail;
        Ok(())
    }
    pub(super) fn native_feedback(&self, record: Record, read: impl FnOnce() -> Feedback) {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Ready cannot detach before this feedback runs its callbacks.
            let marker = record.pin()?;
            record
                .publish_feedback_observed(read(), |_, feedback| {
                    if let Some(error) = &feedback.error {
                        let _ = self.native_error.set(error.clone());
                    }
                    marker.publish_native_ready();
                    self.wake.notify();
                })
                .map_err(|e| match e {
                    FeedbackPublicationError::Rejected(rejected) => rejected.error,
                    FeedbackPublicationError::ObserverPanicked => MailboxError::CallbackPanicked,
                })?;
            marker.publish_ordered_ready();
            self.drain_ready()
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(_)) => self
                .failure
                .publish(Failure::AdapterFailed, || self.wake.notify()),
            Err(payload) => {
                if let Err(payload) = record.retain_boundary_panic(payload) {
                    let _ = Box::leak(payload);
                }
                self.failure
                    .publish(Failure::CallbackPanic, || self.wake.notify());
            }
        }
    }
    fn drain_ready(&self) -> Result<(), MailboxError> {
        {
            let mut drain = self.drain.lock();
            if drain.elected || self.failure.get().is_some() {
                return Ok(());
            }
            drain.elected = true;
        }
        loop {
            let next = {
                let mut drain = self.drain.lock();
                if self.failure.get().is_some() {
                    return Ok(());
                }
                if drain.next.is_none() {
                    drain.next = drain.cursor.take_successor();
                }
                if !drain.next.as_ref().is_some_and(Marker::ordered_ready) {
                    // Publish election availability under the same lock that
                    // every feedback takes after its Ready publication.
                    drain.elected = false;
                    return Ok(());
                }
                drain.next.as_ref().expect("ready successor").try_clone()?
            };
            next.record().run_callbacks(
                |captures, _, feedback| captures.run(self.identity(), feedback),
                || self.wake.notify(),
            )?;
            next.publish_callbacks_finished();
            self.wake.notify();
            let previous = {
                let mut drain = self.drain.lock();
                let retired = drain.next.take().expect("elected successor remains owned");
                std::mem::replace(&mut drain.cursor, retired)
            };
            // Metadata-only drop. No arbitrary capture destructor runs here.
            drop(previous);
            // Concurrent failure permits only this already-started record to
            // finish; the next loop cannot start a later callback.
        }
    }
    pub(super) fn try_wait(
        &self,
        marker: &Marker,
        logical: bool,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        loop {
            let seen = self.wake.snapshot();
            self.check()?;
            if if logical {
                marker.callbacks_finished()
            } else {
                marker.native_ready()
            } {
                return Ok(());
            }
            crate::completion_scope::before_park(self.identity())
                .map_err(|_| crate::kernels::MetalKernelError::CallbackWouldBlock)?;
            self.wake.wait(seen);
        }
    }
}
fn unavailable(message: &'static str) -> crate::kernels::MetalKernelError {
    crate::kernels::MetalKernelError::QueueUnavailable(message)
}

#[derive(Clone, Copy)]
enum CommitPhase {
    Unused,
    HandlerAttached,
    Linked,
    EnteredNative,
    Returned,
}

/// Only associated engine wrappers construct this transaction. Standalone raw
/// CommandBuffer::new retains its explicit native-only compatibility contract.
/// The actual wrapper constructor/routing remains an integration prerequisite.
pub(super) struct PreparedCommit<'a> {
    core: &'a Arc<Completions>,
    raw: &'a objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    marker: &'a Marker,
    order: MutexGuard<'a, Marker>,
    link: Option<Marker>,
    new_tail: Option<Marker>,
    handler: block2::RcBlock<
        dyn Fn(
            std::ptr::NonNull<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
        ),
    >,
    phase: CommitPhase,
}
impl Completions {
    pub(super) fn prepare_commit<'a>(
        self: &'a Arc<Self>,
        raw: &'a objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >,
        marker: &'a Marker,
    ) -> Result<PreparedCommit<'a>, crate::kernels::MetalKernelError> {
        let order = self.prepare_order()?;
        if !matches!(
            raw.status(),
            objc2_metal::MTLCommandBufferStatus::NotEnqueued
                | objc2_metal::MTLCommandBufferStatus::Enqueued
        ) {
            return Err(unavailable("associated M3 buffer already committed"));
        }
        let link = marker
            .try_clone()
            .map_err(|_| unavailable("M3 successor pin refused"))?;
        let new_tail = marker
            .try_clone()
            .map_err(|_| unavailable("M3 tail pin refused"))?;
        let record = marker.record().clone();
        let completion = self.clone();
        // This allocation happens before append/native handoff. Unused/drop
        // cannot leave a sequence hole or a native-installed handler.
        let handler = block2::RcBlock::new(
            move |cb: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
            >| {
                completion.native_feedback(record.clone(), || {
                    // SAFETY: Metal supplies the completed buffer for this native
                    // callback. Only terminal status/error/timestamps are read.
                    let cb = unsafe { cb.as_ref() };
                    Feedback {
                        error: (cb.status() == objc2_metal::MTLCommandBufferStatus::Error).then(
                            || {
                                cb.error()
                                    .map(|e| e.localizedDescription().to_string())
                                    .unwrap_or_else(|| "unknown Metal command buffer error".into())
                            },
                        ),
                        start: cb.GPUStartTime(),
                        end: cb.GPUEndTime(),
                    }
                });
            },
        );
        Ok(PreparedCommit {
            core: self,
            raw,
            marker,
            order,
            link: Some(link),
            new_tail: Some(new_tail),
            handler,
            phase: CommitPhase::Unused,
        })
    }
}
impl PreparedCommit<'_> {
    pub(super) fn commit(mut self) -> Result<(), crate::kernels::MetalKernelError> {
        self.core.check()?;
        // Record phase before entering Objective-C. Handler installation cannot
        // execute GPU work; cancellation can still use host-only destruction.
        self.phase = CommitPhase::HandlerAttached;
        // SAFETY: Metal copies this preallocated metadata-only block. No block
        // disposal owns the last callback/physical payload reference.
        unsafe {
            self.raw
                .addCompletedHandler(block2::RcBlock::as_ptr(&self.handler));
        }
        self.core.check()?;
        let link = self.link.take().expect("prepared successor pin");
        let new_tail = self.new_tail.take().expect("prepared tail pin");
        self.core.append(&mut self.order, link, new_tail)?;
        self.phase = CommitPhase::Linked;
        self.marker
            .record()
            .begin_handoff()
            .map_err(|_| unavailable("M3 native handoff refused"))?;
        self.phase = CommitPhase::EnteredNative;
        // Commit-order guard still held: no second associated producer can
        // append/commit while this raw call has not returned. Public enqueue
        // may already have reserved native placement; logical prefix remains
        // commit order, not a claim that earlier enqueue never changes order.
        self.raw.commit();
        self.marker
            .record()
            .handoff_returned()
            .map_err(|_| unavailable("M3 handoff return rejected"))?;
        self.phase = CommitPhase::Returned;
        Ok(())
    }
}
impl Drop for PreparedCommit<'_> {
    fn drop(&mut self) {
        match self.phase {
            CommitPhase::Unused | CommitPhase::Returned => {}
            CommitPhase::HandlerAttached | CommitPhase::Linked => {
                // Known no raw commit. Prepared captures remain in service
                // custody, including error paths under the producer's locks.
                if self.marker.record().cancel_before_handoff().is_err()
                    || matches!(self.phase, CommitPhase::Linked)
                {
                    self.core
                        .failure
                        .publish(Failure::AdapterFailed, || self.core.wake.notify());
                }
            }
            CommitPhase::EnteredNative => {
                // Unknown native return is NOT terminality. Strict cleanup
                // policy retains CB/physical/capture owners and wakes failures.
                let _ = self
                    .marker
                    .record()
                    .handoff_unknown(|| self.core.wake.notify());
                self.core
                    .failure
                    .publish(Failure::UnknownHandoff, || self.core.wake.notify());
            }
        }
    }
}

/// Queue-associated M3 wrapper stored directly in CommandBuffer's backend enum;
/// no new per-buffer Arc allocation. Native raw identity remains unchanged.
pub(super) struct AssociatedBuffer {
    raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    marker: Marker,
    core: Arc<Completions>,
    anchor: Metal3Anchor,
}
pub(super) struct CallbackRejected {
    pub error: crate::kernels::MetalKernelError,
    pub callback: Option<super::retirement::Callback>,
}
impl AssociatedBuffer {
    pub(super) fn new(
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
        core: Arc<Completions>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Result<Self, crate::kernels::MetalKernelError> {
        let anchor = core
            .raw_owners
            .reserve((), Metal3RawOwners { raw, physical }, |_| Ok(()))
            .map_err(|rejected| {
                crate::kernels::err::CompletionRegistrationError::new(
                    unavailable("M3 raw owner registration refused"),
                    rejected.owner,
                )
            })?;
        // The original retain entered registered custody before a wrapper retain
        // can escape. Even a post-registration pin failure owns no raw local.
        let raw = match anchor.edit((), |_, owner, _| owner.raw.clone()) {
            Ok((raw, _)) => raw,
            Err(_) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    unavailable("M3 raw anchor borrow refused"),
                    Some(anchor),
                )
                .into())
            }
        };
        let native_anchor = match anchor.try_clone() {
            Ok(permit) => permit,
            Err(_) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    unavailable("M3 native anchor permit exhausted"),
                    Some((
                        Metal3RawOwners {
                            raw,
                            physical: None,
                        },
                        anchor,
                    )),
                )
                .into())
            }
        };
        let marker = match core.reserve(Captures(Vec::new()), NativeOwners::metal3(native_anchor)) {
            Ok(marker) => marker,
            Err(rejected) => {
                return Err(crate::kernels::err::CompletionRegistrationError::new(
                    rejected.error,
                    Some((
                        Metal3RawOwners {
                            raw,
                            physical: None,
                        },
                        anchor,
                        rejected.rejected,
                    )),
                )
                .into())
            }
        };
        if marker.add_wrapper_handle().is_err() {
            let _ = marker.record().cancel_before_handoff();
            return Err(crate::kernels::err::CompletionRegistrationError::new(
                unavailable("M3 wrapper count exhausted"),
                Some((
                    Metal3RawOwners {
                        raw,
                        physical: None,
                    },
                    anchor,
                )),
            )
            .into());
        }
        Ok(Self {
            raw,
            marker,
            core,
            anchor,
        })
    }
    pub(super) fn raw(
        &self,
    ) -> &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>
    {
        &self.raw
    }
    pub(super) fn prepare_commit(
        &self,
    ) -> Result<PreparedCommit<'_>, crate::kernels::MetalKernelError> {
        self.core.prepare_commit(&self.raw, &self.marker)
    }
    pub(super) fn try_commit(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.prepare_commit()?.commit()
    }
    /// Internal fallible registration. An unaccepted callback is returned so
    /// the engine can release it outside producer locks; no hidden Drop here.
    pub(super) fn try_on_completion(
        &self,
        callback: super::retirement::Callback,
    ) -> Result<(), CallbackRejected> {
        if let Err(error) = self.core.check() {
            return Err(CallbackRejected {
                error,
                callback: Some(callback),
            });
        }
        match self
            .marker
            .record()
            .edit_prepared(callback, |captures, _, input| {
                // Fallible/allocation work precedes transfer. After take, push into
                // reserved capacity cannot allocate or invoke arbitrary user code.
                if captures.0.try_reserve(1).is_err() {
                    return Err(());
                }
                let Some(callback) = input.take() else {
                    return Err(());
                };
                captures.0.push(callback);
                Ok(())
            }) {
            Ok((Ok(()), None)) => Ok(()),
            Ok((_, callback)) => Err(CallbackRejected {
                error: unavailable("M3 callback metadata allocation refused"),
                callback,
            }),
            Err(crate::retirement_mailbox::PreparedEditError::Rejected(rejected)) => {
                Err(CallbackRejected {
                    error: unavailable("M3 callback registration phase refused"),
                    callback: Some(rejected.value),
                })
            }
            Err(crate::retirement_mailbox::PreparedEditError::Panicked { input }) => {
                Err(CallbackRejected {
                    error: unavailable("M3 callback recording edit panicked"),
                    callback: input,
                })
            }
        }
    }
    pub(super) fn check_data_wait(&self) -> Result<(), crate::kernels::MetalKernelError> {
        self.core.check()?;
        if self.marker.native_ready() {
            return Ok(());
        }
        crate::completion_scope::before_park(self.core.identity())
            .map_err(|_| crate::kernels::MetalKernelError::CallbackWouldBlock)
    }
    pub(super) fn try_wait(&self, logical: bool) -> Result<(), crate::kernels::MetalKernelError> {
        self.core.try_wait(&self.marker, logical)
    }
}
impl Clone for AssociatedBuffer {
    fn clone(&self) -> Self {
        // Acquire all fallible retain/pin state before counting a new wrapper.
        // Existing self keeps the anchor live if any checked counter refuses.
        let raw = self.raw.clone();
        let anchor = self
            .anchor
            .try_clone()
            .expect("M3 anchor permit count exhausted");
        let marker = self
            .marker
            .try_clone()
            .expect("M3 metadata pin count exhausted");
        let core = self.core.clone();
        marker
            .add_wrapper_handle()
            .expect("M3 wrapper count exhausted");
        Self {
            raw,
            marker,
            core,
            anchor,
        }
    }
}
impl Drop for AssociatedBuffer {
    fn drop(&mut self) {
        if self.marker.release_wrapper_handle() {
            // Only final associated wrapper cancels a still-Prepared recording.
            // Native blocks/producer/drainer links are metadata pins, not user
            // wrapper references. After handoff this returns WrongPhase and
            // real native feedback/strict service custody determines cleanup.
            let _ = self.marker.record().cancel_before_handoff();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retirement_service::HostRetirementService;
    use std::{sync::mpsc, time::Duration};

    fn core() -> (Arc<HostRetirementService>, Arc<Completions>) {
        let service = HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let wake = Arc::new(Wake::default());
        let registry = OwnerRegistry::manual_with_waiters(
            failure.clone(),
            service.clone(),
            wake.clone(),
            CleanupPolicy::ReturnedHandoff,
        );
        let raw_owners = HostAnchorRegistry::manual(failure.clone(), service.clone(), wake.clone());
        let core = Completions::from_registry(registry, raw_owners, failure, wake).unwrap();
        (service, core)
    }
    fn reserve(core: &Completions, callback: super::super::retirement::Callback) -> Marker {
        core.reserve(Captures(vec![callback]), NativeOwners::vacant())
            .unwrap_or_else(|_| panic!("fake owner reserve"))
    }
    fn handoff(core: &Completions, marker: &Marker) {
        // Fake native handoff: exercises actual append and mailbox transitions,
        // not ObjC PreparedCommit/AssociatedBuffer or real queue placement.
        let mut order = core.prepare_order().unwrap();
        core.append(
            &mut order,
            marker.try_clone().unwrap(),
            marker.try_clone().unwrap(),
        )
        .unwrap();
        marker.record().begin_handoff().unwrap();
        marker.record().handoff_returned().unwrap();
    }
    fn feedback(error: Option<&str>) -> Feedback {
        Feedback {
            error: error.map(str::to_owned),
            start: 0.,
            end: 1.,
        }
    }
    fn cleanup(service: &HostRetirementService) {
        for _ in 0..64 {
            if !service.manual_pass() {
                return;
            }
        }
        panic!("bounded manual cleanup did not drain");
    }

    #[test]
    fn data_wait_from_callback_refuses_pending_same_queue_marker() {
        let (service, core) = core();
        let later = reserve(&core, Box::new(|_| {}));
        let held_later = later.try_clone().unwrap();
        let weak = Arc::downgrade(&core);
        let (tx, rx) = mpsc::channel();
        let first = reserve(
            &core,
            Box::new(move |_| {
                let result = weak.upgrade().unwrap().try_wait(&held_later, false);
                tx.send(matches!(
                    result,
                    Err(crate::kernels::MetalKernelError::CallbackWouldBlock)
                ))
                .unwrap();
            }),
        );
        handoff(&core, &first);
        handoff(&core, &later);
        let worker_core = core.clone();
        let first_record = first.record().clone();
        let worker = std::thread::spawn(move || {
            worker_core.native_feedback(first_record, || feedback(None))
        });
        let observed = rx.recv_timeout(Duration::from_secs(2));
        if observed.is_err() {
            // Bounded cleanup even if a mutation parks the production wait.
            core.failure
                .publish(Failure::Closing, || core.wake.notify());
        }
        worker.join().unwrap();
        assert_eq!(
            observed,
            Ok(true),
            "callback data wait did not return typed no-park refusal"
        );
        core.native_feedback(later.record().clone(), || feedback(None));
        assert!(first.callbacks_finished());
        assert!(later.callbacks_finished());
        cleanup(&service);
        assert!(
            first.callbacks_finished(),
            "old marker lost logical state after payload cleanup"
        );
    }

    #[test]
    fn out_of_order_feedback_and_arrival_during_callback_need_no_extra_wake() {
        let (service, core) = core();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let c0 = calls.clone();
        let first = reserve(&core, Box::new(move |_| c0.lock().push(0)));
        let c1 = calls.clone();
        let second = reserve(&core, Box::new(move |_| c1.lock().push(1)));
        handoff(&core, &first);
        handoff(&core, &second);
        core.native_feedback(second.record().clone(), || feedback(None));
        assert!(second.native_ready());
        assert!(!second.callbacks_finished());
        assert!(
            calls.lock().is_empty(),
            "later callback overtook first native completion"
        );
        core.native_feedback(first.record().clone(), || feedback(None));
        assert_eq!(*calls.lock(), vec![0, 1]);

        let (entered_tx, entered_rx) = mpsc::channel();
        let (leave_tx, leave_rx) = mpsc::channel();
        let leave_rx = Mutex::new(leave_rx);
        let c2 = calls.clone();
        let third = reserve(
            &core,
            Box::new(move |_| {
                c2.lock().push(2);
                entered_tx.send(()).unwrap();
                leave_rx
                    .lock()
                    .recv_timeout(Duration::from_secs(2))
                    .unwrap();
            }),
        );
        let c3 = calls.clone();
        let fourth = reserve(&core, Box::new(move |_| c3.lock().push(3)));
        handoff(&core, &third);
        handoff(&core, &fourth);
        let worker_core = core.clone();
        let third_record = third.record().clone();
        let worker = std::thread::spawn(move || {
            worker_core.native_feedback(third_record, || feedback(None))
        });
        let entered = entered_rx.recv_timeout(Duration::from_secs(2));
        let during = if entered.is_ok() {
            core.native_feedback(fourth.record().clone(), || feedback(None));
            let observed = (fourth.native_ready(), fourth.callbacks_finished());
            let released = leave_tx.send(()).is_ok();
            Some((observed, released))
        } else {
            core.failure
                .publish(Failure::Closing, || core.wake.notify());
            None
        };
        worker.join().unwrap();
        assert_eq!(
            during,
            Some(((true, false), true)),
            "later data visibility or callback exclusion failed"
        );
        assert!(
            entered.is_ok(),
            "first callback did not enter bounded fixture"
        );
        assert_eq!(
            *calls.lock(),
            vec![0, 1, 2, 3],
            "elected drainer lost feedback arriving inside callback"
        );
        assert!(fourth.callbacks_finished());
        cleanup(&service);
        assert!(fourth.callbacks_finished());
    }

    #[test]
    fn native_error_is_visible_to_other_marker_without_stopping_terminal_drain() {
        let (service, core) = core();
        let first = reserve(&core, Box::new(|_| {}));
        let second = reserve(&core, Box::new(|_| {}));
        handoff(&core, &first);
        handoff(&core, &second);
        core.native_feedback(second.record().clone(), || {
            feedback(Some("fixture native failure"))
        });
        assert!(
            matches!(core.try_wait(&first, false), Err(crate::kernels::MetalKernelError::CommandBufferError(message)) if message == "fixture native failure")
        );
        assert!(!first.native_ready());
        assert!(!second.callbacks_finished());
        assert_eq!(
            core.failure.get(),
            None,
            "native terminal error became adapter poison"
        );
        core.native_feedback(first.record().clone(), || feedback(None));
        assert!(first.callbacks_finished());
        assert!(second.callbacks_finished());
        cleanup(&service);
    }
}
