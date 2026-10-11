//! Native payload custody for the shared host retirement registry.
//!
//! Engine prepared-commit sites use this source draft; physical context/IO
//! custody and public compatibility paths remain incomplete, not native-validated.
use super::metal4;
use crate::{
    completion_frontier::{CompletionFrontier, FrontierError},
    completion_wake::Wake,
    retirement_mailbox::{FeedbackPublicationError, MailboxError},
    retirement_owners::{Failure, FailureSignal},
    retirement_registry::{OwnerRegistry, Registration, RegistryError},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLSharedEvent};
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) type Callback = Box<dyn Fn(&metal4::Feedback) + Send + Sync>;
pub(super) type Registry = OwnerRegistry<Captures, NativeOwners, metal4::Feedback>;
pub(super) type Record = Registration<Captures, NativeOwners, metal4::Feedback>;

pub(super) struct Captures(pub Vec<Callback>);
impl Captures {
    pub fn run(&self, queue: usize, feedback: &metal4::Feedback) {
        crate::completion_scope::with_callback(queue, || {
            // Borrow callback boxes; destruction belongs to the host service.
            for callback in &self.0 {
                callback(feedback);
            }
        });
    }
}

/// No arbitrary callback boxes here. Declaration order keeps native resources
/// and submission state alive before releasing the physical aggregate.
pub(super) type Metal3Anchor = crate::host_anchor::HostAnchorPermit<(), Metal3RawOwners>;

/// The registry, never a callback or final wrapper stack, destroys these owners.
/// Raw native lifetime ends before its corresponding physical accounting owner.
pub(super) struct Metal3RawOwners {
    pub raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub physical: Option<Arc<dyn Send + Sync>>,
}
// SAFETY: same cross-thread retain/release contract as CommandBuffer.
unsafe impl Send for Metal3RawOwners {}
unsafe impl Sync for Metal3RawOwners {}

pub(super) struct NativeOwners {
    backend: NativeBackend,
    physical: Option<Arc<dyn Send + Sync>>,
}
enum NativeBackend {
    Vacant,
    Metal3(Metal3Anchor),
    Metal4 {
        slot: Option<metal4::Slot>,
        queue: Arc<metal4::Queue>,
        submission: Option<Arc<metal4::Submission>>,
        signals: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    },
}
// SAFETY: Metal CB wrappers already permit cross-thread retention. The M4 slot
// is accessed only through the mailbox's phase-exclusive mutable callback
// borrow; cleanup cannot run until that borrow ends. Physical owners are Send + Sync.
unsafe impl Send for NativeOwners {}
unsafe impl Sync for NativeOwners {}

impl NativeOwners {
    pub fn vacant() -> Self {
        Self {
            backend: NativeBackend::Vacant,
            physical: None,
        }
    }

    pub fn metal3(anchor: Metal3Anchor) -> Self {
        Self {
            backend: NativeBackend::Metal3(anchor),
            physical: None,
        }
    }
    pub fn metal4(
        slot: metal4::Slot,
        submission: Arc<metal4::Submission>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Self {
        submission.install_native_slot(slot);
        Self {
            backend: NativeBackend::Metal4 {
                slot: None,
                queue: submission.queue.clone(),
                submission: Some(submission),
                signals: Vec::new(),
            },
            physical,
        }
    }
    /// Abandonment transfers the actual recording owners, not just an empty
    /// metadata ticket. No Arc<Submission> is resurrected during its Drop.
    pub fn metal4_abandoned(
        slot: metal4::Slot,
        queue: Arc<metal4::Queue>,
        signals: Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
        physical: Option<Arc<dyn Send + Sync>>,
    ) -> Self {
        Self {
            backend: NativeBackend::Metal4 {
                slot: Some(slot),
                queue,
                submission: None,
                signals,
            },
            physical,
        }
    }

    pub fn publish_native_feedback(&self, feedback: &metal4::Feedback) {
        if let NativeBackend::Metal4 {
            submission: Some(submission),
            ..
        } = &self.backend
        {
            submission.publish_retirement_feedback(feedback);
        }
    }
    /// Ordered callbacks finished, but reuse also needs host handoff return.
    /// No native slot capacity depends on the host destruction worker.
    pub fn return_slot(&mut self) {
        if let NativeBackend::Metal4 {
            submission: Some(submission),
            ..
        } = &self.backend
        {
            submission.publish_native_reuse(crate::native_reuse::ReuseSide::CallbacksDone);
            submission.notify_retirement_callbacks_finished();
        }
    }
}

pub(super) fn registry(
    failure: Arc<FailureSignal>,
    waiters: Arc<Wake>,
) -> Result<Registry, RegistryError> {
    Registry::new_with_waiters(
        failure,
        waiters,
        crate::retirement_mailbox::CleanupPolicy::ReturnedHandoff,
    )
}

impl Drop for NativeOwners {
    fn drop(&mut self) {
        if let NativeBackend::Metal4 {
            slot,
            queue,
            submission,
            ..
        } = &mut self.backend
        {
            if let Some(submission) = submission {
                submission.discard_cancelled_native_slot();
            }
            if let Some(slot) = slot.take() {
                // Only cancellation before native handoff reaches destruction
                // with a slot still present. Unknown/failed callbacks stay rooted.
                queue.discard_retirement_slot(slot);
            }
        }
    }
}

/// Metadata-only feedback core. A native block may retain this Arc without
/// retaining the Queue or physical context. The registry independently roots
/// each Record's payload until host cleanup completes.
///
/// Only native feedback takes this lock, never the recording thread. Capacity
/// is the M4 native-slot bound, not the growable registry metadata capacity.
/// The M3 queue adapter needs its separately sized prefix representation.
pub(super) struct Metal4Completions {
    frontier: Mutex<CompletionFrontier<Record, 8>>,
    wake: Arc<Wake>,
    failure: Arc<FailureSignal>,
    queue: usize,
}
#[derive(Debug)]
pub(super) enum RetirementError {
    Mailbox(MailboxError),
    Frontier(FrontierError),
}
impl Metal4Completions {
    pub fn new(queue: usize, failure: Arc<FailureSignal>, wake: Arc<Wake>) -> Self {
        Self {
            frontier: Mutex::new(CompletionFrontier::new().expect("positive native capacity")),
            wake,
            failure,
            queue,
        }
    }

    /// Entire Rust callback boundary. Physical owners never enter this stack.
    /// A malformed/already-retired record cannot accept a panic box; retaining
    /// that box for process lifetime is safer than dropping arbitrary captures
    /// on the driver thread. This exceptional leak accompanies queue refusal.
    pub fn native_feedback(
        &self,
        sequence: u64,
        record: Record,
        read_feedback: impl FnOnce() -> metal4::Feedback,
    ) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.publish(sequence, record.clone(), read_feedback())
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(_error)) => self
                .failure
                .publish(Failure::AdapterFailed, || self.wake.notify()),
            Err(payload) => {
                if let Err(rejected) = record.retain_boundary_panic(payload) {
                    let _ = Box::leak(rejected);
                }
                self.failure
                    .publish(Failure::CallbackPanic, || self.wake.notify());
            }
        }
    }

    /// Called once per real native feedback. Commit-order registration assigns
    /// sequence before native entry; no handler assumes native delivery order.
    /// Errors leave physical owners under registry custody, never on this stack.
    /// The native boundary must poison/wake on Err; ignoring it is not success.
    pub fn publish(
        &self,
        sequence: u64,
        record: Record,
        feedback: metal4::Feedback,
    ) -> Result<(), RetirementError> {
        record
            .publish_feedback_observed(feedback, |owners, feedback| {
                owners.publish_native_feedback(feedback);
            })
            .map_err(|error| {
                RetirementError::Mailbox(match error {
                    FeedbackPublicationError::Rejected(rejected) => rejected.error,
                    FeedbackPublicationError::ObserverPanicked => MailboxError::CallbackPanicked,
                })
            })?;
        let elected = self
            .frontier
            .lock()
            .publish(sequence, record)
            .map_err(|rejected| RetirementError::Frontier(rejected.reason))?;
        if !elected || self.failure.get().is_some() {
            return Ok(());
        }
        loop {
            // A record whose callbacks already started may finish safe logical
            // retirement after concurrent queue failure. Do not start a later
            // record: its captures remain rooted until explicit recovery.
            if self.failure.get().is_some() {
                return Ok(());
            }
            let next = self
                .frontier
                .lock()
                .pop_ready_or_release_drainer()
                .map_err(RetirementError::Frontier)?;
            let Some((sequence, record)) = next else {
                return Ok(());
            };
            record
                .run_callbacks(
                    |captures, owners, feedback| {
                        captures.run(self.queue, feedback);
                        self.frontier
                            .lock()
                            .finish(sequence)
                            .expect("elected prefix retirement");
                        owners.return_slot();
                    },
                    || self.wake.notify(),
                )
                .map_err(RetirementError::Mailbox)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn failure_during_callback_finishes_started_record_but_not_next() {
        // CPU-only composition: no native Device/Queue/CB, no shared worker.
        let service = crate::retirement_service::HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let wake = Arc::new(Wake::default());
        let registry = Registry::manual(failure.clone(), service.clone());
        let core = Metal4Completions::new(1, failure.clone(), wake.clone());
        let first_calls = Arc::new(AtomicUsize::new(0));
        let later_calls = Arc::new(AtomicUsize::new(0));
        let first = first_calls.clone();
        let signal = failure.clone();
        let notify = wake.clone();
        let record0 = registry
            .prepare(
                Captures(vec![Box::new(move |_| {
                    first.fetch_add(1, Ordering::SeqCst);
                    signal.publish(Failure::Closing, || notify.notify());
                })]),
                NativeOwners::vacant(),
                |_| Ok(()),
            )
            .unwrap_or_else(|_| panic!("first prepare"));
        let later = later_calls.clone();
        let record1 = registry
            .prepare(
                Captures(vec![Box::new(move |_| {
                    later.fetch_add(1, Ordering::SeqCst);
                })]),
                NativeOwners::vacant(),
                |_| Ok(()),
            )
            .unwrap_or_else(|_| panic!("second prepare"));
        for record in [&record0, &record1] {
            record.begin_handoff().unwrap();
            record.handoff_returned().unwrap();
        }
        let feedback = || metal4::Feedback {
            error: None,
            start: 0.,
            end: 1.,
        };
        core.publish(1, record1, feedback()).unwrap();
        core.publish(0, record0, feedback()).unwrap();
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(later_calls.load(Ordering::SeqCst), 0);
        assert_eq!(failure.get(), Some(Failure::Closing));
        // Started record reached cleanup despite failure inside its callback;
        // queued later record remains retained, never called or synthesized.
        assert!(service.manual_pass());
        assert!(!service.manual_pass());
    }
}
