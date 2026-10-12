// Private fixed-worker custody. Every constructor requires an explicit owning
// fallback; no public server route or production fallback is registered here.
use super::{
    accept_managed_once, AcceptAttempt, AcceptSetupFailure, AcceptedSetup, ClosedOriginSlot,
    OriginInventory,
};
use crate::client::ClientConnection;
use crate::connection::Listener;
use crate::{ConnectionCancellation, Request};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::io;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) struct ManagedClient {
    pending: Option<Request>,
    client: ClientConnection,
    control: ConnectionCancellation,
    #[cfg(test)]
    panic_after_read: bool,
    #[cfg(test)]
    pause: Option<Arc<TestPause>>,
}
struct ManagedRequests {
    state: Mutex<RequestState>,
    changed: Condvar,
    limit: NonZeroUsize,
}
struct RequestState {
    closed: bool,
    queue: VecDeque<Request>,
}
enum PublishStep {
    Published,
    Full,
    Closed,
}
impl ManagedRequests {
    fn new(limit: NonZeroUsize) -> io::Result<Arc<Self>> {
        let mut queue = VecDeque::new();
        queue
            .try_reserve_exact(limit.get())
            .map_err(io::Error::other)?;
        Ok(Arc::new(Self {
            state: Mutex::new(RequestState {
                closed: false,
                queue,
            }),
            changed: Condvar::new(),
            limit,
        }))
    }
    // Request stays in worker-owned storage until every refusal check is past.
    fn publish_step(&self, pending: &mut Option<Request>) -> PublishStep {
        let mut state = self.state.lock();
        if state.closed {
            return PublishStep::Closed;
        }
        if state.queue.len() == self.limit.get() {
            return PublishStep::Full;
        }
        assert!(state.queue.capacity() >= self.limit.get());
        state
            .queue
            .push_back(pending.take().expect("worker retains request"));
        self.changed.notify_all();
        PublishStep::Published
    }
    fn publish_until(&self, worker: &mut ManagedClient, deadline: Instant) -> io::Result<()> {
        loop {
            let remaining = worker.control.remaining(deadline)?;
            match self.publish_step(&mut worker.pending) {
                PublishStep::Published => return Ok(()),
                PublishStep::Closed => {
                    return Err(io::Error::new(
                        io::ErrorKind::ConnectionAborted,
                        "request intake closed",
                    ));
                }
                PublishStep::Full => (),
            }
            let mut state = self.state.lock();
            if !state.closed && state.queue.len() == self.limit.get() {
                self.changed
                    .wait_for(&mut state, remaining.min(Duration::from_millis(10)));
            }
        }
    }
    fn pop(&self) -> Option<Request> {
        let mut state = self.state.lock();
        loop {
            if state.closed {
                return None;
            }
            if let Some(request) = state.queue.pop_front() {
                self.changed.notify_all();
                return Some(request);
            }
            self.changed.wait(&mut state);
        }
    }
    // Returned queued custody remains rooted until runtime cancellation policy.
    fn close_and_take(&self) -> VecDeque<Request> {
        let mut state = self.state.lock();
        state.closed = true;
        let queued = std::mem::take(&mut state.queue);
        self.changed.notify_all();
        queued
    }
}

enum WorkerStop {
    ReadyOnClose,
    ParserEnded,
    Refused(io::Error),
    Panicked(Box<dyn std::any::Any + Send>),
}
struct WorkerCustody {
    worker: ManagedClient,
    outcome: WorkerStop,
}
impl ManagedClient {
    fn pump(&mut self, requests: &ManagedRequests) -> io::Result<()> {
        loop {
            self.control.check()?;
            if self.pending.is_none() {
                self.pending = self.client.next();
                if self.pending.is_none() {
                    return Ok(());
                }
            }
            #[cfg(test)]
            if let Some(pause) = &self.pause {
                pause.stop_once();
            }
            #[cfg(test)]
            if self.panic_after_read {
                panic!("worker pending-request fixture panic");
            }
            let deadline = Instant::now()
                .checked_add(self.control.policy().io_timeout)
                .expect("validated origin timeout");
            requests.publish_until(self, deadline)?;
        }
    }
    // Caller retains this output inside its original worker handle/join outcome.
    // A refusal never causes implicit Request/ClientConnection destruction here.
    fn run(mut self, requests: &ManagedRequests) -> WorkerCustody {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.pump(requests)));
        let outcome = match result {
            Ok(Ok(())) => WorkerStop::ParserEnded,
            Ok(Err(error)) => WorkerStop::Refused(error),
            Err(payload) => WorkerStop::Panicked(payload),
        };
        WorkerCustody {
            worker: self,
            outcome,
        }
    }
}

// One acknowledged result slot per fixed worker keeps original job custody
// until host release, independently of function/native thread completion.
struct WorkerMailbox {
    state: parking_lot::Mutex<WorkerMailboxState>,
    changed: parking_lot::Condvar,
}
struct WorkerMailboxState {
    closed: bool,
    epoch: u64,
    slot: WorkerMailboxSlot,
}
enum WorkerMailboxSlot {
    Vacant,
    Ready(ManagedClient),
    Running,
    Returned(WorkerCustody),
    Releasing,
}
struct WorkerRelease {
    mailbox: std::sync::Arc<WorkerMailbox>,
    epoch: u64,
    custody: Option<WorkerCustody>,
}
impl WorkerRelease {
    fn try_finish(mut self) -> Result<WorkerStop, (io::Error, Self)> {
        if !matches!(
            self.custody.as_ref().expect("original custody").outcome,
            WorkerStop::ParserEnded
        ) {
            if let Err(error) = self
                .custody
                .as_ref()
                .expect("original custody")
                .worker
                .control
                .cancel()
            {
                return Err((error, self));
            }
        }
        let WorkerCustody { worker, outcome } = self.custody.take().expect("original custody");
        drop(worker);
        let mut state = self.mailbox.state.lock();
        assert_eq!(state.epoch, self.epoch);
        assert!(matches!(state.slot, WorkerMailboxSlot::Releasing));
        state.slot = WorkerMailboxSlot::Vacant;
        self.mailbox.changed.notify_all();
        Ok(outcome)
    }
}
impl Drop for WorkerRelease {
    fn drop(&mut self) {
        if let Some(custody) = self.custody.take() {
            let mut state = self.mailbox.state.lock();
            assert_eq!(state.epoch, self.epoch);
            assert!(matches!(state.slot, WorkerMailboxSlot::Releasing));
            state.slot = WorkerMailboxSlot::Returned(custody);
            self.mailbox.changed.notify_all();
        }
    }
}
impl WorkerMailbox {
    fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: parking_lot::Mutex::new(WorkerMailboxState {
                closed: false,
                epoch: 0,
                slot: WorkerMailboxSlot::Vacant,
            }),
            changed: parking_lot::Condvar::new(),
        })
    }
    fn try_submit(&self, pending: &mut Option<ManagedClient>) -> std::io::Result<bool> {
        let mut state = self.state.lock();
        if state.closed {
            drop(state);
            return Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "worker intake closed",
            ));
        }
        if !matches!(state.slot, WorkerMailboxSlot::Vacant) {
            return Ok(false);
        }
        let Some(epoch) = state.epoch.checked_add(1) else {
            drop(state);
            return Err(std::io::Error::other("worker epoch exhausted"));
        };
        state.epoch = epoch;
        state.slot = WorkerMailboxSlot::Ready(pending.take().expect("accepted client retained"));
        self.changed.notify_all();
        Ok(true)
    }
    fn take_ready(&self) -> Option<(u64, ManagedClient)> {
        let mut state = self.state.lock();
        loop {
            // Closing preserves Ready custody instead of beginning another job.
            if state.closed {
                return None;
            }
            if matches!(state.slot, WorkerMailboxSlot::Ready(_)) {
                let WorkerMailboxSlot::Ready(client) =
                    std::mem::replace(&mut state.slot, WorkerMailboxSlot::Running)
                else {
                    unreachable!("ready worker slot");
                };
                return Some((state.epoch, client));
            }
            self.changed.wait(&mut state);
        }
    }
    fn return_original(&self, epoch: u64, custody: WorkerCustody) {
        let mut state = self.state.lock();
        assert_eq!(state.epoch, epoch);
        assert!(matches!(state.slot, WorkerMailboxSlot::Running));
        state.slot = WorkerMailboxSlot::Returned(custody);
        self.changed.notify_all();
    }
    fn run(&self, requests: &ManagedRequests) {
        while let Some((epoch, worker)) = self.take_ready() {
            let custody = worker.run(requests);
            self.return_original(epoch, custody);
        }
    }
    fn close_intake(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        self.changed.notify_all();
    }
    fn close_ready_into_returned(&self) {
        let mut state = self.state.lock();
        assert!(state.closed);
        if !matches!(state.slot, WorkerMailboxSlot::Ready(_)) {
            return;
        }
        let WorkerMailboxSlot::Ready(worker) =
            std::mem::replace(&mut state.slot, WorkerMailboxSlot::Running)
        else {
            unreachable!("closed ready custody");
        };
        let custody = WorkerCustody {
            worker,
            outcome: WorkerStop::ReadyOnClose,
        };
        state.slot = WorkerMailboxSlot::Returned(custody);
    }
    fn take_returned(self: &std::sync::Arc<Self>) -> Option<WorkerRelease> {
        let mut state = self.state.lock();
        if !matches!(state.slot, WorkerMailboxSlot::Returned(_)) {
            return None;
        }
        let WorkerMailboxSlot::Returned(custody) =
            std::mem::replace(&mut state.slot, WorkerMailboxSlot::Releasing)
        else {
            unreachable!("returned worker slot");
        };
        Some(WorkerRelease {
            mailbox: std::sync::Arc::clone(self),
            epoch: state.epoch,
            custody: Some(custody),
        })
    }
}

// Production implementations must retain every part on refusal/unwind and own
// physical joins; the only implementation below is explicit cfg(test) blocking.
// No production default is installed and this is not a bounded join certificate.
trait WorkerFallback: Send + Sync {
    fn retain(&self, retirement: WorkerRetirement);
}
struct WorkerEntry {
    mailbox: Arc<WorkerMailbox>,
    handle: Option<std::thread::JoinHandle<()>>,
    native_outcome: Option<std::thread::Result<()>>,
    host_outcome: Option<WorkerStop>,
}
struct WorkerParts {
    entries: Vec<WorkerEntry>,
    requests: Arc<ManagedRequests>,
    origins: Arc<OriginInventory>,
    held_requests: VecDeque<Request>,
    pending_accepted: Option<ManagedClient>,
    retained_setup: Option<AcceptedSetup>,
}
struct WorkerRetirement {
    parts: WorkerParts,
}
struct WorkerOwner {
    parts: Option<WorkerParts>,
    fallback: Arc<dyn WorkerFallback>,
}
struct WorkerStartFailure {
    error: io::Error,
    owner: WorkerOwner,
}
impl WorkerOwner {
    fn start(
        workers: NonZeroUsize,
        requests: Arc<ManagedRequests>,
        origins: Arc<OriginInventory>,
        fallback: Arc<dyn WorkerFallback>,
    ) -> Result<Self, WorkerStartFailure> {
        Self::start_inner(workers, requests, origins, fallback, None)
    }
    fn start_inner(
        workers: NonZeroUsize,
        requests: Arc<ManagedRequests>,
        origins: Arc<OriginInventory>,
        fallback: Arc<dyn WorkerFallback>,
        fail_at: Option<usize>,
    ) -> Result<Self, WorkerStartFailure> {
        let mut entries = Vec::new();
        let mut owner = Self {
            parts: Some(WorkerParts {
                entries: Vec::new(),
                requests: requests.clone(),
                origins,
                held_requests: VecDeque::new(),
                pending_accepted: None,
                retained_setup: None,
            }),
            fallback,
        };
        let origin_limit = owner
            .parts
            .as_ref()
            .expect("original parts")
            .origins
            .state
            .lock()
            .slots
            .len();
        if workers.get() > origin_limit {
            return Err(WorkerStartFailure {
                error: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "worker count exceeds original origin capacity",
                ),
                owner,
            });
        }
        if let Err(error) = entries.try_reserve_exact(workers.get()) {
            return Err(WorkerStartFailure {
                error: io::Error::other(error),
                owner,
            });
        }
        for _ in 0..workers.get() {
            entries.push(WorkerEntry {
                mailbox: WorkerMailbox::new(),
                handle: None,
                native_outcome: None,
                host_outcome: None,
            });
        }
        owner.parts.as_mut().expect("original parts").entries = entries;
        for index in 0..workers.get() {
            let parts = owner.parts.as_mut().expect("original parts");
            let mailbox = parts.entries[index].mailbox.clone();
            let requests = parts.requests.clone();
            let result = if fail_at == Some(index) {
                Err(io::Error::other("controlled worker startup refusal"))
            } else {
                std::thread::Builder::new().spawn(move || mailbox.run(&requests))
            };
            match result {
                Ok(handle) => parts.entries[index].handle = Some(handle),
                Err(error) => return Err(WorkerStartFailure { error, owner }),
            }
        }
        Ok(owner)
    }
    fn try_submit(&self, pending: &mut Option<ManagedClient>) -> io::Result<bool> {
        for entry in &self.parts.as_ref().expect("original parts").entries {
            if entry.mailbox.try_submit(pending)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn pump_one(&self, index: usize) -> io::Result<Option<WorkerStop>> {
        let parts = self.parts.as_ref().expect("original parts");
        let entry = &parts.entries[index];
        let Some(release) = entry.mailbox.take_returned() else {
            return Ok(None);
        };
        match release.try_finish() {
            Ok(outcome) => Ok(Some(outcome)),
            Err((error, release)) => {
                drop(release);
                Err(error)
            }
        }
    }
    fn close(&mut self) {
        let parts = self.parts.as_mut().expect("original parts");
        parts.origins.close_intake();
        for entry in &parts.entries {
            entry.mailbox.close_intake();
        }
        let already_closed = parts.requests.state.lock().closed;
        if !already_closed {
            parts.held_requests = parts.requests.close_and_take();
        }
        for entry in &parts.entries {
            entry.mailbox.close_ready_into_returned();
        }
    }
    fn cancel_origins(&self) -> io::Result<()> {
        let parts = self.parts.as_ref().expect("original parts");
        let count = parts.origins.state.lock().slots.len();
        let mut first = None;
        for index in 0..count {
            if let Some(ClosedOriginSlot::Live(control)) = parts.origins.slot_after_close(index) {
                if let Err(error) = control.cancel() {
                    if first.is_none() {
                        first = Some(error);
                    }
                }
            }
        }
        first.map_or(Ok(()), Err)
    }
    fn poll_functions(&self) -> bool {
        self.parts
            .as_ref()
            .expect("original parts")
            .entries
            .iter()
            .all(|entry| {
                entry
                    .handle
                    .as_ref()
                    .is_none_or(std::thread::JoinHandle::is_finished)
            })
    }
    // This operation may block through native/TLS termination after is_finished.
    fn join_functions(&mut self) {
        for entry in &mut self.parts.as_mut().expect("original parts").entries {
            if let Some(handle) = entry.handle.take() {
                entry.native_outcome = Some(handle.join());
            }
        }
    }
}
impl Drop for WorkerOwner {
    fn drop(&mut self) {
        if let Some(parts) = self.parts.take() {
            self.fallback.retain(WorkerRetirement { parts });
        }
    }
}
impl ManagedClient {
    pub(super) fn from_registered(
        client: ClientConnection,
        control: ConnectionCancellation,
    ) -> Self {
        Self {
            pending: None,
            client,
            control,
            #[cfg(test)]
            panic_after_read: false,
            #[cfg(test)]
            pause: None,
        }
    }
}
#[cfg(test)]
struct BlockingTestFallback {
    calls: std::sync::atomic::AtomicUsize,
    joined: std::sync::atomic::AtomicUsize,
}
#[cfg(test)]
impl WorkerFallback for BlockingTestFallback {
    fn retain(&self, retirement: WorkerRetirement) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut owner = WorkerOwner {
            parts: Some(retirement.parts),
            fallback: Arc::new(Self {
                calls: 0.into(),
                joined: 0.into(),
            }),
        };
        owner.close();
        while owner
            .parts
            .as_ref()
            .expect("original parts")
            .retained_setup
            .as_ref()
            .is_some_and(|setup| setup.cancel_retained().is_err())
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        while owner.cancel_origins().is_err() {
            std::thread::sleep(Duration::from_millis(1));
        }
        owner.join_functions();
        let mut parts = owner.parts.take().expect("same retired parts");
        for entry in &parts.entries {
            while let Some(release) = entry.mailbox.take_returned() {
                match release.try_finish() {
                    Ok(_) => (),
                    Err((_, release)) => {
                        drop(release);
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            }
        }
        self.joined.fetch_add(
            parts
                .entries
                .iter()
                .filter(|entry| entry.native_outcome.is_some())
                .count(),
            std::sync::atomic::Ordering::Relaxed,
        );
        parts.held_requests.clear();
        drop(parts);
    }
}

#[cfg(test)]
struct TestPause {
    state: Mutex<(bool, bool)>,
    changed: Condvar,
}
#[cfg(test)]
impl TestPause {
    fn stop_once(&self) {
        let mut state = self.state.lock();
        state.0 = true;
        self.changed.notify_all();
        let deadline = Instant::now() + Duration::from_millis(500);
        while !state.1 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "bounded worker fixture pause expired");
            self.changed.wait_for(&mut state, remaining);
        }
    }
    fn await_entered(&self) {
        let mut state = self.state.lock();
        let deadline = Instant::now() + Duration::from_millis(500);
        while !state.0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "actual worker must reach retained pending request"
            );
            self.changed.wait_for(&mut state, remaining);
        }
    }
    fn release(&self) {
        self.state.lock().1 = true;
        self.changed.notify_all();
    }
}
#[cfg(test)]
mod managed_worker_controls {
    use super::*;
    use crate::connection::{Connection, ConnectionProbe};
    use crate::util::RefinedTcpStream;
    use crate::{Response, TransportPolicy};
    use std::sync::atomic::Ordering;

    fn policy() -> TransportPolicy {
        TransportPolicy {
            io_timeout: Duration::from_millis(100),
            disposal_timeout: Duration::from_millis(50),
            max_discard_bytes: 4,
        }
    }
    fn registered(
        origins: &Arc<OriginInventory>,
        path: &str,
    ) -> (ManagedClient, Arc<ConnectionProbe>) {
        let probe = Arc::new(ConnectionProbe::default());
        probe.state.lock().input = std::io::Cursor::new(
            format!("GET {path} HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n").into_bytes(),
        );
        let socket = Connection::Probe(probe.clone());
        let control = ConnectionCancellation::new(&socket, policy()).unwrap();
        origins.reserve().unwrap().attach(&control).unwrap();
        let (reader, writer) =
            RefinedTcpStream::try_new_controlled(socket, control.clone()).unwrap();
        (
            ManagedClient::from_registered(
                ClientConnection::new(writer, reader, Some(control.clone())),
                control,
            ),
            probe,
        )
    }
    fn fixture(workers: usize, origins: usize) -> (WorkerOwner, Arc<BlockingTestFallback>) {
        let requests = ManagedRequests::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let origins = OriginInventory::new(NonZeroUsize::new(origins).unwrap()).unwrap();
        let fallback = Arc::new(BlockingTestFallback {
            calls: 0.into(),
            joined: 0.into(),
        });
        let owner = match WorkerOwner::start(
            NonZeroUsize::new(workers).unwrap(),
            requests,
            origins,
            fallback.clone(),
        ) {
            Ok(owner) => owner,
            Err(_) => panic!("bounded fixture startup"),
        };
        (owner, fallback)
    }
    fn await_returned(mailbox: &WorkerMailbox) {
        let mut state = mailbox.state.lock();
        let deadline = Instant::now() + Duration::from_millis(500);
        while !matches!(state.slot, WorkerMailboxSlot::Returned(_)) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "original worker must return retained custody"
            );
            mailbox.changed.wait_for(&mut state, remaining);
        }
    }
    #[test]
    fn worker_origin_limit_refuses_before_any_native_spawn() {
        let requests = ManagedRequests::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let origins = OriginInventory::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let fallback = Arc::new(BlockingTestFallback {
            calls: 0.into(),
            joined: 0.into(),
        });
        let failure = match WorkerOwner::start(
            NonZeroUsize::new(2).unwrap(),
            requests.clone(),
            origins.clone(),
            fallback.clone(),
        ) {
            Err(failure) => failure,
            Ok(_) => panic!("worker/origin limit must refuse before native spawn"),
        };
        assert_eq!(failure.error.kind(), io::ErrorKind::InvalidInput);
        let parts = failure.owner.parts.as_ref().unwrap();
        assert!(
            parts.entries.is_empty(),
            "no native worker entry may be born"
        );
        assert!(Arc::ptr_eq(&parts.requests, &requests));
        assert!(Arc::ptr_eq(&parts.origins, &origins));
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 0);
        drop(failure);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn ready_close_extraction_retains_on_cancel_refusal() {
        let origins = OriginInventory::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let (client, probe) = registered(&origins, "/ready");
        let original = client.control.origin.clone();
        let mailbox = WorkerMailbox::new();
        let mut pending = Some(client);
        assert!(mailbox.try_submit(&mut pending).unwrap());
        mailbox.close_intake();
        assert!(mailbox.take_ready().is_none());
        mailbox.close_ready_into_returned();
        probe.state.lock().refuse_shutdown = true;
        let release = mailbox.take_returned().unwrap();
        let (error, release) = match release.try_finish() {
            Err(value) => value,
            Ok(_) => panic!("cancel refusal must retain original owners"),
        };
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(Arc::ptr_eq(
            &release.custody.as_ref().unwrap().worker.control.origin,
            &original
        ));
        assert_eq!(release.epoch, 1);
        drop(release);
        assert!(matches!(
            mailbox.state.lock().slot,
            WorkerMailboxSlot::Returned(_)
        ));
        probe.state.lock().refuse_shutdown = false;
        assert!(matches!(
            mailbox.take_returned().unwrap().try_finish(),
            Ok(WorkerStop::ReadyOnClose)
        ));
        assert!(matches!(
            mailbox.state.lock().slot,
            WorkerMailboxSlot::Vacant
        ));
    }
    #[test]
    fn running_return_after_close_preserves_same_pending_request() {
        let (mut owner, fallback) = fixture(1, 1);
        let origins = owner.parts.as_ref().unwrap().origins.clone();
        let (mut client, _) = registered(&origins, "/running");
        let original = client.control.origin.clone();
        let pause = Arc::new(TestPause {
            state: Mutex::new((false, false)),
            changed: Condvar::new(),
        });
        client.pause = Some(pause.clone());
        let mailbox = owner.parts.as_ref().unwrap().entries[0].mailbox.clone();
        assert!(owner.try_submit(&mut Some(client)).unwrap());
        pause.await_entered();
        owner.close();
        pause.release();
        await_returned(&mailbox);
        {
            let state = mailbox.state.lock();
            let WorkerMailboxSlot::Returned(custody) = &state.slot else {
                panic!("same running return");
            };
            assert!(matches!(custody.outcome, WorkerStop::Refused(_)));
            assert_eq!(custody.worker.pending.as_ref().unwrap().url(), "/running");
            assert!(Arc::ptr_eq(&custody.worker.control.origin, &original));
            assert_eq!(state.epoch, 1);
        }
        assert!(matches!(
            owner.pump_one(0).unwrap(),
            Some(WorkerStop::Refused(_))
        ));
        drop(owner);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn returned_ack_reuses_original_worker_and_preserves_successful_request() {
        let (owner, fallback) = fixture(1, 2);
        let parts = owner.parts.as_ref().unwrap();
        let thread = parts.entries[0].handle.as_ref().unwrap().thread().id();
        let mailbox = parts.entries[0].mailbox.clone();
        for (epoch, path) in [(1, "/one"), (2, "/two")] {
            let (client, probe) = registered(&parts.origins, path);
            let control = client.control.clone();
            assert!(owner.try_submit(&mut Some(client)).unwrap());
            await_returned(&mailbox);
            let request = parts.requests.pop().unwrap();
            assert_eq!(request.url(), path);
            assert!(matches!(
                owner.pump_one(0).unwrap(),
                Some(WorkerStop::ParserEnded)
            ));
            control.check().unwrap();
            request.respond(Response::from_string("ok")).unwrap();
            assert!(probe.state.lock().writes > 0);
            assert_eq!(mailbox.state.lock().epoch, epoch);
            assert_eq!(
                parts.entries[0].handle.as_ref().unwrap().thread().id(),
                thread
            );
        }
        drop(owner);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn partial_spawn_failure_retains_and_joins_original_handle() {
        let requests = ManagedRequests::new(NonZeroUsize::new(1).unwrap()).unwrap();
        let origins = OriginInventory::new(NonZeroUsize::new(2).unwrap()).unwrap();
        let fallback = Arc::new(BlockingTestFallback {
            calls: 0.into(),
            joined: 0.into(),
        });
        let failure = match WorkerOwner::start_inner(
            NonZeroUsize::new(2).unwrap(),
            requests,
            origins,
            fallback.clone(),
            Some(1),
        ) {
            Err(error) => error,
            Ok(_) => panic!("controlled second spawn must refuse"),
        };
        assert_eq!(failure.error.kind(), io::ErrorKind::Other);
        assert!(failure.owner.parts.as_ref().unwrap().entries[0]
            .handle
            .is_some());
        assert!(failure.owner.parts.as_ref().unwrap().entries[1]
            .handle
            .is_none());
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 0);
        drop(failure);
        assert_eq!(fallback.calls.load(Ordering::Relaxed), 1);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn parser_worker_panic_retains_pending_across_cancel_refusal() {
        let (owner, fallback) = fixture(1, 1);
        let parts = owner.parts.as_ref().unwrap();
        let (mut client, probe) = registered(&parts.origins, "/panic");
        client.panic_after_read = true;
        let original = client.control.origin.clone();
        let mailbox = parts.entries[0].mailbox.clone();
        assert!(owner.try_submit(&mut Some(client)).unwrap());
        await_returned(&mailbox);
        probe.state.lock().refuse_shutdown = true;
        let error = match owner.pump_one(0) {
            Err(error) => error,
            Ok(_) => panic!("cancel refusal must retain original custody"),
        };
        assert_eq!(error.kind(), io::ErrorKind::Other);
        {
            let state = mailbox.state.lock();
            let WorkerMailboxSlot::Returned(custody) = &state.slot else {
                panic!("cancel refusal must preserve return");
            };
            let WorkerStop::Panicked(payload) = &custody.outcome else {
                panic!("original panic payload retained");
            };
            assert_eq!(
                payload.downcast_ref::<&str>(),
                Some(&"worker pending-request fixture panic")
            );
            assert_eq!(custody.worker.pending.as_ref().unwrap().url(), "/panic");
            assert!(Arc::ptr_eq(&custody.worker.control.origin, &original));
        }
        probe.state.lock().refuse_shutdown = false;
        assert!(matches!(
            owner.pump_one(0).unwrap(),
            Some(WorkerStop::Panicked(_))
        ));
        drop(owner);
        assert_eq!(fallback.joined.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn startup_caller_unwind_uses_same_native_join_guard() {
        let fallback = Arc::new(BlockingTestFallback {
            calls: 0.into(),
            joined: 0.into(),
        });
        let observed = fallback.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let requests = ManagedRequests::new(NonZeroUsize::new(1).unwrap()).unwrap();
            let origins = OriginInventory::new(NonZeroUsize::new(2).unwrap()).unwrap();
            let _owner = match WorkerOwner::start(
                NonZeroUsize::new(2).unwrap(),
                requests,
                origins,
                fallback,
            ) {
                Ok(owner) => owner,
                Err(_) => panic!("bounded startup"),
            };
            panic!("startup caller fixture unwind");
        }));
        let payload = result.unwrap_err();
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"startup caller fixture unwind")
        );
        assert_eq!(observed.calls.load(Ordering::Relaxed), 1);
        assert_eq!(observed.joined.load(Ordering::Relaxed), 2);
    }
}

// Private real-caller draft: existing accept/parser path with explicit fallback.
// No public Server constructor/default is redirected to this owner.
struct ManagedIntake {
    listener: Listener,
    workers: WorkerOwner,
    policy: crate::TransportPolicy,
}
enum IntakeStep {
    Idle,
    Backpressure,
    Dispatched,
}
impl ManagedIntake {
    fn from_started(
        listener: Listener,
        workers: WorkerOwner,
        policy: crate::TransportPolicy,
    ) -> io::Result<Self> {
        let policy = policy.validate()?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            workers,
            policy,
        })
    }
    fn take_worker_outcome(&mut self, index: usize) -> Option<WorkerStop> {
        self.workers
            .parts
            .as_mut()
            .expect("original parts")
            .entries
            .get_mut(index)?
            .host_outcome
            .take()
    }
    fn step(&mut self) -> io::Result<IntakeStep> {
        let count = self
            .workers
            .parts
            .as_ref()
            .expect("original parts")
            .entries
            .len();
        for index in 0..count {
            if self.workers.parts.as_ref().expect("original parts").entries[index]
                .host_outcome
                .is_none()
            {
                if let Some(outcome) = self.workers.pump_one(index)? {
                    self.workers.parts.as_mut().expect("original parts").entries[index]
                        .host_outcome = Some(outcome);
                }
            }
        }
        let parts = self.workers.parts.as_mut().expect("original parts");
        if let Some(setup) = &parts.retained_setup {
            setup.cancel_retained()?;
            drop(parts.retained_setup.take());
            return Ok(IntakeStep::Idle);
        }
        if parts.pending_accepted.is_none() {
            match accept_managed_once(&self.listener, &parts.origins, self.policy) {
                Ok(AcceptAttempt::Idle) => return Ok(IntakeStep::Idle),
                Ok(AcceptAttempt::Ready(client)) => parts.pending_accepted = Some(client),
                Err(AcceptSetupFailure::BeforeAccept(error))
                    if error.kind() == io::ErrorKind::WouldBlock =>
                {
                    return Ok(IntakeStep::Backpressure)
                }
                Err(AcceptSetupFailure::BeforeAccept(error)) => return Err(error),
                Err(AcceptSetupFailure::Retained { error, setup }) => {
                    parts.retained_setup = Some(setup);
                    return Err(error);
                }
            }
        }
        for entry in &parts.entries {
            if entry.mailbox.try_submit(&mut parts.pending_accepted)? {
                return Ok(IntakeStep::Dispatched);
            }
        }
        Ok(IntakeStep::Backpressure)
    }
}
