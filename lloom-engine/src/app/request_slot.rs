//! Shared request checkout: no mutex across a callback, complete state restored
//! on every exit. Incomplete decoding is marked for recovery BEFORE the next
//! use, outside callback unwind. Existing CacheReset may release backing;
//! this is neither transactional rollback nor allocation admission.
use crate::{DecodeConfig, DecodeSession, RingPool};
use lloom::nn::{CacheReset, CausalLM, Sequence};
use lloom_tensor::{Device, Error, Result};
use parking_lot::Mutex;

pub(super) trait AbortState {
    fn abort(&mut self);
}
impl AbortState for () {
    fn abort(&mut self) {}
}

pub(super) struct State<C, E = ()> {
    pub sequence: Option<Sequence<C>>,
    pub rings: Option<RingPool>,
    pub extra: E,
    pub needs_recovery: bool,
}
enum LeaseState<'m, M: CausalLM, E> {
    Held(State<M::Cache, E>),
    Decoding(DecodeSession<'m, M>, E),
}
pub(super) struct Lease<'m, M: CausalLM, E: AbortState = ()>
where
    M::Cache: CacheReset,
{
    slot: &'m Mutex<Option<State<M::Cache, E>>>,
    state: Option<LeaseState<'m, M, E>>,
    recover: bool,
    complete: bool,
    started: bool,
    needs_recovery: bool,
}
impl<'m, M: CausalLM, E: AbortState> Lease<'m, M, E>
where
    M::Cache: CacheReset,
{
    pub fn checkout(slot: &'m Mutex<Option<State<M::Cache, E>>>, recover: bool) -> Result<Self> {
        let state = slot.lock().take().ok_or_else(|| {
            Error::msg(
                "causal owner is busy; queue or retry the request (callback reentry is refused)",
            )
        })?;
        let needs_recovery = state.needs_recovery;
        Ok(Self {
            slot,
            state: Some(LeaseState::Held(state)),
            recover,
            complete: false,
            started: false,
            needs_recovery,
        })
    }
    /// Native slot owners may need a fallible, recovery-only data fence BEFORE
    /// cache/extra reset. A refused fence leaves this marker and custody intact.
    pub fn needs_recovery(&self) -> bool {
        self.recover && self.needs_recovery
    }
    pub fn extra(&self) -> &E {
        match self.state.as_ref() {
            Some(LeaseState::Held(state)) => &state.extra,
            Some(LeaseState::Decoding(_, extra)) => extra,
            None => unreachable!("lease owns its request state"),
        }
    }
    pub fn extra_mut(&mut self) -> &mut E {
        match self.state.as_mut() {
            Some(LeaseState::Held(state)) => &mut state.extra,
            Some(LeaseState::Decoding(_, extra)) => extra,
            None => unreachable!("lease owns its request state"),
        }
    }
    /// Borrow the held sequence between same-cache stages; no unscoped owner
    /// or independent checkout is exposed.
    pub fn sequence(&self) -> &Sequence<M::Cache> {
        match self.state.as_ref() {
            Some(LeaseState::Held(state)) => state
                .sequence
                .as_ref()
                .expect("held request owns its sequence"),
            _ => unreachable!("same-cache stage did not restore held custody"),
        }
    }
    /// Run a shorter-lived model view (such as ImageRows) over the SAME cache
    /// without releasing the outer request checkout. All state is installed in
    /// the nested guard before recovery or callbacks; Drop only restores it.
    /// The caller marks the complete multi-stage request, not each stage.
    pub fn with_model<'n, N: CausalLM<Cache = M::Cache>, T, RunError>(
        &mut self,
        model: &'n N,
        device: &Device,
        decode: DecodeConfig,
        run: impl FnOnce(&mut DecodeSession<'n, N>, &mut E) -> std::result::Result<T, RunError>,
    ) -> std::result::Result<T, RunError> {
        let device = device.clone();
        assert!(
            matches!(self.state, Some(LeaseState::Held(_))),
            "same-cache stage starts from held custody"
        );
        let Some(LeaseState::Held(mut state)) = self.state.take() else {
            unreachable!("held stage checked before taking custody")
        };
        let sequence = state
            .sequence
            .take()
            .expect("held request owns its sequence");
        let rings = state.rings.take().unwrap_or_default();
        self.started = true;
        // A previously completed stage cannot certify a newly started one.
        self.complete = false;
        let mut stage = ModelStage {
            owner: self,
            session: Some(
                DecodeSession::from_parts(model, sequence, device, rings).with_config(decode),
            ),
            extra: Some(state.extra),
            complete: false,
        };
        if stage.owner.recover && stage.owner.needs_recovery {
            let (session, extra) = stage.parts_mut();
            session.reset();
            extra.abort();
            stage.owner.needs_recovery = false;
        }
        let (session, extra) = stage.parts_mut();
        let result = run(session, extra);
        stage.complete = result.is_ok();
        result
    }
    pub fn start_decode(&mut self, model: &'m M, device: &Device, decode: DecodeConfig) {
        // Clone before moving held custody; no backend handle cloning follows it.
        let device = device.clone();
        let Some(LeaseState::Held(mut state)) = self.state.take() else {
            unreachable!("request starts exactly one decode session")
        };
        let sequence = state
            .sequence
            .take()
            .expect("held request owns its sequence");
        let rings = state.rings.take().unwrap_or_default();
        self.started = true;
        self.complete = false;
        self.state = Some(LeaseState::Decoding(
            DecodeSession::from_parts(model, sequence, device, rings).with_config(decode),
            state.extra,
        ));
        // The complete state is already held by this guard if reset panics.
        // Never enter CacheReset/backend destruction from our Drop during unwind.
        if self.recover && self.needs_recovery {
            let (session, extra) = self.parts_mut();
            session.reset();
            extra.abort();
            self.needs_recovery = false;
        }
    }
    pub fn parts_mut(&mut self) -> (&mut DecodeSession<'m, M>, &mut E) {
        match self.state.as_mut() {
            Some(LeaseState::Decoding(session, extra)) => (session, extra),
            _ => unreachable!("decode session was not started"),
        }
    }
    pub fn session(&mut self) -> &mut DecodeSession<'m, M> {
        self.parts_mut().0
    }
    pub fn mark_complete(&mut self) {
        self.complete = true;
    }
}
impl<M: CausalLM, E: AbortState> Drop for Lease<'_, M, E>
where
    M::Cache: CacheReset,
{
    fn drop(&mut self) {
        let state = match self.state.take() {
            Some(LeaseState::Held(mut state)) => {
                state.needs_recovery |=
                    self.needs_recovery || (self.recover && self.started && !self.complete);
                state
            }
            Some(LeaseState::Decoding(session, extra)) => {
                let (sequence, rings) = session.into_parts();
                State {
                    sequence: Some(sequence),
                    rings: Some(rings),
                    extra,
                    needs_recovery: self.needs_recovery || (self.recover && !self.complete),
                }
            }
            None => return,
        };
        *self.slot.lock() = Some(state);
    }
}

/// A borrowed-model stage returns custody to its already-checked-out owner,
/// never directly to the public slot. A failed stage marks deferred recovery.
struct ModelStage<'a, 'm, 'n, M: CausalLM, N: CausalLM<Cache = M::Cache>, E: AbortState>
where
    M::Cache: CacheReset,
{
    owner: &'a mut Lease<'m, M, E>,
    session: Option<DecodeSession<'n, N>>,
    extra: Option<E>,
    complete: bool,
}
impl<'n, M: CausalLM, N: CausalLM<Cache = M::Cache>, E: AbortState> ModelStage<'_, '_, 'n, M, N, E>
where
    M::Cache: CacheReset,
{
    fn parts_mut(&mut self) -> (&mut DecodeSession<'n, N>, &mut E) {
        (
            self.session
                .as_mut()
                .expect("stage owns its decode session"),
            self.extra.as_mut().expect("stage owns its extra state"),
        )
    }
}
impl<M: CausalLM, N: CausalLM<Cache = M::Cache>, E: AbortState> Drop
    for ModelStage<'_, '_, '_, M, N, E>
where
    M::Cache: CacheReset,
{
    fn drop(&mut self) {
        let session = self.session.take().expect("stage owns its decode session");
        let extra = self.extra.take().expect("stage owns its extra state");
        let (sequence, rings) = session.into_parts();
        self.owner.needs_recovery |= self.owner.recover && !self.complete;
        self.owner.state = Some(LeaseState::Held(State {
            sequence: Some(sequence),
            rings: Some(rings),
            extra,
            needs_recovery: self.owner.needs_recovery,
        }));
    }
}
