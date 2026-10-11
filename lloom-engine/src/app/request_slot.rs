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
            needs_recovery,
        })
    }
    pub fn extra(&self) -> &E {
        match self.state.as_ref() {
            Some(LeaseState::Held(state)) => &state.extra,
            Some(LeaseState::Decoding(_, extra)) => extra,
            None => unreachable!("lease owns its request state"),
        }
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
            Some(LeaseState::Held(state)) => state,
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
