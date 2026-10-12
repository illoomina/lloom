//! Shared host planner lifecycle. Native allocation/locking stays at call sites.
//! Current remains occupied while Closing; pending survives an aborted close.
use std::sync::Arc;

pub(crate) struct PlannerState<S, K> {
    pub identity: Arc<()>,
    pub closing: Option<Arc<()>>,
    pub current: Option<S>,
    pub pending: Option<K>,
    pub base_generation: u64,
}
impl<S, K> PlannerState<S, K> {
    pub fn new() -> Self {
        Self {
            identity: Arc::new(()),
            closing: None,
            current: None,
            pending: None,
            base_generation: 0,
        }
    }
    pub fn owns_close(&self, cache: &Arc<()>, token: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.identity, cache)
            && self.closing.as_ref().is_some_and(|t| Arc::ptr_eq(t, token))
    }
    /// Never erase a successor; historical pending key is intentionally retained.
    pub fn abort_close(&mut self, cache: &Arc<()>, token: &Arc<()>) -> bool {
        if !self.owns_close(cache, token) {
            return false;
        }
        self.closing = None;
        self.current = None;
        true
    }
}

pub(crate) struct ClosingIdentity<N, R> {
    pub cache: Arc<()>,
    pub token: Arc<()>,
    pub namespace: Arc<N>,
    pub session: Arc<()>,
    pub base_generation: u64,
    pub recorder: Arc<R>,
}
impl<N, R> ClosingIdentity<N, R> {
    /// Caller holds the actual planner/session/recorder guards across this check
    /// and publication. No callback or native resource destruction occurs here.
    pub fn validates<S, K>(
        &self,
        state: Option<&PlannerState<S, K>>,
        namespace: &Arc<N>,
        session: &Arc<()>,
        recorder: Option<&Arc<R>>,
    ) -> bool {
        state.is_some_and(|s| {
            s.owns_close(&self.cache, &self.token) && s.base_generation == self.base_generation
        }) && Arc::ptr_eq(namespace, &self.namespace)
            && Arc::ptr_eq(session, &self.session)
            && recorder.is_some_and(|r| Arc::ptr_eq(r, &self.recorder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_admission::{stage, Prediction};
    use crate::memory_admission::{process_account, AccountSnapshot, ModelDomain};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Raw {
        domain: ModelDomain,
        committed: Arc<AtomicBool>,
    }
    impl Drop for Raw {
        fn drop(&mut self) {
            let committed = self.committed.load(Ordering::Relaxed);
            assert_eq!(
                self.domain.snapshot(),
                AccountSnapshot {
                    pending: if committed { 0 } else { 8 },
                    committed: if committed { 8 } else { 0 },
                },
                "raw must precede ticket restoration or charge release"
            );
        }
    }

    fn invalidation(after_commit: bool) {
        // Six independent invalidations; only 8 numeric bytes promised at once.
        for change in 0..6 {
            let domain = process_account().unadmitted_domain(Some(8)).unwrap();
            let pass = domain.reserve(8).unwrap();
            let context = pass.context();
            let mut state = PlannerState::<u8, u8>::new();
            state.pending = Some(7);
            state.current = Some(9);
            let mut namespace = Arc::new(());
            let mut session = Arc::new(());
            let mut recorder = Some(Arc::new(()));
            let token = Arc::new(());
            state.closing = Some(token.clone());
            let identity = ClosingIdentity {
                cache: state.identity.clone(),
                token,
                namespace: namespace.clone(),
                session: session.clone(),
                base_generation: 0,
                recorder: recorder.as_ref().unwrap().clone(),
            };
            let committed = Arc::new(AtomicBool::new(false));
            let staged = stage(
                &context,
                Prediction::heap(8),
                |_| {
                    Some(Raw {
                        domain: domain.clone(),
                        committed: committed.clone(),
                    })
                },
                |_| 8,
            )
            .unwrap();
            assert!(identity.validates(Some(&state), &namespace, &session, recorder.as_ref()));
            let mut staged = Some(staged);
            let mut selected = None;
            if after_commit {
                selected = Some(staged.take().unwrap().commit().unwrap());
                committed.store(true, Ordering::Relaxed);
            }
            match change {
                0 => namespace = Arc::new(()),
                1 => session = Arc::new(()),
                2 => state.base_generation += 1,
                3 => recorder = Some(Arc::new(())),
                4 => {
                    state = PlannerState::new();
                    state.pending = Some(88);
                    state.current = Some(99);
                    state.closing = Some(Arc::new(()));
                }
                _ => state.closing = Some(Arc::new(())),
            }
            let recorder_after_change = recorder.as_ref().unwrap().clone();
            assert!(!identity.validates(Some(&state), &namespace, &session, recorder.as_ref()));
            let ours = state.abort_close(&identity.cache, &identity.token);
            assert_eq!(ours, change < 4);
            assert_eq!(state.pending, Some(if change == 4 { 88 } else { 7 }));
            assert_eq!(
                state.current,
                if change < 4 {
                    None
                } else if change == 4 {
                    Some(99)
                } else {
                    Some(9)
                }
            );
            assert_eq!(state.closing.is_none(), change < 4);
            assert!(
                Arc::ptr_eq(recorder.as_ref().unwrap(), &recorder_after_change),
                "abort must not detach a replacement recorder"
            );
            drop(staged); // unselected: raw before ticket, full promise returns
            drop(selected); // selected: raw before charge, no promise resurrection
            if after_commit {
                assert!(context.take(1).is_err());
                assert_eq!(domain.snapshot(), AccountSnapshot::default());
            } else {
                let returned = context.take(8).unwrap();
                drop(returned);
                assert_eq!(domain.snapshot().pending, 8);
            }
            drop(pass);
            assert_eq!(domain.snapshot(), AccountSnapshot::default());
        }
    }
    #[test]
    fn stale_before_commit_restores_entitlement_and_preserves_successor() {
        invalidation(false);
    }
    #[test]
    fn stale_after_commit_releases_charge_without_resurrecting_entitlement() {
        invalidation(true);
    }
}
