//! Decision-only transport shape, without fabricated generation/KV/seed APIs.
//! Incumbent Served and generic programmatic model bounds remain unchanged.
use crate::decision::{OrderedJson, SystemOneError, SystemOneResponse};
use anyhow::Result;
use serde_json::Value;

/// HTTP ownership alone requires Send+Sync. A model's generic programmatic
/// decision API need not carry these bounds. Checked health producers remain
/// the endpoint owner's admitted responsibility; transport buffers are separate.
pub trait DecisionEndpoint: Send + Sync {
    /// Caller-supplied provenance label, not a copied checkpoint configuration.
    fn model_name(&self) -> &str;
    fn health_checked(&self) -> Result<Value>;
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError>;
    /// Stop endpoint-owned workers only. This is NOT native DATA/callback or
    /// retirement quiescence, and cannot certify safe destruction after refusal.
    fn shutdown(&self);
}

/// Move request custody out before invoking callbacks. The short mutex never
/// encloses backend work. Reentry/overlap refuses instead of deadlocking.
pub(crate) struct Slot<T> {
    value: parking_lot::Mutex<Option<T>>,
}
impl<T> Slot<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: parking_lot::Mutex::new(Some(value)),
        }
    }
    pub(crate) fn checkout(&self) -> Result<Held<'_, T>> {
        let value = { self.value.lock().take() };
        let value = value.ok_or_else(|| anyhow::anyhow!("decision owner is busy"))?;
        Ok(Held {
            slot: self,
            value: Some(value),
        })
    }
}
pub(crate) struct Held<'a, T> {
    slot: &'a Slot<T>,
    value: Option<T>,
}
impl<T> std::ops::Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("held decision custody")
    }
}
impl<T> std::ops::DerefMut for Held<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("held decision custody")
    }
}
impl<T> Drop for Held<'_, T> {
    fn drop(&mut self) {
        // The same complete tuple returns on Ok/Err/unwind. No backend reset,
        // callback wait or model destruction occurs in this restoration body.
        *self.slot.value.lock() = self.value.take();
    }
}

#[cfg(test)]
#[path = "decision_endpoint_tests.rs"]
mod tests;
