//! Non-generating d1-omni components.
//!
//! Attention policy is supplied by the caller; these components do not silently
//! select dense attention or allocate a sequence-squared attention mask.

mod head;
pub use head::{DecisionHead, DecisionHeadBatch};
