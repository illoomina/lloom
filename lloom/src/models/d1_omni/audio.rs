//! Owned omni audio components. Full frontend/Conformer/family assembly remains
//! in progress; relative attention is an explicit caller-selected policy.
mod relative;
pub use relative::{RelativeAttentionPlan, RelativeInputs, RelativeScratch};
#[cfg(test)]
mod relative_tests;

mod encoder;
pub use encoder::{AudioConfig, AudioEncoder};
