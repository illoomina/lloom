//! Owned omni audio components. Full family/admission assembly remains in
//! progress; attention and the host frontend are explicit policies.
mod relative;
pub use relative::{RelativeAttentionPlan, RelativeInputs, RelativeScratch};
#[cfg(test)]
mod relative_tests;

mod encoder;
pub(super) use encoder::AudioWeights;
pub(crate) use encoder::QuantizedAudioEncoder;
pub use encoder::{AudioConfig, AudioEncoder};

mod frontend;
pub use frontend::{FrontendScratch, MelFeatures, MelFrontend};
