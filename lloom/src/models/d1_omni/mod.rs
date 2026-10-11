//! Non-generating d1-omni components.
//!
//! Attention policy is supplied by the caller; these components do not silently
//! select dense attention or allocate a sequence-squared attention mask.

mod head;
mod host;
pub use head::{DecisionHead, DecisionHeadBatch};

mod trunk;
pub use trunk::{QuerySegment, TextConfig, Trunk, TrunkBatch};

mod core;
pub use core::{DecisionCore, DecisionCoreConfig, DecisionRow, PreparedRows};

mod vision;
pub use vision::{PreparedVisionImage, Vision, VisionImagePlan};

mod audio;
pub use audio::{
    AudioConfig, AudioEncoder, FrontendScratch, MelFeatures, MelFrontend, RelativeAttentionPlan,
    RelativeInputs, RelativeScratch,
};

mod model;
pub use model::{OmniConfig, OmniImage, OmniModel};

mod loaded;
