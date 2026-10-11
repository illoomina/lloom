//! Typed decision questions and readout, independent of token generation.
//!
//! Option order is part of the model input. Constructors take ordered vectors;
//! adapters must not round-trip request objects through an unordered map.

mod json;
mod omni;
mod omni_readout;
mod question;
mod readout;
mod temperature;

pub use json::OrderedJson;
pub use omni::{OmniMedia, OmniPreparedQuestion, OmniSpecialTokens};
pub use omni_readout::OmniReadout;
pub use question::{Answer, Question};
pub use readout::{CausalReadout, PreparedQuestion};

pub use temperature::Temperature;

#[cfg(test)]
mod contract_tests;

#[cfg(test)]
mod session_tests;
