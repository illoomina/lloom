//! Typed decision questions and readout, independent of token generation.
//!
//! Option order is part of the model input. Constructors take ordered vectors;
//! adapters must not round-trip request objects through an unordered map.

mod json;
mod omni;
mod omni_readout;
mod question;
mod readout;
mod request;
mod response;
mod state;
mod temperature;
#[cfg(feature = "serve")]
mod template;

pub use json::OrderedJson;
pub use omni::{OmniMedia, OmniPreparedQuestion, OmniSpecialTokens};
pub use omni_readout::OmniReadout;
pub use question::{Answer, Question};
pub use readout::{CausalLabels, CausalReadout, PreparedQuestion};
pub use request::{parse_systemone_questions, NoulAliases};
pub use response::{SystemOneError, SystemOneResponse};
pub use state::{parse_systemone_state, MediaSource, ParsedState, StateNullability};

pub use temperature::Temperature;
#[cfg(feature = "serve")]
pub use template::{SystemOneMedia, SystemOneTemplate, SystemOneTemplateKind};

#[cfg(test)]
mod contract_tests;

#[cfg(test)]
mod session_tests;
