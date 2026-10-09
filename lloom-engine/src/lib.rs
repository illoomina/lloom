//! The serving engine (RESTRUCTURING §4, §7.1) — the vLLM-analog, `lloom-engine`
//! (= §14.6's retired `llumos`). It links `lloom`'s tier-1 API and turns a model
//! into a decode service: the decode loop, sampling, the token-streaming
//! primitive, and the per-request seam.
//!
//! What lives here and what does not (RESTRUCTURING §4):
//! - **here**: the decode loop ([`decode`]), driven over `lloom`'s `CausalLM`
//!   seam so it is model-blind; the streaming primitive; the request seam
//!   ([`seam`]).
//! - **not here**: the OpenAI `/v1` facade and tool execution stay in amber
//!   until §14.6's trigger (a second `/v1` consumer, #328). amber becomes a
//!   consumer of this crate, rendering prompts and parsing tools around the
//!   request seam.
//!
//! The optional `Speculator` (Wave C, `lloom::nn::Speculator`) plugs into the
//! same decode session as the target's drafter; the speculative driver is
//! `lloom::nn::Session::speculative_step`. The non-speculative loop here is
//! `verify_step`'s sibling — the plain per-token path — and its first production
//! caller (`RESTRUCTURING_DESIGN.md` §1.9, §10.1 wave F).

pub mod batch;
pub mod couplings;
pub mod decode;
pub mod option_grammar;
pub mod option_schema;
pub mod option_tree;
pub mod options;
#[cfg(feature = "serve")]
pub mod app;
pub mod seam;
pub mod sink;
pub mod slots;
pub mod stream;

pub use batch::{
    spawn as spawn_batch, BatchConfig, BatchCounters, BatchDone, BatchEvent, BatchHandle,
    BatchMode, BatchRequest, ClosedJob, CountersHook, Driver, Idle, MakeDriver, RowSpec,
    StaticBody, drive_closed_rows,
};
pub use decode::{
    ring_eos_read, DecodeParams, DecodeSession, Decoded, ReadbackArm, RingArm, RingPool, Stop,
};
pub use options::{
    BenchConfig, DecodeConfig, EngineOptions, EosKind, ExecutorConfig, LoadConfig, MemoryConfig,
    ModelConfig, PrefillConfig, PrefixConfig, PromptConfig, ReadbackConfig, ReadbackKind,
    ReasoningConfig, ReportConfig, SamplingConfig, SchedulerConfig, ServerConfig,
    SpeculativeConfig, TuningOptions,
};
pub use seam::Completer;
pub use slots::{Routed, SlotPool, DEFAULT_SIMILARITY};
pub use stream::TextStream;
