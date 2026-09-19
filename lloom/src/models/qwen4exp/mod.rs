//! Qwen3.8-Flash-Next (Qwen4-exp architecture) — a hybrid Gated-DeltaNet +
//! Qwen-Sparse-Attention MoE with 4-branch Gated Residual (HyperConnections),
//! an N-gram PLE table, and an MTP head.
//!
//! Port doc: `design/docs/models/qwen3.8-flash-next.md`. Reference dataflow:
//! llama.cpp `src/models/qwen4exp.cpp` (LLM_ARCH_QWEN4EXP). This module builds
//! the blocks behind issue #747's seams (the `ForwardObserver` observation seam,
//! the `MoeLayer`/`ExpertResidency` residency seam) on the `Resident` path.

pub mod config;
pub mod deltanet;
pub mod deltanet_block;
pub mod hc;
pub mod moe;

pub use config::{Config, LayerKind};
pub use deltanet::{DeltaNetDims, decode_step};
pub use deltanet_block::GatedDeltaNet;
pub use hc::HyperConnection;
pub use moe::{ExpertResidency, MoeLayer, Resident};
