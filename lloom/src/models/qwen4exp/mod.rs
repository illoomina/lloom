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
pub mod loader;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod ple;
pub mod qsa;
pub mod testing;

pub use config::{Config, LayerKind};
pub use deltanet::{decode_step, DeltaNetDims};
pub use deltanet_block::GatedDeltaNet;
pub use hc::HyperConnection;
pub use loader::{LoadOptions, SlotBudget, SlotPlan, StreamLoader, StreamOptions};
pub use model::{Cache, Layer, Mixer, Qwen4Exp};
pub use moe::{ExpertResidency, MoeLayer, Resident, Streamed};
pub use mtp::{MtpDrafter, MtpHead};
pub use ple::{CachedPleTable, MmapPleTable, NgramHash, NgramWindow, PleLayer, PleTable};
pub use qsa::{Indexer, KvCache, QsaAttention};

/// Sub-block activation trace, for bisecting a non-finite forward on the real
/// model. Off unless `QWEN4EXP_TRACE` is set (read once); when on, each
/// sub-block's output is read back and summarized — a debugging instrument
/// behind a gate (§3.3a), never on the shipping path.
pub(crate) fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_TRACE").is_some())
}

pub(crate) fn trace(tag: &str, l: usize, t: &lloom_tensor::Tensor) {
    if !trace_enabled() {
        return;
    }
    let v = t
        .flatten_all()
        .and_then(|t| t.to_dtype(lloom_tensor::DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap_or_default();
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let maxabs = v
        .iter()
        .filter(|x| x.is_finite())
        .fold(0f32, |m, x| m.max(x.abs()));
    if nonfinite > 0 || l <= 4 {
        eprintln!(
            "TRACE layer={l} {tag} n={} nonfinite={nonfinite} maxabs={maxabs:.4e}",
            v.len()
        );
    }
}
