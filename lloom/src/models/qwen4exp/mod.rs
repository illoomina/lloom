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
pub mod dflash;
pub mod hc;
pub mod hotlist;
pub(crate) mod kernels;
pub mod loader;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod ple;
pub mod qsa;
pub mod schema;
pub mod testing;

pub use config::{Config, ConfigArgs, LayerKind};
pub use deltanet::{decode_step, DeltaNetDims};
pub use deltanet_block::GatedDeltaNet;
pub use hc::HyperConnection;
pub use loader::{
    admit_load, LoadOptions, LoadPlan, SlotBudget, SlotPlan, StreamLoader, StreamOptions,
    DEFAULT_MEMORY_FRACTION,
};
pub use model::{
    prefill_chunk, prefill_pair_on, prefill_tiles, set_prefill_chunk, set_prefill_pair,
    set_prefill_tiles, Layer, MemCheck, Mixer, PassPlan, Qwen4Exp, Shape, State, StatePools,
    DEFAULT_PREFILL_PAIR, DEFAULT_PREFILL_TILES, PREFILL_CHUNK, PREFILL_TRANSIENT_BYTES_PER_TOKEN,
};
pub use moe::{ExpertResidency, MoeLayer, Resident, Streamed};
pub use dflash::QwenHead;
pub use mtp::{GenerationProbe, MtpDrafter, MtpHead, MtpLanes};
pub use ple::{CachedPleTable, MmapPleTable, NgramHash, NgramWindow, PleLayer, PleTable};
pub use qsa::{Indexer, KvCache, KvPool, QsaAttention};

/// Sub-block activation trace, for bisecting a non-finite forward on the real
/// model. Off unless `QWEN4EXP_TRACE` is set (read once); when on, each
/// sub-block's output is read back and summarized — a debugging instrument
/// behind a gate (§3.3a), never on the shipping path.
pub(crate) fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.trace)
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
