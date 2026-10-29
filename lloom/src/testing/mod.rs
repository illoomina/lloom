//! Test-support surfaces shared by the crate's tests and `lloom-fuzz`
//! (RESTRUCTURING §7.1 tier 2: `#[doc(hidden)] pub`, not API).

// The real-scale backends locate checkpoints through `hub`, which is
// host-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod fixture;
pub mod ghost;
pub mod lfm2_gguf;
#[cfg(not(target_arch = "wasm32"))]
pub mod scale;
pub mod tiny;
