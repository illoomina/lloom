/// The memory budget: predict the peak before allocating, and refuse (§9.5).
///
/// Deliberately **not** in the glob re-export below: `Budget` and `Footprint`
/// are general names, and a caller reaching for them should say `admission::`
/// rather than acquire them from a `metal::*`.
pub mod admission;
pub mod arena;
pub mod arena_cursor;
#[cfg(test)]
mod arena_gpu_tests;
pub mod buffer;
pub mod buffer_pool;
/// A stateful model-based fuzzer over the pool's public surface (#494).
#[cfg(test)]
mod buffer_pool_fuzz;
/// Cache-line padding for control state written by two threads (issue #442).
/// The unit is load-bearing -- 128 B per *struct instance*, never per element;
/// the module docs carry the 64x blowup the second half prevents.
pub mod cache_pad;
pub mod command_buffer;
pub mod commands;
/// A stateful model-based fuzzer over `Commands`' submission surface, holding
/// `in_flight` bounded across arbitrary commit/prune/wait interleavings (#508).
#[cfg(test)]
mod commands_fuzz;
/// Component digests: one fingerprint per mechanism per variant (#105).
#[cfg(test)]
mod component_digest_tests;
pub mod compute_pipeline;
/// Where the command buffer is cut: the partition arithmetic (#420 stage 3).
///
/// Pure functions, no device -- `scratch_tests.rs`'s split between the
/// arithmetic and what executes it.
#[cfg(test)]
mod cut_policy_tests;
pub mod device;
pub mod encoder;
pub mod executor;
#[cfg(test)]
mod executor_tests;
pub mod fence;
#[cfg(test)]
mod flash_decoding_tests;
/// Strict ordering verification (issue #185). Every entry point is inert
/// without the `hazard-audit` feature, so the module is declared
/// unconditionally and compiles to nothing when the feature is off --
/// `run_telemetry`'s shape, for its reasons.
pub mod hazard_audit;
#[cfg(test)]
mod hazard_audit_tests;
pub mod icb;
#[cfg(test)]
mod icb_executor_tests;
#[cfg(test)]
mod icb_tests;
pub mod library;
/// ASan shadow-memory poisoning for allocator invariants (issue #475). Every
/// entry point is inert without the `asan-poison` feature, so the module is
/// declared unconditionally and compiles to nothing when the feature is off --
/// `run_telemetry`'s shape, for its reasons.
///
/// Deliberately **not** in the glob re-export below: `poison` and `counters`
/// are general names, and a caller reaching for them should say `poison::`
/// rather than acquire them from a `metal::*`.
pub mod poison;
pub mod profile;
pub mod residency_set;
/// Memory-class and event telemetry (issue #171). Every entry point is inert
/// without the `run-telemetry` feature, so the module is declared
/// unconditionally and compiles to nothing when the feature is off.
pub mod run_telemetry;
pub mod scratch;
#[cfg(test)]
mod scratch_gpu_tests;
#[cfg(test)]
mod scratch_tests;
pub mod trace;

pub use arena::*;
pub use arena_cursor::*;
pub use buffer::*;
pub use buffer_pool::*;
pub use command_buffer::*;
pub use commands::*;
pub use compute_pipeline::*;
pub use device::*;
pub use encoder::*;
pub use executor::*;
pub use fence::*;
pub use library::*;
pub use residency_set::*;
pub use scratch::*;
