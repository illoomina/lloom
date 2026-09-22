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
#[cfg(any(test, feature = "fuzzing"))]
pub mod buffer_pool_fuzz;
/// Cache-line padding for control state written by two threads (issue #442).
/// The unit is load-bearing -- 128 B per *struct instance*, never per element;
/// the module docs carry the 64x blowup the second half prevents.
pub mod cache_pad;
pub mod command_buffer;
pub mod commands;
/// A stateful model-based fuzzer over `Commands`' submission surface, holding
/// `in_flight` bounded across arbitrary commit/prune/wait interleavings (#508).
#[cfg(any(test, feature = "fuzzing"))]
pub mod commands_fuzz;
/// How the host learns a command buffer finished: the blocking incumbent and
/// two `MTLSharedEvent` arms behind one seam (#559, `DESIGN.md` §11.5o).
///
/// Deliberately **not** in the glob re-export below: `CompletionWait` is the
/// name a caller should reach for as `completion::CompletionWait`, and
/// `SharedEvent` is general enough that acquiring it from a `metal::*` would
/// read as an unrelated type.
pub mod completion;
/// The completion seam on the device: engagement, and that every arm computes
/// the same bytes (#559). `completion.rs`'s own tests cover the routing, which
/// needs no GPU.
#[cfg(test)]
mod completion_gpu_tests;
/// Component digests: one fingerprint per mechanism per variant (#105).
#[cfg(test)]
mod component_digest_tests;
pub mod compute_pipeline;
#[cfg(test)]
mod constants_pool_tests;
/// Where the command buffer is cut: the partition arithmetic (#420 stage 3).
///
/// Pure functions, no device -- `scratch_tests.rs`'s split between the
/// arithmetic and what executes it.
#[cfg(test)]
mod cut_policy_tests;
pub mod device;
pub mod encoder;
pub mod eos_gate;
#[cfg(test)]
mod eos_gate_gpu_tests;
pub mod executor;
#[cfg(test)]
mod executor_tests;
pub mod fence;
#[cfg(test)]
mod fence_probe_tests;
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
pub mod io_queue;
#[cfg(test)]
mod icb_executor_tests;
#[cfg(test)]
mod icb_tests;
/// The paged KV pool: allocation, free list, refcounts and the page table
/// (`DESIGN.md` §10.3c–§10.3g, issue #148). Pure functions of a configuration
/// and an allocation history — no device, no buffer, no dispatch — which is the
/// half §10.3i says to build **before** the page-walking kernel.
pub mod kv_pages;
#[cfg(test)]
mod kv_pages_tests;
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
pub mod staging;
/// A stateful fuzzer over the scratch class's long-context state machine, on
/// the `kv_len` axis the four existing harnesses do not carry (#575).
#[cfg(any(test, feature = "fuzzing"))]
pub mod scratch_fuzz;
#[cfg(test)]
mod scratch_gpu_tests;
#[cfg(test)]
mod scratch_tests;
/// The session region: one owned allocation for the class the arena declines
/// (#47, `DESIGN.md` §9.2x).
///
/// Deliberately **not** in the glob re-export below: `Span` and `TenantId` are
/// general names, and a caller reaching for them should say `session_region::`
/// rather than acquire them from a `metal::*`. `SessionRegion` itself is
/// re-exported by name for the same reason `admission::Budget` is not.
pub mod session_region;
#[cfg(test)]
mod session_region_tests;
pub mod trace;
pub mod upload_shadow;

pub use arena::*;
pub use arena_cursor::*;
pub use buffer::*;
pub use buffer_pool::*;
pub use command_buffer::*;
pub use commands::*;
pub use compute_pipeline::*;
pub use device::*;
pub use encoder::*;
pub use eos_gate::{DispatchGate, EosGateReport};
pub use executor::*;
pub use fence::*;
pub use kv_pages::*;
pub use library::*;
pub use residency_set::*;
pub use scratch::*;
pub use session_region::SessionRegion;
