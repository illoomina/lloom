//! The `.metal` sources, embedded at build time and compiled at RUNTIME by
//! `newLibraryWithSource` (`kernel.rs`).
//!
//! **Shared helpers are composed here with `concat!`, not with `#include`**
//! (issue #458). There is no build step and no `.metal` file on disk when a
//! library is compiled, so a quoted `#include "utils.metal"` has no directory
//! to resolve against and fails with `'utils.metal' file not found` -- measured,
//! and it fails *even when the process's working directory is `metal_src/`*.
//! `concat!` is textual inclusion by another route: same one-translation-unit
//! result, no filesystem dependency, and the shared text still ships inside the
//! binary exactly as it did when it was copy-pasted into each file.
//!
//! `UTILS` must come FIRST in each `concat!` and each consumer keeps its own
//! `#include <metal_stdlib>`: `utils.metal` deliberately has no include of its
//! own, so the consumer's header line has to be the one that is seen -- which
//! is why `UTILS` is spliced *after* it rather than blindly prepended.
//!
//! Six files carried a byte-identical private copy of `get_strided_index`
//! before this; `ternary.metal` carried a seventh that differed, and
//! `utils.metal` itself carried an eighth that was reachable by nothing. See
//! `measurements/issue-458-helper-duplication.md`.

/// Compose one source: an include prologue, then `utils.metal`'s shared
/// helpers (`get_strided_index`, `div_ceil`, `work_per_thread`), then the body.
///
/// The prologue is written out per source rather than parsed off the body,
/// because `concat!` needs every piece to be a literal known to the macro
/// expander -- there is no split point it could compute. `utils.metal` carries
/// no `#include` of its own precisely so that the prologue can come first.
///
/// `utils.metal` is deliberately NOT a `Source` variant: it is never compiled
/// alone and would not compile alone.
macro_rules! with_utils {
    ($prologue:expr, $body:expr) => {
        concat!(
            $prologue,
            "\n",
            include_str!("metal_src/utils.metal"),
            "\n",
            $body
        )
    };
}

pub const AFFINE: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/affine.metal")
);
pub const ARENA_ALLOC: &str = include_str!("metal_src/arena_alloc.metal");
pub const BINARY: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/binary.metal")
);
pub const CAST: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/cast.metal")
);
pub const CONV: &str = include_str!("metal_src/conv.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
pub const FLASH_DECODING: &str = include_str!("metal_src/flash_decoding.metal");
pub const INDEXING: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/indexing.metal")
);
pub const GEMV: &str = include_str!("metal_src/gemv.metal");
pub const MLX_GEMM: &str = include_str!("metal_src/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");
pub const RANDOM: &str = include_str!("metal_src/random.metal");
pub const REDUCE: &str = with_utils!(
    "#include <metal_stdlib>\n#include <metal_limits>\nusing namespace metal;",
    include_str!("metal_src/reduce.metal")
);
pub const SCRATCH: &str = include_str!("metal_src/scratch.metal");
pub const SORT: &str = include_str!("metal_src/sort.metal");
pub const TERNARY: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/ternary.metal")
);
pub const UNARY: &str = with_utils!(
    "#include <metal_stdlib>\n#include <metal_math>\nusing namespace metal;",
    include_str!("metal_src/unary.metal")
);
pub const SDPA: &str = include_str!("metal_src/scaled_dot_product_attention.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    /// GPU-side bump allocation over the activation arena (`DESIGN.md` §9.2d,
    /// issue #70). Not dispatched by any tensor op -- the arena drives it.
    ArenaAlloc,
    Binary,
    Cast,
    Conv,
    Fill,
    /// FlashDecoding: attention split over independent contiguous KV chunks,
    /// with an index-ordered combine (`DESIGN.md` §10.4, §17 Phase 5 item 16,
    /// issue #116). A **new** kernel rather than a rewiring of
    /// `sdpa_vector_2pass`, whose blocks read a strided interleave at a fixed
    /// count of 32 and therefore do not own a contiguous range (§10.3b).
    FlashDecoding,
    Gemm,
    Gemv,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    /// FlashDecoding partials: the scratch class and its sizing policy
    /// (`DESIGN.md` §9.1, issue #71). Not dispatched by any tensor op -- the
    /// scratch arena drives it, and the kernel is a stub until Phase 4/5
    /// implements FlashDecoding itself.
    Scratch,
    Sort,
    Ternary,
    Unary,
    Sdpa,
}
