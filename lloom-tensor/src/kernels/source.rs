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
/// The short causal convolution block (`metal_src/conv/short_conv.metal`): the
/// window mechanics and folds LFM2's conv kernels and the Gated DeltaNet
/// prologue compose. A fragment with no `#include`, spliced after a consumer's
/// header by `with_short_conv!` — once per translation unit, so the merged
/// unit below carries it once ahead of both consumers.
macro_rules! with_short_conv {
    ($body:expr) => {
        concat!(
            "#include <metal_stdlib>\nusing namespace metal;\n",
            include_str!("metal_src/conv/short_conv.metal"),
            "\n",
            $body
        )
    };
}

pub const CONV: &str = with_short_conv!(include_str!("metal_src/conv.metal"));
/// The Gated DeltaNet block (`metal_src/gdn/`): the decode prologue on the
/// short-conv block, the recurrence step, the gated norm, the rows and
/// multi-position forms, the chunked prefill. Split out of `qwen4exp.metal`
/// so a second GDN model composes it (§7.3).
pub const GDN: &str = with_short_conv!(include_str!("metal_src/gdn/gdn.metal"));
/// The batched EOS-index reduction (the `GpuBatched` EOS arm): one dispatch per
/// commit window returns the first stop-token index or `n`. Self-contained (its
/// own `#include`, an inlined threadgroup min-reduction), so NOT a `with_utils!`
/// consumer and a translation unit of its own.
pub const EOS_INDEX: &str = include_str!("metal_src/eos_index.metal");
/// The EOS gate (lloom #735 step 4): one threadgroup that zeroes the next
/// decode step's indirect-dispatch table when the sampled token is a stop
/// token (or the sticky flag is already up). Self-contained like `EOS_INDEX`.
pub const EOS_GATE: &str = include_str!("metal_src/eos_gate.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
/// The Qwen4-exp model's own kernels (`lloom::models::qwen4exp`): the
/// HyperConnection brackets, the MoE routing, the QSA preparation — each one
/// launch. A `with_utils!` consumer: its SwiGLU calls the shared `silu<T>`.
/// The Gated DeltaNet kernels are [`GDN`]'s.
pub const QWEN4EXP: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/qwen4exp.metal")
);
pub const FLASH_DECODING: &str = include_str!("metal_src/attention/flash_decoding.metal");
pub const INDEXING: &str = with_utils!(
    "#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/indexing.metal")
);
/// The GEMV source and `mlx_gemm.metal` below live in the `metal_src/gemm/`
/// role directory (§7.2): the two translation units the projection fusion
/// touches, grouped by role. The embedded text is byte-identical to before the
/// move, so the compiled library and every canonical digest are unmoved — the
/// relocation changes only the `include_str!` path (§7.3 move gate).
///
/// `gemv` became a `with_utils!` consumer for issue #599: `EpilogueSilu` is now
/// a call site of `utils.metal`'s shared `silu<T>()` rather than a second copy
/// of the expression (`DESIGN.md` §15.2 #15).
///
/// It contributes `<metal_simdgroup>` to the prologue, which the other
/// consumers do not need and which this file's own header line carried. Note
/// `gemv.metal` keeps its MLX attribution comment and its `MLX_MTL_*` macros
/// above the splice point exactly as before; only the include line moves out,
/// on `with_utils!`'s stated requirement that the consumer's header be the one
/// the compiler sees first.
pub const GEMV: &str = with_utils!(
    "#include <metal_simdgroup>\n#include <metal_stdlib>\nusing namespace metal;",
    include_str!("metal_src/gemm/gemv.metal")
);
pub const MLX_GEMM: &str = include_str!("metal_src/gemm/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");

/// The quantized matvec with a block on its store (`metal_src/quantized_store.metal`):
/// the quantized source, the store types, and one instantiation per kind
/// whose plain matvec has a genericized core × store block, generated from
/// the kinds table rather than listed — `kernel_mul_mv_<stem>_f32_silu` and
/// `kernel_mul_mv_<stem>_f32_mul`. Its own translation unit, compiled only
/// when a dispatch asks for a store block, so the default library is
/// untouched.
pub fn quantized_store() -> &'static str {
    static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TEXT.get_or_init(|| {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(QUANTIZED.len() + 8192);
        s.push_str(QUANTIZED);
        s.push('\n');
        s.push_str(include_str!("metal_src/quantized_store.metal"));
        s.push('\n');
        for (stem, store, suffix) in super::kernels::quantized::mv_store_instantiations() {
            let _ = writeln!(
                s,
                "template [[host_name(\"kernel_mul_mv_{stem}_f32{suffix}\")]] kernel \
                 decltype(kernel_mul_mv_{stem}_generic<float, {store}>) \
                 kernel_mul_mv_{stem}_generic<float, {store}>;"
            );
        }
        s
    })
}
pub const RANDOM: &str = concat!(
    include_str!("metal_src/random.metal"),
    "\n",
    include_str!("metal_src/sampling_topk.metal"),
    "\n",
    include_str!("metal_src/sampling_multinomial.metal"),
    "\n",
    include_str!("metal_src/sampling_gumbel.metal")
);
pub const REDUCE: &str = with_utils!(
    "#include <metal_stdlib>\n#include <metal_limits>\nusing namespace metal;",
    include_str!("metal_src/reduce.metal")
);
/// The three RoPE entry points, split out of `reduce.metal` into the
/// `metal_src/rope/` role directory (§7.2). Unlike the `attention/` and `gemm/`
/// moves, this is a translation-unit split: the rope family is a self-contained
/// fragment that references no reduction preamble and no `utils.metal` helper,
/// so it carries its own `#include <metal_stdlib>` and is NOT a `with_utils!`
/// consumer. The residual `reduce.metal` keeps reduce/arg_reduce/softmax/norm,
/// whose shared preamble cannot split (§7.2 finding, #653).
pub const ROPE: &str = include_str!("metal_src/rope/rope.metal");
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
/// `scaled_dot_product_attention.metal` and `flash_decoding.metal` above live
/// in the `metal_src/attention/` role directory (§7.2): the two attention
/// translation units, grouped by role. The embedded text is byte-identical to
/// before the move, so the compiled libraries and every canonical digest are
/// unmoved — the relocation changes only the `include_str!` path (§7.3 move
/// gate). Both keep their own `Source` variant, so every dispatch site, the
/// `Pipelines` key and the `Libraries` key are untouched.
pub const SDPA: &str = include_str!("metal_src/attention/scaled_dot_product_attention.metal");

/// The non-MLX sources as **one translation unit** (issue #477,
/// `DESIGN.md` §13.1e/§13.1j).
///
/// # What this is for
///
/// #452 asked whether the per-file split hides cross-kernel dataflow and
/// measured **no kernel's `__compute` moving on 17 of 17** — but it could not
/// run the experiment as specified, because at that commit the 19 sources did
/// **not** compile as one unit (3434 errors, 25 redefinitions). #458 hoisted
/// the shared helpers and dissolved 18 of the elementwise cluster's 19 edges;
/// #466/#477 resolved the last one (a `strided_indexer` KIND mismatch between
/// `binary.metal` and `reduce.metal`) and verified by direct n-way compile that
/// **all 16 non-MLX sources compile together with 0 errors**.
///
/// **That landed and nothing consumed it.** A merged unit that no code path
/// compiles is a fusion the optimizer has never seen whole, and every
/// `__compute` figure in the corpus was taken with **separate translation units
/// and `-fno-lto` semantics**. This const is the consumer.
///
/// # Why the composition is what it is
///
/// **`utils.metal` is spliced exactly ONCE.** It carries no `#pragma once`
/// deliberately (see the module header: *"there is one textual inclusion per
/// library by construction"*), so concatenating two `with_utils!` consumers
/// naively redefines every shared helper — a redefinition manufactured by the
/// composition rather than present in the tree. `issue-466-raw/pairwise.sh`
/// records this as the thing that would have measured a strawman.
///
/// **The prologue is the UNION of the members' prologues**, not any one
/// member's: `reduce.metal` needs `<metal_limits>`, `unary.metal` needs
/// `<metal_math>` and `gemv.metal` needs `<metal_simdgroup>`, and a merged unit
/// that took only the bare `<metal_stdlib>` line would fail to compile for a
/// reason that is about the probe rather than about the merge.
///
/// `<metal_simdgroup>` joined that union in #599, when `gemv` became a
/// `with_utils!` consumer and stopped carrying its own include line. Before
/// then `gemv.metal`'s own header supplied it and the union did not have to
/// name it -- so this is a case where making a source a fragment moves a
/// requirement from the file into this list, and the partition test below is
/// what stops that being silent.
///
/// # What is NOT in it, and why
///
/// The three MLX sources (`mlx_gemm`, `mlx_sort`, `sdpa`) are excluded: they
/// carry mutually incompatible bodies for `BlockMMA`/`Limits`/`Transform*` and
/// **collide on function-constant indices 200/201/300** (§13.1e's fourth
/// collision class — a namespace the per-file split hides, where in one unit a
/// caller supplying one kernel's constants would specialise the other's). That
/// is a port rather than a merge (#409) and is #477's declared non-goal.
///
/// `quantized.metal` IS included even though LFM2 dispatches none of it
/// (§11.2f): excluding it would make the merged arm's kernel set differ from
/// the union of the split arm's for a reason unrelated to the merge, and a
/// name present in one arm and absent in the other is a difference the
/// engagement proof would have to explain.
pub const MERGED_NON_MLX: &str = concat!(
    // The union prologue. `unary` contributes <metal_math>, `reduce`
    // contributes <metal_limits>, `gemv` contributes <metal_simdgroup> (#599),
    // everything else the bare stdlib line.
    "#include <metal_stdlib>\n#include <metal_limits>\n#include <metal_math>\n#include <metal_simdgroup>\nusing namespace metal;\n",
    include_str!("metal_src/utils.metal"),
    "\n",
    include_str!("metal_src/affine.metal"),
    "\n",
    include_str!("metal_src/arena_alloc.metal"),
    "\n",
    include_str!("metal_src/binary.metal"),
    "\n",
    include_str!("metal_src/cast.metal"),
    "\n",
    include_str!("metal_src/conv/short_conv.metal"),
    "\n",
    include_str!("metal_src/conv.metal"),
    "\n",
    include_str!("metal_src/eos_gate.metal"),
    "\n",
    include_str!("metal_src/eos_index.metal"),
    "\n",
    include_str!("metal_src/fill.metal"),
    "\n",
    include_str!("metal_src/attention/flash_decoding.metal"),
    "\n",
    include_str!("metal_src/gemm/gemv.metal"),
    "\n",
    include_str!("metal_src/gdn/gdn.metal"),
    "\n",
    include_str!("metal_src/indexing.metal"),
    "\n",
    include_str!("metal_src/quantized.metal"),
    "\n",
    include_str!("metal_src/qwen4exp.metal"),
    "\n",
    include_str!("metal_src/random.metal"),
    "\n",
    include_str!("metal_src/sampling_topk.metal"),
    "\n",
    include_str!("metal_src/sampling_multinomial.metal"),
    "\n",
    include_str!("metal_src/sampling_gumbel.metal"),
    "\n",
    include_str!("metal_src/reduce.metal"),
    "\n",
    include_str!("metal_src/rope/rope.metal"),
    "\n",
    include_str!("metal_src/scratch.metal"),
    "\n",
    include_str!("metal_src/sort.metal"),
    "\n",
    include_str!("metal_src/ternary.metal"),
    "\n",
    include_str!("metal_src/unary.metal"),
    "\n",
);

/// The [`Source`] variants [`MERGED_NON_MLX`] contains, in composition order.
///
/// **This list is what makes the arm checkable rather than asserted.** A
/// `Source` added to the enum and not here keeps its own library under the
/// merged arm — which is correct behaviour and silent, so
/// `merged_unit_covers_every_non_mlx_source` in `kernel.rs` asserts the
/// partition explicitly: every variant is either in this list or in the MLX
/// set, and a new one fails the test rather than quietly opting out.
pub const MERGED_MEMBERS: &[Source] = &[
    Source::Affine,
    Source::ArenaAlloc,
    Source::Binary,
    Source::Cast,
    Source::Conv,
    Source::EosGate,
    Source::EosIndex,
    Source::Fill,
    Source::FlashDecoding,
    Source::Gemv,
    Source::Gdn,
    Source::Indexing,
    Source::Quantized,
    Source::Qwen4Exp,
    Source::Random,
    Source::Reduce,
    Source::Rope,
    Source::Scratch,
    Source::Sort,
    Source::Ternary,
    Source::Unary,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    /// GPU-side bump allocation over the activation arena (`DESIGN.md` §9.2d,
    /// issue #70). Not dispatched by any tensor op -- the arena drives it.
    ArenaAlloc,
    Binary,
    Cast,
    Conv,
    /// The batched EOS-index reduction (the `GpuBatched` EOS arm, issue: the
    /// per-token flag train's replacement). Its own translation unit.
    EosGate,
    EosIndex,
    Fill,
    /// FlashDecoding: attention split over independent contiguous KV chunks,
    /// with an index-ordered combine (`DESIGN.md` §10.4, §17 Phase 5 item 16,
    /// issue #116). A **new** kernel rather than a rewiring of
    /// `sdpa_vector_2pass`, whose blocks read a strided interleave at a fixed
    /// count of 32 and therefore do not own a contiguous range (§10.3b).
    FlashDecoding,
    Gemm,
    Gemv,
    /// The Gated DeltaNet block (`metal_src/gdn/`), composed with the short
    /// conv block. Its own translation unit.
    Gdn,
    Indexing,
    MlxSort,
    Quantized,
    /// The quantized matvec with a block on its store ([`quantized_store`]):
    /// its own translation unit, compiled on the first dispatch that asks.
    QuantizedStore,
    /// The Qwen4-exp block kernels (`lloom::models::qwen4exp::kernels`).
    Qwen4Exp,
    Random,
    Reduce,
    /// RoPE: the three rotary-position-embedding entry points (rope, rope_i,
    /// rope_thd), split out of `reduce.metal` into `metal_src/rope/` (§7.2). A
    /// separate `Source` so the family is one translation unit compiled on its
    /// own; the residual reduction preamble stays in `Source::Reduce`.
    Rope,
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
