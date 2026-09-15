// Shared helpers for the elementwise-family kernels.
//
// HOW THIS FILE IS INCLUDED, and it is not with `#include`.
//
// `.metal` sources are `include_str!`-embedded and compiled at RUNTIME by
// `newLibraryWithSource` (`kernel.rs`, `source.rs`) -- there is no build step
// and no file on disk at the point of compilation. A quoted `#include` has no
// directory to resolve against and fails with `'utils.metal' file not found`,
// **including** when the process's working directory is `metal_src/` itself.
// Measured; see `measurements/issue-458-helper-duplication.md`.
//
// So this file is composed into its consumers on the RUST side:
//
//     pub const UNARY: &str = concat!(include_str!("metal_src/utils.metal"),
//                                     include_str!("metal_src/unary.metal"));
//
// which is textual inclusion by another route, with the same one-translation-
// unit result and no filesystem dependency. `#pragma once` is therefore NOT
// used here: there is one textual inclusion per library by construction, and a
// guard would imply a mechanism that does not exist. Do not add one, and do not
// concatenate this file into a source twice.
//
// WHY THESE BODIES AND NOT THE ONES THIS FILE USED TO CARRY (issue #458).
//
// This file previously held a `#pragma once` and a set of helpers that NOTHING
// included -- no `Source` variant, no Rust reference, no `#include` anywhere in
// the tree. Its `get_strided_index` took `constant const uint &num_dims` where
// every live copy takes `size_t` and a templated `PtrT`, and its
// `max_shared_mem<T>` computed `prev_p2(MAX_SHARED_MEM / sizeof(T))` against
// `reduce.metal`'s `div_ceil<MAX_SHARED_MEM, sizeof(T)>()` -- 8192 against
// 16384 at `sizeof(T) == 2`, a factor of two at every dtype. The bodies below
// are the ones the six live copies actually carried, so composing them changes
// no emitted code; the abandoned ones are gone rather than promoted.
//
// `max_shared_mem`, `next_p2`, `prev_p2`, `nonzero` and `granularity` are NOT
// here. They were duplicated between the dead file and `reduce.metal` only, so
// `reduce.metal` is already their single definition site and moving them would
// be motion without a second consumer.

// Templated on the pointer *type* rather than written once per address space.
// The classical entry points pass `constant size_t *`; the packed ones pass
// `device const size_t *`, because an ICB command can bind a buffer but has no
// `setBytes` at all (`DESIGN.md` §3.7c). Deducing the address space from the
// pointer type is what lets one body serve both -- the same move issue #38 made
// for `reduce.metal`'s `strided_indexer`, and it is why no arithmetic is
// duplicated here.
template <typename PtrT>
METAL_FUNC uint get_strided_index(
    uint idx,
    size_t num_dims,
    PtrT dims,
    PtrT strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

// TWO FORMS, AND THE SECOND ONE IS THE DRIFT ISSUE #458 FOUND.
//
// Five of the six private copies had no loop pragma. `ternary.metal`'s carried
// `#pragma clang loop unroll(full)`. That is a REAL codegen difference and it
// is real in BOTH directions -- measured, not assumed:
//
//   shared body WITH the pragma     -> moves 5 kernels in the other five files
//                                      (`is_u32_f16`, `is_i64_f16`, and the
//                                      strided arms of affine/cast/unary/binary)
//   shared body WITHOUT the pragma  -> moves `where_u8_f16`
//   two named forms (this file)     -> moves NOTHING, 26 of 26 byte-identical
//
// So there is no single body that reproduces the tree, and picking either one
// would have silently changed code on whichever side lost. Keeping both as two
// NAMES is still deduplication -- one definition per form, the arithmetic
// written once each, nothing copy-pasted across six files -- and it is the only
// arrangement of the three that is behaviour-preserving.
//
// `where_u8_f16` is why the probe hashes section CONTENT rather than comparing
// section SIZE: dropping the pragma moves its `__compute` bytes at an
// **unchanged 5728 B**, so a size-only comparison (#431's method, and what #458
// asked for) reports "unmoved" and misses it. See
// `measurements/issue-458-helper-duplication.md`.
//
// If a future change makes the two forms provably equivalent, collapse them --
// but measure both directions first, because this one did not look like it had
// a difference either.
template <typename PtrT>
METAL_FUNC uint get_strided_index_unrolled(
    uint idx,
    size_t num_dims,
    PtrT dims,
    PtrT strides
) {
    uint strided_i = 0;
    #pragma clang loop unroll(full)
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

// ---------------------------------------------------------------------------
// Activation primitives (issue #599, `DESIGN.md` §15.2 #15 / `CONTRIBUTING.md`
// §3.6): ONE definition, called from every placement.
// ---------------------------------------------------------------------------
//
// WHY THESE ARE HERE AND NOT IN A PLACEMENT. SiLU is a reusable numeric
// primitive that three different kernel shapes evaluate -- a standalone unary
// op, a fused GEMV epilogue, and (dead, see below) a ggml-imported kernel. A
// name like `SiluKernel` names one of those homes and forces the other two to
// duplicate it, which is exactly what had happened: `unary.metal:326`,
// `gemv.metal:175` and `quantized.metal:1563`/`:1571` each carried their own
// copy of `x / (1 + exp(-x))`.
//
// **This is a HYGIENE change and not a correctness fix, and the distinction is
// measured rather than asserted (#604, `DESIGN.md` §12.2k).** The two LIVE
// copies -- `usilu` and `EpilogueSilu` -- are the same expression and compile
// bit-identically at f32 and at f16; the two that diverge (1 and 3 f16 ulp,
// from the `1.0f` literal) are `quantized.metal`'s and have **zero callers**.
// So nothing here computes a different number and no digest can move because
// of it. #599's original premise -- *"a digest could move today purely because
// two kernels spell SiLU differently"* -- was falsified before this landed, and
// consolidating on a correctness argument would have been consolidating on a
// false one (`CONTRIBUTING.md` §1.2).
//
// WHAT THE TEMPLATE PARAMETER IS FOR, because it is the whole design. The two
// live call sites evaluate at DIFFERENT precisions, deliberately:
//
//   `usilu`         -- `T`, the storage type. `define_unary_op` templates
//                      `operator()` on `T`, so `silu_f16` computes at half on
//                      an already-rounded input and rounds twice.
//   `EpilogueSilu`  -- `AccT`, which is `float` at every dtype
//                      (`GemvDefaultAccT`). It sees the f32 accumulator BEFORE
//                      the narrowing store, so it computes at float and rounds
//                      once.
//
// That difference is `DESIGN.md` §12.2d's finding and is why `--gemv-silu`
// moves `logits_sha256` (§12.2i). **It must survive this consolidation
// untouched**: it is a real precision difference that was argued and chosen,
// not an accident of duplication. A shared function templated on its value type
// preserves it by construction -- `silu<half>` and `silu<float>` are two
// instantiations of one body, and each call site keeps the type it always
// passed. Writing this as a non-templated `float` helper would have silently
// promoted `usilu` to the epilogue's precision and moved the shipping digests,
// which is the destructive form of this change.
//
// SPELLING. The body is `usilu`'s, character for character: a bare `1`, not
// `1.0f`. #604 measured that literal as the one lever that moves bits -- the
// `f` suffix promotes the addition to float where a bare `1` converts to `T` --
// so preserving the character-level spelling is what makes the consolidation
// bit-preserving at f16. `gemv.metal:127-130` had already chosen to match
// `usilu` character for character for the same reason; this makes that
// agreement structural instead of a comment asking two files to stay in step.
//
// THE `static_cast<T>` IS LOAD-BEARING AND IT IS NOT NEW. `define_unary_op`
// wraps every operation in `static_cast<T>(op)`, so `usilu`'s emitted body was
// always `static_cast<T>(x / (1 + exp(-x)))`; `EpilogueSilu` returned `AccT`
// from an expression already of type `AccT`, where the cast is the identity.
// Hoisting the expression without the cast does not merely lose a formality --
// **it does not compile at `bfloat`**: `1 + exp(-x)` promotes to `float`, and
// MSL has no implicit `float` -> `bfloat` conversion, so the return fails with
// *"cannot initialize return object of type 'bfloat' with an rvalue of type
// 'float'"* and takes the whole `unary` library with it. Found by the fuzzers
// rather than by reading, on the first run after the hoist.
//
// That is the same `bfloat`-narrowing class the #613 arms hit with `abs`,
// `clamp` and `floor`, arriving through conversion rather than through overload
// resolution -- and it is why the primitive carries the cast rather than
// leaving it to each call site: a call site that forgets it fails at
// **`bfloat` only**, which is a dtype neither canonical digest exercises.
//
// At `T = half` and `T = float` the cast is exactly the one `define_unary_op`
// already emitted at the same point in the expression, so the arithmetic is
// unchanged and the digests cannot move. `EpilogueSilu` instantiates it at
// `AccT = float`, where `static_cast<float>` of a `float` is a no-op.
//
// This is #458's shape applied to an activation rather than to an indexing
// helper, and #458's method applies with it: the body below is the one the live
// copies actually carried, so composing them changes no emitted code.
template <typename T>
METAL_FUNC T silu(T x) {
    return static_cast<T>(x / (1 + exp(-x)));
}

// The sigmoid the same two files spell two ways: `unary.metal:308`'s
// `recip(1 + exp(-x))` and `quantized.metal:1503`'s `1.0f / (1.0f + exp(-x))`.
//
// `unary.metal`'s spelling is the one kept, for the reason above -- it is the
// live one, `usigmoid` dispatches through it, and the `quantized.metal` copy is
// dead. #604 measured `recip(1+exp(-x))` bit-identical to `x/(1+exp(-x))`'s
// division form at both dtypes, so the two rearrangements were never the
// divergence; the literal was.
//
// `recip` is inlined here rather than called, because it lives in
// `unary.metal` and a primitive in this file must not depend on a consumer of
// this file. The expansion is written out to be exactly what the original
// composition produced: `recip<T>(v)` is `static_cast<T>(1.0 / v)`, and
// `sigmoid` then wrapped that in a second `static_cast<T>` which is a no-op on
// an expression already of type `T`. `1.0` is a `double` literal, so the
// division is spelled as it was -- narrowing it to `1.0f` or `1` is precisely
// the class of edit #604 measured moving bits, and is not made here.
template <typename T>
METAL_FUNC T sigmoid(T x) {
    return static_cast<T>(static_cast<T>(1.0 / (1 + exp(-x))));
}

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// lloom #735 step 4: the EOS gate. A decode kernel instantiated with function
// constant 777 = true takes one extra argument -- the gate's flag word, bound
// once per encoder at buffer index 30 -- and returns before doing anything
// when it is set: the forwards a no-K decode loop queued past an EOS launch
// their threadgroups but write nothing. The plain instantiation has no such
// argument and no such read. Guarded because the merged translation unit
// (`source.rs`) sees this block through `utils.metal` and again in
// `conv.metal`, which is not a `with_utils!` consumer.
#ifndef LLOOM_EOS_GATE_DECL
#define LLOOM_EOS_GATE_DECL
constant bool eos_gated [[function_constant(777)]];
#define EOS_GATE_ARG device const uint *eos_gate_flag [[buffer(30), function_constant(eos_gated)]]
#define EOS_GATE_CHECK if (eos_gated && eos_gate_flag[0] != 0) { return; }
#endif
