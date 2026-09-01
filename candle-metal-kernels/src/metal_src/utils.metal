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
