// NOTE: this file does NOT open with `#include <metal_stdlib>`.
// `source.rs` supplies the include prologue and splices `utils.metal`'s
// shared helpers in after it (issue #458), so this file is a FRAGMENT and
// does not compile standalone. `newLibraryWithSource` sees prologue +
// utils + this text as one translation unit.

// `get_strided_index`, `div_ceil` and `work_per_thread` are shared: they live
// in `utils.metal`, concatenated ahead of this file by `source.rs` (#458).

// Packed parameter blocks.
//
// An ICB command cannot carry an inline constant -- `MTLIndirectComputeCommand`
// has no `setBytes` in any form (`DESIGN.md` §3.7c) -- so every scalar a kernel
// takes as `constant size_t &n` has to arrive in a buffer instead. These are the
// structs the `_packed` entry points read, mirrored by `#[repr(C)]` types in
// `kernels/params.rs` and checked against them by `elementwise_params_layout`.
//
// `size_t` is 8 bytes in MSL, so these are 8-aligned -- unlike `reduce.metal`'s
// `uint` structs, which are 4-aligned. That difference is the reason the layout
// check ships the numbers across the boundary rather than trusting either side.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length.
struct CastParams {
    size_t dim;
};

struct CastStridedParams {
    size_t dim;
    size_t num_dims;
};

// Kernels
//
// One body per kernel, two entry points around it. The classical wrapper binds
// its scalars with `setBytes` exactly as before; the `_packed` one reads them
// from a single `device const Params*`. Neither the arithmetic nor the loop
// structure is duplicated, so the two styles cannot compute different things --
// which is what makes the bit-identical parity test meaningful rather than
// merely reassuring.
//
// Unlike `reduce.metal`, nothing here declares threadgroup memory, so the
// whole-body-plus-two-thin-wrappers factoring that #38 could not use does work
// in this file. (MSL permits a `threadgroup` variable only inside a
// `[[kernel]]`-qualified function; there is no such variable in any of these
// four families.)
template <typename T, typename U, typename IR, int W>
METAL_FUNC void cast_body(
    size_t dim,
    device const T* input,
    device U* output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(static_cast<IR>(input[i]));
    }
}

template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    cast_body<T, U, IR, W>(dim, input, output, tid);
}

template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel_packed(
    device const CastParams *pp,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    cast_body<T, U, IR, W>(pp->dim, input, output, tid);
}

// `InPtrT` is templated alongside `PtrT` so the classical wrapper keeps its
// `constant const T *input` exactly as it was. That address space is not
// incidental -- changing it would be a behaviour change to the classical path
// smuggled in beside a binding change, and the acceptance bar for this work is
// that the classical path is untouched.
template <typename T, typename U, typename IR, typename PtrT, typename InPtrT>
METAL_FUNC void cast_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    InPtrT input,
    device U *output,
    uint tid
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        static_cast<IR>(input[get_strided_index(tid, num_dims, dims, strides)])
    );
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    cast_strided_body<T, U, IR, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, input, output, tid);
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided_packed(
    device const CastStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    cast_strided_body<T, U, IR, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, input, output, tid);
}

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `cast_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
//
// `size_t` is 8 bytes here, hence 8-aligned. That is the substantive difference
// from `reduce.metal`, whose packed structs are all 4-aligned `uint`.
static_assert(sizeof(CastParams) == 8, "CastParams layout");
static_assert(alignof(CastParams) == 8, "CastParams alignment");

static_assert(sizeof(CastStridedParams) == 16, "CastStridedParams layout");
static_assert(alignof(CastStridedParams) == 8, "CastStridedParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void cast_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    CastParams        probe_CastParams;
    CastStridedParams probe_CastStridedParams;
    out[0] = sizeof(CastParams);
    out[1] = offsetof_rt(CastParams, dim);

    out[2] = sizeof(CastStridedParams);
    out[3] = offsetof_rt(CastStridedParams, dim);
    out[4] = offsetof_rt(CastStridedParams, num_dims);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both binding styles from one instantiation row, so a variant cannot exist in
// one style and not the other. `_packed` is a name segment appended after the
// dtype and any `_strided`, and `packed_names_resolve` checks every result
// against the compiled library rather than against this macro -- which is
// `DESIGN.md` §8.1b's argument, and #26 shipped 48 names absent from a metallib
// that compiled cleanly.
#define init_cast(tname, t, uname, u)                                                          \
    init_kernel("cast_" #tname "_" #uname, cast_kernel, t, u)                                  \
    init_kernel("cast_" #tname "_" #uname "_packed", cast_kernel_packed, t, u)                 \
    init_kernel("cast_" #tname "_" #uname "_strided", cast_kernel_strided, t, u)               \
    init_kernel("cast_" #tname "_" #uname "_strided_packed", cast_kernel_strided_packed, t, u)

#if defined(__HAVE_BFLOAT__)
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, bf16, bfloat)   \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#else
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#endif


init_cast_all(f32, float);
init_cast_all(f16, half);
#if defined(__HAVE_BFLOAT__)
init_cast_all(bf16, bfloat);
#endif
init_cast_all(i64, int64_t);
init_cast_all(u32, uint32_t);
init_cast_all(u8, uint8_t);
