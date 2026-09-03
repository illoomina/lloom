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
// `kernels/params.rs` and checked against them by `affine_params_layout`.
//
// **This file is the one where the padding is not trivial, and it is why the
// cross-boundary check earns its keep here more than anywhere else.** These
// structs mix 8-byte `size_t` with 4-byte `float`, so they carry interior and
// trailing padding that neither side can be assumed to get right by inspection:
// `{size_t, float, float}` is 16 bytes, not 12, and `{size_t, size_t, float}`
// is 24, not 20. `reduce.metal`'s structs were uniform (all `uint`, or all
// `size_t`) and had none of this.
//
// Field order is the `set_params!` order at the call site, because that is the
// order the capture appends in. It is not a free choice.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length.
struct AffineParams {
    size_t dim;
    float mul;
    float add;
};

struct AffineStridedParams {
    size_t dim;
    size_t num_dims;
    float mul;
    float add;
};

// `powf` and `elu` bind one float where `affine` binds two, so they need their
// own structs rather than sharing `AffineParams` with a dead field: a shared
// struct would ship 16 bytes where the capture produces 12, and every field
// after the first would be read from the wrong place.
struct ScaleParams {
    size_t dim;
    float mul;
};

struct ScaleStridedParams {
    size_t dim;
    size_t num_dims;
    float mul;
};

// Kernels
//
// One body per kernel, two entry points around it. The classical wrapper binds
// its scalars with `setBytes` exactly as before; the `_packed` one reads them
// from a single `device const Params*`. Neither the arithmetic nor the loop
// structure is duplicated, so the two styles cannot compute different things.
//
// Unlike `reduce.metal`, nothing here declares threadgroup memory, so the
// whole-body-plus-two-thin-wrappers factoring that #38 could not use does work
// in this file.
template <typename T, int W>
METAL_FUNC void affine_body(
    size_t dim,
    float mul,
    float add,
    device const T *input,
    device T *output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<T>(fma(float(input[i]), mul, add));
    }
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void affine_kernel(
    constant size_t &dim,
    constant float &mul,
    constant float &add,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    affine_body<T, W>(dim, mul, add, input, output, tid);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void affine_kernel_packed(
    device const AffineParams *pp,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    affine_body<T, W>(pp->dim, pp->mul, pp->add, input, output, tid);
}

// `InPtrT` is templated alongside `PtrT` so the classical wrapper keeps its
// `constant const T *input` exactly as it was. That address space is not
// incidental -- changing it would be a behaviour change to the classical path
// smuggled in beside a binding change.
template <typename T, typename PtrT, typename InPtrT>
METAL_FUNC void affine_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    float mul,
    float add,
    InPtrT input,
    device T *output,
    uint tid
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    float result = fma(float(input[idx]), mul, add);
    output[tid] = static_cast<T>(result);
}

template <typename T>
[[kernel]] void affine_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant float &add,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    affine_strided_body<T, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, mul, add, input, output, tid);
}

template <typename T>
[[kernel]] void affine_kernel_strided_packed(
    device const AffineStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    affine_strided_body<T, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, pp->mul, pp->add, input, output, tid);
}

template <typename T, int W>
METAL_FUNC void powf_body(
    size_t dim,
    float mul,
    device const T *input,
    device T *output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<T>(pow(static_cast<float>(input[i]), mul));
    }
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void powf_kernel(
    constant size_t &dim,
    constant float &mul,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    powf_body<T, W>(dim, mul, input, output, tid);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void powf_kernel_packed(
    device const ScaleParams *pp,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    powf_body<T, W>(pp->dim, pp->mul, input, output, tid);
}

template <typename T, typename PtrT, typename InPtrT>
METAL_FUNC void powf_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    float mul,
    InPtrT input,
    device T *output,
    uint tid
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[tid] = static_cast<T>(pow(static_cast<float>(input[idx]), mul));
}

template <typename T>
[[kernel]] void powf_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    powf_strided_body<T, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, mul, input, output, tid);
}

template <typename T>
[[kernel]] void powf_kernel_strided_packed(
    device const ScaleStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    powf_strided_body<T, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, pp->mul, input, output, tid);
}

template <typename T, int W>
METAL_FUNC void elu_body(
    size_t dim,
    float mul,
    device const T *input,
    device T *output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        const T x = input[i];
        output[i] = static_cast<T>((x > 0) ? x : mul * (exp(x) - 1));
    }
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void elu_kernel(
    constant size_t &dim,
    constant float &mul,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    elu_body<T, W>(dim, mul, input, output, tid);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void elu_kernel_packed(
    device const ScaleParams *pp,
    device const T *input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    elu_body<T, W>(pp->dim, pp->mul, input, output, tid);
}

template <typename T, typename PtrT, typename InPtrT>
METAL_FUNC void elu_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    float mul,
    InPtrT input,
    device T *output,
    uint tid
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    const T x = input[idx];
    output[tid] = static_cast<T>((x > 0) ? x : mul * (exp(x) - 1));
}

template <typename T>
[[kernel]] void elu_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant float &mul,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    elu_strided_body<T, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, mul, input, output, tid);
}

template <typename T>
[[kernel]] void elu_kernel_strided_packed(
    device const ScaleStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    elu_strided_body<T, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, pp->mul, input, output, tid);
}

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// **These are the padded ones.** `{size_t, float, float}` is 16 bytes rather
// than the 12 its fields sum to, and `{size_t, size_t, float}` is 24 rather
// than 20 -- trailing padding to the struct's own 8-byte alignment. Getting
// that wrong on either side is exactly the silent case, and it is why the
// packed block is padded to `align_of` at capture close rather than shipped at
// its natural length.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `affine_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
static_assert(sizeof(AffineParams) == 16, "AffineParams layout");
static_assert(alignof(AffineParams) == 8, "AffineParams alignment");

static_assert(sizeof(AffineStridedParams) == 24, "AffineStridedParams layout");
static_assert(alignof(AffineStridedParams) == 8, "AffineStridedParams alignment");

static_assert(sizeof(ScaleParams) == 16, "ScaleParams layout");
static_assert(alignof(ScaleParams) == 8, "ScaleParams alignment");

static_assert(sizeof(ScaleStridedParams) == 24, "ScaleStridedParams layout");
static_assert(alignof(ScaleStridedParams) == 8, "ScaleStridedParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void affine_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    AffineParams        probe_AffineParams;
    AffineStridedParams probe_AffineStridedParams;
    ScaleParams         probe_ScaleParams;
    ScaleStridedParams  probe_ScaleStridedParams;
    out[ 0] = sizeof(AffineParams);
    out[ 1] = offsetof_rt(AffineParams, dim);
    out[ 2] = offsetof_rt(AffineParams, mul);
    out[ 3] = offsetof_rt(AffineParams, add);

    out[ 4] = sizeof(AffineStridedParams);
    out[ 5] = offsetof_rt(AffineStridedParams, dim);
    out[ 6] = offsetof_rt(AffineStridedParams, num_dims);
    out[ 7] = offsetof_rt(AffineStridedParams, mul);
    out[ 8] = offsetof_rt(AffineStridedParams, add);

    out[ 9] = sizeof(ScaleParams);
    out[10] = offsetof_rt(ScaleParams, dim);
    out[11] = offsetof_rt(ScaleParams, mul);

    out[12] = sizeof(ScaleStridedParams);
    out[13] = offsetof_rt(ScaleStridedParams, dim);
    out[14] = offsetof_rt(ScaleStridedParams, num_dims);
    out[15] = offsetof_rt(ScaleStridedParams, mul);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both binding styles from one instantiation row, so a variant cannot exist in
// one style and not the other. `_packed` is a name segment appended after the
// dtype and any `_strided`, and `packed_names_resolve` checks every result
// against the compiled library rather than against these macros -- which is
// `DESIGN.md` §8.1b's argument, and #26 shipped 48 names absent from a metallib
// that compiled cleanly.
#define init_affine(tname, t)                                                                \
    init_kernel("affine_" #tname, affine_kernel, t)                                          \
    init_kernel("affine_" #tname "_packed", affine_kernel_packed, t)                         \
    init_kernel("affine_" #tname "_strided", affine_kernel_strided, t)                       \
    init_kernel("affine_" #tname "_strided_packed", affine_kernel_strided_packed, t)

#define init_powf(tname, t)                                                              \
    init_kernel("powf_" #tname, powf_kernel, t)                                          \
    init_kernel("powf_" #tname "_packed", powf_kernel_packed, t)                         \
    init_kernel("powf_" #tname "_strided", powf_kernel_strided, t)                       \
    init_kernel("powf_" #tname "_strided_packed", powf_kernel_strided_packed, t)

#define init_elu(tname, t)                                                           \
    init_kernel("elu_" #tname, elu_kernel, t)                                        \
    init_kernel("elu_" #tname "_packed", elu_kernel_packed, t)                       \
    init_kernel("elu_" #tname "_strided", elu_kernel_strided, t)                     \
    init_kernel("elu_" #tname "_strided_packed", elu_kernel_strided_packed, t)


init_affine(u8, uint8_t);
init_affine(u32, uint32_t);
init_affine(i64, int64_t);
init_affine(f32, float);
init_affine(f16, half);

init_powf(f32, float);
init_powf(f16, half);

init_elu(f32, float);
init_elu(f16, half);

#if defined(__HAVE_BFLOAT__)
init_affine(bf16, bfloat);
init_powf(bf16, bfloat);
init_elu(bf16, bfloat);
#endif
