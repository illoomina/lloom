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
// One conversion expression for both index widths and binding styles.
// The default observer is empty and inlined away. The counted observer/entry
// exists ONLY when a native test compiles this same source with the macro.
template <typename T, typename U, typename IR>
METAL_FUNC U cast_value(T value) {
    return static_cast<U>(static_cast<IR>(value));
}
struct CastNoWriteCount {
    METAL_FUNC void written(size_t) const {}
};

template <typename T, typename U, typename IR, int W, typename Observer = CastNoWriteCount>
METAL_FUNC void cast_body(
    size_t dim,
    device const T* input,
    device U* output,
    uint tid,
    Observer observer = Observer{}
) {
    const uint step = div_ceil<W>(dim);
    // The host grid rounds the launch up to a complete final threadgroup.
    // Its padding threads must not repeat another thread's next iteration.
    if (tid >= step) return;
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = cast_value<T, U, IR>(input[i]);
        observer.written(i);
    }
}

// Wide entries reconstruct the linear index from group coordinates, avoiding
// the uint thread_position_in_grid truncation. Padding groups/lanes return
// before any unchecked product/addition. The host describes a 1/2/3-D grid.
METAL_FUNC bool cast_wide_thread(
    size_t work, uint3 group, uint3 groups, uint lane, uint width,
    thread size_t& tid
) {
    const size_t max_index = ~size_t(0);
    if (groups.x == 0 || groups.y == 0 || width == 0) return false;
    if (size_t(group.z) > (max_index - group.y) / groups.y) return false;
    const size_t yz = size_t(group.z) * groups.y + group.y;
    if (yz > (max_index - group.x) / groups.x) return false;
    const size_t linear_group = yz * groups.x + group.x;
    const size_t needed_groups = work / width + (work % width != 0);
    if (linear_group >= needed_groups) return false;
    const size_t first = linear_group * width; // first < work, no wrap
    if (lane >= work - first) return false;
    tid = first + lane;
    return true;
}

template <typename T, typename U, typename IR, int W>
METAL_FUNC void cast_wide_body(
    size_t dim, device const T* input, device U* output,
    uint3 group, uint3 groups, uint lane, uint width
) {
    const size_t step = dim / W + (dim % W != 0);
    size_t tid;
    if (!cast_wide_thread(step, group, groups, lane, width, tid)) return;
    for (size_t i = tid; i < dim;) {
        output[i] = cast_value<T, U, IR>(input[i]);
        // Unlike uint's legacy step, this also remains defined at SIZE_MAX.
        if (dim - i <= step) break;
        i += step;
    }
}

template <typename T, typename U, typename IR = T, int W = work_per_thread<T>()>
[[kernel]] void cast_kernel_wide(
    constant size_t& dim, device const T* input, device U* output,
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint3 width [[threads_per_threadgroup]]
) {
    cast_wide_body<T,U,IR,W>(dim,input,output,group,groups,lane,width.x);
}

template <typename T, typename U, typename IR = T, int W = work_per_thread<T>()>
[[kernel]] void cast_kernel_wide_packed(
    device const CastParams* pp, device const T* input, device U* output,
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint3 width [[threads_per_threadgroup]]
) {
    cast_wide_body<T,U,IR,W>(pp->dim,input,output,group,groups,lane,width.x);
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
    output[tid] = cast_value<T, U, IR>(input[get_strided_index(tid, num_dims, dims, strides)]);
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

// Preserve arbitrary supported strides without uint accumulator narrowing.
// Positive logical dimensions are checked by the Rust descriptor before encode.
template <typename PtrT>
METAL_FUNC size_t cast_strided_index_wide(size_t index, size_t rank, PtrT dims, PtrT strides) {
    size_t result = 0;
    for (size_t d = rank; d != 0; --d) {
        const size_t axis = d - 1;
        result += (index % dims[axis]) * strides[axis];
        index /= dims[axis];
    }
    return result;
}

template <typename T, typename U, typename IR, typename PtrT, typename InPtrT>
METAL_FUNC void cast_strided_wide_body(
    size_t dim, size_t rank, PtrT dims, PtrT strides, InPtrT input, device U* output,
    uint3 group, uint3 groups, uint lane, uint width
) {
    size_t tid;
    if (!cast_wide_thread(dim,group,groups,lane,width,tid)) return;
    output[tid] = cast_value<T,U,IR>(input[cast_strided_index_wide(tid,rank,dims,strides)]);
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided_wide(
    constant size_t& dim, constant size_t& rank,
    constant size_t* dims, constant size_t* strides, constant const T* input, device U* output,
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint3 width [[threads_per_threadgroup]]
) {
    cast_strided_wide_body<T,U,IR,constant size_t*,constant const T*>(
        dim,rank,dims,strides,input,output,group,groups,lane,width.x);
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided_wide_packed(
    device const CastStridedParams* pp,
    device const size_t* dims, device const size_t* strides, constant const T* input, device U* output,
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint3 width [[threads_per_threadgroup]]
) {
    cast_strided_wide_body<T,U,IR,device const size_t*,constant const T*>(
        pp->dim,pp->num_dims,dims,strides,input,output,group,groups,lane,width.x);
}

#ifdef LLOOM_CAST_WRITE_COUNTS
// Native causal test ONLY. Production source never defines the macro, so no
// test kernel/atomic instrumentation is included in the shipped library.
struct CastWriteCounts {
    device atomic_uint* counts;
    METAL_FUNC void written(size_t i) const {
        atomic_fetch_add_explicit(counts + i, 1u, memory_order_relaxed);
    }
};
[[kernel]] void cast_count_writers(
    constant size_t& dim, device const float* input, device float* output,
    device atomic_uint* counts, uint tid [[thread_position_in_grid]]
) {
    cast_body<float,float,float,2,CastWriteCounts>(dim,input,output,tid,CastWriteCounts{counts});
}
#endif

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
    init_kernel("cast_" #tname "_" #uname "_strided_packed", cast_kernel_strided_packed, t, u)   \
    init_kernel("cast_" #tname "_" #uname "_wide", cast_kernel_wide, t, u)                      \
    init_kernel("cast_" #tname "_" #uname "_wide_packed", cast_kernel_wide_packed, t, u)        \
    init_kernel("cast_" #tname "_" #uname "_strided_wide", cast_kernel_strided_wide, t, u)      \
    init_kernel("cast_" #tname "_" #uname "_strided_wide_packed", cast_kernel_strided_wide_packed, t, u)

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
