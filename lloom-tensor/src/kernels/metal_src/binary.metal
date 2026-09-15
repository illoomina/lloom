// NOTE: this file does NOT open with `#include <metal_stdlib>`.
// `source.rs` supplies the include prologue and splices `utils.metal`'s
// shared helpers in after it (issue #458), so this file is a FRAGMENT and
// does not compile standalone. `newLibraryWithSource` sees prologue +
// utils + this text as one translation unit.

// Utils
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

// `get_strided_index`, `div_ceil` and `work_per_thread` are shared: they live
// in `utils.metal`, concatenated ahead of this file by `source.rs` (#458).
//
// The indexers below are member templates for the same reason the shared
// `get_strided_index` is templated on `PtrT`: the functor type is a kernel
// template parameter (`l_indexer`, `r_indexer`), so it must not itself be
// parameterised on the address space, or every instantiation row would double.
// Templating `operator()` instead keeps `strided_operand_indexer` one type that
// works with either pointer.
//
// WHY `_operand_` IS IN THESE NAMES (issue #466).
//
// These three are per-OPERAND index POLICIES: `binary_kernel_strided` takes one
// as `l_indexer` and one as `r_indexer`, so a row picks how the left and the
// right operand are addressed independently. They are selected by name in the
// `init_kernel` rows below and appear in no kernel name and in no Rust code.
//
// `reduce.metal` has an unrelated type that was ALSO called `strided_indexer`
// -- a `template<ushort D, typename IndexT, typename PtrT>` with a `<1, ..>`
// partial specialisation, chaining `next` to unroll rank 1..4 (its
// `get_strided_index_t` switch). It takes `operator()(idx)` with dims/strides
// as MEMBERS where the one below takes `operator()(idx, num_dims, dims,
// strides)` and holds no state. That one KEEPS the bare name: three other files
// (`conv`, `gemv`, `utils`) cite it by that name as the precedent for the
// `PtrT` templating idea, so renaming it would falsify comments elsewhere.
//
// Two different designs, one name: not drift, and not deduplicable. In a merged
// translation unit the shared name was `redefinition of 'strided_indexer' as a
// different kind of symbol` -- a struct against a class template -- and it was
// the ONLY remaining elementwise conflict after #458 hoisted the shared helpers.
// All three siblings are renamed rather than just the colliding one, so the
// family keeps one prefix and the next author is not invited to re-collide.

struct cont_operand_indexer {
    template <typename PtrT>
    METAL_FUNC uint operator()(
        uint idx,
        size_t num_dims,
        PtrT dims,
        PtrT strides
    ) {
        return idx;
    }
};

struct strided_operand_indexer {
    template <typename PtrT>
    METAL_FUNC uint operator()(
        uint idx,
        size_t num_dims,
        PtrT dims,
        PtrT strides
    ) {
        return get_strided_index(idx, num_dims, dims, strides);
    }
};

struct scalar_operand_indexer {
    template <typename PtrT>
    METAL_FUNC uint operator()(
        uint idx,
        size_t num_dims,
        PtrT dims,
        PtrT strides
    ) {
        return 0;
    }
};


// Packed parameter blocks.
//
// An ICB command cannot carry an inline constant -- `MTLIndirectComputeCommand`
// has no `setBytes` in any form (`DESIGN.md` §3.7c) -- so every scalar a kernel
// takes as `constant size_t &n` has to arrive in a buffer instead. These are the
// structs the `_packed` entry points read, mirrored by `#[repr(C)]` types in
// `kernels/params.rs` and checked against them by `binary_params_layout`.
//
// `size_t` is 8 bytes in MSL, so these are 8-aligned -- unlike `reduce.metal`'s
// `uint` structs, which are 4-aligned.
//
// `dims` and the two stride arrays are deliberately not fields: their length
// comes from the tensor's layout, not from the struct. They stay separate
// bindings, which an ICB can express -- `setKernelBuffer` binds a buffer of any
// length. Note this family binds *three* such arrays, one more than the others.
struct BinaryParams {
    size_t dim;
};

struct BinaryStridedParams {
    size_t dim;
    size_t num_dims;
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
//
// Note `U` is `bool` for the comparison families (`eq`, `ne`, `le`, `lt`, `ge`,
// `gt`). That is the *output element* type -- a `device U*` binding -- not a
// scalar parameter, so MSL's 1-byte `bool` never reaches a packed struct here.
// The `bool` hazard #38 flags is about `primitive!(bool)` in `EncoderParam`,
// and no kernel in these four files takes a `bool` scalar.
template <typename T, typename U, typename binary, uint W>
METAL_FUNC void binary_body(
    size_t dim,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid
) {
    binary op;
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(op(left[i], right[i]));
    }
}

template <typename T, typename U, typename binary, uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel(
    constant size_t &dim,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[thread_position_in_grid]],
    EOS_GATE_ARG
) {
    EOS_GATE_CHECK
    binary_body<T, U, binary, W>(dim, left, right, output, tid);
}

template <typename T, typename U, typename binary, uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel_packed(
    device const BinaryParams *pp,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[thread_position_in_grid]],
    EOS_GATE_ARG
) {
    EOS_GATE_CHECK
    binary_body<T, U, binary, W>(pp->dim, left, right, output, tid);
}

template <
    typename T,
    typename U,
    typename binary,
    typename l_indexer,
    typename r_indexer,
    uint W,
    typename PtrT>
METAL_FUNC void binary_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT left_strides,
    PtrT right_strides,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid
) {
    binary op;
    l_indexer l_index;
    r_indexer r_index;
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        uint l_idx = l_index(i, num_dims, dims, left_strides);
        uint r_idx = r_index(i, num_dims, dims, right_strides);
        output[i] = static_cast<U>(op(left[l_idx], right[r_idx]));
    }
}

template <
    typename T,
    typename U,
    typename binary,
    typename l_indexer = strided_operand_indexer,
    typename r_indexer = strided_operand_indexer,
    uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *left_strides,
    constant size_t *right_strides,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    binary_strided_body<T, U, binary, l_indexer, r_indexer, W, constant size_t *>(
        dim, num_dims, dims, left_strides, right_strides, left, right, output, tid);
}

template <
    typename T,
    typename U,
    typename binary,
    typename l_indexer = strided_operand_indexer,
    typename r_indexer = strided_operand_indexer,
    uint W = work_per_thread<T>()>
[[kernel]] void binary_kernel_strided_packed(
    device const BinaryStridedParams *pp,
    device const size_t *dims,
    device const size_t *left_strides,
    device const size_t *right_strides,
    device const T *left,
    device const T *right,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    binary_strided_body<T, U, binary, l_indexer, r_indexer, W, device const size_t *>(
        pp->dim, pp->num_dims, dims, left_strides, right_strides, left, right, output, tid);
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
// constant expression. They are reported by `binary_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
static_assert(sizeof(BinaryParams) == 8, "BinaryParams layout");
static_assert(alignof(BinaryParams) == 8, "BinaryParams alignment");

static_assert(sizeof(BinaryStridedParams) == 16, "BinaryStridedParams layout");
static_assert(alignof(BinaryStridedParams) == 8, "BinaryStridedParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void binary_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    BinaryParams        probe_BinaryParams;
    BinaryStridedParams probe_BinaryStridedParams;
    out[0] = sizeof(BinaryParams);
    out[1] = offsetof_rt(BinaryParams, dim);

    out[2] = sizeof(BinaryStridedParams);
    out[3] = offsetof_rt(BinaryStridedParams, dim);
    out[4] = offsetof_rt(BinaryStridedParams, num_dims);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// Both binding styles from one instantiation row, so a variant cannot exist in
// one style and not the other. `_packed` is a name segment appended after the
// dtype and any indexer suffix, and `packed_names_resolve` checks every result
// against the compiled library rather than against this macro -- which is
// `DESIGN.md` §8.1b's argument, and #26 shipped 48 names absent from a metallib
// that compiled cleanly.
//
// This family is the widest of the four: nine indexer combinations per dtype,
// each now in two styles.
#define init_binary_k(op_name, binary_op, tname, t, u)                                                                                    \
    init_kernel(#op_name "_" #tname, binary_kernel, t, u, binary_op)                                                                      \
    init_kernel(#op_name "_" #tname "_packed", binary_kernel_packed, t, u, binary_op)                                                     \
    init_kernel(#op_name "_" #tname "_strided", binary_kernel_strided, t, u, binary_op)                                                   \
    init_kernel(#op_name "_" #tname "_strided_packed", binary_kernel_strided_packed, t, u, binary_op)                                     \
    init_kernel(#op_name "_" #tname "_lstrided", binary_kernel_strided, t, u, binary_op, strided_operand_indexer, cont_operand_indexer)                   \
    init_kernel(#op_name "_" #tname "_lstrided_packed", binary_kernel_strided_packed, t, u, binary_op, strided_operand_indexer, cont_operand_indexer)     \
    init_kernel(#op_name "_" #tname "_rstrided", binary_kernel_strided, t, u, binary_op, cont_operand_indexer, strided_operand_indexer)                   \
    init_kernel(#op_name "_" #tname "_rstrided_packed", binary_kernel_strided_packed, t, u, binary_op, cont_operand_indexer, strided_operand_indexer)     \
    init_kernel(#op_name "_" #tname "_scalar", binary_kernel_strided, t, u, binary_op, scalar_operand_indexer, scalar_operand_indexer)                    \
    init_kernel(#op_name "_" #tname "_scalar_packed", binary_kernel_strided_packed, t, u, binary_op, scalar_operand_indexer, scalar_operand_indexer)      \
    init_kernel(#op_name "_" #tname "_cs", binary_kernel_strided, t, u, binary_op, cont_operand_indexer, scalar_operand_indexer)                          \
    init_kernel(#op_name "_" #tname "_cs_packed", binary_kernel_strided_packed, t, u, binary_op, cont_operand_indexer, scalar_operand_indexer)            \
    init_kernel(#op_name "_" #tname "_sc", binary_kernel_strided, t, u, binary_op, scalar_operand_indexer, cont_operand_indexer)                          \
    init_kernel(#op_name "_" #tname "_sc_packed", binary_kernel_strided_packed, t, u, binary_op, scalar_operand_indexer, cont_operand_indexer)            \
    init_kernel(#op_name "_" #tname "_rss", binary_kernel_strided, t, u, binary_op, scalar_operand_indexer, strided_operand_indexer)                      \
    init_kernel(#op_name "_" #tname "_rss_packed", binary_kernel_strided_packed, t, u, binary_op, scalar_operand_indexer, strided_operand_indexer)        \
    init_kernel(#op_name "_" #tname "_lss", binary_kernel_strided, t, u, binary_op, strided_operand_indexer, scalar_operand_indexer)                      \
    init_kernel(#op_name "_" #tname "_lss_packed", binary_kernel_strided_packed, t, u, binary_op, strided_operand_indexer, scalar_operand_indexer)

#if defined(__HAVE_BFLOAT__)
#define init_binary(bop)                            \
    init_binary_k(bop, bop, f32, float, float)      \
    init_binary_k(bop, bop, f16, half, half)        \
    init_binary_k(bop, bop, bf16, bfloat, bfloat)   \
    init_binary_k(bop, bop, u8, uint8_t, uint8_t)   \
    init_binary_k(bop, bop, u32, uint32_t, uint32_t)\
    init_binary_k(bop, bop, i64, int64_t, int64_t)
#else
#define init_binary(bop)                                                       \
    init_binary_k(bop, bop, f32, float, float)      \
    init_binary_k(bop, bop, f16, half, half)        \
    init_binary_k(bop, bop, u8, uint8_t, uint8_t)   \
    init_binary_k(bop, bop, u32, uint32_t, uint32_t)\
    init_binary_k(bop, bop, i64, int64_t, int64_t)
#endif

#if defined(__HAVE_BFLOAT__)
#define init_boolean_binary(op_name, binary_op)             \
    init_binary_k(op_name, binary_op, f32, float, bool)     \
    init_binary_k(op_name, binary_op, f16, half, bool)      \
    init_binary_k(op_name, binary_op, bf16, bfloat, bool)   \
    init_binary_k(op_name, binary_op, u8, uint8_t, bool)    \
    init_binary_k(op_name, binary_op, u32, uint32_t, bool)  \
    init_binary_k(op_name, binary_op, i64, int64_t, bool)
#else
#define init_boolean_binary(op_name, binary_op)             \
    init_binary_k(op_name, binary_op, f32, float, bool)     \
    init_binary_k(op_name, binary_op, f16, half, bool)      \
    init_binary_k(op_name, binary_op, u8, uint8_t, bool)    \
    init_binary_k(op_name, binary_op, u32, uint32_t, bool)  \
    init_binary_k(op_name, binary_op, i64, int64_t, bool)
#endif

// Define binary ops
#define define_binary_op(name, op)      \
struct name {                           \
    template <typename T>               \
    METAL_FUNC T operator()(T x, T y) { \
        return static_cast<T>(op);      \
    }                                   \
};
#define define_binary_bool_op(name, op)     \
struct name {                               \
    template <typename T>                   \
    METAL_FUNC bool operator()(T x, T y) {  \
        return op;                          \
    }                                       \
};

// Define binary ops
define_binary_op(badd, x + y);
define_binary_op(bsub, x - y);
define_binary_op(bmul, x * y);
define_binary_op(bdiv, x / y);
define_binary_op(bminimum, MIN(x, y));
define_binary_op(bmaximum, MAX(x, y));

// Define binary ops that return a bool
define_binary_bool_op(beq, x == y);
define_binary_bool_op(bne, x != y);
define_binary_bool_op(ble, x <= y);
define_binary_bool_op(blt, x < y);
define_binary_bool_op(bge, x >= y);
define_binary_bool_op(bgt, x > y)

// Initialize kernels
init_binary(badd);
init_binary(bsub);
init_binary(bmul);
init_binary(bdiv);
init_binary(bminimum);
init_binary(bmaximum);

init_boolean_binary(eq, beq);
init_boolean_binary(ne, bne);
init_boolean_binary(le, ble);
init_boolean_binary(lt, blt);
init_boolean_binary(ge, bge);
init_boolean_binary(gt, bgt);
