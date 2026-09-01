// NOTE: this file does NOT open with `#include <metal_stdlib>`.
// `source.rs` supplies the include prologue and splices `utils.metal`'s
// shared helpers in after it (issue #458), so this file is a FRAGMENT and
// does not compile standalone. `newLibraryWithSource` sees prologue +
// utils + this text as one translation unit.

constant bool IDS_CONTIGUOUS [[function_constant(0)]];
constant bool T_CONTIGUOUS [[function_constant(1)]];
constant bool F_CONTIGUOUS [[function_constant(2)]];


// `get_strided_index_unrolled`, `div_ceil` and `work_per_thread` are shared:
// they live in `utils.metal`, concatenated ahead of this file by `source.rs`
// (#458).
//
// This file calls the `_unrolled` form deliberately. Its private copy of the
// helper carried `#pragma clang loop unroll(full)` where the other five files'
// copies did not, and measured, that pragma changes `where_u8_f16`'s emitted
// code -- at an unchanged section size, so only a content hash sees it. Calling
// the un-pragma'd form here would be a silent codegen change.


template<typename T, typename ID, uint W = work_per_thread<T>()>
METAL_FUNC void where_cond(
    constant size_t &numel,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant size_t *strides_t,
    constant size_t *strides_f,
    device const ID *ids,
    device const T *t,
    device const T *f,
    device T *out,
    uint tid [[ thread_position_in_grid ]]
) {
    uint idx = 0;
    uint t_idx = 0;
    uint f_idx = 0;

    const uint step = div_ceil<W>(numel);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < numel; i += step) {
        if (IDS_CONTIGUOUS) {
            idx = i;
        } else {
            idx = get_strided_index_unrolled(i, num_dims, dims, strides);
        }
        if (T_CONTIGUOUS) {
            t_idx = i;
        } else {
            t_idx = get_strided_index_unrolled(i, num_dims, dims, strides_t);
        }
        if (F_CONTIGUOUS) {
            f_idx = i;
        } else {
            f_idx = get_strided_index_unrolled(i, num_dims, dims, strides_f);
        }
        out[i] = select(f[f_idx], t[t_idx], ids[idx]);
    }

}

#define WHERE_OP(T, ID, FN_NAME)                                                                \
kernel void FN_NAME(                                                                            \
    constant size_t &numel,                                                                     \
    constant size_t &num_dims,                                                                  \
    constant size_t *dims,                                                                      \
    constant size_t *strides,                                                                   \
    constant size_t *strides_t,                                                                 \
    constant size_t *strides_f,                                                                 \
    device const ID *ids,                                                                       \
    device const T *t,                                                                          \
    device const T *f,                                                                          \
    device T *out,                                                                              \
    uint i [[ thread_position_in_grid ]]                                                        \
) {                                                                                             \
   where_cond<T, ID>(numel, num_dims, dims, strides, strides_t, strides_f, ids, t, f, out, i);  \
}                                                                                               \

WHERE_OP(half, uint32_t, where_u32_f16)
WHERE_OP(float, uint32_t, where_u32_f32)
WHERE_OP(uint8_t, uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)

WHERE_OP(half, uint8_t, where_u8_f16)
WHERE_OP(float, uint8_t, where_u8_f32)
WHERE_OP(uint8_t, uint8_t, where_u8_u8)
WHERE_OP(uint32_t, uint8_t, where_u8_u32)

#if __METAL_VERSION__ >= 220
WHERE_OP(int64_t, uint8_t, where_u8_i64)
WHERE_OP(int64_t, uint32_t, where_u32_i64)

WHERE_OP(half, int64_t, where_i64_f16)
WHERE_OP(float, int64_t, where_i64_f32)
WHERE_OP(uint8_t, int64_t, where_i64_u8)
WHERE_OP(uint32_t, int64_t, where_i64_u32)
WHERE_OP(int64_t, int64_t, where_i64_i64)
#if defined(__HAVE_BFLOAT__)
WHERE_OP(bfloat, int64_t, where_i64_bf16)
#endif
#endif

#if defined(__HAVE_BFLOAT__)
WHERE_OP(bfloat, uint8_t, where_u8_bf16)
WHERE_OP(bfloat, uint32_t, where_u32_bf16)
#endif
