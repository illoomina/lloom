// RoPE: the three rotary-position-embedding entry points (rope, rope_i,
// rope_thd), split out of reduce.metal into its own translation unit (§7.2
// role directory). This fragment references no reduction preamble and no
// `utils.metal` helper, so it compiles from its own header alone.
//
// Each kernel has a classical form binding scalar `constant &` params and a
// `_packed` form binding a single struct (#38). `size_t` is 8 bytes in MSL, so
// the packed structs are 8-aligned where the reductions' `uint` params are
// 4-aligned; the Rust mirrors in `kernels/params.rs` are `u64` for the same
// reason and are checked field-for-field by `rope_params_layout_matches_metal`.

#include <metal_stdlib>

using namespace metal;

// The registration macro: a `[[host_name]]` template instantiation for one
// (name, dtype). Copied here because this TU no longer shares reduce.metal's
// preamble.
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

template<typename T>
METAL_FUNC void ropei(
    const size_t bh,
    const size_t td,
    const size_t stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid
) {
    if (2 * tid >= bh * td) {
        return;
    }
    size_t rope_idx = tid % (td / 2);
    if (stride_b > 0) {
      size_t b_idx = (2 * tid) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];
    dst[2 * tid] = src[2 * tid] * c - src[2 * tid + 1] * s;
    dst[2 * tid + 1] = src[2 * tid] * s + src[2 * tid + 1] * c;
}

template<typename T>
METAL_FUNC void rope(
    const size_t bh,
    const size_t td,
    const size_t d,
    const size_t stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= bh * td) {
        return;
    }
    size_t i_bh = idx / (td / 2);
    size_t i_td = idx - (td / 2) * i_bh;
    size_t i_t = i_td / (d / 2);
    size_t i_d = i_td - (d / 2) * i_t;
    size_t i1 = i_bh * td + i_t * d + i_d;
    size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template<typename T>
METAL_FUNC void rope_thd(
    const size_t b,
    const size_t t,
    const size_t h,
    const size_t d,
    const size_t stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx
) {
    if (2 * idx >= b * t * h * d) {
        return;
    }
    const size_t i_bth = idx / (d / 2);
    const size_t i_d = idx - (d / 2) * i_bth;
    const size_t i_t = (i_bth / h) % t;
    const size_t i1 = i_bth * d + i_d;
    const size_t i2 = i1 + d / 2;
    size_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      const size_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];
    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

// RoPE binds `size_t`, which is **8 bytes** in MSL where the reductions' `uint`
// is 4. The struct keeps `size_t` rather than narrowing to `uint`, so the
// packed form binds exactly the values the classical form does and the
// conversion stays a binding-style change with no numeric range change. The
// Rust mirror is `u64` for the same reason, and the sizes are asserted against
// each other — this is precisely the over-alignment class #38 warns about, and
// silently narrowing here would be a wrong answer rather than a crash.
struct RopeIParams {
    size_t bh;
    size_t td;
    size_t stride_b;
};

struct RopeParams {
    size_t bh;
    size_t td;
    size_t d;
    size_t stride_b;
};

struct RopeThdParams {
    size_t b;
    size_t t;
    size_t h;
    size_t d;
    size_t stride_b;
};

template<typename T>
[[kernel]] void rope_i_kernel(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
    ropei<T>(bh, td, stride_b, src, cos, sin, dst, tid);
}

template<typename T>
[[kernel]] void rope_i_kernel_packed(
    device const RopeIParams *pp,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint tid [[ thread_position_in_grid ]]
) {
    RopeIParams p = *pp;
    ropei<T>(p.bh, p.td, p.stride_b, src, cos, sin, dst, tid);
}

template<typename T>
[[kernel]] void rope_kernel(
    constant size_t &bh,
    constant size_t &td,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    rope<T>(bh, td, d, stride_b, src, cos, sin, dst, idx);
}

template<typename T>
[[kernel]] void rope_kernel_packed(
    device const RopeParams *pp,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    RopeParams p = *pp;
    rope<T>(p.bh, p.td, p.d, p.stride_b, src, cos, sin, dst, idx);
}

template<typename T>
[[kernel]] void rope_thd_kernel(
    constant size_t &b,
    constant size_t &t,
    constant size_t &h,
    constant size_t &d,
    constant size_t &stride_b,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    rope_thd<T>(b, t, h, d, stride_b, src, cos, sin, dst, idx);
}

template<typename T>
[[kernel]] void rope_thd_kernel_packed(
    device const RopeThdParams *pp,
    device const T *src,
    device const T *cos,
    device const T *sin,
    device T *dst,
    uint idx [[ thread_position_in_grid ]]
) {
    RopeThdParams p = *pp;
    rope_thd<T>(p.b, p.t, p.h, p.d, p.stride_b, src, cos, sin, dst, idx);
}

// The three variants a single `ROPE(...)` used to generate. Note the name
// shapes differ between them — `rope_f32`, `rope_i_f32`, `rope_thd_f32` — so
// the dtype suffix is not a common tail and each is spelled out.
#define init_rope(tname, t)                                                 \
    init_kernel("rope_" #tname, rope_kernel, t)                             \
    init_kernel("rope_" #tname "_packed", rope_kernel_packed, t)            \
    init_kernel("rope_i_" #tname, rope_i_kernel, t)                         \
    init_kernel("rope_i_" #tname "_packed", rope_i_kernel_packed, t)        \
    init_kernel("rope_thd_" #tname, rope_thd_kernel, t)                     \
    init_kernel("rope_thd_" #tname "_packed", rope_thd_kernel_packed, t)

// Layout, asserted on the device side.
//
// A packed parameter at the wrong offset is silent corruption, not a crash
// (`DESIGN.md` §3.5, §15.1): the kernel reads a well-formed number from the
// wrong place and produces a plausible wrong answer. The layout is pinned here
// in the language that decides it and again in `kernels/params.rs`;
// `rope_params_layout_matches_metal` dispatches `rope_params_layout` to write
// these values into a buffer and compares them against Rust's own
// `size_of`/`offset_of`, so the two sides are checked against each other.
//
// `size_t` is 8 bytes here, so these structs are 8-aligned. Only sizes and
// alignments are `static_assert`ed; offsets are reported by the kernel below,
// since MSL's null-pointer `offsetof` is not a constant expression.
static_assert(sizeof(RopeIParams) == 24, "RopeIParams layout");
static_assert(alignof(RopeIParams) == 8, "RopeIParams alignment");

static_assert(sizeof(RopeParams) == 32, "RopeParams layout");
static_assert(alignof(RopeParams) == 8, "RopeParams alignment");

static_assert(sizeof(RopeThdParams) == 40, "RopeThdParams layout");
static_assert(alignof(RopeThdParams) == 8, "RopeThdParams alignment");

// Reports each struct's size and every field's offset as the compiled kernel
// sees them, for the host-side layout test to compare against Rust's figures.
// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation.
//
// Writes 15 slots; see `ROPE_LAYOUT_SLOTS` in `kernels/params.rs` for what each
// index means.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void rope_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    RopeIParams   probe_RopeIParams;
    RopeParams    probe_RopeParams;
    RopeThdParams probe_RopeThdParams;
    out[ 0] = sizeof(RopeIParams);
    out[ 1] = offsetof_rt(RopeIParams, bh);
    out[ 2] = offsetof_rt(RopeIParams, td);
    out[ 3] = offsetof_rt(RopeIParams, stride_b);

    out[ 4] = sizeof(RopeParams);
    out[ 5] = offsetof_rt(RopeParams, bh);
    out[ 6] = offsetof_rt(RopeParams, td);
    out[ 7] = offsetof_rt(RopeParams, d);
    out[ 8] = offsetof_rt(RopeParams, stride_b);

    out[ 9] = sizeof(RopeThdParams);
    out[10] = offsetof_rt(RopeThdParams, b);
    out[11] = offsetof_rt(RopeThdParams, t);
    out[12] = offsetof_rt(RopeThdParams, h);
    out[13] = offsetof_rt(RopeThdParams, d);
    out[14] = offsetof_rt(RopeThdParams, stride_b);
}

init_rope(f32, float)
init_rope(f16, half)

#if defined(__HAVE_BFLOAT__)
init_rope(bf16, bfloat)
#endif
