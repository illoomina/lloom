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
// `kernels/params.rs` and checked against them by `unary_params_layout`.
//
// `size_t` and `int64_t` are both 8 bytes in MSL, so these are 8-aligned --
// unlike `reduce.metal`'s `uint` structs, which are 4-aligned. That difference
// is why the layout check ships the numbers across the boundary rather than
// trusting either side to be right about padding.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length.
struct UnaryParams {
    size_t dim;
};

struct UnaryStridedParams {
    size_t dim;
    size_t num_dims;
};

// `copy2d` is the largest single kernel in a decode token -- 140 of 674
// dispatches (`DESIGN.md` §11.2), the KV `Tensor::cat` and the conv-state
// shuffle. Its four scalars are `int64_t`, not `size_t`, which is a real
// distinction: they are signed, and narrowing them here would be a numeric
// change smuggled in beside a binding change.
struct Copy2dParams {
    int64_t d1;
    int64_t d2;
    int64_t src_s;
    int64_t dst_s;
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
// in this file.
template <typename T, typename U, typename unary, int W>
METAL_FUNC void unary_body(
    size_t dim,
    device const T* input,
    device U* output,
    uint tid
) {
    unary op;
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(op(input[i]));
    }
}

template <typename T, typename U, typename unary, int W = work_per_thread<T>()>
[[kernel]] void unary_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    unary_body<T, U, unary, W>(dim, input, output, tid);
}

template <typename T, typename U, typename unary, int W = work_per_thread<T>()>
[[kernel]] void unary_kernel_packed(
    device const UnaryParams *pp,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    unary_body<T, U, unary, W>(pp->dim, input, output, tid);
}

// `InPtrT` is templated alongside `PtrT` so the classical wrapper keeps its
// `constant const T *input` exactly as it was. That address space is not
// incidental -- changing it would be a behaviour change to the classical path
// smuggled in beside a binding change.
template <typename T, typename U, typename unary, typename PtrT, typename InPtrT>
METAL_FUNC void unary_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    InPtrT input,
    device U *output,
    uint tid
) {
    unary op;
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[tid] = static_cast<U>(op(input[idx]));
}

template <typename T, typename U, typename unary>
[[kernel]] void unary_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    unary_strided_body<T, U, unary, constant size_t *, constant const T *>(
        dim, num_dims, dims, strides, input, output, tid);
}

template <typename T, typename U, typename unary>
[[kernel]] void unary_kernel_strided_packed(
    device const UnaryStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    unary_strided_body<T, U, unary, device const size_t *, constant const T *>(
        pp->dim, pp->num_dims, dims, strides, input, output, tid);
}

template <typename T, int W>
METAL_FUNC void const_set_body(
    size_t dim,
    device const T &input,
    device T *output,
    uint tid
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = input;
    }
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void const_set(
    constant size_t &dim,
    device const T &input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const_set_body<T, W>(dim, input, output, tid);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void const_set_packed(
    device const UnaryParams *pp,
    device const T &input,
    device T *output,
    uint tid [[thread_position_in_grid]]
) {
    const_set_body<T, W>(pp->dim, input, output, tid);
}

template <typename T, typename PtrT>
METAL_FUNC void const_set_strided_body(
    size_t dim,
    size_t num_dims,
    PtrT dims,
    PtrT strides,
    device const T &input,
    device T *output,
    uint tid
) {
    if (tid >= dim) return;
    uint idx = get_strided_index(tid, num_dims, dims, strides);
    output[idx] = input;
}

template <typename T>
[[kernel]] void const_set_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    device const T &input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    const_set_strided_body<T, constant size_t *>(
        dim, num_dims, dims, strides, input, output, tid);
}

template <typename T>
[[kernel]] void const_set_strided_packed(
    device const UnaryStridedParams *pp,
    device const size_t *dims,
    device const size_t *strides,
    device const T &input,
    device T *output,
    uint tid [[ thread_position_in_grid ]]
) {
    const_set_strided_body<T, device const size_t *>(
        pp->dim, pp->num_dims, dims, strides, input, output, tid);
}

template <typename T>
METAL_FUNC void copy2d_body(
    int64_t d1,
    int64_t d2,
    int64_t src_s,
    int64_t dst_s,
    device const T *input,
    device T *output,
    uint2 idx
) {
    if (idx.x >= d1 || idx.y >= d2) return;
    int64_t src_idx = idx.x * src_s + idx.y;
    int64_t dst_idx = idx.x * dst_s + idx.y;
    output[dst_idx] = input[src_idx];
}

template <typename T>
[[kernel]] void copy2d(
    constant int64_t &d1,
    constant int64_t &d2,
    constant int64_t &src_s,
    constant int64_t &dst_s,
    device const T *input,
    device T *output,
    uint2 idx [[thread_position_in_grid]]
) {
    copy2d_body<T>(d1, d2, src_s, dst_s, input, output, idx);
}

template <typename T>
[[kernel]] void copy2d_packed(
    device const Copy2dParams *pp,
    device const T *input,
    device T *output,
    uint2 idx [[thread_position_in_grid]]
) {
    copy2d_body<T>(pp->d1, pp->d2, pp->src_s, pp->dst_s, input, output, idx);
}

// Unary functions
template <typename T> METAL_FUNC T erf(T in){
    // constants
    constexpr const float a1 =  0.254829592;
    constexpr const float a2 = -0.284496736;
    constexpr const float a3 =  1.421413741;
    constexpr const float a4 = -1.453152027;
    constexpr const float a5 =  1.061405429;
    constexpr const float p  =  0.3275911;

    float x = static_cast<float>(in);

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return T(sign*y);
}
template <typename T> METAL_FUNC T id(T in) { return in; }
template <typename T> METAL_FUNC T gelu_erf(T x) {
    return static_cast<T>(x * (1 + erf(x * M_SQRT1_2_F)) / 2);
}
template <typename T> METAL_FUNC T gelu(T x) {
    if (x > 5) {
        return x;
    }
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    T beta =  (static_cast<T>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + T(precise::tanh(beta)));
}
template <typename T> METAL_FUNC T relu(T x) {
    if (x > 5) {
        return x;
    }
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    T beta =  (static_cast<T>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + T(precise::tanh(beta)));
}
template <typename T> METAL_FUNC T recip(T x) {
    return static_cast<T>(1.0 / x);
}
template <typename T> METAL_FUNC T sigmoid(T x) {
    return static_cast<T>(recip(1 + exp(-x)));
}

// Define unary ops
#define define_unary_op(name, op)   \
struct name {                       \
    template <typename T>           \
    METAL_FUNC T operator()(T x) {  \
        return static_cast<T>(op);  \
    }                               \
};

define_unary_op(usqr, x * x);
define_unary_op(urecip, recip(x));
define_unary_op(uneg, -x);
define_unary_op(uid, x);
define_unary_op(ugelu, gelu(x));
define_unary_op(urelu, x < 0 ? 0 : x);
define_unary_op(usilu, x / (1 + exp(-x)));

///////////////////////////////////////////////////////////////////////////////
/// The SiLU variant axis (issue #599, `DESIGN.md` §12.2l)
///////////////////////////////////////////////////////////////////////////////
//
// `usilu` above is the SHIPPING arm and is untouched: it is `SiluImpl::Exact`,
// it keeps its `[[host_name]]`, and an unconfigured process resolves exactly
// the name it always did (§7.1a -- no default is flipped here).
//
// The arms below exist to answer #599's titled question -- *how much error does
// the model absorb before it degenerates* -- which needs variants that differ
// by a KNOWN amount, spanning several orders of magnitude. #604 (§12.2k)
// measured the five spellings already in the tree and found them bit-identical
// at f32 and 0-3 ulp apart at f16: the existing spread is far too narrow to
// locate an absorption threshold, so the curve needs arms built for it.
//
// **These are variants, not replacements** -- §3.6's law. The functor is the
// template parameter `unary_kernel` already takes, so each arm is one
// `define_unary_op` row plus one `init_unary_float` row, composed at
// instantiation and inlined by the compiler. Nothing about the kernel body,
// the binding style or the dispatch changes.
//
// **Why the ERROR is the axis and not the SPEED.** These arms are deliberately
// not proposed as faster: `exp` on this hardware is a hardware instruction and
// a polynomial that replaces it is not obviously cheaper. The axis is
// numerical, and the question is what the model tolerates -- which bounds every
// future approximation (quantized activations, a cheaper `exp`, a fused
// epilogue at lower precision) with one measured curve instead of one
// experiment each.

// **Arm 2 -- `Recip`.** `x * recip(1+exp(-x))`, `unary.metal:308`'s sigmoid
// spelling composed. #604 measured this bit-identical to `usilu` at both
// dtypes; it is carried here as an INTERNAL CONTROL rather than as a variant.
// A sweep whose arms all differ cannot show that the instrument reports "no
// difference" when there is none, and §3.1a rule 4 is that all-different is as
// suspicious as all-identical. This arm must read zero.
define_unary_op(usilu_recip, x * recip(1 + exp(-x)));

// **Arm 3 -- `Literal`.** `x / (1.0f + exp(-x))`, `quantized.metal:1563`'s
// spelling. #604 measured this at 1 ulp over 440 of 4096 points at f16 and
// bit-identical at f32, and identified the `1.0f` suffix as the mechanism: it
// promotes the addition to float where a bare `1` converts to `T`. It is the
// smallest NON-ZERO perturbation available from a real site in this tree, so it
// is the bottom rung of the ladder -- and the one arm whose magnitude was
// measured before this issue rather than by it.
define_unary_op(usilu_literal, x / (1.0f + exp(-x)));

// **Arms 4 and 5 -- `Coarse4` and `Coarse1`.** The two middle rungs, and the
// only ones whose error is a free parameter.
//
// `exp(-x)` is rewritten as `exp2(-x * log2(e))` -- an identity, and `exp2` is
// what the hardware instruction computes anyway -- and then the ARGUMENT is
// rounded to a fixed number of fractional bits before `exp2` sees it. The
// number of bits is the knob: 4 fractional bits gives ~4.6e-3 peak absolute
// error, 1 bit gives ~3.9e-2, and the limit of infinitely many bits is the
// exact arm. Nothing else about the expression changes.
//
// **Why this shape rather than a polynomial or a fast-sigmoid.** The first
// version of this arm WAS the classic fast-sigmoid `0.5*(1 + x/(1+|x|))`, and
// it was wrong for this job in a way worth recording: it converges to -0.476
// as x -> -inf where SiLU converges to 0, so its "error" is a systematic bias
// in the whole negative tail rather than an approximation that shrinks where
// the function does. An arm like that degenerates the model for a reason that
// is not *approximation error*, which makes it useless as a rung -- it would
// put a point on the curve whose x-coordinate does not mean what the other
// points' means. Measured before it was replaced: peak absolute error 4.76e-1,
// and 4096 of 4096 points differing at f32 where the exact arm's own error is
// 1.7e-6.
//
// The quantized-`exp2` form has the property the ladder needs: it decays to
// zero in the tail exactly as SiLU does, so its error is *approximation* error
// at every x, and the knob moves it smoothly.
//
// A note on spelling, because it cost a build: at `T = bfloat` a bare `abs`,
// `clamp` or `floor` is ambiguous between the float overload and the integer
// ones, and the whole `unary` library then fails to compile -- taking every
// unary op with it, not just this arm. Caught by the name-resolution table in
// `silu_variant_error.rs` on its first run, where `silu_f16` -- a name that has
// always existed -- read NO alongside the new ones. **A resolution check
// covering only the NEW names would have reported them absent and sent the
// reader hunting a missing instantiation** (§8.1b, and #517's positive-control
// rule: include a name that must already be there).
#define SILU_COARSE(x, scale)                                                  \
    ((x) / (static_cast<T>(1) +                                                \
        exp2(static_cast<T>(rint(static_cast<float>(-(x)) * 1.4426950408889634f \
             * (scale)) / (scale)))))

define_unary_op(usilu_coarse4, SILU_COARSE(x, 16.0f));
define_unary_op(usilu_coarse1, SILU_COARSE(x, 2.0f));

// **Arm 6 -- `HardSwish`.** `x * clamp(x/6 + 0.5, 0, 1)`, the piecewise-linear
// hard-swish from MobileNetV3. Exp-free, cheap, and visibly wrong: peak
// absolute error against SiLU is ~0.14 and it is exactly zero for x <= -3 where
// SiLU is small but non-zero.
//
// Spelled with `fmin`/`fmax` rather than `clamp` for `fabs`'s reason one arm
// up: at `T = bfloat` a bare `clamp` is ambiguous against the integer
// overloads. `fmin(fmax(v, 0), 1)` is the same function and names the float
// overload unambiguously at all three dtypes.
define_unary_op(usilu_hardswish, x * fmin(fmax(x / static_cast<T>(6) +
    static_cast<T>(0.5), static_cast<T>(0)), static_cast<T>(1)));

// **Arm 7 -- `Relu`.** Not an approximation to SiLU at all: a DIFFERENT
// activation function, peak absolute error ~0.28 against SiLU and about twice
// hard-swish's.
//
// **Added because the ladder ran out before the model did.** The first sweep
// went to `hardswish` and every arm passed the text gate, which bounds the
// absorption capacity from below and does not locate it. This arm is the top
// because it abandons the *shape*: it is exactly zero for all x < 0, where SiLU
// has its characteristic non-monotone dip, so if the model tolerates this the
// finding is not "SiLU error is absorbed" but "this position in the network
// barely cares which nonlinearity it gets".
//
// `x < 0 ? 0 : x` is `urelu`'s own spelling one screen up, copied
// character-for-character rather than paraphrased (§3.1c). Note that
// `unary.metal:294`'s `relu<T>()` is NOT relu -- its body is a copy of `gelu`'s
// tanh approximation and nothing calls it (#599's earlier finding, recorded in
// the pit) -- so taking the functor's spelling rather than that function's is
// deliberate.
define_unary_op(usilu_relu, x < 0 ? 0 : x);

define_unary_op(ugelu_erf, gelu_erf(x));
define_unary_op(usqrt, sqrt(x));
define_unary_op(ucos, cos(x));
define_unary_op(usin, sin(x));
define_unary_op(uexp, exp(x));
define_unary_op(ulog, log(x));
define_unary_op(uabs, abs(static_cast<float>(x)));
define_unary_op(uceil, ceil(x));
define_unary_op(ufloor, floor(x));
define_unary_op(uround, round(x));
define_unary_op(uerf, erf(x));
define_unary_op(usign, sign(x));
define_unary_op(usigmoid, sigmoid(x));
// tanh may create NaN on large values, e.g. 45 rather than outputting 1.
// This has been an issue for the encodec example.
define_unary_op(utanh, precise::tanh(x));

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `unary_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
static_assert(sizeof(UnaryParams) == 8, "UnaryParams layout");
static_assert(alignof(UnaryParams) == 8, "UnaryParams alignment");

static_assert(sizeof(UnaryStridedParams) == 16, "UnaryStridedParams layout");
static_assert(alignof(UnaryStridedParams) == 8, "UnaryStridedParams alignment");

static_assert(sizeof(Copy2dParams) == 32, "Copy2dParams layout");
static_assert(alignof(Copy2dParams) == 8, "Copy2dParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void unary_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    UnaryParams        probe_UnaryParams;
    UnaryStridedParams probe_UnaryStridedParams;
    Copy2dParams       probe_Copy2dParams;
    out[0] = sizeof(UnaryParams);
    out[1] = offsetof_rt(UnaryParams, dim);

    out[2] = sizeof(UnaryStridedParams);
    out[3] = offsetof_rt(UnaryStridedParams, dim);
    out[4] = offsetof_rt(UnaryStridedParams, num_dims);

    out[5] = sizeof(Copy2dParams);
    out[6] = offsetof_rt(Copy2dParams, d1);
    out[7] = offsetof_rt(Copy2dParams, d2);
    out[8] = offsetof_rt(Copy2dParams, src_s);
    out[9] = offsetof_rt(Copy2dParams, dst_s);
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
#define init_unary(op_name, unary_op, tname, t)                                                        \
    init_kernel(#op_name "_" #tname, unary_kernel, t, t, unary_op)                                     \
    init_kernel(#op_name "_" #tname "_packed", unary_kernel_packed, t, t, unary_op)                    \
    init_kernel(#op_name "_" #tname "_strided", unary_kernel_strided, t, t, unary_op)                  \
    init_kernel(#op_name "_" #tname "_strided_packed", unary_kernel_strided_packed, t, t, unary_op)

#if defined(__HAVE_BFLOAT__)
#define init_unary_float(op_name, unary_op)   \
    init_unary(op_name, unary_op, f32, float) \
    init_unary(op_name, unary_op, f16, half)  \
    init_unary(op_name, unary_op, bf16, bfloat)
#else
#define init_unary_float(op_name, unary_op)   \
    init_unary(op_name, unary_op, f32, float) \
    init_unary(op_name, unary_op, f16, half)
#endif

#define init_copy2d(tname, t)                                       \
    init_kernel("copy2d_" #tname, copy2d, t)                        \
    init_kernel("copy2d_" #tname "_packed", copy2d_packed, t)

#define init_const_set(tname, t)                                                        \
    init_kernel("const_set_" #tname, const_set, t)                                      \
    init_kernel("const_set_" #tname "_packed", const_set_packed, t)                     \
    init_kernel("const_set_" #tname "_strided", const_set_strided, t)                   \
    init_kernel("const_set_" #tname "_strided_packed", const_set_strided_packed, t)

// Initialize all unary kernels for floating point types
init_unary_float(gelu_erf, ugelu_erf);
init_unary_float(sqrt, usqrt);
init_unary_float(sqr, usqr);
init_unary_float(neg, uneg);
init_unary_float(recip, urecip);
init_unary_float(copy, uid);
init_unary_float(silu, usilu);
// The #599 variant arms. Each is a DISTINCT `[[host_name]]` set, so `silu_f16`
// -- the name an unconfigured process resolves -- is byte-identical to what it
// was and the arms are selected by name rather than by rebuilding (§7.1a).
// Instantiated unconditionally, as `gemv_silu`'s names are since #516: a
// run-time selector whose kernel exists only under a `-D` is two sources of
// truth for one decision, and #516 removed exactly that shape.
init_unary_float(silu_recip, usilu_recip);
init_unary_float(silu_literal, usilu_literal);
init_unary_float(silu_coarse4, usilu_coarse4);
init_unary_float(silu_coarse1, usilu_coarse1);
init_unary_float(silu_hardswish, usilu_hardswish);
init_unary_float(silu_relu, usilu_relu);
init_unary_float(gelu, ugelu);
init_unary_float(relu, urelu);
init_unary_float(cos, ucos);
init_unary_float(sin, usin);
init_unary_float(exp, uexp);
init_unary_float(log, ulog);
init_unary_float(abs, uabs);
init_unary_float(ceil, uceil);
init_unary_float(floor, ufloor);
init_unary_float(round, uround);
init_unary_float(erf, uerf);
init_unary_float(sign, usign);
init_unary_float(sigmoid, usigmoid);
init_unary_float(tanh, utanh);

// Initialize copy2d kernels
init_copy2d(f32, float);
init_copy2d(f16, half);

// Initialize const_set kernels
init_const_set(f32, float);
init_const_set(f16, half);

#if defined(__HAVE_BFLOAT__)
init_copy2d(bf16, bfloat);
init_const_set(bf16, bfloat);
#endif

// Initialize unary kernels for integer dtypes
init_unary(copy, uid, u8, uint8_t);
init_unary(copy, uid, u32, uint32_t);

init_copy2d(u8, uint8_t);
init_copy2d(u32, uint32_t);

init_const_set(u8, uint8_t);
init_const_set(u32, uint32_t);

#if __METAL_VERSION__ >= 220
init_unary(copy, uid, i64, int64_t);
init_copy2d(i64, int64_t);
init_const_set(i64, int64_t);
#endif

init_copy2d(i32, int32_t);
init_copy2d(i16, int16_t);
