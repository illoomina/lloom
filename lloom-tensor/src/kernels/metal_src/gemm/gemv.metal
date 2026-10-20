// GEMV kernel adapted from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.metal
// Copyright © 2023-2024 Apple Inc.

// NOTE: this file does NOT open with its own `#include`s. As of issue #599 it
// is a `with_utils!` consumer: `source.rs` supplies
// `<metal_simdgroup>`/`<metal_stdlib>` and splices `utils.metal` in after them,
// so this file is a FRAGMENT and does not compile standalone -- the same shape
// `unary.metal`, `binary.metal` and `reduce.metal` have had since #458.
//
// It was made one so that `EpilogueSilu` could CALL `utils.metal`'s shared
// `silu<T>()` instead of carrying a second copy of the expression
// (`DESIGN.md` §15.2 #15 / `CONTRIBUTING.md` §3.6). `#include` is not available
// here for #458's reason: the sources are `include_str!`-embedded and compiled
// at runtime by `newLibraryWithSource`, so there is no directory for a quoted
// include to resolve against. Composition is `concat!` on the Rust side.

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// elem_to_loc for nc=1 batch handling
//
// Templated on the *pointer types* rather than fixing them to `constant`,
// because the packed entry points below receive `batch_shape` and the three
// stride arrays in `device` buffers instead. Deducing the address space from
// the argument keeps this one body: rewriting the arithmetic per address space
// was the alternative, and it is the destructive one. Same remedy issue #38
// used for `strided_indexer` (`DESIGN.md` §11.3d).
//
// `stride_t` stays an explicit parameter -- it is the *value* type and callers
// name it -- while `ShapeP`/`StrideP` are deduced.
template <typename stride_t, typename ShapeP, typename StrideP>
METAL_FUNC stride_t gemv_elem_to_loc(
    uint elem,
    ShapeP shape,
    StrideP strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename U>
struct GemvDefaultAccT {
  using type = float;
};

///////////////////////////////////////////////////////////////////////////////
/// Prologue functors (issue #266, `DESIGN.md` §12.2 #2)
///////////////////////////////////////////////////////////////////////////////

// The mirror of `mlx_gemm.metal`'s `TransformNone`/`TransformAdd`/
// `TransformAxpby` at the *input* end. §7.5 states the mechanism for epilogues
// and §11.3v (#221) records that it is built and inherited from MLX for the
// matmul; two of §12.2's four fusion entries are **prologues**, which that item
// does not name, and this is the seam they need.
//
// A prologue transforms `in_vec` as it is loaded into registers, so the
// transformed vector is never written to memory. That is what removes the
// round-trip §12.2 #2 is about, and §12.1 is explicit that the round-trip is
// the win rather than the dispatch count.
//
// A functor here sees one thread's `TN` elements at offset `bn`; a reduction
// over all of `K` -- RMSNorm's `rsqrt(mean(x^2))` -- is not visible to it.
// Two shapes follow. `PrologueRmsNorm` imports the reduction as a scale a
// separate dispatch computed: a GEMV partitions its OUTPUT across `n_tgp`
// threadgroups with no barrier between them, and at a tile whose simdgroups
// split K a per-threadgroup recompute re-reads the input `n_tgp` times
// (`measurements/issue-266-raw/prediction.py`: 432x the reduction work).
// `PrologueNormInline` recomputes it in every
// simdgroup from the coefficients the K-walk already loads, which at the
// decode tiles (`bn1 sm1 sn32`: one simdgroup spans K) costs no read, and
// applies `1/rms` at the store (`docs/tuning.md` §13.5t).

// The identity. Selected when no fusion is asked for, and byte-for-byte what
// shipped: it holds no state and its `operator()` returns its argument, so it
// inlines away entirely.
template <typename T, typename AccT>
struct PrologueNone {
  MLX_MTL_CONST bool kInlineNorm = false;
  MLX_MTL_CONST bool kGroupNorm = false;
  template <typename S>
  METAL_FUNC void prepare(const device S*, uint, uint, uint3, int) const {}
  METAL_FUNC PrologueNone(const device T*, const device float*) {}
  METAL_FUNC AccT operator()(AccT x, int) const { return x; }
  METAL_FUNC AccT scale(AccT, int) const { return AccT(1); }
};

// RMSNorm computed INSIDE the GEMV, reduction included (`DESIGN.md` §12.2
// row 2, `docs/tuning.md` §13.5t). At the decode tiles (`bn1 sm1 sn32`) each
// simdgroup's 32 lanes walk the whole of K for the dot product, so the same
// loaded coefficients give `sum(x^2)` for one more FMA per element and one
// more shuffle tree, with no extra read: the reduction the separate
// `rmsnorm` dispatch exists for is already in every simdgroup's registers.
// `y = (W (x * w)) * rsqrt(mean(x^2) + eps)`: the weight rides the prologue,
// the 1/rms the store. Not bit-identical to norm-then-GEMV (no f16 rounding
// of the normalized vector, a different summation tree).
template <typename T, typename AccT>
struct PrologueNormInline {
  MLX_MTL_CONST bool kInlineNorm = true;
  MLX_MTL_CONST bool kGroupNorm = false;
  template <typename S>
  METAL_FUNC void prepare(const device S*, uint, uint, uint3, int) const {}
  const device T* w;
  AccT eps;

  METAL_FUNC PrologueNormInline(const device T* w_, float eps_)
      : w(w_), eps(static_cast<AccT>(eps_)) {}

  METAL_FUNC AccT operator()(AccT x, int i) const {
    return x * static_cast<AccT>(w[i]);
  }

  METAL_FUNC AccT scale(AccT ssq, int k) const {
    return metal::precise::rsqrt(ssq / static_cast<AccT>(k) + eps);
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Epilogue functors (issue #484, `DESIGN.md` §12.2 #3)
///////////////////////////////////////////////////////////////////////////////

// The OTHER end of the same kernel, and the asymmetry is the entry's whole
// point rather than an implementation detail.
//
// A PROLOGUE sits inside the K-walk. Every one of the `n_tgp` threadgroups
// walks the whole input vector, so a prologue's work is multiplied by `n_tgp`
// **whatever it computes** -- §12.2b's own words, and the property that priced
// #266 at 432x the reduction work and #282 at 128x the elementwise work. That
// is not a fact about reductions; it is a fact about how a GEMV partitions its
// OUTPUT.
//
// An EPILOGUE sits on the other side of that same partition. `GEMVKernel::run`
// ends with
//
//     if (simdN == 0 && thrN == 0) { out_vec[out_row + tm] = ...; }
//
// so exactly ONE thread writes each output element, and a transform placed on
// `result[tm]` there is evaluated exactly once per output element -- the same
// count as the standalone elementwise kernel it replaces. The multiplier that
// killed both prologues is 1 here, by the same argument that makes it `n_tgp`
// there.
//
// It also costs no traffic of its own: `result[tm]` is already in registers
// and `out_vec[out_row + tm]` is a store that already happened. What it removes
// is the consumer kernel's read and its write of the intermediate.
//
// WHAT AN EPILOGUE CANNOT DO, stated so the seam's limit is on the record. It
// sees one thread's `TM` outputs at `out_row`. It cannot see the whole output
// vector, so a reduction over N -- a softmax over the logits, say -- is no more
// expressible here than a reduction over K is in a prologue. The two ends have
// the same restriction and it binds on different axes: a prologue cannot reduce
// over K, an epilogue cannot reduce over N. Elementwise is what fits, and silu
// is elementwise.

// The identity. Byte-for-byte what shipped, on the same argument as
// `PrologueNone`: no state, returns its argument, inlines away.
template <typename T, typename AccT>
struct EpilogueNone {
  METAL_FUNC AccT operator()(AccT x, int) const { return x; }
};

// SiLU / swish: `x / (1 + exp(-x))`.
//
// The EXPRESSION is spelled to match `unary.metal`'s `usilu` character for
// character. ~~`x * (1/(1+exp(-x)))` and `x/(1+exp(-x))` are not the same
// float operation, and a different spelling would move the digests for a
// reason that is not the fusion.~~
//
// **The struck clause was a prediction and it is FALSE -- measured 2026-09-03
// (#599, `DESIGN.md` §12.2k).** Compiled side by side at held-constant
// precision over 4096 inputs, `x * (1/(1+exp(-x)))` is **bit-identical** to
// `x/(1+exp(-x))` at f32 AND at f16, as is `x * recip(1+exp(-x))`
// (`unary.metal:308`'s sigmoid spelling). The algebraic rearrangement this
// comment warned about costs nothing.
//
// What DOES move bits is the LITERAL: `quantized.metal:1563`'s
// `x / (1.0f + exp(-x))` differs by 1 ulp at f16 over 440 of 4096 points,
// because the `f` suffix promotes the addition to float where a bare `1`
// converts to `T`. At `T = float` the two are the same expression, which is
// why f32 shows nothing. Mutation-verified.
//
// **Matching `usilu` character for character remains the right choice** -- it
// costs nothing and removes a variable -- so the decision below is unchanged
// and only its justification is corrected. The PRECISION note that follows is
// unaffected and is the real reason this arm moves the digests.
//
// **The PRECISION deliberately does not match, and this is the one place the
// fused arm is not a rewrite of the unfused one.** Read from source:
// `define_unary_op(name, op)` templates `operator()` on `T` and returns
// `static_cast<T>(op)`, so `silu_f16` evaluates `x/(1+exp(-x))` **at half**
// and its input is the f16 value `w1` already rounded and stored. Here the
// input is `result[tm]`, which is the f32 accumulator BEFORE that rounding,
// and `AccT` is `float` at every dtype this instantiates
// (`GemvDefaultAccT`) -- so the fused arm computes silu at f32 on an
// unrounded input and rounds once, where the unfused pair rounds twice and
// computes the transcendental at f16.
//
// That is strictly the more accurate order and it is still a DIFFERENT
// computation, so this arm **moves the digests by construction** and owes
// §2.3.5a's discriminator rather than an unchanged-digest gate. Recording it
// here because the alternative -- casting down to T, computing at half, and
// casting back, purely to reproduce the unfused rounding -- is available and
// was not taken: it would spend accuracy to make a gate cheaper, which is
// §1.2's "never weaken a test to make a digest match" pointed the other way.
//
// Per §2.3.3b the bare `exp` at FLOAT is the flag-reachable spelling, so this
// arm tracks `MathMode` where `silu_f16`'s half `exp` is pinned. That is a
// second, independent reason the two cannot be bit-compared.
// **This is a CALL SITE of `utils.metal`'s shared `silu<T>()`, not a copy of
// it -- consolidated 2026-09-03 (#599, `DESIGN.md` §15.2 #15).** The expression
// above used to be written out here and in `unary.metal`, which is the
// duplication §3.6 names; there is now one definition and this is one of its
// two live callers.
//
// **The precision note below is UNCHANGED by that, and it is why the primitive
// is templated.** `silu` is instantiated here at `AccT` -- `float` at every
// dtype -- and at `T` in `unary.metal`, so each site keeps the type it always
// passed and the f32-accumulator-versus-f16-store difference this comment
// describes survives verbatim. A shared helper fixed at `float` would have
// silently promoted `usilu` and moved the shipping digests; templating it is
// what makes the consolidation bit-preserving rather than a numerics change
// wearing a refactor's clothes.
template <typename T, typename AccT>
struct EpilogueSilu {
  METAL_FUNC AccT operator()(AccT x, int) const {
    return silu(x);
  }
};

// The mixed-precision arms' epilogues (`gemv_mixed_*`, lloom's Qwen4-exp
// dense projections): `silu(s · x)` and `sigmoid(s · x)` on the accumulator,
// `s` the dispatch's `alpha` — the HyperConnection's `silu(down·x / n)` and
// `sigmoid(up·lo)` as the store, where the composed path wrote the gemv's
// output and read it back through `affine`/`silu`/`sigmoid` launches. At the
// accumulator's precision on the unrounded sum, so a different (more
// accurate) computation than the chain it replaces (§2.3.5a).
template <typename T, typename AccT>
struct EpilogueSiluScale {
  AccT s;
  METAL_FUNC EpilogueSiluScale(float s_) : s(static_cast<AccT>(s_)) {}
  METAL_FUNC AccT operator()(AccT x, int) const {
    return silu(s * x);
  }
};

template <typename T, typename AccT>
struct EpilogueSigmoidScale {
  AccT s;
  METAL_FUNC EpilogueSigmoidScale(float s_) : s(static_cast<AccT>(s_)) {}
  METAL_FUNC AccT operator()(AccT x, int) const {
    const AccT y = s * x;
    return AccT(1) / (AccT(1) + metal::exp(-y));
  }
};

// The residual add: `x + residual[out_row]` (issue #509, `DESIGN.md` §12.2e's
// own recommended candidate, which that catalogue has no row for).
//
// **This is the first epilogue with a BINDING, and the arity argument is why
// it is admissible where §12.2 #7 is not.** G1 requires the functor see ONE
// value; a residual add has two operands, so the row covering this op's
// CONSUMER end fails it. What makes the PRODUCER end different is the index:
// the second operand is read at `out_row + tm` -- **the storing thread's own
// output position** -- so each thread reads exactly the one element it is
// about to write, with no cross-thread data and no reduction.
//
// `kDoAxpby` is already exactly this shape and already instantiated:
// `bias[(out_row + tm) * bias_stride]` at the store below. That is the
// structural difference from §12.2b's `gate * up`, whose second operand is a
// full vector of the INPUT extent and is therefore multiplied by `n_tgp`:
// **a residual is indexable from the output position and a gate is not.**
//
// The residual is read at the ACCUMULATOR's precision and added before the
// narrowing cast, matching `EpilogueSilu`'s placement. `badd_f16` adds at `T`
// on the already-rounded store, so this arm rounds once where the pair rounds
// twice -- strictly the more accurate order, a genuinely different
// computation, and therefore one that moves the digests by construction and
// owes §2.3.5a's discriminator rather than an unchanged-digest gate. Recorded
// rather than worked around, for the reason `EpilogueSilu` gives.
template <typename T, typename AccT>
struct EpilogueResidual {
  const device T* res;

  METAL_FUNC EpilogueResidual(const device T* res_) : res(res_) {}

  METAL_FUNC AccT operator()(AccT x, int i) const {
    return x + static_cast<AccT>(res[i]);
  }
};

// The gate product: `x * gate[out_row]` (§12.2e row 3b at the PRODUCER end).
//
// The same binding and the same arity argument as `EpilogueResidual`: the
// second operand is read at the storing thread's own output position, one
// element per thread, no cross-thread data. §12.2b's `gate * up` failed G1
// as a PROLOGUE on `w2` — there the gate is a full vector of `w2`'s INPUT
// extent, walked once per threadgroup — and row 3b was classified only at
// that seam. At `w3`'s epilogue the gate IS row-indexed: `silu(w1·x)[i]` for
// output row `i`. The product lands at the accumulator's precision before the
// narrowing cast, where the `bmul_f16` it replaces multiplies two rounded
// stores at `T` — one rounding fewer, a different computation, so the arm
// moves the digests by construction and owes §2.3.5a's discriminator.
template <typename T, typename AccT>
struct EpilogueMul {
  const device T* gate;

  METAL_FUNC EpilogueMul(const device T* gate_) : gate(gate_) {}

  METAL_FUNC AccT operator()(AccT x, int i) const {
    return x * static_cast<AccT>(gate[i]);
  }
};

// RMSNorm's elementwise tail: `x * inv_rms * weight[i]`.
//
// `inv_rms` is a single f32 the reduction dispatch wrote, read once at
// construction rather than per element -- §11.3e's hoist rule, which is why
// this costs one dependent load per dispatch and not one per element.
//
// The weight is indexed by the element's position in the INPUT vector, which
// is `bn + tn`, and the caller passes it. Getting that index wrong is the
// failure mode a fixture whose `norm_w` is all-ones cannot see, so the parity
// test uses a non-constant weight.
template <typename T, typename AccT>
struct PrologueRmsNorm {
  MLX_MTL_CONST bool kInlineNorm = false;
  MLX_MTL_CONST bool kGroupNorm = false;
  template <typename S>
  METAL_FUNC void prepare(const device S*, uint, uint, uint3, int) const {}
  const device T* w;
  AccT inv_rms;

  METAL_FUNC PrologueRmsNorm(const device T* w_, const device float* scale)
      : w(w_), inv_rms(static_cast<AccT>(scale[0])) {}

  METAL_FUNC AccT operator()(AccT x, int i) const {
    return x * inv_rms * static_cast<AccT>(w[i]);
  }
  METAL_FUNC AccT scale(AccT, int) const { return AccT(1); }
};

// A GROUPED RMSNorm computed inside the GEMV (`docs/tuning.md` §13.5u): the
// input `[groups · group]` is normalized per group of `group` elements, then
// scaled by the `[groups · group]` weight — Qwen4-exp's HyperConnection norm
// (`hc_norm_f32`: per stream, then γ), whose consumers are mixed GEMVs whose
// simdgroups SPLIT K (`bn8`), so `PrologueNormInline`'s per-simdgroup sum
// cannot be had. Two phases instead. `prepare`, before the K-walk, reduces
// every group's sum of squares across the whole threadgroup — thread `t`
// strides by the threadgroup's width, `simd_sum`, the simdgroups' partials
// summed in index order, `metal::rsqrt(total / group + eps)`: `hc_norm_f32`'s
// own tree when the threadgroup is its 256 threads, so the scale is the same
// number — into threadgroup memory; the K-walk then transforms each loaded
// coefficient to `(x · s_g) · w`, `hc_norm_f32`'s product in its order, and
// accumulates as `PrologueNone` would. The input is re-read from cache once a
// threadgroup (it is tens of kB against the weight's MB). With `scales_out`
// the first threadgroup also writes the `groups` scales (per batch row), for a
// later consumer of the same normalized input that is not a GEMV.
MLX_MTL_CONST int kGroupNormMaxGroups = 8;
MLX_MTL_CONST int kGroupNormMaxSimd = 8;
MLX_MTL_CONST int kGroupNormScratch =
    kGroupNormMaxGroups * kGroupNormMaxSimd + kGroupNormMaxGroups;

template <typename VT, typename AccT>
struct PrologueGroupNorm {
  MLX_MTL_CONST bool kInlineNorm = false;
  MLX_MTL_CONST bool kGroupNorm = true;
  const device VT* w;
  float eps;
  int group;
  int groups;
  device float* scales_out;
  threadgroup AccT* scratch;

  METAL_FUNC PrologueGroupNorm(
      const device VT* w_, float eps_, int group_, int groups_,
      device float* scales_out_, threadgroup AccT* scratch_)
      : w(w_), eps(eps_), group(group_), groups(groups_),
        scales_out(scales_out_), scratch(scratch_) {}

  template <typename S>
  METAL_FUNC void prepare(
      const device S* x, uint simd_gid, uint simd_lid, uint3 tid,
      int n_threads) const {
    const int t = int(simd_gid) * 32 + int(simd_lid);
    const int n_simd = n_threads / 32;
    threadgroup AccT* scale = scratch + kGroupNormMaxGroups * kGroupNormMaxSimd;
    for (int g = 0; g < groups; g++) {
      const device S* xs = x + g * group;
      AccT acc = 0;
      for (int i = t; i < group; i += n_threads) {
        const AccT v = static_cast<AccT>(xs[i]);
        acc += v * v;
      }
      acc = simd_sum(acc);
      if (simd_lid == 0) {
        scratch[g * kGroupNormMaxSimd + simd_gid] = acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < groups) {
      AccT total = 0;
      for (int k = 0; k < n_simd; k++) {
        total += scratch[t * kGroupNormMaxSimd + k];
      }
      const AccT s = metal::rsqrt(total / static_cast<AccT>(group) + static_cast<AccT>(eps));
      scale[t] = s;
      if (scales_out != nullptr && tid.x == 0) {
        scales_out[tid.z * groups + t] = static_cast<float>(s);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  METAL_FUNC AccT operator()(AccT x, int i) const {
    return x * scratch[kGroupNormMaxGroups * kGroupNormMaxSimd + i / group] *
        static_cast<AccT>(w[i]);
  }
  METAL_FUNC AccT scale(AccT, int) const { return AccT(1); }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix-vector: mat [M_out, K] × vec [K] → out [M_out]
/// (mat rows = output dimension, mat cols = K)
///////////////////////////////////////////////////////////////////////////////

// `NR` is the row axis (`docs/tuning.md` §13.5y): `NR` activation rows,
// `vec_row_stride` elements apart, against one sweep of the threadgroup's
// weight block. Each row has its own accumulators and sees, lane for lane,
// the coefficients and the order of operations the one-row kernel gives it.
// That a row's result is then the one-row kernel's bits is a property of how
// this loop nest compiles under fast math, not of the source order alone (a
// nest holding the weight block in registers instead does not keep them):
// `gemv_rows_keep_the_one_row_bits_at_every_capacity` asserts it. `NR = 1`
// is the one-row kernel. A group holding fewer than `NR` rows (`n_rows`)
// re-reads its last row for the missing ones and stores only its own.
template <
    typename T,
    const int BM,       // Threadgroup rows (in simdgroups)
    const int BN,       // Threadgroup cols (in simdgroups)
    const int SM,       // Simdgroup rows (in threads)
    const int SN,       // Simdgroup cols (in threads)
    const int TM,       // Thread rows (in elements)
    const int TN,       // Thread cols (in elements)
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type,
    typename Prologue = PrologueNone<T, AccT>,
    typename Epilogue = EpilogueNone<T, AccT>,
    // The vector type: the input vector, the bias and the output. `T` for
    // every classical arm; `float` against `half` weights on the mixed arms
    // (`gemv_mixed_*`), which read an f32 activation and store an f32 result
    // with no cast launch on either side. The matrix stays `T`.
    typename VT = T,
    const int NR = 1>
struct GEMVKernel {
  using acc_type = AccT;
  using prologue_type = Prologue;
  using epilogue_type = Epilogue;
  using vec_type = VT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");
  static_assert(SN == 4 || SN == 8 || SN == 16 || SN == 32,
                "gemv block must have width 4, 8, 16, or 32");

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  // The inline norm's reduction is complete in one simdgroup only when the
  // simdgroup spans K by itself.
  static_assert(!Prologue::kInlineNorm || (BN == 1 && SN == 32),
                "the inline norm needs a tile whose simdgroup walks all of K");
  // The row axis keeps each row's sum in its simdgroup (no threadgroup
  // reduction to widen) and stores no bias.
  static_assert(NR >= 1, "at least one row");
  static_assert(NR == 1 || (BN == 1 && !kDoAxpby && !Prologue::kGroupNorm),
                "the row axis is instantiated at the tiles whose simdgroup walks all of K, "
                "without axpby or the grouped norm");

  template <typename U = T>
  static METAL_FUNC void load_unsafe(
      const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  // The vector loads, typed on the vector's own element type (`VT`), cast
  // to the accumulator as they land — the matrix loads above stay on `T`.
  template <typename S, typename U>
  static METAL_FUNC void load_vec_unsafe(
      const device S* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename S, typename U>
  static METAL_FUNC void load_vec_safe(
      const device S* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat,
      const device VT* in_vec,
      const device VT* bias,
      device VT* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int matrix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      thread const Prologue& prologue,
      thread const Epilogue& epilogue,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid,
      const int64_t vec_row_stride = 0,
      const int n_rows = 1) {
    (void)lid;

    // The grouped norm's first phase: every thread of the threadgroup takes
    // part (it has threadgroup barriers), so it runs before any thread can
    // leave on the row bound below. Compiled out for every other prologue.
    if (Prologue::kGroupNorm) {
      prologue.prepare(in_vec, simd_gid, simd_lid, tid, BM * BN * 32);
    }

    thread AccT result[NR][TM] = {};
    thread T inter[TN];
    thread AccT v_coeff[NR][TN];
    AccT ssq[NR] = {};

    const device VT* in_row[NR];
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < NR; r++) {
      in_row[r] = in_vec + (NR == 1 ? 0 : (r < n_rows ? r : n_rows - 1) * vec_row_stride);
    }

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (int)(simd_gid / BN) : (int)(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (int)(simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position (output row)
    int out_row = tid.x * blockM + bm;

    if (out_row >= out_vec_size) return;

    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    mat += out_row * matrix_ld;

    const int loop_stride = blockN;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    for (int i = 0; i < n_iter; ++i) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < NR; r++) {
        load_vec_unsafe<VT, AccT>(in_row[r], v_coeff[r], bn);

        if (Prologue::kInlineNorm) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            ssq[r] += v_coeff[r][tn] * v_coeff[r][tn];
          }
        }

        // The prologue transforms the input vector in registers, so a fused
        // producer's output is never written to memory (issue #266). Under
        // `PrologueNone` this is the identity and inlines away, which is what
        // makes the unfused arm byte-for-byte what shipped.
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[r][tn] = prologue(v_coeff[r][tn], bn + tn);
        }
      }

      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_unsafe(mat, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < NR; r++) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[r][tm] += inter[tn] * v_coeff[r][tn];
          }
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }

    if (leftover > 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < NR; r++) {
        load_vec_safe<VT, AccT>(in_row[r], v_coeff[r], bn, in_size);
        if (Prologue::kInlineNorm) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            ssq[r] += v_coeff[r][tn] * v_coeff[r][tn];
          }
        }
        // Guarded by `bn + tn < in_size`: `load_safe` zero-fills past the end,
        // and the prologue must not index its weight out of bounds there. A
        // zero times anything is zero, so skipping the transform on the padding
        // lanes is also arithmetically the same.
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          if (bn + tn < in_size) {
            v_coeff[r][tn] = prologue(v_coeff[r][tn], bn + tn);
          }
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < NR; r++) {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[r][tm] += inter[tn] * v_coeff[r][tn];
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < NR; r++) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
          result[r][tm] += simd_shuffle_down(result[r][tm], sn);
        }
      }
    }
    if (Prologue::kInlineNorm) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < NR; r++) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
          ssq[r] += simd_shuffle_down(ssq[r], sn);
        }
      }
    }

    // `NR == 1` wherever a threadgroup reduction exists (asserted above).
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[0][tm];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[0][tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // THE EPILOGUE SEAM (issue #484). This branch is what makes an epilogue
    // cheap where a prologue is not: it is taken by ONE thread per output
    // element, so `epilogue()` is evaluated exactly `out_vec_size` times per
    // dispatch -- the standalone elementwise kernel's own count -- against a
    // prologue's `n_tgp * in_vec_size` (§12.2a, §12.2b).
    //
    // It is applied to `result[tm]`, the AccT accumulator, BEFORE the
    // narrowing cast to T. Applying it after would round twice and would also
    // put the transform on the wrong side of `kDoAxpby`'s bias add.
    //
    // Under `kDoAxpby` the epilogue transforms the matmul result and the bias
    // is added to the transformed value. No kernel LFM2 dispatches uses axpby
    // (see the `bias` note on `gemv_packed`), and no fused arm is instantiated
    // at `axpby1`, so this composition is unreachable today; it is written the
    // one way that is arithmetically defensible rather than left to whichever
    // order fell out.
    //
    // A row's output, and the index its epilogue reads at, sit `out_vec_size`
    // past the previous row's.
    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < NR; r++) {
        if (NR == 1 || r < n_rows) {
          if (Prologue::kInlineNorm) {
            const AccT inv_rms = prologue.scale(ssq[r], in_vec_size);
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[r][tm] *= inv_rms;
            }
          }
          const int row_out = r * out_vec_size + out_row;
          MLX_MTL_PRAGMA_UNROLL
          for (int tm = 0; tm < TM; tm++) {
            const AccT y = epilogue(result[r][tm], row_out + tm);
            if (kDoAxpby) {
              out_vec[row_out + tm] =
                  static_cast<VT>(alpha) * static_cast<VT>(y) +
                  static_cast<VT>(beta) * bias[(out_row + tm) * bias_stride];
            } else {
              out_vec[row_out + tm] = static_cast<VT>(y);
            }
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector-matrix: vec [K] × mat [K, N_out] → out [N_out]
/// (mat rows = K, mat cols = output dimension)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type>
struct GEMVTKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM * (blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int marix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid) {
    (void)lid;

    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgM = BN != 1 ? (int)(simd_gid / BN) : (int)simd_gid;
    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    const int loop_stride = blockM;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    if (out_col < out_vec_size) {
      out_col = out_col + TN <= out_vec_size ? out_col : out_vec_size - TN;

      for (int i = 0; i < n_iter; ++i) {
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          auto vc = static_cast<AccT>(v_coeff[tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc * inter[tn];
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += v_coeff[tm] * inter[tn];
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] =
              static_cast<T>(alpha) * static_cast<T>(result[j]) +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = static_cast<T>(result[j]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Packed parameter block
///////////////////////////////////////////////////////////////////////////////

// The scalars `gemv` and `gemv_t` take as `constant int&` today, as one struct
// an `MTLIndirectComputeCommand` can bind.
//
// `MTLIndirectComputeCommand` has no `setBytes` in any form (`DESIGN.md`
// §3.7c); `setKernelBuffer` is its only binding primitive, so a kernel whose
// scalars arrive inline cannot be encoded into an ICB at all. Both entry points
// below are instantiated from the same body, so in §7.1's terms the binding
// style is a compile-tier variant axis alongside dtype — not a migration.
//
// Field order mirrors the classical argument order, so `set_params!` fills the
// block in the order the arguments are already written. `batch_shape` and the
// three stride arrays are deliberately *not* fields: their length is a property
// of the call, so they stay separate bindings — which an ICB can express, since
// `setKernelBuffer` binds a buffer of any length (`DESIGN.md` §11.3d).
//
// Every field is a 4-byte scalar, so this is 28 bytes at 4-byte alignment with
// no padding anywhere. Checked rather than assumed: the `static_assert` below
// pins size and alignment, and `gemv_params_layout` reports every offset for
// the Rust side to compare against `offset_of!`.
struct GemvParams {
  int in_vec_size;
  int out_vec_size;
  int matrix_ld;
  float alpha;
  float beta;
  int batch_ndim;
  int bias_stride;
};

static_assert(sizeof(GemvParams) == 28, "GemvParams layout");
static_assert(alignof(GemvParams) == 4, "GemvParams alignment");

///////////////////////////////////////////////////////////////////////////////
/// gemv kernel: mat [M, K] × vec [K] → out [M]
///////////////////////////////////////////////////////////////////////////////

// The batch offsets and the call into `GEMVKernel::run`, shared by both binding
// styles.
//
// Note what this takes and what it does *not*: the scalars arrive **by value**,
// already unpacked by whichever wrapper called it. That is the whole answer to
// issue #41's inner-loop question. `GEMVKernel::run` already takes
// `const int in_vec_size` / `const int matrix_ld` by value, so the loop bound
// and the row stride are in registers before the first iteration under either
// style; neither `constant int&` nor `device GemvParams*` is ever dereferenced
// inside the loop. Promoting them to a buffer therefore moves *where the load
// happens once*, not how often. `regpressure.m` checks that against the
// compiler rather than leaving it as a reading of the source.
//
// `ShapeP`/`StrideP` are deduced, so `constant`-space arrays (classical) and
// `device`-space ones (packed) both reach the same arithmetic.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby,
    typename Prologue,
    typename Epilogue,
    const int NR = 1,
    typename VT,
    typename ShapeP, typename StrideP>
METAL_FUNC void gemv_body(
    const device T* mat,
    const device VT* in_vec,
    const device VT* bias,
    device VT* out_vec,
    const int in_vec_size,
    const int out_vec_size,
    const int matrix_ld,
    const float alpha,
    const float beta,
    const int batch_ndim,
    ShapeP batch_shape,
    StrideP vector_batch_stride,
    StrideP matrix_batch_stride,
    StrideP bias_batch_stride,
    const int bias_stride,
    threadgroup typename GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    thread const Prologue& prologue,
    thread const Epilogue& epilogue,
    uint3 tid,
    uint3 lid,
    uint simd_gid,
    uint simd_lid) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby,
                              typename GemvDefaultAccT<T>::type, Prologue,
                              Epilogue, VT, NR>;

  // Under the row axis `tid.z` is a group of `NR` batch items (the rows),
  // every one against the batch's one matrix: the host binds a zero matrix
  // batch stride.
  static_assert(NR == 1 || !kDoNCBatch, "the row axis takes the contiguous batch");
  const uint row0 = tid.z * NR;
  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += row0 * vector_batch_stride[0];
    mat    += row0 * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += row0 * out_vec_size;
  const int n_rows = NR == 1 ? 1 : min(NR, int(batch_shape[0]) - int(row0));

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                tgp_memory, prologue, epilogue,
                tid, lid, simd_gid, simd_lid,
                vector_batch_stride[0], n_rows);
}

// The threadgroup array cannot move into `gemv_body`: MSL permits a
// threadgroup-address-space variable only inside a `[[kernel]]`-qualified
// function, which is issue #38's finding (`DESIGN.md` §11.3d) reproducing here
// unchanged. So the allocation stays in each wrapper and is passed down, and
// what the two wrappers differ in is the lines that unpack the scalars.
#define gemv_tgp_memory(KT) \
  threadgroup typename KT::acc_type tgp_memory[KT::tgp_mem_size == 0 ? 1 : KT::tgp_mem_size]

// `NR` is the row axis (`GEMVKernel`): `NR` batch items against the one
// matrix per threadgroup, the classical names its `NR = 1` instance and the
// `_r<NR>` names its others (`instantiate_gemv_rows_blocks`).
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby,
    const int NR = 1>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueNone<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(nullptr, nullptr);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t, NR>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue_t(),
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv with a fused SiLU epilogue (issue #484, §12.2 #3)
///////////////////////////////////////////////////////////////////////////////

// The fused arm, and **its signature is byte-identical to `gemv`'s above** --
// no extra bindings, because a pure elementwise epilogue reads nothing the
// kernel was not already reading. That is the sharpest single contrast with
// `gemv_rmsnorm`, which needs two extra buffers to import a reduction it
// cannot compute.
//
// It is nonetheless a DISTINCT function template rather than a second
// `[[host_name]]` on `gemv`, for the reason `instantiate_gemv_packed_helper`
// records: MSL rejects two names on one explicit instantiation outright.
//
// This arm exists because `lfm2.rs::Mlp::forward` is
// `silu(gate_proj.forward(x))` -- 30 of the step's 539 dispatches are a
// `silu_f16` whose sole producer is this kernel, with nothing between them
// (`measurements/issue-249-raw/kernel-census.txt`; 30 silu against 30 layers).
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_silu(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueSilu<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(nullptr, nullptr);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue_t(),
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv with a fused residual-add epilogue (issue #509)
///////////////////////////////////////////////////////////////////////////////

// **Unconditionally instantiated, exactly as `gemv_silu` and `gemv_rmsnorm`
// are** -- issue #516. It sat behind `-DLLOOM_509_RESIDUAL_EPILOGUE` until
// then, and that `-D` is REMOVED rather than made more legible, because both
// reasons it existed have expired:
//
//  1. **It dissolved §17 item 7's gate** (§12.2h) -- but that item is
//     discharged in substance by #484 (§17 Phase 1 item 7): `GEMVKernel`
//     carries an `Epilogue` template parameter defaulting to `EpilogueNone`,
//     and `gemv_silu` ships through it with no `-D` at all. The gate the flag
//     routed around is gone, so the route is no longer load-bearing.
//  2. **It kept the metallib narrow** -- measured at #516 rather than
//     asserted: the arm adds **6 names of 307, and 55 376 bytes (+2.04 %)** to
//     `gemv`'s metallib, against §11.3g's **+4.01 MiB and 144 new variants**
//     for the packed-params doubling this same file already carries. §11.3g's
//     verdict is that the doubling is affordable *structurally* -- a library
//     is compiled once per `Source` and pipelines are lazy, so an
//     un-dispatched variant costs nothing at dispatch. This arm is ~74x
//     smaller than the widening already accepted here, on the same file.
//
// What the `-D` cost was a **second source of truth for one decision**: a
// run-time `Config::gemv_residual` selecting a kernel whose *existence* was
// decided at build time, so `GemvResidual=on` without the flag failed at
// `newFunctionWithName:` -- after model load and prefill, with a Metal error
// naming a mangled kernel rather than the flag. §7.1's tiers are the frame:
// the arm is a **dispatch-tier** choice and now resolves at exactly one tier.
//
// One binding, at 15 -- after every argument the classical signature declares,
// so the classical slot numbering is untouched, exactly as `gemv_rmsnorm`'s
// two are.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_residual(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    const device T* residual [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueResidual<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(nullptr, nullptr);
  // `tid.z` is the batch index and the residual is one vector per batch item,
  // matching `out_vec += tid.z * out_vec_size` in `gemv_body`.
  const epilogue_t epilogue(residual + tid.z * out_vec_size);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue,
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv with a fused gate-product epilogue (issue #732, §12.2e row 3b)
///////////////////////////////////////////////////////////////////////////////

// `gemv_residual`'s signature with `EpilogueMul`: the operand at 15, after
// every classical slot, so the host binds it at one index for both epilogues
// that read one.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_mul(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    const device T* gate [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueMul<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(nullptr, nullptr);
  // `tid.z` is the batch index and the gate is one vector per batch item,
  // matching `out_vec += tid.z * out_vec_size` in `gemv_body`.
  const epilogue_t epilogue(gate + tid.z * out_vec_size);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue,
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv with a fused RMSNorm prologue (issue #266, §12.2 #2)
///////////////////////////////////////////////////////////////////////////////

// The fused arm. Two extra bindings on the END -- the norm weight and a
// one-element f32 buffer holding `1/rms` -- so the *classical* signature above
// is untouched and the two arms stay independently dispatchable, which is
// §12.3's requirement that fused and unfused variants coexist and be
// swappable.
//
// The scale buffer is produced by a separate reduction dispatch
// (`reduce.metal`'s `rmsnorm_scale`), so this removes the normalized vector's
// round-trip and not the norm's dispatch. `gemv_norm` (below) removes both by
// recomputing the reduction in the K-walk.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_rmsnorm(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    const device T* norm_weight [[buffer(15)]],
    const device float* norm_scale [[buffer(16)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueRmsNorm<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueNone<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  // The scale is read ONCE here, at kernel entry, rather than per element --
  // §11.3e's hoist rule, which is what keeps this one dependent load per
  // dispatch instead of one per element. `tid.z` is the batch index, matching
  // the one scale per row the reduction writes.
  const prologue_t prologue(norm_weight, norm_scale + tid.z);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue_t(),
      tid, lid, simd_gid, simd_lid);
}

// `gemv_rmsnorm` without the imported scale: the RMSNorm's reduction is
// computed in the GEMV (`PrologueNormInline`), so the norm is not its own
// dispatch. The norm weight at 15 and `eps` at 16; `EPI` the output end.
#define gemv_norm_wrapper(func, EPI)                                              \
template <                                                                        \
    typename T,                                                                   \
    const int BM, const int BN,                                                   \
    const int SM, const int SN,                                                   \
    const int TM, const int TN,                                                   \
    const bool kDoNCBatch,                                                        \
    const bool kDoAxpby>                                                          \
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]                       \
void func(                                                                        \
    const device T* mat [[buffer(0)]],                                            \
    const device T* in_vec [[buffer(1)]],                                         \
    const device T* bias [[buffer(2)]],                                           \
    device T* out_vec [[buffer(3)]],                                              \
    const constant int& in_vec_size [[buffer(4)]],                                \
    const constant int& out_vec_size [[buffer(5)]],                               \
    const constant int& matrix_ld [[buffer(6)]],                                  \
    const constant float& alpha [[buffer(7)]],                                    \
    const constant float& beta [[buffer(8)]],                                     \
    const constant int& batch_ndim [[buffer(9)]],                                 \
    const constant int* batch_shape [[buffer(10)]],                               \
    const constant int64_t* vector_batch_stride [[buffer(11)]],                   \
    const constant int64_t* matrix_batch_stride [[buffer(12)]],                   \
    const constant int64_t* bias_batch_stride [[buffer(13)]],                     \
    const constant int& bias_stride [[buffer(14)]],                               \
    const device T* norm_weight [[buffer(15)]],                                   \
    const constant float& norm_eps [[buffer(16)]],                                \
    uint3 tid [[threadgroup_position_in_grid]],                                   \
    uint3 lid [[thread_position_in_threadgroup]],                                 \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                             \
    uint simd_lid [[thread_index_in_simdgroup]],                                  \
    EOS_GATE_ARG) {                                                               \
    EOS_GATE_CHECK                                                                \
  using acc_t = typename GemvDefaultAccT<T>::type;                                \
  using prologue_t = PrologueNormInline<T, acc_t>;                                \
  using epilogue_t = EPI<T, acc_t>;                                               \
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;               \
  gemv_tgp_memory(kernel_t);                                                      \
  const prologue_t prologue(norm_weight, norm_eps);                               \
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,          \
                 epilogue_t>(                                                     \
      mat, in_vec, bias, out_vec,                                                 \
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,                          \
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,          \
      bias_batch_stride, bias_stride,                                             \
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,                         \
      prologue, epilogue_t(),                                                     \
      tid, lid, simd_gid, simd_lid);                                              \
}

gemv_norm_wrapper(gemv_norm, EpilogueNone)
gemv_norm_wrapper(gemv_norm_silu, EpilogueSilu)

// The ICB-expressible sibling: one `device const GemvParams*` in place of the
// seven `constant &` scalars, everything else unchanged.
//
// The buffer indices are renumbered rather than left with holes — params at 0,
// then the bound tensors, then the arrays — because diverting scalars out of
// the argument list shifts every binding that remains. `ParamCapture` on the
// Rust side performs exactly that renumbering, so the two agree by construction
// rather than by matching two hand-written lists (`DESIGN.md` §11.3d, third
// finding).
//
// **`bias` sits last, at 8, and that is not cosmetic.** `call_mlx_gemv` passes
// `()` for bias — no kernel LFM2 dispatches uses `axpby`, so nothing is bound
// there — and `EncoderParam for ()` binds nothing and consumes no slot. Under
// capture the renumbering therefore assigns consecutive slots to the buffers
// that *are* bound: mat 1, in_vec 2, out_vec 3, then the four arrays at 4..7.
// Declaring `bias` at 3 in source order would leave every later binding one
// slot low and the kernel would read `out_vec`'s contents as its input vector.
//
// This is a *different* failure from the hole #38 describes: there a diverted
// scalar shifted the bindings and the Rust side compensated. Here an argument
// that was never bound at all shifts them, and the compensation has to happen
// in the *kernel signature*, because the Rust side cannot renumber a slot no
// one asked for. Under `HazardTrackingModeUntracked` the symptom is not a
// crash — it hung the GPU, which is how it was found.
//
// `bias` stays a parameter rather than being deleted because `kDoAxpby` is a
// template axis and half the instantiations do read it; it is unbound only for
// the `axpby0` half, exactly as on the classical path.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueNone<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  const prologue_t prologue(nullptr, nullptr);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue_t(),
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// The packed siblings of the two fused epilogues (issue #732 P3)
///////////////////////////////////////////////////////////////////////////////

// `gemv_packed`'s signature with the epilogue functor — what `gemv_silu` and
// `gemv_residual` are to `gemv`. They exist because an ICB command cannot
// carry an inline constant (§3.7c): under replay the classical epilogue arms
// were the largest excluded class on the decode step (90 of 345 positions,
// `inline-constants`), so the fusions the optimizer derives and the
// re-expression that follows them could not meet. The residual keeps its
// binding at 15, after every slot the packed layout uses, so the host binds
// the operand at one index whichever style it runs.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_silu_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueSilu<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  const prologue_t prologue(nullptr, nullptr);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue_t(),
      tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_residual_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    const device T* residual [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    EOS_GATE_ARG) {
    EOS_GATE_CHECK
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueResidual<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  const prologue_t prologue(nullptr, nullptr);
  const epilogue_t epilogue(residual + tid.z * p.out_vec_size);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue,
      tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_mul_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    const device T* gate [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  using epilogue_t = EpilogueMul<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  const prologue_t prologue(nullptr, nullptr);
  const epilogue_t epilogue(gate + tid.z * p.out_vec_size);
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,
                 epilogue_t>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, epilogue,
      tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv_swiglu: TWO matrices per threadgroup, one store — §12.2 row 5 with
/// row 3b's product in the same epilogue (issue #732)
///////////////////////////////////////////////////////////////////////////////

// `silu(gate·x) * (up·x)` for one block of output rows, both weights streamed
// by the same threadgroup over the same input coefficients. There is no
// barrier anywhere inside it: the two sweeps that `gemv_mul` put a barrier
// between are one dispatch here, and the product is applied to the two
// unrounded f32 accumulators — one rounding where the three-dispatch chain
// (`gemv`, `silu`, `gemv`, `bmul`) rounds three times and the two-dispatch
// chain (`gemv_silu`, `gemv_mul`) twice. `GEMVKernel::run`'s K-walk, with
// the matrix loop run twice per block; the row selection, the loop bounds
// and the simdgroup reduction are its, verbatim, so the two kernels cannot
// disagree about which rows a threadgroup owns.
//
// Only the two decode tiles are instantiated, both `bn1`, so the threadgroup
// reduction is compiled out; it is written for the general tile anyway with
// a second region of threadgroup memory, so a third tile is one macro line.
//
// `Prologue` is `GEMVKernel`'s seam, applied the same way: `PrologueNone`
// (what `gemv_swiglu` instantiates) inlines away, and `PrologueNormInline`
// (`gemv_norm_swiglu`) takes `sum(x^2)` off the coefficients both sweeps
// already share and scales BOTH accumulators by `1/rms` before the product —
// `silu(s·g) * (s·u)` for the normalized input, the norm's dispatch removed
// (`docs/tuning.md` §13.5t). It needs every simdgroup to walk all of K, as
// `GEMVKernel`'s does.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    typename AccT = typename GemvDefaultAccT<T>::type,
    typename Prologue = PrologueNone<T, AccT>>
struct GEMVSwigluKernel {
  using acc_type = AccT;
  using base_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, false>;

  static_assert(!Prologue::kInlineNorm || (BN == 1 && SN == 32),
                "the inline norm needs each simdgroup to walk all of K");

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;
  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;
  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? 2 * BN * (blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  static METAL_FUNC void run(
      const device T* mat_up,
      const device T* mat_gate,
      const device T* in_vec,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int matrix_ld,
      threadgroup AccT* tgp_memory,
      thread const Prologue& prologue,
      uint3 tid,
      uint simd_gid,
      uint simd_lid) {
    thread AccT up[TM] = {0};
    thread AccT gate[TM] = {0};
    AccT ssq = 0;
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;
    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (int)(simd_gid / BN) : (int)(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (int)(simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;
    int out_row = tid.x * blockM + bm;
    if (out_row >= out_vec_size) return;
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    mat_up += out_row * matrix_ld;
    mat_gate += out_row * matrix_ld;

    const int loop_stride = blockN;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    for (int i = 0; i < n_iter; ++i) {
      base_t::template load_unsafe<AccT>(in_vec, v_coeff, bn);
      if (Prologue::kInlineNorm) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          ssq += v_coeff[tn] * v_coeff[tn];
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        v_coeff[tn] = prologue(v_coeff[tn], bn + tn);
      }
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        base_t::load_unsafe(mat_up, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          up[tm] += inter[tn] * v_coeff[tn];
        }
        base_t::load_unsafe(mat_gate, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          gate[tm] += inter[tn] * v_coeff[tn];
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }

    if (leftover > 0) {
      base_t::template load_safe<AccT>(in_vec, v_coeff, bn, in_size);
      if (Prologue::kInlineNorm) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          ssq += v_coeff[tn] * v_coeff[tn];
        }
      }
      // Past the end `load_safe` zero-fills, and the prologue must not index
      // its weight there (`GEMVKernel::run`'s guard).
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        if (bn + tn < in_size) {
          v_coeff[tn] = prologue(v_coeff[tn], bn + tn);
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        base_t::load_safe(&mat_up[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          up[tm] += inter[tn] * v_coeff[tn];
        }
        base_t::load_safe(&mat_gate[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          gate[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        up[tm] += simd_shuffle_down(up[tm], sn);
        gate[tm] += simd_shuffle_down(gate[tm], sn);
      }
    }
    if (Prologue::kInlineNorm) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        ssq += simd_shuffle_down(ssq, sn);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_up = tgp_memory + sgN * (blockM + TM) + bm;
      threadgroup AccT* tgp_gate = tgp_memory + BN * (blockM + TM) + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_up[tm] = up[tm];
          tgp_gate[tm] = gate[tm];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              up[tm] += tgp_up[sgn * (blockM + TM) + tm];
              gate[tm] += tgp_gate[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // The product on the two accumulators, one thread per output element,
    // one rounding at the store.
    if (simdN == 0 && thrN == 0) {
      if (Prologue::kInlineNorm) {
        const AccT inv_rms = prologue.scale(ssq, in_vec_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          up[tm] *= inv_rms;
          gate[tm] *= inv_rms;
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = static_cast<T>(silu(gate[tm]) * up[tm]);
      }
    }
  }
};

// The batch offsets and the call, shared by both binding styles; `gemv_body`'s
// shape with the second matrix under the same batch stride.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    typename Prologue,
    typename ShapeP, typename StrideP>
METAL_FUNC void gemv_swiglu_body(
    const device T* mat_up,
    const device T* mat_gate,
    const device T* in_vec,
    device T* out_vec,
    const int in_vec_size,
    const int out_vec_size,
    const int matrix_ld,
    const int batch_ndim,
    ShapeP batch_shape,
    StrideP vector_batch_stride,
    StrideP matrix_batch_stride,
    threadgroup typename GEMVSwigluKernel<T, BM, BN, SM, SN, TM, TN>::acc_type* tgp_memory,
    thread const Prologue& prologue,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  using kernel_t = GEMVSwigluKernel<
      T, BM, BN, SM, SN, TM, TN, typename GemvDefaultAccT<T>::type, Prologue>;
  if (kDoNCBatch) {
    in_vec   += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat_up   += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    mat_gate += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
  } else {
    in_vec   += tid.z * vector_batch_stride[0];
    mat_up   += tid.z * matrix_batch_stride[0];
    mat_gate += tid.z * matrix_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;
  kernel_t::run(mat_up, mat_gate, in_vec, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                tgp_memory, prologue, tid, simd_gid, simd_lid);
}

// `gemv`'s classical signature — `mat` is the UP weight at 0 — with the gate
// weight at 15, after every slot the classical layout declares. `bias`, the
// axpby scalars and the bias strides are accepted and unread: the fused arms
// are instantiated at `axpby0` only, as the two epilogues are.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_swiglu(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    const device T* mat_gate [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)bias; (void)alpha; (void)beta; (void)bias_batch_stride; (void)bias_stride; (void)lid;
  using kernel_t = GEMVSwigluKernel<T, BM, BN, SM, SN, TM, TN>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(nullptr, nullptr);
  gemv_swiglu_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, prologue_t>(
      mat, mat_gate, in_vec, out_vec,
      in_vec_size, out_vec_size, matrix_ld,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, tid, simd_gid, simd_lid);
}

// `gemv_swiglu` with the RMSNorm computed inside it (`PrologueNormInline`):
// `silu(gate·n(x)) * (up·n(x))`, `n(x) = x * w / sqrt(mean(x^2) + eps)`. The
// classical layout, the gate weight at 15 as `gemv_swiglu` binds it, the
// norm's weight and epsilon after it at 16 and 17 (`gemv_norm` binds them at
// 15 and 16; 15 is taken here).
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_norm_swiglu(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    const device T* mat_gate [[buffer(15)]],
    const device T* norm_weight [[buffer(16)]],
    const constant float& norm_eps [[buffer(17)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)bias; (void)alpha; (void)beta; (void)bias_batch_stride; (void)bias_stride; (void)lid;
  using acc_t = typename GemvDefaultAccT<T>::type;
  using prologue_t = PrologueNormInline<T, acc_t>;
  using kernel_t = GEMVSwigluKernel<T, BM, BN, SM, SN, TM, TN, acc_t, prologue_t>;
  gemv_tgp_memory(kernel_t);
  const prologue_t prologue(norm_weight, norm_eps);
  gemv_swiglu_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, prologue_t>(
      mat, mat_gate, in_vec, out_vec,
      in_vec_size, out_vec_size, matrix_ld,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, tid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_swiglu_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    const device T* mat_gate [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)bias; (void)bias_batch_stride; (void)lid;
  using kernel_t = GEMVSwigluKernel<T, BM, BN, SM, SN, TM, TN>;
  using prologue_t = PrologueNone<T, typename kernel_t::acc_type>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  const prologue_t prologue(nullptr, nullptr);
  gemv_swiglu_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, prologue_t>(
      mat, mat_gate, in_vec, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      prologue, tid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv_t kernel: vec [K] × mat [K, N] → out [N]
///////////////////////////////////////////////////////////////////////////////

// As `gemv_body`, for the transposed kernel. Separate rather than templated on
// the kernel type because the two select different `GEMVKernel`/`GEMVTKernel`
// accumulator types for their threadgroup array, and folding them would put a
// second template parameter in the way of that for no saving.
template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby,
    typename ShapeP, typename StrideP>
METAL_FUNC void gemv_t_body(
    const device T* mat,
    const device T* in_vec,
    const device T* bias,
    device T* out_vec,
    const int in_vec_size,
    const int out_vec_size,
    const int matrix_ld,
    const float alpha,
    const float beta,
    const int batch_ndim,
    ShapeP batch_shape,
    StrideP vector_batch_stride,
    StrideP matrix_batch_stride,
    StrideP bias_batch_stride,
    const int bias_stride,
    threadgroup typename GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>::acc_type* tgp_memory,
    uint3 tid,
    uint3 lid,
    uint simd_gid,
    uint simd_lid) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;

  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc<int64_t>(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat    += tid.z * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                tgp_memory,
                tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  gemv_t_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_t_packed(
    const device GemvParams* pp [[buffer(0)]],
    const device T* mat [[buffer(1)]],
    const device T* in_vec [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const device int* batch_shape [[buffer(4)]],
    const device int64_t* vector_batch_stride [[buffer(5)]],
    const device int64_t* matrix_batch_stride [[buffer(6)]],
    const device int64_t* bias_batch_stride [[buffer(7)]],
    const device T* bias [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  gemv_tgp_memory(kernel_t);
  GemvParams p = *pp;
  gemv_t_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby>(
      mat, in_vec, bias, out_vec,
      p.in_vec_size, p.out_vec_size, p.matrix_ld, p.alpha, p.beta,
      p.batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,
      bias_batch_stride, p.bias_stride,
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid, lid, simd_gid, simd_lid);
}

// Reports `GemvParams`'s size and every field's offset *as the compiled kernel
// sees them*, for the host-side layout test to compare against Rust's own
// `size_of`/`offset_of!`.
//
// A `static_assert` proves only that the device side agrees with itself; only
// shipping the numbers across the boundary proves the two sides agree, which is
// the failure this exists to catch — a field at the wrong offset does not
// crash, it reads a well-formed number from the wrong place and computes a
// plausible wrong answer (`DESIGN.md` §3.5, §15.1). Offsets cannot be
// `static_assert`ed in MSL: there is no `<cstddef>` and the null-pointer-member
// form is not a constant expression, so they are measured at runtime from a
// real `thread` instance (`DESIGN.md` §11.3b).
//
// Writes 8 slots; `GEMV_LAYOUT_SLOTS` in `kernels/params.rs` says what each is.
#define gemv_offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void gemv_params_layout(
    device uint *out,
    uint tid [[thread_position_in_grid]]
) {
  if (tid != 0) { return; }
  GemvParams probe_GemvParams;
  out[0] = sizeof(GemvParams);
  out[1] = gemv_offsetof_rt(GemvParams, in_vec_size);
  out[2] = gemv_offsetof_rt(GemvParams, out_vec_size);
  out[3] = gemv_offsetof_rt(GemvParams, matrix_ld);
  out[4] = gemv_offsetof_rt(GemvParams, alpha);
  out[5] = gemv_offsetof_rt(GemvParams, beta);
  out[6] = gemv_offsetof_rt(GemvParams, batch_ndim);
  out[7] = gemv_offsetof_rt(GemvParams, bias_stride);
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiations
///////////////////////////////////////////////////////////////////////////////

// Use decltype-based instantiation (MLX defines.h pattern):
//   template [[host_name(...)]] [[kernel]] decltype(func<...>) func<...>;
// This avoids redeclaring parameter attributes, which Metal rejects.
#define instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, nc, axpby) \
  template [[host_name(                                                               \
      #func "_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                         \
            "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby)]]                         \
  [[kernel]] decltype(func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>)   \
             func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>;

// The packed sibling, named by appending `_packed` to the classical name.
//
// It has to be a *distinct function template* rather than a second
// `[[host_name]]` on the same instantiation: MSL rejects that outright with
// "duplicate explicit instantiation of `gemv_t<bfloat, 1, 16, 4, 8, 4, 4, true,
// true>`", since an explicit instantiation is unique per template-argument list
// however many names are attached to it. Measured while pricing the doubling —
// see `measurements/probes/issue-41/`.
//
// Generated from the same axis lists as the classical names below, so the two
// sets cannot drift: adding a tile configuration adds both, and a name that
// exists on one side only fails `gemv_names_resolve` against the compiled
// library (`DESIGN.md` §8.1b).
#define instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, nc, axpby) \
  template [[host_name(                                                                      \
      #func "_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                                \
            "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby "_packed")]]                      \
  [[kernel]] decltype(func##_packed<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>)  \
             func##_packed<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>;

// The fused-prologue sibling (issue #266, §12.2 #2), named by appending
// `_rmsnorm`.
//
// DELIBERATELY NOT part of `instantiate_gemv_nc_axpby`'s cross. That macro
// emits 4 nc/axpby combinations per tile across 7 tiles and 3 dtypes; adding a
// fused arm to it would emit 84 more names per dtype, of which LFM2 dispatches
// **two**. §11.3g measured the doubling of this family at +730 ms of cold
// compile and +4.01 MiB, which is the strongest recorded case for not
// instantiating what nothing asks for, and §11.3k decided a scope on exactly
// that arithmetic.
//
// So the fused arm is instantiated only at `nc=0, axpby=0` and only at the two
// tiles the decode path selects -- read from `call_mlx_gemv`'s own tile rule
// and confirmed against the committed census
// (`measurements/issue-249-raw/kernel-census.txt`, which records `bm4` and
// `bm8` and no others).
#define instantiate_gemv_rmsnorm_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                        \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                      \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_rmsnorm")]]                             \
  [[kernel]] decltype(gemv_rmsnorm<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_rmsnorm<itype, bm, bn, sm, sn, tm, tn, false, false>;

// The fused-epilogue sibling (issue #484, §12.2 #3), named by appending
// `_silu`.
//
// Scoped on exactly #266's arithmetic and for exactly its reason: the full
// nc/axpby cross over 7 tiles and 3 dtypes would emit 84 more names per dtype
// of which LFM2 dispatches **one**, against §11.3g's measured +730 ms of cold
// compile and +4.01 MiB for doubling this family.
//
// `bm8` is the tile `call_mlx_gemv` selects for w1 (K=2048, N=10752) and is
// the only one the decode path reaches. **`bm4` is instantiated anyway, and
// not as a convenience** -- it is what lets the parity fixture cover a second
// tile, and §9.2c's rule is that a fixture built only from the model's own
// shapes is blind to a whole defect class. f32 is instantiated alongside f16
// for the same reason #266 gives: it is the precision §2.3.5a's discriminator
// wants.
#define instantiate_gemv_silu_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                     \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                   \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_silu")]]                             \
  [[kernel]] decltype(gemv_silu<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_silu<itype, bm, bn, sm, sn, tm, tn, false, false>;

// The fused residual-epilogue sibling (issue #509), named by appending
// `_residual`. Scoped exactly as `_silu` and `_rmsnorm` are, for §11.3g's
// measured +730 ms / +4.01 MiB reason -- two tiles per dtype and no more.
//
// **No longer behind a `-D`** (issue #516): see the kernel's own comment above
// for why the gate is removed rather than made legible. The cost of the arm
// existing unconditionally is measured at **6 names of 307, +2.04 % of this
// file's metallib**, which is ~74x below the widening §11.3g already accepts
// on this same file.
//
// **`bm4` is the tile that matters here and it is not the fixture's.** All
// three producers of a residual add sit at `N_out` = 2048 -- `w2`
// (K=10752), `conv.out_proj` and `out_proj` (K=2048) -- which
// `derive_gemv_tile` sends to `bm4`. `bm8` is instantiated beside it so the
// parity fixture covers a second tile, per §9.2c's rule that a fixture built
// only from the model's own shapes is blind to a whole defect class. That is
// the mirror of `_silu`, where `bm8` is the decode tile and `bm4` the
// fixture's.
#define instantiate_gemv_residual_helper(nm, itype, bm, bn, sm, sn, tm, tn)      \
  template [[host_name(                                                         \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                       \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_residual")]]                             \
  [[kernel]] decltype(gemv_residual<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_residual<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_residual_blocks(nm, itype)              \
  instantiate_gemv_residual_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_residual_helper(nm, itype, 8, 1, 1, 32, 4, 4)

// The gate-product epilogue (issue #732), named by appending `_mul`. `bm8`
// is the decode tile: the site is `w3` (K=2048, N=10752), the same shape as
// `w1`'s; `bm4` is instantiated beside it for the parity fixture's second
// tile, as the two arms above are.
#define instantiate_gemv_mul_helper(nm, itype, bm, bn, sm, sn, tm, tn)      \
  template [[host_name(                                                    \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                  \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_mul")]]                             \
  [[kernel]] decltype(gemv_mul<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_mul<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_mul_blocks(nm, itype)              \
  instantiate_gemv_mul_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_mul_helper(nm, itype, 8, 1, 1, 32, 4, 4)

// The two-weight SwiGLU kernel (issue #732, §12.2 row 5 + row 3b), named by
// appending `_swiglu`; the same two tiles.
#define instantiate_gemv_swiglu_helper(nm, itype, bm, bn, sm, sn, tm, tn)      \
  template [[host_name(                                                       \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                     \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_swiglu")]]                             \
  [[kernel]] decltype(gemv_swiglu<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_swiglu<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_swiglu_blocks(nm, itype)              \
  instantiate_gemv_swiglu_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_swiglu_helper(nm, itype, 8, 1, 1, 32, 4, 4)

// `gemv_norm_swiglu`: the dispatcher builds `_norm` then the epilogue's
// `_swiglu`, the order `gemv_norm_silu` takes; the same two tiles.
#define instantiate_gemv_norm_swiglu_helper(nm, itype, bm, bn, sm, sn, tm, tn)      \
  template [[host_name("gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn          \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_norm_swiglu")]]                              \
  [[kernel]] decltype(gemv_norm_swiglu<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_norm_swiglu<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_norm_swiglu_blocks(nm, itype)              \
  instantiate_gemv_norm_swiglu_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_norm_swiglu_helper(nm, itype, 8, 1, 1, 32, 4, 4)

// The packed siblings of the two epilogue arms (issue #732 P3), named as every
// packed name is: `_packed` LAST, `…_nc0_axpby0_silu_packed` — the replay
// executor classifies a position by that suffix (`icb.rs` `is_packed`), so a
// name carrying it anywhere else is a packed dispatch it cannot see. The same
// two tiles as the classical arms, so `gemv_packed_names_resolve` can pin both
// sets from one list.
#define instantiate_gemv_silu_packed_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                             \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                           \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_silu_packed")]]                              \
  [[kernel]] decltype(gemv_silu_packed<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_silu_packed<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_residual_packed_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                                 \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                               \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_residual_packed")]]                              \
  [[kernel]] decltype(gemv_residual_packed<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_residual_packed<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_mul_packed_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                            \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                          \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_mul_packed")]]                              \
  [[kernel]] decltype(gemv_mul_packed<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_mul_packed<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_swiglu_packed_helper(nm, itype, bm, bn, sm, sn, tm, tn)     \
  template [[host_name(                                                               \
      "gemv_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                             \
      "_tm" #tm "_tn" #tn "_nc0_axpby0_swiglu_packed")]]                              \
  [[kernel]] decltype(gemv_swiglu_packed<itype, bm, bn, sm, sn, tm, tn, false, false>) \
             gemv_swiglu_packed<itype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_epilogue_packed_blocks(nm, itype)              \
  instantiate_gemv_silu_packed_helper(nm, itype, 4, 1, 1, 32, 4, 4)     \
  instantiate_gemv_silu_packed_helper(nm, itype, 8, 1, 1, 32, 4, 4)     \
  instantiate_gemv_residual_packed_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_residual_packed_helper(nm, itype, 8, 1, 1, 32, 4, 4) \
  instantiate_gemv_mul_packed_helper(nm, itype, 4, 1, 1, 32, 4, 4)      \
  instantiate_gemv_mul_packed_helper(nm, itype, 8, 1, 1, 32, 4, 4)      \
  instantiate_gemv_swiglu_packed_helper(nm, itype, 4, 1, 1, 32, 4, 4)   \
  instantiate_gemv_swiglu_packed_helper(nm, itype, 8, 1, 1, 32, 4, 4)

#define instantiate_gemv_nc_axpby(func, nm, itype, bm, bn, sm, sn, tm, tn)    \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 0)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 1)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 0)      \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 1)      \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_packed_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 1)

// gemv blocks: mat×vec (output size = M)
// bm=4/8 for large output; bm=1,bn=8 for K-heavy; bm=1,bn=1,sm=8,sn=4 for small K
#define instantiate_gemv_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 8,  1, 1, 32, 4, 4)

// gemv_t blocks: vec×mat (output size = N)
// bn=2/4/16 for various output sizes; sm/sn tuned for K size
#define instantiate_gemv_t_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 1)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  4,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  4, 8, 4, 4)


///////////////////////////////////////////////////////////////////////////////
/// The mixed-precision arms: `half` weights, `float` vector in and out
/// (lloom's Qwen4-exp dense projections — the HyperConnection low-rank pair,
/// the routers, DeltaNet's α/β — are BF16 in the checkpoint and the model's
/// activations are f32). The classical path cast the activation to half, ran
/// `gemv_float16` and cast the result back: two launches around every one of
/// ~375 gemvs a token. These read the f32 vector as it lies and store f32,
/// accumulating in f32 as the half arm does. `alpha` is the epilogue's
/// pre-scale on the `_silu`/`_sigmoid` arms (`kDoAxpby` is never set here).
///////////////////////////////////////////////////////////////////////////////

#define gemv_mixed_wrapper(func, EPI, EPI_CTOR)                                   \
template <                                                                        \
    typename T, typename VT,                                                      \
    const int BM, const int BN,                                                   \
    const int SM, const int SN,                                                   \
    const int TM, const int TN,                                                   \
    const bool kDoNCBatch,                                                        \
    const bool kDoAxpby>                                                          \
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]                       \
void func(                                                                        \
    const device T* mat [[buffer(0)]],                                            \
    const device VT* in_vec [[buffer(1)]],                                        \
    const device VT* bias [[buffer(2)]],                                          \
    device VT* out_vec [[buffer(3)]],                                             \
    const constant int& in_vec_size [[buffer(4)]],                                \
    const constant int& out_vec_size [[buffer(5)]],                               \
    const constant int& matrix_ld [[buffer(6)]],                                  \
    const constant float& alpha [[buffer(7)]],                                    \
    const constant float& beta [[buffer(8)]],                                     \
    const constant int& batch_ndim [[buffer(9)]],                                 \
    const constant int* batch_shape [[buffer(10)]],                               \
    const constant int64_t* vector_batch_stride [[buffer(11)]],                   \
    const constant int64_t* matrix_batch_stride [[buffer(12)]],                   \
    const constant int64_t* bias_batch_stride [[buffer(13)]],                     \
    const constant int& bias_stride [[buffer(14)]],                               \
    uint3 tid [[threadgroup_position_in_grid]],                                   \
    uint3 lid [[thread_position_in_threadgroup]],                                 \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                             \
    uint simd_lid [[thread_index_in_simdgroup]],                                  \
    EOS_GATE_ARG) {                                                               \
    EOS_GATE_CHECK                                                                \
  using acc_t = typename GemvDefaultAccT<T>::type;                                \
  using prologue_t = PrologueNone<T, acc_t>;                                      \
  using epilogue_t = EPI<T, acc_t>;                                               \
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby, acc_t,         \
                              prologue_t, epilogue_t, VT>;                        \
  gemv_tgp_memory(kernel_t);                                                      \
  const prologue_t prologue(nullptr, nullptr);                                    \
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,          \
                 epilogue_t>(                                                     \
      mat, in_vec, bias, out_vec,                                                 \
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,                          \
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,          \
      bias_batch_stride, bias_stride,                                             \
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,                         \
      prologue, EPI_CTOR,                                                         \
      tid, lid, simd_gid, simd_lid);                                              \
}

gemv_mixed_wrapper(gemv_mixed, EpilogueNone, epilogue_t())
gemv_mixed_wrapper(gemv_mixed_silu, EpilogueSiluScale, epilogue_t(alpha))
gemv_mixed_wrapper(gemv_mixed_sigmoid, EpilogueSigmoidScale, epilogue_t(alpha))

#define instantiate_gemv_mixed_helper(func, nm, wtype, vtype, bm, bn, sm, sn, tm, tn, suffix) \
  template [[host_name(                                                                      \
      "gemv_mixed_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                              \
            "_tm" #tm "_tn" #tn "_nc0_axpby0" suffix)]]                                      \
  [[kernel]] decltype(func<wtype, vtype, bm, bn, sm, sn, tm, tn, false, false>)               \
             func<wtype, vtype, bm, bn, sm, sn, tm, tn, false, false>;

#define instantiate_gemv_mixed_tile(nm, wtype, vtype, bm, bn, sm, sn, tm, tn)                 \
  instantiate_gemv_mixed_helper(gemv_mixed, nm, wtype, vtype, bm, bn, sm, sn, tm, tn, "")     \
  instantiate_gemv_mixed_helper(gemv_mixed_silu, nm, wtype, vtype, bm, bn, sm, sn, tm, tn, "_silu") \
  instantiate_gemv_mixed_helper(gemv_mixed_sigmoid, nm, wtype, vtype, bm, bn, sm, sn, tm, tn, "_sigmoid")

// The mixed GEMV with the grouped norm inside it (`PrologueGroupNorm`): the
// classical layout, then the norm weight (`VT`, `[groups · group]`) at 15, its
// epsilon at 16, the group size and count at 17 and 18, the scales' output at
// 19 and whether to write it at 20.
#define gemv_mixed_gnorm_wrapper(func, EPI, EPI_CTOR)                             \
template <                                                                        \
    typename T, typename VT,                                                      \
    const int BM, const int BN,                                                   \
    const int SM, const int SN,                                                   \
    const int TM, const int TN,                                                   \
    const bool kDoNCBatch,                                                        \
    const bool kDoAxpby>                                                          \
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]                       \
void func(                                                                        \
    const device T* mat [[buffer(0)]],                                            \
    const device VT* in_vec [[buffer(1)]],                                        \
    const device VT* bias [[buffer(2)]],                                          \
    device VT* out_vec [[buffer(3)]],                                             \
    const constant int& in_vec_size [[buffer(4)]],                                \
    const constant int& out_vec_size [[buffer(5)]],                               \
    const constant int& matrix_ld [[buffer(6)]],                                  \
    const constant float& alpha [[buffer(7)]],                                    \
    const constant float& beta [[buffer(8)]],                                     \
    const constant int& batch_ndim [[buffer(9)]],                                 \
    const constant int* batch_shape [[buffer(10)]],                               \
    const constant int64_t* vector_batch_stride [[buffer(11)]],                   \
    const constant int64_t* matrix_batch_stride [[buffer(12)]],                   \
    const constant int64_t* bias_batch_stride [[buffer(13)]],                     \
    const constant int& bias_stride [[buffer(14)]],                               \
    const device VT* norm_weight [[buffer(15)]],                                  \
    const constant float& norm_eps [[buffer(16)]],                                \
    const constant int& norm_group [[buffer(17)]],                                \
    const constant int& norm_groups [[buffer(18)]],                               \
    device float* scales_out [[buffer(19)]],                                      \
    const constant int& write_scales [[buffer(20)]],                              \
    uint3 tid [[threadgroup_position_in_grid]],                                   \
    uint3 lid [[thread_position_in_threadgroup]],                                 \
    uint simd_gid [[simdgroup_index_in_threadgroup]],                             \
    uint simd_lid [[thread_index_in_simdgroup]],                                  \
    EOS_GATE_ARG) {                                                               \
    EOS_GATE_CHECK                                                                \
  static_assert(BM * BN <= kGroupNormMaxSimd, "the scratch holds 8 simdgroups");  \
  using acc_t = typename GemvDefaultAccT<T>::type;                                \
  using prologue_t = PrologueGroupNorm<VT, acc_t>;                                \
  using epilogue_t = EPI<T, acc_t>;                                               \
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby, acc_t,         \
                              prologue_t, epilogue_t, VT>;                        \
  gemv_tgp_memory(kernel_t);                                                      \
  threadgroup acc_t gn_scratch[kGroupNormScratch];                                \
  const prologue_t prologue(norm_weight, norm_eps, norm_group, norm_groups,       \
                            write_scales != 0 ? scales_out : nullptr, gn_scratch);\
  gemv_body<T, BM, BN, SM, SN, TM, TN, kDoNCBatch, kDoAxpby, prologue_t,          \
                 epilogue_t>(                                                     \
      mat, in_vec, bias, out_vec,                                                 \
      in_vec_size, out_vec_size, matrix_ld, alpha, beta,                          \
      batch_ndim, batch_shape, vector_batch_stride, matrix_batch_stride,          \
      bias_batch_stride, bias_stride,                                             \
      kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,                         \
      prologue, EPI_CTOR,                                                         \
      tid, lid, simd_gid, simd_lid);                                              \
}

gemv_mixed_gnorm_wrapper(gemv_mixed_gnorm, EpilogueNone, epilogue_t())
gemv_mixed_gnorm_wrapper(gemv_mixed_gnorm_silu, EpilogueSiluScale, epilogue_t(alpha))

// Only the tile Qwen4-exp's HyperConnection GEMVs select (`[320, 10240]` and
// `[4, 10240]`: `k >= 16 n`, `bm1 bn8 sm1 sn32 tm4`): 256 threads, the
// threadgroup `hc_norm_f32` reduces with. The host refuses any other.
instantiate_gemv_mixed_helper(gemv_mixed_gnorm, f16w_f32, half, float, 1, 8, 1, 32, 4, 4, "_gnorm")
instantiate_gemv_mixed_helper(gemv_mixed_gnorm_silu, f16w_f32, half, float, 1, 8, 1, 32, 4, 4, "_silu_gnorm")

// The seven `gemv` tiles `derive_gemv_tile` can select, at `tm` 4 and 1.
#define instantiate_gemv_mixed_blocks(nm, wtype, vtype)                  \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 1,  8, 1, 32, 4, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 1,  8, 1, 32, 1, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 1,  1, 8,  4, 4, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 1,  1, 8,  4, 1, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 4,  1, 1, 32, 1, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 4,  1, 1, 32, 4, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 8,  1, 1, 32, 4, 4)      \
  instantiate_gemv_mixed_tile(nm, wtype, vtype, 8,  1, 1, 32, 1, 4)

instantiate_gemv_mixed_blocks(f16w_f32, half, float)

instantiate_gemv_blocks(float32, float)
instantiate_gemv_blocks(float16, half)
instantiate_gemv_blocks(bfloat16, bfloat)

// The row axis (`GEMVKernel`'s `NR`, `docs/tuning.md` §13.5y): `_r<NR>`
// after the classical name, at the two decode tiles (the only ones whose
// simdgroup walks all of K), `nc0_axpby0`, split params, half only — the
// dense f16 matmul is its one caller (`call_mlx_gemv_rows`). Every count
// from 2 to 16, so a window of `m` rows is one group of exactly `m`.
#define instantiate_gemv_rows_helper(nm, itype, bm, nr)                        \
  template [[host_name("gemv_" #nm "_bm" #bm                                   \
      "_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_r" #nr)]]                              \
  [[kernel]] decltype(gemv<itype, bm, 1, 1, 32, 4, 4, false, false, nr>)       \
             gemv<itype, bm, 1, 1, 32, 4, 4, false, false, nr>;

#define instantiate_gemv_rows_tile(nm, itype, bm)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 2)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 3)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 4)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 5)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 6)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 7)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 8)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 9)   \
  instantiate_gemv_rows_helper(nm, itype, bm, 10)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 11)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 12)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 13)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 14)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 15)  \
  instantiate_gemv_rows_helper(nm, itype, bm, 16)

instantiate_gemv_rows_tile(float16, half, 4)
instantiate_gemv_rows_tile(float16, half, 8)

// The two decode-path tiles, per dtype. `bm8` serves w1/w3 (N=10752) and
// conv.in_proj (N=6144); `bm4` serves q_proj (N=2048) and k/v_proj (N=512).
// f32 is instantiated alongside f16 so the parity fixture can run at f32,
// which is the precision §2.3.5a's discriminator wants.
#define instantiate_gemv_rmsnorm_blocks(nm, itype)              \
  instantiate_gemv_rmsnorm_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_rmsnorm_helper(nm, itype, 8, 1, 1, 32, 4, 4)

instantiate_gemv_rmsnorm_blocks(float32, float)
instantiate_gemv_rmsnorm_blocks(float16, half)
instantiate_gemv_rmsnorm_blocks(bfloat16, bfloat)

// The fused-epilogue tiles. `bm8` is the one w1 selects and therefore the only
// one decode reaches; `bm4` is the fixture's second tile (see the helper).
// The inline-norm arms (`gemv_norm_wrapper`), at the two decode tiles: the
// only tiles whose simdgroup walks all of K.
#define instantiate_gemv_norm_helper(func, nm, itype, bm, suffix)               \
  template [[host_name(                                                        \
      "gemv_" #nm "_bm" #bm "_bn1_sm1_sn32_tm4_tn4_nc0_axpby0" suffix)]]        \
  [[kernel]] decltype(func<itype, bm, 1, 1, 32, 4, 4, false, false>)           \
             func<itype, bm, 1, 1, 32, 4, 4, false, false>;

#define instantiate_gemv_norm_blocks(nm, itype)                                \
  instantiate_gemv_norm_helper(gemv_norm, nm, itype, 4, "_norm")               \
  instantiate_gemv_norm_helper(gemv_norm, nm, itype, 8, "_norm")               \
  instantiate_gemv_norm_helper(gemv_norm_silu, nm, itype, 4, "_norm_silu")     \
  instantiate_gemv_norm_helper(gemv_norm_silu, nm, itype, 8, "_norm_silu")

instantiate_gemv_norm_blocks(float32, float)
instantiate_gemv_norm_blocks(float16, half)
instantiate_gemv_norm_blocks(bfloat16, bfloat)

#define instantiate_gemv_silu_blocks(nm, itype)              \
  instantiate_gemv_silu_helper(nm, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_silu_helper(nm, itype, 8, 1, 1, 32, 4, 4)

instantiate_gemv_silu_blocks(float32, float)
instantiate_gemv_silu_blocks(float16, half)
instantiate_gemv_silu_blocks(bfloat16, bfloat)

// Issue #509, unconditional since #516 -- six names, the same two tiles per
// dtype `_silu` and `_rmsnorm` above instantiate. The `-D` that used to gate
// this is gone; §7.1's dispatch tier is now the only tier that decides whether
// the fused arm runs, and `Config::gemv_residual` is the only switch.
instantiate_gemv_residual_blocks(float32, float)
instantiate_gemv_residual_blocks(float16, half)
instantiate_gemv_residual_blocks(bfloat16, bfloat)

instantiate_gemv_mul_blocks(float32, float)
instantiate_gemv_mul_blocks(float16, half)
instantiate_gemv_mul_blocks(bfloat16, bfloat)

instantiate_gemv_swiglu_blocks(float32, float)
instantiate_gemv_swiglu_blocks(float16, half)
instantiate_gemv_swiglu_blocks(bfloat16, bfloat)
instantiate_gemv_norm_swiglu_blocks(float32, float)
instantiate_gemv_norm_swiglu_blocks(float16, half)
instantiate_gemv_norm_swiglu_blocks(bfloat16, bfloat)

// Their packed siblings (issue #732 P3): twelve names, the same two tiles.
instantiate_gemv_epilogue_packed_blocks(float32, float)
instantiate_gemv_epilogue_packed_blocks(float16, half)
instantiate_gemv_epilogue_packed_blocks(bfloat16, bfloat)

instantiate_gemv_t_blocks(float32, float)
instantiate_gemv_t_blocks(float16, half)
instantiate_gemv_t_blocks(bfloat16, bfloat)
