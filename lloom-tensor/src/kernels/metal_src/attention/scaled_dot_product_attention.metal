// Updated from MLX commit has f70764a

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;
typedef half float16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

// ============ "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"

struct MLXFastAttentionParams {
  const int M;
  const int N;
  const int K;

  const int ldq; // ldq == ldo
  const int ldk;
  const int ldv;
  const int lds;
  const int ldo;

  const int tiles_n;
  const int tiles_m;

  const int batch_stride_q;
  const int batch_stride_k;
  const int batch_stride_v;
  const int batch_stride_o;

  const int swizzle_log;
  const int gemm_n_iterations_aligned;
  const int gemm_k_iterations_aligned;
  const int gemm_sv_m_block_iterations;

  const int batch_ndim;
  const float alpha;
  const float softcapping;
};

struct MLXScaledDotProductAttentionParams {
  // Associated dimensions & transposition information
  const uint QUERY_SEQUENCE_LENGTH = 1;
  const uint N_Q_HEADS = 32;
  const uint N_KV_HEADS = 32;
  const uint KV_TILES = 1;
  const float INV_ALPHA = 0.08838834764831843f;
};

// ============ "mlx/backend/metal/kernels/scaled_dot_product_attention_params.sdpa_vector"

constant bool sdpa_vector_has_mask [[function_constant(20)]];

// Packed parameter struct, mirrored in `kernels/params.rs` and checked against
// it by `sdpa_params_layout`.
//
// `MTLIndirectComputeCommand` has no `setBytes` in any form (`DESIGN.md`
// §3.7c), so a kernel taking its scalars as `constant int &` cannot be encoded
// into an ICB at all. #97 put `sdpa_vector` on 8 decode dispatch positions per
// token (`DESIGN.md` §6.2a), which made this the one decode-path family
// without a packed sibling and therefore the ICB path's remaining constants
// gate (§11.3h).
//
// **The width matters here, and it is the trap #38 found.** `k_stride` and
// `v_stride` are `size_t` — 8 bytes — where `gqa_factor` and `N` are `int`.
// `call_sdpa_vector` binds them as `usize` and `i32` respectively, so the
// classical path is already correct; a `usize` bound to a `constant int &`
// would be benign under `setBytes` (which writes 8 bytes and lets a
// little-endian 4-byte read take the low half) and **wrong once packed**.
// Checked by enumerating every `constant &` parameter in this file rather than
// assumed.
//
// The mixed widths are why this struct pads: `{i32, i32, u64, u64, f32, f32}`
// is 32 bytes where its fields sum to 28, with four bytes of tail padding to
// the struct's own 8-byte alignment. That number is not obvious by inspection,
// which is the argument for shipping it across the boundary rather than
// asserting it on each side.
//
// The **mask** arguments are deliberately not fields. They are gated on the
// `sdpa_vector_has_mask` function constant, so when it is false they do not
// exist in the signature at all and consume no slot — which is why the `()`
// hazard #41 found (an unbound argument that binds nothing and consumes no
// slot, shifting every later binding one low and hanging the GPU) does not
// arise: nothing passes `()` here, and the function constant removes the
// parameters rather than leaving them unbound. `call_sdpa_vector` sets the
// constant to `false` unconditionally, so the packed instantiations below are
// reachable only in that configuration; a masked packed variant would need
// `mask_seq_stride` and `mask_head_stride` in the struct and the mask itself
// left as a buffer binding, which is a change to make when a caller wants it.
struct SdpaVectorParams {
  int gqa_factor;
  int N;
  size_t k_stride;
  size_t v_stride;
  float scale;
  float softcapping;
};

// §11.3d's factoring rather than §11.3f's: `sdpa_vector` declares threadgroup
// memory, and MSL permits a `threadgroup` variable only inside a
// `[[kernel]]`-qualified function, so the whole-body-plus-two-thin-wrappers
// shape does not compile. The arrays are declared in each entry point and
// passed to the shared body as `threadgroup` pointers, which is the same seam
// `reduce.metal` uses (`reduce()` takes its shared memory as a parameter).
//
// The struct is taken by `thread const &` and its fields copied into locals at
// entry, per `DESIGN.md` §11.3e: one structure load into registers rather than
// a dereference per use. That matters here — `N` bounds the key loop and
// `softcapping` is tested inside it — because `constant` is a cached read-only
// space and `device` is not.
template <typename T, int D>
METAL_FUNC void sdpa_vector_body(
    thread const SdpaVectorParams& p,
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    const device bool* mask,
    int mask_seq_stride,
    int mask_head_stride,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;

  typedef float U;

  const int gqa_factor = p.gqa_factor;
  const int N = p.N;
  const size_t k_stride = p.k_stride;
  const size_t v_stride = p.v_stride;
  const float scale = p.scale;
  const float softcapping = p.softcapping;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + simd_gid * D + simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + simd_gid * D + simd_lid * elem_per_thread;
  if (sdpa_vector_has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_seq_stride;
  }
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    if (!sdpa_vector_has_mask || mask[0]) {
      // Read the key
      for (int j = 0; j < elem_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < elem_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (softcapping != 1.) {
        score = precise::tanh(score);
        score = score * softcapping;
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < elem_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += stride;
    values += stride;
    if (sdpa_vector_has_mask) {
      mask += BN * mask_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& softcapping,
    const device bool* mask [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  // Brace-initialized positionally, in the classical argument order. C++11
  // narrowing on this initializer is a third guard beside the `static_assert`
  // and the cross-boundary layout check: inserting a field of a different type,
  // or reordering two fields whose types differ, fails to compile here before
  // either of the other two runs (`DESIGN.md` §11.3k finding 2).
  SdpaVectorParams p{gqa_factor, N, k_stride, v_stride, scale, softcapping};
  sdpa_vector_body<T, D>(
      p,
      queries,
      keys,
      values,
      out,
      sdpa_vector_has_mask ? mask : nullptr,
      sdpa_vector_has_mask ? mask_seq_stride : 0,
      sdpa_vector_has_mask ? mask_head_stride : 0,
      outputs,
      max_scores,
      sum_exp_scores,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int D>
[[kernel]] void sdpa_vector_packed(
    const device SdpaVectorParams* pp [[buffer(0)]],
    const device T* queries [[buffer(1)]],
    const device T* keys [[buffer(2)]],
    const device T* values [[buffer(3)]],
    device T* out [[buffer(4)]],
    const device bool* mask [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  SdpaVectorParams p = *pp;
  sdpa_vector_body<T, D>(
      p,
      queries,
      keys,
      values,
      out,
      sdpa_vector_has_mask ? mask : nullptr,
      sdpa_vector_has_mask ? mask_seq_stride : 0,
      sdpa_vector_has_mask ? mask_head_stride : 0,
      outputs,
      max_scores,
      sum_exp_scores,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& softcapping,
    const device bool* mask [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],
    const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int stride = BN * D;
  constexpr int blocks = 32;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  values += kv_head_idx * v_stride + (block_idx * BN + simd_gid) * D +
      simd_lid * elem_per_thread;
  out += head_idx * blocks * D + block_idx * D + simd_lid * elem_per_thread;
  if (sdpa_vector_has_mask) {
    mask += head_idx * mask_head_stride +
        (block_idx * BN + simd_gid) * mask_seq_stride;
  }
  sums += head_idx * blocks + block_idx;
  maxs += head_idx * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    if (!sdpa_vector_has_mask || mask[0]) {
      // Read the key
      for (int i = 0; i < elem_per_thread; i++) {
        k[i] = keys[i];
      }

      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < elem_per_thread; i++) {
        score += q[i] * k[i];
      }
      score = simd_sum(score);
      if (softcapping != 1.) {
        score = precise::tanh(score);
        score = score * softcapping;
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < elem_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * stride;
    values += blocks * stride;
    if (sdpa_vector_has_mask) {
      mask += BN * blocks * mask_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.y;
  partials += head_idx * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  out += head_idx * D + simd_gid * elem_per_thread;

  // First everybody reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ============ quantized KV (issue #218, `DESIGN.md` §10.7, §17 Phase 5 item 18)

// The KV cache stored as integer codes plus one (scale, bias) pair per group,
// with **the group equal to one head vector**.
//
// # Why the group is the head vector, and why that is not a tuning choice
//
// `head_dim` is 64 (§5.3), and §16 #6 records the arithmetic: a group size of
// 128 **does not divide it**, so AWQ's default and llama.cpp's 128-wide blocks
// are unavailable along this axis. MLX defaults to 64, which is exactly one
// group per head vector — §10.7a calls that agreement a *coincidence* that
// would fail at head_dim 128, so it is asserted on the host side rather than
// relied on here.
//
// The consequence that matters for this kernel: `scale` and `bias` are
// **uniform across the 32 lanes** that cooperate on one key, so a dequantized
// dot product needs no cross-lane traffic beyond the `simd_sum` the f16 path
// already does.
//
// # The scales are a parallel plane, not interleaved
//
// §17 item 18(c) and §10.7a: interleaving a scale beside each head vector
// breaks `sdpa_vector`'s contiguous `keys += stride` walk, which §10.3b reads
// out of the kernel and §9.1d records as a *requirement* rather than a
// preference (`head_dim × dtype` is exactly one 128 B cache line at f16). The
// codes therefore keep the `[B, H, T, D]` order the f16 cache has, at a
// narrower element, and the scales live in their own `[B, H, T]` plane.
//
// # Determinism
//
// **No variant tier is owed** (§16 #6, §10.7a): a KV scale is computed once at
// write time and only read thereafter, so it sits *outside* the per-token
// reduction §2.3.3 #1 governs. What sits inside is the dequantized dot
// product, and the loop below is `sdpa_vector_body`'s line for line — same
// `typedef float U`, same `simd_sum` tree, same online-softmax rescale. The
// only change is where `k[j]` and `values[j]` come from.
//
// **The dequant is per element on purpose.** For gs == D the score factorises
// into `s * sum(qv*c) + b * sum(qv)`, which would hoist the scale out of the
// key loop for one extra `simd_sum` — algebraically exact and a *different*
// rounding. This kernel does not take it: the issue's scope is *get it correct
// first*, and dequantizing into the same accumulator makes parity against a
// scalar reference exact rather than exact-modulo-reassociation.
struct SdpaQuantParams {
  int gqa_factor;
  int N;
  // Per-head strides, in ELEMENTS of the plane they address. The code plane's
  // element is a byte at int8 and half a byte at int4, so `k_code_stride` is a
  // *byte* stride and `k_scale_stride` is an f16-element stride: two different
  // units in adjacent fields, which is exactly the shape #38's width trap took.
  // Both are `size_t` and both are read, so `sdpa_quant_params_layout` can see
  // a transposition.
  size_t k_code_stride;
  size_t v_code_stride;
  size_t k_scale_stride;
  size_t v_scale_stride;
  float scale;
  float softcapping;
};

// The write side's parameters. Separate from `SdpaQuantParams` because the two
// kernels are dispatched independently and share no field whose meaning is the
// same: a write knows a destination offset and one plane, a read knows N and
// both planes.
struct SdpaQuantWriteParams {
  // Per-head stride of the f16 source, in elements.
  size_t src_head_stride;
  // Per-head stride of the code plane, in BYTES.
  size_t code_head_stride;
  // Per-head stride of the scale and bias planes, in ELEMENTS.
  size_t scale_head_stride;
  // Token index this append starts at — the slot's live length before it.
  int dst_offset;
  int _pad;
};

// Unpack one code from the packed plane.
//
// `BITS` is 8 or 4 and is a template parameter rather than a field, so the
// shift-and-mask disappears at int8 and the two widths are two instantiations
// rather than one kernel with a branch in its inner loop (§7.2: dtype and
// memory layout are compile-tier).
template <int BITS>
METAL_FUNC uint sdpa_unpack_code(const device uchar* codes, int idx);

template <>
METAL_FUNC uint sdpa_unpack_code<8>(const device uchar* codes, int idx) {
  return uint(codes[idx]);
}

template <>
METAL_FUNC uint sdpa_unpack_code<4>(const device uchar* codes, int idx) {
  // Two codes per byte, low nibble first. The low/high choice is mirrored by
  // the writer in `quantize_kv.metal` and by the CPU reference; a swap is
  // caught by the parity test, whose fixture uses values that differ between
  // the nibbles.
  const uchar packed = codes[idx >> 1];
  return uint((idx & 1) ? (packed >> 4) : (packed & 0x0F));
}

// Bytes of code plane occupied by `n` codes at this width.
template <int BITS>
METAL_FUNC int sdpa_code_bytes(int n) {
  return (BITS == 8) ? n : (n >> 1);
}

template <typename T, int D, int BITS>
METAL_FUNC void sdpa_vector_quant_body(
    thread const SdpaQuantParams& p,
    const device T* queries,
    const device uchar* k_codes,
    const device T* k_scales,
    const device T* k_biases,
    const device uchar* v_codes,
    const device T* v_scales,
    const device T* v_biases,
    device T* out,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  const int gqa_factor = p.gqa_factor;
  const int N = p.N;
  const float scale = p.scale;
  const float softcapping = p.softcapping;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;

  queries += head_idx * D + simd_lid * elem_per_thread;
  out += head_idx * D + simd_gid * elem_per_thread;

  // The code planes are indexed in codes and converted to bytes once, so an
  // int4 plane advances half as far per token as an int8 one and the arithmetic
  // says so in one place.
  const int lane_first_code = simd_lid * elem_per_thread;
  const device uchar* k_code_base =
      k_codes + kv_head_idx * p.k_code_stride + sdpa_code_bytes<BITS>(simd_gid * D);
  const device uchar* v_code_base =
      v_codes + kv_head_idx * p.v_code_stride + sdpa_code_bytes<BITS>(simd_gid * D);

  // One (scale, bias) per (head, token): the group IS the head vector.
  const device T* k_scale_p = k_scales + kv_head_idx * p.k_scale_stride + simd_gid;
  const device T* k_bias_p = k_biases + kv_head_idx * p.k_scale_stride + simd_gid;
  const device T* v_scale_p = v_scales + kv_head_idx * p.v_scale_stride + simd_gid;
  const device T* v_bias_p = v_biases + kv_head_idx * p.v_scale_stride + simd_gid;

  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // Byte advance per BN keys, per plane. `stride = BN * D` in the f16 kernel.
  const int code_step = sdpa_code_bytes<BITS>(BN * D);

  for (int i = simd_gid; i < N; i += BN) {
    const U k_scale = static_cast<U>(*k_scale_p);
    const U k_bias = static_cast<U>(*k_bias_p);
    const U v_scale = static_cast<U>(*v_scale_p);
    const U v_bias = static_cast<U>(*v_bias_p);

    for (int j = 0; j < elem_per_thread; j++) {
      const uint code = sdpa_unpack_code<BITS>(k_code_base, lane_first_code + j);
      k[j] = static_cast<U>(code) * k_scale + k_bias;
    }

    U score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);
    if (softcapping != 1.) {
      score = precise::tanh(score);
      score = score * softcapping;
    }

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (int j = 0; j < elem_per_thread; j++) {
      const uint code = sdpa_unpack_code<BITS>(v_code_base, lane_first_code + j);
      const U value = static_cast<U>(code) * v_scale + v_bias;
      o[j] = o[j] * factor + exp_score * value;
    }

    k_code_base += code_step;
    v_code_base += code_step;
    k_scale_p += BN;
    k_bias_p += BN;
    v_scale_p += BN;
    v_bias_p += BN;
  }

  // The combine is `sdpa_vector_body`'s, unchanged. Quantization touches the
  // loads and nothing about the reduction order (§2.3.3 #1).
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int BITS>
[[kernel]] void sdpa_vector_quant(
    const device T* queries [[buffer(0)]],
    const device uchar* k_codes [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uchar* v_codes [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const device SdpaQuantParams* pp [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  // Taken as a pointer rather than as `constant &` scalars, so this family is
  // ICB-encodable from the first commit rather than needing §11.3's packed
  // conversion later (`DESIGN.md` §3.7c: `MTLIndirectComputeCommand` has no
  // `setBytes` in any form). §11.3h's whole programme is that debt paid
  // retroactively for seven families; there is no reason to incur it again.
  SdpaQuantParams p = *pp;
  sdpa_vector_quant_body<T, D, BITS>(
      p,
      queries,
      k_codes,
      k_scales,
      k_biases,
      v_codes,
      v_scales,
      v_biases,
      out,
      outputs,
      max_scores,
      sum_exp_scores,
      tid,
      simd_gid,
      simd_lid);
}

#define instantiate_sdpa_vector_quant(name, type, d, bits)              \
  template [[host_name("sdpa_vector_quant_" #name "_" #d "_" #bits)]]   \
  [[kernel]] void sdpa_vector_quant<type, d, bits>(                     \
      const device type* queries [[buffer(0)]],                         \
      const device uchar* k_codes [[buffer(1)]],                        \
      const device type* k_scales [[buffer(2)]],                        \
      const device type* k_biases [[buffer(3)]],                        \
      const device uchar* v_codes [[buffer(4)]],                        \
      const device type* v_scales [[buffer(5)]],                        \
      const device type* v_biases [[buffer(6)]],                        \
      device type* out [[buffer(7)]],                                   \
      const device SdpaQuantParams* pp [[buffer(8)]],                   \
      uint3 tid [[threadgroup_position_in_grid]],                       \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lid [[thread_index_in_simdgroup]]);

// LFM2 decodes at f16 with head_dim 64 (§5.3, §2.3.3b: every `half` site is
// pinned whatever its spelling). f32 is instantiated beside it because the
// parity test compares against the CPU backend at f32, and 128 because it is
// the next head_dim a second checkpoint would bring — §16 #5 records that the
// question reopens there.
instantiate_sdpa_vector_quant(float16_t, half, 64, 8)
instantiate_sdpa_vector_quant(float16_t, half, 64, 4)
instantiate_sdpa_vector_quant(float16_t, half, 128, 8)
instantiate_sdpa_vector_quant(float16_t, half, 128, 4)
instantiate_sdpa_vector_quant(float, float, 64, 8)
instantiate_sdpa_vector_quant(float, float, 64, 4)
instantiate_sdpa_vector_quant(float, float, 128, 8)
instantiate_sdpa_vector_quant(float, float, 128, 4)

// ---- the write side: quantize one token's head vectors into the planes ----

// Quantize `[H, D]` — one token's worth of one of K or V — into a code plane
// plus a (scale, bias) pair per head vector.
//
// # Why this is its own dispatch rather than folded into the append
//
// §6.2b's in-place append is a `copy2d` of exactly one token's bytes at a
// moving offset, and its value is that **the buffer identity stops changing**
// (§11.1a.1). Quantizing is a *different* computation over the same bytes, so
// folding it into `slice_set` would mean a strided, dtype-changing copy where
// there is now a contiguous one — and `slice_set` requires both sides
// contiguous (`tensor_cat.rs:249`). One dispatch per (layer, K/V) per token is
// the honest cost and it is what §12's fusion catalogue would later remove.
//
// # One threadgroup per head vector
//
// The group is the head vector (§16 #6), so a threadgroup owns exactly one
// group and the min/max reduction is one `simd_min`/`simd_max` over D/32
// elements per lane. **The reduction is a fixed simdgroup tree** — §2.3.3 #3,
// which #265 records as the one rule whose cost is structurally zero because
// `simd_min`/`simd_max` *are* the primitive.
//
// Determinism: this runs once per token at write time, and its output is then
// only read (§16 #6). It is not inside the per-token reduction §2.3.3 #1
// governs.
template <typename T, int D, int BITS>
[[kernel]] void quantize_kv(
    const device T* src [[buffer(0)]],
    device uchar* codes [[buffer(1)]],
    device T* scales [[buffer(2)]],
    device T* biases [[buffer(3)]],
    const device SdpaQuantWriteParams* pp [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr float levels = (BITS == 8) ? 255.0f : 15.0f;

  const SdpaQuantWriteParams p = *pp;

  // One threadgroup per (head, token-within-this-append).
  const int head_idx = tid.x;
  const int tok_idx = tid.y;

  const device T* in = src + head_idx * p.src_head_stride + tok_idx * D;
  // The destination token index is the slot's live length plus this token's
  // position in the batch being appended, which is what makes a `seq_len > 1`
  // prefill append and a 1-token decode append the same call (§6.2b).
  const int dst_tok = p.dst_offset + tok_idx;

  device uchar* out_codes =
      codes + head_idx * p.code_head_stride + sdpa_code_bytes<BITS>(dst_tok * D);
  device T* out_scale = scales + head_idx * p.scale_head_stride + dst_tok;
  device T* out_bias = biases + head_idx * p.scale_head_stride + dst_tok;

  thread float x[elem_per_thread];
  for (int j = 0; j < elem_per_thread; j++) {
    x[j] = static_cast<float>(in[simd_lid * elem_per_thread + j]);
  }

  float lo = x[0];
  float hi = x[0];
  for (int j = 1; j < elem_per_thread; j++) {
    lo = min(lo, x[j]);
    hi = max(hi, x[j]);
  }
  lo = simd_min(lo);
  hi = simd_max(hi);

  // An affine (asymmetric) range. K after RoPE and per-head QK-norm is not
  // zero-centred per group, so a symmetric range would spend a code bit on
  // an unused sign; the cost of the affine form is the second f16, and that
  // second array is exactly the term §10.7a counts to get 0.531× rather than
  // 0.5× (§9.5b, and #228 for admission's copy of the same arithmetic).
  //
  // A degenerate group — every element equal, which a zeroed KV slot produces
  // before anything is written — gives `hi == lo`. Scale 0 with bias `lo`
  // then reconstructs the constant exactly, which is the correct answer and
  // avoids a divide by zero. It is not a special case bolted on: it is what
  // the affine form already means when the range is empty.
  const float range = hi - lo;
  const float scale = (range > 0.0f) ? (range / levels) : 0.0f;
  const float inv_scale = (range > 0.0f) ? (levels / range) : 0.0f;

  if (simd_lid == 0) {
    // Stored at T, which is what the read side loads them as. Rounding the
    // scale to f16 BEFORE it is used to encode would be the correct thing if
    // the reader used the rounded value against unrounded codes; it does not
    // — both sides read this same stored pair — so the encode below uses the
    // f32 `inv_scale` and the decode uses the f16 `scale`, and the parity
    // reference must do the same. That asymmetry is deliberate and is
    // asserted by the CPU reference rather than left implicit.
    *out_scale = static_cast<T>(scale);
    *out_bias = static_cast<T>(lo);
  }

  // Re-read the scale as the reader will see it, so the codes are chosen
  // against the value that will actually dequantize them.
  const float enc_scale = static_cast<float>(static_cast<T>(scale));
  const float enc_bias = static_cast<float>(static_cast<T>(lo));
  const float enc_inv = (enc_scale > 0.0f) ? (1.0f / enc_scale) : 0.0f;
  (void)inv_scale;

  for (int j = 0; j < elem_per_thread; j++) {
    const int idx = simd_lid * elem_per_thread + j;
    float t = (x[j] - enc_bias) * enc_inv;
    t = clamp(rint(t), 0.0f, levels);
    const uint code = uint(t);

    if (BITS == 8) {
      out_codes[idx] = uchar(code);
    } else {
      // Two codes share a byte and two *different lanes* may own them, so the
      // write must not be a read-modify-write race. `elem_per_thread` is
      // D/32 = 2 at D=64 and 4 at D=128 — even in both cases — so a lane owns
      // a whole number of bytes and packs them itself. Asserted rather than
      // assumed, because it is the property that makes this branch safe.
      if ((idx & 1) == 0) {
        // The odd sibling is this same lane's next element.
        float t1 = (x[j + 1] - enc_bias) * enc_inv;
        t1 = clamp(rint(t1), 0.0f, levels);
        out_codes[idx >> 1] = uchar(code | (uint(t1) << 4));
      }
    }
  }
}

// Read the code planes back as `T`, for the paths that cannot consume codes.
//
// **The prefill path needs this and it is not a convenience.** A turn boundary
// re-prefills onto the *accumulated* cache (§2.3.8d: turn 2 is the KV-suspect
// test, and §10.9 records that `Cache::clear()` is never called by either
// multi-turn harness), so a `seq_len > 1` pass must attend over the whole
// `kv_len` -- the causal mask is `[seq_len, index_pos + seq_len]` by
// construction. Attending over only the current pass's K/V is a shape error at
// best and a silently truncated context at worst.
//
// A quantized `seq_len > 1` attention kernel is out of scope (§17 item 18 is
// the decode path), so the honest arrangement is: the cache stays quantized,
// and prefill dequantizes it. The decode steps that follow read the codes
// directly through `sdpa_vector_quant`, which is where the mechanism pays.
//
// One threadgroup per (head, token), 32 lanes, mirroring `quantize_kv`.
template <typename T, int D, int BITS>
[[kernel]] void dequantize_kv(
    const device uchar* codes [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device SdpaQuantWriteParams* pp [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  const SdpaQuantWriteParams p = *pp;
  const int head_idx = tid.x;
  const int tok_idx = tid.y;

  // `dst_offset` is 0 here: a dequant reads from the base of the live region.
  // `src_head_stride` is reused as the DESTINATION head stride, in elements,
  // because the direction is reversed -- named in the caller rather than
  // renamed here, since the struct is shared and its layout is pinned.
  const device uchar* in_codes =
      codes + head_idx * p.code_head_stride + sdpa_code_bytes<BITS>(tok_idx * D);
  const T scale = scales[head_idx * p.scale_head_stride + tok_idx];
  const T bias = biases[head_idx * p.scale_head_stride + tok_idx];
  device T* dst = out + head_idx * p.src_head_stride + tok_idx * D;

  for (int j = 0; j < elem_per_thread; j++) {
    const int idx = simd_lid * elem_per_thread + j;
    const uint code = sdpa_unpack_code<BITS>(in_codes, idx);
    dst[idx] = static_cast<T>(static_cast<float>(code) * static_cast<float>(scale) +
                              static_cast<float>(bias));
  }
}

#define instantiate_dequantize_kv(name, type, d, bits)              \
  template [[host_name("dequantize_kv_" #name "_" #d "_" #bits)]]   \
  [[kernel]] void dequantize_kv<type, d, bits>(                     \
      const device uchar* codes [[buffer(0)]],                      \
      const device type* scales [[buffer(1)]],                      \
      const device type* biases [[buffer(2)]],                      \
      device type* out [[buffer(3)]],                               \
      const device SdpaQuantWriteParams* pp [[buffer(4)]],          \
      uint3 tid [[threadgroup_position_in_grid]],                   \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_dequantize_kv(float16_t, half, 64, 8)
instantiate_dequantize_kv(float16_t, half, 64, 4)
instantiate_dequantize_kv(float16_t, half, 128, 8)
instantiate_dequantize_kv(float16_t, half, 128, 4)
instantiate_dequantize_kv(float, float, 64, 8)
instantiate_dequantize_kv(float, float, 64, 4)
instantiate_dequantize_kv(float, float, 128, 8)
instantiate_dequantize_kv(float, float, 128, 4)

#define instantiate_quantize_kv(name, type, d, bits)              \
  template [[host_name("quantize_kv_" #name "_" #d "_" #bits)]]   \
  [[kernel]] void quantize_kv<type, d, bits>(                     \
      const device type* src [[buffer(0)]],                       \
      device uchar* codes [[buffer(1)]],                          \
      device type* scales [[buffer(2)]],                          \
      device type* biases [[buffer(3)]],                          \
      const device SdpaQuantWriteParams* pp [[buffer(4)]],        \
      uint3 tid [[threadgroup_position_in_grid]],                 \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_quantize_kv(float16_t, half, 64, 8)
instantiate_quantize_kv(float16_t, half, 64, 4)
instantiate_quantize_kv(float16_t, half, 128, 8)
instantiate_quantize_kv(float16_t, half, 128, 4)
instantiate_quantize_kv(float, float, 64, 8)
instantiate_quantize_kv(float, float, 64, 4)
instantiate_quantize_kv(float, float, 128, 8)
instantiate_quantize_kv(float, float, 128, 4)

// ============ "mlx/backend/metal/kernels/utils.h"

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);


// ============ "mlx/backend/metal/kernels/steel/attn/loader.h"

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

template <int R, int C>
struct CShape {
  STEEL_CONST int kRows = R;
  STEEL_CONST int kCols = C;
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoaderT(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] =
            op.apply(dst[i * kDstStrRow + j * kDstStrCol]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }

  /* Re-seat this thread's read pointer at a new block base. A paged K/V
   * (`is_paged`) resolves each key block's base through a page table rather
   * than advancing linearly, so the loader is pointed at the block rather than
   * stepped to it; the thread's own offset inside the block is unchanged. */
  METAL_FUNC void seek(const device T* src_) {
    src = src_ + bi * src_ld + bj;
  }
};

// ============ "mlx/backend/metal/kernels/steel/utils/type_traits.h"

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

// ============ "mlx/backend/metal/kernels/steel/utils/integral_constant.h"

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

// ============ "mlx/backend/metal/kernels/steel/gemm/transforms.h"

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x * alpha + (beta * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

// ============ "mlx/backend/metal/kernels/steel/attn/mma.h"

template <typename RInt, typename CInt>
struct Shape2D {
  RInt r;
  CInt c;

  Shape2D(RInt r_, CInt c_) : r(r_), c(c_) {}
};

template <typename Shape, typename Layout>
struct Layout2D {
  Shape shape;
  Layout layout;
};

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread dtype_mat_t<Atype>& A,
      thread dtype_mat_t<Btype>& B,
      thread dtype_mat_t<Ctype>& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC mat_type mat_at(const short i, const short j) {
    mat_type val_mat;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < kElemsPerFrag; ++ii) {
      val_mat.thread_elements()[ii] = frag_at(i, j)[ii];
    }
    return val_mat;
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(
                dst[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load_safe(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m; //(n % 2) ? (M - 1 - m) : m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Threadgroup A strides
  STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

        // Read C
        U c_elems[kelems] = {0};

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          D[offset_d + k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[offset_d + k] =
                  epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

// ============ "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale
  float softcapping; ///< Softcapping value (1.0 = disabled)

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

/// The page-walking K/V address (`is_paged`): K and V are one pool,
/// `[1, H_kv, pool_tokens, D]`, and sequence `b`'s key `t` lives at slot
/// `page_table[b * table_stride + t / page_size] * page_size + t % page_size`.
/// A key block never straddles a page (`page_size % BK == 0`, checked by the
/// host), so the block loaders are re-seated once per block and walk the page
/// linearly from there — the indirection is hoisted out of the inner loop
/// exactly as `flash_decoding.metal` hoists its chunk lookup (§10.3d). The
/// batch stride of K/V is 0 on this arm: a sequence is reached through its
/// table row, never through a stride (§10.3c).
struct AttnPagedParams {
  int page_size; ///< KV tokens per page
  int table_stride; ///< Entries between two sequences' tables
  int n_keys_stride; ///< Entries between two sequences' live key counts (0: one count for the dispatch)
};

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool is_paged [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  // Guard: when y (row max) is -inf, all scores in the row are -inf (entirely masked). Return 0 instead of exp2(-inf - (-inf)) = exp2(NaN).
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -metal::numeric_limits<T>::infinity())
        ? T(0)
        : fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const constant AttnPagedParams* paged_params [[buffer(7), function_constant(is_paged)]],
    const device uint* page_table [[buffer(8), function_constant(is_paged)]],
    const device int* n_keys_per_seq [[buffer(9), function_constant(is_paged)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Seqeunce

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Seqeunce

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  TransformScale<T> ts(static_cast<T>(params->scale * 1.44269504089));

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks apply scale
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::min;
  }

  // The key range this sequence attends: the dispatch's on the contiguous
  // arm; on the paged arm the sequence's OWN live count (`n_keys_per_seq`,
  // stride 0 = one count for the batch), so rows of different lengths run in
  // one dispatch, each over its own blocks with its own causal offset. The
  // host compiles `align_K` off whenever the counts differ.
  int kL = params->kL;
  int NK = params->NK;
  int NK_aligned = params->NK_aligned;
  int kL_rem = params->kL_rem;
  int qL_off = params->qL_off;
  if (is_paged) {
    kL = n_keys_per_seq[tidl.z * paged_params->n_keys_stride];
    NK = (kL + BK - 1) / BK;
    NK_aligned = kL / BK;
    kL_rem = kL - NK_aligned * BK;
    qL_off = kL - params->qL;
  }

  // The causal window is [kb_min_causal, kb_lim): blocks below it are fully
  // visible, blocks at or above it are masked element-wise. Both are clamped
  // to the key range: a partial query block's `q_max` overshoots `kL`.
  int kb_lim = NK;
  int kb_min_causal = NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(NK, kb_lim);

    int q_min = tid.x * BQ + qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    // Paged K/V: seat both loaders at this block's page slot. The partial
    // last block (`kL_rem` rows) still sits inside one page, so the masked
    // `load_safe` below reads exactly the live rows from that base.
    if (is_paged) {
      const int t0 = kb * BK;
      const int page_size = paged_params->page_size;
      const uint page = page_table[tidl.z * paged_params->table_stride + t0 / page_size];
      const int64_t slot = int64_t(page) * page_size + (t0 % page_size);
      loader_k.seek(K + slot * params->K_strides[2]);
      loader_v.seek(V + slot * params->V_strides[2]);
    }

    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (NK_aligned)) {
      loader_k.load_safe(short2(BD, kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Mask out length sequence
    if (!align_K && kb == (NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += 1.44269504089 * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!align_K && kb == (NK_aligned)) {
      loader_v.load_safe(short2(BD, kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    // Guard: when max_score == -inf (no valid K seen yet), the previous accumulation is all zeros so the correct rescaling factor is 0.
    // Without this, -inf - (-inf) = NaN which poisons the output.
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (max_score[i] == -metal::numeric_limits<AccumType>::infinity())
          ? AccumType(0)
          : fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration: the contiguous arm walks linearly; the paged
    // arm is re-seated at the top of the loop.
    if (!is_paged) {
      loader_k.next();
      loader_v.next();
    }
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

// clang-format off

// SDPA full instantiations

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  96, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  72, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  32, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);
instantiate_attn_mask_helper(float32, float);

// BD=512: reduced tiles (BQ=8, BK=8) to fit 32KB threadgroup memory.
// Only f16/bf16 — float32 exceeds the limit.
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)

// ---------------------------------------------------------------------------
// The layout check (issue #103).
//
// A field at the wrong offset does not crash. The kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `sdpa_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
//
// `SdpaVectorParams` is 32 rather than the 28 its fields sum to: `size_t`
// forces 8-byte alignment, `k_stride` cannot start until 8, and the two
// trailing `float`s leave four bytes of tail padding. Neither number is a field
// width; both are the padding rule, which is what this kernel exists to ship
// across the boundary.
static_assert(sizeof(SdpaVectorParams) == 32, "SdpaVectorParams layout");
static_assert(alignof(SdpaVectorParams) == 8, "SdpaVectorParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define sdpa_offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void sdpa_params_layout(
    device uint* out,
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) { return; }
  SdpaVectorParams probe_SdpaVectorParams;

  out[0] = sizeof(SdpaVectorParams);
  out[1] = sdpa_offsetof_rt(SdpaVectorParams, gqa_factor);
  out[2] = sdpa_offsetof_rt(SdpaVectorParams, N);
  out[3] = sdpa_offsetof_rt(SdpaVectorParams, k_stride);
  out[4] = sdpa_offsetof_rt(SdpaVectorParams, v_stride);
  out[5] = sdpa_offsetof_rt(SdpaVectorParams, scale);
  out[6] = sdpa_offsetof_rt(SdpaVectorParams, softcapping);
}

// The same check for issue #218's two structs.
//
// A **second** kernel rather than more slots on `sdpa_params_layout`, because
// that kernel's slot list is what `sdpa_params_layout_check_detects_a_moved_
// field` indexes its assertion against; appending to it would change what an
// existing mutation test reads.
//
// `SdpaQuantParams` is 48: two `int` (8), then four `size_t` from 8 to 40,
// then two `float` to 48 — no tail padding, because 48 is already a multiple
// of the 8-byte alignment. `SdpaQuantWriteParams` is 32: three `size_t` to 24,
// two `int` to 32.
static_assert(sizeof(SdpaQuantParams) == 48, "SdpaQuantParams layout");
static_assert(alignof(SdpaQuantParams) == 8, "SdpaQuantParams alignment");
static_assert(sizeof(SdpaQuantWriteParams) == 32, "SdpaQuantWriteParams layout");
static_assert(alignof(SdpaQuantWriteParams) == 8, "SdpaQuantWriteParams alignment");

[[kernel]] void sdpa_quant_params_layout(
    device uint* out,
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) { return; }
  SdpaQuantParams probe_SdpaQuantParams;
  SdpaQuantWriteParams probe_SdpaQuantWriteParams;

  out[0] = sizeof(SdpaQuantParams);
  out[1] = sdpa_offsetof_rt(SdpaQuantParams, gqa_factor);
  out[2] = sdpa_offsetof_rt(SdpaQuantParams, N);
  out[3] = sdpa_offsetof_rt(SdpaQuantParams, k_code_stride);
  out[4] = sdpa_offsetof_rt(SdpaQuantParams, v_code_stride);
  out[5] = sdpa_offsetof_rt(SdpaQuantParams, k_scale_stride);
  out[6] = sdpa_offsetof_rt(SdpaQuantParams, v_scale_stride);
  out[7] = sdpa_offsetof_rt(SdpaQuantParams, scale);
  out[8] = sdpa_offsetof_rt(SdpaQuantParams, softcapping);

  out[9] = sizeof(SdpaQuantWriteParams);
  out[10] = sdpa_offsetof_rt(SdpaQuantWriteParams, src_head_stride);
  out[11] = sdpa_offsetof_rt(SdpaQuantWriteParams, code_head_stride);
  out[12] = sdpa_offsetof_rt(SdpaQuantWriteParams, scale_head_stride);
  out[13] = sdpa_offsetof_rt(SdpaQuantWriteParams, dst_offset);
}

// SDPA vector instantiations
//
// A second `[[host_name]]` on the same instantiation does not compile -- an
// explicit instantiation is unique per template-argument list however many
// names are attached (`DESIGN.md` §11.3g). So the packed sibling is a distinct
// function template, which is what `sdpa_vector_packed` above is.
#define instantiate_sdpa_vector(type, head_dim)                              \
  template [[host_name("sdpa_vector_" #type "_" #head_dim)]]                 \
  [[kernel]] void sdpa_vector<type, head_dim>(                               \
      const device type* queries [[buffer(0)]],                              \
      const device type* keys [[buffer(1)]],                                 \
      const device type* values [[buffer(2)]],                               \
      device type* out [[buffer(3)]],                                        \
      const constant int& gqa_factor,                                        \
      const constant int& N,                                                 \
      const constant size_t& k_stride,                                       \
      const constant size_t& v_stride,                                       \
      const constant float& scale,                                           \
      const constant float& softcapping,                                     \
      const device bool* mask [[function_constant(sdpa_vector_has_mask)]],              \
      const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],   \
      const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],  \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \
  template [[host_name("sdpa_vector_" #type "_" #head_dim "_packed")]]       \
  [[kernel]] void sdpa_vector_packed<type, head_dim>(                        \
      const device SdpaVectorParams* pp [[buffer(0)]],                       \
      const device type* queries [[buffer(1)]],                              \
      const device type* keys [[buffer(2)]],                                 \
      const device type* values [[buffer(3)]],                               \
      device type* out [[buffer(4)]],                                        \
      const device bool* mask [[function_constant(sdpa_vector_has_mask)]],              \
      const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],   \
      const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],  \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \
  template [[host_name("sdpa_vector_2pass_1_" #type "_" #head_dim)]]         \
  [[kernel]] void sdpa_vector_2pass_1<type, head_dim>(                       \
      const device type* queries [[buffer(0)]],                              \
      const device type* keys [[buffer(1)]],                                 \
      const device type* values [[buffer(2)]],                               \
      device float* out [[buffer(3)]],                                       \
      device float* sums [[buffer(4)]],                                      \
      device float* maxs [[buffer(5)]],                                      \
      const constant int& gqa_factor,                                        \
      const constant int& N,                                                 \
      const constant size_t& k_stride,                                       \
      const constant size_t& v_stride,                                       \
      const constant float& scale,                                           \
      const constant float& softcapping,                                     \
      const device bool* mask [[function_constant(sdpa_vector_has_mask)]],              \
      const constant int& mask_seq_stride [[function_constant(sdpa_vector_has_mask)]],   \
      const constant int& mask_head_stride [[function_constant(sdpa_vector_has_mask)]],  \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \
  template [[host_name("sdpa_vector_2pass_2_" #type "_" #head_dim)]]         \
  [[kernel]] void sdpa_vector_2pass_2<type, head_dim>(                       \
      const device float* partials [[buffer(0)]],                            \
      const device float* sums [[buffer(1)]],                                \
      const device float* maxs [[buffer(2)]],                                \
      device type* out [[buffer(3)]],                                           \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);                          \

#define instantiate_sdpa_vector_heads(type) \
  instantiate_sdpa_vector(type, 32)         \
  instantiate_sdpa_vector(type, 64)         \
  instantiate_sdpa_vector(type, 72)         \
  instantiate_sdpa_vector(type, 80)         \
  instantiate_sdpa_vector(type, 96)         \
  instantiate_sdpa_vector(type, 128)         \
  instantiate_sdpa_vector(type, 256)         \
  instantiate_sdpa_vector(type, 512)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on