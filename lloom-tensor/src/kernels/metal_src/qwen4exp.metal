// Kernels the Qwen4-exp blocks dispatch (`lloom::models::qwen4exp`): the
// pieces of the decode and prefill paths that were composed from tens of
// elementwise ops, each written as ONE launch. Every kernel here has a
// scalar reference in the model crate it is tested against.

// ---- HyperConnection brackets (`hc.rs`) ---------------------------------------
//
// The carry is `n` streams of `d` (`[rows, n·d]`). The composed path spent
// a launch per elementwise op on it; these are the three that remain
// between the two low-rank projections (which are `gemv_mixed` launches).

// `out[r, s·d + i] = x[r, s·d + i] · rsqrt(mean_i(x[r, s·d + ·]²) + eps) · gamma[s·d + i]`
// — the grouped RMS norm: per stream over `d`, the γ over the whole `n·d`
// (`hc::grouped_rms_norm`). One threadgroup per (row, stream); the sum of
// squares by simdgroup shuffles then a small threadgroup tree, in f32.
kernel void hc_norm_f32(
    const device float* x [[buffer(0)]],
    const device float* gamma [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const uint row = group / n;
  const uint s = group % n;
  const device float* xs = x + (size_t)row * n * d + (size_t)s * d;
  device float* os = out + (size_t)row * n * d + (size_t)s * d;
  const device float* gs = gamma + (size_t)s * d;
  float acc = 0.0f;
  for (uint i = tid; i < d; i += width) {
    const float v = xs[i];
    acc += v * v;
  }
  acc = simd_sum(acc);
  if (simd_lane == 0) {
    partial[simd_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint k = 0; k < n_simd; k++) {
    total += partial[k];
  }
  const float scale = metal::rsqrt(total / (float)d + eps);
  for (uint i = tid; i < d; i += width) {
    os[i] = xs[i] * scale * gs[i];
  }
}

// `out[r, i] = (1/n) · Σ_s xn[r, s·d + i] · gate[r, s·d + i]` — the gated mean
// over the streams (the prologue's collapse). One thread per output element.
kernel void hc_collapse_f32(
    const device float* xn [[buffer(0)]],
    const device float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= rows * d) {
    return;
  }
  const uint r = idx / d;
  const uint i = idx % d;
  const size_t base = (size_t)r * n * d + i;
  float acc = 0.0f;
  for (uint s = 0; s < n; s++) {
    acc += xn[base + (size_t)s * d] * gate[base + (size_t)s * d];
  }
  out[idx] = acc * (1.0f / (float)n);
}

// `out[r, s·d + i] = h[r, s·d + i] + bo[r, i] · 2·sigmoid(inject[r, s] / n)`
// — the epilogue's scatter of the block output across the streams, weighted
// by the prediction, centred on 1. One thread per output element.
kernel void hc_combine_f32(
    const device float* h [[buffer(0)]],
    const device float* bo [[buffer(1)]],
    const device float* inject [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  const size_t total = (size_t)rows * n * d;
  if (idx >= total) {
    return;
  }
  const uint nd = n * d;
  const uint r = idx / nd;
  const uint rem = idx % nd;
  const uint s = rem / d;
  const uint i = rem % d;
  const float z = inject[(size_t)r * n + s] * (1.0f / (float)n);
  const float w = 2.0f / (1.0f + metal::exp(-z));
  out[idx] = h[idx] + bo[(size_t)r * d + i] * w;
}

// ---- Mixture of experts (`moe.rs`) --------------------------------------------

// The router's selection: softmax over the `n_experts` logits of each row,
// the top `k` by probability (ties to the lowest index), the selected
// probabilities renormalised to sum 1 — `ids [rows, k]` u32 and `w [rows, k]`
// f32. One threadgroup per row; `k` rounds of an argmax reduction over the
// experts. The composed path: softmax, a full descending arg-sort, a narrow,
// a gather, a sum and a divide — six launches.
// ---------------------------------------------------------------------------
// The HC bracket's prologue as two launches (#749 P29 (3)): the grouped norm,
// the low-rank down projection with its SiLU, and the inject projection in
// ONE kernel (`hc_mix_a`), then the up projection with its sigmoid and the
// gated collapse over the streams in ONE (`hc_mix_b`). Composed, the same
// bracket was five to six launches a bracket, ~11 a layer, 528 a token.
//
// `hc_mix_a`: threadgroup (x, row) — every threadgroup recomputes the row's
// `n` stream scales (a 2 560-wide sum of squares each; the carry is 40 KB and
// stays in cache), stages the normed carry `h · scale_s · γ` a chunk at a
// time in threadgroup memory (threadgroup x == 0 also writes it out as `xn`
// for `hc_mix_b`), and each simdgroup takes HC_MIX_OUT output rows of
// `[w_down | w_inject]`: lane-strided `half4` dots (coalesced across the
// simdgroup) accumulated over the chunks, `simd_sum`, then SiLU(scale · dot)
// for a down row and the raw dot for an inject row. With no inject weight the
// inject slots are written as zeros (the composed path's `Tensor::zeros`).
// The first cut (one output a simdgroup, the normed value recomputed per
// dot) read the carry three times per weight byte and ran at 254 GB/s.
//
// Output layout (flat): [rows · n·d] xn | [rows · lr] lo | [rows · n] inject —
// three contiguous blocks, so each is a contiguous narrow of the one buffer.
//
// The dot's order — lane-strided partials then `simd_sum` — differs from the
// mixed gemv's tiles, so the bracket's rounding moves at the last bits (the
// logits digest moves, the tokens hold); the parity test bounds it.
#define HC_MAX_N 8
#define HC_MIX_SG 8
#define HC_MIX_MAX_LR 1024
#define HC_MIX_OUT 1
#define HC_MIX_CHUNK 2048

inline float hc_silu(float v) { return v / (1.0f + metal::exp(-v)); }
inline float hc_sigmoid(float v) { return 1.0f / (1.0f + metal::exp(-v)); }

kernel void hc_mix_a_f32(
    const device float* h [[buffer(0)]],
    const device float* gamma [[buffer(1)]],
    const device half* w_down [[buffer(2)]],
    const device half* w_inject [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& lr [[buffer(7)]],
    constant uint& n_inj [[buffer(8)]],
    constant uint& rows [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  threadgroup float scales[HC_MAX_N];
  // The normed carry, staged a chunk at a time (8 KB), read once per
  // threadgroup instead of once per output: the weights are then the only
  // stream the dots pull from device memory.
  threadgroup float4 xn_s[HC_MIX_CHUNK / 4];
  const uint tid = tid2.x;
  const uint width = tptg.x;
  const uint row = tg.y;
  if (row >= rows) {
    return;
  }
  const uint nd = n * d;
  const device float* hr = h + (size_t)row * nd;
  // The stream scales: one reduction per stream, every thread a strided slice.
  const uint n_simd = (width + 31) / 32;
  for (uint s = 0; s < n; s++) {
    float acc = 0.0f;
    const device float* hs = hr + (size_t)s * d;
    for (uint i = tid; i < d; i += width) {
      const float v = hs[i];
      acc += v * v;
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) {
      partial[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float total = 0.0f;
      for (uint k = 0; k < n_simd; k++) {
        total += partial[k];
      }
      scales[s] = metal::rsqrt(total / (float)d + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  device float* xn_out = out;
  device float* lo_out = out + (size_t)rows * nd;
  device float* inj_out = lo_out + (size_t)rows * lr;
  // This simdgroup's HC_MIX_OUT output rows of [w_down | w_inject].
  const uint j0 = (tg.x * HC_MIX_SG + simd_id) * HC_MIX_OUT;
  const device half4* wr4[HC_MIX_OUT];
  bool live[HC_MIX_OUT];
  float acc[HC_MIX_OUT];
  for (uint o = 0; o < HC_MIX_OUT; o++) {
    const uint j = j0 + o;
    acc[o] = 0.0f;
    if (j < lr) {
      live[o] = true;
      wr4[o] = (const device half4*)(w_down + (size_t)j * nd);
    } else if (j < lr + n_inj) {
      live[o] = true;
      wr4[o] = (const device half4*)(w_inject + (size_t)(j - lr) * nd);
    } else {
      live[o] = false;
      wr4[o] = (const device half4*)w_down;
    }
  }
  const device float4* hr4 = (const device float4*)hr;
  const device float4* g4 = (const device float4*)gamma;
  const uint nd4 = nd / 4;
  const uint d4 = d / 4;
  device float4* xn_out4 = (device float4*)(xn_out + (size_t)row * nd);
  for (uint c0 = 0; c0 < nd4; c0 += HC_MIX_CHUNK / 4) {
    const uint c_len = min((uint)(HC_MIX_CHUNK / 4), nd4 - c0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i4 = tid; i4 < c_len; i4 += width) {
      const uint g = c0 + i4;
      const float4 xn = hr4[g] * scales[g / d4] * g4[g];
      xn_s[i4] = xn;
      if (tg.x == 0) {
        xn_out4[g] = xn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < HC_MIX_SG) {
      for (uint i4 = simd_lane; i4 < c_len; i4 += 32) {
        const float4 xn = xn_s[i4];
        const uint g = c0 + i4;
        for (uint o = 0; o < HC_MIX_OUT; o++) {
          if (live[o]) {
            acc[o] += dot(float4(wr4[o][g]), xn);
          }
        }
      }
    }
  }
  if (simd_id >= HC_MIX_SG) {
    return;
  }
  for (uint o = 0; o < HC_MIX_OUT; o++) {
    const float total = simd_sum(acc[o]);
    const uint j = j0 + o;
    if (simd_lane == 0 && j < lr + n) {
      if (j < lr) {
        lo_out[(size_t)row * lr + j] = hc_silu(scale * total);
      } else {
        // An inject row, or — with no inject weight — the composed path's zeros.
        inj_out[(size_t)row * n + (j - lr)] = live[o] ? total : 0.0f;
      }
    }
  }
}

// `hc_mix_b`: threadgroup (x, row); `lo` staged in threadgroup memory; each
// simdgroup takes one output column i of the collapsed carry: for every stream
// s the gate σ(w_up[s·d+i] · lo) (a lane-strided dot over lr, `simd_sum`),
// then Σ_s xn[s·d+i] · gate_s scaled by 1/n — `hc_collapse`'s arithmetic with
// the gate never stored.
kernel void hc_mix_b_f32(
    const device float* packed [[buffer(0)]],
    const device half* w_up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& lr [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant float& inv_n [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float lo_s[HC_MIX_MAX_LR];
  const uint tid = tid2.x;
  const uint width = tptg.x;
  const uint row = tg.y;
  if (row >= rows) {
    return;
  }
  const uint nd = n * d;
  const device float* xn = packed + (size_t)row * nd;
  const device float* lo = packed + (size_t)rows * nd + (size_t)row * lr;
  for (uint k = tid; k < lr; k += width) {
    lo_s[k] = lo[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint i = tg.x * HC_MIX_SG + simd_id;
  if (simd_id >= HC_MIX_SG || i >= d) {
    return;
  }
  float acc = 0.0f;
  const uint lr2 = lr / 2;
  for (uint s = 0; s < n; s++) {
    // lr is even: half2 loads, lane l at [2l, 2l+2) of every 64-wide chunk.
    const device half2* wr2 = (const device half2*)(w_up + (size_t)(s * d + i) * lr);
    float dotv = 0.0f;
    for (uint k2 = simd_lane; k2 < lr2; k2 += 32) {
      const float2 w = float2(wr2[k2]);
      dotv += w.x * lo_s[2 * k2] + w.y * lo_s[2 * k2 + 1];
    }
    dotv = simd_sum(dotv);
    acc += xn[(size_t)s * d + i] * hc_sigmoid(dotv);
  }
  if (simd_lane == 0) {
    out[(size_t)row * d + i] = acc * inv_n;
  }
}

kernel void moe_topk_f32(
    const device float* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* w [[buffer(2)]],
    constant uint& n_experts [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red_f[32];
  threadgroup uint red_i[32];
  threadgroup float probs[1024];
  threadgroup float picked_sum;
  const device float* lg = logits + (size_t)row * n_experts;
  const uint n_simd = (width + 31) / 32;

  // max
  float m = -INFINITY;
  for (uint e = tid; e < n_experts; e += width) {
    m = metal::max(m, lg[e]);
  }
  m = simd_max(m);
  if (simd_lane == 0) red_f[simd_id] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red_f[0];
  for (uint s = 1; s < n_simd; s++) m = metal::max(m, red_f[s]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // exp and sum
  float sum = 0.0f;
  for (uint e = tid; e < n_experts; e += width) {
    const float p = metal::exp(lg[e] - m);
    probs[e] = p;
    sum += p;
  }
  sum = simd_sum(sum);
  if (simd_lane == 0) red_f[simd_id] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = 0.0f;
  for (uint s = 0; s < n_simd; s++) sum += red_f[s];
  const float inv = 1.0f / sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < n_experts; e += width) {
    probs[e] *= inv;
  }
  if (tid == 0) picked_sum = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // k rounds of argmax over the remaining experts (a picked one is set to -1).
  for (uint round = 0; round < k; round++) {
    float best = -1.0f;
    uint best_i = 0xffffffffu;
    for (uint e = tid; e < n_experts; e += width) {
      const float p = probs[e];
      if (p > best || (p == best && e < best_i)) {
        best = p;
        best_i = e;
      }
    }
    // simdgroup argmax: value first, lowest index on ties
    for (uint off = 16; off > 0; off >>= 1) {
      const float ob = simd_shuffle_down(best, off);
      const uint oi = simd_shuffle_down(best_i, off);
      if (ob > best || (ob == best && oi < best_i)) {
        best = ob;
        best_i = oi;
      }
    }
    if (simd_lane == 0) {
      red_f[simd_id] = best;
      red_i[simd_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float b = red_f[0];
      uint bi = red_i[0];
      for (uint s = 1; s < n_simd; s++) {
        if (red_f[s] > b || (red_f[s] == b && red_i[s] < bi)) {
          b = red_f[s];
          bi = red_i[s];
        }
      }
      ids[(size_t)row * k + round] = bi;
      w[(size_t)row * k + round] = b;
      picked_sum += b;
      probs[bi] = -1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // renormalise the picked weights
  if (tid < k) {
    w[(size_t)row * k + tid] = w[(size_t)row * k + tid] / picked_sum;
  }
}

// One element of `silu(g) · u` at `T`, with `utils.metal`'s `silu<T>`.
template <typename T>
METAL_FUNC T silu_times(T g, T u) {
  return static_cast<T>(silu(g) * u);
}

// At half the product must round the SiLU `silu_f16` stored: left to
// reassociate, fast math folds the SiLU's division into the multiply and
// lands one ulp away on about one element in six.
template <>
METAL_FUNC half silu_times(half g, half u) {
  #pragma clang fp reassociate(off)
  return static_cast<half>(silu(g) * u);
}

// `out = silu(g) · u`, elementwise (the routed experts' SwiGLU and the shared
// expert's): two launches into one, at `T`, so the f16 form is `silu_f16`
// then `bmul_f16` bit for bit.
template <typename T>
kernel void silu_mul(
    const device T* g [[buffer(0)]],
    const device T* u [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n) return;
  out[idx] = silu_times(g[idx], u[idx]);
}

template [[host_name("silu_mul_f32")]] [[kernel]] decltype(silu_mul<float>) silu_mul<float>;
template [[host_name("silu_mul_f16")]] [[kernel]] decltype(silu_mul<half>) silu_mul<half>;

// `out[r, i] = Σ_k w[r, k] · y[r, k, i] + shared[r, i] · sigmoid(gate[r])` —
// the routed combine over the `k` selected experts plus the gated shared
// expert (the layer's MoE output), in f32. `Y` is the routed output's store
// type: an f16 `y` widens exactly on load, so the sum is the f32 form's over
// `y` cast to f32. One thread per output element; the composed path was a
// broadcast multiply, a sum, a sigmoid, a multiply and an add.
constant bool moe_masked [[function_constant(778)]];
template <typename Y>
kernel void moe_combine(
    const device Y* y [[buffer(0)]],
    const device float* w [[buffer(1)]],
    const device float* shared [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    // The mask arm (`moe_masked`, function constant 778): a flag per plane
    // `[rows, k]` — the device plan's record, 1 where the plane's expert
    // was not resident and slot 0 was read — whose plane contributes
    // nothing: the pass without that expert, defined and finite. With the
    // constant false the argument is absent and the loop is the plain one.
    const device uint* flags [[buffer(8), function_constant(moe_masked)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= rows * d) return;
  const uint r = idx / d;
  const uint i = idx % d;
  float acc = 0.0f;
  for (uint e = 0; e < k; e++) {
    if (moe_masked && flags[(size_t)r * k + e] != 0u) continue;
    acc += w[(size_t)r * k + e] * static_cast<float>(y[((size_t)r * k + e) * d + i]);
  }
  const float gz = gate[r];
  const float sg = 1.0f / (1.0f + metal::exp(-gz));
  out[idx] = acc + shared[idx] * sg;
}

template [[host_name("moe_combine_f32")]] [[kernel]] decltype(moe_combine<float>) moe_combine<float>;
template [[host_name("moe_combine_f16")]] [[kernel]] decltype(moe_combine<half>) moe_combine<half>;

// ---- QSA (full attention) decode (`qsa.rs`) ------------------------------------

// The decode prologue: from `qg [nh, 2·hd]` (q | gate interleaved per head)
// and `k [nkv, hd]`, the per-head RMS norm (γ_q / γ_k over `hd`) and the
// partial NEOX rope at `pos` on the first `n_rot` dims — the angle computed
// here in precise f32 as the host did (`rope_cs_at`) — into
// `out [2·nh + nkv, hd]`: rows `0..nh` q (normed, roped), `nh..2nh` the gate
// (as is), `2nh..` k (normed, roped). One threadgroup per head, `hd` threads.
kernel void qsa_prep_f32(
    const device float* qg [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* q_gamma [[buffer(2)]],
    const device float* k_gamma [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& hd [[buffer(5)]],
    constant uint& n_rot [[buffer(6)]],
    constant uint& nh [[buffer(7)]],
    constant uint& nkv [[buffer(8)]],
    constant uint& pos [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint g [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  threadgroup float tmp[1024];
  const bool is_q = g < nh;
  float x = 0.0f;
  const device float* gamma = is_q ? q_gamma : k_gamma;
  uint out_row;
  if (is_q) {
    out_row = g;
    if (i < hd) {
      x = qg[(size_t)g * 2 * hd + i];
      out[(size_t)(nh + g) * hd + i] = qg[(size_t)g * 2 * hd + hd + i]; // the gate
    }
  } else {
    const uint kh = g - nh;
    out_row = 2 * nh + kh;
    if (i < hd) x = k[(size_t)kh * hd + i];
  }
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  const float scale = 1.0f / metal::sqrt(total / (float)hd + eps);
  if (i < hd) tmp[i] = x * scale * gamma[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= hd) return;
  const uint rot_half = n_rot / 2;
  device float* o = out + (size_t)out_row * hd;
  if (i < rot_half) {
    const float inv = metal::precise::exp(-((2.0f * (float)i) / (float)n_rot) * metal::precise::log(theta));
    const float f = inv * (float)pos;
    const float c = metal::precise::cos(f);
    const float s = metal::precise::sin(f);
    const float lo = tmp[i];
    const float hi = tmp[i + rot_half];
    o[i] = lo * c - hi * s;
    o[i + rot_half] = lo * s + hi * c;
  } else if (i >= n_rot) {
    o[i] = tmp[i];
  }
}

// `qsa_prep_f32` over B ROWS in one launch (the pooled rows arm): row `r`'s
// `qg [nh, 2·hd]` / `k [nkv, hd]` at its own position `pos[r]` → three
// regions of `out`, each row-major: `q [B, nh, hd] | gate [B, nh, hd] | k
// [B, nkv, hd]` (the attention's queries, the gate, the scatter's tokens —
// no re-layout after), row for row the single kernel's arithmetic. One
// threadgroup per (head, row), `hd` threads.
kernel void qsa_prep_batch_f32(
    const device float* qg [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device uint* pos [[buffer(2)]],
    const device float* q_gamma [[buffer(3)]],
    const device float* k_gamma [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& hd [[buffer(6)]],
    constant uint& n_rot [[buffer(7)]],
    constant uint& nh [[buffer(8)]],
    constant uint& nkv [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant uint& n_rows [[buffer(12)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = tid.x;
  const uint width = tptg.x;
  const uint g = gid.x;
  const uint r = gid.y;
  threadgroup float partial[32];
  threadgroup float tmp[1024];
  const bool is_q = g < nh;
  const device float* qg_r = qg + (size_t)r * nh * 2 * hd;
  const device float* k_r = k + (size_t)r * nkv * hd;
  const size_t gate_base = (size_t)n_rows * nh * hd;
  const size_t k_base = 2 * gate_base;
  float x = 0.0f;
  const device float* gamma = is_q ? q_gamma : k_gamma;
  device float* o;
  if (is_q) {
    o = out + ((size_t)r * nh + g) * hd;
    if (i < hd) {
      x = qg_r[(size_t)g * 2 * hd + i];
      out[gate_base + ((size_t)r * nh + g) * hd + i] = qg_r[(size_t)g * 2 * hd + hd + i]; // the gate
    }
  } else {
    const uint kh = g - nh;
    o = out + k_base + ((size_t)r * nkv + kh) * hd;
    if (i < hd) x = k_r[(size_t)kh * hd + i];
  }
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  const float scale = 1.0f / metal::sqrt(total / (float)hd + eps);
  if (i < hd) tmp[i] = x * scale * gamma[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= hd) return;
  const uint rot_half = n_rot / 2;
  if (i < rot_half) {
    const float inv = metal::precise::exp(-((2.0f * (float)i) / (float)n_rot) * metal::precise::log(theta));
    const float f = inv * (float)pos[r];
    const float c = metal::precise::cos(f);
    const float s = metal::precise::sin(f);
    const float lo = tmp[i];
    const float hi = tmp[i + rot_half];
    o[i] = lo * c - hi * s;
    o[i + rot_half] = lo * s + hi * c;
  } else if (i >= n_rot) {
    o[i] = tmp[i];
  }
}

// `out = sigmoid(a) · b`, elementwise (the attention's output gate).
kernel void sigmoid_mul_f32(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n) return;
  out[idx] = b[idx] / (1.0f + metal::exp(-a[idx]));
}

// ---- QSA prefill (`qsa.rs::prefill`) ----------------------------------------------

// Over a chunk of `seq` positions from `pos0`: `qg [seq, nh, 2·hd]`, `k [seq,
// nkv, hd]`, `v [seq, nkv, hd]` → flat: q_h [nh, seq, hd] (normed, roped,
// head-first for the attention kernel) | gate [seq, nh·hd] | k_new [nkv, seq,
// hd] (normed, roped, head-first for the cache) | v_new [nkv, seq, hd]. One
// threadgroup per (head, position), `hd` threads.
kernel void qsa_prep_rows_f32(
    const device float* qg [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    const device float* q_gamma [[buffer(3)]],
    const device float* k_gamma [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& hd [[buffer(6)]],
    constant uint& n_rot [[buffer(7)]],
    constant uint& nh [[buffer(8)]],
    constant uint& nkv [[buffer(9)]],
    constant uint& seq [[buffer(10)]],
    constant uint& pos0 [[buffer(11)]],
    constant float& theta [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = tid.x;
  const uint width = tptg.x;
  threadgroup float partial[32];
  threadgroup float tmp[1024];
  const uint g = gid.x;
  const uint p = gid.y;
  const bool is_q = g < nh;
  const size_t q_len = (size_t)nh * seq * hd;
  const size_t kv_len = (size_t)nkv * seq * hd;
  float x = 0.0f;
  const device float* gamma = is_q ? q_gamma : k_gamma;
  device float* o;
  if (is_q) {
    o = out + ((size_t)g * seq + p) * hd;
    if (i < hd) {
      const size_t src = ((size_t)p * nh + g) * 2 * hd;
      x = qg[src + i];
      out[q_len + ((size_t)p * nh + g) * hd + i] = qg[src + hd + i]; // the gate
    }
  } else {
    const uint kh = g - nh;
    o = out + 2 * q_len + ((size_t)kh * seq + p) * hd;
    if (i < hd) {
      const size_t src = ((size_t)p * nkv + kh) * hd + i;
      x = k[src];
      out[2 * q_len + kv_len + ((size_t)kh * seq + p) * hd + i] = v[src];
    }
  }
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  const float scale = 1.0f / metal::sqrt(total / (float)hd + eps);
  if (i < hd) tmp[i] = x * scale * gamma[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= hd) return;
  const uint rot_half = n_rot / 2;
  const float pos = (float)(pos0 + p);
  if (i < rot_half) {
    const float inv = metal::precise::exp(-((2.0f * (float)i) / (float)n_rot) * metal::precise::log(theta));
    const float f = inv * pos;
    const float c = metal::precise::cos(f);
    const float s = metal::precise::sin(f);
    const float lo = tmp[i];
    const float hi = tmp[i + rot_half];
    o[i] = lo * c - hi * s;
    o[i + rot_half] = lo * s + hi * c;
  } else if (i >= n_rot) {
    o[i] = tmp[i];
  }
}

// ---- The streamed expert tier's device-side plan (`nn::expert_stream`) ---------

// `ids [n]` (expert ids) through `table [n_experts]` (the slot holding each
// expert, `0xFFFFFFFF` = not resident) → flat `[3·n]`: the slot ids the routed
// kernel reads (a miss reads slot 0 — a valid slot, the token is replayed
// after the host loads it) | the ids themselves | a miss flag per id.
kernel void moe_slot_map_u32(
    const device uint* ids [[buffer(0)]],
    const device uint* table [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& n_experts [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  const uint e = ids[i];
  const uint s = e < n_experts ? table[e] : 0xFFFFFFFFu;
  const bool miss = s == 0xFFFFFFFFu;
  out[i] = miss ? 0u : s;
  out[n + i] = e;
  out[2 * n + i] = miss ? 1u : 0u;
}
