#include <metal_stdlib>

using namespace metal;

// Kernels the Qwen4-exp blocks dispatch (`lloom::models::qwen4exp`): the
// pieces of the decode and prefill paths that were composed from tens of
// elementwise ops, each written as ONE launch. Every kernel here has a
// scalar reference in the model crate it is tested against.

// `(I + A)⁻¹` for a strictly-lower-triangular `A [batch, n, n]` (row-major),
// by forward substitution — `T[i, :] = e_i − Σ_{m<i} A[i, m] · T[m, :]`, each
// row from the rows before it (the numerically stable form; the nilpotent
// product overflows on real keys, see `deltanet::unit_lower_inverse`).
//
// One threadgroup per matrix; thread `j` owns column `j` (and every column
// `j + k·width`), rows run in order with a device-memory barrier between
// them, so row `i` reads the rows `< i` the group has already stored. `n`
// is the chunk (64): 64 rows × 64 threads, ~2k FMAs a thread. The composed
// path this replaces was an `n`-iteration host loop of `narrow`/`cat`/
// `matmul`, ~2k `copy2d` launches per matrix batch (the prefill's storm).
kernel void unit_lower_inverse_f32(
    const device float* a [[buffer(0)]],
    device float* t [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint b [[threadgroup_position_in_grid]],
    uint j [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  const device float* am = a + (size_t)b * n * n;
  device float* tm = t + (size_t)b * n * n;
  for (uint i = 0; i < n; i++) {
    for (uint jj = j; jj < n; jj += width) {
      float acc = (i == jj) ? 1.0f : 0.0f;
      for (uint m = 0; m < i; m++) {
        acc -= am[i * n + m] * tm[m * n + jj];
      }
      tm[i * n + jj] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

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

// `out = silu(g) · u`, elementwise (the routed experts' SwiGLU and the shared
// expert's): two launches into one.
kernel void silu_mul_f32(
    const device float* g [[buffer(0)]],
    const device float* u [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= n) return;
  const float x = g[idx];
  out[idx] = (x / (1.0f + metal::exp(-x))) * u[idx];
}

// `out[r, i] = Σ_k w[r, k] · y[r, k, i] + shared[r, i] · sigmoid(gate[r])` —
// the routed combine over the `k` selected experts plus the gated shared
// expert (the layer's MoE output). One thread per output element; the
// composed path was a broadcast multiply, a sum, a sigmoid, a multiply and
// an add.
kernel void moe_combine_f32(
    const device float* y [[buffer(0)]],
    const device float* w [[buffer(1)]],
    const device float* shared [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= rows * d) return;
  const uint r = idx / d;
  const uint i = idx % d;
  float acc = 0.0f;
  for (uint e = 0; e < k; e++) {
    acc += w[(size_t)r * k + e] * y[((size_t)r * k + e) * d + i];
  }
  const float gz = gate[r];
  const float sg = 1.0f / (1.0f + metal::exp(-gz));
  out[idx] = acc + shared[idx] * sg;
}

// ---- Gated DeltaNet decode (`deltanet_block.rs`) ------------------------------
//
// One position through a linear-attention layer as three launches between
// the projections: the prologue (conv window, SiLU, the head split, the L2
// norms, the GQA tile, the query scale), the recurrence on the `S` slab in
// place, and the gated norm. The composed path was ~40 launches.

METAL_FUNC float gdn_silu(float x) {
  return x / (1.0f + metal::exp(-x));
}

// `qkv [conv_ch]` (the fused projection's output) against the layer's conv
// window `win [conv_ch, kk]` (newest input at column kk−1; shifted and
// appended IN PLACE): the folded, SiLU'd channels split into q/k (`h_k`
// heads of `d`, L2-normed per head over `d`, tiled to `h_v` heads — v-head
// `j` reads k-head `j % h_k` — q scaled by `q_scale`) and v (`h_v` heads of
// `d`), written as `heads [3, h_v, d]` (q, k, v). Threadgroups: `h_k` for q,
// `h_k` for k, `h_v` for v; `d` threads each, one channel per thread.
kernel void gdn_prologue_f32(
    const device float* qkv [[buffer(0)]],
    const device float* conv_w [[buffer(1)]],
    device float* win [[buffer(2)]],
    device float* heads [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& h_k [[buffer(5)]],
    constant uint& h_v [[buffer(6)]],
    constant uint& kk [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const uint key_dim = h_k * d;
  // which plane, which head, which channel base
  uint plane, head, base;
  if (group < h_k) {
    plane = 0; head = group; base = head * d;
  } else if (group < 2 * h_k) {
    plane = 1; head = group - h_k; base = key_dim + head * d;
  } else {
    plane = 2; head = group - 2 * h_k; base = 2 * key_dim + head * d;
  }
  float y = 0.0f;
  if (i < d) {
    const uint c = base + i;
    // shift the window left by one, append the input, fold against the taps
    float folded = 0.0f;
    device float* w = win + (size_t)c * kk;
    const device float* cw = conv_w + (size_t)c * kk;
    for (uint j = 0; j + 1 < kk; j++) {
      const float v = w[j + 1];
      w[j] = v;
      folded += cw[j] * v;
    }
    const float x = qkv[c];
    w[kk - 1] = x;
    folded += cw[kk - 1] * x;
    y = gdn_silu(folded);
  }
  if (plane == 2) {
    if (i < d) {
      heads[(size_t)(2 * h_v + head) * d + i] = y;
    }
    return;
  }
  // L2 norm over the head's d channels
  float ss = y * y;
  ss = simd_sum(ss);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  const float inv = 1.0f / metal::sqrt(total + eps);
  float out = y * inv;
  if (plane == 0) out *= q_scale;
  if (i < d) {
    const uint rep = h_v / h_k;
    for (uint r = 0; r < rep; r++) {
      const uint j = r * h_k + head; // TILE: v-head j reads k-head j % h_k
      heads[(size_t)(plane * h_v + j) * d + i] = out;
    }
  }
}

// The recurrence on the layer's `S [d, d, h_v]` slab (head-last, as stored)
// IN PLACE, from `heads [3, h_v, d]` and the raw gate logits:
//   g = ssm_a · softplus(alpha + dt_bias); β = sigmoid(beta)
//   S ← e^g · S;  r = kᵀ S;  δ = β (v − r);  S ← S + k ⊗ δ;  o = qᵀ S
// One threadgroup per `dv`, `h_v` threads (one head each): the head is the
// innermost slab index, so every read across the group is contiguous.
kernel void gdn_step_f32(
    const device float* heads [[buffer(0)]],
    const device float* alpha [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* dt_bias [[buffer(3)]],
    const device float* ssm_a [[buffer(4)]],
    device float* s [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    constant uint& h_v [[buffer(8)]],
    uint dv [[threadgroup_position_in_grid]],
    uint h [[thread_position_in_threadgroup]]) {
  if (h >= h_v) return;
  const device float* q = heads + (size_t)h * d;
  const device float* k = heads + (size_t)(h_v + h) * d;
  const device float* v = heads + (size_t)(2 * h_v + h) * d;
  // gate and write strength (the composed `gate_and_beta` / `softplus`)
  const float a = alpha[h] + dt_bias[h];
  const float sp = metal::max(a, 0.0f) + metal::log(1.0f + metal::exp(-metal::fabs(a)));
  const float eg = metal::exp(ssm_a[h] * sp);
  const float b = 1.0f / (1.0f + metal::exp(-beta[h]));
  // r = Σ_dk k[dk] · (e^g S[dk, dv, h])
  float r = 0.0f;
  for (uint dk = 0; dk < d; dk++) {
    r += k[dk] * (eg * s[((size_t)dk * d + dv) * h_v + h]);
  }
  const float delta = b * (v[dv] - r);
  float acc = 0.0f;
  for (uint dk = 0; dk < d; dk++) {
    const size_t idx = ((size_t)dk * d + dv) * h_v + h;
    const float sn = eg * s[idx] + k[dk] * delta;
    s[idx] = sn;
    acc += q[dk] * sn;
  }
  o[(size_t)h * d + dv] = acc;
}

// `out[h·d + i] = rmsnorm_d(o[h, ·])[i] · γ[i] · sigmoid(z[h·d + i])` — the
// gated RMS norm over each value head, scaled by the output gate. One
// threadgroup per head, `d` threads.
kernel void gdn_gated_norm_f32(
    const device float* o [[buffer(0)]],
    const device float* gamma [[buffer(1)]],
    const device float* z [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint h [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const size_t idx = (size_t)h * d + i;
  const float x = (i < d) ? o[idx] : 0.0f;
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint sidx = 0; sidx < n_simd; sidx++) total += partial[sidx];
  if (i < d) {
    const float scale = 1.0f / metal::sqrt(total / (float)d + eps);
    const float sg = 1.0f / (1.0f + metal::exp(-z[idx]));
    out[idx] = x * scale * gamma[i] * sg;
  }
}

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
