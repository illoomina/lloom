#include <metal_stdlib>

using namespace metal;

// The Gated DeltaNet block's kernels: the decode prologue, the recurrence
// step, the gated norm, their rows and multi-position forms, and the chunked
// prefill. The prologue's conv is the short-conv block (`conv/short_conv.metal`,
// spliced ahead of this text); what follows it here is GDN's epilogue.

// `(I + A)⁻¹` for a strictly-lower-triangular `A [batch, n, n]` (row-major),
// by forward substitution — `T[i, :] = e_i − Σ_{m<i} A[i, m] · T[m, :]`, each
// row from the rows before it (the numerically stable form; the nilpotent
// product overflows on real keys, see `deltanet::unit_lower_inverse`).
//
// One threadgroup per matrix; thread `j` owns column `j` (and every column
// `j + k·width`), rows run in order with a device-memory barrier between
// them, so row `i` reads the rows `< i` the group has already stored. `n`
// is the prefill chunk.
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

// ---- Decode --------------------------------------------------------------------
//
// One position through a linear-attention layer as three launches between
// the projections: the prologue (the conv, SiLU, the head split, the L2
// norms, the GQA tile, the query scale), the recurrence on the `S` slab, and
// the gated norm.

METAL_FUNC float gdn_silu(float x) {
  return x / (1.0f + metal::exp(-x));
}

// The part of the concat-QKV row a prologue threadgroup owns: `h_k` groups
// for q, `h_k` for k, `h_v` for v, each one head of `d` channels.
struct gdn_group {
  uint plane; // 0 q, 1 k, 2 v
  uint head;
  uint base;  // the head's first channel
};

METAL_FUNC gdn_group gdn_group_of(uint group, uint d, uint h_k) {
  const uint key_dim = h_k * d;
  if (group < h_k) {
    return gdn_group { 0, group, group * d };
  } else if (group < 2 * h_k) {
    return gdn_group { 1, group - h_k, key_dim + (group - h_k) * d };
  }
  return gdn_group { 2, group - 2 * h_k, 2 * key_dim + (group - 2 * h_k) * d };
}

// Where the prologue writes head-row `r` of `[3, h_v, d]` (q, k, v).
struct gdn_heads_rows {
  device float *heads;
  uint d;
  METAL_FUNC device float *operator()(uint r) const { return heads + (size_t)r * d; }
};

// The same for the prefill's chunked head-first `[3, h_v, nc, cs, d]` at
// chunk `pc`, position `pi` within it.
struct gdn_heads_chunked {
  device float *out;
  uint d, nc, cs, pc, pi;
  METAL_FUNC device float *operator()(uint r) const {
    return out + (((size_t)r * nc + pc) * cs) * d + (size_t)pi * d;
  }
};

// GDN's epilogue over the conv's SiLU'd channel `y`: v stored as it is; q
// and k L2-normed over the head's `d` channels, q scaled by `q_scale`, and
// tiled to the `h_v` value heads (v-head `j` reads k-head `j % h_k`). The
// threadgroup is one head, so the norm's reduction is the group's.
template <typename Dst>
METAL_FUNC void gdn_heads_epilogue(
    float y, gdn_group g, uint i, uint d, uint h_k, uint h_v, float eps, float q_scale,
    thread const Dst &dst, threadgroup float *partial, uint width, uint simd_lane,
    uint simd_id) {
  if (g.plane == 2) {
    if (i < d) {
      dst(2 * h_v + g.head)[i] = y;
    }
    return;
  }
  float ss = y * y;
  ss = simd_sum(ss);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  const float inv = 1.0f / metal::sqrt(total + eps);
  float out = y * inv;
  if (g.plane == 0) out *= q_scale;
  if (i < d) {
    const uint rep = h_v / h_k;
    for (uint r = 0; r < rep; r++) {
      dst(g.plane * h_v + r * h_k + g.head)[i] = out;
    }
  }
}

// `qkv [conv_ch]` (the fused projection's output) against the layer's conv
// window `win [conv_ch, kk]` (newest input at column kk−1), the window after
// the step written to `win_out`: the block's shift window, then SiLU and the
// heads epilogue into `heads [3, h_v, d]`. Threadgroups: `h_k` for q, `h_k`
// for k, `h_v` for v; `d` threads each, one channel per thread.
kernel void gdn_prologue_f32(
    const device float* qkv [[buffer(0)]],
    const device float* conv_w [[buffer(1)]],
    const device float* win [[buffer(2)]],
    device float* heads [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& h_k [[buffer(5)]],
    constant uint& h_v [[buffer(6)]],
    constant uint& kk [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    device float* win_out [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[32];
  const gdn_group g = gdn_group_of(group, d, h_k);
  float y = 0.0f;
  if (i < d) {
    const uint c = g.base + i;
    const short_conv_shift_taps<float> taps { win + (size_t)c * kk, qkv[c] };
    y = gdn_silu(short_conv_fold_into<float>(
        taps, conv_w + (size_t)c * kk, win_out + (size_t)c * kk, kk));
  }
  const gdn_heads_rows dst { heads, d };
  gdn_heads_epilogue(y, g, i, d, h_k, h_v, eps, q_scale, dst, partial, width, simd_lane, simd_id);
}

// The recurrence on the layer's `S [d, d, h_v]` slab (head-last, as stored),
// read from `s` and written to `s_out`, from `heads [3, h_v, d]` and the raw
// gate logits:
//   g = ssm_a · softplus(alpha + dt_bias); β = sigmoid(beta)
//   S ← e^g · S;  r = kᵀ S;  δ = β (v − r);  S ← S + k ⊗ δ;  o = qᵀ S
// One threadgroup per `dv`, `h_v × split` threads: thread (h, j) owns head
// `h` over the `dk ≡ j (mod split)` slice — the head is the innermost slab
// index, so a warp's reads are contiguous — the two reductions over `dk`
// (the read-out `r`, the output `o`) closing across the slices through
// threadgroup memory. `split` is the width over `h_v`, the host's choice.
kernel void gdn_step_f32(
    const device float* heads [[buffer(0)]],
    const device float* alpha [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    const device float* dt_bias [[buffer(3)]],
    const device float* ssm_a [[buffer(4)]],
    const device float* s [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant uint& d [[buffer(7)]],
    constant uint& h_v [[buffer(8)]],
    device float* s_out [[buffer(9)]],
    uint dv [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
  threadgroup float part[1024];
  const uint split = width / h_v;
  const uint h = t % h_v;
  const uint j = t / h_v;
  const bool live = j < split; // width is h_v · split; a stray thread only idles
  const device float* q = heads + (size_t)h * d;
  const device float* k = heads + (size_t)(h_v + h) * d;
  const device float* v = heads + (size_t)(2 * h_v + h) * d;
  // gate and write strength (the composed `gate_and_beta` / `softplus`)
  const float a = alpha[h] + dt_bias[h];
  const float sp = metal::max(a, 0.0f) + metal::log(1.0f + metal::exp(-metal::fabs(a)));
  const float eg = metal::exp(ssm_a[h] * sp);
  const float b = 1.0f / (1.0f + metal::exp(-beta[h]));
  // r = Σ_dk k[dk] · (e^g S[dk, dv, h]), the slice's part then the slices
  float r = 0.0f;
  if (live) {
    for (uint dk = j; dk < d; dk += split) {
      r += k[dk] * (eg * s[((size_t)dk * d + dv) * h_v + h]);
    }
    part[t] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  r = 0.0f;
  for (uint jj = 0; jj < split; jj++) r += part[jj * h_v + h];
  const float delta = b * (v[dv] - r);
  float acc = 0.0f;
  if (live) {
    for (uint dk = j; dk < d; dk += split) {
      const size_t idx = ((size_t)dk * d + dv) * h_v + h;
      const float sn = eg * s[idx] + k[dk] * delta;
      s_out[idx] = sn;
      acc += q[dk] * sn;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (live) part[t] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (j == 0) {
    float sum = 0.0f;
    for (uint jj = 0; jj < split; jj++) sum += part[jj * h_v + h];
    o[(size_t)h * d + dv] = sum;
  }
}

// One recurrence step of column `dv` for head `h` over the `dk` slice
// `j, j+split, …`, with the slice's values of `S` (at most `R`) held in
// registers between the read-out and the update, so the slab is read once
// rather than twice: 6 MB a layer at the real geometry where the loop form
// (`gdn_step_column`) moves 9. Every register-form kernel runs this column,
// so a row's arithmetic is the same alone, among rows and in a window. The
// rounding is the compiler's for this shape (fast math), so the result is not
// the loop form's bit for bit (§2.3.5a). The host dispatches the register
// form only when `ceil(d / split) ≤ R`.
template <uint R>
static inline void gdn_step_column_reg(
    const device float* q, const device float* k, const device float* v,
    const float eg, const float bt,
    const device float* s, device float* s_out,
    const uint d, const uint h_v, const uint dv, const uint h, const uint j,
    const uint split, const bool live, const uint t,
    threadgroup float* part, device float* o_out) {
  float sv[R];
  float r = 0.0f;
  if (live) {
    uint u = 0;
    for (uint dk = j; dk < d; dk += split) {
      const float x = s[((size_t)dk * d + dv) * h_v + h];
      sv[u++] = x;
      r += k[dk] * (eg * x);
    }
    part[t] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  r = 0.0f;
  for (uint jj = 0; jj < split; jj++) r += part[jj * h_v + h];
  const float delta = bt * (v[dv] - r);
  float acc = 0.0f;
  if (live) {
    uint u = 0;
    for (uint dk = j; dk < d; dk += split) {
      const float sn = eg * sv[u++] + k[dk] * delta;
      s_out[((size_t)dk * d + dv) * h_v + h] = sn;
      acc += q[dk] * sn;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (live) part[t] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (j == 0) {
    float sum = 0.0f;
    for (uint jj = 0; jj < split; jj++) sum += part[jj * h_v + h];
    o_out[(size_t)h * d + dv] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// `gdn_step_f32` in the register form (`gdn_step_column_reg`).
#define GDN_STEP_REG(R)                                                                     \
  kernel void gdn_step_r##R##_f32(                                                          \
      const device float* heads [[buffer(0)]], const device float* alpha [[buffer(1)]],     \
      const device float* beta [[buffer(2)]], const device float* dt_bias [[buffer(3)]],    \
      const device float* ssm_a [[buffer(4)]], const device float* s [[buffer(5)]],         \
      device float* o [[buffer(6)]], constant uint& d [[buffer(7)]],                        \
      constant uint& h_v [[buffer(8)]], device float* s_out [[buffer(9)]],                  \
      uint dv [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], \
      uint width [[threads_per_threadgroup]]) {                                             \
    threadgroup float part[1024];                                                           \
    const uint split = width / h_v;                                                         \
    const uint h = t % h_v;                                                                 \
    const uint j = t / h_v;                                                                 \
    const float a = alpha[h] + dt_bias[h];                                                  \
    const float sp = metal::max(a, 0.0f) + metal::log(1.0f + metal::exp(-metal::fabs(a))); \
    const float eg = metal::exp(ssm_a[h] * sp);                                             \
    const float b = 1.0f / (1.0f + metal::exp(-beta[h]));                                   \
    gdn_step_column_reg<R>(heads + (size_t)h * d, heads + (size_t)(h_v + h) * d,           \
                           heads + (size_t)(2 * h_v + h) * d, eg, b, s, s_out, d, h_v, dv,  \
                           h, j, split, j < split, t, part, o);                             \
  }
GDN_STEP_REG(16)
GDN_STEP_REG(32)

// The output gate of the gated RMS norm: `sigmoid(z)` (Qwen4-exp) or
// `silu(z)` (Qwen3.5, `output_gate_type: swish`). One body per kernel,
// instantiated per gate; the gate is the block's (`GdnGate`).
template <bool SILU>
METAL_FUNC float gdn_out_gate(float z) {
  const float sg = 1.0f / (1.0f + metal::exp(-z));
  return SILU ? z * sg : sg;
}

// `out[h·d + i] = rmsnorm_d(o[h, ·])[i] · γ[i] · gate(z[h·d + i])` — the
// gated RMS norm over each value head, scaled by the output gate. One
// threadgroup per head, `d` threads.
template <bool SILU>
METAL_FUNC void gdn_gated_norm_body(
    const device float* o,
    const device float* gamma,
    const device float* z,
    device float* out,
    uint d,
    float eps,
    uint h,
    uint i,
    uint width,
    uint simd_lane,
    uint simd_id,
    threadgroup float* partial) {
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
    out[idx] = x * scale * gamma[i] * gdn_out_gate<SILU>(z[idx]);
  }
}

#define GDN_GATED_NORM(NAME, SILU)                                                  \
  kernel void NAME(                                                                 \
      const device float* o [[buffer(0)]],                                          \
      const device float* gamma [[buffer(1)]],                                      \
      const device float* z [[buffer(2)]],                                          \
      device float* out [[buffer(3)]],                                              \
      constant uint& d [[buffer(4)]],                                               \
      constant float& eps [[buffer(5)]],                                            \
      uint h [[threadgroup_position_in_grid]],                                      \
      uint i [[thread_position_in_threadgroup]],                                    \
      uint width [[threads_per_threadgroup]],                                       \
      uint simd_lane [[thread_index_in_simdgroup]],                                 \
      uint simd_id [[simdgroup_index_in_threadgroup]]) {                            \
    threadgroup float partial[32];                                                  \
    gdn_gated_norm_body<SILU>(o, gamma, z, out, d, eps, h, i, width, simd_lane,     \
                              simd_id, partial);                                    \
  }
GDN_GATED_NORM(gdn_gated_norm_f32, false)
GDN_GATED_NORM(gdn_gated_norm_silu_f32, true)

// ---- Chunked prefill ------------------------------------------------------------

// The gate and the write strength over a chunk, laid out chunked
// `[2, h_v, nc, cs]` (plane 0: g = ssm_a · softplus(alpha + dt_bias); plane 1:
// β = sigmoid(beta)); positions ≥ `seq` are zero (no decay, no write).
kernel void gdn_gate_f32(
    const device float* alpha [[buffer(0)]],
    const device float* beta_raw [[buffer(1)]],
    const device float* dt_bias [[buffer(2)]],
    const device float* ssm_a [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq [[buffer(5)]],
    constant uint& h_v [[buffer(6)]],
    constant uint& nc [[buffer(7)]],
    constant uint& cs [[buffer(8)]],
    uint idx [[thread_position_in_grid]]) {
  const uint total = h_v * nc * cs;
  if (idx >= total) return;
  const uint h = idx / (nc * cs);
  const uint p = idx % (nc * cs);
  float g = 0.0f;
  float b = 0.0f;
  if (p < seq) {
    const float a = alpha[(size_t)p * h_v + h] + dt_bias[h];
    const float sp = metal::max(a, 0.0f) + metal::log(metal::exp(-metal::abs(a)) + 1.0f);
    g = ssm_a[h] * sp;
    b = 1.0f / (1.0f + metal::exp(-beta_raw[(size_t)p * h_v + h]));
  }
  out[idx] = g;
  out[total + idx] = b;
}

// The conv over the chunk against the carried window (read only: the block's
// virtual sequence), SiLU and the heads epilogue, written chunked head-first
// `[3, h_v, nc, cs, d]` (planes q, k, v), zero past `seq`. One threadgroup per
// (plane-head, position), `d` threads.
kernel void gdn_prefill_prologue_f32(
    const device float* qkv [[buffer(0)]],
    const device float* win [[buffer(1)]],
    const device float* conv_w [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& h_k [[buffer(6)]],
    constant uint& h_v [[buffer(7)]],
    constant uint& kk [[buffer(8)]],
    constant uint& cs [[buffer(9)]],
    constant uint& nc [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant float& q_scale [[buffer(12)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = tid.x;
  const uint width = tptg.x;
  threadgroup float partial[32];
  const uint p = gid.y;
  const uint C = (2 * h_k + h_v) * d;
  const gdn_group g = gdn_group_of(gid.x, d, h_k);
  float y = 0.0f;
  if (p < seq && i < d) {
    const uint c = g.base + i;
    const short_conv_seq_taps<float> taps { win + (size_t)c * kk, qkv + c, C, p };
    y = gdn_silu(short_conv_fold<float>(taps, conv_w + (size_t)c * kk, kk));
  }
  const gdn_heads_chunked dst { out, d, nc, cs, p / cs, p % cs };
  gdn_heads_epilogue(y, g, i, d, h_k, h_v, eps, q_scale, dst, partial, width, simd_lane, simd_id);
}

// The window after the chunk: the taps of its last position — the last `kk`
// inputs, from the chunk or from the old window when the chunk is shorter —
// newest last, a new `[C, kk]`.
kernel void gdn_conv_window_f32(
    const device float* win [[buffer(0)]],
    const device float* qkv [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& C [[buffer(4)]],
    constant uint& kk [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= C * kk) return;
  const uint c = idx / kk;
  const uint j = idx % kk;
  const short_conv_seq_taps<float> taps { win + (size_t)c * kk, qkv + c, C, seq - 1 };
  out[idx] = taps.tap(j, kk);
}

// From the chunked gate `g [h_v, nc, cs]`, the intra-chunk decay, flat:
//   g_exp [h_v, nc, cs] = exp(G)            (G = cumsum of g within the chunk)
//   g_diff [h_v, nc, cs] = exp(G_last − G)
//   g_last [h_v, nc] = exp(G_last)
//   decay [h_v, nc, cs, cs][b][a] = exp(G[b] − G[a]) for a ≤ b, else 0
//   decay_strict: the same with the diagonal zero.
// One threadgroup per (head, chunk), `cs` threads; `cs` ≤ 256.
kernel void gdn_chunk_decay_f32(
    const device float* g [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& h_v [[buffer(2)]],
    constant uint& nc [[buffer(3)]],
    constant uint& cs [[buffer(4)]],
    uint hc [[threadgroup_position_in_grid]],
    uint i [[thread_position_in_threadgroup]]) {
  threadgroup float gcs[256];
  const device float* gg = g + (size_t)hc * cs;
  if (i < cs) {
    float s = 0.0f;
    for (uint j = 0; j <= i; j++) s += gg[j];
    gcs[i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i >= cs) return;
  const uint n = h_v * nc;
  const float gi = gcs[i];
  const float gl = gcs[cs - 1];
  out[(size_t)hc * cs + i] = metal::exp(gi);
  out[(size_t)n * cs + (size_t)hc * cs + i] = metal::exp(gl - gi);
  if (i == 0) out[(size_t)2 * n * cs + hc] = metal::exp(gl);
  device float* dec = out + (size_t)2 * n * cs + n + ((size_t)hc * cs + i) * cs;
  device float* decs = dec + (size_t)n * cs * cs;
  for (uint a = 0; a < cs; a++) {
    const float v = a <= i ? metal::exp(gi - gcs[a]) : 0.0f;
    dec[a] = v;
    decs[a] = a < i ? v : 0.0f;
  }
}

// The per-position scalings of the chunked q / k / v (`[3, n, d]`, n = h_v·nc·cs)
// by β and the decay terms, flat: k_beta = k·β | w_in = k·β·g_exp |
// q_g = q·g_exp | kg = k·g_diff | v_beta = v·β, each `[n, d]`.
kernel void gdn_chunk_scale_f32(
    const device float* qkv [[buffer(0)]],
    const device float* gb [[buffer(1)]],
    const device float* small [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  const uint total = n * d;
  if (idx >= total) return;
  const uint r = idx / d;
  const float q = qkv[idx];
  const float k = qkv[total + idx];
  const float v = qkv[(size_t)2 * total + idx];
  const float b = gb[n + r];
  const float ge = small[r];
  const float gd = small[n + r];
  const float kb = k * b;
  out[idx] = kb;
  out[total + idx] = kb * ge;
  out[(size_t)2 * total + idx] = q * ge;
  out[(size_t)3 * total + idx] = k * gd;
  out[(size_t)4 * total + idx] = v * b;
}

// The chunks' outputs `o [h_v, padded, d]` (head-first) → the gated RMS norm
// per (position, head) × γ × gate(z) → `[seq, h_v·d]`. One threadgroup
// per (head, position), `d` threads.
template <bool SILU>
METAL_FUNC void gdn_chunk_epilogue_body(
    const device float* o,
    const device float* gamma,
    const device float* z,
    device float* out,
    uint padded,
    uint d,
    uint h_v,
    float eps,
    uint2 gid,
    uint2 tid,
    uint2 tptg,
    uint simd_lane,
    uint simd_id,
    threadgroup float* partial) {
  const uint i = tid.x;
  const uint width = tptg.x;
  const uint h = gid.x;
  const uint p = gid.y;
  const float x = (i < d) ? o[((size_t)h * padded + p) * d + i] : 0.0f;
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint s = 0; s < n_simd; s++) total += partial[s];
  if (i < d) {
    const float scale = 1.0f / metal::sqrt(total / (float)d + eps);
    const size_t oi = ((size_t)p * h_v + h) * d + i;
    out[oi] = x * scale * gamma[i] * gdn_out_gate<SILU>(z[oi]);
  }
}

#define GDN_CHUNK_EPILOGUE(NAME, SILU)                                              \
  kernel void NAME(                                                                 \
      const device float* o [[buffer(0)]],                                          \
      const device float* gamma [[buffer(1)]],                                      \
      const device float* z [[buffer(2)]],                                          \
      device float* out [[buffer(3)]],                                              \
      constant uint& padded [[buffer(4)]],                                          \
      constant uint& d [[buffer(5)]],                                               \
      constant uint& h_v [[buffer(6)]],                                             \
      constant float& eps [[buffer(7)]],                                            \
      uint2 gid [[threadgroup_position_in_grid]],                                   \
      uint2 tid [[thread_position_in_threadgroup]],                                 \
      uint2 tptg [[threads_per_threadgroup]],                                       \
      uint simd_lane [[thread_index_in_simdgroup]],                                 \
      uint simd_id [[simdgroup_index_in_threadgroup]]) {                            \
    threadgroup float partial[32];                                                  \
    gdn_chunk_epilogue_body<SILU>(o, gamma, z, out, padded, d, h_v, eps, gid, tid,  \
                                  tptg, simd_lane, simd_id, partial);               \
  }
GDN_CHUNK_EPILOGUE(gdn_chunk_epilogue_f32, false)
GDN_CHUNK_EPILOGUE(gdn_chunk_epilogue_silu_f32, true)

// ---- Decode over the rows of one state pool, and over the positions of one row ----

// A state pool's layer is `[phases, slots, ..row]`; row `slot` at `phase` is
// at `(phase * n_slots + slot) * row_elems`, and the next phase is
// `(phase + 1) % n_phases`.

// `gdn_prologue_f32` over `B` rows: `qkv [B, C]`, the window pool (row `b`
// reads `phases[b]` at `slots[b]`, writes the next phase), `heads [B, 3,
// h_v, d]`. Grid: (2·h_k + h_v) × B threadgroups.
kernel void gdn_prologue_rows_f32(
    const device float* qkv [[buffer(0)]],
    const device float* conv_w [[buffer(1)]],
    const device float* win_pool [[buffer(2)]],
    const device uint* slots [[buffer(3)]],
    const device uint* phases [[buffer(4)]],
    device float* heads [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& h_k [[buffer(7)]],
    constant uint& h_v [[buffer(8)]],
    constant uint& kk [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant float& q_scale [[buffer(11)]],
    constant uint& n_slots [[buffer(12)]],
    constant uint& n_phases [[buffer(13)]],
    device float* win_pool_out [[buffer(14)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = tid.x;
  const uint width = tptg.x;
  threadgroup float partial[32];
  const uint b = gid.y;
  const uint C = (2 * h_k + h_v) * d;
  const size_t row = (size_t)C * kk;
  const uint slot = slots[b];
  const uint phase = phases[b];
  const device float* win = win_pool + ((size_t)phase * n_slots + slot) * row;
  device float* wout = win_pool_out + ((size_t)((phase + 1) % n_phases) * n_slots + slot) * row;
  const device float* x_row = qkv + (size_t)b * C;
  const gdn_group g = gdn_group_of(gid.x, d, h_k);
  float y = 0.0f;
  if (i < d) {
    const uint c = g.base + i;
    const short_conv_shift_taps<float> taps { win + (size_t)c * kk, x_row[c] };
    y = gdn_silu(short_conv_fold_into<float>(
        taps, conv_w + (size_t)c * kk, wout + (size_t)c * kk, kk));
  }
  const gdn_heads_rows dst { heads + (size_t)b * 3 * h_v * d, d };
  gdn_heads_epilogue(y, g, i, d, h_k, h_v, eps, q_scale, dst, partial, width, simd_lane, simd_id);
}

// `gdn_prologue_f32` over `T` POSITIONS of one row: `qkv [T, C]`, the
// window at `phase` of `slot` is the state before position 0; position t's
// conv reads the block's virtual sequence, and the window after every
// position is written to phase `(phase + 1 + t) % n_phases` — so an accept
// of `a` positions is phase `phase + a`. `heads [T, 3, h_v, d]`. Grid:
// (2·h_k + h_v) × T.
kernel void gdn_prologue_seq_f32(
    const device float* qkv [[buffer(0)]],
    const device float* conv_w [[buffer(1)]],
    const device float* win_pool [[buffer(2)]],
    device float* heads [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& h_k [[buffer(5)]],
    constant uint& h_v [[buffer(6)]],
    constant uint& kk [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    constant uint& n_slots [[buffer(10)]],
    constant uint& n_phases [[buffer(11)]],
    constant uint& slot [[buffer(12)]],
    constant uint& phase [[buffer(13)]],
    device float* win_pool_out [[buffer(14)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  const uint i = tid.x;
  const uint width = tptg.x;
  threadgroup float partial[32];
  const uint t = gid.y;
  const uint C = (2 * h_k + h_v) * d;
  const size_t row = (size_t)C * kk;
  const device float* win = win_pool + ((size_t)phase * n_slots + slot) * row;
  device float* wout = win_pool_out + ((size_t)((phase + 1 + t) % n_phases) * n_slots + slot) * row;
  const gdn_group g = gdn_group_of(gid.x, d, h_k);
  float y = 0.0f;
  if (i < d) {
    const uint c = g.base + i;
    const short_conv_seq_taps<float> taps { win + (size_t)c * kk, qkv + c, C, t };
    y = gdn_silu(short_conv_fold_into<float>(
        taps, conv_w + (size_t)c * kk, wout + (size_t)c * kk, kk));
  }
  const gdn_heads_rows dst { heads + (size_t)t * 3 * h_v * d, d };
  gdn_heads_epilogue(y, g, i, d, h_k, h_v, eps, q_scale, dst, partial, width, simd_lane, simd_id);
}

// One recurrence step of column `dv` for head `h` over the `dk` slice
// `j, j+split, …` — the body `gdn_step_f32` runs; here shared by the rows
// and the multi-step forms. `part` is the group's `[h_v · split]` scratch.
static inline void gdn_step_column(
    const device float* q, const device float* k, const device float* v,
    const float eg, const float bt,
    const device float* s, device float* s_out,
    const uint d, const uint h_v, const uint dv, const uint h, const uint j,
    const uint split, const bool live, const uint t,
    threadgroup float* part, device float* o_out) {
  float r = 0.0f;
  if (live) {
    for (uint dk = j; dk < d; dk += split) {
      r += k[dk] * (eg * s[((size_t)dk * d + dv) * h_v + h]);
    }
    part[t] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  r = 0.0f;
  for (uint jj = 0; jj < split; jj++) r += part[jj * h_v + h];
  const float delta = bt * (v[dv] - r);
  float acc = 0.0f;
  if (live) {
    for (uint dk = j; dk < d; dk += split) {
      const size_t idx = ((size_t)dk * d + dv) * h_v + h;
      const float sn = eg * s[idx] + k[dk] * delta;
      s_out[idx] = sn;
      acc += q[dk] * sn;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (live) part[t] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (j == 0) {
    float sum = 0.0f;
    for (uint jj = 0; jj < split; jj++) sum += part[jj * h_v + h];
    o_out[(size_t)h * d + dv] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float gdn_gate_eg(const float alpha, const float dt_bias, const float ssm_a) {
  const float a = alpha + dt_bias;
  const float sp = metal::max(a, 0.0f) + metal::log(1.0f + metal::exp(-metal::fabs(a)));
  return metal::exp(ssm_a * sp);
}

// The column a step kernel runs, by form: `R = 0` the loop form
// (`gdn_step_column`), else the register form holding up to `R` values.
template <uint R>
struct GdnColumn {
  static inline void run(const device float* q, const device float* k, const device float* v,
                         const float eg, const float bt, const device float* s,
                         device float* s_out, const uint d, const uint h_v, const uint dv,
                         const uint h, const uint j, const uint split, const bool live,
                         const uint t, threadgroup float* part, device float* o_out) {
    gdn_step_column_reg<R>(q, k, v, eg, bt, s, s_out, d, h_v, dv, h, j, split, live, t, part,
                           o_out);
  }
};
template <>
struct GdnColumn<0> {
  static inline void run(const device float* q, const device float* k, const device float* v,
                         const float eg, const float bt, const device float* s,
                         device float* s_out, const uint d, const uint h_v, const uint dv,
                         const uint h, const uint j, const uint split, const bool live,
                         const uint t, threadgroup float* part, device float* o_out) {
    gdn_step_column(q, k, v, eg, bt, s, s_out, d, h_v, dv, h, j, split, live, t, part, o_out);
  }
};

// `gdn_step_f32` over `B` rows: `heads [B, 3, h_v, d]`, `alpha` / `beta`
// `[B, h_v]`, the `S` pool (row `b` reads `phases[b]` at `slots[b]`, writes
// the next phase), `o [B, h_v, d]`. Grid: d × B threadgroups of `h_v × split`.
template <uint R>
static inline void gdn_step_rows_body(
    const device float* heads, const device float* alpha, const device float* beta,
    const device float* dt_bias, const device float* ssm_a, const device float* s_pool,
    const device uint* slots, const device uint* phases, device float* o, const uint d,
    const uint h_v, const uint n_slots, const uint n_phases, device float* s_pool_out,
    const uint2 gid, const uint2 tid, const uint2 tptg, threadgroup float* part) {
  const uint dv = gid.x;
  const uint b = gid.y;
  const uint t = tid.x;
  const uint width = tptg.x;
  const uint split = width / h_v;
  const uint h = t % h_v;
  const uint j = t / h_v;
  const bool live = j < split;
  const size_t slab = (size_t)d * d * h_v;
  const uint slot = slots[b];
  const uint phase = phases[b];
  const device float* s = s_pool + ((size_t)phase * n_slots + slot) * slab;
  device float* s_out = s_pool_out + ((size_t)((phase + 1) % n_phases) * n_slots + slot) * slab;
  const device float* hrow = heads + (size_t)b * 3 * h_v * d;
  const float eg = gdn_gate_eg(alpha[(size_t)b * h_v + h], dt_bias[h], ssm_a[h]);
  const float bt = 1.0f / (1.0f + metal::exp(-beta[(size_t)b * h_v + h]));
  GdnColumn<R>::run(hrow + (size_t)h * d, hrow + (size_t)(h_v + h) * d,
                    hrow + (size_t)(2 * h_v + h) * d, eg, bt, s, s_out, d, h_v, dv, h, j, split,
                    live, t, part, o + (size_t)b * h_v * d);
}

// `gdn_step_f32` over `T` POSITIONS of one row, in one launch: position t
// reads `S` at phase `phase + t` and writes `phase + 1 + t` (mod
// `n_phases`), so every intermediate state is kept and an accept of `a`
// positions is phase `phase + a`. `heads [T, 3, h_v, d]`, `alpha` / `beta`
// `[T, h_v]`, `o [T, h_v, d]`. A threadgroup owns column `dv` for every
// head — the recurrence of a column depends on that column alone — so the
// positions run in sequence inside the group, each step the arithmetic of
// the one-row step in the same form. Grid: d threadgroups of `h_v × split`.
template <uint R>
static inline void gdn_step_seq_body(
    const device float* heads, const device float* alpha, const device float* beta,
    const device float* dt_bias, const device float* ssm_a, const device float* s_pool,
    device float* o, const uint d, const uint h_v, const uint n_slots, const uint n_phases,
    const uint slot, const uint phase, const uint steps, device float* s_pool_out,
    const uint dv, const uint t, const uint width, threadgroup float* part) {
  const uint split = width / h_v;
  const uint h = t % h_v;
  const uint j = t / h_v;
  const bool live = j < split;
  const size_t slab = (size_t)d * d * h_v;
  for (uint p = 0; p < steps; p++) {
    const device float* s = s_pool + ((size_t)((phase + p) % n_phases) * n_slots + slot) * slab;
    device float* s_out = s_pool_out + ((size_t)((phase + p + 1) % n_phases) * n_slots + slot) * slab;
    const device float* hrow = heads + (size_t)p * 3 * h_v * d;
    const float eg = gdn_gate_eg(alpha[(size_t)p * h_v + h], dt_bias[h], ssm_a[h]);
    const float bt = 1.0f / (1.0f + metal::exp(-beta[(size_t)p * h_v + h]));
    GdnColumn<R>::run(hrow + (size_t)h * d, hrow + (size_t)(h_v + h) * d,
                      hrow + (size_t)(2 * h_v + h) * d, eg, bt, s, s_out, d, h_v, dv, h, j,
                      split, live, t, part, o + (size_t)p * h_v * d);
  }
}

// The rows and multi-position kernels in form `R` (0 = the loop form, whose
// names carry no `_r<R>`).
#define GDN_STEP_ROWS_SEQ(R, SUFFIX)                                                           \
  kernel void gdn_step_rows##SUFFIX##_f32(                                                     \
      const device float* heads [[buffer(0)]], const device float* alpha [[buffer(1)]],        \
      const device float* beta [[buffer(2)]], const device float* dt_bias [[buffer(3)]],       \
      const device float* ssm_a [[buffer(4)]], const device float* s_pool [[buffer(5)]],       \
      const device uint* slots [[buffer(6)]], const device uint* phases [[buffer(7)]],         \
      device float* o [[buffer(8)]], constant uint& d [[buffer(9)]],                           \
      constant uint& h_v [[buffer(10)]], constant uint& n_slots [[buffer(11)]],                \
      constant uint& n_phases [[buffer(12)]], device float* s_pool_out [[buffer(13)]],         \
      uint2 gid [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], \
      uint2 tptg [[threads_per_threadgroup]]) {                                                \
    threadgroup float part[1024];                                                              \
    gdn_step_rows_body<R>(heads, alpha, beta, dt_bias, ssm_a, s_pool, slots, phases, o, d, h_v, \
                          n_slots, n_phases, s_pool_out, gid, tid, tptg, part);                \
  }                                                                                            \
  kernel void gdn_step_seq##SUFFIX##_f32(                                                      \
      const device float* heads [[buffer(0)]], const device float* alpha [[buffer(1)]],        \
      const device float* beta [[buffer(2)]], const device float* dt_bias [[buffer(3)]],       \
      const device float* ssm_a [[buffer(4)]], const device float* s_pool [[buffer(5)]],       \
      device float* o [[buffer(6)]], constant uint& d [[buffer(7)]],                           \
      constant uint& h_v [[buffer(8)]], constant uint& n_slots [[buffer(9)]],                  \
      constant uint& n_phases [[buffer(10)]], constant uint& slot [[buffer(11)]],              \
      constant uint& phase [[buffer(12)]], constant uint& steps [[buffer(13)]],                \
      device float* s_pool_out [[buffer(14)]], uint dv [[threadgroup_position_in_grid]],       \
      uint t [[thread_position_in_threadgroup]], uint width [[threads_per_threadgroup]]) {     \
    threadgroup float part[1024];                                                              \
    gdn_step_seq_body<R>(heads, alpha, beta, dt_bias, ssm_a, s_pool, o, d, h_v, n_slots,       \
                         n_phases, slot, phase, steps, s_pool_out, dv, t, width, part);        \
  }
GDN_STEP_ROWS_SEQ(0, )
GDN_STEP_ROWS_SEQ(16, _r16)
GDN_STEP_ROWS_SEQ(32, _r32)

// `gdn_gated_norm_f32` over `B` rows: `o [B, h_v, d]`, `z [B, h_v·d]` →
// `[B, h_v·d]`. Grid: h_v × B threadgroups.
template <bool SILU>
METAL_FUNC void gdn_gated_norm_rows_body(
    const device float* o,
    const device float* gamma,
    const device float* z,
    device float* out,
    uint d,
    uint h_v,
    float eps,
    uint2 gid,
    uint2 tid,
    uint2 tptg,
    uint simd_lane,
    uint simd_id,
    threadgroup float* partial) {
  const uint h = gid.x;
  const uint b = gid.y;
  const uint i = tid.x;
  const uint width = tptg.x;
  const size_t idx = ((size_t)b * h_v + h) * d + i;
  const float x = (i < d) ? o[idx] : 0.0f;
  float ss = simd_sum(x * x);
  if (simd_lane == 0) partial[simd_id] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (width + 31) / 32;
  float total = 0.0f;
  for (uint sidx = 0; sidx < n_simd; sidx++) total += partial[sidx];
  if (i < d) {
    const float scale = 1.0f / metal::sqrt(total / (float)d + eps);
    out[idx] = x * scale * gamma[i] * gdn_out_gate<SILU>(z[idx]);
  }
}

#define GDN_GATED_NORM_ROWS(NAME, SILU)                                             \
  kernel void NAME(                                                                 \
      const device float* o [[buffer(0)]],                                          \
      const device float* gamma [[buffer(1)]],                                      \
      const device float* z [[buffer(2)]],                                          \
      device float* out [[buffer(3)]],                                              \
      constant uint& d [[buffer(4)]],                                               \
      constant uint& h_v [[buffer(5)]],                                             \
      constant float& eps [[buffer(6)]],                                            \
      uint2 gid [[threadgroup_position_in_grid]],                                   \
      uint2 tid [[thread_position_in_threadgroup]],                                 \
      uint2 tptg [[threads_per_threadgroup]],                                       \
      uint simd_lane [[thread_index_in_simdgroup]],                                 \
      uint simd_id [[simdgroup_index_in_threadgroup]]) {                            \
    threadgroup float partial[32];                                                  \
    gdn_gated_norm_rows_body<SILU>(o, gamma, z, out, d, h_v, eps, gid, tid, tptg,   \
                                   simd_lane, simd_id, partial);                    \
  }
GDN_GATED_NORM_ROWS(gdn_gated_norm_rows_f32, false)
GDN_GATED_NORM_ROWS(gdn_gated_norm_rows_silu_f32, true)

// ---- The chunk scan: the inter-chunk recurrence as one launch --------------------
//
// Walks the chunks for one head per threadgroup, the simdgroups splitting
// each product into 8 × 8 tiles straight from device memory (`S` and `V` are
// L2-resident):
//   V  = U_c − W_c · S                    [cs, d]
//   O_c = Qg_c · S + KQ_c · V             [cs, d]
//   S  = g_c · S + Kg_cᵀ · V              [d, d]   (in place; the tile a
//                                          simdgroup writes is the one it read)
// `d` and `cs` multiples of 8; `S` starts as `s_in` and ends in `s_out`.
// Every product is column-separable in `d`, so a head is split over
// `splits` threadgroups, each owning a slice of `S`'s / `V`'s / `O`'s
// columns (48 heads alone leave most of the machine idle).
kernel void gdn_chunk_scan_f32(
    const device float* w [[buffer(0)]],
    const device float* u [[buffer(1)]],
    const device float* qg [[buffer(2)]],
    const device float* kq [[buffer(3)]],
    const device float* kg [[buffer(4)]],
    const device float* gl [[buffer(5)]],
    const device float* s_in [[buffer(6)]],
    device float* o [[buffer(7)]],
    device float* s_out [[buffer(8)]],
    device float* vn [[buffer(9)]],
    constant uint& nc [[buffer(10)]],
    constant uint& cs [[buffer(11)]],
    constant uint& d [[buffer(12)]],
    constant uint& splits [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float diag[64];
  const uint h = gid / splits;
  const uint part = gid % splits;
  device float* S = s_out + (size_t)h * d * d;
  device float* V = vn + (size_t)h * cs * d;
  const uint rt = cs / 8;
  const uint dt = d / 8;
  // this threadgroup's column tiles [c0, c1)
  const uint c0 = (dt * part) / splits;
  const uint c1 = (dt * (part + 1)) / splits;
  const uint nct = c1 - c0;
  for (uint i = tid; i < d * (c1 - c0) * 8; i += width) {
    const uint row = i / ((c1 - c0) * 8);
    const uint col = c0 * 8 + i % ((c1 - c0) * 8);
    S[row * d + col] = s_in[(size_t)h * d * d + row * d + col];
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint c = 0; c < nc; c++) {
    const size_t cd = ((size_t)h * nc + c) * cs * d;
    const device float* Wc = w + cd;
    const device float* Uc = u + cd;
    const device float* Qc = qg + cd;
    const device float* Kgc = kg + cd;
    const device float* KQc = kq + ((size_t)h * nc + c) * cs * cs;
    device float* Oc = o + ((size_t)h * nc + c) * cs * d;
    const float g = gl[h * nc + c];
    if (tid < 64) {
      diag[tid] = (tid % 9 == 0) ? g : 0.0f;
    }
    // V = W_c · S   (this slice's columns; the K runs over all of S's rows,
    // whose columns this threadgroup does not own — S's other columns are
    // read, never written, by the K loops: only the owned slice is written)
    for (uint t = sg; t < rt * nct; t += nsg) {
      const uint r = t / nct;
      const uint cc = c0 + t % nct;
      simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
      for (uint k = 0; k < dt; k++) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, Wc + (size_t)r * 8 * d + k * 8, d);
        simdgroup_load(b, S + (size_t)k * 8 * d + cc * 8, d);
        simdgroup_multiply_accumulate(acc, a, b, acc);
      }
      simdgroup_store(acc, V + (size_t)r * 8 * d + cc * 8, d);
    }
    threadgroup_barrier(mem_flags::mem_device);
    // V = U_c − V (the owned columns)
    for (uint i = tid; i < cs * nct * 8; i += width) {
      const uint row = i / (nct * 8);
      const uint col = c0 * 8 + i % (nct * 8);
      V[row * d + col] = Uc[row * d + col] - V[row * d + col];
    }
    threadgroup_barrier(mem_flags::mem_device);
    // O_c = Qg_c · S + KQ_c · V
    for (uint t = sg; t < rt * nct; t += nsg) {
      const uint r = t / nct;
      const uint cc = c0 + t % nct;
      simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
      for (uint k = 0; k < dt; k++) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, Qc + (size_t)r * 8 * d + k * 8, d);
        simdgroup_load(b, S + (size_t)k * 8 * d + cc * 8, d);
        simdgroup_multiply_accumulate(acc, a, b, acc);
      }
      for (uint k = 0; k < rt; k++) {
        simdgroup_float8x8 a, b;
        simdgroup_load(a, KQc + (size_t)r * 8 * cs + k * 8, cs);
        simdgroup_load(b, V + (size_t)k * 8 * d + cc * 8, d);
        simdgroup_multiply_accumulate(acc, a, b, acc);
      }
      simdgroup_store(acc, Oc + (size_t)r * 8 * d + cc * 8, d);
    }
    threadgroup_barrier(mem_flags::mem_device);
    // S = g · S + Kg_cᵀ · V
    simdgroup_float8x8 G;
    simdgroup_load(G, diag, 8);
    for (uint t = sg; t < dt * nct; t += nsg) {
      const uint r = t / nct;
      const uint cc = c0 + t % nct;
      simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
      for (uint k = 0; k < rt; k++) {
        simdgroup_float8x8 a, b;
        // Kg_cᵀ[r, k] = Kg_c[k, r]: the block at rows k·8.., columns r·8.., transposed.
        simdgroup_load(a, Kgc + (size_t)k * 8 * d + r * 8, d, ulong2(0, 0), true);
        simdgroup_load(b, V + (size_t)k * 8 * d + cc * 8, d);
        simdgroup_multiply_accumulate(acc, a, b, acc);
      }
      simdgroup_float8x8 sold;
      simdgroup_load(sold, S + (size_t)r * 8 * d + cc * 8, d);
      simdgroup_multiply_accumulate(acc, sold, G, acc);
      simdgroup_store(acc, S + (size_t)r * 8 * d + cc * 8, d);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
