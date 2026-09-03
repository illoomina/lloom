// FlashDecoding: attention split over independent contiguous KV chunks, with
// an index-ordered combine (`DESIGN.md` §10.4, §17 Phase 5 item 16, issue #116).
//
// # Why this is a new kernel and not a rewiring of `sdpa_vector_2pass`
//
// `sdpa_vector_2pass` is dispatched today at `k_seq >= 1024` (`ops.rs`) and is
// easy to mistake for FlashDecoding already shipping. It is not, on three
// counts (§10.3b), and the second is disqualifying:
//
// | | `sdpa_vector_2pass` | what §10.4 specifies |
// |---|---|---|
// | chunk count | **fixed at 32**, whatever `kv_len` is | `ceil(kv_len / chunk_size)`, per step |
// | chunk extent | **strided interleave** -- block `b` reads every 32nd group of 8 keys | **contiguous** -- chunk `c` is `[c*S, (c+1)*S)` |
// | partial buffers | allocated per call | §9.1a's planned scratch class |
//
// A 2pass block does not own a contiguous KV range, so it does not correspond
// to a page, cannot be resolved through a chunk table, and cannot be the unit
// §10.4 makes equal to the page. Its merge is index-ordered **by accident of
// the fixed count** rather than by design, and a variable-count combine
// inherits no such accident -- which is why the combine below asserts its order
// rather than relying on one.
//
// The existing 2pass is a legitimate long-`kv_len` optimisation and is left
// alone.
//
// # The two decisions this file takes, both of which #116 owns
//
// **1. The token stride is a PARAMETER, not a `constexpr`.** `sdpa_vector`
// steps keys with `constexpr int stride = BN * D`, compile-time, with no
// token-stride parameter -- which is why #200 could refute a dim-outer KV order
// there and could not *time* it: there is no second arm to build. §8.3 item 1
// already requires `kv_len` from a buffer rather than a compile-time constant;
// §9.1d makes the same argument for the token stride and records that **#116's
// kernel is where it is decided**. The layout shipped is `[B, H, T, D]` --
// dim-innermost, because `head_dim x dtype` is exactly one 128 B cache line at
// LFM2's `64 x f16` -- and that is a property of *this* head_dim, reopening at
// a different one and under §10.7's int8 KV. Parameterising the stride is what
// keeps a second arm buildable when it does.
//
// **2. `chunk_size = k * page_size`, with `k` a parameter.** §10.4 fixes
// *"page size = FlashDecoding chunk size"* **by fiat**. A page is an
// ALLOCATION unit and a tile is a COMPUTATION unit, optimised against disjoint
// cost functions -- a page wants to be small (last-page waste, sharing
// granularity) and a tile wants to be large enough to fill the machine (§9.1d).
// The general form is `chunk_size = k * page_size` for integer `k >= 1`, which
// this file carries and ships at `k = 1`. At `k > 1` a chunk spans `k` pages
// and the kernel resolves `k` table entries **at chunk entry**, still hoisted
// out of the key loop, so §10.3d's `1 : 8192 bytes` indirection ratio is
// unchanged.
//
// # The determinism hazard, and why the chunk table is here at B=1
//
// §10.4 calls a completion-ordered merge *"the single most likely place for
// nondeterminism to enter the design"*. §10.3h/§3.7f add a **second** wrong
// order that is not completion order: the combine must **index** its chunk
// table and never **walk** it. Both visit every chunk; only the first is
// bit-stable, because a table's *contents* depend on allocation history, which
// under B>1 depends on what other sequences did -- so a sequence's logits would
// depend on its batch neighbours, violating §2.3.3 #7.
//
// **A B=1 gate structurally cannot detect the difference**, because at B=1 the
// table is `chunk_table[c] == c` and the two orders coincide. That is why the
// table is a real binding here rather than something paging adds later: with it
// present, a **permuted** table (#197's `page_table[3] = 6`) is an arm the
// tests can build, and it is the only arm that separates an index from a walk.
// A fixture that does not permute its table is testing the identity function
// (§9.2c's alignment lesson, in a second quantity).

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// ============================================================================
// Parameters
// ============================================================================

/// The partial pass's scalars.
///
/// One struct taken as `device const*` and copied into locals at kernel entry,
/// per §11.3e: one structure load into registers rather than a dereference per
/// use. That rule is load-bearing here -- `n_keys` bounds the key loop and
/// `softcapping` is tested inside it.
///
/// Field order is chosen so the two 8-byte strides sit together and the struct
/// needs no interior padding beyond the tail: 8 x int32 then 4 x size_t then
/// 2 x float. `flash_params_layout` ships the real offsets across the boundary
/// rather than either side asserting its own (§11.3d).
///
/// **The int32 count was 6 and is 8 as of #234**, which is why the sentence
/// above is a statement about the current field list rather than a constant: an
/// even count is what keeps the `size_t` block naturally aligned, and adding
/// `chunk_capacity` alone would have made it 7 and inserted 4 bytes of padding
/// the layout check would then have had to explain.
struct FlashPartialParams {
  /// Query heads per KV head. 4 for LFM2 (32 q heads, 8 kv heads, §5.2).
  int gqa_factor;
  /// Live KV length. The loop bound, and the reason a chunk count is a per-step
  /// quantity rather than a compile-time one (§8.3 item 1).
  int n_keys;
  /// KV tokens per chunk. `k * page_size`; see the header note.
  int chunk_size;
  /// Chunks this step splits `n_keys` into: `ceil(n_keys / chunk_size)`.
  ///
  /// **The DISPATCH DEPTH, and not the stride** -- see `chunk_capacity`. Under
  /// `Sizing::Reserve` the region holds more chunks than this step computes,
  /// and the partial pass computes only the live ones: a chunk the pass never
  /// dispatches is a chunk it never writes, so the combine must not read it
  /// (which it does not -- its own loop bound is this same live count).
  int n_chunks;
  /// Chunks the region is SIZED for. The partial write stride.
  ///
  /// # Why sizing needs a second number rather than a bigger first one
  ///
  /// `ScratchSizing` (§9.1a) chooses how many chunks a region reserves, and
  /// `Reserve` and `Bucket` both reserve more than the step needs. The stride
  /// between two heads' partials is a property of **how the region was laid
  /// out**, so it is the reserved count; the number of chunks written and
  /// merged is a property of **this step**, so it is `n_chunks`. Conflating
  /// them under a reserving policy makes head `h`'s partials land where head
  /// `h-1`'s were expected and the answer moves — a plausible wrong answer,
  /// which §3.5 says nothing reports.
  ///
  /// `FlashCombineParams` has carried exactly this separation since #116 and
  /// every caller passed `n_chunks` for both, because nothing selected a
  /// reserving policy. **This is the same field on the pass that lays the
  /// region out**, which is the half that was missing.
  ///
  /// Equal to `n_chunks` under `Sizing::Grow`, which is what shipped.
  int chunk_capacity;
  /// Pages per chunk -- the `k` of `chunk_size = k * page_size`. **1 today.**
  int pages_per_chunk;
  /// KV tokens per page. Equals `chunk_size` at `k = 1`.
  int page_size;
  /// Padding to keep the `size_t` block naturally aligned and the struct's
  /// layout stated rather than inferred. Written by the Rust side as 0 and read
  /// by nothing; `flash_params_layout` asserts the offsets either way, so this
  /// is a declaration of intent rather than a load-bearing field.
  int _pad;
  /// Distance between two KV heads, in elements. `k_stride[1]` -- the same
  /// quantity `sdpa_vector` takes, and it is the RESERVED capacity where
  /// `n_keys` is the LIVE length. Those being different numbers is what lets a
  /// pre-allocated cache be read without a copy (§10.3b, §6.2b).
  size_t k_head_stride;
  size_t v_head_stride;
  /// Distance between two adjacent KV tokens, in elements.
  ///
  /// **This is the field #200 could not vary and §9.1d asks for.** `head_dim`
  /// for the shipped `[B, H, T, D]` order. A different dimension order is a
  /// different value here rather than a different kernel.
  size_t k_token_stride;
  size_t v_token_stride;
  float scale;
  float softcapping;
};

/// The combine pass's scalars.
struct FlashCombineParams {
  /// Chunks to merge. The combine iterates `0..n_chunks` and **indexes** the
  /// table by that -- never walks the table (§10.3h).
  int n_chunks;
  /// Chunks each region is SIZED for, which under `Sizing::Reserve` exceeds
  /// `n_chunks` (§9.1a).
  ///
  /// The partial stride must be the reserved count, because that is how the
  /// partial pass laid the region out; the merge must run to the **live** count,
  /// because merging over the reservation folds in uninitialised memory --
  /// a silent wrong answer that no size check catches (§10.4, §10.3d).
  int chunk_capacity;
};

// ============================================================================
// Pass 1 -- partials over contiguous chunks
// ============================================================================

/// One threadgroup computes one (head, chunk) partial: its own `m`, `l` and
/// accumulator over a **contiguous** KV range.
///
/// Chunks are independent -- each computes a local `(m, l, acc)` and online
/// softmax merges associatively -- so they need **no fences between them**
/// (§9.4, §10.3h). What they do need is a fence after the KV append and one
/// before the combine, which is two per attention layer per step and is the
/// caller's obligation.
///
/// `T` is the storage type and accumulation is in `float` regardless: §8.1
/// principle 4, and §9.1a records that this class cannot be shrunk to f16
/// because the combine merges what the accumulation bought.
template <typename T, int D>
[[kernel]] void flash_decoding_partial(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    // Partials, laid out as §9.1a's PLANES layout: the accumulator plane is
    // [head][chunk][dim] and the statistics plane is [head][chunk][2].
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    // The chunk table: `chunk_table[c]` is the FIRST PAGE of chunk `c`.
    // At B=1 contiguous it is the identity, and it is bound anyway -- see the
    // header note on why a B=1 gate cannot otherwise discriminate.
    const device uint* chunk_table [[buffer(6)]],
    const device FlashPartialParams* params [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // One structure load into registers (§11.3e). Not `params->field` per use:
  // `device` is general memory where `constant` is a cached read-only space,
  // and these are read inside the key loop.
  const FlashPartialParams p = *params;
  const int gqa_factor = p.gqa_factor;
  const int n_keys = p.n_keys;
  const int chunk_size = p.chunk_size;
  const int pages_per_chunk = p.pages_per_chunk;
  const int page_size = p.page_size;
  const size_t k_head_stride = p.k_head_stride;
  const size_t v_head_stride = p.v_head_stride;
  const size_t k_token_stride = p.k_token_stride;
  const size_t v_token_stride = p.v_token_stride;
  const float scale = p.scale;
  const float softcapping = p.softcapping;

  const int head_idx = tid.y;
  const int chunk_idx = tid.z;
  const int kv_head_idx = head_idx / gqa_factor;

  // The chunk's KV range, resolved ONCE at chunk entry rather than per key
  // (§10.3d). At `k = 1` this is one table read; at `k > 1` it is `k` of them,
  // and either way it is hoisted out of the loop below, so the indirection is
  // 1 : 8192 bytes of key data at page size 256 rather than a dependent load in
  // the innermost loop of a bandwidth-bound kernel.
  //
  // `chunk_table[c]` names a PAGE, and a chunk spans `pages_per_chunk` of them.
  // Under a contiguous cache the pages of one chunk are consecutive, so the
  // first page determines the range; a paged cache would resolve each page
  // separately here, which is the loop this shape leaves room for.
  const uint first_page = chunk_table[chunk_idx];
  const int chunk_start = (int)first_page * page_size;
  int chunk_end = chunk_start + pages_per_chunk * page_size;
  if (chunk_end > n_keys) {
    chunk_end = n_keys;
  }
  // The last chunk is partial, and the clamp above is what walks the LIVE
  // count rather than the reserved one -- merging over a full last chunk folds
  // in uninitialised memory, which §9.1a records as a silent wrong answer that
  // no size check catches.
  //
  // The extent is `pages_per_chunk * page_size` and NOT `chunk_size`, though
  // the two are equal by construction. Deriving it from the two fields the
  // table is indexed with keeps one source for the range: `chunk_size` is
  // carried for the descriptor's completeness and a kernel that read it here
  // could disagree with the table if a caller ever set them inconsistently.
  (void)chunk_size;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  queries += head_idx * D + simd_lid * elem_per_thread;

  // The token step is `k_token_stride` -- a PARAMETER. This is the line #200
  // could not vary and §9.1d asks for; see the header note.
  const device T* k_base =
      keys + kv_head_idx * k_head_stride + simd_lid * elem_per_thread;
  const device T* v_base =
      values + kv_head_idx * v_head_stride + simd_lid * elem_per_thread;

  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // Walk this chunk's keys only. The trip count is uniform across the
  // simdgroup, so §3.3 makes the bound a compare and a branch rather than a
  // divergence -- which is the CUDA-derived intuition that section corrects,
  // and it is why a dynamic `kv_len` needs no bucketing (§10.4).
  for (int t = chunk_start + (int)simd_gid; t < chunk_end; t += BN) {
    const device T* kp = k_base + (size_t)t * k_token_stride;
    const device T* vp = v_base + (size_t)t * v_token_stride;

    for (int j = 0; j < elem_per_thread; j++) {
      k[j] = kp[j];
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
      o[j] = o[j] * factor + exp_score * vp[j];
    }
  }

  // Combine the simdgroups' partial states within this threadgroup. A fixed
  // tree, per §2.3.3 #3 -- `simd_max` and `simd_sum` are order-stable and are
  // not to be replaced with an order-dependent variant.
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // An EMPTY chunk contributes nothing and must not poison the merge. It arises
  // when `n_keys` does not fill the last chunk's first key -- reachable
  // whenever `chunk_start >= n_keys`, which a caller can produce by sizing for
  // more chunks than the step needs. `m = -INFINITY` with `l = 0` is the
  // identity of the online-softmax merge, and writing it is what makes a
  // reserved-but-unused chunk safe rather than a source of NaN.
  const bool empty = (chunk_start >= chunk_end);

  // The stride between two heads' partials is what the region was SIZED for,
  // which under `Sizing::Reserve` and `Sizing::Bucket` exceeds what this step
  // computes (§9.1a, #234). Using the live count here would make head `h`'s
  // partials land where head `h-1`'s were expected under any reserving policy
  // -- a plausible wrong answer, and §3.5 says nothing reports it.
  //
  // Equal to `p.n_chunks` under `Grow`, which is the arm that shipped and the
  // reason a single field served both until a policy could select otherwise.
  const int stride_chunks = p.chunk_capacity;
  device float* acc_out =
      partials + (size_t)head_idx * stride_chunks * D + (size_t)chunk_idx * D;

  if (simd_gid == 0) {
    sums[(size_t)head_idx * stride_chunks + chunk_idx] =
        empty ? 0.0f : sum_exp_score;
    maxs[(size_t)head_idx * stride_chunks + chunk_idx] =
        empty ? -INFINITY : new_max;
  }

  // Aggregate the per-thread accumulators and write the chunk's partial.
  //
  // The partial is stored WITHOUT dividing by `sum_exp_score` and without
  // rescaling to a global max -- both are the combine's job, because only it
  // sees every chunk. Storing a normalised partial would make the merge
  // non-associative and is the shape that forces a completion-order merge.
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U v = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) {
      acc_out[simd_gid * elem_per_thread + i] = empty ? 0.0f : v;
    }
  }
}

// ============================================================================
// Pass 2 -- the index-ordered combine
// ============================================================================

/// Merge the chunk partials into one output per head.
///
/// # The order is the whole point of this kernel
///
/// The loop is `for (c = 0; c < n_chunks; ++c)` -- over the **chunk index**.
/// The chunk table is *indexed* by `c` where a chunk's KV range is needed; it
/// is never *walked*. §10.3h:
///
/// ```text
/// // CORRECT -- the loop is over chunk index; the table is a lookup.
/// for (uint c = 0; c < n_chunks; ++c) { merge(partial[c]); }
///
/// // WRONG -- same set, order now a property of the table's contents.
/// for (page in page_list) { merge(partial_for(page)); }
/// ```
///
/// Both visit every chunk and only the first is bit-stable. At B=1 the table is
/// the identity and the two coincide, so **a B=1 determinism gate structurally
/// cannot tell them apart** -- which is why the tests build a permuted table
/// rather than trusting this comment.
///
/// A single thread per (head, dim-slice) walks the chunks in ascending order,
/// so the accumulation order is a property of the loop and not of the
/// scheduler. That is §2.3.3 #1 by construction: no float atomics, no
/// completion order, a fixed index sequence.
template <typename T, int D>
[[kernel]] void flash_decoding_combine(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device FlashCombineParams* params [[buffer(4)]],
    // The chunk indices this kernel actually walked, in the order it walked
    // them. **The merge order is asserted against this and not against the
    // output**, because §10.4 records that reversing the loop *"is caught by
    // that assertion and by nothing else"* -- the bit-equality tests stayed
    // green under the reversal, floating-point non-associativity happening not
    // to bite on that fixture. **Reproduced here: the first mutation run
    // reversed both combine loops and left all seven tests passing.**
    //
    // An ordering that cannot be observed cannot be asserted (§9.1a).
    device uint* walk_order [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  const FlashCombineParams p = *params;
  const int n_chunks = p.n_chunks;
  const int stride_chunks = p.chunk_capacity;

  const int head_idx = tid.y;

  const device float* head_partials =
      partials + (size_t)head_idx * stride_chunks * D;
  const device float* head_sums = sums + (size_t)head_idx * stride_chunks;
  const device float* head_maxs = maxs + (size_t)head_idx * stride_chunks;

  // Pass A: the global maximum, in ascending chunk order.
  //
  // `max` is associative AND commutative in floating point -- unlike `+` -- so
  // this pass's order does not affect the bits. It is written ascending anyway,
  // because a reader checking §2.3.3 #1 should not have to prove the exemption,
  // and because the next two passes genuinely depend on the order.
  U global_max = -INFINITY;
  for (int c = 0; c < n_chunks; ++c) {
    global_max = max(global_max, head_maxs[c]);
  }

  // Pass B: the denominator, summed in ASCENDING CHUNK INDEX ORDER.
  //
  // This one is order-dependent: float addition is not associative, so the
  // sequence `c = 0, 1, 2, ...` is what makes the result bit-stable. Reversing
  // it is a legal-looking change that produces different bits, which is
  // precisely the mutation §10.4 records the bit-equality tests failing to
  // catch on their own.
  U denom = 0;
  for (int c = 0; c < n_chunks; ++c) {
    // An empty chunk carries `m = -INFINITY, l = 0` and contributes exactly
    // zero: `exp(-inf - finite) = 0`. Guarded explicitly rather than relying on
    // that, because when EVERY chunk is empty `global_max` is `-INFINITY` and
    // `-inf - -inf` is NaN.
    if (head_sums[c] != 0.0f) {
      denom += head_sums[c] * fast::exp(head_maxs[c] - global_max);
    }
  }

  // Pass C: the accumulator, also in ascending chunk index order.
  thread U acc[elem_per_thread];
  for (int i = 0; i < elem_per_thread; i++) {
    acc[i] = 0;
  }

  const int lane_base = (int)simd_lid * elem_per_thread;
  int visited = 0;
  for (int c = 0; c < n_chunks; ++c) {
    // Recorded before the skip, so the log is the order the loop VISITS rather
    // than the order it accumulates: an empty chunk contributes nothing and
    // still occupies a position in the sequence. Logging after the skip would
    // make a reversed loop over a fixture with no empty chunks indistinguishable
    // from a forward one whose empties fell differently.
    if (simd_lid == 0 && simd_gid == 0) {
      walk_order[(size_t)head_idx * n_chunks + visited] = (uint)c;
    }
    visited += 1;
    if (head_sums[c] == 0.0f) {
      continue;
    }
    const U rescale = fast::exp(head_maxs[c] - global_max);
    const device float* chunk_acc = head_partials + (size_t)c * D;
    for (int i = 0; i < elem_per_thread; i++) {
      acc[i] += chunk_acc[lane_base + i] * rescale;
    }
  }

  const U inv_denom = denom > 0 ? 1.0f / denom : 0.0f;
  device T* head_out = out + (size_t)head_idx * D + lane_base;
  if (simd_gid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      head_out[i] = static_cast<T>(acc[i] * inv_denom);
    }
  }
}

// ============================================================================
// Layout, shipped across the boundary
// ============================================================================

// 72 as of #234, and it was 64 before: **eight** `int` fill 0..32, the four
// `size_t` land 8-aligned at 32..64, and the two `float` at 64..72 need no
// trailing pad only because 72 is already a multiple of the struct's 8-byte
// alignment. `_pad` is what makes the `int` count even and keeps that true —
// with seven ints the `size_t` block would have started at 28 and the compiler
// would have inserted four bytes here rather than the field stating them.
//
// **This literal was wrong on the first attempt and the compiler said so**,
// which is the argument for the cross-boundary check rather than against it: a
// `static_assert` catches the author's arithmetic, and
// `every_family_params_layout_matches_metal` catches the two *sides*
// disagreeing — a different failure, and the one #38 found live (§11.3d).
static_assert(sizeof(FlashPartialParams) == 72, "FlashPartialParams layout");
static_assert(alignof(FlashPartialParams) == 8, "FlashPartialParams alignment");
static_assert(sizeof(FlashCombineParams) == 8, "FlashCombineParams layout");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation (§11.3b).
// A `static_assert` proves only that one side agrees with itself; measuring the
// offsets the COMPILED kernel sees and comparing against Rust's `offset_of!` is
// what proves the two agree (§11.3d).
#define flash_offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void flash_params_layout(
    device uint* out,
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) { return; }
  FlashPartialParams probe_FlashPartialParams;
  FlashCombineParams probe_FlashCombineParams;

  out[0] = sizeof(FlashPartialParams);
  out[1] = flash_offsetof_rt(FlashPartialParams, gqa_factor);
  out[2] = flash_offsetof_rt(FlashPartialParams, n_keys);
  out[3] = flash_offsetof_rt(FlashPartialParams, chunk_size);
  out[4] = flash_offsetof_rt(FlashPartialParams, n_chunks);
  out[5] = flash_offsetof_rt(FlashPartialParams, chunk_capacity);
  out[6] = flash_offsetof_rt(FlashPartialParams, pages_per_chunk);
  out[7] = flash_offsetof_rt(FlashPartialParams, page_size);
  out[8] = flash_offsetof_rt(FlashPartialParams, k_head_stride);
  out[9] = flash_offsetof_rt(FlashPartialParams, v_head_stride);
  out[10] = flash_offsetof_rt(FlashPartialParams, k_token_stride);
  out[11] = flash_offsetof_rt(FlashPartialParams, v_token_stride);
  out[12] = flash_offsetof_rt(FlashPartialParams, scale);
  out[13] = flash_offsetof_rt(FlashPartialParams, softcapping);

  out[14] = sizeof(FlashCombineParams);
  out[15] = flash_offsetof_rt(FlashCombineParams, n_chunks);
  out[16] = flash_offsetof_rt(FlashCombineParams, chunk_capacity);
}

// ============================================================================
// Instantiations
// ============================================================================
//
// Named per §7.4's convention. The head dimensions match `call_sdpa_vector`'s
// `match` arms rather than the file's own instantiation list, because #103
// found those two disagreeing in `scaled_dot_product_attention.metal` -- eight
// instantiated, six reachable -- and a registry that declares an unreachable
// name is the inverse of §8.1b's absent-variant class. `FlashKernel::ALL`
// resolves every name below against the compiled library.

// clang-format off
#define instantiate_flash_decoding(type, head_dim)                            \
  template [[host_name("flash_decoding_partial_" #type "_" #head_dim)]]       \
  [[kernel]] void flash_decoding_partial<type, head_dim>(                     \
      const device type* queries [[buffer(0)]],                               \
      const device type* keys [[buffer(1)]],                                  \
      const device type* values [[buffer(2)]],                                \
      device float* partials [[buffer(3)]],                                   \
      device float* sums [[buffer(4)]],                                       \
      device float* maxs [[buffer(5)]],                                       \
      const device uint* chunk_table [[buffer(6)]],                           \
      const device FlashPartialParams* params [[buffer(7)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);                           \
  template [[host_name("flash_decoding_combine_" #type "_" #head_dim)]]       \
  [[kernel]] void flash_decoding_combine<type, head_dim>(                     \
      const device float* partials [[buffer(0)]],                             \
      const device float* sums [[buffer(1)]],                                 \
      const device float* maxs [[buffer(2)]],                                 \
      device type* out [[buffer(3)]],                                         \
      const device FlashCombineParams* params [[buffer(4)]],                  \
      device uint* walk_order [[buffer(5)]],                                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_flash_decoding_heads(type) \
  instantiate_flash_decoding(type, 32)         \
  instantiate_flash_decoding(type, 64)         \
  instantiate_flash_decoding(type, 96)         \
  instantiate_flash_decoding(type, 128)

instantiate_flash_decoding_heads(float)
instantiate_flash_decoding_heads(half)
// clang-format on

// **`bfloat` is deliberately not instantiated, and that is still a scope
// statement rather than an omission -- but every reason this comment used to
// give for it is FALSE.** Measured 2026-08-30 (#307, `DESIGN.md` §3.9).
//
// It read: *"`scaled_dot_product_attention.metal` reaches it through a ~500-line
// `_MLX_BFloat16` shim it carries for the pre-`__HAVE_BFLOAT__` case;
// duplicating that here would be the largest part of this file for a dtype
// nothing dispatches. LFM2 ships BF16 on disk and decode runs F16 --
// `lfm2-determinism/main.rs` converts at load, because "Metal's bf16 kernel
// coverage is patchy and unsupported ops fall back to the CPU silently"
// (§9.1b)."*
//
// Three corrections, and the last one is why this is worth keeping rather than
// deleting:
//
//   1. **`__HAVE_BFLOAT__` IS defined on this machine**, so that shim is the
//      `#else` branch and is inert. Nothing would be duplicated.
//   2. **"a dtype nothing dispatches" is true and circular.** Nothing
//      dispatches bfloat here *because this file does not instantiate it*; the
//      rest of the decode path dispatches bf16 at 12 of 12 families.
//   3. **The quoted claim is false.** It is a comment in a harness, quoted into
//      §9.1b as a fact and made load-bearing for three decisions -- including
//      this one. bf16 decode runs: `lfm2-smoke` PASSes at `--dtype bf16` on all
//      three `--attn` arms with byte-identical text, and no dispatch falls back
//      to the CPU.
//
// **The disposition is unchanged and its grounds are new.** §10.4b measures this
// arm **+6.3 % slower** than `Sdpa` at `kv_len` 16 034, so instantiating
// `bfloat` would be built-and-unused (§15.2 #11). `half` is the measured arm and
// `float` is here because CPU-parity fixtures use it. Adding `bfloat` is **three
// lines** -- the `#include` is not needed -- when a caller wants it, and a
// caller wanting it owes the timing first.
