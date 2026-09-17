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
/// Field order is chosen so the 8-byte strides sit together and the struct
/// needs no interior padding beyond the tail: 10 x int32 then 6 x size_t then
/// 2 x float. `flash_params_layout` ships the real offsets across the boundary
/// rather than either side asserting its own (§11.3d).
///
/// The int32 count is kept even so the `size_t` block stays naturally aligned;
/// `n_keys_stride` sits in what was the padding slot.
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
  /// Query heads per SEQUENCE. The grid's `y` axis is `(batch x q head)`
  /// flattened, so `head_idx / heads_per_seq` is the sequence and
  /// `head_idx % heads_per_seq` the head within it. Splitting the flat index
  /// here rather than folding the batch into `kv_head_idx` is what lets B
  /// sequences read ONE pool through B different tables (§10.3c).
  int heads_per_seq;
  /// Entries between two sequences' chunk tables. 0 when every sequence reads
  /// the same table (the contiguous cache, where the table is the identity).
  int table_stride;
  /// Entries between two sequences' live key counts in the `n_keys_per_seq`
  /// buffer: 0 when every sequence reads entry 0 (a lockstep batch, the
  /// contiguous cache), 1 when row `b` reads its own (`batched-decode-design.md`
  /// §12.2: a ragged batch through one dispatch). Took the `_pad` slot that
  /// kept the `int` count even, so the layout is unchanged.
  int n_keys_stride;
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
  /// Distance between two sequences' KV, in elements. `k_stride[0]` for a
  /// `[B, H_kv, T, D]` cache; **0 for a shared page pool**, where a sequence is
  /// reached through its table rather than through a stride.
  size_t k_seq_stride;
  size_t v_seq_stride;
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
/// The shared attention core: one threadgroup computes online-softmax
/// attention over a **contiguous KV range** `[range_start, range_end)`, in
/// `float` regardless of the storage type `T` (§8.1 principle 4). Both output
/// tails below call this — `flash_decoding_partial` writes the *unnormalised*
/// state to scratch for a later combine, and `flash_decoding_single` divides by
/// the denominator and writes `out` directly. Factoring it here is what keeps
/// the two arms one arithmetic body: a divergence between the split path and the
/// single-chunk path would be a silent wrong answer on exactly the band the
/// single path is FOR (§3.5).
///
/// Returns, per lane, the accumulated `o[]` (NOT yet divided by the
/// denominator, NOT yet reduced across the simdgroup transpose), and by
/// reference the threadgroup-merged `sum_exp_score`, `new_max`, the cross-simd
/// `factor` the caller's transpose must apply, and `empty`. The caller owns the
/// transpose+write because the two tails write to different places (scratch vs
/// `out`) and in different dtypes.
///
/// The token step is `k_token_stride` — a PARAMETER, not a `constexpr`. This is
/// the line #200 could not vary and §9.1d asks for, and it is the reason this
/// family exists separately from `sdpa_vector` (whose stride is compile-time):
/// a strided or `narrow`ed KV cache, and §10.7's int8 KV, are a different value
/// here rather than a different kernel. The single-chunk tail below inherits
/// that capability, which is why it is NOT a rewiring to `sdpa_vector`.
template <typename T, int D, int BN, int BD, int elem_per_thread>
inline void flash_decoding_core(
    const device T* queries,
    const device T* keys,
    const device T* values,
    const FlashPartialParams p,
    int range_start,
    int range_end,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    thread float (&o)[elem_per_thread],
    thread float& out_sum_exp_score,
    thread float& out_new_max,
    thread float& out_factor,
    thread bool& out_empty,
    uint head_idx,
    uint simd_gid,
    uint simd_lid) {
  typedef float U;

  const int gqa_factor = p.gqa_factor;
  const size_t k_head_stride = p.k_head_stride;
  const size_t v_head_stride = p.v_head_stride;
  const size_t k_token_stride = p.k_token_stride;
  const size_t v_token_stride = p.v_token_stride;
  const float scale = p.scale;
  const float softcapping = p.softcapping;

  // The flat grid index splits into (sequence, head-within-sequence), and only
  // the second half is divided by the GQA factor. Folding the batch into
  // `kv_head_idx` is address-identical for a contiguous `[B, H_kv, T, D]` cache
  // and wrong for a shared pool, where a sequence has no stride and is reached
  // through its own table (§10.3c).
  const int seq_idx = (int)head_idx / p.heads_per_seq;
  const int kv_head_idx = ((int)head_idx % p.heads_per_seq) / gqa_factor;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];

  queries += head_idx * D + simd_lid * elem_per_thread;

  const device T* k_base = keys + (size_t)seq_idx * p.k_seq_stride +
                           kv_head_idx * k_head_stride + simd_lid * elem_per_thread;
  const device T* v_base = values + (size_t)seq_idx * p.v_seq_stride +
                           kv_head_idx * v_head_stride + simd_lid * elem_per_thread;

  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // Walk this range's keys only. The trip count is uniform across the
  // simdgroup, so §3.3 makes the bound a compare and a branch rather than a
  // divergence -- which is the CUDA-derived intuition that section corrects,
  // and it is why a dynamic `kv_len` needs no bucketing (§10.4).
  for (int t = range_start + (int)simd_gid; t < range_end; t += BN) {
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

  // Lanes past `BN` (a narrow threadgroup, `BN = 8`) carry the merge's
  // identity; at `BN = 32` the selects are identities and the arithmetic is
  // the same instructions as before them.
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -INFINITY;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(((simd_lid < BN) ? sum_exp_scores[simd_lid] : 0.0f) * factor);

  // An EMPTY range contributes nothing and must not poison the merge. It arises
  // when `n_keys` does not fill the range's first key -- reachable whenever
  // `range_start >= range_end`, which a caller can produce by sizing for more
  // chunks than the step needs. `m = -INFINITY` with `l = 0` is the identity of
  // the online-softmax merge, and returning it is what makes a reserved-but-
  // unused chunk safe rather than a source of NaN.
  out_empty = (range_start >= range_end);
  out_sum_exp_score = sum_exp_score;
  out_new_max = new_max;
  out_factor = factor;
}

/// `T` is the storage type and accumulation is in `float` regardless: §8.1
/// principle 4, and §9.1a records that this class cannot be shrunk to f16
/// because the combine merges what the accumulation bought.
/// `BN` is the threadgroup's WIDTH in simdgroups — how many keys of a chunk
/// are walked at once, and so how many keys each simdgroup walks in
/// sequence: at the 256-key chunk, `BN = 32` (1024 threads) walks 8 keys per
/// simdgroup and reduces 32 partial states; `BN = 8` (256 threads, the
/// geometry of `sdpa_vector_2pass_1`, which walks 32 keys per simdgroup at
/// 8k) walks 32 and reduces 8. The two are separate instantiations
/// (`flash_decoding_partial` / `flash_decoding_partial8`) selected by the
/// host; their key order per simdgroup and their reduction trees differ, so
/// they are each bit-stable and NOT bit-identical to each other.
template <typename T, int D, int BN>
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
    const constant uint* chunk_table [[buffer(6)]],
    const constant FlashPartialParams* params [[buffer(7)]],
    // The live key count per sequence: `n_keys_per_seq[seq * n_keys_stride]`.
    // A ragged batch (§12.2) gives every row its own; a chunk past a row's
    // count is an EMPTY range and writes the merge's identity below, so the
    // dispatch depth is the longest row's and the combine needs no change.
    const constant int* n_keys_per_seq [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // One structure load into registers (§11.3e), from the `constant` address
  // space: the block, the chunk table and the live counts are read-only for
  // the dispatch and bound as buffers all the same (the host side is
  // unchanged) — `constant` is the cached read-only space, `device` general
  // memory, and at 8k keys a layer runs 768 threadgroups that each read them
  // once (#148: the same bits, one less general-memory load per threadgroup).
  const FlashPartialParams p = *params;
  const int pages_per_chunk = p.pages_per_chunk;
  const int page_size = p.page_size;

  const int head_idx = tid.y;
  const int chunk_idx = tid.z;
  const int n_keys = n_keys_per_seq[(size_t)(head_idx / p.heads_per_seq) * p.n_keys_stride];

  // The chunk's KV range, resolved ONCE at chunk entry rather than per key
  // (§10.3d). At `k = 1` this is one table read; at `k > 1` it is `k` of them,
  // and either way it is hoisted out of the loop, so the indirection is
  // 1 : 8192 bytes of key data at page size 256 rather than a dependent load in
  // the innermost loop of a bandwidth-bound kernel.
  //
  // `chunk_table[c]` names a PAGE, and a chunk spans `pages_per_chunk` of them.
  // The pages of one chunk are consecutive in the pool -- the allocator checks
  // it (`KvPagePool::chunk_pages_are_contiguous`) -- so the first page
  // determines the range. Each sequence has its own table at `seq * stride`.
  const int seq_idx = head_idx / p.heads_per_seq;
  const uint first_page = chunk_table[(size_t)seq_idx * p.table_stride + chunk_idx];
  // The live count of THIS chunk is a property of the logical token axis --
  // `n_keys - c * chunk_size`, clamped to a chunk -- and not of where the page
  // sits in the pool. A clamp keyed on the physical position assumes pages
  // appear in ascending token order, which a recycled pool does not honour;
  // it read a truncated range for any chunk placed above the tail page. The
  // partial last chunk walks the LIVE count rather than the reserved one:
  // merging over a full last page folds in uninitialised memory, which §9.1a
  // records as a silent wrong answer that no size check catches.
  const int chunk_size = pages_per_chunk * page_size;
  int n_live = n_keys - chunk_idx * chunk_size;
  if (n_live > chunk_size) {
    n_live = chunk_size;
  }
  const int chunk_start = (int)first_page * page_size;
  const int chunk_end = chunk_start + n_live;

  thread U o[elem_per_thread];
  U sum_exp_score, new_max, factor;
  bool empty;
  flash_decoding_core<T, D, BN, BD, elem_per_thread>(
      queries, keys, values, p, chunk_start, chunk_end, outputs, max_scores,
      sum_exp_scores, o, sum_exp_score, new_max, factor, empty, head_idx,
      simd_gid, simd_lid);

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
  if constexpr (BN == BD) {
    // Square: transpose through threadgroup memory so simdgroup `g` sums
    // lane `g`'s element across the BN simdgroups in one `simd_sum`.
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      U v = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_lid == 0) {
        acc_out[simd_gid * elem_per_thread + i] = empty ? 0.0f : v;
      }
    }
  } else {
    // Narrow: every simdgroup rescales its own accumulator to the
    // threadgroup max (its factor is lane `simd_gid`'s above, recomputed
    // from the same inputs) and simdgroup 0 sums the BN entries of each
    // lane in index order — `sdpa_vector_2pass_1`'s tail, the same
    // element layout as the square arm's.
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[simd_lid * BN + simd_gid] =
          o[i] * fast::exp(max_scores[simd_gid] - new_max);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        U v = outputs[simd_lid * BN];
        for (int j = 1; j < BN; j++) {
          v += outputs[simd_lid * BN + j];
        }
        acc_out[simd_lid * elem_per_thread + i] = empty ? 0.0f : v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

/// The single-chunk path: `n_chunks == 1` attention in ONE dispatch, writing
/// `out` directly with no scratch and no combine.
///
/// # Why this exists, and why it is not the combine made cheap
///
/// `flash_chunk_count` returns 1 whenever `kv_len <= chunk_size`, which is the
/// COMMON decode band — at the shipped `chunk_size = 256` a crown decode at
/// `kv_len ≈ 234` is one chunk. The split-K machinery buys nothing there: one
/// chunk is no parallelism to manufacture (§10.4's whole argument is that
/// *multiple* chunks fill a GPU wanting hundreds of threadgroups). But the
/// two-pass path still PAID for it — `flash_decoding_partial` writes the answer
/// to *scratch* device memory, then a second dispatch (`flash_decoding_combine`)
/// reads it back and writes `out`. That second dispatch plus the partials
/// round-trip through global memory is the **+0.989 % `n_chunks = 1` residual**
/// §10.4d/§10.4e measured and left in the INEXPRESSIVE class: it is not a
/// microarchitectural mystery, it is a mandatory extra dispatch and a
/// device-memory round-trip that `sdpa` does not pay. This kernel is the
/// design's own *"single-pass attention is FlashDecoding with n_chunks == 1"*
/// (`flash_decoding.rs`), finally taken.
///
/// # Bit-identity with the two-pass path at n_chunks == 1 (§2.3.5a)
///
/// The combine at one chunk computes `denom = sums[0] * exp(maxs[0] - maxs[0])`
/// = `sums[0]`, and `out[i] = partials[0][i] * exp(0) / denom` = `acc[i] /
/// sums[0]`. The `exp(0) = 1` and the `* 1` are exact in IEEE-754, so this
/// kernel's `o[i] * factor / sum_exp_score` — the SAME transpose+`factor` the
/// partial's scratch tail applies, then one division — reproduces the two-pass
/// output BIT for BIT. It shares `flash_decoding_core`, so the scores,
/// accumulation and reduction are literally the same instructions; only the
/// write differs (`out` in dtype `T` vs unnormalised scratch in `float`).
template <typename T, int D>
[[kernel]] void flash_decoding_single(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant FlashPartialParams* params [[buffer(4)]],
    // Read even though there is one chunk: its first page is wherever the
    // sequence's table says, which is page 0 only on a fresh pool. A
    // single-chunk path that assumed `[0, n_keys)` was correct for every
    // contiguous cache and silently wrong for a recycled one (§10.3h).
    const constant uint* chunk_table [[buffer(5)]],
    const constant int* n_keys_per_seq [[buffer(6)]],
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

  const FlashPartialParams p = *params;
  const int head_idx = tid.y;
  const int seq_idx = head_idx / p.heads_per_seq;
  const int n_keys = n_keys_per_seq[(size_t)seq_idx * p.n_keys_stride];

  // The whole live range, starting at the single chunk's first page. This
  // kernel is dispatched only when the caller computed `n_chunks == 1`, so
  // `n_keys` fits in that chunk and the range cannot be empty for a real step
  // (`n_keys >= 1` at decode).
  const int start = (int)chunk_table[(size_t)seq_idx * p.table_stride] * p.page_size;
  thread U o[elem_per_thread];
  U sum_exp_score, new_max, factor;
  bool empty;
  flash_decoding_core<T, D, BN, BD, elem_per_thread>(
      queries, keys, values, p, start, start + n_keys, outputs, max_scores,
      sum_exp_scores, o, sum_exp_score, new_max, factor, empty, head_idx,
      simd_gid, simd_lid);

  // Normalise and write `out` in dtype `T`. `inv_denom` guards an all-empty
  // range the same way the combine's does (`denom > 0`), so a step with no keys
  // writes zeros rather than NaN -- unreachable at decode, kept for parity with
  // the combine it replaces.
  const U inv_denom = (!empty && sum_exp_score > 0) ? 1.0f / sum_exp_score : 0.0f;
  device T* head_out = out + (size_t)head_idx * D;
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U v = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) {
      head_out[simd_gid * elem_per_thread + i] = static_cast<T>(v * inv_denom);
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
    const constant FlashCombineParams* params [[buffer(4)]],
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
// The KV write: one token's rows scattered to table-chosen slots
// ============================================================================

/// The scatter's scalars. 4 x int32 then 6 x size_t; no interior padding.
struct FlashScatterParams {
  /// Rows in `slots`: `n_seqs * seq_len`. Row `r` is sequence `r / seq_len`,
  /// token `r % seq_len` of this pass.
  int n_rows;
  int seq_len;
  int n_kv_heads;
  int head_dim;
  size_t src_seq_stride;
  size_t src_head_stride;
  size_t src_token_stride;
  /// 0 for a shared page pool: a sequence is reached through its slots.
  size_t dst_seq_stride;
  size_t dst_head_stride;
  size_t dst_token_stride;
};

/// Write `src[b, h, t, :]` to `dst` at token slot `slots[b * seq_len + t]`.
///
/// `copy2d` with a per-token destination -- vLLM's `slot_mapping` shape. The
/// contiguous in-place cache is the case where every row's slot is `len + t`
/// and `dst_seq_stride` is a batch stride; a page pool is the case where the
/// slot is `page * page_size + offset` from the sequence's table and the
/// sequence stride is 0. One kernel serves both, because a token is one token
/// whatever page it lands in (§10.3e). One thread per element; the grid is
/// `(head_dim, n_kv_heads, n_rows)`.
template <typename T>
[[kernel]] void flash_kv_scatter(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const device uint* slots [[buffer(2)]],
    const device FlashScatterParams* params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  const FlashScatterParams p = *params;
  const int d = gid.x;
  const int h = gid.y;
  const int row = gid.z;
  if (d >= p.head_dim || h >= p.n_kv_heads || row >= p.n_rows) {
    return;
  }
  const int b = row / p.seq_len;
  const int t = row % p.seq_len;
  const size_t slot = slots[row];
  const device T* s = src + (size_t)b * p.src_seq_stride + (size_t)h * p.src_head_stride +
                      (size_t)t * p.src_token_stride + d;
  device T* o = dst + (size_t)b * p.dst_seq_stride + (size_t)h * p.dst_head_stride +
                slot * p.dst_token_stride + d;
  *o = *s;
}

// ============================================================================
// Layout, shipped across the boundary
// ============================================================================

// 96: **ten** `int` fill 0..40, the six `size_t` land 8-aligned at 40..88, and
// the two `float` at 88..96 need no trailing pad because 96 is a multiple of the
// struct's 8-byte alignment. `n_keys_stride` is what makes the `int` count
// even and keeps that true.
//
// A `static_assert` catches the author's arithmetic, and
// `every_family_params_layout_matches_metal` catches the two *sides*
// disagreeing — a different failure, and the one #38 found live (§11.3d).
static_assert(sizeof(FlashPartialParams) == 96, "FlashPartialParams layout");
static_assert(alignof(FlashPartialParams) == 8, "FlashPartialParams alignment");
static_assert(sizeof(FlashCombineParams) == 8, "FlashCombineParams layout");
static_assert(sizeof(FlashScatterParams) == 64, "FlashScatterParams layout");
static_assert(alignof(FlashScatterParams) == 8, "FlashScatterParams alignment");

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
  FlashScatterParams probe_FlashScatterParams;

  out[0] = sizeof(FlashPartialParams);
  out[1] = flash_offsetof_rt(FlashPartialParams, gqa_factor);
  out[2] = flash_offsetof_rt(FlashPartialParams, n_keys);
  out[3] = flash_offsetof_rt(FlashPartialParams, chunk_size);
  out[4] = flash_offsetof_rt(FlashPartialParams, n_chunks);
  out[5] = flash_offsetof_rt(FlashPartialParams, chunk_capacity);
  out[6] = flash_offsetof_rt(FlashPartialParams, pages_per_chunk);
  out[7] = flash_offsetof_rt(FlashPartialParams, page_size);
  out[8] = flash_offsetof_rt(FlashPartialParams, heads_per_seq);
  out[9] = flash_offsetof_rt(FlashPartialParams, table_stride);
  out[10] = flash_offsetof_rt(FlashPartialParams, k_head_stride);
  out[11] = flash_offsetof_rt(FlashPartialParams, v_head_stride);
  out[12] = flash_offsetof_rt(FlashPartialParams, k_token_stride);
  out[13] = flash_offsetof_rt(FlashPartialParams, v_token_stride);
  out[14] = flash_offsetof_rt(FlashPartialParams, k_seq_stride);
  out[15] = flash_offsetof_rt(FlashPartialParams, v_seq_stride);
  out[16] = flash_offsetof_rt(FlashPartialParams, scale);
  out[17] = flash_offsetof_rt(FlashPartialParams, softcapping);

  out[18] = sizeof(FlashCombineParams);
  out[19] = flash_offsetof_rt(FlashCombineParams, n_chunks);
  out[20] = flash_offsetof_rt(FlashCombineParams, chunk_capacity);

  out[21] = sizeof(FlashScatterParams);
  out[22] = flash_offsetof_rt(FlashScatterParams, n_rows);
  out[23] = flash_offsetof_rt(FlashScatterParams, seq_len);
  out[24] = flash_offsetof_rt(FlashScatterParams, n_kv_heads);
  out[25] = flash_offsetof_rt(FlashScatterParams, head_dim);
  out[26] = flash_offsetof_rt(FlashScatterParams, src_seq_stride);
  out[27] = flash_offsetof_rt(FlashScatterParams, src_head_stride);
  out[28] = flash_offsetof_rt(FlashScatterParams, src_token_stride);
  out[29] = flash_offsetof_rt(FlashScatterParams, dst_seq_stride);
  out[30] = flash_offsetof_rt(FlashScatterParams, dst_head_stride);
  out[31] = flash_offsetof_rt(FlashScatterParams, dst_token_stride);
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
#define instantiate_flash_decoding_partial(type, head_dim, bn, name)         \
  template [[host_name(name "_" #type "_" #head_dim)]]                        \
  [[kernel]] void flash_decoding_partial<type, head_dim, bn>(                 \
      const device type* queries [[buffer(0)]],                               \
      const device type* keys [[buffer(1)]],                                  \
      const device type* values [[buffer(2)]],                                \
      device float* partials [[buffer(3)]],                                   \
      device float* sums [[buffer(4)]],                                       \
      device float* maxs [[buffer(5)]],                                       \
      const constant uint* chunk_table [[buffer(6)]],                           \
      const constant FlashPartialParams* params [[buffer(7)]],                  \
      const constant int* n_keys_per_seq [[buffer(8)]],                         \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_flash_decoding(type, head_dim)                            \
  instantiate_flash_decoding_partial(type, head_dim, 32, "flash_decoding_partial") \
  instantiate_flash_decoding_partial(type, head_dim, 8, "flash_decoding_partial8") \
  template [[host_name("flash_decoding_combine_" #type "_" #head_dim)]]       \
  [[kernel]] void flash_decoding_combine<type, head_dim>(                     \
      const device float* partials [[buffer(0)]],                             \
      const device float* sums [[buffer(1)]],                                 \
      const device float* maxs [[buffer(2)]],                                 \
      device type* out [[buffer(3)]],                                         \
      const constant FlashCombineParams* params [[buffer(4)]],                  \
      device uint* walk_order [[buffer(5)]],                                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]);                           \
  template [[host_name("flash_decoding_single_" #type "_" #head_dim)]]        \
  [[kernel]] void flash_decoding_single<type, head_dim>(                      \
      const device type* queries [[buffer(0)]],                               \
      const device type* keys [[buffer(1)]],                                  \
      const device type* values [[buffer(2)]],                                \
      device type* out [[buffer(3)]],                                         \
      const constant FlashPartialParams* params [[buffer(4)]],                  \
      const constant uint* chunk_table [[buffer(5)]],                           \
      const constant int* n_keys_per_seq [[buffer(6)]],                         \
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

// The scatter is per element and has no head-dimension template: one
// instantiation per dtype.
#define instantiate_flash_kv_scatter(type)                                    \
  template [[host_name("flash_kv_scatter_" #type)]]                           \
  [[kernel]] void flash_kv_scatter<type>(                                     \
      const device type* src [[buffer(0)]],                                   \
      device type* dst [[buffer(1)]],                                         \
      const device uint* slots [[buffer(2)]],                                 \
      const device FlashScatterParams* params [[buffer(3)]],                  \
      uint3 gid [[thread_position_in_grid]]);

instantiate_flash_kv_scatter(float)
instantiate_flash_kv_scatter(half)
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
