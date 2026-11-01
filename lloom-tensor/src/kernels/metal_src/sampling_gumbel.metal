// Shared centered-f32 scoring. Noise is already-generated Gumbel f32 data in
// original-id order, never uniforms or compact/rank-order noise. Scalar
// stages round separately: ((z - maximum) * inv_t) + g. Positive prepared
// weight is required, so exp32's zero tail cannot win via noise. Status
// codes/custody are the multinomial diagnostics, not a public ID API. Code 4
// here means a nonfinite variate at status[1]; score errors take priority.

// Stage 7 (Gumbel): the support N and its cutoff, published for the tiles.
kernel void sampling_gumbel_support_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &p_bits [[buffer(2)]],
    constant uint &filter_p [[buffer(3)]],
    device uint *work [[buffer(4)]],
    device const uint *members [[buffer(5)]],
    device const float *weights [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    threadgroup uint cid[SAMPLING_CAP];
    threadgroup uint cscore[SAMPLING_CAP];
    threadgroup float cweight[SAMPLING_CAP];
    threadgroup uint crank[SAMPLING_CAP];
    threadgroup uint corder[SAMPLING_CAP];
    threadgroup float lanes[1024];
    threadgroup float groups[32];
    threadgroup float reds[SAMPLING_WIDTH];
    threadgroup uint redu[SAMPLING_WIDTH];
    uint row = tg.y;
    SamplingWork wk = sampling_work(work, row, vocab);
    device const uint *z = scores + ulong(row) * vocab;
    device const float *w = weights + ulong(row) * vocab;
    device const uint *mem = members + ulong(row) * vocab;
    uint code = wk.words[W_CODE];
    uint bad_variate = wk.words[W_GBAD];
    if (code == 0u && bad_variate != 0xffffffffu) {
        code = 4u;
        if (tid == 0u) wk.words[W_BAD_OUT] = bad_variate;
    }
    SamplingCompact c;
    c.id = cid;
    c.score = cscore;
    c.weight = cweight;
    c.rank = crank;
    c.order = corder;
    c.n = min(wk.words[W_COMPACT], SAMPLING_CAP);
    c.chunk = (vocab + 1023u) / 1024u;
    SamplingNucleus n;
    n.cutoff = 0ul;
    n.count = 0u;
    n.total = 0.0f;
    n.nucleus = 0.0f;
    if (code == 0u) {
        n = sampling_prepare_nucleus(z, w, mem, vocab, p_bits, filter_p, wk, c,
                                     lanes, groups, reds, redu, tid, width);
        code = n.code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) sampling_publish_nucleus(wk, n, code);
}

// Stage 8 (Gumbel): each tile's best scored member and the support mask. A
// refused row's mask and weights are zero.
kernel void sampling_gumbel_tile_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &inv_t_bits [[buffer(2)]],
    device const uint *variates [[buffer(3)]],
    device uint *work [[buffer(4)]],
    device float *weights [[buffer(5)]],
    device uint *support [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    threadgroup ulong maxima[SAMPLING_WIDTH];
    uint tile = tg.x, row = tg.y;
    SamplingWork wk = sampling_work(work, row, vocab);
    device const uint *z = scores + ulong(row) * vocab;
    device const uint *g = variates + ulong(row) * vocab;
    device float *w = weights + ulong(row) * vocab;
    device uint *out = support + ulong(row) * vocab;
    uint code = wk.words[W_CODE];
    uint m = wk.words[W_M];
    ulong cutoff = sampling_key_pair(wk.words, W_CUTP_LO);
    uint lo = tile * SAMPLING_TILE, hi = min(lo + SAMPLING_TILE, vocab);
    ulong best = 0ul;
    for (uint i = lo + tid; i < hi; i += width) {
        uint bits = z[i];
        bool member = code == 0u && sampling_topk_finite(bits)
                   && sampling_topk_key(bits, i) >= cutoff;
        out[i] = uint(member);
        if (code != 0u) w[i] = 0.0f;
        if (member && as_type<uint>(w[i]) != 0u) {
            uint delta = sampling_sub_bits(bits, m);
            uint scaled = sampling_mul_bits(delta, inv_t_bits);
            // Inputs are finite on this positive-weight branch. Reuse the
            // bit-preserving finite add/subtract path, including tiny noise.
            uint value = sampling_sub_bits(scaled, g[i] ^ 0x80000000u);
            best = max(best, sampling_topk_key(value, i));
        }
    }
    maxima[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        for (uint t = 1u; t < width; ++t) best = max(best, maxima[t]);
        wk.tile_best[2ul * tile] = uint(best);
        wk.tile_best[2ul * tile + 1u] = uint(best >> 32);
    }
}

// Stage 9 (Gumbel): the row's best over its tiles, and the status.
kernel void sampling_gumbel_pick_f32(
    constant uint &vocab [[buffer(0)]],
    device uint *work [[buffer(1)]],
    device uint *ids [[buffer(2)]],
    device uint *status [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]]
) {
    uint tid = tid2.x;
    if (tid != 0u) return;
    uint row = tg.y;
    uint tiles = sampling_tiles(vocab);
    SamplingWork wk = sampling_work(work, row, vocab);
    uint code = wk.words[W_CODE];
    ulong best = 0ul;
    for (uint t = 0u; t < tiles; ++t)
        best = max(best, ulong(wk.tile_best[2ul * t]) | (ulong(wk.tile_best[2ul * t + 1u]) << 32));
    device uint *result = status + ulong(row) * 4ul;
    ids[row] = code == 0u && best != 0ul ? ~uint(best) : 0xffffffffu;
    result[0] = code == 0u && best == 0ul ? 5u : code;
    result[1] = wk.words[W_BAD_OUT];
    result[2] = wk.words[W_COUNT];
    result[3] = code == 0u ? wk.words[W_NCOUNT] : 0u;
}
