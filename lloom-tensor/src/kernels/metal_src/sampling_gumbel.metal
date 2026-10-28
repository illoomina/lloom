// Shared centered-f32 scoring. The incumbent public Metal route is not yet
// switched to this private draw. Noise is already-generated Gumbel
// f32 data in original-id order, never uniforms or compact/rank-order noise.
// Scalar stages round separately: ((z - maximum) * inv_t) + g. Positive
// prepared weight is required, so exp32's zero tail cannot win via noise.
// Status codes/custody are the multinomial diagnostics, not a public ID API.
// Code 4 here means a nonfinite variate at status[1]; score errors take priority.
kernel void sampling_filtered_gumbel_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &inv_t_bits [[buffer(2)]],
    constant uint &p_bits [[buffer(3)]],
    constant uint &filter_p [[buffer(4)]],
    device const uint *variates [[buffer(5)]],
    device const uint *kept [[buffer(6)]],
    device const uint *selection_status [[buffer(7)]],
    device float *weight_scratch [[buffer(8)]],
    device uint *support [[buffer(9)]],
    device uint *ids [[buffer(10)]],
    device uint *status [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    threadgroup float lanes[1024];
    threadgroup float groups[32];
    threadgroup ulong maxima[256];
    device const uint *z = scores + ulong(row) * vocab;
    device const uint *k = kept + ulong(row) * vocab;
    device const uint *g = variates + ulong(row) * vocab;
    device float *w = weight_scratch + ulong(row) * vocab;
    device uint *out = support + ulong(row) * vocab;
    device const uint *s = selection_status + ulong(row) * 4;
    device uint *result = status + ulong(row) * 4;
    uint bad = 0xffffffffu;
    for (ulong i = tid; i < vocab; i += width)
        if (!sampling_topk_finite(g[i])) bad = min(bad, uint(i));
    maxima[tid] = ulong(bad);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < width; ++t) bad = min(bad, uint(maxima[t]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint code = s[0] != 0u ? s[0] : (bad == 0xffffffffu ? 0u : 4u);
    if (code != 0u) {
        for (ulong i = tid; i < vocab; i += width) { out[i] = 0u; w[i] = 0.0f; }
        if (tid == 0) {
            ids[row] = 0xffffffffu;
            result[0] = code; result[1] = s[0] != 0u ? s[1] : bad;
            result[2] = s[2]; result[3] = 0u;
        }
        return;
    }
    SamplingPrepared prepared = sampling_prepare_support(
        z, k, w, vocab, inv_t_bits, p_bits, filter_p,
        lanes, groups, maxima, tid, width);
    if (!(prepared.total > 0.0f)
        || !sampling_topk_finite(as_type<uint>(prepared.total))) code = 5u;
    // All readers of preparation's maxima must finish before draw reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong best = 0ul;
    for (ulong i = tid; i < vocab; i += width) {
        bool member = code == 0u && k[i] != 0u
                   && sampling_topk_key(z[i], uint(i)) >= prepared.cutoff;
        out[i] = uint(member);
        if (member && as_type<uint>(w[i]) != 0u) {
            uint delta = sampling_sub_bits(z[i], prepared.maximum);
            uint scaled = sampling_mul_bits(delta, inv_t_bits);
            // Inputs are finite on this positive-weight branch. Reuse the
            // bit-preserving finite add/subtract path, including tiny noise.
            uint value = sampling_sub_bits(scaled, g[i] ^ 0x80000000u);
            best = max(best, sampling_topk_key(value, uint(i)));
        }
    }
    maxima[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        best = 0ul;
        for (uint t = 0; t < width; ++t) best = max(best, maxima[t]);
        uint count = 0u;
        for (ulong i = 0; i < vocab; ++i)
            count += uint(code == 0u && k[i] != 0u
                          && sampling_topk_key(z[i], uint(i)) >= prepared.cutoff);
        ids[row] = code == 0u && best != 0ul ? ~uint(best) : 0xffffffffu;
        result[0] = code == 0u && best == 0ul ? 5u : code;
        result[1] = s[1]; result[2] = s[2]; result[3] = count;
    }
}
