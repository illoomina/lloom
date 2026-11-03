// Preserve operand bits until the subnormal/underflow arm has rounded once.
// Normal finite products stay on the existing floating-point path.
inline ulong sampling_round_right_even(ulong value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 64u) return 0ul;
    ulong whole = value >> shift;
    ulong remainder = value & ((1ul << shift) - 1ul);
    ulong halfway = 1ul << (shift - 1u);
    return whole + ulong(remainder > halfway || (remainder == halfway && (whole & 1ul)));
}

// Bit results keep gradual underflow independent of the device's FTZ mode.
// A 24-by-24-bit significand product fits in ulong; no floating-point
// operation consumes a subnormal in the integer arm.
inline uint sampling_mul_bits(uint a, uint b) {
    uint sign = (a ^ b) & 0x80000000u;
    uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
    if (aa > 0x7f800000u || bb > 0x7f800000u) return 0x7fc00000u;
    if (aa == 0x7f800000u || bb == 0x7f800000u)
        return (aa == 0u || bb == 0u) ? 0x7fc00000u : (sign | 0x7f800000u);
    if (aa == 0u || bb == 0u) return sign;
    uint ea = aa >> 23, eb = bb >> 23;
    if (ea != 0u && eb != 0u && ea + eb >= 128u) {
        // Rounded overflow is classified before native exceptional arithmetic.
        if (ea + eb >= 381u) {
            ulong sa = (aa & 0x007fffffu) | 0x00800000u;
            ulong sb = (bb & 0x007fffffu) | 0x00800000u;
            if (ea + eb >= 382u || sa * sb >= ((1ul << 47) - (1ul << 22)))
                return sign | 0x7f800000u;
        }
        volatile float product = as_type<float>(a) * as_type<float>(b);
        return as_type<uint>(product);
    }
    ulong sa = (aa & 0x007fffffu) | (ea != 0u ? 0x00800000u : 0u);
    ulong sb = (bb & 0x007fffffu) | (eb != 0u ? 0x00800000u : 0u);
    int exponent = (ea != 0u ? int(ea) - 150 : -149)
                 + (eb != 0u ? int(eb) - 150 : -149);
    ulong product = sa * sb;
    int high = 63 - int(clz(product));
    int power = high + exponent;
    if (power < -126) {
        uint shift = uint(-149 - exponent);
        return sign | uint(sampling_round_right_even(product, shift));
    }
    ulong significand = sampling_round_right_even(product, uint(high - 23));
    if (significand == 0x01000000ul) { significand >>= 1; ++power; }
    return sign | (uint(power + 127) << 23) | (uint(significand) & 0x007fffffu);
}

// Only tiny-input/cancellation and rounded-overflow cases leave native
// subtraction. Inputs are finite scores; the selector validates the row.
inline uint sampling_sub_bits(uint a, uint b) {
    uint aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
    uint ea = aa >> 23, eb = bb >> 23;
    uint sign_a = a & 0x80000000u, sign_b = (b ^ 0x80000000u) & 0x80000000u;
    if (aa == 0u && bb == 0u) return sign_a & sign_b;
    if (a == b) return 0u;
    bool tiny_input = ea == 0u || eb == 0u;
    uint large_e = max(ea, eb), small_e = min(ea, eb);
    if (tiny_input && large_e >= 26u) return ea > eb ? a : (b ^ 0x80000000u);
    if (tiny_input || (sign_a != sign_b && large_e <= 24u && large_e - small_e <= 1u)) {
        ulong sa = (aa & 0x007fffffu) | (ea != 0u ? 0x00800000u : 0u);
        ulong sb = (bb & 0x007fffffu) | (eb != 0u ? 0x00800000u : 0u);
        sa <<= ea != 0u ? ea - 1u : 0u;
        sb <<= eb != 0u ? eb - 1u : 0u;
        ulong magnitude = sign_a == sign_b ? sa + sb : (sa >= sb ? sa - sb : sb - sa);
        uint sign = sign_a == sign_b || sa >= sb ? sign_a : sign_b;
        if (magnitude == 0ul) return 0u;
        if (magnitude < 0x00800000ul) return sign | uint(magnitude);
        if (tiny_input) {
            int high = 63 - int(clz(magnitude));
            ulong significand = sampling_round_right_even(magnitude, uint(high - 23));
            if (significand == 0x01000000ul) { significand >>= 1; ++high; }
            return sign | (uint(high - 22) << 23) | (uint(significand) & 0x007fffffu);
        }
    }
    if (sign_a == sign_b && large_e == 254u && large_e - small_e <= 24u) {
        uint gap = large_e - small_e;
        ulong sa = (aa & 0x007fffffu) | 0x00800000u;
        ulong sb = (bb & 0x007fffffu) | 0x00800000u;
        ulong sum = ea >= eb ? (sa << gap) + sb : (sb << gap) + sa;
        ulong threshold = gap == 0u ? 0x01000000ul
                          : (0x01000000ul << gap) - (1ul << (gap - 1u));
        if (sum >= threshold) return sign_a | 0x7f800000u;
    }
    volatile float delta = as_type<float>(a) - as_type<float>(b);
    return as_type<uint>(delta);
}

// Numerical contract: dispatch §11.2j. These are rounded operations, not
// interchangeable calls to platform exp or a hardware-width reduction.
inline float sampling_exp32(float x) {
    if (!(x >= -87.0f)) return 0.0f;
    float n = rint(x * as_type<float>(0x3fb8aa3bu));
    float r = fma(n, -as_type<float>(0x3f318000u), x);
    r = fma(n, -as_type<float>(0xb95e8083u), r);
    float a = fma(as_type<float>(0x39506967u), r, as_type<float>(0x3ab743ceu));
    a = fma(a, r, as_type<float>(0x3c088908u));
    a = fma(a, r, as_type<float>(0x3d2aa9c1u));
    a = fma(a, r, as_type<float>(0x3e2aaaaau));
    a = fma(a, r, as_type<float>(0x3f000000u));
    volatile float square = r * r;
    volatile float y = fma(a, square, r) + 1.0f;
    return y * as_type<float>(uint(int(n) + 127) << 23);
}

// The sampler supplies a nonpositive scaled delta or NaN. Classify before
// any native comparison/FMA can consume an exceptional or subnormal value.
inline float sampling_weight_from_bits(uint scaled) {
    uint magnitude = scaled & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return 0.0f;
    if (magnitude < 0x00800000u) return 1.0f;
    return sampling_exp32(as_type<float>(scaled));
}

inline float sampling_weight(uint bits, uint m, uint inv_t_bits) {
    uint delta = sampling_sub_bits(bits, m);
    uint scaled = sampling_mul_bits(delta, inv_t_bits);
    return sampling_weight_from_bits(scaled);
}

// The contract's lane fold of the weights whose key is at least `cutoff`,
// over the ids up to `last_id`: 1024 lanes of ceil(V / 1024) consecutive ids,
// each folded left to right from +0. The weights are already zero outside
// K, so the kept test is the key alone. Emulated by `width` threads.
inline void sampling_lane_sums(
    device const uint *scores, device const float *weights, uint vocab,
    ulong cutoff, uint last_id, threadgroup float *lanes, uint tid, uint width
) {
    ulong chunk = (ulong(vocab) + 1023ul) / 1024ul;
    for (uint lane = tid; lane < 1024u; lane += width) {
        ulong lo = min(ulong(lane) * chunk, ulong(vocab));
        ulong hi = min(lo + chunk, ulong(vocab));
        volatile float sum = 0.0f;
        for (ulong i = lo; i < hi; ++i) {
            float term = uint(i) <= last_id
                       && sampling_topk_key(scores[i], uint(i)) >= cutoff
                       ? weights[i] : 0.0f;
            sum = sum + term;
        }
        lanes[lane] = sum;
    }
}

// The 32 groups of 32 lanes, each folded left to right, then the groups.
inline float sampling_fold_lanes(
    threadgroup const float *lanes, threadgroup float *groups, uint tid, uint width
) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = tid; group < 32u; group += width) {
        volatile float sum = 0.0f;
        for (uint lane = 0u; lane < 32u; ++lane)
            sum = sum + lanes[group * 32u + lane];
        groups[group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    volatile float total = 0.0f;
    for (uint group = 0u; group < 32u; ++group) total = total + groups[group];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline float sampling_mass(
    device const uint *scores, device const float *weights, uint vocab,
    ulong cutoff, uint last_id, threadgroup float *lanes, threadgroup float *groups,
    uint tid, uint width
) {
    sampling_lane_sums(scores, weights, vocab, cutoff, last_id, lanes, tid, width);
    return sampling_fold_lanes(lanes, groups, tid, width);
}

// The nucleus over a kept set too large for the compact path: the largest key
// threshold whose mass reaches the target, by bisection over the key space,
// each step one full row pass. Exact and slow; `W_FALLBACK` records it.
inline ulong sampling_nucleus_bisect(
    device const uint *scores, device const float *weights, uint vocab, ulong best,
    float target, threadgroup float *lanes, threadgroup float *groups, uint tid, uint width
) {
    ulong lo = 0ul, hi = best;
    while (lo < hi) {
        ulong mid = lo + (hi - lo) / 2ul + (hi - lo) % 2ul;
        float mass = sampling_mass(scores, weights, vocab, mid, vocab - 1u,
                                   lanes, groups, tid, width);
        if (mass >= target) lo = mid;
        else hi = mid - 1ul;
    }
    return lo;
}

// The compact kept set in threadgroup memory: members in appended order with
// their score bits and weights, each member's rank by key (0 = the top) and
// the id order (`order[q]` = the member with the q-th smallest id).
struct SamplingCompact {
    threadgroup uint *id;
    threadgroup uint *score;
    threadgroup float *weight;
    threadgroup uint *rank;
    threadgroup uint *order;
    uint n;
    uint chunk;
};

inline void sampling_compact_load(
    SamplingCompact c, device const uint *scores, device const float *weights,
    device const uint *members, uint tid, uint width
) {
    for (uint j = tid; j < c.n; j += width) {
        uint id = members[j];
        c.id[j] = id;
        c.score[j] = scores[id];
        c.weight[j] = weights[id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < c.n; j += width) {
        ulong mine = sampling_topk_key(c.score[j], c.id[j]);
        uint above = 0u, before = 0u;
        for (uint i = 0u; i < c.n; ++i) {
            above += uint(sampling_topk_key(c.score[i], c.id[i]) > mine);
            before += uint(c.id[i] < c.id[j]);
        }
        c.rank[j] = above;
        c.order[before] = j;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// mass(R_j ∩ [0, id of the q-th member in id order]) over the compact set:
// the contract's fold with the lanes and groups no member touches skipped
// (an absent id adds +0 exactly). One thread, O(q).
inline float sampling_compact_mass(SamplingCompact c, uint j, uint q_last) {
    volatile float total = 0.0f;
    volatile float group = 0.0f;
    volatile float lane = 0.0f;
    uint cur_lane = 0xffffffffu, cur_group = 0xffffffffu;
    for (uint q = 0u; q <= q_last; ++q) {
        uint member = c.order[q];
        if (c.rank[member] >= j) continue;
        uint l = c.id[member] / c.chunk;
        uint g = l / 32u;
        if (l != cur_lane) {
            if (cur_lane != 0xffffffffu) group = group + lane;
            if (g != cur_group) {
                if (cur_group != 0xffffffffu) total = total + group;
                group = 0.0f;
                cur_group = g;
            }
            lane = 0.0f;
            cur_lane = l;
        }
        lane = lane + c.weight[member];
    }
    if (cur_lane != 0xffffffffu) {
        group = group + lane;
        total = total + group;
    }
    return total;
}

// The prefix of the full-row lane sums up to lane `L` inclusive, with lane L
// itself replaced by `partial`: the contract's fold of groups then lanes.
inline float sampling_lane_prefix(
    threadgroup const float *lanes, threadgroup const float *groups, uint L, float partial
) {
    uint g = L / 32u;
    volatile float total = 0.0f;
    for (uint i = 0u; i < g; ++i) total = total + groups[i];
    volatile float part = 0.0f;
    for (uint l = g * 32u; l < L; ++l) part = part + lanes[l];
    part = part + partial;
    return total + part;
}

// Stage 5: the weights of K, zero elsewhere; the compact set's members
// appended; the Gumbel variates checked. One thread per id.
kernel void sampling_weights_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &inv_t_bits [[buffer(2)]],
    constant uint &gumbel [[buffer(3)]],
    device const uint *variates [[buffer(4)]],
    device uint *work [[buffer(5)]],
    device atomic_uint *work_atomic [[buffer(6)]],
    device uint *members [[buffer(7)]],
    device float *weights [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    uint row = tg.y;
    uint i = tg.x * width + tid;
    if (i >= vocab) return;
    ulong base = sampling_word_offset(row, vocab, 0u);
    device const uint *words = work + base;
    device float *w = weights + ulong(row) * vocab;
    if (words[W_CODE] != 0u) {
        w[i] = 0.0f;
        return;
    }
    uint bits = scores[ulong(row) * vocab + i];
    bool finite = sampling_topk_finite(bits);
    ulong key = sampling_topk_key(bits, i);
    bool member = finite && key >= sampling_key_pair(words, W_CUTK_LO);
    w[i] = member ? sampling_weight(bits, words[W_M], inv_t_bits) : 0.0f;
    if (words[W_COMPACT_ON] != 0u && finite && key >= sampling_key_pair(words, W_CUTS_LO)) {
        uint at = atomic_fetch_add_explicit(work_atomic + base + W_COMPACT, 1u,
                                            memory_order_relaxed);
        if (at < SAMPLING_CAP) members[ulong(row) * vocab + at] = i;
        else atomic_store_explicit(work_atomic + base + W_COMPACT_OVER, 1u, memory_order_relaxed);
    }
    if (gumbel != 0u && !sampling_topk_finite(variates[ulong(row) * vocab + i]))
        atomic_fetch_min_explicit(work_atomic + base + W_GBAD, i, memory_order_relaxed);
}

// Stage 6: the contract's 1024 lane sums over K, one thread per lane.
kernel void sampling_lanes_f32(
    constant uint &vocab [[buffer(0)]],
    device uint *work [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    uint row = tg.y;
    uint lane = tg.x * width + tid;
    if (lane >= 1024u) return;
    SamplingWork wk = sampling_work(work, row, vocab);
    device const float *w = weights + ulong(row) * vocab;
    ulong chunk = (ulong(vocab) + 1023ul) / 1024ul;
    ulong lo = min(ulong(lane) * chunk, ulong(vocab));
    ulong hi = min(lo + chunk, ulong(vocab));
    volatile float sum = 0.0f;
    for (ulong i = lo; i < hi; ++i) sum = sum + w[i];
    wk.lanes[lane] = sum;
}

// Shared by the multinomial and the Gumbel draw: the total mass of K, the
// nucleus N (the shortest key prefix of K whose mass reaches p × mass(K)),
// its cutoff key, size and mass. Returns the row code (0, or 5 on an invalid
// mass). The compact arrays are the caller's threadgroup memory.
struct SamplingNucleus {
    uint code;
    float total;      // mass(K)
    float nucleus;    // mass(N)
    ulong cutoff;     // N = {finite, key >= cutoff}
    uint count;       // |N|
    uint jstar;       // |N| on the compact path
    bool compact;     // N is held in the compact arrays
};

inline SamplingNucleus sampling_prepare_nucleus(
    device const uint *z, device const float *w, device const uint *members,
    uint vocab, uint p_bits, uint filter_p, SamplingWork wk,
    SamplingCompact c, threadgroup float *lanes, threadgroup float *groups,
    threadgroup float *reds, threadgroup uint *redu, uint tid, uint width
) {
    SamplingNucleus out;
    out.code = 0u;
    out.compact = wk.words[W_COMPACT_ON] != 0u && wk.words[W_COMPACT_OVER] == 0u;
    ulong cutoff_k = sampling_key_pair(wk.words, W_CUTK_LO);
    for (uint l = tid; l < 1024u; l += width) lanes[l] = wk.lanes[l];
    out.total = sampling_fold_lanes(lanes, groups, tid, width);
    out.cutoff = cutoff_k;
    out.count = wk.words[W_KEEP];
    out.nucleus = out.total;
    out.jstar = 0u;
    if (!(out.total > 0.0f) || !sampling_topk_finite(as_type<uint>(out.total))) {
        out.code = 5u;
        return out;
    }
    if (out.compact) {
        sampling_compact_load(c, z, w, members, tid, width);
        out.jstar = c.n;
    }
    if (filter_p == 0u) return out;
    float target = as_type<float>(sampling_mul_bits(p_bits, as_type<uint>(out.total)));
    bool found = false;
    if (out.compact) {
        // The least j with mass(R_j) >= target, each j by its own thread.
        uint best = 0xffffffffu;
        for (uint j = tid + 1u; j <= c.n; j += width)
            if (sampling_compact_mass(c, j, c.n - 1u) >= target) {
                best = j;
                break;
            }
        redu[tid] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            for (uint t = 1u; t < width; ++t) best = min(best, redu[t]);
            redu[0] = best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        best = redu[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (best != 0xffffffffu) {
            found = true;
            out.jstar = best;
            out.count = best;
            if (tid == 0u) reds[0] = sampling_compact_mass(c, best, c.n - 1u);
            for (uint j = tid; j < c.n; j += width)
                if (c.rank[j] == best - 1u) redu[1] = j;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out.nucleus = reds[0];
            uint member = redu[1];
            out.cutoff = sampling_topk_key(c.score[member], c.id[member]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (!found) {
        // The nucleus reaches past the compact set (or there is none): the
        // key bisection over the whole row.
        out.compact = false;
        ulong best = sampling_key_pair(wk.words, W_MAX_LO);
        out.cutoff = sampling_nucleus_bisect(z, w, vocab, best, target, lanes, groups,
                                             tid, width);
        out.nucleus = sampling_mass(z, w, vocab, out.cutoff, vocab - 1u, lanes, groups,
                                    tid, width);
        uint n = 0u;
        for (uint i = tid; i < vocab; i += width)
            n += uint(sampling_topk_finite(z[i]) && sampling_topk_key(z[i], i) >= out.cutoff);
        redu[tid] = n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            for (uint t = 1u; t < width; ++t) n += redu[t];
            redu[0] = n;
            wk.words[W_FALLBACK] = 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out.count = redu[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return out;
}

// Also returns the words the weights stage appends to and the Gumbel check
// writes to their reset state, so a draw repeated from one selection starts
// clean; every thread's reads of them precede the caller's barrier.
inline void sampling_publish_nucleus(SamplingWork wk, SamplingNucleus n, uint code) {
    wk.words[W_CODE] = code;
    sampling_store_key(wk.words, W_CUTP_LO, n.cutoff);
    wk.words[W_NCOUNT] = code == 0u ? n.count : 0u;
    wk.words[W_TOTAL] = as_type<uint>(n.total);
    wk.words[W_NMASS] = as_type<uint>(n.nucleus);
    wk.words[W_COMPACT] = 0u;
    wk.words[W_COMPACT_OVER] = 0u;
    wk.words[W_GBAD] = 0xffffffffu;
}

// Stage 7: one threadgroup per row draws the token: the inverse CDF in
// token-id order over N with the row's uniform. Private status: (0 ok / 1
// NaN / 2 +inf / 3 empty / 4 bad uniform / 5 invalid mass, first bad score
// id or UINT_MAX, finite count, support count). Refusal writes UINT_MAX as
// the id and an empty support, never a candidate.
kernel void sampling_draw_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &p_bits [[buffer(2)]],
    constant uint &filter_p [[buffer(3)]],
    device const uint *uniforms [[buffer(4)]],
    device uint *work [[buffer(5)]],
    device const uint *members [[buffer(6)]],
    device const float *weights [[buffer(7)]],
    device uint *ids [[buffer(8)]],
    device uint *status [[buffer(9)]],
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
    device uint *result = status + ulong(row) * 4ul;
    uint code = wk.words[W_CODE];
    uint ubits = uniforms[row];
    bool uniform_ok = (ubits & 0x7fffffffu) < 0x3f800000u
                   && ((ubits & 0x80000000u) == 0u || ubits == 0x80000000u);
    if (code == 0u && !uniform_ok) code = 4u;
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
    uint pick = 0xffffffffu;
    if (code == 0u) {
        float target = as_type<float>(sampling_mul_bits(ubits, as_type<uint>(n.nucleus)));
        if (n.compact) {
            // The first member in id order whose prefix mass passes the
            // target; else the last member with a positive weight (kept as
            // id + 1, so 0 is none).
            uint first = 0xffffffffu, last1 = 0u;
            for (uint q = tid; q < c.n; q += width) {
                uint member = c.order[q];
                if (c.rank[member] >= n.jstar) continue;
                if (n.nucleus > target) {
                    if (first == 0xffffffffu
                        && sampling_compact_mass(c, n.jstar, q) > target)
                        first = c.id[member];
                } else if (c.weight[member] > 0.0f) {
                    last1 = c.id[member] + 1u;
                }
            }
            redu[tid] = first;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint t = 1u; t < width; ++t) first = min(first, redu[t]);
                redu[0] = first;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            first = redu[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            redu[tid] = last1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint t = 1u; t < width; ++t) last1 = max(last1, redu[t]);
                redu[0] = n.nucleus > target ? first
                        : (last1 != 0u ? last1 - 1u : 0xffffffffu);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pick = redu[0];
        } else {
            // The large path: lane sums over N, the group and lane whose
            // prefix passes the target, then that lane walked in id order.
            if (n.cutoff != sampling_key_pair(wk.words, W_CUTK_LO))
                sampling_lane_sums(z, w, vocab, n.cutoff, vocab - 1u, lanes, tid, width);
            else
                for (uint l = tid; l < 1024u; l += width) lanes[l] = wk.lanes[l];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint group = tid; group < 32u; group += width) {
                volatile float sum = 0.0f;
                for (uint lane = 0u; lane < 32u; ++lane)
                    sum = sum + lanes[group * 32u + lane];
                groups[group] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                if (n.nucleus > target) {
                    uint L = 0xffffffffu;
                    volatile float acc = 0.0f;
                    for (uint g = 0u; g < 32u && L == 0xffffffffu; ++g) {
                        if (!(acc + groups[g] > target)) {
                            acc = acc + groups[g];
                            continue;
                        }
                        volatile float part = 0.0f;
                        for (uint l = g * 32u; l < g * 32u + 32u; ++l) {
                            part = part + lanes[l];
                            if (acc + part > target) { L = l; break; }
                        }
                    }
                    if (L != 0xffffffffu) {
                        ulong chunk = ulong(c.chunk);
                        ulong lo = min(ulong(L) * chunk, ulong(vocab));
                        ulong hi = min(lo + chunk, ulong(vocab));
                        volatile float sum = 0.0f;
                        for (ulong i = lo; i < hi; ++i) {
                            bool member = sampling_topk_key(z[i], uint(i)) >= n.cutoff;
                            sum = sum + (member ? w[i] : 0.0f);
                            if (sampling_lane_prefix(lanes, groups, L, sum) > target) {
                                pick = uint(i);
                                break;
                            }
                        }
                    }
                }
                redu[0] = pick;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pick = redu[0];
            if (pick == 0xffffffffu) {
                uint last = 0xffffffffu;
                for (uint i = tid; i < vocab; i += width)
                    if (w[i] > 0.0f && sampling_topk_key(z[i], i) >= n.cutoff) last = i;
                redu[tid] = last;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    uint best = 0xffffffffu;
                    for (uint t = 0u; t < width; ++t)
                        if (redu[t] != 0xffffffffu)
                            best = best == 0xffffffffu ? redu[t] : max(best, redu[t]);
                    redu[0] = best;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                pick = redu[0];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        sampling_publish_nucleus(wk, n, code);
        ids[row] = code != 0u ? 0xffffffffu : pick;
        result[0] = code;
        result[1] = wk.words[W_BAD_OUT];
        result[2] = wk.words[W_COUNT];
        result[3] = code == 0u ? n.count : 0u;
    }
}

// Stage 8: the support mask, N as a dense u32 mask. A refused row's mask and
// weights are zero.
kernel void sampling_support_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    device const uint *work [[buffer(2)]],
    device uint *support [[buffer(3)]],
    device float *weights [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    uint row = tg.y;
    uint i = tg.x * width + tid;
    if (i >= vocab) return;
    device const uint *words = work + sampling_word_offset(row, vocab, 0u);
    uint code = words[W_CODE];
    uint bits = scores[ulong(row) * vocab + i];
    bool member = code == 0u && sampling_topk_finite(bits)
               && sampling_topk_key(bits, i) >= sampling_key_pair(words, W_CUTP_LO);
    support[ulong(row) * vocab + i] = uint(member);
    if (code != 0u) weights[ulong(row) * vocab + i] = 0.0f;
}
