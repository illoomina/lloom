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

// Contract lanes retain their original vocabulary positions even on sparse
// sets; the threadgroup width only schedules those 1024 logical lanes.
inline float sampling_mass(
    device const uint *scores, device const uint *kept,
    device const float *weights, uint vocab, ulong cutoff, uint last_id,
    threadgroup float *lanes, threadgroup float *groups,
    uint tid, uint width
) {
    ulong chunk = (ulong(vocab) + 1023ul) / 1024ul;
    for (uint lane = tid; lane < 1024; lane += width) {
        ulong lo = min(ulong(lane) * chunk, ulong(vocab));
        ulong hi = min(lo + chunk, ulong(vocab));
        volatile float sum = 0.0f;
        for (ulong i = lo; i < hi; ++i) {
            float term = kept[i] != 0u && uint(i) <= last_id
                       && sampling_topk_key(scores[i], uint(i)) >= cutoff
                       ? weights[i] : 0.0f;
            sum = sum + term;
        }
        lanes[lane] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = tid; group < 32; group += width) {
        volatile float sum = 0.0f;
        for (uint lane = 0; lane < 32; ++lane)
            sum = sum + lanes[group * 32 + lane];
        groups[group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    volatile float total = 0.0f;
    for (uint group = 0; group < 32; ++group) total = total + groups[group];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// Both private draws consume this exact processed-score preparation. The
// caller has already checked the selector status and its variates.
struct SamplingPrepared {
    ulong cutoff;
    float total;
    uint maximum;
};

inline SamplingPrepared sampling_prepare_support(
    device const uint *z, device const uint *k, device float *w,
    uint vocab, uint inv_t_bits, uint p_bits, uint filter_p,
    threadgroup float *lanes, threadgroup float *groups,
    threadgroup ulong *maxima, uint tid, uint width
) {
    ulong best = 0ul;
    for (ulong i = tid; i < vocab; i += width)
        if (k[i] != 0u) best = max(best, sampling_topk_key(z[i], uint(i)));
    maxima[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best = 0ul;
    for (uint t = 0; t < width; ++t) best = max(best, maxima[t]);
    uint top_id = ~uint(best);
    uint m = z[top_id];
    for (ulong i = tid; i < vocab; i += width) {
        float weight = 0.0f;
        if (k[i] != 0u) {
            uint delta = sampling_sub_bits(z[i], m);
            uint scaled = sampling_mul_bits(delta, inv_t_bits);
            weight = sampling_weight_from_bits(scaled);
        }
        w[i] = weight;
    }
    threadgroup_barrier(mem_flags::mem_device);
    ulong cutoff = 0ul;
    if (filter_p != 0u) {
        float total = sampling_mass(z, k, w, vocab, 0ul, vocab - 1u,
                                    lanes, groups, tid, width);
        float target = as_type<float>(sampling_mul_bits(p_bits, as_type<uint>(total)));
        ulong lo = 0ul, hi = best;
        while (lo < hi) {
            ulong mid = lo + (hi - lo) / 2ul + (hi - lo) % 2ul;
            float mass = sampling_mass(z, k, w, vocab, mid, vocab - 1u,
                                       lanes, groups, tid, width);
            if (mass >= target) lo = mid;
            else hi = mid - 1ul;
        }
        cutoff = lo;
    }
    float total = sampling_mass(z, k, w, vocab, cutoff, vocab - 1u,
                               lanes, groups, tid, width);
    return {cutoff, total, m};
}

// Mechanism: exact key-threshold bisection for the conditional nucleus,
// then exact token-id-prefix bisection for one supplied uniform. Neither
// search depends on k fitting local scratch. The top-k stage owns validation
// of the entire score row, including scores outside K.
// Private status: (0 ok / 1 NaN / 2 +inf / 3 empty / 4 bad uniform /
// 5 invalid mass, first bad score id or UINT_MAX, finite count, support count).
// Refusal always writes UINT_MAX as id and an empty support, never a candidate.
kernel void sampling_multinomial_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &inv_t_bits [[buffer(2)]],
    constant uint &p_bits [[buffer(3)]],
    constant uint &filter_p [[buffer(4)]],
    device const uint *uniforms [[buffer(5)]],
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
    device float *w = weight_scratch + ulong(row) * vocab;
    device uint *out = support + ulong(row) * vocab;
    device const uint *s = selection_status + ulong(row) * 4;
    device uint *result = status + ulong(row) * 4;
    uint ubits = uniforms[row];
    bool uniform_ok = (ubits & 0x7fffffffu) < 0x3f800000u
                   && ((ubits & 0x80000000u) == 0u || ubits == 0x80000000u);
    uint code = s[0] != 0u ? s[0] : (uniform_ok ? 0u : 4u);
    if (code != 0u) {
        for (ulong i = tid; i < vocab; i += width) { out[i] = 0u; w[i] = 0.0f; }
        if (tid == 0) {
            ids[row] = 0xffffffffu;
            result[0] = code; result[1] = s[1]; result[2] = s[2]; result[3] = 0u;
        }
        return;
    }
    SamplingPrepared prepared = sampling_prepare_support(
        z, k, w, vocab, inv_t_bits, p_bits, filter_p,
        lanes, groups, maxima, tid, width);
    ulong cutoff = prepared.cutoff;
    float total = prepared.total;
    if (!(total > 0.0f) || !sampling_topk_finite(as_type<uint>(total))) code = 5u;
    uint pick = 0xffffffffu;
    if (code == 0u) {
        float target = as_type<float>(sampling_mul_bits(ubits, as_type<uint>(total)));
        if (total > target) {
            uint lo = 0u, hi = vocab - 1u;
            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2u;
                float mass = sampling_mass(z, k, w, vocab, cutoff, mid,
                                           lanes, groups, tid, width);
                if (mass > target) hi = mid;
                else lo = mid + 1u;
            }
            pick = lo;
        }
    }
    for (ulong i = tid; i < vocab; i += width)
        out[i] = uint(code == 0u && k[i] != 0u
                      && sampling_topk_key(z[i], uint(i)) >= cutoff);
    if (tid == 0) {
        uint count = 0u, last = 0xffffffffu;
        for (ulong i = 0; i < vocab; ++i) {
            bool member = code == 0u && k[i] != 0u
                       && sampling_topk_key(z[i], uint(i)) >= cutoff;
            if (member) { ++count; if (w[i] > 0.0f) last = uint(i); }
        }
        ids[row] = code != 0u ? 0xffffffffu : (pick == 0xffffffffu ? last : pick);
        result[0] = code; result[1] = s[1]; result[2] = s[2]; result[3] = count;
    }
}
