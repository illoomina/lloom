#include <metal_stdlib>
using namespace metal;

// Integer classification preserves NaNs and subnormal score ranks under fast math.
inline bool sampling_topk_finite(uint bits) {
    return (bits & 0x7f800000u) != 0x7f800000u;
}

// Larger keys rank first; signed zeros tie and original ids break every tie.
inline ulong sampling_topk_key(uint bits, uint id) {
    if ((bits & 0x7fffffffu) == 0u) bits = 0u;
    uint score = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (ulong(score) << 32) | ulong(~id);
}

// A selection mask only, not a draw. Status is private diagnostic metadata:
// (code: 0 ok / 1 NaN / 2 +inf / 3 empty, first bad id or UINT_MAX,
//  finite count, kept count). Refused rows write a zero mask.
kernel void sampling_topk_mask_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &k [[buffer(2)]],
    device uint *mask [[buffer(3)]],
    device uint *status [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint bins[256];
    threadgroup ulong prefix;
    threadgroup ulong known;
    threadgroup uint rank;
    threadgroup uint count;
    threadgroup uint keep;
    threadgroup uint code;
    device const uint *z = scores + ulong(row) * vocab;
    device uint *out = mask + ulong(row) * vocab;

    if (tid == 0) {
        atomic_store_explicit(&bins[0], 0u, memory_order_relaxed);
        atomic_store_explicit(&bins[1], 0xffffffffu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint finite = 0;
    uint bad = 0xffffffffu;
    for (ulong i = tid; i < vocab; i += width) {
        uint bits = z[i];
        if (sampling_topk_finite(bits)) ++finite;
        else if (bits != 0xff800000u) bad = min(bad, uint(i));
    }
    atomic_fetch_add_explicit(&bins[0], finite, memory_order_relaxed);
    atomic_fetch_min_explicit(&bins[1], bad, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        count = atomic_load_explicit(&bins[0], memory_order_relaxed);
        uint first_bad = atomic_load_explicit(&bins[1], memory_order_relaxed);
        code = first_bad != 0xffffffffu ? (z[first_bad] == 0x7f800000u ? 2u : 1u)
                                      : (count == 0u ? 3u : 0u);
        keep = code != 0u ? 0u : (k == 0u ? count : min(k, count));
        prefix = 0ul;
        known = 0ul;
        rank = keep;
        status[ulong(row) * 4 + 0] = code;
        status[ulong(row) * 4 + 1] = first_bad;
        status[ulong(row) * 4 + 2] = count;
        status[ulong(row) * 4 + 3] = keep;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (code == 0u && keep < count) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            for (uint b = tid; b < 256; b += width)
                atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (ulong i = tid; i < vocab; i += width) {
                uint bits = z[i];
                if (!sampling_topk_finite(bits)) continue;
                ulong key = sampling_topk_key(bits, uint(i));
                if ((key & known) == prefix) {
                    uint digit = uint((key >> shift) & 255ul);
                    atomic_fetch_add_explicit(&bins[digit], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                for (int b = 255; b >= 0; --b) {
                    uint n = atomic_load_explicit(&bins[b], memory_order_relaxed);
                    if (rank > n) rank -= n;
                    else {
                        prefix |= ulong(b) << shift;
                        break;
                    }
                }
                known |= 255ul << shift;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (ulong i = tid; i < vocab; i += width) {
        uint bits = z[i];
        out[i] = uint(code == 0u && sampling_topk_finite(bits)
                      && (keep == count || sampling_topk_key(bits, uint(i)) >= prefix));
    }
}
