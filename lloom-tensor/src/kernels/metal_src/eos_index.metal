#include <metal_stdlib>

using namespace metal;

// The batched EOS check: over a window of `n` freshly-sampled token ids, find the
// FIRST (lowest) index whose token is one of the `n_eos` stop ids, or report `n`
// ("no EOS in this window"). One dispatch per commit window — NOT the per-token
// `broadcast_eq → max_all → slice_set` train (~5 dispatches/token) the poll arm
// runs — and a full-threadgroup parallel reduction, not a single-lane scan, so the
// dispatch uses the machine it occupies.
//
// The reduction is a MIN over per-lane candidates: lane `i` (i < n) proposes `i`
// if `tokens[i]` matches any stop id, else the sentinel `n`; lanes past the window
// propose `n` too. The min across all lanes is the first matching index, or `n`.
// Ties cannot happen (each index is distinct), so a plain `min` over `uint` is the
// whole of it — no `indexed<T>` tie-break needed, unlike the general arg-reduce.
//
// One threadgroup covers the window: `n` is the wake cadence (a handful to a few
// tens of tokens), well under a threadgroup's width, so the host launches a single
// group of `next_pow2(n)` lanes and the tree reduces in shared memory. `n_eos` is a
// loop-invariant handful (the model's stop set), scanned per lane. The threadgroup
// width the host dispatches (`threads_per_threadgroup`) is the reduction span; the
// shared array is sized to the Metal maximum (1024) so any legal width fits.

kernel void eos_index_u32(
    device const uint *tokens   [[buffer(0)]],  // the window's token ids, [n]
    constant uint     &n        [[buffer(1)]],  // window length (real tokens)
    device const uint *eos_ids  [[buffer(2)]],  // the stop set, [n_eos]
    constant uint     &n_eos    [[buffer(3)]],  // stop set size
    device uint       *out      [[buffer(4)]],  // out[0] = first match index, or n
    uint tid [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    threadgroup uint shared[1024];

    // Per-lane candidate: this lane's index if its token is a stop id, else the
    // sentinel `n` ("no match here"). A lane past the window is a sentinel too. The
    // stop set is tiny, so the inner loop is short and branch-uniform in practice.
    uint cand = n; // sentinel = "none"
    if (tid < n) {
        uint t = tokens[tid];
        for (uint j = 0; j < n_eos; ++j) {
            if (t == eos_ids[j]) {
                cand = tid;
                break;
            }
        }
    }
    shared[tid] = cand;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Harris tree reduction with `min`: after this, shared[0] holds the lowest
    // matching index across the whole threadgroup, or `n` if none matched.
    for (uint s = width / 2; s > 0; s >>= 1) {
        if (tid < s) {
            uint other = shared[tid + s];
            if (other < shared[tid]) {
                shared[tid] = other;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[0] = shared[0];
    }
}
