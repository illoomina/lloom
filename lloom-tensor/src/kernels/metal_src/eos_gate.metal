#include <metal_stdlib>
#include <metal_limits>

using namespace metal;

// The EOS gate (lloom #735, step 4). One threadgroup, dispatched by the decode
// loop right after a step's argmax and before the next forward is encoded.
// It reads the token that forward will consume; if the token is in the stop
// set it raises the sticky flag. The decode kernels instantiated with function
// constant 777 (`utils.metal`'s `EOS_GATE_ARG` / `EOS_GATE_CHECK`) read the
// flag through buffer index 30 and return at once when it is set, so a forward
// queued past an EOS launches its threadgroups and writes nothing. The host
// clears the flag between runs.
kernel void eos_gate_u32(
    device const uint *token    [[buffer(0)]],  // the sampled token the next forward consumes
    device const uint *eos_ids  [[buffer(1)]],  // the stop set, [n_eos]
    constant uint     &n_eos    [[buffer(2)]],
    device uint       *flag     [[buffer(3)]],  // sticky: 0 until an EOS is seen
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) {
        return;
    }
    uint t = token[0];
    for (uint j = 0; j < n_eos; ++j) {
        if (t == eos_ids[j]) {
            flag[0] = 1;
            return;
        }
    }
}

// The fused publish (lloom #735, step 5): argmax over the logits, the token
// into the input slot and the history slot, the EOS test into the gate's flag,
// a device fence, then the generation stamp — the six-launch chain of
// `cast → argmax → three blits → gate` as one dispatch of one threadgroup.
// The argmax is `reduce.metal`'s `indexed<T>` rule exactly: the greatest value,
// the lowest index on ties, each partial starting at {0, lowest()}, values
// compared as float (which is what `cast_f16_f32 → fast_argmax_f32` compared).
// The stamp is written by thread 0 after a device-scope barrier, which orders
// it after the slots for the GPU's threads and NOT for a host reading the
// mapped words: two stores of one dispatch reach the host in either order.
// A caller whose stamp a host reads passes `generation` 0, which writes no
// stamp, and stamps with `ring_stamp` bound to the slots' buffer.
template <typename T>
METAL_FUNC void argmax_publish_body(
    device const T *logits,
    uint n,
    device uint *token_slot,
    device uint *history_slot,
    device const uint *eos_ids,
    uint n_eos,
    device uint *flag,
    device uint *step,
    uint generation,
    uint tid,
    uint width,
    threadgroup float *sh_v,
    threadgroup uint *sh_i
) {
    float best = numeric_limits<float>::lowest();
    uint bi = 0;
    for (uint i = tid; i < n; i += width) {
        float v = float(logits[i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    sh_v[tid] = best;
    sh_i[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = width / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sh_v[tid + s];
            uint oi = sh_i[tid + s];
            if (ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid])) {
                sh_v[tid] = ov;
                sh_i[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        uint t = sh_i[0];
        token_slot[0] = t;
        history_slot[0] = t;
        for (uint j = 0; j < n_eos; ++j) {
            if (t == eos_ids[j]) {
                flag[0] = 1;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0 && generation != 0) {
        step[0] = generation;
    }
}

#define ARGMAX_PUBLISH(NAME, T)                                                  \
kernel void NAME(                                                                \
    device const T    *logits       [[buffer(0)]],                               \
    constant uint     &n            [[buffer(1)]],                               \
    device uint       *token_slot   [[buffer(2)]],                               \
    device uint       *history_slot [[buffer(3)]],                               \
    device const uint *eos_ids      [[buffer(4)]],                               \
    constant uint     &n_eos        [[buffer(5)]],                               \
    device uint       *flag         [[buffer(6)]],                               \
    device uint       *step         [[buffer(7)]],                               \
    constant uint     &generation   [[buffer(8)]],                               \
    uint tid   [[thread_index_in_threadgroup]],                                  \
    uint width [[threads_per_threadgroup]]                                       \
) {                                                                              \
    threadgroup float sh_v[1024];                                                \
    threadgroup uint sh_i[1024];                                                 \
    argmax_publish_body<T>(logits, n, token_slot, history_slot, eos_ids, n_eos,  \
                           flag, step, generation, tid, width, sh_v, sh_i);      \
}

ARGMAX_PUBLISH(argmax_publish_f16, half)
ARGMAX_PUBLISH(argmax_publish_f32, float)

// The token ring's generation stamp, for a publish whose slots were written by
// other dispatches (the unfused chain). `certified_a` and `certified_b` are
// never read: they are bound so the command stream's hazard tracker orders
// this dispatch after their last writers — a barrier inside an encoder, a
// fence wait across encoders — which is what lets a host that observes the
// stamp read the slots behind it. Nothing else orders it: the stamp's value is
// a constant and its only output is `step`. One buffer may be bound twice.
kernel void ring_stamp(
    device uint        *step        [[buffer(0)]],
    constant uint      &generation  [[buffer(1)]],
    device const uchar *certified_a [[buffer(2)]],
    device const uchar *certified_b [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        step[0] = generation;
    }
}

// The speculative round's readback as a publish: the verify's
// `[argmax, proposed]` words copied into a CPU-mapped buffer the host keeps,
// a device-scope barrier, then the generation stamp by thread 0. As in
// `argmax_publish_body`, the barrier orders the stamp after the words for the
// GPU's threads and NOT for a host reading the mapped words, so a caller whose
// stamp a host polls passes `generation` 0, which writes no stamp, and stamps
// with `ring_stamp` bound to `out` (`Device::words_publish` does).
kernel void words_publish(
    device const uint *src        [[buffer(0)]],
    constant uint     &n          [[buffer(1)]],
    device uint       *out        [[buffer(2)]],
    device uint       *step       [[buffer(3)]],
    constant uint     &generation [[buffer(4)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]
) {
    for (uint i = tid; i < n; i += width) {
        out[i] = src[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0 && generation != 0) {
        step[0] = generation;
    }
}
