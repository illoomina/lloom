#include <metal_stdlib>

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
