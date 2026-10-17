// One request owns the sticky error. Validation is a separate dispatch so all
// later consumers, including already queued consumers, see the same decision.
kernel void sampling_validate_tokens(
    device const uint *ids [[buffer(0)]],
    device const uint *status [[buffer(1)]],
    device uint *error [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u || error[0] != 0u) return;
    for (uint row = 0u; row < rows; ++row) {
        device const uint *s = status + ulong(row) * 4ul;
        bool valid = s[0] == 0u && s[1] == 0xffffffffu
                  && s[2] > 0u && s[2] <= vocab
                  && s[3] > 0u && s[3] <= s[2] && ids[row] < vocab;
        if (!valid) {
            error[0] = s[0] >= 1u && s[0] <= 5u ? s[0] : 6u;
            error[1] = row; error[2] = s[1]; error[3] = 0u;
            return;
        }
    }
}

// Copy bytes, not converted values: f16/bf16/f32 embeddings share this guard.
kernel void sampling_guarded_embedding(
    device const uint *ids [[buffer(0)]],
    device const uint *error [[buffer(1)]],
    device const uchar *table [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant ulong &row_bytes [[buffer(4)]],
    constant ulong &bytes [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (ulong(index) >= bytes || error[0] != 0u) return;
    ulong row = ulong(index) / row_bytes;
    out[index] = table[ulong(ids[row]) * row_bytes + ulong(index) % row_bytes];
}

kernel void sampling_guarded_copy(
    device const uint *ids [[buffer(0)]],
    device const uint *error [[buffer(1)]],
    device uint *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < rows && error[0] == 0u) out[row] = ids[row];
}

// One thread preserves repeated-id counting order without float atomics.
kernel void sampling_guarded_counts(
    device const uint *ids [[buffer(0)]],
    device const uint *error [[buffer(1)]],
    device float *counts [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u || error[0] != 0u) return;
    for (uint row = 0u; row < rows; ++row) counts[ids[row]] += 1.0f;
}

// The budget's step and closed flag are device state. An invalid draw cannot
// consume even a forced-tail step, and forced ids validate before any mutation.
kernel void sampling_guarded_feed(
    device const uint *ids [[buffer(0)]],
    device uint *error [[buffer(1)]],
    device const uint *forced [[buffer(2)]],
    device const uint *close [[buffer(3)]],
    device uint *state [[buffer(4)]],
    device uint *out [[buffer(5)]],
    constant uint &free_steps [[buffer(6)]],
    constant uint &forced_len [[buffer(7)]],
    constant uint &close_len [[buffer(8)]],
    constant uint &vocab [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u || error[0] != 0u) return;
    uint step = state[0];
    bool in_tail = step >= free_steps && step - free_steps < forced_len;
    uint fed = in_tail && state[1] == 0u ? forced[step - free_steps] : ids[0];
    if (fed >= vocab || step == 0xffffffffu) {
        error[0] = 6u; error[1] = 0u; error[2] = fed; error[3] = 0u;
        return;
    }
    bool live_after = step < free_steps || step - free_steps + 1u < forced_len;
    if (live_after)
        for (uint i = 0u; i < close_len; ++i)
            if (fed == close[i]) state[1] = 1u;
    state[0] = step + 1u;
    out[0] = fed;
}

kernel void sampling_guarded_eos(
    device const uint *ids [[buffer(0)]],
    device const uint *error [[buffer(1)]],
    device const uint *eos_ids [[buffer(2)]],
    device uchar *flag [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u || error[0] != 0u) return;
    uchar hit = 0;
    for (uint i = 0u; i < count; ++i) hit |= uchar(ids[0] == eos_ids[i]);
    flag[0] = hit;
}

// Certify all three buffers a ring reader may observe behind this marker.
kernel void sampling_guarded_marker(
    device const uint *error [[buffer(0)]],
    device uint *marker [[buffer(1)]],
    constant uint &generation [[buffer(2)]],
    device const uint *history [[buffer(3)]],
    device const uint *input [[buffer(4)]],
    device const uchar *eos [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0u && error[0] == 0u) marker[0] = generation;
}
