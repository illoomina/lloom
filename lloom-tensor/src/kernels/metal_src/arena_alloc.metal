#include <metal_stdlib>
using namespace metal;

// GPU-side bump allocation over the activation arena (`DESIGN.md` §9.2d case 1,
// §11.3c, issue #70).
//
// # What this computes, and what it deliberately does not
//
// One `atomic_uint` cursor over one pre-allocated arena buffer. Each request
// claims `align_up(size, ARENA_ALIGNMENT)` bytes and records the offset it was
// given. That is the whole allocator: **bump, with a per-step reset**.
//
// It is not a free list, and §9.3 is why rather than taste. MSL `device`-space
// atomics accept **only `memory_order_relaxed`** -- acquire and release do not
// exist as spellings, and `seq_cst` is threadgroup-only:
//
//     error: use of undeclared identifier 'memory_order_acquire';
//     note: candidate disabled: 'order' argument must be 'metal::memory_order_relaxed'
//
// A lock-free free list is built out of acquire/release pairs. Without them
// there is no standard way to reason about one, so building it is a research
// project rather than an issue (§9.2d case 3). A bump allocator needs none of
// that machinery: live ranges within a decode step are a DAG planned offline
// (§9.2, #68), and across steps everything resets. One relaxed increment, one
// reset.
//
// It also never selects *which* buffer. Everything is `arena_base + offset`
// against one allocation, so residency stays a CPU-side fact established once
// (§9.2d case 2). A kernel here cannot and does not ask for a buffer identity.
//
// # Determinism: why a bump allocator can be bit-exact at all
//
// `atomic_fetch_add` returns values in an order the hardware chooses, so N
// concurrent threads bumping one cursor get N distinct offsets in an order that
// is **not** reproducible. That would be a nondeterministic allocator, and
// §2.3.2 says a nondeterministic allocation layout is indistinguishable from a
// missing fence -- exactly the failure this issue must not introduce.
//
// So the ordering is taken out of the atomic rather than trusted from it.
// `arena_bump_sequential` runs on **one thread**, walking the requests in
// ordinal order, which is the order the CPU planner used (§9.2c: a slot is
// chosen by allocation ordinal, never by buffer identity). The atomic is still
// the cursor's home -- it is device state the next dispatch reads -- but the
// increment order is program order on a single thread, so the offsets are a
// pure function of the request sizes.
//
// `arena_bump_concurrent` exists beside it *as the negative control*, not as an
// optimization: it is what a naive parallel bump allocator does, and the test
// that runs it demonstrates the ordering is genuinely unspecified. Nothing on
// the decode path dispatches it. Keeping the wrong version compiled and pinned
// is the same discipline §9.3 applies to the non-aliasing reference layout --
// the property is only meaningful if the thing it excludes actually exists.
//
// # The per-step reset, and the fence it requires
//
// See `arena_reset_cursor`.

/// Slot alignment, and it must equal `ARENA_ALIGNMENT` on the Rust side.
///
/// 128 B: every Metal dtype fits, and it is `hw.cachelinesize` on this machine,
/// so a slot boundary is a cache-line boundary and adjacent slots cannot false
/// share (§9.2). A kernel cannot read the Rust constant, so the agreement is
/// checked across the boundary by `arena_alloc_reports_alignment` rather than
/// asserted on each side -- §11.3d's argument, in a constant instead of a
/// struct layout.
constant uint ARENA_ALIGNMENT_MSL = 128;

inline uint align_up(uint x, uint a) {
    return ((x + a - 1) / a) * a;
}

/// Bump-allocate `n` slices in ordinal order, from one thread.
///
/// `sizes[i]` is the byte size ordinal `i` requests; `out_offs[i]` receives the
/// offset it was given. A size of 0 marks an ordinal the arena does not serve
/// -- session state, which §9.1 keeps out entirely -- and it consumes no bytes
/// and receives `ARENA_DECLINED` rather than an offset, so that excluding an
/// ordinal does not renumber the ones after it. That is the same property the
/// CPU plan relies on (`StepPlan::by_ordinal` keeps a `None` in place).
///
/// Single-threaded by construction, and the dispatch that runs it must be
/// 1x1x1. The `tid` guard makes a wider grid harmless rather than merely
/// discouraged: extra threads return without touching the cursor, so a caller
/// that gets the grid wrong loses nothing and corrupts nothing.
kernel void arena_bump_sequential(device atomic_uint*      cursor    [[buffer(0)]],
                                  device const uint*       sizes     [[buffer(1)]],
                                  device uint*             out_offs  [[buffer(2)]],
                                  constant uint&           n         [[buffer(3)]],
                                  constant uint&           capacity  [[buffer(4)]],
                                  uint tid [[thread_position_in_grid]])
{
    if (tid != 0) { return; }

    for (uint i = 0; i < n; ++i) {
        uint want = sizes[i];
        if (want == 0) {
            // Not the arena's ordinal. It keeps its place in the sequence and
            // takes the pool path on the CPU side.
            out_offs[i] = 0xFFFFFFFFu;
            continue;
        }
        uint take = align_up(want, ARENA_ALIGNMENT_MSL);
        // Relaxed is the only spelling MSL accepts (§9.3). It is also all that
        // is needed here: the increments are ordered by program order on this
        // one thread, not by the memory model.
        uint off = atomic_fetch_add_explicit(cursor, take, memory_order_relaxed);
        // Refuse rather than wrap. A bump allocator that ran past its arena
        // would hand out offsets addressing another slot's bytes, which under
        // `HazardTrackingModeUntracked` is silent corruption (§3.5) -- so the
        // overflow case is a declined ordinal, which the CPU sees and can
        // report, and never a wrapped one.
        if (off + take > capacity) {
            out_offs[i] = 0xFFFFFFFFu;
            continue;
        }
        out_offs[i] = off;
    }
}

/// The naive parallel bump allocator, kept as a **negative control**.
///
/// Every thread claims one slice concurrently. The claims are disjoint -- that
/// much a relaxed `atomic_fetch_add` does guarantee -- but *which* thread gets
/// *which* offset is unspecified, so the mapping from ordinal to offset varies
/// run to run.
///
/// That is precisely why the decode path uses `arena_bump_sequential` instead.
/// This is dispatched only by `concurrent_bump_does_not_fix_an_ordinal_to_an_offset`,
/// which demonstrates the disorder rather than asserting its absence, so the
/// single-threaded choice is justified by a measurement rather than by caution.
kernel void arena_bump_concurrent(device atomic_uint*      cursor    [[buffer(0)]],
                                  device const uint*       sizes     [[buffer(1)]],
                                  device uint*             out_offs  [[buffer(2)]],
                                  constant uint&           n         [[buffer(3)]],
                                  constant uint&           capacity  [[buffer(4)]],
                                  uint tid [[thread_position_in_grid]])
{
    if (tid >= n) { return; }
    uint want = sizes[tid];
    if (want == 0) { out_offs[tid] = 0xFFFFFFFFu; return; }
    uint take = align_up(want, ARENA_ALIGNMENT_MSL);
    uint off = atomic_fetch_add_explicit(cursor, take, memory_order_relaxed);
    out_offs[tid] = (off + take > capacity) ? 0xFFFFFFFFu : off;
}

/// Reset the cursor to 0, opening a new decode step.
///
/// # Why this needs an explicit fence, and it is not "it works"
///
/// The reset is a **write-after-read against the whole previous step**. Step
/// `t`'s allocations were handed offsets derived from this cursor, and step
/// `t`'s kernels then read and wrote the arena bytes at those offsets. Setting
/// the cursor back to 0 declares those bytes reusable. If the reset becomes
/// visible before the previous step's last read of the arena has completed,
/// step `t+1` allocates offsets that step `t` is still reading, and two
/// unrelated values occupy the same bytes at the same time.
///
/// Under `HazardTrackingModeUntracked` (§3.5) the driver does no dependency
/// analysis, so nothing would report that. It corrupts intermittently, which
/// §2.3.2 identifies as the most expensive class of bug in this design and the
/// one that looks exactly like float noise.
///
/// Two mechanisms are involved and **both** are required, because they order
/// different things:
///
/// 1. **The encoder-level ordering, which the CPU still owns.** The reset is a
///    dispatch, and dispatches within one encoder overlap (§3.5,
///    `MTLDispatchType::Concurrent`) -- the GPU does not drain between them. So
///    the reset must be separated from the previous step's arena readers by a
///    `memoryBarrierWithScope(Buffers)` or an encoder break. That is what
///    candle's `auto_barrier` already emits, and the Rust side gets it by
///    binding the cursor as an *output* after the arena was bound as an input
///    (see `ArenaCursor::encode_reset`). Nothing in MSL can substitute for it:
///    a fence inside a kernel orders that kernel's own memory operations, not
///    another dispatch's.
///
/// 2. **The device-scope fence inside this kernel**, below. The barrier orders
///    the *dispatches*; this orders the cursor write against every other device
///    memory operation this thread has issued, at device scope. Without
///    acquire/release (§9.3) a relaxed store carries no ordering of its own, so
///    the fence is the only place the ordering can be expressed at all. It is
///    written explicitly rather than being implied by the atomic, because the
///    atomic *cannot* imply it here -- `memory_order_relaxed` is the only
///    argument the compiler accepts.
///
/// The fence is `thread_scope_device` rather than `thread_scope_threadgroup`
/// because the readers being ordered against are other dispatches, on other
/// cores. A threadgroup-scoped fence would compile, would look correct, and
/// would order nothing that matters -- which is the shape of mistake §9.3 says
/// has no safety net. `reset_uses_a_device_scope_fence` pins the spelling
/// against exactly that substitution.
kernel void arena_reset_cursor(device atomic_uint* cursor [[buffer(0)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid != 0) { return; }
    // Order everything this thread has observed of device memory before the
    // store that republishes the arena. Relaxed is the only order MSL accepts
    // in device space (§9.3), so the ordering has to be the fence's rather than
    // the store's.
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_relaxed,
                        thread_scope_device);
    atomic_store_explicit(cursor, 0u, memory_order_relaxed);
}

/// Report the alignment and the sentinel this file compiles with.
///
/// Checked across the language boundary rather than asserted on each side: a
/// `static_assert` in MSL proves only that MSL agrees with itself, and the
/// failure being guarded against is the two sides disagreeing. §11.3d makes the
/// same argument for struct layouts and found a real width mismatch by it.
kernel void arena_alloc_report(device uint* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid != 0) { return; }
    out[0] = ARENA_ALIGNMENT_MSL;
    out[1] = 0xFFFFFFFFu;
}
