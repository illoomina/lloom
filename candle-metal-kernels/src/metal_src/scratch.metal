#include <metal_stdlib>
using namespace metal;

// FlashDecoding partials: the scratch class, and its sizing policy as a
// compile-tier template parameter (`DESIGN.md` §9.1, §10.4, issue #71).
//
// # What this is, and what it is emphatically not
//
// **This does not implement FlashDecoding.** That kernel is Phase 4/5 (§17
// items 14, 16). What is here writes the partials region in exactly the shape a
// real partial kernel would, and merges it in exactly the order §10.4 requires,
// so the *memory behaviour* -- sizing, addressing, fencing, merge order -- can
// be designed and tested before the arithmetic exists. The issue asks for this
// explicitly: "drive the allocation from a stub that writes the right shape".
//
// The stub writes a value derived from `(head, chunk, lane)` rather than a
// constant, and that is deliberate. §15.1 #1 (and #53, §11.3j) records that two
// kernels which both write nothing agree perfectly, so a parity assertion over
// constant output is vacuous -- and §3.7a names all-zero output as the *ICB
// path's characteristic failure*. A stub that wrote zeros would be untestable in
// exactly the way this project has already been bitten by.
//
// # Sizing as a template parameter, and why compile tier
//
// §7.2's rule assigns the tier: **if changing it changes how many registers a
// thread needs, it is compile tier.** Sizing changes addressing and loop bounds,
// so it is compile tier -- all variants pre-built, selected at construction.
//
// The precedent is already in this tree. `reduce.metal` carries
// `template<typename OP, ushort BLOCKSIZE, typename T>` -- a **non-dtype policy
// parameter** -- and §11.3b's `ParamStyle` is the Rust-side mirror of the same
// idea. `SIZING` here is that move applied to a third axis.
//
// Which policy wins was **unmeasured** when this was written, because the
// regime that distinguishes them is long context and the largest `kv_len` this
// project had then recorded is 2720 (§13.2, `.bench/` §3). So none was chosen:
// all three compile, and the A/B that would choose one was the deliverable
// rather than the choice.
//
// **Both halves of that have since moved, and this comment did not (#526).**
// #234 wired the axis to `candle-nn`'s FlashDecoding allocation and made
// `Grow` the default; #274 (§9.1f) then ran all three arms across the curve
// and found they separate at `kv_len` 4034 and at neither of its neighbours --
// so the policies stay unchosen on evidence rather than for want of it.
//
// **That measurement ran through the real kernel, not through this file.** The
// `SIZING` template parameter is the same choice `FlashPartialParams::
// chunk_capacity` now makes at run time. What is still only here is the
// **layout** axis -- `flash_decoding.metal` is hardcoded to planes and takes no
// layout parameter -- and that is what this file is now kept for.

/// Region alignment. Must equal `ARENA_ALIGNMENT` on the Rust side.
///
/// Checked across the language boundary by `scratch_reports_its_constants`
/// rather than asserted twice -- §11.3d's argument: a `static_assert` proves
/// only that one side agrees with itself, and the failure being guarded is the
/// two sides disagreeing.
constant uint SCRATCH_ALIGNMENT_MSL = 128;

/// Extra F32s per (head, chunk) beside the accumulator: online softmax's running
/// maximum `m` and denominator `l` (§10.4).
constant uint SCRATCH_STATS = 2;

/// The sizing policies, as template arguments.
///
/// Distinct types rather than an enum non-type parameter so that `if constexpr`
/// dispatch reads as overload selection and a policy with different addressing
/// can carry it as a member. MSL is C++14 via clang (§3.6), so this composes
/// and inlines at zero cost -- §7.5's "static polymorphism via template
/// parameters", the same shape `KvAccess`/`Element` take there.
struct SizingReserve {
    /// Regions are sized for the configured max context, so a step at a shorter
    /// `kv_len` writes a prefix of its region and the offsets never move.
    static constexpr constant bool fixed_offsets = true;
};
struct SizingGrow {
    /// Regions track `kv_len`, so a growth step reallocates -- which changes
    /// buffer identity, the thing #69 exists to prevent (§9.2c). The kernel's
    /// addressing is identical; what differs is who may rebind between steps.
    static constexpr constant bool fixed_offsets = false;
};
struct SizingBucket {
    /// Regions are sized to the smallest rung covering `kv_len`, so offsets are
    /// fixed *within* a rung and move when the rung does.
    static constexpr constant bool fixed_offsets = false;
};

inline uint align_up_msl(uint x, uint a) {
    return ((x + a - 1) / a) * a;
}

/// Parameters for the partials stub and the combine, packed for an ICB.
///
/// One `device const*` rather than several `constant &` scalars, because
/// `MTLIndirectComputeCommand` has **no `setBytes` in any form** (§3.7c) -- so a
/// kernel that will ever be ICB-dispatched must be able to bind its constants
/// from a buffer. Every decode-path family already carries both styles after
/// #81 (§11.3k); a family added now starts packed rather than acquiring the
/// sibling later.
struct ScratchParams {
    uint n_heads;      // query heads (32 for LFM2) -- see the Rust-side doc
    uint head_dim;     // 64 for LFM2
    uint live_chunks;  // chunks the current kv_len needs
    uint sized_chunks; // chunks the region is sized for (>= live_chunks)
    uint interleaved;  // 0 = separate planes, 1 = one padded record per (h, c)
    uint seed;         // varies the written values, so parity is not vacuous
};

/// Byte stride between consecutive (head, chunk) records under the interleaved
/// layout, in F32 elements.
///
/// `align_up((head_dim + 2) * 4, 128) / 4`. **For LFM2 that is 264 B rounded to
/// 384**, and the 264 is the point: it is the first shape in this project that
/// is *not* a multiple of the alignment. #70's warning, now in §9.2c, is that
/// every LFM2 decode activation is a 128-multiple, so `align_up` is a no-op on
/// every shape our own model produces -- and deleting it left that issue's
/// acceptance test green until a deliberately unaligned size was added.
/// Takes the struct **by value**, not by `constant&`. Two call sites make that
/// necessary rather than stylistic: the kernels call it on a `thread`-space
/// local (the register copy loaded at entry), and `scratch_report` calls it on a
/// `device` dereference. A `constant&` parameter binds neither --
/// *"cannot bind reference in default address space to object in address space
/// 'constant'"* -- so an address-space-generic helper either takes by value or
/// is templated on the pointer type, which is the choice §11.3d made for
/// `reduce`'s indexer. By value is right here because the struct is 24 bytes.
inline uint interleaved_stride_elems(ScratchParams p) {
    return align_up_msl((p.head_dim + SCRATCH_STATS) * 4u, SCRATCH_ALIGNMENT_MSL) / 4u;
}

/// The value a real partial kernel would leave at `(head, chunk, lane)`.
///
/// Deterministic in its inputs and varying across them -- see the file header on
/// why a constant would make every parity assertion vacuous.
inline float stub_value(uint head, uint chunk, uint lane, uint seed) {
    uint mixed = (head * 2654435761u) ^ (chunk * 40503u) ^ (lane * 2246822519u) ^ seed;
    // Bounded and non-degenerate: a partial is a softmax-weighted sum, so a
    // stub in [-1, 1] keeps the combine's arithmetic in the range the real one
    // would see rather than manufacturing a range it never will.
    return float(int(mixed & 0xFFFFu) - 32768) / 32768.0f;
}

/// Write one chunk's partials, in the shape a FlashDecoding partial kernel
/// would.
///
/// One threadgroup per (head, chunk); `SIZING` selects the addressing. **N of
/// these write disjoint regions and need no fences between them** -- §9.4 -- and
/// the disjointness is *our* assertion rather than the driver's (§3.5), which is
/// why `scratch_partials_write_disjoint_regions` checks it by execution rather
/// than by argument.
///
/// # An MSL constraint worth carrying: position attributes must agree in width
///
/// `uint3 gid [[threadgroup_position_in_grid]]` beside
/// `uint lane [[thread_position_in_threadgroup]]` **does not compile**:
///
/// ```text
/// error: expecting input declarations with either all scalar types or
///        all vector types with the same number of elements
/// ```
///
/// So a kernel needing a 2-D grid position must take its thread position as
/// `uint3` too and project it. That is not in `DESIGN.md` -- §3.6 lists what MSL
/// has and §11.3d/§11.3f list the layout hazards, and this is neither. It is
/// benign here (the diagnostic is a compile error, not a silent miscompile) but
/// it is the kind of thing that costs an hour if the error is read as being
/// about the template rather than about the attributes.
template <typename SIZING>
[[kernel]] void scratch_partials(device float*             out    [[buffer(0)]],
                                 device const ScratchParams* pp   [[buffer(1)]],
                                 uint3 gid  [[threadgroup_position_in_grid]],
                                 uint3 lane3 [[thread_position_in_threadgroup]])
{
    // One structure load into registers, per §11.3e's rule: a field read inside
    // a loop from `device` space is a memory access where a register would do,
    // and `constant` vs `device` is an address-space difference that matters
    // exactly there.
    ScratchParams p = *pp;
    uint lane  = lane3.x;
    uint head  = gid.x;
    uint chunk = gid.y;
    if (head >= p.n_heads || chunk >= p.live_chunks) { return; }

    // The three policies differ in how many chunks the region was *sized* for,
    // never in where chunk `c` of head `h` lands. That is what makes them an
    // A/B rather than three implementations: a policy that moved the addressing
    // would compute different bits and could not be compared.
    uint stride_chunks = SIZING::fixed_offsets ? p.sized_chunks : p.live_chunks;

    if (p.interleaved != 0u) {
        uint rec = interleaved_stride_elems(p);
        uint base = (head * stride_chunks + chunk) * rec;
        for (uint i = lane; i < p.head_dim; i += 32u) {
            out[base + i] = stub_value(head, chunk, i, p.seed);
        }
        if (lane == 0u) {
            // m and l, immediately after the accumulator within the record.
            out[base + p.head_dim + 0u] = stub_value(head, chunk, 0xF0u, p.seed);
            out[base + p.head_dim + 1u] = 1.0f + float(chunk);
        }
    } else {
        uint acc_base   = (head * stride_chunks + chunk) * p.head_dim;
        uint stats_off  = p.n_heads * stride_chunks * p.head_dim;
        uint stats_base = stats_off + (head * stride_chunks + chunk) * SCRATCH_STATS;
        for (uint i = lane; i < p.head_dim; i += 32u) {
            out[acc_base + i] = stub_value(head, chunk, i, p.seed);
        }
        if (lane == 0u) {
            out[stats_base + 0u] = stub_value(head, chunk, 0xF0u, p.seed);
            out[stats_base + 1u] = 1.0f + float(chunk);
        }
    }
}

/// Merge the partials **in ascending chunk index**, never in completion order.
///
/// # This is the determinism constraint, in the one place it can be violated
///
/// §2.3.3 #1: every reduction merges in fixed index order. §10.4 calls a
/// completion-ordered merge here **the single most likely place for
/// nondeterminism to enter the whole design**, because online softmax is
/// associative in R and **not** in floating point -- so the same partials
/// merged in a different order give different bits.
///
/// The symptom is a generation that diverges after a few hundred tokens, which
/// §2.3.2 says is indistinguishable from a missing fence. Both are why the loop
/// below is a plain ascending `for` over `live_chunks` on **one thread per
/// head**, and not a tree, a simdgroup reduction over chunks, or an atomic
/// accumulation:
///
/// - a `simd_sum` over chunks would merge in lane order, which is fixed, but
///   the *assignment* of chunks to lanes would then depend on the grid -- a
///   second thing to pin.
/// - a float atomic is order-nondeterministic by construction (§2.3.3 #2) and
///   is *slower* than the ordered walk besides (§2.3.4).
///
/// The ordered walk costs nothing worth counting: §2.3.4 records that
/// fixed-order merging is free because the partials still run fully
/// concurrently and only the merge is ordered, over `n_chunks x [32, 64]`
/// values.
template <typename SIZING>
[[kernel]] void scratch_combine(device const float*         partials [[buffer(0)]],
                                device float*                out      [[buffer(1)]],
                                device const ScratchParams*  pp       [[buffer(2)]],
                                device uint*                 order    [[buffer(3)]],
                                uint3 gid  [[threadgroup_position_in_grid]],
                                uint3 lane3 [[thread_position_in_threadgroup]])
{
    ScratchParams p = *pp;
    uint lane = lane3.x;
    uint head = gid.x;
    if (head >= p.n_heads) { return; }

    uint stride_chunks = SIZING::fixed_offsets ? p.sized_chunks : p.live_chunks;
    uint rec           = interleaved_stride_elems(p);
    uint stats_off     = p.n_heads * stride_chunks * p.head_dim;

    // The running online-softmax state, carried in registers across the walk.
    // Registers and not memory: §11.4 -- chained tiles carrying (m, l, acc)
    // are *not* valid to split across dispatches, so the merge is one kernel
    // looping internally, which is §8.3 item 5's "the correct place for the
    // unrolling instinct".
    float m = -INFINITY;
    float l = 0.0f;
    float acc[8];
    uint per_lane = (p.head_dim + 31u) / 32u;
    for (uint k = 0; k < 8u; ++k) { acc[k] = 0.0f; }

    // ---- the ordered walk ----
    // Ascending `c`, unconditionally. A reviewer changing this loop to consume
    // chunks as they complete would be reintroducing exactly what §10.4 names.
    for (uint c = 0; c < p.live_chunks; ++c) {
        uint acc_base, m_at, l_at;
        if (p.interleaved != 0u) {
            uint base = (head * stride_chunks + c) * rec;
            acc_base = base;
            m_at     = base + p.head_dim + 0u;
            l_at     = base + p.head_dim + 1u;
        } else {
            acc_base = (head * stride_chunks + c) * p.head_dim;
            m_at     = stats_off + (head * stride_chunks + c) * SCRATCH_STATS + 0u;
            l_at     = stats_off + (head * stride_chunks + c) * SCRATCH_STATS + 1u;
        }

        float m_c = partials[m_at];
        float l_c = partials[l_at];

        // Standard online-softmax merge. The rescale is applied to the running
        // accumulator *before* the new chunk is added, so the arithmetic is a
        // fixed sequence of operations per chunk in a fixed order.
        float m_new = max(m, m_c);
        float scale_old = (m == -INFINITY) ? 0.0f : exp(m - m_new);
        float scale_new = exp(m_c - m_new);
        for (uint k = 0; k < per_lane && k < 8u; ++k) {
            uint i = lane + k * 32u;
            if (i < p.head_dim) {
                acc[k] = acc[k] * scale_old + partials[acc_base + i] * scale_new;
            }
        }
        l = l * scale_old + l_c * scale_new;
        m = m_new;

        // The order actually walked, recorded so a test compares against what
        // ran rather than against the source. §2.4: an instrument that cannot be
        // shown to have engaged has not measured anything -- and the converse,
        // an ordering that cannot be observed cannot be asserted.
        if (lane == 0u) { order[head * p.live_chunks + c] = c; }
    }

    for (uint k = 0; k < per_lane && k < 8u; ++k) {
        uint i = lane + k * 32u;
        if (i < p.head_dim) {
            out[head * p.head_dim + i] = (l > 0.0f) ? (acc[k] / l) : 0.0f;
        }
    }
}

/// Report the constants this file compiled with, for the cross-boundary check.
///
/// §11.3d: a `static_assert` in MSL proves only that MSL agrees with itself.
/// `scratch_reports_its_constants` compares these against the Rust side, which
/// is what found a real width mismatch during #38.
[[kernel]] void scratch_report(device uint*                out [[buffer(0)]],
                               device const ScratchParams* pp  [[buffer(1)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid != 0) { return; }
    out[0] = SCRATCH_ALIGNMENT_MSL;
    out[1] = SCRATCH_STATS;
    out[2] = uint(sizeof(ScratchParams));
    // The interleaved record stride, padded -- 384 for LFM2's 264 B record.
    // Reported rather than asserted because it is the number that differs
    // between the two layouts and the one an alignment defect would move.
    out[3] = interleaved_stride_elems(*pp) * 4u;
    // And the unpadded record, so a reader can see the 264 the padding covers.
    out[4] = (pp->head_dim + SCRATCH_STATS) * 4u;
}

#define INSTANTIATE_SCRATCH(POLICY, SUFFIX)                                          \
template [[host_name("scratch_partials_" #SUFFIX)]] [[kernel]]                       \
void scratch_partials<POLICY>(device float*, device const ScratchParams*,            \
                              uint3, uint3);                                          \
template [[host_name("scratch_combine_" #SUFFIX)]] [[kernel]]                        \
void scratch_combine<POLICY>(device const float*, device float*,                     \
                             device const ScratchParams*, device uint*, uint3, uint3);

// All three policies, compiled. **None is chosen here** -- §7.1's compile tier
// means every variant is pre-built and the selection happens at construction,
// which is what makes the A/B free (§11.1). The names are declared once on the
// Rust side in `ScratchKernel` and resolved against *this* compiled library by
// `scratch_names_resolve`, which is §8.1b's checked registry rather than a
// generator: a test that loads every declared name against the metallib the GPU
// will be asked for is a strictly stronger oracle than two lists from one
// source, and it is what caught 48 absent `reduce` variants during #26.
INSTANTIATE_SCRATCH(SizingReserve, reserve)
INSTANTIATE_SCRATCH(SizingGrow, grow)
INSTANTIATE_SCRATCH(SizingBucket, bucket)
