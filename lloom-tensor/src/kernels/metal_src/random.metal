#include <metal_stdlib>
#include <metal_integer>
#include <metal_atomic>

using namespace metal;

// Constants
// 2^32 and 1/2^32. Useful for converting between float and uint.
static constexpr constant ulong UNIF01_NORM32 = 4294967296;
static constexpr constant float UNIF01_INV32 = 2.328306436538696289e-10;
// 2 * pi
static constexpr constant float TWO_PI = 2.0 * M_PI_F;
static constexpr constant int3 S1 = {13, 19, 12};
static constexpr constant int3 S2 = {2, 25, 4};
static constexpr constant int3 S3 = {3, 11, 17};

// Used to prevent bad seeds.
static constexpr constant uint64_t PHI[16] = {
    0x9E3779B97F4A7C15,
    0xF39CC0605CEDC834,
    0x1082276BF3A27251,
    0xF86C6A11D0C18E95,
    0x2767F0B153D27B7F,
    0x0347045B5BF1827F,
    0x01886F0928403002,
    0xC1D64BA40F335E36,
    0xF06AD7AE9717877E,
    0x85839D6EFFBD7DC6,
    0x64D325D1C5371682,
    0xCADD0CCCFDFFBBE1,
    0x626E33B8D04B4331,
    0xBBF73C790D94F79D,
    0x471C4AB3ED3D82A5,
    0xFEC507705E4AE6E5,
};

// Combined Tausworthe and LCG Random Number Generator.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
struct HybridTaus {

    float state;

    HybridTaus() thread = default;
    HybridTaus() threadgroup = default;
    HybridTaus() device = default;
    HybridTaus() constant = default;

    // Generate seeds for each thread.
    METAL_FUNC static uint4 seed_per_thread(const ulong4 seeds) {
        return uint4(ulong4(seeds) * ulong4(PHI[0], PHI[1], PHI[2], PHI[3]) * ulong4(1099087573UL));
    }

    // Tausworthe generator.
    METAL_FUNC static uint taus(const uint z, const int3 s, const uint M) {
        uint b = (((z << s.x) ^ z) >> s.y);
        return (((z & M) << s.z) ^ b);
    }

    // LCG generator.
    METAL_FUNC static uint lcg(const uint z) {
        return (1664525 * z + 1013904223UL);
    }

    // Initialize the RNG state.
    METAL_FUNC static HybridTaus init(const ulong4 seeds) {
        uint4 seed = seed_per_thread(seeds);

        // Seed #1
        uint z1 = taus(seed.x, S1, 4294967294UL);
        uint z2 = taus(seed.y, S2, 4294967288UL);
        uint z3 = taus(seed.z, S3, 4294967280UL);
        uint z4 = lcg(seed.x);

        // Seed #2
        uint r1 = (z1^z2^z3^z4^seed.y);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #3
        r1 = (z1^z2^z3^z4^seed.z);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #4
        r1 = (z1^z2^z3^z4^seed.w);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        HybridTaus rng;
        rng.state = (z1^z2^z3^z4) * UNIF01_INV32;
        return rng;
    }

    METAL_FUNC float rand() {
        uint seed = this->state * UNIF01_NORM32;
        uint z1 = taus(seed, S1, 429496729UL);
        uint z2 = taus(seed, S2, 4294967288UL);
        uint z3 = taus(seed, S3, 429496280UL);
        uint z4 = lcg(seed);

        thread float result = this->state;
        this->state = (z1^z2^z3^z4) * UNIF01_INV32;
        return result;
    }
};
typedef struct
{
    atomic_uint seed[2];
} seed_buffer;


METAL_FUNC ulong atomic_load_seed(device seed_buffer *sb) {
    uint x = atomic_load_explicit(&sb->seed[0], memory_order_relaxed);
    uint y = atomic_load_explicit(&sb->seed[1], memory_order_relaxed);
    return static_cast<ulong>(x) << 32 | y;
}

METAL_FUNC void atomic_store_seed(device seed_buffer *sb, ulong desired) {
    uint x = static_cast<uint>(desired >> 32);
    uint y = static_cast<uint>(desired & 0xFFFFFFFF);
    atomic_store_explicit(&sb->seed[0], x, memory_order_relaxed);
    atomic_store_explicit(&sb->seed[1], y, memory_order_relaxed);
}

// One element per thread, each from its own stream (lloom #345).
//
// `rand_uniform` was not i.i.d. within a vector, and with equal weights the
// argmax over a drawn vector must be uniform over positions -- that is the only
// property GPU gumbel-max sampling needs, and it failed at chi-squared 36 157
// against a p=0.001 critical value of 330.5 at n=256.
//
// TWO independent defects produced it, and fixing either alone leaves the test
// failing. Both are fixed here.
//
// **1. Two elements shared one stream.** The kernel dispatched `size/2` threads
// and had each write `out[tid]` and `out[size - off - tid]` from two
// CONSECUTIVE `rand()` calls. `rand()` returns the current state and advances
// it by a deterministic `f`, so the pair is `(s, f(s))` -- a curve in the unit
// square rather than a fill of it. One element per thread removes it.
//
// **2. The global seed's orbit collapsed.** See the counter comment below. This
// is the larger of the two: with the mirror pairing removed but the seed
// advance unchanged, n=4 still read chi-squared 623 against a critical 16.27.
//
// Marginal checks see neither. `candle-metal-kernels`' own `random` test asserts
// range and mean and passes on the broken kernel; the maximum is a tail
// statistic, so a source can have near-correct marginals and still place its
// maxima wrongly. Even the marginals are in fact wrong here -- 181 of 256
// positions sit beyond 5 s.e. at 40 000 draws -- but the deviation is small
// enough (~0.013 in the mean) that a 4 000-draw check reports 4 of 256 and
// reads as clean.
//
// The cost is `size` threads where the old form dispatched `size/2`: the write
// traffic is identical and the extra work is one `HybridTaus::init` per element.
template<typename T> METAL_FUNC void rand_uniform(
    constant size_t &size,
    constant float &min,
    constant float &max,
    device seed_buffer *sb,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) {
        return;
    }

    float diff = abs(min - max);
    ulong s = atomic_load_seed(sb);
    HybridTaus rng = HybridTaus::init({s, tid, 1, 1});
    out[tid] = static_cast<T>(rng.rand() * diff + min);

    // Advance the stored seed as a COUNTER, not by feeding the generator's own
    // output back into it (lloom #345).
    //
    // This stored `rng.rand() * UNIF01_NORM32` -- the generator's own next
    // state, scaled. Two things made that lossy. `state` is a `float`, so the
    // round trip through 24 mantissa bits discards the low 8; and
    // `seed_per_thread` truncates to `uint4`, so only the low 32 bits of the
    // stored value ever reach the generator anyway. The seed therefore walked a
    // 2^32 space under a map that is not a bijection, and it collapsed:
    // measured from `set_seed(299792458)`, the orbit closes after 9 631 calls
    // onto a cycle of period **257**. Two hundred thousand draws visit 9 632
    // distinct seeds.
    //
    // A counter is a bijection on the whole space by construction, so the
    // period is 2^64 and cannot collapse for any seed. It is also what
    // `DESIGN.md` 2.3.3 #7 asks for -- "counter-based RNG ... so sampling is
    // reproducible regardless of dispatch order" -- and it makes the stream a
    // pure function of (seed, call index, tid) rather than of the generator's
    // own history.
    if (tid == 0) {
        // Re-loaded rather than reusing the `s` read above, which is NOT
        // redundant: keeping `s` live across `HybridTaus::init` and using it
        // here crashes the Metal compiler on this toolchain --
        // `XPC_ERROR_CONNECTION_INTERRUPTED ... after multiple retries`, with
        // no diagnostic, so it presents as a pipeline-creation failure rather
        // than a compile error. Reproduced 3/3 on macOS 26.6.2 / clang 21.0.0
        // with both `atomic_store_seed(sb, s + 1)` and an explicitly-typed
        // `ulong next = s + 1UL`. Only `tid == 0` runs this, so the extra load
        // is one per dispatch. `DESIGN.md` 3.7g is the standing caution that
        // these are properties of (device, OS, toolchain).
        atomic_store_seed(sb, atomic_load_seed(sb) + 1);
    }
}

// Create Gaussian normal distribution using Box-Muller transform:
// https://en.wikipedia.org/wiki/Box–Muller_transform
// One element per thread, as `rand_uniform` above and for the same reason.
//
// Box-Muller produces two independent normals from two independent uniforms,
// and the previous form wrote them to `out[tid]` and `out[size - off - tid]`.
// That is sound ONLY if `u1` and `u2` are independent, and here they were two
// consecutive states of one `HybridTaus` -- so the transform's input assumption
// was violated upstream and the two outputs inherited the dependence. Measured
// on the old kernel, `rand_normal`'s argmax-position chi-squared read 74.3 at
// n=4 against a critical 16.27 (lloom #345).
//
// Each thread now draws both uniforms for its OWN element and discards `z1`.
// That halves the transform's efficiency and is the honest cost of independence:
// the alternative -- keeping both outputs and giving them to two elements --
// is what made the pair dependent in the first place, since two elements of one
// vector must not share a stream.
template<typename T> METAL_FUNC void normal(
    constant size_t &size,
    constant float &mean,
    constant float &stddev,
    device seed_buffer *sb,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) {
        return;
    }
    ulong s = atomic_load_seed(sb);
    HybridTaus rng = HybridTaus::init({s, tid, 1, 1});
    float u1 = rng.rand();
    float u2 = rng.rand();

    float cosval;
    sincos(TWO_PI * u2, cosval);
    float mag = stddev * sqrt(-2.0 * log(u1));
    float z0  = mag * cosval + mean;

    out[tid] = static_cast<T>(z0);

    // Counter advance, as `rand_uniform` above (lloom #345).
    if (tid == 0) {
        // Re-loaded rather than reusing the `s` read above, which is NOT
        // redundant: keeping `s` live across `HybridTaus::init` and using it
        // here crashes the Metal compiler on this toolchain --
        // `XPC_ERROR_CONNECTION_INTERRUPTED ... after multiple retries`, with
        // no diagnostic, so it presents as a pipeline-creation failure rather
        // than a compile error. Reproduced 3/3 on macOS 26.6.2 / clang 21.0.0
        // with both `atomic_store_seed(sb, s + 1)` and an explicitly-typed
        // `ulong next = s + 1UL`. Only `tid == 0` runs this, so the extra load
        // is one per dispatch. `DESIGN.md` 3.7g is the standing caution that
        // these are properties of (device, OS, toolchain).
        atomic_store_seed(sb, atomic_load_seed(sb) + 1);
    }
}

#define UNIFORM_OP(NAME, T)                             \
kernel void rand_uniform_##NAME(                        \
    constant size_t &size,                              \
    constant float &min,                                \
    constant float &max,                                \
    device seed_buffer *sb,                             \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    rand_uniform<T>(size, min, max, sb, out, tid);      \
}                                                       \

#define NORMAL_OP(NAME, T)                              \
kernel void rand_normal_##NAME(                         \
    constant size_t &size,                              \
    constant float &mean,                               \
    constant float &stddev,                             \
    device seed_buffer *sb,                             \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    normal<T>(size, mean, stddev, sb, out, tid);        \
}                                                       \


#define RANDOM_OPS(NAME, T) \
UNIFORM_OP(NAME, T)         \
NORMAL_OP(NAME, T)          \

RANDOM_OPS(f32, float)
RANDOM_OPS(f16, half)

#if __METAL_VERSION__ >= 310
RANDOM_OPS(bf16, bfloat)
#endif
