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

// The selection runs as a pipeline of small dispatches so a row's work is
// spread over every core instead of one threadgroup (dispatch §11.2j): a
// per-tile histogram of the key's top bits, its reduction, one scan that
// finds the bin holding the k-th key, the compaction of that bin, and a
// radix rank over the compacted candidates. Every stage reads and writes
// this row-local scratch; the host sizes it with the same arithmetic
// (`sampling_work_words`).
static constexpr constant uint SAMPLING_TILE = 4096u;
static constexpr constant uint SAMPLING_BINS = 2048u;
static constexpr constant uint SAMPLING_BIN_SHIFT = 53u;
static constexpr constant uint SAMPLING_CAP = 1024u;
static constexpr constant uint SAMPLING_CAND_CAP = 2048u;
static constexpr constant uint SAMPLING_WIDTH = 256u;
static constexpr constant uint SAMPLING_WORDS = 64u;

// Scratch words, one block per row.
static constexpr constant uint W_CODE = 0u;        // row status code (final after the draw)
static constexpr constant uint W_BAD = 1u;         // first NaN/+inf id, or UINT_MAX
static constexpr constant uint W_COUNT = 2u;       // finite scores
static constexpr constant uint W_KEEP = 3u;        // |K|, the top-k set
static constexpr constant uint W_MAX_LO = 4u;      // the row's largest key
static constexpr constant uint W_MAX_HI = 5u;
static constexpr constant uint W_M = 6u;           // the top score's bits
static constexpr constant uint W_RSEL = 7u;        // rank the select stage resolves, 0 = none
static constexpr constant uint W_KSEL = 8u;        // 1: that rank is k (the cutoff is K's)
static constexpr constant uint W_BIN = 9u;         // the histogram bin holding that rank
static constexpr constant uint W_NABOVE = 10u;     // keys strictly above the bin
static constexpr constant uint W_CAND = 11u;       // compacted candidates (atomic)
static constexpr constant uint W_CUTK_LO = 12u;    // K = {finite, key >= cutoff_k}
static constexpr constant uint W_CUTK_HI = 13u;
static constexpr constant uint W_CUTS_LO = 14u;    // the compact set = {finite, key >= cutoff_sel}
static constexpr constant uint W_CUTS_HI = 15u;
static constexpr constant uint W_COMPACT = 16u;    // compact members written (atomic)
static constexpr constant uint W_COMPACT_ON = 17u; // the compact set fits SAMPLING_CAP
static constexpr constant uint W_GBAD = 18u;       // first nonfinite Gumbel variate (atomic min)
static constexpr constant uint W_CUTP_LO = 19u;    // N = {finite, key >= cutoff_p}
static constexpr constant uint W_CUTP_HI = 20u;
static constexpr constant uint W_NCOUNT = 21u;     // |N|
static constexpr constant uint W_FALLBACK = 22u;   // the nucleus took the full-row bisection
static constexpr constant uint W_TOTAL = 23u;      // mass(K) bits
static constexpr constant uint W_NMASS = 24u;      // mass(N) bits
static constexpr constant uint W_COMPACT_OVER = 25u;
static constexpr constant uint W_BAD_OUT = 26u;    // status[1] as published

inline uint sampling_tiles(uint vocab) {
    return (vocab + SAMPLING_TILE - 1u) / SAMPLING_TILE;
}

struct SamplingWork {
    device uint *words;
    device uint *tile_words;  // per tile: finite count, first bad, max key lo, hi
    device uint *hist;        // per tile: SAMPLING_BINS counts
    device uint *ghist;       // the reduced histogram
    device float *lanes;      // the contract's 1024 lane sums over K
    device uint *tile_best;   // per tile: best Gumbel key lo, hi
};

inline ulong sampling_work_words_per_row(uint tiles) {
    return ulong(SAMPLING_WORDS) + 4ul * tiles + ulong(SAMPLING_BINS) * tiles
         + ulong(SAMPLING_BINS) + 1024ul + 2ul * tiles;
}

inline SamplingWork sampling_work(device uint *work, uint row, uint vocab) {
    uint tiles = sampling_tiles(vocab);
    device uint *base = work + ulong(row) * sampling_work_words_per_row(tiles);
    SamplingWork w;
    w.words = base;
    w.tile_words = base + SAMPLING_WORDS;
    w.hist = w.tile_words + 4ul * tiles;
    w.ghist = w.hist + ulong(SAMPLING_BINS) * tiles;
    w.lanes = (device float *)(w.ghist + SAMPLING_BINS);
    w.tile_best = w.ghist + SAMPLING_BINS + 1024u;
    return w;
}

inline ulong sampling_word_offset(uint row, uint vocab, uint word) {
    return ulong(row) * sampling_work_words_per_row(sampling_tiles(vocab)) + word;
}

inline ulong sampling_key_pair(device const uint *w, uint lo) {
    return ulong(w[lo]) | (ulong(w[lo + 1u]) << 32);
}

inline void sampling_store_key(device uint *w, uint lo, ulong key) {
    w[lo] = uint(key);
    w[lo + 1u] = uint(key >> 32);
}

// Stage 1: one threadgroup per tile counts its finite keys into bins of the
// key's top eleven bits, and reduces its finite count, first bad id and
// largest key.
kernel void sampling_select_hist_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    device uint *work [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    threadgroup atomic_uint bins[SAMPLING_BINS];
    threadgroup uint finites[SAMPLING_WIDTH];
    threadgroup uint bads[SAMPLING_WIDTH];
    threadgroup ulong maxima[SAMPLING_WIDTH];
    uint tile = tg.x, row = tg.y;
    SamplingWork w = sampling_work(work, row, vocab);
    device const uint *z = scores + ulong(row) * vocab;
    for (uint b = tid; b < SAMPLING_BINS; b += width)
        atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lo = tile * SAMPLING_TILE, hi = min(lo + SAMPLING_TILE, vocab);
    uint finite = 0u, bad = 0xffffffffu;
    ulong best = 0ul;
    for (uint i = lo + tid; i < hi; i += width) {
        uint bits = z[i];
        if (sampling_topk_finite(bits)) {
            ++finite;
            ulong key = sampling_topk_key(bits, i);
            best = max(best, key);
            atomic_fetch_add_explicit(&bins[uint(key >> SAMPLING_BIN_SHIFT)], 1u,
                                      memory_order_relaxed);
        } else if (bits != 0xff800000u) {
            bad = min(bad, i);
        }
    }
    finites[tid] = finite;
    bads[tid] = bad;
    maxima[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint *hist = w.hist + ulong(tile) * SAMPLING_BINS;
    for (uint b = tid; b < SAMPLING_BINS; b += width)
        hist[b] = atomic_load_explicit(&bins[b], memory_order_relaxed);
    if (tid == 0u) {
        for (uint t = 1u; t < width; ++t) {
            finite += finites[t];
            bad = min(bad, bads[t]);
            best = max(best, maxima[t]);
        }
        device uint *tw = w.tile_words + 4ul * tile;
        tw[0] = finite;
        tw[1] = bad;
        tw[2] = uint(best);
        tw[3] = uint(best >> 32);
    }
}

// Stage 2a: the tiles' histograms summed per bin; the tile words reduced.
kernel void sampling_select_reduce_f32(
    constant uint &vocab [[buffer(0)]],
    device uint *work [[buffer(1)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    uint row = tg.y;
    uint tiles = sampling_tiles(vocab);
    SamplingWork w = sampling_work(work, row, vocab);
    uint b = tg.x * width + tid;
    if (b < SAMPLING_BINS) {
        uint n = 0u;
        for (uint t = 0u; t < tiles; ++t) n += w.hist[ulong(t) * SAMPLING_BINS + b];
        w.ghist[b] = n;
    }
    if (tg.x == 0u && tid == 0u) {
        uint finite = 0u, bad = 0xffffffffu;
        ulong best = 0ul;
        for (uint t = 0u; t < tiles; ++t) {
            device const uint *tw = w.tile_words + 4ul * t;
            finite += tw[0];
            bad = min(bad, tw[1]);
            best = max(best, ulong(tw[2]) | (ulong(tw[3]) << 32));
        }
        w.words[W_COUNT] = finite;
        w.words[W_BAD] = bad;
        w.words[W_MAX_LO] = uint(best);
        w.words[W_MAX_HI] = uint(best >> 32);
    }
}

// Stage 2b: one threadgroup per row classifies the row, decides which rank
// the select resolves (k, or the compact capacity for a nucleus over a
// large set), and finds the histogram bin holding it. Also publishes the
// selection status and resets the atomics the later stages count on.
kernel void sampling_select_scan_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    constant uint &k [[buffer(2)]],
    constant uint &filter_p [[buffer(3)]],
    device uint *work [[buffer(4)]],
    device uint *status [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    threadgroup uint partial[SAMPLING_WIDTH];
    threadgroup uint shared[4];
    uint row = tg.y;
    SamplingWork w = sampling_work(work, row, vocab);
    device const uint *z = scores + ulong(row) * vocab;
    uint count = w.words[W_COUNT];
    uint bad = w.words[W_BAD];
    uint code = bad != 0xffffffffu ? (z[bad] == 0x7f800000u ? 2u : 1u)
                                   : (count == 0u ? 3u : 0u);
    uint keep = code != 0u ? 0u : (k == 0u ? count : min(k, count));
    bool need_k = code == 0u && k != 0u && k < count;
    bool need_cap = code == 0u && !need_k && filter_p != 0u && count > SAMPLING_CAP;
    uint r_sel = need_k ? k : (need_cap ? SAMPLING_CAP : 0u);
    uint compact_target = r_sel != 0u ? r_sel : count;
    bool compact_on = code == 0u && compact_target <= SAMPLING_CAP;
    uint per = (SAMPLING_BINS + width - 1u) / width;
    uint first = tid * per, last = min(first + per, SAMPLING_BINS);
    uint sum = 0u;
    for (uint b = first; b < last; ++b) sum += w.ghist[b];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint bin = 0u, above = 0u;
        if (r_sel != 0u) {
            uint acc = 0u;
            uint t = width;
            while (t > 0u) {
                --t;
                if (acc + partial[t] >= r_sel) break;
                acc += partial[t];
            }
            uint lo = t * per, hi = min(lo + per, SAMPLING_BINS);
            uint b = hi;
            while (b > lo) {
                --b;
                uint n = w.ghist[b];
                if (acc + n >= r_sel) break;
                acc += n;
            }
            bin = b;
            above = acc;
        }
        shared[0] = bin;
        shared[1] = above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        w.words[W_CODE] = code;
        w.words[W_KEEP] = keep;
        w.words[W_M] = code == 0u ? z[~w.words[W_MAX_LO]] : 0u;
        w.words[W_RSEL] = r_sel;
        w.words[W_KSEL] = uint(need_k);
        w.words[W_BIN] = shared[0];
        w.words[W_NABOVE] = shared[1];
        w.words[W_CAND] = 0u;
        w.words[W_CUTK_LO] = 0u;
        w.words[W_CUTK_HI] = 0u;
        w.words[W_CUTS_LO] = 0u;
        w.words[W_CUTS_HI] = 0u;
        w.words[W_COMPACT] = 0u;
        w.words[W_COMPACT_ON] = uint(compact_on);
        w.words[W_GBAD] = 0xffffffffu;
        w.words[W_CUTP_LO] = 0u;
        w.words[W_CUTP_HI] = 0u;
        w.words[W_NCOUNT] = 0u;
        w.words[W_FALLBACK] = 0u;
        w.words[W_TOTAL] = 0u;
        w.words[W_NMASS] = 0u;
        w.words[W_COMPACT_OVER] = 0u;
        w.words[W_BAD_OUT] = bad;
        device uint *s = status + ulong(row) * 4ul;
        s[0] = code;
        s[1] = bad;
        s[2] = count;
        s[3] = keep;
    }
}

// Stage 3: the ids whose key falls in the selected bin, appended in any
// order; the rank stage orders them by key, so the order does not matter.
kernel void sampling_select_compact_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    device uint *work [[buffer(2)]],
    device atomic_uint *work_atomic [[buffer(3)]],
    device uint *candidates [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    uint tile = tg.x, row = tg.y;
    SamplingWork w = sampling_work(work, row, vocab);
    if (w.words[W_RSEL] == 0u) return;
    ulong bin = ulong(w.words[W_BIN]);
    device const uint *z = scores + ulong(row) * vocab;
    device uint *cand = candidates + ulong(row) * vocab;
    device atomic_uint *counter = work_atomic + sampling_word_offset(row, vocab, W_CAND);
    uint lo = tile * SAMPLING_TILE, hi = min(lo + SAMPLING_TILE, vocab);
    for (uint i = lo + tid; i < hi; i += width) {
        uint bits = z[i];
        if (!sampling_topk_finite(bits)) continue;
        if ((sampling_topk_key(bits, i) >> SAMPLING_BIN_SHIFT) != bin) continue;
        uint at = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        cand[at] = i;
    }
}

// Stage 4: one threadgroup per row ranks the candidates by their full key
// (a radix select over the bits below the bin) and publishes the cutoff.
kernel void sampling_select_rank_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    device uint *work [[buffer(2)]],
    device const uint *candidates [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 width2 [[threads_per_threadgroup]]
) {
    uint tid = tid2.x;
    uint width = width2.x;
    threadgroup atomic_uint bins[256];
    threadgroup ulong keys[SAMPLING_CAND_CAP];
    threadgroup ulong prefix;
    threadgroup ulong known;
    threadgroup uint rank;
    uint row = tg.y;
    SamplingWork w = sampling_work(work, row, vocab);
    uint r_sel = w.words[W_RSEL];
    if (r_sel == 0u) return;
    device const uint *z = scores + ulong(row) * vocab;
    device const uint *cand = candidates + ulong(row) * vocab;
    uint n = w.words[W_CAND];
    uint above = w.words[W_NABOVE];
    bool local = n <= SAMPLING_CAND_CAP;
    if (local)
        for (uint j = tid; j < n; j += width) {
            uint id = cand[j];
            keys[j] = sampling_topk_key(z[id], id);
        }
    if (tid == 0u) {
        prefix = ulong(w.words[W_BIN]) << SAMPLING_BIN_SHIFT;
        known = 0x7fful << SAMPLING_BIN_SHIFT;
        rank = r_sel - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (uint b = tid; b < 256u; b += width)
            atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = tid; j < n; j += width) {
            ulong key;
            if (local) {
                key = keys[j];
            } else {
                uint id = cand[j];
                key = sampling_topk_key(z[id], id);
            }
            if ((key & known) == prefix) {
                uint digit = uint((key >> shift) & 255ul);
                atomic_fetch_add_explicit(&bins[digit], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            for (int b = 255; b >= 0; --b) {
                uint c = atomic_load_explicit(&bins[b], memory_order_relaxed);
                if (rank > c) rank -= c;
                else {
                    prefix |= ulong(b) << shift;
                    break;
                }
            }
            known |= 255ul << shift;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        sampling_store_key(w.words, W_CUTS_LO, prefix);
        if (w.words[W_KSEL] != 0u) sampling_store_key(w.words, W_CUTK_LO, prefix);
    }
}

// The selection mask alone: K as a dense u32 mask (tests and the mask API).
kernel void sampling_select_mask_f32(
    device const uint *scores [[buffer(0)]],
    constant uint &vocab [[buffer(1)]],
    device const uint *work [[buffer(2)]],
    device uint *mask [[buffer(3)]],
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
    uint bits = scores[ulong(row) * vocab + i];
    bool member = words[W_CODE] == 0u && sampling_topk_finite(bits)
               && sampling_topk_key(bits, i) >= sampling_key_pair(words, W_CUTK_LO);
    mask[ulong(row) * vocab + i] = uint(member);
}
