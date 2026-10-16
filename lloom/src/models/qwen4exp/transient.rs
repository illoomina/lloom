//! What a Qwen4-exp prefill takes from the device beside its state, derived
//! from the config, the route its passes take and how long the route keeps
//! what it allocates (`memory.md` §9.10, §9.5q) — the machine gate's and the
//! per-pass admission's transient.
//!
//! **What a pass allocates** is enumerated per block from the launches the
//! forward makes on Metal: the HyperConnection brackets, the Gated DeltaNet's
//! fused chunked recurrence, the QSA attention (fused, or composed where the
//! head width has no fused kernel) with its indexer, the routed MoE and its
//! shared expert, the PLE, the head. Each tensor is one pool allocation,
//! rounded to the pool's alignment. The KV slab is not here: it is the
//! session's `qwen4exp-kv` claim. A test holds this against the pool's own
//! count on held passes of several geometries
//! (`tests/qwen4exp_prefill_transient.rs`), so a launch added to the forward
//! without a row here fails there.
//!
//! **How long it is kept** is the route's (`Lifetime`):
//! - Metal, resident experts: nothing inside a pass reads the host, and the
//!   pool hands a buffer back only when the command buffer that last read
//!   it completes (§6.3b), so a pass keeps everything it allocates until the
//!   GPU catches up — the sum over the layers, for every forward in flight.
//! - Metal, streamed experts: the host plan reads every layer's routing back
//!   and its wait retires everything encoded before it, so a pass keeps a
//!   window of layers per tile.
//! - CPU: a tensor is freed when its last handle drops; at most one layer's
//!   allocations at a time beside the carries (an upper bound: the CPU frees
//!   inside a layer too).
//!
//! **What moves it** (`CONTRIBUTING.md` §2.6a): a launch added to, removed
//! from or resized in the prefill forward; the pool's release rule — reuse
//! ordered by dependencies (`dependency-ordered-reuse-design.md`) collapses
//! the sum over layers toward one layer's set. The trigger is observable:
//! once the pool's census reports early-reuse hits on activations in a
//! production pass, the resident arm's sum over layers is stale and is
//! re-derived from that schedule; it is one term ([`Lifetime::Deferred`]'s
//! arm of [`PrefillTransient::forward`]), replaced without touching the
//! per-block rows. Also: the attention route, whose workspace descriptor
//! supplies [`Attention`]'s terms where there is one; the host plan's
//! per-layer wait.

use super::config::{Config, LayerKind};
use super::deltanet::CHUNK_SIZE;
use super::qsa::SPARSE_TILE_BYTES;

/// The pool's allocation alignment (`metal_backend::device::BUFFER_ALIGNMENT`):
/// every buffer is a multiple of it, and a tiny tensor costs all of it.
pub const POOL_ALIGNMENT: u64 = 128;

/// Forwards the prefill pacer keeps in flight at its floor
/// (`nn::decode::IN_FLIGHT_FLOOR`): each holds its pass until its command
/// buffers complete.
pub const FORWARDS_IN_FLIGHT: u64 = crate::nn::decode::IN_FLIGHT_FLOOR as u64;

/// Layers of one tile a streamed pass keeps pending between two of the
/// host plan's waits: the layer whose routing the wait read, and the next
/// layer's mixer and router encoded before the next wait.
pub const SYNCED_WINDOW_LAYERS: usize = 2;

/// Positions past which a forward or a context is not sized but refused:
/// far past any context the family serves.
pub const MAX_POSITIONS: usize = 1 << 24;

/// How long a pass keeps what it allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Metal with no host read inside the pass: the sum over its layers.
    Deferred,
    /// Metal with the host plan's per-layer readback: a window of layers.
    LayerSynced,
    /// The CPU: one layer's set beside the carries.
    Eager,
}

impl Lifetime {
    pub fn name(self) -> &'static str {
        match self {
            Lifetime::Deferred => "deferred",
            Lifetime::LayerSynced => "layer-synced",
            Lifetime::Eager => "eager",
        }
    }
}

/// The route a prefill pass takes, as far as its allocations go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    pub lifetime: Lifetime,
    /// Positions per chunk ([`super::prefill_chunk`]).
    pub chunk: usize,
    /// Chunks a pipelined pass interleaves (1 with the pair off).
    pub tiles: usize,
    /// The full-attention layers' fused path (`QsaAttention::prefill_fused`).
    pub qsa_fused: bool,
    /// The rows past the indexer's dense range on the paged kernel.
    pub sparse_paged: bool,
    /// Rows from which the routed experts take the GEMM arm (`MoeArm`).
    pub moe_gemm_from: usize,
    /// The MTP head observes the carry (`State::keep_nextn`).
    pub mtp: bool,
    /// A turn checkpoint inside a pass reruns each DeltaNet layer's
    /// recurrence over the prefix once per forward.
    pub checkpoint: bool,
    /// The routed experts' matmul-vec tiles compute in f16 (`--moe-act`).
    pub routed_f16: bool,
    /// The PLE is loaded and runs at its layer.
    pub ple: bool,
    /// The full-attention layers carry the indexer (the loader's always
    /// do): without it every row attends densely and observes nothing.
    pub indexer: bool,
}

impl Route {
    /// The route of this process's forwards on `metal` (else the CPU), with
    /// the experts planned on the host per layer when `host_planned` (the
    /// streamed tier), from the same switches the forward reads; what the
    /// load attached (`mtp`, the PLE, the routed arm's dtype) is the
    /// caller's to set.
    pub fn of(cfg: &Config, metal: bool, host_planned: bool) -> Self {
        let fused = metal && !super::kernels::fused_off();
        let hd = cfg.head_dim;
        Route {
            lifetime: match (metal, host_planned) {
                (false, _) => Lifetime::Eager,
                (true, false) => Lifetime::Deferred,
                (true, true) => Lifetime::LayerSynced,
            },
            chunk: super::prefill_chunk(),
            tiles: if super::prefill_pair_on() {
                super::prefill_tiles()
            } else {
                1
            },
            qsa_fused: fused && matches!(hd, 32 | 64 | 72 | 80 | 96 | 128 | 256),
            sparse_paged: fused && matches!(hd, 32 | 64 | 96 | 128 | 256 | 512),
            moe_gemm_from: moe_gemm_from(),
            mtp: false,
            checkpoint: true,
            routed_f16: false,
            ple: false,
            indexer: true,
        }
    }
}

/// The two-pass vector kernel's blocks per row (its partials' width).
#[cfg(feature = "metal")]
fn sdpa_2pass_blocks() -> usize {
    lloom_tensor::kernels::SDPA_2PASS_BLOCKS
}
/// Off Metal the paged kernel does not run.
#[cfg(not(feature = "metal"))]
fn sdpa_2pass_blocks() -> usize {
    32
}

#[cfg(feature = "metal")]
fn moe_mm_bn(pairs: usize, n_experts: usize) -> usize {
    lloom_tensor::quantized::metal::moe_mm_bn(pairs, n_experts)
}
/// Off Metal no GEMM row map exists; the narrower block bounds it.
#[cfg(not(feature = "metal"))]
fn moe_mm_bn(_pairs: usize, _n_experts: usize) -> usize {
    16
}

#[cfg(feature = "metal")]
fn moe_gemm_from() -> usize {
    lloom_tensor::quantized::metal::moe_mm_from()
}
#[cfg(not(feature = "metal"))]
fn moe_gemm_from() -> usize {
    lloom_tensor::tuning::tuning().moe.mm_from
}

/// One pool allocation of `elems` elements of `bytes` bytes each.
fn alloc(elems: usize, bytes: usize) -> u64 {
    let b = (elems as u64).saturating_mul(bytes as u64).max(1);
    b.div_ceil(POOL_ALIGNMENT).saturating_mul(POOL_ALIGNMENT)
}

const F32: usize = 4;
const F16: usize = 2;
const U32: usize = 4;
const U8: usize = 1;

/// The attention terms a full-attention layer adds for a chunk of `rows`
/// positions starting at `pos`, beside its projections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Attention {
    pub dense: u64,
    pub sparse: u64,
}

/// A prefill's transient on one config and route.
#[derive(Debug, Clone)]
pub struct PrefillTransient {
    cfg: Config,
    route: Route,
}

/// One forward's figure, by term, for a report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardBytes {
    /// The positions, where they start, and the chunks they ran as.
    pub seq: usize,
    pub pos: usize,
    pub chunks: usize,
    /// What the route keeps at once.
    pub held: u64,
    /// Everything the forward allocates, whatever completes in between.
    pub allocated: u64,
}

impl PrefillTransient {
    pub fn new(cfg: &Config, route: Route) -> Self {
        Self {
            cfg: cfg.clone(),
            route,
        }
    }

    pub fn route(&self) -> Route {
        self.route
    }

    /// The chunks a forward of `seq` positions runs as: `prefill_chunk`
    /// positions each, the last shorter.
    fn chunks(&self, seq: usize) -> Vec<usize> {
        let chunk = self.route.chunk.max(1);
        let mut out = Vec::with_capacity(seq.div_ceil(chunk));
        let mut at = 0;
        while at < seq {
            let len = (seq - at).min(chunk);
            out.push(len);
            at += len;
        }
        out
    }

    /// One HyperConnection bracket's prologue over `r` rows: the grouped
    /// norm, `silu(down·xn / n)`, the f16 round trip of `up`, its sigmoid,
    /// the collapse, the injection.
    fn hc_mix(&self, r: usize) -> u64 {
        let c = &self.cfg;
        let (n, d, lr) = (c.hc_count, c.hidden_size, c.hc_lowrank);
        let nd = n * d;
        alloc(r * nd, F32)
            + 3 * alloc(r * lr, F32)
            + alloc(r * lr, F16)
            + alloc(r * nd, F16)
            + 2 * alloc(r * nd, F32)
            + alloc(r * d, F32)
            + alloc(r * n, F32)
    }

    /// The bracket's combine: the new carry.
    fn hc_combine(&self, r: usize) -> u64 {
        alloc(r * self.cfg.hc_count * self.cfg.hidden_size, F32)
    }

    /// The DeltaNet's chunked recurrence over `r` rows (`fused_recurrence`).
    fn gdn_recurrence(&self, r: usize) -> u64 {
        let c = &self.cfg;
        let dims = c.deltanet_dims();
        let (dk, hv) = (dims.d_k, dims.h_v);
        let cs = CHUNK_SIZE;
        let nc = r.div_ceil(cs);
        let p = nc * cs;
        let n = hv * nc;
        alloc(2 * hv * p, F32) // the gate and β, chunked
            + alloc(3 * hv * p * dk, F32) // q, k, v packed
            + alloc(2 * n * cs + n + 2 * n * cs * cs, F32) // the decays
            + alloc(5 * hv * p * dk, F32) // the five scalings
            + 5 * alloc(n * cs * cs, F32) // k·kᵀ and q·kᵀ, their decays, the inverse
            + 2 * alloc(n * cs * dk, F32) // u, w
            + alloc(hv * p * dk + hv * dk * dk, F32) // the scan's output and S
            + alloc(hv * cs * dk, F32) // the scan's one-chunk scratch
    }

    /// One DeltaNet layer over `r` rows: the state's copies, the
    /// projections, the recurrence, the gated norm, the out-projection. A
    /// fresh state (the sequence's first pass) starts from zeros and takes
    /// its own buffers as it is written; one already written is copied in
    /// place.
    fn gdn(&self, r: usize, fresh: bool) -> u64 {
        let c = &self.cfg;
        let dims = c.deltanet_dims();
        let (dk, hv) = (dims.d_k, dims.h_v);
        let ch = c.conv_channels();
        let vd = c.ssm_inner_size;
        let (s, w) = (alloc(hv * dk * dk, F32), alloc(ch * c.ssm_conv_kernel, F32));
        // `S` in, the window after, `S` out; fresh: the window's zeros and
        // the two buffers the state keeps, `S` in as zeros.
        let state = if fresh { 3 * (s + w) } else { 2 * s + w };
        state
            + alloc(r * ch, F32)
            + 2 * alloc(r * vd, F32)
            + 2 * alloc(r * hv, F32)
            + alloc(r * c.hidden_size, F32)
            + self.gdn_recurrence(r)
    }

    /// The second recurrence a turn checkpoint inside the chunk runs over
    /// its prefix (at most the chunk), with the prefix's own window and `S`.
    fn gdn_checkpoint(&self, r: usize) -> u64 {
        let c = &self.cfg;
        let dims = c.deltanet_dims();
        alloc(r * c.conv_channels(), F32)
            + alloc(c.conv_channels() * c.ssm_conv_kernel, F32)
            + alloc(dims.h_v * dims.d_k * dims.d_k, F32)
            + self.gdn_recurrence(r)
    }

    /// Rows `[0, d)` of a chunk at `past` attend densely; the rest attend to
    /// the indexer's selection.
    fn dense_rows(&self, past: usize, r: usize) -> usize {
        if !self.route.indexer {
            return r;
        }
        let dense_limit = self.cfg.indexer_budget + self.cfg.indexer_compress_ratio - 1;
        r.min(dense_limit.saturating_sub(past))
    }

    /// The indexer's selection for `n` rows from `pos0` (`select_rows`).
    fn select_rows(&self, n: usize, pos0: usize) -> u64 {
        let c = &self.cfg;
        let (ih, id, ratio) = (
            c.indexer_n_heads,
            c.indexer_head_dim,
            c.indexer_compress_ratio.max(1),
        );
        let half = self.n_rot() / 2;
        let nb = (pos0.saturating_add(n) / ratio).max(1);
        let topk = c.indexer_budget / ratio;
        let stride = topk * ratio + ratio - 1;
        4 * alloc(n * ih * id, F32) // q, its norm, the rope's join, the transposed copy
            + 2 * alloc(n * half, F32)
            + 6 * alloc(n * ih * half, F32)
            + 2 * alloc(nb.saturating_mul(n * ih), F32) // the score and its relu
            + 6 * alloc(n.saturating_mul(nb), F32) // the sum, its transpose, -1, where, neg, the order
            + alloc(n.saturating_mul(nb), U8)
            + alloc(n, U32)
            + alloc(nb, U32)
            + 3 * alloc(n * topk, F32)
            + alloc(ratio, F32)
            + 2 * alloc(n * topk * ratio, F32)
            + alloc(n * (ratio - 1), U32)
            + alloc(n * stride, U32)
    }

    /// The rows past the dense range, `s` of them from `pos0`: in tiles the
    /// score and the two-pass partials bound (`sparse_rows`), each selected
    /// and attended (`attend_rows_paged`, or row by row composed).
    fn sparse(&self, s: usize, pos0: usize) -> u64 {
        if s == 0 {
            return 0;
        }
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let ratio = c.indexer_compress_ratio.max(1);
        let topk = c.indexer_budget / ratio;
        let stride = topk * ratio + ratio - 1;
        let n_blocks = pos0.saturating_add(s) / ratio;
        let score_row = n_blocks
            .max(1)
            .saturating_mul(c.indexer_n_heads)
            .saturating_mul(4);
        let partials_row = nh * 32 * hd * 4;
        let tile = (SPARSE_TILE_BYTES / score_row.max(partials_row)).max(1);
        let mut total = 0;
        let mut t0 = 0;
        while t0 < s {
            let n = tile.min(s - t0);
            total += self.select_rows(n, pos0.saturating_add(t0));
            total += if self.route.sparse_paged {
                // Per kernel arm (single- or two-pass by the row's key
                // count): the rows' key counts and context, the two-pass
                // partials; with both arms present, each group's row index,
                // its queries and table rows gathered, and the rows joined.
                let long = |t: usize| {
                    let nk = topk * ratio + pos0.wrapping_add(t0 + t + 1) % ratio;
                    crate::nn::ops::sdpa_vector_two_pass(nk, hd, lloom_tensor::DType::F32)
                };
                let n_long = (0..n).filter(|&t| long(t)).count();
                let arm = |ng: usize, two_pass: bool| {
                    if ng == 0 {
                        return 0;
                    }
                    let blocks = sdpa_2pass_blocks();
                    let split = if ng < n {
                        alloc(ng, U32) + alloc(ng * nh * hd, F32) + alloc(ng * stride, U32)
                    } else {
                        0
                    };
                    let scratch = if two_pass {
                        alloc(ng * nh * blocks * hd, F32) + 2 * alloc(ng * nh * blocks, F32)
                    } else {
                        0
                    };
                    split + alloc(ng, U32) + alloc(ng * nh * hd, F32) + scratch
                };
                let join = if n_long > 0 && n_long < n {
                    alloc(n * nh * hd, F32)
                } else {
                    0
                };
                arm(n - n_long, false) + arm(n_long, true) + join
            } else {
                // Per row: its cells' keys and values, their head
                // expansion and transposed copy, the score, its scale, the
                // softmax, the context; then the rows joined. A row's cells
                // are its selected blocks and its incomplete tail.
                let per_row: u64 = (0..n)
                    .map(|t| {
                        let nk = topk * ratio + pos0.wrapping_add(t0 + t + 1) % ratio;
                        2 * alloc(nkv * nk * hd, F32)
                            + 3 * alloc(nh * nk * hd, F32)
                            + 3 * alloc(nh * nk, F32)
                            + alloc(nh * hd, F32)
                    })
                    .sum();
                per_row + alloc(n * nh * hd, F32)
            };
            t0 += n;
        }
        if s > tile {
            total += alloc(s * nh * hd, F32);
        }
        total
    }

    /// The dense range's attention: `d` queries over `past + d` keys.
    fn dense(&self, d: usize, past: usize, r: usize) -> u64 {
        if d == 0 {
            return 0;
        }
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let kv = past.saturating_add(d);
        // A store that is not f32 is read up to f32 (dequantized) whole.
        let views = if self.cfg.kv_precision == super::config::DEFAULT_KV_PRECISION {
            0
        } else {
            2 * alloc(nkv * kv * hd, F32)
        };
        if self.route.qsa_fused {
            // The queries narrowed when the range ends inside the chunk,
            // the kernel's output and its head-last copy.
            let narrowed = u64::from(d < r);
            views + (2 + narrowed) * alloc(nh * d * hd, F32)
        } else {
            // The dense views copied out of the slab and their head
            // expansion, the queries and the transposed keys, the scores
            // through mask, fill and softmax.
            let _ = views;
            2 * alloc(nkv * kv * hd, F32)
                + 2 * alloc(nh * kv * hd, F32)
                + 3 * alloc(nh * d * hd, F32)
                + alloc(nh * hd * kv, F32)
                + 5 * alloc(nh * d * kv, F32)
                + alloc(d * kv, U8)
                + alloc(nh * d * kv, U8)
        }
    }

    fn n_rot(&self) -> usize {
        (self.cfg.head_dim as f64 * self.cfg.partial_rotary_factor) as usize
    }

    /// The attention terms of a chunk of `r` rows at `past`.
    pub fn attention(&self, r: usize, past: usize) -> Attention {
        let d = self.dense_rows(past, r);
        Attention {
            dense: self.dense(d, past, r),
            sparse: self.sparse(r - d, past.saturating_add(d)),
        }
    }

    /// One full-attention layer over `r` rows at `past`: the projections,
    /// the rows' preparation, the indexer's observe, the attention, the
    /// output gate, the out-projection.
    fn qsa(&self, r: usize, past: usize) -> u64 {
        let c = &self.cfg;
        let (nh, nkv, hd, h) = (c.n_heads, c.n_kv_heads, c.head_dim, c.hidden_size);
        let half = self.n_rot() / 2;
        let ratio = c.indexer_compress_ratio.max(1);
        let blocks = (past.saturating_add(r) / ratio).saturating_sub(past / ratio);
        let (id, ih) = (c.indexer_head_dim, c.indexer_n_heads);
        let _ = ih;
        let proj = alloc(r * nh * 2 * hd, F32) + 2 * alloc(r * nkv * hd, F32);
        let prep = if self.route.qsa_fused {
            alloc((2 * nh + 2 * nkv) * r * hd, F32)
        } else {
            // q and gate split, both norms, the rope's tables and its
            // halves, the joins, the head-first keys and values.
            2 * alloc(r * nh * hd, F32)
                + alloc(r * nh * hd, F32)
                + alloc(r * nkv * hd, F32)
                + 2 * alloc(r * half, F32)
                + 6 * alloc(r * nh * half, F32)
                + 6 * alloc(r * nkv * half, F32)
                + alloc(r * nh * hd, F32)
                + alloc(r * nkv * hd, F32)
                + 2 * alloc(r * nkv * hd, F32)
        };
        // A single completed block takes its rope from one position's
        // angles, which allocates no table.
        let tables = if blocks > 1 {
            2 * alloc(blocks * half, F32)
        } else {
            0
        };
        let observe = match (self.route.indexer, blocks) {
            (false, _) => 0,
            (true, 0) => alloc(r * id, F32),
            (true, _) => {
                alloc(r * id, F32)
                    + 4 * alloc(blocks * id, F32)
                    + tables
                    + 6 * alloc(blocks * half, F32)
            }
        };
        let att = self.attention(r, past);
        let d = self.dense_rows(past, r);
        let join = if d > 0 && d < r {
            alloc(r * nh * hd, F32)
        } else {
            0
        };
        let sparse_q = if self.route.qsa_fused && d < r {
            alloc((r - d) * nh * hd, F32)
        } else {
            0
        };
        let gate = if self.route.qsa_fused {
            alloc(r * nh * hd, F32)
        } else {
            2 * alloc(r * nh * hd, F32)
        };
        proj + prep + observe + att.dense + att.sparse + join + sparse_q + gate + alloc(r * h, F32)
    }

    /// The routed MoE and the shared expert over `r` rows.
    fn moe(&self, r: usize) -> u64 {
        let c = &self.cfg;
        let (e, k, mff, sff, h) = (
            c.n_experts,
            c.n_experts_per_tok,
            c.moe_intermediate_size,
            c.shared_expert_intermediate_size,
            c.hidden_size,
        );
        let gemm = self.route.moe_gemm_from > 0 && r >= self.route.moe_gemm_from;
        // On the GEMM arm gate and up are two launches, and a launch whose K
        // is a multiple of the GEMM's 32-wide block takes the row-grouped
        // kernel with its row map (`moe_rowmap`: the rows, the blocks at the
        // tile's block width, the count); one whose K is not keeps the
        // matmul-vec. Off the GEMM arm gate and up are one pair into one
        // buffer, and an f16 routed arm casts its activations first. An f16
        // output is counted at f32.
        let rowmap = || {
            let bn = moe_mm_bn(r * k, e);
            alloc(r * k, U32) + alloc(3 * ((r * k).div_ceil(bn) + e), U32) + alloc(1, U32)
        };
        let mapped = |kdim: usize| {
            if gemm && kdim.is_multiple_of(32) {
                rowmap()
            } else {
                0
            }
        };
        let routed = if gemm {
            2 * alloc(r * k * mff, F32) + 2 * mapped(h)
        } else {
            let cast = if self.route.routed_f16 {
                alloc(r * h, F16)
            } else {
                0
            };
            alloc(2 * r * k * mff, F32) + cast
        };
        alloc(r * e, F32) // the router
            + 2 * alloc(r * k, U32) // ids and weights
            + routed
            + alloc(r * k * mff, F32) // their SwiGLU
            + alloc(r * k * h, F32) // down
            + mapped(mff)
            + 3 * alloc(r * sff, F32) // the shared gate, up, SwiGLU
            + alloc(r * sff, F16)
            + alloc(r * h, F16)
            + alloc(r * h, F32)
            + alloc(r, F32) // its gate
            + alloc(r * h, F32) // the combine
    }

    /// The PLE at its layer over `r` rows: the gathered rows, the key and
    /// value projections, the grouped norms, the gate, the dilated conv.
    fn ple(&self, r: usize) -> u64 {
        let c = &self.cfg;
        if !self.route.ple || c.ple_layers.is_empty() {
            return 0;
        }
        let (n, d) = (c.hc_count, c.hidden_size);
        let nd = n * d;
        let width = c.ple_ngram_size.saturating_sub(1) * c.ple_heads_per_ngram * c.ple_head_dim;
        let k = c.ple_conv_kernel.max(1);
        let hist = (k - 1) * c.ple_ngram_size;
        alloc(r * width, F32)
            + alloc(r * nd, F32)
            + alloc(r * d, F32)
            + 7 * alloc(r * nd, F32) // three norms (two launches each), the product
            + 11 * alloc(r * n, F32)
            + 2 * alloc(r * n, U8)
            + alloc(r * nd, F32) // the gated value
            + alloc(r * nd, F32) // the transposed input
            + 2 * alloc(nd * hist, F32)
            + alloc(nd * (hist + r), F32)
            + (2 * k as u64 - 1) * alloc(nd * r, F32)
            + 4 * alloc(r * nd, F32) // the conv's transpose, SiLU, sum, the carry
    }

    /// The last row through the head bracket and `lm_head`.
    fn head(&self) -> u64 {
        let c = &self.cfg;
        let nd = c.hc_count * c.hidden_size;
        2 * alloc(nd, F32)
            + alloc(c.hc_lowrank, F32)
            + alloc(c.hidden_size, F32)
            + alloc(c.vocab_size, F32)
    }

    /// One layer of a chunk of `r` rows at `past` (position 0: a fresh
    /// state).
    pub fn layer(&self, l: usize, r: usize, past: usize) -> u64 {
        let mixer = match self.cfg.layer_types.get(l) {
            Some(LayerKind::Full) => self.qsa(r, past),
            _ => self.gdn(r, past == 0),
        };
        let ple = if self.cfg.ple_layers.contains(&l) {
            self.ple(r)
        } else {
            0
        };
        ple + 2 * self.hc_mix(r) + mixer + 2 * self.hc_combine(r) + self.moe(r)
    }

    /// What one chunk of `r` rows at `past` allocates over every layer, the
    /// head aside: the carry, the layers, the MTP's copy.
    pub fn chunk(&self, r: usize, past: usize) -> u64 {
        let c = &self.cfg;
        let carry = alloc(r * c.hc_count * c.hidden_size, F32);
        let layers: u64 = (0..c.n_layers).map(|l| self.layer(l, r, past)).sum();
        let nextn = if self.route.mtp { carry } else { 0 };
        carry + layers + nextn
    }

    /// How a forward of `seq` positions runs: its chunks, grouped as
    /// `forward_embeds` groups them — up to `tiles` multi-position chunks
    /// a pipelined pass, each group one head.
    fn groups(&self, seq: usize) -> Vec<Vec<usize>> {
        let chunks = self.chunks(seq);
        let tiles = self.route.tiles.max(1);
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut i = 0;
        while i < chunks.len() {
            let mut g = vec![chunks[i]];
            i += 1;
            if g[0] > 1 {
                while g.len() < tiles && i < chunks.len() && chunks[i] > 1 {
                    g.push(chunks[i]);
                    i += 1;
                }
            }
            groups.push(g);
        }
        groups
    }

    /// The most `window` consecutive layers of a chunk allocate together.
    fn window(&self, r: usize, past: usize, window: usize) -> u64 {
        let per: Vec<u64> = (0..self.cfg.n_layers)
            .map(|l| self.layer(l, r, past))
            .collect();
        per.windows(window.clamp(1, per.len().max(1)))
            .map(|w| w.iter().sum::<u64>())
            .max()
            .unwrap_or(0)
    }

    /// A forward of `seq` positions from `pos`: what its route keeps at
    /// once, and everything it allocates. Past [`MAX_POSITIONS`] the figure
    /// is `u64::MAX`, which no room admits.
    pub fn forward(&self, seq: usize, pos: usize) -> ForwardBytes {
        if seq > MAX_POSITIONS || pos > MAX_POSITIONS {
            return ForwardBytes {
                seq,
                pos,
                chunks: 0,
                held: u64::MAX,
                allocated: u64::MAX,
            };
        }
        let c = &self.cfg;
        let groups = self.groups(seq);
        let embed = alloc(seq * c.hidden_size, F32);
        let mut allocated = embed;
        let mut at = pos;
        let mut largest = 0usize;
        let mut chunks = 0;
        for g in &groups {
            for &r in g {
                allocated = allocated.saturating_add(self.chunk(r, at));
                largest = largest.max(r);
                at = at.saturating_add(r);
                chunks += 1;
            }
            allocated = allocated.saturating_add(self.head());
        }
        // One chunk of the forward carries the turn checkpoint's second
        // recurrence, in every DeltaNet layer: the largest chunk, the worst.
        let linear = c
            .layer_types
            .iter()
            .filter(|k| **k == LayerKind::Linear)
            .count();
        let ckpt_layer = if self.route.checkpoint && linear > 0 {
            self.gdn_checkpoint(largest)
        } else {
            0
        };
        allocated = allocated.saturating_add(ckpt_layer.saturating_mul(linear as u64));
        let held = match self.route.lifetime {
            // Everything the pass allocated, until completion-deferred
            // release catches up; stale once early reuse engages.
            Lifetime::Deferred => allocated,
            Lifetime::LayerSynced | Lifetime::Eager => {
                let layers = match self.route.lifetime {
                    Lifetime::LayerSynced => SYNCED_WINDOW_LAYERS,
                    _ => 1,
                };
                // A window holds the checkpoint's recurrence of the
                // DeltaNet layers inside it.
                let ckpt = ckpt_layer * layers.min(linear) as u64;
                // The tiles one pass interleaves each hold their window;
                // the forward's last group is the deepest in the context,
                // and a window is taken where it is largest.
                let mut held = 0u64;
                let mut at = pos;
                for g in &groups {
                    let mut group = embed + ckpt + self.head();
                    for &r in g {
                        group +=
                            alloc(r * c.hc_count * c.hidden_size, F32) + self.window(r, at, layers);
                        at = at.saturating_add(r);
                    }
                    held = held.max(group);
                }
                held
            }
        };
        ForwardBytes {
            seq,
            pos,
            chunks,
            held,
            allocated,
        }
    }

    /// The machine gate's claim for a session prefilling in forwards of
    /// `tile` positions (0: one chunk) up to `kv_tokens`: the forwards in
    /// flight, each the larger of a tile's two extremes — the first, on a
    /// fresh state, and the last, deepest in the context at
    /// `kv_tokens − tile`. `(claim, that forward)`.
    pub fn claim(&self, tile: usize, kv_tokens: usize) -> (u64, ForwardBytes) {
        let tile = if tile == 0 { self.route.chunk } else { tile }.max(1);
        let first = self.forward(tile, 0);
        let last = self.forward(tile, kv_tokens.saturating_sub(tile));
        let one = if last.held > first.held { last } else { first };
        let in_flight = match self.route.lifetime {
            Lifetime::Deferred => FORWARDS_IN_FLIGHT,
            // A streamed forward's first wait retires the one before it;
            // the CPU runs one at a time.
            Lifetime::LayerSynced | Lifetime::Eager => 1,
        };
        (one.held.saturating_mul(in_flight), one)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(lifetime: Lifetime, qsa_fused: bool) -> Route {
        Route {
            lifetime,
            chunk: 512,
            tiles: 2,
            qsa_fused,
            sparse_paged: qsa_fused,
            moe_gemm_from: 64,
            mtp: false,
            checkpoint: false,
            routed_f16: false,
            ple: true,
            indexer: true,
        }
    }

    /// The pool's count of one traced tiny pass of 64 positions from 0, by
    /// call site (`LLOOM_POOL_TRACE=1`, `trace_one_forward`): every block's
    /// row here is that site's bytes. The KV slab (303104 B) is the
    /// session's claim, not the pass's.
    #[test]
    fn the_tiny_pass_is_the_pools_count_by_block() {
        let cfg = Config::tiny();
        let t = PrefillTransient::new(&cfg, route(Lifetime::Deferred, false));
        assert_eq!(
            t.hc_mix(64),
            16384 + 3 * 2048 + 1024 + 8192 + 2 * 16384 + 8192 + 512
        );
        assert_eq!(t.hc_combine(64), 16384);
        assert_eq!(
            t.gdn(64, true),
            3 * 2048
                + 16384
                + 2 * 8192
                + 2 * 1024
                + 8192
                + 2048
                + 24576
                + 133248
                + 40960
                + 5 * 65536
                + 2 * 8192
                + 9216
                + 8192
        );
        assert_eq!(t.head(), 2 * 256 + 128 + 128 + 512);
        assert_eq!(t.qsa(64, 0), 1_229_824);
        assert_eq!(t.moe(64), 80_896);
        assert_eq!(t.forward(64, 0).allocated, 4_433_536 - 303_104);
    }

    /// Pairing never absorbs a one-position tail. Every group owes one
    /// head, and deferred lifetime retains all groups, not only the last.
    #[test]
    fn deferred_forward_keeps_every_chunk_and_head_across_pair_boundaries() {
        for cfg in [Config::tiny(), Config::flash_next()] {
            let t = PrefillTransient::new(&cfg, route(Lifetime::Deferred, cfg.head_dim == 256));
            for (seq, expected) in [
                (511, vec![vec![511]]),
                (512, vec![vec![512]]),
                (513, vec![vec![512], vec![1]]),
                (1023, vec![vec![512, 511]]),
                (1024, vec![vec![512, 512]]),
                (1025, vec![vec![512, 512], vec![1]]),
            ] {
                assert_eq!(t.groups(seq), expected, "seq={seq}");
                for pos in [0, cfg.indexer_budget + cfg.indexer_compress_ratio - 1] {
                    let mut allocated = alloc(seq * cfg.hidden_size, F32);
                    let mut at = pos;
                    let mut chunks = 0;
                    for group in &expected {
                        for &rows in group {
                            allocated += t.chunk(rows, at);
                            at += rows;
                            chunks += 1;
                        }
                        allocated += t.head();
                    }
                    assert_eq!(
                        t.forward(seq, pos),
                        ForwardBytes {
                            seq,
                            pos,
                            chunks,
                            held: allocated,
                            allocated,
                        },
                        "layers={} seq={seq} pos={pos}",
                        cfg.n_layers
                    );
                }
            }
        }
    }

    /// Checkpoint recurrence is retained once per linear layer in the
    /// whole forward, rather than once per chunk or once per pipeline group.
    #[test]
    fn deferred_checkpoint_adds_one_largest_recurrence_per_linear_layer() {
        for cfg in [Config::tiny(), Config::flash_next()] {
            let off = PrefillTransient::new(&cfg, route(Lifetime::Deferred, cfg.head_dim == 256));
            let on = PrefillTransient::new(
                &cfg,
                Route {
                    checkpoint: true,
                    ..off.route()
                },
            );
            let linear = cfg
                .layer_types
                .iter()
                .filter(|&&kind| kind == LayerKind::Linear)
                .count() as u64;
            assert_eq!(linear, if cfg.n_layers == 4 { 3 } else { 36 });
            for seq in [63usize, 64, 65, 511, 512, 513, 1024, 1025] {
                let without = off.forward(seq, 0);
                let with = on.forward(seq, 0);
                let extra = off.gdn_checkpoint(seq.min(512)) * linear;
                assert_eq!(with.chunks, without.chunks);
                assert_eq!(with.allocated - without.allocated, extra, "seq={seq}");
                assert_eq!(with.held - without.held, extra, "seq={seq}");
            }
        }
    }

    /// Route selection is numeric: a Metal route can be inspected without
    /// creating a device. Host-planned experts affect Metal lifetime only.
    #[test]
    fn production_route_selects_lifetime_without_creating_a_device() {
        for cfg in [Config::tiny(), Config::flash_next()] {
            for (metal, host_planned, lifetime) in [
                (false, false, Lifetime::Eager),
                (false, true, Lifetime::Eager),
                (true, false, Lifetime::Deferred),
                (true, true, Lifetime::LayerSynced),
            ] {
                assert_eq!(Route::of(&cfg, metal, host_planned).lifetime, lifetime);
            }
        }
    }

    /// Flash-Next's own geometry at the engine's 512 tile, every route. The
    /// streamed route is held against the one measurement of this family's
    /// prefill at scale (`measurements/qwen4exp-dispatches.md` P18: a
    /// 937-token prompt in 512 tiles, `--experts stream`, the pool's pending
    /// peaking at 1.34 GB and the device growing 1.51 GB): the derivation
    /// covers it and is within a quarter above it. The resident route has no
    /// measurement at scale; its figure is the sum over the layers the held
    /// tiny passes check (`memory.md` §9.10b).
    #[test]
    fn flash_next_at_a_512_tile() {
        let cfg = Config::flash_next();
        // P18 predates the turn checkpoint inside a pass: compared without it.
        let held = |lifetime, checkpoint| {
            let t = PrefillTransient::new(
                &cfg,
                Route {
                    checkpoint,
                    ..route(lifetime, true)
                },
            );
            let one = t.forward(512, 0);
            let (claim, _) = t.claim(512, 8704);
            eprintln!(
                "flash_next {} 512@0 held={} allocated={} claim(512, kv 8704)={claim}",
                lifetime.name(),
                one.held,
                one.allocated
            );
            (one.held, claim)
        };
        let (streamed, _) = held(Lifetime::LayerSynced, false);
        assert!(
            streamed >= 1_510_000_000,
            "the streamed forward's {streamed} B is under P18's measured 1.51 GB"
        );
        assert!(
            streamed <= 1_510_000_000 * 5 / 4,
            "the streamed forward's {streamed} B is over P18's 1.51 GB by a quarter"
        );
        let (with_ckpt, streamed_claim) = held(Lifetime::LayerSynced, true);
        assert!(streamed_claim >= with_ckpt && with_ckpt > streamed);
        let (resident, resident_claim) = held(Lifetime::Deferred, true);
        assert!(resident > 10 * with_ckpt, "{resident} against {with_ckpt}");
        assert!(resident_claim >= 2 * resident);
        held(Lifetime::Eager, true);
    }
}
