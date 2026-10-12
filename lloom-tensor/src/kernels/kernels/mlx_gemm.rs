use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, GemvParams, ParamStyle,
};
use crate::kernels::metal::{Buffer, ComputeCommandEncoder, Device, MetalDeviceType};
use crate::kernels::utils::{EncoderProvider, Input};
use crate::kernels::{
    debug_group, set_params, ConstantValues, EncoderParam, Kernels, MetalKernelError, Output,
    Source, Value,
};
use objc2_metal::MTLSize;

use crate::gemv_tile::{derive_gemv_tile, gemv_tile_dims};
pub use crate::gemv_tile::{gemv_mixed_gnorm_serves, gemv_norm_serves, GEMV_GROUP_NORM_MAX_GROUPS};

/// Trailing alignment of the packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirror rather than written as a
/// literal, and `gemv_params_layout_matches_metal` is what proves that mirror
/// agrees with `gemv.metal`.
const GEMV_PARAMS_ALIGN: usize = core::mem::align_of::<GemvParams>();

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

/// Tile configuration for GEMM kernel.
///
/// These parameters control the block sizes and warp tiling for the Metal GEMM kernel.
/// Different configurations are optimal for different matrix sizes and data types.
///
/// Reference: MLX steel_gemm_fused.metal
#[derive(Copy, Clone, Debug)]
struct TileConfig {
    bm: usize, // Block size M
    bn: usize, // Block size N
    bk: usize, // Block size K
    wm: usize, // Warp tiles M
    wn: usize, // Warp tiles N
}

impl TileConfig {
    const fn new(bm: usize, bn: usize, bk: usize, wm: usize, wn: usize) -> Self {
        Self { bm, bn, bk, wm, wn }
    }
}

// Predefined tile configurations matching MLX's steel_gemm_fused.metal
// Note: TILE_32_32_16_2_2 is kept for backward compatibility and as a fallback.
// It's used by MLX for small devices ('g'/'p') but we default to medium device configs.
#[allow(dead_code)]
const TILE_32_32_16_2_2: TileConfig = TileConfig::new(32, 32, 16, 2, 2);
const TILE_64_64_16_2_2: TileConfig = TileConfig::new(64, 64, 16, 2, 2);
const TILE_64_64_16_1_2: TileConfig = TileConfig::new(64, 64, 16, 1, 2);
const TILE_64_32_32_2_2: TileConfig = TileConfig::new(64, 32, 32, 2, 2);
const TILE_32_64_16_1_2: TileConfig = TileConfig::new(32, 64, 16, 1, 2);

// Tiles instantiated by issue #383 so the BN axis can be measured. They are
// unreachable unless `LLOOM_383_TILE` names one; nothing in
// `select_tile_config`'s own logic returns them.
const TILE_32_16_16_2_2: TileConfig = TileConfig::new(32, 16, 16, 2, 2);
const TILE_16_32_16_2_2: TileConfig = TileConfig::new(16, 32, 16, 2, 2);
const TILE_16_16_16_2_2: TileConfig = TileConfig::new(16, 16, 16, 2, 2);

/// Issue #383's tile-forcing arm: pin `select_tile_config`'s answer so the
/// TILE is the variable and `m` is held, which is the discriminator shape
/// §13.5a used for the GEMV/GEMM switch (`LLOOM_253_FORCE_GEMM`).
///
/// **Unset is what ships** and selects byte-for-byte what shipped, because the
/// forcing happens after every other branch would have returned.
///
/// It exists because `DESIGN.md` §13.5b and lloom #378 both dispositioned a
/// 32×16 tile **without one being instantiated** — #378 on the argument that
/// *"every LFM2 `N_out` is an exact multiple of 32, so BN=32 wastes nothing on
/// N"*. That is an argument about **padding**; llama.cpp #27441's stated
/// mechanism is **occupancy** (*"launches `ne01/64` threadgroups, which
/// under-fills the GPU at these sizes"*), and halving BN doubles the
/// threadgroup count along N. The two arguments do not meet, and until this
/// knob existed neither could be measured.
///
/// An unparseable or unknown value **panics** rather than falling back:
/// `DESIGN.md` §2.4 — an arm that silently ran the baseline is #69's vacuous
/// determinism run, and the whole point of this knob is to be A/B'd.
fn forced_tile() -> Option<TileConfig> {
    static TILE: std::sync::OnceLock<Option<TileConfig>> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| {
        crate::tuning::tuning()
            .gemm
            .tile
            .as_deref()
            .map(|v| match v {
                "32x32" => TILE_32_32_16_2_2,
                "32x16" => TILE_32_16_16_2_2,
                "16x32" => TILE_16_32_16_2_2,
                "16x16" => TILE_16_16_16_2_2,
                "64x32" => TILE_64_32_32_2_2,
                "64x64" => TILE_64_64_16_2_2,
                other => panic!(
                    "LLOOM_383_TILE must be one of \
                 32x32|32x16|16x32|16x16|64x32|64x64, got {other:?}"
                ),
            })
    })
}

/// The largest `m` `select_tile_config` serves with the 16-row tile
/// (`TILE_16_32_16_2_2`), inclusive. Measured cheapest of the three
/// instantiated row tiles at every `m` to 32 and at 33..48 for a prefill;
/// 64-row wins at 64 and 49..63 is unmeasured, so the band ends where the
/// canonical 34-token determinism prefill stays on its recorded tile
/// (§13.5o). On a batched decode step the 16-row tile is the slower tile at
/// `m` ∈ {40, 48} by a third (§13.5q), which is why the band is not wider.
pub const SKINNY_TILE_MAX_M: usize = 32;

/// The `m` below which `select_tile_config` takes the 16-row skinny tile.
///
/// `SKINNY_TILE_MAX_M + 1` is what ships (§13.5o). `LLOOM_364_SKINNY_MAX`
/// moves it for issue #364's threshold arm and is read **once**.
///
/// An unparseable value **panics** rather than falling back to the default:
/// `DESIGN.md` §2.4 — *"make an invalid setting panic rather than fall back
/// silently"* — because an arm that silently ran the baseline is #69's vacuous
/// determinism run, and the whole point of this knob is to be A/B'd.
fn skinny_max() -> usize {
    crate::tuning::tuning().gemm.skinny_max
}

/// Which side of the `m == 1 || n == 1` GEMV route to suppress, so the routing
/// decision itself becomes an arm (issue #386).
///
/// **Unset is what ships**, and the shipping path is byte-for-byte unchanged:
/// the suppression is read only inside that branch, so an unconfigured process
/// takes exactly the route it took before.
///
/// It rebuilds `LLOOM_253_FORCE_GEMM`, which `DESIGN.md` §13.5a's discriminator
/// used and which **exists on no branch** — `measurements/issue-253-raw/README.md`
/// records it as a temporary edit, restored from a hashed snapshot. §13.5a is
/// the precedent and this is the mechanism, made durable this time so the next
/// re-take costs an environment variable rather than four re-applied lines.
///
/// Two arms rather than one, because the route has two halves and only one has
/// ever been measured:
///
/// - `m` — route `m == 1` to the GEMM, holding `B` fixed so the batch is not a
///   variable. This is §13.5a's arm exactly. `n == 1` still takes the GEMV.
/// - `n` — route `n == 1` to the GEMM. **Never measured on any axis**: every
///   citation in the corpus is about `m`, and `n == 1` fires the same branch.
///
/// An unknown value **panics** rather than falling back: `DESIGN.md` §2.4 — an
/// arm that silently ran the baseline is #69's vacuous determinism run, and the
/// whole point of this knob is to be A/B'd. Engagement is nonetheless proved
/// from the **kernel census** rather than from the flag, per that same rule.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ForceGemm {
    M,
    N,
    Both,
}

fn forced_gemm() -> Option<ForceGemm> {
    match crate::tuning::tuning().gemm.force.as_deref() {
        Some("m") => Some(ForceGemm::M),
        Some("n") => Some(ForceGemm::N),
        Some("both") => Some(ForceGemm::Both),
        _ => None,
    }
}

/// Whether the GEMV route should still be taken for this shape.
///
/// Split out so the branch at the call site reads as the routing decision it is,
/// and so the shipping answer — *"`m == 1 || n == 1` takes the GEMV"* — is one
/// expression rather than a condition with a flag threaded through it.
fn takes_gemv_route(m: usize, n: usize) -> bool {
    let m_is_vec = m == 1;
    let n_is_vec = n == 1;
    let (suppress_m, suppress_n) = match forced_gemm() {
        None => (false, false),
        Some(ForceGemm::M) => (true, false),
        Some(ForceGemm::N) => (false, true),
        Some(ForceGemm::Both) => (true, true),
    };
    (m_is_vec && !suppress_m) || (n_is_vec && !suppress_n)
}

/// Select optimal tile configuration based on matrix dimensions, data type, transpose mode,
/// and device type.
///
/// This implements MLX's GEMM_TPARAM_MACRO tile selection logic.
/// Reference: refs/mlx/mlx/backend/metal/matmul.cpp lines 88-170
///
/// The selection is based on:
/// - Device type (phone/base-pro for small, ultra for large, others for medium)
/// - Total output size (batch_size * M * N)
/// - Data type (F32 vs F16/BF16)
/// - Transpose mode (nn, nt, tn, tt)
/// - K dimension relative to M and N
fn select_tile_config(
    dtype: GemmDType,
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    a_trans: bool,
    b_trans: bool,
    device_type: MetalDeviceType,
) -> TileConfig {
    // Issue #383's forcing arm, ahead of every other branch so the tile is
    // held while `m` varies. Unset (the shipping case) costs one `OnceLock`
    // read and changes nothing.
    if let Some(tile) = forced_tile() {
        return tile;
    }

    // Small M: the 16-row tile. A BM-row tile pads `m` to BM and a tiled GEMM
    // at small `m` is thread-bound (§13.5n), and two 16-row tiles still beat one
    // 32- or 64-row tile up to the band's end (§13.5o); the 64-row selection
    // below takes over above it. `LLOOM_364_SKINNY_MAX` moves the ceiling for
    // an A/B, read once (`OnceLock`); `LLOOM_383_TILE` pins the tile outright.
    if m < skinny_max() {
        return TILE_16_32_16_2_2;
    }

    // MLX uses batch_size * M * N >= 1M as the threshold for "large matmul"
    let total_output = batch_size * m * n;
    let is_large_matmul = total_output >= (1 << 20); // 1M elements

    match device_type {
        // Small devices: phone ('p') and base/pro ('g')
        MetalDeviceType::Phone | MetalDeviceType::BasePro => {
            // MLX: if (devc == 'g' || devc == 'p')
            if !a_trans && b_trans {
                // nt mode
                TILE_64_32_32_2_2
            } else if dtype != GemmDType::F32 {
                // half and bfloat
                TILE_64_64_16_1_2
            } else {
                // float32 default
                TILE_64_64_16_2_2
            }
        }
        // Large device: ultra ('d')
        MetalDeviceType::Ultra => {
            // MLX: if (devc == 'd')
            if is_large_matmul {
                // Large matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if 2 * m.max(n) > k {
                        // Reasonable K
                        TILE_64_64_16_1_2
                    } else if !a_trans && b_trans {
                        // nt with large K
                        TILE_64_32_32_2_2
                    } else {
                        // nn with large K
                        TILE_32_64_16_1_2
                    }
                } else {
                    // float32 takes default
                    TILE_64_64_16_2_2
                }
            } else {
                // Smaller matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if !a_trans && b_trans {
                        // nt
                        TILE_64_32_32_2_2
                    } else {
                        // nn
                        TILE_64_64_16_1_2
                    }
                } else {
                    // floats
                    if !a_trans && b_trans {
                        // nt
                        TILE_32_64_16_1_2
                    } else {
                        // nn
                        TILE_64_32_32_2_2
                    }
                }
            }
        }
        // Medium devices: max ('s') and unknown
        MetalDeviceType::Max | MetalDeviceType::Medium => {
            // MLX: default medium device config
            // Use the same logic as before but with medium device defaults
            match dtype {
                GemmDType::F32 => {
                    if !is_large_matmul {
                        if !a_trans && b_trans {
                            TILE_32_64_16_1_2
                        } else {
                            TILE_64_32_32_2_2
                        }
                    } else {
                        TILE_64_64_16_2_2
                    }
                }
                GemmDType::F16 | GemmDType::BF16 => {
                    if is_large_matmul {
                        if 2 * m.max(n) > k {
                            TILE_64_64_16_1_2
                        } else if !a_trans && b_trans {
                            TILE_64_32_32_2_2
                        } else {
                            TILE_32_64_16_1_2
                        }
                    } else if !a_trans && b_trans {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_64_64_16_1_2
                    }
                }
            }
        }
    }
}

/// Check if batch can be collapsed into M dimension.
///
/// MLX's batch collapse optimization (from matmul.cpp lines 700-740):
/// When B is broadcasted (2D), we can collapse batch into M dimension:
/// - [batch, M, K] @ [K, N] -> [batch*M, K] @ [K, N]
///
/// Conditions for batch collapse:
/// 1. batch_size > 1
/// 2. !transpose_a (A is not transposed, i.e., row-major for M dimension)
/// 3. A is contiguous in batch dimension (batch_stride_a == M * K)
/// 4. B is broadcasted (batch_stride_b == 0, meaning B is 2D)
///
/// Returns (effective_batch, effective_m, should_collapse)
fn check_batch_collapse(
    b: usize,
    m: usize,
    k: usize,
    a_trans: bool,
    lhs_stride: &[usize],
    rhs_stride: &[usize],
) -> (usize, usize, bool) {
    if b <= 1 {
        return (b, m, false);
    }

    // A must not be transposed for batch collapse
    if a_trans {
        return (b, m, false);
    }

    // Check A's batch stride - must be contiguous (batch_stride_a == M * K)
    let a_batch_stride = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };

    // Check B's batch stride - must be 0 (broadcasted) for collapse
    let b_batch_stride = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        0 // B is 2D, effectively broadcasted
    };

    // For batch collapse:
    // - A must be contiguous: batch_stride_a == M * K
    // - B must be broadcasted: batch_stride_b == 0
    let a_contiguous = a_batch_stride == m * k;
    let b_broadcasted = b_batch_stride == 0;

    if a_contiguous && b_broadcasted {
        // Collapse batch into M: new_m = batch * m, new_batch = 1
        (1, b * m, true)
    } else {
        (b, m, false)
    }
}

/// Check if we can use split-K strategy for better performance.
///
/// MLX uses split-K when:
/// - batch_size == 1
/// - (M/16) * (N/16) <= 32 (small output)
/// - K/16 >= 8 (large K)
///
/// This is useful for tall-skinny matrices where K >> M*N
#[allow(dead_code)]
fn should_use_split_k(b: usize, m: usize, n: usize, k: usize) -> bool {
    if b != 1 {
        return false;
    }
    let tm = m / 16;
    let tn = n / 16;
    let tk = k / 16;
    (tm * tn) <= 32 && tk >= 8
}

/// The fused RMSNorm prologue's two extra bindings (issue #266, `DESIGN.md`
/// §12.2 #2).
///
/// `scale` is a one-element `f32` per batch item holding `1/rms`, written by a
/// separate reduction dispatch. It is a binding rather than a packed-params
/// field because it is a *device* value the previous dispatch produced:
/// putting it in the block would mean reading it back to the host between the
/// two, which is the round trip the fusion exists to remove.
///
/// **The reduction cannot move into the prologue**, and that is arithmetic
/// rather than a preference. A GEMV partitions the output across `n_tgp`
/// threadgroups — 336 of them for LFM2's `w1` — and Metal has no
/// cross-threadgroup barrier inside a dispatch, so a prologue can only
/// *recompute* the whole `sum(x^2)`, once per threadgroup.
/// `measurements/issue-266-raw/prediction.py` prices that at 432x the
/// reduction work against a 0.492 MB/token saving.
#[derive(Debug, Clone, Copy)]
pub struct GemvRmsNorm<'a> {
    /// The norm's learned weight, `[K]`, indexed by position in the input vector.
    pub weight: &'a Buffer,
    pub weight_offset: usize,
    /// One `f32` per batch item: `1 / sqrt(mean(x^2) + eps)`.
    pub scale: &'a Buffer,
    pub scale_offset: usize,
}

/// RMSNorm computed INSIDE the GEMV (`gemv_norm*`, `docs/tuning.md` §13.5t):
/// the norm weight `[K]` and `eps`. Unlike [`GemvRmsNorm`] it imports no
/// scale, so the norm is no dispatch of its own: at the decode tiles each
/// simdgroup walks all of K and reduces `sum(x^2)` beside the dot product.
pub struct GemvNormInline<'a> {
    pub weight: &'a Buffer,
    pub weight_offset: usize,
    pub eps: f32,
}

/// Which input-side transform a GEMV runs.
enum GemvPrologue<'a> {
    None,
    RmsNorm(GemvRmsNorm<'a>),
    NormInline(GemvNormInline<'a>),
}

/// The elementwise transform fused onto a GEMV's OUTPUT (issue #484,
/// `DESIGN.md` §12.2 #3).
///
/// **This is not the mirror of [`GemvRmsNorm`] and the difference is the
/// entry's whole point.** A prologue sits inside the K-walk, which each of the
/// `n_tgp` threadgroups performs over the whole input vector, so its work is
/// multiplied by `n_tgp` *whatever it computes* (§12.2b). An epilogue sits on
/// the other side of that partition: `GEMVKernel::run` writes each output
/// element from exactly one thread, so the transform runs once per output
/// element — the standalone elementwise kernel's own count.
///
/// A pure elementwise transform therefore carries **no bindings at all**,
/// where `GemvRmsNorm` carries two. [`GemvEpilogue::Residual`] is the
/// exception and its one binding is what §12.2e's G1 gate is about — see
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GemvEpilogue {
    /// No transform. Resolves the classical `[[host_name]]`, so this is
    /// byte-for-byte the path that shipped.
    #[default]
    None,
    /// SiLU / swish, `x / (1 + exp(-x))`.
    ///
    /// **Computed at the accumulator's precision on the unrounded
    /// accumulator**, where the `silu_f16` dispatch it replaces computes at
    /// `T` on the already-rounded stored value. That is a more accurate order
    /// and a genuinely different computation, so this arm moves the digests by
    /// construction and owes §2.3.5a's discriminator rather than an
    /// unchanged-digest gate.
    Silu,
    /// The residual add, `x + residual[out_row]` (issue #509).
    ///
    /// **The one epilogue that carries a binding, and the arity argument is
    /// why it is admissible.** §12.2e's G1 requires the functor see one value;
    /// the second operand here is read at the storing thread's *own output
    /// position*, so each thread reads exactly the element it is about to
    /// write — no cross-thread data, no reduction. `kDoAxpby`'s
    /// `bias[(out_row + tm) * bias_stride]` is already that shape and already
    /// instantiated. §12.2b's `gate * up` is a full vector of the INPUT extent
    /// and is therefore multiplied by `n_tgp`: **a residual is indexable from
    /// the output position and a gate is not.**
    ///
    /// Like [`GemvEpilogue::Silu`] this adds at the accumulator's precision
    /// before the narrowing cast, where the `badd_f16` it replaces adds at `T`
    /// on the rounded store — so it moves the digests by construction.
    ///
    /// **Instantiated unconditionally since #516**, at the same two tiles per
    /// dtype as [`GemvEpilogue::Silu`]. It was behind
    /// `-DLLOOM_509_RESIDUAL_EPILOGUE` until then, which made a dispatch-tier
    /// choice depend on a compile-tier flag; the arm now resolves at one tier.
    Residual,
    /// The gate product, `x * gate[out_row]` (issue #732, `DESIGN.md` §12.2e
    /// row 3b at the producer end).
    ///
    /// The residual's binding and the residual's arity argument, with `*` for
    /// `+`: the operand is read at the storing thread's own output row. Row 3b
    /// failed G1 as a PROLOGUE on `w2`, where the gate is a full vector of the
    /// input extent; at `w3`'s epilogue `silu(w1·x)[i]` is row `i`'s own value.
    /// Replaces the `bmul_f16` of `silu(w1·x) * (w3·x)`, computing at the
    /// accumulator's precision on the unrounded up-projection — one rounding
    /// fewer, so it moves the digests by construction (§2.3.5a).
    Mul,
    /// The two-weight SwiGLU, `silu(gate·x) * (up·x)` — `mat` is the up
    /// weight and the operand at 15 the GATE WEIGHT, `[n, k]` like `mat`,
    /// streamed by the same threadgroup over the same input coefficients
    /// (issue #732, `DESIGN.md` §12.2 row 5 with row 3b's product). Not an
    /// epilogue on one accumulator: two accumulators per output row and no
    /// barrier between the two sweeps, which is what `Mul` could not have.
    /// One rounding at the store where the two-dispatch chain rounds twice.
    Swiglu,
}

impl GemvEpilogue {
    /// The `[[host_name]]` suffix, matching `gemv.metal`'s
    /// `instantiate_gemv_silu_helper` / `instantiate_gemv_residual_helper`.
    ///
    /// `pub` so the engine-side `Epilogue::SUFFIX` (`lloom::nn::fusion`) can be
    /// reconciled against it: the typed seam and the runtime enum must resolve
    /// the same `[[host_name]]` or a fused type would dispatch a kernel that
    /// does not exist.
    pub fn name_suffix(self) -> &'static str {
        match self {
            GemvEpilogue::None => "",
            GemvEpilogue::Silu => "_silu",
            GemvEpilogue::Residual => "_residual",
            GemvEpilogue::Mul => "_mul",
            GemvEpilogue::Swiglu => "_swiglu",
        }
    }

    /// Whether this arm reads a second operand at buffer slot 15.
    ///
    /// An exhaustive `match` rather than an inequality test, so a future arm
    /// is a **compile error** here rather than a silently unbound buffer.
    /// That is §7.1a-i's own repair: a two-armed conditional over a
    /// three-armed axis mislabelled every `RESULT` line since #116, and the
    /// fix was to read the resolved enum through a `match`.
    pub fn needs_operand(self) -> bool {
        match self {
            GemvEpilogue::None | GemvEpilogue::Silu => false,
            GemvEpilogue::Residual | GemvEpilogue::Mul | GemvEpilogue::Swiglu => true,
        }
    }
}

/// M=1 -> gemv_t (vec[K] x mat[K,N] -> vec[N])
/// N=1 -> gemv   (mat[M,K] x vec[K] -> vec[M])
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_mlx_gemv_with(
        device,
        ep,
        kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_mlx_gemv`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind — which is the property that makes the bit-identical test
/// meaningful. Same shape as `call_reduce_contiguous_with` (issue #38).
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    call_mlx_gemv_full(
        device,
        ep,
        kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
        style,
        None,
        GemvEpilogue::None,
    )
}

/// As [`call_mlx_gemv_with`], optionally fusing an RMSNorm prologue onto the
/// input vector (issue #266, `DESIGN.md` §12.2 #2).
///
/// `Some(..)` selects the `_rmsnorm` sibling, which scales each input element
/// by `1/rms` and the norm weight **as it is loaded into registers**, so the
/// normalized vector is never written to memory. `None` is byte-for-byte the
/// path that shipped: the `_rmsnorm` suffix is absent from the name, so the
/// same `[[host_name]]` resolves and the same pipeline runs.
///
/// One body serves both, per issue #38's discipline: the arms cannot disagree
/// about which values are bound or in what order, because only one argument
/// list exists.
///
/// **Refused rather than silently ignored** when the fused arm is asked for on
/// a path that has no `_rmsnorm` instantiation — the transposed kernel, or a
/// tile outside the two the decode path selects. A fused request that quietly
/// ran unfused would produce an un-normalized result, which §3.5 says nothing
/// reports.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    style: ParamStyle,
    fused_norm: Option<GemvRmsNorm<'_>>,
    epilogue: GemvEpilogue,
) -> Result<(), MetalKernelError> {
    call_mlx_gemv_epilogue(
        device,
        ep,
        kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
        style,
        fused_norm,
        epilogue,
        None,
    )
}

/// As [`call_mlx_gemv_full`], supplying the second operand an epilogue that
/// needs one reads (issue #509).
///
/// A **sibling rather than a widened signature**, deliberately: nine callers
/// pass the existing argument list and none of them has an operand to pass, so
/// widening it would edit nine sites to write `None` and gain nothing.
/// [`call_mlx_gemv_full`] delegates here, so there is still **one body** and
/// the two cannot drift in what they bind — issue #38's discipline, which is
/// the property that makes the bit-identical test meaningful.
///
/// The operand is bound at slot **15**, after every argument the classical
/// signature declares, exactly as `gemv_rmsnorm`'s two are at 15 and 16 — so
/// the unfused arm's slot numbering is untouched.
///
/// **Refused rather than silently ignored** in two ways, each because a
/// request that quietly ran the wrong kernel produces a well-formed wrong
/// number that §3.5 says nothing reports: an epilogue needing an operand with
/// none supplied, and an operand supplied to an epilogue that reads none.
///
/// A third case used to sit here — a `_residual` name absent from the metallib
/// because `-DLLOOM_509_RESIDUAL_EPILOGUE` was not on the compile line — and is
/// gone as of #516, which instantiates the arm unconditionally. That case was
/// never refused *here* anyway: it surfaced two frames down as a
/// `newFunctionWithName:` failure naming a mangled kernel, which is the
/// illegibility #516 removes rather than relabels.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_epilogue(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    mnk: (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    style: ParamStyle,
    fused_norm: Option<GemvRmsNorm<'_>>,
    epilogue: GemvEpilogue,
    epilogue_operand: Option<(&Buffer, usize)>,
) -> Result<(), MetalKernelError> {
    let prologue = match fused_norm {
        Some(norm) => GemvPrologue::RmsNorm(norm),
        None => GemvPrologue::None,
    };
    gemv_dispatch(
        device,
        ep,
        kernels,
        dtype,
        mnk,
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
        style,
        prologue,
        epilogue,
        epilogue_operand,
    )
}

/// A GEMV with its RMSNorm computed inside it ([`GemvNormInline`]): `out =
/// epilogue((W (x * norm.w)) * rsqrt(mean(x^2) + eps))`, one dispatch where the
/// unfused pair is an `rmsnorm` and a GEMV. `GemvEpilogue::None`, `Silu` and
/// `Swiglu` are instantiated (`Swiglu`: `rhs` is the UP weight and
/// `epilogue_operand` the GATE weight, both sweeps normalized,
/// `gemv_norm_swiglu`), at the non-transposed `bm4`/`bm8` decode tiles,
/// `ParamStyle::Split`; anything else is refused.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    mnk: (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    norm: GemvNormInline<'_>,
    epilogue: GemvEpilogue,
    epilogue_operand: Option<(&Buffer, usize)>,
) -> Result<(), MetalKernelError> {
    gemv_dispatch(
        device,
        ep,
        kernels,
        dtype,
        mnk,
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
        ParamStyle::Split,
        GemvPrologue::NormInline(norm),
        epilogue,
        epilogue_operand,
    )
}

#[allow(clippy::too_many_arguments)]
fn gemv_dispatch(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    style: ParamStyle,
    prologue: GemvPrologue<'_>,
    epilogue: GemvEpilogue,
    epilogue_operand: Option<(&Buffer, usize)>,
) -> Result<(), MetalKernelError> {
    debug_assert!(m == 1 || n == 1, "call_mlx_gemv requires M=1 or N=1");

    // Both directions checked, not just the missing one. An operand passed to
    // an arm that does not read it would be silently dropped, and the caller
    // would believe a fusion ran that did not — §2.4's "an instrument that
    // cannot be shown to have engaged has measured nothing", at the binding.
    match (epilogue.needs_operand(), epilogue_operand.is_some()) {
        (true, false) => {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv epilogue {epilogue:?} reads a second operand and none \
                 was supplied (issue #509)"
            )))
        }
        (false, true) => {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv epilogue {epilogue:?} reads no second operand but one \
                 was supplied; it would be silently unused (issue #509)"
            )))
        }
        _ => {}
    }

    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);

    // Determine transpose flags from strides (same logic as call_mlx_gemm)
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    // Figure out if transpose is needed.
    let is_b_matrix = n != 1;
    let transpose_mat = if is_b_matrix { !b_trans } else { a_trans };
    let mat_ld = if is_b_matrix {
        ldb as usize
    } else {
        lda as usize
    };
    let in_vec_size = k;
    let out_vec_size = if is_b_matrix { n } else { m };

    let (mat_buffer, mat_offset, vec_buffer, vec_offset) = if is_b_matrix {
        (rhs_buffer, rhs_offset, lhs_buffer, lhs_offset)
    } else {
        (lhs_buffer, lhs_offset, rhs_buffer, rhs_offset)
    };

    // Batch strides (elements per batch item)
    let vec_batch_stride: i64 = if is_b_matrix {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    } else {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    };
    // Weight matrix is often 2D (shared across batch) -> stride = 0
    let mat_batch_stride: i64 = if is_b_matrix {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            0
        }
    } else {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            0
        }
    };

    // Tile selection
    let (bm, bn, sm, sn, tm, tn) = if transpose_mat {
        // gemv_t: vec[K] x mat[K,N_out] -> out[N_out]
        let (sm, sn) = if in_vec_size >= 8192 && out_vec_size >= 2048 {
            (4usize, 8usize)
        } else {
            (8, 4)
        };
        let bn = if out_vec_size >= 2048 {
            16usize
        } else if out_vec_size >= 512 {
            4
        } else {
            2
        };
        let tn: usize = if out_vec_size < 4 { 1 } else { 4 };
        (1usize, bn, sm, sn, 4usize, tn)
    } else {
        // gemv: mat[M_out,K] x vec[K] -> out[M_out]
        gemv_tile_dims(out_vec_size, in_vec_size)
    };

    let dtype_str = match dtype {
        GemmDType::F32 => "float32",
        GemmDType::F16 => "float16",
        GemmDType::BF16 => "bfloat16",
    };
    let kernel_prefix = if transpose_mat { "gemv_t" } else { "gemv" };

    // The fused arm is instantiated only for the non-transposed kernel at the
    // two tiles the decode path selects (`gemv.metal`'s
    // `instantiate_gemv_rmsnorm_blocks`), and only at `nc0_axpby0`. Anything
    // else is REFUSED rather than silently falling back to the unfused name:
    // a fused request that ran unfused would compute an un-normalized result,
    // and under `HazardTrackingModeUntracked` (§3.5) nothing would report it.
    let decode_tile = !transpose_mat
        && bn == 1
        && sm == 1
        && sn == 32
        && tn == 4
        && tm == 4
        && (bm == 4 || bm == 8);
    if let GemvPrologue::NormInline(_) = &prologue {
        if !decode_tile || style != ParamStyle::Split {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv inline norm is instantiated only at the non-transposed bm4/bm8 decode \
                 tiles, split params; asked for {kernel_prefix} \
                 bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn} {style:?} (mnk={m}x{n}x{k})"
            )));
        }
        if !matches!(
            epilogue,
            GemvEpilogue::None | GemvEpilogue::Silu | GemvEpilogue::Swiglu
        ) {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv inline norm composes with no epilogue, SiLU or the two-weight SwiGLU \
                 only, not {epilogue:?}"
            )));
        }
    }
    let norm_suffix = if let GemvPrologue::RmsNorm(_) = &prologue {
        let tile_is_instantiated = !transpose_mat
            && bn == 1
            && sm == 1
            && sn == 32
            && tn == 4
            && tm == 4
            && (bm == 4 || bm == 8);
        if !tile_is_instantiated {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv fused rmsnorm prologue is not instantiated for \
                 {kernel_prefix} bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn} \
                 (mnk={m}x{n}x{k}); only the non-transposed bm4/bm8 decode \
                 tiles carry it -- see DESIGN.md 12.2 #2, issue #266"
            )));
        }
        if style != ParamStyle::Split {
            return Err(MetalKernelError::LoadFunctionError(
                "gemv fused rmsnorm prologue has no packed sibling; the two \
                 extra bindings are device buffers rather than scalars, so \
                 there is nothing for a params block to carry (issue #266)"
                    .to_string(),
            ));
        }
        "_rmsnorm"
    } else if let GemvPrologue::NormInline(_) = &prologue {
        "_norm"
    } else {
        ""
    };

    // The epilogue arm is instantiated only for the non-transposed kernel at
    // the two tiles `instantiate_gemv_silu_blocks` names, and only at
    // `nc0_axpby0` — the same scope and the same arithmetic as the prologue
    // above (§11.3g's +730 ms / +4.01 MiB for doubling this family).
    //
    // REFUSED rather than silently falling back, for the reason the prologue
    // gives: a fused request that ran unfused would write un-transformed
    // values, and under `HazardTrackingModeUntracked` (§3.5) nothing reports
    // a wrong number that is well-formed.
    //
    // The two fused arms are also refused TOGETHER. They are composable in
    // principle — a prologue and an epilogue touch different ends of the same
    // body — but no `_rmsnorm_silu` name is instantiated, so asking for both
    // would resolve a name that does not exist. Failing here says which
    // combination is missing; `load_pipeline` would only say the name is
    // absent.
    let epilogue_suffix = if epilogue != GemvEpilogue::None {
        if norm_suffix == "_rmsnorm" {
            return Err(MetalKernelError::LoadFunctionError(
                "gemv fused prologue and epilogue cannot be combined: no \
                 `_rmsnorm_silu` variant is instantiated (issue #484, \
                 DESIGN.md 12.2 #3). Each is available alone."
                    .to_string(),
            ));
        }
        let tile_is_instantiated = !transpose_mat
            && bn == 1
            && sm == 1
            && sn == 32
            && tn == 4
            && tm == 4
            && (bm == 4 || bm == 8);
        if !tile_is_instantiated {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "gemv fused epilogue is not instantiated for \
                 {kernel_prefix} bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn} \
                 (mnk={m}x{n}x{k}); only the non-transposed bm4/bm8 decode \
                 tiles carry it -- see DESIGN.md 12.2 #3, issue #484"
            )));
        }
        // Both styles: the packed siblings (`gemv_silu_packed`,
        // `gemv_residual_packed`) read the same params block `gemv_packed`
        // does, so the name is the classical one with `_packed` appended, as
        // `packed_name` would build it. Without them the fused arms were the largest
        // class an ICB replay could not cover (§3.7c: no inline constants in
        // a command) — 90 of 345 decode positions (issue #732 P3).
        epilogue.name_suffix()
    } else {
        ""
    };

    // The style segment LAST: `packed_name` appends `_packed` to a finished
    // classical name everywhere else, and the replay executor classifies a
    // position by that trailing suffix (`icb.rs` `is_packed`), so a fused
    // packed name is `…_axpby0_silu_packed`, never `…_packed_silu`.
    let name = format!(
        "{}_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc0_axpby0{}{}{}",
        kernel_prefix,
        dtype_str,
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
        norm_suffix,
        epilogue_suffix,
        style.name_suffix()
    );

    let pipeline = kernels.load_pipeline(device, Source::Gemv, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let batch_shape = [b as i32];
    let vec_batch_strides = [vec_batch_stride];
    let mat_batch_strides = [mat_batch_stride];
    let bias_batch_strides = [0i64];

    // The packed block is built by letting this *same* `set_params!` run and
    // diverting each scalar as it passes (`EncoderParam::set_param`), so the two
    // styles cannot disagree about which values are bound or in what order:
    // only one argument list exists. A hand-written packing struct beside this
    // call would be two declarations of one thing, which is the hand-sync
    // `DESIGN.md` §8.1b exists to remove.
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            Input::with_offset(mat_buffer, mat_offset),
            Input::with_offset(vec_buffer, vec_offset),
            (), // bias
            Output::new(output),
            in_vec_size as i32,
            out_vec_size as i32,
            mat_ld as i32,
            1.0f32, // alpha
            0.0f32, // beta
            1i32,   // batch_ndim
            &batch_shape[..],
            &vec_batch_strides[..],
            &mat_batch_strides[..],
            &bias_batch_strides[..],
            1i32 // bias_stride
        )
    );
    // Held until after the dispatch is encoded: the params block and any array
    // promoted out of `setBytes` must outlive it.
    let _staged = finish_packed_params(device, encoder, style, GEMV_PARAMS_ALIGN)?;

    // The fused prologue's two bindings, at 15 and 16 -- after every argument
    // the classical signature declares, so the classical slot numbering is
    // untouched and the unfused arm binds exactly what it bound before.
    //
    // They go in AFTER `finish_packed_params` deliberately: under
    // `ParamStyle::Packed` the capture renumbers the slots it diverted, and
    // these are not part of that argument list. The fused arm is refused for
    // `Packed` above, so the two mechanisms never meet -- stated here because
    // the ordering looks incidental and is not (issue #41's `()` hazard, in
    // the other direction).
    // `set_input_buffer` rather than a raw bind: both are READS, and that
    // function is what records the binding for hazard tracking and consults
    // `prev_ce_outputs` for a pending writer (§6.4). The scale buffer is
    // written by the reduction dispatch immediately before this one, so the
    // RAW edge between them is exactly what must not be dropped -- and under
    // `HazardTrackingModeUntracked` a missed one is silent corruption (§3.5).
    match &prologue {
        GemvPrologue::RmsNorm(norm) => {
            encoder.set_input_buffer(15, Some(norm.weight), norm.weight_offset);
            encoder.set_input_buffer(16, Some(norm.scale), norm.scale_offset);
        }
        // After the epilogue's operand when it reads one (`gemv_norm_swiglu`:
        // the gate weight at 15, the norm at 16 and 17), else at 15 and 16.
        GemvPrologue::NormInline(norm) => {
            let at = if epilogue.needs_operand() { 16 } else { 15 };
            encoder.set_input_buffer(at, Some(norm.weight), norm.weight_offset);
            encoder.set_bytes(at + 1, &norm.eps);
        }
        GemvPrologue::None => {}
    }

    // The fused epilogue's one binding, also at 15 -- the `_rmsnorm` prologue
    // and an epilogue are refused together above, and the inline norm binds
    // past it when both are present, so nothing else is bound there. The same index under
    // both styles: the packed sibling declares `residual` at 15, past every
    // slot the packed layout uses, and this bind happens after the capture
    // has closed, so it is not renumbered. `set_input_buffer`
    // for the reason the prologue's comment gives, and here the RAW edge is
    // the one the whole issue is about: the residual is written by the layer's
    // previous op, and dropping that edge is silent corruption under
    // `HazardTrackingModeUntracked` (§3.5).
    if let Some((buf, offset)) = epilogue_operand {
        encoder.set_input_buffer(15, Some(buf), offset);
    }

    let n_out_per_tgp = if transpose_mat {
        bn * sn * tn
    } else {
        bm * sm * tm
    };
    let n_tgp = out_vec_size.div_ceil(n_out_per_tgp);
    let grid_size = MTLSize {
        width: n_tgp,
        height: 1,
        depth: b,
    };
    let group_size = MTLSize {
        width: 32,
        height: bn,
        depth: bm,
    };
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

/// The epilogue of a [`call_mlx_gemv_mixed`] dispatch (`gemv.metal`'s
/// `EpilogueSiluScale` / `EpilogueSigmoidScale`): `silu(scale · y)` or
/// `sigmoid(scale · y)` on the f32 accumulator at the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MixedEpilogue {
    #[default]
    None,
    Silu,
    Sigmoid,
}

impl MixedEpilogue {
    pub fn name_suffix(self) -> &'static str {
        match self {
            MixedEpilogue::None => "",
            MixedEpilogue::Silu => "_silu",
            MixedEpilogue::Sigmoid => "_sigmoid",
        }
    }
}

/// `out[b, :] = epilogue(scale · mat · vec[b, :])` for `b` f32 vectors of `k`
/// against ONE `[n_out, k]` row-major **f16** matrix, f32 out — the
/// `gemv_mixed_f16w_f32_*` arms: the f32 activation read as it lies, the f32
/// result stored, f32 accumulation, no cast launch on either side (the
/// classical route cast to f16, ran `gemv_float16` and cast back — three
/// launches for one). The tile is `derive_gemv_tile`'s, as the classical
/// non-transposed path picks it; `ParamStyle::Split` only.
///
/// `vec` is `b` contiguous rows of `k` (`vec_batch_stride` elements apart);
/// `mat` is shared across the batch.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_mixed(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bnk: (usize, usize, usize),
    mat_buffer: &Buffer,
    mat_offset: usize,
    vec_buffer: &Buffer,
    vec_offset: usize,
    vec_batch_stride: usize,
    output: &Buffer,
    epilogue: MixedEpilogue,
    scale: f32,
) -> Result<(), MetalKernelError> {
    gemv_mixed_dispatch(
        device,
        ep,
        kernels,
        bnk,
        mat_buffer,
        mat_offset,
        vec_buffer,
        vec_offset,
        vec_batch_stride,
        output,
        epilogue,
        scale,
        None,
    )
}

/// The grouped RMSNorm a [`call_mlx_gemv_mixed_gnorm`] computes inside the
/// GEMV (`gemv.metal`'s `PrologueGroupNorm`): `vec` normalized per group of
/// `group` elements (`groups` of them, `group · groups == k`), then scaled by
/// the f32 `weight` `[k]`. With `scales`, the per-row group scales
/// (`rsqrt(mean(x_g^2) + eps)`, `b · groups` f32) are written there too.
pub struct GemvGroupNorm<'a> {
    pub weight: &'a Buffer,
    pub weight_offset: usize,
    pub eps: f32,
    pub group: usize,
    pub groups: usize,
    pub scales: Option<(&'a Buffer, usize)>,
}

/// [`call_mlx_gemv_mixed`] with a grouped RMSNorm of `vec` computed inside it
/// ([`GemvGroupNorm`]): `out = epilogue(scale · mat · norm(vec))`, one launch
/// where the composed route is the norm's launch and the GEMV. Refused where
/// [`gemv_mixed_gnorm_serves`] is false, and for the sigmoid epilogue (not
/// instantiated).
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv_mixed_gnorm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    bnk: (usize, usize, usize),
    mat_buffer: &Buffer,
    mat_offset: usize,
    vec_buffer: &Buffer,
    vec_offset: usize,
    vec_batch_stride: usize,
    output: &Buffer,
    epilogue: MixedEpilogue,
    scale: f32,
    norm: GemvGroupNorm<'_>,
) -> Result<(), MetalKernelError> {
    gemv_mixed_dispatch(
        device,
        ep,
        kernels,
        bnk,
        mat_buffer,
        mat_offset,
        vec_buffer,
        vec_offset,
        vec_batch_stride,
        output,
        epilogue,
        scale,
        Some(norm),
    )
}

#[allow(clippy::too_many_arguments)]
fn gemv_mixed_dispatch(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, n_out, k): (usize, usize, usize),
    mat_buffer: &Buffer,
    mat_offset: usize,
    vec_buffer: &Buffer,
    vec_offset: usize,
    vec_batch_stride: usize,
    output: &Buffer,
    epilogue: MixedEpilogue,
    scale: f32,
    norm: Option<GemvGroupNorm<'_>>,
) -> Result<(), MetalKernelError> {
    let (bm, bn, sm, sn) = derive_gemv_tile(n_out, k);
    let tm: usize = if n_out < 4 { 1 } else { 4 };
    let tn: usize = 4;
    let gnorm_suffix = match &norm {
        None => "",
        Some(g) => {
            if !gemv_mixed_gnorm_serves(n_out, k, g.groups) || g.group * g.groups != k {
                return Err(MetalKernelError::LoadFunctionError(format!(
                    "gemv_mixed grouped norm: not instantiated for [{n_out}, {k}] in {} groups \
                     of {} (tile bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}; only bm1_bn8_sm1_sn32_tm4)",
                    g.groups, g.group
                )));
            }
            if epilogue == MixedEpilogue::Sigmoid {
                return Err(MetalKernelError::LoadFunctionError(
                    "gemv_mixed grouped norm: no sigmoid variant is instantiated".to_string(),
                ));
            }
            "_gnorm"
        }
    };
    let name = format!(
        "gemv_mixed_f16w_f32_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc0_axpby0{}{gnorm_suffix}",
        epilogue.name_suffix()
    );
    let pipeline = kernels.load_pipeline(device, Source::Gemv, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let batch_shape = [b as i32];
    let vec_batch_strides = [vec_batch_stride as i64];
    let mat_batch_strides = [0i64];
    let bias_batch_strides = [0i64];
    set_params!(
        encoder,
        (
            Input::with_offset(mat_buffer, mat_offset),
            Input::with_offset(vec_buffer, vec_offset),
            (), // bias
            Output::new(output),
            k as i32,
            n_out as i32,
            k as i32, // matrix_ld: row-major [n_out, k]
            scale,    // alpha: the epilogue's pre-scale
            0.0f32,   // beta
            1i32,     // batch_ndim
            &batch_shape[..],
            &vec_batch_strides[..],
            &mat_batch_strides[..],
            &bias_batch_strides[..],
            1i32 // bias_stride
        )
    );
    // The grouped norm's bindings, after every slot the classical layout
    // declares (`gemv_mixed_gnorm_wrapper`). Reads through `set_input_buffer`
    // (the hazard edge to the input's producer), the scales through
    // `set_output_buffer` (a later consumer's edge to this dispatch); with no
    // scales asked for, the output is bound at 19 and never written there.
    if let Some(g) = &norm {
        encoder.set_input_buffer(15, Some(g.weight), g.weight_offset);
        encoder.set_bytes(16, &g.eps);
        encoder.set_bytes(17, &(g.group as i32));
        encoder.set_bytes(18, &(g.groups as i32));
        match g.scales {
            Some((buf, off)) => encoder.set_output_buffer(19, Some(buf), off),
            None => encoder.set_output_buffer(19, Some(output), 0),
        }
        encoder.set_bytes(20, &(g.scales.is_some() as i32));
    }
    let n_out_per_tgp = bm * sm * tm;
    let n_tgp = n_out.div_ceil(n_out_per_tgp);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n_tgp,
            height: 1,
            depth: b.max(1),
        },
        MTLSize {
            width: 32,
            height: bn,
            depth: bm,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_mlx_gemm_out(
        device,
        ep,
        kernels,
        dtype,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        output,
    )
}

/// [`call_mlx_gemm`] with the output in `out_dtype`: the operands `dtype`,
/// the accumulation float, the store converted once per element (the steel
/// kernel's `OutT`). Instantiated for `(F16, F32)` — half operands, a float
/// result, the dequantise-once stems' route (issue #749 P26) — beside the
/// same-type arms. A mixed pair never takes the gemv route.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm_out(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    out_dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    if dtype == GemmDType::F16
        && out_dtype == dtype
        && !crate::tuning::contract().is_fast()
        && forced_gemm().is_some()
    {
        return Err(MetalKernelError::InvalidInput(
            "--force-gemm conflicts with deterministic f16 row identity; use --determinism fast for the forcing experiment, or unset --force-gemm".into()));
    }

    // #942: deterministic half output uses the existing one-row reduction
    // at every row count. Rows occupy GEMV's batch axis, not a new core.
    // The storage caller materializes a transposed activation first. Weight
    // strides and both input offsets remain those of the caller's views.
    if dtype == GemmDType::F16
        && out_dtype == dtype
        && !crate::tuning::contract().is_fast()
        && !a_trans
        && m > 1
    {
        let a_batch = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3]
        } else {
            m * k
        };
        let b_batch = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3]
        } else {
            n * k
        };
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        for batch in 0..b {
            let out = output.view(batch * m * n * 2, m * n * 2);
            call_mlx_gemv(
                device,
                encoder,
                kernels,
                dtype,
                (m, 1, n, k),
                &[k, k, 1],
                lhs_offset + batch * a_batch * 2,
                lhs_buffer,
                &[0, rhs_m2, rhs_m1],
                rhs_offset + batch * b_batch * 2,
                rhs_buffer,
                &out,
            )?;
        }
        return Ok(());
    }

    // The routing decision this project has never audited (issue #386): no
    // threshold, no config axis, and `select_tile_config` is never consulted.
    // `takes_gemv_route` is `m == 1 || n == 1` unless `LLOOM_386_FORCE_GEMM`
    // suppresses one half, which is the only way to build the other arm.
    if out_dtype == dtype && takes_gemv_route(m, n) {
        return call_mlx_gemv(
            device,
            ep,
            kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            lhs_offset,
            lhs_buffer,
            rhs_stride,
            rhs_offset,
            rhs_buffer,
            output,
        );
    }

    // Check for batch collapse optimization (MLX matmul.cpp lines 700-740)
    // When B is broadcasted (2D), collapse batch into M dimension
    let (effective_batch, effective_m, batch_collapsed) =
        check_batch_collapse(b, m, k, a_trans, lhs_stride, rhs_stride);

    // Use effective dimensions after potential batch collapse
    let m = effective_m;
    let b = effective_batch;

    // Dynamic tile selection based on matrix dimensions, dtype, transpose mode, and device type
    // Reference: MLX GEMM_TPARAM_MACRO in matmul.cpp
    let device_type = device.device_type();
    let tile = select_tile_config(dtype, m, n, k, b, a_trans, b_trans, device_type);
    let (bm, bn, bk, wm, wn) = (tile.bm, tile.bn, tile.bk, tile.wm, tile.wn);

    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    // has_batch should be true when b > 1, matching the original candle behavior
    let has_batch = b > 1;

    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(has_batch)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile_swizzle = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile_swizzle;
    let tm = tm.div_ceil(tile_swizzle);

    // Calculate batch strides based on whether batch was collapsed
    let (batch_stride_a, batch_stride_b) = if batch_collapsed {
        // After batch collapse, there's no batch dimension
        (0isize, 0isize)
    } else {
        let a_stride = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as isize
        } else {
            (m * k) as isize
        };
        let b_stride = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as isize
        } else {
            (n * k) as isize
        };
        (a_stride, b_stride)
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if batch_collapsed { k as i32 } else { lda }, // After collapse, lda = K
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    // Dynamically generate kernel name based on dtype, transpose mode, and tile config
    // Format: gemm_{trans}_{itype}_{otype}_{bm}_{bn}_{bk}_{wm}_{wn}
    let dtype_name = |d: GemmDType| match d {
        GemmDType::F32 => "f32",
        GemmDType::F16 => "f16",
        GemmDType::BF16 => "bf16",
    };
    let dtype_str = dtype_name(dtype);
    let out_str = dtype_name(out_dtype);
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, out_str, bm, bn, bk, wm, wn
    );

    let pipeline =
        kernels.load_pipeline_with_constants(device, Source::Gemm, name.clone(), constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "mlx_gemm {name} B={b} M={m} N={n} K={k}");

    impl EncoderParam for GemmParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    // Batch strides for buffer 7 (same as main branch)
    let batch_strides = [batch_stride_a, batch_stride_b];

    set_params!(
        encoder,
        (
            (lhs_buffer, lhs_offset),
            (rhs_buffer, rhs_offset),
            (),
            Output::new(output),
            gemm_params,
            (),
            b as i32,
            &batch_strides[..]
        )
    );

    let grid_size = MTLSize {
        width: tn,
        height: tm,
        depth: /* batch_size_out */ b,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

/// The gate `DESIGN.md` §13.5b names and did not have — issue #380.
///
/// §13.5b's re-measure trigger reads *"any change to `select_tile_config` or the
/// `m == 1` routing at `mlx_gemm.rs:705`, which `occupancy.py` turns into a
/// failing check"*. Measured, it does not: `occupancy.py` carries that rule as a
/// **comment** and re-derives the tile split from its own copy of the
/// arithmetic, so it validates a model of this function rather than this
/// function. #364 edited exactly this function and that script exited 0.
///
/// These tests exercise `select_tile_config` **itself**. They are the second
/// gate §15.1a prescribes for its COARSE class — the script's own arithmetic is
/// correct and stays, and what was missing is a check keyed on the claim it is
/// cited for.
///
/// **No device, no GPU, no lease.** `select_tile_config` takes `MetalDeviceType`
/// as a *parameter* rather than a `&Device`, so the whole selection is a pure
/// function of `(dtype, m, n, k, batch, a_trans, b_trans, device_type)` and the
/// tests below are arithmetic that happens to be written in Rust.
///
/// **What each test asserts is stated in its own name and doc comment**, per
/// #380's third requirement: a gate that does not say which claim it carries is
/// one whose failure nobody can read.
#[cfg(test)]
mod tile_config_gate {
    use super::*;

    /// M1 Max — `architecture_name()` ends in `'s'` (`metal/device.rs:262`), and
    /// it is the machine every figure in `DESIGN.md` §13.5a/§13.5b/§13.5c was
    /// taken on. Passed as a value rather than read from a device, which is what
    /// keeps these tests GPU-free.
    const DEV: MetalDeviceType = MetalDeviceType::Max;

    /// LFM2's 167 weight matmuls, `DESIGN.md` §5.4 and §6.6.
    ///
    /// `(label, count, n = N_out, k = K)`. Every decode matmul is `nt`
    /// (`a_trans = false`, `b_trans = true`) — a `[m, K]` activation against a
    /// `[N_out, K]` weight — which is what `call_mlx_gemm`'s stride analysis
    /// resolves them to and what §13.5b's branch reading depends on.
    const SITES: &[(&str, usize, usize, usize)] = &[
        ("conv.in_proj", 22, 6144, 2048),
        ("conv.out_proj", 22, 2048, 2048),
        ("w1 (gate)", 30, 10752, 2048),
        ("w3 (up)", 30, 10752, 2048),
        ("w2 (down)", 30, 2048, 10752),
        ("q_proj", 8, 2048, 2048),
        ("k_proj", 8, 512, 2048),
        ("v_proj", 8, 512, 2048),
        ("out_proj", 8, 2048, 2048),
        ("lm_head", 1, 128000, 2048),
    ];

    fn tile_at(m: usize, n: usize, k: usize) -> TileConfig {
        select_tile_config(
            GemmDType::F16,
            m,
            n,
            k,
            /* batch_size */ 1,
            /* a_trans */ false,
            /* b_trans */ true,
            DEV,
        )
    }

    /// The site table is LFM2's, checked against `DESIGN.md` §6.6's own count.
    ///
    /// Not a claim about `select_tile_config`; it is the fixture's own
    /// non-vacuity check (§15.1a). A table that had drifted from the model would
    /// make every assertion below true of a workload we do not run.
    #[test]
    fn the_site_table_is_lfm2s_167_weight_matmuls() {
        let total: usize = SITES.iter().map(|(_, c, _, _)| c).sum();
        assert_eq!(
            total, 167,
            "DESIGN.md §6.6 counts 167 GEMVs per decode step (22*2 + 8*4 + 30*3 + 1)"
        );
    }

    /// **The claim: `select_tile_config` returns the 16×32 tile at every
    /// `m` in 1..=32.**
    ///
    /// `DESIGN.md` §13.5n measured the instantiated 16-row tile a third cheaper
    /// than the 32×32 one at every `m` in 2..15 (batched decode) and on the
    /// `m = 10` DSpark verify; §13.5o measured it against both wider tiles on
    /// prefill at every `m` to 32 and closes the band there. The contract is
    /// read here from the function rather than from a comment beside it.
    ///
    /// Every one of LFM2's 167 weight matmuls is asserted rather than a
    /// representative, because the claim is quantified over all 167 and the
    /// sites differ in `N_out` by 250× (512 to 128000).
    #[test]
    fn up_to_thirty_two_every_lfm2_matmul_gets_the_16_32_tile() {
        for m in 1..=SKINNY_TILE_MAX_M {
            for (label, _, n, k) in SITES {
                let t = tile_at(m, *n, *k);
                assert_eq!(
                    (t.bm, t.bn, t.bk, t.wm, t.wn),
                    (16, 32, 16, 2, 2),
                    "§13.5o: at m={m} the {label} matmul ({n}x{k}) must take \
                     TILE_16_32_16_2_2; got {t:?}"
                );
            }
        }
    }

    /// **The claim: the threshold is exactly at 32/33, and crossing it leaves
    /// BM=16 for the 64-row tile.**
    ///
    /// `DESIGN.md` §13.5c pinned the old `m < 16` boundary by bisection and
    /// measured it 2× too low; §13.5o moves it to 32, the last `m` the sweep
    /// resolved the 16-row tile cheapest at without moving the canonical
    /// 34-token determinism prefill. **Both sides are asserted** (§8.1g): a test
    /// that only checked the low side would pass on a function that returned the
    /// 16×32 tile everywhere, which is precisely the `LLOOM_364_SKINNY_MAX` and
    /// `LLOOM_383_TILE` arms. Above the threshold the selection is exactly as
    /// #364 left it.
    #[test]
    fn the_threshold_is_at_thirty_two_and_the_tile_changes_across_it() {
        let (n, k) = (2048, 2048);
        let shape = |t: TileConfig| (t.bm, t.bn, t.bk, t.wm, t.wn);
        // The shipped ceiling is `SKINNY_TILE_MAX_M + 1`: #148 had raised it
        // to 64 for the batch-collapsed decode `m`, and #694 measured that
        // arm 30 % slower on the batched step at B = 40 and 48 (§13.5q), so
        // the band is the prefill sweep's again and the canonical 34-token
        // prefill stays on the tile its digests were taken on.
        assert_eq!(
            shape(tile_at(16, n, k)),
            (16, 32, 16, 2, 2),
            "m=16 (§13.5c's old cliff)"
        );
        assert_eq!(
            shape(tile_at(32, n, k)),
            (16, 32, 16, 2, 2),
            "m=32 is the last m on 16x32 (SKINNY_TILE_MAX_M)"
        );
        assert_eq!(
            shape(tile_at(33, n, k)),
            (64, 32, 32, 2, 2),
            "m=33 takes TILE_64_32_32_2_2, #364's selection above the band"
        );
        assert_eq!(
            shape(tile_at(34, n, k)),
            (64, 32, 32, 2, 2),
            "§2.3.8c's 34-token canonical prefill stays on its recorded tile"
        );
        assert_ne!(
            shape(tile_at(32, n, k)),
            shape(tile_at(33, n, k)),
            "the boundary must be a boundary: if these agree, every §13.5c/§13.5o \
             figure is a comparison of a tile against itself"
        );
    }

    /// **The claim: `align_M` is false and `align_N` is true at all 167 sites.**
    ///
    /// This is §13.5b's load-bearing arithmetic — it is what selects
    /// `mlx_gemm.metal:1300`'s `LoopAlignment<false, true, true>`, which puts
    /// the scalar `load_safe` on operand A (the 4 KB SLC-resident activation)
    /// and leaves `load_unsafe` on operand B (the 5.394 GB weight sweep). That
    /// is the whole of why §13.5b dispositions candidate 3 as *"real, and out of
    /// reach of the weight traffic"*.
    ///
    /// The two constants are computed here exactly as `call_mlx_gemm` computes
    /// them at `:784-785` — `m % bm` and `n % bn` against **the tile this
    /// function returned**, not against a literal 32.
    #[test]
    fn at_m_one_align_m_is_false_and_align_n_is_true_at_every_site() {
        let m = 1;
        for (label, _, n, k) in SITES {
            let t = tile_at(m, *n, *k);
            assert!(
                m % t.bm != 0,
                "§13.5b: align_M must be false at {label} (m={m}, bm={})",
                t.bm
            );
            assert!(
                n % t.bn == 0,
                "§13.5b: align_N must be true at {label} -- every LFM2 N_out is \
                 a multiple of BN (n={n}, bn={})",
                t.bn
            );
        }
    }

    /// **The claim: `maxTotalThreadsPerThreadgroup` is 128 for the GEMM tile.**
    ///
    /// §13.5b reads the GEMM arm's 128 against the `bm8` GEMV's 256 and
    /// concludes *"both are declared by the same formula"* — `WM * WN * 32` at
    /// `mlx_gemm.metal:1024` — so the halving is a tile shape and not an
    /// occupancy signal. The formula's inputs are this function's output, which
    /// is why the reading moves if the tile does.
    #[test]
    fn the_gemm_arms_declared_thread_bound_is_a_tile_shape() {
        let t = tile_at(1, 2048, 2048);
        assert_eq!(
            t.wm * t.wn * 32,
            128,
            "§13.5b: mlx_gemm.metal:1024 declares WM * WN * 32; the 128 the \
             step-1 probe observed is this product and nothing else"
        );
    }

    /// **The claim: the `m == 1` routing means the shipping decode step never
    /// reaches `select_tile_config` at all.**
    ///
    /// §13.5b's trigger names *"the `m == 1` routing at `mlx_gemm.rs:705"*
    /// beside the function itself, and §13.5c's census records the consequence:
    /// B=1, K=0 decode dispatches `gemv_float16_*` and no tile. That routing is
    /// a branch in `call_mlx_gemm`, which needs a device — so what is asserted
    /// here is the *predicate*, read from the same source line, rather than the
    /// dispatch.
    ///
    /// Kept deliberately: it is the half of §13.5b's trigger that a tile-only
    /// gate would leave uncovered, and a reader arriving at a failure of the
    /// tests above needs to know which of the two clauses moved.
    #[test]
    fn the_gemv_routing_predicate_still_excludes_the_b1_decode_step() {
        // `call_mlx_gemm`: `if m == 1 || n == 1 { return call_mlx_gemv(..) }`.
        let routes_to_gemv = |m: usize, n: usize| m == 1 || n == 1;
        for (label, _, n, _) in SITES {
            assert!(
                routes_to_gemv(1, *n),
                "§13.5c: at B=1, K=0 the {label} matmul must route to GEMV \
                 before any tile is selected"
            );
        }
        assert!(
            !routes_to_gemv(2, 2048),
            "the predicate must be capable of the other answer: B=2 reaches \
             select_tile_config, which is the road §13.5c measures"
        );
    }

    /// **The claim: the GEMV tile is derived from `out_vec_size`, and the
    /// boundary is at 4096** (`DESIGN.md` §12.2f, issue #492).
    ///
    /// This is the fact that makes an N sweep cross a **kernel** boundary, and
    /// it had never been pinned: #488 swept N from 512 to 131072 and its two
    /// steep points turned out to be the only `bm4` points in it. The boundary
    /// is read here from the same expression `call_mlx_gemv_full` uses rather
    /// than from a comment, so a change to it goes red beside the sections that
    /// quote it.
    ///
    /// **`select_tile_config` is deliberately not consulted**: the GEMV route
    /// returns before it, which is why `LLOOM_383_TILE` cannot reach a GEMV and
    /// why #492 had to add a second knob.
    ///
    /// **It calls `derive_gemv_tile`, the function production calls**, rather
    /// than restating the expression. The first version restated it and a
    /// mutation moving the boundary to 2048 **survived** — recorded at that
    /// function, because a test checking its own copy of the thing under test
    /// is §15.1a's VACUOUS class and the repair is the inputs, not more
    /// assertions.
    #[test]
    fn the_gemv_tile_boundary_is_at_four_thousand_and_ninety_six() {
        let derived_bm = |out_vec_size: usize, in_vec_size: usize| -> usize {
            derive_gemv_tile(out_vec_size, in_vec_size).0
        };
        let k = 2048; // LFM2's hidden_size (§5.2)

        // Either side of the boundary, and ON it -- the `>=` matters.
        assert_eq!(derived_bm(4095, k), 4, "just below the boundary is bm4");
        assert_eq!(derived_bm(4096, k), 8, "the boundary itself is bm8");
        assert_eq!(derived_bm(4097, k), 8, "just above the boundary is bm8");

        // #488's own sweep points, which is the claim §12.2f makes about that
        // artifact: its two steep points are the only bm4 ones.
        let sweep = [512usize, 2048, 6144, 10752, 32768, 65536, 131072];
        let bm4_points: Vec<usize> = sweep
            .iter()
            .copied()
            .filter(|n| derived_bm(*n, k) == 4)
            .collect();
        assert_eq!(
            bm4_points,
            vec![512, 2048],
            "§12.2f: exactly the two steep points of #488's sweep are bm4, \
             which is why its curve cannot separate N from the tile"
        );

        // LFM2's own w1/w3 site sits above the boundary, so the decode path
        // takes bm8 -- stated because §12.2d's headline figure is at that site.
        assert_eq!(
            derived_bm(10752, k),
            8,
            "LFM2's w1/w3 (N_out = 10752, §5.4) takes the bm8 tile"
        );
    }

    /// **The claim: `LLOOM_492_GEMV_BM` unset changes nothing.**
    ///
    /// `DESIGN.md` §7.1a's rule for every knob in this file — unset selects
    /// byte-for-byte what shipped. The override is applied *after* the shipping
    /// branch has chosen, so this is a property of where the code sits rather
    /// than of a default value, and the test reads the function rather than
    /// asserting the intent.
    ///
    /// The environment is **not** set here, deliberately: `forced_gemv_bm`
    /// reads through a `OnceLock`, so a test that set it would poison every
    /// later test in the process (#69's consumed-`OnceLock` shape, §2.4). What
    /// is checkable without mutating global state is that the unset path
    /// returns `None`, which is the whole of the no-op claim.
    #[test]
    fn the_gemv_tile_override_is_absent_unless_asked_for() {
        assert!(
            std::env::var("LLOOM_492_GEMV_BM").is_err(),
            "this test asserts the DEFAULT; it cannot run with the override set"
        );
        assert_eq!(
            crate::gemv_tile::forced_gemv_bm(),
            None,
            "unset must select the derived tile, so every recorded figure \
             belongs to the shipping path (§7.1a)"
        );
    }
}
