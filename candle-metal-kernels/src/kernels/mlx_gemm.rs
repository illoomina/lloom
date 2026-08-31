use crate::kernels::params::{begin_packed_params, finish_packed_params, GemvParams, ParamStyle};
use crate::metal::{Buffer, ComputeCommandEncoder, Device, MetalDeviceType};
use crate::utils::{EncoderProvider, Input};
use crate::{
    debug_group, set_params, ConstantValues, EncoderParam, Kernels, MetalKernelError, Output,
    Source, Value,
};
use objc2_metal::MTLSize;

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
    use std::sync::OnceLock;
    static TILE: OnceLock<Option<TileConfig>> = OnceLock::new();
    *TILE.get_or_init(|| match std::env::var("LLOOM_383_TILE") {
        Ok(v) => Some(match v.as_str() {
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
        }),
        Err(_) => None,
    })
}

/// The `m` below which `select_tile_config` keeps `TILE_32_32_16_2_2`.
///
/// **16 is what ships**, which is `select_tile_config`'s own literal and what
/// every recorded figure in this repository belongs to. `LLOOM_364_SKINNY_MAX`
/// raises it for issue #364's threshold arm and is read **once**.
///
/// An unparseable value **panics** rather than falling back to the default:
/// `DESIGN.md` §2.4 — *"make an invalid setting panic rather than fall back
/// silently"* — because an arm that silently ran the baseline is #69's vacuous
/// determinism run, and the whole point of this knob is to be A/B'd.
fn skinny_max() -> usize {
    use std::sync::OnceLock;
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| match std::env::var("LLOOM_364_SKINNY_MAX") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("LLOOM_364_SKINNY_MAX must be an integer, got {v:?}")),
        Err(_) => 16,
    })
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
    use std::sync::OnceLock;
    static FORCE: OnceLock<Option<ForceGemm>> = OnceLock::new();
    *FORCE.get_or_init(|| match std::env::var("LLOOM_386_FORCE_GEMM") {
        Ok(v) => Some(match v.as_str() {
            "m" => ForceGemm::M,
            "n" => ForceGemm::N,
            "both" => ForceGemm::Both,
            other => panic!("LLOOM_386_FORCE_GEMM must be one of m|n|both, got {other:?}"),
        }),
        Err(_) => None,
    })
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

    // Special case: For very small M (vector-matrix multiply),
    // use the original 32x32 tile to avoid thread waste.
    // When M is very small (< bm), using larger bm values causes significant
    // thread underutilization because most threads in the M dimension have no work.
    // This is critical for benchmarks like [1, 2048] @ [2048, 2048] (m=1).
    //
    // We use m < 16 as the threshold because:
    // - For m=1 to m=15, even 32x32 tile has some waste but it's the smallest available
    // - For m >= 16, the larger tiles can provide better throughput despite some waste
    //
    // # The threshold is a measurement arm as of issue #364
    //
    // `LLOOM_364_SKINNY_MAX` raises it, so the 32x32 tile is kept past 16 rather
    // than yielding to `TILE_64_32_32_2_2`. **Off by default** (`DESIGN.md`
    // §7.1a: no default is flipped without its own argued decision), so an
    // unconfigured process selects byte-for-byte what shipped.
    //
    // It exists because the comment above is a *prediction* — *"the larger
    // tiles can provide better throughput"* — and measured at one token's
    // resolution it fails: crossing `m = 15 -> 16` costs **+17.9 % GPU busy for
    // +6.7 % of work**, RESOLVED against a null, while the same one-token step
    // inside either tile regime is unresolved. The larger tile is the expensive
    // one at its own boundary.
    //
    // Read once (`OnceLock`), so this is not a per-dispatch `getenv` on a path
    // §15.2 #10 forbids O(N) work on.
    if m < skinny_max() {
        return TILE_32_32_16_2_2;
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
        device, ep, kernels, dtype, (b, m, n, k),
        lhs_stride, lhs_offset, lhs_buffer,
        rhs_stride, rhs_offset, rhs_buffer,
        output, style, None,
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
) -> Result<(), MetalKernelError> {
    debug_assert!(m == 1 || n == 1, "call_mlx_gemv requires M=1 or N=1");

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
        let (bm, bn, sm, sn): (usize, usize, usize, usize) = if in_vec_size <= 64 {
            (1, 1, 8, 4)
        } else if in_vec_size >= 16 * out_vec_size {
            (1, 8, 1, 32)
        } else if out_vec_size >= 4096 {
            (8, 1, 1, 32)
        } else {
            (4, 1, 1, 32)
        };
        let tm: usize = if out_vec_size < 4 { 1 } else { 4 };
        (bm, bn, sm, sn, tm, 4usize)
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
    let norm_suffix = if fused_norm.is_some() {
        let tile_is_instantiated =
            !transpose_mat && bn == 1 && sm == 1 && sn == 32 && tn == 4 && tm == 4 && (bm == 4 || bm == 8);
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
    } else {
        ""
    };

    let name = format!(
        "{}_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc0_axpby0{}{}",
        kernel_prefix,
        dtype_str,
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
        style.name_suffix(),
        norm_suffix
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
    if let Some(norm) = fused_norm {
        encoder.set_input_buffer(15, Some(norm.weight), norm.weight_offset);
        encoder.set_input_buffer(16, Some(norm.scale), norm.scale_offset);
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

    // The routing decision this project has never audited (issue #386): no
    // threshold, no config axis, and `select_tile_config` is never consulted.
    // `takes_gemv_route` is `m == 1 || n == 1` unless `LLOOM_386_FORCE_GEMM`
    // suppresses one half, which is the only way to build the other arm.
    if takes_gemv_route(m, n) {
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
    let dtype_str = match dtype {
        GemmDType::F32 => "f32",
        GemmDType::F16 => "f16",
        GemmDType::BF16 => "bf16",
    };
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
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

    /// **The claim: `select_tile_config` returns a BM=32 tile below `m` = 16.**
    ///
    /// This is what `DESIGN.md` §13.5b asserts as *"`select_tile_config`, rs:85:
    /// `m < 16` → `TILE_32_32_16_2_2` unconditionally"* and what every figure in
    /// that section's `align_M`/`align_N` branch reading is computed from. It is
    /// read here from the function rather than from a comment beside it.
    ///
    /// Every one of LFM2's 167 weight matmuls is asserted rather than a
    /// representative, because §13.5b's own claim is quantified over all 167 and
    /// the sites differ in `N_out` by 250× (512 to 128000).
    #[test]
    fn below_the_threshold_every_lfm2_matmul_gets_the_32_32_tile() {
        for m in 1..16 {
            for (label, _, n, k) in SITES {
                let t = tile_at(m, *n, *k);
                assert_eq!(
                    (t.bm, t.bn, t.bk, t.wm, t.wn),
                    (32, 32, 16, 2, 2),
                    "§13.5b: at m={m} the {label} matmul ({n}x{k}) must take \
                     TILE_32_32_16_2_2; got {t:?}"
                );
            }
        }
    }

    /// **The claim: the threshold is exactly 16, and crossing it leaves BM=32.**
    ///
    /// `DESIGN.md` §13.5c pins the boundary *"exact at 16, pinned by
    /// bisection"* from a GPU census; this is the same boundary read off the
    /// function with no GPU. **Both sides are asserted** (§8.1g): a test that
    /// only checked the low side would pass on a function that returned the
    /// 32×32 tile everywhere, which is precisely the `LLOOM_364_SKINNY_MAX`
    /// arm and precisely the change §13.5c argues about.
    #[test]
    fn the_threshold_is_at_sixteen_and_the_tile_changes_across_it() {
        let (n, k) = (2048, 2048);
        let below = tile_at(15, n, k);
        let at = tile_at(16, n, k);
        assert_eq!(
            (below.bm, below.bn),
            (32, 32),
            "§13.5c: m=15 is the last m on the 32x32 tile"
        );
        assert_eq!(
            (at.bm, at.bn, at.bk, at.wm, at.wn),
            (64, 32, 32, 2, 2),
            "§13.5c: m=16 takes TILE_64_32_32_2_2 -- the larger tile, and the \
             one measured +17.9 % of GPU busy for +6.7 % of work"
        );
        assert_ne!(
            (below.bm, below.bn, below.bk, below.wm, below.wn),
            (at.bm, at.bn, at.bk, at.wm, at.wn),
            "the boundary must be a boundary: if these agree, every §13.5c \
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
}
