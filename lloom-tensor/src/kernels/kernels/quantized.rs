use crate::kernels::metal::arms::{self, Feature, Features};
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, Kernels,
    MetalKernelError, Output, Source, Value,
};
use objc2_metal::MTLSize;

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4NL,
    IQ4XS,
    Q2_0,
}

impl GgmlDType {
    /// Whether this weight dtype has a genericized #750 matmul-vec core, and so
    /// can be dispatched under a non-f32 [`GemvTriple`]. The dense (F16/BF16/F32)
    /// and Q8-family weights keep their f32-store-only kernels, so only these
    /// rows offer the `(f16, f32, f16)` arm.
    pub fn has_gemv_triple(self) -> bool {
        matches!(
            self,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ3S
                | GgmlDType::IQ4NL
                | GgmlDType::IQ4XS
                | GgmlDType::Q2_0
        )
    }
}

/// One float leg of a matmul-vec dtype triple (issue #750, `DESIGN.md` §8.1.4).
/// The accumulate leg is not spelled here because §8.1.4 fixes it: *accumulate in
/// F32, store in the working dtype*. Every legal float triple this dispatch names
/// therefore accumulates in `f32`; the leg that varies is the activation the
/// kernel reads (`src1`) and the store it writes (`dst`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemvFloat {
    F32,
    F16,
}

impl GemvFloat {
    /// The `[[host_name]]` suffix segment this leg contributes.
    fn suffix(self) -> &'static str {
        match self {
            GemvFloat::F32 => "f32",
            GemvFloat::F16 => "f16",
        }
    }
}

/// The `(activation, accumulate, store)` dtype triple a quantized matmul-vec
/// dispatch runs under (issue #750). The accumulate leg is always `f32` for the
/// float triples (§8.1.4), so the enum names the `(activation, store)` pair the
/// kernel is instantiated over; a quantized *weight* dtype ([`GgmlDType`]) is an
/// orthogonal axis carried alongside it.
///
/// Only the two float triples the Metal side instantiates are nameable:
/// `(f32, f32, f32)` — the historical lock, byte-identical to the pre-#750 kernel
/// — and `(f16, f32, f16)` — f16 activations + store, matching the dense
/// `mul_mv_f16_f16` arm, so a quantized weight no longer forces f32 activations.
/// The M5 native-int triple `(int8, int32, …)` is registered as declared-but-
/// unimplemented in `lloom`'s axis registry (issue #750, out of scope here — no M5
/// hardware to gate it), and its kernel body is a deferred follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemvTriple {
    /// `(f32, f32, f32)` — the historical arm, byte-identical to the pre-#750 lock.
    F32F32F32,
    /// `(f16, f32, f16)` — f16 activations, f32 accumulate, f16 store.
    F16F32F16,
    /// `(i8, i32, i8)` — the M5 native-integer arm: int8 activations and store,
    /// int32 accumulate. **Declared but unimplemented** (issue #750): the triple
    /// is named so the M5 arm is one more instantiation of the generic core rather
    /// than a fork, but its kernel body is a deferred follow-up (no M5 hardware
    /// here to gate or measure it, and at B=1 decode never dispatches
    /// `simdgroup_matrix`, so the int arm is a prefill/B>1 lever — `DESIGN.md`
    /// §13.5a). Dispatching it errors rather than silently falling back.
    I8I32I8,
}

impl GemvTriple {
    /// Whether this triple's Metal kernel body exists. The M5 native-int arm is
    /// declared (it is a legal triple, and the optimizer models it as
    /// `NumericClass::BitExact`) but not built, so a dispatch under it must be
    /// refused rather than resolve a kernel that does not exist.
    pub fn is_implemented(self) -> bool {
        !matches!(self, GemvTriple::I8I32I8)
    }

    /// What the GPU must execute natively for this arm to be dispatched
    /// (§15.2a): the float legs' types, or the int8 matrix unit.
    pub fn requires(self) -> Features {
        match self {
            GemvTriple::F32F32F32 => Features::empty().with(Feature::F32),
            GemvTriple::F16F32F16 => Features::empty().with(Feature::F16).with(Feature::F32),
            GemvTriple::I8I32I8 => Features::empty().with(Feature::Int8Mma),
        }
    }

    /// The triple as a refusal names it.
    pub fn label(self) -> &'static str {
        match self {
            GemvTriple::F32F32F32 => "(f32, f32, f32)",
            GemvTriple::F16F32F16 => "(f16, f32, f16)",
            GemvTriple::I8I32I8 => "(i8, i32, i8)",
        }
    }

    /// Refuse this arm on a `device` that does not execute its legs, then
    /// refuse a declared triple whose kernel body is not built. `op` names the
    /// dispatch in the refusal.
    fn check(self, device: &Device, op: &'static str) -> Result<(), MetalKernelError> {
        arms::require_all(device, self.requires(), || format!("{op} {}", self.label()))?;
        if !self.is_implemented() {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "declared-unimplemented GemvTriple (M5 native-int)",
                op,
            ));
        }
        Ok(())
    }

    /// The activation (`src1`) dtype leg, for the two float arms. The native-int
    /// arm's legs are integer and have no [`GemvFloat`]; it is declared-only, so
    /// callers gate on [`is_implemented`](Self::is_implemented) first.
    pub fn activation(self) -> GemvFloat {
        match self {
            GemvTriple::F32F32F32 => GemvFloat::F32,
            GemvTriple::F16F32F16 => GemvFloat::F16,
            GemvTriple::I8I32I8 => GemvFloat::F32, // unreachable: declared-only
        }
    }

    /// The store (`dst`) dtype leg (float arms only — see [`activation`](Self::activation)).
    pub fn store(self) -> GemvFloat {
        match self {
            GemvTriple::F32F32F32 => GemvFloat::F32,
            GemvTriple::F16F32F16 => GemvFloat::F16,
            GemvTriple::I8I32I8 => GemvFloat::F32, // unreachable: declared-only
        }
    }

    /// The `[[host_name]]` suffix this triple selects on a `kernel_mul_mv_<q>`
    /// name. The `(f32, f32, f32)` arm keeps the historical single-`_f32` suffix
    /// (so it resolves the byte-identical pre-#750 kernel); every other float
    /// triple spells both float legs, matching the dense `mul_mv_<act>_<store>`
    /// scheme. The native-int arm has no suffix — it is declared-unimplemented.
    fn name_suffix(self) -> String {
        match self {
            GemvTriple::F32F32F32 => "f32".to_string(),
            GemvTriple::I8I32I8 => "i8_i8".to_string(), // no kernel; is_implemented() gates it
            other => format!("{}_{}", other.activation().suffix(), other.store().suffix()),
        }
    }
}

/// Threadgroup geometry of the matmul-vec kernels per weight dtype:
/// `(threads.x, threads.y, rows per threadgroup)` — the same table the plain
/// and the routed (`mul_mv_id`) dispatches use, since the routed kernel wraps
/// the plain per-dtype core.
fn mv_threadgroup(dtype: GgmlDType) -> (usize, usize, usize) {
    match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => match q4k_nr0() {
            // Two simdgroups of `nr0` rows (P31 (5)).
            Some(nr0) => (4, 16, 2 * nr0),
            // The incumbent: one simdgroup, four rows.
            None => (4, 8, 4),
        },
        GgmlDType::Q3K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        // Two simdgroups of `nr0` rows (2 the incumbent; P31 (5)).
        GgmlDType::Q5K => (2, 32, 2 * q5k_nr0().unwrap_or(2)),
        GgmlDType::Q6K => {
            // Two simdgroups a threadgroup, `q6k_nr0()` rows each (P29 (6)).
            let nth0 = 2;
            let nth1 = 32;
            let align = 2 * q6k_nr0();
            (nth0, nth1, align)
        }
        // The dense-weight template (`kernel_mul_mv_impl`, f32 / f16): one
        // simdgroup per OUTPUT ROW (`r0 = tgpig.x`), so the grid is one
        // threadgroup per row — `align` 1. An `align` of 8 here dispatched
        // n/8 threadgroups and left 7 rows in 8 of every output unwritten
        // (the f32-expert test models decoded whatever the buffer held).
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 1;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 1;
            (nth0, nth1, align)
        }
        // IQ family: 64 threads = 2 simdgroups. Rows per threadgroup is the
        // core's rows-per-simdgroup x 2 (llama.cpp's N_R0_* x N_SG_*): 4 x 2 for
        // the IQ2/IQ3 cores, 2 x 2 for IQ4_NL/IQ4_XS. Over-dispatching (a smaller
        // `align`) makes threadgroups whose rows are all past `ne01`; the cores
        // guard their stores, but the routed dispatch has `batch * topk` planes
        // and wasted threadgroups per plane are wasted per plane.
        // IQ2_XS and IQ4_NL: 8 simdgroups a threadgroup (`NSG_IQ2XS`,
        // `NSG_IQ4NL` in the kernel), 32 and 16 rows: the per-threadgroup
        // table staging amortised over four times the rows (#749 P29 (5)).
        GgmlDType::IQ2XS => {
            let nsg = iq_nsg(dtype);
            (4, 8 * nsg, 4 * nsg)
        }
        GgmlDType::IQ4NL => {
            let nsg = iq_nsg(dtype);
            (4, 8 * nsg, 2 * nsg)
        }
        GgmlDType::IQ2S => {
            let nsg = iq_nsg(dtype);
            (4, 8 * nsg, 4 * nsg)
        }
        GgmlDType::IQ2XXS | GgmlDType::IQ3XXS | GgmlDType::IQ3S => {
            let nth0 = 4;
            let nth1 = 16;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::IQ4XS => {
            let nth0 = 4;
            let nth1 = 16;
            let align = 4;
            (nth0, nth1, align)
        }
        // Q2_0: 8 rows per simdgroup x 2 simdgroups (the fork's N_R0_Q2_0 /
        // N_SG_Q2_0), so 16 rows per threadgroup.
        GgmlDType::Q2_0 => {
            let nth0 = 32;
            let nth1 = 2;
            let align = 16;
            (nth0, nth1, align)
        }
    }
}

/// Simdgroups a threadgroup of the IQ2_XS / IQ4_NL matvec cores
/// (`LLOOM_QMV_IQ2XS_NSG`, `LLOOM_QMV_IQ4NL_NSG`, read once; 1..16): the
/// per-threadgroup table staging amortised over more rows against fewer
/// threadgroups in flight (#749 P29 (5)). The kernel takes it as a function
/// constant (`NSG_IQ2XS_FC` / `NSG_IQ4NL_FC`); the default is the
/// incumbent 2.
pub(crate) fn iq_nsg(dtype: GgmlDType) -> usize {
    match dtype {
        // 4 by default (P30 (3)): on the warm census the routed gate|up read
        // 97.6 → 90.3 µs and the down 75.0 → 66.3 against 2 (8 the same as
        // 4, 1 slower); every arm byte-identical (the row mapping alone).
        GgmlDType::IQ2XS => crate::tuning::tuning().qmv.iq2xs_nsg,
        GgmlDType::IQ4NL => crate::tuning::tuning().qmv.iq4nl_nsg,
        // The IQ2_S core stages its 8 KB grid per threadgroup (P31 (4)); 4
        // amortises the staging over 16 rows as the IQ2_XS core's does.
        GgmlDType::IQ2S => crate::tuning::tuning().qmv.iq2s_nsg,
        _ => 2,
    }
}

/// Rows a simdgroup of the Q6_K matvec core (`LLOOM_QMV_Q6K_NR0`, read
/// once; 1..4): ggml master's core takes 2, the port took 1 (P29 (6)). The
/// kernel takes it as a function constant (`NR0_Q6K_FC`). **The default is
/// 2**: the checkpoint's Q6_K stems at K = 2560 read 200 → 116–141 µs and
/// at K = 6144 90 → 50 (the census), the decode −1.3 % on a rotation;
/// the contraction differs from the one-row core's at the last bits, so
/// the logits digest moves (Paris `054a6ef1…` → `5f5ddb57…`, the tokens
/// unmoved); `1` is the incumbent arm.
pub(crate) fn q6k_nr0() -> usize {
    crate::tuning::tuning().qmv.q6k_nr0
}

/// Rows a simdgroup for the Q4_K core (`LLOOM_QMV_Q4K_NR0`, read once;
/// 1..4): set, the core runs TWO simdgroups of that many rows (ggml
/// master's 2 × 2) in a 64-thread threadgroup; unset, the incumbent one
/// simdgroup of four rows in a 32-thread threadgroup (P31 (5)).
pub(crate) fn q4k_nr0() -> Option<usize> {
    crate::tuning::tuning().qmv.q4k_nr0
}

/// Rows a simdgroup for the Q5_K core (`LLOOM_QMV_Q5K_NR0`, read once;
/// 1..2): the incumbent is 2, ggml master's 1 (P31 (5)).
pub(crate) fn q5k_nr0() -> Option<usize> {
    crate::tuning::tuning().qmv.q5k_nr0
}

/// The function constants a matvec pipeline of `dtype` is loaded with:
/// the simdgroup count for the IQ2_XS / IQ4_NL cores, the rows a simdgroup
/// for Q6_K, none elsewhere.
fn mv_constants(dtype: GgmlDType) -> Option<ConstantValues> {
    match dtype {
        GgmlDType::IQ2XS => Some(ConstantValues::new(vec![(910, Value::USize(iq_nsg(dtype)))])),
        GgmlDType::IQ4NL => Some(ConstantValues::new(vec![(911, Value::USize(iq_nsg(dtype)))])),
        GgmlDType::Q6K => Some(ConstantValues::new(vec![(912, Value::USize(q6k_nr0()))])),
        GgmlDType::IQ2S => Some(ConstantValues::new(vec![(913, Value::USize(iq_nsg(dtype)))])),
        // Always supplied (a function naming a constant must be specialised
        // to build a pipeline); 0 selects the incumbent geometry.
        GgmlDType::Q4K => Some(ConstantValues::new(vec![(914, Value::USize(q4k_nr0().unwrap_or(0)))])),
        GgmlDType::Q5K => Some(ConstantValues::new(vec![(915, Value::USize(q5k_nr0().unwrap_or(0)))])),
        _ => None,
    }
}

/// `LLOOM_QMV_ROWS=on` (read once): the small-batch rows arm — `NR1`
/// activation rows per weight read (`kernel_mul_mv_*_rows_impl`). Off by
/// default, and not taken by the `fast` contract either: a row's bits
/// depend on how many rows share its launch, which only `fast` licenses,
/// and on the real checkpoint it is slower than the per-row launches at
/// three rows and up (`architecture.md` §2.3.5b).
fn rows_arm_on() -> bool {
    crate::tuning::tuning().qmv.rows
}

/// The threadgroup geometry each rows kernel is written for: the rows a
/// threadgroup covers are a property of the kernel's own row indexing, not
/// of the one-row core's tunable geometry (`q6k_nr0`, `iq_nsg`), which the
/// rows kernels do not read.
fn rows_threadgroup(dtype: GgmlDType) -> (usize, usize, usize) {
    match dtype {
        // `row = 2 * r0 + sgitg`: two simdgroups of one row.
        GgmlDType::Q6K => (2, 32, 2),
        // `first_row = r0 * N_DST`: one simdgroup of four rows.
        GgmlDType::Q4K => (4, 8, 4),
        // `first_row = (r0 * 2 + sgitg) * 2`: two simdgroups of two rows.
        GgmlDType::IQ4XS | GgmlDType::IQ4NL => (4, 16, 4),
        // `first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST`: two of four.
        GgmlDType::IQ3S => (4, 16, 8),
        _ => mv_threadgroup(dtype),
    }
}

/// Whether the dtype's matmul-vec core stages a codebook into threadgroup
/// memory (`shared_values [[threadgroup(0)]]`, 8192 bytes per llama.cpp).
fn needs_iq_shared(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::IQ2XXS
            | GgmlDType::IQ2XS
            | GgmlDType::IQ2S
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ3S
            | GgmlDType::IQ4NL
            | GgmlDType::IQ4XS
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    triple: GemvTriple,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    // The strides in bytes (the f32 / f16 template steps rows and batches
    // by them; the quantized impls address from the block geometry).
    let row_bytes = ((k / dtype_block_size(dtype)) * dtype_type_size(dtype)) as i64;
    let nb00 = dtype_type_size(dtype) as i64;
    let nb01 = row_bytes;
    let nb02 = n as i64 * row_bytes;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = core::mem::size_of::<f32>() as i64;
    let nb11 = k as i64 * nb10;
    let nb12 = m as i64 * nb11;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = mv_threadgroup(dtype);
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    triple.check(device, "quantized matmul-vec")?;

    // The weight-dtype stem, e.g. `kernel_mul_mv_q4_0`. The `(activation, store)`
    // triple suffix is appended below. The dense-weight rows (F16/BF16/F32/Q8K)
    // keep their historical `_f32`-store names — they are not part of the #750
    // quantized-core genericization and have no `_f16_f16` arm — so the triple's
    // f16 arm is only offered on the quantized-weight rows.
    let quant_stem = match dtype {
        GgmlDType::Q4_0 => Some("kernel_mul_mv_q4_0"),
        GgmlDType::Q4_1 => Some("kernel_mul_mv_q4_1"),
        GgmlDType::Q5_0 => Some("kernel_mul_mv_q5_0"),
        GgmlDType::Q5_1 => Some("kernel_mul_mv_q5_1"),
        GgmlDType::Q8_0 => Some("kernel_mul_mv_q8_0"),
        GgmlDType::Q2K => Some("kernel_mul_mv_q2_K"),
        GgmlDType::Q3K => Some("kernel_mul_mv_q3_K"),
        GgmlDType::Q4K => Some("kernel_mul_mv_q4_K"),
        GgmlDType::Q5K => Some("kernel_mul_mv_q5_K"),
        GgmlDType::Q6K => Some("kernel_mul_mv_q6_K"),
        GgmlDType::IQ2XXS => Some("kernel_mul_mv_iq2_xxs"),
        GgmlDType::IQ2XS => Some("kernel_mul_mv_iq2_xs"),
        GgmlDType::IQ2S => Some("kernel_mul_mv_iq2_s"),
        GgmlDType::IQ3XXS => Some("kernel_mul_mv_iq3_xxs"),
        GgmlDType::IQ3S => Some("kernel_mul_mv_iq3_s"),
        GgmlDType::IQ4NL => Some("kernel_mul_mv_iq4_nl"),
        GgmlDType::IQ4XS => Some("kernel_mul_mv_iq4_xs"),
        GgmlDType::Q2_0 => Some("kernel_mul_mv_q2_0"),
        // Dense and Q8-family weights: no genericized #750 core, f32 store only.
        GgmlDType::Q8_1 | GgmlDType::Q8K | GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32 => {
            None
        }
    };
    // A `String` naming the kernel: the quantized rows are `stem_<triple suffix>`;
    // the dense/Q8 rows the triple does not reach keep their f32-store names.
    let name: String = match quant_stem {
        // Every quant stem ships both `_f32` (byte-identical) and `_f16_f16`
        // (issue #750), so the suffix the triple selects is always instantiated.
        Some(stem) => format!("{stem}_{}", triple.name_suffix()),
        None => {
            // The triple does not apply to a dense/Q8 weight; a caller asking for
            // a non-f32 triple on one of these has selected an arm that does not
            // exist, which is a dispatch error rather than a silent f32 fallback.
            if triple != GemvTriple::F32F32F32 {
                return Err(MetalKernelError::UnsupportedDTypeForOp(
                    "non-f32 GemvTriple",
                    "dense/Q8 quantized matmul-vec",
                ));
            }
            match dtype {
                GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
                GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
                GgmlDType::F16 => "kernel_mul_mv_f16_f32",
                GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
                GgmlDType::F32 => "kernel_mul_mv_f32_f32",
                _ => unreachable!("quant_stem None only for the dense/Q8 rows above"),
            }
            .to_string()
        }
    };

    // The small-batch arm: `m` activation rows (2..=8) share the weight
    // read (`kernel_mul_mv_*_rows_impl`, the f32 triple, the five stem
    // types it is built for) — `NR1` rows per threadgroup, the grid's
    // second axis over the row groups, at the rows kernel's own geometry; a
    // row is a reordering of the one-row kernel's, not its bits. Elsewhere
    // the rows stay the grid's second axis, one weight read each.
    let (name, rows_per_group) = match (triple, dtype, m) {
        (
            GemvTriple::F32F32F32,
            GgmlDType::Q6K | GgmlDType::Q4K | GgmlDType::IQ4XS | GgmlDType::IQ3S | GgmlDType::IQ4NL,
            2..=8,
        // The rows kernels keep the incumbent Q4_K geometry (one simdgroup of
        // four rows); under the Q4_K rows-a-simdgroup arm (P31 (5)) they
        // would be dispatched at the other shape, so the two arms do not
        // compose — the per-row launches instead.
        ) if rows_arm_on() && !(matches!(dtype, GgmlDType::Q4K) && q4k_nr0().is_some()) => {
            let nr1 = if m <= 2 { 2 } else { 4 };
            (format!("{name}_r{nr1}"), nr1)
        }
        _ => (name, 1),
    };
    let (thread_groups_count, threads_per_threadgroup) = if rows_per_group > 1 {
        let (nth0, nth1, align) = rows_threadgroup(dtype);
        (
            MTLSize {
                width: divide(ne01 as usize, align),
                height: (ne11 as usize).div_ceil(rows_per_group),
                depth: thread_groups_count.depth,
            },
            MTLSize {
                width: nth0,
                height: nth1,
                depth: 1,
            },
        )
    } else {
        (thread_groups_count, threads_per_threadgroup)
    };

    let iq_shared = needs_iq_shared(dtype);

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Quantized, name.clone(), mv_constants(dtype))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    if iq_shared {
        encoder.set_threadgroup_memory_length(0, 8192);
    }
    debug_group!(encoder, "qmm_mv {name} B={b} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// The bytes of one activation element and one stored element under `triple`:
/// the routed launches step `src1` and `dst` by them.
fn triple_leg_bytes(triple: GemvTriple) -> (u64, u64) {
    let bytes = |leg: GemvFloat| match leg {
        GemvFloat::F32 => core::mem::size_of::<f32>() as u64,
        GemvFloat::F16 => core::mem::size_of::<half::f16>() as u64,
    };
    (bytes(triple.activation()), bytes(triple.store()))
}

/// The `kernel_mul_mv_id_<stem>` / `kernel_mul_mv_id2_<stem>` stem of a weight kind.
fn mv_id_stem(dtype: GgmlDType) -> Result<&'static str, MetalKernelError> {
    Ok(match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2K => "q2_K",
        GgmlDType::Q3K => "q3_K",
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q5K => "q5_K",
        GgmlDType::Q6K => "q6_K",
        GgmlDType::IQ2XXS => "iq2_xxs",
        GgmlDType::IQ2XS => "iq2_xs",
        GgmlDType::IQ2S => "iq2_s",
        GgmlDType::IQ3XXS => "iq3_xxs",
        GgmlDType::IQ3S => "iq3_s",
        GgmlDType::IQ4NL => "iq4_nl",
        GgmlDType::IQ4XS => "iq4_xs",
        GgmlDType::Q2_0 => "q2_0",
        GgmlDType::Q8_1 | GgmlDType::Q8K | GgmlDType::BF16 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "no kernel_mul_mv_id arm for this weight dtype",
                "routed quantized matmul-vec",
            ))
        }
    })
}

/// The routed-expert matmul-vec (`kernel_mul_mv_id_<q>_f32`): for every
/// `(token, slot)` pair, one GEMV of `src1`'s row against expert
/// `ids[token][slot]` of the stacked weight `[n_experts, n, k]`.
///
/// `src1` is `[batch, in_dim1, k]` f32 with `in_dim1 ∈ {1, topk}` (one
/// activation per token, or one per selected slot — the down-projection
/// case); `ids` is `[batch, topk]` i32/u32; `dst` is `[batch, topk, n]` f32.
/// Grid: rows per threadgroup as the plain dispatch, one threadgroup column,
/// `batch * topk` planes — the kernel decodes `(token, slot)` from the plane
/// index and the per-dtype core sees a plain `m == 1` GEMV.
///
/// `triple` picks the `(activation, accumulate, store)` arm: `(f32, f32,
/// f32)` wraps the cores' f32 aliases; `(f16, f32, f16)` (issue #748, #750
/// item B) wraps the same genericized cores at `<half, half>` — quantized
/// weight rows only, as the plain dispatch. `src1` is read and `dst` written
/// in the triple's legs.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_id_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    triple: GemvTriple,
    (n_experts, n, k): (usize, usize, usize),
    (batch, topk, in_dim1): (usize, usize, usize),
    src0: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    triple.check(device, "routed quantized matmul-vec")?;
    if triple != GemvTriple::F32F32F32 && !dtype.has_gemv_triple() {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "non-f32 GemvTriple",
            "dense/Q8 routed quantized matmul-vec",
        ));
    }
    let stem = mv_id_stem(dtype)?;
    let name = format!("kernel_mul_mv_id_{stem}_{}", triple.name_suffix());
    let (act_size, store_size) = triple_leg_bytes(triple);
    if in_dim1 != 1 && in_dim1 != topk {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "routed matmul-vec activation dim1 must be 1 or topk",
            "routed quantized matmul-vec",
        ));
    }

    // Everything is in reverse (ggml order).
    let nei0 = topk as i64;
    let nei1 = batch as i64;
    let nbi1 = (topk * core::mem::size_of::<u32>()) as u64;
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = n_experts as i64;
    // The weight strides in bytes. The quantized impls address rows from
    // `ne00` and the block geometry, but the f32 / f16 template
    // (`kernel_mul_mv_impl`) steps rows by `nb01` — a zero here made every
    // output row read row 0 of the plane (every f32-expert test model on
    // Metal decoded garbage, finitely).
    let row_bytes = ((k / dtype_block_size(dtype)) * dtype_type_size(dtype)) as u64;
    let nb00 = dtype_type_size(dtype) as u64;
    let nb01 = row_bytes;
    let nb02 = n as u64 * row_bytes;
    let ne10 = k as i64;
    let ne11 = in_dim1 as i64;
    let ne12 = batch as i64;
    let ne13 = 1i64;
    let nb10 = act_size;
    let nb11 = k as u64 * act_size;
    let nb12 = (in_dim1 * k) as u64 * act_size;
    let ne0 = n as i64;
    let ne1 = topk as i64;
    let nb1 = n as u64 * store_size;

    let (nth0, nth1, align) = mv_threadgroup(dtype);
    let thread_groups_count = MTLSize {
        width: divide(n, align),
        height: 1,
        depth: batch * topk,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Quantized, name.clone(), mv_constants(dtype))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    if needs_iq_shared(dtype) {
        encoder.set_threadgroup_memory_length(0, 8192);
    }
    debug_group!(
        encoder,
        "qmm_mv_id {name} E={n_experts} B={batch} topk={topk} K={k} N={n}"
    );

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            Output::new(dst),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// The routed GEMM's row block (`BLOCK_SIZE_N` of `kernel_mul_mm_id_impl`).
pub const MM_ID_BLOCK_ROWS: usize = 32;

/// Blocks of `bn` rows a routed tile of `rows × topk` pairs over
/// `n_experts` can need at most: one part-block per expert plus the full
/// blocks — the grid's x and the map's capacity (`kernel_moe_rowmap`).
pub fn mm_id_block_bound(rows: usize, topk: usize, n_experts: usize, bn: usize) -> usize {
    (rows * topk).div_ceil(bn.max(1)) + n_experts
}

/// `kernel_moe_rowmap`: the router's `ids [rows, topk]` (u32) → `rowids`
/// (`rows·topk` ushort2), `blocks` (`3 · bound` u32), `meta` (1 u32, the
/// block count). One threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn call_moe_rowmap(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    topk: usize,
    n_experts: usize,
    ids: &Buffer,
    ids_offset: usize,
    rowids: &Buffer,
    blocks: &Buffer,
    meta: &Buffer,
    bound: usize,
    block_n: usize,
) -> Result<(), MetalKernelError> {
    if n_experts > 1024 {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "moe_rowmap: n_experts {n_experts} > 1024"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Quantized, "kernel_moe_rowmap")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "moe_rowmap rows={rows} topk={topk} E={n_experts}");
    set_params!(
        encoder,
        (
            (ids, ids_offset),
            Output::new(rowids),
            Output::new(blocks),
            Output::new(meta),
            (rows * topk) as u32,
            topk as u32,
            n_experts as u32,
            block_n as u32,
            bound as u32
        )
    );
    let width = 1024usize.min(pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The routed GEMM over the rows grouped by expert (`kernel_mul_mm_id_mapped`,
/// lloom #749 P22): `src0` the stacked expert weights `[n_experts, n, k]`,
/// `src1` the f32 activations `[batch, in_dim1, k]` (`in_dim1` 1 or `topk`),
/// the map from [`call_moe_rowmap`], `dst [batch, topk, n]` f32 as the
/// matmul-vec lays it. One expert's weight is read once per 32-row block of
/// the rows routed to it, where the matmul-vec reads it once per row: the
/// prefill tile's arm (`LLOOM_MOE_MM_FROM`). Not bit-identical to the
/// matmul-vec: the weights dequantize to f16 for the simdgroup multiply.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_id_mapped(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n_experts, n, k): (usize, usize, usize),
    (batch, topk, in_dim1): (usize, usize, usize),
    src0: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    rowids: &Buffer,
    blocks: &Buffer,
    meta: &Buffer,
    bound: usize,
    bn: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    if bn != 16 && bn != MM_ID_BLOCK_ROWS {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "routed matmul row block must be 16 or 32",
            "routed quantized matmul (rows grouped by expert)",
        ));
    }
    let stem = match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2_0 => "q2_0",
        GgmlDType::Q2K => "q2_K",
        GgmlDType::Q3K => "q3_K",
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q5K => "q5_K",
        GgmlDType::Q6K => "q6_K",
        GgmlDType::IQ2XXS => "iq2_xxs",
        GgmlDType::IQ2XS => "iq2_xs",
        GgmlDType::IQ2S => "iq2_s",
        GgmlDType::IQ3XXS => "iq3_xxs",
        GgmlDType::IQ3S => "iq3_s",
        GgmlDType::IQ4NL => "iq4_nl",
        GgmlDType::IQ4XS => "iq4_xs",
        GgmlDType::Q8_1 | GgmlDType::Q8K | GgmlDType::BF16 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "no kernel_mul_mm_id_mapped arm for this weight dtype",
                "routed quantized matmul (rows grouped by expert)",
            ))
        }
    };
    if in_dim1 != 1 && in_dim1 != topk {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "routed matmul activation dim1 must be 1 or topk",
            "routed quantized matmul (rows grouped by expert)",
        ));
    }
    let name = if bn == 16 {
        format!("kernel_mul_mm_id_mapped16_{stem}_f32")
    } else {
        format!("kernel_mul_mm_id_mapped_{stem}_f32")
    };
    let row_bytes = ((k / dtype_block_size(dtype)) * dtype_type_size(dtype)) as u64;
    let act = core::mem::size_of::<f32>() as u64;
    let ne00 = k as i64;
    let ne02 = n_experts as i64;
    let nb01 = row_bytes;
    let nb02 = n as u64 * row_bytes;
    let ne11 = in_dim1 as i64;
    let ne12 = batch as i64;
    let nb10 = act;
    let nb11 = k as u64 * act;
    let nb12 = (in_dim1 * k) as u64 * act;
    let ne0 = n as i64;
    let ne1 = topk as i64;
    let pipeline = kernels.load_pipeline(device, Source::Quantized, name.clone())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qmm_mm_id {name} E={n_experts} B={batch} topk={topk} K={k} N={n} bn={bn}"
    );
    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            Output::new(dst),
            rowids,
            blocks,
            meta,
            ne00,
            ne02,
            nb01,
            nb02,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1
        )
    );
    encoder.set_threadgroup_memory_length(0, 8192 + bn * 4);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: bound,
            height: divide(n, 64),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// `kernel_dequant_rows_f16_*`: the quantized `[n, k]` `src` to half in
/// `dst` (`n · k` halves). The prefill stems' dequantise-once (P26).
pub fn call_quantized_dequant_rows_f16(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n, k): (usize, usize),
    src: &Buffer,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let stem = match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2_0 => "q2_0",
        GgmlDType::Q2K => "q2_K",
        GgmlDType::Q3K => "q3_K",
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q5K => "q5_K",
        GgmlDType::Q6K => "q6_K",
        GgmlDType::IQ2XXS => "iq2_xxs",
        GgmlDType::IQ2XS => "iq2_xs",
        GgmlDType::IQ2S => "iq2_s",
        GgmlDType::IQ3XXS => "iq3_xxs",
        GgmlDType::IQ3S => "iq3_s",
        GgmlDType::IQ4NL => "iq4_nl",
        GgmlDType::IQ4XS => "iq4_xs",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "no kernel_dequant_rows_f16 arm for this weight dtype",
                "dequantise rows to f16",
            ))
        }
    };
    if !k.is_multiple_of(16) {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "k must be a multiple of 16",
            "dequantise rows to f16",
        ));
    }
    let name = format!("kernel_dequant_rows_f16_{stem}");
    let row_bytes = (k / dtype_block_size(dtype)) * dtype_type_size(dtype);
    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qdequant_rows_f16 {stem} n={n} k={k}");
    set_params!(encoder, (src, Output::new(dst), k as i64, row_bytes as u64));
    let chunks = k / 16;
    encoder.dispatch_thread_groups(
        MTLSize {
            width: chunks.div_ceil(128).min(1024),
            height: n,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Elements per block of a weight dtype (the ggml `blck_size`).
fn dtype_block_size(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => 1,
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1
        | GgmlDType::IQ4NL => 32,
        GgmlDType::Q2_0 => 64,
        GgmlDType::Q2K
        | GgmlDType::Q3K
        | GgmlDType::Q4K
        | GgmlDType::Q5K
        | GgmlDType::Q6K
        | GgmlDType::Q8K
        | GgmlDType::IQ2XXS
        | GgmlDType::IQ2XS
        | GgmlDType::IQ2S
        | GgmlDType::IQ3XXS
        | GgmlDType::IQ3S
        | GgmlDType::IQ4XS => 256,
    }
}

/// Bytes per block of a weight dtype (the ggml `type_size`).
fn dtype_type_size(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::F32 => 4,
        GgmlDType::F16 | GgmlDType::BF16 => 2,
        GgmlDType::Q4_0 => 18,
        GgmlDType::Q4_1 => 20,
        GgmlDType::Q5_0 => 22,
        GgmlDType::Q5_1 => 24,
        GgmlDType::Q8_0 => 34,
        GgmlDType::Q8_1 => 36,
        GgmlDType::Q2K => 84,
        GgmlDType::Q3K => 110,
        GgmlDType::Q4K => 144,
        GgmlDType::Q5K => 176,
        GgmlDType::Q6K => 210,
        GgmlDType::Q8K => 292,
        GgmlDType::IQ2XXS => 66,
        GgmlDType::IQ2XS => 74,
        GgmlDType::IQ2S => 82,
        GgmlDType::IQ3XXS => 98,
        GgmlDType::IQ3S => 110,
        GgmlDType::IQ4NL => 18,
        GgmlDType::IQ4XS => 136,
        GgmlDType::Q2_0 => 18,
    }
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
        GgmlDType::IQ2XXS => "kernel_mul_mm_iq2_xxs_f32",
        GgmlDType::IQ2XS => "kernel_mul_mm_iq2_xs_f32",
        GgmlDType::IQ2S => "kernel_mul_mm_iq2_s_f32",
        GgmlDType::IQ3XXS => "kernel_mul_mm_iq3_xxs_f32",
        GgmlDType::IQ3S => "kernel_mul_mm_iq3_s_f32",
        GgmlDType::IQ4NL => "kernel_mul_mm_iq4_nl_f32",
        GgmlDType::IQ4XS => "kernel_mul_mm_iq4_xs_f32",
        GgmlDType::Q2_0 => "kernel_mul_mm_q2_0_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm {name} M={ne11} K={ne00} N={ne01}");

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_get_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    hidden_size: usize,
    row_stride: usize,
    ids_len: usize,
    src: &Buffer,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_row_stride = hidden_size * core::mem::size_of::<f32>();
    let name = match dtype {
        GgmlDType::F32 => "kernel_get_rows_f32",
        GgmlDType::F16 => "kernel_get_rows_f16",
        GgmlDType::BF16 => "kernel_get_rows_bf16",
        GgmlDType::Q4_0 => "kernel_get_rows_q4_0",
        GgmlDType::Q4_1 => "kernel_get_rows_q4_1",
        GgmlDType::Q5_0 => "kernel_get_rows_q5_0",
        GgmlDType::Q5_1 => "kernel_get_rows_q5_1",
        GgmlDType::Q8_0 => "kernel_get_rows_q8_0",
        GgmlDType::Q2K => "kernel_get_rows_q2_K",
        GgmlDType::Q3K => "kernel_get_rows_q3_K",
        GgmlDType::Q4K => "kernel_get_rows_q4_K",
        GgmlDType::Q5K => "kernel_get_rows_q5_K",
        GgmlDType::Q6K => "kernel_get_rows_q6_K",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "get_rows"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "get_rows"))?,
        GgmlDType::IQ2XXS => "kernel_get_rows_iq2_xxs",
        GgmlDType::IQ2XS => "kernel_get_rows_iq2_xs",
        GgmlDType::IQ2S => "kernel_get_rows_iq2_s",
        GgmlDType::IQ3XXS => "kernel_get_rows_iq3_xxs",
        GgmlDType::IQ3S => "kernel_get_rows_iq3_s",
        GgmlDType::IQ4NL => "kernel_get_rows_iq4_nl",
        GgmlDType::IQ4XS => "kernel_get_rows_iq4_xs",
        GgmlDType::Q2_0 => "kernel_get_rows_q2_0",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qget_rows {name} ids={ids_len} hidden={hidden_size}"
    );

    let thread_groups_count = MTLSize {
        width: ids_len,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            src,
            (ids, ids_offset),
            Output::new(dst),
            hidden_size as i64,
            row_stride as u64,
            0u64,
            ids_len as i64,
            core::mem::size_of::<u32>() as u64,
            0u64,
            dst_row_stride as u64,
            0u64
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}

/// Gate and up of one routed layer in ONE launch (`kernel_mul_mv_id2_<q>_<triple>`):
/// `src0a` and `src0b` are two stacked weights of the same kind and
/// `[n_experts, n, k]`; `dst` is `[2, batch, topk, n]` in the triple's store
/// leg (plane set 0 from `src0a`, 1 from `src0b`). `triple` is
/// [`call_quantized_matmul_mv_id_t`]'s, with the same reach — the f16 arm on
/// quantized weight rows only — and each plane computes exactly what that
/// launch computes for it under the same triple.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_id2_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    triple: GemvTriple,
    (n_experts, n, k): (usize, usize, usize),
    (batch, topk, in_dim1): (usize, usize, usize),
    src0a: &Buffer,
    src0b: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    triple.check(device, "routed quantized matmul-vec pair")?;
    if triple != GemvTriple::F32F32F32 && !dtype.has_gemv_triple() {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "non-f32 GemvTriple",
            "dense/Q8 routed quantized matmul-vec pair",
        ));
    }
    let stem = mv_id_stem(dtype)?;
    let name = format!("kernel_mul_mv_id2_{stem}_{}", triple.name_suffix());
    let (act_size, store_size) = triple_leg_bytes(triple);
    if in_dim1 != 1 && in_dim1 != topk {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "routed matmul-vec activation dim1 must be 1 or topk",
            "routed quantized matmul-vec pair",
        ));
    }
    let nei0 = topk as i64;
    let nei1 = batch as i64;
    let nbi1 = (topk * core::mem::size_of::<u32>()) as u64;
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = n_experts as i64;
    let row_bytes = ((k / dtype_block_size(dtype)) * dtype_type_size(dtype)) as u64;
    let nb00 = dtype_type_size(dtype) as u64;
    let nb01 = row_bytes;
    let nb02 = n as u64 * row_bytes;
    let ne10 = k as i64;
    let ne11 = in_dim1 as i64;
    let ne12 = batch as i64;
    let ne13 = 1i64;
    let nb10 = act_size;
    let nb11 = k as u64 * act_size;
    let nb12 = (in_dim1 * k) as u64 * act_size;
    let ne0 = n as i64;
    let ne1 = topk as i64;
    let nb1 = n as u64 * store_size;

    let (nth0, nth1, align) = mv_threadgroup(dtype);
    let thread_groups_count = MTLSize {
        width: divide(n, align),
        height: 1,
        depth: 2 * batch * topk,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Quantized, name.clone(), mv_constants(dtype))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    if needs_iq_shared(dtype) {
        encoder.set_threadgroup_memory_length(0, 8192);
    }
    debug_group!(
        encoder,
        "qmm_mv_id2 {name} E={n_experts} B={batch} topk={topk} K={k} N={n}"
    );

    set_params!(
        encoder,
        (
            src0a,
            src0b,
            (src1, src1_offset),
            Output::new(dst),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[cfg(test)]
mod arm_tests {
    use super::*;
    use crate::kernels::metal::{Commands, ResidencySet};
    use crate::kernels::RESOURCE_OPTIONS;
    use std::sync::Arc;

    fn setup() -> (Device, Kernels, Commands, Buffer) {
        let device = Device::system_default().expect("a Metal device");
        let queue = device.new_command_queue().unwrap();
        let commands = Commands::new(queue, Arc::new(ResidencySet::new(&device))).unwrap();
        let buffer = device.new_buffer(4096, RESOURCE_OPTIONS).unwrap();
        (device, Kernels::new(), commands, buffer)
    }

    fn dispatch_plain(triple: GemvTriple) -> Result<(), MetalKernelError> {
        let (device, kernels, commands, buf) = setup();
        let encoder = commands.command_encoder().unwrap();
        call_quantized_matmul_mv_t(
            &device,
            &encoder,
            &kernels,
            GgmlDType::Q4K,
            triple,
            (1, 1, 32, 256),
            &buf,
            0,
            &buf,
            0,
            &buf,
        )
    }

    fn dispatch_routed(triple: GemvTriple) -> Result<(), MetalKernelError> {
        let (device, kernels, commands, buf) = setup();
        let encoder = commands.command_encoder().unwrap();
        call_quantized_matmul_mv_id_t(
            &device,
            &encoder,
            &kernels,
            GgmlDType::Q4K,
            triple,
            (2, 32, 256),
            (1, 1, 1),
            &buf,
            &buf,
            0,
            &buf,
            0,
            &buf,
        )
    }

    /// The int8 arm needs a feature no family registers, so both dispatches
    /// refuse it by that feature's name before any kernel is resolved — not by
    /// the kernel body being unbuilt, which is the second check.
    #[test]
    fn the_int8_arm_is_refused_by_the_feature_it_lacks() {
        for (err, op) in [
            (dispatch_plain(GemvTriple::I8I32I8), "quantized matmul-vec"),
            (
                dispatch_routed(GemvTriple::I8I32I8),
                "routed quantized matmul-vec",
            ),
        ] {
            let err = err.unwrap_err();
            assert!(matches!(err, MetalKernelError::ArmRefused(_)), "{err}");
            let msg = err.to_string();
            assert!(
                msg.starts_with(&format!(
                    "Metal arm `{op} (i8, i32, i8)` refused: this GPU does not execute `int8-mma`"
                )),
                "{msg}"
            );
        }
    }

    /// The float arms pass the device check here, which registers `f16` on
    /// every Apple family; `LLOOM_METAL_DISABLE=f16` makes the same dispatch
    /// refuse, naming `f16`.
    #[test]
    fn the_f16_arm_follows_the_device_set() {
        let (device, ..) = setup();
        assert!(GemvTriple::F32F32F32.check(&device, "op").is_ok());
        let f16 = GemvTriple::F16F32F16.check(&device, "op");
        if arms::for_device(&device).has(Feature::F16) {
            assert!(f16.is_ok(), "{f16:?}");
        } else {
            let msg = f16.unwrap_err().to_string();
            assert!(msg.contains("`op (f16, f32, f16)` refused"), "{msg}");
            assert!(msg.contains("does not execute `f16`"), "{msg}");
            let routed = dispatch_routed(GemvTriple::F16F32F16)
                .unwrap_err()
                .to_string();
            assert!(routed.contains("does not execute `f16`"), "{routed}");
            eprintln!("f16 withdrawn: {msg}");
        }
    }
}
