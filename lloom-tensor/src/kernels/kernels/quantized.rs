use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
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

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
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
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
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
    // The M5 native-int arm is declared-unimplemented (issue #750): its kernel
    // body is not built, so refuse the dispatch rather than resolve a name that
    // does not exist in the metallib.
    if !triple.is_implemented() {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "declared-unimplemented GemvTriple (M5 native-int)",
            "quantized matmul-vec",
        ));
    }

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
        // Dense and Q8-family weights: no genericized #750 core, f32 store only.
        GgmlDType::Q8_1
        | GgmlDType::Q8K
        | GgmlDType::F16
        | GgmlDType::BF16
        | GgmlDType::F32 => None,
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

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name.clone())?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
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
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm {name} M={ne11} K={ne00} N={ne01}");

    set_params!(
        encoder,
        (
            src0,
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
