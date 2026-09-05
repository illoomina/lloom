use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, ParamStyle, SdpaQuantParams, SdpaQuantWriteParams,
    SdpaVectorParams,
};
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, EncoderParam,
    Kernels, MetalKernelError, Output, Source, Value,
};
use objc2_metal::MTLSize;

/// Trailing alignment of the packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirror rather than written as a
/// literal, and `every_family_params_layout_matches_metal` is what proves that
/// mirror agrees with `scaled_dot_product_attention.metal`.
///
/// This one does work rather than being a formality: the fields sum to 28 and
/// the struct is 32, because two 4-byte `int`s precede a `size_t`. Without the
/// trailing pad the capture would hand the kernel 28 bytes for a 32-byte read.
const SDPA_VECTOR_PARAMS_ALIGN: usize = core::mem::align_of::<SdpaVectorParams>();

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SdpaDType {
    BF16,
    F16,
    F32,
}

/// The tiled multi-query attention (`steel_attention`): one threadgroup per
/// (`BQ` query block, head), online softmax over `BK`-wide K/V blocks, f32
/// accumulation in simdgroup matrices, GQA-native (`kv_head_idx = head /
/// gqa_factor`), K/V read at their own dtype through `[B, H, L]` strides.
///
/// Supported: head dim in {32, 64, 72, 80, 96, 128, 256, 512} (512 not at f32),
/// f16/bf16/f32, `q_seq <= k_seq`. `do_causal` masks in-kernel with the query
/// rows offset to the END of the key range (`qL_off = kL - qL`), which is the
/// cache shape: a verify or prefill pass over the last `qL` positions of `kL`.
/// An explicit mask disables `do_causal` (it already encodes causality).
///
/// Every reduction is fixed-order — blocks ascend, the row reductions are the
/// simdgroup tree — so a given build is bit-stable run to run (§2.3.3).
///
/// # No packed sibling, and it needs none — issue #103
///
/// This is the prefill path, which #97 deliberately left on the generic route
/// (`DESIGN.md` §6.2a), so it is on no decode dispatch position. That alone
/// would make a packed sibling optional rather than required. The stronger
/// reason is that **it has nothing to promote**: the `attention` kernel takes
/// `const constant AttnParams* params [[buffer(4)]]` and, when masked,
/// `const constant AttnMaskParams* mask_params [[buffer(5)]]`. Those are
/// *pointers*, not `constant &` scalars, so the file's 32 `constant &`
/// parameters are all on the `sdpa_vector` signatures.
///
/// A pointer parameter is bindable by `setKernelBuffer`, which is the only
/// binding primitive an `MTLIndirectComputeCommand` has (`DESIGN.md` §3.7c) —
/// the constraint is `setBytes`, not buffer count (§11.3d). So what stands
/// between this entry point and an ICB is the *host* side: candle reaches
/// `setBytes` here through `impl EncoderParam for AttnParams`, which writes the
/// struct inline. Converting it is therefore a change to how the existing
/// struct is uploaded rather than a second kernel from one body, which is a
/// different shape of work from every conversion #38 through #103 — and it buys
/// nothing until prefill is on a replayed position. Recorded rather than done.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_strides: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_strides: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    v_strides: &[usize],
    mask_type: Option<SdpaDType>,
    mask_buffer: Option<&Buffer>,
    m_strides: Option<&[usize]>,
    output: &Buffer,
    o_strides: &[usize],
    scale: f32,
    do_causal: bool,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct AttnParams {
        b: i32,
        h: i32,
        d: i32,
        ql: i32,
        kl: i32,
        gqa_factor: i32,
        scale: f32,
        softcapping: f32, // Must match Metal struct layout (1.0 = disabled)
        nq: i32,
        nk: i32,
        nq_aligned: i32,
        nk_aligned: i32,
        ql_rem: i32,
        kl_rem: i32,
        ql_off: i32,
        q_strides: [i64; 3],
        k_strides: [i64; 3],
        v_strides: [i64; 3],
        o_strides: [i64; 3],
    }

    #[derive(Debug)]
    #[repr(C)]
    struct AttnMaskParams {
        m_strides: [i64; 3],
    }

    let bd = q_shape[q_shape.len() - 1];
    if ![32, 64, 72, 80, 96, 128, 256, 512].contains(&bd) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "full",
            got: bd,
            expected: vec![32, 64, 72, 80, 96, 128, 256, 512],
        });
    };

    // BD=512 uses reduced tiles to fit 32KB threadgroup memory (f16/bf16 only).
    let (bq, bk, wm, wn): (usize, usize, usize, usize) = if bd == 512 {
        if itype == SdpaDType::F32 {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "full (f32 unsupported at head_dim=512)",
                got: bd,
                expected: vec![32, 64, 72, 80, 96, 128, 256],
            });
        }
        (8, 8, 1, 1)
    } else {
        let bk = if bd < 128 { 32 } else { 16 };
        (32, bk, 4, 1)
    };

    let b = q_shape[0];
    let h = q_shape[1];
    let d = q_shape[3];
    let gqa_factor = q_shape[1] / k_shape[1];

    let ql = q_shape[2];
    let kl = k_shape[2];

    let align_q = (ql % bq) == 0;
    let align_k = (kl % bk) == 0;
    let has_mask = mask_buffer.is_some();

    // When an explicit additive mask is supplied it already encodes causality.
    // Also enabling the in-kernel `do_causal` path double-applies causality and,
    // at periodic sequence lengths, leaves a query row fully masked -> softmax
    // normalizer sum(exp) == 0 -> final divide computes 0/0 = NaN logits.
    let do_causal = do_causal && !has_mask;

    let itype_repr = match itype {
        SdpaDType::BF16 => "bfloat16",
        SdpaDType::F16 => "float16",
        SdpaDType::F32 => "float32",
    };
    let mask_repr = match mask_type {
        Some(SdpaDType::BF16) => "bfloat16",
        Some(SdpaDType::F16) => "float16",
        Some(SdpaDType::F32) => "float32",
        None => itype_repr,
    };
    let name =
        format!("steel_attention_{itype_repr}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{wn}_mask{mask_repr}");

    let constants = Some(ConstantValues::new(vec![
        (200, Value::Bool(/* align_Q */ align_q)),
        (201, Value::Bool(/* align_K */ align_k)),
        (300, Value::Bool(/* has_mask */ has_mask)),
        (301, Value::Bool(/* do_causal */ do_causal)),
    ]));

    #[cfg(feature = "debug-labels")]
    let name_for_label = name.clone();
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "sdpa_full {name_for_label} B={b} H={h} D={d} QL={ql} KL={kl}"
    );

    let nq = (ql + bq - 1) / bq;
    let nk = (kl + bk - 1) / bk;

    let nq_aligned = ql / bq;
    let nk_aligned = kl / bk;

    let params = AttnParams {
        b: b as i32,
        h: h as i32,
        d: d as i32,
        ql: ql as i32,
        kl: kl as i32,
        gqa_factor: gqa_factor as i32,
        scale,
        softcapping: 1.0, // SDPA full doesn't support softcapping, always 1.0
        nq: nq as i32,
        nk: nk as i32,
        nq_aligned: nq_aligned as i32,
        nk_aligned: nk_aligned as i32,
        ql_rem: ql.wrapping_sub(nq_aligned * bq) as i32,
        kl_rem: kl.wrapping_sub(nk_aligned * bk) as i32,
        ql_off: kl.wrapping_sub(ql) as i32,
        q_strides: [
            q_strides[0] as i64,
            q_strides[1] as i64,
            q_strides[2] as i64,
        ],
        k_strides: [
            k_strides[0] as i64,
            k_strides[1] as i64,
            k_strides[2] as i64,
        ],
        v_strides: [
            v_strides[0] as i64,
            v_strides[1] as i64,
            v_strides[2] as i64,
        ],
        o_strides: [
            o_strides[0] as i64,
            o_strides[1] as i64,
            o_strides[2] as i64,
        ],
    };

    impl EncoderParam for AttnParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    impl EncoderParam for AttnMaskParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    if let Some(mask) = mask_buffer {
        let mask_strides = m_strides.unwrap();
        let mask_params = AttnMaskParams {
            m_strides: [
                mask_strides[0] as i64,
                mask_strides[1] as i64,
                mask_strides[2] as i64,
            ],
        };

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(output),
                params,
                mask_params,
                mask
            )
        );
    } else {
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(output),
                params
            )
        );
    }

    let grid_dims = MTLSize {
        width: nq,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: 32,
        height: wm,
        depth: wn,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);

    Ok(())
}

/// SDPA full is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    call_sdpa_vector_with(
        device,
        ep,
        kernels,
        q_offset,
        q_shape,
        q_buffer,
        k_offset,
        k_shape,
        k_stride,
        k_buffer,
        v_offset,
        v_stride,
        v_buffer,
        output,
        alpha,
        softcapping,
        itype,
        ParamStyle::default(),
    )
}

/// As [`call_sdpa_vector`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind — which is the property that makes the bit-identical test
/// meaningful.
///
/// # Why this family needed converting, and why it did not before
///
/// `MTLIndirectComputeCommand` has no `setBytes` in any form (`DESIGN.md`
/// §3.7c), so every kernel on a replayed dispatch position must bind its
/// constants from a buffer. §11.3h deferred this family on the ground that
/// LFM2 dispatched none of it and that Phase 4 was about to rewrite the
/// signature. #97 (§6.2a) discharged the first half — LFM2 now dispatches
/// `sdpa_vector_float16_t_64` 8 times per decode token — and discharged the
/// second half in the opposite direction: it needed **no `.metal` change at
/// all**, because `call_sdpa_vector` already computed `gqa_factor` and the
/// kernel already accumulated in F32. So the signature was never moving; it
/// was already right and merely unused.
///
/// # Scope
///
/// Every `sdpa_vector` variant carries both styles, not only the one LFM2
/// dispatches. `call_sdpa_vector_2pass` is **not** converted: it is
/// FlashDecoding's entry point (`DESIGN.md` §10.4), which is Phase 5 item 16
/// and does not exist yet, so it is on no dispatch position at all.
/// `call_sdpa_full` is not converted either, and for a different reason — it
/// already takes its parameters as a `constant AttnParams*`, which is a
/// pointer rather than a `setBytes` scalar, so it has no `constant &` to
/// promote. See the note on that function.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
    let n = k_shape[2] as i32;
    let b = (q_shape[0] * q_shape[1]) as i32;
    // `usize`, bound against a `constant size_t &` -- 8 bytes on both sides.
    // The width mismatch #38 found (`usize` against a 4-byte `constant uint &`,
    // benign under `setBytes` and wrong once packed) does not occur in this
    // family; `gqa_factor` and `n` are `i32` against `constant int &`, which
    // also agrees. Checked by enumerating the file's `constant &` parameters.
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "sdpa_vector_float16_t_32",
        (64, SdpaDType::F16) => "sdpa_vector_float16_t_64",
        (96, SdpaDType::F16) => "sdpa_vector_float16_t_96",
        (128, SdpaDType::F16) => "sdpa_vector_float16_t_128",
        (256, SdpaDType::F16) => "sdpa_vector_float16_t_256",
        (512, SdpaDType::F16) => "sdpa_vector_float16_t_512",
        (32, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_32",
        (64, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_64",
        (96, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_96",
        (128, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_128",
        (256, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_256",
        (512, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_512",
        (32, SdpaDType::F32) => "sdpa_vector_float_32",
        (64, SdpaDType::F32) => "sdpa_vector_float_64",
        (96, SdpaDType::F32) => "sdpa_vector_float_96",
        (128, SdpaDType::F32) => "sdpa_vector_float_128",
        (256, SdpaDType::F32) => "sdpa_vector_float_256",
        (512, SdpaDType::F32) => "sdpa_vector_float_512",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector",
                got: *other,
                expected: vec![32, 64, 96, 128, 256, 512],
            })
        }
    };

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    // `false` on both styles, and that is what makes the packed variants
    // reachable at all: the mask arguments are gated on this function constant,
    // so when it is false they are removed from the signature rather than left
    // unbound. That is why #41's `()` hazard -- an argument that binds nothing
    // and consumes no slot, shifting every later binding one low and hanging
    // the GPU -- cannot arise here.
    let constants = Some(ConstantValues::new(vec![(
        20,
        Value::Bool(/* sdpa_vector_has_mask */ false),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(
        device,
        Source::Sdpa,
        style.kernel_name(name),
        constants,
    )?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "sdpa_vector bk={bk} B={b} N={n}");

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, kv_seq, hidden)

    // The scalars come last in the classical argument list while the packed
    // struct is bound at slot 0, so the capture renumbers the four buffers to
    // 1..4 -- which is what `sdpa_vector_packed`'s explicit `[[buffer(N)]]`
    // attributes declare. The classical entry point keeps 0..3.
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            Output::new(output),
            gqa_factor,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SDPA_VECTOR_PARAMS_ALIGN)?;

    let grid_dims = MTLSize {
        width: 1,
        height: b as usize,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// The bit width a quantized KV plane stores its codes at (issue #218).
///
/// Two arms rather than a `bits: usize`, so an unsupported width is a compile
/// error at the call site rather than a runtime `bail!` — #58's mechanism, and
/// the one #241 found missing when a two-armed conditional rendered a
/// three-armed axis.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum KvQuantBits {
    /// 8 bits per code, one byte per element.
    Eight,
    /// 4 bits per code, two elements per byte, low nibble first.
    Four,
}

impl KvQuantBits {
    /// Every arm, for exhaustive tests and a harness flag (`Sizing::ALL`'s
    /// argument, #58 / §11.3i).
    pub const ALL: [KvQuantBits; 2] = [KvQuantBits::Eight, KvQuantBits::Four];

    /// Bits per code.
    pub fn bits(self) -> usize {
        match self {
            KvQuantBits::Eight => 8,
            KvQuantBits::Four => 4,
        }
    }

    /// The arm's name, as a kernel name and a `RESULT` line spell it.
    pub fn name(self) -> &'static str {
        match self {
            KvQuantBits::Eight => "8",
            KvQuantBits::Four => "4",
        }
    }

    /// Bytes a code plane of `n` elements occupies at this width.
    ///
    /// **The one place this arithmetic lives.** The kernel computes the same
    /// quantity with `sdpa_code_bytes<BITS>`, and
    /// `kv_quant_code_bytes_matches_the_kernel` is what keeps the two honest —
    /// an off-by-one here is a plane that overlaps its neighbour, which is
    /// §3.5's silent-wrong-answer class rather than a crash.
    pub fn code_bytes(self, n: usize) -> usize {
        match self {
            KvQuantBits::Eight => n,
            // Callers pass whole head vectors, so `n` is a multiple of
            // `head_dim` (64 or 128) and this division is exact. Asserted
            // rather than assumed because a half-byte tail would silently
            // truncate the last code.
            KvQuantBits::Four => {
                debug_assert_eq!(n % 2, 0, "an int4 plane needs an even code count");
                n / 2
            }
        }
    }

    /// The number of distinct codes, as the affine range divides by.
    pub fn levels(self) -> f32 {
        match self {
            KvQuantBits::Eight => 255.0,
            KvQuantBits::Four => 15.0,
        }
    }
}

impl SdpaQuantWriteParams {
    /// The fields for one append, computed in one place.
    ///
    /// `FlashPartialParams::for_step`'s shape and for its reason: a caller that
    /// computed these itself could compute them differently from the tests, and
    /// the units differ per field (`code_head_stride` is **bytes**, the other
    /// two are **elements**), which is exactly where that goes wrong.
    ///
    /// `capacity` is the slot's token capacity, not its live length: the planes
    /// are allocated once at capacity, so a head's stride does not move as the
    /// cache fills — §6.2b's stable buffer identity, preserved at a narrower
    /// element.
    pub fn for_append(
        src_head_stride: usize,
        capacity: usize,
        head_dim: usize,
        dst_offset: usize,
        bits: KvQuantBits,
    ) -> Self {
        Self {
            src_head_stride: src_head_stride as u64,
            code_head_stride: bits.code_bytes(capacity * head_dim) as u64,
            scale_head_stride: capacity as u64,
            dst_offset: dst_offset as i32,
            pad: 0,
        }
    }
}

impl SdpaQuantParams {
    /// The fields for one decode step's attention, computed in one place.
    ///
    /// K and V share a geometry here — both are `[B, H, capacity, D]` planes
    /// of the same slot — so the two stride pairs are equal by construction on
    /// this path. **They are still separate fields**, because the kernel reads
    /// four and a struct that folded them would make a transposition
    /// unobservable, which is #81 finding 3's condition restated: the parity
    /// arm can only see a swap of fields that differ in value, and the layout
    /// check can only see a swap of fields that exist.
    pub fn for_step(
        gqa_factor: usize,
        n: usize,
        capacity: usize,
        head_dim: usize,
        scale: f32,
        softcapping: f32,
        bits: KvQuantBits,
    ) -> Self {
        let code_stride = bits.code_bytes(capacity * head_dim) as u64;
        Self {
            gqa_factor: gqa_factor as i32,
            n: n as i32,
            k_code_stride: code_stride,
            v_code_stride: code_stride,
            k_scale_stride: capacity as u64,
            v_scale_stride: capacity as u64,
            scale,
            softcapping,
        }
    }
}

/// Quantize one append's worth of K or V into a code plane and a scale plane
/// (issue #218, `DESIGN.md` §10.7).
///
/// `src` is `[B, H, seq, D]` contiguous — one token at decode, `P` at a
/// prefill — and is written at `dst_offset` in the destination planes, which is
/// the slot's live length before this append (§6.2b's moving offset, at a
/// narrower element).
///
/// **One threadgroup per (head, token), 32 threads.** The group is the head
/// vector (§16 #6), so the min/max is one `simd_min`/`simd_max` — §2.3.3 #3's
/// fixed simdgroup tree, which #265 records as the rule whose cost is
/// structurally zero.
#[allow(clippy::too_many_arguments)]
pub fn call_quantize_kv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src_offset: usize,
    src_buffer: &Buffer,
    codes: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    params: &Buffer,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    bits: KvQuantBits,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let itype_repr = match itype {
        SdpaDType::F16 => "float16_t",
        SdpaDType::F32 => "float",
        SdpaDType::BF16 => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "quantize_kv (bf16 not instantiated)",
                got: head_dim,
                expected: vec![64, 128],
            })
        }
    };
    if head_dim != 64 && head_dim != 128 {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "quantize_kv",
            got: head_dim,
            expected: vec![64, 128],
        });
    }
    let name = format!("quantize_kv_{itype_repr}_{head_dim}_{}", bits.name());

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "quantize_kv hd={head_dim} bits={} heads={n_heads} seq={seq_len}",
        bits.name()
    );

    set_params!(
        encoder,
        (
            (src_buffer, src_offset),
            Output::new(codes),
            Output::new(scales),
            Output::new(biases),
            params
        )
    );

    let grid_dims = MTLSize {
        width: n_heads,
        height: seq_len,
        depth: 1,
    };
    // 32 threads: one simdgroup owns one head vector, which is what makes the
    // range reduction a `simd_min`/`simd_max` with no threadgroup memory.
    let group_dims = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Read a quantized KV plane back as `T` (issue #218).
///
/// **What the prefill path needs.** A turn boundary re-prefills onto the
/// accumulated cache (§2.3.8d, §10.9), so a `seq_len > 1` pass must attend over
/// the whole `kv_len` -- the causal mask is `[seq_len, index_pos + seq_len]`.
/// A quantized `seq_len > 1` kernel is out of scope (§17 item 18 scopes this to
/// decode), so the cache stays quantized and prefill dequantizes it; the decode
/// steps that follow read codes directly, which is where the mechanism pays.
///
/// `n_tokens` is the LIVE length, so this reads the region that has been
/// written and not the whole capacity.
#[allow(clippy::too_many_arguments)]
pub fn call_dequantize_kv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    codes: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n_heads: usize,
    n_tokens: usize,
    head_dim: usize,
    bits: KvQuantBits,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let itype_repr = match itype {
        SdpaDType::F16 => "float16_t",
        SdpaDType::F32 => "float",
        SdpaDType::BF16 => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "dequantize_kv (bf16 not instantiated)",
                got: head_dim,
                expected: vec![64, 128],
            })
        }
    };
    if head_dim != 64 && head_dim != 128 {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "dequantize_kv",
            got: head_dim,
            expected: vec![64, 128],
        });
    }
    let name = format!("dequantize_kv_{itype_repr}_{head_dim}_{}", bits.name());

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "dequantize_kv hd={head_dim} bits={} heads={n_heads} n={n_tokens}",
        bits.name()
    );

    set_params!(
        encoder,
        (codes, scales, biases, Output::new(output), params)
    );

    let grid_dims = MTLSize {
        width: n_heads,
        height: n_tokens,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Attention over a quantized KV cache (issue #218, `DESIGN.md` §10.7).
///
/// The dispatch geometry is `call_sdpa_vector`'s exactly — one threadgroup per
/// (batch × q head), 1024 threads — because the kernel body is that kernel's
/// with the loads replaced. See the note in
/// `scaled_dot_product_attention.metal` for why the dequant is per element.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_quant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_codes: &Buffer,
    k_scales: &Buffer,
    k_biases: &Buffer,
    v_codes: &Buffer,
    v_scales: &Buffer,
    v_biases: &Buffer,
    output: &Buffer,
    params: &Buffer,
    bits: KvQuantBits,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let head_dim = *q_shape.last().unwrap();
    let itype_repr = match itype {
        SdpaDType::F16 => "float16_t",
        SdpaDType::F32 => "float",
        SdpaDType::BF16 => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "sdpa_vector_quant (bf16 not instantiated)",
                got: head_dim,
                expected: vec![64, 128],
            })
        }
    };
    if head_dim != 64 && head_dim != 128 {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "sdpa_vector_quant",
            got: head_dim,
            expected: vec![64, 128],
        });
    }
    let name = format!("sdpa_vector_quant_{itype_repr}_{head_dim}_{}", bits.name());

    let b = q_shape[0] * q_shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "sdpa_vector_quant hd={head_dim} bits={} B={b}",
        bits.name()
    );

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            k_codes,
            k_scales,
            k_biases,
            v_codes,
            v_scales,
            v_biases,
            Output::new(output),
            params
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// The CPU reference for one quantize-then-dequantize round trip.
///
/// **This is the parity reference and it is deliberately scalar.** It mirrors
/// `quantize_kv`'s arithmetic exactly, including the part that is easy to get
/// wrong: the codes are chosen against the scale **as f16 will store it**, not
/// against the f32 scale that computed the range. The kernel does the same
/// (`enc_scale`), and a reference that skipped it would disagree with a correct
/// kernel by one code on roughly half the groups.
///
/// Returns `(codes, scale, bias)` for one group of `n` values.
pub fn quantize_group_reference(
    values: &[f32],
    bits: KvQuantBits,
    // How the scale and bias are rounded for storage. `f16` for the shipping
    // path; the parameter exists so the f32 instantiation can be tested with
    // its own rounding rather than inheriting f16's.
    round: impl Fn(f32) -> f32,
) -> (Vec<u8>, f32, f32) {
    let levels = bits.levels();
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = hi - lo;
    let scale = if range > 0.0 { range / levels } else { 0.0 };

    let enc_scale = round(scale);
    let enc_bias = round(lo);
    let enc_inv = if enc_scale > 0.0 {
        1.0 / enc_scale
    } else {
        0.0
    };

    let codes: Vec<u8> = values
        .iter()
        .map(|&v| {
            let t = ((v - enc_bias) * enc_inv).round_ties_even();
            t.clamp(0.0, levels) as u8
        })
        .collect();
    (codes, enc_scale, enc_bias)
}

pub const SDPA_2PASS_BLOCKS: usize = 32;

/// SDPA vector 2pass is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    // First pass
    {
        let name_pass1 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_512",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_1_float_512",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_1",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256, 512],
                })
            }
        };

        let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
        let n = k_shape[2] as i32;
        let b = (q_shape[0] * q_shape[1]) as i32;
        let kstride = k_stride[1];
        let vstride = v_stride[1];

        let alpha = if softcapping != 1. {
            alpha / softcapping
        } else {
            alpha
        };

        let constants = Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]));

        let pipeline =
            kernels.load_pipeline_with_constants(device, Source::Sdpa, name_pass1, constants)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(encoder, "sdpa_vector_2pass pass1 bk={bk} B={b} N={n}");

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(intermediate),
                Output::new(sums),
                Output::new(maxs),
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b as usize,
            depth: SDPA_2PASS_BLOCKS,
        };
        let group_dims = MTLSize {
            width: 8 * 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Final pass
    {
        let name_pass2 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_512",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (512, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_2_float_512",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_2",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256, 512],
                })
            }
        };

        let b = q_shape[0] * q_shape[1];

        let pipeline = kernels.load_pipeline(device, Source::Sdpa, name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(encoder, "sdpa_vector_2pass pass2 bk={bk} B={b}");

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(encoder, (intermediate, sums, maxs, Output::new(output)));

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }
    Ok(())
}
