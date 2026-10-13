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

/// The `head_dim`s `steel_attention` (`call_sdpa_full`) is instantiated at;
/// f32 has no 512 (`sdpa_full_tile`).
pub const SDPA_FULL_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

/// The `head_dim`s the `sdpa_vector` entry points dispatch (`call_sdpa_vector`,
/// its two-pass and paged siblings). Two fewer than the metallib instantiates:
/// 72 and 80 have kernels no entry point names
/// (`sdpa_vector_declares_variants_the_name_table_cannot_reach`).
pub const SDPA_VECTOR_HEAD_DIMS: [usize; 6] = [32, 64, 96, 128, 256, 512];

/// Whether every `sdpa_vector` pipeline at `(head_dim, itype)` admits the
/// threadgroup width it is dispatched at: the one-pass kernel and the two-pass
/// kernels' second pass at 1024 threads, the first pass at 256. A pipeline
/// admitting fewer cannot run its dispatch, and the output is left unwritten.
/// Register demand grows with the head dim, so this is a property of the
/// device and toolchain, read from the compiled pipelines; a load-time check,
/// never a per-dispatch one.
pub fn sdpa_vector_admits_its_widths(
    device: &Device,
    kernels: &Kernels,
    head_dim: usize,
    itype: SdpaDType,
) -> Result<bool, MetalKernelError> {
    if !SDPA_VECTOR_HEAD_DIMS.contains(&head_dim) {
        return Ok(false);
    }
    let t = match itype {
        SdpaDType::F16 => "float16_t",
        SdpaDType::BF16 => "bfloat16_t",
        SdpaDType::F32 => "float",
    };
    let constants = || {
        Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]))
    };
    let cells = [
        (format!("sdpa_vector_{t}_{head_dim}"), true, 1024),
        (format!("sdpa_vector_2pass_1_{t}_{head_dim}"), true, 8 * 32),
        (format!("sdpa_vector_2pass_2_{t}_{head_dim}"), false, 1024),
    ];
    for (name, specialized, width) in cells {
        let pipeline = if specialized {
            kernels.load_pipeline_with_constants(device, Source::Sdpa, name, constants())?
        } else {
            kernels.load_pipeline(device, Source::Sdpa, name)?
        };
        if pipeline.max_total_threads_per_threadgroup() < width {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The tile `steel_attention` runs `head_dim` at: `(BQ, BK, WM, WN)`.
///
/// `Err` for a head dim the kernel does not instantiate, and for f32 at 512
/// (the reduced tile does not fit f32 in threadgroup memory). One function so
/// the dispatch and a paged caller's `page_size % BK` check read the same
/// number.
pub fn sdpa_full_tile(
    head_dim: usize,
    itype: SdpaDType,
) -> Result<(usize, usize, usize, usize), MetalKernelError> {
    if !SDPA_FULL_HEAD_DIMS.contains(&head_dim) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
            crate::kernels::err::SdpaHeadSizeMismatch {
                variation: "full",
                got: head_dim,
                expected: SDPA_FULL_HEAD_DIMS.to_vec(),
            },
        )));
    }
    // BD=512 uses reduced tiles to fit 32KB Metal threadgroup memory (f16/bf16 only).
    if head_dim == 512 {
        if itype == SdpaDType::F32 {
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "full (f32 unsupported at head_dim=512)",
                    got: head_dim,
                    expected: vec![32, 64, 72, 80, 96, 128, 256],
                },
            )));
        }
        Ok((8, 8, 1, 1))
    } else if head_dim == 256 && itype == SdpaDType::F32 {
        // The 32x16 tile's f32 Q block alone is 32KB: the reduced tile.
        Ok((8, 8, 1, 1))
    } else {
        let bk = if head_dim < 128 { 32 } else { 16 };
        Ok((32, bk, 4, 1))
    }
}

/// A page pool's table for `call_sdpa_full_with` (`is_paged`, issue #708).
///
/// K and V are one pool `[1, H_kv, pool_tokens, D]` shared by every sequence
/// of the dispatch; sequence `b`'s key `t` sits at slot `table[b *
/// table_stride + t / page_size] * page_size + t % page_size`. The kernel
/// re-seats its K/V block loaders once per `BK`-key block through the table
/// and walks the page linearly from there, so `page_size` must be a multiple
/// of the tile's `BK` ([`sdpa_full_tile`]) — a block that straddled two pages
/// would read the wrong keys silently (§3.5). The K/V batch stride is ignored
/// on this arm: the batch row is the table row.
#[derive(Copy, Clone, Debug)]
pub struct SdpaPageTable<'a> {
    /// `u32` page indices, every sequence's table back to back.
    pub table: &'a Buffer,
    pub table_offset: usize,
    /// Entries between two sequences' tables.
    pub table_stride: usize,
    /// KV tokens per page.
    pub page_size: usize,
    /// `i32` live key counts: sequence `b` reads entry `b * n_keys_stride`,
    /// so a stride of 0 is one count for the whole dispatch (`k_shape[2]`)
    /// and a stride of 1 a count per row — B rows of different live lengths
    /// in one dispatch, each over its own key blocks and its own causal
    /// offset (`batched-decode-design.md` §12.2 change 3).
    pub n_keys: &'a Buffer,
    pub n_keys_offset: usize,
    pub n_keys_stride: usize,
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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_full(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_sdpa_full_with(
            device,
            ep,
            kernels,
            q_offset,
            q_shape,
            q_strides,
            q_buffer,
            k_offset,
            k_shape,
            k_strides,
            k_buffer,
            v_offset,
            v_buffer,
            v_strides,
            mask_type,
            mask_buffer,
            m_strides,
            output,
            o_strides,
            scale,
            do_causal,
            itype,
            None,
        )
    }
}

/// As [`call_sdpa_full`], with the K/V address selected: contiguous strides
/// (`None`) or a page pool read through per-sequence tables (`Some`, the
/// `is_paged` function constant — issue #708, the `q_seq > 1` half of #148's
/// page-walking read). One body for both arms, so the paged arm cannot drift
/// from the kernel it is one address away from.
///
/// On the paged arm `k_shape[2]` is the LIVE key count (`kL`; the longest
/// row's when the counts are per row) and `k_strides`
/// / `v_strides` are the pool's `[_, head, token]` strides; the batch stride is
/// not read. The table is bound at `buffer(8)` and its params at `buffer(7)`,
/// above the mask's `5`/`6`, so the two optional arms bind independently.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_full_with(
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
    paged: Option<SdpaPageTable<'_>>,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above), passed on unchanged.
    unsafe {
        call_sdpa_full_with_mask_offset(
            device,
            ep,
            kernels,
            q_offset,
            q_shape,
            q_strides,
            q_buffer,
            k_offset,
            k_shape,
            k_strides,
            k_buffer,
            v_offset,
            v_buffer,
            v_strides,
            mask_type,
            mask_buffer,
            0,
            m_strides,
            output,
            o_strides,
            scale,
            do_causal,
            itype,
            paged,
        )
    }
}

/// [`call_sdpa_full_with`] with an explicit additive-mask byte offset.
///
/// The offset belongs to the mask's tensor view, separately from its backing
/// buffer and outer strides. The compatibility entry point above retains its
/// original zero-offset contract; callers holding a layout use this entry.
/// Four mask strides preserve the key axis, including broadcast stride zero;
/// a legacy three-stride slice retains its implied unit key stride.
///
/// # Safety
/// As [`call_sdpa_full_with`]: every buffer bound here stays unreused and
/// unwritten by the host until the command buffer it is encoded into
/// completes, and while an ICB that captured it can replay.
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_full_with_mask_offset(
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
    mask_offset: usize,
    m_strides: Option<&[usize]>,
    output: &Buffer,
    o_strides: &[usize],
    scale: f32,
    do_causal: bool,
    itype: SdpaDType,
    paged: Option<SdpaPageTable<'_>>,
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
        m_strides: [i64; 4],
    }

    /// Mirrors `AttnPagedParams` in `scaled_dot_product_attention.metal`.
    #[derive(Debug)]
    #[repr(C)]
    struct AttnPagedParams {
        page_size: i32,
        table_stride: i32,
        n_keys_stride: i32,
    }

    let bd = q_shape[q_shape.len() - 1];
    let (bq, bk, wm, wn) = sdpa_full_tile(bd, itype)?;

    if let Some(p) = &paged {
        // A key block must sit inside one page: the loader is seated once per
        // block and walks linearly from there.
        if p.page_size == 0 || !p.page_size.is_multiple_of(bk) {
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "full (paged: page_size must be a multiple of the key block)",
                    got: p.page_size,
                    expected: vec![bk],
                },
            )));
        }
    }

    let b = q_shape[0];
    let h = q_shape[1];
    let d = q_shape[3];
    let gqa_factor = q_shape[1] / k_shape[1];

    let ql = q_shape[2];
    let kl = k_shape[2];

    let align_q = ql.is_multiple_of(bq);
    // Per-row live counts differ in their alignment; the kernel then derives
    // each row's partial block from its own count, and the aligned rows take
    // the same loads they would under `align_K`.
    let align_k = kl.is_multiple_of(bk) && paged.is_none_or(|p| p.n_keys_stride == 0);
    let has_mask = mask_buffer.is_some();
    // Legacy direct callers may supply the three outer strides only. Their
    // original contract implied a contiguous key axis; tensor layouts carry
    // all four, including nonunit and broadcast-zero key strides.
    let mask_k_stride = m_strides.and_then(|s| s.get(3)).copied().unwrap_or(1);

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
        (302, Value::Bool(/* is_paged */ paged.is_some())),
        (303, Value::Bool(/* mask_k_contiguous */ mask_k_stride == 1)),
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

    let nq = ql.div_ceil(bq);
    let nk = kl.div_ceil(bk);

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
        unsafe fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    impl EncoderParam for AttnMaskParams {
        unsafe fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    impl EncoderParam for AttnPagedParams {
        unsafe fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
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
                mask_k_stride as i64,
            ],
        };

        // SAFETY: this function's contract (# Safety above).
        unsafe {
            set_params!(
                encoder,
                (
                    (q_buffer, q_offset),
                    (k_buffer, k_offset),
                    (v_buffer, v_offset),
                    Output::new(output),
                    params,
                    mask_params,
                    (mask, mask_offset)
                )
            );
        }
    } else {
        // SAFETY: this function's contract (# Safety above).
        unsafe {
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
    }

    // The paged bindings sit at fixed slots above the mask's, so they bind
    // whether or not a mask is bound: `[[buffer(7)]]` and `[[buffer(8)]]` are
    // absolute indices, and a function constant removing 5/6 shifts nothing.
    if let Some(p) = &paged {
        crate::kernels::utils::set_param(
            encoder,
            7,
            AttnPagedParams {
                page_size: p.page_size as i32,
                table_stride: p.table_stride as i32,
                n_keys_stride: p.n_keys_stride as i32,
            },
        );
        crate::kernels::utils::set_param(encoder, 8, (p.table, p.table_offset));
        crate::kernels::utils::set_param(encoder, 9, (p.n_keys, p.n_keys_offset));
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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector_with(
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
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "vector",
                    got: *other,
                    expected: SDPA_VECTOR_HEAD_DIMS.to_vec(),
                },
            )))
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
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

/// `sdpa_vector_paged`: [`call_sdpa_vector`] for `rows` sequences in ONE
/// dispatch over a shared page pool — `q [rows · nh, D]`, `k` / `v` one
/// layer's pool `[n_kv_heads, tokens, D]` (`k_stride` / `v_stride` the head
/// stride in elements), `table` (`u32`, `rows × table_stride`: entry `r ·
/// table_stride + p` is row `r`'s page `p`) and `n_keys` (`i32 [rows]`, the
/// live count per row). Row `r`'s `nh` heads are threadgroups `r · nh ..`;
/// its key `i` is pool token `table[r][i / page_size] · page_size + i %
/// page_size`. Per row the arithmetic is `sdpa_vector`'s over that row's keys
/// laid contiguously, bit for bit.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector_paged(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    nh: usize,
    gqa_factor: usize,
    head_dim: usize,
    q_offset: usize,
    q_buffer: &Buffer,
    k_offset: usize,
    k_buffer: &Buffer,
    k_stride: usize,
    v_offset: usize,
    v_buffer: &Buffer,
    v_stride: usize,
    table: &Buffer,
    table_offset: usize,
    table_stride: usize,
    n_keys: &Buffer,
    n_keys_offset: usize,
    page_size: usize,
    output: &Buffer,
    alpha: f32,
    itype: SdpaDType,
    // The 2-pass arm (every row at `≥ TWO_PASS_K_THRESHOLD` keys, the
    // per-row `ops::sdpa`'s choice there): the pass-1 scratch
    // `(partials [rows·nh, 32, D] f32, sums [rows·nh, 32], maxs [rows·nh,
    // 32])`, then `sdpa_vector_2pass_2` as the contiguous kernel runs it.
    two_pass: Option<(&Buffer, &Buffer, &Buffer)>,
) -> Result<(), MetalKernelError> {
    if let Some((partials, sums, maxs)) = two_pass {
        let name = match (head_dim, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_1_paged_float16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_1_paged_float_512",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                    crate::kernels::err::SdpaHeadSizeMismatch {
                        variation: "vector_2pass_1_paged",
                        got: other,
                        expected: SDPA_VECTOR_HEAD_DIMS.to_vec(),
                    },
                )))
            }
        };
        let name2 = match (head_dim, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (512, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_512",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (512, SdpaDType::F32) => "sdpa_vector_2pass_2_float_512",
            _ => unreachable!("the pass-1 match above covers the head sizes"),
        };
        if rows == 0 || nh == 0 || gqa_factor == 0 || page_size == 0 || table_stride == 0 {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "sdpa_vector_paged: rows {rows} nh {nh} gqa {gqa_factor} page {page_size} stride {table_stride}"
            )));
        }
        {
            let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            debug_group!(
                encoder,
                "sdpa_vector_2pass_1_paged bk={head_dim} rows={rows} nh={nh}"
            );
            // SAFETY: this function's contract (# Safety above).
            unsafe {
                set_params!(
                    encoder,
                    (
                        (q_buffer, q_offset),
                        (k_buffer, k_offset),
                        (v_buffer, v_offset),
                        Output::new(partials),
                        Output::new(sums),
                        Output::new(maxs),
                        (table, table_offset),
                        (n_keys, n_keys_offset),
                        gqa_factor as i32,
                        nh as i32,
                        table_stride as i32,
                        page_size as i32,
                        k_stride,
                        v_stride,
                        alpha,
                        1.0f32
                    )
                );
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: rows * nh,
                    depth: SDPA_2PASS_BLOCKS,
                },
                MTLSize {
                    width: 8 * 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        {
            let pipeline = kernels.load_pipeline(device, Source::Sdpa, name2)?;
            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            debug_group!(
                encoder,
                "sdpa_vector_2pass_2 (paged rows) bk={head_dim} rows={rows}"
            );
            // SAFETY: this function's contract (# Safety above).
            unsafe {
                set_params!(encoder, (partials, sums, maxs, Output::new(output)));
            }
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: rows * nh,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
        }
        return Ok(());
    }
    let name = match (head_dim, itype) {
        (32, SdpaDType::F16) => "sdpa_vector_paged_float16_t_32",
        (64, SdpaDType::F16) => "sdpa_vector_paged_float16_t_64",
        (96, SdpaDType::F16) => "sdpa_vector_paged_float16_t_96",
        (128, SdpaDType::F16) => "sdpa_vector_paged_float16_t_128",
        (256, SdpaDType::F16) => "sdpa_vector_paged_float16_t_256",
        (512, SdpaDType::F16) => "sdpa_vector_paged_float16_t_512",
        (32, SdpaDType::F32) => "sdpa_vector_paged_float_32",
        (64, SdpaDType::F32) => "sdpa_vector_paged_float_64",
        (96, SdpaDType::F32) => "sdpa_vector_paged_float_96",
        (128, SdpaDType::F32) => "sdpa_vector_paged_float_128",
        (256, SdpaDType::F32) => "sdpa_vector_paged_float_256",
        (512, SdpaDType::F32) => "sdpa_vector_paged_float_512",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "vector_paged",
                    got: other,
                    expected: SDPA_VECTOR_HEAD_DIMS.to_vec(),
                },
            )))
        }
    };
    if rows == 0 || nh == 0 || gqa_factor == 0 || page_size == 0 || table_stride == 0 {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "sdpa_vector_paged: rows {rows} nh {nh} gqa {gqa_factor} page {page_size} stride {table_stride}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "sdpa_vector_paged bk={head_dim} rows={rows} nh={nh}"
    );
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(output),
                (table, table_offset),
                (n_keys, n_keys_offset),
                gqa_factor as i32,
                nh as i32,
                table_stride as i32,
                page_size as i32,
                k_stride,
                v_stride,
                alpha,
                1.0f32
            )
        );
    }
    let grid_dims = MTLSize {
        width: 1,
        height: rows * nh,
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

/// The `head_dim`s the quantized-KV kernels are instantiated at: LFM2's 64,
/// 128, and Qwen4-exp's 256 (`DESIGN.md` §10.7).
pub const KV_QUANT_HEAD_DIMS: [usize; 3] = [64, 128, 256];

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
            // `head_dim` (64, 128 or 256) and this division is exact. Asserted
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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_quantize_kv(
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
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "quantize_kv (bf16 not instantiated)",
                    got: head_dim,
                    expected: KV_QUANT_HEAD_DIMS.to_vec(),
                },
            )))
        }
    };
    if !KV_QUANT_HEAD_DIMS.contains(&head_dim) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
            crate::kernels::err::SdpaHeadSizeMismatch {
                variation: "quantize_kv",
                got: head_dim,
                expected: KV_QUANT_HEAD_DIMS.to_vec(),
            },
        )));
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

    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }

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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_dequantize_kv(
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
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "dequantize_kv (bf16 not instantiated)",
                    got: head_dim,
                    expected: KV_QUANT_HEAD_DIMS.to_vec(),
                },
            )))
        }
    };
    if !KV_QUANT_HEAD_DIMS.contains(&head_dim) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
            crate::kernels::err::SdpaHeadSizeMismatch {
                variation: "dequantize_kv",
                got: head_dim,
                expected: KV_QUANT_HEAD_DIMS.to_vec(),
            },
        )));
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

    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (codes, scales, biases, Output::new(output), params)
        );
    }

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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector_quant(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        encode_sdpa_vector_quant(
            device, ep, kernels, q_offset, q_shape, q_buffer, k_codes, k_scales, k_biases, v_codes,
            v_scales, v_biases, output, params, None, bits, itype,
        )
    }
}

/// [`call_sdpa_vector_quant`] over a selection of the cache: key `i` is read
/// at cell `sel[i]` (`u32`, at byte offset `sel_offset`), and the params' `N`
/// is the selection's length. QSA's indexer cells, read from the planes
/// without gathering them first (`DESIGN.md` §10.7).
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector_quant_sel(
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
    sel: (&Buffer, usize),
    bits: KvQuantBits,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        encode_sdpa_vector_quant(
            device,
            ep,
            kernels,
            q_offset,
            q_shape,
            q_buffer,
            k_codes,
            k_scales,
            k_biases,
            v_codes,
            v_scales,
            v_biases,
            output,
            params,
            Some(sel),
            bits,
            itype,
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
unsafe fn encode_sdpa_vector_quant(
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
    sel: Option<(&Buffer, usize)>,
    bits: KvQuantBits,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let head_dim = *q_shape.last().unwrap();
    let itype_repr = match itype {
        SdpaDType::F16 => "float16_t",
        SdpaDType::F32 => "float",
        SdpaDType::BF16 => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                crate::kernels::err::SdpaHeadSizeMismatch {
                    variation: "sdpa_vector_quant (bf16 not instantiated)",
                    got: head_dim,
                    expected: KV_QUANT_HEAD_DIMS.to_vec(),
                },
            )))
        }
    };
    if !KV_QUANT_HEAD_DIMS.contains(&head_dim) {
        return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
            crate::kernels::err::SdpaHeadSizeMismatch {
                variation: "sdpa_vector_quant",
                got: head_dim,
                expected: KV_QUANT_HEAD_DIMS.to_vec(),
            },
        )));
    }
    let family = if sel.is_some() {
        "sdpa_vector_quant_sel"
    } else {
        "sdpa_vector_quant"
    };
    let name = format!("{family}_{itype_repr}_{head_dim}_{}", bits.name());

    let b = q_shape[0] * q_shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Sdpa, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "{family} hd={head_dim} bits={} B={b}", bits.name());

    match sel {
        None => {
            // SAFETY: this function's contract (# Safety above).
            unsafe {
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
            }
        }
        Some(sel) => {
            // SAFETY: this function's contract (# Safety above).
            unsafe {
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
                        params,
                        sel
                    )
                );
            }
        }
    }

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
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_sdpa_vector_2pass(
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
                return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                    crate::kernels::err::SdpaHeadSizeMismatch {
                        variation: "vector_2pass_1",
                        got: *other,
                        expected: SDPA_VECTOR_HEAD_DIMS.to_vec(),
                    },
                )))
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

        // SAFETY: this function's contract (# Safety above).
        unsafe {
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
        }

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
                return Err(MetalKernelError::SdpaHeadSizeMismatch(Box::new(
                    crate::kernels::err::SdpaHeadSizeMismatch {
                        variation: "vector_2pass_2",
                        got: *other,
                        expected: SDPA_VECTOR_HEAD_DIMS.to_vec(),
                    },
                )))
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

        // SAFETY: this function's contract (# Safety above).
        unsafe {
            set_params!(encoder, (intermediate, sums, maxs, Output::new(output)));
        }

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
