//! FlashDecoding dispatch: partials over contiguous chunks, then an
//! index-ordered combine (`DESIGN.md` §10.4, §17 Phase 5 item 16, issue #116).
//!
//! # What this is not
//!
//! **It is not a rewiring of `call_sdpa_vector_2pass`.** That entry point ships
//! and is dispatched at `k_seq >= 1024`, which makes it easy to read as
//! FlashDecoding already existing. §10.3b establishes it is not: its chunk
//! count is **fixed at 32** whatever `kv_len` is, and its blocks read a
//! **strided interleave** rather than a contiguous range, so a block is not a
//! page, cannot be resolved through a chunk table, and cannot be the unit §10.4
//! makes equal to the page. Its index-ordered merge is safe **by accident of
//! the fixed count**, which a variable-count combine does not inherit.
//!
//! The existing 2pass is left alone as the legitimate long-`kv_len`
//! optimisation it is.
//!
//! # What it is unblocked by
//!
//! §6.2b's pre-allocated KV buffer (#142/#150), not paging. FlashDecoding needs
//! **contiguous chunks of a KV cache**, and that buffer supplies them: chunk
//! `c` is `[c*S, (c+1)*S)` at an offset into one allocation. **Paging is
//! required to stop *reserving* for them** — a footprint question costed at
//! 1.1 % of the pool at B=1 and deferred to B≥4 (§10.3i) — not to get them. The
//! chain is `#150 → #116`, and `#157 → #148`.

use crate::kernels::kernel::KernelName;
use crate::kernels::kernels::params::{FlashCombineParams, FlashPartialParams, FlashScatterParams};
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// Storage dtype of the query, key and value tensors.
///
/// `bfloat` is absent. It is still a scope statement rather than an omission,
/// and **both of the reasons originally given for it are false** — measured
/// 2026-08-30 (#307, `DESIGN.md` §3.9).
///
/// This comment used to read: *"`flash_decoding.metal` does not instantiate it,
/// because reaching it needs the ~500-line `_MLX_BFloat16` shim
/// `scaled_dot_product_attention.metal` carries. LFM2 ships BF16 on disk and
/// decode runs F16 (§9.1b)."* Measured, `__HAVE_BFLOAT__` **is defined** on this
/// machine, so that shim is the `#else` branch and is **inert** — the `.metal`
/// file's own estimate, *"a `#include` of the shim and three lines"*, is an
/// over-estimate by the `#include`. And bf16 decode is **reachable**: 12 of 12
/// decode kernel families dispatch a native bf16 sibling at an identical count,
/// and `lfm2-smoke` PASSes at `--dtype bf16` on all three `--attn` arms.
///
/// **What keeps `bfloat` out is now a cost argument rather than a capability
/// one**: §10.4b measures this arm **+6.3 % slower** than `Sdpa` at `kv_len`
/// 16 034, so a bf16 instantiation would be built-and-unused (§15.2 #11).
/// `lfm2.rs`'s `flash_decoding_applies` excludes `BF16` to match, which makes
/// `--attn flash --dtype bf16` a silent decline to the generic path — see the
/// note there before reading a flash run's config line.
///
/// The attention kernels are instantiated at `F16` and `F32` only
/// ([`FlashDType::head_dims`] is empty at `BF16`). The KV scatter moves
/// elements without arithmetic and carries `BF16` as well, so a page pool can
/// be written at bf16 and read by a kernel that has a bf16 arm.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FlashDType {
    F16,
    F32,
    BF16,
}

impl FlashDType {
    /// The name segment this dtype contributes to a `[[host_name]]`.
    ///
    /// `half`, not `float16_t`: the instantiation macro is passed `half`, and
    /// the token-pasted name is what the metallib carries. Checked against the
    /// compiled library by `flash_names_resolve` rather than against a second
    /// copy of the list — §8.1b's argument, and what caught 48 absent `reduce`
    /// variants and two absent `indexing` ones.
    fn suffix(self) -> &'static str {
        match self {
            FlashDType::F16 => "half",
            FlashDType::F32 => "float",
            FlashDType::BF16 => "bfloat",
        }
    }

    /// The head dims the attention kernels are instantiated at in this
    /// dtype: [`FLASH_HEAD_DIMS`] at f16, without 512 at f32, none at bf16.
    pub fn head_dims(self) -> &'static [usize] {
        match self {
            FlashDType::F16 => &FLASH_HEAD_DIMS,
            FlashDType::F32 => &FLASH_F32_HEAD_DIMS,
            FlashDType::BF16 => &[],
        }
    }
}

/// Head dimensions `flash_decoding.metal` instantiates, at f16 and f32.
///
/// **This list and the `.metal` instantiation list must agree, and the registry
/// is what checks it.** #103 found `scaled_dot_product_attention.metal`
/// instantiating eight head dimensions where `call_sdpa_vector`'s `match`
/// reaches six — the inverse of §8.1b's absent-variant class, whose symptom is
/// a capability that silently does not exist and dead instantiations paid for
/// at every cold compile. Declaring the list once here and resolving every name
/// against the compiled library is what stops that.
///
/// A lane holds `ceil(head_dim / 32)` elements, so 72 and 80 run with a masked
/// tail; every pipeline must admit the threadgroup width it is dispatched at
/// (`every_flash_pipeline_admits_its_dispatch_width`). This is the f16 list;
/// f32 is [`FLASH_F32_HEAD_DIMS`].
pub const FLASH_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

/// [`FLASH_HEAD_DIMS`] at f32: no 512. Sixteen `float`s of `q`, `k` and `o`
/// per lane leave the pipeline admitting fewer threads per threadgroup than
/// the 1024 the single-chunk and square partial passes dispatch
/// (`docs/porting-models.md` §21.8).
pub const FLASH_F32_HEAD_DIMS: [usize; 7] = [32, 64, 72, 80, 96, 128, 256];

/// The threadgroup width, in threads, each flash kernel is dispatched at:
/// the single-chunk pass and the square partial run 32 simdgroups, the
/// narrow partial and the combine 8. A pipeline whose
/// `maxTotalThreadsPerThreadgroup` is below its width cannot be dispatched.
pub const FLASH_SINGLE_THREADS: usize = 32 * 32;
pub const FLASH_COMBINE_THREADS: usize = 8 * 32;

/// The partial pass's threadgroup widths, in simdgroups: 32 (1024 threads,
/// 8 keys per simdgroup over a 256-key chunk) and 8 (256 threads, 32 keys
/// per simdgroup — `sdpa_vector_2pass_1`'s shape).
pub const FLASH_PARTIAL_WIDTHS: [usize; 2] = [32, 8];

/// Every `[[host_name]]` `flash_decoding.metal` declares.
///
/// A checked registry rather than a generated one, for §8.1b's reason: candle
/// has no build step to generate into, and because `.metal` is compiled at
/// runtime a test can resolve every declared name against **the actual metallib
/// the GPU will be asked for** — a strictly stronger oracle than generation,
/// which only proves two lists came from one source.
pub struct FlashKernel;

impl FlashKernel {
    /// The partial-pass kernel for one `(dtype, head_dim)`.
    /// The partial pass at a threadgroup `width` of 32 simdgroups (the
    /// kernel as shipped) or 8 (`flash_decoding_partial8`; the geometry of
    /// `sdpa_vector_2pass_1`, `FLASH_PARTIAL_WIDTHS`).
    pub fn partial(
        dtype: FlashDType,
        head_dim: usize,
        width: usize,
    ) -> Result<String, MetalKernelError> {
        let stem = match width {
            32 => "flash_decoding_partial",
            8 => "flash_decoding_partial8",
            _ => {
                return Err(MetalKernelError::LoadFunctionError(format!(
                    "flash_decoding partial width {width}: instantiated at {:?}",
                    FLASH_PARTIAL_WIDTHS
                )))
            }
        };
        Self::name(stem, dtype, head_dim)
    }

    /// The combine-pass kernel for one `(dtype, head_dim)`.
    pub fn combine(dtype: FlashDType, head_dim: usize) -> Result<String, MetalKernelError> {
        Self::name("flash_decoding_combine", dtype, head_dim)
    }

    /// The single-pass kernel for one `(dtype, head_dim)` — the `n_chunks == 1`
    /// path that writes `out` directly, skipping the partial+combine round-trip.
    pub fn single(dtype: FlashDType, head_dim: usize) -> Result<String, MetalKernelError> {
        Self::name("flash_decoding_single", dtype, head_dim)
    }

    /// The KV write for one dtype: one token's rows scattered to table-chosen
    /// slots. Per element, so it has no head-dimension instantiation.
    pub fn scatter(dtype: FlashDType) -> String {
        format!("flash_kv_scatter_{}", dtype.suffix())
    }

    fn name(stem: &str, dtype: FlashDType, head_dim: usize) -> Result<String, MetalKernelError> {
        if !dtype.head_dims().contains(&head_dim) {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "flash_decoding",
                got: head_dim,
                expected: dtype.head_dims().to_vec(),
            });
        }
        Ok(format!("{stem}_{}_{head_dim}", dtype.suffix()))
    }

    /// Every declared name, for `flash_names_resolve` and for exhaustive tests.
    pub fn all() -> Vec<String> {
        let mut names = Vec::new();
        for dtype in [FlashDType::F16, FlashDType::F32, FlashDType::BF16] {
            for &head_dim in dtype.head_dims() {
                names.push(Self::name("flash_decoding_partial", dtype, head_dim).unwrap());
                names.push(Self::name("flash_decoding_partial8", dtype, head_dim).unwrap());
                names.push(Self::name("flash_decoding_combine", dtype, head_dim).unwrap());
                names.push(Self::name("flash_decoding_single", dtype, head_dim).unwrap());
            }
            names.push(Self::scatter(dtype));
        }
        names
    }
}

/// Whether every FlashDecoding pipeline at `(head_dim, dtype)` is
/// instantiated and admits the threadgroup width it is dispatched at
/// ([`FLASH_SINGLE_THREADS`], both [`FLASH_PARTIAL_WIDTHS`],
/// [`FLASH_COMBINE_THREADS`]). A load-time check, never a per-dispatch one.
pub fn flash_admits_its_widths(
    device: &Device,
    kernels: &Kernels,
    head_dim: usize,
    dtype: FlashDType,
) -> Result<bool, MetalKernelError> {
    if !dtype.head_dims().contains(&head_dim) {
        return Ok(false);
    }
    let mut cells = vec![
        (FlashKernel::single(dtype, head_dim)?, FLASH_SINGLE_THREADS),
        (
            FlashKernel::combine(dtype, head_dim)?,
            FLASH_COMBINE_THREADS,
        ),
    ];
    for width in FLASH_PARTIAL_WIDTHS {
        cells.push((FlashKernel::partial(dtype, head_dim, width)?, width * 32));
    }
    for (name, threads) in cells {
        let pipeline =
            kernels.load_pipeline(device, Source::FlashDecoding, KernelName::Value(name))?;
        if pipeline.max_total_threads_per_threadgroup() < threads {
            return Ok(false);
        }
    }
    Ok(true)
}

/// How a chunk table is built for one step.
///
/// # Why this exists at B=1, where the table is the identity
///
/// §10.3h/§3.7f: the combine must **index** its chunk table and never **walk**
/// it. Both visit every chunk and only the first is bit-stable, because a
/// table's *contents* depend on allocation history, which under B>1 depends on
/// what other sequences did — so a sequence's logits would depend on its batch
/// neighbours, violating §2.3.3 #7.
///
/// **A B=1 gate structurally cannot detect the difference**, because at B=1 the
/// table is `chunk_table[c] == c` and the two orders coincide. #197's probe
/// demonstrated that a **permuted** arm is what separates them. So the table is
/// a real binding from the start and [`ChunkTable::Permuted`] exists as a test
/// fixture: a fixture that does not permute its table is testing the identity
/// function, which is §9.2c's alignment lesson in a second quantity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkTable {
    /// What a **contiguous** cache produces: chunk `c` begins at page
    /// `c * pages_per_chunk`.
    ///
    /// **It is `chunk_table[c] == c` only at `k = 1`**, which is the shipped
    /// value and is why the two are easy to conflate. An entry names a *page*
    /// and a chunk spans `k` of them, so at `k = 2` chunk 1 begins at page
    /// **2**, not page 1. Getting this wrong is not a crash: the kernel reads a
    /// valid but wrong range and the answer moves — which is what
    /// `flash_k_greater_than_one_agrees_with_k_one` caught on the first run,
    /// and the reason `k` is exercised by a test rather than only carried.
    Contiguous,
    /// An explicit table. Chunk `c` begins at page `entries[c]`.
    ///
    /// Used by the permuted-arm test, and the shape a paged cache would
    /// produce.
    Explicit(Vec<u32>),
}

impl ChunkTable {
    /// The `u32` entries for `n_chunks` chunks at `pages_per_chunk` pages each.
    pub fn entries(&self, n_chunks: usize, pages_per_chunk: usize) -> Vec<u32> {
        match self {
            ChunkTable::Contiguous => (0..n_chunks)
                .map(|c| (c * pages_per_chunk.max(1)) as u32)
                .collect(),
            ChunkTable::Explicit(v) => v.clone(),
        }
    }
}

/// Chunks a `kv_len` splits into at a given chunk size.
///
/// `ceil(kv_len / chunk_size)`. §10.6: contiguous is paged with one page and
/// single-pass attention is FlashDecoding with `n_chunks == 1`, so the
/// degenerate case falls out rather than needing a branch.
pub fn flash_chunk_count(kv_len: usize, chunk_size: usize) -> usize {
    kv_len.div_ceil(chunk_size.max(1))
}

impl FlashPartialParams {
    /// Build the partial pass's parameters for one step.
    ///
    /// **The arithmetic lives here rather than at the call site**, so a caller
    /// cannot compute the fields differently from the tests — which is the
    /// hand-sync §8.1b exists to remove, applied to a params block rather than
    /// to a kernel name.
    ///
    /// `k_stride`/`v_stride` are the tensors' full stride arrays. Index 1 is
    /// the **head** stride (the reserved capacity, which `sdpa_vector` also
    /// takes) and index 2 is the **token** stride — the field #200 could not
    /// vary and §9.1d asks for.
    ///
    /// `n_chunks` is what this step computes and `chunk_capacity` is what the
    /// region was sized for — the pair `ScratchSizing` decides (#234). They are
    /// equal under `Sizing::Grow` and differ under the other two, and passing
    /// the live count for both is what made every reserving policy
    /// inexpressible however many were compiled. **A capacity below the live
    /// count is refused here**: it would make the partial pass write past its
    /// own region, which §3.5 makes silent corruption rather than an error, and
    /// a caller computing the two independently is exactly the hand-sync this
    /// function exists to remove.
    ///
    /// `k_stride[0]`/`v_stride[0]` is the **sequence** stride, and the kernel
    /// reaches a batch row through it rather than by folding the batch into
    /// the head index. For a `[B, H_kv, T, D]` cache it is the batch stride; for
    /// a shared page pool it is **0**, and `table_stride` is how row `b` finds
    /// its own chunk table (`chunk_table[b * table_stride + c]`). `table_stride`
    /// is 0 when every row reads the same table — the contiguous cache, whose
    /// table is the identity for every row.
    #[allow(clippy::too_many_arguments)]
    pub fn for_step(
        q_shape: &[usize],
        k_shape: &[usize],
        k_stride: &[usize],
        v_stride: &[usize],
        page_size: usize,
        pages_per_chunk: usize,
        n_chunks: usize,
        chunk_capacity: usize,
        table_stride: usize,
        alpha: f32,
        softcapping: f32,
    ) -> Result<Self, MetalKernelError> {
        if q_shape[1] == 0 {
            return Err(MetalKernelError::LoadFunctionError(
                "flash-decoding: q has no heads, so the grid's (batch x head) index cannot be \
                 split into a sequence and a head"
                    .to_string(),
            ));
        }
        if chunk_capacity < n_chunks {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "flash-decoding scratch sized for {chunk_capacity} chunks but the step \
                 computes {n_chunks}: the partial pass would write past its region, which \
                 under HazardTrackingModeUntracked is silent corruption (DESIGN.md 3.5)"
            )));
        }
        // Folded the way `call_sdpa_vector` folds it, so the two paths scale
        // identically and a comparison between them is a comparison of the
        // split rather than of the scale.
        let scale = if softcapping != 1. {
            alpha / softcapping
        } else {
            alpha
        };
        Ok(Self {
            gqa_factor: (q_shape[1] / k_shape[1]) as i32,
            n_keys: k_shape[2] as i32,
            chunk_size: (page_size * pages_per_chunk) as i32,
            n_chunks: n_chunks as i32,
            chunk_capacity: chunk_capacity as i32,
            pages_per_chunk: pages_per_chunk as i32,
            page_size: page_size as i32,
            heads_per_seq: q_shape[1] as i32,
            table_stride: table_stride as i32,
            n_keys_stride: 0,
            k_head_stride: k_stride[1] as u64,
            v_head_stride: v_stride[1] as u64,
            k_token_stride: k_stride[2] as u64,
            v_token_stride: v_stride[2] as u64,
            k_seq_stride: k_stride[0] as u64,
            v_seq_stride: v_stride[0] as u64,
            scale,
            softcapping,
        })
    }
}

impl FlashScatterParams {
    /// Build the KV write's parameters for one pass.
    ///
    /// `src` is `[n_seqs, n_kv_heads, seq_len, head_dim]` with the given
    /// strides; `dst_*` are the destination's strides in the same order, where
    /// `dst_seq_stride` is 0 for a shared page pool. The slot table the kernel
    /// reads has `n_seqs * seq_len` rows, one per `(sequence, token)`.
    #[allow(clippy::too_many_arguments)]
    pub fn for_write(
        n_seqs: usize,
        n_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
        src_stride: &[usize],
        dst_seq_stride: usize,
        dst_head_stride: usize,
        dst_token_stride: usize,
    ) -> Self {
        Self {
            n_rows: (n_seqs * seq_len) as i32,
            seq_len: seq_len as i32,
            n_kv_heads: n_kv_heads as i32,
            head_dim: head_dim as i32,
            src_seq_stride: src_stride[0] as u64,
            src_head_stride: src_stride[1] as u64,
            src_token_stride: src_stride[2] as u64,
            dst_seq_stride: dst_seq_stride as u64,
            dst_head_stride: dst_head_stride as u64,
            dst_token_stride: dst_token_stride as u64,
        }
    }
}

impl FlashCombineParams {
    /// Build the combine pass's parameters for one step.
    ///
    /// `n_chunks` is the **live** count and `chunk_capacity` is what the region
    /// is **sized** for. They differ under `Sizing::Reserve`, and the merge runs
    /// to the live count: merging over the reservation folds in uninitialised
    /// memory, a silent wrong answer no size check catches (§9.1a, §10.3d).
    pub fn for_step(n_chunks: usize, chunk_capacity: usize) -> Self {
        Self {
            n_chunks: n_chunks as i32,
            chunk_capacity: chunk_capacity as i32,
        }
    }
}

/// Dispatch one attention layer's FlashDecoding: partials, then combine.
///
/// # The caller's fence obligation
///
/// **Two barriers per attention layer per step** (§9.4): one after the KV
/// append and before the partials read it, and one after the partials and
/// before the combine reads them. The N partials write **disjoint** regions and
/// need no fences *between* them — but disjointness is **our** assertion and not
/// the driver's (§3.5), so it is asserted by `plan_scratch`'s `check_disjoint`
/// rather than assumed here.
///
/// Both barriers are candle's `auto_barrier`, emitted because `partials`/`sums`
/// /`maxs` are bound as outputs of the first dispatch and inputs of the second.
/// That is the same mechanism #71 verified at `auto_barrier`'s emission site,
/// with a negative control reading 0 when the declaration is dropped.
///
/// # `pages_per_chunk` is `k`
///
/// `chunk_size = k * page_size` (§9.1d). Shipped at `k = 1`, which is what
/// §10.4 specifies — the difference is that it is now a *stated* value on a
/// selectable axis rather than an equality welded into a kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_flash_decoding(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_buffer: &Buffer,
    k_offset: usize,
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    output: &Buffer,
    partials: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    chunk_table: &Buffer,
    // Both params blocks are **caller-owned buffers**, not `setBytes` scalars.
    //
    // That is `call_scratch_partials`'s shape and it is deliberate on two
    // counts. `MTLIndirectComputeCommand` has no `setBytes` in any form
    // (§3.7c), so a kernel taking its scalars inline would be unencodable on a
    // replayed dispatch position however stable its operands were — the second
    // axis of coverage §11.3l found empty. And the buffer is the *caller's* so
    // it can be pooled: §11.3l finding 3 records that a params buffer allocated
    // per call makes ICB coverage **zero**, because a command binds a buffer by
    // identity at encode time.
    //
    // Filled by [`FlashPartialParams::for_step`] and
    // [`FlashCombineParams::for_step`], which is where the arithmetic lives, so
    // a caller cannot compute the fields differently from the tests.
    partial_params: &Buffer,
    combine_params: &Buffer,
    // Where the combine records the chunk indices it walked, in walk order.
    //
    // **The merge-order assertion compares against this and not against the
    // output.** §10.4: reversing the combine loop *"is caught by that
    // assertion and by nothing else"*, because floating-point
    // non-associativity happens not to bite on every fixture — measured here,
    // a reversed loop left all seven of this family's tests green. An ordering
    // that cannot be observed cannot be asserted (§9.1a).
    //
    // Sized `n_q_heads * n_chunks` `u32`. It is written on every step rather
    // than behind a flag: it is one store per chunk per head by one lane, and
    // a debug-only buffer is one the shipping path never proves it can fill.
    walk_order: &Buffer,
    // The live key count per sequence, `i32` entries read at
    // `params.n_keys_stride` per row: one entry at stride 0 for a lockstep
    // batch, one per row at stride 1 for a ragged one (§12.2). Caller-owned
    // for the reason the params blocks are.
    n_keys_per_seq: &Buffer,
    n_q_heads: usize,
    head_dim: usize,
    n_chunks: usize,
    itype: FlashDType,
    // The partial pass's threadgroup width in simdgroups
    // (`FLASH_PARTIAL_WIDTHS`); the single-chunk and combine passes have
    // their own fixed geometry.
    partial_width: usize,
) -> Result<(), MetalKernelError> {
    // The single-chunk fast path: at `n_chunks == 1` the split-K machinery
    // manufactures no parallelism (one chunk is one threadgroup per head), and
    // the two-pass path would still write the answer to `partials` scratch, then
    // pay a SECOND dispatch (`flash_decoding_combine`) plus a device-memory
    // round-trip to read it back and write `out`. `flash_decoding_single` does
    // the whole thing in one dispatch, writing `out` directly — bit-identical to
    // the two-pass path at one chunk (§2.3.5a; the combine's rescale is
    // `exp(0) = 1` and its denom is `sums[0]`, so both reduce to `acc / sums[0]`).
    //
    // `scratch`/`combine_params`/`walk_order` are still allocated by the caller
    // but go unread here — sizing them is cheap and the caller does not know the
    // chunk count until it has the params. `chunk_table` IS read: the single
    // chunk starts at the page its table names, which is page 0 only on a
    // fresh pool (§10.3h).
    if n_chunks == 1 {
        let name = FlashKernel::single(itype, head_dim)?;
        let pipeline =
            kernels.load_pipeline(device, Source::FlashDecoding, KernelName::Value(name))?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(encoder, "flash_decoding single hd={head_dim}");

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(output),
                partial_params,
                chunk_table,
                n_keys_per_seq
            )
        );

        // One threadgroup per head — the single chunk IS the cache, so there is
        // no `depth` axis to split over (contrast the partial pass's
        // `depth = n_chunks`).
        let grid_dims = MTLSize {
            width: 1,
            height: n_q_heads,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: FLASH_SINGLE_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
        return Ok(());
    }

    // Pass 1 -- partials.
    {
        let name = FlashKernel::partial(itype, head_dim, partial_width)?;
        let pipeline =
            kernels.load_pipeline(device, Source::FlashDecoding, KernelName::Value(name))?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(
            encoder,
            "flash_decoding partial hd={head_dim} chunks={n_chunks}"
        );

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(partials),
                Output::new(sums),
                Output::new(maxs),
                chunk_table,
                partial_params,
                n_keys_per_seq
            )
        );

        // One threadgroup per (head, chunk). **This is where the parallelism
        // comes from**: §10.4's structural argument is that at B=1 attention
        // with one threadgroup per head is 32 threadgroups on a GPU wanting
        // hundreds, and splitting KV *manufactures* the rest. At `kv_len` 2720
        // and chunk 256 this is 32 x 11 = 352.
        let grid_dims = MTLSize {
            width: 1,
            height: n_q_heads,
            depth: n_chunks,
        };
        // `partial_width` simdgroups of 32: the kernel's `BN`.
        let group_dims = MTLSize {
            width: partial_width * 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Pass 2 -- the index-ordered combine.
    {
        let name = FlashKernel::combine(itype, head_dim)?;
        let pipeline =
            kernels.load_pipeline(device, Source::FlashDecoding, KernelName::Value(name))?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(
            encoder,
            "flash_decoding combine hd={head_dim} chunks={n_chunks}"
        );

        set_params!(
            encoder,
            (
                partials,
                sums,
                maxs,
                Output::new(output),
                combine_params,
                Output::new(walk_order)
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: n_q_heads,
            depth: 1,
        };
        // Eight simdgroups per head: the kernel's `BN` (chunks striped across
        // them, a fixed-order sum at the end).
        let group_dims = MTLSize {
            width: FLASH_COMBINE_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(())
}

/// Dispatch the KV write: `src`'s rows to `dst` at the token slots `slots`
/// names, one thread per element.
///
/// `slots` holds `n_rows` `u32` entries from byte `slots_offset` (a view of a
/// larger slot table binds at its offset, as the source and destination do)
/// and `scatter_params` a [`FlashScatterParams`] built by
/// [`FlashScatterParams::for_write`]. Both are caller-owned buffers for the
/// same reason the attention params are (§3.7c).
#[allow(clippy::too_many_arguments)]
pub fn call_flash_kv_scatter(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src_offset: usize,
    src: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
    slots: &Buffer,
    slots_offset: usize,
    scatter_params: &Buffer,
    n_rows: usize,
    n_kv_heads: usize,
    head_dim: usize,
    itype: FlashDType,
) -> Result<(), MetalKernelError> {
    if n_rows == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    let name = FlashKernel::scatter(itype);
    let pipeline = kernels.load_pipeline(device, Source::FlashDecoding, KernelName::Value(name))?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "flash_kv_scatter rows={n_rows}");
    set_params!(
        encoder,
        (
            (src, src_offset),
            Output::with_offset(dst, dst_offset),
            (slots, slots_offset),
            scatter_params
        )
    );
    let grid_dims = MTLSize {
        width: head_dim,
        height: n_kv_heads,
        depth: n_rows,
    };
    // A threadgroup covers one head's vector: `head_dim` threads along `x`
    // (at most the pipeline's width), one row of one head per group.
    let width = head_dim.min(pipeline.max_total_threads_per_threadgroup());
    let group_dims = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}
