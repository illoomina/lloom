// NOTE: this file does NOT open with `#include <metal_stdlib>`.
// `source.rs` supplies the include prologue and splices `utils.metal`'s
// shared helpers in after it (issue #458), so this file is a FRAGMENT and
// does not compile standalone. `newLibraryWithSource` sees prologue +
// utils + this text as one translation unit.

template <typename T>
inline T max_value();

template <>
inline int64_t max_value<int64_t>() {
    return 0x7FFFFFFFFFFFFFFF;
}

template <>
inline uint32_t max_value<uint32_t>() {
    return 0xFFFFFFFFu;
}

template <>
inline uint8_t max_value<uint8_t>() {
    return 0xFF;
}

// Packed parameter structs, mirrored in `kernels/params.rs` and checked against
// them by `indexing_params_layout`.
//
// `size_t` is 8 bytes in MSL, so these are 8-aligned -- like `conv.metal`'s and
// the four elementwise families', unlike `reduce.metal`'s 4-aligned `uint`
// structs.
//
// **`IndexParams` is where the `bool` hazard finally fires.** `DESIGN.md`
// §11.3b names two layout hazards -- over-aligning vector types and `bool` at 1
// byte -- and #40 recorded that the second "does not fire" in any of the four
// elementwise families, checked by enumerating their parameters. It fires here:
// `index` takes a `constant bool &contiguous` between four `size_t` and a
// fifth, so `contiguous` sits at offset 32, `src_num_dims` pads to 40, and the
// struct is 48 bytes where its fields sum to 41. `conv.metal`'s
// `UpsampleBilinear2dParams` is the only other struct in the crate with a
// `bool` in it, and that one is off the decode path entirely.
//
// **`left_size` is gone from all four structs as of #219**, having been bound by
// all five kernels and read by none since before the packed conversion -- §11.3k
// finding 3 found it with a mutation that survived. Each struct is one `size_t`
// narrower and the interior padding is unchanged in kind: the `bool` still pads,
// it simply pads at 32 rather than at 40. The figures above are post-removal;
// before it they were 40 / 48 / 56 / 49.
//
// `dims` and `strides` are deliberately not fields: their length comes from the
// tensor's layout, not from the struct. They stay separate bindings, which an
// ICB can express -- `setKernelBuffer` binds a buffer of any length, and the
// constraint is `setBytes` rather than buffer count (`DESIGN.md` §11.3d).
struct IndexParams {
    size_t dst_size;
    size_t src_dim_size;
    size_t right_size;
    size_t ids_size;
    bool contiguous;
    size_t src_num_dims;
};

struct GatherParams {
    size_t dst_size;
    size_t src_dim_size;
    size_t right_size;
    size_t ids_size;
};

// One struct for `scatter` and `scatter_add`: they bind the same five scalars
// and differ only in whether the destination is assigned or accumulated into,
// which is the body rather than the binding.
struct ScatterParams {
    size_t dst_size;
    size_t src_dim_size;
    size_t right_size;
    size_t dst_dim_size;
};

struct IndexAddParams {
    size_t dst_size;
    size_t src_dim_size;
    size_t right_size;
    size_t dst_dim_size;
    size_t ids_dim_size;
};

// `get_strided_index` is shared: it lives in `utils.metal`, concatenated ahead
// of this file by `source.rs` (issue #458). It is templated on the pointer type
// rather than written once per address space, and this file is why -- the
// classical `index` entry point receives `src_dims`/`src_strides` as
// `constant size_t *`, the packed one as `device const size_t *`, because a
// `setBytes`-bound array becomes a real buffer under packing. Letting MSL
// deduce the address space keeps the index arithmetic one body -- the same
// remedy #38 and #41 used for `reduce` and `gemv`.

// The five bodies.
//
// Each is a `METAL_FUNC` taking its scalars as one `thread const` struct, with
// two `[[kernel]]` wrappers above it that differ only in how they obtain that
// struct: the classical one builds it from `constant &` arguments, the packed
// one loads it from a `device const *`. `DESIGN.md` §11.3b's pattern, and
// §11.3f's factoring rather than §11.3d's -- no kernel here declares
// threadgroup memory, so the whole body can live below the `[[kernel]]`
// boundary.
//
// The struct is taken by `thread const &` and its fields copied into locals at
// entry, per `DESIGN.md` §11.3e: one structure load into registers rather than
// a dereference per use. `constant` is a cached read-only space and `device` is
// not, so the difference is real for a value read inside a loop -- which
// `scatter`, `scatter_add` and `index_add` all do.

template<typename TYPENAME, typename INDEX_TYPENAME, typename PtrT>
METAL_FUNC void index_body(
    thread const IndexParams &p,
    PtrT src_dims,
    PtrT src_strides,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid
) {
    const size_t dst_size = p.dst_size;
    const size_t src_dim_size = p.src_dim_size;
    const size_t right_size = p.right_size;
    const size_t ids_size = p.ids_size;
    const bool contiguous = p.contiguous;
    const size_t src_num_dims = p.src_num_dims;
    if (tid >= dst_size) {
        return;
    }
    const size_t id_i = (tid / right_size) % ids_size;
    if (input_ids[id_i] == max_value<INDEX_TYPENAME>()) {
      output[tid] = static_cast<TYPENAME>(0);
    } else {
      const INDEX_TYPENAME input_i = min(input_ids[id_i], (INDEX_TYPENAME)(src_dim_size - 1));
      const size_t right_rank_i = tid % right_size;
      const size_t left_rank_i = tid / right_size / ids_size;
      /*
      // Force prevent out of bounds indexing
      // since there doesn't seem to be a good way to force crash
      // No need to check for zero we're only allowing unsized.
      */
      const size_t src_i = left_rank_i * src_dim_size * right_size + input_i * right_size + right_rank_i;
      // `src_num_dims` is the source tensor's **rank**, which is what
      // `get_strided_index` walks: it decomposes a logical index into per-axis
      // coordinates by dividing through `dims` from the last axis inward, so it
      // needs one iteration per axis.
      //
      // This argument used to be `src_dim_size` -- the *extent of the indexed
      // dimension*. The two are unrelated quantities that share a type, so it
      // compiled and silently read the wrong elements whenever they differed.
      // The CUDA kernel for this same op passes the rank
      // (`candle-kernels/src/indexing.cu:65`: `get_strided_index(src_i,
      // num_dims, dims, strides)`), so the old form was a Metal-only divergence
      // from the reference rather than a shared convention -- and every other
      // `get_strided_index` call site in this crate's `.metal` sources passes
      // `num_dims` too.
      //
      // Latent exactly when `rank != dims[dim]`; at rank 2 with a 2-long
      // indexed dimension the two coincide, which is why no existing test
      // caught it. LFM2's embedding lookup is contiguous and takes the fast
      // arm, so it never reached this line.
      const size_t strided_src_i = contiguous ? src_i : get_strided_index<PtrT>(src_i, src_num_dims, src_dims, src_strides);
      output[tid] = input[strided_src_i];
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index(
    constant size_t &dst_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    constant bool &contiguous,
    constant size_t &src_num_dims,
    constant size_t *src_dims,
    constant size_t *src_strides,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    IndexParams p { dst_size, src_dim_size, right_size, ids_size,
                    contiguous, src_num_dims };
    index_body<TYPENAME, INDEX_TYPENAME, constant size_t *>(
        p, src_dims, src_strides, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index_packed(
    device const IndexParams *pp,
    device const size_t *src_dims,
    device const size_t *src_strides,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    IndexParams p = *pp;
    index_body<TYPENAME, INDEX_TYPENAME, device const size_t *>(
        p, src_dims, src_strides, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void gather_body(
    thread const GatherParams &p,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid
) {
    const size_t dst_size = p.dst_size;
    const size_t src_dim_size = p.src_dim_size;
    const size_t right_size = p.right_size;
    const size_t ids_size = p.ids_size;
    if (tid >= dst_size) {
        return;
    }
    const INDEX_TYPENAME input_i = input_ids[tid];
    if (input_i == max_value<INDEX_TYPENAME>()) {
      output[tid] = static_cast<TYPENAME>(0);
    } else {
      const size_t right_rank_i = tid % right_size;
      const size_t left_rank_i = tid / right_size / ids_size;
      const size_t src_i = (left_rank_i * src_dim_size + input_i) * right_size + right_rank_i;
      output[tid] = input[src_i];
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void gather(
    constant size_t &dst_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &ids_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    GatherParams p { dst_size, src_dim_size, right_size, ids_size };
    gather_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void gather_packed(
    device const GatherParams *pp,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    GatherParams p = *pp;
    gather_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void scatter_body(
    thread const ScatterParams &p,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid
) {
    const size_t dst_size = p.dst_size;
    const size_t src_dim_size = p.src_dim_size;
    const size_t right_size = p.right_size;
    const size_t dst_dim_size = p.dst_dim_size;
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < src_dim_size; ++j) {
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const INDEX_TYPENAME idx = input_ids[src_i];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] = input[src_i];
        }
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter(
    constant size_t &dst_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    ScatterParams p { dst_size, src_dim_size, right_size, dst_dim_size };
    scatter_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter_packed(
    device const ScatterParams *pp,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    ScatterParams p = *pp;
    scatter_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void scatter_add_body(
    thread const ScatterParams &p,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid
) {
    const size_t dst_size = p.dst_size;
    const size_t src_dim_size = p.src_dim_size;
    const size_t right_size = p.right_size;
    const size_t dst_dim_size = p.dst_dim_size;
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < src_dim_size; ++j) {
        const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
        const INDEX_TYPENAME idx = input_ids[src_i];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] += input[src_i];
        }
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter_add(
    constant size_t &dst_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    ScatterParams p { dst_size, src_dim_size, right_size, dst_dim_size };
    scatter_add_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void scatter_add_packed(
    device const ScatterParams *pp,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    ScatterParams p = *pp;
    scatter_add_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
METAL_FUNC void index_add_body(
    thread const IndexAddParams &p,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid
) {
    const size_t dst_size = p.dst_size;
    const size_t src_dim_size = p.src_dim_size;
    const size_t right_size = p.right_size;
    const size_t dst_dim_size = p.dst_dim_size;
    const size_t ids_dim_size = p.ids_dim_size;
    if (tid >= dst_size) {
        return;
    }
    const size_t right_rank_i = tid % right_size;
    const size_t left_rank_i = tid / right_size;
    for (unsigned int j = 0; j < ids_dim_size; ++j) {
        const INDEX_TYPENAME idx = input_ids[j];
        if (idx < max_value<INDEX_TYPENAME>()) {
          const size_t src_i = (left_rank_i * src_dim_size + j) * right_size + right_rank_i;
          const size_t dst_i = (left_rank_i * dst_dim_size + idx) * right_size + right_rank_i;
          output[dst_i] += input[src_i];
        }
    }
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index_add(
    constant size_t &dst_size,
    constant size_t &src_dim_size,
    constant size_t &right_size,
    constant size_t &dst_dim_size,
    constant size_t &ids_dim_size,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    IndexAddParams p { dst_size, src_dim_size, right_size,
                       dst_dim_size, ids_dim_size };
    index_add_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

template<typename TYPENAME, typename INDEX_TYPENAME>
[[kernel]] void index_add_packed(
    device const IndexAddParams *pp,
    const device TYPENAME *input,
    const device INDEX_TYPENAME *input_ids,
    device TYPENAME *output,
    uint tid [[ thread_position_in_grid ]]
) {
    IndexAddParams p = *pp;
    index_add_body<TYPENAME, INDEX_TYPENAME>(p, input, input_ids, output, tid);
}

// Layout, asserted rather than hoped.
//
// A field at the wrong offset does not crash: the kernel reads a well-formed
// number from the wrong place and computes a plausible wrong answer, which
// under `HazardTrackingModeUntracked` is the failure mode `DESIGN.md` §3.5 and
// §15.1 both single out.
//
// Only sizes and alignments are `static_assert`ed. Offsets cannot be: MSL has
// no `<cstddef>` and the null-pointer-member form of `offsetof` is not a
// constant expression. They are reported by `indexing_params_layout` below and
// compared against Rust's `offset_of!`, which is the stronger check regardless
// -- a `static_assert` on either side proves only that side agrees with itself.
//
// `IndexParams` is 48 rather than the 41 its fields sum to: the `bool` at 32
// leaves seven bytes before `src_num_dims` can start at its own 8-byte
// alignment. That is the number this file exists to ship across the boundary.
//
// All four shrank by one `size_t` when #219 removed `left_size` -- 56/40/40/48
// became 48/32/32/40. The padding argument is untouched by that: what pads is
// the `bool`, and it pads wherever it lands.
static_assert(sizeof(IndexParams) == 48, "IndexParams layout");
static_assert(alignof(IndexParams) == 8, "IndexParams alignment");

static_assert(sizeof(GatherParams) == 32, "GatherParams layout");
static_assert(alignof(GatherParams) == 8, "GatherParams alignment");

static_assert(sizeof(ScatterParams) == 32, "ScatterParams layout");
static_assert(alignof(ScatterParams) == 8, "ScatterParams alignment");

static_assert(sizeof(IndexAddParams) == 40, "IndexAddParams layout");
static_assert(alignof(IndexAddParams) == 8, "IndexAddParams alignment");

// The offset is taken from a real `thread` instance rather than the usual
// null-pointer form, which MSL rejects in constant evaluation. Measuring it at
// runtime is what this kernel is for.
#define offsetof_rt(S, F) \
    ((uint)((thread const char *)&(probe_##S.F) - (thread const char *)&probe_##S))

[[kernel]] void indexing_params_layout(
    device uint *out,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid != 0) { return; }
    IndexParams    probe_IndexParams;
    GatherParams   probe_GatherParams;
    ScatterParams  probe_ScatterParams;
    IndexAddParams probe_IndexAddParams;

    out[0]  = sizeof(IndexParams);
    out[1]  = offsetof_rt(IndexParams, dst_size);
    out[2]  = offsetof_rt(IndexParams, src_dim_size);
    out[3]  = offsetof_rt(IndexParams, right_size);
    out[4]  = offsetof_rt(IndexParams, ids_size);
    out[5]  = offsetof_rt(IndexParams, contiguous);
    out[6]  = offsetof_rt(IndexParams, src_num_dims);

    out[7]  = sizeof(GatherParams);
    out[8]  = offsetof_rt(GatherParams, dst_size);
    out[9]  = offsetof_rt(GatherParams, src_dim_size);
    out[10] = offsetof_rt(GatherParams, right_size);
    out[11] = offsetof_rt(GatherParams, ids_size);

    out[12] = sizeof(ScatterParams);
    out[13] = offsetof_rt(ScatterParams, dst_size);
    out[14] = offsetof_rt(ScatterParams, src_dim_size);
    out[15] = offsetof_rt(ScatterParams, right_size);
    out[16] = offsetof_rt(ScatterParams, dst_dim_size);

    out[17] = sizeof(IndexAddParams);
    out[18] = offsetof_rt(IndexAddParams, dst_size);
    out[19] = offsetof_rt(IndexAddParams, src_dim_size);
    out[20] = offsetof_rt(IndexAddParams, right_size);
    out[21] = offsetof_rt(IndexAddParams, dst_dim_size);
    out[22] = offsetof_rt(IndexAddParams, ids_dim_size);
}

// Explicit instantiation. `decltype(func<...>)` restates the template's own
// signature, so a variant is declared by naming the type arguments and the
// `[[host_name]]` string only -- the parameter list is written once, in the
// template above.
//
// The macro form this replaces spelled every signature twice: a `kernel void`
// wrapper per instantiation whose only job was to restate the template's
// parameters and forward to it. `DESIGN.md` §8.1c records why that is worse
// than verbose -- a parameter added to one list and not the other is a compile
// error only when the argument *count* changes, and **reordering two
// same-typed parameters in one list is silent**. `index` takes five
// consecutive `constant size_t &` parameters and `gather` five more, so that
// shape was reachable in this file.
//
// Same spelling as unary.metal, binary.metal, affine.metal, conv.metal and
// reduce.metal, which candle already migrated.
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// The five op families, each keyed on a `(index dtype, value dtype)` **pair**.
//
// That pair is what makes this file different from every other family candle
// has templated: `conv` and `reduce` vary over one dtype, so their names are
// `<stem>_<dtype>`. Here both the index type and the value type vary
// independently, and the name carries both -- `is_u32_f16` is `index` over an
// f16 tensor with u32 indices. The registry on the Rust side
// (`IndexingKernel`) is keyed the same way for the same reason.
//
// Note the argument order: the *name* reads `<stem>_<index>_<value>` while the
// template takes `<value, index>`. That inversion is inherited from the macro
// form and is preserved deliberately -- changing it would rename all 72
// kernels for no gain, and the registry checks the spelling rather than
// assuming it.
//
// Each row declares **both** binding styles, so a variant cannot exist in one
// and not the other -- the `_packed` name is a second name segment appended
// after the two dtypes, matching `ParamStyle::kernel_name` on the Rust side.
//
// A second `[[host_name]]` on the *same* instantiation does not compile: MSL
// rejects it as a duplicate explicit instantiation regardless of how many names
// are attached (#41, `DESIGN.md` §11.3g). So the packed sibling is a genuinely
// distinct function template, which is what `*_packed` above are.
#define init_index(iname, tname, itype, ttype) \
    init_kernel("is_" #iname "_" #tname, index, ttype, itype) \
    init_kernel("is_" #iname "_" #tname "_packed", index_packed, ttype, itype)

#define init_gather(iname, tname, itype, ttype) \
    init_kernel("gather_" #iname "_" #tname, gather, ttype, itype) \
    init_kernel("gather_" #iname "_" #tname "_packed", gather_packed, ttype, itype)

#define init_scatter(iname, tname, itype, ttype) \
    init_kernel("s_" #iname "_" #tname, scatter, ttype, itype) \
    init_kernel("s_" #iname "_" #tname "_packed", scatter_packed, ttype, itype)

#define init_scatter_add(iname, tname, itype, ttype) \
    init_kernel("sa_" #iname "_" #tname, scatter_add, ttype, itype) \
    init_kernel("sa_" #iname "_" #tname "_packed", scatter_add_packed, ttype, itype)

#define init_index_add(iname, tname, itype, ttype) \
    init_kernel("ia_" #iname "_" #tname, index_add, ttype, itype) \
    init_kernel("ia_" #iname "_" #tname "_packed", index_add_packed, ttype, itype)

// ---------------------------------------------------------------------------
// index_select -- 18 variants, three index types over six value types.
//
// `is_i64_u8` and `is_i64_u32` were **absent before this conversion** while
// `candle-core` named both, so `index_select` on a U8 or U32 tensor with I64
// indices was a runtime `LoadFunctionError` on `lloom/integration`. That is the
// fourth firing of the absent-variant class `DESIGN.md` §8.1b tracks, after
// #26's 48 reduce variants and conv's. They are declared here, and
// `indexing_names_resolve` is what keeps the two sides from drifting again.
//
// `is_u32_f16` is the embedding lookup -- the one kernel in this file on the
// LFM2 decode path (§11.3h), one dispatch per token.
// ---------------------------------------------------------------------------
init_index(i64, i64, int64_t, int64_t)
init_index(i64, f32, int64_t, float)
init_index(i64, f16, int64_t, half)
init_index(i64, u8, int64_t, uint8_t)
init_index(i64, u32, int64_t, uint32_t)
#if defined(__HAVE_BFLOAT__)
init_index(i64, bf16, int64_t, bfloat)
#endif

init_index(u32, u8, uint32_t, uint8_t)
init_index(u32, u32, uint32_t, uint32_t)
init_index(u32, i64, uint32_t, int64_t)
init_index(u32, f32, uint32_t, float)
init_index(u32, f16, uint32_t, half)
#if defined(__HAVE_BFLOAT__)
init_index(u32, bf16, uint32_t, bfloat)
#endif

init_index(u8, u8, uint8_t, uint8_t)
init_index(u8, u32, uint8_t, uint32_t)
init_index(u8, i64, uint8_t, int64_t)
init_index(u8, f32, uint8_t, float)
init_index(u8, f16, uint8_t, half)
#if defined(__HAVE_BFLOAT__)
init_index(u8, bf16, uint8_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// gather -- 16 variants.
//
// The pre-conversion list spelled `uint` for the u32 index type here and
// `uint32_t` in the index_select block. They are the same type in MSL, so this
// is a spelling difference rather than a behaviour one; `uint32_t` is used
// throughout below so the instantiation rows read uniformly.
// ---------------------------------------------------------------------------
init_gather(u8, f32, uint8_t, float)
init_gather(u8, f16, uint8_t, half)
init_gather(i64, f32, int64_t, float)
init_gather(i64, f16, int64_t, half)
init_gather(u32, f32, uint32_t, float)
init_gather(u32, f16, uint32_t, half)
#if defined(__HAVE_BFLOAT__)
init_gather(u8, bf16, uint8_t, bfloat)
init_gather(i64, bf16, int64_t, bfloat)
init_gather(u32, bf16, uint32_t, bfloat)
#endif
init_gather(u8, u8, uint8_t, uint8_t)
init_gather(u8, i64, uint8_t, int64_t)
init_gather(u8, u32, uint8_t, uint32_t)
init_gather(u32, u32, uint32_t, uint32_t)
init_gather(u32, i64, uint32_t, int64_t)
init_gather(i64, u32, int64_t, uint32_t)
init_gather(i64, i64, int64_t, int64_t)

// ---------------------------------------------------------------------------
// scatter_add -- 10 variants.
// ---------------------------------------------------------------------------
init_scatter_add(u32, f32, uint32_t, float)
init_scatter_add(u8, f32, uint8_t, float)
init_scatter_add(i64, f32, int64_t, float)
init_scatter_add(u32, u32, uint32_t, uint32_t)
init_scatter_add(u32, f16, uint32_t, half)
init_scatter_add(u8, f16, uint8_t, half)
init_scatter_add(i64, f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
init_scatter_add(u32, bf16, uint32_t, bfloat)
init_scatter_add(u8, bf16, uint8_t, bfloat)
init_scatter_add(i64, bf16, int64_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// scatter -- 10 variants. Same key set as scatter_add; the two differ in
// whether the destination is assigned or accumulated into, which is the kernel
// body rather than the name.
// ---------------------------------------------------------------------------
init_scatter(u32, f32, uint32_t, float)
init_scatter(u8, f32, uint8_t, float)
init_scatter(i64, f32, int64_t, float)
init_scatter(u32, u32, uint32_t, uint32_t)
init_scatter(u32, f16, uint32_t, half)
init_scatter(u8, f16, uint8_t, half)
init_scatter(i64, f16, int64_t, half)
#if defined(__HAVE_BFLOAT__)
init_scatter(u32, bf16, uint32_t, bfloat)
init_scatter(u8, bf16, uint8_t, bfloat)
init_scatter(i64, bf16, int64_t, bfloat)
#endif

// ---------------------------------------------------------------------------
// index_add -- 18 variants, the same three-by-six grid as index_select.
// ---------------------------------------------------------------------------
init_index_add(i64, f16, int64_t, half)
init_index_add(i64, f32, int64_t, float)
init_index_add(i64, i64, int64_t, int64_t)
init_index_add(i64, u32, int64_t, uint32_t)
init_index_add(i64, u8, int64_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(i64, bf16, int64_t, bfloat)
#endif

init_index_add(u32, f16, uint32_t, half)
init_index_add(u32, f32, uint32_t, float)
init_index_add(u32, i64, uint32_t, int64_t)
init_index_add(u32, u32, uint32_t, uint32_t)
init_index_add(u32, u8, uint32_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(u32, bf16, uint32_t, bfloat)
#endif

init_index_add(u8, f16, uint8_t, half)
init_index_add(u8, f32, uint8_t, float)
init_index_add(u8, i64, uint8_t, int64_t)
init_index_add(u8, u32, uint8_t, uint32_t)
init_index_add(u8, u8, uint8_t, uint8_t)
#if defined(__HAVE_BFLOAT__)
init_index_add(u8, bf16, uint8_t, bfloat)
#endif
