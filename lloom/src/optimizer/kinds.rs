//! What each kernel family a walk emits IS, for the remarks
//! (`design/docs/optimizer-design.md` §10i): the shape of its parallelism
//! (`design/docs/fusion.md` §12.1's rule table — elementwise, reduction,
//! product) and whether it takes a composed block at its store or its load.
//! A seam's verdict is read off these two facts and the catalogue, so every
//! family a walk emits must be declared here; an undeclared one is reported
//! as such, never guessed.
//!
//! The IR's own ops carry their facts; an `Opaque` node is looked up by its
//! family, each entry citing the kernel it describes.

use super::ir::{Core, Node, Op};

/// The parallel shape of a kernel, in §12.1's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// A matrix-vector product: partitions its OUTPUT across threadgroups
    /// and walks `K` per output element (§12.2d). Its store is written once
    /// per output element; its load is read by every threadgroup.
    Product,
    /// One output element from the same element of each input.
    Elementwise,
    /// Reduces over an extent: an output element reads many inputs, held by
    /// many threads (a norm, a sum, a gated mean over streams, a softmax or
    /// a top-k, an attention over the cache).
    Reduction,
    /// Moves or indexes data: a copy, a gather, an index map.
    Movement,
    /// A state update carried across steps.
    Recurrence,
    /// Reads two elements of one row to write both (a rotation).
    Pairwise,
    /// Writes more than it reads per input element: a scatter of one value
    /// across streams.
    Broadcast,
}

/// The facts the remarks read about one dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    pub role: Role,
    /// The kernel takes a composed block on its store (an epilogue functor,
    /// a `Store` functor).
    pub store_hook: bool,
    /// The kernel takes a composed block on its load (a prologue functor).
    pub load_hook: bool,
}

const fn facts(role: Role, store_hook: bool, load_hook: bool) -> Facts {
    Facts {
        role,
        store_hook,
        load_hook,
    }
}

/// A product on `core`: `gemv.metal`'s `GEMVKernel` takes a `Prologue` and an
/// `Epilogue` (`PrologueRmsNorm`, `PrologueNormInline`, `EpilogueSilu`, …);
/// the mixed arm the `MixedEpilogue` blocks at its store; the quantized
/// matvec (`quantized.metal`'s `kernel_mul_mv_*`) neither.
fn product(core: Core) -> Facts {
    match core {
        Core::Gemv => facts(Role::Product, true, true),
        Core::Mixed => facts(Role::Product, true, false),
        Core::Quant => facts(Role::Product, false, false),
    }
}

/// The opaque families, by the name the walk emits. Each row cites the
/// kernel its facts are read from.
fn opaque(family: &str) -> Option<Facts> {
    use Role::*;
    Some(match family {
        // `quantized.metal`: the routed matvecs over the selected experts.
        // `steel_gemm`: a matrix product (the indexer's block scores); its
        // store and load take no block.
        "mul_mv_id" | "mul_mv_id2" | "gemm" => facts(Product, false, false),
        // The composed attention's batched products (`q · kᵀ` on the GEMV,
        // `softmax · v` on the transposed GEMV), opaque: no block at either
        // end.
        "gemv" | "gemv_t" => facts(Product, false, false),
        // `reduce.metal`: the softmax over the last extent.
        "softmax" => facts(Reduction, false, false),
        // `mlx_sort.metal`: the indexer's block order.
        "arg_sort" => facts(Reduction, false, false),
        // `qwen4exp.metal` `silu_mul`, `sigmoid_mul_f32`: one element of each
        // operand per output element.
        "silu_mul" | "sigmoid_mul" => facts(Elementwise, false, false),
        // `qwen4exp.metal` `hc_norm_f32`: a threadgroup per (row, stream),
        // the sum of squares over `d`.
        "hc_norm" => facts(Reduction, false, false),
        // `hc_collapse_f32`: the gated mean over the `n` streams; an output
        // element reads `n` inputs a stream apart.
        "hc_collapse" => facts(Reduction, false, false),
        // `hc_combine_f32`: the block output scattered across the streams,
        // one input element to `n` outputs.
        "hc_combine" => facts(Broadcast, false, false),
        // `moe_topk_f32`: softmax and `k` argmax rounds over the experts.
        // `moe_combine`: the weighted sum over the `k` selected experts.
        "moe_topk" | "moe_combine" => facts(Reduction, false, false),
        // `moe_slot_map_u32`: an index map through the residency table.
        "moe_slot_map" => facts(Movement, false, false),
        // `gdn.metal`: the DeltaNet state's prologue and step.
        "gdn_prologue" | "gdn_step" => facts(Recurrence, false, false),
        // `gdn.metal`: the per-head RMS norm and its gate.
        "gdn_gated_norm" => facts(Reduction, false, false),
        // `qwen4exp.metal` `qsa_prep_f32` / `qsa_prep_rows_f32`: a
        // threadgroup per head (and row), the head's norm then its rotation.
        "qsa_prep" | "qsa_prep_rows" => facts(Reduction, false, false),
        // `scaled_dot_product_attention.metal`: the softmax over the cache,
        // in one launch or as the 2-pass kernel's partials and their merge;
        // `flash_decoding.metal`'s split-K attention.
        "sdpa_vector"
        | "sdpa_quant"
        | "sdpa_vector_2pass_1"
        | "sdpa_vector_2pass_2"
        | "flash_decoding" => facts(Reduction, false, false),
        // `reduce.metal`'s `rms_norm<…, StoreRope>`: the head's norm, its
        // rotation at the store.
        "rmsnorm_rope" => facts(Reduction, false, false),
        // The KV appends of the `Cat` and `Paged` arms.
        "kv_cat" | "flash_kv_scatter" => facts(Movement, false, false),
        // `reduce.metal`: a sum over the last extent, an argmax.
        "fast_sum" | "fast_argmax" => facts(Reduction, false, false),
        // `rope.metal`: reads `i` and `i + d/2`, writes both.
        "rope" => facts(Pairwise, false, false),
        // `unary.metal` / `binary.metal` / `cast.metal` single ops the walk
        // names rather than types.
        "abs" | "gt" | "lt" | "sqrt" | "relu" | "neg" | "bsub" | "bmaximum" | "bminimum" => {
            facts(Elementwise, false, false)
        }
        // `indexing.metal`: the embedding gather (`is_u32`), a row gather.
        "is_u32" | "get_rows" => facts(Movement, false, false),
        // `kv_quant.metal`: a row quantized into the store's codes.
        "quantize_kv" => facts(Movement, false, false),
        // The quantized store's attention, over the live cells or the
        // selected ones.
        "sdpa_quant_selected" => facts(Reduction, false, false),
        // The conv kernels (`conv.metal`): the fused ring step and writer.
        "shortconv_step" | "shortconv_ring" => facts(Recurrence, false, false),
        f if f.starts_with("cast") => facts(Elementwise, false, false),
        // `copy2d_*`, `copy_*_strided`: strided copies.
        f if f.starts_with("copy") => facts(Movement, false, false),
        _ => return None,
    })
}

/// The facts of `node`'s dispatch; `None` for a node that does not dispatch
/// or an opaque family not declared here.
pub fn of(node: &Node) -> Option<Facts> {
    use Role::*;
    Some(match &node.op {
        Op::Input | Op::Weight | Op::View => return None,
        Op::Gemv => product(node.core),
        // The store carries the fragment already; the load is the core's.
        Op::Fused => Facts {
            store_hook: false,
            ..product(node.core)
        },
        Op::Silu | Op::Sigmoid | Op::Scale | Op::Mul | Op::Add => facts(Elementwise, false, false),
        // `reduce.metal`'s `rms_norm<…, Store>`: `StoreNorm` / `StoreRope`.
        Op::RmsNorm => facts(Reduction, true, false),
        Op::Opaque => return opaque(&node.name),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Graph, Shape};
    use lloom_tensor::DType;

    /// The cores differ in their hooks, a fused product's store is taken,
    /// and an unknown family is undeclared rather than defaulted.
    #[test]
    fn the_facts_follow_the_core_and_the_family() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F32);
        let w = g.weight("w", Shape::of(&[8, 8]), DType::F32);
        let q = g.gemv_on(Core::Quant, w, x);
        let m = g.gemv_on(Core::Mixed, w, x);
        let f = g.fused("silu", crate::optimizer::NumericClass::Rounding, w, x, None);
        let hc = g.opaque("hc_collapse", &[x], Shape::of(&[1, 1, 8]), DType::F32);
        let unknown = g.opaque("no_such_kernel", &[x], Shape::of(&[1, 1, 8]), DType::F32);
        assert_eq!(of(g.node(q)), Some(facts(Role::Product, false, false)));
        assert_eq!(of(g.node(m)), Some(facts(Role::Product, true, false)));
        assert_eq!(of(g.node(f)), Some(facts(Role::Product, false, true)));
        assert_eq!(of(g.node(hc)).map(|f| f.role), Some(Role::Reduction));
        assert_eq!(of(g.node(unknown)), None);
        assert_eq!(of(g.node(x)), None, "an input does not dispatch");
    }
}
