//! The optimizer (`design/docs/optimizer-design.md`, P1): a tensor-op IR the
//! model emits from its own configuration, egglog as the rewrite substrate,
//! and derivations generated from the fragment catalogue — so which
//! mechanisms a decode step takes is DERIVED from what the ops compute and
//! what the kernels can compose, rather than selected by a flag per site.
//!
//! What it derives today is the elementwise family: an elementwise dispatch
//! whose single-use input is a GEMV store becomes that block riding the
//! store, for every block the runtime seam admits (`nn::fusion`'s
//! `Epilogue` set). The result is a [`FusionPlan`], resolved once at model
//! construction into the same per-site bools the forward pass already reads
//! — nothing runs per token (`CONTRIBUTING.md` §3.3a).
//!
//! The numeric obligation is a class ([`NumericClass`]) each fragment
//! carries, checked against the policy's budget before its rule exists; an
//! exclusion set (P2's product) removes fragments by name the same way.

pub mod egraph;
pub mod fragments;
pub mod ir;
pub mod numeric;
pub mod plan;

use lloom_tensor::Result;

pub use fragments::{catalogue, Fragment};
pub use ir::{Census, Dim, Graph, Node, NodeId, Op, Shape};
pub use numeric::NumericClass;
pub use plan::{FusionPlan, Seam, Sites};

/// What the derivation may do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Policy {
    /// The weakest class a derived graph may land in. A fragment whose
    /// composition lands beyond it contributes no rule.
    pub budget: NumericClass,
    /// Fragment names the exclusion set refuses (P2's product; empty until
    /// one is computed).
    pub excluded: Vec<String>,
    /// egglog `run` iterations. The rules are local, so a handful saturates.
    pub iterations: usize,
}

impl Default for Policy {
    /// `Rounding`: admits what this repository ships — both epilogues compute
    /// on the accumulator and are measured as roundings, not defects
    /// (`design/docs/fusion.md` §12.2i). A `BitExact` budget admits none of
    /// today's fragments, which is a fact about the catalogue and not a
    /// reason to widen the budget silently.
    fn default() -> Policy {
        Policy {
            budget: NumericClass::Rounding,
            excluded: Vec::new(),
            iterations: 8,
        }
    }
}

/// Everything one derivation produced, for the report.
#[derive(Debug)]
pub struct Derivation {
    pub plan: FusionPlan,
    /// The derived graph.
    pub graph: Graph,
    /// Fragments the policy refused, with the reason.
    pub refused: Vec<(String, String)>,
    /// The egglog program that ran.
    pub program: String,
    pub elapsed: std::time::Duration,
}

/// Derive over `graph` under `policy` with the shipped catalogue.
pub fn derive(graph: &Graph, policy: &Policy) -> Result<Derivation> {
    derive_with(graph, policy, &catalogue())
}

/// Derive over `graph` under `policy` with an explicit catalogue.
pub fn derive_with(graph: &Graph, policy: &Policy, catalogue: &[Fragment]) -> Result<Derivation> {
    let (admitted, refused) = egraph::admitted(policy, catalogue);
    let sat = egraph::derive(graph, &admitted, policy.iterations)?;
    let plan = FusionPlan {
        sites: plan::sites(&sat.graph),
        class: sat.graph.numeric(),
        before: graph.census(),
        after: sat.graph.census(),
        seams: plan::seams(&sat.graph),
    };
    Ok(Derivation {
        plan,
        graph: sat.graph,
        refused,
        program: sat.program,
        elapsed: sat.elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::DType;

    /// One SwiGLU MLP with its residual: `add(gemv(w2, mul(silu(gemv(w1, x)),
    /// gemv(w3, x))), r)`.
    fn mlp(rows: usize) -> Graph {
        let mut g = Graph::new();
        let r = g.input("h", Shape::of(&[1, rows, 8]), DType::F16);
        let nw = g.weight("norm", Shape::of(&[8]), DType::F16);
        let x = g.rms_norm(nw, r);
        let w1 = g.weight("w1", Shape::of(&[16, 8]), DType::F16);
        let w3 = g.weight("w3", Shape::of(&[16, 8]), DType::F16);
        let w2 = g.weight("w2", Shape::of(&[8, 16]), DType::F16);
        let gate = g.gemv(w1, x);
        let gate = g.silu(gate);
        let up = g.gemv(w3, x);
        let gated = g.mul(gate, up);
        let down = g.gemv(w2, gated);
        let out = g.add(r, down);
        g.mark_root(out);
        g
    }

    #[test]
    fn both_epilogues_are_derived_on_the_mlp() {
        let g = mlp(1);
        let d = derive(&g, &Policy::default()).unwrap();
        assert_eq!(d.plan.sites.get("w1").map(String::as_str), Some("silu"));
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.sites.len(), 2);
        assert_eq!(d.plan.dispatches_before(), 7);
        assert_eq!(d.plan.dispatches_after(), 5);
        assert_eq!(d.plan.class, NumericClass::Rounding);
        assert_eq!(d.plan.after.get("gemv_silu"), Some(&1));
        assert_eq!(d.plan.after.get("gemv_residual"), Some(&1));
        assert!(!d.plan.after.contains_key("silu"));
        assert!(!d.plan.after.contains_key("badd"));
        // The gate*up product is the one seam left: a fused GEMV feeding a
        // multiply the catalogue has no fragment for.
        assert_eq!(
            d.plan.seams,
            vec![
                Seam {
                    producer: "gemv".into(),
                    consumer: "bmul".into(),
                    count: 1
                },
                Seam {
                    producer: "gemv_silu".into(),
                    consumer: "bmul".into(),
                    count: 1
                }
            ]
        );
    }

    #[test]
    fn a_bit_exact_budget_admits_no_shipped_fragment() {
        let g = mlp(1);
        let policy = Policy {
            budget: NumericClass::BitExact,
            ..Default::default()
        };
        let d = derive(&g, &policy).unwrap();
        assert!(d.plan.sites.is_empty());
        assert_eq!(d.refused.len(), 2);
        assert_eq!(d.plan.dispatches_after(), 7);
        assert_eq!(d.plan.class, NumericClass::BitExact);
    }

    #[test]
    fn the_exclusion_set_removes_a_fragment_by_name() {
        let g = mlp(1);
        let policy = Policy {
            excluded: vec!["silu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &policy).unwrap();
        assert_eq!(d.plan.sites.get("w1"), None);
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(
            d.refused,
            vec![(
                "silu".to_string(),
                "excluded by the exclusion set".to_string()
            )]
        );
    }

    #[test]
    fn a_batched_step_is_not_the_gemv_route_and_fuses_nothing() {
        let g = mlp(4);
        let d = derive(&g, &Policy::default()).unwrap();
        assert!(
            d.plan.sites.is_empty(),
            "m = 4 is a GEMM; the fused arms are m = 1"
        );
    }

    #[test]
    fn a_store_with_two_readers_is_not_fused() {
        // silu(gemv(w1, x)) where gemv(w1, x) is ALSO read by a second add:
        // fusing the SiLU would leave the second reader without the unfused
        // value. G3 adjacency.
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w1 = g.weight("w1", Shape::of(&[8, 8]), DType::F16);
        let gate = g.gemv(w1, x);
        let s = g.silu(gate);
        let other = g.add(gate, x);
        let out = g.mul(s, other);
        g.mark_root(out);
        let d = derive(&g, &Policy::default()).unwrap();
        assert!(d.plan.sites.is_empty(), "{:?}", d.plan.sites);
        assert_eq!(d.plan.dispatches_after(), 4);
    }

    #[test]
    fn the_residual_fuses_from_either_operand_order() {
        for swap in [false, true] {
            let mut g = Graph::new();
            let r = g.input("h", Shape::of(&[1, 1, 8]), DType::F16);
            let w = g.weight("w", Shape::of(&[8, 8]), DType::F16);
            let y = g.gemv(w, r);
            let out = if swap { g.add(y, r) } else { g.add(r, y) };
            g.mark_root(out);
            let d = derive(&g, &Policy::default()).unwrap();
            assert_eq!(
                d.plan.sites.get("w").map(String::as_str),
                Some("residual"),
                "swap={swap}"
            );
        }
    }

    #[test]
    fn a_view_between_the_store_and_the_load_is_a_second_reader() {
        // The in_proj shape: one GEMV read through three narrows. Nothing
        // fuses onto it, even though each narrow is single-use.
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 4]), DType::F16);
        let w = g.weight("in_proj", Shape::of(&[12, 4]), DType::F16);
        let bcx = g.gemv(w, x);
        let b = g.view("narrow", bcx, Shape::of(&[1, 1, 4]));
        let c = g.view("narrow", bcx, Shape::of(&[1, 1, 4]));
        let xx = g.view("narrow", bcx, Shape::of(&[1, 1, 4]));
        let bx = g.mul(b, xx);
        let out = g.add(bx, c);
        g.mark_root(out);
        let d = derive(&g, &Policy::default()).unwrap();
        assert!(d.plan.sites.is_empty());
        assert_eq!(d.plan.dispatches_after(), 3);
        // And the graph round-trips its views and shapes.
        assert_eq!(d.graph.dispatches(), 3);
        assert_eq!(
            d.graph.nodes().iter().filter(|n| n.op == Op::View).count(),
            3
        );
    }

    #[test]
    fn opaque_nodes_round_trip_with_their_shapes() {
        let mut g = Graph::new();
        let q = g.input("q", Shape::of(&[1, 8, 1, 64]), DType::F16);
        let k = g.input("k", Shape::of(&[1, 2, 40, 64]), DType::F16);
        let v = g.input("v", Shape::of(&[1, 2, 40, 64]), DType::F16);
        let y = g.opaque("sdpa", &[q, k, v], Shape::of(&[1, 8, 1, 64]), DType::F16);
        let w = g.weight("o", Shape::of(&[512, 512]), DType::F16);
        let yv = g.view("reshape", y, Shape::of(&[1, 1, 512]));
        let o = g.gemv(w, yv);
        g.mark_root(o);
        let d = derive(&g, &Policy::default()).unwrap();
        assert_eq!(d.plan.after.get("sdpa"), Some(&1));
        let sd = d.graph.nodes().iter().find(|n| n.name == "sdpa").unwrap();
        assert_eq!(sd.shape, Shape::of(&[1, 8, 1, 64]));
        assert_eq!(sd.inputs.len(), 3);
    }
}
