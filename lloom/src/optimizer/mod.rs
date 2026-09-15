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
//! exclusion set (P2's product) removes fragments by name the same way —
//! looked up under the fp model the fragment composes in, so a verdict
//! recorded for another kernel's tree is never reused (§5a). The plan also
//! reports the step's recurrences and which fused sites sit in one.

pub mod cost;
pub mod egraph;
pub mod exclusions;
pub mod fragments;
pub mod ir;
pub mod numeric;
pub mod plan;

use lloom_tensor::Result;

pub use cost::{CostModel, Marginal, StepCost};
pub use exclusions::{ExclusionSet, Lookup, Verdict};
pub use fragments::{catalogue, Fragment};
pub use ir::{Census, Dim, Graph, Node, NodeId, Op, Shape};
pub use numeric::{NumericClass, Region, RegionKind, TargetFpModel};
pub use plan::{FusionPlan, Regions, Seam, Sites, StatefulSite};

/// What the derivation may do.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    /// The weakest class a derived graph may land in. A fragment whose
    /// composition lands beyond it contributes no rule.
    pub budget: NumericClass,
    /// Fragment names the exclusion set refuses (P2's product; empty until
    /// one is computed).
    pub excluded: Vec<String>,
    /// egglog `run` iterations. The rules are local, so a handful saturates.
    pub iterations: usize,
    /// The barrier-placement term (`cost`): after extraction, a fragment
    /// whose sites cost more under it than they save — a dispatch removed
    /// but its barrier re-seated between two sweeps — is unfused and
    /// refused with the marginal named. `None` prices by dispatch count
    /// alone, which is what the extractor does.
    pub barrier_model: Option<CostModel>,
}

impl Default for Policy {
    /// `Rounding`: admits what this repository ships — both epilogues compute
    /// on the accumulator and are measured as roundings, not defects
    /// (`design/docs/fusion.md` §12.2i). A `BitExact` budget admits none of
    /// today's fragments, which is a fact about the catalogue and not a
    /// reason to widen the budget silently.
    ///
    /// `excluded` is read off the embedded exclusion set
    /// ([`exclusions::embedded`], P2): every catalogue fragment whose
    /// transform the offline validation refused — under the model the
    /// fragment composes in. A verdict recorded under another model neither
    /// excludes nor admits; it is reported by [`Policy::unvalidated`].
    fn default() -> Policy {
        let set = exclusions::embedded();
        let excluded = catalogue()
            .into_iter()
            .filter(|f| {
                matches!(
                    set.lookup(&ExclusionSet::fragment_transform(f.name), &f.model),
                    Lookup::Validated(v) if v.verdict == Verdict::Exclude
                )
            })
            .map(|f| f.name.to_string())
            .collect();
        Policy {
            budget: NumericClass::Rounding,
            excluded,
            iterations: 8,
            barrier_model: Some(CostModel::default()),
        }
    }
}

impl Policy {
    /// The catalogue fragments the exclusion set has NO verdict for under the
    /// model they compose in — admitted by class budget alone, and worth
    /// naming in a report, since "unvalidated" and "validated admissible" are
    /// different facts. Each with the reason: no entry, or an entry recorded
    /// under another model.
    pub fn unvalidated(&self) -> Vec<(String, String)> {
        let set = exclusions::embedded();
        catalogue()
            .into_iter()
            .filter_map(|f| {
                let t = ExclusionSet::fragment_transform(f.name);
                match set.lookup(&t, &f.model) {
                    Lookup::Validated(_) => None,
                    Lookup::Unvalidated => Some((f.name.to_string(), "no entry".to_string())),
                    Lookup::ModelMismatch(v) => Some((
                        f.name.to_string(),
                        format!("recorded under {}, composes under {}", v.model, f.model),
                    )),
                }
            })
            .collect()
    }
}

/// Everything one derivation produced, for the report.
#[derive(Debug)]
pub struct Derivation {
    pub plan: FusionPlan,
    /// The derived graph.
    pub graph: Graph,
    /// Fragments the policy refused, with the reason — before the rules
    /// (class, exclusion) or after extraction (the barrier term).
    pub refused: Vec<(String, String)>,
    /// Every admitted fragment's marginal under the barrier model, when one
    /// is set: what its sites cost the step against the same step with them
    /// unfused.
    pub marginals: Vec<Marginal>,
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
    let (admitted, mut refused) = egraph::admitted(policy, catalogue);
    let sat = egraph::derive(graph, &admitted, policy.iterations)?;
    let mut derived = sat.graph;

    // The barrier term, over the extracted step in emission order (§10e):
    // a fragment whose marginal is positive is unfused — the dispatch it
    // removed cost less than the barrier it re-seated — and the marginal is
    // the reason. Each fragment is judged against the step with the others
    // in, most costly first, so a refusal is charged to the fragment that
    // moved the barrier.
    let mut marginals = Vec::new();
    if let Some(model) = &policy.barrier_model {
        let names: Vec<&str> = admitted.iter().map(|f| f.name).collect();
        loop {
            let m = cost::marginals(model, &derived, &names);
            let worst = m
                .iter()
                .filter(|x| x.sites > 0 && x.delta_us > 0.0)
                .max_by(|a, b| a.delta_us.total_cmp(&b.delta_us))
                .cloned();
            match worst {
                Some(w) => {
                    let between = w
                        .between
                        .as_ref()
                        .map(|(a, b)| format!(", re-seated between {a} and {b}"))
                        .unwrap_or_default();
                    refused.push((
                        w.fragment.clone(),
                        format!(
                            "barrier term: +{:.1} µs per site over {} sites{between}",
                            w.per_site_us(),
                            w.sites
                        ),
                    ));
                    derived = cost::unfuse(&derived, &w.fragment);
                    marginals.push(w);
                }
                None => {
                    marginals.extend(m.into_iter().filter(|x| x.sites > 0));
                    break;
                }
            }
        }
    }

    let sites = plan::sites(&derived);
    let regions = plan::regions(graph, &sites);
    let cost = policy
        .barrier_model
        .as_ref()
        .map(|m| (cost::step_cost(m, graph), cost::step_cost(m, &derived)));
    let plan = FusionPlan {
        sites,
        class: derived.numeric(),
        before: graph.census(),
        after: derived.census(),
        seams: plan::seams(&derived),
        regions,
        cost,
    };
    Ok(Derivation {
        plan,
        graph: derived,
        refused,
        marginals,
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

    /// The whole catalogue on one MLP: the extractor takes the two-weight
    /// kernel for the gate (one dispatch where SiLU + the product are two),
    /// the residual on `w2`; 7 dispatches → 3, `w1` never dispatched alone.
    #[test]
    fn the_swiglu_kernel_is_derived_over_the_pair_on_the_mlp() {
        let g = mlp(1);
        let d = derive(&g, &Policy::default()).unwrap();
        assert_eq!(d.plan.sites.get("w3").map(String::as_str), Some("swiglu"));
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.sites.get("w1"), None, "{:?}", d.plan.sites);
        assert_eq!(d.plan.sites.len(), 2);
        assert_eq!(d.plan.dispatches_before(), 7);
        assert_eq!(d.plan.dispatches_after(), 3);
        assert_eq!(d.plan.class, NumericClass::Rounding);
        assert_eq!(d.plan.after.get("gemv_swiglu"), Some(&1));
        assert_eq!(d.plan.after.get("gemv_residual"), Some(&1));
        assert_eq!(d.plan.after.get("gemv"), None);
        assert!(!d.plan.after.contains_key("silu"));
        assert!(!d.plan.after.contains_key("bmul"));
        assert!(d.plan.seams.is_empty(), "{:?}", d.plan.seams);
        // The fused node's operand is the gate WEIGHT.
        let sw = d
            .graph
            .nodes()
            .iter()
            .find(|n| n.op == Op::Fused && n.name == "swiglu")
            .unwrap();
        let operand = d.graph.node(sw.inputs[2]);
        assert_eq!(
            (operand.op.clone(), operand.name.as_str()),
            (Op::Weight, "w1")
        );
    }

    /// With the two-weight kernel excluded, the pair it supersedes: SiLU on
    /// `w1`, the product on `w3` with the fused gate as its operand.
    #[test]
    fn all_three_epilogues_are_derived_on_the_mlp() {
        let g = mlp(1);
        let policy = Policy {
            excluded: vec!["swiglu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &policy).unwrap();
        assert_eq!(d.plan.sites.get("w1").map(String::as_str), Some("silu"));
        assert_eq!(d.plan.sites.get("w3").map(String::as_str), Some("mul"));
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.sites.len(), 3);
        assert_eq!(d.plan.dispatches_before(), 7);
        assert_eq!(d.plan.dispatches_after(), 4);
        assert_eq!(d.plan.class, NumericClass::Rounding);
        assert_eq!(d.plan.after.get("gemv_silu"), Some(&1));
        assert_eq!(d.plan.after.get("gemv_mul"), Some(&1));
        assert_eq!(d.plan.after.get("gemv_residual"), Some(&1));
        assert!(!d.plan.after.contains_key("silu"));
        assert!(!d.plan.after.contains_key("bmul"));
        assert!(!d.plan.after.contains_key("badd"));
        // No seam is left: the gate*up product rides `w3`'s store, its
        // operand the fused SiLU'd gate.
        assert!(d.plan.seams.is_empty(), "{:?}", d.plan.seams);
        // The fused product's operand is the fused gate, not the bare GEMV.
        let mul = d
            .graph
            .nodes()
            .iter()
            .find(|n| n.op == Op::Fused && n.name == "mul")
            .unwrap();
        let operand = d.graph.node(mul.inputs[2]);
        assert_eq!(
            (operand.op.clone(), operand.name.as_str()),
            (Op::Fused, "silu")
        );
    }

    /// The barrier term, on an MLP at LFM2's shape (`w1`/`w3` `[10752,
    /// 2048]` f16 = 44 MB each), with the two-weight kernel excluded so the
    /// gate is the `mul` fragment: it removes a dispatch and re-seats the
    /// product's barrier between the two sweeps, which the model prices above
    /// the dispatch — refused, with the marginal named and the pair it landed
    /// between; SiLU and the residual save. On the small MLP above the same
    /// fragment is admitted: the verdict is the weights', not the fragment's.
    /// `barrier_model: None` is the extractor's own count and admits all
    /// three. And the two-weight kernel at this shape is admitted at a margin
    /// the crown's loss could not dent: one dispatch for four, its only
    /// barrier the residual's beside `w2`.
    #[test]
    fn the_barrier_term_refuses_the_gate_at_lfm2s_shape_and_admits_it_small() {
        let mut g = Graph::new();
        let r = g.input("h", Shape::of(&[1, 1, 2048]), DType::F16);
        let nw = g.weight("norm", Shape::of(&[2048]), DType::F16);
        let x = g.rms_norm(nw, r);
        let w1 = g.weight("w1", Shape::of(&[10752, 2048]), DType::F16);
        let w3 = g.weight("w3", Shape::of(&[10752, 2048]), DType::F16);
        let w2 = g.weight("w2", Shape::of(&[2048, 10752]), DType::F16);
        let gate = g.gemv(w1, x);
        let gate = g.silu(gate);
        let up = g.gemv(w3, x);
        let gated = g.mul(gate, up);
        let down = g.gemv(w2, gated);
        let out = g.add(r, down);
        g.mark_root(out);

        let no_swiglu = Policy {
            excluded: vec!["swiglu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &no_swiglu).unwrap();
        assert_eq!(d.plan.sites.get("w1").map(String::as_str), Some("silu"));
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.sites.get("w3"), None, "{:?}", d.plan.sites);
        assert_eq!(d.plan.dispatches_after(), 5);
        let (_, why) = d.refused.iter().find(|(n, _)| n == "mul").unwrap();
        assert!(
            why.contains("re-seated between gemv_silu and gemv_mul"),
            "{why}"
        );
        let mul = d.marginals.iter().find(|m| m.fragment == "mul").unwrap();
        // By hand from the two declared constants. Fused, the product's
        // node sits between two 44 MB sweeps and the model charges BOTH
        // barriers at the smaller neighbour's bytes: before `gemv_mul`
        // (`gemv_silu`'s w1 + x + out) and before `gemv_residual` (w2 +
        // gated + r + out). Unfused, the `bmul` is the spacer on both sides
        // and each barrier is its three 21.5 KB vectors. Less the dispatch.
        let bw = 790e3;
        let (w, x, o, r) = (
            10752.0 * 2048.0 * 2.0,
            2048.0 * 2.0,
            10752.0 * 2.0,
            2048.0 * 2.0,
        );
        let expect = (w + x + o) / bw + (w + o + r + r) / bw - 49.0 - 2.0 * (3.0 * o) / bw;
        assert!(
            (mul.per_site_us() - expect).abs() < 1e-6,
            "{} vs {expect}",
            mul.per_site_us()
        );
        for f in ["silu", "residual"] {
            let m = d.marginals.iter().find(|m| m.fragment == f).unwrap();
            assert!(m.delta_us < 0.0, "{f}: {m:?}");
        }
        assert!(d.plan.to_string().contains(" cost[before: "), "{}", d.plan);
        let (before, after) = d.plan.cost.as_ref().unwrap();
        assert!(after.total_us() < before.total_us());
        // The seams the refusal leaves are the product's two.
        assert_eq!(d.plan.seams.len(), 2, "{:?}", d.plan.seams);

        let none = Policy {
            barrier_model: None,
            excluded: vec!["swiglu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &none).unwrap();
        assert_eq!(d.plan.sites.len(), 3);
        assert_eq!(d.refused.len(), 1);
        assert!(d.marginals.is_empty());
        assert!(d.plan.cost.is_none());

        let d = derive(&g, &Policy::default()).unwrap();
        assert_eq!(d.plan.sites.get("w3").map(String::as_str), Some("swiglu"));
        assert_eq!(d.plan.dispatches_after(), 3);
        assert!(d.refused.is_empty(), "{:?}", d.refused);
        let sw = d.marginals.iter().find(|m| m.fragment == "swiglu").unwrap();
        assert!(sw.delta_us < -80.0, "{sw:?}");
    }

    /// With both gate fragments excluded, the SwiGLU product is the seam the
    /// first catalogue left — pinned so the seam report stays exercised.
    #[test]
    fn without_the_gate_fragment_the_product_is_the_seam_left() {
        let g = mlp(1);
        let policy = Policy {
            excluded: vec!["mul".to_string(), "swiglu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &policy).unwrap();
        assert_eq!(d.plan.sites.len(), 2);
        assert_eq!(d.plan.dispatches_after(), 5);
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

    /// A fused site inside a recurrence is reported, not hidden: the
    /// residual onto a GEMV whose add the emitter marked as a ring's state
    /// write is `fused-in-stateful`; the same graph unmarked reports none.
    #[test]
    fn a_fused_site_in_a_marked_region_is_reported() {
        let build = |mark: bool| {
            let mut g = Graph::new();
            let r = g.input("h", Shape::of(&[1, 1, 8]), DType::F16);
            let w = g.weight("w", Shape::of(&[8, 8]), DType::F16);
            let y = g.gemv(w, r);
            let out = g.add(r, y);
            g.mark_root(out);
            if mark {
                g.mark_region(out, Region::ConvRing(2));
            }
            g
        };
        let d = derive(&build(false), &Policy::default()).unwrap();
        assert_eq!(d.plan.sites.get("w").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.regions, Regions::default());
        let d = derive(&build(true), &Policy::default()).unwrap();
        assert_eq!(d.plan.sites.get("w").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.regions.conv_ring, 1);
        assert_eq!(d.plan.regions.nodes, 1);
        assert_eq!(
            d.plan.regions.fused_in_stateful,
            vec![StatefulSite {
                site: "w".to_string(),
                fragment: "residual".to_string(),
                region: Region::ConvRing(2),
            }]
        );
        assert!(d
            .plan
            .to_string()
            .contains("fused-in-stateful=1 w:residual@conv-ring@2"));
    }

    /// The default policy asks the set under each fragment's own model; a
    /// fragment the set only knows under another model is unvalidated, with
    /// the two models named.
    #[test]
    fn a_fragment_validated_under_another_model_is_unvalidated_here() {
        let policy = Policy::default();
        assert!(
            policy.unvalidated().is_empty(),
            "{:?}",
            policy.unvalidated()
        );
        let set = exclusions::embedded();
        let other = TargetFpModel::of_family("shortconv_step").unwrap();
        for f in catalogue() {
            let t = ExclusionSet::fragment_transform(f.name);
            assert!(matches!(set.lookup(&t, &f.model), Lookup::Validated(_)));
            assert!(matches!(set.lookup(&t, &other), Lookup::ModelMismatch(_)));
        }
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
        assert_eq!(d.refused.len(), 4);
        assert_eq!(d.plan.dispatches_after(), 7);
        assert_eq!(d.plan.class, NumericClass::BitExact);
    }

    #[test]
    fn the_exclusion_set_removes_a_fragment_by_name() {
        let g = mlp(1);
        let policy = Policy {
            excluded: vec!["silu".to_string(), "swiglu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &policy).unwrap();
        assert_eq!(d.plan.sites.get("w1"), None);
        assert_eq!(d.plan.sites.get("w2").map(String::as_str), Some("residual"));
        assert_eq!(d.plan.sites.get("w3").map(String::as_str), Some("mul"));
        assert_eq!(
            d.refused,
            vec![
                (
                    "silu".to_string(),
                    "excluded by the exclusion set".to_string()
                ),
                (
                    "swiglu".to_string(),
                    "excluded by the exclusion set".to_string()
                )
            ]
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
