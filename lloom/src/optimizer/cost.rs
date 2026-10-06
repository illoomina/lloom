//! The barrier-placement term (`design/docs/optimizer-design.md` §10e, §8.2
//! query 4's efficiency half): what a derived step costs in a model that sees
//! WHERE each barrier lands, not only how many dispatches there are.
//!
//! Measured on the crown (#732, `measurements/issue-732-swiglu-gate.md`): the
//! SwiGLU product fused onto `w3`'s store removes 30 dispatches per token and
//! the token gets slower, because the product's RAW barrier is not removed —
//! it moves from between `w3` and a 2 µs `bmul` to between `w1` and `w3`, the
//! layer's two 44 MB weight sweeps, which then cannot overlap. A cost that
//! counts dispatches cannot see that; this one can.
//!
//! The model is a heuristic with two declared constants and no free
//! parameter, and it is flow-SENSITIVE — it walks the step in emission order
//! — which is why it runs over the extracted graph rather than inside the
//! e-graph (§4: barrier placement is a property of program order, not of an
//! e-class). A barrier before a dispatch costs the overlap it forbids between
//! the dispatch before it and the one after it, bounded by the smaller of the
//! two kernels' work; a dispatch costs its per-dispatch wall. The verdict on a
//! fragment is its MARGINAL cost: the derived step against the same step with
//! that fragment's sites unfused, both priced in emission order.

use std::collections::HashSet;
use std::fmt;

use lloom_tensor::DType;

use super::ir::{family, Dim, Graph, Node, NodeId, Op, Shape};

/// The model's two constants, both measured on this machine and declared
/// here with their sources; a different machine re-declares them
/// (`CONTRIBUTING.md` §2.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// What one dispatch costs the step, in µs — the GPU-side per-dispatch
    /// wall §9a measures at 47–49 µs on the decode step, invariant to the
    /// arena, the hazard key and the cut period (`issue-727-pass-plans.md`
    /// §19).
    pub dispatch_us: f64,
    /// The rate a weight sweep proceeds at when its neighbours overlap it:
    /// #715 prices the step's four GEMV families' summed intervals at
    /// ~790 GB/s over a 5.4 GB sweep (`issue-715-cut-period.md` — "a kernel's
    /// interval depends on what preceded it"). In bytes per µs.
    pub bytes_per_us: f64,
}

impl Default for CostModel {
    fn default() -> CostModel {
        CostModel {
            dispatch_us: 49.0,
            bytes_per_us: 790e3,
        }
    }
}

/// One barrier the sequential hazard model emits: before `after`, because it
/// reads what `before`-or-later wrote since the last barrier; priced as the
/// overlap it forbids between `before` (the dispatch just ahead of it) and
/// `after`.
#[derive(Debug, Clone, PartialEq)]
pub struct Barrier {
    pub before: NodeId,
    pub after: NodeId,
    pub cost_us: f64,
}

/// A step's cost under the model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepCost {
    pub dispatches: u64,
    pub dispatch_us: f64,
    pub barriers: Vec<Barrier>,
    pub barrier_us: f64,
}

impl StepCost {
    pub fn total_us(&self) -> f64 {
        self.dispatch_us + self.barrier_us
    }
}

impl fmt::Display for StepCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} dispatches {:.0} µs + {} barriers {:.0} µs = {:.0} µs",
            self.dispatches,
            self.dispatch_us,
            self.barriers.len(),
            self.barrier_us,
            self.total_us()
        )
    }
}

/// A fragment's marginal verdict under the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Marginal {
    pub fragment: String,
    pub sites: usize,
    /// The derived step's cost minus the same step with this fragment's
    /// sites unfused: negative saves, positive costs.
    pub delta_us: f64,
    /// The family pair the fragment's barrier lands between, at the first
    /// site, when it costs: "gemv_silu→gemv_mul".
    pub between: Option<(String, String)>,
}

impl Marginal {
    pub fn per_site_us(&self) -> f64 {
        if self.sites == 0 {
            0.0
        } else {
            self.delta_us / self.sites as f64
        }
    }
}

pub(crate) fn elems(shape: &Shape) -> f64 {
    shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Const(n) => *n as f64,
            // A symbolic extent (`kv_len`) is the same in the source and the
            // derived step; priced at 1 so it neither dominates nor vanishes.
            Dim::Sym(_) => 1.0,
        })
        .product()
}

pub(crate) fn bytes(shape: &Shape, dtype: DType) -> f64 {
    elems(shape) * dtype.size_in_bytes() as f64
}

/// The bytes a dispatch moves: what it reads (its inputs, weights included)
/// and what it writes. A GEMV is its weight sweep; an elementwise kernel its
/// two vectors; a view or a leaf moves nothing.
pub fn bytes_moved(graph: &Graph, id: NodeId) -> f64 {
    let node = graph.node(id);
    if !node.op.dispatches() {
        return 0.0;
    }
    let mut total = bytes(&node.shape, node.dtype);
    for &i in &node.inputs {
        let inp = graph.node(i);
        total += bytes(&inp.shape, inp.dtype);
    }
    total
}

/// Emission order: a post-order walk from the roots, inputs in order, each
/// node once. On the graph the emitter produces this is the emitter's order;
/// on the rebuilt graph it is the same procedure, so the two are comparable
/// — and it is what makes the fused product's operand (the gate) come before
/// the fused `w3`, as the layer runs it.
pub fn order(graph: &Graph) -> Vec<NodeId> {
    fn visit(graph: &Graph, id: NodeId, seen: &mut Vec<bool>, out: &mut Vec<NodeId>) {
        if seen[id] {
            return;
        }
        seen[id] = true;
        for &i in &graph.node(id).inputs {
            visit(graph, i, seen, out);
        }
        out.push(id);
    }
    let mut seen = vec![false; graph.nodes().len()];
    let mut out = Vec::with_capacity(graph.nodes().len());
    for &r in graph.roots() {
        visit(graph, r, &mut seen, &mut out);
    }
    out
}

/// The producing dispatch a node reads through views, if any.
fn producer(graph: &Graph, mut id: NodeId) -> Option<NodeId> {
    loop {
        let n = graph.node(id);
        match n.op {
            Op::View => id = n.inputs[0],
            Op::Input | Op::Weight => return None,
            _ => return Some(id),
        }
    }
}

/// Price `graph` in emission order under the sequential hazard model: a
/// barrier goes before a dispatch that reads a value written since the last
/// barrier (RAW; what candle's `auto_barrier` emits for the decode step,
/// which #727 measured as 88 % pure RAW), and costs the overlap it forbids
/// between the dispatch just before it and the one it guards.
pub fn step_cost(model: &CostModel, graph: &Graph) -> StepCost {
    let mut cost = StepCost::default();
    let mut dirty: HashSet<NodeId> = HashSet::new();
    let mut prev: Option<NodeId> = None;
    for id in order(graph) {
        let node: &Node = graph.node(id);
        if !node.op.dispatches() {
            continue;
        }
        cost.dispatches += 1;
        cost.dispatch_us += model.dispatch_us;
        let hazard = node
            .inputs
            .iter()
            .filter_map(|&i| producer(graph, i))
            .any(|p| dirty.contains(&p));
        if hazard {
            if let Some(before) = prev {
                let w = bytes_moved(graph, before).min(bytes_moved(graph, id)) / model.bytes_per_us;
                cost.barrier_us += w;
                cost.barriers.push(Barrier {
                    before,
                    after: id,
                    cost_us: w,
                });
            }
            dirty.clear();
        }
        dirty.insert(id);
        prev = Some(id);
    }
    cost
}

/// `graph` with every `Fused` node of `fragment` replaced by the pair it
/// stands for — the GEMV, then the elementwise dispatch reading its store
/// (and the operand, if the fragment binds one). The inverse of the
/// derivation for one fragment, without egglog; what the marginal is priced
/// against.
pub fn unfuse(graph: &Graph, fragment: &str) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Option<NodeId>> = vec![None; graph.nodes().len()];
    // A prologue's norm, once per (weight, input): its readers shared it.
    let mut norms: std::collections::HashMap<(NodeId, NodeId), NodeId> =
        std::collections::HashMap::new();
    let class_of = |name: &str| {
        super::fragments::catalogue()
            .into_iter()
            .find(|f| f.name == name)
            .map(|f| f.class())
            .unwrap_or(super::numeric::NumericClass::BitExact)
    };
    for id in order(graph) {
        let n = graph.node(id);
        let inputs: Vec<NodeId> = n
            .inputs
            .iter()
            .map(|&i| map[i].expect("post-order: inputs first"))
            .collect();
        let new = match n.op {
            Op::Fused if n.carries(fragment) => {
                // The node's own parts: the epilogue's operand and the
                // prologue's weight, in the mapped graph.
                let operand = n.operand().map(|i| map[i].expect("post-order"));
                let pw = n.prologue_weight().map(|i| map[i].expect("post-order"));
                let (w, x) = (inputs[0], inputs[1]);
                if n.prologue.as_deref() == Some(fragment) {
                    // The prologue back to the norm it computed.
                    let pw = pw.expect("a prologue carries its weight");
                    let norm = *norms.entry((pw, x)).or_insert_with(|| out.rms_norm(pw, x));
                    if n.name.is_empty() {
                        out.gemv_on(n.core, w, norm)
                    } else {
                        out.fused_on(n.core, &n.name, class_of(&n.name), w, norm, operand, n.scalar)
                    }
                } else {
                    // The epilogue back to its pair; a prologue stays on the
                    // product (and on the gate's, for the two-weight kernel).
                    let product = |out: &mut Graph, w: NodeId| match (&n.prologue, pw) {
                        (Some(p), Some(pw)) => out.fused_with_prologue(
                            n.core,
                            "",
                            super::numeric::NumericClass::BitExact,
                            w,
                            x,
                            None,
                            None,
                            (p.as_str(), pw, class_of(p)),
                        ),
                        _ => out.gemv_on(n.core, w, x),
                    };
                    let g = product(&mut out, w);
                    // A scaling block stands for its constant's `affine` too,
                    // unless the constant is 1 (nothing was absorbed).
                    let scaled = |out: &mut Graph, g: NodeId| match n.scalar {
                        Some(c) if c != 1.0 => out.scale(g, c),
                        _ => g,
                    };
                    match (n.name.as_str(), operand) {
                        ("mixed_silu", None) => {
                            let s = scaled(&mut out, g);
                            out.silu(s)
                        }
                        ("mixed_sigmoid", None) => {
                            let s = scaled(&mut out, g);
                            out.sigmoid(s)
                        }
                        ("silu", None) => out.silu(g),
                        ("residual", Some(r)) => out.add(g, r),
                        // The emitter's operand order: `mul(gate, up)`, the
                        // gate first — so the pair is priced in the order the
                        // layer runs.
                        ("mul", Some(r)) => out.mul(r, g),
                        // The two-weight kernel stands for four dispatches:
                        // `gemv(w1)`, `silu`, `gemv(w3)`, `mul(gate, up)` —
                        // the emitter's order. The operand is the gate WEIGHT.
                        ("swiglu", Some(w1)) => {
                            let gate = product(&mut out, w1);
                            let gate = out.silu(gate);
                            out.mul(gate, g)
                        }
                        (other, _) => panic!("optimizer: no unfused pair for fragment `{other}`"),
                    }
                }
            }
            _ => {
                let mut node = n.clone();
                node.inputs = inputs;
                out.push(node)
            }
        };
        map[id] = Some(new);
    }
    for &r in graph.roots() {
        out.mark_root(map[r].expect("roots are visited"));
    }
    out
}

/// Each fragment's marginal cost on `derived`: the step as derived against
/// the step with that fragment's sites unfused. The pair a costing
/// fragment's barrier lands between is named from the first such site.
pub fn marginals(model: &CostModel, derived: &Graph, fragments: &[&str]) -> Vec<Marginal> {
    let full = step_cost(model, derived);
    fragments
        .iter()
        .map(|&f| {
            let sites = derived
                .nodes()
                .iter()
                .filter(|n| n.carries(f))
                .count();
            if sites == 0 {
                return Marginal {
                    fragment: f.to_string(),
                    sites: 0,
                    delta_us: 0.0,
                    between: None,
                };
            }
            let without = step_cost(model, &unfuse(derived, f));
            let delta_us = full.total_us() - without.total_us();
            let between = full
                .barriers
                .iter()
                .find(|b| {
                    derived.node(b.after).carries(f)
                })
                .map(|b| {
                    (
                        family(derived.node(b.before)).unwrap_or_default(),
                        family(derived.node(b.after)).unwrap_or_default(),
                    )
                });
            Marginal {
                fragment: f.to_string(),
                sites,
                delta_us,
                between,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::numeric::NumericClass;

    /// Two big sweeps and a small consumer, unfused and fused, priced by hand.
    #[test]
    fn a_barrier_between_two_sweeps_costs_the_smaller_sweep() {
        let model = CostModel {
            dispatch_us: 10.0,
            bytes_per_us: 1000.0,
        };
        // Unfused: gemv(w1,x) → silu → gemv(w3,x) → mul(silu, up).
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 100]), DType::F16);
        let w1 = g.weight("w1", Shape::of(&[500, 100]), DType::F16);
        let w3 = g.weight("w3", Shape::of(&[500, 100]), DType::F16);
        let a = g.gemv(w1, x);
        let s = g.silu(a);
        let b = g.gemv(w3, x);
        let m = g.mul(s, b);
        g.mark_root(m);
        let c = step_cost(&model, &g);
        assert_eq!(c.dispatches, 4);
        // Barriers: before silu (reads a; between a and silu: min(w_a, w_silu))
        // and before mul (reads b; between b and mul). Nothing between the
        // two GEMVs: silu's write is dirty but b reads x only.
        assert_eq!(c.barriers.len(), 2);
        let w_gemv = bytes_moved(&g, a);
        let w_silu = bytes_moved(&g, s);
        assert!(w_gemv > w_silu * 10.0, "{w_gemv} vs {w_silu}");
        assert!((c.barriers[0].cost_us - w_silu / 1000.0).abs() < 1e-9);
        // Fused: fused_silu(w1,x) → fused_mul(w3, x, gate): the barrier now
        // sits between the two sweeps and costs a whole sweep.
        let mut f = Graph::new();
        let x = f.input("x", Shape::of(&[1, 1, 100]), DType::F16);
        let w1 = f.weight("w1", Shape::of(&[500, 100]), DType::F16);
        let w3 = f.weight("w3", Shape::of(&[500, 100]), DType::F16);
        let gate = f.fused("silu", NumericClass::Rounding, w1, x, None);
        let out = f.fused("mul", NumericClass::Rounding, w3, x, Some(gate));
        f.mark_root(out);
        let cf = step_cost(&model, &f);
        assert_eq!(cf.dispatches, 2);
        assert_eq!(cf.barriers.len(), 1);
        assert_eq!((cf.barriers[0].before, cf.barriers[0].after), (gate, out));
        let w_fused = bytes_moved(&f, out).min(bytes_moved(&f, gate));
        assert!((cf.barriers[0].cost_us - w_fused / 1000.0).abs() < 1e-9);
        // The marginal of `mul`: two dispatches saved against one sweep-sized
        // barrier; here the sweep (100 KB) is 100 µs at 1 KB/µs, the two
        // dispatches 20 µs — so the fusion COSTS.
        let m = marginals(&model, &f, &["mul", "silu"]);
        assert_eq!(m[0].sites, 1);
        assert!(m[0].delta_us > 0.0, "{:?}", m[0]);
        assert_eq!(
            m[0].between,
            Some(("gemv_silu".to_string(), "gemv_mul".to_string()))
        );
        // In the fully fused graph `silu` is charged too: with `mul` fused,
        // unfusing `silu` puts a tiny kernel between the two sweeps and the
        // big barrier goes cheap — both fragments share the blame, which is
        // why the derivation refuses the worst one and re-prices the rest.
        assert!(m[1].delta_us > 0.0, "{:?}", m[1]);
        let without_mul = unfuse(&f, "mul");
        let m2 = marginals(&model, &without_mul, &["silu"]);
        assert!(m2[0].delta_us < 0.0, "{:?}", m2[0]);
        // Unfusing `mul` restores the pair's shape.
        let u = unfuse(&f, "mul");
        assert_eq!(u.dispatches(), 3);
        assert_eq!(u.census().get("bmul"), Some(&1));
        assert_eq!(u.census().get("gemv_silu"), Some(&1));
    }

    /// Small weights: the barrier between two tiny sweeps costs less than
    /// the dispatch it removes, so the same fusion SAVES — the verdict is
    /// shape-aware, not a rule against the fragment.
    #[test]
    fn the_verdict_follows_the_weights_not_the_fragment() {
        let model = CostModel::default();
        let build = |n: usize, k: usize| {
            let mut f = Graph::new();
            let x = f.input("x", Shape::of(&[1, 1, k]), DType::F16);
            let w1 = f.weight("w1", Shape::of(&[n, k]), DType::F16);
            let w3 = f.weight("w3", Shape::of(&[n, k]), DType::F16);
            let gate = f.fused("silu", NumericClass::Rounding, w1, x, None);
            let out = f.fused("mul", NumericClass::Rounding, w3, x, Some(gate));
            f.mark_root(out);
            f
        };
        let small = marginals(&model, &build(512, 512), &["mul"]);
        assert!(small[0].delta_us < 0.0, "{:?}", small[0]);
        // LFM2's w3: [10752, 2048] f16 = 44 MB → 55.8 µs at 790 GB/s against
        // 49 µs per dispatch, less the tiny barrier the unfused `bmul` owed
        // (its three 21.5 KB vectors): +6.7 µs per site, the sign the crown
        // measured. By hand from the two declared constants.
        let lfm2 = marginals(&model, &build(10752, 2048), &["mul"]);
        assert!(lfm2[0].delta_us > 0.0, "{:?}", lfm2[0]);
        let (w, x, out) = (10752.0 * 2048.0 * 2.0, 2048.0 * 2.0, 10752.0 * 2.0);
        let expect = (w + x + out) / 790e3 - 49.0 - 3.0 * out / 790e3;
        assert!(
            (lfm2[0].per_site_us() - expect).abs() < 1e-6,
            "{} vs {expect}",
            lfm2[0].per_site_us()
        );
        assert!(lfm2[0].per_site_us() > 5.0 && lfm2[0].per_site_us() < 8.0);
    }
}
