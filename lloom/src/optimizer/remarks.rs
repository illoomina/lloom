//! Optimization remarks — the `-Rpass-missed` analogue
//! (`design/docs/optimizer-design.md` §10i): every producer→consumer seam a
//! derived step leaves between two dispatches, why it is unfused, and what
//! it costs in the census, ranked.
//!
//! A seam is an edge between two dispatches: the consumer reads the
//! producer's store, directly or through views. Each is given ONE reason,
//! the first gate it fails in a fixed order — the extent (G2), the arity
//! (G1), the adjacency (G3) of `design/docs/fusion.md` §12.2e, then whether
//! the kernel at the seam takes a composed block at all, then whether the
//! catalogue holds one of the consumer's semantics. The facts are
//! [`kinds`](super::kinds)'s; the order is the one a fusion would have to
//! clear, so the reason named is the one standing in the way.
//!
//! The cost is counts: the dispatches a fusion of the class would remove,
//! the barriers the sequential hazard model places between the pair
//! ([`cost::step_cost`]), and the intermediate's round trip in bytes. No
//! time is claimed: at B = 1 the decode step sits at the weight sweep's
//! roofline, where neither a dispatch nor a barrier is a byte (§10e-iii).

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use crate::nn::fusion::{OperandKind, Semantics};

use super::cost::{self, CostModel};
use super::fragments::{catalogue, Fragment};
use super::ir::{family, Graph, Node, NodeId, Op};
use super::kinds::{self, Facts, Role};
use super::Derivation;

/// Which end of a product the seam is at.
pub use super::fragments::Side;

/// Why a seam is unfused.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Why {
    /// G2: the consumer reduces over the extent the producer partitions
    /// across threadgroups (store side: an epilogue cannot reduce over `N`),
    /// or a reduction feeds a product's load (a prologue cannot import a
    /// reduction one dispatch computes for all threadgroups, §12.2a).
    CrossThreadgroup(Side),
    /// G1: the consumer reads an operand that is not indexed by the
    /// producer's output position.
    BinaryOperand,
    /// G3: the producer's store is read by another dispatch too (or leaves
    /// the step), so the value must be stored whatever rides it.
    MultiUse,
    /// An elementwise producer feeding a product's load: as a prologue it
    /// would re-run in every threadgroup (×`n_tgp`, §12.2b).
    PrologueRecompute,
    /// No kernel at the seam takes a composed block — collecting it means a
    /// new fused body (or a hook on `kernel`).
    NewBody(String),
    /// The gates pass and `kernel` takes a block at this end, but the
    /// catalogue holds none of the consumer's semantics for it.
    NoFragment(String),
    /// The catalogue holds a block for the seam and the derivation did not
    /// place it: the policy refused `fragment` (or the rule did not match).
    Refused { fragment: String, reason: String },
    /// A family [`kinds`] does not declare.
    Undeclared(String),
}

impl fmt::Display for Why {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Why::CrossThreadgroup(Side::Store) => {
                f.write_str("cross-threadgroup reduction (store)")
            }
            Why::CrossThreadgroup(Side::Load) => f.write_str("cross-threadgroup reduction (load)"),
            Why::BinaryOperand => f.write_str("binary operand"),
            Why::MultiUse => f.write_str("multi-use store"),
            Why::PrologueRecompute => f.write_str("prologue recompute ×n_tgp"),
            Why::NewBody(k) => write!(f, "needs a new body ({k})"),
            Why::NoFragment(k) => write!(f, "no fragment kind ({k})"),
            Why::Refused { fragment, reason } => write!(f, "refused {fragment}: {reason}"),
            Why::Undeclared(k) => write!(f, "undeclared kernel {k}"),
        }
    }
}

/// One class of seams: a producer family, a consumer family and a reason.
#[derive(Debug, Clone, PartialEq)]
pub struct Remark {
    pub producer: String,
    pub consumer: String,
    pub why: Why,
    /// Edges of the class.
    pub edges: u64,
    /// Dispatches a fusion of the class would remove: one per distinct
    /// consumer on the store side and between non-products, one per
    /// distinct producer on the load side.
    pub dispatches: u64,
    /// Edges of the class with a barrier between the pair in the hazard
    /// model's emission order.
    pub barriers: u64,
    /// The intermediate's round trip (written by the producer, read by the
    /// consumer), summed over the edges.
    pub bytes: f64,
    /// Edges with either end in a recurrence.
    pub stateful: u64,
    /// A weight at the first edge, to find the site.
    pub site: String,
}

impl Remark {
    /// The census cost the ranking orders by.
    pub fn census(&self) -> u64 {
        self.dispatches + self.barriers
    }
}

/// Every seam of a derived step, classified and ranked.
#[derive(Debug, Clone)]
pub struct Remarks {
    /// Most costly first.
    pub rows: Vec<Remark>,
    /// The derived step's dispatches.
    pub dispatches: u64,
    /// The fused sites the derivation placed.
    pub fused: u64,
    /// The hazard model's barriers over the derived step.
    pub barriers: u64,
}

impl Remarks {
    pub fn edges(&self) -> u64 {
        self.rows.iter().map(|r| r.edges).sum()
    }

    /// Edges and dispatches per reason, the reason's display as the key.
    pub fn by_reason(&self) -> BTreeMap<String, (u64, u64)> {
        let mut out: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for r in &self.rows {
            let key = match &r.why {
                Why::Refused { fragment, .. } => format!("refused {fragment}"),
                w => w.to_string(),
            };
            let e = out.entry(key).or_default();
            e.0 += r.edges;
            e.1 += r.dispatches;
        }
        out
    }
}

impl fmt::Display for Remarks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "remarks: {} dispatches, {} fused sites, {} barriers (model); {} seams in {} classes",
            self.dispatches,
            self.fused,
            self.barriers,
            self.edges(),
            self.rows.len()
        )?;
        writeln!(
            f,
            "rank dispatches barriers edges stateful      bytes  seam  why  [site]"
        )?;
        for (i, r) in self.rows.iter().enumerate() {
            writeln!(
                f,
                "{:4} {:10} {:8} {:5} {:8} {:10.0}  {}>{}  {}  [{}]",
                i + 1,
                r.dispatches,
                r.barriers,
                r.edges,
                r.stateful,
                r.bytes,
                r.producer,
                r.consumer,
                r.why,
                r.site
            )?;
        }
        Ok(())
    }
}

/// The dispatch a value is read from, looking through views; `None` for a
/// leaf.
fn producer_of(graph: &Graph, mut id: NodeId) -> Option<NodeId> {
    loop {
        let n = graph.node(id);
        match n.op {
            Op::View => id = n.inputs[0],
            Op::Input | Op::Weight => return None,
            _ => return Some(id),
        }
    }
}

/// Every dispatch's distinct dispatching readers (through views), and
/// whether its value leaves the step (a root, or a view of one).
fn readers(graph: &Graph) -> (Vec<Vec<NodeId>>, Vec<bool>) {
    let n = graph.nodes().len();
    let mut out: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for (c, node) in graph.nodes().iter().enumerate() {
        if !node.op.dispatches() {
            continue;
        }
        for &i in &node.inputs {
            if let Some(p) = producer_of(graph, i) {
                if !out[p].contains(&c) {
                    out[p].push(c);
                }
            }
        }
    }
    let mut external = vec![false; n];
    for &r in graph.roots() {
        if let Some(p) = producer_of(graph, r) {
            external[p] = true;
        }
    }
    (out, external)
}

/// The catalogue semantics an elementwise consumer computes, if any.
fn semantics(node: &Node) -> Option<(Semantics, bool)> {
    Some(match node.op {
        Op::Silu => (Semantics::Silu, false),
        Op::Sigmoid => (Semantics::Sigmoid, false),
        Op::Add => (Semantics::AddOperand, true),
        Op::Mul => (Semantics::MulOperand, true),
        // The SwiGLU's activation and product as one kernel (a kernel
        // fragment): on a product's store it is the SiLU block on the gate's
        // (and the product on the up's).
        Op::Opaque if node.name == "silu_mul" => (Semantics::Silu, false),
        _ => return None,
    })
}

/// Whether every operand of the elementwise `c` other than `p`'s value is
/// indexed by `c`'s output position: the same element count as `c`'s
/// output (which a view of `p`'s store, a third of it, may be).
fn row_indexed(graph: &Graph, p: NodeId, c: NodeId) -> bool {
    let want = cost::elems(&graph.node(c).shape);
    graph
        .node(c)
        .inputs
        .iter()
        .all(|&i| producer_of(graph, i) == Some(p) || cost::elems(&graph.node(i).shape) == want)
}

/// The first weight a dispatch reads, for naming its site.
fn site(graph: &Graph, id: NodeId) -> Option<String> {
    graph
        .node(id)
        .inputs
        .iter()
        .find(|&&i| graph.node(i).op == Op::Weight)
        .map(|&i| graph.node(i).name.clone())
}

/// The reason the seam `p → c` is unfused, and which dispatch a fusion of
/// it would remove.
#[allow(clippy::too_many_arguments)]
fn classify(
    graph: &Graph,
    p: NodeId,
    c: NodeId,
    pf: Facts,
    cf: Facts,
    others: bool,
    catalogue: &[Fragment],
    refused: &[(String, String)],
) -> (Why, NodeId) {
    let (pn, cn) = (graph.node(p), graph.node(c));
    let fam = |n: &Node| family(n).unwrap_or_default();
    // The load side: the producer would ride the consumer product's load.
    if cf.role == Role::Product {
        let why = match pf.role {
            // A reduction rides a product's load only where the catalogue
            // has a prologue computing it there and the product's tile admits
            // it (G2 passed): then it is the policy's, or the prologue's own
            // refusal, that leaves it.
            Role::Reduction => match load_prologue(graph, p, c, catalogue) {
                Some(f) => Why::Refused {
                    fragment: f.name.to_string(),
                    reason: f
                        .refusal
                        .map(str::to_string)
                        .or_else(|| {
                            refused
                                .iter()
                                .find(|(n, _)| n == f.name)
                                .map(|(_, r)| r.clone())
                        })
                        .unwrap_or_else(|| "the rule did not place it".to_string()),
                },
                None => Why::CrossThreadgroup(Side::Load),
            },
            Role::Elementwise if cf.load_hook => Why::PrologueRecompute,
            Role::Elementwise => Why::NewBody(format!("no load hook on {}", fam(cn))),
            Role::Product => Why::NewBody("two products".to_string()),
            _ => Why::NewBody(format!("{} into a product", role_word(pf.role))),
        };
        return (why, p);
    }
    // The store side: the consumer would ride the producer product's store.
    if pf.role == Role::Product {
        let why = match cf.role {
            Role::Reduction => Why::CrossThreadgroup(Side::Store),
            Role::Elementwise => {
                if !row_indexed(graph, p, c) {
                    Why::BinaryOperand
                } else if others {
                    Why::MultiUse
                } else if pn.op == Op::Fused {
                    Why::NewBody(format!("the store carries {}", pn.name))
                } else if !pf.store_hook
                    || (pn.core == super::ir::Core::Quant
                        && graph.node(pn.inputs[0]).dtype != lloom_tensor::DType::U8)
                {
                    Why::NewBody(format!("no store hook on {}", fam(pn)))
                } else {
                    store_fragment(pn, cn, catalogue, refused)
                }
            }
            r => Why::NewBody(format!("a {} after a product", role_word(r))),
        };
        return (why, c);
    }
    // Neither end is a product. A store hook takes an elementwise block, and
    // a rotation where the storing thread holds the pair (`rms_norm`'s
    // `StoreRope`, `fusion.md` §12.2p).
    let at_store = match cf.role {
        Role::Elementwise => row_indexed(graph, p, c),
        Role::Pairwise => true,
        _ => false,
    };
    let why = if pf.store_hook && at_store {
        if others {
            Why::MultiUse
        } else {
            Why::NoFragment(fam(pn))
        }
    } else {
        Why::NewBody(format!("{} to {}", role_word(pf.role), role_word(cf.role)))
    };
    (why, c)
}

/// The catalogue's prologue that computes producer `p` in consumer `c`'s
/// load where `c`'s tile admits it: the inline norm (an `RmsNorm` into a
/// GEMV whose simdgroup walks K), the grouped norm (`hc_norm` into the mixed
/// GEMV, any tile). A prologue that only imports the reduction removes no
/// dispatch and is not this seam's.
fn load_prologue<'c>(graph: &Graph, p: NodeId, c: NodeId, catalogue: &'c [Fragment]) -> Option<&'c Fragment> {
    let (pn, cn) = (graph.node(p), graph.node(c));
    let semantics = match (&pn.op, pn.name.as_str()) {
        (Op::RmsNorm, _) if graph.gemv_walks_k(c) => Semantics::RmsNorm,
        (Op::Opaque, "hc_norm") => Semantics::GroupRmsNorm,
        _ => return None,
    };
    // A fused consumer takes the prologue only beside an epilogue the
    // kernel instantiates with it.
    catalogue.iter().find(|f| {
        f.side == Side::Load
            && f.semantics == semantics
            && f.core == cn.core
            && (cn.op != Op::Fused || f.composes_with.contains(&cn.name.as_str()))
    })
}

/// The store-side verdict once the gates pass: the catalogue's block for
/// the consumer's semantics on the producer's core, refused or never there.
fn store_fragment(
    pn: &Node,
    cn: &Node,
    catalogue: &[Fragment],
    refused: &[(String, String)],
) -> Why {
    let kernel = family(pn).unwrap_or_default();
    let Some((sem, binary)) = semantics(cn) else {
        return Why::NoFragment(kernel);
    };
    let operand = if binary {
        OperandKind::RowIndexed
    } else {
        OperandKind::None
    };
    match catalogue
        .iter()
        .find(|f| {
            f.side == Side::Store
                && f.core == pn.core
                && f.semantics == sem
                && (f.operand == operand
                    || (operand == OperandKind::RowIndexed && f.operand == OperandKind::Into))
        })
    {
        Some(f) => Why::Refused {
            fragment: f.name.to_string(),
            reason: refused
                .iter()
                .find(|(n, _)| n == f.name)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| "the rule did not place it".to_string()),
        },
        None => Why::NoFragment(kernel),
    }
}

fn role_word(r: Role) -> &'static str {
    match r {
        Role::Product => "product",
        Role::Elementwise => "elementwise",
        Role::Reduction => "reduction",
        Role::Movement => "movement",
        Role::Recurrence => "recurrence",
        Role::Pairwise => "pairwise",
        Role::Broadcast => "broadcast",
    }
}

/// The remarks over `d`'s derived step, the barriers placed by `model`.
pub fn remarks(d: &Derivation, model: &CostModel) -> Remarks {
    remarks_of(&d.graph, &d.refused, model, &catalogue())
}

/// [`remarks`] over any graph, with the policy's refusals and a catalogue.
pub fn remarks_of(
    graph: &Graph,
    refused: &[(String, String)],
    model: &CostModel,
    catalogue: &[Fragment],
) -> Remarks {
    let (readers, external) = readers(graph);
    let step = cost::step_cost(model, graph);
    let barriers: HashSet<(NodeId, NodeId)> =
        step.barriers.iter().map(|b| (b.before, b.after)).collect();
    struct Acc {
        edges: u64,
        removed: HashSet<NodeId>,
        barriers: u64,
        bytes: f64,
        stateful: u64,
        site: String,
    }
    let mut classes: BTreeMap<(String, String, Why), Acc> = BTreeMap::new();
    for (p, pn) in graph.nodes().iter().enumerate() {
        if !pn.op.dispatches() {
            continue;
        }
        let pfam = family(pn).unwrap_or_default();
        for &c in &readers[p] {
            let cn = graph.node(c);
            let cfam = family(cn).unwrap_or_default();
            let others = external[p] || readers[p].len() > 1;
            let (why, removed) = match (kinds::of(pn), kinds::of(cn)) {
                (Some(pf), Some(cf)) => classify(graph, p, c, pf, cf, others, catalogue, refused),
                (None, _) => (Why::Undeclared(pfam.clone()), c),
                (_, None) => (Why::Undeclared(cfam.clone()), c),
            };
            let acc = classes
                .entry((pfam.clone(), cfam, why))
                .or_insert_with(|| Acc {
                    edges: 0,
                    removed: HashSet::new(),
                    barriers: 0,
                    bytes: 0.0,
                    stateful: 0,
                    site: site(graph, p)
                        .or_else(|| site(graph, c))
                        .unwrap_or_default(),
                });
            acc.edges += 1;
            acc.removed.insert(removed);
            acc.barriers += u64::from(barriers.contains(&(p, c)));
            acc.bytes += 2.0 * cost::bytes(&pn.shape, pn.dtype);
            acc.stateful += u64::from(pn.region.is_stateful() || cn.region.is_stateful());
        }
    }
    let mut rows: Vec<Remark> = classes
        .into_iter()
        .map(|((producer, consumer, why), a)| Remark {
            producer,
            consumer,
            why,
            edges: a.edges,
            dispatches: a.removed.len() as u64,
            barriers: a.barriers,
            bytes: a.bytes,
            stateful: a.stateful,
            site: a.site,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.census()
            .cmp(&a.census())
            .then_with(|| b.bytes.total_cmp(&a.bytes))
            .then_with(|| (&a.producer, &a.consumer).cmp(&(&b.producer, &b.consumer)))
    });
    Remarks {
        rows,
        dispatches: graph.dispatches(),
        fused: graph.nodes().iter().filter(|n| n.op == Op::Fused).count() as u64,
        barriers: step.barriers.len() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{derive, Core, Policy, Shape};
    use lloom_tensor::DType;

    fn why_of<'a>(r: &'a Remarks, p: &str, c: &str) -> Vec<&'a Why> {
        r.rows
            .iter()
            .filter(|x| x.producer == p && x.consumer == c)
            .map(|x| &x.why)
            .collect()
    }

    /// One seam per reason, on a graph built for it, each named by the gate
    /// it fails — and the ranking puts the class with a barrier first.
    #[test]
    fn every_reason_is_reached_by_the_gate_it_names() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F32);
        let nw = g.weight("norm", Shape::of(&[8]), DType::F32);
        let wq = g.weight("wq", Shape::of(&[8, 8]), DType::F32);
        let wm = g.weight("wm", Shape::of(&[8, 8]), DType::F32);
        let wg = g.weight("wg", Shape::of(&[8, 8]), DType::F32);
        // A norm into a product's load: cross-threadgroup (load).
        let xn = g.rms_norm(nw, x);
        // A dense quantized product (an f32 `QTensor`) into a SiLU: the store
        // blocks ride the packed kinds' cores only.
        let q = g.gemv_on(Core::Quant, wq, xn);
        let qs = g.silu(q);
        // An elementwise value into the mixed product's load (no load hook),
        // and the mixed product into a norm: cross-threadgroup (store).
        let m = g.gemv_on(Core::Mixed, wm, qs);
        let mn = g.rms_norm(nw, m);
        // An f32 GEMV whose store has two readers: multi-use.
        let gg = g.gemv(wg, mn);
        let a = g.silu(gg);
        let b = g.sigmoid(gg);
        // A product feeding a broadcast operand: binary operand.
        let s = g.input("s", Shape::of(&[1, 1, 1]), DType::F32);
        let gb = g.gemv(wg, a);
        let bm = g.mul(gb, s);
        let out = g.add(bm, b);
        g.mark_root(out);
        let r = remarks_of(
            &g,
            &[],
            &CostModel::default(),
            &crate::optimizer::catalogue(),
        );
        assert_eq!(
            why_of(&r, "rmsnorm", "mul_mv"),
            [&Why::CrossThreadgroup(Side::Load)]
        );
        assert_eq!(
            why_of(&r, "mul_mv", "silu"),
            [&Why::NewBody("no store hook on mul_mv".to_string())]
        );
        assert_eq!(
            why_of(&r, "gemv_mixed", "rmsnorm"),
            [&Why::CrossThreadgroup(Side::Store)]
        );
        assert_eq!(why_of(&r, "gemv", "silu"), [&Why::MultiUse]);
        assert_eq!(why_of(&r, "gemv", "sigmoid"), [&Why::MultiUse]);
        assert_eq!(why_of(&r, "gemv", "bmul"), [&Why::BinaryOperand]);
        assert_eq!(
            why_of(&r, "silu", "gemv_mixed"),
            [&Why::NewBody("no load hook on gemv_mixed".to_string())]
        );
        assert_eq!(why_of(&r, "silu", "gemv"), [&Why::PrologueRecompute]);
        assert_eq!(
            why_of(&r, "rmsnorm", "gemv"),
            [&Why::CrossThreadgroup(Side::Load)]
        );
        // Two elementwise ops: no kernel composes them.
        assert_eq!(
            why_of(&r, "bmul", "badd"),
            [&Why::NewBody("elementwise to elementwise".to_string())]
        );
        // Every edge is classified exactly once.
        let edges: u64 = r.rows.iter().map(|x| x.edges).sum();
        assert_eq!(edges, 11);
        assert_eq!(r.dispatches, g.dispatches());
        // The ranking: census cost descending.
        assert!(r.rows.windows(2).all(|w| w[0].census() >= w[1].census()));
        // A norm's store takes a rotation (`StoreRope`); the catalogue has
        // no such block.
        let mut h = Graph::new();
        let x = h.input("x", Shape::of(&[1, 4, 8]), DType::F32);
        let gain = h.weight("g", Shape::of(&[8]), DType::F32);
        let n = h.rms_norm(gain, x);
        let cs = h.input("cos", Shape::of(&[1, 1, 4]), DType::F32);
        let rope = h.opaque("rope", &[n, cs], Shape::of(&[1, 4, 8]), DType::F32);
        h.mark_root(rope);
        let rh = remarks_of(
            &h,
            &[],
            &CostModel::default(),
            &crate::optimizer::catalogue(),
        );
        assert_eq!(
            why_of(&rh, "rmsnorm", "rope"),
            [&Why::NoFragment("rmsnorm".to_string())]
        );
        // An elementwise consumer of a view of a product's store, its other
        // operand the view's extent: indexed by its own output, not G1.
        let mut v = Graph::new();
        let x = v.input("x", Shape::of(&[1, 1, 8]), DType::F32);
        let w = v.weight("w", Shape::of(&[24, 8]), DType::F32);
        let y = v.gemv(w, x);
        let third = v.view("narrow", y, Shape::of(&[1, 1, 8]));
        let other = v.input("other", Shape::of(&[1, 1, 8]), DType::F32);
        let m = v.mul(third, other);
        v.mark_root(m);
        let rv = remarks_of(
            &v,
            &[],
            &CostModel::default(),
            &crate::optimizer::catalogue(),
        );
        assert!(
            matches!(
                why_of(&rv, "gemv", "bmul")[..],
                [Why::Refused { fragment, .. }] if fragment == "mul"
            ),
            "{rv}"
        );
    }

    /// A seam the catalogue covers and the policy refused is named with the
    /// policy's reason; with the block admitted it is fused and gone.
    #[test]
    fn a_refused_block_names_the_policy_and_an_admitted_one_leaves_no_seam() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w = g.weight("w1", Shape::of(&[16, 8]), DType::F16);
        let y = g.gemv(w, x);
        let s = g.silu(y);
        g.mark_root(s);
        let refused = Policy {
            excluded: vec!["silu".to_string()],
            ..Default::default()
        };
        let d = derive(&g, &refused).unwrap();
        let r = remarks(&d, &CostModel::default());
        assert_eq!(
            why_of(&r, "gemv", "silu"),
            [&Why::Refused {
                fragment: "silu".to_string(),
                reason: "excluded by the exclusion set".to_string()
            }]
        );
        assert_eq!(r.rows[0].dispatches, 1);
        assert_eq!(r.rows[0].barriers, 1, "the pair is adjacent and dependent");
        let d = derive(&g, &Policy::default()).unwrap();
        let r = remarks(&d, &CostModel::default());
        assert!(r.rows.is_empty(), "{r}");
        assert_eq!(r.fused, 1);
    }
}
