//! The IR diff (`design/docs/profiles-and-memory-design.md` §11 item 1,
//! `porting-models.md` §21.10 row 1): two walks of a forward compared up to a
//! dependency-preserving permutation of their nodes, and the first edge that
//! differs named — before any GPU runs.
//!
//! A node is identified by what it computes and from what: its op, name,
//! kernel core, scalar, shape, dtype and region, and the identities of its
//! inputs in order (a Merkle hash over the DAG). Two graphs whose multisets of
//! identities agree are the same computation in some order that keeps every
//! edge; a reorder that breaks an edge changes the identity of the node whose
//! input moved, and of everything downstream of it. The mismatch reported
//! first is the one nearest the inputs, which is where the change is.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use super::ir::{family, Graph, Node, Op};

/// One node's identity and the dispatch depth it sits at.
fn identities(graph: &Graph) -> (Vec<u64>, Vec<u32>) {
    let nodes = graph.nodes();
    let mut ids = vec![0u64; nodes.len()];
    let mut level = vec![0u32; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        let mut h = DefaultHasher::new();
        local(node).hash(&mut h);
        for &j in &node.inputs {
            ids[j].hash(&mut h);
        }
        ids[i] = h.finish();
        let deepest = node.inputs.iter().map(|&j| level[j]).max().unwrap_or(0);
        level[i] = deepest + u32::from(node.op.dispatches());
    }
    (ids, level)
}

/// What a node is, without its inputs.
fn local(node: &Node) -> String {
    format!(
        "{:?}|{}|{:?}|{}|{}|{:?}|{}",
        node.op,
        node.name,
        node.core,
        node.scalar.map(f32::to_bits).unwrap_or(0),
        node.shape,
        node.dtype,
        node.region
    )
}

/// A node's one-line description: its family (or op), name, shape, dtype,
/// region, and its inputs' families.
pub fn describe(graph: &Graph, id: usize) -> String {
    let node = graph.node(id);
    let what = family(node).unwrap_or_else(|| format!("{:?}", node.op).to_lowercase());
    let mut s = what;
    if !node.name.is_empty() && !matches!(node.op, Op::Opaque) {
        s.push(' ');
        s.push_str(&node.name);
    }
    if let Some(c) = node.scalar {
        s.push_str(&format!(" c={c}"));
    }
    s.push_str(&format!(" {} {:?}", node.shape, node.dtype));
    if node.region.is_stateful() {
        s.push_str(&format!(" @{}", node.region));
    }
    if !node.inputs.is_empty() {
        let ins: Vec<String> = node
            .inputs
            .iter()
            .map(|&j| {
                let n = graph.node(j);
                match n.op {
                    Op::Input | Op::Weight => n.name.clone(),
                    _ => family(n).unwrap_or_else(|| format!("{:?}", n.op).to_lowercase()),
                }
            })
            .collect();
        s.push_str(&format!(" <- {}", ins.join(", ")));
    }
    s
}

/// A node one side has more of than the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmatched {
    /// The dispatch depth it sits at (1 = reads only leaves).
    pub level: u32,
    /// How many more of it this side has.
    pub count: usize,
    pub describe: String,
}

/// What [`diff`] found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IrDiff {
    /// Nodes the left graph has and the right does not, nearest the inputs first.
    pub only_left: Vec<Unmatched>,
    pub only_right: Vec<Unmatched>,
    /// Whether the two graphs' roots are the same computations.
    pub roots_match: bool,
}

impl IrDiff {
    pub fn is_empty(&self) -> bool {
        self.only_left.is_empty() && self.only_right.is_empty() && self.roots_match
    }

    /// The first differing edge: the unmatched node nearest the inputs on
    /// each side.
    pub fn first(&self) -> (Option<&Unmatched>, Option<&Unmatched>) {
        (self.only_left.first(), self.only_right.first())
    }
}

impl fmt::Display for IrDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("identical");
        }
        let (l, r) = self.first();
        write!(
            f,
            "{} node kinds only left, {} only right{}",
            self.only_left.len(),
            self.only_right.len(),
            if self.roots_match {
                ""
            } else {
                ", roots differ"
            }
        )?;
        if let Some(l) = l {
            write!(f, "; first left: L{} ×{} {}", l.level, l.count, l.describe)?;
        }
        if let Some(r) = r {
            write!(f, "; first right: L{} ×{} {}", r.level, r.count, r.describe)?;
        }
        Ok(())
    }
}

/// Compare `left` and `right` up to a dependency-preserving permutation.
pub fn diff(left: &Graph, right: &Graph) -> IrDiff {
    let (li, ll) = identities(left);
    let (ri, rl) = identities(right);
    let count = |ids: &[u64]| {
        let mut m: BTreeMap<u64, usize> = BTreeMap::new();
        for &i in ids {
            *m.entry(i).or_insert(0) += 1;
        }
        m
    };
    let (lc, rc) = (count(&li), count(&ri));
    let unmatched = |g: &Graph,
                     ids: &[u64],
                     levels: &[u32],
                     mine: &BTreeMap<u64, usize>,
                     theirs: &BTreeMap<u64, usize>| {
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<Unmatched> = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            let have = mine[&id];
            let other = theirs.get(&id).copied().unwrap_or(0);
            if have > other && seen.insert(id) {
                out.push(Unmatched {
                    level: levels[i],
                    count: have - other,
                    describe: describe(g, i),
                });
            }
        }
        out.sort_by(|a, b| {
            a.level
                .cmp(&b.level)
                .then_with(|| a.describe.cmp(&b.describe))
        });
        out
    };
    let mut lroots: Vec<u64> = left.roots().iter().map(|&r| li[r]).collect();
    let mut rroots: Vec<u64> = right.roots().iter().map(|&r| ri[r]).collect();
    lroots.sort_unstable();
    rroots.sort_unstable();
    IrDiff {
        only_left: unmatched(left, &li, &ll, &lc, &rc),
        only_right: unmatched(right, &ri, &rl, &rc, &lc),
        roots_match: lroots == rroots,
    }
}

/// The graph as text in a canonical order — by dispatch depth, then by
/// description — one line per node, inputs named by their line: two graphs
/// equal up to a dependency-preserving permutation list identically, and a
/// change shows as a line-level diff of the listing.
pub fn listing(graph: &Graph) -> String {
    let (ids, levels) = identities(graph);
    let mut order: Vec<usize> = (0..graph.nodes().len()).collect();
    let desc: Vec<String> = order.iter().map(|&i| describe(graph, i)).collect();
    order.sort_by(|&a, &b| {
        levels[a]
            .cmp(&levels[b])
            .then_with(|| desc[a].cmp(&desc[b]))
            .then_with(|| ids[a].cmp(&ids[b]))
    });
    let mut line = vec![0usize; order.len()];
    for (pos, &i) in order.iter().enumerate() {
        line[i] = pos;
    }
    let mut out = String::new();
    for (pos, &i) in order.iter().enumerate() {
        let node = graph.node(i);
        let ins: Vec<String> = node.inputs.iter().map(|&j| line[j].to_string()).collect();
        let root = if graph.roots().contains(&i) {
            " root"
        } else {
            ""
        };
        out.push_str(&format!(
            "{pos} L{} {} [{}]{root}\n",
            levels[i],
            desc[i],
            ins.join(" ")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Region, Shape};
    use lloom_tensor::DType;

    fn mlp(order_swapped: bool, region: bool) -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w1 = g.weight("w1", Shape::of(&[16, 8]), DType::F16);
        let w3 = g.weight("w3", Shape::of(&[16, 8]), DType::F16);
        // The two independent GEMVs emitted in either order.
        let (a, b) = if order_swapped {
            let b = g.gemv(w3, x);
            let a = g.gemv(w1, x);
            (a, b)
        } else {
            let a = g.gemv(w1, x);
            let b = g.gemv(w3, x);
            (a, b)
        };
        let s = g.silu(a);
        let m = g.mul(s, b);
        if region {
            g.mark_region(m, Region::Kv(0));
        }
        g.mark_root(m);
        g
    }

    /// A permutation that keeps every edge is no difference; the listing is
    /// the same text.
    #[test]
    fn independent_nodes_may_reorder() {
        let (a, b) = (mlp(false, false), mlp(true, false));
        assert!(diff(&a, &b).is_empty(), "{}", diff(&a, &b));
        assert_eq!(listing(&a), listing(&b));
    }

    /// A broken edge is named: the SiLU moved from `w1`'s product to `w3`'s,
    /// and the first unmatched node on each side is that SiLU, at depth 2.
    #[test]
    fn a_moved_edge_is_named_at_its_depth() {
        let a = mlp(false, false);
        let mut b = Graph::new();
        let x = b.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w1 = b.weight("w1", Shape::of(&[16, 8]), DType::F16);
        let w3 = b.weight("w3", Shape::of(&[16, 8]), DType::F16);
        let p1 = b.gemv(w1, x);
        let p3 = b.gemv(w3, x);
        let s = b.silu(p3);
        let m = b.mul(s, p1);
        b.mark_root(m);
        let d = diff(&a, &b);
        assert!(!d.is_empty());
        let (l, r) = d.first();
        let (l, r) = (l.unwrap(), r.unwrap());
        assert_eq!((l.level, r.level), (2, 2), "{d}");
        assert!(l.describe.starts_with("silu"), "{}", l.describe);
        assert!(r.describe.starts_with("silu"), "{}", r.describe);
        assert_ne!(listing(&a), listing(&b));
    }

    /// A region mark is part of the identity: the same ops with the product
    /// marked stateful differ.
    #[test]
    fn a_region_is_part_of_the_identity() {
        let d = diff(&mlp(false, false), &mlp(false, true));
        assert_eq!(d.only_left.len(), 1, "{d}");
        assert!(!d.roots_match);
    }

    /// The counts are a multiset: one side with an extra copy of a node is
    /// a difference, though every node of the other has a match.
    #[test]
    fn an_extra_copy_is_a_difference() {
        let a = mlp(false, false);
        let mut b = mlp(false, false);
        let x = 0;
        let w1 = 1;
        let extra = b.gemv(w1, x);
        b.mark_root(extra);
        let d = diff(&a, &b);
        assert!(d.only_left.is_empty());
        assert_eq!(d.only_right.len(), 1);
        assert_eq!(d.only_right[0].count, 1);
    }
}
