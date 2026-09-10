//! The tensor-op IR (`design/docs/optimizer-design.md` §6.3): a node is WHAT
//! computes, thin; the value it produces carries shape, dtype and numerical
//! class. Emitted by the model from its own configuration, never recovered
//! from an execution (§8.1).
//!
//! The graph is a DAG in an arena: a node names its inputs by [`NodeId`], and
//! a value's use count is a property of the graph, not of the node — which is
//! what the G3 adjacency gate (`design/docs/fusion.md` §12.2e) reads.
//!
//! A node also carries the [`Region`] it computes in: straight-line, or one
//! of the step's recurrences (a conv layer's state window, an attention
//! layer's KV cache). The emitter marks them; the plan reads them off the
//! SOURCE graph (`plan::regions`), because a rewrite decision is a fact about
//! where a site sits, and a per-rewrite verdict does not compose across a
//! recurrence the way it does along a straight line (§5a).

use std::collections::BTreeMap;
use std::fmt;

use lloom_tensor::DType;

use super::numeric::{NumericClass, Region};

/// Index of a node in a [`Graph`].
pub type NodeId = usize;

/// One extent of a shape: a number, or a host symbol the graph is optimized
/// under (the Kind-A environment, §3.2 — `B`, `K`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Const(usize),
    Sym(&'static str),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(n) => write!(f, "{n}"),
            Dim::Sym(s) => f.write_str(s),
        }
    }
}

/// A symbolic shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn of(dims: &[usize]) -> Shape {
        Shape(dims.iter().map(|&d| Dim::Const(d)).collect())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn last(&self) -> Option<Dim> {
        self.0.last().copied()
    }

    /// The same shape with the last extent replaced.
    fn with_last(&self, d: Dim) -> Shape {
        let mut v = self.0.clone();
        if let Some(l) = v.last_mut() {
            *l = d;
        }
        Shape(v)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// What a node computes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// A value that enters the graph from outside (the token's hidden state,
    /// a cache). Not a dispatch.
    Input,
    /// A weight. Not a dispatch.
    Weight,
    /// A view of its input — a `narrow`, a `reshape`, a `transpose` that is
    /// contiguous at this shape. Not a dispatch. Its `name` says which.
    View,
    /// `x @ w.t()`. Inputs `[w, x]`; `w` is `[n, k]`, `x` is `[.., rows, k]`.
    /// One dispatch; the GEMV route when `rows == 1`.
    Gemv,
    /// `silu(x)`. One dispatch.
    Silu,
    /// `a * b`. One dispatch.
    Mul,
    /// `a + b`. One dispatch.
    Add,
    /// RMS norm over the last extent. Inputs `[w, x]`. One dispatch.
    RmsNorm,
    /// A dispatch of some other kernel family, named by `name`. Opaque to
    /// every rewrite; counted by the census.
    Opaque,
    /// A GEMV with an [`Elementwise`](crate::nn::fusion::Elementwise) block
    /// riding its store. Inputs `[w, x]`, plus the block's operand if it binds
    /// one. `name` is the fragment's name. One dispatch.
    Fused,
}

impl Op {
    /// Whether this node is one GPU dispatch.
    pub fn dispatches(&self) -> bool {
        !matches!(self, Op::Input | Op::Weight | Op::View)
    }

    /// Whether this node is a standalone elementwise dispatch — the family
    /// P1's derivations fuse.
    pub fn is_elementwise(&self) -> bool {
        matches!(self, Op::Silu | Op::Mul | Op::Add)
    }
}

/// A node: the op, its inputs, the value it produces.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
    /// The site name for an `Input`/`Weight` (the checkpoint tensor's name for
    /// a weight, so a fusion decision names the site it applies to); the
    /// kernel family for an `Opaque`/`View`; the fragment name for a `Fused`.
    /// Empty otherwise.
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub numeric: NumericClass,
    /// Where the node computes: straight-line unless the emitter marked it
    /// as part of a recurrence ([`Graph::mark_region`]).
    pub region: Region,
}

/// The kernel family a node's dispatch is counted under — the census key.
pub fn family(node: &Node) -> Option<String> {
    Some(match &node.op {
        Op::Input | Op::Weight | Op::View => return None,
        Op::Gemv => "gemv".to_string(),
        Op::Silu => "silu".to_string(),
        Op::Mul => "bmul".to_string(),
        Op::Add => "badd".to_string(),
        Op::RmsNorm => "rmsnorm".to_string(),
        Op::Opaque => node.name.clone(),
        Op::Fused => format!("gemv_{}", node.name),
    })
}

/// A per-family dispatch count.
pub type Census = BTreeMap<String, u64>;

/// The IR of one pass: nodes in an arena, roots named.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    roots: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn mark_root(&mut self, id: NodeId) {
        self.roots.push(id);
    }

    /// Place `id` in `region`. The emitter calls this for every node of a
    /// recurrence: the state it reads, the write that carries it to the next
    /// token, and the reduction over it.
    pub fn mark_region(&mut self, id: NodeId, region: Region) {
        self.nodes[id].region = region;
    }

    /// The stateful regions and their nodes, in region order.
    pub fn regions(&self) -> BTreeMap<Region, Vec<NodeId>> {
        let mut out: BTreeMap<Region, Vec<NodeId>> = BTreeMap::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if node.region.is_stateful() {
                out.entry(node.region).or_default().push(id);
            }
        }
        out
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn input(&mut self, name: &str, shape: Shape, dtype: DType) -> NodeId {
        self.push(Node {
            op: Op::Input,
            inputs: vec![],
            name: name.to_string(),
            shape,
            dtype,
            numeric: NumericClass::BitExact,
            region: Region::Straight,
        })
    }

    pub fn weight(&mut self, name: &str, shape: Shape, dtype: DType) -> NodeId {
        self.push(Node {
            op: Op::Weight,
            inputs: vec![],
            name: name.to_string(),
            shape,
            dtype,
            numeric: NumericClass::BitExact,
            region: Region::Straight,
        })
    }

    /// A view of `x` at `shape`. No dispatch.
    pub fn view(&mut self, what: &str, x: NodeId, shape: Shape) -> NodeId {
        let dtype = self.nodes[x].dtype;
        let numeric = self.nodes[x].numeric;
        self.push(Node {
            op: Op::View,
            inputs: vec![x],
            name: what.to_string(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    /// `x @ w.t()`: the output takes `x`'s shape with the last extent `w`'s
    /// row count.
    pub fn gemv(&mut self, w: NodeId, x: NodeId) -> NodeId {
        let n = self.nodes[w].shape.0[0];
        let shape = self.nodes[x].shape.with_last(n);
        let dtype = self.nodes[x].dtype;
        let numeric = self.nodes[x].numeric.join(self.nodes[w].numeric);
        self.push(Node {
            op: Op::Gemv,
            inputs: vec![w, x],
            name: String::new(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    fn unary(&mut self, op: Op, x: NodeId) -> NodeId {
        let shape = self.nodes[x].shape.clone();
        let dtype = self.nodes[x].dtype;
        let numeric = self.nodes[x].numeric;
        self.push(Node {
            op,
            inputs: vec![x],
            name: String::new(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    fn binary(&mut self, op: Op, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.nodes[a].shape.clone();
        let dtype = self.nodes[a].dtype;
        let numeric = self.nodes[a].numeric.join(self.nodes[b].numeric);
        self.push(Node {
            op,
            inputs: vec![a, b],
            name: String::new(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    pub fn silu(&mut self, x: NodeId) -> NodeId {
        self.unary(Op::Silu, x)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(Op::Mul, a, b)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.binary(Op::Add, a, b)
    }

    pub fn rms_norm(&mut self, w: NodeId, x: NodeId) -> NodeId {
        let shape = self.nodes[x].shape.clone();
        let dtype = self.nodes[x].dtype;
        let numeric = self.nodes[x].numeric;
        self.push(Node {
            op: Op::RmsNorm,
            inputs: vec![w, x],
            name: String::new(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    /// One dispatch of kernel family `family` over `inputs`, producing a value
    /// of `shape`.
    pub fn opaque(
        &mut self,
        family: &str,
        inputs: &[NodeId],
        shape: Shape,
        dtype: DType,
    ) -> NodeId {
        let numeric = inputs.iter().fold(NumericClass::BitExact, |c, &i| {
            c.join(self.nodes[i].numeric)
        });
        self.push(Node {
            op: Op::Opaque,
            inputs: inputs.to_vec(),
            name: family.to_string(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    /// A GEMV with fragment `fragment` riding its store. `operand` is the
    /// fragment's second buffer, if it binds one.
    pub fn fused(
        &mut self,
        fragment: &str,
        class: NumericClass,
        w: NodeId,
        x: NodeId,
        operand: Option<NodeId>,
    ) -> NodeId {
        let n = self.nodes[w].shape.0[0];
        let shape = self.nodes[x].shape.with_last(n);
        let dtype = self.nodes[x].dtype;
        let mut inputs = vec![w, x];
        let mut numeric = self.nodes[x].numeric.join(class);
        if let Some(r) = operand {
            inputs.push(r);
            numeric = numeric.join(self.nodes[r].numeric);
        }
        self.push(Node {
            op: Op::Fused,
            inputs,
            name: fragment.to_string(),
            shape,
            dtype,
            numeric,
            region: Region::Straight,
        })
    }

    /// How many nodes read each value. A view counts as a use of what it
    /// views — the G3 adjacency gate asks whether anything else can observe
    /// the producer's store, and a view can.
    pub fn use_counts(&self) -> Vec<usize> {
        let mut uses = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            for &i in &node.inputs {
                uses[i] += 1;
            }
        }
        for &r in &self.roots {
            uses[r] += 1;
        }
        uses
    }

    /// The GEMV's row count (`m`), the Kind-A parameter the fused arms are
    /// instantiated for at `m == 1`.
    pub fn gemv_rows(&self, id: NodeId) -> Option<Dim> {
        let node = &self.nodes[id];
        if !matches!(node.op, Op::Gemv | Op::Fused) {
            return None;
        }
        let x = &self.nodes[node.inputs[1]];
        let r = x.shape.rank();
        if r < 2 {
            Some(Dim::Const(1))
        } else {
            Some(x.shape.0[r - 2])
        }
    }

    /// Dispatches per kernel family.
    pub fn census(&self) -> Census {
        let mut census = Census::new();
        for node in &self.nodes {
            if let Some(f) = family(node) {
                *census.entry(f).or_insert(0) += 1;
            }
        }
        census
    }

    /// Total dispatches.
    pub fn dispatches(&self) -> u64 {
        self.nodes.iter().filter(|n| n.op.dispatches()).count() as u64
    }

    /// The weakest numerical class any root lands in.
    pub fn numeric(&self) -> NumericClass {
        self.roots.iter().fold(NumericClass::BitExact, |c, &r| {
            c.join(self.nodes[r].numeric)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_dtypes_and_uses_follow_the_ops() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w = g.weight("w", Shape::of(&[4, 8]), DType::F16);
        let y = g.gemv(w, x);
        assert_eq!(g.node(y).shape, Shape::of(&[1, 1, 4]));
        assert_eq!(g.gemv_rows(y), Some(Dim::Const(1)));
        let s = g.silu(y);
        let v = g.view("narrow", y, Shape::of(&[1, 1, 2]));
        g.mark_root(s);
        g.mark_root(v);
        let uses = g.use_counts();
        assert_eq!(uses[y], 2, "a view is a use");
        assert_eq!(uses[s], 1, "a root is a use");
        assert_eq!(g.dispatches(), 2);
        assert_eq!(g.census().get("gemv"), Some(&1));
        assert_eq!(g.census().get("silu"), Some(&1));
        assert!(
            !g.census().contains_key("narrow"),
            "a view is not a dispatch"
        );
    }

    #[test]
    fn a_region_is_marked_by_the_emitter_and_read_off_the_graph() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let st = g.input("state", Shape::of(&[1, 8, 3]), DType::F16);
        let w = g.weight("w", Shape::of(&[8, 8]), DType::F16);
        let y = g.gemv(w, x);
        let ring = g.opaque("copy2d", &[y, st], Shape::of(&[1, 8, 3]), DType::F16);
        assert_eq!(
            g.node(ring).region,
            Region::Straight,
            "unmarked is straight"
        );
        g.mark_region(st, Region::ConvRing(4));
        g.mark_region(ring, Region::ConvRing(4));
        let regions = g.regions();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[&Region::ConvRing(4)], vec![st, ring]);
        assert_eq!(g.node(y).region, Region::Straight);
    }

    #[test]
    fn a_fused_node_carries_its_fragments_class() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::of(&[1, 1, 8]), DType::F16);
        let w = g.weight("w", Shape::of(&[4, 8]), DType::F16);
        let f = g.fused("silu", NumericClass::Rounding, w, x, None);
        assert_eq!(g.node(f).numeric, NumericClass::Rounding);
        assert_eq!(g.census().get("gemv_silu"), Some(&1));
        g.mark_root(f);
        assert_eq!(g.numeric(), NumericClass::Rounding);
    }
}
