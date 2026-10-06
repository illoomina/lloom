//! The one emitter: a model's IR is the walk of its own blocks, each
//! emitting what its decode step dispatches (`design/docs/optimizer-design.md`
//! §6.3, §8.1 — emitted, never recovered; `porting-models.md` §21.9's
//! level-0 contract — every block has `execute` and `visit`).
//!
//! A block implements [`Visit`] beside its forward and emits the dispatches
//! that forward runs at a one-token decode step on Metal: the IR's nodes for
//! what the optimizer derives over (the products, the elementwise chain), an
//! opaque node per kernel it does not, the recurrences marked. The walk reads
//! the block as built — which projections were stacked at load, whether a
//! bracket has an injection, what a weight's storage dispatches — so the IR
//! is the model's, not a second description of it kept beside the forward.
//!
//! [`Walker`] is the shared half: the graph being built and the name scope a
//! block's weights are named in (the checkpoint's own path, as the loader
//! read it), so a fusion decision names the site it applies to. Inputs (the
//! token, a cache, a state window) are named whole, by the step.

use std::ops::{Deref, DerefMut};

use lloom_tensor::quantized::QMatMul;
use lloom_tensor::DType;

use super::ir::{Core, Graph, NodeId, Shape};
use super::numeric::Region;

/// A block that can say what its decode step dispatches.
///
/// `In` and `Out` are the IR values the block reads and produces, in the
/// shape its forward takes and returns them: a projection reads one value
/// and produces one; a residual bracket reads the carry and returns the
/// block's input and its mixing weights; a model reads nothing and produces
/// its logits.
pub trait Visit {
    type In;
    type Out;
    fn visit(&self, w: &mut Walker, input: Self::In) -> Self::Out;
}

/// The graph a walk builds, and the scope its weights are named in.
#[derive(Debug, Default)]
pub struct Walker {
    graph: Graph,
    scope: Vec<String>,
    /// Writes the step makes for a later step that nothing in this step
    /// reads (an indexer's key append, a conv history's next window): the
    /// step's outputs beside its result.
    state: Vec<NodeId>,
}

impl Walker {
    pub fn new() -> Walker {
        Walker::default()
    }

    /// Run `f` with `part` appended to the name scope.
    pub fn scoped<T>(&mut self, part: impl AsRef<str>, f: impl FnOnce(&mut Walker) -> T) -> T {
        self.scope.push(part.as_ref().to_string());
        let out = f(self);
        self.scope.pop();
        out
    }

    /// `leaf` named in the current scope: `model.layers.3` + `w1.weight`.
    pub fn path(&self, leaf: &str) -> String {
        let mut s = self.scope.join(".");
        if !s.is_empty() && !leaf.is_empty() {
            s.push('.');
        }
        s.push_str(leaf);
        s
    }

    /// A weight named `leaf` in the current scope.
    pub fn weight(&mut self, leaf: &str, shape: Shape, dtype: DType) -> NodeId {
        let name = self.path(leaf);
        self.graph.weight(&name, shape, dtype)
    }

    /// The product of the weight named `leaf` (`shape` `[n, k]`, `dtype`) and
    /// `x`, on `core`.
    pub fn project(
        &mut self,
        core: Core,
        leaf: &str,
        shape: Shape,
        dtype: DType,
        x: NodeId,
    ) -> NodeId {
        let w = self.weight(leaf, shape, dtype);
        self.graph.gemv_on(core, w, x)
    }

    /// The product of the matrix `q` (named `leaf`) and `x`, on the core its
    /// storage dispatches at a decode step ([`qmat_operand`]).
    pub fn project_qmat(&mut self, leaf: &str, q: &QMatMul, x: NodeId) -> NodeId {
        let (core, shape, dtype) = qmat_operand(q);
        self.project(core, leaf, shape, dtype, x)
    }

    /// `id` is a write the step leaves for a later one and does not read
    /// itself: an output of the step, rooted beside its result so the
    /// derivation keeps it.
    pub fn state_out(&mut self, id: NodeId) {
        self.state.push(id);
    }

    /// Place every node of `ids` in `region`.
    pub fn mark(&mut self, ids: &[NodeId], region: Region) {
        for &id in ids {
            self.graph.mark_region(id, region);
        }
    }

    pub fn finish(self) -> Graph {
        self.graph
    }
}

impl Deref for Walker {
    type Target = Graph;
    fn deref(&self) -> &Graph {
        &self.graph
    }
}

impl DerefMut for Walker {
    fn deref_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }
}

/// The core a [`QMatMul`]'s storage runs a one-row product on (Metal), and
/// the weight's IR shape and dtype: an f16 stem on the mixed GEMV
/// (`QMatMul::gemv_mixed`), a packed weight on the quantized matvec, an f32
/// matrix on the GEMV. A packed weight's dtype is `U8` — its kind is the
/// header's, and only the cost model's byte count reads the dtype.
pub fn qmat_operand(q: &QMatMul) -> (Core, Shape, DType) {
    match q {
        QMatMul::TensorF16(t) => (Core::Mixed, Shape::of(t.dims()), t.dtype()),
        QMatMul::Tensor(t) => (Core::Gemv, Shape::of(t.dims()), t.dtype()),
        QMatMul::QTensor(t) => (Core::Quant, Shape::of(t.shape().dims()), DType::U8),
    }
}

/// The IR of `model`'s decode step: its visit from nothing to its output,
/// which becomes the graph's root.
pub fn walk<M: Visit<In = (), Out = NodeId> + ?Sized>(model: &M) -> Graph {
    let mut w = Walker::new();
    let out = model.visit(&mut w, ());
    w.mark_root(out);
    for id in std::mem::take(&mut w.state) {
        w.mark_root(id);
    }
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Proj {
        n: usize,
        k: usize,
    }

    impl Visit for Proj {
        type In = NodeId;
        type Out = NodeId;
        fn visit(&self, w: &mut Walker, x: NodeId) -> NodeId {
            let y = w.project(
                Core::Gemv,
                "weight",
                Shape::of(&[self.n, self.k]),
                DType::F16,
                x,
            );
            w.silu(y)
        }
    }

    struct Model {
        layers: Vec<Proj>,
    }

    impl Visit for Model {
        type In = ();
        type Out = NodeId;
        fn visit(&self, w: &mut Walker, (): ()) -> NodeId {
            let mut x = w.input("x", Shape::of(&[1, 1, 4]), DType::F16);
            for (l, p) in self.layers.iter().enumerate() {
                x = w.scoped(format!("layers.{l}"), |w| p.visit(w, x));
            }
            x
        }
    }

    /// Weights are named by the scope they were emitted in; the scope pops
    /// after the block; the model's output is the root.
    #[test]
    fn a_walk_names_weights_by_their_scope() {
        let m = Model {
            layers: vec![Proj { n: 4, k: 4 }, Proj { n: 4, k: 4 }],
        };
        let g = walk(&m);
        let names: Vec<&str> = g
            .nodes()
            .iter()
            .filter(|n| n.op == super::super::Op::Weight)
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(names, ["layers.0.weight", "layers.1.weight"]);
        assert_eq!(g.roots().len(), 1);
        assert_eq!(g.dispatches(), 4);
        let input = g.nodes().iter().find(|n| n.op == super::super::Op::Input);
        assert_eq!(input.map(|n| n.name.as_str()), Some("x"));
    }
}
