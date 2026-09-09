//! The rewrite substrate: egglog (`design/docs/optimizer-design.md` §2.13).
//!
//! The IR is lowered to an egglog program — one constructor per op, one
//! `let` per node, the use counts and GEMV row counts as facts — the
//! derivation rules are generated from the fragment catalogue, the e-graph is
//! saturated, and the cheapest term (dispatches, tree-additive) is extracted
//! back into a [`Graph`].
//!
//! A rule fires only when its obligations hold: the GEMV's store has exactly
//! one reader (G3 adjacency — nothing else can observe the unfused value),
//! its row count is 1 (the fused arms are instantiated for the GEMV route
//! only), and the fragment is admitted by the policy (its class within the
//! budget, its name not excluded). Two of the three are discharged by the
//! LOWERING rather than by a rule condition: a value with two or more readers
//! is cut into a leaf its readers cannot look through (`cut_points`), and an
//! inadmissible fragment contributes no rule — so the rewrite that would
//! cross the budget never exists in the e-graph, which is how an unsound
//! reorder "simply never fires" (§4). The row count is the one fact a rule
//! reads.

use std::collections::HashMap;
use std::fmt::Write as _;

use lloom_tensor::{Error, Result};

use crate::nn::fusion::{OperandKind, Semantics};

use super::fragments::Fragment;
use super::ir::{Graph, Node, NodeId, Op};
use super::Policy;

/// The largest opaque arity the program declares a constructor for.
const MAX_OPAQUE_ARITY: usize = 6;

/// What one derivation run produced.
pub struct Saturation {
    /// The program that was run, for the report.
    pub program: String,
    /// The extracted graph.
    pub graph: Graph,
    /// The extractor's cost of the root term (tree-additive, so it counts a
    /// shared subterm once per path and is NOT the dispatch count; the
    /// dispatch count is the graph's census).
    pub cost: u64,
    /// Wall time of the egglog run.
    pub elapsed: std::time::Duration,
}

/// The fragments the policy admits, with why the others were refused.
pub fn admitted(policy: &Policy, catalogue: &[Fragment]) -> (Vec<Fragment>, Vec<(String, String)>) {
    let mut ok = Vec::new();
    let mut refused = Vec::new();
    for f in catalogue {
        if policy.excluded.iter().any(|e| e == f.name) {
            refused.push((
                f.name.to_string(),
                "excluded by the exclusion set".to_string(),
            ));
        } else if !f.class().within(policy.budget) {
            refused.push((
                f.name.to_string(),
                format!("class {} exceeds the budget {}", f.class(), policy.budget),
            ));
        } else {
            ok.push(f.clone());
        }
    }
    (ok, refused)
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The datatype declarations and the derivation rules for `fragments`.
fn prelude(fragments: &[Fragment]) -> String {
    let mut p = String::new();
    // Leaves, views and epilogue markers cost 0; every dispatching constructor
    // costs 1, and a fused constructor costs 1 — so a fusion is cheaper than
    // the pair it replaces by exactly one dispatch, never a tie. The extractor
    // prices a literal child at 1, so a marker carries NO literal: one
    // constructor per fragment, its operand (if any) as an `Expr` child.
    p.push_str("(datatype* (Epi");
    for f in fragments {
        match f.operand {
            OperandKind::None => {
                let _ = write!(p, " (Epi_{} :cost 0)", f.name);
            }
            OperandKind::RowIndexed => {
                let _ = write!(p, " (Epi_{} Expr :cost 0)", f.name);
            }
        }
    }
    p.push_str(")\n (Expr\n");
    p.push_str("  (In String :cost 0)\n  (W String :cost 0)\n");
    p.push_str("  (View String i64 Expr :cost 0)\n");
    p.push_str("  (Gemv Expr Expr :cost 1)\n  (Silu Expr :cost 1)\n");
    p.push_str("  (Mul Expr Expr :cost 1)\n  (Add Expr Expr :cost 1)\n");
    p.push_str("  (Norm Expr Expr :cost 1)\n");
    for arity in 1..=MAX_OPAQUE_ARITY {
        let _ = write!(p, "  (Op{arity} String i64");
        for _ in 0..arity {
            p.push_str(" Expr");
        }
        p.push_str(" :cost 1)\n");
    }
    p.push_str("  (Fused Expr Expr Epi :cost 1)))\n");
    p.push_str("(function rows (Expr) i64 :merge (max old new))\n");
    p.push_str("(rewrite (Add a b) (Add b a))\n");
    p.push_str("(rewrite (Mul a b) (Mul b a))\n");
    for f in fragments {
        let name = f.name;
        match (f.semantics, f.operand) {
            (Semantics::Silu, OperandKind::None) => {
                let _ = writeln!(
                    p,
                    "(rule ((= e (Silu g)) (= g (Gemv w x)) (= (rows g) 1))\n      \
                     ((union e (Fused w x (Epi_{name})))))"
                );
            }
            (Semantics::AddOperand, OperandKind::RowIndexed) => {
                let _ = writeln!(
                    p,
                    "(rule ((= e (Add g r)) (= g (Gemv w x)) (= (rows g) 1))\n      \
                     ((union e (Fused w x (Epi_{name} r)))))"
                );
            }
            // Identity is the unfused arm itself; any other pairing has no
            // rule shape yet and contributes nothing rather than a guess.
            _ => {}
        }
    }
    p
}

/// A value with two or more readers is a fusion boundary: no rule looks
/// through it (G3 adjacency — the store is observable), and it is where the
/// tree-additive extractor would otherwise count the subgraph below it once
/// per path and saturate on a 30-layer step. So the graph is lowered as a
/// forest cut at every such value: consumers read it as a leaf, and it is
/// extracted as its own root. Leaves themselves are never cut — duplicating a
/// weight name costs nothing and keeps the site name on the term.
fn cut_points(graph: &Graph) -> Vec<bool> {
    let uses = graph.use_counts();
    graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(id, node)| uses[id] >= 2 && !matches!(node.op, Op::Input | Op::Weight))
        .collect()
}

/// The roots the program extracts, in emission order: every cut point and
/// every graph root, ascending — a cut point's id is below its readers', so
/// rebuilding in this order finds a segment before anything reads it.
fn segment_roots(graph: &Graph) -> Vec<NodeId> {
    let cut = cut_points(graph);
    let mut roots: Vec<NodeId> = (0..graph.nodes().len()).filter(|&i| cut[i]).collect();
    for &r in graph.roots() {
        if !roots.contains(&r) {
            roots.push(r);
        }
    }
    roots.sort_unstable();
    roots
}

/// The leaf a consumer reads a cut value through.
fn segment_leaf(id: NodeId) -> String {
    format!("#n{id}")
}

/// The `let`s and facts for `graph`.
fn lower(graph: &Graph) -> Result<String> {
    let cut = cut_points(graph);
    let mut p = String::new();
    for (id, node) in graph.nodes().iter().enumerate() {
        // A reader names a cut value by its leaf, not its term.
        let n = |i: NodeId| {
            if cut[i] {
                format!("c{i}")
            } else {
                format!("n{i}")
            }
        };
        let term = match node.op {
            Op::Input => format!("(In {})", quote(&node.name)),
            Op::Weight => format!("(W {})", quote(&node.name)),
            // A view or an opaque carries its source ordinal: two `narrow`s of
            // one value are two different values, and without the ordinal the
            // e-graph would hash-cons them into one.
            Op::View => format!("(View {} {id} {})", quote(&node.name), n(node.inputs[0])),
            Op::Gemv => format!("(Gemv {} {})", n(node.inputs[0]), n(node.inputs[1])),
            Op::Silu => format!("(Silu {})", n(node.inputs[0])),
            Op::Mul => format!("(Mul {} {})", n(node.inputs[0]), n(node.inputs[1])),
            Op::Add => format!("(Add {} {})", n(node.inputs[0]), n(node.inputs[1])),
            Op::RmsNorm => format!("(Norm {} {})", n(node.inputs[0]), n(node.inputs[1])),
            Op::Opaque => {
                let arity = node.inputs.len();
                if arity == 0 || arity > MAX_OPAQUE_ARITY {
                    return Err(Error::Msg(format!(
                        "optimizer: opaque `{}` has {arity} inputs; the program declares \
                         arities 1..={MAX_OPAQUE_ARITY}",
                        node.name
                    )));
                }
                let mut t = format!("(Op{arity} {} {id}", quote(&node.name));
                for &i in &node.inputs {
                    let _ = write!(t, " {}", n(i));
                }
                t.push(')');
                t
            }
            Op::Fused => {
                return Err(Error::Msg(
                    "optimizer: the source graph already carries a fused node".to_string(),
                ))
            }
        };
        let _ = writeln!(p, "(let n{id} {term})");
        if cut[id] {
            let _ = writeln!(p, "(let c{id} (In {}))", quote(&segment_leaf(id)));
        }
        if node.op == Op::Gemv {
            let rows = match graph.gemv_rows(id) {
                Some(super::ir::Dim::Const(m)) => m as i64,
                // A symbolic row count is not known to be 1: the fused arms
                // are not instantiated for it, so the rule must not fire.
                _ => -1,
            };
            let _ = writeln!(p, "(set (rows n{id}) {rows})");
        }
    }
    Ok(p)
}

/// The egglog program for `graph` under `fragments`: the declarations and
/// rules, the graph, `iterations` of saturation, one extraction per root.
pub fn program(graph: &Graph, fragments: &[Fragment], iterations: usize) -> Result<String> {
    let roots = graph.roots();
    if roots.is_empty() {
        return Err(Error::Msg("optimizer: the graph has no root".to_string()));
    }
    let mut program = prelude(fragments);
    program.push_str(&lower(graph)?);
    let _ = writeln!(program, "(run {iterations})");
    for r in segment_roots(graph) {
        let _ = writeln!(program, "(extract n{r})");
    }
    Ok(program)
}

/// Run the derivation: saturate `graph` under `fragments`, extract the
/// cheapest realization of every root, rebuild it as a [`Graph`].
pub fn derive(graph: &Graph, fragments: &[Fragment], iterations: usize) -> Result<Saturation> {
    let program = program(graph, fragments, iterations)?;

    // The extractor reconstructs the term DAG recursively and so does the
    // rebuild below; a 30-layer step nests a few hundred deep, past a test
    // thread's default stack. One thread with room, once per construction.
    let started = std::time::Instant::now();
    let outputs = std::thread::Builder::new()
        .name("lloom-optimizer".to_string())
        .stack_size(256 << 20)
        .spawn({
            let program = program.clone();
            move || {
                let mut egraph = egglog::EGraph::default();
                egraph
                    .parse_and_run_program(None, &program)
                    .map_err(|e| Error::Msg(format!("optimizer: egglog: {e}")))
            }
        })
        .map_err(|e| {
            Error::Msg(format!(
                "optimizer: cannot spawn the derivation thread: {e}"
            ))
        })?
        .join()
        .map_err(|_| Error::Msg("optimizer: the derivation thread panicked".to_string()))??;
    let elapsed = started.elapsed();

    let (rebuilt_graph, cost, extracted) = std::thread::Builder::new()
        .name("lloom-optimizer-rebuild".to_string())
        .stack_size(256 << 20)
        .spawn({
            let graph = graph.clone();
            let fragments = fragments.to_vec();
            move || -> Result<(Graph, u64, usize)> {
                let mut rebuilt = Rebuild::new(&graph, &fragments);
                let segments = segment_roots(&graph);
                let mut cost = 0u64;
                let mut extracted = 0usize;
                for out in &outputs {
                    if let egglog::CommandOutput::ExtractBest(dag, c, id) = out {
                        let Some(&segment) = segments.get(extracted) else {
                            return Err(Error::Msg(
                                "optimizer: more extractions than segment roots".to_string(),
                            ));
                        };
                        // Term ids are per extraction: each `ExtractBest`
                        // carries its own DAG.
                        rebuilt.memo.clear();
                        let node = rebuilt.term(dag, *id)?;
                        rebuilt.segments.insert(segment, node);
                        if graph.roots().contains(&segment) {
                            rebuilt.graph.mark_root(node);
                        }
                        cost = cost.saturating_add(*c);
                        extracted += 1;
                    }
                }
                if extracted != segments.len() {
                    return Err(Error::Msg(format!(
                        "optimizer: extracted {extracted} segments of {}",
                        segments.len()
                    )));
                }
                Ok((rebuilt.graph, cost, extracted))
            }
        })
        .map_err(|e| Error::Msg(format!("optimizer: cannot spawn the rebuild thread: {e}")))?
        .join()
        .map_err(|_| Error::Msg("optimizer: the rebuild thread panicked".to_string()))??;
    if extracted != segment_roots(graph).len() {
        return Err(Error::Msg(format!(
            "optimizer: extracted {extracted} segments of {}",
            segment_roots(graph).len()
        )));
    }
    if cost == u64::MAX {
        return Err(Error::Msg(
            "optimizer: the extraction cost saturated; a segment is too deep for the \
             extractor to rank realizations"
                .to_string(),
        ));
    }
    Ok(Saturation {
        program,
        graph: rebuilt_graph,
        cost,
        elapsed,
    })
}

/// Rebuilds a [`Graph`] from an extracted term DAG, hash-consing on the
/// term id so a value shared in the source is shared in the result.
struct Rebuild<'a> {
    source: &'a Graph,
    fragments: &'a [Fragment],
    graph: Graph,
    /// Source nodes by name, for the leaves.
    inputs: HashMap<&'a str, NodeId>,
    weights: HashMap<&'a str, NodeId>,
    /// Source cut point → its rebuilt node, filled as segments are rebuilt.
    segments: HashMap<NodeId, NodeId>,
    memo: HashMap<egglog::TermId, NodeId>,
}

impl<'a> Rebuild<'a> {
    fn new(source: &'a Graph, fragments: &'a [Fragment]) -> Self {
        let mut inputs = HashMap::new();
        let mut weights = HashMap::new();
        for (id, node) in source.nodes().iter().enumerate() {
            match node.op {
                Op::Input => {
                    inputs.insert(node.name.as_str(), id);
                }
                Op::Weight => {
                    weights.insert(node.name.as_str(), id);
                }
                _ => {}
            }
        }
        Rebuild {
            source,
            fragments,
            graph: Graph::new(),
            inputs,
            weights,
            segments: HashMap::new(),
            memo: HashMap::new(),
        }
    }

    fn string(dag: &egglog::TermDag, id: egglog::TermId) -> Result<String> {
        match dag.get(id) {
            egglog::Term::Lit(egglog::ast::Literal::String(s)) => Ok(s.clone()),
            other => Err(Error::Msg(format!(
                "optimizer: expected a string literal in the extracted term, got {other:?}"
            ))),
        }
    }

    /// The source node an extracted view/opaque names by ordinal.
    fn ordinal(&self, dag: &egglog::TermDag, id: egglog::TermId, name: &str) -> Result<&'a Node> {
        let ordinal = match dag.get(id) {
            egglog::Term::Lit(egglog::ast::Literal::Int(i)) => *i,
            other => {
                return Err(Error::Msg(format!(
                    "optimizer: expected an ordinal in the extracted term, got {other:?}"
                )))
            }
        };
        let node = usize::try_from(ordinal)
            .ok()
            .and_then(|i| self.source.nodes().get(i))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "optimizer: extracted `{name}` names ordinal {ordinal}, not in the source"
                ))
            })?;
        if node.name != name {
            return Err(Error::Msg(format!(
                "optimizer: extracted `{name}` names ordinal {ordinal}, which is `{}` in the source",
                node.name
            )));
        }
        Ok(node)
    }

    fn term(&mut self, dag: &egglog::TermDag, id: egglog::TermId) -> Result<NodeId> {
        if let Some(&n) = self.memo.get(&id) {
            return Ok(n);
        }
        let term_id = id;
        let egglog::Term::App(head, children) = dag.get(id) else {
            return Err(Error::Msg(format!(
                "optimizer: expected an application in the extracted term, got {:?}",
                dag.get(id)
            )));
        };
        let children = children.clone();
        let node = match head.as_str() {
            "In" => {
                let name = Self::string(dag, children[0])?;
                if let Some(id) = name.strip_prefix("#n") {
                    // A cut value: the segment rebuilt before this one.
                    let src: NodeId = id.parse().map_err(|_| {
                        Error::Msg(format!("optimizer: malformed segment leaf `{name}`"))
                    })?;
                    let node = *self.segments.get(&src).ok_or_else(|| {
                        Error::Msg(format!(
                            "optimizer: segment `{name}` read before it was rebuilt"
                        ))
                    })?;
                    self.memo.insert(term_id, node);
                    return Ok(node);
                }
                let src = *self.inputs.get(name.as_str()).ok_or_else(|| {
                    Error::Msg(format!(
                        "optimizer: extracted input `{name}` not in the source"
                    ))
                })?;
                let s: &Node = self.source.node(src);
                self.graph.input(&s.name, s.shape.clone(), s.dtype)
            }
            "W" => {
                let name = Self::string(dag, children[0])?;
                let src = *self.weights.get(name.as_str()).ok_or_else(|| {
                    Error::Msg(format!(
                        "optimizer: extracted weight `{name}` not in the source"
                    ))
                })?;
                let s: &Node = self.source.node(src);
                self.graph.weight(&s.name, s.shape.clone(), s.dtype)
            }
            "View" => {
                let name = Self::string(dag, children[0])?;
                let shape = self.ordinal(dag, children[1], &name)?.shape.clone();
                let x = self.term(dag, children[2])?;
                self.graph.view(&name, x, shape)
            }
            "Gemv" => {
                let w = self.term(dag, children[0])?;
                let x = self.term(dag, children[1])?;
                self.graph.gemv(w, x)
            }
            "Silu" => {
                let x = self.term(dag, children[0])?;
                self.graph.silu(x)
            }
            "Mul" => {
                let a = self.term(dag, children[0])?;
                let b = self.term(dag, children[1])?;
                self.graph.mul(a, b)
            }
            "Add" => {
                let a = self.term(dag, children[0])?;
                let b = self.term(dag, children[1])?;
                self.graph.add(a, b)
            }
            "Norm" => {
                let w = self.term(dag, children[0])?;
                let x = self.term(dag, children[1])?;
                self.graph.rms_norm(w, x)
            }
            "Fused" => {
                let w = self.term(dag, children[0])?;
                let x = self.term(dag, children[1])?;
                let egglog::Term::App(ehead, echildren) = dag.get(children[2]) else {
                    return Err(Error::Msg("optimizer: malformed epilogue term".to_string()));
                };
                let echildren = echildren.clone();
                let name = ehead.strip_prefix("Epi_").ok_or_else(|| {
                    Error::Msg(format!("optimizer: unknown epilogue constructor `{ehead}`"))
                })?;
                let fragment = self
                    .fragments
                    .iter()
                    .find(|f| f.name == name)
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "optimizer: extracted fragment `{name}` not in the catalogue"
                        ))
                    })?
                    .clone();
                let operand = match echildren.first() {
                    Some(&r) => Some(self.term(dag, r)?),
                    None => None,
                };
                self.graph
                    .fused(fragment.name, fragment.class(), w, x, operand)
            }
            op if op.starts_with("Op") => {
                let name = Self::string(dag, children[0])?;
                let s = self.ordinal(dag, children[1], &name)?;
                let (shape, dtype) = (s.shape.clone(), s.dtype);
                let mut inputs = Vec::with_capacity(children.len() - 2);
                for &c in &children[2..] {
                    inputs.push(self.term(dag, c)?);
                }
                self.graph.opaque(&name, &inputs, shape, dtype)
            }
            other => {
                return Err(Error::Msg(format!(
                    "optimizer: unknown constructor `{other}` in the extracted term"
                )))
            }
        };
        self.memo.insert(id, node);
        Ok(node)
    }
}
