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

use crate::nn::fusion::{OperandKind, Semantics, TileNeed};
use lloom_tensor::DType;

use super::fragments::{Fragment, KernelForm, KernelFragment, Side};
use super::ir::{Core, Graph, Node, NodeId, Op};
use super::Policy;

/// Every core, in the order the program declares their constructors.
const CORES: [Core; 3] = [Core::Gemv, Core::Mixed, Core::Quant];

/// A scale constant as the program carries it: the f32's bits, exactly.
fn scale_bits(c: f32) -> i64 {
    i64::from(c.to_bits())
}

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
    /// Each kernel fragment's placements in the extracted graph.
    pub kernels: std::collections::BTreeMap<String, usize>,
    /// Prologue sites the extraction took and the rebuild reverted: the
    /// producer they would have absorbed has another reader that did not
    /// take it, so it stays a dispatch and the prologue would remove nothing.
    pub reverted: std::collections::BTreeMap<String, usize>,
}

/// The fragments the policy admits, with why the others were refused.
pub fn admitted(policy: &Policy, catalogue: &[Fragment]) -> (Vec<Fragment>, Vec<(String, String)>) {
    let mut ok = Vec::new();
    let mut refused = Vec::new();
    for f in catalogue {
        if let Some(why) = f.refusal {
            refused.push((f.name.to_string(), why.to_string()));
        } else if let Some((_, why)) = policy.unavailable.iter().find(|(n, _)| n == f.name) {
            refused.push((f.name.to_string(), why.clone()));
        } else if policy.excluded.iter().any(|e| e == f.name) {
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

/// The kernel fragments the policy admits, with why the others were refused.
pub fn admitted_kernels(
    policy: &Policy,
    kernels: &[KernelFragment],
) -> (Vec<KernelFragment>, Vec<(String, String)>) {
    let mut ok = Vec::new();
    let mut refused = Vec::new();
    for k in kernels {
        if let Some((_, why)) = policy.unavailable.iter().find(|(n, _)| n == k.name) {
            refused.push((k.name.to_string(), why.clone()));
        } else if policy.excluded.iter().any(|e| e == k.name) {
            refused.push((k.name.to_string(), "excluded by the exclusion set".to_string()));
        } else if !k.class.within(policy.budget) {
            refused.push((
                k.name.to_string(),
                format!("class {} exceeds the budget {}", k.class, policy.budget),
            ));
        } else {
            ok.push(k.clone());
        }
    }
    (ok, refused)
}

/// A dtype as the program's facts carry it.
fn dtype_code(d: DType) -> i64 {
    match d {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        _ => 0,
    }
}

/// Whether the catalogue entry is a placeable prologue: on the load, a norm
/// the rules know how to move.
fn placeable_prologue(f: &Fragment) -> bool {
    f.side == Side::Load && f.refusal.is_none() && f.semantics == Semantics::RmsNorm
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

/// The datatype declarations and the derivation rules for `fragments` and
/// `kernels`.
fn prelude(fragments: &[Fragment], kernels: &[KernelFragment], sites: &[(String, Vec<String>)]) -> String {
    let epilogues: Vec<&Fragment> = fragments.iter().filter(|f| f.side == Side::Store).collect();
    // A block restricted to some sites rides a product whose weight carries
    // the site fact; the rule reads it.
    let at = |name: &str, w: &str| {
        if sites.iter().any(|(f, _)| f == name) {
            format!(" (= (ok_{name} {w}) 1)")
        } else {
            String::new()
        }
    };
    let prologues: Vec<&Fragment> = fragments.iter().filter(|f| placeable_prologue(f)).collect();
    let mut p = String::new();
    // Leaves, views and block markers cost 0; every dispatching constructor
    // costs 2, and a fused constructor costs 2 — so a fusion is cheaper than
    // the pair it replaces by one dispatch's 2, never a tie. A kernel
    // fragment costs 3: still cheaper than its pair (4), and dearer than a
    // product block removing the same dispatch, so where both apply the
    // block on the product the step runs anyway is taken. The extractor
    // prices a literal child at 1, so a marker carries NO literal: one
    // constructor per fragment, its operand (if any) as an `Expr` child.
    // A scaling block carries its constant as a literal; it gets a second
    // marker, `EpiS_`, used only when a `Scale` was absorbed — whose own
    // literal priced the unfused side the same — so the fusion stays cheaper
    // than what it replaces either way.
    // A prologue marker names the block a product computes in its load,
    // carrying the block's weight; a product with one is `GemvP` / `FusedP`
    // (the GEMV core's), which no epilogue rule matches — only the
    // compositions the kernel instantiates are written for it.
    p.push_str("(datatype* ");
    if !prologues.is_empty() {
        p.push_str("(Pro");
        for f in &prologues {
            let _ = write!(p, " (Pro_{} Expr :cost 0)", f.name);
        }
        p.push_str(")\n ");
    }
    p.push_str("(Epi");
    for f in &epilogues {
        match f.operand {
            OperandKind::None => {
                let _ = write!(p, " (Epi_{} :cost 0)", f.name);
                if f.scales {
                    let _ = write!(p, " (EpiS_{} i64 :cost 0)", f.name);
                }
            }
            OperandKind::RowIndexed | OperandKind::Weight => {
                let _ = write!(p, " (Epi_{} Expr :cost 0)", f.name);
            }
        }
    }
    p.push_str(")\n (Expr\n");
    p.push_str("  (In String :cost 0)\n  (W String :cost 0)\n");
    p.push_str("  (View String i64 Expr :cost 0)\n");
    for core in CORES {
        let _ = writeln!(p, "  (Gemv{} Expr Expr :cost 2)", core.tag());
    }
    p.push_str("  (Silu Expr :cost 2)\n  (Sigmoid Expr :cost 2)\n");
    p.push_str("  (Scale i64 Expr :cost 2)\n");
    p.push_str("  (Mul Expr Expr :cost 2)\n  (Add Expr Expr :cost 2)\n");
    p.push_str("  (Norm Expr Expr :cost 2)\n");
    if !prologues.is_empty() {
        p.push_str("  (GemvP Expr Pro Expr :cost 2)\n");
        p.push_str("  (FusedP Expr Pro Expr Epi :cost 2)\n");
    }
    // A kernel fragment: the pair as the one kernel, or a cast its consumer
    // reads through (a marker, no dispatch).
    for k in kernels {
        match k.form {
            KernelForm::SiluMul => {
                let _ = writeln!(p, "  (K_{} Expr Expr :cost 3)", k.name);
            }
            KernelForm::LoadCast { .. } => {
                let _ = writeln!(p, "  (L_{} Expr :cost 0)", k.name);
            }
        }
    }
    for arity in 1..=MAX_OPAQUE_ARITY {
        let _ = write!(p, "  (Op{arity} String i64");
        for _ in 0..arity {
            p.push_str(" Expr");
        }
        p.push_str(" :cost 2)\n");
    }
    for (i, core) in CORES.iter().enumerate() {
        let close = if i + 1 == CORES.len() { "))" } else { "" };
        let _ = writeln!(p, "  (Fused{} Expr Expr Epi :cost 2){close}", core.tag());
    }
    p.push_str("(function rows (Expr) i64 :merge (max old new))\n");
    // The facts the prologue and kernel rules read: whether the product's
    // tile walks all of K in one simdgroup (1), a value's dtype and its
    // element count (-1 when symbolic).
    p.push_str("(function kwalk (Expr) i64 :merge (max old new))\n");
    p.push_str("(function dt (Expr) i64 :merge (max old new))\n");
    p.push_str("(function sz (Expr) i64 :merge (max old new))\n");
    for (f, _) in sites {
        let _ = writeln!(p, "(function ok_{f} (Expr) i64 :merge (max old new))");
    }
    for f in &prologues {
        // A norm into a product whose simdgroup walks K: the product
        // computes it in its load.
        if f.needs != TileNeed::SimdgroupKWalk || f.core != Core::Gemv {
            continue;
        }
        let pro = format!("(Pro_{} nw)", f.name);
        let _ = writeln!(
            p,
            "(rule ((= g (Gemv w n)) (= n (Norm nw x)) (= (kwalk g) 1))\n      \
             ((union g (GemvP w {pro} x))))"
        );
        // Through a view that keeps every element (a reshape, the last row
        // at one row): the norm of the viewed value is the view of the
        // normalized one, so the product normalizes the viewed input.
        let _ = writeln!(
            p,
            "(rule ((= g (Gemv w v)) (= v (View vn o n)) (= n (Norm nw x)) (= (kwalk g) 1) \
             (= z (sz v)) (= z (sz n)))\n      ((union g (GemvP w {pro} (View vn o x)))))"
        );
        // The epilogues the kernel instantiates beside it.
        for e in epilogues.iter().filter(|e| e.core == Core::Gemv) {
            if !f.composes_with.contains(&e.name) {
                continue;
            }
            match (e.semantics, e.operand) {
                (Semantics::Silu, OperandKind::None) => {
                    let _ = writeln!(
                        p,
                        "(rule ((= e (Silu g)) (= g (GemvP w pr x)))\n      \
                         ((union e (FusedP w pr x (Epi_{})))))",
                        e.name
                    );
                }
                (Semantics::SwigluGate, OperandKind::Weight) => {
                    let _ = writeln!(
                        p,
                        "(rule ((= e (Mul s u)) (= s (Silu g)) (= g (GemvP w1 pr x)) \
                         (= u (GemvP w3 pr x)))\n      ((union e (FusedP w3 pr x (Epi_{} w1)))))",
                        e.name
                    );
                }
                _ => {}
            }
        }
    }
    for k in kernels {
        match k.form {
            KernelForm::SiluMul => {
                for &d in k.dtypes {
                    let _ = writeln!(
                        p,
                        "(rule ((= e (Mul s b)) (= s (Silu a)) (= (dt a) {c}) (= (dt b) {c}) \
                         (= z (sz a)) (= z (sz b)))\n      ((union e (K_{name} a b))))",
                        c = dtype_code(d),
                        name = k.name
                    );
                }
            }
            KernelForm::LoadCast { into, operand } => {
                for arity in (operand + 1)..=MAX_OPAQUE_ARITY {
                    let vars: Vec<String> = (0..arity).map(|i| format!("v{i}")).collect();
                    let mut with = vars.clone();
                    with[operand] = format!("(L_{} y)", k.name);
                    for &d in k.dtypes {
                        let _ = writeln!(
                            p,
                            "(rule ((= e (Op{arity} {q} o {vars})) (= v{operand} (Op1 \"cast\" oc y)) \
                             (= (dt y) {c}))\n      ((union e (Op{arity} {q} o {with}))))",
                            q = quote(into),
                            vars = vars.join(" "),
                            with = with.join(" "),
                            c = dtype_code(d),
                        );
                    }
                }
            }
        }
    }
    p.push_str("(rewrite (Add a b) (Add b a))\n");
    p.push_str("(rewrite (Mul a b) (Mul b a))\n");
    // An epilogue rides its core's store; the rule is read off the block's
    // semantics and operand, the same for every core (the constructors carry
    // the core's tag).
    for f in &epilogues {
        let (name, tag) = (f.name, f.core.tag());
        match (f.semantics, f.operand) {
            (Semantics::Silu | Semantics::Sigmoid, OperandKind::None) => {
                let act = if f.semantics == Semantics::Silu { "Silu" } else { "Sigmoid" };
                let ok = at(name, "w");
                let _ = writeln!(
                    p,
                    "(rule ((= e ({act} g)) (= g (Gemv{tag} w x)) (= (rows g) 1){ok})\n      \
                     ((union e (Fused{tag} w x (Epi_{name})))))"
                );
                // A scaling block absorbs the constant scale between them.
                if f.scales {
                    let _ = writeln!(
                        p,
                        "(rule ((= e ({act} s)) (= s (Scale c g)) (= g (Gemv{tag} w x)) (= (rows g) 1){ok})\n      \
                         ((union e (Fused{tag} w x (EpiS_{name} c)))))"
                    );
                }
            }
            (Semantics::AddOperand, OperandKind::RowIndexed) => {
                let _ = writeln!(
                    p,
                    "(rule ((= e (Add g r)) (= g (Gemv{tag} w x)) (= (rows g) 1){ok})\n      \
                     ((union e (Fused{tag} w x (Epi_{name} r)))))",
                    ok = at(name, "w")
                );
            }
            // The product of a store with a row-indexed operand: on the
            // SwiGLU `mul(silu(w1·x), w3·x)` the bare product child is `w3`'s
            // (the gate side is `Silu(..)`, or a fused block once its own rule
            // has fired), so the operand is the gate and the block rides
            // `w3`. `Mul`'s commutativity rewrite lets either order match.
            (Semantics::MulOperand, OperandKind::RowIndexed) => {
                let _ = writeln!(
                    p,
                    "(rule ((= e (Mul g r)) (= g (Gemv{tag} w x)) (= (rows g) 1){ok})\n      \
                     ((union e (Fused{tag} w x (Epi_{name} r)))))",
                    ok = at(name, "w")
                );
            }
            // The two-weight SwiGLU: the product of a SiLU'd product and a
            // bare one over ONE input becomes the bare product with the
            // other's weight as its operand. Both products are single-use by
            // the cut, and the SiLU side matches whether it is still
            // `Silu(..)` or already fused — one e-class.
            (Semantics::SwigluGate, OperandKind::Weight) => {
                let _ = writeln!(
                    p,
                    "(rule ((= e (Mul s u)) (= s (Silu g)) (= g (Gemv{tag} w1 x)) (= u (Gemv{tag} w3 x)) \
                     (= (rows u) 1){ok})\n      ((union e (Fused{tag} w3 x (Epi_{name} w1)))))",
                    ok = at(name, "w3")
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
///
/// One exception: a norm every reader of which is a GEMV-core product is not
/// cut. Its readers may each compute it in their load (a prologue), which
/// observes nothing of its store; the rebuild keeps one norm node for the
/// readers that do not, and reverts the prologues of a norm that survives.
fn cut_points(graph: &Graph) -> Vec<bool> {
    let uses = graph.use_counts();
    let mut gemv_readers = vec![0usize; graph.nodes().len()];
    for node in graph.nodes() {
        if node.op == Op::Gemv && node.core == Core::Gemv {
            gemv_readers[node.inputs[1]] += 1;
        }
    }
    graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(id, node)| {
            let recomputable = node.op == Op::RmsNorm
                && gemv_readers[id] == uses[id]
                && !graph.roots().contains(&id);
            uses[id] >= 2 && !matches!(node.op, Op::Input | Op::Weight) && !recomputable
        })
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

/// The `let`s and facts for `graph`; `sites` the site-restricted blocks.
fn lower(graph: &Graph, sites: &[(String, Vec<String>)]) -> Result<String> {
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
            Op::Gemv => format!(
                "(Gemv{} {} {})",
                node.core.tag(),
                n(node.inputs[0]),
                n(node.inputs[1])
            ),
            Op::Silu => format!("(Silu {})", n(node.inputs[0])),
            Op::Sigmoid => format!("(Sigmoid {})", n(node.inputs[0])),
            Op::Scale => format!(
                "(Scale {} {})",
                scale_bits(node.scalar.unwrap_or(1.0)),
                n(node.inputs[0])
            ),
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
        if node.op == Op::Weight {
            for (f, suffixes) in sites {
                if suffixes.iter().any(|x| node.name.ends_with(x.as_str())) {
                    let _ = writeln!(p, "(set (ok_{f} n{id}) 1)");
                }
            }
        }
        if cut[id] {
            let _ = writeln!(p, "(let c{id} (In {}))", quote(&segment_leaf(id)));
        }
        let size = node
            .shape
            .0
            .iter()
            .try_fold(1i64, |acc, d| match d {
                super::ir::Dim::Const(c) => Some(acc * *c as i64),
                _ => None,
            })
            .unwrap_or(-1);
        for name in [Some(format!("n{id}")), cut[id].then(|| format!("c{id}"))]
            .into_iter()
            .flatten()
        {
            let _ = writeln!(p, "(set (dt {name}) {})", dtype_code(node.dtype));
            let _ = writeln!(p, "(set (sz {name}) {size})");
        }
        if node.op == Op::Gemv {
            let rows = match graph.gemv_rows(id) {
                Some(super::ir::Dim::Const(m)) => m as i64,
                // A symbolic row count is not known to be 1: the fused arms
                // are not instantiated for it, so the rule must not fire.
                _ => -1,
            };
            let _ = writeln!(p, "(set (rows n{id}) {rows})");
            // The tile fact: the dispatcher's own selection for this weight's
            // shape, at one row, on the GEMV core, at a dtype the inline norm
            // is instantiated for.
            if graph.gemv_walks_k(id) {
                let _ = writeln!(p, "(set (kwalk n{id}) 1)");
            }
        }
    }
    Ok(p)
}

/// The egglog program for `graph` under `fragments`: the declarations and
/// rules, the graph, `iterations` of saturation, one extraction per root.
pub fn program(
    graph: &Graph,
    fragments: &[Fragment],
    kernels: &[KernelFragment],
    sites: &[(String, Vec<String>)],
    iterations: usize,
) -> Result<String> {
    let roots = graph.roots();
    if roots.is_empty() {
        return Err(Error::Msg("optimizer: the graph has no root".to_string()));
    }
    let mut program = prelude(fragments, kernels, sites);
    program.push_str(&lower(graph, sites)?);
    let _ = writeln!(program, "(run {iterations})");
    for r in segment_roots(graph) {
        let _ = writeln!(program, "(extract n{r})");
    }
    Ok(program)
}

/// Run the derivation: saturate `graph` under `fragments`, extract the
/// cheapest realization of every root, rebuild it as a [`Graph`].
pub fn derive(
    graph: &Graph,
    fragments: &[Fragment],
    kernels: &[KernelFragment],
    sites: &[(String, Vec<String>)],
    iterations: usize,
) -> Result<Saturation> {
    // One restriction per block: a policy that names a block twice allows
    // the union of its sites.
    let mut merged: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (f, suffixes) in sites {
        let at = merged.entry(f.clone()).or_default();
        for x in suffixes {
            if !at.contains(x) {
                at.push(x.clone());
            }
        }
    }
    let sites: Vec<(String, Vec<String>)> = merged.into_iter().collect();
    let program = program(graph, fragments, kernels, &sites, iterations)?;

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

    type Rebuilt = (Graph, u64, usize, Placements, Placements);
    let (rebuilt_graph, cost, extracted, placed, reverted) = std::thread::Builder::new()
        .name("lloom-optimizer-rebuild".to_string())
        .stack_size(256 << 20)
        .spawn({
            let graph = graph.clone();
            let fragments = fragments.to_vec();
            let kernels = kernels.to_vec();
            move || -> Result<Rebuilt> {
                let mut rebuilt = Rebuild::new(&graph, &fragments, &kernels);
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
                let reverted = rebuilt.revert_partial_prologues();
                Ok((rebuilt.graph, cost, extracted, rebuilt.placed, reverted))
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
        kernels: placed,
        reverted,
    })
}

/// Placements by fragment name.
type Placements = std::collections::BTreeMap<String, usize>;

/// The core a constructor's suffix names.
fn core_of(tag: &str) -> Result<Core> {
    CORES
        .into_iter()
        .find(|c| c.tag() == tag)
        .ok_or_else(|| Error::Msg(format!("optimizer: unknown core tag `{tag}`")))
}

/// Rebuilds a [`Graph`] from an extracted term DAG, hash-consing on the
/// term id so a value shared in the source is shared in the result. A leaf,
/// a view and an opaque kernel name their source node, and keep its
/// recurrence mark; the products and the elementwise ops do not, and come
/// back straight-line (`plan::regions` reads the source for that reason).
struct Rebuild<'a> {
    source: &'a Graph,
    fragments: &'a [Fragment],
    kernels: &'a [KernelFragment],
    graph: Graph,
    /// Norm nodes by (weight name, input value): a norm left as a dispatch
    /// is one node however many segments read it.
    norms: HashMap<(String, String), NodeId>,
    /// Each kernel fragment's placements.
    placed: Placements,
    /// Source nodes by name, for the leaves.
    inputs: HashMap<&'a str, NodeId>,
    weights: HashMap<&'a str, NodeId>,
    /// Source cut point → its rebuilt node, filled as segments are rebuilt.
    segments: HashMap<NodeId, NodeId>,
    memo: HashMap<egglog::TermId, NodeId>,
}

impl<'a> Rebuild<'a> {
    fn new(source: &'a Graph, fragments: &'a [Fragment], kernels: &'a [KernelFragment]) -> Self {
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
            kernels,
            graph: Graph::new(),
            norms: HashMap::new(),
            placed: Placements::new(),
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

    /// A scale constant from its literal bits.
    fn scale(dag: &egglog::TermDag, id: egglog::TermId) -> Result<f32> {
        match dag.get(id) {
            egglog::Term::Lit(egglog::ast::Literal::Int(i)) => u32::try_from(*i)
                .map(f32::from_bits)
                .map_err(|_| Error::Msg(format!("optimizer: scale literal {i} is not f32 bits"))),
            other => Err(Error::Msg(format!(
                "optimizer: expected a scale literal in the extracted term, got {other:?}"
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

    /// A prologue marker: the fragment it names and its weight.
    fn prologue(&mut self, dag: &egglog::TermDag, id: egglog::TermId) -> Result<(Fragment, NodeId)> {
        let egglog::Term::App(head, children) = dag.get(id) else {
            return Err(Error::Msg("optimizer: malformed prologue term".to_string()));
        };
        let children = children.clone();
        let name = head
            .strip_prefix("Pro_")
            .ok_or_else(|| Error::Msg(format!("optimizer: unknown prologue constructor `{head}`")))?;
        let fragment = self
            .fragments
            .iter()
            .find(|f| f.name == name && f.side == Side::Load)
            .ok_or_else(|| {
                Error::Msg(format!("optimizer: extracted prologue `{name}` not in the catalogue"))
            })?
            .clone();
        let pw = self.term(dag, children[0])?;
        Ok((fragment, pw))
    }

    /// A value's identity across segments: a leaf is rebuilt once per
    /// segment that reads it, so it is named; any other value is one node.
    fn value_key(graph: &Graph, id: NodeId) -> String {
        let n = graph.node(id);
        match n.op {
            Op::Input | Op::Weight => format!("leaf:{}", n.name),
            _ => format!("node:{id}"),
        }
    }

    /// A kernel fragment by name.
    fn kernel(&self, name: &str) -> Result<&'a KernelFragment> {
        self.kernels
            .iter()
            .find(|k| k.name == name)
            .ok_or_else(|| Error::Msg(format!("optimizer: extracted kernel `{name}` not in the catalogue")))
    }

    /// Revert every prologue whose producer survived as a dispatch: a norm
    /// one reader still reads is launched anyway, so the readers that compute
    /// it in their load remove nothing and only round differently. The graph
    /// is re-emitted in order, the norm placed at its first reader, so every
    /// node still follows its inputs. Returns the reverted sites by prologue.
    fn revert_partial_prologues(&mut self) -> Placements {
        let old = &self.graph;
        let key = |w: NodeId, x: NodeId| -> (String, String) {
            (old.node(w).name.clone(), Self::value_key(old, x))
        };
        let surviving: std::collections::HashSet<(String, String)> = old
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| n.op == Op::RmsNorm)
            .map(|(_, n)| key(n.inputs[0], n.inputs[1]))
            .collect();
        let targets: std::collections::HashSet<NodeId> = old
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(id, n)| {
                let pw = n.prologue_weight()?;
                surviving.contains(&key(pw, n.inputs[1])).then_some(id)
            })
            .collect();
        let mut reverted = Placements::new();
        if targets.is_empty() {
            return reverted;
        }
        let mut out = Graph::new();
        let mut map: Vec<NodeId> = Vec::with_capacity(old.nodes().len());
        let mut made: HashMap<(String, String), NodeId> = HashMap::new();
        for (id, n) in old.nodes().iter().enumerate() {
            let inputs: Vec<NodeId> = n.inputs.iter().map(|&i| map[i]).collect();
            let new = if n.op == Op::RmsNorm {
                let k = key(n.inputs[0], n.inputs[1]);
                match made.get(&k) {
                    Some(&m) => m,
                    None => {
                        let mut node = n.clone();
                        node.inputs = inputs;
                        let m = out.push(node);
                        made.insert(k, m);
                        m
                    }
                }
            } else if targets.contains(&id) {
                let pw = n.prologue_weight().expect("a target carries a prologue");
                let k = key(pw, n.inputs[1]);
                let norm = match made.get(&k) {
                    Some(&m) => m,
                    None => {
                        let m = out.rms_norm(map[pw], inputs[1]);
                        made.insert(k, m);
                        m
                    }
                };
                *reverted.entry(n.prologue.clone().unwrap_or_default()).or_insert(0) += 1;
                let mut node = n.clone();
                node.inputs = inputs;
                node.inputs.pop();
                node.inputs[1] = norm;
                node.prologue = None;
                if node.name.is_empty() {
                    node.op = Op::Gemv;
                }
                let epilogue = self
                    .fragments
                    .iter()
                    .find(|f| f.name == node.name && f.core == node.core)
                    .map(Fragment::class)
                    .unwrap_or(super::numeric::NumericClass::BitExact);
                node.numeric = node
                    .inputs
                    .iter()
                    .map(|&i| out.node(i).numeric)
                    .fold(epilogue, |a, b| a.join(b));
                out.push(node)
            } else {
                let mut node = n.clone();
                node.inputs = inputs;
                out.push(node)
            };
            map.push(new);
        }
        for &r in old.roots() {
            out.mark_root(map[r]);
        }
        self.graph = out;
        reverted
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
                let id = self.graph.input(&s.name, s.shape.clone(), s.dtype);
                self.graph.mark_region(id, s.region);
                id
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
                let s = self.ordinal(dag, children[1], &name)?;
                let (shape, region) = (s.shape.clone(), s.region);
                let x = self.term(dag, children[2])?;
                let id = self.graph.view(&name, x, shape);
                self.graph.mark_region(id, region);
                id
            }
            "GemvP" => {
                let w = self.term(dag, children[0])?;
                let (pro, pw) = self.prologue(dag, children[1])?;
                let x = self.term(dag, children[2])?;
                self.graph.fused_with_prologue(
                    Core::Gemv,
                    "",
                    super::numeric::NumericClass::BitExact,
                    w,
                    x,
                    None,
                    None,
                    (pro.name, pw, pro.class()),
                )
            }
            head if head.starts_with("Gemv") => {
                let core = core_of(&head["Gemv".len()..])?;
                let w = self.term(dag, children[0])?;
                let x = self.term(dag, children[1])?;
                self.graph.gemv_on(core, w, x)
            }
            "Silu" => {
                let x = self.term(dag, children[0])?;
                self.graph.silu(x)
            }
            "Sigmoid" => {
                let x = self.term(dag, children[0])?;
                self.graph.sigmoid(x)
            }
            "Scale" => {
                let c = Self::scale(dag, children[0])?;
                let x = self.term(dag, children[1])?;
                self.graph.scale(x, c)
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
                let key = (self.graph.node(w).name.clone(), Self::value_key(&self.graph, x));
                if let Some(&n) = self.norms.get(&key) {
                    n
                } else {
                    let n = self.graph.rms_norm(w, x);
                    self.norms.insert(key, n);
                    n
                }
            }
            head if head.starts_with("K_") => {
                let k = self.kernel(&head["K_".len()..])?;
                let a = self.term(dag, children[0])?;
                let b = self.term(dag, children[1])?;
                let (shape, dtype) = (self.graph.node(a).shape.clone(), self.graph.node(a).dtype);
                let id = self.graph.opaque(k.family, &[a, b], shape, dtype);
                let class = self.graph.node(a).numeric.join(self.graph.node(b).numeric).join(k.class);
                self.graph.set_numeric(id, class);
                *self.placed.entry(k.name.to_string()).or_insert(0) += 1;
                id
            }
            head if head.starts_with("L_") => {
                let k = self.kernel(&head["L_".len()..])?;
                let y = self.term(dag, children[0])?;
                *self.placed.entry(k.name.to_string()).or_insert(0) += 1;
                y
            }
            head if head.starts_with("Fused") => {
                // `FusedP w pro x epi` (the GEMV core, a prologue) or
                // `Fused{core} w x epi`.
                let (core, prologue, x_at, epi_at) = if head == "FusedP" {
                    (Core::Gemv, Some(self.prologue(dag, children[1])?), 2, 3)
                } else {
                    (core_of(&head["Fused".len()..])?, None, 1, 2)
                };
                let w = self.term(dag, children[0])?;
                let x = self.term(dag, children[x_at])?;
                let egglog::Term::App(ehead, echildren) = dag.get(children[epi_at]) else {
                    return Err(Error::Msg("optimizer: malformed epilogue term".to_string()));
                };
                let echildren = echildren.clone();
                let (name, scaled) = match ehead.strip_prefix("EpiS_") {
                    Some(name) => (name, true),
                    None => (
                        ehead.strip_prefix("Epi_").ok_or_else(|| {
                            Error::Msg(format!("optimizer: unknown epilogue constructor `{ehead}`"))
                        })?,
                        false,
                    ),
                };
                let fragment = self
                    .fragments
                    .iter()
                    .find(|f| f.name == name && f.core == core)
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "optimizer: extracted fragment `{name}` not in the catalogue"
                        ))
                    })?
                    .clone();
                let (operand, scale) = if scaled {
                    (None, Some(Self::scale(dag, echildren[0])?))
                } else {
                    let operand = match echildren.first() {
                        Some(&r) => Some(self.term(dag, r)?),
                        None => None,
                    };
                    (operand, fragment.scales.then_some(1.0))
                };
                match prologue {
                    None => self
                        .graph
                        .fused_on(core, fragment.name, fragment.class(), w, x, operand, scale),
                    Some((pro, pw)) => self.graph.fused_with_prologue(
                        core,
                        fragment.name,
                        fragment.class(),
                        w,
                        x,
                        operand,
                        scale,
                        (pro.name, pw, pro.class()),
                    ),
                }
            }
            op if op.starts_with("Op") => {
                let name = Self::string(dag, children[0])?;
                let s = self.ordinal(dag, children[1], &name)?;
                let (shape, dtype, region) = (s.shape.clone(), s.dtype, s.region);
                let mut inputs = Vec::with_capacity(children.len() - 2);
                for &c in &children[2..] {
                    inputs.push(self.term(dag, c)?);
                }
                let id = self.graph.opaque(&name, &inputs, shape, dtype);
                self.graph.mark_region(id, region);
                id
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
