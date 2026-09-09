//! What a derivation produces: the fusion plan, and the seams it could not
//! fuse.

use std::collections::BTreeMap;
use std::fmt;

use super::ir::{Census, Graph, Op};
use super::numeric::NumericClass;

/// A weight site the plan fuses an epilogue onto, keyed by the weight's
/// checkpoint name; the value is the fragment's name.
pub type Sites = BTreeMap<String, String>;

/// An adjacent producer→consumer seam in the derived graph that the
/// catalogue has no fragment for: a single-use dispatch feeding a standalone
/// elementwise dispatch. Each is a composition the optimizer wanted and could
/// not express — the "missing derivation rule / missing fragment" verdict of
/// `design/docs/optimizer-design.md` §4.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Seam {
    /// The producing node's kernel family.
    pub producer: String,
    /// The consuming elementwise node's kernel family.
    pub consumer: String,
    /// How many such seams the graph has.
    pub count: u64,
}

/// The result of deriving over one graph.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Weight name → fragment name, for every GEMV the plan fuses.
    pub sites: Sites,
    /// The class the derived graph's roots land in relative to the source.
    pub class: NumericClass,
    /// Dispatches per family, before.
    pub before: Census,
    /// Dispatches per family, after.
    pub after: Census,
    /// The seams left unfused for want of a fragment, most frequent first.
    pub seams: Vec<Seam>,
}

impl FusionPlan {
    pub fn dispatches_before(&self) -> u64 {
        self.before.values().sum()
    }

    pub fn dispatches_after(&self) -> u64 {
        self.after.values().sum()
    }

    /// The sites fused with `fragment`.
    pub fn sites_of(&self, fragment: &str) -> Vec<&str> {
        self.sites
            .iter()
            .filter(|(_, f)| f.as_str() == fragment)
            .map(|(s, _)| s.as_str())
            .collect()
    }
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut per_fragment: BTreeMap<&str, usize> = BTreeMap::new();
        for frag in self.sites.values() {
            *per_fragment.entry(frag.as_str()).or_insert(0) += 1;
        }
        write!(
            f,
            "dispatches {} -> {} class {}",
            self.dispatches_before(),
            self.dispatches_after(),
            self.class
        )?;
        for (frag, n) in per_fragment {
            write!(f, " gemv_{frag}={n}")?;
        }
        if !self.seams.is_empty() {
            write!(f, " unfused[")?;
            for (i, s) in self.seams.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}>{}={}", s.producer, s.consumer, s.count)?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

/// The fused sites of a graph: each `Fused` node's weight name and fragment.
pub fn sites(graph: &Graph) -> Sites {
    let mut sites = Sites::new();
    for node in graph.nodes() {
        if node.op == Op::Fused {
            let w = graph.node(node.inputs[0]);
            sites.insert(w.name.clone(), node.name.clone());
        }
    }
    sites
}

/// The seams of a graph: every standalone elementwise dispatch whose input is
/// a single-use dispatch (a view of one counts, since the seam is between the
/// producer's store and the consumer's load), grouped by family pair.
pub fn seams(graph: &Graph) -> Vec<Seam> {
    let uses = graph.use_counts();
    let mut counts: BTreeMap<(String, String), u64> = BTreeMap::new();
    for node in graph.nodes() {
        if !node.op.is_elementwise() {
            continue;
        }
        let consumer = super::ir::family(node).expect("elementwise ops dispatch");
        for &input in &node.inputs {
            // Look through a single-use view to the value it views.
            let mut id = input;
            while graph.node(id).op == Op::View && uses[id] == 1 {
                id = graph.node(id).inputs[0];
            }
            let producer = graph.node(id);
            if uses[id] != 1 {
                continue;
            }
            let Some(family) = super::ir::family(producer) else {
                continue;
            };
            *counts.entry((family, consumer.clone())).or_insert(0) += 1;
        }
    }
    let mut seams: Vec<Seam> = counts
        .into_iter()
        .map(|((producer, consumer), count)| Seam {
            producer,
            consumer,
            count,
        })
        .collect();
    seams.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.cmp(b)));
    seams
}
