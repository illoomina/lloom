//! What a derivation produces: the fusion plan, and the seams it could not
//! fuse.

use std::collections::BTreeMap;
use std::fmt;

use super::cost::StepCost;
use super::ir::{Census, Graph, Op};
use super::numeric::{NumericClass, Region};

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

/// A fused site that sits in a recurrence: the verdict admitting its
/// fragment was measured on a straight-line region and does not compose
/// across the recurrence by addition (§5a). Reported, never approximated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatefulSite {
    pub site: String,
    pub fragment: String,
    pub region: Region,
}

/// The step's recurrences, as the emitter marked them, and what the plan put
/// inside them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Regions {
    /// Conv layers with a state window.
    pub conv_ring: usize,
    /// Attention layers with a KV cache.
    pub kv: usize,
    /// Linear-attention layers with a recurrent state.
    pub ssm: usize,
    /// Nodes marked stateful, over all regions.
    pub nodes: usize,
    /// Fused sites whose GEMV, or the elementwise dispatch it absorbed, was
    /// marked stateful.
    pub fused_in_stateful: Vec<StatefulSite>,
}

/// The result of deriving over one graph.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Weight name → block name, for every GEMV the plan fuses (a prologue
    /// first: `norm_silu`).
    pub sites: Sites,
    /// Each kernel fragment's placements (`silu_mul`, `combine_cast`).
    pub kernels: BTreeMap<String, usize>,
    /// The class the derived graph's roots land in relative to the source.
    pub class: NumericClass,
    /// Dispatches per family, before.
    pub before: Census,
    /// Dispatches per family, after.
    pub after: Census,
    /// The seams left unfused for want of a fragment, most frequent first.
    pub seams: Vec<Seam>,
    /// The recurrences the source graph marked, and the fused sites inside
    /// them.
    pub regions: Regions,
    /// The step's cost under the barrier model, before and after, when the
    /// policy set one.
    pub cost: Option<(StepCost, StepCost)>,
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
        for (k, n) in &self.kernels {
            write!(f, " {k}={n}")?;
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
        write!(
            f,
            " regions[conv-ring={} kv={}",
            self.regions.conv_ring, self.regions.kv
        )?;
        if self.regions.ssm > 0 {
            write!(f, " ssm={}", self.regions.ssm)?;
        }
        write!(
            f,
            " nodes={} fused-in-stateful={}",
            self.regions.nodes,
            self.regions.fused_in_stateful.len()
        )?;
        for s in &self.regions.fused_in_stateful {
            write!(f, " {}:{}@{}", s.site, s.fragment, s.region)?;
        }
        write!(f, "]")?;
        if let Some((before, after)) = &self.cost {
            write!(
                f,
                " cost[before: {before}; after: {after}; {:+.0} µs]",
                after.total_us() - before.total_us()
            )?;
        }
        Ok(())
    }
}

/// The recurrences of the SOURCE graph and the fused `sites` that fall in
/// one: a site is stateful if its GEMV's node, or any elementwise reader the
/// fusion absorbed (a `Silu`/`Add` whose input is that GEMV), was marked.
/// Read off the source because the rebuilt graph is a fresh arena and the
/// marks are the emitter's.
pub fn regions(source: &Graph, sites: &Sites) -> Regions {
    let marked = source.regions();
    let mut out = Regions {
        conv_ring: marked
            .keys()
            .filter(|r| matches!(r, Region::ConvRing(_)))
            .count(),
        kv: marked.keys().filter(|r| matches!(r, Region::Kv(_))).count(),
        ssm: marked
            .keys()
            .filter(|r| matches!(r, Region::Ssm(_)))
            .count(),
        nodes: marked.values().map(Vec::len).sum(),
        fused_in_stateful: Vec::new(),
    };
    for (id, node) in source.nodes().iter().enumerate() {
        if node.op != Op::Gemv {
            continue;
        }
        let w = source.node(node.inputs[0]);
        let Some(fragment) = sites.get(&w.name) else {
            continue;
        };
        let mut region = node.region;
        if !region.is_stateful() {
            // The elementwise dispatch the fusion absorbed.
            for reader in source.nodes() {
                if reader.op.is_elementwise()
                    && reader.inputs.contains(&id)
                    && reader.region.is_stateful()
                {
                    region = reader.region;
                    break;
                }
            }
        }
        if region.is_stateful() {
            out.fused_in_stateful.push(StatefulSite {
                site: w.name.clone(),
                fragment: fragment.clone(),
                region,
            });
        }
    }
    out
}

/// The fused sites of a graph: each `Fused` node's weight name and fragment.
pub fn sites(graph: &Graph) -> Sites {
    let mut sites = Sites::new();
    for node in graph.nodes() {
        if node.op == Op::Fused {
            let w = graph.node(node.inputs[0]);
            sites.insert(w.name.clone(), node.block());
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
