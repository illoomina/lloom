//! Backend-independent command-buffer partition policy and grammar.

/// Where a command buffer is cut (`DESIGN.md` §13.5m, issue #420 stage 3).
///
/// # Why there are two
///
/// The shipping cut is a **global dispatch counter**: `count >=
/// compute_per_buffer`, with `count` running from process start. §13.5i
/// measured what that produces — every seam at a multiple of 50 *counted from
/// process start*, so the flash arm's land at 51/101/152 rather than
/// 50/100/150, and of a token's seams 6 fall inside attention layers and 7
/// inside conv, none at a position the graph would have chosen. The placement
/// is a **counter's residue**, not a decision.
///
/// §13.5j then measured the cost of the cut itself and found it is **~7.2–7.4
/// µs per encoder session and not a function of the barrier count** — but
/// barriers and sessions are *collinear* along `COMPUTE_PER_BUFFER` (R² 0.983
/// and 0.985), because one cut produces both. **No setting of that knob can
/// separate the two terms.**
///
/// A node-index partition is the arm that can, and that is the only reason
/// this enum exists: it moves *where* the seams land while holding *how many*
/// there are, so the session term and the barrier term stop moving together.
///
/// # What a "node" is here, and why it is the layer
///
/// llama.cpp partitions a materialised graph — `n_nodes_per_cb = (n_nodes +
/// n_cb - 1) / n_cb`, shipping `n_cb` of 1 or 2 (#419). Candle is eager and has
/// no graph object to index, so the node index has to come from the only party
/// that knows the graph position: the model. LFM2's decoder is a loop over 30
/// layers, so **node := layer** and the partition is over the layer sequence.
///
/// That is what makes the count held rather than approximately held: cutting a
/// 30-layer sequence into `n_cb` contiguous runs yields **exactly `n_cb`
/// sessions per forward pass, by construction**, independent of how many
/// dispatches a layer happens to contain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutPolicy {
    /// Cut every `compute_per_buffer` dispatches, counted from process start.
    /// The default and the shipping path; §13.5j's arm and every recorded
    /// digest belong to it.
    DispatchCounter,
    /// Cut at the first node boundary once `compute_per_buffer` dispatches
    /// have accumulated, and unconditionally at four periods.
    ///
    /// The counter's period as a *lower bound* on a session's length and the
    /// graph's own boundaries as the places a seam may fall: the count stays
    /// the counter's to within a node, the placement is the graph's, and a
    /// stream that marks no nodes (a model that never calls
    /// `mark_graph_node`) is still cut by the hard bound rather than growing
    /// one command buffer for its whole pass. `CANDLE_METAL_CUT_POLICY=
    /// node-aligned`.
    NodeAligned,
    /// Cut on a layer-index partition: `n_cb` contiguous runs of the layer
    /// sequence, so every seam lands on a layer boundary and the session count
    /// is `n_cb` by construction.
    ///
    /// `nodes_per_cb` is the run length in layers, derived from `n_cb` the same
    /// way llama.cpp derives it — `ceil(n_layers / n_cb)` — and computed by the
    /// caller, since only the model knows `n_layers`.
    NodeIndex { nodes_per_cb: usize },
}

impl CutPolicy {
    /// Whether reaching graph node `index` should arm a cut.
    ///
    /// A free function of the policy and the index so the partition arithmetic
    /// is testable **without a device** — `Commands` needs a live
    /// `MTLCommandQueue` and this needs none, which is `scratch_tests.rs`'s
    /// split between the arithmetic and what executes it.
    ///
    /// **Node 0 never cuts.** `n_cb` contiguous runs have `n_cb - 1` interior
    /// boundaries, and a cut at the first node would add a seam the partition
    /// does not name — the forward pass already opens against whatever encoder
    /// the previous token's `flush_and_wait_current` left. Cutting there would
    /// make the session count `n_cb + 1` and break the held-count property the
    /// arm exists to have.
    ///
    /// `since_cut` is how many dispatches the open session holds when the node
    /// is reached; only `NodeAligned` reads it, against `compute_per_buffer`.
    #[inline]
    pub fn cuts_at_node(&self, index: usize, since_cut: usize, compute_per_buffer: usize) -> bool {
        match self {
            // A dispatch counter has no notion of graph position, which is
            // §13.5i's whole finding. It cannot answer this question and does
            // not try.
            CutPolicy::DispatchCounter => false,
            CutPolicy::NodeIndex { nodes_per_cb } => {
                index > 0 && index.is_multiple_of(*nodes_per_cb)
            }
            CutPolicy::NodeAligned => index > 0 && since_cut >= compute_per_buffer,
        }
    }

    /// The dispatch count at which `NodeAligned` cuts whether or not a node
    /// boundary has arrived, as a multiple of the period.
    pub const NODE_ALIGNED_HARD_BOUND: usize = 4;

    /// Whether a session `since_cut` dispatches long must be cut at this
    /// dispatch, node or no node.
    #[inline]
    pub fn cuts_at_dispatch(&self, since_cut: usize, compute_per_buffer: usize) -> bool {
        match self {
            CutPolicy::DispatchCounter => since_cut >= compute_per_buffer,
            CutPolicy::NodeIndex { .. } => false,
            CutPolicy::NodeAligned => {
                since_cut >= compute_per_buffer.saturating_mul(Self::NODE_ALIGNED_HARD_BOUND)
            }
        }
    }

    /// The arm as a `RESULT`-line token, with the period the counter arms
    /// use: `Counter(50)`, `Node(3)`, `NodeAligned(50)`.
    pub fn render(policy: CutPolicy, compute_per_buffer: usize) -> String {
        match policy {
            CutPolicy::DispatchCounter => format!("Counter({compute_per_buffer})"),
            CutPolicy::NodeIndex { nodes_per_cb } => format!("Node({nodes_per_cb})"),
            CutPolicy::NodeAligned => format!("NodeAligned({compute_per_buffer})"),
        }
    }

    /// Parse the environment's spelling: `counter`, `node-aligned`, or
    /// `nodes:<k>`.
    pub fn from_env_str(s: &str) -> Result<CutPolicy, String> {
        match s {
            "counter" => Ok(CutPolicy::DispatchCounter),
            "node-aligned" => Ok(CutPolicy::NodeAligned),
            other => match other.strip_prefix("nodes:") {
                Some(k) => {
                    let n: usize = k
                        .parse()
                        .map_err(|_| format!("nodes:<k> needs a positive integer, got {k:?}"))?;
                    if n == 0 {
                        return Err("nodes:0 would cut at every node and then some".to_string());
                    }
                    Ok(CutPolicy::NodeIndex { nodes_per_cb: n })
                }
                None => Err(format!(
                    "CANDLE_METAL_CUT_POLICY must be `counter`, `node-aligned` or `nodes:<k>`, \
                     got {other:?}"
                )),
            },
        }
    }

    /// How many sessions a forward pass over `n_nodes` nodes will cut into.
    ///
    /// The held-count property, as arithmetic rather than as a claim: a
    /// partition of `n_nodes` into runs of `nodes_per_cb` has
    /// `ceil(n_nodes / nodes_per_cb)` runs, and each run is one session. Used
    /// by the tests and by anything choosing a `nodes_per_cb` to match a
    /// counter arm's count.
    pub fn sessions_for(
        &self,
        n_nodes: usize,
        compute_per_buffer: usize,
        n_dispatches: usize,
    ) -> usize {
        match self {
            // The counter's count is a function of the dispatch total, which is
            // why it cannot be held while placement moves (§13.5j).
            CutPolicy::DispatchCounter => n_dispatches.div_ceil(compute_per_buffer.max(1)),
            CutPolicy::NodeIndex { nodes_per_cb } => n_nodes.div_ceil((*nodes_per_cb).max(1)),
            // At most the counter's count, at least one; the exact figure
            // depends on where the nodes fall against the period and is
            // read from the run rather than predicted here.
            CutPolicy::NodeAligned => n_dispatches.div_ceil(compute_per_buffer.max(1)).max(1),
        }
    }
}
