//! Expert residency — the seam a top-k MoE layer applies its routed experts
//! through, and its two strategies: `Resident` (every expert in memory) and
//! `Streamed` (a slot pool filled from the checkpoint on demand)
//! (`design/docs/expert-streaming.md` §20.4, §20.6).
//!
//! Model-independent: a layer hands the strategy the router's expert ids and
//! gets back the ids the routed kernel reads (`plan` + `ensure`), then
//! dispatches the same `mul_mv_id` kernels over whatever tensor the strategy
//! holds (`gate_up`, `down`). `Resident` monomorphizes to one group and the
//! ids untouched — no readback, no map, the path a non-streamed model takes.
//!
//! `Streamed` keeps per layer three packed slot pools (gate, up, down — an
//! expert is resident in all three or none) and ONE residency map keyed by
//! expert id. Per forward it reads the top-k ids back once (the host has to
//! know the routing to load anything; a prefetch policy is what removes that
//! sync, §20.3, phase 1), groups the rows so each group's expert union fits
//! the pool, and per group loads the misses into LRU-chosen unpinned slots
//! before the dispatch. Two loader arms (§20.4a): `Pread` reads the expert's
//! bytes from the file into a staging buffer that a compute-encoder copy lands
//! in the slot, so the write is ordered after every earlier reader of the slot
//! by the encoder's own hazard chain; `Io` (`MTLIOCommandQueue`) loads the
//! file straight into the slot, ordered by shared events on both queues. On
//! neither arm does an eviction wait on the GPU.

use crate::nn::slot_cache::{ExpertId, ExpertSlots, SlotId};
use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::{GgmlDType, MoeArm, QTensor};
use lloom_tensor::retire::{Completed, Epoch, Retire, Tick};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The rows of one routed dispatch and the expert ids they select.
#[derive(Debug, Clone)]
pub struct RoutedGroup {
    /// The rows of the layer's `ids [rows, topk]` this group covers.
    pub rows: Range<usize>,
    ids: GroupIds,
}

#[derive(Debug, Clone)]
enum GroupIds {
    /// The ids as the router produced them, on the device (`Resident`).
    Device(Tensor),
    /// The ids read back to the host, row-major `[rows · topk]` (`Streamed`).
    Host { ids: Vec<u32>, topk: usize },
}

/// One layer's plan of a speculative window, on the host: the `[3·n]` record
/// a device plan writes (the slot ids | the expert ids | a miss flag per id,
/// `n = rows × topk`, row-major) or the same layout a host plan filled in.
/// Settled after the accept ([`ExpertResidency::settle_window`]).
#[derive(Debug, Clone)]
pub struct WindowRecord {
    pub data: Vec<u32>,
    pub rows: usize,
    pub topk: usize,
    /// A device plan's record: the settle closes the plan.
    pub device: bool,
    /// Rows the host plan already resolved with loads (touched and counted
    /// then): the settle touches the accepted rows from this one on. 0 for
    /// a device plan.
    pub resolved_rows: usize,
    /// Experts the host plan loaded for those rows (a device plan loads
    /// nothing): the arming reads the real rows' loads alone.
    pub real_loads: u64,
    /// Experts the host plan loaded for the draft rows beside them (0 when
    /// the drafts were resolved against what is resident, or resident).
    pub draft_loads: u64,
}

impl WindowRecord {
    /// A device plan's `[3·n]` record read back.
    pub fn device(data: Vec<u32>, rows: usize, topk: usize) -> Result<Self> {
        if data.len() != 3 * rows * topk {
            lloom_tensor::bail!(
                "window record: [3·n] with n = {rows} × {topk}, got {}",
                data.len()
            )
        }
        Ok(Self {
            data,
            rows,
            topk,
            device: true,
            resolved_rows: 0,
            real_loads: 0,
            draft_loads: 0,
        })
    }

    fn n(&self) -> usize {
        self.rows * self.topk
    }

    pub fn ids(&self) -> &[u32] {
        &self.data[self.n()..2 * self.n()]
    }

    pub fn flags(&self) -> &[u32] {
        &self.data[2 * self.n()..]
    }

    /// The first row with a flagged (missing) expert, if any. A row's
    /// state and carry are wrong from the layer that flagged it, and so is
    /// every row after it (attention and the recurrence read the rows
    /// before them): the window is valid up to this row.
    pub fn first_flagged_row(&self) -> Option<usize> {
        let flags = self.flags();
        (0..self.rows).find(|&r| {
            flags[r * self.topk..(r + 1) * self.topk]
                .iter()
                .any(|&f| f != 0)
        })
    }

    /// One record over the rows of several (a reference path that planned
    /// the window position by position): the resolved rows must be a
    /// prefix.
    pub fn concat_rows(recs: &[WindowRecord]) -> Result<Self> {
        let Some(first) = recs.first() else {
            lloom_tensor::bail!("window record: no rows to join")
        };
        let topk = first.topk;
        let rows: usize = recs.iter().map(|r| r.rows).sum();
        let n = rows * topk;
        let mut data = vec![0u32; 3 * n];
        let mut at = 0usize;
        let mut resolved_rows = 0usize;
        let mut resolved_done = false;
        let mut real_loads = 0u64;
        let mut draft_loads = 0u64;
        for r in recs {
            real_loads += r.real_loads;
            draft_loads += r.draft_loads;
            if r.topk != topk || r.device != first.device {
                lloom_tensor::bail!("window record: rows of different shape or plan")
            }
            let m = r.n();
            data[at..at + m].copy_from_slice(&r.data[..m]);
            data[n + at..n + at + m].copy_from_slice(&r.data[m..2 * m]);
            data[2 * n + at..2 * n + at + m].copy_from_slice(&r.data[2 * m..]);
            at += m;
            if r.resolved_rows == r.rows && !resolved_done {
                resolved_rows += r.rows;
            } else if r.resolved_rows == 0 {
                resolved_done = true;
            } else {
                lloom_tensor::bail!("window record: the resolved rows are not a prefix")
            }
        }
        Ok(Self {
            data,
            rows,
            topk,
            device: first.device,
            resolved_rows,
            real_loads,
            draft_loads,
        })
    }

    /// Flags set on the rows from `from` on.
    pub fn flags_from(&self, from: usize) -> u64 {
        let flags = self.flags();
        flags[(from.min(self.rows) * self.topk)..]
            .iter()
            .filter(|&&f| f != 0)
            .count() as u64
    }
}

/// How a layer's routed experts are held and applied.
///
/// The contract is the routed matmul's: the strategy receives the activation
/// and the ids of the selected experts and returns the per-slot projections,
/// `[rows, topk, n]`, plane `(r, t)` for `ids[r, t]` — so the combine above
/// the seam never learns whether the experts were resident or streamed.
pub trait ExpertResidency {
    /// Group the rows of `ids [rows, topk]` (u32, on the device) into
    /// dispatchable groups: `Resident` returns one group; `Streamed` groups
    /// rows so each group's distinct experts fit its slot pool.
    fn plan(&self, ids: &Tensor) -> Result<Vec<RoutedGroup>>;

    /// Make the group's experts resident and return the ids the dispatch reads,
    /// `[rows, topk]` u32 on the device — the group's ids themselves for
    /// `Resident`, slot indices for `Streamed`.
    fn ensure(&self, group: &RoutedGroup) -> Result<Tensor>;

    /// `gate` and `up` projections of `x [rows, 1, hidden]` through the experts
    /// `slot_ids [rows, topk]`: each `[rows, topk, n_ff]`. `arm` is the
    /// tile's ([`MoeArm::for_rows`] over every row of the pass), not the
    /// group's: a split changes which slots are read, never how a row is
    /// computed (§20.15). `act` is the dtype the experts compute in; an `x`
    /// in another dtype is read as if cast to it (§20.15e).
    fn gate_up(
        &self,
        x: &Tensor,
        slot_ids: &Tensor,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)>;

    /// `down` projection of `h [rows, topk, n_ff]` through `slot_ids`:
    /// `[rows, topk, hidden]`, on the tile's `arm` as [`Self::gate_up`].
    fn down(&self, h: &Tensor, slot_ids: &Tensor, arm: MoeArm) -> Result<Tensor>;

    /// The elastic surface (`profiles-and-memory-design.md` §4.2): slots in
    /// service, and shrinking / growing them. A resident tier has none.
    fn active_slots(&self) -> usize {
        0
    }
    fn shrink_slots(&self, _active: usize, _floor: usize) -> Result<usize> {
        Ok(0)
    }
    fn grow_slots(&self, _active: usize) -> usize {
        0
    }

    /// The pass that follows replays the token the last settle counted
    /// (the host plan over the same token): its resolves must not count
    /// those experts into the hotlist's usage again (#899's finding, a
    /// device-planned step that misses counted them twice). `Resident`
    /// keeps no usage.
    fn note_replay(&self) {}

    /// Whether [`Self::plan_device`] can answer: the strategy keeps a
    /// residency table on the device. `Resident` never needs to.
    fn plans_on_device(&self) -> bool {
        false
    }

    /// [`Self::plan`] over ids already on the host (`rows × topk`, row
    /// major) — for a caller that read them back itself, staged
    /// (`metal_backend::stage_readback`) so the wait was its own marker's.
    /// `None` when the strategy plans on the device tensor (`Resident`):
    /// the caller then uses `plan`, which reads nothing back there.
    fn plan_host(
        &self,
        _ids: &[u32],
        _rows: usize,
        _topk: usize,
    ) -> Result<Option<Vec<RoutedGroup>>> {
        Ok(None)
    }

    /// How many of a host-planned group's experts are not resident now —
    /// what `ensure` would load. 0 for a strategy with no pool.
    fn group_misses(&self, _group: &RoutedGroup) -> usize {
        0
    }

    /// Experts loaded so far (the miss counter), 0 when nothing streams —
    /// a pass that leaves it unchanged found everything resident.
    fn misses_so_far(&self) -> u64 {
        0
    }

    /// Experts evicted so far, 0 when nothing streams.
    fn evictions_so_far(&self) -> u64 {
        0
    }

    /// The device-side plan of `ids [rows, topk]` (u32, on the device): the
    /// slot ids the dispatch reads — NO host sync, an expert not resident
    /// maps to slot 0 and is flagged — and the plan's record for
    /// [`Self::settle`]. `None` where the strategy has no device plan (the
    /// caller takes [`Self::plan`] / [`Self::ensure`]).
    fn plan_device(&self, _ids: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    /// Where a one-row step's router should write its `topk` ids: a tensor
    /// over the strategy's host-read slot, which [`Self::plan`] then reads in
    /// place (#814 part 5). `None` where the strategy reads nothing back
    /// (`Resident`) or off Metal: the router allocates its own output.
    fn decode_ids_slot(&self, _topk: usize) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// The device plan's `record`, read in place from the strategy's
    /// host-read slot when `record` is that slot's value; `None` otherwise
    /// (the caller reads it back). **The caller has waited** for the pass that
    /// wrote it (#814 part 5).
    fn read_record_in_place(&self, _record: &Tensor) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Read a device plan's record back (the one host sync, at the end of
    /// the pass, for every layer at once), touch the residency of what the
    /// pass used and load what it lacked. `true` when the pass computed on
    /// a missing expert — its result is wrong and the pass is replayed; the
    /// experts are resident now.
    fn settle(&self, _record: &Tensor) -> Result<bool> {
        Ok(false)
    }

    /// [`Self::settle`] from a record already on the host: the caller read
    /// every layer's record in ONE readback after the pass (a `cat` and one
    /// wait — #749 P29: 48 readbacks a token were 47 seams the GPU idled
    /// across) and hands each layer its `[3·n]` slice, in layer order.
    fn settle_host(&self, _record: &[u32]) -> Result<bool> {
        Ok(false)
    }

    /// [`Self::settle_host`] over several layers' records at once, in layer
    /// order: per part, whether its pass computed on a missing expert (a
    /// replay) and how many experts were loaded. A strategy may begin every
    /// part's loads before finishing any, so a token's misses read together
    /// (`Streamed`); the default settles one part after another.
    fn settle_batch(parts: &[(&Self, &[u32])]) -> Result<Vec<(bool, u64)>>
    where
        Self: Sized,
    {
        parts
            .iter()
            .map(|(r, v)| {
                let before = r.misses_so_far();
                let redo = r.settle_host(v)?;
                Ok((redo, r.misses_so_far() - before))
            })
            .collect()
    }

    /// The host plan of a speculative window `ids [rows, topk]` (u32, on
    /// the device; row 0 the real next token, the rows after it drafts):
    /// the first `real_rows` rows' experts made resident (loaded, counted
    /// as used); the rows up to `loaded_rows` made resident too but counted
    /// only once the accept keeps them; every other row resolved against
    /// what is resident — a miss maps to slot 0 and FLAGS the row instead
    /// of loading, so a wrong draft loads nothing and evicts nothing
    /// (`expert-streaming.md` §20.16). Returns the slot ids the dispatch
    /// reads and the window's record for [`Self::settle_window`]; `None`
    /// where the strategy keeps no pool (`Resident`: the caller takes
    /// [`Self::plan`] / [`Self::ensure`]).
    fn plan_window(
        &self,
        _ids: &Tensor,
        _real_rows: usize,
        _loaded_rows: usize,
    ) -> Result<Option<(Tensor, WindowRecord)>> {
        Ok(None)
    }

    /// Settle a window's records AFTER the accept, in layer order: the plans
    /// closed, and the residency touched and the usage counted for the
    /// experts of the `accepted` rows alone — a rejected row's experts are
    /// neither touched, nor counted, nor loaded. The caller has already
    /// bounded `accepted` by the first flagged row ([`WindowRecord::first_flagged_row`]):
    /// every accepted row's expert was resident when its plan ran. Returns
    /// the flags set on the rows past `accepted` (the wrong-path misses not
    /// loaded).
    fn settle_window(parts: &[(&Self, &WindowRecord)], _accepted: usize) -> Result<u64>
    where
        Self: Sized,
    {
        let _ = parts;
        Ok(0)
    }

    /// A hint that a multi-token pass is about to run this layer: stage the
    /// experts it will most likely miss, off the encoding thread
    /// (`nn::expert_prefetch`, §20.3 P1). Never changes what the layer
    /// computes; `Resident` and a tier without a prefetcher ignore it.
    fn prefetch_ahead(&self) -> Result<()> {
        Ok(())
    }

    /// A pass whose expert union exceeds the pool resolves in several
    /// groups (`plan`): once a group is resident, stage the LATER groups'
    /// missing experts while this group's dispatch runs (§20.4, DS4's
    /// overlap of the routed loads with the experts already in RAM). A
    /// hint like `prefetch_ahead`; `Resident` ignores it.
    fn prefetch_groups(&self, _later: &[RoutedGroup]) -> Result<()> {
        Ok(())
    }

    /// A hint naming the experts a pass is predicted to route to (§20.3
    /// P2, the layer's router on a carry from before its mixer): stage the
    /// ones not held. A hint like `prefetch_ahead`; `Resident` ignores it.
    /// [`Self::prefetch_ids`] for the named experts ALONE — no fill from the
    /// usage profile (a decode hint: ten experts a layer, not the pool).
    fn prefetch_named(&self, _ids: &[u32]) -> Result<()> {
        Ok(())
    }

    fn prefetch_ids(&self, _ids: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Stage the hotlist's experts not resident — the usage profile's
    /// top, up to the pool — ahead of a device-planned token, so
    /// `plan_device` can land them before the plan maps the pool (#849's
    /// D on the device plan, the prediction the hotlist gives; the
    /// residual-speculation hint is the host plan's today). `Resident`
    /// has nothing to stage.
    fn prefetch_hot(&self) -> Result<()> {
        Ok(())
    }

    /// The prefetcher's counters (shared by the tier's layers), zeros
    /// without one.
    fn prefetch_stats(&self) -> PrefetchStats {
        PrefetchStats::default()
    }

    /// The fetch inside the token (`QWEN4EXP_FETCH=inside`,
    /// `nn::expert_fetch`): whether this tier can load a running pass's
    /// misses beside it — reader threads, Shared slot pools the host
    /// writes, a residency table the device reads in place. `Resident`
    /// never needs to.
    fn fetch_ready(&self) -> bool {
        false
    }

    /// Before a fetching pass, with no command that writes it pending:
    /// fill the routing's host-read slot with a value no expert has, so the
    /// worker can tell the pass's routing from the last one. `false` when
    /// the tier does not fetch.
    fn fetch_prepare(&self, _topk: usize) -> Result<bool> {
        Ok(false)
    }

    /// At a fetching pass's layer: hand `lane` this layer's job — the
    /// routing `ids`, written into the tier's host-read slot — and encode
    /// the device's wait for it. `false` (nothing encoded) when the tier
    /// does not fetch or `ids` is not its slot's.
    #[cfg(feature = "metal")]
    fn fetch_register(
        &self,
        _lane: &crate::nn::expert_fetch::FetchLane,
        _ids: &Tensor,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// The prefetcher's counters; the real type on Metal, zeros elsewhere.
#[cfg(feature = "metal")]
pub use crate::nn::expert_prefetch::PrefetchStats;
/// The prefetcher's counters (no prefetcher off Metal).
#[cfg(not(feature = "metal"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    pub issued: u64,
    pub staged: u64,
    pub used: u64,
    pub wasted: u64,
    pub stale: u64,
    pub bytes: u64,
    pub parked: u64,
    pub now: u64,
}

/// All experts resident in RAM: the small-model / big-RAM case, zero streaming
/// overhead. Holds the checkpoint's stacked expert tensors as they come —
/// `ffn_{gate,up}_exps [n_expert, n_ff, hidden]`, `ffn_down_exps [n_expert,
/// hidden, n_ff]` — packed; the routed kernel reads the selected experts'
/// blocks in place.
#[derive(Debug, Clone)]
pub struct Resident {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

impl Resident {
    pub fn new(gate: Arc<QTensor>, up: Arc<QTensor>, down: Arc<QTensor>) -> Self {
        Self { gate, up, down }
    }
}

impl ExpertResidency for Resident {
    #[inline]
    fn plan(&self, ids: &Tensor) -> Result<Vec<RoutedGroup>> {
        let rows = ids.dim(0)?;
        Ok(vec![RoutedGroup {
            rows: 0..rows,
            ids: GroupIds::Device(ids.clone()),
        }])
    }
    #[inline]
    fn ensure(&self, group: &RoutedGroup) -> Result<Tensor> {
        match &group.ids {
            GroupIds::Device(t) => Ok(t.clone()),
            GroupIds::Host { .. } => lloom_tensor::bail!("Resident::ensure: a host-planned group"),
        }
    }
    #[inline]
    fn gate_up(
        &self,
        x: &Tensor,
        ids: &Tensor,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)> {
        // One launch for the pair on the matmul-vec arm on Metal; two elsewhere.
        self.gate
            .indexed_moe_forward_pair_as(&self.up, x, ids, arm, act)
    }
    #[inline]
    fn down(&self, h: &Tensor, ids: &Tensor, arm: MoeArm) -> Result<Tensor> {
        self.down.indexed_moe_forward_with(h, ids, arm)
    }
}

/// One stacked expert tensor in a GGUF file, addressed by expert: plane `e`
/// is the contiguous byte range `[e · plane_bytes, (e+1) · plane_bytes)` of
/// the tensor's data (rows are whole blocks, so a plane is whole blocks).
pub struct ExpertSource {
    file: Arc<File>,
    path: PathBuf,
    #[cfg(feature = "metal")]
    io: Option<lloom_tensor::kernels::metal::io_queue::IoFile>,
    data_start: u64,
    plane_bytes: usize,
    n_experts: usize,
    n: usize,
    k: usize,
    dtype: GgmlDType,
}

impl std::fmt::Debug for ExpertSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ExpertSource {{ {} [{}, {}, {}] {:?} @{} plane={} }}",
            self.path.display(),
            self.n_experts,
            self.n,
            self.k,
            self.dtype,
            self.data_start,
            self.plane_bytes
        )
    }
}

impl ExpertSource {
    /// Locate tensor `name` (`[n_experts, n, k]`) in the shard `content` was
    /// read from at `path`. Reads no expert bytes.
    pub fn from_gguf(content: &Content, path: &Path, name: &str) -> Result<Self> {
        let info = match content.tensor_infos.get(name) {
            Some(i) => i,
            None => lloom_tensor::bail!("{} has no tensor `{name}`", path.display()),
        };
        let (n_experts, n, k) = info.shape.dims3()?;
        let dtype = info.ggml_dtype;
        if !k.is_multiple_of(dtype.block_size()) {
            lloom_tensor::bail!(
                "`{name}`: k {k} is not a multiple of the {dtype:?} block ({})",
                dtype.block_size()
            )
        }
        let plane_bytes = n * (k / dtype.block_size()) * dtype.type_size();
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let data_start = content.tensor_data_offset + info.offset;
        let need = data_start + (n_experts * plane_bytes) as u64;
        if need > len {
            lloom_tensor::bail!(
                "`{name}` needs {need} bytes of {}, which has {len}",
                path.display()
            )
        }
        Ok(Self {
            file: Arc::new(file),
            path: path.to_path_buf(),
            #[cfg(feature = "metal")]
            io: None,
            data_start,
            plane_bytes,
            n_experts,
            n,
            k,
            dtype,
        })
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    /// Bytes of one expert's plane.
    pub fn plane_bytes(&self) -> usize {
        self.plane_bytes
    }
    /// `(n, k)` of one expert's matrix.
    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }
    /// The byte offset of expert `e`'s plane in the file.
    pub fn plane_offset(&self, e: usize) -> u64 {
        self.data_start + (e * self.plane_bytes) as u64
    }

    /// This source's addressing for the prefetch thread.
    #[cfg(feature = "metal")]
    pub fn plane_source(&self) -> crate::nn::expert_prefetch::PlaneSource {
        crate::nn::expert_prefetch::PlaneSource {
            file: Arc::clone(&self.file),
            data_start: self.data_start,
            plane_bytes: self.plane_bytes,
        }
    }

    /// Read expert `e`'s plane into `dst` (exactly `plane_bytes`).
    pub fn read_plane(&self, e: usize, dst: &mut [u8]) -> Result<()> {
        if e >= self.n_experts {
            lloom_tensor::bail!("expert {e} out of {}", self.n_experts)
        }
        if dst.len() != self.plane_bytes {
            lloom_tensor::bail!(
                "read_plane: {} bytes requested, a plane is {}",
                dst.len(),
                self.plane_bytes
            )
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(dst, self.plane_offset(e))?;
            Ok(())
        }
        #[cfg(not(unix))]
        {
            lloom_tensor::bail!("ExpertSource::read_plane needs a positional read (unix)")
        }
    }
}

/// Which primitive moves an expert's bytes into its slot (§20.4a).
#[derive(Clone)]
pub enum LoaderArm {
    /// `pread` into a staging buffer, landed by a compute-encoder copy
    /// (candidate A; the only arm on the CPU).
    Pread,
    /// `MTLIOCommandQueue` loads straight into the slot, ordered by shared
    /// events (candidate B). One queue shared by every layer.
    #[cfg(feature = "metal")]
    Io(Arc<lloom_tensor::kernels::metal::io_queue::IoQueue>),
}

impl std::fmt::Debug for LoaderArm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoaderArm::Pread => write!(f, "Pread"),
            #[cfg(feature = "metal")]
            LoaderArm::Io(q) => write!(f, "Io({q:?})"),
        }
    }
}

impl LoaderArm {
    pub fn name(&self) -> &'static str {
        match self {
            LoaderArm::Pread => "pread",
            #[cfg(feature = "metal")]
            LoaderArm::Io(_) => "io",
        }
    }
}

/// Which expert is in which slot, with the liveness and the policy: a
/// `SlotCache<ExpertId>` (`nn::slot_cache`) plus the per-expert use count
/// the hotlist reads (§20.3 R2). The tier's own vocabulary stays `usize` /
/// `u32` at this boundary; the cache's ids are the newtypes.
#[derive(Debug, Clone)]
pub struct ResidencyMap {
    cache: ExpertSlots,
    /// Uses per expert (a hit or a miss each time a pass needs it).
    usage: Vec<u32>,
}

const NONE: u32 = crate::nn::slot_cache::NONE;

impl ResidencyMap {
    pub fn new(n_experts: usize, n_slots: usize) -> Self {
        Self {
            cache: ExpertSlots::new(n_experts, n_slots),
            usage: vec![0; n_experts],
        }
    }

    /// Select the arm: `spare` slots kept free (0: in place). Only on an
    /// empty map (a pool that has loaded nothing).
    pub fn set_spare(&mut self, spare: usize) {
        self.cache.set_spare(spare)
    }

    /// Whether misses land in free slots (the quarantined arm).
    pub fn quarantined(&self) -> bool {
        self.cache.quarantined()
    }

    /// Slots the quarantined arm keeps free (0: the in-place arm).
    pub fn spare(&self) -> usize {
        self.cache.spare()
    }

    /// Slots evicted and not yet reusable (quarantined or deferred).
    pub fn held(&self) -> usize {
        self.cache.held()
    }

    /// Free slots on the quarantined arm's list.
    fn free_len(&self) -> usize {
        self.cache.free_len()
    }

    /// Whether any slot holds an expert.
    fn any_resident(&self) -> bool {
        self.cache.any_resident()
    }

    /// Move every quarantined slot whose epoch has completed to the free
    /// list; how many.
    fn recycle(&mut self, completed: Completed) -> usize {
        self.cache.recycle(completed)
    }

    /// A free slot, if any.
    fn take_free(&mut self) -> Option<usize> {
        self.cache.take_free().map(SlotId::index)
    }

    /// The least-recently-used RESIDENT unpinned slot not used in `tick` and
    /// not in flight.
    fn lru_resident(&self, tick: Tick) -> Option<usize> {
        self.cache.lru_resident(tick).map(SlotId::index)
    }

    /// Evict the LRU expert (not used in `tick`) out of the map; its slot
    /// quarantined at `epoch`, or deferred while a device plan is open.
    /// Returns the expert evicted.
    fn evict_lru(&mut self, tick: Tick, epoch: Epoch, plan_open: bool) -> Option<u32> {
        self.cache.evict_lru(tick, epoch, plan_open).map(|e| e.0)
    }

    /// The device plans are settled: every reader of a slot evicted while
    /// one was open is encoded now, in an epoch at most `epoch`.
    fn release_deferred(&mut self, epoch: Epoch) {
        self.cache.release_deferred(epoch)
    }

    pub fn n_slots(&self) -> usize {
        self.cache.n_slots()
    }

    /// The slot holding expert `e`, if resident.
    pub fn slot_of(&self, e: usize) -> Option<usize> {
        self.cache.slot_of(ExpertId(e as u32)).map(SlotId::index)
    }

    /// The expert in `slot`, or `NONE`.
    fn expert_in(&self, slot: usize) -> u32 {
        self.cache
            .key_in(SlotId(slot as u32))
            .map(|e| e.0)
            .unwrap_or(NONE)
    }

    /// Resident experts, in slot order.
    pub fn resident(&self) -> Vec<u32> {
        self.cache.resident().into_iter().map(|e| e.0).collect()
    }

    /// The expert → slot table as the device reads it.
    fn slot_table(&self) -> &[u32] {
        self.cache.slot_table()
    }

    /// Open a new use epoch (one per `ensure`).
    fn begin(&mut self) -> Tick {
        self.cache.begin()
    }

    fn tick(&self) -> Tick {
        self.cache.tick()
    }

    fn touch(&mut self, slot: usize, tick: Tick) {
        self.cache.touch(SlotId(slot as u32), tick)
    }

    /// [`SlotCache::victim_settled`].
    fn victim_settled(&self, tick: Tick, settled_before: Completed) -> Option<(usize, bool)> {
        self.cache
            .victim_settled(tick, settled_before)
            .map(|(s, e)| (s.index(), e))
    }

    /// [`SlotCache::victim_in`].
    fn victim_in(
        &self,
        tick: Tick,
        skip_inflight: bool,
        settled_before: Completed,
    ) -> Option<(usize, bool)> {
        self.cache
            .victim_in(tick, skip_inflight, settled_before)
            .map(|(s, e)| (s.index(), e))
    }

    /// [`SlotCache::victim_beside`].
    fn victim_beside(
        &self,
        tick: Tick,
        floor: Epoch,
        completed: Completed,
        skip_empty: bool,
    ) -> Option<(usize, bool)> {
        self.cache
            .victim_beside(tick, floor, completed, skip_empty)
            .map(|(s, e)| (s.index(), e))
    }

    /// The epoch of slot `s`'s last GPU reader or writer.
    fn reader_epoch(&self, s: usize) -> Epoch {
        self.cache.reader_epoch(SlotId(s as u32))
    }

    /// Slot `s` was read or written by a command buffer in `epoch`.
    fn stamp(&mut self, s: usize, epoch: Epoch) {
        self.cache.stamp(SlotId(s as u32), epoch)
    }

    /// A device plan in epoch `epoch` may read every slot.
    fn plan_read_all(&mut self, epoch: Epoch) {
        self.cache.plan_read_all(epoch)
    }

    /// The epoch of the last device plan.
    fn plan_epoch(&self) -> Epoch {
        self.cache.plan_epoch()
    }

    /// Slots in service ([`SlotCache::active`]).
    fn active(&self) -> usize {
        self.cache.active()
    }

    /// [`SlotCache::shrink_to`]: the experts to evict from the slots
    /// leaving service.
    fn shrink_to(&mut self, active: usize, floor: usize) -> Vec<u32> {
        self.cache
            .shrink_to(active, floor)
            .into_iter()
            .map(|e| e.0)
            .collect()
    }

    /// [`SlotCache::grow_to`]: the slots brought back into service.
    fn grow_to(&mut self, active: usize) -> Vec<SlotId> {
        self.cache.grow_to(active)
    }

    /// [`SlotCache::settled_past_active`].
    fn settled_past_active(&self) -> bool {
        self.cache.settled_past_active()
    }

    /// Evict the expert in `slot` ([`SlotCache::evict`]).
    fn evict(&mut self, slot: usize, epoch: Epoch, plan_open: bool) -> Option<u32> {
        self.cache
            .evict(SlotId(slot as u32), epoch, plan_open)
            .map(|e| e.0)
    }

    fn install(&mut self, slot: usize, e: usize) {
        self.cache.install(SlotId(slot as u32), ExpertId(e as u32))
    }

    /// Empty `slot` (its key out of the map), returning the key it held.
    fn uninstall(&mut self, slot: usize) -> Option<u32> {
        self.cache.uninstall(SlotId(slot as u32)).map(|e| e.0)
    }

    fn inflight(&self, slot: usize) -> u64 {
        self.cache.inflight(SlotId(slot as u32))
    }

    fn set_inflight(&mut self, slot: usize, value: u64) {
        self.cache.set_inflight(SlotId(slot as u32), value)
    }

    /// The largest in-flight value over the slots (0: none in flight).
    fn inflight_max(&self) -> u64 {
        self.cache.inflight_max()
    }

    /// Slots whose expert is in flight (an `io` hint not yet waited for).
    pub fn inflight_count(&self) -> usize {
        self.cache.inflight_count()
    }

    /// Forget the in-flight state of every slot whose loads landed (event
    /// value ≤ `done`): a landed expert is a resident like any other.
    /// Returns how many.
    fn settle_inflight(&mut self, done: u64) -> u64 {
        self.cache.settle_inflight(done)
    }

    fn set_pinned(&mut self, slot: usize, pinned: bool) {
        self.cache.set_pinned(SlotId(slot as u32), pinned)
    }

    fn is_pinned(&self, slot: usize) -> bool {
        self.cache.is_pinned(SlotId(slot as u32))
    }

    pub fn pinned_count(&self) -> usize {
        self.cache.pinned_count()
    }

    /// Slots a load can take without evicting (the arm's rule,
    /// [`SlotCache::free_slots`]).
    pub fn free_slots(&self) -> usize {
        self.cache.free_slots()
    }
}

/// Per-`Streamed` counters, relaxed atomics (an instrument, read off the
/// hot path).
#[derive(Debug, Default)]
pub struct StreamCounters {
    pub plans: AtomicU64,
    pub groups: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub bytes_loaded: AtomicU64,
    /// Device-side plans (`plan_device`): no readback at the layer.
    pub device_plans: AtomicU64,
    /// Settles that found the pass computed on a missing expert (the pass
    /// is replayed by the caller).
    pub settled_misses: AtomicU64,
    /// Misses read through the prefetcher's threads at the layer
    /// (`Prefetcher::load_now`) rather than one by one on this thread.
    pub prefetch_now: AtomicU64,
    /// Host nanoseconds: the ids readback in `plan` (the GPU drain), the
    /// `load_now` waits, the staged landings' encode, the synchronous
    /// loads — where a layer's wall goes beside its GPU time.
    pub readback_ns: AtomicU64,
    pub load_now_ns: AtomicU64,
    pub land_ns: AtomicU64,
    pub sync_load_ns: AtomicU64,
    /// The `io` arm's hints: experts issued to slots ahead of their layer.
    pub io_hints: AtomicU64,
    /// Uses that found their expert in flight (or landed) from a hint.
    pub io_inflight_hits: AtomicU64,
    /// In-flight entries overwritten before any use (read for nothing).
    pub io_inflight_wasted: AtomicU64,
    /// In-flight entries the host saw landed before any use waited for
    /// them (residents from then on, their use a plain hit).
    pub io_inflight_landed: AtomicU64,
    /// The `io` arm's in-flight entries a device plan waited for on the
    /// queue (the plan may map any of them), and the waits it encoded —
    /// one per plan with a load pending at encode time, none otherwise.
    pub io_inflight_planned: AtomicU64,
    pub plan_waits: AtomicU64,
    /// Misses read straight into their slots (`pread_direct_on`), and the
    /// reader threads' time waiting for the slots' last readers' epochs.
    pub direct_loads: AtomicU64,
    pub direct_wait_ns: AtomicU64,
    /// Misses a settle loaded, and the host time it spent loading them
    /// (the device plan's per-miss cost before any replay).
    pub settle_loads: AtomicU64,
    pub settle_ns: AtomicU64,
    /// Settles that loaded while the pool clock still showed a committed
    /// command buffer incomplete (its completion handler not yet run).
    pub settle_clock_lag: AtomicU64,
    /// The quarantined arm: misses landed in a free slot, and in a
    /// victim's slot in place because none was free; evictions quarantined,
    /// and slots recycled from the quarantine.
    pub free_takes: AtomicU64,
    pub inplace_takes: AtomicU64,
    pub quarantined: AtomicU64,
    pub recycled: AtomicU64,
    /// H1 (`profiles-and-memory-design.md` §4.7): routed kernels of a
    /// device-planned pass encoded in a later command buffer than the plan's
    /// — the slots' reader epoch moved up to the kernel's.
    pub h1_restamps: AtomicU64,
    /// H2: a load on the in-place arm that would have evicted a resident
    /// slot while a device plan was open, refused.
    pub h2_refusals: AtomicU64,
    /// H3: live-table writes deferred while a device plan was open,
    /// published at its settle.
    pub h3_deferred_writes: AtomicU64,
    /// Reallocations of the slot pools under the arbiter's yield (§4.2):
    /// shrinks and grows, and the bytes the shrinks released.
    pub slot_reallocations: AtomicU64,
    pub slot_bytes_released: AtomicU64,
    /// Experts the prefetcher staged on the hotlist's prediction and a
    /// device plan landed before mapping the pool, and those taken back
    /// (already resident, or no slot) — `QWEN4EXP_PREDICT=hotlist`.
    pub predict_landed: AtomicU64,
    pub predict_wasted: AtomicU64,
}

/// The fetch worker's counters (`nn::expert_fetch::FetchLane::stats`; zeros
/// off Metal or before a fetching step): jobs taken (one a layer a step),
/// layers that missed, experts loaded, experts left missing for want of a
/// slot, routings not seen in time, and the host time spent watching for
/// the routings and serving them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FetchStats {
    pub jobs: u64,
    pub missed_layers: u64,
    pub loads: u64,
    pub short: u64,
    pub timeouts: u64,
    pub watch_ns: u64,
    pub serve_ns: u64,
}

/// A staging ring's counters (the `pread` arm on Metal; zeros elsewhere).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StagingStats {
    /// Buffers the ring keeps.
    pub buffers: usize,
    /// Bytes those buffers hold.
    pub bytes: usize,
    pub allocations: u64,
    /// Acquisitions served by reuse.
    pub reuses: u64,
    /// One-shots handed out past the cap (held until their epoch completes).
    pub overflows: u64,
}

/// A snapshot of [`StreamCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    pub plans: u64,
    pub groups: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
    pub device_plans: u64,
    pub settled_misses: u64,
    pub prefetch_now: u64,
    pub readback_ns: u64,
    pub load_now_ns: u64,
    pub land_ns: u64,
    pub sync_load_ns: u64,
    pub io_hints: u64,
    pub io_inflight_hits: u64,
    pub io_inflight_wasted: u64,
    pub io_inflight_landed: u64,
    pub io_inflight_planned: u64,
    pub plan_waits: u64,
    pub direct_loads: u64,
    pub direct_wait_ns: u64,
    pub settle_loads: u64,
    pub settle_ns: u64,
    pub settle_clock_lag: u64,
    pub free_takes: u64,
    pub inplace_takes: u64,
    pub quarantined: u64,
    pub recycled: u64,
    pub h1_restamps: u64,
    pub h2_refusals: u64,
    pub h3_deferred_writes: u64,
    pub slot_reallocations: u64,
    pub slot_bytes_released: u64,
    pub predict_landed: u64,
    pub predict_wasted: u64,
}

impl StreamCounters {
    pub fn snapshot(&self) -> StreamStats {
        StreamStats {
            plans: self.plans.load(Ordering::Relaxed),
            groups: self.groups.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_loaded: self.bytes_loaded.load(Ordering::Relaxed),
            device_plans: self.device_plans.load(Ordering::Relaxed),
            settled_misses: self.settled_misses.load(Ordering::Relaxed),
            prefetch_now: self.prefetch_now.load(Ordering::Relaxed),
            readback_ns: self.readback_ns.load(Ordering::Relaxed),
            load_now_ns: self.load_now_ns.load(Ordering::Relaxed),
            land_ns: self.land_ns.load(Ordering::Relaxed),
            sync_load_ns: self.sync_load_ns.load(Ordering::Relaxed),
            io_hints: self.io_hints.load(Ordering::Relaxed),
            io_inflight_hits: self.io_inflight_hits.load(Ordering::Relaxed),
            io_inflight_wasted: self.io_inflight_wasted.load(Ordering::Relaxed),
            io_inflight_landed: self.io_inflight_landed.load(Ordering::Relaxed),
            io_inflight_planned: self.io_inflight_planned.load(Ordering::Relaxed),
            plan_waits: self.plan_waits.load(Ordering::Relaxed),
            direct_loads: self.direct_loads.load(Ordering::Relaxed),
            direct_wait_ns: self.direct_wait_ns.load(Ordering::Relaxed),
            settle_loads: self.settle_loads.load(Ordering::Relaxed),
            settle_ns: self.settle_ns.load(Ordering::Relaxed),
            settle_clock_lag: self.settle_clock_lag.load(Ordering::Relaxed),
            free_takes: self.free_takes.load(Ordering::Relaxed),
            inplace_takes: self.inplace_takes.load(Ordering::Relaxed),
            quarantined: self.quarantined.load(Ordering::Relaxed),
            recycled: self.recycled.load(Ordering::Relaxed),
            h1_restamps: self.h1_restamps.load(Ordering::Relaxed),
            h2_refusals: self.h2_refusals.load(Ordering::Relaxed),
            h3_deferred_writes: self.h3_deferred_writes.load(Ordering::Relaxed),
            slot_reallocations: self.slot_reallocations.load(Ordering::Relaxed),
            slot_bytes_released: self.slot_bytes_released.load(Ordering::Relaxed),
            predict_landed: self.predict_landed.load(Ordering::Relaxed),
            predict_wasted: self.predict_wasted.load(Ordering::Relaxed),
        }
    }
}

impl std::ops::AddAssign for StreamStats {
    fn add_assign(&mut self, o: Self) {
        self.device_plans += o.device_plans;
        self.settled_misses += o.settled_misses;
        self.prefetch_now += o.prefetch_now;
        self.readback_ns += o.readback_ns;
        self.load_now_ns += o.load_now_ns;
        self.land_ns += o.land_ns;
        self.sync_load_ns += o.sync_load_ns;
        self.io_hints += o.io_hints;
        self.io_inflight_hits += o.io_inflight_hits;
        self.io_inflight_wasted += o.io_inflight_wasted;
        self.io_inflight_landed += o.io_inflight_landed;
        self.io_inflight_planned += o.io_inflight_planned;
        self.plan_waits += o.plan_waits;
        self.direct_loads += o.direct_loads;
        self.direct_wait_ns += o.direct_wait_ns;
        self.settle_loads += o.settle_loads;
        self.settle_ns += o.settle_ns;
        self.settle_clock_lag += o.settle_clock_lag;
        self.free_takes += o.free_takes;
        self.inplace_takes += o.inplace_takes;
        self.quarantined += o.quarantined;
        self.recycled += o.recycled;
        self.h1_restamps += o.h1_restamps;
        self.h2_refusals += o.h2_refusals;
        self.h3_deferred_writes += o.h3_deferred_writes;
        self.slot_reallocations += o.slot_reallocations;
        self.slot_bytes_released += o.slot_bytes_released;
        self.predict_landed += o.predict_landed;
        self.predict_wasted += o.predict_wasted;
        self.plans += o.plans;
        self.groups += o.groups;
        self.hits += o.hits;
        self.misses += o.misses;
        self.evictions += o.evictions;
        self.bytes_loaded += o.bytes_loaded;
    }
}

/// A strategy that keeps streaming counters (`Streamed`); `Resident` reports
/// zeros so a harness can print one line for either.
pub trait HasStreamStats {
    fn stream_stats(&self) -> StreamStats;
    /// The staging ring's counters, if any.
    fn staging_stats(&self) -> StagingStats {
        StagingStats::default()
    }
    /// An identity for the staging ring, equal across the layers sharing it.
    fn staging_id(&self) -> Option<usize> {
        None
    }
    /// Slots per projection, 0 when nothing is streamed.
    fn slots(&self) -> usize {
        0
    }
    /// Uses per expert (the frequency profile); empty when nothing is streamed.
    fn usage(&self) -> Vec<u32> {
        Vec::new()
    }
    /// Warm the pool with `experts` in order (see `Streamed::warm`); the
    /// number loaded, 0 when nothing is streamed.
    fn warm(&self, _experts: &[u32]) -> Result<usize> {
        Ok(0)
    }

    /// Load `experts` and pin them — never evicted until `unpin_all` (see
    /// `Streamed::pin`: the static hot set, §20.3 R2); the number pinned, 0
    /// when nothing is streamed.
    fn pin(&self, _experts: &[u32]) -> Result<usize> {
        Ok(0)
    }
}

impl HasStreamStats for Resident {
    fn stream_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}

impl HasStreamStats for Streamed {
    fn stream_stats(&self) -> StreamStats {
        self.stats()
    }
    fn staging_stats(&self) -> StagingStats {
        Streamed::staging_stats(self)
    }
    fn staging_id(&self) -> Option<usize> {
        Streamed::staging_id(self)
    }
    fn slots(&self) -> usize {
        self.n_slots
    }
    fn usage(&self) -> Vec<u32> {
        Streamed::usage(self)
    }
    fn warm(&self, experts: &[u32]) -> Result<usize> {
        Streamed::warm(self, experts)
    }
    fn pin(&self, experts: &[u32]) -> Result<usize> {
        Streamed::pin(self, experts)?;
        Ok(experts.len())
    }
}

struct SlotPool {
    qt: QTensor,
    plane_bytes: usize,
}

struct Inner {
    /// The experts the last settle counted into the hotlist's usage (the
    /// record's distinct ids: the resident ones touched, the missing ones
    /// loaded), and whether the pass that follows is that token's REPLAY —
    /// the host plan over the same token, whose resolves must not count
    /// them again (#899's finding, `design/measurements/issue-746-mtp.md`:
    /// a device-planned step that misses counted its replayed token's
    /// experts twice, once at the settle and once at the replay's loads).
    /// `note_replay` arms it; the first resolve under it clears both.
    counted: Vec<u32>,
    replay_pending: bool,
    /// gate, up, down.
    pools: [SlotPool; 3],
    sources: [ExpertSource; 3],
    map: ResidencyMap,
    loader: LoaderArm,
    device: Device,
    /// The residency table on the device (`slot_of`, `NONE` = absent), for
    /// the device-side plan; rebuilt after a settle that loaded.
    table: Option<Tensor>,
    table_stale: bool,
    /// H3: live-table writes made while a device plan was open, published
    /// when the last one settles — a plan in flight reads the table it was
    /// built on.
    live_pending: Vec<(u32, u32)>,
    /// The `pread` arm's staging ring on Metal (reused once its copy's
    /// command buffer completes; never a pool bucket, never a wait) — one
    /// ring shared by every layer of the tier when built by the loader.
    #[cfg(feature = "metal")]
    staging: Option<lloom_tensor::kernels::metal::staging::SharedStagingRing>,
    /// The epoch of a group whose slot ids `ensure` handed out and whose
    /// dispatches are not all encoded yet (`down` closes it): the `io` arm's
    /// hint between the two must not give its slots away — the load is
    /// ordered after what IS encoded, nothing orders it after what is not.
    open_epoch: Option<Tick>,
    /// That group's slots: stamped with the GPU epoch at `down` (their last
    /// reader's command buffer), for the direct read's wait.
    open_slots: Vec<usize>,
    /// The pool clock (Metal), for those stamps.
    #[cfg(feature = "metal")]
    clock: Option<Arc<lloom_tensor::kernels::metal::GpuClock>>,
    /// The slot pools are Shared and a miss is read straight into its slot
    /// (`pread_direct_on`, the `pread` arm on Metal).
    direct: bool,
    /// The slot pools are Shared: the direct read's, or the fetch arm's
    /// (`fetch_on`), whose worker writes a slot from the host.
    shared_pools: bool,
    /// A settle's distinct ids and misses, kept between calls so a settle
    /// allocates nothing once they have grown.
    scratch_ids: Vec<u32>,
    scratch_missing: Vec<u32>,
    /// Device plans encoded and not yet settled. While one is open its pass
    /// may still map any resident slot, so no resident slot may be written
    /// and an eviction's quarantine waits for the settle.
    plans_open: u32,
    /// The residency table the device plan reads, in place (Metal, the
    /// quarantined arm on `pread`): an entry is written when its expert's
    /// bytes are ordered before every reader that can see it. `None`: the
    /// table is rebuilt from the map after a change (`table`).
    live: Option<LiveTable>,
    /// Entries whose bytes land by a copy encoded while a device plan was
    /// open: stored once the copy's epoch is complete.
    landing: Retire<(u32, u32)>,
    landed_scratch: Vec<(u32, u32)>,
    /// A one-row host plan's buffers (Metal), kept for the tier's life so a
    /// decode step asks no pool for them (`DESIGN.md` §9.2aa): the routing
    /// ids' readback destination, and the slot ids the routed kernels read.
    #[cfg(feature = "metal")]
    decode_io: DecodeIo,
}

/// See [`Inner::decode_io`].
#[cfg(feature = "metal")]
#[derive(Default)]
struct DecodeIo {
    ids_dst: Option<lloom_tensor::kernels::metal::BufferHandle>,
    /// The routing ids' host-read slot (#814 part 5): `moe_topk` writes a
    /// one-row step's ids straight into it and [`Streamed::plan`] reads it in
    /// place — no copy dispatch into `ids_dst`, no staging buffer. One per
    /// tier, so one per MoE layer: a layer's write never lands on another
    /// layer's value inside a step.
    ids_slot: Option<Arc<lloom_tensor::metal_backend::HostSlot>>,
    /// The device plan's record host-read slot: `SlotMap` writes a one-row
    /// step's record into it and the settle reads it in place (#814 part 5).
    record_slot: Option<Arc<lloom_tensor::metal_backend::HostSlot>>,
    slots: Option<(lloom_tensor::kernels::metal::BufferHandle, Tensor)>,
    /// The epoch of the command buffer the slot ids' last readers were
    /// encoded into (`down` stamps it), or `BUSY` from the write until then.
    slots_read_at: Option<Epoch>,
}

/// [`DecodeIo::slots_read_at`] between a write and its group's `down`.
#[cfg(feature = "metal")]
const BUSY: Epoch = Epoch(u64::MAX);

#[cfg(feature = "metal")]
impl Inner {
    /// The routing ids' host-read slot for a one-row step of `topk` ids,
    /// made on first use and kept for the tier's life.
    fn decode_ids_slot(
        &mut self,
        topk: usize,
    ) -> Result<Option<Arc<lloom_tensor::metal_backend::HostSlot>>> {
        let Ok(m) = self.device.as_metal_device() else {
            return Ok(None);
        };
        if let Some(s) = &self.decode_io.ids_slot {
            if s.cap() >= topk {
                return Ok(Some(Arc::clone(s)));
            }
        }
        let slot = Arc::new(lloom_tensor::metal_backend::HostSlot::new(
            m,
            topk,
            lloom_tensor::DType::U32,
        )?);
        self.decode_io.ids_slot = Some(Arc::clone(&slot));
        Ok(Some(slot))
    }

    /// The device plan's record slot for a one-row step's `len` words, made
    /// on first use and kept for the tier's life.
    fn decode_record_slot(
        &mut self,
        len: usize,
    ) -> Result<Option<Arc<lloom_tensor::metal_backend::HostSlot>>> {
        let Ok(m) = self.device.as_metal_device() else {
            return Ok(None);
        };
        if let Some(s) = &self.decode_io.record_slot {
            if s.cap() >= len {
                return Ok(Some(Arc::clone(s)));
            }
        }
        let slot = Arc::new(lloom_tensor::metal_backend::HostSlot::new(
            m,
            len,
            lloom_tensor::DType::U32,
        )?);
        self.decode_io.record_slot = Some(Arc::clone(&slot));
        Ok(Some(slot))
    }

    /// The destination a one-row plan's routing ids are read back into:
    /// the tier's own Shared buffer of at least `bytes`, so the readback asks
    /// no pool for one per step.
    fn decode_ids_dst(
        &mut self,
        bytes: usize,
    ) -> Result<Option<lloom_tensor::kernels::metal::BufferHandle>> {
        let Ok(m) = self.device.as_metal_device() else {
            return Ok(None);
        };
        Ok(Some(match &self.decode_io.ids_dst {
            Some(b) if b.length() >= bytes => b.clone(),
            _ => {
                let _escape = escape_guard();
                let b = m.new_persistent_shared(bytes)?;
                self.decode_io.ids_dst = Some(b.clone());
                b
            }
        }))
    }

    /// A one-row group's slot ids written into the tier's own Shared buffer,
    /// once the command buffer holding the last ids' readers has completed
    /// (`slots_read_at`, stamped by `down` after a group's last dispatch).
    /// `None` — the caller uploads through the pool, as always — while a
    /// reader may still run: another group of the same call, whose
    /// dispatches sit in the open command buffer, or any reader not yet
    /// retired.
    fn write_decode_slots(&mut self, slot_ids: &[u32]) -> Result<Option<Tensor>> {
        let (Some(clock), Ok(m)) = (&self.clock, self.device.as_metal_device()) else {
            return Ok(None);
        };
        if let Some(e) = self.decode_io.slots_read_at {
            if e == BUSY || !clock.is_complete(e) {
                return Ok(None);
            }
        }
        let n = slot_ids.len();
        let fits = matches!(&self.decode_io.slots, Some((_, t)) if t.dims() == [1, n]);
        if !fits {
            let _escape = escape_guard();
            let buf = m.new_persistent_shared(std::mem::size_of_val(slot_ids))?;
            if buf.contents().is_null() {
                return Ok(None);
            }
            let storage = lloom_tensor::Storage::Metal(lloom_tensor::MetalStorage::new(
                buf.clone(),
                m.clone(),
                n,
                lloom_tensor::DType::U32,
            ));
            let t =
                Tensor::from_storage(storage, (1, n), lloom_tensor::op::BackpropOp::none(), false);
            self.decode_io.slots = Some((buf, t));
        }
        let (buf, t) = self.decode_io.slots.as_ref().expect("made above");
        // SAFETY: a CPU-mapped Shared buffer of `n` u32 (checked when made);
        // the command buffer holding its last readers has completed (above),
        // and nothing on the device writes it.
        unsafe {
            std::ptr::copy_nonoverlapping(slot_ids.as_ptr(), buf.contents() as *mut u32, n);
        }
        self.decode_io.slots_read_at = Some(BUSY);
        Ok(Some(t.clone()))
    }
}

/// The residency table in a persistent Shared buffer, written by the host
/// in place (`expert → slot`, `NONE` = absent).
struct LiveTable {
    tensor: Tensor,
    #[cfg(feature = "metal")]
    buf: lloom_tensor::kernels::metal::BufferHandle,
    n: usize,
}

impl LiveTable {
    #[cfg(feature = "metal")]
    fn new(device: &lloom_tensor::MetalDevice, n: usize) -> Result<Self> {
        let buf = device.new_persistent_shared(n * std::mem::size_of::<u32>())?;
        if buf.contents().is_null() {
            lloom_tensor::bail!("the live residency table has no CPU mapping")
        }
        let storage = lloom_tensor::Storage::Metal(lloom_tensor::MetalStorage::new(
            buf.clone(),
            device.clone(),
            n,
            lloom_tensor::DType::U32,
        ));
        let tensor = Tensor::from_storage(storage, n, lloom_tensor::op::BackpropOp::none(), false);
        let t = Self { tensor, buf, n };
        for e in 0..n {
            t.store(e, NONE);
        }
        Ok(t)
    }

    /// `table[e] = v`.
    #[cfg(feature = "metal")]
    fn store(&self, e: usize, v: u32) {
        assert!(e < self.n, "live table: expert {e} out of {}", self.n);
        // SAFETY: a CPU-mapped Shared buffer of `n` u32 (checked at `new`),
        // `e < n`. Nothing on the device writes it; a kernel reading the word
        // sees the old value or the new one (an aligned u32 store).
        unsafe { (self.buf.contents() as *mut u32).add(e).write_volatile(v) }
    }

    #[cfg(not(feature = "metal"))]
    fn store(&self, _e: usize, _v: u32) {}
}

impl Inner {
    /// Allocate the three plane pools at `active` slots and carry the
    /// first `keep` slots' planes over (a blit on Metal; on the CPU the
    /// pools come back empty and the caller evicts everything). The old
    /// pools drop here: their handles' reservations leave the ledger at
    /// once, the residency set forgets them at the next cut. Returns the
    /// pools' bytes before and after.
    fn reallocate_pools(&mut self, active: usize, keep: usize) -> Result<(usize, usize)> {
        let before: usize = self
            .pools
            .iter()
            .map(|p| p.qt.storage_size_in_bytes())
            .sum();
        let mut fresh = Vec::with_capacity(3);
        for (i, src) in self.sources.iter().enumerate() {
            let (n, k) = src.dims();
            let qt = QTensor::zeros_persistent_with(
                (active, n, k),
                src.dtype(),
                &self.device,
                self.direct,
            )?;
            #[cfg(not(feature = "metal"))]
            let _ = keep;
            #[cfg(feature = "metal")]
            if let Device::Metal(m) = &self.device {
                let bytes = keep * self.pools[i].plane_bytes;
                if bytes > 0 {
                    let (old, new) = (self.pools[i].qt.metal_buffer()?, qt.metal_buffer()?);
                    let mut blit = m.blit_command_encoder()?;
                    blit.set_label("slot pools reallocated");
                    blit.copy_from_buffer(old, 0, new, 0, bytes);
                }
            }
            fresh.push(SlotPool {
                qt,
                plane_bytes: self.pools[i].plane_bytes,
            });
        }
        // The copies are encoded on the queue, which retains the old pools
        // until they run; the handles may drop now (their reservations
        // leave the ledger at once, the residency set forgets them at the
        // next cut). No wait here: the tier never drains the device
        // (`qwen4exp_eviction_no_drain`); the yield's one host wait is the
        // admission's, before it asks (`Qwen4Exp::apply_expert_grant`).
        let fresh: [SlotPool; 3] = fresh
            .try_into()
            .map_err(|_| lloom_tensor::Error::Msg("three slot pools".into()))?;
        self.pools = fresh;
        let after: usize = self
            .pools
            .iter()
            .map(|p| p.qt.storage_size_in_bytes())
            .sum();
        Ok((before, after))
    }

    /// Store the landed entries whose copies' epochs have completed (and
    /// whose expert is still in that slot).
    fn store_landed(&mut self) {
        if self.landing.is_empty() {
            return;
        }
        let done = self.completed_epochs();
        let mut ready = std::mem::take(&mut self.landed_scratch);
        self.landing.drain_into(done, &mut ready);
        for &(e, slot) in &ready {
            if self.map.slot_of(e as usize) == Some(slot as usize) {
                self.table_set(e, slot);
            }
        }
        ready.clear();
        self.landed_scratch = ready;
    }

    /// Record `e → slot` in the device's table: in place when a table is
    /// live, else by marking it for a rebuild.
    fn table_set(&mut self, e: u32, slot: u32) {
        match &self.live {
            Some(_) if self.plans_open > 0 => {
                // H3: a device plan in flight reads this table on the GPU;
                // the write waits for its settle.
                self.live_pending.push((e, slot));
            }
            Some(t) => {
                debug_assert_eq!(
                    self.plans_open, 0,
                    "H3: a live-table write under an open plan"
                );
                t.store(e as usize, slot)
            }
            None => self.table_stale = true,
        }
    }

    /// Write `e → slot` into the live table NOW, whatever plan is open, and
    /// drop any write for `e` deferred under one (`table_set`'s H3): the
    /// fetch worker's, whose pass reads the table only after it is done.
    fn live_store(&mut self, e: u32, slot: u32) {
        self.live_pending.retain(|&(x, _)| x != e);
        if let Some(t) = &self.live {
            t.store(e as usize, slot);
        }
    }

    /// Every device plan has settled: deferred evictions retire at this
    /// epoch (H2) and the live-table writes made under the plans are
    /// published (H3). Returns the writes published.
    fn plans_settled(&mut self) -> u64 {
        let epoch = self.current_epoch();
        self.map.release_deferred(epoch);
        let n = self.live_pending.len() as u64;
        if let Some(t) = &self.live {
            for &(e, slot) in &self.live_pending {
                t.store(e as usize, slot);
            }
        }
        self.live_pending.clear();
        n
    }
}

/// A resolve between its halves (`Streamed::resolve_begin` →
/// `resolve_finish`): each expert's slot, and the misses' reads in flight.
struct Pending {
    slots: Vec<usize>,
    /// Misses read synchronously at the finish.
    misses: Vec<(usize, usize)>,
    /// Every `(expert, slot)` installed, and the experts whose bytes the
    /// reader threads write straight into their slots: the device table
    /// learns each once its bytes are ordered before any reader of it.
    installed: Vec<(u32, u32)>,
    direct_landed: Vec<u32>,
    #[cfg(feature = "metal")]
    landed: Vec<(usize, usize, crate::nn::expert_prefetch::Staged)>,
    #[cfg(feature = "metal")]
    wait_value: u64,
    #[cfg(feature = "metal")]
    direct: Option<DirectInFlight>,
    #[cfg(feature = "metal")]
    now: Option<NowInFlight>,
}

/// Direct reads submitted: the ticket, the experts, the submit time.
#[cfg(feature = "metal")]
type DirectInFlight = (
    crate::nn::expert_prefetch::DirectTicket,
    u64,
    std::time::Duration,
);

/// Ring reads submitted: the ticket, the `(expert, slot)`s asked for, the
/// submit time.
#[cfg(feature = "metal")]
type NowInFlight = (
    crate::nn::expert_prefetch::NowTicket,
    Vec<(usize, usize)>,
    std::time::Duration,
);

/// The routed experts streamed from the checkpoint into a slot pool (§20.6).
pub struct Streamed {
    inner: parking_lot::Mutex<Inner>,
    counters: StreamCounters,
    n_experts: usize,
    n_slots: usize,
    /// The tier's prefetcher, shared by its layers (`set_prefetcher`), and
    /// this layer's key in it.
    #[cfg(feature = "metal")]
    prefetch: Option<Arc<crate::nn::expert_prefetch::Prefetcher>>,
    /// The `io` arm's hints (`set_io_hints`): a prefetch hint chooses the
    /// slots NOW and issues the loads on the IO queue, the residency map
    /// carrying the entries in flight until the layer's use waits for them.
    #[cfg(feature = "metal")]
    io_hints: bool,
    key: usize,
}

/// Distinct per `Streamed` for the life of the process (the prefetcher's
/// tier key).
static NEXT_KEY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl std::fmt::Debug for Streamed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Streamed {{ experts: {}, slots: {}, {:?} }}",
            self.n_experts,
            self.n_slots,
            self.counters.snapshot()
        )
    }
}

impl Streamed {
    /// A pool of `n_slots` experts per projection over the three sources
    /// (gate, up, down — the same expert count each), filled by `loader`.
    pub fn new(
        gate: ExpertSource,
        up: ExpertSource,
        down: ExpertSource,
        n_slots: usize,
        loader: LoaderArm,
        device: &Device,
    ) -> Result<Self> {
        #[cfg(feature = "metal")]
        let staging = match (device, &loader) {
            (Device::Metal(d), LoaderArm::Pread) => Some(
                lloom_tensor::kernels::metal::staging::StagingRing::shared(d),
            ),
            _ => None,
        };
        #[cfg(not(feature = "metal"))]
        let staging = None::<()>;
        Self::with_staging(gate, up, down, n_slots, loader, device, staging)
    }

    /// [`Self::new`] over a staging ring shared with other layers (the
    /// `pread` arm on Metal; `None` elsewhere, and the arm stages one-shot).
    pub fn with_staging(
        gate: ExpertSource,
        up: ExpertSource,
        down: ExpertSource,
        n_slots: usize,
        loader: LoaderArm,
        device: &Device,
        #[cfg(feature = "metal")] staging: Option<
            lloom_tensor::kernels::metal::staging::SharedStagingRing,
        >,
        #[cfg(not(feature = "metal"))] staging: Option<()>,
    ) -> Result<Self> {
        #[cfg(feature = "metal")]
        let (direct, shared) = (pread_direct_on(), pread_direct_on() || fetch_on());
        #[cfg(not(feature = "metal"))]
        let (direct, shared) = (false, false);
        Self::with_staging_pools(
            gate, up, down, n_slots, loader, device, staging, direct, shared,
        )
    }

    /// [`Self::with_staging`] with the direct read chosen here rather than
    /// by `LLOOM_PREAD_DIRECT` (Shared slot pools a miss is read straight
    /// into, on Metal's `pread` arm).
    #[allow(clippy::too_many_arguments)]
    pub fn with_staging_direct(
        gate: ExpertSource,
        up: ExpertSource,
        down: ExpertSource,
        n_slots: usize,
        loader: LoaderArm,
        device: &Device,
        #[cfg(feature = "metal")] staging: Option<
            lloom_tensor::kernels::metal::staging::SharedStagingRing,
        >,
        #[cfg(not(feature = "metal"))] staging: Option<()>,
        direct: bool,
    ) -> Result<Self> {
        Self::with_staging_pools(
            gate, up, down, n_slots, loader, device, staging, direct, direct,
        )
    }

    /// [`Self::with_staging_direct`] with the pools' storage chosen apart
    /// from the direct read: `shared` makes them Shared (the fetch arm's
    /// worker writes a slot from the host) whether or not a miss on the
    /// encoding thread is read straight into its slot (`direct`).
    #[allow(clippy::too_many_arguments)]
    fn with_staging_pools(
        gate: ExpertSource,
        up: ExpertSource,
        down: ExpertSource,
        n_slots: usize,
        loader: LoaderArm,
        device: &Device,
        #[cfg(feature = "metal")] staging: Option<
            lloom_tensor::kernels::metal::staging::SharedStagingRing,
        >,
        #[cfg(not(feature = "metal"))] staging: Option<()>,
        direct: bool,
        shared: bool,
    ) -> Result<Self> {
        #[cfg(not(feature = "metal"))]
        let _ = (staging, direct, shared);
        let n_experts = gate.n_experts();
        if up.n_experts() != n_experts || down.n_experts() != n_experts {
            lloom_tensor::bail!(
                "expert counts differ: gate {} up {} down {}",
                n_experts,
                up.n_experts(),
                down.n_experts()
            )
        }
        if n_slots == 0 || n_slots > n_experts {
            lloom_tensor::bail!("slot pool of {n_slots} for {n_experts} experts")
        }
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut sources = [gate, up, down];
        #[cfg(feature = "metal")]
        if let LoaderArm::Io(q) = &loader {
            for s in sources.iter_mut() {
                s.io = Some(q.open(&s.path).map_err(lloom_tensor::Error::wrap)?);
            }
        }
        // Persistent storage of known geometry: never a pool bucket. Shared
        // under the direct read and the fetch arm (the CPU writes a miss
        // into its slot).
        #[cfg(feature = "metal")]
        let pread_metal = matches!(loader, LoaderArm::Pread) && matches!(device, Device::Metal(_));
        #[cfg(feature = "metal")]
        let (direct, shared) = (direct && pread_metal, (shared || direct) && pread_metal);
        #[cfg(not(feature = "metal"))]
        let (direct, shared) = (false, false);
        let pool = |s: &ExpertSource| -> Result<SlotPool> {
            let (n, k) = s.dims();
            Ok(SlotPool {
                qt: QTensor::zeros_persistent_with((n_slots, n, k), s.dtype(), device, shared)?,
                plane_bytes: s.plane_bytes(),
            })
        };
        let pools = [pool(&sources[0])?, pool(&sources[1])?, pool(&sources[2])?];
        #[cfg(feature = "metal")]
        let clock = match device {
            Device::Metal(d) => Some(d.gpu_clock()),
            _ => None,
        };
        // Every slot was last written by the pools' zeroing, encoded just
        // now: a direct read (the CPU writing a slot outside the command
        // stream) must wait for it like for any other writer, or the blit
        // still pending would zero what the read landed.
        let mut map = ResidencyMap::new(n_experts, n_slots);
        #[cfg(feature = "metal")]
        map.plan_read_all(clock.as_ref().map_or(Epoch::ZERO, |c| c.current_epoch()));
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner {
                pools,
                sources,
                map,
                loader,
                device: device.clone(),
                #[cfg(feature = "metal")]
                staging,
                table: None,
                table_stale: true,
                live_pending: Vec::new(),
                open_epoch: None,
                open_slots: Vec::new(),
                #[cfg(feature = "metal")]
                clock,
                direct,
                shared_pools: shared,
                scratch_ids: Vec::new(),
                scratch_missing: Vec::new(),
                counted: Vec::new(),
                replay_pending: false,
                plans_open: 0,
                live: None,
                landing: Retire::new(),
                landed_scratch: Vec::new(),
                #[cfg(feature = "metal")]
                decode_io: DecodeIo::default(),
            }),
            counters: StreamCounters::default(),
            n_experts,
            n_slots,
            #[cfg(feature = "metal")]
            prefetch: None,
            #[cfg(feature = "metal")]
            io_hints: false,
            key: NEXT_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        })
        .and_then(|mut t| {
            t.set_spare_slots(default_spare_slots())?;
            Ok(t)
        })
    }

    /// Keep `k` slots free (the quarantined arm: a miss lands in a free
    /// slot, an eviction is quarantined until its readers complete); 0 is
    /// the in-place arm. A live device table comes with the quarantined arm
    /// on Metal's `pread` loader. Only on a pool that has loaded nothing.
    pub fn set_spare_slots(&mut self, k: usize) -> Result<()> {
        let inner = self.inner.get_mut();
        if inner.map.any_resident() {
            lloom_tensor::bail!("set_spare_slots on a pool that already holds experts")
        }
        inner.map.set_spare(k);
        inner.live = None;
        // The fetch arm's worker tells the device of a load mid-pass: the
        // table is the one the device reads, written in place.
        #[cfg(feature = "metal")]
        if (inner.map.quarantined() || (fetch_on() && inner.shared_pools))
            && matches!(inner.loader, LoaderArm::Pread)
        {
            if let Device::Metal(d) = &inner.device {
                inner.live = Some(LiveTable::new(d, self.n_experts)?);
            }
        }
        inner.table = None;
        inner.table_stale = true;
        Ok(())
    }

    /// Slots kept free (0: the in-place arm).
    pub fn spare_slots(&self) -> usize {
        self.inner.lock().map.spare()
    }

    /// Bytes of the three planes per slot.
    pub fn bytes_per_slot(&self) -> usize {
        self.inner.lock().pools.iter().map(|p| p.plane_bytes).sum()
    }

    /// Slots evicted and not yet reusable.
    pub fn held_slots(&self) -> usize {
        self.inner.lock().map.held()
    }

    /// Whether the device plan reads a table written in place.
    pub fn live_table(&self) -> bool {
        self.inner.lock().live.is_some()
    }

    /// Attach the tier's prefetcher (the `pread` arm on Metal; a loader
    /// builds one per tier and hands every layer the same).
    #[cfg(feature = "metal")]
    pub fn set_prefetcher(&mut self, p: Arc<crate::nn::expert_prefetch::Prefetcher>) {
        self.prefetch = Some(p);
    }

    /// Let the prefetch hints act on the `io` arm: the hinted experts are
    /// given slots at hint time and their loads issued on the IO queue,
    /// waited for on the compute stream only at the layer that uses them
    /// (the in-flight residency entry, §20.4a). No effect on the `pread`
    /// arm (its hints go to the prefetcher's ring).
    #[cfg(feature = "metal")]
    pub fn set_io_hints(&mut self, on: bool) {
        self.io_hints = on;
    }

    #[cfg(feature = "metal")]
    pub fn io_hints(&self) -> bool {
        self.io_hints
    }

    /// Slots in flight from `io` hints, not yet waited for.
    pub fn inflight_count(&self) -> usize {
        self.inner.lock().map.inflight_count()
    }

    /// The tier key the prefetcher files this layer's requests under.
    pub fn prefetch_key(&self) -> usize {
        self.key
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn stats(&self) -> StreamStats {
        self.counters.snapshot()
    }

    /// The loader arm's name.
    pub fn loader_name(&self) -> &'static str {
        self.inner.lock().loader.name()
    }

    /// Resident bytes across the three pools.
    pub fn pool_bytes(&self) -> usize {
        let inner = self.inner.lock();
        inner
            .pools
            .iter()
            .map(|p| p.qt.storage_size_in_bytes())
            .sum()
    }

    /// The staging ring's `(buffers, bytes, allocations, reuses, overflows)`
    /// — zeros where there is no ring (the CPU, the io arm).
    pub fn staging_stats(&self) -> StagingStats {
        #[cfg(feature = "metal")]
        {
            let inner = self.inner.lock();
            if let Some(r) = &inner.staging {
                let r = r.lock();
                let (a, u, o) = r.counters();
                return StagingStats {
                    buffers: r.len(),
                    bytes: r.bytes(),
                    allocations: a,
                    reuses: u,
                    overflows: o,
                };
            }
        }
        StagingStats::default()
    }

    /// An identity for the staging ring, shared rings equal — so a model
    /// reports a ring once, not once per layer.
    pub fn staging_id(&self) -> Option<usize> {
        #[cfg(feature = "metal")]
        {
            let inner = self.inner.lock();
            inner
                .staging
                .as_ref()
                .map(|r| std::sync::Arc::as_ptr(r) as usize)
        }
        #[cfg(not(feature = "metal"))]
        None
    }

    /// The resident experts, in slot order.
    pub fn resident(&self) -> Vec<u32> {
        self.inner.lock().map.resident()
    }

    /// Uses per expert since construction (the frequency profile: a hit or
    /// a miss each time a pass needed the expert).
    pub fn usage(&self) -> Vec<u32> {
        self.inner.lock().map.usage.clone()
    }

    /// Warm the pool: load `experts` in order into the FREE slots — the
    /// head of a frequency profile (§20.3 R2) before the first token —
    /// evicting nothing and pinning nothing (the pool's policy runs as it
    /// would over a pool that had seen them). Stops at the first expert
    /// past the free slots; already-resident experts are skipped. Returns
    /// the experts loaded.
    pub fn warm(&self, experts: &[u32]) -> Result<usize> {
        let _escape = escape_guard();
        let mut inner = self.inner.lock();
        let mut take: Vec<u32> = Vec::new();
        let mut free = inner.map.free_slots();
        for &e in experts {
            if free == 0 {
                break;
            }
            if e as usize >= self.n_experts {
                lloom_tensor::bail!("warm: expert {e} out of {}", self.n_experts)
            }
            if inner.map.slot_of(e as usize).is_some() || take.contains(&e) {
                continue;
            }
            take.push(e);
            free -= 1;
        }
        if take.is_empty() {
            return Ok(0);
        }
        let tick = inner.map.begin();
        // Loaded as one group (one staging pass); a warm is not a use.
        let before: Vec<u32> = inner.map.usage.clone();
        let pending = self.resolve_begin(&mut inner, &take, tick, false)?;
        self.resolve_finish(&mut inner, pending)?;
        inner.map.usage = before;
        // The `io` arm: no more than two layers' warms pending on the queue
        // (their bytes are held on the device while they wait).
        #[cfg(feature = "metal")]
        if let LoaderArm::Io(q) = &inner.loader {
            q.wait_all_but_latest();
        }
        Ok(take.len())
    }

    /// Load `experts` and pin them: they are never evicted until
    /// [`Self::unpin_all`]. The static hot set (§20.3 R2).
    pub fn pin(&self, experts: &[u32]) -> Result<()> {
        if experts.len() >= self.n_slots {
            lloom_tensor::bail!(
                "pinning {} experts leaves no free slot of {}",
                experts.len(),
                self.n_slots
            )
        }
        let mut inner = self.inner.lock();
        let tick = inner.map.begin();
        let slots = self.resolve(&mut inner, experts, tick)?;
        for s in slots {
            inner.map.set_pinned(s, true);
        }
        Ok(())
    }

    pub fn unpin_all(&self) {
        let mut inner = self.inner.lock();
        for s in 0..self.n_slots {
            inner.map.set_pinned(s, false);
        }
    }

    /// Make every expert of `experts` resident within epoch `tick`, loading
    /// the misses; returns each one's slot.
    fn resolve(&self, inner: &mut Inner, experts: &[u32], tick: Tick) -> Result<Vec<usize>> {
        let pending = self.resolve_begin(inner, experts, tick, true)?;
        self.resolve_finish(inner, pending)
    }

    /// [`Self::resolve`]'s first half: every expert given its slot and the
    /// map updated, the staged ones taken, and the misses' reads SUBMITTED
    /// to the reader threads — not waited for. [`Self::resolve_finish`]
    /// waits for them and lands them; between the two another layer's
    /// reads can be submitted, so a token's misses read together
    /// (`settle_batch`).
    /// On the quarantined arm a miss takes a free slot, and `replenish`
    /// then evicts LRU experts into the quarantine until `spare` slots are
    /// free or held again (a warm, which evicts nothing, does not).
    fn resolve_begin(
        &self,
        inner: &mut Inner,
        experts: &[u32],
        tick: Tick,
        replenish: bool,
    ) -> Result<Pending> {
        let mut slots = Vec::with_capacity(experts.len());
        let mut misses: Vec<(usize, usize)> = Vec::new();
        // What the prefetcher staged for this layer: a miss with a staged
        // expert lands it (the copy alone) instead of reading it here.
        #[cfg(feature = "metal")]
        let mut staged: std::collections::HashMap<u32, crate::nn::expert_prefetch::Staged> =
            match &self.prefetch {
                Some(p) => {
                    let needed: Vec<u32> = experts
                        .iter()
                        .copied()
                        .filter(|&e| {
                            (e as usize) < self.n_experts && inner.map.slot_of(e as usize).is_none()
                        })
                        .collect();
                    p.take_needed(self.key, &needed)
                        .into_iter()
                        .map(|s| (s.expert, s))
                        .collect()
                }
                None => Default::default(),
            };
        #[cfg(feature = "metal")]
        let mut landed: Vec<(usize, usize, crate::nn::expert_prefetch::Staged)> = Vec::new();
        // The `io` arm's in-flight entries this group uses: the loads
        // already landed (the event past their value) need no wait; the
        // rest are waited for once, below, before the group's dispatch.
        #[cfg(feature = "metal")]
        let mut wait_value = 0u64;
        // Under the direct read, a victim whose last reader has completed
        // costs the reader thread no wait: prefer those.
        let settled_before = inner.completed_epochs();
        if inner.map.quarantined() {
            let n = inner.map.recycle(settled_before);
            self.counters
                .recycled
                .fetch_add(n as u64, Ordering::Relaxed);
        }
        // Every expert installed here, for the table once its bytes are
        // ordered before its readers.
        let mut installed: Vec<(u32, u32)> = Vec::new();
        #[cfg(feature = "metal")]
        if inner.map.inflight_count() > 0 {
            if let LoaderArm::Io(q) = &inner.loader {
                let done = q.signaled();
                let n = inner.map.settle_inflight(done);
                self.counters
                    .io_inflight_landed
                    .fetch_add(n, Ordering::Relaxed);
            }
        }
        // A replay's resolve: the settle counted this token's experts
        // already; those are not counted again, the rest (a layer past the
        // miss, whose true experts the replay finds) are. One resolve clears
        // it: the replay visits each layer's tier once.
        let replaying = std::mem::take(&mut inner.replay_pending);
        let counted = if replaying {
            std::mem::take(&mut inner.counted)
        } else {
            Vec::new()
        };
        for &e in experts {
            let e = e as usize;
            if e >= self.n_experts {
                lloom_tensor::bail!("expert {e} out of {}", self.n_experts)
            }
            if !(replaying && counted.contains(&(e as u32))) {
                inner.map.usage[e] = inner.map.usage[e].saturating_add(1);
            }
            let slot = match inner.map.slot_of(e) {
                Some(s) => {
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "metal")]
                    if inner.map.inflight(s) != 0 {
                        wait_value = wait_value.max(inner.map.inflight(s));
                        inner.map.set_inflight(s, 0);
                        self.counters
                            .io_inflight_hits
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    s
                }
                None => {
                    let free = if inner.map.quarantined() {
                        inner.map.take_free()
                    } else {
                        None
                    };
                    let (s, evicted) = match free {
                        Some(s) => {
                            self.counters.free_takes.fetch_add(1, Ordering::Relaxed);
                            (s, false)
                        }
                        // A resident slot is written only while no device
                        // plan is open: an open plan's pass may map any.
                        None if inner.map.quarantined() && inner.plans_open > 0 => {
                            self.counters.h2_refusals.fetch_add(1, Ordering::Relaxed);
                            lloom_tensor::bail!(
                                "no free slot of {} ({} held) for a load while a device plan is open",
                                self.n_slots,
                                inner.map.held()
                            )
                        }
                        // No free slot, no plan open: a RESIDENT victim in
                        // place (an empty slot here is quarantined — its
                        // readers may not have run).
                        None if inner.map.quarantined() => match inner.map.lru_resident(tick) {
                            Some(s) => {
                                self.counters.inplace_takes.fetch_add(1, Ordering::Relaxed);
                                (s, true)
                            }
                            None => lloom_tensor::bail!(
                                "the slot pool ({} slots, {} pinned, {} held) cannot hold this group's {} experts",
                                self.n_slots,
                                inner.map.pinned_count(),
                                inner.map.held(),
                                experts.len()
                            ),
                        },
                        None => match inner.map.victim_settled(tick, settled_before) {
                            // H2: a device plan in flight may read any slot;
                            // the in-place arm cannot defer the overwrite, so
                            // it refuses, as the quarantined arm does above.
                            Some((s, true)) if inner.plans_open > 0 => {
                                self.counters.h2_refusals.fetch_add(1, Ordering::Relaxed);
                                lloom_tensor::bail!(
                                    "no empty slot of {} for a load while a device plan is open \
                                     (slot {s} holds expert {}, which the plan may read)",
                                    self.n_slots,
                                    inner.map.expert_in(s)
                                )
                            }
                            Some(v) => v,
                            None => lloom_tensor::bail!(
                                "the slot pool ({} slots, {} pinned) cannot hold this group's {} experts",
                                self.n_slots,
                                inner.map.pinned_count(),
                                experts.len()
                            ),
                        },
                    };
                    if evicted {
                        // The victim's entry goes before its slot is written.
                        debug_assert_eq!(inner.plans_open, 0, "H2: an eviction under an open plan");
                        let old = inner.map.expert_in(s);
                        inner.table_set(old, NONE);
                    }
                    installed.push((e as u32, s as u32));
                    if inner.map.inflight(s) != 0 {
                        self.counters
                            .io_inflight_wasted
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    inner.map.install(s, e);
                    self.counters.misses.fetch_add(1, Ordering::Relaxed);
                    if evicted {
                        self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    #[cfg(feature = "metal")]
                    if let Some(st) = staged.remove(&(e as u32)) {
                        landed.push((e, s, st));
                        inner.map.touch(s, tick);
                        slots.push(s);
                        continue;
                    }
                    misses.push((e, s));
                    s
                }
            };
            inner.map.touch(slot, tick);
            slots.push(slot);
        }
        if replenish && inner.map.quarantined() && !installed.is_empty() {
            let epoch = inner.current_epoch();
            let open = inner.plans_open > 0;
            while inner.map.free_len() + inner.map.held() < inner.map.spare() {
                let Some(old) = inner.map.evict_lru(tick, epoch, open) else {
                    break;
                };
                inner.table_set(old, NONE);
                self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                self.counters.quarantined.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(feature = "metal")]
        let mut pending = Pending {
            slots,
            misses: Vec::new(),
            installed,
            direct_landed: Vec::new(),
            landed,
            wait_value,
            direct: None,
            now: None,
        };
        #[cfg(not(feature = "metal"))]
        let pending = Pending {
            slots,
            misses: Vec::new(),
            installed,
            direct_landed: Vec::new(),
        };
        #[cfg(feature = "metal")]
        {
            if !staged.is_empty() {
                if let Some(p) = &self.prefetch {
                    p.release(staged.into_values().collect());
                }
            }
            // The misses nothing staged, under the direct read: those whose
            // slot's last reader has already completed go straight into
            // their Shared slots through the reader threads — no ring, no
            // landing copy (§20.4a); the rest (the GPU has not run their
            // slot's last reader yet: the host is ahead of it, which is the
            // pipelined prefill's whole point) take the ring and the copy,
            // ordered behind that reader on the queue, so the host never
            // waits for the GPU here.
            if !misses.is_empty() && inner.direct {
                if let Some(p) = &self.prefetch {
                    let t0 = std::time::Instant::now();
                    let settled = inner.completed_epochs();
                    let (now, later): (Vec<_>, Vec<_>) = misses
                        .iter()
                        .copied()
                        .partition(|&(_, s)| settled.covers(inner.map.reader_epoch(s)));
                    misses = later;
                    let mut jobs = Vec::with_capacity(now.len());
                    let mut bytes = 0u64;
                    for &(e, s) in &now {
                        let mut dsts =
                            [crate::nn::expert_prefetch::DirectDst(std::ptr::null_mut()); 3];
                        for (i, pool) in inner.pools.iter().enumerate() {
                            let base = pool.qt.metal_buffer()?.contents();
                            if base.is_null() {
                                lloom_tensor::bail!("direct read: the slot pool has no CPU mapping")
                            }
                            // SAFETY: `s * plane` is inside the pool (`s < n_slots`).
                            dsts[i] = crate::nn::expert_prefetch::DirectDst(unsafe {
                                base.add(s * pool.plane_bytes)
                            });
                            bytes += pool.plane_bytes as u64;
                        }
                        jobs.push(crate::nn::expert_prefetch::DirectJob {
                            expert: e as u32,
                            planes: [
                                inner.sources[0].plane_source(),
                                inner.sources[1].plane_source(),
                                inner.sources[2].plane_source(),
                            ],
                            dsts,
                            epoch: inner.map.reader_epoch(s),
                        });
                    }
                    let n_direct = jobs.len();
                    pending
                        .direct_landed
                        .extend(now.iter().map(|&(e, _)| e as u32));
                    let ticket = p.submit_direct(self.key, jobs);
                    self.counters
                        .bytes_loaded
                        .fetch_add(bytes, Ordering::Relaxed);
                    if n_direct > 0 {
                        inner.table_stale = true;
                    }
                    pending.direct = Some((ticket, n_direct as u64, t0.elapsed()));
                }
            }
            // The misses nothing staged: read now, through the reader
            // threads in parallel (one `pread` after another on this thread
            // read ~1 GB/s; four threads ~2.5, P19), landed like the staged.
            if !misses.is_empty() {
                if let Some(p) = &self.prefetch {
                    let want: Vec<u32> = misses.iter().map(|&(e, _)| e as u32).collect();
                    let planes = [
                        inner.sources[0].plane_source(),
                        inner.sources[1].plane_source(),
                        inner.sources[2].plane_source(),
                    ];
                    let t0 = std::time::Instant::now();
                    let ticket = p.submit_now(self.key, &want, &planes);
                    pending.now = Some((ticket, std::mem::take(&mut misses), t0.elapsed()));
                }
            }
        }
        #[cfg_attr(not(feature = "metal"), allow(unused_mut))]
        let mut pending = pending;
        pending.misses = misses;
        Ok(pending)
    }

    /// [`Self::resolve`]'s second half: wait for what `resolve_begin`
    /// submitted, land it, read synchronously whatever the threads did not;
    /// returns each expert's slot.
    fn resolve_finish(&self, inner: &mut Inner, pending: Pending) -> Result<Vec<usize>> {
        #[cfg(feature = "metal")]
        let Pending {
            slots,
            mut misses,
            installed,
            direct_landed,
            mut landed,
            wait_value,
            direct,
            now,
        } = pending;
        #[cfg(not(feature = "metal"))]
        let Pending {
            slots,
            misses,
            installed,
            direct_landed,
        } = pending;
        #[cfg(feature = "metal")]
        {
            if let (Some(p), Some((ticket, n_direct, submit))) = (&self.prefetch, direct) {
                let t0 = std::time::Instant::now();
                let waited = p.wait_direct(ticket);
                self.counters
                    .direct_loads
                    .fetch_add(n_direct, Ordering::Relaxed);
                self.counters
                    .direct_wait_ns
                    .fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
                self.counters
                    .load_now_ns
                    .fetch_add((submit + t0.elapsed()).as_nanos() as u64, Ordering::Relaxed);
            }
            if let (Some(p), Some((ticket, asked, submit))) = (&self.prefetch, now) {
                let t0 = std::time::Instant::now();
                let mut got: std::collections::HashMap<u32, crate::nn::expert_prefetch::Staged> = p
                    .wait_now(ticket)
                    .into_iter()
                    .map(|s| (s.expert, s))
                    .collect();
                self.counters
                    .load_now_ns
                    .fetch_add((submit + t0.elapsed()).as_nanos() as u64, Ordering::Relaxed);
                let mut rest = Vec::new();
                for &(e, s) in &asked {
                    match got.remove(&(e as u32)) {
                        Some(st) => landed.push((e, s, st)),
                        None => rest.push((e, s)),
                    }
                }
                self.counters
                    .prefetch_now
                    .fetch_add((asked.len() - rest.len()) as u64, Ordering::Relaxed);
                misses.extend(rest);
            }
            if !landed.is_empty() {
                let t0 = std::time::Instant::now();
                self.land_staged(inner, landed)?;
                self.counters
                    .land_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                inner.table_stale = true;
            }
            if wait_value != 0 {
                self.await_inflight(inner, wait_value)?;
            }
        }
        if !misses.is_empty() {
            let t0 = std::time::Instant::now();
            self.load(inner, &misses)?;
            self.counters
                .sync_load_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            // The device mirror of the table no longer matches (whichever
            // path installed: a host plan, a settle, a pin).
            inner.table_stale = true;
        }
        self.publish(inner, &installed, &direct_landed);
        Ok(slots)
    }

    /// Tell the device table about experts just installed. A table rebuilt
    /// from the map is marked stale by the landing itself. A live one is
    /// written now when no device plan is open (every later plan is encoded
    /// after the landing, so its readers are ordered after the bytes) or
    /// when the host wrote the bytes itself; an entry landed by a copy
    /// while a plan is open waits until the copy's epoch completes (a pass
    /// already encoded could otherwise map the slot before the copy runs).
    fn publish(&self, inner: &mut Inner, installed: &[(u32, u32)], direct_landed: &[u32]) {
        if inner.live.is_none() {
            return;
        }
        let epoch = inner.current_epoch();
        for &(e, slot) in installed {
            if inner.plans_open == 0 || direct_landed.contains(&e) {
                inner.table_set(e, slot);
            } else {
                inner.landing.retire(epoch, (e, slot));
            }
        }
    }

    /// Make the compute stream wait for the `io` arm's loads up to `value`
    /// (the in-flight entries a group uses), once per group.
    #[cfg(feature = "metal")]
    fn await_inflight(&self, inner: &mut Inner, value: u64) -> Result<()> {
        let queue = match &inner.loader {
            LoaderArm::Io(q) => q,
            LoaderArm::Pread => {
                lloom_tensor::bail!("an in-flight residency entry on the pread arm")
            }
        };
        let commands = match &inner.device {
            Device::Metal(d) => d.commands().clone(),
            _ => lloom_tensor::bail!("the io loader arm needs a Metal device"),
        };
        queue
            .await_value(&commands, value)
            .map_err(lloom_tensor::Error::wrap)
    }

    /// The `io` arm's hint: give each of `experts` (missing, in the order
    /// the hint ranks them) a slot NOW — the LRU victim among the slots not
    /// in flight, stopping when none is left — and issue their loads as
    /// one IO command buffer, ordered after everything encoded so far. The
    /// slots are in flight until the layer's `resolve` waits for them (or
    /// the host sees the event past their value). A hint is not a use:
    /// usage is untouched, hits and misses are counted when the layer
    /// comes.
    #[cfg(feature = "metal")]
    fn issue_hint(&self, inner: &mut Inner, experts: &[u32]) -> Result<()> {
        use lloom_tensor::kernels::metal::io_queue::IoLoad;
        let queue = match &inner.loader {
            LoaderArm::Io(q) => Arc::clone(q),
            LoaderArm::Pread => return Ok(()),
        };
        let commands = match &inner.device {
            Device::Metal(d) => d.commands().clone(),
            _ => lloom_tensor::bail!("the io loader arm needs a Metal device"),
        };
        let done = queue.signaled();
        let n = inner.map.settle_inflight(done);
        self.counters
            .io_inflight_landed
            .fetch_add(n, Ordering::Relaxed);
        // NOT a new epoch, and the open one (a group whose slot ids are out
        // and whose dispatch may not be encoded yet — the within-layer hint
        // for the later groups runs between the two) is protected: its
        // slots are never victims. Any other epoch's dispatches are encoded
        // and the load is ordered after them.
        let tick = inner.map.tick();
        let protect = inner.open_epoch.unwrap_or(Tick::NEVER);
        let mut chosen: Vec<(usize, usize)> = Vec::with_capacity(experts.len());
        for &e in experts {
            let e = e as usize;
            if e >= self.n_experts || inner.map.slot_of(e).is_some() {
                continue;
            }
            // FREE slots only, no eviction: a hint that evicts on a pool
            // smaller than the tile's working set displaces what the tile
            // needs now for what it may need next (measured: misses 23 802
            // → 30 368 and the prefill 12.6 → 21–37 s on the 937-token
            // prompt; 15 459 → 18 808 and 5.8 → 11.0 s on the 25-token one
            // over a hotlist-warmed pool, `p26-io2-*.txt`). The `pread`
            // arm's ring is the room its hints read into; the `io` arm's
            // room is its empty slots.
            let s = if inner.map.quarantined() {
                match inner.map.take_free() {
                    Some(s) => s,
                    None => break,
                }
            } else {
                let Some((s, false)) = inner.map.victim_in(protect, true, Completed::ALL) else {
                    break;
                };
                s
            };
            inner.map.install(s, e);
            // In flight from here (the value below once issued), so the
            // next pick of this loop skips it.
            inner.map.set_inflight(s, u64::MAX);
            inner.map.touch(s, tick);
            chosen.push((e, s));
        }
        if chosen.is_empty() {
            return Ok(());
        }
        let mut bytes = 0u64;
        let value = {
            let mut loads = Vec::with_capacity(chosen.len() * 3);
            for (pool, src) in inner.pools.iter().zip(inner.sources.iter()) {
                let dst = pool.qt.metal_buffer()?;
                let file = match &src.io {
                    Some(f) => f,
                    None => lloom_tensor::bail!("io arm: {} was not opened", src.path.display()),
                };
                let plane = pool.plane_bytes;
                for &(e, s) in &chosen {
                    loads.push(IoLoad {
                        dst,
                        dst_offset: s * plane,
                        file,
                        src_offset: src.plane_offset(e),
                        len: plane,
                    });
                    bytes += plane as u64;
                }
            }
            queue
                .issue(&commands, &loads)
                .map_err(lloom_tensor::Error::wrap)?
        };
        for &(_, s) in &chosen {
            inner.map.set_inflight(s, value);
        }
        inner.table_stale = true;
        self.counters
            .io_hints
            .fetch_add(chosen.len() as u64, Ordering::Relaxed);
        self.counters
            .bytes_loaded
            .fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// A hint's missing experts to whichever mechanism the arm has: the
    /// prefetcher's ring (`pread`), or slots chosen now on the IO queue
    /// (`io`, `set_io_hints`).
    #[cfg(feature = "metal")]
    fn dispatch_hint(
        &self,
        missing: Vec<u32>,
        planes: [crate::nn::expert_prefetch::PlaneSource; 3],
    ) -> Result<()> {
        if missing.is_empty() {
            return Ok(());
        }
        if let Some(p) = &self.prefetch {
            p.request(self.key, &missing, &planes);
            return Ok(());
        }
        if self.io_hints {
            let mut inner = self.inner.lock();
            self.issue_hint(&mut inner, &missing)?;
        }
        Ok(())
    }

    /// Before a device plan on the `io` arm: the hinted loads in flight
    /// (`issue_hint`) whose slots the plan may map. The entries landed by
    /// now (the event past their value) need no wait; for the rest the
    /// pass's commands wait once on the queue, for the largest value (the
    /// queue is in order), encoded only when one is pending at encode
    /// time (R1) — never a host wait (#849's D on the `io` arm).
    #[cfg(feature = "metal")]
    fn await_plan_inflight(&self, inner: &mut Inner) -> Result<()> {
        if inner.map.inflight_count() == 0 {
            return Ok(());
        }
        if let LoaderArm::Io(q) = &inner.loader {
            let done = q.signaled();
            let n = inner.map.settle_inflight(done);
            self.counters
                .io_inflight_landed
                .fetch_add(n, Ordering::Relaxed);
        }
        let pending = inner.map.inflight_max();
        if pending == 0 {
            return Ok(());
        }
        self.await_inflight(inner, pending)?;
        let n = inner.map.settle_inflight(pending);
        self.counters
            .io_inflight_planned
            .fetch_add(n, Ordering::Relaxed);
        self.counters.plan_waits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Before a device plan: land every expert the prefetcher staged for
    /// this tier and the pool has room for — a free slot under the
    /// quarantine, else the LRU victim whose last reader has run, never a
    /// slot in flight — the copies encoded here, ahead of the plan's
    /// readers; the table told at once (no plan is open at this instant).
    /// Staged experts already resident, or without a slot, go back to the
    /// ring (`predict_wasted`). An arm (`QWEN4EXP_PREDICT=hotlist`): off,
    /// nothing is staged for a device-planned token and this lands nothing.
    #[cfg(feature = "metal")]
    fn land_predicted(&self, inner: &mut Inner) -> Result<()> {
        let Some(p) = &self.prefetch else {
            return Ok(());
        };
        let staged = p.take(self.key);
        if staged.is_empty() {
            return Ok(());
        }
        // H2: a plan still open on this tier may map any resident slot;
        // nothing is evicted under it (a window's plan settles after its
        // accept) — what was staged waits in the ring for the next plan.
        if inner.plans_open > 0 {
            self.counters
                .predict_wasted
                .fetch_add(staged.len() as u64, Ordering::Relaxed);
            p.release(staged);
            return Ok(());
        }
        let tick = inner.map.tick();
        let mut landed: Vec<(usize, usize, crate::nn::expert_prefetch::Staged)> = Vec::new();
        let mut back = Vec::new();
        let mut installed: Vec<(u32, u32)> = Vec::new();
        for st in staged {
            let e = st.expert as usize;
            if e >= self.n_experts || inner.map.slot_of(e).is_some() {
                back.push(st);
                continue;
            }
            // A FREE slot (the quarantined arm's list) or an EMPTY one, never
            // an eviction: a prediction that displaces what the pool holds
            // for what it may need is the `io` hint's finding again
            // (`issue_hint`), and the first form of this — the LRU resident
            // evicted for a staged expert the usage profile ranked above it
            // — cost the warm decode +20–29 % (`memory.md` §9.9: ~35
            // landings and ~95 wasted stagings a token on a pool that
            // missed nothing). The copy is encoded in the pass's own stream,
            // ordered after every reader encoded before it.
            let free = if inner.map.quarantined() {
                inner.map.take_free()
            } else {
                match inner.map.victim_in(tick, true, Completed::ALL) {
                    Some((s, false)) => Some(s),
                    _ => None,
                }
            };
            let Some(s) = free else {
                back.push(st);
                continue;
            };
            inner.map.install(s, e);
            inner.map.touch(s, tick);
            installed.push((e as u32, s as u32));
            landed.push((e, s, st));
        }
        let n = landed.len() as u64;
        if !landed.is_empty() {
            self.land_staged(inner, landed)?;
            self.publish(inner, &installed, &[]);
        }
        self.counters.predict_landed.fetch_add(n, Ordering::Relaxed);
        self.counters
            .predict_wasted
            .fetch_add(back.len() as u64, Ordering::Relaxed);
        if !back.is_empty() {
            p.release(back);
        }
        Ok(())
    }

    /// Land staged experts `(expert, slot, staged)` in all three pools: the
    /// copy from each ring buffer into its slot, encoded here on the compute
    /// encoder (`QTensor::write_staged`), the ring entries stamped by it.
    #[cfg(feature = "metal")]
    fn land_staged(
        &self,
        inner: &mut Inner,
        landed: Vec<(usize, usize, crate::nn::expert_prefetch::Staged)>,
    ) -> Result<()> {
        let Inner {
            pools,
            staging,
            map,
            clock,
            ..
        } = inner;
        let ring = match staging.as_ref() {
            Some(r) => r,
            None => lloom_tensor::bail!("land_staged: the tier has no staging ring"),
        };
        let mut ring = ring.lock();
        let n = landed.len() as u64;
        let mut bytes = 0u64;
        let epoch = clock
            .as_ref()
            .map(|c| c.current_epoch())
            .unwrap_or(Epoch::ZERO);
        for (_, s, st) in landed {
            for (i, pool) in pools.iter_mut().enumerate() {
                let plane = pool.plane_bytes;
                let (entry, buf) = &st.planes[i];
                pool.qt
                    .write_staged(s * plane, plane, buf.clone(), &mut ring, *entry)?;
                bytes += plane as u64;
            }
            // The copy writes the slot in this command buffer.
            map.stamp(s, epoch);
        }
        drop(ring);
        self.counters
            .bytes_loaded
            .fetch_add(bytes, Ordering::Relaxed);
        if let Some(p) = &self.prefetch {
            p.used(n);
        }
        Ok(())
    }

    /// Land `(expert, slot)` pairs in all three pools.
    fn load(&self, inner: &mut Inner, misses: &[(usize, usize)]) -> Result<()> {
        let mut bytes = 0u64;
        // Whichever arm lands them, the slots are written in this command
        // buffer (the copy, or the IO load ordered into it).
        let epoch = inner.current_epoch();
        for &(_, s) in misses {
            inner.map.stamp(s, epoch);
        }
        match inner.loader.clone() {
            LoaderArm::Pread => {
                #[cfg(feature = "metal")]
                let Inner {
                    pools,
                    sources,
                    staging,
                    ..
                } = inner;
                #[cfg(not(feature = "metal"))]
                let Inner { pools, sources, .. } = inner;
                for (pool, src) in pools.iter_mut().zip(sources.iter()) {
                    let plane = pool.plane_bytes;
                    for &(e, s) in misses {
                        #[cfg(feature = "metal")]
                        if let Some(ring) = staging.as_ref() {
                            let mut ring = ring.lock();
                            pool.qt
                                .write_with_ring(s * plane, plane, &mut ring, &mut |dst| {
                                    src.read_plane(e, dst)
                                })?;
                            bytes += plane as u64;
                            continue;
                        }
                        pool.qt
                            .write_with(s * plane, plane, &mut |dst| src.read_plane(e, dst))?;
                        bytes += plane as u64;
                    }
                }
            }
            #[cfg(feature = "metal")]
            LoaderArm::Io(queue) => {
                use lloom_tensor::kernels::metal::io_queue::IoLoad;
                let commands = match &inner.device {
                    Device::Metal(d) => d.commands().clone(),
                    _ => lloom_tensor::bail!("the io loader arm needs a Metal device"),
                };
                let mut loads = Vec::with_capacity(misses.len() * 3);
                for (pool, src) in inner.pools.iter().zip(inner.sources.iter()) {
                    let dst = pool.qt.metal_buffer()?;
                    let file = match &src.io {
                        Some(f) => f,
                        None => {
                            lloom_tensor::bail!("io arm: {} was not opened", src.path.display())
                        }
                    };
                    let plane = pool.plane_bytes;
                    for &(e, s) in misses {
                        loads.push(IoLoad {
                            dst,
                            dst_offset: s * plane,
                            file,
                            src_offset: src.plane_offset(e),
                            len: plane,
                        });
                        bytes += plane as u64;
                    }
                }
                queue
                    .load_ordered(&commands, &loads)
                    .map_err(lloom_tensor::Error::wrap)?;
            }
        }
        self.counters
            .bytes_loaded
            .fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
}

impl Streamed {
    /// The routing `ids [rows, topk]` read back to the host — the one host
    /// wait of a host-planned layer, timed (`readback_ns`).
    fn read_ids_host(&self, ids: &Tensor, rows: usize, topk: usize) -> Result<Vec<u32>> {
        let t0 = std::time::Instant::now();
        // One row: read through the tier's own destination (the lock only
        // to fetch it; the wait runs without it). The same bytes `to_vec1`
        // reads.
        #[cfg(feature = "metal")]
        let slot = if rows == 1 {
            self.inner
                .lock()
                .decode_io
                .ids_slot
                .as_ref()
                .filter(|s| s.holds(ids))
                .map(Arc::clone)
        } else {
            None
        };
        // The router wrote the ids straight into the tier's slot
        // (`MoeLayer::prefill_route`): wait for that work and read them in
        // place, with no copy (#814 part 5).
        #[cfg(feature = "metal")]
        let decode = if let Some(slot) = slot {
            Some(slot.read_committed::<u32>(topk)?)
        } else if rows == 1 {
            let flat = ids.flatten_all()?.contiguous()?;
            let bytes = flat.elem_count() * std::mem::size_of::<u32>();
            let dst = self.inner.lock().decode_ids_dst(bytes)?;
            match dst {
                Some(dst) => {
                    Some(lloom_tensor::metal_backend::stage_readback_into(&flat, &dst)?.finish()?)
                }
                None => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "metal"))]
        let decode: Option<Vec<u32>> = None;
        let host: Vec<u32> = match decode {
            Some(v) => v,
            None => ids.flatten_all()?.to_vec1::<u32>()?,
        };
        if host.len() != rows * topk {
            lloom_tensor::bail!("read_ids_host: {} ids for {rows} rows × {topk}", host.len())
        }
        self.counters
            .readback_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(host)
    }

    /// One layer's window settle ([`ExpertResidency::settle_window`]): its
    /// device plan closed, the accepted rows' experts touched and counted
    /// (those a host plan resolved with loads were, at the plan). Returns
    /// the flags on the rows past `accepted`.
    fn settle_window_one(&self, rec: &WindowRecord, accepted: usize) -> Result<u64> {
        let n = rec.n();
        if rec.data.len() != 3 * n {
            lloom_tensor::bail!("settle_window: a record is [3·n], got {}", rec.data.len())
        }
        let mut inner = self.inner.lock();
        if rec.device {
            // The plan this record came from is closed: its pass is encoded,
            // so a slot evicted while it was open outlives the current epoch.
            inner.plans_open = inner.plans_open.saturating_sub(1);
            if inner.plans_open == 0 {
                let n = inner.plans_settled();
                self.counters
                    .h3_deferred_writes
                    .fetch_add(n, Ordering::Relaxed);
            }
        }
        let accepted = accepted.min(rec.rows);
        let from = rec.resolved_rows.min(accepted);
        let ids = rec.ids();
        let mut distinct = std::mem::take(&mut inner.scratch_ids);
        distinct.clear();
        distinct.extend_from_slice(&ids[from * rec.topk..accepted * rec.topk]);
        distinct.sort_unstable();
        distinct.dedup();
        let tick = inner.map.begin();
        for &e in &distinct {
            match inner.map.slot_of(e as usize) {
                Some(s) => {
                    inner.map.touch(s, tick);
                    inner.map.usage[e as usize] = inner.map.usage[e as usize].saturating_add(1);
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                }
                // Quarantined, an expert the plan mapped can have been
                // evicted after its map ran (its bytes intact until the
                // readers complete): it is loaded when next needed. In
                // place nothing is evicted while a plan is open, so an
                // accepted row's expert is resident or the caller's
                // bound was wrong.
                None if inner.map.quarantined() => {}
                None => lloom_tensor::bail!(
                    "settle_window: accepted row's expert {e} is not resident (the window was \
                     accepted past its first flagged row)"
                ),
            }
        }
        inner.scratch_ids = distinct;
        let rejected = rec.flags_from(accepted);
        if trace_settle() {
            eprintln!(
                "SETTLEWIN tick={tick} rows={} accepted={accepted} device={} first_flagged={:?} rejected_flags={rejected} resident={}",
                rec.rows,
                rec.device,
                rec.first_flagged_row(),
                inner.map.resident().len()
            );
        }
        Ok(rejected)
    }

    /// The host plan over ids on the host: the greedy row grouping to the
    /// pool, counted.
    fn plan_rows_host(&self, host: &[u32], rows: usize, topk: usize) -> Vec<RoutedGroup> {
        self.counters.plans.fetch_add(1, Ordering::Relaxed);
        // The pinned experts are resident whatever the group: a row's
        // pinned experts do not count against the unpinned capacity; nor do
        // the spare slots, which may be held for readers not yet run.
        let (capacity, pinned) = {
            let inner = self.inner.lock();
            let pinned: Vec<bool> = (0..self.n_experts)
                .map(|e| inner.map.slot_of(e).is_some_and(|s| inner.map.is_pinned(s)))
                .collect();
            // The slots in service (`active`, not the pool's size): after a
            // shrink a group planned at the old width cannot be served
            // (measured: the first yield run refused a 316-wide prefill
            // group in a pool shrunk below it, #892 A5).
            let capacity = inner
                .map
                .active()
                .saturating_sub(inner.map.pinned_count() + inner.map.spare())
                .max(1);
            (capacity, pinned)
        };
        let groups = plan_rows_pinned(host, rows, topk, capacity, &pinned);
        self.counters
            .groups
            .fetch_add(groups.len() as u64, Ordering::Relaxed);
        groups
            .into_iter()
            .map(|r| RoutedGroup {
                ids: GroupIds::Host {
                    ids: host[r.start * topk..r.end * topk].to_vec(),
                    topk,
                },
                rows: r,
            })
            .collect()
    }
}

/// `LLOOM_PREAD_DIRECT=1` (off by default): on the `pread` arm the slot
/// pools are Shared and a miss whose slot's last GPU reader has completed
/// (the pool clock) is read straight into its slot by the reader threads —
/// no staging ring entry and no landing copy for it; the rest take the ring
/// (§20.4a). Measured (#749 P27): the copies are ~13 % of the prefill's GPU
/// time and this removes them (GPU busy 4.4 → 4.0 s), the decode after a
/// long prompt −4–5 %, byte-identical — and the prefill wall +70 % (4.1 →
/// 7.1 s; 5.2 → 8.4 without the pair): the reader threads' aggregate rate
/// falls to the SSD's while the ring arm's reads hit the page cache more,
/// for a reason not isolated (the write target is not it: a `pread` into a
/// 4 GiB persistent Shared buffer runs at 40 GB/s like one into a `Vec`).
/// Opt-in until it is. Read once.
#[cfg(feature = "metal")]
pub fn pread_direct_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.pread_direct)
}

/// `QWEN4EXP_FETCH=inside` (off by default, read once): while the pool
/// misses, a device-planned decode step waits on the device at each layer
/// for the misses its router chose, read beside the pass
/// (`nn::expert_fetch`, `expert-streaming.md` §20.14c). The tier's half: its
/// slot pools are Shared and its device table is live.
pub fn fetch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.fetch == "inside")
}

/// Whether a value allocated now is read after the open pass closes: the
/// open pass is a model's own (`MetalDevice::model_pass_open`), whose settle
/// runs after it. False with no pass open, off Metal, or under a driver's.
fn record_outlives_pass(t: &Tensor) -> bool {
    #[cfg(feature = "metal")]
    if let Ok(m) = t.device().as_metal_device() {
        return m.model_pass_open();
    }
    let _ = t;
    false
}

/// An escape guard while the streamed tier allocates (Metal): the pool
/// serves it, no pass plan records or serves it. Nothing elsewhere.
#[cfg(feature = "metal")]
fn escape_guard() -> Option<lloom_tensor::metal_backend::EscapeGuard> {
    Some(lloom_tensor::metal_backend::EscapeGuard::new())
}
#[cfg(not(feature = "metal"))]
fn escape_guard() -> Option<()> {
    None
}

impl Inner {
    /// Epochs the pool clock shows complete (`u64::MAX` off Metal: nothing
    /// to wait for).
    fn completed_epochs(&self) -> Completed {
        #[cfg(feature = "metal")]
        {
            self.clock
                .as_ref()
                .map(|c| c.completed_count())
                .unwrap_or(Completed::ALL)
        }
        #[cfg(not(feature = "metal"))]
        {
            Completed::ALL
        }
    }

    /// The pool clock's current epoch (0 off Metal).
    fn current_epoch(&self) -> Epoch {
        #[cfg(feature = "metal")]
        {
            self.clock
                .as_ref()
                .map(|c| c.current_epoch())
                .unwrap_or(Epoch::ZERO)
        }
        #[cfg(not(feature = "metal"))]
        {
            Epoch::ZERO
        }
    }
}

/// Greedy row grouping: consecutive rows whose distinct experts fit `n_slots`.
fn plan_rows(ids: &[u32], rows: usize, topk: usize, n_slots: usize) -> Vec<Range<usize>> {
    plan_rows_pinned(ids, rows, topk, n_slots, &[])
}

/// [`plan_rows`] with `capacity` unpinned slots: a pinned expert (`pinned[e]`)
/// is resident in every group and takes none of them.
fn plan_rows_pinned(
    ids: &[u32],
    rows: usize,
    topk: usize,
    capacity: usize,
    pinned: &[bool],
) -> Vec<Range<usize>> {
    let counts = |e: u32| !pinned.get(e as usize).copied().unwrap_or(false);
    let mut groups = Vec::new();
    let mut start = 0usize;
    let mut distinct: Vec<u32> = Vec::with_capacity(capacity);
    for r in 0..rows {
        let row = &ids[r * topk..(r + 1) * topk];
        let mut added: Vec<u32> = Vec::new();
        for &e in row {
            if counts(e) && !distinct.contains(&e) && !added.contains(&e) {
                added.push(e);
            }
        }
        if distinct.len() + added.len() > capacity && r > start {
            groups.push(start..r);
            start = r;
            distinct.clear();
            distinct.extend(row.iter().copied().filter(|&e| counts(e)));
            distinct.sort_unstable();
            distinct.dedup();
        } else {
            distinct.extend(added);
        }
    }
    groups.push(start..rows);
    groups
}

/// Slots each streamed layer keeps free by default (`QWEN4EXP_SPARE_SLOTS`,
/// read once; 0 = the in-place arm).
fn default_spare_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().stream.spare_slots)
}

/// A token's settle begins every layer's loads before finishing any
/// (`QWEN4EXP_SETTLE_BATCH`, read once; off = one layer after another).
fn settle_batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.settle_batch)
}

/// `QWEN4EXP_TRACE_SETTLE=1` (read once): print every settle's ids, flags
/// and misses — a debugging instrument, never the shipping path.
fn trace_settle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.trace_settle)
}

/// The device-side plan: `ids [n]` (applied to) through the residency
/// `table [n_experts]` (captured; `NONE` = absent) → flat `[3·n]` u32: the
/// slot ids (a miss reads slot 0) | the ids | a miss flag per id. The CPU
/// path is the same lookup.
struct SlotMap {
    table: Tensor,
    n_experts: usize,
    /// Where the record goes, when the tier brings its host-read slot (a
    /// one-row step): the settle reads it in place (#814 part 5).
    out: Option<Tensor>,
}

impl lloom_tensor::CustomOp1 for SlotMap {
    fn name(&self) -> &'static str {
        "expert-stream-slot-map"
    }
    fn cpu_fwd(
        &self,
        s: &lloom_tensor::CpuStorage,
        l: &lloom_tensor::Layout,
    ) -> Result<(lloom_tensor::CpuStorage, lloom_tensor::Shape)> {
        let ids = match s {
            lloom_tensor::CpuStorage::U32(v) => v,
            _ => lloom_tensor::bail!("slot_map: u32 ids only"),
        };
        if !l.is_contiguous() {
            lloom_tensor::bail!("slot_map: contiguous ids only")
        }
        let n = l.shape().elem_count();
        let ids = &ids[l.start_offset()..l.start_offset() + n];
        let table: Vec<u32> = self.table.flatten_all()?.to_vec1()?;
        let mut out = vec![0u32; 3 * n];
        for (i, &e) in ids.iter().enumerate() {
            let slot = if (e as usize) < self.n_experts {
                table.get(e as usize).copied().unwrap_or(NONE)
            } else {
                NONE
            };
            let miss = slot == NONE;
            out[i] = if miss { 0 } else { slot };
            out[n + i] = e;
            out[2 * n + i] = miss as u32;
        }
        Ok((
            lloom_tensor::CpuStorage::U32(out),
            lloom_tensor::Shape::from(3 * n),
        ))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &lloom_tensor::MetalStorage,
        l: &lloom_tensor::Layout,
    ) -> Result<(lloom_tensor::MetalStorage, lloom_tensor::Shape)> {
        use lloom_tensor::backend::BackendStorage;
        if s.dtype() != lloom_tensor::DType::U32 || !l.is_contiguous() {
            lloom_tensor::bail!("slot_map: contiguous u32 ids only")
        }
        let n = l.shape().elem_count();
        let (ts, tl) = self.table.storage_and_layout();
        let ts = match &*ts {
            lloom_tensor::Storage::Metal(m) => m,
            _ => lloom_tensor::bail!("slot_map: the table must be on the metal device"),
        };
        if tl.shape().elem_count() != self.n_experts || !tl.is_contiguous() {
            lloom_tensor::bail!("slot_map: the table must be contiguous [n_experts]")
        }
        let device = s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("moe_slot_map");
        let output = match &self.out {
            Some(t) => {
                let (os, ol) = t.storage_and_layout();
                let os = match &*os {
                    lloom_tensor::Storage::Metal(m) => m,
                    _ => lloom_tensor::bail!("slot_map: the record slot must be on the metal device"),
                };
                if ol.shape().elem_count() != 3 * n
                    || !ol.is_contiguous()
                    || ol.start_offset() != 0
                    || os.dtype() != lloom_tensor::DType::U32
                {
                    lloom_tensor::bail!("slot_map: the record slot must be a fresh u32 [3·n]")
                }
                os.buffer_handle().clone()
            }
            None => device
                .new_buffer_builder()
                .with_size_for(3 * n, lloom_tensor::DType::U32)
                .with_label("moe_slot_map")
                .build()?,
        };
        lloom_tensor::kernels::call_moe_slot_map(
            device.metal_device(),
            &encoder,
            device.kernels(),
            n,
            self.n_experts,
            lloom_tensor::kernels::BufferOffset {
                buffer: s.buffer(),
                offset_in_bytes: l.start_offset() * s.dtype().size_in_bytes(),
            },
            lloom_tensor::kernels::BufferOffset {
                buffer: ts.buffer(),
                offset_in_bytes: tl.start_offset() * ts.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(lloom_tensor::Error::wrap)?;
        Ok((
            lloom_tensor::MetalStorage::new(output, device.clone(), 3 * n, s.dtype()),
            lloom_tensor::Shape::from(3 * n),
        ))
    }
}

/// A settle between its halves: the resolve in flight, and what it loads.
struct SettlePending {
    pending: Pending,
    loads: u64,
    t0: std::time::Instant,
}

impl Streamed {
    /// A settle's first half: what the pass used touched, its misses given
    /// slots and their reads submitted; `None` when the pass found every
    /// expert resident (nothing to finish, no replay).
    fn settle_begin(&self, v: &[u32]) -> Result<Option<SettlePending>> {
        if !v.len().is_multiple_of(3) {
            lloom_tensor::bail!("settle: a plan record is [3·n], got {}", v.len())
        }
        let n = v.len() / 3;
        let ids = &v[n..2 * n];
        let flags = &v[2 * n..];
        let mut inner = self.inner.lock();
        // The plan this record came from is closed: its pass is encoded,
        // so a slot evicted while it was open outlives the current epoch.
        inner.plans_open = inner.plans_open.saturating_sub(1);
        if inner.plans_open == 0 {
            let n = inner.plans_settled();
            self.counters
                .h3_deferred_writes
                .fetch_add(n, Ordering::Relaxed);
        }
        let mut distinct = std::mem::take(&mut inner.scratch_ids);
        let mut missing = std::mem::take(&mut inner.scratch_missing);
        distinct.clear();
        missing.clear();
        distinct.extend_from_slice(ids);
        distinct.sort_unstable();
        distinct.dedup();
        let tick = inner.map.begin();
        // A device-planned replay's settle: the settle before it counted the
        // token's experts already (`note_replay`); those are not counted
        // again, as a host-planned replay's resolve does not count them.
        let replaying = std::mem::take(&mut inner.replay_pending);
        let prior = if replaying {
            std::mem::take(&mut inner.counted)
        } else {
            Vec::new()
        };
        // What the pass used and found: touched first, so a load below never
        // evicts an expert this pass needs.
        for &e in &distinct {
            match inner.map.slot_of(e as usize) {
                Some(s) => {
                    inner.map.touch(s, tick);
                    if !prior.contains(&e) {
                        inner.map.usage[e as usize] =
                            inner.map.usage[e as usize].saturating_add(1);
                    }
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                }
                // Counted by `resolve` below.
                None => missing.push(e),
            }
        }
        // Every id of the record is counted by this settle — the resident
        // above, the missing by the loads below. A replay of the token must
        // not count them again (`note_replay`, `resolve_begin`).
        inner.counted.clear();
        inner.counted.extend_from_slice(&distinct);
        inner.scratch_ids = distinct;
        let flagged = flags.iter().any(|&f| f != 0);
        if trace_settle() {
            let stale: Vec<u32> = ids
                .iter()
                .zip(flags)
                .filter(|(&e, &f)| f != 0 && inner.map.slot_of(e as usize).is_some())
                .map(|(&e, _)| e)
                .collect();
            eprintln!(
                "SETTLE tick={tick} ids={ids:?} flags={flags:?} missing={missing:?} flagged_but_resident={stale:?} resident={}",
                inner.map.resident().len()
            );
        }
        // In place, nothing is evicted while a plan is open, so a missing
        // expert is exactly a flagged one. Quarantined, an expert the pass
        // mapped can be evicted after its map ran (the eviction waits in
        // `deferred`, its bytes intact): the pass was right iff nothing was
        // flagged, and an expert missing now is loaded when next needed.
        if inner.map.quarantined() && !flagged {
            inner.scratch_missing = missing;
            return Ok(None);
        }
        if missing.is_empty() {
            inner.scratch_missing = missing;
            if flagged && !inner.map.quarantined() {
                lloom_tensor::bail!(
                    "settle: the device plan flagged a miss the residency map does not show"
                )
            }
            if !flagged {
                return Ok(None);
            }
            // Quarantined: flagged, and loaded since the map ran — the
            // pass computed on a missing expert; replay, nothing to load.
            let pending = self.resolve_begin(&mut inner, &[], tick, true)?;
            return Ok(Some(SettlePending {
                pending,
                loads: 0,
                t0: std::time::Instant::now(),
            }));
        }
        if inner.completed_epochs().0 < inner.current_epoch().0 {
            self.counters
                .settle_clock_lag
                .fetch_add(1, Ordering::Relaxed);
        }
        let t0 = std::time::Instant::now();
        let pending = self.resolve_begin(&mut inner, &missing, tick, true);
        let loads = missing.len() as u64;
        inner.scratch_missing = missing;
        Ok(Some(SettlePending {
            pending: pending?,
            loads,
            t0,
        }))
    }

    /// A settle's second half: its reads waited for and landed.
    fn settle_finish(&self, p: SettlePending) -> Result<()> {
        let mut inner = self.inner.lock();
        self.resolve_finish(&mut inner, p.pending)?;
        self.counters
            .settle_ns
            .fetch_add(p.t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.counters
            .settle_loads
            .fetch_add(p.loads, Ordering::Relaxed);
        self.counters.settled_misses.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The residency table on the device, rebuilt when a settle changed it.
    fn device_table(&self, inner: &mut Inner) -> Result<Tensor> {
        if let Some(t) = &inner.live {
            return Ok(t.tensor.clone());
        }
        if inner.table_stale || inner.table.is_none() {
            let t = Tensor::from_vec(
                inner.map.slot_table().to_vec(),
                self.n_experts,
                &inner.device,
            )?;
            inner.table = Some(t);
            inner.table_stale = false;
        }
        Ok(inner.table.as_ref().expect("built above").clone())
    }
}

impl Streamed {
    /// H1: a device-planned pass's routed kernels read slots the host does
    /// not know; the plan stamped every slot with the plan's epoch, and a
    /// kernel encoded after a command-buffer cut reads in a later one. The
    /// kernel stamps its own.
    fn stamp_device_plan_readers(&self, inner: &mut Inner) {
        if inner.plans_open == 0 {
            return;
        }
        let epoch = inner.current_epoch();
        if epoch > inner.map.plan_epoch() {
            self.counters.h1_restamps.fetch_add(1, Ordering::Relaxed);
            inner.map.plan_read_all(epoch);
        }
        debug_assert!(
            inner.map.plan_epoch() >= epoch,
            "H1: a routed kernel outran its stamp"
        );
    }
}

#[cfg(feature = "metal")]
impl Streamed {
    /// The fetch worker's turn at this tier's layer of a running pass whose
    /// router chose `ids` (`nn::expert_fetch`): the experts the pool lacks
    /// given slots — a free slot, else the LRU slot not routed to now whose
    /// last reader ran before `floor` or has completed — their planes read
    /// straight into the Shared slots by the prefetcher's reader threads,
    /// and the device's live table told. Returns `(loaded, short)`: experts
    /// loaded, and experts left missing for want of a slot (the pass maps
    /// them missing and the step is redone).
    ///
    /// Why a resident slot may be written while a device plan is open: in a
    /// fetching pass this layer's slot map and routed experts are encoded
    /// after the wait the worker satisfies once this returns, so no command
    /// of the pass reads the tier before then, and every command before
    /// `floor` has completed (`FetchLane::begin_pass`). The routed experts
    /// of this layer are touched first and never chosen. No use is counted
    /// here: the pass's settle counts every routed expert once.
    pub(crate) fn fetch_beside(&self, ids: &[u32], floor: Epoch) -> (u64, u64) {
        let Some(p) = &self.prefetch else {
            return (0, ids.len() as u64);
        };
        let mut jobs: Vec<crate::nn::expert_prefetch::DirectJob> = Vec::new();
        let mut installed: Vec<(u32, u32)> = Vec::new();
        let mut short = 0u64;
        let mut bytes = 0u64;
        {
            let mut inner = self.inner.lock();
            if inner.live.is_none() {
                return (0, ids.len() as u64);
            }
            let completed = inner.completed_epochs();
            let quarantined = inner.map.quarantined();
            if quarantined {
                let n = inner.map.recycle(completed);
                self.counters
                    .recycled
                    .fetch_add(n as u64, Ordering::Relaxed);
            }
            let tick = inner.map.begin();
            let mut missing: Vec<u32> = Vec::new();
            for &e in ids {
                match inner.map.slot_of(e as usize) {
                    Some(s) => inner.map.touch(s, tick),
                    None if (e as usize) < self.n_experts && !missing.contains(&e) => {
                        missing.push(e)
                    }
                    None => {}
                }
            }
            for &e in &missing {
                let free = if quarantined {
                    inner.map.take_free().map(|s| (s, false))
                } else {
                    None
                };
                let Some((s, evicted)) =
                    free.or_else(|| inner.map.victim_beside(tick, floor, completed, quarantined))
                else {
                    short += 1;
                    continue;
                };
                if evicted {
                    let old = inner.map.expert_in(s);
                    inner.live_store(old, NONE);
                    self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                }
                inner.map.install(s, e as usize);
                inner.map.touch(s, tick);
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                let mut dsts = [crate::nn::expert_prefetch::DirectDst(std::ptr::null_mut()); 3];
                for (i, pool) in inner.pools.iter().enumerate() {
                    let base = match pool.qt.metal_buffer() {
                        Ok(b) => b.contents(),
                        Err(_) => std::ptr::null_mut(),
                    };
                    // SAFETY: `s * plane` is inside the pool (`s < n_slots`);
                    // a null base is caught below.
                    dsts[i] = crate::nn::expert_prefetch::DirectDst(if base.is_null() {
                        base
                    } else {
                        unsafe { base.add(s * pool.plane_bytes) }
                    });
                    bytes += pool.plane_bytes as u64;
                }
                if dsts.iter().any(|d| d.0.is_null()) {
                    // No CPU mapping (cannot happen with Shared pools): the
                    // expert stays out, mapped missing.
                    inner.map.uninstall(s);
                    short += 1;
                    continue;
                }
                jobs.push(crate::nn::expert_prefetch::DirectJob {
                    expert: e,
                    planes: [
                        inner.sources[0].plane_source(),
                        inner.sources[1].plane_source(),
                        inner.sources[2].plane_source(),
                    ],
                    dsts,
                    // The slot's readers have run (chosen so): no wait.
                    epoch: Epoch::ZERO,
                });
                installed.push((e, s as u32));
            }
        }
        if !jobs.is_empty() {
            let ticket = p.submit_direct(self.key, jobs);
            p.wait_direct(ticket);
            self.counters
                .direct_loads
                .fetch_add(installed.len() as u64, Ordering::Relaxed);
            self.counters
                .bytes_loaded
                .fetch_add(bytes, Ordering::Relaxed);
            let mut inner = self.inner.lock();
            for &(e, slot) in &installed {
                if inner.map.slot_of(e as usize) == Some(slot as usize) {
                    inner.live_store(e, slot);
                }
            }
        }
        (installed.len() as u64, short)
    }
}

impl ExpertResidency for Streamed {
    /// Slots in service (`profiles-and-memory-design.md` §4.2, the expert
    /// pool as an elastic tenant): `0..active` of the pool's `n_slots`.
    fn active_slots(&self) -> usize {
        self.inner.lock().map.active()
    }

    /// Shrink the slots in service to `active` (never below `floor`): the
    /// experts in the slots leaving service are evicted at their last
    /// reader's epoch, which the clock must already show complete — the
    /// caller waits the queue out first, on the refusal path alone
    /// (§21.13; the tier itself never drains the device,
    /// `qwen4exp_eviction_no_drain`) — and the three plane pools are
    /// reallocated at `active` slots with the kept slots' planes carried
    /// over: the old pools' reservations leave the ledger. The cached pass
    /// plans, which bound the old pools, are dropped. Returns the bytes
    /// released. Refused while a device plan is open, a load is in flight,
    /// or a slot leaving service still has a reader the clock has not
    /// completed.
    fn shrink_slots(&self, active: usize, floor: usize) -> Result<usize> {
        let mut inner = self.inner.lock();
        if inner.plans_open > 0 {
            lloom_tensor::bail!("shrink_slots while a device plan is open")
        }
        if inner.map.inflight_count() > 0 {
            lloom_tensor::bail!(
                "shrink_slots with {} loads in flight",
                inner.map.inflight_count()
            )
        }
        let was = inner.map.active();
        let evict: Vec<u32> = inner.map.shrink_to(active, floor);
        let active = inner.map.active();
        if active >= was {
            return Ok(0);
        }
        for e in evict {
            if let Some(s) = inner.map.slot_of(e as usize) {
                let epoch = inner.map.reader_epoch(s);
                inner.map.evict(s, epoch, false);
                inner.table_set(e, NONE);
                self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        let completed = inner.completed_epochs();
        inner.map.recycle(completed);
        let epoch = inner.current_epoch();
        if !matches!(inner.device, Device::Metal(_)) {
            // The CPU pools come back empty (no blit): every resident
            // expert goes.
            let all: Vec<u32> = inner.map.resident();
            for e in all {
                if let Some(s) = inner.map.slot_of(e as usize) {
                    inner.map.evict(s, epoch, false);
                    inner.table_set(e, NONE);
                    self.counters.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
            inner.map.recycle(Completed::ALL);
        }
        if !inner.map.settled_past_active() {
            // Undone: the slots come back into service with their experts
            // gone (they are in the quarantine and recycle as usual).
            inner.map.grow_to(was);
            lloom_tensor::bail!(
                "shrink_slots: a slot leaving service still has a reader the clock has not \
                 completed; wait the queue out first"
            )
        }
        let (before, after) = inner.reallocate_pools(active, active)?;
        inner.table_stale = true;
        inner.table = None;
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &inner.device {
            m.drop_pass_plans();
        }
        self.counters
            .slot_reallocations
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .slot_bytes_released
            .fetch_add(before.saturating_sub(after) as u64, Ordering::Relaxed);
        Ok(before.saturating_sub(after))
    }

    /// Grow the slots in service to `active`: the pools are reallocated
    /// at `active` slots with every slot in service carried over; the
    /// slots brought back are empty. Returns the bytes taken.
    fn note_replay(&self) {
        let mut inner = self.inner.lock();
        inner.replay_pending = !inner.counted.is_empty();
    }

    fn grow_slots(&self, active: usize) -> usize {
        let mut inner = self.inner.lock();
        let was = inner.map.active();
        let back = inner.map.grow_to(active);
        let active = inner.map.active();
        if active <= was || back.is_empty() && active == was {
            return 0;
        }
        let keep = if matches!(inner.device, Device::Metal(_)) {
            was
        } else {
            0
        };
        let realloc = inner.reallocate_pools(active, keep);
        let taken = match realloc {
            Ok((before, after)) => after.saturating_sub(before),
            Err(_) => 0,
        };
        inner.table_stale = true;
        inner.table = None;
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &inner.device {
            m.drop_pass_plans();
        }
        self.counters
            .slot_reallocations
            .fetch_add(1, Ordering::Relaxed);
        taken
    }

    fn plans_on_device(&self) -> bool {
        true
    }

    /// The experts this layer does not hold, most-used first (the usage
    /// profile, §20.3 R2) — for a multi-token pass nearly all of them are
    /// about to be needed — handed to the prefetcher. No pool state moves:
    /// the victims are chosen when the pass reaches the layer.
    #[cfg(feature = "metal")]
    fn prefetch_ahead(&self) -> Result<()> {
        if self.prefetch.is_none() && !self.io_hints {
            return Ok(());
        }
        let (missing, planes) = {
            let inner = self.inner.lock();
            let mut missing: Vec<u32> = (0..self.n_experts as u32)
                .filter(|&e| inner.map.slot_of(e as usize).is_none())
                .collect();
            missing.sort_by_key(|&e| std::cmp::Reverse(inner.map.usage[e as usize]));
            // No more than the pool can take: past its unpinned slots a
            // staged expert can only land by evicting one landed in the same
            // pass (measured on an empty pool: 512 asked, 327 slots, the
            // rest read for nothing — issue #749 P19).
            missing.truncate(inner.map.active().saturating_sub(inner.map.pinned_count()));
            let planes = [
                inner.sources[0].plane_source(),
                inner.sources[1].plane_source(),
                inner.sources[2].plane_source(),
            ];
            (missing, planes)
        };
        self.dispatch_hint(missing, planes)
    }

    #[cfg(feature = "metal")]
    fn prefetch_groups(&self, later: &[RoutedGroup]) -> Result<()> {
        let mut distinct: Vec<u32> = Vec::new();
        for g in later {
            if let GroupIds::Host { ids, .. } = &g.ids {
                distinct.extend_from_slice(ids);
            }
        }
        self.prefetch_ids(&distinct)
    }

    #[cfg(feature = "metal")]
    fn prefetch_named(&self, ids: &[u32]) -> Result<()> {
        if self.prefetch.is_none() && !self.io_hints {
            return Ok(());
        }
        let mut distinct: Vec<u32> = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let (missing, planes) = {
            let inner = self.inner.lock();
            let missing: Vec<u32> = distinct
                .iter()
                .copied()
                .filter(|&e| {
                    (e as usize) < self.n_experts && inner.map.slot_of(e as usize).is_none()
                })
                .collect();
            let planes = [
                inner.sources[0].plane_source(),
                inner.sources[1].plane_source(),
                inner.sources[2].plane_source(),
            ];
            (missing, planes)
        };
        self.dispatch_hint(missing, planes)
    }

    #[cfg(feature = "metal")]
    fn prefetch_hot(&self) -> Result<()> {
        // Only what the pool can land without evicting (`land_predicted`):
        // the usage profile's top missing experts, as many as there are
        // free slots — none on a full pool, so the warm path stages nothing.
        let (missing, planes) = {
            let inner = self.inner.lock();
            let free = inner.map.free_slots();
            if free == 0 {
                return Ok(());
            }
            let mut rest: Vec<u32> = (0..self.n_experts as u32)
                .filter(|&e| inner.map.slot_of(e as usize).is_none())
                .collect();
            rest.sort_by_key(|&e| std::cmp::Reverse(inner.map.usage[e as usize]));
            rest.truncate(free);
            let planes = [
                inner.sources[0].plane_source(),
                inner.sources[1].plane_source(),
                inner.sources[2].plane_source(),
            ];
            (rest, planes)
        };
        self.dispatch_hint(missing, planes)
    }

    #[cfg(feature = "metal")]
    fn prefetch_ids(&self, ids: &[u32]) -> Result<()> {
        if self.prefetch.is_none() && !self.io_hints {
            return Ok(());
        }
        let mut distinct: Vec<u32> = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let (missing, planes) = {
            let inner = self.inner.lock();
            let mut missing: Vec<u32> = distinct
                .iter()
                .copied()
                .filter(|&e| {
                    (e as usize) < self.n_experts && inner.map.slot_of(e as usize).is_none()
                })
                .collect();
            // The named experts first; the usage profile fills what the
            // pool could still take (a 512-token tile needs most of what
            // the router did not name either, P20).
            let bound = inner.map.active().saturating_sub(inner.map.pinned_count());
            if missing.len() < bound {
                let mut rest: Vec<u32> = (0..self.n_experts as u32)
                    .filter(|&e| {
                        inner.map.slot_of(e as usize).is_none()
                            && distinct.binary_search(&e).is_err()
                    })
                    .collect();
                rest.sort_by_key(|&e| std::cmp::Reverse(inner.map.usage[e as usize]));
                missing.extend(rest);
            }
            missing.truncate(bound);
            let planes = [
                inner.sources[0].plane_source(),
                inner.sources[1].plane_source(),
                inner.sources[2].plane_source(),
            ];
            (missing, planes)
        };
        self.dispatch_hint(missing, planes)
    }

    #[cfg(feature = "metal")]
    fn prefetch_stats(&self) -> PrefetchStats {
        self.prefetch
            .as_ref()
            .map(|p| p.stats())
            .unwrap_or_default()
    }

    fn fetch_ready(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            if self.prefetch.is_none() {
                return false;
            }
            let inner = self.inner.lock();
            inner.shared_pools
                && inner.live.is_some()
                && matches!(inner.loader, LoaderArm::Pread)
                && matches!(inner.device, Device::Metal(_))
        }
        #[cfg(not(feature = "metal"))]
        false
    }

    fn fetch_prepare(&self, topk: usize) -> Result<bool> {
        #[cfg(feature = "metal")]
        {
            if !self.fetch_ready() {
                return Ok(false);
            }
            let slot = self.inner.lock().decode_ids_slot(topk)?;
            match slot {
                Some(s) => {
                    s.fill_u32(NONE, topk)?;
                    Ok(true)
                }
                None => Ok(false),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = topk;
            Ok(false)
        }
    }

    #[cfg(feature = "metal")]
    fn fetch_register(
        &self,
        lane: &crate::nn::expert_fetch::FetchLane,
        ids: &Tensor,
    ) -> Result<bool> {
        if !self.fetch_ready() {
            return Ok(false);
        }
        let slot = self
            .inner
            .lock()
            .decode_io
            .ids_slot
            .as_ref()
            .filter(|s| s.holds(ids))
            .map(Arc::clone);
        let Some(slot) = slot else {
            return Ok(false);
        };
        lane.register(self, slot, ids.elem_count(), self.n_experts)?;
        Ok(true)
    }

    fn group_misses(&self, group: &RoutedGroup) -> usize {
        let GroupIds::Host { ids, .. } = &group.ids else {
            return 0;
        };
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let inner = self.inner.lock();
        distinct
            .iter()
            .filter(|&&e| (e as usize) < self.n_experts && inner.map.slot_of(e as usize).is_none())
            .count()
    }

    fn misses_so_far(&self) -> u64 {
        self.counters.misses.load(Ordering::Relaxed)
    }

    fn evictions_so_far(&self) -> u64 {
        self.counters.evictions.load(Ordering::Relaxed)
    }

    fn read_record_in_place(&self, record: &Tensor) -> Result<Option<Vec<u32>>> {
        #[cfg(feature = "metal")]
        {
            let slot = self
                .inner
                .lock()
                .decode_io
                .record_slot
                .as_ref()
                .filter(|s| s.holds(record))
                .map(Arc::clone);
            slot.map(|s| s.read_unchecked::<u32>(record.elem_count()))
                .transpose()
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = record;
            Ok(None)
        }
    }

    fn decode_ids_slot(&self, topk: usize) -> Result<Option<Tensor>> {
        #[cfg(feature = "metal")]
        {
            let slot = self.inner.lock().decode_ids_slot(topk)?;
            slot.map(|s| s.tensor((1, topk))).transpose()
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = topk;
            Ok(None)
        }
    }

    fn plan_device(&self, ids: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        let (rows, topk) = ids.dims2()?;
        let n = rows * topk;
        let ids = ids.flatten_all()?.contiguous()?;
        // The record is read by the settle after the pass body: inside a
        // driver's pass, where a plan serves it, and after a model's own
        // pass has closed, where it must be declared (`DESIGN.md` §9.2aa).
        let _escape = if record_outlives_pass(&ids) {
            escape_guard()
        } else {
            None
        };
        let table = {
            // The table outlives the pass: the tier's, rebuilt when stale.
            let _escape = escape_guard();
            let mut inner = self.inner.lock();
            // What the prefetcher staged for this tier (the hotlist's
            // prediction, `prefetch_hot`) lands now — the copies encoded
            // ahead of this plan's readers on the one queue, the table
            // told — so the pass finds them resident (#849's D).
            #[cfg(feature = "metal")]
            self.land_predicted(&mut inner)?;
            #[cfg(feature = "metal")]
            self.await_plan_inflight(&mut inner)?;
            // The pass may read any resident slot (a direct read into one
            // waits for it).
            let epoch = inner.current_epoch();
            inner.map.plan_read_all(epoch);
            inner.store_landed();
            inner.plans_open += 1;
            self.device_table(&mut inner)?
        };
        self.counters.device_plans.fetch_add(1, Ordering::Relaxed);
        // One row: the record goes into the tier's host-read slot, which the
        // settle reads in place (#814 part 5).
        #[cfg(feature = "metal")]
        let out = if rows == 1 {
            let slot = self.inner.lock().decode_record_slot(3 * n)?;
            slot.map(|s| s.tensor(3 * n)).transpose()?
        } else {
            None
        };
        #[cfg(not(feature = "metal"))]
        let out = None;
        let plan = ids.apply_op1_no_bwd(&SlotMap {
            table,
            n_experts: self.n_experts,
            out,
        })?; // [3·n]
        let slots = plan.narrow(0, 0, n)?.reshape((rows, topk))?;
        Ok(Some((slots, plan)))
    }

    fn settle(&self, record: &Tensor) -> Result<bool> {
        let _escape = escape_guard();
        let v: Vec<u32> = record.flatten_all()?.to_vec1()?;
        self.settle_host(&v)
    }

    fn settle_host(&self, v: &[u32]) -> Result<bool> {
        let _escape = escape_guard();
        Ok(match self.settle_begin(v)? {
            Some(p) => {
                self.settle_finish(p)?;
                true
            }
            None => false,
        })
    }

    fn settle_batch(parts: &[(&Self, &[u32])]) -> Result<Vec<(bool, u64)>> {
        let _escape = escape_guard();
        if !settle_batch_on() {
            let mut out = Vec::with_capacity(parts.len());
            for (r, v) in parts {
                out.push(match r.settle_begin(v)? {
                    Some(p) => {
                        let n = p.loads;
                        r.settle_finish(p)?;
                        (true, n)
                    }
                    None => (false, 0),
                });
            }
            return Ok(out);
        }
        let mut begun = Vec::with_capacity(parts.len());
        for (r, v) in parts {
            begun.push(r.settle_begin(v)?);
        }
        let mut out = Vec::with_capacity(parts.len());
        for ((r, _), p) in parts.iter().zip(begun) {
            out.push(match p {
                Some(p) => {
                    let n = p.loads;
                    r.settle_finish(p)?;
                    (true, n)
                }
                None => (false, 0),
            });
        }
        Ok(out)
    }

    fn plan(&self, ids: &Tensor) -> Result<Vec<RoutedGroup>> {
        let (rows, topk) = ids.dims2()?;
        let host = self.read_ids_host(ids, rows, topk)?;
        Ok(self.plan_rows_host(&host, rows, topk))
    }

    fn plan_host(&self, ids: &[u32], rows: usize, topk: usize) -> Result<Option<Vec<RoutedGroup>>> {
        if ids.len() != rows * topk {
            lloom_tensor::bail!("plan_host: {} ids for {rows} rows × {topk}", ids.len())
        }
        Ok(Some(self.plan_rows_host(ids, rows, topk)))
    }

    fn plan_window(
        &self,
        ids: &Tensor,
        real_rows: usize,
        loaded_rows: usize,
    ) -> Result<Option<(Tensor, WindowRecord)>> {
        // What the tier allocates depends on the pool's state, not the
        // pass's shape: declared outside any pass plan (§9.2aa).
        let _escape = escape_guard();
        let (rows, topk) = ids.dims2()?;
        let n = rows * topk;
        let host = self.read_ids_host(ids, rows, topk)?;
        self.counters.plans.fetch_add(1, Ordering::Relaxed);
        self.counters.groups.fetch_add(1, Ordering::Relaxed);
        let loaded_rows = loaded_rows.min(rows);
        // The real rows (the first `real_rows`, row 0 of a verify window):
        // resident within this epoch, their misses loaded, their uses
        // counted. The draft rows loaded beside them (`loaded_rows` past
        // `real_rows`) are loaded and touched but counted only once the
        // accept keeps them (`settle_window`): a wrong draft's experts never
        // enter the usage profile the hotlist is written from.
        let real_rows = real_rows.min(loaded_rows);
        let mut distinct: Vec<u32> = host[..real_rows * topk].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut drafts: Vec<u32> = host[real_rows * topk..loaded_rows * topk]
            .iter()
            .copied()
            .filter(|e| distinct.binary_search(e).is_err())
            .collect();
        drafts.sort_unstable();
        drafts.dedup();
        let mut inner = self.inner.lock();
        // The loaded rows must be resident at once: wider than the pool's
        // unpinned capacity, they are the grouped host plan's (`plan` /
        // `ensure`, a group at a time) — no record, nothing speculative.
        let capacity = self
            .n_slots
            .saturating_sub(inner.map.pinned_count() + inner.map.spare());
        if distinct.len() + drafts.len() > capacity {
            return Ok(None);
        }
        let tick = inner.map.begin();
        let misses_before = self.counters.misses.load(Ordering::Relaxed);
        let slots = self.resolve(&mut inner, &distinct, tick)?;
        let real_loads = self.counters.misses.load(Ordering::Relaxed) - misses_before;
        let draft_slots = if drafts.is_empty() {
            Vec::new()
        } else {
            let usage_before = inner.map.usage.clone();
            let s = self.resolve(&mut inner, &drafts, tick)?;
            inner.map.usage = usage_before;
            s
        };
        let draft_loads = self.counters.misses.load(Ordering::Relaxed) - misses_before - real_loads;
        let mut data = vec![0u32; 3 * n];
        let mut open_slots: Vec<usize> = slots.clone();
        open_slots.extend_from_slice(&draft_slots);
        for (i, &e) in host.iter().enumerate() {
            let (slot, miss) = if i / topk < real_rows {
                let j = distinct
                    .binary_search(&e)
                    .expect("distinct covers the real rows");
                (slots[j] as u32, false)
            } else if i / topk < loaded_rows && distinct.binary_search(&e).is_ok() {
                let j = distinct.binary_search(&e).expect("checked");
                (slots[j] as u32, false)
            } else if i / topk < loaded_rows {
                let j = drafts
                    .binary_search(&e)
                    .expect("drafts covers the loaded rows");
                (draft_slots[j] as u32, false)
            } else if (e as usize) < self.n_experts {
                // A draft row reads what is resident; a miss reads slot 0
                // and flags the row — nothing is loaded for it.
                match inner.map.slot_of(e as usize) {
                    Some(s) => {
                        open_slots.push(s);
                        (s as u32, false)
                    }
                    None => (0, true),
                }
            } else {
                lloom_tensor::bail!("expert {e} out of {}", self.n_experts)
            };
            data[i] = slot;
            data[n + i] = e;
            data[2 * n + i] = miss as u32;
        }
        // Every slot the dispatch reads is a reader of this epoch (`down`
        // stamps them: a direct read into one waits for it).
        open_slots.sort_unstable();
        open_slots.dedup();
        inner.open_epoch = Some(tick);
        inner.open_slots = open_slots;
        let slot_ids = Tensor::from_vec(data[..n].to_vec(), (rows, topk), &inner.device)?;
        Ok(Some((
            slot_ids,
            WindowRecord {
                data,
                rows,
                topk,
                device: false,
                resolved_rows: real_rows,
                real_loads,
                draft_loads,
            },
        )))
    }

    fn settle_window(parts: &[(&Self, &WindowRecord)], accepted: usize) -> Result<u64> {
        let _escape = escape_guard();
        let mut rejected_flags = 0u64;
        for (r, rec) in parts {
            rejected_flags += r.settle_window_one(rec, accepted)?;
        }
        Ok(rejected_flags)
    }

    fn ensure(&self, group: &RoutedGroup) -> Result<Tensor> {
        // What the tier allocates depends on the pool's state, not the
        // pass's shape: declared outside any pass plan (§9.2aa).
        let _escape = escape_guard();
        let (ids, topk) = match &group.ids {
            GroupIds::Host { ids, topk } => (ids, *topk),
            GroupIds::Device(_) => lloom_tensor::bail!("Streamed::ensure: a device-planned group"),
        };
        let mut distinct: Vec<u32> = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let mut inner = self.inner.lock();
        let tick = inner.map.begin();
        let slots = self.resolve(&mut inner, &distinct, tick)?;
        inner.open_epoch = Some(tick);
        inner.open_slots = slots.clone();
        let slot_ids: Vec<u32> = ids
            .iter()
            .map(|e| {
                let i = distinct.binary_search(e).expect("distinct covers ids");
                slots[i] as u32
            })
            .collect();
        #[cfg(feature = "metal")]
        if group.rows.len() == 1 {
            if let Some(t) = inner.write_decode_slots(&slot_ids)? {
                return Ok(t);
            }
        }
        Tensor::from_vec(slot_ids, (group.rows.len(), topk), &inner.device)
    }

    fn gate_up(
        &self,
        x: &Tensor,
        slot_ids: &Tensor,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)> {
        let mut inner = self.inner.lock();
        // One launch for the pair on the matmul-vec arm on Metal; two elsewhere.
        let out = inner.pools[0].qt.indexed_moe_forward_pair_as(
            &inner.pools[1].qt,
            x,
            slot_ids,
            arm,
            act,
        )?;
        // H1 (§4.7): the slots this plan reads are stamped with the open
        // epoch at the dispatch, so a load never writes one under it.
        self.stamp_device_plan_readers(&mut inner);
        Ok(out)
    }

    fn down(&self, h: &Tensor, slot_ids: &Tensor, arm: MoeArm) -> Result<Tensor> {
        let mut inner = self.inner.lock();
        let y = inner.pools[2]
            .qt
            .indexed_moe_forward_with(h, slot_ids, arm)?;
        self.stamp_device_plan_readers(&mut inner);
        // The group's last dispatch is encoded: its slots may be given away
        // — to a direct read once this command buffer has run.
        inner.open_epoch = None;
        let epoch = inner.current_epoch();
        let slots = std::mem::take(&mut inner.open_slots);
        for s in slots {
            inner.map.stamp(s, epoch);
        }
        // The group's slot ids, if they were the tier's own buffer: its last
        // reader is encoded into this epoch's command buffer.
        #[cfg(feature = "metal")]
        if inner.decode_io.slots_read_at == Some(BUSY) {
            inner.decode_io.slots_read_at = Some(epoch);
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::gguf_file;

    /// A stacked `[e, out, inp]` expert weight quantized to `dtype`.
    fn stacked(e: usize, out: usize, inp: usize, seed: f32, dtype: GgmlDType) -> QTensor {
        let n = e * out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
            .collect();
        let t = Tensor::from_vec(v, (e, out, inp), &Device::Cpu).unwrap();
        QTensor::quantize(&t, dtype).unwrap()
    }

    /// Write `tensors` to a fresh GGUF at a unique temp path (tests run in
    /// parallel in one process, so the pid alone would collide).
    fn write_gguf(tensors: &[(&str, &QTensor)]) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lloom-expert-stream-{}-{}.gguf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let mut f = File::create(&path).unwrap();
        gguf_file::write(&mut f, &[], tensors).unwrap();
        path
    }

    fn open(path: &Path, name: &str) -> ExpertSource {
        let mut f = File::open(path).unwrap();
        let content = Content::read(&mut f).unwrap();
        ExpertSource::from_gguf(&content, path, name).unwrap()
    }

    fn ids(rows: usize, topk: usize, n_experts: usize, seed: usize) -> Tensor {
        let v: Vec<u32> = (0..rows * topk)
            .map(|i| ((i * 7 + seed * 13 + i / topk) % n_experts) as u32)
            .collect();
        Tensor::from_vec(v, (rows, topk), &Device::Cpu).unwrap()
    }

    fn x(rows: usize, k: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * k)
            .map(|i| ((i as f32) * 0.07 + seed).sin() * 0.4)
            .collect();
        Tensor::from_vec(v, (rows, 1, k), &Device::Cpu).unwrap()
    }

    fn run<R: ExpertResidency>(r: &R, x3: &Tensor, ids: &Tensor) -> Vec<f32> {
        let groups = r.plan(ids).unwrap();
        let arm = MoeArm::for_rows(x3.dim(0).unwrap());
        let mut ys = Vec::new();
        for g in &groups {
            let slot_ids = r.ensure(g).unwrap();
            let xg = x3.narrow(0, g.rows.start, g.rows.len()).unwrap();
            let (gt, u) = r.gate_up(&xg, &slot_ids, arm, xg.dtype()).unwrap();
            let h = (crate::nn::ops::silu(&gt).unwrap() * u).unwrap();
            ys.push(r.down(&h, &slot_ids, arm).unwrap());
        }
        let y = Tensor::cat(&ys, 0).unwrap();
        y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Streamed == Resident bit for bit over decode rows and a chunk whose
    /// expert union exceeds the pool (several groups, evictions), for a float
    /// stack and a real block type; the counters prove the streaming ran.
    /// #899's finding: a device-planned step that misses counted its
    /// replayed token's experts twice — once at the settle, once at the
    /// replay's loads. A settle followed by `note_replay` and the host plan
    /// over the same ids counts each expert once; the same host plan
    /// without `note_replay` (a new token) counts again.
    #[test]
    fn a_replayed_token_counts_its_experts_once() {
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::Q8_0);
        let up = stacked(e, ff, d, 2.0, GgmlDType::Q8_0);
        let down = stacked(e, d, ff, 3.0, GgmlDType::Q8_0);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let streamed = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            5,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        // Experts 1 and 2 resident; the token wants 1, 2, 7: the device plan
        // would have found 7 missing — its record says so.
        let pre = Tensor::new(&[[1u32, 2]], &Device::Cpu).unwrap();
        for g in streamed.plan(&pre).unwrap() {
            streamed.ensure(&g).unwrap();
        }
        let before = streamed.usage();
        // The record: [slots.. | ids.. | flags..] for n = topk ids.
        let ids_tok = [1u32, 2, 7];
        let slots: Vec<u32> = ids_tok
            .iter()
            .map(|&i| {
                streamed
                    .inner
                    .lock()
                    .map
                    .slot_of(i as usize)
                    .map(|s| s as u32)
                    .unwrap_or(0)
            })
            .collect();
        let flags: Vec<u32> = ids_tok
            .iter()
            .map(|&i| (streamed.inner.lock().map.slot_of(i as usize).is_none()) as u32)
            .collect();
        let record: Vec<u32> = slots
            .iter()
            .chain(ids_tok.iter())
            .chain(flags.iter())
            .copied()
            .collect();
        assert!(streamed.settle_host(&record).unwrap(), "a miss settles");
        let settled = streamed.usage();
        for &i in &ids_tok {
            assert_eq!(
                settled[i as usize],
                before[i as usize] + 1,
                "expert {i} counted once at the settle"
            );
        }
        // The replay: the host plan over the same token.
        streamed.note_replay();
        let tok = Tensor::new(&[[1u32, 2, 7]], &Device::Cpu).unwrap();
        for g in streamed.plan(&tok).unwrap() {
            streamed.ensure(&g).unwrap();
        }
        let replayed = streamed.usage();
        for &i in &ids_tok {
            assert_eq!(
                replayed[i as usize], settled[i as usize],
                "expert {i} not counted again by the replay"
            );
        }
        // A new token with the same experts: counted.
        for g in streamed.plan(&tok).unwrap() {
            streamed.ensure(&g).unwrap();
        }
        let next = streamed.usage();
        for &i in &ids_tok {
            assert_eq!(
                next[i as usize],
                settled[i as usize] + 1,
                "expert {i} counted by the next token"
            );
        }
        let _ = topk;
    }

    /// A device-planned replay (`QWEN4EXP_REPLAY=device`, the fetch arm's
    /// redo) settles the same token's record again: its experts are counted
    /// once — at the first settle — and a later token's are counted again.
    #[test]
    fn a_device_planned_replay_counts_its_experts_once() {
        let (e, ff, d) = (12usize, 32usize, 64usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::Q8_0);
        let up = stacked(e, ff, d, 2.0, GgmlDType::Q8_0);
        let down = stacked(e, d, ff, 3.0, GgmlDType::Q8_0);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let streamed = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            5,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        let pre = Tensor::new(&[[1u32, 2]], &Device::Cpu).unwrap();
        for g in streamed.plan(&pre).unwrap() {
            streamed.ensure(&g).unwrap();
        }
        let ids_tok = [1u32, 2, 7];
        let record = |streamed: &Streamed| -> Vec<u32> {
            let inner = streamed.inner.lock();
            let slots = ids_tok
                .iter()
                .map(|&i| inner.map.slot_of(i as usize).map_or(0, |s| s as u32));
            let flags = ids_tok
                .iter()
                .map(|&i| inner.map.slot_of(i as usize).is_none() as u32);
            slots.chain(ids_tok.iter().copied()).chain(flags).collect()
        };
        let before = streamed.usage();
        // The step missed 7: its settle loads it and counts the three.
        assert!(streamed.settle_host(&record(&streamed)).unwrap());
        // The replay on the device plan: every expert resident now.
        streamed.note_replay();
        assert!(!streamed.settle_host(&record(&streamed)).unwrap());
        let replayed = streamed.usage();
        for &i in &ids_tok {
            assert_eq!(
                replayed[i as usize],
                before[i as usize] + 1,
                "expert {i} counted once over the settle and its replay"
            );
        }
        // The next token's settle, no replay: counted.
        assert!(!streamed.settle_host(&record(&streamed)).unwrap());
        let next = streamed.usage();
        for &i in &ids_tok {
            assert_eq!(
                next[i as usize],
                before[i as usize] + 2,
                "expert {i} counted by the next token"
            );
        }
    }

    fn streamed_matches_resident(dtype: GgmlDType) {
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let resident = Resident::new(Arc::new(gate), Arc::new(up), Arc::new(down));
        let n_slots = 5; // < e, and < a chunk's union
        let streamed = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            n_slots,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();

        // decode rows, one at a time (temporal reuse → hits)
        for t in 0..8 {
            let xs = x(1, d, t as f32);
            let id = ids(1, topk, e, t % 3);
            assert_eq!(
                run(&resident, &xs, &id),
                run(&streamed, &xs, &id),
                "token {t}"
            );
        }
        // a chunk of 9 rows: union 12 > 5 slots → groups + evictions
        let xs = x(9, d, 0.5);
        let id = ids(9, topk, e, 4);
        let groups = streamed.plan(&id).unwrap();
        assert!(
            groups.len() >= 2,
            "expected several groups, got {}",
            groups.len()
        );
        assert_eq!(run(&resident, &xs, &id), run(&streamed, &xs, &id), "chunk");

        let s = streamed.stats();
        // 8 decode plans + the chunk's plan above + the chunk plan inside `run`.
        assert!(s.plans == 10, "plans {}", s.plans);
        assert!(s.hits > 0, "no hits: {s:?}");
        assert!(
            s.misses > 0 && s.evictions > 0,
            "no eviction exercised: {s:?}"
        );
        let plane: usize = [ff * d, ff * d, d * ff]
            .iter()
            .map(|n| n / dtype.block_size() * dtype.type_size())
            .sum();
        assert_eq!(
            s.bytes_loaded,
            s.misses * plane as u64,
            "bytes loaded must be misses × the three planes"
        );
        std::fs::remove_file(path).ok();
    }

    /// The device plan against the host plan: the same slot ids once the
    /// misses are settled, the misses flagged before, and the routed output
    /// on the settled slots equal to `Resident`'s bit for bit. On a pool
    /// smaller than the working set, so settles load and evict.
    #[test]
    fn device_plan_settles_to_the_host_plan() {
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let resident = Resident::new(Arc::new(gate), Arc::new(up), Arc::new(down));
        let streamed = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            5,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        assert!(streamed.plans_on_device());
        let mut settled_with_misses = 0;
        for t in 0..8 {
            let xs = x(1, d, t as f32);
            let id = ids(1, topk, e, t % 3);
            let (slots, record) = streamed.plan_device(&id).unwrap().expect("a device plan");
            let rec: Vec<u32> = record.to_vec1().unwrap();
            let n = topk;
            assert_eq!(
                &rec[n..2 * n],
                &id.flatten_all().unwrap().to_vec1::<u32>().unwrap()[..]
            );
            let flagged = rec[2 * n..].iter().any(|&f| f != 0);
            let missed = streamed.settle(&record).unwrap();
            assert_eq!(
                flagged, missed,
                "token {t}: the flags and the settle disagree"
            );
            if missed {
                settled_with_misses += 1;
                // Replay: every expert is resident now, the plan is clean.
                let (slots2, record2) = streamed.plan_device(&id).unwrap().unwrap();
                assert!(
                    !streamed.settle(&record2).unwrap(),
                    "token {t}: a second miss"
                );
                let s2: Vec<u32> = slots2.flatten_all().unwrap().to_vec1().unwrap();
                let host = streamed.ensure(&streamed.plan(&id).unwrap()[0]).unwrap();
                let h: Vec<u32> = host.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(
                    s2, h,
                    "token {t}: the device plan's slots vs the host plan's"
                );
                let out = {
                    let x3 = xs.reshape((1, 1, d)).unwrap();
                    let (g, u) = streamed
                        .gate_up(&x3, &slots2, MoeArm::MatVec, x3.dtype())
                        .unwrap();
                    let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
                    streamed.down(&hh, &slots2, MoeArm::MatVec).unwrap()
                };
                let want = {
                    let x3 = xs.reshape((1, 1, d)).unwrap();
                    let (g, u) = resident
                        .gate_up(&x3, &id, MoeArm::MatVec, x3.dtype())
                        .unwrap();
                    let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
                    resident.down(&hh, &id, MoeArm::MatVec).unwrap()
                };
                assert_eq!(
                    out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    "token {t}: routed output"
                );
            } else {
                let s1: Vec<u32> = slots.flatten_all().unwrap().to_vec1().unwrap();
                let host = streamed.ensure(&streamed.plan(&id).unwrap()[0]).unwrap();
                let h: Vec<u32> = host.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(s1, h, "token {t}: a clean device plan vs the host plan");
            }
        }
        assert!(
            settled_with_misses > 0,
            "the pool never missed: the test is vacuous"
        );
        let st = streamed.stats();
        assert_eq!(st.settled_misses, settled_with_misses);
        assert!(st.device_plans >= 8);
        std::fs::remove_file(path).ok();
    }

    /// `warm` loads the profile's head into the free slots — no eviction,
    /// no pin, nothing counted as a use — and stops at the pool's edge; a
    /// warmed pool then hits. `usage` counts every use through both plans.
    #[test]
    fn warm_fills_the_free_slots_and_counts_nothing() {
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let streamed = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            5,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        // Eight asked, five slots: the first five land, no eviction.
        let loaded = streamed.warm(&[7, 3, 3, 9, 1, 0, 5, 6, 8]).unwrap();
        assert_eq!(loaded, 5);
        let st = streamed.stats();
        assert_eq!((st.misses, st.evictions), (5, 0), "{st:?}");
        assert!(
            streamed.usage().iter().all(|&u| u == 0),
            "a warm counted as a use"
        );
        let mut resident = streamed.resident();
        resident.sort_unstable();
        assert_eq!(resident, vec![0, 1, 3, 7, 9]);
        // A second warm of the same finds no free slot: nothing loaded.
        assert_eq!(streamed.warm(&[2, 4]).unwrap(), 0);
        // A token over warmed experts: hits only, and the usage counts it.
        let id = Tensor::from_vec(vec![7u32, 3, 9], (1, topk), &Device::Cpu).unwrap();
        let (_, record) = streamed.plan_device(&id).unwrap().unwrap();
        assert!(!streamed.settle(&record).unwrap());
        let st = streamed.stats();
        assert_eq!(st.misses, 5, "a warmed expert missed: {st:?}");
        let u = streamed.usage();
        assert_eq!((u[7], u[3], u[9], u[1]), (1, 1, 1, 0));
        let groups = streamed.plan(&id).unwrap();
        streamed.ensure(&groups[0]).unwrap();
        assert_eq!(
            streamed.usage()[7],
            2,
            "the host plan did not count the use"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn streamed_matches_resident_f32() {
        streamed_matches_resident(GgmlDType::F32);
    }

    #[test]
    fn streamed_matches_resident_q8_0() {
        streamed_matches_resident(GgmlDType::Q8_0);
    }

    /// Pinned experts survive an eviction pass; the pool refuses a pin set
    /// that leaves no free slot.
    #[test]
    fn pinned_experts_are_never_evicted() {
        let (e, ff, d, topk) = (10usize, 32usize, 32usize, 2usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::F32);
        let up = stacked(e, ff, d, 2.0, GgmlDType::F32);
        let down = stacked(e, d, ff, 3.0, GgmlDType::F32);
        let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
        let streamed = Streamed::new(
            open(&path, "g"),
            open(&path, "u"),
            open(&path, "d"),
            4,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        streamed.pin(&[7, 8]).unwrap();
        assert!(
            streamed.pin(&[0, 1, 2, 3]).is_err(),
            "a pin set must leave a free slot"
        );
        for t in 0..6 {
            let id = ids(1, topk, 7, t); // experts 0..7 only, churning the 2 free slots
            let xs = x(1, d, 0.0);
            let _ = run(&streamed, &xs, &id);
        }
        let inner = streamed.inner.lock();
        assert!(inner.map.slot_of(7).is_some() && inner.map.slot_of(8).is_some());
        drop(inner);
        assert!(streamed.stats().evictions > 0);
        std::fs::remove_file(path).ok();
    }

    /// The elastic surface (§4.2): a shrink takes the top slots out of
    /// service — their experts evicted, later loads and victims confined to
    /// the slots in service — and a grow brings them back; the floor holds.
    #[test]
    fn a_shrink_parks_the_top_slots_and_a_grow_brings_them_back() {
        let (e, ff, d, topk) = (10usize, 32usize, 32usize, 2usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::F32);
        let up = stacked(e, ff, d, 2.0, GgmlDType::F32);
        let down = stacked(e, d, ff, 3.0, GgmlDType::F32);
        let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
        for spare in [0usize, 1] {
            let mut streamed = Streamed::new(
                open(&path, "g"),
                open(&path, "u"),
                open(&path, "d"),
                4,
                LoaderArm::Pread,
                &Device::Cpu,
            )
            .unwrap();
            streamed.set_spare_slots(spare).unwrap();
            let warm: &[u32] = if spare == 0 {
                &[0, 1, 2, 3]
            } else {
                &[0, 1, 2]
            };
            assert_eq!(streamed.warm(warm).unwrap(), warm.len());
            assert_eq!(streamed.active_slots(), 4);
            // Shrink to two slots (a floor of one): the experts in slots 2..
            // are evicted and the pools are reallocated at two slots — on
            // the CPU they come back empty (no blit), so every resident
            // expert goes; two slots' bytes are released.
            let per_slot = streamed.bytes_per_slot();
            let released = streamed.shrink_slots(2, 1).unwrap();
            assert_eq!(released, 2 * per_slot, "spare {spare}");
            assert_eq!(streamed.active_slots(), 2);
            let resident = streamed.inner.lock().map.resident();
            assert!(resident.is_empty(), "spare {spare}: {resident:?}");
            let evictions = streamed.stats().evictions;
            assert!(evictions >= 2, "spare {spare}: {evictions}");
            // Loads now live in the two slots in service.
            for t in 0..4 {
                let id = ids(1, topk, 10, 7 + t);
                let xs = x(1, d, 0.0);
                let _ = run(&streamed, &xs, &id);
            }
            {
                let inner = streamed.inner.lock();
                for e in 0..e {
                    if let Some(s) = inner.map.slot_of(e) {
                        assert!(s < 2, "spare {spare}: expert {e} landed in parked slot {s}");
                    }
                }
            }
            // Never below the floor (`topk + 1 = 3`): a shrink to 1 stops at 3
            // — which is a grow from 2.
            assert_eq!(streamed.shrink_slots(1, topk + 1).unwrap(), 0);
            assert_eq!(streamed.active_slots(), 2, "a shrink never grows");
            assert_eq!(
                streamed.grow_slots(4),
                2 * per_slot,
                "two slots' bytes taken"
            );
            assert_eq!(streamed.active_slots(), 4);
            assert_eq!(streamed.stats().slot_reallocations, 2);
            for t in 0..4 {
                let id = ids(1, topk, 10, 11 + t);
                let xs = x(1, d, 0.0);
                let _ = run(&streamed, &xs, &id);
            }
            let used: std::collections::BTreeSet<usize> = {
                let inner = streamed.inner.lock();
                (0..e).filter_map(|x| inner.map.slot_of(x)).collect()
            };
            assert!(
                used.iter().any(|&s| s >= 2),
                "spare {spare}: the slots brought back are never used: {used:?}"
            );
        }
        std::fs::remove_file(path).ok();
    }

    /// Metal: a shrink reallocates the plane pools at the smaller slot
    /// count with the kept slots' planes carried over — the routed output
    /// of the resident experts is the same bits before and after — and
    /// the ledger's committed bytes fall by the slots released; a grow
    /// takes them back.
    #[cfg(feature = "metal")]
    #[test]
    fn a_shrink_reallocates_the_pools_and_the_ledger_falls_by_the_slots_released() {
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let (e, ff, d, topk) = (8usize, 32usize, 64usize, 2usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::F32);
        let up = stacked(e, ff, d, 2.0, GgmlDType::F32);
        let down = stacked(e, d, ff, 3.0, GgmlDType::F32);
        let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
        let streamed = Streamed::new(
            open(&path, "g"),
            open(&path, "u"),
            open(&path, "d"),
            4,
            LoaderArm::Pread,
            &dev,
        )
        .unwrap();
        let per_slot = streamed.bytes_per_slot();
        assert_eq!(streamed.warm(&[0, 1, 2]).unwrap(), 3);
        let xs = x(1, d, 0.25).to_device(&dev).unwrap();
        let id = Tensor::from_vec(vec![0u32, 2], (1, topk), &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let before = run(&streamed, &xs, &id);
        // The caller's drain (the admission's, on the refusal path): the
        // tier itself never waits for the device, so the test commits and
        // completes what is open the way the open-plan test does — a
        // readback, then the pool clock caught up. (The readback's own
        // buffer stays on the ledger in the free list: the figure is read
        // after it.)
        let clock = m.gpu_clock();
        Tensor::zeros(1, lloom_tensor::DType::F32, &dev)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        while clock.completed_count().0 < clock.current_epoch().0 {
            std::thread::yield_now();
        }
        let committed = m.ledger().committed();
        let released = streamed.shrink_slots(3, 1).unwrap();
        assert_eq!(released, per_slot, "one slot's planes released");
        assert_eq!(
            m.ledger().committed() + per_slot,
            committed,
            "the old pools' reservations left the ledger"
        );
        assert_eq!(streamed.active_slots(), 3);
        // Experts 0..3 are still resident in slots 0..3: the planes came
        // over, the output is the same bits.
        assert_eq!(streamed.inner.lock().map.resident().len(), 3);
        let after = run(&streamed, &xs, &id);
        assert_eq!(after, before, "the kept slots' planes moved bit for bit");
        let stats = streamed.stats();
        assert_eq!(
            (stats.slot_reallocations, stats.slot_bytes_released),
            (1, per_slot as u64)
        );
        // A grow takes the slot back; a fourth expert lands in it.
        assert_eq!(streamed.grow_slots(4), per_slot);
        assert_eq!(m.ledger().committed(), committed);
        let id4 = Tensor::from_vec(vec![3u32, 1], (1, topk), &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let _ = run(&streamed, &xs, &id4);
        assert_eq!(streamed.inner.lock().map.slot_of(3), Some(3));
        std::fs::remove_file(path).ok();
    }

    /// After a shrink the host plan groups rows to the slots in service:
    /// a tile whose rows together name more experts than the shrunk pool
    /// holds is split into groups that fit, never refused (the first
    /// yield run on the real model refused a prefill group planned at the
    /// pool's old width).
    #[test]
    fn a_shrunk_pool_plans_its_groups_to_the_slots_in_service() {
        let (e, ff, d, topk) = (10usize, 32usize, 32usize, 1usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::F32);
        let up = stacked(e, ff, d, 2.0, GgmlDType::F32);
        let down = stacked(e, d, ff, 3.0, GgmlDType::F32);
        let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
        let streamed = Streamed::new(
            open(&path, "g"),
            open(&path, "u"),
            open(&path, "d"),
            6,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        // Six rows, six distinct experts: one group in a 6-slot pool.
        let ids = Tensor::from_vec((0..6u32).collect::<Vec<_>>(), (6, topk), &Device::Cpu).unwrap();
        assert_eq!(streamed.plan(&ids).unwrap().len(), 1);
        streamed.shrink_slots(2, 1).unwrap();
        let groups = streamed.plan(&ids).unwrap();
        assert_eq!(
            groups.len(),
            3,
            "two slots in service: three groups of two rows"
        );
        let xs = x(6, d, 0.5);
        for g in &groups {
            let slots = streamed.ensure(g).unwrap();
            assert_eq!(slots.dims(), &[g.rows.len(), topk]);
        }
        let _ = xs;
        std::fs::remove_file(path).ok();
    }

    /// A group wider than the pool is refused rather than silently narrowed.
    #[test]
    fn a_row_wider_than_the_pool_is_refused() {
        let (e, ff, d) = (8usize, 32usize, 32usize);
        let gate = stacked(e, ff, d, 1.0, GgmlDType::F32);
        let up = stacked(e, ff, d, 2.0, GgmlDType::F32);
        let down = stacked(e, d, ff, 3.0, GgmlDType::F32);
        let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
        let streamed = Streamed::new(
            open(&path, "g"),
            open(&path, "u"),
            open(&path, "d"),
            3,
            LoaderArm::Pread,
            &Device::Cpu,
        )
        .unwrap();
        let id = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &Device::Cpu).unwrap();
        let g = streamed.plan(&id).unwrap();
        assert!(streamed.ensure(&g[0]).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn plan_rows_groups_by_union() {
        // rows: {0,1} {1,2} {3,4} {5,6} with 4 slots → {0,1,2} fits, {3,4}
        // would make 5 → cut; {3,4,5,6} fits → [0..2] [2..4]
        let ids = [0u32, 1, 1, 2, 3, 4, 5, 6];
        assert_eq!(plan_rows(&ids, 4, 2, 4), vec![0..2, 2..4]);
        assert_eq!(plan_rows(&ids, 4, 2, 3), vec![0..2, 2..3, 3..4]);
        assert_eq!(plan_rows(&ids, 4, 2, 8), vec![0..4]);
        // A pinned expert costs no capacity: with expert 1 pinned, {0,2,3,4}
        // is 4 of 4 → the first three rows are one group where they were two.
        let mut pinned = vec![false; 8];
        pinned[1] = true;
        assert_eq!(plan_rows_pinned(&ids, 4, 2, 4, &pinned), vec![0..3, 3..4]);
    }

    /// Metal: the prefetcher stages a layer's missing experts off the
    /// encoding thread and `ensure` lands them (the copy alone, on the
    /// compute encoder); the routed output is the synchronous tier's bit for
    /// bit, every landed expert was one the pass would have read, and what
    /// was staged and not needed goes back to the ring. A second hint on the
    /// warmed pool asks only for what is missing.
    #[cfg(feature = "metal")]
    #[test]
    fn prefetched_experts_land_bit_for_bit_and_leftovers_return() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let ring = StagingRing::shared(&m);
        let tier = |prefetch: bool| -> Streamed {
            let mut t = Streamed::with_staging(
                open(&path, "gate"),
                open(&path, "up"),
                open(&path, "down"),
                6,
                LoaderArm::Pread,
                &dev,
                Some(Arc::clone(&ring)),
            )
            .unwrap();
            if prefetch {
                t.set_prefetcher(Arc::new(Prefetcher::new(&m, Arc::clone(&ring))));
            }
            t
        };
        let plain = tier(false);
        let pre = tier(true);
        let xs = x(4, d, 0.5).to_device(&dev).unwrap();
        let id = ids(4, topk, e, 1).to_device(&dev).unwrap();
        // The hint is bounded by the pool: 6 of the 12 missing experts.
        let n = 6u64;
        pre.prefetch_ahead().unwrap();
        let t0 = std::time::Instant::now();
        while pre.prefetch_stats().staged < n {
            assert!(
                t0.elapsed() < std::time::Duration::from_secs(20),
                "the prefetcher staged {} of {n}",
                pre.prefetch_stats().staged
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(
            run(&plain, &xs, &id),
            run(&pre, &xs, &id),
            "the hint changed the output"
        );
        let s = pre.stats();
        let pf = pre.prefetch_stats();
        assert!(pf.staged >= n, "{pf:?}");
        // Every miss came through the threads: landed from the hint, or
        // read at the layer by `load_now` (`now`) — never by this thread.
        assert_eq!(pf.used, s.misses, "{pf:?} vs {s:?}");
        assert!(pf.now > 0 && pf.used - pf.now <= n, "{pf:?}");
        assert_eq!(s.prefetch_now, pf.now);
        assert_eq!(
            s.bytes_loaded,
            plain.stats().bytes_loaded,
            "a landed expert counts its bytes like a read one"
        );
        let resident = pre.resident().len() as u64;
        // The next hint on the warmed pool asks only for what is missing
        // (bounded by the pool), and releases what the pass left staged and
        // unused.
        pre.prefetch_ahead().unwrap();
        let pf2 = pre.prefetch_stats();
        assert_eq!(pf2.issued - pf.issued, (e as u64 - resident).min(n));
        assert_eq!((pf2.used - pf2.now) + pf2.wasted, n, "{pf2:?}");
        std::fs::remove_file(path).ok();
    }

    /// Metal, the predict arm's landing (`land_predicted`, #849's D with
    /// the hotlist's prediction): an expert the prefetcher staged for the
    /// tier and the usage profile ranks above the LRU resident lands
    /// before the plan maps the pool — the plan finds it resident (no
    /// flag), the pass computes its bytes (the resident tier's output
    /// for the token, bit for bit); a staged expert the profile ranks
    /// below every resident one goes back to the ring, nothing evicted
    /// for it.
    #[cfg(feature = "metal")]
    #[test]
    fn a_predicted_expert_lands_before_the_plan_maps_the_pool() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d) = (12usize, 32usize, 64usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let ring = StagingRing::shared(&m);
        let mut pre = Streamed::with_staging(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            6,
            LoaderArm::Pread,
            &dev,
            Some(Arc::clone(&ring)),
        )
        .unwrap();
        pre.set_prefetcher(Arc::new(Prefetcher::new(&m, Arc::clone(&ring))));
        let plain = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            6,
            LoaderArm::Pread,
            &dev,
        )
        .unwrap();
        // The pool warm with 0..5 on the host plan — one slot of six left
        // empty (two tokens run through; each token's slots released by its
        // `down`).
        let xw = x(1, d, 0.25).to_device(&dev).unwrap();
        let _ = run(&pre, &xw, &Tensor::new(&[[0u32, 1, 2]], &dev).unwrap());
        let _ = run(&pre, &xw, &Tensor::new(&[[3u32, 4, 4]], &dev).unwrap());
        assert_eq!(pre.resident().len(), 5, "{:?}", pre.resident());
        // The profile: expert 9 the hottest missing one. The hint stages as
        // many as the pool can land without evicting — one; `staged` counts
        // the warm's reads through the threads too, so from here.
        pre.inner.lock().map.usage[9] = 50;
        let staged0 = pre.prefetch_stats().staged;
        pre.prefetch_hot().unwrap();
        let t0 = std::time::Instant::now();
        while pre.prefetch_stats().staged < staged0 + 1 {
            assert!(
                t0.elapsed() < std::time::Duration::from_secs(20),
                "the prefetcher staged {} of 1",
                pre.prefetch_stats().staged - staged0
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // A full pool stages nothing: a second hint now adds no request.
        let issued = pre.prefetch_stats().issued;
        // A token wanting 9 (staged, hot) with 3 and 4 (resident): the plan
        // lands 9 in the empty slot and maps every expert.
        let tok = Tensor::new(&[[9u32, 3, 4]], &dev).unwrap();
        let (slots, rec) = pre.plan_device(&tok).unwrap().expect("a device plan");
        let s = pre.stats();
        assert_eq!((s.predict_landed, s.predict_wasted), (1, 0), "{s:?}");
        assert_eq!(s.evictions, 0, "a prediction never evicts: {s:?}");
        let resident = pre.resident();
        assert_eq!(resident.len(), 6, "{resident:?}");
        assert!(resident.iter().any(|&r| r as usize == 9), "{resident:?}");
        pre.prefetch_hot().unwrap();
        assert_eq!(
            pre.prefetch_stats().issued,
            issued,
            "a full pool stages nothing"
        );
        let r: Vec<u32> = rec.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(&r[3..6], &[9, 3, 4], "the record's ids: {r:?}");
        assert_eq!(&r[6..9], &[0, 0, 0], "a landed expert is no miss: {r:?}");
        let xs = x(1, d, 0.5).to_device(&dev).unwrap();
        let arm = MoeArm::for_rows(1);
        let (gt, u) = pre.gate_up(&xs, &slots, arm, xs.dtype()).unwrap();
        let h = (crate::nn::ops::silu(&gt).unwrap() * u).unwrap();
        let y = pre
            .down(&h, &slots, arm)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(!pre.settle(&rec).unwrap(), "nothing to settle");
        assert_eq!(
            y,
            run(&plain, &xs, &tok),
            "the landed expert's bytes differ from the resident tier's"
        );
        std::fs::remove_file(path).ok();
    }

    /// Metal: a batched settle's reads — three tiers' `load_now`s submitted
    /// before any is waited for (`settle_batch`) — all land for their
    /// layers: a `load_now`'s results never expire with the hints' leftovers
    /// (the "behind the pass" rule gave a landed read back to the ring
    /// before its layer's wait took it; every layer then waited the
    /// deadline out — #892's follow-up, the armed replay's settle with
    /// loads). Nothing wasted, every wait complete at once.
    #[cfg(feature = "metal")]
    #[test]
    fn a_batched_settles_reads_land_for_every_layer() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d) = (12usize, 32usize, 64usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let ring = StagingRing::shared(&m);
        let p = Prefetcher::new(&m, Arc::clone(&ring));
        let planes = [
            open(&path, "gate").plane_source(),
            open(&path, "up").plane_source(),
            open(&path, "down").plane_source(),
        ];
        // Three tiers' reads, submitted as the batched settle submits them,
        // then a pause for the threads to land them all.
        let tickets: Vec<_> = (0..3usize)
            .map(|key| p.submit_now(key, &[key as u32 * 3, key as u32 * 3 + 1], &planes))
            .collect();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let t0 = std::time::Instant::now();
        for (key, t) in tickets.into_iter().enumerate() {
            let have = p.wait_now(t);
            let mut got: Vec<u32> = have.iter().map(|s| s.expert).collect();
            got.sort_unstable();
            assert_eq!(
                got,
                vec![key as u32 * 3, key as u32 * 3 + 1],
                "tier {key}'s reads did not all land for its wait: {:?}",
                p.stats()
            );
            p.release(have);
        }
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "a wait ran to its deadline: {:?}",
            p.stats()
        );
        std::fs::remove_file(path).ok();
    }

    /// Metal, the `io` arm under a device plan (#849's D on that arm): a
    /// hint's loads in flight when the plan is made are waited for once on
    /// the queue — a wait encoded for the largest value, every entry
    /// settled by it (`plan_waits`, `io_inflight_planned`) — and the pass
    /// computes the landed bytes (the resident tier's output, bit for bit).
    #[cfg(feature = "metal")]
    #[test]
    fn a_device_plan_waits_once_for_the_io_arms_loads_in_flight() {
        use lloom_tensor::kernels::metal::io_queue::IoQueue;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d) = (12usize, 32usize, 64usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let queue = Arc::new(IoQueue::new(m.device()).unwrap());
        let mut io = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            6,
            LoaderArm::Io(Arc::clone(&queue)),
            &dev,
        )
        .unwrap();
        io.set_io_hints(true);
        let plain = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            6,
            LoaderArm::Pread,
            &dev,
        )
        .unwrap();
        // The GPU held busy ahead of the hint: the IO loads are ordered after
        // the compute encoded so far (`IoQueue::issue`), so they are pending
        // when the plan is made — the wait is required, and a plan without
        // it reads the slots before the bytes land.
        let a = Tensor::ones((2048, 2048), DType::F32, &dev).unwrap();
        let mut busy = a.clone();
        for _ in 0..6 {
            busy = busy.matmul(&a).unwrap();
        }
        // The hint, named (no fill from the profile): 9 and 10 take empty
        // slots and load on the IO queue; in flight until a use waits.
        io.prefetch_named(&[9, 10]).unwrap();
        assert_eq!(io.inflight_count(), 2, "{:?}", io.stats());
        // The device plan for a token wanting 9 and 10: the wait encoded
        // once, both entries settled by it, no miss flagged.
        let tok = Tensor::new(&[[9u32, 10, 9]], &dev).unwrap();
        let (slots, rec) = io.plan_device(&tok).unwrap().expect("a device plan");
        let s = io.stats();
        assert_eq!(io.inflight_count(), 0, "{s:?}");
        assert_eq!(s.io_inflight_planned + s.io_inflight_landed, 2, "{s:?}");
        assert_eq!(
            s.plan_waits,
            (s.io_inflight_planned > 0) as u64,
            "a wait iff a load was pending: {s:?}"
        );
        assert!(
            s.io_inflight_planned > 0,
            "the loads landed before the plan was made (the GPU was not held busy): {s:?}"
        );
        drop(busy);
        let r: Vec<u32> = rec.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(&r[6..9], &[0, 0, 0], "a hinted expert is no miss: {r:?}");
        let xs = x(1, d, 0.5).to_device(&dev).unwrap();
        let arm = MoeArm::for_rows(1);
        let (gt, u) = io.gate_up(&xs, &slots, arm, xs.dtype()).unwrap();
        let h = (crate::nn::ops::silu(&gt).unwrap() * u).unwrap();
        let y = io
            .down(&h, &slots, arm)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(!io.settle(&rec).unwrap(), "nothing to settle");
        assert_eq!(
            y,
            run(&plain, &xs, &tok),
            "the hinted experts' bytes differ from the resident tier's"
        );
        // A second plan with nothing in flight encodes no wait (R1).
        let tok2 = Tensor::new(&[[9u32, 10, 10]], &dev).unwrap();
        let (_, rec2) = io.plan_device(&tok2).unwrap().expect("a device plan");
        assert_eq!(io.stats().plan_waits, 1, "{:?}", io.stats());
        assert!(!io.settle(&rec2).unwrap());
        std::fs::remove_file(path).ok();
    }

    /// Metal: a miss read in pieces — each plane cut into `split` requests
    /// read by the pool at once, the last piece shorter — lands the same
    /// bytes as the expert read whole on the encoding thread: the routed
    /// output over decode rows and an evicting chunk equals the synchronous
    /// tier's bit for bit at every split, and every miss came through the
    /// threads. Planes of 128 KiB, so a split of 3 is three uneven pieces
    /// (48 + 48 + 32 KiB).
    #[cfg(feature = "metal")]
    #[test]
    fn a_miss_read_in_pieces_lands_the_same_bytes() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (10usize, 128usize, 256usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let plain = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            4,
            LoaderArm::Pread,
            &dev,
        )
        .unwrap();
        let x_decode: Vec<Tensor> = (0..6)
            .map(|t| x(1, d, t as f32).to_device(&dev).unwrap())
            .collect();
        let id_decode: Vec<Tensor> = (0..6)
            .map(|t| ids(1, topk, e, t).to_device(&dev).unwrap())
            .collect();
        let x_chunk = x(7, d, 0.5).to_device(&dev).unwrap();
        let id_chunk = ids(7, topk, e, 3).to_device(&dev).unwrap();
        let mut want = Vec::new();
        for (xs, id) in x_decode.iter().zip(&id_decode) {
            want.push(run(&plain, xs, id));
        }
        want.push(run(&plain, &x_chunk, &id_chunk));
        assert!(plain.stats().evictions > 0, "the reference never evicted");
        for split in [0usize, 1, 3] {
            let ring = StagingRing::shared(&m);
            let mut tier = Streamed::with_staging(
                open(&path, "gate"),
                open(&path, "up"),
                open(&path, "down"),
                4,
                LoaderArm::Pread,
                &dev,
                Some(Arc::clone(&ring)),
            )
            .unwrap();
            tier.set_prefetcher(Arc::new(Prefetcher::with_split(
                &m,
                Arc::clone(&ring),
                crate::nn::expert_prefetch::DEFAULT_MAX_STAGED_BYTES,
                split,
            )));
            let mut got = Vec::new();
            for (xs, id) in x_decode.iter().zip(&id_decode) {
                got.push(run(&tier, xs, id));
            }
            got.push(run(&tier, &x_chunk, &id_chunk));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g, w,
                    "split {split}: pass {i} differs from the synchronous tier"
                );
            }
            let s = tier.stats();
            let pf = tier.prefetch_stats();
            assert!(s.misses > 0 && s.evictions > 0, "split {split}: {s:?}");
            assert_eq!(
                s.prefetch_now, s.misses,
                "split {split}: a miss was not read by the threads: {s:?} {pf:?}"
            );
            assert_eq!(pf.staged, s.misses, "split {split}: {pf:?}");
        }
        std::fs::remove_file(path).ok();
    }

    /// Metal: a batched settle (`settle_batch`) over two layers sharing one
    /// prefetcher submits the second layer's reads before it waits for the
    /// first's (the prefetcher saw two tickets open at once), and settles to
    /// exactly what one settle after another does: the same experts in the
    /// same slots and the routed outputs on them bit for bit (the sequential
    /// settle against `Resident` is `device_plan_settles_to_the_host_plan`).
    #[cfg(feature = "metal")]
    #[test]
    fn a_batched_settle_reads_every_layers_misses_before_waiting() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (12usize, 32usize, 64usize, 3usize);
        let weights: Vec<[QTensor; 3]> = (0..2usize)
            .map(|l| {
                let o = l as f32 * 10.0;
                [
                    stacked(e, ff, d, 1.0 + o, dtype),
                    stacked(e, ff, d, 2.0 + o, dtype),
                    stacked(e, d, ff, 3.0 + o, dtype),
                ]
            })
            .collect();
        let paths: Vec<PathBuf> = weights
            .iter()
            .map(|w| write_gguf(&[("gate", &w[0]), ("up", &w[1]), ("down", &w[2])]))
            .collect();
        // Two layers over one ring and one prefetcher, as the loader builds a tier.
        let pair = || -> Vec<Streamed> {
            let ring = StagingRing::shared(&m);
            let pf = Arc::new(Prefetcher::new(&m, Arc::clone(&ring)));
            paths
                .iter()
                .map(|p| {
                    let mut t = Streamed::with_staging(
                        open(p, "gate"),
                        open(p, "up"),
                        open(p, "down"),
                        5,
                        LoaderArm::Pread,
                        &dev,
                        Some(Arc::clone(&ring)),
                    )
                    .unwrap();
                    t.set_prefetcher(Arc::clone(&pf));
                    t
                })
                .collect()
        };
        let batched = pair();
        let sequential = pair();
        let mut batched_misses = 0;
        for t in 0..6 {
            let id: Vec<Tensor> = (0..2)
                .map(|l| ids(1, topk, e, t * 2 + l).to_device(&dev).unwrap())
                .collect();
            let records = |tiers: &[Streamed]| -> Vec<Vec<u32>> {
                tiers
                    .iter()
                    .zip(&id)
                    .map(|(tier, id)| {
                        let (_, rec) = tier.plan_device(id).unwrap().unwrap();
                        rec.to_vec1().unwrap()
                    })
                    .collect()
            };
            let rb = records(&batched);
            let rs = records(&sequential);
            assert_eq!(rb, rs, "token {t}: the two pairs planned differently");
            let parts: Vec<(&Streamed, &[u32])> =
                batched.iter().zip(&rb).map(|(t, r)| (t, &r[..])).collect();
            let loads = Streamed::settle_batch(&parts).unwrap();
            let one_by_one: Vec<bool> = sequential
                .iter()
                .zip(&rs)
                .map(|(tier, r)| tier.settle_host(r).unwrap())
                .collect();
            assert_eq!(
                loads.iter().map(|&(redo, _)| redo).collect::<Vec<_>>(),
                one_by_one,
                "token {t}: which layers missed"
            );
            if loads.iter().all(|&(redo, _)| redo) {
                batched_misses += 1;
            }
            for (l, (b, q)) in batched.iter().zip(&sequential).enumerate() {
                assert_eq!(b.resident(), q.resident(), "token {t} layer {l}: the slots");
                let xs = x(1, d, (t * 2 + l) as f32).to_device(&dev).unwrap();
                // The same ops on both twins, so their LRU clocks stay equal.
                let out = |tier: &Streamed| -> Vec<f32> {
                    let (slots, rec) = tier.plan_device(&id[l]).unwrap().unwrap();
                    assert!(
                        !tier.settle(&rec).unwrap(),
                        "token {t} layer {l}: still missing"
                    );
                    let (g, u) = tier
                        .gate_up(&xs, &slots, MoeArm::MatVec, xs.dtype())
                        .unwrap();
                    let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
                    tier.down(&hh, &slots, MoeArm::MatVec)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                assert_eq!(out(b), out(q), "token {t} layer {l}: routed output");
            }
        }
        assert!(
            batched_misses > 0,
            "no token missed in both layers: the batch was never exercised"
        );
        let peak = batched[0].prefetch_stats().now_peak;
        if settle_batch_on() {
            assert!(
                peak >= 2,
                "the batched settle waited for a layer before submitting the next (peak {peak})"
            );
        }
        let seq_peak = sequential[0].prefetch_stats().now_peak;
        assert_eq!(
            seq_peak, 1,
            "one settle after another overlapped (peak {seq_peak})"
        );
        for p in paths {
            std::fs::remove_file(p).ok();
        }
    }

    /// The quarantined arm at the map: a miss takes a FREE slot; an evicted
    /// slot is quarantined at the current epoch and reused only once the
    /// clock's completed count is past it; a slot evicted while a device
    /// plan is open is not even quarantined until the plan settles
    /// (`release_deferred`), whatever the clock says.
    #[test]
    fn the_quarantined_arm_recycles_a_slot_only_after_its_readers() {
        let mut m = ResidencyMap::new(8, 4);
        m.set_spare(1);
        assert!(m.quarantined());
        assert_eq!(m.free_slots(), 4);
        for e in [5, 6, 7] {
            let t = m.begin();
            let s = m.take_free().unwrap();
            m.install(s, e);
            m.touch(s, t);
        }
        assert_eq!(
            (m.slot_of(5), m.slot_of(6), m.slot_of(7)),
            (Some(0), Some(1), Some(2))
        );
        let t = m.begin();
        assert_eq!(m.take_free(), Some(3), "the miss lands in the free slot");
        m.install(3, 1);
        m.touch(3, t);
        // Refill the spare: the LRU expert out, its slot held at epoch 10.
        assert_eq!(m.evict_lru(t, Epoch(10), false), Some(5));
        assert_eq!(m.slot_of(5), None);
        assert_eq!((m.held(), m.free_slots()), (1, 0));
        assert_eq!(
            m.recycle(Completed(10)),
            0,
            "epoch 10 is not finished at a count of 10"
        );
        assert_eq!(m.take_free(), None);
        assert_eq!(m.recycle(Completed(11)), 1);
        assert_eq!(m.take_free(), Some(0), "the recycled slot");
        // A plan open: the eviction is deferred, not quarantined.
        let t = m.begin();
        m.install(0, 2);
        m.touch(0, t);
        assert_eq!(m.evict_lru(t, Epoch(12), true), Some(6));
        assert_eq!(m.held(), 1);
        assert_eq!(
            m.recycle(Completed::ALL),
            0,
            "a slot evicted under an open plan was recycled before the plan settled"
        );
        m.release_deferred(Epoch(20));
        assert_eq!(m.recycle(Completed(20)), 0);
        assert_eq!(m.recycle(Completed(21)), 1);
        assert_eq!(m.take_free(), Some(1));
    }

    /// Metal, the direct read (Shared slots): a write-after-read planted
    /// between a device plan's slot map and its routed dispatch. The pass P
    /// maps experts a and b and its map has run; the host then loads c and
    /// d before P's experts are encoded. On the in-place arm the second load
    /// would write d into a slot P maps — P would compute on d's bytes and
    /// its record show no miss, a silent corruption (the hazard H2 names,
    /// `profiles-and-memory-design.md` §4.7; before #892 this arm did it,
    /// and this test read the corrupted output as its non-vacuity) — and is
    /// refused, counted (`h2_refusals`): P's output is the clean pass's bit
    /// for bit; so is c's, the pool being full. On the quarantined arm c
    /// takes the free slot, the eviction that refills the spare waits for
    /// P's settle — and so does its live-table write (H3): the table P's map
    /// read still holds the evicted expert until the settle publishes the
    /// write — and d, with no free slot left and a plan open, is refused
    /// the same way.
    #[cfg(feature = "metal")]
    #[test]
    fn a_load_never_writes_a_slot_an_open_device_plan_maps() {
        use crate::nn::expert_prefetch::Prefetcher;
        use lloom_tensor::kernels::metal::staging::StagingRing;
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let m = dev.as_metal_device().unwrap().clone();
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (6usize, 32usize, 64usize, 2usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let (a, b, c, dd, x_exp) = (0u32, 1u32, 2u32, 3u32, 4u32);
        let xs = x(1, d, 0.25).to_device(&dev).unwrap();
        let ids_ab = Tensor::from_vec(vec![a, b], (1, topk), &Device::Cpu)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let one = |id: u32| {
            Tensor::from_vec(vec![id], (1, 1), &Device::Cpu)
                .unwrap()
                .to_device(&dev)
                .unwrap()
        };
        let tier = |spare: usize| -> Streamed {
            let ring = StagingRing::shared(&m);
            let mut t = Streamed::with_staging_direct(
                open(&path, "gate"),
                open(&path, "up"),
                open(&path, "down"),
                3,
                LoaderArm::Pread,
                &dev,
                Some(Arc::clone(&ring)),
                true,
            )
            .unwrap();
            t.set_spare_slots(spare).unwrap();
            t.set_prefetcher(Arc::new(Prefetcher::new(&m, Arc::clone(&ring))));
            t
        };
        let routed = |t: &Streamed, slots: &Tensor| -> Vec<f32> {
            let arm = MoeArm::for_rows(xs.dim(0).unwrap());
            let (g, u) = t.gate_up(&xs, slots, arm, xs.dtype()).unwrap();
            let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
            t.down(&hh, slots, arm)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let clock = m.gpu_clock();
        let drain = || {
            // Commit and wait what is open, then let the pool clock catch up
            // (its completion handler runs on another thread).
            Tensor::zeros(1, lloom_tensor::DType::F32, &dev)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            while clock.completed_count().0 < clock.current_epoch().0 {
                std::thread::yield_now();
            }
        };
        // The clean pass: P alone.
        let want = {
            let t = tier(0);
            t.warm(&[a, b]).unwrap();
            let (slots, rec) = t.plan_device(&ids_ab).unwrap().unwrap();
            let out = routed(&t, &slots);
            assert!(!t.settle(&rec).unwrap());
            out
        };
        for spare in [0usize, 1] {
            let t = tier(spare);
            // In place the pool is full; quarantined it keeps its spare free.
            let warm: &[u32] = if spare == 0 { &[a, b, x_exp] } else { &[a, b] };
            assert_eq!(t.warm(warm).unwrap(), warm.len());
            let (slots, rec) = t.plan_device(&ids_ab).unwrap().unwrap();
            // P's slot map has run; its experts are not encoded yet.
            let mapped: Vec<u32> = slots.flatten_all().unwrap().to_vec1().unwrap();
            drain();
            let g = t.plan(&one(c)).unwrap();
            let c_load = t.ensure(&g[0]);
            // H3, the quarantined arm: c's landing refilled the spare by
            // evicting a or b under the open plan; the host map says so, the
            // live table P reads does not — yet.
            let evicted_entry = (spare == 1).then(|| {
                let inner = t.inner.lock();
                let old = [a, b]
                    .into_iter()
                    .find(|&x| inner.map.slot_of(x as usize).is_none())
                    .expect("the replenish evicted a or b");
                let live = inner
                    .live
                    .as_ref()
                    .expect("the quarantined arm keeps a live table");
                let v = unsafe {
                    (live.buf.contents() as *const u32)
                        .add(old as usize)
                        .read_volatile()
                };
                (old, v)
            });
            drain();
            let g = t.plan(&one(dd)).unwrap();
            let d_load = t.ensure(&g[0]);
            // H1: P's routed kernels are encoded now, after the drains' cuts
            // — in a later command buffer than its slot map; the slots they
            // read must carry the kernels' epoch, not the plan's.
            let kernel_epoch = clock.current_epoch();
            let got = routed(&t, &slots);
            {
                let inner = t.inner.lock();
                for &s in &mapped {
                    assert!(
                        inner.map.reader_epoch(s as usize) >= kernel_epoch,
                        "H1: slot {s} is stamped {} under a kernel encoded in {kernel_epoch}",
                        inner.map.reader_epoch(s as usize)
                    );
                }
                assert!(t.stats().h1_restamps >= 1, "{:?}", t.stats());
            }
            let flagged = rec.to_vec1::<u32>().unwrap()[2 * topk..]
                .iter()
                .any(|&f| f != 0);
            assert!(!flagged, "spare {spare}: P's map ran before any eviction");
            if spare == 0 {
                // H2 on the in-place arm: every slot is resident, so c and d
                // could only take a slot by evicting under the open plan, and
                // the overwrite cannot be deferred in place — refused.
                for (name, load) in [("c", c_load), ("d", d_load)] {
                    let err = load
                        .err()
                        .unwrap_or_else(|| {
                            panic!(
                                "the in-place arm loaded {name} into a slot under the open plan P"
                            )
                        })
                        .to_string();
                    assert!(
                        err.contains("while a device plan is open"),
                        "the refusal names the open plan: {err}"
                    );
                }
                let inner = t.inner.lock();
                assert!(inner.map.slot_of(c as usize).is_none());
                assert!(inner.map.slot_of(dd as usize).is_none());
                drop(inner);
                assert!(
                    mapped.iter().all(|s| *s < 3),
                    "P maps {mapped:?} in a 3-slot pool"
                );
                assert_eq!(got, want, "a load wrote a slot the open plan P maps");
                assert_eq!(t.stats().h2_refusals, 2, "{:?}", t.stats());
                assert!(!t.settle(&rec).unwrap());
            } else {
                c_load.expect("the quarantined arm loads c into its free slot");
                let (old, v) = evicted_entry.unwrap();
                let old_slot = mapped[[a, b].iter().position(|&x| x == old).unwrap()];
                assert_eq!(
                    v, old_slot,
                    "H3: the live table lost expert {old} while P was open"
                );
                assert!(
                    d_load.is_err(),
                    "a load with no free slot under an open plan was not refused"
                );
                assert_eq!(got, want, "a load wrote a slot the open plan P maps");
                assert!(!t.settle(&rec).unwrap());
                // The settle published the deferred write.
                let v = {
                    let inner = t.inner.lock();
                    let live = inner.live.as_ref().unwrap();
                    unsafe {
                        (live.buf.contents() as *const u32)
                            .add(old as usize)
                            .read_volatile()
                    }
                };
                assert_eq!(v, NONE, "H3: the settle did not publish the eviction");
                // The warm's a and b, and c: every landing took a free slot.
                let st = t.stats();
                assert_eq!((st.free_takes, st.inplace_takes), (3, 0), "{st:?}");
                // Two writes waited for the settle: the eviction's NONE and
                // c's own entry.
                assert_eq!((st.h2_refusals, st.h3_deferred_writes), (1, 2), "{st:?}");
            }
        }
        std::fs::remove_file(path).ok();
    }

    /// Metal: the quarantined arm's device table is written in place — the
    /// tensor a device plan reads is the same buffer before and after a
    /// settle loads — and a plan after the settle maps the loaded experts
    /// (no flag) onto the slots the host map holds them in.
    #[cfg(feature = "metal")]
    #[test]
    fn the_live_table_is_written_in_place() {
        let dev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let dtype = GgmlDType::F32;
        let (e, ff, d, topk) = (8usize, 32usize, 64usize, 3usize);
        let gate = stacked(e, ff, d, 1.0, dtype);
        let up = stacked(e, ff, d, 2.0, dtype);
        let down = stacked(e, d, ff, 3.0, dtype);
        let path = write_gguf(&[("gate", &gate), ("up", &up), ("down", &down)]);
        let mut t = Streamed::new(
            open(&path, "gate"),
            open(&path, "up"),
            open(&path, "down"),
            5,
            LoaderArm::Pread,
            &dev,
        )
        .unwrap();
        t.set_spare_slots(1).unwrap();
        assert!(t.live_table());
        let id = ids(1, topk, e, 2).to_device(&dev).unwrap();
        let table_ptr = |t: &Streamed| {
            let inner = t.inner.lock();
            let tt = &inner.live.as_ref().unwrap().tensor;
            let (st, _) = tt.storage_and_layout();
            match &*st {
                lloom_tensor::Storage::Metal(m) => m.buffer().contents() as usize,
                _ => unreachable!(),
            }
        };
        let before = table_ptr(&t);
        let (_, rec) = t.plan_device(&id).unwrap().unwrap();
        assert!(t.settle(&rec).unwrap(), "a cold pool missed nothing");
        let (slots, rec) = t.plan_device(&id).unwrap().unwrap();
        assert!(
            !t.settle(&rec).unwrap(),
            "the loaded experts are still flagged"
        );
        assert_eq!(table_ptr(&t), before, "the table was re-uploaded");
        let want: Vec<u32> = {
            let inner = t.inner.lock();
            id.flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()
                .iter()
                .map(|&x| inner.map.slot_of(x as usize).unwrap() as u32)
                .collect()
        };
        assert_eq!(slots.flatten_all().unwrap().to_vec1::<u32>().unwrap(), want);
        std::fs::remove_file(path).ok();
    }
}
