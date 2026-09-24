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

use lloom_tensor::quantized::gguf_file::Content;
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
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
    /// `slot_ids [rows, topk]`: each `[rows, topk, n_ff]`.
    fn gate_up(&self, x: &Tensor, slot_ids: &Tensor) -> Result<(Tensor, Tensor)>;

    /// `down` projection of `h [rows, topk, n_ff]` through `slot_ids`:
    /// `[rows, topk, hidden]`.
    fn down(&self, h: &Tensor, slot_ids: &Tensor) -> Result<Tensor>;

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

    /// Experts loaded so far (the miss counter), 0 when nothing streams —
    /// a pass that leaves it unchanged found everything resident.
    fn misses_so_far(&self) -> u64 {
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

    /// Read a device plan's record back (the one host sync, at the end of
    /// the pass, for every layer at once), touch the residency of what the
    /// pass used and load what it lacked. `true` when the pass computed on
    /// a missing expert — its result is wrong and the pass is replayed; the
    /// experts are resident now.
    fn settle(&self, _record: &Tensor) -> Result<bool> {
        Ok(false)
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

    /// The prefetcher's counters (shared by the tier's layers), zeros
    /// without one.
    fn prefetch_stats(&self) -> PrefetchStats {
        PrefetchStats::default()
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
    fn gate_up(&self, x: &Tensor, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.gate.indexed_moe_forward(x, ids)?,
            self.up.indexed_moe_forward(x, ids)?,
        ))
    }
    #[inline]
    fn down(&self, h: &Tensor, ids: &Tensor) -> Result<Tensor> {
        self.down.indexed_moe_forward(h, ids)
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

/// Expert id → slot, slot → expert, an LRU tick per slot and pin flags. The
/// query is one indexed load; the victim search is over the (small) slot
/// count and runs only on a miss.
#[derive(Debug, Clone)]
pub struct ResidencyMap {
    slot_of: Vec<u32>,
    /// Uses per expert (a hit or a miss each time a pass needs it): the
    /// frequency profile the hotlist is (§20.3 R2).
    usage: Vec<u32>,
    expert_in: Vec<u32>,
    last_use: Vec<u64>,
    pinned: Vec<bool>,
    /// Per slot, the `io` arm's event value its expert lands on — an
    /// in-flight entry (a hint issued, the DMA not yet waited for); 0 when
    /// the slot holds a landed expert or nothing (§20.4a, DS4's
    /// `inflight_seq`). The victim search skips in-flight slots while any
    /// other is available; `install` clears it.
    inflight: Vec<u64>,
    /// Per slot, the GPU epoch (the pool clock's) of the last command
    /// buffer that read or wrote it: a direct read into the slot waits for
    /// it to complete before the bytes land (`Prefetcher::read_direct`).
    last_epoch: Vec<u64>,
    tick: u64,
}

const NONE: u32 = u32::MAX;

impl ResidencyMap {
    pub fn new(n_experts: usize, n_slots: usize) -> Self {
        Self {
            slot_of: vec![NONE; n_experts],
            usage: vec![0; n_experts],
            expert_in: vec![NONE; n_slots],
            last_use: vec![0; n_slots],
            pinned: vec![false; n_slots],
            inflight: vec![0; n_slots],
            last_epoch: vec![0; n_slots],
            tick: 0,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.expert_in.len()
    }

    /// The slot holding expert `e`, if resident.
    pub fn slot_of(&self, e: usize) -> Option<usize> {
        match self.slot_of[e] {
            NONE => None,
            s => Some(s as usize),
        }
    }

    /// Resident experts, in slot order.
    pub fn resident(&self) -> Vec<u32> {
        self.expert_in
            .iter()
            .copied()
            .filter(|&e| e != NONE)
            .collect()
    }

    /// Open a new use epoch (one per `ensure`).
    fn begin(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn touch(&mut self, slot: usize, tick: u64) {
        self.last_use[slot] = tick;
    }

    /// The least-recently-used unpinned slot not used in epoch `tick`, and
    /// whether it held an expert (an eviction). An in-flight slot is taken
    /// only when no other is (the IO queue is in order, so the second load
    /// lands after the first; the hint's bytes were read for nothing).
    fn victim(&self, tick: u64) -> Option<(usize, bool)> {
        self.victim_settled(tick, u64::MAX)
    }

    /// [`Self::victim`] preferring a slot whose last GPU reader's epoch is
    /// below `settled_before` (the pool clock's completed count): a direct
    /// read into it waits for nothing. Falls back to any slot.
    fn victim_settled(&self, tick: u64, settled_before: u64) -> Option<(usize, bool)> {
        self.victim_in(tick, true, settled_before)
            .or_else(|| self.victim_in(tick, true, u64::MAX))
            .or_else(|| self.victim_in(tick, false, u64::MAX))
    }

    /// [`Self::victim`] over the slots not in flight (`skip_inflight`) or
    /// over all of them, and with a last reader's epoch below
    /// `settled_before`.
    fn victim_in(
        &self,
        tick: u64,
        skip_inflight: bool,
        settled_before: u64,
    ) -> Option<(usize, bool)> {
        let mut best: Option<(usize, u64)> = None;
        for s in 0..self.expert_in.len() {
            if self.pinned[s]
                || self.last_use[s] == tick
                || (skip_inflight && self.inflight[s] != 0)
                || self.last_epoch[s] >= settled_before
            {
                continue;
            }
            // An empty slot wins outright.
            if self.expert_in[s] == NONE {
                return Some((s, false));
            }
            match best {
                Some((_, t)) if t <= self.last_use[s] => {}
                _ => best = Some((s, self.last_use[s])),
            }
        }
        best.map(|(s, _)| (s, true))
    }

    fn install(&mut self, slot: usize, e: usize) {
        let old = self.expert_in[slot];
        if old != NONE {
            self.slot_of[old as usize] = NONE;
        }
        self.expert_in[slot] = e as u32;
        self.slot_of[e] = slot as u32;
        self.inflight[slot] = 0;
    }

    /// Slots whose expert is in flight (an `io` hint not yet waited for).
    pub fn inflight_count(&self) -> usize {
        self.inflight.iter().filter(|&&v| v != 0).count()
    }

    /// Forget the in-flight state of every slot whose loads landed (event
    /// value ≤ `done`): a landed expert is a resident like any other.
    /// Returns how many.
    fn settle_inflight(&mut self, done: u64) -> u64 {
        let mut n = 0;
        for v in self.inflight.iter_mut() {
            if *v != 0 && *v <= done {
                *v = 0;
                n += 1;
            }
        }
        n
    }

    fn set_pinned(&mut self, slot: usize, pinned: bool) {
        self.pinned[slot] = pinned;
    }

    pub fn pinned_count(&self) -> usize {
        self.pinned.iter().filter(|&&p| p).count()
    }

    /// Slots holding no expert.
    pub fn free_slots(&self) -> usize {
        self.expert_in.iter().filter(|&&e| e == NONE).count()
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
    /// Misses read straight into their slots (`pread_direct_on`), and the
    /// reader threads' time waiting for the slots' last readers' epochs.
    pub direct_loads: AtomicU64,
    pub direct_wait_ns: AtomicU64,
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
    pub direct_loads: u64,
    pub direct_wait_ns: u64,
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
            direct_loads: self.direct_loads.load(Ordering::Relaxed),
            direct_wait_ns: self.direct_wait_ns.load(Ordering::Relaxed),
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
        self.direct_loads += o.direct_loads;
        self.direct_wait_ns += o.direct_wait_ns;
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
    /// The `pread` arm's staging ring on Metal (reused once its copy's
    /// command buffer completes; never a pool bucket, never a wait) — one
    /// ring shared by every layer of the tier when built by the loader.
    #[cfg(feature = "metal")]
    staging: Option<lloom_tensor::kernels::metal::staging::SharedStagingRing>,
    /// The epoch of a group whose slot ids `ensure` handed out and whose
    /// dispatches are not all encoded yet (`down` closes it): the `io` arm's
    /// hint between the two must not give its slots away — the load is
    /// ordered after what IS encoded, nothing orders it after what is not.
    open_epoch: Option<u64>,
    /// That group's slots: stamped with the GPU epoch at `down` (their last
    /// reader's command buffer), for the direct read's wait.
    open_slots: Vec<usize>,
    /// The pool clock (Metal), for those stamps.
    #[cfg(feature = "metal")]
    clock: Option<Arc<lloom_tensor::kernels::metal::GpuClock>>,
    /// The slot pools are Shared and a miss is read straight into its slot
    /// (`pread_direct_on`, the `pread` arm on Metal).
    direct: bool,
}

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
        #[cfg(not(feature = "metal"))]
        let _ = staging;
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
        // under the direct read (the CPU writes a miss into its slot).
        #[cfg(feature = "metal")]
        let direct = matches!(loader, LoaderArm::Pread)
            && matches!(device, Device::Metal(_))
            && pread_direct_on();
        #[cfg(not(feature = "metal"))]
        let direct = false;
        let pool = |s: &ExpertSource| -> Result<SlotPool> {
            let (n, k) = s.dims();
            Ok(SlotPool {
                qt: QTensor::zeros_persistent_with((n_slots, n, k), s.dtype(), device, direct)?,
                plane_bytes: s.plane_bytes(),
            })
        };
        let pools = [pool(&sources[0])?, pool(&sources[1])?, pool(&sources[2])?];
        #[cfg(feature = "metal")]
        let clock = match device {
            Device::Metal(d) => Some(d.gpu_clock()),
            _ => None,
        };
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner {
                pools,
                sources,
                map: ResidencyMap::new(n_experts, n_slots),
                loader,
                device: device.clone(),
                #[cfg(feature = "metal")]
                staging,
                table: None,
                table_stale: true,
                open_epoch: None,
                open_slots: Vec::new(),
                #[cfg(feature = "metal")]
                clock,
                direct,
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
        self.resolve(&mut inner, &take, tick)?;
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
    fn resolve(&self, inner: &mut Inner, experts: &[u32], tick: u64) -> Result<Vec<usize>> {
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
        for &e in experts {
            let e = e as usize;
            if e >= self.n_experts {
                lloom_tensor::bail!("expert {e} out of {}", self.n_experts)
            }
            inner.map.usage[e] = inner.map.usage[e].saturating_add(1);
            let slot = match inner.map.slot_of(e) {
                Some(s) => {
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "metal")]
                    if inner.map.inflight[s] != 0 {
                        wait_value = wait_value.max(inner.map.inflight[s]);
                        inner.map.inflight[s] = 0;
                        self.counters
                            .io_inflight_hits
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    s
                }
                None => {
                    let (s, evicted) = match inner.map.victim_settled(tick, settled_before) {
                        Some(v) => v,
                        None => lloom_tensor::bail!(
                            "the slot pool ({} slots, {} pinned) cannot hold this group's {} experts",
                            self.n_slots,
                            inner.map.pinned_count(),
                            experts.len()
                        ),
                    };
                    if inner.map.inflight[s] != 0 {
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
                    let (now, later): (Vec<(usize, usize)>, Vec<(usize, usize)>) = misses
                        .iter()
                        .copied()
                        .partition(|&(_, s)| inner.map.last_epoch[s] < settled);
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
                            epoch: inner.map.last_epoch[s],
                        });
                    }
                    let n_direct = jobs.len();
                    let waited = p.read_direct(self.key, jobs);
                    self.counters
                        .direct_loads
                        .fetch_add(n_direct as u64, Ordering::Relaxed);
                    self.counters
                        .direct_wait_ns
                        .fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
                    self.counters
                        .load_now_ns
                        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    self.counters
                        .bytes_loaded
                        .fetch_add(bytes, Ordering::Relaxed);
                    if n_direct > 0 {
                        inner.table_stale = true;
                    }
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
                    let mut got: std::collections::HashMap<
                        u32,
                        crate::nn::expert_prefetch::Staged,
                    > = p
                        .load_now(self.key, &want, &planes)
                        .into_iter()
                        .map(|s| (s.expert, s))
                        .collect();
                    self.counters
                        .load_now_ns
                        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let mut rest = Vec::new();
                    for &(e, s) in &misses {
                        match got.remove(&(e as u32)) {
                            Some(st) => landed.push((e, s, st)),
                            None => rest.push((e, s)),
                        }
                    }
                    self.counters
                        .prefetch_now
                        .fetch_add((misses.len() - rest.len()) as u64, Ordering::Relaxed);
                    misses = rest;
                }
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
        Ok(slots)
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
        let tick = inner.map.tick;
        let protect = inner.open_epoch.unwrap_or(u64::MAX);
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
            let Some((s, false)) = inner.map.victim_in(protect, true, u64::MAX) else {
                break;
            };
            inner.map.install(s, e);
            // In flight from here (the value below once issued), so the
            // next pick of this loop skips it.
            inner.map.inflight[s] = u64::MAX;
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
            inner.map.inflight[s] = value;
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
        let epoch = clock.as_ref().map(|c| c.current_epoch()).unwrap_or(0);
        for (_, s, st) in landed {
            for (i, pool) in pools.iter_mut().enumerate() {
                let plane = pool.plane_bytes;
                let (entry, buf) = &st.planes[i];
                pool.qt
                    .write_staged(s * plane, plane, Arc::clone(buf), &mut ring, *entry)?;
                bytes += plane as u64;
            }
            // The copy writes the slot in this command buffer.
            map.last_epoch[s] = epoch;
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
            inner.map.last_epoch[s] = epoch;
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
    /// The host plan over ids on the host: the greedy row grouping to the
    /// pool, counted.
    fn plan_rows_host(&self, host: &[u32], rows: usize, topk: usize) -> Vec<RoutedGroup> {
        self.counters.plans.fetch_add(1, Ordering::Relaxed);
        // The pinned experts are resident whatever the group: a row's
        // pinned experts do not count against the unpinned capacity.
        let (capacity, pinned) = {
            let inner = self.inner.lock();
            let pinned: Vec<bool> = (0..self.n_experts)
                .map(|e| inner.map.slot_of(e).is_some_and(|s| inner.map.pinned[s]))
                .collect();
            (self.n_slots - inner.map.pinned_count(), pinned)
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
    *ON.get_or_init(|| {
        std::env::var("LLOOM_PREAD_DIRECT")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

impl Inner {
    /// Epochs the pool clock shows complete (`u64::MAX` off Metal: nothing
    /// to wait for).
    fn completed_epochs(&self) -> u64 {
        #[cfg(feature = "metal")]
        {
            self.clock
                .as_ref()
                .map(|c| c.completed_count())
                .unwrap_or(u64::MAX)
        }
        #[cfg(not(feature = "metal"))]
        {
            u64::MAX
        }
    }

    /// The pool clock's current epoch (0 off Metal).
    fn current_epoch(&self) -> u64 {
        #[cfg(feature = "metal")]
        {
            self.clock.as_ref().map(|c| c.current_epoch()).unwrap_or(0)
        }
        #[cfg(not(feature = "metal"))]
        {
            0
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

/// `QWEN4EXP_TRACE_SETTLE=1` (read once): print every settle's ids, flags
/// and misses — a debugging instrument, never the shipping path.
fn trace_settle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN4EXP_TRACE_SETTLE").is_ok())
}

/// The device-side plan: `ids [n]` (applied to) through the residency
/// `table [n_experts]` (captured; `NONE` = absent) → flat `[3·n]` u32: the
/// slot ids (a miss reads slot 0) | the ids | a miss flag per id. The CPU
/// path is the same lookup.
struct SlotMap {
    table: Tensor,
    n_experts: usize,
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
        let output = device
            .new_buffer_builder()
            .with_size_for(3 * n, lloom_tensor::DType::U32)
            .with_label("moe_slot_map")
            .build()?;
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

impl Streamed {
    /// The residency table on the device, rebuilt when a settle changed it.
    fn device_table(&self, inner: &mut Inner) -> Result<Tensor> {
        if inner.table_stale || inner.table.is_none() {
            let t = Tensor::from_vec(inner.map.slot_of.clone(), self.n_experts, &inner.device)?;
            inner.table = Some(t);
            inner.table_stale = false;
        }
        Ok(inner.table.as_ref().expect("built above").clone())
    }
}

impl ExpertResidency for Streamed {
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
            missing.truncate(self.n_slots.saturating_sub(inner.map.pinned_count()));
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
            let bound = self.n_slots.saturating_sub(inner.map.pinned_count());
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

    fn misses_so_far(&self) -> u64 {
        self.counters.misses.load(Ordering::Relaxed)
    }

    fn plan_device(&self, ids: &Tensor) -> Result<Option<(Tensor, Tensor)>> {
        let (rows, topk) = ids.dims2()?;
        let n = rows * topk;
        let ids = ids.flatten_all()?.contiguous()?;
        let table = {
            let mut inner = self.inner.lock();
            // The pass may read any resident slot: stamp them all with this
            // epoch (a direct read into one waits for it).
            let epoch = inner.current_epoch();
            for e in inner.map.last_epoch.iter_mut() {
                *e = epoch;
            }
            self.device_table(&mut inner)?
        };
        self.counters.device_plans.fetch_add(1, Ordering::Relaxed);
        let plan = ids.apply_op1_no_bwd(&SlotMap {
            table,
            n_experts: self.n_experts,
        })?; // [3·n]
        let slots = plan.narrow(0, 0, n)?.reshape((rows, topk))?;
        Ok(Some((slots, plan)))
    }

    fn settle(&self, record: &Tensor) -> Result<bool> {
        let v: Vec<u32> = record.flatten_all()?.to_vec1()?;
        if !v.len().is_multiple_of(3) {
            lloom_tensor::bail!("settle: a plan record is [3·n], got {}", v.len())
        }
        let n = v.len() / 3;
        let ids = &v[n..2 * n];
        let flags = &v[2 * n..];
        let mut distinct: Vec<u32> = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let mut inner = self.inner.lock();
        let tick = inner.map.begin();
        // What the pass used and found: touched first, so a load below never
        // evicts an expert this pass needs.
        let mut missing: Vec<u32> = Vec::new();
        for &e in &distinct {
            match inner.map.slot_of(e as usize) {
                Some(s) => {
                    inner.map.touch(s, tick);
                    inner.map.usage[e as usize] = inner.map.usage[e as usize].saturating_add(1);
                    self.counters.hits.fetch_add(1, Ordering::Relaxed);
                }
                // Counted by `resolve` below.
                None => missing.push(e),
            }
        }
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
        if missing.is_empty() {
            if flagged {
                lloom_tensor::bail!(
                    "settle: the device plan flagged a miss the residency map does not show"
                )
            }
            return Ok(false);
        }
        self.resolve(&mut inner, &missing, tick)?;
        self.counters.settled_misses.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn plan(&self, ids: &Tensor) -> Result<Vec<RoutedGroup>> {
        let (rows, topk) = ids.dims2()?;
        let t0 = std::time::Instant::now();
        let host: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?;
        self.counters
            .readback_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(self.plan_rows_host(&host, rows, topk))
    }

    fn plan_host(&self, ids: &[u32], rows: usize, topk: usize) -> Result<Option<Vec<RoutedGroup>>> {
        if ids.len() != rows * topk {
            lloom_tensor::bail!("plan_host: {} ids for {rows} rows × {topk}", ids.len())
        }
        Ok(Some(self.plan_rows_host(ids, rows, topk)))
    }

    fn ensure(&self, group: &RoutedGroup) -> Result<Tensor> {
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
        Tensor::from_vec(slot_ids, (group.rows.len(), topk), &inner.device)
    }

    fn gate_up(&self, x: &Tensor, slot_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let inner = self.inner.lock();
        Ok((
            inner.pools[0].qt.indexed_moe_forward(x, slot_ids)?,
            inner.pools[1].qt.indexed_moe_forward(x, slot_ids)?,
        ))
    }

    fn down(&self, h: &Tensor, slot_ids: &Tensor) -> Result<Tensor> {
        let mut inner = self.inner.lock();
        let y = inner.pools[2].qt.indexed_moe_forward(h, slot_ids)?;
        // The group's last dispatch is encoded: its slots may be given away
        // — to a direct read once this command buffer has run.
        inner.open_epoch = None;
        let epoch = inner.current_epoch();
        let slots = std::mem::take(&mut inner.open_slots);
        for s in slots {
            inner.map.last_epoch[s] = epoch;
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
        let mut ys = Vec::new();
        for g in &groups {
            let slot_ids = r.ensure(g).unwrap();
            let xg = x3.narrow(0, g.rows.start, g.rows.len()).unwrap();
            let (gt, u) = r.gate_up(&xg, &slot_ids).unwrap();
            let h = (crate::nn::ops::silu(&gt).unwrap() * u).unwrap();
            ys.push(r.down(&h, &slot_ids).unwrap());
        }
        let y = Tensor::cat(&ys, 0).unwrap();
        y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Streamed == Resident bit for bit over decode rows and a chunk whose
    /// expert union exceeds the pool (several groups, evictions), for a float
    /// stack and a real block type; the counters prove the streaming ran.
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
                    let (g, u) = streamed.gate_up(&x3, &slots2).unwrap();
                    let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
                    streamed.down(&hh, &slots2).unwrap()
                };
                let want = {
                    let x3 = xs.reshape((1, 1, d)).unwrap();
                    let (g, u) = resident.gate_up(&x3, &id).unwrap();
                    let hh = (crate::nn::ops::silu(&g).unwrap() * u).unwrap();
                    resident.down(&hh, &id).unwrap()
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
}
