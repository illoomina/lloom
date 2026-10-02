//! Dense weight groups (`profiles-and-memory-design.md` §4.8, #892 A6a):
//! a set of tensors that load and evict together — a dense layer's
//! matrices, a row block of `lm_head` — streamed through a
//! [`SlotCache<WeightGroupId, Dense, Belady>`] over slots of one size, the
//! plan's order of use giving the prefetch (`d` groups ahead) and the
//! eviction (Belady: the group needed furthest ahead goes).
//!
//! The groups' storage is one backing tensor `[slots, slot_elems]` of the
//! groups' dtype; a group's part is a contiguous view into its slot
//! (`Tensor::narrow`), so a dispatch binds **the slot's buffer at the
//! part's offset** — an identity fixed for as long as the group stays in
//! that slot (§4.8 R2: a recorded plan binds slots, never a per-load
//! buffer). A load is `slice_set` into the slot: on Metal an encoder copy
//! ordered on the queue behind the slot's last reader, so the host waits
//! for nothing and nothing is encoded for a group already resident
//! (R1: at full quota the pass encodes no load and no wait).
//!
//! Liveness is the cache's: a slot leaving a group retires at its last
//! reader's epoch and is reused once the clock covers it (`Retire<I>`,
//! §4.5). The source of a group's bytes is the owner's (`GroupSource`): a
//! safetensors map, a GGUF plane source, a test's tensors.

use crate::nn::slot_cache::{Belady, Dense, SlotCache, SlotId, WeightGroupId};
use crate::nn::zeros as zs;
use lloom_tensor::retire::{Completed, Epoch};
use lloom_tensor::{DType, Device, Result, Shape, Tensor};

/// One tensor of a group: its name, its shape, and where it sits in the
/// slot (in elements of the groups' dtype).
#[derive(Debug, Clone)]
pub struct GroupPart {
    pub name: String,
    pub shape: Shape,
    pub offset: usize,
}

impl GroupPart {
    pub fn elems(&self) -> usize {
        self.shape.elem_count()
    }
}

/// A group: its parts laid end to end in a slot.
#[derive(Debug, Clone)]
pub struct GroupSpec {
    pub id: WeightGroupId,
    pub parts: Vec<GroupPart>,
    /// Elements the group occupies (the sum of its parts').
    pub elems: usize,
}

impl GroupSpec {
    /// Lay `parts` (name, shape) end to end.
    pub fn new(id: WeightGroupId, parts: Vec<(String, Shape)>) -> Self {
        let mut offset = 0usize;
        let parts = parts
            .into_iter()
            .map(|(name, shape)| {
                let p = GroupPart {
                    name,
                    shape,
                    offset,
                };
                offset += p.elems();
                p
            })
            .collect();
        Self {
            id,
            parts,
            elems: offset,
        }
    }
}

/// Where a group's bytes come from. `write` is the fast path: the part's
/// bytes written by the host straight into a Shared slot in the tier's
/// dtype — no allocation, nothing encoded (a mapped checkpoint; the host
/// converts when the checkpoint's dtype is not the tier's). `part` is the
/// general path: a tensor the tier copies in with `slice_set` (a test's
/// tensors; a device without a host mapping).
pub trait GroupSource: Send + Sync {
    fn part(&self, group: WeightGroupId, part: usize) -> Result<Tensor>;
    /// Write part `part` of `group` into `dst` — a slot's host mapping —
    /// in `dtype`, returning whether it did. The default declines.
    fn write(
        &self,
        _group: WeightGroupId,
        _part: usize,
        _dtype: DType,
        _dst: &mut [u8],
    ) -> Result<bool> {
        Ok(false)
    }
}

/// `bf16` bits → `f16`, on every core: each bf16 is widened to f32 by a
/// shift and rounded to f16 by `half` (the hardware convert on this CPU).
/// Byte-wise reads: the mapped checkpoint's tensor data starts wherever
/// the header ends. The 2.6B checkpoint is bf16 and the Metal path
/// computes in f16; the first form of the tier took every such part to
/// the device, cast it there and copied it into the slot — two transient
/// pool allocations a part, 15 307 of them over 64 steps at a 4 GB quota
/// (`memory.md` §9.7).
fn convert_bf16_to_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = dst.len() / 2;
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let per = n.div_ceil(threads).max(1 << 16);
    std::thread::scope(|scope| {
        for (t, out) in dst.chunks_mut(per * 2).enumerate() {
            let inp = &src[t * per * 2..][..out.len()];
            scope.spawn(move || {
                for (o, i) in out.chunks_exact_mut(2).zip(inp.chunks_exact(2)) {
                    let b = u16::from_le_bytes([i[0], i[1]]);
                    let f = half::f16::from_f32(f32::from_bits((b as u32) << 16));
                    o.copy_from_slice(&f.to_le_bytes());
                }
            });
        }
    });
}

/// A safetensors checkpoint as a group source: each group's parts by
/// tensor name, read from the mapped files (the page cache is the second
/// copy §4.8 names; `F_NOCACHE` is §14 Q7's measurement, after this).
pub struct SafetensorsGroups {
    map: lloom_tensor::safetensors::MmapedSafetensors,
    /// Per group, its parts' tensor names.
    names: Vec<Vec<String>>,
    device: Device,
}

impl SafetensorsGroups {
    /// # Safety
    /// The files are memory-mapped (`MmapedSafetensors::multi`).
    pub unsafe fn open<P: AsRef<std::path::Path>>(
        files: &[P],
        names: Vec<Vec<String>>,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            map: lloom_tensor::safetensors::MmapedSafetensors::multi(files)?,
            names,
            device: device.clone(),
        })
    }

    fn name(&self, group: WeightGroupId, part: usize) -> Result<&str> {
        self.names
            .get(group.index())
            .and_then(|p| p.get(part))
            .map(|s| s.as_str())
            .ok_or_else(|| {
                lloom_tensor::Error::Msg(format!(
                    "weight groups: no part {part} of group {}",
                    group.0
                ))
            })
    }
}

impl GroupSource for SafetensorsGroups {
    fn part(&self, group: WeightGroupId, part: usize) -> Result<Tensor> {
        self.map.load(self.name(group, part)?, &self.device)
    }
    fn write(
        &self,
        group: WeightGroupId,
        part: usize,
        dtype: DType,
        dst: &mut [u8],
    ) -> Result<bool> {
        let view = self.map.get(self.name(group, part)?)?;
        let st: DType = view.dtype().try_into()?;
        let data = view.data();
        let want = dst.len() / dtype.size_in_bytes() * st.size_in_bytes();
        if data.len() != want {
            lloom_tensor::bail!(
                "weight groups: group {} part {} is {} bytes in the checkpoint, the slot's part \
                 wants {} ({} elements of {:?})",
                group.0,
                part,
                data.len(),
                want,
                dst.len() / dtype.size_in_bytes(),
                dtype
            )
        }
        match (st, dtype) {
            (a, b) if a == b => dst.copy_from_slice(data),
            (DType::BF16, DType::F16) => convert_bf16_to_f16(data, dst),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// The tier's counters (relaxed atomics; an instrument).
#[derive(Debug, Default)]
pub struct GroupCounters {
    pub hits: std::sync::atomic::AtomicU64,
    pub misses: std::sync::atomic::AtomicU64,
    pub evictions: std::sync::atomic::AtomicU64,
    pub bytes_loaded: std::sync::atomic::AtomicU64,
    pub waits: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_loaded: u64,
    /// Loads that found no slot whose last reader the clock covered and
    /// took one anyway (an encoder copy ordered behind the reader, never a
    /// host wait; counted because a tight ring is a sign the depth is
    /// short).
    pub waits: u64,
}

/// The dense weight-group tier.
#[derive(Debug)]
pub struct WeightGroups {
    groups: Vec<GroupSpec>,
    cache: SlotCache<WeightGroupId, Dense, Belady>,
    /// `[slots, slot_elems]` of the groups' dtype.
    backing: Tensor,
    slot_elems: usize,
    dtype: DType,
    device: Device,
    /// The plan's order of use, cyclic, and the position of the next use.
    plan: Vec<WeightGroupId>,
    pos: u64,
    /// Per group, the slot it was served from at its last use: a recorded
    /// pass plan bound that slot (§4.8 R2), so a change is reported.
    last_slot: Vec<Option<SlotId>>,
    /// The slot of the group in use: `ensure` returns its views before the
    /// stage's kernels are encoded, so until the next `ensure` it is
    /// pinned — a prefetch may never take it (the first form of this tier
    /// did, and the stage computed on the next group's bytes).
    current: Option<SlotId>,
    #[cfg(feature = "metal")]
    clock: Option<std::sync::Arc<lloom_tensor::kernels::metal::GpuClock>>,
    counters: GroupCounters,
    site: &'static str,
    /// The tier's row on the ledger (§4.8 "as a tenant": class `Cache`,
    /// floor the ring — three slots — want every group, unit a slot); the
    /// owner applies a grant with `resize_slots`.
    #[cfg(feature = "metal")]
    gauge: Option<std::sync::Arc<lloom_tensor::kernels::metal::ledger::Gauge>>,
}

/// The tier's storage: `[slots, slot_elems]` of `dtype`. On Metal a
/// persistent Shared buffer from the device's `Mapped` backing — a weight
/// placement, not a pool allocation: the pool's residual cap (§9.5k) counts
/// what no class planned, and the admission plans the slots as weights
/// (`resident + slots x slot_bytes`); the reservation is the backing's,
/// so the ledger holds the bytes the way it holds every weight's. The
/// first run through the pool was refused by that cap at a 4 GB quota
/// ("1.97 GB unplanned", `memory.md` §9.7). Elsewhere, zeros through
/// `nn::zeros` (its sentinel arms the site).
fn slot_backing(
    site: &'static str,
    slots: usize,
    slot_elems: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        use lloom_tensor::{op::BackpropOp, MetalStorage, Storage};
        let bytes = slots * slot_elems * dtype.size_in_bytes();
        let handle = m.new_persistent_shared(bytes)?;
        let storage = MetalStorage::new(handle, m.clone(), slots * slot_elems, dtype);
        return Ok(Tensor::from_storage(
            Storage::Metal(storage),
            (slots, slot_elems),
            BackpropOp::none(),
            false,
        ));
    }
    zs::zeros(site, (slots, slot_elems), dtype, device)
}

impl WeightGroups {
    /// `slots` slots sized for the largest group, zero-filled, as one
    /// allocation (`site` names it for the zeros census). Three at least
    /// serve a host loader: the group in use (pinned), the previous
    /// stage's (its reader still open), and one to load into.
    pub fn new(
        groups: Vec<GroupSpec>,
        slots: usize,
        dtype: DType,
        device: &Device,
        site: &'static str,
    ) -> Result<Self> {
        let slot_elems = groups.iter().map(|g| g.elems).max().unwrap_or(0).max(1);
        let slots = slots.max(1);
        let backing = slot_backing(site, slots, slot_elems, dtype, device)?;
        #[cfg(feature = "metal")]
        let clock = match device {
            Device::Metal(d) => Some(d.gpu_clock()),
            _ => None,
        };
        let slot_bytes = slot_elems * dtype.size_in_bytes();
        #[cfg(feature = "metal")]
        let gauge = match device {
            Device::Metal(m) => {
                use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
                let gauge = Gauge::new(
                    site,
                    Claim {
                        floor: 3 * slot_bytes,
                        want: groups.len().max(1) * slot_bytes,
                        unit: slot_bytes,
                        class: Class::Cache,
                    },
                    slots * slot_bytes,
                );
                let tenant: std::sync::Arc<dyn Tenant> = gauge.clone();
                m.ledger().register(&tenant);
                Some(gauge)
            }
            _ => None,
        };
        Ok(Self {
            cache: SlotCache::with(Dense::new(groups.len()), slots),
            last_slot: vec![None; groups.len()],
            current: None,
            groups,
            backing,
            slot_elems,
            dtype,
            device: device.clone(),
            plan: Vec::new(),
            pos: 0,
            #[cfg(feature = "metal")]
            clock,
            counters: GroupCounters::default(),
            site,
            #[cfg(feature = "metal")]
            gauge,
        })
    }

    /// The slots the tier holds, changed: the backing is allocated again
    /// at `slots` with the resident groups' slots carried over (a shrink
    /// evicts the groups past the new count first, their next use the
    /// furthest), and the ledger's row follows. Three slots at least.
    /// Refused while a group is in use (between a stage's `ensure` and the
    /// next).
    pub fn resize_slots(&mut self, slots: usize) -> Result<usize> {
        let slots = slots.max(3);
        if self.current.is_some() {
            lloom_tensor::bail!("weight groups: resize_slots while a stage's group is in use")
        }
        let was = self.cache.n_slots();
        if slots == was {
            return Ok(was);
        }
        // Evict the groups in slots past the new count (a shrink); their
        // readers must have completed — a cold path, the owner's wait.
        if slots < was {
            let completed = self.completed();
            for s in slots..was {
                let sid = SlotId(s as u32);
                if self.cache.key_in(sid).is_some() {
                    if !completed.covers(self.cache.reader_epoch(sid)) {
                        lloom_tensor::bail!(
                            "weight groups: resize_slots with slot {s}'s reader open"
                        )
                    }
                    self.counters
                        .evictions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let epoch = self.cache.reader_epoch(sid);
                    self.cache.evict(sid, epoch, false);
                }
            }
        }
        let fresh = slot_backing(self.site, slots, self.slot_elems, self.dtype, &self.device)?;
        let keep = was.min(slots);
        fresh
            .narrow(0, 0, keep)?
            .slice_set(&self.backing.narrow(0, 0, keep)?, 0, 0)?;
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &self.device {
            // The copy is encoded; the old backing must outlive it.
            m.wait_until_completed()?;
        }
        self.backing = fresh;
        // The map: the same groups in the same slots below `keep`.
        let mut cache: SlotCache<WeightGroupId, Dense, Belady> =
            SlotCache::with(Dense::new(self.groups.len()), slots);
        for s in 0..keep {
            let sid = SlotId(s as u32);
            if let Some(g) = self.cache.key_in(sid) {
                cache.install(sid, g);
                cache.set_next_use(sid, self.cache.next_use(sid));
                cache.stamp(sid, self.cache.reader_epoch(sid));
            }
        }
        self.cache = cache;
        for l in self.last_slot.iter_mut() {
            if l.is_some_and(|s| s.index() >= slots) {
                *l = None;
            }
        }
        #[cfg(feature = "metal")]
        if let Some(g) = &self.gauge {
            g.set_held(slots * self.slot_bytes());
        }
        Ok(slots)
    }

    pub fn n_slots(&self) -> usize {
        self.cache.n_slots()
    }

    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Bytes of one slot.
    pub fn slot_bytes(&self) -> usize {
        self.slot_elems * self.dtype.size_in_bytes()
    }

    /// Bytes of the backing (every slot).
    pub fn bytes(&self) -> usize {
        self.backing.elem_count() * self.dtype.size_in_bytes()
    }

    /// The plan: the groups in their order of use, which repeats
    /// (a per-step stage order). Resets the position.
    pub fn set_plan(&mut self, order: Vec<WeightGroupId>) {
        self.plan = order;
        self.pos = 0;
        self.restamp_next_uses();
    }

    pub fn plan(&self) -> &[WeightGroupId] {
        &self.plan
    }

    /// The plan position of `g`'s next use after `pos` (the cycle wraps;
    /// `u64::MAX` for a group not in the plan).
    fn next_use(&self, g: WeightGroupId, pos: u64) -> u64 {
        let n = self.plan.len() as u64;
        if n == 0 {
            return u64::MAX;
        }
        let start = (pos % n) as usize;
        for k in 1..=n as usize {
            let i = (start + k) % n as usize;
            if self.plan[i] == g {
                return pos + k as u64;
            }
        }
        u64::MAX
    }

    fn restamp_next_uses(&mut self) {
        for s in 0..self.cache.n_slots() {
            if let Some(g) = self.cache.key_in(SlotId(s as u32)) {
                let at = self.next_use(g, self.pos);
                self.cache.set_next_use(SlotId(s as u32), at);
            }
        }
    }

    fn completed(&self) -> Completed {
        #[cfg(feature = "metal")]
        if let Some(c) = &self.clock {
            return c.completed_count();
        }
        Completed::ALL
    }

    fn epoch(&self) -> Epoch {
        #[cfg(feature = "metal")]
        if let Some(c) = &self.clock {
            return c.current_epoch();
        }
        Epoch::ZERO
    }

    /// The slot `g` is in, if resident.
    pub fn slot_of(&self, g: WeightGroupId) -> Option<SlotId> {
        self.cache.slot_of(g)
    }

    /// Make `g` resident for the stage at the plan's current position —
    /// loaded through `source` into a slot the clock covers if one is
    /// free, else into the Belady victim with the copy ordered behind its
    /// last reader — stamp the slot with the open epoch (its reader), and
    /// advance the plan past this use. Returns the slot and whether it
    /// differs from the slot `g` was served from at its last use (a
    /// recorded plan bound that one).
    pub fn ensure(&mut self, g: WeightGroupId, source: &dyn GroupSource) -> Result<(SlotId, bool)> {
        use std::sync::atomic::Ordering::Relaxed;
        // The previous stage's slot may be a victim again: its kernels are
        // encoded by now — in the epoch open at *this* instant at the
        // latest, which may be later than the one its `ensure` stamped (a
        // command-buffer cut between the two: H1's shape, §4.7). Stamp it
        // again here, so the clock is asked about the right epoch.
        if let Some(prev) = self.current.take() {
            let now = self.epoch();
            if now > self.cache.reader_epoch(prev) {
                self.cache.stamp(prev, now);
            }
            self.cache.set_pinned(prev, false);
        }
        let slot = match self.cache.slot_of(g) {
            Some(s) => {
                self.counters.hits.fetch_add(1, Relaxed);
                s
            }
            None => {
                self.counters.misses.fetch_add(1, Relaxed);
                let (s, evicted) = match self.cache.victim(self.completed()) {
                    Some(v) => v,
                    None => {
                        // Every slot's last reader is still open: take the
                        // Belady victim anyway — the copy is ordered behind
                        // the reader on the queue.
                        self.counters.waits.fetch_add(1, Relaxed);
                        self.cache.victim(Completed::ALL).ok_or_else(|| {
                            lloom_tensor::Error::Msg(format!(
                                "weight groups: no slot of {} for group {}",
                                self.cache.n_slots(),
                                g.0
                            ))
                        })?
                    }
                };
                if evicted {
                    self.counters.evictions.fetch_add(1, Relaxed);
                    let epoch = self.cache.reader_epoch(s);
                    self.cache.evict(s, epoch, false);
                }
                self.load(s, g, source)?;
                self.cache.install(s, g);
                s
            }
        };
        self.cache.stamp(slot, self.epoch());
        self.cache.set_pinned(slot, true);
        self.current = Some(slot);
        let moved = self.last_slot[g.index()].is_some_and(|s| s != slot);
        self.last_slot[g.index()] = Some(slot);
        // This use is consumed: the next one is further along the cycle.
        if !self.plan.is_empty() {
            let pos = self.pos;
            // Advance to this group's position if the plan is being
            // followed; a group used out of plan order leaves `pos` alone.
            if self.plan[(pos % self.plan.len() as u64) as usize] == g {
                self.pos = pos + 1;
            }
            let at = self.next_use(g, self.pos.saturating_sub(1).max(pos));
            self.cache.set_next_use(slot, at);
            if self.plan[(pos % self.plan.len() as u64) as usize] == g {
                self.restamp_next_uses();
            }
        }
        Ok((slot, moved))
    }

    /// Load the next `depth` groups of the plan not yet resident (the
    /// loader runs ahead of the GPU). Returns how many were loaded.
    pub fn prefetch(&mut self, depth: usize, source: &dyn GroupSource) -> Result<usize> {
        if self.plan.is_empty() {
            return Ok(0);
        }
        let n = self.plan.len() as u64;
        let mut loaded = 0;
        for k in 0..depth as u64 {
            let g = self.plan[((self.pos + k) % n) as usize];
            if self.cache.slot_of(g).is_some() {
                continue;
            }
            let Some((s, evicted)) = self.cache.victim(self.completed()) else {
                break;
            };
            if evicted {
                // A victim needed sooner than this prefetch would be a
                // loss: stop at the ring's edge.
                if self.cache.next_use(s) <= self.pos + k {
                    break;
                }
                self.counters
                    .evictions
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let epoch = self.cache.reader_epoch(s);
                self.cache.evict(s, epoch, false);
            }
            self.load(s, g, source)?;
            self.cache.install(s, g);
            let at = self.next_use(g, self.pos.saturating_sub(1));
            self.cache.set_next_use(s, at);
            loaded += 1;
        }
        Ok(loaded)
    }

    /// The slot's host mapping on Metal (a Shared backing), else `None`.
    #[cfg(feature = "metal")]
    fn slot_host_ptr(&self, slot: SlotId) -> Option<*mut u8> {
        let (storage, _) = self.backing.storage_and_layout();
        let lloom_tensor::Storage::Metal(m) = &*storage else {
            return None;
        };
        let base = m.buffer().contents();
        if base.is_null() {
            return None;
        }
        // The backing is contiguous `[slots, slot_elems]` from offset 0.
        Some(unsafe { base.add(slot.index() * self.slot_bytes()) })
    }

    #[cfg(not(feature = "metal"))]
    fn slot_host_ptr(&self, _slot: SlotId) -> Option<*mut u8> {
        None
    }

    fn load(&mut self, slot: SlotId, g: WeightGroupId, source: &dyn GroupSource) -> Result<()> {
        let spec = &self.groups[g.index()];
        let elem = self.dtype.size_in_bytes();
        // The host path: the slot's last reader must have completed (the
        // Belady victim's did, stages ago; a tight ring waits once, counted).
        let host = match self.slot_host_ptr(slot) {
            Some(p) if self.completed().covers(self.cache.reader_epoch(slot)) => Some(p),
            Some(p) => {
                #[cfg(feature = "metal")]
                if let Device::Metal(m) = &self.device {
                    self.counters
                        .waits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    m.wait_until_completed()?;
                }
                Some(p)
            }
            None => None,
        };
        let row = self.backing.narrow(0, slot.index(), 1)?.squeeze(0)?;
        let mut bytes = 0usize;
        for (i, part) in spec.parts.iter().enumerate() {
            if let Some(p) = host {
                let len = part.elems() * elem;
                // SAFETY: `p + offset .. + len` is inside slot `slot`'s
                // range of the Shared backing, alive for the tier's life;
                // its last GPU reader has completed (above) and no host
                // reader exists (the slot is being loaded).
                let dst = unsafe { std::slice::from_raw_parts_mut(p.add(part.offset * elem), len) };
                if source.write(g, i, self.dtype, dst)? {
                    bytes += len;
                    // A diagnosis knob: `LLOOM_WG_SYNC=1` waits the queue out
                    // after every host write (never on by default).
                    #[cfg(feature = "metal")]
                    if std::env::var("LLOOM_WG_SYNC").is_ok() {
                        if let Device::Metal(m) = &self.device {
                            m.wait_until_completed()?;
                        }
                    }
                    continue;
                }
            }
            let src = source.part(g, i)?;
            if src.elem_count() != part.elems() {
                lloom_tensor::bail!(
                    "weight groups: group {} part {} ({}) has {} elements, the source {}",
                    g.0,
                    i,
                    part.name,
                    part.elems(),
                    src.elem_count()
                )
            }
            let src = src.to_dtype(self.dtype)?.flatten_all()?.contiguous()?;
            row.slice_set(&src, 0, part.offset)?;
            bytes += part.elems() * self.dtype.size_in_bytes();
        }
        self.counters
            .bytes_loaded
            .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Part `part` of `g`, as a view over its slot (the slot's buffer at
    /// the part's offset, the part's shape). `g` must be resident.
    pub fn tensor(&self, g: WeightGroupId, part: usize) -> Result<Tensor> {
        let Some(slot) = self.cache.slot_of(g) else {
            lloom_tensor::bail!("weight groups: group {} is not resident", g.0)
        };
        let p = &self.groups[g.index()].parts[part];
        self.backing
            .narrow(0, slot.index(), 1)?
            .squeeze(0)?
            .narrow(0, p.offset, p.elems())?
            .reshape(p.shape.clone())
    }

    /// A diagnosis: every resident group whose slot's bytes (read on the
    /// host, after the queue is waited out) differ from `source`'s. Empty
    /// when the slots hold what they should.
    pub fn audit(&self, source: &dyn GroupSource) -> Result<Vec<(WeightGroupId, usize)>> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = &self.device {
            m.wait_until_completed()?;
        }
        let mut wrong = Vec::new();
        for g in self.cache.resident() {
            for (i, _) in self.groups[g.index()].parts.iter().enumerate() {
                let have = self
                    .tensor(g, i)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let want = source
                    .part(g, i)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                if have != want {
                    wrong.push((g, i));
                }
            }
        }
        Ok(wrong)
    }

    pub fn stats(&self) -> GroupStats {
        use std::sync::atomic::Ordering::Relaxed;
        GroupStats {
            hits: self.counters.hits.load(Relaxed),
            misses: self.counters.misses.load(Relaxed),
            evictions: self.counters.evictions.load(Relaxed),
            bytes_loaded: self.counters.bytes_loaded.load(Relaxed),
            waits: self.counters.waits.load(Relaxed),
        }
    }

    pub fn resident(&self) -> Vec<WeightGroupId> {
        self.cache.resident()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Groups of two parts, [k, 4] and [4, k] with k = 2 + id, filled with
    /// values that name the group and the part.
    struct Synthetic {
        device: Device,
    }

    impl Synthetic {
        fn spec(n: usize) -> Vec<GroupSpec> {
            (0..n as u32)
                .map(|g| {
                    let k = 2 + g as usize;
                    GroupSpec::new(
                        WeightGroupId(g),
                        vec![
                            (format!("g{g}.a"), Shape::from((k, 4))),
                            (format!("g{g}.b"), Shape::from((4, k))),
                        ],
                    )
                })
                .collect()
        }
        fn expected(&self, g: WeightGroupId, part: usize) -> Tensor {
            let k = 2 + g.0 as usize;
            let (r, c) = if part == 0 { (k, 4) } else { (4, k) };
            let v: Vec<f32> = (0..r * c)
                .map(|i| (g.0 as f32) * 100.0 + part as f32 * 10.0 + i as f32)
                .collect();
            Tensor::from_vec(v, (r, c), &self.device).unwrap()
        }
    }

    impl GroupSource for Synthetic {
        fn part(&self, g: WeightGroupId, part: usize) -> Result<Tensor> {
            Ok(self.expected(g, part))
        }
    }

    /// Design §4.8's claim, read at the tier: on a cyclic order of `n`
    /// groups through `c` slots with the loader one group ahead, Belady
    /// loads between `n - c` and `n - c + 2` groups a cycle once warm and
    /// every `ensure` is a hit. `n - c` is the optimum (the policy's own
    /// test, `slot_cache::tests::belady_hits_on_a_cyclic_order_where_lru_hits_nothing`:
    /// the resident set turns with the cycle, the group used last being
    /// the one needed last); the two above it are the slot pinned under
    /// the stage in use and the one the next group loads into — the tier
    /// settles at either, by where the warm-up left the ring (18 of 30
    /// slots: 12 a cycle; 6 of 30: 24 rising to 26). LRU on the same
    /// order loads every group every cycle. This is the tier's figure,
    /// the one the quota arms are read against (`memory.md` §9.7).
    #[test]
    fn a_warm_cycle_loads_only_the_groups_the_slots_cannot_hold() {
        let dev = Device::Cpu;
        let src = Synthetic {
            device: dev.clone(),
        };
        let n = 30usize;
        for c in [18usize, 6] {
            let mut tier =
                WeightGroups::new(Synthetic::spec(n), c, DType::F32, &dev, "test.groups").unwrap();
            tier.set_plan((0..n as u32).map(WeightGroupId).collect());
            let mut per_cycle = Vec::new();
            for _cycle in 0..5 {
                let misses0 = tier.stats().misses;
                let mut loads = 0usize;
                for g in 0..n as u32 {
                    tier.ensure(WeightGroupId(g), &src).unwrap();
                    loads += tier.prefetch(1, &src).unwrap();
                }
                let misses = (tier.stats().misses - misses0) as usize;
                per_cycle.push((loads + misses, misses));
            }
            // The first cycle is cold: `n` loads. Warm: `n - c ..= n - c + 2`
            // loads and no miss.
            assert_eq!(per_cycle[0].0, n, "c={c}: the cold cycle loads every group");
            for (cycle, &(loads, misses)) in per_cycle.iter().enumerate().skip(1) {
                assert!(
                    (n - c..=n - c + 2).contains(&loads) && misses == 0,
                    "c={c} cycle {cycle}: {per_cycle:?}"
                );
            }
        }
    }

    #[test]
    fn groups_stream_through_the_slots_and_bind_the_right_bytes() {
        let dev = Device::Cpu;
        let src = Synthetic {
            device: dev.clone(),
        };
        let (n, slots) = (6usize, 3usize);
        let mut tier =
            WeightGroups::new(Synthetic::spec(n), slots, DType::F32, &dev, "test.groups").unwrap();
        assert_eq!(
            tier.slot_bytes(),
            (2 * 7 * 4) * 4,
            "sized for the largest group"
        );
        tier.set_plan((0..n as u32).map(WeightGroupId).collect());
        for round in 0..3 {
            for g in 0..n as u32 {
                let g = WeightGroupId(g);
                let (slot, moved) = tier.ensure(g, &src).unwrap();
                assert_eq!(tier.slot_of(g), Some(slot));
                // A resident group is served from the slot it had: `moved`
                // only ever reports a load into another slot (Belady's
                // assignment is not promised periodic; the owner drops its
                // recorded plans on a move).
                if !moved {
                    assert_eq!(Some(slot), tier.slot_of(g));
                }
                for part in 0..2 {
                    let got = tier.tensor(g, part).unwrap();
                    let want = src.expected(g, part);
                    assert_eq!(got.dims(), want.dims());
                    assert_eq!(
                        got.to_vec2::<f32>().unwrap(),
                        want.to_vec2::<f32>().unwrap(),
                        "round {round} group {} part {part}",
                        g.0
                    );
                }
            }
        }
        let st = tier.stats();
        assert_eq!(st.hits + st.misses, 3 * n as u64);
        // Belady on the cycle: at least C − 1 of every N hit after the first
        // round; never fewer than one slot streams.
        assert!(st.hits >= 2 * (slots as u64 - 1), "{st:?}");
        assert!(st.misses >= n as u64 + 2, "{st:?}");
        assert_eq!(tier.resident().len(), slots);
    }

    #[test]
    fn a_resize_keeps_the_resident_groups_and_their_bytes() {
        let dev = Device::Cpu;
        let src = Synthetic {
            device: dev.clone(),
        };
        let n = 6usize;
        let mut tier =
            WeightGroups::new(Synthetic::spec(n), 5, DType::F32, &dev, "test.groups").unwrap();
        tier.set_plan((0..n as u32).map(WeightGroupId).collect());
        for g in 0..5u32 {
            tier.ensure(WeightGroupId(g), &src).unwrap();
        }
        // The group in use blocks a resize; after the stage it does not.
        assert!(tier.resize_slots(3).is_err());
        tier.ensure(WeightGroupId(5), &src).unwrap();
        tier.current = None;
        // Shrink to three: the groups in slots 3 and 4 go, 0..3 stay with
        // their bytes.
        assert_eq!(tier.resize_slots(3).unwrap(), 3);
        assert_eq!(tier.n_slots(), 3);
        let resident = tier.resident();
        assert_eq!(resident.len(), 3, "{resident:?}");
        for g in resident {
            for part in 0..2 {
                assert_eq!(
                    tier.tensor(g, part).unwrap().to_vec2::<f32>().unwrap(),
                    src.expected(g, part).to_vec2::<f32>().unwrap()
                );
            }
        }
        // Grow to four: the three stay, a fourth slot is free.
        assert_eq!(tier.resize_slots(4).unwrap(), 4);
        assert_eq!(tier.resident().len(), 3);
        let (slot, _) = tier.ensure(WeightGroupId(3), &src).unwrap();
        assert_eq!(slot, SlotId(3), "the new slot is empty and taken first");
    }

    #[test]
    fn the_prefetch_runs_ahead_of_the_plan_and_stops_at_the_rings_edge() {
        let dev = Device::Cpu;
        let src = Synthetic {
            device: dev.clone(),
        };
        let (n, slots) = (5usize, 3usize);
        let mut tier =
            WeightGroups::new(Synthetic::spec(n), slots, DType::F32, &dev, "test.groups").unwrap();
        tier.set_plan((0..n as u32).map(WeightGroupId).collect());
        // An empty cache: a depth-2 prefetch loads the first two groups.
        assert_eq!(tier.prefetch(2, &src).unwrap(), 2);
        assert_eq!(tier.resident(), vec![WeightGroupId(0), WeightGroupId(1)]);
        // Using them is a hit; the third group is a miss loaded on use.
        assert_eq!(
            tier.ensure(WeightGroupId(0), &src).unwrap(),
            (SlotId(0), false)
        );
        assert_eq!(
            tier.ensure(WeightGroupId(1), &src).unwrap(),
            (SlotId(1), false)
        );
        let st = tier.stats();
        assert_eq!((st.hits, st.misses), (2, 0));
        // Group 2 misses (slot 2) and is in use: pinned. The plan is at
        // position 3. A depth-3 prefetch loads 3 (evicting 1, needed
        // furthest at 7 among the unpinned — 2 is in use) and 4 (evicting
        // 0, needed at 6 — after 4's use at 4); then the edge: two loads.
        tier.ensure(WeightGroupId(2), &src).unwrap();
        let loaded = tier.prefetch(3, &src).unwrap();
        assert_eq!(loaded, 2, "the ring's edge");
        let mut resident = tier.resident();
        resident.sort();
        assert_eq!(
            resident,
            vec![WeightGroupId(2), WeightGroupId(3), WeightGroupId(4)]
        );
        assert_eq!(tier.stats().misses, 1);
    }
}
