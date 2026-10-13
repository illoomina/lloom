//! Recurrent-state channel: fixed-size, per-sequence state that migrates with
//! the sequence and rolls back on speculative reject.
//!
//! Several blocks carry state that is *not* KV: LFM2's short-conv window, a
//! Mamba SSM's hidden state, Gated DeltaNet's `S` matrix. They share a lifecycle
//! KV does not:
//!
//! - **fixed size** — the state does not grow with context (DeltaNet's `S` is
//!   `[H_v, d, d]` at position 1 and at position 262 144);
//! - **migrates WITH the sequence** — on eviction / prefix-reuse / slot move it
//!   travels with the tokens, and a cache that restores KV without it "answers
//!   confidently with a corrupt trajectory" (the llama.cpp linear-attention
//!   landmine, `design/docs/porting-models.md` §21.4a);
//! - **rolls back on a rejected speculative window** — MTP / DSpark verify runs
//!   `K+1` positions and keeps the accepted prefix, so the state must snapshot at
//!   the window start and rewind to `accepted.len()` (§10.2a, the conv ring's
//!   `advance`/`resolve` discipline).
//!
//! LFM2 hand-rolled this for its conv window (`conv_states` + `conv_phase` +
//! `staged_snaps`); Mamba and recurrent-gemma each hand-rolled their own. This is
//! the discipline stated once, as the seam the next recurrent block plugs into —
//! §3.6a applied to sequence state. DeltaNet's `S` is its first consumer here; a
//! later PR can move the conv window onto the same seam (its migration/rollback
//! logic is the reference this generalizes, so the move is mechanical).
//!
//! It is deliberately shape-agnostic: a channel holds one `Option<Tensor>` per
//! layer of whatever shape the block chose, plus a single snapshot per layer for
//! the open speculative window. It does not interpret the state — only owns its
//! lifecycle.

use lloom_tensor::{Result, Tensor};

/// One block's per-layer recurrent state, plus the single-window rollback
/// snapshot the speculative path needs.
///
/// `state[l]` is `None` until the first pass on layer `l` writes it (a fresh
/// sequence starts from the block's implicit zero state). The snapshot is taken
/// by [`Self::open_window`] and consumed by [`Self::resolve_window`] /
/// [`Self::rewind_to`]; only one window is open at a time (the verify loop's
/// invariant, mirrored from the conv ring's non-`Copy` `Window` token).
/// Hot-path discipline (the reference LFM2's conv window will migrate onto):
/// - the per-layer state buffer is allocated ONCE (pool-backed) and written IN
///   PLACE via `slice_set` under an [`EscapeGuard`] — zero allocation per token
///   in steady state (§3.3a), never an `Option<Tensor>` replaced per step;
/// - the speculative window copies nothing: its base is the slab's own
///   buffer (a shared handle); a write inside the window lands in the
///   channel's spare buffer and is swapped in ([`Self::take_spare`] /
///   [`Self::swap_in`] for a kernel writing out of place, copy-on-write in
///   [`Self::write`] for a composed step), so a rewind is the base handle
///   back and a resolve retires the base as the next spare — no allocation
///   in the steady state, no snapshot copy per token;
/// - no `std::sync`, no lock — the channel is `&mut self` owned by the sequence's
///   cache, so interior mutability is not needed at all.
///
/// A block computes the next state as an arena transient and hands it to
/// [`Self::write`], which copies it into the persistent slab in place. (The copy
/// is the recurrence's, not an allocation; a future fused GDN kernel would write
/// the slab directly, as LFM2's `shortconv_ring_fused` does for its ring.)
#[derive(Debug, Clone)]
pub struct RecurrentState {
    /// Per-layer persistent state buffer, allocated once on first [`Self::write`]
    /// and reused in place thereafter. `None` before the first write (the block
    /// substitutes its implicit zero state).
    slab: Vec<Option<Tensor>>,
    /// The pre-window slab per layer while a window is open: a shared handle
    /// on the buffer that WAS the slab at `open_window` — no copy. Every write
    /// inside the window lands in another buffer ([`Self::write`] copies on
    /// write when the slab is the base; a fused kernel writes the spare from
    /// [`Self::take_spare`] and [`Self::swap_in`]s it), so the base stays
    /// what it was and a rewind is a handle swap.
    base: Vec<Option<Tensor>>,
    /// A free buffer per layer, the slab's shape: the buffer the last swap
    /// retired, or the base a resolved window released. Handed out by
    /// [`Self::take_spare`], so steady-state decode allocates nothing.
    spare: Vec<Option<Tensor>>,
    /// Whether a speculative window is currently open (guards double-open and
    /// resolve/rewind-without-open). A plain bool — the channel is single-owner.
    window_open: bool,
    /// A window HELD behind the open one ([`Self::hold_window`]): its base per
    /// layer, still restorable while the newer window runs — a decode step
    /// whose settle trails the next step's encode (a provisional step on a
    /// lookahead ring) is undone two steps deep. A write never lands in it,
    /// and no buffer it names is handed out as the spare. Empty when no
    /// window is held.
    held: Vec<Option<Tensor>>,
    held_open: bool,
    /// The pool lease when the state lives in a [`StatePool`].
    pooled: Option<Pooled>,
}

/// An independently stored current state, outside the finite phase rings.
/// Private fields prevent callers from substituting aliasing tensor handles.
#[derive(Debug)]
pub struct RecurrentRetained {
    slab: Vec<Option<Tensor>>,
    phases: Option<Vec<usize>>,
}

impl crate::nn::branch::RetainState for RecurrentState {
    type Retained = RecurrentRetained;

    fn retain_state(&mut self) -> Result<Self::Retained> {
        if self.window_open || self.held_open {
            lloom_tensor::bail!("recurrent retention requires settled windows")
        }
        // Validate every backing before allocating the first copy.
        for tensor in self.slab.iter().flatten() {
            let bytes = tensor
                .dims()
                .iter()
                .try_fold(tensor.dtype().size_in_bytes(), |n, d| n.checked_mul(*d))
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| {
                    lloom_tensor::Error::Msg("retained recurrent bytes overflow".into())
                })?;
            let _ = bytes;
        }
        let mut slab = crate::nn::branch::retained_vec(self.slab.len())?;
        let phases = if let Some(p) = &self.pooled {
            let mut phases = crate::nn::branch::retained_vec(p.phase.len())?;
            phases.extend_from_slice(&p.phase);
            Some(phases)
        } else {
            None
        };
        let _persistent = lloom_tensor::EscapeGuard::new();
        for t in &self.slab {
            slab.push(t.as_ref().map(own_copy).transpose()?);
        }
        Ok(RecurrentRetained { slab, phases })
    }

    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        if self.window_open || self.held_open {
            lloom_tensor::bail!("recurrent retention requires settled windows")
        }
        if saved.slab.len() != self.slab.len() || saved.phases.is_some() != self.pooled.is_some() {
            lloom_tensor::bail!("retained recurrent state has incompatible geometry")
        }
        if let (Some(phases), Some(p)) = (&saved.phases, &self.pooled) {
            if phases.len() != self.slab.len() || phases.iter().any(|v| *v >= p.pool.phases()) {
                lloom_tensor::bail!("retained recurrent state has invalid phases")
            }
        }
        for (old, saved) in self.slab.iter().zip(&saved.slab) {
            if let (Some(old), Some(saved)) = (old, saved) {
                if old.dims() != saved.dims()
                    || old.dtype() != saved.dtype()
                    || !old.device().same_device(saved.device())
                {
                    lloom_tensor::bail!("retained recurrent state changed shape, dtype or device")
                }
            }
        }
        for (l, saved_tensor) in saved.slab.iter().enumerate() {
            if let Some(p) = &mut self.pooled {
                let phase = saved.phases.as_ref().expect("validated pooled metadata")[l];
                if let Some(t) = saved_tensor {
                    let dst = Self::pool_view(p, l, phase)?;
                    if dst.dims() != t.dims()
                        || dst.dtype() != t.dtype()
                        || !dst.device().same_device(t.device())
                    {
                        lloom_tensor::bail!("retained recurrent state differs from pool geometry")
                    }
                    let _persistent = lloom_tensor::EscapeGuard::new();
                    dst.slice_set(t, 0, 0)?;
                    self.slab[l] = Some(dst);
                } else {
                    self.slab[l] = None;
                }
                p.phase[l] = phase;
                p.written[l] = saved_tensor.is_some();
            } else if let Some(t) = saved_tensor {
                self.write(l, t.clone())?;
            } else {
                self.slab[l] = None;
            }
        }
        Ok(())
    }
}

/// Whether every state write replaces the slab with a fresh copy instead of
/// writing it in place (`QWEN4EXP_STATE_WRITE=copy`, read once). A debugging
/// instrument behind a gate (§3.3a): an allocation per layer per token,
/// never the shipping path.
fn state_write_by_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.state_write == "copy")
}

/// A copy of `t` in storage of its own. `Tensor::copy` is NOT that on Metal —
/// `MetalStorage::try_clone` hands back the same buffer — so a snapshot taken
/// with it aliases what it snapshots; the CPU backend copies. This allocates
/// and copies on every backend.
pub(crate) fn own_copy(t: &Tensor) -> Result<Tensor> {
    let out = Tensor::zeros(t.shape(), t.dtype(), t.device())?;
    out.slice_set(&t.contiguous()?, 0, 0)?;
    Ok(out)
}

/// The slot-indexed pool a batch's rows keep their recurrent state in
/// (`docs/batched-decode-design.md` §12.2 applied to the recurrent
/// channels), and the ring a speculative window rolls back through: per
/// layer ONE tensor `[phases, slots, ..shape]` — a row's state at layer `l`
/// is row `slot` of the slab at the row's PHASE for that layer, a step
/// writes the next phase and advances (the out-of-place write the fused
/// kernels take, with no buffer per row), a window's base is the phase at
/// its open and an accept of `a` positions is `phase = base + a` — no copy,
/// no replay, on every channel written per position. A batched kernel takes
/// the layer's tensor and per-row `slots` / `phases` tables and runs the
/// rows as one launch; a multi-step kernel writes phases `base+1..base+T`
/// as it goes. `phases` ≥ 2 (a decode needs the current and the next);
/// a verify of `T` positions needs `T + 1`. Allocated per layer on first use.
#[derive(Debug, Clone)]
pub struct StatePool {
    inner: std::sync::Arc<parking_lot::Mutex<PoolInner>>,
}

#[derive(Debug)]
struct PoolInner {
    slots: usize,
    phases: usize,
    shape: Vec<usize>,
    dtype: lloom_tensor::DType,
    device: lloom_tensor::Device,
    /// Per layer, `[phases, slots, ..shape]`.
    slabs: Vec<Option<Tensor>>,
    free: Vec<usize>,
}

impl StatePool {
    /// A pool of `slots` rows over `n_layers` layers of `shape` per row, with
    /// `phases` slabs per layer (≥ 2).
    pub fn new(
        slots: usize,
        n_layers: usize,
        shape: &[usize],
        dtype: lloom_tensor::DType,
        device: &lloom_tensor::Device,
    ) -> Self {
        Self::with_phases(slots, 2, n_layers, shape, dtype, device)
    }

    pub fn with_phases(
        slots: usize,
        phases: usize,
        n_layers: usize,
        shape: &[usize],
        dtype: lloom_tensor::DType,
        device: &lloom_tensor::Device,
    ) -> Self {
        Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(PoolInner {
                slots,
                phases: phases.max(2),
                shape: shape.to_vec(),
                dtype,
                device: device.clone(),
                slabs: vec![None; n_layers],
                free: (0..slots).rev().collect(),
            })),
        }
    }

    /// Whether two handles are the same pool.
    pub fn same(&self, other: &StatePool) -> bool {
        std::sync::Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn slots(&self) -> usize {
        self.inner.lock().slots
    }

    pub fn phases(&self) -> usize {
        self.inner.lock().phases
    }

    pub fn n_layers(&self) -> usize {
        self.inner.lock().slabs.len()
    }

    /// The elements of one row's state.
    pub fn row_elems(&self) -> usize {
        self.inner.lock().shape.iter().product()
    }

    fn lease(&self) -> Result<usize> {
        self.inner
            .lock()
            .free
            .pop()
            .ok_or_else(|| lloom_tensor::Error::msg("StatePool: no free slot"))
    }

    fn release(&self, slot: usize) {
        self.inner.lock().free.push(slot);
    }

    /// Layer `l` as the rows and multi-position kernels read it.
    pub fn pool_layer(&self, l: usize) -> Result<PoolLayer> {
        Ok(PoolLayer {
            tensor: self.layer(l)?,
            n_slots: self.slots(),
            n_phases: self.phases(),
        })
    }

    /// Bytes the pool's allocated layers hold on the device.
    pub fn bytes(&self) -> usize {
        self.inner
            .lock()
            .slabs
            .iter()
            .flatten()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    /// Layer `l`'s tensor `[phases, slots, ..shape]`, allocated (zero) on
    /// first use.
    pub fn layer(&self, l: usize) -> Result<Tensor> {
        let mut g = self.inner.lock();
        if l >= g.slabs.len() {
            lloom_tensor::bail!("StatePool: layer {l} out of range");
        }
        if g.slabs[l].is_none() {
            let mut dims = vec![g.phases, g.slots];
            dims.extend_from_slice(&g.shape);
            let _declared = lloom_tensor::EscapeGuard::new();
            g.slabs[l] = Some(Tensor::zeros(dims.as_slice(), g.dtype, &g.device)?);
        }
        Ok(g.slabs[l].as_ref().expect("allocated above").clone())
    }
}

/// One layer of a [`StatePool`]: `tensor [phases, slots, ..row]`, row `slot`
/// at `phase` at `(phase · n_slots + slot) · row_elems`, the next phase
/// `(phase + 1) % n_phases`.
#[derive(Debug, Clone)]
pub struct PoolLayer {
    pub tensor: Tensor,
    pub n_slots: usize,
    pub n_phases: usize,
}

/// A channel's lease on a [`StatePool`] slot.
#[derive(Debug)]
struct Pooled {
    pool: StatePool,
    slot: usize,
    /// Per layer, which phase holds the row's current state.
    phase: Vec<usize>,
    /// Per layer, whether the row has written the layer since its reset.
    written: Vec<bool>,
    /// The phases and written flags at the open window (an accept of `a`
    /// positions sets `phase = base + a`; the phases past the base were
    /// written only inside the window).
    base_phase: Vec<usize>,
    base_written: Vec<bool>,
}

impl Drop for Pooled {
    fn drop(&mut self) {
        self.pool.release(self.slot);
    }
}

/// A clone leases a slot of its own and copies the written layers into it
/// (a `Cache` clone is a sequence of its own). Panics when the pool has no
/// free slot — a clone cannot fail, and a pool sized for its batch has one.
impl Clone for Pooled {
    fn clone(&self) -> Self {
        let slot = self
            .pool
            .lease()
            .expect("StatePool: no free slot for a cloned channel");
        for (l, &w) in self.written.iter().enumerate() {
            if !w {
                continue;
            }
            let t = self.pool.layer(l).expect("the layer's tensor");
            let src = t
                .narrow(0, self.phase[l], 1)
                .and_then(|x| x.narrow(1, self.slot, 1))
                .expect("the source row");
            // Through a buffer of its own: a slice into the tensor it came from.
            let copy = own_copy(&src).expect("the source row's copy");
            let _declared = lloom_tensor::EscapeGuard::new();
            t.narrow(0, self.phase[l], 1)
                .expect("the phase")
                .slice_set(&copy, 1, slot)
                .expect("the clone's row");
        }
        Self {
            pool: self.pool.clone(),
            slot,
            phase: self.phase.clone(),
            written: self.written.clone(),
            base_phase: self.base_phase.clone(),
            base_written: self.base_written.clone(),
        }
    }
}

impl RecurrentState {
    /// A channel whose state lives in `pool` (a slot leased here, released
    /// when the channel is dropped): the same API, the slabs the pool's.
    pub fn in_pool(pool: &StatePool) -> Result<Self> {
        let n_layers = pool.n_layers();
        let slot = pool.lease()?;
        Ok(Self {
            slab: vec![None; n_layers],
            base: vec![None; n_layers],
            spare: vec![None; n_layers],
            window_open: false,
            held: vec![None; n_layers],
            held_open: false,
            pooled: Some(Pooled {
                pool: pool.clone(),
                slot,
                phase: vec![0; n_layers],
                written: vec![false; n_layers],
                base_phase: vec![0; n_layers],
                base_written: vec![false; n_layers],
            }),
        })
    }

    /// The pool this channel's state lives in, its slot, its phase at layer
    /// `l` and whether the layer is written — for a batched or multi-step
    /// kernel over the pool's tensor.
    pub fn pool_slot(&self, l: usize) -> Option<(&StatePool, usize, usize, bool)> {
        self.pooled
            .as_ref()
            .map(|p| (&p.pool, p.slot, p.phase[l], p.written[l]))
    }

    /// Row `slot` of layer `l`'s slab at `phase`, a view of the row's shape.
    fn pool_view(p: &Pooled, l: usize, phase: usize) -> Result<Tensor> {
        let t = p.pool.layer(l)?;
        t.narrow(0, phase, 1)?
            .narrow(1, p.slot, 1)?
            .squeeze(0)?
            .squeeze(0)
    }

    /// Bytes the channel holds on the device: its pool's when pooled, else
    /// every distinct buffer it keeps — the slabs, the spares and the
    /// windows' bases (a base is the slab's or a spare's buffer at most
    /// once, so nothing is counted twice).
    pub fn bytes_held(&self) -> usize {
        if let Some(p) = &self.pooled {
            return p.pool.bytes();
        }
        let mut seen = Vec::new();
        let mut bytes = 0;
        for t in self
            .slab
            .iter()
            .chain(&self.spare)
            .chain(&self.base)
            .chain(&self.held)
            .flatten()
        {
            if !seen.contains(&t.id()) {
                seen.push(t.id());
                bytes += t.elem_count() * t.dtype().size_in_bytes();
            }
        }
        bytes
    }

    /// Whether the channel is pooled.
    pub fn is_pooled(&self) -> bool {
        self.pooled.is_some()
    }

    /// Whether a window of `steps` positions can be rolled back by phase on
    /// this channel (pooled, with `steps + 1` phases at least).
    pub fn ring_holds(&self, steps: usize) -> bool {
        match &self.pooled {
            Some(p) => p.pool.phases() > steps,
            None => false,
        }
    }

    /// Advance layer `l` by `steps` phases (a kernel wrote phases
    /// `phase + 1 ..= phase + steps`).
    pub fn advance_phase(&mut self, l: usize, steps: usize) -> Result<()> {
        let p = match &mut self.pooled {
            Some(p) => p,
            None => lloom_tensor::bail!("RecurrentState::advance_phase on an unpooled channel"),
        };
        let n = p.pool.phases();
        p.phase[l] = (p.phase[l] + steps) % n;
        p.written[l] = true;
        self.slab[l] = Some(Self::pool_view(p, l, p.phase[l])?);
        Ok(())
    }

    /// Flip layer `l` to the next phase (the batched kernel wrote it).
    pub fn flip_phase(&mut self, l: usize) -> Result<()> {
        self.advance_phase(l, 1)
    }

    /// Accept `a` of the positions written inside the open window on every
    /// layer: `phase = base + a` (the ring holds them), the window closed.
    /// `a = 0` is the rewind; `a` = the window's length the resolve.
    pub fn accept_window(&mut self, a: usize) -> Result<()> {
        if !self.window_open {
            lloom_tensor::bail!("RecurrentState: no speculative window is open");
        }
        let p = match &mut self.pooled {
            Some(p) => p,
            None => lloom_tensor::bail!("RecurrentState::accept_window needs a pooled channel"),
        };
        let n = p.pool.phases();
        if a >= n {
            lloom_tensor::bail!("RecurrentState::accept_window: {a} positions over {n} phases");
        }
        for l in 0..self.slab.len() {
            // A layer the window never stepped keeps its base (its phase did
            // not move); a stepped one lands at `base + a`.
            let stepped = p.phase[l] != p.base_phase[l] || p.written[l] != p.base_written[l];
            if stepped {
                p.phase[l] = (p.base_phase[l] + a) % n;
                p.written[l] = p.base_written[l] || a > 0;
            }
            self.slab[l] = if p.written[l] {
                Some(Self::pool_view(p, l, p.phase[l])?)
            } else {
                None
            };
        }
        self.window_open = false;
        Ok(())
    }

    /// A fresh channel for `n_layers`, every layer at its implicit zero state.
    /// No device buffers yet — the slab is allocated lazily on first write, at
    /// the shape the block chose (so the channel stays shape-agnostic).
    pub fn new(n_layers: usize) -> Self {
        Self {
            slab: vec![None; n_layers],
            base: vec![None; n_layers],
            spare: vec![None; n_layers],
            window_open: false,
            held: vec![None; n_layers],
            held_open: false,
            pooled: None,
        }
    }

    /// Whether layer `l`'s slab is the open window's base (a write must not
    /// land in it).
    fn slab_is_base(&self, l: usize) -> bool {
        match (&self.slab[l], &self.base[l]) {
            (Some(s), Some(b)) => s.id() == b.id(),
            _ => false,
        }
    }

    /// Whether `buf` is the held window's base at layer `l`.
    fn held_has(&self, l: usize, buf: &Tensor) -> bool {
        self.held[l].as_ref().is_some_and(|b| b.id() == buf.id())
    }

    /// Whether `t` is a buffer a window still restores — the open window's
    /// base or the held one's — at layer `l`: never written, never retired as
    /// the spare.
    fn is_kept(&self, l: usize, t: &Tensor) -> bool {
        self.base[l].as_ref().is_some_and(|b| b.id() == t.id())
            || self.held[l].as_ref().is_some_and(|b| b.id() == t.id())
    }

    /// A buffer of the slab's shape for layer `l` that is neither the slab
    /// nor the window's base — for a kernel that writes the next state out
    /// of place; hand it back with [`Self::swap_in`]. The retired buffer of
    /// the last swap when there is one (no allocation), else a fresh one.
    /// Layer `l` must have been written.
    pub fn take_spare(&mut self, l: usize) -> Result<Tensor> {
        if l >= self.slab.len() {
            lloom_tensor::bail!("RecurrentState::take_spare: layer {l} out of range");
        }
        if let Some(p) = &self.pooled {
            if !p.written[l] {
                lloom_tensor::bail!("RecurrentState::take_spare: layer {l} unwritten");
            }
            return Self::pool_view(p, l, (p.phase[l] + 1) % p.pool.phases());
        }
        let cur = match &self.slab[l] {
            Some(c) => c,
            None => lloom_tensor::bail!("RecurrentState::take_spare: layer {l} unwritten"),
        };
        if let Some(sp) = self.spare[l].take() {
            if sp.dims() == cur.dims() && sp.dtype() == cur.dtype() {
                return Ok(sp);
            }
        }
        let _declared = lloom_tensor::EscapeGuard::new();
        Tensor::zeros(cur.shape(), cur.dtype(), cur.device())
    }

    /// Layer `l`'s next state IS `next` (a buffer of its own — the spare a
    /// kernel wrote, from [`Self::take_spare`]): the slab becomes `next`, the
    /// buffer it replaces is retired as the spare unless it is the open
    /// window's base, which stays the base.
    pub fn swap_in(&mut self, l: usize, next: Tensor) -> Result<()> {
        if l >= self.slab.len() {
            lloom_tensor::bail!("RecurrentState::swap_in: layer {l} out of range");
        }
        if let Some(p) = &self.pooled {
            // `next` is the other phase's view (from `take_spare`), written
            // by a kernel: flip to it.
            let other = Self::pool_view(p, l, (p.phase[l] + 1) % p.pool.phases())?;
            if other.dims() != next.dims() {
                lloom_tensor::bail!(
                    "RecurrentState::swap_in: a pooled layer takes its own spare back ({:?} vs {:?})",
                    next.dims(),
                    other.dims()
                );
            }
            return self.flip_phase(l);
        }
        if let Some(cur) = &self.slab[l] {
            if cur.dims() != next.dims() {
                lloom_tensor::bail!(
                    "RecurrentState::swap_in: layer {l} shape {:?} != slab {:?}",
                    next.dims(),
                    cur.dims()
                );
            }
        }
        let old = self.slab[l].replace(next);
        if let Some(old) = old {
            if !self.is_kept(l, &old) && old.id() != self.slab[l].as_ref().expect("set").id() {
                self.spare[l] = Some(old);
            }
        }
        Ok(())
    }

    /// The current state for layer `l`, or `None` if the sequence has not written
    /// it yet (the block substitutes its zero state).
    pub fn get(&self, l: usize) -> Option<&Tensor> {
        self.slab.get(l).and_then(|s| s.as_ref())
    }

    /// Write layer `l`'s next state, IN PLACE into the persistent slab.
    ///
    /// The first write allocates the slab (pool-backed) and takes ownership of
    /// `next` as the buffer; every subsequent write `slice_set`s `next` into the
    /// existing slab under an [`EscapeGuard`] (so no pass plan tries to serve the
    /// persistent buffer, `DESIGN.md` §9.2aa) — no allocation. Skips the copy
    /// when `next` already IS the slab buffer (a fused kernel wrote it in place).
    pub fn write(&mut self, l: usize, next: Tensor) -> Result<()> {
        if l >= self.slab.len() {
            lloom_tensor::bail!("RecurrentState::write: layer {l} out of range");
        }
        if let Some(p) = &self.pooled {
            // Into the other phase (the current one may be a window's base),
            // then flip. (A kernel that wrote the spare hands it back through
            // `swap_in`, never here.)
            let other = Self::pool_view(p, l, (p.phase[l] + 1) % p.pool.phases())?;
            if other.elem_count() != next.elem_count() {
                lloom_tensor::bail!(
                    "RecurrentState::write: layer {l} shape {:?} != the pool's {:?}",
                    next.dims(),
                    other.dims()
                );
            }
            {
                let _declared = lloom_tensor::EscapeGuard::new();
                other.slice_set(&next.contiguous()?.reshape(other.shape())?, 0, 0)?;
            }
            return self.flip_phase(l);
        }
        match &self.slab[l] {
            // Under an open window the slab may be the base: copy on write
            // into the spare and swap, the base untouched.
            Some(buf)
                if ((self.window_open && self.slab_is_base(l)) || self.held_has(l, buf))
                    && buf.id() != next.id() =>
            {
                if buf.dims() != next.dims() {
                    lloom_tensor::bail!(
                        "RecurrentState::write: layer {l} shape {:?} != slab {:?}",
                        next.dims(),
                        buf.dims()
                    );
                }
                let target = self.take_spare(l)?;
                {
                    let _declared = lloom_tensor::EscapeGuard::new();
                    target.slice_set(&next.contiguous()?, 0, 0)?;
                }
                self.swap_in(l, target)?;
            }
            // Steady state: copy into the existing buffer in place.
            Some(buf) => {
                if state_write_by_copy() {
                    // Instrument: replace the slab with a fresh own-storage copy
                    // instead of writing it in place (the A/B for an in-place
                    // write hazard; `QWEN4EXP_STATE_WRITE=copy`, read once).
                    let _declared = lloom_tensor::EscapeGuard::new();
                    self.slab[l] = Some(own_copy(&next)?);
                    return Ok(());
                }
                if buf.id() != next.id() {
                    if buf.dims() != next.dims() {
                        lloom_tensor::bail!(
                            "RecurrentState::write: layer {l} shape {:?} != slab {:?}",
                            next.dims(),
                            buf.dims()
                        );
                    }
                    let _declared = lloom_tensor::EscapeGuard::new();
                    // Copy the whole buffer in place (dim 0, offset 0). The slab
                    // is the persistent buffer; `next` is the arena transient.
                    buf.slice_set(&next.contiguous()?, 0, 0)?;
                }
            }
            // First write on this layer: take a persistent OWN-storage copy of
            // `next` (which is an arena transient the pass will recycle — adopting
            // it directly would alias recycled memory). The copy allocates the
            // slab buffer once; every later write reuses it in place.
            None => {
                let _declared = lloom_tensor::EscapeGuard::new();
                self.slab[l] = Some(own_copy(&next)?);
            }
        }
        Ok(())
    }

    /// Number of layers this channel covers.
    pub fn n_layers(&self) -> usize {
        self.slab.len()
    }

    /// Return every layer to its implicit zero state (a sequence reset). Keeps
    /// the allocated buffers — a reset is not a free — but marks them unwritten
    /// so the next block sees the zero state. Clears any open window.
    pub fn reset(&mut self) {
        for s in &mut self.slab {
            *s = None;
        }
        for s in &mut self.base {
            *s = None;
        }
        for s in &mut self.held {
            *s = None;
        }
        self.held_open = false;
        if let Some(p) = &mut self.pooled {
            for w in &mut p.written {
                *w = false;
            }
        }
        self.window_open = false;
    }

    /// Open a speculative window: snapshot every written layer's state in place,
    /// so a rejected verify can rewind here. Refuses a second open — one window
    /// at a time (the verify loop's invariant). Allocates the snapshot buffer
    /// once (first open), reuses it in place after.
    pub fn open_window(&mut self) -> Result<()> {
        if self.window_open {
            lloom_tensor::bail!("RecurrentState: a speculative window is already open");
        }
        if let Some(p) = &mut self.pooled {
            p.base_phase.clone_from(&p.phase);
            p.base_written.clone_from(&p.written);
            self.window_open = true;
            return Ok(());
        }
        // The base is the slab's buffer itself (a shared handle, no copy):
        // every write inside the window goes to another buffer.
        for l in 0..self.slab.len() {
            self.base[l] = self.slab[l].clone();
        }
        self.window_open = true;
        Ok(())
    }

    /// Accept the window: the current state stands, the snapshot is retained as a
    /// preallocated buffer for the next window (not freed). After a fully-accepted
    /// verify the slab already reflects all `K` steps.
    pub fn resolve_window(&mut self) -> Result<()> {
        if !self.window_open {
            lloom_tensor::bail!("RecurrentState: no speculative window is open");
        }
        if self.pooled.is_some() {
            self.window_open = false;
            return Ok(());
        }
        // A base the window moved away from is a free buffer: the spare —
        // unless the held window still restores it.
        for l in 0..self.slab.len() {
            if let Some(b) = self.base[l].take() {
                let moved = self.slab[l]
                    .as_ref()
                    .map(|s| s.id() != b.id())
                    .unwrap_or(true);
                if moved && !self.held_has(l, &b) && self.spare[l].is_none() {
                    self.spare[l] = Some(b);
                }
            }
        }
        self.window_open = false;
        Ok(())
    }

    /// Reject (partially or wholly) the window: restore each layer's slab from
    /// its snapshot IN PLACE. The caller replays the accepted prefix afterwards —
    /// the decode recurrence is K-step-replayable, so a checkpoint + replay is the
    /// rollback, no per-step undo (§10.2a).
    pub fn rewind_to_window(&mut self) -> Result<()> {
        if !self.window_open {
            lloom_tensor::bail!("RecurrentState: no speculative window is open");
        }
        if self.pooled.is_some() {
            // The phases past the base were written only inside the window.
            return self.accept_window(0);
        }
        for l in 0..self.slab.len() {
            let base = self.base[l].take();
            let cur = self.slab[l].take();
            // The buffer the window wrote is free (unless it IS the base, or
            // the held window's).
            if let Some(c) = &cur {
                let is_base = base.as_ref().map(|b| b.id() == c.id()).unwrap_or(false);
                if !is_base && !self.held_has(l, c) && self.spare[l].is_none() {
                    self.spare[l] = cur.clone();
                }
            }
            // A layer written only inside the window: its pre-window state
            // was the zero state, so it drops back to unwritten.
            self.slab[l] = base;
        }
        self.window_open = false;
        Ok(())
    }

    /// Whether a speculative window is currently open.
    pub fn window_open(&self) -> bool {
        self.window_open
    }

    /// Hold the open window behind the next one: its base stays restorable
    /// ([`Self::unhold_window`] after the newer window is rewound) or is
    /// released ([`Self::resolve_held`]) once the step it covers is settled,
    /// while a new window opens over the current state. One held window at a
    /// time; refused on a pooled channel (its rollback is by phase, and a
    /// held window would need a third).
    pub fn hold_window(&mut self) -> Result<()> {
        if !self.window_open {
            lloom_tensor::bail!("RecurrentState: no window open to hold");
        }
        if self.held_open {
            lloom_tensor::bail!("RecurrentState: a window is already held");
        }
        if self.pooled.is_some() {
            lloom_tensor::bail!("RecurrentState: a pooled channel does not hold a window");
        }
        for l in 0..self.slab.len() {
            self.held[l] = self.base[l].take();
        }
        self.window_open = false;
        self.held_open = true;
        Ok(())
    }

    /// The held window's step stands: its base is a free buffer (the spare,
    /// when the spare is empty and the buffer is neither the slab nor the
    /// open window's base).
    pub fn resolve_held(&mut self) -> Result<()> {
        if !self.held_open {
            lloom_tensor::bail!("RecurrentState: no window is held");
        }
        for l in 0..self.slab.len() {
            if let Some(b) = self.held[l].take() {
                let in_use = self.slab[l].as_ref().is_some_and(|s| s.id() == b.id())
                    || self.base[l].as_ref().is_some_and(|x| x.id() == b.id());
                if !in_use && self.spare[l].is_none() {
                    self.spare[l] = Some(b);
                }
            }
        }
        self.held_open = false;
        Ok(())
    }

    /// The held window becomes the open one again: the newer window was
    /// rewound (the state is the held step's), and the held step is now the
    /// one a rewind undoes. Refused while a window is open.
    pub fn unhold_window(&mut self) -> Result<()> {
        if !self.held_open {
            lloom_tensor::bail!("RecurrentState: no window is held");
        }
        if self.window_open {
            lloom_tensor::bail!("RecurrentState: unhold under an open window");
        }
        for l in 0..self.slab.len() {
            self.base[l] = self.held[l].take();
        }
        self.held_open = false;
        self.window_open = true;
        Ok(())
    }

    /// Whether a window is held behind the open one.
    pub fn window_held(&self) -> bool {
        self.held_open
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::Device;

    fn t(v: f32, dev: &Device) -> Tensor {
        Tensor::full(v, (2, 2), dev).unwrap()
    }

    fn val(rs: &RecurrentState) -> f32 {
        rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0]
    }

    /// A window held behind the open one is undone two steps deep: the newer
    /// window rewound, the held one made the open one again and rewound.
    #[test]
    fn a_held_window_restores_two_steps_back() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(0.0, &dev)).unwrap();
        rs.open_window().unwrap();
        rs.write(0, t(1.0, &dev)).unwrap();
        rs.hold_window().unwrap();
        rs.open_window().unwrap();
        rs.write(0, t(2.0, &dev)).unwrap();
        assert_eq!(val(&rs), 2.0);
        rs.rewind_to_window().unwrap();
        assert_eq!(val(&rs), 1.0, "the newer window rewound");
        rs.unhold_window().unwrap();
        assert!(!rs.window_held() && rs.window_open());
        rs.rewind_to_window().unwrap();
        assert_eq!(val(&rs), 0.0, "the held window rewound");
    }

    /// Steady state under a held window, out of place as the fused kernels
    /// write (`take_spare` + `swap_in`): each step holds the last window,
    /// opens its own, writes, and the held step is settled. The held base
    /// keeps its value while the newer step writes, and the layer rotates
    /// three buffers with no allocation after the first lap.
    #[test]
    fn a_held_window_rotates_three_buffers_and_keeps_its_base() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(0.0, &dev)).unwrap();
        rs.open_window().unwrap();
        let mut ids = std::collections::HashSet::new();
        for step in 1..=12 {
            let before = val(&rs);
            rs.hold_window().unwrap();
            rs.open_window().unwrap();
            let sp = rs.take_spare(0).unwrap();
            sp.slice_set(&t(step as f32, &dev), 0, 0).unwrap();
            rs.swap_in(0, sp).unwrap();
            ids.insert(rs.get(0).unwrap().id());
            // The held base is the state two steps back, untouched.
            assert_eq!(
                rs.held[0].as_ref().unwrap().to_vec2::<f32>().unwrap()[0][0],
                before - 1.0_f32.min(before),
                "step {step}: the held base moved"
            );
            rs.resolve_held().unwrap();
            assert_eq!(val(&rs), step as f32);
        }
        assert!(ids.len() <= 3, "{} buffers for one layer", ids.len());
    }

    /// A held step that did not write the layer leaves the slab the held
    /// base; with no window open over it, a write there copies out of it —
    /// never in place.
    #[test]
    fn a_write_with_no_window_open_copies_out_of_a_held_base() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(0.0, &dev)).unwrap();
        rs.open_window().unwrap();
        rs.hold_window().unwrap();
        rs.write(0, t(2.0, &dev)).unwrap();
        assert_eq!(val(&rs), 2.0);
        assert_eq!(
            rs.held[0].as_ref().unwrap().to_vec2::<f32>().unwrap()[0][0],
            0.0
        );
    }

    /// A write lands in neither base, in place or not.
    #[test]
    fn a_write_never_lands_in_a_held_base() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(0.0, &dev)).unwrap();
        rs.open_window().unwrap();
        // The step does not write the layer: the open base IS the slab.
        rs.hold_window().unwrap();
        rs.open_window().unwrap();
        rs.write(0, t(5.0, &dev)).unwrap();
        rs.write(0, t(6.0, &dev)).unwrap();
        assert_eq!(val(&rs), 6.0);
        assert_eq!(
            rs.held[0].as_ref().unwrap().to_vec2::<f32>().unwrap()[0][0],
            0.0
        );
        rs.rewind_to_window().unwrap();
        rs.unhold_window().unwrap();
        rs.rewind_to_window().unwrap();
        assert_eq!(val(&rs), 0.0);
    }

    #[test]
    fn fresh_state_is_zero_slots() {
        let rs = RecurrentState::new(3);
        assert_eq!(rs.n_layers(), 3);
        assert!(rs.get(0).is_none());
        assert!(!rs.window_open());
    }

    #[test]
    fn write_get_roundtrips() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(2);
        rs.write(0, t(1.0, &dev)).unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 1.0);
        assert!(rs.get(1).is_none());
    }

    /// The hot-path property: after the first write allocates the slab, every
    /// subsequent write reuses the SAME buffer (in-place `slice_set`), so no
    /// allocation churns per token. Checked by buffer identity.
    #[test]
    fn write_is_in_place_after_first() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(1.0, &dev)).unwrap();
        let id_first = rs.get(0).unwrap().id();
        // A second, third write must land in the same buffer.
        rs.write(0, t(2.0, &dev)).unwrap();
        assert_eq!(
            rs.get(0).unwrap().id(),
            id_first,
            "write allocated a new buffer"
        );
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 2.0);
        rs.write(0, t(3.0, &dev)).unwrap();
        assert_eq!(rs.get(0).unwrap().id(), id_first);
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 3.0);
    }

    #[test]
    fn window_reject_restores_snapshot() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(1.0, &dev)).unwrap();
        rs.open_window().unwrap();
        // A speculative step advances the state...
        rs.write(0, t(9.0, &dev)).unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 9.0);
        // ...then the verify rejects it: state returns to the snapshot.
        rs.rewind_to_window().unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 1.0);
        assert!(!rs.window_open());
    }

    #[test]
    fn window_accept_keeps_advanced_state() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(1.0, &dev)).unwrap();
        rs.open_window().unwrap();
        rs.write(0, t(9.0, &dev)).unwrap();
        rs.resolve_window().unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 9.0);
        assert!(!rs.window_open());
    }

    /// A layer written only INSIDE the window (zero pre-window state) drops back
    /// to unwritten on reject — its pre-window state was the implicit zero.
    #[test]
    fn window_reject_of_first_ever_write_drops_to_zero() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.open_window().unwrap(); // nothing written yet
        rs.write(0, t(5.0, &dev)).unwrap(); // first write is inside the window
        assert!(rs.get(0).is_some());
        rs.rewind_to_window().unwrap();
        assert!(
            rs.get(0).is_none(),
            "should return to the implicit zero state"
        );
    }

    /// The window is zero-copy: opening it does not touch the slab, a write
    /// inside it lands in ANOTHER buffer (the base is intact), a rewind is
    /// the base handle back, and a resolve retires the base as the spare
    /// the next swap reuses — no allocation in the steady state.
    #[test]
    fn window_is_zero_copy_over_swapped_slabs() {
        let dev = Device::Cpu;
        let mut rs = RecurrentState::new(1);
        rs.write(0, t(1.0, &dev)).unwrap();
        let base_id = rs.get(0).unwrap().id();
        rs.open_window().unwrap();
        assert_eq!(rs.get(0).unwrap().id(), base_id, "open copied the slab");
        rs.write(0, t(9.0, &dev)).unwrap();
        let moved_id = rs.get(0).unwrap().id();
        assert_ne!(
            moved_id, base_id,
            "a write inside the window landed in the base"
        );
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 9.0);
        rs.rewind_to_window().unwrap();
        assert_eq!(
            rs.get(0).unwrap().id(),
            base_id,
            "the rewind is not the base handle"
        );
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 1.0);
        // The retired buffer is the spare: the next out-of-place step reuses it.
        let sp = rs.take_spare(0).unwrap();
        assert_eq!(
            sp.id(),
            moved_id,
            "the retired buffer was not kept as the spare"
        );
        rs.swap_in(0, sp).unwrap();
        assert_eq!(rs.get(0).unwrap().id(), moved_id);
        // A resolved window releases its base as the spare.
        rs.open_window().unwrap();
        let sp = rs.take_spare(0).unwrap();
        assert_eq!(
            sp.id(),
            base_id,
            "the base handed out as a spare under its window"
        );
        rs.swap_in(0, sp).unwrap();
        rs.resolve_window().unwrap();
        assert_eq!(rs.get(0).unwrap().id(), base_id);
        assert_eq!(rs.take_spare(0).unwrap().id(), moved_id);
    }

    /// A pooled channel: its state is a row of the pool at a phase; a write
    /// lands in the other phase and flips; `take_spare` / `swap_in` flip
    /// without a copy; a window rewinds by flipping back; a clone leases a
    /// slot of its own with the state copied; a drop frees the slot.
    #[test]
    fn pooled_channel_flips_phases_and_rewinds() {
        let dev = Device::Cpu;
        let pool = StatePool::new(2, 1, &[2, 2], lloom_tensor::DType::F32, &dev);
        let mut rs = RecurrentState::in_pool(&pool).unwrap();
        assert!(rs.is_pooled());
        assert!(rs.get(0).is_none());
        rs.write(0, t(1.0, &dev)).unwrap();
        let (_, slot, phase, written) = rs.pool_slot(0).unwrap();
        assert!(written);
        assert_eq!(phase, 1, "the first write flips to phase 1");
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 1.0);
        // The pool's tensor at the row: the value landed there.
        let t0 = pool.layer(0).unwrap();
        assert_eq!(
            t0.narrow(0, 1, 1)
                .unwrap()
                .narrow(1, slot, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0],
            1.0
        );
        // A kernel's out-of-place step: the spare is the other phase's row.
        let sp = rs.take_spare(0).unwrap();
        sp.slice_set(&t(5.0, &dev), 0, 0).unwrap();
        rs.swap_in(0, sp).unwrap();
        assert_eq!(rs.pool_slot(0).unwrap().2, 0);
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 5.0);
        // A window: the write inside flips; the rewind flips back.
        rs.open_window().unwrap();
        rs.write(0, t(9.0, &dev)).unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 9.0);
        rs.rewind_to_window().unwrap();
        assert_eq!(rs.pool_slot(0).unwrap().2, 0);
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 5.0);
        // A clone leases the other slot and carries the state.
        let c = rs.clone();
        assert_ne!(c.pool_slot(0).unwrap().1, slot);
        assert_eq!(c.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 5.0);
        assert!(
            RecurrentState::in_pool(&pool).is_err(),
            "two slots, two leases"
        );
        drop(c);
        assert!(
            RecurrentState::in_pool(&pool).is_ok(),
            "the clone's slot freed"
        );
        // A reset: unwritten again, the slot kept.
        rs.reset();
        assert!(rs.get(0).is_none());
        assert!(rs.take_spare(0).is_err());
    }

    /// A ring of 4 phases: three writes inside a window land in phases
    /// 1..=3 and `accept_window(2)` lands on the second, its value.
    #[test]
    fn ring_accepts_a_prefix_by_phase() {
        let dev = Device::Cpu;
        let pool = StatePool::with_phases(1, 4, 1, &[2, 2], lloom_tensor::DType::F32, &dev);
        let mut rs = RecurrentState::in_pool(&pool).unwrap();
        rs.write(0, t(1.0, &dev)).unwrap();
        assert!(rs.ring_holds(3) && !rs.ring_holds(4));
        rs.open_window().unwrap();
        for v in [2.0f32, 3.0, 4.0] {
            rs.write(0, t(v, &dev)).unwrap();
        }
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 4.0);
        rs.accept_window(2).unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 3.0);
        assert!(!rs.window_open());
        rs.open_window().unwrap();
        rs.write(0, t(9.0, &dev)).unwrap();
        rs.accept_window(0).unwrap();
        assert_eq!(rs.get(0).unwrap().to_vec2::<f32>().unwrap()[0][0], 3.0);
    }

    #[test]
    fn double_open_and_orphan_resolve_refused() {
        let mut rs = RecurrentState::new(1);
        rs.open_window().unwrap();
        assert!(rs.open_window().is_err());
        rs.resolve_window().unwrap();
        assert!(rs.resolve_window().is_err());
        assert!(rs.rewind_to_window().is_err());
    }
}
