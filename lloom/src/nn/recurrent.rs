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
}

/// Whether every state write replaces the slab with a fresh copy instead of
/// writing it in place (`QWEN4EXP_STATE_WRITE=copy`, read once). A debugging
/// instrument behind a gate (§3.3a): an allocation per layer per token,
/// never the shipping path.
fn state_write_by_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("QWEN4EXP_STATE_WRITE")
            .map(|v| v == "copy")
            .unwrap_or(false)
    })
}

/// A copy of `t` in storage of its own. `Tensor::copy` is NOT that on Metal —
/// `MetalStorage::try_clone` hands back the same buffer — so a snapshot taken
/// with it aliases what it snapshots; the CPU backend copies. This allocates
/// and copies on every backend.
fn own_copy(t: &Tensor) -> Result<Tensor> {
    let out = Tensor::zeros(t.shape(), t.dtype(), t.device())?;
    out.slice_set(&t.contiguous()?, 0, 0)?;
    Ok(out)
}

impl RecurrentState {
    /// A fresh channel for `n_layers`, every layer at its implicit zero state.
    /// No device buffers yet — the slab is allocated lazily on first write, at
    /// the shape the block chose (so the channel stays shape-agnostic).
    pub fn new(n_layers: usize) -> Self {
        Self {
            slab: vec![None; n_layers],
            base: vec![None; n_layers],
            spare: vec![None; n_layers],
            window_open: false,
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

    /// A buffer of the slab's shape for layer `l` that is neither the slab
    /// nor the window's base — for a kernel that writes the next state out
    /// of place; hand it back with [`Self::swap_in`]. The retired buffer of
    /// the last swap when there is one (no allocation), else a fresh one.
    /// Layer `l` must have been written.
    pub fn take_spare(&mut self, l: usize) -> Result<Tensor> {
        if l >= self.slab.len() {
            lloom_tensor::bail!("RecurrentState::take_spare: layer {l} out of range");
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
        if let Some(cur) = &self.slab[l] {
            if cur.dims() != next.dims() {
                lloom_tensor::bail!(
                    "RecurrentState::swap_in: layer {l} shape {:?} != slab {:?}",
                    next.dims(),
                    cur.dims()
                );
            }
        }
        let was_base = self.slab_is_base(l);
        let old = self.slab[l].replace(next);
        if let Some(old) = old {
            if !was_base && old.id() != self.slab[l].as_ref().expect("set").id() {
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
        match &self.slab[l] {
            // Under an open window the slab may be the base: copy on write
            // into the spare and swap, the base untouched.
            Some(buf) if self.window_open && self.slab_is_base(l) && buf.id() != next.id() => {
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
        // A base the window moved away from is a free buffer: the spare.
        for l in 0..self.slab.len() {
            if let Some(b) = self.base[l].take() {
                let moved = self.slab[l]
                    .as_ref()
                    .map(|s| s.id() != b.id())
                    .unwrap_or(true);
                if moved && self.spare[l].is_none() {
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
        for l in 0..self.slab.len() {
            let base = self.base[l].take();
            let cur = self.slab[l].take();
            // The buffer the window wrote is free (unless it IS the base).
            if let Some(c) = &cur {
                let is_base = base.as_ref().map(|b| b.id() == c.id()).unwrap_or(false);
                if !is_base && self.spare[l].is_none() {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::Device;

    fn t(v: f32, dev: &Device) -> Tensor {
        Tensor::full(v, (2, 2), dev).unwrap()
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
