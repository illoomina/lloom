//! The turn-checkpoint ring (`sequence-reuse-design.md` §2.3b): the
//! bookkeeping every family that restores a re-rendered turn shares —
//! which positions a checkpoint stands at, the ring of frames the
//! recurrent layers' copies live in, the hint a caller gave for where the
//! next pass should leave one, and the marks (a KV cache's lengths) a
//! restore needs beside the frame. What a layer writes into its slice of a
//! frame, and how it is put back, is the layer's (`nn::decoder::MixerBlock`,
//! `qwen4exp::State`); this owns the slots.

use crate::nn::decoder::WindowMark;
use crate::nn::ring::Ring;
use crate::nn::zeros as zs;
use lloom_tensor::{DType, Device, Result, Tensor};

/// A checkpoint: the position it stands at and, per layer, the mark a
/// restore needs beside the ring frame.
#[derive(Debug, Clone)]
struct Checkpoint {
    pos: usize,
    valid: bool,
    marks: Vec<WindowMark>,
}

/// `slots` frames of one [`Ring`], each every layer's `elems` f32s laid end
/// to end (`layout`: offset and count per layer); allocated at a sequence's
/// first checkpoint and kept, a slot reused oldest first.
#[derive(Debug, Clone)]
pub struct CheckpointRing {
    slots: usize,
    layout: Vec<(usize, usize)>,
    ring: Option<Ring>,
    checkpoints: Vec<Checkpoint>,
    next: usize,
    /// Where the passes that follow should leave their checkpoint
    /// ([`Self::hint`]).
    want: Option<usize>,
    /// Until a caller hints, every multi-token pass checkpoints its end; a
    /// hinted state is driven by its hints alone (a tiled prefill's other
    /// tiles take none), until a reset.
    auto: bool,
}

impl CheckpointRing {
    /// `slots` checkpoints, each `elems[l]` f32s for layer `l` (0: the
    /// layer's checkpoint is its mark alone). `slots` 0 keeps none.
    pub fn new(elems: &[usize], slots: usize) -> Self {
        let mut off = 0;
        let layout = elems
            .iter()
            .map(|&n| {
                let o = off;
                off += n;
                (o, n)
            })
            .collect();
        Self {
            slots,
            layout,
            ring: None,
            checkpoints: Vec::new(),
            next: 0,
            want: None,
            auto: true,
        }
    }

    pub fn none() -> Self {
        Self::new(&[], 0)
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn layers(&self) -> usize {
        self.layout.len()
    }

    fn elems(&self) -> usize {
        self.layout.iter().map(|(_, n)| n).sum()
    }

    /// One frame's bytes.
    pub fn frame_bytes(&self) -> usize {
        self.elems() * DType::F32.size_in_bytes()
    }

    /// The ring's bytes (allocated at the first checkpoint).
    pub fn bytes(&self) -> usize {
        self.slots * self.frame_bytes()
    }

    /// The positions the valid checkpoints stand at.
    pub fn positions(&self) -> Vec<usize> {
        self.checkpoints
            .iter()
            .filter(|c| c.valid)
            .map(|c| c.pos)
            .collect()
    }

    /// Have the passes that follow leave their checkpoint at `pos`. Ignored
    /// without slots.
    pub fn hint(&mut self, pos: usize) {
        if self.slots > 0 {
            self.want = Some(pos);
            self.auto = false;
        }
    }

    pub fn hinted(&self) -> Option<usize> {
        self.want
    }

    /// A hint at a position already passed is stale; one a checkpoint
    /// already stands at is met. Called at a forward's entry, at `pos`.
    pub fn drop_stale_hint(&mut self, pos: usize) {
        if let Some(w) = self.want {
            if w < pos || self.index(w).is_some() {
                self.want = None;
            }
        }
    }

    /// Where a chunk of `len` positions starting at `at` should leave its
    /// checkpoint, as rows into the chunk: at the hinted position when the
    /// chunk covers it, else — with no hint pending, on the last chunk of
    /// a multi-token pass, for a state no caller has hinted — at its end.
    /// `None`: nowhere. Refused (`None`) under `windowed`.
    pub fn rows_for(
        &self,
        at: usize,
        len: usize,
        last_of_pass: bool,
        seq: usize,
        windowed: bool,
    ) -> Option<usize> {
        if self.slots == 0 || windowed {
            return None;
        }
        if let Some(w) = self.want.filter(|&w| w >= at && w <= at + len) {
            Some(w - at)
        } else if self.auto && self.want.is_none() && last_of_pass && seq >= 2 {
            Some(len)
        } else {
            None
        }
    }

    /// Frame `slot`'s view for layer `l`, `[elems]` f32; `None` for a layer
    /// whose checkpoint is its mark alone.
    pub fn frame(&self, slot: usize, l: usize) -> Result<Option<Tensor>> {
        let (off, n) = self.layout[l];
        if n == 0 {
            return Ok(None);
        }
        let ring = self
            .ring
            .as_ref()
            .ok_or_else(|| lloom_tensor::Error::msg("turn checkpoints: no ring allocated"))?;
        Ok(Some(ring.frame(slot)?.narrow(0, off, n)?))
    }

    /// The slot the next checkpoint, at `pos`, goes in: the oldest, its
    /// frame rewritten; the ring allocated at the first (one request,
    /// outside every plan, kept for the sequence). [`Self::close`] makes it
    /// valid.
    pub fn open(&mut self, pos: usize, dev: &Device) -> Result<usize> {
        if self.slots == 0 {
            lloom_tensor::bail!("turn checkpoints: no slots")
        }
        let elems = self.elems();
        if elems > 0 && self.ring.is_none() {
            self.ring = Some(Ring::allocate(
                self.slots,
                elems,
                DType::F32,
                dev,
                zs::NN_DECODER_CHECKPOINTS,
                "turn-checkpoints",
            )?);
        }
        let slot = self.next % self.slots;
        let c = Checkpoint {
            pos,
            valid: false,
            marks: Vec::new(),
        };
        if slot < self.checkpoints.len() {
            self.checkpoints[slot] = c;
        } else {
            self.checkpoints.push(c);
        }
        self.next = (slot + 1) % self.slots;
        Ok(slot)
    }

    pub fn close(&mut self, slot: usize, marks: Vec<WindowMark>) {
        let c = &mut self.checkpoints[slot];
        c.marks = marks;
        c.valid = true;
        if self.want == Some(c.pos) {
            self.want = None;
        }
    }

    /// Checkpoints past `pos` describe positions the state no longer holds.
    pub fn forget_past(&mut self, pos: usize) {
        for c in &mut self.checkpoints {
            if c.pos > pos {
                c.valid = false;
            }
        }
    }

    /// Every checkpoint forgotten and the hint dropped (a reset); the ring
    /// stays for the next sequence.
    pub fn forget_all(&mut self) {
        for c in &mut self.checkpoints {
            c.valid = false;
        }
        self.want = None;
        self.auto = true;
    }

    /// The slot of the valid checkpoint standing at exactly `pos`.
    pub fn index(&self, pos: usize) -> Option<usize> {
        self.checkpoints
            .iter()
            .position(|c| c.valid && c.pos == pos)
    }

    /// The slot and position of the latest valid checkpoint at or before
    /// `pos`: where a request diverging at `pos` can be put back.
    pub fn latest_at_or_before(&self, pos: usize) -> Option<(usize, usize)> {
        self.checkpoints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.valid && c.pos <= pos)
            .max_by_key(|(_, c)| c.pos)
            .map(|(i, c)| (i, c.pos))
    }

    /// The marks of slot `slot`.
    pub fn marks(&self, slot: usize) -> &[WindowMark] {
        &self.checkpoints[slot].marks
    }
}
