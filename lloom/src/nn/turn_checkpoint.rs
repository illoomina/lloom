//! The turn-checkpoint ring (`sequence-reuse-design.md` §2.3b): the
//! bookkeeping every family that restores a re-rendered turn shares —
//! which positions a checkpoint stands at, the ring of frames the
//! recurrent layers' copies live in, where the passes that follow should
//! leave one (a request's [`CheckpointPolicy`], or a caller's hint), the
//! pinned checkpoint a session keeps of its head (`sessions.md` §8), and
//! the marks (a KV cache's lengths) a restore needs beside the frame. What
//! a layer writes into its slice of a frame, and how it is put back, is the
//! layer's (`nn::decoder::MixerBlock`, `qwen4exp::State`); this owns the
//! slots.

use crate::nn::decoder::WindowMark;
use crate::nn::ring::Ring;
use crate::nn::zeros as zs;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The turn checkpoint a request's passes leave. Every request that opens
/// one ([`CheckpointRing::open_request`]) states it; nothing an earlier
/// request asked carries over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    /// Leave none.
    None,
    /// At the end of the request's prefill, in the pass that ends it.
    End,
    /// At this absolute position, in whichever pass covers it.
    At(usize),
}

/// A checkpoint: the position it stands at and, per layer, the mark a
/// restore needs beside the ring frame.
#[derive(Debug, Clone)]
struct Checkpoint {
    pos: usize,
    valid: bool,
    marks: Vec<WindowMark>,
}

/// One checkpoint a chunk of a pass leaves: the state after the chunk's
/// first `rows` positions, into the pin's frame (`pin`) or a turn slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Take {
    pub rows: usize,
    pub pin: bool,
}

/// The checkpoints a chunk leaves ([`CheckpointRing::takes_for`]): at most
/// two — the request's own and the pin's — in row order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Takes([Option<Take>; 2]);

impl Takes {
    fn of(a: Option<Take>, b: Option<Take>) -> Self {
        match (a, b) {
            (Some(x), Some(y)) if y.rows < x.rows => Self([Some(y), Some(x)]),
            (None, b) => Self([b, None]),
            (a, b) => Self([a, b]),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Take> + '_ {
        self.0.iter().flatten().copied()
    }

    pub fn len(&self) -> usize {
        self.0.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.0[0].is_none()
    }
}

/// The widest offsets a pass that leaves both a request's checkpoint and
/// the pin's can name in its plan key ([`CheckpointRing::pass_variant`]):
/// the request's from the pass's end, the pin's from its start.
const TWO_TURN_BITS: u32 = 14;
const TWO_PIN_BITS: u32 = 16;

/// `slots` frames of one [`Ring`], each every layer's `elems` f32s laid end
/// to end (`layout`: offset and count per layer); allocated at a sequence's
/// first checkpoint and kept, a forgotten slot reused before a valid one and
/// the oldest valid one last. With a pin frame ([`Self::with_pin_frame`])
/// the ring holds one frame more, for the pinned checkpoint, which `open`
/// never gives out: the turn checkpoints rotate through `slots` as they do
/// without one.
#[derive(Debug, Clone)]
pub struct CheckpointRing {
    slots: usize,
    layout: Vec<(usize, usize)>,
    ring: Option<Ring>,
    checkpoints: Vec<Checkpoint>,
    next: usize,
    /// Where the passes that follow should leave their checkpoint: the open
    /// request's target, or a caller's [`Self::hint`].
    want: Option<usize>,
    /// The policy of the request whose passes are running, from
    /// [`Self::open_request`] to [`Self::close_request`]. While it is open
    /// the ring leaves that request's checkpoint and no other.
    request: Option<CheckpointPolicy>,
    /// With no request open: until a caller hints, every multi-token pass
    /// checkpoints its end; a hinted state is driven by its hints alone (a
    /// tiled prefill's other tiles take none), until a reset or a request.
    auto: bool,
    /// The ring holds a frame past the slots for the pinned checkpoint.
    pin_frame: bool,
    /// The frame each slot's checkpoint lives in, the pin's last (index
    /// `slots`): pinning a slot's checkpoint swaps the two, so it leaves the
    /// slots without a copy and its slot is forgotten.
    phys: Vec<usize>,
    /// The pinned checkpoint, in frame `phys[slots]`.
    pinned: Option<Checkpoint>,
    /// The open request's second target: a checkpoint at this position,
    /// taken into the pin's frame by the pass that covers it and pinned.
    pin_want: Option<usize>,
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
            request: None,
            auto: true,
            pin_frame: false,
            phys: (0..slots).collect(),
            pinned: None,
            pin_want: None,
        }
    }

    pub fn none() -> Self {
        Self::new(&[], 0)
    }

    /// One frame more, for a pinned checkpoint ([`Self::pin`]). Set before
    /// the first checkpoint allocates the ring.
    pub fn with_pin_frame(mut self, on: bool) -> Self {
        self.pin_frame = on;
        self.phys = (0..self.frames()).collect();
        self
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Whether the ring holds a frame for a pinned checkpoint.
    pub fn has_pin_frame(&self) -> bool {
        self.pin_frame
    }

    /// The frames the ring allocates: the slots, and the pin's.
    fn frames(&self) -> usize {
        self.slots + usize::from(self.pin_frame)
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

    /// The ring's bytes (allocated at the first checkpoint), the pin's frame
    /// included.
    pub fn bytes(&self) -> usize {
        self.frames() * self.frame_bytes()
    }

    /// The positions the valid checkpoints stand at, the pinned one's
    /// included.
    pub fn positions(&self) -> Vec<usize> {
        self.checkpoints
            .iter()
            .chain(self.pinned.iter())
            .filter(|c| c.valid)
            .map(|c| c.pos)
            .collect()
    }

    /// Open a request whose prefill ends at `end` (absolute): its passes
    /// leave the checkpoint `policy` names, in the pass that covers it, and
    /// no other — no tile's end, and nothing once it is met. Replaces
    /// whatever an earlier request or hint left; holds across every pass
    /// until [`Self::close_request`] or a reset.
    pub fn open_request(&mut self, policy: CheckpointPolicy, end: usize) {
        self.request = Some(policy);
        self.want = match policy {
            CheckpointPolicy::None => None,
            CheckpointPolicy::End => Some(end),
            CheckpointPolicy::At(p) => Some(p),
        };
        self.pin_want = None;
    }

    /// The request's end: a target no pass met — its own or the pin's — is
    /// dropped, and the ring returns to the state a fresh ring is in, so
    /// what follows does not depend on what the request asked.
    pub fn close_request(&mut self) {
        self.request = None;
        self.want = None;
        self.pin_want = None;
        self.auto = true;
    }

    /// The policy of the open request, if one is open.
    pub fn request_policy(&self) -> Option<CheckpointPolicy> {
        self.request
    }

    /// Have the passes that follow leave their checkpoint at `pos`. Ignored
    /// without slots. Inside an open request it replaces the request's
    /// target.
    pub fn hint(&mut self, pos: usize) {
        if self.slots > 0 {
            self.want = Some(pos);
            self.auto = false;
        }
    }

    pub fn hinted(&self) -> Option<usize> {
        self.want
    }

    /// Keep the checkpoint standing at `pos` until a rewind or a restore
    /// goes below it, or a reset: `open` never takes its frame. One pin; a
    /// new one replaces it. `false` where no checkpoint stands at `pos`, and
    /// on a ring without a pin frame.
    pub fn pin(&mut self, pos: usize) -> bool {
        if !self.pin_frame {
            return false;
        }
        if self.pinned_at(pos) {
            return true;
        }
        let Some(s) = self
            .checkpoints
            .iter()
            .position(|c| c.valid && c.pos == pos)
        else {
            return false;
        };
        self.phys.swap(s, self.slots);
        self.pinned = Some(self.checkpoints[s].clone());
        self.checkpoints[s].valid = false;
        if self.pin_want == Some(pos) {
            self.pin_want = None;
        }
        true
    }

    /// Release the pin, and a second target not yet taken.
    pub fn unpin(&mut self) {
        self.pinned = None;
        self.pin_want = None;
    }

    /// Where the pinned checkpoint stands.
    pub fn pinned(&self) -> Option<usize> {
        self.pinned.as_ref().filter(|c| c.valid).map(|c| c.pos)
    }

    fn pinned_at(&self, pos: usize) -> bool {
        self.pinned() == Some(pos)
    }

    /// Pin the checkpoint at `pos`: the one standing there now; else, while
    /// a request is open, the one its passes leave there beside the
    /// request's own — the second target, taken into the pin's frame in the
    /// same pass and pinned. Returns whether one is pinned at `pos` now.
    pub fn pin_at(&mut self, pos: usize) -> bool {
        if self.pin(pos) {
            return true;
        }
        if self.pin_frame && self.request.is_some() {
            self.pin_want = Some(pos);
        }
        false
    }

    /// The second target, pending.
    pub fn pin_wanted(&self) -> Option<usize> {
        self.pin_want
    }

    /// A hint at a position already passed is stale; one a checkpoint
    /// already stands at is met. A second target already passed is
    /// dropped; one a checkpoint stands at is pinned there. Called at a
    /// forward's entry, at `pos`.
    pub fn drop_stale_hint(&mut self, pos: usize) {
        if let Some(w) = self.want {
            if w < pos || self.index(w).is_some() {
                self.want = None;
            }
        }
        if let Some(w) = self.pin_want {
            if w < pos || self.pin(w) {
                self.pin_want = None;
            }
        }
    }

    /// Where a chunk of `len` positions starting at `at` should leave its
    /// checkpoint, as rows into the chunk: at the target when the chunk
    /// covers it, else — with no target pending and no request open, on the
    /// last chunk of a multi-token pass, for a state no caller has hinted —
    /// at its end. `None`: nowhere. Refused (`None`) under `windowed`.
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
        } else if self.takes_auto() && last_of_pass && seq >= 2 {
            Some(len)
        } else {
            None
        }
    }

    /// [`Self::rows_for`] and the second target's, for the chunk of `len`
    /// positions at `at` of the pass of `seq` positions at `pass`: what the
    /// chunk leaves. The two at one position are one take, into the pin's
    /// frame. A second target whose pass also leaves the request's own at
    /// offsets the plan key cannot name ([`Self::pass_variant`]) is not
    /// taken; [`Self::settle_pass`] drops it before the pass runs.
    pub fn takes_for(
        &self,
        at: usize,
        len: usize,
        last_of_pass: bool,
        pass: usize,
        seq: usize,
        windowed: bool,
    ) -> Takes {
        let own = self
            .rows_for(at, len, last_of_pass, seq, windowed)
            .map(|rows| Take { rows, pin: false });
        let pin = self
            .pin_target(pass, seq, windowed)
            .filter(|&w| w >= at && w <= at + len)
            .map(|w| Take {
                rows: w - at,
                pin: true,
            });
        match (own, pin) {
            (Some(o), Some(p)) if o.rows == p.rows => Takes::of(Some(p), None),
            (o, p) => Takes::of(o, p),
        }
    }

    /// The second target as the pass of `seq` positions at `pos` takes it:
    /// pending, inside the pass, with no checkpoint standing there (one that
    /// stands is pinned at the pass's entry instead, [`Self::drop_stale_hint`]),
    /// and nameable beside the request's own target (see
    /// [`Self::pass_variant`]).
    fn pin_target(&self, pos: usize, seq: usize, windowed: bool) -> Option<usize> {
        if !self.pin_frame || windowed || seq == 0 {
            return None;
        }
        let w = self
            .pin_want
            .filter(|&w| w >= pos && w <= pos + seq && self.index(w).is_none())?;
        match self.own_target(pos, seq) {
            Some(t) if t != w => {
                let fits = (pos + seq - t) < (1 << TWO_TURN_BITS) && w - pos < (1 << TWO_PIN_BITS);
                fits.then_some(w)
            }
            _ => Some(w),
        }
    }

    /// The request's own (or a hint's) target inside the pass of `seq` at
    /// `pos`, not yet standing.
    fn own_target(&self, pos: usize, seq: usize) -> Option<usize> {
        self.want
            .filter(|&w| w >= pos && w <= pos + seq && self.index(w).is_none())
    }

    /// Before the pass of `seq` positions at `pos` runs: a second target it
    /// covers that it cannot take ([`Self::takes_for`]) is dropped, so the
    /// request reports it unpinned rather than a later pass taking it.
    pub fn settle_pass(&mut self, pos: usize, seq: usize, windowed: bool) {
        let covered = self
            .pin_want
            .is_some_and(|w| w >= pos && w <= pos + seq && !windowed && seq > 0);
        if covered && self.pin_target(pos, seq, windowed).is_none() {
            self.pin_want = None;
        }
    }

    /// Whether a pass with no target pending checkpoints its end: no
    /// request is open and no caller has hinted.
    fn takes_auto(&self) -> bool {
        self.request.is_none() && self.auto && self.want.is_none()
    }

    /// Which checkpoints a pass of `seq` positions from `pos` will take, as
    /// the pass key's variant (`PassKey::variant`): 0 for none, 1 at the
    /// pass's end — whether the ring's own, a target there or the pin's,
    /// the same copy in the same chunk — `2 | offset << 2` for one inside
    /// the pass, and `3 | from_end << 2 | pin_offset << 16` for a pass that
    /// leaves both the request's own and the pin's. A checkpoint adds a copy
    /// (at a chunk's end) or a second recurrence over the prefix (inside a
    /// chunk), so two passes of one shape that leave theirs at different
    /// places allocate different sequences, and a plan recorded from one
    /// must not serve the other. Decided as the forward decides it
    /// ([`Self::takes_for`] after [`Self::drop_stale_hint`] and
    /// [`Self::settle_pass`]); the chunking inside the pass is the model's
    /// constant, so the offsets fix it.
    pub fn pass_variant(&self, pos: usize, seq: usize, windowed: bool) -> u32 {
        if windowed || seq == 0 {
            return 0;
        }
        let one = |w: usize| -> u32 {
            if w == pos + seq {
                1
            } else {
                2 | (((w - pos) as u32) << 2)
            }
        };
        let pin = self.pin_target(pos, seq, windowed);
        if self.slots == 0 {
            return pin.map_or(0, one);
        }
        let own = self.own_target(pos, seq);
        match (own, pin) {
            (Some(t), Some(p)) if t != p => {
                3 | (((pos + seq - t) as u32) << 2) | (((p - pos) as u32) << TWO_PIN_BITS)
            }
            (_, Some(p)) => one(p),
            (Some(t), None) => one(t),
            (None, None)
                if self.want.is_none() && self.request.is_none() && self.auto && seq >= 2 =>
            {
                1
            }
            _ => 0,
        }
    }

    /// Frame `slot`'s view for layer `l`, `[elems]` f32; `None` for a layer
    /// whose checkpoint is its mark alone. The pin's slot is `slots`.
    pub fn frame(&self, slot: usize, l: usize) -> Result<Option<Tensor>> {
        let (off, n) = self.layout[l];
        if n == 0 {
            return Ok(None);
        }
        let ring = self
            .ring
            .as_ref()
            .ok_or_else(|| lloom_tensor::Error::msg("turn checkpoints: no ring allocated"))?;
        let Some(&phys) = self.phys.get(slot) else {
            lloom_tensor::bail!("turn checkpoints: no slot {slot}")
        };
        Ok(Some(ring.frame(phys)?.narrow(0, off, n)?))
    }

    /// The ring, allocated at the first checkpoint (one request, outside
    /// every plan, kept for the sequence).
    fn allocate(&mut self, dev: &Device) -> Result<()> {
        let elems = self.elems();
        if elems > 0 && self.ring.is_none() {
            self.ring = Some(Ring::allocate(
                self.frames(),
                elems,
                DType::F32,
                dev,
                zs::NN_DECODER_CHECKPOINTS,
                "turn-checkpoints",
            )?);
        }
        Ok(())
    }

    /// The slot the next checkpoint, at `pos`, goes in: one whose checkpoint
    /// a rewind or a restore forgot, else the oldest, its frame rewritten —
    /// so the checkpoint a restore returned to outlives the pass after it.
    /// Never the pin's. [`Self::close`] makes it valid.
    pub fn open(&mut self, pos: usize, dev: &Device) -> Result<usize> {
        if self.slots == 0 {
            lloom_tensor::bail!("turn checkpoints: no slots")
        }
        self.allocate(dev)?;
        let n = self.slots;
        let slot = if self.checkpoints.len() < n {
            self.checkpoints.len()
        } else {
            (0..n)
                .map(|k| (self.next + k) % n)
                .find(|&s| !self.checkpoints[s].valid)
                .unwrap_or(self.next % n)
        };
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
        if slot == self.next % n {
            self.next = (slot + 1) % n;
        }
        Ok(slot)
    }

    /// The pin's slot, for the second target's checkpoint at `pos`: the
    /// pinned one it replaces is gone. [`Self::close`] makes it valid and
    /// pinned.
    pub fn open_pin(&mut self, pos: usize, dev: &Device) -> Result<usize> {
        if !self.pin_frame {
            lloom_tensor::bail!("turn checkpoints: no frame for a pin")
        }
        self.allocate(dev)?;
        self.pinned = Some(Checkpoint {
            pos,
            valid: false,
            marks: Vec::new(),
        });
        Ok(self.slots)
    }

    pub fn close(&mut self, slot: usize, marks: Vec<WindowMark>) {
        let c = if slot == self.slots && self.pin_frame {
            match self.pinned.as_mut() {
                Some(c) => c,
                None => return,
            }
        } else {
            &mut self.checkpoints[slot]
        };
        c.marks = marks;
        c.valid = true;
        let pos = c.pos;
        if self.want == Some(pos) {
            self.want = None;
        }
        if self.pin_want == Some(pos) && slot == self.slots {
            self.pin_want = None;
        }
    }

    /// Checkpoints past `pos` describe positions the state no longer holds;
    /// a pin past it is released.
    pub fn forget_past(&mut self, pos: usize) {
        for c in &mut self.checkpoints {
            if c.pos > pos {
                c.valid = false;
            }
        }
        if self.pinned.as_ref().is_some_and(|c| c.pos > pos) {
            self.pinned = None;
        }
    }

    /// Every checkpoint forgotten, the pin released, the hint and the open
    /// request dropped (a reset); the ring stays for the next sequence.
    pub fn forget_all(&mut self) {
        for c in &mut self.checkpoints {
            c.valid = false;
        }
        self.pinned = None;
        self.pin_want = None;
        self.want = None;
        self.request = None;
        self.auto = true;
    }

    /// The slot of the valid checkpoint standing at exactly `pos` (the
    /// pin's is `slots`).
    pub fn index(&self, pos: usize) -> Option<usize> {
        self.checkpoints
            .iter()
            .position(|c| c.valid && c.pos == pos)
            .or_else(|| self.pinned_at(pos).then_some(self.slots))
    }

    /// The slot and position of the latest valid checkpoint at or before
    /// `pos`: where a request diverging at `pos` can be put back.
    pub fn latest_at_or_before(&self, pos: usize) -> Option<(usize, usize)> {
        let pinned = self.pinned().filter(|&p| p <= pos).map(|p| (self.slots, p));
        self.checkpoints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.valid && c.pos <= pos)
            .map(|(i, c)| (i, c.pos))
            .chain(pinned)
            .max_by_key(|&(_, p)| p)
    }

    /// The marks of slot `slot` (the pin's is `slots`).
    pub fn marks(&self, slot: usize) -> &[WindowMark] {
        if slot == self.slots && self.pin_frame {
            return self.pinned.as_ref().map_or(&[], |c| &c.marks);
        }
        &self.checkpoints[slot].marks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(r: &mut CheckpointRing, pos: usize) {
        let slot = r.open(pos, &Device::Cpu).unwrap();
        r.close(slot, Vec::new());
    }

    fn standing(r: &CheckpointRing) -> Vec<usize> {
        let mut p = r.positions();
        p.sort_unstable();
        p
    }

    /// Two slots; a pass leaves 256, the next 258; a restore at 256 forgets
    /// 258; the pass after the restore takes 258's slot, so a divergence just
    /// before its own checkpoint (373) still restores to 256. With every
    /// slot valid, the oldest goes. (The sequence Qwen3.6-27B's persisted
    /// turns 3–5 ran, `design/measurements/issue-946.md` §16.)
    #[test]
    fn a_new_checkpoint_takes_a_forgotten_slot_before_a_valid_one() {
        let mut r = CheckpointRing::new(&[0], 2);
        take(&mut r, 256);
        take(&mut r, 258);
        r.forget_past(256);
        take(&mut r, 374);
        assert_eq!(standing(&r), vec![256, 374]);
        assert_eq!(r.latest_at_or_before(373).map(|(_, p)| p), Some(256));
        take(&mut r, 500);
        assert_eq!(standing(&r), vec![374, 500], "the oldest valid goes");
        r.forget_all();
        take(&mut r, 10);
        take(&mut r, 20);
        assert_eq!(
            standing(&r),
            vec![10, 20],
            "a reset leaves every slot to reuse"
        );
    }

    /// What a pass of `seq` from `pos`, tiled by `tiles`, takes: per tile
    /// the takes (`takes_for` after `drop_stale_hint` and `settle_pass`,
    /// then each checkpoint taken there, as the forward does) and the
    /// tile's variant.
    fn run_takes(r: &mut CheckpointRing, pos: usize, tiles: &[usize]) -> Vec<(Vec<Take>, u32)> {
        let mut at = pos;
        let mut out = Vec::new();
        for &len in tiles {
            r.drop_stale_hint(at);
            r.settle_pass(at, len, false);
            let variant = r.pass_variant(at, len, false);
            let takes = r.takes_for(at, len, true, at, len, false);
            for t in takes.iter() {
                let slot = if t.pin {
                    r.open_pin(at + t.rows, &Device::Cpu).unwrap()
                } else {
                    r.open(at + t.rows, &Device::Cpu).unwrap()
                };
                r.close(slot, Vec::new());
            }
            out.push((takes.iter().collect(), variant));
            at += len;
        }
        out
    }

    /// [`run_takes`] as the turn target's rows alone, for the cases that
    /// leave no pin.
    fn run(r: &mut CheckpointRing, pos: usize, tiles: &[usize]) -> Vec<(Option<usize>, u32)> {
        run_takes(r, pos, tiles)
            .into_iter()
            .map(|(t, v)| {
                assert!(t.iter().all(|t| !t.pin), "no pin was asked");
                (t.first().map(|t| t.rows), v)
            })
            .collect()
    }

    /// One state's requests, no reset between them: a hinted (stripped)
    /// request, then a persisted one whose policy is the prompt's end,
    /// tiled — each leaves its one checkpoint and the tiles before it
    /// none. Without the request policy the zero-tail request after a
    /// hint leaves nothing (the lifecycle defect), and a tiled prefill on
    /// an unhinted state checkpoints every tile's end.
    #[test]
    fn every_request_leaves_its_own_checkpoint_whatever_the_last_asked() {
        let mut r = CheckpointRing::new(&[4], 2);
        r.open_request(CheckpointPolicy::At(18), 20);
        assert_eq!(
            run(&mut r, 0, &[8, 12]),
            vec![(None, 0), (Some(10), 2 | (10 << 2))]
        );
        r.close_request();
        assert_eq!(standing(&r), vec![18]);

        r.open_request(CheckpointPolicy::End, 40);
        assert_eq!(
            run(&mut r, 20, &[8, 8, 4]),
            vec![(None, 0), (None, 0), (Some(4), 1)]
        );
        r.close_request();
        assert_eq!(standing(&r), vec![18, 40], "the persisted request's end");

        r.open_request(CheckpointPolicy::None, 50);
        assert_eq!(run(&mut r, 40, &[6, 4]), vec![(None, 0), (None, 0)]);
        r.close_request();
        assert_eq!(standing(&r), vec![18, 40], "a request that keeps none");

        r.open_request(CheckpointPolicy::At(56), 60);
        assert_eq!(run(&mut r, 50, &[6, 4]), vec![(Some(6), 1), (None, 0)]);
        assert_eq!(standing(&r), vec![40, 56]);
        assert_eq!(
            run(&mut r, 60, &[1, 1]),
            vec![(None, 0), (None, 0)],
            "met: neither the rest of the request nor its decode steps take another"
        );
        r.close_request();

        // The control: the same sequence through the hint alone.
        let mut h = CheckpointRing::new(&[4], 2);
        h.hint(18);
        run(&mut h, 0, &[8, 12]);
        assert_eq!(run(&mut h, 20, &[8, 8, 4]), vec![(None, 0); 3]);
        assert_eq!(
            standing(&h),
            vec![18],
            "a zero-tail request after a hint keeps none"
        );
        let mut a = CheckpointRing::new(&[4], 2);
        assert_eq!(
            run(&mut a, 0, &[8, 8, 4]),
            vec![(Some(8), 1), (Some(8), 1), (Some(4), 1)]
        );
        assert_eq!(standing(&a), vec![16, 20], "auto: the last two tiles' ends");
    }

    /// Closing a request returns the ring to a fresh ring's default and
    /// drops a target no pass met; a reset drops an open request.
    #[test]
    fn closing_a_request_returns_to_the_default() {
        let mut r = CheckpointRing::new(&[4], 2);
        r.hint(3);
        r.open_request(CheckpointPolicy::At(30), 20);
        assert_eq!(r.request_policy(), Some(CheckpointPolicy::At(30)));
        run(&mut r, 0, &[20]);
        r.close_request();
        assert_eq!(r.request_policy(), None);
        assert_eq!(r.hinted(), None, "the unmet target is dropped");
        assert_eq!(
            run(&mut r, 20, &[12]),
            vec![(Some(12), 1)],
            "after the request, a pass takes the fresh ring's default"
        );
        r.open_request(CheckpointPolicy::None, 40);
        r.forget_all();
        assert_eq!(r.request_policy(), None, "a reset closes the request");
        assert_eq!(run(&mut r, 0, &[6]), vec![(Some(6), 1)]);
    }

    /// A checkpoint at the pass's end is one variant whoever asked for it:
    /// the ring's own, `End`, or a target there take the same copy in the
    /// same chunk. Inside the pass the offset names it; without slots,
    /// nothing.
    #[test]
    fn a_checkpoint_at_the_pass_end_is_one_variant() {
        let auto = CheckpointRing::new(&[4], 2);
        let mut end = CheckpointRing::new(&[4], 2);
        end.open_request(CheckpointPolicy::End, 24);
        let mut hinted = CheckpointRing::new(&[4], 2);
        hinted.hint(24);
        for r in [&auto, &end, &hinted] {
            assert_eq!(r.pass_variant(0, 24, false), 1);
            assert_eq!(r.rows_for(0, 24, true, 24, false), Some(24));
        }
        hinted.hint(22);
        assert_eq!(hinted.pass_variant(0, 24, false), 2 | (22 << 2));
        let mut none = CheckpointRing::new(&[4], 0);
        none.open_request(CheckpointPolicy::End, 24);
        assert_eq!(none.pass_variant(0, 24, false), 0);
        assert_eq!(none.rows_for(0, 24, true, 24, false), None);
        assert_eq!(end.pass_variant(0, 24, true), 0, "none under a window");
    }

    /// A session's first request: its own checkpoint at the prompt's end
    /// and the head's at 7, both in the one pass — two takes, the head's
    /// into the pin's frame, pinned — and a variant of its own.
    fn first_turn(r: &mut CheckpointRing, head: usize, end: usize) -> Vec<(Vec<Take>, u32)> {
        r.open_request(CheckpointPolicy::End, end);
        assert!(!r.pin_at(head), "nothing stands at the head yet");
        assert_eq!(r.pin_wanted(), Some(head));
        let out = run_takes(r, 0, &[end]);
        r.close_request();
        out
    }

    /// The pin survives more turns than the ring has slots: three requests
    /// each leave their own checkpoint (two slots, so the third evicts the
    /// first), and the head's, pinned by the first, stands throughout.
    #[test]
    fn a_pin_survives_more_turns_than_the_ring_has_slots() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        assert_eq!(r.bytes(), 3 * 16, "the slots and the pin's frame");
        assert_eq!(
            first_turn(&mut r, 7, 20),
            vec![(
                vec![
                    Take { rows: 7, pin: true },
                    Take {
                        rows: 20,
                        pin: false
                    }
                ],
                // The request's own at the pass's end: 0 from it.
                3 | (7 << TWO_PIN_BITS)
            )]
        );
        assert_eq!(r.pinned(), Some(7));
        assert_eq!(standing(&r), vec![7, 20]);
        for (pos, end) in [(20, 40), (40, 60), (60, 80), (80, 100)] {
            r.open_request(CheckpointPolicy::End, end);
            assert!(r.pin_at(7), "the pin stands: nothing to take");
            assert_eq!(
                run(&mut r, pos, &[end - pos]),
                vec![(Some(end - pos), 1)],
                "an extend's pass takes its own alone, at today's variant"
            );
            r.close_request();
            assert_eq!(r.pinned(), Some(7), "the pin outlives the turns");
        }
        assert_eq!(
            standing(&r),
            vec![7, 80, 100],
            "the two slots rotated; the pinned one stood"
        );
        assert_eq!(r.latest_at_or_before(30).map(|(_, p)| p), Some(7));
        let (slot, _) = r.latest_at_or_before(30).unwrap();
        assert_eq!(slot, r.slots(), "it is the pin's slot");
        assert_eq!(r.index(7), Some(2));
        assert!(r.frame(2, 0).unwrap().is_some(), "the pin's frame");
    }

    /// `open` never gives out the pin's frame: whatever the slots go
    /// through, the frame the pinned checkpoint lives in is never one a
    /// slot is handed.
    #[test]
    fn open_never_gives_out_the_pins_frame() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        first_turn(&mut r, 7, 20);
        let pin_phys = r.phys[r.slots()];
        for k in 0..9 {
            let slot = r.open(30 + k, &Device::Cpu).unwrap();
            assert_ne!(r.phys[slot], pin_phys, "turn {k} took the pin's frame");
            r.close(slot, Vec::new());
            if k % 3 == 1 {
                r.forget_past(30 + k - 1);
            }
        }
        assert_eq!(r.pinned(), Some(7));
    }

    /// Below the pin it goes; at it and above, it stands. A restore at the
    /// pin keeps it, a rewind below releases it, a reset releases it.
    #[test]
    fn a_rewind_or_restore_below_the_pin_releases_it() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        first_turn(&mut r, 7, 20);
        r.forget_past(7);
        assert_eq!(r.pinned(), Some(7), "a restore at the pin keeps it");
        assert_eq!(standing(&r), vec![7]);
        r.forget_past(6);
        assert_eq!(r.pinned(), None, "below it, it is released");
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        first_turn(&mut r, 7, 20);
        r.forget_all();
        assert_eq!(r.pinned(), None, "a reset releases it");
        assert!(standing(&r).is_empty());
    }

    /// One pin: a new one replaces it. Pinning a checkpoint a slot holds
    /// moves it to the pin's frame and forgets the slot, which the next
    /// checkpoint takes first; `unpin` releases it.
    #[test]
    fn one_pin_and_a_new_one_replaces_it() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        take(&mut r, 10);
        take(&mut r, 20);
        assert!(!r.pin(15), "nothing stands at 15");
        assert!(r.pin(10));
        assert_eq!(r.pinned(), Some(10));
        assert_eq!(standing(&r), vec![10, 20]);
        let slot = r.open(30, &Device::Cpu).unwrap();
        r.close(slot, Vec::new());
        assert_eq!(
            standing(&r),
            vec![10, 20, 30],
            "the forgotten slot went first"
        );
        assert!(r.pin(20), "a new pin replaces it");
        assert_eq!(r.pinned(), Some(20));
        assert_eq!(standing(&r), vec![20, 30], "the old pinned one is gone");
        r.unpin();
        assert_eq!(r.pinned(), None);
        let mut plain = CheckpointRing::new(&[4], 2);
        take(&mut plain, 10);
        assert!(!plain.pin(10), "no pin without a pin frame");
        assert!(!plain.pin_at(10));
    }

    /// The second target in a tiled prefill: taken by the tile that covers
    /// it, at a single checkpoint's variant there; the request's own by the
    /// last. A target already passed or one that cannot be named beside the
    /// request's own is dropped, and an unmet one goes with the request.
    #[test]
    fn the_second_target_across_tiles_and_its_limits() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        r.open_request(CheckpointPolicy::At(38), 40);
        r.pin_at(5);
        assert_eq!(
            run_takes(&mut r, 0, &[16, 24]),
            vec![
                (vec![Take { rows: 5, pin: true }], 2 | (5 << 2)),
                (
                    vec![Take {
                        rows: 22,
                        pin: false
                    }],
                    2 | (22 << 2)
                ),
            ]
        );
        r.close_request();
        assert_eq!((r.pinned(), standing(&r)), (Some(5), vec![5, 38]));

        // Already passed: dropped at the pass's entry, nothing pinned.
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        r.open_request(CheckpointPolicy::End, 60);
        r.pin_at(5);
        run_takes(&mut r, 40, &[20]);
        assert_eq!((r.pinned(), r.pin_wanted()), (None, None));
        r.close_request();

        // Past the key's reach beside the request's own: not taken, the
        // pass is today's.
        let far = 1 << TWO_PIN_BITS;
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        r.open_request(CheckpointPolicy::End, far + 10);
        r.pin_at(far + 1);
        assert_eq!(
            run_takes(&mut r, 0, &[far + 10]),
            vec![(
                vec![Take {
                    rows: far + 10,
                    pin: false
                }],
                1
            )]
        );
        assert_eq!(r.pinned(), None);
        r.close_request();

        // A request that ends before the target: dropped with it.
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        r.open_request(CheckpointPolicy::End, 10);
        r.pin_at(30);
        run_takes(&mut r, 0, &[10]);
        r.close_request();
        assert_eq!((r.pinned(), r.pin_wanted()), (None, None));

        // No request open: nothing is scheduled.
        assert!(!r.pin_at(3));
        assert_eq!(r.pin_wanted(), None);
    }

    /// The head and the request's own at one position are one take, into
    /// the pin's frame; it meets both.
    #[test]
    fn the_two_targets_at_one_position_are_one_take() {
        let mut r = CheckpointRing::new(&[4], 2).with_pin_frame(true);
        r.open_request(CheckpointPolicy::At(12), 20);
        r.pin_at(12);
        assert_eq!(
            run_takes(&mut r, 0, &[20]),
            vec![(
                vec![Take {
                    rows: 12,
                    pin: true
                }],
                2 | (12 << 2)
            )]
        );
        assert_eq!((r.pinned(), r.hinted()), (Some(12), None));
        r.close_request();
    }
}
