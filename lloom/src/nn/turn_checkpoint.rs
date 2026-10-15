//! The turn-checkpoint ring (`sequence-reuse-design.md` §2.3b): the
//! bookkeeping every family that restores a re-rendered turn shares —
//! which positions a checkpoint stands at, the ring of frames the
//! recurrent layers' copies live in, where the passes that follow should
//! leave one (a request's [`CheckpointPolicy`], or a caller's hint), and
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

/// `slots` frames of one [`Ring`], each every layer's `elems` f32s laid end
/// to end (`layout`: offset and count per layer); allocated at a sequence's
/// first checkpoint and kept, a forgotten slot reused before a valid one and
/// the oldest valid one last.
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
    }

    /// The request's end: a target no pass met is dropped, and the ring
    /// returns to the state a fresh ring is in, so what follows does not
    /// depend on what the request asked.
    pub fn close_request(&mut self) {
        self.request = None;
        self.want = None;
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

    /// Whether a pass with no target pending checkpoints its end: no
    /// request is open and no caller has hinted.
    fn takes_auto(&self) -> bool {
        self.request.is_none() && self.auto && self.want.is_none()
    }

    /// Which checkpoint a pass of `seq` positions from `pos` will take, as
    /// the pass key's variant (`PassKey::variant`): 0 for none, 1 at the
    /// pass's end — whether the ring's own or a target there, the same copy
    /// in the same chunk — else nonzero and a function of the target's
    /// offset into the pass. A checkpoint adds a copy (at a chunk's end) or
    /// a second recurrence over the prefix (inside a chunk), so two passes
    /// of one shape that leave theirs at different places allocate
    /// different sequences, and a plan recorded from one must not serve the
    /// other. Decided as the forward decides it ([`Self::rows_for`] after
    /// [`Self::drop_stale_hint`]); the chunking inside the pass is the
    /// model's constant, so the offset fixes it.
    pub fn pass_variant(&self, pos: usize, seq: usize, windowed: bool) -> u32 {
        if self.slots == 0 || windowed || seq == 0 {
            return 0;
        }
        let want = self.want.filter(|&w| w >= pos && self.index(w).is_none());
        match want {
            Some(w) if w == pos + seq => 1,
            Some(w) if w < pos + seq => 2 | (((w - pos) as u32) << 2),
            None if self.request.is_none() && self.auto && seq >= 2 => 1,
            _ => 0,
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

    /// The slot the next checkpoint, at `pos`, goes in: one whose checkpoint
    /// a rewind or a restore forgot, else the oldest, its frame rewritten —
    /// so the checkpoint a restore returned to outlives the pass after it.
    /// The ring is allocated at the first (one request, outside every plan,
    /// kept for the sequence). [`Self::close`] makes it valid.
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

    /// Every checkpoint forgotten, the hint and the open request dropped (a
    /// reset); the ring stays for the next sequence.
    pub fn forget_all(&mut self) {
        for c in &mut self.checkpoints {
            c.valid = false;
        }
        self.want = None;
        self.request = None;
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
    /// the rows into it (`rows_for` after `drop_stale_hint`, then the
    /// checkpoint taken there, as the forward does) and the tile's
    /// variant.
    fn run(r: &mut CheckpointRing, pos: usize, tiles: &[usize]) -> Vec<(Option<usize>, u32)> {
        let mut at = pos;
        let mut out = Vec::new();
        for &len in tiles {
            r.drop_stale_hint(at);
            let variant = r.pass_variant(at, len, false);
            let rows = r.rows_for(at, len, true, len, false);
            if let Some(rows) = rows {
                take(r, at + rows);
            }
            out.push((rows, variant));
            at += len;
        }
        out
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
}
