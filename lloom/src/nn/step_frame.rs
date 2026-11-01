//! The step frame ring (`profiles-and-memory-design.md` §4.6): a step's
//! per-sequence inputs — the live key count, the KV write slots — written by
//! the host into a frame of one persistent Shared ring and bound by the step's
//! kernels at the frame's offset, so a decode step uploads nothing.
//!
//! Frame `k` is the GPU's from the step that takes it until the epoch open
//! when the next step begins has completed, then the host's again. A frame is
//! never written while a queued step can still read it, so a rewind or a reset
//! is just a different value in the next frame: there is no host store racing
//! a queued reader. The ring is sized by what is in flight, which the cache
//! does not know (the decode driver's lead is a run-time value): when every
//! frame is still queued it grows by a segment, up to [`MAX_STEP_FRAMES`];
//! past that [`StepFrames::take`] returns `None` and the caller uploads that
//! step's values as before, counted in [`StepFrames::busy`].
//!
//! Two layout rules (§4.6). Every field is a `u32`, so a GPU read never sees
//! half of a host store. A frame's host-written words fill whole
//! [`LINE_BYTES`] lines, so a field the GPU writes — none yet — can be given a
//! line of its own, never sharing one with the host's.

use lloom_tensor::retire::{Completed, Epoch};

/// The line a frame's host-written region is padded to.
pub const LINE_BYTES: usize = 128;
/// [`LINE_BYTES`] in `u32` words.
pub const LINE_WORDS: usize = LINE_BYTES / std::mem::size_of::<u32>();

/// Frames a ring starts with, and adds when every frame is queued.
pub const STEP_FRAMES: usize = 8;

/// Frames a ring grows to at most: past the decode driver's default lead
/// (`DecodeOptions::adaptive_lead`, 16 queued + 8 run ahead) with room for
/// a deeper configured one.
pub const MAX_STEP_FRAMES: usize = 128;

/// Which frame a step may write, decided on the GPU clock alone.
/// Backend-neutral: a CPU test drives it with numbers of its own.
#[derive(Debug, Clone)]
pub struct FrameOwners {
    /// Per frame, the epoch that must complete before the host writes it
    /// again; `None` for a frame no step has taken.
    held: Vec<Option<Epoch>>,
    /// The frame the previous step took, not yet closed.
    open: Option<usize>,
    next: usize,
}

impl FrameOwners {
    pub fn new(frames: usize) -> Self {
        Self {
            held: vec![None; frames],
            open: None,
            next: 0,
        }
    }

    pub fn frames(&self) -> usize {
        self.held.len()
    }

    /// Add `n` frames no step has taken.
    pub fn grow(&mut self, n: usize) {
        self.held.extend(std::iter::repeat_n(None, n));
    }

    /// Close the previous step's frame at `now` and take the next frame no
    /// queued step can read, or `None` when every frame is still queued.
    ///
    /// `now` is the epoch open at the call: every dispatch of the previous
    /// step was encoded before it, so none outlives it. `done` is the GPU's
    /// completed count.
    pub fn take(&mut self, now: Epoch, done: Completed) -> Option<usize> {
        if let Some(i) = self.open.take() {
            self.held[i] = Some(now);
        }
        let n = self.held.len();
        let free = (0..n)
            .map(|k| (self.next + k) % n)
            .find(|&i| self.held[i].is_none_or(|e| done.covers(e)))?;
        self.next = (free + 1) % n;
        self.open = Some(free);
        Some(free)
    }
}

#[cfg(feature = "metal")]
pub use metal::{OwnedFrames, StepFrames};

#[cfg(feature = "metal")]
mod metal {
    use super::{FrameOwners, LINE_WORDS, MAX_STEP_FRAMES, STEP_FRAMES};
    use lloom_tensor::kernels::metal::GpuClock;
    use lloom_tensor::metal_backend::HostSlot;
    use lloom_tensor::{DType, MetalDevice, Result, Tensor};
    use std::sync::Arc;

    /// A ring of frames of `u32` words, in segments of [`STEP_FRAMES`] frames,
    /// each one persistent Shared buffer.
    #[derive(Debug)]
    pub struct StepFrames {
        /// Each segment's mapping and a tensor over its bytes.
        segments: Vec<(HostSlot, Tensor)>,
        device: MetalDevice,
        /// Words per frame, a whole number of lines.
        stride: usize,
        owners: FrameOwners,
        clock: Arc<GpuClock>,
        busy: u64,
    }

    impl StepFrames {
        /// A ring of frames of at least `words` host-written words each.
        pub fn new(device: &MetalDevice, clock: Arc<GpuClock>, words: usize) -> Result<Self> {
            let stride = words.max(1).div_ceil(LINE_WORDS) * LINE_WORDS;
            let mut ring = Self {
                segments: Vec::new(),
                device: device.clone(),
                stride,
                owners: FrameOwners::new(0),
                clock,
                busy: 0,
            };
            ring.add_segment()?;
            Ok(ring)
        }

        fn add_segment(&mut self) -> Result<()> {
            let words = STEP_FRAMES * self.stride;
            let slot = HostSlot::new(&self.device, words, DType::U32)?;
            let t = slot.tensor(words)?;
            self.segments.push((slot, t));
            self.owners.grow(STEP_FRAMES);
            Ok(())
        }

        /// Frames the ring holds now.
        pub fn frames(&self) -> usize {
            self.owners.frames()
        }

        /// Words a frame holds.
        pub fn words(&self) -> usize {
            self.stride
        }

        /// The frame for the step about to be encoded, growing the ring when
        /// every frame is still queued; `None` when it is at its bound (the
        /// caller uploads instead).
        pub fn take(&mut self) -> Result<Option<usize>> {
            let (now, done) = (self.clock.current_epoch(), self.clock.completed_count());
            if let Some(f) = self.owners.take(now, done) {
                return Ok(Some(f));
            }
            if self.owners.frames() + STEP_FRAMES <= MAX_STEP_FRAMES {
                self.add_segment()?;
                return Ok(self.owners.take(now, done));
            }
            self.busy += 1;
            Ok(None)
        }

        fn at(&self, frame: usize) -> (&Tensor, usize) {
            (
                &self.segments[frame / STEP_FRAMES].1,
                (frame % STEP_FRAMES) * self.stride,
            )
        }

        /// Write `words` into frame `frame` from word `at`.
        ///
        /// Only for a frame [`Self::take`] just returned: no queued step
        /// reads it.
        pub fn write(&self, frame: usize, at: usize, words: &[u32]) -> Result<()> {
            if at + words.len() > self.stride {
                lloom_tensor::bail!(
                    "step frame: {} words at {at} past a {}-word frame",
                    words.len(),
                    self.stride
                )
            }
            let (seg, base) = self.at(frame);
            // SAFETY: `take` handed out `frame` only once every step that
            // could read it had completed, and the range is inside it.
            if !unsafe { seg.write_host_slice::<u32>(base + at, words) } {
                lloom_tensor::bail!("step frame: the ring is not host-mapped")
            }
            Ok(())
        }

        /// `len` words of frame `frame` from word `at`, as a tensor a kernel
        /// binds at that offset.
        pub fn field(&self, frame: usize, at: usize, len: usize) -> Result<Tensor> {
            let (seg, base) = self.at(frame);
            seg.narrow(0, base + at, len)
        }

        /// Steps that found every frame queued and uploaded instead.
        pub fn busy(&self) -> u64 {
            self.busy
        }
    }

    /// A [`StepFrames`] its owner creates on first use. A clone starts with
    /// none: two owners writing one ring would overwrite each other's queued
    /// frames.
    #[derive(Debug, Default)]
    pub struct OwnedFrames(pub Option<StepFrames>);

    impl Clone for OwnedFrames {
        fn clone(&self) -> Self {
            Self(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_handed_out_again_only_after_its_step_completes() {
        let mut o = FrameOwners::new(3);
        let none = Completed::NONE;
        // Three steps in epochs 0, 1, 2, nothing completed: three frames.
        assert_eq!(o.take(Epoch(0), none), Some(0));
        assert_eq!(o.take(Epoch(1), none), Some(1));
        assert_eq!(o.take(Epoch(2), none), Some(2));
        // The fourth finds every frame queued (frame 2 closed at epoch 3).
        assert_eq!(o.take(Epoch(3), none), None);
        // Epoch 1 done: frame 0 (closed at 1) is free, frame 1 (at 2) is not.
        assert_eq!(o.take(Epoch(3), Completed(2)), Some(0));
        assert_eq!(o.take(Epoch(4), Completed(2)), None);
        // Everything up to epoch 4 done: frames 1, 2 and 0 come back in turn.
        assert_eq!(o.take(Epoch(5), Completed(5)), Some(1));
        assert_eq!(o.take(Epoch(6), Completed(5)), Some(2));
        assert_eq!(o.take(Epoch(7), Completed(6)), Some(0));
    }

    #[test]
    fn the_previous_frame_is_closed_at_the_epoch_open_when_the_next_step_begins() {
        let mut o = FrameOwners::new(2);
        assert_eq!(o.take(Epoch(10), Completed(10)), Some(0));
        // The step that took frame 0 encoded into epochs up to 12.
        assert_eq!(o.take(Epoch(12), Completed(12)), Some(1));
        // Epoch 11 done but not 12: frame 0 is still the GPU's.
        assert_eq!(o.take(Epoch(13), Completed(12)), None);
        assert_eq!(o.take(Epoch(13), Completed(13)), Some(0));
    }

    #[test]
    fn grown_frames_are_free_and_the_queued_ones_stay_held() {
        let mut o = FrameOwners::new(2);
        assert_eq!(o.take(Epoch(0), Completed::NONE), Some(0));
        assert_eq!(o.take(Epoch(1), Completed::NONE), Some(1));
        assert_eq!(o.take(Epoch(2), Completed::NONE), None);
        o.grow(2);
        assert_eq!(o.take(Epoch(2), Completed::NONE), Some(2));
        assert_eq!(o.take(Epoch(3), Completed::NONE), Some(3));
        assert_eq!(o.take(Epoch(4), Completed::NONE), None);
        assert_eq!(o.frames(), 4);
    }

    #[test]
    fn a_frame_is_a_whole_number_of_lines() {
        assert_eq!(LINE_WORDS * 4, LINE_BYTES);
        assert_eq!(LINE_BYTES, 128);
    }
}
