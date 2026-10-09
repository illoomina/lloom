//! The short causal convolution block: a channel-wise `k`-tap causal conv over
//! each channel's newest inputs, carried across steps as a window of state.
//! LFM2's conv layers and Gated DeltaNet's prologue are both this block
//! (`docs/fusion.md` §12.2 row 1, `docs/sequence-state.md` §10.2).
//!
//! The block is one type over a [`Layout`] — where the window lives:
//!
//! - [`Shift`]: a `[channels, k]` window, the next one written out of place
//!   each step (taps `1..k` of the old window, then the input). LFM2's
//!   `Shuffle` arm; GDN's window, a [`RecurrentState`](super::RecurrentState)
//!   channel whose phases give a speculative window its rollback.
//! - [`Ring`]: a `[channels, width]` column ring written in place at the
//!   input's phase slot, `width − k` columns of history a rollback returns to
//!   (§10.2a). Folded in slot order against a per-phase permuted weight table,
//!   a reduction order of its own (§10.2g).
//!
//! The Metal half is `metal_src/conv/short_conv.metal`, composed into
//! `conv.metal`'s `shortconv_step` / `shortconv_ring` and `gdn.metal`'s
//! prologues. A kernel keeps its epilogue (LFM2's `*C` gate; GDN's SiLU and
//! head norms): the fold is the block's.

use lloom_tensor::{Result, Tensor};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Shift {}
    impl Sealed for super::Ring {}
}

/// Where a [`ShortConv`] keeps its window, and the weights its fold reads.
pub trait Layout: sealed::Sealed {
    /// The weights in the form this layout's fold reads.
    type Weights: std::fmt::Debug + Clone;
    const NAME: &'static str;
}

/// The `[channels, k]` window, shifted out of place each step.
#[derive(Debug, Clone, Copy)]
pub struct Shift;

/// The `[channels, width]` column ring, written in place.
#[derive(Debug, Clone, Copy)]
pub struct Ring;

impl Layout for Shift {
    type Weights = ();
    const NAME: &'static str = "shift";
}

impl Layout for Ring {
    type Weights = RingWeights;
    const NAME: &'static str = "ring";
}

/// A ring's weights: for each phase (the slot the newest input occupies) the
/// taps permuted onto the slots holding their inputs, zero on the history.
#[derive(Debug, Clone)]
pub struct RingWeights {
    width: usize,
    /// `[channels, width]` per phase.
    per_phase: Vec<Tensor>,
    /// The same stacked `[width, channels, width]`: the operand the fused
    /// kernel indexes by phase.
    table: Tensor,
}

/// One layer's short causal conv: its `[channels, k]` taps (oldest first) and
/// what its layout's fold reads besides.
#[derive(Debug, Clone)]
pub struct ShortConv<L: Layout> {
    weight: Tensor,
    channels: usize,
    k: usize,
    layout: L::Weights,
    _layout: PhantomData<L>,
}

impl<L: Layout> ShortConv<L> {
    /// The taps, `[channels, k]`, oldest first.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

impl ShortConv<Shift> {
    /// `weight` is `[channels, k]`; its shape is checked where a kernel is
    /// dispatched against it.
    pub fn new(weight: Tensor, k: usize) -> Self {
        let channels = weight.elem_count() / k.max(1);
        Self {
            weight,
            channels,
            k,
            layout: (),
            _layout: PhantomData,
        }
    }

    /// The step gated by `C` (LFM2's conv block, `shortconv_step`): `bx = b·x`
    /// shifted into `state [b, channels, k]`, folded, `·c`. `b`, `x`, `c` are
    /// `[b, channels, 1]` views sharing one layout. Returns `(out [b, channels,
    /// 1], next_state)`; `state` is read, never written.
    pub fn gated_step(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        crate::nn::ops::shortconv_step(b, x, c, state, &self.weight, self.k)
    }

    /// [`Self::gated_step`] writing the next state into `next`, a contiguous
    /// `[b, channels, k]` buffer that is not `state`'s: a caller that owns two
    /// states per layer alternates them and the step allocates only its
    /// output. Returns the output.
    pub fn gated_step_into(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        state: &Tensor,
        next: &Tensor,
    ) -> Result<Tensor> {
        crate::nn::ops::shortconv_step_into(b, x, c, state, &self.weight, self.k, next)
    }
}

impl ShortConv<Ring> {
    /// `weight` is `[channels, k]`; the ring holds `width ≥ k` columns.
    pub fn new(weight: Tensor, width: usize) -> Result<Self> {
        let (channels, k) = weight.dims2()?;
        if k == 0 || width < k {
            lloom_tensor::bail!("short conv: a {width}-column ring cannot hold {k} taps");
        }
        // The newest input is AT `phase` and the live window is the `k` slots
        // ending there: slot `s` is `(phase + width − s) % width` inputs old,
        // weighted by the tap that age reads, or zero if it is history. A
        // naive rotation pairs slot `s` with tap `s` — a different operator,
        // right only where the rotation is the identity (§10.2g).
        let mut per_phase = Vec::with_capacity(width);
        for phase in 0..width {
            let mut cols = Vec::with_capacity(width);
            for s in 0..width {
                let back = (phase + width - s) % width;
                let col = if back < k {
                    weight.narrow(1, k - 1 - back, 1)?
                } else {
                    Tensor::zeros((channels, 1), weight.dtype(), weight.device())?
                };
                cols.push(col);
            }
            per_phase.push(Tensor::cat(&cols, 1)?.contiguous()?);
        }
        let table = Tensor::stack(&per_phase, 0)?.contiguous()?;
        Ok(Self {
            weight,
            channels,
            k,
            layout: RingWeights {
                width,
                per_phase,
                table,
            },
            _layout: PhantomData,
        })
    }

    pub fn width(&self) -> usize {
        self.layout.width
    }

    /// The taps permuted for `phase`, `[channels, width]`.
    pub fn phase_weight(&self, phase: usize) -> &Tensor {
        &self.layout.per_phase[phase]
    }

    /// Every phase's permuted taps, `[width, channels, width]`.
    pub fn table(&self) -> &Tensor {
        &self.layout.table
    }

    /// `seq_len` positions into `ring [b, channels, width]` (written in place)
    /// gated by `C` (`shortconv_ring`): position `i` at slot `phase_last −
    /// (seq_len − 1 − i)`. `b`, `x`, `c` are `[b, channels, seq_len]` views
    /// sharing one layout. Returns `[b, seq_len, channels]`.
    pub fn gated_pass(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        ring: &Tensor,
        phase_last: usize,
    ) -> Result<Tensor> {
        crate::nn::ops::shortconv_ring(b, x, c, ring, &self.layout.table, phase_last)
    }

    /// [`Self::gated_pass`] of one position for rows of a pool of rings
    /// `[slots, channels, width]`: row `r` writes ring `row_slots[r]` at
    /// `row_phases[r]` (`u32` tensors), one dispatch.
    pub fn gated_rows(
        &self,
        b: &Tensor,
        x: &Tensor,
        c: &Tensor,
        pool: &Tensor,
        row_slots: &Tensor,
        row_phases: &Tensor,
    ) -> Result<Tensor> {
        crate::nn::ops::shortconv_ring_rows(
            b,
            x,
            c,
            pool,
            &self.layout.table,
            row_slots,
            row_phases,
        )
    }
}

/// The [`Shift`] window's step as tensor ops, the reference its kernels are
/// tested against: `x [channels]` shifted into `window [channels, k]`
/// (oldest..newest, `None` = zero), folded against `weight [channels, k]`.
/// Returns `(out [channels], next_window)`.
pub fn shift_step_reference(
    x: &Tensor,
    weight: &Tensor,
    window: Option<&Tensor>,
    channels: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let dt = x.dtype();
    let win = match window {
        Some(w) => w.clone(),
        None => Tensor::zeros((channels, k), dt, dev)?,
    };
    let shifted = win.narrow(1, 1, k - 1)?;
    let x_col = x.reshape((channels, 1))?;
    let new_win = Tensor::cat(&[&shifted, &x_col], 1)?;
    let folded = (new_win.contiguous()? * weight)?.sum(1)?;
    Ok((folded, new_win))
}

/// The window after the first `rows` positions of a chunk `x [seq,
/// channels]` against the carried window — what [`shift_prefill_reference`]
/// over `x[..rows]` leaves, without the fold: the newest `k` inputs,
/// oldest first, `[channels, k]`.
pub fn window_after(
    x: &Tensor,
    window: Option<&Tensor>,
    channels: usize,
    k: usize,
    rows: usize,
) -> Result<Tensor> {
    let dev = x.device();
    let dt = x.dtype();
    let seq = x.dim(0)?;
    if rows == 0 || rows > seq {
        lloom_tensor::bail!("window_after: {rows} rows of a {seq}-position chunk")
    }
    let x_t = x.narrow(0, 0, rows)?.t()?.contiguous()?;
    let history = match window {
        Some(w) => w.to_dtype(dt)?.narrow(1, 1, k - 1)?,
        None => Tensor::zeros((channels, k - 1), dt, dev)?,
    };
    let padded = Tensor::cat(&[&history, &x_t], 1)?;
    padded.narrow(1, rows - 1, k)?.contiguous()
}

/// [`shift_step_reference`] over a chunk `x [seq, channels]` against the
/// carried window: `(out [seq, channels], the window after the chunk)`,
/// position by position what the steps give.
pub fn shift_prefill_reference(
    x: &Tensor,
    weight: &Tensor,
    window: Option<&Tensor>,
    channels: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let dt = x.dtype();
    let seq = x.dim(0)?;
    let x_t = x.t()?.contiguous()?;
    let history = match window {
        Some(w) => w.to_dtype(dt)?.narrow(1, 1, k - 1)?,
        None => Tensor::zeros((channels, k - 1), dt, dev)?,
    };
    let padded = Tensor::cat(&[&history, &x_t], 1)?;
    let w = weight.to_dtype(dt)?;
    let mut acc: Option<Tensor> = None;
    for tap in 0..k {
        let cols = padded.narrow(1, tap, seq)?;
        let term = cols.broadcast_mul(&w.narrow(1, tap, 1)?)?;
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    let out = acc.expect("k >= 1").t()?.contiguous()?;
    let new_window = padded.narrow(1, seq - 1, k)?.contiguous()?;
    Ok((out, new_window))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::Device;

    /// Slot `s` at phase `p` carries the tap for the input `(p − s) mod width`
    /// back, zero past the window — pinned at a width and phase where a naive
    /// rotation would differ.
    #[test]
    fn the_ring_table_pairs_each_slot_with_its_age() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(vec![1f32, 2., 3., 10., 20., 30.], (2, 3), &dev)?;
        let conv = ShortConv::<Ring>::new(w, 5)?;
        // Phase 1: the newest in slot 1, then slot 0, then slot 4; 2 and 3 history.
        let got = conv.phase_weight(1).to_vec2::<f32>()?;
        assert_eq!(got[0], vec![2., 3., 0., 0., 1.]);
        assert_eq!(got[1], vec![20., 30., 0., 0., 10.]);
        assert_eq!(conv.table().dims(), &[5, 2, 5]);
        assert!(
            ShortConv::<Ring>::new(Tensor::zeros((2, 3), lloom_tensor::DType::F32, &dev)?, 2)
                .is_err()
        );
        Ok(())
    }

    /// Two inputs through one channel at k = 3, by hand: `w0·x_{t−2} +
    /// w1·x_{t−1} + w2·x_t`.
    #[test]
    fn the_shift_reference_folds_the_causal_window() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(vec![0.5f32, 1.0, 2.0], (1, 3), &dev)?;
        let (o1, win1) = shift_step_reference(&Tensor::new(&[1f32], &dev)?, &w, None, 1, 3)?;
        assert_eq!(o1.to_vec1::<f32>()?, vec![2.0]);
        let (o2, _) = shift_step_reference(&Tensor::new(&[3f32], &dev)?, &w, Some(&win1), 1, 3)?;
        assert_eq!(o2.to_vec1::<f32>()?, vec![7.0]);
        Ok(())
    }

    /// The chunk equals its positions' steps from the same carried window,
    /// including the window it leaves behind.
    #[test]
    fn the_shift_prefill_reference_is_its_steps() {
        let dev = Device::Cpu;
        let (ch, k, seq) = (3usize, 4usize, 9usize);
        let w = Tensor::from_vec(
            (0..ch * k)
                .map(|i| (i as f32 * 0.3).cos())
                .collect::<Vec<_>>(),
            (ch, k),
            &dev,
        )
        .unwrap();
        let x = Tensor::from_vec(
            (0..seq * ch)
                .map(|i| (i as f32 * 0.17).sin())
                .collect::<Vec<_>>(),
            (seq, ch),
            &dev,
        )
        .unwrap();
        let w0 = Tensor::from_vec(
            (0..ch * k)
                .map(|i| (i as f32 * 0.5).sin())
                .collect::<Vec<_>>(),
            (ch, k),
            &dev,
        )
        .unwrap();
        // Chunk.
        let (out, win) = shift_prefill_reference(&x, &w, Some(&w0), ch, k).unwrap();
        // Steps.
        let mut cur = w0.clone();
        let mut rows = Vec::new();
        for t in 0..seq {
            let xt = x.narrow(0, t, 1).unwrap().squeeze(0).unwrap();
            let (o, nw) = shift_step_reference(&xt, &w, Some(&cur), ch, k).unwrap();
            rows.push(o);
            cur = nw;
        }
        let out_ref = Tensor::stack(&rows, 0).unwrap();
        let (a, b) = (
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            out_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
        let (wa, wb) = (
            win.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            cur.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert_eq!(wa, wb);
    }
}
