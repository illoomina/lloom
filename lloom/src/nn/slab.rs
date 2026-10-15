//! `Slab`: a dense tensor along one axis with a live length
//! (`profiles-and-memory-design.md` §4.4, L3 `Slab<B, G: Grow>`).
//!
//! Positions append in place at the live length and a rewind is a length
//! decrement: the bytes above the length are unreachable (`DESIGN.md` §10.2a)
//! and the allocation is never touched by a rollback, so a kernel binds the
//! same buffer at the same base every token (§6.2b). The slab widens by one
//! event, [`Slab::grow_to`] — a fresh allocation at the new width with the
//! old one copied into its prefix — paid once per grow rather than per
//! token; a replay executor sees it as a stale step and re-records (§11.3y).
//! Whether a slab grows on its own ([`Grow::By`], a step at a time, the
//! zoo's and Qwen4-exp's) or only when its owner widens it before the pass
//! ([`Grow::Fixed`], LFM2's bucket, admitted first) is the rule, not the
//! type. One mechanism behind LFM2's `KvSlot` and `nn::kv_cache::Cache`.

use crate::nn::zeros as zs;
use lloom_tensor::{Result, Tensor};

/// How a slab meets an append past its capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grow {
    /// Refuse: the owner widens the slab before a pass appends (`grow_to`),
    /// so an append past the capacity ran outside a prepared pass.
    Fixed,
    /// Widen by multiples of the step until the append fits.
    By(usize),
}

#[derive(Debug, Clone)]
pub struct Slab {
    /// `[.., capacity, ..]` with `capacity` on `dim`; `None` until the first
    /// append, because the other dims (the batch) are known only then.
    data: Option<Tensor>,
    dim: usize,
    len: usize,
    capacity: usize,
    grow: Grow,
    /// The zeros site the backing is allocated at (`nn::zeros`).
    site: &'static str,
}

impl Slab {
    pub fn new(dim: usize, capacity: usize, grow: Grow, site: &'static str) -> Self {
        Self {
            data: None,
            dim,
            len: 0,
            capacity,
            grow,
            site,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Live positions.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions the backing holds (or will, at the first append).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn grow(&self) -> Grow {
        self.grow
    }

    /// The whole backing, `None` before the first append.
    pub fn data(&self) -> &Option<Tensor> {
        &self.data
    }

    /// Whether the backing exists.
    pub fn is_allocated(&self) -> bool {
        self.data.is_some()
    }

    /// The live prefix, a view; `None` before the first append.
    pub fn current(&self) -> Result<Option<Tensor>> {
        match self.data.as_ref() {
            None => Ok(None),
            Some(d) => Ok(Some(d.narrow(self.dim, 0, self.len)?)),
        }
    }

    /// Bytes the backing holds; zero before the first append.
    pub fn bytes(&self) -> usize {
        self.data
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0)
    }

    /// Forget every position and the backing: the next append allocates
    /// afresh (its batch may differ).
    pub fn reset(&mut self) {
        self.len = 0;
        self.data = None;
    }

    /// Keep the first `len` positions. A `len` past the live length is a
    /// no-op; the bytes stay, the length is what readers see.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }

    /// Discard the last `count` positions: a length decrement and nothing
    /// else. Past the start is the caller's error.
    pub fn rewind(&mut self, count: usize) -> Result<()> {
        if count > self.len {
            lloom_tensor::bail!(
                "slab rewind past the start: {count} > len {}; a rollback may only discard \
                 positions it appended",
                self.len
            )
        }
        self.len -= count;
        Ok(())
    }

    /// The width the backing will have at the first append. Refused once
    /// it exists: that is a grow (`grow_to`), not a setting.
    pub fn set_capacity(&mut self, capacity: usize) {
        debug_assert!(self.data.is_none(), "set_capacity on an allocated slab");
        if self.data.is_none() {
            self.capacity = capacity;
        }
    }

    /// Widen to `capacity` positions, every position where it was: before
    /// the first append a number; after it a fresh backing at the new width
    /// with the old copied into its prefix (one `copy2d` per plane). The
    /// buffer identity moves here and only here. At or below the current
    /// capacity: a no-op, never a shrink. The copy carries the rows the
    /// slab holds, so it runs ungated under an armed EOS gate: a grow can
    /// fall inside the forwards queued past an EOS
    /// (`sequence-reuse-design.md` §2.3e).
    pub fn grow_to(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.capacity {
            return Ok(());
        }
        if let Some(old) = self.data.as_ref() {
            let _ungated = old.device().eos_gate_suspended();
            let mut shape = old.dims().to_vec();
            shape[self.dim] = capacity;
            let grown = zs::zeros(self.site, shape, old.dtype(), old.device())?;
            grown.slice_set(old, self.dim, 0)?;
            self.data = Some(grown);
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Append `src` along `dim` at the live length and return the live
    /// prefix as a view. Allocates the backing at the first append; past
    /// the capacity, [`Grow::By`] widens first and [`Grow::Fixed`] refuses.
    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        if self.len + seq_len > self.capacity {
            match self.grow {
                Grow::By(step) if step > 0 => {
                    let short = self.len + seq_len - self.capacity;
                    let steps = short.div_ceil(step);
                    self.grow_to(self.capacity + steps * step)?;
                }
                _ => lloom_tensor::bail!(
                    "KV cache exhausted: {} + {seq_len} > capacity {} (a fixed slab); a pass \
                     widens the slab before it appends (`grow_to`), this append ran outside \
                     a prepared pass",
                    self.len,
                    self.capacity
                ),
            }
        }
        if self.data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.capacity;
            self.data = Some(zs::zeros(self.site, shape, src.dtype(), src.device())?);
        }
        let data = self.data.as_ref().expect("allocated above");
        data.slice_set(src, self.dim, self.len)?;
        self.len += seq_len;
        data.narrow(self.dim, 0, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{DType, Device};

    fn rows(n: usize, from: f32, dev: &Device) -> Tensor {
        let v: Vec<f32> = (0..n).map(|i| from + i as f32).collect();
        Tensor::from_vec(v, (1, n, 1), dev).unwrap()
    }

    #[test]
    fn a_fixed_slab_appends_in_place_rewinds_by_length_and_grows_by_copy() -> Result<()> {
        let dev = Device::Cpu;
        let mut s = Slab::new(1, 4, Grow::Fixed, zs::LFM2_KV_SLAB);
        assert!(!s.is_allocated());
        assert_eq!(s.bytes(), 0);
        let v = s.append(&rows(3, 1.0, &dev))?;
        assert_eq!(v.dims(), [1, 3, 1]);
        assert_eq!(s.bytes(), 4 * 4, "allocated at the capacity");
        assert!(
            s.append(&rows(2, 9.0, &dev)).is_err(),
            "past the capacity is refused"
        );
        s.rewind(1)?;
        assert_eq!(s.len(), 2);
        s.grow_to(8)?;
        assert_eq!(s.capacity(), 8);
        let v = s.append(&rows(2, 9.0, &dev))?;
        assert_eq!(
            v.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 2.0, 9.0, 10.0],
            "the prefix survived the grow, the rewound position was overwritten"
        );
        s.reset();
        assert!(!s.is_allocated());
        Ok(())
    }

    #[test]
    fn a_growing_slab_widens_in_steps() -> Result<()> {
        let dev = Device::Cpu;
        let mut s = Slab::new(1, 2, Grow::By(2), zs::LFM2_KV_SLAB);
        s.append(&rows(2, 1.0, &dev))?;
        let v = s.append(&rows(3, 3.0, &dev))?;
        assert_eq!(s.capacity(), 6, "two steps of 2 past the first 2");
        assert_eq!(
            v.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        s.truncate(1);
        assert_eq!(s.current()?.unwrap().dims(), [1, 1, 1]);
        assert_eq!(s.dtype_or(DType::F32), DType::F32);
        Ok(())
    }

    impl Slab {
        fn dtype_or(&self, d: DType) -> DType {
            self.data.as_ref().map(|t| t.dtype()).unwrap_or(d)
        }
    }
}
