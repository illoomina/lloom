//! `Ring`: frames of one shape over one backing tensor, allocated once and
//! rotated through by index (`profiles-and-memory-design.md` §4.4, L3).
//!
//! A ring is per-sequence state that outlives every pass: the turn
//! checkpoints' conv tails, the host-fed rows, a staging window. What it
//! guarantees is the allocation shape — one request for the whole ring at
//! the first use of a sequence, none per pass or per frame — so a served pass
//! asks the pools for nothing a plan did not serve (`pass-plan-design.md`
//! §18). Which frame a pass writes is the owner's rotating index, kept beside
//! the state it describes (`DESIGN.md` §10.2c), not here.
//!
//! On the ledger as a capability: a sequence's state is never reduced under
//! pressure, only released with the sequence (`ledger.rs`).

use crate::nn::zeros as zs;
use lloom_tensor::{DType, Device, Result, Shape, Tensor};
#[cfg(feature = "metal")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Ring {
    /// `[frames, frame_shape…]`, contiguous, so every frame is a contiguous
    /// view a kernel or a `slice_set` can take as it is.
    backing: Tensor,
    frames: usize,
    frame: Shape,
    /// Held, never read: the ledger keeps its tenants as `Weak`
    /// (`Ledger::register`), so this strong reference is what keeps the
    /// ring's row on the ledger for as long as the ring lives.
    #[cfg(feature = "metal")]
    _ledger: Option<Arc<lloom_tensor::kernels::metal::ledger::Gauge>>,
}

impl Ring {
    /// `frames` frames of `frame` each, zeroed at `site`, on `device`: one
    /// allocation, outside every plan (it outlives the passes that write it).
    /// Registered on the device's ledger as `name` when the device is Metal.
    pub fn allocate<S: Into<Shape>>(
        frames: usize,
        frame: S,
        dtype: DType,
        device: &Device,
        site: &'static str,
        name: &str,
    ) -> Result<Self> {
        let frame: Shape = frame.into();
        let mut dims = Vec::with_capacity(frame.rank() + 1);
        dims.push(frames.max(1));
        dims.extend_from_slice(frame.dims());
        #[cfg(feature = "metal")]
        let _declared = lloom_tensor::EscapeGuard::new();
        let backing = zs::zeros(site, dims, dtype, device)?;
        let bytes = backing.elem_count() * dtype.size_in_bytes();
        #[cfg(feature = "metal")]
        let ledger = match device {
            Device::Metal(m) => {
                use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
                let gauge = Gauge::new(
                    name,
                    Claim {
                        floor: bytes,
                        want: bytes,
                        unit: bytes / frames.max(1),
                        class: Class::Capability,
                    },
                    bytes,
                );
                let tenant: Arc<dyn Tenant> = gauge.clone();
                m.ledger().register(&tenant);
                Some(gauge)
            }
            _ => None,
        };
        #[cfg(not(feature = "metal"))]
        let _ = name;
        Ok(Self {
            backing,
            frames: frames.max(1),
            frame,
            #[cfg(feature = "metal")]
            _ledger: ledger,
        })
    }

    /// Whether this ring already has `frames` frames of `frame` at `dtype`:
    /// the owner reuses it across sequences and reallocates only on a change
    /// of geometry.
    pub fn fits<S: Into<Shape>>(&self, frames: usize, frame: S, dtype: DType) -> bool {
        self.frames == frames.max(1) && self.frame == frame.into() && self.backing.dtype() == dtype
    }

    /// Frame `i`, a contiguous view of the backing.
    pub fn frame(&self, i: usize) -> Result<Tensor> {
        self.backing.narrow(0, i % self.frames, 1)?.squeeze(0)
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn frame_shape(&self) -> &Shape {
        &self.frame
    }

    /// The ring's bytes on the device.
    pub fn bytes(&self) -> usize {
        self.backing.elem_count() * self.backing.dtype().size_in_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_contiguous_views_of_one_allocation() -> Result<()> {
        let dev = Device::Cpu;
        let ring = Ring::allocate(4, (2, 3), DType::F32, &dev, zs::LFM2_CONV_STATE, "t")?;
        assert_eq!(ring.frames(), 4);
        assert_eq!(ring.bytes(), 4 * 2 * 3 * 4);
        for i in 0..4 {
            let f = ring.frame(i)?;
            assert_eq!(f.dims(), [2, 3]);
            assert!(f.is_contiguous());
        }
        // A write lands in its frame alone.
        let ones = Tensor::ones((2, 3), DType::F32, &dev)?;
        ring.frame(1)?.slice_set(&ones, 0, 0)?;
        assert_eq!(ring.frame(1)?.sum_all()?.to_scalar::<f32>()?, 6.0);
        assert_eq!(ring.frame(0)?.sum_all()?.to_scalar::<f32>()?, 0.0);
        assert_eq!(
            ring.frame(5)?.sum_all()?.to_scalar::<f32>()?,
            6.0,
            "index wraps"
        );
        assert!(ring.fits(4, (2, 3), DType::F32));
        assert!(!ring.fits(4, (2, 4), DType::F32));
        assert!(!ring.fits(8, (2, 3), DType::F32));
        Ok(())
    }
}
