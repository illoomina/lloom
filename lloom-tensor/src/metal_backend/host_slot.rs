//! Host-read slots: a small Shared, persistent buffer a kernel writes and the
//! host reads **in place** (`DESIGN.md` §15.2 #12a, #814 part 5).
//!
//! # Why a slot rather than a readback
//!
//! `to_cpu` reads its source directly only when that source is CPU-mapped,
//! which for a Private allocation means *under 16384 B* — and which bucket the
//! pool rounded a request into is a per-RUN outcome, not a property of the call
//! site (§15.2 #12a, #454: four consecutive runs of one arm read blit
//! 12 / 1 / 2 / 5 with no code difference). So a per-token readback of a small
//! value takes the fast path by the allocator's luck, and the blit it falls
//! back to is a second command buffer's submit-and-wait (0.175–0.188 ms,
//! §11.5c).
//!
//! A slot removes the question. The buffer is `StorageModeShared`, allocated
//! once for the session (`MetalDevice::new_persistent_shared`, which also wires
//! its pages), never returned to a pool and never a pass plan's — so
//! `contents()` is non-null for the whole run, by construction rather than by
//! rounding. The producing kernel writes it as its output; the host waits for
//! that work and reads the mapping. No copy dispatch, no staging allocation,
//! no blit.
//!
//! This generalizes the pattern `VerifyWords` (`models/lfm2.rs`) and the token
//! ring's `Mapped` already use: a mapped U32 buffer a kernel publishes into and
//! the host reads through a pointer held beside the tensor that owns it.
//!
//! # What a slot does NOT give
//!
//! **Ordering is the caller's.** A slot is one buffer reused every step, so the
//! step that writes it and the host read that follows are ordered by the
//! caller's own wait (a marker, or the queue drain a blocking arm already
//! performs). Writing it again before the host has read the previous value
//! overwrites what a reader may still be reading — the epoch-gated pattern the
//! tier's own decode buffers use (`write_decode_slots`). A slot's contract is
//! *write, wait, read*, in that order, per step.

use crate::kernels::metal::BufferHandle;
use crate::{DType, MetalStorage, Result, Shape, Storage, Tensor};

use super::MetalDevice;

/// A Shared, persistent buffer for one readback site, and the tensor a kernel
/// writes into it.
///
/// Held by the session (the model, the tier, the sampler) for the run, so
/// nothing about its mapping is a pool decision. [`Self::tensor`] is the
/// output to hand a kernel; [`Self::read`] is the host's read of what the
/// kernel wrote, with no copy.
#[derive(Debug)]
pub struct HostSlot {
    buffer: BufferHandle,
    device: MetalDevice,
    dtype: DType,
    /// Elements the buffer can hold at `dtype`.
    cap: usize,
}

impl HostSlot {
    /// A slot for at least `cap` elements of `dtype`.
    ///
    /// Allocated outside every pool and plan: `new_persistent_shared` takes it
    /// straight from the device and wires its pages, so the allocation is not
    /// an ordinal any arena could serve and not a buffer any pool can recycle.
    pub fn new(device: &MetalDevice, cap: usize, dtype: DType) -> Result<Self> {
        let bytes = (cap * dtype.size_in_bytes()).max(1);
        // Outside any pass plan: the value is the host's, never a later
        // dispatch's ordinal (§9.2aa).
        let _escape = crate::EscapeGuard::new();
        let buffer = device.new_persistent_shared(bytes)?;
        if buffer.contents().is_null() {
            crate::bail!("HostSlot: a {bytes}-byte Shared allocation is not CPU-mapped")
        }
        Ok(Self {
            buffer: buffer.clone(),
            device: device.clone(),
            dtype,
            cap,
        })
    }

    /// Grow to at least `cap` elements, keeping the slot's identity otherwise.
    /// A no-op when it already fits, so the steady state allocates nothing.
    pub fn ensure(&mut self, cap: usize) -> Result<()> {
        if cap <= self.cap {
            return Ok(());
        }
        *self = Self::new(&self.device, cap, self.dtype)?;
        Ok(())
    }

    /// Whether `t`'s bytes are this slot's: the kernel that produced `t` wrote
    /// the slot, so a read of the slot reads `t`.
    pub fn holds(&self, t: &Tensor) -> bool {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Metal(m) => {
                layout.start_offset() == 0
                    && crate::kernels::metal::BufferHandle::ptr_eq(m.buffer_handle(), &self.buffer)
            }
            _ => false,
        }
    }

    /// Elements the slot holds.
    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The slot as a tensor of `shape` — the output to hand the producing
    /// kernel. Errors when `shape` does not fit the slot.
    ///
    /// Every call builds a fresh `Tensor` over the SAME buffer, which is the
    /// point: the bytes are the slot's, so two tensors of one slot alias by
    /// construction. A caller that needs two live values needs two slots.
    pub fn tensor<S: Into<Shape>>(&self, shape: S) -> Result<Tensor> {
        let shape: Shape = shape.into();
        let n = shape.elem_count();
        if n > self.cap {
            crate::bail!(
                "HostSlot::tensor: {n} elements of {:?} in a slot of {}",
                self.dtype,
                self.cap
            )
        }
        let storage = MetalStorage::new(self.buffer.clone(), self.device.clone(), n, self.dtype);
        Ok(Tensor::from_storage(
            Storage::Metal(storage),
            shape,
            crate::op::BackpropOp::none(),
            false,
        ))
    }

    /// The slot's first `n` elements, read in place.
    ///
    /// **The caller must already have waited** for the work that wrote them: a
    /// marker ([`Self::read_after`]), or the queue drain a blocking arm
    /// performs anyway. This function does not wait, and reading unwritten
    /// bytes is not an error it can detect.
    pub fn read_unchecked<T: Clone + crate::WithDType>(&self, n: usize) -> Result<Vec<T>> {
        if T::DTYPE != self.dtype {
            crate::bail!(
                "HostSlot::read: slot is {:?}, asked as {:?}",
                self.dtype,
                T::DTYPE
            )
        }
        if n > self.cap {
            crate::bail!("HostSlot::read: {n} elements from a slot of {}", self.cap)
        }
        let ptr = std::hint::black_box(self.buffer.contents()) as *const T;
        if ptr.is_null() {
            crate::bail!("HostSlot::read: the slot lost its CPU mapping")
        }
        super::TO_CPU_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::note_readback("slot");
        // SAFETY: the mapping is non-null and holds `cap` elements of
        // `dtype`; the caller has waited for the GPU work that wrote them.
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    /// Wait for `marker` (the command buffer that wrote the slot), then read
    /// the first `n` elements in place.
    pub fn read_after<T: Clone + crate::WithDType>(
        &self,
        marker: &crate::kernels::metal::CommandMarker,
        n: usize,
    ) -> Result<Vec<T>> {
        self.device.wait_marker(marker)?;
        self.read_unchecked(n)
    }

    /// Commit the open command buffer and read the first `n` elements: the
    /// blocking arm's shape, where the step's work is what the host waits for.
    pub fn read_committed<T: Clone + crate::WithDType>(&self, n: usize) -> Result<Vec<T>> {
        let marker = self.device.flush_marker()?;
        self.read_after(&marker, n)
    }
}
