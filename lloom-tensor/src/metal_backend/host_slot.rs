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
        let bytes = cap
            .checked_mul(dtype.size_in_bytes())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| crate::Error::Msg("HostSlot: byte capacity overflow".into()))?
            .max(1);
        // Outside any pass plan: the value is the host's, never a later
        // dispatch's ordinal (§9.2aa).
        let _escape = crate::EscapeGuard::new();
        let buffer = device.new_persistent_shared(bytes)?;
        // SAFETY: the pointer is only compared with null.
        if unsafe { buffer.contents() }.is_null() {
            crate::bail!("HostSlot: a {bytes}-byte Shared allocation is not CPU-mapped")
        }
        let device = device.allocated_output_device(&buffer)?;
        Ok(Self {
            buffer: buffer.clone(),
            device,
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
        // SAFETY: the slot is session-owned Shared memory no pool recycles; the
        // read below follows the caller's wait (this function's precondition).
        let ptr = std::hint::black_box(unsafe { self.buffer.contents() }) as *const T;
        if ptr.is_null() {
            crate::bail!("HostSlot::read: the slot lost its CPU mapping")
        }
        super::TO_CPU_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::note_readback("slot");
        // SAFETY: the mapping is non-null and holds `cap` elements of
        // `dtype`; the caller has waited for the GPU work that wrote them.
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    /// The tensor `t`'s values, computed into this slot: the argmax's `[1]`
    /// u32 draw, so the host reads the id from a mapping that is the
    /// session's rather than from a buffer the pool rounded (#814 part 5).
    ///
    /// One `copy2d` dispatch, in the step's own command buffer — the same
    /// copy the ring's publish encodes. It is a dispatch rather than a
    /// blit so that it is ordered by the compute barrier chain when a pass
    /// plan is installed (`pass-plan-design.md` §9).
    pub fn write_from(&self, t: &Tensor) -> Result<()> {
        let (storage, layout) = t.storage_and_layout();
        let m = match &*storage {
            Storage::Metal(m) => m,
            _ => crate::bail!("HostSlot::write_from: a Metal tensor"),
        };
        if !layout.is_contiguous() {
            crate::bail!("HostSlot::write_from: a contiguous tensor")
        }
        if m.dtype != self.dtype {
            crate::bail!(
                "HostSlot::write_from: slot is {:?}, tensor {:?}",
                self.dtype,
                m.dtype
            )
        }
        let n = layout.shape().elem_count();
        if n > self.cap {
            crate::bail!(
                "HostSlot::write_from: {n} elements into a slot of {}",
                self.cap
            )
        }
        let elem = self.dtype.size_in_bytes();
        let kernel = match self.dtype {
            DType::U32 => crate::kernels::copy2d::U32,
            DType::F32 => crate::kernels::copy2d::FLOAT,
            DType::F16 => crate::kernels::copy2d::HALF,
            DType::BF16 => crate::kernels::copy2d::BFLOAT,
            dtype => crate::bail!("HostSlot::write_from: {dtype:?} is not wired"),
        };
        let encoder = self.device.command_encoder()?;
        // SAFETY: `m` is the caller's storage and the slot is session-owned
        // (binding authority, `mod.rs` head); the host reads the slot only after
        // waiting for this command buffer (write, wait, read; module docs).
        unsafe {
            crate::kernels::call_copy2d(
                &self.device.device,
                &encoder,
                &self.device.kernels,
                kernel,
                m.buffer(),
                &self.buffer,
                1,
                n,
                n,
                n,
                layout.start_offset() * elem,
                0,
            )
        }
        .map_err(crate::MetalError::from)?;
        Ok(())
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

    /// Store `v` into the first `n` u32 words from the host. For a slot a
    /// poller watches while its producer runs ([`Self::peek_u32`]): the
    /// caller writes only while no command that writes the slot is pending,
    /// so the producer's next write is the one the poller sees.
    pub fn fill_u32(&self, v: u32, n: usize) -> Result<()> {
        self.check_u32(n)?;
        // SAFETY: session-owned Shared memory; the host writes it only while no
        // command that writes the slot is pending (this function's precondition).
        let ptr = unsafe { self.buffer.contents() } as *mut u32;
        for i in 0..n {
            // SAFETY: a CPU-mapped Shared buffer of `cap >= n` u32 (checked).
            unsafe { ptr.add(i).write_volatile(v) }
        }
        Ok(())
    }

    /// The first `n` u32 words as the host sees them now. The caller owns
    /// ordering; words may individually be old or new during a GPU write.
    pub fn peek_u32(&self, n: usize, out: &mut Vec<u32>) -> Result<()> {
        peek_u32(&self.buffer, self.dtype, self.cap, n, out)
    }

    /// Fixed destination sibling; caller retains the same completion authority.
    pub(crate) fn peek_u32_into(&self, out: &mut [u32]) -> Result<()> {
        peek_u32_into(&self.buffer, self.dtype, self.cap, out)
    }

    /// Retain only future host-read ownership. No write, tensor, reset or
    /// native-buffer capability escapes; the physical allocation stays live
    /// until the final reader drops, independently of the producing slot.
    pub(crate) fn read_view(&self) -> HostReadView {
        HostReadView {
            buffer: self.buffer.clone(),
            dtype: self.dtype,
            cap: self.cap,
        }
    }

    fn check_u32(&self, n: usize) -> Result<()> {
        check_u32(&self.buffer, self.dtype, self.cap, n)
    }
}

/// A retained read owner, not a native-binding or emission-closed receipt.
#[derive(Clone)]
pub(crate) struct HostReadView {
    buffer: BufferHandle,
    dtype: DType,
    cap: usize,
}

impl HostReadView {
    pub(crate) fn peek_u32(&self, n: usize, out: &mut Vec<u32>) -> Result<()> {
        peek_u32(&self.buffer, self.dtype, self.cap, n, out)
    }

    pub(crate) fn peek_u32_into(&self, out: &mut [u32]) -> Result<()> {
        peek_u32_into(&self.buffer, self.dtype, self.cap, out)
    }
}

fn check_u32(buffer: &BufferHandle, dtype: DType, cap: usize, n: usize) -> Result<()> {
    if dtype != DType::U32 {
        crate::bail!("HostSlot: a u32 access to a {dtype:?} slot")
    }
    if n > cap || n.checked_mul(4).is_none_or(|bytes| bytes > buffer.length()) {
        crate::bail!("HostSlot: {n} words exceed logical or physical slot capacity {cap}")
    }
    // SAFETY: the pointer is only compared with null.
    if unsafe { buffer.contents() }.is_null() {
        crate::bail!("HostSlot: the slot lost its CPU mapping")
    }
    Ok(())
}

fn peek_u32(
    buffer: &BufferHandle,
    dtype: DType,
    cap: usize,
    n: usize,
    out: &mut Vec<u32>,
) -> Result<()> {
    check_u32(buffer, dtype, cap, n)?;
    out.resize(n, 0);
    // SAFETY: exact dtype/logical/physical/mapping checks above. Resize affects
    // only independently owned host output, not the retained native buffer.
    unsafe { copy_checked_u32(buffer, out) };
    Ok(())
}

fn peek_u32_into(buffer: &BufferHandle, dtype: DType, cap: usize, out: &mut [u32]) -> Result<()> {
    check_u32(buffer, dtype, cap, out.len())?;
    // SAFETY: same checks and retained buffer as the public Vec route.
    unsafe { copy_checked_u32(buffer, out) };
    Ok(())
}

/// Requires check_u32(buffer, dtype, cap, out.len()) before entry. The retained
/// Shared buffer and disjoint initialized host destination outlive this copy.
/// Completion interpretation stays with the caller; each word can be old/new.
unsafe fn copy_checked_u32(buffer: &BufferHandle, out: &mut [u32]) {
    let ptr = unsafe { buffer.contents() } as *const u32;
    for (i, word) in out.iter_mut().enumerate() {
        *word = unsafe { ptr.add(i).read_volatile() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_read_view_survives_slot_and_checks_actual_bytes() -> Result<()> {
        crate::isolation::claim("retained HostSlot read owner");
        let mut tuning = crate::tuning::Tuning::default();
        tuning.from_env().expect("valid environment");
        tuning.contract.profile = "deterministic".into();
        crate::tuning::install(tuning).expect("before device");
        let d = crate::Device::new_metal(0)?;
        {
            let encoder = d.as_metal_device()?.command_encoder()?;
            println!(
                "HOST_READ_VIEW_SETUP metal4={}",
                encoder.as_ref().command_buffer.is_metal4()
            );
        }
        let slot = HostSlot::new(d.as_metal_device()?, 1, DType::U32)?;
        slot.fill_u32(17, 1)?;
        let mut view = slot.read_view();
        drop(slot);
        let mut words = Vec::with_capacity(1);
        view.peek_u32(1, &mut words)?;
        assert_eq!(words, [17]);
        let mut fixed = [0xdecafbadu32; 1];
        view.peek_u32_into(&mut fixed)?;
        assert_eq!(fixed, [17], "fixed slice must share exact retained copy");
        view.peek_u32_into(&mut [])?;
        let mut too_long = [0xdecafbadu32; 2];
        assert!(view.peek_u32_into(&mut too_long).is_err());
        assert_eq!(too_long, [0xdecafbad; 2], "logical refusal modified output");
        let wrong_dtype = HostReadView {
            buffer: view.buffer.clone(),
            dtype: DType::F32,
            cap: 1,
        };
        assert!(wrong_dtype.peek_u32_into(&mut fixed).is_err());
        assert_eq!(fixed, [17], "dtype refusal modified output");
        let mut beyond_physical = vec![0xdecafbad; view.buffer.length() / 4 + 1];
        let advertised_large = HostReadView {
            buffer: view.buffer.clone(),
            dtype: DType::U32,
            cap: usize::MAX,
        };
        assert!(advertised_large
            .peek_u32_into(&mut beyond_physical)
            .is_err());
        assert!(beyond_physical.iter().all(|&v| v == 0xdecafbad));
        assert!(view.peek_u32(2, &mut words).is_err());
        // Corrupt only private logical metadata to establish that physical
        // bounds, not the advertised capacity alone, fence every reader.
        view.cap = usize::MAX;
        assert!(view.peek_u32(usize::MAX, &mut words).is_err());
        assert!(view
            .peek_u32(view.buffer.length() / 4 + 1, &mut words)
            .is_err());
        assert!(HostSlot::new(d.as_metal_device()?, usize::MAX, DType::U32).is_err());
        Ok(())
    }
}
