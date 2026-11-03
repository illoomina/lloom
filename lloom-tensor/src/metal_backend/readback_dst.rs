//! A host readback's own destination, and the cache a per-step reader keeps
//! it in (`DESIGN.md` §11.5z-ix item 2).

use super::{stage_readback, stage_readback_into, MetalDevice, ReadbackElement};
use crate::kernels::metal::BufferHandle;
use crate::Result;

/// A readback destination one reader owns: a persistent Shared buffer that
/// only [`stage_readback_into`] writes and only that stage's
/// [`super::StagedReadback::finish`] reads.
///
/// It is not `Clone` and hands out no buffer, so no other writer can be
/// bound to it, and a stage borrows it exclusively until `finish`.
///
/// `in_flight` is its custody of the GPU copy, which outlives the borrow: a
/// stage sets it before encoding, and only that stage's successful `finish`
/// (the copy's recording waited on, its bytes read) clears it. A stage that
/// is dropped, fails, or is finished with an error leaves it set. A stage
/// into a destination still in flight is admitted on the destination's own
/// queue only, where the encoder fences order the new copy after the
/// pending one; a tensor on another queue is refused whatever the custody,
/// because the buffer's residency and in-flight hold are its queue's.
///
/// ```no_run
/// # use lloom_tensor::{Device, Tensor};
/// # use lloom_tensor::metal_backend::stage_readback_into;
/// let dev = Device::new_metal(0).unwrap();
/// let Device::Metal(m) = &dev else { unreachable!() };
/// let t = Tensor::new(&[1u32, 2], &dev).unwrap();
/// let mut dst = m.new_readback_dst(8).unwrap();
/// let a: Vec<u32> = stage_readback_into(&t, &mut dst).unwrap().finish().unwrap();
/// let b: Vec<u32> = stage_readback_into(&t, &mut dst).unwrap().finish().unwrap();
/// assert_eq!(a, b);
/// drop(dst);
/// ```
///
/// A second stage while the first is pending does not compile:
///
/// ```compile_fail
/// # use lloom_tensor::{Device, Tensor};
/// # use lloom_tensor::metal_backend::stage_readback_into;
/// let dev = Device::new_metal(0).unwrap();
/// let Device::Metal(m) = &dev else { unreachable!() };
/// let t = Tensor::new(&[1u32, 2], &dev).unwrap();
/// let mut dst = m.new_readback_dst(8).unwrap();
/// let a = stage_readback_into(&t, &mut dst).unwrap();
/// let b = stage_readback_into(&t, &mut dst).unwrap();
/// let _: Vec<u32> = a.finish().unwrap();
/// let _: Vec<u32> = b.finish().unwrap();
/// ```
///
/// Nor does a clone of the destination to stage through:
///
/// ```compile_fail
/// # use lloom_tensor::{Device, Tensor};
/// # use lloom_tensor::metal_backend::stage_readback_into;
/// let dev = Device::new_metal(0).unwrap();
/// let Device::Metal(m) = &dev else { unreachable!() };
/// let t = Tensor::new(&[1u32, 2], &dev).unwrap();
/// let dst = m.new_readback_dst(8).unwrap();
/// let mut other = dst.clone();
/// let _: Vec<u32> = stage_readback_into(&t, &mut other).unwrap().finish().unwrap();
/// ```
///
/// Nor does dropping the destination before its stage finishes:
///
/// ```compile_fail
/// # use lloom_tensor::{Device, Tensor};
/// # use lloom_tensor::metal_backend::stage_readback_into;
/// let dev = Device::new_metal(0).unwrap();
/// let Device::Metal(m) = &dev else { unreachable!() };
/// let t = Tensor::new(&[1u32, 2], &dev).unwrap();
/// let mut dst = m.new_readback_dst(8).unwrap();
/// let a = stage_readback_into(&t, &mut dst).unwrap();
/// drop(dst);
/// let _: Vec<u32> = a.finish().unwrap();
/// ```
///
/// Nor does reaching its buffer, to bind it elsewhere:
///
/// ```compile_fail
/// # use lloom_tensor::Device;
/// let dev = Device::new_metal(0).unwrap();
/// let Device::Metal(m) = &dev else { unreachable!() };
/// let dst = m.new_readback_dst(8).unwrap();
/// let _ = &dst.buffer;
/// ```
pub struct ReadbackDst {
    buffer: BufferHandle,
    device: MetalDevice,
    in_flight: bool,
}

impl ReadbackDst {
    /// `buffer` must be a persistent Shared buffer of `device` that nothing
    /// else holds.
    pub(super) fn new(buffer: BufferHandle, device: MetalDevice) -> Result<Self> {
        // SAFETY: a mapping query on a live buffer; no byte is read.
        if unsafe { buffer.contents() }.is_null() {
            crate::bail!(
                "ReadbackDst: a {}-byte buffer with no CPU mapping",
                buffer.length()
            )
        }
        Ok(Self {
            buffer,
            device,
            in_flight: false,
        })
    }

    /// The bytes a stage may copy into it.
    pub fn capacity(&self) -> usize {
        self.buffer.length()
    }

    /// Whether a copy into it may still be pending: its last stage was not
    /// finished successfully.
    pub fn in_flight(&self) -> bool {
        self.in_flight
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub(super) fn buffer(&self) -> &BufferHandle {
        &self.buffer
    }

    /// Whether a tensor on `device` may stage into it: the same queue, so
    /// the same encoder fences order its copies.
    pub(super) fn serves(&self, device: &MetalDevice) -> bool {
        std::sync::Arc::ptr_eq(&self.device.commands, &device.commands)
    }

    /// Take custody for a copy about to be encoded; whether the previous one
    /// was still in flight.
    pub(super) fn claim(&mut self) -> bool {
        std::mem::replace(&mut self.in_flight, true)
    }

    /// The copy's recording completed and its bytes were read.
    pub(super) fn settle(&mut self) {
        self.in_flight = false;
    }
}

impl std::fmt::Debug for ReadbackDst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadbackDst")
            .field("capacity", &self.capacity())
            .field("in_flight", &self.in_flight)
            .finish()
    }
}

/// What a [`ReadbackDstCache`] has done since it was made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackDstStats {
    /// Checkouts that found nothing cached and allocated the first
    /// destination.
    pub created: u64,
    /// Checkouts that replaced the cached destination with a larger one.
    pub grown: u64,
    /// Checkouts served by the cached destination as it was.
    pub reused: u64,
    /// Requests that found the destination checked out by another reader.
    pub busy: u64,
    /// Requests for a tensor on another queue than the cached destination's.
    pub foreign: u64,
}

/// One reader's [`ReadbackDst`], kept across a readback it repeats every
/// step, so the step asks no pool for a destination.
///
/// Cold (nothing cached), idle (cached) or checked out. Only a checkout
/// installs a destination, so a returning lease never meets a newer one.
/// The lock is held to take the destination and to return it, never across
/// an allocation or the wait. A request that finds it checked out, or that
/// is for another queue, gets no lease, and [`Self::read`] then stages into
/// a buffer of its own, as an uncached readback does.
#[derive(Default)]
pub struct ReadbackDstCache {
    state: parking_lot::Mutex<CacheState>,
}

#[derive(Default)]
struct CacheState {
    dst: Option<ReadbackDst>,
    checked_out: bool,
    stats: ReadbackDstStats,
}

/// A destination checked out of a [`ReadbackDstCache`]; it goes back when
/// the lease drops, whether the readback succeeded, failed or unwound.
pub struct ReadbackDstLease<'c> {
    cache: &'c ReadbackDstCache,
    dst: Option<ReadbackDst>,
    made: Option<Made>,
}

#[derive(Clone, Copy)]
enum Made {
    Created,
    Grown,
}

impl ReadbackDstCache {
    /// Read the contiguous Metal tensor `t` back as `T` through the cached
    /// destination, or through a staging buffer of its own when the cache
    /// cannot lend it.
    pub fn read<T: ReadbackElement>(&self, t: &crate::Tensor) -> Result<Vec<T>> {
        let crate::Device::Metal(device) = t.device() else {
            crate::bail!("ReadbackDstCache::read: a Metal tensor")
        };
        let bytes = t.elem_count() * t.dtype().size_in_bytes();
        match self.lease(device, bytes)? {
            Some(mut lease) => stage_readback_into(t, lease.dst_mut())?.finish(),
            None => stage_readback(t)?.finish(),
        }
    }

    /// Check the destination out for a readback of `bytes` from `device`'s
    /// queue, allocating or growing it as needed. `None` when another reader
    /// has it, or when it belongs to another queue.
    pub fn lease(
        &self,
        device: &MetalDevice,
        bytes: usize,
    ) -> Result<Option<ReadbackDstLease<'_>>> {
        let taken = {
            let mut s = self.state.lock();
            if s.checked_out {
                s.stats.busy += 1;
                return Ok(None);
            }
            match &s.dst {
                Some(dst) if !dst.serves(device) => {
                    s.stats.foreign += 1;
                    return Ok(None);
                }
                Some(dst) if dst.capacity() >= bytes => s.stats.reused += 1,
                _ => {}
            }
            s.checked_out = true;
            s.dst.take()
        };
        let mut lease = ReadbackDstLease {
            cache: self,
            dst: taken,
            made: None,
        };
        let grown = match &lease.dst {
            Some(dst) if dst.capacity() >= bytes => return Ok(Some(lease)),
            Some(_) => Made::Grown,
            None => Made::Created,
        };
        // A failed allocation drops the lease, which returns what it holds.
        let fresh = device.new_readback_dst(bytes)?;
        lease.dst = Some(fresh);
        lease.made = Some(grown);
        Ok(Some(lease))
    }

    pub fn stats(&self) -> ReadbackDstStats {
        self.state.lock().stats
    }
}

impl std::fmt::Debug for ReadbackDstCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.state.lock();
        f.debug_struct("ReadbackDstCache")
            .field("dst", &s.dst)
            .field("checked_out", &s.checked_out)
            .field("stats", &s.stats)
            .finish()
    }
}

impl ReadbackDstLease<'_> {
    pub fn dst_mut(&mut self) -> &mut ReadbackDst {
        // A lease is handed out only once it holds a destination.
        self.dst.as_mut().expect("a lease holds its destination")
    }
}

impl Drop for ReadbackDstLease<'_> {
    fn drop(&mut self) {
        let mut s = self.cache.state.lock();
        s.checked_out = false;
        match self.made {
            Some(Made::Created) => s.stats.created += 1,
            Some(Made::Grown) => s.stats.grown += 1,
            None => {}
        }
        if let Some(dst) = self.dst.take() {
            debug_assert!(s.dst.is_none(), "only the checked-out lease installs");
            s.dst = Some(dst);
        }
    }
}
