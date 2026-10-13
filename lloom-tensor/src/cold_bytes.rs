//! Actual requested-layout host bytes for a cold plan shadow. No Vec growth,
//! post-allocation overcapacity refusal or producer-final cleanup is involved.
use crate::memory_admission::{
    AllocationContext, AllocationError, AllocationTicket, PhysicalCharge,
};
use std::alloc::{alloc_zeroed, Layout};

#[derive(Debug)]
pub(crate) enum ColdBytesError {
    Admission(AllocationError),
    Layout,
    Allocation,
}
/// Field order is custody: Box frees the exact byte allocation before the
/// charge wrapper frees its control allocation and refunds both requested terms.
/// The containing cold plan owns final Drop off engine/encoder/producer guards.
pub(crate) struct ColdBytes {
    bytes: Box<[u8]>,
    _charge: PhysicalCharge,
}
impl ColdBytes {
    pub(crate) fn requested_bytes(len: usize) -> Result<usize, ColdBytesError> {
        if len == 0 {
            return Err(ColdBytesError::Layout);
        }
        let layout = Layout::array::<u8>(len).map_err(|_| ColdBytesError::Layout)?;
        layout
            .size()
            .checked_add(
                AllocationTicket::requested_charge_control_bytes()
                    .map_err(ColdBytesError::Admission)?,
            )
            .ok_or(ColdBytesError::Layout)
    }
    pub(crate) fn zeroed_in(
        context: &AllocationContext,
        len: usize,
    ) -> Result<Self, ColdBytesError> {
        let quote = Self::requested_bytes(len)?;
        let mut ticket = context.take(quote).map_err(ColdBytesError::Admission)?;
        let layout = Layout::array::<u8>(len).map_err(|_| ColdBytesError::Layout)?;
        // SAFETY: checked nonzero Layout; null refusal allocates no storage.
        let raw = std::ptr::NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or(ColdBytesError::Allocation)?;
        // SAFETY: unique Global allocation for exactly [u8;len], all initialized
        // by alloc_zeroed. Slice Box owns the same pointer/Layout deallocation.
        let bytes = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(raw.as_ptr(), len)) };
        let charge = match ticket.try_commit_with_control(len) {
            Ok(charge) => charge,
            Err(error) => {
                drop(bytes); // free requested storage before unused promise refund
                return Err(ColdBytesError::Admission(error));
            }
        };
        Ok(Self {
            bytes,
            _charge: charge,
        })
    }
}
impl std::ops::Deref for ColdBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}
impl std::ops::DerefMut for ColdBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
