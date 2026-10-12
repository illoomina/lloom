//! Requested host allocation Layouts, NOT allocator rounding or resident RSS.
//! Arc quoting has an audited std implementation dependency, not a stable ABI.
//! build.rs admits exact audited official compiler/std/target entries only;
//! failure leaves legacy builds/APIs available and refuses owned quotes before
//! allocation. Absent rust-src uses the exact published-source audit table;
//! present divergent source refuses until reconciled. Recovery: restore an
//! audited matching distribution, or audit a new
//! compiler+target and update the explicit policy. Never guess on a new std.
//! No private Arc pointer is dereferenced, allocated or freed using this quote.
use std::{alloc::Layout, sync::atomic::AtomicUsize};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ManagedLayoutError {
    Overflow,
    Unsupported,
}
impl From<ManagedLayoutError> for crate::memory_admission::AllocationError {
    fn from(error: ManagedLayoutError) -> Self {
        match error {
            ManagedLayoutError::Overflow => Self::Overflow,
            ManagedLayoutError::Unsupported => Self::UnsupportedLayout,
        }
    }
}
pub(crate) fn arc_requested_layout<T>() -> Result<Layout, ManagedLayoutError> {
    if !cfg!(lloom_audited_arc_layout) {
        return Err(ManagedLayoutError::Unsupported);
    }
    // Audited repr(C, align(2)) ArcInner: Atomic<usize> twice, then T;
    // Atomic<usize> storage/alignment equals the stable AtomicUsize on this target.
    let counters = Layout::array::<AtomicUsize>(2).map_err(|_| ManagedLayoutError::Overflow)?;
    counters
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align())
        .map_err(|_| ManagedLayoutError::Overflow)
}
