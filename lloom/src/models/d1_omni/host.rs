use lloom_tensor::Result;

// Preflight every host backing before the first reservation. Layout includes
// the isize byte ceiling that a checked usize element product alone misses.
pub(super) fn checked_host_capacity<T>(capacity: usize) -> Result<usize> {
    std::alloc::Layout::array::<T>(capacity)
        .map_err(|_| lloom_tensor::Error::Msg("omni host capacity exceeds layout limits".into()))?;
    Ok(capacity)
}
pub(super) fn prepared_vec<T>(capacity: usize) -> Result<Vec<T>> {
    checked_host_capacity::<T>(capacity)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|e| lloom_tensor::Error::Msg(format!("omni host reservation failed: {e}")))?;
    Ok(result)
}

/// Numeric host backing with optional shared-domain ownership. The vector is
/// deliberately declared before its charge: storage releases before accounting.
/// Used only for scalar f32/f64 tables here. Nested allocations in a generic T
/// would not be covered. Capacity bytes exclude allocator metadata; this is
/// not a physical RAM cap.
pub(super) struct HostVec<T> {
    values: Vec<T>,
    charge: Option<lloom_tensor::memory_admission::PhysicalCharge>,
}
impl<T: std::fmt::Debug> std::fmt::Debug for HostVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostVec")
            .field("values", &self.values)
            .field("charged_bytes", &self.charge.as_ref().map(|c| c.bytes()))
            .finish()
    }
}
impl<T> std::ops::Deref for HostVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> std::ops::DerefMut for HostVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T: Default + Clone> HostVec<T> {
    pub(super) fn zeroed(
        count: usize,
        context: Option<&lloom_tensor::memory_admission::AllocationContext>,
    ) -> Result<Self> {
        let layout = std::alloc::Layout::array::<T>(count).map_err(lloom_tensor::Error::wrap)?;
        // Reservation precedes every backing allocation, including construction
        // scratch. Failure leaves no live uncharged vector behind.
        let mut ticket = context
            .map(|context| context.take(layout.size()))
            .transpose()
            .map_err(lloom_tensor::Error::wrap)?;
        let mut values = prepared_vec(count)?;
        values.resize(count, T::default());
        let actual = std::alloc::Layout::array::<T>(values.capacity())
            .map_err(lloom_tensor::Error::wrap)?
            .size();
        let charge = if let Some(ticket) = ticket.as_mut() {
            match ticket.try_commit(actual) {
                Ok(charge) => Some(charge),
                Err(error) => {
                    drop(values);
                    return Err(lloom_tensor::Error::wrap(error));
                }
            }
        } else {
            None
        };
        Ok(Self { values, charge })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static INITIALIZED: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone)]
    struct ScalarProbe(u32);
    impl Default for ScalarProbe {
        fn default() -> Self {
            INITIALIZED.fetch_add(1, Ordering::SeqCst);
            Self(0)
        }
    }
    #[test]
    fn refused_ticket_precedes_numeric_initialization() -> Result<()> {
        use lloom_tensor::memory_admission::{process_account, AccountSnapshot};
        INITIALIZED.store(0, Ordering::SeqCst);
        let domain = process_account()
            .unadmitted_domain(Some(8))
            .map_err(lloom_tensor::Error::wrap)?;
        let pass = domain.reserve(8).map_err(lloom_tensor::Error::wrap)?;
        let context = pass.context();
        assert!(HostVec::<ScalarProbe>::zeroed(4, Some(&context)).is_err());
        assert_eq!(INITIALIZED.load(Ordering::SeqCst), 0);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 8,
                committed: 0
            }
        );
        let value = HostVec::<ScalarProbe>::zeroed(2, Some(&context))?;
        assert!(INITIALIZED.load(Ordering::SeqCst) > 0);
        assert_eq!(value[0].0, 0);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 8
            }
        );
        drop(pass);
        drop(value);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
        Ok(())
    }
}
