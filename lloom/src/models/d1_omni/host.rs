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
