//! The fence an owner of failed work runs before it resets or reuses the state
//! that work may have touched (`batched-decode-design.md` §11.10).
//!
//! A forward that fails can return with its own work, and other sequences'
//! work, still queued on the device, so a reset before that work retires can
//! race it. The single request's lease, the generic causal owner and the
//! static batch scheduler all wait here first.

use lloom_tensor::{Device, Result};

/// Wait for the work registered on `device`'s queue so far: on Metal, a fresh
/// marker and the data of the prefix it closes (`wait_marker_prefix_data`);
/// elsewhere, a full synchronize. An error is the queue's own failure.
pub fn fence_failed_execution(device: &Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(device) = device {
        let marker = device.flush_marker()?;
        return device.wait_marker_prefix_data(&marker);
    }
    device.synchronize()
}
