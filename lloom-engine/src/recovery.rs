//! What an owner of failed work runs before it resets or reuses the state that
//! work may have touched (`batched-decode-design.md` §11.10, §11.11): the fence,
//! and the test of whether the failure touched any state at all.
//!
//! A forward that fails can return with its own work, and other sequences'
//! work, still queued on the device, so a reset before that work retires can
//! race it. The single request's lease, the generic causal owner and the
//! batch scheduler (static and continuous) all wait here first.

use lloom_tensor::{Device, Error, Result};

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

/// Whether a failure may have moved the state of the rows it ended. One that
/// did not — a refusal before any state moved — leaves each row's cached
/// prefix reusable; one that did is fenced and the rows reset.
///
/// Every failure is taken as touching: no model reports yet that a refusal
/// left its state untouched. This is the one place that report is read.
pub fn failure_touched_state(error: &Error) -> bool {
    let _ = error;
    true
}
