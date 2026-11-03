//! What an owner of failed work runs before it resets or reuses the state that
//! work may have touched (`batched-decode-design.md` §11.10, §11.11): the fence,
//! and the test of whether the failure touched any state at all.
//!
//! A forward that fails can return with its own work, and other sequences'
//! work, still queued on the device, so a reset before that work retires can
//! race it. The single request's lease, the generic causal owner and the
//! batch scheduler (static and continuous) all wait here first.

use lloom_tensor::{Device, Error, Result, StateTouched};

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
/// Only a failure the step reported untouched ([`StateTouched::No`]) is
/// taken as untouched; an unreported one is touched. Ask of the error as the
/// step returned it: the generic `Result` context erases the report.
pub fn failure_touched_state(error: &Error) -> bool {
    error.state_touched() != Some(StateTouched::No)
}

#[cfg(test)]
mod tests {
    use super::failure_touched_state;
    use lloom_tensor::{Context as _, Error, StateTouched};

    fn refusal() -> Error {
        Error::Msg("ceiling".into())
    }

    #[test]
    fn only_a_failure_reported_untouched_keeps_the_state() {
        assert!(!failure_touched_state(
            &refusal().with_state_touched(StateTouched::No)
        ));
        assert!(failure_touched_state(
            &refusal().with_state_touched(StateTouched::Yes)
        ));
        assert!(
            failure_touched_state(&refusal()),
            "an unreported failure counts as touched"
        );
    }

    /// The typed wrappers keep the report; the generic `Result` context
    /// erases the inner type and with it the report, so a failure queried
    /// through it counts as touched. The query belongs before that wrap.
    #[test]
    fn a_report_reads_through_typed_context_and_not_through_an_erased_one() {
        let typed = refusal()
            .with_state_touched(StateTouched::No)
            .context("a static batch");
        assert!(!failure_touched_state(&typed));
        let erased = Err::<(), _>(refusal().with_state_touched(StateTouched::No))
            .context("a static batch")
            .unwrap_err();
        assert!(failure_touched_state(&erased));
    }
}
