//! The decode drivers' per-step and per-round clock (decision 123).
//!
//! Every clock read a decode step or a speculative round takes for its own
//! statistics goes through [`HotInstant`]. In a debug build and under the
//! `profile` feature it is `std::time::Instant`; in a release build without
//! `profile` it is a zero-sized value whose reads compile to nothing and whose
//! elapsed time is 0 — the same gate as the K=1 loop's step clock and
//! [`crate::PROFILE_INSTRUMENTS`]. A consumer that would print a per-step or
//! per-phase time checks [`crate::PROFILE_INSTRUMENTS`] and says the figure is
//! unavailable rather than printing the 0 (`DESIGN.md` §2.4: a null that reads
//! as a result).
//!
//! Not for functional time: a timeout, a hedge or a deadline is behaviour and
//! keeps `std::time::Instant`. Not for once-per-run brackets (the prefill, the
//! tail, a request's wall): those are the coarse metrics a release build keeps.

/// A per-step instant: real under [`crate::PROFILE_INSTRUMENTS`], nothing
/// otherwise.
#[derive(Clone, Copy, Debug)]
pub struct HotInstant {
    #[cfg(any(debug_assertions, feature = "profile"))]
    at: std::time::Instant,
}

impl HotInstant {
    /// Now, or nothing in a release build.
    #[inline(always)]
    pub fn now() -> Self {
        HotInstant {
            #[cfg(any(debug_assertions, feature = "profile"))]
            at: std::time::Instant::now(),
        }
    }

    /// Nanoseconds since [`Self::now`]; 0 in a release build.
    #[inline(always)]
    pub fn elapsed_ns(&self) -> u64 {
        #[cfg(any(debug_assertions, feature = "profile"))]
        {
            self.at.elapsed().as_nanos() as u64
        }
        #[cfg(not(any(debug_assertions, feature = "profile")))]
        {
            0
        }
    }

    /// Nanoseconds from `earlier` to this instant; 0 in a release build.
    #[inline(always)]
    pub fn ns_since(&self, earlier: &HotInstant) -> u64 {
        #[cfg(any(debug_assertions, feature = "profile"))]
        {
            self.at.saturating_duration_since(earlier.at).as_nanos() as u64
        }
        #[cfg(not(any(debug_assertions, feature = "profile")))]
        {
            let _ = earlier;
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests run with debug assertions (the `ci` profile keeps them), so
    /// the clock is real here and the release arm is a compile-time check.
    #[test]
    fn the_clock_is_real_under_the_instrumented_gate() {
        assert!(crate::PROFILE_INSTRUMENTS);
        let a = HotInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = HotInstant::now();
        assert!(a.elapsed_ns() >= 1_000_000);
        assert!(b.ns_since(&a) >= 1_000_000);
        assert_eq!(a.ns_since(&b), 0, "saturating, never negative");
    }
}
