//! The wait behind the non-blocking token path.
//!
//! `DESIGN.md` §11.5g designs a decode step whose sampled token never comes
//! back to the host: `argmax()` already returns a **device** tensor and
//! `to_scalar` is the only thing that forces it host-side
//! (`candle-transformers/src/generation/mod.rs`), so removing the read removes
//! the *reason* for the write. What is left on the host is a decision -- has
//! the GPU published this step's token yet, and was it EOS -- and that decision
//! needs a wait.
//!
//! **The wait is the part that is not decided.** §11.5g reports
//! `surviving_count` = 3 and declines to pick:
//!
//! | mechanism | cpu/wall | note |
//! |---|---|---|
//! | bare spin | 0.9998 | a full core |
//! | `WFE`-moderated spin | same core cost | **596-2048x** fewer line reads |
//! | `nanosleep` | 0.3606 | wake latency at an 18 ms cadence **unmeasured** |
//!
//! §3.7h measured all three and the framing it inverted is the reason this is a
//! seam rather than a constant: **there is no cheap park.** A thread in `WFE`
//! with the exclusive monitor armed reads cpu/wall 0.9998 -- indistinguishable
//! from a bare spin -- and retires ~1 `WFE` per microsecond whether or not
//! anything is written. What `WFE` buys is *fewer reads of the watched line*,
//! not power. `FEAT_WFxT` reads 0 on this machine, so there is no
//! hardware-bounded wait either. The question was never gate-versus-poll; it is
//! **which poll**, and that question is open.
//!
//! So this module is the one place the three arms differ, and nothing above it
//! knows which is installed. Picking a default is its own argued decision
//! (§7.1a) on its own evidence, and the evidence is not taken: see
//! [`PollMechanism::default`].
//!
//! # Why a poll at all
//!
//! `DESIGN.md` §15.2 #10a: *"anything that polls for a condition owes an
//! explanation of why the producer cannot announce it"* -- a polled state is
//! usually a missing event, and optimising a poll has a ceiling.
//!
//! The explanation here is that **the producer structurally cannot announce.**
//! §3.7b establishes three ways (issue #32) that a compute kernel cannot signal
//! an `MTLEvent` or `MTLSharedEvent`: 856 `__metal_*` builtins carry no
//! event/signal/semaphore primitive, no MSL stdlib header declares an event
//! type, and every compile probe for one fails. `[[event]]` *appears* to
//! compile and binds as an ordinary buffer -- silently ignored. Metal's
//! GPU-driven story covers "what work exists" and never "when a separate
//! submission is released".
//!
//! What *is* available is the reverse direction, and it is measured: §3.7g
//! (#191) reads **1998-2000 of 2000** distinct mid-flight values when the host
//! polls a kernel-written buffer, on 6 of 6 arms. The converse -- a kernel
//! observing host writes -- sees **at most one per launch**, so this seam may
//! not be inverted into a host-fed resident kernel.
//!
//! This is therefore a missing *primitive*, not a missing event, which is the
//! discharge §15.2 #10a asks for.

/// How the host waits for the GPU to publish a token.
///
/// The three arms §11.5g leaves surviving. They differ only here; the ring, the
/// buffer and the protocol are identical behind all three.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PollMechanism {
    /// Read the watched line in a tight loop.
    ///
    /// cpu/wall 0.9998 (§3.7h) -- a full core for the duration of the wait.
    Spin,
    /// Spin, but yield the pipeline between reads.
    ///
    /// On aarch64 this is the `WFE`-moderated arm: §3.7h measures the same
    /// core cost as [`Self::Spin`] and **596-2048x fewer reads of the watched
    /// line**. It is not a clock gate and does not park.
    ///
    /// `WFE` proper needs the exclusive monitor armed by a `LDXR` on the
    /// watched address, which is not expressible over a `&AtomicU32` in safe
    /// Rust; `core::hint::spin_loop()` emits the same `YIELD`/`WFE`-family hint
    /// the arm is named for and is what a portable build can reach. The
    /// distinction is recorded rather than papered over: this arm is
    /// *hint-moderated*, and whether it reproduces §3.7h's line-read reduction
    /// through this spelling is **unmeasured**.
    SpinHint,
    /// Sleep for a fixed interval between reads.
    ///
    /// cpu/wall 0.3606 (§3.7h) -- the only arm that genuinely gates. Its wake
    /// latency at an 18 ms cadence is **unmeasured**, which is why it is not
    /// the default.
    Sleep(std::time::Duration),
}

impl Default for PollMechanism {
    /// [`PollMechanism::Spin`].
    ///
    /// **This is not a measured choice and must not be read as one.** §11.5g
    /// reports the mechanism as `~ underdetermined` with `surviving_count` = 3,
    /// and `CONTRIBUTING.md` §7.1a's discipline is that a default is not
    /// flipped -- or fixed -- without its own argued decision on its own
    /// evidence. That evidence is owed, not taken.
    ///
    /// `Spin` is here because it is the arm whose *cost* is known exactly
    /// (cpu/wall 0.9998, §3.7h) and whose behaviour has no unmeasured term:
    /// `Sleep`'s wake latency at an 18 ms cadence is unmeasured, and
    /// `SpinHint`'s line-read reduction through this spelling is unmeasured.
    /// Choosing the arm with no unknowns is the conservative reading, not the
    /// fast one -- `Spin` is the arm that costs the most CPU.
    fn default() -> Self {
        Self::Spin
    }
}

impl PollMechanism {
    /// One iteration's wait. Called between reads of the watched word.
    #[inline]
    pub fn wait(&self) {
        match self {
            Self::Spin => {}
            Self::SpinHint => std::hint::spin_loop(),
            Self::Sleep(d) => std::thread::sleep(*d),
        }
    }

    /// The arm's name, for a `RESULT` line.
    ///
    /// Read back through an exhaustive `match` so a new arm is a compile error
    /// rather than a silently mis-rendered one. §7.1a-i measured the
    /// alternative: `AttnImpl` was rendered by a two-armed conditional over a
    /// three-armed axis, so `--attn flash` reported `AttnImpl=Generic` on every
    /// `RESULT` line since #116 -- and a *wrong* arm compares equal to every
    /// run of that arm, where an absent one renders `UNRECORDED` and compares
    /// equal to nothing.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Spin => "spin",
            Self::SpinHint => "spin-hint",
            Self::Sleep(_) => "sleep",
        }
    }
}

impl std::str::FromStr for PollMechanism {
    type Err = String;

    /// Parses `spin`, `spin-hint`, `sleep` and `sleep:<micros>`.
    ///
    /// An unrecognised value is an error rather than a fallback to the default:
    /// §2.4's rule after #69's vacuous determinism run is to make an invalid
    /// setting fail loudly, because a silent fallback produces a run that
    /// reports one arm and executes another.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "spin" => Ok(Self::Spin),
            "spin-hint" => Ok(Self::SpinHint),
            "sleep" => Ok(Self::Sleep(std::time::Duration::from_micros(50))),
            other => match other.strip_prefix("sleep:") {
                Some(us) => us
                    .parse::<u64>()
                    .map(|us| Self::Sleep(std::time::Duration::from_micros(us)))
                    .map_err(|e| format!("poll mechanism {other:?}: bad microseconds: {e}")),
                None => Err(format!(
                    "unknown poll mechanism {other:?} \
                     (expected spin, spin-hint, sleep, or sleep:<micros>)"
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parses_every_arm() {
        assert_eq!(
            PollMechanism::from_str("spin").unwrap(),
            PollMechanism::Spin
        );
        assert_eq!(
            PollMechanism::from_str("spin-hint").unwrap(),
            PollMechanism::SpinHint
        );
        assert_eq!(
            PollMechanism::from_str("sleep:7").unwrap(),
            PollMechanism::Sleep(std::time::Duration::from_micros(7))
        );
    }

    /// An invalid setting must fail rather than fall back (§2.4, after #69).
    ///
    /// The control is the arm above: if `from_str` accepted everything this
    /// test would pass while `parses_every_arm` also passed, so the pair is
    /// what makes either readable.
    #[test]
    fn rejects_unknown_arm() {
        assert!(PollMechanism::from_str("wfe").is_err());
        assert!(PollMechanism::from_str("").is_err());
        assert!(PollMechanism::from_str("sleep:").is_err());
        assert!(PollMechanism::from_str("sleep:-1").is_err());
    }

    /// Every arm renders a distinct name, and the `match` in `name` is
    /// exhaustive by construction. §7.1a-i's defect was a *collision* between
    /// two arms' rendered names, so distinctness is the property to assert.
    #[test]
    fn every_arm_renders_distinctly() {
        let arms = [
            PollMechanism::Spin,
            PollMechanism::SpinHint,
            PollMechanism::Sleep(std::time::Duration::from_micros(50)),
        ];
        let names: std::collections::HashSet<_> = arms.iter().map(|a| a.name()).collect();
        assert_eq!(names.len(), arms.len(), "two arms render the same name");
    }

    /// The default is `Spin` and is deliberately not measured (§11.5g:
    /// `surviving_count` = 3). Pinned so that flipping it is a visible edit to
    /// a test that says why, rather than a one-character change.
    #[test]
    fn default_is_the_unmeasured_conservative_arm() {
        assert_eq!(PollMechanism::default(), PollMechanism::Spin);
    }
}
