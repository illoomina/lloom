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
    /// # This arm is NOT `WFE`, and the doc comment that said so was wrong
    ///
    /// **Corrected 2026-09-02 (#578), read off the emitted instructions.** This
    /// arm's text claimed `core::hint::spin_loop()` *"emits the same
    /// `YIELD`/`WFE`-family hint the arm is named for"*. It does not. On
    /// aarch64 `spin_loop()` lowers to **`isb`**:
    ///
    /// ```text
    /// __ZN9wfe_probe4hint17h74c9e1b2b133caefE:
    ///         isb
    ///         ret
    /// ```
    ///
    /// `isb` is an instruction-synchronization barrier, not a wait hint, and
    /// §3.7h prices it at **9.04 ns** -- the **most expensive** of the seven
    /// gadgets that survey measured, against `yield` at 0.013-0.014 ns and
    /// `dmb ish` at 0.755 ns. So this arm was the *costliest* per iteration
    /// while being named for the cheapest, and §11.5n's measured
    /// `spin`-vs-`spin-hint` **0.1 %** was read through `objc_msgSend` where a
    /// hint has nothing to moderate -- it did not test this claim either.
    ///
    /// The arm is kept, correctly described, because it is what a **portable**
    /// build reaches: [`Self::Wfe`] is aarch64-only. What is removed is the
    /// claim that it reproduces §3.7h's line-read reduction.
    SpinHint,
    /// Wait on the exclusive monitor: `LDXR` the watched word, then `WFE`.
    ///
    /// **This is the arm §3.7h actually measured**, and #578 is why it exists
    /// as its own variant rather than as a description of [`Self::SpinHint`].
    /// The monitor must be armed by a load-exclusive on the address being
    /// watched, so the wait takes that address -- which is what makes this a
    /// different signature from the other arms and why `wait` cannot express
    /// it (see [`PollMechanism::wait`]).
    ///
    /// §3.7h measures cpu/wall **0.9998** -- the same full core as a bare spin,
    /// because there is no cheap park on this part -- and **596-2048x fewer
    /// reads of the watched line**. That reduction is the property this arm is
    /// for: it is a coherence-traffic result, not a power or latency one, and
    /// §3.7h's own follow-up (§9.2m) measures a poller's cost to a concurrent
    /// **GPU** kernel at a null, so the win is against other **CPU** threads
    /// (§9.2l) and must not be quoted as a decode speedup.
    ///
    /// `WFE` also retires on its own roughly every ~1 us whether or not
    /// anything is written (§3.7h: 711-1325 ns per retirement across a 16x
    /// span), so this is a bounded spin rather than a gate, and a poll loop
    /// around it still needs its own exit condition.
    #[cfg(target_arch = "aarch64")]
    Wfe,
    /// Sleep for a fixed interval between reads.
    ///
    /// cpu/wall 0.3606 (§3.7h) -- the only arm that genuinely gates. Its wake
    /// latency at an 18 ms cadence is **unmeasured**, which is why it is not
    /// the token ring's default. It is the host fill producer's wait while no
    /// forward waits on it (`nn::decode::FeedWait::OnDemand`): a core that
    /// spins through the streamed tier's host-planned pass holds the GPU at
    /// low clocks (`expert-streaming.md` §20.14a).
    Sleep(std::time::Duration),
}

impl PollMechanism {
    /// [`Default::default`] as a constant: `Wfe` on aarch64, `SpinHint`
    /// elsewhere (the reasoning is on the `Default` impl).
    #[cfg(target_arch = "aarch64")]
    pub const DEFAULT: PollMechanism = PollMechanism::Wfe;
    #[cfg(not(target_arch = "aarch64"))]
    pub const DEFAULT: PollMechanism = PollMechanism::SpinHint;
}

impl Default for PollMechanism {
    /// [`PollMechanism::Wfe`] on aarch64, [`PollMechanism::SpinHint`] elsewhere.
    ///
    /// # Why this changed, and what it does and does not claim (#578)
    ///
    /// **The previous default was `Spin`, whose `wait` was an empty body** --
    /// so the default arm had no wait at all and the loop ran `peek -> peek ->
    /// peek` at **~283 000 iterations per token**, one every ~62 ns, against a
    /// token boundary at ~55 Hz. That is not a choice between measured arms; it
    /// is a missing implementation, and `CONTRIBUTING.md` §7.1a's *"a default
    /// is not flipped without its own argued decision"* protects a **decision**
    /// rather than a defect.
    ///
    /// **What is argued here is the mechanism, not a speedup.** §3.7h measured
    /// `WFE` at **596-2048x fewer reads of the watched line** at the same core
    /// cost (cpu/wall 0.9998), and this is a load from a line the waiter owns
    /// -- exactly the shape that reduction applies to. §11.5n's null on
    /// `spin`-vs-`spin-hint` was measured through `signaledValue`, an
    /// `objc_msgSend` where *"a hint has nothing to moderate"*, so it does not
    /// bear on this channel.
    ///
    /// **What is NOT claimed: any throughput result.** §9.2m measures a
    /// poller's cost to a concurrent GPU kernel at a **null** (-0.07 %,
    /// [-1.26, +1.02]), so the read reduction is worth what it is worth against
    /// other **CPU** threads (§9.2l) and buys nothing against the GPU. #578
    /// fixes a defect; the A/B remains #522's to take.
    ///
    /// **`surviving_count` = 3 is not resolved by this.** §11.5g's question is
    /// which arm is *better* and that measurement is still not taken. What
    /// changed is that the default is now an arm that actually waits.
    fn default() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Wfe
        }
        // `WFE` is aarch64-only, so a portable build takes the hint arm --
        // which at least yields, unlike the empty body this replaced.
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::SpinHint
        }
    }
}

impl PollMechanism {
    /// One iteration's wait, for a caller that has no address to watch.
    ///
    /// **[`Self::Wfe`] cannot be served here and degrades to a hint**, because
    /// `WFE` without an armed exclusive monitor is a wait for an event that may
    /// never be signalled. Arming it requires the watched address, which this
    /// signature does not carry -- [`Self::wait_on`] is the form that does, and
    /// it is what the poll loop calls. The degradation is explicit rather than
    /// silent for §2.4's reason: a run that reports one arm and executes
    /// another is the failure mode, so the two call sites are distinguishable
    /// and this one is documented as not serving the arm.
    #[inline]
    pub fn wait(&self) {
        match self {
            // Not a no-op any more (#578). An empty body makes the loop
            // `peek -> peek -> peek` at ~62 ns per iteration, which is a poll
            // ~290 000x faster than the ~55 Hz event it waits for.
            Self::Spin => std::hint::spin_loop(),
            Self::SpinHint => std::hint::spin_loop(),
            #[cfg(target_arch = "aarch64")]
            Self::Wfe => std::hint::spin_loop(),
            Self::Sleep(d) => std::thread::sleep(*d),
        }
    }

    /// One iteration's wait, watching `addr` (#578).
    ///
    /// This is the form a poll loop should call. [`Self::Wfe`] arms the
    /// exclusive monitor on `addr` with a `LDXR` and then executes `WFE`, which
    /// is the sequence §3.7h measured; every other arm ignores the address and
    /// behaves exactly as [`Self::wait`].
    ///
    /// # The compare is what makes the wait safe
    ///
    /// `WFE` returns on an event, and a store to the watched line by another
    /// agent is one -- but so is a timeout, an interrupt, or an unrelated
    /// `SEV`. So this is a *bounded* wait and never a substitute for the
    /// caller's own re-check: the loop must re-read and re-test after it
    /// returns. It also compares the loaded word against `expect` and skips the
    /// `WFE` when they already differ, which closes the race where the value is
    /// published between the caller's read and the monitor being armed --
    /// without that compare a poller can sleep on an event that has already
    /// happened.
    ///
    /// # Safety
    ///
    /// `addr` must be a readable, correctly aligned address that stays mapped
    /// for the call. It is read once, non-atomically and by the hardware's
    /// exclusive load; the value may be stale or concurrently written, which is
    /// the point.
    #[inline]
    pub unsafe fn wait_on(&self, addr: *const u32, expect: u32) {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Wfe => {
                // SAFETY: the caller guarantees `addr` is readable and mapped.
                // `ldxr` arms the local monitor on that line; `wfe` then waits
                // for an event, which a store by the GPU to the line raises.
                // The branch skips the wait when the word already differs from
                // `expect`, so a value published before the monitor was armed
                // does not cost a full WFE quantum. `clrex` releases the
                // monitor on that path so no stale reservation is left behind.
                unsafe {
                    let v: u32;
                    core::arch::asm!(
                        "ldxr {v:w}, [{addr}]",
                        "cmp {v:w}, {expect:w}",
                        "b.ne 2f",
                        "wfe",
                        "b 3f",
                        "2:",
                        "clrex",
                        "3:",
                        addr = in(reg) addr,
                        expect = in(reg) expect,
                        v = out(reg) v,
                        options(nostack),
                    );
                    let _ = v;
                }
            }
            other => other.wait(),
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
            #[cfg(target_arch = "aarch64")]
            Self::Wfe => "wfe",
            Self::Sleep(_) => "sleep",
        }
    }
}

impl std::fmt::Display for PollMechanism {
    /// The spelling [`FromStr`](std::str::FromStr) reads back (`sleep:<µs>`
    /// keeps its duration, where [`Self::name`] renders the arm alone).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sleep(d) => write!(f, "sleep:{}", d.as_micros()),
            other => f.write_str(other.name()),
        }
    }
}

lloom_tensor::serde_via_str!(PollMechanism);

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
            // Accepted only where it exists. On another architecture this falls
            // through to the error below rather than silently selecting a
            // different arm -- §2.4: a run that reports one arm and executes
            // another is the failure this rejects loudly.
            #[cfg(target_arch = "aarch64")]
            "wfe" => Ok(Self::Wfe),
            "sleep" => Ok(Self::Sleep(std::time::Duration::from_micros(50))),
            other => match other.strip_prefix("sleep:") {
                Some(us) => us
                    .parse::<u64>()
                    .map(|us| Self::Sleep(std::time::Duration::from_micros(us)))
                    .map_err(|e| format!("poll mechanism {other:?}: bad microseconds: {e}")),
                None => Err(format!(
                    "unknown poll mechanism {other:?} \
                     (expected spin, spin-hint, wfe on aarch64, sleep, or \
                      sleep:<micros>)"
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
        // `wfe` used to be here as an unknown arm; #578 makes it a real one on
        // aarch64, so the negative case moves to a name that is still not an
        // arm rather than being deleted.
        assert!(PollMechanism::from_str("wait").is_err());
        assert!(PollMechanism::from_str("").is_err());
        assert!(PollMechanism::from_str("sleep:").is_err());
        assert!(PollMechanism::from_str("sleep:-1").is_err());
    }

    /// `wfe` parses where it exists and is refused where it does not (#578).
    ///
    /// The refusal half is the one that matters: selecting a different arm for
    /// a name the platform cannot serve is §7.1a-i's wrong-arm shape, where a
    /// run reports one arm and executes another.
    #[test]
    fn wfe_parses_only_where_it_exists() {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(PollMechanism::from_str("wfe").unwrap(), PollMechanism::Wfe);
        #[cfg(not(target_arch = "aarch64"))]
        assert!(PollMechanism::from_str("wfe").is_err());
    }

    /// Every arm renders a distinct name, and the `match` in `name` is
    /// exhaustive by construction. §7.1a-i's defect was a *collision* between
    /// two arms' rendered names, so distinctness is the property to assert.
    #[test]
    fn every_arm_renders_distinctly() {
        let arms = [
            PollMechanism::Spin,
            PollMechanism::SpinHint,
            #[cfg(target_arch = "aarch64")]
            PollMechanism::Wfe,
            PollMechanism::Sleep(std::time::Duration::from_micros(50)),
        ];
        let names: std::collections::HashSet<_> = arms.iter().map(|a| a.name()).collect();
        assert_eq!(names.len(), arms.len(), "two arms render the same name");
    }

    /// The default waits (#578).
    ///
    /// **This replaces `default_is_the_unmeasured_conservative_arm`**, which
    /// pinned `Spin` on the reasoning that it was the arm with no unmeasured
    /// term. That reasoning was sound about the *choice* and wrong about the
    /// *code*: `Spin`'s `wait` was an empty body, so the default was not a
    /// conservative arm but an absent one, spinning at ~283 000 iterations per
    /// token against a ~55 Hz event.
    ///
    /// Pinned the same way and for the same reason -- flipping it should be a
    /// visible edit to a test that says why -- but the property asserted is now
    /// the one that matters: **the default is an arm that waits.**
    #[test]
    fn the_default_is_an_arm_that_actually_waits() {
        let d = PollMechanism::default();
        assert_ne!(
            d,
            PollMechanism::Spin,
            "the default is `Spin`, whose wait was the empty body #578 removed"
        );
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
            d,
            PollMechanism::Wfe,
            "aarch64 has WFE and should default to it"
        );
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(d, PollMechanism::SpinHint);
    }

    /// No arm's `wait` is a no-op (#578), which is defect 2 stated as a test.
    ///
    /// A `wait` that compiles to nothing cannot be distinguished from one that
    /// waits by any assertion over values -- §15.1a's INEXPRESSIVE class -- so
    /// this pins the *source* the way the peek's gate does: the `match` in
    /// `wait` must have no empty arm.
    #[test]
    fn no_wait_arm_is_empty() {
        let src = include_str!("poll.rs");
        let start = src
            .find("pub fn wait(&self)")
            .expect("wait not found -- this gate is pinned to a name that moved");
        let body = &src[start..];
        let end = body
            .find("\n    /// One iteration's wait, watching")
            .expect("could not find the end of wait (the following item moved)");
        let body = &body[..end];
        assert!(
            !body.contains("=> {}"),
            "an arm of `wait` has an empty body -- that arm does not wait at all, \
             which is #578's defect 2 and no assertion over values can see it"
        );
        // The control: the needle must be findable, or the assertion above
        // passes over nothing (§15.1 #1).
        assert!(
            "Self::Spin => {}".contains("=> {}"),
            "the needle does not match its own example, so the gate is vacuous"
        );
    }
}
