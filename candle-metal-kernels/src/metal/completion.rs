//! How the host learns a command buffer has finished.
//!
//! `DESIGN.md` §11.5 has spent nine sections narrowing one round trip and never
//! removing it. §11.5j built the token ring's call site and recorded what it is
//! in its own words -- *"a hint before a blocking read, not a spin-until-published
//! loop"* -- because `PollMechanism::wait()` is called once per step and the flag
//! read that follows reaches `to_cpu`'s `flush_and_wait_current`. §11.5k measured
//! the consequence (**+0.018 %** against a null of **+0.187 %**) and §11.5l
//! measured why the three poll arms cannot be told apart: *"the arm as wired never
//! spins"*, so there is no burn for a park to save. Both name the same repair, and
//! it is the one thing neither built:
//!
//! > **replacing the blocking flag read with a genuinely non-blocking one** --
//! > the `candle-core` change §11.5j deliberately did not bundle, and the largest
//! > single candidate, since the seam as built narrows a round trip rather than
//! > removing it.
//!
//! This module is that seam. It is **the wait**, factored out of
//! [`super::commands::Commands`], so the blocking `waitUntilCompleted` becomes
//! one arm among several rather than the only thing the code can express.
//!
//! # Why this is a seam and not a replacement
//!
//! `synchronize()` stays selectable and stays the **default**. It is the
//! incumbent, every committed figure in `DESIGN.md` belongs to it, and §7.1a's
//! discipline is that a default flip is its own argued decision with a number in
//! hand -- which this module deliberately does not take. What changes is that the
//! alternative is now *expressible*, which is the precondition for ever taking
//! that number.
//!
//! # The primitive, verified rather than inherited
//!
//! §3.7b establishes that a compute **kernel** cannot signal an `MTLEvent`:
//! 856 `__metal_*` builtins carry no event/signal/semaphore primitive, no MSL
//! header declares an event type, and `[[event]]` binds as an ordinary buffer.
//! That is a fact about kernels and it is **not** a fact about command buffers --
//! §3.7b's own table says an `MTLSharedEvent` is signalled at a *"GPU
//! [command-buffer] boundary **or** CPU"*, which is a different actor.
//!
//! Issue #559's research input asserts the command-buffer route works and marks
//! itself external `$w`. Probed on this machine before anything was built here
//! (`measurements/probes/issue-559/shared_event.m`, output in
//! `measurements/issue-559-raw/`):
//!
//! | arm | reads |
//! |---|---|
//! | `newSharedEvent` | **OK** |
//! | signal observed with **no** `waitUntilCompleted` | **YES**, value 1 after 23.97 ms |
//! | **non-vacuity** -- same buffer, no signal encoded | value does **NOT** advance |
//! | **ordering** -- kernel's sentinel read on observing the signal, without waiting | **visible** (`0xABCDEF01`) |
//! | host-side `setSignaledValue` | **settable** |
//!
//! The non-vacuity arm is what makes the first row readable: a counter that was
//! never zero would report success for a signal that never fired. The ordering
//! arm is what makes the mechanism *sound* -- the signal implies the preceding
//! writes are visible, which is the property every caller of this seam relies on
//! and which §3.5 makes silent corruption rather than an error if it fails.
//!
//! # What the wait arms cost, measured on this channel
//!
//! §3.7h measured the CPU-side gadgets in the abstract and found **no cheap
//! park**: `WFE` reads cpu/wall 0.9998 -- a full core -- against a `nanosleep`
//! control at 0.3606, and what the hint buys is **596-2048x fewer reads of the
//! watched line** rather than power.
//!
//! **That line-read reduction does not reproduce through this channel, and the
//! reason is structural.** `signaledValue` is an `objc_msgSend` into the Metal
//! runtime, not a load from a line the waiter owns, so a hint has nothing to
//! moderate. Measured over a 16.3 ms wait, 8 interleaved reps per arm
//! (`measurements/probes/issue-559/poll_cost.m`):
//!
//! | arm | polls/ms | median wait (ms) |
//! |---|---|---|
//! | `spin` | **50 917** | 16.3100 |
//! | `spin-hint` | **50 855** | 16.3211 |
//! | `sleep:50` | **15.0** | 16.3493 |
//! | `sleep:200` | **3.9** | 16.4305 |
//!
//! `spin` and `spin-hint` are **0.1 % apart** -- indistinguishable, where §3.7h
//! separates the analogous CPU-side pair by three orders of magnitude on line
//! reads. So the hint arm is not carried here: it would be a third name for the
//! second arm, and §7.1a-i's defect is precisely an axis whose arms render
//! distinctly and behave identically.
//!
//! **And this answers a question §11.5g recorded as open.** That section reports
//! `nanosleep`'s *"wake latency at an 18 ms cadence unmeasured"*. Measured here it
//! is the `wait` column's excess over `spin`: **+0.039 ms at 50 us** and **+0.121
//! ms at 200 us**, against a 3400x and 13000x reduction in polls. That is a real
//! trade rather than a free win, and it is now a number.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLSharedEvent};

use super::{CommandBuffer, Device};
use crate::MetalKernelError;

/// How the host waits for a committed command buffer.
///
/// The arms differ **only** in [`Self::wait_for`]; encoding, committing, error
/// handling and pruning are identical behind all of them. That is what makes a
/// third arm a variant rather than a restructuring -- see the module docs and
/// [`Self::from_str_arm`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompletionWait {
    /// `waitUntilCompleted` -- block the calling thread in the Metal runtime.
    ///
    /// **The incumbent and the default.** Every canonical digest, every
    /// `.bench/` row and every figure in `DESIGN.md` §11.5 belongs to this arm.
    /// It is a Mach trap: the thread yields its quantum and the kernel wakes it,
    /// which is the cost issue #559 exists to make optional.
    #[default]
    Synchronize,
    /// Signal an `MTLSharedEvent` at the command-buffer boundary and poll
    /// `signaledValue` from the host without blocking.
    ///
    /// The host never enters the kernel to wait. Between reads it spins: §3.7h
    /// establishes there is no cheap park on this part (`FEAT_WFxT` reads 0, so
    /// there is no hardware-bounded wait), and the probe above establishes that
    /// a hint does not reduce reads on *this* channel because the read is an
    /// `objc_msgSend` rather than a load from a watched line.
    SharedEvent,
    /// As [`Self::SharedEvent`], but `nanosleep` for `micros` between reads.
    ///
    /// The arm that trades wake latency for polls: **3400x fewer polls for
    /// +0.039 ms** at 50 us, measured above. Carried because it is the only arm
    /// that genuinely yields the core, which is what a machine running other
    /// work wants and what §3.7h's 0.3606 control measures.
    SharedEventSleep {
        /// Microseconds to sleep between polls.
        micros: u64,
    },
}

impl CompletionWait {
    /// The arm's name, for a `RESULT` line.
    ///
    /// Rendered through an **exhaustive** `match` so a new arm is a compile
    /// error rather than a silently mis-rendered one. §7.1a-i measured the
    /// alternative: `AttnImpl` was rendered by a two-armed conditional over a
    /// three-armed axis, so `--attn flash` reported `AttnImpl=Generic` on every
    /// `RESULT` line since #116 -- and **a wrong arm compares equal to every run
    /// of that arm**, where an absent one compares equal to nothing.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Synchronize => "synchronize",
            Self::SharedEvent => "shared-event",
            Self::SharedEventSleep { .. } => "shared-event-sleep",
        }
    }

    /// Whether this arm needs an `MTLSharedEvent` allocated on the device.
    ///
    /// `Synchronize` does not, so the default path allocates nothing and the
    /// event is created lazily only when an async arm is selected. That keeps
    /// the incumbent byte-for-byte what it was.
    pub fn needs_event(&self) -> bool {
        match self {
            Self::Synchronize => false,
            Self::SharedEvent | Self::SharedEventSleep { .. } => true,
        }
    }

    /// Parses `synchronize`, `shared-event`, `shared-event-sleep` and
    /// `shared-event-sleep:<micros>`.
    ///
    /// An unrecognised value is an **error**, never a fallback to the default:
    /// §2.4's rule after #69's vacuous determinism run is that an invalid
    /// setting must fail loudly, because a silent fallback produces a run that
    /// reports one arm and executes another -- which is #500's whole finding in
    /// a different place.
    pub fn from_str_arm(s: &str) -> Result<Self, String> {
        match s {
            "synchronize" => Ok(Self::Synchronize),
            "shared-event" => Ok(Self::SharedEvent),
            "shared-event-sleep" => Ok(Self::SharedEventSleep { micros: 50 }),
            other => match other.strip_prefix("shared-event-sleep:") {
                Some(us) => us
                    .parse::<u64>()
                    .map(|micros| Self::SharedEventSleep { micros })
                    .map_err(|e| format!("completion wait {other:?}: bad microseconds: {e}")),
                None => Err(format!(
                    "unknown completion wait {other:?} (expected synchronize, \
                     shared-event, shared-event-sleep, or shared-event-sleep:<micros>)"
                )),
            },
        }
    }

    /// The arm selected by `CANDLE_METAL_COMPLETION_WAIT`, read **once**.
    ///
    /// Read through a `OnceLock` because this sits on the per-token path and a
    /// per-call `var()` is what #69's vacuous determinism run risks (§2.4). An
    /// unparseable value **panics** rather than falling back, for
    /// [`Self::from_str_arm`]'s reason: a run that silently executes the
    /// incumbent while its operator believes otherwise is the failure mode this
    /// whole section exists to avoid.
    pub fn from_env() -> Self {
        static ARM: std::sync::OnceLock<CompletionWait> = std::sync::OnceLock::new();
        *ARM.get_or_init(|| match std::env::var("CANDLE_METAL_COMPLETION_WAIT") {
            Ok(v) => Self::from_str_arm(&v)
                .unwrap_or_else(|e| panic!("CANDLE_METAL_COMPLETION_WAIT: {e}")),
            Err(_) => Self::default(),
        })
    }

    /// Encode this arm's signal onto `cb`, if it has one, returning the value
    /// the host should wait for.
    ///
    /// **Must be called with no open encoder on `cb`.** `encodeSignalEvent:value:`
    /// is documented *"may only be called if there is no current command encoder
    /// on the receiver"*, and the caller in `commands.rs` ends encoding before
    /// committing, which is the same point.
    ///
    /// Returns `None` for [`Self::Synchronize`], which encodes nothing and
    /// leaves the incumbent path untouched.
    pub fn encode_signal(&self, cb: &CommandBuffer, event: Option<&SharedEvent>) -> Option<u64> {
        if !self.needs_event() {
            return None;
        }
        let event = event?;
        let value = event.next_value();
        // SAFETY-adjacent note rather than an `unsafe` block: the objc2 binding
        // for `encodeSignalEvent:value:` is safe, and its one documented
        // precondition -- no open encoder -- is the caller's, discharged above.
        cb.as_ref()
            .encodeSignalEvent_value(ProtocolObject::from_ref(event.raw()), value);
        Some(value)
    }

    /// Wait until `cb`'s work is done, by this arm's mechanism.
    ///
    /// `signalled` is [`Self::encode_signal`]'s return: `None` means no signal
    /// was encoded and the blocking path is the only one available, which is
    /// what makes a missing event **fail safe** rather than spin forever.
    ///
    /// Returns the number of polls performed, so a caller can put engagement on
    /// a `RESULT` line -- §2.4: an instrument that cannot be shown to have
    /// engaged has not measured anything, and #500's finding is that a green
    /// board says nothing when nothing is counted. `Synchronize` reports 0,
    /// which is the value that distinguishes it from an async arm that happened
    /// to observe the signal on its first read.
    pub fn wait_for(
        &self,
        cb: &CommandBuffer,
        event: Option<&SharedEvent>,
        signalled: Option<u64>,
    ) -> Result<u64, MetalKernelError> {
        match self.wait_route(event.is_some(), signalled.is_some()) {
            // The incumbent, and the fallback whenever a signal was not
            // encoded. Falling back to a blocking wait can only ever be
            // *early*; falling the other way -- polling for a value nothing
            // will publish -- is a hang, so the asymmetry decides the
            // direction (§3.5).
            WaitRoute::Block => {
                cb.wait_until_completed();
                Ok(0)
            }
            WaitRoute::Poll => {
                // Both are `Some` on this route by construction, but the
                // routing function is the thing under test and this must not
                // re-derive the decision -- so the bindings are taken with a
                // fallback that blocks, keeping the fail-safe direction even if
                // the two ever disagreed.
                let (Some(ev), Some(target)) = (event, signalled) else {
                    cb.wait_until_completed();
                    return Ok(0);
                };
                let sleep = match self {
                    Self::SharedEventSleep { micros } => {
                        Some(std::time::Duration::from_micros(*micros))
                    }
                    _ => None,
                };
                let mut polls = 0u64;
                while ev.signaled_value() < target {
                    polls += 1;
                    if let Some(d) = sleep {
                        std::thread::sleep(d);
                    }
                }
                Ok(polls)
            }
        }
    }

    /// Which route [`Self::wait_for`] takes, as a pure function of the arm and
    /// what it was handed.
    ///
    /// Factored out so the fail-safe rule is **testable without a live queue**.
    /// The failure this guards against is a poll for a value nothing will
    /// publish, and that failure presents as a hang -- which a test cannot
    /// report. Reading the decision instead of executing it is the only way to
    /// assert it, and it is §3.1a rule 3's repair: the test calls the production
    /// classifier rather than restating its condition.
    fn wait_route(&self, has_event: bool, has_signal: bool) -> WaitRoute {
        match self {
            Self::Synchronize => WaitRoute::Block,
            Self::SharedEvent | Self::SharedEventSleep { .. } => {
                if has_event && has_signal {
                    WaitRoute::Poll
                } else {
                    WaitRoute::Block
                }
            }
        }
    }
}

/// The two ways [`CompletionWait::wait_for`] can wait. See
/// [`CompletionWait::wait_route`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaitRoute {
    /// `waitUntilCompleted` -- the incumbent, and the fail-safe fallback.
    Block,
    /// Poll the shared event's signalled value.
    Poll,
}

/// An `MTLSharedEvent` and the monotonically increasing value the host waits on.
///
/// One per `Commands`, allocated lazily and only when an async arm is selected
/// -- the default path never creates one, so it is byte-for-byte the incumbent.
///
/// # Why the counter is here and not on the event
///
/// `signaledValue` is readable and writable from the host (probe arm A4), so it
/// would be possible to derive the next target from it. Doing so would be a
/// read-modify-write over a value the **GPU also writes**, which is the two-clocks
/// defect §6.7 L4 names: the value can advance between the read and the use. The
/// counter here is written only by the host, only under the `Commands` lock the
/// caller already holds, and is compared against the event rather than derived
/// from it.
pub struct SharedEvent {
    raw: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    next: std::sync::atomic::AtomicU64,
}

// SAFETY: `MTLSharedEvent` is documented as usable across threads and processes
// -- it is the cross-device, cross-process member of §3.7b's table -- and the
// only mutable state beside it is an atomic. The same reasoning `CommandBuffer`
// above it uses.
unsafe impl Send for SharedEvent {}
unsafe impl Sync for SharedEvent {}

impl SharedEvent {
    /// Allocates the event, or `None` if the device declines to make one.
    ///
    /// A `None` is **not** an error: it is the capability being absent, and
    /// [`CompletionWait::wait_for`] falls back to the blocking wait, which is
    /// correct everywhere. Returning an error instead would make a device
    /// without shared events fail rather than run slower.
    pub fn new(device: &Device) -> Option<Self> {
        device.as_ref().newSharedEvent().map(|raw| Self {
            raw,
            next: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.raw
    }

    /// The next value to signal. Monotone, host-written only.
    fn next_value(&self) -> u64 {
        self.next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1)
    }

    /// The value the GPU has signalled so far.
    pub fn signaled_value(&self) -> u64 {
        self.raw.signaledValue()
    }
}

impl std::fmt::Debug for SharedEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEvent")
            .field("signaled", &self.signaled_value())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_arm() {
        assert_eq!(
            CompletionWait::from_str_arm("synchronize").unwrap(),
            CompletionWait::Synchronize
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event").unwrap(),
            CompletionWait::SharedEvent
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-sleep:7").unwrap(),
            CompletionWait::SharedEventSleep { micros: 7 }
        );
    }

    /// An invalid setting must fail rather than fall back (§2.4, after #69).
    ///
    /// The control is the arm above: if `from_str_arm` accepted everything, this
    /// test would pass while `parses_every_arm` also passed, so the pair is what
    /// makes either readable.
    #[test]
    fn rejects_unknown_arm() {
        assert!(CompletionWait::from_str_arm("wfe").is_err());
        assert!(CompletionWait::from_str_arm("").is_err());
        assert!(CompletionWait::from_str_arm("shared-event-sleep:").is_err());
        assert!(CompletionWait::from_str_arm("shared-event-sleep:-1").is_err());
    }

    /// Every arm renders a distinct name (§7.1a-i's defect was a *collision*).
    #[test]
    fn every_arm_renders_distinctly() {
        let arms = [
            CompletionWait::Synchronize,
            CompletionWait::SharedEvent,
            CompletionWait::SharedEventSleep { micros: 50 },
        ];
        let names: std::collections::HashSet<_> = arms.iter().map(|a| a.name()).collect();
        assert_eq!(names.len(), arms.len(), "two arms render the same name");
    }

    /// The default is the incumbent, pinned with an **absolute** reference so a
    /// flip is a visible edit to a test that says why (§3.1b rule 1: prefer an
    /// absolute reference to a shape).
    ///
    /// §7.1a's discipline is that a default flip needs its own argued decision
    /// with a number in hand, and issue #559 explicitly does not take that
    /// number. This is the mechanical half of that promise.
    #[test]
    fn default_is_the_incumbent() {
        assert_eq!(CompletionWait::default(), CompletionWait::Synchronize);
        assert_eq!(CompletionWait::default().name(), "synchronize");
    }

    /// Only the async arms allocate an event. The default path must not, or the
    /// incumbent is no longer byte-for-byte what every recorded figure belongs
    /// to.
    #[test]
    fn only_async_arms_need_an_event() {
        assert!(!CompletionWait::Synchronize.needs_event());
        assert!(CompletionWait::SharedEvent.needs_event());
        assert!(CompletionWait::SharedEventSleep { micros: 1 }.needs_event());
    }

    /// **The fail-safe direction, asserted rather than assumed.**
    ///
    /// An async arm with **no event** must take the blocking path -- never poll
    /// for a value nothing will publish. This is the asymmetry `wait_for`'s
    /// comment names: falling back to a wait is early, falling the other way is
    /// a hang, and a hang is the one failure a test cannot report.
    ///
    /// Asserted through [`CompletionWait::wait_route`], which is the *same*
    /// three-way classification `wait_for` matches on, factored out so it can be
    /// checked without a live queue. That factoring is what makes this a test of
    /// the rule rather than a restatement of it (§3.1a rule 3): mutating
    /// `wait_for`'s guard to poll without an event changes `wait_route`, and
    /// this test goes red instead of hanging.
    #[test]
    fn a_missing_signal_or_event_blocks_rather_than_polls() {
        use WaitRoute::*;
        // The incumbent blocks whatever it is handed.
        assert_eq!(CompletionWait::Synchronize.wait_route(false, false), Block);
        assert_eq!(CompletionWait::Synchronize.wait_route(true, true), Block);

        for arm in [
            CompletionWait::SharedEvent,
            CompletionWait::SharedEventSleep { micros: 1 },
        ] {
            // No event, or a signal that was never encoded: block.
            assert_eq!(arm.wait_route(false, false), Block, "{}", arm.name());
            assert_eq!(arm.wait_route(false, true), Block, "{}", arm.name());
            assert_eq!(arm.wait_route(true, false), Block, "{}", arm.name());
            // Both present is the only case that may poll.
            assert_eq!(arm.wait_route(true, true), Poll, "{}", arm.name());
        }
    }

    /// The routing table above is only meaningful if a poll is reachable at all
    /// -- an implementation that always returned `Block` would pass every
    /// assertion in the previous test except the last, so this is that arm's
    /// non-vacuity control stated separately (§3.1a rule 1).
    #[test]
    fn the_poll_route_is_reachable() {
        assert_eq!(
            CompletionWait::SharedEvent.wait_route(true, true),
            WaitRoute::Poll
        );
    }
}
