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
//!
//! # Three designs were collapsed into one word, and the third was never built
//!
//! **Separated 2026-09-02 (#583, `DESIGN.md` §11.5t.)** Every async arm above
//! **polls**, including the two named for mechanisms that do not have to, and
//! the reason is that three distinct designs shared the vocabulary of "waiting":
//!
//! | design | mechanism | who owns the exit condition |
//! |---|---|---|
//! | **poll a GPU-written counter** | host reads a mapped line in a loop | the host, by re-reading |
//! | **`WFE` on a watched line** | `LDXR` + `WFE`, wake on a store to that line | the host -- `WFE` retires on a **clock** (711-1325 ns, §3.7h), so the loop still re-tests |
//! | **wait on an `MTLSharedEvent`** | `waitUntilSignaledValue:timeoutMS:` | **the runtime** -- the thread does not run until the value arrives |
//!
//! The first two are **loops around a wait**; the third is **a wait**. `WFE` is
//! a watched-line spin whose quantum happens to be ~1 us, so naming it beside an
//! event wait obscures that it iterates at all -- §11.5s measured **1.39 M poll
//! iterations per token** under it, and a wait does not iterate 1.39 M times.
//!
//! **[`Self::SharedEventWait`] is the third**, and it is the arm §15.2 #10a asks
//! for. That rule says *"anything that polls for a condition owes an explanation
//! of why the producer cannot announce it"*. `poll.rs` discharges it for the
//! **kernel** case and the discharge is sound -- §3.7b establishes a compute
//! kernel carries no signal primitive. **It was never a discharge for this
//! seam**: the producer here is a **command buffer**, §11.5o's own probe shows it
//! announcing, and the arms above poll for a value the runtime would have parked
//! them for. So the poll here was a missing event rather than a missing
//! primitive, which is exactly the distinction §15.2 #10a is drawn to catch.

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
    /// `waitUntilSignaledValue:timeoutMS:` -- park in the runtime until the GPU
    /// signals, with **no poll loop at all** (#583).
    ///
    /// # Why this is a different design and not a third tuning of the same one
    ///
    /// The two arms above are loops: they read `signaledValue` and decide. This
    /// one hands the condition to the runtime and does not run until it holds,
    /// so [`Self::wait_for`] returns **0 polls** by construction rather than by
    /// luck -- and 0 here is a *structural* zero, which is why the counter is
    /// worth reading on this arm where §11.5o warns it is ambiguous on the
    /// others.
    ///
    /// This is the arm `DESIGN.md` §15.2 #10a asks for. `poll.rs` discharges
    /// that rule for a **kernel**, which genuinely cannot announce (§3.7b); a
    /// **command buffer** can, and §11.5o's own probe measured it doing so. The
    /// polling arms were therefore optimising a poll whose producer was able to
    /// announce the whole time.
    ///
    /// # The timeout is a fail-safe and not a tuning knob
    ///
    /// `waitUntilSignaledValue:timeoutMS:` returns `false` on timeout, and a
    /// timeout **must not** be read as completion: §3.5 makes a read before the
    /// GPU's write silent corruption rather than an error. So a `false` falls
    /// back to [`Self::Synchronize`]'s blocking wait rather than proceeding,
    /// which is the same asymmetry [`Self::wait_for`] applies to a missing
    /// event -- falling back to a block can only ever be *early*.
    ///
    /// The default is deliberately long ([`DEFAULT_WAIT_TIMEOUT_MS`]): it exists
    /// so a lost signal becomes a slow run rather than a hang, not so that it
    /// can be tuned against the ~18 ms token cadence.
    SharedEventWait {
        /// Milliseconds before the wait gives up and blocks instead.
        timeout_ms: u64,
    },
}

/// The fail-safe timeout for [`CompletionWait::SharedEventWait`], in
/// milliseconds.
///
/// **Two orders of magnitude above the 18.76 ms token cadence** (§6.6), because
/// this is the bound on a *pathology* -- a signal that never arrives -- and not
/// a latency budget. A timeout close to the cadence would convert ordinary
/// jitter into a fallback and the arm would silently become the incumbent,
/// which is §7.1a-i's wrong-arm shape: a run reporting one arm and executing
/// another.
pub const DEFAULT_WAIT_TIMEOUT_MS: u64 = 2_000;

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
            Self::SharedEventWait { .. } => "shared-event-wait",
        }
    }

    /// Whether this arm reaches the GPU through a poll loop.
    ///
    /// **The axis #583 separates**, readable rather than inferred from a name.
    /// `Synchronize` and `SharedEventWait` both park -- in the Mach kernel and
    /// in the Metal runtime respectively -- while the two `SharedEvent*` arms
    /// iterate. A run can therefore assert *"this arm must report 0 polls"*
    /// from the arm itself instead of from a table a reader has to trust, which
    /// is what makes the engagement counter checkable (§2.4).
    pub fn polls(&self) -> bool {
        match self {
            Self::Synchronize | Self::SharedEventWait { .. } => false,
            Self::SharedEvent | Self::SharedEventSleep { .. } => true,
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
            Self::SharedEvent | Self::SharedEventSleep { .. } | Self::SharedEventWait { .. } => {
                true
            }
        }
    }

    /// Parses `synchronize`, `shared-event`, `shared-event-sleep`,
    /// `shared-event-sleep:<micros>`, `shared-event-wait` and
    /// `shared-event-wait:<timeout_ms>`.
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
            "shared-event-wait" => Ok(Self::SharedEventWait {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
            }),
            // The two parameterised arms are matched **before** any shorter
            // prefix, and both are matched exhaustively here rather than by
            // `starts_with("shared-event")` -- `shared-event-wait:5` must not
            // be readable as `shared-event` with a suffix nobody looked at.
            // §7.1a-i: a run that reports one arm and executes another.
            other => {
                if let Some(us) = other.strip_prefix("shared-event-sleep:") {
                    return us
                        .parse::<u64>()
                        .map(|micros| Self::SharedEventSleep { micros })
                        .map_err(|e| format!("completion wait {other:?}: bad microseconds: {e}"));
                }
                if let Some(ms) = other.strip_prefix("shared-event-wait:") {
                    return ms
                        .parse::<u64>()
                        .map(|timeout_ms| Self::SharedEventWait { timeout_ms })
                        .map_err(|e| format!("completion wait {other:?}: bad milliseconds: {e}"));
                }
                Err(format!(
                    "unknown completion wait {other:?} (expected synchronize, \
                     shared-event, shared-event-sleep, shared-event-sleep:<micros>, \
                     shared-event-wait, or shared-event-wait:<timeout_ms>)"
                ))
            }
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
            WaitRoute::Park { timeout_ms } => {
                // As `Poll` above: both are `Some` on this route by
                // construction, and re-deriving the decision here would make
                // the routing function untestable. The fallback blocks.
                let (Some(ev), Some(target)) = (event, signalled) else {
                    cb.wait_until_completed();
                    return Ok(0);
                };
                // The whole arm, in one call: the thread does not run until the
                // runtime observes the value. No loop, so no poll to count --
                // the 0 below is structural rather than a lucky first read.
                if !ev.wait_until_signaled(target, timeout_ms) {
                    // A timeout is **not** completion. §3.5 makes reading before
                    // the GPU's write silent corruption rather than an error, so
                    // this falls back to the blocking wait -- the same
                    // fail-safe direction as a missing event, and for the same
                    // reason: blocking can only ever be early.
                    cb.wait_until_completed();
                }
                Ok(0)
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
            // The same fail-safe guard as the polling arms, and it is needed
            // for a *different* reason worth stating: a park without a signal
            // does not spin, it sleeps until the timeout and then blocks -- so
            // the failure would be a slow run rather than a hang, and a slow
            // run is the harder one to notice.
            Self::SharedEventWait { timeout_ms } => {
                if has_event && has_signal {
                    WaitRoute::Park {
                        timeout_ms: *timeout_ms,
                    }
                } else {
                    WaitRoute::Block
                }
            }
        }
    }
}

/// The three ways [`CompletionWait::wait_for`] can wait. See
/// [`CompletionWait::wait_route`].
///
/// **`Block` and `Park` both park and are not the same route** (#583): `Block`
/// enters the Mach kernel via `waitUntilCompleted` and knows nothing about the
/// event, while `Park` waits on the event's value in the Metal runtime and
/// falls back to `Block` on timeout. Collapsing them would make the fallback
/// unobservable, and the fallback is the arm's only unsound direction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaitRoute {
    /// `waitUntilCompleted` -- the incumbent, and the fail-safe fallback.
    Block,
    /// Poll the shared event's signalled value.
    Poll,
    /// Park in the runtime until the event reaches the value, or `timeout_ms`
    /// elapses and the wait falls back to [`Self::Block`].
    Park {
        /// Milliseconds before falling back to the blocking wait.
        timeout_ms: u64,
    },
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

    /// Park until the event reaches `value`, or `timeout_ms` elapses (#583).
    ///
    /// Returns **`true` if the value was reached** and `false` on timeout. The
    /// caller must treat `false` as *"still unknown"* and not as completion --
    /// [`CompletionWait::wait_for`] blocks on it, for §3.5's reason.
    ///
    /// This is `waitUntilSignaledValue:timeoutMS:`, which was bound in
    /// `objc2-metal` and called from nowhere in this tree until #583. It is the
    /// only wait here that does not iterate: the thread is descheduled by the
    /// runtime and the exit condition belongs to it rather than to a loop the
    /// host writes.
    pub fn wait_until_signaled(&self, value: u64, timeout_ms: u64) -> bool {
        self.raw.waitUntilSignaledValue_timeoutMS(value, timeout_ms)
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
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-wait").unwrap(),
            CompletionWait::SharedEventWait {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS
            }
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-wait:7").unwrap(),
            CompletionWait::SharedEventWait { timeout_ms: 7 }
        );
    }

    /// The two parameterised arms share a prefix, and neither may absorb the
    /// other (#583).
    ///
    /// `shared-event-wait` starts with `shared-event`, so a parser reaching for
    /// the shorter name first -- or using `starts_with` -- would resolve the
    /// wait arm to the **polling** arm and silently drop the suffix. That is
    /// §7.1a-i's wrong-arm shape, and a wrong arm compares equal to every run
    /// of that arm where an absent one compares equal to nothing.
    ///
    /// Asserted with an **absolute** reference on each side (§3.1b rule 1)
    /// rather than as *"the two differ"*, which a parser mapping both to a
    /// third arm would also satisfy.
    #[test]
    fn the_prefixed_arms_do_not_absorb_each_other() {
        assert_eq!(
            CompletionWait::from_str_arm("shared-event").unwrap(),
            CompletionWait::SharedEvent
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-wait").unwrap(),
            CompletionWait::SharedEventWait {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS
            }
        );
        // The suffix must reach the field rather than being parsed and dropped:
        // a `shared-event-wait:5` read as the default timeout would pass an
        // equality test against the arm and be wrong by 1995 ms.
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-wait:5").unwrap(),
            CompletionWait::SharedEventWait { timeout_ms: 5 }
        );
        assert_ne!(DEFAULT_WAIT_TIMEOUT_MS, 5, "the control above is vacuous");
    }

    /// Which arms poll, pinned as an **absolute** list rather than a shape
    /// (#583).
    ///
    /// This is the axis #583 separates: `Synchronize` and `SharedEventWait`
    /// park, the two `SharedEvent*` arms iterate. The property is asserted per
    /// arm because a shape invariant -- *"the arms differ"* -- is blind to a
    /// defect that moves every member (§3.1b), and inverting `polls()` wholesale
    /// is exactly such a defect.
    #[test]
    fn exactly_the_looping_arms_report_polling() {
        assert!(!CompletionWait::Synchronize.polls());
        assert!(CompletionWait::SharedEvent.polls());
        assert!(CompletionWait::SharedEventSleep { micros: 1 }.polls());
        assert!(!CompletionWait::SharedEventWait { timeout_ms: 1 }.polls());
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
            CompletionWait::SharedEventWait { timeout_ms: 2_000 },
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
        assert!(CompletionWait::SharedEventWait { timeout_ms: 1 }.needs_event());
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

        // The parking arm takes the same three fail-safe cases (#583). Its
        // failure mode is a *slow* run rather than a hang -- it would sleep to
        // the timeout and then block -- which is the harder one to notice, so
        // it is asserted rather than left to the timeout.
        let park = CompletionWait::SharedEventWait { timeout_ms: 1 };
        assert_eq!(park.wait_route(false, false), Block, "{}", park.name());
        assert_eq!(park.wait_route(false, true), Block, "{}", park.name());
        assert_eq!(park.wait_route(true, false), Block, "{}", park.name());
        assert_eq!(
            park.wait_route(true, true),
            Park { timeout_ms: 1 },
            "{}",
            park.name()
        );
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

    /// The park route's non-vacuity control, and it carries the timeout (#583).
    ///
    /// The same argument as `the_poll_route_is_reachable` one arm along: an
    /// implementation collapsing `Park` into `Block` would satisfy every
    /// fail-safe assertion above, because those all *expect* `Block`. Only a
    /// positive assertion that the route is reached can see it.
    ///
    /// **The timeout is asserted through the route rather than the field**,
    /// because that is the value `wait_for` actually hands to the runtime: a
    /// routing function that returned a hard-coded default would leave the enum
    /// correct and the wait wrong.
    #[test]
    fn the_park_route_is_reachable_and_carries_its_timeout() {
        assert_eq!(
            CompletionWait::SharedEventWait { timeout_ms: 1234 }.wait_route(true, true),
            WaitRoute::Park { timeout_ms: 1234 }
        );
        assert_ne!(
            1234, DEFAULT_WAIT_TIMEOUT_MS,
            "the timeout above equals the default, so this cannot see a hard-coded one"
        );
    }

    /// A parking arm must never route to a poll, whatever it is handed (#583).
    ///
    /// This is the property that makes `completion_polls == 0` readable as
    /// evidence rather than as a coincidence: if the wait arm could reach
    /// `Poll` under any input, a zero would mean *"the signal was up on the
    /// first read"* instead of *"this arm does not iterate"*. Stated over all
    /// four input combinations because that is the whole domain.
    #[test]
    fn a_parking_arm_never_routes_to_a_poll() {
        let park = CompletionWait::SharedEventWait {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
        };
        for (has_event, has_signal) in [(false, false), (false, true), (true, false), (true, true)] {
            assert_ne!(
                park.wait_route(has_event, has_signal),
                WaitRoute::Poll,
                "the wait arm polled with event={has_event} signal={has_signal}"
            );
        }
        // The control: `Poll` is reachable for an arm that should reach it, so
        // the assertion above is not passing over an unreachable variant.
        assert_eq!(
            CompletionWait::SharedEvent.wait_route(true, true),
            WaitRoute::Poll
        );
    }
}
