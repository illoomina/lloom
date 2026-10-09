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
use objc2_metal::{MTLDevice, MTLSharedEvent, MTLSharedEventListener};

use super::cache_pad::CachePadded;
use super::{CommandBuffer, Device};
use crate::kernels::MetalKernelError;

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
    /// Signal an `MTLSharedEvent` at the command-buffer boundary, poll
    /// `signaledValue` for at most [`SHARED_EVENT_POLL_BOUND`] reads, then park
    /// until a listener block reports the value.
    ///
    /// The poll is for a buffer about to complete; a buffer that does not
    /// complete inside it is waited for parked. Polling through the whole
    /// buffer held a core for the wait, and through a host-planned pass that
    /// keeps the GPU at low clocks: +28…+30 ms a token on Qwen4-exp's miss
    /// regime (`expert-streaming.md` §20.14a, D5).
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
    /// `notifyListener:atValue:block:` -- the GPU's signal invokes a **block on
    /// a GCD queue**, which wakes this thread through a condvar (#583).
    ///
    /// # What is different, and it is the host thread rather than the wait
    ///
    /// The three arms above are all things the *waiting thread* does: it loops
    /// ([`Self::SharedEvent`], [`Self::SharedEventSleep`]) or it parks inside
    /// the runtime ([`Self::SharedEventWait`]). This arm is the only one where
    /// the completion is delivered **to** the host rather than waited **for**
    /// by it -- Metal invokes a callback on a background dispatch queue, and
    /// what that callback does is a decision this code makes rather than one
    /// the runtime makes for it.
    ///
    /// Here it signals a condvar, because [`super::Commands::flush_and_wait_current`]'s
    /// contract is *the value is ready when I return* -- `to_cpu` reads the
    /// buffer on the next line, and §3.5 makes a read before the GPU's write
    /// silent corruption rather than an error. **So this arm still blocks the
    /// calling thread, and it must**; what it does not do is burn a core or sit
    /// in the Metal runtime's own wait.
    ///
    /// # What it is FOR is the callback, and this arm is the seam for it
    ///
    /// The issue this was built for describes the shape where *"the host does
    /// other work instead of waiting at all"*. That shape needs a caller with
    /// other work to do, and `to_cpu` -- the one call site on the decode path
    /// (§11.5f's census) -- has none: it needs the bytes. **So this arm is the
    /// mechanism in place with its consumer honestly absent**, which is #500's
    /// finding recorded in advance rather than discovered by a measurement:
    /// what is built here is the *notification path*, and a decode loop that
    /// overlaps host work with GPU work is a change to the loop, not to this.
    ///
    /// # Lifetime is the hazard, and it fails silently
    ///
    /// *"If the listener is deallocated, registered completion blocks will
    /// never fire"* -- and **a mechanism that stops firing is indistinguishable
    /// from one that never engaged** (§2.4, and #500's whole finding). This arm
    /// therefore holds its listener for the life of the [`SharedEvent`] rather
    /// than per wait, and [`Self::wait_for`] returns a **fired count** so the
    /// block's arrival is asserted rather than assumed.
    ///
    /// # The timeout is a fail-safe, exactly as [`Self::SharedEventWait`]'s is
    ///
    /// A block that never fires would hang, and a hang is the one failure a
    /// test cannot report. The condvar wait is bounded and a timeout falls back
    /// to the blocking wait -- which can only ever be *early*.
    SharedEventListener {
        /// Milliseconds before the condvar wait gives up and blocks instead.
        timeout_ms: u64,
    },
}

/// Reads of `signaledValue` [`CompletionWait::SharedEvent`] makes before it
/// parks: ~40 µs at the ~50 000 reads a millisecond measured above.
pub const SHARED_EVENT_POLL_BOUND: u64 = 2_048;

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
    /// Every arm a user can select, as the **enum's own** enumeration (#583).
    ///
    /// # Why this exists, and it is a defect found by making it
    ///
    /// #583 added a fourth arm and repaired the GPU suite's hand-written
    /// coverage array by *"checking the list against the parser"*. That repair
    /// carried a second hand-written list -- the parser names it checked
    /// against -- and **it went stale on the very next arm**: adding a fifth
    /// left `every_arm_is_covered_here` green while the new arm was executed by
    /// no test, because the test compared two hand-written lists **to each
    /// other**. That is #519's *"a comparison that looks like one and is not"*
    /// (§3.1b), inside the check written to prevent exactly this.
    ///
    /// **The durable form is an enumeration the enum owns**, so a new variant
    /// is one edit here and every consumer -- the GPU sweep, the registry
    /// check, a harness's `--help` -- is derived rather than maintained. It is
    /// §8.1b's checked-registry argument applied to an axis's arms, and the
    /// `no_arm_is_missing_from_all_arms` test below is what keeps it honest,
    /// since a `const` array cannot be exhaustively checked by the compiler.
    ///
    /// Parameterised arms appear once, at their default period: the period is a
    /// value on the arm, not a separate arm, and `from_str_arm` round-trips
    /// both spellings.
    pub const ALL_ARMS: [Self; 5] = [
        Self::Synchronize,
        Self::SharedEvent,
        Self::SharedEventSleep { micros: 50 },
        Self::SharedEventWait {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
        },
        Self::SharedEventListener {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
        },
    ];

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
            Self::SharedEventListener { .. } => "shared-event-listener",
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
            Self::Synchronize | Self::SharedEventWait { .. } | Self::SharedEventListener { .. } => {
                false
            }
            Self::SharedEvent | Self::SharedEventSleep { .. } => true,
        }
    }

    /// Whether this arm is woken by a **callback** rather than by waiting.
    ///
    /// The axis the listener arm adds (#583), and it is orthogonal to
    /// [`Self::polls`]: `Synchronize` and `SharedEventWait` both answer `false`
    /// to both, and they are not the same mechanism -- one parks in the Mach
    /// kernel on the command buffer, the other in the Metal runtime on the
    /// event's value. This arm is the only one where the completion is
    /// **delivered** to the host, on a GCD worker, rather than waited for on
    /// the calling thread.
    ///
    /// Readable rather than inferred from a name, for [`Self::polls`]'s reason:
    /// a run can assert *"this arm must report a nonzero fired count"* from the
    /// arm itself instead of from a table a reader has to trust (§2.4).
    pub fn notifies(&self) -> bool {
        match self {
            Self::SharedEventListener { .. } => true,
            Self::Synchronize
            | Self::SharedEvent
            | Self::SharedEventSleep { .. }
            | Self::SharedEventWait { .. } => false,
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
            Self::SharedEvent
            | Self::SharedEventSleep { .. }
            | Self::SharedEventWait { .. }
            | Self::SharedEventListener { .. } => true,
        }
    }

    /// The arm's period, if it carries one — and it is the enum that knows
    /// (#583).
    ///
    /// **Three consumers were each rendering this with their own `match`**, and
    /// each is a list that goes stale silently when an arm is added: adding a
    /// parameterised arm is a compile error at an exhaustive `match` and a
    /// *wrong answer* at one with a `_ =>` fallback, which is what these had.
    /// An arm whose period does not reach the axis vector renders **equal** to
    /// the same arm at a different period — §7.1a-i's present-and-wrong
    /// direction, and §13.5m's measured defect at `CutPolicy`.
    ///
    /// The unit is the arm's, not this function's: `shared-event-sleep`'s is
    /// microseconds and both `shared-event-wait`'s and
    /// `shared-event-listener`'s are milliseconds. That is why the rendering is
    /// `arm:period` — the name already says which unit it is, and a second
    /// field for the unit would be a second copy of a list (§11.3h).
    pub fn period(&self) -> Option<u64> {
        match self {
            Self::Synchronize | Self::SharedEvent => None,
            Self::SharedEventSleep { micros } => Some(*micros),
            Self::SharedEventWait { timeout_ms } => Some(*timeout_ms),
            Self::SharedEventListener { timeout_ms } => Some(*timeout_ms),
        }
    }

    /// The arm as an axis vector element: `name` or `name:period` (#583).
    ///
    /// Derived from [`Self::name`] and [`Self::period`] so a new arm reaches
    /// every consumer at once. `lloom-runs` keys comparisons on this string, so
    /// two runs differing only in a period must not render equal.
    pub fn arm_with_period(&self) -> String {
        match self.period() {
            Some(p) => format!("{}:{p}", self.name()),
            None => self.name().to_string(),
        }
    }

    /// Whether this arm needs an `MTLSharedEventListener` beside its event.
    ///
    /// Only the listener arm does, and it is allocated with the event so its
    /// lifetime is the engine's: *"if the listener is deallocated, registered
    /// completion blocks will never fire"*, and a block that stops firing reads
    /// exactly like one that never engaged (§2.4).
    pub fn needs_listener(&self) -> bool {
        // `SharedEvent` parks through a listener once its poll bound is spent.
        self.notifies() || matches!(self, Self::SharedEvent)
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
            "shared-event-listener" => Ok(Self::SharedEventListener {
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
                if let Some(ms) = other.strip_prefix("shared-event-listener:") {
                    return ms
                        .parse::<u64>()
                        .map(|timeout_ms| Self::SharedEventListener { timeout_ms })
                        .map_err(|e| format!("completion wait {other:?}: bad milliseconds: {e}"));
                }
                Err(format!(
                    "unknown completion wait {other:?} (expected synchronize, \
                     shared-event, shared-event-sleep, shared-event-sleep:<micros>, \
                     shared-event-wait, shared-event-wait:<timeout_ms>, \
                     shared-event-listener, or shared-event-listener:<timeout_ms>)"
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
        *ARM.get_or_init(|| {
            Self::from_str_arm(&crate::tuning::tuning().metal.completion_wait)
                .unwrap_or_else(|e| panic!("tuning.metal.completion_wait: {e}"))
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
        cb.signal_event(event.raw(), value);
        Some(value)
    }

    /// Wait until `cb`'s work is done, by this arm's mechanism.
    ///
    /// `signalled` is [`Self::encode_signal`]'s return: `None` means no signal
    /// was encoded and the blocking path is the only one available, which is
    /// what makes a missing event **fail safe** rather than spin forever.
    ///
    /// Returns [`WaitOutcome`] -- polls performed and blocks fired -- so a
    /// caller can put engagement on a `RESULT` line. §2.4: an instrument that
    /// cannot be shown to have engaged has not measured anything, and #500's
    /// finding is that a green board says nothing when nothing is counted.
    /// `Synchronize` reports zeros on both, which is the value that
    /// distinguishes it from an async arm that happened to observe the signal
    /// on its first read.
    ///
    /// **Two counters rather than one, because the arms fail differently**
    /// (#583): a polling arm that stopped engaging reads `polls = 0`, and a
    /// listener arm that stopped engaging *also* reads `polls = 0` -- so on the
    /// listener arm the poll count is not an engagement proof at all and
    /// `fired` is. That is §3.1c's rule (*a zero is not evidence unless
    /// something nonzero came from the same command*) applied to a counter
    /// rather than to a run.
    pub fn wait_for(
        &self,
        cb: &CommandBuffer,
        event: Option<&SharedEvent>,
        signalled: Option<u64>,
    ) -> Result<WaitOutcome, MetalKernelError> {
        match self.wait_route(event.is_some(), signalled.is_some()) {
            // The incumbent, and the fallback whenever a signal was not
            // encoded. Falling back to a blocking wait can only ever be
            // *early*; falling the other way -- polling for a value nothing
            // will publish -- is a hang, so the asymmetry decides the
            // direction (§3.5).
            WaitRoute::Block => {
                cb.wait_until_completed();
                Ok(WaitOutcome::default())
            }
            WaitRoute::Poll => {
                // Both are `Some` on this route by construction, but the
                // routing function is the thing under test and this must not
                // re-derive the decision -- so the bindings are taken with a
                // fallback that blocks, keeping the fail-safe direction even if
                // the two ever disagreed.
                let (Some(ev), Some(target)) = (event, signalled) else {
                    cb.wait_until_completed();
                    return Ok(WaitOutcome::default());
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
                    match sleep {
                        Some(d) => std::thread::sleep(d),
                        None if polls >= SHARED_EVENT_POLL_BOUND => {
                            // Not imminent: park for the rest, woken by the
                            // listener, with the parking arms' fail-safe.
                            if ev.notify_then_wait(target, DEFAULT_WAIT_TIMEOUT_MS) == 0
                                && ev.signaled_value() < target
                            {
                                cb.wait_until_completed();
                            }
                            break;
                        }
                        None => {}
                    }
                }
                Ok(WaitOutcome {
                    polls,
                    fired: 0,
                    already_complete: 0,
                })
            }
            WaitRoute::Park { timeout_ms } => {
                // As `Poll` above: both are `Some` on this route by
                // construction, and re-deriving the decision here would make
                // the routing function untestable. The fallback blocks.
                let (Some(ev), Some(target)) = (event, signalled) else {
                    cb.wait_until_completed();
                    return Ok(WaitOutcome::default());
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
                Ok(WaitOutcome::default())
            }
            WaitRoute::Notify { timeout_ms } => {
                // As the two routes above: both are `Some` here by
                // construction, and re-deriving the decision would make
                // `wait_route` untestable. The fallback blocks.
                let (Some(ev), Some(target)) = (event, signalled) else {
                    cb.wait_until_completed();
                    return Ok(WaitOutcome::default());
                };
                // The block is registered on the event and fires on a GCD
                // worker; this thread parks on the condvar it signals. Nothing
                // here loops, so `polls` stays 0 -- and unlike the parking arm
                // above, that zero is NOT this arm's engagement proof, because
                // a listener that never fired reads zero too. `fired` is.
                let fired = ev.notify_then_wait(target, timeout_ms);
                if fired == 0 {
                    // The block did not arrive inside the budget. Same
                    // fail-safe direction and the same reason as every arm
                    // above -- blocking can only ever be early -- and it is
                    // reported rather than hidden, because a listener silently
                    // falling back to the incumbent is §7.1a-i's wrong-arm
                    // shape: a run that says one thing and does another.
                    cb.wait_until_completed();
                }
                Ok(WaitOutcome {
                    polls: 0,
                    fired,
                    already_complete: 0,
                })
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
            // The same fail-safe guard again, and a third distinct reason it is
            // needed: with no signal encoded, no block is ever registered, so a
            // condvar wait would time out rather than return -- a slow run, and
            // the timeout is what stops it being a hang.
            Self::SharedEventListener { timeout_ms } => {
                if has_event && has_signal {
                    WaitRoute::Notify {
                        timeout_ms: *timeout_ms,
                    }
                } else {
                    WaitRoute::Block
                }
            }
        }
    }
}

/// What one wait engaged, returned by [`CompletionWait::wait_for`] (#583).
///
/// **Two counters because the arms fail differently**, and one would hide the
/// listener arm's failure. A polling arm proves it ran by `polls > 0`; the
/// listener arm polls zero times whether it worked or not, so its proof is
/// `fired > 0`. Reporting a single number would make *"the block never
/// arrived and we quietly blocked instead"* read exactly like a working parked
/// wait -- §2.4's rule, and #500's finding that a green board says nothing when
/// nothing is counted.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct WaitOutcome {
    /// Reads of `signaledValue` performed by a polling arm.
    pub polls: u64,
    /// Notification blocks that fired and woke this wait (#583).
    ///
    /// `0` on every arm that does not register one, and `0` on the listener arm
    /// **when the block did not arrive inside the timeout** -- which is the
    /// failure this counter exists to make visible rather than a second normal
    /// value.
    pub fired: u64,
    /// `1` when the command buffer had **already completed** and no arm ran at
    /// all (#583).
    ///
    /// **Measured 0 on the decode path, and it is kept because of what it
    /// excluded.** It was added to test an explanation for `completion_fires`
    /// reading ~28 against `completion_signals` ~384 -- that most readbacks
    /// arrive after the GPU has retired the buffer -- and it **falsified**
    /// that: the buffers are not already complete.
    ///
    /// The real reason is that the two counters have **different
    /// denominators**. A signal is encoded in every `commit_swap_locked` (five
    /// call sites), and only `flush_and_wait_current`'s reaches a wait; at 675
    /// dispatches/token under `CutPolicy::Counter(50)` that is ~13.5 commits
    /// per token against **one** wait. So `fired` ~= one per token is the
    /// listener firing on essentially every wait there is, and the ratio to
    /// `signals` is arithmetic rather than a failure rate.
    ///
    /// Kept rather than deleted because a counter that reads zero for a
    /// *measured* reason is evidence, where the absence of one leaves the same
    /// question open for the next reader (§2.4).
    pub already_complete: u64,
}

/// The four ways [`CompletionWait::wait_for`] can wait. See
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
    /// Register a notification block and park on a condvar until it fires, or
    /// `timeout_ms` elapses and the wait falls back to [`Self::Block`] (#583).
    ///
    /// **Distinct from [`Self::Park`] for the same reason `Park` is distinct
    /// from [`Self::Block`]**: all three end with the thread descheduled, and
    /// they differ in *what wakes it* -- the Mach kernel on the command buffer,
    /// the Metal runtime on the event's value, and a GCD worker running our own
    /// block. Only the third has a fired count, and collapsing the routes would
    /// make that count unobservable.
    Notify {
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
    /// The listener notification blocks are delivered on, held for this
    /// event's whole life (#583).
    ///
    /// **The strong reference is the mechanism, not tidiness.** Apple's
    /// guidance is explicit -- *"if the listener is deallocated, registered
    /// completion blocks will never fire"* -- and a block that stops firing is
    /// indistinguishable from one that never engaged (§2.4, #500). Holding it
    /// per *wait* rather than per *event* would drop it between tokens and the
    /// arm would silently become the blocking fallback: §7.1a-i's wrong-arm
    /// shape, where a run reports one arm and executes another.
    ///
    /// `None` on every arm that does not need one, so the incumbent allocates
    /// nothing (the same discipline as the event itself).
    listener: Option<Retained<MTLSharedEventListener>>,
    /// Where a fired block leaves the value it saw, and the condvar it wakes
    /// the waiting thread through.
    ///
    /// **`Arc` because a registered block may fire after this event is
    /// dropped.** Metal owns the block once it is registered and there is no
    /// API to withdraw one, so a block capturing a borrow of this struct would
    /// be a write into freed memory on exactly the path -- a late or lost
    /// signal -- that the arm's timeout exists to survive. The `Arc` makes the
    /// state outlive both, which is structural rather than an argument about
    /// timing.
    ///
    /// `CachePadded` because the block runs on a **GCD worker** and the waiter
    /// is the decode thread, so this line is genuinely cross-thread -- §9.2l
    /// prices a shared line at 4.93x at two threads, and #582 is the standing
    /// rule that a counter a completion handler shares with the host gets
    /// padded.
    shared: CachePadded<std::sync::Arc<NotifySlot>>,
}

/// The condvar-guarded state a notification block writes and a waiter reads.
///
/// `parking_lot`, as every lock in this crate is (`Cargo.toml`): std's macOS
/// `Mutex` forces a poison `Result` per acquisition, and this one is taken on
/// a **GCD worker** inside the notification block — a branch on a path whose
/// whole purpose is to deliver a wake promptly. `parking_lot`'s pair is also
/// word-sized and allocation-free, where std's `Condvar` on macOS carries a
/// `pthread_cond_t`. The poison arms this removes were dead weight in both
/// directions: neither the block nor the waiter can panic while holding it.
type NotifySlot = (parking_lot::Mutex<NotifyState>, parking_lot::Condvar);

/// What a fired notification block leaves for the waiting thread.
///
/// The value is carried rather than a bare flag for §11.5n's reason one level
/// out: a flag cannot distinguish *"the block fired for the value I want"* from
/// *"the block fired for an earlier one"*, and blocks for different values are
/// delivered independently. Comparing against a target the waiter already holds
/// is the same repair the ring's generation counter is.
#[derive(Default)]
pub struct NotifyState {
    /// Highest value any fired block has reported.
    seen: u64,
    /// How many blocks have fired on this event since it was created.
    ///
    /// The engagement proof for the listener arm, and it is a **count** rather
    /// than a flag so that a wait can tell *its own* block fired rather than
    /// inheriting a previous one's. Completion still requires `seen >= target`;
    /// this count alone cannot identify which target fired.
    fired: u64,
}

impl NotifyState {
    fn waiting_for(&self, target: u64) -> bool {
        self.seen < target
    }

    fn completed_notifications(&self, target: u64, before: u64) -> u64 {
        // A late callback for an earlier target is engagement, not completion
        // of this wait. Returning zero preserves the caller's blocking fallback.
        if self.waiting_for(target) {
            0
        } else {
            self.fired.saturating_sub(before)
        }
    }
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
        Self::with_listener(device, false)
    }

    /// As [`Self::new`], allocating an `MTLSharedEventListener` when `listener`
    /// (#583).
    ///
    /// The listener is created **with the event** and dropped **with it**, which
    /// is the whole lifetime discipline: Apple's guidance is that a deallocated
    /// listener silently stops delivering blocks, and this is the allocation
    /// whose lifetime is easiest to reason about because it is the same as the
    /// thing it notifies for.
    ///
    /// `MTLSharedEventListener::new()` takes the process's default dispatch
    /// queue rather than one we make. That is deliberate: a queue of our own
    /// would be a second lifetime to get right for no benefit, and #547 already
    /// establishes the block runs on a GCD worker either way
    /// (`MTLDispatchListApply` -> `libsystem_blocks`), so the queue's identity
    /// is not what the soundness rests on.
    pub fn with_listener(device: &Device, listener: bool) -> Option<Self> {
        device.as_ref().newSharedEvent().map(|raw| Self {
            raw,
            next: std::sync::atomic::AtomicU64::new(0),
            listener: listener.then(MTLSharedEventListener::new),
            shared: CachePadded::new(std::sync::Arc::new((
                parking_lot::Mutex::new(NotifyState::default()),
                parking_lot::Condvar::new(),
            ))),
        })
    }

    /// Whether a listener was allocated for this event.
    ///
    /// Reported so *"the listener arm was selected"* and *"the listener arm is
    /// running"* stay separable, the same distinction
    /// [`super::Commands::completion_event_present`] draws for the event.
    pub fn has_listener(&self) -> bool {
        self.listener.is_some()
    }

    fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.raw
    }

    /// The next value to signal. Monotone, host-written only.
    ///
    /// # This is NOT the token ring's generation counter, and they cannot merge
    ///
    /// #583 asks whether the ring's counter and this value *"are the same
    /// number"* before adding a second. **They are not**, and the reason is the
    /// same *different heights* argument that decided the ring still needs its
    /// counter under a real event wait (§11.5t):
    ///
    /// | | ring's generation counter | this value |
    /// |---|---|---|
    /// | written by | a **kernel**, mid-buffer, via `slice_set` | the **host**, before commit |
    /// | advances | once per **token** | once per **`commit_swap_locked`** — ~13.5/token under `CutPolicy::Counter(50)` |
    /// | observed by | the host polling a mapped line (§3.7g) | the Metal runtime, or the host reading `signaledValue` |
    /// | means | *this step's output is published* | *this command buffer is retired* |
    ///
    /// **Measured, they differ by more than an offset**: at `n` = 24 this
    /// counter reaches ~384 while the ring publishes 24 times, and the ratio is
    /// the cut policy's rather than a constant. A single number cannot carry
    /// both, and §3.7b is why the ring's cannot simply become an event: a
    /// compute kernel has no signal primitive, so there is no event at the
    /// height the ring needs one.
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

    /// Register a notification block for `value` and park until it fires, or
    /// until `timeout_ms` elapses (#583).
    ///
    /// Returns **the number of blocks that had fired** by the time this
    /// returned, which is `0` exactly when the block did not arrive inside the
    /// budget -- so a caller can tell *"the listener delivered"* from *"the
    /// listener is not delivering and I quietly blocked instead"*. That
    /// distinction is the reason this returns a count rather than a `bool`
    /// (§2.4, #500).
    ///
    /// # The shape, and why the block does not do the work
    ///
    /// Metal invokes the block on a GCD worker. What the block does here is the
    /// smallest thing that can be done correctly: record the value it was given
    /// and wake the condvar. **It must not run the caller's work** -- the caller
    /// is `to_cpu`, whose contract is *give me the finished value*, and moving
    /// that onto a background queue would change what
    /// [`super::Commands::flush_and_wait_current`] returning means.
    ///
    /// # Why the state is compared and not merely flagged
    ///
    /// Blocks for different values are delivered independently, so a bare flag
    /// cannot distinguish *this* step's completion from a previous one's. The
    /// waiter therefore compares `seen` against a `value` it already holds --
    /// the same repair §11.5n made when a one-byte EOS flag turned out not to
    /// be pollable, one level out.
    ///
    /// # The GCD boundary is a known blind spot for TSan
    ///
    /// #547 read the stack (`MTLDispatchListApply` -> `libsystem_blocks`) and
    /// §15.1i measured the consequence: **TSan cannot model the happens-before
    /// edge through Metal's dispatch, so a report at this boundary survives
    /// `-Zbuild-std` and is still false.** Any result taken on this arm must
    /// state that, because a genuine ordering defect here would be
    /// indistinguishable from the false ones.
    pub fn notify_then_wait(&self, value: u64, timeout_ms: u64) -> u64 {
        let Some(listener) = self.listener.as_ref() else {
            // No listener: nothing will ever fire, so report the failure rather
            // than parking for the whole timeout. The caller blocks.
            return 0;
        };

        // The state the block writes and this thread reads. Cloned into the
        // block because Metal owns the block once it is registered and may
        // invoke it after this call -- or after this event is dropped.
        let shared = std::sync::Arc::clone(&self.shared);
        let block = block2::RcBlock::new(
            move |_event: std::ptr::NonNull<ProtocolObject<dyn MTLSharedEvent>>, seen: u64| {
                let (lock, cv) = &*shared;
                {
                    let mut st = lock.lock();
                    st.seen = st.seen.max(seen);
                    st.fired += 1;
                }
                // Woken under no lock: the waiter re-checks the predicate, so a
                // wake that races the unlock is harmless.
                cv.notify_all();
            },
        );

        let before = {
            let (lock, _) = &**self.shared;
            let st = lock.lock();
            st.fired
        };

        // SAFETY: `block` is a valid `RcBlock` for the duration of the call, and
        // `notifyListener:atValue:block:` copies it -- Metal retains what it
        // needs, which is why the block may safely outlive this stack frame and
        // why it is sound for it to fire after a timeout below.
        unsafe {
            self.raw
                .notifyListener_atValue_block(listener, value, &*block as *const _ as *mut _);
        }

        let (lock, cv) = &**self.shared;
        let deadline = std::time::Duration::from_millis(timeout_ms);
        let mut guard = lock.lock();
        // `wait_while_for` re-checks the predicate on every wake, so a
        // spurious wake does not return early and a block that fired *before*
        // this thread parked is not missed -- the predicate reads state the
        // block already wrote. `parking_lot` takes the guard by `&mut` and
        // returns the timeout as a plain result, so the poison arm std needed
        // here is gone; a timeout still falls through to the count below,
        // which reads 0 and sends the caller to the blocking wait.
        let _timed_out = cv.wait_while_for(&mut guard, |st| st.waiting_for(value), deadline);
        // The count this wait observed, and 0 when nothing arrived. Reported
        // rather than a bool for the reason in the doc comment above.
        guard.completed_notifications(value, before)
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
    fn older_listener_callback_cannot_complete_a_newer_target() {
        let mut state = NotifyState { seen: 7, fired: 2 };
        assert!(state.waiting_for(8));
        assert_eq!(state.completed_notifications(8, 1), 0);
        // Timeout after an old callback still requests the native fallback.
        state.seen = 8;
        state.fired = 3;
        assert!(!state.waiting_for(8));
        assert_eq!(state.completed_notifications(8, 1), 2);
        assert_eq!(state.completed_notifications(9, 1), 0);
    }

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
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-listener").unwrap(),
            CompletionWait::SharedEventListener {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS
            }
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-listener:7").unwrap(),
            CompletionWait::SharedEventListener { timeout_ms: 7 }
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

        // The listener arm shares the same prefix and widens the hazard from a
        // pair to a set of three (#583): `shared-event-listener` starts with
        // `shared-event`, and `shared-event-l...` is not distinguished from
        // `shared-event-w...` until the fourth character of the suffix.
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-listener").unwrap(),
            CompletionWait::SharedEventListener {
                timeout_ms: DEFAULT_WAIT_TIMEOUT_MS
            }
        );
        assert_eq!(
            CompletionWait::from_str_arm("shared-event-listener:5").unwrap(),
            CompletionWait::SharedEventListener { timeout_ms: 5 }
        );

        // **Every arm resolves to itself, over the whole set.** The pairwise
        // assertions above are absolute and still cannot see a parser that
        // mapped two *unrelated* prefixed names together; sweeping the set is
        // what closes that, and it grows with the enum rather than by hand.
        for arm in CompletionWait::ALL_ARMS {
            assert_eq!(
                CompletionWait::from_str_arm(arm.name()).unwrap().name(),
                arm.name(),
                "arm {} does not resolve to itself",
                arm.name()
            );
        }
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
        assert!(!CompletionWait::SharedEventListener { timeout_ms: 1 }.polls());
    }

    /// **`notifies()` and `polls()` are different axes**, and exactly one arm
    /// is woken by a callback (#583).
    ///
    /// Asserted per arm rather than as a shape: `Synchronize`,
    /// `SharedEventWait` and `SharedEventListener` all answer `false` to
    /// `polls()`, and reading that as *"they are the same"* is the collapse
    /// this axis exists to prevent -- they park in the Mach kernel, in the
    /// Metal runtime, and on a condvar a GCD worker signals, which are three
    /// mechanisms with three failure modes.
    #[test]
    fn exactly_the_listener_arm_is_notified() {
        assert!(!CompletionWait::Synchronize.notifies());
        assert!(!CompletionWait::SharedEvent.notifies());
        assert!(!CompletionWait::SharedEventSleep { micros: 1 }.notifies());
        assert!(!CompletionWait::SharedEventWait { timeout_ms: 1 }.notifies());
        assert!(CompletionWait::SharedEventListener { timeout_ms: 1 }.notifies());
        // The notified arm allocates a listener, and the bounded poll parks
        // through one; the incumbent and the two other async arms create
        // nothing extra.
        for arm in CompletionWait::ALL_ARMS {
            assert_eq!(
                arm.needs_listener(),
                arm.notifies() || arm == CompletionWait::SharedEvent,
                "arm {} disagrees about needing a listener",
                arm.name()
            );
        }
    }

    /// No two arms both poll and notify, and every async arm does exactly one
    /// of {poll, park, notify} (#583).
    ///
    /// The partition is what makes each arm's engagement counter readable: a
    /// polling arm proves itself by `polls > 0`, a notified arm by `fired > 0`,
    /// and an arm that did both would have no unambiguous proof.
    #[test]
    fn polling_and_notifying_are_exclusive() {
        for arm in CompletionWait::ALL_ARMS {
            assert!(
                !(arm.polls() && arm.notifies()),
                "arm {} both polls and notifies, so neither counter is its proof",
                arm.name()
            );
        }
        // The control: each property is true of something, or the assertion
        // above holds vacuously for an implementation returning false always.
        assert!(CompletionWait::ALL_ARMS.iter().any(|a| a.polls()));
        assert!(CompletionWait::ALL_ARMS.iter().any(|a| a.notifies()));
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
        let arms = CompletionWait::ALL_ARMS;
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

        // The listener arm takes the same three fail-safe cases (#583), and its
        // failure mode is a third one worth naming: with no signal encoded no
        // block is ever registered, so a condvar wait would sleep to the
        // timeout and *then* block -- a slow run rather than a hang, which is
        // the harder one to notice and the reason this is asserted rather than
        // left to the timeout.
        let notify = CompletionWait::SharedEventListener { timeout_ms: 1 };
        assert_eq!(notify.wait_route(false, false), Block, "{}", notify.name());
        assert_eq!(notify.wait_route(false, true), Block, "{}", notify.name());
        assert_eq!(notify.wait_route(true, false), Block, "{}", notify.name());
        assert_eq!(
            notify.wait_route(true, true),
            Notify { timeout_ms: 1 },
            "{}",
            notify.name()
        );
    }

    /// The notify route's non-vacuity control, carrying its timeout (#583).
    ///
    /// The same argument as `the_park_route_is_reachable_and_carries_its_timeout`
    /// one arm along: an implementation collapsing `Notify` into `Block` or into
    /// `Park` would satisfy every fail-safe assertion above, because those all
    /// expect `Block`. Only a positive assertion that the route is reached, with
    /// the timeout that reaches the condvar rather than the one on the enum, can
    /// see it.
    #[test]
    fn the_notify_route_is_reachable_and_carries_its_timeout() {
        assert_eq!(
            CompletionWait::SharedEventListener { timeout_ms: 4321 }.wait_route(true, true),
            WaitRoute::Notify { timeout_ms: 4321 }
        );
        assert_ne!(
            4321, DEFAULT_WAIT_TIMEOUT_MS,
            "the timeout above equals the default, so this cannot see a hard-coded one"
        );
    }

    /// **The three parking routes are distinct**, asserted directly (#583).
    ///
    /// `Block`, `Park` and `Notify` all end with the thread descheduled, which
    /// is exactly why collapsing them is tempting and why it must not happen:
    /// they differ in *what wakes the thread*, and only `Notify` has a fired
    /// count. A routing function mapping two of them together would leave every
    /// fail-safe test above passing while making one arm's engagement proof
    /// unobservable.
    #[test]
    fn the_three_parking_routes_are_distinct() {
        let block = CompletionWait::Synchronize.wait_route(true, true);
        let park = CompletionWait::SharedEventWait { timeout_ms: 7 }.wait_route(true, true);
        let notify = CompletionWait::SharedEventListener { timeout_ms: 7 }.wait_route(true, true);
        // Absolute references on all three, not "they differ": a function
        // mapping all three to a fourth route would satisfy pairwise inequality
        // (§3.1b rule 1, and the shape #583's own prefix test guards against).
        assert_eq!(block, WaitRoute::Block);
        assert_eq!(park, WaitRoute::Park { timeout_ms: 7 });
        assert_eq!(notify, WaitRoute::Notify { timeout_ms: 7 });
    }

    /// A notified arm must never route to a poll, whatever it is handed (#583).
    ///
    /// The counterpart of `a_parking_arm_never_routes_to_a_poll`, and it is what
    /// makes `completion_polls == 0` readable on this arm too. Stated over the
    /// whole four-case input domain.
    #[test]
    fn a_notified_arm_never_routes_to_a_poll() {
        let notify = CompletionWait::SharedEventListener {
            timeout_ms: DEFAULT_WAIT_TIMEOUT_MS,
        };
        for (has_event, has_signal) in [(false, false), (false, true), (true, false), (true, true)]
        {
            assert_ne!(
                notify.wait_route(has_event, has_signal),
                WaitRoute::Poll,
                "the listener arm polled with event={has_event} signal={has_signal}"
            );
        }
        // The control, as above: `Poll` is reachable for an arm that should
        // reach it.
        assert_eq!(
            CompletionWait::SharedEvent.wait_route(true, true),
            WaitRoute::Poll
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

    /// **`ALL_ARMS` must contain every variant**, checked against the compiler
    /// rather than against a second hand-written list (#583).
    ///
    /// A `const` array is the one enumeration a new variant does not break: it
    /// is a compile error at every `match` and **silent** here. The previous
    /// repair -- checking the array against a list of parser names -- compared
    /// two hand-written lists to each other and went stale on the next arm
    /// added, which is #519's *"looks like a comparison and is not"* (§3.1b).
    ///
    /// **The reference here is the compiler's**: the `match` below is
    /// exhaustive, so adding a variant fails to build until it is handled, and
    /// handling it means naming its discriminant. The test then asserts every
    /// named discriminant appears in `ALL_ARMS`. So the array cannot fall
    /// behind the enum without something failing -- a build error if the match
    /// is not updated, this test if the array is not.
    #[test]
    fn no_arm_is_missing_from_all_arms() {
        // The exhaustive match is the mechanism: a new variant does not compile
        // until it is added here, and adding it here is what puts it in `want`.
        fn discriminant(arm: &CompletionWait) -> &'static str {
            match arm {
                CompletionWait::Synchronize => "Synchronize",
                CompletionWait::SharedEvent => "SharedEvent",
                CompletionWait::SharedEventSleep { .. } => "SharedEventSleep",
                CompletionWait::SharedEventWait { .. } => "SharedEventWait",
                CompletionWait::SharedEventListener { .. } => "SharedEventListener",
            }
        }
        let want = [
            "Synchronize",
            "SharedEvent",
            "SharedEventSleep",
            "SharedEventWait",
            "SharedEventListener",
        ];
        let have: Vec<&str> = CompletionWait::ALL_ARMS.iter().map(discriminant).collect();
        for w in want {
            assert!(
                have.contains(&w),
                "{w} is a variant of CompletionWait and is absent from ALL_ARMS, \
                 so every sweep over ALL_ARMS silently skips it"
            );
        }
        assert_eq!(
            have.len(),
            want.len(),
            "ALL_ARMS has a duplicate or an extra member: {have:?}"
        );
    }

    /// Every arm in `ALL_ARMS` round-trips through the parser (#583).
    ///
    /// The other half of the registry property: `ALL_ARMS` is what sweeps
    /// iterate and `from_str_arm` is what a **user** can select, so an arm in
    /// one and not the other is either untestable from a flag or untested by
    /// every sweep. Checking the two against each other is sound *here* --
    /// unlike the stale test this replaces -- because neither is hand-written
    /// relative to the other: `ALL_ARMS` is pinned to the enum above and the
    /// parser is pinned to the names a `RESULT` line carries.
    #[test]
    fn every_arm_round_trips_through_its_name() {
        for arm in CompletionWait::ALL_ARMS {
            let parsed = CompletionWait::from_str_arm(arm.name()).unwrap_or_else(|e| {
                panic!(
                    "arm {} is in ALL_ARMS and does not parse from its own name: {e}",
                    arm.name()
                )
            });
            assert_eq!(
                parsed.name(),
                arm.name(),
                "arm {} parses to a different arm: {}",
                arm.name(),
                parsed.name()
            );
        }
        // The control: an unknown name is still an error, or the loop above
        // would pass for a parser that accepted everything (§3.1a rule 3).
        assert!(CompletionWait::from_str_arm("shared-event-park").is_err());
    }

    /// **Every parameterised arm's period reaches its rendered name** (#583).
    ///
    /// The defect this pins is §13.5m's, measured at `CutPolicy` and repeated
    /// at `CompletionWait` in #567: an arm rendered without its period compares
    /// **equal** to the same arm at a different period, so two runs the design
    /// separates are pooled by the store. §11.5o measures `sleep:50` against
    /// `sleep:200` at 15.0 against 3.9 polls/ms — two arms, not one.
    ///
    /// Asserted by **varying the period and requiring the rendering to move**,
    /// which is the only form that can see a `match` arm returning a constant.
    /// A test comparing `arm_with_period()` against a literal would pass on an
    /// implementation that ignored the field and happened to print the default.
    #[test]
    fn a_periods_arm_renders_its_period_and_a_bare_arm_does_not() {
        // Bare arms render bare, so the axis vector does not gain a spurious
        // field for an arm that has no period.
        assert_eq!(CompletionWait::Synchronize.arm_with_period(), "synchronize");
        assert_eq!(
            CompletionWait::SharedEvent.arm_with_period(),
            "shared-event"
        );
        assert_eq!(CompletionWait::Synchronize.period(), None);
        assert_eq!(CompletionWait::SharedEvent.period(), None);

        // Every parameterised arm: two distinct periods must render
        // differently, which is the property that stops the pooling.
        let cases: [(CompletionWait, CompletionWait, &str); 3] = [
            (
                CompletionWait::SharedEventSleep { micros: 50 },
                CompletionWait::SharedEventSleep { micros: 200 },
                "shared-event-sleep",
            ),
            (
                CompletionWait::SharedEventWait { timeout_ms: 50 },
                CompletionWait::SharedEventWait { timeout_ms: 200 },
                "shared-event-wait",
            ),
            (
                CompletionWait::SharedEventListener { timeout_ms: 50 },
                CompletionWait::SharedEventListener { timeout_ms: 200 },
                "shared-event-listener",
            ),
        ];
        for (lo, hi, name) in cases {
            // Absolute references on both sides, per §3.1b rule 1: an
            // implementation mapping both to a third string would satisfy a
            // bare `assert_ne!`.
            assert_eq!(lo.arm_with_period(), format!("{name}:50"));
            assert_eq!(hi.arm_with_period(), format!("{name}:200"));
            assert_eq!(lo.period(), Some(50));
            assert_eq!(hi.period(), Some(200));
            // And the bare name is still the arm's, so the period is an
            // addition rather than a replacement.
            assert_eq!(lo.name(), name);
        }
    }

    /// Every arm in `ALL_ARMS` renders a period iff it carries one (#583).
    ///
    /// The sweep that keeps [`CompletionWait::period`] honest as arms are
    /// added: a new parameterised arm falling into a `None` branch would be
    /// silently pooled with itself at every period, and this is what fails.
    #[test]
    fn the_period_sweep_agrees_with_the_rendering() {
        for arm in CompletionWait::ALL_ARMS {
            let rendered = arm.arm_with_period();
            match arm.period() {
                Some(p) => assert_eq!(
                    rendered,
                    format!("{}:{p}", arm.name()),
                    "arm {} has a period and does not render it",
                    arm.name()
                ),
                None => assert_eq!(
                    rendered,
                    arm.name(),
                    "arm {} has no period and renders one",
                    arm.name()
                ),
            }
        }
        // The control: at least one arm of each kind exists, or the loop above
        // holds vacuously (§3.1a rule 1).
        assert!(CompletionWait::ALL_ARMS
            .iter()
            .any(|a| a.period().is_some()));
        assert!(CompletionWait::ALL_ARMS
            .iter()
            .any(|a| a.period().is_none()));
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
        for (has_event, has_signal) in [(false, false), (false, true), (true, false), (true, true)]
        {
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
