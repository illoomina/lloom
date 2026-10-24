//! `server.keep_warm`: while no request runs, keep the GPU ready every
//! period — ask the OS to keep the residency set's memory wired
//! (`residency`) or run one one-element dispatch (`dispatch`) — until
//! `server.keep_warm_for_s` after the last request. macOS unwires a
//! residency set nothing used for a few seconds, and the request after it
//! waits while it is wired back (`docs/machine-model.md` §3.4d); the bound
//! gives the memory back to an idle server, and the next request starts the
//! heartbeat again. Each beat reads the kernel's memory-pressure level; at
//! warn or critical the heartbeat stops and ends the set's residency, so
//! the memory goes back to the machine that needs it, and at normal it
//! resumes.
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// What the heartbeat does each period.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum By {
    Residency,
    Dispatch,
}

impl By {
    fn name(self) -> &'static str {
        match self {
            By::Residency => "residency",
            By::Dispatch => "dispatch",
        }
    }
}

/// The kernel's memory-pressure level as a beat reads it
/// (`kern.memorystatus_vm_pressure_level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pressure {
    Normal,
    Warn,
    Critical,
    /// Unreadable, or a level the kernel does not document.
    Unknown,
}

impl Pressure {
    pub(super) fn of(level: u8) -> Self {
        match level {
            1 => Pressure::Normal,
            2 => Pressure::Warn,
            4 => Pressure::Critical,
            _ => Pressure::Unknown,
        }
    }
    fn code(self) -> u8 {
        match self {
            Pressure::Normal => 1,
            Pressure::Warn => 2,
            Pressure::Critical => 4,
            Pressure::Unknown => 0,
        }
    }
    /// Whether the heartbeat stands down: warn or critical. An unreadable
    /// level does not stop it.
    pub(super) fn elevated(self) -> bool {
        matches!(self, Pressure::Warn | Pressure::Critical)
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Pressure::Normal => "normal",
            Pressure::Warn => "warn",
            Pressure::Critical => "critical",
            Pressure::Unknown => "unknown",
        }
    }
}

/// The level a beat reads: the kernel's, 0 where it cannot be read.
pub(super) fn vm_pressure() -> u8 {
    lloom_tensor::machine::vm_pressure_level()
        .and_then(|level| u8::try_from(level).ok())
        .unwrap_or(0)
}

/// The file [`file_pressure`] reads (`server.keep_warm_pressure_file`), set
/// once before the heartbeat starts.
pub(super) static PRESSURE_FILE: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The level in [`PRESSURE_FILE`]: its first number, 0 when unreadable.
pub(super) fn file_pressure() -> u8 {
    PRESSURE_FILE
        .get()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0)
}

/// The level a heartbeat that ignores pressure reads: always normal.
pub(super) fn pressure_ignored() -> u8 {
    1
}

/// The keep-warm a server runs, resolved from the options and the loaded
/// owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Policy {
    Off {
        why: String,
    },
    On {
        period: Duration,
        /// How long after the last request it keeps going; `None` = no bound.
        window: Option<Duration>,
        by: By,
        /// Whether each beat reads the memory-pressure level and stands
        /// down at warn or critical (`server.keep_warm_pressure`).
        pressure: bool,
    },
}

impl Policy {
    /// `period_ms` (`server.keep_warm`; unset or 0 = off), `for_s`
    /// (`server.keep_warm_for_s`, 0 = no bound) and `by`
    /// (`server.keep_warm_by`) for an owner at `batch` sequences on a GPU
    /// (`gpu`). A period asked for at a batch is refused: the batch thread
    /// owns the device. On a device that is not a GPU there is nothing to
    /// keep warm, and the policy is off.
    pub(super) fn resolve(
        period_ms: Option<u64>,
        for_s: u64,
        by: &str,
        pressure: bool,
        batch: usize,
        gpu: bool,
    ) -> Result<Policy, String> {
        let off = |why: String| Ok(Policy::Off { why });
        let period_ms = match period_ms {
            Some(ms) if ms > 0 => ms,
            given => {
                return off(if batch > 1 {
                    format!("not built at --batch {batch}")
                } else if !gpu {
                    "the device is not a GPU".into()
                } else if given.is_some() {
                    "server.keep_warm = 0".into()
                } else {
                    "server.keep_warm unset".into()
                })
            }
        };
        if batch > 1 {
            return Err(format!(
                "server.keep_warm at --batch {batch} is not built: the batch thread owns the device; \
                 serve at --batch 1 or set server.keep_warm=0"
            ));
        }
        if !gpu {
            return off("the device is not a GPU".into());
        }
        Ok(Policy::On {
            period: Duration::from_millis(period_ms),
            window: (for_s > 0).then(|| Duration::from_secs(for_s)),
            by: if by == "dispatch" {
                By::Dispatch
            } else {
                By::Residency
            },
            pressure,
        })
    }

    /// The loaded line's field: `1000ms/180s/residency`, `1000ms/unbounded/…`
    /// (`…/pressure` when it stands down under memory pressure) or `off`.
    pub(super) fn short(&self) -> String {
        match self {
            Policy::Off { .. } => "off".into(),
            Policy::On {
                period,
                window,
                by,
                pressure,
            } => format!(
                "{}ms/{}/{}{}",
                period.as_millis(),
                window.map_or("unbounded".into(), |w| format!("{}s", w.as_secs())),
                by.name(),
                if *pressure { "/pressure" } else { "" }
            ),
        }
    }

    /// The operator's line.
    pub(super) fn sentence(&self) -> String {
        match self {
            Policy::Off { why } => format!("off: {why}"),
            Policy::On {
                period,
                window,
                by,
                pressure,
            } => format!(
                "{} every {} ms while idle, {}{}",
                match by {
                    By::Residency => "the residency set's memory requested resident",
                    By::Dispatch => "one GPU dispatch",
                },
                period.as_millis(),
                match window {
                    Some(w) => format!(
                        "for {} s after the last request, then stopped until the next",
                        w.as_secs()
                    ),
                    None => "with no bound".into(),
                },
                if *pressure {
                    "; stopped at memory pressure warn or critical, the set's residency ended, resumed at normal"
                } else {
                    "; memory pressure not read"
                }
            ),
        }
    }

    /// `/health`'s object: the policy, and the heartbeat's state now.
    pub(super) fn json(&self, activity: &Activity, now: u64) -> Value {
        let changed = activity.pressure_changed.load(Ordering::Acquire);
        let transition = (changed != 0).then(|| {
            json!({
                "from": Pressure::of(activity.pressure_from.load(Ordering::Acquire)).name(),
                "to": Pressure::of(activity.pressure.load(Ordering::Acquire)).name(),
                "ms_ago": now.saturating_sub(changed) / 1_000_000,
            })
        });
        let state = json!({
            "heartbeat": if activity.running.load(Ordering::Acquire) { "running" } else { "stopped" },
            "actions": activity.actions.load(Ordering::Relaxed),
            "windows_ended": activity.windows.load(Ordering::Acquire),
            "since_request_ms": now.saturating_sub(activity.last_request.load(Ordering::Acquire)) / 1_000_000,
            "pressure": Pressure::of(activity.pressure.load(Ordering::Acquire)).name(),
            "pressure_transition": transition,
            "pressure_transitions": activity.transitions.load(Ordering::Relaxed),
            "residency_ends": activity.ends.load(Ordering::Relaxed),
        });
        match self {
            Policy::Off { why } => json!({ "policy": "off", "why": why, "state": state }),
            Policy::On {
                period,
                window,
                by,
                pressure,
            } => json!({
                "policy": "on",
                "period_ms": period.as_millis() as u64,
                "for_s": window.map(|w| w.as_secs()),
                "by": by.name(),
                "pressure_stop": pressure,
                "state": state,
            }),
        }
    }
}

/// When this server last gave the GPU work, and the heartbeat's state:
/// written by every request thread at its end, read by the heartbeat.
#[derive(Default)]
pub(super) struct Activity {
    /// The last request's end or keep-warm action (`uptime_ns`): how long the
    /// GPU had nothing from this server when a request arrives.
    pub(super) last_active: AtomicU64,
    /// The last request's end; the load before the first. The window runs
    /// from here.
    pub(super) last_request: AtomicU64,
    /// Keep-warm actions taken.
    pub(super) actions: AtomicU64,
    /// Windows that ran out: each time the heartbeat stopped for want of a
    /// request.
    pub(super) windows: AtomicU64,
    /// Whether the heartbeat is inside its window.
    pub(super) running: AtomicBool,
    /// The memory-pressure level the last beat read ([`Pressure::code`]).
    pub(super) pressure: AtomicU8,
    /// The level before the last change, and when it changed
    /// (`uptime_ns`; 0 = never).
    pub(super) pressure_from: AtomicU8,
    pub(super) pressure_changed: AtomicU64,
    /// Level changes seen.
    pub(super) transitions: AtomicU64,
    /// Residency ends the heartbeat took under pressure.
    pub(super) ends: AtomicU64,
    stop: AtomicBool,
    beat: OnceLock<std::thread::Thread>,
}

impl Activity {
    pub(super) fn new(now: u64) -> Self {
        let a = Activity::default();
        a.last_active.store(now, Ordering::Relaxed);
        a.last_request.store(now, Ordering::Relaxed);
        a.pressure.store(Pressure::Normal.code(), Ordering::Relaxed);
        a
    }

    /// Records the level a beat read at `now`, logging a change once.
    fn note_pressure(&self, level: Pressure, now: u64) {
        let was = Pressure::of(self.pressure.swap(level.code(), Ordering::AcqRel));
        if was == level {
            return;
        }
        self.pressure_from.store(was.code(), Ordering::Release);
        self.pressure_changed.store(now.max(1), Ordering::Release);
        self.transitions.fetch_add(1, Ordering::Relaxed);
        let what = if level.elevated() {
            "residency requests stopped, the set's residency ended once the slot is free"
        } else if was.elevated() {
            "residency requests resume"
        } else {
            "no change to the heartbeat"
        };
        eprintln!(
            "[keep-warm: memory pressure {} (was {}): {what}]",
            level.name(),
            was.name()
        );
    }

    /// A request ended at `now`: the window starts again, and a stopped
    /// heartbeat with it.
    pub(super) fn request_ended(&self, now: u64) {
        self.last_active.store(now, Ordering::Release);
        self.last_request.store(now, Ordering::Release);
        if let Some(t) = self.beat.get() {
            t.unpark();
        }
    }
}

/// One period's work.
pub(super) enum Act {
    Done,
    /// A request holds or waits for the slot.
    Busy,
    Failed(String),
}

/// The heartbeat thread.
pub(super) struct Heartbeat {
    activity: Arc<Activity>,
    thread: std::thread::JoinHandle<()>,
}

/// What a heartbeat reads and does: the clock (`uptime_ns`), the
/// memory-pressure level ([`vm_pressure`]), the period's work, and the work
/// that gives the memory back when pressure rises.
pub(super) struct Beat<A, E> {
    pub clock: fn() -> u64,
    pub pressure: fn() -> u8,
    pub act: A,
    pub end: E,
}

impl Heartbeat {
    /// Every `period` that the GPU had nothing from this server, run `act`,
    /// until `window` after the last request; then park until a request
    /// ends. While the pressure level is warn or critical no `act` runs, and
    /// `end` runs once, and again after any request in the meantime.
    pub(super) fn spawn<A, E>(
        activity: Arc<Activity>,
        period: Duration,
        window: Option<Duration>,
        mut work: Beat<A, E>,
    ) -> std::io::Result<Heartbeat>
    where
        A: FnMut() -> Act + Send + 'static,
        E: FnMut() -> Act + Send + 'static,
    {
        let a = Arc::clone(&activity);
        let thread = std::thread::Builder::new()
            .name("keep-warm".into())
            .spawn(move || {
                beat(
                    &a,
                    period,
                    window,
                    (work.clock, work.pressure),
                    &mut work.act,
                    &mut work.end,
                )
            })?;
        let _ = activity.beat.set(thread.thread().clone());
        Ok(Heartbeat { activity, thread })
    }

    pub(super) fn stop(self) {
        self.activity.stop.store(true, Ordering::Release);
        self.thread.thread().unpark();
        let _ = self.thread.join();
    }
}

fn beat(
    a: &Activity,
    period: Duration,
    window: Option<Duration>,
    (clock, pressure): (fn() -> u64, fn() -> u8),
    act: &mut dyn FnMut() -> Act,
    end: &mut dyn FnMut() -> Act,
) {
    let period_ns = period.as_nanos() as u64;
    let window_ns = window.map(|w| w.as_nanos() as u64);
    // The time left in the window at `now`; `None` once it has run out.
    let left = |now: u64| -> Option<u64> {
        let since = now.saturating_sub(a.last_request.load(Ordering::Acquire));
        match window_ns {
            None => Some(u64::MAX),
            Some(w) => (since < w).then(|| w - since),
        }
    };
    // When the residency was last ended under pressure.
    let mut ended: Option<u64> = None;
    while !a.stop.load(Ordering::Acquire) {
        let Some(rest) = left(clock()) else {
            if a.running.swap(false, Ordering::AcqRel) {
                a.windows.fetch_add(1, Ordering::Release);
            }
            std::thread::park();
            continue;
        };
        a.running.store(true, Ordering::Release);
        std::thread::park_timeout(period.min(Duration::from_nanos(rest)));
        if a.stop.load(Ordering::Acquire) {
            break;
        }
        let now = clock();
        if left(now).is_none() {
            continue;
        }
        let level = Pressure::of(pressure());
        a.note_pressure(level, now);
        if level.elevated() {
            // Once per spell of pressure, and again after a request: its
            // command buffers made the set resident again.
            if ended.is_none_or(|t| a.last_active.load(Ordering::Acquire) > t) {
                match end() {
                    Act::Done => {
                        ended = Some(clock());
                        a.ends.fetch_add(1, Ordering::Relaxed);
                    }
                    Act::Busy => {}
                    Act::Failed(e) => {
                        eprintln!("[keep-warm: stopped: {e}]");
                        break;
                    }
                }
            }
            continue;
        }
        ended = None;
        if now.saturating_sub(a.last_active.load(Ordering::Acquire)) < period_ns {
            continue;
        }
        match act() {
            Act::Done => {
                a.last_active.store(clock(), Ordering::Release);
                a.actions.fetch_add(1, Ordering::Relaxed);
            }
            Act::Busy => {}
            Act::Failed(e) => {
                eprintln!("[keep-warm: stopped: {e}]");
                break;
            }
        }
    }
    a.running.store(false, Ordering::Release);
}

#[cfg(test)]
#[path = "keep_warm_tests.rs"]
mod tests;
