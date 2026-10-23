//! `server.keep_warm`: while no request runs, keep the GPU ready every
//! period — ask the OS to keep the residency set's memory wired
//! (`residency`) or run one one-element dispatch (`dispatch`) — until
//! `server.keep_warm_for_s` after the last request. macOS unwires a
//! residency set nothing used for a few seconds, and the request after it
//! waits while it is wired back (`docs/machine-model.md` §3.4d); the bound
//! gives the memory back to an idle server, and the next request starts the
//! heartbeat again.
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
        })
    }

    /// The loaded line's field: `1000ms/180s/residency`, `1000ms/unbounded/…`
    /// or `off`.
    pub(super) fn short(&self) -> String {
        match self {
            Policy::Off { .. } => "off".into(),
            Policy::On { period, window, by } => format!(
                "{}ms/{}/{}",
                period.as_millis(),
                window.map_or("unbounded".into(), |w| format!("{}s", w.as_secs())),
                by.name()
            ),
        }
    }

    /// The operator's line.
    pub(super) fn sentence(&self) -> String {
        match self {
            Policy::Off { why } => format!("off: {why}"),
            Policy::On { period, window, by } => format!(
                "{} every {} ms while idle, {}",
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
                }
            ),
        }
    }

    /// `/health`'s object: the policy, and the heartbeat's state now.
    pub(super) fn json(&self, activity: &Activity, now: u64) -> Value {
        let state = json!({
            "heartbeat": if activity.running.load(Ordering::Acquire) { "running" } else { "stopped" },
            "actions": activity.actions.load(Ordering::Relaxed),
            "windows_ended": activity.windows.load(Ordering::Acquire),
            "since_request_ms": now.saturating_sub(activity.last_request.load(Ordering::Acquire)) / 1_000_000,
        });
        match self {
            Policy::Off { why } => json!({ "policy": "off", "why": why, "state": state }),
            Policy::On { period, window, by } => json!({
                "policy": "on",
                "period_ms": period.as_millis() as u64,
                "for_s": window.map(|w| w.as_secs()),
                "by": by.name(),
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
    stop: AtomicBool,
    beat: OnceLock<std::thread::Thread>,
}

impl Activity {
    pub(super) fn new(now: u64) -> Self {
        let a = Activity::default();
        a.last_active.store(now, Ordering::Relaxed);
        a.last_request.store(now, Ordering::Relaxed);
        a
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

impl Heartbeat {
    /// Every `period` that the GPU had nothing from this server, run `act`,
    /// until `window` after the last request; then park until a request
    /// ends. `clock` is `uptime_ns`.
    pub(super) fn spawn(
        activity: Arc<Activity>,
        period: Duration,
        window: Option<Duration>,
        clock: fn() -> u64,
        mut act: impl FnMut() -> Act + Send + 'static,
    ) -> std::io::Result<Heartbeat> {
        let a = Arc::clone(&activity);
        let thread = std::thread::Builder::new()
            .name("keep-warm".into())
            .spawn(move || beat(&a, period, window, clock, &mut act))?;
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
    clock: fn() -> u64,
    act: &mut dyn FnMut() -> Act,
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
        if left(now).is_none()
            || now.saturating_sub(a.last_active.load(Ordering::Acquire)) < period_ns
        {
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
