//! Values that outlive a served pass undeclared (`pass-plan-design.md` §26).
//!
//! Every served pass hands out slots of one base that all keys share, so a
//! value a served pass produced is valid only until a later pass hands its
//! bytes on. A value still referenced when a later pass opens has escaped its
//! pass unless its producer took it out of the base (an `EscapeGuard`, a
//! hold). Nothing else connects the arena's allocation order to the driver's
//! read order, so the check is made here, in two halves:
//!
//! - **at handout, in every build**: a slot whose bytes a carried value still
//!   holds is declined. The allocation takes the pool, the carried value is
//!   never overwritten, and the decline is counted (`undeclared_escapes`) and
//!   named once per producing `(key, ordinal)`. A silent fallback would hide
//!   the site to fix.
//! - **at use, in debug and `profile` builds**: a bind of a carried value
//!   after a later pass began is the escape itself, whatever the layouts. It
//!   names the consuming site, and does not depend on whether two plans'
//!   offsets happen to collide.
//!
//! What a hit does is [`EscapeCheck`]: counted and warned once, reported with
//! its sites, or a hard error.

use super::{Arena, Buffer, PassKey};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// What an undeclared escape does when it is found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EscapeCheck {
    /// Counted, and named once per producing `(key, ordinal)` on stderr. The
    /// shipping build's mode.
    Count,
    /// As `Count`, plus every hit's sites on stderr and in [`escape_hits`].
    /// The `profile` build's mode, and the census's.
    Report,
    /// A panic naming the producer, the pass that superseded it and the
    /// consuming site. Debug builds and every test.
    Error,
}

/// 0 = not yet read, else `EscapeCheck as u8 + 1`.
static MODE: AtomicU8 = AtomicU8::new(0);

impl EscapeCheck {
    fn from_bits(b: u8) -> Self {
        match b {
            1 => Self::Count,
            2 => Self::Report,
            _ => Self::Error,
        }
    }

    fn bits(self) -> u8 {
        match self {
            Self::Count => 1,
            Self::Report => 2,
            Self::Error => 3,
        }
    }

    /// The build's mode: a hard error where debug assertions are on, the
    /// report under `profile`, the count otherwise.
    pub fn for_build() -> Self {
        if cfg!(debug_assertions) {
            Self::Error
        } else if cfg!(feature = "profile") {
            Self::Report
        } else {
            Self::Count
        }
    }
}

/// The mode in force: `LLOOM_ESCAPE_CHECK=count|report|error` (read once),
/// else [`EscapeCheck::for_build`]. A report or an error asked for in a build
/// without the use check still checks every handout.
pub fn escape_check() -> EscapeCheck {
    match MODE.load(Ordering::Relaxed) {
        0 => {
            let mode = match std::env::var("LLOOM_ESCAPE_CHECK").ok().as_deref() {
                Some("count") => EscapeCheck::Count,
                Some("report") => EscapeCheck::Report,
                Some("error") => EscapeCheck::Error,
                _ => EscapeCheck::for_build(),
            };
            MODE.store(mode.bits(), Ordering::Relaxed);
            mode
        }
        b => EscapeCheck::from_bits(b),
    }
}

/// Set the mode for the process (a test's switch; the census runs `Report`).
pub fn set_escape_check(mode: EscapeCheck) {
    MODE.store(mode.bits(), Ordering::Relaxed);
}

/// A served value still referenced after its pass: the pass that produced
/// it, its ordinal there (the last one served its slot), and its bytes in
/// the base.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeldValue {
    pub key: PassKey,
    pub ordinal: usize,
    pub offset: usize,
    pub size: usize,
}

/// What one carried value cost while it was carried, and whether it was used.
#[derive(Debug, Default)]
pub struct CarriedStats {
    /// Handouts declined because they would have overwritten it.
    pub declines: AtomicU64,
    /// Binds of it after a later pass began (debug and `profile` builds).
    pub reads: AtomicU64,
    /// The first declined allocation's site and the pass it was made in.
    #[cfg(any(debug_assertions, feature = "profile"))]
    first_decline: Mutex<Option<String>>,
}

impl CarriedStats {
    fn first_decline(&self) -> String {
        #[cfg(any(debug_assertions, feature = "profile"))]
        if let Some(s) = self.first_decline.lock().as_ref() {
            return s.clone();
        }
        "(not recorded in this build)".to_string()
    }
}

/// One entry of a served arena's watch for its step: its slot `slot`
/// overlaps the bytes of `held`, carried in from an earlier pass, which the
/// slot handle `victim_slot` of `victim` still names.
pub struct ArenaWatch {
    pub slot: usize,
    pub victim: Arena,
    pub victim_slot: usize,
    pub held: HeldValue,
    /// The pass this arena is serving.
    pub by: PassKey,
    pub stats: Arc<CarriedStats>,
    pub log: Arc<EscapeLog>,
}

/// One hit, as the report lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EscapeHit {
    pub kind: EscapeKind,
    pub held: HeldValue,
    /// The pass that superseded the held value.
    pub by: PassKey,
    /// The site of the declined allocation or of the use.
    pub site: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EscapeKind {
    /// A later pass needed its bytes and was declined them.
    Declined,
    /// Bound after a later pass began.
    Used,
    /// Dropped unused, after costing declines: a hold to move earlier.
    HeldUnused,
}

/// The plan cache's record of undeclared escapes. Shared by the cache and
/// the watches it installs, so a decline is counted without the planner lock.
#[derive(Debug, Default)]
pub struct EscapeLog {
    undeclared: AtomicU64,
    used: AtomicU64,
    held_unused: AtomicU64,
    warned: Mutex<HashSet<(PassKey, usize)>>,
    hits: Mutex<Vec<EscapeHit>>,
}

impl EscapeLog {
    /// Handouts declined because a carried value held their bytes.
    pub fn undeclared_escapes(&self) -> u64 {
        self.undeclared.load(Ordering::Relaxed)
    }

    /// Carried values bound after a later pass began.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Carried values dropped unused after costing declines.
    pub fn held_unused(&self) -> u64 {
        self.held_unused.load(Ordering::Relaxed)
    }

    /// The producers named so far, one per `(key, ordinal)`.
    pub fn named(&self) -> Vec<(PassKey, usize)> {
        let mut v: Vec<_> = self.warned.lock().iter().copied().collect();
        v.sort_by_key(|(k, o)| (format!("{k:?}"), *o));
        v
    }

    /// Every hit recorded under `Report` or `Error`, in order. Taking
    /// clears it.
    pub fn take_hits(&self) -> Vec<EscapeHit> {
        std::mem::take(&mut *self.hits.lock())
    }

    fn warn_once(&self, held: &HeldValue, by: PassKey, what: &str) {
        if self.warned.lock().insert((held.key, held.ordinal)) {
            eprintln!(
                "[pass plans] undeclared escape: ordinal {} of {} ({} B at {}) is still \
                 referenced after its pass; {} {what}. Declare it at its producer, drop it \
                 before the next pass, or place it outside the base (pass-plan-design.md §26).",
                held.ordinal,
                key_name(&held.key),
                held.size,
                held.offset,
                key_name(&by),
            );
        }
    }

    fn record(&self, hit: EscapeHit) {
        if escape_check() != EscapeCheck::Count {
            eprintln!(
                "[pass plans] {:?}: ordinal {} of {} ({} B at {}), superseded by {}, at {}",
                hit.kind,
                hit.held.ordinal,
                key_name(&hit.held.key),
                hit.held.size,
                hit.held.offset,
                key_name(&hit.by),
                hit.site,
            );
            self.hits.lock().push(hit);
        }
    }

    /// A served handout was declined for `w`'s carried value. Called from the
    /// arena on the declining path only.
    pub(crate) fn note_decline(&self, w: &ArenaWatch) {
        self.undeclared.fetch_add(1, Ordering::Relaxed);
        let first = w.stats.declines.fetch_add(1, Ordering::Relaxed) == 0;
        self.warn_once(&w.held, w.by, "needed its bytes and took the pool");
        if escape_check() == EscapeCheck::Count {
            return;
        }
        let site = site();
        #[cfg(any(debug_assertions, feature = "profile"))]
        if first {
            *w.stats.first_decline.lock() = Some(format!("{site} (in {})", key_name(&w.by)));
        }
        let _ = first;
        self.record(EscapeHit {
            kind: EscapeKind::Declined,
            held: w.held,
            by: w.by,
            site,
        });
    }

    /// A carried value died. If it cost declines and was never used, it was
    /// a hold to move before the pass; under `Error` that is a hard error.
    pub(crate) fn note_dropped(&self, held: &HeldValue, by: PassKey, stats: &CarriedStats) {
        let declines = stats.declines.load(Ordering::Relaxed);
        if declines == 0 || stats.reads.load(Ordering::Relaxed) > 0 {
            return;
        }
        self.held_unused.fetch_add(1, Ordering::Relaxed);
        let first = stats.first_decline();
        let message = format!(
            "ordinal {} of {} ({} B at {}) was held across {} and dropped unused, after \
             {declines} declined allocation(s), first at {first}. Drop it before the next pass \
             (pass-plan-design.md §26).",
            held.ordinal,
            key_name(&held.key),
            held.size,
            held.offset,
            key_name(&by),
        );
        match escape_check() {
            EscapeCheck::Count => {}
            EscapeCheck::Report => self.record(EscapeHit {
                kind: EscapeKind::HeldUnused,
                held: *held,
                by,
                site: first,
            }),
            EscapeCheck::Error => panic!("[pass plans] undeclared escape (held): {message}"),
        }
    }
}

/// A key as the plan report names it: the kind, then the model's variant
/// and a model-opened pass's mark when they are not the default.
pub fn key_name(k: &PassKey) -> String {
    let mut name = format!("{:?}", k.kind);
    if k.variant != 0 {
        name.push_str(&format!("#v{}", k.variant));
    }
    if k.opener == super::PassOpener::Model {
        name.push_str("@model");
    }
    name
}

/// The site of the current call, as the served-request instrument names it.
fn site() -> String {
    #[cfg(any(debug_assertions, feature = "profile"))]
    {
        crate::metal_backend::call_site()
    }
    #[cfg(not(any(debug_assertions, feature = "profile")))]
    {
        "(no backtrace in this build)".to_string()
    }
}

/// The carried values the use check watches: `(base, start, end)` with
/// what to charge a use to. Process-wide like the arena's other globals,
/// because binds are seen in the encoder and the cache lives on the device.
#[cfg(any(debug_assertions, feature = "profile"))]
struct Stale {
    base: usize,
    start: usize,
    end: usize,
    held: HeldValue,
    by: PassKey,
    stats: Arc<CarriedStats>,
    log: Arc<EscapeLog>,
}

#[cfg(any(debug_assertions, feature = "profile"))]
static STALE: Mutex<Vec<Stale>> = Mutex::new(Vec::new());
#[cfg(any(debug_assertions, feature = "profile"))]
static STALE_WATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A carried value the use check watches from now on.
pub struct StaleValue {
    pub base: usize,
    pub held: HeldValue,
    pub by: PassKey,
    pub stats: Arc<CarriedStats>,
    pub log: Arc<EscapeLog>,
}

/// Replace the watched set with `values` (the plan cache's carried values,
/// at a pass's opening). A no-op in a build without the use check.
pub fn watch_stale(values: Vec<StaleValue>) {
    #[cfg(any(debug_assertions, feature = "profile"))]
    {
        let mut g = STALE.lock();
        g.clear();
        g.extend(values.into_iter().map(|v| Stale {
            base: v.base,
            start: v.held.offset,
            end: v.held.offset + v.held.size,
            held: v.held,
            by: v.by,
            stats: v.stats,
            log: v.log,
        }));
        STALE_WATCHING.store(!g.is_empty(), Ordering::Relaxed);
    }
    #[cfg(not(any(debug_assertions, feature = "profile")))]
    drop(values);
}

/// A dispatch or a blit binds `buf`. If its bytes are a carried value's, the
/// value is used after a later pass began: an undeclared escape. One relaxed
/// load when nothing is carried; compiled out of the shipping build.
#[inline]
pub fn note_use(buf: &Buffer) {
    #[cfg(any(debug_assertions, feature = "profile"))]
    {
        if !STALE_WATCHING.load(Ordering::Relaxed) {
            return;
        }
        note_use_slow(buf.raw_addr(), buf.base_offset(), buf.length());
    }
    #[cfg(not(any(debug_assertions, feature = "profile")))]
    let _ = buf;
}

#[cfg(any(debug_assertions, feature = "profile"))]
#[cold]
fn note_use_slow(base: usize, start: usize, len: usize) {
    let end = start + len;
    let hit = {
        let g = STALE.lock();
        g.iter()
            .find(|s| s.base == base && s.start < end && start < s.end)
            .map(|s| (s.held, s.by, s.stats.clone(), s.log.clone()))
    };
    let Some((held, by, stats, log)) = hit else {
        return;
    };
    stats.reads.fetch_add(1, Ordering::Relaxed);
    log.used.fetch_add(1, Ordering::Relaxed);
    log.warn_once(&held, by, "began after it, and it was then used");
    let site = site();
    match escape_check() {
        EscapeCheck::Count => {}
        EscapeCheck::Report => log.record(EscapeHit {
            kind: EscapeKind::Used,
            held,
            by,
            site,
        }),
        EscapeCheck::Error => panic!(
            "[pass plans] undeclared escape (used): ordinal {} of {} ({} B at {}) is used at \
             {site} after {} began on the same base; it cost {} declined allocation(s), \
             first at {}. Declare it at its producer or copy it out before the next pass \
             (pass-plan-design.md §26).",
            held.ordinal,
            key_name(&held.key),
            held.size,
            held.offset,
            key_name(&by),
            stats.declines.load(Ordering::Relaxed),
            stats.first_decline(),
        ),
    }
}
