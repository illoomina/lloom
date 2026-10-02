//! The decode thread's scheduling class, claimed for the length of a run.
//!
//! The hot thread does ~1 ms of encode and then waits ~15 ms for the GPU to
//! publish a token, five hundred times a second (`docs/dispatch.md` §11.5y).
//! When it is not scheduled promptly after the publish, the next forward is
//! committed late and the GPU idles -- measured as a *starved* gap of
//! 100–380 ms on a loaded machine (§11.5z: the host thread was simply not
//! running), never as a per-step cost. Every production caller runs the loop
//! on a thread it did not tune for that: `lfm2-chat`'s main thread and
//! ambrogio's `spawn_blocking` pool thread both carry `QOS_CLASS_DEFAULT`,
//! while candle's own rayon workers already run at `USER_INTERACTIVE`
//! (`lloom-tensor/src/utils.rs`). So the loop claims the class itself.
//!
//! **Where it lives and why**: the entry points of `Session::decode*` -- the
//! loop knows it is the hot thread; a harness does not know which thread the
//! engine will run on. **What it costs**: two syscalls per generation
//! (claim, restore), none per step. **What it restores**: the class the
//! thread had, on drop -- a pool thread reused for other work is left as it
//! was found. **Arm**: `LLOOM_HOT_THREAD=off` leaves the thread alone, for
//! the A/B; the default is `interactive`.
//!
//! **What it is not**: a real-time policy (`THREAD_TIME_CONSTRAINT_POLICY`).
//! The wait is a poll (`docs/dispatch.md` §11.5o), and a real-time thread
//! that spins past its budget is demoted by the kernel; the QoS class is the
//! arm that cannot make things worse. Off macOS every call is a no-op and
//! the report says so.

/// A thread's QoS class, as `pthread_get_qos_class_np` reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum QosClass {
    /// Not read: off macOS, or the read failed.
    #[default]
    Unknown,
    UserInteractive,
    UserInitiated,
    Default,
    Utility,
    Background,
    Unspecified,
}

impl std::str::FromStr for QosClass {
    type Err = String;

    /// The spellings [`QosClass::as_str`] renders, for a knob that names a
    /// class; `unknown` and `unspecified` are not classes a thread is put in.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user-interactive" | "interactive" => Ok(QosClass::UserInteractive),
            "user-initiated" | "initiated" => Ok(QosClass::UserInitiated),
            "default" => Ok(QosClass::Default),
            "utility" => Ok(QosClass::Utility),
            "background" => Ok(QosClass::Background),
            other => Err(format!(
                "qos class {other:?}: expected user-interactive, user-initiated, default, \
                 utility or background"
            )),
        }
    }
}

/// The calling thread's class.
pub fn current() -> QosClass {
    #[cfg(target_os = "macos")]
    {
        macos::class_of(macos::get().0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        QosClass::Unknown
    }
}

/// Put the calling thread in `class` for the rest of its life and return the
/// class it then reads: for a thread the engine spawned and owns, which has
/// nothing to restore (`docs/invariants.md` §15.2 #10b). `Err(errno)` when
/// the kernel refuses; off macOS a no-op reading [`QosClass::Unknown`].
pub fn set_current(class: QosClass) -> Result<QosClass, i32> {
    #[cfg(target_os = "macos")]
    {
        let raw = match class {
            QosClass::UserInteractive => libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
            QosClass::UserInitiated => libc::qos_class_t::QOS_CLASS_USER_INITIATED,
            QosClass::Default => libc::qos_class_t::QOS_CLASS_DEFAULT,
            QosClass::Utility => libc::qos_class_t::QOS_CLASS_UTILITY,
            QosClass::Background => libc::qos_class_t::QOS_CLASS_BACKGROUND,
            QosClass::Unknown | QosClass::Unspecified => return Err(libc::EINVAL),
        };
        macos::set(raw, 0)?;
        Ok(current())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = class;
        Ok(QosClass::Unknown)
    }
}

/// A spawned thread's class, from a knob: `inherit` leaves the thread as
/// spawned, any class [`QosClass::from_str`] reads is claimed. Returns the
/// class the thread then reads and the errno of a refused claim (0 if none).
pub fn claim_for_spawned(arm: &str) -> (QosClass, i32) {
    if arm.trim().eq_ignore_ascii_case("inherit") {
        return (current(), 0);
    }
    match arm.parse::<QosClass>() {
        Ok(class) => match set_current(class) {
            Ok(after) => (after, 0),
            Err(e) => (current(), e),
        },
        // The knob's parser refuses a name that is not a class.
        Err(_) => (current(), -1),
    }
}

/// The knob's parser: `inherit` or a class [`QosClass::from_str`] reads.
pub fn parse_spawned_arm(v: &str) -> Result<String, String> {
    let s = v.trim().to_ascii_lowercase();
    if s == "inherit" {
        return Ok(s);
    }
    s.parse::<QosClass>().map(|c| c.as_str().to_string())
}

impl QosClass {
    /// The inverse of `class as u8`, for a class kept in an atomic.
    pub fn from_index(i: u8) -> Self {
        match i {
            1 => QosClass::UserInteractive,
            2 => QosClass::UserInitiated,
            3 => QosClass::Default,
            4 => QosClass::Utility,
            5 => QosClass::Background,
            6 => QosClass::Unspecified,
            _ => QosClass::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            QosClass::Unknown => "unknown",
            QosClass::UserInteractive => "user-interactive",
            QosClass::UserInitiated => "user-initiated",
            QosClass::Default => "default",
            QosClass::Utility => "utility",
            QosClass::Background => "background",
            QosClass::Unspecified => "unspecified",
        }
    }
}

/// Which arm the claim took, from `LLOOM_HOT_THREAD`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HotThreadArm {
    /// Claim `QOS_CLASS_USER_INTERACTIVE` for the run (the default).
    #[default]
    Interactive,
    /// Leave the thread as found (`LLOOM_HOT_THREAD=off`).
    Off,
    /// Hold the thread at `QOS_CLASS_DEFAULT` for the run
    /// (`LLOOM_HOT_THREAD=default`) -- the class a spawned thread has, i.e.
    /// ambrogio's `spawn_blocking` worker without this claim. A measurement
    /// arm: it lets a harness whose main thread is already interactive (every
    /// process's main thread is) run the production shape for an A/B.
    Default,
}

/// What the claim did, so a harness can show it engaged rather than assert
/// that a flag was passed (`CONTRIBUTING.md` §2.4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HotThreadReport {
    pub arm: HotThreadArm,
    /// The class the thread had when the run started.
    pub before: QosClass,
    /// The class it has for the run: `before` on the `Off` arm, after a
    /// failed set, or off macOS.
    pub after: QosClass,
    /// `errno` from `pthread_set_qos_class_self_np` when it refused; 0
    /// otherwise. A thread under a fixed-priority or real-time policy refuses
    /// with `EPERM`.
    pub set_errno: i32,
}

impl HotThreadReport {
    /// The run's thread is at `USER_INTERACTIVE` because of this claim.
    pub fn engaged(&self) -> bool {
        self.arm == HotThreadArm::Interactive
            && self.after == QosClass::UserInteractive
            && self.before != QosClass::UserInteractive
    }
}

/// The arm, read once from `LLOOM_HOT_THREAD`.
pub fn arm() -> HotThreadArm {
    static ARM: std::sync::OnceLock<HotThreadArm> = std::sync::OnceLock::new();
    *ARM.get_or_init(|| match crate::tuning::tuning().ring.hot_thread.as_str() {
        "off" | "0" | "none" => HotThreadArm::Off,
        "default" | "demote" => HotThreadArm::Default,
        _ => HotThreadArm::Interactive,
    })
}

/// The claim: holds the thread at `USER_INTERACTIVE` until dropped.
#[derive(Debug)]
pub struct HotThread {
    report: HotThreadReport,
    #[cfg(target_os = "macos")]
    restore: Option<(libc::qos_class_t, libc::c_int)>,
}

impl HotThread {
    /// Claim the current thread for a decode run.
    pub fn claim() -> Self {
        let arm = arm();
        #[cfg(target_os = "macos")]
        {
            let (before_raw, relpri) = macos::get();
            let before = macos::class_of(before_raw);
            let mut report = HotThreadReport {
                arm,
                before,
                after: before,
                set_errno: 0,
            };
            let mut restore = None;
            let want = match arm {
                HotThreadArm::Interactive => Some((
                    libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
                    QosClass::UserInteractive,
                )),
                HotThreadArm::Default => {
                    Some((libc::qos_class_t::QOS_CLASS_DEFAULT, QosClass::Default))
                }
                HotThreadArm::Off => None,
            };
            if let Some((raw, class)) = want {
                if before != class {
                    match macos::set(raw, 0) {
                        Ok(()) => {
                            report.after = macos::class_of(macos::get().0);
                            restore = before_raw.map(|raw| (raw, relpri));
                        }
                        Err(e) => report.set_errno = e,
                    }
                }
            }
            Self { report, restore }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self {
                report: HotThreadReport {
                    arm,
                    ..HotThreadReport::default()
                },
            }
        }
    }

    pub fn report(&self) -> HotThreadReport {
        self.report
    }
}

impl Drop for HotThread {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some((raw, relpri)) = self.restore.take() {
            // Best effort: a thread that could be raised can be put back, and
            // a failure here leaves it at the higher class, which is the safe
            // side.
            let _ = macos::set(raw, relpri);
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::QosClass;
    use libc::qos_class_t;

    /// The current thread's class and relative priority; `None` when the
    /// read fails.
    pub fn get() -> (Option<qos_class_t>, libc::c_int) {
        let mut class = qos_class_t::QOS_CLASS_UNSPECIFIED;
        let mut relpri: libc::c_int = 0;
        // SAFETY: `pthread_self` is always a valid thread; the two out
        // pointers are to live locals.
        let rc = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relpri)
        };
        if rc == 0 {
            (Some(class), relpri)
        } else {
            (None, 0)
        }
    }

    pub fn set(class: qos_class_t, relpri: libc::c_int) -> Result<(), i32> {
        // SAFETY: plain FFI on the calling thread; no pointers.
        let rc = unsafe { libc::pthread_set_qos_class_self_np(class, relpri) };
        if rc == 0 {
            Ok(())
        } else {
            // Documented to return the error code directly, `errno` also set.
            let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(rc);
            Err(if e == 0 { rc } else { e })
        }
    }

    pub fn class_of(raw: Option<qos_class_t>) -> QosClass {
        match raw {
            None => QosClass::Unknown,
            Some(qos_class_t::QOS_CLASS_USER_INTERACTIVE) => QosClass::UserInteractive,
            Some(qos_class_t::QOS_CLASS_USER_INITIATED) => QosClass::UserInitiated,
            Some(qos_class_t::QOS_CLASS_DEFAULT) => QosClass::Default,
            Some(qos_class_t::QOS_CLASS_UTILITY) => QosClass::Utility,
            Some(qos_class_t::QOS_CLASS_BACKGROUND) => QosClass::Background,
            Some(qos_class_t::QOS_CLASS_UNSPECIFIED) => QosClass::Unspecified,
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// The claim raises a fresh thread's class and the drop puts it back --
    /// on a thread of our own, so the test never touches the harness's.
    #[test]
    fn the_claim_raises_and_the_drop_restores() {
        std::thread::spawn(|| {
            let start = macos::class_of(macos::get().0);
            assert_ne!(
                start,
                QosClass::UserInteractive,
                "a fresh std thread is not interactive"
            );
            let report = {
                let claim = HotThread::claim();
                let r = claim.report();
                assert_eq!(r.arm, HotThreadArm::Interactive);
                assert_eq!(r.before, start);
                assert_eq!(r.after, QosClass::UserInteractive, "errno {}", r.set_errno);
                assert!(r.engaged());
                assert_eq!(macos::class_of(macos::get().0), QosClass::UserInteractive);
                r
            };
            assert_eq!(
                macos::class_of(macos::get().0),
                report.before,
                "restored on drop"
            );
        })
        .join()
        .unwrap();
    }

    /// A thread `std` spawns does not take its spawner's class: it reads
    /// `default` even when spawned from an interactive thread, so an engine
    /// thread is at `default` unless it claims a class (§15.2 #10b).
    #[test]
    fn a_spawned_thread_does_not_inherit_its_spawners_class() {
        std::thread::spawn(|| {
            macos::set(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0).unwrap();
            assert_eq!(current(), QosClass::UserInteractive);
            let child = std::thread::spawn(current).join().unwrap();
            assert_eq!(child, QosClass::Default, "the spawned thread's class");
        })
        .join()
        .unwrap();
    }

    /// `claim_for_spawned` claims what the knob names and reports what the
    /// thread then reads; `inherit` claims nothing.
    #[test]
    fn a_spawned_thread_claims_the_class_its_knob_names() {
        let read = |arm: &'static str| std::thread::spawn(move || claim_for_spawned(arm));
        assert_eq!(
            read("user-interactive").join().unwrap(),
            (QosClass::UserInteractive, 0)
        );
        assert_eq!(read("utility").join().unwrap(), (QosClass::Utility, 0));
        assert_eq!(read("inherit").join().unwrap(), (QosClass::Default, 0));
        assert!(parse_spawned_arm("unknown").is_err());
        assert!(parse_spawned_arm("unspecified").is_err());
        assert_eq!(parse_spawned_arm("Interactive").unwrap(), "user-interactive");
        assert_eq!(parse_spawned_arm("inherit").unwrap(), "inherit");
    }

    /// The index a class is kept in an atomic as reads back as the class.
    #[test]
    fn a_class_round_trips_through_its_index() {
        for c in [
            QosClass::Unknown,
            QosClass::UserInteractive,
            QosClass::UserInitiated,
            QosClass::Default,
            QosClass::Utility,
            QosClass::Background,
            QosClass::Unspecified,
        ] {
            assert_eq!(QosClass::from_index(c as u8), c);
        }
    }

    /// A thread already interactive is left alone and the report says so.
    #[test]
    fn an_interactive_thread_is_not_claimed_twice() {
        std::thread::spawn(|| {
            macos::set(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0).unwrap();
            let claim = HotThread::claim();
            let r = claim.report();
            assert_eq!(r.before, QosClass::UserInteractive);
            assert_eq!(r.after, QosClass::UserInteractive);
            assert!(!r.engaged(), "nothing to engage");
            assert!(claim.restore.is_none());
        })
        .join()
        .unwrap();
    }
}
