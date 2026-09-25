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

impl QosClass {
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
