//! Preserve the prior signal ABI and delivery policy while interposing capture.
//! Installation is a once-only process setup operation, after executor setup.
use std::sync::OnceLock;

pub(super) const SIGNALS: [libc::c_int; 5] = [
    libc::SIGSEGV,
    libc::SIGBUS,
    libc::SIGABRT,
    libc::SIGILL,
    libc::SIGFPE,
];

#[derive(Debug, Clone, Copy)]
pub(super) enum Stage {
    Query,
    Install,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct InstallError {
    pub(super) signal: libc::c_int,
    pub(super) stage: Stage,
    pub(super) errno: libc::c_int,
    pub(super) rollback_failed: u8,
}
impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "crash recorder signal {} {:?} failed: errno {}; rollback failure mask {:#x}",
            self.signal, self.stage, self.errno, self.rollback_failed
        )
    }
}
impl std::error::Error for InstallError {}

static SAVED: OnceLock<[libc::sigaction; SIGNALS.len()]> = OnceLock::new();
static INSTALLED: OnceLock<Result<(), InstallError>> = OnceLock::new();

// Flags and masks belong to the predecessor, not to the capture implementation.
fn wrapped(prior: &libc::sigaction, handler: usize) -> libc::sigaction {
    let mut action = *prior;
    action.sa_sigaction = handler;
    action.sa_flags |= libc::SA_SIGINFO;
    action
}

/// Called at the existing single-client setup boundary. Other components must
/// not replace these dispositions concurrently or after this once-only install.
/// Saved actions are immutable and published before any capture handler is live.
pub(super) fn install(
    handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void),
) -> Result<(), InstallError> {
    *INSTALLED.get_or_init(|| {
        // SAFETY: sigaction writes a complete C action on successful query.
        let mut saved: [libc::sigaction; SIGNALS.len()] = unsafe { std::mem::zeroed() };
        for (i, &signal) in SIGNALS.iter().enumerate() {
            // SAFETY: query only; valid signal and writable action storage.
            if unsafe { libc::sigaction(signal, std::ptr::null(), &mut saved[i]) } != 0 {
                return Err(InstallError {
                    signal,
                    stage: Stage::Query,
                    errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                    rollback_failed: 0,
                });
            }
        }
        let saved = SAVED.get_or_init(|| saved);
        for (i, &signal) in SIGNALS.iter().enumerate() {
            let action = wrapped(&saved[i], handler as *const () as usize);
            // SAFETY: handler is the private extern-C capture function; actions
            // stay process-rooted before their dispositions become visible.
            if unsafe { libc::sigaction(signal, &action, std::ptr::null_mut()) } != 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                let mut rollback_failed = 0;
                for old in (0..i).rev() {
                    // SAFETY: restore precisely the previously queried action.
                    if unsafe { libc::sigaction(SIGNALS[old], &saved[old], std::ptr::null_mut()) }
                        != 0
                    {
                        rollback_failed |= 1 << old;
                    }
                }
                return Err(InstallError {
                    signal,
                    stage: Stage::Install,
                    errno,
                    rollback_failed,
                });
            }
        }
        Ok(())
    })
}

/// Original info/context pointers stay valid for this synchronous handler call.
/// The kernel already applied the predecessor's mask, alternate-stack, restart
/// and reset policy through wrapped(). This function takes no lock or allocation.
pub(super) unsafe fn chain(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    let Some(index) = SIGNALS.iter().position(|&candidate| candidate == signal) else {
        return;
    };
    let Some(saved) = SAVED.get() else {
        return;
    };
    // SAFETY: the selected saved action and original delivery arguments retain
    // their kernel registration ABI for this synchronous chain.
    unsafe { dispatch(&saved[index], signal, info, context) };
}

unsafe fn dispatch(
    prior: &libc::sigaction,
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    let handler = prior.sa_sigaction;
    if handler == libc::SIG_DFL {
        // SAFETY: restore the original default action before re-raising. If
        // masked, the default action is delivered when this handler returns.
        if unsafe { libc::sigaction(signal, prior, std::ptr::null_mut()) } != 0
            || unsafe { libc::raise(signal) } != 0
        {
            // SAFETY: failed disposition restoration cannot safely resume a fault.
            unsafe { libc::_exit(128 + signal) };
        }
    } else if handler != libc::SIG_IGN {
        if prior.sa_flags & libc::SA_SIGINFO != 0 {
            // SAFETY: saved sigaction selected precisely this three-argument ABI.
            let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                unsafe { std::mem::transmute(handler) };
            handler(signal, info, context);
        } else {
            // SAFETY: without SA_SIGINFO the saved function accepts only signal.
            let handler: extern "C" fn(libc::c_int) = unsafe { std::mem::transmute(handler) };
            handler(signal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ONE: AtomicUsize = AtomicUsize::new(0);
    static THREE: AtomicUsize = AtomicUsize::new(0);
    static INFO: AtomicUsize = AtomicUsize::new(0);
    static CONTEXT: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn one(signal: libc::c_int) {
        ONE.store(signal as usize, Ordering::Relaxed);
    }
    extern "C" fn three(
        signal: libc::c_int,
        info: *mut libc::siginfo_t,
        context: *mut libc::c_void,
    ) {
        THREE.store(signal as usize, Ordering::Relaxed);
        INFO.store(info as usize, Ordering::Relaxed);
        CONTEXT.store(context as usize, Ordering::Relaxed);
    }
    fn action(handler: usize, flags: libc::c_int) -> libc::sigaction {
        // SAFETY: initialized C action; no process signal handler is installed.
        let mut value: libc::sigaction = unsafe { std::mem::zeroed() };
        value.sa_sigaction = handler;
        value.sa_flags = flags;
        // SAFETY: valid writable sigset storage, with an actual signal member.
        unsafe {
            libc::sigemptyset(&mut value.sa_mask);
            libc::sigaddset(&mut value.sa_mask, libc::SIGALRM);
        }
        value
    }
    #[test]
    fn wrapper_preserves_predecessor_delivery_policy() {
        let flags = libc::SA_ONSTACK | libc::SA_RESTART | libc::SA_RESETHAND;
        let prior = action(one as *const () as usize, flags);
        let wrapper = wrapped(&prior, three as *const () as usize);
        assert_eq!(wrapper.sa_flags, flags | libc::SA_SIGINFO);
        assert_eq!(wrapper.sa_flags & libc::SA_NODEFER, 0);
        // SAFETY: both sets were initialized; membership inspection only.
        assert_eq!(
            unsafe { libc::sigismember(&wrapper.sa_mask, libc::SIGALRM) },
            1
        );
        assert_eq!(wrapper.sa_sigaction, three as *const () as usize);
    }
    #[test]
    fn dispatch_uses_one_argument_predecessor() {
        let prior = action(one as *const () as usize, 0);
        // SAFETY: actual one-argument extern-C fixture, not a delivered signal.
        unsafe { dispatch(&prior, 17, std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(ONE.load(Ordering::Relaxed), 17);
    }
    #[test]
    fn dispatch_preserves_three_argument_context() {
        let prior = action(three as *const () as usize, libc::SA_SIGINFO);
        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        let mut context = 0u8;
        let context_ptr = (&mut context as *mut u8).cast();
        // SAFETY: fixture records pointer identities only; no signal/context read.
        unsafe { dispatch(&prior, 19, info.as_mut_ptr(), context_ptr) };
        assert_eq!(THREE.load(Ordering::Relaxed), 19);
        assert_eq!(INFO.load(Ordering::Relaxed), info.as_mut_ptr() as usize);
        assert_eq!(CONTEXT.load(Ordering::Relaxed), context_ptr as usize);
    }
}
