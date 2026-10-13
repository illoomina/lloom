//! The persistent record: the last N inputs an in-process loop executed, dumped
//! together when the loop dies (lloom #706).
//!
//! # Why a crashing input is not a bug report for a stateful target
//!
//! AFL++ has `AFL_PERSISTENT_RECORD`; LibAFL 0.16.1 has nothing under that name
//! (grepped over `libafl`, `libafl_bolts`, `libafl_targets`). The gap matters
//! here more than for a parser: every harness in this crate runs its sequence
//! **in the same process as the previous ten thousand**, against a Metal device,
//! a command queue, a buffer pool and a residency set that are process-wide. A
//! failure that depends on what an *earlier* input left behind — a page the
//! previous sequence quarantined, a segment the previous ring lapped, a queue
//! depth the previous commit pattern reached — reproduces from the crashing
//! input alone exactly never. The record is what makes it reproduce: the last N
//! inputs, in order, replayed in one fresh process.
//!
//! # Shape
//!
//! A ring of the raw input bytes, pushed *before* each execution (so the input
//! that dies is in the ring when the handler runs). Three exits write it:
//!
//! 1. the harness returning an oracle failure — the common case, written by the
//!    harness itself through [`Record::dump`];
//! 2. a Rust panic anywhere under the harness — the hook installed by
//!    [`Record::install_crash_dump`];
//! 3. a signal (`SIGSEGV`, `SIGBUS`, `SIGABRT`, `SIGILL`, `SIGFPE`) — the
//!    handlers installed by the same function, which write the ring through
//!    a fixed-storage writer on a directory descriptor opened at install, then
//!    chain to the action they replaced. LibAFL's own in-process handlers run
//!    too; ours are chained ahead of them so the record exists before the
//!    fuzzer's report.
//!
//! The dump is a directory `record-<pid>-<seq>/` holding `0000.bin … NNNN.bin`
//! (oldest first, the dying input last), `reason.txt`, and a versioned
//! completion manifest. [`replay`] validates counts/sizes, then executes
//! a directory in order through the same harness and reports which entry fails.
//!
//! The ring lives in a global because the signal path cannot take a lock or
//! allocate: the slot buffers are pre-allocated at [`Record::install`] and a
//! push is a `memcpy` under a nonblocking atomic permit. Contention refuses
//! recording rather than racing the payload. The signal route uses only
//! descriptor syscalls, fixed stack buffers and atomics; [`Record::dump`] and
//! the panic hook keep the allocating writer, which is not a signal context.

mod archive;
pub(crate) mod identity;
pub(crate) use archive::read_single;
mod fixed_capture;
mod inputs;
mod signals;

use inputs::InputRing;
pub use inputs::RecordError;
use std::fs;
#[cfg(test)]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The largest input the ring accepts. Larger inputs return an explicit error;
/// the campaign must not execute bytes it failed to record.
pub const MAX_INPUT: usize = 64 * 1024;

/// Requested recorder payload/metadata-array quota; not process RSS. Explicit
/// larger limits retain large-history support without weakening replay checks.
#[derive(Clone, Copy, Debug)]
pub struct RecordLimits {
    pub max_ring_bytes: usize,
}
impl Default for RecordLimits {
    fn default() -> Self {
        Self {
            max_ring_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Logical replay work/metadata quotas, not a process RSS or allocator quota.
/// Larger old records remain supported through explicit larger limits. These
/// limits never bypass identity, completion, per-input size or digest checks.
#[derive(Clone, Copy, Debug)]
pub struct ReplayLimits {
    pub max_inputs: usize,
    pub max_directory_entries: usize,
    pub max_metadata_bytes: usize,
    pub max_total_input_bytes: u64,
}
impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_inputs: 4096,
            max_directory_entries: 8192,
            max_metadata_bytes: 8 * 1024 * 1024,
            max_total_input_bytes: 256 * 1024 * 1024,
        }
    }
}

struct Ring {
    inputs: InputRing,
    dir: PathBuf,
    /// `dir` opened at install for the signal route; negative when it could
    /// not be opened, in which case a signal writes no record.
    dir_fd: libc::c_int,
    dumps: AtomicUsize,
    dumping: AtomicBool,
    identity: Option<Vec<u8>>,
}

impl Drop for Ring {
    // Only a candidate that lost the install race is ever dropped; the
    // installed ring is process-lifetime and keeps its descriptor.
    fn drop(&mut self) {
        if self.dir_fd >= 0 {
            // SAFETY: the descriptor was opened by open_parent for this ring
            // alone and was never published.
            unsafe { libc::close(self.dir_fd) };
        }
    }
}

/// Open the record directory for descriptor-relative writes. Cold setup only.
fn open_parent(dir: &Path) -> libc::c_int {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(dir.as_os_str().as_bytes()) else {
        return -1;
    };
    // SAFETY: NUL-terminated path; the returned descriptor is owned by a Ring.
    unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    }
}

static RING: std::sync::OnceLock<Ring> = std::sync::OnceLock::new();

/// The record's handle. One per process; [`Record::install`] once, then
/// [`Record::push`] per execution and [`Record::dump`] on an oracle failure.
#[derive(Clone, Copy, Debug)]
pub struct Record;

impl Record {
    /// Install the ring: `n` slots of [`MAX_INPUT`] bytes under `dir`.
    /// Idempotent after the first call (the first `n`/`dir` win). The panic
    /// hook and the signal handlers are [`Record::install_crash_dump`], called
    /// separately and AFTER LibAFL's executor has installed its own, so that
    /// ours run first and chain to theirs -- a handler installed before the
    /// executor is simply replaced by it (measured: 2 252 objectives and zero
    /// records on the first Metal campaign).
    pub fn install(dir: &Path, n: usize) -> Record {
        let n = n.max(1);
        RING.get_or_init(|| {
            fs::create_dir_all(dir).ok();
            Ring {
                inputs: InputRing::new(n, MAX_INPUT),
                dir: dir.to_path_buf(),
                dir_fd: open_parent(dir),
                dumps: AtomicUsize::new(0),
                dumping: AtomicBool::new(false),
                identity: None,
            }
        });
        Record
    }

    /// Install an identified campaign. Unlike the legacy first-wins API, a
    /// different second identity/size/path is an error before target execution.
    pub fn install_with_identity(
        dir: &Path,
        n: usize,
        identity: Vec<u8>,
    ) -> std::io::Result<Record> {
        Self::install_with_identity_and_limits(dir, n, identity, RecordLimits::default())
    }

    /// Install with an explicit requested ring-layout budget. Increasing it
    /// recovers larger histories; identity/path/capacity checks still apply.
    /// Legacy `install` remains first-wins and infallible. Path/identity and
    /// dump allocations, concurrent setup peak and process RSS are not bounded.
    pub fn install_with_identity_and_limits(
        dir: &Path,
        n: usize,
        identity: Vec<u8>,
        limits: RecordLimits,
    ) -> std::io::Result<Record> {
        if identity.is_empty() || identity.len() > archive::MAX_IDENTITY {
            return Err(std::io::Error::other(
                "record identity is empty or exceeds its bound",
            ));
        }
        let n = n.max(1);
        let requested = InputRing::requested_bytes(n, MAX_INPUT).map_err(std::io::Error::other)?;
        if requested > limits.max_ring_bytes {
            return Err(std::io::Error::other(inputs::SetupError::Budget {
                requested,
                limit: limits.max_ring_bytes,
            }));
        }
        let matches = |ring: &Ring, identity: &[u8]| {
            ring.identity.as_deref() == Some(identity)
                && ring.dir == dir
                && ring.inputs.capacity() == n
        };
        let mismatch = || {
            std::io::Error::other(
                "persistent recorder already installed with different identity/path/capacity",
            )
        };
        if let Some(ring) = RING.get() {
            return if matches(ring, &identity) {
                Ok(Record)
            } else {
                Err(mismatch())
            };
        }
        // Fallible ring birth and full quota refusal precede directory effects.
        // Setup is cold; no lock/allocator change in push/snapshot hot paths.
        let inputs = InputRing::try_new(n, MAX_INPUT, limits.max_ring_bytes)
            .map_err(std::io::Error::other)?;
        fs::create_dir_all(dir)?;
        let dir_fd = open_parent(dir);
        if dir_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let candidate = Ring {
            inputs,
            dir: dir.to_path_buf(),
            dir_fd,
            dumps: AtomicUsize::new(0),
            dumping: AtomicBool::new(false),
            identity: Some(identity),
        };
        match RING.set(candidate) {
            Ok(()) => Ok(Record),
            Err(candidate) => {
                // A concurrent installer won. Never replace it or retry births.
                // Candidate owns all partial allocations and drops on this path.
                let winner = RING
                    .get()
                    .expect("OnceLock set lost to initialized recorder");
                if matches(
                    winner,
                    candidate.identity.as_deref().expect("identified candidate"),
                ) {
                    Ok(Record)
                } else {
                    Err(mismatch())
                }
            }
        }
    }

    /// How many slots the ring holds.
    pub fn capacity(self) -> usize {
        RING.get().map_or(0, |r| r.inputs.capacity())
    }

    /// Record `input` as the next execution. Call BEFORE running it.
    /// A refusal leaves history unchanged; callers must not execute that input.
    pub fn push(self, input: &[u8]) -> Result<(), RecordError> {
        let r = RING.get().ok_or(RecordError::NotInstalled)?;
        r.inputs.push(input)
    }

    /// Write the ring, oldest first, and the reason. Returns the directory.
    ///
    /// Allocates, formats and synchronizes files; this can block on filesystem
    /// I/O. Returns a directory only after the completion manifest is written.
    /// Returns no directory if a writer or another dump
    /// owns the history. Not for signal context: the signal route has its own
    /// fixed-storage writer.
    pub fn dump(self, reason: &str) -> Option<PathBuf> {
        let r = RING.get()?;
        let dir = dump_raw(r, reason.as_bytes())?;
        Some(dir)
    }

    /// The panic hook and signal handlers. Install AFTER the fuzzer's executor
    /// exists (see [`Record::install`]); idempotent. Query/install failures are
    /// returned before target execution; partial installs attempt rollback.
    /// Disposition replacement must remain serialized with process setup.
    pub fn install_crash_dump() -> std::io::Result<()> {
        IMAGE.get_or_init(Image::of_handler);
        signals::install(on_signal).map_err(std::io::Error::other)?;
        static PANIC_HOOK: std::sync::Once = std::sync::Once::new();
        PANIC_HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if let Some(r) = RING.get() {
                    let msg = format!("panic: {info}");
                    let _ = dump_raw(r, msg.as_bytes());
                }
                prev(info);
            }));
        });
        Ok(())
    }
}

/// The image holding the capture handler, recorded at install so a signal-time
/// backtrace can be written as raw addresses and symbolized offline
/// (`atos -o <image_path> -l <image_base> <pc>...`): symbolizing in the handler
/// would take dyld's lock.
struct Image {
    base: usize,
    path: [u8; 1024],
    len: usize,
}

impl Image {
    fn of_handler() -> Self {
        let mut image = Image {
            base: 0,
            path: [0; 1024],
            len: 0,
        };
        // SAFETY: dladdr fills the zeroed Dl_info on success; dli_fname is a
        // NUL-terminated string owned by dyld for the image's lifetime.
        unsafe {
            let mut found: libc::Dl_info = std::mem::zeroed();
            if libc::dladdr(on_signal as *const libc::c_void, &mut found) != 0 {
                image.base = found.dli_fbase as usize;
                if !found.dli_fname.is_null() {
                    let name = std::ffi::CStr::from_ptr(found.dli_fname).to_bytes();
                    image.len = name.len().min(image.path.len());
                    image.path[..image.len].copy_from_slice(&name[..image.len]);
                }
            }
        }
        image
    }
}

static IMAGE: std::sync::OnceLock<Image> = std::sync::OnceLock::new();

/// Where the last signal capture's frame buffer lived, for the delivery
/// controls to classify: thread stack, alternate stack, or ASan's fake stack.
/// The fake-stack query only recognises a live frame, so it is made inside the
/// capture, through entry points resolved before any delivery.
#[cfg(test)]
mod handler_frame {
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Current = unsafe extern "C" fn() -> *mut libc::c_void;
    type IsIn = unsafe extern "C" fn(
        *mut libc::c_void,
        *mut libc::c_void,
        *mut *mut libc::c_void,
        *mut *mut libc::c_void,
    ) -> *mut libc::c_void;

    pub(super) static ADDR: AtomicUsize = AtomicUsize::new(0);
    /// 0: no ASan runtime, 1: not in a fake frame, 2: in a live fake frame.
    pub(super) static IN_FAKE: AtomicUsize = AtomicUsize::new(0);
    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static IS_IN: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn resolve() {
        // SAFETY: dlsym of exported ASan interface functions; null if absent.
        unsafe {
            let current = libc::dlsym(
                libc::RTLD_DEFAULT,
                c"__asan_get_current_fake_stack".as_ptr(),
            );
            let is_in = libc::dlsym(libc::RTLD_DEFAULT, c"__asan_addr_is_in_fake_stack".as_ptr());
            CURRENT.store(current as usize, Ordering::Relaxed);
            IS_IN.store(is_in as usize, Ordering::Relaxed);
        }
    }

    pub(super) fn note(addr: usize) {
        ADDR.store(addr, Ordering::Relaxed);
        let (current, is_in) = (
            CURRENT.load(Ordering::Relaxed),
            IS_IN.load(Ordering::Relaxed),
        );
        if current == 0 || is_in == 0 {
            IN_FAKE.store(0, Ordering::Relaxed);
            return;
        }
        // SAFETY: both were resolved from the ASan runtime's exported interface
        // with these C signatures; the query reads the thread's fake stack only.
        let hit = unsafe {
            let current: Current = std::mem::transmute(current);
            let is_in: IsIn = std::mem::transmute(is_in);
            let fake = current();
            !fake.is_null()
                && !is_in(
                    fake,
                    addr as *mut libc::c_void,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
                .is_null()
        };
        IN_FAKE.store(if hit { 2 } else { 1 }, Ordering::Relaxed);
    }
}

extern "C" fn on_signal(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let interrupted_errno = signals::PreservedErrno::new();
    if let Some(r) = RING.get() {
        capture_signal(r, sig, info);
    }
    // SAFETY: forward the original kernel delivery to the ABI and policy
    // retained from the predecessor.
    interrupted_errno.restore();
    unsafe { signals::chain(sig, info, ctx) };
}

/// The signal route: descriptor syscalls, fixed stack storage and atomics only.
/// The record is complete before the backtrace is attempted, so a fault while
/// walking the stack loses the backtrace, never the record.
fn capture_signal(r: &Ring, sig: libc::c_int, info: *mut libc::siginfo_t) {
    if r.dir_fd < 0 {
        return;
    }
    let mut reason = *b"signal 00\n";
    reason[7] = b'0' + ((sig / 10) % 10) as u8;
    reason[8] = b'0' + (sig % 10) as u8;
    // SAFETY: getpid has no failure mode and is async-signal-safe.
    let pid = unsafe { libc::getpid() } as u64;
    let written = with_history(r, |snapshot, seq| {
        fixed_capture::capture(
            r.dir_fd,
            pid,
            seq as u64,
            snapshot.inputs(),
            r.identity.as_deref(),
            &reason,
        )
        .then_some(seq)
    });
    let Some(seq) = written else {
        return;
    };
    // SAFETY: `info` is the kernel's siginfo for this delivery, or null.
    let addr = if info.is_null() {
        0
    } else {
        unsafe { (*info).si_addr as usize }
    };
    let mut frames = [std::ptr::null_mut::<libc::c_void>(); 96];
    #[cfg(test)]
    handler_frame::note(frames.as_ptr() as usize);
    // SAFETY: a fixed-size frame buffer; the count written is bounded by it.
    let n = unsafe { libc::backtrace(frames.as_mut_ptr(), frames.len() as libc::c_int) };
    let frames = &frames[..(n.max(0) as usize).min(frames.len())];
    let (base, path) = IMAGE
        .get()
        .map_or((0, &[][..]), |image| (image.base, &image.path[..image.len]));
    fixed_capture::context(r.dir_fd, pid, seq as u64, addr, base, path, frames);
}

/// Run `f` on a coherent snapshot with a fresh dump sequence number, or not at
/// all when another dump, a writer or a reader owns the history. Never blocks:
/// the caller may have interrupted the owner.
fn with_history<T>(
    r: &Ring,
    f: impl FnOnce(&inputs::Snapshot<'_>, usize) -> Option<T>,
) -> Option<T> {
    if r.dumping.swap(true, Ordering::AcqRel) {
        return None; // re-entrant (a crash inside a dump): leave the first one
    }
    // Release the reentrancy latch on every return, including filesystem errors.
    struct DumpGuard<'a>(&'a AtomicBool);
    impl Drop for DumpGuard<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _dump_guard = DumpGuard(&r.dumping);
    let snapshot = r.inputs.snapshot().ok()?;
    snapshot.inputs().next()?;
    let seq = r
        .dumps
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        .ok()?;
    f(&snapshot, seq)
}

/// Write a coherent ring snapshot through the allocating archive writer, for
/// the oracle and panic exits. Busy snapshots return no directory.
fn dump_raw(r: &Ring, reason: &[u8]) -> Option<PathBuf> {
    with_history(r, |snapshot, seq| {
        let dir = r.dir.join(format!("record-{}-{seq}", std::process::id()));
        archive::write_identified(&dir, snapshot.inputs(), reason, r.identity.as_deref()).ok()?;
        Some(dir)
    })
}

/// Replay a record directory (or a single input file) through `harness`, in
/// order, in this process. Returns `Ok(n)` with the number of inputs run when
/// none failed, or `Err((index, message))` for the first that did.
///
/// The harness's own `Err` is the oracle failure; a panic propagates as it
/// would in the campaign (the hook above dumps a fresh record of the replay,
/// which is the state-carryover evidence: a record that reproduces where the
/// single crashing input does not).
pub fn replay<F>(path: &Path, harness: F) -> Result<usize, (usize, String)>
where
    F: FnMut(&[u8]) -> Result<(), String>,
{
    replay_with_identity(path, None, false, harness)
}

/// Replay with an explicitly expected campaign identity. A mismatch is refused
/// before the first harness invocation. The explicit diagnostic override never
/// bypasses format/completion/size validation and never certifies exact replay.
pub fn replay_with_identity<F>(
    path: &Path,
    expected: Option<&[u8]>,
    allow_mismatch: bool,
    harness: F,
) -> Result<usize, (usize, String)>
where
    F: FnMut(&[u8]) -> Result<(), String>,
{
    replay_with_identity_and_limits(
        path,
        expected,
        allow_mismatch,
        ReplayLimits::default(),
        harness,
    )
}

/// Same replay contract with caller-selected finite metadata/work limits.
/// Raising a limit supports larger archives; it is not an integrity override.
pub fn replay_with_identity_and_limits<F>(
    path: &Path,
    expected: Option<&[u8]>,
    allow_mismatch: bool,
    limits: ReplayLimits,
    mut harness: F,
) -> Result<usize, (usize, String)>
where
    F: FnMut(&[u8]) -> Result<(), String>,
{
    let replay = archive::prepare_with_limits(path, MAX_INPUT, limits)
        .map_err(|error| (0, format!("{}: {error}", path.display())))?;
    let matched = archive::check_identity(replay.identity.as_deref(), expected, allow_mismatch)
        .map_err(|error| (0, format!("{}: {error}", path.display())))?;
    if !matched {
        eprintln!("UNVERIFIED REPLAY IDENTITY: legacy/unidentified input or explicit diagnostic mismatch override; not exact build/configuration/schedule replay");
    }
    if replay.legacy {
        eprintln!(
            "legacy record: numeric order checked; completeness/build/schema metadata unavailable"
        );
    }
    if !replay.content_verified() {
        eprintln!(
            "UNVERIFIED REPLAY CONTENT: legacy/v1/v2/single input has no recorded content digests"
        );
    }
    for (i, p) in replay.paths.iter().enumerate() {
        let bytes = replay
            .read_input(i, MAX_INPUT)
            .map_err(|error| (i, format!("{}: {error}", p.display())))?;
        harness(&bytes).map_err(|m| (i, format!("{}: {m}", p.display())))?;
    }
    Ok(replay.paths.len())
}

#[cfg(test)]
mod signal_errno_controls {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    static PREDECESSOR_ERRNO: AtomicI32 = AtomicI32::new(0);
    static PREDECESSOR_SIGNAL: AtomicI32 = AtomicI32::new(0);

    extern "C" fn predecessor(signal: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
        PREDECESSOR_ERRNO.store(errno::errno().0, Ordering::Relaxed);
        PREDECESSOR_SIGNAL.store(signal, Ordering::Relaxed);
        errno::set_errno(errno::Errno(libc::EIO));
    }

    #[test]
    fn delivered_capture_preserves_errno_before_and_after_predecessor() {
        const CHILD: &str = "LLOOM_RECORD_ERRNO_CONTROL_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "record::signal_errno_controls::delivered_capture_preserves_errno_before_and_after_predecessor",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            eprintln!("{}", String::from_utf8_lossy(&output.stdout));
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            assert!(
                output.status.success(),
                "isolated delivered-signal child failed"
            );
            return;
        }
        let path = std::env::temp_dir().join(format!("lloom-record-errno-{}", std::process::id()));
        let record = Record::install(&path, 1);
        record.push(b"actual recorded input").unwrap();
        std::fs::remove_dir(&path).unwrap();
        // The actual recorder's archive birth will fail with ENOENT. This
        // deliberately exercises capture changing errno, not a dummy callback.
        // SAFETY: zeroed C action is initialized before registration below.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = predecessor as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO;
        // SAFETY: initialized action storage; handler has the selected C ABI.
        assert_eq!(
            unsafe {
                libc::sigemptyset(&mut action.sa_mask);
                libc::sigaction(libc::SIGILL, &action, std::ptr::null_mut())
            },
            0
        );
        Record::install_crash_dump().unwrap();
        errno::set_errno(errno::Errno(libc::EDOM));
        // SAFETY: this disposable child installed a returning SIGILL handler.
        // Delivery occurs outside allocator work; no real fault is generated.
        let raised = unsafe { libc::raise(libc::SIGILL) };
        let returned_errno = errno::errno().0;
        assert_eq!(raised, 0);
        assert_eq!(PREDECESSOR_SIGNAL.load(Ordering::Relaxed), libc::SIGILL);
        assert_eq!(
            PREDECESSOR_ERRNO.load(Ordering::Relaxed),
            libc::EDOM,
            "capture I/O must not change the predecessor's interrupted errno"
        );
        assert_eq!(
            returned_errno,
            libc::EDOM,
            "returning predecessor must not leak errno into interrupted code"
        );
        assert!(!path.exists());
        record.push(b"after returning signal").unwrap();
    }
}

#[cfg(test)]
mod signal_delivery_controls {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::atomic::{AtomicI32, Ordering};

    static CALLS: AtomicI32 = AtomicI32::new(0);
    static SIGNAL: AtomicI32 = AtomicI32::new(0);
    static ALARM_MASKED: AtomicI32 = AtomicI32::new(0);
    static SELF_MASKED: AtomicI32 = AtomicI32::new(0);

    extern "C" fn one_argument(signal: libc::c_int) {
        // SAFETY: writable sigset initialized by the successful kernel query.
        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        let queried = unsafe { libc::sigprocmask(libc::SIG_SETMASK, std::ptr::null(), &mut mask) };
        if queried == 0 {
            // SAFETY: query initialized mask storage; membership reads only.
            ALARM_MASKED.store(
                unsafe { libc::sigismember(&mask, libc::SIGALRM) },
                Ordering::Relaxed,
            );
            SELF_MASKED.store(
                unsafe { libc::sigismember(&mask, signal) },
                Ordering::Relaxed,
            );
        }
        SIGNAL.store(signal, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn delivered(mode: &str, termination: Option<libc::c_int>) {
        const CHILD: &str = "LLOOM_RECORD_POLICY_CONTROL_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().unwrap().to_owned();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name, "--nocapture", "--test-threads=1"])
                .env(CHILD, mode)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("{stdout}");
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            assert!(
                stdout.contains("CAPTURE_HANDLER_REGISTERED"),
                "child must verify actual recorder registration before delivery"
            );
            match termination {
                Some(signal) => assert_eq!(
                    output.status.signal(),
                    Some(signal),
                    "only the intended default signal is acceptance"
                ),
                None => assert!(output.status.success(), "returning delivery child failed"),
            }
            return;
        }
        assert_eq!(std::env::var(CHILD).unwrap(), mode);
        // SAFETY: cold process setup disables cores only in this disposable child.
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &zero) }, 0);
        // No ring is installed: these tests cover actual delivery/chaining policy,
        // not capture allocation or successful archive creation.
        // SAFETY: initialized C actions are fully populated before registration.
        let mut prior: libc::sigaction = unsafe { std::mem::zeroed() };
        prior.sa_sigaction = match mode {
            "ignore" => libc::SIG_IGN,
            "default" => libc::SIG_DFL,
            _ => one_argument as *const () as usize,
        };
        prior.sa_flags = match mode {
            "nodefer" => libc::SA_NODEFER,
            _ => 0,
        };
        // SAFETY: writable mask/action storage and a correct one-argument ABI.
        assert_eq!(
            unsafe {
                libc::sigemptyset(&mut prior.sa_mask);
                libc::sigaddset(&mut prior.sa_mask, libc::SIGALRM);
                libc::sigaction(libc::SIGILL, &prior, std::ptr::null_mut())
            },
            0
        );
        let expected_mask = if matches!(mode, "one" | "nodefer") {
            // SAFETY: returning predecessor is registered for this first delivery.
            assert_eq!(unsafe { libc::raise(libc::SIGILL) }, 0);
            assert_eq!(CALLS.load(Ordering::Relaxed), 1);
            let reference = (
                ALARM_MASKED.load(Ordering::Relaxed),
                SELF_MASKED.load(Ordering::Relaxed),
            );
            println!("UNWRAPPED_REFERENCE_MASK={reference:?}");
            CALLS.store(0, Ordering::Relaxed);
            SIGNAL.store(0, Ordering::Relaxed);
            ALARM_MASKED.store(0, Ordering::Relaxed);
            SELF_MASKED.store(0, Ordering::Relaxed);
            // SAFETY: reinstall the SAME predecessor after its possible reset.
            assert_eq!(
                unsafe { libc::sigaction(libc::SIGILL, &prior, std::ptr::null_mut()) },
                0
            );
            Some(reference)
        } else {
            None
        };
        Record::install_crash_dump().unwrap();
        // SAFETY: successful query initializes the complete registered action.
        let mut installed: libc::sigaction = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::sigaction(libc::SIGILL, std::ptr::null(), &mut installed) },
            0
        );
        assert_eq!(installed.sa_sigaction, on_signal as *const () as usize);
        assert_ne!(installed.sa_flags & libc::SA_SIGINFO, 0);
        println!("CAPTURE_HANDLER_REGISTERED");
        std::io::stdout().flush().unwrap();
        // SAFETY: intentional delivery to this child; parent validates termination.
        assert_eq!(unsafe { libc::raise(libc::SIGILL) }, 0);
        assert_ne!(mode, "default", "default disposition must not return");
        if mode == "ignore" {
            assert_eq!(CALLS.load(Ordering::Relaxed), 0);
            return;
        }
        assert_eq!(CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(SIGNAL.load(Ordering::Relaxed), libc::SIGILL);
        assert_eq!(
            (
                ALARM_MASKED.load(Ordering::Relaxed),
                SELF_MASKED.load(Ordering::Relaxed)
            ),
            expected_mask.unwrap(),
            "capture must preserve the SAME unwrapped kernel delivery policy",
        );
    }

    #[test]
    fn delivered_one_argument_predecessor_observes_saved_mask() {
        delivered("one", None);
    }
    #[test]
    fn delivered_nodefer_predecessor_observes_saved_mask() {
        delivered("nodefer", None);
    }
    #[test]
    fn delivered_ignore_disposition_returns() {
        delivered("ignore", None);
    }
    #[test]
    fn delivered_default_disposition_terminates() {
        delivered("default", Some(libc::SIGILL));
    }
}

/// The recorder carries every delivery policy the `sigaction` query reports.
/// Darwin's query never reports `SA_RESETHAND` (§15.1l; the register-then-query
/// probe `query_roundtrip.c`), so a one-shot predecessor stays wrapped after
/// its first delivery. Each test runs the predecessor alone (reference) and
/// behind capture (wrapped), each in its own child, and pins both arms at the
/// level the kernel shows: for SIGILL `sigaction(2)` ignores the flag and the
/// arms agree; for SIGSEGV, SIGBUS, SIGABRT and SIGFPE the reference reverts
/// to `SIG_DFL` and dies on the second delivery while the wrapped arm keeps
/// the capture handler and returns. A query that starts reporting the flag
/// fails these tests, and the boundary must be measured again. A predecessor
/// without the flag is identical in both arms on both deliveries.
#[cfg(test)]
mod signal_reset_reference {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::atomic::{AtomicI32, Ordering};

    const CHILD: &str = "LLOOM_RECORD_RESET_REFERENCE_CHILD";
    const FIRST_KEPT: &str = "RESET_FIRST calls=1 mask=(1,1) post=kept";
    const FIRST_DEFAULT: &str = "RESET_FIRST calls=1 mask=(1,1) post=default";
    const SECOND_RETURNED: &str = "RESET_SECOND=returned calls=2";
    static CALLS: AtomicI32 = AtomicI32::new(0);
    static ALARM_MASKED: AtomicI32 = AtomicI32::new(-1);
    static SELF_MASKED: AtomicI32 = AtomicI32::new(-1);

    extern "C" fn predecessor(signal: libc::c_int) {
        // SAFETY: writable sigset; the successful query initializes it.
        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        if unsafe { libc::sigprocmask(libc::SIG_SETMASK, std::ptr::null(), &mut mask) } == 0 {
            // SAFETY: membership reads of the initialized mask only.
            ALARM_MASKED.store(
                unsafe { libc::sigismember(&mask, libc::SIGALRM) },
                Ordering::Relaxed,
            );
            SELF_MASKED.store(
                unsafe { libc::sigismember(&mask, signal) },
                Ordering::Relaxed,
            );
        }
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn disposition(signal: libc::c_int) -> usize {
        // SAFETY: query only; writable action storage.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::sigaction(signal, std::ptr::null(), &mut action) },
            0
        );
        action.sa_sigaction
    }

    fn child(arm: &str, signal: libc::c_int, flags: libc::c_int) {
        // SAFETY: cold setup in a disposable child; no core files.
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &zero) }, 0);
        // SAFETY: the action is fully initialized before registration.
        let mut prior: libc::sigaction = unsafe { std::mem::zeroed() };
        prior.sa_sigaction = predecessor as *const () as usize;
        prior.sa_flags = flags;
        assert_eq!(
            unsafe {
                libc::sigemptyset(&mut prior.sa_mask);
                libc::sigaddset(&mut prior.sa_mask, libc::SIGALRM);
                libc::sigaction(signal, &prior, std::ptr::null_mut())
            },
            0
        );
        let kept = match arm {
            "reference" => {
                assert_eq!(disposition(signal), predecessor as *const () as usize);
                println!("UNWRAPPED_HANDLER_REGISTERED");
                predecessor as *const () as usize
            }
            "wrapped" => {
                Record::install_crash_dump().unwrap();
                assert_eq!(disposition(signal), on_signal as *const () as usize);
                println!("CAPTURE_HANDLER_REGISTERED");
                on_signal as *const () as usize
            }
            other => panic!("unknown arm {other}"),
        };
        // SAFETY: intentional delivery to this disposable child.
        assert_eq!(unsafe { libc::raise(signal) }, 0);
        let after = disposition(signal);
        let post = if after == libc::SIG_DFL {
            "default"
        } else if after == kept {
            "kept"
        } else {
            "other"
        };
        println!(
            "RESET_FIRST calls={} mask=({},{}) post={post}",
            CALLS.load(Ordering::Relaxed),
            ALARM_MASKED.load(Ordering::Relaxed),
            SELF_MASKED.load(Ordering::Relaxed),
        );
        std::io::stdout().flush().unwrap();
        // SAFETY: second intentional delivery; a reset disposition terminates.
        assert_eq!(unsafe { libc::raise(signal) }, 0);
        println!(
            "RESET_SECOND=returned calls={}",
            CALLS.load(Ordering::Relaxed)
        );
        std::io::stdout().flush().unwrap();
    }

    /// One child's outcome: the terminating signal, clean exit, and the
    /// first- and second-delivery lines it printed.
    struct Arm {
        killed_by: Option<libc::c_int>,
        success: bool,
        first: Option<String>,
        second: Option<String>,
    }

    fn line(stdout: &str, prefix: &str) -> Option<String> {
        stdout
            .lines()
            .find_map(|l| l.find(prefix).map(|at| l[at..].to_owned()))
    }

    fn run(arm: &str, signal: libc::c_int, flags: libc::c_int) -> Arm {
        let name = std::thread::current().name().unwrap().to_owned();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(CHILD, format!("{arm}:{signal}:{flags}"))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        eprintln!(
            "[{arm} signal {signal} flags {flags:#x}] status {:?}\n{stdout}",
            output.status
        );
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        let marker = match arm {
            "reference" => "UNWRAPPED_HANDLER_REGISTERED",
            _ => "CAPTURE_HANDLER_REGISTERED",
        };
        assert!(stdout.contains(marker), "{arm} child did not register");
        if arm == "reference" {
            assert!(
                !stdout.contains("CAPTURE_HANDLER_REGISTERED"),
                "the reference arm must not install capture"
            );
        }
        Arm {
            killed_by: output.status.signal(),
            success: output.status.success(),
            first: line(&stdout, "RESET_FIRST "),
            second: line(&stdout, "RESET_SECOND"),
        }
    }

    /// Runs the child when this process is one; otherwise the two arms.
    fn arms(signal: libc::c_int, flags: libc::c_int) -> Option<(Arm, Arm)> {
        if let Some(mode) = std::env::var_os(CHILD) {
            let mode = mode.into_string().unwrap();
            let mut fields = mode.split(':');
            let arm = fields.next().unwrap().to_owned();
            let number: libc::c_int = fields.next().unwrap().parse().unwrap();
            let child_flags: libc::c_int = fields.next().unwrap().parse().unwrap();
            if (number, child_flags) == (signal, flags) {
                child(&arm, signal, flags);
            }
            return None;
        }
        Some((
            run("reference", signal, flags),
            run("wrapped", signal, flags),
        ))
    }

    fn returns_twice(arm: &Arm, label: &str) {
        assert!(arm.success, "{label}: kept handler must return twice");
        assert_eq!(arm.first.as_deref(), Some(FIRST_KEPT), "{label}");
        assert_eq!(arm.second.as_deref(), Some(SECOND_RETURNED), "{label}");
    }

    fn reset_pinned(signal: libc::c_int) {
        let Some((reference, wrapped)) = arms(signal, libc::SA_RESETHAND) else {
            return;
        };
        if signal == libc::SIGILL {
            returns_twice(&reference, "reference SIGILL ignores SA_RESETHAND");
            returns_twice(&wrapped, "wrapped SIGILL");
            return;
        }
        assert_eq!(
            reference.first.as_deref(),
            Some(FIRST_DEFAULT),
            "unwrapped kernel no longer resets signal {signal}"
        );
        assert_eq!(
            reference.killed_by,
            Some(signal),
            "reference second delivery must terminate by the signal"
        );
        assert!(
            reference.second.is_none(),
            "reference second delivery returned"
        );
        returns_twice(
            &wrapped,
            "wrapped: the query hides SA_RESETHAND, so capture stays installed; \
             a query that reports it moves this boundary",
        );
    }

    #[test]
    fn reset_sigill_matches_unwrapped_kernel() {
        reset_pinned(libc::SIGILL);
    }
    #[test]
    fn reset_sigsegv_is_kept_where_the_kernel_resets() {
        reset_pinned(libc::SIGSEGV);
    }
    #[test]
    fn reset_sigbus_is_kept_where_the_kernel_resets() {
        reset_pinned(libc::SIGBUS);
    }
    #[test]
    fn reset_sigabrt_is_kept_where_the_kernel_resets() {
        reset_pinned(libc::SIGABRT);
    }
    #[test]
    fn reset_sigfpe_is_kept_where_the_kernel_resets() {
        reset_pinned(libc::SIGFPE);
    }

    /// Positive control for the boundary above: without `SA_RESETHAND` the
    /// wrapped arm is line-for-line the unwrapped kernel on every recorded
    /// signal, on both deliveries.
    #[test]
    fn no_reset_predecessor_matches_unwrapped_kernel() {
        for signal in signals::SIGNALS {
            let Some((reference, wrapped)) = arms(signal, 0) else {
                continue;
            };
            returns_twice(&reference, "reference without SA_RESETHAND");
            returns_twice(&wrapped, "wrapped without SA_RESETHAND");
            assert_eq!(reference.first, wrapped.first, "signal {signal}");
            assert_eq!(reference.second, wrapped.second, "signal {signal}");
            println!("NO_RESET_IDENTICAL signal={signal}");
        }
    }
}

/// The installed signal route under real delivery. A returning predecessor lets
/// the child inspect what capture wrote. libmalloc's `malloc_logger` counts
/// every malloc and free from arming (just before the raise) to the
/// predecessor's entry — the whole capture — so "no allocation" is measured
/// at the allocator, below Rust's, with a positive control on the instrument.
/// ASan replaces malloc below that hook, so under ASan the instrument is blind
/// (pinned at zero) and the allocation count belongs to the uninstrumented
/// build; delivery, replay, refusal and the backtrace are asserted under both.
#[cfg(test)]
mod signal_capture_controls {
    use super::*;
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    const CHILD: &str = "LLOOM_RECORD_SIGNAL_CAPTURE_CHILD";
    const IDENTITY: &[u8] = b"signal-capture-control";
    static EVENTS: AtomicUsize = AtomicUsize::new(0);
    static EVENTS_AT_PREDECESSOR: AtomicUsize = AtomicUsize::new(usize::MAX);
    static ERRNO_AT_PREDECESSOR: AtomicI32 = AtomicI32::new(-1);
    static CALLS: AtomicI32 = AtomicI32::new(0);

    extern "C" fn count(_: u32, _: usize, _: usize, _: usize, _: usize, _: u32) {
        EVENTS.fetch_add(1, Ordering::Relaxed);
    }

    static HOOK: AtomicUsize = AtomicUsize::new(0);

    fn asan_runtime() -> bool {
        // SAFETY: dlsym of an exported function name; null if absent.
        !unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"__asan_init".as_ptr()) }.is_null()
    }

    fn instrument_marker() -> &'static str {
        if asan_runtime() {
            "INSTRUMENT_BLIND_UNDER_ASAN"
        } else {
            "INSTRUMENT_POSITIVE_CONTROL"
        }
    }

    /// The ASan runtime's `detect_stack_use_after_return` as this process
    /// reads it; `None` without the runtime.
    fn use_after_return_option() -> Option<libc::c_int> {
        // SAFETY: dlsym of an exported data symbol; null if absent.
        let option = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                c"__asan_option_detect_stack_use_after_return".as_ptr(),
            )
        };
        // SAFETY: the runtime's option is a C int it initialised before main.
        (!option.is_null())
            .then(|| unsafe { std::ptr::read_volatile(option as *const libc::c_int) })
    }

    /// Where the last capture's frame buffer lived.
    fn handler_frame_location() -> &'static str {
        let addr = handler_frame::ADDR.load(Ordering::Relaxed);
        assert_ne!(addr, 0, "no capture frame was noted");
        if handler_frame::IN_FAKE.load(Ordering::Relaxed) == 2 {
            return "fake";
        }
        // SAFETY: queries of this thread's stack and alternate stack only.
        unsafe {
            let me = libc::pthread_self();
            let top = libc::pthread_get_stackaddr_np(me) as usize;
            let size = libc::pthread_get_stacksize_np(me);
            if addr < top && addr >= top - size {
                return "thread";
            }
            let mut alt: libc::stack_t = std::mem::zeroed();
            assert_eq!(libc::sigaltstack(std::ptr::null(), &mut alt), 0);
            let base = alt.ss_sp as usize;
            if base != 0 && addr >= base && addr < base + alt.ss_size {
                return "altstack";
            }
        }
        "other"
    }

    /// A fresh alternate signal stack for this thread, outside the heap.
    fn install_altstack() {
        const SIZE: usize = 256 * 1024;
        // SAFETY: an anonymous private mapping, kept for the child's life.
        unsafe {
            let base = libc::mmap(
                std::ptr::null_mut(),
                SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert_ne!(base, libc::MAP_FAILED);
            let stack = libc::stack_t {
                ss_sp: base,
                ss_size: SIZE,
                ss_flags: 0,
            };
            assert_eq!(libc::sigaltstack(&stack, std::ptr::null_mut()), 0);
        }
    }

    /// Resolve libmalloc's `malloc_logger` variable once, before anything is
    /// measured: dlsym itself takes dyld's lock and may allocate.
    fn resolve_logger() {
        // SAFETY: dlsym of an exported data symbol; null if absent.
        let hook = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"malloc_logger".as_ptr()) };
        assert!(!hook.is_null(), "libmalloc's malloc_logger is not exported");
        HOOK.store(hook as usize, Ordering::Relaxed);
    }

    fn logger(on: bool) {
        let value = if on { count as *const () as usize } else { 0 };
        // SAFETY: HOOK is libmalloc's function-pointer variable (resolved and
        // checked non-null); a volatile store of null or a matching C function
        // is how its own tools arm it.
        unsafe { std::ptr::write_volatile(HOOK.load(Ordering::Relaxed) as *mut usize, value) };
    }

    extern "C" fn predecessor(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
        logger(false);
        EVENTS_AT_PREDECESSOR.store(EVENTS.load(Ordering::Relaxed), Ordering::Relaxed);
        ERRNO_AT_PREDECESSOR.store(errno::errno().0, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn deliver() -> usize {
        EVENTS.store(0, Ordering::Relaxed);
        EVENTS_AT_PREDECESSOR.store(usize::MAX, Ordering::Relaxed);
        errno::set_errno(errno::Errno(libc::EDOM));
        logger(true);
        // SAFETY: a returning predecessor is registered behind capture.
        assert_eq!(unsafe { libc::raise(libc::SIGILL) }, 0);
        assert_eq!(errno::errno().0, libc::EDOM, "interrupted errno leaked");
        assert_eq!(ERRNO_AT_PREDECESSOR.load(Ordering::Relaxed), libc::EDOM);
        EVENTS_AT_PREDECESSOR.load(Ordering::Relaxed)
    }

    fn records(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.starts_with("record-"))
            .collect();
        names.sort();
        names
    }

    fn child(mode: &str) {
        // SAFETY: cold setup in a disposable child; no core files.
        let zero = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_CORE, &zero) }, 0);
        let dir = std::env::temp_dir().join(format!("lloom-signal-capture-{}", std::process::id()));
        let record = Record::install_with_identity(&dir, 4, IDENTITY.to_vec()).unwrap();
        for input in [b"first".as_slice(), b"", b"dying input"] {
            record.push(input).unwrap();
        }
        let onstack = mode == "capture-onstack";
        if onstack {
            install_altstack();
        }
        // SAFETY: the action is fully initialized before registration.
        let mut prior: libc::sigaction = unsafe { std::mem::zeroed() };
        prior.sa_sigaction = predecessor as *const () as usize;
        prior.sa_flags = libc::SA_SIGINFO | if onstack { libc::SA_ONSTACK } else { 0 };
        assert_eq!(
            unsafe {
                libc::sigemptyset(&mut prior.sa_mask);
                libc::sigaction(libc::SIGILL, &prior, std::ptr::null_mut())
            },
            0
        );
        Record::install_crash_dump().unwrap();
        handler_frame::resolve();
        resolve_logger();
        logger(true);
        EVENTS.store(0, Ordering::Relaxed);
        drop(std::hint::black_box(vec![0u8; 64]));
        logger(false);
        let control = EVENTS.load(Ordering::Relaxed);
        let asan = asan_runtime();
        if asan {
            assert_eq!(
                control, 0,
                "ASan's allocator now reaches malloc_logger: assert the allocation count here too"
            );
        } else {
            assert!(
                control >= 2,
                "instrument blind: {control} events for one allocation"
            );
        }
        println!("{} events={control}", instrument_marker());
        let allocation_free = |during: usize| {
            if !asan {
                assert_eq!(during, 0, "the signal route reached the allocator");
            }
        };
        let pid = std::process::id();
        match mode {
            "capture" | "capture-onstack" => {
                let during = deliver();
                println!("CAPTURE_ALLOCATOR_EVENTS={during}");
                assert_eq!(CALLS.load(Ordering::Relaxed), 1);
                allocation_free(during);
                let option = use_after_return_option();
                let location = handler_frame_location();
                println!(
                    "HANDLER_FRAME={location} UAR_OPTION={option:?} IN_FAKE_QUERY={}",
                    handler_frame::IN_FAKE.load(Ordering::Relaxed)
                );
                let fake = option.unwrap_or(0) != 0;
                let own = if onstack { "altstack" } else { "thread" };
                if fake && onstack {
                    // Reported, not pinned: where ASan places a fake frame
                    // taken on an alternate signal stack is what this measures.
                    assert!(matches!(location, "fake" | "altstack"), "{location}");
                } else if fake {
                    assert_eq!(location, "fake", "use-after-return on, frame not fake");
                } else {
                    assert_eq!(location, own, "capture frame outside its stack");
                }
                let archive = dir.join(format!("record-{pid}-0"));
                let mut replayed = Vec::new();
                let n = replay_with_identity(&archive, Some(IDENTITY), false, |bytes| {
                    replayed.push(bytes.to_vec());
                    Ok(())
                })
                .unwrap();
                assert_eq!(n, 3);
                assert_eq!(
                    replayed,
                    [b"first".to_vec(), Vec::new(), b"dying input".to_vec()]
                );
                assert_eq!(
                    fs::read(archive.join("reason.txt")).unwrap(),
                    b"signal 04\n"
                );
                let bt = fs::read_to_string(archive.join("bt.txt")).unwrap();
                let image = std::env::current_exe().unwrap();
                assert!(bt.starts_with("si_addr=0x"), "{bt}");
                assert!(!bt.contains("image_base=0x0000000000000000"), "{bt}");
                assert!(
                    bt.contains(&format!("image_path={}\n", image.display())),
                    "{bt}"
                );
                let pcs = bt.lines().filter(|l| l.starts_with("pc=0x")).count();
                assert!(pcs > 0, "{bt}");
                print!("BT_TXT_BEGIN\n{bt}BT_TXT_END\n");
                println!("SIGNAL_CAPTURE_REPLAYED inputs={n} pcs={pcs}");
            }
            "busy" => {
                let held = RING.get().unwrap().inputs.snapshot().unwrap();
                let during = deliver();
                println!("BUSY_ALLOCATOR_EVENTS={during}");
                assert_eq!(CALLS.load(Ordering::Relaxed), 1);
                allocation_free(during);
                drop(held);
                assert!(
                    records(&dir).is_empty(),
                    "a held snapshot must refuse capture"
                );
                let during = deliver();
                assert_eq!(CALLS.load(Ordering::Relaxed), 2);
                allocation_free(during);
                assert_eq!(
                    records(&dir),
                    [format!("record-{pid}-0")],
                    "a refused capture must not consume the dump sequence"
                );
                println!("BUSY_REFUSED_THEN_CAPTURED");
            }
            other => panic!("unknown mode {other}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    fn run(mode: &str, marker: &str) {
        if std::env::var_os(CHILD).is_some() {
            child(mode);
            return;
        }
        let name = std::thread::current().name().unwrap().to_owned();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(CHILD, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("{stdout}");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success(), "delivered capture child failed");
        assert!(stdout.contains(instrument_marker()));
        assert!(stdout.contains(marker), "child did not reach {marker}");
    }

    /// The handler's libSystem calls must not reach dyld's lazy binder: this
    /// image (the one the delivered controls run in) carries no lazy binds.
    #[test]
    fn this_image_binds_every_import_at_load() {
        const LC_DYLD_INFO: u32 = 0x22;
        const LC_DYLD_INFO_ONLY: u32 = 0x8000_0022;
        const LC_DYLD_CHAINED_FIXUPS: u32 = 0x8000_0034;
        let image = fs::read(std::env::current_exe().unwrap()).unwrap();
        let word = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
        assert_eq!(
            word(0),
            0xfeed_facf,
            "expected a thin 64-bit Mach-O test binary"
        );
        let (mut at, mut lazy, mut chained) = (32, None, false);
        for _ in 0..word(16) {
            let (cmd, size) = (word(at), word(at + 4) as usize);
            if cmd == LC_DYLD_INFO || cmd == LC_DYLD_INFO_ONLY {
                lazy = Some(word(at + 36));
            }
            chained |= cmd == LC_DYLD_CHAINED_FIXUPS;
            at += size;
        }
        println!("LAZY_BIND_SIZE={lazy:?} CHAINED_FIXUPS={chained}");
        assert!(
            chained || lazy == Some(0),
            "lazy binds present ({lazy:?} bytes): the signal route's first libSystem \
             call would run dyld's stub binder inside the handler"
        );
    }
    #[test]
    fn delivered_signal_writes_a_replayable_record_without_allocating() {
        run("capture", "SIGNAL_CAPTURE_REPLAYED inputs=3");
    }
    /// The campaign's own predecessors set `SA_ONSTACK`, so its handler runs on
    /// an alternate stack; the record must hold there as on the thread stack.
    #[test]
    fn delivered_signal_on_an_alternate_stack_writes_a_replayable_record() {
        run("capture-onstack", "SIGNAL_CAPTURE_REPLAYED inputs=3");
    }
    #[test]
    fn delivered_signal_refuses_a_held_snapshot_without_blocking() {
        run("busy", "BUSY_REFUSED_THEN_CAPTURED");
    }
}

/// The ASan arm's positive controls: planted overflows in this crate, built
/// with the arm's instrumentation, must be reported, or a clean record suite
/// under ASan says nothing. Compiled only into that arm (`lloom_fuzz_asan`,
/// set by `sancov-wrapper.sh`), so no other build can pass them vacuously.
#[cfg(all(test, lloom_fuzz_asan))]
mod sanitizer_controls {
    const CHILD: &str = "LLOOM_RECORD_ASAN_CONTROL_CHILD";

    #[inline(never)]
    fn read_at(base: *const u8, at: usize) -> u8 {
        // SAFETY: none; the out-of-bounds read is the planted defect.
        unsafe { std::ptr::read_volatile(base.add(at)) }
    }

    fn planted(kind: &str) {
        match kind {
            "heap" => {
                let heap = std::hint::black_box(vec![7u8; 32]);
                println!(
                    "UNREPORTED={}",
                    read_at(heap.as_ptr(), std::hint::black_box(32))
                );
            }
            "stack" => {
                let stack = std::hint::black_box([7u8; 32]);
                println!(
                    "UNREPORTED={}",
                    read_at(stack.as_ptr(), std::hint::black_box(32))
                );
            }
            "return" => {
                let gone = dangling();
                println!("UNREPORTED={}", read_at(gone, 0));
            }
            other => panic!("unknown control {other}"),
        }
    }

    /// A pointer into a frame that has returned. The address leaves through a
    /// call, not `black_box`: `black_box` is inline asm, and ASan gives a
    /// function holding inline asm no fake frame, so nothing is poisoned at
    /// its return and the read below would be invisible to the arm.
    #[inline(never)]
    fn dangling() -> *const u8 {
        let local = [7u8; 32];
        escape(&local)
    }

    #[inline(never)]
    fn escape(local: &[u8; 32]) -> *const u8 {
        local.as_ptr()
    }

    /// The runtime's `detect_stack_use_after_return`, as this process reads it.
    fn use_after_return_option() -> libc::c_int {
        // SAFETY: dlsym of an exported data symbol; null if absent.
        let option = unsafe {
            libc::dlsym(
                libc::RTLD_DEFAULT,
                c"__asan_option_detect_stack_use_after_return".as_ptr(),
            )
        };
        assert!(!option.is_null(), "ASan runtime without the option symbol");
        // SAFETY: the runtime's option is a C int it initialised before main.
        unsafe { std::ptr::read_volatile(option as *const libc::c_int) }
    }

    fn child_output(kind: &str) -> std::process::Output {
        // SAFETY: dlsym of an exported function name; null if absent.
        let runtime =
            !unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"__asan_init".as_ptr()) }.is_null();
        assert!(runtime, "lloom_fuzz_asan build without the ASan runtime");
        let name = std::thread::current().name().unwrap().to_owned();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(CHILD, kind)
            .output()
            .unwrap();
        eprintln!(
            "[{kind}] status {:?}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn reported(kind: &str, report: &str) {
        if let Some(mode) = std::env::var_os(CHILD) {
            planted(mode.to_str().unwrap());
            return;
        }
        let output = child_output(kind);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "planted {kind} overflow ran clean"
        );
        assert!(
            !stdout.contains("UNREPORTED="),
            "planted {kind} overflow returned"
        );
        assert!(
            stderr.contains(&format!("ERROR: AddressSanitizer: {report}")),
            "no {report} report"
        );
        println!("ASAN_POSITIVE_CONTROL={report}");
    }

    #[test]
    fn asan_reports_a_planted_heap_overflow() {
        reported("heap", "heap-buffer-overflow");
    }
    #[test]
    fn asan_reports_a_planted_stack_overflow() {
        reported("stack", "stack-buffer-overflow");
    }
    /// A read through a returned frame is reported exactly when the runtime
    /// detects use-after-return, and runs clean when it does not. An arm states
    /// the level it set in `LLOOM_EXPECT_UAR`, so one that believes it set the
    /// option and did not fails here rather than passing the other branch.
    #[test]
    fn asan_reports_a_planted_use_after_return_iff_detection_is_on() {
        if let Some(mode) = std::env::var_os(CHILD) {
            planted(mode.to_str().unwrap());
            return;
        }
        let option = use_after_return_option();
        println!("UAR_OPTION={option}");
        if let Some(expected) = std::env::var_os("LLOOM_EXPECT_UAR") {
            assert_eq!(
                option.to_string(),
                expected.to_str().unwrap(),
                "the arm's detect_stack_use_after_return did not take"
            );
        }
        if option != 0 {
            reported("return", "stack-use-after-return");
            return;
        }
        let output = child_output("return");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "detection off, child failed");
        assert!(
            stdout.contains("UNREPORTED="),
            "detection off, read did not return"
        );
        assert!(
            !stderr.contains("ERROR: AddressSanitizer"),
            "detection off, yet reported"
        );
        println!("ASAN_USE_AFTER_RETURN_OFF_PINNED");
    }
}
