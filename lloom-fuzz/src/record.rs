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
//!    handlers installed by the same function, which write the ring with
//!    `write(2)` and re-raise. LibAFL's own in-process handlers run too; ours are
//!    chained ahead of them so the record exists before the fuzzer's report.
//!
//! The dump is a directory `record-<pid>-<seq>/` holding `0000.bin … NNNN.bin`
//! (oldest first, the dying input last), `reason.txt`, and a versioned
//! completion manifest. [`replay`] validates counts/sizes, then executes
//! a directory in order through the same harness and reports which entry fails.
//!
//! The ring lives in a global because the signal path cannot take a lock or
//! allocate: the slot buffers are pre-allocated at [`Record::install`] and a
//! push is a `memcpy` under a nonblocking atomic permit. Contention refuses
//! recording rather than racing the payload. Signal-safe I/O and handler
//! composition remain separate obligations: the current dump still allocates.

mod archive;
pub(crate) mod identity;
pub(crate) use archive::read_single;
mod inputs;
mod signals;

use inputs::InputRing;
pub use inputs::RecordError;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The largest input the ring accepts. Larger inputs return an explicit error;
/// the campaign must not execute bytes it failed to record.
pub const MAX_INPUT: usize = 64 * 1024;

struct Ring {
    inputs: InputRing,
    dir: PathBuf,
    dumps: AtomicUsize,
    dumping: AtomicBool,
    identity: Option<Vec<u8>>,
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
        if identity.is_empty() || identity.len() > archive::MAX_IDENTITY {
            return Err(std::io::Error::other(
                "record identity is empty or exceeds its bound",
            ));
        }
        std::fs::create_dir_all(dir)?;
        let n = n.max(1);
        let ring = RING.get_or_init(|| Ring {
            inputs: InputRing::new(n, MAX_INPUT),
            dir: dir.to_path_buf(),
            dumps: AtomicUsize::new(0),
            dumping: AtomicBool::new(false),
            identity: Some(identity.clone()),
        });
        if ring.identity.as_deref() != Some(identity.as_slice())
            || ring.dir != dir
            || ring.inputs.capacity() != n
        {
            return Err(std::io::Error::other(
                "persistent recorder already installed with different identity/path/capacity",
            ));
        }
        Ok(Record)
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
    /// owns the history. The signal path currently shares this allocating I/O;
    /// async-signal safety is not claimed by the payload ownership repair.
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

extern "C" fn on_signal(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    if let Some(r) = RING.get() {
        let mut reason = *b"signal 00\n";
        reason[7] = b'0' + ((sig / 10) % 10) as u8;
        reason[8] = b'0' + (sig % 10) as u8;
        if let Some(dir) = dump_raw(r, &reason) {
            // The faulting address and the native backtrace, through
            // `backtrace_symbols_fd` (libSystem's, writes straight to the fd).
            // Not strictly async-signal-safe -- it is the last thing this
            // handler does before chaining, and a crash here loses only the
            // backtrace, never the record written above.
            if let Ok(f) = fs::File::create(dir.join("bt.txt")) {
                use std::os::unix::io::AsRawFd;
                let fd = f.as_raw_fd();
                // SAFETY: `info` is the kernel's siginfo for this delivery.
                let addr = if info.is_null() {
                    0usize
                } else {
                    unsafe { (*info).si_addr as usize }
                };
                let mut line = *b"si_addr=0x0000000000000000\n";
                for i in 0..16 {
                    let nib = ((addr >> (4 * (15 - i))) & 0xf) as u8;
                    line[10 + i] = if nib < 10 {
                        b'0' + nib
                    } else {
                        b'a' + nib - 10
                    };
                }
                let _ = (&f).write_all(&line);
                let mut frames = [std::ptr::null_mut::<libc::c_void>(); 96];
                // SAFETY: a fixed-size frame buffer; libSystem walks the stack.
                unsafe {
                    let n = libc::backtrace(frames.as_mut_ptr(), frames.len() as libc::c_int);
                    libc::backtrace_symbols_fd(frames.as_ptr(), n, fd);
                }
            }
        }
    }
    // SAFETY: forward the original kernel delivery to the ABI and policy
    // retained from the predecessor. Allocating dump/backtrace I/O above is
    // still a separate async-signal-safety defect, not repaired by this chain.
    unsafe { signals::chain(sig, info, ctx) };
}

/// Write a coherent ring snapshot. This still allocates and is NOT yet an
/// async-signal-safe I/O implementation. Busy snapshots return no directory.
fn dump_raw(r: &Ring, reason: &[u8]) -> Option<PathBuf> {
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
    // The hook can interrupt a writer or another thread can be recording.
    // Never read racing bytes or block waiting for the interrupted thread.
    let snapshot = r.inputs.snapshot().ok()?;
    if snapshot.inputs().next().is_none() {
        return None;
    }
    let seq = r
        .dumps
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        .ok()?;
    let dir = r.dir.join(format!("record-{}-{seq}", std::process::id()));
    archive::write_identified(&dir, snapshot.inputs(), reason, r.identity.as_deref()).ok()?;
    Some(dir)
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
    mut harness: F,
) -> Result<usize, (usize, String)>
where
    F: FnMut(&[u8]) -> Result<(), String>,
{
    let replay = archive::prepare(path, MAX_INPUT)
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
    for (i, (p, length)) in replay.paths.iter().zip(&replay.lengths).enumerate() {
        let bytes = archive::read(p, *length, MAX_INPUT)
            .map_err(|error| (i, format!("{}: {error}", p.display())))?;
        harness(&bytes).map_err(|m| (i, format!("{}: {m}", p.display())))?;
    }
    Ok(replay.paths.len())
}
