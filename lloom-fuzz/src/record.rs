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
//! (oldest first, the dying input last) and `reason.txt`. [`replay`] executes
//! a directory in order through the same harness and reports which entry fails.
//!
//! The ring lives in a global because the signal path cannot take a lock or
//! allocate: the slot buffers are pre-allocated at [`Record::install`] and a
//! push is a `memcpy` plus two atomics.

use std::cell::UnsafeCell;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The largest input the ring keeps whole; longer inputs are truncated in the
/// record (and the harnesses cap their inputs below this anyway).
pub const MAX_INPUT: usize = 64 * 1024;

struct Slot {
    len: AtomicUsize,
    bytes: UnsafeCell<Box<[u8]>>,
}

// SAFETY: a slot is written only by the fuzz thread (`push`) and read by the
// same thread's dump path or by a signal handler interrupting that thread; the
// handler reads `len` first, and a torn read of a slot being written can only
// mis-record the *dying* input, which the fuzzer's own crash file also holds.
unsafe impl Sync for Slot {}

struct Ring {
    slots: Vec<Slot>,
    /// Total pushes so far; `head % slots.len()` is the next slot to write.
    pushes: AtomicUsize,
    dir: PathBuf,
    dumps: AtomicUsize,
    dumping: AtomicBool,
}

static RING: std::sync::OnceLock<Ring> = std::sync::OnceLock::new();

/// The record's handle. One per process; [`Record::install`] once, then
/// [`Record::push`] per execution and [`Record::dump`] on an oracle failure.
#[derive(Clone, Copy, Debug)]
pub struct Record;

impl Record {
    /// Install the ring: `n` slots of [`MAX_INPUT`] bytes under `dir`, and the
    /// panic hook and signal handlers that dump it. Idempotent after the first
    /// call (the first `n`/`dir` win).
    pub fn install(dir: &Path, n: usize) -> Record {
        let n = n.max(1);
        RING.get_or_init(|| {
            fs::create_dir_all(dir).ok();
            Ring {
                slots: (0..n)
                    .map(|_| Slot {
                        len: AtomicUsize::new(usize::MAX),
                        bytes: UnsafeCell::new(vec![0u8; MAX_INPUT].into_boxed_slice()),
                    })
                    .collect(),
                pushes: AtomicUsize::new(0),
                dir: dir.to_path_buf(),
                dumps: AtomicUsize::new(0),
                dumping: AtomicBool::new(false),
            }
        });
        Record::install_crash_dump();
        Record
    }

    /// How many slots the ring holds.
    pub fn capacity(self) -> usize {
        RING.get().map_or(0, |r| r.slots.len())
    }

    /// Record `input` as the next execution. Call BEFORE running it.
    pub fn push(self, input: &[u8]) {
        let Some(r) = RING.get() else { return };
        let i = r.pushes.fetch_add(1, Ordering::AcqRel) % r.slots.len();
        let slot = &r.slots[i];
        // Mark the slot as being written so a handler interrupting the copy
        // does not read a half-old, half-new input as a whole one.
        slot.len.store(usize::MAX, Ordering::Release);
        let n = input.len().min(MAX_INPUT);
        // SAFETY: the fuzz thread is the only writer (see `Slot`).
        let buf: &mut Box<[u8]> = unsafe { &mut *slot.bytes.get() };
        buf[..n].copy_from_slice(&input[..n]);
        slot.len.store(n, Ordering::Release);
    }

    /// Write the ring, oldest first, and the reason. Returns the directory.
    ///
    /// Safe to call from the harness (allocates, formats). The signal path uses
    /// [`dump_raw`] instead.
    pub fn dump(self, reason: &str) -> Option<PathBuf> {
        let r = RING.get()?;
        let dir = dump_raw(r, reason.as_bytes())?;
        Some(dir)
    }

    /// The panic hook and signal handlers. Installed by [`Record::install`];
    /// exposed so a harness that manages its own ring can still chain them.
    pub fn install_crash_dump() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if let Some(r) = RING.get() {
                    let msg = format!("panic: {info}");
                    let _ = dump_raw(r, msg.as_bytes());
                }
                prev(info);
            }));
            // SAFETY: installing async-signal handlers that only call
            // async-signal-safe functions (`open`, `write`, `close`, `mkdir`,
            // atomics) and then re-raise with the default disposition.
            unsafe {
                for sig in [
                    libc::SIGSEGV,
                    libc::SIGBUS,
                    libc::SIGABRT,
                    libc::SIGILL,
                    libc::SIGFPE,
                ] {
                    let mut sa: libc::sigaction = std::mem::zeroed();
                    let mut prev: libc::sigaction = std::mem::zeroed();
                    sa.sa_sigaction = on_signal as *const () as usize;
                    sa.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER;
                    libc::sigemptyset(&mut sa.sa_mask);
                    if libc::sigaction(sig, &sa, &mut prev) == 0 {
                        PREV_HANDLERS[sig_index(sig)].store(
                            prev.sa_sigaction,
                            Ordering::Release,
                        );
                    }
                }
            }
        });
    }
}

const SIGNALS: [libc::c_int; 5] = [
    libc::SIGSEGV,
    libc::SIGBUS,
    libc::SIGABRT,
    libc::SIGILL,
    libc::SIGFPE,
];
static PREV_HANDLERS: [AtomicUsize; 5] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

fn sig_index(sig: libc::c_int) -> usize {
    SIGNALS.iter().position(|&s| s == sig).unwrap_or(0)
}

extern "C" fn on_signal(sig: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    if let Some(r) = RING.get() {
        let mut reason = *b"signal 00\n";
        reason[7] = b'0' + ((sig / 10) % 10) as u8;
        reason[8] = b'0' + (sig % 10) as u8;
        let _ = dump_raw(r, &reason);
    }
    // Chain to whatever was installed before us (LibAFL's handler, or the
    // default), so the fuzzer still sees the crash.
    let prev = PREV_HANDLERS[sig_index(sig)].load(Ordering::Acquire);
    // SAFETY: `prev` is what `sigaction` reported, and it is invoked with the
    // signature it was registered under; the default disposition re-raises.
    unsafe {
        if prev == libc::SIG_DFL || prev == 0 {
            libc::signal(sig, libc::SIG_DFL);
            libc::raise(sig);
        } else if prev != libc::SIG_IGN {
            let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                std::mem::transmute(prev);
            f(sig, info, ctx);
        }
    }
}

/// Write the ring with nothing but `open`/`write`/`close`/`mkdir` — the
/// signal-safe subset. Returns the directory written.
fn dump_raw(r: &Ring, reason: &[u8]) -> Option<PathBuf> {
    if r.dumping.swap(true, Ordering::AcqRel) {
        return None; // re-entrant (a crash inside a dump): leave the first one
    }
    let seq = r.dumps.fetch_add(1, Ordering::AcqRel);
    let dir = r.dir.join(format!("record-{}-{seq}", std::process::id()));
    fs::create_dir_all(&dir).ok()?;
    let total = r.pushes.load(Ordering::Acquire);
    let n = r.slots.len();
    let oldest = total.saturating_sub(n);
    let mut k = 0usize;
    for p in oldest..total {
        let slot = &r.slots[p % n];
        let len = slot.len.load(Ordering::Acquire);
        if len == usize::MAX {
            continue; // never written, or mid-copy
        }
        // SAFETY: read-only view of a slot whose `len` was published.
        let buf: &Box<[u8]> = unsafe { &*slot.bytes.get() };
        let bytes = &buf[..len];
        if let Ok(mut f) = fs::File::create(dir.join(format!("{k:04}.bin"))) {
            let _ = f.write_all(bytes);
        }
        k += 1;
    }
    if let Ok(mut f) = fs::File::create(dir.join("reason.txt")) {
        let _ = f.write_all(reason);
        let _ = f.write_all(b"\n");
    }
    r.dumping.store(false, Ordering::Release);
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
pub fn replay<F>(path: &Path, mut harness: F) -> Result<usize, (usize, String)>
where
    F: FnMut(&[u8]) -> Result<(), String>,
{
    let inputs: Vec<PathBuf> = if path.is_dir() {
        let mut v: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| (0, format!("{}: {e}", path.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .collect();
        v.sort();
        v
    } else {
        vec![path.to_path_buf()]
    };
    for (i, p) in inputs.iter().enumerate() {
        let bytes = fs::read(p).map_err(|e| (i, format!("{}: {e}", p.display())))?;
        harness(&bytes).map_err(|m| (i, format!("{}: {m}", p.display())))?;
    }
    Ok(inputs.len())
}
