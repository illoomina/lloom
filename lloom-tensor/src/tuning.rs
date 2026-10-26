//! The tensor crate's tuning knobs, as one option group.
//!
//! Every `LLOOM_*` / `CANDLE_*` variable a kernel, the command stream, the
//! buffer pool or the loader used to read through its own `OnceLock` is a
//! field here, with that variable as its env alias — so the scripts that set
//! them keep working. Each field also declares its key in the option tree
//! (`#[opt(path = "kernels.quant_matvec.q6_k.rows_per_simdgroup")]`), which
//! `--set`, `--config`, `LLOOM_<KEY>` and `lloom options` take; the storage
//! path (`tuning.qmv.q6k_nr0`) and the old flag stay accepted.
//!
//! **Nothing on the hot path changes.** Every read site keeps its `OnceLock`
//! (or atomic) and initialises it from [`tuning()`] instead of from
//! `std::env::var` — the same one-time read, the same cached load after. The
//! group itself is read once: [`install`] before the first kernel runs (the
//! `lloom` binary, after resolving its option layers), else the defaults
//! with the env aliases applied on first use — byte for byte what the
//! per-site reads did.
//!
//! An env alias set to a value a knob does not accept is an error at first
//! use (§2.4: a run that silently executes the baseline while its operator
//! believes otherwise is the failure the whole knob exists to avoid); where a
//! site used to fall back silently, this is a deliberate tightening.

use crate::options::{before, is_none, non_empty, parse_arm, parse_count, parse_number, BeforeRow};
use lloom_macros::Options;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Whether optional attribution machinery is compiled into this tensor build.
pub const PROFILE_INSTRUMENTS: bool = cfg!(any(debug_assertions, feature = "profile"));

/// Require the detailed Metal recorder at a cold consumer boundary.
/// This does not enable profiling or reject scalar-only profiling requests.
/// The capability/key come from the existing generated instrument declaration.
pub fn require_profile_recorder() -> Result<(), TuningSetupError> {
    let meta = crate::options::meta_at(InstrumentTuning::META, "profile")
        .expect("profile is declared by the generated instrument schema");
    if !cfg!(feature = "metal") || !PROFILE_INSTRUMENTS {
        return Err(TuningSetupError::NotCompiled {
            key: meta.path,
            capability: if cfg!(feature = "metal") {
                meta.needs
            } else {
                "metal"
            },
        });
    }
    Ok(())
}

/// An ordinary setup refusal, distinct from a backend invariant failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningSetupError {
    AlreadyRead,
    InvalidEnvironment(String),
    NotCompiled {
        key: &'static str,
        capability: &'static str,
    },
}

impl std::fmt::Display for TuningSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRead => f.write_str("tuning: the tensor crate's knobs were already read before install"),
            Self::InvalidEnvironment(reason) => write!(f, "tuning: {reason}"),
            Self::NotCompiled { key, capability } => write!(f,
                "{key}: {capability} instrumentation is not compiled; rebuild with cargo build --profile release-with-debug -p lloom-tensor --features metal,profile"),
        }
    }
}
impl std::error::Error for TuningSetupError {}

/// A failed install retains the incoming configuration, unchanged.
#[derive(Debug)]
pub struct InstallError {
    pub reason: TuningSetupError,
    tuning: Box<Tuning>,
}
impl InstallError {
    pub fn into_tuning(self) -> Tuning {
        *self.tuning
    }
}
impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.reason.fmt(f)
    }
}
impl std::error::Error for InstallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.reason)
    }
}

/// Validate one direct instrumentation request against its generated declaration.
pub(crate) fn check_profile_request(storage: &str, on: bool) -> Result<(), TuningSetupError> {
    let meta = crate::options::meta_at(InstrumentTuning::META, storage)
        .expect("instrument control must name its generated option");
    if on && !PROFILE_INSTRUMENTS && meta.needs == "profile" && meta.needs_part.is_empty() {
        return Err(TuningSetupError::NotCompiled {
            key: meta.path,
            capability: meta.needs,
        });
    }
    Ok(())
}

/// Legacy cached status means effectively active, not merely requested.
/// An unavailable request is reported once by each caller's status cache;
/// primary control/setup APIs return the typed error instead.
#[cfg(feature = "metal")]
pub(crate) fn effective_instrument(storage: &str, requested: bool) -> bool {
    match check_profile_request(storage, requested) {
        Ok(()) => requested,
        Err(error) => {
            eprintln!("{error}");
            false
        }
    }
}

impl Tuning {
    /// Cold setup/legacy-cache notice, generated from the same partial-capability
    /// metadata as the engine. No notice or side effect in diagnostic builds.
    pub(crate) fn note_partial_instruments(&self) {
        if PROFILE_INSTRUMENTS || self.instrument == InstrumentTuning::default() {
            return;
        }
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| {
            let values = serde_json::to_value(&self.instrument).expect("instrument group serializes");
            let defaults = serde_json::to_value(InstrumentTuning::default()).expect("instrument defaults serialize");
            for meta in InstrumentTuning::META {
                if meta.needs == "profile" && !meta.needs_part.is_empty()
                    && values.get(meta.key) != defaults.get(meta.key) {
                    eprintln!("[diagnostics] {}: {} require profile instrumentation absent from this build; remaining scalar output is available; rebuild with cargo build --profile release-with-debug -p lloom-tensor --features metal,profile",meta.path,meta.needs_part);
                }
            }
        });
    }

    /// Reject unavailable whole instruments, preserving partial scalar-counter routes.
    pub fn validate_instruments(&self) -> Result<(), TuningSetupError> {
        if PROFILE_INSTRUMENTS || self.instrument == InstrumentTuning::default() {
            return Ok(());
        }
        let values = serde_json::to_value(&self.instrument).expect("instrument group serializes");
        let defaults = serde_json::to_value(InstrumentTuning::default())
            .expect("instrument defaults serialize");
        for meta in InstrumentTuning::META {
            if values.get(meta.key) != defaults.get(meta.key) {
                check_profile_request(meta.key, true)?;
            }
        }
        Ok(())
    }
}

static TUNING: OnceLock<Tuning> = OnceLock::new();

/// The determinism contract a run is held to (`--profile`, `LLOOM_PROFILE`;
/// `architecture.md` §2.3.5b).
///
/// `Fast` is the default (RESTRUCTURING_DESIGN.md §2): the same
/// distribution, not the same bits — it licenses the arms whose bits depend
/// on who else is in the batch, on completion order, or on which math path
/// verifies a speculation. `Deterministic` is the opt-in and the contract
/// every gate, test and digest runs, pinned explicitly ([`pin_contract`],
/// [`gate_contract`]) rather than inherited: the same inputs and seed give
/// the same bits, whatever the scheduling, the pool size, the tile split or
/// the lookahead. Every fast arm has a deterministic twin the bit-identity
/// suites cover; it is judged by the quality gate, never in place of the
/// bit gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contract {
    Deterministic,
    Fast,
}

impl Contract {
    pub const SPELLINGS: [&'static str; 2] = ["deterministic", "fast"];

    pub fn parse(v: &str) -> Result<Contract, String> {
        match v.trim().to_ascii_lowercase().as_str() {
            "deterministic" => Ok(Contract::Deterministic),
            "fast" => Ok(Contract::Fast),
            other => Err(format!(
                "profile: `{other}` is not one of {}",
                Contract::SPELLINGS.join(", ")
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Contract::Deterministic => "deterministic",
            Contract::Fast => "fast",
        }
    }

    pub fn is_fast(self) -> bool {
        self == Contract::Fast
    }

    /// The arms `fast` takes, by name (`architecture.md` §2.3.5b). An arm
    /// joins this list in the change that makes it read [`contract`], and
    /// that change pins the bit suites that reach it.
    ///
    /// - `moe-act-f16`: Qwen4-exp's routed experts on f16 activations where
    ///   the matmul-vec runs (`lloom`'s `--moe-act auto`).
    /// - `decision-pack`: `/v1/systemone` requests that wait together are read
    ///   as one tree, so an answer's prefill partition depends on what else
    ///   arrived (`lloom-engine`'s `decide_together`).
    /// - `decision-ragged`: a `/v1/systemone` request's questions are read as
    ///   one ragged batch over its state, which is kept for the next request
    ///   over it, so an answer's pass partition depends on the batch and on
    ///   what came before (`lloom-engine`'s `CausalRequest::execute_ragged`).
    pub const FAST_ARMS: &'static [&'static str] =
        &["moe-act-f16", "decision-pack", "decision-ragged"];

    /// The arms `deterministic` takes in a speculative round's passes and
    /// in a prompt's prefill ([`Window`]), each named on the report line
    /// while its switch is on: `kernels.gemm.verify_window`,
    /// `kernels.gemm.draft_window` and `kernels.gemm.prefill_window`.
    pub const VERIFY_WINDOW_ARM: &'static str = "verify-gemm";
    pub const DRAFT_WINDOW_ARM: &'static str = "draft-gemm";
    pub const PREFILL_WINDOW_ARM: &'static str = "prefill-gemm";

    /// The arm `fast` takes in a speculative verify window while
    /// `kernels.gemm.f16_rows` is on ([`f16_rows_max`]): the window's f16
    /// rows on the GEMV's row axis, named on the report line then.
    pub const F16_ROWS_ARM: &'static str = "f16-rows";

    /// The arm `fast` takes in a batched decode step while
    /// `kernels.gemm.f16_rows_decode` is on with the row axis
    /// ([`f16_rows_decode`]): the step's rows on the GEMV's row axis.
    pub const F16_ROWS_DECODE_ARM: &'static str = "f16-rows-decode";

    /// `profile=<name> arms=<list> cpu_f16_accum=<type>/<promote_every>:<instruction>`,
    /// for the report lines that say which contract a run held and which
    /// accumulator its CPU f16 matmuls ran (`cpu::f16_matmul`).
    pub fn report(self) -> String {
        let arms = match self {
            Contract::Deterministic => {
                let on: Vec<&str> = [
                    (Window::Verify, Contract::VERIFY_WINDOW_ARM),
                    (Window::Draft, Contract::DRAFT_WINDOW_ARM),
                    (Window::Prefill, Contract::PREFILL_WINDOW_ARM),
                ]
                .into_iter()
                .filter(|(w, _)| w.arm())
                .map(|(_, name)| name)
                .collect();
                if on.is_empty() {
                    "none".to_string()
                } else {
                    on.join(",")
                }
            }
            Contract::Fast => {
                let mut on: Vec<&str> = Contract::FAST_ARMS.to_vec();
                if f16_rows_max().is_some() {
                    on.push(Contract::F16_ROWS_ARM);
                    if f16_rows_decode() {
                        on.push(Contract::F16_ROWS_DECODE_ARM);
                    }
                }
                if on.is_empty() {
                    "none".to_string()
                } else {
                    on.join(",")
                }
            }
        };
        format!(
            "profile={} arms={arms} cpu_f16_accum={}",
            self.name(),
            crate::cpu::f16_matmul::report()
        )
    }
}

/// A pass that is not on the K=1 decode path, open on this thread: a
/// speculative round's — the target's verify window over `[last, draft…]`
/// (or several rows' windows laid end to end), the drafter's block and its
/// re-key over the accepted rows — or one forward of a prompt's prefill,
/// a tile of one sequence's prompt tokens.
///
/// Under `deterministic` a half-precision matmul runs the one-row GEMV
/// reduction at every row count, so a row's bits never depend on how many
/// rows share the launch — the K=1 decode path's promise. Inside a window it
/// takes the route `fast` takes instead (the GEMM from two rows), while the
/// window's switch is on: a P-row pass as P one-row sweeps is the cost that
/// promise would charge a pass that is not on the K=1 path, and the pass's
/// rows are held identical per window shape — a verify window's, a prefill
/// tile's row count — rather than across row counts (`architecture.md`
/// §2.3.5b).
///
/// Thread-local and nestable, closed when the guard drops (an early return
/// or an unwind included): the window is the encoding thread's, and a pass
/// encoded on another thread keeps the one-row reduction rather than
/// inheriting another sequence's window.
#[must_use = "the window holds only while the guard is alive"]
pub struct VerifyWindow {
    kind: Window,
    _thread: std::marker::PhantomData<*const ()>,
}

/// Which pass a [`VerifyWindow`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// The target's verify window: `kernels.gemm.verify_window`.
    Verify,
    /// The drafter's block and re-key: `kernels.gemm.draft_window`.
    Draft,
    /// One forward of a prompt's prefill: `kernels.gemm.prefill_window`.
    Prefill,
}

const WINDOW_KINDS: usize = 3;

thread_local! {
    static WINDOWS: [std::cell::Cell<u32>; WINDOW_KINDS] = const {
        [std::cell::Cell::new(0), std::cell::Cell::new(0), std::cell::Cell::new(0)]
    };
}

/// 0 unread, 1 off, 2 on; per [`Window`].
static WINDOW_ARMS: [std::sync::atomic::AtomicU8; WINDOW_KINDS] = [
    std::sync::atomic::AtomicU8::new(0),
    std::sync::atomic::AtomicU8::new(0),
    std::sync::atomic::AtomicU8::new(0),
];

/// Launches each window's arm took off the one-row reduction since process
/// start: the arms' engagement counts.
static WINDOW_GEMMS: [std::sync::atomic::AtomicU64; WINDOW_KINDS] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

impl Window {
    /// Every kind, in the order [`window_gemm`] attributes a launch when
    /// more than one is open.
    pub const ALL: [Window; WINDOW_KINDS] = [Window::Verify, Window::Draft, Window::Prefill];

    fn index(self) -> usize {
        match self {
            Window::Verify => 0,
            Window::Draft => 1,
            Window::Prefill => 2,
        }
    }

    /// Whether a window of this kind is open on this thread.
    pub fn is_open(self) -> bool {
        WINDOWS.with(|w| w[self.index()].get() > 0)
    }

    /// This window's switch, read once; or [`Window::set_arm`].
    pub fn arm(self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let at = &WINDOW_ARMS[self.index()];
        match at.load(Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let g = &tuning().gemm;
                let on = match self {
                    Window::Verify => g.verify_window,
                    Window::Draft => g.draft_window,
                    Window::Prefill => g.prefill_window,
                };
                at.store(if on { 2 } else { 1 }, Relaxed);
                on
            }
        }
    }

    /// Select this window's arm for the process (a test runs both).
    pub fn set_arm(self, on: bool) {
        WINDOW_ARMS[self.index()]
            .store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    }

    /// The launches this window's arm has taken off the one-row reduction.
    pub fn gemms(self) -> u64 {
        WINDOW_GEMMS[self.index()].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What this window's half matmuls run under the contract in force:
    /// `split` under `fast` (the GEMV at one row, the GEMM from two), `gemm`
    /// or `one-row` under `deterministic`, by the window's arm.
    pub fn report(self) -> &'static str {
        match (contract().is_fast(), self.arm()) {
            (true, _) => "split",
            (false, true) => "gemm",
            (false, false) => "one-row",
        }
    }
}

impl VerifyWindow {
    /// The target's verify window.
    pub fn open() -> Self {
        Self::of(Window::Verify)
    }

    /// The drafter's pass over the round's block or its accepted rows.
    pub fn draft() -> Self {
        Self::of(Window::Draft)
    }

    /// One forward of a prompt's prefill: a tile of one sequence's prompt
    /// tokens, opened by the driver that feeds them. Never a pass that
    /// carries a decode row (a ragged step's pieces ride its decode rows).
    pub fn prefill() -> Self {
        Self::of(Window::Prefill)
    }

    fn of(kind: Window) -> Self {
        WINDOWS.with(|w| {
            let c = &w[kind.index()];
            c.set(c.get() + 1)
        });
        VerifyWindow {
            kind,
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for VerifyWindow {
    fn drop(&mut self) {
        WINDOWS.with(|w| {
            let c = &w[self.kind.index()];
            c.set(c.get() - 1)
        });
    }
}

/// Whether a target's verify window is open on this thread.
pub fn in_verify_window() -> bool {
    Window::Verify.is_open()
}

/// `kernels.gemm.verify_window`.
pub fn verify_window_arm() -> bool {
    Window::Verify.arm()
}

/// Select the verify window's arm for this process.
pub fn set_verify_window_arm(on: bool) {
    Window::Verify.set_arm(on)
}

/// The window whose arm a `deterministic` half matmul encoded here takes —
/// the GEMM rather than the one-row reduction — or `None`.
pub fn window_gemm() -> Option<Window> {
    Window::ALL.into_iter().find(|w| w.is_open() && w.arm())
}

/// Count one launch `w`'s arm routed to the GEMM.
pub fn note_window_gemm(w: Window) {
    WINDOW_GEMMS[w.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The verify window's engagement count now.
pub fn verify_window_gemms() -> u64 {
    Window::Verify.gemms()
}

/// The prefill window's receipt: its arm under the contract in force and the
/// launches it took since `since` (a [`Window::gemms`] read).
pub fn prefill_window_report(since: u64) -> String {
    format!(
        "arm={} gemm_launches={}",
        Window::Prefill.report(),
        Window::Prefill.gemms().saturating_sub(since)
    )
}

/// The receipt a speculative run prints: each window's arm under the
/// contract in force and its launches so far.
pub fn window_report() -> String {
    // The f16 row axis is a `fast` arm; under `deterministic` nothing takes it.
    let rows = match f16_rows_max() {
        Some(max) if contract().is_fast() => format!("rows-{max}"),
        _ => "off".to_string(),
    };
    format!(
        "arm={} gemm_launches={} draft_arm={} draft_gemm_launches={} f16_rows={rows} \
         f16_rows_launches={}",
        Window::Verify.report(),
        Window::Verify.gemms(),
        Window::Draft.report(),
        Window::Draft.gemms(),
        f16_rows_launches()
    )
}

/// `kernels.gemm.f16_rows` and its ceiling, read once; or [`set_f16_rows`].
/// 0 unread, else the ceiling when on and 1 when off.
static F16_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Launches a verify window's f16 matmuls took on the GEMV's row axis since
/// process start: the arm's engagement count.
static F16_ROWS_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The most rows the f16 row axis takes in a verify window under `fast`, or
/// `None` while `kernels.gemm.f16_rows` is off.
pub fn f16_rows_max() -> Option<usize> {
    use std::sync::atomic::Ordering::Relaxed;
    let v = match F16_ROWS.load(Relaxed) {
        0 => {
            let g = &tuning().gemm;
            let v = if g.f16_rows { g.f16_rows_max } else { 1 };
            F16_ROWS.store(v, Relaxed);
            v
        }
        v => v,
    };
    (v > 1).then_some(v)
}

/// Select the f16 row axis for this process: `Some(ceiling)` on, `None` off
/// (a test runs both).
pub fn set_f16_rows(max: Option<usize>) {
    let v = match max {
        Some(m) => {
            assert!(
                (2..=16).contains(&m),
                "f16 rows a weight read: {m} is not in 2..=16"
            );
            m
        }
        None => 1,
    };
    F16_ROWS.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Count one launch the f16 row axis took.
pub fn note_f16_rows() {
    F16_ROWS_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The f16 row axis's engagement count now.
pub fn f16_rows_launches() -> u64 {
    F16_ROWS_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

thread_local! {
    /// Open [`DecodeRows`] scopes on this thread.
    static DECODE_ROWS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A batched decode step's forward, open on this thread: one row per live
/// sequence, one token each. Under `kernels.gemm.f16_rows_decode` its half
/// matmuls of 2 to [`f16_rows_max`] rows take the GEMV's row axis.
/// Thread-local and nestable, closed when the guard drops: a pass encoded on
/// another thread keeps its own route.
#[must_use = "the scope holds only while the guard is alive"]
pub struct DecodeRows {
    _thread: std::marker::PhantomData<*const ()>,
}

impl DecodeRows {
    pub fn open() -> Self {
        DECODE_ROWS.with(|c| c.set(c.get() + 1));
        DecodeRows {
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for DecodeRows {
    fn drop(&mut self) {
        DECODE_ROWS.with(|c| c.set(c.get() - 1));
    }
}

/// Whether a batched decode step's forward is open on this thread.
pub fn in_decode_rows() -> bool {
    DECODE_ROWS.with(|c| c.get() > 0)
}

thread_local! {
    /// The open [`RaggedDecodeRows`] scope on this thread, `(live, rows,
    /// heads)`; `live == 0` when none is.
    static RAGGED_DECODE_ROWS: std::cell::Cell<(usize, usize, usize)> =
        const { std::cell::Cell::new((0, 0, 0)) };
}

/// A ragged batched step's forward, open on this thread: `live` decode rows
/// of one token each lead its `rows` input rows, the prompt pieces after
/// them, and its head projects `heads` rows (one a sequence), the live rows
/// first. Under `kernels.gemm.f16_rows_decode` the live rows of its half
/// matmuls take the route a decode step's rows take, so a sequence's bits do
/// not depend on whether a prompt piece rode its step; the pieces keep the
/// GEMM. Thread-local; the enclosing scope is restored when the guard drops.
#[must_use = "the scope holds only while the guard is alive"]
pub struct RaggedDecodeRows {
    outer: (usize, usize, usize),
    _thread: std::marker::PhantomData<*const ()>,
}

impl RaggedDecodeRows {
    pub fn open(live: usize, rows: usize, heads: usize) -> Self {
        let outer = RAGGED_DECODE_ROWS.with(|c| c.replace((live, rows, heads)));
        RaggedDecodeRows {
            outer,
            _thread: std::marker::PhantomData,
        }
    }
}

impl Drop for RaggedDecodeRows {
    fn drop(&mut self) {
        RAGGED_DECODE_ROWS.with(|c| c.set(self.outer));
    }
}

/// The decode rows leading a matmul of `m` rows in the open
/// [`RaggedDecodeRows`] scope: `Some(live)` when `m` is the step's input
/// rows or its head's and the live rows are some but not all of them.
pub fn ragged_decode_lead(m: usize) -> Option<usize> {
    let (live, rows, heads) = RAGGED_DECODE_ROWS.with(|c| c.get());
    (live > 0 && live < m && (m == rows || m == heads)).then_some(live)
}

/// `kernels.gemm.f16_rows_decode`: 0 unread, 1 off, 2 on.
static F16_ROWS_DECODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Launches a batched decode step's f16 matmuls took on the row axis since
/// process start: the decode arm's engagement count.
static F16_ROWS_DECODE_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `kernels.gemm.f16_rows_decode`, read once; or [`set_f16_rows_decode`].
/// The decode rows take the axis only while it is on ([`f16_rows_max`]).
pub fn f16_rows_decode() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match F16_ROWS_DECODE.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = tuning().gemm.f16_rows_decode;
            F16_ROWS_DECODE.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// Select the decode rows' arm for this process (a test runs both).
pub fn set_f16_rows_decode(on: bool) {
    F16_ROWS_DECODE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// Count one launch a batched decode step's rows took on the row axis (a
/// ragged step's live rows included).
pub fn note_f16_rows_decode() {
    F16_ROWS_DECODE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The decode rows' engagement count now.
pub fn f16_rows_decode_launches() -> u64 {
    F16_ROWS_DECODE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// `kernels.gemm.f16_rows_round`: 0 unread, 1 off, 2 on.
static F16_ROWS_ROUND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Launches on the f16 row axis that took a capacity above their rows since
/// process start: the rounding arm's engagement count.
static F16_ROWS_ROUNDED_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `kernels.gemm.f16_rows_round`, read once; or [`set_f16_rows_round`].
pub fn f16_rows_round() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match F16_ROWS_ROUND.load(Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = tuning().gemm.f16_rows_round;
            F16_ROWS_ROUND.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// Select the capacity rounding for this process (a test runs both).
pub fn set_f16_rows_round(on: bool) {
    F16_ROWS_ROUND.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// Count one launch on the f16 row axis at a capacity above its rows.
pub fn note_f16_rows_rounded() {
    F16_ROWS_ROUNDED_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The rounding arm's engagement count now.
pub fn f16_rows_rounded_launches() -> u64 {
    F16_ROWS_ROUNDED_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The contract in force for this process: [`tuning`]'s `contract.profile`,
/// parsed once. Every fast arm reads it here, so the contract is one value
/// read once (a compile-time profile type would replace this function).
pub fn contract() -> Contract {
    *CONTRACT.get_or_init(|| {
        REGIME.get().copied().unwrap_or_else(|| {
            Contract::parse(&tuning().contract.profile).expect("the parser accepted it")
        })
    })
}

static CONTRACT: OnceLock<Contract> = OnceLock::new();
/// The contract a family's regime type stated ([`pin_regime`]) before the
/// contract was first read: what [`contract`] takes ahead of the tuning's
/// default, so a model built at `Exact` in a process that opened its device
/// first (the tuning read, the pin refused) still states its regime.
static REGIME: OnceLock<Contract> = OnceLock::new();

/// A family built at regime `c` states it (`lloom::profile::hold`): the
/// contract becomes `c` unless one is already in force — `LLOOM_PROFILE`
/// named one, a harness pinned one, or [`contract`] was read — in which
/// case that one is returned and the caller compares. Unlike
/// [`pin_contract`] this does not need the tuning unread: the Metal device
/// reads the tuning when it opens, which is before any model is built.
pub fn pin_regime(c: Contract) -> Result<Contract, String> {
    if let Some(v) = std::env::var_os("LLOOM_PROFILE") {
        return Contract::parse(&v.to_string_lossy());
    }
    if let Some(read) = CONTRACT.get() {
        return Ok(*read);
    }
    let _ = REGIME.set(c);
    CONTRACT_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(contract())
}

static CONTRACT_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Pin the contract for the process, before anything reads the tuning: the
/// defaults with the env aliases applied, then `c` — unless `LLOOM_PROFILE`
/// names one, which wins (a quality run asking a gate harness for `fast` has
/// named it). `Err` if the group was already fixed. What a gate harness or a
/// bit-identity test calls first, so its contract is stated, not inherited.
pub fn pin_contract(c: Contract) -> Result<Contract, String> {
    let mut t = from_process_env().map_err(|e| format!("tuning: {e}"))?;
    if std::env::var_os("LLOOM_PROFILE").is_none() {
        t.contract.profile = c.name().to_string();
    }
    let pinned = Contract::parse(&t.contract.profile)?;
    install(t).map_err(|e| e.to_string())?;
    CONTRACT_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(pinned)
}

/// The contract a gate runs, refused unless it was stated: pinned
/// ([`pin_contract`]) or named by `LLOOM_PROFILE`. A gate that inherited the
/// default would run `fast` silently, which is the failure the pin exists
/// for.
pub fn gate_contract(who: &str) -> Result<Contract, String> {
    let named = std::env::var_os("LLOOM_PROFILE").is_some();
    if !(CONTRACT_PINNED.load(std::sync::atomic::Ordering::Relaxed) || named) {
        return Err(format!(
            "{who}: a gate runs on a stated contract; it inherited the default ({}) — \
             pin it (`tuning::pin_contract`) or set LLOOM_PROFILE (architecture.md §2.3.5b)",
            ContractTuning::default().profile
        ));
    }
    Ok(contract())
}

/// The legacy env names of this group whose values meant something else
/// before they joined the option tree (`options::legacy_notes`).
pub const LEGACY_BEFORE: &[BeforeRow] = &[
    (
        "CANDLE_METAL_ENABLE_FAST_MATH",
        "metal.fast_math",
        before::truthy,
    ),
    (
        "CANDLE_METAL_EAGER_RESIDENCY_RELEASE",
        "metal.eager_residency_release",
        |v| Some((v == "1" || v.eq_ignore_ascii_case("true")).to_string()),
    ),
    (
        "CANDLE_METAL_PROFILE_DUMP",
        "instrument.profile_dump",
        before::path_verbatim,
    ),
    ("LLOOM_383_TILE", "gemm.tile", before::empty_or_off_unset),
    (
        "LLOOM_386_FORCE_GEMM",
        "gemm.force",
        before::empty_or_off_unset,
    ),
    (
        "CANDLE_METAL_HAZARD_KEY",
        "metal.hazard_key",
        before::empty_or_off_unset,
    ),
    (
        "LLOOM_47_REUSE_OVERSIZE_BOUND",
        "pool.reuse_oversize_bound",
        before::empty_or_off_unset,
    ),
    (
        "LLOOM_METAL_ALLOW_EMULATED",
        "metal.allow_emulated",
        before::empty_or_off_unset,
    ),
    ("LLOOM_492_GEMV_BM", "gemm.gemv_bm", before::empty_unset),
    ("LLOOM_QMV_Q4K_NR0", "qmv.q4k_nr0", before::empty_unset),
    ("LLOOM_QMV_Q5K_NR0", "qmv.q5k_nr0", before::empty_unset),
    ("LLOOM_QMV_MM_FROM", "qmv.mm_from", before::empty_unset),
    // The empty value was the arm that advises nothing, as `none` is.
    ("CANDLE_MMAP_ADVICE", "cpu.mmap_advice", |v| {
        v.is_empty().then(|| "\"none\"".into())
    }),
];

/// [`LEGACY_BEFORE`]'s lines for the values `env` sets.
pub fn legacy_notes(env: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
    crate::options::legacy_notes(LEGACY_BEFORE, Tuning::META, env, &|alone| {
        let mut t = Tuning::default();
        t.from_env_with(alone)?;
        serde_json::to_value(&t).map_err(|e| e.to_string())
    })
}

/// The defaults with the process env applied, the moved meanings noted.
fn from_process_env() -> Result<Tuning, String> {
    for note in
        legacy_notes(&|name: &str| std::env::var_os(name).map(|v| v.to_string_lossy().into_owned()))
    {
        eprintln!("{note}");
    }
    let mut t = Tuning::default();
    t.from_env()?;
    Ok(t)
}

/// The group in force for this process. First use resolves it: the
/// installed group if [`install`] ran, else the defaults with the env
/// aliases applied (a malformed alias panics, naming the variable).
pub fn tuning() -> &'static Tuning {
    TUNING.get_or_init(|| from_process_env().unwrap_or_else(|e| panic!("tuning: {e}")))
}

/// Resolve and validate library setup before creating backend resources.
/// Failures do not install a group or claim test isolation. An earlier legacy
/// `tuning()` read is also checked, never silently replaced.
pub fn try_tuning() -> Result<&'static Tuning, TuningSetupError> {
    if let Some(t) = TUNING.get() {
        t.validate_instruments()?;
        t.note_partial_instruments();
        return Ok(t);
    }
    let t = from_process_env().map_err(TuningSetupError::InvalidEnvironment)?;
    t.validate_instruments()?;
    let installed = TUNING.get_or_init(|| t);
    installed.validate_instruments()?;
    installed.note_partial_instruments();
    Ok(installed)
}

/// Install before first use. Refusal returns the original configuration and
/// a typed reason; an unsupported request has no global/isolation side effects.
pub fn install(t: Tuning) -> Result<(), InstallError> {
    let reason = if TUNING.get().is_some() {
        Some(TuningSetupError::AlreadyRead)
    } else {
        t.validate_instruments().err()
    };
    if let Some(reason) = reason {
        return Err(InstallError {
            reason,
            tuning: Box::new(t),
        });
    }
    crate::isolation::claim("lloom_tensor::tuning");
    TUNING.set(t).map_err(|t| InstallError {
        reason: TuningSetupError::AlreadyRead,
        tuning: Box::new(t),
    })?;
    TUNING
        .get()
        .expect("successful tuning installation")
        .note_partial_instruments();
    Ok(())
}

/// Whether [`install`] (or a first read) has already fixed the group.
pub fn installed() -> bool {
    TUNING.get().is_some()
}

fn ranged(value: &str, what: &str, lo: usize, hi: usize) -> Result<usize, String> {
    let n: usize = parse_count(value).map_err(|e| format!("{what}: {e}"))?;
    if !(lo..=hi).contains(&n) {
        return Err(format!("{what}: {n} is outside {lo}..={hi}"));
    }
    Ok(n)
}

fn parse_nsg(v: &str) -> Result<usize, String> {
    ranged(v, "simdgroups a threadgroup", 1, 16)
}
fn parse_nr0(v: &str) -> Result<usize, String> {
    ranged(v, "rows a simdgroup", 1, 4)
}
/// `0` is the rule's own choice, as `none` is.
fn parse_opt_nr0_4(v: &str) -> Result<Option<usize>, String> {
    if is_none(v) || v.trim() == "0" {
        return Ok(None);
    }
    ranged(v, "rows a simdgroup", 1, 4).map(Some)
}
fn parse_opt_rows_max(v: &str) -> Result<Option<usize>, String> {
    let v = non_empty(v)?;
    if is_none(v) || v == "0" {
        return Ok(None);
    }
    ranged(v, "rows a weight read", 1, 8).map(Some)
}
fn parse_gdn_split(v: &str) -> Result<usize, String> {
    ranged(v, "gdn split", 1, 32)
}
fn parse_positive(v: &str) -> Result<usize, String> {
    ranged(v, "count", 1, usize::MAX)
}
fn parse_heap_mb(v: &str) -> Result<usize, String> {
    ranged(v, "heap chunk MiB", 16, usize::MAX)
}
fn parse_pow2_align(v: &str) -> Result<usize, String> {
    let n = ranged(v, "buffer alignment", 16, usize::MAX)?;
    if !n.is_power_of_two() {
        return Err(format!("buffer alignment {n} is not a power of two"));
    }
    Ok(n)
}
fn one_of(v: &str, what: &str, arms: &[&str]) -> Result<String, String> {
    parse_arm(v, arms)
        .map(str::to_string)
        .map_err(|e| format!("{what}: {e}"))
}
/// An optional enum: `none` unsets, anything else is `f`'s.
fn optional(v: &str, f: impl Fn(&str) -> Result<String, String>) -> Result<Option<String>, String> {
    if is_none(v) {
        Ok(None)
    } else {
        f(v).map(Some)
    }
}
/// An optional comma list (a text): `none` or the empty list unsets.
fn optional_list(
    v: &str,
    f: impl Fn(&str) -> Result<String, String>,
) -> Result<Option<String>, String> {
    if v.trim().is_empty() || is_none(v) {
        Ok(None)
    } else {
        f(v.trim()).map(Some)
    }
}
fn parse_tile(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| {
        one_of(
            s,
            "gemm tile",
            &["32x32", "32x16", "16x32", "16x16", "64x32", "64x64"],
        )
    })
}
fn parse_f16_rows_max(v: &str) -> Result<usize, String> {
    ranged(v, "f16 rows a weight read", 2, 16)
}
fn parse_gemv_bm(v: &str) -> Result<Option<usize>, String> {
    if is_none(v) {
        return Ok(None);
    }
    match parse_count::<usize>(v).map_err(|e| format!("gemv bm: {e}"))? {
        n @ (4 | 8) => Ok(Some(n)),
        other => Err(format!(
            "gemv bm: `{other}` is not 4 or 8 (the two instantiated decode tiles)"
        )),
    }
}
fn parse_force_gemm(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| one_of(s, "force gemm", &["m", "n", "both"]))
}
/// Without the Metal backend its parsers are not compiled; a value of one of
/// its keys is still held to the arms the field declares (`variants`:
/// `name`, `name:<param>`, `name[:<param>]`, the parameter a count), so a
/// misspelling is refused on every build. The domain checks are the backend's.
#[cfg(not(feature = "metal"))]
fn declared_arm(v: &str, field: &str) -> Result<String, String> {
    let variants = crate::options::meta_at(MetalTuning::META, field)
        .map(|m| m.variants)
        .unwrap_or(&[]);
    let v = non_empty(v)?.to_ascii_lowercase();
    let (name, param) = match v.split_once(':') {
        Some((n, p)) => (n, Some(p)),
        None => (v.as_str(), None),
    };
    let ok = variants.iter().any(|d| {
        let base = d.split([':', '[']).next().unwrap_or(d);
        base == name
            && match param {
                None => !d.contains(':') || d.contains("[:"),
                Some(p) => d.contains(':') && parse_count::<u64>(p).is_ok(),
            }
    });
    if ok {
        Ok(v)
    } else {
        Err(format!("`{v}` is not one of {}", variants.join(", ")))
    }
}
fn parse_cut_policy(v: &str) -> Result<String, String> {
    let v = non_empty(v)?.to_ascii_lowercase();
    crate::command_policy::CutPolicy::from_env_str(&v)?;
    Ok(v)
}
fn parse_command_api(v: &str) -> Result<String, String> {
    parse_arm(v, &["metal3", "metal4", "auto"]).map(str::to_owned)
}
fn parse_host_input(v: &str) -> Result<String, String> {
    parse_arm(v, &["commit", "gpu-wait"]).map(str::to_owned)
}

fn parse_completion_wait(v: &str) -> Result<String, String> {
    let v = non_empty(v)?.to_ascii_lowercase();
    #[cfg(feature = "metal")]
    crate::kernels::metal::completion::CompletionWait::from_str_arm(&v)?;
    #[cfg(not(feature = "metal"))]
    declared_arm(&v, "completion_wait")?;
    Ok(v)
}
/// `backend.metal.completion_wait` in the key's spelling; a bare
/// `shared-event-wait` is the timed wait at its default timeout.
fn completion_wait_to(s: &str) -> String {
    #[cfg(feature = "metal")]
    let bare_ms = crate::kernels::metal::completion::DEFAULT_WAIT_TIMEOUT_MS;
    // Inert without Metal; the figure is the Metal default's.
    #[cfg(not(feature = "metal"))]
    let bare_ms = 2_000;
    crate::options::v1::completion_wait_to(s, bare_ms)
}
/// `nodes_per_cb = k` is `cut_policy = nodes:<k>`.
fn nodes_to_cut_policy(s: &str) -> String {
    format!("nodes:{s}")
}
#[cfg(feature = "metal")]
fn parse_metal_disable(v: &str) -> Result<Option<String>, String> {
    optional_list(v, |s| {
        crate::kernels::metal::arms::Features::parse_list(s).map_err(|e| e.to_string())?;
        Ok(s.to_string())
    })
}
#[cfg(not(feature = "metal"))]
fn parse_metal_disable(v: &str) -> Result<Option<String>, String> {
    optional_list(v, |s| Ok(s.to_string()))
}
fn parse_cpu_disable(v: &str) -> Result<Option<String>, String> {
    optional_list(v, |s| {
        crate::cpu::arms::Features::parse_list(s).map_err(|e| e.to_string())?;
        Ok(s.to_string())
    })
}
fn parse_accum_type(v: &str) -> Result<String, String> {
    one_of(v, "f16 matmul accumulator type", &["auto", "f32", "f16"])
}
fn parse_promote_every(v: &str) -> Result<String, String> {
    one_of(v, "f16 matmul promotion interval", &["auto", "1", "16"])
}
fn parse_allow_emulated(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| {
        let s = non_empty(s)?.to_ascii_lowercase();
        #[cfg(feature = "metal")]
        crate::kernels::metal::arms::parse_allow_emulated(&s)?;
        #[cfg(not(feature = "metal"))]
        for item in s.split(',') {
            declared_arm(item.trim(), "allow_emulated")?;
        }
        Ok(s)
    })
}
fn parse_eviction(v: &str) -> Result<String, String> {
    one_of(
        v,
        "eviction order",
        &["age", "oldest", "off", "size", "largest"],
    )
}
fn parse_hazard_key(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| one_of(s, "hazard key", &["range", "pointer", "ptr"]))
}
/// `none` is the arm that advises nothing, kept as `"none"` rather than
/// cleared: unset takes the default, an advised arm.
fn parse_mmap_advice(v: &str) -> Result<Option<String>, String> {
    if v.trim().is_empty() {
        return Err(
            "mmap advice: the empty value is refused; `none` is the arm that advises nothing"
                .into(),
        );
    }
    one_of(
        v,
        "mmap advice",
        &["none", "willneed", "will_need", "sequential", "seq"],
    )
    .map(Some)
}
fn parse_profile(v: &str) -> Result<String, String> {
    non_empty(v)?;
    Contract::parse(v).map(|c| c.name().to_string())
}
fn parse_dequant_gemm(v: &str) -> Result<String, String> {
    one_of(v, "dequant gemm", &["qmm", "steel"])
}
fn parse_pair_order(v: &str) -> Result<String, String> {
    non_empty(v)?;
    crate::quantized::PairOrder::parse(v).map(|o| o.name().to_string())
}
fn parse_icb_residency(v: &str) -> Result<String, String> {
    one_of(v, "icb residency", &["per-run", "plan-wide"])
}
fn parse_residency_attach(v: &str) -> Result<String, String> {
    one_of(v, "residency attach", &["queue", "heartbeat"])
}
fn parse_residency_cap_pct(v: &str) -> Result<u32, String> {
    let pct: u32 = parse_count(v)?;
    if pct > 100 {
        return Err(format!("residency set cap {pct} % is over 100"));
    }
    Ok(pct)
}
fn parse_residency_hot(v: &str) -> Result<String, String> {
    non_empty(v)?;
    let names = ["weights", "kv", "heap", "scratch", "experts", "other"];
    let mut out = Vec::new();
    for name in v.split(',').map(str::trim) {
        if !names.contains(&name) {
            return Err(format!(
                "`{name}` is not a residency class ({})",
                names.join(", ")
            ));
        }
        out.push(name);
    }
    Ok(out.join(","))
}
fn parse_oversize_bound(v: &str) -> Result<Option<f64>, String> {
    if is_none(v) {
        return Ok(None);
    }
    let f = parse_number(v).map_err(|e| format!("reuse oversize bound: {e}"))?;
    if !matches!(
        f.partial_cmp(&1.0),
        Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
    ) {
        return Err(format!("reuse oversize bound {f} must be >= 1.0"));
    }
    Ok(Some(f))
}
fn parse_cflags(v: &str) -> Result<Option<String>, String> {
    if v.trim().is_empty() {
        // An empty value is almost always an unexpanded shell variable
        // (`LLOOM_434_METAL_CFLAGS=$UNSET`); refused, as the site did.
        return Err("metal cflags: set but empty — unset it for the default arm".into());
    }
    if is_none(v) {
        return Ok(None);
    }
    Ok(Some(v.to_string()))
}

/// The determinism contract ([`Contract`]).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct ContractTuning {
    /// The determinism contract: `fast` (the default: the same distribution; licenses the arms whose bits depend on scheduling or batch composition) or `deterministic` (same inputs + seed → same bits; every gate pins it).
    #[opt(flag = "profile", env = "LLOOM_PROFILE", parser = "parse_profile", variants = ["deterministic", "fast"], path = "determinism", status = "stable", cli = "determinism")]
    pub profile: String,
}

impl Default for ContractTuning {
    fn default() -> Self {
        ContractTuning {
            profile: "fast".into(),
        }
    }
}

/// The dense GEMM/GEMV tile selection (`mlx_gemm.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct GemmTuning {
    /// The `m` (rows) up to which the skinny 16-row GEMM tile is taken; above it the 64-row tile (§13.5o/p).
    #[opt(
        flag = "gemm-skinny-max",
        env = "LLOOM_364_SKINNY_MAX",
        path = "kernels.gemm.skinny_max_rows",
        status = "advanced"
    )]
    pub skinny_max: usize,
    /// Pin the GEMM tile (an A/B arm): `32x32 | 32x16 | 16x32 | 16x16 | 64x32 | 64x64`; none = the derived tile.
    #[opt(flag = "gemm-tile", env = "LLOOM_383_TILE", parser = "parse_tile", variants = ["none", "32x32", "32x16", "16x32", "16x16", "64x32", "64x64"], path = "kernels.gemm.tile", status = "arm")]
    pub tile: Option<String>,
    /// Pin the decode GEMV's `bm` (an A/B arm): 4 or 8; none = the derived tile.
    #[opt(flag = "gemv-bm", env = "LLOOM_492_GEMV_BM", parser = "parse_gemv_bm", variants = ["none", "4", "8"], path = "kernels.gemm.gemv_bm", status = "arm")]
    pub gemv_bm: Option<usize>,
    /// Force the GEMM route where the GEMV would be taken (an A/B arm): `m | n | both`; none = the rule.
    #[opt(flag = "force-gemm", env = "LLOOM_386_FORCE_GEMM", parser = "parse_force_gemm", variants = ["none", "m", "n", "both"], path = "kernels.gemm.force_route", status = "arm")]
    pub force: Option<String>,
    /// Under `deterministic`, a speculative verify window's half matmuls take the GEMM `fast` takes from two rows, its rows identical per window shape; off = the one-row GEMV reduction at every row count, the window's too (k + 1 weight sweeps a k-row window). The K=1 path keeps the reduction either way; `fast` does not read it.
    #[opt(
        flag = "gemm-verify-window",
        env = "LLOOM_GEMM_VERIFY_WINDOW",
        bool,
        path = "kernels.gemm.verify_window",
        status = "arm"
    )]
    pub verify_window: bool,
    /// Under `deterministic`, a speculative drafter's half matmuls over its block and its re-key take the GEMM `fast` takes from two rows; off = the one-row GEMV reduction, the drafter's block as one weight sweep a row. The drafter is not on the K=1 path; `fast` does not read it.
    #[opt(
        flag = "gemm-draft-window",
        env = "LLOOM_GEMM_DRAFT_WINDOW",
        bool,
        path = "kernels.gemm.draft_window",
        status = "arm"
    )]
    pub draft_window: bool,
    /// Under `deterministic`, a prompt's prefill forward — a tile of one sequence's prompt tokens — takes the GEMM `fast` takes from two rows, its rows identical per tile shape; off = the one-row GEMV reduction, a P-row tile as P weight sweeps. The K=1 decode path (a step, batched rows, a ragged step's pieces) keeps the reduction either way; `fast` does not read it.
    #[opt(
        flag = "gemm-prefill-window",
        env = "LLOOM_GEMM_PREFILL_WINDOW",
        bool,
        path = "kernels.gemm.prefill_window",
        status = "arm"
    )]
    pub prefill_window: bool,
    /// Under `fast`, a speculative verify window's f16 matmuls of 2 to `f16_rows_max` rows take the GEMV's row axis: each weight block read once for every row of the window, each row the one-row GEMV's bits; off = the GEMM from two rows. On by default under `fast` (maintainer, 2026-10-19: DSpark ×1.25 on LFM2.5, the stream parting from K=1 only at an exact tie); `deterministic` never takes it. The K=1 path, the prefill, the drafter and `deterministic` do not read it.
    #[opt(
        flag = "gemm-f16-rows",
        env = "LLOOM_GEMM_F16_ROWS",
        bool,
        path = "kernels.gemm.f16_rows",
        status = "arm"
    )]
    pub f16_rows: bool,
    /// The most rows the f16 row axis takes (2..16), in a verify window and in a batched decode step (`kernels.gemm.f16_rows_decode`, whose serve and bench default holds at a batch of 2 to this many sequences); more rows take the GEMM. 10 in the library; `serve` and `bench` resolve 12 on Metal (decision 60: on LFM2.5-2.6B the served step at 11 and 12 rows is 21.1 and 22.8 ms on the axis against 26.8 and 27.1 on the GEMM, ×1.26 and ×1.18 decode tok/s at B = 11 and 12; from 13 rows the axis is ×1.65 the GEMM's step).
    #[opt(
        flag = "gemm-f16-rows-max",
        env = "LLOOM_GEMM_F16_ROWS_MAX",
        parser = "parse_f16_rows_max",
        path = "kernels.gemm.f16_rows_max",
        status = "arm"
    )]
    pub f16_rows_max: usize,
    /// Under `fast`, with `kernels.gemm.f16_rows` on, a batched decode step's f16 matmuls of 2 to `f16_rows_max` rows (one row per live sequence) take the GEMV's row axis: each weight block read once for every row, each row the one-row GEMV's bits; off = the GEMM from two rows. A step that carries prompt pieces puts its 1 to `f16_rows_max` live rows on the same route and its pieces on the GEMM, so a row's bits do not depend on whether a piece rode its step; a single sequence's step and `deterministic` keep their routes. Off in the library; `serve` and `bench` turn it on on Metal at a batch of 2 to `f16_rows_max` sequences (decision 54; 2 to 12 since decision 60), `--gemm-f16-rows-decode=false` to measure without it.
    #[opt(
        flag = "gemm-f16-rows-decode",
        env = "LLOOM_GEMM_F16_ROWS_DECODE",
        bool,
        path = "kernels.gemm.f16_rows_decode",
        status = "arm"
    )]
    pub f16_rows_decode: bool,
    /// Under `fast`, a launch of an odd number of rows (3 to 15) on the f16 row axis takes the next even capacity: the padded lanes re-read the launch's last row and store nothing, so every row keeps the one-row GEMV's bits; off = capacity `m`. A verify window's and a batched decode step's launches alike; one row keeps the one-row GEMV.
    #[opt(
        flag = "gemm-f16-rows-round",
        env = "LLOOM_GEMM_F16_ROWS_ROUND",
        bool,
        path = "kernels.gemm.f16_rows_round",
        status = "arm"
    )]
    pub f16_rows_round: bool,
}

impl Default for GemmTuning {
    fn default() -> Self {
        GemmTuning {
            // `mlx_gemm::SKINNY_TILE_MAX_M + 1` (33): the 16-row tile up to 32 rows.
            skinny_max: 33,
            tile: None,
            gemv_bm: None,
            force: None,
            verify_window: true,
            draft_window: true,
            prefill_window: true,
            f16_rows: true,
            // Measured (`docs/tuning.md` §13.5y): at LFM2.5's shapes the row
            // axis is no slower than the GEMM at every row count to 10 and
            // slower at 12 on most of them.
            f16_rows_max: 10,
            f16_rows_decode: false,
            f16_rows_round: false,
        }
    }
}

/// The quantized matvec cores (`quantized.rs`, `quantized/metal.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct QmvTuning {
    /// Simdgroups a threadgroup on the IQ2_XS routed core (a function constant; 1..16).
    #[opt(
        flag = "qmv-iq2xs-nsg",
        env = "LLOOM_QMV_IQ2XS_NSG",
        parser = "parse_nsg",
        path = "kernels.quant_matvec.iq2_xs.simdgroups",
        status = "advanced"
    )]
    pub iq2xs_nsg: usize,
    /// Simdgroups a threadgroup on the IQ4_NL core (a function constant; 1..16).
    #[opt(
        flag = "qmv-iq4nl-nsg",
        env = "LLOOM_QMV_IQ4NL_NSG",
        parser = "parse_nsg",
        path = "kernels.quant_matvec.iq4_nl.simdgroups",
        status = "advanced"
    )]
    pub iq4nl_nsg: usize,
    /// Simdgroups a threadgroup on the IQ2_S core, its 8 KB grid staged per threadgroup (P31 (4); 1..16).
    #[opt(
        flag = "qmv-iq2s-nsg",
        env = "LLOOM_QMV_IQ2S_NSG",
        parser = "parse_nsg",
        path = "kernels.quant_matvec.iq2_s.simdgroups",
        status = "advanced"
    )]
    pub iq2s_nsg: usize,
    /// Rows a simdgroup on the Q4_K core: set, two simdgroups of that many rows (ggml master's 2 × 2); none = the incumbent one simdgroup of four (P31 (5)).
    #[opt(flag = "qmv-q4k-nr0", env = "LLOOM_QMV_Q4K_NR0", parser = "parse_opt_nr0_4", variants = ["none", "1", "2", "3", "4"], path = "kernels.quant_matvec.q4_k.rows_per_simdgroup", status = "arm")]
    pub q4k_nr0: Option<usize>,
    /// Rows a simdgroup on the Q5_K core: 1 (ggml master's) or 2; none = the incumbent 2 (P31 (5)).
    #[opt(flag = "qmv-q5k-nr0", env = "LLOOM_QMV_Q5K_NR0", parser = "parse_opt_nr0_4", variants = ["none", "1", "2", "4"], path = "kernels.quant_matvec.q5_k.rows_per_simdgroup", status = "arm")]
    pub q5k_nr0: Option<usize>,
    /// Rows a simdgroup on the Q6_K core (a function constant; 1..4; 2 = ggml master's).
    #[opt(
        flag = "qmv-q6k-nr0",
        env = "LLOOM_QMV_Q6K_NR0",
        parser = "parse_nr0",
        path = "kernels.quant_matvec.q6_k.rows_per_simdgroup",
        status = "advanced"
    )]
    pub q6k_nr0: usize,
    /// The row axis of the quantized matmul-vec under `fast`: 2 to 8 activation rows (a verify window, a small batch, the routed planes sharing a slot) read each weight block once; off = one weight read per row. `deterministic` never takes it (a row's bits then never depend on the row count).
    #[opt(
        flag = "qmv-rows",
        env = "LLOOM_QMV_ROWS",
        bool,
        path = "kernels.quant_matvec.batched_rows",
        status = "arm"
    )]
    pub rows: bool,
    /// The row axis's ceiling for every kind (1..8): none = each kind's measured ceiling (`QuantKind::rows_max`). An A/B arm, and how a test reaches every capacity of every kind.
    #[opt(
        flag = "qmv-rows-max",
        env = "LLOOM_QMV_ROWS_MAX",
        parser = "parse_opt_rows_max",
        variants = ["none", "1", "2", "4", "8"],
        path = "kernels.quant_matvec.rows_max",
        status = "arm"
    )]
    pub rows_max: Option<usize>,
    /// Rows from which a quantized stem takes the mm kernel instead of the row gemvs (none = 9: up to 8 rows on the gemv).
    #[opt(
        flag = "qmv-mm-from",
        env = "LLOOM_QMV_MM_FROM",
        path = "kernels.quant_matvec.matmul_from_rows",
        status = "advanced"
    )]
    pub mm_from: Option<usize>,
    /// The routed matvec of 2 to 8 rows (a verify window, a few rows) in its union form: the first plane naming a slot runs every plane naming it, so a slot the rows share is read once (each plane the incumbent's arithmetic, under either contract). Now the routed row axis at capacity 1 (`kernel_mul_mv_id{,2}_*_r1`); where `fast` and a ceiling put the launch on the axis, the axis wins.
    #[opt(
        flag = "qmv-id-union",
        env = "LLOOM_QMV_ID_UNION",
        bool,
        path = "kernels.quant_matvec.routed_union",
        status = "arm"
    )]
    pub id_union: bool,
}

impl Default for QmvTuning {
    fn default() -> Self {
        QmvTuning {
            iq2xs_nsg: 4,
            iq4nl_nsg: 4,
            iq2s_nsg: 4,
            q4k_nr0: None,
            q5k_nr0: None,
            q6k_nr0: 2,
            rows: true,
            rows_max: None,
            mm_from: None,
            id_union: false,
        }
    }
}

/// The quantized GEMM at prefill (`quantized/metal.rs`, `quantized/mod.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct QmmTuning {
    /// Rows from which a quantized stem's GEMM dequantises the weight to f16 once and runs the f16 arm (0 = never).
    #[opt(
        flag = "qmm-dequant-from",
        env = "LLOOM_QMM_DEQUANT_FROM",
        path = "kernels.quant_matmul.dequantize_from_rows",
        status = "advanced"
    )]
    pub dequant_from: usize,
    /// The GEMM over the dequantised scratch: `qmm` (the quantized GEMM's f16 arm) or `steel`.
    #[opt(flag = "qmm-dequant-gemm", env = "LLOOM_QMM_DEQUANT_GEMM", parser = "parse_dequant_gemm", variants = ["qmm", "steel"], path = "kernels.quant_matmul.dequantized_gemm", status = "arm")]
    pub dequant_gemm: String,
    /// `QMatMul::TensorF16` takes the f16-weight GEMM past the mixed gemv's rows (off = the classical cast-matmul-cast route).
    #[opt(
        flag = "qmm-f16-mm",
        env = "LLOOM_QMM_F16_MM",
        bool,
        path = "kernels.quant_matmul.f16_weight_gemm",
        status = "advanced"
    )]
    pub f16_mm: bool,
    /// Dequantise every quantized matmul to f32 (a reference arm).
    #[opt(
        flag = "dequantize-all",
        env = "CANDLE_DEQUANTIZE_ALL",
        bool,
        path = "kernels.quant_matmul.dequantize_all",
        status = "reference"
    )]
    pub dequantize_all: bool,
    /// Dequantise every quantized matmul to f16 (a reference arm).
    #[opt(
        flag = "dequantize-all-f16",
        env = "CANDLE_DEQUANTIZE_ALL_F16",
        bool,
        path = "kernels.quant_matmul.dequantize_all_f16",
        status = "reference"
    )]
    pub dequantize_all_f16: bool,
}

impl Default for QmmTuning {
    fn default() -> Self {
        QmmTuning {
            dequant_from: 64,
            // `quantized::metal::DEFAULT_DEQUANT_GEMM_STEEL` is false.
            dequant_gemm: "qmm".into(),
            f16_mm: true,
            dequantize_all: false,
            dequantize_all_f16: false,
        }
    }
}

/// The routed-expert GEMM (`quantized/metal.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct MoeTuning {
    /// Pairs (rows × experts) from which the routed matmul takes the grouped GEMM instead of the matvec planes (0 = never).
    #[opt(
        flag = "moe-mm-from",
        env = "LLOOM_MOE_MM_FROM",
        path = "kernels.moe.gemm_from_pairs",
        status = "advanced"
    )]
    pub mm_from: usize,
    /// Pin the routed GEMM's row block: 16 or 32 (0 = the rule: 16 when the experts see few rows each).
    #[opt(flag = "moe-mm-bn", env = "LLOOM_MOE_MM_BN", variants = ["0", "16", "32"], path = "kernels.moe.gemm_row_block", status = "arm")]
    pub mm_bn: usize,
    /// Gate and up of a routed layer in one launch on the matmul-vec arm (`kernel_mul_mv_id2`; a GEMM tile always takes two); off = two launches, the A/B arm.
    #[opt(
        flag = "moe-pair",
        env = "LLOOM_MOE_PAIR",
        bool,
        path = "kernels.moe.gate_up_pair",
        status = "advanced"
    )]
    pub pair: bool,
    /// Where the pair's half sits in its launch's grid: `expert` (an expert's gate and up adjacent, the default), `half` (the slowest axis, read by a division: the previous default), `half-cmp` (the same, read by a compare), `tile` (a row tile's gate and up adjacent), or `split` (the pair's kernel once a half, a discriminator). Same bits on every arm.
    #[opt(flag = "moe-pair-order", env = "LLOOM_MOE_PAIR_ORDER", parser = "parse_pair_order", variants = ["expert", "half", "half-cmp", "tile", "split"], path = "kernels.moe.pair_order", status = "arm")]
    pub pair_order: String,
}

impl Default for MoeTuning {
    fn default() -> Self {
        MoeTuning {
            mm_from: 64,
            mm_bn: 0,
            pair: true,
            pair_order: "expert".into(),
        }
    }
}

/// The Gated DeltaNet step kernels (`kernels/gdn.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct GdnTuning {
    /// The most `dk` slices a head's threads take in `gdn_step` (1..32; more slices, longer cross-slice reduction).
    #[opt(
        flag = "gdn-split",
        env = "LLOOM_GDN_SPLIT",
        parser = "parse_gdn_split",
        path = "kernels.deltanet.split",
        status = "advanced"
    )]
    pub split: usize,
    /// The step's register form (the default): each thread's slice of `S` held in registers, the slab read once, not twice; `false` = the loop form, the A/B arm (`docs/models/qwen3.8-flash-next.md`).
    #[opt(
        flag = "gdn-step-reg",
        env = "LLOOM_GDN_STEP_REG",
        bool,
        path = "kernels.deltanet.register_step",
        status = "advanced"
    )]
    pub step_reg: bool,
}

impl Default for GdnTuning {
    fn default() -> Self {
        GdnTuning {
            split: 8,
            step_reg: true,
        }
    }
}

/// The Metal command stream and the device (`commands.rs`, `completion.rs`,
/// `encoder.rs`, `heap.rs`, `io_queue.rs`, `icb.rs`, `kernel.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct MetalTuning {
    /// Command API: metal3 (default), metal4 (experimental reusable native buffers), or auto (Metal 4 when available and compatible with diagnostics, otherwise Metal 3).
    #[opt(flag = "metal-api", env = "LLOOM_METAL_API", parser = "parse_command_api", variants = ["metal3", "metal4", "auto"], path = "backend.metal.command_buffer.api", status = "arm", flows = ["commands-out"])]
    pub command_api: String,
    /// How the command stream orders work behind an input a host thread or the IO queue produces: `commit` (the producer commits the dependent recording once the input exists; the GPU never waits on a host-produced signal) or `gpu-wait` (the dependent recording is committed at once behind a GPU wait on the producer's event — the A/B control; a timed-out Metal 4 queue wait releases the work).
    #[opt(flag = "host-input", env = "LLOOM_HOST_INPUT", parser = "parse_host_input", variants = ["commit", "gpu-wait"], path = "backend.metal.command_buffer.host_input", status = "arm", facets = ["wait"], flows = ["commands-out", "weights-in"])]
    pub host_input: String,
    /// How the host waits for a command buffer: `block` (the blocking wait), `event` (park on the buffer's shared event), `event:<dur>` (the same, timed out after whole milliseconds), `event-sleep[:<dur>]` (poll the event, sleeping between looks), `event-listener[:<dur>]` (an event listener, timed out). The older spellings `synchronize`, `shared-event`, `shared-event-wait[:ms]`, `shared-event-sleep[:µs]`, `shared-event-listener[:ms]` are the same arms.
    #[opt(flag = "completion-wait", env = "CANDLE_METAL_COMPLETION_WAIT", parser = "parse_completion_wait", variants = ["synchronize", "shared-event", "shared-event-sleep[:µs]", "shared-event-wait[:ms]", "shared-event-listener[:ms]"], path = "backend.metal.completion_wait", status = "arm", facets = ["wait"], flows = ["token-out", "prompt-in", "commands-out"], forms = ["block", "event", "event-sleep", "event-listener"], from_v1 = "crate::options::v1::completion_wait_from", to_v1 = "completion_wait_to", schema = "Wait", grammar = "crate::options::wait::COMPLETION_WAIT")]
    pub completion_wait: String,
    /// Dispatches per command buffer on the blocking loop and the prefill (the cut period).
    #[opt(flag = "compute-per-buffer",
        env = "CANDLE_METAL_COMPUTE_PER_BUFFER",
        parser = "parse_positive", path = "backend.metal.command_buffer.dispatches", status = "advanced", facets = ["cadence"], flows = ["commands-out", "prompt-in"])]
    pub compute_per_buffer: usize,
    /// Dispatches per command buffer on the pipelined loop (lookahead ≥ 1).
    #[opt(flag = "pipelined-compute-per-buffer",
        env = "CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER",
        parser = "parse_positive", path = "backend.metal.command_buffer.pipelined_dispatches", status = "advanced", facets = ["cadence"], flows = ["commands-out"])]
    pub pipelined_compute_per_buffer: usize,
    /// Where a command buffer is cut: `counter` (every period), `node-aligned` (at a layer boundary once the period passed), `nodes:<k>` (a layer partition).
    #[opt(flag = "cut-policy", env = "CANDLE_METAL_CUT_POLICY", parser = "parse_cut_policy", variants = ["counter", "node-aligned", "nodes:<k>"], path = "backend.metal.command_buffer.cut_policy", status = "arm", facets = ["cadence"], flows = ["commands-out"])]
    pub cut_policy: String,
    /// The layer partition's run length (`nodes:<k>` by another name; 0 = off). Takes precedence over `cut_policy` when set, as the variable did.
    #[opt(flag = "nodes-per-cb", env = "CANDLE_METAL_NODES_PER_CB", path = "backend.metal.command_buffer.cut_policy", status = "arm", facets = ["cadence"], flows = ["commands-out"], to_v1 = "nodes_to_cut_policy", legacy_only)]
    pub nodes_per_cb: usize,
    /// The hazard tracker's key: `range` (byte ranges; the arena's) or `pointer` (buffer identity); none = the default for the arm.
    #[opt(flag = "hazard-key", env = "CANDLE_METAL_HAZARD_KEY", parser = "parse_hazard_key", variants = ["none", "range", "pointer"], path = "backend.metal.hazard_key", status = "arm", facets = ["signal"], flows = ["commands-out"])]
    pub hazard_key: Option<String>,
    /// Command buffers with unretained references (the liveness discipline is ours; fuzzed by `cb_liveness`).
    #[opt(flag = "cb-unretained", env = "LLOOM_CB_UNRETAINED", bool, path = "backend.metal.command_buffer.unretained", status = "arm", facets = ["liveness"], flows = ["commands-out", "memory-return"])]
    pub cb_unretained: bool,
    /// Read a tensor back through a blit into a staging buffer rather than in place (an A/B arm).
    #[opt(flag = "to-cpu-blit", env = "CANDLE_METAL_TO_CPU_BLIT", bool, path = "backend.metal.readback_blit", status = "arm", facets = ["transport"], flows = ["token-out"])]
    pub to_cpu_blit: bool,
    /// Opt into implementation-dependent CPU mapping of non-heap Private buffers. Off by default: Private readback stages through Shared storage. Incompatible with Metal API Validation; never permits heap mappings.
    #[opt(flag = "private-cpu-mapping", env = "LLOOM_PRIVATE_CPU_MAPPING", bool, path = "backend.metal.private_cpu_mapping", status = "arm", facets = ["transport"], flows = ["token-out", "prompt-in"])]
    pub private_cpu_mapping: bool,
    /// Route every contiguous `copy2d` through the compute copy kernel instead of the blit encoder (an A/B arm).
    #[opt(
        flag = "copy2d-no-blit",
        env = "LLOOM_COPY2D_NO_BLIT",
        bool,
        path = "backend.metal.copy2d_compute",
        status = "arm"
    )]
    pub copy2d_no_blit: bool,
    /// `mlock` the quantized weight uploads (the slot pools' fix applied to the dense weights; #749 P29).
    #[opt(flag = "wire-weights", env = "LLOOM_WIRE_WEIGHTS", bool, path = "weights.wire", status = "advanced", facets = ["buffer"], flows = ["weights-in"])]
    pub wire_weights: bool,
    /// The placement heap's chunk size in MiB (≥ 16).
    #[opt(
        flag = "heap-chunk-mb",
        env = "LLOOM_HEAP_CHUNK_MB",
        parser = "parse_heap_mb",
        path = "backend.metal.heaps.chunk_mb",
        status = "advanced"
    )]
    pub heap_chunk_mb: usize,
    /// Serve pool misses from heap chunks instead of a buffer each (an A/B arm; #749 P26).
    #[opt(
        flag = "transient-heaps",
        env = "LLOOM_TRANSIENT_HEAPS",
        bool,
        path = "backend.metal.heaps.transient",
        status = "arm"
    )]
    pub transient_heaps: bool,
    /// IO command buffers a point-of-use batch of the `io` loader is split across (≥ 1).
    #[opt(flag = "io-lanes", env = "LLOOM_IO_LANES", parser = "parse_positive", path = "weights.io.lanes", status = "advanced", facets = ["transport", "cadence"], flows = ["weights-in"])]
    pub io_lanes: usize,
    /// Extra Metal compiler flags for the kernel library (an A/B arm; none = the default arm).
    #[opt(
        flag = "metal-cflags",
        env = "LLOOM_434_METAL_CFLAGS",
        parser = "parse_cflags",
        path = "backend.metal.compiler.cflags",
        status = "arm"
    )]
    pub cflags: Option<String>,
    /// Compile the kernel sources as one translation unit (an A/B arm).
    #[opt(
        flag = "merged-unit",
        env = "LLOOM_477_MERGED_UNIT",
        bool,
        path = "backend.metal.compiler.merged_unit",
        status = "arm"
    )]
    pub merged_unit: bool,
    /// Lift the ICB replay quarantine on the `_swiglu_` kernels (the #732 reproducer's switch).
    #[opt(flag = "icb-quarantine-off", env = "LLOOM_ICB_QUARANTINE_OFF", bool, path = "executor.replay.quarantine_off", status = "instrument", facets = ["liveness"], flows = ["memory-return", "commands-out"])]
    pub icb_quarantine_off: bool,
    /// The ICB run's residency declaration: `per-run` (the buffers a run binds) or `plan-wide` (every buffer the plan binds, at every run).
    #[opt(flag = "icb-residency", env = "LLOOM_688_RESIDENCY", parser = "parse_icb_residency", variants = ["per-run", "plan-wide"], path = "executor.replay.residency", status = "arm")]
    pub icb_residency: String,
    /// How the device's residency set keeps its memory resident: `queue` (attached to the Metal 3 command queue, so a command buffer waits until the whole set is resident) or `heartbeat` (not attached: a command buffer waits only for what it binds, and the set is kept wired by requests — serve's keep-warm). Metal 4 keeps `queue`: its queue binds through per-slot sets, and the loaded line says so. `queue` in the library; `chat` and `serve` default to `heartbeat` on Metal (decision 56; measured on Metal 3, `tuning.md` §6.3n: the first token with the set unwired 203–216 ms queue-attached, 159–190 ms detached; warm, under a desktop-GPU load and in decode no change), `--residency-attach queue` to measure without it.
    #[opt(flag = "residency-attach", env = "LLOOM_RESIDENCY_ATTACH", parser = "parse_residency_attach", variants = ["queue", "heartbeat"], path = "kernels.residency.attach", status = "arm", facets = ["liveness"], flows = ["commands-out"])]
    pub residency_attach: String,
    /// Each residency set of a class holds at most this percentage of the device's recommended working set; past it the class takes another set (at most 24 device sets). 0 = one set for everything; 5 is MLX's per-set cap.
    #[opt(flag = "residency-set-cap-pct", env = "LLOOM_RESIDENCY_SET_CAP_PCT", parser = "parse_residency_cap_pct", path = "kernels.residency.set_cap_pct", status = "arm", facets = ["liveness"], flows = ["commands-out"])]
    pub residency_set_cap_pct: u32,
    /// The residency classes the keep-warm asks to keep resident when the sets are split (`kernels.residency.set_cap_pct` > 0): a comma list of weights, kv, heap, scratch, experts, other. The rest are left to the OS, and under memory pressure their residency is ended.
    #[opt(flag = "residency-hot", env = "LLOOM_RESIDENCY_HOT", parser = "parse_residency_hot", path = "kernels.residency.hot", status = "arm", facets = ["liveness"], flows = ["commands-out"])]
    pub residency_hot: String,
    /// Count what the Metal 4 queue's per-slot residency sets do: the allocations `Slot::resource` adds per command buffer, the set commits, and the set's `allocatedSize` at each commit (last and largest), read out in `/health` (an instrument for lane B, #1092).
    #[opt(
        flag = "residency-slot-trace",
        env = "LLOOM_METAL4_SLOT_TRACE",
        bool,
        path = "kernels.residency.slot_trace",
        status = "instrument"
    )]
    pub residency_slot_trace: bool,
    /// Run these types although this GPU only emulates them (a comma list; `bf16` on Apple7, ~2.6x the f32 time); none = refuse them (§15.2a).
    #[opt(flag = "allow-emulated", env = "LLOOM_METAL_ALLOW_EMULATED", parser = "parse_allow_emulated", variants = ["none", "bf16"], path = "backend.metal.allow_emulated", status = "advanced")]
    pub allow_emulated: Option<String>,
    /// Compile the kernel library under Metal's fast math (on: Metal's own default and the arm every recorded digest belongs to; off: the safe math arm, a different digest pair).
    #[opt(
        flag = "metal-fast-math",
        env = "CANDLE_METAL_ENABLE_FAST_MATH",
        bool,
        path = "backend.metal.compiler.fast_math",
        status = "advanced"
    )]
    pub fast_math: bool,
    /// The legacy residency switch, honoured so an old script selects the arm it names: `true` releases an evicted buffer's residency at once, `false` keeps it resident; none = `backend.pool.retain_evicted` decides.
    #[opt(flag = "eager-residency-release", env = "CANDLE_METAL_EAGER_RESIDENCY_RELEASE", bool, path = "backend.metal.eager_residency_release", status = "arm", facets = ["liveness"], flows = ["memory-return"])]
    pub eager_residency_release: Option<bool>,
    /// Withdraw these device features (a comma list) from the device's set, so the arms that need them are refused: a reference switch.
    #[opt(
        flag = "metal-disable",
        env = "LLOOM_METAL_DISABLE",
        parser = "parse_metal_disable",
        path = "backend.metal.disable_features",
        status = "reference"
    )]
    pub disable_features: Option<String>,
}

/// The cut period: dispatches per command buffer. 50 is the shipping path and
/// every recorded digest belongs to it; it is also the overlap granularity of a
/// loop that WAITS on every step -- the GPU runs the first fifty dispatches
/// while the host encodes the rest -- which is why a prefill and a blocking
/// loop want it small (lloom #715: `lfm2-decode-profile` at 512 read −8.5 %,
/// 20.0 vs 18.5 ms/token; a prompt in one buffer +1.6–2.6 ms of TTFT).
pub const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
/// The cut period for a PIPELINED decode loop -- one that encodes step `s+1`
/// while the GPU runs step `s` (`lookahead ≥ 1`), so a cut buys no overlap and
/// costs a seam. 512 is "at most one interior cut per decode step" for LFM2's
/// ~345 dispatches, with margin; measured on the crown loop (#715,
/// `measurements/issue-715-cut-period.md`): **+2.2 % [+1.9, +2.6] tok/s at
/// 350, 512 and 1024 against 50**, streams byte-identical, the visible cadence
/// the commit budget's, prefill memory unchanged. The decode loop raises the
/// period to this for its steps (`Commands::set_compute_per_buffer`) and
/// restores the default after; the prefill runs at the default.
/// `CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER` overrides it.
pub const DEFAULT_CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER: usize = 512;

/// Alignment applied to every buffer allocation.
///
/// This is the Apple Silicon cache line. Aligning to it means every buffer
/// starts on a line boundary, so no two buffers share a line and the memory
/// controller can coalesce accesses cleanly. It also covers the alignment any
/// dtype could need: the kernels use `float4`/`half4` (16 bytes) widely and
/// quantized block structures are wider still, all well under 128.
///
/// **This is the DEFAULT and no longer the only reachable value** — see
/// `metal_backend::device::buffer_alignment`, which reads `LLOOM_586_BUFFER_ALIGNMENT` once. The
/// constant is kept as the name every comment and test refers to, and as what
/// the axis reports when nothing selects an arm.
pub const BUFFER_ALIGNMENT: usize = 128;

impl Default for MetalTuning {
    fn default() -> Self {
        MetalTuning {
            command_api: "metal3".into(),
            host_input: "commit".into(),
            completion_wait: "synchronize".into(),
            compute_per_buffer: DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
            pipelined_compute_per_buffer: DEFAULT_CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER,
            cut_policy: "counter".into(),
            nodes_per_cb: 0,
            hazard_key: None,
            cb_unretained: false,
            to_cpu_blit: false,
            private_cpu_mapping: false,
            copy2d_no_blit: false,
            wire_weights: false,
            heap_chunk_mb: 512,
            transient_heaps: false,
            io_lanes: 8,
            cflags: None,
            merged_unit: false,
            icb_quarantine_off: false,
            icb_residency: "per-run".into(),
            residency_attach: "queue".into(),
            residency_set_cap_pct: 0,
            residency_hot: "weights,kv".into(),
            residency_slot_trace: false,
            allow_emulated: None,
            fast_math: true,
            eager_residency_release: None,
            disable_features: None,
        }
    }
}

/// The buffer pool (`buffer_pool.rs`, `device.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct PoolTuning {
    /// Reuse a free buffer up to this many times the request's size (none = exact-class reuse only).
    #[opt(flag = "pool-reuse-oversize-bound", env = "LLOOM_47_REUSE_OVERSIZE_BOUND", parser = "parse_oversize_bound", variants = ["none", "<factor ≥ 1.0>"], path = "backend.pool.reuse_oversize_bound", status = "arm", facets = ["liveness"], flows = ["memory-return"])]
    pub reuse_oversize_bound: Option<f64>,
    /// Which free buffer eviction takes first: `age` (oldest, what ships) or `size` (largest).
    #[opt(flag = "pool-eviction-order", env = "LLOOM_47_EVICTION_ORDER", parser = "parse_eviction", variants = ["age", "size"], path = "backend.pool.eviction_order", status = "arm", facets = ["liveness"], flows = ["memory-return"])]
    pub eviction_order: String,
    /// Never reuse a freed buffer (the corruption-class diagnostic; §2.3.8b).
    #[opt(flag = "pool-no-reuse", env = "LLOOM_POOL_NO_REUSE", bool, path = "diagnostics.checks.pool_no_reuse", status = "instrument", facets = ["liveness"], flows = ["memory-return"])]
    pub no_reuse: bool,
    /// The admission's second rung: a pass that does not fit after the free
    /// lists are trimmed is waited out for the pending lists before it is
    /// refused (`MetalDevice::admit_bytes`, `memory.md` §9.5t). It costs
    /// nothing while the budget suffices (it runs on the refusal path alone)
    /// and is counted in the ledger's `drains`; off, the refusal stands on
    /// the first figure or the tenants' pledge alone — the arm for the
    /// maintainer's numbers (#892). Stored as its negation.
    #[opt(flag = "pool-no-admission-drain",
        env = "LLOOM_POOL_NO_ADMISSION_DRAIN",
        bool, path = "memory.admission.drain", status = "arm", facets = ["wait", "liveness"], flows = ["memory-return"], invert)]
    pub no_admission_drain: bool,
    /// Completed command buffers a released buffer must outlive before reuse (0 ships; the #748 A/B).
    #[opt(flag = "pool-quarantine", env = "LLOOM_POOL_QUARANTINE", path = "backend.pool.quarantine_command_buffers", status = "arm", facets = ["liveness"], flows = ["memory-return"])]
    pub quarantine: u64,
    /// Keep evicted buffers in the residency set (the pre-#206 arm).
    #[opt(flag = "pool-retain-evicted",
        env = "CANDLE_METAL_RETAIN_EVICTED",
        bool, path = "backend.pool.retain_evicted", status = "arm", facets = ["liveness"], flows = ["memory-return"])]
    pub retain_evicted: bool,
    /// Every host upload takes the unpooled `newBufferWithBytes` path (an A/B instrument).
    #[opt(
        flag = "unpooled-uploads",
        env = "LLOOM_UNPOOLED_UPLOADS",
        bool,
        path = "backend.pool.unpooled_uploads",
        status = "arm"
    )]
    pub unpooled_uploads: bool,
    /// Stage streamed weight loads through the Shared buffer pool instead of an unpooled buffer per load (the #748 A/B).
    #[opt(flag = "stream-pooled-staging",
        env = "LLOOM_STREAM_POOLED_STAGING",
        bool, path = "weights.experts.pooled_staging", status = "arm", facets = ["buffer"], flows = ["weights-in"])]
    pub stream_pooled_staging: bool,
    /// The pool's buffer alignment in bytes (a power of two ≥ 16; §9.2r).
    #[opt(
        flag = "buffer-alignment",
        env = "LLOOM_586_BUFFER_ALIGNMENT",
        parser = "parse_pow2_align",
        path = "backend.pool.alignment",
        status = "advanced"
    )]
    pub buffer_alignment: usize,
}

impl Default for PoolTuning {
    fn default() -> Self {
        PoolTuning {
            reuse_oversize_bound: None,
            eviction_order: "age".into(),
            no_reuse: false,
            no_admission_drain: false,
            quarantine: 0,
            retain_evicted: false,
            unpooled_uploads: false,
            stream_pooled_staging: false,
            buffer_alignment: BUFFER_ALIGNMENT,
        }
    }
}

/// The instruments (`profile.rs`, `trace.rs`, `buffer_pool.rs`, `upload_shadow.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(krate = "crate")]
#[serde(default)]
pub struct InstrumentTuning {
    /// The GPU profiler: per-command-buffer GPU intervals and per-label census (`PROFILE` lines).
    #[opt(
        flag = "gpu-profile",
        env = "CANDLE_METAL_PROFILE",
        bool,
        path = "diagnostics.gpu_profile.enabled",
        status = "instrument",
        cli = "gpu-profile",
        needs = "profile",
        needs_part = "per-label census, command-buffer intervals and sampled encoder data"
    )]
    pub profile: bool,
    /// Under `profile`: time every compute encoder with stage-boundary counter samples.
    #[opt(
        flag = "profile-counters",
        env = "CANDLE_METAL_PROFILE_COUNTERS",
        bool,
        path = "diagnostics.gpu_profile.counters",
        status = "instrument",
        needs = "profile"
    )]
    pub profile_counters: bool,
    /// Under `profile`: end the compute encoder after every dispatch, so each counter-sampled encoder is one kernel (an arm: a seam per dispatch).
    #[opt(
        flag = "profile-encoder-per-dispatch",
        env = "CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH",
        bool,
        path = "diagnostics.gpu_profile.encoder_per_dispatch",
        status = "instrument",
        needs = "profile"
    )]
    pub profile_encoder_per_dispatch: bool,
    /// The dispatch trace recorder (kernel, grid, threadgroup, barrier, encoder per dispatch).
    #[opt(
        flag = "trace",
        env = "CANDLE_METAL_TRACE",
        bool,
        path = "diagnostics.trace.dispatch",
        status = "instrument",
        needs = "profile",
        cli = "trace"
    )]
    pub trace: bool,
    /// Trace every buffer-pool allocation (who missed, at what size).
    #[opt(
        flag = "pool-trace",
        env = "LLOOM_POOL_TRACE",
        bool,
        path = "diagnostics.trace.pool",
        status = "instrument",
        needs = "profile"
    )]
    pub pool_trace: bool,
    /// Attribute every pool request a served pass makes to the model call site that made it (a backtrace per request; the residue of `served_pass_pool_requests`, §9.2aa).
    #[opt(
        flag = "served-request-trace",
        env = "LLOOM_SERVED_REQUEST_TRACE",
        bool,
        path = "diagnostics.trace.served_requests",
        status = "instrument",
        needs = "profile"
    )]
    pub served_request_trace: bool,
    /// Attribute every host readback (`to_cpu`'s fast and blit paths, a staged read's stage, a host-read slot's read) to the model call site that made it (a backtrace per readback, #814 part 5).
    #[opt(flag = "readback-trace", env = "LLOOM_READBACK_TRACE", bool, path = "diagnostics.trace.readbacks", status = "instrument", needs = "profile", flows = ["token-out"])]
    pub readback_trace: bool,
    /// Shadow every host upload and check it back (the #748 corruption instrument).
    #[opt(
        flag = "upload-shadow",
        env = "LLOOM_UPLOAD_SHADOW",
        bool,
        path = "diagnostics.checks.upload_shadow",
        status = "instrument"
    )]
    pub upload_shadow: bool,
    /// Count binds of a pooled buffer after its release (`late_bind.rs`, §15.1m).
    #[opt(flag = "late-bind", env = "LLOOM_LATE_BIND", bool, path = "diagnostics.checks.late_bind", status = "instrument", facets = ["liveness"], flows = ["memory-return"])]
    pub late_bind: bool,
    /// Under `profile`: write the per-encoder and per-step tables as CSV files with this path prefix.
    #[opt(
        flag = "profile-dump",
        env = "CANDLE_METAL_PROFILE_DUMP",
        path = "diagnostics.gpu_profile.dump",
        status = "instrument",
        needs = "profile",
        needs_part = "per-label and interval tables, sampled encoder data, and the steps table's host step time column"
    )]
    pub profile_dump: Option<String>,
}

/// The CPU side (`utils.rs`, `safetensors.rs`, `backprop.rs`, `cpu/f16_matmul.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct CpuTuning {
    /// Threads the CPU backend uses (0 = the machine's).
    #[opt(
        flag = "num-threads",
        env = "CANDLE_NUM_THREADS",
        path = "backend.cpu.threads",
        status = "advanced"
    )]
    pub num_threads: usize,
    /// The `madvise` for a memory-mapped safetensors file: `willneed | sequential`; none = no advice.
    #[opt(flag = "mmap-advice", env = "CANDLE_MMAP_ADVICE", parser = "parse_mmap_advice", variants = ["none", "willneed", "sequential"], path = "weights.mmap_advice", status = "advanced", facets = ["source"], flows = ["weights-in"])]
    pub mmap_advice: Option<String>,
    /// Keep the autograd graph's detached nodes (a debugging arm).
    #[opt(
        flag = "grad-do-not-detach",
        env = "CANDLE_GRAD_DO_NOT_DETACH",
        bool,
        path = "diagnostics.checks.grad_keep_detached",
        status = "instrument"
    )]
    pub grad_do_not_detach: bool,
    /// Withdraw these host features (a comma list) from the CPU backend's set: a reference switch.
    #[opt(
        flag = "cpu-disable",
        env = "LLOOM_CPU_DISABLE",
        parser = "parse_cpu_disable",
        path = "backend.cpu.disable_features",
        status = "reference"
    )]
    pub disable_features: Option<String>,
    /// The CPU f16 matmul's accumulator type: `auto` (f32 under both contracts), `f32` (f32 throughout, rounded to f16 once: the test reference), `f16` (f16 chains promoted into f32 every `promote_every` products; aarch64 with fp16 only, refused by name elsewhere).
    #[opt(flag = "f16-matmul-accumulate", env = "LLOOM_F16_MATMUL_ACCUMULATE", parser = "parse_accum_type", variants = ["auto", "f32", "f16"], path = "kernels.matmul_f16.accumulate.type", status = "arm")]
    pub f16_matmul_accumulate: String,
    /// The CPU f16 matmul's promotion interval, in products per chain: `auto` (1 for f32, 16 for f16), `1` (f32 only), `16` (f16 only); another pairing is refused by name.
    #[opt(flag = "f16-matmul-promote-every", env = "LLOOM_F16_MATMUL_PROMOTE_EVERY", parser = "parse_promote_every", variants = ["auto", "1", "16"], path = "kernels.matmul_f16.accumulate.promote_every", status = "arm")]
    pub f16_matmul_promote_every: String,
}

impl Default for CpuTuning {
    fn default() -> Self {
        CpuTuning {
            num_threads: 0,
            mmap_advice: None,
            grad_do_not_detach: false,
            disable_features: None,
            f16_matmul_accumulate: "auto".into(),
            f16_matmul_promote_every: "auto".into(),
        }
    }
}

/// Every tuning knob of the tensor crate.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(krate = "crate")]
#[serde(default)]
pub struct Tuning {
    /// The determinism contract.
    #[opt(nested)]
    pub contract: ContractTuning,
    /// The dense GEMM/GEMV tiles.
    #[opt(nested)]
    pub gemm: GemmTuning,
    /// The quantized matvec cores.
    #[opt(nested)]
    pub qmv: QmvTuning,
    /// The quantized GEMM at prefill.
    #[opt(nested)]
    pub qmm: QmmTuning,
    /// The routed-expert GEMM.
    #[opt(nested)]
    pub moe: MoeTuning,
    /// The DeltaNet step kernel.
    #[opt(nested)]
    pub gdn: GdnTuning,
    /// The Metal command stream and the device.
    #[opt(nested)]
    pub metal: MetalTuning,
    /// The buffer pool.
    #[opt(nested)]
    pub pool: PoolTuning,
    /// The instruments.
    #[opt(nested)]
    pub instrument: InstrumentTuning,
    /// The CPU side.
    #[opt(nested)]
    pub cpu: CpuTuning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_shipping_values() {
        let t = Tuning::default();
        assert_eq!(t.qmv.q6k_nr0, 2);
        assert_eq!(t.qmv.iq2xs_nsg, 4);
        assert_eq!(t.gdn.split, 8);
        assert_eq!(t.metal.compute_per_buffer, 50);
        assert_eq!(t.metal.pipelined_compute_per_buffer, 512);
        assert_eq!(t.metal.cut_policy, "counter");
        assert_eq!(t.metal.completion_wait, "synchronize");
        assert_eq!(t.pool.buffer_alignment, 128);
        assert_eq!(t.gemm.skinny_max, 33);
        // On by default under `fast` (maintainer, 2026-10-19); `deterministic`
        // never takes it (`the_report_names_the_f16_row_axis_under_fast_only`).
        assert!(t.gemm.f16_rows);
        assert_eq!(t.gemm.f16_rows_max, 10);
        assert!(!t.gemm.f16_rows_round);
        assert_eq!(t.qmm.dequant_from, 64);
        assert_eq!(t.moe.mm_from, 64);
        assert!(t.qmm.f16_mm);
        assert_eq!(t.contract.profile, "fast");
        assert_eq!(t.cpu.f16_matmul_accumulate, "auto");
        assert_eq!(t.cpu.f16_matmul_promote_every, "auto");
    }

    #[test]
    fn the_contract_parses_two_spellings_and_refuses_the_rest() {
        let mut t = Tuning::default();
        assert!(t.set("contract.profile", "relaxed").is_err());
        assert!(t.set("contract.profile", "").is_err());
        t.set("contract.profile", "Fast").unwrap();
        assert_eq!(t.contract.profile, "fast");
        assert_eq!(Contract::parse(&t.contract.profile), Ok(Contract::Fast));
        t.set("contract.profile", "deterministic").unwrap();
        assert_eq!(
            Contract::parse(&t.contract.profile),
            Ok(Contract::Deterministic)
        );
        assert!(Contract::Fast.is_fast() && !Contract::Deterministic.is_fast());
        // The line names the CPU f16 matmul's accumulator after the arms:
        // `f32/1` under both contracts (`auto`), by the instruction this
        // CPU has. The deterministic arms are `the_report_names_the_window_arms`'.
        let accum = crate::cpu::f16_matmul::report();
        assert!(accum.starts_with("f32/1:"), "{accum}");
        assert_eq!(
            Contract::Fast.report(),
            format!(
                "profile=fast arms=moe-act-f16,decision-pack,decision-ragged,f16-rows cpu_f16_accum={accum}"
            )
        );
    }

    /// The report line names each window's arm while it is on, so a receipt
    /// says whether a `deterministic` speculative round ran the GEMM.
    #[test]
    fn the_report_names_the_window_arms() {
        let t = Tuning::default();
        assert!(
            t.gemm.verify_window && t.gemm.draft_window && t.gemm.prefill_window,
            "every window's arm is on by default"
        );
        for (verify, draft, prefill, arms) in [
            (true, true, true, "verify-gemm,draft-gemm,prefill-gemm"),
            (true, true, false, "verify-gemm,draft-gemm"),
            (true, false, true, "verify-gemm,prefill-gemm"),
            (false, true, true, "draft-gemm,prefill-gemm"),
            (false, false, true, "prefill-gemm"),
            (true, false, false, "verify-gemm"),
            (false, true, false, "draft-gemm"),
            (false, false, false, "none"),
        ] {
            Window::Verify.set_arm(verify);
            Window::Draft.set_arm(draft);
            Window::Prefill.set_arm(prefill);
            let accum = crate::cpu::f16_matmul::report();
            assert_eq!(
                Contract::Deterministic.report(),
                format!("profile=deterministic arms={arms} cpu_f16_accum={accum}")
            );
            assert!(
                Contract::Fast
                    .report()
                    .starts_with("profile=fast arms=moe-act-f16,"),
                "the window switches do not reach `fast`'s arms"
            );
        }
    }

    /// `fast`'s report line names the f16 row axis while it is on, after the
    /// static arms, and `deterministic`'s never does (nothing there takes it).
    #[test]
    fn the_report_names_the_f16_row_axis_under_fast_only() {
        let accum = crate::cpu::f16_matmul::report();
        let fast = Contract::FAST_ARMS.join(",");
        set_f16_rows(None);
        assert_eq!(f16_rows_max(), None);
        assert_eq!(
            Contract::Fast.report(),
            format!("profile=fast arms={fast} cpu_f16_accum={accum}")
        );
        let deterministic = Contract::Deterministic.report();
        set_f16_rows(Some(10));
        assert_eq!(f16_rows_max(), Some(10));
        assert_eq!(
            Contract::Fast.report(),
            format!("profile=fast arms={fast},f16-rows cpu_f16_accum={accum}")
        );
        assert_eq!(Contract::Deterministic.report(), deterministic);
    }

    /// The window is the opening thread's, nests, and closes with its guard.
    #[test]
    fn a_verify_window_is_scoped_to_its_thread_and_guard() {
        assert!(!in_verify_window());
        let outer = VerifyWindow::open();
        {
            let _inner = VerifyWindow::open();
            assert!(in_verify_window());
        }
        assert!(in_verify_window(), "the outer window still holds");
        assert!(
            !std::thread::spawn(in_verify_window).join().unwrap(),
            "another thread inherits no window"
        );
        {
            let _draft = VerifyWindow::draft();
            assert!(Window::Draft.is_open() && Window::Verify.is_open());
        }
        assert!(
            !Window::Draft.is_open(),
            "the draft window closed with its guard"
        );
        drop(outer);
        assert!(!in_verify_window());
    }

    /// A prefill window is its own kind: it opens nothing else, another
    /// thread does not see it, and it closes on every way out of the forward
    /// that opened it — a return, an `Err` propagated by `?`, an unwind.
    #[test]
    fn a_prefill_window_closes_on_every_exit() {
        Window::Prefill.set_arm(true);
        let none_open = || Window::ALL.iter().all(|w| !w.is_open());
        assert!(none_open() && window_gemm().is_none());
        {
            let _p = VerifyWindow::prefill();
            assert!(Window::Prefill.is_open());
            assert!(!Window::Verify.is_open() && !Window::Draft.is_open());
            assert_eq!(window_gemm(), Some(Window::Prefill));
            assert!(
                std::thread::spawn(|| window_gemm().is_none())
                    .join()
                    .unwrap(),
                "a decode encoded on another thread inherits no prefill window"
            );
            {
                let _v = VerifyWindow::open();
                assert_eq!(
                    window_gemm(),
                    Some(Window::Verify),
                    "a nested verify attributes to itself"
                );
            }
            assert_eq!(window_gemm(), Some(Window::Prefill));
        }
        assert!(none_open());

        fn failing_forward() -> Result<(), String> {
            let _p = VerifyWindow::prefill();
            let forward: Result<(), String> = Err("the forward failed".to_string());
            forward?;
            unreachable!("the forward's error propagates")
        }
        assert!(failing_forward().is_err());
        assert!(none_open(), "a failed prefill leaves no window behind");

        let unwound = std::panic::catch_unwind(|| {
            let _p = VerifyWindow::prefill();
            panic!("the forward panicked");
        });
        assert!(unwound.is_err());
        assert!(none_open(), "an unwound prefill leaves no window behind");

        Window::Prefill.set_arm(false);
        let _p = VerifyWindow::prefill();
        assert!(Window::Prefill.is_open());
        assert_eq!(
            window_gemm(),
            None,
            "with its arm off the window routes nothing"
        );
    }

    /// Threshold comparison keeps the finite-number grammar's earlier refusal.
    /// Pure parser calls only: no process tuning installation or native work.
    #[test]
    fn reuse_oversize_bound_keeps_finite_grammar_threshold_and_default() {
        assert_eq!(PoolTuning::default().reuse_oversize_bound, None);
        for input in ["none", " NoNe ", "\tnone\n"] {
            assert_eq!(parse_oversize_bound(input), Ok(None), "{input:?}");
        }
        for (input, expected) in [
            ("1", 1.0_f64),
            ("1e0", 1.0),
            (" 1.5 ", 1.5),
            ("1.0000000000000002", f64::from_bits(1.0_f64.to_bits() + 1)),
            ("1.7976931348623157e308", f64::MAX),
        ] {
            let got = parse_oversize_bound(input).unwrap().unwrap();
            assert_eq!(got.to_bits(), expected.to_bits(), "{input:?}");
        }
        for (input, message) in [
            ("-1", "reuse oversize bound -1 must be >= 1.0"),
            ("-0", "reuse oversize bound -0 must be >= 1.0"),
            ("0", "reuse oversize bound 0 must be >= 1.0"),
            ("0.9999999999999999", "reuse oversize bound 0.9999999999999999 must be >= 1.0"),
            ("NaN", "reuse oversize bound: `NaN` is not a finite number"),
            ("inf", "reuse oversize bound: `inf` is not a finite number"),
            ("-inf", "reuse oversize bound: `-inf` is not a finite number"),
            ("1e999", "reuse oversize bound: `1e999` is not a finite number"),
            ("off", "reuse oversize bound: `off` is not a finite number"),
            ("not-a-number", "reuse oversize bound: `not-a-number` is not a finite number"),
            ("", "reuse oversize bound: an empty value is refused (`none` unsets an optional key)"),
            (" ", "reuse oversize bound: an empty value is refused (`none` unsets an optional key)"),
        ] {
            assert_eq!(parse_oversize_bound(input), Err(message.into()), "{input:?}");
        }
    }

    #[test]
    fn the_parsers_refuse_what_the_sites_refused() {
        let mut t = Tuning::default();
        assert!(t.set("qmv.q6k_nr0", "5").is_err());
        assert!(t.set("qmv.iq2xs_nsg", "0").is_err());
        assert!(t.set("gemm.tile", "48x48").is_err());
        assert!(t.set("gemm.gemv_bm", "16").is_err());
        assert!(t.set("metal.cut_policy", "nodes:0").is_err());
        assert!(t.set("metal.completion_wait", "spin").is_err());
        assert!(t.set("pool.buffer_alignment", "100").is_err());
        assert!(t.set("metal.cflags", "").is_err());
        #[cfg(feature = "metal")]
        assert!(t.set("metal.allow_emulated", "f16").is_err());
        t.set("gemm.tile", "64x64").unwrap();
        t.set("metal.cut_policy", "nodes:15").unwrap();
        t.set("metal.completion_wait", "shared-event-wait:5")
            .unwrap();
        t.set("pool.reuse_oversize_bound", "1.5").unwrap();
        assert!(
            t.set("pool.reuse_oversize_bound", "off").is_err(),
            "`none` unsets an optional key; `off` is not a number"
        );
        t.set("pool.reuse_oversize_bound", "none").unwrap();
        assert_eq!(t.pool.reuse_oversize_bound, None);
        assert_eq!(t.gemm.tile.as_deref(), Some("64x64"));
        t.set("metal.allow_emulated", "bf16").unwrap();
        assert_eq!(t.metal.allow_emulated.as_deref(), Some("bf16"));
        t.set("metal.allow_emulated", "none").unwrap();
        assert_eq!(t.metal.allow_emulated, None);
    }

    /// The refusal names the opt-in by these spellings; they are the knob's.
    #[cfg(feature = "metal")]
    #[test]
    fn the_refusal_names_the_knob_by_its_own_spellings() {
        use crate::kernels::metal::arms::{
            ALLOW_EMULATED_ENV, ALLOW_EMULATED_FLAG, ALLOW_EMULATED_KEY,
        };
        let rows = crate::options::rows(
            "tuning",
            Tuning::META,
            &serde_json::to_value(Tuning::default()).unwrap(),
        );
        let row = rows
            .iter()
            .find(|r| r.key == ALLOW_EMULATED_KEY)
            .unwrap_or_else(|| panic!("no knob keyed {ALLOW_EMULATED_KEY}"));
        assert_eq!(row.flag, ALLOW_EMULATED_FLAG);
        assert_eq!(row.env, Some(ALLOW_EMULATED_ENV));
    }

    #[test]
    fn every_knob_has_an_env_alias() {
        let aliases = crate::options::env_aliases(Tuning::META);
        let rows = crate::options::rows(
            "tuning",
            Tuning::META,
            &serde_json::to_value(Tuning::default()).unwrap(),
        );
        assert_eq!(
            aliases.len(),
            rows.len(),
            "every leaf carries the variable it replaces"
        );
        let mut sorted = aliases.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), aliases.len(), "no two knobs share a variable");
    }
}

#[cfg(test)]
mod diagnostic_build_tests {
    #[test]
    fn an_explicit_detail_requirement_uses_the_generated_capability_without_setup() {
        let meta = crate::options::meta_at(super::InstrumentTuning::META, "profile").unwrap();
        let result = super::require_profile_recorder();
        if cfg!(feature = "metal") && super::PROFILE_INSTRUMENTS {
            result.unwrap();
        } else {
            assert_eq!(
                result,
                Err(super::TuningSetupError::NotCompiled {
                    key: meta.path,
                    capability: if cfg!(feature = "metal") {
                        meta.needs
                    } else {
                        "metal"
                    },
                })
            );
        }
        // A cold requirement must not freeze tuning or turn the recorder on.
        super::install(super::Tuning::default()).unwrap();
    }

    #[test]
    fn an_unavailable_direct_attribution_request_is_loud() {
        assert!(super::check_profile_request("served_request_trace", false).is_ok());
        assert_eq!(
            super::check_profile_request("served_request_trace", true).is_ok(),
            super::PROFILE_INSTRUMENTS
        );
    }

    #[test]
    fn whole_instruments_refuse_but_partial_counts_remain_usable() {
        let mut t = super::Tuning::default();
        assert!(t.validate_instruments().is_ok());
        t.instrument.profile = true;
        assert!(t.validate_instruments().is_ok());
        t.instrument.trace = true;
        assert_eq!(t.validate_instruments().is_ok(), super::PROFILE_INSTRUMENTS);
    }

    #[test]
    fn install_returns_incoming_payload_and_preserves_already_read() {
        let t = super::Tuning::default();
        super::install(t.clone()).unwrap();
        let mut incoming = t;
        incoming.instrument.trace = true;
        let error = super::install(incoming.clone()).unwrap_err();
        assert_eq!(error.reason, super::TuningSetupError::AlreadyRead);
        assert_eq!(error.into_tuning(), incoming);
    }
}
