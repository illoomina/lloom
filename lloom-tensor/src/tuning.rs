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
    pub const FAST_ARMS: &'static [&'static str] = &["moe-act-f16"];

    /// `profile=<name> arms=<list>`, for the report lines that say which
    /// contract a run held.
    pub fn report(self) -> String {
        let arms = match self {
            Contract::Deterministic => "none".to_string(),
            Contract::Fast if Contract::FAST_ARMS.is_empty() => "none".to_string(),
            Contract::Fast => Contract::FAST_ARMS.join(","),
        };
        format!("profile={} arms={arms}", self.name())
    }
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
    install(t).map_err(|_| "pin_contract: the tuning was read before the pin".to_string())?;
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

/// Install the group for the process, before anything reads it. `Err` hands
/// the group back if a read already happened (the knobs a `OnceLock` site
/// took are fixed; installing after would make the report line lie).
pub fn install(t: Tuning) -> Result<(), Tuning> {
    TUNING.set(t)
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
fn parse_cut_policy(v: &str) -> Result<String, String> {
    let v = non_empty(v)?.to_ascii_lowercase();
    #[cfg(feature = "metal")]
    crate::kernels::metal::commands::CutPolicy::from_env_str(&v)?;
    Ok(v)
}
fn parse_completion_wait(v: &str) -> Result<String, String> {
    let v = non_empty(v)?.to_ascii_lowercase();
    #[cfg(feature = "metal")]
    crate::kernels::metal::completion::CompletionWait::from_str_arm(&v)?;
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
fn parse_allow_emulated(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| {
        let s = non_empty(s)?.to_ascii_lowercase();
        #[cfg(feature = "metal")]
        crate::kernels::metal::arms::parse_allow_emulated(&s)?;
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
fn parse_oversize_bound(v: &str) -> Result<Option<f64>, String> {
    if is_none(v) {
        return Ok(None);
    }
    let f = parse_number(v).map_err(|e| format!("reuse oversize bound: {e}"))?;
    if !(f >= 1.0) {
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
}

impl Default for GemmTuning {
    fn default() -> Self {
        GemmTuning {
            // `mlx_gemm::SKINNY_TILE_MAX_M + 1` (33): the 16-row tile up to 32 rows.
            skinny_max: 33,
            tile: None,
            gemv_bm: None,
            force: None,
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
    /// The small-batch quantized gemv (NR1 rows per weight read) instead of per-row launches — measured no faster here; not bit-identical.
    #[opt(
        flag = "qmv-rows",
        env = "LLOOM_QMV_ROWS",
        bool,
        path = "kernels.quant_matvec.batched_rows",
        status = "arm"
    )]
    pub rows: bool,
    /// Rows from which a quantized stem takes the mm kernel instead of the row gemvs (none = 9: up to 8 rows on the gemv).
    #[opt(
        flag = "qmv-mm-from",
        env = "LLOOM_QMV_MM_FROM",
        path = "kernels.quant_matvec.matmul_from_rows",
        status = "advanced"
    )]
    pub mm_from: Option<usize>,
    /// The routed matvec of 2 to 8 rows (a verify window, a few rows) in its union form: the first plane naming a slot runs every plane naming it, so a slot the rows share is read once (each plane the incumbent's arithmetic; a function constant).
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
            rows: false,
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
    /// How the host waits for a command buffer: `block` (the blocking wait), `event` (park on the buffer's shared event), `event:<dur>` (the same, timed out after whole milliseconds), `event-sleep[:<dur>]` (poll the event, sleeping between looks), `event-listener[:<dur>]` (an event listener, timed out). The older spellings `synchronize`, `shared-event`, `shared-event-wait[:ms]`, `shared-event-sleep[:µs]`, `shared-event-listener[:ms]` are the same arms.
    #[opt(flag = "completion-wait", env = "CANDLE_METAL_COMPLETION_WAIT", parser = "parse_completion_wait", variants = ["synchronize", "shared-event", "shared-event-sleep[:µs]", "shared-event-wait[:ms]", "shared-event-listener[:ms]"], path = "backend.metal.completion_wait", status = "arm", facets = ["wait"], flows = ["token-out", "prompt-in", "commands-out"], forms = ["block", "event", "event-sleep", "event-listener"], from_v1 = "crate::options::v1::completion_wait_from", to_v1 = "completion_wait_to", schema = "Wait")]
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
            completion_wait: "synchronize".into(),
            compute_per_buffer: DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
            pipelined_compute_per_buffer: DEFAULT_CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER,
            cut_policy: "counter".into(),
            nodes_per_cb: 0,
            hazard_key: None,
            cb_unretained: false,
            to_cpu_blit: false,
            copy2d_no_blit: false,
            wire_weights: false,
            heap_chunk_mb: 512,
            transient_heaps: false,
            io_lanes: 8,
            cflags: None,
            merged_unit: false,
            icb_quarantine_off: false,
            icb_residency: "per-run".into(),
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
        cli = "gpu-profile"
    )]
    pub profile: bool,
    /// Under `profile`: time every compute encoder with stage-boundary counter samples.
    #[opt(
        flag = "profile-counters",
        env = "CANDLE_METAL_PROFILE_COUNTERS",
        bool,
        path = "diagnostics.gpu_profile.counters",
        status = "instrument"
    )]
    pub profile_counters: bool,
    /// Under `profile`: end the compute encoder after every dispatch, so each counter-sampled encoder is one kernel (an arm: a seam per dispatch).
    #[opt(
        flag = "profile-encoder-per-dispatch",
        env = "CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH",
        bool,
        path = "diagnostics.gpu_profile.encoder_per_dispatch",
        status = "instrument"
    )]
    pub profile_encoder_per_dispatch: bool,
    /// The dispatch trace recorder (kernel, grid, threadgroup, barrier, encoder per dispatch).
    #[opt(
        flag = "trace",
        env = "CANDLE_METAL_TRACE",
        bool,
        path = "diagnostics.trace.dispatch",
        status = "instrument",
        cli = "trace"
    )]
    pub trace: bool,
    /// Trace every buffer-pool allocation (who missed, at what size).
    #[opt(
        flag = "pool-trace",
        env = "LLOOM_POOL_TRACE",
        bool,
        path = "diagnostics.trace.pool",
        status = "instrument"
    )]
    pub pool_trace: bool,
    /// Attribute every pool request a served pass makes to the model call site that made it (a backtrace per request; the residue of `served_pass_pool_requests`, §9.2aa).
    #[opt(
        flag = "served-request-trace",
        env = "LLOOM_SERVED_REQUEST_TRACE",
        bool,
        path = "diagnostics.trace.served_requests",
        status = "instrument"
    )]
    pub served_request_trace: bool,
    /// Attribute every host readback (`to_cpu`'s fast and blit paths, a staged read's stage, a host-read slot's read) to the model call site that made it (a backtrace per readback, #814 part 5).
    #[opt(flag = "readback-trace", env = "LLOOM_READBACK_TRACE", bool, path = "diagnostics.trace.readbacks", status = "instrument", flows = ["token-out"])]
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
        status = "instrument"
    )]
    pub profile_dump: Option<String>,
}

/// The CPU side (`utils.rs`, `safetensors.rs`, `backprop.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
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
        assert_eq!(t.qmm.dequant_from, 64);
        assert_eq!(t.moe.mm_from, 64);
        assert!(t.qmm.f16_mm);
        assert_eq!(t.contract.profile, "fast");
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
        assert_eq!(
            Contract::Deterministic.report(),
            "profile=deterministic arms=none"
        );
        assert_eq!(Contract::Fast.report(), "profile=fast arms=moe-act-f16");
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
