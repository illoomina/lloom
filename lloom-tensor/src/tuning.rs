//! The tensor crate's tuning knobs, as one option group.
//!
//! Every `LLOOM_*` / `CANDLE_*` variable a kernel, the command stream, the
//! buffer pool or the loader used to read through its own `OnceLock` is a
//! field here, with that variable as its env alias — so the scripts that set
//! them keep working, and the same knob is also a flag (`--q6k-nr0 4`), a
//! dotted key (`--set tuning.qmv.q6k_nr0=4`), a JSON group (`--tuning-config
//! '{"qmv": {"q6k_nr0": 4}}'`) and a row of `lloom knobs`.
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

use crate::options::parse_bool;
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
    static C: OnceLock<Contract> = OnceLock::new();
    *C.get_or_init(|| Contract::parse(&tuning().contract.profile).expect("the parser accepted it"))
}

static CONTRACT_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Pin the contract for the process, before anything reads the tuning: the
/// defaults with the env aliases applied, then `c` — unless `LLOOM_PROFILE`
/// names one, which wins (a quality run asking a gate harness for `fast` has
/// named it). `Err` if the group was already fixed. What a gate harness or a
/// bit-identity test calls first, so its contract is stated, not inherited.
pub fn pin_contract(c: Contract) -> Result<Contract, String> {
    let mut t = Tuning::default();
    t.from_env().map_err(|e| format!("tuning: {e}"))?;
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

/// The group in force for this process. First use resolves it: the
/// installed group if [`install`] ran, else the defaults with the env
/// aliases applied (a malformed alias panics, naming the variable).
pub fn tuning() -> &'static Tuning {
    TUNING.get_or_init(|| {
        let mut t = Tuning::default();
        if let Err(e) = t.from_env() {
            panic!("tuning: {e}");
        }
        t
    })
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
    let n: usize = value
        .trim()
        .parse()
        .map_err(|_| format!("{what}: `{value}` is not an integer"))?;
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
fn parse_opt_nr0_4(v: &str) -> Result<Option<usize>, String> {
    if v.is_empty() || v.eq_ignore_ascii_case("none") || v == "0" {
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
    let s = v.trim().to_ascii_lowercase();
    if arms.contains(&s.as_str()) {
        Ok(s)
    } else {
        Err(format!("{what}: `{v}` is not one of {}", arms.join(", ")))
    }
}
fn optional(v: &str, f: impl Fn(&str) -> Result<String, String>) -> Result<Option<String>, String> {
    if v.is_empty() || v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
        Ok(None)
    } else {
        f(v).map(Some)
    }
}
fn parse_tile(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| one_of(s, "gemm tile", &["32x32", "32x16", "16x32", "16x16", "64x32", "64x64"]))
}
fn parse_gemv_bm(v: &str) -> Result<Option<usize>, String> {
    if v.is_empty() || v.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    match v.trim() {
        "4" => Ok(Some(4)),
        "8" => Ok(Some(8)),
        other => Err(format!("gemv bm: `{other}` is not 4 or 8 (the two instantiated decode tiles)")),
    }
}
fn parse_force_gemm(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| one_of(s, "force gemm", &["m", "n", "both"]))
}
fn parse_cut_policy(v: &str) -> Result<String, String> {
    #[cfg(feature = "metal")]
    crate::kernels::metal::commands::CutPolicy::from_env_str(v.trim())?;
    Ok(v.trim().to_string())
}
fn parse_completion_wait(v: &str) -> Result<String, String> {
    #[cfg(feature = "metal")]
    crate::kernels::metal::completion::CompletionWait::from_str_arm(v.trim())?;
    Ok(v.trim().to_string())
}
fn parse_allow_emulated(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| {
        #[cfg(feature = "metal")]
        crate::kernels::metal::arms::parse_allow_emulated(s)?;
        Ok(s.trim().to_string())
    })
}
fn parse_eviction(v: &str) -> Result<String, String> {
    one_of(v, "eviction order", &["age", "oldest", "off", "size", "largest"])
}
fn parse_hazard_key(v: &str) -> Result<Option<String>, String> {
    optional(v, |s| one_of(s, "hazard key", &["range", "pointer", "ptr"]))
}
/// `none` (and the empty string a shell passes for `CANDLE_MMAP_ADVICE=`) is
/// kept as `"none"`, not cleared: unset takes the default, which is an advised
/// arm, so clearing would turn a request to disable into the default.
fn parse_mmap_advice(v: &str) -> Result<Option<String>, String> {
    let v = v.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("none") {
        return Ok(Some("none".to_string()));
    }
    one_of(v, "mmap advice", &["willneed", "will_need", "sequential", "seq"]).map(Some)
}
fn parse_profile(v: &str) -> Result<String, String> {
    Contract::parse(v).map(|c| c.name().to_string())
}
fn parse_dequant_gemm(v: &str) -> Result<String, String> {
    one_of(v, "dequant gemm", &["qmm", "steel"])
}
fn parse_pair_order(v: &str) -> Result<String, String> {
    crate::quantized::PairOrder::parse(v).map(|o| o.name().to_string())
}
fn parse_icb_residency(v: &str) -> Result<String, String> {
    one_of(v, "icb residency", &["per-run", "plan-wide"])
}
fn parse_oversize_bound(v: &str) -> Result<Option<f64>, String> {
    if v.is_empty() || v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let f: f64 = v
        .trim()
        .parse()
        .map_err(|_| format!("reuse oversize bound: `{v}` is not a number"))?;
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
    if v.eq_ignore_ascii_case("none") {
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
    #[opt(flag = "profile", env = "LLOOM_PROFILE", parser = "parse_profile", variants = ["deterministic", "fast"])]
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
    #[opt(flag = "gemm-skinny-max", env = "LLOOM_364_SKINNY_MAX")]
    pub skinny_max: usize,
    /// Pin the GEMM tile (an A/B arm): `32x32 | 32x16 | 16x32 | 16x16 | 64x32 | 64x64`; none = the derived tile.
    #[opt(flag = "gemm-tile", env = "LLOOM_383_TILE", parser = "parse_tile", variants = ["none", "32x32", "32x16", "16x32", "16x16", "64x32", "64x64"])]
    pub tile: Option<String>,
    /// Pin the decode GEMV's `bm` (an A/B arm): 4 or 8; none = the derived tile.
    #[opt(flag = "gemv-bm", env = "LLOOM_492_GEMV_BM", parser = "parse_gemv_bm", variants = ["none", "4", "8"])]
    pub gemv_bm: Option<usize>,
    /// Force the GEMM route where the GEMV would be taken (an A/B arm): `m | n | both`; none = the rule.
    #[opt(flag = "force-gemm", env = "LLOOM_386_FORCE_GEMM", parser = "parse_force_gemm", variants = ["none", "m", "n", "both"])]
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
    #[opt(flag = "qmv-iq2xs-nsg", env = "LLOOM_QMV_IQ2XS_NSG", parser = "parse_nsg")]
    pub iq2xs_nsg: usize,
    /// Simdgroups a threadgroup on the IQ4_NL core (a function constant; 1..16).
    #[opt(flag = "qmv-iq4nl-nsg", env = "LLOOM_QMV_IQ4NL_NSG", parser = "parse_nsg")]
    pub iq4nl_nsg: usize,
    /// Simdgroups a threadgroup on the IQ2_S core, its 8 KB grid staged per threadgroup (P31 (4); 1..16).
    #[opt(flag = "qmv-iq2s-nsg", env = "LLOOM_QMV_IQ2S_NSG", parser = "parse_nsg")]
    pub iq2s_nsg: usize,
    /// Rows a simdgroup on the Q4_K core: set, two simdgroups of that many rows (ggml master's 2 × 2); none = the incumbent one simdgroup of four (P31 (5)).
    #[opt(flag = "qmv-q4k-nr0", env = "LLOOM_QMV_Q4K_NR0", parser = "parse_opt_nr0_4", variants = ["none", "1", "2", "3", "4"])]
    pub q4k_nr0: Option<usize>,
    /// Rows a simdgroup on the Q5_K core: 1 (ggml master's) or 2; none = the incumbent 2 (P31 (5)).
    #[opt(flag = "qmv-q5k-nr0", env = "LLOOM_QMV_Q5K_NR0", parser = "parse_opt_nr0_4", variants = ["none", "1", "2", "4"])]
    pub q5k_nr0: Option<usize>,
    /// Rows a simdgroup on the Q6_K core (a function constant; 1..4; 2 = ggml master's).
    #[opt(flag = "qmv-q6k-nr0", env = "LLOOM_QMV_Q6K_NR0", parser = "parse_nr0")]
    pub q6k_nr0: usize,
    /// The small-batch quantized gemv (NR1 rows per weight read) instead of per-row launches — measured no faster here; not bit-identical.
    #[opt(flag = "qmv-rows", env = "LLOOM_QMV_ROWS", bool)]
    pub rows: bool,
    /// Rows from which a quantized stem takes the mm kernel instead of the row gemvs (none = 9: up to 8 rows on the gemv).
    #[opt(flag = "qmv-mm-from", env = "LLOOM_QMV_MM_FROM")]
    pub mm_from: Option<usize>,
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
        }
    }
}

/// The quantized GEMM at prefill (`quantized/metal.rs`, `quantized/mod.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct QmmTuning {
    /// Rows from which a quantized stem's GEMM dequantises the weight to f16 once and runs the f16 arm (0 = never).
    #[opt(flag = "qmm-dequant-from", env = "LLOOM_QMM_DEQUANT_FROM")]
    pub dequant_from: usize,
    /// The GEMM over the dequantised scratch: `qmm` (the quantized GEMM's f16 arm) or `steel`.
    #[opt(flag = "qmm-dequant-gemm", env = "LLOOM_QMM_DEQUANT_GEMM", parser = "parse_dequant_gemm", variants = ["qmm", "steel"])]
    pub dequant_gemm: String,
    /// `QMatMul::TensorF16` takes the f16-weight GEMM past the mixed gemv's rows (off = the classical cast-matmul-cast route).
    #[opt(flag = "qmm-f16-mm", env = "LLOOM_QMM_F16_MM", bool)]
    pub f16_mm: bool,
    /// Dequantise every quantized matmul to f32 (a reference arm).
    #[opt(flag = "dequantize-all", env = "CANDLE_DEQUANTIZE_ALL", bool)]
    pub dequantize_all: bool,
    /// Dequantise every quantized matmul to f16 (a reference arm).
    #[opt(flag = "dequantize-all-f16", env = "CANDLE_DEQUANTIZE_ALL_F16", bool)]
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
    #[opt(flag = "moe-mm-from", env = "LLOOM_MOE_MM_FROM")]
    pub mm_from: usize,
    /// Pin the routed GEMM's row block: 16 or 32 (0 = the rule: 16 when the experts see few rows each).
    #[opt(flag = "moe-mm-bn", env = "LLOOM_MOE_MM_BN", variants = ["0", "16", "32"])]
    pub mm_bn: usize,
    /// Gate and up of a routed layer in one launch on the matmul-vec arm (`kernel_mul_mv_id2`; a GEMM tile always takes two); off = two launches, the A/B arm.
    #[opt(flag = "moe-pair", env = "LLOOM_MOE_PAIR", bool)]
    pub pair: bool,
    /// Where the pair's half sits in its launch's grid: `expert` (an expert's gate and up adjacent, the default), `half` (the slowest axis, read by a division: the previous default), `half-cmp` (the same, read by a compare), `tile` (a row tile's gate and up adjacent), or `split` (the pair's kernel once a half, a discriminator). Same bits on every arm.
    #[opt(flag = "moe-pair-order", env = "LLOOM_MOE_PAIR_ORDER", parser = "parse_pair_order", variants = ["expert", "half", "half-cmp", "tile", "split"])]
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
    #[opt(flag = "gdn-split", env = "LLOOM_GDN_SPLIT", parser = "parse_gdn_split")]
    pub split: usize,
    /// The step's register form (the default): each thread's slice of `S` held in registers, the slab read once, not twice; `false` = the loop form, the A/B arm (`docs/models/qwen3.8-flash-next.md`).
    #[opt(flag = "gdn-step-reg", env = "LLOOM_GDN_STEP_REG", bool)]
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
    /// How the host waits for a command buffer: `synchronize | shared-event | shared-event-sleep[:µs] | shared-event-wait[:ms] | shared-event-listener[:ms]`.
    #[opt(flag = "completion-wait", env = "CANDLE_METAL_COMPLETION_WAIT", parser = "parse_completion_wait", variants = ["synchronize", "shared-event", "shared-event-sleep[:µs]", "shared-event-wait[:ms]", "shared-event-listener[:ms]"])]
    pub completion_wait: String,
    /// Dispatches per command buffer on the blocking loop and the prefill (the cut period).
    #[opt(flag = "compute-per-buffer", env = "CANDLE_METAL_COMPUTE_PER_BUFFER", parser = "parse_positive")]
    pub compute_per_buffer: usize,
    /// Dispatches per command buffer on the pipelined loop (lookahead ≥ 1).
    #[opt(flag = "pipelined-compute-per-buffer", env = "CANDLE_METAL_PIPELINED_COMPUTE_PER_BUFFER", parser = "parse_positive")]
    pub pipelined_compute_per_buffer: usize,
    /// Where a command buffer is cut: `counter` (every period), `node-aligned` (at a layer boundary once the period passed), `nodes:<k>` (a layer partition).
    #[opt(flag = "cut-policy", env = "CANDLE_METAL_CUT_POLICY", parser = "parse_cut_policy", variants = ["counter", "node-aligned", "nodes:<k>"])]
    pub cut_policy: String,
    /// The layer partition's run length (`nodes:<k>` by another name; 0 = off). Takes precedence over `cut_policy` when set, as the variable did.
    #[opt(flag = "nodes-per-cb", env = "CANDLE_METAL_NODES_PER_CB")]
    pub nodes_per_cb: usize,
    /// The hazard tracker's key: `range` (byte ranges; the arena's) or `pointer` (buffer identity); none = the default for the arm.
    #[opt(flag = "hazard-key", env = "CANDLE_METAL_HAZARD_KEY", parser = "parse_hazard_key", variants = ["none", "range", "pointer"])]
    pub hazard_key: Option<String>,
    /// Command buffers with unretained references (the liveness discipline is ours; fuzzed by `cb_liveness`).
    #[opt(flag = "cb-unretained", env = "LLOOM_CB_UNRETAINED", bool)]
    pub cb_unretained: bool,
    /// Read a tensor back through a blit into a staging buffer rather than in place (an A/B arm).
    #[opt(flag = "to-cpu-blit", env = "CANDLE_METAL_TO_CPU_BLIT", bool)]
    pub to_cpu_blit: bool,
    /// Route every contiguous `copy2d` through the compute copy kernel instead of the blit encoder (an A/B arm).
    #[opt(flag = "copy2d-no-blit", env = "LLOOM_COPY2D_NO_BLIT", bool)]
    pub copy2d_no_blit: bool,
    /// `mlock` the quantized weight uploads (the slot pools' fix applied to the dense weights; #749 P29).
    #[opt(flag = "wire-weights", env = "LLOOM_WIRE_WEIGHTS", bool)]
    pub wire_weights: bool,
    /// The placement heap's chunk size in MiB (≥ 16).
    #[opt(flag = "heap-chunk-mb", env = "LLOOM_HEAP_CHUNK_MB", parser = "parse_heap_mb")]
    pub heap_chunk_mb: usize,
    /// Serve pool misses from heap chunks instead of a buffer each (an A/B arm; #749 P26).
    #[opt(flag = "transient-heaps", env = "LLOOM_TRANSIENT_HEAPS", bool)]
    pub transient_heaps: bool,
    /// IO command buffers a point-of-use batch of the `io` loader is split across (≥ 1).
    #[opt(flag = "io-lanes", env = "LLOOM_IO_LANES", parser = "parse_positive")]
    pub io_lanes: usize,
    /// Extra Metal compiler flags for the kernel library (an A/B arm; none = the default arm).
    #[opt(flag = "metal-cflags", env = "LLOOM_434_METAL_CFLAGS", parser = "parse_cflags")]
    pub cflags: Option<String>,
    /// Compile the kernel sources as one translation unit (an A/B arm).
    #[opt(flag = "merged-unit", env = "LLOOM_477_MERGED_UNIT", bool)]
    pub merged_unit: bool,
    /// Lift the ICB replay quarantine on the `_swiglu_` kernels (the #732 reproducer's switch).
    #[opt(flag = "icb-quarantine-off", env = "LLOOM_ICB_QUARANTINE_OFF", bool)]
    pub icb_quarantine_off: bool,
    /// The ICB run's residency declaration: `per-run` (the buffers a run binds) or `plan-wide` (every buffer the plan binds, at every run).
    #[opt(flag = "icb-residency", env = "LLOOM_688_RESIDENCY", parser = "parse_icb_residency", variants = ["per-run", "plan-wide"])]
    pub icb_residency: String,
    /// Run these types although this GPU only emulates them (a comma list; `bf16` on Apple7, ~2.6x the f32 time); none = refuse them (§15.2a).
    #[opt(flag = "allow-emulated", env = "LLOOM_METAL_ALLOW_EMULATED", parser = "parse_allow_emulated", variants = ["none", "bf16"])]
    pub allow_emulated: Option<String>,
}

impl Default for MetalTuning {
    fn default() -> Self {
        MetalTuning {
            completion_wait: "synchronize".into(),
            compute_per_buffer: 50,
            pipelined_compute_per_buffer: 512,
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
        }
    }
}

/// The buffer pool (`buffer_pool.rs`, `device.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[options(krate = "crate")]
#[serde(default)]
pub struct PoolTuning {
    /// Reuse a free buffer up to this many times the request's size (none = exact-class reuse only).
    #[opt(flag = "pool-reuse-oversize-bound", env = "LLOOM_47_REUSE_OVERSIZE_BOUND", parser = "parse_oversize_bound", variants = ["none", "<factor ≥ 1.0>"])]
    pub reuse_oversize_bound: Option<f64>,
    /// Which free buffer eviction takes first: `age` (oldest, what ships) or `size` (largest).
    #[opt(flag = "pool-eviction-order", env = "LLOOM_47_EVICTION_ORDER", parser = "parse_eviction", variants = ["age", "size"])]
    pub eviction_order: String,
    /// Never reuse a freed buffer (the corruption-class diagnostic; §2.3.8b).
    #[opt(flag = "pool-no-reuse", env = "LLOOM_POOL_NO_REUSE", bool)]
    pub no_reuse: bool,
    /// Completed command buffers a released buffer must outlive before reuse (0 ships; the #748 A/B).
    #[opt(flag = "pool-quarantine", env = "LLOOM_POOL_QUARANTINE")]
    pub quarantine: u64,
    /// Keep evicted buffers in the residency set (the pre-#206 arm).
    #[opt(flag = "pool-retain-evicted", env = "CANDLE_METAL_RETAIN_EVICTED", bool)]
    pub retain_evicted: bool,
    /// Every host upload takes the unpooled `newBufferWithBytes` path (an A/B instrument).
    #[opt(flag = "unpooled-uploads", env = "LLOOM_UNPOOLED_UPLOADS", bool)]
    pub unpooled_uploads: bool,
    /// Stage streamed weight loads through the Shared buffer pool instead of an unpooled buffer per load (the #748 A/B).
    #[opt(flag = "stream-pooled-staging", env = "LLOOM_STREAM_POOLED_STAGING", bool)]
    pub stream_pooled_staging: bool,
    /// The pool's buffer alignment in bytes (a power of two ≥ 16; §9.2r).
    #[opt(flag = "buffer-alignment", env = "LLOOM_586_BUFFER_ALIGNMENT", parser = "parse_pow2_align")]
    pub buffer_alignment: usize,
}

impl Default for PoolTuning {
    fn default() -> Self {
        PoolTuning {
            reuse_oversize_bound: None,
            eviction_order: "age".into(),
            no_reuse: false,
            quarantine: 0,
            retain_evicted: false,
            unpooled_uploads: false,
            stream_pooled_staging: false,
            buffer_alignment: 128,
        }
    }
}

/// The instruments (`profile.rs`, `trace.rs`, `buffer_pool.rs`, `upload_shadow.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(krate = "crate")]
#[serde(default)]
pub struct InstrumentTuning {
    /// The GPU profiler: per-command-buffer GPU intervals and per-label census (`PROFILE` lines).
    #[opt(flag = "gpu-profile", env = "CANDLE_METAL_PROFILE", bool)]
    pub profile: bool,
    /// Under `profile`: time every compute encoder with stage-boundary counter samples.
    #[opt(flag = "profile-counters", env = "CANDLE_METAL_PROFILE_COUNTERS", bool)]
    pub profile_counters: bool,
    /// Under `profile`: end the compute encoder after every dispatch, so each counter-sampled encoder is one kernel (an arm: a seam per dispatch).
    #[opt(flag = "profile-encoder-per-dispatch", env = "CANDLE_METAL_PROFILE_ENCODER_PER_DISPATCH", bool)]
    pub profile_encoder_per_dispatch: bool,
    /// The dispatch trace recorder (kernel, grid, threadgroup, barrier, encoder per dispatch).
    #[opt(flag = "trace", env = "CANDLE_METAL_TRACE", bool)]
    pub trace: bool,
    /// Trace every buffer-pool allocation (who missed, at what size).
    #[opt(flag = "pool-trace", env = "LLOOM_POOL_TRACE", bool)]
    pub pool_trace: bool,
    /// Attribute every pool request a served pass makes to the model call site that made it (a backtrace per request; the residue of `served_pass_pool_requests`, §9.2aa).
    #[opt(flag = "served-request-trace", env = "LLOOM_SERVED_REQUEST_TRACE", bool)]
    pub served_request_trace: bool,
    /// Attribute every host readback (`to_cpu`'s fast and blit paths, a staged read's stage, a host-read slot's read) to the model call site that made it (a backtrace per readback, #814 part 5).
    #[opt(flag = "readback-trace", env = "LLOOM_READBACK_TRACE", bool)]
    pub readback_trace: bool,
    /// Shadow every host upload and check it back (the #748 corruption instrument).
    #[opt(flag = "upload-shadow", env = "LLOOM_UPLOAD_SHADOW", bool)]
    pub upload_shadow: bool,
    /// Count binds of a pooled buffer after its release (`late_bind.rs`, §15.1m).
    #[opt(flag = "late-bind", env = "LLOOM_LATE_BIND", bool)]
    pub late_bind: bool,
}

/// The CPU side (`utils.rs`, `safetensors.rs`, `backprop.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[options(krate = "crate")]
#[serde(default)]
pub struct CpuTuning {
    /// Threads the CPU backend uses (0 = the machine's).
    #[opt(flag = "num-threads", env = "CANDLE_NUM_THREADS")]
    pub num_threads: usize,
    /// The `madvise` for a memory-mapped safetensors file: `willneed | sequential`; none = no advice.
    #[opt(flag = "mmap-advice", env = "CANDLE_MMAP_ADVICE", parser = "parse_mmap_advice", variants = ["none", "willneed", "sequential"])]
    pub mmap_advice: Option<String>,
    /// Keep the autograd graph's detached nodes (a debugging arm).
    #[opt(flag = "grad-do-not-detach", env = "CANDLE_GRAD_DO_NOT_DETACH", bool)]
    pub grad_do_not_detach: bool,
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

impl Tuning {
    /// The bool spelling every env alias accepts, for the sites that still
    /// read one variable of their own.
    pub fn bool_env(name: &str) -> Option<bool> {
        std::env::var(name).ok().and_then(|v| parse_bool(&v).ok())
    }
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
        t.set("metal.completion_wait", "shared-event-wait:5").unwrap();
        t.set("pool.reuse_oversize_bound", "1.5").unwrap();
        t.set("pool.reuse_oversize_bound", "off").unwrap();
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
        let rows = crate::options::rows("tuning", Tuning::META, &serde_json::to_value(Tuning::default()).unwrap());
        assert_eq!(aliases.len(), rows.len(), "every leaf carries the variable it replaces");
        let mut sorted = aliases.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), aliases.len(), "no two knobs share a variable");
    }
}
