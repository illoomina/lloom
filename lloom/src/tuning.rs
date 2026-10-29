//! The model crate's tuning knobs, as option groups — the `LLOOM_*` and
//! `QWEN4EXP_*` variables the model ports, the streamed tier, the drafter
//! and the ring read through their own `OnceLock`s, each now a field with
//! that variable as its env alias (`lloom_tensor::tuning` is the same shape
//! for the kernels, the command stream and the pool).
//!
//! Read once ([`tuning()`]): the group [`install`]ed by the binary after its
//! option layers resolve, else the defaults with the env aliases applied on
//! first use. Every read site keeps its one-time read; nothing per token
//! changes.

use lloom_macros::Options;
use lloom_tensor::options::{before, non_empty, parse_arm, parse_count, BeforeRow};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static TUNING: OnceLock<Tuning> = OnceLock::new();

/// The legacy env names of this group whose values meant something else
/// before they joined the option tree (`lloom_tensor::options::legacy_notes`).
pub const LEGACY_BEFORE: &[BeforeRow] = &[
    ("LLOOM_WG_SYNC", "wg.sync", before::presence),
    (
        "LLOOM_WG_CUT_AFTER_ENSURE",
        "wg.cut_after_ensure",
        before::presence,
    ),
    ("LLOOM_WG_NO_LOADER", "wg.no_loader", before::presence),
    ("LLOOM_WG_TRACE", "wg.trace", before::presence),
    // The field took the bool spellings and the batch scheduler checked for
    // presence beside it, so a false spelling left the scheduler's trace on.
    ("LLOOM_148_SHARE_TRACE", "lfm2.share_trace", |v| {
        before::derive_bool(v).map(|_| "true".into())
    }),
    (
        "QWEN4EXP_MASK",
        "stream.per_plane_mask",
        before::exactly_one,
    ),
    (
        "QWEN4EXP_READY_FIRST",
        "stream.ready_first",
        before::exactly_one,
    ),
    ("QWEN4EXP_REPLAY", "stream.replay", |v| {
        Some(
            if v == "device" {
                "\"device\""
            } else {
                "\"host\""
            }
            .into(),
        )
    }),
    ("QWEN4EXP_PREDICT", "stream.predict", |v| {
        Some(
            if v == "hotlist" {
                "\"hotlist\""
            } else {
                "\"off\""
            }
            .into(),
        )
    }),
    (
        "LLOOM_148_POOL_PAGES",
        "lfm2.pool_pages",
        before::strict_count,
    ),
    (
        "LLOOM_148_POOL_CHURN",
        "lfm2.pool_churn",
        before::strict_count,
    ),
    (
        "LLOOM_WG_SLOW_LOADER_MS",
        "wg.slow_loader_ms",
        before::count_or_unset,
    ),
    // An empty path opened nothing.
    ("LLOOM_LFM2_LOGITS_DUMP", "lfm2.logits_dump", |v| {
        if v.is_empty() {
            Some("null".into())
        } else {
            before::path_verbatim(v)
        }
    }),
];

/// [`LEGACY_BEFORE`]'s lines for the values `env` sets.
pub fn legacy_notes(env: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
    lloom_tensor::options::legacy_notes(LEGACY_BEFORE, Tuning::META, env, &|alone| {
        let mut t = Tuning::default();
        t.from_env_with(alone)?;
        serde_json::to_value(&t).map_err(|e| e.to_string())
    })
}

/// The group in force for this process (see the module docs).
pub fn tuning() -> &'static Tuning {
    TUNING.get_or_init(|| {
        for note in legacy_notes(&|name: &str| {
            std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
        }) {
            eprintln!("{note}");
        }
        let mut t = Tuning::default();
        if let Err(e) = t.from_env() {
            panic!("tuning: {e}");
        }
        t
    })
}

/// Install the group before anything reads it; `Err` hands it back if a read
/// already fixed the defaults.
pub fn install(t: Tuning) -> Result<(), Tuning> {
    lloom_tensor::isolation::claim("lloom::tuning");
    TUNING.set(t)
}

fn one_of(v: &str, what: &str, arms: &[&str]) -> Result<String, String> {
    parse_arm(v, arms)
        .map(str::to_string)
        .map_err(|e| format!("{what}: {e}"))
}
fn parse_at_least_one(v: &str) -> Result<usize, String> {
    let n: usize = parse_count(v)?;
    if n == 0 {
        return Err("must be >= 1".into());
    }
    Ok(n)
}
fn parse_at_least_two(v: &str) -> Result<usize, String> {
    let n = parse_at_least_one(v)?;
    if n < 2 {
        return Err("must be >= 2".into());
    }
    Ok(n)
}
fn parse_spec_ahead(v: &str) -> Result<usize, String> {
    match parse_count::<usize>(v).map_err(|e| format!("spec ahead: {e}"))? {
        n @ (0 | 1) => Ok(n),
        other => Err(format!("spec ahead: `{other}` is not 0 or 1")),
    }
}
fn parse_flash_bn(v: &str) -> Result<usize, String> {
    match parse_count::<usize>(v).map_err(|e| format!("flash partial bn: {e}"))? {
        n @ (8 | 32) => Ok(n),
        other => Err(format!("flash partial bn: `{other}` is not 8 or 32")),
    }
}
fn parse_ctx_kv(v: &str) -> Result<String, String> {
    let v = non_empty(v)?.to_ascii_lowercase();
    if v == crate::models::lfm2_dspark::CtxKv::AUTO {
        return Ok(v);
    }
    crate::models::lfm2_dspark::CtxKv::parse(&v)
        .map(|_| v.clone())
        .map_err(|e| e.to_string())
}
fn parse_hot_thread(v: &str) -> Result<String, String> {
    one_of(
        v,
        "hot thread",
        &["interactive", "default", "demote", "off", "0", "none"],
    )
}
fn parse_spike0(v: &str) -> Result<String, String> {
    one_of(v, "spike 0", &["off", "wait"])
}
fn parse_mem_check(v: &str) -> Result<String, String> {
    one_of(v, "mem check", &["off", "report", "refuse"])
}
fn parse_draft_loads(v: &str) -> Result<String, String> {
    one_of(v, "MTP draft loads", &["auto", "on", "off"])
}

fn parse_fusion_budget(v: &str) -> Result<String, String> {
    one_of(v, "fusion budget", &["rounding", "reorder"])
}
fn parse_mtp_experts(v: &str) -> Result<String, String> {
    one_of(v, "MTP experts", &["resident", "streamed"])
}
/// `none` is the family's own policy, as for the key it shares with LFM2's
/// axis (`kernels.fusion.policy`, unset by default): `manual`.
fn parse_fusion(v: &str) -> Result<String, String> {
    if lloom_tensor::options::is_none(v) {
        return Ok("manual".into());
    }
    one_of(v, "fusion", &["manual", "derived"])
}
fn parse_verify_key(v: &str) -> Result<String, String> {
    one_of(v, "verify key", &["kind", "variant"])
}
fn parse_qsa_window(v: &str) -> Result<String, String> {
    one_of(v, "QSA window", &["rows", "batched"])
}
fn parse_plan(v: &str) -> Result<String, String> {
    one_of(v, "expert plan", &["device", "host"])
}
fn parse_fetch_settle(v: &str) -> Result<String, String> {
    one_of(v, "fetch settle", &["behind", "now"])
}
fn parse_provisional(v: &str) -> Result<String, String> {
    one_of(v, "provisional steps", &["on", "off"])
}
fn parse_fetch(v: &str) -> Result<String, String> {
    one_of(v, "expert fetch", &["off", "inside"])
}
fn parse_state_write(v: &str) -> Result<String, String> {
    one_of(v, "state write", &["in-place", "copy"])
}
fn parse_rows_arm(v: &str) -> Result<String, String> {
    one_of(v, "rows arm", &["batched", "per-row"])
}
fn parse_predict(v: &str) -> Result<String, String> {
    one_of(v, "expert prediction", &["off", "hotlist"])
}
fn parse_replay(v: &str) -> Result<String, String> {
    one_of(v, "replay plan", &["host", "device"])
}
fn parse_poison_value(v: &str) -> Result<String, String> {
    one_of(v, "poison value", &["nan", "finite"])
}

/// LFM2's runtime toggles (`models/lfm2.rs`, `nn/ops.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Lfm2Tuning {
    /// The decode conv's fused kernel (off: the composed path, the A/B for the digest gate). Stored as its negation.
    #[opt(
        flag = "skip-conv-c-fusion",
        env = "LLOOM_SKIP_CONV_C_FUSION",
        bool,
        path = "kernels.fusion.conv_c",
        status = "arm",
        invert
    )]
    pub skip_conv_c_fusion: bool,
    /// The MLP's `silu(w1·x) · w3·x` as one elementwise launch (`silu_mul`, `silu_f16` then `bmul_f16` bit for bit) wherever the MLP's composed path runs the pair (not under `--gemv-gate mul` or the norm in the GEMVs, which keep it); off: the pair, the A/B arm. Stored as its negation.
    #[opt(
        flag = "skip-lfm2-silu-mul",
        env = "LLOOM_LFM2_SKIP_SILU_MUL",
        bool,
        path = "kernels.fusion.silu_mul",
        status = "arm",
        invert
    )]
    pub skip_silu_mul: bool,
    /// The pool rows' kernels: `batched` (the batch in one dispatch per kernel) or `per-row` (the per-row loop, the A/B arm).
    #[opt(flag = "rows-arm", env = "LLOOM_148_ROWS_ARM", parser = "parse_rows_arm", variants = ["batched", "per-row"], path = "models.lfm2.rows_kernels", status = "arm")]
    pub rows_arm: String,
    /// The FlashDecoding partial kernel's block width: 8 or 32 (the kernels differ only in this geometry).
    #[opt(flag = "flash-partial-bn", env = "LLOOM_FLASH_PARTIAL_BN", parser = "parse_flash_bn", variants = ["8", "32"], path = "kernels.attention.flash.partial_block", status = "arm")]
    pub flash_partial_bn: usize,
    /// Read B=1 continuation prefill directly from compatible paged KV, retaining the gathered path otherwise. An opt-in arm until end-to-end measurements justify a default.
    #[opt(
        flag = "lfm2-paged-prefill",
        env = "LLOOM_LFM2_PAGED_PREFILL",
        bool,
        path = "kernels.attention.paged_prefill",
        status = "arm"
    )]
    pub paged_prefill: bool,
    /// Keys from which a one-query attention takes the 2-pass vector kernel instead of the single-pass one, at every head dim; 0 = the measured threshold per head dim and dtype (`nn::ops::sdpa_vector_two_pass_from`, the arm). Moves digests at the lengths it moves.
    #[opt(
        flag = "sdpa-two-pass-k",
        env = "LLOOM_SDPA_TWO_PASS_K",
        path = "kernels.attention.sdpa.two_pass_from_keys",
        status = "advanced"
    )]
    pub sdpa_two_pass_k: usize,
    /// The decode step's RMSNorms computed inside the projections that consume them (`nn::ops::gemv_norm`: one dispatch where the pair is two; moves digests). An arm, off by default.
    #[opt(
        flag = "lfm2-norm-gemv",
        env = "LLOOM_LFM2_NORM_GEMV",
        bool,
        path = "kernels.fusion.norm_in_gemv",
        status = "arm"
    )]
    pub norm_gemv: bool,
    /// The weakest numerical class `kernels.fusion.policy = derived` may land in: `rounding` (the epilogues) or `reorder` (also the norm inside its consumers' GEMVs, whose sum of squares takes another tree). An arm.
    #[opt(flag = "lfm2-fusion-budget", env = "LLOOM_LFM2_FUSION_BUDGET", parser = "parse_fusion_budget", variants = ["rounding", "reorder"], path = "kernels.fusion.budget", status = "arm")]
    pub fusion_budget: String,
    /// Skip these dispatches — `norm` (the layers' two RMSNorms at a decode step); at a prefill pass `prefill-norm`, `prefill-silu` (the MLP's SiLU), `prefill-mul` (the SwiGLU product), `prefill-add` (the two residual adds) — to price what removing them would buy: a timing bound, the output is WRONG. An instrument.
    #[opt(
        flag = "lfm2-ablate",
        env = "LLOOM_LFM2_ABLATE",
        path = "diagnostics.ablate",
        status = "instrument"
    )]
    pub ablate: String,
    /// Print every prefix share (pages, snapshots) — an instrument.
    #[opt(
        flag = "share-trace",
        env = "LLOOM_148_SHARE_TRACE",
        bool,
        path = "diagnostics.trace.prefix_share",
        status = "instrument"
    )]
    pub share_trace: bool,
    /// Print every verify poll's spins and fallback — an instrument.
    #[opt(flag = "verify-poll-trace", env = "LLOOM_VERIFY_POLL_TRACE", bool, path = "diagnostics.trace.verify_poll", status = "instrument", flows = ["token-out"])]
    pub verify_poll_trace: bool,
    /// Fill these zero sites with a sentinel instead of zeros (`nn/zeros.rs`): `all`, or site names separated by commas — an instrument.
    #[opt(
        flag = "poison-zeros",
        env = "LLOOM_POISON_ZEROS",
        path = "diagnostics.poison.sites",
        status = "instrument"
    )]
    pub poison_zeros: String,
    /// The poison's sentinel: `nan` (any read that reaches arithmetic shows) or `finite` (a read whose value matters shows).
    #[opt(flag = "poison-value", env = "LLOOM_POISON_VALUE", parser = "parse_poison_value", variants = ["nan", "finite"], path = "diagnostics.poison.value", status = "instrument")]
    pub poison_value: String,
    /// Append every decode step's logits, `vocab` little-endian f32, to this file — an instrument (a readback per forward).
    #[opt(
        flag = "lfm2-logits-dump",
        env = "LLOOM_LFM2_LOGITS_DUMP",
        path = "diagnostics.dump.logits",
        status = "instrument"
    )]
    pub logits_dump: Option<String>,
    /// Pin the rows' paged KV pool at this many pages instead of sizing it from the KV ceiling — a fault-injection instrument.
    #[opt(
        flag = "pool-pages",
        env = "LLOOM_148_POOL_PAGES",
        path = "diagnostics.faults.pool_pages",
        status = "instrument"
    )]
    pub pool_pages: Option<usize>,
    /// Pages the rows' paged pool allocates and recycles before any sequence exists, so the sequences' page tables are not the identity — a fault-injection instrument (0 = none).
    #[opt(
        flag = "pool-churn",
        env = "LLOOM_148_POOL_CHURN",
        path = "diagnostics.faults.pool_churn_pages",
        status = "instrument"
    )]
    pub pool_churn: usize,
}

impl Default for Lfm2Tuning {
    fn default() -> Self {
        Lfm2Tuning {
            skip_conv_c_fusion: false,
            skip_silu_mul: false,
            rows_arm: "batched".into(),
            flash_partial_bn: 32,
            paged_prefill: false,
            sdpa_two_pass_k: crate::nn::ops::SDPA_TWO_PASS_K_THRESHOLD,
            norm_gemv: false,
            fusion_budget: "rounding".into(),
            ablate: String::new(),
            share_trace: false,
            verify_poll_trace: false,
            poison_zeros: String::new(),
            poison_value: "nan".into(),
            logits_dump: None,
            pool_pages: None,
            pool_churn: 0,
        }
    }
}

/// The DSpark drafter (`models/lfm2_dspark.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct DsparkTuning {
    /// Where the drafter keeps its context KV: `auto` (paged per sequence on Metal, contiguous on the CPU), `contiguous` (cat-grown) or `paged[:<page_size>][:shared]`.
    #[opt(flag = "dspark-ctx", env = "LLOOM_DSPARK_CTX_KV", parser = "parse_ctx_kv", variants = ["auto", "contiguous", "paged[:page][:shared]"], path = "speculative.dspark.context_kv", status = "advanced")]
    pub ctx_kv: String,
    /// The drafter's eager attention arm (the parity gate's A/B).
    #[opt(
        flag = "dspark-eager-attn",
        env = "LLOOM_DSPARK_EAGER_ATTN",
        bool,
        path = "speculative.dspark.eager_attention",
        status = "reference"
    )]
    pub eager_attn: bool,
    /// The draft stops at the first position whose confidence is below this (llama.cpp's `--spec-draft-p-min`); 0 is off. Read from a drafter's confidence head where it has one (LFM2's DSpark); a drafter with none drafts its whole block.
    #[opt(
        flag = "spec-p-min",
        env = "LLOOM_SPEC_P_MIN",
        path = "speculative.p_min",
        status = "arm"
    )]
    pub p_min: f64,
}

impl Default for DsparkTuning {
    fn default() -> Self {
        DsparkTuning {
            ctx_kv: "auto".into(),
            eager_attn: false,
            p_min: 0.0,
        }
    }
}

/// The streamed expert tier (`nn/expert_stream.rs`, `nn/expert_prefetch.rs`,
/// `models/qwen4exp/{model,moe}.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct StreamTuning {
    /// `pread` straight into the slot's Shared buffer (no staging copy) — measured slower under pressure, opt-in.
    #[opt(flag = "pread-direct", env = "LLOOM_PREAD_DIRECT", bool, path = "weights.experts.pread_direct", status = "arm", facets = ["transport"], flows = ["weights-in"])]
    pub pread_direct: bool,
    /// Reader threads of the expert prefetcher (≥ 1; the SSD's rate flattens past eight).
    #[opt(flag = "prefetch-threads",
        env = "LLOOM_PREFETCH_THREADS",
        parser = "parse_at_least_one", path = "weights.experts.prefetch.threads", status = "advanced", facets = ["transport"], flows = ["weights-in"])]
    pub prefetch_threads: usize,
    /// The prefetcher's reader threads' QoS: a class (`user-interactive`, `user-initiated`, `default`, `utility`) or `inherit` (as spawned: `default`).
    #[opt(flag = "prefetch-qos",
        env = "LLOOM_PREFETCH_QOS",
        parser = "parse_spawned_qos", path = "weights.experts.prefetch.qos", status = "advanced", facets = ["qos"], flows = ["weights-in"], forms = ["user-interactive", "user-initiated", "default", "utility", "background", "inherit"], schema = "ThreadQos")]
    pub prefetch_qos: String,
    /// Pieces per plane a reader request covers (0 = an expert is one request, its planes read one after another).
    #[opt(flag = "prefetch-split", env = "LLOOM_PREFETCH_SPLIT", path = "weights.experts.prefetch.split", status = "arm", facets = ["transport", "cadence"], flows = ["weights-in"])]
    pub prefetch_split: usize,
    /// Layers of lead the prefill's prefetcher runs ahead of the layer being computed (≥ 1).
    #[opt(flag = "prefetch-lead",
        env = "QWEN4EXP_PREFETCH_LEAD",
        parser = "parse_at_least_one", path = "weights.experts.prefetch.lead_layers", status = "advanced", facets = ["lead"], flows = ["weights-in"])]
    pub prefetch_lead: usize,
    /// Where the routed experts are planned: `device` (the settle once a token) or `host` (a readback per layer, the A/B arm).
    #[opt(flag = "expert-plan", env = "QWEN4EXP_PLAN", parser = "parse_plan", variants = ["device", "host"], path = "weights.experts.plan", status = "arm", facets = ["locality"], flows = ["weights-in"], schema = "Locality")]
    pub plan: String,
    /// At decode under the host plan, layer l+2's router names the experts to prefetch (recall 0.56, P27).
    #[opt(flag = "decode-hint", env = "QWEN4EXP_DECODE_HINT", bool, path = "weights.experts.decode_hint", status = "arm", facets = ["lead"], flows = ["weights-in"])]
    pub decode_hint: bool,
    /// Score the residual-speculation hint against the real routing — an instrument (extra router runs and readbacks).
    #[opt(
        flag = "hint-recall",
        env = "QWEN4EXP_HINT_RECALL",
        bool,
        path = "diagnostics.experts.hint_recall",
        status = "instrument"
    )]
    pub hint_recall: bool,
    /// Slots each streamed layer keeps free: a miss lands in one, an eviction is quarantined until its readers complete (0 = land in the victim's slot in place; the protocol a load beside the pass needs, off while none runs: each spare slot is one expert less cached).
    #[opt(flag = "spare-slots", env = "QWEN4EXP_SPARE_SLOTS", path = "weights.experts.spare_slots", status = "advanced", facets = ["buffer", "liveness"], flows = ["weights-in", "memory-return"])]
    pub spare_slots: usize,
    /// A token's settle submits every layer's loads before waiting for any (off = one layer after another, the A/B arm).
    #[opt(flag = "settle-batch", env = "QWEN4EXP_SETTLE_BATCH", bool, path = "weights.experts.settle_batch", status = "arm", facets = ["cadence"], flows = ["weights-in"])]
    pub settle_batch: bool,
    /// Spike 0 of fetch-and-wait: `wait` = a GPU wait on an already-raised event at every device-planned layer's join point (an instrument; off = none).
    #[opt(flag = "spike0", env = "QWEN4EXP_SPIKE0", parser = "parse_spike0", variants = ["off", "wait"], path = "diagnostics.experts.spike0", status = "instrument")]
    pub spike0: String,
    /// Print every settle's ids, flags and misses — an instrument.
    #[opt(flag = "trace-settle", env = "QWEN4EXP_TRACE_SETTLE", bool, path = "diagnostics.trace.expert_settle", status = "instrument", flows = ["weights-in"])]
    pub trace_settle: bool,
    /// The fetch inside the token: `inside` = while the pool misses, a device-planned decode step waits on the device at each layer for the misses its router chose, read beside the pass, instead of replaying the token or taking the host plan (`off` = the settle and replay).
    #[opt(flag = "expert-fetch", env = "QWEN4EXP_FETCH", parser = "parse_fetch", variants = ["off", "inside"], path = "weights.experts.fetch.mode", status = "arm", facets = ["wait"], flows = ["weights-in"])]
    pub fetch: String,
    /// How the fetch worker waits between looks at a layer's routing: `sleep:<dur>` (it never spins; the older spelling is the bare count of microseconds).
    #[opt(flag = "fetch-poll-us",
        env = "QWEN4EXP_FETCH_POLL_US",
        parser = "parse_at_least_one", path = "weights.experts.fetch.wait", status = "advanced", facets = ["wait"], flows = ["weights-in"], forms = ["sleep"], from_v1 = "lloom_tensor::options::v1::fetch_wait_from", to_v1 = "lloom_tensor::options::v1::fetch_wait_to", schema = "Wait", grammar = "lloom_tensor::options::wait::FETCH_WAIT")]
    pub fetch_poll_us: usize,
    /// Fetching steps in a row that load nothing before the waits stop (the pool holds the stream again).
    #[opt(flag = "fetch-exit",
        env = "QWEN4EXP_FETCH_EXIT",
        parser = "parse_at_least_one", path = "weights.experts.fetch.exit_after_steps", status = "advanced", facets = ["cadence"], flows = ["weights-in"])]
    pub fetch_exit: usize,
    /// When a fetching step settles (`QWEN4EXP_FETCH=inside`) on a driver that does not settle its steps itself (`provisional`): `behind` = after the next step is encoded, waiting for that step's own pass only, so the host encodes step t+1 while the GPU runs step t (the step's misses were loaded inside it, so its settle is bookkeeping; a pass that still computed on a missing expert — past the worker's fail-safe — is an error, not a redo, its token handed out); `now` = at the step's own end, the host waiting for the GPU every step (the previous behaviour).
    #[opt(flag = "fetch-settle", env = "QWEN4EXP_FETCH_SETTLE", parser = "parse_fetch_settle", variants = ["behind", "now"], path = "weights.experts.fetch.settle", status = "arm", facets = ["wait", "cadence"], flows = ["weights-in", "commands-out"])]
    pub fetch_settle: String,
    /// Provisional steps (`CausalLM::provisional`): a decode driver that can take its draw back (the blocking arm) settles each device-planned step itself, after its readback — the pass complete, so the settle waits for nothing — and redraws from the redone step when it computed on a missing expert; the model's step then returns without waiting for its pass. `off` = the step settles at its own end (the host waiting for the GPU every step, then again for the draw).
    #[opt(flag = "provisional", env = "QWEN4EXP_PROVISIONAL", parser = "parse_provisional", variants = ["on", "off"], path = "weights.experts.provisional_steps", status = "arm", facets = ["wait"], flows = ["weights-in", "token-out"])]
    pub provisional: String,
    /// The fetch worker's QoS: a class (`user-interactive`, …) or `inherit`.
    #[opt(flag = "fetch-qos",
        env = "QWEN4EXP_FETCH_QOS",
        parser = "parse_spawned_qos", path = "weights.experts.fetch.qos", status = "advanced", facets = ["qos"], flows = ["weights-in"], forms = ["user-interactive", "user-initiated", "default", "utility", "background", "inherit"], schema = "ThreadQos")]
    pub fetch_qos: String,
    /// On a device-planned pass, a plane whose expert the plan found missing contributes nothing to the combine (the pass without that expert, defined and finite) instead of slot 0's expert under its weight; the token is settled and replayed as without it.
    #[opt(flag = "expert-mask", env = "QWEN4EXP_MASK", bool, path = "weights.experts.per_plane_mask", status = "arm", facets = ["locality"], flows = ["weights-in"])]
    pub per_plane_mask: bool,
    /// Before a device-planned token, stage the experts the hotlist predicts that are not resident, and let the plan land them: `hotlist`, or `off`.
    #[opt(flag = "expert-predict", env = "QWEN4EXP_PREDICT", parser = "parse_predict", variants = ["off", "hotlist"], path = "weights.experts.predict", status = "arm", facets = ["lead"], flows = ["weights-in"])]
    pub predict: String,
    /// A host-planned tile's groups run in readiness order — the group with the fewest experts to load first — so the later groups' loads overlap the earlier groups' compute; off = the rows' order. Each row's arithmetic is unchanged.
    #[opt(flag = "expert-ready-first", env = "QWEN4EXP_READY_FIRST", bool, path = "weights.experts.readiness_order", status = "arm", facets = ["lead"], flows = ["weights-in", "commands-out"])]
    pub ready_first: bool,
    /// Where a decode step that missed replays: `host` (the host plan, the plan disarmed) or `device` (the device plan, kept armed).
    #[opt(flag = "expert-replay", env = "QWEN4EXP_REPLAY", parser = "parse_replay", variants = ["host", "device"], path = "weights.experts.replay_arming", status = "arm", facets = ["locality"], flows = ["weights-in", "commands-out"])]
    pub replay: String,
}

impl Default for StreamTuning {
    fn default() -> Self {
        StreamTuning {
            pread_direct: false,
            // `nn::expert_prefetch::DEFAULT_THREADS` (metal-only module): 8.
            prefetch_threads: 8,
            prefetch_qos: "default".into(),
            prefetch_split: 1,
            prefetch_lead: 2,
            plan: "device".into(),
            decode_hint: false,
            hint_recall: false,
            spare_slots: 0,
            settle_batch: true,
            spike0: "off".into(),
            trace_settle: false,
            fetch: "off".into(),
            fetch_poll_us: 50,
            fetch_exit: 8,
            fetch_settle: "behind".into(),
            provisional: "on".into(),
            fetch_qos: "user-interactive".into(),
            per_plane_mask: false,
            predict: "off".into(),
            ready_first: false,
            replay: "host".into(),
        }
    }
}

/// Qwen4-exp's own arms (`models/qwen4exp/*`, `nn/recurrent.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Qwen4ExpTuning {
    /// Who decides the bracket epilogues a decode step takes: `manual` (every one the forward can fuse, the incumbent) or `derived` (the optimizer's plan over the model's walk, once at load; issue #732).
    #[opt(flag = "qwen-fusion", env = "QWEN4EXP_FUSION", parser = "parse_fusion", variants = ["manual", "derived"], path = "kernels.fusion.policy", status = "arm")]
    pub fusion: String,
    /// The fused HyperConnection prologue pair instead of the composed chain (measured +3.6 % slower in situ, P29 (4); the arm the next bracket kernel is measured against).
    #[opt(
        flag = "hc-mix",
        env = "QWEN4EXP_HC_MIX",
        bool,
        path = "models.qwen4exp.hc.fused_mix",
        status = "arm"
    )]
    pub hc_mix: bool,
    /// Every fused op declines and the composed path runs (the A/B for the fused kernels).
    #[opt(
        flag = "fused-off",
        env = "QWEN4EXP_FUSED_OFF",
        bool,
        path = "kernels.fusion.disabled",
        status = "reference"
    )]
    pub fused_off: bool,
    /// The f16 routed step's SwiGLU as one launch and its combine reading the f16 routed output; off = `silu` + multiply and a cast to f32, two launches more a layer (the A/B arm).
    #[opt(
        flag = "qwen-f16-fuse",
        env = "QWEN4EXP_F16_FUSE",
        bool,
        path = "models.qwen4exp.moe.f16_swiglu_fused",
        status = "advanced"
    )]
    pub f16_fuse: bool,
    /// The shared expert's SwiGLU on its quantized projections' stores: the SiLU on the gate's, the product multiplied into the gate's buffer on the up's — one launch less a layer (`silu_mul`); off = the two products and `silu_mul` (the A/B arm). An arm, off by default: measured 1.1 % slower a token (`optimizer-design.md` §10k).
    #[opt(
        flag = "qwen-shared-store",
        env = "QWEN4EXP_SHARED_STORE",
        bool,
        path = "models.qwen4exp.moe.shared_expert_store",
        status = "arm"
    )]
    pub shared_store: bool,
    /// Where the MTP head's routed experts live when the target's are streamed: `streamed` (the head layer takes the target's per-layer slot count and misses as the pool does; the default) or `resident` (every expert of the head layer loaded at load and claimed beside the slot pool — 2.67 GB for the Q8 sidecar, 12 of 314 slots at 0.65 — so no observe pass is flagged and every draft comes from the fold). An arm: on the prose prompt it reads 129 rounds / 30 accepted against streamed's 123 / 36, tokens identical.
    #[opt(flag = "qwen-mtp-experts", env = "QWEN4EXP_MTP_EXPERTS", parser = "parse_mtp_experts", variants = ["resident", "streamed"], path = "speculative.mtp.experts", status = "advanced")]
    pub mtp_experts: String,
    /// The f16 routed step's activation cast folded into the gate|up launch (an f32 activation rounded to half as it loads); off = the `cast_f32_f16` launch in front of it (the A/B arm). Same bits either way.
    #[opt(
        flag = "qwen-f16-fold",
        env = "QWEN4EXP_F16_FOLD",
        bool,
        path = "models.qwen4exp.moe.f16_cast_fold",
        status = "arm"
    )]
    pub f16_fold: bool,
    /// The DeltaNet prefill's chunk scan as one launch (off = the per-chunk launches).
    #[opt(
        flag = "chunk-scan",
        env = "QWEN4EXP_CHUNK_SCAN",
        bool,
        path = "kernels.deltanet.chunk_scan",
        status = "advanced"
    )]
    pub chunk_scan: bool,
    /// The delta rule as a broadcast product + reduce instead of batched matmuls (a recorded non-deterministic arm; never shipping).
    #[opt(
        flag = "delta-reduce",
        env = "QWEN4EXP_DELTA_REDUCE",
        bool,
        path = "kernels.deltanet.delta_reduce",
        status = "reference"
    )]
    pub delta_reduce: bool,
    /// How a recurrent state is written: `in-place` or `copy` (an allocation per layer per token; a debugging arm).
    #[opt(flag = "state-write", env = "QWEN4EXP_STATE_WRITE", parser = "parse_state_write", variants = ["in-place", "copy"], path = "kernels.recurrent.state_write", status = "reference")]
    pub state_write: String,
    /// The prefill CHUNK the paired tiles are cut into (0 = the model's default).
    #[opt(flag = "qwen-prefill-chunk", env = "QWEN4EXP_PREFILL_CHUNK", path = "prefill.recurrent_chunk", status = "advanced", facets = ["cadence"], flows = ["prompt-in"])]
    pub prefill_chunk: usize,
    /// The paired prefill: two chunks in one forward, the second's loads under the first's compute.
    #[opt(flag = "prefill-pair", env = "QWEN4EXP_PREFILL_PAIR", bool, path = "prefill.pair", status = "advanced", facets = ["lead"], flows = ["prompt-in", "weights-in"])]
    pub prefill_pair: bool,
    /// Pipelined prefill tiles in flight (≥ 2; the reserve and admission follow).
    #[opt(flag = "prefill-tiles",
        env = "QWEN4EXP_PREFILL_TILES",
        parser = "parse_at_least_two", path = "prefill.tiles_in_flight", status = "advanced", facets = ["lead"], flows = ["prompt-in", "commands-out"])]
    pub prefill_tiles: usize,
    /// Speculate the next layer's routing ahead: 0 (off) or 1.
    #[opt(flag = "spec-ahead", env = "QWEN4EXP_SPEC_AHEAD", parser = "parse_spec_ahead", variants = ["0", "1"], path = "weights.experts.route_ahead_layers", status = "arm", facets = ["lead"], flows = ["weights-in"])]
    pub spec_ahead: usize,
    /// What a pass the memory budget cannot take does: `off`, `report` or `refuse`.
    #[opt(flag = "mem-check", env = "QWEN4EXP_MEM", parser = "parse_mem_check", variants = ["off", "report", "refuse"], path = "memory.on_overflow", status = "advanced")]
    pub mem_check: String,
    /// The QSA block score as a broadcast product + reduce instead of the matmul (an A/B instrument).
    #[opt(
        flag = "score-no-matmul",
        env = "QWEN4EXP_SCORE_NO_MATMUL",
        bool,
        path = "kernels.attention.qsa.score_no_matmul",
        status = "reference"
    )]
    pub score_no_matmul: bool,
    /// A prefill chunk's queries past the QSA dense range attend through one masked dense attention over every cell instead of the paged kernel over each query's cells (an A/B arm).
    #[opt(
        flag = "sparse-masked",
        env = "QWEN4EXP_SPARSE_MASKED",
        bool,
        path = "kernels.attention.qsa.masked_dense",
        status = "arm"
    )]
    pub sparse_masked: bool,
    /// Read the QSA selection back and check it every position — an instrument (a host sync per full layer per token).
    #[opt(
        flag = "check-sel",
        env = "QWEN4EXP_CHECK_SEL",
        bool,
        path = "diagnostics.checks.qsa_selection",
        status = "instrument"
    )]
    pub check_sel: bool,
    /// Check every layer's outputs for non-finite values and stop at the first — an instrument.
    #[opt(
        flag = "watch-nan",
        env = "QWEN4EXP_WATCH_NAN",
        bool,
        path = "diagnostics.checks.non_finite",
        status = "instrument"
    )]
    pub watch_nan: bool,
    /// The sub-block activation trace — an instrument (a readback per sub-block).
    #[opt(
        flag = "qwen-trace",
        env = "QWEN4EXP_TRACE",
        bool,
        path = "diagnostics.trace.activations",
        status = "instrument"
    )]
    pub trace: bool,
    /// Print each stacked stem's kinds at load — an instrument.
    #[opt(
        flag = "trace-load",
        env = "QWEN4EXP_TRACE_LOAD",
        bool,
        path = "diagnostics.trace.load",
        status = "instrument"
    )]
    pub trace_load: bool,
    /// The MTP prompt-row ablation probe — an instrument.
    #[opt(
        flag = "mtp-probe",
        env = "QWEN4EXP_MTP_PROBE",
        bool,
        path = "diagnostics.mtp_probe",
        status = "instrument"
    )]
    pub mtp_probe: bool,
    /// Print every MTP verify round's window, argmax and accepted prefix — an instrument.
    #[opt(
        flag = "mtp-trace",
        env = "QWEN4EXP_MTP_TRACE",
        bool,
        path = "diagnostics.trace.mtp_rounds",
        status = "instrument"
    )]
    pub mtp_trace: bool,
    /// Whether a host-planned MTP verify window loads the DRAFT rows' missing experts beside row 0's: `auto` (while the real row's loads per host-planned window average 2 or more — the pool is missing the stream — off again under 1), `on` (always), `off` (never: a draft row's miss flags the row and loads nothing).
    #[opt(flag = "mtp-draft-loads", env = "QWEN4EXP_MTP_DRAFT_LOADS", parser = "parse_draft_loads", variants = ["auto", "on", "off"], path = "speculative.mtp.draft_loads", status = "advanced", facets = ["lead"], flows = ["weights-in"])]
    pub mtp_draft_loads: String,
    /// Where the MTP head's own streamed experts are planned in the drafter's observe: `device` (no readback at its layer; its record settled after the draft's argmax) or `host` (a readback per round, the A/B arm).
    #[opt(flag = "mtp-observe-plan", env = "QWEN4EXP_MTP_OBSERVE_PLAN", parser = "parse_plan", variants = ["device", "host"], path = "speculative.mtp.observe_plan", status = "arm", facets = ["locality"], flows = ["weights-in", "token-out"], schema = "Locality")]
    pub mtp_observe_plan: String,
    /// Each HyperConnection bracket's grouped RMSNorm computed inside its consumers at a one-row step — the `down` and `inject` mixed GEMVs (the norm in their prologue, `down` also writing the stream scales) and the collapse (reading them) — so `hc_norm_f32` is not launched (`docs/tuning.md` §13.5u). An arm, off by default.
    #[opt(
        flag = "qwen-hc-norm-gemv",
        env = "QWEN4EXP_HC_NORM_GEMV",
        bool,
        path = "kernels.fusion.norm_in_gemv",
        status = "arm"
    )]
    pub hc_norm_gemv: bool,
    /// Skip a decode step's work by name, comma-separated (`hcnorm`: each HC bracket's grouped RMSNorm, its input passed through) — an instrument pricing a fusion's bound; the output is wrong.
    #[opt(
        flag = "qwen-ablate",
        env = "QWEN4EXP_ABLATE",
        path = "diagnostics.ablate",
        status = "instrument"
    )]
    pub ablate: String,
    /// How a verify window's planned pass is keyed: `kind` (one key per window length, the incumbent) or `variant` (also by the sequence the window runs: the rows that complete an indexer block, the device or host plan, the drafts' loads, the dense range — so each is served its own plan instead of evicting another's).
    #[opt(flag = "mtp-verify-key", env = "QWEN4EXP_MTP_VERIFY_KEY", parser = "parse_verify_key", variants = ["kind", "variant"], path = "speculative.mtp.verify_plan_key", status = "arm")]
    pub mtp_verify_key: String,
    /// The QSA layers of a verify window: `rows` (each position's prep, append, indexer and attention launched one after another, the incumbent) or `batched` (the window's positions in one launch per kernel, each row's arithmetic unchanged).
    #[opt(flag = "qsa-window", env = "QWEN4EXP_QSA_WINDOW", parser = "parse_qsa_window", variants = ["rows", "batched"], path = "kernels.attention.qsa.verify_window", status = "arm")]
    pub qsa_window: String,
    /// Run the MTP head on every verify window row inside the verify's pass (draft-for-every-prefix) and take the accepted row's draft after the accept, in place of the observe's own submission and readback.
    #[opt(
        flag = "mtp-fold",
        env = "QWEN4EXP_MTP_FOLD",
        bool,
        path = "speculative.mtp.fold",
        status = "arm"
    )]
    pub mtp_fold: bool,
}

impl Default for Qwen4ExpTuning {
    fn default() -> Self {
        Qwen4ExpTuning {
            fusion: "manual".into(),
            hc_mix: false,
            fused_off: false,
            f16_fuse: true,
            shared_store: false,
            mtp_experts: "streamed".into(),
            f16_fold: false,
            chunk_scan: true,
            delta_reduce: false,
            state_write: "in-place".into(),
            prefill_chunk: 0,
            prefill_pair: crate::models::qwen4exp::DEFAULT_PREFILL_PAIR,
            prefill_tiles: crate::models::qwen4exp::DEFAULT_PREFILL_TILES,
            spec_ahead: 0,
            mem_check: "refuse".into(),
            score_no_matmul: false,
            sparse_masked: false,
            check_sel: false,
            watch_nan: false,
            trace: false,
            trace_load: false,
            mtp_probe: false,
            mtp_trace: false,
            mtp_draft_loads: "auto".into(),
            mtp_observe_plan: "host".into(),
            hc_norm_gemv: false,
            ablate: String::new(),
            mtp_verify_key: "kind".into(),
            qsa_window: "rows".into(),
            mtp_fold: false,
        }
    }
}

/// The token ring and the decode thread (`generation/{readback,hot_thread}.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct RingTuning {
    /// Consumer-slack bytes of the token ring's history (0 = the tight geometry, the A/B baseline).
    #[opt(flag = "ring-slack-bytes", env = "LLOOM_RING_SLACK_BYTES", path = "decode.ring.slack_bytes", status = "advanced", facets = ["buffer"], flows = ["token-out"])]
    pub slack_bytes: usize,
    /// The decode thread's QoS: `user-interactive` (claim it for the run), `default` (hold the class a spawned thread has, a measurement arm) or `off` (leave the thread as found). The older spellings `interactive`, `demote`, `0` and `none` are the same arms.
    #[opt(flag = "hot-thread", env = "LLOOM_HOT_THREAD", parser = "parse_hot_thread", variants = ["interactive", "default", "off", "demote", "0", "none"], path = "decode.thread_qos", status = "advanced", facets = ["qos"], flows = ["token-out", "commands-out"], forms = ["user-interactive", "default", "off"], from_v1 = "lloom_tensor::options::v1::hot_thread_from", to_v1 = "lloom_tensor::options::v1::hot_thread_to", schema = "ThreadQos")]
    pub hot_thread: String,
    /// The host fill producer's QoS: a class (`user-interactive`, `user-initiated`, `default`, `utility`) or `inherit` (as spawned: `default`).
    #[opt(flag = "host-feed-qos",
        env = "LLOOM_HOST_FEED_QOS",
        parser = "parse_spawned_qos", path = "decode.host_feed.qos", status = "advanced", facets = ["qos"], flows = ["host-feed-in"], forms = ["user-interactive", "user-initiated", "default", "utility", "background", "inherit"], schema = "ThreadQos")]
    pub feed_qos: String,
    /// How the host fill producer waits between tokens: `adaptive[:<dur>]` (`WFE` while a forward waits on an unfilled value, else a sleep of that long: measured fastest), `event[:<dur>]` (park, woken by the ring's publish event or a forward's demand; slower where the pool misses), or one arm throughout: `sleep[:<dur>]`, `wfe`, `spin-hint`, `spin`. The older spellings (`on-demand`, durations as bare µs) are the same arms.
    #[opt(flag = "host-feed-poll",
        env = "LLOOM_HOST_FEED_POLL",
        parser = "parse_poll", path = "decode.host_feed.wait", status = "advanced", facets = ["wait"], flows = ["host-feed-in"], forms = ["adaptive", "event", "sleep", "wfe", "spin-hint", "spin"], from_v1 = "lloom_tensor::options::v1::feed_wait_from", to_v1 = "lloom_tensor::options::v1::feed_wait_to", schema = "Wait", grammar = "lloom_tensor::options::wait::HOST_FEED_WAIT")]
    pub feed_poll: String,
    /// Time every host fill (encoded, observed, signalled) and the producer's loop, and print a `HOSTFEED` line a run — an instrument.
    #[opt(flag = "host-feed-trace", env = "LLOOM_HOST_FEED_TRACE", bool, path = "diagnostics.trace.host_feed", status = "instrument", flows = ["host-feed-in"])]
    pub feed_trace: bool,
    /// How the fixed cadence's EOS wait learns the stamp: `inherit` (poll it, as the readback's own wait does) or `event` (park on the ring's publish event; measured 8 % slower a token on LFM2's `ring-fixed`, the event landing ~1 ms after the stamp).
    #[opt(flag = "ring-eos-park", env = "LLOOM_RING_EOS_PARK", bool, path = "decode.ring.eos_wait", status = "arm", facets = ["wait"], flows = ["token-out"], values = ["inherit=false", "event=true"], normalize_values, forms = ["inherit", "event"], schema = "Wait")]
    pub eos_park: bool,
    /// How DSpark's verify completion is waited for: `event` (park on an event behind the round's stamp) or `inherit` (poll the stamp, the A/B arm).
    #[opt(flag = "verify-park", env = "LLOOM_VERIFY_PARK", bool, path = "speculative.verify_wait", status = "advanced", facets = ["wait"], flows = ["token-out"], values = ["inherit=false", "event=true"], normalize_values, forms = ["inherit", "event"], schema = "Wait")]
    pub verify_park: bool,
    /// The EOS gate on Qwen3.5 (#993's arm B): off (the family's gate is clamped off; the standing window undoes the fixed cadence's one forward past an EOS) or on (`decode.readback.eos_gate` then rides the adaptive commit: the GDN's gated decode step carries its state, the overrun taken back by its counters). An arm until its real-model gates are run.
    #[opt(flag = "qwen3-5-eos-gate", env = "LLOOM_QWEN3_5_EOS_GATE", bool, path = "models.qwen3_5.eos_gate", status = "arm", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub qwen3_5_eos_gate: bool,
    /// The gate's cores arm on Qwen3.5 (#993 run 3): off (the gate stops the state; the quantized mat-vec cores of a forward queued past the EOS still run) or on (they return at once too, so a forward past the EOS costs its ungated kernels and the GDN's identity carry). Rides `models.qwen3_5.eos_gate`; taken only by a model whose dependency inventory admits it (`CausalLM::eos_gate_cores`). An arm until it is measured.
    #[opt(flag = "qwen3-5-eos-gate-cores", env = "LLOOM_QWEN3_5_EOS_GATE_CORES", bool, path = "models.qwen3_5.eos_gate_cores", status = "arm", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub qwen3_5_eos_gate_cores: bool,
    /// Whether the adaptive commit's loop watches the ring for a stop id itself (#993): off (it stops only when the reader raises the flag, so a late reader lets it run past an EOS without bound) or on (it reads what the GPU published before each step and stops at a stop id, so at most `adaptive_lead + 1` forwards run past an EOS). An arm until it is measured.
    #[opt(flag = "ring-eos-watch", env = "LLOOM_RING_EOS_WATCH", bool, path = "decode.ring.eos_watch", status = "arm", facets = ["lead"], flows = ["token-out", "commands-out"])]
    pub eos_watch: bool,
}

fn parse_spawned_qos(v: &str) -> Result<String, String> {
    crate::generation::hot_thread::parse_spawned_arm(&non_empty(v)?.to_ascii_lowercase())
}
fn parse_poll(v: &str) -> Result<String, String> {
    non_empty(v)?
        .to_ascii_lowercase()
        .parse::<crate::nn::decode::FeedWait>()
        .map(|w| w.to_string())
}

impl Default for RingTuning {
    fn default() -> Self {
        RingTuning {
            slack_bytes: crate::generation::readback::RING_SLACK_BYTES_DEFAULT,
            hot_thread: "interactive".into(),
            feed_qos: "user-interactive".into(),
            feed_poll: "on-demand:50".into(),
            feed_trace: false,
            eos_park: false,
            verify_park: true,
            qwen3_5_eos_gate: false,
            qwen3_5_eos_gate_cores: false,
            eos_watch: false,
        }
    }
}

/// The dense weight-group tier (`nn/weight_groups.rs`): its diagnosis knobs.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct WgTuning {
    /// Wait the queue out after every host write and every load request (a diagnosis arm, never on by default).
    #[opt(flag = "wg-sync", env = "LLOOM_WG_SYNC", bool, path = "weights.groups.sync", status = "arm", facets = ["wait"], flows = ["weights-in"])]
    pub sync: bool,
    /// Take the blocking path instead of the loader thread (a diagnosis arm).
    #[opt(
        flag = "wg-no-loader",
        env = "LLOOM_WG_NO_LOADER",
        bool,
        path = "weights.groups.no_loader",
        status = "instrument"
    )]
    pub no_loader: bool,
    /// Delay every loader write by this many milliseconds — a fault-injection instrument.
    #[opt(flag = "wg-slow-loader-ms", env = "LLOOM_WG_SLOW_LOADER_MS", path = "diagnostics.faults.slow_loader_ms", status = "instrument", flows = ["weights-in"])]
    pub slow_loader_ms: Option<u64>,
    /// Cut the command buffer after every `ensure` without the restamp at the next one — a fault-injection instrument.
    #[opt(
        flag = "wg-cut-after-ensure",
        env = "LLOOM_WG_CUT_AFTER_ENSURE",
        bool,
        path = "diagnostics.faults.cut_after_ensure",
        status = "instrument"
    )]
    pub cut_after_ensure: bool,
    /// Narrate every request, wait and write of the tier and its loader on stderr — an instrument.
    #[opt(flag = "wg-trace", env = "LLOOM_WG_TRACE", bool, path = "diagnostics.trace.weight_groups", status = "instrument", flows = ["weights-in"])]
    pub trace: bool,
}

/// Every tuning knob of the model crate.
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(default)]
pub struct Tuning {
    /// LFM2's runtime toggles.
    #[opt(nested)]
    pub lfm2: Lfm2Tuning,
    /// The DSpark drafter.
    #[opt(nested)]
    pub dspark: DsparkTuning,
    /// The streamed expert tier.
    #[opt(nested)]
    pub stream: StreamTuning,
    /// Qwen4-exp's own arms.
    #[opt(nested)]
    pub qwen4exp: Qwen4ExpTuning,
    /// The token ring and the decode thread.
    #[opt(nested)]
    pub ring: RingTuning,
    /// The dense weight-group tier.
    #[opt(nested)]
    pub wg: WgTuning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_knob_has_a_distinct_env_alias() {
        let aliases = lloom_tensor::options::env_aliases(Tuning::META);
        let rows = lloom_tensor::options::rows(
            "tuning",
            Tuning::META,
            &serde_json::to_value(Tuning::default()).unwrap(),
        );
        assert_eq!(aliases.len(), rows.len());
        let mut sorted = aliases.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), aliases.len());
    }

    #[test]
    fn the_parsers_refuse_what_the_sites_refused() {
        let mut t = Tuning::default();
        assert!(t.set("qwen4exp.spec_ahead", "2").is_err());
        assert!(t.set("qwen4exp.prefill_tiles", "1").is_err());
        assert!(t.set("lfm2.flash_partial_bn", "16").is_err());
        assert!(t.set("dspark.ctx_kv", "sparse").is_err());
        assert!(t.set("qwen4exp.mem_check", "maybe").is_err());
        assert!(t.set("lfm2.poison_value", "inf").is_err());
        t.set("dspark.ctx_kv", "paged:256").unwrap();
        t.set("ring.hot_thread", "off").unwrap();
        t.set("stream.plan", "host").unwrap();
        assert_eq!(t.stream.plan, "host");
    }
}
