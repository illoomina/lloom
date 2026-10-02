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
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static TUNING: OnceLock<Tuning> = OnceLock::new();

/// The group in force for this process (see the module docs).
pub fn tuning() -> &'static Tuning {
    TUNING.get_or_init(|| {
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
    TUNING.set(t)
}

fn one_of(v: &str, what: &str, arms: &[&str]) -> Result<String, String> {
    let s = v.trim().to_ascii_lowercase();
    if arms.contains(&s.as_str()) {
        Ok(s)
    } else {
        Err(format!("{what}: `{v}` is not one of {}", arms.join(", ")))
    }
}
fn parse_at_least_one(v: &str) -> Result<usize, String> {
    let n: usize = v.trim().parse().map_err(|_| format!("`{v}` is not an integer"))?;
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
    match v.trim() {
        "0" => Ok(0),
        "1" => Ok(1),
        other => Err(format!("spec ahead: `{other}` is not 0 or 1")),
    }
}
fn parse_flash_bn(v: &str) -> Result<usize, String> {
    match v.trim() {
        "8" => Ok(8),
        "32" => Ok(32),
        other => Err(format!("flash partial bn: `{other}` is not 8 or 32")),
    }
}
fn parse_ctx_kv(v: &str) -> Result<String, String> {
    if v.trim()
        .eq_ignore_ascii_case(crate::models::lfm2_dspark::CtxKv::AUTO)
    {
        return Ok(crate::models::lfm2_dspark::CtxKv::AUTO.to_string());
    }
    crate::models::lfm2_dspark::CtxKv::parse(v)
        .map(|_| v.trim().to_string())
        .map_err(|e| e.to_string())
}
fn parse_hot_thread(v: &str) -> Result<String, String> {
    one_of(v, "hot thread", &["interactive", "default", "demote", "off", "0", "none"])
}
fn parse_spike0(v: &str) -> Result<String, String> {
    one_of(v, "spike 0", &["off", "wait"])
}
fn parse_mem_check(v: &str) -> Result<String, String> {
    one_of(v, "mem check", &["off", "report", "refuse"])
}
fn parse_draft_loads(v: &str) -> Result<String, String> {
    match v {
        "auto" | "on" | "off" => Ok(v.to_string()),
        _ => Err(format!("{v:?}: expected auto, on or off")),
    }
}

fn parse_plan(v: &str) -> Result<String, String> {
    one_of(v, "expert plan", &["device", "host"])
}
fn parse_state_write(v: &str) -> Result<String, String> {
    one_of(v, "state write", &["in-place", "copy"])
}
fn parse_rows_arm(v: &str) -> Result<String, String> {
    one_of(v, "rows arm", &["batched", "per-row"])
}
fn parse_poison_value(v: &str) -> Result<String, String> {
    one_of(v, "poison value", &["nan", "finite"])
}

/// LFM2's runtime toggles (`models/lfm2.rs`, `nn/ops.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Lfm2Tuning {
    /// Run the decode conv's composed path instead of the fused kernel (the A/B for the digest gate).
    #[opt(flag = "skip-conv-c-fusion", env = "LLOOM_SKIP_CONV_C_FUSION", bool)]
    pub skip_conv_c_fusion: bool,
    /// The pool rows' kernels: `batched` (the batch in one dispatch per kernel) or `per-row` (the per-row loop, the A/B arm).
    #[opt(flag = "rows-arm", env = "LLOOM_148_ROWS_ARM", parser = "parse_rows_arm", variants = ["batched", "per-row"])]
    pub rows_arm: String,
    /// The FlashDecoding partial kernel's block width: 8 or 32 (the kernels differ only in this geometry).
    #[opt(flag = "flash-partial-bn", env = "LLOOM_FLASH_PARTIAL_BN", parser = "parse_flash_bn", variants = ["8", "32"])]
    pub flash_partial_bn: usize,
    /// Print every prefix share (pages, snapshots) — an instrument.
    #[opt(flag = "share-trace", env = "LLOOM_148_SHARE_TRACE", bool)]
    pub share_trace: bool,
    /// Print every verify poll's spins and fallback — an instrument.
    #[opt(flag = "verify-poll-trace", env = "LLOOM_VERIFY_POLL_TRACE", bool)]
    pub verify_poll_trace: bool,
    /// Fill these zero sites with a sentinel instead of zeros (`nn/zeros.rs`): `all`, or site names separated by commas — an instrument.
    #[opt(flag = "poison-zeros", env = "LLOOM_POISON_ZEROS")]
    pub poison_zeros: String,
    /// The poison's sentinel: `nan` (any read that reaches arithmetic shows) or `finite` (a read whose value matters shows).
    #[opt(flag = "poison-value", env = "LLOOM_POISON_VALUE", parser = "parse_poison_value", variants = ["nan", "finite"])]
    pub poison_value: String,
}

impl Default for Lfm2Tuning {
    fn default() -> Self {
        Lfm2Tuning {
            skip_conv_c_fusion: false,
            rows_arm: "batched".into(),
            flash_partial_bn: 32,
            share_trace: false,
            verify_poll_trace: false,
            poison_zeros: String::new(),
            poison_value: "nan".into(),
        }
    }
}

/// The DSpark drafter (`models/lfm2_dspark.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct DsparkTuning {
    /// Where the drafter keeps its context KV: `auto` (paged per sequence on Metal, contiguous on the CPU), `contiguous` (cat-grown) or `paged[:<page_size>][:shared]`.
    #[opt(flag = "dspark-ctx", env = "LLOOM_DSPARK_CTX_KV", parser = "parse_ctx_kv", variants = ["auto", "contiguous", "paged[:page][:shared]"])]
    pub ctx_kv: String,
    /// The drafter's eager attention arm (the parity gate's A/B).
    #[opt(flag = "dspark-eager-attn", env = "LLOOM_DSPARK_EAGER_ATTN", bool)]
    pub eager_attn: bool,
}

impl Default for DsparkTuning {
    fn default() -> Self {
        DsparkTuning {
            ctx_kv: "auto".into(),
            eager_attn: false,
        }
    }
}

/// The streamed expert tier (`nn/expert_stream.rs`, `nn/expert_prefetch.rs`,
/// `models/qwen4exp/{model,moe}.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct StreamTuning {
    /// `pread` straight into the slot's Shared buffer (no staging copy) — measured slower under pressure, opt-in.
    #[opt(flag = "pread-direct", env = "LLOOM_PREAD_DIRECT", bool)]
    pub pread_direct: bool,
    /// Reader threads of the expert prefetcher (≥ 1; the SSD's rate flattens past eight).
    #[opt(flag = "prefetch-threads", env = "LLOOM_PREFETCH_THREADS", parser = "parse_at_least_one")]
    pub prefetch_threads: usize,
    /// The prefetcher's reader threads' QoS: a class (`user-interactive`, `user-initiated`, `default`, `utility`) or `inherit` (as spawned: `default`).
    #[opt(flag = "prefetch-qos", env = "LLOOM_PREFETCH_QOS", parser = "parse_spawned_qos")]
    pub prefetch_qos: String,
    /// Pieces per plane a reader request covers (0 = an expert is one request, its planes read one after another).
    #[opt(flag = "prefetch-split", env = "LLOOM_PREFETCH_SPLIT")]
    pub prefetch_split: usize,
    /// Layers of lead the prefill's prefetcher runs ahead of the layer being computed (≥ 1).
    #[opt(flag = "prefetch-lead", env = "QWEN4EXP_PREFETCH_LEAD", parser = "parse_at_least_one")]
    pub prefetch_lead: usize,
    /// Where the routed experts are planned: `device` (the settle once a token) or `host` (a readback per layer, the A/B arm).
    #[opt(flag = "expert-plan", env = "QWEN4EXP_PLAN", parser = "parse_plan", variants = ["device", "host"])]
    pub plan: String,
    /// At decode under the host plan, layer l+2's router names the experts to prefetch (recall 0.56, P27).
    #[opt(flag = "decode-hint", env = "QWEN4EXP_DECODE_HINT", bool)]
    pub decode_hint: bool,
    /// Score the residual-speculation hint against the real routing — an instrument (extra router runs and readbacks).
    #[opt(flag = "hint-recall", env = "QWEN4EXP_HINT_RECALL", bool)]
    pub hint_recall: bool,
    /// Slots each streamed layer keeps free: a miss lands in one, an eviction is quarantined until its readers complete (0 = land in the victim's slot in place; the protocol a load beside the pass needs, off while none runs: each spare slot is one expert less cached).
    #[opt(flag = "spare-slots", env = "QWEN4EXP_SPARE_SLOTS")]
    pub spare_slots: usize,
    /// A token's settle submits every layer's loads before waiting for any (off = one layer after another, the A/B arm).
    #[opt(flag = "settle-batch", env = "QWEN4EXP_SETTLE_BATCH", bool)]
    pub settle_batch: bool,
    /// Spike 0 of fetch-and-wait: `wait` = a GPU wait on an already-raised event at every device-planned layer's join point (an instrument; off = none).
    #[opt(flag = "spike0", env = "QWEN4EXP_SPIKE0", parser = "parse_spike0", variants = ["off", "wait"])]
    pub spike0: String,
    /// Print every settle's ids, flags and misses — an instrument.
    #[opt(flag = "trace-settle", env = "QWEN4EXP_TRACE_SETTLE", bool)]
    pub trace_settle: bool,
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
        }
    }
}

/// Qwen4-exp's own arms (`models/qwen4exp/*`, `nn/recurrent.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Qwen4ExpTuning {
    /// The fused HyperConnection prologue pair instead of the composed chain (measured +3.6 % slower in situ, P29 (4); the arm the next bracket kernel is measured against).
    #[opt(flag = "hc-mix", env = "QWEN4EXP_HC_MIX", bool)]
    pub hc_mix: bool,
    /// Every fused op declines and the composed path runs (the A/B for the fused kernels).
    #[opt(flag = "fused-off", env = "QWEN4EXP_FUSED_OFF", bool)]
    pub fused_off: bool,
    /// The f16 routed step's SwiGLU as one launch and its combine reading the f16 routed output; off = `silu` + multiply and a cast to f32, two launches more a layer (the A/B arm).
    #[opt(flag = "qwen-f16-fuse", env = "QWEN4EXP_F16_FUSE", bool)]
    pub f16_fuse: bool,
    /// The DeltaNet prefill's chunk scan as one launch (off = the per-chunk launches).
    #[opt(flag = "chunk-scan", env = "QWEN4EXP_CHUNK_SCAN", bool)]
    pub chunk_scan: bool,
    /// The delta rule as a broadcast product + reduce instead of batched matmuls (a recorded non-deterministic arm; never shipping).
    #[opt(flag = "delta-reduce", env = "QWEN4EXP_DELTA_REDUCE", bool)]
    pub delta_reduce: bool,
    /// How a recurrent state is written: `in-place` or `copy` (an allocation per layer per token; a debugging arm).
    #[opt(flag = "state-write", env = "QWEN4EXP_STATE_WRITE", parser = "parse_state_write", variants = ["in-place", "copy"])]
    pub state_write: String,
    /// The prefill CHUNK the paired tiles are cut into (0 = the model's default).
    #[opt(flag = "qwen-prefill-chunk", env = "QWEN4EXP_PREFILL_CHUNK")]
    pub prefill_chunk: usize,
    /// The paired prefill: two chunks in one forward, the second's loads under the first's compute.
    #[opt(flag = "prefill-pair", env = "QWEN4EXP_PREFILL_PAIR", bool)]
    pub prefill_pair: bool,
    /// Pipelined prefill tiles in flight (≥ 2; the reserve and admission follow).
    #[opt(flag = "prefill-tiles", env = "QWEN4EXP_PREFILL_TILES", parser = "parse_at_least_two")]
    pub prefill_tiles: usize,
    /// Speculate the next layer's routing ahead: 0 (off) or 1.
    #[opt(flag = "spec-ahead", env = "QWEN4EXP_SPEC_AHEAD", parser = "parse_spec_ahead", variants = ["0", "1"])]
    pub spec_ahead: usize,
    /// What a pass the memory budget cannot take does: `off`, `report` or `refuse`.
    #[opt(flag = "mem-check", env = "QWEN4EXP_MEM", parser = "parse_mem_check", variants = ["off", "report", "refuse"])]
    pub mem_check: String,
    /// The QSA block score as a broadcast product + reduce instead of the matmul (an A/B instrument).
    #[opt(flag = "score-no-matmul", env = "QWEN4EXP_SCORE_NO_MATMUL", bool)]
    pub score_no_matmul: bool,
    /// A prefill chunk's queries past the QSA dense range attend through one masked dense attention over every cell instead of the paged kernel over each query's cells (an A/B arm).
    #[opt(flag = "sparse-masked", env = "QWEN4EXP_SPARSE_MASKED", bool)]
    pub sparse_masked: bool,
    /// Read the QSA selection back and check it every position — an instrument (a host sync per full layer per token).
    #[opt(flag = "check-sel", env = "QWEN4EXP_CHECK_SEL", bool)]
    pub check_sel: bool,
    /// Check every layer's outputs for non-finite values and stop at the first — an instrument.
    #[opt(flag = "watch-nan", env = "QWEN4EXP_WATCH_NAN", bool)]
    pub watch_nan: bool,
    /// The sub-block activation trace — an instrument (a readback per sub-block).
    #[opt(flag = "qwen-trace", env = "QWEN4EXP_TRACE", bool)]
    pub trace: bool,
    /// Print each stacked stem's kinds at load — an instrument.
    #[opt(flag = "trace-load", env = "QWEN4EXP_TRACE_LOAD", bool)]
    pub trace_load: bool,
    /// The MTP prompt-row ablation probe — an instrument.
    #[opt(flag = "mtp-probe", env = "QWEN4EXP_MTP_PROBE", bool)]
    pub mtp_probe: bool,
    /// Print every MTP verify round's window, argmax and accepted prefix — an instrument.
    #[opt(flag = "mtp-trace", env = "QWEN4EXP_MTP_TRACE", bool)]
    pub mtp_trace: bool,
    /// Whether a host-planned MTP verify window loads the DRAFT rows' missing experts beside row 0's: `auto` (while the real row's loads per host-planned window average 2 or more — the pool is missing the stream — off again under 1), `on` (always), `off` (never: a draft row's miss flags the row and loads nothing).
    #[opt(flag = "mtp-draft-loads", env = "QWEN4EXP_MTP_DRAFT_LOADS", parser = "parse_draft_loads", variants = ["auto", "on", "off"])]
    pub mtp_draft_loads: String,
    /// Where the MTP head's own streamed experts are planned in the drafter's observe: `device` (no readback at its layer; its record settled after the draft's argmax) or `host` (a readback per round, the A/B arm).
    #[opt(flag = "mtp-observe-plan", env = "QWEN4EXP_MTP_OBSERVE_PLAN", parser = "parse_plan", variants = ["device", "host"])]
    pub mtp_observe_plan: String,
}

impl Default for Qwen4ExpTuning {
    fn default() -> Self {
        Qwen4ExpTuning {
            hc_mix: false,
            fused_off: false,
            f16_fuse: true,
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
        }
    }
}

/// The token ring and the decode thread (`generation/{readback,hot_thread}.rs`).
#[derive(Options, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct RingTuning {
    /// Consumer-slack bytes of the token ring's history (0 = the tight geometry, the A/B baseline).
    #[opt(flag = "ring-slack-bytes", env = "LLOOM_RING_SLACK_BYTES")]
    pub slack_bytes: usize,
    /// The decode thread's QoS: `interactive` (claim `USER_INTERACTIVE`), `default`, or `off`.
    #[opt(flag = "hot-thread", env = "LLOOM_HOT_THREAD", parser = "parse_hot_thread", variants = ["interactive", "default", "off"])]
    pub hot_thread: String,
    /// The host fill producer's QoS: a class (`user-interactive`, `user-initiated`, `default`, `utility`) or `inherit` (as spawned: `default`).
    #[opt(flag = "host-feed-qos", env = "LLOOM_HOST_FEED_QOS", parser = "parse_spawned_qos")]
    pub feed_qos: String,
    /// How the host fill producer waits between tokens: `sleep:<µs>` (a spinning wait costs the streamed tier's host-planned pass GPU time), `sleep`, `wfe`, `spin-hint` or `spin`.
    #[opt(flag = "host-feed-poll", env = "LLOOM_HOST_FEED_POLL", parser = "parse_poll")]
    pub feed_poll: String,
    /// Time every host fill (encoded, observed, signalled) and the producer's loop, and print a `HOSTFEED` line a run — an instrument.
    #[opt(flag = "host-feed-trace", env = "LLOOM_HOST_FEED_TRACE", bool)]
    pub feed_trace: bool,
}

fn parse_spawned_qos(v: &str) -> Result<String, String> {
    crate::generation::hot_thread::parse_spawned_arm(v)
}
fn parse_poll(v: &str) -> Result<String, String> {
    v.trim()
        .parse::<crate::generation::poll::PollMechanism>()
        .map(|m| m.to_string())
}

impl Default for RingTuning {
    fn default() -> Self {
        RingTuning {
            slack_bytes: crate::generation::readback::RING_SLACK_BYTES_DEFAULT,
            hot_thread: "interactive".into(),
            feed_qos: "user-interactive".into(),
            feed_poll: "sleep:50".into(),
            feed_trace: false,
        }
    }
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
