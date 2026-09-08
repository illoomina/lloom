//! `DESIGN.md` §14.5's RELEASE mode: the number a user gets, with no instrumentation.
//!
//! Measurement harness for lloom issue #321.
//!
//! Every throughput figure this project holds was taken under `lloom-arb` with a
//! sampler, a run store and often `CANDLE_METAL_TRACE` attached. §14.5 defines a
//! second mode and nothing had ever run it. This binary is that mode:
//!
//! ```text
//! RELEASE = --no-default-features, run-telemetry/hazard-audit/debug-labels OFF,
//!           no CANDLE_METAL_* set, no sampler, no run store, no per-token series.
//!           Emits tokens, wall time, and a digest taken AFTER the window closes.
//! ```
//!
//! # What makes this different from `lfm2-decode-profile`, and it is the point
//!
//! **The timed window spans the whole loop, including sampling.** The profiler
//! opens its window *after* `sample()` (`lfm2-decode-profile/main.rs:1779` samples,
//! `:1825` starts the clock), so its `wall` excludes the sample readback that
//! §11.5a prices at **441 µs/token** — a term #299 classifies as EXPOSED, meaning
//! overlap cannot hide it. That is correct for an A/B between arms, where the term
//! is common to both. It is wrong for a number published against llama.cpp,
//! **because a user waits for it.**
//!
//! So there is exactly one `Instant` here, it opens before the first sampling call
//! and closes after the last token is pushed, and everything between is what the
//! user pays for.
//!
//! # The digest is taken after the window closes, and that is load-bearing
//!
//! It is what makes the run verifiable without instrumenting the thing being
//! measured. Token ids are pushed into a `Vec` reserved before the window opens
//! (so no allocation can occur inside it) and hashed after the clock stops. A
//! *logits* digest would require a per-step device-to-host readback of 128k f32 —
//! instrumentation inside the window — which is why §14.5 specifies a **token**
//! digest and why this harness computes only that one.
//!
//! # What this harness deliberately does not have
//!
//! No run store, no `--per-token` series, no GPU-busy timing, no memory probe, no
//! progress output inside the loop, no `lloom-sample`. Those are the diagnostic
//! mode's, and §14.5 requires the two modes never be pooled. `CANDLE_METAL_PROFILE`
//! is *env-gated rather than compiled out* (`profile.rs:31`, #287), so its hooks
//! ship in this binary whatever the variable says; leaving the variable unset is
//! all a user does, and it is what this measures.
//!
//! ```bash
//! cargo build --release --example lfm2-release \
//!     --no-default-features --features metal
//! ./target/release/examples/lfm2-release --n 500 --attn generic --turns 3
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod sha256;

use anyhow::{Context, Result};
use candle::{DType, Device};
use clap::Parser;
use lloom::generation::poll::PollMechanism;
use lloom::generation::WAKE_LIVE_EVERY_DEFAULT;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackArms, ReadbackCounters,
    ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{AttnImpl, Cache, Config, ConvState, KvAppend, Lfm2Config, Model};
use lloom::nn::decode::IN_FLIGHT_STEPS;
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode, release mode: DESIGN.md §14.5")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt fed to the model. The default matches `lfm2-determinism`'s, so the
    /// digest is comparable to the canonical pairs in `DESIGN.md` §2.3.8c.
    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Tokens to generate.
    #[arg(long, short = 'n', default_value_t = 500)]
    n: usize,

    /// Sampling seed.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Sampling temperature. 0 selects argmax.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling cutoff.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Sample by device-resident gumbel-max (plain temperature, no top-p)
    /// instead of the host nucleus path. This is the arm the token-ring can ride
    /// at t > 0 (§11.2c discharged): the token stays on the device, `to_cpu_blit`
    /// stays 0.
    #[arg(long)]
    gumbel: bool,

    /// Conversational turns against one KV cache. The canonical digest pairs in
    /// §2.3.8c are `--turns 3`, which is what the gate compares against.
    #[arg(long, default_value_t = 1)]
    turns: usize,

    /// Attention implementation: `generic`, `sdpa` or `flash`.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV append policy: `cat` or `in-place`.
    #[arg(long, default_value = "in-place")]
    kv_append: String,

    /// Conv state policy.
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// Model dtype override.
    #[arg(long)]
    dtype: Option<String>,

    /// Run on the CPU backend.
    #[arg(long)]
    cpu: bool,

    /// Steps excluded from the reported rate. The window still spans them; they
    /// are dropped from the steady-state figure, not from the run.
    ///
    /// Reported separately rather than folded, because a whole-generation average
    /// over a cold start is the shape `measurements/issue-7-reconciliation.md` §1
    /// exists to discredit.
    #[arg(long, default_value_t = 0)]
    warmup: usize,

    /// Check the SHA-256 implementation against the NIST vectors and exit.
    #[arg(long)]
    self_test: bool,

    /// **Diagnostic, and it perturbs the thing it measures.** Time `sample()`
    /// separately from the forward pass, to price the readback term inside THIS
    /// harness's window rather than inheriting #172's figure from a different
    /// window shape.
    ///
    /// Off for every published number. §11.2a records that the equivalent split
    /// on `bench/issue-172-sample-cost` "inflates the total, so only the split is
    /// read", and #299 fired a gate on an analysis that pooled the split arm's
    /// totals with the clean arm's. So a run with this flag reports the split and
    /// its total must not be compared against a run without it.
    #[arg(long)]
    sample_split: bool,

    /// **The second arm.** Run the decode loop on §11.5g's non-blocking token
    /// path: the sampled token is left on the device, written into a persistent
    /// input slot in place, and the EOS decision is made GPU-side.
    ///
    /// **Off by default and the default arm is byte-identical to the path that
    /// ships** (§7.1a: no default is flipped without its own argued decision on
    /// its own evidence, and §11.5h's prize is ~1.0 % of a step with three
    /// discounts stacked on it before anyone measures). What this flag exists
    /// for is that #500 found the mechanism had **zero production callers**, so
    /// the A/B it was sent to take had one arm; the flag is what makes a second
    /// arm exist *in the same binary*, which is #276's requirement that arms
    /// interleave rather than being two builds compared across time.
    ///
    /// **Run-time rather than compile-tier, deliberately.** §14.5a's property is
    /// that the arm is reproducible from the manifest alone, after #455's
    /// codegen flags went unrecorded. A compile-tier arm reproduces only if
    /// someone writes the build line into the artifact -- which is exactly how
    /// #455's were lost -- and it cannot put both arms in one binary at all. A
    /// run-time flag is read back from the resolved value onto the `RESULT`
    /// line by the binary itself, so the arm travels *with the number*. §7.1a-i
    /// is the reason it is read back rather than echoed: `AttnImpl` was rendered
    /// by a two-armed conditional over a three-armed axis and `--attn flash`
    /// reported `AttnImpl=Generic` on every `RESULT` line since #116, and a
    /// *wrong* arm compares equal to every run of that arm where an absent one
    /// compares equal to nothing.
    #[arg(long)]
    token_ring: bool,

    /// How the host waits for the GPU to publish, on `--token-ring` runs.
    ///
    /// `spin`, `spin-hint`, `wfe` (aarch64), `sleep`, or `sleep:<micros>`.
    /// §11.5g reports `surviving_count` = 3 on *which arm is best* and that is
    /// still not resolved; the flag exposes the seam without resolving it,
    /// which is what lets a later measurement discriminate the arms without a
    /// rebuild. An unrecognised value is an error rather than a fallback
    /// (§2.4, after #69's vacuous determinism run).
    ///
    /// **Unset takes `PollMechanism::default()`, and #578 is why that is not
    /// spelled here.** This carried `default_value = "spin"`, which *shadowed*
    /// the type's own default: flipping the default in `poll.rs` would have
    /// changed nothing on either harness, and the `RESULT` line would have gone
    /// on reporting the arm the CLI pinned. That is §7.1a-i's wrong-arm shape
    /// pointed at a default rather than at a rendering -- two sources of truth
    /// for one choice, with the losing one still on the line. The default now
    /// lives in exactly one place.
    #[arg(long)]
    poll: Option<String>,

    /// Read the EOS flag **without blocking**, on `--token-ring` runs (#535).
    ///
    /// §11.5j records that the ring as wired installs *"a hint before a blocking
    /// read, not a spin-until-published loop"* -- `wait()` fires once per token
    /// and the flag read that follows reaches `to_cpu`'s
    /// `flush_and_wait_current` anyway. §11.5k names replacing that read as
    /// **the largest single candidate** on this seam, and §11.5l measures the
    /// three poll arms as indiscriminable *because* nothing ever spins.
    ///
    /// This is that replacement, behind its own run-time flag so **both arms
    /// exist in one binary** (#513's requirement: a mechanism reachable only by
    /// rebuilding is one nothing measures, and #276's rotations need arms that
    /// interleave in one process).
    ///
    /// Off by default. §7.1a: a default is not flipped without its own argued
    /// decision on its own evidence, and the evidence here is a capability
    /// result, not a timing.
    ///
    /// Requires `--token-ring`; refused otherwise rather than silently ignored,
    /// since a run reporting an arm it did not execute is §7.1a-i's wrong-arm
    /// shape.
    #[arg(long)]
    nonblocking_eos: bool,

    /// Learn EOS host-side from the drained ids (`EosRead::HostOnDrain`, #677),
    /// on `--token-ring` runs: the device never computes the EOS flag (~5
    /// dispatches per token gone) and the loop learns EOS on the commit
    /// cadence, so it overruns an EOS by up to a commit period and the cache
    /// is rewound at the turn boundary. Needs `--lookahead 1` to mean
    /// anything, and a rewindable cache (`--kv-append in-place --conv-state
    /// rotating:K`, `K` at least the commit period) for `--turns` above 1.
    /// Mutually exclusive with `--nonblocking-eos`. Off by default (§7.1a).
    #[arg(long)]
    host_drain: bool,

    /// Carry the conv state across a multi-token pass as the convolution's left
    /// context (`ConvPrefillCtx`, `DESIGN.md` §13.2f defect 1, #677). Off, a
    /// follow-up turn's prefill zero-pads where the previous turn's last
    /// `l_cache - 1` tokens belong, so `--turns 2` computes something a single
    /// pass over the conversation does not. Off by default: every recorded
    /// `--turns 3` digest belongs to the zero-padded arm (§7.1a).
    #[arg(long, default_value_t = false)]
    conv_prefill_ctx: bool,

    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS, on `--token-ring` runs: `0` (the incumbent order — publish,
    /// wait, then encode the next forward) or `1` (encode the next forward,
    /// then wait, while it runs). `1` is the arm that closes the non-GPU gap
    /// (`design/docs/decode-driver-design.md`): the GPU always has the next
    /// step queued, so nothing idles between tokens.
    ///
    /// **`1` leaves the cache one position past an EOS** — the forward that
    /// consumed the EOS token was already enqueued when the host learned it —
    /// and the crown configuration's cache (`KvAppend=Cat`, `ConvState=Shuffle`)
    /// cannot rewind. So it is refused with `--turns` above 1 rather than
    /// letting turn two prefill onto a polluted cache and move the digest.
    /// Reported as `overrun` on the `RESULT` line.
    ///
    /// Off (`0`) by default per §7.1a: the evidence is what this flag lets a
    /// run take.
    #[arg(long, default_value_t = 0)]
    lookahead: usize,

    /// How often the decode loop commits and prunes the in-flight queue:
    /// `per-token` (the incumbent ring arm's cadence) or `steps:<N>`.
    ///
    /// Only real under `--lookahead 1`: with the wait before the next encode
    /// the publish must be committed before the host can wait on it, so the
    /// loop commits per token whatever this says, and `commits` on the
    /// `RESULT` line reports which happened.
    #[arg(long, default_value = "per-token")]
    commit_budget: String,

    /// How the host waits for a committed command buffer (`DESIGN.md` §11.5o,
    /// issue #559; §11.5t, issue #583).
    ///
    /// `synchronize` (the default and the incumbent), `shared-event`,
    /// `shared-event-sleep[:<micros>]`, `shared-event-wait[:<timeout_ms>]`, or
    /// `shared-event-listener[:<timeout_ms>]`.
    ///
    /// **The five arms are four designs, not five tunings** (#583). The two
    /// middle arms are **poll loops** over `signaledValue`; `shared-event-wait`
    /// is `waitUntilSignaledValue:timeoutMS:` and **parks in the runtime**, so
    /// it reports `completion_polls=0` structurally. Its period is a **timeout
    /// in milliseconds** — a fail-safe that falls back to blocking, not a
    /// latency knob — where the sleeping arm's is a sleep in microseconds.
    ///
    /// **`shared-event-listener` is the fourth design and the only one where
    /// the completion is DELIVERED** — `notifyListener:atValue:block:` runs a
    /// block on a GCD worker, which wakes this thread through a condvar. It
    /// reports `completion_polls=0` like the parking arm, and **that zero is
    /// not its engagement proof**: a listener that has stopped delivering falls
    /// back to blocking, computes the right answer, moves no digest and reads
    /// the same 0. `completion_fires` is what separates them.
    ///
    /// **A result from this arm owes a stated blind spot**: the block runs on a
    /// GCD queue, and §15.1i measures TSan unable to model the happens-before
    /// edge through Metal's dispatch — a report there survives `-Zbuild-std`
    /// and is still false.
    ///
    /// **Three axes on one seam, and they are at different heights.** §11.5j
    /// records the ring as wired installing *"a hint before a blocking read"*,
    /// and the three flags now attack the three parts of that sentence:
    ///
    /// | flag | what it changes |
    /// |---|---|
    /// | `--poll` | how the host idles *between* reads (§11.5l: indiscriminable, because nothing spins) |
    /// | `--nonblocking-eos` | whether the **flag read** blocks (#535) |
    /// | `--completion-wait` | whether the **command-buffer wait underneath** blocks (#559) |
    ///
    /// They compose and are not substitutes: `--nonblocking-eos` removes the
    /// block from the *one* read the ring makes per token, while this removes it
    /// from every `flush_and_wait_current` on the path -- which is what the
    /// baseline arm reaches through the sampler and what `to_cpu` reaches at
    /// every other call site.
    ///
    /// **`synchronize` stays the default.** Every canonical digest and every
    /// `.bench/` row belongs to it, and §7.1a's discipline is that a default
    /// flip is its own argued decision with a number in hand. #559 does not take
    /// that number.
    #[arg(long, default_value = "synchronize")]
    completion_wait: String,

    /// Fuse the SiLU onto its producing GEMV's store (§12.2i, #509).
    ///
    /// **The mirror of #567's first half.** That change brought the three async
    /// arms onto `lfm2-decode-profile` so one binary could select all four; the
    /// epilogues were selectable there and on `lfm2-determinism` and **not
    /// here**, so this harness — the one with the undrained queue — could select
    /// three of the four. The reopening comment asks the question directly
    /// (*"does release select all four? Grep and say"*), and the answer was no:
    /// `grep -c gemv_silu` read **0** on this file against 5 on the profiler.
    ///
    /// **The digests MOVE and that is the engagement proof, not a regression.**
    /// §12.2i measures `logits_sha256` moving on every epilogue arm while
    /// `tokens_sha256` does not — the add lands at the f32 accumulator where
    /// `badd_f16` adds at `T` on the rounded store (§12.2h). This harness
    /// computes only the token digest (§14.5's clause, and the module header
    /// says why), so **its digest is expected to be UNMOVED on this arm**, and
    /// an unmoved token digest here is therefore not evidence the arm engaged.
    /// The `RESULT` line's axis field is what separates the runs.
    ///
    /// Off by default (§7.1a).
    #[arg(long, default_value_t = false)]
    gemv_silu: bool,

    /// Fuse the two residual adds onto their producing GEMV's store (§12.2i, #509).
    ///
    /// The producers are `out_proj`, `conv.out_proj` and `w2` — 60 dispatches
    /// per token (§12.2e). **Needs no compile flag since #516**; it used to
    /// require `-DLLOOM_509_RESIDUAL_EPILOGUE` through `LLOOM_434_METAL_CFLAGS`
    /// and failed at pipeline load without it.
    ///
    /// Off by default (§7.1a), and see `--gemv-silu` on why an unmoved token
    /// digest does not prove this arm inert.
    #[arg(long, default_value_t = false)]
    gemv_residual: bool,

    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// `conv1d` dispatch (§12.2 row 1, `ConvFused`, #668) — the crown's third
    /// fusion, selectable here since #677 so this harness can run the
    /// configuration `lfm2-chat` calls the crown. Moves the logits digest (a
    /// serial f32 accumulation against a `simd_sum` tree) and not the tokens;
    /// off by default (§7.1a).
    #[arg(long, default_value_t = false)]
    conv_fused: bool,
}

/// Parse LFM2.5-VL-3B's `config.json`.
///
/// Byte-identical to `lfm2-determinism`'s (verified by SHA), and #287 counts it
/// as one of the 102 lines duplicated across four harnesses. Copied rather than
/// shared because §14.5's migration order puts the shared module at step 2 and
/// the release build at step 5; doing step 2 here would make this PR a refactor
/// of four harnesses plus a measurement, and a moved digest would then have two
/// possible causes instead of none.
fn parse_config(raw: &str) -> Result<Config> {
    let root: Value = serde_json::from_str(raw).context("parsing config.json")?;
    let text = root.get("text_config").unwrap_or(&root);
    let mut obj = text
        .as_object()
        .context("expected a JSON object for the model config")?
        .clone();

    if !obj.contains_key("rope_theta") {
        if let Some(theta) = obj
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .cloned()
        {
            obj.insert("rope_theta".into(), theta);
        }
    }

    if !obj.contains_key("tie_embedding") {
        if let Some(v) = obj.get("tie_word_embeddings").cloned() {
            obj.insert("tie_embedding".into(), v);
        }
    }

    for key in ["bos_token_id", "eos_token_id"] {
        if !obj.contains_key(key) {
            if let Some(v) = root.get(key).cloned() {
                obj.insert(key.into(), v);
            }
        }
    }

    let normalized = Value::Object(obj);
    let base: Lfm2Config = serde_json::from_value(normalized.clone())
        .context("config.json does not match candle's LFM2 config schema")?;
    let mut config = base.into_config(false);

    if let Some(stated) = normalized
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    {
        config.intermediate_size = stated;
    }

    Ok(config)
}

/// Tensor names, read from the safetensors header without mapping the weights.
fn tensor_names(path: &Path) -> Result<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    anyhow::ensure!(
        header_len > 0 && header_len < 100 * 1024 * 1024,
        "implausible safetensors header length {header_len}"
    );

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let parsed: Value = serde_json::from_slice(&header)?;
    Ok(parsed
        .as_object()
        .context("safetensors header is not a JSON object")?
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect())
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let raw = std::fs::read_to_string(&index)?;
    let parsed: Value = serde_json::from_str(&raw)?;
    let map = parsed
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("safetensors index has no weight_map")?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

/// Default HF cache location for the checkpoint this project targets.
fn default_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-3B/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// The follow-up turns, identical to `lfm2-determinism`'s, so a `--turns 3` run
/// here decodes the same sequence the canonical pair was taken over.
const FOLLOW_UPS: [&str; 8] = [
    "Expand on that, and give a concrete example.",
    "What are the main tradeoffs involved?",
    "How did this evolve historically?",
    "What do people most often get wrong about it?",
    "Compare that with the main alternative approach.",
    "What would you measure to tell if it is working well?",
    "Summarize the key points so far.",
    "What is the next thing someone should learn about this?",
];

/// An on/off axis arm. `on`/`off` rather than `true`/`false`, matching the
/// profiler's rendering so the two harnesses' `config=[…]` fields compare
/// directly — `.bench/configurations.md` §1 declares the arms in this spelling
/// and `lloom_runs::arms` checks a rendered line against it.
fn on_off(v: bool) -> String {
    if v { "on" } else { "off" }.to_string()
}

/// The `MetalCflags` arm: the compile flags as the compiler resolved them.
///
/// `none` rather than the empty string, for the profiler's reason: an arm that
/// renders empty is indistinguishable from a field the harness failed to fill
/// in, which is §7.1a-i's confusion. Rendered **verbatim** — two different `-D`s
/// are two different metallibs and must not compare equal.
///
/// Read from the crate's own `OnceLock` rather than from an environment
/// variable, so it reports what the compiler received. Note that release mode
/// refuses every `CANDLE_METAL_*` variable but `LLOOM_434_METAL_CFLAGS` is not
/// one of them, so this is genuinely selectable here.
fn metal_cflags_arm() -> String {
    #[cfg(feature = "metal")]
    {
        candle::kernels::metal_compile_flags()
            .map(|f| f.to_string())
            .unwrap_or_else(|| "none".to_string())
    }
    #[cfg(not(feature = "metal"))]
    {
        // No Metal compiler in this build, so there is no metallib for a flag to
        // reach. Distinct from `none`, which means "a Metal build that set no
        // flags".
        "n/a".to_string()
    }
}

/// The `HazardKey` arm the process installed, from the resolved value.
fn hazard_key_arm() -> &'static str {
    #[cfg(feature = "metal")]
    {
        match candle::metal_backend::hazard_key() {
            candle::metal_backend::HazardKey::Pointer => "Pointer",
            candle::metal_backend::HazardKey::Range => "Range",
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        "n/a"
    }
}

/// The `MergedUnit` arm (#477): whether the sixteen non-MLX sources were
/// compiled as one translation unit.
fn merged_unit_arm() -> &'static str {
    #[cfg(feature = "metal")]
    {
        if candle::kernels::merged_translation_unit() {
            "on"
        } else {
            "off"
        }
    }
    #[cfg(not(feature = "metal"))]
    {
        "n/a"
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.self_test {
        sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
        println!("sha256 self-test: OK (NIST vectors + 1e6*'a' + streaming)");
        return Ok(());
    }

    // **The release-mode precondition, checked rather than assumed.** §14.5 says
    // "no CANDLE_METAL_* set". A variable left in the environment by a previous
    // command would silently select a different arm and this run would report a
    // release number for a configuration no user runs -- #69's vacuous run in a
    // new quantity (§2.4). Refused rather than warned: the whole value of this
    // binary is that its configuration is the shipping one.
    let leaked: Vec<String> = std::env::vars()
        .map(|(k, _)| k)
        .filter(|k| k.starts_with("CANDLE_METAL_"))
        .collect();
    anyhow::ensure!(
        leaked.is_empty(),
        "release mode requires no CANDLE_METAL_* variable set (DESIGN.md §14.5); found: {}",
        leaked.join(", ")
    );

    // The three compiled-out tiers, asserted from `cfg!` in this crate rather
    // than from the build command, so the binary states what it *is*. #171/#205
    // establish the structural engagement proof (0 `run_telemetry` symbols with
    // the feature off, 13 with it on); this is the cheap precondition for it.
    anyhow::ensure!(
        !cfg!(feature = "metal-run-telemetry"),
        "release mode requires run-telemetry OFF (DESIGN.md §14.5)"
    );
    anyhow::ensure!(
        !cfg!(feature = "metal-hazard-audit"),
        "release mode requires hazard-audit OFF (DESIGN.md §14.5)"
    );
    anyhow::ensure!(
        !cfg!(feature = "metal-debug-labels"),
        "release mode requires debug-labels OFF (DESIGN.md §14.5)"
    );

    let model_dir = args.model_dir.clone().or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    // **The completion arm is parsed and installed BEFORE the device exists**
    // (`DESIGN.md` §11.5o, issue #559). `Commands::new` reads the arm once at
    // construction through a `OnceLock`, for the reason every other per-token
    // setting is read once (§2.4: a per-call `var()` on the hot path is what
    // #69's vacuous determinism run risks) -- so setting it afterwards would
    // silently leave the incumbent installed while the `RESULT` line reported
    // otherwise. That is §7.1a-i's wrong-arm shape, and it is the failure this
    // ordering exists to prevent.
    //
    // Parsed first, so an unrecognised value fails here rather than panicking
    // inside the device constructor: the error names the flag the operator
    // typed. Parsing is under `cfg(metal)` because the type lives behind the
    // feature -- §14.1 makes not breaking the non-Metal build a hard constraint
    // -- and a CPU run has no command buffer to wait for, so the flag is inert
    // rather than merely unavailable.
    #[cfg(feature = "metal")]
    let completion_wait =
        candle::metal_backend::CompletionWait::from_str_arm(&args.completion_wait)
            .map_err(anyhow::Error::msg)?;
    // SAFETY: single-threaded, before any device or thread exists. `set_var` is
    // unsafe from Rust 2024 because a concurrent `getenv` is UB; there is no
    // other thread here, and the value is read once immediately below.
    #[cfg(feature = "metal")]
    unsafe {
        std::env::set_var("CANDLE_METAL_COMPLETION_WAIT", &args.completion_wait)
    };

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // **Read back from the RESOLVED value, and refuse if it disagrees.**
    // §7.1a-i: `AttnImpl` was rendered by a two-armed conditional over a
    // three-armed axis and `--attn flash` reported `AttnImpl=Generic` on every
    // `RESULT` line since #116. A wrong arm compares equal to every run of that
    // arm; an absent one compares equal to nothing. So the check is that what
    // the device installed is what was asked for, not that the string parsed.
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        anyhow::ensure!(
            m.completion_wait_name() == completion_wait.name(),
            "--completion-wait {} resolved to {} inside the device: the arm reported \
             would not be the arm executed (DESIGN.md §7.1a-i)",
            completion_wait.name(),
            m.completion_wait_name()
        );
        // An arm that needs an event and did not get one runs the blocking
        // fallback -- correct, but not what was asked for, and a null measured
        // on it would be attributed to the wrong mechanism (§2.4).
        anyhow::ensure!(
            !completion_wait.needs_event() || m.completion_event_present(),
            "--completion-wait {} needs an MTLSharedEvent and the device declined to \
             create one, so every wait would take the blocking fallback",
            completion_wait.name()
        );
    }

    // f16 on Metal is what ambrogio loads, so it is the path worth measuring.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => match &device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        },
    };

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_raw)?;

    // Refused rather than defaulted, as in the other harnesses: a silently
    // ignored axis makes a gate report on an arm it did not run (§2.4).
    config.attn_impl = match args.attn.as_str() {
        "generic" => AttnImpl::Generic,
        "sdpa" => AttnImpl::Sdpa,
        "flash" => AttnImpl::FlashDecoding,
        other => anyhow::bail!("--attn must be `generic`, `sdpa` or `flash`, got `{other}`"),
    };
    config.kv_append = match args.kv_append.as_str() {
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        "paged" => KvAppend::Paged,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    // The two GEMV epilogue fusions (§12.2i, #509's wiring), brought onto this
    // harness by #567's second half so it selects the same four mechanisms the
    // profiler does. Construction-tier: read by `Mlp::new` at model construction
    // (`lfm2.rs:1438-1439`), so they must be set before `Model::new` below.
    //
    // Assigned from `args` and rendered on the `RESULT` line from `config`
    // rather than from the flag (§7.1a-i): the field the model reads is the arm
    // that ran, and a flag echoed back to itself is the two-parses-of-one-value
    // shape that made `--attn flash` report `AttnImpl=Generic` for the life of
    // #116.
    config.gemv_silu = args.gemv_silu;
    config.gemv_residual = args.gemv_residual;
    // Read by `ShortConv::new`, like the two above; rendered from `config`.
    config.conv_fused = args.conv_fused;
    // The conv writer's left context across a turn boundary (#677). Read by
    // `ShortConv::new`, so it must be set before `Model::new` below, and
    // rendered from `config` on the `RESULT` line (§7.1a-i).
    config.conv_prefill_ctx = args.conv_prefill_ctx;

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::new(&config, vb).context("constructing LFM2 model")?;
    let cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else if args.gumbel {
        // Plain temperature via gumbel-max, no nucleus. This is the arm that
        // stays on the device (§11.2c discharged), so the token-ring works at
        // t > 0 with `to_cpu_blit=0`.
        Sampling::All {
            temperature: args.temperature,
        }
    } else {
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    // Whether the ring's device-resident sampler covers this arm (checked
    // before `sampling` is moved into the processor).
    let ring_sampler_is_device = matches!(sampling, Sampling::ArgMax | Sampling::All { .. });
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);
    // Gumbel-max samples from the DEVICE rng (`rand_like`), not the processor's
    // host `StdRng`, so `--seed` reaches it only through `Device::set_seed`.
    // Without this, `--gumbel` runs are reproducible-by-accident from the RNG's
    // default state and ignore `--seed` (the §11.2c seed-reach gap, on the
    // harness side).
    device.set_seed(args.seed)?;

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

    // The poll arm, parsed before the window and refused rather than defaulted
    // (§2.4, after #69's vacuous run: an invalid setting must fail loudly,
    // because a silent fallback produces a run that reports one arm and
    // executes another).
    // Unset takes the type's default rather than a string spelled here (#578).
    let poll = match &args.poll {
        Some(s) => PollMechanism::from_str(s).map_err(anyhow::Error::msg)?,
        None => PollMechanism::default(),
    };

    // **The ring needs a DEVICE-RESIDENT sampler, and this refuses the arms
    // that round-trip rather than silently running the host path.** The ring is
    // built on `sample_argmax_on_device` leaving the token on the device; that
    // now holds for t=0 (`ArgMax`) and plain temperature via gumbel-max (`All`,
    // §11.2c discharged) but NOT top-{k,p}, which need a host `sort_by` (§11.2c:
    // 74 % of that arm). So `--token-ring` at t > 0 requires `--gumbel` (the
    // `All` arm); a `--token-ring --temperature 0.7` without `--gumbel` would
    // build `TopP`, take the host path, and report the ring arm -- #7.1a-i's
    // wrong-arm shape, which compares equal to every real run of that arm.
    anyhow::ensure!(
        !args.token_ring || ring_sampler_is_device,
        "--token-ring needs a device-resident sampler: t=0 (argmax) or t>0 with --gumbel \
         (plain-temperature gumbel-max). top-p/top-k take a host sort (§11.2c) and cannot \
         ride the ring; drop --top-p by passing --gumbel, or drop --token-ring."
    );
    // The non-blocking EOS read is a property OF the ring arm, so asking for it
    // without the ring is a contradiction rather than a no-op. Refused for the
    // same reason as above: a run that reports `nonblocking_eos=true` while
    // executing the baseline's blocking sampler is a wrong arm, and a wrong arm
    // compares equal to every real run of that arm (§7.1a-i).
    anyhow::ensure!(
        !args.nonblocking_eos || args.token_ring,
        "--nonblocking-eos requires --token-ring (DESIGN.md §11.5, #535): the flag it \
         polls is published by the ring, so without it there is nothing to poll"
    );
    anyhow::ensure!(
        !args.host_drain || args.token_ring,
        "--host-drain requires --token-ring (#677): it learns EOS from the ring's drained \
         ids, so without the ring there is nothing to drain"
    );
    anyhow::ensure!(
        !(args.host_drain && args.nonblocking_eos),
        "--host-drain and --nonblocking-eos are two EOS-learning arms of one axis (EosRead); \
         pick one"
    );
    // B=1 is what this harness runs and what the slot is sized for; §6.6c
    // records the readback as `B` round trips per step at B>1, which makes the
    // slot `B` slots and is #263's change. Stated so the scope is in the
    // binary rather than only in the write-up.

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let prompt_ids = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    // ---- the readback strategy and the loop's knobs, before the window ---
    //
    // The loop is `Session::decode` (`design/docs/decode-driver-design.md`,
    // stage 1) and this harness only SELECTS: which readback arm, whether the
    // host encodes ahead of the EOS it has not yet learned, and how often the
    // in-flight queue is pruned. The `RESULT` line renders the arms from the
    // strategy that ran and the counters from the loop that ran them.
    let budget = CommitBudget::from_str(&args.commit_budget).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        args.lookahead <= 1,
        "--lookahead must be 0 or 1: the ring holds one input slot and the overrun past an \
         EOS is bounded at one forward only for a lookahead of at most one"
    );
    anyhow::ensure!(
        args.lookahead == 0 || args.token_ring,
        "--lookahead 1 needs --token-ring: the blocking arm learns EOS inside its readback, \
         so there is nothing to encode ahead of"
    );
    // A multi-turn run on an arm that learns EOS late leaves the cache past
    // the EOS, and the next turn prefills onto it unless the overrun can be
    // rewound at the turn boundary (#677, `Session::reconcile`). Rewindable
    // means `KvAppend::InPlace` (a length to decrement) with
    // `ConvState::RotatingRing { k }` (history the phase can return to), `k` at
    // least the overrun an arm can produce: 1 under `--lookahead 1`; under
    // `--host-drain` a commit period PLUS the in-flight cap, because the drain
    // that shows the EOS reads only what the GPU has published — a period's
    // worth of forwards may be enqueued past the EOS before the commit, and the
    // cap's worth may still be unpublished at that commit (measured: `steps:8`
    // with cap 8 overran a `rotating:8` ring, `steps:4` with cap 4 did not).
    // With no cap the host's lead is unbounded, so the drain arm is refused
    // for a multi-turn run without one. The crown's `Cat` × `Shuffle` cannot
    // rewind at all, so it is refused with `--turns` above 1 rather than
    // moving the digest.
    // The drain arm learns EOS at a commit, from the published slots. The
    // completion marker goes live every `WAKE_LIVE_EVERY_DEFAULT` commits and a
    // drain reads the marker the GPU has executed — at worst the previous live
    // one — so up to two live windows of forwards may be enqueued past the EOS
    // before a drain shows it, and the pacer's depth may still be unpublished
    // at it.
    let overrun_bound = if args.host_drain {
        2 * budget.period() * WAKE_LIVE_EVERY_DEFAULT + IN_FLIGHT_STEPS
    } else if args.lookahead > 0 {
        1
    } else {
        0
    };
    let rewindable_to = match (config.kv_append, config.conv_state) {
        (KvAppend::InPlace | KvAppend::Paged, ConvState::RotatingRing { k }) => k,
        _ => 0,
    };
    anyhow::ensure!(
        overrun_bound == 0 || args.turns.max(1) == 1 || rewindable_to >= overrun_bound,
        "--lookahead {}{} with --turns {}: this arm can leave the cache up to {} positions \
         past an EOS, and the next turn would prefill onto them; the cache must be able to \
         rewind that far (--kv-append in-place --conv-state rotating:K with K >= {}; this run \
         has {:?} x {:?}, DESIGN.md §10.2a). Use --turns 1, or a rewindable cache.",
        args.lookahead,
        if args.host_drain { " --host-drain" } else { "" },
        args.turns,
        overrun_bound,
        overrun_bound,
        config.kv_append,
        config.conv_state,
    );
    let opts = DecodeOptions {
        max_tokens: args.n,
        budget,
        lookahead: args.lookahead,
        sample_split: args.sample_split,
        ..DecodeOptions::new(args.n)
    };
    // The ring's device-side history is sized for the whole run — every turn,
    // plus the overrun an arm may publish past each turn's EOS — and refuses
    // to overflow rather than silently dropping an id.
    let capacity = args.n + args.turns * (overrun_bound + 2) + 1;

    // ---- the timed window --------------------------------------------------
    //
    // One clock over the whole loop, inside `run_turns`. It opens before the
    // first prefill and closes after the last token is produced, so it contains
    // every prefill, every forward pass, every sample and every EOS test --
    // which is what a user waits for. The ring arms learn their tokens once,
    // after each turn's loop, in a single readback of the device-side history;
    // nothing else inside the window reads anything back except what the
    // decode itself requires.
    let run = if !args.token_ring {
        let rb = Blocking::new(&eos_ids, &device)?;
        run_turns(
            &args,
            &model,
            cache,
            rb,
            &mut logits_processor,
            &tokenizer,
            &prompt_ids,
            &opts,
        )?
    } else if args.host_drain {
        let rb = Ring::new(&eos_ids, &device, capacity, EosRead::HostOnDrain)?;
        run_turns(
            &args,
            &model,
            cache,
            rb,
            &mut logits_processor,
            &tokenizer,
            &prompt_ids,
            &opts,
        )?
    } else if args.nonblocking_eos {
        let rb = Ring::new(&eos_ids, &device, capacity, EosRead::Poll(poll))?;
        run_turns(
            &args,
            &model,
            cache,
            rb,
            &mut logits_processor,
            &tokenizer,
            &prompt_ids,
            &opts,
        )?
    } else {
        let rb = Ring::new(&eos_ids, &device, capacity, EosRead::Blocking)?;
        run_turns(
            &args,
            &model,
            cache,
            rb,
            &mut logits_processor,
            &tokenizer,
            &prompt_ids,
            &opts,
        )?
    };
    let Run {
        tokens,
        step_ns,
        sample_ns,
        kv_len,
        hit_eos,
        turns_run,
        prefill_submit_ns,
        elapsed,
        commits,
        max_in_flight,
        overrun,
        rewound,
        step_markers,
        step_waits,
        arms,
        eos_read,
        counters,
    } = run;
    // **Read back from the strategy that ran, not from the flags.** A wrong
    // arm compares equal to every run of that arm (§7.1a-i); the strategy is
    // where the arm actually is. Both non-blocking EOS arms render
    // `nonblocking_eos=true`; `eos_read` is what tells them apart.
    let want_eos_read = if !args.token_ring {
        "host"
    } else if args.host_drain {
        "drain"
    } else if args.nonblocking_eos {
        "poll"
    } else {
        "flag"
    };
    anyhow::ensure!(
        arms.token_ring == args.token_ring
            && arms.nonblocking_eos == (args.nonblocking_eos || args.host_drain)
            && eos_read == want_eos_read,
        "the strategy that ran renders token_ring={} nonblocking_eos={} eos_read={} against \
         the flags {}/{}/{}: the arm reported would not be the arm executed (DESIGN.md §7.1a-i)",
        arms.token_ring,
        arms.nonblocking_eos,
        eos_read,
        args.token_ring,
        args.nonblocking_eos || args.host_drain,
        want_eos_read
    );
    let ReadbackCounters {
        ring_publishes,
        ring_eos_publishes,
        poll_iters,
        poll_fallbacks,
        ring_discarded,
        ..
    } = counters;
    // ---- the window is closed; everything below is after the clock ----------

    // Every publish is either a token in the digest, an EOS that ended a
    // turn, or — on the host-on-drain arm — an overrun the loop discarded.
    // **This reconciles the engagement count against the digest**, which
    // is what makes `ring_publishes` evidence rather than decoration: a run
    // whose counter and token stream disagree has one of the two wrong, and
    // #500's whole finding is that a green board says nothing when nothing is
    // counted. (An EOS counted per turn rather than inferred from a bool: a
    // `--turns 3` run can publish three.)
    if args.token_ring {
        anyhow::ensure!(
            ring_publishes == tokens.len() as u64 + ring_eos_publishes + ring_discarded,
            "published {} tokens but accounted for {} + {} EOS + {} discarded: the \
             engagement count and the digest disagree",
            ring_publishes,
            tokens.len(),
            ring_eos_publishes,
            ring_discarded
        );
        anyhow::ensure!(
            ring_publishes > 0,
            "--token-ring ran and published nothing: the ring arm is not on the path"
        );
    }

    // The digest, taken AFTER the window closes. This is §14.5's load-bearing
    // clause: it is what makes the run verifiable without instrumenting the thing
    // being measured.
    let mut token_hasher = sha256::Sha256::new();
    for t in &tokens {
        token_hasher.update(&t.to_le_bytes());
    }
    let token_digest = sha256::hex(&token_hasher.finalize());

    let generated = tokens.len();
    anyhow::ensure!(generated > 0, "no tokens generated");

    // Steady state excludes `--warmup` steps. Reported beside the end-to-end
    // figure rather than instead of it: the end-to-end number is what a user
    // gets, and the steady-state one is what compares to §11.5a's per-step terms.
    let steady: Vec<u64> = step_ns.iter().copied().skip(args.warmup).collect();
    anyhow::ensure!(
        !steady.is_empty(),
        "--warmup {} left no steps of {}",
        args.warmup,
        step_ns.len()
    );
    let steady_sum: u64 = steady.iter().sum();
    let steady_mean_ms = steady_sum as f64 / steady.len() as f64 / 1e6;
    let mut sorted = steady.clone();
    sorted.sort_unstable();
    let med_ms = sorted[sorted.len() / 2] as f64 / 1e6;
    let min_ms = sorted[0] as f64 / 1e6;
    let max_ms = sorted[sorted.len() - 1] as f64 / 1e6;
    let p5_ms = sorted[sorted.len() * 5 / 100] as f64 / 1e6;
    let sd_ms = {
        let m = steady_mean_ms;
        (steady
            .iter()
            .map(|&v| (v as f64 / 1e6 - m).powi(2))
            .sum::<f64>()
            / steady.len() as f64)
            .sqrt()
    };

    let elapsed_s = elapsed.as_secs_f64();
    // **The publishable number**: the whole loop, including every prefill and
    // every sample, divided by the tokens a user received.
    let end_to_end_tok_s = generated as f64 / elapsed_s;
    // The steady-state decode rate, prefill and warmup excluded, sampling still
    // inside. This is the figure comparable to §11.5a's `wall + sample + eos`.
    let steady_tok_s = 1e3 / steady_mean_ms;

    // ---- §11.2b's denominator, on this harness (#567's reopening) ----------
    //
    // **`step_ms_per_token` is `steady_mean_ms` here, and the equality is
    // structural rather than a coincidence to be checked.** §11.2b defines a
    // step as `wall + sample + eos`, and the three are summed on
    // `lfm2-decode-profile` because that harness's windows are **disjoint**: it
    // samples at `main.rs:3080-3277` and opens its step clock afterwards at
    // `:3395`, so its `wall` excludes the sample readback §11.5a prices at
    // 441 µs/token, and `step_ms_per_token` has to add the terms back
    // (`main.rs:4558`).
    //
    // **This harness has one `Instant` and it opens BEFORE the sample call**
    // (`step_start`, above the `--sample-split` drain), so `step_ns` already
    // contains the forward pass, `sample()` and the EOS test. The three terms
    // are not disjoint here; they are nested. **Adding them again would be
    // #299's double count** — subtracting or adding a term to a denominator
    // that already contains it — which is the exact error §2.5 exists to
    // prevent and the one §11.2b was written to end.
    //
    // So the field is emitted as the identity rather than as a sum, and the
    // comparability is the point: the same quantity, `wall + sample + eos`,
    // reached two ways because the two harnesses bracket it differently. What
    // makes the pair readable is that neither had to change what `wall` means
    // (§2.6: adding an axis is not moving one) — the profiler's `wall` still
    // excludes the sample and this one's still includes it.
    //
    // **The mean rather than the median, deliberately, and it is the weaker
    // statistic.** §6.6b measures this distribution with a floor and a
    // right tail, mean 22 % above median, and `CONTRIBUTING.md` §3.2a's rule is
    // to prefer the median where `mean_over_median` is far from 1.0. The mean
    // is used here because the profiler's field is a mean of `wall` plus a mean
    // of `sample` (`main.rs:3744`, `:3812`) and a comparison between two
    // harnesses' denominators must be between the same statistic. `step_med_ms`
    // is on the line beside it for the reader who wants the location estimate,
    // and it is unchanged.
    let step_ms_per_token = steady_mean_ms;

    println!("model dir             {}", model_dir.display());
    println!("device                {:?}  dtype {:?}", device, dtype);
    println!(
        "config                attn={:?} kv_append={:?} conv_state={:?}",
        config.attn_impl, config.kv_append, config.conv_state
    );
    println!(
        "sampling              seed={} temp={} top_p={}",
        args.seed, args.temperature, args.top_p
    );
    println!();
    println!("generated             {generated} tokens over {turns_run} turn(s)");
    println!("final kv_len          {kv_len}");
    println!("hit_eos               {hit_eos}");
    println!("elapsed (whole loop)  {:.4} s", elapsed_s);
    println!(
        "prefill (submit)      {:.4} ms   (submission, not execution -- see the source note)",
        prefill_submit_ns as f64 / 1e6
    );
    println!();
    println!(
        "END-TO-END            {:.4} tok/s   ({} tokens / {:.4} s, everything included)",
        end_to_end_tok_s, generated, elapsed_s
    );
    println!(
        "steady decode         {:.4} tok/s   ({:.4} ms/step, warmup {} excluded)",
        steady_tok_s, steady_mean_ms, args.warmup
    );
    println!(
        "  per-step ms         sd {:.4}  p5 {:.4}  min {:.4}  med {:.4}  max {:.4}  n {}",
        sd_ms,
        p5_ms,
        min_ms,
        med_ms,
        max_ms,
        steady.len()
    );
    if args.sample_split {
        let s: Vec<u64> = sample_ns.iter().copied().skip(args.warmup).collect();
        if !s.is_empty() {
            let mut sorted_s = s.clone();
            sorted_s.sort_unstable();
            let mean = s.iter().sum::<u64>() as f64 / s.len() as f64 / 1e6;
            let med = sorted_s[sorted_s.len() / 2] as f64 / 1e6;
            println!();
            println!(
                "sample split          mean {:.4} ms  med {:.4}  min {:.4}  max {:.4}  n {}",
                mean,
                med,
                sorted_s[0] as f64 / 1e6,
                sorted_s[sorted_s.len() - 1] as f64 / 1e6,
                s.len()
            );
            println!(
                "  share of step       {:.2} % of {:.4} ms   (DIAGNOSTIC: this run's total is \
                 inflated by the split and must not be compared to a run without it, \
                 DESIGN.md §11.2a)",
                100.0 * mean / steady_mean_ms,
                steady_mean_ms
            );
        }
    }

    println!();
    println!("tokens sha256         {token_digest}");

    // **The engagement counters, and they are the reason a null on this axis is
    // readable at all.** §11.5c's first sweep measured the blit against itself
    // and read `unresolved` on every dimension -- it was caught by
    // `to_cpu_fast=0` on both arms and by nothing else. #500 is the same shape
    // one level out: every gate was green because the mechanism had no callers,
    // and a run reporting `ring_publishes=0` under `--token-ring` is that state
    // made visible instead of being discovered by a census.
    //
    // `to_cpu_fast`/`to_cpu_blit` are reported **per arm** because #454
    // measures which arm fires as unstable run to run -- blit 12/1/2/5 across
    // four consecutive runs of one binary, 3..26 in a committed 12-run arm --
    // which is the same order as the effect. **If the split differs between
    // arms the comparison moved the allocator, not the ring**, and the result
    // is void regardless of sign.
    // `cfg`'d for §14.1: the counters live in `metal_backend`, which is behind
    // the `metal` feature. A CPU build reports zeros, which is honest -- there
    // is no `to_cpu` arm to choose on that backend.
    #[cfg(feature = "metal")]
    let (fast, blit) = candle::metal_backend::to_cpu_counts();
    #[cfg(not(feature = "metal"))]
    let (fast, blit) = (0u64, 0u64);
    #[cfg(feature = "metal")]
    let null_after_wait = candle::metal_backend::to_cpu_null_after_wait();
    #[cfg(not(feature = "metal"))]
    let null_after_wait = 0u64;
    // #535's peek counters. `peek_unmapped` is the fail-closed arm: a nonzero
    // value says the allocator handed back a buffer Metal did not map, which
    // §15.2 #12a makes a SIZE property and #454 measures as unstable run to
    // run. Reported beside `poll_fallbacks` so a reader can tell "the peek
    // declined" from "the peek never ran".
    #[cfg(feature = "metal")]
    let (peek_hit, peek_unmapped) = candle::metal_backend::peek_counts();
    #[cfg(not(feature = "metal"))]
    let (peek_hit, peek_unmapped) = (0u64, 0u64);
    // See the `peek arms` line below: #578's fast path moves neither counter
    // above, so this is the one that says the poll ran.
    let poll_reads = lloom::generation::token_ring::poll_read_count();

    // #559's completion arm, **read back from the device rather than from the
    // flag** (§7.1a-i), and its engagement pair beside it. `(0, 0)` under an
    // async arm means the arm did not run: #500 measured that a mechanism with
    // no production callers produces a clean null indistinguishable from a real
    // one, and the counter is what separates them.
    //
    // A different height from #535's counters above rather than a duplicate of
    // them: `peek_hit` counts reads that skipped a wait, this counts waits that
    // were satisfied without blocking.
    // `cw_fires` is the listener arm's own engagement proof (#583), and the
    // pair above cannot be it: `shared-event-listener` reports
    // `(signals > 0, polls == 0)` whether its notification block delivered or
    // never fired and the wait fell back to blocking. The fallback is correct,
    // so the values agree and both digests are unmoved either way.
    #[cfg(feature = "metal")]
    let (cw_name, cw_signals, cw_polls, cw_fires) = match &device {
        Device::Metal(m) => {
            let (s, p) = m.completion_counts();
            (m.completion_wait_name(), s, p, m.completion_fires())
        }
        _ => ("none", 0, 0, 0),
    };
    #[cfg(not(feature = "metal"))]
    let (cw_name, cw_signals, cw_polls, cw_fires) = ("none", 0u64, 0u64, 0u64);

    println!();
    println!(
        "engagement            token_ring={} ring_publishes={} poll={} poll_iters={} \
         max_in_flight={} nonblocking_eos={} poll_fallbacks={} eos_read={} ring_discarded={}",
        arms.token_ring,
        ring_publishes,
        poll.name(),
        poll_iters,
        max_in_flight,
        arms.nonblocking_eos,
        poll_fallbacks,
        eos_read,
        ring_discarded
    );
    println!(
        "  decode loop         lookahead={} commit_budget={} commits={} overrun={} rewound={} \
         in_flight_steps={} step_markers={} step_waits={} conv_prefill_ctx={} \
         (Session::decode, DESIGN.md decode-driver stage 1; #677 sequence layer)",
        args.lookahead,
        budget.arm(),
        commits,
        overrun,
        rewound,
        IN_FLIGHT_STEPS,
        step_markers,
        step_waits,
        config.conv_prefill_ctx
    );
    println!(
        "  completion arm      completion_wait={cw_name} completion_signals={cw_signals} \
         completion_polls={cw_polls} completion_fires={cw_fires}"
    );
    println!(
        "  readback arms       to_cpu_fast={fast} to_cpu_blit={blit} \
         to_cpu_null_after_wait={null_after_wait}"
    );
    // `poll_reads` beside the peek pair for #578's reason: the poll's fast path
    // reads through an address resolved once and so increments neither
    // `peek_hit` nor `peek_unmapped`. On a mapped run those two read zero for a
    // path that ran every iteration, which is indistinguishable from a path
    // that never ran (§3.1c) -- this is the counter that separates them.
    println!(
        "  peek arms           peek_hit={peek_hit} peek_unmapped={peek_unmapped} \
         poll_reads={poll_reads}"
    );

    // ---- the axis vector (#567's reopening) --------------------------------
    //
    // **This harness printed no `config=[…]` field at all**, which
    // `grep -c 'config=\['` read as **0** here against 7 on the profiler. That
    // is a wider hole than the missing denominator the reopening comment names,
    // and it is the same defect one level out: `lloom_runs::ingest`'s
    // `from_result_line` accepts **any** `RESULT` line and reads its axes from
    // `config=[…]` (`ingest.rs:97`), so every release run ingested to date is
    // `Unrecorded` on all 22 registry axes — and #171's store pools runs whose
    // axis vectors agree. Two release runs differing in `--attn`, in the ring
    // arm, or in an epilogue therefore rendered *equal* axis vectors.
    //
    // **That is §2.3.9's `MathMode` failure at the level of a harness rather
    // than of an axis**: not an axis missing from a line, but the line missing
    // from a binary. The axes were all registered — #567's first half put the
    // four async arms in `axes::REGISTRY`, `.bench/configurations.md` §1 and
    // `axis_pairs()` — and this harness simply did not render them.
    //
    // Rendered from the **resolved** values (§7.1a-i) rather than from the flag
    // strings: `config.attn_impl` is what the model was built with, and
    // `cw_name` comes from the device. The axes this harness cannot select are
    // stated at their pinned defaults rather than omitted, for the profiler's
    // reason — *inert* and *unrecorded* are different facts, and only the
    // second should stop a merge.
    let conv_state_arm = match config.conv_state {
        ConvState::Shuffle => "Shuffle".to_string(),
        ConvState::SlidingRing { k, slack } => format!("SlidingRing(k={k},slack={slack})"),
        ConvState::RotatingRing { k } => format!("RotatingRing(k={k})"),
    };
    // **The sleeping arm carries its PERIOD**, which `name()` drops: it maps
    // every `SharedEventSleep { micros }` to bare `shared-event-sleep`, so `:50`
    // and `:200` would render equal while §11.5o measures them at +0.039 against
    // +0.121 ms of wake latency. #572 repaired this on the profiler and in
    // `lloom-runs`; rendering `cw_name` here would have re-opened it on the
    // third harness. §13.5m's `Counter(50)`/`Counter(55)` rule.
    //
    // Built from the parsed arm rather than from `cw_name` for the period, and
    // asserted equal to the device's own name above, so the widening cannot
    // drift from what actually ran.
    // **The period comes from the ARM, and this was the THIRD copy of that
    // list** (#583). It was a local `match` naming each parameterised arm with
    // an `other => name()` fallback — so a newly added parameterised arm does
    // not fail to compile here, it silently renders **without its period** and
    // pools with every other period of itself in #171's store. That is §13.5m's
    // defect, and the reason it kept recurring is that three harnesses each
    // kept their own copy of one list. `CompletionWait::arm_with_period()` is
    // now the single place that knows, and its `match` is exhaustive, so a new
    // arm is a compile error *there* rather than a wrong axis vector *here*.
    #[cfg(feature = "metal")]
    let completion_wait_arm = completion_wait.arm_with_period();
    #[cfg(not(feature = "metal"))]
    let completion_wait_arm = "n/a".to_string();
    // `PollMechanism::name()` has the identical shape and the identical repair
    // (#572): `Sleep(d)` renders bare.
    let poll_arm = match poll {
        PollMechanism::Sleep(d) => format!("sleep:{}", d.as_micros()),
        other => other.name().to_string(),
    };
    #[cfg(feature = "metal")]
    let cut_policy_arm = match &device {
        Device::Metal(m) => {
            candle::metal_backend::CutPolicy::render(m.cut_policy(), m.compute_per_buffer())
        }
        _ => "n/a".to_string(),
    };
    #[cfg(not(feature = "metal"))]
    let cut_policy_arm = "n/a".to_string();
    let config_line = [
        // Not selectable here; nothing on the LFM2 path dispatches the packed
        // variants (§11.3k). Stated at its pinned default.
        ("ParamStyle", "Split".to_string()),
        ("Executor", "Classical".to_string()),
        // This harness never installs an arena: it has no `--arena` flag, so
        // every allocation comes from the pool. `none(pool)` is the profiler's
        // own rendering for that state.
        ("ArenaLayout", "none(pool)".to_string()),
        ("ArenaOffsets", "Cpu".to_string()),
        // **Read back from the resolved value** (§7.1a-i), as the profiler
        // does. This rendered the literal `Resource` — an arm `HazardKey` does
        // not have (`Pointer` | `Range`) — on every release line since #567,
        // and nothing noticed until a release line became a registry fixture
        // (#673): a wrong arm compares equal to every run of that arm, and an
        // unknown one to nothing.
        ("HazardKey", hazard_key_arm().to_string()),
        ("AttnImpl", format!("{:?}", config.attn_impl)),
        ("KvAppend", format!("{:?}", config.kv_append)),
        ("ConvState", conv_state_arm),
        ("BarrierScope", "RunStart".to_string()),
        ("ScratchSizing", "Grow".to_string()),
        // Release mode forbids `CANDLE_METAL_*` entirely (asserted at the top of
        // `main`), and `MathMode` is selected by one of those variables, so this
        // binary can only ever run the default arm. `Fast` is candle's default
        // (§2.3.9, `kernel.rs:203`) and is the arm every canonical digest
        // belongs to.
        ("MathMode", "Fast".to_string()),
        ("ReplayBarriers", "Always".to_string()),
        ("MemoryBudget", "None".to_string()),
        ("ScratchLayout", "Planes".to_string()),
        // Only reachable under `AttnImpl=FlashDecoding`; this harness has no
        // flag for either, so they render the model's defaults.
        ("FlashPageSize", config.flash_page_size.to_string()),
        ("FlashK", config.flash_pages_per_chunk.to_string()),
        // B=1 is what this harness runs (the ring's slot is sized for it, and
        // the source says so above).
        ("Batch", "1".to_string()),
        // Read back from the device: `CANDLE_METAL_*` is refused by this
        // binary, so this can only ever be the default arm, but the mechanism
        // that decides it is the one that renders it (§7.1a-i).
        ("CutPolicy", cut_policy_arm),
        ("GemvSilu", on_off(config.gemv_silu)),
        ("GemvResidual", on_off(config.gemv_residual)),
        // #668's axis, rendered here since #677 (no harness rendered it before,
        // which `lloom-runs`' renderability check reported as absent).
        ("ConvFused", on_off(config.conv_fused)),
        // #677: the conv writer's left context across a multi-token pass,
        // read back from `config` (`ShortConv::new` consumed it).
        ("ConvPrefillCtx", on_off(config.conv_prefill_ctx)),
        ("TokenRing", on_off(arms.token_ring)),
        ("Poll", poll_arm),
        ("NonblockingEos", on_off(arms.nonblocking_eos)),
        // #677: how the loop learned EOS, from the strategy that ran.
        ("EosRead", eos_read.to_string()),
        // The decode loop's two knobs (#673). Rendered from the resolved
        // values the loop ran under; `Lookahead` carries the count and
        // `CommitBudget` its period verbatim (§13.5m).
        ("Lookahead", args.lookahead.to_string()),
        ("CommitBudget", budget.arm()),
        ("CompletionWait", completion_wait_arm),
        ("MetalCflags", metal_cflags_arm()),
        ("MergedUnit", merged_unit_arm().to_string()),
    ]
    .iter()
    .map(|(k, v)| format!("{k}={v}"))
    .collect::<Vec<_>>()
    .join(" ");

    // One machine-readable line, in the shape the other harnesses print, so an
    // analysis script reads a line rather than parsing prose.
    println!(
        "RESULT mode=release n={} warmup={} turns={} generated={} attn={:?} kv_append={:?} \
         conv_state={:?} dtype={:?} seed={} temp={} top_p={} elapsed_s={:.6} \
         end_to_end_tok_s={:.4} steady_ms_per_step={:.6} steady_tok_s={:.4} \
         step_ms_per_token={:.4} \
         step_sd_ms={:.6} step_p5_ms={:.6} step_med_ms={:.6} step_max_ms={:.6} \
         prefill_submit_ms={:.4} final_kv_len={} hit_eos={} token_ring={} \
         ring_publishes={} poll={} poll_iters={} max_in_flight={} to_cpu_fast={} \
         to_cpu_blit={} to_cpu_null_after_wait={} nonblocking_eos={} \
         poll_fallbacks={} peek_hit={} peek_unmapped={} poll_reads={} completion_wait={} \
         completion_signals={} completion_polls={} completion_fires={} \
         gemv_silu={} gemv_residual={} conv_fused={} \
         lookahead={} commit_budget={} commits={} overrun={} rewound={} \
         in_flight_steps={} step_markers={} step_waits={} eos_read={} ring_discarded={} \
         conv_prefill_ctx={} config=[{}] token_digest={}",
        args.n,
        args.warmup,
        turns_run,
        generated,
        config.attn_impl,
        config.kv_append,
        config.conv_state,
        dtype,
        args.seed,
        args.temperature,
        args.top_p,
        elapsed_s,
        end_to_end_tok_s,
        steady_mean_ms,
        steady_tok_s,
        // §11.2b's denominator. Equal to `steady_ms_per_step` by construction on
        // this harness rather than by coincidence -- see the derivation above.
        step_ms_per_token,
        sd_ms,
        p5_ms,
        med_ms,
        max_ms,
        prefill_submit_ns as f64 / 1e6,
        kv_len,
        hit_eos,
        arms.token_ring,
        ring_publishes,
        poll.name(),
        poll_iters,
        max_in_flight,
        fast,
        blit,
        null_after_wait,
        arms.nonblocking_eos,
        poll_fallbacks,
        peek_hit,
        peek_unmapped,
        poll_reads,
        cw_name,
        cw_signals,
        cw_polls,
        cw_fires,
        // Read back from `config`, which is what `Mlp::new` consumed, rather
        // than from `args` (§7.1a-i).
        config.gemv_silu,
        config.gemv_residual,
        config.conv_fused,
        args.lookahead,
        budget.arm(),
        commits,
        overrun,
        rewound,
        IN_FLIGHT_STEPS,
        step_markers,
        step_waits,
        eos_read,
        ring_discarded,
        config.conv_prefill_ctx,
        config_line,
        token_digest
    );

    Ok(())
}

/// What one run produced, across every turn.
struct Run {
    tokens: Vec<u32>,
    /// Per-step wall, one entry per token, concatenated across turns.
    step_ns: Vec<u64>,
    /// Only under `--sample-split`.
    sample_ns: Vec<u64>,
    kv_len: usize,
    hit_eos: bool,
    turns_run: usize,
    prefill_submit_ns: u64,
    elapsed: std::time::Duration,
    commits: u64,
    max_in_flight: usize,
    /// Positions the cache ran past a turn's EOS, summed over turns.
    overrun: usize,
    /// Of those, the positions rewound at a turn boundary so the next turn
    /// prefilled onto the last kept token (#677). The last turn's overrun is
    /// left as it is — nothing follows it — so `rewound <= overrun`.
    rewound: usize,
    /// The pacer's engagement pair (`Session::decode`, `IN_FLIGHT_STEPS`):
    /// markers taken, one per forward, and the ones the host had to wait on.
    step_markers: u64,
    step_waits: u64,
    arms: ReadbackArms,
    eos_read: &'static str,
    counters: ReadbackCounters,
}

/// The conversational turns against one cache, driven through
/// `Session::decode` over the selected strategy.
///
/// Generic over the strategy so the loop monomorphizes per arm (§15.2 #9): the
/// three call sites above are the three instantiations, and nothing here names
/// an arm. Turn 0 is the prompt; every later turn prefills a follow-up onto the
/// same cache, exactly as `lfm2-determinism` does, so a `--turns 3` run decodes
/// the sequence the canonical §2.3.8c pairs were taken over.
///
/// **EOS ends a turn without being pushed or hashed**, which is the order the
/// canonical pairs were taken in; `Session::decode` returns the ids without
/// it. A turn that reaches the token budget ends the run.
#[allow(clippy::too_many_arguments)]
fn run_turns<R: ReadbackStrategy>(
    args: &Args,
    model: &Model,
    cache: Cache,
    mut rb: R,
    sampler: &mut LogitsProcessor,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    opts: &DecodeOptions,
) -> Result<Run> {
    // Reserved BEFORE the window opens. A `Vec` growth inside the timed region
    // would be an allocation the user does not pay for on a steady-state path
    // and this harness would be measuring its own bookkeeping (§6.4a).
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n + args.turns + 1);
    let mut step_ns: Vec<u64> = Vec::with_capacity(args.n + args.turns + 1);
    let mut sample_ns: Vec<u64> = Vec::with_capacity(args.n + args.turns + 1);
    let mut hit_eos = false;
    let mut turns_run = 0usize;
    let mut prefill_submit_ns: u64 = 0;
    let mut commits: u64 = 0;
    let mut max_in_flight = 0usize;
    let mut step_markers = 0u64;
    let mut step_waits = 0u64;
    let mut overrun = 0usize;
    let mut rewound = 0usize;
    // The cache starts empty: turn 0's prefill is what appends the prompt, and
    // the session's position advances by each turn's length after it.
    let mut session = Session::new(model, cache);
    let mut spec = IdentityProposer;

    let run_start = std::time::Instant::now();
    for turn in 0..args.turns.max(1) {
        let turn_ids: Vec<u32> = if turn == 0 {
            prompt_ids.to_vec()
        } else {
            let follow_up = FOLLOW_UPS[(turn - 1) % FOLLOW_UPS.len()];
            let text = format!("<|im_start|>user\n{follow_up}<|im_end|>\n<|im_start|>assistant\n");
            tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing follow-up: {e}"))?
                .get_ids()
                .to_vec()
        };
        // The budget left for this turn. A turn opened with none left still
        // prefills — the previous loop shape did, and `final_kv_len` and
        // `turns_run` are what would otherwise move.
        let turn_opts = DecodeOptions {
            max_tokens: args.n - tokens.len(),
            ..*opts
        };
        // No streaming: this is the perf harness, whose number is the whole
        // generation and whose digest is taken from the ids after the clock. A
        // per-batch drain would add readbacks the published figure must not carry.
        let out = session
            .decode(
                &mut spec,
                &mut rb,
                sampler,
                &turn_ids,
                &turn_opts,
                &mut |_| Ok(()),
            )
            .with_context(|| format!("decoding turn {turn}"))?;
        turns_run = turn + 1;
        // Accumulated across turns; prefill is a different shape of work (§6.6)
        // and is reported apart from the decode rate. **Submission time, not
        // execution time**: candle is asynchronous and `forward` returns once
        // the work is encoded (see the profiler's bracket for the other side).
        prefill_submit_ns += out.stats.prefill_ns;
        tokens.extend_from_slice(&out.ids);
        step_ns.extend_from_slice(&out.stats.step_ns);
        sample_ns.extend_from_slice(&out.stats.sample_ns);
        commits += out.stats.commits;
        max_in_flight = max_in_flight.max(out.stats.max_in_flight);
        step_markers += out.stats.step_markers;
        step_waits += out.stats.step_waits;
        overrun += out.stats.overrun;
        if out.eos {
            hit_eos = true;
            if turn + 1 < args.turns.max(1) {
                // The turn boundary (#677): an arm that learned EOS late left
                // the cache past it; rewind so the next turn prefills onto the
                // last kept token. The flag check in `main` guarantees the
                // cache can, so a refusal here is a defect and not a policy.
                rewound += session
                    .reconcile()
                    .with_context(|| format!("rewinding turn {turn}'s overrun"))?;
                continue;
            }
        }
        break;
    }
    let elapsed = run_start.elapsed();

    Ok(Run {
        tokens,
        step_ns,
        sample_ns,
        kv_len: session.pos(),
        hit_eos,
        turns_run,
        prefill_submit_ns,
        elapsed,
        commits,
        max_in_flight,
        overrun,
        rewound,
        step_markers,
        step_waits,
        arms: rb.arms(),
        eos_read: rb.eos_read_arm(),
        counters: rb.counters(),
    })
}
