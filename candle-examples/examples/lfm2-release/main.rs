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
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::lfm2::{
    AttnImpl, Cache, Config, ConvState, KvAppend, Lfm2Config, Model,
};
use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
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

    let model_dir = args.model_dir.or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

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
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;

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
    let mut cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::TopP {
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

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

    // Reserved BEFORE the window opens. A `Vec` growth inside the timed region
    // would be an allocation the user does not pay for on a steady-state path
    // and this harness would be measuring its own bookkeeping (§6.4a).
    let mut tokens: Vec<u32> = Vec::with_capacity(args.n + args.turns + 1);
    // Per-step wall, for the steady-state figure and the distribution. This is a
    // push into a reserved `Vec` of `u64` nanoseconds -- one `Instant::elapsed`
    // per token, which is the same `clock_gettime_nsec_np` against a 24 MHz
    // timebase the profiler stamps (§3.4b) and is not a per-token *series* in
    // §14.5's sense: nothing is written, sampled or read back.
    let mut step_ns: Vec<u64> = Vec::with_capacity(args.n + args.turns + 1);
    // Only written under `--sample-split`, and reserved either way so the flag
    // does not change the allocation behaviour of the path it measures.
    let mut sample_ns: Vec<u64> = Vec::with_capacity(args.n + args.turns + 1);

    // Starts at 0, not at `prompt_ids.len()`: turn 0's prefill is what appends
    // the prompt, so the forward pass must be told the cache is empty and
    // `kv_len` advances by the turn's length after it (`lfm2-determinism`
    // `main.rs:891`, `:982-985`). Seeding it with the prompt length instead
    // double-counts and the attention mask fails to broadcast.
    let mut kv_len = 0usize;
    let mut hit_eos = false;
    let mut turns_run = 0usize;
    let mut prefill_submit_ns: u64 = 0;

    // ---- the timed window --------------------------------------------------
    //
    // One clock over the whole loop. It opens before the first prefill and closes
    // after the last token is pushed, so it contains every prefill, every forward
    // pass, every `sample()` and every EOS test -- which is what a user waits
    // for. Nothing inside it writes, allocates or reads anything back except what
    // the decode itself requires.
    let run_start = std::time::Instant::now();

    'turns: for turn in 0..args.turns.max(1) {
        let turn_ids: Vec<u32> = if turn == 0 {
            prompt_ids.clone()
        } else {
            let follow_up = FOLLOW_UPS[(turn - 1) % FOLLOW_UPS.len()];
            let text = format!("<|im_start|>user\n{follow_up}<|im_end|>\n<|im_start|>assistant\n");
            tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizing follow-up: {e}"))?
                .get_ids()
                .to_vec()
        };

        let prefill_start = std::time::Instant::now();
        let input = Tensor::new(turn_ids.as_slice(), &device)?.unsqueeze(0)?;
        let mut logits = model
            .forward(&input, kv_len, &mut cache)
            .context("prefill forward pass")?
            .squeeze(0)?;
        kv_len += turn_ids.len();
        turns_run = turn + 1;
        // Accumulated across turns; reported apart from the decode rate because
        // prefill is a different shape of work (§6.6) and folding it in is the
        // whole-generation average this project has twice been misled by.
        //
        // **This is SUBMISSION time, not execution time, and the field name says
        // so.** Candle is asynchronous: `forward` returns once the work is
        // encoded. `lfm2-decode-profile` brackets its prefill with
        // `device.synchronize()` on both sides (`main.rs:1666`, `:1694`) and so
        // reports ~90 ms where this reports ~30 ms; the difference is the
        // prefill's GPU tail, which here lands in the first decode steps and is
        // what `--warmup` excludes. Draining here would make the two comparable
        // -- and would also put a synchronization in the release path that a
        // user's decode loop does not have, which is the thing this binary
        // exists not to do. So it is named honestly instead.
        prefill_submit_ns += prefill_start.elapsed().as_nanos() as u64;

        loop {
            if tokens.len() >= args.n {
                break 'turns;
            }

            let step_start = std::time::Instant::now();

            // **Inside the window, and this is the difference from the profiler.**
            // §11.5a prices this readback at 441 µs/token and classifies it
            // EXPOSED: the host waits on a specific completion, so overlap cannot
            // hide it. The profiler's `wall` opens after this call; a user does
            // not get to.
            // Under `--sample-split` the forward pass is drained BEFORE the
            // sample call is timed. Without this the two are not separable at
            // all: `forward()` returns having only *encoded* the work, and
            // `sample()` is then the call that blocks on its completion, so a
            // naive split attributes the whole forward pass to sampling. The
            // first version of this diagnostic did exactly that and read
            // **16.53 ms, 93 % of the step** -- a figure that is real as an
            // interval and wrong as an attribution, and it is recorded in the
            // write-up because it is the trap the profiler's own window shape
            // avoids by synchronizing explicitly (`lfm2-decode-profile:1839`).
            if args.sample_split {
                device.synchronize()?;
            }
            let sample_start = std::time::Instant::now();
            let next = logits_processor.sample(&logits).context("sampling")?;
            if args.sample_split {
                sample_ns.push(sample_start.elapsed().as_nanos() as u64);
            }

            // **EOS ends the turn WITHOUT being pushed or hashed**, which is
            // `lfm2-determinism`'s order (`main.rs:1227-1237`: the `break`
            // precedes `token_hasher.update`) and therefore the order the
            // canonical pairs in §2.3.8c were taken in. Pushing it here would
            // move the digest by one token and the gate would read a defect in
            // the release build that was this harness's own bug.
            if eos_ids.contains(&next) {
                hit_eos = true;
                if turn + 1 < args.turns.max(1) {
                    break;
                }
                break 'turns;
            }

            tokens.push(next);

            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            logits = model
                .forward(&input, kv_len, &mut cache)
                .context("decode forward pass")?
                .squeeze(0)?;
            kv_len += 1;

            // No explicit `device.synchronize()`, and its absence is the point.
            // The next iteration's `sample()` reads the logits back to the host,
            // which is itself the synchronization -- so the serialization stays
            // inherent to the workload rather than being one this harness added.
            // `lfm2-determinism` has no synchronize either, and a step timed here
            // is therefore bounded by the readback that opens the NEXT step. The
            // end-to-end figure is unaffected by where that boundary falls; only
            // the per-step distribution is, which is why the publishable number
            // is the end-to-end one.
            step_ns.push(step_start.elapsed().as_nanos() as u64);
        }
    }

    let elapsed = run_start.elapsed();
    // ---- the window is closed; everything below is after the clock ----------

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

    // One machine-readable line, in the shape the other harnesses print, so an
    // analysis script reads a line rather than parsing prose.
    println!(
        "RESULT mode=release n={} warmup={} turns={} generated={} attn={:?} kv_append={:?} \
         conv_state={:?} dtype={:?} seed={} temp={} top_p={} elapsed_s={:.6} \
         end_to_end_tok_s={:.4} steady_ms_per_step={:.6} steady_tok_s={:.4} \
         step_sd_ms={:.6} step_p5_ms={:.6} step_med_ms={:.6} step_max_ms={:.6} \
         prefill_submit_ms={:.4} final_kv_len={} hit_eos={} token_digest={}",
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
        sd_ms,
        p5_ms,
        med_ms,
        max_ms,
        prefill_submit_ns as f64 / 1e6,
        kv_len,
        hit_eos,
        token_digest
    );

    Ok(())
}
