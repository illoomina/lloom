//! Does LFM2 still answer a question? A two-turn text smoke test.
//!
//! Harness for lloom issue #121, and the tool `CONTRIBUTING.md` §3.1a asks for.
//!
//! # The gap this fills
//!
//! Every LFM2 run on this project reports **digests and never text**.
//! `lfm2-determinism` loads a tokenizer, uses it to *encode*, and decodes
//! nothing — so fifty commits of kernel work have been validated entirely on
//! `5e5ba45a…` matching `5e5ba45a…`.
//!
//! That is the right gate for *"did this change the computation"* and it is
//! blind to *"is the model still answering questions."* **A digest is equally
//! stable whether the model is coherent or emitting the same garbage every
//! run** — §2.3.1's three kinds of determinism are independent, and §2.3.5a
//! says it plainly: a build can be reproducible and wrong at once.
//!
//! # Why a sibling binary rather than a `--show-text` flag
//!
//! Issue #121 asks for the argument. Three reasons, and the third is the one
//! that decides it:
//!
//! 1. **The digest path stays byte-for-byte what it was.** A flag would put a
//!    `tokenizer.decode` inside the loop whose output is the gate every kernel
//!    change is measured against, and §2.4's standing rule is to check whether
//!    the measurement tool is the cost. Here the property is stronger than
//!    "measured and found not to perturb": `lfm2-determinism` is *not edited*,
//!    so its digests are unchanged by construction rather than by experiment.
//! 2. **The two harnesses want opposite prompts.** The determinism gate wants a
//!    long open-ended prompt generating hundreds of tokens across turns, which
//!    is what makes it a strong test (§2.3.6 — long multi-turn is strongest).
//!    This wants a *short* prompt with one checkable answer. One binary serving
//!    both means one prompt set, and whichever wins the other check is weaker.
//! 3. **This project already made this split once, and paid for getting it
//!    wrong.** `lfm2-determinism` printed a `tok_per_s` that was not throughput
//!    for twenty-odd issues and misled two readers, including a reviewer of #97
//!    (§2.4, lloom #102). The lesson recorded there is that the two harnesses
//!    have separate jobs — *"this one measures digests, `lfm2-decode-profile`
//!    measures throughput"* — and a number needing a caveat every time it is
//!    read is a number better not printed beside a different quantity. Adding a
//!    text check to the digest harness is the same shape of decision, and it is
//!    declined for the same reason.
//!
//! The cost of a sibling is a duplicated model loader. That is the convention
//! all three existing LFM2 harnesses already follow (`lfm2-decode-profile` and
//! `lfm2-dispatch-trace` each carry their own `parse_config`), and it is what
//! keeps them independently revertible (`CONTRIBUTING.md` §4.1).
//!
//! # The two turns, and why that shape
//!
//! **Turn 1 — a completion with an obvious continuation.** A broken model fails
//! this *visibly and instantly*: repetition, a wrong token, an immediate EOS —
//! where a digest reports a stable hash of the same failure.
//!
//! Note it still goes through the chat template. `lfm2-determinism`'s
//! `--raw-prompt` records the finding: without the template LFM2 emits EOS at
//! the first step and then degenerates, which measures nothing.
//!
//! **Turn 2 — a fact against the accumulated cache.** By this point KV has
//! grown, conv state has been shuffled 22 times per token (§5.3 — twenty-*two*
//! conv layers, corrected by #88; §6.1 is the shuffle), and a second prefill
//! runs on top of both. **If turn 2 is wrong while turn 1 was fine, the cache is
//! the suspect** — which is a sharper signal than a moved digest, because it
//! says *where*.
//!
//! # Three constraints, each load-bearing
//!
//! 1. **Greedy and seeded.** `Sampling::ArgMax`, not top-p. Under sampling turn
//!    2's substring check is a coin flip and the harness becomes flaky — which
//!    is *worse than absent*, because a flaky gate gets ignored and then gets
//!    deleted.
//! 2. **The checks exit non-zero.** `"Paris"` is a low bar, and a low bar that
//!    actually fires is worth more than a high bar nobody runs.
//!
//!    **Both turns carry a substring check, and the first one was added after a
//!    mutation defeated the harness.** Turn 1 originally asserted only that some
//!    text came back. Narrowing §6.1's conv-state shuffle the wrong way
//!    degenerated the completion from `"…jumps over the lazy dog."` to `"The
//!    quick brown"` — 10 tokens to 3 — and this harness reported **PASS** while
//!    the digest gate reported a well-formed, changed digest pair and exit 0.
//!    Neither caught it. That is §11.3j's vacuous-parity-arm lesson in a second
//!    place: a guard that survives the defect it exists to catch is not a guard,
//!    and "the model emitted something" is that shape of guard.
//! 3. **Not a quality benchmark.** No perplexity, no scoring, no model
//!    comparison, and deliberately no timing — the question is *did the model
//!    stop working*, nothing more. §2.4 and lloom #102 are why there is no
//!    `tok_per_s` here either.
//!
//! # What it is not
//!
//! **Not a replacement for the digest gate.** Both, and they catch different
//! things: this one cannot tell you *which* kernel moved. Run it beside
//! `lfm2-determinism`, not instead of it.
//!
//! ```bash
//! cargo run --release --features metal --example lfm2-smoke -- --attn generic
//! cargo run --release --features metal --example lfm2-smoke -- --attn sdpa
//! ```
//!
//! Both arms must print the same text while their digests legitimately differ
//! (§2.3.8c) — which is the single best demonstration of why this test exists.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_examples::lfm2_setup::{default_model_dir, parse_config, tensor_names, weight_files};
use clap::Parser;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{AttnImpl, ConvState, KvAppend, Model, Shape as Lfm2Shape, State};
use lloom::nn::ops::KvQuant;
use lloom::nn::VarBuilder;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "LFM2 two-turn text smoke test: did the model stop working?")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Turn 1: a prompt with an obvious continuation.
    ///
    /// The point is not the fox. It is that a broken model fails this visibly
    /// and instantly — repetition, a wrong token, an immediate EOS — where a
    /// digest would report a stable hash of the same failure.
    #[arg(
        long,
        default_value = "Complete this sentence with one word: The quick brown fox jumps over the lazy"
    )]
    turn1: String,

    /// Substring turn 1's completion must contain, case-insensitively.
    ///
    /// **This carries the same weight as `--expect` and was added because the
    /// first version of this harness did not have it.** Turn 1 originally only
    /// checked that *some* text came back, and a deliberate conv-state mutation
    /// (§6.1's shuffle, narrowed the wrong way) degenerated the completion from
    /// `"…jumps over the lazy dog."` to `"The quick brown"` — 10 tokens to 3 —
    /// while the harness reported **PASS**. A guard that survives the defect it
    /// exists to catch is §11.3j's vacuous-parity-arm lesson in a second place,
    /// and the fix is the same one: make the check compare against something.
    #[arg(long, default_value = "dog")]
    expect_turn1: String,

    /// Turn 2: a single unambiguous fact, asked against the accumulated cache.
    #[arg(long, default_value = "What is the capital of France?")]
    turn2: String,

    /// Substring turn 2's answer must contain. Checked case-insensitively;
    /// a miss is a non-zero exit.
    #[arg(long, default_value = "Paris")]
    expect: String,

    /// Max tokens per turn. Small deliberately: this is a smoke test, and a
    /// long generation would be measuring something else.
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,

    /// Sampling seed.
    ///
    /// Recorded and reported even though decoding is greedy: `LogitsProcessor`
    /// takes one, and printing the value it was constructed with is cheaper
    /// than arguing later about whether it mattered.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,

    /// Weight dtype. Defaults to f16 on Metal, matching how ambrogio loads it.
    #[arg(long)]
    dtype: Option<String>,

    /// Which attention implementation decode takes: `generic` (the default) or
    /// `sdpa` (the GQA-native `sdpa_vector` kernel, issue #97).
    ///
    /// **Running both arms is the demonstration this harness exists for.** Their
    /// digests legitimately differ — `5e5ba45a…`/`5903d463…` against
    /// `8a6e7eca…`/`6dc77c78…` (§2.3.8c) — while the *text* should not. Two
    /// legitimately different digests, one sensible answer, and the digest gate
    /// cannot tell you the second thing.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV tokens per page for `--attn flash` — the **allocation** granularity.
    ///
    /// §10.4 proposes 256 and marks it **UNVERIFIED**. A flag rather than a
    /// constant because §10.3d establishes page size as two dispatch-tier
    /// numbers, so 16, 256 and 1024 are one field apart — which is what
    /// *"must not foreclose it"* requires. Verifying which wins is #61's axis.
    #[arg(long, default_value_t = 256)]
    flash_page_size: usize,

    /// Pages per chunk for `--attn flash` — the `k` of
    /// `chunk_size = k * page_size`.
    ///
    /// §10.4 fixes the page and the chunk equal **by fiat**; §9.1d records that
    /// a page (allocation) and a tile (computation) are optimised against
    /// disjoint cost functions, so **a sweep holding `k = 1` cannot separate a
    /// page-size effect from a tile-size one**. 1 is what §10.4 specifies; the
    /// flag is what makes the two separable.
    #[arg(long, default_value_t = 1)]
    flash_k: usize,

    /// How `--attn flash` sizes the scratch class: `grow` (the default),
    /// `reserve` or `bucket` — `DESIGN.md` §9.1a's three policies, issue #234.
    ///
    /// Sizing decides **where bytes live and not what they are**, so all three
    /// arms owe the same text here and the same digests in
    /// `lfm2-determinism` — which is why this flag exists on the gates rather
    /// than only on the profiler. `grow` is what #116 allocated, so the default
    /// is byte-for-byte what shipped.
    #[arg(long, default_value = "grow")]
    flash_scratch_sizing: String,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset — issue #142).
    ///
    /// **This harness is the primary correctness evidence for that arm**, and
    /// the reason is #128's own demonstration: its mutation was *"appending the
    /// value cache in the wrong order"*, which produced three identical digests
    /// of a model emitting `picture picture picture`. The digest gate cannot see
    /// that bug class; this one can, and `in-place` changes exactly where KV
    /// bytes live.
    ///
    /// Turn 2 is the part that matters here. It re-prefills onto the
    /// *accumulated* cache without clearing it, so under `in-place` it is a
    /// `seq_len > 1` write at a non-zero offset — the turn boundary §11.1a's
    /// single-turn limitation is the precedent for.
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// How the Q/K/V `transpose` views reach the norms and the KV append
    /// (`QkvLayout`, `DESIGN.md` §6.2c): `contiguous`, `strided` or `fused`.
    #[arg(long, default_value = "contiguous")]
    qkv_layout: String,

    /// KV cache element width: `off` (the f16 default), `int8` or `int4`
    /// (issue #218, `DESIGN.md` §10.7).
    ///
    /// **A moved digest is EXPECTED on this axis and is not itself the
    /// finding.** Quantization is a real precision loss rather than a
    /// reordering, so what is owed is the ulp delta's **magnitude** and its
    /// **growth slope** against §10.7a's ladder, per §2.3.3c's tier-2
    /// sub-distinction. Magnitude is primary: #128's defect ratio is 0.787, so
    /// reading growth first would have called a real defect benign.
    #[arg(long, default_value = "off")]
    kv_quant: String,

    /// How decode writes conv state: `shuffle` (§6.1's `narrow` + `Tensor::cat`,
    /// the default), `ring[:K[:slack]]` (the sliding window, §10.2e) or
    /// `rotating[:K]` (§10.2a's rotating index, §10.2g).
    ///
    /// **This gate is what the digests cannot say for the rotating arm.** Its
    /// digests move by construction, so §15.1 #7 can only report that they are
    /// stable; whether the model still answers questions is this harness's
    /// question and nothing else's (§2.3.8d).
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// Fuse SiLU onto `w1`'s GEMV store (#484's kernel, wired by #509).
    ///
    /// **This gate is what the digests cannot say for this arm**, exactly as
    /// for `rotating` above. The fusion moves the digests by construction
    /// (§12.2d), so §15.1 #7 can only report that they are *stable*; whether
    /// the model still answers the question is this harness's job and nothing
    /// else's (§2.3.8d).
    #[arg(long, default_value_t = false)]
    gemv_silu: bool,

    /// Fuse the two residual adds onto their producing GEMV's store (#509).
    ///
    /// Needs no compile flag since #516; it used to require
    /// `-DLLOOM_509_RESIDUAL_EPILOGUE` through `LLOOM_434_METAL_CFLAGS`. Moves
    /// the digests by construction (§12.2h) — see `--gemv-silu` for why this
    /// gate is the one that matters then.
    #[arg(long, default_value_t = false)]
    gemv_residual: bool,

    /// How the SwiGLU product is dispatched (`GemvGate`, #732): `off` |
    /// `mul` | `swiglu`. The digests move; the text is expected identical.
    #[arg(long, default_value = "off")]
    gemv_gate: String,

    /// Fold the decode conv's `broadcast_mul + sum_keepdim` into one depthwise
    /// `conv1d` dispatch (`ConvFused`, §12.2 row 1). The digests move; the text
    /// is expected identical, which is what this harness checks.
    #[arg(long, default_value_t = false)]
    conv_fused: bool,

    /// Who selects the GEMV epilogue fusions (`Fusion`, issue #732): `manual`
    /// (the two flags above as given) or `derived` (the optimizer's plan,
    /// derived at construction). The text is expected identical either way.
    #[arg(long, default_value = "manual")]
    fusion: String,

    /// Which SiLU implementation the 30 `usilu` dispatches resolve to
    /// (issue #599, `DESIGN.md` §12.2l).
    ///
    /// `exact` | `recip` | `literal` | `coarse4` | `coarse1` | `hardswish`,
    /// in ascending error order. **`exact` is the default and is what ships**
    /// — it resolves the same `silu_f16` name it always did, so an
    /// unconfigured run is byte-identical (§7.1a).
    ///
    /// This is the text half of #599's absorption question. A coarser arm
    /// moves the digests by construction, so `lfm2-determinism` cannot say
    /// whether the model still WORKS under it — only this harness can
    /// (§2.3.8d: three identical digests of a model emitting
    /// `picture picture picture`).
    #[arg(long, default_value = "exact")]
    silu_impl: String,

    /// Print each turn's prompt and completion. On by default; `--quiet`
    /// leaves only the verdict lines, for a scripted gate.
    #[arg(long)]
    quiet: bool,
}

/// One turn's outcome, kept so the verdict can be decided after both have run.
struct Turn {
    prompt: String,
    text: String,
    tokens: usize,
    hit_eos: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // A gate states its contract: deterministic, unless LLOOM_PROFILE names one.
    candle::tuning::pin_contract(candle::tuning::Contract::Deterministic)
        .map_err(anyhow::Error::msg)?;
    let key = shape_key(&args)?;
    println!("PROFILE {key}");
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => run::<P>(args))
}

fn run<P: Lfm2Shape>(args: Args) -> Result<()> {
    let contract = candle::tuning::gate_contract("lfm2-smoke").map_err(anyhow::Error::msg)?;
    println!("CONTRACT {}", contract.report());

    let model_dir = args.model_dir.clone().or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    // f16 on Metal is what ambrogio loads. Measuring f32 here would exercise a
    // path nothing runs.
    //
    // This comment used to add: "LFM2 ships bf16, but Metal's bf16 kernel
    // coverage is patchy and unsupported ops fall back to the CPU silently."
    // **That is false, and it had never been tested** (#307, §3.9) -- it was
    // quoted into `DESIGN.md` §9.1b as a fact and made load-bearing for three
    // decisions. THIS harness is what refutes it: it PASSes at `--dtype bf16`
    // on all three `--attn` arms with text byte-identical to the f16 arm.
    // The default stays f16 because it is what the consumer loads, which is the
    // reason this comment led with (§7.1a: a default is its own argued
    // decision).
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

    // Refused rather than defaulted. A silent fallback would report the generic
    // path's text under the sdpa arm's name, and running both arms is the whole
    // point of this harness — #69's vacuous determinism run is exactly the
    // failure this avoids (§2.4, §9.2f).
    config.attn_impl = match args.attn.as_str() {
        "generic" => AttnImpl::Generic,
        "sdpa" => AttnImpl::Sdpa,
        // Issue #116. Selectable and never a default: 10.4's argument for it
        // is structural rather than measured, and the kv_len at which it pays
        // is #61's to find.
        "flash" => AttnImpl::FlashDecoding,
        other => anyhow::bail!("--attn must be `generic`, `sdpa` or `flash`, got `{other}`"),
    };
    config.flash_page_size = args.flash_page_size;
    config.flash_pages_per_chunk = args.flash_k;
    // §9.1a's sizing policy (#234). Refused rather than defaulted, for the same
    // reason as `--attn`: a silently-ignored flag makes a gate report on an arm
    // it did not run (§2.4).
    config.flash_scratch_sizing =
        lloom::nn::ops::FlashScratchSizing::parse(&args.flash_scratch_sizing)
            .map_err(anyhow::Error::msg)?;
    // Refused rather than defaulted, for the same reason as `--attn` above.
    config.kv_append = match args.kv_append.as_str() {
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        "paged" => KvAppend::Paged,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    config.qkv_layout =
        lloom::models::lfm2::QkvLayout::parse(&args.qkv_layout).map_err(anyhow::Error::msg)?;

    // Refused rather than defaulted, for `--kv-append`'s reason: an
    // unrecognized value silently pinning the default would report the f16
    // arm's digest under a quantized arm's name -- #69's vacuous run (§2.4).
    config.kv_quant = KvQuant::parse(&args.kv_quant).map_err(anyhow::Error::msg)?;

    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    println!("conv state: {:?}", config.conv_state);

    // #599's SiLU arm. Refused rather than defaulted, for `--attn`'s reason: a
    // silently-ignored flag makes this harness report the exact arm's text
    // under a coarse arm's name, which is precisely the absorption question
    // answered wrong. Printed whether or not it is `exact`, because *inert* and
    // *unrecorded* are different facts (#241) and a PASS is only interpretable
    // against the arm it was taken in.
    let silu = candle::kernels::SiluImpl::parse(&args.silu_impl).ok_or_else(|| {
        anyhow::anyhow!(
            "--silu-impl must be one of `exact`, `recip`, `literal`, `coarse4`, \
             `coarse1`, `hardswish`, got `{}`",
            args.silu_impl
        )
    })?;
    // The SiLU arm rides `Config::silu_impl` now (RESTRUCTURING §1.6):
    // `Model::new` pushes the field into the kernel cache, so setting the atomic
    // directly here would be clobbered by that push. Set the field instead.
    config.silu_impl = silu;
    println!("silu impl: {}", silu.name());

    // #509's wiring. Printed whether or not set: *inert* and *unrecorded* are
    // different facts (#241), and this harness's verdict is only interpretable
    // against the arm it was taken in.
    config.gemv_silu = args.gemv_silu;
    config.gemv_residual = args.gemv_residual;
    config.gemv_gate =
        lloom::models::lfm2::GateFusion::parse(&args.gemv_gate).map_err(anyhow::Error::msg)?;
    config.conv_fused = args.conv_fused;
    config.fusion =
        lloom::models::lfm2::FusionPolicy::parse(&args.fusion).map_err(anyhow::Error::msg)?;
    println!(
        "gemv epilogue: silu={} residual={} gate={}",
        config.gemv_silu,
        config.gemv_residual,
        config.gemv_gate.name()
    );

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    // A VL checkpoint nests the language stack under `model.language_model.`;
    // candle's text-only loader asks for `model.*`.
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    let model = Model::<P>::new(&config, vb).context("constructing LFM2 model")?;
    let mut cache =
        State::<P>::new(true, dtype, &config, &device).context("allocating KV cache")?;

    // Greedy, and it is a constraint rather than a default (issue #121). Under
    // sampling the turn-2 substring check is a coin flip and this harness
    // becomes flaky — which is worse than absent, because a flaky gate gets
    // ignored and then gets deleted.
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, Sampling::ArgMax);

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }
    anyhow::ensure!(
        !eos_ids.is_empty(),
        "no EOS id found, so a turn could only end by exhausting --max-tokens"
    );

    println!("model     {}", model_dir.display());
    println!("attention {:?}", config.attn_impl);
    println!("kv append {:?}", config.kv_append);
    println!("kv quant  {:?}", config.kv_quant);
    println!(
        "dtype     {dtype:?}   device {}",
        if args.cpu { "cpu" } else { "metal" }
    );
    println!("sampling  greedy (argmax), seed {}", args.seed);
    println!();

    // Both turns run against one cache, which is what makes turn 2 a test of
    // the cache rather than a second independent question.
    let mut kv_len = 0usize;
    let mut turns: Vec<Turn> = Vec::with_capacity(2);

    for (idx, prompt) in [args.turn1.as_str(), args.turn2.as_str()]
        .into_iter()
        .enumerate()
    {
        // Through the chat template, like everything else. `lfm2-determinism`'s
        // `--raw-prompt` records why: fed a bare string LFM2 emits EOS at the
        // first step and then degenerates, which measures nothing.
        let templated = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");

        // `add_special_tokens` only on the first turn: the second appends to a
        // conversation that already has its BOS, and a second one mid-stream
        // would be a token the model was never trained to see there.
        let ids = tokenizer
            .encode(templated.as_str(), idx == 0)
            .map_err(|e| anyhow::anyhow!("tokenizing turn {}: {e}", idx + 1))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(!ids.is_empty(), "turn {} tokenized to nothing", idx + 1);

        let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
        let mut logits = model
            .forward(&input, kv_len, &mut cache)
            .with_context(|| format!("turn {} prefill", idx + 1))?
            .squeeze(0)?;
        kv_len += ids.len();

        let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
        let mut hit_eos = false;

        while generated.len() < args.max_tokens {
            let next = logits_processor.sample(&logits).context("sampling")?;
            if eos_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);

            let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
            logits = model
                .forward(&input, kv_len, &mut cache)
                .with_context(|| format!("turn {} decode", idx + 1))?
                .squeeze(0)?;
            kv_len += 1;
        }

        // Decoded in one shot at the end of the turn rather than streamed. The
        // per-token path is the one every kernel measurement runs against, and
        // there is no reason for this harness to put a tokenizer call inside it.
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decoding turn {}: {e}", idx + 1))?;

        turns.push(Turn {
            prompt: prompt.to_string(),
            text,
            tokens: generated.len(),
            hit_eos,
        });
    }

    if !args.quiet {
        for (i, turn) in turns.iter().enumerate() {
            println!("--- turn {} ---", i + 1);
            println!("prompt: {}", turn.prompt);
            println!("output: {}", turn.text.trim());
            println!(
                "        ({} tokens, {})",
                turn.tokens,
                if turn.hit_eos {
                    "stopped at EOS"
                } else {
                    "hit --max-tokens"
                }
            );
            println!();
        }
    }

    // The verdict. One check per turn, and they fail for different reasons —
    // which is the point of having two turns rather than one.
    //
    // Both are substring checks, deliberately. What a "good" continuation looks
    // like is a judgement, and a smoke test that argues about its own verdict is
    // not a smoke test (issue #121); whether the text contains a fixed word is
    // not a judgement. The bar is low on purpose, and a low bar that actually
    // fires is worth more than a high bar nobody runs.
    let contains = |text: &str, want: &str| text.to_lowercase().contains(&want.to_lowercase());

    let mut failures: Vec<String> = Vec::new();

    // Turn 1: the model completed the sentence. A broken model fails this
    // visibly and instantly — repetition, a wrong token, an early EOS.
    let turn1_ok = contains(&turns[0].text, &args.expect_turn1);
    if !turn1_ok {
        failures.push(format!(
            "turn 1 does not contain {:?} ({} tokens, hit_eos={}): {:?}",
            args.expect_turn1,
            turns[0].tokens,
            turns[0].hit_eos,
            turns[0].text.trim()
        ));
    }

    // Turn 2: the same check against the *accumulated* cache. By here KV has
    // grown, conv state has been shuffled 22 times per token (§5.3, §6.1), and
    // a second prefill has run on top of both.
    //
    // **If turn 2 fails while turn 1 passed, the cache is the suspect** — which
    // is a sharper signal than a moved digest, because it says where.
    let turn2_ok = contains(&turns[1].text, &args.expect);
    if !turn2_ok {
        failures.push(format!(
            "turn 2 does not contain {:?}: {:?}",
            args.expect,
            turns[1].text.trim()
        ));
    }

    println!(
        "SMOKE attn={:?} kv_append={:?} qkv_layout={} turn1_expect={:?} turn1_ok={} \
         turn1_tokens={} turn2_expect={:?} turn2_ok={} turn2_tokens={} verdict={}",
        config.attn_impl,
        config.kv_append,
        config.qkv_layout.as_str(),
        args.expect_turn1,
        turn1_ok,
        turns[0].tokens,
        args.expect,
        turn2_ok,
        turns[1].tokens,
        if failures.is_empty() { "PASS" } else { "FAIL" },
    );

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        // Non-zero exit, per issue #121: a low bar that actually fires is worth
        // more than a high bar nobody runs.
        anyhow::bail!("{} of 2 checks failed", failures.len());
    }

    Ok(())
}

/// The profile the run's axes name (`profiles-and-memory-design.md` §3):
/// `--kv-append` and `--conv-state` as `run` resolves them, read off
/// `Config::profile_key` at the contract in force; a divergence from `run`'s
/// resolution is refused by `Model::new` (`Config::check_shape`).
fn shape_key(args: &Args) -> Result<lloom::profile::ProfileKey> {
    let mut config = lloom::models::lfm2::Config::default();
    config.kv_append = match args.kv_append.as_str() {
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        "paged" => KvAppend::Paged,
        other => anyhow::bail!("--kv-append must be `cat`, `in-place` or `paged`, got `{other}`"),
    };
    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    Ok(config.profile_key())
}
