//! Is candle's decode dispatch sequence identical across tokens?
//!
//! Measurement harness for lloom issue #6 / `DESIGN.md` §11.1a, §16 P0 #2.
//!
//! An `MTLIndirectCommandBuffer` is only replayable if the command sequence is
//! identical between invocations. Candle is eager — ops execute as called, and
//! nothing guarantees the next forward pass issues the same dispatches against
//! the same buffers. This records (pipeline, buffers, grid) per dispatch across
//! N decode tokens and diffs consecutive tokens.
//!
//! The model driver is the one from the determinism probe
//! (`candle-examples/examples/lfm2-determinism`, lloom issue #5), so this
//! measures the same decode path that was shown to be bit-deterministic rather
//! than a second, differently-wrong reimplementation of it.
//!
//! Three outcomes are distinguished, because they have different consequences:
//!
//! * **identical** — replay is possible; the executor work is justified.
//! * **differs only in buffer identity** — the allocator hands out different
//!   buffers for the same logical slot. Fixable, and it makes the allocator a
//!   prerequisite rather than a parallel track.
//! * **differs structurally** — kernel, grid or binding shape changes. Replay
//!   would need the model to declare stability.
//!
//! ```bash
//! CANDLE_METAL_TRACE=1 cargo run --release --example lfm2-dispatch-trace -- --n 24
//! ```
//!
//! Without `CANDLE_METAL_TRACE=1` the instrumentation is inert and the harness
//! says so rather than reporting an empty trace as a result.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use lloom::nn::VarBuilder;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{
    AttnImpl, Cache, ConvState, KvAppend, Model,
};
use candle_examples::lfm2_setup::{
    default_model_dir, parse_config, tensor_names, weight_files,
};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use candle::metal_backend::{set_hazard_key, trace, ArenaLayout, HazardKey};

#[derive(Parser, Debug)]
#[command(about = "LFM2 decode dispatch-sequence probe")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt fed to the model.
    #[arg(
        long,
        default_value = "Explain, in careful detail, how a modern operating system schedules threads across CPU cores, and why fairness and throughput are in tension."
    )]
    prompt: String,

    /// Number of decode tokens to trace.
    #[arg(long, short = 'n', default_value_t = 24)]
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

    /// Weight dtype. Defaults to f16 on Metal, matching how ambrogio loads it.
    #[arg(long)]
    dtype: Option<String>,

    /// Print the full recorded sequence for these decode steps, comma separated.
    ///
    /// The raw record is what makes the verdict checkable rather than believed,
    /// so at least two consecutive steps are worth dumping.
    #[arg(long, default_value = "1,2")]
    dump_steps: String,

    /// Write the full trace of every step to this file.
    #[arg(long)]
    dump_all: Option<PathBuf>,

    /// Also trace the prefill pass, reported separately.
    ///
    /// Prefill runs a different shape (`seq` = prompt length against `seq` = 1),
    /// so mixing it into the decode comparison would manufacture a difference
    /// that says nothing about replayability across decode tokens.
    #[arg(long)]
    include_prefill: bool,

    /// Serve decode activations from an activation arena (`DESIGN.md` §9.2).
    ///
    /// One steady-state decode step is recorded to derive the plan, the arena is
    /// installed, and the remaining steps run against it. This is the arm whose
    /// "buffer identity varies" count should read 0.
    #[arg(long)]
    arena: bool,

    /// Arena layout: `packed` (§9.2's plan) or `non-aliasing` (§9.3's reference,
    /// every value in its own slot so aliasing is impossible by construction).
    ///
    /// The two must produce identical output. That comparison is the parity
    /// check §9.3 requires, and it is an execution comparison rather than a size
    /// one -- the whole point being that offset arithmetic has no other detector
    /// under `HazardTrackingModeUntracked`.
    #[arg(long, default_value = "packed")]
    arena_layout: String,

    /// How intra-encoder hazards are keyed: `pointer` (candle's, and the
    /// default) or `range` (`DESIGN.md` §9.2e route (a)).
    ///
    /// `pointer` treats two values at different offsets of one allocation as
    /// the same resource, which is correct but cannot tell "the same bytes"
    /// from "the same allocation" -- under an arena every activation shares one
    /// pointer, so every write-then-read pair inside it becomes a false
    /// dependency. `range` decides overlap by an interval test, so it can only
    /// remove barriers and never add one.
    ///
    /// Defaults to `CANDLE_METAL_HAZARD_KEY` when that is set, so a harness
    /// without the flag still gets the A/B -- the determinism probe is driven
    /// that way.
    #[arg(long)]
    hazard_key: Option<String>,

    /// Where arena offsets are computed: `cpu` (#69's path, the default) or
    /// `gpu` (issue #70 -- a kernel bump-allocates them over the arena).
    ///
    /// The GPU table is verified element-wise against the CPU plan before it is
    /// adopted, and the arena refuses to switch if they disagree. So this flag
    /// selects *which allocator computed the offsets*, never a different set of
    /// bytes: bit-identical activations follow by construction rather than by
    /// measurement (`DESIGN.md` §9.2d).
    ///
    /// Requires `--arena`, and `--arena-layout non-aliasing`: a packed plan
    /// reuses slots, so its offsets are not monotone and no forward-only cursor
    /// can reproduce them. Both are refused loudly rather than silently falling
    /// back, because a silent fallback reports the CPU path's numbers under the
    /// GPU path's name -- which is #69's vacuous determinism run exactly (§2.4).
    #[arg(long, default_value = "cpu")]
    arena_offsets: String,

    /// Which attention implementation decode takes: `generic` (the default, and
    /// what every trace before lloom #97 recorded) or `sdpa` (the GQA-native
    /// `sdpa_vector` kernel).
    ///
    /// The claim under test is that `sdpa` removes 24 varying-grid positions --
    /// 16 `cast_f16_f32` from the F32 upcast of K and V, and 8 `affine_f32` from
    /// the softmax scale over a growing `kv_len` (`DESIGN.md` §11.1a.1). Both
    /// arms are compiled, so this selects between them at model construction and
    /// the A/B costs nothing.
    #[arg(long, default_value = "generic")]
    attn: String,

    /// KV cache growth: `cat` (the default `Tensor::cat` reallocation) or
    /// `in-place` (pre-allocated, written at a moving offset -- issue #142).
    ///
    /// **The digests must not move across this axis**, which is what makes it
    /// different from `--attn`. `Sdpa` changes the arithmetic and legitimately
    /// moves them (§2.3.8c); `in-place` changes only *where the bytes live*, so
    /// a moved digest here is a defect rather than a variant (§2.3.5a). Gate it
    /// against the pair for whichever `--attn` arm is selected, not against the
    /// canonical generic pair.
    #[arg(long, default_value = "cat")]
    kv_append: String,

    /// How decode writes conv state: `shuffle` (§6.1's `narrow` + `Tensor::cat`,
    /// the default), `ring[:K[:slack]]` (the sliding window, §10.2e) or
    /// `rotating[:K]` (§10.2a's rotating index, §10.2g).
    ///
    /// The two rings differ on exactly the axes this harness measures: `ring`
    /// compacts, so its per-token dispatch count is non-constant and its write
    /// offsets slide; `rotating` never compacts, so its count is constant and
    /// its offsets take one of `l_cache + K` fixed values.
    #[arg(long, default_value = "shuffle")]
    conv_state: String,

    /// How every kernel's scalars reach it: `split` (the default) or `packed`
    /// (one `device const Params*`, issue #115).
    ///
    /// This is the axis that decides whether a dispatch position is
    /// **ICB-encodable at all**, which is a different question from whether it
    /// is *replayable*. §9.2h and §9.2i classify the stream by what varies
    /// across steady-state steps -- identity, grid, offset -- and that is the
    /// right axis for replay. It is not sufficient for encoding: an ICB command
    /// has no `setBytes` in any form (§3.7c), so a position whose kernel binds
    /// its constants inline cannot be encoded however stable it is.
    ///
    /// Under `packed` every dispatch that has a packed sibling resolves a
    /// `*_packed` `[[host_name]]`, which is visible in the kernel census this
    /// harness prints -- so the census is what shows the switch engaged, rather
    /// than a flag being echoed back (§2.4, §9.2f).
    #[arg(long, default_value = "split")]
    param_style: String,

    /// Replay the stable, packed subset of a decode step from an
    /// `MTLIndirectCommandBuffer` (`DESIGN.md` §17 Phase 2 item 10, issue #115).
    ///
    /// Records the first few decode steps, encodes the positions that were
    /// invariant across them, and executes those by range on later steps with
    /// the classical dispatch suppressed. **Partial by construction**: what it
    /// covers is reported, and what it does not is attributed to a reason rather
    /// than left as a remainder.
    ///
    /// Needs `--param-style packed`, and says so rather than covering nothing:
    /// an ICB command cannot carry an inline constant (§3.7c), so under `split`
    /// every position is excluded and the run would report a working executor
    /// that replayed zero dispatches.
    #[arg(long)]
    icb: bool,

    /// Exit non-zero unless the `metal-hazard-audit` feature reached this
    /// binary (issue #185).
    ///
    /// **An unused feature flag reads exactly like a correct one.** §9.5l
    /// finding 5 is the worked case one crate over: admission compiled to its
    /// non-Metal stub in every harness because `candle-examples`' `metal`
    /// feature did not enable `candle-transformers/metal`, and nothing said so.
    /// #205 exists because `metal-run-telemetry` stops at `candle-core` and was
    /// therefore never reachable from any example.
    ///
    /// So the propagation is asserted by a **run**, not by reading four
    /// `Cargo.toml` files and believing them. `cfg!` is evaluated in *this*
    /// crate, so a `true` here means the chain
    /// `candle-examples -> candle -> candle-metal-kernels` resolved -- which is
    /// exactly the link that was missing.
    #[arg(long)]
    assert_hazard_audit: bool,
}

/// One decode step's recorded dispatches.
struct Step {
    label: String,
    dispatches: Vec<trace::Dispatch>,
}

impl Step {
    fn signature(&self) -> Vec<String> {
        self.dispatches.iter().map(|d| d.signature()).collect()
    }

    fn shape_signature(&self) -> Vec<String> {
        self.dispatches
            .iter()
            .map(|d| d.shape_signature())
            .collect()
    }

    fn kernel_signature(&self) -> Vec<String> {
        self.dispatches
            .iter()
            .map(|d| d.kernel_signature())
            .collect()
    }
}

/// How two steps differ.
///
/// Finer than the issue's three outcomes on purpose. "Structural" turned out to
/// cover two findings with opposite consequences for the ICB thesis: an op
/// sequence that genuinely changes shape, versus one whose kernels and binding
/// slots are fixed and only whose *grids* scale with `kv_len`. The latter is a
/// dispatch-tier parameter (`DESIGN.md` §7.1) and is replayable if the grid can
/// be made indirect; reporting it as "structural" would retire the ICB thesis on
/// a fault that does not warrant it.
#[derive(PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
enum Diff {
    /// Byte-for-byte the same commands, including buffers.
    Identical,
    /// Same kernels, grids, slots and offsets; different backing buffers.
    BufferIdentityOnly,
    /// Same kernel sequence and binding slots; some grids or offsets scale.
    GridOrOffsetScaling,
    /// The kernel sequence itself differs.
    KernelSequence,
}

fn compare(a: &Step, b: &Step) -> Diff {
    if a.signature() == b.signature() {
        Diff::Identical
    } else if a.shape_signature() == b.shape_signature() {
        Diff::BufferIdentityOnly
    } else if a.kernel_signature() == b.kernel_signature() {
        Diff::GridOrOffsetScaling
    } else {
        Diff::KernelSequence
    }
}

fn render(step: &Step) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== {} : {} dispatches ===\n",
        step.label,
        step.dispatches.len()
    ));
    for (i, d) in step.dispatches.iter().enumerate() {
        out.push_str(&format!(
            "{i:>4}  {} grid={:?} tg={:?} {} enc={}{}\n",
            d.pipeline,
            d.grid,
            d.threadgroup,
            if d.by_threadgroups {
                "threadgroups"
            } else {
                "threads"
            },
            d.encoder,
            // The kinds are appended AFTER the `BARRIER` token, never woven
            // into it (issue #185). `measurements/issue-144-raw/edge-cover.py`
            // and `hazard_audit_crosscheck` both key on that literal, and
            // #11.3h's recurring lesson is that a figure edited away from the
            // shape its readers expect is one nobody can re-check. Appending
            // leaves every existing parser matching and gives a new one the
            // attribution.
            if d.barrier {
                let kinds: Vec<&str> = d.kinds.iter().map(|k| k.as_str()).collect();
                if kinds.is_empty() {
                    "  BARRIER".to_string()
                } else {
                    format!("  BARRIER [{}]", kinds.join(","))
                }
            } else {
                String::new()
            }
        ));
        for b in &d.bindings {
            out.push_str(&format!(
                "        [{}] {} buf#{} @{:#x} off={}\n",
                b.index,
                if b.is_output { "out" } else { "in " },
                b.buffer_id,
                b.buffer_addr,
                b.offset
            ));
        }
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Before anything that needs a model or a GPU: did the feature arrive?
    // Checked first so the answer costs nothing and cannot be confused with a
    // load failure (issue #185).
    if args.assert_hazard_audit {
        let reached = cfg!(feature = "metal-hazard-audit");
        println!(
            "metal-hazard-audit reached candle-examples: {}",
            if reached { "YES" } else { "NO" }
        );
        if !reached {
            anyhow::bail!(
                "the `metal-hazard-audit` feature did not reach this binary.\n\
                 Build with:  cargo build --release --features metal,metal-hazard-audit \
                 --example lfm2-dispatch-trace\n\
                 If it was already in the command line, the propagation chain is \
                 broken in a Cargo.toml -- which is #205's defect, and the reason \
                 this flag exists rather than a comment claiming the wiring is right."
            );
        }
        return Ok(());
    }

    if !trace::trace_requested() {
        anyhow::bail!(
            "CANDLE_METAL_TRACE is not set, so no dispatches would be recorded.\n\
             Re-run with:  CANDLE_METAL_TRACE=1 cargo run --release --example \
             lfm2-dispatch-trace -- --n {}",
            args.n
        );
    }

    let model_dir = args.model_dir.or_else(default_model_dir).context(
        "no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache",
    )?;

    let device = Device::new_metal(0)?;

    // f16 on Metal is what ambrogio loads and what the determinism probe
    // measured. Tracing a dtype nothing runs would trace a different kernel mix.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => DType::F16,
    };

    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let mut config = parse_config(&config_raw)?;

    // Unknown values are refused rather than defaulted: a silent fallback would
    // report the generic path's dispatch counts under the sdpa arm's name, which
    // is #69's vacuous determinism run in another quantity (§2.4).
    config.attn_impl = match args.attn.as_str() {
        "generic" => AttnImpl::Generic,
        "sdpa" => AttnImpl::Sdpa,
        // Issue #116.
        "flash" => AttnImpl::FlashDecoding,
        other => anyhow::bail!("--attn must be `generic`, `sdpa` or `flash`, got `{other}`"),
    };
    config.conv_state = ConvState::parse(&args.conv_state).map_err(anyhow::Error::msg)?;
    println!("conv state: {:?}", config.conv_state);
    println!("attention implementation: {:?}", config.attn_impl);
    // Refused rather than defaulted, for the same reason as `--attn`: an
    // unrecognized value that silently pinned the default would report the
    // `cat` arm's digest under the `in-place` arm's name, which is #69's
    // vacuous run (§2.4, §9.2f).
    config.kv_append = match args.kv_append.as_str() {
        "cat" => KvAppend::Cat,
        "in-place" => KvAppend::InPlace,
        other => anyhow::bail!("--kv-append must be `cat` or `in-place`, got `{other}`"),
    };
    println!("kv append: {:?}", config.kv_append);

    // The binding-style axis (issue #115), refused rather than defaulted for the
    // same reason as the arm above. Read back from the crate rather than echoed
    // from `args`, so the printed line is what the kernels will use.
    {
        use candle::kernels::{default_param_style, set_default_param_style, ParamStyle};
        let want = match args.param_style.as_str() {
            "split" => ParamStyle::Split,
            "packed" => ParamStyle::Packed,
            other => anyhow::bail!("--param-style must be `split` or `packed`, got `{other}`"),
        };
        set_default_param_style(want);
        let got = default_param_style();
        anyhow::ensure!(
            got == want,
            "--param-style {want:?} did not take effect; default_param_style() reports {got:?}"
        );
        println!("param style: {got:?}");
    }

    // ICB replay (issue #115). Both switches have to be thrown *here*, before
    // the first pipeline is built:
    //
    // * `supportIndirectCommandBuffers` is a property of a pipeline and
    //   pipelines are cached per process, so one built earlier keeps the old
    //   setting -- and encoding that into an ICB is §3.7d's segfault at encode
    //   time, with no error to catch. `set_pipelines_support_icb` refuses to
    //   change under an existing pipeline rather than flipping and hoping.
    // * the constants pool has to be serving before the recorded steps run, or
    //   the params buffer is a fresh allocation per dispatch and every packed
    //   position varies (§11.3d).
    if args.icb {
        anyhow::ensure!(
            args.param_style == "packed",
            "--icb needs --param-style packed: an ICB command cannot carry an inline constant \
             (DESIGN.md §3.7c), so under `split` every position is excluded and the run would \
             report a working executor that replayed nothing"
        );
        candle::kernels::set_pipelines_support_icb(true)
            .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
        candle::kernels::set_constants_pool_enabled(true);
        println!("icb: enabled (pipelines support ICBs, constants pooled)");
    }

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

    // The checkpoint's chat template, reduced for a single user turn. Fed a bare
    // string LFM2 emits EOS at step 0 and there is no decode to trace.
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

    let mut steps: Vec<Step> = Vec::new();
    let mut kv_len = 0usize;

    // Prefill. Traced only on request and never mixed into the decode
    // comparison: `seq` is the prompt length here and 1 in decode, so the two
    // are different command sequences by construction.
    // Each traced window covers exactly one `model.forward`, and nothing else.
    //
    // Two things have to be kept out of it or the count is not the model's.
    // Sampling runs its own kernels (softmax, sort, affine) and is not part of
    // the forward pass. And candle is lazy on Metal, so the forward's dispatches
    // are not all encoded until something reads the result back — a readback
    // inside the window would encode the tail of the forward but also add cast
    // dispatches of its own.
    //
    // `device.synchronize()` resolves this: it forces the queued work to be
    // submitted and completed without encoding any compute of its own, so the
    // window closes on a drained queue containing only the forward.
    let trace_window = |device: &Device| -> Result<()> {
        device.synchronize()?;
        Ok(())
    };

    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    if args.include_prefill {
        trace::set_region(Some("prefill".to_string()));
        trace::set_recording(true);
    }
    let mut logits = model
        .forward(&input, kv_len, &mut cache)
        .context("prefill forward pass")?
        .squeeze(0)?;
    trace_window(&device)?;
    if args.include_prefill {
        trace::set_recording(false);
        let dispatches = trace::take_dispatches();
        steps.push(Step {
            label: "prefill".to_string(),
            dispatches,
        });
    }
    kv_len += prompt_ids.len();

    let metal_device = match &device {
        Device::Metal(d) => Some(d.clone()),
        _ => None,
    };
    let layout = match args.arena_layout.as_str() {
        "packed" => ArenaLayout::Packed,
        "non-aliasing" | "nonaliasing" | "reference" => ArenaLayout::NonAliasing,
        other => anyhow::bail!("unknown --arena-layout {other:?}; want packed or non-aliasing"),
    };
    // Issue #70. Validated here rather than at the install site so an
    // unsatisfiable combination fails before the model runs, not after.
    let gpu_offsets = match args.arena_offsets.as_str() {
        "cpu" => false,
        "gpu" => true,
        other => anyhow::bail!("unknown --arena-offsets {other:?}; want cpu or gpu"),
    };
    if gpu_offsets {
        if !args.arena {
            anyhow::bail!("--arena-offsets gpu needs --arena: there is nothing to allocate over");
        }
        if layout != ArenaLayout::NonAliasing {
            // Refused rather than downgraded. A packed plan reuses slots, so its
            // offsets are not monotone and a forward-only cursor cannot
            // reproduce them; falling back to the CPU here would report #69's
            // numbers under this flag's name.
            anyhow::bail!(
                "--arena-offsets gpu needs --arena-layout non-aliasing: a packed plan reuses \
                 slots, so its offsets are not monotone and no bump allocator can reproduce them"
            );
        }
    }
    // Set before any encoder session opens, since the keying is read once per
    // session rather than per bind.
    // An explicit flag wins; otherwise leave the selector alone so that
    // `CANDLE_METAL_HAZARD_KEY` is honoured. Calling `set_hazard_key`
    // unconditionally would consume the env switch and silently pin the
    // default -- which it did, and which made a determinism run that believed
    // it was testing route (a) actually test the default.
    if let Some(requested) = args.hazard_key.as_deref() {
        let key = match requested {
            "pointer" | "ptr" => HazardKey::Pointer,
            "range" => HazardKey::Range,
            other => anyhow::bail!("unknown --hazard-key {other:?}; want pointer or range"),
        };
        set_hazard_key(key);
    }
    // Reported rather than assumed, for the same reason: a run has to be able
    // to say which arm it was.
    eprintln!("hazard key: {:?}", candle::metal_backend::hazard_key());
    // Decode steps 0..record_steps are recorded, and the arena is installed
    // after them.
    //
    // **Two, not one.** Comparing two steps is what separates an activation from
    // session state: an allocation whose size moved between them is sized by
    // `kv_len` and must not enter the arena (`DESIGN.md` §9.1, #68 finding 4).
    // With one step there is no growth to observe, and the KV cache and its
    // `Tensor::cat` copies would be packed as though they were fixed -- which is
    // exactly the modelling error that makes 969 bindings unstable.
    //
    // They are the *first* steps so that every step the stability analysis
    // compares is served by the arena; the recording steps necessarily run on
    // the pool, since the arena does not exist until its plan does.
    let record_steps = 2usize;

    // Steps the ICB executor records before deciding what is replayable.
    //
    // Three rather than the minimum of two: a position that happens to agree
    // between two consecutive steps but drifts on the third would be admitted by
    // a two-step window, and admitting one wrongly is silent corruption under
    // `HazardTrackingModeUntracked` (§3.5) rather than a visible failure. Three
    // is cheap -- the cost is three classically-dispatched steps at the head of
    // the run, which are not measured anyway.
    const ICB_RECORD_STEPS: usize = 3;

    // The ICB executor records *after* the arena is installed, when one is.
    //
    // Recording earlier would capture the pool's buffer identities, which vary
    // per step by construction -- so every position would be excluded as
    // `varies` and the run would report zero coverage for a reason that has
    // nothing to do with the executor. The arena is what makes identity stable
    // (§9.2c), so it has to be in place before there is anything worth
    // recording.
    let icb_executor = args
        .icb
        .then(|| candle::kernels::IcbExecutor::new(ICB_RECORD_STEPS));
    let icb_installed_at = if args.arena { record_steps } else { 0 };

    let mut tokens: Vec<u32> = Vec::new();
    for step in 0..args.n {
        let next = logits_processor.sample(&logits).context("sampling")?;
        if eos_ids.contains(&next) {
            eprintln!(
                "note: EOS at decode step {step}; traced {} decode steps",
                steps
                    .iter()
                    .filter(|s| s.label.starts_with("decode"))
                    .count()
            );
            break;
        }
        tokens.push(next);

        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;

        // Drain anything sampling left queued, so it is attributed to the step
        // it belongs to rather than to the window opened next.
        trace_window(&device)?;

        if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
            if step == 0 {
                dev.begin_arena_recording();
            } else if step < record_steps {
                // Close the previous recorded step and open another, keeping the
                // first step's sizes so the two can be compared.
                dev.next_arena_recording_step();
            }
            dev.begin_decode_step();
        }

        // Install once, at the first step whose operands are stable.
        if let (Some(exec), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
            if step == icb_installed_at {
                dev.set_executor(std::sync::Arc::new(
                    candle::kernels::metal::ExecutorSlot::Custom(exec.clone()),
                ));
            }
        }

        let label = format!("decode[{step}] kv_len={kv_len}");
        trace::set_region(Some(label.clone()));
        trace::set_recording(true);
        logits = model
            .forward(&input, kv_len, &mut cache)
            .context("decode forward pass")?
            .squeeze(0)?;
        trace_window(&device)?;
        trace::set_recording(false);

        // Close the ICB step here: after the forward pass has drained and
        // *before* the next iteration samples, so an ICB step is exactly the
        // traced forward pass and nothing else.
        //
        // The placement is load-bearing and the obvious alternatives are both
        // wrong. Closing at the top of the loop puts the previous iteration's
        // `logits_processor.sample` inside the window -- measured, that made a
        // step 556 dispatches against another's 554, because sampling adds a
        // `softmax_f32` and an `affine_f32`. Positions are compared by index, so
        // a two-dispatch shift pairs unrelated dispatches and *every* position
        // reads as varying: the run reported 548 of 556 excluded as `varies`,
        // for kernels the same trace classifies as stable. The executor's own
        // dispatch-count guard is what turned the first form into an error
        // rather than a plausible zero-coverage result, and the second form is
        // what the count alone would not have caught.
        if let (Some(exec), Some(dev)) = (icb_executor.as_ref(), metal_device.as_ref()) {
            if step >= icb_installed_at {
                exec.end_step(dev.metal_device())
                    .map_err(|e| anyhow::anyhow!("closing ICB step {step}: {e}"))?;
            }
        }

        if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
            dev.end_decode_step();
            if step == record_steps - 1 {
                let excluded = dev.arena_recording_excluded();
                let by_test = dev.arena_recording_excluded_by_test();
                let plan = dev
                    .finish_arena_recording(layout)
                    .context("arena recording produced no allocations")?;
                let (covered, total) = plan.covered();
                if let Some((n_excl, n_all)) = excluded {
                    eprintln!(
                        "arena: {n_all} allocations recorded over {record_steps} steps, \
                         {n_excl} excluded as session state"
                    );
                    // Split by detector, because neither subsumes the other and
                    // a zero on either side is a result. Size growth finds the
                    // KV cache; cross-step liveness finds the conv state, which
                    // never grows and which the size test therefore admits
                    // (`DESIGN.md` §5.7, §9.2c). The two overlap, so they do not
                    // sum to `n_excl`.
                    if let Some((by_size, by_step)) = by_test {
                        eprintln!(
                            "arena:   {by_size} caught by size growth (kv_len), \
                             {by_step} by outliving the step (session state at a fixed size)"
                        );
                    }
                }
                eprintln!(
                    "arena: {covered} of {total} ordinals served -> {} slots, {} B ({:?})",
                    plan.slots().len(),
                    plan.arena_bytes(),
                    layout,
                );
                for (i, s) in plan.slots().iter().enumerate() {
                    eprintln!("  slot {i:>2}  offset {:>7}  size {:>7}", s.offset, s.size);
                }
                let monotone = plan.is_bump_reproducible();
                dev.install_arena(plan, layout)
                    .context("installing the arena")?;

                if gpu_offsets {
                    // Checked against the plan that was actually recorded, not
                    // against the layout's name: the property a cursor needs is
                    // that the offsets increase, and asserting it here means a
                    // plan that somehow is not monotone fails before a wrong
                    // offset can be bound rather than after (`DESIGN.md` §9.3 --
                    // there is no other detector).
                    if !monotone {
                        anyhow::bail!(
                            "the recorded plan's offsets are not monotone, so a bump allocator \
                             cannot reproduce them"
                        );
                    }
                    let served = dev
                        .install_gpu_arena_offsets()
                        .context("computing arena offsets on the GPU")?;
                    // The engagement check §2.4 requires. `arena_offsets()`
                    // reads `Gpu` only after the table was verified equal to the
                    // plan element-wise, so this line is evidence that the GPU
                    // allocator ran and agreed -- not that a flag was passed.
                    eprintln!(
                        "arena: offsets computed on the GPU and verified against the plan \
                         ({served} ordinals); source now {:?}",
                        dev.arena_offsets()
                    );
                }
            }
        }

        let dispatches = trace::take_dispatches();
        steps.push(Step { label, dispatches });
        kv_len += 1;
    }

    if let (true, Some(dev)) = (args.arena, metal_device.as_ref()) {
        if let Some(c) = dev.arena_counters() {
            println!(
                "\n== arena ==\noffset source {:?}  gpu_verified {}",
                dev.arena_offsets().unwrap_or_default(),
                c.gpu_verified
            );
            println!(
                "steps {}  offers {}  hits {}  declined(session) {}  declined(size) {}  declined(past plan) {}",
                c.steps, c.offers, c.hits, c.declined_session, c.declined_size, c.declined_exhausted
            );
            let (_, private) = dev.pool_counters();
            println!(
                "private pool: lookups {}  hits {}  allocations {}",
                private.lookups, private.hits, private.allocations
            );
        }
    }

    trace::set_region(None);

    let decode_steps: Vec<&Step> = steps
        .iter()
        .filter(|s| s.label.starts_with("decode"))
        .collect();

    anyhow::ensure!(
        decode_steps.len() >= 2,
        "need at least 2 decode steps to diff; got {}",
        decode_steps.len()
    );

    println!("== dispatch counts per decode token ==");
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for s in &decode_steps {
        *counts.entry(s.dispatches.len()).or_default() += 1;
        println!("{:<28} {} dispatches", s.label, s.dispatches.len());
    }
    if let Some(prefill) = steps.iter().find(|s| s.label == "prefill") {
        println!(
            "{:<28} {} dispatches (not compared against decode)",
            prefill.label,
            prefill.dispatches.len()
        );
    }
    println!("\ndistinct decode dispatch counts: {counts:?}");

    // Barriers per token -- `DESIGN.md` §9.2e's standing check on arena work.
    //
    // `MTLDispatchType::Concurrent` means dispatches within an encoder overlap
    // and the GPU does not drain between them (§3.5), so serialization comes
    // from barriers. A layout that stabilises buffer identity while multiplying
    // this number is a regression wearing the acceptance criterion as a
    // disguise, and the two numbers therefore belong side by side.
    //
    // These are *observed*, not simulated: `auto_barrier` reports each barrier
    // it emits. Encoder sessions are reported beside them because hazard state
    // is per encoder -- candle opens a new one every
    // `CANDLE_METAL_COMPUTE_PER_BUFFER` dispatches (50 by default), and a
    // barrier orders nothing across that boundary.
    println!("\n== barriers per decode token ==");
    let mut barrier_counts: BTreeMap<usize, usize> = BTreeMap::new();
    for s in &decode_steps {
        let barriers = s.dispatches.iter().filter(|d| d.barrier).count();
        let encoders = s
            .dispatches
            .iter()
            .map(|d| d.encoder)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        *barrier_counts.entry(barriers).or_default() += 1;
        println!(
            "{:<28} {barriers:>4} barriers / {} dispatches   ({encoders} encoder sessions)",
            s.label,
            s.dispatches.len()
        );
    }
    println!("distinct decode barrier counts: {barrier_counts:?}");
    // The last step, not step 1. Under `--arena` the first `record_steps`
    // decode steps run *before* the arena is installed, so step 1 reports the
    // classical count in every arm and would make the arena look free. Reading
    // the last step is what makes this number the one §9.2e asks for.
    if let Some(steady) = decode_steps.last() {
        let barriers = steady.dispatches.iter().filter(|d| d.barrier).count();
        println!(
            "steady-state ({}): {barriers} barriers over {} dispatches ({:.1} %)",
            steady.label,
            steady.dispatches.len(),
            100.0 * barriers as f64 / steady.dispatches.len().max(1) as f64
        );

        // Per-kind attribution -- the artifact no previous instrument could
        // produce (issue #185). §11.3p had to attribute the 505 by *position*
        // (covered non-head, run head, gap) because the direction was computed
        // at the two emission sites and discarded into a bool.
        //
        // The columns do NOT partition the total: a barrier is a latch, so one
        // may be owed to several directions at once and is counted in each.
        // `only` is the partition -- barriers attributable to exactly one
        // direction -- and it is the column that answers "these N are WAR, and
        // WAR is the one a different layout would remove."
        use candle::kernels::metal::encoder::HazardKind;
        let mut any = [0usize; 3];
        let mut only = [0usize; 3];
        let mut multi = 0usize;
        let mut unattributed = 0usize;
        for d in steady.dispatches.iter().filter(|d| d.barrier) {
            let ks: Vec<HazardKind> = d.kinds.iter().collect();
            if ks.is_empty() {
                unattributed += 1;
                continue;
            }
            for k in &ks {
                let i = match k {
                    HazardKind::Raw => 0,
                    HazardKind::Waw => 1,
                    HazardKind::War => 2,
                };
                any[i] += 1;
                if ks.len() == 1 {
                    only[i] += 1;
                }
            }
            if ks.len() > 1 {
                multi += 1;
            }
        }
        println!("  by direction (a barrier owed to two is counted in each):");
        for (i, name) in ["RAW", "WAW", "WAR"].iter().enumerate() {
            println!(
                "    {name}: {:>4} barriers  ({:>4} owed to {name} alone)",
                any[i], only[i]
            );
        }
        println!("    owed to more than one direction: {multi}");
        if unattributed > 0 {
            // Not expected: a barrier fires only because a hazard set the latch,
            // so an unattributed one means the kinds were drained somewhere the
            // latch was not. Reported rather than hidden -- it would be a defect
            // in this change, and a silent zero is how it would survive.
            println!("    UNATTRIBUTED: {unattributed}  <- expected 0; see issue #185");
        }
    }

    // Kernel histogram: what the per-token dispatch budget is actually spent on.
    // §11.2 estimates ~240 from 30 layers x ~8 ops; the issue asks for the
    // counted number rather than the inherited estimate.
    println!("\n== kernels per decode token (step 1) ==");
    let mut per_kernel: BTreeMap<&str, usize> = BTreeMap::new();
    for d in &decode_steps[1].dispatches {
        *per_kernel.entry(d.pipeline.as_str()).or_default() += 1;
    }
    let mut hist: Vec<(&&str, &usize)> = per_kernel.iter().collect();
    hist.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, n) in hist {
        println!("{n:>5}  {name}");
    }

    // Consecutive diffs. Step 0 is included in the report but the verdict is
    // taken over steps >= 1: the first decode token follows prefill and can
    // legitimately differ (cache warm-up, first-use allocations) without saying
    // anything about whether steady-state decode is replayable.
    println!("\n== consecutive diffs ==");
    let mut verdicts: Vec<(String, Diff)> = Vec::new();
    for pair in decode_steps.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let diff = compare(a, b);
        println!("{:<28} -> {:<28} {:?}", a.label, b.label, diff);
        verdicts.push((format!("{} -> {}", a.label, b.label), diff));
    }

    // Which dispatch positions actually move, aggregated over every steady-state
    // pair rather than reported from the first difference found. The first
    // difference is whichever position happens to come earliest, which says
    // nothing about how much of the sequence is unstable or why.
    println!("\n== what varies across steady-state decode steps ==");
    // Index of the step the comparison baselines on.
    //
    // 1 normally: `decode[0]` follows prefill and is not steady state. With an
    // arena, the steps the plan was recorded from ran on the pool -- the arena
    // does not exist until its plan does -- so the baseline moves past them.
    // Comparing an arena-served step against a pool-served one would report
    // variance that is an artefact of when the arena was installed rather than a
    // property of the arena.
    let base_idx = if args.arena { record_steps + 1 } else { 1 };
    anyhow::ensure!(
        decode_steps.len() > base_idx + 1,
        "need at least {} decode steps to compare; got {}",
        base_idx + 2,
        decode_steps.len()
    );
    println!(
        "baseline: {} (comparing {} steps)",
        decode_steps[base_idx].label,
        decode_steps.len() - base_idx
    );
    let n = decode_steps[base_idx].dispatches.len();
    let uniform_len = decode_steps
        .iter()
        .skip(base_idx)
        .all(|s| s.dispatches.len() == n);
    if uniform_len {
        let mut grid_varies: Vec<usize> = Vec::new();
        let mut offset_varies: Vec<usize> = Vec::new();
        let mut buffer_varies: Vec<usize> = Vec::new();
        let mut kernel_varies: Vec<usize> = Vec::new();
        for i in 0..n {
            let base = &decode_steps[base_idx].dispatches[i];
            let rest = || {
                decode_steps
                    .iter()
                    .skip(base_idx + 1)
                    .map(|s| &s.dispatches[i])
            };
            if rest().any(|d| d.pipeline != base.pipeline) {
                kernel_varies.push(i);
            }
            if rest().any(|d| {
                d.grid != base.grid
                    || d.threadgroup != base.threadgroup
                    || d.by_threadgroups != base.by_threadgroups
            }) {
                grid_varies.push(i);
            }
            if rest().any(|d| {
                d.bindings
                    .iter()
                    .map(|b| b.offset)
                    .ne(base.bindings.iter().map(|b| b.offset))
            }) {
                offset_varies.push(i);
            }
            if rest().any(|d| {
                d.bindings
                    .iter()
                    .map(|b| b.buffer_id)
                    .ne(base.bindings.iter().map(|b| b.buffer_id))
            }) {
                buffer_varies.push(i);
            }
        }

        let summarize = |what: &str, idx: &[usize]| {
            println!("{what:<22} {:>4} of {n} dispatches", idx.len());
            if !idx.is_empty() {
                let mut by_kernel: BTreeMap<&str, usize> = BTreeMap::new();
                for &i in idx {
                    *by_kernel
                        .entry(decode_steps[base_idx].dispatches[i].pipeline.as_str())
                        .or_default() += 1;
                }
                let mut v: Vec<_> = by_kernel.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
                for (k, c) in v {
                    println!("{:>26}  {c:>4}  {k}", "");
                }
            }
        };
        summarize("kernel name varies:", &kernel_varies);
        summarize("grid varies:", &grid_varies);
        summarize("binding offset varies:", &offset_varies);
        summarize("buffer identity varies:", &buffer_varies);

        // A grid that grows by a constant amount per token is `kv_len` reaching
        // the dispatch, which is the shape §6.2 predicts for the KV `cat`.
        if let Some(&i) = grid_varies.first() {
            let seq: Vec<usize> = decode_steps
                .iter()
                .skip(1)
                .map(|s| s.dispatches[i].grid.1)
                .collect();
            let deltas: Vec<i64> = seq.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect();
            println!(
                "\nexample growing dispatch: #{i} {} grid.height {:?}",
                decode_steps[base_idx].dispatches[i].pipeline,
                &seq[..seq.len().min(8)]
            );
            println!("  per-token delta: {:?}", &deltas[..deltas.len().min(7)]);
        }
    } else {
        println!("dispatch counts differ between steps; per-position comparison skipped");
    }

    let steady: Vec<&(String, Diff)> = verdicts.iter().skip(1).collect();
    let worst = steady
        .iter()
        .map(|(_, d)| *d)
        .max()
        .unwrap_or(Diff::Identical);
    let verdict = match worst {
        Diff::Identical => "IDENTICAL",
        Diff::BufferIdentityOnly => "BUFFER_IDENTITY_ONLY",
        Diff::GridOrOffsetScaling => "STABLE_KERNEL_SEQUENCE_VARYING_GRIDS",
        Diff::KernelSequence => "STRUCTURAL",
    };

    println!("\n== verdict ==");
    println!(
        "steady-state decode (steps 1..{}): {verdict}",
        decode_steps.len() - 1
    );
    println!(
        "first transition (decode[0] -> decode[1]): {:?}",
        verdicts[0].1
    );
    println!(
        "dispatches per decode token: {}",
        if counts.len() == 1 {
            format!("{}", decode_steps[1].dispatches.len())
        } else {
            format!("varies: {counts:?}")
        }
    );
    println!(
        "dispatches seen while not recording: {}",
        trace::skipped_count()
    );

    let dump: Vec<usize> = args
        .dump_steps
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .context("parsing --dump-steps")?;
    for i in dump {
        if let Some(s) = decode_steps.get(i) {
            println!("\n{}", render(s));
        }
    }

    if let Some(path) = args.dump_all {
        let mut out = String::new();
        for s in &steps {
            out.push_str(&render(s));
            out.push('\n');
        }
        std::fs::write(&path, out).with_context(|| format!("writing {}", path.display()))?;
        println!("full trace written to {}", path.display());
    }

    // What the ICB executor covered, with the exclusions attributed.
    //
    // A bare count would not be reportable: "431 of 554" says nothing about
    // *why* the rest were left, and §9.2h's lesson is that the reason decides
    // which phase removes them. An exclusion for `inline-constants` is §3.7c and
    // wants a packed sibling; one for `varies` is §6.2's population and wants
    // in-place KV append or indirect dispatch. They are different work.
    if let Some(exec) = icb_executor.as_ref() {
        let cov = exec.coverage();
        println!("\n== icb replay ==");
        println!("replaying:                {}", exec.is_replaying());
        println!("positions per decode step: {}", cov.positions);
        println!(
            "covered:                  {} ({:.1} %)",
            cov.covered,
            if cov.positions == 0 {
                0.0
            } else {
                100.0 * cov.covered as f64 / cov.positions as f64
            }
        );
        println!("execute calls per step:   {} (one per run)", exec.runs());
        println!("excluded, varies:         {}", cov.varies);
        println!("excluded, inline consts:  {}", cov.inline_constants);
        println!("excluded, no pipeline:    {}", cov.no_pipeline);
        // Zero is the expected value. Nonzero means a replayed position stopped
        // matching its recording mid-run, so coverage is lower than the plan
        // claims and the figure above overstates what ran.
        println!("stale positions:          {}", exec.stale_positions());
        println!("poisoned runs:            {}", exec.poisoned_runs());
        // One is expected: the window opened at install time is partial.
        println!("discarded windows:        {}", exec.discarded_windows());
        // The ordering auto_barrier would have emitted between replayed
        // dispatches, re-expressed on the commands themselves.
        println!("barriers encoded in icb:  {}", exec.encoded_barriers());
        if !cov.covered_by_kernel.is_empty() {
            println!("  covered, by kernel:");
            let mut rows: Vec<_> = cov.covered_by_kernel.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            for (name, n) in rows {
                println!("    {n:5}  {name}");
            }
        }
        if !cov.excluded_by_kernel.is_empty() {
            println!("  excluded, by kernel and reason:");
            let mut rows: Vec<_> = cov.excluded_by_kernel.iter().collect();
            rows.sort_by(|a, b| b.1.cmp(a.1));
            for ((name, reason), n) in rows {
                println!("    {n:5}  {name:<48} {reason}");
            }
        }
    }

    // The token stream, so two runs can be compared by *what they computed*
    // rather than only by what they dispatched.
    //
    // This is what makes §9.3's parity check an execution comparison: the packed
    // layout aliases values that the non-aliasing reference keeps apart, so if
    // the offset arithmetic is wrong the two runs compute different things.
    // Under `HazardTrackingModeUntracked` there is no other detector -- a bad
    // offset is a silent wrong answer, not an error. Sampling is deterministic
    // given the seed (§2.3.8), so a difference here is the arena's.
    println!("\ngenerated {} tokens", tokens.len());
    println!(
        "tokens: {}",
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
