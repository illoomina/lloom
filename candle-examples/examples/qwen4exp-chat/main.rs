//! Qwen3.8-Flash-Next light gate (issue #747): load the real GGUF, decode a
//! prompt greedily, print the text, and hash the token stream — the digest +
//! text pair on the real model (CONTRIBUTING §3.1a), run N times from a fresh
//! cache so the digest also answers "is the build run-to-run identical".
//!
//! `--hotlist <file>` (with `--experts stream`) warms every layer's pool
//! from the expert frequency profile of earlier runs before the first token
//! (§20.3 R2) and writes the profile back at exit with this run merged in.
//! `--experts stream` runs the routed experts through the streamed tier
//! (issue #749): a per-layer slot pool of `--slots` experts filled from the
//! shards by `--loader pread|io`; the gate is that its digests equal the
//! resident arm's. `--ple-cache-rows` puts a hot-row cache in front of the
//! N-gram table. A sharded checkpoint is found from its first shard.
//!
//! No timing beyond the wall clock of load and decode (phase-0 is
//! correctness-only); no perf harness. Exits non-zero when `--expect` is not a
//! substring of the output, or when the runs' digests disagree.
//!
//! ```bash
//! cargo run --release --features metal --example qwen4exp-chat -- --runs 3
//! cargo run --release --features metal --example qwen4exp-chat -- --experts stream --slots 64 --runs 2
//! ```

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use clap::Parser;
use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::loader::{LoadOptions, SlotBudget, StreamLoader, StreamOptions};
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::{
    Cache, ExpertResidency, MemCheck, MtpDrafter, Qwen4Exp, Resident, Streamed,
};
use lloom::nn::expert_stream::HasStreamStats;
use lloom::nn::memory::{MemGauge, MemReport, MemSnapshot};
use lloom::nn::model::CausalLM;
use lloom::nn::observer::{ForwardObserver, HookPoint};

/// `CANDLE_METAL_PROFILE=1`: one line per phase from the backend's profiler
/// (§13.1a) — dispatches, command buffers, GPU-busy union against the wall,
/// and the dispatch count per kernel label — then a reset for the next phase.
/// Inert otherwise.
fn profile_line(run: usize, phase: &str, tokens: usize, wall_ms: f64) {
    #[cfg(feature = "metal")]
    {
        use candle::kernels::metal::profile;
        if !profile::enabled() {
            return;
        }
        let s = profile::snapshot();
        let busy = 1e3 * s.gpu_busy_union_s;
        println!(
            "PROFILE run={run} phase={phase} tokens={tokens} wall_ms={wall_ms:.1} gpu_busy_ms={busy:.1} gpu_span_ms={:.1} busy_over_wall={:.3} dispatches={} per_token={:.0} encoders={} command_buffers={} commits={} commit_ms={:.1} end_enc_ms={:.1} residency_ms={:.1} cb_commit_ms={:.1} create_cb_ms={:.1}",
            1e3 * s.gpu_span_s,
            busy / wall_ms.max(1e-9),
            s.dispatches,
            s.dispatches as f64 / tokens.max(1) as f64,
            s.encoders,
            s.command_buffers,
            s.commits,
            s.commit_ns as f64 / 1e6,
            s.end_enc_ns as f64 / 1e6,
            s.residency_ns as f64 / 1e6,
            s.cb_commit_ns as f64 / 1e6,
            s.create_cb_ns as f64 / 1e6,
        );
        let mut by = s.by_label.clone();
        by.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<String> = by
            .iter()
            .take(24)
            .map(|(l, n)| format!("{l}={:.1}", *n as f64 / tokens.max(1) as f64))
            .collect();
        println!(
            "PROFILE run={run} phase={phase} labels_per_token {}",
            top.join(" ")
        );
        // GPU time by label: where the busy time goes (ms per token).
        let mut gt = profile::gpu_time_by_label_detailed();
        gt.sort_by(|a, b| {
            b.total_s
                .partial_cmp(&a.total_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top: Vec<String> = gt
            .iter()
            .take(24)
            .map(|k| format!("{}={:.3}", k.label, 1e3 * k.total_s / tokens.max(1) as f64))
            .collect();
        println!(
            "PROFILE run={run} phase={phase} gpu_ms_per_token_by_label {}",
            top.join(" ")
        );
        profile::reset();
    }
    #[cfg(not(feature = "metal"))]
    let _ = (run, phase, tokens, wall_ms);
}
use lloom::nn::speculator::{Anchor, Draft, Probes};
use lloom::nn::{DecodeOptions, Session};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[path = "../lfm2-determinism/sha256.rs"]
mod sha256;

#[derive(Parser, Debug)]
#[command(about = "Qwen3.8-Flash-Next real-model light gate: digest + text, greedy")]
struct Args {
    /// The first GGUF shard (`-00001-of-0000N`); the siblings are found next
    /// to it. Default: the HF cache's ISTA-DASLab IQ3_XXS snapshot.
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// tokenizer.json. Default: the HF cache's Qwen/Qwen3.8-Flash-Next snapshot.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// The GGUF shard holding the N-gram PLE table (`per_layer_token_embd`).
    /// Default: the shard among the discovered siblings that holds it.
    #[arg(long)]
    ple_shard: Option<PathBuf>,

    /// Load without the N-gram PLE (the #747 baseline the digests were
    /// recorded on).
    #[arg(long)]
    no_ple: bool,

    /// Rows of the PLE hot-row cache (0 = read the memory map every gather).
    #[arg(long, default_value_t = 0)]
    ple_cache_rows: usize,

    /// The GGUF carrying the MTP head (`blk.{n_layers}.*` + `nextn.*`, ds4's
    /// naming — `qwen4exp-mtp-quantize` writes one). After the greedy runs the
    /// prompt is decoded once more through `Session::decode_speculative` with
    /// the `MtpDrafter`; its stream must equal the greedy stream, and the
    /// draft acceptance rate is printed (`MTP …`).
    #[arg(long)]
    mtp: Option<PathBuf>,

    /// Tokens the MTP drafter proposes per round (`--mtp`): 1 from the head
    /// once, `K` by re-applying the one MTP layer on its own output. The
    /// verify runs `K + 1` positions; byte-identical to greedy whatever `K`.
    #[arg(long, default_value_t = 1)]
    mtp_k: usize,

    /// Rollback probe: after the prompt, run N verify rounds whose draft is
    /// always wrong (each round rewinds and replays its anchor), then compare
    /// every state channel's checksum against a fresh cache that decoded the
    /// same tokens greedily. Prints the channels that differ. 0 = off.
    #[arg(long, default_value_t = 0)]
    rollback_probe: usize,

    /// How the routed experts are held: `resident` (every expert in RAM) or
    /// `stream` (a per-layer slot pool filled from the shards).
    #[arg(long, default_value = "resident")]
    experts: String,

    /// Streamed experts per layer per projection: a count, or `auto` (sized
    /// from `recommendedMaxWorkingSetSize × --memory-fraction` minus the
    /// resident tier; the `SLOTPLAN` line says what it resolved to).
    #[arg(long, default_value = "64")]
    slots: String,

    /// The fraction of the device's recommended working set `--slots auto`
    /// may plan for.
    #[arg(long, default_value_t = 0.65)]
    memory_fraction: f64,

    /// The streaming loader primitive: `pread` (read + compute-encoder copy)
    /// or `io` (`MTLIOCommandQueue`).
    #[arg(long, default_value = "pread")]
    loader: String,

    /// Decode `N` sequences of the prompt as one batch (`Session::
    /// decode_batched` over `BatchedCausalLM::forward_rows`: one weight
    /// sweep per step for all rows, the state per row) and report the
    /// aggregate rate. 1 = the single-sequence loop.
    #[arg(long, default_value_t = 1)]
    rows: usize,

    /// Run even `--rows 1` through the batched driver (the batched arm's
    /// own baseline).
    #[arg(long)]
    batched_driver: bool,

    /// Accepted for the recorded commands: the driver is the default
    /// gate since P29 (7) (`Session::decode` on the blocking arm — the loop
    /// `lloom serve` runs). See `--legacy-loop`.
    #[arg(long)]
    driver: bool,

    /// The driver's readback arm: `blocking` (the default, the id read back
    /// per step), `ring-fixed` (the device-resident token polled, lookahead 1,
    /// a commit every 8 steps) or `ring` (the same under the adaptive commit,
    /// the crown's cadence). On a ring arm each decode forward's PLE rows
    /// come from the host fill (`--host-fill`).
    #[arg(long, default_value = "blocking")]
    readback: String,

    /// The PLE's host fill on a ring arm (`expert-streaming.md` §20.13): a
    /// producer thread fills each decode step's rows from the published
    /// token and the GPU waits for them. `false` is the A/B arm: the forward
    /// reads its input id back per step.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    host_fill: bool,

    /// The harness's own per-token loop (the full logits read back and
    /// hashed a token: the logits digest, `--state-digest`, `--probe`,
    /// `--margins`, `--prompt-split`, `--rollback-probe`, `--mtp-k`) —
    /// the instrument arm, 40 % slower than the driver (P29). Off by
    /// default since P29 (7): one loop, the driver's.
    #[arg(long)]
    legacy_loop: bool,

    /// `--rows N`: the rows' recurrent state in one slot-indexed pool (the
    /// DeltaNet step as one launch per kernel over the rows); off, every
    /// row keeps its own slabs and the step launches per row.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    rows_pool: bool,

    /// The expert hotlist file (`--experts stream`): read at load to warm
    /// every layer's pool before the first token (§20.3 R2; a missing file
    /// warms nothing), written at exit with this run's usage merged in.
    #[arg(long)]
    hotlist: Option<PathBuf>,

    /// Pin the hotlist's head: this many of each layer's most-used experts
    /// are never evicted (`--experts stream` with `--hotlist`), so a long
    /// prompt's sweep cannot churn the decode's hot set out of the pool. 0
    /// = warm only.
    #[arg(long, default_value_t = 0)]
    pin: usize,

    /// Planned passes (DESIGN.md §9.2aa): every prefill tile and decode
    /// step keyed by shape, its activations served from one base after a
    /// first recorded sight — the streamed tier's and the routed GEMMs'
    /// allocations declared escapes. Prints `[pass plans: …]` at exit.
    #[arg(long)]
    pass_plans: bool,

    /// How the prompt is run: `chunk` (Kernel B + chunk attention, the
    /// reference prefill graph) or `replay` (the decode recurrence one position
    /// at a time, #747's path).
    #[arg(long, default_value = "chunk")]
    prefill: String,

    /// The routed experts' activation dtype: `auto` (the contract's: `f16`
    /// under `fast`, `f32` under `deterministic`, this gate's pin), `f32`
    /// (the byte-identical routed arm) or `f16` (the `(f16, f32, f16)` arm).
    #[arg(long, default_value = "auto")]
    moe_act: String,

    #[arg(
        long,
        default_value = "What is the capital of France? Answer in one short sentence."
    )]
    prompt: String,

    /// Send the prompt raw (no chat template, no thinking-off prefix).
    #[arg(long)]
    raw: bool,

    /// Substring the output must contain (case-insensitive); exits 1 otherwise.
    #[arg(long)]
    expect: Option<String>,

    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// Fresh-cache repetitions; all must produce one token digest.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long)]
    cpu: bool,

    /// After the prompt and after generation, print a checksum per state
    /// channel and layer (`STATE`), and a per-step logits checksum (`STEP`),
    /// so two runs can be diffed for where they first diverge.
    #[arg(long)]
    state_digest: bool,

    /// The prompt's tile: every prefill forward is at most this many
    /// positions, each admitted (`prepare_pass`) and paced at depth 2 so a
    /// long prompt's activations are never all pending at once (DESIGN.md
    /// §9.5m, the per-request guard). 0 = the whole prompt in one forward
    /// (the model still chunks at 512 inside it, unpaced).
    #[arg(long, default_value_t = 512)]
    prefill_tile: usize,

    /// The streamed tier's prefetch thread (the `pread` arm): a prefill
    /// tile stages each layer's likely misses while the layer before it
    /// runs. `--prefetch false` is the A/B arm (every miss read at the layer).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    prefetch: bool,

    /// The prefetch hints on the `io` arm: the hinted experts take the
    /// pool's free slots at hint time and load on the IO queue, waited for
    /// at the layer (the in-flight residency entry). Measured net-negative
    /// on this pool (issue #749 P26); the A/B arm.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    io_hints: bool,

    /// The prefetch hint on a prefill pass (§20.3): 0 = the usage profile,
    /// 1 = residual speculation (layer l+2's router on the carry entering
    /// layer l, landed after l). Default: `QWEN4EXP_SPEC_AHEAD`, else 0.
    #[arg(long)]
    spec_ahead: Option<usize>,

    /// KV positions per row `--slots auto` reserves beside the slot pool
    /// (with two prefill tiles' transients and the pool's free list): the
    /// session's share of the budget, so a pass fits by construction.
    #[arg(long, default_value_t = 4096)]
    kv_reserve_tokens: usize,

    /// The memory gauge on the `MEM` lines: a 10 ms sampler holding the
    /// peaks over each phase (device allocated, pool pending, footprint).
    #[arg(long)]
    mem_sample: bool,

    /// What a pass the memory budget cannot take does: `report` (a
    /// `MEMCHECK` line on stderr, the arm the transient term is measured
    /// under), `refuse` (stop before anything is written), `off`. Default:
    /// `QWEN4EXP_MEM`, else `refuse`.
    #[arg(long)]
    mem_check: Option<String>,

    /// Feed the prompt in segments ending at these positions (comma list, e.g.
    /// `512,2048`) and, with `--state-digest`, print a `STATE ... phase=at<N>`
    /// checksum after each — a bisection aid locating WHERE in the prompt two
    /// runs first diverge. The model chunks each segment exactly as it chunks
    /// the whole prompt when the splits fall on chunk boundaries.
    #[arg(long, value_delimiter = ',')]
    prompt_split: Vec<usize>,

    /// Print, per generated step, the argmax, the runner-up and their logit
    /// margin (`MARGIN` lines): the discriminator for a token digest that
    /// moved between two arms — a flip at a near-tie is rounding, a flip at
    /// a wide margin is a defect (§2.3.5a).
    #[arg(long)]
    margins: bool,

    /// Print per-layer activation statistics (finite count, max |x|, mean |x|)
    /// of the HC carry for the FIRST prompt token, through the observer seam,
    /// plus the final logits' statistics. A bisection aid, not a gate.
    #[arg(long)]
    probe: bool,

    /// Decode this token sequence instead of the greedy one (`--legacy-loop`):
    /// every arm walks the same positions, so two `--dump-logits` files
    /// subtract position by position (`tools/logit-delta`). The argmax is still
    /// taken and counted against the forced id (`overrides`). One id per line
    /// or a comma / space list — the `TOKENS` line feeds back in.
    #[arg(long)]
    force_tokens: Option<PathBuf>,

    /// Write every logits row the legacy loop digests (the prompt's last
    /// position, then each step) as `step <i> <f32 bits hex>…` — the format
    /// `tools/logit-delta` reads.
    #[arg(long)]
    dump_logits: Option<PathBuf>,

    /// The model's runtime axes (`--kv-quant off|f16|int8|int4`, …): the
    /// `#[derive(Config)]` flags of `qwen4exp::Config`.
    #[command(flatten)]
    axes: lloom::models::qwen4exp::ConfigArgs,
}

/// Token ids from a `--force-tokens` file: whitespace / comma separated, `#`
/// comments.
fn read_forced_tokens(path: &std::path::Path) -> Result<Vec<u32>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.split('#').next().unwrap_or("");
        for f in line.split([',', ' ', '\t']).filter(|f| !f.is_empty()) {
            out.push(
                f.parse()
                    .with_context(|| format!("{f:?} is not a token id"))?,
            );
        }
    }
    anyhow::ensure!(!out.is_empty(), "{}: no token ids", path.display());
    Ok(out)
}

/// One `--dump-logits` row.
fn dump_row(
    sink: &mut Option<std::io::BufWriter<std::fs::File>>,
    step: usize,
    t: &Tensor,
) -> Result<()> {
    use std::io::Write;
    let Some(sink) = sink.as_mut() else {
        return Ok(());
    };
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let mut line = String::with_capacity(v.len() * 9 + 16);
    line.push_str(&format!("step {step}"));
    for x in v {
        line.push_str(&format!(" {:08x}", x.to_bits()));
    }
    line.push('\n');
    sink.write_all(line.as_bytes())?;
    Ok(())
}

/// Activation statistics through the `ForwardObserver` seam: verbose prints
/// every point; otherwise it folds a step's points into (first non-finite
/// layer, max |x| over the HC carry) for a per-step summary line.
struct Probe {
    verbose: bool,
    first_nonfinite: Option<(usize, lloom::nn::observer::HookKind)>,
    max_abs: f32,
}

impl Probe {
    fn new(verbose: bool) -> Self {
        Self {
            verbose,
            first_nonfinite: None,
            max_abs: 0.0,
        }
    }
    fn reset(&mut self) {
        self.first_nonfinite = None;
        self.max_abs = 0.0;
    }
}

/// One `MEM` line: the reading now (and the phase's peaks when a gauge ran).
fn mem_line(run: usize, phase: &str, device: &Device, report: Option<MemReport>) {
    let gb = |b: u64| b as f64 / 1e9;
    let now = MemSnapshot::read(device);
    let mut line = format!(
        "MEM run={run} phase={phase} device_gb={:.2} pool_live_gb={:.2} pool_free_gb={:.2} pool_pending_gb={:.2} phys_gb={:.2} working_set_gb={:.2}",
        gb(now.device_allocated),
        gb(now.pool_live),
        gb(now.pool_free),
        gb(now.pool_pending),
        gb(now.phys_footprint),
        gb(now.working_set),
    );
    #[cfg(feature = "metal")]
    if let Device::Metal(d) = device {
        // The residency set's traffic (cumulative): how many `commit`s the
        // command-buffer cuts and the pool's misses made, and the set's size
        // — the seam's cost at prefill is this commit (issue #749 P26).
        let c = d.residency_counters();
        let (hs, hp) = d.heap_stats();
        line.push_str(&format!(
            " residency_allocations={} residency_commits={} residency_added={} residency_removed={} heap_chunks={} heap_size_gb={:.2} heap_used_gb={:.2}",
            d.residency_set_len(),
            c.commits,
            c.added,
            c.removed,
            hs.chunks + hp.chunks,
            gb((hs.size + hp.size) as u64),
            gb((hs.used + hp.used) as u64)
        ));
    }
    if let Some(r) = report {
        if let Some(p) = r.peak {
            line.push_str(&format!(
                " peak_device_gb={:.2} peak_pending_gb={:.2} peak_phys_gb={:.2} samples={}",
                gb(p.device_allocated),
                gb(p.pool_pending),
                gb(p.phys_footprint),
                r.samples
            ));
        }
    }
    println!("{line}");
}

/// The pass plans' report (lfm2-chat's line) and the pools' churn.
fn report_pass_plans(device: &Device) {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        let (shared, private) = dev.pool_counters();
        println!(
            "POOLS shared_lookups={} shared_hits={} private_lookups={} private_hits={}",
            shared.lookups, shared.hits, private.lookups, private.hits
        );
        let lb = candle::kernels::metal::late_bind::report();
        if lb.armed {
            println!(
                "LATEBIND releases={} binds={} late={} late_writes={}",
                lb.releases, lb.binds, lb.late, lb.late_writes
            );
        }
        if let (Some(c), Some((keys, base))) = (dev.pass_plan_counters(), dev.pass_plan_report()) {
            let keys: Vec<String> = keys
                .iter()
                .map(|(k, covered, total, escapes)| {
                    // A pass the model bracketed itself is its own key (§9.2aa).
                    let own = match k.opener {
                        candle::metal_backend::PassOpener::Model => "/model",
                        candle::metal_backend::PassOpener::Driver => "",
                    };
                    if *escapes > 0 {
                        format!(
                            "{:?}{own}={covered}/{total}(+{escapes} escaped {:?})",
                            k.kind,
                            dev.pass_plan_escapes(k)
                        )
                    } else {
                        format!("{:?}{own}={covered}/{total}", k.kind)
                    }
                })
                .collect();
            println!(
                "PASSPLANS served={} recorded={} observed={} plans={} evicted={} escapes={} capped={} base_grows={} base_mib={:.1} keys=[{}]",
                c.served,
                c.recorded,
                c.observed,
                c.plans,
                c.evicted,
                c.escapes,
                c.capped,
                c.base_grows,
                base as f64 / (1024.0 * 1024.0),
                keys.join(" ")
            );
        }
    }
    let _ = device;
}

fn mem_check_arm(args: &Args) -> Result<MemCheck> {
    Ok(match args.mem_check.as_deref() {
        None => MemCheck::from_env(),
        Some("off") => MemCheck::Off,
        Some("report") => MemCheck::Report,
        Some("refuse") => MemCheck::Refuse,
        Some(other) => anyhow::bail!("--mem-check {other}: off | report | refuse"),
    })
}

/// `ids` through `forward` in tiles of `tile` positions (0 = one forward),
/// each prepared (`prepare_pass`: the KV grow and the pass admitted before
/// it runs) and paced at depth 2 on Metal — the drivers' `StepPacer` shape,
/// here for the harness's own prompt loop. Returns the last position's
/// logits.
fn prefill_tiled<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cache: &mut Cache,
    ids: &[u32],
    tile: usize,
    device: &Device,
) -> Result<Tensor> {
    use lloom::models::qwen4exp::{prefill_chunk, prefill_pair_on, prefill_tiles};
    // Under the pipelined prefill a tile is `prefill_tiles` chunks — the
    // pass interleaves them a layer apart — and the pacer's depth is 1: the
    // pass IS the tiles in flight the transient constant was measured with.
    let (tile, depth) = if tile != 0 && prefill_pair_on() && tile <= prefill_chunk() {
        (prefill_tiles() * prefill_chunk(), 1)
    } else if tile > prefill_chunk() {
        (tile, 1)
    } else {
        (tile, 2)
    };
    #[cfg(feature = "metal")]
    let mut markers: std::collections::VecDeque<candle::metal_backend::CommandMarker> =
        std::collections::VecDeque::new();
    let mut logits = None;
    let mut off = 0usize;
    while off < ids.len() {
        let left = ids.len() - off;
        let w = if tile == 0 { left } else { left.min(tile) };
        let seg = &ids[off..off + w];
        let input = Tensor::from_vec(seg.to_vec(), (1, w), device)?;
        model.prepare_pass(cache, w)?;
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            while markers.len() >= depth {
                let oldest = markers.pop_front().expect("len >= depth");
                m.wait_marker(&oldest)?;
            }
        }
        logits = Some(model.forward(&input, cache.pos, cache)?);
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            markers.push_back(m.flush_marker()?);
        }
        off += w;
    }
    Ok(logits.expect("a non-empty prompt"))
}

fn stats(t: &Tensor) -> Result<(usize, usize, f32, f32)> {
    let v = t
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let maxabs = v
        .iter()
        .filter(|x| x.is_finite())
        .fold(0f32, |m, x| m.max(x.abs()));
    let meanabs = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .sum::<f32>()
        / v.len().max(1) as f32;
    Ok((v.len(), nonfinite, maxabs, meanabs))
}

impl ForwardObserver for Probe {
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> candle::Result<()> {
        let (n, nonfinite, maxabs, meanabs) =
            stats(value).map_err(|e| candle::Error::Msg(e.to_string()))?;
        if nonfinite > 0 && self.first_nonfinite.is_none() {
            self.first_nonfinite = Some((layer, P::KIND));
        }
        self.max_abs = self.max_abs.max(maxabs);
        if self.verbose {
            println!(
                "PROBE layer={layer} point={:?} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}",
                P::KIND
            );
        }
        Ok(())
    }
    fn wants<P: HookPoint>(&self, _layer: usize) -> bool {
        true
    }
}

fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(base)
        .ok()?
        .flatten()
        .next()
        .map(|e| e.path())
}

/// `first` plus its `-0000k-of-0000N` siblings, in shard order.
fn discover_shards(first: &PathBuf) -> Vec<PathBuf> {
    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    let Some(pos) = name.find("-of-") else {
        return vec![first.clone()];
    };
    // "...-00001-of-00003.gguf": the count follows "-of-".
    let tail = &name[pos + 4..];
    let count: usize = tail.trim_end_matches(".gguf").parse().unwrap_or(1);
    let head = &name[..pos.saturating_sub(5)];
    let mut out = Vec::with_capacity(count);
    for k in 1..=count {
        let sibling = first.with_file_name(format!("{head}{k:05}-of-{count:05}.gguf"));
        if sibling.exists() {
            out.push(sibling);
        }
    }
    if out.is_empty() {
        vec![first.clone()]
    } else {
        out
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    // A gate states its contract: deterministic, unless LLOOM_PROFILE names one.
    candle::tuning::pin_contract(candle::tuning::Contract::Deterministic)
        .map_err(anyhow::Error::msg)?;
    let contract = candle::tuning::gate_contract("qwen4exp-chat").map_err(anyhow::Error::msg)?;
    println!("CONTRACT {}", contract.report());
    let gguf = match &args.gguf {
        Some(p) => p.clone(),
        None => hf_snapshot("models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
            .context("no HF snapshot for the GGUF; pass --gguf")?
            .join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    };
    let tokenizer = match &args.tokenizer {
        Some(p) => p.clone(),
        None => hf_snapshot("models--Qwen--Qwen3.8-Flash-Next")
            .context("no HF snapshot for the tokenizer; pass --tokenizer")?
            .join("tokenizer.json"),
    };
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    // `QWEN4EXP_POOL_POISON=<hex byte>`: the #706 GPU-side free-buffer poison
    // (a `fuzzing`-feature build; inert otherwise). A bisection aid for the
    // long-context divergence: a still-reading command buffer sees the
    // sentinel instead of a plausible stale value.
    #[cfg(feature = "metal")]
    if let (Ok(hex), Device::Metal(m)) = (std::env::var("QWEN4EXP_POOL_POISON"), &device) {
        let byte = u8::from_str_radix(hex.trim_start_matches("0x"), 16)
            .map_err(|e| anyhow::anyhow!("QWEN4EXP_POOL_POISON={hex}: {e}"))?;
        m.set_free_poison(Some(byte));
        eprintln!("pool free-poison armed: 0x{byte:02x}");
    }

    let tokenizer = Tokenizer::from_file(&tokenizer).map_err(anyhow::Error::msg)?;
    let mut eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    eos_ids.dedup();

    let t0 = std::time::Instant::now();
    let mut shards = discover_shards(&gguf);
    if let Some(p) = &args.ple_shard {
        if !shards.contains(p) {
            shards.push(p.clone());
        }
    }
    if let Some(p) = &args.mtp {
        shards.push(p.clone());
    }
    let moe_act = match (args.moe_act.as_str(), contract.is_fast()) {
        ("auto", true) | ("f16", _) => Some(candle::DType::F16),
        ("auto", false) | ("f32", _) => None,
        (other, _) => anyhow::bail!("--moe-act {other}: auto | f32 | f16"),
    };
    let opts = LoadOptions {
        ple_shard: None,
        ple: !args.no_ple,
        moe_act,
        ple_cache_rows: (args.ple_cache_rows > 0).then_some(args.ple_cache_rows),
        mtp: args.mtp.is_some(),
        memory_fraction: Some(args.memory_fraction),
        kv_precision: args
            .axes
            .kv_precision
            .as_deref()
            .map(lloom::models::qwen4exp::config::parse_kv_precision)
            .transpose()
            .map_err(anyhow::Error::msg)?,
    };
    let mem_check = mem_check_arm(&args)?;
    let prefill_mode = match args.prefill.as_str() {
        "chunk" => PrefillMode::Chunk,
        "replay" => PrefillMode::Replay,
        other => anyhow::bail!("--prefill {other}: chunk | replay"),
    };
    let loader = match args.loader.as_str() {
        "pread" => StreamLoader::Pread,
        "io" => StreamLoader::Io,
        other => anyhow::bail!("--loader {other}: pread | io"),
    };
    let text = if args.raw {
        args.prompt.clone()
    } else {
        // The checkpoint's chat template with thinking disabled
        // (`enable_thinking=false` → an empty think block after the assistant tag).
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            args.prompt
        )
    };
    let prompt_ids: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    match args.experts.as_str() {
        "resident" => {
            let mut model: Qwen4Exp<Resident> =
                Qwen4Exp::from_gguf_shards(&shards, &device, &opts)?;
            model.apply_axes(&args.axes)?;
            model.set_prefill_mode(prefill_mode);
            model.set_memory_fraction(args.memory_fraction);
            model.set_mem_check(mem_check);
            model.set_host_fill(args.host_fill);
            if args.rows > 1 || args.batched_driver {
                rows_gate(&model, &args, &device, &prompt_ids, &eos_ids)
            } else if !args.legacy_loop {
                driver_gate(&model, &args, &shards, &device, &prompt_ids, &eos_ids, t0)
            } else {
                gate(
                    &model,
                    &args,
                    &shards,
                    &device,
                    &tokenizer,
                    &prompt_ids,
                    &eos_ids,
                    t0,
                )
            }
        }
        "stream" => {
            #[cfg(feature = "metal")]
            if args.pass_plans {
                if let Device::Metal(m) = &device {
                    m.enable_pass_plans(candle::metal_backend::ArenaLayout::Packed);
                }
            }
            let slots = match args.slots.as_str() {
                "auto" => SlotBudget::Auto {
                    fraction: args.memory_fraction,
                    rows: args.rows.max(1),
                    kv_tokens: args.kv_reserve_tokens,
                    prefill_tile: args.prefill_tile,
                },
                n => SlotBudget::Fixed(n.parse().context("--slots: a count or `auto`")?),
            };
            let stream = StreamOptions {
                slots,
                loader,
                prefetch: args.prefetch,
                io_hints: args.io_hints,
            };
            let mut model: Qwen4Exp<Streamed> =
                Qwen4Exp::from_gguf_streamed(&shards, &device, &opts, &stream)?;
            model.apply_axes(&args.axes)?;
            model.set_prefill_mode(prefill_mode);
            model.set_memory_fraction(args.memory_fraction);
            model.set_mem_check(mem_check);
            model.set_host_fill(args.host_fill);
            if let Some(n) = args.spec_ahead {
                model.set_spec_ahead(n);
            }
            if let Some(path) = &args.hotlist {
                if path.exists() {
                    let tw = std::time::Instant::now();
                    let hot = lloom::models::qwen4exp::hotlist::Hotlist::read(path)?;
                    let loaded = model.warm_from_hotlist(&hot)?;
                    let s = model.stream_stats();
                    println!(
                        "WARM hotlist={} layers={} loaded={loaded} bytes={} ms={}",
                        path.display(),
                        hot.layers.len(),
                        s.bytes_loaded,
                        tw.elapsed().as_millis()
                    );
                    if args.pin > 0 {
                        let pinned = model.pin_from_hotlist(&hot, args.pin)?;
                        println!("PIN per_layer={} pinned={pinned}", args.pin);
                    }
                } else {
                    println!("WARM hotlist={} absent: a cold pool", path.display());
                }
            }
            let out = if args.rows > 1 || args.batched_driver {
                rows_gate(&model, &args, &device, &prompt_ids, &eos_ids)
            } else if !args.legacy_loop {
                driver_gate(&model, &args, &shards, &device, &prompt_ids, &eos_ids, t0)
            } else {
                gate(
                    &model,
                    &args,
                    &shards,
                    &device,
                    &tokenizer,
                    &prompt_ids,
                    &eos_ids,
                    t0,
                )
            };
            if let Some(path) = &args.hotlist {
                use lloom::models::qwen4exp::hotlist::Hotlist;
                let mut hot = if path.exists() {
                    Hotlist::read(path)?
                } else {
                    Hotlist::default()
                };
                hot.merge(&Hotlist::from_usage(&model.stream_usage()));
                hot.write(path)?;
                println!(
                    "HOTLIST written={} layers={} entries={}",
                    path.display(),
                    hot.layers.len(),
                    hot.layers.iter().map(|l| l.len()).sum::<usize>()
                );
            }
            out
        }
        other => anyhow::bail!("--experts {other}: resident | stream"),
    }
}

#[allow(clippy::too_many_arguments)]
/// `--rows N`: the prompt decoded `N` times at once through the batched
/// driver; `ROWS` lines with the per-run aggregate rate.
#[allow(clippy::too_many_arguments)]
fn rows_gate<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    device: &Device,
    prompt_ids: &[u32],
    eos_ids: &[u32],
) -> Result<()> {
    use lloom::generation::{LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, Session};
    let cfg = model.config();
    for run in 0..args.runs {
        let pools = args.rows_pool.then(|| model.state_pools(args.rows, device));
        let mut sessions: Vec<Session<'_, Qwen4Exp<R>>> = (0..args.rows)
            .map(|_| {
                let cache = match &pools {
                    Some(p) => lloom::models::qwen4exp::Cache::in_pools(cfg, p)?,
                    None => model.new_cache(cfg, device)?,
                };
                Ok(Session::new(model, cache))
            })
            .collect::<Result<_>>()?;
        let prompts: Vec<&[u32]> = (0..args.rows).map(|_| prompt_ids).collect();
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions {
            prefill_tile: args.prefill_tile,
            ..DecodeOptions::new(args.max_tokens)
        };
        #[cfg(feature = "metal")]
        candle::kernels::metal::profile::reset();
        let gauge = MemGauge::start(device, args.mem_sample);
        let t = std::time::Instant::now();
        let outs = Session::decode_batched(
            &mut sessions,
            &mut sampler,
            &prompts,
            &opts,
            eos_ids,
            device,
            &mut |_row, _toks| Ok(()),
        )?;
        let wall = t.elapsed();
        let mem = gauge.finish(device);
        let generated: usize = outs.iter().map(|o| o.ids.len()).sum();
        for (i, o) in outs.iter().enumerate() {
            println!(
                "ROWTOKENS run={run} row={i} {}",
                o.ids
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        // The whole run: the rows' prefills and the batched steps (per
        // token = per generated token over all rows).
        profile_line(run, "rows", generated, wall.as_secs_f64() * 1e3);
        mem_line(run, "rows", device, Some(mem));
        let digests: Vec<String> = outs
            .iter()
            .map(|o| {
                let mut h = sha256::Sha256::new();
                for id in &o.ids {
                    h.update(&id.to_le_bytes());
                }
                sha256::hex(&h.finalize())[..8].to_string()
            })
            .collect();
        println!(
            "ROWS run={run} rows={} pool={} prompt_tokens={} generated={generated} wall_ms={} tok_s={:.2} per_row_tok_s={:.2} digests={:?}",
            args.rows,
            args.rows_pool,
            prompt_ids.len(),
            wall.as_millis(),
            generated as f64 / wall.as_secs_f64(),
            generated as f64 / wall.as_secs_f64() / args.rows as f64,
            digests
        );
        let s = model.stream_stats();
        let pf = model.prefetch_stats();
        if s.plans + s.device_plans > 0 {
            println!(
                "STREAM run={run} slots={} plans={} hits={} misses={} bytes_loaded={} hit_rate={:.4}",
                model.slots_per_layer(),
                s.plans,
                s.hits,
                s.misses,
                s.bytes_loaded,
                s.hits as f64 / (s.hits + s.misses).max(1) as f64,
            );
            println!(
                "PREFETCH run={run} on={} spec_ahead={} issued={} staged={} used={} wasted={} stale={} bytes={} parked={} stage_read_gb_s={:.2} direct_read_gb_s={:.2}",
                args.prefetch, model.spec_ahead(), pf.issued, pf.staged, pf.used, pf.wasted, pf.stale, pf.bytes, pf.parked,
                pf.stage_read_bytes as f64 / pf.stage_read_ns.max(1) as f64,
                pf.direct_read_bytes as f64 / pf.direct_read_ns.max(1) as f64
            );
        }
        // `--mtp` with `--rows N`: the same prompts through the speculative
        // batched loop (`Session::decode_spec_batched`: the MTP drafter per
        // row, the rows' verifies `Probes::verify_rows_enqueue`). Every row's
        // stream must be its greedy row's; the rate beside the greedy one.
        if let Some(head) = model.mtp_head() {
            use lloom::models::qwen4exp::MtpDrafter;
            // The greedy rows' caches and pools go first: both sets at once
            // would be two sessions' state the budget did not plan for.
            drop(sessions);
            drop(pools);
            let pools = model.spec_pools(args.rows, args.mtp_k, device);
            let mut sessions: Vec<Session<'_, Qwen4Exp<R>>> = (0..args.rows)
                .map(|_| {
                    let mut cache = lloom::models::qwen4exp::Cache::in_pools(cfg, &pools)?;
                    cache.want_nextn = true;
                    Ok(Session::new(model, cache))
                })
                .collect::<Result<_>>()?;
            let mut drafter = MtpDrafter::with_seqs(
                head,
                model.embed_tokens(),
                model.lm_head(),
                cfg,
                device,
                args.rows,
            )
            .with_k(args.mtp_k);
            let t = std::time::Instant::now();
            let spec = Session::decode_spec_batched(
                &mut sessions,
                &mut drafter,
                &prompts,
                &opts,
                eos_ids,
                device,
                &mut |_row, _toks| Ok(()),
            )?;
            let spec_wall = t.elapsed();
            let spec_generated: usize = spec.iter().map(|o| o.ids.len()).sum();
            let same: Vec<bool> = spec
                .iter()
                .zip(&outs)
                .map(|(a, b)| a.ids == b.ids)
                .collect();
            let rounds: usize = spec.iter().map(|o| o.stats.rounds.len()).sum();
            println!(
                "MTPROWS run={run} rows={} k={} identical_to_greedy={:?} generated={spec_generated} rounds={rounds} tokens_per_round={:.3} wall_ms={} tok_s={:.2} greedy_tok_s={:.2}",
                args.rows,
                drafter.k(),
                same,
                spec_generated as f64 / rounds.max(1) as f64,
                spec_wall.as_millis(),
                spec_generated as f64 / spec_wall.as_secs_f64(),
                generated as f64 / wall.as_secs_f64(),
            );
        }
    }
    Ok(())
}

/// The `LOADED` line and the post-load `MEM` line, shared by the gates.
fn loaded_line<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    shards: &[PathBuf],
    device: &Device,
    t0: std::time::Instant,
) {
    let cfg = model.config();
    let load_ms = t0.elapsed().as_millis();
    println!(
        "LOADED gguf={} shards={} layers={} hidden={} vocab={} experts={}x{} device={:?} ple={} ple_cache_rows={} prefill={:?} moe_act={} experts_mode={} slots={} loader={} config=[{}] load_ms={load_ms}",
        shards[0].display(),
        shards.len(),
        cfg.n_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.n_experts,
        cfg.n_experts_per_tok,
        device,
        if model.has_ple() { "on" } else { "off" },
        args.ple_cache_rows,
        model.prefill_mode(),
        format!("{:?}", model.routed_act()).to_lowercase(),
        args.experts,
        args.slots,
        args.loader,
        cfg.axes_line(),
    );
    mem_line(0, "loaded", device, None);
}

/// The prefetcher's counters for the run so far (cumulative over runs).
fn prefetch_line<R: ExpertResidency + HasStreamStats>(
    run: usize,
    model: &Qwen4Exp<R>,
    args: &Args,
) {
    let pf = model.prefetch_stats();
    println!(
        "PREFETCH run={run} on={} spec_ahead={} issued={} staged={} used={} wasted={} stale={} bytes={} parked={} stage_read_gb_s={:.2} direct_read_gb_s={:.2}",
        args.prefetch, model.spec_ahead(), pf.issued, pf.staged, pf.used, pf.wasted, pf.stale, pf.bytes, pf.parked,
        pf.stage_read_bytes as f64 / pf.stage_read_ns.max(1) as f64,
        pf.direct_read_bytes as f64 / pf.direct_read_ns.max(1) as f64
    );
}

/// The streamed tier's counters (cumulative over the process's runs) and
/// the cache's (this run's): where a miss's host time went, by path.
fn stream_line<R: ExpertResidency + HasStreamStats>(
    run: usize,
    model: &Qwen4Exp<R>,
    args: &Args,
    cache: &Cache,
) {
    let s = model.stream_stats();
    println!(
        "STREAM run={run} loader={} slots={} plans={} groups={} hits={} misses={} evictions={} bytes_loaded={} hit_rate={:.4} device_plans={} settled_misses={} prefetch_now={} redos={} host_plan_passes={} readback_ms={:.0} load_now_ms={:.0} land_ms={:.0} sync_load_ms={:.0} io_hints={} io_inflight_hits={} io_inflight_wasted={} io_inflight_landed={} direct_loads={} direct_wait_ms={:.0} settle_loads={} settle_ms={:.0} settle_clock_lag={} settle_after_first={} replay_ms={:.0} host_pass_ms={:.0} free_takes={} inplace_takes={} quarantined={} recycled={}",
        args.loader,
        model.slots_per_layer(),
        s.plans,
        s.groups,
        s.hits,
        s.misses,
        s.evictions,
        s.bytes_loaded,
        s.hits as f64 / (s.hits + s.misses).max(1) as f64,
        s.device_plans,
        s.settled_misses,
        s.prefetch_now,
        cache.redos,
        cache.host_plan_passes,
        s.readback_ns as f64 / 1e6,
        s.load_now_ns as f64 / 1e6,
        s.land_ns as f64 / 1e6,
        s.sync_load_ns as f64 / 1e6,
        s.io_hints,
        s.io_inflight_hits,
        s.io_inflight_wasted,
        s.io_inflight_landed,
        s.direct_loads,
        s.direct_wait_ns as f64 / 1e6,
        s.settle_loads,
        s.settle_ns as f64 / 1e6,
        s.settle_clock_lag,
        cache.settle_after_first,
        cache.replay_ns as f64 / 1e6,
        cache.host_pass_ns as f64 / 1e6,
        s.free_takes,
        s.inplace_takes,
        s.quarantined,
        s.recycled,
    );
}

/// `--driver`: the prompt through `Session::decode` on the blocking arm —
/// the loop `lloom serve` runs (`decode-driver-design.md` Stage 5/6) —
/// with the same `RESULT` / `PROFILE` / `MEM` / `STREAM` lines as `gate`,
/// minus the logits digest (`logits_sha256=-`: the driver never reads the
/// logits back; the tokens digest is the gate).
#[allow(clippy::too_many_arguments)]
fn driver_gate<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    shards: &[PathBuf],
    device: &Device,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    t0: std::time::Instant,
) -> Result<()> {
    use lloom::generation::poll::PollMechanism;
    use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
    use lloom::models::qwen4exp::{prefill_chunk, prefill_pair_on, prefill_tiles};
    use lloom::nn::{DecodeOptions, IdentityProposer, Session};
    let cfg = model.config().clone();
    loaded_line(model, args, shards, device, t0);
    let (lookahead, adaptive) = match args.readback.as_str() {
        "blocking" => (0, false),
        "ring-fixed" => (1, false),
        "ring" => (1, true),
        other => anyhow::bail!("--readback {other}: blocking | ring-fixed | ring"),
    };
    // The tile the harness's own loop feeds (`prefill_tiled`): the paired
    // prefill wants both chunks in one forward.
    let tile =
        if args.prefill_tile != 0 && prefill_pair_on() && args.prefill_tile <= prefill_chunk() {
            prefill_tiles() * prefill_chunk()
        } else {
            args.prefill_tile
        };
    println!(
        "DRIVER arm={} prefill_tile={tile} lookahead={lookahead} adaptive={adaptive} host_fill={}",
        args.readback,
        args.readback != "blocking" && model.host_fill()
    );
    let mut greedy_digest: Option<String> = None;
    for run in 0..args.runs {
        let cache: Cache = model.new_cache(&cfg, device)?;
        let mut session = Session::new(model, cache);
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut spec = IdentityProposer;
        let mut opts = DecodeOptions {
            prefill_tile: tile,
            ..DecodeOptions::new(args.max_tokens)
        };
        if lookahead > 0 {
            opts.lookahead = lookahead;
            opts.budget = CommitBudget::steps(8)?;
            // Under the adaptive commit the ids are taken off the ring at the
            // end (no reader streams here), and a run past an EOS stops at the
            // cap: its ids are cut at the EOS.
            opts.adaptive_commit = adaptive;
            opts.stream_via_reader = adaptive;
        }
        #[cfg(feature = "metal")]
        candle::kernels::metal::profile::reset();
        let gauge = MemGauge::start(device, args.mem_sample);
        // `CANDLE_METAL_TRACE=1`: the dispatch census of three armed decode
        // steps (the 9th to the 11th token) — kernel, grid, threadgroup,
        // barrier, encoder per dispatch, through the same recorder
        // `lfm2-dispatch-trace` uses. The callback runs after a step's
        // readback, so the dispatches between two callbacks are one step.
        let trace_on = trace_requested();
        let mut produced = 0usize;
        let mut traced: Vec<(usize, TraceStep)> = Vec::new();
        // The tier's counters when the first token is out: what the prompt's
        // prefill loaded, so the decode's share is the difference.
        let mut pre: Option<lloom::nn::expert_stream::StreamStats> = None;
        let t1 = std::time::Instant::now();
        let mut on_tokens = |toks: &[u32]| -> candle::Result<()> {
            if pre.is_none() {
                pre = Some(model.stream_stats());
            }
            if trace_on {
                let before = produced;
                produced += toks.len();
                if before >= TRACE_FROM && before < TRACE_FROM + TRACE_STEPS {
                    traced.push((before + 1, trace_take()));
                }
                if produced >= TRACE_FROM && produced < TRACE_FROM + TRACE_STEPS {
                    trace_set(true, Some(format!("decode[{}]", produced + 1)));
                } else {
                    trace_set(false, None);
                    let _ = trace_take();
                }
            }
            Ok(())
        };
        let out = if lookahead == 0 {
            let mut rb = Blocking::new(eos_ids, device)?;
            session.decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                prompt_ids,
                &opts,
                &mut on_tokens,
            )?
        } else {
            let mut rb = Ring::new(eos_ids, device, 0, EosRead::Poll(PollMechanism::default()))?;
            session.decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                prompt_ids,
                &opts,
                &mut on_tokens,
            )?
        };
        let wall_ms = t1.elapsed().as_secs_f64() * 1e3;
        for (step, dispatches) in &traced {
            trace_census(run, *step, dispatches);
        }
        let prefill_ms = out.stats.prefill_ns as f64 / 1e6;
        let decode_ms = (wall_ms - prefill_ms).max(0.0);
        // One profile snapshot for the run: the driver's prefill and decode
        // share the command stream and the snapshot is process-wide.
        profile_line(run, "driver", prompt_ids.len() + out.ids.len(), wall_ms);
        mem_line(run, "driver", device, Some(gauge.finish(device)));
        let mut h = sha256::Sha256::new();
        for t in &out.ids {
            h.update(&t.to_le_bytes());
        }
        let digest = sha256::hex(&h.finalize());
        println!(
            "TOKENS {}",
            out.ids
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "RESULT run={run} tokens_sha256={digest} logits_sha256=- logits_max=- prompt_tokens={} generated={} hit_eos={} prefill_ms={prefill_ms:.0} decode_ms={decode_ms:.0} max_tokens={} driver=session-{} steps={} commits={} host_fill_forwards={} host_fill_fills={} overrun={} config=[{}]",
            prompt_ids.len(),
            out.ids.len(),
            out.eos,
            args.max_tokens,
            args.readback,
            out.stats.steps,
            out.stats.commits,
            out.stats.host_feed.as_ref().map_or(0, |f| f.forwards),
            out.stats.host_feed.as_ref().map_or(0, |f| f.fills),
            out.stats.overrun,
            model.config().axes_line(),
        );
        if let Some(f) = &out.stats.host_feed {
            if !f.fill_ns.is_empty() {
                let mut v = f.fill_ns.clone();
                v.sort_unstable();
                let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64 / 1e3;
                println!(
                    "HOSTFILL run={run} fills={} p50_us={:.1} p90_us={:.1} p99_us={:.1} max_us={:.1}",
                    v.len(),
                    q(0.5),
                    q(0.9),
                    q(0.99),
                    q(1.0)
                );
            }
        }
        let s = model.stream_stats();
        if s.plans + s.device_plans > 0 {
            prefetch_line(run, model, args);
            if let Some(p) = pre {
                println!(
                    "STREAMPRE run={run} misses={} readback_ms={:.0} load_now_ms={:.0} land_ms={:.0} sync_load_ms={:.0} settle_loads={} settle_ms={:.0} direct_loads={} direct_wait_ms={:.0}",
                    p.misses,
                    p.readback_ns as f64 / 1e6,
                    p.load_now_ns as f64 / 1e6,
                    p.land_ns as f64 / 1e6,
                    p.sync_load_ns as f64 / 1e6,
                    p.settle_loads,
                    p.settle_ns as f64 / 1e6,
                    p.direct_loads,
                    p.direct_wait_ns as f64 / 1e6,
                );
            }
            stream_line(run, model, args, session.sequence().cache());
        }
        greedy_digest = Some(digest);
    }
    // `--mtp <sidecar>`: the same prompt once more through the driver's
    // speculative body (`Session::decode_speculative`, the DSpark seams:
    // `MtpDrafter: Speculator`, `Qwen4Exp: Probes<NextnHidden>`), its
    // tokens against the greedy run's (byte-identical is the gate).
    if let Some(head) = model.mtp_head() {
        use lloom::models::qwen4exp::MtpDrafter;
        let pools = model.spec_pools(1, args.mtp_k, device);
        let mut cache: Cache = Cache::in_pools(&cfg, &pools)?;
        cache.want_nextn = true;
        let mut session = Session::new(model, cache);
        let mut drafter =
            MtpDrafter::new(head, model.embed_tokens(), model.lm_head(), &cfg, device)
                .with_k(args.mtp_k);
        let mut rb = Blocking::new(eos_ids, device)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions {
            prefill_tile: tile,
            ..DecodeOptions::new(args.max_tokens)
        };
        #[cfg(feature = "metal")]
        candle::kernels::metal::profile::reset();
        let gauge = MemGauge::start(device, args.mem_sample);
        let t2 = std::time::Instant::now();
        let out = session.decode_speculative(
            &mut drafter,
            &mut rb,
            &mut sampler,
            prompt_ids,
            &opts,
            eos_ids,
            &mut |_| Ok(()),
        )?;
        let mtp_ms = t2.elapsed().as_millis();
        profile_line(0, "mtp", out.ids.len(), mtp_ms as f64);
        mem_line(0, "mtp", device, Some(gauge.finish(device)));
        let mut h = sha256::Sha256::new();
        for t in &out.ids {
            h.update(&t.to_le_bytes());
        }
        let mtp_digest = sha256::hex(&h.finalize());
        println!(
            "MTPTOKENS {}",
            out.ids
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let rounds = out.stats.rounds.len();
        let accepted_drafts = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
        let tokens: usize = out.stats.rounds.iter().map(|r| r.accepted).sum();
        let same = greedy_digest.as_deref() == Some(mtp_digest.as_str());
        let sum = |f: &dyn Fn(&lloom::nn::RoundTiming) -> u64| -> f64 {
            out.stats.rounds.iter().map(|r| f(&r.timing)).sum::<u64>() as f64 / 1e6
        };
        let hist: Vec<usize> = (0..=drafter.k() + 1)
            .map(|a| out.stats.rounds.iter().filter(|r| r.accepted == a).count())
            .collect();
        println!(
            "MTP tokens_sha256={mtp_digest} identical_to_greedy={same} generated={} rounds={rounds} accepted_drafts={accepted_drafts} draft_accept_rate={:.4} tokens_per_round={:.3} drafter_rows={} wall_ms={mtp_ms} k={} draft_ms={:.1} verify_ms={:.1} observe_ms={:.1} accepted_hist={hist:?} driver=session-speculative",
            out.ids.len(),
            accepted_drafts as f64 / rounds.max(1) as f64,
            tokens as f64 / rounds.max(1) as f64,
            drafter.rows_seen(),
            drafter.k(),
            sum(&|t| t.draft_ns),
            sum(&|t| t.verify_ns),
            sum(&|t| t.observe_ns),
        );
        let s = model.stream_stats();
        let c = session.cache_mut();
        let rb = |f: &dyn Fn(&lloom::nn::RoundTiming) -> u64| -> u64 {
            out.stats.rounds.iter().map(|r| f(&r.timing)).sum::<u64>()
        };
        println!(
            "MTPSTREAM plans={} device_plans={} hits={} misses={} evictions={} bytes_loaded={} settled_misses={} settle_loads={} redos={} window_redos={} window_host={} window_device={} truncations={} rejected_flags={} host_plan_passes={} device_plan_armed={} readbacks_draft={} readbacks_verify={} readbacks_observe={} readbacks_per_round={:.2}",
            s.plans,
            s.device_plans,
            s.hits,
            s.misses,
            s.evictions,
            s.bytes_loaded,
            s.settled_misses,
            s.settle_loads,
            c.redos,
            c.window_redos,
            c.window_host,
            c.window_device,
            c.window_truncations,
            c.window_rejected_flags,
            c.host_plan_passes,
            c.device_plan_armed,
            rb(&|t| t.draft_readbacks),
            rb(&|t| t.verify_readbacks),
            rb(&|t| t.observe_readbacks),
            (rb(&|t| t.draft_readbacks + t.verify_readbacks + t.observe_readbacks)) as f64
                / rounds.max(1) as f64,
        );
        let (od, oh, orr) = drafter.observe_plans();
        println!("MTPOBSERVE device={od} host={oh} redos={orr}");
        // Under the model's per-round trace (`QWEN4EXP_MTP_TRACE`), each
        // round's phase brackets in round order, to join with its
        // `MTPTRACE` line (the same order).
        if std::env::var_os("QWEN4EXP_MTP_TRACE").is_some_and(|v| v == "1") {
            for (i, r) in out.stats.rounds.iter().enumerate() {
                println!(
                    "MTPROUND i={i} accepted={} draft_ms={:.2} verify_ms={:.2} observe_ms={:.2} rb_draft={} rb_verify={} rb_observe={}",
                    r.accepted,
                    r.timing.draft_ns as f64 / 1e6,
                    r.timing.verify_ns as f64 / 1e6,
                    r.timing.observe_ns as f64 / 1e6,
                    r.timing.draft_readbacks,
                    r.timing.verify_readbacks,
                    r.timing.observe_readbacks,
                );
            }
        }
        if !same {
            anyhow::bail!("the MTP run's tokens differ from the greedy run's");
        }
    }
    Ok(())
}

fn gate<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    shards: &[PathBuf],
    device: &Device,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    t0: std::time::Instant,
) -> Result<()> {
    let cfg = model.config().clone();
    loaded_line(model, args, shards, device, t0);

    if args.rollback_probe > 0 {
        rollback_probe(
            model,
            &cfg,
            device,
            prompt_ids,
            eos_ids,
            args.rollback_probe,
        )?;
    }
    let mut digests = Vec::with_capacity(args.runs);
    let mut last_text = String::new();
    let forced: Option<Vec<u32>> = args
        .force_tokens
        .as_deref()
        .map(read_forced_tokens)
        .transpose()?;
    for run in 0..args.runs {
        // The dump is the first run's (a determinism repeat writes nothing).
        let mut sink = match (&args.dump_logits, run) {
            (Some(p), 0) => Some(std::io::BufWriter::new(
                std::fs::File::create(p).with_context(|| format!("creating {}", p.display()))?,
            )),
            _ => None,
        };
        let (mut forced_steps, mut overrides) = (0usize, 0usize);
        let mut cache: Cache = model.new_cache(&cfg, device)?;
        if args.probe && run == 0 {
            let first = Tensor::from_vec(vec![prompt_ids[0]], (1, 1), device)?;
            let h = model.embed(&first)?;
            let (n, nonfinite, maxabs, meanabs) = stats(&h)?;
            println!("PROBE embed token={} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}", prompt_ids[0]);
            // Every prompt position, one at a time, with a per-step summary
            // (the first step verbose): where does the state-threaded forward
            // first go non-finite, and how does the carry's magnitude move.
            let mut probe_cache: Cache = model.new_cache(&cfg, device)?;
            let mut probe = Probe::new(true);
            for (t, &tok) in prompt_ids.iter().enumerate() {
                probe.reset();
                probe.verbose = t == 0;
                let input = Tensor::from_vec(vec![tok], (1, 1), device)?;
                let h = model.embed(&input)?;
                let (_, e_nonfinite, e_maxabs, _) = stats(&h)?;
                probe_cache.set_pending_tokens(&[tok]);
                let logits = model.decode_embed(&h, &mut probe_cache, &mut probe)?;
                let (n, nonfinite, maxabs, meanabs) = stats(&logits)?;
                println!(
                    "PROBE step={t} token={tok} embed_nonfinite={e_nonfinite} embed_maxabs={e_maxabs:.3e} first_nonfinite={:?} carry_maxabs={:.3e} logits: n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e} argmax={}",
                    probe.first_nonfinite,
                    probe.max_abs,
                    argmax(&logits)?
                );
            }
        }
        let t1 = std::time::Instant::now();
        let mut logits = None;
        let mut fed = 0usize;
        let mut splits: Vec<usize> = args
            .prompt_split
            .iter()
            .copied()
            .filter(|&n| n > 0 && n < prompt_ids.len())
            .collect();
        splits.sort_unstable();
        splits.dedup();
        splits.push(prompt_ids.len());
        let gauge = MemGauge::start(device, args.mem_sample);
        for end in splits {
            let seg = &prompt_ids[fed..end];
            logits = Some(prefill_tiled(
                model,
                &mut cache,
                seg,
                args.prefill_tile,
                device,
            )?); // [1, vocab]
            fed = end;
            if args.state_digest && end < prompt_ids.len() {
                for (name, sum, nonfinite) in cache.checksums()? {
                    println!(
                        "STATE run={run} phase=at{end} {name}={sum:016x} nonfinite={nonfinite}"
                    );
                }
            }
        }
        let mut logits = logits.expect("a non-empty prompt");
        let prefill_ms = t1.elapsed().as_millis();
        profile_line(run, "prefill", prompt_ids.len(), prefill_ms as f64);
        mem_line(run, "prefill", device, Some(gauge.finish(device)));
        let gauge = MemGauge::start(device, args.mem_sample);
        if args.state_digest {
            for (name, sum, nonfinite) in cache.checksums()? {
                println!("STATE run={run} phase=prompt {name}={sum:016x} nonfinite={nonfinite}");
            }
        }
        let mut step_no = 0usize;
        let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
        let mut hit_eos = false;
        // The logits digest covers every position's full logit row (prefill's
        // last position + each generated step), so a computation that moves
        // by rounding while the argmax holds is still visible.
        let mut lh = sha256::Sha256::new();
        let mut logits_max = f32::MIN;
        let mut digest_logits = |t: &Tensor, lh: &mut sha256::Sha256| -> Result<()> {
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            for x in &v {
                lh.update(&x.to_le_bytes());
                logits_max = logits_max.max(*x);
            }
            Ok(())
        };
        digest_logits(&logits, &mut lh)?;
        dump_row(&mut sink, step_no, &logits)?;
        if args.state_digest {
            println!(
                "STEP run={run} step={step_no} logits={:016x}",
                fnv(&logits)?
            );
        }
        for _ in 0..args.max_tokens {
            let sampled = argmax(&logits)?;
            let next = match &forced {
                None => sampled,
                Some(ids) => match ids.get(step_no) {
                    Some(&id) => {
                        forced_steps += 1;
                        overrides += usize::from(id != sampled);
                        id
                    }
                    None => break,
                },
            };
            if args.margins {
                let v = logits.flatten_all()?.to_vec1::<f32>()?;
                let (mut b1, mut i1, mut b2, mut i2) = (f32::MIN, 0usize, f32::MIN, 0usize);
                for (i, &x) in v.iter().enumerate() {
                    if x > b1 {
                        b2 = b1;
                        i2 = i1;
                        b1 = x;
                        i1 = i;
                    } else if x > b2 {
                        b2 = x;
                        i2 = i;
                    }
                }
                println!(
                    "MARGIN run={run} step={step_no} argmax={i1} second={i2} margin={:.4} top={b1:.4}",
                    b1 - b2
                );
            }
            if forced.is_none() && eos_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            let input = Tensor::from_vec(vec![next], (1, 1), device)?;
            logits = model.forward(&input, cache.pos, &mut cache)?;
            digest_logits(&logits, &mut lh)?;
            step_no += 1;
            dump_row(&mut sink, step_no, &logits)?;
            if args.state_digest {
                println!(
                    "STEP run={run} step={step_no} logits={:016x}",
                    fnv(&logits)?
                );
            }
        }
        if args.state_digest {
            for (name, sum, nonfinite) in cache.checksums()? {
                println!("STATE run={run} phase=end {name}={sum:016x} nonfinite={nonfinite}");
            }
        }
        let decode_ms = t1.elapsed().as_millis() - prefill_ms;
        profile_line(run, "decode", generated.len(), decode_ms as f64);
        mem_line(run, "decode", device, Some(gauge.finish(device)));
        let logits_digest = sha256::hex(&lh.finalize());
        let mut h = sha256::Sha256::new();
        for t in &generated {
            h.update(&t.to_le_bytes());
        }
        let digest = sha256::hex(&h.finalize());
        last_text = tokenizer
            .decode(&generated, true)
            .map_err(anyhow::Error::msg)?;
        println!("--- run {run} ---");
        println!("prompt: {}", args.prompt);
        println!("output: {last_text}");
        println!(
            "TOKENS {}",
            generated
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        if let Some(ids) = &forced {
            anyhow::ensure!(forced_steps > 0, "--force-tokens: no step took a forced id");
            println!(
                "FORCED run={run} requested={} forced={forced_steps} overrides={overrides}",
                ids.len()
            );
        }
        println!(
            "RESULT run={run} tokens_sha256={digest} logits_sha256={logits_digest} logits_max={logits_max:.4} prompt_tokens={} generated={} hit_eos={hit_eos} prefill_ms={prefill_ms} decode_ms={decode_ms} max_tokens={} forced={forced_steps} overrides={overrides} config=[{}]",
            prompt_ids.len(),
            generated.len(),
            args.max_tokens,
            cfg.axes_line()
        );
        if args.experts == "stream" {
            let st = model.staging_stats();
            println!(
                "STAGING run={run} buffers={} bytes={} allocations={} reuses={} overflows={}",
                st.buffers, st.bytes, st.allocations, st.reuses, st.overflows
            );
            prefetch_line(run, model, args);
            report_pass_plans(device);
            let dh = lloom::models::qwen4exp::Qwen4Exp::<lloom::models::qwen4exp::Streamed>::decode_hints();
            if dh > 0 {
                println!("DECODEHINT run={run} hints={dh}");
            }
            let hr = model.hint_recall();
            if hr.iter().any(|&(_, n)| n > 0) {
                let r = |k: usize| hr[k].0 as f64 / hr[k].1.max(1) as f64;
                println!(
                    "HINTRECALL run={run} k0={:.3} k1={:.3} k2={:.3} k3={:.3} n={}",
                    r(0),
                    r(1),
                    r(2),
                    r(3),
                    hr[0].1
                );
            }
            stream_line(run, model, args, &cache);
        }
        if let Some((hits, misses)) = model.ple_cache_stats() {
            println!(
                "PLECACHE run={run} rows={} hits={hits} misses={misses} hit_rate={:.4}",
                args.ple_cache_rows,
                hits as f64 / (hits + misses).max(1) as f64
            );
        }
        digests.push(digest);
    }
    let identical = digests.windows(2).all(|w| w[0] == w[1]);
    println!(
        "DETERMINISM runs={} identical={identical} digest={}",
        args.runs,
        digests.first().cloned().unwrap_or_default()
    );
    let mut ok = identical;
    if let Some(head) = model.mtp_head() {
        // One speculative decode of the same prompt: the MTP drafter on the
        // seam, verified by the target; must equal the greedy stream.
        // The verify window's states in phase rings: an accept by phase.
        let pools = model.spec_pools(1, args.mtp_k, device);
        let mut cache: Cache = Cache::in_pools(&cfg, &pools)?;
        cache.want_nextn = true;
        let mut session = Session::new(model, cache);
        let mut drafter =
            MtpDrafter::new(head, model.embed_tokens(), model.lm_head(), &cfg, device)
                .with_k(args.mtp_k);
        let mut rb = Blocking::new(eos_ids, device)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions {
            prefill_tile: args.prefill_tile,
            ..DecodeOptions::new(args.max_tokens)
        };
        #[cfg(feature = "metal")]
        candle::kernels::metal::profile::reset();
        let gauge = MemGauge::start(device, args.mem_sample);
        let t2 = std::time::Instant::now();
        let out = session.decode_speculative(
            &mut drafter,
            &mut rb,
            &mut sampler,
            prompt_ids,
            &opts,
            eos_ids,
            &mut |_| Ok(()),
        )?;
        let mtp_ms = t2.elapsed().as_millis();
        profile_line(0, "mtp", out.ids.len(), mtp_ms as f64);
        mem_line(0, "mtp", device, Some(gauge.finish(device)));
        let mut h = sha256::Sha256::new();
        for t in &out.ids {
            h.update(&t.to_le_bytes());
        }
        let mtp_digest = sha256::hex(&h.finalize());
        let rounds = out.stats.rounds.len();
        let accepted_drafts = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
        let tokens: usize = out.stats.rounds.iter().map(|r| r.accepted).sum();
        let same = digests.first().map(|d| d == &mtp_digest).unwrap_or(false);
        println!(
            "output(mtp): {}",
            tokenizer
                .decode(&out.ids, true)
                .map_err(anyhow::Error::msg)?
        );
        let sum = |f: &dyn Fn(&lloom::nn::RoundTiming) -> u64| -> f64 {
            out.stats.rounds.iter().map(|r| f(&r.timing)).sum::<u64>() as f64 / 1e6
        };
        let hist: Vec<usize> = (0..=drafter.k() + 1)
            .map(|a| out.stats.rounds.iter().filter(|r| r.accepted == a).count())
            .collect();
        println!(
            "MTP tokens_sha256={mtp_digest} identical_to_greedy={same} generated={} rounds={rounds} accepted_drafts={accepted_drafts} draft_accept_rate={:.4} tokens_per_round={:.3} drafter_rows={} wall_ms={mtp_ms} k={} draft_ms={:.1} verify_ms={:.1} observe_ms={:.1} accepted_hist={hist:?}",
            out.ids.len(),
            accepted_drafts as f64 / rounds.max(1) as f64,
            tokens as f64 / rounds.max(1) as f64,
            drafter.rows_seen(),
            drafter.k(),
            sum(&|t| t.draft_ns),
            sum(&|t| t.verify_ns),
            sum(&|t| t.observe_ns),
        );
        ok &= same;
    }
    if let Some(expect) = &args.expect {
        let found = last_text.to_lowercase().contains(&expect.to_lowercase());
        println!("TEXT expect={expect:?} found={found}");
        ok &= found;
    }
    if !ok {
        anyhow::bail!("gate FAILED (identical={identical})");
    }
    Ok(())
}

/// The rollback probe (see `--rollback-probe`).
fn rollback_probe<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &lloom::models::qwen4exp::Config,
    device: &Device,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    rounds: usize,
) -> Result<()> {
    let wrong = cfg.vocab_size as u32 - 2;
    let mode = std::env::var("QWEN4EXP_ROLLBACK_MODE").unwrap_or_else(|_| "rounds".into());
    // subject: verify rounds with a wrong draft, each a partial accept
    let mut sub: Cache = model.new_cache(cfg, device)?;
    sub.want_nextn = true;
    let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;
    let logits = model.forward(&input, 0, &mut sub)?;
    let _ = model.take_probe(&mut sub)?;
    let mut anchor = argmax(&logits)?;
    let mut consumed: Vec<u32> = Vec::new();
    if mode == "roundtrip" {
        // open + rewind with no forward in between: the snapshot/restore alone
        let before = sub.checksums()?;
        for _ in 0..rounds {
            sub.open_window()?;
            sub.rewind_window()?;
        }
        let after = sub.checksums()?;
        let diff = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| a.1 != b.1)
            .count();
        println!(
            "ROLLBACK roundtrip x{rounds}: channels={} differing={diff}",
            before.len()
        );
        // and open + one forward + rewind: the restore after a write
        let before = sub.checksums()?;
        sub.open_window()?;
        let inp = Tensor::from_vec(vec![anchor], (1, 1), device)?;
        model.forward(&inp, sub.pos, &mut sub)?;
        sub.rewind_window()?;
        let after = sub.checksums()?;
        let diff = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| a.1 != b.1)
            .count();
        println!(
            "ROLLBACK open+forward+rewind: channels={} differing={diff} pos={}",
            before.len(),
            sub.pos
        );
        return Ok(());
    }
    for r in 0..rounds {
        let (acc, _probe) = model.verify_and_probe(
            &[Draft::host(vec![wrong])],
            &[Anchor {
                last: anchor,
                pos: sub.pos,
            }],
            &mut sub,
            eos_ids,
        )?;
        let acc = &acc[0];
        println!(
            "ROLLBACK round={r} anchor={anchor} accepted={acc:?} pos_after={}",
            sub.pos
        );
        consumed.push(anchor);
        anchor = acc[0];
        if acc.len() != 1 {
            anyhow::bail!("the wrong draft was accepted");
        }
    }
    // reference: the same tokens decoded greedily, one per forward
    let mut refc: Cache = model.new_cache(cfg, device)?;
    model.forward(&input, 0, &mut refc)?;
    let mut ref_next = Vec::new();
    for &t in &consumed {
        let inp = Tensor::from_vec(vec![t], (1, 1), device)?;
        let lg = model.forward(&inp, refc.pos, &mut refc)?;
        ref_next.push(argmax(&lg)?);
    }
    println!("ROLLBACK consumed={consumed:?} reference_next={ref_next:?} subject_next={anchor} pos sub={} ref={}", sub.pos, refc.pos);
    let a = sub.checksums()?;
    let b = refc.checksums()?;
    let mut diff = 0;
    for ((n1, h1, _), (n2, h2, _)) in a.iter().zip(&b) {
        if n1 != n2 || h1 != h2 {
            diff += 1;
            if diff <= 12 {
                println!("ROLLBACK DIFF {n1}: subject={h1:016x} reference={h2:016x}");
            }
        }
    }
    println!("ROLLBACK channels={} differing={diff}", a.len());
    // magnitudes on a few channels
    for l in [0usize, 1, 2] {
        if let (Some(x), Some(y)) = (sub.s_state.get(l), refc.s_state.get(l)) {
            let (m, n) = maxdiff(x, y)?;
            println!("ROLLBACK S[{l}] max|Δ|={m:.3e} max|ref|={n:.3e}");
        }
        if let (Some(x), Some(y)) = (sub.conv_state.get(l), refc.conv_state.get(l)) {
            let (m, n) = maxdiff(x, y)?;
            println!("ROLLBACK conv[{l}] max|Δ|={m:.3e} max|ref|={n:.3e}");
        }
    }
    Ok(())
}

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let b = b
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let mut m = 0f32;
    let mut n = 0f32;
    for (x, y) in a.iter().zip(&b) {
        m = m.max((x - y).abs());
        n = n.max(y.abs());
    }
    Ok((m, n))
}

/// FNV over the f32 bit patterns of a logits row (the `STEP` line).
fn fnv(logits: &Tensor) -> Result<u64> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut h = 0xcbf29ce484222325u64;
    for x in &v {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    Ok(h)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

/// The first token index the driver's dispatch trace records at, and how
/// many steps it records (`CANDLE_METAL_TRACE=1`).
const TRACE_FROM: usize = 8;
const TRACE_STEPS: usize = 3;

#[cfg(feature = "metal")]
type TraceStep = Vec<candle::kernels::metal::trace::Dispatch>;
#[cfg(not(feature = "metal"))]
type TraceStep = Vec<()>;

fn trace_requested() -> bool {
    #[cfg(feature = "metal")]
    {
        candle::kernels::metal::trace::trace_requested()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

fn trace_set(on: bool, region: Option<String>) {
    #[cfg(feature = "metal")]
    {
        candle::kernels::metal::trace::set_region(region);
        candle::kernels::metal::trace::set_recording(on);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = (on, region);
    }
}

fn trace_take() -> TraceStep {
    #[cfg(feature = "metal")]
    {
        candle::kernels::metal::trace::take_dispatches()
    }
    #[cfg(not(feature = "metal"))]
    {
        Vec::new()
    }
}

/// One decode step's census: the totals, then one line per kernel with its
/// launches, how many of them waited at a barrier, and the shapes it was
/// launched at (threads a dispatch = grid × threadgroup when dispatched by
/// threadgroups).
#[cfg(feature = "metal")]
fn trace_census(run: usize, step: usize, d: &TraceStep) {
    use std::collections::{BTreeMap, BTreeSet};
    let barriers = d.iter().filter(|x| x.barrier).count();
    let encoders: BTreeSet<u64> = d.iter().map(|x| x.encoder).collect();
    println!(
        "TRACE run={run} step={step} dispatches={} barriers={} encoders={}",
        d.len(),
        barriers,
        encoders.len()
    );
    // Where the encoder sessions break: the kernel before and after each
    // boundary (a blit, a cut or a readback ends the compute encoder; the
    // trace sees only the compute dispatches either side of it).
    let mut breaks: BTreeMap<String, usize> = BTreeMap::new();
    for w in d.windows(2) {
        if w[0].encoder != w[1].encoder {
            *breaks
                .entry(format!("after={} before={}", w[0].pipeline, w[1].pipeline))
                .or_default() += 1;
        }
    }
    let mut br: Vec<_> = breaks.into_iter().collect();
    br.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in br {
        println!("TRACE run={run} step={step} break n={n} {k}");
    }
    // kernel -> (count, barriered, shape -> count)
    let mut by: BTreeMap<&str, (usize, usize, BTreeMap<String, usize>)> = BTreeMap::new();
    for x in d {
        let (gw, gh, gd) = x.grid;
        let (tw, th, td) = x.threadgroup;
        let tg = tw * th * td;
        let threads = if x.by_threadgroups {
            gw * gh * gd * tg
        } else {
            gw * gh * gd
        };
        let shape = format!(
            "grid={}x{}x{}{} tg={} threads={}",
            gw,
            gh,
            gd,
            if x.by_threadgroups { "tg" } else { "" },
            tg,
            threads
        );
        let e = by.entry(x.pipeline.as_str()).or_default();
        e.0 += 1;
        e.1 += x.barrier as usize;
        *e.2.entry(shape).or_default() += 1;
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(a.0.cmp(b.0)));
    for (name, (n, barriered, shapes)) in rows {
        let mut sh: Vec<_> = shapes.into_iter().collect();
        sh.sort_by(|a, b| b.1.cmp(&a.1));
        let shapes = sh
            .iter()
            .map(|(s, c)| format!("{c}×[{s}]"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("TRACE run={run} step={step} kernel={name} n={n} barrier={barriered} {shapes}");
    }
}

#[cfg(not(feature = "metal"))]
fn trace_census(_run: usize, _step: usize, _d: &TraceStep) {}
