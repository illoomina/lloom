//! The overlap window of Qwen4-exp's PLE: the GPU time of a decode step's
//! work BEFORE the PLE layer (the token embedding and the layers under it),
//! which is how long a host fill of the PLE rows can take without the GPU
//! waiting for it (`design/docs/expert-streaming.md` §20.13 probe 2).
//!
//! The model's PLE-window instrument (`Qwen4Exp::arm_ple_window`) stamps each
//! decode step's entry and cuts the command buffer at the PLE layer; with
//! `CANDLE_METAL_PROFILE=1` every command buffer's GPU interval and commit
//! stamp is recorded, so the buffers committed between a step's entry and its
//! cut are exactly the pre-PLE work. The decode is the driver's
//! (`Session::decode`, the blocking arm — the loop `lloom serve` runs on this
//! model), over the streamed expert tier.
//!
//! ```bash
//! CANDLE_METAL_PROFILE=1 lloom-arb measure -- ./target/release/examples/ple_window_probe \
//!     --max-tokens 160 --prompt "…"
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, CommitBudget, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen4exp::loader::{LoadOptions, SlotBudget, StreamLoader, StreamOptions};
use lloom::models::qwen4exp::NgramWindow;
use lloom::models::qwen4exp::{prefill_chunk, prefill_pair_on, prefill_tiles, Qwen4Exp, Streamed};
use lloom::nn::model::CausalLM as _;
use lloom::nn::ngram_memory::ReadPolicy;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::Device;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "../../candle-examples/examples/lfm2-determinism/sha256.rs"]
mod sha256;

#[derive(Parser, Debug)]
#[command(about = "GPU time before the PLE layer, per decode step (§20.13 probe 2)")]
struct Args {
    /// The first GGUF shard; its `-0000k-of-0000N` siblings are loaded too.
    /// Default: the HF cache's ISTA-DASLab IQ3_XXS snapshot.
    #[arg(long)]
    gguf: Option<PathBuf>,
    /// tokenizer.json. Default: the HF cache's Qwen/Qwen3.8-Flash-Next snapshot.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(
        long,
        default_value = "Write a short story about a lighthouse keeper who finds a message in a bottle. Use plain prose, at least three paragraphs."
    )]
    prompt: String,
    #[arg(long, default_value_t = 160)]
    max_tokens: usize,
    /// Streamed experts per layer per projection, or `auto`.
    #[arg(long, default_value = "auto")]
    slots: String,
    #[arg(long, default_value_t = 0.65)]
    memory_fraction: f64,
    /// Decode steps at the start excluded from the summary (the pool warms
    /// and the device plan arms after the first token).
    #[arg(long, default_value_t = 4)]
    skip: usize,
    /// Leave the instrument unarmed (the identity control: same tokens).
    #[arg(long)]
    unarmed: bool,
    /// Fresh-cache decodes of the prompt in one process; each is reported
    /// (the first on a cold expert pool, the later ones warm, as the P33
    /// gates ran them).
    #[arg(long, default_value_t = 2)]
    runs: usize,
    /// How the PLE table reads a token's rows: `serial`, `adaptive` (the
    /// default) or `concurrent` (`nn::ngram_memory::ReadPolicy`).
    #[arg(long, default_value = "adaptive")]
    ple_read: String,
    /// Before every run after the first, drop the page-cache pages of every
    /// PLE row the previous run's stream read (`msync(MS_INVALIDATE)` on a
    /// map of the table's shard): the runs then decode the same tokens with
    /// a warm expert pool and a cold PLE.
    #[arg(long)]
    evict_ple: bool,
    /// Print one `STEP` line per decode step.
    #[arg(long)]
    per_step: bool,
    /// The driver's readback arm: `blocking` (the id read back per step),
    /// `ring-fixed` (the device-resident token polled, lookahead 1, a commit
    /// every 8 steps) or `ring` (the same under the adaptive commit).
    #[arg(long, default_value = "blocking")]
    readback: String,
    /// The PLE's host fill on a ring arm (§20.13 option A): `false` reads
    /// each decode step's input id back instead.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    host_fill: bool,
}

fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var_os("HOME")?)
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
fn shards(first: &Path) -> Vec<PathBuf> {
    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let Some(pos) = name.find("-of-") else {
        return vec![first.to_path_buf()];
    };
    let count: usize = name[pos + 4..]
        .trim_end_matches(".gguf")
        .parse()
        .unwrap_or(1);
    let head = &name[..pos.saturating_sub(5)];
    let v: Vec<PathBuf> = (1..=count)
        .map(|k| first.with_file_name(format!("{head}{k:05}-of-{count:05}.gguf")))
        .filter(|p| p.exists())
        .collect();
    if v.is_empty() {
        vec![first.to_path_buf()]
    } else {
        v
    }
}

/// A read-only map of the shards, to drop PLE rows' pages from the page
/// cache between runs.
struct Evictor {
    map: memmap2::Mmap,
    _file: std::fs::File,
}

impl Evictor {
    /// The shard whose header holds the PLE table, mapped read-only.
    fn new(shards: &[PathBuf]) -> Result<Self> {
        use lloom_tensor::quantized::gguf_file::Content;
        for p in shards {
            let mut f = std::fs::File::open(p)?;
            let c = Content::read(&mut f)?;
            if c.tensor_infos
                .contains_key(lloom::nn::ngram_memory::PER_LAYER_TOKEN_EMBD)
            {
                // SAFETY: read-only map of a file this process keeps open.
                let map = unsafe { memmap2::Mmap::map(&f)? };
                return Ok(Self { map, _file: f });
            }
        }
        anyhow::bail!("no shard holds the PLE table")
    }

    /// Drop the pages of every row `stream`'s tokens hash to (in the model's
    /// order: the window starts empty and advances past each token). Returns
    /// (pages, still resident).
    fn evict_stream(
        &self,
        hash: &lloom::models::qwen4exp::NgramHash,
        table: &lloom::nn::ngram_memory::MmapRows,
        stream: &[u32],
    ) -> Result<(usize, usize)> {
        let map = &self.map;
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let mut pages = std::collections::BTreeSet::new();
        let mut w = NgramWindow::new(hash.context_len());
        let mut ids = vec![0i64; hash.n_heads()];
        for &t in stream {
            hash.row_ids(&w.window(Some(t)), &mut ids);
            w.push(Some(t));
            for &id in &ids {
                let off = table.row_file_offset(id)? as usize;
                pages.insert(off / page);
                pages.insert((off + table.row_bytes() - 1) / page);
            }
        }
        let base = map.as_ptr() as *mut libc::c_void;
        let mut still = 0;
        for &p in &pages {
            // SAFETY: one page inside the map.
            unsafe {
                let a = (base as *mut u8).add(p * page) as *mut libc::c_void;
                libc::msync(a, page, libc::MS_INVALIDATE);
                let mut v = 0u8;
                libc::mincore(a, page, &mut v as *mut u8 as *mut libc::c_char);
                still += (v & 1) as usize;
            }
        }
        Ok((pages.len(), still))
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

fn dist(v: &mut [f64]) -> String {
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n.max(1) as f64;
    format!(
        "n={n} p5_ms={:.3} p50_ms={:.3} p90_ms={:.3} max_ms={:.3} mean_ms={mean:.3}",
        pct(v, 0.05),
        pct(v, 0.5),
        pct(v, 0.9),
        pct(v, 1.0)
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        profile::enabled() || args.unarmed,
        "set CANDLE_METAL_PROFILE=1: the window is read off the profile's command-buffer records"
    );
    let gguf = match &args.gguf {
        Some(p) => p.clone(),
        None => hf_snapshot("models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
            .context("no HF snapshot for the GGUF; pass --gguf")?
            .join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    };
    let tok_path = match &args.tokenizer {
        Some(p) => p.clone(),
        None => hf_snapshot("models--Qwen--Qwen3.8-Flash-Next")
            .context("no HF snapshot for the tokenizer; pass --tokenizer")?
            .join("tokenizer.json"),
    };
    let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
    let mut eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect();
    eos_ids.dedup();
    // The checkpoint's chat template with thinking disabled, as the harness.
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        args.prompt
    );
    let prompt_ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let device = Device::new_metal(0)?;
    let opts = LoadOptions {
        ple_shard: None,
        ple: true,
        moe_act: None,
        ple_cache_rows: None,
        mtp: false,
        memory_fraction: Some(args.memory_fraction),
    };
    let slots = match args.slots.as_str() {
        "auto" => SlotBudget::Auto {
            fraction: args.memory_fraction,
            rows: 1,
            kv_tokens: 4096,
            prefill_tile: 512,
        },
        n => SlotBudget::Fixed(n.parse().context("--slots: a count or `auto`")?),
    };
    let stream = StreamOptions {
        slots,
        loader: StreamLoader::Pread,
        prefetch: true,
        io_hints: false,
    };
    let mut model: Qwen4Exp<Streamed> =
        Qwen4Exp::from_gguf_streamed(&shards(&gguf), &device, &opts, &stream)?;
    model.set_memory_fraction(args.memory_fraction);
    model.set_host_fill(args.host_fill);
    let (lookahead, adaptive) = match args.readback.as_str() {
        "blocking" => (0, false),
        "ring-fixed" => (1, false),
        "ring" => (1, true),
        other => anyhow::bail!("--readback {other}: blocking | ring-fixed | ring"),
    };
    if !args.unarmed {
        model.arm_ple_window();
    }
    let cfg = model.config().clone();
    let policy = match args.ple_read.as_str() {
        "serial" => ReadPolicy::Serial,
        "adaptive" => ReadPolicy::Adaptive,
        "concurrent" => ReadPolicy::Concurrent,
        other => anyhow::bail!("--ple-read {other}: serial | adaptive | concurrent"),
    };
    let table = model
        .ple_table()
        .and_then(|t| t.mmap())
        .context("the PLE table is not memory-mapped")?;
    table.set_read_policy(policy);
    let evictor = if args.evict_ple {
        Some(Evictor::new(&shards(&gguf))?)
    } else {
        None
    };
    println!(
        "LOADED slots={} ple={} armed={} prompt_tokens={} ple_read={policy:?} evict_ple={} readback={} host_fill={}",
        model.slots_per_layer(),
        model.has_ple(),
        !args.unarmed,
        prompt_ids.len(),
        args.evict_ple,
        args.readback,
        lookahead > 0 && args.host_fill
    );

    // The harness's tile rule (`qwen4exp-chat` `driver_gate`).
    let tile = if prefill_pair_on() && 512 <= prefill_chunk() {
        prefill_tiles() * prefill_chunk()
    } else {
        512
    };
    let mut last_stream: Vec<u32> = Vec::new();
    for run in 0..args.runs {
        if let (Some(ev), false) = (&evictor, last_stream.is_empty()) {
            let hash = model.ple_hash().context("no PLE hash")?;
            let (pages, still) = ev.evict_stream(hash, table, &last_stream)?;
            println!("EVICT run={run} pages={pages} resident_after={still}");
        }
        let (mapped0, pooled0) = table.read_counters();
        let cache = model.new_cache(&cfg, &device)?;
        let mut session = Session::new(&model, cache);
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut spec = IdentityProposer;
        let mut dopts = DecodeOptions {
            prefill_tile: tile,
            ..DecodeOptions::new(args.max_tokens)
        };
        if lookahead > 0 {
            dopts.lookahead = lookahead;
            dopts.budget = CommitBudget::steps(8)?;
            dopts.adaptive_commit = adaptive;
            dopts.stream_via_reader = adaptive;
        }
        profile::reset();
        let t = std::time::Instant::now();
        let out = if lookahead == 0 {
            let mut rb = Blocking::new(&eos_ids, &device)?;
            session.decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt_ids,
                &dopts,
                &mut |_| Ok(()),
            )?
        } else {
            let mut rb = Ring::new(
                &eos_ids,
                &device,
                0,
                EosRead::Poll(PollMechanism::default()),
            )?;
            session.decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt_ids,
                &dopts,
                &mut |_| Ok(()),
            )?
        };
        let wall_ms = t.elapsed().as_secs_f64() * 1e3;
        let prefill_ms = out.stats.prefill_ns as f64 / 1e6;
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
        let decode_ms = (wall_ms - prefill_ms).max(0.0);
        let (mapped1, pooled1) = table.read_counters();
        println!(
            "RESULT run={run} tokens_sha256={digest} generated={} hit_eos={} prefill_ms={prefill_ms:.0} decode_ms={decode_ms:.0} ms_per_token={:.3} ple_rows_mapped={} ple_rows_pooled={} readback={} host_fill_forwards={} host_fill_fills={} overrun={}",
            out.ids.len(),
            out.eos,
            decode_ms / out.ids.len().max(1) as f64,
            mapped1 - mapped0,
            pooled1 - pooled0,
            args.readback,
            out.stats.host_feed.as_ref().map_or(0, |f| f.forwards),
            out.stats.host_feed.as_ref().map_or(0, |f| f.fills),
            out.stats.overrun,
        );
        if let Some(f) = &out.stats.host_feed {
            if !f.fill_ns.is_empty() {
                let mut v: Vec<f64> = f.fill_ns.iter().map(|&n| n as f64 / 1e6).collect();
                println!("HOSTFILL run={run} {}", dist(&mut v));
            }
        }
        last_stream = prompt_ids.iter().chain(&out.ids).copied().collect();
        if args.unarmed {
            continue;
        }

        let snap = profile::snapshot();
        let recs = &snap.command_buffer_records;
        let marks = model.take_ple_window_marks();
        anyhow::ensure!(
            !marks.is_empty(),
            "no PLE-window marks: was the PLE attached?"
        );
        let (mut win_busy, mut win_span, mut rest_busy, mut step_busy, mut frac) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut win_bufs, mut rest_bufs) = (Vec::new(), Vec::new());
        for (i, &(entry, cut)) in marks.iter().enumerate() {
            let next = marks.get(i + 1).map(|m| m.0).unwrap_or(f64::INFINITY);
            let head: Vec<_> = recs
                .iter()
                .filter(|r| r.committed >= entry && r.committed <= cut)
                .collect();
            let tail: Vec<_> = recs
                .iter()
                .filter(|r| r.committed > cut && r.committed < next)
                .collect();
            let busy = |v: &[&profile::CommandBufferRecord]| -> f64 {
                v.iter()
                    .map(|r| r.interval.end - r.interval.start)
                    .sum::<f64>()
                    * 1e3
            };
            let (hb, tb) = (busy(&head), busy(&tail));
            let span = match (
                head.iter().map(|r| r.interval.start).reduce(f64::min),
                head.iter().map(|r| r.interval.end).reduce(f64::max),
            ) {
                (Some(a), Some(b)) => (b - a) * 1e3,
                _ => 0.0,
            };
            if args.per_step {
                println!(
                "STEP run={run} {i} window_busy_ms={hb:.3} window_span_ms={span:.3} window_buffers={} rest_busy_ms={tb:.3} rest_buffers={} host_entry_to_cut_ms={:.3}",
                head.len(),
                tail.len(),
                (cut - entry) * 1e3
            );
            }
            // The last step's tail runs to the end of the records: not a full step.
            if i < args.skip || i + 1 == marks.len() {
                continue;
            }
            win_busy.push(hb);
            win_span.push(span);
            rest_busy.push(tb);
            step_busy.push(hb + tb);
            frac.push(hb / (hb + tb).max(1e-9));
            win_bufs.push(head.len() as f64);
            rest_bufs.push(tail.len() as f64);
        }
        println!("WINDOW run={run} busy {}", dist(&mut win_busy));
        println!("WINDOW run={run} span {}", dist(&mut win_span));
        println!("REST run={run} busy {}", dist(&mut rest_busy));
        println!("STEPBUSY run={run} {}", dist(&mut step_busy));
        println!(
        "WINDOW run={run} fraction_of_step_p50={:.4} buffers_p50={} rest_buffers_p50={} marks={} records={}",
        pct(&mut frac, 0.5),
        pct(&mut win_bufs, 0.5),
        pct(&mut rest_bufs, 0.5),
        marks.len(),
        recs.len()
    );
        let s = model.stream_stats();
        println!(
            "STREAM run={run} hits={} misses={} device_plans={} settled_misses={}",
            s.hits, s.misses, s.device_plans, s.settled_misses
        );
    }
    Ok(())
}
