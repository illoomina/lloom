//! Load-phase profiler: split the cold-start cost into its terms.
//!
//! `DESIGN.md` §2.4 records first-touch page-in of the 5.39 GB weight set at
//! **13.3–16.5 s** and disposes of it by discarding the cold pass. That is the
//! right discipline for making a *prefill* number readable and it is not a
//! disposition on the cost itself, which §5.5a puts in user-visible units
//! (~2.2 s for the vision tower's share alone, "or 116 decode tokens").
//!
//! Issue #298 asks for the **split before the fix**: the 13.3–16.5 s contains
//! at least page-in, a BF16→F16 `to_dtype` conversion, and §9.1b's ~530
//! `commit()` calls, and no measurement separates them. Attacking a residual
//! before pricing it is the shape §6.6c warns about.
//!
//! # What this measures, and why it is a separate binary
//!
//! Four phases, each timed on `CLOCK_UPTIME_RAW` (§3.4b domain A, the same
//! clock `lloom-sample` brackets its reads in):
//!
//! | phase | what it is |
//! |---|---|
//! | `mmap` | opening the file and mapping it; no page is touched |
//! | `pagein` | faulting the mapped range in, one byte per page |
//! | `model` | `Model::new` — `to_dtype` and the per-tensor allocations |
//! | `cache` | `Cache::new_with` — KV, conv state, the RoPE tables |
//!
//! **`pagein` is a phase this harness creates rather than one the engine
//! has.** In the shipping path there is no page-in step: the pages fault in
//! *inside* `Model::new`, interleaved with the conversion, which is exactly
//! why §2.4 could only ever measure the two together. Sweeping the mapping
//! first separates them — and the separation is the deliverable, so the sweep
//! is reported as its own column rather than folded into either neighbour.
//! `--no-sweep` runs the shipping shape, where `model` carries both.
//!
//! A **separate binary** rather than a flag on `lfm2-decode-profile`, for
//! §2.3.8d's reason: two quantities sharing one harness is what made
//! `tok_per_s` mislead two readers over twenty-odd issues (§2.4). That
//! harness measures steady-state decode and averages its warmup away; this one
//! measures the thing that only happens once, and folding them would put a
//! cold-start figure on a line where readers look for a per-token one.
//!
//! # The cold-cache requirement, which is the whole measurement
//!
//! A warm run measures the page cache and answers nothing. This binary does
//! **not** evict anything itself — eviction is the caller's, so that the
//! harness cannot silently believe its own precondition — and instead
//! **reports the residency it observed** via `mincore(2)` before it starts.
//! `resident_frac_pre` on the RESULT line is the gate: a run reporting
//! anything but 0.000000 is a warm run and its `pagein` figure is not a
//! page-in figure.
//!
//! That is §2.4's rule about an instrument that cannot be shown to have
//! engaged, applied to the precondition rather than to the flag: `purge(8)`
//! needs root and **exits 0 while printing "Operation not permitted"**, so a
//! batch that shells out to it and checks the status believes the cache is
//! cold when it is warm — in the one place where being wrong makes every
//! result look enormous.
//!
//! # The floor
//!
//! `CONTRIBUTING.md` §1.3: a speedup that is too large is a bug signal, and on
//! a 13–16 s baseline that trap is wide open. The physical floor is what the
//! storage can deliver: this binary reports `read_floor_s` — the same bytes
//! pulled through `read(2)` into a heap buffer on the same cold cache — so any
//! claimed improvement can be checked against it in the same artifact rather
//! than against a number quoted from elsewhere. **No arm can beat the floor**,
//! and one that appears to has measured a warm cache.

use anyhow::{Context, Result};
use candle::{DType, Device};
use lloom_engine::nn::VarBuilder;
use lloom_engine::models::lfm2::{Cache, Config, Lfm2Config, Model};
use clap::Parser;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Split the LFM2 cold load into page-in / to_dtype / cache")]
struct Args {
    /// Local checkpoint directory (config.json, tokenizer.json, weights).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Sweep the mapping to separate page-in from conversion.
    ///
    /// On by default *because the split is the deliverable*. `--no-sweep`
    /// reproduces the shipping shape, where the pages fault in inside
    /// `Model::new` and the two terms cannot be told apart — which is the
    /// state §2.4 measured and the reason this issue exists.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    sweep: bool,

    /// Also time a `read(2)` of the whole checkpoint, to report the floor.
    ///
    /// Off by default because it *warms the cache*, so it must be the only
    /// thing a process does. Run it in its own process against its own cold
    /// cache; do not combine it with a load in the same invocation.
    #[arg(long)]
    read_floor: bool,

    /// Decode dtype. The checkpoint ships BF16 (§5.1); decode runs F16.
    #[arg(long, default_value = "f16")]
    dtype: String,
}

fn now() -> Instant {
    Instant::now()
}

/// Fraction of `path` resident in the page cache, read via `mincore(2)`.
///
/// This is the cold-cache gate. Mapping a file to ask about it does not fault
/// its pages: `mmap` establishes the mapping and `mincore` reports residency
/// without touching a page, which is what makes it safe to call immediately
/// before the measurement it guards.
#[cfg(unix)]
fn resident_fraction(path: &Path) -> Result<f64> {
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Ok(0.0);
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let npages = len.div_ceil(page);

    // SAFETY: a read-only shared mapping of a file we just opened; unmapped
    // below. The pointer is not dereferenced -- only handed to `mincore`.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        anyhow::bail!("mmap for residency check failed");
    }
    let mut vec = vec![0i8; npages];
    // SAFETY: `addr` is a valid mapping of `len` bytes and `vec` has one byte
    // per page of it, which is the contract `mincore` documents.
    let rc = unsafe { libc::mincore(addr, len, vec.as_mut_ptr()) };
    // SAFETY: unmapping the mapping created directly above.
    unsafe { libc::munmap(addr, len) };
    if rc != 0 {
        anyhow::bail!("mincore failed");
    }
    let resident = vec.iter().filter(|b| **b & 1 != 0).count();
    Ok(resident as f64 / npages as f64)
}

#[cfg(not(unix))]
fn resident_fraction(_path: &Path) -> Result<f64> {
    Ok(f64::NAN)
}

/// Fault in every page of the mapped checkpoint, and report how long it took.
///
/// One byte per page is the minimum that forces a fault, so this measures
/// fault handling rather than bandwidth to the touched bytes. It is
/// deliberately the same shape the engine's own first sweep has: the weights
/// are read once each, in file order.
#[cfg(unix)]
fn sweep_pages(paths: &[PathBuf]) -> Result<(f64, u64)> {
    use std::os::unix::io::AsRawFd;

    let t0 = now();
    let mut bytes = 0u64;
    let mut acc = 0u64;
    for p in paths {
        let file = std::fs::File::open(p)?;
        let len = file.metadata()?.len() as usize;
        bytes += len as u64;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        // SAFETY: read-only private mapping of a file we just opened, unmapped
        // below; only read through, one byte per page.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            anyhow::bail!("mmap for sweep failed");
        }
        let base = addr as *const u8;
        let mut off = 0usize;
        while off < len {
            // SAFETY: `off < len` and `base` maps `len` bytes.
            acc = acc.wrapping_add(unsafe { std::ptr::read_volatile(base.add(off)) } as u64);
            off += page;
        }
        // SAFETY: unmapping the mapping created directly above.
        unsafe { libc::munmap(addr, len) };
    }
    // `acc` is consumed so the sweep cannot be optimised away.
    std::hint::black_box(acc);
    Ok((t0.elapsed().as_secs_f64(), bytes))
}

#[cfg(not(unix))]
fn sweep_pages(_paths: &[PathBuf]) -> Result<(f64, u64)> {
    Ok((0.0, 0))
}

/// Pull the whole checkpoint through `read(2)`. The storage floor.
fn read_floor(paths: &[PathBuf]) -> Result<(f64, u64)> {
    use std::io::Read;
    let t0 = now();
    let mut total = 0u64;
    let mut buf = vec![0u8; 8 << 20];
    let mut acc = 0u64;
    for p in paths {
        let mut f = std::fs::File::open(p)?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n as u64;
            acc = acc.wrapping_add(buf[0] as u64).wrapping_add(buf[n - 1] as u64);
        }
    }
    std::hint::black_box(acc);
    Ok((t0.elapsed().as_secs_f64(), total))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_dir = args
        .model_dir
        .clone()
        .or_else(default_model_dir)
        .context("no --model-dir and no LFM2 checkpoint in the HF cache")?;

    let dtype = match args.dtype.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        other => anyhow::bail!("--dtype must be f16, bf16 or f32, got `{other}`"),
    };

    let files = weight_files(&model_dir)?;

    // The cold-cache gate, read BEFORE anything touches the file. A run whose
    // pre-residency is not 0 is a warm run; this reports the number rather
    // than asserting on it, so a warm run is visibly warm in the artifact
    // instead of being silently rejected or silently believed.
    let resident_pre = resident_fraction(&files[0]).unwrap_or(f64::NAN);

    if args.read_floor {
        // The floor arm does nothing else, because reading the file warms the
        // cache and would make any load in the same process meaningless.
        let (secs, bytes) = read_floor(&files)?;
        println!(
            "RESULT arm=read-floor resident_frac_pre={resident_pre:.6} \
             read_floor_s={secs:.4} bytes={bytes} read_GBps={:.3}",
            bytes as f64 / secs / 1e9
        );
        return Ok(());
    }

    let device = Device::new_metal(0).or_else(|_| Ok::<_, anyhow::Error>(Device::Cpu))?;

    let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let config = parse_config(&raw)?;

    // ---- phase 1: map -----------------------------------------------------
    // Opening and mapping only. No page is touched, so this is the header
    // parse and the mapping call; it is expected to be small and is reported
    // so that "small" is a measurement rather than an assumption.
    let t_mmap = now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let mmap_s = t_mmap.elapsed().as_secs_f64();

    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");
    let vb = if nested {
        vb.rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        })
    } else {
        vb
    };

    // ---- phase 2: page-in -------------------------------------------------
    // Optional, and it is the phase that makes the split possible at all.
    let (pagein_s, swept_bytes) = if args.sweep {
        sweep_pages(&files)?
    } else {
        (0.0, 0)
    };
    let resident_mid = resident_fraction(&files[0]).unwrap_or(f64::NAN);

    // ---- phase 3: model construction (to_dtype + allocations) -------------
    let t_model = now();
    let model = Model::new(&config, vb).context("constructing LFM2 model")?;
    let model_s = t_model.elapsed().as_secs_f64();

    // ---- phase 4: cache ---------------------------------------------------
    let t_cache = now();
    let _cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;
    let cache_s = t_cache.elapsed().as_secs_f64();

    // Keep the model alive across the cache phase so nothing is dropped early.
    std::hint::black_box(&model);

    let advice = candle::safetensors::MmapAdvice::from_env()?;
    let advice_ns =
        candle::safetensors::MMAP_ADVICE_NANOS.load(std::sync::atomic::Ordering::Relaxed);
    let advice_calls =
        candle::safetensors::MMAP_ADVICE_CALLS.load(std::sync::atomic::Ordering::Relaxed);

    let total_s = mmap_s + pagein_s + model_s + cache_s;

    // One line, every term, so a reader never has to add columns from two
    // places. `advice_ms` and `advice_calls` are the engagement proof: an arm
    // claiming `willneed` with `advice_calls=0` did not run the mechanism it
    // names, which is the vacuous-arm failure §2.4 records.
    println!(
        "RESULT arm=load advice={} advice_calls={advice_calls} advice_ms={:.3} \
         sweep={} resident_frac_pre={resident_pre:.6} resident_frac_mid={resident_mid:.6} \
         mmap_s={mmap_s:.4} pagein_s={pagein_s:.4} model_s={model_s:.4} cache_s={cache_s:.4} \
         total_s={total_s:.4} swept_bytes={swept_bytes} dtype={} device={:?}",
        advice.as_str(),
        advice_ns as f64 / 1e6,
        args.sweep,
        args.dtype,
        device,
    );

    Ok(())
}

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

    // §5.2's trap: the stated `intermediate_size` wins over candle's formula.
    if let Some(stated) = normalized
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
    {
        config.intermediate_size = stated;
    }
    Ok(config)
}

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
