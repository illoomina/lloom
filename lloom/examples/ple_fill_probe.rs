//! The host fill of Qwen4-exp's N-gram PLE rows, measured over real token
//! streams (`design/docs/expert-streaming.md` §20.13 probe 1, §20.8 #4a).
//!
//! Per token the fill is: advance the n-gram window, hash it to 16 row ids,
//! read the 16 packed rows (90 B each, scattered over a 28.8 GB shard) and
//! dequantize them into a staging slot of 2560 floats. The slot here is a
//! host vector — the same unified memory a Shared Metal buffer's contents
//! are; the event signal that would follow costs 0.25 µs (`host_table_probe
//! event`) and is not repeated here.
//!
//! The arms:
//! * `mmap` — `PleTable::Mmap::gather_into`, the production read through the
//!   memory map, one row after another;
//! * `pread` — the same rows by `pread`, one after another;
//! * `pread-par` — the 16 `pread`s concurrently on a rayon pool (ds4's
//!   `d0b7434` shape);
//! * `cached:N` — `PleTable::Cached` with N rows: the hot-row cache's hit
//!   rate on the stream (`PLECACHE`), cold and again warm.
//!
//! Cold is made, not assumed: before an arm's cold pass every page the
//! stream's rows touch is dropped with `msync(MS_INVALIDATE)` (it drops a
//! clean page nothing pins — `host_table_probe nocopy`'s `evict_control`),
//! and each token's rows are classified resident or not with `mincore`
//! before its fill is timed, so a latency is reported beside the number of
//! its rows that had to come from the SSD.
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/ple_fill_probe \
//!     --shard <…-00002-of-00002.gguf> --text prose=<file> --text structured=<file>
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use lloom::models::qwen4exp::{CachedPleTable, MmapPleTable, NgramHash, NgramWindow, PleTable};
use rayon::prelude::*;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Latency and hit rate of the N-gram PLE's host fill (§20.13 probe 1)")]
struct Args {
    /// The GGUF shard holding `per_layer_token_embd.weight`.
    #[arg(long)]
    shard: PathBuf,
    /// tokenizer.json of the model.
    #[arg(long)]
    tokenizer: PathBuf,
    /// `label=path` text streams, tokenized without special tokens.
    #[arg(long)]
    text: Vec<String>,
    /// `label=path` token streams: comma-separated ids (a harness's `TOKENS`
    /// line, the prompt's ids first when they are wanted in the window).
    #[arg(long)]
    tokens: Vec<String>,
    /// A text streamed through each cache arm first (a cache that has seen
    /// earlier conversations), reported as a separate `history` pass.
    #[arg(long)]
    history: Option<PathBuf>,
    /// Tokens per stream, from its start.
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,
    /// Hot-row cache sizes (rows).
    #[arg(long, value_delimiter = ',', default_values_t = [16384usize, 65536, 262144, 1048576])]
    cache_rows: Vec<usize>,
    /// Latency arms.
    #[arg(long, value_delimiter = ',', default_values_t = ["mmap".to_string(), "pread".to_string(), "pread-par".to_string()])]
    arms: Vec<String>,
    /// Threads of the `pread-par` pool.
    #[arg(long, default_value_t = 16)]
    par_threads: usize,
}

/// The checkpoint's hash: the reference derivation, which
/// `ple::tests::derivation_reproduces_checkpoint_constants` pins to the
/// constants the GGUF records.
fn checkpoint_hash() -> Result<NgramHash> {
    Ok(NgramHash::derive(
        1234, 248320, 3, 8, 20_000_000, 0, 248044,
    )?)
}

/// A read-only map of the whole shard for residency queries and eviction —
/// its own mapping; the page cache it reports on is the file's.
struct Pages {
    ptr: *mut libc::c_void,
    len: usize,
    page: usize,
}

impl Pages {
    fn new(file: &std::fs::File) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let len = file.metadata()?.len() as usize;
        // SAFETY: a fresh read-only mapping of an open file, unmapped on drop.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            bail!("mmap: {}", std::io::Error::last_os_error())
        }
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        Ok(Self { ptr, len, page })
    }

    /// The pages `[off, off + n)` spans.
    fn span(&self, off: u64, n: usize) -> std::ops::RangeInclusive<usize> {
        (off as usize / self.page)..=((off as usize + n - 1) / self.page)
    }

    fn resident(&self, page: usize) -> bool {
        let mut v = [0u8; 1];
        // SAFETY: one page inside the mapping.
        unsafe {
            libc::mincore(
                (self.ptr as *mut u8).add(page * self.page) as *mut libc::c_void,
                self.page,
                v.as_mut_ptr() as *mut libc::c_char,
            );
        }
        v[0] as i32 & libc::MINCORE_INCORE != 0
    }

    fn evict(&self, page: usize) -> i32 {
        // SAFETY: one page inside the mapping.
        unsafe {
            libc::msync(
                (self.ptr as *mut u8).add(page * self.page) as *mut libc::c_void,
                self.page,
                libc::MS_INVALIDATE,
            )
        }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// Per-token row ids of a stream, as `ple_delta` hashes them: the window
/// starts empty (the sequence start) and advances past every token.
fn row_ids(hash: &NgramHash, tokens: &[u32]) -> Vec<Vec<i64>> {
    let mut w = NgramWindow::new(hash.context_len());
    tokens
        .iter()
        .map(|&t| {
            let mut ids = vec![0i64; hash.n_heads()];
            hash.row_ids(&w.window(Some(t)), &mut ids);
            w.push(Some(t));
            ids
        })
        .collect()
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
    if n == 0 {
        return "n=0".into();
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    format!(
        "n={n} p5_us={:.2} p50_us={:.2} p90_us={:.2} p99_us={:.2} max_us={:.2} mean_us={mean:.2}",
        pct(v, 0.05),
        pct(v, 0.5),
        pct(v, 0.9),
        pct(v, 0.99),
        pct(v, 1.0)
    )
}

/// The rows of one token by `pread` + dequant into `out`, serially or on the
/// pool.
fn pread_fill(
    table: &MmapPleTable,
    file: &std::fs::File,
    ids: &[i64],
    out: &mut [f32],
    pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    let hd = table.head_dim();
    let rb = table.row_bytes();
    let one = |id: i64, dst: &mut [f32]| -> Result<()> {
        let mut b = [0u8; 256];
        let b = &mut b[..rb];
        file.read_exact_at(b, table.row_file_offset(id)?)?;
        Ok(table.dequant_packed(b, dst)?)
    };
    match pool {
        None => {
            for (i, &id) in ids.iter().enumerate() {
                one(id, &mut out[i * hd..(i + 1) * hd])?;
            }
            Ok(())
        }
        Some(p) => p.install(|| {
            ids.par_iter()
                .zip(out.par_chunks_mut(hd))
                .try_for_each(|(&id, dst)| one(id, dst))
        }),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let hash = checkpoint_hash()?;
    let tok = tokenizers::Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let encode = |path: &std::path::Path| -> Result<Vec<u32>> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let enc = tok.encode(text, false).map_err(anyhow::Error::msg)?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(args.max_tokens);
        Ok(ids)
    };
    let file = std::fs::File::open(&args.shard)?;
    let pages = Pages::new(&file)?;
    let table = MmapPleTable::open(&args.shard)?;
    let hd = table.head_dim();
    let n_heads = hash.n_heads();
    if (hash.total_rows() as usize) > table.rows() {
        bail!(
            "hash addresses {} rows, the table has {}",
            hash.total_rows(),
            table.rows()
        )
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.par_threads)
        .build()?;
    println!(
        "TABLE rows={} head_dim={hd} row_bytes={} dtype={:?} heads={n_heads} page={}",
        table.rows(),
        table.row_bytes(),
        table.dtype(),
        pages.page
    );
    let history = match &args.history {
        Some(p) => Some(row_ids(&hash, &encode(p)?)),
        None => None,
    };

    let mut streams: Vec<(String, String, Vec<u32>)> = Vec::new();
    for spec in &args.text {
        let (label, path) = spec
            .split_once('=')
            .with_context(|| format!("--text {spec}: want label=path"))?;
        streams.push((
            label.into(),
            path.into(),
            encode(std::path::Path::new(path))?,
        ));
    }
    for spec in &args.tokens {
        let (label, path) = spec
            .split_once('=')
            .with_context(|| format!("--tokens {spec}: want label=path"))?;
        let raw = std::fs::read_to_string(path).with_context(|| path.to_string())?;
        let mut ids: Vec<u32> = raw
            .trim()
            .trim_start_matches("TOKENS ")
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()
            .with_context(|| format!("{path}: not a comma-separated id list"))?;
        ids.truncate(args.max_tokens);
        streams.push((label.into(), path.into(), ids));
    }
    anyhow::ensure!(!streams.is_empty(), "no --text or --tokens stream");
    for (label, path, tokens) in &streams {
        let tokens = tokens.clone();
        // The hash alone, timed per token.
        let mut hash_us = Vec::with_capacity(tokens.len());
        let mut w = NgramWindow::new(hash.context_len());
        let mut scratch = vec![0i64; n_heads];
        for &t in &tokens {
            let s = Instant::now();
            hash.row_ids(&w.window(Some(t)), &mut scratch);
            w.push(Some(t));
            hash_us.push(s.elapsed().as_secs_f64() * 1e6);
        }
        let rows = row_ids(&hash, &tokens);
        let total = rows.len() * n_heads;
        let distinct: std::collections::HashSet<i64> = rows.iter().flatten().copied().collect();
        let mut pages_of_stream: Vec<usize> = rows
            .iter()
            .flatten()
            .flat_map(|&id| pages.span(table.row_file_offset(id).unwrap(), table.row_bytes()))
            .collect();
        pages_of_stream.sort_unstable();
        pages_of_stream.dedup();
        println!(
            "STREAM label={label} path={path} tokens={} rows={total} distinct_rows={} compulsory_miss={:.4} pages={} hash {}",
            tokens.len(),
            distinct.len(),
            distinct.len() as f64 / total.max(1) as f64,
            pages_of_stream.len(),
            dist(&mut hash_us)
        );

        let mut slot = vec![0f32; n_heads * hd];
        for arm in &args.arms {
            let evict_rc: i32 = pages_of_stream.iter().map(|&p| pages.evict(p)).sum();
            let still = pages_of_stream
                .iter()
                .filter(|&&p| pages.resident(p))
                .count();
            println!(
                "EVICT label={label} arm={arm} pages={} msync_rc_sum={evict_rc} resident_after={still}",
                pages_of_stream.len()
            );
            for pass in ["cold", "warm"] {
                let mut all = Vec::new();
                let mut by_cold: [Vec<f64>; 4] = Default::default();
                let mut cold_rows = 0usize;
                let mut w = NgramWindow::new(hash.context_len());
                let mut ids = vec![0i64; n_heads];
                for &t in &tokens {
                    // Classify this token's rows before its fill is timed.
                    let window = w.window(Some(t));
                    hash.row_ids(&window, &mut ids);
                    let cold = ids
                        .iter()
                        .filter(|&&id| {
                            pages
                                .span(table.row_file_offset(id).unwrap(), table.row_bytes())
                                .any(|p| !pages.resident(p))
                        })
                        .count();
                    cold_rows += cold;
                    let s = Instant::now();
                    hash.row_ids(&window, &mut ids);
                    match arm.as_str() {
                        "mmap" => table.gather_into(&ids, &mut slot)?,
                        "pread" => pread_fill(&table, &file, &ids, &mut slot, None)?,
                        "pread-par" => pread_fill(&table, &file, &ids, &mut slot, Some(&pool))?,
                        other => bail!("unknown arm {other}"),
                    }
                    let us = s.elapsed().as_secs_f64() * 1e6;
                    w.push(Some(t));
                    all.push(us);
                    let b = match cold {
                        0 => 0,
                        1..=3 => 1,
                        4..=8 => 2,
                        _ => 3,
                    };
                    by_cold[b].push(us);
                }
                println!(
                    "FILL label={label} arm={arm} pass={pass} cold_rows={cold_rows} of={total} {}",
                    dist(&mut all)
                );
                for (b, name) in ["0", "1-3", "4-8", "9-16"].iter().enumerate() {
                    println!(
                        "FILL label={label} arm={arm} pass={pass} cold_rows_per_token={name} {}",
                        dist(&mut by_cold[b])
                    );
                }
            }
        }

        // The hot-row cache: its hit rate on this stream, from empty, again
        // warm, and after a history stream.
        for &n in &args.cache_rows {
            let cached =
                PleTable::Cached(CachedPleTable::new(MmapPleTable::open(&args.shard)?, n)?);
            let mut feed = |rows: &[Vec<i64>]| -> Result<(u64, u64)> {
                let (h0, m0) = cached.cache_stats();
                for ids in rows {
                    cached.gather_into(ids, &mut slot)?;
                }
                let (h1, m1) = cached.cache_stats();
                Ok((h1 - h0, m1 - m0))
            };
            let (h, m) = feed(&rows)?;
            let (hw, mw) = feed(&rows)?;
            let rate = |h: u64, m: u64| h as f64 / (h + m).max(1) as f64;
            let mut line = format!(
                "PLECACHE label={label} rows={n} mib={:.1} first_pass_hits={h} misses={m} hit_rate={:.4} second_pass_hit_rate={:.4}",
                (n * table.row_bytes()) as f64 / 1048576.0,
                rate(h, m),
                rate(hw, mw)
            );
            if let Some(hist) = &history {
                let cached =
                    PleTable::Cached(CachedPleTable::new(MmapPleTable::open(&args.shard)?, n)?);
                for ids in hist {
                    cached.gather_into(ids, &mut slot)?;
                }
                let (h0, m0) = cached.cache_stats();
                for ids in &rows {
                    cached.gather_into(ids, &mut slot)?;
                }
                let (h1, m1) = cached.cache_stats();
                line += &format!(" after_history_hit_rate={:.4}", rate(h1 - h0, m1 - m0));
            }
            println!("{line}");
        }
    }
    Ok(())
}
