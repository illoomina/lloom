//! §14 Q7's measurement (`profiles-and-memory-design.md` §4.8, #892 A6a):
//! does a streamed read go through the unified buffer cache — a second
//! copy of every streamed group beside the slot — and does `F_NOCACHE`
//! avoid it, at what bandwidth? Run by hand; reads the first
//! `LLOOM_PROBE_GB` (default 2) GB of `LLOOM_PROBE_FILE` (a checkpoint
//! shard) four ways — `pread` warm, `pread` + `F_NOCACHE`, a mapped read
//! (`mmap` + a sequential touch), and `pread` + `F_NOCACHE` again — and
//! prints each arm's bandwidth beside the machine's file-backed and free
//! pages (`vm_stat`) before and after:
//!
//!   LLOOM_PROBE_FILE=~/.cache/huggingface/hub/…/model.safetensors \
//!     cargo test -p lloom --features metal --test page_cache_probe -- --ignored --nocapture
#![cfg(target_os = "macos")]
#![allow(clippy::needless_range_loop)]
use std::os::unix::fs::FileExt;

fn vm_stat() -> (u64, u64, u64) {
    let out = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let pages = |key: &str| -> u64 {
        out.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
            .unwrap_or(0)
            * 16384
    };
    (
        pages("File-backed pages"),
        pages("Pages free"),
        pages("Pages wired down"),
    )
}

fn line(what: &str, bytes: usize, secs: f64) {
    let (file, free, wired) = vm_stat();
    let gb = |b: u64| b as f64 / 1e9;
    eprintln!(
        "PAGECACHE {what}: {:.2} GB in {secs:.2} s = {:.2} GB/s; file-backed={:.2} GB free={:.2} GB wired={:.2} GB",
        bytes as f64 / 1e9,
        bytes as f64 / 1e9 / secs.max(1e-9),
        gb(file),
        gb(free),
        gb(wired)
    );
}

#[test]
#[ignore = "a measurement, not a check: reads a checkpoint four ways and prints bandwidths"]
fn page_cache_probe() {
    let Ok(path) = std::env::var("LLOOM_PROBE_FILE") else {
        eprintln!("PAGECACHE: set LLOOM_PROBE_FILE");
        return;
    };
    let gb: usize = std::env::var("LLOOM_PROBE_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let file = std::fs::File::open(&path).unwrap();
    let len = (file.metadata().unwrap().len() as usize).min(gb << 30);
    let chunk = 64 << 20;
    let mut buf = vec![0u8; chunk];
    let (f0, fr0, w0) = vm_stat();
    eprintln!(
        "PAGECACHE before: file-backed={:.2} GB free={:.2} GB wired={:.2} GB ({len} bytes of {path})",
        f0 as f64 / 1e9,
        fr0 as f64 / 1e9,
        w0 as f64 / 1e9
    );
    let nocache = |on: bool| {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, on as libc::c_int) };
        assert_eq!(rc, 0, "fcntl F_NOCACHE");
    };
    let pread_all = |buf: &mut [u8]| -> f64 {
        let t = std::time::Instant::now();
        let mut off = 0usize;
        let mut sink = 0u64;
        while off < len {
            let n = chunk.min(len - off);
            file.read_exact_at(&mut buf[..n], off as u64).unwrap();
            sink = sink
                .wrapping_add(buf[0] as u64)
                .wrapping_add(buf[n - 1] as u64);
            off += n;
        }
        std::hint::black_box(sink);
        t.elapsed().as_secs_f64()
    };
    nocache(false);
    let s = pread_all(&mut buf);
    line("pread (cached)", len, s);
    nocache(true);
    let s = pread_all(&mut buf);
    line("pread + F_NOCACHE", len, s);
    nocache(false);
    // The mapped read: the loader's path today (`SafetensorsGroups::bytes`).
    let map = unsafe { memmap2::MmapOptions::new().len(len).map(&file).unwrap() };
    let t = std::time::Instant::now();
    let mut sink = 0u64;
    let mut off = 0usize;
    while off < len {
        sink = sink.wrapping_add(map[off] as u64);
        off += 16384;
    }
    std::hint::black_box(sink);
    line(
        "mmap touch (one byte a page)",
        len,
        t.elapsed().as_secs_f64(),
    );
    drop(map);
    nocache(true);
    let s = pread_all(&mut buf);
    line("pread + F_NOCACHE (after the map)", len, s);
    nocache(false);
}
