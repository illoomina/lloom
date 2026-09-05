//! Does the Rust cover test agree with #144's Python model? (issue #185)
//!
//! The unit tests in `metal::hazard_audit_tests` check the *mechanism* on
//! synthetic streams -- that each of the three primitives is modelled and that
//! each is load-bearing. They do not check that this implementation agrees with
//! the analysis the project already trusts, and that is the acceptance criterion
//! that makes #185 a test rather than a display:
//!
//! > validated against a **known-good** predicate (#144's, which the edge
//! > analysis proved) **and a known-bad one** (its run-heads mutation, which the
//! > digest gate could not see).
//!
//! So this replays the **same committed trace** through this crate's `cover()`
//! and prints the same six rows `measurements/issue-144-raw/edge-cover.py`
//! prints. Agreement on all six -- over 85595 edges -- is what licenses the Rust
//! path; a disagreement is a finding about one of the two models.
//!
//! # The keying, and why it is `(buffer, offset)` here
//!
//! The committed trace predates this issue and carries **no extent**
//! (§9.2j finding 3), so an interval test is not available on it and this
//! harness reproduces the Python model's key exactly rather than a better one.
//! That is deliberate: the point is agreement with the artifact, and a run
//! keyed differently would be a different measurement wearing the same name.
//!
//! `trace::Binding` now carries `len`, so a *freshly taken* trace supports the
//! interval test the audit runs at runtime. Both are correct; they answer
//! different questions, and mixing them would silently change the denominator.
//!
//! Usage:
//!   hazard_audit_crosscheck <all-steps-trace.gz|txt> <packed-trace.gz|txt>

use lloom_tensor::kernels::metal::encoder::{HazardKind, HazardKinds};
use lloom_tensor::kernels::metal::hazard_audit::{cover, AuditBinding, AuditDispatch};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};

#[derive(Clone)]
struct Raw {
    pos: usize,
    kernel: String,
    grid: (u64, u64, u64),
    tg: (u64, u64, u64),
    mode: String,
    enc: u64,
    barrier: bool,
    binds: Vec<(usize, bool, u64, u64)>, // index, is_output, buf_id, offset
}

/// A decompressed reader over a trace, gzipped or not.
///
/// The child is returned alongside its pipe and reaped when the reader drops,
/// rather than left for the OS. A `gunzip` that fails midway would otherwise
/// look like a short file, and a truncated trace silently changes every count
/// this harness exists to check.
struct Gunzip {
    child: std::process::Child,
    out: std::process::ChildStdout,
}

impl Read for Gunzip {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.out.read(buf)
    }
}

impl Drop for Gunzip {
    fn drop(&mut self) {
        match self.child.wait() {
            Ok(status) if !status.success() => {
                panic!("gunzip failed with {status}; the trace may be truncated")
            }
            Ok(_) => {}
            Err(e) => panic!("could not reap gunzip: {e}"),
        }
    }
}

fn open(path: &str) -> Box<dyn Read> {
    if path.ends_with(".gz") {
        let mut child = Command::new("gunzip")
            .arg("-c")
            .arg(path)
            .stdout(Stdio::piped())
            .spawn()
            .expect("gunzip");
        let out = child.stdout.take().expect("stdout");
        Box::new(Gunzip { child, out })
    } else {
        Box::new(std::fs::File::open(path).expect("open"))
    }
}

/// Parse the dispatch-trace text format into steps.
fn parse(path: &str) -> BTreeMap<u64, Vec<Raw>> {
    let mut steps: BTreeMap<u64, Vec<Raw>> = BTreeMap::new();
    let mut cur: Option<u64> = None;
    let reader = BufReader::new(open(path));

    for line in reader.lines() {
        let line = line.expect("read");
        if let Some(rest) = line.strip_prefix("=== decode[") {
            let idx: u64 = rest
                .split(']')
                .next()
                .and_then(|s| s.parse().ok())
                .expect("step index");
            cur = Some(idx);
            steps.entry(idx).or_default();
            continue;
        }
        let Some(step) = cur else { continue };

        if let Some(b) = parse_binding(&line) {
            if let Some(list) = steps.get_mut(&step) {
                if let Some(last) = list.last_mut() {
                    last.binds.push(b);
                }
            }
            continue;
        }
        if let Some(d) = parse_dispatch(&line) {
            steps.entry(step).or_default().push(d);
        }
    }
    steps
}

fn parse_dispatch(line: &str) -> Option<Raw> {
    // "   0  is_u32_f16 grid=(2, 1, 1) tg=(1024, 1, 1) threadgroups enc=1  BARRIER"
    let t = line.trim();
    if t.starts_with('[') || t.is_empty() {
        return None;
    }
    let mut it = t.split_whitespace();
    let pos: usize = it.next()?.parse().ok()?;
    let kernel = it.next()?.to_string();
    let rest = t;
    let grid = triple(rest, "grid=(")?;
    let tg = triple(rest, "tg=(")?;
    let mode = if rest.contains(" threadgroups ") {
        "threadgroups"
    } else if rest.contains(" threads ") {
        "threads"
    } else {
        return None;
    }
    .to_string();
    let enc: u64 = rest
        .split("enc=")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(Raw {
        pos,
        kernel,
        grid,
        tg,
        mode,
        enc,
        barrier: rest.contains("BARRIER"),
        binds: Vec::new(),
    })
}

fn triple(s: &str, key: &str) -> Option<(u64, u64, u64)> {
    let body = s.split(key).nth(1)?.split(')').next()?;
    let mut p = body.split(',').map(|x| x.trim().parse::<u64>().ok());
    Some((p.next()??, p.next()??, p.next()??))
}

fn parse_binding(line: &str) -> Option<(usize, bool, u64, u64)> {
    // "        [9] in  buf#0 @0xb51458c40 off=0"
    let t = line.trim();
    let rest = t.strip_prefix('[')?;
    let (idx, rest) = rest.split_once(']')?;
    let index: usize = idx.trim().parse().ok()?;
    let rest = rest.trim();
    let is_output = if rest.starts_with("out") {
        true
    } else if rest.starts_with("in") {
        false
    } else {
        return None;
    };
    let buf: u64 = rest
        .split("buf#")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let off: u64 = rest
        .split("off=")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some((index, is_output, buf, off))
}

/// `icb.rs`'s replay-compatibility test: everything an ICB command freezes.
fn replay_compatible(a: &Raw, b: &Raw) -> bool {
    a.kernel == b.kernel
        && a.grid == b.grid
        && a.tg == b.tg
        && a.mode == b.mode
        && a.binds == b.binds
}

/// Build the audit's dispatch list, keyed as the Python model keys: `(buffer,
/// offset)` with a **unit extent**, so equality of `(buf, off)` is exactly
/// overlap and nothing else can alias.
fn to_audit(step: &[Raw], suppress: &dyn Fn(usize) -> bool, runs: &Runs) -> Vec<AuditDispatch> {
    step.iter()
        .enumerate()
        .map(|(i, d)| {
            let bindings = d
                .binds
                .iter()
                .map(|&(_, is_output, buf, off)| AuditBinding {
                    // The buffer id and the offset together identify a slot;
                    // a unit extent makes `overlaps` decide exactly `==`.
                    ptr: buf as usize,
                    offset: off as usize,
                    len: 1,
                    is_output,
                })
                .collect();
            let barrier = d.barrier && !suppress(i);
            let mut kinds = HazardKinds::NONE;
            if barrier {
                // The trace predates per-kind recording, so the attribution
                // columns are not exercised here; the edge cover is.
                kinds.insert(HazardKind::Raw);
            }
            AuditDispatch {
                seq: d.pos as u64,
                kernel: d.kernel.clone(),
                bindings,
                barrier,
                kinds,
                barrier_suppressed: d.barrier && suppress(i),
                encoder: d.enc,
                run: runs.run_of.get(&i).copied(),
                is_run_head: runs.heads.contains(&i),
                icb_barrier: runs.icb_on && runs.run_of.contains_key(&i),
            }
        })
        .collect()
}

struct Runs {
    run_of: BTreeMap<usize, u64>,
    heads: std::collections::BTreeSet<usize>,
    icb_on: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <all-steps.gz> <packed.gz>", args[0]);
        std::process::exit(2);
    }

    let steps = parse(&args[1]);
    let packed = parse(&args[2]);

    let idx: Vec<u64> = steps.keys().copied().filter(|&k| k >= 3).collect();
    let base = &steps[&idx[0]];
    let n = base.len();

    // Positions whose kernel is not `_packed`: excluded for inline constants.
    let pbase = packed
        .iter()
        .find(|(k, _)| **k >= 2)
        .map(|(_, v)| v)
        .expect("packed step");
    assert_eq!(pbase.len(), n, "packed trace length differs");
    let unpacked: std::collections::BTreeSet<usize> = pbase
        .iter()
        .enumerate()
        .filter(|(_, d)| !d.kernel.ends_with("_packed"))
        .map(|(i, _)| i)
        .collect();

    // Positions that vary across steady-state steps.
    let mut varies = std::collections::BTreeSet::new();
    for k in &idx[1..] {
        for (p, (a, b)) in base.iter().zip(steps[k].iter()).enumerate() {
            if !replay_compatible(a, b) {
                varies.insert(p);
            }
        }
    }

    let covered: Vec<usize> = (0..n)
        .filter(|p| !unpacked.contains(p) && !varies.contains(p))
        .collect();
    let mut run_list: Vec<Vec<usize>> = Vec::new();
    for &p in &covered {
        match run_list.last_mut() {
            Some(last) if *last.last().unwrap() + 1 == p => last.push(p),
            _ => run_list.push(vec![p]),
        }
    }
    let heads: std::collections::BTreeSet<usize> = run_list.iter().map(|r| r[0]).collect();
    let mut run_of = BTreeMap::new();
    for (i, r) in run_list.iter().enumerate() {
        for &p in r {
            run_of.insert(p, i as u64);
        }
    }

    println!("{}", "=".repeat(74));
    println!("SETUP -- must reproduce #135 before anything is added");
    println!("{}", "=".repeat(74));
    println!(
        "positions {n}   covered {}   runs {}   non-head {}",
        covered.len(),
        run_list.len(),
        covered.len() - run_list.len()
    );
    println!(
        "excluded: inline-consts {}   varies {}",
        unpacked.len(),
        varies.len()
    );
    let ok = covered.len() == 431 && run_list.len() == 31;
    println!(
        "reproduces #135 (431 covered, 31 runs): {}",
        if ok { "YES" } else { "NO" }
    );
    assert!(
        ok,
        "does not reproduce #135; the analysis below is untrustworthy"
    );

    // §11.3p's 393: covered, non-head, carrying a classical barrier.
    let candidates: Vec<usize> = covered
        .iter()
        .copied()
        .filter(|p| !heads.contains(p) && base[*p].barrier)
        .collect();
    println!();
    println!("{}", "=".repeat(74));
    println!("CANDIDATE SUPPRESSION SITES -- §11.3p's 393");
    println!("{}", "=".repeat(74));
    println!(
        "covered, non-head, carrying a classical barrier: {}",
        candidates.len()
    );
    assert_eq!(candidates.len(), 393, "expected §11.3p's 393");
    println!("reproduces §11.3p's 393: YES");

    // The four arms `edge-cover.py` runs, as two switches: `icb_on` drops
    // primitive 3, `seam_on` drops primitive 2 by collapsing every position
    // into one encoder session.
    let run = |label: &str,
               suppressed: &std::collections::BTreeSet<usize>,
               icb_on: bool,
               seam_on: bool| {
        let runs = Runs {
            run_of: if icb_on {
                run_of.clone()
            } else {
                BTreeMap::new()
            },
            heads: heads.clone(),
            icb_on,
        };
        let mut ds = to_audit(base, &|i| suppressed.contains(&i), &runs);
        if !seam_on {
            // Collapse every position into one session, so the seam clause
            // cannot fire. `edge-cover.py`'s `seam_on=False` arm.
            for d in ds.iter_mut() {
                d.encoder = 0;
            }
        }
        let rep = cover(label, &ds);
        (rep.edges, rep.uncovered.len())
    };

    let none = std::collections::BTreeSet::new();
    let a: std::collections::BTreeSet<usize> = candidates.iter().copied().collect();
    let head_barriers: std::collections::BTreeSet<usize> =
        heads.iter().copied().filter(|p| base[*p].barrier).collect();
    let all_barriers: std::collections::BTreeSet<usize> =
        (0..n).filter(|p| base[*p].barrier).collect();
    let a_and_heads: std::collections::BTreeSet<usize> = a.union(&head_barriers).copied().collect();

    let (edges, _) = run("baseline", &none, true, true);
    println!();
    println!("{}", "=".repeat(74));
    println!("DEPENDENCY EDGES -- candle's three-hazard test, keyed (buffer, offset)");
    println!("{}", "=".repeat(74));
    println!("edges: {edges}");
    assert_eq!(edges, 85595, "edge count disagrees with #144's 85595");

    println!();
    println!("{}", "=".repeat(74));
    println!("RESULT, and the non-vacuity controls beside it (§15.1 #1)");
    println!("{}", "=".repeat(74));
    let rows: Vec<(&str, usize, usize)> = vec![
        (
            "baseline: suppress nothing",
            run("b", &none, true, true).1,
            0,
        ),
        (
            "PROPOSAL: suppress all 393 covered non-head",
            run("p", &a, true, true).1,
            0,
        ),
        (
            "MUTATION: also suppress the 30 run heads",
            run("m1", &a_and_heads, true, true).1,
            208,
        ),
        (
            "MUTATION: suppress all 505 barriers",
            run("m2", &all_barriers, true, true).1,
            5535,
        ),
        (
            "MUTATION: 393, but ICB emits no setBarrier",
            run("m3", &a, false, true).1,
            1960,
        ),
        (
            "MUTATION: 393, but seams do not order",
            run("m4", &a, true, false).1,
            11,
        ),
    ];
    let mut all_ok = true;
    for (label, got, want) in &rows {
        let mark = if got == want { "OK" } else { "MISMATCH" };
        if got != want {
            all_ok = false;
        }
        println!("  {got:>7}  edges unordered   {label:<44} (#144: {want}) {mark}");
    }
    println!();
    println!(
        "agrees with measurements/issue-144-raw/edge-cover.txt on all six rows: {}",
        if all_ok { "YES" } else { "NO" }
    );
    assert!(all_ok, "the Rust cover test disagrees with #144's model");
}
