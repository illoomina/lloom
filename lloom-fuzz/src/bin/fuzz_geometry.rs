//! The shape campaign over a family's tiny models (`lloom_fuzz::geometry`,
//! `docs/porting-models.md` §21.8): geometries searched by the shape probes
//! and the dispatch census, judged by CPU-vs-Metal parity.
//!
//! `--family qwen4exp | lfm2` picks the family; `--manifest` turns a corpus
//! into the family's manifest (the smallest geometry reaching each point).

use clap::Parser;
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{Geometry, TinyFamily, ValueLaw, Weights, Workload, MAX_PROMPT};
use lloom_fuzz::geometry::{self, Outcome};
use lloom_tensor::Device;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    cli: lloom_fuzz::campaign::Cli,
    /// `qwen4exp`, `lfm2`, `qwen3_5` or `qwen3_5_moe`.
    #[arg(long, default_value = "qwen4exp")]
    family: String,
    /// Numerical contract; recorded in the manifest because it selects routes.
    #[arg(long, default_value = "deterministic", value_parser = ["fast", "deterministic"])]
    contract: String,
    /// The value law of the weights (`hash[:gain]`, `uniform[:half]`,
    /// `ramp-sin`, `ramp-mod17`).
    #[arg(long, default_value = "hash:1")]
    law: String,
    /// The weights' seed.
    #[arg(long, default_value_t = Weights::default().seed)]
    weight_seed: u64,
    /// The largest CPU-vs-Metal logit difference accepted, relative to the
    /// CPU logits' largest magnitude. The reference computes at f32; the
    /// Metal arms that cast activations to f16 (the f16-weight GEMM's
    /// classical route) sit up to a few percent from it at hidden 512, so
    /// the bound is 5e-2 rather than the spike's 2e-2 — a gross class (a
    /// non-finite logit, a wrong route) is far outside either.
    #[arg(long, default_value_t = 5e-2)]
    tol: f32,
    /// Take the fused HC prologue (`QWEN4EXP_HC_MIX`), off by default in the
    /// engine; the campaign reaches its shape checks only with it on.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    hc_mix: bool,
    /// The seed's dense kind, an index into the family's kinds.
    #[arg(long, default_value_t = 0)]
    seed_dense: usize,
    /// The seed's prompt length.
    #[arg(long, default_value_t = 1)]
    seed_prompt: usize,
    /// Build the Metal side as the family's CPU reference is built (f32
    /// stems): a discriminator — a deviation that goes away was the served
    /// arm's precision, one that stays is not.
    #[arg(long)]
    metal_reference: bool,
    /// Run one input (or every file in a directory), print the geometry, the
    /// outcome, the per-step comparison, the probes and the census; exit.
    #[arg(long)]
    show: Option<PathBuf>,
    /// Run every input in this directory (a corpus) and print what the set
    /// reached: each probe `(site, condition, outcome)` with the smallest
    /// geometry that reached it, the census labels, the outcome classes.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Run every input in this directory and write the family's manifest
    /// (JSON) to `--manifest-out`: the smallest agreeing geometry for each
    /// probe outcome and census label the corpus reached.
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long, default_value = "manifest.json")]
    manifest_out: PathBuf,
    /// Append one line per execution to this file, tab-separated: the
    /// outcome, the largest relative logit difference, the CPU logits'
    /// scale, the geometry, the message, the Unix time in ms, the input hex.
    #[arg(long)]
    outcomes: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // The census and the tuning are read once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let args = Args::parse();
    geometry::FAST_CONTRACT.store(
        args.contract == "fast",
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut t = lloom::tuning::Tuning::default();
    t.from_env().map_err(anyhow::Error::msg)?;
    t.qwen4exp.hc_mix = args.hc_mix;
    if lloom::tuning::install(t).is_err() {
        anyhow::bail!("the tuning was read before it was installed");
    }
    match args.family.as_str() {
        "qwen4exp" => run::<Qwen4ExpTiny>(args),
        "lfm2" => run::<Lfm2Tiny>(args),
        "qwen3_5" => run::<Qwen3_5Tiny>(args),
        "qwen3_5_moe" => run::<Qwen3_5MoeTiny>(args),
        other => anyhow::bail!("--family {other}: qwen4exp | lfm2 | qwen3_5 | qwen3_5_moe"),
    }
}

fn inputs(dir: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && !p
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        })
        .collect();
    files.sort();
    Ok(files)
}

fn run<F: TinyFamily + 'static>(mut args: Args) -> anyhow::Result<()> {
    let weights = Weights {
        law: ValueLaw::parse(&args.law).map_err(anyhow::Error::msg)?,
        seed: args.weight_seed,
        ..Weights::default()
    };
    let tol = args.tol;
    geometry::METAL_REFERENCE.store(args.metal_reference, std::sync::atomic::Ordering::Relaxed);
    let devs: std::cell::OnceCell<(Device, Device)> = std::cell::OnceCell::new();
    let exec = move |bytes: &[u8]| -> (Geometry, Outcome) {
        let (cpu, gpu) =
            devs.get_or_init(|| (Device::Cpu, Device::new_metal(0).expect("a Metal device")));
        let g = Geometry::decode::<F>(bytes);
        let o = geometry::execute::<F>(&g, weights, cpu, gpu, tol);
        (g, o)
    };

    if let Some(path) = &args.show {
        let files = if path.is_dir() {
            inputs(path)?
        } else {
            vec![path.clone()]
        };
        for f in files {
            let (g, o) = exec(&std::fs::read(&f)?);
            println!("== {}", f.display());
            println!(
                "family: {} law: {} weight_seed: {}",
                F::NAME,
                args.law,
                args.weight_seed
            );
            println!("geometry: {}", g.describe::<F>());
            println!("outcome: {o:?}");
            for (i, st) in geometry::last_steps().iter().enumerate() {
                println!(
                    "step {i}: maxdiff {:e} scale {:e} rel {:e} argmax cpu {} metal {}",
                    st.maxdiff,
                    st.scale,
                    st.maxdiff / st.scale.max(1e-30),
                    st.argmax_cpu,
                    st.argmax_metal
                );
            }
            let (probes, labels) = geometry::last_reach();
            for (site, cond, ok) in probes {
                println!("probe: {site} | {cond} | {ok}");
            }
            println!("census: {}", labels.join(" "));
        }
        println!("dictionary: {:?}", geometry::dictionary());
        return Ok(());
    }

    if let Some(dir) = &args.report {
        let mut exec = exec;
        return report::<F>(dir, weights, &mut exec);
    }

    if let Some(dir) = &args.manifest {
        let mut exec = exec;
        return manifest::<F>(dir, &args, weights, &mut exec);
    }

    if args.cli.seeds.is_none() {
        let dir = args
            .cli
            .corpus
            .with_file_name(format!("geometry-seeds-{}", F::NAME));
        std::fs::create_dir_all(&dir)?;
        let mut seed = Geometry::template::<F>(Workload {
            prompt: args.seed_prompt.clamp(1, MAX_PROMPT),
            decode: 1,
        });
        seed.dense = args.seed_dense % F::DENSE_KINDS.len();
        std::fs::write(dir.join("template"), seed.encode())?;
        args.cli.seeds = Some(dir);
    }

    lloom_fuzz::campaign::set_structured_mutator(geometry::mutate::<F>);
    // The model's kernels are compiled at first use, in the process that
    // executes; `--single` keeps that process a re-executed one rather than
    // a forked child (`campaign::Cli::single`). `LLOOM_FUZZ_GEOMETRY_LAUNCHER=1`
    // runs the LLMP launcher instead.
    args.cli.single = std::env::var_os("LLOOM_FUZZ_GEOMETRY_LAUNCHER").is_none();
    let mut log = match &args.outcomes {
        Some(p) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)?,
        ),
        None => None,
    };
    let name = format!("geometry-{}", F::NAME);
    lloom_fuzz::campaign::run_with_config(
        &args.cli,
        &name,
        &serde_json::json!({
            "family": F::NAME, "law": args.law, "weight_seed": args.weight_seed,
            "tol_bits": args.tol.to_bits(), "hc_mix": args.hc_mix,
            "metal_reference": args.metal_reference
        }),
        move |bytes| {
            let (g, o) = exec(bytes);
            if let Some(f) = log.as_mut() {
                use std::io::Write;
                let (d, s) = match &o {
                    Outcome::Agree { maxdiff, scale } => (*maxdiff, *scale),
                    _ => (f32::NAN, f32::NAN),
                };
                let what = match &o {
                    Outcome::CpuRefused(m) | Outcome::MetalRefused(m) | Outcome::Defect(m) => {
                        m.replace(['\n', '\t'], " ")
                    }
                    _ => String::new(),
                };
                let ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |t| t.as_millis());
                let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                let _ = writeln!(
                    f,
                    "{}\t{d:e}\t{s:e}\t{}\t{what}\t{ms}\t{hex}",
                    o.class(),
                    g.describe::<F>()
                );
            }
            match o {
                Outcome::Defect(m) => Err(format!("{m}\ngeometry: {}", g.describe::<F>())),
                _ => Ok(()),
            }
        },
    )
}

/// A geometry's size, for "smallest": its weight elements, then its tokens.
fn size<F: TinyFamily>(g: &Geometry, weights: Weights) -> usize {
    g.checkpoint::<F>(weights)
        .and_then(|c| c.schema().ok())
        .map_or(usize::MAX, |s| s.elements())
        .saturating_add(g.work.prompt + g.work.decode)
}

/// The corpus report (`--report`).
fn report<F: TinyFamily>(
    dir: &std::path::Path,
    weights: Weights,
    exec: &mut impl FnMut(&[u8]) -> (Geometry, Outcome),
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;
    let files = inputs(dir)?;
    let mut probes: BTreeMap<(String, String, bool), (usize, usize, String)> = BTreeMap::new();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut classes: BTreeMap<&'static str, usize> = BTreeMap::new();
    for f in &files {
        let (g, o) = exec(&std::fs::read(f)?);
        *classes.entry(o.class()).or_default() += 1;
        let size = size::<F>(&g, weights);
        let (ps, ls) = geometry::last_reach();
        for p in ps {
            let e = probes.entry(p).or_insert((0, usize::MAX, String::new()));
            e.0 += 1;
            if size < e.1 {
                e.1 = size;
                e.2 = g.describe::<F>();
            }
        }
        for l in ls {
            *labels.entry(l).or_default() += 1;
        }
    }
    println!("family: {}", F::NAME);
    println!("inputs: {}", files.len());
    println!("outcomes: {classes:?}");
    println!("probes reached ({}):", probes.len());
    for ((site, cond, ok), (n, _, g)) in &probes {
        println!("  {site} | {cond} | {ok} | inputs {n} | smallest: {g}");
    }
    println!("census labels ({}):", labels.len());
    for (l, n) in &labels {
        println!("  {l} {n}");
    }
    Ok(())
}

/// The manifest (`--manifest`): every probe outcome and census label the
/// corpus reached on an AGREEING execution, each assigned the smallest
/// geometry that reached it; one entry per distinct geometry, with the
/// points it carries. Points reached only by a refusal or a defect are
/// listed apart — a drift test cannot replay them as passing.
fn manifest<F: TinyFamily>(
    dir: &std::path::Path,
    args: &Args,
    weights: Weights,
    exec: &mut impl FnMut(&[u8]) -> (Geometry, Outcome),
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;
    let files = inputs(dir)?;
    // point -> (size, geometry bytes)
    let mut best: BTreeMap<String, (usize, Vec<u8>)> = BTreeMap::new();
    let mut elsewhere: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut runs: BTreeMap<Vec<u8>, (String, Vec<String>)> = BTreeMap::new();
    for f in &files {
        let (g, o) = exec(&std::fs::read(f)?);
        let (ps, ls) = geometry::last_reach();
        let points: Vec<String> = ps
            .iter()
            .map(|(s, c, ok)| format!("probe|{s}|{c}|{ok}"))
            .chain(ls.iter().map(|l| format!("census|{l}")))
            .collect();
        if !matches!(o, Outcome::Agree { .. }) {
            eprintln!(
                "MANIFEST_NON_AGREE input={} geometry={} outcome={o:?} steps={:?}",
                f.display(),
                g.describe::<F>(),
                geometry::last_steps()
            );
            for p in points {
                elsewhere.entry(p).or_insert(o.class());
            }
            continue;
        }
        let bytes = g.encode();
        let sz = size::<F>(&g, weights);
        for p in &points {
            let e = best.entry(p.clone()).or_insert((usize::MAX, Vec::new()));
            if sz < e.0 || (sz == e.0 && bytes < e.1) {
                *e = (sz, bytes.clone());
            }
        }
        runs.insert(bytes, (g.describe::<F>(), points));
    }
    // The gap list: points no agreeing execution reached.
    elsewhere.retain(|p, _| !best.contains_key(p));
    let mut entries: BTreeMap<Vec<u8>, Vec<String>> = BTreeMap::new();
    for (p, (_, g)) in &best {
        entries.entry(g.clone()).or_default().push(p.clone());
    }
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let json = serde_json::json!({
        "family": F::NAME,
        "contract": args.contract,
        "commit": commit,
        "law": args.law,
        "weight_seed": args.weight_seed,
        "tol": args.tol,
        "hc_mix": args.hc_mix,
        "gdn_step_reg": lloom_tensor::tuning::tuning().gdn.step_reg,
        "corpus_inputs": files.len(),
        "entries": entries.iter().map(|(g, pts)| {
            let (describe, reached) = &runs[g];
            serde_json::json!({
                "geometry": g.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                "describe": describe,
                "smallest_for": pts,
                "reaches": reached,
            })
        }).collect::<Vec<_>>(),
        "reached_only_by_non_agreeing": elsewhere.iter().map(|(p, c)| format!("{p} ({c})")).collect::<Vec<_>>(),
    });
    std::fs::write(&args.manifest_out, serde_json::to_string_pretty(&json)?)?;
    println!(
        "{}: {} inputs, {} points on agreeing runs, {} entries, {} points only on refusals/defects -> {}",
        F::NAME,
        files.len(),
        best.len(),
        entries.len(),
        elsewhere.len(),
        args.manifest_out.display()
    );
    Ok(())
}
