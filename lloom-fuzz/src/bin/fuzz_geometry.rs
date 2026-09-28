//! The shape campaign over the tiny Qwen4-exp model (`lloom_fuzz::geometry`,
//! `docs/porting-models.md` §21.8): geometries searched by the shape probes
//! and the dispatch census, judged by CPU-vs-Metal parity.

use clap::Parser;
use lloom_fuzz::geometry::{self, Geometry, Outcome, Workload};
use lloom_tensor::Device;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[command(flatten)]
    cli: lloom_fuzz::campaign::Cli,
    /// The largest CPU-vs-Metal logit difference accepted, relative to the
    /// CPU logits' largest magnitude.
    #[arg(long, default_value_t = 2e-2)]
    tol: f32,
    /// Take the fused HC prologue (`QWEN4EXP_HC_MIX`), off by default in the
    /// engine; the campaign reaches its shape checks only with it on.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    hc_mix: bool,
    /// The seed's dense stems as `TensorF16` (the real checkpoint's kind);
    /// the tiny fixtures default to f32.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    seed_f16_stems: bool,
    /// The seed's prompt length.
    #[arg(long, default_value_t = 1)]
    seed_prompt: usize,
    /// Run one input (or every file in a directory), print the geometry, the
    /// outcome, the per-step comparison, the probes and the census; exit.
    #[arg(long)]
    show: Option<PathBuf>,
    /// Run every input in this directory (a corpus) and print what the set
    /// reached: each probe `(site, condition, outcome)` with the smallest
    /// geometry that reached it, the census labels, the outcome classes.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Append one line per execution to this file, tab-separated: the
    /// outcome, the largest relative logit difference, the CPU logits'
    /// scale, the geometry, the message, the Unix time in ms, the input hex.
    #[arg(long)]
    outcomes: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // The census and the tuning are read once, on first use.
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let mut args = Args::parse();
    let mut t = lloom::tuning::Tuning::default();
    t.from_env().map_err(anyhow::Error::msg)?;
    t.qwen4exp.hc_mix = args.hc_mix;
    if lloom::tuning::install(t).is_err() {
        anyhow::bail!("the tuning was read before it was installed");
    }

    if args.cli.seeds.is_none() {
        let dir = args.cli.corpus.with_file_name("geometry-seeds");
        std::fs::create_dir_all(&dir)?;
        let seed = Geometry::template(Workload {
            prompt: args.seed_prompt.clamp(1, geometry::MAX_PROMPT),
            decode: 1,
            f16_stems: args.seed_f16_stems,
        });
        std::fs::write(dir.join("template"), seed.encode())?;
        args.cli.seeds = Some(dir);
    }

    let tol = args.tol;
    let devs: std::cell::OnceCell<(Device, Device)> = std::cell::OnceCell::new();
    let run = move |bytes: &[u8]| -> (Geometry, Outcome) {
        let (cpu, gpu) =
            devs.get_or_init(|| (Device::Cpu, Device::new_metal(0).expect("a Metal device")));
        let g = Geometry::decode(bytes);
        let o = geometry::execute(&g, cpu, gpu, tol);
        (g, o)
    };

    if let Some(path) = &args.show {
        let mut files = vec![path.clone()];
        if path.is_dir() {
            files = std::fs::read_dir(path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect();
            files.sort();
        }
        for f in files {
            let (g, o) = run(&std::fs::read(&f)?);
            println!("== {}", f.display());
            println!("geometry: {}", g.describe());
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
        let mut run = run;
        return report(dir, &mut run);
    }

    lloom_fuzz::campaign::set_structured_mutator(geometry::mutate);
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
    lloom_fuzz::campaign::run(&args.cli, "geometry", move |bytes| {
        let (g, o) = run(bytes);
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
                g.describe()
            );
        }
        match o {
            Outcome::Defect(m) => Err(format!("{m}\ngeometry: {}", g.describe())),
            _ => Ok(()),
        }
    })
}

/// The corpus report (`--report`).
fn report(
    dir: &std::path::Path,
    run: &mut impl FnMut(&[u8]) -> (Geometry, Outcome),
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;
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
    let mut probes: BTreeMap<(String, String, bool), (usize, usize, String)> = BTreeMap::new();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut classes: BTreeMap<&'static str, usize> = BTreeMap::new();
    for f in &files {
        let (g, o) = run(&std::fs::read(f)?);
        *classes.entry(o.class()).or_default() += 1;
        let size: usize = g.axes.iter().sum::<usize>() + g.work.prompt + g.work.decode;
        let (ps, ls) = geometry::last_reach();
        for p in ps {
            let e = probes.entry(p).or_insert((0, usize::MAX, String::new()));
            e.0 += 1;
            if size < e.1 {
                e.1 = size;
                e.2 = g.describe();
            }
        }
        for l in ls {
            *labels.entry(l).or_default() += 1;
        }
    }
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
