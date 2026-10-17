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
use lloom_tensor::tuning::{self, Contract};
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
    #[arg(long, default_value = "deterministic", value_parser = Contract::parse)]
    contract: Contract,
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
    /// Publish one complete observation for a single-input manifest invocation.
    #[arg(long, requires = "manifest", conflicts_with_all = ["merge_observations", "show", "report"])]
    manifest_observations: Option<PathBuf>,
    /// Merge completed per-input observations against the exact --manifest corpus.
    /// This CPU-only arm never constructs a Metal device.
    #[arg(long, requires = "manifest", conflicts_with_all = ["show", "report"])]
    merge_observations: Option<PathBuf>,
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
    args.contract = setup_contract(args.contract)?;
    eprintln!("CONTRACT profile={}", args.contract.name());
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

fn setup_contract(requested: Contract) -> anyhow::Result<Contract> {
    let resolved = tuning::pin_contract(requested).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        tuning::contract() == resolved,
        "the process contract was already fixed before campaign setup"
    );
    geometry::FAST_CONTRACT.store(resolved.is_fast(), std::sync::atomic::Ordering::Relaxed);
    Ok(resolved)
}

fn inputs(dir: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    input_paths(std::fs::read_dir(dir)?.map(|e| e.map(|e| e.path())))
}

fn input_paths(
    entries: impl IntoIterator<Item = std::io::Result<PathBuf>>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in entries {
        let path = entry?;
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        // Follow valid file symlinks, but never turn a metadata error into omission.
        if std::fs::metadata(&path)?.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn run<F: TinyFamily + 'static>(mut args: Args) -> anyhow::Result<()> {
    let weights = Weights {
        law: ValueLaw::parse(&args.law).map_err(anyhow::Error::msg)?,
        seed: args.weight_seed,
        ..Weights::default()
    };
    if let Some(dir) = &args.merge_observations {
        return merge_observations::<F>(dir, &args, weights);
    }
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
            "metal_reference": args.metal_reference, "contract": args.contract.name()
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
        .and_then(|c| {
            if geometry::FAST_CONTRACT.load(std::sync::atomic::Ordering::Relaxed) {
                c.schema_for::<lloom::profile::Crown>().ok()
            } else {
                c.schema_for::<lloom::profile::CrownExact>().ok()
            }
        })
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
    manifest_reached::<F>(dir, args, weights, &mut |bytes| {
        let (g, o) = exec(bytes);
        (g, o, geometry::last_reach())
    })
}

fn manifest_reached<F: TinyFamily>(
    dir: &std::path::Path,
    args: &Args,
    weights: Weights,
    exec: &mut impl FnMut(&[u8]) -> (Geometry, Outcome, geometry::Reach),
) -> anyhow::Result<()> {
    let files = inputs(dir)?;
    anyhow::ensure!(
        args.manifest_observations.is_none() || files.len() == 1,
        "observations require exactly one input"
    );
    let mut observations = Vec::new();
    for f in &files {
        let raw = std::fs::read(f)?;
        let (g, o, (ps, ls)) = exec(&raw);
        let points = ps
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
        }
        observations.push(Observation {
            name: f
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| anyhow::anyhow!("non-UTF8 input name"))?
                .into(),
            raw,
            size: size::<F>(&g, weights),
            g,
            outcome: o,
            points,
            steps: geometry::last_steps()
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "maxdiff_bits": s.maxdiff.to_bits(), "scale_bits": s.scale.to_bits(),
                        "argmax_cpu": s.argmax_cpu, "argmax_metal": s.argmax_metal,
                    })
                })
                .collect(),
        });
    }
    if let Some(path) = &args.manifest_observations {
        publish_new(
            path,
            &serde_json::json!({
                "schema": "lloom-geometry-observation-v1", "complete": true, "input_count": 1,
                "identity": observation_identity::<F>(args)?, "result": observations[0].json::<F>(),
            }),
        )?;
    }
    write_manifest::<F>(&observations, args)
}

struct Observation {
    name: String,
    raw: Vec<u8>,
    g: Geometry,
    size: usize,
    outcome: Outcome,
    points: Vec<String>,
    steps: Vec<serde_json::Value>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn unhex(value: &serde_json::Value) -> anyhow::Result<Vec<u8>> {
    let s = value
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing hex string"))?;
    anyhow::ensure!(s.len() % 2 == 0 && s.is_ascii(), "invalid hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(Into::into))
        .collect()
}

fn observation_identity<F: TinyFamily>(args: &Args) -> anyhow::Result<serde_json::Value> {
    static BINARY: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    let binary = BINARY
        .get_or_init(|| {
            (|| -> anyhow::Result<String> {
                use sha2::Digest;
                use std::io::Read;
                let mut file = std::fs::File::open(std::env::current_exe()?)?;
                let mut hash = sha2::Sha256::new();
                let mut buf = [0; 16 * 1024];
                loop {
                    let n = file.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    hash.update(&buf[..n]);
                }
                Ok(format!("{:x}", hash.finalize()))
            })()
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("executable identity: {e}"))?;
    Ok(serde_json::json!({
        "executable_sha256": binary,
        "family": F::NAME, "contract": args.contract.name(), "law": args.law,
        "weight_seed": args.weight_seed, "tol_bits": args.tol.to_bits(),
        "hc_mix": args.hc_mix, "metal_reference": args.metal_reference,
        "gdn_step_reg": lloom_tensor::tuning::tuning().gdn.step_reg,
        "commit": current_commit(),
        "coverage": "explicit configuration only; environment/driver/hardware are separate admission receipts",
    }))
}

fn current_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

/// Same-filesystem hard-link publication is atomic and refuses an existing final path.
fn publish_new(path: &std::path::Path, json: &serde_json::Value) -> anyhow::Result<()> {
    publish_before_link(path, json, || Ok(()))
}

fn publish_before_link(
    path: &std::path::Path,
    json: &serde_json::Value,
    before_link: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("pending-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    let result = (|| -> anyhow::Result<()> {
        file.write_all(serde_json::to_string_pretty(json)?.as_bytes())?;
        file.sync_all()?;
        before_link()?;
        std::fs::hard_link(&tmp, path)?;
        Ok(())
    })();
    drop(file);
    // Only the temporary file created by this invocation is removed.
    let cleanup = std::fs::remove_file(&tmp);
    result?;
    cleanup?;
    Ok(())
}

impl Observation {
    fn json<F: TinyFamily>(&self) -> serde_json::Value {
        let (message, comparison) = match &self.outcome {
            Outcome::CpuRefused(m) | Outcome::MetalRefused(m) | Outcome::Defect(m) => {
                (m.as_str(), serde_json::Value::Null)
            }
            Outcome::Agree { maxdiff, scale } => (
                "",
                serde_json::json!({"maxdiff_bits": maxdiff.to_bits(), "scale_bits": scale.to_bits()}),
            ),
            Outcome::Skipped => ("", serde_json::Value::Null),
        };
        serde_json::json!({"name": self.name, "input_hex": hex(&self.raw), "input_sha256": digest(&self.raw),
            "geometry": hex(&self.g.encode()), "describe": self.g.describe::<F>(), "size": self.size,
            "outcome": self.outcome.class(), "message": message, "comparison": comparison,
            "points": self.points, "steps": self.steps})
    }
}

fn merge_observations<F: TinyFamily>(
    dir: &std::path::Path,
    args: &Args,
    weights: Weights,
) -> anyhow::Result<()> {
    let corpus = args
        .manifest
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing source corpus"))?;
    let sources = inputs(corpus)?;
    let mut names = std::collections::BTreeSet::new();
    let identity = observation_identity::<F>(args)?;
    let mut observations = Vec::new();
    for source in &sources {
        let name = source
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 source"))?;
        anyhow::ensure!(names.insert(name.to_owned()), "duplicate source name");
        let receipt = dir.join(name);
        let read = |name: &str| -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::from_slice(&std::fs::read(receipt.join(name))?)?)
        };
        let doc = read("observation.json")?;
        anyhow::ensure!(
            doc["schema"] == "lloom-geometry-observation-v1"
                && doc["complete"] == true
                && doc["input_count"] == 1,
            "incomplete or unsupported observation for {name}"
        );
        anyhow::ensure!(
            doc["identity"] == identity,
            "observation identity mismatch for {name}"
        );
        let r = &doc["result"];
        let raw = std::fs::read(source)?;
        anyhow::ensure!(
            r["name"] == name
                && unhex(&r["input_hex"])? == raw
                && r["input_sha256"] == digest(&raw),
            "source identity mismatch for {name}"
        );
        let g = Geometry::decode::<F>(&raw);
        let size = size::<F>(&g, weights);
        anyhow::ensure!(
            unhex(&r["geometry"])? == g.encode()
                && r["describe"] == g.describe::<F>()
                && r["size"].as_u64() == Some(size as u64),
            "geometry/size mismatch for {name}"
        );
        let message = r["message"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing outcome message"))?
            .to_owned();
        let outcome = match r["outcome"].as_str() {
            Some("agree") => {
                let bits = |key: &str| -> anyhow::Result<f32> {
                    Ok(f32::from_bits(u32::try_from(
                        r["comparison"][key]
                            .as_u64()
                            .ok_or_else(|| anyhow::anyhow!("invalid comparison"))?,
                    )?))
                };
                let maxdiff = bits("maxdiff_bits")?;
                let scale = bits("scale_bits")?;
                anyhow::ensure!(
                    maxdiff.is_finite()
                        && scale.is_finite()
                        && maxdiff >= 0.
                        && scale >= 0.
                        && maxdiff <= args.tol,
                    "invalid Agree comparison"
                );
                Outcome::Agree { maxdiff, scale }
            }
            Some("skipped") => Outcome::Skipped,
            Some("cpu-refused") => Outcome::CpuRefused(message),
            Some("metal-refused") => Outcome::MetalRefused(message),
            Some("defect") => anyhow::bail!("defect observation for {name}: {message}"),
            _ => anyhow::bail!("unknown outcome for {name}"),
        };
        let points: Vec<String> = r["points"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing points"))?
            .iter()
            .map(|p| {
                p.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("invalid point"))
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            points
                .iter()
                .all(|p| p.starts_with("probe|") || p.starts_with("census|")),
            "invalid point kind"
        );
        let exit = read("exit.json")?;
        let covered = matches!(outcome, Outcome::Agree { .. }) && !points.is_empty();
        anyhow::ensure!(
            exit["exit_code"].as_i64() == Some(if covered { 0 } else { 1 }),
            "unexpected process exit for {name}"
        );
        let partial = receipt.join("manifest.json");
        if covered {
            let manifest = read("manifest.json")?;
            anyhow::ensure!(
                manifest["corpus_inputs"] == 1
                    && manifest["entries"]
                        .as_array()
                        .is_some_and(|e| !e.is_empty()),
                "missing covered partial manifest"
            );
        } else {
            anyhow::ensure!(!partial.exists(), "noncoverage invocation wrote a manifest");
            let stderr = std::fs::read_to_string(receipt.join("stderr.txt"))?;
            let exact = format!("Error: {}: 1 inputs, {} agreeing runs, 0 manifest entries; refusing to overwrite {}", F::NAME,
                usize::from(matches!(outcome, Outcome::Agree {..})), partial.display());
            anyhow::ensure!(
                stderr.lines().any(|l| l == exact),
                "missing exact empty-guard failure for {name}"
            );
        }
        let observation = Observation {
            name: name.into(),
            raw,
            g,
            size,
            outcome,
            points,
            steps: r["steps"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("missing steps"))?
                .clone(),
        };
        if covered {
            anyhow::ensure!(
                read("manifest.json")?
                    == manifest_json::<F>(std::slice::from_ref(&observation), args)?,
                "covered partial manifest differs from observation for {name}"
            );
        }
        observations.push(observation);
    }
    let actual: std::collections::BTreeSet<String> = std::fs::read_dir(dir)?
        .map(|e| {
            let e = e?;
            anyhow::ensure!(e.file_type()?.is_dir(), "unexpected receipt entry");
            e.file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 receipt name"))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(actual == names, "extra or missing input receipts");
    write_manifest::<F>(&observations, args)
}

fn write_manifest<F: TinyFamily>(observations: &[Observation], args: &Args) -> anyhow::Result<()> {
    let json = manifest_json::<F>(observations, args)?;
    std::fs::write(&args.manifest_out, serde_json::to_string_pretty(&json)?)?;
    let points: usize = json["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["smallest_for"].as_array().unwrap().len())
        .sum();
    println!("{}: {} inputs, {} points on agreeing runs, {} entries, {} points only on refusals/defects -> {}",
        F::NAME, observations.len(), points, json["entries"].as_array().unwrap().len(),
        json["reached_only_by_non_agreeing"].as_array().unwrap().len(), args.manifest_out.display());
    Ok(())
}

fn manifest_json<F: TinyFamily>(
    observations: &[Observation],
    args: &Args,
) -> anyhow::Result<serde_json::Value> {
    use std::collections::BTreeMap;
    let mut best: BTreeMap<String, (usize, Vec<u8>)> = BTreeMap::new();
    let mut elsewhere: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut runs: BTreeMap<Vec<u8>, (String, Vec<String>)> = BTreeMap::new();
    for o in observations {
        if !matches!(o.outcome, Outcome::Agree { .. }) {
            for p in &o.points {
                elsewhere.entry(p.clone()).or_insert(o.outcome.class());
            }
            continue;
        }
        let bytes = o.g.encode();
        for p in &o.points {
            let e = best.entry(p.clone()).or_insert((usize::MAX, Vec::new()));
            if o.size < e.0 || (o.size == e.0 && bytes < e.1) {
                *e = (o.size, bytes.clone());
            }
        }
        runs.insert(bytes, (o.g.describe::<F>(), o.points.clone()));
    }
    // The gap list: points no agreeing execution reached.
    elsewhere.retain(|p, _| !best.contains_key(p));
    let mut entries: BTreeMap<Vec<u8>, Vec<String>> = BTreeMap::new();
    for (p, (_, g)) in &best {
        entries.entry(g.clone()).or_default().push(p.clone());
    }
    anyhow::ensure!(
        !entries.is_empty(),
        "{}: {} inputs, {} agreeing runs, 0 manifest entries; refusing to overwrite {}",
        F::NAME,
        observations.len(),
        runs.len(),
        args.manifest_out.display()
    );
    let commit = current_commit();
    let json = serde_json::json!({
        "family": F::NAME,
        "contract": args.contract.name(),
        "commit": commit,
        "law": args.law,
        "weight_seed": args.weight_seed,
        "tol": args.tol,
        "hc_mix": args.hc_mix,
        "gdn_step_reg": lloom_tensor::tuning::tuning().gdn.step_reg,
        "corpus_inputs": observations.len(),
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
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom::profile::{Crown, CrownExact};
    use lloom::testing::tiny::TinyCheckpoint;

    fn schema_at(contract: Contract) {
        std::env::remove_var("LLOOM_PROFILE");
        assert_eq!(setup_contract(contract).unwrap(), contract);
        assert_eq!(tuning::contract(), contract);
        assert_eq!(
            geometry::FAST_CONTRACT.load(std::sync::atomic::Ordering::Relaxed),
            contract.is_fast()
        );
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        let schema = match contract {
            Contract::Fast => ckpt.schema_for::<Crown>(),
            Contract::Deterministic => ckpt.schema_for::<CrownExact>(),
        }
        .expect("schema must use the requested regime");
        assert!(!schema.tensors.is_empty());
        match contract {
            Contract::Fast => {
                assert_eq!(schema, ckpt.schema_for::<Crown>().unwrap());
                assert!(ckpt.schema_for::<CrownExact>().is_err());
                assert!(ckpt.schema().is_err());
            }
            Contract::Deterministic => assert_eq!(schema, ckpt.schema().unwrap()),
        }
        match contract {
            Contract::Fast => {
                assert!(Lfm2Tiny::reference::<CrownExact>(&ckpt, &Device::Cpu).is_err());
                let model = Lfm2Tiny::reference::<Crown>(&ckpt, &Device::Cpu).unwrap();
                let logits = lloom::testing::tiny::run(
                    &model,
                    &ckpt.cfg,
                    ckpt.cfg.vocab_size,
                    &Device::Cpu,
                    &Workload {
                        prompt: 1,
                        decode: 1,
                    },
                )
                .unwrap();
                assert_eq!(logits.len(), 2);
                assert!(logits.iter().flatten().all(|v| v.is_finite()));
            }
            Contract::Deterministic => {
                assert!(Lfm2Tiny::reference::<Crown>(&ckpt, &Device::Cpu).is_err());
                let model = Lfm2Tiny::reference::<CrownExact>(&ckpt, &Device::Cpu).unwrap();
                let logits = lloom::testing::tiny::run(
                    &model,
                    &ckpt.cfg,
                    ckpt.cfg.vocab_size,
                    &Device::Cpu,
                    &Workload {
                        prompt: 1,
                        decode: 1,
                    },
                )
                .unwrap();
                assert_eq!(logits.len(), 2);
                assert!(logits.iter().flatten().all(|v| v.is_finite()));
            }
        }
        use sha2::Digest;
        let digest = sha2::Sha256::digest(format!("{schema:?}").as_bytes());
        println!("schema_sha256={digest:x}");
        println!(
            "{}: {} tensors, {} elements; schema and CPU reference built",
            contract.name(),
            schema.tensors.len(),
            schema.elements()
        );
    }

    #[test]
    fn fast_setup_and_schema_match() {
        schema_at(Contract::Fast);
    }

    #[test]
    fn deterministic_setup_and_schema_match() {
        schema_at(Contract::Deterministic);
    }

    #[test]
    fn environment_contract_selects_the_same_arm() {
        std::env::set_var("LLOOM_PROFILE", " FAST ");
        assert_eq!(
            setup_contract(Contract::Deterministic).unwrap(),
            Contract::Fast
        );
        assert!(geometry::FAST_CONTRACT.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(tuning::contract(), Contract::Fast);
        TinyCheckpoint::<Lfm2Tiny>::template()
            .schema_for::<Crown>()
            .unwrap();
    }

    #[test]
    fn unbound_schema_then_exact_keeps_legacy_ordering() {
        std::env::remove_var("LLOOM_PROFILE");
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        assert_eq!(ckpt.schema().unwrap().tensors.len(), 29);
        ckpt.model::<CrownExact>(&Device::Cpu).unwrap();
        assert!(ckpt.model::<Crown>(&Device::Cpu).is_err());
    }

    #[test]
    fn unbound_write_then_exact_load_keeps_legacy_ordering() {
        std::env::remove_var("LLOOM_PROFILE");
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        let dir =
            std::env::temp_dir().join(format!("geometry-legacy-write-{}", std::process::id()));
        ckpt.write(&dir).unwrap();
        Lfm2Tiny::load::<CrownExact>(&dir, &Device::Cpu).unwrap();
        assert!(Lfm2Tiny::load::<Crown>(&dir, &Device::Cpu).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_fast_pin_before_first_contract_read_records_fast() {
        std::env::remove_var("LLOOM_PROFILE");
        tuning::pin_contract(Contract::Fast).unwrap();
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        assert_eq!(ckpt.schema_for::<Crown>().unwrap().tensors.len(), 29);
        ckpt.model::<Crown>(&Device::Cpu).unwrap();
        assert!(ckpt.model::<CrownExact>(&Device::Cpu).is_err());
    }

    #[test]
    fn explicit_exact_pin_before_first_contract_read_records_exact() {
        std::env::remove_var("LLOOM_PROFILE");
        tuning::pin_contract(Contract::Deterministic).unwrap();
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        ckpt.schema_for::<CrownExact>().unwrap();
        ckpt.model::<CrownExact>(&Device::Cpu).unwrap();
        assert!(ckpt.model::<Crown>(&Device::Cpu).is_err());
    }

    #[test]
    fn initialized_tuning_still_allows_explicit_exact_schema() {
        std::env::remove_var("LLOOM_PROFILE");
        let _ = tuning::tuning();
        let ckpt = TinyCheckpoint::<Lfm2Tiny>::template();
        ckpt.schema_for::<CrownExact>().unwrap();
        ckpt.model::<CrownExact>(&Device::Cpu).unwrap();
        assert!(ckpt.model::<Crown>(&Device::Cpu).is_err());
    }

    #[test]
    fn covered_agreeing_input_writes_a_nonempty_manifest() {
        std::env::remove_var("LLOOM_PROFILE");
        setup_contract(Contract::Fast).unwrap();
        let dir = std::env::temp_dir().join(format!("geometry-covered-{}", std::process::id()));
        let corpus = dir.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        let g = Geometry {
            axes: vec![64, 128, 256, 3, 2, 1, 3],
            work: Workload {
                prompt: 1,
                decode: 1,
            },
            dense: 0,
            experts: 0,
            arm: 0,
        };
        let bytes = g.encode();
        std::fs::write(corpus.join("input"), &bytes).unwrap();
        let mut args = Args::try_parse_from(["fuzz_geometry", "--contract", "fast"]).unwrap();
        args.manifest_out = dir.join("manifest.json");
        args.manifest_observations = Some(dir.join("observation.json"));
        std::fs::write(&args.manifest_out, b"old manifest").unwrap();
        manifest_reached::<Lfm2Tiny>(&corpus, &args, Weights::default(), &mut |input| {
            assert_eq!(input, bytes);
            (
                g.clone(),
                Outcome::Agree {
                    maxdiff: 0.,
                    scale: 1.,
                },
                (
                    vec![("fixture".into(), "covered".into(), true)],
                    vec!["fixture-kernel".into()],
                ),
            )
        })
        .unwrap();
        let observation: serde_json::Value = serde_json::from_slice(
            &std::fs::read(args.manifest_observations.as_ref().unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(observation["complete"], true);
        assert_eq!(observation["result"]["outcome"], "agree");
        assert_eq!(
            observation["result"]["geometry"],
            "40008000000103000200010003000101000000"
        );
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&args.manifest_out).unwrap()).unwrap();
        assert_eq!(m["contract"], "fast");
        assert_eq!(m["corpus_inputs"], 1);
        let entries = m["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0]["geometry"],
            "40008000000103000200010003000101000000"
        );
        assert_eq!(
            entries[0]["smallest_for"],
            serde_json::json!(["census|fixture-kernel", "probe|fixture|covered|true"])
        );
        assert_eq!(
            entries[0]["reaches"],
            serde_json::json!(["probe|fixture|covered|true", "census|fixture-kernel"])
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn observations_merge_exactly_and_refuse_incomplete_receipts() {
        std::env::remove_var("LLOOM_PROFILE");
        setup_contract(Contract::Fast).unwrap();
        let root =
            std::env::temp_dir().join(format!("geometry-observation-{}", std::process::id()));
        let source = root.join("source");
        let receipts = root.join("receipts");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&receipts).unwrap();
        let mut args = Args::try_parse_from(["fuzz_geometry", "--contract", "fast"]).unwrap();
        args.manifest_out = root.join("direct.json");
        let geometry = |vocab| Geometry {
            axes: vec![vocab, 128, 256, 3, 2, 1, 3],
            work: Workload {
                prompt: 1,
                decode: 1,
            },
            dense: 0,
            experts: 0,
            arm: 0,
        };
        let small = geometry(64);
        let large = geometry(128);
        // Literal source order: refusal class is selected by name, not receipt creation order.
        let fixtures = [
            (
                "a",
                large.clone(),
                Outcome::CpuRefused("first".into()),
                vec!["census|gap", "census|shared"],
            ),
            (
                "b",
                large.clone(),
                Outcome::MetalRefused("second".into()),
                vec!["census|gap"],
            ),
            (
                "c",
                large.clone(),
                Outcome::Agree {
                    maxdiff: 0.,
                    scale: 1.,
                },
                vec!["census|shared", "census|large-only"],
            ),
            (
                "d",
                small.clone(),
                Outcome::Agree {
                    maxdiff: 0.,
                    scale: 1.,
                },
                vec!["census|shared", "census|small-only"],
            ),
        ];
        let mut observations = Vec::new();
        for (name, g, outcome, points) in fixtures.iter().rev() {
            let raw = g.encode();
            std::fs::write(source.join(name), &raw).unwrap();
            let o = Observation {
                name: (*name).into(),
                raw,
                g: g.clone(),
                size: size::<Lfm2Tiny>(g, Weights::default()),
                outcome: outcome.clone(),
                points: points.iter().map(|s| (*s).into()).collect(),
                steps: vec![],
            };
            let receipt = receipts.join(name);
            std::fs::create_dir(&receipt).unwrap();
            args.manifest_out = receipt.join("manifest.json");
            publish_new(&receipt.join("observation.json"), &serde_json::json!({
                "schema":"lloom-geometry-observation-v1","complete":true,"input_count":1,
                "identity":observation_identity::<Lfm2Tiny>(&args).unwrap(),"result":o.json::<Lfm2Tiny>()
            })).unwrap();
            let covered = matches!(outcome, Outcome::Agree { .. });
            if covered {
                write_manifest::<Lfm2Tiny>(std::slice::from_ref(&o), &args).unwrap();
            }
            std::fs::write(
                receipt.join("exit.json"),
                format!("{{\"exit_code\":{}}}", if covered { 0 } else { 1 }),
            )
            .unwrap();
            std::fs::write(receipt.join("stderr.txt"),format!("Error: lfm2: 1 inputs, 0 agreeing runs, 0 manifest entries; refusing to overwrite {}\n",args.manifest_out.display())).unwrap();
            observations.push(o);
        }
        observations.sort_by(|a, b| a.name.cmp(&b.name));
        args.manifest_out = root.join("direct.json");
        write_manifest::<Lfm2Tiny>(&observations, &args).unwrap();
        let direct = std::fs::read(&args.manifest_out).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&direct).unwrap();
        assert_eq!(json["corpus_inputs"], 4);
        assert_eq!(
            json["reached_only_by_non_agreeing"],
            serde_json::json!(["census|gap (cpu-refused)"])
        );
        let entries = json["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0]["geometry"],
            "40008000000103000200010003000101000000"
        );
        assert_eq!(
            entries[0]["smallest_for"],
            serde_json::json!(["census|shared", "census|small-only"])
        );
        args.manifest = Some(source.clone());
        args.manifest_out = root.join("merged.json");
        merge_observations::<Lfm2Tiny>(&receipts, &args, Weights::default()).unwrap();
        assert_eq!(std::fs::read(&args.manifest_out).unwrap(), direct);
        let reject = |args: &Args| {
            std::fs::write(&args.manifest_out, b"sentinel").unwrap();
            assert!(merge_observations::<Lfm2Tiny>(&receipts, args, Weights::default()).is_err());
            assert_eq!(std::fs::read(&args.manifest_out).unwrap(), b"sentinel");
        };
        let path = receipts.join("a/observation.json");
        let original = std::fs::read(&path).unwrap();
        for field in [
            "contract",
            "tol_bits",
            "law",
            "weight_seed",
            "hc_mix",
            "metal_reference",
            "gdn_step_reg",
            "executable_sha256",
        ] {
            let mut bad: serde_json::Value = serde_json::from_slice(&original).unwrap();
            bad["identity"][field] = serde_json::json!("wrong");
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            reject(&args);
        }
        for field in [
            "input_hex",
            "input_sha256",
            "geometry",
            "size",
            "name",
            "outcome",
        ] {
            let mut bad: serde_json::Value = serde_json::from_slice(&original).unwrap();
            bad["result"][field] = serde_json::json!("wrong");
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            reject(&args);
        }
        std::fs::write(&path, b"{").unwrap();
        reject(&args);
        let mut bad: serde_json::Value = serde_json::from_slice(&original).unwrap();
        bad["complete"] = serde_json::json!(false);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        reject(&args);
        std::fs::remove_file(&path).unwrap();
        reject(&args);
        std::fs::write(&path, &original).unwrap();
        std::fs::create_dir(receipts.join("extra")).unwrap();
        reject(&args);
        std::fs::remove_dir(receipts.join("extra")).unwrap();
        std::fs::write(receipts.join("a/exit.json"), b"{\"exit_code\":100}").unwrap();
        reject(&args);
        std::fs::write(receipts.join("a/exit.json"), b"{\"exit_code\":1}").unwrap();
        std::fs::write(
            receipts.join("a/stderr.txt"),
            b"selector mentions empty guard, not runtime error",
        )
        .unwrap();
        reject(&args);
        assert!(publish_new(&path, &serde_json::json!({})).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn observation_reducer_preserves_literal_ties_duplicates_and_empty_guard() {
        std::env::remove_var("LLOOM_PROFILE");
        setup_contract(Contract::Fast).unwrap();
        let args = Args::try_parse_from(["fuzz_geometry", "--contract", "fast"]).unwrap();
        let g = |prompt, decode| Geometry {
            axes: vec![64, 128, 256, 3, 2, 1, 3],
            work: Workload { prompt, decode },
            dense: 0,
            experts: 0,
            arm: 0,
        };
        let observe = |name: &str, g: Geometry, points: Vec<&str>| Observation {
            name: name.into(),
            raw: g.encode(),
            size: size::<Lfm2Tiny>(&g, Weights::default()),
            g,
            outcome: Outcome::Agree {
                maxdiff: 0.,
                scale: 1.,
            },
            points: points.into_iter().map(str::to_owned).collect(),
            steps: vec![],
        };
        let observations = vec![
            observe("a", g(2, 1), vec!["census|tie"]),
            observe("b", g(1, 2), vec!["census|tie", "census|early"]),
            observe("c", g(1, 2), vec!["census|tie", "census|last"]),
        ];
        assert_eq!(observations[0].size, observations[1].size);
        let json = manifest_json::<Lfm2Tiny>(&observations, &args).unwrap();
        assert_eq!(json["entries"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["entries"][0]["geometry"],
            "40008000000103000200010003000102000000"
        );
        assert_eq!(
            json["entries"][0]["smallest_for"],
            serde_json::json!(["census|early", "census|last", "census|tie"])
        );
        // Legacy runs.insert semantics: last canonical duplicate's reaches, not a union.
        assert_eq!(
            json["entries"][0]["reaches"],
            serde_json::json!(["census|tie", "census|last"])
        );
        assert!(manifest_json::<Lfm2Tiny>(&[], &args).is_err());
        let mut empty = observe("empty", g(1, 1), vec![]);
        assert!(manifest_json::<Lfm2Tiny>(std::slice::from_ref(&empty), &args).is_err());
        empty.points.push("census|not-pass".into());
        empty.outcome = Outcome::CpuRefused("refused".into());
        assert!(manifest_json::<Lfm2Tiny>(&[empty], &args).is_err());
    }

    // Borrow the child so the caller can verify its reaped status after unwinding.
    // Only this owned child is signalled; never a process-name or process-group kill.
    struct ChildCleanup<'a>(&'a mut std::process::Child);

    impl Drop for ChildCleanup<'_> {
        fn drop(&mut self) {
            if !matches!(self.0.try_wait(), Ok(Some(_))) {
                // Best effort during unwinding: do not replace the original failure.
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }

    #[test]
    fn publication_child_cleanup_reaps_on_return_and_unwind() {
        for unwind in [false, true] {
            let mut child = std::process::Command::new("/bin/sleep")
                .arg("60")
                .spawn()
                .unwrap();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _cleanup = ChildCleanup(&mut child);
                if unwind {
                    panic!("injected failure before publication readiness");
                }
            }));
            let reaped = child.try_wait();
            // Keep the regression control safe even if cleanup is broken/mutated.
            if !matches!(reaped, Ok(Some(_))) {
                let _ = child.kill();
                let _ = child.wait();
            }
            assert_eq!(outcome.is_err(), unwind);
            assert!(matches!(reaped, Ok(Some(_))), "owned child was not reaped");
        }
    }

    #[test]
    fn publication_sigterm_child() {
        let Some(dir) = std::env::var_os("LLOOM_TEST_PUBLICATION_CHILD") else {
            return;
        };
        let dir = PathBuf::from(dir);
        publish_before_link(
            &dir.join("observation.json"),
            &serde_json::json!({"complete":true}),
            || {
                std::fs::write(dir.join("ready"), b"temporary synced, final not published")?;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            },
        )
        .unwrap();
    }

    #[test]
    fn actual_sigterm_cannot_publish_a_partial_observation() {
        use std::os::unix::process::ExitStatusExt;
        let dir = std::env::temp_dir().join(format!(
            "geometry-publication-sigterm-{}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::publication_sigterm_child", "--nocapture"])
            .env("LLOOM_TEST_PUBLICATION_CHILD", &dir)
            .spawn()
            .unwrap();
        let cleanup = ChildCleanup(&mut child);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !dir.join("ready").exists() && std::time::Instant::now() < deadline {
            assert!(
                cleanup.0.try_wait().unwrap().is_none(),
                "producer exited before readiness"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            dir.join("ready").exists(),
            "producer did not reach synced temporary phase"
        );
        assert!(std::process::Command::new("/bin/kill")
            .args(["-TERM", &cleanup.0.id().to_string()])
            .status()
            .unwrap()
            .success());
        let status = cleanup.0.wait().unwrap();
        assert_eq!(status.signal(), Some(15));
        assert!(!dir.join("observation.json").exists());
        let pending = dir
            .join("observation.json")
            .with_extension(format!("pending-{}", cleanup.0.id()));
        assert!(
            pending.exists(),
            "signal did not interrupt actual synced temporary state"
        );
        println!(
            "SIGTERM signal=15 pending={} final_absent=true",
            pending.display()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn input_enumeration_preserves_sort_and_propagates_entry_errors() {
        let dir =
            std::env::temp_dir().join(format!("geometry-input-enumeration-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        for name in ["z", "a", ".hidden"] {
            std::fs::write(dir.join(name), b"input").unwrap();
        }
        std::fs::create_dir(dir.join("subdir")).unwrap();
        assert_eq!(inputs(&dir).unwrap(), vec![dir.join("a"), dir.join("z")]);
        std::os::unix::fs::symlink(dir.join("absent"), dir.join(".hidden-broken")).unwrap();
        assert_eq!(inputs(&dir).unwrap(), vec![dir.join("a"), dir.join("z")]);
        std::os::unix::fs::symlink(dir.join("a"), dir.join("valid-link")).unwrap();
        assert_eq!(
            inputs(&dir).unwrap(),
            vec![dir.join("a"), dir.join("valid-link"), dir.join("z")]
        );
        std::os::unix::fs::symlink(dir.join("absent"), dir.join("broken")).unwrap();
        let metadata_error = inputs(&dir).unwrap_err();
        assert_eq!(
            metadata_error
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::NotFound
        );

        let entries = [
            Ok(dir.join("z")),
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected directory entry error",
            )),
            Ok(dir.join("a")),
        ];
        let err = input_paths(entries).unwrap_err();
        assert_eq!(err.to_string(), "injected directory entry error");
        assert_eq!(
            err.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::PermissionDenied
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn contract_parser_uses_the_engine_grammar() {
        for word in Contract::SPELLINGS {
            let args = Args::try_parse_from(["fuzz_geometry", "--contract", &word.to_uppercase()])
                .unwrap();
            assert_eq!(args.contract, Contract::parse(word).unwrap());
        }
        assert!(Args::try_parse_from(["fuzz_geometry", "--contract", "balanced"]).is_err());
    }

    #[test]
    fn refused_skipped_and_empty_corpora_cannot_replace_a_manifest() {
        let dir = std::env::temp_dir().join(format!("geometry-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        let out = dir.join("manifest.json");
        std::fs::write(&out, b"retained manifest").unwrap();
        let mut args = Args::try_parse_from(["fuzz_geometry"]).unwrap();
        args.manifest_out = out.clone();
        let g = Geometry::template::<Lfm2Tiny>(Workload {
            prompt: 1,
            decode: 1,
        });
        for outcome in [
            Outcome::CpuRefused("regime mismatch".into()),
            Outcome::MetalRefused("same refusal".into()),
            Outcome::Skipped,
            Outcome::Agree {
                maxdiff: 0.,
                scale: 1.,
            },
        ] {
            std::fs::write(corpus.join("input"), g.encode()).unwrap();
            std::fs::write(corpus.join("second"), g.encode()).unwrap();
            let err = manifest::<Lfm2Tiny>(&corpus, &args, Weights::default(), &mut |_| {
                (g.clone(), outcome.clone())
            })
            .unwrap_err();
            assert!(err.to_string().contains("0 manifest entries"), "{err}");
            assert_eq!(std::fs::read(&out).unwrap(), b"retained manifest");
        }
        std::fs::remove_file(corpus.join("input")).unwrap();
        std::fs::remove_file(corpus.join("second")).unwrap();
        assert!(
            manifest::<Lfm2Tiny>(&corpus, &args, Weights::default(), &mut |_| unreachable!())
                .is_err()
        );
        assert_eq!(std::fs::read(&out).unwrap(), b"retained manifest");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
