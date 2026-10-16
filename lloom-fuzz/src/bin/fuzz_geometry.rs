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
    use std::collections::BTreeMap;
    let files = inputs(dir)?;
    // point -> (size, geometry bytes)
    let mut best: BTreeMap<String, (usize, Vec<u8>)> = BTreeMap::new();
    let mut elsewhere: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut runs: BTreeMap<Vec<u8>, (String, Vec<String>)> = BTreeMap::new();
    for f in &files {
        let (g, o, (ps, ls)) = exec(&std::fs::read(f)?);
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
    anyhow::ensure!(
        !entries.is_empty(),
        "{}: {} inputs, {} agreeing runs, 0 manifest entries; refusing to overwrite {}",
        F::NAME,
        files.len(),
        runs.len(),
        args.manifest_out.display()
    );
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let json = serde_json::json!({
        "family": F::NAME,
        "contract": args.contract.name(),
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
