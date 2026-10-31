//! The register and occupancy census of the pipelines a run built (#1154 C).
//!
//! Input: the `[pipeline: …]` lines a run prints under
//! `LLOOM_PIPELINE_CENSUS=1` (`kernels::pipeline_census`), from any number of
//! log files. Each named pipeline is built again here from the same source,
//! function and function constants through the production compile options
//! (`Kernels::load_library`), and so is its f16-activation counterpart when
//! the library instantiates one. For each, the three fields the driver
//! reports at pipeline creation are printed — `maxTotalThreadsPerThreadgroup`,
//! the execution width and the static threadgroup memory — and the pipeline's
//! binary archive is written so its AGX code can be read offline
//! (`design/measurements/issue-1154-raw/census/analyse.py`).
//!
//! `maxTotalThreadsPerThreadgroup` is the driver's own occupancy statement and
//! falls with live named scalars (`docs/tuning.md` §13.1b), but a kernel that
//! declares `max_total_threads_per_threadgroup` reads its declaration and a
//! kernel staging threadgroup memory is bounded by that instead (§13.1d). So
//! the field is printed beside the threadgroup memory, and the declared
//! bounds are read from the source by the analysis, never inferred here.
//!
//! Two controls run first, before any row is read:
//!
//! * every rebuilt pipeline must report the `max_threads` its census line
//!   carried, so a row is a statement about the pipeline the run built;
//! * the live-scalar ladder (§13.1b's fixture, the name byte-identical across
//!   rungs) at `float` and at `half`: the field must fall on the `float`
//!   ladder, or no 1024 below means anything. The `half` ladder is the
//!   measured answer to whether a half value takes half a register.
//!
//! No GPU work is dispatched: compiling and serializing use the driver, not
//! the queue `lloom-arb` arbitrates, so this needs a build lease, not a
//! measure lease.
//!
//! ```bash
//! LLOOM_PIPELINE_CENSUS=1 lloom chat … 2>| run.log
//! ./target/release/examples/register_census --out /tmp/census run.log [more.log …]
//! ```

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use lloom_tensor::kernels::metal::{ComputePipeline, ConstantValues, Device, Library, Value};
use lloom_tensor::kernels::pipeline_census::{self, Entry, SOURCES};
use lloom_tensor::kernels::source::Source;
use lloom_tensor::kernels::Kernels;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLCompileOptions, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    /// Where the binary archives and the table go.
    #[arg(long)]
    out: PathBuf,
    /// Live-scalar counts for the ladder control.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,8,16,32,48,64,80,96,112,128,160,192,256"
    )]
    ladder: Vec<usize>,
    /// Pipelines no run reached, by `<Source>:<function-name prefix>`: every
    /// function of that source's library starting with the prefix is built
    /// (the row capacities, the fused stores) with the EOS gate's constant
    /// off where the function names one.
    #[arg(long)]
    extra: Vec<String>,
    /// Logs carrying `[pipeline: …]` lines.
    logs: Vec<PathBuf>,
}

struct Built {
    max_threads: usize,
    exec_width: usize,
    tg_mem: usize,
    archive: String,
}

fn archive(device: &Device, function: &ProtocolObject<dyn MTLFunction>, path: &Path) -> Result<()> {
    let desc = MTLComputePipelineDescriptor::new();
    desc.setComputeFunction(Some(function));
    desc.setSupportIndirectCommandBuffers(lloom_tensor::kernels::metal::pipelines_support_icb());
    let raw: &ProtocolObject<dyn MTLDevice> = device.as_ref();
    let ar = raw
        .newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new())
        .map_err(|e| anyhow::anyhow!("archive: {e}"))?;
    ar.addComputePipelineFunctionsWithDescriptor_error(&desc)
        .map_err(|e| anyhow::anyhow!("archive add: {e}"))?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    ar.serializeToURL_error(&url)
        .map_err(|e| anyhow::anyhow!("archive write {}: {e}", path.display()))?;
    Ok(())
}

fn fields(p: &ComputePipeline) -> (usize, usize, usize) {
    (
        p.max_total_threads_per_threadgroup(),
        p.thread_execution_width(),
        p.static_threadgroup_memory_length(),
    )
}

fn clone_values(values: &[(usize, Value)]) -> Vec<(usize, Value)> {
    values
        .iter()
        .map(|(i, v)| {
            (
                *i,
                match v {
                    Value::Bool(b) => Value::Bool(*b),
                    Value::USize(u) => Value::USize(*u),
                    Value::U16(h) => Value::U16(*h),
                    Value::F32(f) => Value::F32(*f),
                },
            )
        })
        .collect()
}

fn build(
    device: &Device,
    lib: &Library,
    function: &str,
    constants: Option<&[(usize, Value)]>,
    out: &Path,
    tag: &str,
) -> Result<Built> {
    let constants = constants.map(|c| ConstantValues::new(clone_values(c)));
    let func = lib
        .get_function(function, constants.as_ref())
        .map_err(|e| anyhow::anyhow!("{function}: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("{function}: {e}"))?;
    let (max_threads, exec_width, tg_mem) = fields(&pipeline);
    let file = format!("{tag}.metallib");
    archive(device, func.as_ref(), &out.join(&file))?;
    Ok(Built {
        max_threads,
        exec_width,
        tg_mem,
        archive: file,
    })
}

/// The f16-activation counterpart names to look for, in order.
fn f16_candidates(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    if name.starts_with("kernel_mul_mv") {
        // The quantized matvec triple token: `_f32` (or `_f32_r<N>`) at the end.
        if let Some(pos) = name.rfind("_f32") {
            let (head, tail) = name.split_at(pos);
            out.push(format!("{head}_f16_f16{}", &tail[4..]));
        }
    }
    if let Some(stem) = name
        .strip_prefix("kernel_mul_mm_")
        .and_then(|s| s.strip_suffix("_f32"))
    {
        out.push(format!("kernel_mul_mm_{stem}_f16_f16"));
        out.push(format!("kernel_mul_mm_{stem}_f16"));
    }
    let swapped = name
        .replace("float32", "float16")
        .replace("_f32", "_f16")
        .replace("f32_", "f16_");
    if swapped != name {
        out.push(swapped);
    }
    // The dense MLX names spell the dtype `float` / `float16_t`.
    if name.contains("_float_") {
        out.push(name.replace("_float_", "_float16_t_"));
        out.push(name.replace("_float_", "_half_"));
    }
    if let Some(s) = name.strip_suffix("_float") {
        out.push(format!("{s}_float16_t"));
    }
    out.dedup();
    out
}

fn ladder_source(ty: &str, n: usize) -> String {
    let mut decl = String::new();
    let mut seed = String::new();
    let mut step = String::new();
    let mut comb = String::new();
    for i in 0..n {
        decl += &format!("  {ty} a{i};\n");
        seed += &format!("  a{i} = in[(t + {}) & 1023];\n", i * 7 + 1);
        step += &format!(
            "    a{i} = fma(a{i}, {ty}(1.0000{:03}), v * {ty}({}.5));\n",
            (i % 999) + 1,
            (i % 13) + 1
        );
        comb += &format!("  s += a{i};\n");
    }
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         [[kernel]] void k_ladder(device {ty}* out [[buffer(0)]],\n\
           const device {ty}* in [[buffer(1)]],\n\
           uint t [[thread_position_in_grid]]) {{\n{decl}{seed}\
           for (uint k = 0; k < 64u; ++k) {{\n    {ty} v = in[(t + k) & 1023];\n{step}  }}\n\
           {ty} s = {ty}(0);\n{comb}  out[t] = s;\n}}\n"
    )
}

/// The production compile options' math settings (`kernel.rs`
/// `get_compile_options`, default `fast_math` on), for the ladder's inline
/// source, which has no `Source` to compile through.
fn ladder_options() -> objc2::rc::Retained<MTLCompileOptions> {
    let o = MTLCompileOptions::new();
    if lloom_tensor::kernels::fast_math_enabled() {
        o.setMathMode(MTLMathMode::Fast);
        o.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    } else {
        o.setMathMode(MTLMathMode::Relaxed);
        o.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    }
    o
}

fn run_ladder(
    device: &Device,
    counts: &[usize],
    out: &Path,
) -> Result<BTreeMap<(String, usize), usize>> {
    let raw: &ProtocolObject<dyn MTLDevice> = device.as_ref();
    let mut got = BTreeMap::new();
    println!("LADDER ty live max_threads tg_mem archive");
    for ty in ["float", "half"] {
        for &n in counts {
            let src = ladder_source(ty, n);
            let lib = raw
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&src),
                    Some(&ladder_options()),
                )
                .map_err(|e| anyhow::anyhow!("ladder {ty} {n}: {e}"))?;
            let func = lib
                .newFunctionWithName(&NSString::from_str("k_ladder"))
                .context("k_ladder")?;
            let p = raw
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| anyhow::anyhow!("ladder {ty} {n}: {e}"))?;
            let file = format!("ladder-{ty}-{n:03}.metallib");
            archive(device, &func, &out.join(&file))?;
            let m = p.maxTotalThreadsPerThreadgroup();
            println!(
                "LADDER {ty} {n} {m} {} {file}",
                p.staticThreadgroupMemoryLength()
            );
            got.insert((ty.to_string(), n), m);
        }
    }
    Ok(got)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.logs.is_empty(),
        "name at least one log carrying [pipeline: …] lines"
    );
    std::fs::create_dir_all(&args.out)?;
    let device = Device::system_default().context("no Metal device")?;
    let kernels = Kernels::new();

    println!(
        "CONTEXT fast_math={} icb_pipelines={} compile_flags={}",
        lloom_tensor::kernels::fast_math_enabled(),
        lloom_tensor::kernels::metal::pipelines_support_icb(),
        lloom_tensor::kernels::metal_compile_flags().unwrap_or("none")
    );

    // The ladder first: if the field cannot fall, nothing below is readable.
    let ladder = run_ladder(&device, &args.ladder, &args.out)?;
    let first = ladder[&("float".to_string(), args.ladder[0])];
    let last = ladder[&("float".to_string(), *args.ladder.last().unwrap_or(&1))];
    ensure!(
        last < first,
        "the float ladder did not fall ({first} at {} live, {last} at {}): \
         every 1024 below would be unreadable",
        args.ladder[0],
        args.ladder.last().unwrap_or(&1)
    );

    // The population: every distinct pipeline any log names, with the
    // `max_threads` its run reported and which logs named it.
    let mut entries: BTreeMap<String, (Entry, usize, BTreeSet<String>)> = BTreeMap::new();
    let mut lines = 0usize;
    for log in &args.logs {
        let text = std::fs::read_to_string(log).with_context(|| log.display().to_string())?;
        let tag = log
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        for l in text
            .lines()
            .filter(|l| l.trim_start().starts_with("[pipeline: "))
        {
            lines += 1;
            let Some(entry) = pipeline_census::parse(l) else {
                bail!("{}: unparsable census line: {l}", log.display());
            };
            let reported: usize = l
                .split(' ')
                .find_map(|p| p.strip_prefix("max_threads="))
                .and_then(|v| v.parse().ok())
                .with_context(|| format!("no max_threads in {l}"))?;
            let key = l
                .split(" max_threads=")
                .next()
                .unwrap_or(l)
                .trim()
                .to_string();
            entries
                .entry(key)
                .or_insert((entry, reported, BTreeSet::new()))
                .2
                .insert(tag.clone());
        }
    }
    ensure!(
        lines > 0,
        "no [pipeline: …] line in any log (was LLOOM_PIPELINE_CENSUS=1 set?)"
    );
    println!("POPULATION lines={lines} distinct={}", entries.len());

    // Every library, so a counterpart is found wherever it is instantiated.
    let mut libs: Vec<(Source, Library, BTreeSet<String>)> = Vec::new();
    for s in SOURCES {
        let lib = kernels
            .load_library(&device, s)
            .map_err(|e| anyhow::anyhow!("{s:?}: {e}"))?;
        let names = lib.function_names().into_iter().collect();
        libs.push((s, lib, names));
    }
    let find = |name: &str| libs.iter().find(|(_, _, n)| n.contains(name));

    println!("ROW arm source function constants max_threads exec_width tg_mem archive logs");
    let mut mismatches = 0;
    for (i, (_key, (entry, reported, logs))) in entries.iter().enumerate() {
        let (_, lib, _) = libs
            .iter()
            .find(|(s, _, _)| *s == entry.source)
            .context("source")?;
        let constants = entry.constants.as_deref();
        let cstr = match constants {
            None => "-".to_string(),
            Some(c) => c
                .iter()
                .map(|(i, v)| format!("{i}={v:?}"))
                .collect::<Vec<_>>()
                .join(","),
        };
        let tag = format!("p{i:03}");
        let b = build(&device, lib, &entry.function, constants, &args.out, &tag)?;
        if b.max_threads != *reported {
            mismatches += 1;
            println!(
                "MISMATCH {} reported={} rebuilt={}",
                entry.function, reported, b.max_threads
            );
        }
        let logs = logs.iter().cloned().collect::<Vec<_>>().join("+");
        println!(
            "ROW run {:?} {} {} {} {} {} {} {}",
            entry.source,
            entry.function,
            cstr,
            b.max_threads,
            b.exec_width,
            b.tg_mem,
            b.archive,
            logs
        );
        let mut found = false;
        for cand in f16_candidates(&entry.function) {
            if cand == entry.function {
                continue;
            }
            if let Some((src, clib, _)) = find(&cand) {
                // The gate's constant is part of every gated source's
                // function; keep the run's constants where the counterpart
                // shares the source, the gate's alone otherwise.
                let cvals: Option<Vec<(usize, Value)>> = if *src == entry.source {
                    constants.map(clone_values)
                } else {
                    constants.map(|c| {
                        clone_values(c)
                            .into_iter()
                            .filter(|(i, _)| {
                                *i == lloom_tensor::kernels::metal::eos_gate::EOS_GATE_FUNCTION_CONSTANT
                            })
                            .collect()
                    })
                };
                let ctag = format!("{tag}-f16");
                match build(&device, clib, &cand, cvals.as_deref(), &args.out, &ctag) {
                    Ok(cb) => println!(
                        "ROW f16-of:{} {:?} {} {} {} {} {} {} {}",
                        entry.function,
                        src,
                        cand,
                        cstr,
                        cb.max_threads,
                        cb.exec_width,
                        cb.tg_mem,
                        cb.archive,
                        logs
                    ),
                    Err(e) => println!(
                        "ROW f16-of:{} {:?} {} BUILD-FAILED {e}",
                        entry.function, src, cand
                    ),
                }
                found = true;
                break;
            }
        }
        if !found && (entry.function.contains("f32") || entry.function.contains("float")) {
            println!("ROW f16-of:{} - ABSENT - - - - - {}", entry.function, logs);
        }
    }
    // The extra population: the same three fields and archive, no counterpart
    // search (the prefix names both triples where both exist).
    let gate_off = [(
        lloom_tensor::kernels::metal::eos_gate::EOS_GATE_FUNCTION_CONSTANT,
        Value::Bool(false),
    )];
    let mut extras = 0usize;
    for spec in &args.extra {
        let (src_name, prefix) = spec
            .split_once(':')
            .with_context(|| format!("--extra wants <Source>:<prefix>, got {spec}"))?;
        let (src, lib, names) = libs
            .iter()
            .find(|(s, _, _)| format!("{s:?}") == src_name)
            .with_context(|| format!("no source named {src_name}"))?;
        let matched: Vec<&String> = names.iter().filter(|n| n.starts_with(prefix)).collect();
        ensure!(
            !matched.is_empty(),
            "--extra {spec}: no function of {src_name} starts with {prefix}"
        );
        for name in matched {
            let tag = format!("x{extras:04}");
            extras += 1;
            let built = build(&device, lib, name, Some(&gate_off), &args.out, &tag)
                .map(|b| (b, "777=Bool(false)"))
                .or_else(|_| build(&device, lib, name, None, &args.out, &tag).map(|b| (b, "-")));
            match built {
                Ok((b, cstr)) => println!(
                    "ROW extra {src:?} {name} {cstr} {} {} {} {} {spec}",
                    b.max_threads, b.exec_width, b.tg_mem, b.archive
                ),
                Err(e) => println!("ROW extra {src:?} {name} BUILD-FAILED {e}"),
            }
        }
    }
    println!("EXTRA rows={extras}");

    ensure!(
        mismatches == 0,
        "{mismatches} rebuilt pipeline(s) disagree with what their run reported"
    );
    println!("DONE rows={} mismatches=0", entries.len());
    Ok(())
}
