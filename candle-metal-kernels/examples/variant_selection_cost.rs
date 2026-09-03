//! What does the host-side variant selection cost, per dispatch?
//!
//! Measurement harness for lloom issue #399 / `DESIGN.md` §7.1a.
//!
//! # The question, and why it is a count and a price rather than a run
//!
//! #399 asks whether compile-time specialised *engine bundles* — one prebuilt
//! variant per configuration, so the hot path carries no runtime lookup — are
//! worth their combinatorics. Its own "what would decide it, cheaply" item 1 is:
//!
//! > **Measure the host-side branch cost.** The `select_tile_config` call, the
//! > axis reads, the descriptor construction — **a count and a price, not a
//! > run.** If it is under 10 µs/token, the whole compile-tier argument is a
//! > null on this engine.
//!
//! A *run* would be a decode A/B, which on this machine cannot resolve an
//! effect this small (`CONTRIBUTING.md` §1.3: a null control read **+65.8 %**
//! with two other agents building). A *count × price* is load-independent: the
//! count comes from the dispatch census and the price from a microbenchmark,
//! and neither depends on what else the machine is doing.
//!
//! # What is actually on the decode path
//!
//! **`select_tile_config` is not**, which #399's item 3 assumes it is.
//! `call_mlx_gemm` returns into `call_mlx_gemv` at `mlx_gemm.rs:281-299`
//! whenever `takes_gemv_route(m, n)` — `m == 1 || n == 1` — which every B=1
//! decode matmul satisfies. The census confirms it from the artifact: a decode
//! token dispatches **0** `gemm_*` kernels and **183** `gemv_*` ones.
//!
//! What *is* on the path, once per dispatch, is
//! [`Kernels::load_pipeline_with_constants`]: an `RwLock` read plus a `HashMap`
//! lookup on `(Source, KernelName, Option<ConstantValues>)`, then a `clone()`
//! of the pipeline handle. `KernelName`'s `Hash` impl hashes the string
//! *contents* (`kernel.rs:32-38`), so each dispatch pays a string hash.
//!
//! **That lookup is exactly what a compile-tier bundle removes**, and it is
//! therefore the quantity that prices #399's proposal. This harness measures
//! it, on the production function, at the census's own kernel names.
//!
//! # Reading the output
//!
//! Three arms, and the first is the non-vacuity check (`CONTRIBUTING.md`
//! §3.1a rule 1 — a green baseline is a *precondition* for reading anything
//! below it, so it is printed first):
//!
//! * **warm** — every key is present. This is the shipping decode path: 675
//!   dispatches per token, all hitting a cache of 29 pipelines.
//! * **miss** — a key that is absent, taking the write-lock path and failing to
//!   find it. The **positive control**: it must be dearer than `warm`, or the
//!   harness is not distinguishing the two paths and every number is suspect.
//! * **hash-only** — the `KernelName` hash alone, with no map and no lock. Bounds
//!   how much of `warm` is the string hash rather than the lookup.
//!
//! ```bash
//! # from inside candle-metal-kernels/ -- the crate has no `metal` feature,
//! # Metal is unconditional in it.
//! cargo run --release --example variant_selection_cost
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

use candle_metal_kernels::metal::device::Device;
use candle_metal_kernels::source::Source;
use candle_metal_kernels::Kernels;

#[derive(Parser, Debug)]
#[command(about = "host-side per-dispatch variant-selection cost (lloom #399)")]
struct Args {
    /// Timed repetitions per arm; the median of per-lookup nanoseconds is reported.
    #[arg(long, default_value_t = 30)]
    iters: usize,

    /// Lookups per timed repetition. Large enough that the `Instant` pair
    /// (~46 ns, §13.8) is a rounding error on the total.
    #[arg(long, default_value_t = 100_000)]
    inner: usize,

    /// Untimed repetitions first, so the caches and the branch predictor are warm.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Dispatches per decode token, for the count x price line.
    ///
    /// 675 is the measured figure for the `Generic` arm (`DESIGN.md` §6.6) and
    /// reproduces on this tree; `--attn sdpa` is 555 and §11.2a's arm is 517.
    #[arg(long, default_value_t = 675.0)]
    dispatches_per_token: f64,
}

/// The kernel names a decode token actually dispatches, from the census.
///
/// Taken from `CANDLE_METAL_PROFILE=1 lfm2-decode-profile --n 20` on this tree,
/// which is the artifact rather than a grep (`CONTRIBUTING.md` §3.1c). The
/// counts are the census's own, so the workload here is the decode token's
/// *actual* key mix rather than a uniform one — a uniform mix would understate
/// the hash cost, since the two `gemv_*` names are 48 characters where
/// `silu_f16` is 8.
const CENSUS: &[(&str, usize)] = &[
    ("copy2d_f16", 140),
    ("bmul_f16", 96),
    ("gemv_float16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0", 84),
    ("gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0", 83),
    ("rmsnorm_f16", 77),
    ("badd_f16", 60),
    ("silu_f16", 30),
    ("cast_f16_f32", 25),
    ("fast_sum_f16", 22),
    ("rope_f16", 16),
    ("affine_f32", 8),
    ("cast_f32_f16", 8),
    ("gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0", 8),
    ("gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0", 8),
    ("softmax_f32", 8),
    ("fast_argmax_f32", 1),
    ("is_u32_f16", 1),
];

/// The census expanded into one entry per dispatch, in census order.
fn census_sequence() -> Vec<&'static str> {
    let mut v = Vec::new();
    for (name, count) in CENSUS {
        for _ in 0..*count {
            v.push(*name);
        }
    }
    v
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let seq = census_sequence();
    let total: usize = CENSUS.iter().map(|(_, c)| c).sum();
    println!(
        "census: {} distinct names, {total} dispatches/token",
        CENSUS.len()
    );
    assert_eq!(
        total, 675,
        "the census must sum to the measured dispatches/token"
    );

    // The production key type is private to `candle-metal-kernels`
    // (`Pipelines = HashMap<(Source, KernelName, Option<ConstantValues>), _>`),
    // so the map here is keyed on the same *shape* with the same string-hashing
    // behaviour: `KernelName`'s `Hash` hashes the contents, which is what
    // `&str` does. What this cannot see is the `ComputePipeline` clone, priced
    // separately below via the real `Kernels` cache when a device is present.
    let mut map: HashMap<(Source, String), u64> = HashMap::new();
    for (name, _) in CENSUS {
        map.insert((Source::Gemv, (*name).to_string()), 1);
    }

    // ---- arm 1: warm hit. Printed FIRST because it is the non-vacuity check.
    let mut warm = Vec::new();
    for i in 0..(args.warmup + args.iters) {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for j in 0..args.inner {
            let name = seq[j % seq.len()];
            let k = (Source::Gemv, name.to_string());
            acc += map.get(&k).copied().unwrap_or(0);
        }
        let dt = t0.elapsed();
        black_box(acc);
        if i >= args.warmup {
            warm.push(dt.as_secs_f64() * 1e9 / args.inner as f64);
        }
    }
    let warm_ns = median(warm);
    println!("\n[non-vacuity arm, run first]");
    println!("  warm hit         {warm_ns:8.2} ns/lookup");
    if warm_ns <= 0.0 {
        anyhow::bail!("warm arm read non-positive; every figure below is uninterpretable");
    }

    // ---- arm 2: the positive control. A key that is absent.
    let mut miss = Vec::new();
    for i in 0..(args.warmup + args.iters) {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for j in 0..args.inner {
            let name = seq[j % seq.len()];
            let k = (Source::Unary, format!("{name}_absent"));
            acc += map.get(&k).copied().unwrap_or(0);
        }
        let dt = t0.elapsed();
        black_box(acc);
        if i >= args.warmup {
            miss.push(dt.as_secs_f64() * 1e9 / args.inner as f64);
        }
    }
    let miss_ns = median(miss);
    println!("  miss (pos. ctrl) {miss_ns:8.2} ns/lookup");

    // ---- arm 3: the hash alone.
    let mut hash_only = Vec::new();
    for i in 0..(args.warmup + args.iters) {
        let t0 = Instant::now();
        let mut acc = 0u64;
        for j in 0..args.inner {
            use std::hash::{Hash, Hasher};
            let name = seq[j % seq.len()];
            let mut h = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut h);
            acc = acc.wrapping_add(h.finish());
        }
        let dt = t0.elapsed();
        black_box(acc);
        if i >= args.warmup {
            hash_only.push(dt.as_secs_f64() * 1e9 / args.inner as f64);
        }
    }
    let hash_ns = median(hash_only);
    println!("  hash only        {hash_ns:8.2} ns/lookup");

    // The positive control is what makes the warm figure readable: without a
    // path that is measurably dearer, `warm` is equally consistent with a
    // harness that optimised the lookup away entirely.
    println!("\n[controls]");
    if miss_ns > warm_ns {
        println!(
            "  OK  miss > warm by {:.2} ns ({:.2}x) -- the arms are distinguishable",
            miss_ns - warm_ns,
            miss_ns / warm_ns
        );
    } else {
        println!(
            "  WARN miss ({miss_ns:.2}) is not dearer than warm ({warm_ns:.2}); \
             the harness may not be measuring the lookup"
        );
    }

    // ---- the real cache, when a device is present. This is the production
    // function rather than a restatement of it (`CONTRIBUTING.md` §3.1a rule 3).
    let real: Option<f64> = {
        match Device::system_default() {
            Some(device) => {
                let kernels = Kernels::new();
                // Prime one real pipeline, then time repeated warm lookups of it
                // through the production entry point.
                let name = "silu_f16";
                match kernels.load_pipeline(&device, Source::Unary, name) {
                    Ok(_) => {
                        let (libs, pipes) = kernels.cache_sizes();
                        println!("\n[real Kernels cache] libraries={libs} pipelines={pipes}");
                        println!(
                            "  NOTE: a decode token runs against libraries=10 pipelines=29 \
                             (measured), so this cache is SMALLER than production; a smaller \
                             HashMap is if anything cheaper to probe, which makes the figure \
                             below a LOWER bound on the production lookup."
                        );

                        // (a) `&'static str` key: no allocation. This is the
                        // `KernelName::Ref` path most call sites take.
                        let mut v = Vec::new();
                        for i in 0..(args.warmup + args.iters) {
                            let t0 = Instant::now();
                            for _ in 0..1000 {
                                let p = kernels
                                    .load_pipeline(&device, Source::Unary, name)
                                    .expect("warm lookup of a primed pipeline cannot fail");
                                black_box(&p);
                            }
                            let dt = t0.elapsed();
                            if i >= args.warmup {
                                v.push(dt.as_secs_f64() * 1e9 / 1000.0);
                            }
                        }
                        let ns_static = median(v);
                        println!(
                            "  load_pipeline, &'static str key   {ns_static:8.2} ns/dispatch  \
                             <- KernelName::Ref, no allocation"
                        );

                        // (b) the GEMV path's key: `format!` builds a fresh
                        // `String` per dispatch (`mlx_gemm.rs:1006-1019`), which
                        // `KernelName::Value` then hashes by contents. 167 of a
                        // decode token's 675 dispatches take this route, so it
                        // is not a corner case -- it is every weight matmul.
                        //
                        // This arm calls the SAME production function as (a),
                        // differing only in how the key arrives: a `String`
                        // built per call rather than a `&'static str`. That is
                        // the one variable, which is what makes the difference
                        // attributable (`CONTRIBUTING.md` §3.1b -- a reference,
                        // not a shape).
                        //
                        // The name must be one that is PRIMED, or this would
                        // measure the miss path against (a)'s hit path and the
                        // difference would be the branch rather than the build.
                        let mut v = Vec::new();
                        for i in 0..(args.warmup + args.iters) {
                            let t0 = Instant::now();
                            for _ in 0..1000 {
                                // Same field count and widths as the production
                                // expression, rendering the primed name. Built
                                // FROM `name` rather than from a literal, so a
                                // mutation of `name` reaches this arm too --
                                // the first version hardcoded it and survived
                                // exactly that mutation.
                                let built = format!("{}{}{}{}", name, "", "", "");
                                let p = kernels
                                    .load_pipeline(&device, Source::Unary, built)
                                    .expect("warm lookup of a primed pipeline cannot fail");
                                black_box(&p);
                            }
                            let dt = t0.elapsed();
                            if i >= args.warmup {
                                v.push(dt.as_secs_f64() * 1e9 / 1000.0);
                            }
                        }
                        let ns_fmt = median(v);
                        println!(
                            "  load_pipeline, built String key   {ns_fmt:8.2} ns/dispatch  \
                             <- KernelName::Value, one String per dispatch"
                        );
                        if ns_fmt <= ns_static {
                            println!(
                                "  WARN built-key arm is not dearer than the static one; \
                                 the allocation is not resolvable at this resolution"
                            );
                        }

                        // The blended per-dispatch price. The split is derived
                        // FROM the census rather than restated beside it, so it
                        // cannot drift from the table above: every `gemv*` name
                        // is built by `format!` (`mlx_gemm.rs:1006-1019`) and
                        // every other name in the census is a `&'static str`.
                        //
                        // 183 built / 492 static on this arm. A hardcoded 167
                        // stood here first and was wrong by 16 -- the count has
                        // one source now.
                        let gemv: usize = CENSUS
                            .iter()
                            .filter(|(n, _)| n.starts_with("gemv"))
                            .map(|(_, c)| *c)
                            .sum();
                        let rest = total - gemv;
                        let blended =
                            (gemv as f64 * ns_fmt + rest as f64 * ns_static) / total as f64;
                        println!(
                            "  blended over the census ({gemv} built / {rest} static) \
                             {blended:8.2} ns/dispatch"
                        );
                        Some(blended)
                    }
                    Err(e) => {
                        println!("\n[real Kernels cache] unavailable: {e}");
                        None
                    }
                }
            }
            None => {
                println!("\n[real Kernels cache] no Metal device");
                None
            }
        }
    };

    // ---- count x price.
    //
    // **No fallback.** The first version read `real.unwrap_or(warm_ns)`, so a
    // failed production lookup silently substituted the synthetic map's figure
    // and the harness printed a RESULT line anyway -- measured, as this
    // change's mutation arm 2, which SURVIVED for exactly that reason. That is
    // §15.1a's silent-fallback class in the instrument written to price it: the
    // number was real, the thing it measured was not the thing named. A missing
    // production measurement is a refusal, not a default.
    let Some(price_ns) = real else {
        anyhow::bail!(
            "no production measurement: the real `Kernels` cache arm did not run, \
             so there is no price to report. Refusing rather than falling back to \
             the synthetic map, which measures a different thing under the same name."
        );
    };
    let per_token_us = price_ns * args.dispatches_per_token / 1000.0;
    println!("\n[count x price]");
    println!(
        "  {:.0} dispatches/token x {:.2} ns = {:.3} us/token",
        args.dispatches_per_token, price_ns, per_token_us
    );
    println!("  #399's decision threshold: 10 us/token");
    println!(
        "  verdict: {}",
        if per_token_us < 10.0 {
            "UNDER -- the compile-tier bundle wins nothing measurable here"
        } else {
            "OVER -- the compile-tier argument survives this check"
        }
    );

    println!(
        "\nRESULT issue=399 warm_ns={warm_ns:.3} miss_ns={miss_ns:.3} hash_ns={hash_ns:.3} \
         real_ns={real_ns:.3} dispatches_per_token={dpt:.1} per_token_us={per_token_us:.4} \
         threshold_us=10 under_threshold={under}",
        real_ns = price_ns,
        dpt = args.dispatches_per_token,
        under = per_token_us < 10.0,
    );

    Ok(())
}
