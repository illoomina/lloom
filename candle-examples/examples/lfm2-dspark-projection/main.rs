//! Run the DSpark context projection on the REAL drafter and the REAL target —
//! `DESIGN.md` §10.2m/§10.2o, issue #363.
//!
//! # What this is for
//!
//! §10.2n built the target-side tap and stopped, naming the projection as the
//! next increment and the **first falsifiable surface the drafter offers**: it
//! is reachable before the backbone exists, and it is where a wrong tap becomes
//! observable. `lfm2_dspark_projection.rs` tests the arithmetic on a
//! hand-computed reference; **this harness is the other half — it proves the
//! seam carries real weights and real activations end to end.**
//!
//! That distinction is the one `CONTRIBUTING.md` §7.1 F4 and §3.1c are about.
//! A unit test proves the projection is *correct*; only a run against the
//! checkpoint proves it is *reachable*, and this project has shipped a correct,
//! tested, fully-gated mechanism wired into nothing (#499/#500) — where the
//! digests were unmoved and the tests passed **because the new path was not on
//! the path**.
//!
//! ```bash
//! cargo run --release --features metal --example lfm2-dspark-projection
//! cargo run --release --example lfm2-dspark-projection -- --cpu   # no metal
//! ```
//!
//! # What it checks, and what it deliberately does not
//!
//! It reports the projected `H_ctx` — shape, dtype, and a SHA-256 over the
//! bytes — plus per-slice statistics of the five taps that fed it. Three
//! structural checks exit non-zero:
//!
//! 1. **the tap arity matches the drafter's `fc.weight`** — five taps against
//!    `[2048, 10240]`, which is the seam between §10.2n's producer and this
//!    consumer, and the one place a mismatch would otherwise surface late;
//! 2. **every tapped slice is distinct** — five identical slices would project
//!    to a well-formed `H_ctx` from a tap that captured the same layer five
//!    times, which no shape check can see;
//! 3. **`H_ctx` is finite** — a NaN here is a silent-wrong-answer that
//!    propagates into an acceptance rate and nowhere else.
//!
//! **It does NOT claim a parity result.** §10.2m's soundness gate is an
//! `H_ctx` ulp-match against llama.cpp for the same input and layers, and that
//! needs llama.cpp's `H_ctx` for the same input — which is not in this tree.
//! Printing a digest and calling it parity would be a comparison against
//! nothing (§3.1b: *a test whose reference comes from the same run as its
//! subject looks like a comparison and is not*). **The digest here is a
//! reproducibility anchor, not a verdict**, and §10.2o records the parity as
//! owed.
//!
//! **And it takes no timing.** §7.1a: no `.bench/` row, no lease, no default
//! flipped. The tap is off unless a caller names layers, and this is that
//! caller.
//!
//! # The witness conflict this harness makes visible
//!
//! Our tap is the layer-**input** residual (§10.2n), following llama.cpp
//! PR #27383's `t_layer_inp[il]`, whose consumer passes the drafter's indices
//! unshifted under the comment *"turn on extraction of the target layers'
//! **input** embeddings"*. The native MLX port captures the layer **output**
//! instead, with no compensating shift. Both are §10.2m witnesses and they
//! differ by one layer. `--tap-shift` runs the other convention so the two are
//! comparable on one machine the moment a reference exists; it changes which
//! layers are tapped and nothing else.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_examples::lfm2_setup::{parse_config, weight_files};
use candle_nn::VarBuilder;
use candle_transformers::models::lfm2::{Cache, Model};
use candle_transformers::models::lfm2_dspark::{DSparkProjection, CHECKPOINT_DTYPE};
use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Project real target taps through the real DSpark fc + encoder norm")]
struct Args {
    /// The 2.6B target checkpoint. This is the target the drafter is trained
    /// against (§5.11, §10.2l) — VL-3B has no DSpark draft and one does not
    /// transfer, so the default is deliberately not `default_model_dir()`.
    #[arg(long)]
    target_dir: Option<PathBuf>,

    /// The DSpark drafter checkpoint (supplies `fc.weight`, `hidden_norm.weight`).
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Run on the CPU even when built with `metal`.
    #[arg(long)]
    cpu: bool,

    /// Tokens to run through the target. The projection's `M` is this, and
    /// §10.2m sets the real `M = accepted_len`, 1..n_max.
    #[arg(long, default_value_t = 4)]
    tokens: usize,

    /// Shift every tap index by this much before running.
    ///
    /// `0` is our convention and llama.cpp's: the layer-**input** residual at
    /// `target_layer_ids`. `1` reproduces mlx-dspark's, whose capture is after
    /// the block — the input of layer `l+1` is the output of layer `l`, so a
    /// shift of one is exactly that convention on an input-tapping engine.
    /// Recorded as an arm rather than a constant because the two witnesses
    /// disagree and this tree cannot yet adjudicate (§10.2o).
    #[arg(long, default_value_t = 0)]
    tap_shift: usize,
}

fn newest_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// `lfm2-determinism`'s SHA-256, included by path rather than copied.
///
/// One implementation, and it carries the NIST self-test that runs below — a
/// digest from an unverified hasher is a number, not an anchor.
#[path = "../lfm2-determinism/sha256.rs"]
mod sha256;

fn sha256(bytes: &[u8]) -> String {
    sha256::digest_hex(bytes)
}

/// `target_layer_ids` from the drafter's own config — read, never assumed.
///
/// §10.2n's rule: the indices are a property of the *drafter* checkpoint, and
/// welding them into the target is the conflation §10.1 renames to avoid.
fn drafter_tap_layers(draft_dir: &Path) -> Result<(Vec<usize>, usize, f64)> {
    let raw = std::fs::read_to_string(draft_dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", draft_dir.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&raw)?;
    let dfc = cfg.get("dflash_config").unwrap_or(&cfg);
    let ids: Vec<usize> = dfc
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .context("drafter config has no dflash_config.target_layer_ids")?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    let hidden = cfg
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .context("drafter config has no hidden_size")? as usize;
    let eps = cfg
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5);
    Ok((ids, hidden, eps))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // The hasher is checked before anything is hashed. A digest is only an
    // anchor if the thing producing it is known good, and this is the same
    // self-test `lfm2-determinism` runs (NIST vectors + 1e6*'a' + streaming).
    sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
    println!("sha256 self-test  : OK");

    let target_dir = args
        .target_dir
        .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B"))
        .context(
            "no LFM2.5-2.6B snapshot in the HF cache and no --target-dir. \
             The drafter is trained against THIS target (§10.2l); VL-3B has no \
             DSpark draft and one does not transfer.",
        )?;
    let draft_dir = args
        .draft_dir
        .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark"))
        .context(
            "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir. \
             §10.2n: the design sections' constants came from a committed \
             6480-byte HEADER BLOB — 'characterised' and 'loadable' are \
             different claims. Fetch model.safetensors (655 421 522 B).",
        )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        candle_examples::device(false)?
    };

    println!("target : {}", target_dir.display());
    println!("draft  : {}", draft_dir.display());
    println!("device : {device:?}");

    let (mut tap_layers, draft_hidden, eps) = drafter_tap_layers(&draft_dir)?;
    if args.tap_shift != 0 {
        println!(
            "\nNOTE: --tap-shift {} -- reproducing mlx-dspark's after-the-block \
             capture convention. Our default (0) is llama.cpp's layer-INPUT \
             residual. See DESIGN.md 10.2o; the two witnesses disagree.",
            args.tap_shift
        );
        tap_layers = tap_layers.iter().map(|l| l + args.tap_shift).collect();
    }
    println!("\ntap_layers        : {tap_layers:?}");
    println!("drafter hidden    : {draft_hidden}");
    println!("drafter rms eps   : {eps:e}");

    // ---- the target, tapping at the drafter's indices -----------------------
    let raw = std::fs::read_to_string(target_dir.join("config.json"))?;
    let mut cfg = parse_config(&raw)?;
    if cfg.hidden_size != draft_hidden {
        bail!(
            "target hidden {} != drafter hidden {draft_hidden}: fc.weight's \
             column count could not be a whole number of target hidden states",
            cfg.hidden_size
        );
    }
    for &l in &tap_layers {
        if l >= cfg.num_hidden_layers {
            bail!(
                "tap layer {l} is past the target's {} layers",
                cfg.num_hidden_layers
            );
        }
    }
    cfg.tap_layers = tap_layers.clone();
    cfg.tap_dtype = DType::F32; // §10.2m: deliberate, buys acceptance rate

    let files = weight_files(&target_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F16, &device)? };
    let model = Model::new(&cfg, vb)?;
    let mut cache = Cache::new(true, DType::F16, &cfg, &device)?;

    // ---- the drafter's projection, from its own checkpoint ------------------
    let draft_files = vec![draft_dir.join("model.safetensors")];
    if !draft_files[0].exists() {
        bail!(
            "{} does not exist. §10.2n: a committed header blob reads exactly \
             like the real thing in prose and has no payload.",
            draft_files[0].display()
        );
    }
    let dvb =
        unsafe { VarBuilder::from_mmaped_safetensors(&draft_files, CHECKPOINT_DTYPE, &device)? };
    let proj = DSparkProjection::new(draft_hidden, tap_layers.len(), eps, dvb)?;
    println!(
        "projection loaded : fc [{}, {}], {} taps",
        proj.hidden(),
        proj.n_tap() * proj.hidden(),
        proj.n_tap()
    );

    // ---- run the target, drain the taps, project ----------------------------
    let ids: Vec<u32> = (1..=args.tokens as u32).collect();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let _logits = model.forward(&input, 0, &mut cache)?;

    let taps = cache.take_taps();
    println!("\ntaps drained      : {}", taps.len());
    if taps.len() != proj.n_tap() {
        bail!(
            "FAIL: the target produced {} taps and fc.weight consumes {} — the \
             producer (§10.2n) and this consumer disagree about arity",
            taps.len(),
            proj.n_tap()
        );
    }

    // Every slice distinct. Five identical slices project to a well-formed
    // H_ctx and mean the tap captured one layer five times — a defect no shape
    // check can see, and §3.1b's whole subject.
    let mut slice_digests = Vec::new();
    println!("\n  slice  layer  shape                 mean        absmax     sha256[..16]");
    for (i, t) in taps.iter().enumerate() {
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let absmax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = sha256(&bytes);
        println!(
            "  {i:>5}  {:>5}  {:<20}  {mean:>10.6}  {absmax:>9.4}  {}",
            tap_layers[i],
            format!("{:?}", t.dims()),
            &d[..16]
        );
        slice_digests.push(d);
    }
    let mut uniq = slice_digests.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() != slice_digests.len() {
        bail!(
            "FAIL: only {} of {} tapped slices are distinct — the tap is \
             capturing the same layer more than once, which projects to a \
             well-formed H_ctx and is invisible to every shape check",
            uniq.len(),
            slice_digests.len()
        );
    }
    println!("  all {} slices distinct", slice_digests.len());

    let h_ctx = proj.forward(&taps)?;
    let hv = h_ctx.flatten_all()?.to_vec1::<f32>()?;
    if let Some(bad) = hv.iter().position(|x| !x.is_finite()) {
        bail!("FAIL: H_ctx[{bad}] is not finite ({})", hv[bad]);
    }
    let hbytes: Vec<u8> = hv.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mean = hv.iter().sum::<f32>() / hv.len() as f32;
    let absmax = hv.iter().fold(0f32, |m, x| m.max(x.abs()));

    println!("\nH_ctx");
    println!("  shape           : {:?}", h_ctx.dims());
    println!("  dtype           : {:?}", h_ctx.dtype());
    println!("  mean            : {mean:.6}");
    println!("  absmax          : {absmax:.4}");
    println!("  sha256          : {}", sha256(&hbytes));
    println!(
        "\nRESULT tokens={} taps={} tap_shift={} h_ctx={:?} sha256={}",
        args.tokens,
        taps.len(),
        args.tap_shift,
        h_ctx.dims(),
        sha256(&hbytes)
    );
    println!(
        "\nNOTE: the digest is a reproducibility anchor, NOT a parity verdict. \
         §10.2m's soundness gate is an H_ctx ulp-match against llama.cpp for \
         the same input, and llama.cpp's H_ctx is not in this tree (§10.2o)."
    );
    Ok(())
}
