//! Run the DSpark drafter on the REAL drafter and the REAL target —
//! `DESIGN.md` §10.2p, issue #363.
//!
//! # What this is for
//!
//! §10.2o built the projection and named the backbone as the next increment.
//! `lfm2_dspark_backbone.rs` tests the arithmetic against hand-computed
//! references; **this harness is the other half — it proves the seam carries
//! real weights and real activations end to end.**
//!
//! That distinction is `CONTRIBUTING.md` §7.1 F4 and §3.1c. A unit test proves
//! the drafter is *correct*; only a run against the checkpoint proves it is
//! *reachable*, and this project has shipped a correct, tested, fully-gated
//! mechanism wired into nothing (#499/#500) — where the digests were unmoved
//! and the tests passed **because the new path was not on the path.**
//!
//! ```bash
//! cargo run --release --features metal --example lfm2-dspark-draft
//! cargo run --release --example lfm2-dspark-draft -- --cpu   # no metal
//! ```
//!
//! # What it checks, and what it deliberately does not
//!
//! It runs one full drafting round — tap, project, key/value the context,
//! embed the block with the *target's* table, run the five draft layers, take
//! the target's `lm_head` over the prediction slots, and walk the sequential
//! Markov pass — then reports the drafted block, its confidence scores, and a
//! digest. Five structural checks exit non-zero:
//!
//! 1. **the loaded parameter count matches the checkpoint's 327 707 521** — the
//!    config this build hardcodes describes the file it just loaded;
//! 2. **every drafted position is a real token id** — inside the target's
//!    vocabulary, so the block can be fed back for verification;
//! 3. **the block hidden states are finite and distinct per position** — γ
//!    identical rows would draft γ copies of one token and is invisible to any
//!    shape check;
//! 4. **the drafted block is not all one token** — the degenerate output a
//!    broken backbone produces while every shape stays right;
//! 5. **confidence scores are in (0, 1)** — a sigmoid that has been fed the
//!    wrong feature width would still produce numbers.
//!
//! **It does NOT claim an acceptance rate.** That is #363's own soundness
//! criterion and it needs the verify loop this harness deliberately does not
//! build: a drafted block is only *accepted* against a target forward pass over
//! the proposal, and wiring that is the next increment. Reporting a number here
//! would be a comparison against nothing (§3.1b).
//!
//! **And it takes no timing.** §7.1a: no `.bench/` row, no lease, no default
//! flipped. The tap is off unless a caller names layers, and this is a caller.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Context, Result};
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_examples::lfm2_setup::{parse_config, weight_files};
use clap::Parser;
use lloom::models::lfm2::{Model, Shape as Lfm2Shape, State};
use lloom::models::lfm2_dspark::{drafter_params, DSparkConfig, DSparkDrafter, CHECKPOINT_DTYPE};
use lloom::nn::VarBuilder;
use std::path::{Path, PathBuf};

/// The released drafter's own parameter total, summed from the payload-bearing
/// safetensors header.
const CHECKPOINT_PARAMS: usize = 327_707_521;

#[derive(Parser, Debug)]
#[command(about = "Run one DSpark drafting round on the real target and drafter")]
struct Args {
    /// The 2.6B target checkpoint — the target the drafter is trained against
    /// (§5.11, §10.2l). VL-3B has no DSpark draft and one does not transfer.
    #[arg(long)]
    target_dir: Option<PathBuf>,

    /// The DSpark drafter checkpoint.
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Run on the CPU even when built with `metal`.
    #[arg(long)]
    cpu: bool,

    /// Context tokens to run through the target before drafting.
    #[arg(long, default_value_t = 8)]
    tokens: usize,

    /// How many tokens to draft. Capped at the block's prediction slots
    /// (`block_size - logits_start` = 9 on this checkpoint).
    #[arg(long, default_value_t = 9)]
    draft: usize,

    /// Shift every tap index relative to `DSparkConfig::tap_layers()` — the
    /// layer-input capture at `target_layer_ids + 1`, i.e. those layers' output
    /// residual, which llama.cpp and mlx-dspark both feed (§10.2s). `-1` is the
    /// one-layer-early convention this tree shipped before §10.2s.
    #[arg(long, default_value_t = 0)]
    tap_shift: i64,

    /// Where the drafter keeps its context KV: `contiguous` or
    /// `paged[:<page_size>]` (#708, Metal only). The drafted ids must not move
    /// between the two — the page walk changes addresses, not arithmetic.
    #[arg(long, default_value = "contiguous")]
    ctx_kv: String,
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

/// `lfm2-determinism`'s SHA-256, included by path rather than copied — and it
/// carries the NIST self-test that runs below.
#[path = "../lfm2-determinism/sha256.rs"]
mod sha256;

fn sha256(bytes: &[u8]) -> String {
    sha256::digest_hex(bytes)
}

/// Read the drafter's constants from its own `config.json`.
///
/// Never defaulted. The two fields a wrong default would silently change —
/// `rope_theta` and `rope_is_neox_style` — alter no shape and no dispatch
/// count, and on a drafter there is not even a text gate to catch them, because
/// the target verifies every token (§10.2n).
fn read_drafter_config(dir: &Path) -> Result<DSparkConfig> {
    let raw = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let dfc = v.get("dflash_config").cloned().unwrap_or(v.clone());

    let usize_at = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .with_context(|| format!("drafter config has no {k}"))
    };

    let target_layer_ids: Vec<usize> = dfc
        .get("target_layer_ids")
        .and_then(|x| x.as_array())
        .context("drafter config has no dflash_config.target_layer_ids")?
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();

    Ok(DSparkConfig {
        hidden_size: usize_at("hidden_size")?,
        num_hidden_layers: usize_at("num_hidden_layers")?,
        num_attention_heads: usize_at("num_attention_heads")?,
        num_key_value_heads: usize_at("num_key_value_heads")?,
        head_dim: usize_at("head_dim")?,
        intermediate_size: usize_at("intermediate_size")?,
        vocab_size: usize_at("vocab_size")?,
        rms_norm_eps: v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5),
        rope_theta: v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .context("drafter config has no rope_theta")?,
        rope_is_neox_style: v
            .get("rope_is_neox_style")
            .and_then(|x| x.as_bool())
            .context("drafter config has no rope_is_neox_style")?,
        block_size: usize_at("block_size")?,
        mask_token_id: dfc
            .get("mask_token_id")
            .and_then(|x| x.as_u64())
            .context("drafter config has no dflash_config.mask_token_id")?
            as u32,
        num_target_layers: dfc
            .get("num_target_layers")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(30),
        target_layer_ids,
        markov_rank: v
            .get("markov_rank")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0),
        enable_confidence_head: v
            .get("enable_confidence_head")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        // NOT a key in config.json — a property of the packaging. This
        // checkpoint is anchor-as-pos0, so block_size 9 yields 9 predictions.
        // See DSparkConfig::logits_start for the measured cost of getting it
        // wrong (accept 1.35 against 3.42 on the sibling case).
        logits_start: 0,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let key = shape_key();
    println!("PROFILE {key}");
    lloom::profile::dispatch!(key, lloom::profile::LFM2, P => run::<P>(args))
}

fn run<P: Lfm2Shape>(args: Args) -> Result<()> {
    sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
    println!("sha256 self-test  : OK");

    let target_dir = args
        .target_dir
        .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B"))
        .context(
            "no LFM2.5-2.6B snapshot in the HF cache and no --target-dir. \
             The drafter is trained against THIS target (§10.2l).",
        )?;
    let draft_dir = args
        .draft_dir
        .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark"))
        .context(
            "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir. \
             §10.2n: 'characterised' and 'loadable' are different claims — the \
             design sections' constants came from a 6480-byte header blob. \
             Fetch model.safetensors (655 421 522 B).",
        )?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        candle_examples::device(false)?
    };

    println!("target : {}", target_dir.display());
    println!("draft  : {}", draft_dir.display());
    println!("device : {device:?}");

    // ---- the drafter's own constants, read rather than assumed --------------
    let dcfg = read_drafter_config(&draft_dir)?;
    println!("\ndrafter config (read from its own config.json)");
    println!(
        "  hidden / layers : {} / {}",
        dcfg.hidden_size, dcfg.num_hidden_layers
    );
    println!(
        "  heads / kv      : {} / {}  (GQA {}:1)",
        dcfg.num_attention_heads,
        dcfg.num_key_value_heads,
        dcfg.num_kv_groups()
    );
    println!("  head_dim        : {}", dcfg.head_dim);
    println!("  intermediate    : {}", dcfg.intermediate_size);
    println!("  rope_theta      : {:e}", dcfg.rope_theta);
    println!(
        "  rope style      : {} {}",
        if dcfg.rope_is_neox_style {
            "neox"
        } else {
            "INTERLEAVED"
        },
        if dcfg.rope_is_neox_style {
            "(rope)"
        } else {
            "(rope_i -- the target runs neox; DESIGN.md 10.2l item 3)"
        }
    );
    println!("  block_size      : {}", dcfg.block_size);
    println!(
        "  logits_start    : {}  (anchor-as-pos0)",
        dcfg.logits_start
    );
    println!("  max_draft       : {}", dcfg.max_draft());
    println!("  mask_token_id   : {}", dcfg.mask_token_id);
    println!("  markov_rank     : {}", dcfg.markov_rank);
    println!("  target_layer_ids: {:?}", dcfg.target_layer_ids);

    // Check 1: the config describes the file. Two independently derived
    // quantities -- one from config.json, one summed from the header.
    let derived = drafter_params(&dcfg);
    println!("\nparameter count");
    println!("  from config     : {derived}");
    println!("  from checkpoint : {CHECKPOINT_PARAMS}");
    if derived != CHECKPOINT_PARAMS {
        bail!(
            "FAIL: the config-derived parameter count {derived} does not match \
             the checkpoint's {CHECKPOINT_PARAMS} (delta {}). Either the config \
             is being read wrong or the layer structure is.",
            derived as i64 - CHECKPOINT_PARAMS as i64
        );
    }
    println!("  MATCH (delta 0)");

    // The layer-input tap at `target_layer_ids + 1` is those layers' output
    // residual -- the representation the drafter is trained on (§10.2s).
    let tap_layers: Vec<usize> = dcfg
        .tap_layers()
        .iter()
        .map(|&l| {
            let shifted = l as i64 + args.tap_shift;
            if shifted < 0 {
                bail!(
                    "--tap-shift {} puts tap layer {l} below zero",
                    args.tap_shift
                );
            }
            Ok(shifted as usize)
        })
        .collect::<Result<Vec<_>>>()?;
    if args.tap_shift != 0 {
        println!(
            "\nNOTE: --tap-shift {} moves the tap off the reference convention \
             (target_layer_ids + 1, DESIGN.md 10.2s); -1 is the one-layer-early arm.",
            args.tap_shift
        );
    }
    println!("  tap_layers      : {tap_layers:?}  (target_layer_ids + 1 + shift)");

    // ---- the target -------------------------------------------------------
    let raw = std::fs::read_to_string(target_dir.join("config.json"))?;
    let mut cfg = parse_config(&raw)?;
    if cfg.hidden_size != dcfg.hidden_size {
        bail!(
            "target hidden {} != drafter hidden {}: the projection's columns \
             could not be a whole number of target hidden states",
            cfg.hidden_size,
            dcfg.hidden_size
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
    let model = Model::<P>::new(&cfg, vb)?;
    let mut cache = State::<P>::new(true, DType::F16, &cfg, &device)?;
    println!("\ntarget loaded     : {} layers", cfg.num_hidden_layers);

    // ---- the drafter ------------------------------------------------------
    let draft_file = draft_dir.join("model.safetensors");
    if !draft_file.exists() {
        bail!(
            "{} does not exist. §10.2n: a committed header blob reads exactly \
             like the real thing in prose and has no payload.",
            draft_file.display()
        );
    }
    // The RoPE table must cover the largest absolute position the draft
    // attends: the context rows sit at their own offsets and the block sits
    // after them.
    let max_pos = args.tokens + dcfg.block_size + 64;
    // The checkpoint is stored at BF16 (CHECKPOINT_DTYPE) and is loaded here at
    // F16 -- the dtype the TARGET runs at, so the block embedding it produces
    // needs no cast, and the one the CPU backend can matmul at all. The CPU has
    // no BF16 matmul while Metal does, which is §14.1's constraint deciding a
    // dtype rather than a kernel.
    let compute_dtype = DType::F16;
    println!(
        "\ndrafter dtype     : stored {CHECKPOINT_DTYPE:?}, computing at {compute_dtype:?} \
         (CPU has no BF16 matmul; the target runs F16)"
    );
    let dvb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[draft_file], compute_dtype, &device)? };
    let drafter = DSparkDrafter::new(dcfg.clone(), max_pos, dvb)?;
    println!(
        "drafter loaded    : {} layers, {} params",
        dcfg.num_hidden_layers, derived
    );

    // ---- run the target, drain the taps ------------------------------------
    let ids: Vec<u32> = (1..=args.tokens as u32).collect();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0, &mut cache)?;
    let anchor = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;

    let taps = cache.take_taps();
    println!("\ntaps drained      : {}", taps.len());
    if taps.len() != dcfg.target_layer_ids.len() {
        bail!(
            "FAIL: the target produced {} taps and the projection consumes {}",
            taps.len(),
            dcfg.target_layer_ids.len()
        );
    }
    println!("anchor token      : {anchor}  (the target's own next-token argmax)");

    // ---- one drafting round ------------------------------------------------
    let ctx_kv = lloom::models::lfm2_dspark::CtxKv::parse(&args.ctx_kv)?;
    let mut caches = drafter
        .make_ctx_caches_for(ctx_kv, 1, &device)?
        .pop()
        .expect("one sequence");
    drafter.update_context(&taps, 0, &mut caches)?;
    println!(
        "context keyed     : {} positions per layer, {} layers, {} kv",
        caches[0].len(),
        caches.len(),
        ctx_kv.name()
    );

    let cap = args.draft.min(dcfg.max_draft());
    let width = dcfg.block_size;
    let block_ids = drafter.block_input_ids(anchor, width);
    println!(
        "block input       : [{}] then {} x mask({})",
        block_ids[0],
        width - 1,
        dcfg.mask_token_id
    );

    // The block is embedded with the TARGET's RAW table -- the drafter ships
    // no embed_tokens. Raw, not embedding_norm: see Model::draft_embed.
    let block_tensor = Tensor::new(block_ids.as_slice(), &device)?.unsqueeze(0)?;
    let block_embeds = drafter.embed_block(&model.draft_embed(&block_tensor)?)?;

    let block_offset = caches[0].len();
    let hidden = drafter.block_hidden(&block_embeds, block_offset, &caches)?;
    println!("block hidden      : {:?}", hidden.dims());

    // Check 3: finite, and distinct per position.
    let hv = hidden
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some(bad) = hv.iter().position(|x| !x.is_finite()) {
        bail!("FAIL: block hidden[{bad}] is not finite ({})", hv[bad]);
    }
    let mut row_digests = Vec::new();
    println!("\n  pos    mean        absmax     sha256[..16]");
    for k in 0..width {
        let row = hidden.i((0, k))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let absmax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
        let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = sha256(&bytes);
        println!("  {k:>3}  {mean:>10.6}  {absmax:>9.4}  {}", &d[..16]);
        row_digests.push(d);
    }
    let mut uniq = row_digests.clone();
    uniq.sort();
    uniq.dedup();
    if uniq.len() != row_digests.len() {
        bail!(
            "FAIL: only {} of {} block positions are distinct. Identical rows \
             draft identical tokens and are invisible to every shape check -- \
             a bidirectional block whose positions do not differ means the \
             position information never reached the backbone.",
            uniq.len(),
            row_digests.len()
        );
    }
    println!("  all {} positions distinct", row_digests.len());

    // ---- the prediction slots, through the TARGET's lm_head ----------------
    let pred = drafter.head_slice(&hidden, cap)?;
    let pred2 = pred.squeeze(0)?;
    let base_logits = model
        .draft_lm_head(&pred2.to_dtype(DType::F16)?)?
        .to_dtype(DType::F32)?;
    println!("\nbase logits       : {:?}", base_logits.dims());

    let anchor_t = Tensor::new(&[anchor], base_logits.device())?;
    let drafted = drafter
        .sample_block(&base_logits, &anchor_t)?
        .to_vec1::<u32>()?;
    println!("drafted block     : {drafted:?}");

    // Check 2: every drafted id is a real token.
    for (k, &t) in drafted.iter().enumerate() {
        if (t as usize) >= cfg.vocab_size {
            bail!(
                "FAIL: drafted token {t} at position {k} is outside the target's \
                 vocabulary ({}), so it could never be verified",
                cfg.vocab_size
            );
        }
    }
    // Check 4: not degenerate.
    let all_same = drafted.windows(2).all(|w| w[0] == w[1]);
    if drafted.len() > 1 && all_same {
        bail!(
            "FAIL: every drafted position is token {} -- the degenerate output a \
             broken backbone produces while every shape stays right",
            drafted[0]
        );
    }
    println!("  all ids in vocabulary, block is not degenerate");

    // ---- confidence --------------------------------------------------------
    let mut prev_tokens = Vec::with_capacity(cap);
    prev_tokens.push(anchor);
    for &t in drafted.iter().take(cap.saturating_sub(1)) {
        prev_tokens.push(t);
    }
    let prev_tokens = Tensor::new(prev_tokens.as_slice(), pred2.device())?;
    if let Some(conf) = drafter.confidence_scores(&pred2, &prev_tokens)? {
        let cv = conf.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        println!(
            "\nconfidence        : {:?}",
            cv.iter()
                .map(|x| (x * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        // Check 5: a sigmoid's range.
        for (k, &c) in cv.iter().enumerate() {
            if !(c.is_finite() && (0.0..=1.0).contains(&c)) {
                bail!("FAIL: confidence[{k}] = {c} is not a probability");
            }
        }
        let survival: Vec<f32> = cv
            .iter()
            .scan(1.0f32, |acc, &c| {
                *acc *= c;
                Some(*acc)
            })
            .collect();
        println!(
            "survival prefix   : {:?}",
            survival
                .iter()
                .map(|x| (x * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        println!(
            "  (paper Algorithm 1 sorts candidate prefixes by these products; \
             at B=1 it is the 'how many of the {cap} to submit' knob)"
        );
    }

    let dbytes: Vec<u8> = drafted.iter().flat_map(|t| t.to_le_bytes()).collect();
    println!(
        "\nRESULT ctx_tokens={} anchor={} cap={} tap_shift={} ctx_kv={} drafted={:?} sha256={}",
        args.tokens,
        anchor,
        cap,
        args.tap_shift,
        ctx_kv.name(),
        drafted,
        sha256(&dbytes)
    );
    println!(
        "\nNOTE: this is a drafting round, NOT an acceptance rate. #363's \
         soundness criterion is the accept rate against the published 4.42 \
         mean, which needs the verify loop over a target forward pass on the \
         proposal -- the next increment. A digest here is a reproducibility \
         anchor (§3.1b: a reference from the same run is not a comparison)."
    );
    Ok(())
}

/// The profile this harness runs at: the config's default axes name the
/// one-sequence, no-rewind shape (`profiles-and-memory-design.md` §3), at
/// the contract in force.
fn shape_key() -> lloom::profile::ProfileKey {
    lloom::profile::ProfileKey::at_contract(
        <lloom::profile::NoRewind as lloom::profile::Rewind>::NAME,
        <lloom::profile::One as lloom::profile::Rows>::NAME,
    )
}
