//! The DSpark **verify loop** — the acceptance rate, on the real target and the
//! real drafter. `DESIGN.md` §10.2q, issue #363.
//!
//! # What this is, and why it is the increment the issue was waiting for
//!
//! §10.2p built the drafter and closed with the reason #363 stays open:
//!
//! > *"The accept rate needs the **verify loop**: a target forward pass over the
//! > proposed block, `accept()` taking the longest matching prefix, and the
//! > conv/KV rollback on rejection."*
//!
//! That is this harness. Every piece it composes already existed and **none of
//! them had ever run together**: `Model::verify_step` (`lfm2.rs:3199`) had
//! **zero callers** — §10.2l grepped it with a positive control — and the
//! drafter had run exactly one round against a fixed context (§10.2p). A loop is
//! what turns *"a drafted block"* into *"an acceptance rate", and the accept
//! rate is #363's own soundness criterion, because the identity gate is blind to
//! the drafter by construction (§10.2n: the target corrects every proposal, so
//! **any** drafter yields the byte-identical stream).
//!
//! ```bash
//! cargo run --release --features metal --example lfm2-dspark-verify -- \
//!     --prompt-file design/workloads/prompts/short-factual.txt --rounds 24
//! ```
//!
//! # The three things it reports that nothing else in the tree can
//!
//! 1. **The acceptance rate** — accepted tokens per verify pass, against the
//!    published **4.42 mean** (M4 Max). This is the number §10.2m's soundness
//!    gate names.
//! 2. **Output identity against non-speculative decode** — the same prompt
//!    decoded both ways must produce the **byte-identical** token stream. Under
//!    greedy verification that is a theorem, not a hope (§10.2j), so **a moved
//!    digest here is a defect rather than a tradeoff**. This is the gate that
//!    proves the *verifier* while the accept rate probes the *drafter*.
//! 3. **The ring's behaviour with its real consumer** (#535, folded in). #535
//!    measured the token ring against a blocking flush and found a null,
//!    concluding the ring's right consumer is the speculation loop — where
//!    verify→accept publishes a *variable* number of tokens per round rather
//!    than one. This is that consumer, and §5 of the report is what it does.
//!
//! # The prompt is selected, not defaulted (#549, folded in)
//!
//! **The default harness prompt is broken and measuring an accept rate on it
//! would measure nonsense.** #549 measured the tree's dominant prompt — 64.4 %
//! of all committed `RESULT` lines — producing **40 distinct ids out of 120**
//! with a **49-token verbatim repeated block**, on a text-only question it
//! answers by describing an image that does not exist. An accept rate on a
//! degeneration loop is a measurement of the loop: a drafter trivially predicts
//! the next token of a block it has already seen repeated, so the number would
//! be high and meaningless.
//!
//! So this harness has **no default prompt**. `--prompt-file` or `--prompt` is
//! required, `design/workloads/prompts/` is the corpus #549 produced for exactly
//! this, and the run **reports the degeneracy of its own output stream** so the
//! reader can see whether the prompt held up rather than trusting that it did.
//!
//! # What it does not do
//!
//! **No timing, no `.bench/` row, no lease, no default flipped** (§7.1a, §2.6).
//! An acceptance rate is a property of the model pair, not of the machine — it
//! is a count of matching tokens, identical on Metal and CPU, and it needs no
//! GPU exclusivity to be correct. The GPU-busy-per-accepted-token figure §10.2j
//! measures with an *oracle* proposer is the follow-on that wants a lease, and
//! it is deliberately not taken here: this run establishes the accept rate that
//! every such figure needs as its input, which #363 says has never been measured
//! (*"every acceptance rate there is imposed rather than earned"*).

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor, D};
use candle_examples::lfm2_setup::{parse_config, weight_files};
use candle_nn::VarBuilder;
use candle_transformers::models::lfm2::{Cache, ConvState, KvAppend, Model};
use candle_transformers::models::lfm2_dspark::{
    drafter_params, DSparkConfig, DSparkDrafter, CHECKPOINT_DTYPE,
};
use clap::Parser;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// The released drafter's own parameter total, summed from the payload-bearing
/// safetensors header (§10.2p, delta 0 against the config-derived count).
const CHECKPOINT_PARAMS: usize = 327_707_521;

/// The published mean accepted-tokens-per-step for this drafter, M4 Max.
///
/// §10.2m names it as the soundness gate's reference. It is a **different
/// machine and an unstated prompt**, so it is a target to be compared against
/// with those caveats stated, never a threshold to pass — see the report.
const PUBLISHED_ACCEPT_MEAN: f64 = 4.42;

#[derive(Parser, Debug)]
#[command(about = "The DSpark verify loop: acceptance rate and output identity")]
struct Args {
    /// The 2.6B target checkpoint — the target this drafter is trained against.
    #[arg(long)]
    target_dir: Option<PathBuf>,

    /// The DSpark drafter checkpoint.
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Run on the CPU even when built with `metal`.
    #[arg(long)]
    cpu: bool,

    /// The prompt, as a file. `design/workloads/prompts/*.txt` is the corpus.
    ///
    /// **There is deliberately no default** — see the module docs. #549 measured
    /// the tree's dominant prompt as a hallucination loop, and an acceptance
    /// rate over a degenerate stream measures the degeneracy.
    #[arg(long)]
    prompt_file: Option<PathBuf>,

    /// The prompt, inline. Mutually exclusive with `--prompt-file`.
    #[arg(long)]
    prompt: Option<String>,

    /// A label for the prompt, recorded on the RESULT line.
    ///
    /// `manifest.json` carries a `sha256` per prompt and the `RESULT` line
    /// carries only a *length*, so two different 34-token prompts are
    /// indistinguishable in the record (`workloads/README.md`). Defaults to the
    /// prompt file's stem.
    #[arg(long)]
    prompt_id: Option<String>,

    /// Speculation rounds to run. Each round drafts a block and verifies it.
    #[arg(long, default_value_t = 24)]
    rounds: usize,

    /// Draft block size, capped at the checkpoint's prediction slots (9).
    ///
    /// This is the verifier's `K` **and** the conv ring's `K` — `DESIGN.md`
    /// §16 6b, enforced by `Cache::advance`, which refuses a window wider than
    /// the ring's reserved history.
    #[arg(long, short = 'k', default_value_t = 8)]
    draft: usize,

    /// Skip the non-speculative identity arm.
    ///
    /// Off by default because the identity gate is the half that proves the
    /// verifier, and it is cheap. Present so a long sweep can skip the second
    /// decode.
    #[arg(long)]
    no_identity: bool,

    /// Shift every tap index — `0` is llama.cpp's layer-input convention and
    /// ours, `1` reproduces mlx-dspark's after-the-block capture (§10.2o).
    #[arg(long, default_value_t = 0)]
    tap_shift: usize,

    /// How much of the prefill to key into the drafter's context cache.
    ///
    /// **This is an open convention and the arm exists because it is.** The
    /// drafter attends `H_ctx` as its K/V, and neither §10.2m nor the code
    /// pins how many prompt positions that should be. `last` keys the final
    /// position only (the one the anchor follows from); `all` keys every prompt
    /// position, which is what `lfm2-dspark-draft` does by passing the whole
    /// tap through. They give different acceptance rates and the run reports
    /// which it used.
    #[arg(long, default_value = "all")]
    prefill_ctx: String,
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

fn token_digest(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    sha256(&bytes)
}

/// Read the drafter's constants from its own `config.json`.
///
/// Never defaulted, for `lfm2-dspark-draft`'s reason: `rope_theta` and
/// `rope_is_neox_style` alter no shape and no dispatch count, and on a drafter
/// there is not even a text gate to catch them, because the target verifies
/// every token (§10.2n).
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
        // NOT a key in config.json — a property of the packaging (§10.2p).
        logits_start: 0,
    })
}

/// What one speculation round did.
///
/// Recorded per round rather than accumulated, because a mean over rounds hides
/// the shape and `CONTRIBUTING.md` §3.2a rule 3 is explicit that distribution
/// shape matters more than centre. An accept rate that is 8,1,8,1 is a different
/// mechanism from one that is 4,5,4,5 with the same mean.
#[derive(Debug, Clone)]
struct Round {
    /// Tokens the drafter proposed this round.
    proposed: Vec<u32>,
    /// Tokens `verify_step` returned — the accepted prefix plus the target's own
    /// correction, so this is never empty.
    accepted: Vec<u32>,
    /// How many of `proposed` the target agreed with, before the correction.
    matched: usize,
}

impl Round {
    /// The **advance**: tokens the stream moved forward this round.
    ///
    /// This is `accepted.len()`, and it is the quantity the acceptance rate is
    /// about — `verify_step`'s docs call the last one the bonus token, free
    /// because the final verified position predicts it. It is never zero: the
    /// target's own prediction at position 0 is correct by definition.
    fn advance(&self) -> usize {
        self.accepted.len()
    }
}

/// The degeneracy of a token stream, for the #549 check.
///
/// Reported rather than asserted for the most part: a low-entropy prompt
/// (`low-entropy-list` counts to forty) legitimately has a low type/token ratio,
/// so a threshold here would fail a good prompt. What IS asserted is the
/// verbatim-repeated-block test, because that is the shape #549 measured — a
/// 49-token block repeated verbatim is a loop rather than a low-entropy answer.
struct Degeneracy {
    distinct: usize,
    total: usize,
    longest_repeat: usize,
}

fn degeneracy(ids: &[u32]) -> Degeneracy {
    let mut uniq = ids.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    // The longest block that appears at least twice, non-overlapping. O(n^2)
    // over a few hundred tokens, run once, off any timed path.
    let mut longest = 0usize;
    let n = ids.len();
    for w in (1..=n / 2).rev() {
        if w <= longest {
            break;
        }
        let mut found = false;
        'outer: for i in 0..=n.saturating_sub(2 * w) {
            for j in (i + w)..=n.saturating_sub(w) {
                if ids[i..i + w] == ids[j..j + w] {
                    found = true;
                    break 'outer;
                }
            }
        }
        if found {
            longest = w;
            break;
        }
    }
    Degeneracy {
        distinct: uniq.len(),
        total: n,
        longest_repeat: longest,
    }
}

#[allow(clippy::too_many_arguments)]
fn main() -> Result<()> {
    let args = Args::parse();

    sha256::self_test().map_err(|e| anyhow::anyhow!("sha256 self-test failed: {e}"))?;
    println!("sha256 self-test  : OK");

    // ---- the prompt: selected, never defaulted (#549) ----------------------
    let (prompt_text, prompt_id) = match (&args.prompt_file, &args.prompt) {
        (Some(_), Some(_)) => bail!("--prompt-file and --prompt are mutually exclusive"),
        (Some(p), None) => {
            let t = std::fs::read_to_string(p)
                .with_context(|| format!("reading prompt file {}", p.display()))?;
            let id = args.prompt_id.clone().unwrap_or_else(|| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unnamed".into())
            });
            (t.trim().to_string(), id)
        }
        (None, Some(t)) => (
            t.clone(),
            args.prompt_id.clone().unwrap_or_else(|| "inline".into()),
        ),
        (None, None) => bail!(
            "no prompt. This harness has NO DEFAULT prompt deliberately (issue #549): \
             the tree's dominant prompt -- 64.4 % of committed RESULT lines -- decodes to \
             40 distinct ids of 120 with a 49-token verbatim repeated block, and an \
             acceptance rate over a degeneration loop measures the loop rather than the \
             drafter (a drafter trivially predicts the next token of a block it has already \
             seen repeated). Pass --prompt-file design/workloads/prompts/<id>.txt -- that \
             corpus exists for this -- or --prompt with your own."
        ),
    };
    if prompt_text.is_empty() {
        bail!("the prompt is empty");
    }

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
        .context("no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir")?;

    let device = if args.cpu {
        Device::Cpu
    } else {
        candle_examples::device(false)?
    };

    println!("target      : {}", target_dir.display());
    println!("draft       : {}", draft_dir.display());
    println!("device      : {device:?}");
    println!("prompt id   : {prompt_id}");
    println!("prompt sha  : {}", sha256(prompt_text.as_bytes()));
    println!("prompt      : {prompt_text:?}");

    // ---- the drafter's constants ------------------------------------------
    let mut dcfg = read_drafter_config(&draft_dir)?;
    let derived = drafter_params(&dcfg);
    if derived != CHECKPOINT_PARAMS {
        bail!(
            "FAIL: config-derived parameter count {derived} != checkpoint's \
             {CHECKPOINT_PARAMS} (delta {})",
            derived as i64 - CHECKPOINT_PARAMS as i64
        );
    }
    println!("\ndrafter params    : {derived} (matches checkpoint, delta 0)");

    let mut tap_layers = dcfg.target_layer_ids.clone();
    if args.tap_shift != 0 {
        println!(
            "NOTE: --tap-shift {} reproduces mlx-dspark's after-the-block capture \
             (§10.2o; the two witnesses disagree by one layer)",
            args.tap_shift
        );
        tap_layers = tap_layers.iter().map(|l| l + args.tap_shift).collect();
    }

    // ---- the target -------------------------------------------------------
    let raw = std::fs::read_to_string(target_dir.join("config.json"))?;
    let mut cfg = parse_config(&raw)?;
    if cfg.hidden_size != dcfg.hidden_size {
        bail!(
            "target hidden {} != drafter hidden {}",
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

    let k = args.draft.min(dcfg.max_draft());
    if k == 0 {
        bail!("--draft must be at least 1");
    }

    // **The ring's K and the verifier's K are the same number** (`DESIGN.md`
    // §16 6b), enforced by `Cache::advance`, which refuses a window wider than
    // the reserved history rather than letting a speculative write overwrite the
    // live window. Set from `k` rather than hardcoded so the two cannot drift.
    cfg.conv_state = ConvState::RotatingRing { k };
    // Speculation requires the in-place arm: the `Cat` arm has no length to
    // rewind, and `Cache::resolve` refuses it (`lfm2.rs:1546`) rather than
    // silently keeping discarded positions inside the concatenated allocation.
    cfg.kv_append = KvAppend::InPlace;

    let tokenizer_path = target_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    // The chat template, the same one `lfm2-smoke` and `lfm2-decode-profile`
    // apply. Fed a bare string LFM2 emits EOS at the first position
    // (`lfm2-smoke --raw-prompt` records it), so an untemplated prompt would
    // make the accept rate a property of a one-token stream.
    let templated = format!("<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_ids: Vec<u32> = tokenizer
        .encode(templated.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to nothing");
    }
    println!(
        "prompt tokens     : {} (wrapped in the chat template)",
        prompt_ids.len()
    );

    println!("target config     : {} layers", cfg.num_hidden_layers);

    // ---- the drafter ------------------------------------------------------
    let draft_file = draft_dir.join("model.safetensors");
    if !draft_file.exists() {
        bail!("{} does not exist", draft_file.display());
    }
    let max_pos = prompt_ids.len() + args.rounds * k + dcfg.block_size + 64;
    let compute_dtype = DType::F16;
    println!("drafter dtype     : stored {CHECKPOINT_DTYPE:?}, computing at {compute_dtype:?}");
    let dvb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[draft_file], compute_dtype, &device)? };
    dcfg.target_layer_ids = tap_layers.clone();
    let drafter = DSparkDrafter::new(dcfg.clone(), max_pos, dvb)?;
    println!("drafter loaded    : {} layers", dcfg.num_hidden_layers);
    println!("draft block K     : {k}  (conv ring reserves {k} history slots)");

    // =====================================================================
    // ARM 1 — the speculative loop
    // =====================================================================
    println!("\n=== ARM 1: speculative decode (drafter -> verify_step) ===");

    // The tap is ON for this arm, because the drafter consumes it. It is off by
    // default and this is a caller (§7.1a).
    let mut spec_cfg = cfg.clone();
    spec_cfg.tap_layers = tap_layers.clone();
    spec_cfg.tap_dtype = DType::F32; // §10.2m: deliberate, buys acceptance rate
    let spec_model = {
        let files = weight_files(&target_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F16, &device)? };
        Model::new(&spec_cfg, vb)?
    };
    let mut cache = Cache::new(true, DType::F16, &spec_cfg, &device)?;

    // Prefill. The taps this pass leaves describe the whole prompt; the drafter
    // keys the LAST position, which is the one the anchor follows from.
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let logits = spec_model.forward(&input, 0, &mut cache)?;
    let anchor = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
    let prefill_taps = cache.take_taps();
    if prefill_taps.len() != tap_layers.len() {
        bail!(
            "FAIL: the target produced {} taps and the projection consumes {}",
            prefill_taps.len(),
            tap_layers.len()
        );
    }

    let mut caches = drafter.make_ctx_caches();
    // How much of the prompt conditions the first draft. See `--prefill-ctx`:
    // the convention is not pinned by §10.2m or by the code, so it is an arm
    // rather than a silent choice (`CONTRIBUTING.md` §6).
    let seed_taps: Vec<Tensor> = match args.prefill_ctx.as_str() {
        "all" => prefill_taps.clone(),
        "last" => prefill_taps
            .iter()
            .map(|t| {
                let seq = t.dim(1)?;
                t.narrow(1, seq - 1, 1)
            })
            .collect::<candle::Result<Vec<_>>>()?,
        other => bail!("--prefill-ctx must be `all` or `last`, got {other:?}"),
    };
    drafter.update_context(&seed_taps, 0, &mut caches)?;
    let mut ctx_positions = caches[0].len();

    println!(
        "prefill done      : anchor={anchor}, drafter context {ctx_positions} position(s) \
         (--prefill-ctx {})",
        args.prefill_ctx
    );

    let mut stream: Vec<u32> = Vec::new();
    let mut rounds: Vec<Round> = Vec::new();
    let mut last = anchor;
    stream.push(anchor);
    let mut index_pos = prompt_ids.len();

    println!("\n rnd  proposed                              matched  advance  stream");
    for r in 0..args.rounds {
        // ---- draft ------------------------------------------------------
        let width = dcfg.block_size;
        let block_ids = drafter.block_input_ids(last, width);
        let block_tensor = Tensor::new(block_ids.as_slice(), &device)?.unsqueeze(0)?;
        let block_embeds = drafter.embed_block(&spec_model.draft_embed(&block_tensor)?)?;
        let hidden = drafter.block_hidden(&block_embeds, ctx_positions, &caches)?;
        let pred = drafter.head_slice(&hidden, k)?.squeeze(0)?;
        let base_logits = spec_model
            .draft_lm_head(&pred.to_dtype(DType::F16)?)?
            .to_dtype(DType::F32)?;
        let proposed = drafter.sample_block(&base_logits, last)?;

        // ---- verify -----------------------------------------------------
        // ONE target forward pass over the whole proposed block, longest
        // matching prefix accepted, KV rewound and conv phase rolled for the
        // rest. This is `lfm2.rs:3199` and this loop is its first caller.
        let accepted = spec_model.verify_step(last, &proposed, index_pos, &mut cache)?;

        // How many proposals the target agreed with, which is `advance - 1`
        // when the last token is the free bonus and `advance` when a rejection
        // ended the prefix early. Derived by comparison rather than assumed, so
        // it stays right if `verify_step`'s contract changes.
        let matched = accepted
            .iter()
            .zip(proposed.iter())
            .take_while(|(a, p)| a == p)
            .count();

        index_pos += accepted.len();
        stream.extend_from_slice(&accepted);
        last = *accepted.last().expect("verify_step never returns empty");

        // ---- re-key the drafter from the ACCEPTED rows only ---------------
        // §10.2m: *"we take the tap only from accepted rows from the start"*.
        // The verify pass tapped all `k` positions; the rejected ones describe
        // state the target has just rolled back, so keying them would condition
        // the next draft on a future that did not happen.
        let pass_taps = cache.take_taps();
        if pass_taps.len() != tap_layers.len() {
            bail!(
                "FAIL: verify pass produced {} taps, expected {}",
                pass_taps.len(),
                tap_layers.len()
            );
        }
        let n_acc = accepted.len();
        let kept: Vec<Tensor> = pass_taps
            .iter()
            .map(|t| {
                let seq = t.dim(1)?;
                // The pass ran `k` positions; keep the first `n_acc`, which are
                // the accepted prefix. `min` guards the case where the pass was
                // shorter than the acceptance (it cannot be, and an assertion
                // that silently truncates is worse than one that is exact).
                t.narrow(1, 0, n_acc.min(seq))
            })
            .collect::<candle::Result<Vec<_>>>()?;
        drafter.update_context(&kept, ctx_positions, &mut caches)?;
        ctx_positions = caches[0].len();

        let prop_s = format!("{proposed:?}");
        println!(
            " {r:>3}  {:<36}  {matched:>7}  {:>7}  {:>6}",
            if prop_s.len() > 36 {
                format!("{}…", &prop_s[..35])
            } else {
                prop_s
            },
            accepted.len(),
            stream.len()
        );

        rounds.push(Round {
            proposed,
            accepted,
            matched,
        });
    }

    // =====================================================================
    // ARM 2 — non-speculative decode, the identity gate
    // =====================================================================
    let identity = if args.no_identity {
        None
    } else {
        println!("\n=== ARM 2: non-speculative decode (the identity gate) ===");
        // A SEPARATE model and cache with the tap OFF — this arm must be the
        // ordinary decode path, not the speculative one with speculation
        // disabled, or the comparison is against itself (§3.1b: a reference from
        // the same mechanism "looks like a comparison and is not").
        let plain_model = {
            let files = weight_files(&target_dir)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F16, &device)? };
            Model::new(&cfg, vb)?
        };
        let mut plain_cache = Cache::new(true, DType::F16, &cfg, &device)?;
        let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
        let logits = plain_model.forward(&input, 0, &mut plain_cache)?;
        let mut tok = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let mut plain: Vec<u32> = vec![tok];
        let mut pos = prompt_ids.len();
        while plain.len() < stream.len() {
            let inp = Tensor::new(&[tok], &device)?.unsqueeze(0)?;
            let l = plain_model.forward(&inp, pos, &mut plain_cache)?;
            tok = l.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            plain.push(tok);
            pos += 1;
        }
        plain.truncate(stream.len());
        println!("decoded {} tokens one at a time", plain.len());
        Some(plain)
    };

    // =====================================================================
    // The report
    // =====================================================================
    let n_rounds = rounds.len() as f64;
    let total_advance: usize = rounds.iter().map(|r| r.advance()).sum();
    let total_matched: usize = rounds.iter().map(|r| r.matched).sum();
    let mean_advance = total_advance as f64 / n_rounds;
    let mean_matched = total_matched as f64 / n_rounds;

    let mut adv: Vec<usize> = rounds.iter().map(|r| r.advance()).collect();
    adv.sort_unstable();
    let median = adv[adv.len() / 2];
    let full_blocks = rounds.iter().filter(|r| r.advance() == k).count();
    let single = rounds.iter().filter(|r| r.advance() == 1).count();

    println!("\n=== 1. ACCEPTANCE RATE ===");
    println!("rounds                     : {}", rounds.len());
    println!("K (proposals per round)    : {k}");
    println!("tokens advanced            : {total_advance}");
    println!("mean advance per round     : {mean_advance:.3}");
    println!("median advance             : {median}");
    println!(
        "min / max advance          : {} / {}",
        adv[0],
        adv[adv.len() - 1]
    );
    println!("mean matched proposals     : {mean_matched:.3}  (of {k})");
    println!(
        "acceptance fraction        : {:.4}  (matched / proposed)",
        total_matched as f64 / (rounds.len() * k) as f64
    );
    println!(
        "full-block rounds          : {full_blocks} / {}",
        rounds.len()
    );
    println!("single-token rounds        : {single} / {}", rounds.len());
    println!("\npublished reference        : {PUBLISHED_ACCEPT_MEAN} mean accepted/step (M4 Max)");
    println!(
        "ours                       : {mean_advance:.3}  (ratio {:.3})",
        mean_advance / PUBLISHED_ACCEPT_MEAN
    );
    println!(
        "  NOT a pass/fail: a different machine, an unstated prompt and an unstated K.\n  \
         The comparable claim is the shape, and the caveats are in the artifact."
    );

    // Per-round advance, because a mean hides the shape (§3.2a rule 3).
    println!("\nadvance per round (the distribution the mean summarises):");
    let hist: Vec<String> = (1..=k)
        .map(|n| {
            let c = rounds.iter().filter(|r| r.advance() == n).count();
            format!("{n}:{c}")
        })
        .collect();
    println!("  {}", hist.join("  "));

    println!("\n=== 2. OUTPUT IDENTITY (the verifier's gate) ===");
    let spec_digest = token_digest(&stream);
    println!(
        "speculative stream  : {} tokens, sha256 {}",
        stream.len(),
        spec_digest
    );
    match &identity {
        None => println!("non-speculative     : SKIPPED (--no-identity)"),
        Some(plain) => {
            let plain_digest = token_digest(plain);
            println!(
                "non-speculative     : {} tokens, sha256 {}",
                plain.len(),
                plain_digest
            );
            if plain != &stream {
                let first = stream
                    .iter()
                    .zip(plain.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(stream.len().min(plain.len()));
                bail!(
                    "FAIL: the speculative and non-speculative streams differ at position \
                     {first} (spec {:?} vs plain {:?}). Under greedy verification a \
                     speculative scheme is output-identical BY CONSTRUCTION (§10.2j), so \
                     this is a defect in the verifier and not a tradeoff.",
                    stream.get(first),
                    plain.get(first)
                );
            }
            println!(
                "IDENTICAL           : byte-for-byte, {} tokens",
                stream.len()
            );
            println!(
                "  This proves the VERIFIER. It says nothing about the drafter -- any \n  \
                 drafter yields this stream, which is why the accept rate above is the \n  \
                 drafter's only soundness signal (§10.2n)."
            );
        }
    }

    println!("\n=== 3. THE STREAM, and whether the prompt held up (#549) ===");
    let d = degeneracy(&stream);
    let text = tokenizer
        .decode(&stream, true)
        .map_err(|e| anyhow::anyhow!("decoding: {e}"))?;
    println!("distinct ids        : {} of {}", d.distinct, d.total);
    println!(
        "type/token ratio    : {:.4}",
        d.distinct as f64 / d.total.max(1) as f64
    );
    println!("longest repeated    : {} tokens", d.longest_repeat);
    println!("text                : {text:?}");
    // #549's shape, asserted: a long verbatim repeated block is a loop rather
    // than a low-entropy answer. Thresholded well above `low-entropy-list`'s
    // legitimate repetition (counting to forty repeats short runs, not blocks).
    let repeat_limit = (d.total / 4).max(8);
    if d.longest_repeat >= repeat_limit {
        bail!(
            "FAIL: the output stream repeats a {}-token block verbatim in {} tokens \
             (limit {repeat_limit}). This is #549's degeneration shape, and an acceptance \
             rate over it measures the loop rather than the drafter -- a drafter trivially \
             predicts the next token of a block it has already seen. Pick another prompt \
             from design/workloads/prompts/.",
            d.longest_repeat,
            d.total
        );
    }
    println!("  no degenerate repeated block (limit {repeat_limit})");

    println!("\n=== 4. WHAT THE DRAFTER PROPOSED, round by round ===");
    println!(" rnd  matched  advance  first rejected proposal");
    for (i, r) in rounds.iter().enumerate().take(12) {
        let rej = if r.matched < r.proposed.len() {
            format!(
                "proposed {}, target said {}",
                r.proposed[r.matched],
                r.accepted.get(r.matched).copied().unwrap_or(0)
            )
        } else {
            "none -- full block".to_string()
        };
        println!(" {i:>3}  {:>7}  {:>7}  {rej}", r.matched, r.advance());
    }
    if rounds.len() > 12 {
        println!("  … {} more rounds", rounds.len() - 12);
    }

    println!("\n=== 5. THE RING'S CONSUMER (#535, folded in) ===");
    // #535's finding is that the ring was measured against the WRONG consumer:
    // a blocking flush publishing exactly one token per step, where the ring's
    // whole shape is a VARIABLE number of tokens becoming available at once.
    // This loop is the right consumer, and what it produces is that variability,
    // which is a property of the loop rather than a timing measurement.
    let publishes: Vec<usize> = rounds.iter().map(|r| r.advance()).collect();
    let distinct_widths: usize = {
        let mut w = publishes.clone();
        w.sort_unstable();
        w.dedup();
        w.len()
    };
    println!("publish events      : {} (one per verify)", publishes.len());
    println!(
        "tokens per publish  : min {} max {} mean {mean_advance:.3}",
        adv[0],
        adv[adv.len() - 1]
    );
    println!("distinct widths     : {distinct_widths}");
    println!(
        "tokens per publish, in order: {:?}",
        publishes.iter().take(24).collect::<Vec<_>>()
    );
    println!(
        "\n#535 measured the ring against a one-token-per-step consumer through a\n\
         blocking flush and found a null. This loop is the consumer it named:\n\
         a verify publishes {} to {} tokens at once, {distinct_widths} distinct widths over\n\
         {} rounds. That variability is the ring's reason to exist and it is\n\
         present here -- but NO TIMING IS TAKEN in this run (§7.1a), so this is\n\
         the consumer's SHAPE, not the ring's A/B. The A/B needs a lease and a\n\
         ring arm to compare against, and it is named in the PR as not done.",
        adv[0],
        adv[adv.len() - 1],
        rounds.len()
    );

    println!(
        "\nRESULT prompt_id={prompt_id} prompt_sha={} k={k} rounds={} \
         mean_advance={mean_advance:.4} median_advance={median} \
         mean_matched={mean_matched:.4} accept_fraction={:.4} \
         tokens={} tap_shift={} stream_sha256={spec_digest} identity={}",
        &sha256(prompt_text.as_bytes())[..16],
        rounds.len(),
        total_matched as f64 / (rounds.len() * k) as f64,
        stream.len(),
        args.tap_shift,
        match &identity {
            None => "skipped",
            Some(_) => "IDENTICAL",
        }
    );

    Ok(())
}
