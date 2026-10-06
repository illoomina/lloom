//! MTP over the STREAMED tier on a pool too tight to hold a wrong draft's
//! experts beside the real stream's (`expert-streaming.md` §20.16): the
//! verify window's plans are settled AFTER the accept, so
//!
//! - the stream is greedy's, byte for byte, with every drafter (the real
//!   `MtpDrafter`, an oracle, a garbage drafter), at every K;
//! - a rejected draft loads nothing, touches nothing and counts no usage:
//!   under a drafter whose every draft is wrong the tier's loads, evictions
//!   and usage profile equal the greedy stream's own — the absolute
//!   reference (CONTRIBUTING §3.1b), taken from a second run of the same
//!   tier over the same tokens;
//! - a draft that WOULD have been accepted but computed on a missing expert
//!   cuts the accepted prefix at itself instead of redoing the window.
//!
//! CPU: the question is the settle's bookkeeping; the CPU backend is
//! bit-stable. The window runs the reference path (position by position),
//! whose per-position plans join into one record per layer.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Shape as _;
use lloom::models::qwen4exp::{Config, MtpDrafter, Qwen4Exp, State, Streamed};
use lloom::nn::expert_stream::{ExpertSource, LoaderArm, StreamStats};
use lloom::nn::model::CausalLM;
use lloom::nn::speculator::{Draft, KvSource, NextnHidden, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{CrownExact, RewindableExact};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::{GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

fn temp_gguf(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lloom-mtp-streamed-{}-{tag}.gguf",
        std::process::id()
    ))
}

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 12;
    cfg.n_experts_per_tok = 3;
    cfg.eos_token_id = 127;
    cfg
}

/// Every layer's expert stacks to one GGUF, as the loader names them.
fn write_checkpoint(cfg: &Config, path: &Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut owned: Vec<(String, QTensor)> = Vec::new();
    for l in 0..cfg.n_layers {
        let [g, u, d] = testing::expert_stacks(cfg, l, &dev, GgmlDType::F32);
        owned.push((format!("blk.{l}.ffn_gate_exps.weight"), g));
        owned.push((format!("blk.{l}.ffn_up_exps.weight"), u));
        owned.push((format!("blk.{l}.ffn_down_exps.weight"), d));
    }
    let refs: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut File::create(path)?, &[], &refs)?;
    Ok(())
}

fn streamed_model(
    cfg: &Config,
    path: &Path,
    slots: usize,
) -> Result<Qwen4Exp<RewindableExact, Streamed>> {
    let mut f = File::open(path)?;
    let content = Content::read(&mut f)?;
    let dev = Device::Cpu;
    testing::model_with::<RewindableExact, _>(cfg, &dev, |l| {
        let src = |n: &str| ExpertSource::from_gguf(&content, path, &format!("blk.{l}.{n}"));
        Streamed::new(
            src("ffn_gate_exps.weight")?,
            src("ffn_up_exps.weight")?,
            src("ffn_down_exps.weight")?,
            slots,
            LoaderArm::Pread,
            &dev,
        )
    })
}

/// The tier's counters and usage profile, summed over the layers.
fn tier_state(model: &Qwen4Exp<RewindableExact, Streamed>) -> (StreamStats, Vec<Vec<u32>>) {
    let mut stats = StreamStats::default();
    for l in model.layers() {
        stats += l.moe().experts().stats();
    }
    (stats, model.stream_usage())
}

fn greedy(
    model: &Qwen4Exp<RewindableExact, Streamed>,
    cfg: &Config,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool)> {
    let dev = Device::Cpu;
    let cache = RewindableExact::single(&model, &dev, 8)?;
    let mut session = Session::new(model, cache);
    let mut spec = IdentityProposer;
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(max);
    let out = session.decode(&mut spec, &mut rb, &mut sampler, prompt, &opts, &mut |_| {
        Ok(())
    })?;
    Ok((out.ids, out.eos))
}

/// The prompt as one chunk, then `tokens` one by one through the decode
/// step on a fresh tier: the tier's state after exactly those positions.
fn decode_positions(
    cfg: &Config,
    path: &Path,
    slots: usize,
    prompt: &[u32],
    tokens: &[u32],
) -> Result<Qwen4Exp<RewindableExact, Streamed>> {
    let dev = Device::Cpu;
    let mut model = streamed_model(cfg, path, slots)?;
    // On the host plan: a device-planned decode step that misses counts
    // its replayed token's experts twice (once at the settle, once at the
    // replay's loads) — the reference is one use per position.
    model.set_device_plan(false);
    let mut cache = RewindableExact::single(&model, &dev, 8)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &dev)?;
    model.forward(&input, 0, &mut cache)?;
    for &t in tokens {
        let input = Tensor::from_vec(vec![t], (1, 1), &dev)?;
        model.forward(&input, cache.pos, &mut cache)?;
    }
    Ok(model)
}

/// A prompt whose greedy stream spans several rounds, on a fresh tier.
fn long_prompt(
    cfg: &Config,
    path: &Path,
    slots: usize,
    max: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..5).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let model = streamed_model(cfg, path, slots)?;
        let (ids, _) = greedy(&model, cfg, &prompt, max)?;
        if ids.len() >= 8 {
            return Ok((prompt, ids));
        }
    }
    lloom_tensor::bail!("no prompt gave a greedy stream of >= 8 tokens")
}

struct SpecRun {
    ids: Vec<u32>,
    accepted_drafts: usize,
    /// Positions the cache consumed past the prompt: the stream's prefix of
    /// this length was fed (the final anchor is streamed, never fed).
    fed: usize,
    /// The cache's window counters (`State::::window_*`), read before the
    /// pooled cache is dropped (a pooled channel cannot be cloned: the
    /// pool has one slot).
    truncations: u64,
    rejected_flags: u64,
    redos: u64,
    window_host: u64,
    window_device: u64,
}

/// The speculative loop over the phase rings (the verify as one window with
/// the accept by phase), on a fresh cache.
fn speculative<S: Speculator<Probe = NextnHidden>>(
    model: &Qwen4Exp<RewindableExact, Streamed>,
    cfg: &Config,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
    k: usize,
) -> Result<SpecRun> {
    let dev = Device::Cpu;
    let mut cache: State<RewindableExact> =
        State::<RewindableExact>::in_ring(cfg, &model.spec_pools(1, k, &dev))?;
    cache.want_nextn = true;
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(max);
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &opts,
        &eos,
        &mut |_| Ok(()),
    )?;
    let accepted_drafts = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
    let fed = session.pos() - prompt.len();
    let c = session.cache_mut();
    Ok(SpecRun {
        ids: out.ids,
        accepted_drafts,
        fed,
        truncations: c.window_truncations,
        rejected_flags: c.window_rejected_flags,
        redos: c.window_redos,
        window_host: c.window_host,
        window_device: c.window_device,
    })
}

/// Proposes the greedy stream's next token(s) — every draft right.
struct Oracle {
    stream: Vec<u32>,
    produced: usize,
    k: usize,
}

impl Speculator for Oracle {
    type Probe = NextnHidden;
    type Family = lloom::nn::speculator::Mtp;
    fn n_seq(&self) -> usize {
        1
    }
    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }
    fn max_draft(&self) -> usize {
        self.k
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        self.produced = 0;
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        let next: Vec<u32> = (0..self.k)
            .map(|i| self.stream.get(self.produced + i).copied().unwrap_or(0))
            .collect();
        Ok(vec![Draft::host(next)])
    }
    fn observe(&mut self, _: SeqId, _: &Tensor, accepted: &[u32]) -> Result<()> {
        self.produced += accepted.len();
        Ok(())
    }
}

/// Proposes `k` tokens the target never emits — every draft wrong, and each
/// one routes to experts the real stream never asked for.
struct Garbage {
    tokens: Vec<u32>,
}

impl Speculator for Garbage {
    type Probe = NextnHidden;
    type Family = lloom::nn::speculator::Mtp;
    fn n_seq(&self) -> usize {
        1
    }
    fn max_draft(&self) -> usize {
        self.tokens.len()
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(vec![Draft::host(self.tokens.clone())])
    }
    fn observe(&mut self, _: SeqId, _: &Tensor, _: &[u32]) -> Result<()> {
        Ok(())
    }
}

/// Ids the greedy stream never emits (nor the prompt holds), for garbage drafts.
fn never_emitted(cfg: &Config, prompt: &[u32], ids: &[u32], n: usize) -> Vec<u32> {
    (0..cfg.vocab_size as u32)
        .filter(|t| !ids.contains(t) && !prompt.contains(t) && *t != cfg.eos_token_id)
        .take(n)
        .collect()
}

/// The policy these tests pin: a draft row's miss flags the row and loads
/// nothing (`off`); the shipped `auto` loads a draft's misses after a window
/// whose real row missed, which on this pool is most of them.
fn pin_policy() {
    std::env::set_var("QWEN4EXP_MTP_DRAFT_LOADS", "off");
}

#[test]
fn a_rejected_draft_loads_touches_and_counts_nothing() -> Result<()> {
    pin_policy();
    let cfg = config();
    let path = temp_gguf("garbage");
    write_checkpoint(&cfg, &path)?;
    // 4 slots for 12 experts at top-3: a draft's three experts are a
    // quarter of the pool, so a wrong draft loaded would evict the stream's.
    let slots = 4;
    let max = 24;
    let (prompt, ref_ids) = long_prompt(&cfg, &path, slots, max)?;

    for k in [1usize, 2, 3] {
        let model = streamed_model(&cfg, &path, slots)?;
        let garbage = never_emitted(&cfg, &prompt, &ref_ids, k);
        let run = speculative(
            &model,
            &cfg,
            &mut Garbage { tokens: garbage },
            &prompt,
            max,
            k,
        )?;
        assert_eq!(
            run.ids, ref_ids,
            "k = {k}: garbage drafter: not the greedy stream"
        );
        assert_eq!(run.accepted_drafts, 0, "k = {k}: a garbage draft accepted");
        let (got_stats, got_usage) = tier_state(&model);
        // The absolute reference: the same tier, fresh, over exactly the
        // positions the speculative run fed (the greedy stream's prefix),
        // by a different route — its loads, evictions and usage are what a
        // verify whose rejected rows cost nothing must reproduce exactly.
        let reference = decode_positions(&cfg, &path, slots, &prompt, &ref_ids[..run.fed])?;
        let (want_stats, want_usage) = tier_state(&reference);
        assert!(
            want_stats.misses > 0 && want_stats.evictions > 0,
            "the pool must be tight enough to evict: {want_stats:?}"
        );
        assert_eq!(
            got_usage, want_usage,
            "k = {k}: the usage profile counted a rejected draft's experts"
        );
        assert_eq!(
            (
                got_stats.misses,
                got_stats.evictions,
                got_stats.bytes_loaded
            ),
            (
                want_stats.misses,
                want_stats.evictions,
                want_stats.bytes_loaded
            ),
            "k = {k}: a rejected draft loaded or evicted (got {got_stats:?}, want {want_stats:?})"
        );
        // Non-vacuity: the wrong drafts DID route to missing experts — every
        // round had a draft row flagged, and none was loaded.
        assert!(
            run.rejected_flags > 0,
            "k = {k}: no rejected draft row ever missed — the pool is not tight"
        );
        assert_eq!(run.truncations, 0, "k = {k}: a garbage draft truncated");
        println!(
            "MTPSTREAMED garbage k={k} misses={} evictions={} rejected_flags={} window_host={} window_device={}",
            got_stats.misses,
            got_stats.evictions,
            run.rejected_flags,
            run.window_host,
            run.window_device
        );
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn a_right_draft_on_a_missing_expert_cuts_the_prefix_and_the_stream_is_greedy() -> Result<()> {
    pin_policy();
    let cfg = config();
    let path = temp_gguf("oracle");
    write_checkpoint(&cfg, &path)?;
    let slots = 4;
    let max = 24;
    let (prompt, ref_ids) = long_prompt(&cfg, &path, slots, max)?;
    let mut any_truncation = false;
    for k in [1usize, 2, 3] {
        let model = streamed_model(&cfg, &path, slots)?;
        let mut oracle = Oracle {
            stream: ref_ids.clone(),
            produced: 0,
            k,
        };
        let run = speculative(&model, &cfg, &mut oracle, &prompt, max, k)?;
        assert_eq!(run.ids, ref_ids, "k = {k}: oracle: not the greedy stream");
        assert!(
            run.accepted_drafts > 0,
            "k = {k}: the oracle's drafts must be accepted"
        );
        // On the host plan (the reference path) row 0 is loaded, never
        // flagged: a cut window is never redone.
        assert_eq!(run.redos, 0, "k = {k}: a host-planned window was redone");
        any_truncation |= run.truncations > 0;
        println!(
            "MTPSTREAMED oracle k={k} accepted_drafts={} truncations={} redos={} rejected_flags={} window_host={} window_device={}",
            run.accepted_drafts,
            run.truncations,
            run.redos,
            run.rejected_flags,
            run.window_host,
            run.window_device
        );
    }
    // On a pool a quarter the experts, a right draft's experts are not all
    // resident on every round: some window was cut at its draft instead of
    // redone (the mechanism, not only the identity).
    assert!(
        any_truncation,
        "no oracle window was ever cut at a draft on a 4-slot pool"
    );
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn the_real_drafter_over_the_streamed_tier_is_greedy() -> Result<()> {
    pin_policy();
    let cfg = config();
    let path = temp_gguf("mtp");
    write_checkpoint(&cfg, &path)?;
    let slots = 4;
    let max = 24;
    let dev = Device::Cpu;
    let (prompt, ref_ids) = long_prompt(&cfg, &path, slots, max)?;
    for k in [1usize, 2] {
        let model = streamed_model(&cfg, &path, slots)?;
        let head = testing::mtp_head::<CrownExact, _>(
            &cfg,
            &dev,
            testing::resident_experts(&cfg, 7, &dev),
        );
        let mut mtp =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let run = speculative(&model, &cfg, &mut mtp, &prompt, max, k)?;
        assert_eq!(
            run.ids, ref_ids,
            "k = {k}: MTP drafter: not the greedy stream"
        );
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}
