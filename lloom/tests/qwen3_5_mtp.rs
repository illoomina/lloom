//! MTP on the generic decoder (`nn::decoder`, `nn::mtp`) is byte-identical
//! to greedy on the tiny Qwen3.5 models, dense and MoE, with the block in
//! the file and from a sidecar, on the CPU and on Metal. The identity gate
//! follows `qwen4exp_mtp_identity.rs`: the reference is the K=1 greedy loop
//! (`Session::decode`), the subject `Session::decode_speculative` driven by
//! (a) the real `NextnDrafter` on the checkpoint's own MTP block at k = 1,
//! 2, 3, (b) an oracle that always proposes the greedy stream's next token
//! (every round fully accepted — the resolve path), and (c) a garbage
//! drafter (every round rejected — the rewind + replay path over every
//! layer's state: the DeltaNet's `S` and conv window, the attention's KV).
//! A drafter changes the acceptance rate and nothing else.
use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen3_5::testing::{gguf_bytes, gguf_metadata, Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen3_5::{Config, LoadOptions, Qwen3_5, Qwen3_5Moe};
use lloom::nn::decoder::{Layer, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::mtp::{NextnDrafter, NextnHead};
use lloom::nn::speculator::{Draft, KvSource, NextnHidden, Probes, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::{RewindableExact, Shaped};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::quantized::gguf_file::{self, Content};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Result, Tensor};
use std::path::PathBuf;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn dense_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_nextn_layers = 1;
    // An EOS id the model can still emit, but not at every prompt's second
    // token.
    cfg.eos_token_id = 127;
    cfg
}

fn moe_cfg() -> Config {
    let mut cfg = Config::tiny_moe();
    cfg.n_nextn_layers = 1;
    cfg.eos_token_id = 127;
    cfg
}

/// A target and its MTP head, on any `Decoder` family of this file.
trait Target:
    CausalLM<Config = Config> + Probes<NextnHidden> + Shaped<Profile = RewindableExact>
{
    type Mtp: Layer;
    fn embed_table(&self) -> &QMatMul;
    fn lm_head(&self) -> &QMatMul;
    fn capacity(&self) -> StateCapacity;
    /// Retain `h_nextn` on every forward of `cache`.
    fn arm_nextn(cache: &mut Self::Cache);
}

impl Target for Qwen3_5<RewindableExact> {
    type Mtp = lloom::models::qwen3_5::Qwen3_5MtpLayer<RewindableExact>;
    fn embed_table(&self) -> &QMatMul {
        self.embed_block().table()
    }
    fn lm_head(&self) -> &QMatMul {
        self.head_block().proj()
    }
    fn capacity(&self) -> StateCapacity {
        Qwen3_5::capacity(self)
    }
    fn arm_nextn(cache: &mut Self::Cache) {
        cache.set_want_nextn(true);
    }
}

impl Target for Qwen3_5Moe<RewindableExact> {
    type Mtp = lloom::models::qwen3_5::Qwen3_5MoeMtpLayer<RewindableExact>;
    fn embed_table(&self) -> &QMatMul {
        self.embed_block().table()
    }
    fn lm_head(&self) -> &QMatMul {
        self.head_block().proj()
    }
    fn capacity(&self) -> StateCapacity {
        Qwen3_5Moe::capacity(self)
    }
    fn arm_nextn(cache: &mut Self::Cache) {
        cache.set_want_nextn(true);
    }
}

fn greedy<M: Target>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool)> {
    let cache = model.new_cache(cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut spec = IdentityProposer;
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(max);
    let out = session.decode(&mut spec, &mut rb, &mut sampler, prompt, &opts, &mut |_| {
        Ok(())
    })?;
    Ok((out.ids, out.eos))
}

/// A prompt whose greedy stream is long enough to span several rounds.
fn long_prompt<M: Target>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    max: usize,
) -> Result<(Vec<u32>, Vec<u32>, bool)> {
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..5).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let (ids, eos) = greedy(model, cfg, dev, &prompt, max)?;
        if ids.len() >= 6 {
            return Ok((prompt, ids, eos));
        }
    }
    lloom_tensor::bail!("no prompt gave a greedy stream of >= 6 tokens")
}

fn speculative<M: Target, S: Speculator<Probe = NextnHidden>>(
    model: &M,
    cfg: &Config,
    dev: &Device,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool, usize)> {
    let mut cache = model.new_cache(cfg, dev)?;
    // The prefill retains `h_nextn` at every prompt position for the
    // drafter's seeding (the verify retains its own rows).
    M::arm_nextn(&mut cache);
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(max);
    let mut streamed = Vec::new();
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &opts,
        &eos,
        &mut |b| {
            streamed.extend_from_slice(b);
            Ok(())
        },
    )?;
    assert_eq!(
        out.ids, streamed,
        "the callback stream must equal the returned ids"
    );
    let accepted_drafts = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
    Ok((out.ids, out.eos, accepted_drafts))
}

/// Proposes the greedy stream's next token — every draft accepted.
struct Oracle {
    stream: Vec<u32>,
    produced: usize,
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
        1
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        self.produced = 0;
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        let next = self.stream.get(self.produced).copied().unwrap_or(0);
        Ok(vec![Draft::host(vec![next])])
    }
    fn observe(&mut self, _: SeqId, _: &Tensor, accepted: &[u32]) -> Result<()> {
        self.produced += accepted.len();
        Ok(())
    }
}

/// Proposes a token the target never emits — every draft rejected.
struct Garbage {
    token: u32,
}

impl Speculator for Garbage {
    type Probe = NextnHidden;
    type Family = lloom::nn::speculator::Mtp;
    fn n_seq(&self) -> usize {
        1
    }
    fn max_draft(&self) -> usize {
        1
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, _: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(vec![Draft::host(vec![self.token])])
    }
    fn observe(&mut self, _: SeqId, _: &Tensor, _: &[u32]) -> Result<()> {
        Ok(())
    }
}

/// The three drafters against the greedy stream, on one target with its head.
fn identity_on<M: Target>(
    model: &M,
    head: &NextnHead<M::Mtp>,
    cfg: &Config,
    dev: &Device,
    tag: &str,
) -> Result<()> {
    let max = 24;
    let (prompt, ref_ids, ref_eos) = long_prompt(model, cfg, dev, max)?;
    // (a) the real drafter on the checkpoint's own block, k = 1, 2, 3
    for k in [1usize, 2, 3] {
        let mut mtp = NextnDrafter::new(
            head,
            model.embed_table(),
            model.lm_head(),
            dev,
            model.capacity(),
        )?
        .with_k(k);
        let (ids, eos, accepted) = speculative(model, cfg, dev, &mut mtp, &prompt, max)?;
        assert_eq!(
            ids, ref_ids,
            "{tag}: MTP drafter at k = {k}: not the greedy stream"
        );
        assert_eq!(eos, ref_eos, "{tag}: k = {k}");
        assert!(
            mtp.rows_seen() >= prompt.len() + ref_ids.len() - 1,
            "{tag} k = {k}: the drafter was keyed with {} rows for {} prompt + {} generated",
            mtp.rows_seen(),
            prompt.len(),
            ref_ids.len()
        );
        println!(
            "QWEN35MTP {tag} k={k} tokens={} accepted_rounds={accepted} rows_seen={}",
            ids.len(),
            mtp.rows_seen()
        );
    }
    // (b) the oracle: every round fully accepted (the resolve path)
    let mut oracle = Oracle {
        stream: ref_ids.clone(),
        produced: 0,
    };
    let (ids, eos, accepted) = speculative(model, cfg, dev, &mut oracle, &prompt, max)?;
    assert_eq!(ids, ref_ids, "{tag}: oracle: not the greedy stream");
    assert_eq!(eos, ref_eos);
    assert!(accepted > 0, "{tag}: the oracle's drafts must be accepted");
    // (c) garbage: every round rejected (the rewind + replay path)
    let token = (0..cfg.vocab_size as u32)
        .find(|t| !ref_ids.contains(t) && *t != cfg.eos_token_id)
        .expect("a vocabulary id the stream never emits");
    let mut garbage = Garbage { token };
    let (ids, eos, accepted) = speculative(model, cfg, dev, &mut garbage, &prompt, max)?;
    assert_eq!(
        ids, ref_ids,
        "{tag}: garbage drafter: not the greedy stream"
    );
    assert_eq!(eos, ref_eos);
    assert_eq!(accepted, 0, "{tag}: a garbage draft must never be accepted");
    Ok(())
}

fn load_dense(
    ckpt: &TinyCheckpoint<Qwen3_5Tiny>,
    dev: &Device,
) -> Result<(
    Qwen3_5<RewindableExact>,
    NextnHead<lloom::models::qwen3_5::Qwen3_5MtpLayer<RewindableExact>>,
)> {
    let bytes = gguf_bytes(ckpt, None)?;
    let mut r = std::io::Cursor::new(bytes);
    let content = Content::read(&mut r)?;
    let (m, head) = Qwen3_5::from_gguf_with_mtp(
        content,
        &mut r,
        dev,
        &LoadOptions {
            mtp: true,
            ..Default::default()
        },
    )?;
    Ok((m, head.expect("the tiny file keeps its MTP block")))
}

fn load_moe(
    ckpt: &TinyCheckpoint<Qwen3_5MoeTiny>,
    dev: &Device,
) -> Result<(
    Qwen3_5Moe<RewindableExact>,
    NextnHead<lloom::models::qwen3_5::Qwen3_5MoeMtpLayer<RewindableExact>>,
)> {
    let bytes = gguf_bytes(ckpt, None)?;
    let mut r = std::io::Cursor::new(bytes);
    let content = Content::read(&mut r)?;
    let (m, head) = Qwen3_5Moe::from_gguf_with_mtp(
        content,
        &mut r,
        dev,
        &LoadOptions {
            mtp: true,
            ..Default::default()
        },
    )?;
    Ok((m, head.expect("the tiny file keeps its MTP block")))
}

/// The block in the file (unsloth's Qwen3.8-27B shape: `block_count` =
/// `n_layers` + 1, `nextn_predict_layers` 1), dense and MoE, on the CPU.
#[test]
fn mtp_on_the_decoder_is_byte_identical_to_greedy_on_the_cpu() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(dense_cfg(), Weights::default());
    let (model, head) = load_dense(&dense, &dev)?;
    identity_on(&model, &head, &dense.cfg, &dev, "dense-cpu")?;
    let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(moe_cfg(), Weights::default());
    let (model, head) = load_moe(&moe, &dev)?;
    identity_on(&model, &head, &moe.cfg, &dev, "moe-cpu")?;
    Ok(())
}

/// The same identity on Metal (the kernels' arms: the GDN's step and chunk,
/// the attention's decode and prefill, the rewind of every state).
#[cfg(feature = "metal")]
#[test]
fn mtp_on_the_decoder_is_byte_identical_to_greedy_on_metal() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(dense_cfg(), Weights::default());
    let (model, head) = load_dense(&dense, &dev)?;
    identity_on(&model, &head, &dense.cfg, &dev, "dense-metal")?;
    let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(moe_cfg(), Weights::default());
    let (model, head) = load_moe(&moe, &dev)?;
    identity_on(&model, &head, &moe.cfg, &dev, "moe-metal")?;
    Ok(())
}

/// The window as ONE pass over the phase rings (`StateCapacity::window`,
/// the `Ring` profile's state): on Metal the verify of `k + 1` positions
/// runs the decode kernels' window forms and accepts by phase; the stream is
/// greedy's and every round keeps what the sequential verify keeps, at
/// k = 1, 2, 3, dense and MoE.
#[cfg(feature = "metal")]
#[test]
fn the_window_runs_as_one_pass_on_the_rings_on_metal() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    fn rounds<M: Target>(
        model: &M,
        head: &NextnHead<M::Mtp>,
        cfg: &Config,
        dev: &Device,
        k: usize,
        prompt: &[u32],
        max: usize,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let mut d = NextnDrafter::new(
            head,
            model.embed_table(),
            model.lm_head(),
            dev,
            model.capacity(),
        )?
        .with_k(k);
        let mut cache = model.new_cache(cfg, dev)?;
        M::arm_nextn(&mut cache);
        let mut session = Session::new(model, cache);
        let eos = [cfg.eos_token_id];
        let mut rb = Blocking::new(&eos, dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut d,
            &mut rb,
            &mut sampler,
            prompt,
            &DecodeOptions::new(max),
            &eos,
            &mut |_| Ok(()),
        )?;
        Ok((
            out.ids,
            out.stats.rounds.iter().map(|r| r.accepted).collect(),
        ))
    }
    let max = 24;
    for k in [1usize, 2, 3] {
        // dense
        let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(dense_cfg(), Weights::default());
        let bytes = gguf_bytes(&dense, None)?;
        let load = |window: usize| -> Result<_> {
            let mut r = std::io::Cursor::new(bytes.clone());
            let content = Content::read(&mut r)?;
            let (m, h) = Qwen3_5::<RewindableExact>::from_gguf_with_mtp(
                content,
                &mut r,
                &dev,
                &LoadOptions {
                    mtp: true,
                    window,
                    ..Default::default()
                },
            )?;
            Ok((m, h.expect("the block")))
        };
        let (seq_m, seq_h) = load(0)?;
        let (ring_m, ring_h) = load(k + 1)?;
        let (prompt, ref_ids, _) = long_prompt(&seq_m, &dense.cfg, &dev, max)?;
        assert!(!seq_m.new_cache(&dense.cfg, &dev)?.ring_holds(k + 1));
        assert!(
            ring_m.new_cache(&dense.cfg, &dev)?.ring_holds(k + 1),
            "k = {k}: the rings hold the window"
        );
        let (ids_s, acc_s) = rounds(&seq_m, &seq_h, &dense.cfg, &dev, k, &prompt, max)?;
        let (ids_r, acc_r) = rounds(&ring_m, &ring_h, &dense.cfg, &dev, k, &prompt, max)?;
        assert_eq!(
            ids_s, ref_ids,
            "dense k = {k}: sequential verify: not the greedy stream"
        );
        assert_eq!(
            ids_r, ref_ids,
            "dense k = {k}: ring verify: not the greedy stream"
        );
        assert_eq!(
            acc_r, acc_s,
            "dense k = {k}: ring vs sequential verify, the rounds' accepted counts"
        );
        println!(
            "QWEN35RING dense k={k} rounds={} accepted={acc_r:?}",
            acc_r.len()
        );
        if k == 3 {
            // The real drafter at k = 1..3, the oracle (every window accepted
            // whole, by phase) and the garbage drafter (every window cut at
            // its anchor) on rings that hold the widest window.
            identity_on(&ring_m, &ring_h, &dense.cfg, &dev, "dense-ring")?;
            // Rings built for a shorter window refuse a longer one rather
            // than stepping past their phases.
            let (short_m, short_h) = load(2)?;
            let mut d = NextnDrafter::new(
                &short_h,
                short_m.embed_table(),
                short_m.lm_head(),
                &dev,
                short_m.capacity(),
            )?
            .with_k(2);
            let err = speculative(&short_m, &dense.cfg, &dev, &mut d, &prompt, max)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(err.contains("rings built for a shorter one"), "{err}");
        }
        // MoE
        let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(moe_cfg(), Weights::default());
        let bytes = gguf_bytes(&moe, None)?;
        let load = |window: usize| -> Result<_> {
            let mut r = std::io::Cursor::new(bytes.clone());
            let content = Content::read(&mut r)?;
            let (m, h) = Qwen3_5Moe::<RewindableExact>::from_gguf_with_mtp(
                content,
                &mut r,
                &dev,
                &LoadOptions {
                    mtp: true,
                    window,
                    ..Default::default()
                },
            )?;
            Ok((m, h.expect("the block")))
        };
        let (seq_m, seq_h) = load(0)?;
        let (ring_m, ring_h) = load(k + 1)?;
        let (prompt, ref_ids, _) = long_prompt(&seq_m, &moe.cfg, &dev, max)?;
        let (ids_s, acc_s) = rounds(&seq_m, &seq_h, &moe.cfg, &dev, k, &prompt, max)?;
        let (ids_r, acc_r) = rounds(&ring_m, &ring_h, &moe.cfg, &dev, k, &prompt, max)?;
        assert_eq!(
            ids_s, ref_ids,
            "moe k = {k}: sequential verify: not the greedy stream"
        );
        assert_eq!(
            ids_r, ref_ids,
            "moe k = {k}: ring verify: not the greedy stream"
        );
        assert_eq!(
            acc_r, acc_s,
            "moe k = {k}: ring vs sequential verify, the rounds' accepted counts"
        );
        println!(
            "QWEN35RING moe k={k} rounds={} accepted={acc_r:?}",
            acc_r.len()
        );
        if k == 3 {
            identity_on(&ring_m, &ring_h, &moe.cfg, &dev, "moe-ring")?;
        }
    }
    Ok(())
}

/// The sidecar shape (bartowski's `mtp-*.gguf` beside a stripped main
/// file): the main file written without the block (`block_count` =
/// `n_layers`, no `nextn_predict_layers`), the sidecar with `blk.{n}.*`,
/// `blk.{n}.nextn.*` and its own copies of `token_embd`, `output` and
/// `output_norm` (which the loader drops in favour of the target's). The
/// head so loaded drafts what the in-file one drafts, and the stream is
/// greedy's.
#[test]
fn the_mtp_block_loads_from_a_sidecar_gguf() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let with = TinyCheckpoint::<Qwen3_5Tiny>::new(dense_cfg(), Weights::default());
    let mut stripped_cfg = dense_cfg();
    stripped_cfg.n_nextn_layers = 0;
    let stripped = TinyCheckpoint::<Qwen3_5Tiny>::new(stripped_cfg, Weights::default());
    let dir = std::env::temp_dir().join(format!("lloom-qwen35-mtp-sidecar-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let main_path = dir.join("main.gguf");
    std::fs::write(&main_path, gguf_bytes(&stripped, None)?)?;
    // The sidecar: the MTP block's tensors and the three copies, under the
    // with-block header.
    let n = with.cfg.n_layers;
    let schema = with.schema()?;
    let prefix = format!("blk.{n}.");
    let tensors: Vec<(String, lloom_tensor::quantized::QTensor)> = schema
        .tensors
        .iter()
        .filter(|t| {
            t.name.starts_with(&prefix)
                || ["token_embd.weight", "output.weight", "output_norm.weight"]
                    .contains(&t.name.as_str())
        })
        .map(|t| Ok((t.name.clone(), with.qtensor(t, &Device::Cpu)?)))
        .collect::<Result<_>>()?;
    assert_eq!(tensors.len(), 11 + 4 + 3, "the sidecar's tensors");
    let md = gguf_metadata(&with.cfg, None);
    let md_refs: Vec<(&str, &gguf_file::Value)> = md.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let t_refs: Vec<(&str, &lloom_tensor::quantized::QTensor)> =
        tensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let sidecar_path: PathBuf = dir.join("mtp-sidecar.gguf");
    let mut f = std::fs::File::create(&sidecar_path)?;
    gguf_file::write(&mut f, &md_refs, &t_refs)?;
    drop(f);
    let opts = LoadOptions {
        mtp: true,
        ..Default::default()
    };
    // Without the sidecar the stripped file carries no block.
    let err = Qwen3_5::<RewindableExact>::from_gguf_shards_with_mtp(
        std::slice::from_ref(&main_path),
        None,
        &dev,
        &opts,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(err.contains("no MTP block"), "{err}");
    let (model, head) = Qwen3_5::<RewindableExact>::from_gguf_shards_with_mtp(
        &[main_path],
        Some(&sidecar_path),
        &dev,
        &opts,
    )?;
    let head = head.expect("the sidecar's block");
    identity_on(&model, &head, &with.cfg, &dev, "dense-sidecar")?;
    // The sidecar's head drafts what the in-file head drafts: both keyed
    // from the same prompt rows, the same next token.
    let (model_in, head_in) = load_dense(&with, &dev)?;
    let max = 12;
    let (prompt, _, _) = long_prompt(&model_in, &with.cfg, &dev, max)?;
    let drafts = |m: &Qwen3_5<RewindableExact>,
                  h: &NextnHead<lloom::models::qwen3_5::Qwen3_5MtpLayer<RewindableExact>>|
     -> Result<Vec<usize>> {
        let mut d = NextnDrafter::new(h, m.embed_table(), m.lm_head(), &dev, m.capacity())?;
        let mut cache = m.new_cache(&with.cfg, &dev)?;
        cache.set_want_nextn(true);
        let mut session = Session::new(m, cache);
        let eos = [with.cfg.eos_token_id];
        let mut rb = Blocking::new(&eos, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut d,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(max),
            &eos,
            &mut |_| Ok(()),
        )?;
        Ok(out.stats.rounds.iter().map(|r| r.accepted).collect())
    };
    assert_eq!(
        drafts(&model, &head)?,
        drafts(&model_in, &head_in)?,
        "the sidecar's head and the in-file head accept the same rounds"
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// `begin_at` keeps a lane keyed from the reused prefix (cut back to it)
/// and resets one keyed from something else — the drafter's half of the
/// engine's prefix reuse (`Speculator::begin_at`).
#[test]
fn the_drafters_lane_follows_a_reused_prefix_and_resets_otherwise() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(dense_cfg(), Weights::default());
    let (model, head) = load_dense(&dense, &dev)?;
    let cfg = &dense.cfg;
    let max = 8;
    let (prompt, ids, _) = long_prompt(&model, cfg, &dev, max)?;
    let mut d = NextnDrafter::new(
        &head,
        model.embed_table(),
        model.lm_head(),
        &dev,
        model.capacity(),
    )?;
    let (out, _, _) = speculative(&model, cfg, &dev, &mut d, &prompt, max)?;
    assert_eq!(out, ids);
    let seen = d.rows_seen();
    // The whole conversation so far is the reused prefix: the lane keeps it.
    let mut kept: Vec<u32> = prompt.clone();
    kept.extend_from_slice(&ids[..ids.len() - 1]);
    d.begin_at(SeqId(0), &ids[ids.len() - 1..], &kept)?;
    assert_eq!(d.lane_resets(), 0, "a lane keyed from the prefix is kept");
    assert_eq!(
        d.rows_seen(),
        kept.len(),
        "cut back to the prefix: {seen} rows before"
    );
    // A prefix the lane was not keyed from resets it.
    let other: Vec<u32> = kept.iter().map(|t| (t + 1) % 120).collect();
    d.begin_at(SeqId(0), &[3], &other)?;
    assert_eq!(d.lane_resets(), 1);
    assert_eq!(d.rows_seen(), 0);
    Ok(())
}
