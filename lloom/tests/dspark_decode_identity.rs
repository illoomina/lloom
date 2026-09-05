//! `Session::decode_speculative` is output-identical to greedy `Session::decode`
//! — the correctness gate for the DSpark decode-driver integration (stage 2 of
//! `design/docs/decode-driver-design.md`).
//!
//! # What this pins, and why it is the right reference
//!
//! Greedy speculative decoding accepts a drafted token iff it equals the
//! target's own argmax, so the emitted stream is **byte-identical to
//! non-speculative greedy decoding by construction** (`lfm2.rs` `verify_step`:
//! "greedy needs no epsilon"; `design/docs/speculation.md` §10.2n). The
//! implication the design leans on hardest is that the identity holds **for any
//! drafter, however bad** — the target corrects every mismatch, so a drafter that
//! proposes garbage changes the acceptance rate and nothing else.
//!
//! So the reference here is a genuinely different code path (`Session::decode`'s
//! K=1 identity body, `forward` per token) run against the SAME model, and the
//! subject (`Session::decode_speculative`, `forward_all` over a K-block with
//! rollback) is driven by a **deliberately wrong** proposer. If the accept /
//! rollback / re-key logic in `decode_speculative` is off by one, keeps a
//! rejected token, or fails to roll the cache back, the two streams diverge and
//! this fails. `CONTRIBUTING.md` §3.1b: the reference comes from a different
//! mechanism, not from restating the subject.
//!
//! # The mutation this survived (kept in the comments, per §3.1a)
//!
//! Running `decode_speculative` while feeding the drafter's proposals straight
//! through as "accepted" (skipping the argmax verify) makes the two streams
//! diverge on the first mismatch — asserted below to be non-vacuous by first
//! checking the two arms produce a non-empty stream.
//!
//! It runs on the CPU: the question is which tokens survive the accept logic, not
//! a kernel, and the CPU backend is bit-stable (`CONTRIBUTING.md` §3.1).

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{Cache, Config, ConvState, KvAppend, LayerType, Model};
use lloom::nn::speculator::{Draft, DraftTokens, LayerTaps, SeqId, Speculator, TargetProbe};
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom_tensor::{DType, Device, Result, Tensor};

/// A tiny LFM2 whose layer count leaves room for a five-layer tap. Six layers
/// (conv/attention interleaved), `hidden_size = 8`. The spec arms
/// (`RotatingRing` + `InPlace`) are the ONLY cache that `Cache::advance` /
/// `Cache::resolve` accept (the shuffle is destructive, `Cat` has no length to
/// rewind), and the taps are on so `verify_and_probe` has a probe to hand the
/// drafter — the same setup `lfm2-chat --dspark` builds.
fn tiny_config(k: usize) -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        // A real EOS id in the tiny vocabulary so both arms can terminate; picked
        // so the fixed-weight ramp reaches it within the token budget.
        eos_token_id: Some(7),
        use_flash_attn: false,
        attn_impl: Default::default(),
        flash_page_size: 256,
        flash_pages_per_chunk: 1,
        flash_scratch_sizing: Default::default(),
        // The rewindable arms speculation requires.
        kv_append: KvAppend::InPlace,
        kv_quant: Default::default(),
        conv_state: ConvState::RotatingRing { k: k + 1 },
        memory_budget: None,
        gemv_silu: false,
        gemv_residual: false,
        // A five-layer tap on valid layers, so the target produces a probe
        // (`LayerTaps<5>`). The drafter ignores it; what matters is the tap
        // machinery runs the same way `--dspark` drives it.
        tap_layers: vec![1, 2, 3, 4, 5],
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// The same fixed-ramp weights the conv-rollback test uses — deterministic, so
/// the two decode arms agree on the model and any divergence is the paths'.
fn build(cfg: &Config, dev: &Device) -> Result<Model> {
    let mut tensors = std::collections::HashMap::new();
    let mut fill = |name: &str, shape: Vec<usize>| -> Result<()> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
        tensors.insert(name.to_string(), Tensor::from_vec(v, shape, dev)?);
        Ok(())
    };
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    fill("model.embed_tokens.weight", vec![cfg.vocab_size, h])?;
    fill("model.embedding_norm.weight", vec![h])?;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        fill(&format!("{p}.operator_norm.weight"), vec![h])?;
        fill(&format!("{p}.ffn_norm.weight"), vec![h])?;
        fill(&format!("{p}.feed_forward.w1.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w3.weight"), vec![i, h])?;
        fill(&format!("{p}.feed_forward.w2.weight"), vec![h, i])?;
        match cfg.layer_types[l] {
            LayerType::Conv => {
                fill(&format!("{p}.conv.in_proj.weight"), vec![3 * h, h])?;
                fill(
                    &format!("{p}.conv.conv.weight"),
                    vec![h, 1, cfg.conv_l_cache],
                )?;
                fill(&format!("{p}.conv.out_proj.weight"), vec![h, h])?;
            }
            LayerType::FullAttention => {
                let kv = cfg.num_key_value_heads * cfg.head_dim();
                fill(&format!("{p}.self_attn.q_proj.weight"), vec![h, h])?;
                fill(&format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
                fill(&format!("{p}.self_attn.out_proj.weight"), vec![h, h])?;
                fill(
                    &format!("{p}.self_attn.q_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
                fill(
                    &format!("{p}.self_attn.k_layernorm.weight"),
                    vec![cfg.head_dim()],
                )?;
            }
        }
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    Model::new(cfg, vb)
}

/// A deliberately wrong proposer: it proposes `k` copies of a fixed token every
/// round and keys nothing from the probe. Its `Probe` is `LayerTaps<5>`, so it
/// pairs with LFM2 through `Session::speculative_step`'s `M: Probes<S::Probe>`
/// bound — the same pairing `DSparkSpeculator` uses. Because greedy verify accepts
/// a token iff it is the target's argmax, this drafter's proposals are almost
/// always rejected and the target's own correction is what advances — which is
/// exactly the state the accept path must handle correctly.
struct GarbageDrafter {
    k: usize,
    token: u32,
}

impl Speculator for GarbageDrafter {
    type Probe = LayerTaps<5>;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active
            .iter()
            .map(|_| Draft::host(vec![self.token; self.k]))
            .collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        // A drafter that learns nothing. The identity must hold regardless.
        Ok(())
    }
}

/// A drafter that records what the loop hands it: every `observe` call's
/// (rows per tap, accepted) and whether a `draft` had already run. It drafts
/// garbage, so the identity still holds and the record is the whole test.
struct RecordingDrafter {
    k: usize,
    drafts: usize,
    observed: Vec<(usize, usize, Vec<u32>)>, // (drafts before this observe, rows, accepted)
}

impl Speculator for RecordingDrafter {
    type Probe = LayerTaps<5>;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        self.drafts += 1;
        Ok(active
            .iter()
            .map(|_| Draft::host(vec![0; self.k]))
            .collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        assert_eq!(probe.len(), 5, "LayerTaps<5> hands five tensors");
        let rows = probe[0].dim(1)?;
        for t in probe {
            assert_eq!(t.dim(1)?, rows, "every tap covers the same positions");
        }
        self.observed.push((self.drafts, rows, accepted.to_vec()));
        Ok(())
    }
}

/// The drafter is keyed from the WHOLE prompt before it drafts anything: the
/// first `observe` arrives before the first `draft`, carries one row per prompt
/// token in every tap, and names the prefill's argmax as the accepted tail —
/// `lfm2-dspark-verify`'s `--prefill-ctx all`. Every later `observe` carries
/// exactly the accepted rows of its round. A lazily-seeded loop (round 0 drafts
/// on an empty context) fails the first assertion; a loop trimming the prefill
/// probe to the last position fails the second.
#[test]
fn decode_speculative_seeds_the_drafter_from_every_prompt_position() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7 + 1, 9];
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();

    let cache = Cache::new(true, DType::F32, &cfg, &dev)?;
    let mut session = Session::new(&model, cache);
    let mut spec = RecordingDrafter {
        k,
        drafts: 0,
        observed: Vec::new(),
    };
    let mut rb = Blocking::new(&eos_ids, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(24);
    let out = session.decode_speculative(
        &mut spec,
        &mut rb,
        &mut sampler,
        &prompt,
        &opts,
        &eos_ids,
        &mut |_: &[u32]| Ok(()),
    )?;
    assert!(out.ids.len() > k, "the run must span more than one round");

    let (drafts_before, rows, accepted) = spec.observed.first().expect("an observe ran");
    assert_eq!(
        *drafts_before, 0,
        "the prompt seed precedes the first draft"
    );
    assert_eq!(
        *rows,
        prompt.len(),
        "the seed carries every prompt position"
    );
    assert_eq!(
        accepted,
        &out.ids[..1],
        "the seed's accepted tail is the prefill's argmax"
    );
    assert_eq!(
        spec.observed.len(),
        out.stats.rounds.len() + 1,
        "one observe per round, plus the seed"
    );
    for ((_, rows, accepted), round) in spec.observed[1..].iter().zip(&out.stats.rounds) {
        assert_eq!(
            *rows, round.accepted,
            "a round's probe is trimmed to its accepted rows"
        );
        assert_eq!(accepted.len(), round.accepted);
    }
    Ok(())
}

/// The token-identity gate: greedy `decode` and speculative `decode_speculative`
/// produce byte-identical ids for the same prompt.
#[test]
fn decode_speculative_is_byte_identical_to_greedy() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;

    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_tokens = 40usize;
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    assert!(!eos_ids.is_empty(), "the tiny config must carry an EOS id");

    // --- reference: greedy K=1 decode (the identity proposer, blocking) ------
    let (ref_ids, ref_eos) = {
        let cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let mut ids = Vec::new();
        let out = session.decode(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &opts,
            &mut |b: &[u32]| {
                ids.extend_from_slice(b);
                Ok(())
            },
        )?;
        assert_eq!(
            out.ids, ids,
            "the callback stream must equal the returned ids"
        );
        (out.ids, out.eos)
    };

    // The gate is only meaningful if the reference actually generated something
    // (§3.1a: run the non-vacuity arm and say so). A zero-length stream would
    // make any "identical" claim vacuous.
    assert!(
        ref_ids.len() > k,
        "the greedy reference produced only {} tokens; the gate needs a stream long \
         enough to span more than one speculative round (got EOS={ref_eos})",
        ref_ids.len()
    );

    // --- subject: speculative decode with a GARBAGE drafter ------------------
    let (spec_ids, spec_eos) = {
        let cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let mut ids = Vec::new();
        let out = session.decode_speculative(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &opts,
            &eos_ids,
            &mut |b: &[u32]| {
                ids.extend_from_slice(b);
                Ok(())
            },
        )?;
        assert_eq!(
            out.ids, ids,
            "the callback stream must equal the returned ids"
        );
        (out.ids, out.eos)
    };

    assert_eq!(
        ref_ids, spec_ids,
        "DSpark speculative decode must be BYTE-IDENTICAL to greedy decode \
         (design/docs/speculation.md §10.2n): the target corrects every mismatch, so \
         a garbage drafter cannot change the output. A divergence here is a bug in the \
         accept / rollback / re-key logic of decode_speculative."
    );
    assert_eq!(
        ref_eos, spec_eos,
        "the two arms must agree on whether EOS ended the run"
    );

    Ok(())
}

/// The identity holds for a range of block sizes K — a K-off-by-one in the window
/// or the accept prefix would break exactly one of these.
#[test]
fn decode_speculative_identity_across_block_sizes() -> Result<()> {
    let dev = Device::Cpu;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_tokens = 32usize;

    // The greedy reference is independent of K, so compute it once at K=1's cache
    // shape... except the cache arm depends on K (RotatingRing{K+1}). The stream
    // is token-identical across cache arms for greedy decode (every arm is
    // token-identical, §13.5), so take the reference from the largest K's cache
    // and compare every K's speculative run to it.
    let ref_cfg = tiny_config(8);
    let ref_model = build(&ref_cfg, &dev)?;
    let eos_ids: Vec<u32> = ref_cfg.eos_token_id.into_iter().collect();
    let ref_ids = {
        let cache = Cache::new(true, DType::F32, &ref_cfg, &dev)?;
        let mut session = Session::new(&ref_model, cache);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        session
            .decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt,
                &opts,
                &mut |_| Ok(()),
            )?
            .ids
    };
    assert!(
        ref_ids.len() >= 9,
        "reference too short to exercise the K sweep"
    );

    for k in [1usize, 2, 3, 5, 8] {
        let cfg = tiny_config(k);
        let model = build(&cfg, &dev)?;
        let cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let ids = session
            .decode_speculative(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt,
                &opts,
                &eos_ids,
                &mut |_| Ok(()),
            )?
            .ids;
        assert_eq!(
            ref_ids, ids,
            "speculative decode at K={k} diverged from greedy — the accept prefix or the \
             verify window is K-dependent-wrong"
        );
    }
    Ok(())
}

/// A proposal block for the greedy stream `truth` (the `k` tokens the target
/// emits after the anchor) that is right up to position `j` and wrong there —
/// so the accept length is known by construction: `j + 1` for `j < k`, and
/// `k + 1` (every proposal right, plus the bonus) for `j == k`.
fn adversarial_block(truth: &[u32], j: usize, vocab: u32) -> Vec<u32> {
    let mut proposed = truth.to_vec();
    if j < truth.len() {
        proposed[j] = (truth[j] + 1) % vocab;
        assert_ne!(proposed[j], truth[j]);
    }
    proposed
}

/// The device-resident verify (`verify_tokens`, the proposals as a tensor)
/// accepts exactly what the host verify (`verify_step`, the proposals as a
/// slice) accepts and leaves the cache in the same state — over adversarial
/// blocks whose accepted length is fixed by construction rather than read off
/// the other arm (`CONTRIBUTING.md` §3.1b: an absolute reference, not a shape).
///
/// The reference stream is greedy `Session::decode` on the same model; every
/// accepted token must be that stream's, and the next `forward` on both caches
/// must agree bit-for-bit AND continue the stream. A verify that fed the window
/// one position off, dropped the anchor, split the joint readback at the wrong
/// index, or resolved the cache to the wrong length fails one of these.
fn verify_arms_agree_on(dev: &Device) -> Result<()> {
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();

    let ref_ids = {
        let cache = Cache::new(true, DType::F32, &cfg, dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&eos_ids, dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(24);
        session
            .decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt,
                &opts,
                &mut |_| Ok(()),
            )?
            .ids
    };
    // `ref_ids[0]` is the prefill's argmax (the anchor); the window's `k + 1`
    // argmaxes are `ref_ids[1..=k + 1]`, and one more token checks the cache.
    assert!(
        ref_ids.len() >= k + 3,
        "the greedy reference produced only {} tokens; the arms need k + 3 = {}",
        ref_ids.len(),
        k + 3
    );
    let pos = prompt.len();
    let last = ref_ids[0];
    let truth = &ref_ids[1..1 + k];
    let prefill = || -> Result<Cache> {
        let mut cache = Cache::new(true, DType::F32, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        model.forward(&input, 0, &mut cache)?;
        Ok(cache)
    };
    let next_logits = |cache: &mut Cache, tail: u32, at: usize| -> Result<Vec<f32>> {
        let input = Tensor::new(&[tail], dev)?.unsqueeze(0)?;
        model
            .forward(&input, at, cache)?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let argmax = |logits: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    };

    let mut lengths_seen = Vec::new();
    for j in 0..=k {
        let proposed = adversarial_block(truth, j, cfg.vocab_size as u32);
        let want_len = if j < k { j + 1 } else { k + 1 };

        let mut cache_h = prefill()?;
        let acc_h = model.verify_step(last, &proposed, pos, &mut cache_h)?;

        let mut cache_d = prefill()?;
        let proposed_t = Tensor::new(proposed.as_slice(), dev)?;
        let acc_d = model.verify_tokens(last, &proposed_t, pos, &mut cache_d)?;

        assert_eq!(
            acc_h.len(),
            want_len,
            "j={j}: the host arm accepted {} where the block was built to accept {want_len}",
            acc_h.len()
        );
        assert_eq!(
            acc_h, acc_d,
            "j={j}: the two arms accepted different tokens"
        );
        assert_eq!(
            &acc_h[..],
            &ref_ids[1..1 + want_len],
            "j={j}: an accepted token is not the greedy stream's"
        );

        let tail = *acc_h.last().expect("never empty");
        let at = pos + want_len;
        let next_h = next_logits(&mut cache_h, tail, at)?;
        let next_d = next_logits(&mut cache_d, tail, at)?;
        assert_eq!(
            next_h, next_d,
            "j={j}: the caches diverged after resolve — the arms resolved to different lengths"
        );
        assert_eq!(
            argmax(&next_h),
            ref_ids[1 + want_len],
            "j={j}: the resolved cache does not continue the greedy stream"
        );
        lengths_seen.push(want_len);
    }
    // Non-vacuity: the sweep exercised every accept length from 1 to k + 1.
    assert_eq!(lengths_seen, (1..=k + 1).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn verify_tokens_accepts_exactly_what_verify_step_accepts() -> Result<()> {
    verify_arms_agree_on(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn verify_tokens_accepts_exactly_what_verify_step_accepts_on_metal() -> Result<()> {
    verify_arms_agree_on(&Device::new_metal(0)?)
}

/// A drafter that proposes garbage as a DEVICE-resident block — the shape
/// `DSparkSpeculator` hands the loop — so the identity gate covers
/// `verify_and_probe`'s `DraftTokens::Device` arm end to end, including the
/// joint readback and the rollback that follows it.
struct DeviceGarbageDrafter {
    k: usize,
    token: u32,
    dev: Device,
}

impl Speculator for DeviceGarbageDrafter {
    type Probe = LayerTaps<5>;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        active
            .iter()
            .map(|_| {
                Ok(Draft {
                    tokens: DraftTokens::Device(Tensor::new(
                        vec![self.token; self.k].as_slice(),
                        &self.dev,
                    )?),
                    confidence: Vec::new(),
                })
            })
            .collect()
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

/// The token-identity gate through the device-resident draft seam: greedy
/// `decode` and `decode_speculative` with a device-block garbage drafter agree
/// byte for byte, across block sizes.
#[test]
fn decode_speculative_is_byte_identical_to_greedy_with_device_drafts() -> Result<()> {
    let dev = Device::Cpu;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_tokens = 32usize;
    let ref_cfg = tiny_config(8);
    let ref_model = build(&ref_cfg, &dev)?;
    let eos_ids: Vec<u32> = ref_cfg.eos_token_id.into_iter().collect();
    let ref_ids = {
        let cache = Cache::new(true, DType::F32, &ref_cfg, &dev)?;
        let mut session = Session::new(&ref_model, cache);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        session
            .decode(
                &mut spec,
                &mut rb,
                &mut sampler,
                &prompt,
                &opts,
                &mut |_| Ok(()),
            )?
            .ids
    };
    assert!(
        ref_ids.len() >= 9,
        "reference too short to exercise the K sweep"
    );

    for k in [1usize, 3, 8] {
        let cfg = tiny_config(k);
        let model = build(&cfg, &dev)?;
        let cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = DeviceGarbageDrafter {
            k,
            token: 0,
            dev: dev.clone(),
        };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let out = session.decode_speculative(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &opts,
            &eos_ids,
            &mut |_| Ok(()),
        )?;
        assert_eq!(
            ref_ids, out.ids,
            "device-block speculative decode at K={k} diverged from greedy"
        );
        assert!(
            out.stats.rounds.iter().all(|r| r.accepted >= 1),
            "every round accepts at least the target's own token"
        );
    }
    Ok(())
}
