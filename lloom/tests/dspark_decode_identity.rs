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
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::nn::speculator::{Draft, DraftTokens, LayerTaps, SeqId, Speculator, TargetProbe};
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom::profile::RewindableExact;
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
fn build(cfg: &Config, dev: &Device) -> Result<Model<RewindableExact>> {
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
    Model::<RewindableExact>::new(cfg, vb)
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
    type Family = lloom::nn::speculator::DSpark;

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
    type Family = lloom::nn::speculator::DSpark;

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

    let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
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
        lloom::profile::RewindProof::of(),
        &mut spec,
        &mut rb,
        &mut sampler,
        &prompt,
        &opts,
        &eos_ids,
        &mut |_: &[u32]| Ok(()),
    )?;
    assert!(out.ids.len() > k, "the run must span more than one round");
    assert_eq!(out.stats.prefill_units, 1, "one-shot: one forward");

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
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let mut ids = Vec::new();
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &ref_cfg, &dev)?;
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let ids = session
            .decode_speculative(
                lloom::profile::RewindProof::of(),
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?;
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
    let prefill = || -> Result<State<RewindableExact>> {
        let mut cache = State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?;
        let input = Tensor::new(prompt.as_slice(), dev)?.unsqueeze(0)?;
        model.forward(&input, 0, &mut cache)?;
        Ok(cache)
    };
    let next_logits =
        |cache: &mut State<RewindableExact>, tail: u32, at: usize| -> Result<Vec<f32>> {
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
        let acc_h = model.verify_step(last, &proposed, pos, &mut cache_h, &eos_ids)?;

        let mut cache_d = prefill()?;
        let proposed_t = Tensor::new(proposed.as_slice(), dev)?;
        let acc_d = model.verify_tokens(last, &proposed_t, pos, &mut cache_d, &eos_ids)?;

        // The pipelined arm: the same pass split at the readback, the wait on
        // its own marker (`verify_tokens_enqueue` / `verify_tokens_complete`).
        let mut cache_p = prefill()?;
        let pending = model.verify_tokens_enqueue(last, &proposed_t, pos, &mut cache_p)?;
        let (acc_p, probe_p) = model.verify_tokens_complete(pending, &mut cache_p, &eos_ids)?;

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
            acc_h, acc_p,
            "j={j}: the split (enqueue/complete) arm accepted different tokens"
        );
        // On Metal the completion waits on the event signalled behind the
        // round's stamp: every round signalled it, and each was either already
        // visible or parked until it landed, none timing out
        // (`expert-streaming.md` §20.14b).
        #[cfg(feature = "metal")]
        if dev.is_metal() {
            for (arm, w) in [
                ("device", cache_d.verify_waits()),
                ("split", cache_p.verify_waits()),
            ] {
                assert!(
                    w.signalled > 0,
                    "j={j}: the {arm} arm signalled no publish event: {w:?}"
                );
                assert_eq!(
                    w.already + w.parked,
                    w.signalled,
                    "j={j}: the {arm} arm's completion did not wait on its event: {w:?}"
                );
                assert_eq!(
                    w.fell_back, 0,
                    "j={j}: the {arm} arm's park timed out: {w:?}"
                );
            }
        }
        assert_eq!(probe_p.len(), 5, "the split arm hands back the five taps");
        for t in &probe_p {
            assert_eq!(
                t.dim(1)?,
                want_len,
                "j={j}: the split arm's probe is trimmed to the accepted rows"
            );
        }
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
        let next_p = next_logits(&mut cache_p, tail, at)?;
        assert_eq!(
            next_h, next_p,
            "j={j}: the split arm's cache diverged after resolve"
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
    /// Sequence slots — the pipelined driver needs one per session.
    n: usize,
}

impl Speculator for DeviceGarbageDrafter {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        self.n
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &ref_cfg, &dev)?;
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
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = DeviceGarbageDrafter {
            k,
            token: 0,
            dev: dev.clone(),
            n: 1,
        };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(max_tokens);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
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

/// A drafter that proposes garbage for `n` slots and records every `observe`
/// it receives as `(slot, accepted)`, so a test can check that the pipelined
/// driver re-keys each sequence from its OWN verify and no other's.
struct SlotRecordingDrafter {
    k: usize,
    n: usize,
    observed: Vec<(usize, Vec<u32>)>,
}

impl Speculator for SlotRecordingDrafter {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        self.n
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
            .map(|_| Draft::host(vec![0; self.k]))
            .collect())
    }

    fn observe(
        &mut self,
        seq: SeqId,
        probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        assert_eq!(probe.len(), 5, "LayerTaps<5> hands five tensors");
        self.observed.push((seq.0, accepted.to_vec()));
        Ok(())
    }
}

/// Three prompts of different lengths and content: the solo reference streams
/// differ from one another (a pipelined run that leaked one sequence's state
/// into another would have somewhere to leak TO), and they end at different
/// rounds, so the driver's handling of a lane finishing while others continue
/// is on the path.
const PIPELINE_PROMPTS: [&[u32]; 3] = [&[1, 2, 3, 4, 5], &[9, 8, 7, 6, 5, 4, 3, 2], &[2, 4, 6]];

/// The pipelined driver's streams against the two references it must equal,
/// on `dev`: each sequence's `decode_speculative` run solo, and greedy
/// `decode`. Byte-identical per sequence, round for round, under `eos_ids`
/// and `max_tokens`; `want_eos` / `want_cap` demand that at least one sequence
/// end on an EOS inside the budget / run to the cap, so the two cuts the driver
/// makes are on the path.
fn pipelined_matches_solo_on(
    dev: &Device,
    max_tokens: usize,
    extra_eos: &[u32],
    want_eos: bool,
    want_cap: bool,
) -> Result<()> {
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, dev)?;
    let mut eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    eos_ids.extend_from_slice(extra_eos);

    // Reference 1: greedy K=1 per prompt.
    let greedy: Vec<Vec<u32>> = PIPELINE_PROMPTS
        .iter()
        .map(|prompt| -> Result<Vec<u32>> {
            let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?;
            let mut session = Session::new(&model, cache);
            let mut spec = IdentityProposer;
            let mut rb = Blocking::new(&eos_ids, dev)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let opts = DecodeOptions::new(max_tokens);
            Ok(session
                .decode(&mut spec, &mut rb, &mut sampler, prompt, &opts, &mut |_| {
                    Ok(())
                })?
                .ids)
        })
        .collect::<Result<_>>()?;
    // Non-vacuity: the streams are long enough to span more than one block and
    // are not all the same stream (a leak needs somewhere to leak TO).
    for g in &greedy {
        assert!(g.len() > k, "a greedy reference is only {} tokens", g.len());
    }
    assert!(
        greedy[0] != greedy[1] && greedy[1] != greedy[2],
        "the reference streams must differ for a cross-sequence leak to be visible"
    );

    // Reference 2: each sequence through the serial speculative loop, solo.
    let solo: Vec<lloom::nn::DecodeOut> = PIPELINE_PROMPTS
        .iter()
        .map(|prompt| -> Result<lloom::nn::DecodeOut> {
            let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?;
            let mut session = Session::new(&model, cache);
            let mut spec = DeviceGarbageDrafter {
                k,
                token: 0,
                dev: dev.clone(),
                n: 1,
            };
            let mut rb = Blocking::new(&eos_ids, dev)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let opts = DecodeOptions::new(max_tokens);
            session.decode_speculative(
                lloom::profile::RewindProof::of(),
                &mut spec,
                &mut rb,
                &mut sampler,
                prompt,
                &opts,
                &eos_ids,
                &mut |_| Ok(()),
            )
        })
        .collect::<Result<_>>()?;
    for (s, g) in solo.iter().zip(&greedy) {
        assert_eq!(
            &s.ids, g,
            "the solo speculative stream is the greedy stream"
        );
    }
    if want_eos {
        assert!(
            solo.iter().any(|o| o.eos),
            "no sequence reached EOS within {max_tokens} tokens; the EOS-in-block path is untested"
        );
    }
    if want_cap {
        assert!(
            solo.iter().any(|o| !o.eos && o.ids.len() == max_tokens),
            "no sequence ran to the cap; the max_tokens cut is untested"
        );
    }

    // Subject: the three sequences pipelined through one drafter with three slots.
    let mut sessions: Vec<Session<'_, lloom::models::lfm2::Model<RewindableExact>>> = (0
        ..PIPELINE_PROMPTS.len())
        .map(|_| {
            Ok(Session::new(
                &model,
                State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut spec = DeviceGarbageDrafter {
        k,
        token: 0,
        dev: dev.clone(),
        n: PIPELINE_PROMPTS.len(),
    };
    let opts = DecodeOptions::new(max_tokens);
    let mut streamed: Vec<Vec<u32>> = vec![Vec::new(); PIPELINE_PROMPTS.len()];
    let outs = Session::decode_speculative_pipelined(
        &mut sessions,
        lloom::profile::RewindProof::of(),
        &mut spec,
        &PIPELINE_PROMPTS,
        &opts,
        &eos_ids,
        dev,
        &mut |i, ids| {
            streamed[i].extend_from_slice(ids);
            Ok(())
        },
    )?;
    assert_eq!(outs.len(), PIPELINE_PROMPTS.len());
    for (i, out) in outs.iter().enumerate() {
        assert_eq!(
            out.ids, greedy[i],
            "sequence {i}: the pipelined stream must be BYTE-IDENTICAL to greedy and to its \
             solo speculative run — a divergence is a cross-sequence leak or a bookkeeping \
             error in decode_speculative_pipelined"
        );
        assert_eq!(
            out.ids, streamed[i],
            "sequence {i}: the callback stream equals the ids"
        );
        assert_eq!(out.eos, solo[i].eos, "sequence {i}: EOS verdicts agree");
        // Round for round: the same drafter proposes the same garbage, so the
        // accept lengths must be the solo run's — a sequence whose rounds
        // differ was verified against something not its own.
        let solo_rounds: Vec<usize> = solo[i].stats.rounds.iter().map(|r| r.accepted).collect();
        let pipe_rounds: Vec<usize> = out.stats.rounds.iter().map(|r| r.accepted).collect();
        assert_eq!(
            pipe_rounds, solo_rounds,
            "sequence {i}: accepted-per-round differs from solo"
        );
        assert_eq!(
            out.stats.overrun, solo[i].stats.overrun,
            "sequence {i}: overrun differs"
        );
        assert_eq!(
            sessions[i].tokens().len() + sessions[i].overrun(),
            sessions[i].pos(),
            "sequence {i}: the token log plus the overrun covers the cache"
        );
    }
    let total_rounds: usize = outs.iter().map(|o| o.stats.rounds.len()).sum();
    assert!(total_rounds >= 3, "the run spanned {total_rounds} rounds");
    Ok(())
}

/// The token the tiny model's streams settle into ([0, 10, 0, 10, …]) after
/// their prompt-specific openings: as an extra EOS it ends each sequence at a
/// different position (13–15 tokens in), so with a cap of 14 one sequence
/// stops on an EOS inside a block and another runs to the cap. Hardcoded rather
/// than derived from the streams, so a fixture change fails the non-vacuity
/// assertions above instead of silently choosing a different test.
const SETTLED_TOKEN: u32 = 10;

#[test]
fn pipelined_decode_is_byte_identical_to_solo_speculative_and_greedy() -> Result<()> {
    // Long streams, every sequence to the cap: the pipeline runs deep.
    pipelined_matches_solo_on(&Device::Cpu, 96, &[], false, true)?;
    // Short streams cut two ways: one on an EOS inside a block, one at the cap.
    pipelined_matches_solo_on(&Device::Cpu, 14, &[SETTLED_TOKEN], true, true)
}

/// The same gate on Metal, where `verify_complete`'s wait is the staged read's
/// own command buffer (`Tensor::stage_read`) rather than the queue's tail —
/// the mechanism the CPU run cannot exercise.
#[cfg(feature = "metal")]
#[test]
fn pipelined_decode_is_byte_identical_to_solo_speculative_and_greedy_on_metal() -> Result<()> {
    let dev = Device::new_metal(0)?;
    pipelined_matches_solo_on(&dev, 96, &[], false, true)?;
    pipelined_matches_solo_on(&dev, 14, &[SETTLED_TOKEN], true, true)
}

/// Each sequence is re-keyed from its own verifies only: the `observe` calls
/// the driver makes for slot `i` carry exactly slot `i`'s accepted tokens, in
/// order, starting from its prompt seed — and the seed precedes its first
/// draft, as the serial loop's does.
#[test]
fn pipelined_observe_is_keyed_per_sequence() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let max_tokens = 24usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let n = PIPELINE_PROMPTS.len();
    let mut sessions: Vec<Session<'_, lloom::models::lfm2::Model<RewindableExact>>> = (0..n)
        .map(|_| {
            Ok(Session::new(
                &model,
                State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut spec = SlotRecordingDrafter {
        k,
        n,
        observed: Vec::new(),
    };
    let opts = DecodeOptions::new(max_tokens);
    let outs = Session::decode_speculative_pipelined(
        &mut sessions,
        lloom::profile::RewindProof::of(),
        &mut spec,
        &PIPELINE_PROMPTS,
        &opts,
        &eos_ids,
        &dev,
        &mut |_, _| Ok(()),
    )?;
    for (i, out) in outs.iter().enumerate() {
        let mine: Vec<&Vec<u32>> = spec
            .observed
            .iter()
            .filter(|(slot, _)| *slot == i)
            .map(|(_, acc)| acc)
            .collect();
        assert_eq!(
            mine.len(),
            out.stats.rounds.len() + 1,
            "sequence {i}: one observe per round plus the prompt seed"
        );
        // The seed names the anchor — the prefill's argmax, the stream's first id.
        assert_eq!(
            mine[0].as_slice(),
            &out.ids[..1],
            "sequence {i}: the seed's tail is the anchor"
        );
        // Every later observe is one round's accepted block, and the rounds
        // concatenate to the stream (plus whatever an EOS or the cap cut off).
        let mut accepted_all: Vec<u32> = Vec::new();
        for (acc, round) in mine[1..].iter().zip(&out.stats.rounds) {
            assert_eq!(
                acc.len(),
                round.accepted,
                "sequence {i}: an observe's block is its round's"
            );
            accepted_all.extend_from_slice(acc);
        }
        let stream_after_anchor = &out.ids[1..];
        assert!(
            accepted_all.starts_with(stream_after_anchor),
            "sequence {i}: the rounds' accepted blocks do not spell this sequence's stream"
        );
    }
    Ok(())
}

/// The streamed prefill seeds the drafter from EVERY unit, not the last one.
///
/// `prefill_ubatch = 3` on an 8-token prompt runs `decode`'s two units (the
/// 3-token spark, then the 5-token remainder). The drafter must see one
/// `observe` per unit before it drafts — 3 rows with an empty accepted tail,
/// then 5 rows with the prefill's argmax — and the two must sum to the prompt.
/// A target that overwrites its taps per forward seeds the last unit alone
/// unless each unit is taken as it lands; that lazily-dropped first unit is
/// #364's defect (§10.2w) with every digest unmoved, and it fails the row
/// count here.
#[test]
fn decode_speculative_streamed_prefill_seeds_every_unit() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 8, 9];
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();

    let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let mut session = Session::new(&model, cache);
    let mut spec = RecordingDrafter {
        k,
        drafts: 0,
        observed: Vec::new(),
    };
    let mut rb = Blocking::new(&eos_ids, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions {
        prefill_ubatch: 3,
        ..DecodeOptions::new(24)
    };
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        &mut spec,
        &mut rb,
        &mut sampler,
        &prompt,
        &opts,
        &eos_ids,
        &mut |_: &[u32]| Ok(()),
    )?;
    assert!(out.ids.len() > k, "the run must span more than one round");
    assert_eq!(
        out.stats.prefill_units, 2,
        "two forwards: the spark, then the remainder"
    );

    let seed: Vec<&(usize, usize, Vec<u32>)> = spec
        .observed
        .iter()
        .take_while(|(drafts_before, _, _)| *drafts_before == 0)
        .collect();
    assert_eq!(
        seed.len(),
        2,
        "one observe per prefill unit: the spark, then the remainder"
    );
    assert_eq!(seed[0].1, 3, "the spark unit's rows");
    assert!(
        seed[0].2.is_empty(),
        "an intermediate unit leaves the anchor where `begin` put it"
    );
    assert_eq!(seed[1].1, 5, "the remainder's rows");
    assert_eq!(
        seed[1].2,
        out.ids[..1],
        "the last unit carries the prefill's argmax"
    );
    assert_eq!(
        seed[0].1 + seed[1].1,
        prompt.len(),
        "the seed covers every prompt position"
    );
    assert_eq!(
        spec.observed.len(),
        out.stats.rounds.len() + 2,
        "one observe per round, plus the two seed units"
    );
    Ok(())
}

/// Streamed speculative decode is byte-identical to streamed greedy decode:
/// the target's prefill runs in the same units on both loops, so the same
/// prompt gives the same ids whichever loop feeds it. (Against the one-shot
/// greedy stream the streamed arms differ only where the target's own chunked
/// prefill does — a property of the target, not of the drafter.)
#[test]
fn decode_speculative_streamed_is_byte_identical_to_greedy_streamed() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let max_tokens = 40usize;
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let ubatch = 4usize;

    let ref_ids = {
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = IdentityProposer;
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions {
            prefill_ubatch: ubatch,
            ..DecodeOptions::new(max_tokens)
        };
        let out = session.decode(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &opts,
            &mut |_: &[u32]| Ok(()),
        )?;
        assert_eq!(
            out.stats.prefill_units, 2,
            "greedy: the spark, then the remainder"
        );
        out.ids
    };
    assert!(
        ref_ids.len() > k,
        "the greedy reference must span more than one round"
    );

    let spec_ids = {
        let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions {
            prefill_ubatch: ubatch,
            ..DecodeOptions::new(max_tokens)
        };
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &opts,
            &eos_ids,
            &mut |_: &[u32]| Ok(()),
        )?;
        assert_eq!(
            out.stats.prefill_units, 2,
            "speculative: the spark, then the remainder"
        );
        out.ids
    };
    assert_eq!(spec_ids, ref_ids, "streamed speculative != streamed greedy");
    Ok(())
}

/// A target whose every `forward` is two forwards: the second clears the
/// first's taps untaken, which is exactly "a forward ran between two probes"
/// — the shape a streamed prefill that skipped a unit's probe would have. It
/// delegates everything else to the tiny LFM2, so the only thing it changes
/// is that one drop per forward.
struct TwiceForward<'a>(&'a lloom::models::lfm2::Model<RewindableExact>);

impl lloom::nn::Model for TwiceForward<'_> {
    type Config = Config;
    fn from_weights(_vb: lloom::nn::VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("TwiceForward wraps a built model")
    }
}

impl lloom::profile::Shaped for TwiceForward<'_> {
    type Profile = RewindableExact;
}

impl lloom::nn::CausalLM for TwiceForward<'_> {
    type Cache = State<RewindableExact>;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<State<RewindableExact>> {
        self.0.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.0.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut State<RewindableExact>,
    ) -> Result<Tensor> {
        // The second pass appends the same tokens AFTER the first's, so the
        // cache stays well-formed (the driver's refusal must be the drop, not
        // a shape error) and the second pass's `begin_taps` drops the first's.
        let n = embeds.dim(1)?;
        let _ = self.0.forward_embeds(embeds, pos, cache)?;
        self.0.forward_embeds(embeds, pos + n, cache)
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
}

impl lloom::nn::speculator::Probes<LayerTaps<5>> for TwiceForward<'_> {
    type Pending = lloom::models::lfm2::PendingVerify;
    type PendingRows = ();
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[lloom::nn::speculator::Anchor],
        cache: &mut State<RewindableExact>,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Vec<Tensor>)> {
        self.0.verify_and_probe(drafts, anchors, cache, eos_ids)
    }
    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &lloom::nn::speculator::Anchor,
        cache: &mut State<RewindableExact>,
    ) -> Result<Self::Pending> {
        self.0.verify_enqueue(draft, anchor, cache)
    }
    fn verify_complete(
        &self,
        pending: Self::Pending,
        cache: &mut State<RewindableExact>,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<Tensor>)> {
        self.0.verify_complete(pending, cache, eos_ids)
    }
    fn take_probe(&self, cache: &mut State<RewindableExact>) -> Result<Vec<Tensor>> {
        self.0.take_probe(cache)
    }
    fn probe_drops(&self, cache: &State<RewindableExact>) -> u64 {
        self.0.probe_drops(cache)
    }
}

/// A probe a forward clears untaken is counted, and a seed across which the
/// count moves is refused. Two arms: the counter itself, on a bare cache; and
/// the refusal, through a target whose every forward is two forwards (the
/// second drops the first's taps). Taps left by whatever ran BEFORE the seed
/// — a warmup's, a previous request's — are drained by the seed and do not
/// refuse anything: the counter is checked across the seed's own probes only.
#[test]
fn decode_speculative_refuses_a_seed_that_dropped_a_probe() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();

    let mut cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    assert_eq!(cache.taps_dropped(), 0);
    let taint = Tensor::new(&[1u32, 2, 3], &dev)?.unsqueeze(0)?;
    model.forward(&taint, 0, &mut cache)?;
    assert_eq!(cache.taps_dropped(), 0, "the taps are still there, untaken");
    model.forward(&taint, 3, &mut cache)?;
    assert_eq!(
        cache.taps_dropped(),
        1,
        "the second forward cleared the first's taps"
    );
    let _ = cache.take_taps();

    // Stale taps before the seed are drained, not refused: a warmup's leftovers.
    {
        let mut cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
        model.forward(&taint, 0, &mut cache)?;
        cache.clear();
        let mut session = Session::new(&model, cache);
        let mut spec = GarbageDrafter { k, token: 0 };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &[4, 5, 6],
            &DecodeOptions::new(8),
            &eos_ids,
            &mut |_: &[u32]| Ok(()),
        )?;
        assert!(
            !out.ids.is_empty(),
            "a seed over a warmup's leftover taps runs"
        );
    }

    // A forward between two of the seed's own probes is refused.
    let twice = TwiceForward(&model);
    let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
    let mut session = Session::new(&twice, cache);
    let mut spec = GarbageDrafter { k, token: 0 };
    let mut rb = Blocking::new(&eos_ids, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let err = session
        .decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &[4, 5, 6],
            &DecodeOptions::new(8),
            &eos_ids,
            &mut |_: &[u32]| Ok(()),
        )
        .expect_err("a seed over a dropped probe must be refused");
    assert!(
        err.to_string().contains("dropped 1 probe set"),
        "the refusal names the drop: {err}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// EOS is an accept boundary: nothing past it is fed, kept or keyed.
// ---------------------------------------------------------------------------

/// A drafter that proposes the target's OWN greedy continuation, taken from a
/// stream decoded with no stop set — so it runs through the natural EOS and
/// proposes the tokens the target really emits after it. Every proposal
/// agrees, so the only thing that can end a round's prefix before `k + 1` is
/// the EOS boundary, which is what the tests below exercise. It records every
/// `observe` (rows per tap, accepted) and tracks the loop's cursor from them.
struct StreamOracle {
    k: usize,
    stream: Vec<u32>,
    /// Index in `stream` of the next token to propose.
    next: usize,
    observed: Vec<(usize, Vec<u32>)>,
}

impl Speculator for StreamOracle {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        self.next = 0;
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let block: Vec<u32> = (0..self.k)
            .map(|i| self.stream.get(self.next + i).copied().unwrap_or(0))
            .collect();
        Ok(active.iter().map(|_| Draft::host(block.clone())).collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        let rows = probe[0].dim(1)?;
        self.next += accepted.len();
        self.observed.push((rows, accepted.to_vec()));
        Ok(())
    }
}

/// The target's greedy stream with NO reachable stop, `n` tokens long.
fn unstopped_stream(
    model: &Model<RewindableExact>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    n: usize,
) -> Result<Vec<u32>> {
    let never = vec![u32::MAX];
    let cache = State::<RewindableExact>::new(true, DType::F32, cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(&never, dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(n),
        &mut |_| Ok(()),
    )?;
    assert_eq!(out.ids.len(), n, "the unstopped stream ran to the budget");
    Ok(out.ids)
}

/// An (EOS id, position) in `stream` that ends a run INSIDE a fully-agreeing
/// block of `k`: the id's first appearance is at `cut`, and `cut` falls at an
/// interior block position — neither the block's first token nor its bonus —
/// so an oracle drafter proposes tokens past it that the target agrees with.
/// Positions are the stream's: index 0 is the prefill's argmax (the first
/// anchor), and round `r` covers `1 + r(k+1) ..= (r+1)(k+1)` when every
/// proposal agrees.
fn mid_block_eos(stream: &[u32], k: usize) -> Option<(u32, usize)> {
    (2..stream.len()).find_map(|cut| {
        let j = (cut - 1) % (k + 1);
        let interior = j >= 1 && j < k;
        let fresh = !stream[..cut].contains(&stream[cut]);
        (interior && fresh).then_some((stream[cut], cut))
    })
}

/// The mid-block-EOS fixture: `K = 4` and this prompt give an unstopped stream
/// `[12, 0, 12, 0, …, 10, 0, 10, …]` whose first `10` is at index 14 — block
/// position 3 of the third round, with two fully-accepted rounds before it
/// and the target agreeing past it. Hardcoded rather than searched, so a
/// fixture change fails here instead of silently choosing a different case.
const EOS_FIXTURE_K: usize = 4;
const EOS_FIXTURE_PROMPT: [u32; 8] = [9, 8, 7, 6, 5, 4, 3, 2];
const EOS_FIXTURE_EOS: u32 = 10;
const EOS_FIXTURE_CUT: usize = 14;
/// A second turn whose continuation does not reach the fixture's EOS within
/// eight tokens, so a two-turn comparison compares something.
const EOS_FIXTURE_TURN2: [u32; 5] = [1, 2, 3, 4, 5];

fn eos_fixture(dev: &Device) -> Result<(Config, Model<RewindableExact>, Vec<u32>)> {
    let cfg = tiny_config(EOS_FIXTURE_K);
    let model = build(&cfg, dev)?;
    let stream = unstopped_stream(&model, &cfg, dev, &EOS_FIXTURE_PROMPT, 40)?;
    assert_eq!(
        mid_block_eos(&stream, EOS_FIXTURE_K),
        Some((EOS_FIXTURE_EOS, EOS_FIXTURE_CUT)),
        "the fixture moved: stream {stream:?}"
    );
    Ok((cfg, model, stream))
}

/// A run stopped by an EOS inside a block leaves the cache, the cursor and the
/// drafter's context at the EOS — nothing past it fed, kept or keyed — on
/// `dev`, through the serial driver:
///
/// - the stream is the greedy stream up to the EOS;
/// - the cache holds exactly the prompt plus the streamed tokens (`pos`), with
///   no overrun to reconcile — the conservative K=1 arm's shape;
/// - the next forward on that cache is bit-identical to the K=1 arm's, so the
///   KV length and the conv phase agree, not only the count;
/// - the drafter's last `observe` ends at the EOS, every observe's rows are its
///   accepted length, and the rows keyed sum to the positions fed.
///
/// The oracle proposes the target's true continuation past the EOS, so without
/// the boundary the target agrees past it and the block runs to `k + 1`: the
/// cache, cursor and drafter then hold `k - j` positions the reply does not have.
fn eos_mid_block_on(dev: &Device) -> Result<()> {
    let prompt: Vec<u32> = EOS_FIXTURE_PROMPT.to_vec();
    let turn2: Vec<u32> = EOS_FIXTURE_TURN2.to_vec();
    let (cfg, model, stream) = eos_fixture(dev)?;
    let (k, eos, cut) = (EOS_FIXTURE_K, EOS_FIXTURE_EOS, EOS_FIXTURE_CUT);
    let j = (cut - 1) % (k + 1);
    let eos_ids = vec![eos];

    // The absolute reference: greedy K=1, blocking (lookahead 0 — no overrun).
    let mut reference = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?,
    );
    let ref_out = {
        let mut rb = Blocking::new(&eos_ids, dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        reference.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(40),
            &mut |_| Ok(()),
        )?
    };
    assert!(ref_out.eos);
    assert_eq!(ref_out.ids, stream[..cut]);
    assert_eq!(ref_out.stats.overrun, 0);
    assert_eq!(reference.pos(), prompt.len() + cut);

    // The subject: the oracle through the serial speculative driver.
    let mut subject = Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, dev)?,
    );
    let mut spec = StreamOracle {
        k,
        stream: stream.clone(),
        next: 0,
        observed: Vec::new(),
    };
    let out = {
        let mut rb = Blocking::new(&eos_ids, dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        subject.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompt,
            &DecodeOptions::new(40),
            &eos_ids,
            &mut |_| Ok(()),
        )?
    };
    assert!(out.eos, "the run ends on the EOS");
    assert_eq!(
        out.ids,
        stream[..cut],
        "the stream is the greedy stream up to the EOS"
    );
    // Non-vacuity: every round before the last accepted the full block (the
    // oracle is right), and the last round is the one the EOS cut.
    let rounds: Vec<usize> = out.stats.rounds.iter().map(|r| r.accepted).collect();
    assert!(rounds.len() >= 2, "the run spans {} rounds", rounds.len());
    assert!(
        rounds[..rounds.len() - 1].iter().all(|&n| n == k + 1),
        "every round before the EOS accepts the whole block: {rounds:?}"
    );
    assert_eq!(
        *rounds.last().unwrap(),
        j + 1,
        "the last round accepts through the EOS at block position {j}: {rounds:?}"
    );

    // The cache ends at the last streamed token: no overrun, `pos` the K=1 arm's.
    assert_eq!(
        out.stats.overrun, 0,
        "an EOS inside a block leaves no overrun (the cache was fed past the EOS)"
    );
    assert_eq!(subject.overrun(), 0);
    assert_eq!(
        subject.pos(),
        prompt.len() + cut,
        "the cache holds exactly the prompt and the streamed tokens"
    );
    assert_eq!(subject.pos(), reference.pos());

    // The drafter: keyed through the EOS and not past it.
    let (rows_last, acc_last) = spec.observed.last().expect("an observe ran");
    assert_eq!(
        acc_last.last(),
        Some(&eos),
        "the last accepted block ends at the EOS"
    );
    assert_eq!(*rows_last, acc_last.len());
    for (rows, acc) in &spec.observed[1..] {
        assert_eq!(
            *rows,
            acc.len(),
            "a round's probe is trimmed to its accepted rows"
        );
    }
    let keyed: usize = spec.observed[1..].iter().map(|(rows, _)| rows).sum();
    assert_eq!(
        keyed, cut,
        "the rows keyed into the drafter after the seed are exactly the positions fed"
    );
    let fed: Vec<u32> = spec.observed[1..]
        .iter()
        .flat_map(|(_, acc)| acc.iter().copied())
        .collect();
    assert_eq!(
        fed,
        stream[1..=cut],
        "the accepted blocks, end to end, are the stream through the EOS and nothing past it"
    );
    assert_eq!(
        spec.observed[0].1,
        stream[..1],
        "the seed's accepted tail is the first anchor"
    );

    // The state, not only the count: the KV length and the conv phase are the
    // K=1 arm's, and the next forward on both caches agrees to well under the
    // accumulation-order spread (a window's K/V come from a multi-position pass,
    // the K=1 arm's from single positions — ulp-level, §2.3.5a tier 2). The
    // control that makes the tolerance evidence: the same forward on the
    // reference cache fed one position further is off by orders more.
    assert_eq!(subject.cache_mut().kv_len(), reference.cache_mut().kv_len());
    assert_eq!(
        subject.cache_mut().conv_phase(),
        reference.cache_mut().conv_phase()
    );
    let next = |session: &mut Session<'_, Model<RewindableExact>>| -> Result<Vec<f32>> {
        let at = session.pos();
        let input = Tensor::new(&[turn2[0]], dev)?.unsqueeze(0)?;
        model
            .forward(&input, at, session.cache_mut())?
            .flatten_all()?
            .to_vec1::<f32>()
    };
    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    };
    let a = next(&mut reference)?;
    let b = next(&mut subject)?;
    let d = max_abs(&a, &b);
    assert!(
        d < 1e-5,
        "the cache after a mid-block EOS is not the K=1 arm's: next-forward max |d| = {d}"
    );
    // Control: one position further along the reference cache.
    let input = Tensor::new(&[turn2[0]], dev)?.unsqueeze(0)?;
    let c = model
        .forward(&input, reference.pos() + 1, reference.cache_mut())?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let dc = max_abs(&b, &c);
    assert!(
        dc > 1e-3,
        "the control (one extra position) must be visibly different: max |d| = {dc}"
    );
    Ok(())
}

#[test]
fn decode_speculative_stops_the_cache_and_the_drafter_at_a_mid_block_eos() -> Result<()> {
    eos_mid_block_on(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn decode_speculative_stops_the_cache_and_the_drafter_at_a_mid_block_eos_on_metal() -> Result<()> {
    eos_mid_block_on(&Device::new_metal(0)?)
}

/// The pipelined driver takes the same boundary through `verify_complete`:
/// one sequence, the oracle, the EOS inside a block — the stream, the
/// overrun and the cursor are the serial driver's.
#[test]
fn decode_speculative_pipelined_stops_at_a_mid_block_eos() -> Result<()> {
    let dev = Device::Cpu;
    let prompt: Vec<u32> = EOS_FIXTURE_PROMPT.to_vec();
    let (cfg, model, stream) = eos_fixture(&dev)?;
    let (k, eos, cut) = (EOS_FIXTURE_K, EOS_FIXTURE_EOS, EOS_FIXTURE_CUT);
    let eos_ids = vec![eos];
    let mut sessions = vec![Session::new(
        &model,
        State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
    )];
    let mut spec = StreamOracle {
        k,
        stream: stream.clone(),
        next: 0,
        observed: Vec::new(),
    };
    let prompts: [&[u32]; 1] = [&prompt];
    let outs = Session::decode_speculative_pipelined(
        &mut sessions,
        lloom::profile::RewindProof::of(),
        &mut spec,
        &prompts,
        &DecodeOptions::new(40),
        &eos_ids,
        &dev,
        &mut |_, _| Ok(()),
    )?;
    let out = &outs[0];
    assert!(out.eos);
    assert_eq!(out.ids, stream[..cut]);
    assert_eq!(out.stats.overrun, 0, "no overrun past a mid-block EOS");
    assert_eq!(sessions[0].overrun(), 0);
    assert_eq!(sessions[0].pos(), prompt.len() + cut);
    let j = (cut - 1) % (k + 1);
    assert_eq!(out.stats.rounds.last().map(|r| r.accepted), Some(j + 1));
    assert_eq!(spec.observed.last().and_then(|(_, a)| a.last()), Some(&eos));
    Ok(())
}

/// A second turn after a speculative turn that ended on a mid-block EOS is
/// byte-identical to the second turn after the K=1 arm's: `prepare` finds the
/// cache exactly at the reply's end (an Extend with nothing to reconcile), and
/// the continuation — greedy, or speculative again — is the reference's.
///
/// Recorded alongside: with the boundary absent the cache holds `k - j`
/// positions past the EOS, and `prepare` reconciles them by a rewind before
/// the turn (`Session::reconcile`, §10.10) — so the continuation was already
/// identical; what the boundary changes is that there is nothing to rewind.
#[test]
fn a_turn_after_a_mid_block_eos_continues_identically_with_nothing_to_reconcile() -> Result<()> {
    use lloom::nn::ReuseAction;
    let dev = Device::Cpu;
    let prompt: Vec<u32> = EOS_FIXTURE_PROMPT.to_vec();
    let turn2: Vec<u32> = EOS_FIXTURE_TURN2.to_vec();
    let (cfg, model, stream) = eos_fixture(&dev)?;
    let (k, eos, cut) = (EOS_FIXTURE_K, EOS_FIXTURE_EOS, EOS_FIXTURE_CUT);
    let eos_ids = vec![eos];
    let greedy = || LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // Reference: K=1 blocking, turn 1 to the EOS, turn 2 appended.
    let reference = {
        let mut s = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let t1 = s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(40),
            &mut |_| Ok(()),
        )?;
        assert_eq!(t1.ids, stream[..cut]);
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(p.action, ReuseAction::Extend);
        assert_eq!(p.suffix, turn2);
        s.decode(
            &mut IdentityProposer,
            &mut rb,
            &mut greedy(),
            &p.suffix,
            &DecodeOptions::new(8),
            &mut |_| Ok(()),
        )?
        .ids
    };
    assert_eq!(reference.len(), 8, "the second turn runs to its budget");

    // Subject: speculative turn 1 (the oracle, EOS mid-block), then turn 2 on
    // the K=1 arm and, separately, speculative again with a garbage drafter.
    for speculative_turn2 in [false, true] {
        let mut s = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?,
        );
        let mut spec = StreamOracle {
            k,
            stream: stream.clone(),
            next: 0,
            observed: Vec::new(),
        };
        let mut rb = Blocking::new(&eos_ids, &dev)?;
        let t1 = s.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(40),
            &eos_ids,
            &mut |_| Ok(()),
        )?;
        assert!(t1.eos);
        assert_eq!(t1.ids, stream[..cut]);
        assert_eq!(s.overrun(), 0, "nothing to reconcile after a mid-block EOS");
        let pos_after_t1 = s.pos();
        let full: Vec<u32> = [prompt.clone(), t1.ids.clone(), turn2.clone()].concat();
        let p = s.prepare(&full)?;
        assert_eq!(p.action, ReuseAction::Extend);
        assert_eq!(p.suffix, turn2);
        assert_eq!(s.pos(), pos_after_t1, "prepare rewound nothing");
        let t2 = if speculative_turn2 {
            let mut garbage = GarbageDrafter { k, token: 0 };
            s.decode_speculative(
                lloom::profile::RewindProof::of(),
                &mut garbage,
                &mut rb,
                &mut greedy(),
                &p.suffix,
                &DecodeOptions::new(8),
                &eos_ids,
                &mut |_| Ok(()),
            )?
            .ids
        } else {
            s.decode(
                &mut IdentityProposer,
                &mut rb,
                &mut greedy(),
                &p.suffix,
                &DecodeOptions::new(8),
                &mut |_| Ok(()),
            )?
            .ids
        };
        assert_eq!(
            t2, reference,
            "turn 2 after a speculative mid-block-EOS turn 1 (speculative turn 2: \
             {speculative_turn2}) is not the K=1 arm's"
        );
    }
    Ok(())
}

/// A drafter that proposes the reference stream with one token wrong per
/// round, at an offset that cycles with the round: so rounds keep every
/// length from one row to the whole window, and the rows a round keeps past
/// the anchor are the ones whose per-row state (the processors' history, a
/// budget's count) the window must have built right.
struct HalfRightDrafter {
    k: usize,
    truth: Vec<u32>,
    produced: usize,
    round: usize,
}

impl Speculator for HalfRightDrafter {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        self.produced = 0;
        self.round = 0;
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let wrong = self.round % (self.k + 1);
        self.round += 1;
        let block: Vec<u32> = (0..self.k)
            .map(|i| {
                let t = self.truth.get(self.produced + i).copied().unwrap_or(0);
                if i == wrong {
                    (t + 1) % 48
                } else {
                    t
                }
            })
            .collect();
        Ok(active.iter().map(|_| Draft::host(block.clone())).collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        self.produced += accepted.len();
        Ok(())
    }
}

/// One request's sampler: an arm, processors, a thinking budget, a digest.
#[derive(Clone, Debug)]
struct Cell {
    sampling: Sampling,
    processors: lloom::generation::Processors,
    budget: Option<lloom::generation::ThinkingBudget>,
    digest: bool,
}

impl Cell {
    fn arm(sampling: Sampling) -> Self {
        Cell {
            sampling,
            processors: Default::default(),
            budget: None,
            digest: false,
        }
    }

    fn sampler(&self, seed: u64) -> LogitsProcessor {
        LogitsProcessor::from_sampling(seed, self.sampling.clone())
            .with_processors(self.processors.clone())
            .with_logits_digest(self.digest)
    }
}

/// What a run returned: the ids, whether EOS ended it, and under a digest
/// the digest and the rows it took.
type RunOut = (Vec<u32>, bool, Option<(String, usize)>);

/// The non-speculative run: the K=1 body on the blocking arm, its budget
/// the arm's, every draw keyed to its position.
fn k1_run(
    model: &Model<RewindableExact>,
    cfg: &Config,
    dev: &Device,
    cell: &Cell,
    seed: u64,
    prompt: &[u32],
    max_tokens: usize,
) -> Result<RunOut> {
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let cache = State::<RewindableExact>::new(true, DType::F32, cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(&eos_ids, dev)?;
    use lloom::generation::ReadbackStrategy;
    rb.set_thinking_budget(cell.budget.as_ref())?;
    let mut sampler = cell.sampler(seed);
    sampler.observe_prompt(prompt);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max_tokens),
        &mut |_: &[u32]| Ok(()),
    )?;
    let digest = sampler.logits_digest().map(|d| (d, sampler.digest_rows()));
    Ok((out.ids, out.eos, digest))
}

/// The speculative run of the same request, with `spec` drafting; returns
/// the run and its number of rounds.
#[allow(clippy::too_many_arguments)]
fn spec_run<S: Speculator<Probe = LayerTaps<5>>>(
    model: &Model<RewindableExact>,
    cfg: &Config,
    dev: &Device,
    cell: &Cell,
    seed: u64,
    prompt: &[u32],
    max_tokens: usize,
    spec: &mut S,
) -> Result<(RunOut, usize)> {
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let cache = State::<RewindableExact>::new(true, DType::F32, cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(&eos_ids, dev)?;
    let mut sampler = cell.sampler(seed).with_window_budget(cell.budget.as_ref());
    sampler.observe_prompt(prompt);
    let mut streamed = Vec::new();
    let out = session.decode_speculative(
        lloom::profile::RewindProof::of(),
        spec,
        &mut rb,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max_tokens),
        &eos_ids,
        &mut |b: &[u32]| {
            streamed.extend_from_slice(b);
            Ok(())
        },
    )?;
    assert_eq!(out.ids, streamed, "the stream must equal the returned ids");
    let digest = sampler.logits_digest().map(|d| (d, sampler.digest_rows()));
    Ok(((out.ids, out.eos, digest), out.stats.rounds.len()))
}

/// **The sampled identity (#943).** Speculative decode is byte-identical to
/// non-speculative decode at the same seed, for every sampler arm and every
/// feature the K=1 loop carries: the verify draws the target's token per
/// window position keyed to that position, row `i` against the history it
/// has in a step-by-step decode, and feeds what the K=1 loop would feed.
///
/// - The arms: multinomial (`-t`, the default method), gumbel-max, top-k,
///   top-p and top-k-then-top-p (the filter, then the inverse CDF).
/// - The features: the processors (each row's history the output plus the
///   drafts before it), a thinking budget (each row's fed token the
///   budget's rule over the rows before it), the logits digest (over the
///   rows the rounds keep).
/// - Two drafters: garbage (every round keeps one row and throws `k` away —
///   the rows a draw-count stream would be shifted by) and half-right
///   (rounds keep every length, so kept rows past the anchor exercise each
///   row's own state).
///
/// N seeds per cell; every cell must agree, and enough must be informative
/// (more than one round, a non-greedy stream).
#[test]
fn sampled_speculative_decode_is_byte_identical_to_sampled_decode_at_one_seed() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let max_tokens = 40usize;

    let greedy = k1_run(
        &model,
        &cfg,
        &dev,
        &Cell::arm(Sampling::ArgMax),
        0,
        &prompt,
        max_tokens,
    )?
    .0;
    let procs = lloom::generation::Processors {
        repetition_penalty: 1.3,
        presence_penalty: 0.3,
        frequency_penalty: 0.4,
        min_p: 0.05,
        logit_bias: vec![(9, 1.0)],
    };
    // A budget the runs reach: six tokens of thinking, then a transition
    // and the closing id forced (one of the tiny vocabulary's ids; a run
    // that emits it first closes the block itself).
    let budget = lloom::generation::ThinkingBudget::new(6, vec![11], vec![17, 11])?;
    let cells = vec![
        Cell::arm(Sampling::All { temperature: 1.2 }),
        Cell::arm(Sampling::GumbelSoftmax { temperature: 0.9 }),
        Cell::arm(Sampling::TopK {
            k: 8,
            temperature: 1.1,
        }),
        Cell::arm(Sampling::TopP {
            p: 0.85,
            temperature: 1.0,
        }),
        Cell::arm(Sampling::TopKThenTopP {
            k: 12,
            p: 0.9,
            temperature: 1.0,
        }),
        Cell {
            processors: procs.clone(),
            ..Cell::arm(Sampling::All { temperature: 1.0 })
        },
        Cell {
            processors: procs.clone(),
            ..Cell::arm(Sampling::ArgMax)
        },
        Cell {
            budget: Some(budget.clone()),
            ..Cell::arm(Sampling::All { temperature: 1.0 })
        },
        Cell {
            budget: Some(budget.clone()),
            ..Cell::arm(Sampling::ArgMax)
        },
        Cell {
            digest: true,
            ..Cell::arm(Sampling::GumbelSoftmax { temperature: 1.0 })
        },
        Cell {
            processors: procs.clone(),
            budget: Some(budget.clone()),
            digest: true,
            ..Cell::arm(Sampling::TopK {
                k: 10,
                temperature: 0.9,
            })
        },
    ];
    let mut compared = 0usize;
    let mut informative = 0usize;
    let mut kept_past_anchor = 0usize;
    for cell in &cells {
        let mut moved_from_greedy = false;
        for seed in [0u64, 7, 42, 1234] {
            let reference = k1_run(&model, &cfg, &dev, cell, seed, &prompt, max_tokens)?;
            let mut garbage = GarbageDrafter { k, token: 0 };
            let (g, g_rounds) = spec_run(
                &model,
                &cfg,
                &dev,
                cell,
                seed,
                &prompt,
                max_tokens,
                &mut garbage,
            )?;
            let mut half = HalfRightDrafter {
                k,
                truth: reference.0.clone(),
                produced: 0,
                round: 0,
            };
            let (h, h_rounds) = spec_run(
                &model, &cfg, &dev, cell, seed, &prompt, max_tokens, &mut half,
            )?;
            for (name, got, rounds) in [("garbage", &g, g_rounds), ("half-right", &h, h_rounds)] {
                assert_eq!(
                    (&reference.0, reference.1),
                    (&got.0, got.1),
                    "{cell:?} seed {seed} ({name} drafter): the speculative run is not the K=1 \
                     run at one seed"
                );
                // The digest hashes the same positions — one row a token,
                // the EOS step's included. Its bytes are each pass's own: a
                // verify window's rows come from an `m = k + 1` pass, the
                // K=1 steps' from `m = 1`, and `deterministic` promises
                // nothing across a pass split (`profiles-and-memory-design.md`
                // §3), the CPU's gemm summing in a row-count-dependent order.
                assert_eq!(
                    reference.2.as_ref().map(|d| d.1),
                    got.2.as_ref().map(|d| d.1),
                    "{cell:?} seed {seed} ({name}): the digest took different rows"
                );
                if let Some((_, rows)) = &got.2 {
                    assert_eq!(
                        *rows,
                        got.0.len() + usize::from(got.1),
                        "{cell:?} seed {seed} ({name}): one digest row a token, EOS included"
                    );
                }
                compared += 1;
                if rounds > 1 && reference.0.len() > k && reference.0 != greedy {
                    informative += 1;
                }
            }
            // Rounds that kept rows past the anchor: fewer rounds than
            // tokens.
            if h_rounds + 1 < h.0.len() {
                kept_past_anchor += 1;
            }
            if cell.digest {
                assert!(reference.2.is_some(), "{cell:?}: no digest");
            }
            moved_from_greedy |= reference.0 != greedy;
        }
        assert!(
            moved_from_greedy,
            "{cell:?}: every seed decoded greedy's stream; the cell exercises nothing"
        );
    }
    eprintln!(
        "[{compared} runs compared, {informative} informative, {kept_past_anchor} half-right \
         runs kept rows past the anchor]"
    );
    assert!(
        informative >= compared / 2,
        "only {informative} of {compared} runs ran past one round with a non-greedy stream"
    );
    assert!(
        kept_past_anchor >= cells.len() * 2,
        "the half-right drafter kept rows past the anchor in only {kept_past_anchor} runs"
    );
    Ok(())
}

/// The window's features move a run: a budget forces its tail, processors
/// change the draws, a digest differs by seed — so the identity above is
/// between runs the features act on, not between two runs that ignore them.
#[test]
fn the_features_under_speculation_move_what_they_should() -> Result<()> {
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let all = Cell::arm(Sampling::All { temperature: 1.0 });
    let budget = lloom::generation::ThinkingBudget::new(6, vec![11], vec![17, 11])?;
    let with_budget = Cell {
        budget: Some(budget.clone()),
        ..all.clone()
    };
    let mut garbage = GarbageDrafter { k, token: 0 };
    let ((plain, _, _), _) = spec_run(&model, &cfg, &dev, &all, 7, &prompt, 30, &mut garbage)?;
    let ((forced, _, _), _) = spec_run(
        &model,
        &cfg,
        &dev,
        &with_budget,
        7,
        &prompt,
        30,
        &mut garbage,
    )?;
    let report = budget.report(&forced);
    assert!(
        report.fired || report.closed_by.is_some(),
        "the budget neither fired nor saw its block close: {forced:?}"
    );
    if report.fired {
        assert_eq!(
            &forced[6..8],
            &[17, 11],
            "the forced tail at the budget: {forced:?}"
        );
    }
    assert_ne!(plain, forced, "the budget moved nothing");
    let digest = Cell {
        digest: true,
        ..all.clone()
    };
    let ((_, _, d7), _) = spec_run(&model, &cfg, &dev, &digest, 7, &prompt, 30, &mut garbage)?;
    let ((_, _, d7b), _) = spec_run(&model, &cfg, &dev, &digest, 7, &prompt, 30, &mut garbage)?;
    let ((_, _, d8), _) = spec_run(&model, &cfg, &dev, &digest, 8, &prompt, 30, &mut garbage)?;
    assert!(
        d7.is_some() && d7 != d8,
        "the digest did not follow the run: {d7:?} {d8:?}"
    );
    assert_eq!(d7, d7b, "the digest of one run differs run to run");
    Ok(())
}

/// Proposes the reference stream with a confidence per position that falls
/// below one half at the third: under `speculative.p_min = 0.5` the driver
/// cuts every draft to two tokens before the verify.
struct ConfidentDrafter {
    k: usize,
    /// The positions it is confident of: its confidence is 0.9 there, 0.2 after.
    confident: usize,
    truth: Vec<u32>,
    produced: usize,
    widest: usize,
}

impl Speculator for ConfidentDrafter {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;
    fn n_seq(&self) -> usize {
        1
    }
    fn max_draft(&self) -> usize {
        self.k
    }
    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        self.produced = 0;
        Ok(())
    }
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let tokens: Vec<u32> = (0..self.k)
            .map(|i| self.truth.get(self.produced + i).copied().unwrap_or(0))
            .collect();
        let confidence: Vec<f32> = (0..self.k)
            .map(|i| if i < self.confident { 0.9 } else { 0.2 })
            .collect();
        Ok(active
            .iter()
            .map(|_| Draft {
                tokens: DraftTokens::Host(tokens.clone()),
                confidence: confidence.clone(),
            })
            .collect())
    }
    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        accepted: &[u32],
    ) -> Result<()> {
        self.produced += accepted.len();
        self.widest = self.widest.max(accepted.len());
        Ok(())
    }
}

/// `speculative.p_min` reaches the window body: drafts are cut at the first
/// position the drafter is less confident of, so no round keeps more than
/// the confident prefix and the target's own token — and the stream is the
/// K=1 run's, at a temperature, as without the cut. A drafter confident of
/// nothing keeps one draft: LFM2's verify needs one (on the real model the cut
/// to zero failed the turn, `lfm2 verify_tokens: no proposed tokens`).
#[test]
fn p_min_cuts_the_draft_and_keeps_the_stream() -> Result<()> {
    let mut t = lloom::tuning::Tuning::default();
    t.dspark.p_min = 0.5;
    let _ = lloom::tuning::install(t);
    assert_eq!(lloom::tuning::tuning().dspark.p_min, 0.5);
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    for confident in [2usize, 0] {
        for cell in [
            Cell::arm(Sampling::ArgMax),
            Cell::arm(Sampling::All { temperature: 1.0 }),
        ] {
            let reference = k1_run(&model, &cfg, &dev, &cell, 3, &prompt, 30)?;
            let mut d = ConfidentDrafter {
                k,
                confident,
                truth: reference.0.clone(),
                produced: 0,
                widest: 0,
            };
            let ((ids, eos, _), rounds) =
                spec_run(&model, &cfg, &dev, &cell, 3, &prompt, 30, &mut d)?;
            assert_eq!((&ids, eos), (&reference.0, reference.1), "{cell:?}");
            let kept = confident.max(1);
            assert_eq!(
                d.widest,
                kept + 1,
                "{cell:?}, confident of {confident}: a round kept {} tokens; the cut keeps \
                 {kept} drafts and the target's own",
                d.widest
            );
            assert!(rounds > 4, "{cell:?}: {rounds} rounds");
        }
    }
    Ok(())
}

/// **A prompt the cache holds whole, under speculation.** The K=1 body
/// resumes from the logits the prefill held (`ReuseAction::Resume`); the
/// speculative body cannot (its first round needs the target's probe at the
/// last position) and re-feeds the last token instead
/// (`Session::prepare_with(.., false)`: one position back through a turn
/// checkpoint, else a fresh prefill). Both decode the same tokens — greedy
/// and at a temperature, where the anchor's draw is keyed to the same
/// position either way.
#[test]
fn a_fully_cached_prompt_decodes_under_speculation_as_its_k1_resume() -> Result<()> {
    use lloom::nn::ReuseAction;
    let dev = Device::Cpu;
    let k = 4usize;
    let cfg = tiny_config(k);
    let model = build(&cfg, &dev)?;
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let eos_ids: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let opts = DecodeOptions::new(30);
    for cell in [
        Cell::arm(Sampling::ArgMax),
        Cell::arm(Sampling::All { temperature: 1.0 }),
    ] {
        let k1 = {
            let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
            let mut session = Session::new(&model, cache);
            let mut rb = Blocking::new(&eos_ids, &dev)?;
            session.prefill(&mut rb, &prompt, &opts)?;
            let prepared = session.prepare(&prompt)?;
            assert_eq!(prepared.action, ReuseAction::Resume);
            let mut sampler = cell.sampler(5);
            sampler.observe_prompt(&prompt);
            session
                .decode(
                    &mut IdentityProposer,
                    &mut rb,
                    &mut sampler,
                    &prepared.suffix,
                    &opts,
                    &mut |_: &[u32]| Ok(()),
                )?
                .ids
        };
        let (spec, action) = {
            let cache = State::<RewindableExact>::new(true, DType::F32, &cfg, &dev)?;
            let mut session = Session::new(&model, cache);
            let mut rb = Blocking::new(&eos_ids, &dev)?;
            session.prefill(&mut rb, &prompt, &opts)?;
            let prepared = session.prepare_with(&prompt, false)?;
            assert_ne!(prepared.action, ReuseAction::Resume);
            assert!(!prepared.suffix.is_empty(), "nothing re-fed");
            let mut garbage = GarbageDrafter { k, token: 0 };
            let mut sampler = cell.sampler(5);
            sampler.observe_prompt(&prompt);
            let out = session.decode_speculative(
                lloom::profile::RewindProof::of(),
                &mut garbage,
                &mut rb,
                &mut sampler,
                &prepared.suffix,
                &opts,
                &eos_ids,
                &mut |_: &[u32]| Ok(()),
            )?;
            (out.ids, prepared.action)
        };
        eprintln!("{cell:?}: the speculative body re-fed through {action:?}");
        assert_eq!(spec, k1, "{cell:?}: not the K=1 resume's tokens");
        assert!(k1.len() > k, "{cell:?}: too short a run to show anything");
    }
    Ok(())
}
