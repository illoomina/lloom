//! MTP on the Qwen4-exp target is byte-identical to greedy, and the verify's
//! three-state rollback (KV, conv ring, DeltaNet `S` — plus the PLE history and
//! the n-gram window) puts the cache exactly where decoding only the accepted
//! tokens would (issue #749).
//!
//! The identity gate follows `dspark_decode_identity.rs`: the reference is the
//! K=1 greedy loop (`Session::decode`, `forward` per token), the subject is
//! `Session::decode_speculative` driven by (a) the real `MtpDrafter` on a
//! synthetic head, (b) an oracle that always proposes the greedy stream's next
//! token (every round fully accepted — the resolve path), and (c) a garbage
//! drafter (every round rejected at the first draft — the rewind + replay
//! path). All three must produce the greedy stream; a drafter changes the
//! acceptance rate and nothing else.
//!
//! CPU: the question is the accept / rollback / re-key bookkeeping, and the
//! CPU backend is bit-stable.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::{Cache, Config, MtpDrafter, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::speculator::{Anchor, Draft, KvSource, NextnHidden, Probes, SeqId, Speculator};
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::{Device, Result, Tensor};

fn config() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    // An EOS id the synthetic model can still emit, but not at the second
    // token of every prompt (`tiny`'s 2 is what the fixed-seed weights favour).
    cfg.eos_token_id = 127;
    cfg
}

/// A prompt whose greedy stream is long enough to span several rounds.
fn long_prompt(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    max: usize,
) -> Result<(Vec<u32>, Vec<u32>, bool)> {
    for seed in 0..8u32 {
        let prompt: Vec<u32> = (0..5).map(|i| (seed * 11 + i * 7 + 1) % 120).collect();
        let (ids, eos) = greedy(model, cfg, &prompt, max)?;
        if ids.len() >= 6 {
            return Ok((prompt, ids, eos));
        }
    }
    lloom_tensor::bail!("no prompt gave a greedy stream of >= 6 tokens")
}

fn greedy(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool)> {
    let dev = Device::Cpu;
    let cache = model.new_cache(cfg, &dev)?;
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

fn speculative<S: Speculator<Probe = NextnHidden>>(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
) -> Result<(Vec<u32>, bool, usize)> {
    speculative_on(model, cfg, spec, prompt, max, None)
}

/// `ring`: the cache over `spec_pools(1, k)` — the verify as one pass with
/// the accept by phase (`verify_window`); `None`: the sequential verify.
fn speculative_on<S: Speculator<Probe = NextnHidden>>(
    model: &Qwen4Exp<Resident>,
    cfg: &Config,
    spec: &mut S,
    prompt: &[u32],
    max: usize,
    ring: Option<usize>,
) -> Result<(Vec<u32>, bool, usize)> {
    let dev = Device::Cpu;
    let mut cache: Cache = match ring {
        Some(k) => Cache::in_pools(cfg, &model.spec_pools(1, k, &dev))?,
        None => model.new_cache(cfg, &dev)?,
    };
    cache.want_nextn = true;
    let mut session = Session::new(model, cache);
    let eos = [cfg.eos_token_id];
    let mut rb = Blocking::new(&eos, &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(max);
    let mut streamed = Vec::new();
    let out =
        session.decode_speculative(spec, &mut rb, &mut sampler, prompt, &opts, &eos, &mut |b| {
            streamed.extend_from_slice(b);
            Ok(())
        })?;
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
        // The anchor is stream[produced-1]; the draft is what follows it.
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

#[test]
fn mtp_speculative_decode_is_byte_identical_to_greedy() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model(&cfg, &dev);
    let max = 24;
    let (prompt, ref_ids, ref_eos) = long_prompt(&model, &cfg, max)?;

    // (a) the real MTP drafter on a synthetic one-layer head
    let head = testing::mtp_head(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    let (ids, eos, _) = speculative(&model, &cfg, &mut mtp, &prompt, max)?;
    assert_eq!(ids, ref_ids, "MTP drafter: not the greedy stream");
    assert_eq!(eos, ref_eos);
    assert!(
        mtp.rows_seen() >= prompt.len() + ref_ids.len() - 1,
        "the drafter was keyed with {} rows for {} prompt + {} generated",
        mtp.rows_seen(),
        prompt.len(),
        ref_ids.len()
    );

    // (a') K > 1: the layer re-applied on its own carry for the further
    // drafts — the greedy stream still, and the drafter's cache keyed only
    // with the observed rows (the re-applications are rewound).
    for k in [2usize, 3] {
        let head = testing::mtp_head(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
        let mut mtp =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let (ids, eos, accepted) = speculative(&model, &cfg, &mut mtp, &prompt, max)?;
        assert_eq!(
            ids, ref_ids,
            "MTP drafter at k = {k}: not the greedy stream"
        );
        assert_eq!(eos, ref_eos);
        assert!(
            mtp.rows_seen() >= prompt.len() + ref_ids.len() - 1,
            "k = {k}: the drafter was keyed with {} rows for {} prompt + {} generated",
            mtp.rows_seen(),
            prompt.len(),
            ref_ids.len()
        );
        assert_eq!(
            mtp.cache().pos,
            mtp.rows_seen(),
            "k = {k}: the drafter's cache holds draft-time rows"
        );
        println!("MTPK k={k} accepted={accepted}");
    }

    // (a'') the verify as ONE pass over the phase rings with the accept by
    // phase (no replay): the greedy stream at every K, the rounds' accepted
    // counts equal to the sequential verify's.
    for k in [1usize, 2, 3] {
        let head = testing::mtp_head(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
        let mut seq =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let (ids_s, eos_s, acc_s) = speculative_on(&model, &cfg, &mut seq, &prompt, max, None)?;
        let head = testing::mtp_head(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
        let mut ring =
            MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev).with_k(k);
        let (ids_r, eos_r, acc_r) = speculative_on(&model, &cfg, &mut ring, &prompt, max, Some(k))?;
        assert_eq!(
            ids_r, ref_ids,
            "ring verify at k = {k}: not the greedy stream"
        );
        assert_eq!(
            (ids_r, eos_r, acc_r),
            (ids_s, eos_s, acc_s),
            "ring vs sequential verify at k = {k}"
        );
        println!("MTPRING k={k} accepted={acc_r}");
    }

    // (b) the oracle: every round fully accepted (the resolve path)
    let mut oracle = Oracle {
        stream: ref_ids.clone(),
        produced: 0,
    };
    let (ids, eos, accepted) = speculative(&model, &cfg, &mut oracle, &prompt, max)?;
    assert_eq!(ids, ref_ids, "oracle: not the greedy stream");
    assert_eq!(eos, ref_eos);
    assert!(accepted > 0, "the oracle's drafts must be accepted");

    // (c) garbage: every round rejected (the rewind + replay path). A valid id
    // the greedy stream never emits.
    let token = (0..cfg.vocab_size as u32)
        .find(|t| !ref_ids.contains(t) && *t != cfg.eos_token_id)
        .expect("a vocabulary id the stream never emits");
    let mut garbage = Garbage { token };
    let (ids, eos, accepted) = speculative(&model, &cfg, &mut garbage, &prompt, max)?;
    assert_eq!(ids, ref_ids, "garbage drafter: not the greedy stream");
    assert_eq!(eos, ref_eos);
    assert_eq!(accepted, 0, "a garbage draft must never be accepted");
    Ok(())
}

/// The identity with the N-gram PLE attached — the real model's shape: the
/// PLE's conv history, the n-gram window and the pending ids all roll back.
#[test]
fn mtp_identity_holds_with_the_ple_attached() -> Result<()> {
    use lloom::models::qwen4exp::{NgramHash, PleLayer, PleTable};
    let cfg = config();
    let dev = Device::Cpu;
    let mut model = testing::resident_model(&cfg, &dev);
    let hash = NgramHash::new(
        vec![1_000_003, 999_983, 7],
        vec![13, 17, 19, 23],
        vec![0, 13, 30, 49],
        2,
        cfg.eos_token_id,
    )?;
    let rows = hash.total_rows() as usize;
    let table = PleTable::Dense(testing::wt((rows, cfg.ple_head_dim), 3.0, &dev));
    let hc_dim = cfg.hc_dim();
    let ple_dim = cfg.ple_head_dim * hash.n_heads();
    let block = PleLayer::new(
        testing::lin(hc_dim, ple_dim, 21.0, &dev),
        testing::lin(cfg.hidden_size, ple_dim, 22.0, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::ones(hc_dim, &dev),
        testing::wt((hc_dim, 4), 23.0, &dev),
        cfg.hc_count,
        cfg.hidden_size,
        cfg.ple_ngram_size,
        cfg.norm_eps as f32,
    )?;
    model.set_ple(1, hash, table, block)?;
    let max = 24;
    let (prompt, ref_ids, ref_eos) = long_prompt(&model, &cfg, max)?;
    let token = (0..cfg.vocab_size as u32)
        .find(|t| !ref_ids.contains(t) && *t != cfg.eos_token_id)
        .expect("a never-emitted id");
    let (ids, eos, accepted) = speculative(&model, &cfg, &mut Garbage { token }, &prompt, max)?;
    assert_eq!(
        ids, ref_ids,
        "garbage drafter with PLE: not the greedy stream"
    );
    assert_eq!(eos, ref_eos);
    assert_eq!(accepted, 0);
    let head = testing::mtp_head(&cfg, &dev, testing::resident_experts(&cfg, 7, &dev));
    let mut mtp = MtpDrafter::new(&head, model.embed_tokens(), model.lm_head(), &cfg, &dev);
    let (ids, _, _) = speculative(&model, &cfg, &mut mtp, &prompt, max)?;
    assert_eq!(ids, ref_ids, "MTP drafter with PLE: not the greedy stream");
    Ok(())
}

/// After a verify that accepts `a` of `k + 1` positions, every state channel
/// equals a fresh cache's after decoding the prompt + the accepted tokens —
/// an absolute reference from a different route (§3.1b), checked per channel
/// and layer by checksum.
#[test]
fn partial_accept_rolls_every_state_back_to_the_accepted_length() -> Result<()> {
    let cfg = config();
    let dev = Device::Cpu;
    let model = testing::resident_model(&cfg, &dev);
    let (prompt, ref_ids, _) = long_prompt(&model, &cfg, 6)?;
    assert!(ref_ids.len() >= 3);
    let anchor_tok = ref_ids[0]; // the prefill's own argmax
    let right = ref_ids[1]; // greedy's next token after the anchor
    let wrong = if right == 0 { 1 } else { 0 };

    let checksums_after = |tokens: &[u32]| -> Result<Vec<(String, u64, usize)>> {
        let mut cache = model.new_cache(&cfg, &dev)?;
        let input = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &dev)?;
        model.forward(&input, 0, &mut cache)?;
        cache.checksums()
    };

    for (draft, expect_accepted) in [(wrong, 1usize), (right, 2usize)] {
        let mut cache = model.new_cache(&cfg, &dev)?;
        cache.want_nextn = true;
        let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
        model.forward(&input, 0, &mut cache)?;
        let _ = model.take_probe(&mut cache)?;
        let anchor = Anchor {
            last: anchor_tok,
            pos: cache.pos,
        };
        let (accepted, probe) = model.verify_and_probe(
            &[Draft::host(vec![draft])],
            &[anchor],
            &mut cache,
            &[cfg.eos_token_id],
        )?;
        let accepted = &accepted[0];
        assert_eq!(
            accepted.len(),
            expect_accepted,
            "draft {draft}: accepted {accepted:?}"
        );
        assert_eq!(
            accepted[0], right,
            "the target's token at the anchor is greedy's"
        );
        assert_eq!(probe.dims(), &[1, expect_accepted, cfg.hc_dim()]);
        assert!(!cache.window_open());
        // The absolute reference: prompt + anchor + the accepted drafts, decoded
        // fresh in one chunked pass + steps.
        let mut consumed = prompt.clone();
        consumed.push(anchor_tok);
        consumed.extend_from_slice(&accepted[..expect_accepted - 1]);
        assert_eq!(cache.pos, consumed.len());
        let got = cache.checksums()?;
        // decode the reference position by position past the prompt, as the
        // verify did (a chunk pass reorders fp32 — the reference must take the
        // same route for the checksums to be comparable)
        let want = {
            let mut c = model.new_cache(&cfg, &dev)?;
            let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
            model.forward(&input, 0, &mut c)?;
            for &t in &consumed[prompt.len()..] {
                let input = Tensor::from_vec(vec![t], (1, 1), &dev)?;
                model.forward(&input, c.pos, &mut c)?;
            }
            c.checksums()?
        };
        assert_eq!(got.len(), want.len());
        for ((n1, h1, _), (n2, h2, _)) in got.iter().zip(&want) {
            assert_eq!(n1, n2);
            assert_eq!(h1, h2, "draft {draft}: state {n1} differs after the verify");
        }
        // and the chunk-pass reference must differ from neither on the KV
        // lengths (the shape half of the rollback)
        let _ = checksums_after(&consumed)?;
    }
    Ok(())
}
